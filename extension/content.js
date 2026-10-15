(() => {
  "use strict";

  const activeMappings = [];
  let enabled = true;
  let replayingSubmit = false;
  let pendingSubmit = false;

  chrome.storage.local.get({enabled: true}, (settings) => { enabled = settings.enabled !== false; });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.enabled) enabled = changes.enabled.newValue !== false;
  });

  function isComposer(element) {
    if (!(element instanceof HTMLElement)) return false;
    if (element.matches("textarea, input[type='text'], [contenteditable='true'], [role='textbox']")) {
      const label = `${element.getAttribute("aria-label") || ""} ${element.getAttribute("data-placeholder") || ""} ${element.getAttribute("placeholder") || ""}`.toLowerCase();
      return !/(search|ara|title|başlık)/.test(label);
    }
    return false;
  }

  function findComposer(preferred) {
    if (isComposer(preferred)) return preferred;
    if (preferred instanceof HTMLFormElement) {
      const inForm = [...preferred.querySelectorAll("textarea, input[type='text'], [contenteditable='true'], [role='textbox']")]
        .filter((node) => isComposer(node) && node.getClientRects().length);
      return inForm.at(-1) || null;
    }
    const active = document.activeElement;
    if (isComposer(active)) return active;
    const selectors = [
      "textarea#prompt-textarea",
      "[contenteditable='true'][data-testid*='composer']",
      "[contenteditable='true'][role='textbox']",
      "textarea",
      "[contenteditable='true']"
    ];
    for (const selector of selectors) {
      const candidates = [...document.querySelectorAll(selector)].filter((node) => isComposer(node) && node.getClientRects().length);
      if (candidates.length) return candidates[candidates.length - 1];
    }
    return null;
  }

  function readComposer(composer) {
    return composer.isContentEditable ? composer.innerText : composer.value;
  }

  function writeComposer(composer, value) {
    composer.focus();
    if (composer instanceof HTMLTextAreaElement || composer instanceof HTMLInputElement) {
      const prototype = composer instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(prototype, "value").set.call(composer, value);
    } else {
      composer.textContent = value;
    }
    composer.dispatchEvent(new InputEvent("input", {bubbles: true, inputType: "insertText", data: value}));
    composer.dispatchEvent(new Event("change", {bubbles: true}));
  }

  async function maskPrompt(text) {
    const payload = await chrome.runtime.sendMessage({type: "turkce-dlp-detect", text});
    if (!payload || payload.error || !Array.isArray(payload.detections)) throw new Error("DLP API request failed");
    const nonce = crypto.getRandomValues(new Uint32Array(1))[0].toString(36).slice(0, 8).padEnd(4, "0");
    return TurkceDlpCore.maskDetections(text, payload.detections || [], nonce);
  }

  function showError(message = "Türkçe LLM DLP: Kontrol tamamlanamadı; mesaj gönderilmedi.") {
    let banner = document.getElementById("turkce-dlp-error");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "turkce-dlp-error";
      banner.setAttribute("role", "alert");
      Object.assign(banner.style, {
        position: "fixed", zIndex: "2147483647", inset: "12px 12px auto auto",
        maxWidth: "360px", padding: "12px 16px", borderRadius: "8px",
        color: "#fff", background: "#9b1c1c", font: "14px sans-serif",
        boxShadow: "0 2px 12px #0006"
      });
      document.documentElement.appendChild(banner);
    }
    banner.textContent = message;
    setTimeout(() => banner.remove(), 6000);
  }

  function nextMappings() {
    return activeMappings.flatMap((mapping) => [...mapping.values.entries()]);
  }

  function isGenerating() {
    return Boolean(document.querySelector(
      "[data-testid*='stop'], button[aria-label*='Stop' i], button[aria-label*='Durdur' i]"
    ));
  }

  function retireWhenComplete(mapping) {
    if (!mapping.seenResponse || isGenerating()) return;
    clearTimeout(mapping.finishTimer);
    mapping.finishTimer = setTimeout(() => {
      if (!isGenerating()) {
        const index = activeMappings.indexOf(mapping);
        if (index >= 0) activeMappings.splice(index, 1);
        clearTimeout(mapping.expiryTimer);
      }
    }, 30_000);
  }

  function restoreVisibleText(root) {
    const mappings = nextMappings();
    if (!mappings.length || !(root instanceof Node)) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || parent.closest("textarea, input, [contenteditable='true'], #turkce-dlp-error")) return NodeFilter.FILTER_REJECT;
        return mappings.some(([placeholder]) => node.nodeValue.includes(placeholder)) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      for (const mapping of activeMappings) {
        const inAssistantMessage = node.parentElement?.closest(
          "[data-message-author-role='assistant'], [data-testid*='assistant-message']"
        );
        if (inAssistantMessage && [...mapping.values.keys()].some((placeholder) => node.nodeValue.includes(placeholder))) {
          mapping.seenResponse = true;
          retireWhenComplete(mapping);
        }
      }
      node.nodeValue = TurkceDlpCore.restoreText(node.nodeValue, new Map(mappings));
    }
  }

  const observer = new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === "characterData") restoreVisibleText(record.target.parentNode);
      else for (const node of record.addedNodes) if (node.nodeType === Node.ELEMENT_NODE || node.nodeType === Node.TEXT_NODE) restoreVisibleText(node);
    }
  });
  observer.observe(document.documentElement, {subtree: true, childList: true, characterData: true});

  async function intercept(event, composer, replay) {
    if (!enabled || replayingSubmit || pendingSubmit || !composer) return;
    const text = readComposer(composer);
    if (!text || !text.trim()) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    pendingSubmit = true;
    try {
      const masked = await maskPrompt(text);
      if (masked.values.size) {
        writeComposer(composer, masked.masked);
        await new Promise((resolve) => requestAnimationFrame(resolve));
        if (readComposer(composer) !== masked.masked) {
          writeComposer(composer, text);
          throw new Error("Composer update could not be verified");
        }
        const mapping = {values: masked.values, seenResponse: false, finishTimer: null};
        activeMappings.push(mapping);
        mapping.expiryTimer = setTimeout(() => {
          const index = activeMappings.indexOf(mapping);
          if (index >= 0) activeMappings.splice(index, 1);
        }, 5 * 60_000);
      }
      replayingSubmit = true;
      replay();
      queueMicrotask(() => { replayingSubmit = false; });
    } catch (_error) {
      showError();
    } finally {
      pendingSubmit = false;
    }
  }

  function isSendButton(button) {
    if (!(button instanceof HTMLButtonElement)) return false;
    const marker = `${button.getAttribute("aria-label") || ""} ${button.getAttribute("data-testid") || ""} ${button.innerText || ""}`.toLowerCase();
    return /(send|submit|gönder|mesajı gönder)/.test(marker) && !/(stop|cancel|durdur|iptal)/.test(marker);
  }

  document.addEventListener("click", (event) => {
    const button = event.target instanceof Element ? event.target.closest("button") : null;
    if (!isSendButton(button)) return;
    const composer = findComposer(button.form || button.closest("form"));
    if (enabled && !composer) {
      event.preventDefault();
      event.stopImmediatePropagation();
      showError("Türkçe LLM DLP: Mesaj alanı doğrulanamadı; gönderim durduruldu.");
      return;
    }
    void intercept(event, composer, () => button.click());
  }, true);

  document.addEventListener("submit", (event) => {
    if (replayingSubmit || pendingSubmit) return;
    const form = event.target;
    const composer = findComposer(form);
    if (!composer) {
      if (enabled && isSendButton(event.submitter)) {
        event.preventDefault();
        event.stopImmediatePropagation();
        showError("Türkçe LLM DLP: Mesaj alanı doğrulanamadı; gönderim durduruldu.");
      }
      return;
    }
    const submitter = event.submitter;
    void intercept(event, composer, () => {
      if (submitter instanceof HTMLButtonElement) submitter.click();
      else form.requestSubmit();
    });
  }, true);

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing || !isComposer(event.target)) return;
    const composer = event.target;
    const form = composer.closest("form");
    const button = form?.querySelector("button[type='submit']") || [...document.querySelectorAll("button")].find(isSendButton);
    if (!button) {
      event.preventDefault();
      event.stopImmediatePropagation();
      showError("Türkçe LLM DLP: Gönderim düğmesi doğrulanamadı; mesaj gönderilmedi.");
      return;
    }
    void intercept(event, composer, () => button.click());
  }, true);
})();
