(() => {
  "use strict";

  const {ConversationVault, TYPE_LABELS, conversationKey} = TurkceDlpCore;
  let enabled = true;
  let preview = true;
  let replayingSubmit = false;
  let pendingSubmit = false;
  let currentKey = conversationKey(location.href);
  let vault = new ConversationVault();
  let vaultReady = loadVault();

  chrome.storage.local.get({enabled: true, preview: true}, (settings) => {
    enabled = settings.enabled !== false;
    preview = settings.preview !== false;
  });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    if (changes.enabled) enabled = changes.enabled.newValue !== false;
    if (changes.preview) preview = changes.preview.newValue !== false;
  });

  // ---------------------------------------------------------------------------
  // Sohbet kasası (arka plandaki oturum belleğinde)
  // ---------------------------------------------------------------------------
  async function loadVault() {
    const response = await chrome.runtime.sendMessage({type: "turkce-dlp-vault-get", key: currentKey});
    vault = new ConversationVault(response?.state || {});
    restoreVisibleText(document.body);
  }

  async function saveVault() {
    const response = await chrome.runtime.sendMessage({type: "turkce-dlp-vault-put", key: currentKey, state: vault.toJSON()});
    if (!response?.ok) throw new Error("Vault could not be saved");
  }

  // Yeni sohbetin ilk mesajından sonra site adresi /c/<id> olur: eşleme o sohbete taşınır.
  setInterval(() => {
    const key = conversationKey(location.href);
    if (!key || key === currentKey) return;
    const previous = currentKey;
    currentKey = key;
    vaultReady = (async () => {
      if (previous?.endsWith(":new") && vault.size) {
        await chrome.runtime.sendMessage({type: "turkce-dlp-vault-rename", from: previous, key});
      }
      await loadVault();
    })().catch(() => { vault = new ConversationVault(); });
  }, 800);

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "turkce-dlp-clear-conversation") {
      chrome.runtime.sendMessage({type: "turkce-dlp-vault-clear", key: currentKey}).then(() => {
        vault = new ConversationVault();
        sendResponse({ok: true});
      }, () => sendResponse({error: true}));
      return true;
    }
    if (message?.type === "turkce-dlp-status") {
      sendResponse({key: currentKey, size: vault.size});
      return false;
    }
    return false;
  });

  // ---------------------------------------------------------------------------
  // Mesaj alanı
  // ---------------------------------------------------------------------------
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
      // Zengin metin editörleri (ProseMirror) kendi seçim olaylarını dinler: tüm içeriği seçip metin olarak yaz
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(composer);
      selection.removeAllRanges();
      selection.addRange(range);
      if (!document.execCommand("insertText", false, value)) composer.textContent = value;
    }
    composer.dispatchEvent(new InputEvent("input", {bubbles: true, inputType: "insertText", data: value}));
    composer.dispatchEvent(new Event("change", {bubbles: true}));
  }

  // ---------------------------------------------------------------------------
  // Arayüz: hata bandı ve gönderim öncesi önizleme (Shadow DOM: site CSS'inden etkilenmez)
  // ---------------------------------------------------------------------------
  function showError(message = "Türkçe LLM DLP: Kontrol tamamlanamadı; mesaj gönderilmedi.", tone = "error") {
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
    banner.style.background = tone === "info" ? "#166534" : "#9b1c1c";
    banner.textContent = message;
    setTimeout(() => banner.remove(), tone === "info" ? 2500 : 6000);
  }

  const PANEL_CSS = `
    :host { all: initial; }
    .backdrop { position: fixed; inset: 0; z-index: 2147483647; background: rgb(15 23 42 / .45);
      display: flex; align-items: center; justify-content: center; font: 14px/1.45 system-ui, sans-serif; }
    .panel { width: min(480px, calc(100vw - 32px)); max-height: calc(100vh - 64px); overflow: auto; background: #fff;
      color: #172033; border-radius: 12px; box-shadow: 0 12px 40px rgb(0 0 0 / .3); padding: 20px; }
    h2 { margin: 0 0 4px; font-size: 17px; }
    p { margin: 0 0 14px; color: #49566d; }
    ul { list-style: none; margin: 0 0 16px; padding: 0; border: 1px solid #e3e7ef; border-radius: 8px; }
    li { display: flex; gap: 10px; align-items: flex-start; padding: 9px 12px; border-top: 1px solid #eef1f6; }
    li:first-child { border-top: 0; }
    .type { flex: none; min-width: 92px; font-weight: 600; color: #334; }
    .value { word-break: break-word; }
    .ph { display: block; color: #6b7690; font-size: 12px; }
    li.blocked .type { color: #9b1c1c; }
    .actions { display: flex; justify-content: flex-end; gap: 8px; }
    button { font: inherit; border-radius: 8px; padding: 8px 14px; cursor: pointer; border: 1px solid #c9d1df; background: #fff; color: #172033; }
    button.primary { background: #1d4ed8; border-color: #1d4ed8; color: #fff; }
    @media (prefers-color-scheme: dark) {
      .panel { background: #1b2130; color: #e8ecf4; } p, .ph { color: #a3adc2; } .type { color: #d5dbe8; }
      ul, li { border-color: #2d3548; } button { background: #232a3b; color: #e8ecf4; border-color: #3a4459; }
      button.primary { background: #3b6cf0; border-color: #3b6cf0; }
    }`;

  // Dönüş: maskelenecek tespitlerin listesi ya da null (vazgeçildi)
  function confirmDetections(text, items, blocked) {
    return new Promise((resolve) => {
      const host = document.createElement("div");
      host.id = "turkce-dlp-preview";
      const root = host.attachShadow({mode: "open"});
      const style = document.createElement("style");
      style.textContent = PANEL_CSS;
      const backdrop = document.createElement("div");
      backdrop.className = "backdrop";
      const panel = document.createElement("div");
      panel.className = "panel";
      panel.setAttribute("role", "dialog");
      panel.setAttribute("aria-modal", "true");
      const title = document.createElement("h2");
      const info = document.createElement("p");
      const list = document.createElement("ul");
      const actions = document.createElement("div");
      actions.className = "actions";
      const toCodePoints = [...text];
      const valueOf = (item) => toCodePoints.slice(item.start, item.end).join("");
      const checks = [];

      if (blocked.length) {
        title.textContent = "Bu mesaj gönderilemez";
        info.textContent = "Kurum politikası aşağıdaki bilgilerin dış yapay zekâ servislerine gönderilmesini engelliyor. Mesajı düzenleyip tekrar deneyin.";
        for (const item of blocked) {
          const row = document.createElement("li");
          row.className = "blocked";
          const type = document.createElement("span");
          type.className = "type";
          type.textContent = TYPE_LABELS[item.type] || item.type;
          const value = document.createElement("span");
          value.className = "value";
          value.textContent = valueOf(item);
          row.append(type, value);
          list.append(row);
        }
      } else {
        title.textContent = "Kişisel veri bulundu";
        info.textContent = "İşaretli bilgiler gönderilmeden önce yer tutucuyla değiştirilecek; yanıt geldiğinde ekranda geri yerleştirilir. Kişisel veri olmayanların işaretini kaldırabilirsiniz.";
        for (const item of items) {
          const row = document.createElement("li");
          const box = document.createElement("input");
          box.type = "checkbox";
          box.checked = true;
          box.setAttribute("aria-label", `${TYPE_LABELS[item.type] || item.type} maskelensin`);
          checks.push([box, item]);
          const type = document.createElement("span");
          type.className = "type";
          type.textContent = TYPE_LABELS[item.type] || item.type;
          const value = document.createElement("span");
          value.className = "value";
          value.textContent = valueOf(item);
          row.append(box, type, value);
          list.append(row);
        }
      }

      const close = (result) => {
        host.remove();
        document.removeEventListener("keydown", onKey, true);
        resolve(result);
      };
      const cancel = document.createElement("button");
      cancel.textContent = blocked.length ? "Tamam" : "Vazgeç";
      cancel.addEventListener("click", () => close(null));
      actions.append(cancel);
      let confirm = null;
      if (!blocked.length) {
        confirm = document.createElement("button");
        confirm.className = "primary";
        confirm.textContent = "Maskele ve gönder";
        confirm.addEventListener("click", () => close(checks.filter(([box]) => box.checked).map(([, item]) => item)));
        actions.append(confirm);
      }
      const onKey = (event) => {
        if (event.key === "Escape") { event.preventDefault(); event.stopImmediatePropagation(); close(null); }
        else if (event.key === "Enter" && confirm && !event.shiftKey) { event.preventDefault(); event.stopImmediatePropagation(); confirm.click(); }
      };
      document.addEventListener("keydown", onKey, true);
      panel.append(title, info, list, actions);
      backdrop.append(panel);
      root.append(style, backdrop);
      document.documentElement.appendChild(host);
      (confirm || cancel).focus();
    });
  }

  // ---------------------------------------------------------------------------
  // Yanıtta geri eşleme
  // ---------------------------------------------------------------------------
  function restoreVisibleText(root) {
    if (!vault.size || !(root instanceof Node)) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || parent.closest("textarea, input, [contenteditable='true'], #turkce-dlp-error, #turkce-dlp-preview")) {
          return NodeFilter.FILTER_REJECT;
        }
        return /[[_]/.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const restored = vault.restore(node.nodeValue);
      if (restored !== node.nodeValue) node.nodeValue = restored;
    }
  }

  const observer = new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === "characterData") restoreVisibleText(record.target.parentNode);
      else for (const node of record.addedNodes) if (node.nodeType === Node.ELEMENT_NODE || node.nodeType === Node.TEXT_NODE) restoreVisibleText(node);
    }
  });
  observer.observe(document.documentElement, {subtree: true, childList: true, characterData: true});

  // ---------------------------------------------------------------------------
  // Gönderimi yakalama
  // ---------------------------------------------------------------------------
  async function intercept(event, composer, replay) {
    if (!enabled || replayingSubmit || pendingSubmit || !composer) return;
    const text = readComposer(composer);
    if (!text || !text.trim()) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    pendingSubmit = true;
    try {
      await vaultReady;
      const payload = await chrome.runtime.sendMessage({type: "turkce-dlp-detect", text});
      if (!payload || payload.error || !Array.isArray(payload.detections)) {
        throw new Error(payload?.reason || "api_error");
      }
      const blocked = payload.detections.filter((item) => item.action === "block");
      let selected = payload.detections.filter((item) => item.action !== "block" && item.action !== "allow");
      if (blocked.length) {
        await confirmDetections(text, selected, blocked);
        return;
      }
      if (selected.length && preview) {
        selected = await confirmDetections(text, selected, []);
        if (selected === null) return; // kullanıcı vazgeçti; mesaj alanı olduğu gibi kalır
      }
      if (!selected.length) showError("Türkçe LLM DLP: Kişisel veri bulunmadı, mesaj olduğu gibi gönderildi.", "info");
      if (selected.length) {
        const {masked} = vault.mask(text, selected);
        await saveVault();
        writeComposer(composer, masked);
        await new Promise((resolve) => requestAnimationFrame(resolve));
        if (readComposer(composer).trim() !== masked.trim()) {
          writeComposer(composer, text);
          throw new Error("Composer update could not be verified");
        }
      }
      replayingSubmit = true;
      replay();
      queueMicrotask(() => { replayingSubmit = false; });
    } catch (error) {
      showError(explain(error));
    } finally {
      pendingSubmit = false;
    }
  }

  // Kullanıcıya nedeni söyleyen hata metni (gönderim her durumda durdurulmuştur)
  function explain(error) {
    const message = String(error?.message || "");
    if (!chrome.runtime?.id || /context invalidated/i.test(message)) {
      return "Türkçe LLM DLP güncellendi: lütfen bu sayfayı yenileyin. Mesaj gönderilmedi.";
    }
    if (message === "api_unreachable") {
      return "Türkçe LLM DLP: Yerel servise (127.0.0.1:8000) ulaşılamadı. Terminalde servisi başlatın. Mesaj gönderilmedi.";
    }
    if (message === "api_error") {
      return "Türkçe LLM DLP: Yerel servis isteği reddetti (mesaj çok uzun ya da servis hatası). Mesaj gönderilmedi.";
    }
    return "Türkçe LLM DLP: Kontrol tamamlanamadı; mesaj gönderilmedi.";
  }

  function isSendButton(button) {
    if (!(button instanceof HTMLButtonElement)) return false;
    const marker = `${button.getAttribute("aria-label") || ""} ${button.getAttribute("data-testid") || ""} ${button.innerText || ""}`.toLowerCase();
    return /(send|submit|gönder|mesajı gönder)/.test(marker) && !/(stop|cancel|durdur|iptal)/.test(marker);
  }

  document.addEventListener("click", (event) => {
    const button = event.target instanceof Element ? event.target.closest("button") : null;
    if (!isSendButton(button) || event.composedPath().some((node) => node?.id === "turkce-dlp-preview")) return;
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
    if (pendingSubmit) { event.preventDefault(); event.stopImmediatePropagation(); return; }
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
