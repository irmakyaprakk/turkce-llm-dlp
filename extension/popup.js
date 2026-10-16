"use strict";

const enabledInput = document.getElementById("enabled");
const previewInput = document.getElementById("preview");
const status = document.getElementById("status");
const conversation = document.getElementById("conversation");
const clearButton = document.getElementById("clear");
let activeTabId = null;

function render() {
  status.textContent = enabledInput.checked ? "Koruma açık" : "Koruma kapalı";
}

chrome.storage.local.get({enabled: true, preview: true}, (settings) => {
  enabledInput.checked = settings.enabled !== false;
  previewInput.checked = settings.preview !== false;
  render();
});
enabledInput.addEventListener("change", () => chrome.storage.local.set({enabled: enabledInput.checked}, render));
previewInput.addEventListener("change", () => chrome.storage.local.set({preview: previewInput.checked}));

chrome.tabs.query({active: true, currentWindow: true}, ([tab]) => {
  if (!tab?.id) return;
  activeTabId = tab.id;
  chrome.tabs.sendMessage(tab.id, {type: "turkce-dlp-status"}, (response) => {
    if (chrome.runtime.lastError || !response) {
      conversation.textContent = "Bu sayfada koruma yok (yalnız ChatGPT ve Claude).";
      return;
    }
    conversation.textContent = `Bu sohbette ${response.size} değer maskelendi.`;
    clearButton.disabled = response.size === 0;
  });
});

clearButton.addEventListener("click", () => {
  if (activeTabId === null) return;
  chrome.tabs.sendMessage(activeTabId, {type: "turkce-dlp-clear-conversation"}, (response) => {
    if (response?.ok) {
      conversation.textContent = "Eşleme silindi. Önceki yanıtlardaki yer tutucular artık geri çevrilmez.";
      clearButton.disabled = true;
    }
  });
});

const service = document.getElementById("service");
chrome.runtime.sendMessage({type: "turkce-dlp-health"}, (response) => {
  const up = Boolean(response?.ok);
  service.textContent = up ? "Yerel servis: çalışıyor" : "Yerel servis: kapalı — mesajlar gönderilmez";
  service.className = up ? "ok" : "down";
});
