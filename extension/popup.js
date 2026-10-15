"use strict";

const enabledInput = document.getElementById("enabled");
const status = document.getElementById("status");

chrome.storage.local.get({enabled: true}, (settings) => {
  enabledInput.checked = settings.enabled !== false;
  status.textContent = enabledInput.checked ? "Koruma açık" : "Koruma kapalı";
});

enabledInput.addEventListener("change", () => {
  chrome.storage.local.set({enabled: enabledInput.checked}, () => {
    status.textContent = enabledInput.checked ? "Koruma açık" : "Koruma kapalı";
  });
});
