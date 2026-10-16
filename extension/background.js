"use strict";

// Test sürümünde (scripts/e2e) bu küme yerel sahte sohbet sayfasıyla genişletilir.
const ALLOWED_SITES = new Set(["chatgpt.com", "www.chatgpt.com", "claude.ai", "www.claude.ai"]);
const API_URL = "http://127.0.0.1:8000/v1/detect";
const VAULT_TTL_MS = 24 * 60 * 60 * 1000;
const VAULT_PREFIX = "vault:";

function senderHost(sender) {
  try {
    return new URL(sender.tab?.url).hostname;
  } catch (_error) {
    return "";
  }
}

// Sohbet kimliği yoksa (yeni sohbet) eşleme sekmeye bağlanır; kimlik oluşunca yeniden adlandırılır.
function storageKey(key, sender) {
  if (typeof key !== "string" || !/^[a-z0-9.-]+:[A-Za-z0-9_-]+$/.test(key)) return null;
  return key.endsWith(":new") ? `${VAULT_PREFIX}${key}:${sender.tab?.id ?? "x"}` : `${VAULT_PREFIX}${key}`;
}

async function readVault(storeKey) {
  const stored = (await chrome.storage.session.get(storeKey))[storeKey];
  if (!stored || Date.now() - (stored.updatedAt || 0) > VAULT_TTL_MS) {
    if (stored) await chrome.storage.session.remove(storeKey);
    return null;
  }
  return stored.state;
}

async function handle(message, sender) {
  const host = senderHost(sender);
  if (!ALLOWED_SITES.has(host)) return {error: true};

  if (message.type === "turkce-dlp-detect") {
    if (typeof message.text !== "string" || message.text.length > 20_000) return {error: true};
    let response;
    try {
      response = await fetch(API_URL, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text: message.text}),
        cache: "no-store",
        credentials: "omit",
        signal: AbortSignal.timeout(8000)
      });
    } catch (_error) {
      return {error: true, reason: "api_unreachable"};
    }
    if (!response.ok) return {error: true, reason: "api_error"};
    const payload = await response.json();
    if (!Array.isArray(payload.detections)) return {error: true};
    return {detections: payload.detections};
  }

  // Kasa yalnız bellekte tutulan oturum deposunda (chrome.storage.session) saklanır; diske yazılmaz,
  // tarayıcı kapanınca silinir. Anahtar her zaman gönderen sekmenin sitesiyle başlamalıdır.
  for (const field of ["key", "from"]) {
    if (message[field] !== undefined && (typeof message[field] !== "string" || !message[field].startsWith(`${host}:`))) {
      return {error: true};
    }
  }
  if (message.type === "turkce-dlp-vault-get") {
    const storeKey = storageKey(message.key, sender);
    return storeKey ? {state: await readVault(storeKey)} : {error: true};
  }
  if (message.type === "turkce-dlp-vault-put") {
    const storeKey = storageKey(message.key, sender);
    if (!storeKey || typeof message.state !== "object") return {error: true};
    await chrome.storage.session.set({[storeKey]: {state: message.state, updatedAt: Date.now()}});
    return {ok: true};
  }
  if (message.type === "turkce-dlp-vault-rename") {
    const from = storageKey(message.from, sender);
    const to = storageKey(message.key, sender);
    if (!from || !to || from === to) return {error: true};
    const state = await readVault(from);
    if (state) {
      await chrome.storage.session.set({[to]: {state, updatedAt: Date.now()}});
      await chrome.storage.session.remove(from);
    }
    return {ok: true};
  }
  if (message.type === "turkce-dlp-vault-clear") {
    const storeKey = storageKey(message.key, sender);
    if (storeKey) await chrome.storage.session.remove(storeKey);
    return {ok: true};
  }
  return {error: true};
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (typeof message?.type !== "string" || !message.type.startsWith("turkce-dlp-")) return false;
  handle(message, sender).then(sendResponse, () => sendResponse({error: true}));
  return true;
});
