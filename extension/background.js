"use strict";

importScripts("extract.js");

// Test sürümünde (scripts/e2e) bu küme yerel sahte sohbet sayfasıyla genişletilir.
const ALLOWED_SITES = new Set(["chatgpt.com", "www.chatgpt.com", "claude.ai", "www.claude.ai"]);
const API_URL = "http://127.0.0.1:8000/v1/detect";
const AUDIT_URL = API_URL.replace("/v1/detect", "/v1/audit/event");
const AUDIT_OUTCOMES = new Set(["masked", "clean", "blocked", "cancelled", "leak_detected"]);
// Tespit ağı: sayfa kilidini atlayan (ör. Web Worker'dan giden) gönderimleri görmek için izlenen siteler
const WATCHED_URLS = ["https://chatgpt.com/*", "https://claude.ai/*"];
const APPROVAL_MS = 30_000;
const APPROVAL_KEY = "network-approvals";

// Yalnız {"PERSON": 2} biçimli sayılar geçer; değer ya da metin taşıyabilecek her şey atılır.
function typeCounts(value) {
  const counts = {};
  if (!value || typeof value !== "object") return counts;
  for (const [kind, count] of Object.entries(value)) {
    if (/^[A-Z_]{2,20}$/.test(kind) && Number.isInteger(count) && count > 0 && count <= 10000) counts[kind] = count;
  }
  return counts;
}
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

async function postAudit(event) {
  try {
    await fetch(AUDIT_URL, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(event),
      cache: "no-store",
      credentials: "omit",
      signal: AbortSignal.timeout(3000)
    });
  } catch (_error) {
    // servis kapalıysa kayıt düşer; mesaj akışı bundan etkilenmez
  }
}

// Onaylanan (maskelenmiş ya da temiz) metinler: yalnız oturum belleğinde, ~30 sn. Ham metin buraya gelmez.
async function readApprovals() {
  const stored = (await chrome.storage.session.get(APPROVAL_KEY))[APPROVAL_KEY] || {};
  const now = Date.now();
  return Object.fromEntries(Object.entries(stored).filter(([, until]) => until > now));
}

async function addApproval(text) {
  const approvals = await readApprovals();
  approvals[TurkceDlpExtract.normalize(text)] = Date.now() + APPROVAL_MS;
  await chrome.storage.session.set({[APPROVAL_KEY]: approvals});
}

async function handle(message, sender) {
  const host = senderHost(sender);
  if (!ALLOWED_SITES.has(host)) return {error: true};

  if (message.type === "turkce-dlp-approve") {
    if (typeof message.text !== "string" || message.text.length > 20_000) return {error: true};
    await addApproval(message.text);
    return {ok: true};
  }

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
    // ner: "unavailable" ise kişi/kurum/yer modeli bu istekte çalışmadı (içerik betiği kullanıcıyı uyarır)
    return {detections: payload.detections, ner: typeof payload.ner === "string" ? payload.ner : "off"};
  }

  // Anonim denetim kaydı: site adı + karar + tür başına sayılar. Başarısızlık gönderimi etkilemez.
  if (message.type === "turkce-dlp-audit") {
    if (!AUDIT_OUTCOMES.has(message.outcome)) return {error: true};
    const event = {
      source: "extension",
      site: host.replace(/^www\./, ""),
      outcome: message.outcome,
      detected: typeCounts(message.detected),
      masked: typeCounts(message.masked),
      unchecked: typeCounts(message.unchecked)
    };
    await postAudit(event);
    return {ok: true};
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

// Popup'taki servis durumu: yalnızca sağlık uç noktası; metin göndermez.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "turkce-dlp-health" || !sender.url?.startsWith(chrome.runtime.getURL(""))) return false;
  fetch(API_URL.replace("/v1/detect", "/health"), {cache: "no-store", credentials: "omit", signal: AbortSignal.timeout(2000)})
    .then(async (response) => {
      const body = response.ok ? await response.json().catch(() => ({})) : {};
      sendResponse({ok: response.ok, ner: body.ner});
    }, () => sendResponse({ok: false}));
  return true;
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (typeof message?.type !== "string" || !message.type.startsWith("turkce-dlp-") || message.type === "turkce-dlp-health") {
    return false;
  }
  handle(message, sender).then(sendResponse, () => sendResponse({error: true}));
  return true;
});

// ---------------------------------------------------------------------------
// Tespit ağı (webRequest): MV3'te istek ENGELLENEMEZ, yalnız gözlenir. Sayfa kilidini (guard.js) atlayan bir
// gönderim (Web Worker / SharedWorker, kilit kurulmadan açılmış sekme) görülürse kullanıcı uyarılır ve denetim
// günlüğüne yalnız "leak_detected" sonucu yazılır (metin ya da değer yazılmaz).
// ---------------------------------------------------------------------------
const recentLeaks = new Map(); // sekme → son uyarı anı (site isteği yeniden denediğinde tek uyarı)

function bodyAnalysis(requestBody) {
  if (requestBody?.formData) return TurkceDlpExtract.fromFormFields(requestBody.formData);
  if (Array.isArray(requestBody?.raw) && requestBody.raw.every((part) => part.bytes)) {
    const decoder = new TextDecoder();
    const text = requestBody.raw.map((part) => decoder.decode(part.bytes, {stream: true})).join("") + decoder.decode();
    return TurkceDlpExtract.fromString(text);
  }
  return null; // gövde görülemiyor (akış ya da dosya): doğrulanamaz
}

async function isUnapproved(analysis) {
  const approvals = await readApprovals();
  return analysis.texts.some((text) => {
    const key = TurkceDlpExtract.normalize(text);
    return key && !approvals[key];
  });
}

async function inspectRequest(details) {
  if (details.method !== "POST" || details.tabId < 0) return;
  const kind = TurkceDlpExtract.endpointKind(details.url);
  if (!kind) return;
  const {enabled} = await chrome.storage.local.get({enabled: true});
  if (enabled === false) return;
  const analysis = bodyAnalysis(details.requestBody);
  if (!analysis) return;
  if (kind === "retry" && !analysis.recognized) return;
  if (analysis.recognized && !(await isUnapproved(analysis))) return;
  // Onay mesajı istekten biraz sonra gelebilir: kısa bekleyip yeniden bak
  await new Promise((resolve) => setTimeout(resolve, 1500));
  if (analysis.recognized && !(await isUnapproved(analysis))) return;
  const last = recentLeaks.get(details.tabId) || 0;
  if (Date.now() - last < 10_000) return;
  recentLeaks.set(details.tabId, Date.now());
  chrome.action.setBadgeBackgroundColor({tabId: details.tabId, color: "#9b1c1c"}).catch(() => {});
  chrome.action.setBadgeText({tabId: details.tabId, text: "!"}).catch(() => {});
  chrome.tabs.sendMessage(details.tabId, {type: "turkce-dlp-leak"}).catch(() => {});
  let site = "";
  try {
    site = new URL(details.url).hostname.replace(/^www\./, "");
  } catch (_error) {
    return;
  }
  await postAudit({source: "extension", site, outcome: "leak_detected", detected: {}, masked: {}, unchecked: {}});
}

chrome.webRequest.onBeforeRequest.addListener(
  (details) => {
    inspectRequest(details).catch(() => {});
  },
  {urls: WATCHED_URLS},
  ["requestBody"]
);

// Sekme yeni sayfaya geçince uyarı rozeti temizlenir
chrome.tabs.onUpdated.addListener((tabId, info) => {
  if (info.status === "loading") chrome.action.setBadgeText({tabId, text: ""}).catch(() => {});
});
