"use strict";

const ALLOWED_SITES = new Set(["chatgpt.com", "www.chatgpt.com", "claude.ai", "www.claude.ai"]);

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "turkce-dlp-detect") return false;
  const senderUrl = sender.tab?.url;
  let hostname = "";
  try {
    hostname = new URL(senderUrl).hostname;
  } catch (_error) {
    sendResponse({error: true});
    return false;
  }
  if (!ALLOWED_SITES.has(hostname) || typeof message.text !== "string" || message.text.length > 20_000) {
    sendResponse({error: true});
    return false;
  }

  fetch("http://127.0.0.1:8000/v1/detect", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: message.text}),
    cache: "no-store",
    credentials: "omit",
    signal: AbortSignal.timeout(5000)
  }).then(async (response) => {
    if (!response.ok) throw new Error("DLP API unavailable");
    const payload = await response.json();
    if (!Array.isArray(payload.detections)) throw new Error("Invalid DLP response");
    sendResponse({detections: payload.detections});
  }).catch(() => sendResponse({error: true}));
  return true;
});
