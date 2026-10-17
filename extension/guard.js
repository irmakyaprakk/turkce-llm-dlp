// Ağ kilidi (fail-closed): sayfanın kendi dünyasında (MAIN), site betiklerinden önce çalışır.
// ChatGPT/Claude'a giden POST gövdesindeki kullanıcı metni, içerik betiğinin kısa süre önce onayladığı
// metinle eşleşmiyorsa istek gönderilmez. Böylece eklentinin olay yakalamasını atlayan bir gönderim yolu
// (geç yüklenen betik, başka bir arayüz, bilinmeyen kısayol) ham metni dışarı çıkaramaz.
(() => {
  "use strict";
  if (window.__turkceDlpGuard) return;
  window.__turkceDlpGuard = true;

  const APPROVAL_MS = 30_000; // site isteği yeniden deneyebilir: onay tek kullanımlık değil
  const approvals = new Map(); // normalize metin → son geçerlilik anı
  let enabled = true; // içerik betiği aksini söyleyene kadar kilitli

  const normalize = (text) => String(text).replace(/\s+/g, " ").trim();

  document.addEventListener("turkce-dlp-approve", (event) => {
    try {
      const text = normalize(JSON.parse(event.detail));
      if (text) approvals.set(text, Date.now() + APPROVAL_MS);
    } catch (_error) { /* bozuk olay yok sayılır */ }
  });
  document.addEventListener("turkce-dlp-config", (event) => {
    try {
      enabled = JSON.parse(event.detail).enabled !== false;
    } catch (_error) { /* bozuk olay yok sayılır */ }
  });

  // İçerik betiği bu betikten önce yüklendiyse ayarı yeniden göndersin
  document.dispatchEvent(new CustomEvent("turkce-dlp-config-request"));

  function isApproved(text) {
    const now = Date.now();
    for (const [key, until] of approvals) if (until < now) approvals.delete(key);
    return approvals.has(normalize(text));
  }

  // Gövdeden kullanıcı metni: form/JSON "prompt" alanı (ChatGPT oturumsuz, Claude) ya da
  // son kullanıcı mesajının content.parts alanı (ChatGPT oturumlu). Bulunamazsa null.
  function fromJson(data) {
    if (!data || typeof data !== "object") return null;
    if (typeof data.prompt === "string") return data.prompt;
    if (Array.isArray(data.messages)) {
      const user = data.messages.filter((m) => m?.author?.role === "user" || m?.role === "user").at(-1);
      const parts = user?.content?.parts;
      if (Array.isArray(parts)) {
        const texts = parts.filter((part) => typeof part === "string");
        if (texts.length) return texts.join("\n");
      }
      if (typeof user?.content === "string") return user.content;
    }
    return null;
  }

  function fromString(body) {
    const trimmed = body.trimStart();
    if (trimmed.startsWith("{")) {
      try {
        return fromJson(JSON.parse(trimmed));
      } catch (_error) {
        return null;
      }
    }
    if (/(^|&)prompt=/.test(body)) return new URLSearchParams(body).get("prompt");
    return null;
  }

  function extractSync(body) {
    if (body == null) return null;
    if (typeof body === "string") return fromString(body);
    if (body instanceof URLSearchParams) return body.get("prompt");
    if (body instanceof FormData) {
      const value = body.get("prompt");
      return typeof value === "string" ? value : null;
    }
    if (body instanceof ArrayBuffer || ArrayBuffer.isView(body)) {
      return fromString(new TextDecoder().decode(body));
    }
    return undefined; // Blob/ReadableStream: eşzamanlı okunamaz
  }

  function isWrite(method) {
    return !/^(GET|HEAD|OPTIONS)$/i.test(method || "GET");
  }

  function blocked(url) {
    document.dispatchEvent(new CustomEvent("turkce-dlp-blocked", {detail: JSON.stringify({url: String(url).slice(0, 200)})}));
    return new TypeError("Türkçe LLM DLP: Kontrol edilmemiş gönderim engellendi");
  }

  // true: gönderilebilir
  function allowText(text) {
    return !enabled || text == null || !normalize(text) || isApproved(text);
  }

  const originalFetch = window.fetch;
  window.fetch = function fetch(input, init) {
    const request = input instanceof Request ? input : null;
    const method = init?.method || request?.method || "GET";
    if (!enabled || !isWrite(method)) return originalFetch.apply(this, arguments);
    const url = request ? request.url : input;
    const body = init && "body" in init ? init.body : null;
    let text = extractSync(body);
    if (text === null && request && !(init && "body" in init)) text = undefined; // gövde Request içinde
    if (text !== undefined) {
      if (!allowText(text)) return Promise.reject(blocked(url));
      return originalFetch.apply(this, arguments);
    }
    // Eşzamansız gövde: okunur, karar sonra verilir (gövdenin kopyası okunur, asıl istek bozulmaz)
    const args = arguments;
    const self = this;
    return (async () => {
      let raw = "";
      try {
        if (body instanceof Blob) raw = await body.text();
        else if (body == null && request) raw = await request.clone().text();
        else if (body instanceof ReadableStream) {
          const [a, b] = body.tee();
          raw = await new Response(a).text();
          args[1] = {...init, body: b, duplex: "half"};
        }
      } catch (_error) {
        raw = "";
      }
      const found = fromString(raw);
      if (!allowText(found)) throw blocked(url);
      return originalFetch.apply(self, args);
    })();
  };

  const xhrOpen = XMLHttpRequest.prototype.open;
  const xhrSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function open(method, url) {
    this.__turkceDlp = {method, url};
    return xhrOpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function send(body) {
    const info = this.__turkceDlp || {};
    if (enabled && isWrite(info.method)) {
      const text = extractSync(body);
      if (text !== undefined && !allowText(text)) throw blocked(info.url);
    }
    return xhrSend.apply(this, arguments);
  };

  if (navigator.sendBeacon) {
    const beacon = navigator.sendBeacon.bind(navigator);
    navigator.sendBeacon = function sendBeacon(url, data) {
      const text = extractSync(data);
      if (text !== undefined && !allowText(text)) {
        blocked(url);
        return false;
      }
      return beacon(url, data);
    };
  }
})();
