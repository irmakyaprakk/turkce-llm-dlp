// Ağ kilidi (fail-closed): sayfanın kendi dünyasında (MAIN), site betiklerinden önce çalışır.
// ChatGPT/Claude'a giden POST gövdesindeki kullanıcı metni, içerik betiğinin kısa süre önce onayladığı
// metinle eşleşmiyorsa istek gönderilmez. Bilinen gönderim adresinde gövde biçimi tanınmazsa da istek durur.
// Böylece olay yakalamayı atlayan bir gönderim yolu (geç yüklenen betik, başka bir arayüz, bilinmeyen kısayol,
// biçim değişikliği, iç çerçevenin temiz fetch'i) ham metni dışarı çıkaramaz.
// Kapsam dışı: Web Worker / SharedWorker istekleri (yalnız arka plandaki webRequest tespit ağı görür).
(() => {
  "use strict";
  const {normalize, endpointKind, fromString} = window.TurkceDlpExtract;
  const APPROVAL_MS = 30_000; // site isteği yeniden deneyebilir: onay tek kullanımlık değil

  // Durum üst pencerede tutulur; aynı kökenli iç çerçeveler aynı nesneyi kullanır.
  let state = null;
  try {
    if (window !== window.top) state = window.top.__turkceDlpGuardState || null;
  } catch (_error) {
    state = null; // farklı köken: kendi durumu, varsayılan kilitli
  }
  if (!state) {
    state = window.__turkceDlpGuardState || {approvals: new Map(), enabled: true};
    if (window === window.top && !window.__turkceDlpGuardState) {
      document.addEventListener("turkce-dlp-approve", (event) => {
        try {
          const text = normalize(JSON.parse(event.detail));
          if (text) state.approvals.set(text, Date.now() + APPROVAL_MS);
        } catch (_error) { /* bozuk olay yok sayılır */ }
      });
      document.addEventListener("turkce-dlp-config", (event) => {
        try {
          state.enabled = JSON.parse(event.detail).enabled !== false;
        } catch (_error) { /* bozuk olay yok sayılır */ }
      });
      // İçerik betiği bu betikten önce yüklendiyse ayarı yeniden göndersin
      document.dispatchEvent(new CustomEvent("turkce-dlp-config-request"));
    }
  }
  if (!window.__turkceDlpGuardState) {
    Object.defineProperty(window, "__turkceDlpGuardState", {value: state, configurable: false});
  }

  function isApproved(text) {
    const now = Date.now();
    for (const [key, until] of state.approvals) if (until < now) state.approvals.delete(key);
    return state.approvals.has(normalize(text));
  }

  function notify(reason) {
    const event = () => new CustomEvent("turkce-dlp-blocked", {detail: JSON.stringify({reason})});
    try {
      window.top.document.dispatchEvent(event());
    } catch (_error) {
      document.dispatchEvent(event());
    }
  }

  // null: geçebilir; aksi hâlde engelleme nedeni ("unapproved" | "format")
  function verdict(url, analysis) {
    if (!state.enabled) return null;
    if (endpointKind(url) === "send" && !analysis.recognized) return "format";
    if (analysis.texts.some((text) => normalize(text) && !isApproved(text))) return "unapproved";
    return null;
  }

  function blockError(reason) {
    notify(reason);
    return new TypeError("Türkçe LLM DLP: gönderim durduruldu");
  }

  const isWrite = (method) => !/^(GET|HEAD|OPTIONS)$/i.test(method || "GET");
  // Pencereler arası tür denetimi: başka çerçevede oluşturulan nesnede instanceof yanlış sonuç verir
  const tag = (value) => Object.prototype.toString.call(value).slice(8, -1);
  const isBlob = (value) => tag(value) === "Blob" || tag(value) === "File";
  const UNREADABLE = {recognized: false, texts: []};

  // Bir pencereye (üst ya da aynı kökenli iç çerçeve) kilidi kurar; o pencerenin kurucuları kullanılır.
  function install(win) {
    if (!win || win.__turkceDlpGuardInstalled) return;
    try {
      Object.defineProperty(win, "__turkceDlpGuardInstalled", {value: true});
    } catch (_error) {
      return;
    }

    // Eşzamanlı okunabilen gövde: {recognized, texts}; okunamıyorsa undefined
    function analyzeSync(body) {
      if (body == null) return UNREADABLE;
      if (typeof body === "string") return fromString(body);
      if (tag(body) === "URLSearchParams") return fromString(body.toString());
      if (tag(body) === "FormData") {
        const value = body.get("prompt");
        return typeof value === "string" ? {recognized: true, texts: [value]} : UNREADABLE;
      }
      if (tag(body) === "ArrayBuffer" || ArrayBuffer.isView(body)) {
        return fromString(new TextDecoder().decode(body));
      }
      return undefined; // Blob/ReadableStream
    }

    const originalFetch = win.fetch;
    win.fetch = function fetch(input, init) {
      const request = tag(input) === "Request" ? input : null;
      const method = init?.method || request?.method || "GET";
      if (!state.enabled || !isWrite(method)) return originalFetch.apply(this, arguments);
      const url = request ? request.url : input;
      const hasInitBody = Boolean(init && "body" in init && init.body != null);
      const body = hasInitBody ? init.body : null;
      const analysis = hasInitBody ? analyzeSync(body) : request ? undefined : UNREADABLE;
      if (analysis !== undefined) {
        const reason = verdict(url, analysis);
        return reason ? Promise.reject(blockError(reason)) : originalFetch.apply(this, arguments);
      }
      // Eşzamansız gövde: kopyası okunur, karar sonra verilir (asıl istek bozulmaz)
      const args = [...arguments];
      const self = this;
      return (async () => {
        let raw = "";
        try {
          if (isBlob(body)) raw = await body.text();
          else if (!hasInitBody && request) raw = await request.clone().text();
          else if (tag(body) === "ReadableStream") {
            const [a, b] = body.tee();
            raw = await new win.Response(a).text();
            args[1] = {...init, body: b, duplex: "half"};
          }
        } catch (_error) {
          raw = "";
        }
        const reason = verdict(url, raw ? fromString(raw) : UNREADABLE);
        if (reason) throw blockError(reason);
        return originalFetch.apply(self, args);
      })();
    };

    const proto = win.XMLHttpRequest.prototype;
    const xhrOpen = proto.open;
    const xhrSend = proto.send;
    proto.open = function open(method, url) {
      this.__turkceDlp = {method, url};
      return xhrOpen.apply(this, arguments);
    };
    proto.send = function send(body) {
      const info = this.__turkceDlp || {};
      if (state.enabled && isWrite(info.method)) {
        // Blob gövdesi eşzamanlı okunamaz: gönderim adresindeyse fail-closed
        const reason = verdict(info.url, analyzeSync(body) ?? UNREADABLE);
        if (reason) throw blockError(reason);
      }
      return xhrSend.apply(this, arguments);
    };

    if (win.navigator.sendBeacon) {
      const beacon = win.navigator.sendBeacon.bind(win.navigator);
      win.navigator.sendBeacon = function sendBeacon(url, data) {
        const reason = verdict(url, analyzeSync(data) ?? UNREADABLE);
        if (reason) {
          blockError(reason);
          return false;
        }
        return beacon(url, data);
      };
    }

    // Aynı kökenli iç çerçeve (about:blank dahil) erişildiği anda kilitlenir: site çerçevenin temiz fetch'ine
    // contentWindow/contentDocument üzerinden ulaşır. Farklı kökenli çerçeveye manifest girdisi (all_frames) kurulur.
    for (const Frame of [win.HTMLIFrameElement, win.HTMLFrameElement].filter(Boolean)) {
      for (const prop of ["contentWindow", "contentDocument"]) {
        const descriptor = Object.getOwnPropertyDescriptor(Frame.prototype, prop);
        if (!descriptor?.get) continue;
        Object.defineProperty(Frame.prototype, prop, {
          ...descriptor,
          get() {
            const value = descriptor.get.call(this);
            try {
              const child = prop === "contentWindow" ? value : value?.defaultView;
              if (child && child.document) install(child);
            } catch (_error) { /* farklı köken */ }
            return value;
          }
        });
      }
    }
  }

  install(window);
})();
