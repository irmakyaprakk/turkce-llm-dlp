// İstek gövdesinden kullanıcı metnini çıkarır. Ağ kilidi (guard.js, sayfa dünyası) ve arka plandaki tespit ağı
// (background.js, webRequest) aynı kuralları kullanır.
(() => {
  "use strict";
  const root = typeof globalThis !== "undefined" ? globalThis : self;
  if (root.TurkceDlpExtract) return;

  const normalize = (text) => String(text).replace(/\s+/g, " ").trim();

  // Mesaj gönderen uç noktalar (gözlenen: oturumsuz ChatGPT POST /unauth-mweb/conversation/updates, form "prompt").
  // conversation/prepare gibi metin taşımayan uç noktalar listede yok.
  const CHATGPT_SEND = /^\/(backend-api|backend-anon|unauth-mweb)\/(f\/)?conversation(\/updates)?$/;
  const CLAUDE_SEND = /^\/api\/organizations\/[^/]+\/chat_conversations\/[^/]+\/(completion|retry_completion)$/;

  // "send": gövdede kullanıcı metni okunamazsa istek durdurulur. "retry": yeniden üretim, yeni metin taşımaz.
  function endpointKind(url) {
    let parsed;
    try {
      // about:blank iç çerçevede location "about:blank"dır; baseURI üst sayfanın adresini taşır
      parsed = new URL(String(url), root.document?.baseURI || root.location?.href);
    } catch (_error) {
      return null;
    }
    const host = parsed.hostname.replace(/^www\./, "");
    if (host === "chatgpt.com" && CHATGPT_SEND.test(parsed.pathname)) return "send";
    if (host === "claude.ai" && CLAUDE_SEND.test(parsed.pathname)) {
      return parsed.pathname.endsWith("/retry_completion") ? "retry" : "send";
    }
    return null;
  }

  function promptOf(data) {
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

  // Dönüş: {recognized, texts}. recognized=false: bilinen bir biçim değil (gönderim adresinde fail-closed).
  function fromJson(data) {
    if (!data || typeof data !== "object" || Array.isArray(data)) return {recognized: false, texts: []};
    const attached = Array.isArray(data.attachments)
      ? data.attachments.map((item) => item?.extracted_content).filter((text) => typeof text === "string" && text.trim())
      : [];
    const main = promptOf(data);
    if (main !== null || attached.length) return {recognized: true, texts: [main ?? "", ...attached]};
    // Oturumlu ChatGPT'de yeniden üretim / devam: yeni kullanıcı mesajı taşımaz
    if (["variant", "continue"].includes(data.action) && Array.isArray(data.messages)) {
      return {recognized: true, texts: []};
    }
    return {recognized: false, texts: []};
  }

  function fromString(body) {
    const trimmed = body.trimStart();
    if (trimmed.startsWith("{")) {
      try {
        return fromJson(JSON.parse(trimmed));
      } catch (_error) {
        return {recognized: false, texts: []};
      }
    }
    if (/(^|&)prompt=/.test(body)) return {recognized: true, texts: [new URLSearchParams(body).get("prompt") ?? ""]};
    return {recognized: false, texts: []};
  }

  // webRequest requestBody.formData: {alan: [değerler]}
  function fromFormFields(fields) {
    const prompt = fields?.prompt;
    if (Array.isArray(prompt)) return {recognized: true, texts: prompt.map(String)};
    return {recognized: false, texts: []};
  }

  root.TurkceDlpExtract = {normalize, endpointKind, fromJson, fromString, fromFormFields};
})();
