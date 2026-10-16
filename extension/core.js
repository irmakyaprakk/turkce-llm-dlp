(() => {
  "use strict";

  function pythonOffsetToJsIndex(text, pythonOffset) {
    if (!Number.isInteger(pythonOffset) || pythonOffset < 0) {
      throw new RangeError("Invalid Python character offset");
    }
    let codePoints = 0;
    let codeUnits = 0;
    for (const character of text) {
      if (codePoints === pythonOffset) return codeUnits;
      codePoints += 1;
      codeUnits += character.length;
    }
    if (codePoints === pythonOffset) return codeUnits;
    throw new RangeError("Python character offset exceeds text length");
  }

  function maskDetections(text, detections, nonce) {
    if (typeof text !== "string" || !Array.isArray(detections)) {
      throw new TypeError("Text and detection list are required");
    }
    if (!/^[A-Za-z0-9]{4,12}$/.test(nonce)) throw new TypeError("Invalid placeholder nonce");
    const ordered = [...detections].sort((a, b) => a.start - b.start || b.end - a.end);
    const values = new Map();
    const identityToPlaceholder = new Map();
    const counters = new Map();
    const replacements = [];

    for (const item of ordered) {
      if (!item || typeof item.type !== "string" || !/^[A-Z][A-Z0-9_]{0,31}$/.test(item.type)) {
        throw new TypeError("Valid detection type is required");
      }
      const start = pythonOffsetToJsIndex(text, item.start);
      const end = pythonOffsetToJsIndex(text, item.end);
      if (end <= start) throw new RangeError("Detection span must be non-empty");
      if (replacements.length && start < replacements[replacements.length - 1].end) {
        throw new RangeError("Detection spans overlap");
      }
      const original = text.slice(start, end);
      let canonical = original.normalize("NFKC").replace(/[\u200B\u200C\u200D\uFEFF\u2060]/g, "");
      if (item.type === "TCKN") canonical = canonical.replace(/\D/g, "");
      else if (item.type === "IBAN") canonical = canonical.replace(/[\s-]/g, "").toUpperCase();
      else if (item.type === "CARD") canonical = canonical.replace(/[\s-]/g, "");
      else if (item.type === "PHONE") canonical = canonical.replace(/[^0-9Xx*]/g, "").slice(-10).toUpperCase();
      const identity = `${item.type}\u0000${canonical}`;
      let placeholder = identityToPlaceholder.get(identity);
      if (!placeholder) {
        const next = (counters.get(item.type) || 0) + 1;
        counters.set(item.type, next);
        placeholder = `<${item.type}_${next}_${nonce}>`;
        identityToPlaceholder.set(identity, placeholder);
        values.set(placeholder, original);
      }
      replacements.push({start, end, placeholder});
    }

    let masked = "";
    let cursor = 0;
    for (const item of replacements) {
      masked += text.slice(cursor, item.start) + item.placeholder;
      cursor = item.end;
    }
    masked += text.slice(cursor);
    return {masked, values};
  }

  function restoreText(text, values) {
    let restored = text;
    for (const [placeholder, original] of [...values.entries()].sort((a, b) => b[0].length - a[0].length)) {
      restored = restored.split(placeholder).join(original);
    }
    return restored;
  }

  const api = {pythonOffsetToJsIndex, maskDetections, restoreText};
  globalThis.TurkceDlpCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
