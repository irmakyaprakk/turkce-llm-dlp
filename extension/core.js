(() => {
  "use strict";

  // Yer tutucu etiketleri: LLM'in anlayacağı, okunabilir Türkçe adlar ("[KİŞİ_1]'e yaz").
  const TAGS = {
    PERSON: "KİŞİ",
    ORG: "KURUM",
    LOCATION: "YER",
    ADDRESS: "ADRES",
    TCKN: "TCKN",
    IBAN: "IBAN",
    CARD: "KART",
    EMAIL: "EPOSTA",
    PHONE: "TELEFON",
    SECRET: "GİZLİ"
  };
  const TYPE_LABELS = {
    PERSON: "Kişi",
    ORG: "Kurum",
    LOCATION: "Yer",
    ADDRESS: "Adres",
    TCKN: "T.C. kimlik no",
    IBAN: "IBAN",
    CARD: "Kredi kartı",
    EMAIL: "E-posta",
    PHONE: "Telefon",
    SECRET: "Gizli bilgi"
  };
  const trUpper = (value) => value.toLocaleUpperCase("tr-TR");
  const trLower = (value) => value.toLocaleLowerCase("tr-TR");
  const asciiFold = (value) => value.replace(/[İI]/g, "I").replace(/Ş/g, "S").replace(/Ğ/g, "G").replace(/Ü/g, "U")
    .replace(/Ö/g, "O").replace(/Ç/g, "C");
  // LLM etiketi küçük harfe ya da ASCII'ye çevirebilir: "[kişi_1]", "[KISI_1]" de tanınır.
  const TAG_BY_FOLDED = new Map(Object.entries(TAGS).map(([type, tag]) => [asciiFold(tag), {type, tag}]));
  const PLACEHOLDER_RE = /\[\s*([A-Za-zÇĞİÖŞÜçğıöşü]{3,8})[ _-]?(\d{1,4})\s*\](?:(['’])?([a-zçğıöşü]{1,8})(?![\p{L}]))?/gu;

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

  // ---------------------------------------------------------------------------
  // Türkçe ek uyumu: "[KİŞİ_1]'e" geri eşlenirken asıl değere göre yeniden çekimlenir ("Ayşe'ye", "Murat'a")
  // ---------------------------------------------------------------------------
  const BACK = new Set("aıou");
  const FRONT = new Set("eiöü");
  const HARD = new Set("fstkçşhp");
  const DIGIT_WORDS = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"];
  const POSSESSIVE_ENDINGS = ["si", "sı", "su", "sü", "ğı", "ği", "ğu", "ğü"];
  const SUFFIX_FORMS = [
    ["DAKI", ["daki", "deki", "taki", "teki", "ndaki", "ndeki"]],
    ["DAN", ["dan", "den", "tan", "ten", "ndan", "nden"]],
    ["DA", ["da", "de", "ta", "te", "nda", "nde"]],
    ["IN", ["ın", "in", "un", "ün", "nın", "nin", "nun", "nün"]],
    ["LA", ["la", "le", "yla", "yle"]],
    ["E", ["a", "e", "ya", "ye", "na", "ne"]],
    ["I", ["ı", "i", "u", "ü", "yı", "yi", "yu", "yü", "nı", "ni", "nu", "nü"]]
  ];

  function suffixKind(written) {
    const lowered = trLower(written);
    for (const [kind, forms] of SUFFIX_FORMS) if (forms.includes(lowered)) return kind;
    return null;
  }

  // Okunuş: rakamla biten değerde son rakamın adı ("…67" → "yedi"), aksi hâlde son sözcük
  function pronounced(value) {
    const trimmed = value.trim().replace(/[.\s]+$/u, "");
    const lastDigit = trimmed.match(/(\d)\D*$/u);
    if (lastDigit && /\d[^\p{L}]*$/u.test(trimmed)) return DIGIT_WORDS[Number(lastDigit[1])];
    const words = trLower(trimmed).match(/\p{L}+/gu);
    return words ? words.join(" ") : "a";
  }

  function lastVowel(word) {
    for (const character of [...word].reverse()) if (BACK.has(character) || FRONT.has(character)) return character;
    return null;
  }
  const twoWay = (vowel) => (vowel === null || BACK.has(vowel) ? "a" : "e");
  const fourWay = (vowel) => ({a: "ı", ı: "ı", o: "u", u: "u", e: "i", i: "i", ö: "ü", ü: "ü"})[vowel || "a"];

  function inflect(value, kind) {
    const word = pronounced(value);
    const lastWord = word.split(" ").at(-1);
    const vowel = lastVowel(word);
    const two = twoWay(vowel);
    const four = fourWay(vowel);
    if (lastWord.length > 4 && POSSESSIVE_ENDINGS.some((ending) => lastWord.endsWith(ending))) {
      // "Moda Mahallesi'nde", "Lale Sokağı'na": tamlama ekinden sonra kaynaştırma "n"
      return {DA: `nd${two}`, DAN: `nd${two}n`, DAKI: `nd${two}ki`, E: `n${two}`, I: `n${four}`, IN: `n${four}n`, LA: `yl${two}`}[kind];
    }
    const last = word.at(-1);
    const endsVowel = BACK.has(last) || FRONT.has(last);
    const hard = HARD.has(last);
    switch (kind) {
      case "DA": return `${hard ? "t" : "d"}${two}`;
      case "DAN": return `${hard ? "t" : "d"}${two}n`;
      case "DAKI": return `${hard ? "t" : "d"}${two}ki`;
      case "E": return `${endsVowel ? "y" : ""}${two}`;
      case "I": return `${endsVowel ? "y" : ""}${four}`;
      case "IN": return `${endsVowel ? "n" : ""}${four}n`;
      case "LA": return `${endsVowel ? "y" : ""}l${two}`;
      default: return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Sohbet kasası: aynı sohbet boyunca aynı değer → aynı yer tutucu; yalnız tarayıcı belleğinde tutulur
  // ---------------------------------------------------------------------------
  function canonicalValue(type, original) {
    let canonical = original.normalize("NFKC").replace(/[​‌‍﻿⁠]/g, "");
    if (type === "TCKN") return canonical.replace(/\D/g, "");
    if (type === "IBAN") return canonical.replace(/[\s-]/g, "").toUpperCase();
    if (type === "CARD") return canonical.replace(/[\s-]/g, "");
    if (type === "PHONE") return canonical.replace(/[^0-9Xx*]/g, "").slice(-10).toUpperCase();
    if (type === "EMAIL") return canonical.toLowerCase();
    if (["PERSON", "ORG", "LOCATION", "ADDRESS"].includes(type)) {
      canonical = trLower(canonical).replace(/\s+/g, " ").trim();
    }
    return canonical;
  }

  class ConversationVault {
    constructor(state = {}) {
      this.identities = {...(state.identities || {})}; // "TYPE\0canonical" → placeholder
      this.values = {...(state.values || {})}; // placeholder → özgün değer (ilk görüldüğü yazım)
      this.counters = {...(state.counters || {})}; // etiket → son numara
    }

    toJSON() {
      return {identities: this.identities, values: this.values, counters: this.counters};
    }

    get size() {
      return Object.keys(this.values).length;
    }

    placeholderFor(type, original, sourceText) {
      const identity = `${type}\u0000${canonicalValue(type, original)}`;
      if (this.identities[identity]) return this.identities[identity];
      const tag = TAGS[type] || type;
      let number = this.counters[tag] || 0;
      let placeholder;
      do {
        number += 1;
        placeholder = `[${tag}_${number}]`;
        // Kullanıcının kendi yazdığı "[KİŞİ_1]" benzeri metinle çakışmasın
      } while (sourceText.includes(placeholder) || this.values[placeholder]);
      this.counters[tag] = number;
      this.identities[identity] = placeholder;
      this.values[placeholder] = original;
      return placeholder;
    }

    // detections: [{type, start, end}] (Python kod noktası ofsetleri). Dönüş: maskeli metin ve kullanılan yer tutucular.
    mask(text, detections) {
      if (typeof text !== "string" || !Array.isArray(detections)) {
        throw new TypeError("Text and detection list are required");
      }
      const ordered = [...detections].sort((a, b) => a.start - b.start || b.end - a.end);
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
        replacements.push({start, end, type: item.type, original, placeholder: this.placeholderFor(item.type, original, text)});
      }
      let masked = "";
      let cursor = 0;
      for (const item of replacements) {
        masked += text.slice(cursor, item.start) + item.placeholder;
        cursor = item.end;
      }
      masked += text.slice(cursor);
      return {masked, replacements};
    }

    // Yalnız bu kasanın ürettiği yer tutucular geri eşlenir; LLM'in uydurduğu "[KİŞİ_9]" olduğu gibi kalır.
    restore(text) {
      if (!this.size || typeof text !== "string" || !text.includes("[")) return text;
      return text.replace(PLACEHOLDER_RE, (whole, rawTag, number, apostrophe, written) => {
        const known = TAG_BY_FOLDED.get(asciiFold(trUpper(rawTag)));
        const placeholder = known ? `[${known.tag}_${Number(number)}]` : null;
        const original = placeholder ? this.values[placeholder] : undefined;
        if (original === undefined) return whole;
        if (!written) return original;
        const kind = suffixKind(written);
        if (!kind) return `${original}${apostrophe || ""}${written}`;
        return `${original}'${inflect(original, kind)}`;
      });
    }
  }

  function restoreText(text, vault) {
    return vault instanceof ConversationVault ? vault.restore(text) : text;
  }

  // Sohbet kimliği: ChatGPT "/c/<id>", Claude "/chat/<id>"; henüz kimliği olmayan yeni sohbet "new"
  function conversationKey(url) {
    let parsed;
    try {
      parsed = new URL(url);
    } catch (_error) {
      return null;
    }
    const match = parsed.pathname.match(/\/(?:c|chat)\/([A-Za-z0-9_-]{6,})/);
    return `${parsed.hostname}:${match ? match[1] : "new"}`;
  }

  const api = {
    TAGS,
    TYPE_LABELS,
    ConversationVault,
    conversationKey,
    inflect,
    pythonOffsetToJsIndex,
    restoreText,
    suffixKind
  };
  globalThis.TurkceDlpCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
