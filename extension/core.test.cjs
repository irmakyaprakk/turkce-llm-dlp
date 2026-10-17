"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {ConversationVault, conversationKey, inflect, pythonOffsetToJsIndex} = require("./core.js");

// Python kod noktası ofsetleri: metindeki değerin yerini bulur
function span(text, value, type, from = 0) {
  const index = text.indexOf(value, from);
  assert.notEqual(index, -1, value);
  return {type, start: [...text.slice(0, index)].length, end: [...text.slice(0, index + value.length)].length};
}

test("converts Python code point offsets to JavaScript UTF-16 indexes", () => {
  const text = "😀 TCKN 10000000146";
  const pythonStart = [...text].indexOf("1");
  assert.equal(pythonOffsetToJsIndex(text, pythonStart), text.indexOf("1"));
});

test("readable Turkish placeholders, consistent for repeated values", () => {
  const text = "TCKN 10000000146 / TCKN 100-000-00146, Ayşe ve ayşe";
  const vault = new ConversationVault();
  const {masked} = vault.mask(text, [
    span(text, "10000000146", "TCKN"),
    span(text, "100-000-00146", "TCKN"),
    span(text, "Ayşe", "PERSON"),
    span(text, "ayşe", "PERSON")
  ]);
  assert.equal(masked, "TCKN [TCKN_1] / TCKN [TCKN_1], [KİŞİ_1] ve [KİŞİ_1]");
  assert.equal(vault.size, 2);
});

test("same conversation keeps placeholders across messages and survives serialization", () => {
  const first = new ConversationVault();
  const a = "Murat'a söyle";
  first.mask(a, [span(a, "Murat", "PERSON")]);
  const second = new ConversationVault(JSON.parse(JSON.stringify(first)));
  const b = "Murat ve Elif geliyor";
  const {masked} = second.mask(b, [span(b, "Murat", "PERSON"), span(b, "Elif", "PERSON")]);
  assert.equal(masked, "[KİŞİ_1] ve [KİŞİ_2] geliyor");
});

test("restores with Turkish suffix harmony of the original value", () => {
  const vault = new ConversationVault();
  const text = "Ayşe, Murat, Kadıköy, Moda Mahallesi, 0532 123 45 67";
  vault.mask(text, [
    span(text, "Ayşe", "PERSON"),
    span(text, "Murat", "PERSON"),
    span(text, "Kadıköy", "LOCATION"),
    span(text, "Moda Mahallesi", "LOCATION"),
    span(text, "0532 123 45 67", "PHONE")
  ]);
  const reply = "[KİŞİ_1]'e ve [KİŞİ_2]'e yaz; [YER_1]'da [YER_2]'da buluşun, [TELEFON_1]'i ara. [KİŞİ_1]'nin numarası";
  assert.equal(
    vault.restore(reply),
    "Ayşe'ye ve Murat'a yaz; Kadıköy'de Moda Mahallesi'nde buluşun, 0532 123 45 67'yi ara. Ayşe'nin numarası"
  );
});

test("tolerates LLM rewriting of placeholders and ignores unknown ones", () => {
  const vault = new ConversationVault();
  const text = "Şule";
  vault.mask(text, [span(text, "Şule", "PERSON")]);
  assert.equal(vault.restore("[kişi_1] ve [KISI_1]ye ve [KİŞİ 1] ama [KİŞİ_9]"), "Şule ve Şule'ye ve Şule ama [KİŞİ_9]");
  assert.equal(vault.restore("[KİŞİ_1]'dır"), "Şule'dır"); // tanınmayan ek olduğu gibi kalır
});

test("does not collide with placeholder-looking text the user typed", () => {
  const text = "Şablon [KİŞİ_1] alanına Ali yazılacak";
  const vault = new ConversationVault();
  const {masked} = vault.mask(text, [span(text, "Ali", "PERSON")]);
  assert.equal(masked, "Şablon [KİŞİ_1] alanına [KİŞİ_2] yazılacak");
  assert.equal(vault.restore("[KİŞİ_1] / [KİŞİ_2]"), "[KİŞİ_1] / Ali");
});

test("inflects digits and possessive compounds", () => {
  assert.equal(inflect("0532 123 45 67", "DA"), "de"); // yedi → de
  assert.equal(inflect("10000000140", "E"), "a"); // sıfır → a
  assert.equal(inflect("Lale Sokağı", "E"), "na");
  assert.equal(inflect("Bucak", "DAKI"), "taki");
});

test("conversation keys for ChatGPT and Claude", () => {
  assert.equal(conversationKey("https://chatgpt.com/c/68f1a2b3-aaaa"), "chatgpt.com:68f1a2b3-aaaa");
  assert.equal(conversationKey("https://claude.ai/chat/0b9c8d7e-1111"), "claude.ai:0b9c8d7e-1111");
  assert.equal(conversationKey("https://chatgpt.com/"), "chatgpt.com:new");
  assert.equal(conversationKey("https://chatgpt.com/uc/6ad23d15-8bfc-83ea"), "chatgpt.com:6ad23d15-8bfc-83ea"); // oturumsuz
});

test("rejects overlapping or out-of-range offsets", () => {
  const vault = new ConversationVault();
  assert.throws(() => vault.mask("abcdef", [
    {type: "PERSON", start: 0, end: 4}, {type: "ORG", start: 3, end: 6}
  ]), /overlap/);
  assert.throws(() => vault.mask("abc", [{type: "PERSON", start: 0, end: 9}]), /exceeds/);
});

test("restores placeholders whose brackets the LLM dropped", () => {
  const vault = new ConversationVault();
  const text = "Kardeşler Market ve user_1";
  vault.mask(text, [span(text, "Kardeşler Market", "ORG")]);
  assert.equal(vault.restore("yarın KURUM_1'in önünde, kurum_1 de olur; user_1 ve XKURUM_1 değişmez"),
    "yarın Kardeşler Market'in önünde, Kardeşler Market de olur; user_1 ve XKURUM_1 değişmez");
});

test("already masked text is not masked again", () => {
  const vault = new ConversationVault();
  const first = "ayşe kardeşler market'te";
  const {masked} = vault.mask(first, [span(first, "ayşe", "PERSON"), span(first, "kardeşler market", "ORG")]);
  assert.equal(masked, "[KİŞİ_1] [KURUM_1]'te");
  // Model yer tutucunun içindeki "KURUM_1"i yeniden kurum sanarsa yok sayılır
  const again = vault.mask(masked, [span(masked, "KURUM_1", "ORG"), span(masked, "[KİŞİ_1]", "PERSON")]);
  assert.equal(again.masked, masked);
  assert.equal(vault.size, 2);
});
