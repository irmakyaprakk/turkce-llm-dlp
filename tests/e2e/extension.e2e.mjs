// Eklentinin uçtan uca testi (gerçek Chromium + gerçek yerel DLP API'si + sahte sohbet sayfası).
// Çalıştırma: scripts/run_e2e.sh
import assert from "node:assert/strict";
import {cpSync, mkdtempSync, readFileSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {createRequire} from "node:module";

const require = createRequire(import.meta.url);
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const here = path.dirname(fileURLToPath(import.meta.url));
const MOCK = "http://127.0.0.1:8765";

// Test kopyası: içerik betiği yalnız sahte sayfaya da izin verecek şekilde genişletilir (asıl manifest değişmez)
const workDir = mkdtempSync(path.join(tmpdir(), "dlp-ext-"));
const extDir = path.join(workDir, "extension");
cpSync(path.join(here, "../../extension"), extDir, {recursive: true});
const manifest = JSON.parse(readFileSync(path.join(extDir, "manifest.json"), "utf8"));
manifest.content_scripts[0].matches.push(`${MOCK}/*`);
manifest.host_permissions.push(`${MOCK}/*`);
writeFileSync(path.join(extDir, "manifest.json"), JSON.stringify(manifest));
const background = readFileSync(path.join(extDir, "background.js"), "utf8")
  .replace('"www.claude.ai"]', '"www.claude.ai", "127.0.0.1"]');
writeFileSync(path.join(extDir, "background.js"), background);

const context = await chromium.launchPersistentContext(path.join(workDir, "profile"), {
  headless: true,
  channel: "chromium",
  args: [`--disable-extensions-except=${extDir}`, `--load-extension=${extDir}`]
});
const results = [];
async function step(name, fn) {
  console.error(`… ${name}`);
  try {
    await fn();
    results.push(`ok - ${name}`);
  } catch (error) {
    results.push(`not ok - ${name}\n  ${error.message.split("\n").join("\n  ")}`);
    process.exitCode = 1;
  }
}

const page = await context.newPage();
page.setDefaultTimeout(10_000);
await page.goto(`${MOCK}/`);
await page.waitForTimeout(1500); // eklentinin yüklenmesi

const TCKN = "10000000146";
const PHONE = "0532 123 45 67";
const EMAIL = "deniz.kaya@example.com";
const ADDRESS = "Gül Sk. No:8 D:3 Kadıköy/İstanbul";

await step("önizleme açılır, maskeli metin gönderilir, ham veri sayfadan çıkmaz", async () => {
  await page.fill("#prompt-textarea", `Kaydı güncelle: TC ${TCKN}, tel ${PHONE}, e-posta ${EMAIL}, adres ${ADDRESS}.`);
  await page.click("[data-testid='send-button']");
  const panel = page.locator("#turkce-dlp-preview");
  await panel.getByRole("dialog").waitFor({timeout: 8000});
  const rows = await panel.locator("li").count();
  assert.ok(rows >= 4, `önizlemede ${rows} satır`);
  await panel.getByRole("button", {name: "Maskele ve gönder"}).click();
  await page.waitForFunction(() => window.__sent.length === 1);
  const sent = (await page.evaluate(() => window.__sent))[0];
  for (const raw of [TCKN, PHONE, EMAIL, "Gül Sk."]) assert.ok(!sent.includes(raw), `gönderilen metinde ham veri: ${raw}`);
  for (const ph of ["[TCKN_1]", "[TELEFON_1]", "[EPOSTA_1]", "[ADRES_1]"]) assert.ok(sent.includes(ph), `yer tutucu yok: ${ph}\n${sent}`);
});

await step("yanıtta yer tutucular özgün değerlere ve doğru eke çevrilir", async () => {
  const reply = page.locator("[data-message-author-role='assistant']").first();
  await reply.waitFor();
  await page.waitForTimeout(300);
  const text = await reply.textContent();
  assert.ok(text.includes(`${PHONE}'ye`), `telefon eki yanlış: ${text}`); // "yedi" → 'ye
  assert.ok(text.includes(`${TCKN}'ya`), `TC eki yanlış: ${text}`); // "altı" → 'ya
  assert.ok(!/\[(TCKN|TELEFON|EPOSTA|ADRES)_\d+\]/.test(text), `geri çevrilmemiş yer tutucu: ${text}`);
  const mine = await page.locator("[data-message-author-role='user']").first().textContent();
  assert.ok(mine.includes(TCKN), "kullanıcının kendi mesajı ekranda özgün hâliyle görünmeli");
});

await step("aynı sohbette aynı değer aynı yer tutucuyu alır", async () => {
  await page.waitForURL(/\/c\/sohbet0001abc/);
  await page.waitForTimeout(1200); // eşleme yeni sohbet kimliğine taşınır
  await page.fill("#prompt-textarea", `Bir de şu numarayı ara: ${PHONE}`);
  await page.click("[data-testid='send-button']");
  await page.locator("#turkce-dlp-preview").getByRole("button", {name: "Maskele ve gönder"}).click();
  await page.waitForFunction(() => window.__sent.length === 2);
  const sent = (await page.evaluate(() => window.__sent))[1];
  assert.ok(sent.includes("[TELEFON_1]") && !sent.includes("[TELEFON_2]"), sent);
});

await step("işareti kaldırılan bilgi maskelenmez", async () => {
  await page.fill("#prompt-textarea", `Sadece e-postam ${EMAIL}`);
  await page.click("[data-testid='send-button']");
  const panel = page.locator("#turkce-dlp-preview");
  await panel.getByRole("checkbox").first().uncheck();
  await panel.getByRole("button", {name: "Maskele ve gönder"}).click();
  await page.waitForFunction(() => window.__sent.length === 3);
  assert.ok((await page.evaluate(() => window.__sent))[2].includes(EMAIL));
});

await step("politika 'block' türünde gönderim tamamen durur", async () => {
  await page.fill("#prompt-textarea", "config: password=GizliParola123");
  await page.click("[data-testid='send-button']");
  const panel = page.locator("#turkce-dlp-preview");
  await panel.getByText("Bu mesaj gönderilemez").waitFor();
  await panel.getByRole("button", {name: "Tamam"}).click();
  await page.waitForTimeout(500);
  assert.equal(await page.evaluate(() => window.__sent.length), 3);
  assert.ok((await page.inputValue("#prompt-textarea")).includes("GizliParola123"), "mesaj alanı korunmalı");
});

await step("sonradan gelen yanıtlar da geri çevrilir", async () => {
  await page.evaluate(() => {
    const div = document.createElement("div");
    div.dataset.messageAuthorRole = "assistant";
    div.id = "late";
    div.textContent = "Eski yanıt: [EPOSTA_1]";
    document.getElementById("thread").append(div);
  });
  await page.waitForTimeout(300);
  assert.equal(await page.locator("#late").textContent(), `Eski yanıt: ${EMAIL}`);
});

await step("sayfa yenilenince eşleme oturum belleğinden geri gelir", async () => {
  await page.reload();
  await page.waitForTimeout(1500);
  await page.evaluate(() => {
    const div = document.createElement("div");
    div.dataset.messageAuthorRole = "assistant";
    div.id = "after-reload";
    div.textContent = "Yenileme sonrası: [TELEFON_1]'e";
    document.getElementById("thread").append(div);
  });
  await page.waitForTimeout(300);
  assert.equal(await page.locator("#after-reload").textContent(), `Yenileme sonrası: ${PHONE}'ye`);
});

await step("başka sohbette eşleme kullanılmaz", async () => {
  await page.goto(`${MOCK}/c/baskasohbet99`);
  await page.waitForTimeout(1500);
  await page.evaluate(() => {
    const div = document.createElement("div");
    div.id = "other";
    div.textContent = "[TELEFON_1]";
    document.getElementById("thread").append(div);
  });
  await page.waitForTimeout(300);
  assert.equal(await page.locator("#other").textContent(), "[TELEFON_1]");
});

await step("zengin metin alanı (contenteditable, Claude benzeri) da maskelenir", async () => {
  await page.goto(`${MOCK}/c/zengin0001abc?editor=ce`);
  await page.waitForTimeout(1500);
  await page.click("#ce-box");
  await page.keyboard.type(`telefonum ${PHONE}`);
  await page.click("#ce-send");
  await page.locator("#turkce-dlp-preview").getByRole("button", {name: "Maskele ve gönder"}).click();
  await page.waitForFunction(() => window.__sent.length === 1);
  const sent = (await page.evaluate(() => window.__sent))[0];
  assert.ok(sent.includes("[TELEFON_1]") && !sent.includes(PHONE.slice(0, 4) + " 123"), sent);
});

console.log(results.join("\n"));
await context.close();
