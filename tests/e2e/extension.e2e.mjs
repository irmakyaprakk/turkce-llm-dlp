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
for (const script of manifest.content_scripts) script.matches.push(`${MOCK}/*`);
manifest.host_permissions.push(`${MOCK}/*`);
writeFileSync(path.join(extDir, "manifest.json"), JSON.stringify(manifest));
const background = readFileSync(path.join(extDir, "background.js"), "utf8")
  .replace('"www.claude.ai"]', '"www.claude.ai", "127.0.0.1"]');
writeFileSync(path.join(extDir, "background.js"), background.replace(
  '"https://claude.ai/*"];', `"https://claude.ai/*", "${MOCK}/*"];`));
// Sahte sayfa ChatGPT'nin gönderim adreslerini (/backend-api/conversation, /unauth-mweb/conversation/updates) kullanır
const extract = readFileSync(path.join(extDir, "extract.js"), "utf8");
writeFileSync(path.join(extDir, "extract.js"), extract.replace(
  'if (host === "chatgpt.com" &&', 'if ((host === "chatgpt.com" || host === "127.0.0.1") &&'));

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

const received = async () => (await fetch(`${MOCK}/__received`)).json();
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

await step("uzun yapıştırma eke dönüşmez, mesaj alanında kontrol edilir", async () => {
  await page.goto(`${MOCK}/c/yapistir0001abc?editor=ce`);
  await page.waitForTimeout(1500);
  const before = (await received()).length;
  const long = `Toplantı notları: ${"gündem maddeleri görüşüldü. ".repeat(10)}iletişim ${EMAIL}`;
  await page.click("#ce-box");
  await page.evaluate((text) => {
    const data = new DataTransfer();
    data.setData("text/plain", text);
    document.getElementById("ce-box").dispatchEvent(new ClipboardEvent("paste", {clipboardData: data, bubbles: true, cancelable: true}));
  }, long);
  assert.equal(await page.evaluate(() => window.__attachments.length), 0, "yapıştırma eke dönüştü");
  assert.ok((await page.locator("#ce-box").innerText()).includes(EMAIL), "metin mesaj alanına yazılmadı");
  await page.click("#ce-send");
  await page.locator("#turkce-dlp-preview").getByRole("button", {name: "Maskele ve gönder"}).click();
  await page.waitForFunction(() => window.__sent.length === 1);
  const sent = (await received()).slice(before);
  assert.equal(sent.length, 1);
  assert.ok(sent[0].includes("[EPOSTA_1]") && !sent[0].includes(EMAIL), sent[0]);
});

await step("ağ kilidi: onaysız ek metni (attachments) engellenir", async () => {
  const before = (await received()).length;
  const result = await page.evaluate(async (email) => {
    try {
      await fetch("/api/completion", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({prompt: "", attachments: [{extracted_content: `ek: ${email}`}]})});
      return "geçti";
    } catch (_error) {
      return "engellendi";
    }
  }, EMAIL);
  assert.equal(result, "engellendi");
  assert.equal((await received()).length, before);
});

await step("popup yerel servisin durumunu gösterir", async () => {
  const extensionId = new URL(context.serviceWorkers()[0].url()).host;
  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${extensionId}/popup.html`);
  await popup.locator("#service.ok").waitFor();
  assert.match(await popup.locator("#service").textContent(), /çalışıyor/);
  await popup.close();
});

await step("denetim günlüğüne yalnız sayılar gider", async () => {
  await page.waitForTimeout(500); // kayıtlar arka planda, gönderimi beklemeden yazılır
  const summary = await (await fetch("http://127.0.0.1:8000/v1/audit/summary?days=1")).json();
  assert.equal(summary.outcomes.blocked, 1, JSON.stringify(summary.outcomes));
  assert.ok(summary.outcomes.masked >= 2, JSON.stringify(summary.outcomes));
  assert.equal(summary.unchecked.EMAIL, 1, "işareti kaldırılan e-posta yanlış alarm olarak sayılmalı");
  assert.deepEqual(Object.keys(summary.sites), ["127.0.0.1"]);
  const raw = JSON.stringify(summary);
  for (const value of [TCKN, EMAIL, "GizliParola123", "Gül Sk."]) assert.ok(!raw.includes(value), `denetimde değer: ${value}`);
});

await step("tespit ağı: Web Worker'dan giden onaysız gönderim engellenemez ama yakalanır", async () => {
  await page.goto(`${MOCK}/c/worker0001abc`);
  await page.waitForTimeout(1500);
  const before = (await received()).length;
  // Sunucuya gerçekten ulaşacağı için kişisel veri yerine işaret metni kullanılır
  await page.evaluate((marker) => {
    // blob: adresli worker'da göreli adres çözülmez: tam adres verilir
    const code = `fetch("${location.origin}/backend-api/conversation", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: [{author: {role: "user"}, content: {parts: ["${marker}"]}}]})})`;
    new Worker(URL.createObjectURL(new Blob([code], {type: "text/javascript"})));
  }, "WORKER-ONAYSIZ-METIN");
  await page.locator("#turkce-dlp-error").getByText(/tespit edildi ve engellenemedi/).waitFor({timeout: 10_000});
  assert.equal((await received()).length, before + 1, "worker isteği (MV3'te engellenemez) sunucuya ulaşmalıydı");
  const worker = context.serviceWorkers()[0];
  const tabId = await worker.evaluate(async () => (await chrome.tabs.query({url: "http://127.0.0.1:8765/*"}))[0].id);
  assert.equal(await worker.evaluate((id) => chrome.action.getBadgeText({tabId: id}), tabId), "!");
  await page.waitForTimeout(500);
  const summary = await (await fetch("http://127.0.0.1:8000/v1/audit/summary?days=1")).json();
  assert.equal(summary.outcomes.leak_detected, 1, JSON.stringify(summary.outcomes));
  assert.ok(!JSON.stringify(summary).includes("WORKER-ONAYSIZ"), "denetim günlüğünde metin");
});

await step("'yapay zekânın gördüğünü göster' açıkken yer tutucular geri çevrilmez", async () => {
  await page.goto(`${MOCK}/c/sohbet0001abc`); // [EPOSTA_1] bu sohbetin eşlemesinde
  await page.waitForTimeout(1500);
  const worker = context.serviceWorkers()[0];
  await Promise.all([
    page.waitForEvent("load"),
    worker.evaluate(() => chrome.storage.local.set({showMasked: true}))
  ]);
  await page.waitForTimeout(1500);
  await page.locator("#turkce-dlp-masked-badge").waitFor();
  await page.evaluate(() => {
    const div = document.createElement("div");
    div.dataset.messageAuthorRole = "assistant";
    div.id = "masked-view";
    div.textContent = "Görünüm: [EPOSTA_1]";
    document.getElementById("thread").append(div);
  });
  await page.waitForTimeout(300);
  assert.equal(await page.locator("#masked-view").textContent(), "Görünüm: [EPOSTA_1]");
  await worker.evaluate(() => chrome.storage.local.set({showMasked: false}));
  await page.waitForTimeout(500);
  assert.equal(await page.locator("#masked-view").textContent(), `Görünüm: ${EMAIL}`);
  assert.equal(await page.locator("#turkce-dlp-masked-badge").count(), 0);
});

await step("ağ kilidi: eklentiyi atlayan gönderim sunucuya ulaşmaz", async () => {
  await page.goto(`${MOCK}/c/kilit0001abc`);
  await page.waitForTimeout(1500);
  const before = (await received()).length;
  await page.fill("#prompt-textarea", `Hızlıca: TC ${TCKN}`);
  await page.click("#leak");
  await page.waitForFunction(() => window.__leakBlocked === true);
  await page.locator("#turkce-dlp-error").getByText(/Kontrol edilmemiş gönderim engellendi/).waitFor();
  assert.equal((await received()).length, before, "sızdıran istek sunucuya ulaştı");
  assert.equal(await page.evaluate(() => window.__sent.length), 0);
});

await step("ağ kilidi: onaylanan (maskeli) metin geçer, ham metin sunucuda yok", async () => {
  await page.fill("#prompt-textarea", `Kaydım: TC ${TCKN}`);
  await page.press("#prompt-textarea", "Enter");
  await page.locator("#turkce-dlp-preview").getByRole("button", {name: "Maskele ve gönder"}).click();
  await page.waitForFunction(() => window.__sent.length === 1);
  const all = await received();
  assert.ok(all.at(-1).includes("[TCKN_1]"), all.at(-1));
  for (const body of all) assert.ok(!body.includes(TCKN), `sunucuda ham TC: ${body}`);
  assert.equal(await page.evaluate(() => window.__networkBlocked || 0), 0, "onaylı gönderim engellenmemeli");
});

await step("ağ kilidi: koruma kapalıyken kilit de açılır", async () => {
  const worker = context.serviceWorkers()[0];
  await worker.evaluate(() => chrome.storage.local.set({enabled: false}));
  await page.waitForTimeout(300);
  const before = (await received()).length;
  await page.fill("#prompt-textarea", "korumasız deneme 0532 000 00 00");
  await page.click("#leak");
  await page.waitForFunction(() => window.__sent.length === 2);
  assert.equal((await received()).length, before + 1);
  await worker.evaluate(() => chrome.storage.local.set({enabled: true}));
  await page.waitForTimeout(300);
});

await step("ağ kilidi: gönderim adresine tanınmayan gövde biçimi durdurulur (fail-closed)", async () => {
  await page.goto(`${MOCK}/c/bicim0001abc`);
  await page.waitForTimeout(1500);
  const before = (await received()).length;
  const result = await page.evaluate(async () => {
    try {
      await fetch("/backend-api/conversation", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({x: "ham metin 10000000146"})});
      return "geçti";
    } catch (_error) {
      return "engellendi";
    }
  });
  assert.equal(result, "engellendi");
  await page.locator("#turkce-dlp-error").getByText(/Gönderim biçimi tanınmadı/).waitFor();
  assert.equal((await received()).length, before, "tanınmayan gövde sunucuya ulaştı");
  // metin taşımayan uç nokta (ör. conversation/prepare) etkilenmez
  const prepare = await page.evaluate(async () => (await fetch("/unauth-mweb/conversation/prepare",
    {method: "POST", body: new URLSearchParams({conversationState: "{}"})})).ok);
  assert.ok(prepare);
});

await step("ağ kilidi: about:blank iç çerçevenin fetch'i de kilitli", async () => {
  const before = (await received()).length;
  // Gerçek sitede bulunan açık: çerçeveye kilidi manifest kurar (match_about_blank) ve göreli adres
  // about:blank'e göre çözülürdü; gövde de üst pencerede oluşturulmuş bir URLSearchParams olabilir.
  const crossRealm = await page.evaluate(async (tc) => {
    const frame = document.createElement("iframe");
    document.body.append(frame);
    await new Promise((resolve) => setTimeout(resolve, 300)); // manifest girdisi çerçeveye yüklensin
    try {
      await frame.contentWindow.fetch("/unauth-mweb/conversation/updates", {method: "POST", body: new URLSearchParams({prompt: `çerçeve ${tc}`})});
      return "geçti";
    } catch (_error) {
      return "engellendi";
    } finally {
      frame.remove();
    }
  }, TCKN);
  assert.equal(crossRealm, "engellendi");
  const result = await page.evaluate(async (tc) => {
    const frame = document.createElement("iframe");
    document.body.append(frame);
    try {
      await frame.contentWindow.fetch("/unauth-mweb/conversation/updates", {method: "POST",
        headers: {"Content-Type": "application/x-www-form-urlencoded"}, body: new URLSearchParams({prompt: `çerçeve ${tc}`}).toString()});
      return "geçti";
    } catch (_error) {
      return "engellendi";
    } finally {
      frame.remove();
    }
  }, TCKN);
  assert.equal(result, "engellendi");
  assert.equal((await received()).length, before, "iç çerçeve isteği sunucuya ulaştı");
});

await step("model çalışmıyorsa önizleme her zaman açılır ve kullanıcı uyarılır", async () => {
  await page.goto(`${MOCK}/c/model0down01`);
  await page.waitForTimeout(1500);
  const worker = context.serviceWorkers()[0];
  // Servis yanıtına "model yeniden başlatılıyor" durumu eklenir (gerçek arıza: inference zaman aşımı)
  await worker.evaluate(() => {
    globalThis.__realFetch = globalThis.fetch;
    globalThis.fetch = async (url, init) => {
      const response = await globalThis.__realFetch(url, init);
      if (!/\/v1\/detect|\/health/.test(String(url))) return response;
      const body = {...(await response.json()), ner: "unavailable"};
      return new Response(JSON.stringify(body), {status: response.status, headers: {"Content-Type": "application/json"}});
    };
  });
  const sentBefore = await page.evaluate(() => window.__sent.length);
  await page.fill("#prompt-textarea", "yarın toplantı saat kaçta");
  await page.press("#prompt-textarea", "Enter");
  const panel = page.locator("#turkce-dlp-preview");
  await panel.getByText("Ad ve adres tanıma şu an çalışmıyor").waitFor();
  await panel.getByText(/modeli şu an çalışmıyor/).waitFor();
  assert.equal(await page.evaluate(() => window.__sent.length), sentBefore, "uyarı onaylanmadan gönderildi");
  // Gerçek ChatGPT'de görülen: site, eklentinin tekrar gönderiminden hemen sonra ikinci bir submit tetikler.
  // Aynı metin yeniden kontrol edilirse önizleme döngüye girip mesaj hiç gitmiyordu.
  await page.evaluate(() => {
    const form = document.getElementById("composer-form");
    form.addEventListener("submit", () => setTimeout(() => form.requestSubmit(), 0), {once: true});
  });
  await panel.getByRole("button", {name: "Yine de gönder"}).click();
  await page.waitForFunction((n) => window.__sent.length >= n + 1, sentBefore);
  await page.waitForTimeout(800);
  assert.equal(await page.locator("#turkce-dlp-preview").count(), 0, "önizleme ikinci submit'te yeniden açıldı");
  const extensionId = new URL(worker.url()).host;
  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${extensionId}/popup.html`);
  await popup.locator("#service.warn").waitFor();
  assert.match(await popup.locator("#service").textContent(), /yeniden başlatılıyor/);
  await popup.close();
  await worker.evaluate(() => { globalThis.fetch = globalThis.__realFetch; });
});

await step("servis kapalıysa gönderim durur ve neden açıkça söylenir", async () => {
  await page.goto(`${MOCK}/c/kapali0001abc`);
  await page.waitForTimeout(1500);
  const api = await context.serviceWorkers()[0];
  await api.evaluate(() => { globalThis.__origFetch = globalThis.fetch; globalThis.fetch = () => Promise.reject(new TypeError("Failed to fetch")); });
  await page.fill("#prompt-textarea", "tel 0532 123 45 67");
  await page.click("[data-testid='send-button']");
  await page.locator("#turkce-dlp-error").waitFor();
  assert.match(await page.locator("#turkce-dlp-error").textContent(), /ulaşılamadı/);
  assert.equal(await page.evaluate(() => window.__sent.length), 0);
  await api.evaluate(() => { globalThis.fetch = globalThis.__origFetch; });
});

if (process.env.TURKCE_DLP_NER_MODEL_DIR) {
  await step("model ile: kişi adı eki dışarıda maskelenir ve yanıtta doğru çekimlenir", async () => {
    await page.goto(`${MOCK}/c/model0001abc`);
    await page.waitForTimeout(1500);
    await page.fill("#prompt-textarea", "Toplantı notunu Ayşe Demir'e ilet, Murat Kaya da gelsin.");
    await page.click("[data-testid='send-button']");
    await page.locator("#turkce-dlp-preview").getByRole("button", {name: "Maskele ve gönder"}).click();
    await page.waitForFunction(() => window.__sent.length === 1);
    const sent = (await page.evaluate(() => window.__sent))[0];
    assert.ok(sent.includes("[KİŞİ_1]'e ilet") && sent.includes("[KİŞİ_2]") && !sent.includes("Ayşe"), sent);
    const reply = page.locator("[data-message-author-role='assistant']").first();
    await reply.waitFor();
    await page.waitForTimeout(300);
    const text = await reply.textContent();
    assert.ok(text.includes("Ayşe Demir'e") && text.includes("Murat Kaya'ya"), text);
  });
}

console.log(results.join("\n"));
await context.close();
