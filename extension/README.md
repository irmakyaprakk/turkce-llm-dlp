# Chrome eklentisi (v1.1)

Manifest V3 eklentisi ChatGPT ve Claude'da gönderim öncesi mesajı yerel DLP servisine (`http://127.0.0.1:8000/v1/detect`)
sorar, bulunan kişisel verileri okunabilir yer tutucularla değiştirir ve yanıt ekrana geldiğinde özgün değerleri geri yerleştirir.
Ham değerler dış LLM'e gitmez; yalnızca bilgisayardaki yerel servise gider.

## Nasıl çalışır

1. **Tespit:** Mesaj yerel servise gönderilir (model + kurallar). Yanıtta yalnızca tür, konum ve kurum politikası (`mask` / `block` / `allow`) döner; değerin kendisi dönmez.
2. **Önizleme:** Bulunanlar bir pencerede listelenir; kişisel veri olmayanın işareti kaldırılabilir. `block` türü varsa (ör. parola) gönderim tamamen durur. Önizleme popup'tan kapatılabilir.
3. **Maskeleme:** `Ayşe Demir` → `[KİŞİ_1]`, `0532 123 45 67` → `[TELEFON_1]`. Etiketler: KİŞİ, KURUM, YER, ADRES, TCKN, IBAN, KART, EPOSTA, TELEFON, GİZLİ.
4. **Sohbet bazlı eşleme:** Aynı sohbette aynı değer her mesajda aynı yer tutucuyu alır (LLM bağlamı korur). Eşleme sohbet kimliğine (`/c/<id>`, `/chat/<id>`) bağlıdır; başka sohbette kullanılmaz. Yalnız tarayıcının oturum belleğinde (`chrome.storage.session`) tutulur, diske yazılmaz, tarayıcı kapanınca ve en geç 24 saatte silinir.
5. **Geri eşleme ve Türkçe ek uyumu:** Yanıttaki `[KİŞİ_1]'e` asıl değere göre yeniden çekimlenir: `Ayşe'ye`, `Murat'a`, `Moda Mahallesi'nde`, `0532 123 45 67'ye` (son rakamın okunuşu). LLM'in yazımı değiştirmesine (`[kişi_1]`, `[KISI_1]ye`) dayanıklıdır; LLM'in uydurduğu, kasada olmayan yer tutucular olduğu gibi kalır.

## Yerel yükleme

1. Servisi başlat: `python -m turkce_llm_dlp` (model için `TURKCE_DLP_NER_MODEL_DIR=<model klasörü>`).
2. Chrome'da `chrome://extensions` → **Developer mode** → **Load unpacked** → bu `extension/` klasörü.
3. ChatGPT ya da Claude'u aç; popup'tan korumayı ve önizlemeyi aç/kapat, sohbetin eşlemesini sil.

Servis kapalıysa, yanıt geçersizse ya da mesaj alanı doğrulanamazsa eklenti gönderimi durdurup uyarı verir (fail-closed).

**Ağ kilidi (`guard.js`, `extract.js`):** Sayfanın kendi dünyasında (`world: "MAIN"`), site betiklerinden önce (`document_start`) çalışır;
`fetch`, `XMLHttpRequest` ve `sendBeacon`'ı sarar. ChatGPT/Claude'a giden POST gövdesindeki kullanıcı metni (form ya da JSON
`prompt` alanı, son kullanıcı mesajının `content.parts` alanı, `attachments[].extracted_content`) önizlemeden geçip onaylanan
metinle eşleşmiyorsa istek gönderilmez ve "Kontrol edilmemiş gönderim engellendi" uyarısı çıkar. Bilinen gönderim adreslerinde
(ChatGPT `…/conversation`, `…/conversation/updates`; Claude `…/completion`) gövde biçimi tanınmazsa istek yine durur
("Gönderim biçimi tanınmadı"): site biçim değiştirirse kilit sessizce açılmaz. Kilit tüm çerçevelere kurulur (`all_frames`,
`match_about_blank`); aynı kökenli bir iç çerçevenin `fetch`'i de erişildiği anda kilitlenir.

**Tespit ağı (`background.js`, `webRequest`):** Sayfa kilidinin göremediği istekler (Web Worker / SharedWorker) için arka plan,
ChatGPT/Claude'a giden gönderim isteklerinin gövdesini gözler. Onaysız bir gönderim görülürse sayfada uyarı bandı ve eklenti
simgesinde kırmızı "!" rozeti çıkar, denetim günlüğüne `leak_detected` sonucu (yalnız sayı) yazılır. **Bu bir engelleme değil,
tespittir:** Manifest V3'te `webRequest` isteği durduramaz; istek o anda gitmiş olur. Onay ~30 saniye geçerlidir (site isteği yeniden
deneyebilir). Böylece olay yakalamayı atlayan bir gönderim yolu ham metni dışarı çıkaramaz. Koruma popup'tan kapatılınca
kilit de açılır. Not: kilit, eklenti yüklendikten ya da yenilendikten **sonra** açılan sekmelerde çalışır; önceden açık
sekmeler yenilenmelidir.

## Testler

- Birim: `node --test extension/core.test.cjs` (yer tutucu, ek uyumu, sohbet kasası).
- Uçtan uca: `scripts/run_e2e.sh` — gerçek Chromium'a eklentiyi yükler, yerel servisi ve sahte bir sohbet sayfasını
  (`tests/e2e/mock_chat.html`) başlatır; gönderilen metinde ham veri olmadığını, yanıtın doğru eklerle geri çevrildiğini,
  sohbet bazlı eşlemeyi, `block` politikasını, sayfa yenilemeyi ve zengin metin alanını (contenteditable) doğrular.
  Gereken: `npm install --no-save playwright@1.56.0 && npx playwright install chromium`.

## Sınırlar

ChatGPT ve Claude resmî bir eklenti API'si sunmaz; gönderim düğmesi ve yanıt DOM'u değişebilir. Seçiciler erişilebilir
etiketlere dayanır ama gerçek sitelerde sürüm değişikliklerinden sonra elle denenmelidir. Yapıştırılan metin düz metin
olarak mesaj alanına yazılır (Claude uzun yapıştırmayı ayrı bir eke çeviremez) ve normal kontrolden geçer; bilinçli bir
tercih olarak biçim (tablo, kod bloğu, kalın yazı) yapıştırmada kaybolur; istekte onaysız
ek metni (`attachments[].extracted_content`) varsa ağ kilidi isteği durdurur. Yüklenen dosyaların ve görsellerin içeriği,
sesli giriş ve eklenti kapalıyken gönderim kapsanmaz. Web Worker'dan giden gönderim yalnız tespit edilir, engellenmez;
`window[0]` gibi çerçeve dizini üzerinden erişilen ve kilit yüklenmeden kullanılan bir iç çerçeve teorik olarak açık kalır. Eklentiyi tek koruma katmanı sayma; kurumsal kullanımda gateway
(`/v1/gateway`) ile birlikte kullanılması önerilir.
