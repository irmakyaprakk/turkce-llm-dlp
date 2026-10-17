# Chrome eklentisi (v1.0)

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

**Ağ kilidi (`guard.js`):** Sayfanın kendi dünyasında (`world: "MAIN"`), site betiklerinden önce (`document_start`) çalışır;
`fetch`, `XMLHttpRequest` ve `sendBeacon`'ı sarar. ChatGPT/Claude'a giden POST gövdesindeki kullanıcı metni (form ya da JSON
`prompt` alanı, son kullanıcı mesajının `content.parts` alanı) önizlemeden geçip onaylanan metinle eşleşmiyorsa istek
gönderilmez ve "Kontrol edilmemiş gönderim engellendi" uyarısı çıkar. Onay ~30 saniye geçerlidir (site isteği yeniden
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
etiketlere dayanır ama gerçek sitelerde sürüm değişikliklerinden sonra elle denenmelidir. Dosya ekleri ve görseller,
sesli giriş ve eklenti kapalıyken gönderim kapsanmaz. Eklentiyi tek koruma katmanı sayma; kurumsal kullanımda gateway
(`/v1/gateway`) ile birlikte kullanılması önerilir.
