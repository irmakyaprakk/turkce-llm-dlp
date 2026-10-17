# Türkçe Kurumsal LLM DLP
[![CI](https://github.com/irmakyaprakk/turkce-llm-dlp/actions/workflows/ci.yml/badge.svg)](https://github.com/irmakyaprakk/turkce-llm-dlp/actions/workflows/ci.yml)

Türkçe kurumsal LLM DLP prototipi: ChatGPT ve Claude'a giden mesajlardaki kişisel verileri yerelde bulur, yer tutucuyla değiştirir ve yanıtta geri yerleştirir. Ham değerler bilgisayardan çıkmaz.

| Bileşen | Durum |
|---|---|
| Kural tabanlı tespit (TCKN, IBAN, kart, e-posta, telefon, gizli anahtar) | Hazır; checksum/Luhn/mod-97 ve bağlam denetimi |
| Türkçe NER (kişi, kurum, yer, adres) | Hazır: BERTurk ince ayarı (augmented-v4) + kural katmanı + yer adı sözlüğü (hibrit). Ağırlıklar depoda yok |
| FastAPI servisi, politika (`mask`/`block`/`allow`), maskeleme/geri eşleme, OpenAI uyumlu gateway | Hazır; gerçek sağlayıcı varsayılan olarak kapalı |
| Chrome eklentisi (`extension/`) | v1.1: önizleme, sohbet bazlı eşleme, Türkçe ek uyumu, fail-closed ağ kilidi. Gerçek ChatGPT'de doğrulandı |
| Anonim denetim günlüğü ve yönetici paneli (`/panel`) | Hazır; yalnız tür başına sayılar |
| macOS arka plan servisi | Hazır (`scripts/macos/`) |
| Belge/OCR (PDF, Word, görsel) | Henüz yok |

Deney sonuçları ve değerlendirme protokolü: `reports/ner-experiments.md`, `docs/evaluation-protocol.md`.

## Çalıştırma

Python 3.10+ gerekir.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m turkce_llm_dlp
```

Servis `http://127.0.0.1:8000` üzerinde açılır. OpenAPI arayüzü: `http://127.0.0.1:8000/docs`.

Docker Compose ile kısıtlı yerel demo:

```bash
docker compose up --build
```

Servis yalnızca makinenin loopback arayüzünden erişilebilir; konteyner root olmayan kullanıcıyla, salt-okunur dosya sistemiyle ve demo sağlayıcıyla çalışır. Bu profil üretim dağıtımı değildir.

## API

- `GET /health`: canlılık kontrolü.
- `POST /v1/detect`: hassas veri türü ve metin içi offset döndürür; ham eşleşen değerleri döndürmez.
- `POST /v1/process`: istemi maskeler, demo yanıtı üretir ve yanıt placeholder'larını aynı istek kapsamındaki değerlerle geri eşler.
- `POST /v1/gateway`: politika kararını uygular, yalnızca maskelenmiş istemi sağlayıcı arayüzüne verir ve yanıtı geri eşler. Şu anki sağlayıcı yerel demodur; gerçek bir LLM'e bağlanmaz.
- `POST /v1/gateway/stream`: aynı politika ve maskeleme adımlarını uygulayıp SSE üzerinden yanıt parçalarını döndürür; parça sınırından bölünen placeholder'lar yanıt akarken geri eşlenir.
- `POST /v1/audit/event`: eklentinin gönderim kararını yalnızca tür başına sayılarla bildirir (metin/değer alanı kabul edilmez).
- `GET /v1/audit/summary?days=30`: panel özeti; `TURKCE_DLP_ADMIN_TOKEN` tanımlıysa `X-Admin-Token` başlığı ister.
- `GET /panel`: yönetici paneli (dış kaynak yüklemez, sıkı CSP).
- İstek gövdesi 100 KB, metin alanı 20.000 karakter ile sınırlıdır. Fazla büyük gövdeler JSON ayrıştırılmadan `413` ile reddedilir.

Örnek:

```bash
curl -s http://127.0.0.1:8000/v1/process \
  -H 'content-type: application/json' \
  -d '{"text":"İletişim: kisi@example.com"}'
```

## Çekirdek davranış

- NFKC ve zero-width temizliği yalnızca tespit kopyasında uygulanır; maskeleme kaynak metindeki özgün offset'lerle yapılır.
- TCKN checksum, Türk IBAN mod-97 ve 13-19 haneli kart adaylarında Luhn doğrulaması kullanılır.
- Türkiye telefon numaraları (cep 5xx, sabit 2xx-4xx; +90 / 0090 / 0 öneki; boşluk, nokta, tire, parantez; kısmen gizlenmiş "05XX" biçimi) yakalanır. Öneksiz 10 haneli sayı yalnızca telefon biçiminde yazılmışsa ya da "tel/cep/whatsapp" bağlamındaysa telefon sayılır; "sipariş/fatura/takip no" bağlamı yanlış alarmı önler. Farklı yazılan aynı numara aynı placeholder'ı alır.
- E-posta, bilinen GitHub/AWS/Slack/Stripe/Google/GitLab/npm/PyPI biçimleri, JWT ve PEM private key desenleri yakalanır. Secret benzeri atamalarda bağlam + Shannon entropisi, parola atamalarında anahtar adı kullanılır.
- `DetectionPipeline` kural tabanlı tespiti optional NER tanıyıcılarıyla birleştirebilir. NER inference exception veya zaman aşımı verirse hassas metni loglamadan uyarı yazar ve deterministik kurallarla devam eder. Zaman aşımında ayrı model süreci sonlandırılır ve servis yeniden başlatılana kadar NER devre dışı kalır. Hatalı model offset'i veri bütünlüğü hatası sayılır; gateway upstream'e geçmez.
- Aynı türdeki aynı özgün değer, tek istek içinde tutarlı placeholder alır.
- Upstream yanıtını parça parça işlerken `MaskingSession.restore_chunks(...)` parçalar arasında bölünen placeholder'ları da geri eşler; tamponda yalnızca olası placeholder eki tutulur.
- `config/policy.example.yml` örnek ayarını `TURKCE_DLP_POLICY` ortam değişkeniyle seçebilirsin. `mask` varsayılandır; `block` upstream çağrısını engeller; `allow` ilgili eşleşmeyi olduğu gibi upstream'e yollar. Bilinmeyen veri türleri daima maskelenir. Hatalı politika ayarıyla servis başlamaz.
- Akış başladıktan sonra upstream hata verirse HTTP durumu artık değiştirilemez; SSE `error` olayı gönderilir ve içerik/ham upstream hatası ifşa edilmez.
- Backend isteği işleyemez veya geçersiz span alırsa sağlayıcıya özgün istem gönderilmez. NER runtime exception'ı kural tabanlı katmana düşer; bu durumda NER'in yakalayacağı kişi/kurum/adresler kaçabilir.

Gerçek sağlayıcıyı açıkça seçmek için ortam değişkenlerini ayarla:

```bash
export TURKCE_DLP_UPSTREAM=openai_compatible
export TURKCE_DLP_UPSTREAM_API_KEY='...'
export TURKCE_DLP_UPSTREAM_MODEL='...'
# İsteğe bağlı: varsayılan OpenAI uyumlu chat completions adresidir.
export TURKCE_DLP_UPSTREAM_URL='https://api.openai.com/v1/chat/completions'
```

Eğitilmiş NER modelini API'ye yerel olarak bağlamak için `TURKCE_DLP_NER_MODEL_DIR` değerini model/tokenizer klasörüne ayarla. Model Hub'dan indirilmez; klasör eksik veya yükleme başarısızsa ham dosya yolu loglanmadan kural tabanlı modda açılır. İsteğe bağlı `TURKCE_DLP_NER_DEVICE=-1` CPU'yu, CUDA ortamında uygun cihaz numarası GPU'yu seçer. `TURKCE_DLP_NER_TIMEOUT_SECONDS` varsayılan 2 saniyelik inference limitini, `TURKCE_DLP_NER_STARTUP_TIMEOUT_SECONDS` varsayılan 60 saniyelik yükleme limitini belirler. Model entegrasyonu varsayılan olarak kapalıdır.

Gerçek sağlayıcı seçildiğinde, politikanın `mask` ettiği istem yine de üçüncü tarafa gönderilir; `allow` olarak ayarlanan alanlar da özgün haliyle gönderilir. Gerçek/kurumsal kullanım öncesi sağlayıcı, veri işleme koşulları ve politika kurum tarafından onaylanmalıdır. API anahtarı yalnızca ortam değişkeninden okunur. Sağlayıcı bağlantısı HTTPS zorunlu kılar (yerel test için localhost HTTP serbest), timeout ve yanıt boyutu sınırı uygular.
- Eşleme bellekte `MaskingSession` ömrü boyunca tutulur. İstek gövdeleri saklanmaz; Uvicorn erişim loglaması kapalıdır.
- `pytest` ile çekirdeğin birim testleri çalıştırılabilir.

## Kaçınma ölçümü

TCKN için ayraç ekleme, zero-width karakter ve fullwidth rakam dönüşümlerinde normalizasyon açık/kapalı yakalama ve yanlış pozitif oranlarını aynı tohumla karşılaştır:

```bash
python training/ner/evaluate_bypass.py --samples-per-class 50 --seed 42
```

Her dönüşüm için en az 50 geçerli ve 50 checksum-geçersiz sentetik örnek kullanılır. Ölçüm yalnızca bu üç biçim dönüşümü için geçerlidir; diğer atlatma türlerini kapsamaz.

Çekirdek tespit ve maskeleme gecikmesini sentetik 500, 5.000 ve 20.000 karakterlik girdilerde ölç:

```bash
python scripts/benchmark_latency.py --iterations 100 --warmup 20
```

Rapor p50/p95 değerlerini ve Python/işletim sistemi bilgisini içerir. Bu, yalnızca yerel çekirdek ölçümüdür; HTTP, sağlayıcı ağı ve NER modeli gecikmesi dahil değildir. Sonuçlar donanıma/çalışma ortamına özeldir. Modelli tespitte yerel CPU'da tek mesaj ısınmadan sonra yaklaşık 50 ms sürer (ayrıntılı modelli gecikme raporu henüz yok).

## Yazım biçimi dayanıklılığı

Üretim boru hattını (model + kurallar + sözlük) aynı cümlelerin orijinal, küçük ve BÜYÜK harfli kopyalarında ölç:

```bash
python training/ner/build_gold_set.py --out-dir data/ner/processed/realistic-v3
python training/ner/evaluate_casing.py --model-dir <model klasörü> --data-dir data/ner/processed/realistic-v3
```

v4 modeli geliştirme setinde hibrit F1: orijinal 0,973, küçük 0,935, BÜYÜK 0,790 (batch1-2); günlük dil partisinde (batch3) 0,851 / 0,838 / 0,681. BÜYÜK harf en zayıf biçimdir; augmented-v5 bu açığı hedefler (bkz. `reports/ner-experiments.md`).

## Yönetici paneli ve anonim denetim günlüğü

Servis çalışırken `http://127.0.0.1:8000/panel` adresi son 7/30/90 günün özetini gösterir: kontrol edilen ve engellenen mesajlar, günlük maskelenen veri, türe göre dağılım, kullanıcının "kişisel veri değil" diyerek işaretini kaldırdığı bulgular (yanlış alarm geri bildirimi), kurum politikası ve son kararlar.

Günlük kişisel veri tutmaz. Her kayıt yalnızca dakikaya yuvarlanmış zaman, site adı, karar (maskelendi / temiz / engellendi / vazgeçildi) ve tür başına sayıları içerir. Metin, değer, değerin özeti (hash) ve kullanıcı kimliği yazılmaz: TC kimlik no gibi küçük uzaylı değerlerin özeti bile kaba kuvvetle çözülebileceği için özet de tutulmaz. Olay şeması fazladan alanları reddeder.

| Ortam değişkeni | Varsayılan | Açıklama |
|---|---|---|
| `TURKCE_DLP_AUDIT` | `1` | `0` ile günlük tamamen kapanır |
| `TURKCE_DLP_AUDIT_DB` | `~/.turkce-dlp/audit.sqlite3` | SQLite dosyası (Docker'da yazılabilir bir volume'a bağlayın) |
| `TURKCE_DLP_AUDIT_RETENTION_DAYS` | `90` | Süresi dolan kayıtlar her yazımda silinir |
| `TURKCE_DLP_ADMIN_TOKEN` | — | Tanımlıysa panel özeti yönetici anahtarı ister |

Eklenti popup'ındaki **"Yapay zekânın gördüğünü göster"** anahtarı yanıtları geri çevirmeyi kapatır; sohbet, dış servise giden yer tutucularla (`[KİŞİ_1]`) görünür. Maskelemenin gerçekten yapıldığını kullanıcıya göstermek ve sunumda kanıtlamak içindir.

## Sınırlar ve sonraki aşamalar

- Secret kuralları eksiksiz değildir; entropy ve bağlam eşikleri yanlış pozitif/negatif üretebilir.
- NER modeli BÜYÜK harfli ve günlük yazımda zayıflar (yukarıdaki ölçüm); model kaçırdığında kural katmanı yalnız açık ipuçlarında (ör. "benim adım …") boşluk doldurur.
- Harf araya sokma, sayıyı yazıyla yazma, kodlanmış veri ve çok mesajlı parçalama kapsam dışıdır.
- Eklenti: yüklenen dosya ve görsellerin içeriği, sesli giriş ve eklenti kapalıyken gönderim kapsanmaz. Ağ kilidi sayfanın kendi isteklerini ve iç çerçeveleri kapsar, bilinen gönderim adresinde tanınmayan gövde biçimini durdurur; Web Worker / SharedWorker istekleri kilitlenmez, yalnız arka plandaki `webRequest` tespit ağı görür ve uyarır (MV3'te engelleme değil tespit). Yapıştırılan metin düz metin olarak alınır, biçim (tablo, kod bloğu) kaybolur. Ağ kilidi eklenti yüklendikten/yenilendikten sonra açılan sekmelerde çalışır. ChatGPT ve Claude resmî eklenti API'si sunmadığından site güncellemelerinden sonra elle denenmelidir; Claude ve oturum açılmış ChatGPT'deki ağ biçimi yalnız sahte sayfada test edildi.
- API loopback'e bağlıdır; dağıtım öncesi auth/TLS ve güvenlik sertleştirmesi gerekir.

Sonraki aşamalar: belge/OCR işleme (Aşama 5); modelli gecikme raporu, Presidio karşılaştırması ve fayda kaybı deneyi (Aşama 6); Docker imajı, model kartı, demo ve nihai rapor (Aşama 7).

## Geliştirme

```bash
python -m pip install -e '.[dev]'
pre-commit install          # her commit öncesi ruff, format ve gizli anahtar kontrolü
ruff check . && ruff format --check .
python -m pytest -q
node --test extension/core.test.cjs
scripts/run_e2e.sh          # eklenti uçtan uca: gerçek Chromium + yerel servis + sahte sohbet sayfası
```

Uçtan uca test 8000 portunda kendi servisini başlatır; macOS arka plan servisi açıksa önce durdurun
(`launchctl bootout gui/$(id -u)/com.turkcedlp.service`, sonra `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.turkcedlp.service.plist`).
Playwright: `npm install --no-save playwright@1.56.0 && npx playwright install chromium` (ya da `PLAYWRIGHT_MODULE` ile mevcut bir kurulum).

## Lisans

Kaynak kod MIT lisanslıdır (`LICENSE`). Eğitim verileri ve başlangıç modeli depoya dahil değildir; kaynakları, lisansları ve atıfları `DATA_LICENSES.md` içindedir.

## Arka planda çalıştırma (macOS, terminalsiz)

Yerel servisi her seferinde terminalden başlatmak yerine bir kez arka plan servisi olarak kurun:

1. Finder'da `scripts/macos/servisi_kur.command` dosyasına çift tıklayın (ilk seferde macOS "tanımlanamayan geliştirici" uyarısı verirse sağ tık → **Aç**).
2. Servis kurulur, model yüklenir ve oturum her açıldığında kendiliğinden başlar; çökerse yeniden başlatılır. Kayıtlar: `~/Library/Logs/turkce-dlp.log`.
3. Kod güncellendiğinde `servisi_yeniden_baslat.command`, kaldırmak için `servisi_kaldir.command`.

Eklenti popup'ı servisin çalışıp çalışmadığını gösterir. macOS "Masaüstü klasörüne erişim" izni isterse **İzin Ver** deyin.
