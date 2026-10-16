# Türkçe Kurumsal LLM DLP
[![CI](https://github.com/irmakyaprakk/turkce-llm-dlp/actions/workflows/ci.yml/badge.svg)](https://github.com/irmakyaprakk/turkce-llm-dlp/actions/workflows/ci.yml)

Türkçe kurumsal LLM DLP prototipi. Tespit, istek kapsamlı maskeleme/geri eşleme, politika, FastAPI ve OpenAI uyumlu sağlayıcı adaptörü vardır. Gerçek sağlayıcı varsayılan olarak kapalıdır. ChatGPT/Claude için Manifest V3 eklentisinin ilk sürümü `extension/` altındadır; gerçek web arayüzüyle manuel doğrulaması henüz yapılmadı. Türkçe NER modeli, kalıcı audit ve belge/OCR modülleri henüz hazır değildir.

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

Rapor p50/p95 değerlerini ve Python/işletim sistemi bilgisini içerir. Bu, yalnızca yerel çekirdek ölçümüdür; HTTP, sağlayıcı ağı ve NER modeli gecikmesi dahil değildir. Sonuçlar donanıma/çalışma ortamına özeldir.

## Sınırlar ve sonraki aşamalar

Secret kuralları eksiksiz değildir; entropy ve bağlam eşikleri yanlış pozitif/negatif üretebilir. NER arayüzüne henüz eğitilmiş Türkçe model bağlanmadı; timeout davranışı mock worker ile test edilmiştir, gerçek model/GPU üzerinde doğrulanmalıdır. Harf araya sokma, sayıyı yazıyla yazma, kodlanmış veri ve çok mesajlı parçalama kapsam dışıdır. Sonraki aşamalar: eklentiyi gerçek sitelerde doğrulama, NER veri incelemesi/eğitimi, audit ve belge işleme. API şu an loopback'e bağlıdır; dağıtım öncesi auth/TLS ve güvenlik sertleştirmesi gerekir.

## Geliştirme

```bash
python -m pip install -e '.[dev]'
pre-commit install          # her commit öncesi ruff, format ve gizli anahtar kontrolü
ruff check . && ruff format --check .
python -m pytest -q
node --test extension/core.test.cjs
```

## Lisans

Kaynak kod MIT lisanslıdır (`LICENSE`). Eğitim verileri ve başlangıç modeli depoya dahil değildir; kaynakları, lisansları ve atıfları `DATA_LICENSES.md` içindedir.
