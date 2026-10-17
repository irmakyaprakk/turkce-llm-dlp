# Proje durumu

Son güncelleme: 17 Ekim 2026. Bu belge, bitirme projesi tanıtım dokümanı ve literatür taramasındaki teslimatları mevcut depoyla karşılaştırır. Ayrıntılı deney sonuçları: `reports/ner-experiments.md`.

## Aşamalar

| Aşama | Kapsam | Durum |
|---|---|---|
| 1 | Kural tabanlı tespit, normalizasyon, maskeleme/geri eşleme, politika, FastAPI, gateway | Bitti |
| 2 | NER veri hattı, BERTurk ince ayarı, değerlendirme protokolü, mühürlü testler | Bitti (v4 üretimde). Veri kaynakları: `reports/ner-experiments.md` |
| 3 | Chrome eklentisi | Bitti (v1.1, gerçek ChatGPT'de doğrulandı) |
| 4 | Anonim denetim günlüğü, yönetici paneli, macOS arka plan servisi | Bitti |
| — | Günlük yazım dayanıklılığı (augmented-v5) | Sürüyor: veri hazır, eğitim Colab'da; bağımsız değerlendirme için sealed3 toplanacak (`docs/sealed3-talimat.md`) |
| 5 | Belge/OCR: PDF, Word ve görsel maskeleme | Başlanmadı |
| 6 | Modelli gecikme raporu, Presidio karşılaştırması, fayda kaybı deneyi | Başlanmadı |
| 7 | Docker imajı, model kartı, demo, nihai rapor | Başlanmadı |

## Bileşenler

| Bileşen | Durum | Kanıt / sınır |
|---|---|---|
| TCKN, IBAN, kredi kartı, e-posta, telefon ve secret tespiti | Hazır | Checksum, mod-97, Luhn; telefon biçim + bağlam denetimi (öneksiz cep/sabit hat dahil); bilinen secret biçimleri ve bağlam/entropy denetimi. |
| Unicode/bypass normalizasyonu | Hazır, sentetik ölçüm | Ayraç, zero-width ve fullwidth için 50 pozitif + 50 negatif örnek (`reports/bypass-evaluation.json`). Yalnız TCKN ve bu üç dönüşüm. |
| İstek kapsamlı maskeleme ve geri eşleme | Hazır | Tekrarlı değer, çakışan placeholder ve parçalı SSE senaryoları test ediliyor. |
| Politika ve sağlayıcı sınırı | Hazır | YAML `mask/block/allow`, HTTPS OpenAI uyumlu adaptör, hata metni sanitizasyonu. Harici sağlayıcı varsayılan değil. |
| Türkçe NER (kişi, kurum, yer, adres) | Üretimde: v4 | BERTurk + augmented-v4 + kural katmanı + yer adı sözlüğü + tesis kuralı + adres birleşimi. Mühürlü test 2 (sealed2, 100 mesaj) resmî hibrit F1 0,865 ± 0,011; geliştirme seti 0,973. Süreç izolasyonu, zaman aşımı ve kural fallback'i gerçek modelle çalışıyor; CPU'da tek mesaj ~50 ms. |
| Yazım biçimi dayanıklılığı | Ölçüldü, iyileştirme sürüyor | `training/ner/evaluate_casing.py`. v4 hibrit F1 (geliştirme, batch1-2): orijinal 0,973 / küçük 0,935 / BÜYÜK 0,790; günlük dil partisi (batch3): 0,851 / 0,838 / 0,681. |
| Chrome eklentisi | v1.1 | Önizleme, sohbet bazlı eşleme, Türkçe ek uyumu, politika engeli, yapıştırmanın mesaj alanında kontrolü. Fail-closed ağ kilidi (`guard.js`): onaysız metin, bilinen gönderim adresinde tanınmayan gövde biçimi ve aynı kökenli iç çerçeve istekleri durur. Web Worker istekleri `webRequest` tespit ağıyla yalnız yakalanır (uyarı + `leak_detected`; MV3'te engelleme değil). Gerçek oturumsuz ChatGPT'de doğrulandı; Claude ve oturumlu ChatGPT ağ biçimi yalnız sahte sayfada test edildi. Dosya/görsel içeriği kapsanmıyor. |
| Denetim günlüğü ve yönetici paneli | Hazır | Yalnız dakikaya yuvarlanmış zaman, site, karar (maskelendi / temiz / engellendi / vazgeçildi / kontrolsüz gönderim) ve tür başına sayılar; metin/değer/hash tutulmaz. `/panel` sıkı CSP ile. |
| macOS arka plan servisi | Hazır | LaunchAgent; oturum açılınca başlar, çökerse yeniden başlar. |
| CI | Çalışıyor | Lint, Python 3.10–3.13 pytest, eklenti birim testi ve Playwright uçtan uca testi. 16–17 Ekim arası e2e işi, betiğin çalıştırma izni kaybolduğu için başarısızdı; düzeltildi. |
| Docker Compose | Yapılandırma hazır | Loopback portu, root olmayan kullanıcı, salt-okunur kök dosya sistemi. İmaj henüz derlenip denenmedi. |

## Son yerel kontroller (17 Ekim 2026)

- Python: 230 test geçti; ruff temiz.
- Eklenti: 11 birim testi ve 21 uçtan uca senaryo geçti (üç ardışık koşuda kararlı).
- Gerçek ChatGPT (oturumsuz; istekler test kancasıyla iptal edildi): önizleme gönderimden önce açılıyor, ağa yalnız maskeli metin çıkıyor; doğrudan fetch/XHR, tanınmayan gövde biçimi, about:blank iç çerçeve ve olay yakalaması kapalıyken sitenin kendi gönderimi ağ kilidinde duruyor; Web Worker gönderimi tespit edilip uyarılıyor.
- Bir Starlette/httpx kullanımdan kaldırma uyarısı var; test başarısızlığı değil.

## Açık işler

1. augmented-v5 sonuçları: v4 ile üç yazım biçiminde karşılaştırma ve realistic-v2'de gerileme kontrolü (batch3 iyimser, aynı kaynak); karar sealed3 ile bir kez.
2. sealed3: bağımsız yazarlardan 50–100 mesaj toplamak, elle etiketlemek, mühürlemek.
3. Eklentinin Claude ve oturumlu ChatGPT'de elle denenmesi (ağ kilidi, gönderim adresleri ve yapıştırma).
4. GitHub push/webhook koruması (tanıtım dokümanındaki teslimat; henüz yok).
5. Aşama 5–7 (yukarıdaki tablo).
6. Adres kuralı önceliğinin güven tabanlı hâle getirilmesi (sealed2 analizinde açık kalan iş).
