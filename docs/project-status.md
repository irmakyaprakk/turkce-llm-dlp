# Proje durumu

Son kod incelemesi: 12 Ekim 2026. Bu belge, bitirme projesi tanıtım dokümanı ve literatür taramasındaki teslimatları mevcut depoyla karşılaştırır.

## Kullanıcı verisi gerektirmeden hazırlananlar

| Bileşen | Durum | Kanıt / sınır |
|---|---|---|
| TCKN, IBAN, kredi kartı, e-posta, telefon ve secret tespiti | Kod hazır | Checksum, mod-97, Luhn, telefon biçim + bağlam denetimi, bilinen secret biçimleri ve bağlam/entropy denetimi. Sınıflandırıcı kuralları gerçek veride ayrıca ölçülmeli. |
| Unicode/bypass normalizasyonu | Kod ve sentetik ölçüm hazır | Ayraç, zero-width ve fullwidth için açık/kapalı koşullarda 50 pozitif + 50 negatif örnek. Yalnızca TCKN ve bu üç dönüşüm için geçerli. |
| İstek kapsamlı maskeleme ve geri eşleme | Çekirdek/test hazır | Tekrarlı değer, çakışan placeholder ve parçalı SSE senaryoları test ediliyor. |
| Politika ve sağlayıcı sınırı | Çekirdek/test hazır | YAML `mask/block/allow`, HTTPS OpenAI uyumlu adaptör ve hata metni sanitizasyonu var. Harici sağlayıcı varsayılan değil. |
| FastAPI | Kod ve TestClient doğrulaması hazır | Boyut limitleri, ham eşleşme değerini döndürmeme ve upstream bloklama testleri var. Canlı port dinleme sandbox tarafından engellendi. |
| NER veri/eğitim hattı | Kod hazır; veri yok | CoNLL dönüştürme, sentetik adres, kalite raporu, split/BIO denetimleri, BERTurk fine-tuning ve manifest var. Ağırlık üretilmedi. |
| NER çalışma zamanı | Kod/mock test hazır | Yerel yükleme, süreç izolasyonu, yükleme/inference timeout'u ve kural fallback'i eklendi. Gerçek model/GPU ile doğrulanmadı. |
| Chrome eklentisi | v1.0 hazır | Okunabilir Türkçe yer tutucular, sohbet bazlı eşleme (oturum belleği), Türkçe ek uyumuyla geri eşleme, gönderim öncesi önizleme, politika engeli. Birim + Playwright uçtan uca testleri (sahte sohbet sayfası). Gerçek ChatGPT/Claude DOM değişikliklerine karşı elle denenmeli; dosya/görsel kapsanmıyor. |
| CI | İş akışı hazır | Python 3.10–3.13 pytest ve Node yardımcı testleri. GitHub üzerinde çalıştırılmadı. |
| Docker Compose | Yapılandırma hazır | Loopback portu, root olmayan kullanıcı, salt-okunur root dosya sistemi ve demo upstream ayarlı. `compose config` geçti; Docker daemon erişim izni olmadığından imaj derlenmedi. |
| Ölçümler | Başlangıç raporları hazır | Bypass raporu `reports/bypass-evaluation.json`; yerel NER-kapalı p50/p95 raporu `reports/latency-evaluation.json`. Bunlar üretim veya model ölçümü değildir. |

## Son yerel kontroller

- Python: 30 test geçti.
- JavaScript yardımcıları: 3 Node testi geçti.
- Python kaynakları derlendi; CLI yardım komutları, paket wheel üretimi/importu ve JSON/YAML yapılandırmaları kontrol edildi.
- Bir Starlette/AnyIO deprecation uyarısı var; test başarısızlığı değil.
- Canlı Docker derlemesi ve canlı tarayıcı sayfalarında eklenti denemesi bu ortamda doğrulanmadı.

## Kullanıcıyla birlikte yapılacak veri kapısı

1. Turkish Wiki NER veri kaynağını lisans ve atıf dosyalarıyla edinmek.
2. Etiketleri, split dosyalarını ve lisans/atıf koşullarını incelemek.
3. Yerel hazırlama betiğini çalıştırıp aggregate kalite raporunu ve sentetik adres örneklerini gözden geçirmek.
4. Veri uygun bulunursa GPU ortamında baseline eğitmek. Bu adıma kadar ham veri veya model ağırlığı repoya alınmayacak.

## Sonraki proje teslimatları

Veri kapısından sonra NER sonuçlarını gerçek modelle ölçmek, bağımsız holdout ve Presidio kıyasını yapmak, eşik/hata analizini tamamlamak ve ancak sonra modeli varsayılan dışı şekilde açmak gerekir. Ayrıca anonim audit günlüğü/yönetici paneli, GitHub push/webhook koruması, PDF/Word/OCR görsel maskeleme, fayda deneyi, performans/dağıtım sertleştirmesi, demo ve nihai rapor henüz yapılmadı.
