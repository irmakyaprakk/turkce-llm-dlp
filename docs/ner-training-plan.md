# Türkçe NER aşaması: veri, eğitim ve entegrasyon

## Şu an hangi aşamadayız?

Çekirdek DLP, politika, upstream adaptörü ve akışlı geri eşleme mevcut. `DetectionPipeline` NER tanıyıcılarını alabiliyor. NER tarafında etiket şeması, CoNLL veri dönüştürücü, sentetik ADDRESS örnek üretimi, split/BIO doğrulaması, aggregate-only kalite raporu ve BERTurk fine-tuning betiği hazır. **İlk kişi/kurum/yer modeli Colab T4 GPU’da eğitildi** (WikiNER test span F1 0,872); gerçek OSM adresli ikinci model için veri doğrulandı ve Colab not defteri hazır.

Şu an veri edinmeden önce kod ve eğitim akışını hazırlıyoruz. Dataset indirildiğinde lisans/atıf korunacak, hazırlama betiği çalıştırılacak, kalite raporu ve sentetik örnekler incelenecek; ondan sonra GPU eğitimine geçilecek.

## Önerilen ortam

Bu çalışma alanındaki makine macOS 26.4.1, Apple Silicon/ARM64 ve Python 3.13. `torch`, `transformers`, `datasets` ve `seqeval` şu anda kurulu değil. Uygulama çekirdeğini burada geliştirmeye devam edeceğiz.

İlk fine-tuning için önerim **Google Colab GPU runtime**; tekrarlanabilir deneyler için eğitim notebook'u ve kodu bu repoda kalacak. Colab GPU ve VM ömrü garantili değil, limitler değişiyor; küçük pilotlar için uygun, uzun/garantili eğitim için değil. Kuruma ait veya gerçek kişisel veri içeren metinleri Colab'a koymayacağız. İlk eğitim yalnızca lisansı kontrol edilmiş açık veri ve sentetik/elle gözden geçirilmiş örneklerle yapılacak. Colab'ın GPU erişimi ve sürelerinin değişken olduğu kendi dokümantasyonunda da belirtiliyor. [Colab kaynak sınırları](https://research.google.com/colaboratory/faq.html)

Yerel alternatif de mümkün: Apple Silicon'da PyTorch'un MPS arka ucu GPU hızlandırması sağlıyor. Ancak şu an bu ortamda ML paketleri yok ve MPS tüm eğitim iş yüklerinde NVIDIA/CUDA ile aynı davranış veya hızı garanti etmez. Küçük smoke test ve CPU/MPS denemeleri için kullanabiliriz; ana karşılaştırmalı eğitim için Colab GPU daha tekrarlanabilir başlangıç olur. [Apple PyTorch/MPS rehberi](https://developer.apple.com/metal/pytorch/)

Model geliştirme için BERTurk cased başlangıç modeli uygun ilk aday: `dbmdz/bert-base-turkish-cased`. Hugging Face kartı bu ağırlıklar için MIT lisansı bildiriyor. [BERTurk model kartı](https://huggingface.co/dbmdz/bert-base-turkish-cased)

## Veri yolu

1. **Kamusal başlangıç ve baseline:** `turkish-nlp-suite/turkish-wikiNER` veri seti. Makalesi 20.000 elle yeniden etiketlenmiş cümle (18.000 train, 1.000 dev, 1.000 test) ve CC BY-SA 4.0 lisansı bildiriyor. Etiketleri 19 sınıfa kadar uzanıyor; örneğin PERSON ve ORG var. Lisans atfını ve ShareAlike koşullarını model/dataset tesliminde korumamız gerekir. [ACL makalesi](https://aclanthology.org/2023.acl-long.768/), [veri deposu ve split bilgisi](https://github.com/turkish-nlp-suite/Turkish-Wiki-NER-Dataset)
2. **Etiketleri DLP hedeflerine indirgeme:** `PERSON`, `ORG`, `LOCATION`, `ADDRESS`. Kaynak `GPE`/`LOC` etiketleri `LOCATION` olarak birleştirilecek; bunlar açık adres değildir. `ADDRESS` için sentetik Türk adresleri ve elle kontrol edilmiş, kurum yazışmalarına benzeyen sentetik cümleler ayrıca üretilecek/etiketlenecek.
3. **Lisans ve mahremiyet:** Kurum içi gerçek prompt, log, isim veya adres eğitim için kullanılmayacak. Sentetik adres/persona değerleri gerçek kişi bilgileriyle karıştırılmayacak. Kaynak veri lisansları manifestte kaydedilecek.
4. **Veri sızıntısını önleyen split:** WikiNER'in resmi split'leri korunur; sentetik adresler split'e özel şablon, değer havuzu ve sayı aralıkları kullanır. Hazırlama betiği bire bir aynı token dizisinin farklı splitlerde bulunmasını reddeder. Gerçek veri gelince near-duplicate kontrolü ayrıca değerlendirilecek. Test seti eşik ayarlamasından ayrı tutulacak.

WikiANN de Türkçe PER/ORG/LOC için kullanılabilir bir baseline kaynağıdır; fakat yalnızca üç entity türü ve Wikipedia alanıyla sınırlı olduğu için, kurumsal adres tespiti hedefinin tamamını temsil etmez. [WikiANN kartı](https://huggingface.co/datasets/tner/wikiann)

## Eğitim ve değerlendirme adımları

1. BIO şemasını sabitle: `O` ile birlikte `B/I-PERSON`, `ORG`, `LOCATION`, `ADDRESS` etiketleri.
2. Public Turkish NER verisini bu şemaya dönüştür; ilk baseline'da yalnızca PERSON/ORG/LOCATION etiketlerini kullan. ADDRESS örneklerini ayrı, lisansı açık ve sentetik korpustan ekle.
3. BERTurk + `AutoModelForTokenClassification` ile ilk fine-tuning'i çalıştır. Token alt-parçalarına BIO etiketlerini hizala; özel tokenları ve devam alt-tokenlarını tutarlı maskele.
4. Her epoch sonunda dev ölçümü, en iyi checkpoint seçimi, sabit seed ve konfigürasyon saklama. Test setine eşik veya hiperparametre seçimi sırasında bakma.
5. Exact entity-span precision/recall/F1'ı entity türü bazında raporla. DLP için recall önemli olsa da, person/location yanlış pozitifleri kullanıcı akışını bozacağından threshold kalibrasyonunu ayrıca raporla. Regex+checksum baseline'ı ve NER+regex birleşimini aynı testte karşılaştır.
6. Başarılı modelde `id2label`, `label2id`, tokenizer ve model dosyalarını birlikte kaydet. Model lisansı, veri atıfları, veri sürümleri, seed, hiperparametre ve metrikleri model manifestine koy.
7. Yalnızca bundan sonra `EntityRecognizer` adaptörünü modele bağla. İlk entegrasyon devre dışı/opsiyonel kalsın. Runtime exception veya timeout'ta ham metin loglamadan kurallara düş; bozuk offset'i veri bütünlüğü hatası sayıp upstream'i durdur. Süreç izolasyonu mock worker ile test edildi; gerçek ağırlıklar ve GPU ile doğrulama model eğitimi sonrasına kaldı. DLP eşiği kalibrasyonu yapılmadan NER'i otomatik bloklama kararına bağlama.

Transformers token classification akışı `AutoModelForTokenClassification` ve Trainer ile uygulanıyor; exact BIO span precision/recall/F1 proje içinde hesaplanıyor. [Hugging Face token classification rehberi](https://huggingface.co/docs/transformers/main/tasks/token_classification)

## Bu projedeki plan

- **Hazır:** veri şeması, CoNLL dönüştürücü, split'e özel sentetik ADDRESS üretimi, BIO/token/split kontrolleri, metin içermeyen `quality_report.json` üretimi ve BERTurk eğitim/evaluation betiği. CLI doğrulama modu ML paketleri veya model indirmeden çalışabilir.
- **Tamamlandı:** WikiNER ve OSM açık veri kaynakları lisans/atıf bilgileriyle işlendi; ham veriler repoya eklenmedi.
- **Veri geldikten sonra:** dosya ve lisans kontrolü, dönüştürme, kalite raporu inceleme ve sentetik örneklerin gözden geçirilmesi.
- **Tamamlandı:** GPU ortamında BERTurk kişi/kurum/yer baseline eğitimi ve test ölçümü; sonuç gateway’e bağlanmadı.
- **Şimdi:** `real-address-v2` ile dört etiketli (kişi/kurum/yer/adres) ikinci eğitimi Colab’da çalıştır; OSM adres testini ve WikiNER testini ayrı yorumla. OSM cümle içi bağlamı yapay kalıplı olduğundan gerçek mesaj başarısı için bağımsız elle etiketli test gerekir.
- **En son:** modeli `EntityRecognizer` olarak entegre etme, gecikme ve fail-safe kontrolleri.

Modeli bu repo içinde `models/ner/` altında tutmak yerine eğitim çıktısını harici bir artifact klasöründe saklayacağız; küçük manifest/metrics dosyaları repoya girebilir, büyük ağırlıklar git'e eklenmez. Eğitim kodu `training/ner/`, lisans notları ve deney raporu `docs/` altında tutulabilir.
