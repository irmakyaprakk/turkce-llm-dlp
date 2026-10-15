# NER veri hazırlama

Eğitim verisi repoya dahil edilmez ve otomatik indirilmez. Kaynağı kullanıcı ayrıca indirir; önce lisans dosyasını ve atıf koşullarını korur.

## Kaynak

İlk baseline için [Turkish Wiki NER Dataset](https://github.com/turkish-nlp-suite/Turkish-Wiki-NER-Dataset) kullanılır. Beklenen CoNLL dosyaları dataset deposundaki `Turkish-Wiki-NER-Dataset/` alt klasöründe `train.conll`, `dev.conll`, `test.conll` biçimindedir. Kaynak lisansı CC BY-SA 4.0'dır; eğitim/veri/model çıktısı için atıf ve lisans notlarını sakla.

## Hazırlama

```bash
python training/ner/prepare_data.py \
  --wiki-dir /path/to/Turkish-Wiki-NER-Dataset/Turkish-Wiki-NER-Dataset \
  --output-dir /tmp/turkce-dlp-ner \
  --address-count 200 \
  --seed 42
```

Script resmi WikiNER split'lerini korur, PERSON/ORG/LOCATION etiketlerini proje şemasına dönüştürür ve her split için ayrı sentetik ADDRESS örnekleri üretir. Adres cümlelerinin template ailesi, değer havuzu ve numara aralığı split'e özeldir; bire bir aynı token dizisi farklı splitlerde bulunursa hazırlama hata verir. Çıktı JSONL'dir (`tokens`, `ner_tags`, `source`, `id`). `quality_report.json` kaynak etiket sayılarını ve proje şemasına alınmayan entity etiketlerini ayrıca raporlar; metin örneklerini içermez. Sentetik örnekleri eğitim öncesi elle gözden geçir; bu sentetik veri gerçek kurumsal dağılımın temsilcisi sayılmaz.

Hazırlanmış JSONL'yi ML paketlerini kurmadan doğrulayabilirsin:

```bash
python training/ner/train.py --data-dir /tmp/turkce-dlp-ner --validate-only
```

Fine-tuning için Colab gibi GPU runtime'ında PyTorch'u o runtime'a uygun kur, ardından `training/ner/requirements.txt` içindeki Transformers/Datasets bağımlılıklarını yükle. Proje dosyalarını ve hazırlanmış veri dosyalarını runtime'a taşıdıktan sonra:

```bash
python training/ner/train.py \
  --data-dir /content/turkce-llm-dlp/data/ner/processed/person-org-location-v1 \
  --labels-path /content/turkce-llm-dlp/training/ner/labels_person_org_location.json \
  --output-dir /content/drive/MyDrive/turkce-dlp-ner-run \
  --epochs 5 \
  --seed 42
```

İlk NER modeli için `person-org-location-v1` kullanılır. Bu set WikiNER'in gerçek cümle bağlamındaki kişi, kurum ve yer etiketlerini içerir. `ADDRESS` etiketi bu ilk modelden çıkarılmıştır. Dört etiketli adresli ikinci eğitim için `training/ner/colab_02_real_address.ipynb` kullanılır. Üçüncü notebook `training/ner/colab_03_address_sentence_context.ipynb`, gerçek OSM adreslerini yapay cümle bağlamlarında ölçer. Bu ölçüm gerçek müşteri mesajı başarısını göstermez.

Gerçek adres kayıtlarını sabit cümle kalıplarında görmek için `python training/ner/prepare_osm_sentence_context.py` çalıştır. Çıktı `data/ner/processed/openstreetmap-tr/sentence-context-v1/` altındadır. Adresler gerçek OSM kayıtlarıdır; cümle bağlamları yapaydır. Bu çıktıyı gerçek mesaj testi olarak sunma.

## Birden fazla test setini değerlendirme

`evaluate_suite.py` her etiketli JSONL seti için tür bazında tam-span precision/recall/F1, ADDRESS içermeyen cümlelerde ADDRESS alarm sayısı, LOCATION↔ADDRESS token karışması, kaynak sayıları ve girdi SHA-256'sını tek JSON raporunda verir. Uzun metinleri kelime sınırlarında örtüşmeli pencerelerle tarar; tek pencere sınırını aşan satır ve altın varlık sayılarını ayrıca raporlar. Küçük harf, büyük harf ve Türkçe karakterleri kaldırılmış kopyalar `--variants` ile aynı gold etiketlerle ölçülür.

```bash
python training/ner/evaluate_suite.py \
  --model seed42=/content/ner-seed42/model \
  --model seed43=/content/ner-seed43/model \
  --model seed44=/content/ner-seed44/model \
  --set wiki=data/ner/processed/person-org-location-v1/test.jsonl \
  --set context=data/ner/processed/openstreetmap-tr/sentence-context-v1/test.jsonl \
  --set realistic=data/ner/processed/realistic-v1/test.jsonl \
  --set hard-negative=data/ner/processed/hard-negatives-v1/test.jsonl \
  --uci-zip data/ner/processed/uci-turkish-user-reviews.zip \
  --variants \
  --output-json /content/evaluation-summary.json
```

Yalnızca elde bulunan setleri komuta ekle; henüz yazılmamış elle etiketli pozitif ve zor negatif setlerin skorunu bu araç üretemez. Birden fazla model verilirse ana metrikler set başına seed ortalaması ve örnek standart sapmasıyla özetlenir. `--uci-zip` ile eklenen etiketsiz UCI yorumlarındaki alarm sayısı precision/recall değildir; ham yorum metinleri rapora yazılmaz. Eğitim kodundaki `--fp16` ve `--warmup-ratio` seçenekleri varsayılan olarak kapalıdır; böylece eski eğitim davranışı ve baseline karşılaştırması korunur.

Komut BERTurk'ü indirip eğitir; model, dev ile seçilen checkpoint, ayrı test metrikleri ve train/dev/test SHA-256 manifestini output klasörüne kaydeder. Model Hub'a yükleme yapmaz. Başlangıç modeli: `dbmdz/bert-base-turkish-cased`.

## Eğitim sonrası uygulamaya bağlama

Yerel model çıktısı hazır olduktan sonra API, `TURKCE_DLP_NER_MODEL_DIR` ayarıyla modeli açıkça yükleyebilir; alternatif olarak `HuggingFaceNERRecognizer.from_local_model(...)` doğrudan kullanılabilir. Model Hub'dan indirilmez. Fast tokenizer ve örtüşmeli pencereler uzun girdiyi işler. Varsayılan `min_score=0.0` sınıflandırıcının tüm tanınan varlıklarını geçirir; eşik ancak dev verisinde ölçülüp seçildikten sonra yükseltilmelidir. Henüz eğitilmiş model klasörü yoksa bu adım çalıştırılamaz.

```python
from turkce_llm_dlp.ner import DetectionPipeline, HuggingFaceNERRecognizer

recognizer = HuggingFaceNERRecognizer.from_local_model("/path/to/run/model")
pipeline = DetectionPipeline([recognizer])
```

API varsayılan olarak modelsiz çalışır. Model dizini ayarlanmış ama yükleme başarısızsa, servis ham yolu loglamadan kural tabanlı modda açılır. Inference varsayılan olarak 2 saniyeyle sınırlı ayrı bir süreçte çalışır; zaman aşımında süreç sonlandırılır ve servis yeniden başlatılana dek kural tabanlı mod kullanılır. `TURKCE_DLP_NER_TIMEOUT_SECONDS` ile süre ayarlanabilir. Bozuk offset fail-closed davranır. Bu süreç izolasyonu henüz gerçek eğitilmiş model ve GPU üzerinde doğrulanmalıdır.

Ham ve işlenmiş veri, büyük model ağırlıkları gibi repoya eklenmemelidir. Yalnızca şema, veri hazırlama kodu, kaynak/lisans bilgisi ve deney özeti repoda tutulur.


## Adresli ikinci eğitim

`real-address-v2` WikiNER'in mevcut cümlelerini ve her split'e ayrılmış gerçek OSM adres dizilerini birleştirir. Splitler 23.967 train, 1.600 dev, 1.600 test kaydı içerir; her splitte WikiNER kayıtları aynı kalır. Eğitim komutu ve test kaynaklarını ayrı gösteren değerlendirme `training/ner/colab_02_real_address.ipynb` içinde hazırdır. OSM test ölçümü yalnızca tek başına adres satırlarında geçerlidir; doğal cümle performansını göstermez.
