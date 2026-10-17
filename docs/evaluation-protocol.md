# Değerlendirme protokolü

Amaç: raporlanan sayıların modelin gerçek genelleme başarısını göstermesi, test setine göre yapılmış ayarlamaları değil.

## Veri bölümleri

| Bölüm | Dosyalar | Kullanım |
|---|---|---|
| Eğitim | `data/ner/processed/augmented-*` (train) | Model eğitimi |
| Eğitim-içi doğrulama | `augmented-*` (dev, ayrık kalıp aileleri) | Checkpoint seçimi (otomatik) |
| **Geliştirme seti** | `data/ner/manual/realistic_batch*.txt` → `realistic-v2` (dev + test) | Hata analizi, son işlem ve kural kararları, model/veri seçimi |
| **Mühürlü final test** | `data/ner/manual/sealed/*.txt` → `build_gold_set.py --sealed` | Yalnızca en sonda, seçilen yapılandırma için **bir kez** |
| **Mühürlü test 2** | `data/ner/manual/sealed2/*.txt` (bağımsız insan yazarlar) | v3 ↔ v4 kararı; bir kez kullanıldı |
| **Mühürlü test 3** | `data/ner/manual/sealed3/*.txt` (bağımsız insan yazarlar; `docs/sealed3-talimat.md`) | v4 ↔ v5 kararı; `prepare_sealed3.py dogrula` sonrası **bir kez** |

Not: `realistic-v2` adındaki "test" bölümü geçmiş deneylerle uyum için korunur; ancak bu set üzerinde hata analizi yapıldığı ve son işlem kuralları buna bakılarak düzeltildiği için **raporda geliştirme seti olarak anılır**. Final iddialar yalnızca mühürlü testle yapılır.

## Kurallar

1. Mühürlü cümleler eğitim verisine girmez; `prepare_augmented.py` bu dosyalardaki cümle, adres, kişi ve kurum adlarını da dolgu sözlüklerinden çıkarır.
2. Mühürlü test için hata analizi, eşik ayarı veya kural değişikliği yapılmaz. Final ölçümden sonra bir değişiklik yapılırsa yeni bir mühürlü set gerekir.
3. Mühürlü set farklı yazarlardan oluşur; sonuçlar yazar grubuna göre ayrı raporlanır (`by_batch`).
4. Final yapılandırma 3 seed ile eğitilir; ortalama ± örnek standart sapması raporlanır.

## Başarı ölçütleri (mühürlü testte)

| Metrik | Hedef | Gerekçe |
|---|---|---|
| Token kapsaması (son işlemli) | ≥ 0,99 | Hassas verinin sızmaması DLP'nin birincil amacı |
| Span F1 (son işlemli) | ≥ 0,92 | Doğru varlık sınırı tutarlı placeholder üretir |
| Precision | ≥ 0,90 | Aşırı maskeleme LLM yanıt kalitesini düşürür |

## Deney planı

Her eğitim tek bir hipotezi sınar:

| # | Deney | Hipotez |
|---|---|---|
| 1 | baseline | WikiNER + kalıplı OSM gerçek mesajlara genellemez |
| 2 | augmented-v1 | Yazım varyantı ve kalıp çeşitliliği genellemeyi artırır |
| 3 | augmented-v2 | OSM ham biçimi eklemek adres sınırlarını düzeltir |
| 4 | augmented-v3 | Serbest yazılmış cümleler ve varlık değiştirme kurum/adres hatalarını azaltır |
| 5 | model karşılaştırması | Sosyal medya verisiyle ön-eğitilmiş model (TurkishBERTweet) gayriresmî dilde daha iyidir |
| 6 | final (3 seed) | Seçilen yapılandırma mühürlü testte hedefleri karşılar |

Ek karşılaştırmalar: hazır Türkçe tweet NER modeli (`busecarik/berturk-sunlp-ner-turkish`), yalnız kural katmanı, model + kural hibriti.

## Değerlendirilip elenen kaynaklar

- **SUNLP Twitter NER (Çarık ve Yeniterzi, LREC 2022):** lisans Apache-2.0, ancak depo yalnızca tweet kimliği ve karakter konumu içerir; metin X API ile yeniden indirilmelidir (ücretli, silinmiş tweetler). Eğitim verisi olarak kullanılmadı; yazarların bu veriyle eğittiği hazır model karşılaştırma için kullanılır.
