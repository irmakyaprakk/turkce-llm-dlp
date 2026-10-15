# NER deney sonuçları

Tam-span (exact match) precision / recall / F1. Altın test seti küçüktür (73 cümle; 17 adres, 27 kurum, 38 kişi, 17 yer varlığı) — sonuçlar yön gösterir, nihai rapor ~400 cümlelik setle yapılacaktır.

## Deney 1 — baseline (`baseline_ctx_seed42`)

Veri: `real-address-context-v1` (WikiNER + OSM adres satırları + 8 sabit kalıpta OSM adresleri). BERTurk cased, 3 epoch, seed 42, fp16, Colab T4.

| Test seti | Genel F1 | Kişi | Kurum | Yer | Adres |
|---|---|---|---|---|---|
| WikiNER test | 0,861 | 0,919 | 0,747 | 0,844 | – |
| OSM adres satırları | 1,000 | – | – | – | 1,000 |
| OSM adres, sabit kalıplı cümle | 0,9999 | – | – | – | 0,9999 |
| **Altın set (gerçekçi mesajlar)** | **0,781** | 0,886 | 0,630 | 0,970 | **0,629** |

Altın sette ayrıntı: kişi P 0,969 / R 0,816; kurum P 0,630 / R 0,630; adres P 0,611 / R 0,647.

**Yorum.** OSM adres testleri doygundur: test adresleri eğitimle aynı tek biçimde olduğu için model kalıbı ezberler ve %100 alır. Gerçekçi mesajlarda adres F1'i 0,63'e düşer; bu, veri zenginleştirmenin (bkz. `docs/data-augmentation.md`) motivasyonudur. Kişi tespitinde kesinlik yüksek ama duyarlılık düşüktür (küçük harfli/Türkçe karaktersiz adlar kaçıyor olabilir).

## Deney 2 — zenginleştirilmiş veri (`augmented_seed42`)

Veri: `augmented-v1` (bkz. `docs/data-augmentation.md`). Diğer her şey Deney 1 ile aynı.

| Test seti | Genel F1 | Kişi | Kurum | Yer | Adres |
|---|---|---|---|---|---|
| WikiNER test | 0,876 (+1,5) | 0,930 | 0,766 | 0,859 | – |
| OSM adres satırları | 1,000 | – | – | – | 1,000 |
| OSM adres, sabit kalıplı cümle | **0,360** (−64) | – | – | – | 0,360 |
| **Altın set** | **0,935 (+15,4)** | 1,000 | 0,852 | 0,971 | **0,882 (+25,3)** |

**Yorum.** Gerçekçi mesajlarda büyük iyileşme; WikiNER'de de gerileme yok. Sabit kalıplı OSM cümle testindeki düşüş teşhis edildi (2.000 adres): adreslerin 1.284'ünde model tüm token'ları ADDRESS olarak işaretliyor ama adresi OSM ham biçimindeki virgülde ikiye bölüyor ("…No: 5" + "X Mahallesi"); hiç kaçan adres yok, **token kapsaması 0,9995**. Yani tam-span F1 düşük, sızıntı yok. Neden: zenginleştirilmiş veride mahalle adresin başında yer alır; model "Mahallesi"yi yeni bir adres başlangıcı sanıyor.

**Alınan önlemler.**
1. Çalışma zamanı: `DetectionPipeline` noktalamayla ayrılmış bitişik ADDRESS parçalarını birleştirir (`merge_adjacent_spans`).
2. Değerlendirme: katı F1'in yanında birleştirilmiş F1 ve token kapsaması raporlanır (`evaluate_model.py`).
3. Veri: `augmented-v2`, adreslerin %15'ini OSM ham biçiminde bırakır (`--osm-original-ratio 0.15`).

**Not (yanlılık riski).** Altın setin ilk partisi ile eğitim kalıpları aynı kaynaktan yazıldı; üslup benzerliği skorları şişirebilir (ör. kişi F1 = 1,000). Sonraki partilere farklı kişilerin yazdığı cümleler eklenecek ve sonuçlar yazar grubuna göre ayrıca raporlanacak.
