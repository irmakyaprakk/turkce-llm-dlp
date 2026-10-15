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

**Yorum.** Gerçekçi mesajlarda büyük iyileşme; WikiNER'de de gerileme yok. Sabit kalıplı OSM cümle testindeki düşüş iki adımda teşhis edildi. İlk teşhis (yalnız span içine bakan) adreslerin bölündüğünü düşündürdü; ancak bitişik parçaları birleştirmek skoru yalnızca 0,360 → 0,377'ye taşıdı, yani hipotez yanlıştı. Sınırları da gösteren ikinci teşhis (1.500 cümle) asıl nedeni buldu: hatalı 965 cümlenin ~%99'unda adresin başı doğru, ancak model **cümle sonu noktasını adrese katıyor** ("…Mahallesi."). Eğitim verisindeki "Sk.", "Mah." gibi kısaltmalarda nokta adresin parçası olduğundan model bunu genellemiş. Hiç kaçan adres yok, **token kapsaması 0,9996**: tam-span F1 düşük, sızıntı yok.

**Alınan önlemler.**
1. Çalışma zamanı: `DetectionPipeline` (a) virgül/eğik çizgi/tireyle ayrılmış bitişik ADDRESS parçalarını birleştirir (`merge_adjacent_spans`; `;` ve satır sonu ayrı adresleri ayırdığı için birleştirmez), (b) adres sonundaki cümle noktalamasını kısaltmaya ait değilse span dışına alır (`trim_address_span`).
2. Değerlendirme: katı F1'in yanında son işlemli F1 (çalışma zamanı davranışı) ve token kapsaması raporlanır (`evaluate_model.py`).
3. Veri: `augmented-v2`, adreslerin %15'ini OSM ham biçiminde bırakır (`--osm-original-ratio 0.15`).

**Not (yanlılık riski).** Altın setin ilk partisi ile eğitim kalıpları aynı kaynaktan yazıldı; üslup benzerliği skorları şişirebilir (ör. kişi F1 = 1,000). Sonraki partilere farklı kişilerin yazdığı cümleler eklenecek ve sonuçlar yazar grubuna göre ayrıca raporlanacak.

### Son işlem sonrası (Deney 2 modeli, yeniden değerlendirme)

| Test seti | Katı F1 | Son işlemli F1 | Token kapsama |
|---|---|---|---|
| OSM adres, sabit kalıplı cümle | 0,360 | **0,996** | 0,9995 |
| Altın set (1. parti) | 0,935 | 0,929* | 0,994 |

\* Bu ölçümde son işlem önce birleştirip sonra kırpıyordu; model ilk adresin sonuna `;` kattığında noktalı virgülle ayrılmış iki adres birleşti (altın sette adres F1 0,882 → 0,849). Sıra "kırp → birleştir → kırp" olarak düzeltildi ve testle güvenceye alındı; yeniden ölçüm bekleniyor.
