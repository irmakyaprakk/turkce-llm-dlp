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

## Kural katmanı (hibrit tespitin ikinci bileşeni)

`src/turkce_llm_dlp/rules_ner.py`: güçlü yapısal ipuçlarına dayalı ORG ve ADDRESS tanıyıcı (kurumsal/yasal ekler; "No + numara" çapası, mahalle/sokak anahtar sözcükleri, daire/kat ve il/ilçe sözlüğü). Model gerektirmez, API'de varsayılan olarak açıktır (`TURKCE_DLP_RULE_NER=0` ile kapatılır).

Geliştirme setinde (realistic-v2 dev+test, 159 cümle) yalnız kural katmanı:

| | Adres | Kurum |
|---|---|---|
| Precision | 0,976 | 0,846 |
| Recall | 0,976 | 0,349 |
| Token kapsama | 0,998 | 0,593 |

Kurallar bu setin hatalarına bakılarak iyileştirildi (protokole uygun: kararlar geliştirme setinde); bu nedenle skorlar iyimserdir, genelleme mühürlü testte ölçülecektir. Gerçek metinde yanlış alarm taraması: UCI Türkçe ürün yorumlarının 33.991 benzersizinde 11 ORG ve 1 ADDRESS işareti (etiketsiz veri; precision değildir).

## Veri v3 (`augmented-v3`)

v2'ye ek olarak: 10.000 cümle 68 yeni çerçeveden (çok cümleli e-posta, şikâyet, yazım hatalı mesaj, İngilizce-Türkçe karışık, hukuk/sağlık, form, kod/log, numarasız adres tarifi), 1.250 yeni zor negatif ve 4.700 varlığı değiştirilmiş WikiNER cümlesi (ekler ünlü uyumuyla yeniden üretilir; kesme işaretsiz ek almış cins isimli varlıklar değiştirilmez). Toplam 55.123 eğitim cümlesi. Geliştirme ve mühürlü setle cümle/adres çakışması yok; mühürlü setteki kişi/kurum adlarının %7'si (4/55: Fenerbahçe, Halkbank, ODTÜ, Emir) WikiNER'in doğal cümlelerinde geçiyor.

## Elenen veri kaynağı

SUNLP Twitter NER (LREC 2022): Apache-2.0, ancak depo yalnızca tweet kimliği ve karakter konumu içeriyor; metin X API ile yeniden indirilmeli. Kullanılmadı (bkz. `docs/evaluation-protocol.md`).

## Deney 3 — `augmented_v2_seed42` (OSM ham biçimi %15)

| Test seti | Katı F1 | Son işlemli F1 | Token kapsama |
|---|---|---|---|
| WikiNER | 0,875 | – | – |
| OSM kalıplı cümle | **0,908** (v1: 0,360) | 0,999 | 0,9999 |
| Geliştirme seti (159 cümle) | 0,890 | 0,909 | 0,993 |

Geliştirme setinde tür bazında (son işlemli): kişi 0,958 · yer 0,974 · adres 0,907 · kurum 0,794. Kapsama: kişi/yer/adres 1,000, kurum 0,960. Hipotez doğrulandı: ham OSM biçimini eklemek sınır hatasını son işlem olmadan da büyük ölçüde giderdi; WikiNER korunur.

### Hata analizi bulguları ve düzeltmeler (geliştirme seti)

1. **Birleşim politikası:** kural ADDRESS'te daha isabetli (0,976 vs 0,907) ama model bir şey bulduğunda yok sayılıyordu. Yeni politika: ADDRESS'te kural öncelikli (`authoritative_types`), ORG'de model öncelikli, kural yalnız boşluk doldurur (`fallback_types`). Oracle testi (model = altın etiket): hibrit 1,000 — kural katmanı doğru bir model tahminini hiçbir zaman bozmaz.
2. **Değerlendirme üretimle birebir değildi:** TCKN/IBAN değerleri "fazla adres" sayılıyordu; üretimde bunları regex kuralları yakalar. `evaluate_rules.py` artık kaydedilmiş model tahminlerini üretimdeki `DetectionPipeline` içinde yeniden oynatır (`pipeline_*` metrikleri).
3. **Yanlış alarmlar:** SQL/kod anahtar sözcükleri, İngilizce gün adları, "İK" gibi birim kısaltmaları, büyük harfli cümleler → v3'e 22 yeni negatif çerçeve eklendi.
4. **Etiket hatası:** geliştirme setinde "Merkez Bankası" zor negatif yazılmıştı; gerçek kurum olduğu için ORG olarak düzeltildi.
5. **Bilinen belirsizlik:** kurum + alt birim ("… Üniversitesi Makine Fakültesi", "… Belediyesi Fen İşleri Müdürlüğü") sınırları; anotasyon kuralı yalnız ana kurumu etiketler. Kapsamayı etkilemez.

### Deney 3 — üretimle birebir ölçüm (DetectionPipeline, geliştirme seti 159 cümle)

| | Yalnız model | Yalnız kural | **Model + kural** |
|---|---|---|---|
| Genel F1 | 0,915 | 0,404 | **0,923** |
| Adres F1 | 0,929 | 1,000* | **0,976** |
| Kurum F1 | 0,803 | 0,460 | 0,803 |
| Kişi / Yer F1 | 0,958 / 0,974 | – | 0,958 / 0,974 |
| Precision (genel) | 0,905 | 0,953 | 0,914 |
| Token kapsama | 0,993 | 0,694 | 0,993 |

\* Kurallar bu setin hatalarına bakılarak geliştirildi; iyimser.

Sonraki düzeltme: modelin ürettiği numarasız/sözcüksüz adres parçaları (": 5", "blok") anotasyon kuralına aykırıdır; `plausible_address` filtresi bunları eler (oracle testinde hiçbir altın adres elenmez). Kalan ana zayıflık kurum: marka adları (Getir, AWS, Sahibinden, Iyzico) ve alt birim sınırları. Geliştirme setindeki markaları bir sözlüğe eklemek test setine göre ayar olacağından yapılmadı; v3 verisi bu türü hedefler.
