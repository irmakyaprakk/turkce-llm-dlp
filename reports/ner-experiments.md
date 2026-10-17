# NER deney sonuçları

Tam-span (exact match) precision / recall / F1. Altın test seti küçüktür (73 cümle; 17 adres, 27 kurum, 38 kişi, 17 yer varlığı) — sonuçlar yön gösterir, nihai rapor ~400 cümlelik setle yapılacaktır.

## Veri kaynakları (değerlendirme partileri)

| Parti | Kullanım | Kaynak türü |
|---|---|---|
| `realistic_batch1`, `realistic_batch2` | Geliştirme seti (realistic-v2) | Yapay zekâ destekli üretildi; Irmak Yaprak gözden geçirdi ve etiketleri onayladı |
| `realistic_batch3` | Geliştirme seti (realistic-v3), günlük dil | Yapay zekâ destekli üretildi; Irmak Yaprak gözden geçirdi ve etiketleri onayladı. augmented-v5 şablonlarıyla aynı kaynak → **iyimser** |
| `sealed/sealed_proje.txt` (eski adı `sealed_asistan.txt`) | Mühürlü test 1 | Yapay zekâ destekli üretildi; Irmak Yaprak gözden geçirdi ve etiketleri onayladı |
| `sealed/sealed_irmak.txt` | Mühürlü test 1 | Bağımsız insan yazar: Irmak Yaprak elle yazdı |
| `sealed2/sealed2_toplanan.txt` | Mühürlü test 2 | Bağımsız insan yazarlar: arkadaşlar elle yazdı |
| `sealed3` (toplanacak) | Mühürlü test 3, v5'in tek bağımsız değerlendirmesi | Bağımsız insan yazarlar (talimat: `docs/sealed3-talimat.md`) |

Sonuç tablolarında yapay zekâ destekli ve bağımsız insan yazar partileri ayrı sütunlarda verilir. Geliştirme setlerinin tamamı yapay zekâ destekli olduğundan genelleme iddiası yalnız bağımsız insan yazar setlerine (sealed_irmak, sealed2, sealed3) dayanır.

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

## Deney 4 — `augmented_v3_seed42`

Geliştirme seti, üretim boru hattı (model + kural):

| | v2 | v3 |
|---|---|---|
| Genel F1 | 0,923 | **0,938** |
| Precision | 0,914 | **0,930** |
| Adres | 0,976 | **1,000** |
| Kurum | 0,803 | **0,844** |
| Kişi | 0,958 | **0,979** |
| Yer | 0,974 | 0,927 |
| Token kapsama | 0,993 | 0,991 |
| WikiNER (katı) | 0,875 | 0,861 |

Yalnız model (kuralsız) v3: genel 0,933, adres 0,976. Eklenen negatifler kod/İngilizce/kısaltma yanlış alarmlarını giderdi; marka adları (Getir, Sahibinden, Iyzico) artık bulunuyor. Gerileme: yer kesinliği — şehir adı içeren kurum adlarında ("Antalya Kurumlar Vergi Dairesi", "İstanbul Anadolu 7. İş Mahkemesi") şehir ayrı LOCATION olarak ayrılıyor (tür hatası, sızıntı değil); WikiNER'de 1,4 puan düşüş (alan uyarlaması ödünleşimi). Tek seed ve 159 cümle: 1–2 puanlık farklar gürültü içinde olabilir; final karşılaştırma 3 seed ile yapılacak.

Son işlem eki: sözcük içermeyen kişi/kurum/yer span'ları ("No : 5") `has_name_word` ile elenir (oracle testinde kayıp yok).

## Kurum sınırları: anotasyon kuralı ve son işlem

**Karar:** Kurum adı büyük harfle yazılmış alt birimiyle birlikte geçtiğinde tamamı tek ORG'dur ("Yıldız Teknik Üniversitesi Makine Fakültesi"); kurum adının içindeki şehir ayrı LOC yapılmaz ("Antalya Kurumlar Vergi Dairesi"). Gerekçe: alt birim de tanımlayıcıdır ve DLP açısından tek placeholder ile maskelenmesi bağlamı daha iyi korur. Kural `ANNOTATION_RULES.md`'ye eklendi; geliştirme setinde 4, mühürlü sette 1 etiket bu kurala göre güncellendi (mühürlü sette model sonucuna bakılmadan, yalnız kural tutarlılığı için).

**Son işlem (`merge_org_compounds`):** (1) boşlukla bitişik "şehir + kurumsal sözcükle biten kurum" tek ORG olur; ek almış yer adı ("Ankara'da") birleşmez; (2) ORG'dan sonra gelen, büyük harfli ve kurumsal bir sözcükle (Fakültesi, Müdürlüğü, Polikliniği, Bölümü…) biten ifade ORG'a katılır. Oracle testinde kayıp yok.

**Bilinen, ele alınmayan:** "Hazine ve Maliye Bakanlığı" gibi bağlaçlı adlar; geliştirme setindeki marka adlarını (AWS, Notion) sözlüğe eklemek test setine göre ayar olacağından yapılmadı — bağımsız kaynaklı kurum sözlüğü gelecek iş.

## Deney 5 — model karşılaştırması: TurkishBERTweet (`bertweet_v3_seed42`)

Aynı veri (augmented-v3), aynı eğitim ayarları; yalnız temel model `VRLLab/TurkishBERTweet` (max_length 128).

| Geliştirme seti (model + kural) | BERTurk v3 | TurkishBERTweet v3 |
|---|---|---|
| Genel F1 | **0,940** | 0,787 |
| Precision | **0,934** | 0,747 |
| Token kapsama | **0,991** | 0,984 |
| WikiNER (katı) | **0,861** | 0,742 |

Hata listesi büyük harfle başlayan sıradan sözcükleri varlık sanan bir model gösteriyor ("Yarın", "Maaş", "Hemşire", "IBAN"). Olası nedenler: model küçük harfe çeviren tweet ön işlemesiyle ön-eğitilmiş, biz büyük/küçük harf korunmuş metin verdik (büyük harf NER'in en güçlü ipuçlarından); 128 token sınırı; hiperparametreler BERTurk için seçildi. Sonuç: bu kurulumda BERTurk açıkça üstün; TurkishBERTweet hakkında genel bir yargı değildir. Ek ayar eğitimleri yapılmadı.

**Final yapılandırma kararı:** BERTurk (`dbmdz/bert-base-turkish-cased`) + augmented-v3 + kural katmanı + son işlem. 3 seed (42, 43, 44) ile eğitilip mühürlü testte bir kez ölçüldü (`colab_final.ipynb`); sonuç aşağıda.

## FİNAL SONUÇ — mühürlü test (v1: BERTurk + augmented-v3 + kural katmanı + son işlem)

3 seed (42, 43, 44), üretim boru hattıyla birebir (DetectionPipeline), ortalama ± std. Mühürlü test bir kez çalıştırıldı;
sonucuna göre hiçbir ayar yapılmadı.

| | Geliştirme (159 cümle; yapay zekâ destekli) | Mühürlü — tümü (71) | Mühürlü — yapay zekâ destekli (56) | Mühürlü — bağımsız insan yazar (15) |
|---|---|---|---|---|
| Genel F1 | 0,971 ± 0,004 | **0,857 ± 0,008** | 0,908 ± 0,019 | 0,622 ± 0,040 |
| Precision | 0,971 ± 0,004 | 0,859 ± 0,003 | 0,912 ± 0,013 | 0,618 ± 0,060 |
| **Token kapsama (DLP)** | 0,993 ± 0,001 | **0,945 ± 0,010** | 0,959 ± 0,009 | 0,833 ± 0,018 |
| Kişi F1 / kapsama | 0,982 / 0,996 | 0,929 / 0,989 | 0,966 / 0,987 | 0,799 / 1,000 |
| Kurum F1 / kapsama | 0,926 / 0,969 | 0,797 / 0,945 | 0,849 / 0,953 | 0,356 / 0,833 |
| Yer F1 / kapsama | 0,983 / 1,000 | 0,833 / 0,788 | 0,977 / 1,000 | 0,451 / 0,417 |
| Adres F1 / kapsama | 1,000 / 1,000 | 0,811 / 0,950 | 0,794 / 0,947 | 1,000 / 1,000 |

Varlık sayıları: yapay zekâ destekli 80 (30 kişi, 25 kurum, 14 yer, 11 adres); bağımsız insan yazar 17 (7 kişi, 3 kurum, 6 yer, 1 adres).

**Yorum.**

1. **Geliştirme–mühürlü farkı (0,971 → 0,857):** Geliştirme setinde hata analizi yaparak kural ve son işlem ayarladık; o set artık iyimser bir tahmin veriyor. Mühürlü test bu iyimserliği ölçtü. Raporlanacak genelleme performansı mühürlü sonuçtur.
2. **DLP açısından asıl metrik token kapsamadır: 0,945.** Kişisel veri token'larının ~%94,5'i maskeleniyor. Kişi adlarında kapsama 0,989; F1'e göre çok daha yüksek olması, hataların çoğunun **sınır** hatası olduğunu gösterir (varlık maskeleniyor ama span altın etiketle birebir örtüşmüyor). Adres de böyle: F1 0,811, kapsama 0,950.
3. **Bağımsız yazarın gayriresmî dili en zor durum (0,622):** Küçük harf, yazım hatası, bitişik ek ("ceyhana", "bucaktaki", "appleda", "tadımın"). Sızıntı en çok yer adlarında (kapsama 0,417) ve küçük harfli kurum/işletme adlarında. Bu, eğitim verisindeki yapay küçük harf varyantlarının gerçek gayriresmî yazımı tam temsil etmediğini gösteriyor. Not: bu grup yalnız 17 varlık içeriyor; tek varlık ≈ %6 oynatır, sonuç yön gösterir ama kesin değildir.
4. **Seed oynaması küçük** (genel F1 std 0,008): sonuç tek bir şanslı eğitime bağlı değil.

**Sonraki adım için çıkarım:** En büyük kazanç alanı gayriresmî/küçük harfli yer ve işletme adları. v1.1 yer adı sözlüğü tam bu boşluğa yönelik; ancak mühürlü cümleler tasarım sırasında görüldüğü için v1.1'in etkisi yalnızca geliştirme setinde raporlanır.

## v1.1 — Yer adı sözlüğü (`LocationGazetteer`)

Modelin boş bıraktığı yerleri dolduran (`fallback_types = {LOCATION}`), bağımsız kaynaklı sözlük katmanı.

- **Kaynak:** PTT posta kodu listesi (81 il, 973 ilçe, ~32 bin mahalle/köy adı; `turkey-neighbourhoods`, MIT) ve elle derlenmiş ülke/bölge listesi. Geliştirme seti ve mühürlü test bu listelere hiçbir şey eklemek için kullanılmadı.
- **Belirsizlik:** Her ad için eşleşme düzeyi veriden hesaplanır (`build_gazetteer.py`). WikiNER'de küçük harfle geçen adlar ("orta", "kale") ve gündelik yorumlarda küçük harfle büyük harften sık yazılanlar ("kulp", "çubuk", "bulanık") yalnız büyük harfle; ay/gün adları ("Aralık", "Pazar") yalnız "ili/ilçesi" bağlamıyla; kişi adı olarak sık geçenler ("Selim", "Menderes") yalnız kesme işaretli ekle eşleşir. Ek atma ("ceyhana", "izmirdeki"), ünsüz yumuşaması ve ASCII yazım desteklenir; ek atınca yaygın sözcüğe inen biçimler ("karşı" ≠ "Kars"+"ı") reddedilir.
- **Kalıplar:** "<ad> Mahallesi / köyü" ve numarasız "<ad> Caddesi / Sokak / Bulvarı" (anotasyon kuralı: LOC).
- **Adres kuralına katkı:** Adres sağa genişletmesi artık belirsiz olmayan 973 ilçenin tamamını tanıyor (önceden ~120 elle yazılmış ad).

| Geliştirme seti (159 cümle), model YOK | LOC P | LOC R | LOC F1 | ADDRESS F1 |
|---|---|---|---|---|
| Kural katmanı | — | 0 | 0 | 1,000 |
| Kural katmanı + sözlük | **1,000** | 0,816 | 0,899 | 1,000 |

- Oracle testi (model = altın etiket): kayıp yok (sözlük modelin doğru tahminini hiçbir yerde ezmiyor).
- Gündelik dil taraması (UCI yorumlarından rastgele 5.000 yorum, etiketsiz): 102 eşleşme, elle bakıldığında ~%90'ı gerçek yer adı (kalan hatalar: "susuz", "durağan", "şok" gibi sözcükler sözlük ayarıyla giderildi).
- WikiNER testi (yalnız sözlük): LOC precision 0,81 (kalan "hataların" bir kısmı WikiNER'in eksik etiketleri: "Portekiz'in", "Irak'ın").

**Model + sözlük (geliştirme seti, 3 seed):**

| | Genel F1 | Yer F1 | Yer kapsama | Genel kapsama |
|---|---|---|---|---|
| v1 (sözlüksüz) | 0,971 ± 0,004 | 0,983 ± 0,020 | 1,000 | 0,993 ± 0,001 |
| v1.1 (sözlüklü) | 0,971 ± 0,004 | 0,983 ± 0,020 | 1,000 | 0,993 ± 0,001 |

Sonuçlar birebir aynı. Sebep: geliştirme setinde model yer adlarının tamamını zaten maskeliyor (kapsama 1,000); kalan yer hataları tür ya da sınır hatası ("modada" → ORG) ve sözlük model önceliğine saygı duyduğu için bunları değiştirmez. Yani sözlük **zarar vermiyor**, ama bu sette **ölçülebilir bir kazanç da sağlamıyor**. Beklenen katkısı modelin yer adını hiç göremediği gayriresmî metinde (mühürlü bağımsız yazar grubunda yer kapsaması 0,417); ancak o set kirlendiği için bu katkı iddia edilmez. Sözlük, zararsız ve açıklanabilir bir güvenlik ağı olarak üretimde açık tutulur (`TURKCE_DLP_GAZETTEER=0` ile kapatılabilir).

**Mühürlü test notu:** Sözlük tasarlanırken mühürlü cümlelerin bir kısmı tasarım sırasında örnek olarak denendi. Bu yüzden v1.1'in mühürlü test sonucu "temiz" sayılmaz ve raporlanmaz; mühürlü testin resmî sonucu sözlüksüz v1 yapılandırmasınındır. Yeni bir bağımsız mühürlü parti yazılmadığından v1.1'in etkisi yalnızca geliştirme setinde raporlanır; bu sonuç iyimser kabul edilmelidir (geliştirme seti hata analizinde kullanıldı).

## v4 — gündelik dil turu ve mühürlü test 2 (sealed2)

**Veri:** augmented-v4 = v3 + 12.000 gündelik dil cümlesi (`docs/data-augmentation.md`), sealed2 toplanmadan önce donduruldu (commit `fbb5f18`).
**Test:** sealed2 — proje sahibinin çevresinden toplanan 100 mesaj (kaynak: bağımsız insan yazarlar, arkadaşlar elle yazdı) (69 kişi, 62 adres, 35 kurum, 28 yer). Bir kez çalıştırıldı;
sonuca göre ayar yapılmadı. Önceden belirlenen karar kuralı: "v4 sealed2'de v3'ten iyiyse üretimde v4".

| sealed2 (3 seed) | v3 hibrit | **v4 hibrit** | v3 yalnız model | v4 yalnız model |
|---|---|---|---|---|
| Genel F1 | 0,857 ± 0,012 | **0,865 ± 0,011** | 0,877 ± 0,003 | 0,880 ± 0,020 |
| **Token kapsama (DLP)** | 0,953 ± 0,014 | **0,971 ± 0,005** | 0,969 ± 0,022 | 0,987 ± 0,004 |
| Kurum F1 / kapsama | 0,807 / 0,866 | **0,858 / 0,963** | 0,803 / 0,915 | 0,837 / 0,992 |
| Kişi F1 | 0,995 | **1,000** | 0,995 | 1,000 |
| Yer F1 / kapsama | 0,453 / 0,820 | 0,354 / 0,887 | 0,454 / 0,747 | 0,335 / 0,847 |
| Adres F1 / kapsama | 0,887 / 0,976 | 0,887 / 0,977 | 0,952 / 1,000 | 0,957 / 1,000 |

Geliştirme setinde (3 seed, hibrit): v3 0,971 → v4 0,973 F1; kapsama 0,993 → 0,996 (gerileme yok).

**Karar:** Üretimde **v4** (karar kuralı sağlandı). Seed'ler geliştirme setinde eşit (std 0,000) olduğundan varsayılan seed 42 kullanılır; seçim test setine bakılarak yapılmadı.

**Yorum.**

1. **v4'ün kazancı DLP metriğinde açık:** sızan kişisel veri token'ı %4,7'den %2,9'a indi (hibrit kapsama 0,953 → 0,971). En büyük iyileşme hedeflenen alanda, yerel işletme/kurum adlarında (kurum kapsaması 0,866 → 0,963). Seed oynaması da azaldı (std 0,014 → 0,005).
2. **Yer F1'i düşük (0,35–0,45) ama kapsama yüksek (0,85–0,89):** Yerler çoğunlukla maskeleniyor, fakat tür/sınır altın etiketle uyuşmuyor. Bunun önemli bir nedeni, sealed2 etiketlenirken alınan yeni karar: numarasız bina/apartman/plaza/park/kampüs adları LOC sayıldı ("çınar plaza", "zümrüt apartmanı"). Eğitim verisinde bu türde örnek yok; model bunları çoğunlukla kurum sanıyor ya da kısmen işaretliyor. Yer sayısı az (28) olduğu için bu satırın oynaklığı yüksek (std 0,09–0,18).
3. **Bu sette kural katmanı zarar veriyor (bulgu):** Yalnız model, hibritten daha iyi (v4: 0,880 vs 0,865 F1; 0,987 vs 0,971 kapsama). Kayıp adreste: hibritte adres sonucu seed'den bağımsız (std 0,000), çünkü adres kuralı öncelikli (`authoritative`). Model mükemmel olsa bile (oracle) hibrit bu sette 0,953 F1'de kalıyor; yani sınırı kural katmanı belirliyor. Bu setin adres biçimi ("…no 12, kadıköy, istanbul; tc …", küçük harf, virgüllü il/ilçe) kural katmanının tasarlandığı biçimden farklı. Geliştirme setinde ve ilk mühürlü testte kural katmanı katkı sağlamıştı. **Bu sete bakılarak kural değiştirilmedi;** önceden belirlenen hibrit yapılandırma korunur. Gelecek iş: adres kuralının önceliğini modelle güven tabanlı birleştirmek ve bunu yeni, bağımsız bir sette ölçmek.
4. **Test setinin sınırlılığı:** sealed2 mesajları büyük ölçüde aynı kalıbı izliyor ("X'e söyle, …; adres: … no N, şehir; tc …, tel …"), yazım hatası azdır. Sonuçlar gündelik yazımı temsil eder ama gerçek mesajlaşmanın tüm dağınıklığını kapsamaz.

## v4.1 — boru hattı düzeltmeleri (yeniden eğitim yok)

sealed2'de görülen iki sorun için, aynı v4 modelleriyle (3 seed) yalnızca boru hattı değiştirildi:

1. **Adres birleşimi:** Öncelikli kural adresi ile çakışan model adresi en geniş kapsamla birleşir; birleşim TCKN/IBAN/telefon gibi kural tespitlerine, `;` ve satır sonuna taşmaz; ayrı yazılmış hâl eki ("trabzon a") dışarıda kalır.
2. **Tesis / bina adı kuralı** (`FacilityRecognizer`): numarasız apartman, plaza, iş merkezi, iş hanı, pasaj, park, kampüs, AVM, rezidans adı LOC; numaralı adresin hemen önündeyse adresin parçası. Kural `ANNOTATION_RULES.md`'ye yazıldı; geliştirme setinde bir etiket buna uyduruldu ("Beşiktaş kampüsündeki").

| | Geliştirme v4.0 | **Geliştirme v4.1** | sealed2 v4.0 (resmî) | sealed2 v4.1 (*sonradan*) |
|---|---|---|---|---|
| Genel F1 | 0,969 | **0,973** | 0,865 ± 0,011 | 0,921 ± 0,007 |
| Kapsama | 0,995 | **0,996** | 0,971 ± 0,005 | 0,994 ± 0,004 |
| Yer F1 | 0,956 | **0,982** | 0,354 | 0,723 |
| Kurum F1 | 0,943 | 0,943 | 0,858 | 0,957 |
| Adres F1 | 1,000 | 1,000 | 0,887 | 0,895 |

(Geliştirme v4.0 değeri, güncellenen "Beşiktaş kampüsündeki" etiketiyle yeniden hesaplandığı için önceki 0,973'ten düşüktür.)

**Yorum.** Geliştirme setinde gerileme yok, küçük kazanç var. sealed2'de büyük iyileşme görülüyor (sızan kişisel veri token'ı %2,9 → %0,6); ancak değişiklikler bu setteki hata türleri görülerek tasarlandığı için bu sonuç **kör değildir ve iyimser kabul edilmelidir**. Resmî sealed2 sonucu v4.0 değeridir. Adres F1'i hâlâ yalnız modelin gerisinde (0,895 vs 0,957): birleşim kapsamayı artırıyor ama sınır uyumunu tam çözmüyor; adres kuralının önceliğinin güven tabanlı hâle getirilmesi gelecek iştir.

**Üretim yapılandırması (final):** BERTurk + augmented-v4 (seed 42) + kural katmanı + yer adı sözlüğü + tesis kuralı + adres birleşimi + son işlem; ayrıca kural tabanlı TCKN, IBAN, kart, e-posta, **telefon** ve gizli anahtar tespiti.

## Üretim–değerlendirme tutarlılığı (model yerelde çalıştırılırken bulundu)

Model ilk kez yerel servise bağlandığında üretimdeki çıkarımın değerlendirmeden farklı sözcük bölütlemesi kullandığı görüldü:
HF `pipeline` kesme işaretinde sözcüğü bölüyordu ("Market" + "'" + "in"); model eğitimde eklere etiket görmediği için "in"
parçasını ayrı bir kurum sanabiliyordu. Çıkarım, eğitim ve değerlendirmedeki kurala birebir uyan sözcük düzeyinde bir
tahminciyle (`WordLevelPredictor`: sözcükler `is_split_into_words` ile verilir, etiket ilk alt parçadan alınır, uzun metin
örtüşmeli sözcük pencerelerine bölünür) değiştirildi. Kesme işaretinden sonraki ek de varlığın dışında bırakılır
("Ayşe Demir'e" → "Ayşe Demir"), böylece aynı kişi farklı eklerle aynı yer tutucuyu alır.

Doğrulama: geliştirme setinin 159 cümlesinin tamamında üretim tahmini, değerlendirme betiğinin tahminiyle varlık düzeyinde
aynı (0 fark). Raporlanan sayılar artık doğrudan üretim davranışını temsil eder. Yerel CPU'da tek mesajlık tespit süresi
ısınmadan sonra ~50 ms (seed 42, v4.1 boru hattı).

## Gerçek kullanımda görülen hatalar ve kural eklemeleri (v4.2, model değişmedi)

Eklentiyle gerçek ChatGPT denemesinde iki kaçırma görüldü ve modelden bağımsız kurallarla giderildi:

- **Telefon:** Öneksiz, ayraçsız numaralar ("ararmısın 5477658697", "hey 2125462345") yakalanmıyordu. Cep operatör önekleri (50x/53x/54x/55x/56x) ve BTK il alan kodlarıyla başlayan 10 haneli sayılar artık telefon sayılır; "sipariş/fatura/takip/hesap no" bağlamı ve tutar birimleri ("5000000000 TL") yanlış alarmı önler.
- **Kişi (tanıtım ipuçları):** Model "benim adım ırmak", "merhaba ben ırmak" gibi küçük harfli, sözcük anlamı da olan adları kaçırıyordu. Model öncelikli (yalnızca boşluk dolduran) bir kural eklendi: "benim adım / ismim / Ad Soyad: / İsim:" sonrası ad; "ben X" ve "X bey/hanım" için ad listesinde (262 yaygın ad) bulunma şartı; "bir adım gerisinde", "işim gereği", "Ben Samsung Galaxy…", "BEN MEMNUN" gibi kalıplar elenir.
  Geliştirme setinde yalnız kural: 10/10 doğru kişi (yanlış alarm 0); 8.000 rastgele ürün yorumunda 4 eşleşme (3'ü gerçek kişi adı). Hibrit F1 değişmedi (0,9725; kural yalnız modelin boş bıraktığı yeri doldurur).

## Gerçek kullanımda görülen hatalar, ikinci tur (v4.3, model değişmedi — sonradan/post-hoc)

ChatGPT denemesinde tamamı küçük harfle yazılmış bir tanıtım mesajı ("selam ben ırmak soyisimim yaprak bucakta okuyorum mehmet akif üniversitesi teknoloji fakültesi") hiç maskelenmeden gitti. Model küçük harfli metinde kişi/kurum/yer bulmuyordu; kural katmanına model öncelikli (boşluk dolduran) üç ekleme yapıldı:

- **Soyadı ipucu:** "soyadım / soyismim / soyisimim / Soyadı:" sonrasındaki 1-2 sözcük; "ben <belirsiz ad>" ardından soyadı ipucu geliyorsa ad kabul edilir. Ad dizisi eki bitişik yer adında durur ("ali yılmaz ankarada" → "ali yılmaz").
- **Küçük harfli kurum:** "<ad> üniversitesi/fakültesi/lisesi/hastanesi/belediyesi…" (ekli biçimler dahil: "boğaziçi üniversitesinde"); soldaki sözcüklerden en az biri bilinen ad, yer adı ya da sık geçen kurum adı olmalı; "çağrı merkezi", "vergi dairesi" gibi genel baş sözcükler küçük harfle kurum sayılmaz; "üniversitesi teknoloji fakültesi" tek kurum olarak birleşir.
- **İkamet bağlamı:** Sözcük anlamı da olan yer adları ("Bucak", "Fatih") küçük harfle, bitişik bulunma/ayrılma ekiyle ve ardından ikamet/okul fiiliyle ("bucakta okuyorum") yer adı sayılır; "bahçede oturuyorum", "kalede oturuyorum" gibi mekân kullanımları elenir. Bitişik ekli yer adında ek artık span dışında kalır ("bucakta" → "bucak"): dış servise "[YER_1]ta" gider, yanıt ekranda "bucak'ta" diye doğru çekimle geri çevrilir.

Doğrulama: 37.055 satırlık UCI ürün yorumlarında yeni kurallar 0 eşleşme üretti (ilk taslaktaki 15 "çağrı merkezi" yanlış alarmı baş sözcük listesi daraltılarak giderildi). Geliştirme setinde (realistic-v2) hibrit F1 değişmedi (0,9725). Mühürlü setlere bakılmadı.

## Yazım biçimi ölçümü ve günlük dil partisi (Sorun 2, adım 1 — 17 Ekim 2026)

Üretim boru hattı (v4 seed 42 + kurallar + sözlük) `training/ner/evaluate_casing.py` ile aynı cümlelerin orijinal, küçük ve BÜYÜK harfli kopyalarında ölçüldü. Mühürlü setler kullanılmadı. Sonuçlar: `reports/casing/`.

**realistic-v2 (batch1 + batch2, dev + test, 159 cümle; yapay zekâ destekli)**

| Yazım | Hibrit F1 | Yalnız model | Yalnız kural |
|---|---|---|---|
| Orijinal | 0,973 | 0,968 | 0,622 |
| küçük | 0,935 | 0,931 | 0,469 |
| BÜYÜK | 0,790 | 0,767 | 0,517 |

**realistic_batch3 (68 cümle, dev + test; yapay zekâ destekli, iyimser — aynı kaynak)**

| Yazım | Hibrit F1 | Adres | Yer | Kurum | Kişi |
|---|---|---|---|---|---|
| Orijinal | 0,851 | 0,667 | 0,899 | 0,743 | 0,907 |
| küçük | 0,838 | 0,667 | 0,870 | 0,743 | 0,896 |
| BÜYÜK | 0,681 | 0,667 | 0,697 | 0,667 | 0,680 |

**Kaynak.** batch3 yapay zekâ destekli üretildi; Irmak Yaprak gözden geçirdi ve etiketleri onayladı (17 Ekim 2026). augmented-v5 şablonları da aynı kaynaktan çıktığı ve batch3 kullanıcının gerçek örneğini birebir içerdiği için batch3 sonuçları **iyimser (aynı kaynak)** sayılır ve kanıt olarak kullanılmaz; asıl karşılaştırma batch1-2 (realistic-v2) ve bağımsız insan yazarlardan toplanacak sealed3 ile yapılır. Sızıntı denetimi: augmented-v4 train/dev ile birebir çakışan cümle yok.

**Hata türleri (batch3, orijinal yazım).** Sözcükler arası yanlış birleştirme ("ırmak bucakta" → YER, "kuzey fethiyede" → YER, "oğuz cumhuriyet mah. …" → ADRES); kurumun kişi + kurum diye bölünmesi ("mehmet akif" + "ersoy üniversitesinde", "süleyman" + "demirel üniversitesini"); "devlet hastanesine" gibi genel adların kurum sanılması; sözcük anlamı da olan adların kaçması ya da yanlış türe girmesi ("kale" → ORG, "kalede" → KİŞİ). Kurum alt birimi farkları ("ege üniversitesi tıp fakültesi"): gözden geçirmede kural korundu, küçük harfli alt birim kuruma dahil edilmez; modelin alt birimi katması bu partide hata sayılır.

## augmented-v5 verisi (Sorun 2, adım 2 — 17 Ekim 2026; eğitim bekliyor)

augmented-v4 komutuna `--v5-informal 8000 600` eklendi (`training/ner/context_templates_v5.py`). v1–v4 satırları ayrı RNG ile aynen üretilir; yeni satırlar 8.000 günlük dil cümlesi ve 536 zor negatiftir. Toplam: train 76.233, dev 5.545 (test v4 ile aynı). SHA-256 train `7e550fe0…`, dev `031f3c7e…`; Colab paketi `turkce-dlp-colab-bundle-v7`, defter `training/ner/colab_v5.ipynb`.

Hedeflenen hata türleri (batch3 ölçümünden; cümleler kopyalanmadı):
- "selam ben <ad> <yer>de yaşıyorum <kurum>de okuyorum" tanıtımları ve yapay zekâya istekler
- kişi adı taşıyan kurumlar ("necmettin erbakan üniversitesi", "<kişi> anadolu lisesi")
- sözcük anlamı da olan 27 ad ("nur", "bulut") ve 55 ilçe ("kemer", "maden"); aynı sözcüklerin sıradan kullanımı zor negatif
- yazım: küçük harf + bitişik ek %35, karışık harf (yalnız varlıkların ilk harfi büyük; yeni `mixed` varyantı) %15, BÜYÜK %12

Sızıntı önlemleri: geliştirme partilerinde (realistic_batch1-3) geçen adlar (ilk sözcük), ilçeler ve kurumlar (önek eşleşmesi) v5 yuvalarında kullanılmaz; 2 üniversite bu yüzden elendi. Altın set dışlama listesine `sealed2/*.txt` etiketli dosyaları da eklendi (önceden yalnız `sealed/` dışlanıyordu; `kaynak/` okunmaz, içerik yalnız dışlama için kullanılır, ayar yapılmadı). Cümle düzeyinde çakışma: 0.

**Dikkat:** batch3 ve v5 şablonları aynı kaynaktan (yapay zekâ destekli) çıktı; batch3'teki iyileşme **iyimser (aynı kaynak)** etiketiyle raporlanır, kanıt sayılmaz. Asıl karşılaştırma ve gerileme kontrolü batch1-2 (realistic-v2) üzerinden; v5'in tek bağımsız değerlendirmesi sealed3 ile bir kez yapılır.

## Not: mühürlü dosya adı (17 Ekim 2026)

Mühürlü final testteki yapay zekâ destekli parti `sealed_asistan.txt` adından `data/ner/manual/sealed/sealed_proje.txt` adına geçti. Kaynağı değişmedi: **yapay zekâ destekli üretildi; Irmak Yaprak gözden geçirdi ve etiketleri onayladı.** Dosyanın ilk yorum satırında "proje içi" yazar; mühürlü dosyanın içeriği okunmadığı ve değiştirilmediği için bu satır olduğu gibi bırakıldı, doğru kaynak bu rapordadır. Yalnız dosya adı ve ilk yorum satırı değişti; cümleler ve etiketler aynıdır. Bu nedenle dosyanın kaynak SHA-256 değeri ve sonuç dosyalarındaki parti adı, bu tarihten önceki raporlardakinden farklıdır; önceki sayılar geçerliliğini korur.

## Bağlamlı TC kuralı (17 Ekim 2026 — sonradan/post-hoc, gerçek kullanımda görüldü)

Gerçek kullanımda "selam ben atilla tc 11028365789" mesajındaki numara maskelenmedi: kontrol haneleri tutmadığı için TC sayılmıyordu. Yanlış yazılmış ya da uydurulmuş bir TC de kimliği büyük ölçüde ele verdiğinden, kontrol haneleri tutmayan 11 haneli sayı **hemen önünde TC ipucu varsa** ("tc", "t.c.", "tckn", "tcm", "kimlik no", "TC kimlik numaram") yine TC olarak maskelenir. Bağlamsız 11 haneli sayılar eskisi gibi kontrol hanesi ister; "sipariş no", "fatura" gibi bağlamlar etkilenmez.

Doğrulama: 37.056 satırlık UCI ürün yorumlarında yeni kural 0 eşleşme üretti; kaçınma ölçümü (`evaluate_bypass.py`) bağlamsız sayılarla çalıştığı için değişmedi. Gerçek ChatGPT'de (oturumsuz, istekler iptal edildi) uydurma geçersiz bir numarayla ağa yalnız `[TCKN_1]` gitti. Mühürlü setlere bakılmadı.

## Model süreci dayanıklılığı (17 Ekim 2026 — gerçek kullanımda görüldü)

Servis günlüğünde "NER tanıyıcısı başarısız" kaydı 6 kez geçiyordu: bilgisayar yoğunken tek bir mesaj 2 saniyelik çıkarım sınırını aşınca model süreci kapatılıyor ve **servis yeniden başlatılana kadar** kişi/kurum/yer tanıma sessizce devre dışı kalıyordu (yalnız kurallar çalışıyordu; kullanıcı fark etmiyordu). Düzeltme:

- Zaman aşımı ya da çökme sonrası süreç arka planda artan aralıklarla (5 sn, 15 sn, 1 dk, 5 dk) yeniden başlatılır; gerçek modelde süreç zorla öldürüldüğünde ~8 saniyede geri geldi.
- `/v1/detect` ve `/health` model durumunu bildirir (`ner`); model çalışmıyorsa eklenti, kişisel veri bulunmasa bile önizlemeyi "Ad ve adres tanıma şu an çalışmıyor" uyarısıyla açar ve gönderim kullanıcı onayına bağlanır; popup da durumu gösterir.
- Gerçek ChatGPT doğrulamasında bulunan yan hata da giderildi: site, eklentinin tekrar gönderiminden hemen sonra ikinci bir `submit` tetikliyor ve aynı metin yeniden kontrol ediliyordu (yanlış "Kişisel veri bulunmadı" bandı; model kapalıyken önizleme döngüsü). Az önce onaylanan metin 5 saniye içinde yeniden kontrol edilmez.

Model ve eşikler değişmedi; ölçüm sonuçları etkilenmez.
