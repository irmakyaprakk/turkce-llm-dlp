# Veri zenginleştirme (augmented-v1)

## Sorun

Baseline eğitim verisinin miktarı yeterli, ancak kullanıcıların LLM araçlarına yazdığı dili temsil etmiyor:

| Veri | Gözlem |
|---|---|
| WikiNER (17.967 cümle) | Varlık token'larının yalnızca %4,7'si küçük harfle başlıyor; tamamı resmi Wikipedia dili. |
| OSM adresleri (90.077) | Tek biçim: kayıtların %100'ünde `No:`, %96'sında `Mahallesi` var; daire/kat bilgisi %0,2. |
| Cümle içi adresler | Yalnızca 8 sabit kalıp. |

Bu nedenle model küçük harfli, Türkçe karaktersiz veya kesme işaretsiz yazılmış adları ve farklı biçimlerdeki adresleri kaçırma riski taşır.

## Yöntem

Kod: `training/ner/augment.py`, `training/ner/context_templates.py`, `training/ner/prepare_augmented.py`.

1. **Yazım varyantları (etiket korumalı).** WikiNER'den 6.000 cümle şu dönüşümlerle çoğaltılır: Türkçe kurallı küçük harf (İ→i, I→ı), Türkçe karakter kaldırma (ş→s), kesme işareti silme (`Kadıköy'de` → `Kadıköyde`) ve büyük harf. Dönüşümler karakter offset'i düzeyinde yapılır ve span'lar otomatik yeniden hesaplanır; etiketler elle kaydırılmaz.
2. **Adres biçim çeşitlendirme.** OSM kayıtları bileşenlerine ayrılır (mahalle, sokak, numara, posta kodu). Posta kodunun ilk iki hanesinden il adı türetilir. Her adres rastgele bir biçimde yeniden yazılır: `Mah./Mahallesi/mah`, `Sokak/Sk./sk`, `No:5/no 5/Numara 5`, isteğe bağlı sentetik daire/kat ve il. Numarasız sokak ve mahalle adları anotasyon kuralına uygun olarak `LOCATION` etiketlenir.
3. **Bağlam kalıpları.** 8 kalıp yerine 96 eğitim + 20 geliştirme kalıbı kullanılır: e-posta, Slack, destek talebi, resmi yazı, imza, form, kod yorumu, toplantı ve gündelik mesaj. Train ve dev kalıp aileleri ayrıktır. Özel isimlere gelen ekler ünlü uyumu ve kaynaştırma kurallarıyla üretilir (`İzmir'de`, `Ayşe'ye`, `Moda Mahallesi'nde`).
4. **Zor negatifler.** Varlık içermeyen ama "No:", "Mahalle", "Barış", "Umut" gibi yanıltıcı ifadeler taşıyan 1.200 cümle.

## Değerlendirme bütünlüğü

- Test split'i baseline ile aynıdır (`real-address-v2/test.jsonl`); sonuçlar doğrudan karşılaştırılabilir.
- Altın test seti (`data/ner/manual/realistic_*.txt`) eğitime girmez. Altın setteki kişi ve kurum adları dolgu sözlüklerinden çıkarılır; cümle veya adres çakışması olursa üretim durur. `build_realistic_set.py --against` ile bağımsız olarak da denetlenir (0 çakışma).
- Kalıplar yapaydır; gerçek kullanıcı başarısı yalnızca altın test setiyle raporlanır.

## Bilinen sınırlar

- Yer adları (il/ilçe) kapalı bir küme olduğundan altın setle örtüşebilir; kişi ve kurum adları örtüşmez.
- Kurum adlarının bir kısmı ad + sektör + şirket türü kalıbıyla üretilir; tanınmış kurumlar örneklemede %40 pay alır.
- Daire/kat bilgileri sentetiktir.

## augmented-v4 — gündelik dil turu (son eğitim turu)

Mühürlü testin bağımsız yazar grubunda ölçülen zayıflığı (F1 0,622; yer kapsaması 0,417) hedefler. v3'ün tamamı aynen korunur
(aynı tohumla aynı satırlar; doğrulandı), üzerine ayrı RNG ile 12.000 eğitim + 800 geliştirme cümlesi eklenir
(`context_templates_v4.py`, `prepare_augmented.py --v4-informal 12000 800`):

- yerel işletme adları (ad + tür, %30 yazım hatalı tür: "Rstoran", "Pastahanesi"), işletmenin yalnız adı, markalar
  (yabancı markada ek okunuşa göre: "Apple'da");
- PTT listesindeki ilçeler, bitişik ekli ("-daki/-deki" dahil), kişi adıyla çakışan ilçeler hariç;
- numarasız mahalle (LOC), çoğul ekli kişi adı ("Aylinlere"), genişletilmiş ad/soyad listesi;
- yer adıyla aynı yazılan sözcüklerle zor negatifler ("kulpu", "pazar günü", "kale");
- yazım varyantları ağırlıkla küçük harf + kesme işaretsiz; cümlelerin %25'inde tek harf düşürme.

Toplam: 67.684 eğitim cümlesi. Eski mühürlü test cümleleri daha önce görüldüğü için onlara benzeyen çerçeveler bilinçli olarak
çıkarıldı. **Veri, yeni mühürlü set (sealed2) görülmeden önce donduruldu**; SHA-256 (train) `e91b9551…f0750ec`, (dev) `a51a72a4…76b2e5`.
Colab'da yeniden üretilen dosya bu özetlerle doğrulanır (`colab_v4.ipynb`).
