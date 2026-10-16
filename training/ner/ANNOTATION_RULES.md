# Türkçe NER ve adres anotasyon kuralları

Bu kural seti elle yazılmış değerlendirme cümleleri ve sonraki anotasyonlar için geçerlidir. Aynı kararlar bütün veri bölmelerinde korunmalıdır.

## Satır içi işaretleme

Varlıklar `[[ETİKET|metin]]` biçiminde gösterilir. Kullanılan görünen etiketler `PER`, `ORG`, `LOC` ve `ADRES`tir. Dönüştürme sırasında bunlar sırasıyla model şemasındaki `PERSON`, `ORG`, `LOCATION` ve `ADDRESS` etiketlerine çevrilir.

Örnek: `[[LOC|Kadıköy]]'deki ofis`, `Kadıköy` sözcüğünü `LOCATION` olarak, kesme işaretinden sonrasını etiket dışı bırakır.

## Karar kuralları

| Durum | Etiketleme |
| --- | --- |
| Tek başına şehir, ilçe veya mahalle adı | `LOC` |
| Kapı numarası içermeyen cadde/sokak adı | `LOC` |
| Kapı numarası, daire, kat veya bina bilgisi olan açık adres | Tam adres ifadesini tek `ADRES` span'ı olarak işaretle |
| Açık adres span'ının içindeki ilçe/şehir/posta kodu | `ADRES` span'ına dahil et |
| `Adres:` gibi açıklayıcı giriş sözü | Span dışında bırak |
| Telefon ve e-posta | NER etiketleri dışında; kural katmanına bırak |
| Adresle aynı cümledeki kurum adı | Kurumu `ORG`, adresi ayrı `ADRES` span'ı yap; span'lar iç içe olmasın |
| Kurum adı + büyük harfle yazılmış alt birim ("Yıldız Teknik Üniversitesi Makine Fakültesi", "Kadıköy Belediyesi Fen İşleri Müdürlüğü") | Alt birim dahil tamamı tek `ORG` (alt birim de tanımlayıcıdır; tek placeholder ile maskelenir). Küçük harfle yazılmış genel ifade ("kardiyoloji polikliniğine") dahil edilmez |
| Kurum adının içindeki şehir/ilçe ("Antalya Kurumlar Vergi Dairesi", "Sosyal Güvenlik Kurumu Konya İl Müdürlüğü") | Ayrı `LOC` yapılmaz; tamamı tek `ORG` |
| `Dr.`, `Sayın`, `Bey` gibi hitap/unvan | Dışarıda bırak; yalnızca kişi adını `PER` işaretle |
| Numarasız tek başına bina / tesis adı: apartman, plaza, iş merkezi, iş hanı, pasaj, park, kampüs, AVM, rezidans ("Zümrüt Apartmanı", "Çınar Plaza", "Beşiktaş kampüsü") | Baş sözcük dahil `LOC` |
| Numaralı adresin hemen önündeki bina adı ("Gül Apartmanı, Kiraz Sokak No 21, Kayseri") | Adresin parçası; tamamı tek `ADRES` |
| İşletme / kurum (eczane, kafe, fırın, tıp merkezi, klinik, kütüphane, şirket) | `ORG` (bina değil kuruluştur) |
| Betimleyici ifade ("göl manzaralı kafe", "sağdaki mavi bina") | Etiketsiz |

Kesme işaretinden sonraki ek varlık span'ına girmez. Örneğin `[[LOC|Kadıköy]]'deki` ve `[[ADRES|... No:5]]'e`.

Numarasız cadde/sokak adının `LOC` sayılması bilinçli proje kararıdır: açık kapı/daire numarası bulunmayan ifade bu anotasyon standardında tek başına açık adres kabul edilmez. Bir adres içindeki şehir/ilçe adı ise bağımsız `LOC` olmaz; `ADRES` span'ının parçasıdır.

## Örnekler

- `Yarın [[PER|Elif Demir]]'e uğrayacağım, [[ADRES|Moda Mah. Bahariye Cad. No:12 D:4 Kadıköy]]'de oturuyor.`
- `Kargoyu [[ADRES|Cumhuriyet Mah. 45. Sk. No:7 K:2]]'ye gönderin lütfen.`
- `Sayın [[PER|Mehmet Kaya]], fatura [[ADRES|Atatürk Cad. No:15/3 06100 Çankaya/Ankara]] adresine gönderilmiştir.`
- `Toplantı [[LOC|Ankara]]'da, [[ORG|Örnek Teknoloji A.Ş.]] merkezinde olacak.`
- `[[LOC|Bursa]]'ya gidiyorum, cuma dönerim.`
- `[[LOC|Bağdat Caddesi]]'nde yürüyüş yaptık.`

Son iki örnek zor negatiflerdir: model bunları `ADRES` olarak etiketlememelidir.

## Model etiketlerine eşleme

| Elle anotasyon etiketi | Model etiketi |
| --- | --- |
| `PER` | `PERSON` |
| `ORG` | `ORG` |
| `LOC` | `LOCATION` |
| `ADRES` | `ADDRESS` |

Model BIO şemasında span'ın ilk token'ı `B-...`, devam token'ları `I-...` alır. Açık adresin içindeki yer adları ayrıca `LOCATION` olarak anotasyonlanmaz.
