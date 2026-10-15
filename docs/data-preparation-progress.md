# Veri hazırlığı notları

## Kaynak

- Veri kümesi: Turkish Wiki NER Dataset
- Kaynak depo: https://github.com/turkish-nlp-suite/Turkish-Wiki-NER-Dataset
- Depoda bildirilen lisans: CC BY-SA 4.0 (lisans metni ham veri klasöründe `LICENSE` dosyası olarak korunuyor)
- Depoda istenen atıf: Duygu Altınok, “A Diverse Set of Freely Available Linguistic Resources for Turkish”, ACL 2023, https://aclanthology.org/2023.acl-long.768/
- Depo, verinin Kuzgunlar NER veri setinden örneklenip yeniden etiketlendiğini belirtiyor. İlgili Mendeley kaydı CC BY 4.0 lisansı bildiriyor: https://data.mendeley.com/datasets/cdcztymf4k/1

## Yerel dosyalar

Ham veri yolu: `data/ner/raw/Turkish-Wiki-NER-Dataset/`

- `LICENSE`
- `README.md`
- `Turkish-Wiki-NER-Dataset/train.conll`
- `Turkish-Wiki-NER-Dataset/dev.conll`
- `Turkish-Wiki-NER-Dataset/test.conll`

Bu veri kümesinin README dosyası 18.000 train, 1.000 dev ve 1.000 test cümlesi bildiriyor. Projenin CoNLL ayrıştırma kuralıyla sayımda 17.967 train, 1.000 dev ve 1.000 test satırı elde edildi. 17.967, README'deki 18.000 değerinin yuvarlanmış gösterimiyle uyumlu; dosya biçiminde satır hatası görülmedi.

Kaynakta 19 varlık türü var. Proje şeması ilk aşamada `PERSON`, `ORG`, `GPE`/`LOC` birleşiminden `LOCATION` ve sentetik `ADDRESS` kullanıyor. Diğer kaynak etiketlerinin nasıl ele alınacağı dönüştürülmüş kalite raporunda ayrıca kontrol edilecek.

## İlk örnek kontrolü

CoNLL satırları token ve BIO etiketi olarak ayrılıyor; örnek satırlar bu yapıya uyuyor. `B-` bir varlığın ilk tokenını, `I-` aynı varlığın devamını, `O` ise etiketlenmeyen tokenı gösteriyor. Split dosyaları beklenen biçimde okunabildi.

## Hazırlanan veri

`training/ner/prepare_data.py` ile `data/ner/processed/` altında JSONL split'leri ve `quality_report.json` oluşturuldu. Her split'e 200 sentetik adres cümlesi eklendi: train 18.167, dev 1.200, test 1.200 satır. Hazırlama sırasında BIO etiketleri, izinli etiketler ve split'ler arası bire bir tekrar kontrol edildi. İlk doğrulama çağrısı import yolu nedeniyle hata verdi; `train.py` doğrudan çalıştırıldığında proje paketini bulacak şekilde düzeltildi ve `train.py --validate-only` başarıyla çalıştı.

## Gerçek adres kaynağı

Kullanıcının gerçek adreslerle çalışma isteği üzerine 2026-10-12 tarihinde OpenStreetData'nın Türkiye OSM dışa aktarımları indirildi:

- `data/ner/raw/openstreetmap-tr/TR-houses.tsv.gz`: 97.851 kayıt; posta kodu, şehir, sokak, kapı numarası ve koordinat alanları.
- `data/ner/raw/openstreetmap-tr/TR-addresses.tsv.gz`: 8.135 kayıt; şehir/sokak ve numara aralığı alanları.
- Kaynak: https://openstreetdata.org/ ve https://github.com/dagnelies/open-street-data
- Veri OpenStreetMap'ten türetilmiş ve ODbL ile lisanslanmış. Atıf ve aynı lisansla paylaşım yükümlülüklerini proje dışında dağıtım öncesi ayrıca incele.
- OSM adresleri gerçek konum kayıtlarıdır; eksiksiz/temsil edici olduğu garanti edilmez. Ham dosyalar git'e alınmamalı ve gerçek adresleri sohbet çıktılarında örnek olarak göstermemeli.
- `training/ner/prepare_osm_addresses.py` ile 90.077 tekil adres kaydı `data/ner/processed/openstreetmap-tr/addresses.jsonl` dosyasına çevrildi. Şehir bilgisi eksik 4.578 kayıt çıkarıldı, 3.196 yinelenen kayıt çıkarıldı; posta kodu isteğe bağlı olduğu için eksik olanlar tutuldu. Bu dosya pozitif adres örneklerinden oluşuyor ve cümle bağlamı taşımıyor; WikiNER train/dev/test dosyalarına henüz eklenmedi.

Kullanıcı yalnızca kamuya açık ve kullanım koşulları belirtilmiş kaynaklarla ilerlemek istedi. Kurum içi gerçek şirket/müşteri yazışmaları kullanılmıyor. OSM kayıtları adres içerir; müşteri dili için ayrı, lisansı belirtilmiş UCI yorum kaynağı kullanılıyor.

## Gerçek müşteri dili kaynağı

UCI Machine Learning Repository'den `Turkish User Reviews` veri kümesi indirildi: `data/ner/raw/uci-turkish-user-reviews/turkish-user-review-dataset.zip`. UCI kaynağı 37.056 Türkçe ürün yorumu bildiriyor; lisans CC BY 4.0 ve DOI `10.24432/C5HK72`. Arşivde 37.056 satır sonu, 12 boş satır ve 37.044 dolu satır var; dolu satırların 3.053'ü tekrarlanıyor. Yorumlar NER etiketi taşımıyor. Otomatik taramada 114 dolu satır adres sözcüğü/numara örüntüsü, bir satır 11 basamaklı sayı örüntüsü içeriyordu; bunlar doğrulanmış kişisel veri veya adres sayılmaz, yalnızca elle inceleme adayıdır. Ham yorum örnekleri sohbette gösterilmedi.

Kullanım koşulları ve atıflar için UCI dosyası `data/ner/raw/uci-turkish-user-reviews/PROVENANCE.md` içinde kaydedildi. OSM kaynağının ODbL atıf bilgisi `data/ner/raw/openstreetmap-tr/PROVENANCE.md` içinde.

## Sıradaki adım

Etiketleri proje şemasına dönüştürüp yapay adres örnekleri eklemek; ardından kalite raporunu ve sentetik örnekleri gözden geçirmek. Ham veri ve lisans dosyaları repoya eklenmemeli.

## Yorumlarda adres adayı incelemesi

UCI yorumlarında adresle ilgili sözcük veya 11 basamaklı sayı örüntüsü bulunan 123 satır, tekrarları birleştirerek 111 tekil yoruma indirildi ve ayrı bir inceleme kuyruğuna alındı: `data/ner/processed/uci-turkish-user-reviews/address-candidates.jsonl`. Satırlar eğitim verisine eklenmedi; `manual_label: REVIEW` ve boş `address_spans` alanları insan kontrolü bekliyor. Kuyruğa kopyalanırken e-posta, Türkiye cep telefonu biçimi ve 11 basamaklı sayı örüntüsü maskelendi. Bu örüntü taraması adres doğrulaması değildir.

İnceleme sırasında normalleştirilmiş OSM adresleriyle tam eşleşme bulunmadı. Bu, yorumlardaki adayların yanlış olduğu anlamına gelmez; adresler farklı yazılmış veya genel adres sözcükleri kullanılmış olabilir. Sonraki adım örnekleri elle etiketlemek, sonra yalnızca doğrulanan etiketli satırları eğitim verisine katmaktır.

## Gerçek adresleri eğitim biçimine hazırlama

`training/ner/prepare_real_address_training.py`, WikiNER train/dev/test split'lerini koruyup önceki sentetik adres satırlarını çıkarır; her split'e sabit seed ile seçilmiş 200 gerçek OpenStreetData adresi ekler. Yeni, ayrı çıktı yolu `data/ner/processed/real-address-v1/` olduğundan önceki `data/ner/processed/` dosyaları değişmeden kalır. Yeni sette toplam 600 OSM adresi vardır; 90.077 adreslik ana OSM listesi ayrıca korunur.

OSM örnekleri tek başına adres dizileridir; cümle bağlamı eklenmedi. Üç split BIO etiket ve splitler arası kopya denetiminden geçti. `python training/ner/train.py --data-dir data/ner/processed/real-address-v1 --validate-only` başarılı oldu. Bu veri, gerçek adres biçimini öğretmek için ilk denemedir; adreslerin doğal metin/cümle içindeki kullanımını tek başına ölçemez. Eğitim sonrası raporda OSM adres test sonuçları ayrı gösterilecek.

## Gerçek adres veri miktarını büyütme

İlk 600 adreslik `real-address-v1` pilotu korunarak `data/ner/processed/real-address-v2/` hazırlandı. Kaynaktaki 90.077 satırdan aynı adres metnini veren 10 tekrar çıkarıldı; kalan 90.067 kayıt `data/ner/processed/openstreetmap-tr/splits/` altında train/dev/test'e 72.053/9.007/9.007 olarak ayrıldı. Birleşik `real-address-v2` sürümü bu ayrımlardan train/dev/test için 6.000/600/600 adres seçiyor; seçilmeyen adresler ana OSM dosyasında korunuyor. Sentetik adresler çıkarıldı; WikiNER satırları korundu. OSM tokenlarının birleşik train içindeki oranı yaklaşık %16; adres örnekleri baskın hale gelmiyor.

BIO/split doğrulaması ve `python training/ner/train.py --data-dir data/ner/processed/real-address-v2 --validate-only` başarılı oldu. Yeni veri seti yapısal olarak eğitime hazır. Gerçek cümle bağlamında adres performansını ölçmek için cümle içinde adres bulunan, kullanım koşulları uygun gerçek metin örnekleri ayrıca gerekli.

## Gerçek adres kayıtlarıyla cümle dosyası

İstek üzerine tüm 90.067 tekil OSM adres kaydı sekiz sabit Türkçe cümle kalıbından biri içine yerleştirildi. Sonuçlar `data/ner/processed/openstreetmap-tr/sentence-context-v1/` altında train/dev/test olarak duruyor. Adres dizisi gerçek OSM kaydıdır; çevresindeki cümle kalıbı yapaydır. Bu nedenle set `context_is_synthetic: true` olarak işaretlendi; doğal kullanıcı metni/evaluation verisi değildir ve ana `real-address-v2` eğitim setine otomatik eklenmedi. BIO/split doğrulaması hazırlama sırasında başarılı oldu.

## Colab eğitim hazırlığı

Yerel Mac'te eğitim paketlerinin (özellikle PyTorch) indirilmesi yavaş olduğu için kurulum durduruldu; eğitim Colab GPU'da çalıştırılacak. Kullanıcının hibrit mimari değerlendirmesi doğrultusunda ilk model yalnızca kişi/kurum/yer tanıyacak. `data/ner/processed/person-org-location-v1/` WikiNER split'lerinden 17.967/1.000/1.000 etiketli cümle içeriyor; açık adres ve sentetik cümle örnekleri bu modele dahil değil. `training/ner/colab_01_person_org_location.ipynb` yükleme, GPU kontrolü, veri doğrulama, Drive'a model kaydetme ve 3 epoch ilk eğitim adımlarını içeriyor. `data/ner/processed/person-org-location-colab.zip` Colab'a yüklenecek kod/veri paketidir. Model manifestinde kullanılan veri kaynağı ve lisansı kaydedilir.


## İlk Colab eğitimi tamamlandı (2026-10-13)

Colab'ın Python 3.12 T4 GPU çalışma ortamında BERTurk, WikiNER kişi/kurum/yer verisi üzerinde 3 epoch ve 6.738 adım eğitildi. Doğrulama genel span F1: 0,852; ayrılmış test genel span F1: 0,872. Testte kişi F1: 0,922; kurum F1: 0,767; yer F1: 0,857. Model, test ve geliştirme metrikleriyle `/content/ner-person-org-location-model/` içine kaydedildi. Bu sonuç WikiNER test setini ölçer; kullanıcı mesajı veya adres performansı değildir.

Model klasörü Colab çalışma alanında geçicidir. Drive yetkilendirmesi yapılmadı. Model ve metrik dosyalarından oluşan 441 MB'lık ZIP arşivi bilgisayara indiriliyor.
