# Türkiye il / ilçe / mahalle listesi

- Paket: `turkey-neighbourhoods` 4.0.3 (npm), https://github.com/muratgozel/turkey-neighbourhoods
- Paket lisansı: MIT (© 2020 Murat Gözel) — bkz. `LICENSE`
- Paketin veri kaynağı: PTT posta kodu listesi (https://postakodu.ptt.gov.tr/Dosyalar/pk_list.zip), kamuya açık resmî liste
- İndirme tarihi: 2026-10-16 (`npm pack turkey-neighbourhoods@4.0.3`)

Dosyalar (paketin `src/data/` klasöründen değiştirilmeden alındı):

- `cityNames.json`: 81 il
- `districtsByCityCode.json`: 973 ilçe (898 benzersiz ad)
- `neighbourhoodsByDistrictAndCityCode.json.gz`: 73.305 mahalle kaydı (gzip ile sıkıştırıldı)

Kullanım: `training/ner/build_gazetteer.py` bu dosyalardan ve WikiNER eğitim bölümünden
(belirsiz sözcük istatistikleri) `src/turkce_llm_dlp/data/tr_gazetteer.json.gz` dosyasını üretir.
Altın geliştirme seti ve mühürlü test bu üretimde kullanılmaz.
