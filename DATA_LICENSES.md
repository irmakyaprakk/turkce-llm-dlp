# Veri ve model lisansları

Bu depodaki **kaynak kod** MIT lisanslıdır (bkz. `LICENSE`). Aşağıdaki veri kümeleri ve önceden eğitilmiş model **depoya dahil değildir**; her biri kendi lisansına tabidir. Ham ve işlenmiş veriler `.gitignore` ile dışarıda tutulur.

| Kaynak | Kullanım | Lisans | Atıf |
|---|---|---|---|
| [Turkish Wiki NER Dataset](https://github.com/turkish-nlp-suite/Turkish-Wiki-NER-Dataset) | PERSON / ORG / LOCATION eğitim ve test | CC BY-SA 4.0 | Altınok, D. (2023). *A Diverse Set of Freely Available Linguistic Resources for Turkish*. ACL 2023. https://aclanthology.org/2023.acl-long.768/ |
| [OpenStreetData – Türkiye](https://openstreetdata.org/) (OpenStreetMap türevi) | ADDRESS dizileri | ODbL 1.0 | © OpenStreetMap katkıcıları — https://www.openstreetmap.org/copyright |
| [Turkish User Reviews (UCI)](https://archive.ics.uci.edu/dataset/769/turkish%2Buser%2Breview%2Bdataset) | Etiketsiz yanlış alarm taraması | CC BY 4.0 | Ekinci, E. ve İlhan Omurca, S. DOI: https://doi.org/10.24432/C5HK72 |
| [BERTurk (`dbmdz/bert-base-turkish-cased`)](https://huggingface.co/dbmdz/bert-base-turkish-cased) | Başlangıç modeli | MIT | Schweter, S. (2020). BERTurk. Zenodo. |

## Eğitilmiş model ağırlıkları

İnce ayarlı NER modeli WikiNER (CC BY-SA 4.0, *ShareAlike*) ve OSM türevi (ODbL, *share-alike*) veriyle eğitildiği için model yayımlanırken bu kaynaklara atıf verilmeli ve model kartında lisans koşulları açıkça belirtilmelidir. Model ağırlıkları bu depoya eklenmez.

## Mahremiyet

- Gerçek kurum içi yazışma, log veya müşteri verisi eğitimde kullanılmaz.
- OSM adresleri gerçek konum kayıtlarıdır; örnek çıktılarda ve dokümantasyonda gösterilmez.
- Testlerdeki TCKN, IBAN ve kart numaraları algoritmik olarak üretilmiş sahte değerlerdir.
