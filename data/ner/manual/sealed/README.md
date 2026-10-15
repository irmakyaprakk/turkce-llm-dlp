# Mühürlü final test

Bu klasördeki `*.txt` dosyaları yalnızca **final değerlendirmede bir kez** kullanılır
(`python training/ner/build_gold_set.py --sealed --out-dir data/ner/processed/final-test`).

- Hata analizi, eşik/kural ayarı veya model seçimi bu cümlelere bakılarak yapılmaz.
- Eğitim verisi üretimi (`prepare_augmented.py`) bu dosyalardaki cümle, adres, kişi ve kurum adlarını engeller.
- Her dosya bir yazar grubudur (ör. `sealed_proje.txt`, `sealed_irmak.txt`); sonuçlar dosya bazında ayrıca raporlanır.
