# Mühürlü test 3 (sealed3): toplama talimatı

Amaç: augmented-v5 modelini, eğitim şablonlarından ve geliştirme setinden bağımsız, gerçek kişilerin yazdığı mesajlarla **bir kez** ölçmek. Geliştirme setleri (realistic_batch1-3) yapay zekâ destekli üretildiği için v5'in gerçek kazancını yalnız bu set gösterebilir.

## Yazarlara gönderilecek metin

> Bir kişisel veri koruma projesi için senden 5–10 kısa mesaj istiyorum. ChatGPT'ye ya da başka bir yapay zekâya bir şey sorar, bir iş yaptırır gibi yaz: kendini tanıt, bir dilekçe, CV, mail ya da mesaj yazdır, bir adrese kargo gönder, birini sor…
>
> - Nasıl yazıyorsan öyle yaz: küçük harf, kısaltma, yazım hatası, kesme işaretsiz ek ("ankarada", "ahmete") serbest.
> - Mesajlarda ad soyad, okul/şirket/hastane adı, şehir/ilçe ve açık adres geçsin; ama **hepsi uydurma** olsun. Kendi ya da tanıdıklarının gerçek bilgilerini yazma.
> - TC kimlik no, telefon, IBAN gibi numaralar yazacaksan rastgele uydur.
> - Mesajları bir metin dosyasına yaz ve aralarına **bir boş satır** bırak. Bir mesaj birden çok satır olabilir.
> - Bu mesajları başka bir yerde (yapay zekâ dahil) üretme ya da düzelttirme; kendi yazımın olsun.

Hedef: 8–12 yazardan toplam 50–100 mesaj. Bir yazarın payı %20'yi geçmesin (tek üslup baskın olmasın).

## Irmak'ın akışı

1. **Topla:** Her yazarın dosyasını `data/ner/manual/sealed3/kaynak/` altına koy (dosya adı serbest; bu klasör `.gitignore`'da, GitHub'a gitmez).
2. **Taslak:** `python training/ner/prepare_sealed3.py taslak` — mesajlar `sealed3/sealed3_toplanan.txt` dosyasında `yazarN<TAB>mesaj` satırlarına dönüşür; yazar adları gizlenir, tekrarlar atılır. Komut yalnız sayı yazdırır.
3. **Etiketle:** Dosyayı `training/ner/ANNOTATION_RULES.md` kurallarıyla elle etiketle (`[[PER|…]]`, `[[ORG|…]]`, `[[LOC|…]]`, `[[ADRES|…]]`). Model ya da kural ön-etiketlemesi **kullanma**: değerlendirilecek sistemin hataları sete taşınmasın. Mesaj metnine dokunma; yalnız işaret ekle.
4. **Mühürle:**
   ```bash
   python training/ner/prepare_sealed3.py muhurle --against data/ner/processed/augmented-v5/train.jsonl data/ner/processed/augmented-v5/dev.jsonl
   ```
   Biçim denetlenir, eğitim verisiyle cümle çakışması varsa mühürlenmez; SHA-256 `sealed3/MANIFEST.json` dosyasına yazılır. `sealed3_toplanan.txt` ve `MANIFEST.json` dosyalarını commit'le. Bundan sonra dosyaya bakılarak hiçbir ayar yapılmaz.
5. **Ölç (bir kez):** Colab'da önce `python training/ner/prepare_sealed3.py dogrula`, sonra
   `python training/ner/build_gold_set.py --sealed --sealed-dir sealed3 --out-dir data/ner/processed/sealed3-test` ve v4 / v5 için `evaluate_casing.py` (ya da `evaluate_model.py` + `evaluate_rules.py`). Sonuç, v5'in resmî değerlendirmesidir; sonrasında yapılan her değişiklik raporda "sonradan (post-hoc)" etiketlenir.

## Notlar

- augmented-v5 verisi sealed3 toplanmadan önce donduruldu (SHA-256 train `7e550fe0…`); sealed3 cümleleri eğitim verisine giremez. `muhurle` adımı yine de çakışmayı denetler.
- Karar kuralı ölçümden **önce** yazılır: "v5, sealed3 hibrit F1'de v4'ten iyiyse ve realistic-v2'de 0,01'den fazla gerilemiyorsa üretimde v5."
  **Onay:** Irmak Yaprak, 17 Ekim 2026 — sealed3 toplanmadan ve ölçümden önce onaylandı. Bu kural ölçümden sonra değiştirilmez.
