"""augmented-v3 için ek cümle çerçeveleri (sözdizimi context_templates.py ile aynı).

v1/v2 kalıplarından farklı olarak: çok cümleli ve uzun metinler, yazım hataları,
İngilizce-Türkçe karışık iş dili, kurum ve adres yoğun çerçeveler, numarasız adres
tarifleri (LOCATION/O) ve daha fazla zor negatif. TRAIN ve DEV aileleri ayrıktır;
geliştirme ve mühürlü altın setle cümle düzeyinde çakışma üretim betiğinde denetlenir.
"""

from __future__ import annotations

TRAIN_TEMPLATES_V3 = (
    # --- uzun, çok cümleli iş e-postaları ---
    "Merhaba {FIRST},\nDünkü görüşmede konuştuğumuz gibi {ORG} ile sözleşme taslağını revize ettik. Islak imzalı nüshayı {ADDR} adresine kargolayacağız. Bir sorun olursa {PER} ile iletişime geçebilirsin.\nKolay gelsin",
    "Sayın yetkili,\n{ORG} bünyesinde çalışan {PER} hakkında referans talebinizi aldık. Kendisi {DATE} tarihinden bu yana {LOC} ofisimizde görev yapmaktadır.\nSaygılarımızla",
    "{FIRST} Hanım merhaba, fatura adresimiz değişti. Eski adres {ADDR} idi, yeni adresimiz {ADDR}. Muhasebe kayıtlarını buna göre güncelleyebilir misiniz?",
    "Ekip, kısa bir hatırlatma: yarın {ORG} denetimi var. {PER} ve {PER} sabah 9'da {LOC} şubesinde olacak. Evraklar hazır olsun lütfen.",
    "Merhaba,\nGeçen ay {ORG:DAN} aldığımız teklif hâlâ geçerli mi? Teslimat {LOC:E} yapılacak, adres {ADDR}. Yetkili kişi {PER}.\nTeşekkürler",
    "Selam {FIRST}, {ORG} tarafı ödemeyi yarın yapacakmış. {PER} bilgi verdi, sen de {ORG} hesabını kontrol eder misin?",
    "{FIRST} bey merhaba,\nNumune paketini {ADDR} adresine gönderdik, takip numarası {TRACK}. Teslim alan kişi {PER} olarak görünüyor.",
    "İyi günler, {ORG} adına yazıyorum. {DATE} tarihli faturanızda bir hata var; alıcı {PER} değil {PER} olmalı.",
    "Merhaba {FIRST},\nStaj başvurun için {ORG} insan kaynaklarından {PER} sana dönüş yapacak. Mülakat {LOC} kampüsünde.",
    "Değerli iş ortağımız,\n{ORG} olarak {DATE} itibarıyla yeni adresimize taşındığımızı bildiririz: {ADDR}.",
    # --- müşteri destek / şikâyet ---
    "Siparişim {TRACK} numarasıyla geliyor ama kurye {ADDR} adresini bulamadığını söylüyor. Lütfen {PER} adına not düşün.",
    "{ORG} müşteri hizmetlerine üç kez ulaştım, kimse dönmedi. Adım {PER}, abone numaram {NUM}.",
    "Ürünü {ORG} {LOC} mağazasından aldım, fiş {PER} adına. Değişim yapabilir miyim?",
    "teslimat adresimi {ADDR} olarak güncellemek istiyorum, eski adrese gitmesin",
    "Kargom {LOC} aktarmada 5 gündür bekliyor. Alıcı {PER}, adres {ADDR}.",
    "{ORG} uygulamasında kartım bloke oldu, {PER} adına kayıtlı hesap.",
    "Merhaba, annem {PER} adına {ORG:DA} açılmış bir hesap var mı öğrenebilir miyim?",
    "İade kargosunu {ORG} şubesine bıraktım, gönderici {PER}, çıkış {LOC}.",
    # --- WhatsApp / gündelik, yazım hatalı ---
    "{FIRST} napıyosun akşam {LOC:DA} mıyız",
    "abi adres {ADDR} zile bas açmazsa ara beni",
    "{FIRST} hocanın ofisi {ORG} binasında 3. katta",
    "{FIRST:E} söyle yarın {ORG:E} gidicez sabah erken",
    "kargo geldi mi {FIRST} senin adrese {ADDR} yazmıştım",
    "{FIRST} {LOC:DAN} döndü mü ya haber alamadım",
    "bu hafta sonu {LOC:E} gidiyoz {FIRST} da geliyo",
    "{FIRST} teyze {LOC:DA} oturuyo artık adres {ADDR}",
    "yarın {ORG:DA} mülakatım var dua edin",
    "{PER} diye biri aradı, {ORG:DAN} arıyorum dedi",
    "tmm {LOC:DAN} çıktım {LOC:E} 20 dk",
    "{FIRST} {ORG} kartı mı kullanıyosun yoksa {ORG} mı",
    "ya {FIRST} adresi atsana {LOC:DA} neredeydi",
    "{FIRST:IN} düğünü {LOC:DA} olcakmış",
    # --- İngilizce-Türkçe karışık iş dili ---
    "Hi {FIRST}, {ORG} meeting'i perşembeye aldık, {PER} sunum yapacak.",
    "FYI: {ORG} contract signed, invoice adresi {ADDR}.",
    "Can you ping {PER}? {ORG} tarafında blocker var.",
    "{FIRST} OOO bu hafta, urgent işler için {PER:E} yazın.",
    "Shipping address: {ADDR}. Contact person: {PER}.",
    "Quick update: {LOC} ofisi {DATE} itibarıyla remote, {PER} koordine ediyor.",
    "Client {ORG} wants the report asap, {LOC} ekibi destek verecek.",
    "Deck'i {PER:E} attım, {ORG} feedback'i bekliyoruz.",
    # --- resmi / hukuk / sağlık ---
    "Müvekkilimiz {PER} adına {ORG} nezdinde başlatılan icra takibine itiraz ediyoruz.",
    "Hasta {PER}, {ORG} acil servisinden {LOC} ilindeki tam teşekküllü hastaneye sevk edilmiştir.",
    "Kiracı {PER}, {ADDR} adresindeki taşınmazı {DATE} tarihinde tahliye edecektir.",
    "{ORG} tarafından düzenlenen {DATE} tarihli tutanak ekte sunulmuştur.",
    "Davalı {PER} vekili Av. {PER} duruşmaya katılmadı.",
    "Tebligat {ADDR} adresine yapılamamış, muhatap {PER} adreste bulunamamıştır.",
    "{ORG} kayıtlarına göre taşınmaz {LOC} ilçesinde, {MAH} sınırları içindedir.",
    "Sigortalı {PER} için {ORG} tarafından provizyon onayı verilmiştir.",
    # --- form / CV / liste ---
    "Ad Soyad: {PER}\nKurum: {ORG}\nİl: {LOC}\nAdres: {ADDR}",
    "Deneyim:\n- {ORG}, Yazılım Uzmanı\n- {ORG}, Stajyer\nEğitim: {ORG}",
    "Referanslar: {PER} ({ORG}), {PER} ({ORG})",
    "Gönderici: {PER}, {ADDR}\nAlıcı: {PER}, {ADDR}",
    "Şube listesi:\n{LOC} – {ADDR}\n{LOC} – {ADDR}",
    "Veli: {PER} | Öğrenci okulu: {ORG} | Adres: {ADDR}",
    "Katılımcılar: {PER} ({ORG}), {PER} ({ORG}), {PER}",
    "Merkez: {ADDR}; Depo: {ADDR}",
    # --- kod / log / teknik ---
    'logger.info("Kullanıcı {PER} giriş yaptı, şube: {LOC}")',
    "INSERT INTO musteri (ad, adres) VALUES ('{PER}', '{ADDR}');",
    "# TODO({FIRST}): {ORG} webhook imzası doğrulanmıyor",
    "// {FIRST} ekledi: {ORG} sandbox anahtarı burada değil, kasada",
    '{{"customer": "{PER}", "company": "{ORG}", "city": "{LOC}"}}',
    "ERROR 2026-10-12 {TIME} {ORG} API zaman aşımı (kullanıcı={PER})",
    # --- adres tarifi ve numarasız yer (LOCATION / O) ---
    "{STREET} üzerindeki eczanenin üstü, zile basmayın arayın.",
    "{MAH:DA} oturuyorum ama tam adresi sonra atarım.",
    "{LOC} çarşısından girip {STREET} yönüne dönün.",
    "{STREET:DA} trafik kilitlendi, geç kalacağım.",
    "{MAH} muhtarlığından ikametgah aldım.",
    # --- yer yoğun ---
    "{LOC:DAN} {LOC:E} giden otobüs iptal oldu, {LOC} aktarmalı gideceğim.",
    "Yeni bölge müdürlüğü {LOC:DA}, {LOC} ve {LOC} illerine bakacak.",
    "{ORG} {LOC:DA} ikinci fabrikasını açtı.",
)

DEV_TEMPLATES_V3 = (
    "Merhaba {FIRST},\nToplantı notlarını {ORG} paylaşım klasörüne koydum. Kargo {ADDR} adresine çıktı.\nİyi çalışmalar",
    "{ORG} müşteri temsilcisi {PER} aradı, adres değişikliği istiyor: {ADDR}",
    "{FIRST} akşam {LOC:DA} buluşalım mı",
    "Hi {FIRST}, invoice'u {ORG} adına kes, adres {ADDR}.",
    "Tebligat adresi {ADDR}, muhatap {PER}.",
    "Ad: {PER}\nŞirket: {ORG}\nŞehir: {LOC}",
    'logger.warn("{ORG} ödeme reddedildi: {PER}")',
    "{STREET} kapalı, {MAH} tarafından dolaşın.",
    "{PER} {ORG:DAN} ayrılıp {ORG:E} geçti.",
    "kargoyu {FIRST:E} bırakın {ADDR}",
)

TRAIN_NEGATIVES_V3 = (
    "Barışçıl bir çözüm için taraflar masaya oturdu.",
    "Engin bilgisiyle herkesi etkiledi.",
    "Ay hilal şeklindeyken gökyüzü çok güzel görünüyor.",
    "Kat planındaki değişiklik için yönetim onayı gerekiyor.",
    "Sokak lezzetleri festivali hafta sonu başlıyor.",
    "Caddeler bayram nedeniyle bugün boş.",
    "Numara {NUM} olan dosyayı arşive kaldır.",
    "Kapıdaki no parking yazısını görmedin mi?",
    "Oturum süresi doldu, tekrar giriş yapın.",
    "Adres alanını boş bırakırsanız form gönderilmez.",
    "Panel montajı hava koşullarına bağlı olarak ertelenebilir.",
    "Daire içi tadilat için komşulara bilgi verildi.",
    "Faiz oranları beklentilerin üzerinde açıklandı.",
    "Sözleşmenin 4. maddesi yeniden düzenlendi.",
    "Sipariş no {NUM}, tahmini teslim {DATE}.",
    "Adresi bilmiyorum, sen biliyor musun?",
    "Paket dağıtım merkezinde bekliyor.",
    "Ofis taşınıyor ama yeni yer henüz belli değil.",
    "Mahalle aralarında hız sınırı 30 km.",
    "toplantı {TIME}'da, oda no {NUM}",
    "ödeme alındı, teşekkürler",
    "şifremi unuttum nasıl sıfırlarım",
    # kod / SQL / log: büyük harfli anahtar sözcükler varlık değildir
    "SELECT id, ad FROM musteri WHERE durum = 'aktif';",
    "UPDATE siparis SET durum = 'iptal' WHERE id = {NUM};",
    "DELETE FROM oturum WHERE bitis < NOW();",
    "CREATE TABLE fatura (id SERIAL PRIMARY KEY, tutar NUMERIC);",
    "GET /api/v1/orders?status=PENDING 200 OK",
    "WARN RetryPolicy: max attempts reached, giving up",
    "if (user == null) return ERROR_NOT_FOUND;",
    "TODO: README güncellenecek, CHANGELOG eklenecek",
    # İngilizce gün/ay ve iş jargonu
    "Deadline is Friday, let's sync on Monday morning.",
    "Q3 review moved to Thursday, agenda TBD.",
    "Please send the report by end of day Tuesday.",
    "Sprint planning every Wednesday at {TIME}.",
    # departman / birim kısaltmaları
    "İK ile görüştüm, izin formu onaylandı.",
    "BT ekibi bilgisayarı yarın teslim edecek.",
    "Satın alma ve muhasebe birimleri toplantıya katılacak.",
    "Ar-Ge bütçesi bu çeyrekte artırıldı.",
    # büyük harfle yazılmış (bağırma) ama varlık içermeyen cümleler
    "LÜTFEN SİPARİŞİMİ İPTAL EDİN",
    "ACİL DÖNÜŞ BEKLİYORUM, ÜRÜN HÂLÂ GELMEDİ",
    "BU KADAR GEÇ TESLİMAT OLMAZ",
    # büyük harfle başlayan sıradan sözcükler (isim/kurum değil)
    "Intranet sayfasındaki duyuruyu okudunuz mu?",
    "Portal üzerinden başvuru yapılabilir.",
    "Dashboard'daki rakamlar güncellenmedi.",
)

DEV_NEGATIVES_V3 = (
    "Toplantı odası 4. katta, asansörün yanında.",
    "Bugün sokaklar çok sessiz.",
    "Fatura no {NUM}, ödeme bekleniyor.",
)
