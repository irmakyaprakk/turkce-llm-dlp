"""augmented-v5: yapay zekâya yazılan günlük tanıtım ve istek dili.

Geliştirme ölçümünde (reports/casing, realistic_batch3) görülen hata TÜRLERİNİ hedefler; cümleler kopyalanmadı:
  - "selam ben <ad> <yer>de yaşıyorum <kurum>de okuyorum" tanıtımları, tamamı küçük harf ve ek bitişik
  - kişi adı taşıyan kurumlar ("necmettin erbakan üniversitesi"): model "kişi + kurum" diye bölüyordu
  - sözcük anlamı da olan adlar ("nur", "bulut") ve ilçeler ("kemer", "maden"): pozitif bağlamla birlikte
    aynı sözcüklerin sıradan kullanımı zor negatif olarak verilir
  - ad ile bitişik ekli yer adının yan yana gelmesi ("can kemerde"): tek span'a birleştirilmemeli
  - BÜYÜK HARF ve karışık harf ("Irmak Bucakta … Mehmet akif ersoy üniversitesinde")

Yeni slotlar: {EDU} okul/üniversite/hastane/belediye (ORG), {AMBF} sözcük anlamı da olan ad (PERSON),
{AMBL} sözcük anlamı da olan ilçe (LOCATION); etiketsiz {LIVE}/{STUDY}/{WORK} tanıtım fiilleri. Geliştirme partilerindeki (realistic_batch*) adlar, ilçeler ve
kurumlar bu slotlarda kullanılmaz (bkz. prepare_augmented.FillerV5). Mühürlü setlere bakılmadan yazıldı.
"""

from __future__ import annotations

# Sözcük anlamı da olan adlar. Geliştirme setinde ilk sözcük olarak geçenler FillerV5'te ayrıca elenir.
AMBIGUOUS_FIRST_NAMES = (
    "Nur",
    "Can",
    "Ay",
    "Ada",
    "Kader",
    "Sevgi",
    "Bulut",
    "Güneş",
    "Şafak",
    "Doğa",
    "Derya",
    "Ümit",
    "Nehir",
    "Ceylan",
    "Fırat",
    "Dicle",
    "Aras",
    "Rüzgar",
    "Çiğdem",
    "Gül",
    "Lale",
    "Menekşe",
    "Nergis",
    "Seher",
    "Şimşek",
    "Tan",
    "Tuna",
    "Yonca",
    "Zeytin",
    "Gökyüzü",
)

# Kişi adı ya da tarihî ad taşıyan kurum adları (baş sözcük FillerV5'te eklenir).
# Okul/hastane türleri yalnız "-si/-sı" ile biter: ek üreteci tamlama ekini yalnız bunlarda doğru kaynaştırır
# ("Lisesi'nde"; "Ortaokulu" → yanlış "Ortaokulu'da" çıkardı).
NAMED_UNIVERSITIES = (
    "Necmettin Erbakan",
    "Katip Çelebi",
    "Hasan Kalyoncu",
    "Ahi Evran",
    "Hacı Bektaş Veli",
    "Tokat Gaziosmanpaşa",
    "Zonguldak Bülent Ecevit",
    "Kahramanmaraş Sütçü İmam",
    "Afyon Kocatepe",
    "Niğde Ömer Halisdemir",
    "Karamanoğlu Mehmetbey",
    "Aydın Adnan Menderes",
    "Hatay Mustafa Kemal",
    "Ondokuz Mayıs",
    "Bolu Abant İzzet Baysal",
    "Muğla Sıtkı Koçman",
    "Manisa Celal Bayar",
    "Osmaniye Korkut Ata",
    "Nuh Naci Yazgan",
    "Piri Reis",
    "Kadir Has",
    "Hitit",
    "Kafkas",
    "Munzur",
    "Harran",
    "İnönü",
    "Erciyes",
    "Dumlupınar",
    "Bartın",
    "Sinop",
    "Giresun",
    "Artvin Çoruh",
    "Bitlis Eren",
    "Van Yüzüncü Yıl",
    "Muş Alparslan",
    "Atatürk",
    "Gazi",
    "Selçuk",
    "Trakya",
    "Kırıkkale",
)
SCHOOL_KINDS = ("Anadolu Lisesi", "Fen Lisesi", "Kız Anadolu Lisesi", "Sosyal Bilimler Lisesi", "Spor Lisesi")
HOSPITAL_KINDS = (
    "Şehir Hastanesi",
    "Eğitim ve Araştırma Hastanesi",
    "Devlet Hastanesi",
    "Ağız ve Diş Sağlığı Hastanesi",
    "Kadın Doğum Hastanesi",
)

# Okul/üniversitedeki ve evdeki fiiller: tanıtım cümlesinin sonu
LIVE = ("yaşıyorum", "oturuyorum", "kalıyorum", "yaşıyoruz", "oturuyoz", "yaşıyom", "takılıyorum")
STUDY = ("okuyorum", "okuyom", "son sınıfım", "hazırlıktayım", "yüksek lisans yapıyorum", "öğrenciyim")
WORK = ("çalışıyorum", "çalışıyom", "hemşireyim", "stajdayım", "memurum", "öğretmenim", "doktorum")

TRAIN_TEMPLATES_V5 = (
    # --- tanıtım ---
    "selam ben {FIRST} {LOC:DA} {LIVE} {EDU:DA} {STUDY}",
    "merhaba ben {PER} {ILCE:DA} {LIVE} {EDU:DA} {WORK}",
    "slm ben {AMBF} {AMBL:DA} {LIVE}",
    "selam ben {AMBF} {EDU:DA} {STUDY} bana yardım eder misin",
    "merhaba benim adım {AMBF} {LOC:DA} {LIVE}",
    "selamm ben {FIRST} {AMBL:DA} doğdum büyüdüm şimdi {LOC:DA} {LIVE}",
    "ben {PER} {EDU} öğrencisiyim staj dilekçesi yazar mısın",
    "hey ben {AMBF} {ILCE:DA} {LIVE} {EDU:DA} {STUDY} cv hazırla",
    "merhabalar ben {PER} {LOC} {EDU:DA} {WORK}",
    "ben {AMBF} soyadım {FIRST} {EDU:DA} {STUDY}",
    "selam ben {AMBF} {AMBL:DAN} geliyorum {EDU:E} yeni başladım",
    "merhaba adım {PER} aslen {AMBL} doğumluyum şimdi {LOC:DA} {LIVE}",
    "ben {FIRST} {LOC:DA} {EDU:DA} {WORK} nöbet çizelgesi lazım",
    "selam ben {PER} {EDU:DAN} mezunum iş başvurusu için ön yazı yaz",
    # --- istek ---
    "bana özgeçmiş yaz adım {PER} {EDU:DAN} mezunum {BRAND:DA} stajyerdim",
    "annem {AMBF} hanım için {EDU} randevusu nasıl alınır",
    "kardeşim {FIRST} {EDU:DA} okuyor ona dilekçe yazar mısın",
    "hocam {PER} {EDU:DA} ders veriyor ona mail yaz",
    "müdürüm {PER} {EDU:DA} çalışıyor resmi yazı lazım",
    "{AMBF} ile {FIRST} {AMBL:DA} tanışmış bunu hikaye yap",
    "{EDU:E} başvuru için niyet mektubu yaz ben {AMBF} {AMBL:DA} {LIVE}",
    "{FIRST} hocaya teşekkür maili yaz {EDU} adına bize yardım etti",
    "arkadaşım {AMBF} {EDU:DA} tedavi oldu teşekkür yazısı yaz",
    "oğlum {AMBF} {EDU:DA} okuyor veli toplantısı için not yaz",
    # --- sohbet ---
    "{AMBF} napıyon akşam {AMBL:E} geliyon mu",
    "{AMBF:LA} konuştum {EDU:E} kayıt olmuş",
    "abi {AMBF:E} söyle {AMBL:DA} bekliyorum",
    "{AMBF} ablanın kızı {EDU:I} kazanmış",
    "{AMBL:DA} kahvaltı yapalım mı {AMBF:LA} {FIRST:I} da çağır",
    "{FIRST} dayıyla {AMBL:DAN} dönüyoz yoldayız",
    "{AMBF} {EDU:DA} yatıyo ziyarete gidcez",
    "{AMBF} {AMBL:DA} oturuyo {EDU:E} yakın",
    "dün {AMBF} aradı {EDU:DAN} çıkmış",
    "{AMBF:IN} annesi {EDU:DA} hemşire",
    # --- karışık / büyük harf odaklı (varyant ayrıca uygulanır) ---
    "SELAM BEN {PER} {LOC:DA} YAŞIYORUM {EDU:DA} OKUYORUM",
    "ACİL {PER} {EDU:DAN} ARANDI",
    "{EDU:E} DİLEKÇE YAZ BENİM ADIM {PER}",
)

DEV_TEMPLATES_V5 = (
    "selam ben {AMBF} {AMBL:DA} {LIVE} {EDU:DA} {STUDY}",
    "merhaba ben {PER} {EDU:DA} {WORK}",
    "ben {AMBF} {LOC:DAN} geliyorum",
    "{AMBF} {EDU:DAN} mezun olmuş",
    "kızım {AMBF} {EDU:DA} okuyor dilekçe yaz",
    "{AMBF:E} sor {AMBL:DA} mı",
    "SELAM BEN {AMBF} {AMBL:DA} YAŞIYORUM",
)

# Aynı sözcüklerin sıradan kullanımı: varlık yok. Geliştirme partisindeki cümlelerden farklı sözcüklerle yazıldı.
TRAIN_NEGATIVES_V5 = (
    "nur topu gibi bir bebeği olmuş",
    "can sıkıntısından ne yapacağımı bilmiyorum",
    "ay tutulması bu gece görünecekmiş",
    "ada vapuruna yetişemedik",
    "kader böyleymiş deyip geçtik",
    "sevgi emek ister",
    "bulut çok kara yağmur yağacak galiba",
    "güneş açtı çamaşırları asalım",
    "şafak sökerken yola çıktık",
    "doğa yürüyüşüne gidelim mi hafta sonu",
    "derya deniz bir bilgisi var",
    "ümit var bu sefer olacak",
    "nehir kenarında piknik yaptık",
    "fırat nehri taşmış haberlerde gördüm",
    "rüzgar çok sert pencereyi kapat",
    "lale soğanlarını ne zaman dikeyim",
    "menekşe saksısını cama koy",
    "zeytin ağacı budanır mı",
    "kemer çok sıktı bir delik açtım",
    "maden suyu iyi gelir mideye",
    "orta şekerli kahve alayım",
    "güney cephede ev aradım",
    "çiftlik evinde tatil yapmak istiyorum",
    "tire tire oldu kazağım",
    "kula kulağa yayıldı haber",
    "olur mu öyle şey hiç sanmam",
    "devlet hastanesine mi gideyim özel mi",
    "üniversitede okurken yurtta kaldım",
    "lisede matematik hocamız çok iyiydi",
    "belediyeye dilekçe nasıl yazılır",
    "hastanede sıra çok uzundu",
    "okulun servisi geç kaldı",
    "üniversitesine yakın bir ev bulmuş",
    "SAKIN KAPIYI AÇIK BIRAKMA",
    "YARIN TOPLANTI VAR UNUTMA",
)

DEV_NEGATIVES_V5 = (
    "şimşek çaktı elektrikler gitti",
    "yonca tarlasında koyunlar otluyor",
    "fakültede kayıt haftası başladı",
    "BU AKŞAM GEÇ GELİRİM",
)
