"""Kural tabanlı kurum (ORG) ve adres (ADDRESS) tanıyıcı.

NER modelinin tamamlayıcısıdır (hibrit tespit): yapısal işaretleri güçlü olan
varlıkları — "A.Ş.", "Bankası", "Üniversitesi" ekli kurumları ve "No: 12" çapalı
adresleri — modelden bağımsız yakalar. `DetectionPipeline` içinde model ile
birlikte bir `EntityRecognizer` olarak kullanılır; çakışmalarda uzun span kazanır.

Tasarım ilkeleri:
- Yüksek kesinlik: zayıf ipuçlarıyla (yalnızca büyük harf, sektör kelimesi) tahmin yapılmaz.
- Kesme işaretiyle gelen ek span dışında kalır ("Bankası'na" → "Bankası").
- Küçük harf ve Türkçe karaktersiz yazım desteklenir (anahtar kelimelerde).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .ner import EntitySpan

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜâîû", "cgiosuCGIOSUaiu")


def _key(word: str) -> str:
    return word.replace("İ", "i").replace("I", "ı").lower().translate(_ASCII)


def _base(token: str) -> str:
    """Kesme işaretli eki ayırır: "Kırşehir'e" → "Kırşehir"."""
    return re.split(r"['’]", token, maxsplit=1)[0]


# ---------------------------------------------------------------------------
# Sözlükler
# ---------------------------------------------------------------------------

ILLER = (
    "Adana Adıyaman Afyonkarahisar Ağrı Amasya Ankara Antalya Artvin Aydın Balıkesir Bilecik Bingöl Bitlis "
    "Bolu Burdur Bursa Çanakkale Çankırı Çorum Denizli Diyarbakır Edirne Elazığ Erzincan Erzurum Eskişehir "
    "Gaziantep Giresun Gümüşhane Hakkari Hatay Isparta Mersin İstanbul İzmir Kars Kastamonu Kayseri "
    "Kırklareli Kırşehir Kocaeli Konya Kütahya Malatya Manisa Kahramanmaraş Mardin Muğla Muş Nevşehir Niğde "
    "Ordu Rize Sakarya Samsun Siirt Sinop Sivas Tekirdağ Tokat Trabzon Tunceli Şanlıurfa Uşak Van Yozgat "
    "Zonguldak Aksaray Bayburt Karaman Kırıkkale Batman Şırnak Bartın Ardahan Iğdır Yalova Karabük Kilis "
    "Osmaniye Düzce"
).split()
ILCELER = (
    "Kadıköy Beşiktaş Üsküdar Şişli Bakırköy Maltepe Kartal Pendik Ümraniye Beylikdüzü Esenyurt Sarıyer "
    "Beykoz Fatih Beyoğlu Ataşehir Bağcılar Bahçelievler Başakşehir Esenler Eyüpsultan Gaziosmanpaşa Güngören "
    "Kağıthane Küçükçekmece Sancaktepe Sultanbeyli Sultangazi Tuzla Zeytinburnu Avcılar Arnavutköy Silivri "
    "Çatalca Büyükçekmece Çekmeköy Adalar Bayrampaşa Çankaya Keçiören Yenimahalle Etimesgut Mamak Sincan "
    "Altındağ Gölbaşı Pursaklar Polatlı Konak Bornova Karşıyaka Buca Bayraklı Çiğli Gaziemir Karabağlar "
    "Narlıdere Balçova Torbalı Menemen Nilüfer Osmangazi Yıldırım İnegöl Gemlik Mudanya Muratpaşa Konyaaltı "
    "Kepez Alanya Manavgat Seyhan Çukurova Yüreğir Sarıçam Selçuklu Meram Karatay Tepebaşı Odunpazarı "
    "Atakum İlkadım Canik Ortahisar Akçaabat Merkezefendi Pamukkale Bodrum Marmaris Fethiye Milas Çeşme "
    "Kuşadası İzmit Gebze Darıca Gölcük Serdivan Adapazarı Çorlu Kocasinan Melikgazi Talas Şahinbey "
    "Şehitkamil Bağlar Kayapınar Yenişehir Akdeniz Toroslar Mezitli Merkez"
).split()
PLACES = frozenset(_key(name) for name in (*ILLER, *ILCELER))

ORG_HEADS = frozenset(
    _key(word)
    for word in (
        "Holding Bankası Bankasi Üniversitesi Hastanesi Belediyesi Bakanlığı Müdürlüğü Başkanlığı Kurumu "
        "Kurulu Derneği Vakfı Odası Borsası Mahkemesi Savcılığı Valiliği Kaymakamlığı Lisesi Koleji "
        "Fakültesi Enstitüsü Şirketi Kooperatifi Federasyonu Sendikası Ajansı Gazetesi Müzesi Kütüphanesi "
        "Akademisi Merkezi Dairesi"
    ).split()
)
# Şirket türü ekleri (çok token'lı olanlar ayrı ele alınır).
COMPANY_SUFFIXES = (
    ("a", ".", "ş", "."),
    ("a", ".", "ş"),
    ("aş",),
    ("ltd", ".", "şti", "."),
    ("ltd", ".", "şti"),
    ("ltd", "."),
    ("inc", "."),
    ("llc",),
    ("gmbh",),
)
COMPANY_SUFFIXES = tuple(tuple(_key(part) for part in suffix) for suffix in COMPANY_SUFFIXES)
ORG_CONNECTORS = frozenset({"ve", "&", "san", "tic", ".", "-"})
# Cümle başında büyük harfle yazılan ama kurum adının parçası olmayan sözcükler.
NON_NAME_WORDS = frozenset(
    _key(word)
    for word in (
        "Merhaba Sayın Yarın Bugün Dün Ekte Ekteki Lütfen Konu İlgi Toplantı Müşteri Teslimat Fatura Adres "
        "Bu Şu O Ben Biz Siz Onlar Ve Ama Fakat Ancak Ayrıca Hasta Kiracı Müvekkilimiz Personelimiz Davacı "
        "Referans Deneyim Eğitim Başvuran Kurum Şube Client FYI Hi Selamlar Not İyi Saygılarımla Siparişiniz "
        "Ürün Doğrusu Kargo Paket Numuneler"
    ).split()
)

MAH_WORDS = frozenset({"mahallesi", "mah", "mh", "mahalle"})
STREET_WORDS = frozenset(
    {
        "sokak",
        "sokagi",
        "sok",
        "sk",
        "caddesi",
        "cadde",
        "cad",
        "cd",
        "bulvari",
        "bulv",
        "blv",
        "yolu",
        "meydani",
        "cikmazi",
        "sitesi",
        "apt",
        "apartmani",
        "blok",
        "organize",
        "osb",
    }
)
NUMBER_WORDS = frozenset({"no", "numara", "nu", "kapi"})
UNIT_WORDS = frozenset({"d", "daire", "k", "kat", "blok", "apt", "bina"})
ADDRESS_STOP = frozenset(
    _key(word)
    for word in (
        "adres adresi adresim adresimiz adresine adresinden adresindeki tam teslimat kargo alıcı "
        "fatura tebligat ikametgah ikamet şube depo ofis evin evim ve ile için lütfen"
    ).split()
)
_NUMBER_RE = re.compile(r"^\d+[a-zA-Z]?$")
# İki sözcüklü mahalle/sokak adlarının sık görülen ilk sözcükleri ("Yavuz Selim", "Kazım Karabekir").
COMPOUND_FIRST = frozenset(
    _key(word)
    for word in (
        "Yavuz Kazım Mimar Fevzi Mustafa Gazi Yeni Eski Yukarı Aşağı Orta Kemal Fatih Sultan Hacı Şehit "
        "Cumhuriyet Atatürk İsmet Adnan Barbaros Hürriyet Zafer Kurtuluş Yunus Ahmet Mehmet Ali Hasan Hüseyin "
        "Osman Murat Mareşal Fevzipaşa Namık Akşemsettin Hoca Molla Kara Ak Gül Çamlı Bahçe Turgut Necip "
        "Abdi İpekçi Halil Rıfat Cemal Rauf Ziya Sakarya Ankara İstanbul İzmir Organize Sanayi"
    ).split()
)
# "vergi no 123", "sipariş no 456": numara ama adres değil.
NON_ADDRESS_NUMBER_CONTEXT = frozenset(
    {
        "vergi",
        "siparis",
        "siparisiniz",
        "telefon",
        "tel",
        "takip",
        "kimlik",
        "tc",
        "fatura",
        "musteri",
        "hesap",
        "kart",
        "dosya",
        "kayit",
        "oda",
        "masa",
        "sira",
        "referans",
        "ref",
        "abone",
        "abonelik",
        "bilet",
        "seri",
    }
)


# ---------------------------------------------------------------------------
# Tanıyıcı
# ---------------------------------------------------------------------------


class RuleEntityRecognizer:
    """`EntityRecognizer` arayüzüne uyan, model gerektirmeyen ORG/ADDRESS tanıyıcı."""

    def recognize(self, text: str) -> Iterable[EntitySpan]:
        tokens = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        spans = list(self._addresses(text, tokens))
        taken = [(s.start, s.end) for s in spans]
        for span in self._orgs(text, tokens):
            if not any(span.start < end and span.end > start for start, end in taken):
                spans.append(span)
        return sorted(spans, key=lambda s: s.start)

    # --- kurumlar ---
    def _orgs(self, text: str, tokens):
        keys = [_key(_base(tok)) for tok, _, _ in tokens]
        consumed = -1
        for index in range(len(tokens)):
            if index <= consumed:
                continue
            end_index = None
            if keys[index] in ORG_HEADS and tokens[index][0][:1].isupper():
                end_index = index
            else:
                for suffix in COMPANY_SUFFIXES:
                    n = len(suffix)
                    if n == 1 and suffix[0] == "as" and not tokens[index][0].isupper():
                        continue  # "AŞ" yalnız büyük harfle: İngilizce "as" ile karışmasın
                    if tuple(keys[index : index + n]) == suffix and (
                        index + n == len(tokens) or keys[index + n] != "."
                    ):
                        end_index = index + n - 1
                        break
                    if tuple(keys[index : index + n]) == suffix:
                        end_index = index + n - 1
                        break
            if end_index is None:
                continue
            # "Gazi Üniversitesi Hastanesi": ardışık baş sözcükleri tek kurum say.
            while (
                end_index + 1 < len(tokens)
                and keys[end_index + 1] in ORG_HEADS
                and tokens[end_index + 1][0][:1].isupper()
                and "'" not in tokens[end_index][0]
            ):
                end_index += 1
            start_index = index
            # Sola doğru büyük harfli ad sözcüklerini ve "San. ve Tic." bağlaçlarını topla.
            cursor = index - 1
            words = 0
            while cursor >= 0 and words < 5:
                token = tokens[cursor][0]
                # Arada başka bir ifade (ör. kesme işaretli ek, satır sonu) varsa dur.
                if text[tokens[cursor][2] : tokens[cursor + 1][1]].count("\n"):
                    break
                if keys[cursor] in ORG_CONNECTORS or token.isdigit():
                    cursor -= 1
                    continue
                if token[:1].isupper() and keys[cursor] not in NON_NAME_WORDS and "'" not in token:
                    start_index = cursor
                    words += 1
                    cursor -= 1
                    continue
                break
            # Baştaki bağlaçları at.
            while start_index < index and keys[start_index] in ORG_CONNECTORS:
                start_index += 1
            if start_index == index and keys[index] in ORG_HEADS:
                continue  # yalnız "Bankası" gibi baş sözcük: ad yok, kurum sayma
            if start_index == index and end_index != index:
                continue  # yalnız "A.Ş.": ad yok
            start = tokens[start_index][1]
            last = tokens[end_index]
            end = last[1] + len(_base(last[0]))
            consumed = end_index
            yield EntitySpan("ORG", start, end)

    # --- adresler ---
    def _addresses(self, text: str, tokens):
        keys = [_key(_base(tok)) for tok, _, _ in tokens]
        used_until = -1
        for index in range(len(tokens)):
            if index <= used_until:
                continue
            anchor_end = self._number_anchor(tokens, keys, index)
            if anchor_end is None:
                continue
            start_index = self._extend_left(tokens, keys, index, text, used_until + 1)
            if start_index is None:
                continue
            end_index = self._extend_right(tokens, keys, anchor_end, text)
            used_until = end_index
            last = tokens[end_index]
            end = last[1] + len(_base(last[0]))
            yield EntitySpan("ADDRESS", tokens[start_index][1], end)

    @staticmethod
    def _number_anchor(tokens, keys, index):
        """ "No: 12", "no 12/3", "Numara 5" veya sokak/bulvar sözcüğünden hemen sonra "54/8"."""
        if keys[index] in NUMBER_WORDS:
            if index > 0 and keys[index - 1] in NON_ADDRESS_NUMBER_CONTEXT:
                return None
            cursor = index + 1
            while cursor < len(tokens) and tokens[cursor][0] in {":", "."}:
                cursor += 1
            if cursor < len(tokens) and _NUMBER_RE.match(tokens[cursor][0]):
                return _slash_number(tokens, cursor)
            return None
        if keys[index] in UNIT_WORDS and keys[index] not in {"blok", "apt", "bina"}:
            # "Kardelen Sitesi C Blok D:14" gibi numarasız ama daire bilgili adresler
            cursor = index + 1
            while cursor < len(tokens) and tokens[cursor][0] in {":", "."}:
                cursor += 1
            window = keys[max(0, index - 8) : index]
            if (
                cursor < len(tokens)
                and _NUMBER_RE.match(tokens[cursor][0])
                and any(key in MAH_WORDS for key in window)
                and any(key in {"sitesi", "blok", "apt", "apartmani"} for key in window)
            ):
                return cursor
            return None
        if _NUMBER_RE.match(tokens[index][0]) and index > 0:
            previous = index - 1
            if tokens[previous][0] == "." and previous > 0:
                previous -= 1
            if keys[previous] in STREET_WORDS and index + 1 < len(tokens) and tokens[index + 1][0] == "/":
                return _slash_number(tokens, index)
        return None

    @staticmethod
    def _extend_left(tokens, keys, index, text, floor=0):
        """Çapadan sola: en az bir sokak/mahalle sözcüğü bulunmalı; önündeki adı (1-2 sözcük) ekle."""
        window_start = max(floor, index - 12)
        keyword_positions = [i for i in range(window_start, index) if keys[i] in STREET_WORDS or keys[i] in MAH_WORDS]
        if not keyword_positions:
            return None
        start = keyword_positions[0]
        if keys[start] in {"organize", "osb"}:
            # "Tuzla Organize Sanayi", "İkitelli OSB": önünde büyük harfli bir yer adı olabilir.
            prev = start - 1
            if (
                prev >= window_start
                and tokens[prev][0][:1].isupper()
                and keys[prev] not in ADDRESS_STOP
                and keys[prev] not in NON_NAME_WORDS
            ):
                return prev
            return start
        # Anahtar sözcüğün önündeki ad: sayı ("1234 sk", "45. Sk") ya da 1 sözcük; ilk sözcük
        # yaygın bileşik ad başıysa ("Yavuz Selim") 2 sözcük.
        cursor = start - 1
        if (
            cursor >= window_start
            and tokens[cursor][0] == "."
            and cursor > 0
            and _NUMBER_RE.match(tokens[cursor - 1][0])
        ):
            return cursor - 1  # "45. Sk."
        candidates = []
        while cursor >= window_start and len(candidates) < 2:
            token = tokens[cursor][0]
            if "\n" in text[tokens[cursor][2] : tokens[cursor + 1][1]]:
                break
            if not re.match(r"^\w", token) or keys[cursor] in ADDRESS_STOP or "'" in token or "’" in token:
                break
            candidates.append(cursor)
            cursor -= 1
        if not candidates:
            return start
        if len(candidates) == 2 and keys[candidates[1]] in COMPOUND_FIRST:
            return candidates[1]
        if len(candidates) == 2 and tokens[candidates[0]][0].isdigit() and tokens[candidates[1]][0][:1].isupper():
            return candidates[1]  # "Alsancak 1442 Sokak": semt + numaralı sokak
        return candidates[0]

    @staticmethod
    def _extend_right(tokens, keys, index, text):
        """Çapadan sağa: daire/kat/blok, posta kodu ve il/ilçe adlarını ekle."""
        end = index
        cursor = index + 1
        while cursor < len(tokens):
            gap = text[tokens[cursor - 1][2] : tokens[cursor][1]]
            if "\n" in gap:
                break
            token, key = tokens[cursor][0], keys[cursor]
            if key in UNIT_WORDS:
                probe = cursor + 1
                while probe < len(tokens) and tokens[probe][0] in {":", "."}:
                    probe += 1
                if probe < len(tokens) and _NUMBER_RE.match(tokens[probe][0]):
                    end = cursor = probe
                    cursor += 1
                    continue
                break
            if (
                _NUMBER_RE.match(token)
                and cursor + 2 < len(tokens)
                and tokens[cursor + 1][0] == "."
                and keys[cursor + 2] == "kat"
            ):
                end = cursor = cursor + 2
                cursor += 1
                continue
            if re.fullmatch(r"\d{5}", token) or key in PLACES:
                end = cursor
                cursor += 1
                continue
            if (
                token in {"/", ",", "-"}
                and cursor + 1 < len(tokens)
                and (keys[cursor + 1] in PLACES or re.fullmatch(r"\d{5}", tokens[cursor + 1][0]))
            ):
                cursor += 1
                continue
            break
        return end


def _slash_number(tokens, index):
    """ "14/2", "46/5" gibi bölü işaretli numaraları tek parça say."""
    end = index
    while end + 2 < len(tokens) and tokens[end + 1][0] == "/" and _NUMBER_RE.match(tokens[end + 2][0]):
        end += 2
    return end
