"""Türkçe NER verisi için etiket korumalı yazım dönüşümleri ve adres yeniden yazımı.

Bütün dönüşümler (metin, span listesi) üzerinde çalışır; span'lar karakter
offset'idir ve dönüşüm sonrası yeniden hesaplanır. Böylece etiketler hiçbir
zaman elle kaydırılmaz. Tokenizasyon WikiNER kuralıyla aynıdır: kesme işaretli
ek kelimeye bağlı kalır ve token'ın tamamı etiketlenir.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
APOSTROPHES = "'’"

Span = tuple[int, int, str]  # (başlangıç, bitiş, etiket) — etiket: PERSON/ORG/LOCATION/ADDRESS

# ---------------------------------------------------------------------------
# Türkçe harf dönüşümleri (uzunluk korunur)
# ---------------------------------------------------------------------------

_LOWER = str.maketrans({"İ": "i", "I": "ı"})
_UPPER = str.maketrans({"i": "İ", "ı": "I"})
_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")


def tr_lower(text: str) -> str:
    """Türkçe kurallarıyla küçük harf (İ→i, I→ı); uzunluğu değiştirmez."""
    return text.translate(_LOWER).lower()


def tr_upper(text: str) -> str:
    return text.translate(_UPPER).upper()


def to_ascii(text: str) -> str:
    """Türkçe karakterleri kaldırır (ş→s, ı→i ...); klavye/telefon yazımını taklit eder."""
    return text.translate(_ASCII)


# ---------------------------------------------------------------------------
# Ünlü uyumu ile ek üretimi
# ---------------------------------------------------------------------------

_BACK = set("aıou")
_FRONT = set("eiöü")
_VOWELS = _BACK | _FRONT
_HARD = set("fstkçşhp")  # Fıstıkçı Şahap: sert ünsüzler
_POSSESSIVE_ENDINGS = ("si", "sı", "su", "sü", "ğı", "ği", "ğu", "ğü")


def _last_vowel(word: str) -> str | None:
    for char in reversed(tr_lower(word)):
        if char in _VOWELS:
            return char
    return None


def _two_way(vowel: str | None) -> str:
    return "a" if vowel in _BACK or vowel is None else "e"


def _four_way(vowel: str | None) -> str:
    return {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "i": "i", "ö": "ü", "ü": "ü"}.get(vowel or "a", "ı")


def suffix(word: str, kind: str) -> str:
    """Özel isme gelecek eki ünlü uyumuyla üretir (kesme işareti hariç).

    kind: DA (bulunma), DAN (ayrılma), E (yönelme), I (belirtme), IN (ilgi), LA (vasıta),
    DAKI (bulunma + ki), LAR / LARA / LARDA / LARDAN (çoğul; "Aylinlere" = Aylin'in ailesine).
    Örnek: suffix("İzmir", "DA") == "de", suffix("Ayşe", "E") == "ye", suffix("Bucak", "DAKI") == "taki".
    """
    if kind == "DAKI":
        return suffix(word, "DA") + "ki"
    if kind.startswith("LAR"):
        plural = "l" + _two_way(_last_vowel(tr_lower(word.rstrip(".")))) + "r"
        rest = kind[3:]
        if not rest:
            return plural
        return plural + suffix(plural, {"A": "E", "DA": "DA", "DAN": "DAN"}[rest])
    lowered = tr_lower(word.rstrip("."))
    last_word = lowered.split()[-1] if lowered.split() else ""
    vowel = _last_vowel(lowered)
    if len(last_word) > 4 and last_word.endswith(_POSSESSIVE_ENDINGS):
        # Tamlama eki almış isim (Mahallesi, Caddesi, Sokağı): kaynaştırma "n" gelir.
        two, four = _two_way(vowel), _four_way(vowel)
        return {
            "DA": "nd" + two,
            "DAN": "nd" + two + "n",
            "E": "n" + two,
            "I": "n" + four,
            "IN": "n" + four + "n",
            "LA": "yl" + two,
        }[kind]
    last_char = lowered[-1:] if lowered else ""
    vowel = _last_vowel(lowered)
    ends_vowel = last_char in _VOWELS
    hard = last_char in _HARD
    two, four = _two_way(vowel), _four_way(vowel)
    if kind == "DA":
        return ("t" if hard else "d") + two
    if kind == "DAN":
        return ("t" if hard else "d") + two + "n"
    if kind == "E":
        return ("y" if ends_vowel else "") + two
    if kind == "I":
        return ("y" if ends_vowel else "") + four
    if kind == "IN":
        return ("n" if ends_vowel else "") + four + "n"
    if kind == "LA":
        return ("y" if ends_vowel else "") + "l" + two
    raise ValueError(f"bilinmeyen ek türü: {kind}")


# ---------------------------------------------------------------------------
# Span korumalı metin dönüşümleri
# ---------------------------------------------------------------------------


def _map_spans(spans: Sequence[Span], index_map: list[int]) -> list[Span]:
    """Eski karakter offset'lerini yeni metne taşır; index_map[i] = eski i'nin yeni konumu."""
    return [(index_map[s], index_map[e], label) for s, e, label in spans]


def drop_apostrophes(text: str, spans: Sequence[Span]) -> tuple[str, list[Span]]:
    """Kesme işaretlerini siler ("Kadıköy'de" → "Kadıköyde"); span'ları kaydırır."""
    out: list[str] = []
    index_map: list[int] = []
    for char in text:
        index_map.append(len(out))
        if char not in APOSTROPHES:
            out.append(char)
    index_map.append(len(out))
    return "".join(out), _map_spans(spans, index_map)


def apply_variant(text: str, spans: Sequence[Span], variant: str) -> tuple[str, list[Span]]:
    """Yazım varyantı uygular. variant: orig, lower, upper, ascii, lower_ascii, noapos, lower_noapos, chat, mixed.

    mixed: ek bitişik, her şey küçük harf, yalnız varlıkların ilk harfi büyük ("selam ben Irmak Bucakta yaşıyorum
    Mehmet akif ersoy üniversitesinde okuyorum"): telefonda otomatik büyük harf ve yarım düzeltme.
    """
    spans = list(spans)
    if variant == "orig":
        return text, spans
    if variant == "mixed":
        text, spans = drop_apostrophes(tr_lower(text), spans)
        chars = list(text)
        for start, _end, _label in spans:
            chars[start] = tr_upper(chars[start])
        return "".join(chars), spans
    parts = {
        "lower": ("lower",),
        "upper": ("upper",),
        "ascii": ("ascii",),
        "lower_ascii": ("lower", "ascii"),
        "noapos": ("noapos",),
        "lower_noapos": ("lower", "noapos"),
        "chat": ("lower", "ascii", "noapos"),
    }.get(variant)
    if parts is None:
        raise ValueError(f"bilinmeyen varyant: {variant}")
    for step in parts:
        if step == "lower":
            text = tr_lower(text)
        elif step == "upper":
            text = tr_upper(text)
        elif step == "ascii":
            text = to_ascii(text)
        elif step == "noapos":
            text, spans = drop_apostrophes(text, spans)
    return text, spans


def tokens_and_tags(text: str, spans: Sequence[Span]) -> tuple[list[str], list[str]]:
    """Metni WikiNER kuralıyla token'lar ve span'ları BIO etiketine çevirir.

    Span'la kesişen her token etiketlenir (ek bitişik yazıldıysa token'ın tamamı).
    Span'lar çakışırsa ValueError verir.
    """
    tokens = [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]
    tags = ["O"] * len(tokens)
    for start, end, label in sorted(spans):
        if not text[start:end].strip():
            raise ValueError(f"boş span: {label}")
        first = True
        hit = False
        for index, (_, t_start, t_end) in enumerate(tokens):
            if t_start < end and t_end > start:
                if tags[index] != "O":
                    raise ValueError("span'lar çakışıyor")
                tags[index] = ("B-" if first else "I-") + label
                first = False
                hit = True
        if not hit:
            raise ValueError(f"span hiçbir token'a denk gelmiyor: {text[start:end]!r}")
    return [token for token, _, _ in tokens], tags


def row_variant(tokens: Sequence[str], tags: Sequence[str], variant: str) -> tuple[list[str], list[str]]:
    """Hazır token/BIO satırına yazım varyantı uygular (WikiNER satırları için).

    Token'lar boşlukla birleştirilip span'a çevrilir, dönüştürülür ve yeniden
    token'lanır. Kesme işareti silinince ek kelimeye bitişir; token sayısı korunur.
    """
    text_parts: list[str] = []
    spans: list[Span] = []
    cursor = 0
    current: list[int] | None = None
    current_label = ""
    for token, tag in zip(tokens, tags, strict=True):
        if text_parts:
            cursor += 1
        start, end = cursor, cursor + len(token)
        text_parts.append(token)
        cursor = end
        if tag.startswith("B-") or (tag.startswith("I-") and (current is None or tag[2:] != current_label)):
            if current is not None:
                spans.append((current[0], current[1], current_label))
            current, current_label = [start, end], tag[2:]
        elif tag.startswith("I-") and current is not None:
            current[1] = end
        else:
            if current is not None:
                spans.append((current[0], current[1], current_label))
            current = None
    if current is not None:
        spans.append((current[0], current[1], current_label))
    text, spans = apply_variant(" ".join(text_parts), spans, variant)
    return tokens_and_tags(text, spans)


# ---------------------------------------------------------------------------
# OSM adreslerini farklı yazım stillerinde yeniden yazma
# ---------------------------------------------------------------------------

_OSM_RE = re.compile(r"^(?P<street>.+?) No: (?P<number>[^,]+), (?:(?P<postal>\d{5}) )?(?P<area>.+)$")

# Posta kodunun ilk iki hanesi il plaka kodudur.
IL_BY_PLATE = {
    1: "Adana",
    2: "Adıyaman",
    3: "Afyonkarahisar",
    4: "Ağrı",
    5: "Amasya",
    6: "Ankara",
    7: "Antalya",
    8: "Artvin",
    9: "Aydın",
    10: "Balıkesir",
    11: "Bilecik",
    12: "Bingöl",
    13: "Bitlis",
    14: "Bolu",
    15: "Burdur",
    16: "Bursa",
    17: "Çanakkale",
    18: "Çankırı",
    19: "Çorum",
    20: "Denizli",
    21: "Diyarbakır",
    22: "Edirne",
    23: "Elazığ",
    24: "Erzincan",
    25: "Erzurum",
    26: "Eskişehir",
    27: "Gaziantep",
    28: "Giresun",
    29: "Gümüşhane",
    30: "Hakkari",
    31: "Hatay",
    32: "Isparta",
    33: "Mersin",
    34: "İstanbul",
    35: "İzmir",
    36: "Kars",
    37: "Kastamonu",
    38: "Kayseri",
    39: "Kırklareli",
    40: "Kırşehir",
    41: "Kocaeli",
    42: "Konya",
    43: "Kütahya",
    44: "Malatya",
    45: "Manisa",
    46: "Kahramanmaraş",
    47: "Mardin",
    48: "Muğla",
    49: "Muş",
    50: "Nevşehir",
    51: "Niğde",
    52: "Ordu",
    53: "Rize",
    54: "Sakarya",
    55: "Samsun",
    56: "Siirt",
    57: "Sinop",
    58: "Sivas",
    59: "Tekirdağ",
    60: "Tokat",
    61: "Trabzon",
    62: "Tunceli",
    63: "Şanlıurfa",
    64: "Uşak",
    65: "Van",
    66: "Yozgat",
    67: "Zonguldak",
    68: "Aksaray",
    69: "Bayburt",
    70: "Karaman",
    71: "Kırıkkale",
    72: "Batman",
    73: "Şırnak",
    74: "Bartın",
    75: "Ardahan",
    76: "Iğdır",
    77: "Yalova",
    78: "Karabük",
    79: "Kilis",
    80: "Osmaniye",
    81: "Düzce",
}


@dataclass(frozen=True)
class AddressParts:
    street: str
    number: str
    postal: str | None
    area: str  # OSM "city" alanı; Türkiye çıktısında çoğunlukla "X Mahallesi"

    @property
    def il(self) -> str | None:
        if self.postal and self.postal[:2].isdigit():
            return IL_BY_PLATE.get(int(self.postal[:2]))
        return None

    @property
    def mahalle(self) -> str | None:
        match = re.match(r"^(.+?)\s+Mahallesi$", self.area)
        return match.group(1) if match else None


def parse_osm_address(text: str) -> AddressParts | None:
    match = _OSM_RE.match(text)
    if not match:
        return None
    return AddressParts(match["street"], match["number"].strip(), match["postal"], match["area"].strip())


_STREET_FORMS = {
    "sokak": ("Sokak", "Sk.", "Sok.", "sk", "Sokağı"),
    "cadde": ("Caddesi", "Cad.", "Cd.", "cad", "Cadde"),
    "bulvar": ("Bulvarı", "Bulv.", "Blv.", "bulvarı"),
}


def _restyle_street(street: str, rng: random.Random) -> str:
    words = street.split()
    last = tr_lower(words[-1].rstrip("."))
    for kind, forms in _STREET_FORMS.items():
        if last.startswith(kind[:3]) or (kind == "sokak" and last in {"sk", "sok"}):
            return " ".join(words[:-1] + [rng.choice(forms)])
        if kind == "cadde" and last in {"cd", "cad"}:
            return " ".join(words[:-1] + [rng.choice(forms)])
    return street


def original_osm_text(parts: AddressParts) -> str:
    """OSM dışa aktarımındaki ham biçim: "Sokak No: 5, 34000 X Mahallesi"."""
    tail = f"{parts.postal} {parts.area}" if parts.postal else parts.area
    return f"{parts.street} No: {parts.number}, {tail}"


def render_address(parts: AddressParts, rng: random.Random, original_ratio: float = 0.0) -> str:
    """Bir OSM adresini Türkiye'de yaygın yazım biçimlerinden birine çevirir.

    Bileşenler (mahalle, sokak, numara, daire/kat, posta kodu, il) rastgele
    biçim ve sırayla yazılır. Daire/kat OSM'de yok; gerçekçi biçim için sentetik eklenir.
    """
    if original_ratio and rng.random() < original_ratio:
        return original_osm_text(parts)
    street = _restyle_street(parts.street, rng)
    mahalle = parts.mahalle
    pieces: list[str] = []
    if mahalle and rng.random() < 0.75:
        pieces.append(f"{mahalle} {rng.choice(('Mah.', 'Mahallesi', 'Mh.', 'mah', 'Mah'))}")
    pieces.append(street)

    number = parts.number
    style = rng.random()
    door = rng.randint(1, 24)
    floor = rng.randint(1, 9)
    if style < 0.30:
        number_text = f"No:{number}"
    elif style < 0.50:
        number_text = f"No: {number}"
    elif style < 0.62:
        number_text = f"no {number}"
    elif style < 0.72:
        number_text = f"No:{number}/{door}" if "/" not in number else f"No:{number}"
    elif style < 0.80:
        number_text = f"Numara {number}"
    else:
        number_text = f"No.{number}"
    pieces.append(number_text)

    extra = rng.random()
    if extra < 0.20:
        pieces.append(rng.choice((f"D:{door}", f"Daire:{door}", f"Daire {door}", f"daire {door}", f"D.{door}")))
    elif extra < 0.32:
        pieces.append(f"K:{floor} D:{door}")
    elif extra < 0.40:
        pieces.append(rng.choice((f"Kat:{floor}", f"Kat {floor}", f"{floor}. kat")))

    il = parts.il
    tail = rng.random()
    if parts.postal and tail < 0.25:
        pieces.append(f"{parts.postal} {il}" if il else parts.postal)
    elif il and tail < 0.75:
        pieces.append(rng.choice((il, f"/{il}", f"- {il}", f"{il} Merkez")))
    if mahalle is None and rng.random() < 0.5:
        pieces.insert(0, parts.area)

    text = " ".join(pieces).replace(" /", "/")
    if rng.random() < 0.3:
        text = text.replace(" No", ", No", 1)
    return re.sub(r"\s+", " ", text).strip()


def street_only(parts: AddressParts, rng: random.Random) -> str:
    """Numarasız sokak/cadde adı — anotasyon kuralına göre ADDRESS değil LOCATION'dır."""
    return _restyle_street(parts.street, rng)
