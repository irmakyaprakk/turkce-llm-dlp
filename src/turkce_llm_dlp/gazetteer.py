"""Türkiye yer adı sözlüğüne (gazetteer) dayalı LOCATION tanıyıcı.

Kaynak: PTT posta kodu listesinden 81 il, 973 ilçe, mahalle ve köy adları; ülke ve bölge adları
(`data/gazetteer/`, üretim: `training/ner/build_gazetteer.py`). Model öncelikli çalışır
(`fallback_types`): yalnızca modelin ve diğer tanıyıcıların boş bıraktığı yeri doldurur.

Belirsizlik yönetimi (bkz. build_gazetteer.py "tier"):
  - "ceyhana", "izmirdeki", "besiktasta" gibi küçük harfli / ASCII / eki bitişik yazımlar
    yalnızca yaygın sözcük olmayan adlarda eşleşir.
  - "Kale", "Orta", "Güney" gibi yaygın sözcükler yalnız büyük harfle (cümle başı değilken) ya da
    kesme işaretli ekle ("Kale'de") eşleşir.
  - Ay/gün adlarıyla aynı ilçeler ("Aralık", "Pazar") ve kişi adı olarak daha sık geçenler
    ("Fatih", "Selim") yalnız kesme işaretli ekle ya da "ili/ilçesi" bağlamıyla eşleşir;
    kulüp adı olarak sık geçenler ("Beşiktaş") ekli biçimde eşleşir, çıplak hâlde eşleşmez.
  - Mahalle/köy adları yalnız "<ad> Mahallesi / köyü" kalıbında kullanılır, tek başına eşleşmez.
  - "Selim Bey", "İzmir Adliyesi" gibi daha büyük bir adın parçası olan yer adları atlanır.

Ek olarak iki kalıp: "<ad> Mahallesi / köyü" ve numarasız "<ad> Caddesi / Sokak / Bulvarı"
(anotasyon kuralı: numarasız cadde/sokak ve mahalle adı LOC'tur).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from ._gazetteer_data import load_gazetteer
from .ner import EntitySpan
from .rules_ner import TOKEN_RE, _key

# Eki bitişik yazılmış yer adları için (ASCII katlanmış anahtar uzayında), uzundan kısaya.
SUFFIXES = tuple(
    sorted(
        {
            "daki",
            "deki",
            "taki",
            "teki",
            "ndaki",
            "ndeki",
            "dan",
            "den",
            "tan",
            "ten",
            "ndan",
            "nden",
            "nda",
            "nde",
            "da",
            "de",
            "ta",
            "te",
            "ya",
            "ye",
            "yi",
            "yu",
            "na",
            "ne",
            "nin",
            "nun",
            "in",
            "un",
            "yla",
            "yle",
            "la",
            "le",
            "a",
            "e",
            "i",
            "u",
        },
        key=len,
        reverse=True,
    )
)
# Ünsüz yumuşaması (anahtar uzayında): "bucağa" → "bucag" → "bucak"
_SOFTEN = {"g": "k", "b": "p", "d": "t"}

HONORIFICS = frozenset(
    _key(w)
    for w in "bey beyi beyin beye hanım hanımı hanıma hoca hocam hocamız abi abla amca teyze efendi usta".split()
)
TITLES = frozenset(_key(w) for w in "sayın dr av prof doç uzm sn bay bayan".split())
ADMIN_CONTEXT = re.compile(r"^(il|ilce)(i|si|sinde|sine|sinin|sinden|inde|ine|inin|inden)?$")
MAH_RE = re.compile(r"^(mahalle(si|sinde|sine|sinin|sinden|sindeki)|mah|mh)$")
KOY_RE = re.compile(r"^koy(u|unde|une|unun|unden|undeki)$")
STREET_RE = re.compile(
    r"^(cadde(si|sinde|sine|sinin|sinden|sindeki)?|cad|cd|bulvar(i|inda|ina|inin|indan|indaki)|blv|bulv"
    r"|sokag(i|inda|ina|inin|indan|indaki)|sokak|sok|sk)$"
)
_SHORT_FORMS = frozenset({"mah", "mh", "cad", "cd", "blv", "bulv", "sok", "sk"})
_SENTENCE_END = frozenset({".", "!", "?", ":", ";", "…"})
_WORD_RE = re.compile(r"^[^\W\d_]", re.UNICODE)


def stems(key: str, minimum: int = 3) -> Iterator[str]:
    """Bitişik ek atılmış aday kökler (yumuşamanın geri alınmışı dahil)."""
    for suffix in SUFFIXES:
        if key.endswith(suffix) and len(key) - len(suffix) >= minimum:
            stem = key[: -len(suffix)]
            yield stem
            if stem[-1] in _SOFTEN:
                yield stem[:-1] + _SOFTEN[stem[-1]]


def strip_suffix(key: str, lookup: Iterable[str] | None = None) -> str | None:
    """İlk aday kök (lookup verilirse içinde bulunan ilk kök)."""
    for stem in stems(key):
        if lookup is None or stem in lookup:
            return stem
    return None


def _key_exact(word: str) -> str:
    """Türkçe harfleri koruyan küçük harf anahtarı (ASCII katlamasız)."""
    return word.replace("İ", "i").replace("I", "ı").lower()


def _is_word(token: str) -> bool:
    return bool(_WORD_RE.match(token))


def _base(token: str) -> str:
    return re.split(r"['’]", token, maxsplit=1)[0]


class LocationGazetteer:
    """`EntityRecognizer` arayüzüne uyan, sözlük tabanlı LOCATION tanıyıcı (model öncelikli)."""

    fallback_types = frozenset({"LOCATION"})

    def __init__(self) -> None:
        data = load_gazetteer()
        self._places: dict[str, dict] = data["places"]
        self._units: frozenset[str] = data["units"]
        self._collisions: frozenset[str] = data["collisions"]

    def recognize(self, text: str) -> list[EntitySpan]:
        tokens = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        keys = [_key(_base(tok)) for tok, _, _ in tokens]
        starts = {0} | {
            i
            for i in range(1, len(tokens))
            if tokens[i - 1][0] in _SENTENCE_END or "\n" in text[tokens[i - 1][2] : tokens[i][1]]
        }
        spans: list[EntitySpan] = []
        taken: set[int] = set()
        for span, covered in self._patterns(text, tokens, keys, starts):
            spans.append(span)
            taken |= covered
        for span, covered in self._multiword(tokens, keys, taken):
            spans.append(span)
            taken |= covered
        for index in range(len(tokens)):
            if index in taken:
                continue
            span = self._single(text, tokens, keys, starts, index)
            if span is not None:
                spans.append(span)
        return sorted(spans, key=lambda s: s.start)

    # --- "<ad> Mahallesi", "<ad> köyü", "<ad> Caddesi" ---
    def _patterns(self, text, tokens, keys, starts):
        for index, key in enumerate(keys):
            is_mah, is_koy, is_street = bool(MAH_RE.match(key)), bool(KOY_RE.match(key)), bool(STREET_RE.match(key))
            if not (is_mah or is_koy or is_street) or index == 0:
                continue
            if key in _SHORT_FORMS:
                # "Sk.", "Cad. 5": kısaltma nokta ya da sayıyla gelir; "şok" (→ "sok") ve "Blv Homme" değil
                following = tokens[index + 1][0] if index + 1 < len(tokens) else ""
                if _key_exact(_base(tokens[index][0])) != key or not (following == "." or following.isdigit()):
                    continue
            # Numaralı adres ise ADDRESS kuralı/model alır; burada LOC üretmek çakışmada zaten elenir.
            first = self._name_start(tokens, keys, starts, index, allow_lower=is_mah or is_koy)
            if first is None:
                continue
            end_token = tokens[index]
            end = end_token[1] + len(_base(end_token[0]))
            yield EntitySpan("LOCATION", tokens[first][1], end), set(range(first, index + 1))

    def _name_start(self, tokens, keys, starts, index, allow_lower):
        words = []
        cursor = index - 1
        while cursor >= 0 and len(words) < 3:
            token = tokens[cursor][0]
            if not (_is_word(token) or re.fullmatch(r"\d+", token)) or "'" in token or "’" in token:
                # "3. Cadde" / "1203. Sokak": sayı + nokta
                if token == "." and cursor >= 1 and re.fullmatch(r"\d+", tokens[cursor - 1][0]):
                    words.append(cursor - 1)
                    cursor -= 2
                    continue
                break
            words.append(cursor)
            if cursor in starts:
                break
            cursor -= 1
        # Sözlükte bulunan en uzun ad
        for length in range(len(words), 0, -1):
            chosen = words[:length]
            joined = " ".join(keys[i] for i in reversed(chosen))
            if joined in self._units or joined in self._places:
                # Sokak/cadde adlarında küçük harfli sözlük eşleşmesi kabul edilmez ("arka sokağında")
                if allow_lower or tokens[chosen[-1]][0][0].isupper() or tokens[chosen[-1]][0].isdigit():
                    return chosen[-1]
        # Sözlükte yoksa: büyük harfli ad (ya da numaralı sokak); küçük harfli ad kabul edilmez
        chosen = []
        for i in words:
            token = tokens[i][0]
            if token[0].isupper() or token.isdigit():
                chosen.append(i)
            else:
                break
        if not chosen:
            return None
        # "Ana cadde kapalı": tek büyük harf cümle başından geliyorsa ad sayılmaz (anahtar sözcük de küçükse)
        if all(i in starts or tokens[i][0].isdigit() for i in chosen) and tokens[index][0][0].islower():
            return None
        if all(tokens[i][0].isdigit() for i in chosen) and not allow_lower:
            return chosen[-1]
        return chosen[-1] if any(tokens[i][0][0].isupper() for i in chosen) else None

    # --- çok sözcüklü ülke / bölge adı: "Güney Kore", "İç Anadolu'da" (yalnız büyük harfle) ---
    def _multiword(self, tokens, keys, taken):
        index = 0
        while index < len(tokens):
            for length in (3, 2):
                window = range(index, index + length)
                if index + length > len(tokens) or any(i in taken for i in window):
                    continue
                words = [tokens[i][0] for i in window]
                if not all(_is_word(w) and w[0].isupper() for w in words):
                    continue
                if any(w != _base(w) for w in words[:-1]):
                    continue
                if " ".join(keys[i] for i in window) in self._places:
                    last = tokens[index + length - 1]
                    yield EntitySpan("LOCATION", tokens[index][1], last[1] + len(_base(last[0]))), set(window)
                    index += length - 1
                    break
            index += 1

    # --- tek sözcüklü il / ilçe / ülke / bölge adı ---
    def _single(self, text, tokens, keys, starts, index):
        token, start, _ = tokens[index]
        if not _is_word(token):
            return None
        base = _base(token)
        has_apostrophe = base != token and len(token) > len(base) + 1
        if has_apostrophe and re.match(r"^l[ıiuü]", token[len(base) + 1 :]):
            return None  # "ABD'li", "İzmir'li": kişiyi niteleyen sıfat, yer adı değil
        capital = base[0].isupper()
        key = keys[index]
        entry = self._places.get(key)
        via_suffix = False
        if entry is None and not has_apostrophe and key not in self._collisions:
            for stem in stems(key, minimum=4):
                if stem in self._places:
                    entry, via_suffix = self._places[stem], True
                    break
        if entry is None:
            return None
        tier = entry["tier"]
        exact = _key_exact(base) == _key_exact(entry["name"]) if not via_suffix else True
        if not exact and tier != "free":  # "Can" ≠ "Çan": ASCII katlamasıyla eşleşme yalnız belirsiz olmayan adlarda
            return None
        next_key = keys[index + 1] if index + 1 < len(keys) else ""
        prev_key = keys[index - 1] if index > 0 and index not in starts else ""
        if next_key in HONORIFICS or prev_key in TITLES:
            return None
        admin_context = bool(ADMIN_CONTEXT.match(next_key))
        if tier == "calendar":  # "Aralık'ta", "Pazar günü": yalnız "Pazar ilçesi" gibi açık bağlamda
            if not admin_context:
                return None
        elif tier in ("context", "person"):
            if not ((has_apostrophe and capital) or admin_context):
                return None
        elif tier == "org":  # "Beşiktaş'ta" / "besiktasta" yer; çıplak "Beşiktaş" çoğunlukla kulüp
            if not ((has_apostrophe and capital) or via_suffix or admin_context):
                return None
        elif tier == "cap":
            if not capital or (index in starts and not has_apostrophe and not admin_context):
                return None
        else:  # free
            if not capital and not has_apostrophe:
                # Küçük harfli çıplak / bitişik ekli yazım ("ceyhana"); sözcük olan adlar "cap" düzeyinde elendi.
                if len(key) < 4 or key in self._collisions:
                    return None
        # "Adnan Menderes'i", "Ali Kepenek'ten": kişi adı olabilen yer adının önünde ad varsa kişidir
        if tier in ("person", "context") and index > 0 and index not in starts and (index - 1) not in starts:
            previous = tokens[index - 1][0]
            if _is_word(previous) and previous[0].isupper() and previous == _base(previous):
                return None
        # Daha büyük bir adın parçası mı? ("Selim Aydın", "İzmir Adliyesi", "Antalya Kurumlar Vergi Dairesi")
        if capital and not has_apostrophe and not via_suffix and not admin_context:
            if self._capital_neighbour(tokens, keys, starts, index):
                return None
        return EntitySpan("LOCATION", start, start + (len(base) if has_apostrophe else len(token)))

    def _capital_neighbour(self, tokens, keys, starts, index) -> bool:
        def foreign_capital(i: int) -> bool:
            token = tokens[i][0]
            return _is_word(token) and token[0].isupper() and keys[i] not in self._places

        if index + 1 < len(tokens) and (index + 1) not in starts and foreign_capital(index + 1):
            return True
        # Önceki sözcük cümle başındaysa ("Bugün Kadıköy", "Our Gaziantep") büyük harfi bilgi taşımaz.
        return index > 0 and index not in starts and (index - 1) not in starts and foreign_capital(index - 1)


# ---------------------------------------------------------------------------
# Tesis / bina adları (anotasyon kuralı: numarasız bina, apartman, plaza, park, kampüs adı LOC'tur)
# ---------------------------------------------------------------------------

_FACILITY_HEAD = re.compile(r"^(apartman\w*|apt|plaza\w*|park[ıi]\w*|kampüs\w*|kampus\w*|pasaj\w*|rezidans\w*|avm)$")
# Tamlama biçimi ("zümrüt apartmanı", "çınar parkında") adı belirtir; çıplak biçim ("apartmana", "kampüste")
# yalnız büyük harfli ad ya da kesme işaretiyle kabul edilir ("kargo apartmana bırakıldı" değil)
_FACILITY_COMPOUND = re.compile(r"^(apartman[ıi]|park[ıi]|kampüsü|kampusu|pasajı|pasaji|plaza|apt|rezidans|avm)")
_GENITIVE = re.compile(r"(ın|in|un|ün)$")
_FACILITY_COMPOUND_FIRST = frozenset({"iş", "is", "alışveriş", "alisveris"})
_FACILITY_COMPOUND_HEAD = re.compile(r"^(merkez\w*|han[ıi]\w*)$")
# Ad yerine geçmeyen niteleyiciler ("yan apartman", "eski kampüs", "çocuk parkı")
FACILITY_STOP = frozenset(
    "bu şu o yan karşı eski yeni bizim sizin onların benim senin her bir aynı büyük küçük arka ön üst alt "
    "çocuk oyun spor milli millet şehir kent ana merkez toplu sosyal halk ve ile de da ki en tüm diğer öbür "
    "ilk son hangi şirket okul site sitenin otopark".split()
)


class FacilityRecognizer:
    """Bina / tesis adlarını LOCATION olarak bulur; kurum tahmininin önüne geçer.

    Anotasyon kuralı (ANNOTATION_RULES.md): numarasız tek başına bina/apartman/plaza/park/kampüs adı LOC'tur;
    numaralı adresin hemen önündeyse adresin parçasıdır (ner.merge_facility_into_address).
    """

    authoritative_types = frozenset({"LOCATION"})

    def recognize(self, text: str) -> list[EntitySpan]:
        tokens = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        lowered = [_key_exact(_base(tok)) for tok, _, _ in tokens]
        spans: list[EntitySpan] = []
        for index, (token, start, _end) in enumerate(tokens):
            head = lowered[index]
            head_start = index
            if _FACILITY_COMPOUND_HEAD.match(head) and index > 0 and lowered[index - 1] in _FACILITY_COMPOUND_FIRST:
                head_start = index - 1
            elif not _FACILITY_HEAD.match(head):
                continue
            has_apostrophe = _base(token) != token
            first = self._name_start(tokens, lowered, head_start)
            if first is None:
                continue
            capital = any(tokens[i][0][:1].isupper() for i in range(first, head_start))
            if head_start == index and not _FACILITY_COMPOUND.match(head) and not (capital or has_apostrophe):
                continue
            spans.append(EntitySpan("LOCATION", tokens[first][1], start + len(_base(token))))
        return spans

    @staticmethod
    def _name_start(tokens, lowered, head_start):
        words = []
        cursor = head_start - 1
        while cursor >= 0 and len(words) < 3:
            token = tokens[cursor][0]
            if not _is_word(token) or _base(token) != token or lowered[cursor] in FACILITY_STOP:
                break
            words.append(cursor)
            cursor -= 1
        if not words:
            return None
        capital_run = []
        for i in words:
            if tokens[i][0][:1].isupper():
                capital_run.append(i)
            else:
                break
        if capital_run:
            return capital_run[-1]
        # Küçük harfli adda sınır belirsiz: yalnız tek sözcük ("kargo zümrüt apartmanı" → "zümrüt apartmanı");
        # ilgi ekli niteleyici ad değildir ("evin apartmanı", "okulun parkı")
        word = tokens[words[0]][0]
        if len(word) < 3 or (len(word) >= 4 and _GENITIVE.search(lowered[words[0]])):
            return None
        return words[0]
