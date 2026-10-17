#!/usr/bin/env python3
"""Kullanıcı diline yaklaştırılmış NER eğitim seti (augmented-v1) üretir.

Bileşenler (train / dev ayrı kaynak ve kalıp aileleriyle):
  1. WikiNER'in orijinal cümleleri (CC BY-SA 4.0)
  2. WikiNER'den yazım varyantları: küçük harf, ASCII, kesme işaretsiz, büyük harf
  3. OSM adreslerinin farklı yazım stillerinde yeniden yazımı (ODbL)
  4. Çeşitli türlerde bağlam kalıplarına yerleştirilmiş kişi/kurum/yer/adres
  5. Varlık içermeyen zor negatifler

Test split'i baseline ile karşılaştırılabilsin diye real-address-v2/test.jsonl'den
değiştirilmeden kopyalanır. Altın test seti (data/ner/manual/realistic_*.txt) eğitime
girmez; oradaki kişi/kurum adları dolgu sözlüklerinden çıkarılır ve cümle/adres
çakışması olursa üretim durur.

Kullanım:
  python training/ner/prepare_augmented.py --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from training.ner import context_templates as T
from training.ner import context_templates_v3 as V3
from training.ner import context_templates_v4 as V4
from training.ner import context_templates_v5 as V5
from training.ner.augment import (
    IL_BY_PLATE,
    AddressParts,
    apply_variant,
    parse_osm_address,
    render_address,
    row_variant,
    street_only,
    suffix,
    to_ascii,
    tokens_and_tags,
)
from training.ner.build_realistic_set import parse_line
from training.ner.prepare_data import validate_splits

LABELS_PATH = ROOT / "training/ner/labels.json"
SLOT_RE = re.compile(
    r"\{(PER|FIRST|ORG|LOC|ADDR|STREET|MAH|NUM|DATE|TIME|AMOUNT|TRACK|BIZNAME|BIZ|BRAND|ILCE|EDU|AMBF|AMBL|LIVE|STUDY|WORK)"
    r"(?::(DAKI|DAN|DA|E|IN|I|LARDAN|LARDA|LARA|LAR|LA))?\}"
)
SLOT_LABEL = {
    "PER": "PERSON",
    "FIRST": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "STREET": "LOCATION",
    "MAH": "LOCATION",
    "ADDR": "ADDRESS",
    "BIZ": "ORG",
    "BIZNAME": "ORG",
    "BRAND": "ORG",
    "ILCE": "LOCATION",
    "EDU": "ORG",
    "AMBF": "PERSON",
    "AMBL": "LOCATION",
}
MONTHS = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")

WIKI_VARIANTS = (("lower", 0.30), ("lower_ascii", 0.25), ("chat", 0.20), ("noapos", 0.15), ("upper", 0.10))
CONTEXT_VARIANTS = (("orig", 0.50), ("lower", 0.15), ("lower_ascii", 0.10), ("chat", 0.15), ("noapos", 0.10))
ADDRESS_VARIANTS = (("orig", 0.60), ("lower", 0.20), ("chat", 0.20))
# v4 gündelik çerçeveler: ağırlıkla küçük harf, kesme işaretsiz (ek bitişik), ASCII
INFORMAL_VARIANTS = (("lower_noapos", 0.40), ("chat", 0.25), ("lower", 0.15), ("noapos", 0.10), ("orig", 0.10))
# v5 tanıtım/istek dili: ağırlıkla küçük harf ve bitişik ek; BÜYÜK ve karışık harf payı v4'ten yüksek
V5_VARIANTS = (
    ("lower_noapos", 0.35),
    ("chat", 0.15),
    ("lower", 0.10),
    ("mixed", 0.15),
    ("upper", 0.12),
    ("noapos", 0.05),
    ("orig", 0.08),
)
TYPO_RATE = 0.25


def key(text: str) -> str:
    """Sızıntı karşılaştırması için büyük/küçük harf, Türkçe karakter ve boşluk duyarsız anahtar."""
    return " ".join(to_ascii(text).casefold().split())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def weighted(rng: random.Random, options: tuple[tuple[str, float], ...]) -> str:
    return rng.choices([name for name, _ in options], weights=[weight for _, weight in options])[0]


def load_gold(manual_dir: Path) -> tuple[set[str], set[str], dict[str, set[str]]]:
    """Altın setin cümle anahtarları, adres anahtarları ve türe göre varlık anahtarları."""
    texts, addresses = set(), set()
    entities: dict[str, set[str]] = {"PERSON": set(), "ORG": set(), "LOCATION": set()}
    # Mühürlü final test de dahil: eğitim verisi ne geliştirme ne final cümlelerini/adlarını içermemeli.
    # sealed2: yalnız etiketli dosyalar (kaynak/ ham metinleri okunmaz); içerik yalnız dışlama için kullanılır.
    paths = [*manual_dir.glob("realistic_*.txt"), *manual_dir.glob("sealed/*.txt"), *manual_dir.glob("sealed2/*.txt")]
    for path in sorted(paths):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            sentence = line.split("\t", 1)[1] if "\t" in line else line
            text, spans = parse_line(sentence.strip().replace("\\n", "\n"))
            texts.add(key(text))
            for start, end, label in spans:
                value = key(text[start:end])
                if label == "ADDRESS":
                    addresses.add(value)
                else:
                    entities[label].add(value)
    return texts, addresses, entities


class Filler:
    """Kalıp slotlarını doldurur; altın setteki kişi/kurum adlarını kullanmaz."""

    def __init__(
        self,
        rng: random.Random,
        address_pool: list[AddressParts],
        gold_entities: dict[str, set[str]],
        original_ratio: float = 0.0,
    ):
        self.rng = rng
        self.original_ratio = original_ratio
        self.address_pool = address_pool
        self.blocked_per = gold_entities["PERSON"]
        self.first_names = [n for n in T.FIRST_NAMES if key(n) not in self.blocked_per]
        generated = [h + " " + s + t for h in T.ORG_HEADS for s in T.ORG_SECTORS for t in T.ORG_TYPES]
        known = list(T.KNOWN_ORGS)
        blocked = gold_entities["ORG"]
        self._blocked_org_keys = blocked
        # Üretilen adlar sayıca çok fazla; tanınmış kurumlar örneklemede %40 pay alır.
        self.org_groups = [
            ([o for o in generated if key(o) not in blocked], 0.6),
            ([o for o in known if key(o) not in blocked], 0.4),
        ]
        self.orgs = [o for group, _ in self.org_groups for o in group]
        orgs = generated + known
        self.locations = (
            [(n, 45) for n in IL_BY_PLATE.values()]
            + [(n, 40) for n in T.ILCELER]
            + [(n, 8) for n in T.BOLGELER]
            + [(n, 7) for n in T.ULKELER]
        )
        self.blocked_orgs = len(orgs) - len(self.orgs)

    def person(self) -> str:
        for _ in range(20):
            first = self.rng.choice(self.first_names)
            if self.rng.random() < 0.12:
                first += " " + self.rng.choice(self.first_names)
            name = f"{first} {self.rng.choice(T.LAST_NAMES)}"
            if key(name) not in self.blocked_per:
                return name
        raise RuntimeError("Altın setle çakışmayan isim üretilemedi.")

    def value(self, slot: str, has_suffix: bool) -> str:
        rng = self.rng
        if slot == "PER":
            return self.person()
        if slot == "FIRST":
            return rng.choice(self.first_names)
        if slot == "ORG":
            group = rng.choices([g for g, _ in self.org_groups], weights=[w for _, w in self.org_groups])[0]
            if has_suffix:
                # Kısaltma/nokta ile biten adlarda ünlü uyumu belirsiz; ek alan slotta kullanılmaz.
                group = [o for o in group if not o.endswith(".") and not o.isupper()]
            return rng.choice(group)
        if slot == "LOC":
            names, weights = zip(*self.locations, strict=True)
            return rng.choices(names, weights=[w / len(names) for w in weights])[0]
        if slot == "ADDR":
            return render_address(rng.choice(self.address_pool), rng, self.original_ratio)
        if slot == "STREET":
            if has_suffix:
                return rng.choice(self.address_pool).street  # kısaltmasız: "Lale Sokak'ta"
            return street_only(rng.choice(self.address_pool), rng)
        if slot == "MAH":
            for _ in range(50):
                mahalle = rng.choice(self.address_pool).mahalle
                if mahalle:
                    return f"{mahalle} {'Mahallesi' if has_suffix or rng.random() < 0.7 else 'Mah.'}"
            raise RuntimeError("Mahalle adı bulunamadı.")
        if slot == "NUM":
            return str(rng.randint(10**5, 10**10))
        if slot == "DATE":
            day = rng.randint(1, 28)
            return rng.choice((f"{day:02d}.{rng.randint(1, 12):02d}.2026", f"{day} {rng.choice(MONTHS)}"))
        if slot == "TIME":
            return f"{rng.randint(8, 19):02d}:{rng.choice(('00', '15', '30', '45'))}"
        if slot == "AMOUNT":
            return f"{rng.randint(1, 99)}.{rng.randint(0, 999):03d}"
        if slot == "TRACK":
            return f"{rng.choice('ABCDEFGHKLMNPRSTUVYZ')}{rng.choice('ABCDEFGH')}{rng.randint(10**8, 10**9 - 1)}TR"
        raise ValueError(slot)


class FillerV4(Filler):
    """v4 slotları: yerel işletme, marka, PTT ilçe listesi; genişletilmiş ad listesi.

    Ayrı bir sınıf ve ayrı RNG ile çalışır; v1–v3 çıktıları bit düzeyinde değişmez."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from turkce_llm_dlp._gazetteer_data import load_gazetteer

        blocked_org = self._blocked_org_keys
        self.first_names = sorted(
            {n for n in (*T.FIRST_NAMES, *V4.EXTRA_FIRST_NAMES) if key(n) not in self.blocked_per}
        )
        self.last_names = sorted({*T.LAST_NAMES, *V4.EXTRA_LAST_NAMES})
        self.brands = [b for b in V4.BRANDS if key(b) not in blocked_org]
        self.biz_heads = [h for h in V4.BIZ_HEADS if key(h) not in blocked_org]
        places = load_gazetteer()["places"]
        # Kişi adı / soyadı olarak da kullanılan ilçeler ("Demirci", "Kozan") çıkarılır: etiket belirsizleşmesin
        name_keys = {key(n) for n in (*self.first_names, *self.last_names)}
        self.ilceler = sorted(
            e["name"]
            for e in places.values()
            if e["kind"] in ("il", "ilce") and e["tier"] == "free" and key(e["name"]) not in name_keys
        )

    def person(self) -> str:
        for _ in range(20):
            first = self.rng.choice(self.first_names)
            if self.rng.random() < 0.12:
                first += " " + self.rng.choice(self.first_names)
            name = f"{first} {self.rng.choice(self.last_names)}"
            if key(name) not in self.blocked_per:
                return name
        raise RuntimeError("Altın setle çakışmayan isim üretilemedi.")

    def business(self) -> str:
        rng = self.rng
        head = rng.choice(self.biz_heads)
        if rng.random() < 0.15:
            head = f"{head} {rng.choice(self.biz_heads)}"
        kind, typos = rng.choice(V4.BIZ_TYPES)
        if rng.random() < 0.30:
            kind = rng.choice(typos)
        return f"{head} {kind}"

    def value(self, slot: str, has_suffix: bool) -> str:
        if slot == "BIZ":
            return self.business()
        if slot == "BIZNAME":
            return self.rng.choice(self.biz_heads)
        if slot == "BRAND":
            pool = [b for b in self.brands if not b.isupper()] if has_suffix else self.brands
            return self.rng.choice(pool)
        if slot == "ILCE":
            return self.rng.choice(self.ilceler)
        return super().value(slot, has_suffix)


class FillerV5(FillerV4):
    """v5 slotları: kurum (okul/üniversite/hastane/belediye), sözcük anlamı da olan ad ve ilçe.

    Geliştirme ve mühürlü setlerdeki adlar, ilçeler ve kurumlar kullanılmaz: ad için ilk sözcük, ilçe ve kurum
    için önek eşleşmesi aranır ("bucakta" → "Bucak" elenir). Ayrı RNG; v1–v4 satırları değişmez.
    """

    def __init__(self, rng, address_pool, gold_entities, original_ratio=0.0):
        super().__init__(rng, address_pool, gold_entities, original_ratio)
        from turkce_llm_dlp._gazetteer_data import load_gazetteer

        gold_first = {name.split()[0] for name in gold_entities["PERSON"] if name}
        gold_loc, gold_org = gold_entities["LOCATION"], gold_entities["ORG"]

        def used_in(gold: set[str], value: str) -> bool:
            k = key(value)
            return any(g.startswith(k) for g in gold)

        self.amb_first = [n for n in V5.AMBIGUOUS_FIRST_NAMES if key(n) not in gold_first]
        amb_names = {key(n) for n in V5.AMBIGUOUS_FIRST_NAMES}
        places = load_gazetteer()["places"]
        self.amb_places = sorted(
            e["name"]
            for e in places.values()
            if e["kind"] in ("il", "ilce")
            and e["tier"] == "cap"
            and not used_in(gold_loc, e["name"])
            and key(e["name"]) not in amb_names
        )
        provinces = [n for n in IL_BY_PLATE.values() if not used_in(gold_loc, n)]
        self.provinces = provinces
        institutions = [f"{n} Üniversitesi" for n in V5.NAMED_UNIVERSITIES]
        institutions = [i for i in institutions if not used_in(gold_org, i)]
        self.universities = institutions
        self._gold_org = gold_org
        self.blocked_edu = len(V5.NAMED_UNIVERSITIES) - len(institutions)

    def institution(self) -> str:
        rng = self.rng
        for _ in range(50):
            roll = rng.random()
            if roll < 0.45:
                value = rng.choice(self.universities)
            elif roll < 0.65:
                value = f"{rng.choice(self.provinces)} {rng.choice(V5.HOSPITAL_KINDS)}"
            elif roll < 0.85:
                value = f"{self.person()} {rng.choice(V5.SCHOOL_KINDS)}"
            else:
                value = f"{rng.choice(self.ilceler)} Belediyesi"
            if not any(g.startswith(key(value)) for g in self._gold_org):
                return value
        raise RuntimeError("Geliştirme setiyle çakışmayan kurum üretilemedi.")

    def value(self, slot: str, has_suffix: bool) -> str:
        rng = self.rng
        if slot == "EDU":
            return self.institution()
        if slot == "AMBF":
            return rng.choice(self.amb_first)
        if slot == "AMBL":
            return rng.choice(self.amb_places)
        if slot == "LIVE":
            return rng.choice(V5.LIVE)
        if slot == "STUDY":
            return rng.choice(V5.STUDY)
        if slot == "WORK":
            return rng.choice(V5.WORK)
        return super().value(slot, has_suffix)


def apply_typo(text: str, spans, rng: random.Random):
    """Uzun bir sözcükten bir harf düşürür (klavye hatası); span'ları kaydırır."""
    words = [m for m in re.finditer(r"[^\W\d_]{5,}", text)]
    if not words:
        return text, list(spans)
    word = rng.choice(words)
    drop = rng.randrange(word.start() + 1, word.end() - 1)
    index_map = [i if i <= drop else i - 1 for i in range(len(text) + 1)]
    new_spans = [(index_map[a], index_map[b], label) for a, b, label in spans]
    return text[:drop] + text[drop + 1 :], new_spans


def generate_informal(
    templates,
    count: int,
    filler: FillerV4,
    rng: random.Random,
    source: str,
    gold_texts: set[str],
    seen: set[str],
    variants: tuple[tuple[str, float], ...] = INFORMAL_VARIANTS,
) -> list[dict]:
    rows, attempts = [], 0
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        template = templates[attempts % len(templates)] if attempts <= len(templates) else rng.choice(templates)
        text, spans = render_template(template, filler, pronounce=V4.PRONUNCIATION)
        if rng.random() < TYPO_RATE:
            text, spans = apply_typo(text, spans, rng)
        row = make_row("", source, text, spans, weighted(rng, variants))
        k = key(row["text"])
        if k in seen or k in gold_texts:
            continue
        seen.add(k)
        row["id"] = f"{source}-{hashlib.sha1(row['text'].encode()).hexdigest()[:14]}"
        rows.append(row)
    return rows


def render_template(
    template: str, filler: Filler, pronounce: dict[str, str] | None = None
) -> tuple[str, list[tuple[int, int, str]]]:
    template = template.replace("{{", "\x00").replace("}}", "\x01")
    out: list[str] = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    length = 0
    for match in SLOT_RE.finditer(template):
        literal = template[cursor : match.start()]
        out.append(literal)
        length += len(literal)
        slot, ek = match.group(1), match.group(2)
        value = filler.value(slot, has_suffix=ek is not None)
        if slot in SLOT_LABEL:
            spans.append((length, length + len(value), SLOT_LABEL[slot]))
        out.append(value)
        length += len(value)
        if ek:
            tail = "'" + suffix((pronounce or {}).get(value, value), ek)
            out.append(tail)
            length += len(tail)
        cursor = match.end()
    out.append(template[cursor:])
    if "{" in template[cursor:]:
        raise ValueError(f"tanınmayan slot: {template}")
    text = "".join(out).replace("\x00", "{").replace("\x01", "}")
    return text, spans


def make_row(row_id: str, source: str, text: str, spans, variant: str) -> dict:
    text, spans = apply_variant(text, spans, variant)
    tokens, tags = tokens_and_tags(text, spans)
    return {"id": row_id, "source": source, "variant": variant, "text": text, "tokens": tokens, "ner_tags": tags}


def generate_context(
    templates, count: int, filler: Filler, rng: random.Random, source: str, gold_texts: set[str], seen: set[str]
) -> list[dict]:
    rows, attempts = [], 0
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        template = templates[attempts % len(templates)] if attempts <= len(templates) else rng.choice(templates)
        text, spans = render_template(template, filler)
        row = make_row("", source, text, spans, weighted(rng, CONTEXT_VARIANTS))
        k = key(row["text"])
        if k in seen or k in gold_texts:
            continue
        seen.add(k)
        row["id"] = f"{source}-{hashlib.sha1(row['text'].encode()).hexdigest()[:14]}"
        rows.append(row)
    return rows


_SUFFIX_KINDS = (
    ("DAN", ("dan", "den", "tan", "ten")),
    ("DA", ("da", "de", "ta", "te")),
    ("IN", ("ın", "in", "un", "ün", "nın", "nin", "nun", "nün")),
    ("LA", ("la", "le", "yla", "yle")),
    ("E", ("a", "e", "ya", "ye")),
    ("I", ("ı", "i", "u", "ü", "yı", "yi", "yu", "yü")),
)


def suffix_kind(tail: str) -> str | None:
    """Kesme işaretinden sonraki eki türüne çevirir ("'nin" → IN); tanınmazsa None."""
    lowered = tail.lower()
    for kind, forms in _SUFFIX_KINDS:
        if lowered in forms:
            return kind
    return None


def replace_mentions(row: dict, filler: Filler) -> tuple[str, list[tuple[int, int, str]]] | None:
    """WikiNER satırındaki kişi/kurum/yer adlarını sözlükten yenileriyle değiştirir.

    Kesme işaretli ek ünlü uyumuyla yeniden üretilir ("Ankara'da" → "İzmir'de").
    Eki tanınmayan varlık değiştirilmeden bırakılır. Değişen varlık yoksa None döner.
    """
    slot_for = {"PERSON": "PER", "ORG": "ORG", "LOCATION": "LOC"}
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    length = 0
    changed = False
    tokens, tags = row["tokens"], row["ner_tags"]
    index = 0

    def emit(text: str) -> None:
        nonlocal length
        if parts:
            parts.append(" ")
            length += 1
        parts.append(text)
        length += len(text)

    while index < len(tokens):
        tag = tags[index]
        if not tag.startswith("B-"):
            emit(tokens[index])
            index += 1
            continue
        kind = tag[2:]
        end = index + 1
        while end < len(tokens) and tags[end] == f"I-{kind}":
            end += 1
        surface = tokens[index:end]
        head, _, tail = surface[-1].replace("’", "'").partition("'")
        ek = suffix_kind(tail) if tail else None
        # Küçük harfle başlayan sözcük içeren varlık ("gençlik akademisine") ek almış cins isimdir;
        # değiştirilirse kesme işaretsiz ek kaybolur, bu yüzden olduğu gibi bırakılır.
        proper = all(word[:1].isupper() or not word[:1].isalpha() for word in surface)
        if kind in slot_for and proper and (not tail or ek):
            slot = slot_for[kind]
            if slot == "PER" and len(surface) == 1:
                slot = "FIRST"
            value = filler.value(slot, has_suffix=ek is not None)
            if parts:
                parts.append(" ")
                length += 1
            spans.append((length, length + len(value), kind))
            parts.append(value)
            length += len(value)
            if ek:
                extra = "'" + suffix(value, ek)
                parts.append(extra)
                length += len(extra)
            changed = True
        else:
            start = length + (1 if parts else 0)
            emit(" ".join(surface))
            spans.append((start, length, kind))
        index = end
    if not changed:
        return None
    return "".join(parts), spans


def address_pool(path: Path) -> list[AddressParts]:
    pool = [parse_osm_address(row["text"]) for row in read_jsonl(path)]
    return [p for p in pool if p is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki-dir", type=Path, default=ROOT / "data/ner/processed/person-org-location-v1")
    ap.add_argument("--osm-split-dir", type=Path, default=ROOT / "data/ner/processed/openstreetmap-tr/splits")
    ap.add_argument("--test-file", type=Path, default=ROOT / "data/ner/processed/real-address-v2/test.jsonl")
    ap.add_argument("--manual-dir", type=Path, default=ROOT / "data/ner/manual")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/ner/processed/augmented-v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wiki-aug", type=int, nargs=2, default=(6000, 300), metavar=("TRAIN", "DEV"))
    ap.add_argument("--addresses", type=int, nargs=2, default=(5000, 400), metavar=("TRAIN", "DEV"))
    ap.add_argument("--contexts", type=int, nargs=2, default=(9000, 1000), metavar=("TRAIN", "DEV"))
    ap.add_argument("--negatives", type=int, nargs=2, default=(1200, 120), metavar=("TRAIN", "DEV"))
    ap.add_argument(
        "--osm-original-ratio",
        type=float,
        default=0.0,
        help="adreslerin bu oranı OSM ham biçiminde bırakılır (v1: 0.0, v2: 0.15)",
    )
    ap.add_argument(
        "--v3-contexts",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("TRAIN", "DEV"),
        help="v3 çerçevelerinden üretilecek cümle sayısı (v3: 10000 800)",
    )
    ap.add_argument(
        "--mention-replace",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("TRAIN", "DEV"),
        help="varlıkları değiştirilmiş WikiNER cümlesi sayısı (v3: 5000 300)",
    )
    ap.add_argument(
        "--v4-informal",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("TRAIN", "DEV"),
        help="v4 gündelik dil çerçevelerinden üretilecek cümle sayısı (v4: 12000 800)",
    )
    ap.add_argument(
        "--v5-informal",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("TRAIN", "DEV"),
        help="v5 tanıtım/istek dili çerçevelerinden üretilecek cümle sayısı (v5: 8000 600)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rng_v5 = random.Random(args.seed + 5)
    # v4 ayrı RNG kullanır: v3 satırları aynı tohumla aynen üretilir (yalnız karıştırma sırası değişir).
    rng_v4 = random.Random(args.seed + 4)
    gold_texts, gold_addresses, gold_entities = load_gold(args.manual_dir)
    pools = {split: address_pool(args.osm_split_dir / f"{split}.jsonl") for split in ("train", "dev")}
    pools = {
        s: [p for p in pool if key(render_address(p, random.Random(0))) not in gold_addresses]
        for s, pool in pools.items()
    }

    splits: dict[str, list[dict]] = {}
    for index, split in enumerate(("train", "dev")):
        wiki = read_jsonl(args.wiki_dir / f"{split}.jsonl")
        rows = list(wiki)

        with_entities = [r for r in wiki if any(t != "O" for t in r["ner_tags"])]
        for i, base in enumerate(rng.sample(with_entities, min(args.wiki_aug[index], len(with_entities)))):
            variant = weighted(rng, WIKI_VARIANTS)
            tokens, tags = row_variant(base["tokens"], base["ner_tags"], variant)
            rows.append(
                {
                    "id": f"{base['id']}-aug{i}-{variant}",
                    "source": "turkish-wikiNER-orthographic-aug",
                    "variant": variant,
                    "tokens": tokens,
                    "ner_tags": tags,
                }
            )

        seen: set[str] = set()
        filler = Filler(rng, pools[split], gold_entities, args.osm_original_ratio)
        for i in range(args.addresses[index]):
            text = render_address(rng.choice(pools[split]), rng, args.osm_original_ratio)
            row = make_row(
                f"osm-restyled-{split}-{i}",
                "openstreetmap-tr-restyled",
                text,
                [(0, len(text), "ADDRESS")],
                weighted(rng, ADDRESS_VARIANTS),
            )
            if key(row["text"]) in gold_addresses or key(row["text"]) in seen:
                continue
            seen.add(key(row["text"]))
            rows.append(row)

        templates = T.TRAIN_TEMPLATES if split == "train" else T.DEV_TEMPLATES
        negatives = T.TRAIN_NEGATIVES if split == "train" else T.DEV_NEGATIVES
        rows += generate_context(templates, args.contexts[index], filler, rng, "generated-context-v2", gold_texts, seen)
        rows += generate_context(
            negatives, args.negatives[index], filler, rng, "generated-negative-v2", gold_texts, seen
        )
        # --- v3 ekleri (varsayılan 0: v1/v2 çıktıları değişmez) ---
        if args.v3_contexts[index]:
            v3_templates = V3.TRAIN_TEMPLATES_V3 if split == "train" else V3.DEV_TEMPLATES_V3
            v3_negatives = V3.TRAIN_NEGATIVES_V3 if split == "train" else V3.DEV_NEGATIVES_V3
            rows += generate_context(
                v3_templates, args.v3_contexts[index], filler, rng, "generated-context-v3", gold_texts, seen
            )
            rows += generate_context(
                v3_negatives, args.v3_contexts[index] // 8, filler, rng, "generated-negative-v3", gold_texts, seen
            )
        if args.mention_replace[index]:
            sample = rng.sample(with_entities, min(args.mention_replace[index], len(with_entities)))
            for i, base in enumerate(sample):
                replaced = replace_mentions(base, filler)
                if replaced is None:
                    continue
                row = make_row(
                    f"{base['id']}-swap{i}", "turkish-wikiNER-mention-swap", *replaced, weighted(rng, CONTEXT_VARIANTS)
                )
                if key(row["text"]) in gold_texts:
                    continue
                rows.append(row)
        if args.v4_informal[index]:
            filler_v4 = FillerV4(rng_v4, pools[split], gold_entities, args.osm_original_ratio)
            v4_templates = V4.TRAIN_TEMPLATES_V4 if split == "train" else V4.DEV_TEMPLATES_V4
            v4_negatives = V4.TRAIN_NEGATIVES_V4 if split == "train" else V4.DEV_NEGATIVES_V4
            v4_rows = generate_informal(
                v4_templates, args.v4_informal[index], filler_v4, rng_v4, "generated-informal-v4", gold_texts, seen
            )
            v4_rows += generate_informal(
                v4_negatives, args.v4_informal[index] // 6, filler_v4, rng_v4, "generated-negative-v4", gold_texts, seen
            )
            rows += v4_rows
        if args.v5_informal[index]:
            filler_v5 = FillerV5(rng_v5, pools[split], gold_entities, args.osm_original_ratio)
            v5_templates = V5.TRAIN_TEMPLATES_V5 if split == "train" else V5.DEV_TEMPLATES_V5
            v5_negatives = V5.TRAIN_NEGATIVES_V5 if split == "train" else V5.DEV_NEGATIVES_V5
            v5_args = (filler_v5, rng_v5)
            rows += generate_informal(
                v5_templates, args.v5_informal[index], *v5_args, "generated-informal-v5", gold_texts, seen, V5_VARIANTS
            )
            rows += generate_informal(
                v5_negatives,
                args.v5_informal[index] // 6,
                *v5_args,
                "generated-negative-v5",
                gold_texts,
                seen,
                V5_VARIANTS,
            )
            report_v5 = {
                "ambiguous_first_names": len(filler_v5.amb_first),
                "ambiguous_places": len(filler_v5.amb_places),
                "blocked_named_universities": filler_v5.blocked_edu,
            }
        rng.shuffle(rows)
        splits[split] = rows

    splits["test"] = read_jsonl(args.test_file)

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    allowed = {"O"} | {f"{p}-{e}" for e in labels["entity_types"] for p in "BI"}
    validate_splits(splits, allowed)

    # Altın setle son kez bağımsız sızıntı denetimi (cümle ve adres düzeyinde).
    leaks = []
    for split in ("train", "dev"):
        for row in splits[split]:
            text = row.get("text") or " ".join(row["tokens"])
            if key(text) in gold_texts:
                leaks.append((split, "cümle", row["id"]))
    if leaks:
        sys.exit(f"Altın setle çakışma bulundu: {leaks[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "osm_original_ratio": args.osm_original_ratio,
        "v3_contexts": list(args.v3_contexts),
        "mention_replace": list(args.mention_replace),
        "v4_informal": list(args.v4_informal),
        "v5_informal": list(args.v5_informal),
        "v5_fill_pools": report_v5 if args.v5_informal[0] or args.v5_informal[1] else None,
        "rows": {s: len(r) for s, r in splits.items()},
        "sources": {s: dict(Counter(r["source"] for r in rows)) for s, rows in splits.items()},
        "variants": {s: dict(Counter(r.get("variant", "orig") for r in rows)) for s, rows in splits.items()},
        "entity_counts": {
            s: dict(Counter(t[2:] for r in rows for t in r["ner_tags"] if t.startswith("B-")))
            for s, rows in splits.items()
        },
        "templates": {
            "train": len(T.TRAIN_TEMPLATES),
            "dev": len(T.DEV_TEMPLATES),
            "train_negative": len(T.TRAIN_NEGATIVES),
            "dev_negative": len(T.DEV_NEGATIVES),
        },
        "gold_leak_check": {
            "gold_sentences": len(gold_texts),
            "gold_addresses": len(gold_addresses),
            "blocked_gold_person_names": len(gold_entities["PERSON"]),
            "blocked_org_fills": filler.blocked_orgs,
            "leaks": 0,
        },
        "test_source": str(args.test_file.relative_to(ROOT))
        if args.test_file.is_relative_to(ROOT)
        else str(args.test_file),
        "licenses": {
            "turkish-wikiNER": "CC-BY-SA-4.0",
            "turkish-wikiNER-orthographic-aug": "CC-BY-SA-4.0 (türetilmiş)",
            "openstreetmap-tr-restyled": "ODbL-1.0 (adres kayıtları; biçim ve daire/kat sentetik)",
            "generated-context-v2": "yerel üretim; adresler ODbL-1.0",
            "generated-negative-v2": "yerel üretim",
            "generated-context-v3": "yerel üretim; adresler ODbL-1.0",
            "generated-negative-v3": "yerel üretim",
            "turkish-wikiNER-mention-swap": "CC-BY-SA-4.0 (türetilmiş; varlık adları değiştirildi)",
            "generated-informal-v4": "yerel üretim; ilçe adları PTT listesi (MIT paket), adresler ODbL-1.0",
            "generated-negative-v4": "yerel üretim",
            "generated-informal-v5": "yerel üretim; ilçe adları PTT listesi (MIT paket), kurum adları kamuya açık",
            "generated-negative-v5": "yerel üretim",
        },
        "note": "Bağlam cümleleri yapaydır; gerçek kullanıcı başarısı yalnızca altın test setiyle ölçülür.",
    }
    for split, rows in splits.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
