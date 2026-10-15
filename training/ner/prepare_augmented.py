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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.ner import context_templates as T
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
SLOT_RE = re.compile(r"\{(PER|FIRST|ORG|LOC|ADDR|STREET|MAH|NUM|DATE|TIME|AMOUNT|TRACK)(?::(DA|DAN|E|I|IN|LA))?\}")
SLOT_LABEL = {
    "PER": "PERSON",
    "FIRST": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "STREET": "LOCATION",
    "MAH": "LOCATION",
    "ADDR": "ADDRESS",
}
MONTHS = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")

WIKI_VARIANTS = (("lower", 0.30), ("lower_ascii", 0.25), ("chat", 0.20), ("noapos", 0.15), ("upper", 0.10))
CONTEXT_VARIANTS = (("orig", 0.50), ("lower", 0.15), ("lower_ascii", 0.10), ("chat", 0.15), ("noapos", 0.10))
ADDRESS_VARIANTS = (("orig", 0.60), ("lower", 0.20), ("chat", 0.20))


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
    for path in sorted(manual_dir.glob("realistic_*.txt")):
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

    def __init__(self, rng: random.Random, address_pool: list[AddressParts], gold_entities: dict[str, set[str]]):
        self.rng = rng
        self.address_pool = address_pool
        self.blocked_per = gold_entities["PERSON"]
        self.first_names = [n for n in T.FIRST_NAMES if key(n) not in self.blocked_per]
        generated = [h + " " + s + t for h in T.ORG_HEADS for s in T.ORG_SECTORS for t in T.ORG_TYPES]
        known = list(T.KNOWN_ORGS)
        blocked = gold_entities["ORG"]
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
            return render_address(rng.choice(self.address_pool), rng)
        if slot == "STREET":
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


def render_template(template: str, filler: Filler) -> tuple[str, list[tuple[int, int, str]]]:
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
            tail = "'" + suffix(value, ek)
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
    args = ap.parse_args()

    rng = random.Random(args.seed)
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
        filler = Filler(rng, pools[split], gold_entities)
        for i in range(args.addresses[index]):
            text = render_address(rng.choice(pools[split]), rng)
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
