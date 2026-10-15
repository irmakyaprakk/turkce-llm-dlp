"""Prepare Turkish WikiNER CoNLL splits plus synthetic ADDRESS examples.

No data is downloaded automatically. Dataset users must obtain the source
under its license and review generated synthetic examples before training.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = Path(__file__).with_name("labels.json")
# Keep apostrophe suffixes on the word, matching WikiNER and the manual-set builder.
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)

SYNTHETIC_TEMPLATES = {
    "train": [
        "Sayın {person}, {org} başvurunuzu {address} adresinde aldı.",
        "{person} için yazışmalar {address} konumundaki {org} birimine gönderilsin.",
        "{org} sözleşmesini {person} adına {address} adresine hazırlayın.",
        "Teslimat bilgisi: {person}, {address}; ilgili kurum: {org}.",
        "{person} bugün {address} adresinde bulunan {org} ofisine gelecek.",
        "İK kaydı {person} adına {org} sisteminde, adres olarak {address} girilmiş.",
        "{org} tarafından {person} için oluşturulan dosya {address} adresine iletilecek.",
        "Lütfen {person} kaydını {address} adresiyle {org} hesabına ekleyin.",
    ],
    "dev": [
        "{person} evrakını {org} için şu adrese postalayın: {address}.",
        "Kayıt sahibi {person}; kurumu {org}; açık adresi {address} olarak güncellendi.",
    ],
    "test": [
        "Yeni müşteri {person} için {org} dosyasındaki posta adresi {address} oldu.",
        "{address} adresinde çalışan {org}, {person} adına bildirim hazırladı.",
    ],
}

SYNTHETIC_VALUES = {
    "train": {
        "people": ["Ece Kurgusal", "Mert Örneksoy", "Deniz Deneme", "Seda Yapay"],
        "orgs": ["Örnek Veri AŞ", "Kurgusal Teknoloji Ltd", "Deneme Lojistik AŞ"],
        "districts": ["Kurgusal Mahallesi", "Sınama Mahallesi", "Model Mahallesi"],
        "streets": ["Örnek Sokak", "Deneme Caddesi", "Yapay Bulvarı"],
        "cities": ["Ankara", "İzmir", "Bursa"],
        "number_base": 1000,
    },
    "dev": {
        "people": ["Ayla Uydurma", "Can Varsayım", "Nil Sentetik"],
        "orgs": ["Varsayım Yazılım AŞ", "Sentetik Hizmet Ltd"],
        "districts": ["Uydurma Mahallesi", "Varsayım Mahallesi"],
        "streets": ["Sentetik Sokak", "Varsayım Caddesi"],
        "cities": ["Eskişehir", "Konya"],
        "number_base": 20_000,
    },
    "test": {
        "people": ["Bora Kurgu", "Duru Tasarım", "Emre Hayali"],
        "orgs": ["Hayali Sistem AŞ", "Kurgu Operasyon Ltd"],
        "districts": ["Tasarım Mahallesi", "Hayali Mahallesi"],
        "streets": ["Kurgu Caddesi", "Tasarım Sokak"],
        "cities": ["Samsun", "Antalya"],
        "number_base": 40_000,
    },
}


def read_conll(path: Path) -> list[tuple[list[str], list[str]]]:
    sentences: list[tuple[list[str], list[str]]] = []
    tokens: list[str] = []
    tags: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: CoNLL satırı token ve etiket içermeli.")
            tokens.append(parts[0])
            tags.append(parts[1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def map_bio_tag(
    tag: str,
    source_mapping: dict[str, str],
    ignored_entity_tags: Counter[str] | None = None,
) -> str:
    if tag == "O":
        return "O"
    if "-" not in tag:
        raise ValueError(f"BIO etiketi geçersiz: {tag}")
    prefix, source_type = tag.split("-", 1)
    if prefix not in {"B", "I"}:
        raise ValueError(f"BIO öneki geçersiz: {tag}")
    target_type = source_mapping.get(source_type)
    if target_type is None and ignored_entity_tags is not None:
        ignored_entity_tags[source_type] += 1
    return f"{prefix}-{target_type}" if target_type else "O"


def convert_conll(
    path: Path,
    split: str,
    source_mapping: dict[str, str],
    source_tag_counts: Counter[str] | None = None,
    ignored_entity_tags: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, (tokens, tags) in enumerate(read_conll(path)):
        if source_tag_counts is not None:
            source_tag_counts.update(tags)
        mapped = [map_bio_tag(tag, source_mapping, ignored_entity_tags) for tag in tags]
        rows.append(
            {"id": f"wiki-{split}-{index:06d}", "source": "turkish-wikiNER", "tokens": tokens, "ner_tags": mapped}
        )
    return rows


def _tag_synthetic_sentence(sentence: str, entities: list[tuple[str, str]], row_id: str) -> dict[str, Any]:
    spans = []
    for kind, value in entities:
        start = sentence.index(value)
        spans.append((start, start + len(value), kind))
    spans.sort()
    tokens: list[str] = []
    tags: list[str] = []
    for match in TOKEN_PATTERN.finditer(sentence):
        token = match.group()
        overlapping = [span for span in spans if match.start() < span[1] and match.end() > span[0]]
        if not overlapping:
            label = "O"
        elif len(overlapping) > 1:
            raise ValueError(f"Çakışan sentetik varlık etiketleri: {sentence}")
        else:
            start, end, kind = overlapping[0]
            if match.start() < start or match.end() > end:
                raise ValueError(f"Sentetik varlık token sınırına denk gelmiyor: {sentence}")
            previous_is_same = bool(tags and tags[-1] in {f"B-{kind}", f"I-{kind}"})
            label = f"I-{kind}" if previous_is_same else f"B-{kind}"
        tokens.append(token)
        tags.append(label)
    return {"id": row_id, "source": "synthetic-address-v1", "tokens": tokens, "ner_tags": tags}


def synthetic_address_rows(split: str, count: int, seed: int) -> list[dict[str, Any]]:
    if split not in SYNTHETIC_TEMPLATES:
        raise ValueError(f"Bilinmeyen split: {split}")
    values = SYNTHETIC_VALUES[split]
    rng = random.Random(seed + {"train": 1, "dev": 2, "test": 3}[split])
    rows = []
    templates = SYNTHETIC_TEMPLATES[split]
    for index in range(count):
        district = rng.choice(values["districts"])
        street = rng.choice(values["streets"])
        city = rng.choice(values["cities"])
        address = f"{district}, {street} No: {values['number_base'] + index}, Daire: {index % 20 + 1}, {city}"
        person = rng.choice(values["people"])
        org = rng.choice(values["orgs"])
        sentence = rng.choice(templates).format(person=person, org=org, address=address)
        rows.append(
            _tag_synthetic_sentence(
                sentence, [("PERSON", person), ("ORG", org), ("ADDRESS", address)], f"address-{split}-{index:06d}"
            )
        )
    return rows


def validate_splits(splits: dict[str, list[dict[str, Any]]], allowed_labels: set[str]) -> None:
    seen: dict[tuple[str, ...], str] = {}
    seen_addresses: dict[str, tuple[str, str]] = {}

    def normalized_address(tokens: list[str]) -> str:
        # Match the realistic-set leak checker: preserve slash distinctions
        # in house numbers while ignoring case, Unicode form and punctuation.
        value = " ".join(tokens)
        value = value.casefold()
        value = re.sub(r"\s*/\s*", "/", value)
        value = re.sub(r"[^\w/]+", " ", value, flags=re.UNICODE)
        return " ".join(value.split())

    for split, rows in splits.items():
        for row in rows:
            tokens, tags = row["tokens"], row["ner_tags"]
            if len(tokens) != len(tags):
                raise ValueError(f"{row['id']}: token/etiket sayısı farklı.")
            unknown = set(tags) - allowed_labels
            if unknown:
                raise ValueError(f"{row['id']}: şemada olmayan etiket {sorted(unknown)}")
            previous = "O"
            active_address: list[str] = []
            address_spans: list[list[str]] = []
            for tag in tags:
                if tag.startswith("I-") and previous not in {f"B-{tag[2:]}", f"I-{tag[2:]}"}:
                    raise ValueError(f"{row['id']}: önceki varlık etiketiyle uyuşmayan {tag} etiketi.")
                previous = tag
            for token, tag in zip(tokens, tags):
                if tag == "B-ADDRESS":
                    if active_address:
                        address_spans.append(active_address)
                    active_address = [token]
                elif tag == "I-ADDRESS" and active_address:
                    active_address.append(token)
                else:
                    if active_address:
                        address_spans.append(active_address)
                        active_address = []
            if active_address:
                address_spans.append(active_address)
            for address_tokens in address_spans:
                key_address = normalized_address(address_tokens)
                if not key_address:
                    continue
                prior = seen_addresses.get(key_address)
                if prior is not None and prior[0] != split:
                    raise ValueError(
                        f"Train/dev/test arasında aynı normalize ADDRESS span'ı bulundu: "
                        f"{prior[0]} ve {split} ({prior[1]}, {row['id']})."
                    )
                seen_addresses[key_address] = (split, row["id"])
            key = tuple(tokens)
            prior = seen.get(key)
            if prior is not None and prior != split:
                raise ValueError(f"Train/dev/test arasında aynı cümle bulundu: {prior} ve {split}.")
            seen[key] = split


def quality_report(
    splits: dict[str, list[dict[str, Any]]],
    source_labels: dict[str, dict[str, dict[str, int]]] | None = None,
) -> dict[str, Any]:
    """Return aggregate-only diagnostics; never include sentence text or tokens."""
    report: dict[str, Any] = {"schema_version": 1, "splits": {}}
    for split, rows in splits.items():
        source_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        entity_counts: dict[str, int] = {}
        lengths = []
        for row in rows:
            source = row.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            lengths.append(len(row["tokens"]))
            previous = "O"
            for tag in row["ner_tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                if tag.startswith("B-"):
                    entity = tag[2:]
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
                elif tag.startswith("I-") and not previous.endswith(tag[1:]):
                    # validate_splits rejects this; retain defensive consistency.
                    raise ValueError(f"{row['id']}: bozuk BIO dizisi")
                previous = tag
        ordered_lengths = sorted(lengths)
        report["splits"][split] = {
            "rows": len(rows),
            "sources": dict(sorted(source_counts.items())),
            "tag_counts": dict(sorted(tag_counts.items())),
            "entity_counts": dict(sorted(entity_counts.items())),
            "token_length": {
                "min": ordered_lengths[0] if ordered_lengths else 0,
                "median": ordered_lengths[len(ordered_lengths) // 2] if ordered_lengths else 0,
                "max": ordered_lengths[-1] if ordered_lengths else 0,
            },
        }
        if source_labels and split in source_labels:
            report["splits"][split]["source_labels"] = source_labels[split]
    return report


def prepare(wiki_dir: Path, output_dir: Path, address_count: int, seed: int) -> dict[str, int]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    source_mapping = schema["source_label_mappings"]
    allowed_labels = set(schema["bio_labels"])
    paths = {"train": wiki_dir / "train.conll", "dev": wiki_dir / "dev.conll", "test": wiki_dir / "test.conll"}
    splits = {}
    source_labels = {}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"CoNLL split bulunamadı: {path}")
        source_tag_counts: Counter[str] = Counter()
        ignored_entity_tags: Counter[str] = Counter()
        splits[split] = convert_conll(path, split, source_mapping, source_tag_counts, ignored_entity_tags)
        source_labels[split] = {
            "tag_counts": dict(sorted(source_tag_counts.items())),
            "ignored_entity_tag_counts": dict(sorted(ignored_entity_tags.items())),
        }
        splits[split].extend(synthetic_address_rows(split, address_count, seed))
    validate_splits(splits, allowed_labels)
    report = quality_report(splits, source_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, rows in splits.items():
        output = output_dir / f"{split}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wiki-dir", type=Path, required=True, help="Lisanslı repo içindeki train/dev/test.conll klasörü"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ner/processed"))
    parser.add_argument("--address-count", type=int, default=200, help="Her split'e eklenecek sentetik adres cümlesi")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.address_count < 0:
        parser.error("--address-count negatif olamaz")
    counts = prepare(args.wiki_dir, args.output_dir, args.address_count, args.seed)
    print(json.dumps({"rows_per_split": counts, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
