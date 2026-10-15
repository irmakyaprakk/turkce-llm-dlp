"""Replace synthetic address examples with real OSM address-only examples.

The OSM records are standalone address strings, not naturally occurring
sentences. They are kept in separate prepared splits and are not mixed into
the original dataset in place.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.ner.prepare_data import validate_splits


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    root = ROOT
    parser = argparse.ArgumentParser(description="WikiNER verisine gerçek OSM adresleri ekle.")
    parser.add_argument("--train-addresses", type=int, default=6000)
    parser.add_argument("--dev-addresses", type=int, default=600)
    parser.add_argument("--test-addresses", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ner/processed/real-address-v2"))
    args = parser.parse_args()
    per_split = {
        "train": args.train_addresses,
        "dev": args.dev_addresses,
        "test": args.test_addresses,
    }
    if any(count < 1 for count in per_split.values()):
        parser.error("Her split için adres sayısı en az 1 olmalı.")
    selected_count = sum(per_split.values())

    base_dir = root / "data/ner/processed"
    osm_path = base_dir / "openstreetmap-tr/addresses.jsonl"
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        split: [row for row in read_jsonl(base_dir / f"{split}.jsonl") if row.get("source") != "synthetic-address-v1"]
        for split in ("train", "dev", "test")
    }

    source_addresses = read_jsonl(osm_path)
    seen_tokens: set[tuple[str, ...]] = set()
    addresses = []
    for row in source_addresses:
        key = tuple(row["tokens"])
        if key not in seen_tokens:
            seen_tokens.add(key)
            addresses.append(row)
    total_addresses = len(addresses)
    random.Random(42).shuffle(addresses)
    train_end = int(total_addresses * 0.8)
    dev_end = train_end + (total_addresses - train_end) // 2
    osm_splits = {
        "train": addresses[:train_end],
        "dev": addresses[train_end:dev_end],
        "test": addresses[dev_end:],
    }
    for split, count in per_split.items():
        if count > len(osm_splits[split]):
            parser.error(f"İstenen {count} adres, OSM {split} bölümündeki {len(osm_splits[split])} adresten fazla.")
        splits[split].extend(osm_splits[split][:count])

    osm_split_dir = base_dir / "openstreetmap-tr/splits"
    osm_split_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in osm_splits.items():
        with (osm_split_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    osm_report = {
        "schema_version": 1,
        "source": "OpenStreetData Turkey houses extract, derived from OpenStreetMap",
        "license": "ODbL-1.0",
        "split_method": "deterministic shuffled 80/10/10 split, seed 42",
        "rows": {split: len(rows) for split, rows in osm_splits.items()},
        "contains_sentence_context": False,
    }
    (osm_split_dir / "quality_report.json").write_text(
        json.dumps(osm_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    schema = json.loads((root / "training/ner/labels.json").read_text(encoding="utf-8"))
    validate_splits(osm_splits, set(schema["bio_labels"]))
    validate_splits(splits, set(schema["bio_labels"]))

    for split, rows in splits.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": 1,
        "base_dataset": "Turkish Wiki NER Dataset (existing prepared split)",
        "address_source": "OpenStreetData Turkey houses extract, derived from OpenStreetMap",
        "address_license": "ODbL-1.0",
        "address_license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "address_rows_per_split": per_split,
        "full_osm_split_rows": {split: len(rows) for split, rows in osm_splits.items()},
        "available_unique_osm_addresses": total_addresses,
        "duplicate_address_texts_removed_before_split": len(source_addresses) - total_addresses,
        "selected_unique_osm_addresses": selected_count,
        "unused_osm_addresses_preserved_in_source_file": total_addresses - selected_count,
        "removed_source": "synthetic-address-v1",
        "contains_synthetic_address_examples": False,
        "contains_natural_sentence_context_for_osm": False,
        "note": "OSM rows are real standalone address strings. They teach address form, not how addresses occur in natural sentences. ODbL attribution/share-alike terms apply to the OSM-derived address database.",
        "rows": {split: len(rows) for split, rows in splits.items()},
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
