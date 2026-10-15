"""Place real OSM address records into clearly marked generated sentences."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.ner.prepare_data import validate_splits

# Use the same apostrophe-suffix tokenization as the NER training data.
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
TEMPLATES = (
    "Teslimat adresi: {address}.",
    "Faturada kullanılacak adres {address} olarak kaydedildi.",
    "Başvuru formuna şu adres yazıldı: {address}.",
    "Kargo şirketine verdiğim adres: {address}.",
    "Adres kaydını güncelleyin: {address}.",
    "İstenen konumun adresi {address}.",
    "Belgede adres alanında {address} yer alıyor.",
    "Ulaşım için verilen adres {address}.",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def make_row(address_row: dict, template_index: int) -> dict:
    template = TEMPLATES[template_index]
    before, marker, after = template.partition("{address}")
    if not marker or "{address}" in after:
        raise ValueError("Şablonda {address} tam olarak bir kez bulunmalı.")
    address = address_row["text"]
    text = before + address + after
    span_start, span_end = len(before), len(before) + len(address)
    tokens = []
    tags = []
    for match in TOKEN_PATTERN.finditer(text):
        tokens.append(match.group())
        inside_address = match.start() >= span_start and match.end() <= span_end
        tags.append(("I-ADDRESS" if any(tag != "O" for tag in tags) else "B-ADDRESS") if inside_address else "O")
    return {
        "id": f"{address_row['id']}-sentence-{template_index + 1}",
        "source": "openstreetmap-tr-generated-context-v1",
        "context_type": "template_generated",
        "context_is_synthetic": True,
        "text": text,
        "tokens": tokens,
        "ner_tags": tags,
    }


def main() -> None:
    source_dir = ROOT / "data/ner/processed/openstreetmap-tr/splits"
    output_dir = ROOT / "data/ner/processed/openstreetmap-tr/sentence-context-v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    template_counts = {}
    for split in ("train", "dev", "test"):
        source_rows = read_jsonl(source_dir / f"{split}.jsonl")
        rows = [make_row(row, index % len(TEMPLATES)) for index, row in enumerate(source_rows)]
        splits[split] = rows
        template_counts[split] = dict(Counter(row["id"].rsplit("-", 1)[-1] for row in rows))
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    labels = json.loads((ROOT / "training/ner/labels.json").read_text(encoding="utf-8"))
    validate_splits(splits, set(labels["bio_labels"]))
    report = {
        "schema_version": 1,
        "address_source": "OpenStreetData Turkey houses extract, derived from OpenStreetMap",
        "address_license": "ODbL-1.0",
        "context_generation": "Fixed Turkish sentence templates; one real address inserted per row",
        "context_is_synthetic": True,
        "address_strings_are_real_osm_records": True,
        "rows": {split: len(rows) for split, rows in splits.items()},
        "templates_per_split": template_counts,
        "warning": "These are generated contexts, not naturally occurring messages. Keep separate from natural-context evaluation and do not describe the sentence text as real user data.",
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
