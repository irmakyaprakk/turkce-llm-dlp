"""Prepare a local, address-only NER set from OpenStreetData's Turkey OSM extract.

The output is positive-only: each row contains one real address span and no
surrounding sentence context. Keep it separate from the WikiNER splits until it
has been reviewed for the intended use.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

# Use the same apostrophe-suffix tokenization as the NER training data.
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
FIELDS = ("postal_code", "city", "street", "house_number")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "data/ner/raw/openstreetmap-tr/TR-houses.tsv.gz"
    output_dir = root / "data/ner/processed/openstreetmap-tr"
    output_dir.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, ...]] = set()
    rows = []
    missing_required = Counter()
    missing_required_rows = 0
    duplicates = 0
    missing_postal_code = 0
    source_rows = 0

    with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream, delimiter="\t"):
            source_rows += 1
            values = {field: (record.get(field) or "").strip() for field in FIELDS}
            required = ("city", "street", "house_number")
            if any(not values[field] for field in required):
                missing_required_rows += 1
                for field in required:
                    if not values[field]:
                        missing_required[field] += 1
                continue
            if not values["postal_code"]:
                missing_postal_code += 1
            key = tuple(values[field].casefold() for field in FIELDS)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)

            address = f"{values['street']} No: {values['house_number']}"
            if values["postal_code"]:
                address += f", {values['postal_code']} {values['city']}"
            else:
                address += f", {values['city']}"
            tokens = [match.group() for match in TOKEN_PATTERN.finditer(address)]
            if not tokens:
                continue
            row_id = hashlib.sha256("\t".join(key).encode("utf-8")).hexdigest()[:20]
            tags = ["B-ADDRESS", *(["I-ADDRESS"] * (len(tokens) - 1))]
            rows.append(
                {
                    "id": f"osm-tr-{row_id}",
                    "source": "openstreetmap-tr-odbl",
                    "text": address,
                    "tokens": tokens,
                    "ner_tags": tags,
                }
            )

    out_path = output_dir / "addresses.jsonl"
    with out_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": 1,
        "source": "OpenStreetData Turkey houses extract (OpenStreetMap-derived)",
        "license": "ODbL-1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "source_rows": source_rows,
        "prepared_unique_addresses": len(rows),
        "duplicates_removed": duplicates,
        "rows_missing_required_fields_removed": missing_required_rows,
        "missing_required_field_counts": dict(sorted(missing_required.items())),
        "rows_missing_postal_code": missing_postal_code,
        "contains_sentence_context": False,
        "note": "Address-only positive examples; review before merging into NER training splits.",
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(out_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
