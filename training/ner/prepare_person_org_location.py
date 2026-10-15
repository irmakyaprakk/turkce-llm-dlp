"""Prepare real-context WikiNER splits for the first PERSON/ORG/LOCATION model."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    input_dir = root / "data/ner/processed"
    output_dir = input_dir / "person-org-location-v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "dev", "test"):
        rows = []
        with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("source") == "turkish-wikiNER":
                    rows.append(row)
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[split] = len(rows)

    report = {
        "schema_version": 1,
        "dataset": "Turkish Wiki NER Dataset",
        "source": "turkish-wikiNER",
        "license": "CC-BY-SA-4.0",
        "entity_types": ["PERSON", "ORG", "LOCATION"],
        "address_examples_included": False,
        "rows": summary,
        "note": "Natural Wikipedia sentence context; this is a baseline for PERSON/ORG/LOCATION, not customer-message evaluation.",
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
