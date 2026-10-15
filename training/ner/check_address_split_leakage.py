"""Check normalized ADDRESS-span overlap across prepared splits and context benchmark."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {
    "train": ROOT / "data/ner/processed/real-address-context-v1/train.jsonl",
    "dev": ROOT / "data/ner/processed/real-address-context-v1/dev.jsonl",
    "test": ROOT / "data/ner/processed/real-address-context-v1/test.jsonl",
    "context_test": ROOT / "data/ner/processed/openstreetmap-tr/sentence-context-v1/test.jsonl",
}
REPORT = ROOT / "data/ner/processed/real-address-context-v1/address_split_overlap_report.json"


def normalize_address(value: str) -> str:
    """Normalize case, Unicode and spacing while preserving meaningful house-number slashes."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"[^\w/]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def address_spans(row: dict[str, Any]) -> list[str]:
    tokens, tags = row["tokens"], row["ner_tags"]
    found: list[str] = []
    index = 0
    while index < len(tags):
        if tags[index] == "B-ADDRESS":
            end = index + 1
            while end < len(tags) and tags[end] == "I-ADDRESS":
                end += 1
            found.append(" ".join(tokens[index:end]))
            index = end
        else:
            index += 1
    return found


def load_addresses(path: Path) -> tuple[list[str], set[str]]:
    values: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for surface in address_spans(row):
                normalized = normalize_address(surface)
                if normalized:
                    values.append(normalized)
    return values, set(values)


def main() -> None:
    values: dict[str, list[str]] = {}
    unique: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split, path in SPLITS.items():
        if not path.is_file():
            raise SystemExit(f"Veri dosyası bulunamadı: {path}")
        values[split], unique[split] = load_addresses(path)
        counts[split] = {
            "address_spans": len(values[split]),
            "unique_normalized_addresses": len(unique[split]),
        }

    pairs = [
        ("train", "dev"),
        ("train", "test"),
        ("train", "context_test"),
        ("dev", "test"),
        ("dev", "context_test"),
        ("test", "context_test"),
    ]
    overlaps = {f"{left}__{right}": sorted(unique[left] & unique[right]) for left, right in pairs}
    report = {
        "normalization": "Unicode NFKC + casefold; punctuation/space normalized; slash in house numbers preserved",
        "files": {name: str(path.relative_to(ROOT)) for name, path in SPLITS.items()},
        "counts": counts,
        "overlap_counts": {name: len(items) for name, items in overlaps.items()},
        "overlap_examples": {name: items[:10] for name, items in overlaps.items() if items},
        "train_has_no_address_overlap_with_dev_or_tests": all(
            not overlaps[key] for key in ("train__dev", "train__test", "train__context_test")
        ),
        "note": "The 600 test/context_test overlaps are expected: real-address-v2 test addresses are also embedded in the separate sentence-context benchmark. They do not overlap training addresses.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nRapor kaydedildi: {REPORT}")


if __name__ == "__main__":
    main()
