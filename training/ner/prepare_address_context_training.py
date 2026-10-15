"""Add generated sentence contexts around OSM addresses to the NER splits.

The original WikiNER rows and standalone-address rows are preserved. Context
examples are sampled from the corresponding OSM split, keeping held-out context
addresses out of training. The generated sentences are explicitly synthetic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.ner.prepare_data import validate_splits
from training.ner.train import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/ner/processed/real-address-v2")
    parser.add_argument(
        "--context-dir", type=Path, default=ROOT / "data/ner/processed/openstreetmap-tr/sentence-context-v1"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/ner/processed/real-address-context-v1")
    parser.add_argument("--train-context-count", type=int, default=6000)
    parser.add_argument("--dev-context-count", type=int, default=600)
    args = parser.parse_args()
    if args.train_context_count < 1 or args.dev_context_count < 1:
        parser.error("train/dev context sayıları pozitif olmalı")

    rows = {split: read_jsonl(args.base_dir / f"{split}.jsonl") for split in ("train", "dev", "test")}
    limits = {"train": args.train_context_count, "dev": args.dev_context_count, "test": 0}
    added: dict[str, int] = {}
    for split, limit in limits.items():
        candidates = read_jsonl(args.context_dir / f"{split}.jsonl")
        if limit > len(candidates):
            raise SystemExit(f"{split} için istenen {limit} bağlam örneği, mevcut {len(candidates)} örnekten fazla.")
        selected = candidates[:limit]
        rows[split].extend(selected)
        added[split] = len(selected)

    labels = json.loads((ROOT / "training/ner/labels.json").read_text(encoding="utf-8"))
    validate_splits(rows, set(labels["bio_labels"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": 1,
        "base_dataset": str(args.base_dir),
        "context_dataset": str(args.context_dir),
        "context_is_synthetic": True,
        "context_examples_added": added,
        "rows": {split: len(split_rows) for split, split_rows in rows.items()},
        "note": "The test split is unchanged. Evaluate separately on sentence-context-v1/test.jsonl; its OSM addresses are held out by source split.",
    }
    (args.output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
