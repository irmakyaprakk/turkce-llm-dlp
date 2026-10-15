#!/usr/bin/env python3
"""Altın test setini partilerden birleştirir; her partinin dev/test ayrımı sabit kalır.

Her `data/ner/manual/realistic_batch*.txt` dosyası build_realistic_set.py ile ayrı ayrı
etiket satırlarına çevrilir ve kendi içinde dev/test'e ayrılır. Böylece yeni bir parti
eklemek önceki partilerin test cümlelerini değiştirmez; eski deney sonuçları yeni
setin ilgili alt kümesiyle karşılaştırılabilir kalır. Her satıra `batch` alanı eklenir.

Kullanım:
  python training/ner/build_gold_set.py --out-dir data/ner/processed/realistic-v2 \\
      --against data/ner/processed/augmented-v2/train.jsonl data/ner/processed/augmented-v2/dev.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 1. partinin ayrımı realistic-v1 ile aynı kalmalı (seed 42, dev 20).
FIXED_DEV = {"realistic_batch1": 20}
DEV_RATIO = 0.15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual-dir", type=Path, default=ROOT / "data/ner/manual")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--against", nargs="*", default=[])
    args = ap.parse_args()

    builder = ROOT / "training/ner/build_realistic_set.py"
    splits: dict[str, list[dict]] = {"dev": [], "test": []}
    batches = {}
    for path in sorted(args.manual_dir.glob("realistic_batch*.txt")):
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        dev_n = FIXED_DEV.get(path.stem, max(1, round(len(lines) * DEV_RATIO)))
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(builder), str(path), "--out-dir", tmp, "--dev-n", str(dev_n)]
            cmd += ["--seed", str(args.seed)]
            if args.against:
                cmd += ["--against", *map(str, args.against)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode:
                sys.exit(f"{path.name} işlenemedi:\n{result.stderr or result.stdout}")
            for split in ("dev", "test"):
                rows = [json.loads(line) for line in (Path(tmp) / f"{split}.jsonl").open(encoding="utf-8")]
                for row in rows:
                    row["batch"] = path.stem
                splits[split] += rows
                batches.setdefault(path.stem, {})[split] = len(rows)
            batches[path.stem]["sha256_source"] = hashlib.sha256(path.read_bytes()).hexdigest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "batches": batches, "rows": {}, "entities": {}, "sha256": {}}
    for split, rows in splits.items():
        target = args.out_dir / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["rows"][split] = len(rows)
        manifest["entities"][split] = dict(Counter(t[2:] for r in rows for t in r["ner_tags"] if t.startswith("B-")))
        manifest["sha256"][split] = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
