#!/usr/bin/env python3
"""Birden fazla seed'in evaluate_rules.py çıktılarını ortalama ± örnek standart sapma olarak özetler.

Kullanım:
  python training/ner/aggregate_seeds.py runs/augmented_v3_seed42/final.json runs/augmented_v3_seed43/final.json ...
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, int | float) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if len(paths) < 2:
        sys.exit("En az iki seed sonucu verin.")
    runs = [flatten(json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    keys = sorted(set.intersection(*(set(r) for r in runs)))
    wanted = ("f1", "precision", "recall", "coverage")
    summary = {}
    for key in keys:
        if not any(w in key.rsplit(".", 1)[-1] for w in wanted):
            continue
        values = [r[key] for r in runs]
        summary[key] = {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4),
            "values": [round(v, 4) for v in values],
        }
    print(json.dumps({"seeds": [p.parent.name for p in paths], "metrics": summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
