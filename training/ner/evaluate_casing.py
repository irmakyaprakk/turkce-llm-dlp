#!/usr/bin/env python3
"""Üretim boru hattını (model + kurallar + sözlük) üç yazım biçiminde ölçer: orijinal, küçük, BÜYÜK.

Günlük sohbet dilinde ad ve kurumlar çoğu zaman küçük harfle yazılır; model büyük harfe yaslanıyorsa
bu ölçüm farkı gösterir. Yalnız geliştirme verisinde çalıştırılır (mühürlü setler kullanılmaz).
Varsayılan çıktı yalnız sayılardır; --show-errors yanlış tahminleri terminale yazar.

Kullanım:
  python training/ner/evaluate_casing.py --model-dir models/ner/v4/augmented_v4_seed42/model \\
      --data-dir data/ner/processed/realistic-v2 --splits dev test --output-json rapor.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.ner.evaluate_model import read_jsonl  # noqa: E402
from training.ner.evaluate_rules import ENTITY_TYPES, pipeline_report, pipeline_tags  # noqa: E402
from training.ner.evaluate_suite import tr_lower, tr_upper  # noqa: E402
from training.ner.train import entity_spans  # noqa: E402
from turkce_llm_dlp.gazetteer import FacilityRecognizer, LocationGazetteer  # noqa: E402
from turkce_llm_dlp.ner import DetectionPipeline, HuggingFaceNERRecognizer  # noqa: E402
from turkce_llm_dlp.rules_ner import RuleEntityRecognizer  # noqa: E402

VARIANTS = {"original": lambda text: text, "lower": tr_lower, "upper": tr_upper}


def variant_row(row: dict, transform) -> dict:
    text = transform(row["text"])
    tokens = [transform(token) for token in row["tokens"]]
    if len(text) != len(row["text"]):
        raise ValueError(f"{row['id']}: yazım dönüşümü metin uzunluğunu değiştirdi")
    return {**row, "text": text, "tokens": tokens}


def _describe(words: list[str], spans) -> str:
    return ", ".join(f"{kind} «{' '.join(words[s:e])}»" for kind, s, e in sorted(spans, key=lambda x: x[1]))


def show_errors(name: str, rows: list[dict], gold: list[list[str]], pred: list[list[str]]) -> None:
    for row, g, p in zip(rows, gold, pred, strict=True):
        gold_spans, pred_spans = set(entity_spans(g)), set(entity_spans(p))
        if gold_spans == pred_spans:
            continue
        print(f"[{name}] {row['id']}: {row['text']}")
        print(f"   kaçan:  {_describe(row['tokens'], gold_spans - pred_spans) or '-'}")
        print(f"   fazla:  {_describe(row['tokens'], pred_spans - gold_spans) or '-'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--batch", nargs="*", default=[], help="yalnız bu partiler (ör. realistic_batch3)")
    ap.add_argument("--output-json", type=Path)
    ap.add_argument("--show-errors", choices=sorted(VARIANTS), nargs="*", default=[])
    args = ap.parse_args()
    if "sealed" in str(args.data_dir):
        sys.exit("Mühürlü setler bu betikle ölçülmez.")

    rows = [row for split in args.splits for row in read_jsonl(args.data_dir / f"{split}.jsonl")]
    if args.batch:
        rows = [row for row in rows if row.get("batch") in args.batch]
    model = HuggingFaceNERRecognizer.from_local_model(args.model_dir, inference_timeout_seconds=30)
    rules = [RuleEntityRecognizer(), LocationGazetteer(), FacilityRecognizer()]
    pipelines = {
        "hybrid": DetectionPipeline([*rules, model]),
        "model_only": DetectionPipeline([model]),
        "rules_only": DetectionPipeline(rules),
    }
    result: dict = {"rows": len(rows), "splits": args.splits, "batches": args.batch or "all", "variants": {}}
    for name, transform in VARIANTS.items():
        variant_rows = [variant_row(row, transform) for row in rows]
        gold = [row["ner_tags"] for row in variant_rows]
        result["variants"][name] = {}
        for label, pipeline in pipelines.items():
            pred = [pipeline_tags(pipeline, row) for row in variant_rows]
            result["variants"][name][label] = pipeline_report(gold, pred)
            if label == "hybrid" and name in args.show_errors:
                show_errors(name, variant_rows, gold, pred)
    model.close()

    print(f"{'yazım':<9} {'boru hattı':<11} {'F1':>6} " + " ".join(f"{kind[:4]:>6}" for kind in ENTITY_TYPES))
    for name, by_pipeline in result["variants"].items():
        for label, rep in by_pipeline.items():
            m = rep["metrics"]
            cells = " ".join(f"{m[f'{kind.lower()}_f1']:6.3f}" for kind in ENTITY_TYPES)
            print(f"{name:<9} {label:<11} {m['overall_f1']:6.3f} {cells}")
    if args.output_json:
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
