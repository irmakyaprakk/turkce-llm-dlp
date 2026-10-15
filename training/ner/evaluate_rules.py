#!/usr/bin/env python3
"""Kural katmanını (RuleEntityRecognizer) etiketli JSONL'de ölçer; isteğe bağlı model tahminleriyle birleştirir.

Model gerektirmez. `--with-predictions` ile evaluate_model.py'nin `--dump-predictions`
çıktısı verilirse model + kural birleşimi (model öncelikli) de raporlanır.

Kullanım:
  python training/ner/evaluate_rules.py --data-dir data/ner/processed/realistic-v2 --splits dev test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.ner.evaluate_model import postprocess, token_coverage  # noqa: E402
from training.ner.train import entity_spans, span_metrics  # noqa: E402
from turkce_llm_dlp.ner import NER_TYPES, DetectionPipeline, EntitySpan  # noqa: E402
from turkce_llm_dlp.rules_ner import RuleEntityRecognizer  # noqa: E402

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
ENTITY_TYPES = ["ADDRESS", "LOCATION", "ORG", "PERSON"]


def spans_to_tags(text: str, tokens: list[str], spans) -> list[str]:
    """Karakter span'larını satırın token'larına BIO etiketi olarak işler."""
    offsets = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    if [text[s:e] for s, e in offsets] != tokens:
        raise ValueError("Metin ve token'lar hizalı değil")
    tags = ["O"] * len(tokens)
    for span in spans:
        first = True
        for index, (start, end) in enumerate(offsets):
            if start < span.end and end > span.start and tags[index] == "O":
                tags[index] = ("B-" if first else "I-") + span.type
                first = False
    return tags


def union(model_tags: list[str], rule_tags: list[str]) -> list[str]:
    """Model öncelikli birleşim: kural span'ı modelin hiçbir token'ıyla çakışmıyorsa eklenir."""
    merged = list(model_tags)
    for kind, start, end in entity_spans(rule_tags):
        if all(tag == "O" for tag in model_tags[start:end]):
            merged[start] = f"B-{kind}"
            for i in range(start + 1, end):
                merged[i] = f"I-{kind}"
    return merged


class ReplayRecognizer:
    """Önceden kaydedilmiş model tahminlerini (kelime düzeyi BIO) çalışma zamanındaki gibi
    karakter span'ı olarak döndürür; böylece değerlendirme üretimdeki DetectionPipeline'ı
    (TCKN/IBAN kuralları, birleştirme/kırpma, kural önceliği) birebir çalıştırır."""

    def __init__(self, predictions: dict[str, list[str]], rows: list[dict]):
        self._by_text = {row["text"]: predictions[row["id"]] for row in rows}

    def recognize(self, text: str):
        tags = self._by_text[text]
        offsets = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        return [EntitySpan(kind, offsets[s][0], offsets[e - 1][1]) for kind, s, e in entity_spans(tags)]


def pipeline_tags(pipeline: DetectionPipeline, row: dict) -> list[str]:
    """Üretim boru hattının NER türündeki tespitlerini satır token'larına BIO olarak işler.
    TCKN/IBAN/e-posta gibi kural tespitleri NER etiketi değildir; yalnızca çakışan
    model/kural span'larını bastırır (üretimdeki gibi)."""
    detections = [d for d in pipeline.detect(row["text"]) if d.type in NER_TYPES]
    return spans_to_tags(row["text"], row["tokens"], detections)


def pipeline_report(gold, pred) -> dict:
    return {"metrics": span_metrics(gold, pred, ENTITY_TYPES), "token_coverage": token_coverage(gold, pred)}


def report(gold, pred, rows) -> dict:
    post = [postprocess(r["tokens"], p) for r, p in zip(rows, pred)]
    return {
        "metrics": span_metrics(gold, pred, ENTITY_TYPES),
        "metrics_postprocessed": span_metrics(gold, post, ENTITY_TYPES),
        "token_coverage": token_coverage(gold, post),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--with-predictions", type=Path, help="evaluate_model.py --dump-predictions çıktısı (JSONL)")
    ap.add_argument(
        "--show-errors", action="store_true", help="kuralın yanlış/eksik bulduğu ADDRESS/ORG span'larını yaz"
    )
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    rows = [json.loads(line) for split in args.splits for line in (args.data_dir / f"{split}.jsonl").open()]
    recognizer = RuleEntityRecognizer()
    gold = [r["ner_tags"] for r in rows]
    rule = [spans_to_tags(r["text"], r["tokens"], recognizer.recognize(r["text"])) for r in rows]
    rules_pipeline = DetectionPipeline([RuleEntityRecognizer()])
    result = {
        "rows": len(rows),
        "rules_only": report(gold, rule, rows),
        "pipeline_rules_only": pipeline_report(gold, [pipeline_tags(rules_pipeline, r) for r in rows]),
    }
    # Kural katmanı yalnız ADDRESS/ORG üretir; o türlerdeki skorlar anlamlıdır.
    if args.with_predictions:
        preds = {}
        for line in args.with_predictions.open():
            item = json.loads(line)
            preds[item["id"]] = item["pred"]
        model = [preds[r["id"]] for r in rows]
        result["model_only"] = report(gold, model, rows)
        result["model_plus_rules"] = report(gold, [union(m, k) for m, k in zip(model, rule)], rows)
        # Üretimle birebir: DetectionPipeline (regex kuralları + kural katmanı + model, son işlem dahil)
        replay = ReplayRecognizer(preds, rows)
        result["pipeline_model_only"] = pipeline_report(
            gold, [pipeline_tags(DetectionPipeline([replay]), r) for r in rows]
        )
        hybrid = DetectionPipeline([RuleEntityRecognizer(), replay])
        hybrid_tags = [pipeline_tags(hybrid, r) for r in rows]
        result["pipeline_model_plus_rules"] = pipeline_report(gold, hybrid_tags)
        # Yazar grubuna / partiye göre (mühürlü testte her dosya bir yazar grubudur)
        batches = sorted({r.get("batch") for r in rows if r.get("batch")})
        result["pipeline_model_plus_rules_by_batch"] = {
            name: pipeline_report(
                [g for g, r in zip(gold, rows) if r.get("batch") == name],
                [t for t, r in zip(hybrid_tags, rows) if r.get("batch") == name],
            )
            for name in batches
        }
        if args.show_errors:
            for r, g, p in zip(rows, gold, hybrid_tags):
                for kind, s, e in sorted(set(entity_spans(g)) - set(entity_spans(p))):
                    print(f"HİBRİT KAÇTI {kind:8} | {' '.join(r['tokens'][s:e]):40} | {r.get('batch', '')}")
                for kind, s, e in sorted(set(entity_spans(p)) - set(entity_spans(g))):
                    print(f"HİBRİT FAZLA {kind:8} | {' '.join(r['tokens'][s:e]):40} | {r.get('batch', '')}")
    if args.show_errors:
        for r, g, p in zip(rows, gold, rule):
            gs = {s for s in entity_spans(g) if s[0] in ("ADDRESS", "ORG")}
            ps = {s for s in entity_spans(p) if s[0] in ("ADDRESS", "ORG")}
            for kind, s, e in sorted(gs - ps):
                print(f"KAÇTI  {kind:7} | {' '.join(r['tokens'][s:e])}")
            for kind, s, e in sorted(ps - gs):
                print(f"FAZLA  {kind:7} | {' '.join(r['tokens'][s:e])}")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
