"""Evaluate a saved Turkish token-classification model on prepared JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.ner.train import entity_spans, span_metrics

GAP_TOKENS = frozenset({",", ";", "/", "-", "–"})


def merge_adjacent(tokens: list[str], tags: list[str], kinds: tuple[str, ...] = ("ADDRESS",)) -> list[str]:
    """Aralarında yalnızca noktalama olan aynı türden bitişik varlıkları birleştirir.

    Çalışma zamanındaki DetectionPipeline birleştirmesinin kelime düzeyi karşılığıdır:
    "…No: 5" + "," + "X Mahallesi" iki ayrı ADDRESS yerine tek ADDRESS sayılır.
    """
    merged = list(tags)
    for index, tag in enumerate(merged):
        if not tag.startswith("B-") or tag[2:] not in kinds:
            continue
        kind = tag[2:]
        back = index - 1
        while back >= 0 and merged[back] == "O" and tokens[back] in GAP_TOKENS:
            back -= 1
        if back >= 0 and merged[back] in (f"B-{kind}", f"I-{kind}"):
            for gap in range(back + 1, index + 1):
                merged[gap] = f"I-{kind}"
    return merged


def token_coverage(true_labels: list[list[str]], predicted_labels: list[list[str]]) -> dict[str, Any]:
    """DLP metriği: altın varlık token'larının kaçı herhangi bir türle maskelenecek (O değil)?"""
    counts: dict[str, list[int]] = {}
    for gold, pred in zip(true_labels, predicted_labels):
        for gold_tag, pred_tag in zip(gold, pred):
            if gold_tag == "O":
                continue
            entry = counts.setdefault(gold_tag[2:], [0, 0])
            entry[0] += 1
            entry[1] += pred_tag != "O"
    total, covered = sum(v[0] for v in counts.values()), sum(v[1] for v in counts.values())
    result: dict[str, Any] = {"overall_coverage": covered / total if total else None}
    for kind, (gold_tokens, hit) in sorted(counts.items()):
        result[f"{kind.lower()}_coverage"] = hit / gold_tokens
        result[f"{kind.lower()}_tokens"] = gold_tokens
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 8:
        parser.error("batch-size >= 1 ve max-length >= 8 olmalı")
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Eğitim gereksinimlerini kur: training/ner/requirements.txt") from exc

    rows = read_jsonl(args.data_dir / f"{args.split}.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("Etiketleri kelime ofsetlerine hizalamak için fast tokenizer gerekli.")
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    true_labels: list[list[str]] = []
    predicted_labels: list[list[str]] = []
    kept_tokens: list[list[str]] = []
    truncated_rows = 0
    omitted_word_count = 0
    gold_entities_past_limit = 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [row["tokens"] for row in batch],
                is_split_into_words=True,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            word_predictions = (
                model(**{key: value.to(device) for key, value in encoded.items()}).logits.argmax(-1).cpu()
            )
            for index, row in enumerate(batch):
                gold: list[str] = []
                pred: list[str] = []
                previous_word = None
                for token_index, word_index in enumerate(encoded.word_ids(batch_index=index)):
                    if word_index is None or word_index == previous_word:
                        continue
                    gold.append(row["ner_tags"][word_index])
                    pred.append(id_to_label[int(word_predictions[index, token_index])])
                    previous_word = word_index
                full_encoding = tokenizer(
                    row["tokens"], is_split_into_words=True, truncation=False, add_special_tokens=True
                )
                full_words = {word_id for word_id in full_encoding.word_ids() if word_id is not None}
                kept_words = {word_id for word_id in encoded.word_ids(batch_index=index) if word_id is not None}
                missing = full_words - kept_words
                if missing:
                    truncated_rows += 1
                    omitted_word_count += len(missing)
                    first_omitted = min(missing)
                    gold_entities_past_limit += sum(
                        1 for _, _, end in entity_spans(row["ner_tags"]) if end > first_omitted
                    )
                true_labels.append(gold)
                predicted_labels.append(pred)
                kept_tokens.append([row["tokens"][w] for w in sorted(kept_words)])
    entity_types = [entity for entity in model.config.id2label.values() if entity != "O"]
    entity_types = sorted({tag.split("-", 1)[1] for tag in entity_types if "-" in tag})
    metrics = span_metrics(true_labels, predicted_labels, entity_types)
    merged_predictions = [merge_adjacent(t, p) for t, p in zip(kept_tokens, predicted_labels)]
    merged_metrics = span_metrics(true_labels, merged_predictions, entity_types)
    support = {
        entity.lower(): sum(1 for tags in true_labels for span in entity_spans(tags) if span[0] == entity)
        for entity in entity_types
    }
    result = {
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "rows": len(rows),
        "sources": sorted({row.get("source", "unknown") for row in rows}),
        "device": str(device),
        "max_length": args.max_length,
        "metrics": metrics,
        "metrics_merged_adjacent": merged_metrics,
        "token_coverage": token_coverage(true_labels, merged_predictions),
        "support": support,
        "metric_notes": "metrics: katı tam-span; merged_adjacent: noktalamayla ayrılmış bitişik ADDRESS parçaları "
        "birleştirilmiş (çalışma zamanı davranışı); token_coverage: maskelenen altın varlık token oranı. "
        "support=0 olan türlerin skorları anlamsızdır.",
        "truncation_diagnostics": {
            "rows_over_max_length": truncated_rows,
            "omitted_word_count": omitted_word_count,
            "gold_entities_with_tokens_after_limit": gold_entities_past_limit,
            "note": "Span metrics cover visible tokens only; use evaluate_suite.py for full-text overlapping-window evaluation.",
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
