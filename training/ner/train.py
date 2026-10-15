"""Fine-tune a Turkish BERT token classifier from prepared JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA_PATH = Path(__file__).with_name("labels.json")
DEFAULT_MODEL = "dbmdz/bert-base-turkish-cased"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: geçersiz JSONL") from exc
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("tokens"), list)
                or not isinstance(row.get("ner_tags"), list)
            ):
                raise ValueError(f"{path}:{line_number}: tokens ve ner_tags dizileri gerekli.")
            if len(row["tokens"]) != len(row["ner_tags"]):
                raise ValueError(f"{path}:{line_number}: token ve etiket sayısı aynı olmalı.")
            if not all(isinstance(token, str) and token for token in row["tokens"]):
                raise ValueError(f"{path}:{line_number}: tokenlar boş olmayan metin olmalı.")
            if not row["tokens"]:
                raise ValueError(f"{path}:{line_number}: boş cümle eğitim verisine alınamaz.")
            if not all(isinstance(tag, str) for tag in row["ner_tags"]):
                raise ValueError(f"{path}:{line_number}: etiketler metin olmalı.")
            rows.append(row)
    return rows


def entity_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    """Return (entity_type, start, end-exclusive) spans from BIO tags."""
    spans: set[tuple[str, int, int]] = set()
    active_type: str | None = None
    start = 0
    for index in range(len(tags) + 1):
        tag = tags[index] if index < len(tags) else "O"
        if tag == "O":
            next_type, prefix = None, "O"
        elif "-" in tag:
            prefix, next_type = tag.split("-", 1)
        else:
            prefix, next_type = "B", tag
        if active_type is not None and (prefix != "I" or next_type != active_type):
            spans.add((active_type, start, index))
            active_type = None
        if next_type is not None and prefix in {"B", "I"} and active_type is None:
            active_type, start = next_type, index
    return spans


def span_metrics(
    true_labels: list[list[str]], predicted_labels: list[list[str]], entity_types: list[str]
) -> dict[str, float]:
    """Compute exact entity-span precision, recall and F1 without seqeval."""
    counts = {entity: [0, 0, 0] for entity in entity_types}  # correct, predicted, gold
    for gold_tags, predicted_tags in zip(true_labels, predicted_labels):
        gold = entity_spans(gold_tags)
        predicted = entity_spans(predicted_tags)
        for entity in entity_types:
            gold_entity = {span for span in gold if span[0] == entity}
            predicted_entity = {span for span in predicted if span[0] == entity}
            counts[entity][0] += len(gold_entity & predicted_entity)
            counts[entity][1] += len(predicted_entity)
            counts[entity][2] += len(gold_entity)
    total = [sum(values[index] for values in counts.values()) for index in range(3)]

    def scores(correct: int, predicted: int, gold: int) -> tuple[float, float, float]:
        precision = correct / predicted if predicted else 0.0
        recall = correct / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    overall = scores(*total)
    metrics = dict(zip(("overall_precision", "overall_recall", "overall_f1"), overall))
    for entity, values in counts.items():
        precision, recall, f1 = scores(*values)
        metrics[f"{entity.lower()}_precision"] = precision
        metrics[f"{entity.lower()}_recall"] = recall
        metrics[f"{entity.lower()}_f1"] = f1
    return metrics


def load_training_dependencies():
    try:
        import numpy as np
        import torch
        import transformers
        from datasets import Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "NER eğitimi için training/ner/requirements.txt içindeki bağımlılıkları, GPU/CPU ortamına uygun PyTorch ile kur."
        ) from exc
    return (
        np,
        torch,
        transformers,
        Dataset,
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ner/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/turkce-dlp-berturk-ner"))
    parser.add_argument("--labels-path", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="0 varsayılanı eski davranışı korur; ör. 0.1")
    parser.add_argument("--fp16", action="store_true", help="karma hassasiyetli eğitim; varsayılan kapalıdır")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="JSONL/split/etiketleri doğrula; ML bağımlılığı veya model indirme gerektirmez",
    )
    args = parser.parse_args()
    if (
        min(args.epochs, args.train_batch_size, args.eval_batch_size) < 1
        or not (0 < args.learning_rate < 1)
        or args.max_length < 8
        or not (0.0 <= args.warmup_ratio < 1.0)
    ):
        parser.error("Eğitim parametreleri aralık dışında.")

    split_paths = {split: args.data_dir / f"{split}.jsonl" for split in ("train", "dev", "test")}
    for path in split_paths.values():
        if not path.is_file():
            raise SystemExit(f"Eğitim split'i bulunamadı: {path}; önce prepare_data.py çalıştır.")
    raw_rows = {split: read_jsonl(path) for split, path in split_paths.items()}
    empty_splits = [split for split, rows in raw_rows.items() if not rows]
    if empty_splits:
        raise SystemExit(f"Boş eğitim split'i: {', '.join(empty_splits)}")
    schema = json.loads(args.labels_path.read_text(encoding="utf-8"))
    labels = schema["bio_labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    unknown = {tag for rows in raw_rows.values() for row in rows for tag in row["ner_tags"]} - set(label_to_id)
    if unknown:
        raise SystemExit(f"Etiket şemasında bulunmayan etiket(ler): {sorted(unknown)}")
    # Reuse the preparation-time BIO and cross-split duplicate checks so a
    # hand-edited JSONL cannot bypass the training data contract.
    from training.ner.prepare_data import validate_splits

    try:
        validate_splits(raw_rows, set(labels))
    except ValueError as exc:
        raise SystemExit(f"Veri doğrulaması başarısız: {exc}") from exc
    if args.validate_only:
        summary = {
            "rows": {split: len(rows) for split, rows in raw_rows.items()},
            "sources": {
                split: sorted({row.get("source", "unknown") for row in rows}) for split, rows in raw_rows.items()
            },
            "sha256": {split: sha256_file(path) for split, path in split_paths.items()},
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    (
        np,
        torch,
        transformers,
        Dataset,
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    ) = load_training_dependencies()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("Alt-token/kelime hizalaması için fast tokenizer gereklidir.")

    def tokenize_batch(batch):
        encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=args.max_length)
        labels_batch = []
        for row_index, word_labels in enumerate(batch["ner_tags"]):
            word_ids = encoded.word_ids(batch_index=row_index)
            aligned = []
            previous_word = None
            for word_index in word_ids:
                if word_index is None:
                    aligned.append(-100)
                elif word_index != previous_word:
                    aligned.append(label_to_id[word_labels[word_index]])
                else:
                    # Score each original word once, avoiding overweighting
                    # words split into several WordPiece tokens.
                    aligned.append(-100)
                previous_word = word_index
            labels_batch.append(aligned)
        encoded["labels"] = labels_batch
        return encoded

    processed = {}
    for split, rows in raw_rows.items():
        dataset = Dataset.from_list(rows)
        processed[split] = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)

    # Surface gold entities that a single max_length window cannot see. This
    # audit does not silently treat omitted tail entities as correct negatives.
    truncation_audit = {}
    for split in ("dev", "test"):
        rows_over_limit = 0
        gold_entities_past_limit = 0
        omitted_words = 0
        for row in raw_rows[split]:
            full = tokenizer(row["tokens"], is_split_into_words=True, truncation=False, add_special_tokens=True)
            limited = tokenizer(row["tokens"], is_split_into_words=True, truncation=True, max_length=args.max_length)
            full_words = {word_id for word_id in full.word_ids() if word_id is not None}
            kept_words = {word_id for word_id in limited.word_ids() if word_id is not None}
            if len(kept_words) < len(full_words):
                rows_over_limit += 1
                omitted_words += len(full_words - kept_words)
                first_omitted = min(full_words - kept_words)
                gold_entities_past_limit += sum(1 for _, _, end in entity_spans(row["ner_tags"]) if end > first_omitted)
        truncation_audit[split] = {
            "rows_over_max_length": rows_over_limit,
            "gold_entities_with_tokens_after_limit": gold_entities_past_limit,
            "omitted_word_count": omitted_words,
            "note": "Trainer token metrics use only the visible single-window portion; use evaluate_suite.py for overlapping-window full-text evaluation.",
        }

    id_to_label = {index: label for label, index in label_to_id.items()}

    def compute_metrics(eval_prediction):
        logits, labels_array = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted = np.argmax(logits, axis=2)
        true_predictions = []
        true_labels = []
        for predictions, gold in zip(predicted, labels_array):
            keep = [index for index, value in enumerate(gold) if value != -100]
            true_predictions.append([id_to_label[int(predictions[index])] for index in keep])
            true_labels.append([id_to_label[int(gold[index])] for index in keep])
        return span_metrics(true_labels, true_predictions, schema["entity_types"])

    id2label = {index: label for index, label in enumerate(labels)}
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(labels), id2label=id2label, label2id=label_to_id
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_overall_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=args.fp16,
        warmup_ratio=args.warmup_ratio,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["dev"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model_dir = args.output_dir / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    results = {"dev": trainer.evaluate(processed["dev"], metric_key_prefix="dev")}
    results["test"] = trainer.evaluate(processed["test"], metric_key_prefix="test")
    for source, key in (
        ("turkish-wikiNER", "test_wiki"),
        ("synthetic-address-v1", "test_synthetic_address"),
        ("openstreetmap-tr-odbl", "test_openstreetmap_real_addresses"),
    ):
        source_rows = [row for row in raw_rows["test"] if row.get("source") == source]
        if source_rows:
            source_data = Dataset.from_list(source_rows)
            source_data = source_data.map(tokenize_batch, batched=True, remove_columns=source_data.column_names)
            results[key] = trainer.evaluate(source_data, metric_key_prefix=key)

    results["truncation_audit"] = truncation_audit
    (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    source_licenses = {
        "turkish-wikiNER": "CC-BY-SA-4.0",
        "synthetic-address-v1": "generated locally",
        "openstreetmap-tr-odbl": "ODbL-1.0",
        "openstreetmap-tr-generated-context-v1": "ODbL-1.0 address records; generated sentence context",
        "turkish-wikiNER-orthographic-aug": "CC-BY-SA-4.0 (derived)",
        "openstreetmap-tr-restyled": "ODbL-1.0 address records; synthetic formatting",
        "generated-context-v2": "generated locally; ODbL-1.0 address records",
        "generated-negative-v2": "generated locally",
    }
    used_sources = sorted({row.get("source", "unknown") for rows in raw_rows.values() for row in rows})
    manifest = {
        "base_model": args.base_model,
        "entity_types": schema["entity_types"],
        "bio_labels": labels,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "fp16": args.fp16,
        "warmup_ratio": args.warmup_ratio,
        "truncation_audit": truncation_audit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "data_sha256": {split: sha256_file(path) for split, path in split_paths.items()},
        "dataset_licenses": {source: source_licenses.get(source, "review required") for source in used_sources},
        "hub_upload": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
