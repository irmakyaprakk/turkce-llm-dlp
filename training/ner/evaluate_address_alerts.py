#!/usr/bin/env python3
"""Measure ADDRESS alerts on labeled examples and an unlabeled UCI review sample.

The report contains aggregate counts and hashes only. It never writes review text.
UCI detections are candidate alerts, not confirmed false positives, because the
reviews do not have entity annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

# Keep apostrophe suffixes attached, matching WikiNER and the manual-set builder.
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def model_predictions(model_dir: Path, token_rows: list[list[str]], batch_size: int, max_length: int):
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Gereksinimleri kur: pip install -r training/ner/requirements.txt") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("Kelime hizalaması için fast tokenizer gerekli.")
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    predictions: list[list[str]] = []
    truncated_rows = 0
    with torch.inference_mode():
        for start in range(0, len(token_rows), batch_size):
            batch = token_rows[start : start + batch_size]
            encoded = tokenizer(
                batch,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            pred_ids = logits.argmax(-1).cpu()
            for index, words in enumerate(batch):
                word_tags: list[str] = []
                previous_word = None
                seen_words: set[int] = set()
                for token_index, word_index in enumerate(encoded.word_ids(batch_index=index)):
                    if word_index is None or word_index == previous_word:
                        continue
                    seen_words.add(word_index)
                    word_tags.append(id_to_label[int(pred_ids[index, token_index])])
                    previous_word = word_index
                if len(seen_words) < len(words):
                    truncated_rows += 1
                predictions.append(word_tags)
    return predictions, str(device), truncated_rows


def bio_spans(tags: list[str], entity: str) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i in range(len(tags) + 1):
        tag = tags[i] if i < len(tags) else "O"
        prefix, kind = tag.split("-", 1) if "-" in tag else ("O", "")
        if start is not None and (prefix != "I" or kind != entity):
            spans.append((start, i))
            start = None
        if kind == entity and prefix == "B":
            start = i
        elif kind == entity and prefix == "I" and start is None:
            start = i
    return spans


def labeled_report(rows: list[dict[str, Any]], predicted: list[list[str]]) -> dict[str, Any]:
    per_category: dict[str, Counter] = {}
    counts = Counter()
    confusion = Counter()
    for row, pred in zip(rows, predicted):
        gold = row["ner_tags"][: len(pred)]
        category = row.get("category", "uncategorized")
        cat = per_category.setdefault(category, Counter())
        gold_address = bool(bio_spans(gold, "ADDRESS"))
        pred_address = bool(bio_spans(pred, "ADDRESS"))
        counts["sentences"] += 1
        counts["gold_address_sentences"] += int(gold_address)
        counts["predicted_address_sentences"] += int(pred_address)
        counts["address_span_predictions"] += len(bio_spans(pred, "ADDRESS"))
        counts["address_span_gold"] += len(bio_spans(gold, "ADDRESS"))
        counts["address_span_exact_matches"] += len(set(bio_spans(gold, "ADDRESS")) & set(bio_spans(pred, "ADDRESS")))
        counts["address_alerts_on_no_address_sentences"] += int(pred_address and not gold_address)
        cat["sentences"] += 1
        cat["gold_address_sentences"] += int(gold_address)
        cat["predicted_address_sentences"] += int(pred_address)
        for g, p in zip(gold, pred):
            gkind = g.split("-", 1)[-1] if g != "O" else "O"
            pkind = p.split("-", 1)[-1] if p != "O" else "O"
            confusion[f"{gkind}_to_{pkind}"] += 1
    return {
        "counts": dict(counts),
        "by_category": {key: dict(value) for key, value in sorted(per_category.items())},
        "token_confusion": {
            "gold_LOCATION_predicted_ADDRESS": confusion["LOCATION_to_ADDRESS"],
            "gold_ADDRESS_predicted_LOCATION": confusion["ADDRESS_to_LOCATION"],
        },
    }


def load_uci_sample(archive: Path, count: int, seed: int):
    with zipfile.ZipFile(archive) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith("dataset.txt")), None)
        if not name:
            raise ValueError("ZIP içinde dataset.txt bulunamadı")
        raw = zf.read(name).decode("utf-8-sig", errors="replace")
    # Preserve the original diagnostic sample: comments shorter than 20 chars
    # and all-caps rows were excluded. Count exclusions for transparent reporting.
    candidates = []
    seen = set()
    dropped_empty = 0
    dropped_headers = 0
    dropped_short = 0
    dropped_uppercase = 0
    duplicates = 0
    for index, line in enumerate(raw.splitlines()):
        text = line.strip()
        if not text:
            dropped_empty += 1
            continue
        if re.search(r"[-=]{3,}\s*$", text):
            dropped_headers += 1
            continue
        if len(text) < 20:
            dropped_short += 1
            continue
        if re.fullmatch(r"[A-ZÇĞİÖŞÜ0-9 &/.,()'-]+", text):
            dropped_uppercase += 1
            continue
        if text not in seen:
            candidates.append((index + 1, text))
            seen.add(text)
        else:
            duplicates += 1
    if count > len(candidates):
        raise ValueError(f"İstenen {count} kayıt, kullanılabilir {len(candidates)} kayıttan fazla")
    rng = random.Random(seed)
    sample = sorted(rng.sample(candidates, count), key=lambda item: item[0])
    digest = hashlib.sha256("\n".join(text for _, text in sample).encode("utf-8")).hexdigest()
    tokens = [[m.group() for m in TOKEN_RE.finditer(text)] for _, text in sample]
    source_info = {
        "nonempty_rows_before_deduplication": len(candidates) + duplicates,
        "unique_reviews_available": len(candidates),
        "duplicate_rows_removed": duplicates,
        "empty_rows_removed": dropped_empty,
        "separator_header_rows_removed": dropped_headers,
        "short_rows_removed_under_20_chars": dropped_short,
        "uppercase_only_rows_removed": dropped_uppercase,
        "sampling_population": "unique nonempty lines of at least 20 characters after separator and uppercase-only rows were excluded",
    }
    return sample, tokens, digest, source_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--labeled-jsonl", type=Path, help="Elle etiketlenmiş satırlar; ör. realistic-v1/test.jsonl")
    parser.add_argument("--uci-zip", type=Path, help="Etiketsiz UCI yorum arşivi")
    parser.add_argument("--uci-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--show-review-candidates",
        action="store_true",
        help="Colab çıktısında tahminli yorumları geçici göster; rapor dosyasına metin yazılmaz",
    )
    args = parser.parse_args()
    if not args.labeled_jsonl and not args.uci_zip:
        parser.error("--labeled-jsonl veya --uci-zip seçeneklerinden en az biri gerekli")

    report: dict[str, Any] = {"model_dir": str(args.model_dir), "seed": args.seed, "max_length": args.max_length}
    device = None
    if args.labeled_jsonl:
        rows = read_jsonl(args.labeled_jsonl)
        pred, device, truncated = model_predictions(
            args.model_dir, [r["tokens"] for r in rows], args.batch_size, args.max_length
        )
        report["labeled"] = {
            "path": str(args.labeled_jsonl),
            "rows": len(rows),
            "sha256": hashlib.sha256(args.labeled_jsonl.read_bytes()).hexdigest(),
            "truncated_rows": truncated,
            **labeled_report(rows, pred),
        }
    if args.uci_zip:
        sample, token_rows, sample_hash, source_info = load_uci_sample(args.uci_zip, args.uci_count, args.seed)
        pred, device, truncated = model_predictions(args.model_dir, token_rows, args.batch_size, args.max_length)
        entity_types = ("ADDRESS", "LOCATION", "PERSON", "ORG")
        sentence_counts = {entity: 0 for entity in entity_types}
        span_counts = {entity: 0 for entity in entity_types}
        review_candidates = []
        for (line_index, text), _tokens, tags in zip(sample, token_rows, pred):
            predicted_entities = []
            for entity in entity_types:
                spans = bio_spans(tags, entity)
                sentence_counts[entity] += int(bool(spans))
                span_counts[entity] += len(spans)
                if spans:
                    predicted_entities.append(entity)
            if predicted_entities:
                review_candidates.append(
                    {
                        "source_line_index": line_index,
                        "predicted_entities": predicted_entities,
                        "text": text,
                    }
                )
        n = len(sample)
        report["uci_unlabeled_sample"] = {
            "archive_sha256": hashlib.sha256(args.uci_zip.read_bytes()).hexdigest(),
            "sample_sha256": sample_hash,
            "seed": args.seed,
            **source_info,
            "sampled_reviews": n,
            "reviews_with_entity_alert": sentence_counts,
            "entity_span_alerts": span_counts,
            "address_alert_rate_upper_95_if_zero": (1 - 0.05 ** (1 / n)) if n and span_counts["ADDRESS"] == 0 else None,
            "truncated_reviews": truncated,
            "sampled_source_line_indices_with_alerts": [item["source_line_index"] for item in review_candidates],
            "interpretation": "Etiketsiz metindeki tahminler aday alarmlardır; doğrulanmış yanlış pozitif değildir. Üst güven sınırı yalnızca bu örnekleme alanı için yaklaşık üç kuralıdır.",
        }
        if args.show_review_candidates:
            # Print LOCATION candidates only to the interactive runtime; report remains text-free.
            print("\nGEÇİCİ İNCELEME — LOCATION tahminli yorumlar; metinleri kaydetmeyin:")
            for item in review_candidates:
                if "LOCATION" in item["predicted_entities"]:
                    print(json.dumps(item, ensure_ascii=False))
    report["device"] = device
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
