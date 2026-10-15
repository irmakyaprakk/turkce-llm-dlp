"""Birden çok test setinde ve yazım varyantlarında NER modellerini değerlendirir.

evaluate_model.py'den farkları:
  * Uzun metinde kaydırmalı pencere: 256 token'dan sonraki kelimeler sessizce düşmez.
    --no-window ile eski davranış (kesme) ölçülebilir; kaybolan kelimeler "O" sayılır, recall şişmez.
  * Yazım varyantları: aynı cümleler küçük harf, BÜYÜK HARF, aksansız ve küçük+aksansız hâliyle de ölçülür.
  * Yanlış alarm: gold'da o türden varlık olmayan cümlede tahmin edilen varlıklar sayılır; tür karışıklığı
    (ör. LOCATION ↔ ADDRESS) token düzeyinde matris olarak verilir.
  * Destek (support) sıfır olan türlerde F1/recall 0.0 yerine null yazılır.
  * Birden çok model (seed) verilirse ortalama ± standart sapma hesaplanır.

suite.json biçimi:
  {"sets": [
    {"name": "wiki_test", "path": "data/ner/processed/person-org-location-v1/test.jsonl"},
    {"name": "gercekci_test", "path": "data/ner/processed/realistic-v1/test.jsonl",
     "variants": ["lower", "upper", "ascii", "lower_ascii"]}
  ]}
Yollar çalışma dizinine göredir. "original" her set için otomatik eklenir.

Kullanım:
  python training/ner/evaluate_suite.py --model-dir runs/seed42/model runs/seed43/model \\
      --suite suite.json --output-json suite-results.json

Not: Çıktıya cümle metni yazılmaz; yalnızca sayılar ve satır id'leri vardır.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.ner.evaluate_model import read_jsonl  # noqa: E402
from training.ner.train import entity_spans, sha256_file, span_metrics  # noqa: E402

OUTSIDE = "O"
BIG = 10**9
MAX_FA_IDS = 50

_ASCII_TABLE = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def tr_lower(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


def tr_upper(text: str) -> str:
    return text.replace("i", "İ").replace("ı", "I").upper()


def ascii_fold(text: str) -> str:
    return text.translate(_ASCII_TABLE)


VARIANTS = {
    "original": lambda t: t,
    "lower": tr_lower,
    "upper": tr_upper,
    "ascii": ascii_fold,
    "lower_ascii": lambda t: ascii_fold(tr_lower(t)),
}


# ---------------------------------------------------------------- pencereleme
def plan_windows(counts: list[int], budget: int, overlap: int) -> list[tuple[int, int]]:
    """Kelime başına alt-token sayılarından (başlangıç, bitiş-dışlayan) pencereleri üretir."""
    n = len(counts)
    if n == 0:
        return [(0, 0)]
    windows = []
    start = 0
    while True:
        end, used = start, 0
        while end < n and (end == start or used + counts[end] <= budget):
            used += counts[end]
            end += 1
        windows.append((start, end))
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return windows


def stitch(n_words: int, windows: list[tuple[int, int]], window_preds: list[list[str]]) -> tuple[list[str], int]:
    """Pencere tahminlerini birleştirir; her kelime için pencere kenarına en uzak tahmin kazanır.
    Hiçbir pencerede tahmini olmayan kelimeler "O" olur ve kayıp sayılır."""
    out = [OUTSIDE] * n_words
    best = [-1] * n_words
    for (start, end), preds in zip(windows, window_preds):
        for k, label in enumerate(preds):
            i = start + k
            if i >= end or i >= n_words:
                break
            left = k if start > 0 else BIG
            right = (end - 1 - i) if end < n_words else BIG
            margin = min(left, right)
            if margin > best[i]:
                best[i], out[i] = margin, label
    return out, sum(1 for b in best if b < 0)


def predict_rows(predictor, token_rows: list[list[str]], windowed: bool, overlap: int):
    plans, flat = [], []
    for words in token_rows:
        wins = (
            plan_windows(predictor.count_subwords(words), predictor.budget, overlap) if windowed else [(0, len(words))]
        )
        plans.append(wins)
        flat.extend(words[s:e] for s, e in wins)
    flat_preds = predictor.predict(flat)
    preds, cursor = [], 0
    stats = {"rows": len(token_rows), "words_total": 0, "words_lost": 0, "rows_windowed": 0, "rows_with_lost_words": 0}
    for words, wins in zip(token_rows, plans):
        chunk = flat_preds[cursor : cursor + len(wins)]
        cursor += len(wins)
        out, lost = stitch(len(words), wins, chunk)
        preds.append(out)
        stats["words_total"] += len(words)
        stats["words_lost"] += lost
        stats["rows_windowed"] += int(len(wins) > 1)
        stats["rows_with_lost_words"] += int(lost > 0)
    return preds, stats


# -------------------------------------------------------------------- metrik
def type_of(tag: str) -> str | None:
    return None if tag == OUTSIDE else tag.split("-", 1)[-1]


def analyze(
    rows: list[dict[str, Any]], gold: list[list[str]], pred: list[list[str]], entity_types: list[str]
) -> dict[str, Any]:
    metrics = span_metrics(gold, pred, entity_types)
    names = [OUTSIDE] + entity_types
    confusion = {g: {p: 0 for p in names} for g in names}
    support = dict.fromkeys(entity_types, 0)
    predicted = dict.fromkeys(entity_types, 0)
    spurious = dict.fromkeys(entity_types, 0)
    fa_rows = dict.fromkeys(entity_types, 0)
    fa_ids: dict[str, list[str]] = {t: [] for t in entity_types}
    for index, (row, g, p) in enumerate(zip(rows, gold, pred)):
        gold_spans, pred_spans = entity_spans(g), entity_spans(p)
        for t in entity_types:
            gt = [s for s in gold_spans if s[0] == t]
            pt = [s for s in pred_spans if s[0] == t]
            support[t] += len(gt)
            predicted[t] += len(pt)
            gold_tokens = {i for _, a, b in gt for i in range(a, b)}
            spurious[t] += sum(1 for _, a, b in pt if not gold_tokens.intersection(range(a, b)))
            if pt and not gt:
                fa_rows[t] += 1
                if len(fa_ids[t]) < MAX_FA_IDS:
                    fa_ids[t].append(str(row.get("id", f"row-{index}")))
        for gt_tag, pt_tag in zip(g, p):
            confusion[type_of(gt_tag) or OUTSIDE][type_of(pt_tag) or OUTSIDE] += 1
    return {
        "metrics": metrics,
        "support": support,
        "predicted": predicted,
        "spurious_spans": spurious,
        "false_alarm_rows": fa_rows,
        "false_alarm_ids": fa_ids,
        "token_confusion": confusion,
        "rows": len(rows),
    }


def summarize(result: dict[str, Any], entity_types: list[str]) -> dict[str, Any]:
    """Düz ve karşılaştırılabilir özet; destek/tahmin yoksa değer null."""
    m, n = result["metrics"], max(result["rows"], 1)
    total_gold, total_pred = sum(result["support"].values()), sum(result["predicted"].values())
    row: dict[str, Any] = {
        "rows": result["rows"],
        "overall_f1": m["overall_f1"] if total_gold else None,
        "overall_recall": m["overall_recall"] if total_gold else None,
        "overall_precision": m["overall_precision"] if total_pred else None,
    }
    for t in entity_types:
        k = t.lower()
        sup, pred = result["support"][t], result["predicted"][t]
        row[f"{k}_support"] = sup
        row[f"{k}_f1"] = m[f"{k}_f1"] if sup else None
        row[f"{k}_recall"] = m[f"{k}_recall"] if sup else None
        row[f"{k}_precision"] = m[f"{k}_precision"] if pred else None
        row[f"{k}_false_alarm_rate"] = result["false_alarm_rows"][t] / n
    return row


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in summaries[0]:
        values = [s[key] for s in summaries if s.get(key) is not None]
        if not values:
            out[key] = {"mean": None, "std": None, "n": 0}
            continue
        out[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }
    return out


# ------------------------------------------------------------ HF tahmincisi
class HFPredictor:
    def __init__(self, model_dir: Path, max_length: int, batch_size: int):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise SystemExit("Gereksinimleri kur: training/ner/requirements.txt") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if not self.tokenizer.is_fast:
            raise SystemExit("Kelime hizalaması için fast tokenizer gerekli.")
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.id_to_label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_length, self.batch_size = max_length, batch_size
        self.budget = max_length - 2
        seen: list[str] = []
        for _, label in sorted(self.id_to_label.items()):
            t = type_of(label)
            if t and t not in seen:
                seen.append(t)
        self.entity_types = seen

    def count_subwords(self, words: list[str]) -> list[int]:
        counts = [0] * len(words)
        if not words:
            return counts
        encoded = self.tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        for word_id in encoded.word_ids():
            if word_id is not None:
                counts[word_id] += 1
        return counts

    def predict(self, batches: list[list[str]]) -> list[list[str]]:
        results: list[list[str]] = [[] for _ in batches]
        order = [i for i in sorted(range(len(batches)), key=lambda i: len(batches[i])) if batches[i]]
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            encoded = self.tokenizer(
                [batches[i] for i in idx],
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits
            ids = logits.argmax(-1).cpu().tolist()
            for j, i in enumerate(idx):
                labels = [OUTSIDE] * len(batches[i])
                previous = None
                for position, word_id in enumerate(encoded.word_ids(batch_index=j)):
                    if word_id is None or word_id == previous:
                        continue
                    labels[word_id] = self.id_to_label[ids[j][position]]
                    previous = word_id
                results[i] = labels
        return results


# ------------------------------------------------------------------ akış
def load_suite(path: Path) -> list[dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    sets = []
    names = set()
    for item in config["sets"]:
        name, data_path = item["name"], Path(item["path"])
        if name in names:
            raise SystemExit(f"Set adı tekrar ediyor: {name}")
        names.add(name)
        if not data_path.is_file():
            raise SystemExit(f"Set bulunamadı: {data_path}")
        variants = ["original"] + [v for v in item.get("variants", []) if v != "original"]
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            raise SystemExit(f"Bilinmeyen varyant {unknown}; geçerli: {sorted(VARIANTS)}")
        rows = read_jsonl(data_path)
        for index, row in enumerate(rows):
            if len(row.get("tokens", [])) != len(row.get("ner_tags", [])):
                raise SystemExit(f"{data_path}: {index}. satırda token/etiket sayısı farklı")
        sets.append(
            {"name": name, "path": str(data_path), "rows": rows, "variants": variants, "sha256": sha256_file(data_path)}
        )
    return sets


def evaluate_model(predictor, sets, windowed: bool, overlap: int) -> dict[str, Any]:
    entity_types = predictor.entity_types
    results: dict[str, Any] = {}
    for spec in sets:
        results[spec["name"]] = {}
        gold = [list(r["ner_tags"]) for r in spec["rows"]]
        for variant in spec["variants"]:
            transform = VARIANTS[variant]
            token_rows = [[transform(t) for t in r["tokens"]] for r in spec["rows"]]
            pred, stats = predict_rows(predictor, token_rows, windowed, overlap)
            result = analyze(spec["rows"], gold, pred, entity_types)
            result["windowing"] = stats
            result["summary"] = summarize(result, entity_types)
            results[spec["name"]][variant] = result
    return results


def _fmt(value: Any) -> str:
    return "  -  " if value is None else f"{value:.3f}"


def print_table(results: dict[str, Any], entity_types: list[str], title: str) -> None:
    keys = [t.lower() for t in entity_types]
    print(f"\n{title}")
    print(f"{'set/varyant':<34}{'satır':>6}{'genel':>7}" + "".join(f"{k[:7]:>9}" for k in keys) + f"{'adr.FA':>8}")
    for set_name, variants in results.items():
        for variant, res in variants.items():
            s = res["summary"]
            line = f"{set_name + '/' + variant:<34}{s['rows']:>6}{_fmt(s['overall_f1']):>7}"
            line += "".join(f"{_fmt(s[f'{k}_f1']):>9}" for k in keys)
            fa = s.get("address_false_alarm_rate")
            line += f"{_fmt(fa):>8}"
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=32, help="pencereler arası örtüşen kelime sayısı")
    parser.add_argument("--window", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_length < 16 or args.batch_size < 1 or args.overlap < 0:
        parser.error("max-length >= 16, batch-size >= 1, overlap >= 0 olmalı")

    sets = load_suite(args.suite)
    report: dict[str, Any] = {
        "suite": str(args.suite),
        "suite_sha256": sha256_file(args.suite),
        "settings": {"max_length": args.max_length, "windowed": args.window, "overlap": args.overlap},
        "sets": {
            s["name"]: {"path": s["path"], "sha256": s["sha256"], "rows": len(s["rows"]), "variants": s["variants"]}
            for s in sets
        },
        "models": {},
    }
    all_results = []
    entity_types: list[str] = []
    for model_dir in args.model_dir:
        predictor = HFPredictor(model_dir, args.max_length, args.batch_size)
        entity_types = predictor.entity_types
        results = evaluate_model(predictor, sets, args.window, args.overlap)
        report["models"][str(model_dir)] = {"device": str(predictor.device), "results": results}
        all_results.append(results)
        print_table(results, entity_types, f"Model: {model_dir}")
    if len(all_results) > 1:
        aggregated: dict[str, Any] = {}
        for set_name in all_results[0]:
            aggregated[set_name] = {}
            for variant in all_results[0][set_name]:
                aggregated[set_name][variant] = aggregate([r[set_name][variant]["summary"] for r in all_results])
        report["aggregate"] = aggregated
        print("\nOrtalama ± std (overall_f1):")
        for set_name, variants in aggregated.items():
            for variant, summary in variants.items():
                a = summary["overall_f1"]
                text = "  -  " if a["mean"] is None else f"{a['mean']:.3f} ± {_fmt(a['std']).strip()}"
                print(f"  {set_name}/{variant:<14} {text}  (n={a['n']})")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nYazıldı: {args.output_json}")


if __name__ == "__main__":
    main()
