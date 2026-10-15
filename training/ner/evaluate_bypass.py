"""Measure TCKN capture and false-positive rates for three format evasions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from turkce_llm_dlp.detectors import detect, valid_tckn  # noqa: E402


def make_tckn(rng: random.Random) -> str:
    first_nine = str(rng.randrange(1, 10)) + "".join(str(rng.randrange(10)) for _ in range(8))
    digits = [int(char) for char in first_nine]
    tenth = (sum(digits[0:9:2]) * 7 - sum(digits[1:8:2])) % 10
    eleventh = (sum(digits) + tenth) % 10
    candidate = first_nine + str(tenth) + str(eleventh)
    assert valid_tckn(candidate)
    return candidate


def mutate(value: str, technique: str, rng: random.Random) -> str:
    if technique == "separators":
        separators = " -."
        return "".join(
            char + (rng.choice(separators) if index < len(value) - 1 else "") for index, char in enumerate(value)
        )
    if technique == "zero_width":
        position = rng.randrange(1, len(value))
        return value[:position] + "\u200b" + value[position:]
    if technique == "fullwidth":
        return value.translate(str.maketrans("0123456789", "０１２３４５６７８９"))
    raise ValueError(f"Bilinmeyen kaçınma tekniği: {technique}")


def is_detected(text: str, normalize: bool) -> bool:
    return any(
        item.type == "TCKN" and item.start == 0 and item.end == len(text) for item in detect(text, normalize=normalize)
    )


def evaluate(samples_per_class: int, seed: int) -> dict:
    if samples_per_class < 50:
        raise ValueError("Proje ölçütü gereği her kaçınma sınıfı için en az 50 örnek gerekli.")
    rng = random.Random(seed)
    results = {"schema_version": 1, "seed": seed, "samples_per_class": samples_per_class, "techniques": {}}
    for technique in ("separators", "zero_width", "fullwidth"):
        positives = []
        negatives = []
        for _ in range(samples_per_class):
            valid = make_tckn(rng)
            invalid = valid[:-1] + str((int(valid[-1]) + 1) % 10)
            positives.append(mutate(valid, technique, rng))
            negatives.append(mutate(invalid, technique, rng))
        metrics = {}
        for normalize in (False, True):
            true_positive = sum(is_detected(sample, normalize) for sample in positives)
            false_positive = sum(is_detected(sample, normalize) for sample in negatives)
            metrics["normalization_on" if normalize else "normalization_off"] = {
                "true_positive": true_positive,
                "false_negative": samples_per_class - true_positive,
                "capture_rate": true_positive / samples_per_class,
                "false_positive": false_positive,
                "true_negative": samples_per_class - false_positive,
                "false_positive_rate": false_positive / samples_per_class,
            }
        results["techniques"][technique] = metrics
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/bypass-evaluation.json"))
    args = parser.parse_args()
    try:
        report = evaluate(args.samples_per_class, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
