"""Benchmark local detection and masking latency on synthetic-only text."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from turkce_llm_dlp.detectors import detect  # noqa: E402
from turkce_llm_dlp.masking import MaskingSession  # noqa: E402


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * quantile) + 0.999999) - 1))
    return ordered[index]


def summarize(values_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(statistics.median(values_ms), 4),
        "p95_ms": round(percentile(values_ms, 0.95), 4),
        "mean_ms": round(statistics.fmean(values_ms), 4),
    }


def benchmark(iterations: int, warmup: int) -> dict:
    if iterations < 10 or warmup < 0:
        raise ValueError("En az 10 ölçüm ve sıfır veya daha fazla ısınma adımı gerekli.")
    results = {}
    for target_length in (500, 5_000, 20_000):
        sensitive = "Örnek kayıt: TCKN 10000000146, e-posta kisi@example.com. "
        filler = "Bu sentetik ve hassas olmayan örnek metindir. "
        text = (filler * ((target_length // len(filler)) + 1))[: max(0, target_length - len(sensitive))] + sensitive
        timings: dict[str, list[float]] = {"detection": [], "detection_and_masking": []}

        for index in range(warmup + iterations):
            start = time.perf_counter_ns()
            detect(text)
            detected_ms = (time.perf_counter_ns() - start) / 1_000_000

            start = time.perf_counter_ns()
            MaskingSession().mask(text)
            masked_ms = (time.perf_counter_ns() - start) / 1_000_000

            if index >= warmup:
                timings["detection"].append(detected_ms)
                timings["detection_and_masking"].append(masked_ms)

        results[str(target_length)] = {
            "text_characters": len(text),
            "iterations": iterations,
            "ner_enabled": False,
            "measurements": {name: summarize(values) for name, values in timings.items()},
        }
    return {
        "schema_version": 1,
        "scope": "local core only; no HTTP, network provider, NER model, or real data",
        "runtime": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
        "warmup_iterations": warmup,
        "workloads": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("reports/latency-evaluation.json"))
    args = parser.parse_args()
    try:
        report = benchmark(args.iterations, args.warmup)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
