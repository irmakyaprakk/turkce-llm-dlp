"""evaluate_suite.py birim testleri (model/torch gerektirmez). tests/ klasörüne koy: pytest tests/test_evaluate_suite.py"""

import statistics

from training.ner.evaluate_suite import (
    aggregate,
    analyze,
    ascii_fold,
    evaluate_model,
    plan_windows,
    predict_rows,
    summarize,
    tr_lower,
    tr_upper,
)

TYPES = ["PERSON", "ORG", "LOCATION", "ADDRESS"]


class Fake:
    """Bağlamdan bağımsız sahte tahminci: alt-token sayısı 1, truncation'ı b[:budget] ile taklit eder."""

    entity_types = TYPES

    def __init__(self, budget=254, fn=lambda w: "O"):
        self.budget, self.fn = budget, fn

    def count_subwords(self, words):
        return [1] * len(words)

    def predict(self, batches):
        return [[self.fn(w) for w in b[: self.budget]] for b in batches]


def test_turkish_case_and_ascii():
    assert tr_lower("İstanbul IŞIK") == "istanbul ışık"
    assert tr_upper("istanbul ışık") == "İSTANBUL IŞIK"
    assert ascii_fold("Çağrı Işığı Öğüş") == "Cagri Isigi Ogus"
    assert ascii_fold(tr_lower("IŞIK")) == "isik"


def test_plan_windows_covers_all_with_overlap():
    wins = plan_windows([1] * 600, budget=50, overlap=10)
    assert wins[0][0] == 0 and wins[-1][1] == 600
    for (_s1, e1), (s2, e2) in zip(wins, wins[1:]):
        assert s2 == e1 - 10 and e2 > e1
    assert all(e - s <= 50 for s, e in wins)


def test_plan_windows_progress_when_overlap_large_and_empty_row():
    assert plan_windows([], 50, 10) == [(0, 0)]
    wins = plan_windows([1] * 30, budget=5, overlap=99)
    assert wins[-1][1] == 30 and len(wins) <= 30


def test_stitch_windowed_equals_full_for_context_free_model():
    words = [f"w{i}" for i in range(300)]
    fn = lambda w: "B-LOCATION" if int(w[1:]) % 7 == 0 else "O"
    full = [fn(w) for w in words]
    preds, stats = predict_rows(Fake(budget=60, fn=fn), [words], windowed=True, overlap=12)
    assert preds[0] == full and stats["words_lost"] == 0 and stats["rows_windowed"] == 1


def test_no_window_counts_lost_words_as_O():
    words = [f"w{i}" for i in range(300)]
    preds, stats = predict_rows(Fake(budget=100, fn=lambda w: "B-LOCATION"), [words], windowed=False, overlap=0)
    assert stats["words_lost"] == 200 and stats["rows_with_lost_words"] == 1
    assert preds[0][:100] == ["B-LOCATION"] * 100 and preds[0][100:] == ["O"] * 200


def test_false_alarm_and_null_support():
    rows = [{"id": "neg-1"}]
    res = analyze(rows, [["O", "O"]], [["B-ADDRESS", "I-ADDRESS"]], TYPES)
    assert res["false_alarm_rows"]["ADDRESS"] == 1 and res["spurious_spans"]["ADDRESS"] == 1
    assert res["false_alarm_ids"]["ADDRESS"] == ["neg-1"]
    assert res["token_confusion"]["O"]["ADDRESS"] == 2
    s = summarize(res, TYPES)
    assert s["overall_f1"] is None and s["address_f1"] is None and s["address_support"] == 0
    assert s["address_false_alarm_rate"] == 1.0 and s["address_precision"] == 0.0


def test_location_address_confusion_and_zero_recall():
    res = analyze([{"id": "a"}], [["B-LOCATION"]], [["B-ADDRESS"]], TYPES)
    assert res["token_confusion"]["LOCATION"]["ADDRESS"] == 1
    s = summarize(res, TYPES)
    assert s["location_recall"] == 0.0 and s["location_f1"] == 0.0 and s["overall_f1"] == 0.0


def test_aggregate_skips_none_and_uses_sample_std():
    agg = aggregate([{"x": 0.8}, {"x": 0.9}, {"x": None}, {"x": 1.0}])
    assert agg["x"]["n"] == 3 and abs(agg["x"]["mean"] - 0.9) < 1e-9
    assert abs(agg["x"]["std"] - statistics.stdev([0.8, 0.9, 1.0])) < 1e-9
    assert aggregate([{"x": None}])["x"] == {"mean": None, "std": None, "n": 0}


def _set(rows, variants):
    return {"name": "s", "rows": rows, "variants": variants}


def test_variants_expose_case_sensitivity():
    rows = [{"id": "r1", "tokens": ["Bursa'ya", "gidiyorum"], "ner_tags": ["B-LOCATION", "O"]}]
    fn = lambda w: "B-LOCATION" if w.startswith("Bursa") else "O"
    out = evaluate_model(Fake(fn=fn), [_set(rows, ["original", "lower", "upper"])], windowed=True, overlap=5)["s"]
    assert out["original"]["summary"]["location_recall"] == 1.0
    assert out["lower"]["summary"]["location_recall"] == 0.0
    assert out["upper"]["summary"]["location_recall"] == 0.0


def test_long_text_entity_found_only_with_windows():
    words = ["kelime"] * 300
    words[280] = "Bursa"
    tags = ["O"] * 300
    tags[280] = "B-LOCATION"
    rows = [{"id": "long", "tokens": words, "ner_tags": tags}]
    fn = lambda w: "B-LOCATION" if w == "Bursa" else "O"
    on = evaluate_model(Fake(budget=100, fn=fn), [_set(rows, ["original"])], True, 20)["s"]["original"]
    off = evaluate_model(Fake(budget=100, fn=fn), [_set(rows, ["original"])], False, 20)["s"]["original"]
    assert on["summary"]["location_recall"] == 1.0 and on["windowing"]["words_lost"] == 0
    assert off["summary"]["location_recall"] == 0.0 and off["windowing"]["words_lost"] == 200
