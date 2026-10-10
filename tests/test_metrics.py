import numpy as np
import pytest

from clfbench.metrics import (bootstrap_ci, calibration, compute_metrics, macro_f1_from, paired_bootstrap_diff,
                              pareto_front, selective, tag_slices)

LABELS = ["a", "b", "c"]


def rec(true, pred, probs=None, err=None, lat=10.0, tin=100, tout=0, id_="x"):
    return {"id": id_, "true": true, "pred": pred, "probs": probs, "conf": None, "conf_kind": "native",
            "latency_ms": lat, "in_tok": tin, "out_tok": tout, "retries": 0, "error": err}


def test_accuracy_and_macro_f1_hand_computed():
    recs = [rec("a", "a"), rec("a", "b"), rec("b", "b"), rec("b", "b"), rec("c", "a"), rec("c", "c")]
    m = compute_metrics(recs, LABELS, None, None)
    assert m.n == 6 and m.accuracy == pytest.approx(4 / 6)
    # a: tp1 fp1 fn1 -> P.5 R.5 F.5 ; b: tp2 fp1 fn0 -> P2/3 R1 F.8 ; c: tp1 fp0 fn1 -> P1 R.5 F2/3
    assert m.macro_f1 == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m.per_class["b"]["recall"] == 1.0


def test_errors_count_as_wrong_and_are_reported():
    recs = [rec("a", "a"), rec("b", None, err="HTTP 500"), rec("c", "c")]
    m = compute_metrics(recs, LABELS, None, None)
    assert m.accuracy == pytest.approx(2 / 3)
    assert m.accuracy_valid == 1.0
    assert m.error_rate == pytest.approx(1 / 3) and m.n_ok == 2
    assert sum(map(sum, m.confusion)) == 3  # geçersiz de matrise girer
    assert m.confusion[1][3] == 1  # 'geçersiz' sütunu


def test_calibration_perfect_and_overconfident():
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    b, e, c = calibration(P, np.array([0, 1, 2]))
    assert b == pytest.approx(0) and e == pytest.approx(0) and c == pytest.approx(1)
    # hep yanlış ama %100 emin: Brier 2, ECE 1
    b, e, _ = calibration(P, np.array([1, 2, 0]))
    assert b == pytest.approx(2.0) and e == pytest.approx(1.0)


def test_selective_prediction():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 1, 1, 0, 0])
    acc80, cov95 = selective(conf, correct)
    assert acc80 == pytest.approx(0.75)  # en emin 4 örnek: 3/4
    assert cov95 == pytest.approx(0.6)  # ilk 3 örnek %100 doğru


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic():
    t = np.array([0] * 50 + [1] * 50)
    p = t.copy()
    p[:10] = 1
    (lo, hi), _ = bootstrap_ci(t, p, 2, seed=1, n_boot=500)
    acc = np.mean(t == p)
    assert lo <= acc <= hi and hi - lo < 0.2
    assert bootstrap_ci(t, p, 2, seed=1, n_boot=500) == bootstrap_ci(t, p, 2, seed=1, n_boot=500)


def test_paired_bootstrap_detects_clear_difference_and_ties():
    a = np.array([1] * 90 + [0] * 10)
    b = np.array([1] * 60 + [0] * 40)
    d, lo, hi, p = paired_bootstrap_diff(a, b, n_boot=500)
    assert d == pytest.approx(0.3) and lo > 0 and p > 0.99
    d, lo, hi, _ = paired_bootstrap_diff(a, a, n_boot=200)
    assert d == 0 and lo == 0 and hi == 0


def test_cost_from_tokens_and_prices():
    recs = [rec("a", "a", tin=1000, tout=10), rec("b", "b", tin=3000, tout=30)]
    m = compute_metrics(recs, LABELS, 0.5, 2.0)
    # ortalama 2000 giriş, 20 çıkış: (2000*0.5 + 20*2)/1e6 = 0.00104 -> 1000 karar = 1.04 USD
    assert m.cost_per_1000 == pytest.approx(1.04)
    assert compute_metrics(recs, LABELS, None, None).cost_per_1000 is None


def test_latency_percentiles_use_only_successful_calls():
    recs = [rec("a", "a", lat=l) for l in range(1, 101)] + [rec("a", None, err="x", lat=99999)]
    m = compute_metrics(recs, LABELS, None, None)
    assert m.lat_p50 == pytest.approx(50.5) and m.lat_p99 < 101


def test_pareto_front():
    pts = {"hızlı-kötü": (0.6, 10.0), "yavaş-iyi": (0.9, 500.0), "dominated": (0.55, 600.0), "orta": (0.8, 100.0)}
    assert pareto_front(pts, (True, False)) == {"hızlı-kötü", "yavaş-iyi", "orta"}


def test_tag_slices_min_n():
    recs = [rec("a", "a", id_=f"i{i}") for i in range(6)] + [rec("a", "b", id_="j")]
    tags = {f"i{i}": ("typo",) for i in range(6)}
    tags["j"] = ("rare",)
    s = tag_slices(recs, tags, min_n=5)
    assert "typo" in s and s["typo"]["accuracy"] == 1.0 and "rare" not in s


def test_macro_f1_ignores_absent_classes():
    f1, per = macro_f1_from(["a", "a"], ["a", "a"], ["a", "b"])
    assert f1 == 1.0 and per["b"]["support"] == 0
