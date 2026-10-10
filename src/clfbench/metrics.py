"""Metrikler: doğruluk, macro-F1, kalibrasyon, seçici tahmin, gecikme, maliyet, bootstrap güven aralığı.

Kural: geçersiz/başarısız yanıtlar *sessizce atılmaz*; sıkı doğrulukta yanlış sayılır, ayrıca oranı raporlanır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

BOOT_N = 2000


@dataclass
class ModelMetrics:
    n: int = 0
    n_ok: int = 0
    error_rate: float = 0.0
    accuracy: float = 0.0           # sıkı: hatalı yanıtlar yanlış sayılır
    accuracy_valid: float = 0.0     # yalnız geçerli yanıtlar
    macro_f1: float = 0.0
    accuracy_ci: tuple[float, float] = (0.0, 0.0)
    macro_f1_ci: tuple[float, float] = (0.0, 0.0)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)  # satır: gerçek, sütun: tahmin (+ 'geçersiz')
    # kalibrasyon (yalnız olasılık/güven veren örneklerde)
    n_prob: int = 0
    brier: Optional[float] = None
    ece: Optional[float] = None
    mean_conf: Optional[float] = None
    conf_kinds: list[str] = field(default_factory=list)
    acc_at_cov80: Optional[float] = None       # güvene göre en iyi %80'in doğruluğu
    cov_at_acc95: Optional[float] = None       # %95 doğruluğu koruyan en yüksek kapsama
    # hız
    lat_p50: Optional[float] = None
    lat_p95: Optional[float] = None
    lat_p99: Optional[float] = None
    lat_mean: Optional[float] = None
    # maliyet
    tok_in_mean: Optional[float] = None
    tok_out_mean: Optional[float] = None
    tokens_estimated: bool = False
    cost_per_1000: Optional[float] = None
    cost_per_1m: Optional[float] = None
    retries_total: int = 0


def macro_f1_from(y_true: Sequence[str], y_pred: Sequence[Optional[str]], labels: Sequence[str]):
    """Etiket başına P/R/F1; tahmin None ise hiçbir sınıfta FP olmaz ama gerçek sınıfta FN olur."""
    out = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        if tp + fn:  # destek sıfırsa sınıf makro ortalamaya girmez
            f1s.append(f1)
    return (float(np.mean(f1s)) if f1s else 0.0), out


def _fast_macro_f1(t_idx: np.ndarray, p_idx: np.ndarray, k: int) -> float:
    """Vektörleştirilmiş macro-F1 (bootstrap için). p_idx == -1 geçersiz tahmindir."""
    f1s = []
    for c in range(k):
        tp = np.sum((t_idx == c) & (p_idx == c))
        fp = np.sum((t_idx != c) & (p_idx == c))
        fn = np.sum((t_idx == c) & (p_idx != c))
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_ci(t_idx: np.ndarray, p_idx: np.ndarray, k: int, seed: int = 0, n_boot: int = BOOT_N):
    rng = np.random.default_rng(seed)
    n = len(t_idx)
    if n == 0:
        return (0.0, 0.0), (0.0, 0.0)
    accs, f1s = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[b] = np.mean(t_idx[idx] == p_idx[idx])
        f1s[b] = _fast_macro_f1(t_idx[idx], p_idx[idx], k)
    lo, hi = 2.5, 97.5
    return ((float(np.percentile(accs, lo)), float(np.percentile(accs, hi))),
            (float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))))


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray, seed: int = 0, n_boot: int = BOOT_N):
    """A−B doğruluk farkı için eşlenmiş bootstrap: (fark, alt, üst, P(A>B))."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    d = correct_a.astype(float) - correct_b.astype(float)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = d[idx].mean()
    return (float(d.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)),
            float(np.mean(diffs > 0)))


def calibration(probs: np.ndarray, y_idx: np.ndarray, bins: int = 10):
    """Brier (çok sınıflı), ECE (en yüksek olasılık üzerinden), ortalama güven."""
    n, k = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_idx] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return brier, float(ece), float(conf.mean())


def selective(conf: np.ndarray, correct: np.ndarray):
    """Güvene göre sırala; (acc@%80 kapsama, %95 doğruluğu koruyan en büyük kapsama)."""
    order = np.argsort(-conf, kind="stable")
    c = correct[order].astype(float)
    n = len(c)
    if n == 0:
        return None, None
    cum = np.cumsum(c) / np.arange(1, n + 1)
    k80 = max(1, int(round(0.8 * n)))
    acc80 = float(cum[k80 - 1])
    ok = np.where(cum >= 0.95)[0]
    cov95 = float((ok[-1] + 1) / n) if len(ok) else 0.0
    return acc80, cov95


def percentile(a: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(a, q)) if len(a) else None


def compute_metrics(records: list[dict], labels: list[str], price_in: Optional[float],
                    price_out: Optional[float], seed: int = 0, n_boot: int = BOOT_N) -> ModelMetrics:
    """records: [{'true', 'pred', 'probs', 'conf', 'conf_kind', 'latency_ms', 'in_tok', 'out_tok',
    'tokens_estimated', 'retries', 'error'}]"""
    m = ModelMetrics(n=len(records))
    if not records:
        return m
    lab_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    y_true = [r["true"] for r in records]
    y_pred = [r["pred"] if r.get("error") is None else None for r in records]
    t_idx = np.array([lab_idx[t] for t in y_true])
    p_idx = np.array([lab_idx[p] if p in lab_idx else -1 for p in y_pred])
    ok = p_idx >= 0
    m.n_ok = int(ok.sum())
    m.error_rate = 1 - m.n_ok / m.n
    m.accuracy = float(np.mean(t_idx == p_idx))
    m.accuracy_valid = float(np.mean(t_idx[ok] == p_idx[ok])) if ok.any() else 0.0
    m.macro_f1, m.per_class = macro_f1_from(y_true, y_pred, labels)
    m.accuracy_ci, m.macro_f1_ci = bootstrap_ci(t_idx, p_idx, k, seed, n_boot)
    conf_mat = np.zeros((k, k + 1), dtype=int)
    for t, p in zip(t_idx, p_idx):
        conf_mat[t, p if p >= 0 else k] += 1
    m.confusion = conf_mat.tolist()
    # kalibrasyon
    have = [i for i, r in enumerate(records) if r.get("probs") and r.get("error") is None]
    if have:
        P = np.array([[records[i]["probs"].get(l, 0.0) for l in labels] for i in have])
        P = P / P.sum(axis=1, keepdims=True)
        m.n_prob = len(have)
        m.brier, m.ece, m.mean_conf = calibration(P, t_idx[have])
        m.conf_kinds = sorted({records[i].get("conf_kind", "native") for i in have})
        cc = P.max(axis=1)
        m.acc_at_cov80, m.cov_at_acc95 = selective(cc, (P.argmax(axis=1) == t_idx[have]))
    # hız (yalnız başarılı çağrılar)
    lats = [r["latency_ms"] for r in records if r.get("error") is None and r.get("latency_ms") is not None]
    if lats:
        m.lat_p50, m.lat_p95, m.lat_p99 = (percentile(lats, q) for q in (50, 95, 99))
        m.lat_mean = float(np.mean(lats))
    # maliyet
    ins = [r["in_tok"] for r in records if r.get("in_tok") is not None]
    outs = [r["out_tok"] for r in records if r.get("out_tok") is not None]
    m.retries_total = int(sum(r.get("retries") or 0 for r in records))
    m.tokens_estimated = any(r.get("tokens_estimated") for r in records)
    if ins:
        m.tok_in_mean = float(np.mean(ins))
    if outs:
        m.tok_out_mean = float(np.mean(outs))
    if price_in is not None and m.tok_in_mean is not None:
        per_call = m.tok_in_mean * price_in / 1e6 + (m.tok_out_mean or 0.0) * (price_out or 0.0) / 1e6
        m.cost_per_1000 = per_call * 1000
        m.cost_per_1m = per_call * 1e6
    return m


def tag_slices(records: list[dict], tags_by_id: dict[str, tuple[str, ...]], min_n: int = 5) -> dict[str, dict]:
    """Etiket (typo, ascii, short, multi, long) dilimlerine göre sıkı doğruluk."""
    buckets: dict[str, list[int]] = {}
    for r in records:
        ok = int(r.get("error") is None and r["pred"] == r["true"])
        for t in tags_by_id.get(r["id"], ()):
            buckets.setdefault(t, []).append(ok)
    return {t: {"n": len(v), "accuracy": float(np.mean(v))} for t, v in sorted(buckets.items()) if len(v) >= min_n}


def pareto_front(points: dict[str, tuple[float, ...]], maximize: tuple[bool, ...]) -> set[str]:
    """points: ad -> değerler; maximize: her boyut için True=büyük iyi. None içeren noktalar dışarıda kalır."""
    cand = {k: v for k, v in points.items() if all(x is not None for x in v)}
    front = set()
    for a, va in cand.items():
        dominated = False
        for b, vb in cand.items():
            if a == b:
                continue
            better_eq = all((y >= x) if mx else (y <= x) for x, y, mx in zip(va, vb, maximize))
            strictly = any((y > x) if mx else (y < x) for x, y, mx in zip(va, vb, maximize))
            if better_eq and strictly:
                dominated = True
                break
        if not dominated:
            front.add(a)
    return front
