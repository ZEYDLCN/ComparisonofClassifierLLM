"""Tek bir model için değerlendirme döngüsü: paralellik, önbellek/devam, ısınma, gecikme ölçümü."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import ModelSpec
from .core import Example, Prediction, ProviderError, TaskSpec, shuffled_order
from .prompts import PROMPT_VERSION, estimate_tokens, pick_shots
from .providers import make_provider


@dataclass
class ModelRun:
    key: str
    order_mode: str
    records: list[dict]
    setup_s: float = 0.0
    cold_ms: Optional[float] = None
    wall_s: float = 0.0
    concurrency: int = 1
    cached: int = 0
    meta: dict = field(default_factory=dict)


def cache_signature(spec: ModelSpec, order_mode: str, seed: int, shots_k: int, data_fp: str) -> str:
    blob = json.dumps([PROMPT_VERSION, spec.provider, spec.model, spec.base_url, spec.path,
                       spec.options, order_mode, seed, shots_k, data_fp], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _record(ex: Example, pred: Prediction, order: list[str] | None) -> dict:
    return {
        "id": ex.id, "true": ex.label, "pred": pred.label, "probs": pred.probs, "conf": pred.confidence,
        "conf_kind": pred.conf_kind, "latency_ms": pred.latency_ms, "in_tok": pred.input_tokens,
        "out_tok": pred.output_tokens, "tokens_estimated": pred.tokens_estimated,
        "retries": pred.retries, "error": pred.error,
    }


def _predict_one(provider, spec: ModelSpec, task: TaskSpec, ex: Example, order_mode: str, seed: int) -> dict:
    order = shuffled_order(task.label_keys, seed, ex.id) if order_mode == "shuffle" else None
    t0 = time.perf_counter()
    try:
        pred = provider.predict(ex.text, task, order)
        if pred.latency_ms is None:
            pred.latency_ms = (time.perf_counter() - t0) * 1000
        if pred.input_tokens is None and spec.price_in is not None and not spec.is_local:
            labels_text = " ".join(f"{k}: {v}" for k, v in task.labels.items())
            pred.input_tokens = estimate_tokens(task.instruction + " " + labels_text + " " + ex.text)
            pred.tokens_estimated = True
    except ProviderError as e:
        pred = Prediction(error=str(e)[:300])
    except Exception as e:  # beklenmeyen hata tüm koşuyu düşürmesin; açıkça kaydedilir
        pred = Prediction(error=f"{type(e).__name__}: {str(e)[:250]}")
    return _record(ex, pred, order)


def run_model(spec: ModelSpec, task: TaskSpec, eval_examples: list[Example], train: list[Example], *,
              order_mode: str = "canonical", seed: int = 0, shots_k: int = 0, data_fp: str = "",
              cache_dir: Path | None = None, use_cache: bool = True, warmup: int = 1,
              concurrency: int | None = None, show_progress: bool = True) -> ModelRun:
    provider = make_provider(spec)
    shots = pick_shots(train, task, shots_k, seed) if shots_k else []
    eval_ids = {e.id for e in eval_examples}
    shots = [s for s in shots if s.id not in eval_ids]

    cache: dict[str, dict] = {}
    cache_file: Path | None = None
    if cache_dir is not None and use_cache:
        sig = cache_signature(spec, order_mode, seed, shots_k, data_fp)
        cache_file = Path(cache_dir) / spec.key / f"{sig}.jsonl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    cache[r["id"]] = r

    pending = [e for e in eval_examples if e.id not in cache]
    meta_file = cache_file.with_suffix(".meta.json") if cache_file else None
    cold_ms = None
    if pending:
        t0 = time.perf_counter()
        provider.prepare(task, train, shots)
        setup_s = time.perf_counter() - t0
    else:
        # Hepsi önbellekte: model yüklemeye/indirmeye gerek yok. Kurulum/soğuk başlangıç değerleri ilk koşudan alınır.
        setup_s, cold_ms = 0.0, None
        if meta_file is not None and meta_file.exists():
            m0 = json.loads(meta_file.read_text(encoding="utf-8"))
            setup_s, cold_ms = m0.get("setup_s", 0.0), m0.get("cold_ms")
    conc = 1 if spec.is_local else max(1, concurrency or spec.concurrency)
    if pending and warmup > 0:
        # ısınma: TLS/bağlantı, tembel model indirme/yükleme, soğuk başlangıç; sonuçlara dahil edilmez, 'soğuk ms' olarak ayrı raporlanır
        for _ in range(warmup):
            w = _predict_one(provider, spec, task, pending[0], order_mode, seed)
            if w["error"] is None and cold_ms is None:
                cold_ms = w["latency_ms"]
            elif w["error"] and any(c in w["error"] for c in ("HTTP 401", "HTTP 402", "HTTP 403", "HTTP 404")):
                # Anahtar/bakiye/erişim/uç nokta hatası: tüm örnekler aynı hatayı alır, boşa istek atma
                provider.close()
                raise ProviderError(
                    f"Erişim/ödeme hatası ({w['error'][:200]}). Anahtarı, bakiyeyi, model adını ve base_url/path'i kontrol edin: "
                    f"'python -m clfbench ping {spec.key}'")

    lock = threading.Lock()
    fh = cache_file.open("a", encoding="utf-8") if cache_file and pending else None
    results: dict[str, dict] = dict(cache)
    t_wall = time.perf_counter()
    try:
        bar = tqdm(total=len(eval_examples), initial=len(cache), desc=f"{spec.key[:28]:28s}",
                   disable=not show_progress, leave=False)
        if conc == 1:
            for ex in pending:
                r = _predict_one(provider, spec, task, ex, order_mode, seed)
                _store(r, results, fh, lock)
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=conc) as pool:
                futs = [pool.submit(_predict_one, provider, spec, task, ex, order_mode, seed) for ex in pending]
                try:
                    for f in as_completed(futs):
                        _store(f.result(), results, fh, lock)
                        bar.update(1)
                except KeyboardInterrupt:
                    for f in futs:
                        f.cancel()
                    raise
        bar.close()
    finally:
        if fh:
            fh.close()
        provider.close()
    wall_s = time.perf_counter() - t_wall
    if meta_file is not None and pending:
        meta_file.write_text(json.dumps({"setup_s": setup_s, "cold_ms": cold_ms}), encoding="utf-8")
    records = [results[e.id] for e in eval_examples if e.id in results]
    return ModelRun(key=spec.key, order_mode=order_mode, records=records, setup_s=setup_s, cold_ms=cold_ms,
                    wall_s=wall_s, concurrency=conc, cached=len(cache),
                    meta={"shots": len(shots), "n_pending": len(pending)})


def _store(r: dict, results: dict, fh, lock: threading.Lock) -> None:
    with lock:
        results[r["id"]] = r
        if fh is not None and r["error"] is None:  # yalnız başarılı sonuçlar önbelleğe yazılır
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
