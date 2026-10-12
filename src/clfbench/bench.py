"""Koşu orkestrasyonu: modelleri çalıştır -> ham tahminleri yaz -> metrikleri hesapla -> rapor üret."""
from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .config import ModelSpec
from .core import Example, TaskSpec
from .data import dataset_fingerprint, select
from .metrics import compute_metrics, paired_bootstrap_diff, pareto_front, tag_slices
from .paths import CACHE_DIR, RESULTS_DIR
from .prompts import PROMPT_VERSION
from .runner import ModelRun, run_model


def _pkg_versions() -> dict[str, str]:
    from importlib import metadata

    out = {}
    for name in ("requests", "numpy", "anthropic", "scikit-learn", "torch", "transformers",
                 "sentence-transformers", "gliclass", "laya", "setfit"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def _git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def run_benchmark(specs: list[ModelSpec], task: TaskSpec, examples: list[Example], *, split: str = "test",
                  seed: int = 0, shots_k: int = 0, robustness: bool = False, warmup: int = 1,
                  concurrency: int | None = None, use_cache: bool = True, run_name: str = "",
                  note: str = "", dataset_label: str = "", out_root: Path | None = None,
                  reference: str | None = None, limit: int | None = None, argv: list[str] | None = None,
                  not_run: list[dict] | None = None) -> Path:
    eval_ex = select(examples, split)
    if limit:
        eval_ex = _stratified_limit(eval_ex, limit, task.label_keys)
    train = [e for e in examples if e.split == "train"]
    if not eval_ex:
        raise ValueError(f"'{split}' bölümünde örnek yok.")
    fp = dataset_fingerprint(task, examples)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root or RESULTS_DIR) / (f"{stamp}-{run_name}" if run_name else stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_meta: list[dict] = []
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as pf:
        for spec in specs:
            meta = _spec_meta(spec)
            print(f"\n▶ {spec.display}  [{spec.key}]")
            runs: list[ModelRun] = []
            try:
                modes = ["canonical"] + (["shuffle"] if robustness and not _order_independent(spec) else [])
                for mode in modes:
                    try:
                        run = run_model(spec, task, eval_ex, train, order_mode=mode, seed=seed, shots_k=shots_k,
                                        data_fp=fp, cache_dir=CACHE_DIR, use_cache=use_cache, warmup=warmup,
                                        concurrency=concurrency)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        if mode == "canonical":
                            raise  # ana koşu yoksa model atlanır
                        # dayanıklılık koşusu başarısızsa ana sonuçlar korunur
                        meta["robustness_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                        print(f"  ⚠ seçenek sırası testi başarısız: {meta['robustness_error']}")
                        continue
                    runs.append(run)
                    for r in run.records:
                        pf.write(json.dumps({"model": spec.key, "mode": mode, **r}, ensure_ascii=False) + "\n")
                    pf.flush()
                main = runs[0]
                meta.update(status="ok", setup_s=round(main.setup_s, 3), cold_ms=main.cold_ms,
                            wall_s=round(main.wall_s, 3), concurrency=main.concurrency, cached=main.cached)
                n_err = sum(1 for r in main.records if r["error"])
                if n_err:
                    first = next(r["error"] for r in main.records if r["error"])
                    meta["first_error"] = first
                    print(f"  ⚠ {n_err}/{len(main.records)} istek başarısız. İlk hata: {first}")
            except KeyboardInterrupt:
                print("\nKullanıcı tarafından durduruldu; o ana kadarki sonuçlar kaydedildi.")
                meta.update(status="interrupted")
                models_meta.append(meta)
                break
            except Exception as e:
                meta.update(status="failed", reason=f"{type(e).__name__}: {str(e)[:300]}")
                print(f"  ✗ atlandı: {meta['reason']}")
            models_meta.append(meta)

    run_json = {
        "clfbench_version": __version__,
        "run_id": out_dir.name,
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": argv or sys.argv[1:],
        "note": note,
        "dataset": {"name": dataset_label or task.name, "fingerprint": fp, "split": split,
                    "n_eval": len(eval_ex), "n_train": len(train),
                    "class_counts": {l: sum(1 for e in eval_ex if e.label == l) for l in task.label_keys}},
        "task": {"instruction": task.instruction, "labels": task.labels, "notes": task.notes},
        "settings": {"seed": seed, "shots_per_label": shots_k, "robustness": robustness, "warmup": warmup,
                     "prompt_version": PROMPT_VERSION, "concurrency_override": concurrency},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": _pkg_versions(), "git_commit": _git_commit()},
        "models": models_meta,
        "not_run": not_run or [],
        "eval_examples": [{"id": e.id, "tags": list(e.tags), "text": e.text} for e in eval_ex],
    }
    (out_dir / "run.json").write_text(json.dumps(run_json, ensure_ascii=False, indent=2), encoding="utf-8")
    from .report import generate_report

    finalize(out_dir, reference=reference)
    generate_report(out_dir)
    return out_dir


def _order_independent(spec: ModelSpec) -> bool:
    """Etiket sırasından etkilenmeyen modeller (eğitimli / gömme) için karıştırma testi anlamsız."""
    return spec.provider in ("tfidf_lr", "setfit", "nli", "embedding")


def _spec_meta(spec: ModelSpec) -> dict:
    return {
        "key": spec.key, "display": spec.display, "group": spec.group, "provider": spec.provider,
        "model": spec.model, "price_in": spec.price_in, "price_out": spec.price_out,
        "price_note": spec.price_note, "verified": spec.verified, "notes": spec.notes,
        "local": spec.is_local, "free": spec.free, "free_note": spec.free_note,
    }


def _stratified_limit(examples: list[Example], n: int, labels: list[str]) -> list[Example]:
    per = max(1, n // len(labels))
    cnt: dict[str, int] = {}
    out = []
    for e in examples:
        cnt[e.label] = cnt.get(e.label, 0) + 1
        if cnt[e.label] <= per:
            out.append(e)
    return out


# ------------------------------------------------------------------------------------ özet
def _is_correct(r: Optional[dict]) -> int:
    return int(r is not None and r["error"] is None and r["pred"] == r["true"])


def finalize(out_dir: Path, reference: str | None = None, n_boot: int = 2000) -> dict:
    """predictions.jsonl + run.json -> summary.json (metrikler, güven aralıkları, eşlenmiş karşılaştırmalar)."""
    out_dir = Path(out_dir)
    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    labels = list(run["task"]["labels"])
    tags_by_id = {e["id"]: tuple(e["tags"]) for e in run["eval_examples"]}
    by: dict[tuple[str, str], list[dict]] = {}
    for line in (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by.setdefault((r["model"], r["mode"]), []).append(r)
    ids = [e["id"] for e in run["eval_examples"]]
    seed = run["settings"]["seed"]

    models = []
    correct: dict[str, np.ndarray] = {}
    for m in run["models"]:
        entry = dict(m)
        recs = by.get((m["key"], "canonical"))
        if not recs or m.get("status") not in ("ok", "interrupted"):
            models.append(entry)
            continue
        pos = {i: n for n, i in enumerate(ids)}
        recs = sorted(recs, key=lambda r: pos[r["id"]])
        met = compute_metrics(recs, labels, m.get("price_in"), m.get("price_out"), seed=seed, n_boot=n_boot)
        entry["metrics"] = dataclasses.asdict(met)
        entry["slices"] = tag_slices(recs, tags_by_id)
        by_id = {r["id"]: r for r in recs}
        correct[m["key"]] = np.array([_is_correct(by_id.get(i)) for i in ids])
        shuf = by.get((m["key"], "shuffle"))
        if shuf:
            sm = {r["id"]: r for r in shuf}
            both = [i for i in ids if i in by_id and i in sm]
            agree = [int(by_id[i]["error"] is None and sm[i]["error"] is None and by_id[i]["pred"] == sm[i]["pred"])
                     for i in both]
            acc_s = float(np.mean([_is_correct(sm[i]) for i in both])) if both else None
            acc_c = float(np.mean([_is_correct(by_id[i]) for i in both])) if both else None
            entry["robustness"] = {"n": len(both), "acc_canonical": acc_c, "acc_shuffled": acc_s,
                                   "delta": (acc_s - acc_c) if acc_s is not None and acc_c is not None else None,
                                   "consistency": float(np.mean(agree)) if agree else None}
        models.append(entry)

    scored = [m for m in models if "metrics" in m]
    ref = reference if reference in correct else (
        max(scored, key=lambda m: m["metrics"]["accuracy"])["key"] if scored else None)
    for m in scored:
        if ref and m["key"] != ref:
            d, lo, hi, p = paired_bootstrap_diff(correct[m["key"]], correct[ref], seed=seed, n_boot=n_boot)
            m["vs_reference"] = {"reference": ref, "diff": d, "lo": lo, "hi": hi, "p_better": p}
    # Pareto: doğruluk ↑, p50 gecikme ↓, 1000 karar maliyeti ↓ (yerel modellerde maliyet 0 kabul edilmez: dışarıda)
    pts = {m["key"]: (m["metrics"]["accuracy"], m["metrics"]["lat_p50"]) for m in scored}
    front_lat = pareto_front(pts, (True, False))
    pts_c = {m["key"]: (m["metrics"]["accuracy"], m["metrics"]["cost_per_1000"]) for m in scored}
    front_cost = pareto_front(pts_c, (True, False))
    summary = {"run_id": run["run_id"], "reference": ref, "labels": labels,
               "pareto_latency": sorted(front_lat), "pareto_cost": sorted(front_cost),
               "models": models, "dataset": run["dataset"], "settings": run["settings"],
               "environment": run["environment"], "note": run.get("note", ""), "not_run": run.get("not_run", [])}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
