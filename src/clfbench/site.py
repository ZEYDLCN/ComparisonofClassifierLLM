"""Statik karşılaştırma sitesi (Vercel/GitHub Pages gibi yerlere olduğu gibi yüklenir).

Bir koşunun `summary.json` + `run.json` dosyalarından, tüm metrikleri yan yana gösteren koyu temalı tek sayfalık bir
site üretir. Yalnızca TOPLU metrikler yayımlanır: değerlendirme örneklerinin metinleri ve tek tek tahminler siteye
girmez; bu yüzden kendi (özel) veri setinle üretilen site de güvenle paylaşılabilir.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import __version__

TEMPLATE = Path(__file__).with_name("assets") / "site.html"
DEFAULT_REPO = "https://github.com/ZEYDLCN/ComparisonofClassifierLLM"

# Siteye yalnızca bu alanlar geçer (beyaz liste): yeni bir alan eklendiğinde yanlışlıkla yayımlanmasın.
_MODEL_FIELDS = ("key", "display", "group", "provider", "model", "local", "status", "reason", "first_error", "notes",
                 "price_in", "price_out", "price_note", "setup_s", "cold_ms", "wall_s", "concurrency", "cached",
                 "metrics", "slices", "robustness", "vs_reference")


def build_site_data(summary: dict, run: dict, repo: str = DEFAULT_REPO) -> dict:
    task = run.get("task", {})
    labels = list(summary["labels"])
    desc, short = task.get("labels", {}), task.get("short", {})
    env = summary.get("environment", {})
    ds = dict(summary["dataset"])
    ds["is_builtin"] = ds.get("name") == "destek_tr"
    return {
        "meta": {
            "run_id": summary["run_id"], "started": run.get("started", ""), "dataset": ds,
            "settings": summary.get("settings", {}), "note": summary.get("note", ""),
            "reference": summary.get("reference"), "version": __version__, "repo": repo,
            "environment": {"python": env.get("python"), "platform": env.get("platform"),
                            "git_commit": env.get("git_commit"), "packages": env.get("packages", {})},
        },
        "labels": [{"key": k, "desc": desc.get(k, ""), "short": short.get(k) or k.replace("_", " ")} for k in labels],
        "models": [{f: m[f] for f in _MODEL_FIELDS if f in m} for m in summary["models"]],
        "pareto": {"latency": summary.get("pareto_latency", []), "cost": summary.get("pareto_cost", [])},
        "skipped": summary.get("skipped", []),
    }


def render(data: dict) -> str:
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # </script> ve HTML ayrıştırıcı sürprizlerine karşı: '<' karakterini kaçır
    blob = blob.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA_JSON__", blob)


def write_site(run_dir: Path, out_dir: Path, repo: str = DEFAULT_REPO) -> Path:
    """`run_dir` (summary.json + run.json) -> `out_dir`/index.html (+ data.json). Var olan dosyaların üzerine yazar."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    data = build_site_data(summary, run, repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(render(data), encoding="utf-8", newline="\n")
    (out_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                                       encoding="utf-8", newline="\n")
    return out_dir / "index.html"
