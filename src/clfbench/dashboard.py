"""Etkileşimli, tek dosyalık HTML panel (report.html). Veriler içine gömülür; internet/CDN gerekmez."""
from __future__ import annotations

import json
from pathlib import Path

from . import __version__

TEMPLATE = Path(__file__).with_name("assets") / "dashboard.html"


def build_data(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    labels = list(run["task"]["labels"])
    lab_idx = {l: i for i, l in enumerate(labels)}
    ex_list = run["eval_examples"]
    pos = {e["id"]: i for i, e in enumerate(ex_list)}
    examples = []
    true_by_id: dict[str, int] = {}
    # gerçek etiketler predictions.jsonl içinde; yoksa -1
    preds_by: dict[str, dict[str, dict]] = {}
    pred_path = out_dir / "predictions.jsonl"
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["mode"] != "canonical":
            continue
        preds_by.setdefault(r["model"], {})[r["id"]] = r
        true_by_id[r["id"]] = lab_idx[r["true"]]
    for e in ex_list:
        examples.append({"id": e["id"], "x": e["text"], "t": true_by_id.get(e["id"], -1), "g": e["tags"]})

    short = run["task"].get("short", {})
    desc = run["task"]["labels"]
    lab_out = [{"key": k, "desc": desc[k], "short": short.get(k, "")} for k in labels]

    models = []
    for m in summary["models"]:
        o = {
            "key": m["key"], "display": m["display"], "group": m["group"], "provider": m["provider"],
            "model": m.get("model") or "", "status": m.get("status", "?"),
            "reason": m.get("reason"), "firstError": m.get("first_error"), "notes": m.get("notes") or "",
            "verified": m.get("verified") or "", "local": bool(m.get("local")),
            "price": {"in": m.get("price_in"), "out": m.get("price_out"), "note": m.get("price_note") or ""},
            "setupS": m.get("setup_s"), "coldMs": m.get("cold_ms"), "concurrency": m.get("concurrency"),
            "m": None, "preds": [], "slices": m.get("slices") or {},
        }
        mt = m.get("metrics")
        if mt:
            o["m"] = {
                "n": mt["n"], "nOk": mt["n_ok"], "accuracy": mt["accuracy"], "accuracyCI": mt["accuracy_ci"],
                "accuracyValid": mt["accuracy_valid"], "macroF1": mt["macro_f1"], "macroF1CI": mt["macro_f1_ci"],
                "errorRate": mt["error_rate"], "ece": mt["ece"], "brier": mt["brier"], "nProb": mt["n_prob"],
                "confKinds": mt["conf_kinds"], "accAtCov80": mt["acc_at_cov80"], "covAtAcc95": mt["cov_at_acc95"],
                "latP50": mt["lat_p50"], "latP95": mt["lat_p95"], "latP99": mt["lat_p99"],
                "tokIn": mt["tok_in_mean"], "tokOut": mt["tok_out_mean"], "tokensEstimated": mt["tokens_estimated"],
                "cost1000": mt["cost_per_1000"], "perClass": mt["per_class"], "confusion": mt["confusion"],
            }
            by_id = preds_by.get(m["key"], {})
            for e in ex_list:
                r = by_id.get(e["id"])
                if r is None:
                    o["preds"].append([-1, None, None, "yanıt yok"])
                elif r["error"]:
                    o["preds"].append([-1, None, None, r["error"]])
                else:
                    o["preds"].append([lab_idx.get(r["pred"], -1),
                                       None if r["conf"] is None else round(r["conf"], 4),
                                       None if r["latency_ms"] is None else round(r["latency_ms"], 1), None])
        rob = m.get("robustness")
        if rob:
            o["robustness"] = {"accCanonical": rob["acc_canonical"], "accShuffled": rob["acc_shuffled"],
                               "delta": rob["delta"], "consistency": rob["consistency"]}
        v = m.get("vs_reference")
        if v:
            o["vsRef"] = {"reference": v["reference"], "diff": v["diff"], "lo": v["lo"], "hi": v["hi"]}
        models.append(o)

    ds = dict(summary["dataset"])
    ds["is_builtin"] = ds.get("name") == "destek_tr"
    return {
        "meta": {"run_id": summary["run_id"], "started": run["started"], "dataset": ds,
                 "settings": summary["settings"], "environment": summary["environment"],
                 "note": summary.get("note", ""), "reference": summary.get("reference"), "version": __version__},
        "labels": lab_out, "examples": examples, "models": models,
        "pareto": {"latency": summary.get("pareto_latency", []), "cost": summary.get("pareto_cost", [])},
    }


def render(data: dict) -> str:
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # </script> ve HTML ayrıştırıcı sürprizlerine karşı: '<' karakterini kaçır
    blob = blob.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA_JSON__", blob)


def write_dashboard(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "report.html"
    path.write_text(render(build_data(out_dir)), encoding="utf-8")
    return path
