"""Koşu birleştirme: uyumsuz veriyi reddetme, model önceliği, yeniden hesaplanan farklar ve gerçek örnek üzerinde sabitlenmiş değerler."""
import copy
import json
from pathlib import Path

import pytest

from clfbench import bench
from clfbench.bench import run_benchmark
from clfbench.cli import main
from clfbench.compare import analyze, load_data
from clfbench.keys import prefix_warning
from clfbench.merge import merge_data

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "jev-gemini-40"


def _model(key, status="ok", preds=None, acc=0.5, source_note=""):
    m = {"key": key, "display": key, "group": "karar-api", "status": status, "m": None, "preds": [], "reason": source_note}
    if status == "ok":
        m["m"] = {"accuracy": acc, "latP50": 10.0, "cost1000": None}
        m["preds"] = preds
    return m


def _run(run_id, models, ids=("a", "b", "c", "d"), truth=(0, 1, 0, 1), fp="fp1", ref=None, platform="Linux"):
    return {"meta": {"run_id": run_id, "started": f"2026-10-0{len(run_id) % 9 + 1}T10:00:00", "dataset": {"fingerprint": fp},
                     "settings": {"seed": 0}, "environment": {"platform": platform}, "note": "", "reference": ref},
            "labels": [{"key": "x"}, {"key": "y"}], "examples": [{"id": i, "x": f"m{i}", "t": t, "g": []} for i, t in zip(ids, truth)],
            "models": models, "pareto": {}}


RIGHT = [[0, 1.0, 5.0, None], [1, 1.0, 5.0, None], [0, 1.0, 5.0, None], [1, 1.0, 5.0, None]]
WRONG = [[1, 1.0, 5.0, None], [0, 1.0, 5.0, None], [1, 1.0, 5.0, None], [0, 1.0, 5.0, None]]


def test_scored_beats_failed_beats_notrun_in_any_order():
    ok, failed, notrun = _model("m", preds=RIGHT, acc=1.0), _model("m", "failed"), _model("m", "notrun")
    for order in ([ok, failed, notrun], [notrun, failed, ok], [failed, notrun, ok], [notrun, ok, failed]):
        merged = merge_data([_run(f"r{i}", [m]) for i, m in enumerate(order)])
        assert [m["status"] for m in merged["models"]] == ["ok"]
    merged = merge_data([_run("r1", [notrun]), _run("r2", [failed])])
    assert merged["models"][0]["status"] == "failed"                      # hata kaydı (gerçek neden) çalıştırılmadı kaydını geçer
    merged = merge_data([_run("r1", [failed]), _run("r2", [notrun])])
    assert merged["models"][0]["status"] == "failed"


def test_newer_scored_run_replaces_older_scored_run():
    old = _run("old", [_model("g", preds=WRONG, acc=0.0), _model("ref", preds=RIGHT, acc=1.0)])
    new = _run("new", [_model("g", preds=RIGHT, acc=1.0)])
    merged = merge_data([old, new])
    g = next(m for m in merged["models"] if m["key"] == "g")
    assert g["source"] == "new" and g["m"]["accuracy"] == 1.0
    assert next(m for m in merged["models"] if m["key"] == "ref")["source"] == "old"


def test_vs_reference_and_pareto_are_recomputed():
    a = _run("a", [_model("ref", preds=RIGHT, acc=1.0)], ref="ref")
    b = _run("b", [_model("other", preds=WRONG, acc=0.0)])
    b["models"][0]["vsRef"] = {"reference": "stale", "diff": 9, "lo": 9, "hi": 9}       # eski değer taşınmamalı
    merged = merge_data([a, b])
    other = next(m for m in merged["models"] if m["key"] == "other")
    assert merged["meta"]["reference"] == "ref"
    assert other["vsRef"]["reference"] == "ref" and other["vsRef"]["diff"] == pytest.approx(-1.0)
    assert "vsRef" not in next(m for m in merged["models"] if m["key"] == "ref")
    assert merged["pareto"]["latency"] == ["ref"]                                       # eşit gecikmede doğruluğu yüksek olan
    explicit = merge_data([a, b], reference="other")
    assert explicit["meta"]["reference"] == "other"


def test_incompatible_runs_are_rejected_and_inputs_not_mutated():
    base = _run("a", [_model("m1", preds=RIGHT)])
    for bad in (_run("b", [_model("m2", preds=RIGHT)], fp="DIFFERENT"),
                _run("b", [_model("m2", preds=RIGHT)], ids=("a", "b", "c", "z")),
                _run("b", [_model("m2", preds=RIGHT)], truth=(1, 1, 0, 1))):
        with pytest.raises(ValueError, match="birleştirilemez"):
            merge_data([base, bad])
    with pytest.raises(ValueError, match="en az 2"):
        merge_data([base])
    a, b = _run("a", [_model("m1", preds=RIGHT)]), _run("b", [_model("m2", preds=WRONG)])
    snap = copy.deepcopy((a, b))
    merge_data([a, b])
    assert (a, b) == snap


def test_note_lists_sources_and_platforms():
    merged = merge_data([_run("a", [_model("m1", preds=RIGHT)], platform="Windows"), _run("b", [_model("m2", preds=WRONG)], platform="Linux")],
                        note="ek not")
    n = merged["meta"]["note"]
    assert "Windows" in n and "Linux" in n and "kıyaslanamaz" in n and n.endswith("ek not")
    assert [s["run_id"] for s in merged["meta"]["sources"]] == ["a", "b"]


def test_real_merged_example_pins_numbers():
    """examples/jev-gemini-40: gerçek Jev (jev-40) + gerçek Gemini koşusu."""
    d = load_data(EXAMPLE)
    by = {m["key"]: m for m in d["models"]}
    scored = [k for k, m in by.items() if m.get("m")]
    assert len(scored) == 10 and d["meta"]["reference"] == "jev_beatapi"
    assert by["gemini_3_5_flash_lite"]["m"]["accuracy"] == pytest.approx(0.95) and by["jev_beatapi"]["m"]["accuracy"] == pytest.approx(0.95)
    assert by["gemini_3_5_flash_lite"]["m"]["errorRate"] == 0 and by["gemini_3_5_flash_lite"]["source"] != by["jev_beatapi"]["source"]
    assert not any(m["key"] == "gemini_3_5_flash_lite" and m["status"] == "notrun" for m in d["models"])
    res = analyze(d, "jev_beatapi")
    row = next(r for r in res["paired"] if r["key"] == "gemini_3_5_flash_lite")
    assert (row["only_ref"], row["only_model"], row["both"], row["neither"]) == (2, 2, 36, 0)
    assert row["p"] == 1.0 and row["verdict"] == "ayırt edilemez"
    assert res["only_ref_right"] == [] and res["nobody"] == []


@pytest.fixture
def three_runs(tmp_path, monkeypatch, make_spec, task_and_data):
    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    out = []
    for key, provider, path, group in (("mock_a", "systemone", "/v1/systemone", "karar-api"),
                                       ("mock_b", "chat_json", "/v1/chat/completions", "uretken-llm"),
                                       ("mock_c", "systemone", "/v1/systemone", "karar-api")):
        s = make_spec(provider, path=path, model="m")
        s.key, s.display, s.group = key, key, group
        out.append(run_benchmark([s], task, ex, split="test", warmup=1, out_root=tmp_path / "out", run_name=key))
    return out


def test_cli_merge_and_chained_merge(three_runs, tmp_path):
    a, b, c = three_runs
    ab = tmp_path / "ab"
    assert main(["merge", str(a), str(b), "--out", str(ab), "--note", "deneme"]) == 0
    assert {"merged.json", "report.html", "derin_karsilastirma.md"} <= {p.name for p in ab.iterdir()}
    assert analyze(load_data(ab))["n_models"] == 2
    abc = tmp_path / "abc"
    assert main(["merge", str(ab), str(c), "--out", str(abc), "--reference", "mock_b"]) == 0   # birleşik klasör de girdi olabilir
    merged = json.loads((abc / "merged.json").read_text(encoding="utf-8"))
    assert [m["key"] for m in merged["models"] if m.get("m")] == ["mock_a", "mock_b", "mock_c"] and merged["meta"]["reference"] == "mock_b"
    assert main(["merge", str(a), str(tmp_path / "yok"), "--out", str(tmp_path / "x")]) == 2


def test_gemini_aq_prefix_is_accepted():
    assert prefix_warning("GEMINI_API_KEY", "AQ.Ab8abc") is None and prefix_warning("GEMINI_API_KEY", "AIzaSyabc") is None
    w = prefix_warning("GEMINI_API_KEY", "nonsense")
    assert w and "AIza" in w and "AQ." in w
