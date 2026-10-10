"""Uçtan uca: sahte API'ler + yerel TF-IDF ile tam koşu, önbellek/devam, rapor dosyaları."""
import json

import pytest

from clfbench import bench
from clfbench.bench import run_benchmark
from clfbench.config import ModelSpec


@pytest.fixture
def specs(make_spec):
    s1 = make_spec("systemone", path="/v1/systemone", model="jev-test")
    s1.key, s1.display = "mock_jev", "Mock Jev"
    s2 = make_spec("chat_json", path="/v1/chat/completions", model="chat-test")
    s2.key, s2.display, s2.group = "mock_chat", "Mock Chat", "uretken-llm"
    s3 = make_spec("systemone", path="/v1/systemone", model="down", key="NOPE_NOT_SET")
    s3.key = "no_key"
    return [s1, s2]


def _calls(mock_log, path):
    return sum(1 for p, _, _ in mock_log if p == path)


def test_full_run_with_robustness_cache_and_reports(tmp_path, monkeypatch, task_and_data, specs):
    from mock_server import _Handler

    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    kw = dict(split="test", warmup=0, robustness=True, out_root=tmp_path / "out", run_name="t")
    _Handler.log.clear()
    out = run_benchmark(specs, task, ex, **kw)

    for f in ("run.json", "predictions.jsonl", "summary.json", "report.md", "report.html"):
        assert (out / f).exists(), f
    for f in ("accuracy.png", "latency.png", "cost.png", "confusion.png", "robustness.png"):
        assert (out / "charts" / f).stat().st_size > 2000, f

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    by = {m["key"]: m for m in summary["models"]}
    n_eval = summary["dataset"]["n_eval"]
    for k in ("mock_jev", "mock_chat"):
        m = by[k]["metrics"]
        assert m["n"] == n_eval and m["error_rate"] == 0
        assert 0.3 < m["accuracy"] < 1.0  # sahte kural tabanlı sınıflandırıcı: anlamlı ama kusurlu
        assert m["lat_p50"] > 0 and m["cost_per_1000"] > 0 and m["accuracy_ci"][0] <= m["accuracy"] <= m["accuracy_ci"][1]
        assert by[k]["robustness"]["consistency"] == pytest.approx(1.0)  # sahte model sıradan etkilenmez
    assert by["mock_chat"]["metrics"]["conf_kinds"] == ["verbalized"]
    assert summary["reference"] in by and "vs_reference" in by[
        "mock_chat" if summary["reference"] == "mock_jev" else "mock_jev"]
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Mock Jev" in html and "data:image/png;base64" in html and "Yöntem ve sınırlar" in html

    # --- önbellek: aynı koşu ikinci kez API'ye gitmemeli ---
    calls_before = len(_Handler.log)
    run_benchmark(specs, task, ex, **{**kw, "run_name": "t2"})
    assert len(_Handler.log) == calls_before, "önbellekten yanıt verilmeliydi"


def test_failed_requests_are_counted_not_dropped(tmp_path, monkeypatch, make_spec, task_and_data):
    from clfbench.core import Example

    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    test_ex = [e for e in ex if e.split == "test"][:6]
    broken = [Example(id="bad1", text="__500__ patla", label="diger", split="test")]
    spec = make_spec("systemone", path="/v1/systemone")
    spec.max_retries = 0
    spec.key, spec.display = "flaky", "Flaky"
    out = run_benchmark([spec], task, test_ex + broken + [e for e in ex if e.split != "test"], split="test",
                        warmup=0, out_root=tmp_path / "out")
    m = json.loads((out / "summary.json").read_text(encoding="utf-8"))["models"][0]
    assert m["metrics"]["n"] == len(test_ex) + 1
    assert m["metrics"]["error_rate"] == pytest.approx(1 / (len(test_ex) + 1))
    assert "500" in m["first_error"]
    assert m["metrics"]["accuracy"] <= m["metrics"]["accuracy_valid"]


def test_unavailable_model_is_skipped_by_resolver(make_spec):
    from clfbench.config import resolve_models

    s = make_spec("systemone", path="/v1/systemone", key="NOPE_NOT_SET")
    run, skipped = resolve_models({s.key: s}, None)
    assert not run and "NOPE_NOT_SET" in skipped[0][1]


def test_local_supervised_model_trains_only_on_train(tmp_path, monkeypatch, task_and_data):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    spec = ModelSpec(key="tfidf", display="TF-IDF", group="egitimli", provider="tfidf_lr", requires=["sklearn"])
    out = run_benchmark([spec], task, ex, split="test", out_root=tmp_path / "out")
    m = json.loads((out / "summary.json").read_text(encoding="utf-8"))["models"][0]
    assert m["metrics"]["accuracy"] > 0.5 and m["metrics"]["cost_per_1000"] is None
    # test örnekleri eğitimde görülmemeli: aynı veriyle ezber (%100) olmamalı
    assert m["metrics"]["accuracy"] < 0.99


def test_bad_key_fails_fast_without_burning_requests(tmp_path, monkeypatch, make_spec, task_and_data):
    from mock_server import _Handler

    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    spec = make_spec("systemone", path="/v1/systemone", key="BAD_API_KEY")
    spec.key, spec.display = "badkey", "Bad Key"
    _Handler.log.clear()
    out = run_benchmark([spec], task, ex, split="test", warmup=1, out_root=tmp_path / "out")
    m = json.loads((out / "summary.json").read_text(encoding="utf-8"))["models"][0]
    assert m["status"] == "failed" and "401" in m["reason"] and "ping" in m["reason"]
    assert len(_Handler.log) == 1, "geçersiz anahtarla yalnızca ısınma isteği atılmalı"
    assert "metrics" not in m


def test_payment_required_fails_fast(tmp_path, monkeypatch, make_spec, task_and_data):
    from mock_server import _Handler

    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    spec = make_spec("systemone", path="/v1/systemone", key="BROKE_API_KEY")
    spec.key, spec.display = "broke", "Broke"
    _Handler.log.clear()
    out = run_benchmark([spec], task, ex, split="test", warmup=1, out_root=tmp_path / "out")
    m = json.loads((out / "summary.json").read_text(encoding="utf-8"))["models"][0]
    assert m["status"] == "failed" and "402" in m["reason"] and "wallet_not_provisioned" in m["reason"]
    assert len(_Handler.log) == 1
