"""Derin karşılaştırma: istatistik fonksiyonları, gerçek örnek koşu üzerinde sabitlenmiş değerler ve iki yükleme yolu."""
from pathlib import Path

import pytest

from clfbench import bench
from clfbench.bench import run_benchmark
from clfbench.cli import main
from clfbench.compare import analyze, holm, load_data, mcnemar_exact, render_markdown

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "jev-40"


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 2) == pytest.approx(0.453125)
    assert mcnemar_exact(2, 5) == mcnemar_exact(5, 2)          # simetrik
    assert mcnemar_exact(10, 10) == 1.0                          # üst sınır 1'e kırpılır
    assert mcnemar_exact(25, 0) < 1e-6


def test_holm_is_monotone_and_never_below_raw():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    p = [0.2, 0.001, 0.5, 0.04]
    adj = holm(p)
    assert all(a >= x for a, x in zip(adj, p)) and all(a <= 1 for a in adj)
    order = sorted(range(len(p)), key=lambda i: p[i])
    assert [adj[i] for i in order] == sorted(adj)               # küçük p'nin düzeltilmişi de küçük kalır


@pytest.fixture(scope="module")
def real():
    return analyze(load_data(EXAMPLE), reference="jev_beatapi")


def test_real_run_pins_published_numbers(real):
    """examples/jev-40 gerçek koşusu: rapordaki sayılarla birebir tutarlı olmalı."""
    assert real["n"] == 40 and real["n_models"] == 9 and real["reference"] == "jev_beatapi"
    assert [r["n_correct"] for r in real["ranking"]][:3] == [38, 35, 33]
    assert real["ranking"][0]["key"] == "jev_beatapi" and real["ranking"][-1]["key"] == "laya_multilingual"
    row = {r["key"]: r for r in real["paired"]}
    st = row["setfit_minilm"]
    assert (st["only_ref"], st["only_model"], st["both"], st["neither"]) == (5, 2, 33, 0)
    assert st["p"] == pytest.approx(0.453125) and st["verdict"] == "ayırt edilemez"
    assert row["gliclass_large_v3"]["verdict"] == "referans daha iyi"
    assert row["laya_multilingual"]["verdict"] == "referans daha iyi" and row["laya_multilingual"]["p_holm"] < 1e-5
    for r in real["paired"]:                                     # her örnek dört hücreden birine düşer
        assert r["only_ref"] + r["only_model"] + r["both"] + r["neither"] == 40


def test_real_run_error_and_collapse_analysis(real):
    errs = real["ref_errors"]
    assert len(errs) == 2 and all(e["true"] == "sikayet_oneri" for e in errs)
    assert {e["ref_pred"] for e in errs} == {"kargo_teslimat", "teknik_destek"}
    lay = next(c for c in real["collapse"] if c["key"] == "laya_multilingual")
    assert lay["flag"] and lay["top"] == "teknik_destek" and lay["share"] == pytest.approx(25 / 40)
    assert set(lay["never_right"]) == {"hesap_guvenlik", "sikayet_oneri", "diger"}
    assert not next(c for c in real["collapse"] if c["key"] == "jev_beatapi")["flag"]
    assert dict(real["difficulty"])[9] == 4 and real["nobody"] == []
    jev_conf = next(c for c in real["confidence"] if c["key"] == "jev_beatapi")
    assert jev_conf["right"] > jev_conf["wrong"] and jev_conf["n_wrong"] == 2


def test_markdown_renders_all_sections(real):
    md = render_markdown(real)
    for h in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## Sınırlar"):
        assert h in md
    assert "McNemar" in md and "Holm" in md and "α = 0,05" in md and "yerel olasılık" in md
    assert "native" not in md                                    # güven türü Türkçe gösterilir


def test_requires_two_scored_models(tmp_path):
    with pytest.raises(ValueError, match="en az 2"):
        analyze({"labels": [{"key": "a"}], "examples": [{"id": "1", "x": "m", "t": 0, "g": []}],
                 "models": [], "meta": {"run_id": "r", "dataset": {"name": "d"}}})


def test_load_data_missing_folder_is_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="predictions.jsonl"):
        load_data(tmp_path)


@pytest.fixture
def live_run(tmp_path, monkeypatch, make_spec, task_and_data):
    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    s1 = make_spec("systemone", path="/v1/systemone", model="jev-test")
    s1.key, s1.display = "mock_jev", "Mock Jev"
    s2 = make_spec("chat_json", path="/v1/chat/completions", model="chat-test")
    s2.key, s2.display, s2.group = "mock_chat", "Mock Chat", "uretken-llm"
    return run_benchmark([s1, s2], task, ex, split="test", warmup=1, out_root=tmp_path / "out", run_name="cmp")


def test_both_loading_paths_agree_and_cli_writes_file(live_run, tmp_path):
    from_jsonl = analyze(load_data(live_run))
    (live_run / "predictions.jsonl").rename(tmp_path / "saklandi.jsonl")      # yalnız report.html kalsın
    from_html = analyze(load_data(live_run))
    assert from_jsonl == from_html and from_jsonl["n_models"] == 2

    out = tmp_path / "karsilastirma.md"
    assert main(["compare", str(live_run), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Derin karşılaştırma")
    assert main(["compare", str(tmp_path / "yok")]) == 2
