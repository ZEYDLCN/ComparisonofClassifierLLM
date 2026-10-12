"""Ücretsiz model bilgisi, --free seçimi ve 'çalıştırılmayan modeller' kaydının rapora/panele yansıması."""
import json
import re

import pytest

from clfbench import bench
from clfbench.bench import run_benchmark
from clfbench.cli import main
from clfbench.config import FREE_KINDS, SAFE_FREE, load_registry, not_run_entries, resolve_models


@pytest.fixture(scope="module")
def reg():
    return load_registry()


def test_registry_free_values_are_valid(reg):
    assert {m.free for m in reg.values()} <= set(FREE_KINDS) | {""}
    assert reg["jev_beatapi"].free == "ucretsiz" and reg["tfidf_lr"].free == "yerel"
    for m in reg.values():
        if m.free:
            assert m.free_note, f"{m.key}: ücretsiz girdinin sınırı (free_note) yazılmalı"


def test_safe_free_models_can_never_cost_money(reg):
    """--free'nin seçtiği hiçbir modelin pozitif tarifesi olmamalı (yanlış girdi para harcatmasın)."""
    for m in reg.values():
        if m.free in SAFE_FREE:
            assert not (m.price_in or 0) and not (m.price_out or 0), f"{m.key} ücretsiz işaretli ama fiyatı var"
    for m in reg.values():
        if m.provider in ("claude", "gemini", "openai_decisions"):
            assert m.free in ("", "olasi")                       # doğrulanmadan 'güvenli ücretsiz' sayılmaz


def test_free_only_selection(reg):
    run, skipped = resolve_models(reg, None, free_only=True)
    chosen = run + [m for m, _ in skipped]
    assert chosen and all(m.free in SAFE_FREE for m in chosen)
    assert not any(m.free == "olasi" for m in chosen)           # olası ücretsiz olanlar yalnız açıkça istenince
    # açıkça istenen 'olasi' model free_only ile birlikte verilirse yine elenir (ücret riski)
    run2, skipped2 = resolve_models(reg, ["mercury_decide"], free_only=True)
    assert not run2 and not skipped2


def test_not_run_entries_cover_missing_and_disabled(reg, monkeypatch):
    for m in reg.values():
        if m.api_key_env:
            monkeypatch.delenv(m.api_key_env, raising=False)
    run, skipped = resolve_models(reg, None)
    nr = {n["key"]: n for n in not_run_entries(reg, None, None, skipped)}
    assert "claude_haiku_5_5" in nr and "tanımlı değil" in nr["claude_haiku_5_5"]["reason"]
    assert "ANTHROPIC_API_KEY" in nr["claude_haiku_5_5"]["fix"]["short"] and nr["claude_haiku_5_5"]["fix"]["cmd"].endswith("claude_haiku_5_5")
    assert nr["sage"]["reason"].startswith("varsayılan kapalı") and nr["sage"]["enabled"] is False
    assert not (set(nr) & {m.key for m in run})                  # çalışan model "çalıştırılmadı" listesine girmez
    # --models ile belirli model istenirse diğerleri 'eksik' sayılmaz
    run_c, skipped_c = resolve_models(reg, ["claude_haiku_5_5"])
    only = not_run_entries(reg, ["claude_haiku_5_5"], None, skipped_c)
    assert [n["key"] for n in only] == ["claude_haiku_5_5"]


def test_fix_hint_for_missing_package(reg):
    from clfbench.keys import fix_for

    spec = reg["gliclass_large_v3"]
    f = fix_for(spec, "paket eksik: gliclass, transformers, torch")
    assert "pip install" in f["short"] and "gliclass" in f["short"] and "local" in f["short"]


@pytest.fixture
def run_with_not_run(tmp_path, monkeypatch, make_spec, task_and_data, reg):
    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    s1 = make_spec("systemone", path="/v1/systemone", model="jev-test")
    s1.key, s1.display, s1.free, s1.free_note = "mock_jev", "Mock Jev", "ucretsiz", "test limiti"
    s2 = make_spec("chat_json", path="/v1/chat/completions", model="chat-test")
    s2.key, s2.display, s2.group = "mock_chat", "Mock Chat", "uretken-llm"
    skipped = [(reg["claude_haiku_5_5"], "ANTHROPIC_API_KEY tanımlı değil"), (reg["laya_multilingual"], "paket eksik: laya")]
    nr = not_run_entries(reg, ["claude_haiku_5_5", "laya_multilingual"], None, skipped)   # açıkça istendi: kapalılar eklenmez
    return run_benchmark([s1, s2], task, ex, split="test", warmup=1, out_root=tmp_path / "out", run_name="free", not_run=nr)


def test_not_run_reaches_json_report_and_dashboard(run_with_not_run):
    d = run_with_not_run
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert [n["key"] for n in run["not_run"]] == [n["key"] for n in summary["not_run"]] == ["claude_haiku_5_5", "laya_multilingual"]
    assert next(m for m in summary["models"] if m["key"] == "mock_jev")["free"] == "ucretsiz"

    md = (d / "report.md").read_text(encoding="utf-8")
    assert "## Bu koşuda çalıştırılmayan modeller" in md and "Claude Haiku 5.5" in md and "`.env` içine `ANTHROPIC_API_KEY=...` ekle" in md
    assert "Yerel — API ücreti yok" in md                         # Laya satırı ücret durumuyla gelir

    html = (d / "report.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    by = {m["key"]: m for m in data["models"]}
    assert by["claude_haiku_5_5"]["status"] == "notrun" and by["claude_haiku_5_5"]["m"] is None and by["claude_haiku_5_5"]["preds"] == []
    assert by["claude_haiku_5_5"]["fix"]["url"].startswith("https://") and by["laya_multilingual"]["free"] == "yerel"
    assert by["mock_jev"]["free"] == "ucretsiz" and by["mock_jev"]["freeNote"] == "test limiti" and by["mock_jev"]["m"]
    for needle in ("id=\"notrun\"", "id=\"notruncards\"", "id=\"freechip\"", "function renderNotrun"):
        assert needle in html                                     # arayüz bölümü ve filtre şablonda var


def test_compare_ignores_not_run_models(run_with_not_run):
    from clfbench.compare import analyze, load_data

    res = analyze(load_data(run_with_not_run))
    assert res["n_models"] == 2                                   # çalıştırılmayanlar karşılaştırmaya girmez


def test_cli_models_and_doctor_show_free_info(capsys):
    assert main(["models", "--all"]) == 0
    out = capsys.readouterr().out
    assert "ücretsiz" in out.splitlines()[0] and "ucretsiz" in out and "olasi" in out
    assert main(["doctor"]) == 0
    assert "Ücretsiz ve şu an hazır" in capsys.readouterr().out
