"""Statik karşılaştırma sitesi: gömülü veri, yalnızca özet metrik, kaçış güvenliği, JS söz dizimi, Vercel yapılandırması."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from clfbench import bench
from clfbench.bench import run_benchmark
from clfbench.cli import main
from clfbench.core import Example
from clfbench.site import build_site_data, render, write_site

ROOT = Path(__file__).resolve().parents[1]
EVIL = "</script><img src=x onerror=alert(1)> kargom gelmedi"
DATA_RE = re.compile(r'<script id="data" type="application/json">(.*?)</script>', re.S)


@pytest.fixture
def run_dir(tmp_path, monkeypatch, make_spec, task_and_data):
    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    s1 = make_spec("systemone", path="/v1/systemone", model="jev-test")
    s1.key, s1.display = "mock_jev", "Mock Jev </script><b>x</b>"  # düşmanca görünen ad de güvenle gömülmeli
    s2 = make_spec("chat_json", path="/v1/chat/completions", model="chat-test")
    s2.key, s2.display, s2.group = "mock_chat", "Mock Chat (sohbet)", "uretken-llm"
    dead = make_spec("systemone", path="/v1/systemone", key="BROKE_API_KEY")
    dead.key, dead.display = "dead", "Dead API"
    nokey = make_spec("systemone", path="/v1/systemone", key="NOPE_NOT_SET")
    nokey.key, nokey.display = "no_key", "Anahtarsız Model"
    evil = Example(id="evil1", text=EVIL, label="kargo_teslimat", split="test")
    return run_benchmark([s1, s2, dead], task, ex + [evil], split="test", warmup=1, robustness=True,
                         out_root=tmp_path / "out", run_name="site",
                         skipped=[(nokey, "NOPE_NOT_SET tanımlı değil")])


def _embedded(html):
    m = DATA_RE.search(html)
    assert m, "gömülü veri bloğu yok"
    return json.loads(m.group(1))


def test_site_files_and_embedded_data(run_dir, tmp_path):
    out = tmp_path / "site"
    path = write_site(run_dir, out)
    assert path == out / "index.html" and (out / "data.json").exists()
    html = path.read_text(encoding="utf-8")
    assert "__DATA_JSON__" not in html and "clfbench" in html
    assert html.count("<script") == 2  # veri bloğu + uygulama kodu
    d = _embedded(html)
    assert d == json.loads((out / "data.json").read_text(encoding="utf-8"))

    by = {m["key"]: m for m in d["models"]}
    assert set(by) == {"mock_jev", "mock_chat", "dead"}
    for k in ("mock_jev", "mock_chat"):
        m = by[k]["metrics"]
        assert 0 < m["accuracy"] <= 1 and len(m["confusion"]) == len(d["labels"]) and m["per_class"]
        assert by[k]["robustness"]["consistency"] == pytest.approx(1.0)
    assert "metrics" not in by["dead"] and by["dead"]["status"] == "failed" and "402" in by["dead"]["reason"]
    assert d["skipped"] == [{"key": "no_key", "display": "Anahtarsız Model", "group": "karar-api",
                             "reason": "NOPE_NOT_SET tanımlı değil"}]
    assert d["meta"]["dataset"]["n_eval"] > 0 and d["labels"][0]["key"] and d["meta"]["repo"].startswith("https://")


def test_site_publishes_only_aggregates(run_dir, tmp_path):
    out = tmp_path / "site"
    write_site(run_dir, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    raw = (out / "data.json").read_text(encoding="utf-8")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    sample = run["eval_examples"][0]["text"]
    for blob in (html, raw):
        assert "kargom gelmedi" not in blob and sample not in blob and "evil1" not in blob
    d = _embedded(html)
    assert not {"examples", "eval_examples", "preds", "predictions"} & set(d)
    assert all("preds" not in m and "verified" not in m for m in d["models"])


def test_embedding_is_script_safe(run_dir, tmp_path):
    write_site(run_dir, tmp_path / "site")
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    block = DATA_RE.search(html).group(1)
    assert "</script" not in block and "<b>" not in block  # '<' kaçırıldı
    assert any(m["display"] == "Mock Jev </script><b>x</b>" for m in _embedded(html)["models"])  # veri bozulmadan geri okunur


def test_site_javascript_syntax(run_dir, tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node yok")
    write_site(run_dir, tmp_path / "site")
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    js = re.findall(r"<script>(.*?)</script>", html, re.S)[0]
    f = tmp_path / "site.js"
    f.write_text(js, encoding="utf-8")
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_render_is_deterministic(run_dir):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    a = render(build_site_data(summary, run))
    assert a == render(build_site_data(summary, run))


def test_old_run_without_skipped_still_renders(tmp_path):
    """`skipped` alanından önce üretilmiş örnek koşu da siteye dönüşebilmeli."""
    src = ROOT / "examples" / "jev-40"
    if not (src / "summary.json").exists():
        pytest.skip("örnek koşu yok")
    d = _embedded(write_site(src, tmp_path / "site").read_text(encoding="utf-8"))
    assert d["skipped"] == [] and sum(1 for m in d["models"] if "metrics" in m) >= 9


def test_cli_site_command(run_dir, tmp_path, capsys):
    out = tmp_path / "s"
    assert main(["site", str(run_dir), "--out", str(out), "--repo", "https://example.com/r"]) == 0
    assert _embedded((out / "index.html").read_text(encoding="utf-8"))["meta"]["repo"] == "https://example.com/r"
    assert main(["site", str(tmp_path / "yok"), "--out", str(out)]) == 2
    assert "bulunamadı" in capsys.readouterr().out


def test_committed_site_and_vercel_config():
    """Depodaki yayınlanabilir site: gömülü veri data.json ile aynı, vercel.json o klasörü yayınlıyor."""
    cfg = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    site = ROOT / cfg["outputDirectory"]
    assert cfg["buildCommand"] is None and cfg["installCommand"] is None  # derleme gerekmez: statik
    assert (site / "index.html").exists(), "vercel.json'daki outputDirectory'de index.html yok"
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "__DATA_JSON__" not in html and html.count("<script") == 2
    assert _embedded(html) == json.loads((site / "data.json").read_text(encoding="utf-8"))


def test_external_result_is_separate_and_labeled(run_dir, tmp_path):
    """Başka koşudan gelen sonuç yalnızca 'önceki koşu' kartında görünür; sıralamaya/tabloya girmez."""
    from clfbench.site import load_external

    src = ROOT / "examples" / "jev-40"
    if not (src / "summary.json").exists():
        pytest.skip("örnek koşu yok")
    ext = load_external(src, "jev_beatapi")
    assert ext["dataset"]["n_eval"] == 40 and ext["model"]["metrics"]["accuracy"] > 0
    html = write_site(run_dir, tmp_path / "site", external=[ext]).read_text(encoding="utf-8")
    d = _embedded(html)
    assert [e["model"]["key"] for e in d["external"]] == ["jev_beatapi"]
    assert "jev_beatapi" not in {m["key"] for m in d["models"]}  # asıl koşunun modelleri arasında değil
    assert "önceki koşu" in html and "Karşılaştırılamaz" in html
    with pytest.raises(ValueError):
        load_external(src, "jev_ai")  # o koşuda başarısız (puanlanmamış) model
    with pytest.raises(ValueError):
        load_external(src, "yok_boyle_model")


def test_cli_also_option(run_dir, tmp_path, capsys):
    src = ROOT / "examples" / "jev-40"
    if not (src / "summary.json").exists():
        pytest.skip("örnek koşu yok")
    out = tmp_path / "s"
    assert main(["site", str(run_dir), "--out", str(out), "--also", f"{src}:jev_beatapi"]) == 0
    assert _embedded((out / "index.html").read_text(encoding="utf-8"))["external"][0]["run_id"].endswith("jev-40")
    assert main(["site", str(run_dir), "--out", str(out), "--also", str(src)]) == 2  # model anahtarı eksik
    assert "YOL:MODEL_ANAHTARI" in capsys.readouterr().out
