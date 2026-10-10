"""Etkileşimli panel: gömülü veri bütünlüğü, kaçış güvenliği ve JS söz dizimi."""
import json
import re
import shutil
import subprocess

import pytest

from clfbench import bench
from clfbench.bench import run_benchmark
from clfbench.core import Example
from clfbench.dashboard import build_data, render


@pytest.fixture
def out_dir(tmp_path, monkeypatch, make_spec, task_and_data):
    monkeypatch.setattr(bench, "CACHE_DIR", tmp_path / "cache")
    task, ex = task_and_data
    s1 = make_spec("systemone", path="/v1/systemone", model="jev-test")
    s1.key, s1.display = "mock_jev", "Mock Jev"
    s2 = make_spec("chat_json", path="/v1/chat/completions", model="chat-test")
    s2.key, s2.display, s2.group = "mock_chat", "Mock Chat", "uretken-llm"
    dead = make_spec("systemone", path="/v1/systemone", key="BROKE_API_KEY")
    dead.key, dead.display = "dead", "Dead API"
    # XSS/ayrıştırıcı sürprizi: gömülü JSON içinde </script> bulunan bir mesaj
    evil = Example(id="evil1", text='</script><img src=x onerror=alert(1)> kargom gelmedi', label="kargo_teslimat", split="test")
    return run_benchmark([s1, s2, dead], task, ex + [evil], split="test", warmup=1, robustness=True,
                         out_root=tmp_path / "out", run_name="dash")


def _embedded(html):
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "gömülü veri bloğu yok"
    return json.loads(m.group(1))


def test_dashboard_file_and_data_shape(out_dir):
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "__DATA_JSON__" not in html and "Classifier LLM" in html
    d = _embedded(html)
    keys = {m["key"]: m for m in d["models"]}
    assert set(keys) == {"mock_jev", "mock_chat", "dead"}
    n = len(d["examples"])
    for k in ("mock_jev", "mock_chat"):
        m = keys[k]
        assert len(m["preds"]) == n and m["m"]["n"] == n
        assert 0 < m["m"]["accuracy"] <= 1 and len(m["m"]["confusion"]) == len(d["labels"])
        assert m["robustness"]["consistency"] == pytest.approx(1.0)
        assert all(len(p) == 4 for p in m["preds"])
    dead = keys["dead"]
    assert dead["m"] is None and dead["status"] == "failed" and "402" in dead["reason"]
    assert d["meta"]["dataset"]["n_eval"] == n and d["labels"][0]["key"]


def test_dashboard_embedding_is_script_safe(out_dir):
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    block = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert "</script" not in block and "<img" not in block  # '<' kaçırıldı
    texts = [e["x"] for e in _embedded(html)["examples"]]
    assert any(t.startswith("</script><img") for t in texts)  # ama veri bozulmadan geri okunur


def test_dashboard_javascript_syntax(out_dir, tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node yok")
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    js = re.findall(r"<script>(.*?)</script>", html, re.S)[0]
    f = tmp_path / "dash.js"
    f.write_text(js, encoding="utf-8")
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_build_data_is_json_serializable_and_deterministic(out_dir):
    a = json.dumps(build_data(out_dir), sort_keys=True)
    assert a == json.dumps(build_data(out_dir), sort_keys=True)
    assert render(build_data(out_dir)).count("<script") == 2
