import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clfbench.config import ModelSpec  # noqa: E402
from clfbench.data import load_dataset  # noqa: E402
from mock_server import start_mock  # noqa: E402


@pytest.fixture(scope="session")
def mock_url():
    srv, url = start_mock()
    yield url
    srv.shutdown()


@pytest.fixture(autouse=True)
def _keys(monkeypatch, mock_url):
    monkeypatch.setenv("MOCK_API_KEY", "good-key")
    monkeypatch.setenv("BAD_API_KEY", "wrong")
    monkeypatch.setenv("BROKE_API_KEY", "broke-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_url)


@pytest.fixture(scope="session")
def task_and_data():
    return load_dataset("destek_tr")


@pytest.fixture
def make_spec(mock_url):
    def _make(provider, path="", model="m", key="MOCK_API_KEY", **kw):
        opts = kw.pop("options", {})
        return ModelSpec(key=f"t_{provider}", display=f"test {provider}", group="karar-api", provider=provider,
                         model=model, base_url=mock_url, path=path, api_key_env=key, price_in=1.0, price_out=2.0,
                         max_retries=2, options=opts, **kw)
    return _make
