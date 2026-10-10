"""Her API protokolü sahte sunucuya karşı: istek şeması, yanıt ayrıştırma, hata yolları."""
import pytest

from clfbench.core import ProviderError
from clfbench.providers import make_provider

KARGO = "Kargom 6 gündür gelmedi, kurye arıyor musunuz?"
FATURA = "Kartımdan iki kez çekim yapılmış, faturayı kontrol edin."


def _predict(spec, task, text, order=None):
    p = make_provider(spec)
    p.prepare(task, [], [])
    return p.predict(text, task, order)


@pytest.mark.parametrize("provider,path", [
    ("systemone", "/v1/systemone"),
    ("systemone", "/v1/decisions"),
    ("openai_decisions", "/oai/v1/decisions"),
    ("chat_json", "/v1/chat/completions"),
    ("gemini", ""),
    ("claude", ""),
])
def test_provider_roundtrip(task_and_data, make_spec, provider, path):
    task, _ = task_and_data
    kw = {"options": {"effort": "low"}} if provider == "claude" else {}
    spec = make_spec(provider, path=path, model="m:x" if provider == "gemini" else "m", **kw)
    if provider == "gemini":
        spec.base_url += "/v1beta"
        spec.path = ""
    if provider == "chat_json":
        spec.base_url += ""
    pred = _predict(spec, task, KARGO)
    assert pred.label == "kargo_teslimat"
    assert pred.error is None and pred.latency_ms and pred.latency_ms > 0
    if provider != "openai_decisions":  # OpenAI Decisions dokümanı kullanım alanı belirtmiyor: çalıştırıcı tahmin eder
        assert pred.input_tokens and pred.input_tokens > 0
    assert pred.probs is None or abs(sum(pred.probs.values()) - 1) < 1e-6
    assert pred.conf_kind == ("native" if provider in ("systemone", "openai_decisions") else "verbalized")


def test_systemone_request_schema(task_and_data, make_spec):
    from mock_server import _Handler

    task, _ = task_and_data
    spec = make_spec("systemone", path="/v1/systemone", model="jev-1.13.0")
    _predict(spec, task, FATURA)
    path, headers, body = _Handler.log[-1]
    q = body["questions"]["routing"]
    assert body["model"] == "jev-1.13.0" and body["state"] == FATURA
    assert q["type"] == "choice" and q["instructions"] == task.instruction
    assert q["criteria"] == task.labels  # tüm etiketler + açıklamalar iletilir
    assert headers["Authorization"] == "Bearer good-key"


def test_label_order_is_forwarded(task_and_data, make_spec):
    from mock_server import _Handler

    task, _ = task_and_data
    order = list(reversed(task.label_keys))
    _predict(make_spec("systemone", path="/v1/systemone"), task, FATURA, order)
    _, _, body = _Handler.log[-1]
    assert list(body["questions"]["routing"]["criteria"]) == order


def test_openai_decisions_schema(task_and_data, make_spec):
    from mock_server import _Handler

    task, _ = task_and_data
    _predict(make_spec("openai_decisions", path="/oai/v1/decisions", model="gpt-6-luna"), task, FATURA)
    _, _, body = _Handler.log[-1]
    q = body["questions"][0]
    assert body["input"] == FATURA and q["type"] == "choice"
    assert [c["value"] for c in q["choices"]] == task.label_keys


def test_claude_uses_json_schema_without_forced_tools_or_temperature(task_and_data, make_spec):
    from mock_server import _Handler

    task, _ = task_and_data
    _predict(make_spec("claude", model="claude-haiku-5-5", options={"effort": "low"}), task, KARGO)
    _, headers, body = _Handler.log[-1]
    assert "tool_choice" not in body and "temperature" not in body  # Claude 5.x: zorunlu araç/örnekleme yok
    oc = body["output_config"]
    assert oc["effort"] == "low" and oc["format"]["type"] == "json_schema"
    assert oc["format"]["schema"]["properties"]["label"]["enum"] == task.label_keys


def test_bad_key_is_clear_error(task_and_data, make_spec):
    task, _ = task_and_data
    with pytest.raises(ProviderError, match="HTTP 401"):
        _predict(make_spec("systemone", path="/v1/systemone", key="BAD_API_KEY"), task, KARGO)


def test_missing_key_env(task_and_data, make_spec):
    task, _ = task_and_data
    spec = make_spec("systemone", path="/v1/systemone", key="NOPE_NOT_SET")
    ok, why = spec.availability()
    assert not ok and "NOPE_NOT_SET" in why


def test_invalid_label_is_error_not_silent(task_and_data, make_spec):
    task, _ = task_and_data
    with pytest.raises(ProviderError, match="Geçersiz etiket"):
        _predict(make_spec("systemone", path="/v1/systemone"), task, "__badlabel__ merhaba")


def test_retry_on_429_then_success(task_and_data, make_spec):
    task, _ = task_and_data
    pred = _predict(make_spec("systemone", path="/v1/systemone"), task, "__flaky__ kargo nerede")
    assert pred.label == "kargo_teslimat" and pred.retries == 1


def test_persistent_500_raises_after_retries(task_and_data, make_spec):
    task, _ = task_and_data
    with pytest.raises(ProviderError, match="500"):
        _predict(make_spec("systemone", path="/v1/systemone"), task, "__500__ x")


def test_systemone_omits_model_when_empty(task_and_data, make_spec):
    from mock_server import _Handler

    task, _ = task_and_data
    spec = make_spec("systemone", path="/v1/systemone", model="")
    pred = _predict(spec, task, KARGO)
    _, _, body = _Handler.log[-1]
    assert "model" not in body and pred.label == "kargo_teslimat"


def test_rate_limit_eta(make_spec):
    from clfbench.cli import rate_limit_minutes

    spec = make_spec("systemone", path="/v1/systemone", options={"rpm_limit": 1})
    assert rate_limit_minutes(spec, 152) == 152.0  # 1 istek/dk → 152 dk
    assert rate_limit_minutes(make_spec("systemone", path="/v1/systemone"), 152) is None
