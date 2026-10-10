"""Claude — resmi Anthropic SDK + yapılandırılmış çıktı (output_config.format, JSON şeması).

Claude 5.x notları (resmi dokümandan): zorunlu tool_choice (any/tool) Opus 5.5 / Sonnet 5.5 / Fable 5.1'de 400 verir,
temperature gibi örnekleme parametreleri bazı modellerde reddedilir. Bu yüzden zorunlu araç kullanımı yerine
JSON-şema çıktısı kullanılır, örnekleme parametresi GÖNDERİLMEZ. 'effort' düşük tutulur: sınıflandırma için yeterli ve ucuzdur.
"""
from __future__ import annotations

import json
import time

from ..core import Prediction, ProviderError, TaskSpec
from ..prompts import json_schema, system_prompt, user_prompt
from .base import Provider, match_label, verbalized_probs


class ClaudeProvider(Provider):
    def __init__(self, spec):
        super().__init__(spec)
        self._client = None
        self._shots = []

    def prepare(self, task, train, shots=None):
        self._shots = shots or []
        import anthropic  # noqa: F401  (paket yoksa erken ve anlaşılır hata)
        self._client = anthropic.Anthropic(
            api_key=self.require_key(), timeout=self.spec.timeout_s, max_retries=min(self.spec.max_retries, 4))

    def predict(self, text: str, task: TaskSpec, order=None) -> Prediction:
        import anthropic

        if self._client is None:
            self.prepare(task, [])
        keys = list(task.ordered(order))
        output_config: dict = {"format": {"type": "json_schema", "schema": json_schema(task, order)}}
        if self.opt("effort"):
            output_config["effort"] = self.opt("effort")
        t0 = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=self.spec.model,
                max_tokens=int(self.opt("max_tokens", 1024)),
                system=system_prompt(task, order, self._shots),
                messages=[{"role": "user", "content": user_prompt(text)}],
                extra_body={"output_config": output_config},
            )
        except anthropic.APIStatusError as e:
            raise ProviderError(f"HTTP {e.status_code}: {str(e.message)[:250]}") from e
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise ProviderError(f"ağ hatası: {type(e).__name__}") from e
        latency = (time.perf_counter() - t0) * 1000
        if getattr(resp, "stop_reason", None) == "refusal":
            raise ProviderError("Model reddetti (stop_reason=refusal)")
        body = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)
        if not body:
            raise ProviderError(f"Metin bloğu yok (stop_reason={resp.stop_reason}); max_tokens artırılabilir")
        try:
            obj = json.loads(body)
        except ValueError as e:
            raise ProviderError(f"JSON ayrıştırılamadı: {body[:120]!r}") from e
        label = match_label(obj.get("label"), keys)
        if label is None:
            raise ProviderError(f"Geçersiz etiket: {str(obj.get('label'))[:80]!r}")
        try:
            conf = float(obj.get("confidence"))
        except (TypeError, ValueError):
            conf = None
        u = resp.usage
        return Prediction(
            label=label, probs=verbalized_probs(label, conf, keys) if conf is not None else None,
            confidence=conf, conf_kind="verbalized", latency_ms=latency,
            input_tokens=getattr(u, "input_tokens", None), output_tokens=getattr(u, "output_tokens", None),
        )
