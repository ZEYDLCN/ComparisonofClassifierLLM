"""OpenAI uyumlu /chat/completions ile üretken sınıflandırma.

OpenAI, OpenRouter, Inception (Mercury 2.5), DeepSeek, Together ... — base_url + model + anahtar yeterli.
Model, JSON {"label", "confidence"} yazar; güven değeri 'sözel' (verbalized) güvendir, yerel olasılık değildir.
"""
from __future__ import annotations

from ..core import Prediction, ProviderError, TaskSpec
from ..http import post_json
from ..prompts import parse_label_json, system_prompt, user_prompt
from .base import Provider, match_label, usage_tokens, verbalized_probs


class ChatJSONProvider(Provider):
    def __init__(self, spec):
        super().__init__(spec)
        self._shots = []

    def prepare(self, task, train, shots=None):
        self._shots = shots or []

    def predict(self, text: str, task: TaskSpec, order=None) -> Prediction:
        keys = list(task.ordered(order))
        payload = {
            "model": self.spec.model,
            "messages": [
                {"role": "system", "content": system_prompt(task, order, self._shots)},
                {"role": "user", "content": user_prompt(text)},
            ],
            self.opt("max_tokens_param", "max_tokens"): int(self.opt("max_tokens", 1024)),
        }
        if self.opt("temperature") is not None:
            payload["temperature"] = self.opt("temperature")
        if self.opt("json_mode", False):
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.opt("extra_body", {}) or {})
        headers = {"Authorization": f"Bearer {self.require_key()}"}
        headers.update(self.opt("extra_headers", {}) or {})
        res = post_json(self.url(), headers=headers, payload=payload,
                        timeout=self.spec.timeout_s, max_retries=self.spec.max_retries)
        i, o = usage_tokens(res.data)
        try:
            msg = res.data["choices"][0]["message"]
            content = msg.get("content")
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"Beklenmeyen yanıt biçimi: {str(res.data)[:200]}") from e
        if not content:
            reason = (res.data["choices"][0].get("finish_reason") or "?")
            raise ProviderError(f"Boş yanıt (finish_reason={reason}); max_tokens artırılabilir")
        try:
            obj = parse_label_json(content)
        except ValueError as e:
            raise ProviderError(f"JSON ayrıştırılamadı: {content[:120]!r}") from e
        label = match_label(obj.get("label"), keys)
        if label is None:
            raise ProviderError(f"Geçersiz etiket: {str(obj.get('label'))[:80]!r}")
        try:
            conf = float(obj.get("confidence"))
        except (TypeError, ValueError):
            conf = None
        probs = verbalized_probs(label, conf, keys) if conf is not None else None
        return Prediction(label=label, probs=probs, confidence=conf, conf_kind="verbalized",
                          latency_ms=res.latency_ms, input_tokens=i, output_tokens=o, retries=res.retries)
