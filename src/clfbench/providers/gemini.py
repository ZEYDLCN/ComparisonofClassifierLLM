"""Google Gemini — generateContent REST, JSON şemalı çıktı (responseSchema)."""
from __future__ import annotations

import json

from ..core import Prediction, ProviderError, TaskSpec
from ..http import post_json
from ..prompts import system_prompt, user_prompt
from .base import Provider, match_label, usage_tokens, verbalized_probs

DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(Provider):
    def __init__(self, spec):
        super().__init__(spec)
        self._shots = []

    def prepare(self, task, train, shots=None):
        self._shots = shots or []

    def predict(self, text: str, task: TaskSpec, order=None) -> Prediction:
        keys = list(task.ordered(order))
        base = (self.spec.base_url or DEFAULT_BASE).rstrip("/")
        url = f"{base}/models/{self.spec.model}:generateContent"
        gen_cfg = {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING", "enum": keys},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["label", "confidence"],
            },
            "maxOutputTokens": int(self.opt("max_tokens", 1024)),
        }
        if self.opt("temperature") is not None:
            gen_cfg["temperature"] = self.opt("temperature")
        gen_cfg.update(self.opt("generation_config", {}) or {})
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt(task, order, self._shots)}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt(text)}]}],
            "generationConfig": gen_cfg,
        }
        res = post_json(url, headers={"x-goog-api-key": self.require_key()}, payload=payload,
                        timeout=self.spec.timeout_s, max_retries=self.spec.max_retries)
        cands = res.data.get("candidates") or []
        if not cands:
            fb = res.data.get("promptFeedback")
            raise ProviderError(f"Aday yok (promptFeedback={str(fb)[:120]})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        body = next((p["text"] for p in reversed(parts) if p.get("text") and not p.get("thought")), None)
        if not body:
            raise ProviderError(f"Metin yok (finishReason={cands[0].get('finishReason')}); max_tokens artırılabilir")
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
        i, o = usage_tokens(res.data)
        return Prediction(label=label, probs=verbalized_probs(label, conf, keys) if conf is not None else None,
                          confidence=conf, conf_kind="verbalized", latency_ms=res.latency_ms,
                          input_tokens=i, output_tokens=o, retries=res.retries)
