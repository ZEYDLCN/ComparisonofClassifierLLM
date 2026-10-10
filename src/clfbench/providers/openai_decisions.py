"""OpenAI Decisions API (POST /v1/decisions, gpt-6-luna) — public beta.

İstek:  {"model", "input", "questions": [{"type": "choice", "name", "instructions", "choices": [{"value", "description"}]}]}
Yanıt:  {"answers": [{"type": "choice", "name", "choice", "probabilities": [{"value", "probability"}], "confidence"}]}
Kaynak: developers.openai.com/api/docs/guides/decisions (alanlar dokümandan alındı).
"""
from __future__ import annotations

from ..core import Prediction, ProviderError, TaskSpec
from ..http import post_json
from ..prompts import QUESTION_NAME
from .base import Provider, match_label, normalize_probs, usage_tokens
from .systemone import _f


class OpenAIDecisionsProvider(Provider):
    def predict(self, text: str, task: TaskSpec, order=None) -> Prediction:
        labels = task.ordered(order)
        payload = {
            "model": self.spec.model,
            "input": text,
            "questions": [{
                "type": "choice",
                "name": QUESTION_NAME,
                "instructions": task.instruction,
                "choices": [{"value": k, "description": v} for k, v in labels.items()],
            }],
        }
        payload.update(self.opt("extra_body", {}) or {})
        res = post_json(
            self.url(),
            headers={"Authorization": f"Bearer {self.require_key()}"},
            payload=payload, timeout=self.spec.timeout_s, max_retries=self.spec.max_retries,
        )
        answers = res.data.get("answers") or []
        ans = next((a for a in answers if isinstance(a, dict) and a.get("name") == QUESTION_NAME), None)
        if ans is None:
            raise ProviderError(f"Yanıtta '{QUESTION_NAME}' cevabı yok: {str(res.data)[:200]}")
        if ans.get("type") == "refusal":
            raise ProviderError(f"Model reddetti: {str(ans)[:200]}")
        keys = list(labels)
        label = match_label(ans.get("choice"), keys)
        probs = normalize_probs(ans.get("probabilities"), keys)
        # 'choice' alanı HİÇ yoksa en yüksek olasılığa düşülür; var ama geçersizse hata sayılır (sessizce düzeltilmez)
        if label is None and ans.get("choice") is None and probs:
            label = max(probs, key=probs.get)
        if label is None:
            raise ProviderError(f"Geçersiz etiket: {str(ans.get('choice'))[:80]!r}")
        conf = ans.get("confidence")
        if conf is None and probs:
            conf = probs.get(label)
        i, o = usage_tokens(res.data)
        return Prediction(label=label, probs=probs, confidence=_f(conf), conf_kind="native",
                          latency_ms=res.latency_ms, input_tokens=i, output_tokens=o, retries=res.retries)
