"""'System One' karar protokolü: Jev (TypeSafe), Mercury Decide, Liquid d1, Instinct ...

İstek:  {"model", "state", "questions": {ad: {"type": "choice", "instructions", "criteria": {etiket: açıklama}}}}
Yanıt:  {"answers": {ad: {"type": "choice", "choice", "probabilities", "confidence"}}, "usage": {...}}
Alanlar TypeSafe, Inception (Mercury Decide) ve ZooData (Instinct) dokümanlarıyla doğrulandı.
"""
from __future__ import annotations

from ..core import Prediction, ProviderError, TaskSpec
from ..http import post_json
from ..prompts import QUESTION_NAME
from .base import Provider, match_label, normalize_probs, usage_tokens


class SystemOneProvider(Provider):
    def predict(self, text: str, task: TaskSpec, order=None) -> Prediction:
        labels = task.ordered(order)
        state = {"message": text} if self.opt("state_format") == "message" else text
        payload = {
            "model": self.spec.model,
            "state": state,
            "questions": {
                QUESTION_NAME: {
                    "type": "choice",
                    "instructions": task.instruction,
                    "criteria": labels,
                }
            },
        }
        payload.update(self.opt("extra_body", {}) or {})
        res = post_json(
            self.url(),
            headers={"Authorization": f"Bearer {self.require_key()}"},
            payload=payload, timeout=self.spec.timeout_s, max_retries=self.spec.max_retries,
        )
        ans = (res.data.get("answers") or {}).get(QUESTION_NAME)
        if not isinstance(ans, dict):
            raise ProviderError(f"Yanıtta '{QUESTION_NAME}' cevabı yok: {str(res.data)[:200]}")
        if ans.get("type") == "refusal" or ans.get("error"):
            raise ProviderError(f"Model reddetti/hata döndürdü: {str(ans)[:200]}")
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


def _f(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None
