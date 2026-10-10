"""Sağlayıcı temel sınıfı ve ortak yardımcılar."""
from __future__ import annotations

from typing import Any, Optional

from ..config import ModelSpec
from ..core import Example, Prediction, ProviderError, TaskSpec


class Provider:
    """Bir modelin tek örnek üzerinden sınıflandırma arayüzü."""

    needs_training = False  # True ise prepare() train bölümünden öğrenir

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    # --- yaşam döngüsü ---------------------------------------------------
    def prepare(self, task: TaskSpec, train: list[Example], shots: list[Example] | None = None) -> None:
        """Model yükleme / eğitim / few-shot örnekleri. Süre ayrıca 'kurulum süresi' olarak raporlanır."""

    def predict(self, text: str, task: TaskSpec, order: list[str] | None = None) -> Prediction:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # --- yardımcılar -----------------------------------------------------
    def opt(self, name: str, default: Any = None) -> Any:
        return self.spec.options.get(name, default)

    def require_key(self) -> str:
        key = self.spec.api_key()
        if not key:
            raise ProviderError(f"{self.spec.api_key_env} tanımlı değil")
        return key

    def url(self) -> str:
        return self.spec.base_url.rstrip("/") + "/" + self.spec.path.lstrip("/") if self.spec.path else self.spec.base_url


def match_label(raw: Any, labels: list[str]) -> Optional[str]:
    """Modelin döndürdüğü etiketi geçerli etiketlerle eşleştirir (büyük/küçük harf ve boşluk toleranslı)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s in labels:
        return s
    low = {l.lower(): l for l in labels}
    return low.get(s.lower().strip("`'\" "))


def normalize_probs(raw: Any, labels: list[str]) -> Optional[dict[str, float]]:
    """{etiket: p} veya [{value|label|choice, probability}] -> tüm etiketleri kapsayan, toplamı 1 olan sözlük."""
    if raw is None:
        return None
    pairs: dict[str, float] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for it in raw:
            if isinstance(it, dict):
                k = it.get("value", it.get("label", it.get("choice")))
                v = it.get("probability", it.get("score", it.get("p")))
                items.append((k, v))
    else:
        return None
    for k, v in items:
        lab = match_label(k, labels)
        try:
            if lab is not None and v is not None:
                pairs[lab] = float(v)
        except (TypeError, ValueError):
            continue
    if not pairs:
        return None
    total = sum(max(p, 0.0) for p in pairs.values())
    if total <= 0:
        return None
    return {l: max(pairs.get(l, 0.0), 0.0) / total for l in labels}


def verbalized_probs(label: str, confidence: float, labels: list[str]) -> dict[str, float]:
    """Sözel güvenden dağılım türet: seçilen etikete c, kalan kütle diğerlerine eşit dağıtılır."""
    c = min(max(float(confidence), 0.0), 1.0)
    rest = (1.0 - c) / max(len(labels) - 1, 1)
    return {l: (c if l == label else rest) for l in labels}


def usage_tokens(data: dict) -> tuple[Optional[int], Optional[int]]:
    """Farklı API'lerdeki kullanım alanlarını ortak (girdi, çıktı) çiftine çevirir."""
    for u in (data.get("usage"), (data.get("meta") or {}).get("usage"), data.get("usageMetadata")):
        if not isinstance(u, dict):
            continue
        i = u.get("input_tokens", u.get("prompt_tokens", u.get("promptTokenCount")))
        o = u.get("output_tokens", u.get("completion_tokens"))
        if o is None and "candidatesTokenCount" in u:
            o = (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
        if i is not None or o is not None:
            return (int(i) if i is not None else None, int(o) if o is not None else None)
    return None, None
