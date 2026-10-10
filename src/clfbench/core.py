"""Ortak veri tipleri."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str
    split: str = "test"
    tags: tuple[str, ...] = ()


@dataclass
class TaskSpec:
    """Sınıflandırma görevi: talimat + etiket sözlüğü (etiket -> açıklama)."""

    name: str
    instruction: str
    labels: dict[str, str]
    # Encoder / NLI modelleri için kısa, doğal dil etiket adları (etiket -> kısa ifade)
    short: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def label_keys(self) -> list[str]:
        return list(self.labels.keys())

    def short_text(self, key: str) -> str:
        return self.short.get(key) or self.labels.get(key) or key

    def ordered(self, order: list[str] | None) -> dict[str, str]:
        keys = order or self.label_keys
        return {k: self.labels[k] for k in keys}


@dataclass
class Prediction:
    """Bir modelin tek bir örnek için çıktısı."""

    label: Optional[str] = None
    probs: Optional[dict[str, float]] = None
    confidence: Optional[float] = None
    # native: modelin kendi olasılığı | verbalized: modelin yazdığı güven değeri | derived: türetilmiş skor
    conf_kind: str = "native"
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tokens_estimated: bool = False
    retries: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.label is not None


class ProviderError(Exception):
    """Sağlayıcı çağrısı başarısız (ağ, HTTP, ayrıştırma, geçersiz etiket...)."""


def shuffled_order(keys: list[str], seed: int, example_id: str) -> list[str]:
    """Örnek başına deterministik etiket sırası (seçenek sırası dayanıklılık testi)."""
    rng = random.Random(f"{seed}:{example_id}")
    out = list(keys)
    rng.shuffle(out)
    return out
