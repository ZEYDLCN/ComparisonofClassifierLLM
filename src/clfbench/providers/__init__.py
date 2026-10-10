"""Sağlayıcı fabrikası."""
from __future__ import annotations

from ..config import ModelSpec
from .base import Provider
from .chat_json import ChatJSONProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .local import (EmbeddingProvider, GLiClassProvider, LayaProvider, NLIProvider, SetFitProvider,
                    TfidfLRProvider)
from .openai_decisions import OpenAIDecisionsProvider
from .systemone import SystemOneProvider

PROVIDERS: dict[str, type[Provider]] = {
    "systemone": SystemOneProvider,
    "openai_decisions": OpenAIDecisionsProvider,
    "chat_json": ChatJSONProvider,
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "tfidf_lr": TfidfLRProvider,
    "embedding": EmbeddingProvider,
    "nli": NLIProvider,
    "gliclass": GLiClassProvider,
    "laya": LayaProvider,
    "setfit": SetFitProvider,
}


def make_provider(spec: ModelSpec) -> Provider:
    try:
        cls = PROVIDERS[spec.provider]
    except KeyError:
        raise KeyError(f"Bilinmeyen sağlayıcı türü: {spec.provider!r}. Geçerli: {sorted(PROVIDERS)}") from None
    return cls(spec)
