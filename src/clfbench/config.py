"""Model kayıt defteri (configs/models.yaml) ve .env yükleme."""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .paths import CONFIG_DIR, ROOT


def load_env() -> None:
    """Depo kökündeki .env dosyasını (varsa) ortam değişkenlerine yükler. Mevcut değişkenleri ezmez."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(ROOT / ".env", override=False)


@dataclass
class ModelSpec:
    key: str
    display: str
    group: str  # karar-api | uretken-llm | encoder | egitimli
    provider: str
    model: str = ""
    base_url: str = ""
    path: str = ""
    api_key_env: str = ""
    enabled: bool = True
    price_in: Optional[float] = None   # USD / 1M girdi tokenı
    price_out: Optional[float] = None  # USD / 1M çıktı tokenı
    price_note: str = ""
    concurrency: int = 4
    timeout_s: float = 60.0
    max_retries: int = 4
    notes: str = ""
    verified: str = ""  # arayüzün neye göre doğrulandığı
    requires: list[str] = field(default_factory=list)  # gerekli Python paketleri
    options: dict = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return not self.api_key_env and not self.base_url and self.provider in LOCAL_PROVIDERS

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "").strip() if self.api_key_env else ""

    def missing_requirements(self) -> list[str]:
        return [m for m in self.requires if importlib.util.find_spec(m.split(".")[0]) is None]

    def availability(self) -> tuple[bool, str]:
        """(kullanılabilir mi, neden değilse açıklama)"""
        if self.api_key_env and not self.api_key():
            return False, f"{self.api_key_env} tanımlı değil"
        miss = self.missing_requirements()
        if miss:
            return False, "paket eksik: " + ", ".join(miss)
        if self.provider in ("systemone", "openai_decisions", "chat_json") and not self.base_url:
            return False, "base_url ayarlanmamış (configs/models.yaml)"
        return True, "hazır"


LOCAL_PROVIDERS = {"tfidf_lr", "embedding", "nli", "gliclass", "laya", "setfit"}

_KNOWN = {f for f in ModelSpec.__dataclass_fields__}


def load_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    path = Path(path) if path else CONFIG_DIR / "models.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults", {}) or {}
    out: dict[str, ModelSpec] = {}
    for item in raw.get("models", []):
        merged = {**defaults, **item}
        price = merged.pop("price", None) or {}
        if "input" in price:
            merged["price_in"] = price.get("input")
        if "output" in price:
            merged["price_out"] = price.get("output")
        if "note" in price:
            merged["price_note"] = price.get("note")
        options = dict(merged.pop("options", {}) or {})
        kwargs = {k: v for k, v in merged.items() if k in _KNOWN}
        unknown = set(merged) - _KNOWN
        if unknown:  # bilinmeyen alanlar sessizce kaybolmasın
            options.update({k: merged[k] for k in unknown})
        spec = ModelSpec(**kwargs, options=options)
        if spec.key in out:
            raise ValueError(f"models.yaml içinde yinelenen anahtar: {spec.key}")
        out[spec.key] = spec
    return out


def resolve_models(registry: dict[str, ModelSpec], requested: list[str] | None,
                   groups: list[str] | None = None, include_disabled: bool = False
                   ) -> tuple[list[ModelSpec], list[tuple[ModelSpec, str]]]:
    """İstenen modelleri seçer. (çalıştırılacaklar, atlananlar+neden) döndürür."""
    if requested:
        unknown = [k for k in requested if k not in registry]
        if unknown:
            raise KeyError(f"Bilinmeyen model anahtarı: {unknown}. 'clfbench models' ile listeleyin.")
        candidates = [registry[k] for k in requested]
        explicit = True
    else:
        candidates = list(registry.values())
        explicit = False
    if groups:
        candidates = [m for m in candidates if m.group in groups]
    run, skipped = [], []
    for m in candidates:
        if not m.enabled and not (explicit or include_disabled):
            continue
        ok, why = m.availability()
        (run if ok else skipped).append(m if ok else (m, why))
    return run, skipped
