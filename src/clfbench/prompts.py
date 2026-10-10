"""Üretken LLM'ler için ortak prompt ve JSON şeması.

Karar API'leri (Jev, Mercury Decide, d1 ...) kendi şemalarını kullanır; burada yalnızca
sohbet/üretken modeller için tek tip bir istem tanımlanır ki modeller aynı talimatla karşılaştırılsın.
"""
from __future__ import annotations

import json
import random

from .core import Example, TaskSpec

# Prompt sürümü: değişirse önbellek (cache) otomatik geçersiz olur.
PROMPT_VERSION = "v1"

# Karar API'lerinde tek soru için kullanılan isim
QUESTION_NAME = "routing"


def system_prompt(task: TaskSpec, order: list[str] | None = None, shots: list[Example] | None = None) -> str:
    labels = task.ordered(order)
    lines = [task.instruction, "", "Kategoriler:"]
    for key, desc in labels.items():
        lines.append(f"- {key}: {desc}")
    if task.notes:
        lines += ["", f"Not: {task.notes}"]
    lines += [
        "",
        "Yalnızca şu biçimde tek bir JSON nesnesi döndür, başka hiçbir şey yazma:",
        '{"label": "<kategori_anahtarı>", "confidence": <0 ile 1 arasında sayı>}',
        "'label' değeri yukarıdaki kategori anahtarlarından biri olmalıdır. "
        "'confidence' seçimine ne kadar güvendiğini gösterir.",
    ]
    if shots:
        lines += ["", "Örnekler:"]
        for s in shots:
            lines.append(f"Mesaj: {s.text}\nCevap: " + json.dumps({"label": s.label, "confidence": 0.95}, ensure_ascii=False))
    return "\n".join(lines)


def user_prompt(text: str) -> str:
    return f"Mesaj:\n{text}"


def json_schema(task: TaskSpec, order: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(task.ordered(order).keys())},
            "confidence": {"type": "number"},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }


def pick_shots(train: list[Example], task: TaskSpec, k_per_label: int, seed: int = 0) -> list[Example]:
    """Etiket başına k adet deterministik örnek; sıra karıştırılır (etiket sırası ipucu olmasın)."""
    if k_per_label <= 0:
        return []
    rng = random.Random(seed)
    shots: list[Example] = []
    for label in task.label_keys:
        pool = sorted((e for e in train if e.label == label), key=lambda e: e.id)
        rng.shuffle(pool)
        shots.extend(pool[:k_per_label])
    rng.shuffle(shots)
    return shots


def parse_label_json(text: str) -> dict:
    """Model çıktısından JSON nesnesini ayıklar (kod çiti / ek metin toleranslı)."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def estimate_tokens(text: str) -> int:
    """Kabaca token tahmini (Türkçe için ~3,2 karakter/token). Yalnızca maliyet ÖN-tahmini için."""
    return max(1, round(len(text) / 3.2))
