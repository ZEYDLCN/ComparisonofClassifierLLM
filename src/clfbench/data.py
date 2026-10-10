"""Veri kümesi yükleme ve seed veri üretimi.

Desteklenen kaynaklar (``--dataset``):
  destek_tr                     yerleşik Türkçe müşteri mesajı yönlendirme seti (data/destek_tr.jsonl)
  yol/dosya.jsonl               JSONL: id?, text, label, split?, tags?  (+ yanında dosya.labels.json)
  yol/dosya.csv                 CSV: text,label[,split,id,tags]        (+ yanında dosya.labels.json)
  hf:ad[:config]:metin:etiket   Hugging Face veri kümesi ('datasets' paketi gerekir)
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from .core import Example, TaskSpec
from .paths import DATA_DIR

SPLITS = ("train", "dev", "test")
SPLIT_FRACTIONS = {"train": 0.40, "dev": 0.15}  # kalan: test


def _stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- seed -> jsonl
_TAG_RE = re.compile(r"^\[([a-z_, ]+)\]\s*(.+)$")


def parse_seed_line(line: str) -> tuple[tuple[str, ...], str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    m = _TAG_RE.match(line)
    tags: tuple[str, ...] = ()
    if m:
        tags = tuple(t.strip() for t in m.group(1).split(",") if t.strip())
        line = m.group(2).strip()
    return tags, line


def auto_tags(text: str, tags: tuple[str, ...]) -> tuple[str, ...]:
    out = list(tags)
    n_words = len(text.split())
    if n_words <= 4 and "short" not in out:
        out.append("short")
    if n_words >= 22 and "long" not in out:
        out.append("long")
    return tuple(out)


def build_seed_dataset(seed_dir: Path | None = None, labels_path: Path | None = None,
                       out_path: Path | None = None) -> Path:
    """data/seed/*.txt -> data/destek_tr.jsonl (etiket bazında deterministik, katmanlı bölme)."""
    seed_dir = seed_dir or DATA_DIR / "seed"
    labels_path = labels_path or DATA_DIR / "destek_tr.labels.json"
    out_path = out_path or DATA_DIR / "destek_tr.jsonl"
    spec = json.loads(labels_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for label in spec["labels"]:
        f = seed_dir / f"{label}.txt"
        if not f.exists():
            raise FileNotFoundError(f"Seed dosyası yok: {f}")
        items = []
        seen: set[str] = set()
        for raw in f.read_text(encoding="utf-8").splitlines():
            parsed = parse_seed_line(raw)
            if not parsed:
                continue
            tags, text = parsed
            if text in seen:
                raise ValueError(f"Aynı etikette yinelenen metin: {text!r}")
            seen.add(text)
            items.append((tags, text))
        items.sort(key=lambda it: _stable_hash(it[1]))
        n = len(items)
        n_train = round(n * SPLIT_FRACTIONS["train"])
        n_dev = round(n * SPLIT_FRACTIONS["dev"])
        for i, (tags, text) in enumerate(items):
            split = "train" if i < n_train else "dev" if i < n_train + n_dev else "test"
            rows.append({
                "id": f"{label}-{_stable_hash(text)[:8]}",
                "text": text,
                "label": label,
                "split": split,
                "tags": list(auto_tags(text, tags)),
            })
    # farklı etiketlerde aynı metin varsa belirsizdir: uyar
    by_text: dict[str, set[str]] = {}
    for r in rows:
        by_text.setdefault(r["text"], set()).add(r["label"])
    dup = [t for t, ls in by_text.items() if len(ls) > 1]
    if dup:
        raise ValueError(f"Farklı etiketlerde aynı metin var: {dup[:3]}")
    rows.sort(key=lambda r: (r["split"], r["label"], r["id"]))
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


# --------------------------------------------------------------------------- yükleme
def _load_task(labels_path: Path) -> TaskSpec:
    d = json.loads(labels_path.read_text(encoding="utf-8"))
    return TaskSpec(
        name=d.get("task", labels_path.stem.replace(".labels", "")),
        instruction=d["instruction"],
        labels=d["labels"],
        short=d.get("short", {}),
        notes=d.get("notes", ""),
    )


def _example_from_row(row: dict, i: int) -> Example:
    text = str(row["text"]).strip()
    label = str(row["label"]).strip()
    tags = row.get("tags") or ()
    if isinstance(tags, str):
        tags = tuple(t for t in re.split(r"[;,| ]+", tags) if t)
    return Example(
        id=str(row.get("id") or f"r{i}-{_stable_hash(text)[:8]}"),
        text=text,
        label=label,
        split=str(row.get("split") or "test"),
        tags=tuple(tags),
    )


def load_dataset(spec: str, limit_per_class: int | None = None) -> tuple[TaskSpec, list[Example]]:
    """Veri kümesini yükler, (görev, örnekler) döndürür."""
    if spec == "destek_tr":
        path = DATA_DIR / "destek_tr.jsonl"
        if not path.exists():
            build_seed_dataset()
        spec = str(path)
    if spec.startswith("hf:"):
        return _load_hf(spec)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Veri kümesi bulunamadı: {path}")
    labels_path = path.with_suffix(".labels.json")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Etiket tanım dosyası yok: {labels_path}\n"
            "Biçim için data/destek_tr.labels.json dosyasına bakın (instruction + labels)."
        )
    task = _load_task(labels_path)
    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        raise ValueError(f"Desteklenmeyen veri biçimi: {path.suffix} (jsonl veya csv kullanın)")
    examples = [_example_from_row(r, i) for i, r in enumerate(rows)]
    _validate(task, examples)
    if limit_per_class:
        examples = _limit_per_class(examples, limit_per_class)
    return task, examples


def _limit_per_class(examples: list[Example], k: int) -> list[Example]:
    counts: dict[tuple[str, str], int] = {}
    out = []
    for ex in examples:
        key = (ex.split, ex.label)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= k:
            out.append(ex)
    return out


def _validate(task: TaskSpec, examples: list[Example]) -> None:
    unknown = {e.label for e in examples} - set(task.labels)
    if unknown:
        raise ValueError(f"Veride tanımsız etiketler var: {sorted(unknown)}")
    ids = [e.id for e in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Veride yinelenen örnek kimlikleri (id) var.")
    bad = {e.split for e in examples} - set(SPLITS)
    if bad:
        raise ValueError(f"Geçersiz split değerleri: {sorted(bad)} (train/dev/test olmalı)")


def _load_hf(spec: str) -> tuple[TaskSpec, list[Example]]:
    """hf:<ad>[:config]:<metin_sütunu>:<etiket_sütunu>  — etiketler ClassLabel adlarından alınır."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Hugging Face veri kümesi için: pip install datasets") from e
    parts = spec.split(":")[1:]
    if len(parts) == 3:
        name, text_col, label_col = parts
        config = None
    elif len(parts) == 4:
        name, config, text_col, label_col = parts
    else:
        raise ValueError("Biçim: hf:<ad>[:config]:<metin_sütunu>:<etiket_sütunu>")
    ds = hf_load(name, config) if config else hf_load(name)
    feat = next(iter(ds.values())).features[label_col]
    names = list(getattr(feat, "names", []))
    if not names:
        raise ValueError("Etiket sütunu ClassLabel olmalı.")
    task = TaskSpec(
        name=name.replace("/", "_"),
        instruction="Aşağıdaki metni verilen kategorilerden tam olarak birine ata.",
        labels={n: n.replace("_", " ") for n in names},
    )
    examples: list[Example] = []
    split_map = {"train": "train", "validation": "dev", "dev": "dev", "test": "test"}
    for split_name, d in ds.items():
        our = split_map.get(split_name)
        if not our:
            continue
        for i, row in enumerate(d):
            examples.append(Example(
                id=f"{name}-{split_name}-{i}", text=str(row[text_col]).strip(),
                label=names[row[label_col]], split=our))
    return task, examples


def select(examples: list[Example], split: str) -> list[Example]:
    """'test' | 'dev' | 'dev+test' — eğitim bölümü hiçbir zaman değerlendirilmez."""
    allowed = {"test": {"test"}, "dev": {"dev"}, "dev+test": {"dev", "test"}}
    if split not in allowed:
        raise ValueError(f"Geçersiz split: {split} (test, dev, dev+test)")
    return [e for e in examples if e.split in allowed[split]]


def dataset_fingerprint(task: TaskSpec, examples: list[Example]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([task.instruction, task.labels], ensure_ascii=False, sort_keys=True).encode())
    for e in sorted(examples, key=lambda x: x.id):
        h.update(f"{e.id}\t{e.label}\t{e.split}\t{e.text}\n".encode())
    return h.hexdigest()[:16]
