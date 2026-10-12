"""Koşuları birleştirme: aynı veri üzerinde ayrı oturumlarda alınmış sonuçları tek panelde ve tek analizde toplar.

Ücretsiz katmanlar (dakikada 1 istek, kota) ve farklı anahtarlar yüzünden modelleri tek seferde koşmak çoğu zaman
mümkün olmaz. Birleştirme yalnızca etiketler, örnek kimlikleri, gerçek etiketler ve veri parmak izi birebir aynıysa yapılır;
modellerin sonuçları değiştirilmez, yalnızca referansa göre farklar ve Pareto sınırları birleşik küme üzerinde yeniden hesaplanır.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .metrics import paired_bootstrap_diff, pareto_front

_RANK = {"ok": 3, "interrupted": 2, "failed": 2, "notrun": 1}


def _signature(d: dict):
    return ([l["key"] for l in d["labels"]], [(e["id"], e["t"]) for e in d["examples"]],
            d["meta"]["dataset"].get("fingerprint"))


def _rank(m: dict) -> int:
    return 3 if m.get("m") else _RANK.get(m.get("status"), 0)


def merge_data(datas: list[dict], reference: str | None = None, note: str = "") -> dict:
    """Panel verisi biçimindeki (dashboard.build_data çıktısı) koşuları birleştirir; girdileri değiştirmez."""
    if len(datas) < 2:
        raise ValueError("Birleştirmek için en az 2 koşu gerekir")
    sig = _signature(datas[0])
    for i, d in enumerate(datas[1:], 2):
        if _signature(d) != sig:
            raise ValueError(f"{i}. koşu farklı bir veri/örnek kümesi üzerinde alınmış (etiketler, örnek kimlikleri, gerçek "
                             "etiketler veya veri parmak izi uyuşmuyor); birleştirilemez")
    datas = copy.deepcopy(datas)
    models: dict[str, dict] = {}
    for d in datas:
        for m in d["models"]:
            m["source"] = d["meta"]["run_id"]
            cur = models.get(m["key"])
            # puanlanmış > hatalı/yarım > çalıştırılmadı; eşit rütbede sonraki koşu (daha yeni) öncekini geçer yalnızca puanlıysa
            if cur is None or _rank(m) > _rank(cur) or (m.get("m") and cur.get("m")):
                models[m["key"]] = m
    base = datas[0]
    scored = [m for m in models.values() if m.get("m") and m.get("preds")]
    truth = np.array([e["t"] for e in base["examples"]])
    ok = {m["key"]: np.array([int(p[0] == t and t >= 0) for p, t in zip(m["preds"], truth)]) for m in scored}
    ref = reference if reference in ok else (base["meta"].get("reference") if base["meta"].get("reference") in ok else
                                             max(scored, key=lambda m: m["m"]["accuracy"])["key"] if scored else None)
    seed = base["meta"]["settings"].get("seed", 0)
    for m in scored:
        m.pop("vsRef", None)
        if m["key"] != ref:
            diff, lo, hi, _ = paired_bootstrap_diff(ok[m["key"]], ok[ref], seed=seed)
            m["vsRef"] = {"reference": ref, "diff": diff, "lo": lo, "hi": hi}
    pareto = {"latency": sorted(pareto_front({m["key"]: (m["m"]["accuracy"], m["m"]["latP50"]) for m in scored}, (True, False))),
              "cost": sorted(pareto_front({m["key"]: (m["m"]["accuracy"], m["m"]["cost1000"]) for m in scored}, (True, False)))}

    sources = [{"run_id": d["meta"]["run_id"], "started": d["meta"]["started"],
                "platform": (d["meta"].get("environment") or {}).get("platform"),
                "python": (d["meta"].get("environment") or {}).get("python")} for d in datas]
    auto = ("Birleştirilmiş koşu — kaynaklar: " + "; ".join(f"{s['run_id']} ({s['platform'] or '?'})" for s in sources) +
            ". Modeller ayrı oturumlarda, aynı örnekler ve aynı ayarlarla koşuldu; gecikmeler farklı makine ve ağlardan geldiği için "
            "doğrudan kıyaslanamaz.")
    meta = dict(base["meta"])
    meta.update(run_id=" + ".join(s["run_id"] for s in sources), started=max(s["started"] for s in sources), reference=ref,
                sources=sources, note=" ".join(x for x in [(base["meta"].get("note") or "").strip(), auto, note.strip()] if x))
    return {"meta": meta, "labels": base["labels"], "examples": base["examples"], "models": list(models.values()), "pareto": pareto}


def write_merged(dirs: list[str | Path], out: str | Path, reference: str | None = None, note: str = "") -> dict:
    """Koşu klasörlerini birleştirip out/ altına merged.json, report.html ve derin_karsilastirma.md yazar."""
    from . import dashboard
    from .compare import analyze, load_data, render_markdown

    merged = merge_data([load_data(d) for d in dirs], reference, note)
    md = render_markdown(analyze(merged, reference))   # önce analiz: başarısız olursa yarım çıktı bırakılmaz
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "merged.json").write_text(json.dumps(merged, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out / "report.html").write_text(dashboard.render(merged), encoding="utf-8")
    (out / "derin_karsilastirma.md").write_text(md, encoding="utf-8")
    return merged
