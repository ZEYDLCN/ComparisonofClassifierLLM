"""Derin karşılaştırma: aynı örnekler üzerinde modelleri çiftler halinde ve örnek bazında inceler (derin_karsilastirma.md).

Ana rapordaki bootstrap aralıklarını tamamlar: McNemar tam testi (Holm düzeltmeli), referansın yanlış yaptığı
örneklerin dökümü, örnek zorluğu, tamamlayıcılık, tek sınıfa yığılma ve güven davranışı. Girdi, panelle aynı
veri biçimidir; `predictions.jsonl` olan koşu klasöründen ya da `report.html` içine gömülü veriden okunur.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .report import Table, num, pct, pp

ALPHA = 0.05
KINDS_TR = {"native": "yerel olasılık", "verbalized": "sözel güven", "derived": "türetilmiş"}
MAX_HATA_DOKUMU = 10  # örnek bazında dökümü verilecek en çok referans hatası
YIGILMA_KAT = 2.0   # bir sınıf, gerçek payının bu kadar katından fazla tahmin edilirse "yığılma" sayılır
YIGILMA_MIN = 0.25  # ... ve tahminlerin en az bu kadarını oluşturuyorsa


def load_data(path: str | Path) -> dict:
    """Koşu klasöründen (predictions.jsonl), birleşik klasörden (merged.json) ya da report.html içindeki gömülü panel verisinden okur."""
    p = Path(path)
    if (p / "merged.json").exists():  # merge komutunun çıktısı (zincirleme birleştirme / analiz için)
        return json.loads((p / "merged.json").read_text(encoding="utf-8"))
    if (p / "predictions.jsonl").exists():
        from .dashboard import build_data
        return build_data(p)
    html = p / "report.html"
    if html.exists():
        m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html.read_text(encoding="utf-8"), re.S)
        if m:
            return json.loads(m.group(1))
    raise FileNotFoundError(f"{p}: ne predictions.jsonl ne de gömülü veri içeren report.html bulundu")


def mcnemar_exact(b: int, c: int) -> float:
    """İki yönlü tam McNemar testi: b = yalnız birincinin, c = yalnız ikincinin doğru bildiği örnek sayısı."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pvals: list[float]) -> list[float]:
    """Holm basamaklı düzeltme (aile genelinde hata oranını korur); girdiyle aynı sırada döner."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run_max = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run_max
    return adj


def _scored(data: dict) -> list[dict]:
    return [m for m in data["models"] if m.get("m") and m.get("preds")]


def analyze(data: dict, reference: str | None = None) -> dict:
    labels = [l["key"] for l in data["labels"]]
    ex = data["examples"]
    truth = np.array([e["t"] for e in ex])
    models = _scored(data)
    if len(models) < 2:
        raise ValueError("Karşılaştırma için en az 2 puanlanmış model gerekir")
    by = {m["key"]: m for m in models}
    ref = reference or data["meta"].get("reference")
    if ref not in by:
        ref = max(models, key=lambda m: m["m"]["accuracy"])["key"]
    pred = {k: np.array([p[0] for p in m["preds"]]) for k, m in by.items()}
    ok = {k: (pred[k] == truth) & (truth >= 0) for k in by}
    n = len(ex)

    ranking = sorted(({"key": k, "display": m["display"], "group": m["group"], "n_correct": int(ok[k].sum()),
                       "acc": m["m"]["accuracy"], "ci": m["m"]["accuracyCI"]} for k, m in by.items()),
                     key=lambda r: -r["acc"])

    others = [k for k in by if k != ref]
    rows = []
    for k in others:
        b = int((ok[ref] & ~ok[k]).sum())      # yalnız referans doğru
        c = int((~ok[ref] & ok[k]).sum())      # yalnız model doğru
        rows.append({"key": k, "display": by[k]["display"], "only_ref": b, "only_model": c,
                     "both": int((ok[ref] & ok[k]).sum()), "neither": int((~ok[ref] & ~ok[k]).sum()),
                     "diff": float(ok[k].mean() - ok[ref].mean()), "p": mcnemar_exact(b, c)})
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        r["verdict"] = ("referans daha iyi" if r["only_ref"] > r["only_model"] else "model daha iyi") \
            if ph < ALPHA else "ayırt edilemez"
    rows.sort(key=lambda r: -by[r["key"]]["m"]["accuracy"])

    ref_errors = []
    for i in np.where(~ok[ref])[0]:
        ref_errors.append({
            "id": ex[i]["id"], "text": ex[i]["x"], "true": labels[truth[i]],
            "ref_pred": labels[pred[ref][i]] if pred[ref][i] >= 0 else "geçersiz", "ref_conf": by[ref]["preds"][i][1],
            "n_right": int(sum(ok[k][i] for k in others)),
            "preds": {by[k]["display"]: labels[pred[k][i]] if pred[k][i] >= 0 else "geçersiz" for k in others}})

    n_right = np.sum([ok[k] for k in by], axis=0)
    only_ref_right = [ex[i]["id"] for i in range(n) if ok[ref][i] and not any(ok[k][i] for k in others)]
    nobody = [ex[i]["id"] for i in range(n) if n_right[i] == 0]
    difficulty = sorted(Counter(int(x) for x in n_right).items())

    true_share = {l: float((truth == i).mean()) for i, l in enumerate(labels)}
    collapse = []
    for k, m in by.items():
        valid = pred[k][pred[k] >= 0]
        cnt = Counter(labels[i] for i in valid)
        top, top_n = cnt.most_common(1)[0] if cnt else ("—", 0)
        share = top_n / len(valid) if len(valid) else 0.0
        recall0 = [l for i, l in enumerate(labels) if (truth == i).any() and not ok[k][truth == i].any()]
        collapse.append({"key": k, "display": m["display"], "top": top, "share": share, "true_share": true_share.get(top, 0.0),
                         "flag": share >= YIGILMA_MIN and share >= YIGILMA_KAT * true_share.get(top, 1.0),
                         "invalid": int((pred[k] < 0).sum()), "never_right": recall0})

    conf = []
    for k, m in by.items():
        cf = np.array([np.nan if p[1] is None else p[1] for p in m["preds"]], dtype=float)
        have = ~np.isnan(cf)
        if have.sum() < 2:
            continue
        right, wrong = cf[have & ok[k]], cf[have & ~ok[k]]
        conf.append({"key": k, "display": m["display"], "kinds": ", ".join(KINDS_TR.get(x, x) for x in m["m"].get("confKinds") or []) or "—",
                     "right": float(right.mean()) if len(right) else None,
                     "wrong": float(wrong.mean()) if len(wrong) else None, "n_wrong": int(len(wrong))})
    return {"n": n, "labels": labels, "reference": ref, "ref_display": by[ref]["display"], "ranking": ranking,
            "paired": rows, "ref_errors": ref_errors, "difficulty": difficulty, "only_ref_right": only_ref_right,
            "nobody": nobody, "collapse": collapse, "confidence": conf, "n_models": len(by),
            "run_id": data["meta"]["run_id"], "dataset": data["meta"]["dataset"]}


def render_markdown(res: dict) -> str:
    ref, n = res["ref_display"], res["n"]
    out = [f"# Derin karşılaştırma — referans: {ref}", "",
           f"Koşu: `{res['run_id']}` · veri: `{res['dataset']['name']}` ({n} değerlendirme örneği, {res['n_models']} puanlanmış model)", "",
           "Bu belge ana raporu (`report.md`) tamamlar: aynı örnekler üzerinde modelleri **çiftler halinde ve örnek bazında** karşılaştırır. "
           f"{n} örnekte tek bir örneğin ağırlığı {num(100 / n)} puandır; bu yüzden küçük farklar yorumlanmamalıdır.", ""]

    out += ["## 1. Doğruluk sıralaması", "",
            Table(["#", "Model", "Doğru / toplam", "Doğruluk [%95 GA]"],
                  [[str(i + 1), r["display"], f"{r['n_correct']} / {n}", f"{pct(r['acc'])} [{pct(r['ci'][0], 0)}–{pct(r['ci'][1], 0)}]"]
                   for i, r in enumerate(res["ranking"])], ["r", "l", "r", "r"]).md(), ""]

    out += [f"## 2. {ref} ile eşlenmiş McNemar testi", "",
            "Yalnızca iki modelin **farklı** davrandığı örnekler bilgi taşır: *yalnız referans doğru* (b) ve *yalnız model doğru* (c). "
            "p değeri iki yönlü tam McNemar testidir; çok sayıda karşılaştırma yapıldığı için **Holm düzeltmeli** p ile karar verilir "
            f"(α = {num(ALPHA, 2)}). \"Ayırt edilemez\" = bu örneklemle fark kanıtlanamıyor, modellerin eşit olduğu anlamına gelmez.", "",
            Table(["Model", "Fark (puan)", "İkisi doğru", "Yalnız ref. doğru (b)", "Yalnız model doğru (c)", "İkisi de yanlış", "p", "p (Holm)", "Sonuç"],
                  [[r["display"], pp(r["diff"]), str(r["both"]), str(r["only_ref"]), str(r["only_model"]), str(r["neither"]),
                    _p(r["p"]), _p(r["p_holm"]), r["verdict"]] for r in res["paired"]],
                  ["l", "r", "r", "r", "r", "r", "r", "r", "l"]).md(), ""]

    out += [f"## 3. {ref} hangi örneklerde yanıldı?", ""]
    if res["ref_errors"]:
        out += [f"{len(res['ref_errors'])} yanlış var. \"Doğru bilen\" sütunu, diğer {res['n_models'] - 1} modelden kaçının bu örneği doğru bildiğidir. "
                "Etiketler gümüş etiketlidir: çok niyetli veya sınırda mesajlarda etiketin kendisi tartışmalı olabilir.", "",
                Table(["Mesaj", "Gerçek", ref, "Güven", "Doğru bilen"],
                      [[e["text"], e["true"], e["ref_pred"], num(e["ref_conf"], 2), f"{e['n_right']} / {res['n_models'] - 1}"]
                       for e in res["ref_errors"]], ["l", "l", "l", "r", "r"]).md(), ""]
        for e in res["ref_errors"][:MAX_HATA_DOKUMU]:
            out += [f"**\"{e['text']}\"** — gerçek: `{e['true']}`", "",
                    Table(["Model", "Tahmin", "Sonuç"],
                          [[ref, e["ref_pred"], "yanlış"]] + [[d, t, "doğru" if t == e["true"] else "yanlış"] for d, t in e["preds"].items()],
                          ["l", "l", "l"]).md(), ""]
        if len(res["ref_errors"]) > MAX_HATA_DOKUMU:
            out += [f"(İlk {MAX_HATA_DOKUMU} hata gösterildi; toplam {len(res['ref_errors'])}.)", ""]
    else:
        out += [f"{ref} bu örneklerde hiç yanılmadı.", ""]

    out += ["## 4. Örnek zorluğu", "",
            f"Her örneği kaç modelin doğru bildiği ({res['n_models']} modelden). Kimsenin bilemediği örnek, modelden çok **etiketin/örneğin** sorunlu olduğuna işaret eder.", "",
            Table(["Doğru bilen model sayısı", "Örnek sayısı"], [[str(k), str(v)] for k, v in res["difficulty"]], ["r", "r"]).md(), "",
            f"Hiçbir modelin bilemediği örnek: **{len(res['nobody'])}**" + (f" ({', '.join(res['nobody'])})" if res["nobody"] else "") + ". "
            f"Yalnızca referansın doğru bildiği örnek: **{len(res['only_ref_right'])}**"
            + (f" ({', '.join(res['only_ref_right'])})" if res["only_ref_right"] else "") + ".", ""]

    out += ["## 5. Tek sınıfa yığılma", "",
            f"Bir model tahminlerinin ≥%{YIGILMA_MIN * 100:.0f}'ini tek sınıfa veriyor ve bu pay o sınıfın gerçek payının ≥{YIGILMA_KAT:.0f} katıysa **yığılma** işaretlenir. "
            "Rastgele hata değil, sistematik bir sapma (ya da istem/etiket adı uyumsuzluğu) belirtisidir. \"Hiç bilemediği sınıf\": o sınıfın hiçbir örneğini doğru tahmin edememe.", "",
            Table(["Model", "En sık tahmin", "Payı", "Gerçek payı", "Yığılma", "Geçersiz", "Hiç bilemediği sınıflar"],
                  [[c["display"], c["top"], pct(c["share"], 0), pct(c["true_share"], 0), "**evet**" if c["flag"] else "hayır",
                    str(c["invalid"]), ", ".join(c["never_right"]) or "—"]
                   for c in sorted(res["collapse"], key=lambda c: -c["share"])], ["l", "l", "r", "r", "l", "r", "l"]).md(), ""]

    out += ["## 6. Güven davranışı", "",
            "Doğru ve yanlış yanıtlardaki ortalama güven. İyi bir modelde yanlış yanıtların güveni belirgin biçimde düşüktür; "
            "iki değer birbirine yakınsa model nerede yanıldığını bilmiyor demektir (seçici tahmin için işe yaramaz). "
            "Güven türleri aynı ölçekte değildir; yanlış sayısı çok az olduğunda ortalama güvenilir değildir.", "",
            Table(["Model", "Güven türü", "Doğrularda ort.", "Yanlışlarda ort.", "Yanlış sayısı"],
                  [[c["display"], c["kinds"], num(c["right"], 2), num(c["wrong"], 2), str(c["n_wrong"])] for c in res["confidence"]],
                  ["l", "l", "r", "r", "r"]).md(), ""]

    out += ["## Sınırlar", "",
            f"- **Örneklem:** {n} örnek. McNemar testinde yalnızca farklı davranılan örnekler sayılır; b+c küçükse test güçsüzdür.",
            "- **Eşit veri bütçesi yok:** eğitimli modeller `train` bölümünden öğrenir, sıfır örnekli modeller (API/encoder) öğrenmez; "
            "bu grupları kıyaslarken doğruluk tek başına adil bir ölçüt değildir.",
            "- **Gümüş etiket:** yerleşik veri insan etiketli altın küme değildir; sıralamayı değil, boru hattını ve eğilimleri gösterir.",
            "- **Tek koşu:** farklı tohum/örneklemde sonuçlar değişebilir; karar için kendi insan etiketli verinizle yeniden çalıştırın.", ""]
    return "\n".join(out)


def _p(x: float) -> str:
    return "<0,001" if x < 0.001 else num(x, 3)


def write_comparison(path: str | Path, reference: str | None = None, out: str | Path | None = None) -> Path:
    p = Path(path)
    target = Path(out) if out else (p if p.is_dir() else p.parent) / "derin_karsilastirma.md"
    target.write_text(render_markdown(analyze(load_data(p), reference)), encoding="utf-8")
    return target
