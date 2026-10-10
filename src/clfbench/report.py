"""Rapor üretimi: Markdown + tek dosyalık HTML (grafikler gömülü) + PNG grafikler."""
from __future__ import annotations

import base64
import html as _html
import json
from pathlib import Path

from . import charts

GROUP_TR = {"karar-api": "Karar API", "uretken-llm": "Üretken LLM", "encoder": "Encoder", "egitimli": "Eğitimli"}
SLICE_TR = {"ascii": "Türkçe karaktersiz (ascii)", "typo": "Yazım hatalı", "short": "Çok kısa (≤4 sözcük)",
            "multi": "Çok niyetli", "long": "Uzun (≥22 sözcük)"}


# ------------------------------------------------------------------------------------ biçimlendirme
def pct(x, nd=1):
    """Türkçe yazım: %70,2"""
    return "—" if x is None else "%" + f"{x * 100:.{nd}f}".replace(".", ",")


def pp(x, nd=1):
    """Yüzde puan farkı: +3,2 puan"""
    return "—" if x is None else f"{x * 100:+.{nd}f}".replace(".", ",") + " puan"


def num(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def ms(x):
    if x is None:
        return "—"
    return f"{x:,.0f}".replace(",", ".") if x >= 100 else num(x, 1)


def usd(x, est=False):
    if x is None:
        return "—"
    s = f"{x:.4f}".replace(".", ",") if x < 1 else f"{x:.2f}".replace(".", ",")
    return ("~" if est else "") + s


class Table:
    def __init__(self, headers: list[str], rows: list[list[str]], align: list[str] | None = None):
        self.headers, self.rows = headers, rows
        self.align = align or ["l"] + ["r"] * (len(headers) - 1)

    def md(self) -> str:
        sep = ["---:" if a == "r" else ":---" for a in self.align]
        lines = ["| " + " | ".join(self.headers) + " |", "|" + "|".join(sep) + "|"]
        lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in self.rows]
        return "\n".join(lines)

    def html(self) -> str:
        th = "".join(f'<th class="{a}">{_html.escape(h)}</th>' for h, a in zip(self.headers, self.align))
        body = "".join("<tr>" + "".join(f'<td class="{a}">{_html.escape(str(c))}</td>' for c, a in zip(r, self.align))
                       + "</tr>" for r in self.rows)
        return f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


# ------------------------------------------------------------------------------------ içerik
def _scored(summary):
    return [m for m in summary["models"] if "metrics" in m]


def build_sections(summary: dict, run: dict, chart_files: dict[str, str]) -> list[dict]:
    """Bölümleri {title, paras:[...], table:Table|None, chart:str|None} olarak üretir (md ve html ortak)."""
    sc = sorted(_scored(summary), key=lambda m: -m["metrics"]["accuracy"])
    ds = summary["dataset"]
    labels = summary["labels"]
    secs: list[dict] = []

    # --- veri ve koşul
    counts = ", ".join(f"{k}: {v}" for k, v in ds["class_counts"].items())
    secs.append({"title": "Koşul özeti", "paras": [
        f"**Veri:** `{ds['name']}` · değerlendirme bölümü: `{ds['split']}` ({ds['n_eval']} örnek) · "
        f"eğitim bölümü: {ds['n_train']} örnek · parmak izi `{ds['fingerprint']}`",
        f"**Sınıf dağılımı (değerlendirme):** {counts}",
        f"**Ayarlar:** tohum {summary['settings']['seed']} · etiket başına few-shot "
        f"{summary['settings']['shots_per_label']} · istem sürümü `{summary['settings']['prompt_version']}` · "
        f"seçenek sırası testi: {'evet' if summary['settings']['robustness'] else 'hayır'}",
        f"**Ortam:** Python {summary['environment']['python']} · {summary['environment']['platform']}",
    ] + ([f"**Not:** {summary['note']}"] if summary.get("note") else [])})

    if not sc:
        secs.append({"title": "Sonuç yok", "paras": ["Hiçbir model başarıyla puanlanamadı. Ayrıntı için atlanan modeller bölümüne bakın."]})
        secs.append(_skipped_section(summary))
        return secs

    best = sc[0]
    ref = summary.get("reference")
    ref_name = next((m["display"] for m in sc if m["key"] == ref), ref)

    # --- bulgular
    ties = [m for m in sc if m["key"] != ref and (m.get("vs_reference") or {}).get("lo", 1) <= 0 <= (m.get("vs_reference") or {}).get("hi", -1)]
    paras = [f"**En yüksek doğruluk:** {best['display']} — {pct(best['metrics']['accuracy'])} "
             f"(%95 GA {pct(best['metrics']['accuracy_ci'][0])}–{pct(best['metrics']['accuracy_ci'][1])}), "
             f"macro-F1 {pct(best['metrics']['macro_f1'])}."]
    if ties:
        paras.append(f"**İstatistiksel olarak ayırt edilemeyenler ({ref_name} ile eşlenmiş bootstrap farkının %95 aralığı 0'ı içeriyor):** "
                     + ", ".join(m["display"] for m in ties) + ". Bu modeller arasında bu örneklemle kesin sıralama yapılamaz.")
    else:
        paras.append(f"{ref_name} ile eşlenmiş farkı sıfırı içeren başka model yok.")
    lat = [m for m in sc if m["metrics"]["lat_p50"] is not None]
    if lat:
        f = min(lat, key=lambda m: m["metrics"]["lat_p50"])
        paras.append(f"**En düşük p50 gecikme:** {f['display']} — {ms(f['metrics']['lat_p50'])} ms "
                     f"(doğruluk {pct(f['metrics']['accuracy'])}).")
    cost = [m for m in sc if m["metrics"]["cost_per_1000"] is not None]
    if cost:
        c = min(cost, key=lambda m: m["metrics"]["cost_per_1000"])
        paras.append(f"**En düşük ölçülen API maliyeti:** {c['display']} — ${usd(c['metrics']['cost_per_1000'], c['metrics']['tokens_estimated'])} / 1.000 karar "
                     f"(doğruluk {pct(c['metrics']['accuracy'])}).")
    pareto = [m["display"] for m in sc if m["key"] in summary.get("pareto_latency", [])]
    if pareto:
        paras.append("**Doğruluk–gecikme Pareto sınırı:** " + ", ".join(pareto) + ".")
    paraC = [m["display"] for m in sc if m["key"] in summary.get("pareto_cost", [])]
    if paraC:
        paras.append("**Doğruluk–maliyet Pareto sınırı (yalnız fiyatı bilinen API modelleri):** " + ", ".join(paraC) + ".")
    secs.append({"title": "Öne çıkan bulgular", "paras": paras, "chart": chart_files.get("accuracy")})

    # --- ana tablo
    rows = []
    for i, m in enumerate(sc, 1):
        mt = m["metrics"]
        rows.append([str(i), m["display"], GROUP_TR.get(m["group"], m["group"]),
                     f"{pct(mt['accuracy'])} [{pct(mt['accuracy_ci'][0], 0)}–{pct(mt['accuracy_ci'][1], 0)}]",
                     pct(mt["macro_f1"]), pct(mt["error_rate"]),
                     num(mt["ece"] * 100) + " pp" if mt["ece"] is not None else "—",
                     ms(mt["lat_p50"]), ms(mt["lat_p95"]),
                     usd(mt["cost_per_1000"], mt["tokens_estimated"]) if mt["cost_per_1000"] is not None
                     else ("yerel" if m.get("local") else "—")])
    secs.append({"title": "Ana sonuç tablosu", "paras": [
        "Doğruluk sıkıdır: geçersiz veya başarısız yanıtlar yanlış sayılır, oranı ayrıca verilir. Köşeli parantez: %95 bootstrap güven aralığı. "
        "ECE: kalibrasyon hatası (yüzde puan; düşük iyi). Gecikme: istemci tarafı uçtan uca ms. Maliyet: ölçülen token × tarife, 1.000 karar için USD; "
        "`~` token sayısının API'den gelmeyip tahmin edildiğini, `yerel` API ücreti olmadığını gösterir (altyapı maliyeti hariç)."],
        "table": Table(["#", "Model", "Grup", "Doğruluk [%95 GA]", "Macro-F1", "Geçersiz", "ECE", "p50 ms", "p95 ms", "$/1.000 karar"], rows,
                       ["r", "l", "l", "r", "r", "r", "r", "r", "r", "r"])})

    # --- ek ölçümler
    rows = []
    for m in sc:
        mt = m["metrics"]
        kinds = {"native": "yerel olasılık", "verbalized": "sözel güven", "derived": "türetilmiş"}
        rows.append([m["display"],
                     num(mt["brier"], 3) if mt["brier"] is not None else "—",
                     ", ".join(kinds.get(k, k) for k in mt["conf_kinds"]) or "—",
                     pct(mt["acc_at_cov80"]), pct(mt["cov_at_acc95"], 0),
                     ms(mt["lat_p99"]), ms(m.get("cold_ms")), num(m.get("setup_s"), 1) + " sn" if m.get("setup_s") is not None else "—",
                     str(m.get("concurrency", "—")),
                     f"{num(mt['tok_in_mean'], 0)} / {num(mt['tok_out_mean'], 0)}" if mt["tok_in_mean"] is not None else "—"])
    secs.append({"title": "Kalibrasyon, hız ve kullanım ayrıntıları", "paras": [
        "Brier: çok sınıflı olasılık hatası (düşük iyi). **Güven türü** karşılaştırmayı sınırlar: üretken LLM'lerde 'sözel güven' modelin yazdığı sayıdır, yerel olasılık değildir; "
        "kalibrasyon sayıları bu yüzden doğrudan kıyaslanmamalıdır. Seçici doğruluk: modelin en emin olduğu %80 örnekteki doğruluk ve doğruluğu ≥%95 tutan en büyük kapsama. "
        "İlk istek (soğuk) gecikmesi ve kurulum/yükleme/eğitim süresi ayrı verilir; p50/p95 ısınmadan sonraki isteklerdir."],
        "table": Table(["Model", "Brier", "Güven türü", "Doğruluk@%80 kapsama", "Kapsama@%95 doğruluk", "p99 ms", "Soğuk ms",
                        "Kurulum", "Eşzamanlılık", "Ort. token (g/ç)"], rows,
                       ["l", "r", "l", "r", "r", "r", "r", "r", "r", "r"])})

    # --- hız/maliyet grafiği
    if chart_files.get("latency"):
        secs.append({"title": "Doğruluk ve gecikme", "paras": [
            "Gecikme ölçümleri farklı donanım, bağlantı ve eşzamanlılık koşullarında alınmıştır (API: bu makineden uçtan uca; yerel: bu makinenin CPU/GPU'su). "
            "Kontrollü bir donanım yarışı değildir."], "chart": chart_files["latency"]})
    if chart_files.get("cost"):
        secs.append({"title": "Doğruluk ve maliyet", "paras": [
            "Yalnızca fiyatı bilinen API modelleri gösterilir; yerel modellerin altyapı maliyeti bu grafikte yoktur."], "chart": chart_files["cost"]})

    # --- eşlenmiş karşılaştırma
    rows = []
    for m in sc:
        v = m.get("vs_reference")
        if v:
            verdict = ("daha iyi" if v["lo"] > 0 else "daha kötü" if v["hi"] < 0 else "ayırt edilemez")
            rows.append([m["display"], pp(v["diff"]), f"{pp(v['lo'])} … {pp(v['hi'])}", verdict])
    if rows:
        secs.append({"title": f"Eşlenmiş karşılaştırma (referans: {ref_name})", "paras": [
            "Aynı örnekler üzerinde doğruluk farkı (model − referans) ve %95 eşlenmiş bootstrap aralığı. Aralık 0'ı içeriyorsa bu örneklemle fark kanıtlanamaz."],
            "table": Table(["Model", "Fark (puan)", "%95 aralık", "Sonuç"], rows, ["l", "r", "r", "l"])})

    # --- sınıf bazında F1
    rows = []
    for m in sc:
        pc = m["metrics"]["per_class"]
        rows.append([m["display"]] + [pct(pc[l]["f1"], 0) if l in pc else "—" for l in labels])
    secs.append({"title": "Sınıf bazında F1", "paras": [
        "Satır başına model, sütun başına sınıf. Destek sayısı küçük olduğundan tek sınıftaki farklar gürültülü olabilir."],
        "table": Table(["Model"] + labels, rows, ["l"] + ["r"] * len(labels))})
    if chart_files.get("confusion"):
        secs.append({"title": "Karışıklık matrisleri (en doğru 3 model)", "paras": [
            "Satırlar gerçek, sütunlar tahmin edilen sınıf; renk satır normalizasyonudur, hücrelerdeki sayılar örnek sayısıdır."],
            "chart": chart_files["confusion"]})

    # --- dilimler
    all_tags = sorted({t for m in sc for t in m.get("slices", {})})
    if all_tags:
        rows = []
        for m in sc:
            sl = m.get("slices", {})
            rows.append([m["display"]] + [f"{pct(sl[t]['accuracy'], 0)} (n={sl[t]['n']})" if t in sl else "—" for t in all_tags])
        secs.append({"title": "Zor örnek dilimleri", "paras": [
            "Doğruluk, örnek etiketine göre. En az 5 örnek içeren dilimler gösterilir; n küçük olduğundan yalnızca yön göstericidir."],
            "table": Table(["Model"] + [SLICE_TR.get(t, t) for t in all_tags], rows, ["l"] + ["r"] * len(all_tags))})

    # --- dayanıklılık
    rob = [m for m in sc if (m.get("robustness") or {}).get("acc_shuffled") is not None]
    if rob:
        rows = [[m["display"], pct(m["robustness"]["acc_canonical"]), pct(m["robustness"]["acc_shuffled"]),
                 pp(m["robustness"]["delta"]), pct(m["robustness"]["consistency"])] for m in rob]
        secs.append({"title": "Seçenek sırası dayanıklılığı", "paras": [
            "Etiket listesi her örnek için rastgele sıralanıp model yeniden çalıştırıldı. Tutarlılık: iki sırada aynı etiketi seçme oranı. Sıradan etkilenmeyen "
            "modeller (TF-IDF, gömme, NLI) bu testten muaftır."],
            "table": Table(["Model", "Doğruluk (varsayılan)", "Doğruluk (karıştırılmış)", "Fark", "Tutarlılık"], rows,
                           ["l", "r", "r", "r", "r"]), "chart": chart_files.get("robustness")})

    secs.append(_skipped_section(summary))
    secs.append(_method_section(summary, run))
    return [s for s in secs if s]


def _skipped_section(summary: dict):
    bad = [m for m in summary["models"] if m.get("status") not in ("ok",) or (m.get("first_error"))]
    if not bad:
        return None
    rows = []
    for m in bad:
        st = {"failed": "atlandı", "interrupted": "yarıda kesildi", "ok": "kısmi hata"}.get(m.get("status"), m.get("status", "?"))
        rows.append([m["display"], st, (m.get("reason") or m.get("first_error") or "")[:260]])
    return {"title": "Atlanan veya hatalı modeller", "paras": [
        "Bu modeller ya hiç puanlanamadı ya da bazı istekleri başarısız oldu. Başarısız istekler doğrulukta yanlış sayılır."],
        "table": Table(["Model", "Durum", "Neden / ilk hata"], rows, ["l", "l", "l"])}


def _method_section(summary: dict, run: dict) -> dict:
    prices = []
    for m in summary["models"]:
        if m.get("price_in") is not None:
            prices.append(f"{m['display']}: {num(m['price_in'], 3)} / {num(m.get('price_out') or 0, 3)}"
                          + (f" ({m['price_note']})" if m.get("price_note") else ""))
    return {"title": "Yöntem ve sınırlar", "paras": [
        "- **Veri:** Yerleşik `destek_tr` seti, bu çalışma için hazırlanmış *gümüş* etiketli bir örnek kümesidir; insan etiketli altın küme DEĞİLDİR. "
        "Bir satın alma kararı için kendi, insan etiketli Türkçe verinizle (`--dataset`) yeniden çalıştırın.",
        "- **Bölme:** Değerlendirme yalnızca `test` (veya istenen) bölümünde yapılır; eğitimli modeller yalnızca `train` bölümünü görür; few-shot örnekleri de `train`'den alınır ve değerlendirmeden çıkarılır.",
        "- **Aynı görev, doğal arayüz:** Karar API'leri kendi `choice` şemasıyla, üretken LLM'ler aynı Türkçe talimat + JSON şemasıyla, encoder modeller etiketlerin kısa Türkçe adlarıyla çağrılır. "
        "Bu, her modelin doğal arayüzünü adil kullanır ama modeller arasındaki farkın bir kısmı istem/arayüz farkından gelebilir.",
        "- **Güven/olasılık:** Karar API'leri ve yerel modeller yerel olasılık verir; üretken LLM'ler sözel güven yazar. Brier/ECE bu yüzden aynı ölçekte değildir.",
        "- **Gecikme:** İstemci tarafı uçtan uca süre. Konum, ağ, eşzamanlılık ve sunucu yükü sonucu etkiler; ısınma isteği hariç tutulur. Yerel gecikmeler bu makinenin donanımına aittir.",
        "- **Maliyet:** Yanıttan okunan token kullanımı × tarife. API token vermiyorsa tahmin edilir ve `~` ile işaretlenir. Abonelik, minimum bakiye, yeniden deneme ve altyapı hariçtir.",
        "- **İstatistik:** Güven aralıkları örnekler üzerinde bootstrap (2000) ile hesaplanır; küçük farklardan kazanan çıkarılamaz.",
        "- **Tarifeler (USD / 1M giriş / çıkış):** " + ("; ".join(prices) if prices else "—"),
        f"- **Sürüm:** clfbench {run.get('clfbench_version')} · git {run.get('environment', {}).get('git_commit') or '—'} · "
        f"paketler: " + ", ".join(f"{k} {v}" for k, v in run.get("environment", {}).get("packages", {}).items()),
    ]}


# ------------------------------------------------------------------------------------ çıktılar
def _md_paras(paras):
    return "\n\n".join(paras)


def render_markdown(summary, secs, run) -> str:
    out = [f"# Classifier LLM karşılaştırma raporu", "",
           f"Koşu: `{summary['run_id']}` · oluşturma: {run['started']}", ""]
    for s in secs:
        out.append(f"## {s['title']}\n")
        if s.get("paras"):
            out.append(_md_paras(s["paras"]) + "\n")
        if s.get("chart"):
            out.append(f"![{s['title']}](charts/{s['chart']})\n")
        if s.get("table"):
            out.append(s["table"].md() + "\n")
    return "\n".join(out)


_CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--accent:#2a78d6;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--accent:#3987e5;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px}
.meta{color:var(--ink2);margin:0 0 18px}p{margin:8px 0}code{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:.92em}
.tablewrap{overflow-x:auto;margin:12px 0;border:1px solid var(--grid);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:7px 10px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-weight:600;color:var(--ink2);text-align:left;position:sticky;top:0;background:var(--surface)}td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}figure{margin:12px 0;padding:10px;background:#fcfcfb;border:1px solid var(--grid);border-radius:8px}
figure img{max-width:100%;height:auto;display:block}ul{padding-left:20px}
"""


def _inline(s: str) -> str:
    """Minimal Markdown -> HTML: **kalın**, `kod`."""
    s = _html.escape(s)
    import re

    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def render_html(summary, secs, run, chart_dir: Path) -> str:
    parts = [f"<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Classifier LLM karşılaştırma raporu</title><style>{_CSS}</style></head><body><main>",
             f"<h1>Classifier LLM karşılaştırma raporu</h1><p class='meta'>Koşu <code>{_html.escape(summary['run_id'])}</code> · {run['started']}</p>"]
    for s in secs:
        parts.append(f"<h2>{_html.escape(s['title'])}</h2>")
        bullets = [p for p in s.get("paras", []) if p.startswith("- ")]
        plain = [p for p in s.get("paras", []) if not p.startswith("- ")]
        for p in plain:
            parts.append(f"<p>{_inline(p)}</p>")
        if bullets:
            parts.append("<ul>" + "".join(f"<li>{_inline(b[2:])}</li>" for b in bullets) + "</ul>")
        if s.get("chart"):
            data = base64.b64encode((chart_dir / s["chart"]).read_bytes()).decode()
            parts.append(f"<figure><img alt='{_html.escape(s['title'])}' src='data:image/png;base64,{data}'></figure>")
        if s.get("table"):
            parts.append(s["table"].html())
    parts.append("</main></body></html>")
    return "".join(parts)


def generate_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    sc = _scored(summary)
    files: dict[str, str] = {}
    if sc:
        charts.accuracy_bars(sc, chart_dir / "accuracy.png")
        files["accuracy"] = "accuracy.png"
        if charts.scatter_tradeoff(sc, chart_dir / "latency.png", x_key="lat_p50",
                                   x_label="p50 gecikme (ms, log ölçek) — sola doğru daha hızlı",
                                   pareto=set(summary.get("pareto_latency", []))):
            files["latency"] = "latency.png"
        priced = [m for m in sc if m["metrics"]["cost_per_1000"] is not None]
        if charts.scatter_tradeoff(priced, chart_dir / "cost.png", x_key="cost_per_1000",
                                   x_label="1.000 karar başına maliyet (USD, log ölçek) — sola doğru daha ucuz",
                                   pareto=set(summary.get("pareto_cost", []))):
            files["cost"] = "cost.png"
        if charts.confusion_panels(sc, summary["labels"], chart_dir / "confusion.png"):
            files["confusion"] = "confusion.png"
        if charts.robustness_dumbbell(sc, chart_dir / "robustness.png"):
            files["robustness"] = "robustness.png"
    secs = build_sections(summary, run, files)
    (out_dir / "report.md").write_text(render_markdown(summary, secs, run), encoding="utf-8")
    (out_dir / "report.html").write_text(render_html(summary, secs, run, chart_dir), encoding="utf-8")
    return out_dir / "report.html"
