"""Komut satırı: clfbench doctor | keys | models | build-data | ping | estimate | run | report"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ModelSpec, load_env, load_registry, resolve_models
from .core import Prediction
from .data import build_seed_dataset, load_dataset, select
from .keys import BY_ENV, prefix_warning, render_guides
from .paths import RESULTS_DIR, ROOT
from .prompts import estimate_tokens, system_prompt


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _split_csv(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _price(s: ModelSpec) -> str:
    if s.price_in is None:
        return "—"
    return f"{s.price_in:g}/{(s.price_out or 0):g}"


# ------------------------------------------------------------------------------------ komutlar
def cmd_models(args) -> int:
    reg = load_registry(args.config)
    print(f"{'anahtar':26s} {'grup':12s} {'sağlayıcı':17s} {'$/1M g/ç':>12s}  durum")
    for s in reg.values():
        if not args.all and not s.enabled:
            continue
        ok, why = s.availability()
        flag = "" if s.enabled else " (varsayılan kapalı)"
        print(f"{s.key:26s} {s.group:12s} {s.provider:17s} {_price(s):>12s}  {'hazır' if ok else why}{flag}")
    return 0


def cmd_keys(args) -> int:
    reg = load_registry(args.config)
    needed = {s.api_key_env for s in reg.values() if s.api_key_env}
    if args.missing:
        import os

        needed = {k for k in needed if not os.environ.get(k, "").strip()}
    print(render_guides(needed) or "Tüm anahtarlar tanımlı.")
    print("Anahtarları .env dosyasına yazın (örnek: .env.example). Anahtarı kimseyle paylaşmayın, commit'lemeyin.")
    return 0


def cmd_doctor(args) -> int:
    import importlib.util
    import os

    reg = load_registry(args.config)
    print(f"Python {sys.version.split()[0]}  ·  .env dosyası: "
          f"{'var' if (Path(__file__).resolve().parents[2] / '.env').exists() else 'YOK (.env.example dosyasını kopyalayın)'}\n")
    print("Paketler:")
    for mod, why in [("requests", "zorunlu"), ("numpy", "zorunlu"), ("matplotlib", "grafikler"), ("anthropic", "Claude"),
                     ("sklearn", "TF-IDF/LR, gömme"), ("sentence_transformers", "gömme modelleri"),
                     ("transformers", "NLI/GLiClass"), ("torch", "yerel modeller"), ("gliclass", "GLiClass"),
                     ("laya", "Laya"), ("setfit", "SetFit"), ("datasets", "SetFit/HF veri")]:
        print(f"  {'✓' if importlib.util.find_spec(mod) else '·'} {mod:22s} {why}")
    print("\nAPI anahtarları:")
    for env in sorted({s.api_key_env for s in reg.values() if s.api_key_env}):
        val = os.environ.get(env, "").strip().strip("\"'")
        print(f"  {'✓ tanımlı' if val else '· yok     '}  {env:22s} "
              f"({', '.join(s.key for s in reg.values() if s.api_key_env == env)})")
        warn = prefix_warning(env, val)
        if warn:
            print(f"      ⚠ {warn}")
    run, skipped = resolve_models(reg, None)
    print(f"\nÇalıştırılabilir modeller ({len(run)}):")
    for s in run:
        print(f"  ✓ {s.key:26s} {s.display}")
    if skipped:
        print(f"\nAtlanacak modeller ({len(skipped)}):")
        for s, why in skipped:
            print(f"  · {s.key:26s} {why}")
    print("\nAnahtar almak için: python -m clfbench keys --missing")
    return 0


def cmd_build_data(args) -> int:
    p = build_seed_dataset()
    task, ex = load_dataset("destek_tr")
    from collections import Counter

    c = Counter(e.split for e in ex)
    print(f"{p} yazıldı: {len(ex)} örnek · train {c['train']} / dev {c['dev']} / test {c['test']}")
    return 0


def cmd_ping(args) -> int:
    reg = load_registry(args.config)
    if args.model not in reg:
        print(f"Bilinmeyen model: {args.model}. 'clfbench models --all' ile bakın.")
        return 2
    spec = reg[args.model]
    ok, why = spec.availability()
    if not ok:
        print(f"{spec.key} kullanılamıyor: {why}")
        if spec.api_key_env in BY_ENV:
            print("\n" + render_guides({spec.api_key_env}))
        return 2
    from .providers import make_provider

    task, ex = load_dataset(args.dataset)
    train = [e for e in ex if e.split == "train"]
    p = make_provider(spec)
    text = args.text or "Kartımdan aynı sipariş için iki kez çekim yapılmış, fazla tutarın iadesini istiyorum."
    print(f"→ {spec.display} ({spec.model or spec.provider})\n  metin: {text}\n")
    try:
        p.prepare(task, train, [])
        import time

        t0 = time.perf_counter()
        pred: Prediction = p.predict(text, task, None)
        wall = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"✗ HATA: {type(e).__name__}: {e}")
        return 1
    top = sorted((pred.probs or {}).items(), key=lambda kv: -kv[1])[:3]
    print(f"✓ etiket: {pred.label}  ·  güven: {pred.confidence if pred.confidence is None else round(pred.confidence, 4)}"
          f" ({pred.conf_kind})")
    if top:
        print("  olasılıklar: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
    print(f"  gecikme: {pred.latency_ms or wall:.0f} ms  ·  token (g/ç): {pred.input_tokens}/{pred.output_tokens}")
    return 0


def estimate_cost(spec: ModelSpec, task, n_eval: int, texts: list[str], shots_k: int, robustness: bool,
                  warmup: int) -> float | None:
    """Çalıştırmadan önce kaba maliyet tahmini (USD). Fiyatı bilinmeyen/yerel modeller için None."""
    if spec.price_in is None or spec.is_local:
        return None
    avg_text = sum(estimate_tokens(t) for t in texts) / max(len(texts), 1)
    base = estimate_tokens(system_prompt(task)) if spec.provider in ("claude", "gemini", "chat_json") else \
        estimate_tokens(task.instruction + " ".join(f"{k}: {v}" for k, v in task.labels.items()))
    shots_tok = shots_k * len(task.labels) * 28 if spec.provider in ("claude", "gemini", "chat_json") else 0
    t_in = base + shots_tok + avg_text
    # Çıktı: karar API'leri ~0-3 token; üretken modellerde düşünme tokenları olabilir → temkinli 300 varsay
    t_out = 3 if spec.provider in ("systemone", "openai_decisions") else 300
    calls = n_eval * (2 if robustness and spec.provider not in ("tfidf_lr",) else 1) + warmup
    return calls * (t_in * spec.price_in + t_out * (spec.price_out or 0)) / 1e6


def cmd_estimate(args) -> int:
    reg = load_registry(args.config)
    task, ex = load_dataset(args.dataset)
    ev = select(ex, args.split)
    run, skipped = resolve_models(reg, _split_csv(args.models), _split_csv(args.groups))
    total = 0.0
    print(f"{len(ev)} değerlendirme örneği · seçenek sırası testi: {'evet' if args.robustness else 'hayır'}\n")
    for s in run:
        c = estimate_cost(s, task, len(ev), [e.text for e in ev], args.few_shot, args.robustness, args.warmup)
        total += c or 0
        print(f"  {s.key:26s} " + (f"~${c:.3f}" if c is not None else "ücretsiz/yerel veya fiyat yok"))
    print(f"\nTOPLAM tahmini API maliyeti: ~${total:.3f}  (üretken modellerde düşünme tokenları nedeniyle gerçek değer sapabilir)")
    for s, why in skipped:
        print(f"  atlanır: {s.key} — {why}")
    return 0


def cmd_run(args) -> int:
    from .bench import run_benchmark

    reg = load_registry(args.config)
    task, ex = load_dataset(args.dataset)
    ev = select(ex, args.split)
    run, skipped = resolve_models(reg, _split_csv(args.models), _split_csv(args.groups))
    if skipped:
        print("Atlanan modeller (anahtar/paket eksik):")
        for s, why in skipped:
            print(f"  · {s.key:26s} {why}")
    if not run:
        print("\nÇalıştırılacak model yok. 'python -m clfbench doctor' ve 'python -m clfbench keys --missing' komutlarına bakın.")
        return 2
    n_eval = len(ev) if not args.limit else min(len(ev), args.limit)
    est = {s.key: estimate_cost(s, task, n_eval, [e.text for e in ev], args.few_shot, args.robustness, args.warmup)
           for s in run}
    total = sum(v or 0 for v in est.values())
    print(f"\n{len(run)} model · {n_eval} örnek ({args.split}) · tahmini API maliyeti ~${total:.3f}")
    if total > args.max_cost and not args.yes:
        print(f"Tahmini maliyet --max-cost ({args.max_cost}) sınırını aşıyor. Onaylıyorsanız --yes ekleyin "
              f"veya --models / --limit ile daralın.")
        return 3
    out = run_benchmark(run, task, ex, split=args.split, seed=args.seed, shots_k=args.few_shot,
                        robustness=args.robustness, warmup=args.warmup, concurrency=args.concurrency,
                        use_cache=not args.no_cache, run_name=args.name or "", note=args.note or "",
                        dataset_label=Path(args.dataset).stem if args.dataset != "destek_tr" else "destek_tr",
                        reference=args.reference, limit=args.limit, argv=sys.argv[1:])
    print(f"\n✓ Bitti. Rapor: {out / 'report.html'}\n  Markdown: {out / 'report.md'}\n  Özet JSON: {out / 'summary.json'}")
    return 0


def cmd_report(args) -> int:
    from .bench import finalize
    from .report import generate_report

    d = Path(args.dir)
    if not (d / "run.json").exists():
        cand = RESULTS_DIR / args.dir
        d = cand if (cand / "run.json").exists() else d
    if not (d / "run.json").exists():
        print(f"Koşu klasörü bulunamadı: {args.dir}")
        return 2
    finalize(d, reference=args.reference)
    print(f"✓ Rapor yeniden üretildi: {generate_report(d)}")
    return 0


# ------------------------------------------------------------------------------------ argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clfbench", description="Classifier LLM karşılaştırma altyapısı")
    p.add_argument("--config", help="models.yaml yolu (varsayılan: configs/models.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="Ortam, paket ve API anahtarı durumunu göster"); s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("keys", help="API anahtarlarını nereden alacağını anlat")
    s.add_argument("--missing", action="store_true", help="yalnız tanımsız olanlar"); s.set_defaults(fn=cmd_keys)
    s = sub.add_parser("models", help="Kayıtlı modelleri listele")
    s.add_argument("--all", action="store_true", help="varsayılan kapalıları da göster"); s.set_defaults(fn=cmd_models)
    s = sub.add_parser("build-data", help="data/seed/*.txt -> data/destek_tr.jsonl"); s.set_defaults(fn=cmd_build_data)

    s = sub.add_parser("ping", help="Tek modele tek istek at (anahtar/uç nokta testi)")
    s.add_argument("model"); s.add_argument("--text"); s.add_argument("--dataset", default="destek_tr")
    s.set_defaults(fn=cmd_ping)

    def common(sp):
        sp.add_argument("--models", help="virgülle model anahtarları (varsayılan: hazır olan tüm modeller)")
        sp.add_argument("--groups", help="karar-api,uretken-llm,encoder,egitimli")
        sp.add_argument("--dataset", default="destek_tr", help="destek_tr | dosya.jsonl | dosya.csv | hf:...")
        sp.add_argument("--split", default="test", choices=["test", "dev", "dev+test"])
        sp.add_argument("--few-shot", type=int, default=0, help="üretken LLM'lerde etiket başına örnek sayısı")
        sp.add_argument("--robustness", action="store_true", help="seçenek sırasını karıştırıp tekrar çalıştır")
        sp.add_argument("--warmup", type=int, default=1, help="API modelleri için ısınma isteği sayısı")

    s = sub.add_parser("estimate", help="Çalıştırmadan önce kaba API maliyeti tahmini"); common(s)
    s.set_defaults(fn=cmd_estimate)

    s = sub.add_parser("run", help="Karşılaştırmayı çalıştır ve rapor üret"); common(s)
    s.add_argument("--limit", type=int, help="hızlı deneme: yaklaşık bu kadar örnek (sınıflara dengeli)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--concurrency", type=int, help="API modelleri için eşzamanlı istek (varsayılan: modele göre)")
    s.add_argument("--no-cache", action="store_true", help="önbelleği kullanma, hepsini yeniden çağır")
    s.add_argument("--name", help="koşu adı (klasör adına eklenir)")
    s.add_argument("--note", help="rapora yazılacak not (örn. istemci bölgesi, donanım)")
    s.add_argument("--reference", help="eşlenmiş karşılaştırma referans modeli (varsayılan: en doğru)")
    s.add_argument("--max-cost", type=float, default=2.0, help="tahmini maliyet bu USD değerini aşarsa onay iste")
    s.add_argument("--yes", action="store_true", help="maliyet onayını atla")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="Var olan koşudan raporu yeniden üret")
    s.add_argument("dir"); s.add_argument("--reference"); s.set_defaults(fn=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8()
    load_env()
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Hata: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        return 130
