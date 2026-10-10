"""Grafikler (matplotlib, PNG). Palet: doğrulanmış referans kategorik palet, açık yüzey.

Kurallar: ince işaretler, hairline ızgara, doğrudan etiket, renk=grup (en çok 3 renk; yerel alt-grupları işaret şekliyle),
tek eksen (çift eksen yok), değerler metin renginde (seri renginde değil).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# kategorik slotlar 1-3 (tüm-çiftler kuralıyla doğrulanmış ilk üç slot)
C_API, C_LLM, C_LOCAL = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

GROUP_LABEL = {"karar-api": "Karar API'leri", "uretken-llm": "Üretken LLM (API)", "yerel": "Yerel modeller"}
GROUP_COLOR = {"karar-api": C_API, "uretken-llm": C_LLM, "yerel": C_LOCAL}

def _pick_font() -> str:
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Segoe UI", "Helvetica Neue", "Arial") if f in have), "DejaVu Sans")


plt.rcParams.update({
    "font.family": _pick_font(),
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "font.size": 10,
})


def plot_group(m: dict) -> str:
    return {"karar-api": "karar-api", "uretken-llm": "uretken-llm"}.get(m["group"], "yerel")


def _marker(m: dict) -> str:
    return "D" if m["group"] == "egitimli" else "o"


def _tr_pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}".replace(".", ",")


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _legend_groups(ax, models, loc="best", extra_local_shapes=False) -> None:
    present = []
    for g in ("karar-api", "uretken-llm", "yerel"):
        if any(plot_group(m) == g for m in models):
            present.append(g)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLOR[g], markeredgecolor=SURFACE,
                      markersize=8, label=GROUP_LABEL[g]) for g in present]
    if extra_local_shapes and any(m["group"] == "egitimli" for m in models):
        handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor=MUTED, markeredgecolor=SURFACE,
                              markersize=7, label="eğitimli (yerel), elmas işaret"))
    leg = ax.legend(handles=handles, loc=loc, frameon=False, fontsize=9, labelcolor=INK2)
    for t in leg.get_texts():
        t.set_color(INK2)


def _place_labels(fig, ax, xs, ys, labels, fontsize=8.5) -> None:
    """Basit etiket çarpışma önleyici: aday ofsetleri sırayla dener."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = []
    cands = [(7, 5, "left", "bottom"), (7, -5, "left", "top"), (-7, 5, "right", "bottom"),
             (-7, -5, "right", "top"), (0, 9, "center", "bottom"), (0, -9, "center", "top"),
             (14, 0, "left", "center"), (-14, 0, "right", "center")]
    order = np.argsort(xs)
    for idx in order:
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(labels[idx], (xs[idx], ys[idx]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=fontsize, color=INK2)
            bb = t.get_window_extent(rend).expanded(1.03, 1.15)
            axbb = ax.get_window_extent(rend)
            inside = axbb.x0 - 4 <= bb.x0 and bb.x1 <= axbb.x1 + 4 and axbb.y0 - 4 <= bb.y0 and bb.y1 <= axbb.y1 + 4
            clash = any(bb.overlaps(p) for p in placed)
            if inside and not clash:
                best = bb
                break
            t.remove()
        if best is None:  # yer bulunamadı: ilk adayı çarpışmayı göze alarak koy
            dx, dy, ha, va = cands[0]
            t = ax.annotate(labels[idx], (xs[idx], ys[idx]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=fontsize, color=INK2)
            best = t.get_window_extent(rend)
        placed.append(best)


def _pareto_line(ax, pts, color=MUTED) -> None:
    """pts: (x küçük iyi, y büyük iyi) Pareto noktaları; basamaklı sınır çizgisi."""
    if len(pts) < 2:
        return
    pts = sorted(pts)
    xs, ys = [], []
    for i, (x, y) in enumerate(pts):
        xs += [x, x] if i else [x]
        ys += [pts[i - 1][1], y] if i else [y]
    ax.plot(xs, ys, color=color, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)


def accuracy_bars(models: list[dict], path: Path) -> None:
    ms = sorted(models, key=lambda m: m["metrics"]["accuracy"])
    h = max(2.6, 0.36 * len(ms) + 1.2)
    fig, ax = plt.subplots(figsize=(9.2, h))
    _style(ax)
    ax.grid(axis="y", visible=False)
    for i, m in enumerate(ms):
        mt = m["metrics"]
        a, (lo, hi) = mt["accuracy"], mt["accuracy_ci"]
        ax.barh(i, a * 100, height=0.62, color=GROUP_COLOR[plot_group(m)], zorder=2)
        ax.plot([lo * 100, hi * 100], [i, i], color=INK2, linewidth=1.1, zorder=3, solid_capstyle="butt")
        for xe in (lo, hi):
            ax.plot([xe * 100] * 2, [i - 0.14, i + 0.14], color=INK2, linewidth=1.1, zorder=3)
        ax.text(min(hi * 100 + 1.2, 99.5), i, "%" + _tr_pct(a), va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([m["display"] for m in ms], fontsize=9, color=INK2)
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"%{v}" for v in range(0, 101, 20)])
    ax.set_xlabel("Doğruluk (sıkı: geçersiz yanıt = yanlış) — çubuk ucundaki çizgi %95 bootstrap güven aralığı")
    _legend_groups(ax, ms, loc="lower right", extra_local_shapes=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scatter_tradeoff(models: list[dict], path: Path, *, x_key: str, x_label: str, pareto: set[str],
                     title_note: str = "") -> bool:
    pts = [m for m in models if m["metrics"].get(x_key) is not None and m["metrics"][x_key] > 0]
    if len(pts) < 2:
        return False
    xs = np.array([m["metrics"][x_key] for m in pts], dtype=float)
    ys = np.array([m["metrics"]["accuracy"] * 100 for m in pts])
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    _style(ax)
    ax.set_xscale("log")
    for m, x, y in zip(pts, xs, ys):
        ax.scatter([x], [y], s=70, marker=_marker(m), color=GROUP_COLOR[plot_group(m)], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
    _pareto_line(ax, [(x, y) for m, x, y in zip(pts, xs, ys) if m["key"] in pareto])
    ymin = max(0, math.floor((ys.min() - 4) / 5) * 5)
    ax.set_ylim(ymin, min(102, ys.max() + 6))
    ax.set_xlim(xs.min() / 1.8, xs.max() * 2.4)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_xlabel(x_label)
    _legend_groups(ax, pts, loc="upper right", extra_local_shapes=True)
    _place_labels(fig, ax, xs, ys, [m["display"] for m in pts])
    ax.text(0.01, 0.015, "Kesikli çizgi: Pareto sınırı (daha doğru ve daha hızlı/ucuz başka model yok)" + title_note,
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def confusion_panels(models: list[dict], labels: list[str], path: Path, top_k: int = 3) -> bool:
    top = sorted(models, key=lambda m: -m["metrics"]["accuracy"])[:top_k]
    if not top:
        return False
    k = len(labels)
    fig, axes = plt.subplots(1, len(top), figsize=(5.1 * len(top), 4.9), squeeze=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, m in zip(axes[0], top):
        cm = np.array(m["metrics"]["confusion"], dtype=float)  # k x (k+1)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(k):
            for j in range(k + 1):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="#ffffff" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(k + 1))
        ax.set_xticklabels(labels + ["geçersiz"], rotation=60, ha="right", fontsize=8, color=INK2)
        ax.set_yticks(range(k))
        ax.set_yticklabels(labels, fontsize=8, color=INK2)
        ax.set_title(f"{m['display']}\nDoğruluk %{_tr_pct(m['metrics']['accuracy'])}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Tahmin", fontsize=9)
        ax.set_ylabel("Gerçek", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def robustness_dumbbell(models: list[dict], path: Path) -> bool:
    ms = [m for m in models if (m.get("robustness") or {}).get("acc_shuffled") is not None]
    if not ms:
        return False
    ms = sorted(ms, key=lambda m: m["robustness"]["acc_canonical"])
    fig, ax = plt.subplots(figsize=(9.2, max(2.4, 0.42 * len(ms) + 1.3)))
    _style(ax)
    ax.grid(axis="y", visible=False)
    for i, m in enumerate(ms):
        r = m["robustness"]
        a, b = r["acc_canonical"] * 100, r["acc_shuffled"] * 100
        ax.plot([a, b], [i, i], color=AXIS, linewidth=2, zorder=2, solid_capstyle="round")
        ax.scatter([a], [i], s=60, color=C_API, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([b], [i], s=60, color="#9ec5f4", edgecolor=INK2, linewidth=0.8, zorder=3)
        ax.text(max(a, b) + 1.2, i, f"tutarlılık %{_tr_pct(r['consistency'], 0)}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([m["display"] for m in ms], fontsize=9, color=INK2)
    lo = min(min(m["robustness"]["acc_canonical"], m["robustness"]["acc_shuffled"]) for m in ms) * 100
    ax.set_xlim(max(0, lo - 8), 112)
    ax.set_xlabel("Doğruluk (%) — koyu: varsayılan seçenek sırası, açık: rastgele karıştırılmış sıra")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=C_API, markersize=8, label="Varsayılan sıra"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#9ec5f4", markeredgecolor=INK2, markersize=8,
                      label="Karıştırılmış sıra")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True
