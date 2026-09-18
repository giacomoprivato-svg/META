#!/usr/bin/env python3
"""
RQ1 — specificity figure (AIM 1)
================================

Panel A : mean similarity to the psychiatric benchmark (x) against mean
          similarity to SUD (y), one point per disorder, coloured by clinical
          cluster, identity line drawn. Vertical distance from the diagonal IS
          Delta. Filled marker + asterisk = Delta survives FDR under both
          nulls; open marker = not.
Panel B : Delta per disorder with bootstrap 95% CIs, ordered by the cortical
          hierarchy, with the null's 2.5-97.5 percentile band behind each row.

Reading panel A: the null band in panel B is NOT centred on zero, and panel A
shows why. Both means are recomputed from the same rotated map, so the null
absorbs the baseline difference between a coherent target set (six substance
maps sharing one control sample) and a heterogeneous one (eight psychiatric
maps). A point above the diagonal is therefore not automatically evidence of
SUD-specific affinity — it has to clear the band.

"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import RQ1_common as C

# ================= CONFIG — edit, then press Run =================
BENCHMARK = "all"          # "all" (primary) | "outcluster"
FIGSIZE = (16, 5.5)        # ~(7.6, 10.6) if STACKED
STACKED = False            # True = A above B
EQUAL_ASPECT = False       # True keeps the identity line at 45 deg, so
                           # "distance from the diagonal" reads as Delta
MANUAL_OFFSETS = {}        # {"SCZ": (4, -2, "left")} in % of axis span,
                           # to pin one label and let the rest place themselves
FS_TITLE, FS_LABEL, FS_TICK, FS_POINT, FS_LEGEND = 16, 14, 13, 13, 12
# ================================================================

BENCH_LBL = {"all": "all other psychiatric disorders",
             "outcluster": "psychiatric disorders outside own cluster"}

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ1", "specificity_adults")
FIGDIR = os.path.join(repo_dir, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def place_labels(fig, ax, texts, pts_disp, marker_px=13, pad=1.5):
    """
    Greedy label placement in DISPLAY space.

    The previous version used a physics-style repulsion in axis-fraction space,
    which is what left BD/SCZ and PD/MDD touching: it only pushed labels away
    from each other, never away from the MARKERS, and it had no notion of how
    wide a label actually is ("ADHD" is four times the width of "AN").

    Now each label's real pixel bbox is measured once from the renderer, then
    eight compass directions at two radii are scored against every marker and
    every already-placed label. Crowded points are placed first, so they get
    first pick of the free space. Deterministic, and it adapts automatically
    when the data move.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    sizes = []
    for t in texts:
        bb = t.get_window_extent(renderer=r)
        sizes.append((bb.width, bb.height))

    ax_bb = ax.get_window_extent(renderer=r)

    def overlap(a, b):
        dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        return max(dx, 0) * max(dy, 0)

    # markers as square proxies, inflated by pad
    marks = [(x - marker_px - pad, y - marker_px - pad,
              2 * (marker_px + pad), 2 * (marker_px + pad)) for x, y in pts_disp]

    # crowded points first
    order = np.argsort([-sum(1 for q in pts_disp
                             if np.hypot(*(np.array(p_) - q)) < 90) for p_ in pts_disp])

    dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    placed = []
    for i in order:
        w, h = sizes[i]
        px, py = pts_disp[i]
        best, best_score = None, np.inf
        for radius in (marker_px + 4, marker_px + 14, marker_px + 26):
            for dx, dy in dirs:
                n = np.hypot(dx, dy)
                cx = px + dx / n * (radius + w / 2 * abs(dx) / max(n, 1))
                cy = py + dy / n * (radius + h / 2 * abs(dy) / max(n, 1))
                rect = (cx - w / 2 - pad, cy - h / 2 - pad, w + 2 * pad, h + 2 * pad)
                score = sum(overlap(rect, m) for m in marks)
                score += 2.0 * sum(overlap(rect, q) for q in placed)
                # keep it inside the axes
                if (rect[0] < ax_bb.x0 or rect[1] < ax_bb.y0
                        or rect[0] + rect[2] > ax_bb.x1 or rect[1] + rect[3] > ax_bb.y1):
                    score += 1e6
                score += 0.06 * radius            # prefer close labels
                if score < best_score:
                    best_score, best = score, (cx, cy, rect)
        cx, cy, rect = best
        placed.append(rect)
        texts[i].set_position(ax.transData.inverted().transform((cx, cy)))
        texts[i].set_ha("center")
        texts[i].set_va("center")


def main():
    f = os.path.join(OUTDIR, f"SPECIFICITY_delta_adults_{BENCHMARK}.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"{f}\nRun RQ1_AIM1_step1_specificity_delta.py with "
            f"BENCHMARKS including '{BENCHMARK}'.")
    df = pd.read_csv(f, index_col=0)
    C.check_cluster_coverage(list(df.index))

    have = [n for n in ("spin", "brainsmash") if f"pFDR_{n}" in df.columns]
    if not have:
        raise KeyError("no pFDR_* columns in the specificity file.")
    df["sig"] = np.logical_and.reduce([df[f"pFDR_{n}"] < .05 for n in have])
    conv = ("FDR < .05 under both nulls" if len(have) == 2
            else f"FDR < .05 ({have[0]} null only)")
    if len(have) == 1:
        print(f"WARNING: only the {have[0]} null is present. Rerun step 1 with "
              f"NULL_MODE = 'both' before this figure goes in the paper.")

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE) if STACKED \
        else plt.subplots(1, 2, figsize=FIGSIZE)

    # ------------------------- Panel A -------------------------
    ax = axes[0]
    lo = min(df["mean_rho_PSY"].min(), df["mean_rho_SUD"].min()) - .07
    hi = max(df["mean_rho_PSY"].max(), df["mean_rho_SUD"].max()) + .10
    lims = (lo, hi)
    ax.plot(lims, lims, ls="--", c="0.55", lw=1.3, zorder=0)
    ax.axhline(0, c="0.88", lw=.9, zorder=0)
    ax.axvline(0, c="0.88", lw=.9, zorder=0)

    pts = df[["mean_rho_PSY", "mean_rho_SUD"]].to_numpy(float)
    ax.set_xlim(lims); ax.set_ylim(lims)
    texts = []
    for k, (name, row) in enumerate(df.iterrows()):
        col = C.CLUSTER_COLORS[row["cluster"]]
        ax.scatter(row["mean_rho_PSY"], row["mean_rho_SUD"], s=150,
                   facecolor=col if row["sig"] else "white",
                   edgecolor=col, linewidth=2.2, zorder=3)
        texts.append(ax.text(pts[k, 0], pts[k, 1], f"{name}*" if row["sig"] else name,
                             fontsize=FS_POINT, color="0.15", zorder=5,
                             ha="center", va="center",
                             fontweight="bold" if row["sig"] else "normal"))
    for name, (dx, dy, ha) in MANUAL_OFFSETS.items():
        if name in list(df.index):
            k = list(df.index).index(name)
            texts[k].set_position(
                (pts[k, 0] + dx * (lims[1] - lims[0]) / 100,
                 pts[k, 1] + dy * (lims[1] - lims[0]) / 100))
            texts[k].set_ha(ha)
            texts[k] = None
    keep = [i for i, t in enumerate(texts) if t is not None]
    if keep:
        place_labels(fig, ax, [texts[i] for i in keep], 
                     ax.transData.transform(pts[keep]))

    if EQUAL_ASPECT:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Mean similarity to {BENCH_LBL[BENCHMARK]} " + r"(Spearman $\rho$)",
                  fontsize=FS_LABEL)
    ax.set_ylabel(r"Mean similarity to SUD (Spearman $\rho$)", fontsize=FS_LABEL)
    ax.set_title("A. Preferential similarity to SUD", fontsize=FS_TITLE, loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=c,
                              markeredgecolor=c, markersize=9, label=k)
                       for k, c in C.CLUSTER_COLORS.items()],
              fontsize=FS_LEGEND, frameon=False, loc="lower right")
    ax.tick_params(labelsize=FS_TICK)

    # ------------------------- Panel B -------------------------
    ax = axes[1]
    d = df.sort_values("mean_rho_SUD")           # bottom-to-top = ascending hierarchy
    band_null = "spin" if "null_lo_spin" in d.columns else have[0]
    for i, (name, row) in enumerate(d.iterrows()):
        col = C.CLUSTER_COLORS[row["cluster"]]
        if f"null_lo_{band_null}" in d.columns:
            ax.barh(i, row[f"null_hi_{band_null}"] - row[f"null_lo_{band_null}"],
                    left=row[f"null_lo_{band_null}"], height=.64,
                    color="0.89", edgecolor="none", zorder=0)
        ax.plot([row["delta_boot_lo"], row["delta_boot_hi"]], [i, i],
                c=col, lw=2.6, zorder=2)
        ax.scatter(row["delta"], i, s=110, color=col, zorder=3,
                   edgecolor="k" if row["sig"] else "none", linewidth=.9)
        if row["sig"]:
            ax.text(row["delta_boot_hi"] + .015, i, "*", fontsize=17,
                    va="center", color="k")

    ax.axvline(0, c="0.3", lw=1.2)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d.index, fontsize=FS_TICK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel(r"$\Delta$ = mean $\rho$(PSY, SUD) $-$ mean $\rho$(PSY, benchmark)",
                  fontsize=FS_LABEL)
    ax.set_title(f"B. $\\Delta$ with bootstrap 95% CI (grey = {band_null} null)",
                 fontsize=FS_TITLE, loc="left")

    fig.tight_layout()
    stem = os.path.join(FIGDIR, f"RQ1_Fig_specificity_{BENCHMARK}")
    for ext, dpi in (("png", 300), ("pdf", 400), ("svg", 400)):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")

    print(f"Significance convention: {conv}")
    print(df[["cluster", "mean_rho_PSY", "mean_rho_SUD", "delta", "sig"]]
          .round(3).to_string())
    print(f"\nSaved -> {stem}.[png|pdf|svg]")


if __name__ == "__main__":
    main()