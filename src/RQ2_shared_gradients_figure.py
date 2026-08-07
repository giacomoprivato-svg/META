#!/usr/bin/env python3
"""
RQ2 — step 4: Figure 3, panel C — adult shared map vs the five gradients
========================================================================

Analysis A of RQ2_step2_gradients.py, drawn. One row per gradient:

  * a filled marker for the adult SHARED map, the tested quantity, with an
    asterisk when it survives its BH family (5 tests, applied upstream)
  * two small open markers for the PSY and SUD components, joined by a thin
    connector, descriptive and NOT part of any family

The components are on the figure for one reason. On C1 they fall on OPPOSITE
sides of zero (PSY negative, SUD positive) and the shared map lands near zero
between them. Without the components on the panel, that cell reads as "the
shared pattern is not organised along C1", which is not what the data say:
the two domains ARE organised along C1, in opposite directions, and the mean
cancels them. The connector is drawn in a warning colour whenever the two
components straddle zero, so the distinction is visible rather than buried in
a supplementary table.

The x-axis is a correlation, so it is fixed to a symmetric range and the zero
line is drawn: a reader should be able to see how far from zero each estimate
is without reading numbers off.

Just press Run. Requires RQ2_step2_gradients.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ================= CONFIG — edit, then press Run =================
FIGSIZE = (7.2, 4.2)
XLIM = 0.8
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 15, 13, 13, 11
COL_SHARED = "#22303F"
COL_PSY = "#4C6EF5"
COL_SUD = "#E8590C"
COL_WARN = "#C92A2A"          # connector when the components straddle zero
GRADIENT_ORDER = ["C1", "C2", "C3", "FC", "MPC"]
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ2", "gradients")
FIGDIR = os.path.join(repo_dir, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load():
    a = os.path.join(OUTDIR, "A_shared_map_vs_gradients.csv")
    c = os.path.join(OUTDIR, "A_component_decomposition.csv")
    for f in (a, c):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f}\nRun RQ2_step2_gradients.py first.")
    A = pd.read_csv(a).set_index("gradient")
    C_ = pd.read_csv(c).set_index("gradient")
    order = [g for g in GRADIENT_ORDER if g in A.index]
    if len(order) != len(A):
        print(f"  note: gradients in the CSV but not plotted: "
              f"{sorted(set(A.index) - set(order))}")
    if "robust" not in A.columns:
        raise KeyError("no `robust` column — rerun step 2.")
    if "pFDR_brainsmash" not in A.columns:
        print("WARNING: BrainSMASH columns absent; asterisks reflect the spin family only.")
    return A.loc[order], C_.loc[order], order


def main():
    A, Cm, order = load()
    y = np.arange(len(order))[::-1]      # first gradient at the top

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axvline(0, color="0.35", lw=1.1, zorder=1)

    for k, g in enumerate(order):
        yy = y[k]
        rp, rs = Cm.loc[g, "r_PSY_component"], Cm.loc[g, "r_SUD_component"]
        straddles = bool(Cm.loc[g, "opposite_signs"])
        ax.plot([rp, rs], [yy, yy], lw=1.6, zorder=2,
                color=COL_WARN if straddles else "0.75")
        ax.scatter(rp, yy, s=55, facecolor="white", edgecolor=COL_PSY,
                   linewidth=1.8, zorder=3)
        ax.scatter(rs, yy, s=55, facecolor="white", edgecolor=COL_SUD,
                   linewidth=1.8, zorder=3)

        r = A.loc[g, "r"]
        sig = bool(A.loc[g, "robust"])
        ax.scatter(r, yy, s=170, color=COL_SHARED, zorder=4,
                   edgecolor="white", linewidth=1.2)
        qcol = "pFDR_brainsmash" if "pFDR_brainsmash" in A.columns else "pFDR_spin"
        lab = f"{r:+.2f}" + ("*" if sig else "")
        ax.annotate(lab, (r, yy), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=FS_TICK,
                    fontweight="bold" if sig else "normal", color="0.15")

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=FS_TICK)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_xlabel("Correlation with the adult shared map (r)", fontsize=FS_LABEL)
    ax.set_title("C. Alignment with cortical gradients", loc="left", fontsize=FS_TITLE)
    ax.tick_params(axis="both", length=0, labelsize=FS_TICK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="", markersize=10, color=COL_SHARED,
               label="Shared map (tested)"),
        Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor="white",
               markeredgecolor=COL_PSY, markeredgewidth=1.8, label="PSY component"),
        Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor="white",
               markeredgecolor=COL_SUD, markeredgewidth=1.8, label="SUD component"),
        Line2D([], [], color=COL_WARN, lw=1.6, label="components straddle zero"),
    ]
    ax.legend(handles=handles, fontsize=FS_LEG, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    fig.tight_layout()
    stem = os.path.join(FIGDIR, "Fig3C_shared_vs_gradients")
    for ext, dpi in (("png", 300), ("pdf", 400), ("svg", 400)):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")

    show = ["r", "p_spin", "pFDR_spin"] + \
           [c for c in ("p_brainsmash", "pFDR_brainsmash", "robust") if c in A.columns]
    print(A[show].round(4).to_string())
    straddling = [g for g in order if bool(Cm.loc[g, "opposite_signs"])]
    if straddling:
        print(f"\ncomponents straddle zero on: {straddling} — a null there is "
              f"cancellation between domains, not absence of organisation.")
    print(f"\nSaved -> {stem}.[png|pdf|svg]")


if __name__ == "__main__":
    main()