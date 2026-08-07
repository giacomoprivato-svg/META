#!/usr/bin/env python3
"""
RQ2 — step 3: Figure 3, panel D
===============================

Correlations only, no brain surfaces.
  columns : SUD mean, adult PSY mean, pediatric PSY mean, and the four cluster
            PSY means — component maps, never shared maps
  rows    : C1, C2, C3   (C1 is retained deliberately)

Asterisks mark cells surviving FDR under BOTH null frameworks; with only one
null present the script falls back to it and warns.

The FDR family is analysis B itself (7 x 3 = 21 tests), applied upstream in
RQ2_step2_gradients.py. Nothing is re-corrected here.

Figure geometry is UNCHANGED from AIM2_step2: figsize (12.5, 3.1), VLIM 0.75,
white minor grid, no frame, values inside the cells, and the standalone
colorbar file at (3.2, 0.28) with end ticks only and the label above.

WHAT CHANGED
------------
1. Reads B_panelD_components.csv (analysis B of the rewritten step 2) instead
   of PART2_panelD_decomposed.csv. The `map` column is gone — analysis B
   contains component maps only by construction, so there is nothing left to
   filter and no way to plot a shared-map row here by accident.

2. Cluster keys are the long RQ1_common names, "Mood/Anxiety" and
   "Neurodevelopmental". The display labels on the axis are unchanged.

3. It fails loudly if a requested column is missing from the CSV. The old
   version would have raised a bare KeyError from the pandas indexer, which on
   a 7-column reindex is not obvious to read.

Just press Run. Requires RQ2_step2_gradients.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= CONFIG — edit, then press Run =================
FS_TITLE = 15
FS_LABEL = 13
FS_TICK = 12
FS_CELL = 12
SHOW_VALUES = True       # False = asterisks only
VLIM = 0.75              # colour scale limit, symmetric around zero
CB_WIDTH_IN = 3.2
CB_HEIGHT_IN = 0.28
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ2", "gradients")
FIGDIR = os.path.join(repo_dir, "figures")
os.makedirs(FIGDIR, exist_ok=True)

COL_ORDER = ["SUD mean", "Adults", "Pediatric", "Psychotic", "AN/OCD",
             "Mood/Anxiety", "Neurodevelopmental"]
COL_LABELS = {"SUD mean": "SUD\nmean", "Adults": "PSY\nadult",
              "Pediatric": "PSY\npediatric", "Psychotic": "Psychotic",
              "AN/OCD": "AN/OCD", "Mood/Anxiety": "Mood/\nanxiety",
              "Neurodevelopmental": "Neuro-\ndevelopmental"}
ROWS = ["C1", "C2", "C3"]

src = os.path.join(OUTDIR, "B_panelD_components.csv")
if not os.path.exists(src):
    raise FileNotFoundError(f"{src}\nRun RQ2_step2_gradients.py first.")
df = pd.read_csv(src)

missing = [c for c in COL_ORDER if c not in set(df["group"])]
if missing:
    raise KeyError(f"groups absent from {os.path.basename(src)}: {missing}. "
                   f"Present: {sorted(set(df['group']))}. If a cluster was renamed, "
                   f"fix it in RQ1_common.CLUSTER_MEMBERS, not here.")

if "robust" in df.columns and "pFDR_brainsmash" in df.columns:
    df["sig"] = df["robust"]
elif "pFDR_spin" in df.columns:
    df["sig"] = df["pFDR_spin"] < .05
    print("WARNING: BrainSMASH columns absent; asterisks reflect the spin family only.")
else:
    raise KeyError("no pFDR columns in the input.")

R = df.pivot(index="gradient", columns="group", values="r").reindex(ROWS)[COL_ORDER]
Sg = df.pivot(index="gradient", columns="group", values="sig").reindex(ROWS)[COL_ORDER]

fig, ax = plt.subplots(figsize=(12.5, 3.1))
im = ax.imshow(R.to_numpy(float), cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, aspect="auto")

ax.set_xticks(np.arange(-.5, len(COL_ORDER), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(ROWS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.6)
ax.tick_params(which="minor", length=0)

for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        r = R.iat[i, j]
        star = bool(Sg.iat[i, j])
        txt = (f"{r:+.2f}" + ("*" if star else "")) if SHOW_VALUES else ("*" if star else "")
        ax.text(j, i, txt, ha="center", va="center", fontsize=FS_CELL,
                fontweight="bold" if star else "normal",
                color="white" if abs(r) > .45 else "0.12")

ax.set_xticks(range(len(COL_ORDER)))
ax.set_xticklabels([COL_LABELS[c] for c in COL_ORDER], fontsize=FS_TICK)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels(ROWS, fontsize=FS_TICK)
ax.set_title("D. Transcriptional components", loc="left", fontsize=FS_TITLE)
ax.tick_params(axis="both", which="major", length=0)
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
stem = os.path.join(FIGDIR, "Fig3D_transcriptional")
for ext, dpi in (("png", 200), ("pdf", 400), ("svg", 400)):
    fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
print("Saved ->", f"{stem}.[png|pdf|svg]")

# ------------------------------------------------------------------
# Colorbar as a separate file, styled to match the ENIGMA Toolbox bars used
# for the shared-map surfaces (thin horizontal bar, no frame, no tick marks,
# the two end values only, label above). Same colormap and limits as the
# heatmap — change VLIM and both update.
# ------------------------------------------------------------------
import matplotlib as mpl
figcb, axcb = plt.subplots(figsize=(CB_WIDTH_IN, CB_HEIGHT_IN))
# build the bar from the same cmap/limits rather than borrowing the heatmap's
# mappable, which belongs to the other figure and makes matplotlib warn
cb = mpl.colorbar.ColorbarBase(
    axcb, cmap=plt.get_cmap("RdBu_r"),
    norm=mpl.colors.Normalize(vmin=-VLIM, vmax=VLIM), orientation="horizontal")
cb.set_ticks([-VLIM, VLIM])
cb.ax.set_xticklabels([f"{-VLIM:g}", f"{VLIM:g}"], fontsize=FS_TICK)
cb.ax.tick_params(length=0, pad=3)
cb.outline.set_visible(False)
cb.ax.xaxis.set_label_position("top")
cb.set_label("Correlation (r)", fontsize=FS_LABEL, labelpad=7)
cbstem = os.path.join(FIGDIR, "Fig3D_colorbar")
for ext, dpi in (("png", 200), ("pdf", 400), ("svg", 400)):
    figcb.savefig(f"{cbstem}.{ext}", dpi=dpi, bbox_inches="tight", transparent=True)
print("Saved ->", f"{cbstem}.[png|pdf|svg]")
print(R.round(3).to_string())