#!/usr/bin/env python3
"""
AIM 2 — step 2: Figure 3, panel D

Correlations only, no brain surfaces.
  x-axis : SUD mean, adult psychiatric mean, pediatric psychiatric mean, and the
           four cluster psychiatric means
  y-axis : C1, C2, C3   (C1 is retained deliberately)

Asterisks mark associations surviving FDR under BOTH null frameworks; if the
BrainSMASH columns are absent the script falls back to the spin family and warns
on stdout.

The FDR family is the panel itself (7 x 3 = 21 tests), handled upstream in
AIM2_step1_loo_panelD.py. Nothing is re-corrected here.

Just press Run. Requires AIM2_step1 to have been run first.
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
SHOW_VALUES = True       # False = asterisks only, as in Figure 2B
SPINE_COLOR = "black"
SPINE_WIDTH = 1.0
VLIM = 0.75              # colour scale limit, symmetric around zero
CB_WIDTH_IN = 3.2        # standalone colorbar, saved as its own file
CB_HEIGHT_IN = 0.28
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ2", "aim2")

COL_ORDER = ["SUD mean", "Adults", "Pediatric", "Psychotic", "AN/OCD",
             "Mood/Anx", "Neurodev"]
COL_LABELS = {"SUD mean": "SUD\nmean", "Adults": "PSY\nadult", "Pediatric": "PSY\npediatric",
              "Psychotic": "Psychotic", "AN/OCD": "AN/OCD",
              "Mood/Anx": "Mood/\nanxiety", "Neurodev": "Neuro-\ndevelopmental"}
ROWS = ["C1", "C2", "C3"]

df = pd.read_csv(os.path.join(OUTDIR, "PART2_panelD_decomposed.csv"))
df = df[df["map"] == "component"]

if "pFDR_brainsmash" in df.columns:
    df["sig"] = (df["pFDR_spin"] < .05) & (df["pFDR_brainsmash"] < .05)
else:
    df["sig"] = df["pFDR_spin"] < .05
    print("WARNING: BrainSMASH columns absent; asterisks reflect the spin family only.")

R = df.pivot(index="gradient", columns="group", values="r").reindex(ROWS)[COL_ORDER]
Sg = df.pivot(index="gradient", columns="group", values="sig").reindex(ROWS)[COL_ORDER]

fig, ax = plt.subplots(figsize=(12.5, 3.1))
im = ax.imshow(R.to_numpy(float), cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, aspect="auto")

# white grid between cells, as in the Figure 2 heatmaps
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
ax.tick_params(axis="both", which="major", length=0)   # labels only, no tick marks

for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)          # no frame around the heatmap

fig.tight_layout()
for ext, dpi in (("png", 200), ("pdf", 400), ("svg", 400)):
    fig.savefig(os.path.join(OUTDIR, f"Fig3D_transcriptional.{ext}"),
                dpi=dpi, bbox_inches="tight")
print("Saved ->", os.path.join(OUTDIR, "Fig3D_transcriptional.[png|pdf|svg]"))

# ------------------------------------------------------------------
# Colorbar as a separate file, styled to match the ENIGMA Toolbox bars used
# for the shared-map surfaces (thin horizontal bar, no frame, no tick marks,
# the two end values only, label above). The Toolbox bars are drawn by VTK
# inside the surface screenshots, so this is a matplotlib reproduction of the
# look rather than the same code.
# Same colormap and limits as the heatmap above — change VLIM and both update.
# ------------------------------------------------------------------
figcb, axcb = plt.subplots(figsize=(CB_WIDTH_IN, CB_HEIGHT_IN))
cb = figcb.colorbar(im, cax=axcb, orientation="horizontal")
cb.set_ticks([-VLIM, VLIM])                       # end points only
cb.ax.set_xticklabels([f"{-VLIM:g}", f"{VLIM:g}"], fontsize=FS_TICK)
cb.ax.tick_params(length=0, pad=3)
cb.outline.set_visible(False)
cb.ax.xaxis.set_label_position("top")             # label above the bar
cb.set_label("Correlation (r)", fontsize=FS_LABEL, labelpad=7)
for ext, dpi in (("png", 200), ("pdf", 400), ("svg", 400)):
    figcb.savefig(os.path.join(OUTDIR, f"Fig3D_colorbar.{ext}"),
                  dpi=dpi, bbox_inches="tight", transparent=True)
print("Saved ->", os.path.join(OUTDIR, "Fig3D_colorbar.[png|pdf|svg]"))
print(R.round(3).to_string())