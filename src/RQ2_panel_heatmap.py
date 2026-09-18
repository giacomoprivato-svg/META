#!/usr/bin/env python3
"""
RQ2 — step 3: Figure 3, panel D

"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= CONFIG — edit, then press Run =================
FS_TITLE = 15
FS_LABEL = 15
FS_TICK = 15
FS_CELL = 15
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
ax.set_title("", loc="left", fontsize=FS_TITLE)
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