#!/usr/bin/env python3
"""
RQ1 - Panel F (PSY cluster fingerprints), standalone

"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import RQ1_common as C

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
BASE = os.path.join(REPO_DIR, "ALL_outputs_RQ1")
OUT_DIR = os.path.join(REPO_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

METRIC = "spearman"
METRIC_LABEL = "Spearman \u03c1"
GROUP_DIRS = ["adults_all", "adults_ctx"]
DROP_PATTERN = "Schizotyp"

FS_LABEL = 20
FS_TITLE = 22
FS_TICK = 20
FS_LEG = 18

# membership from the single source of truth; colours kept verbatim
manual_clusters = {k: list(v) for k, v in C.CLUSTER_MEMBERS.items()}
cluster_colors = {"Psychotic": (1.0, 0.6, 0.0),
                  "Neurodevelopmental": (0.2, 0.8, 0.2),
                  "AN/OCD": (0.2, 0.4, 0.8),
                  "Mood/Anxiety": (0.7, 0.3, 0.7)}
# draw in the original order regardless of dict ordering upstream
PLOT_ORDER = ["Psychotic", "Neurodevelopmental", "AN/OCD", "Mood/Anxiety"]


def rd(fname):
    parts = []
    for g in GROUP_DIRS:
        p = os.path.join(BASE, g, fname)
        if os.path.exists(p):
            parts.append(pd.read_csv(p, index_col=0, sep=None, engine="python"))
    if not parts:
        raise FileNotFoundError(f"{fname} not found in {GROUP_DIRS} under {BASE}")
    M = pd.concat(parts, axis=0).apply(pd.to_numeric, errors="coerce")
    return M[~M.index.str.contains(DROP_PATTERN, case=False, na=False)]


R = rd(f"RAW_cortex_{METRIC}.csv")
psy_names = list(R.index)
sud_names = list(R.columns)
C.check_cluster_coverage(psy_names)

fig, axC = plt.subplots(figsize=(16, 6))
for cname in PLOT_ORDER:
    disorders = manual_clusters[cname]
    missing = [d for d in disorders if d not in psy_names]
    if missing:
        raise KeyError(f"{cname}: {missing} absent from RAW_cortex_{METRIC}.csv. "
                       f"Plotting the cluster mean over the remaining members "
                       f"would silently draw the wrong line.")
    inds = [psy_names.index(d) for d in disorders]
    prof = np.nanmean(R.values[inds, :], axis=0)
    axC.plot(range(len(sud_names)), prof, marker="o", lw=2,
             color=cluster_colors[cname], label=cname)

axC.axhline(0, color="0.6", lw=1, ls="--", zorder=0)
axC.set_xticks(range(len(sud_names)))
axC.set_xticklabels(sud_names, ha="right", fontsize=FS_TICK)
axC.tick_params(axis="y", labelsize=FS_TICK)
axC.set_ylabel(METRIC_LABEL, fontsize=FS_LABEL)
axC.set_title("", fontsize=FS_TITLE)
axC.grid(False)
axC.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.12),
           ncol=len(manual_clusters), fontsize=FS_LEG)

out = os.path.join(OUT_DIR, "RQ1_panelF_spearman.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"saved -> {out}")