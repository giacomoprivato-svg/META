#!/usr/bin/env python3
"""
RQ1 — supplementary 16 x 16 joint similarity matrix with hierarchical clustering
===============================================================================

Full Spearman similarity matrix across all 9 adult psychiatric maps and all 7
SUD maps (6 substance-specific + the transdiagnostic all-SUD map), clustered
with average linkage on 1 - rho.

This belongs in the SUPPLEMENT, not the main text: in the main text it would
shift the paper toward a transdiagnostic taxonomy that is not the stated aim.
Its value is diagnostic — whether the SUD maps form a distinct branch or
intercalate among the psychiatric disorders is the specificity question in its
most legible form, and the answer should be known BEFORE the Results and
Discussion are finalised.

Note on the all-SUD map: it is included here (unlike in the Delta analysis,
where it is dropped) because this is a descriptive display of every map in the
dataset, not a benchmark comparison. It is a weighted composite of the six
substance maps, so its position in the dendrogram is partly redundant by
construction — do not read it as independent evidence of a SUD branch. Set
DROP_ALLSUD = True to rerun the clustering without it as a check.

WHAT CHANGED
------------
1. CLUSTERS COME FROM RQ1_common. This script was the fourth place the same
   four clinical groups were declared, and it still carried PTSD. The label
   only feeds the `clinical_cluster` column of JOINT16_cluster_assignments.csv,
   so a stale entry would not have crashed anything — `CLUSTERS.get(c, "SUD")`
   silently labels any unrecognised psychiatric map as "SUD". With PTSD renamed
   to PD, the PD row of that supplementary table would have read "SUD", i.e.
   the file would have claimed the dataset has eight SUD maps and eight
   psychiatric ones. check_cluster_coverage now raises instead.

2. Everything else is untouched: linkage, cophenetic correlation, the k = 2..5
   branch diagnostics, the within-block means, figsize (8.6, 8.2), colours,
   label rotation, output filenames.

Outputs -> ALL_outputs_RQ1/specificity_adults/
    JOINT16_spearman_matrix.csv
    JOINT16_clustermap.[png|pdf|svg]
    JOINT16_cluster_assignments.csv
    JOINT16_branch_summary.txt

Just press Run.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform

import RQ1_common as C

N_CORTEX = C.N_CORTEX
LINKAGE = "average"

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ1", "specificity_adults")
os.makedirs(OUTDIR, exist_ok=True)

DOMAIN_COLORS = {"PSY": "#4C6EF5", "SUD": "#E8590C"}

# ================= CONFIG — edit, then press Run =================
DROP_ALLSUD = False      # True = rerun as a 15x15 check without the all-SUD map
# ================================================================

# ------------------------------------------------------------------
# Load and combine
# ------------------------------------------------------------------
psy, psy_cols = C.read_maps(os.path.join(data_dir, "PSY_adults.xlsx"), N_CORTEX)
sud, sud_cols = C.read_maps(os.path.join(data_dir, "SUD.xlsx"), N_CORTEX,
                            drop_cols=["SUD"] if DROP_ALLSUD else [])
C.check_cluster_coverage(psy_cols)

M = pd.concat([psy, sud], axis=1)
domain = pd.Series(["PSY"] * psy.shape[1] + ["SUD"] * sud.shape[1], index=M.columns)

# ------------------------------------------------------------------
# Spearman similarity matrix
# ------------------------------------------------------------------
R = rankdata(M.to_numpy(float), axis=0)
S = pd.DataFrame(np.corrcoef(R.T), index=M.columns, columns=M.columns)
S.to_csv(os.path.join(OUTDIR, "JOINT16_spearman_matrix.csv"))

D = 1.0 - S.to_numpy()
np.fill_diagonal(D, 0.0)
D = (D + D.T) / 2.0
Z = linkage(squareform(D, checks=False), method=LINKAGE)
coph, _ = cophenet(Z, squareform(D, checks=False))

# ------------------------------------------------------------------
# Cluster assignments at k = 2..5 and branch diagnostics
# ------------------------------------------------------------------
assign = pd.DataFrame(
    {"domain": domain,
     "clinical_cluster": [C.CLUSTERS[c] if d == "PSY" else "SUD"
                          for c, d in zip(M.columns, domain)]},
    index=M.columns)
for k in range(2, 6):
    assign[f"k{k}"] = fcluster(Z, k, criterion="maxclust")
assign.to_csv(os.path.join(OUTDIR, "JOINT16_cluster_assignments.csv"))

lines = [f"Joint matrix: {S.shape[0]} maps ({(domain=='PSY').sum()} PSY, "
         f"{(domain=='SUD').sum()} SUD); linkage={LINKAGE} on 1-rho; "
         f"cophenetic r = {coph:.3f}", ""]

sud_names = list(domain[domain == "SUD"].index)
for k in range(2, 6):
    lab = assign[f"k{k}"]
    sud_labs = set(lab[sud_names])
    if len(sud_labs) == 1:
        c = sud_labs.pop()
        members = list(lab[lab == c].index)
        intruders = [m for m in members if domain[m] == "PSY"]
        lines.append(
            f"k={k}: all SUD maps share one branch; that branch also contains "
            f"{len(intruders)} psychiatric map(s): {intruders if intruders else 'none (PURE SUD BRANCH)'}")
    else:
        spread = {c: list(lab[lab == c].index) for c in sorted(sud_labs)}
        lines.append(f"k={k}: SUD maps split across {len(sud_labs)} branches -> {spread}")

# how far each PSY map sits from the SUD block vs from the PSY block
psy_names = list(domain[domain == "PSY"].index)
summary = pd.DataFrame({
    "mean_rho_to_SUD": S.loc[psy_names, sud_names].mean(axis=1),
    "mean_rho_to_otherPSY": S.loc[psy_names, psy_names].where(
        ~np.eye(len(psy_names), dtype=bool)).mean(axis=1),
})
summary["delta_incl_allSUD"] = summary["mean_rho_to_SUD"] - summary["mean_rho_to_otherPSY"]
lines += ["", "Descriptive per-disorder means from this matrix "
              "(NB: SUD side here includes the all-SUD map, so these differ "
              "slightly from the Delta analysis):", summary.round(3).to_string()]
lines += ["", "Mean within-block rho:",
          f"  PSY-PSY  = {S.loc[psy_names, psy_names].where(~np.eye(len(psy_names), dtype=bool)).stack().mean():.3f}",
          f"  SUD-SUD  = {S.loc[sud_names, sud_names].where(~np.eye(len(sud_names), dtype=bool)).stack().mean():.3f}",
          f"  PSY-SUD  = {S.loc[psy_names, sud_names].to_numpy().mean():.3f}"]

txt = "\n".join(lines)
with open(os.path.join(OUTDIR, "JOINT16_branch_summary.txt"), "w") as f:
    f.write(txt + "\n")
print(txt)

# ------------------------------------------------------------------
# Clustermap
# ------------------------------------------------------------------
colors = domain.map(DOMAIN_COLORS)
g = sns.clustermap(S, row_linkage=Z, col_linkage=Z, cmap="RdBu_r",
                   vmin=-1, vmax=1, center=0,
                   row_colors=colors, col_colors=colors,
                   figsize=(8.6, 8.2), linewidths=.4, linecolor="white",
                   cbar_kws={"label": r"Spearman $\rho$"})
g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
g.fig.suptitle(f"Joint PSY–SUD similarity matrix ({LINKAGE} linkage on 1–$\\rho$)",
               y=1.01, fontsize=12)
stem = "JOINT15_clustermap" if DROP_ALLSUD else "JOINT16_clustermap"
for ext in ("png", "pdf", "svg"):
    g.savefig(os.path.join(OUTDIR, f"{stem}.{ext}"), dpi=400, bbox_inches="tight")
print("\nSaved ->", os.path.join(OUTDIR, f"{stem}.[png|pdf|svg]"))