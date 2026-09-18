#!/usr/bin/env python3
"""
RQ1 — supplementary joint similarity matrix with hierarchical clustering

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
DROP_ALLSUD = True       # False = descriptive 16x16 display including the aggregate
# ================================================================

# Spellings the aggregate column might appear under, for the post-drop guard.
AGGREGATE_SUD_NAMES = ["SUD", "ALL_SUD", "ALLSUD", "SUD_ALL", "ALL SUD"]
N_SUBSTANCE_MAPS = 6     # ALC ATS CAN COC NIC OPI

# ------------------------------------------------------------------
# Load and combine
# ------------------------------------------------------------------
psy, psy_cols = C.read_maps(os.path.join(data_dir, "PSY_adults.xlsx"), N_CORTEX)
sud, sud_cols = C.read_maps(os.path.join(data_dir, "SUD.xlsx"), N_CORTEX,
                            drop_cols=["SUD"] if DROP_ALLSUD else [])
C.check_cluster_coverage(psy_cols)

if DROP_ALLSUD:
    wanted = {n.strip().upper() for n in AGGREGATE_SUD_NAMES}
    survivors = [c for c in sud.columns if str(c).strip().upper() in wanted]
    if survivors:
        raise RuntimeError(
            f"DROP_ALLSUD is True but {survivors} survived read_maps. The column is "
            f"named something other than 'SUD' — fix the drop_cols argument. Columns "
            f"loaded: {list(sud.columns)}")
    if sud.shape[1] != N_SUBSTANCE_MAPS:
        raise RuntimeError(
            f"expected {N_SUBSTANCE_MAPS} substance-specific maps, got {sud.shape[1]}: "
            f"{list(sud.columns)}")

M = pd.concat([psy, sud], axis=1)
domain = pd.Series(["PSY"] * psy.shape[1] + ["SUD"] * sud.shape[1], index=M.columns)

# Filenames follow the matrix, not a hardcoded literal, so a 15-map run and a
# 16-map run cannot overwrite each other.
stem = f"JOINT{M.shape[1]}"
print(f"{M.shape[1]} maps: {(domain == 'PSY').sum()} PSY + {(domain == 'SUD').sum()} SUD "
      f"(all-SUD aggregate {'EXCLUDED' if DROP_ALLSUD else 'INCLUDED'}) -> {stem}_*")

# ------------------------------------------------------------------
# Spearman similarity matrix
# ------------------------------------------------------------------
R = rankdata(M.to_numpy(float), axis=0)
S = pd.DataFrame(np.corrcoef(R.T), index=M.columns, columns=M.columns)
S.to_csv(os.path.join(OUTDIR, f"{stem}_spearman_matrix.csv"))

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
assign.to_csv(os.path.join(OUTDIR, f"{stem}_cluster_assignments.csv"))

lines = [f"Joint matrix: {S.shape[0]} maps ({(domain=='PSY').sum()} PSY, "
         f"{(domain=='SUD').sum()} SUD); all-SUD aggregate "
         f"{'EXCLUDED' if DROP_ALLSUD else 'INCLUDED'}; linkage={LINKAGE} on 1-rho; "
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
delta_col = "delta_excl_allSUD" if DROP_ALLSUD else "delta_incl_allSUD"
summary[delta_col] = summary["mean_rho_to_SUD"] - summary["mean_rho_to_otherPSY"]
note = ("SUD side here is the six substance-specific maps, matching the Delta "
        "analysis" if DROP_ALLSUD else
        "NB: SUD side here includes the all-SUD map, so these differ slightly "
        "from the Delta analysis")
lines += ["", f"Descriptive per-disorder means from this matrix ({note}):",
          summary.round(3).to_string()]


# ------------------------------------------------------------------
# Centrality, within-set coherence, effective dimensionality
# ------------------------------------------------------------------
def offdiag_mean(block):
    """Mean of the off-diagonal entries of a square similarity submatrix."""
    a = block.to_numpy(float)
    return a[~np.eye(a.shape[0], dtype=bool)].mean()


def effective_dimensionality(block):
    """
    Participation ratio of the eigenvalue spectrum, (sum L)^2 / sum(L^2).

    Runs from 1 (every map identical, one dimension) to the number of maps
    (mutually orthogonal). Reported for each block as a scale-free summary of
    how many independent patterns the block contains. Tiny negative eigenvalues
    from numerical error are clipped to zero before the ratio is formed.
    """
    ev = np.linalg.eigvalsh(block.to_numpy(float))
    ev = np.clip(ev, 0.0, None)
    return float((ev.sum() ** 2) / (ev ** 2).sum())


# Centrality = mean rho to every OTHER map in the joint matrix. With the
# aggregate included this is not comparable across maps: the aggregate is a
# composite of six of them and inflates their scores as well as its own.
S_np = S.to_numpy(float)
centrality = pd.Series(
    S_np[~np.eye(S.shape[0], dtype=bool)].reshape(S.shape[0], -1).mean(axis=1),
    index=S.index, name="centrality_mean_rho_to_others")
centrality_tbl = (pd.DataFrame({"domain": domain, "centrality": centrality})
                  .sort_values("centrality", ascending=False))
centrality_tbl.round(6).to_csv(os.path.join(OUTDIR, f"{stem}_centrality.csv"))

psy_block = S.loc[psy_names, psy_names]
sud_block = S.loc[sud_names, sud_names]

lines += ["", "Centrality (mean rho to all other maps in the joint matrix), ranked:",
          centrality_tbl.round(3).to_string()]
lines += ["", "Within-block coherence and effective dimensionality:",
          f"  PSY-PSY  mean rho = {offdiag_mean(psy_block):.3f}   "
          f"effective dimensionality = {effective_dimensionality(psy_block):.2f} "
          f"of {len(psy_names)}",
          f"  SUD-SUD  mean rho = {offdiag_mean(sud_block):.3f}   "
          f"effective dimensionality = {effective_dimensionality(sud_block):.2f} "
          f"of {len(sud_names)}",
          f"  PSY-SUD  mean rho = {S.loc[psy_names, sud_names].to_numpy().mean():.3f}"]
if not DROP_ALLSUD:
    lines += ["  [warn] the aggregate map is in the SUD block, so SUD-SUD coherence is "
              "inflated and its effective dimensionality deflated by construction. "
              "Do not quote these two numbers in the manuscript from this run."]

txt = "\n".join(lines)
with open(os.path.join(OUTDIR, f"{stem}_branch_summary.txt"), "w") as f:
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
for ext in ("png", "pdf", "svg"):
    g.savefig(os.path.join(OUTDIR, f"{stem}_clustermap.{ext}"), dpi=400,
              bbox_inches="tight")
print("\nSaved ->", os.path.join(OUTDIR, f"{stem}_clustermap.[png|pdf|svg]"))