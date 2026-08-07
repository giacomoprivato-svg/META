#!/usr/bin/env python3
"""
RQ1 - Panel A (PSY x SUD heatmap), standalone
=============================================
SPEARMAN primary version (raw rho; rank correspondence between maps).

Layout, ordering, colour scale, font sizes, aspect ratio and the separate
colorbar file are UNCHANGED from Figure2_Spearman_final_combined. The figure
canvas is still 26 x 14 and the heatmap still occupies the rect
[0.05, 0.1, 0.5, 0.8], i.e. 13 x 11.2 inches, so with bbox_inches="tight" the
panel comes out at exactly the size it did when B/C/D shared the canvas.

ONLY THREE THINGS CHANGED, all forced by the pipeline rewrite:

1. FILENAME. The spin p-values are now read from
   PVAL_cortex_{metric}_spin.csv. They used to be PVAL_cortex_{metric}.csv,
   a name that did not say which null or which compartment produced it — and
   the subcortex script wrote files with colliding names into the same folder.

2. FDR. C.bh_fdr_by_column replaces statsmodels.fdrcorrection. Same procedure
   (BH within each SUD column, then intersect the two nulls), same numbers,
   one less dependency and one less implementation of FDR in the paper.

3. Reads 7 SUD columns including the transdiagnostic all-SUD map, as before.
   Caption note: that column is a weighted composite of the other six, so it
   is not independent of them.

Panel D/E/F (fingerprints) live in RQ1_panelF_fingerprints.py, unchanged.

Just press Run.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import RQ1_common as C

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

sns.set(style="whitegrid")

# ---------------------------
# Metric / stats options
# ---------------------------
METRIC = "spearman"                 # primary metric
METRIC_LABEL = "Spearman \u03c1"
NULLS = ["spin", "brainsmash"]      # asterisk = surviving FDR under BOTH
GROUP_DIRS = ["adults_all", "adults_ctx"]
DROP_PATTERN = "Schizotyp"
np.random.seed(42)

ROW_ORDER_MODE = "optimal_leaf"     # "seriation" | "optimal_leaf" | "hierarchical"

# ---------------------------
# Font sizes
# ---------------------------
FONT_TICKS = 26
FONT_AST = 34

BASE = os.path.join(repo_dir, "ALL_outputs_RQ1")
OUT_DIR = os.path.join(repo_dir, "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def optimal_leaf_order(mat, axis):
    from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import pdist
    data = mat if axis == 0 else mat.T
    if np.isnan(data).any():
        cm = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data = data.copy(); data[inds] = np.take(cm, inds[1])
    if data.shape[0] <= 2:
        return np.arange(data.shape[0])
    d = pdist(data)
    Z = linkage(d, method="average")
    return leaves_list(optimal_leaf_ordering(Z, d))


def hierarchical_order(mat, axis=0, method="ward", metric="euclidean"):
    from scipy.cluster.hierarchy import linkage, leaves_list
    data = mat if axis == 0 else mat.T
    if np.isnan(data).any():
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])
    if data.shape[0] <= 1:
        return np.arange(data.shape[0])
    Z = linkage(data, method=method, metric=metric)
    return leaves_list(Z)


def cosmetic_relabel(ax):
    labels = [t.get_text() for t in ax.get_yticklabels()]
    labels = ["Schizotypy" if l == "Schizotypic" else l for l in labels]
    ax.set_yticklabels(labels, rotation=90, va="center")


def rd(fname):
    parts = []
    for g in GROUP_DIRS:
        p = os.path.join(BASE, g, fname)
        if os.path.exists(p):
            parts.append(pd.read_csv(p, index_col=0, sep=None, engine="python"))
    if not parts:
        raise FileNotFoundError(
            f"{fname} not found in {GROUP_DIRS} under {BASE}. "
            f"Run RQ1_step1_cortex_similarity.py / RQ1_step2_subcortex_similarity.py.")
    M = pd.concat(parts, axis=0).apply(pd.to_numeric, errors="coerce")
    return M[~M.index.str.contains(DROP_PATTERN, case=False, na=False)]


# -----------------------
# Main
# -----------------------
def main():
    sim_primary = rd(f"RAW_cortex_{METRIC}.csv")
    psy_names = list(sim_primary.index)
    sud_names = list(sim_primary.columns)
    print(f"{len(psy_names)} PSY x {len(sud_names)} SUD: {sud_names}")

    # ---- FDR per column under each null, then INTERSECTION ----
    masks = {}
    for null in NULLS:
        P = rd(f"PVAL_cortex_{METRIC}_{null}.csv").reindex(
            index=psy_names, columns=sud_names)
        masks[null] = C.bh_fdr_by_column(P.to_numpy(float)) < 0.05
        print(f"{null}: {masks[null].sum()}")
    fdr_mask = np.logical_and.reduce(list(masks.values()))
    print(f"BOTH (starred): {fdr_mask.sum()}")

    # ---- Ordering ----
    if ROW_ORDER_MODE == "seriation":
        row_order = list(np.argsort(-sim_primary.mean(axis=1).values))
        col_order = list(np.argsort(-sim_primary.mean(axis=0).values))
    elif ROW_ORDER_MODE == "optimal_leaf":
        row_order = optimal_leaf_order(sim_primary.values, axis=0)
        col_order = optimal_leaf_order(sim_primary.values, axis=1)
    else:
        row_order = hierarchical_order(sim_primary.values)
        col_order = hierarchical_order(sim_primary.values.T)

    # Orient so the strongest similarity sits top-left rather than bottom-right.
    # Reversing the whole order preserves adjacency/clustering (just mirrors it),
    # it only fixes which end points which way.
    row_order = np.asarray(row_order)
    col_order = np.asarray(col_order)
    tmp = sim_primary.iloc[row_order, col_order]
    if tmp.mean(axis=1).values[0] < tmp.mean(axis=1).values[-1]:
        row_order = row_order[::-1]
    if tmp.mean(axis=0).values[0] < tmp.mean(axis=0).values[-1]:
        col_order = col_order[::-1]

    sim_primary_ord = sim_primary.iloc[row_order, col_order]
    fdr_mask_ord = fdr_mask[row_order][:, col_order]

    # =====================================================
    # FIGURE — same canvas and same rect as the combined version
    # =====================================================
    fig = plt.figure(figsize=(26, 14))
    axA = fig.add_axes([0.05, 0.1, 0.5, 0.8])

    vlim = np.nanmax(np.abs(sim_primary_ord.values))
    cax = sns.heatmap(sim_primary_ord, ax=axA, cmap="RdBu_r", center=0,
                      xticklabels=True, yticklabels=True, vmin=-vlim, vmax=vlim,
                      cbar_kws={"label": METRIC_LABEL, "shrink": 0.5, "pad": 0.01})
    cbar = cax.collections[0].colorbar
    cbar.set_label(METRIC_LABEL, fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    axA.tick_params(axis="both", labelsize=FONT_TICKS)
    cosmetic_relabel(axA)
    for i in range(sim_primary_ord.shape[0]):
        for j in range(sim_primary_ord.shape[1]):
            if fdr_mask_ord[i, j]:
                axA.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                         color="black", fontsize=FONT_AST, fontweight="bold")
    axA.set_title(f"A. PSY \u00d7 SUD spatial correspondence ({METRIC_LABEL})", fontsize=20)
    axA.grid(False)
    axA.set_aspect("equal")

    outpath = os.path.join(OUT_DIR, "RQ1_panelA_heatmap_spearman.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---------------- standalone colorbar (unchanged) ----------------
    cbar_fig = plt.figure(figsize=(6, 0.9))
    cbar_ax = cbar_fig.add_axes([0.05, 0.5, 0.9, 0.35])
    norm = mpl.colors.Normalize(vmin=-vlim, vmax=vlim)
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=plt.get_cmap("RdBu_r"),
                                   norm=norm, orientation="horizontal")
    cb.set_label(METRIC_LABEL, fontsize=16)
    cb.ax.tick_params(labelsize=13)
    cbar_out = os.path.splitext(outpath)[0] + "_colorbar.png"
    cbar_fig.savefig(cbar_out, dpi=300, bbox_inches="tight")
    cbar_fig.savefig(os.path.splitext(cbar_out)[0] + ".pdf", bbox_inches="tight")
    plt.close(cbar_fig)
    print(f"Saved figure to {outpath}")

    # =====================================================
    # PAIR RANKING -> single Excel sheet (cortex + subcortex columns)
    # =====================================================
    raw_cortex = pd.read_csv(
        os.path.join(BASE, "adults_all", f"RAW_cortex_{METRIC}.csv"),
        index_col=0).reindex([p for p in psy_names if DROP_PATTERN.lower()
                              not in p.lower()]).dropna(how="all")
    raw_subctx = pd.read_csv(
        os.path.join(BASE, "adults_all", f"RAW_subctx_{METRIC}.csv"),
        index_col=0).reindex(raw_cortex.index)

    def ranked_pairs(mat_df):
        long = (mat_df.reset_index()
                .melt(id_vars="index", var_name="SUD", value_name="rho")
                .rename(columns={"index": "PSY"}))
        long["pair"] = long["PSY"] + "\u2013" + long["SUD"]
        return long.sort_values("rho", ascending=False).reset_index(drop=True)

    rc, rs = ranked_pairs(raw_cortex), ranked_pairs(raw_subctx)
    rank_df = pd.DataFrame({
        "Rank": np.arange(1, len(rc) + 1),
        "Cortex_pair": rc["pair"].values,
        f"Cortex_{METRIC}": rc["rho"].round(4).values,
        "Subctx_pair": rs["pair"].values,
        f"Subctx_{METRIC}": rs["rho"].round(4).values,
    })
    xlsx_out = os.path.join(OUT_DIR, "RQ1_pair_ranking_spearman.xlsx")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as xl:
        rank_df.to_excel(xl, sheet_name="pair_ranking", index=False)
    from openpyxl import load_workbook
    from openpyxl.styles import Font
    wb = load_workbook(xlsx_out); ws = wb["pair_ranking"]
    for row in ws.iter_rows():
        for c_ in row:
            c_.font = Font(name="Arial", size=11, bold=(c_.row == 1))
    for col in ws.columns:
        w = max(len(str(c_.value)) for c_ in col if c_.value is not None) + 2
        ws.column_dimensions[col[0].column_letter].width = w
    wb.save(xlsx_out)
    print(f"Saved ranking to {xlsx_out}")


if __name__ == "__main__":
    main()