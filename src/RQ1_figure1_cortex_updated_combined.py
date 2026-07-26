#!/usr/bin/env python3
"""
Figure 2 generator for PSY x SUD similarity panels (A-D)
Adults-only cortex data
SPEARMAN primary version (raw rho; rank correspondence between maps)

WHAT CHANGED vs the Pearson version
-----------------------------------
1. PRIMARY METRIC = Spearman rho (agreed with PI). Euclidean and cosine are
   now the two SENSITIVITY metrics; Pearson no longer appears. RQ1/RQ3 use
   Spearman throughout.

2. Panel A colours cells by RAW Spearman rho (one common scale for all pairs);
   asterisk = pair surviving FDR under BOTH spin and BrainSMASH nulls.

3. Panel B (metric concordance) now plots the PRIMARY (Spearman) against the
   two SENSITIVITY metrics, cosine and -Euclidean. Because cosine (-1..1) and
   -Euclidean (large negative) live on very different scales, they are drawn
   on a twin y-axis (cosine left, -Euclidean right). Each carries an OLS fit
   with a 95% confidence band. r is Pearson r between the two raw metric
   matrices; p is a Mantel (label-permutation) p that respects the
   non-independence of the 63 pairs (each disorder contributes 7).

4. Panel C (cortex vs subcortex) now carries a 95% confidence band on the fit.

5. Ranking: the 63 PSY-SUD pairs are ranked by cortex Spearman and by
   subcortex Spearman and written side by side into one Excel sheet
   (RQ1_pair_ranking_spearman.xlsx).

Spin p-values are the CORRECTED-spin values (idxR + 34) produced by the
rq1_common pipeline; the RAW similarity values are spin-independent.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm

# ---------------------------
# Paths (repo-relative)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

sns.set(style="whitegrid")

# ---------------------------
# Metric / stats options
# ---------------------------
METRIC        = "spearman"          # primary metric
SENSITIVITY   = ["cosine", "euclidean"]   # sensitivity metrics (raw)
METRIC_LABEL  = "Spearman \u03c1"
PANEL_B_P     = "mantel"            # "mantel" or "parametric"
N_MANTEL      = 10000
CI_ALPHA      = 0.05               # 95% confidence band
np.random.seed(42)

ROW_ORDER_MODE = "optimal_leaf"    # "seriation" | "optimal_leaf" | "hierarchical"

# ---------------------------
# Font sizes
# ---------------------------
FONT_TICKS = 26
FONT_AST   = 34

SENS_COLORS = {"cosine": "#e07270", "euclidean": "#2f9e44"}
SENS_LABEL  = {"cosine": "Cosine", "euclidean": "\u2212Euclidean"}


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


def mantel_p(A_vals, B_df, n=N_MANTEL):
    """Label-permutation p for the correlation between two PSY x SUD matrices."""
    a = np.asarray(A_vals).flatten()
    b = B_df.values.flatten()
    ok = np.isfinite(a) & np.isfinite(b)
    r_obs = pearsonr(a[ok], b[ok])[0]
    rng = np.random.default_rng(42)
    null = np.empty(n)
    for k in range(n):
        Bp = B_df.values[np.ix_(rng.permutation(B_df.shape[0]),
                                rng.permutation(B_df.shape[1]))].flatten()
        m = np.isfinite(a) & np.isfinite(Bp)
        null[k] = pearsonr(a[m], Bp[m])[0]
    return r_obs, (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n + 1)


def scatter_fit_ci(ax, x, y, color, label, ci_alpha=CI_ALPHA):
    """Scatter + OLS fit + 95% confidence band on a given axis. Returns fit x/y for legend."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ax.scatter(x, y, marker="o", color=color, s=50, alpha=0.6, label=label)
    Xd = sm.add_constant(x)
    model = sm.OLS(y, Xd).fit()
    xg = np.linspace(x.min(), x.max(), 100)
    pred = model.get_prediction(sm.add_constant(xg))
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=ci_alpha)
    ax.plot(xg, mean, color=color, lw=2, alpha=0.9)
    ax.fill_between(xg, ci[:, 0], ci[:, 1], color=color, alpha=0.15, lw=0)


# -----------------------
# Main
# -----------------------
def make_figure(group_dirs, outpath):

    prefix = "RAW"

    def concat_metric(metric):
        return pd.concat(
            [pd.read_csv(os.path.join(gd, f"{prefix}_cortex_{metric}.csv"), index_col=0)
             for gd in group_dirs], axis=0)

    sim_primary = concat_metric(METRIC)
    sens = {m: concat_metric(m) for m in SENSITIVITY}

    # ---- Remove Schizotypy everywhere ----
    mask_psy = ~sim_primary.index.str.contains("Schizotyp", case=False)
    sim_primary = sim_primary.loc[mask_psy]
    for m in SENSITIVITY:
        sens[m] = sens[m].loc[mask_psy]

    psy_names = list(sim_primary.index)
    sud_names = list(sim_primary.columns)
    sim_primary_orig = sim_primary.copy()

    # ---- p-values: spin (corrected) + BrainSMASH ----
    def read_csv_safely(path):
        df = pd.read_csv(path, index_col=0, sep=None, engine="python")
        return df.apply(pd.to_numeric, errors="coerce")

    PVAL = pd.concat(
        [read_csv_safely(os.path.join(gd, f"PVAL_cortex_{METRIC}.csv")) for gd in group_dirs],
        axis=0).loc[mask_psy]
    PVAL_bs = pd.concat(
        [read_csv_safely(os.path.join(gd, f"PVAL_cortex_{METRIC}_brainsmash.csv")) for gd in group_dirs],
        axis=0).loc[mask_psy]

    PVAL = PVAL.reindex(index=sim_primary.index, columns=sim_primary.columns)
    PVAL_bs = PVAL_bs.reindex(index=sim_primary.index, columns=sim_primary.columns)

    # ---- FDR per column under each null, then INTERSECTION ----
    mask_spin = np.zeros_like(PVAL, dtype=bool)
    mask_bs = np.zeros_like(PVAL_bs, dtype=bool)
    for j in range(PVAL.shape[1]):
        _, pf_s = fdrcorrection(PVAL.iloc[:, j]); mask_spin[:, j] = pf_s < 0.05
        _, pf_b = fdrcorrection(PVAL_bs.iloc[:, j]); mask_bs[:, j] = pf_b < 0.05
    fdr_mask = mask_spin & mask_bs
    print(f"spin only: {mask_spin.sum()} | bsmash only: {mask_bs.sum()} | BOTH (starred): {fdr_mask.sum()}")

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
    row_means_ord = tmp.mean(axis=1).values
    if row_means_ord[0] < row_means_ord[-1]:
        row_order = row_order[::-1]
    col_means_ord = tmp.mean(axis=0).values
    if col_means_ord[0] < col_means_ord[-1]:
        col_order = col_order[::-1]

    sim_primary_ord = sim_primary.iloc[row_order, col_order]
    fdr_mask_ord = fdr_mask[row_order][:, col_order]

    # ---- RAW cortex/subctx (adults_all) for Panel C ----
    raw_cortex = pd.read_csv(
        os.path.join(repo_dir, "ALL_outputs_RQ1", "adults_all", f"RAW_cortex_{METRIC}.csv"),
        index_col=0).reindex(psy_names)
    raw_subctx = pd.read_csv(
        os.path.join(repo_dir, "ALL_outputs_RQ1", "adults_all", f"RAW_subctx_{METRIC}.csv"),
        index_col=0).reindex(psy_names)

    # =====================================================
    # FIGURE LAYOUT
    # =====================================================
    fig = plt.figure(figsize=(26, 14))
    axA = fig.add_axes([0.05, 0.1, 0.5, 0.8])
    axB = fig.add_axes([0.60, 0.55, 0.16, 0.35])
    axC = fig.add_axes([0.80, 0.55, 0.16, 0.35])
    axD = fig.add_axes([0.60, 0.1, 0.36, 0.35])

    # ---------------- PANEL A ----------------
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

    # ---------------- PANEL B (concordance, twin axis + CI) ----------------
    x = sim_primary.values.flatten()
    # cosine on the primary (left) y-axis
    scatter_fit_ci(axB, x, sens["cosine"].values.flatten(),
                   SENS_COLORS["cosine"], SENS_LABEL["cosine"])
    axB.set_ylabel("Cosine similarity", fontsize=15, color=SENS_COLORS["cosine"])
    axB.tick_params(axis="y", labelcolor=SENS_COLORS["cosine"])
    # -Euclidean on twin (right) y-axis
    axB2 = axB.twinx()
    scatter_fit_ci(axB2, x, sens["euclidean"].values.flatten(),
                   SENS_COLORS["euclidean"], SENS_LABEL["euclidean"])
    axB2.set_ylabel("\u2212Euclidean similarity", fontsize=15, color=SENS_COLORS["euclidean"])
    axB2.tick_params(axis="y", labelcolor=SENS_COLORS["euclidean"])
    axB2.grid(False)

    # r / p annotations
    if PANEL_B_P == "mantel":
        r_co, p_co = mantel_p(sim_primary.values, sens["cosine"]); p_tag = "$p_{Mantel}$"
        r_eu, p_eu = mantel_p(sim_primary.values, sens["euclidean"])
    else:
        r_co, p_co = pearsonr(x, sens["cosine"].values.flatten()); p_tag = "$p$"
        r_eu, p_eu = pearsonr(x, sens["euclidean"].values.flatten())
    axB.text(0.03, 0.97,
             f"Cosine:      r = {r_co:.3f}, {p_tag} = {p_co:.4f}\n"
             f"\u2212Euclid:   r = {r_eu:.3f}, {p_tag} = {p_eu:.4f}",
             transform=axB.transAxes, ha="left", va="top", fontsize=13,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axB.set_xlabel(f"Cortex {METRIC_LABEL}", fontsize=15)
    axB.set_title("B. Concordance with sensitivity metrics", fontsize=18)
    axB.grid(False)

    # ---------------- PANEL C (cortex vs subctx + CI) ----------------
    manual_clusters = {"Psychotic": ["SCZ", "BD", "CHR"],
                       "Neurodevelopmental": ["ASD", "ADHD"],
                       "AN/OCD": ["AN", "OCD"],
                       "Mood/Anxiety": ["MDD", "PTSD"]}
    cluster_colors = {"Psychotic": (1.0, 0.6, 0.0),
                      "Neurodevelopmental": (0.2, 0.8, 0.2),
                      "AN/OCD": (0.2, 0.4, 0.8),
                      "Mood/Anxiety": (0.7, 0.3, 0.7)}
    cluster_map = {d: c for c, ds in manual_clusters.items() for d in ds}

    xc = raw_cortex.values.flatten()
    yc = raw_subctx.values.flatten()
    labs = np.repeat(raw_cortex.index.values, raw_cortex.shape[1])
    for cname, color in cluster_colors.items():
        idx = [i for i, d in enumerate(labs) if cluster_map.get(d) == cname]
        if idx:
            axC.scatter(xc[idx], yc[idx], color=color, s=50, alpha=0.75, label=cname)
    ok = np.isfinite(xc) & np.isfinite(yc)
    Xd = sm.add_constant(xc[ok]); model = sm.OLS(yc[ok], Xd).fit()
    xg = np.linspace(np.nanmin(xc), np.nanmax(xc), 100)
    pred = model.get_prediction(sm.add_constant(xg))
    axC.plot(xg, pred.predicted_mean, color="gray", lw=2, alpha=0.9)
    ci = pred.conf_int(alpha=CI_ALPHA)
    axC.fill_between(xg, ci[:, 0], ci[:, 1], color="gray", alpha=0.15, lw=0)
    if PANEL_B_P == "mantel":
        r_cs, p_cs = mantel_p(raw_cortex.values, raw_subctx); tag = "$p_{Mantel}$"
    else:
        r_cs, p_cs = pearsonr(xc[ok], yc[ok]); tag = "$p$"
    axC.text(0.03, 0.97, f"r = {r_cs:.3f}\n{tag} = {p_cs:.4f}",
             transform=axC.transAxes, ha="left", va="top", fontsize=13,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axC.set_xlabel(f"Cortex {METRIC_LABEL} (raw)", fontsize=15)
    axC.set_ylabel(f"Subcortex {METRIC_LABEL} (raw)", fontsize=15)
    axC.set_title("C. Correlation between cortex and subcortex", fontsize=18)
    axC.legend(frameon=False, loc="lower right", fontsize=11)
    axC.grid(False)

    # ---------------- PANEL D (cluster fingerprints) ----------------
    for cname, disorders in manual_clusters.items():
        inds = [psy_names.index(d) for d in disorders if d in psy_names]
        if inds:
            prof = np.nanmean(sim_primary_orig.values[inds, :], axis=0)
            axD.plot(range(len(sud_names)), prof, marker="o", lw=2,
                     color=cluster_colors[cname], label=cname)
    axD.axhline(0, color="0.6", lw=1, ls="--", zorder=0)
    axD.set_xticks(range(len(sud_names)))
    axD.set_xticklabels(sud_names, ha="right", fontsize=18)
    axD.set_ylabel(METRIC_LABEL, fontsize=18)
    axD.legend(frameon=False, loc="lower right", fontsize=14)
    axD.set_title("D. PSY cluster fingerprints", fontsize=18)
    axD.grid(False)

    # ---------------- standalone colorbar ----------------
    import matplotlib as mpl
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

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.savefig(os.path.splitext(outpath)[0] + ".pdf", bbox_inches="tight")
    print(f"Saved figure to {outpath}")

    # =====================================================
    # PAIR RANKING -> single Excel sheet (cortex + subcortex columns)
    # =====================================================
    def ranked_pairs(mat_df):
        long = (mat_df.reset_index()
                .melt(id_vars="index", var_name="SUD", value_name="rho")
                .rename(columns={"index": "PSY"}))
        long["pair"] = long["PSY"] + "\u2013" + long["SUD"]
        long = long.sort_values("rho", ascending=False).reset_index(drop=True)
        return long

    rc = ranked_pairs(raw_cortex)
    rs = ranked_pairs(raw_subctx)
    rank_df = pd.DataFrame({
        "Rank": np.arange(1, len(rc) + 1),
        "Cortex_pair": rc["pair"].values,
        f"Cortex_{METRIC}": rc["rho"].round(4).values,
        "Subctx_pair": rs["pair"].values,
        f"Subctx_{METRIC}": rs["rho"].round(4).values,
    })
    xlsx_out = os.path.join(os.path.dirname(outpath), "RQ1_pair_ranking_spearman.xlsx")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as xl:
        rank_df.to_excel(xl, sheet_name="pair_ranking", index=False)
    # professional font
    from openpyxl import load_workbook
    from openpyxl.styles import Font
    wb = load_workbook(xlsx_out); ws = wb["pair_ranking"]
    for row in ws.iter_rows():
        for c in row:
            c.font = Font(name="Arial", size=11, bold=(c.row == 1))
    for col in ws.columns:
        w = max(len(str(c.value)) for c in col if c.value is not None) + 2
        ws.column_dimensions[col[0].column_letter].width = w
    wb.save(xlsx_out)
    print(f"Saved ranking to {xlsx_out}")


if __name__ == "__main__":
    base = os.path.join(repo_dir, "ALL_outputs_RQ1")
    adults_dirs = [os.path.join(base, "adults_all"), os.path.join(base, "adults_ctx")]
    make_figure(adults_dirs, os.path.join(repo_dir, "figures", "Figure2_Spearman_final_combined.png"))