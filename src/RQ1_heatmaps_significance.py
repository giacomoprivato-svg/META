#!/usr/bin/env python3
"""
Cortex-only heatmap generator — RAW and Z scores

Creates 2 figures per age group:
- Figure 1: RAW values (5 heatmaps)
- Figure 2: Z scores (5 heatmaps)
Last 2 panels show Euclidean distance masked by significance (p < 0.05) and FDR
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
import nimare.transforms

# -----------------------
# Helpers
# -----------------------
def hierarchical_order(mat, method="ward", metric="euclidean"):
    data = mat.copy()
    if np.isnan(data).any():
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])
    if data.shape[0] <= 1:
        return np.arange(data.shape[0])
    Z = linkage(pdist(data, metric=metric), method=method)
    return leaves_list(Z)


def symmetric_limits(mat):
    vmax = np.nanmax(np.abs(mat))
    return -vmax, vmax


def two_tailed_perm_test(true_vals, null_vals):
    """
    true_vals: (n_tests,)
    null_vals: (n_tests, n_perm)
    Returns: z_vals, p_vals
    """
    n_perm = null_vals.shape[1]
    p_upper = ((null_vals >= true_vals[:, None]).sum(axis=1) + 1) / (n_perm + 1)
    p_lower = ((-null_vals >= -true_vals[:, None]).sum(axis=1) + 1) / (n_perm + 1)
    p_vals = np.minimum(np.minimum(p_upper, p_lower) * 2, 1.0)
    z_vals = nimare.transforms.p_to_z(p_vals, tail="two")
    z_vals[p_lower < p_upper] *= -1
    return z_vals, p_vals


def load_csv_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    return df


# -----------------------
# Main plotting function
# -----------------------
def make_figures(group_dirs, out_prefix, group_label):
    """
    group_dirs: list of directories containing CSVs (RAW, Z, NULLS)
    """

    # ---- Load RAW and Z ----
    raw = {
        "spearman": pd.concat([load_csv_matrix(os.path.join(d, "RAW_cortex_spearman.csv")) for d in group_dirs], axis=0),
        "cosine": pd.concat([load_csv_matrix(os.path.join(d, "RAW_cortex_cosine.csv")) for d in group_dirs], axis=0),
        "euclidean": pd.concat([load_csv_matrix(os.path.join(d, "RAW_cortex_euclidean.csv")) for d in group_dirs], axis=0),
    }

    z = {
        "spearman": pd.concat([load_csv_matrix(os.path.join(d, "Z_cortex_spearman.csv")) for d in group_dirs], axis=0),
        "cosine": pd.concat([load_csv_matrix(os.path.join(d, "Z_cortex_cosine.csv")) for d in group_dirs], axis=0),
        "euclidean": pd.concat([load_csv_matrix(os.path.join(d, "Z_cortex_euclidean.csv")) for d in group_dirs], axis=0),
    }

    # ---- Load NULLS (Euclidean only) ----
    sud_names = raw["euclidean"].columns
    nulls_list = []
    for s in sud_names:
        dfs = [load_csv_matrix(os.path.join(d, f"NULLS_cortex_euclidean_{s}.csv")) for d in group_dirs]
        nulls_list.append(np.concatenate([df.values for df in dfs], axis=0))
    nulls = np.stack(nulls_list, axis=1)  # shape: (n_psy, n_sud, n_perm)

    # ---- Compute significance ----
    true_vals = raw["euclidean"].values
    n_psy, n_sud = true_vals.shape

    p_sig = np.zeros_like(true_vals)
    z_sig = np.zeros_like(true_vals)
    for j in range(n_sud):
        z_vals, p_vals = two_tailed_perm_test(true_vals[:, j], nulls[:, j, :])
        z_sig[:, j] = z_vals
        p_sig[:, j] = p_vals

    sig_mask = p_sig < 0.05

    # FDR correction per SUD
    fdr_mask = np.zeros_like(sig_mask, dtype=bool)
    for j in range(n_sud):
        _, p_fdr = fdrcorrection(p_sig[:, j])
        fdr_mask[:, j] = p_fdr < 0.05

    # ---- Clustering (Z Euclidean) ----
    row_order = hierarchical_order(z["euclidean"].values)
    col_order = hierarchical_order(z["euclidean"].values.T)

    def ord_df(df):
        return df.iloc[row_order, col_order]

    # ---- Plotting function ----
    def plot_figure(mats, sig_mask, fname, is_z):
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(2, 3, figure=fig, wspace=0.25, hspace=0.3)

        panels = [
            ("spearman", "A. Spearman Correlation"),
            ("cosine", "B. Cosine Similarity"),
            ("euclidean", "C. Negative Euclidean Distance"),
        ]

        # Top row: RAW/Z unmasked
        for i, (key, title) in enumerate(panels):
            ax = fig.add_subplot(gs[0, i])
            mat = ord_df(mats[key])
            vmin, vmax = symmetric_limits(mat.values) if is_z else (None, None)
            sns.heatmap(mat, ax=ax, cmap="vlag", vmin=vmin, vmax=vmax,
                        linewidths=0.3, linecolor="lightgray",
                        cbar_kws={"shrink":0.7},
                        xticklabels=mat.columns, yticklabels=mat.index)
            ax.set_title(title)
            ax.set_xlabel("SUD")
            ax.set_ylabel("PSY")
            ax.tick_params(axis="x", rotation=45)

        # Bottom row: Euclidean masked
        for i, (mask, title) in enumerate([(sig_mask, "D. Euclidean (p < 0.05)"),
                                           (fdr_mask, "E. Euclidean (FDR < 0.05)")]):
            ax = fig.add_subplot(gs[1, i])
            mat = ord_df(mats["euclidean"])
            masked_mat = mat.values.copy()
            masked_mat[~mask[row_order][:, col_order]] = np.nan
            vmin, vmax = symmetric_limits(mat.values) if is_z else (None, None)
            sns.heatmap(masked_mat, ax=ax, cmap="vlag", vmin=vmin, vmax=vmax,
                        linewidths=0.3, linecolor="lightgray",
                        cbar_kws={"shrink":0.7},
                        xticklabels=mat.columns, yticklabels=mat.index)
            ax.set_title(title)
            ax.set_xlabel("SUD")
            ax.set_ylabel("PSY")
            ax.tick_params(axis="x", rotation=45)

        value_label = "Z scores" if is_z else "RAW values"
        fig.suptitle(f"{group_label} — Cortex only — {value_label}", fontsize=18)
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")

    # ---- Create figures ----
    plot_figure(raw, sig_mask, f"{out_prefix}_RAW.png", is_z=False)
    plot_figure(z, sig_mask, f"{out_prefix}_Z.png", is_z=True)


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    base = r"C:\Users\giaco\Desktop\Git_META\META\ALL_outputs_RQ1"

    adults_dirs = [os.path.join(base, "adults_all"), os.path.join(base, "adults_ctx")]
    adolescents_dirs = [os.path.join(base, "adolescents_all"), os.path.join(base, "adolescents_ctx")]

    make_figures(adults_dirs, "FIG_adults", "Adults")
    make_figures(adolescents_dirs, "FIG_adolescents", "Adolescents")
