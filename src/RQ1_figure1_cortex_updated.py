#!/usr/bin/env python3
"""
Figure 1 generator for PSY × SUD similarity panels (A–D)
Adults-only cortex data
EUCLIDEAN similarity version (permutation-calibrated Z-values)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection

sns.set(style="whitegrid")
SUD_COLORS = sns.color_palette("tab10", 10)

# -----------------------
# Helpers
# -----------------------
def hierarchical_order(mat, axis=0, method='ward', metric='euclidean'):
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
    """Cosmetic relabel for figure only"""
    labels = [t.get_text() for t in ax.get_yticklabels()]
    labels = ['Schizotypy' if l == 'Schizotypic' else l for l in labels]
    ax.set_yticklabels(labels, rotation=0)

# -----------------------
# Main
# -----------------------
def make_figure(group_dirs, outpath):

    # ---- Load Z matrices ----
    Z_euclid_list, Z_spearman_list, Z_cosine_list = [], [], []
    for gd in group_dirs:
        Z_euclid_list.append(pd.read_csv(os.path.join(gd, "Z_cortex_euclidean.csv"), index_col=0))
        Z_spearman_list.append(pd.read_csv(os.path.join(gd, "Z_cortex_spearman.csv"), index_col=0))
        Z_cosine_list.append(pd.read_csv(os.path.join(gd, "Z_cortex_cosine.csv"), index_col=0))

    Z_euclid = pd.concat(Z_euclid_list, axis=0)
    Z_spearman = pd.concat(Z_spearman_list, axis=0)
    Z_cosine = pd.concat(Z_cosine_list, axis=0)

    # ---- Remove Schizotypy from all analyses ----
    mask_psy = ~Z_euclid.index.str.contains('Schizotypic|Schizotypy', case=False)
    Z_euclid = Z_euclid.loc[mask_psy, :]
    Z_spearman = Z_spearman.loc[mask_psy, :]
    Z_cosine = Z_cosine.loc[mask_psy, :]
    psy_names = list(Z_euclid.index)
    sud_names = list(Z_euclid.columns)

    # ---- Load p-values for Euclidean ----
    PVAL_list = []
    for gd in group_dirs:
        PVAL_list.append(pd.read_csv(os.path.join(gd, "PVAL_cortex_euclidean.csv"), index_col=0))
    PVAL = pd.concat(PVAL_list, axis=0)
    PVAL = PVAL.loc[mask_psy, :]

    # ---- FDR mask ----
    fdr_mask = np.zeros_like(PVAL, dtype=bool)
    for j in range(PVAL.shape[1]):
        _, p_fdr = fdrcorrection(PVAL.iloc[:, j])
        fdr_mask[:, j] = p_fdr < 0.05

    # ---- Clustering ----
    row_order = hierarchical_order(Z_euclid.values)
    col_order = hierarchical_order(Z_euclid.values.T)

    def ord_df(df):
        return df.iloc[row_order, col_order]

    Z_euclid_ord = ord_df(Z_euclid)
    Z_spearman_ord = ord_df(Z_spearman)
    Z_cosine_ord = ord_df(Z_cosine)
    fdr_mask_ord = fdr_mask[row_order][:, col_order]

    # ---- Manual clusters for panel C ----
    manual_clusters = {
        'Psychotic': ['SCZ','BD','CHR'],
        'Neurodevelopmental': ['ASD','ADHD'],
        'AN/OCD': ['AN','OCD'],
        'Mood/Anxiety': ['MDD','PTSD']
    }
    cluster_colors = {
        'Psychotic': (1.0, 0.6, 0.0),
        'Neurodevelopmental': (0.2, 0.8, 0.2),
        'AN/OCD': (0.2, 0.4, 0.8),
        'Mood/Anxiety': (0.7, 0.3, 0.7)
    }

    # Figure layout: Panel A grande + B,C,D piccoli sotto
    fig = plt.figure(figsize=(28, 18))
    # height_ratios = [3, 1] → riga A più alta, riga B,C,D mantiene altezza originale
    gs = GridSpec(2, 9, figure=fig, height_ratios=[4,2], hspace=0, wspace=0.5)

    # Panel A: heatmap
    axA = fig.add_subplot(gs[0, :])
    sns.heatmap(Z_euclid_ord, ax=axA, cmap='RdBu_r', center=0,
                xticklabels=Z_euclid_ord.columns, yticklabels=Z_euclid_ord.index,
                cbar_kws={'label': 'Euclidean similarity (Z)',
                        'orientation': 'horizontal', 'shrink':0.4, 'pad': 0.05})
    axA.set_title('A. PSY × SUD Euclidean similarity (Z)', fontsize=16)
    cosmetic_relabel(axA)
    axA.grid(False)

    # Panel B: Euclidean FDR < 0.05 sotto A, senza colormap
    axB = fig.add_subplot(gs[1, 0:3])
    masked_mat = Z_euclid_ord.copy()
    masked_mat[~fdr_mask_ord] = np.nan
    sns.heatmap(masked_mat, ax=axB, cmap='RdBu_r', center=0, 
                xticklabels=Z_euclid_ord.columns, yticklabels=Z_euclid_ord.index,
                cbar=False)  # niente colorbar
    axB.set_title('B. Euclidean similarity (FDR < 0.05)', fontsize=16)
    cosmetic_relabel(axB)
    axB.grid(False)

    # Panel C: cluster fingerprints
    axC = fig.add_subplot(gs[1, 3:6])
    for cname, disorders in manual_clusters.items():
        inds = [psy_names.index(d) for d in disorders if d in psy_names]
        if inds:
            prof = np.nanmean(Z_euclid.values[inds, :], axis=0)
            axC.plot(range(len(sud_names)), prof, marker='o', lw=2,
                    color=cluster_colors[cname], label=cname)
    axC.set_xticks(range(len(sud_names)))
    axC.set_xticklabels(sud_names, rotation=45, ha='right')
    axC.set_ylabel('Euclidean similarity (Z)')
    axC.legend(frameon=False)
    axC.set_title('C. PSY cluster fingerprints', fontsize=16)
    axC.grid(False)
    cosmetic_relabel(axC)

    # Panel D: scatter Euclidean vs Spearman/Cosine con colori neutrali (non quelli dei cluster)
    axD = fig.add_subplot(gs[1, 6:9])
    mean_euclid = np.nanmean(Z_euclid.values, axis=1)
    mean_spearman = np.nanmean(Z_spearman.values, axis=1)
    mean_cosine = np.nanmean(Z_cosine.values, axis=1)
    axD.scatter(mean_euclid, mean_spearman, label='Spearman Z', marker='s', color="#469fdf", alpha=0.8, s=80)
    axD.scatter(mean_euclid, mean_cosine, label='Cosine Z', marker='^', color="#e07270", alpha=0.8, s=80)
    axD.set_xlabel('Mean Euclidean Z')
    axD.set_ylabel('Mean correlation Z')
    axD.set_title('D. Euclidean vs Spearman/Cosine Z', fontsize=16)
    axD.legend(frameon=False)
    lims = [min(axD.get_xlim()[0], axD.get_ylim()[0]), max(axD.get_xlim()[1], axD.get_ylim()[1])]
    axD.plot(lims, lims, ls='--', color='lightgray')
    axD.set_xlim(lims)
    axD.set_ylim(lims)
    axD.grid(False)


    # ---- Save figure ----
    plt.subplots_adjust(top=0.94)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outpath}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    base = r"C:\Users\giaco\Desktop\Git_META\META\ALL_outputs_RQ1"
    adults_dirs = [
        os.path.join(base, "adults_all"),
        os.path.join(base, "adults_ctx")
    ]

    make_figure(adults_dirs, "Figure1_Euclidean_final.png")

