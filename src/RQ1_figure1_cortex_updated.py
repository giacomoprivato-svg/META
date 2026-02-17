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
from adjustText import adjust_text 

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

    # ---- Remove Schizotypy ----
    mask_psy = ~Z_euclid.index.str.contains('Schizotypic|Schizotypy', case=False)
    Z_euclid = Z_euclid.loc[mask_psy]
    Z_spearman = Z_spearman.loc[mask_psy]
    Z_cosine = Z_cosine.loc[mask_psy]

    psy_names = list(Z_euclid.index)
    sud_names = list(Z_euclid.columns)

    # ---- Save original for fingerprint ----
    Z_euclid_orig = Z_euclid.copy()

    # ---- Load p-values ----
    PVAL_list = []
    for gd in group_dirs:
        PVAL_list.append(pd.read_csv(os.path.join(gd, "PVAL_cortex_euclidean.csv"), index_col=0))
    PVAL = pd.concat(PVAL_list, axis=0).loc[mask_psy]

    # ---- FDR correction ----
    fdr_mask = np.zeros_like(PVAL, dtype=bool)
    for j in range(PVAL.shape[1]):
        _, p_fdr = fdrcorrection(PVAL.iloc[:, j])
        fdr_mask[:, j] = p_fdr < 0.05

    # ---- Clustering for heatmap ----
    row_order = hierarchical_order(Z_euclid.values)
    col_order = hierarchical_order(Z_euclid.values.T)

    Z_euclid_ord = Z_euclid.iloc[row_order, col_order]
    Z_spearman_ord = Z_spearman.iloc[row_order, col_order]
    Z_cosine_ord = Z_cosine.iloc[row_order, col_order]
    fdr_mask_ord = fdr_mask[row_order][:, col_order]

    # ---- RAW cortex/subcortex for Panel D ----
    raw_cortex = pd.read_csv(
        r"C:\Users\giaco\Desktop\Git_META\META\ALL_outputs_RQ1\adults_all\RAW_cortex_euclidean.csv",
        index_col=0
    ).loc[psy_names]

    raw_subctx = pd.read_csv(
        r"C:\Users\giaco\Desktop\Git_META\META\ALL_outputs_RQ1\adults_all\RAW_subctx_euclidean.csv",
        index_col=0
    ).loc[psy_names]

    mean_cortex = raw_cortex.mean(axis=1)
    mean_subctx = raw_subctx.mean(axis=1)

    # ------------------------------------------------
    # FIGURE LAYOUT
    # ------------------------------------------------
    fig = plt.figure(figsize=(26, 14))

    # LEFT: heatmap
    axA = fig.add_axes([0.05, 0.1, 0.5, 0.8])

    # RIGHT TOP: B and D
    axB = fig.add_axes([0.55, 0.55, 0.18, 0.35])
    axD = fig.add_axes([0.77, 0.55, 0.18, 0.35])

    # RIGHT BOTTOM: C
    axC = fig.add_axes([0.55, 0.1, 0.4, 0.35])

    # ------------------------------------------------
    # PANEL A — Heatmap with asterisks FDR
    # ------------------------------------------------
    sns.heatmap(Z_euclid_ord, ax=axA, cmap='RdBu_r', center=0,
                xticklabels=True, yticklabels=True,
                vmin=Z_euclid_ord.values.min(), vmax=6,  # massimo colorbar a +6
                cbar_kws={'label': 'Euclidean similarity (Z)', 'shrink':0.5,'pad': 0.01})  # colorbar più stretta

    # Asterischi più grandi
    for i in range(Z_euclid_ord.shape[0]):
        for j in range(Z_euclid_ord.shape[1]):
            if fdr_mask_ord[i, j]:
                axA.text(j + 0.5, i + 0.5, '*',
                        ha='center', va='center',
                        color='black', fontsize=22, fontweight='bold')

    axA.set_title("A. PSY × SUD Euclidean similarity (Z)", fontsize=16)
    axA.grid(False)
    axA.set_aspect('equal')  # rende la heatmap più quadrata

    # ------------------------------------------------
    # PANEL B — Euclidean vs Spearman/Cosine per tutte le coppie PSY × SUD
    # ------------------------------------------------
    # Manual clusters per PSY disorders
    manual_clusters = {
        'Psychotic': ['SCZ','BD','CHR'],
        'Neurodevelopmental': ['ASD','ADHD'],
        'AN/OCD': ['AN','OCD'],
        'Mood/Anxiety': ['MDD','PTSD']
    }

    # Colori dei cluster
    cluster_colors = {
        'Psychotic': (1.0, 0.6, 0.0),
        'Neurodevelopmental': (0.2, 0.8, 0.2),
        'AN/OCD': (0.2, 0.4, 0.8),
        'Mood/Anxiety': (0.7, 0.3, 0.7)
    }
    # Prendi tutte le coppie PSY × SUD
    x_vals_B = Z_euclid.values.flatten()
    y_vals_spearman = Z_spearman.values.flatten()
    y_vals_cosine = Z_cosine.values.flatten()

    # Scatter plot con punti tondi e trasparenti
    axB.scatter(x_vals_B, y_vals_spearman, marker='o', color="#469fdf", s=50, alpha=0.6, label='Spearman')
    axB.scatter(x_vals_B, y_vals_cosine, marker='o', color="#e07270", s=50, alpha=0.6, label='Cosine')

    # Regression line descrittiva leggermente più scura
    coeff_spearman = np.polyfit(x_vals_B, y_vals_spearman, 1)
    x_fit = np.linspace(x_vals_B.min(), x_vals_B.max(), 100)
    axB.plot(x_fit, coeff_spearman[0]*x_fit + coeff_spearman[1], color="#469fdf", lw=2, alpha=0.8)

    coeff_cosine = np.polyfit(x_vals_B, y_vals_cosine, 1)
    axB.plot(x_fit, coeff_cosine[0]*x_fit + coeff_cosine[1], color="#e07270", lw=2, alpha=0.8)

    axB.set_xlabel("Cortex Euclidean similarity (Z)")
    axB.set_ylabel("Cortex Spearman correlation/ Cosine similarity (Z)")
    axB.set_title("B. Correlation between similarity measures", fontsize=14)
    axB.legend(frameon=False)
    axB.grid(False)

    # ------------------------------------------------
    # PANEL C — Cortex vs Subcortex RAW per tutte le coppie PSY × SUD
    # ------------------------------------------------
    x_vals = raw_cortex.values.flatten()
    y_vals = raw_subctx.values.flatten()
    psy_labels = np.repeat(raw_cortex.index.values, raw_cortex.shape[1])

    # Colore in base al cluster
    cluster_colors = {
        'Psychotic': (1.0, 0.6, 0.0),
        'Neurodevelopmental': (0.2, 0.8, 0.2),
        'AN/OCD': (0.2, 0.4, 0.8),
        'Mood/Anxiety': (0.7, 0.3, 0.7)
    }
    cluster_map = {}
    for cname, disorders in manual_clusters.items():
        for d in disorders:
            if d in psy_names:
                cluster_map[d] = cname
    colors = [cluster_colors.get(cluster_map.get(d, None), "black") for d in psy_labels]

    # Scatter plot con legenda
    for cname, color in cluster_colors.items():
        inds = [i for i, d in enumerate(psy_labels) if cluster_map.get(d) == cname]
        axD.scatter(x_vals[inds], y_vals[inds], color=color, s=50, alpha=0.7, label=cname)

    # Regression line descrittiva
    coeffs = np.polyfit(x_vals, y_vals, 1)
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
    axD.plot(x_fit, coeffs[0]*x_fit + coeffs[1], color='gray', lw=2, alpha=0.8)

    # Axes
    axD.set_xlabel("Cortex Euclidean similarity (raw)")
    axD.set_ylabel("Subcortex Euclidean similarity (raw)")
    axD.set_title("C. Correlation between cortex and subcortex", fontsize=14)
    axD.set_xlim(x_vals.min() - 0.05, x_vals.max() + 0.05)
    axD.set_ylim(y_vals.min() - 0.05, y_vals.max() + 0.05)
    axD.grid(False)
    axD.legend(frameon=False)



    # ------------------------------------------------
    # PANEL D — Cluster fingerprints (original)
    # ------------------------------------------------
    for cname, disorders in manual_clusters.items():
        inds = [psy_names.index(d) for d in disorders if d in psy_names]
        if inds:
            prof = np.nanmean(Z_euclid_orig.values[inds, :], axis=0)
            axC.plot(range(len(sud_names)), prof,
                     marker='o', lw=2,
                     color=cluster_colors[cname],
                     label=cname)

    axC.set_xticks(range(len(sud_names)))
    axC.set_xticklabels(sud_names, ha='right')
    axC.set_ylabel('Euclidean similarity (Z)')
    axC.legend(
    frameon=False,
    loc='lower right',              # posizione relativa al bbox
    bbox_to_anchor=(1, 0),          # x=1 (destra), y=0 (sotto)
    fontsize=12)                    # dimensione del testo della legenda
    axC.set_title('D. PSY cluster fingerprints', fontsize=16)
    axC.grid(False)

    # ------------------------------------------------
    # Save figure
    # ------------------------------------------------
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

