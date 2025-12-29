#!/usr/bin/env python3
"""
Figure 1 generator for PSY × SUD similarity panels (A-E) using adults-only data.
Combined single figure with embedded dendrograms for Panel A.
Manual clusters used for Panels C & D with adjusted colors and arc darkness.
Publication-ready aesthetic color palette applied.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec

# -----------------------
# Helpers
# -----------------------
def ensure_symmetric_cmap(v):
    vmax = np.nanmax(np.abs(v))
    return (-vmax, vmax)

def hierarchical_order(mat, axis=0, method='ward', metric='euclidean'):
    data = mat.copy() if axis == 0 else mat.T.copy()
    if np.isnan(data).any():
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])
    if data.shape[0] <= 1:
        return np.arange(data.shape[0]), None
    Z = linkage(pdist(data, metric=metric), method=method)
    order = leaves_list(Z)
    return order, Z

def arc_between(ax, left_y, right_y, left_x, right_x, linewidth=1.5, color='C0', alpha=0.7):
    verts = [(left_x, left_y),
             ((left_x + right_x)/2.0, max(left_y, right_y) + 0.15),
             (right_x, right_y)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=linewidth, alpha=alpha)
    ax.add_patch(patch)

def bipartite_arc_panel(ax, mat, left_order, right_order, psy_names, sud_names, topK=12, cluster_colors_map=None):
    ax.clear()
    vals = mat.flatten()
    idx_sorted = np.argsort(-vals)
    chosen = idx_sorted[:topK]
    pairs = [(int(idx // mat.shape[1]), int(idx % mat.shape[1])) for idx in chosen]

    nL = len(left_order)
    nR = len(right_order)
    left_pos_map = {i: idx for idx, i in enumerate(left_order)}
    right_pos_map = {i: idx for idx, i in enumerate(right_order)}

    max_val = np.nanmax(mat)

    for i,j in pairs:
        yi = left_pos_map[i] / max(1, nL-1)
        yj = right_pos_map[j] / max(1, nR-1)
        strength = mat[i,j]
        lw = max(0.5, min(4.0, (strength / max_val) * 4.0))
        alpha = 0.3 + 0.7 * (strength/max_val)**2  # enhanced contrast
        color = cluster_colors_map.get(i, 'gray') if cluster_colors_map is not None else 'C0'
        arc_between(ax, yi, yj, left_x=0.05, right_x=0.95, linewidth=lw, color=color, alpha=alpha)

    for idx, i in enumerate(left_order):
        y = idx / max(1, nL-1)
        ax.text(0.0, y, psy_names[i], va='center', ha='left', fontsize=8)
    for idx, i in enumerate(right_order):
        y = idx / max(1, nR-1)
        ax.text(1.0, y, sud_names[i], va='center', ha='right', fontsize=8)

    ax.axis('off')

def forest_panel(ax, mat, psy_names, sud_names):
    means = np.nanmean(mat, axis=1)
    order = np.argsort(-means)
    y = np.arange(mat.shape[0])
    mat_ord = mat[order,:]
    ax.hlines(y=y, xmin=np.nanmin(mat_ord, axis=1), xmax=np.nanmax(mat_ord, axis=1), color='lightgray')
    for k in range(mat_ord.shape[1]):
        ax.scatter(mat_ord[:,k], y, s=40, alpha=0.9)
    ax.scatter(np.nanmean(mat_ord, axis=1), y, s=60, c='k', marker='D')
    ax.set_yticks(y)
    ax.set_yticklabels(np.array(psy_names)[order])
    ax.invert_yaxis()
    ax.set_xlabel('Similarity (Z; combined)')
    return order

# -----------------------
# Main function
# -----------------------
def make_figure(group_dirs, outpath, topK=12):
    # Load data
    Zc_list, subctx_list = [], []
    for gd in group_dirs:
        cortex_csv = os.path.join(gd, 'Z_cortex_combined.csv')
        if os.path.exists(cortex_csv):
            Zc_list.append(pd.read_csv(cortex_csv, index_col=0))
            subctx_csv = os.path.join(gd, 'Z_subctx_combined.csv')
            if os.path.exists(subctx_csv):
                subctx_list.append(pd.read_csv(subctx_csv, index_col=0))

    if not Zc_list:
        raise FileNotFoundError("No cortex CSV files found in the specified group_dirs.")

    Zc_combined = pd.concat(Zc_list, axis=0)
    mat = Zc_combined.to_numpy(dtype=float)
    psy_names = list(Zc_combined.index)
    sud_names = list(Zc_combined.columns)
    mat_z = (mat - np.nanmean(mat.flatten())) / np.nanstd(mat.flatten())

    mat_sub = None
    if subctx_list:
        Zs_combined = pd.concat(subctx_list, axis=0)
        Zs_combined = Zs_combined.reindex(index=Zc_combined.index, columns=Zc_combined.columns)
        mat_sub = Zs_combined.to_numpy(dtype=float)

    # Compute dendrograms
    row_order, Zr = hierarchical_order(mat_z, axis=0)
    col_order, Zc = hierarchical_order(mat_z, axis=1)
    mat_ord = mat_z[np.ix_(row_order, col_order)]
    reordered_psy = [psy_names[i] for i in row_order]
    reordered_sud = [sud_names[i] for i in col_order]

    # Manual clusters
    manual_clusters = {
        'Psychotic': ['SCZ','BD','CHR','Schizotypic'],
        'Neurodevelopmental': ['ASD','ADHD'],
        'AN/OCD': ['AN','OCD'],
        'Mood/Anxiety': ['MDD','PTSD']
    }

    # Cluster colors: Psychotic orange, AN/OCD blue, Neurodevelopmental green, Mood/Anxiety purple
    cluster_colors = {
        'Psychotic': (1.0, 0.6, 0.0),
        'Neurodevelopmental': (0.2,0.8,0.2),
        'AN/OCD': (0.2,0.4,0.8),
        'Mood/Anxiety': (0.7,0.3,0.7)
    }

    # Map PSY disorders to cluster colors
    psy_cluster_map = {}
    for cluster_name, disorders in manual_clusters.items():
        for d in disorders:
            if d in psy_names:
                psy_cluster_map[d] = cluster_colors[cluster_name]

    # --- Figure ---
    plt.close('all')
    fig = plt.figure(figsize=(22,14))
    gs = GridSpec(4,4, figure=fig, width_ratios=[0.2,1,1,1.2], height_ratios=[0.4,1,1,1], wspace=0.6, hspace=0.5)

    # Panel A: heatmap + dendrograms
    axA = fig.add_subplot(gs[0:2,1:])
    sns.heatmap(mat_ord, ax=axA, cmap='vlag', xticklabels=reordered_sud, yticklabels=reordered_psy,
                cbar_kws={'shrink':0.6}, linewidths=0.5, linecolor='lightgray')
    axA.set_title('A. PSY × SUD similarity (cortex only, z-scored)')
    axA.set_xlabel('SUD')
    axA.set_ylabel('PSY')

    # Row dendrogram
    ax_row = axA.inset_axes([-0.2,0,0.15,1])
    dendrogram(Zr, orientation='right', labels=None, ax=ax_row, color_threshold=0)
    ax_row.invert_xaxis()
    ax_row.axis('off')

    # Column dendrogram
    ax_col = axA.inset_axes([0,1.02,1,0.15])
    dendrogram(Zc, orientation='top', labels=None, ax=ax_col, color_threshold=0)
    ax_col.axis('off')

    # Panel B: Forest
    axB = fig.add_subplot(gs[2,0:2])
    order_forest = forest_panel(axB, mat_z, psy_names, sud_names)
    axB.set_title('B. PSY disorder summary (min→max; dots per SUD)')
    psy_order_for_arcs = [order_forest[i] for i in range(len(order_forest))]

    # Panel C: Cluster fingerprints
    axC = fig.add_subplot(gs[2,2:])
    for i, (cluster_name, disorders) in enumerate(manual_clusters.items()):
        inds = [psy_names.index(d) for d in disorders if d in psy_names]
        profile = np.nanmean(mat_z[inds,:], axis=0)
        axC.plot(np.arange(len(sud_names)), profile, marker='o', label='-'.join(disorders),
                 linewidth=2, markeredgecolor='k', color=cluster_colors[cluster_name])
    axC.set_xticks(np.arange(len(sud_names)))
    axC.set_xticklabels(sud_names, rotation=45, ha='right')
    axC.set_ylabel('Similarity (Z)')
    axC.set_xlabel('SUD')
    axC.legend(frameon=False, fontsize='small')
    axC.set_title('C. Cluster-level PSY fingerprints')

    # Panel D: Bipartite arcs
    axD = fig.add_subplot(gs[3,0:2])
    bipartite_arc_panel(axD, mat_z, psy_order_for_arcs, col_order,
                        psy_names, sud_names, topK=topK,
                        cluster_colors_map={i: psy_cluster_map.get(psy_names[i],'gray') for i in range(len(psy_names))})
    axD.set_title(f'D. Top {topK} PSY–SUD links')

    # Panel E: Cortex vs Subcortex
    axE = fig.add_subplot(gs[3,2:])
    if mat_sub is not None and np.any(~np.isnan(mat_sub)):
        mean_cortex = np.nanmean(mat, axis=1)
        mean_sub = np.nanmean(mat_sub, axis=1)
        mask = ~np.isnan(mean_cortex) & ~np.isnan(mean_sub)
        colors = [psy_cluster_map.get(psy_names[i],'gray') for i in range(len(psy_names))]
        axE.scatter(np.array(mean_sub)[mask], np.array(mean_cortex)[mask], s=80, c=np.array(colors)[mask])
        for i in range(len(psy_names)):
            if mask[i]:
                axE.text(mean_sub[i]+0.005, mean_cortex[i], psy_names[i], fontsize=8)
        lims = [min(np.nanmin(mean_sub), np.nanmin(mean_cortex)),
                max(np.nanmax(mean_sub), np.nanmax(mean_cortex))]
        axE.plot(lims, lims, '--', color='gray')
        axE.set_xlabel('Mean subcortex similarity (combined Z)')
        axE.set_ylabel('Mean cortex similarity (combined Z)')
        axE.set_title('E. Cortex vs Subcortex mean similarity')
    else:
        axE.text(0.5,0.5,'Subcortex data missing or invalid.\nPanel E skipped.', ha='center', va='center', fontsize=12)
        axE.set_axis_off()

    fig.suptitle('Figure 1 — PSY × SUD similarity (adults only) — Panels A–E', fontsize=16)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved combined figure to {outpath}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 1 panels from adults-only PSY×SUD similarity outputs.")
    parser.add_argument('--group_dirs', type=str, nargs='+', required=True,
                        help='Paths to group output directories (e.g. adults_all and adults_ctx)')
    parser.add_argument('--out', type=str, default='figure1.png', help='Output figure path (PNG or PDF)')
    parser.add_argument('--topK', type=int, default=12, help='Top-K PSY–SUD pairs to draw in Panel D')
    args = parser.parse_args()
    make_figure(args.group_dirs, args.out, topK=args.topK)





