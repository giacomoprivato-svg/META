#!/usr/bin/env python3
"""
Figure 1 generator for PSY × SUD similarity panels (A–E) using adults-only data.
EUCLIDEAN similarity version (permutation-calibrated z-values).

Identical to combined-z version except:
- Loads Z_cortex_euclidean.csv / Z_subctx_euclidean.csv
- No additional z-scoring applied
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# -----------------------
# Helpers
# -----------------------

SUD_COLORS = sns.color_palette("tab10", 10)

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


def bipartite_lollipop_panel(
    ax, mat, left_order, right_order,
    psy_names, sud_names,
    cluster_colors_map,
    top_fraction=0.2,
    max_circle_size=300
):
    ax.clear()
    nL, nR = len(left_order), len(right_order)

    vals = mat.flatten()
    n_top = max(1, int(len(vals) * top_fraction))
    top_inds = np.argsort(-vals)[:n_top]
    top_pairs = [(int(i // nR), int(i % nR)) for i in top_inds]

    top_vals = np.array([mat[i, j] for i, j in top_pairs])
    min_val, max_val = top_vals.min(), top_vals.max()
    norm_vals = (top_vals - min_val) / (max_val - min_val + 1e-8)

    left_pos = {idx: pos for pos, idx in enumerate(left_order)}
    right_pos = {idx: pos for pos, idx in enumerate(right_order)}

    for k, (i, j) in enumerate(top_pairs):
        yi = left_pos[i] / max(1, nL - 1)
        xi = right_pos[j] / max(1, nR - 1)
        size = 50 + max_circle_size * norm_vals[k]
        color = cluster_colors_map.get(i, 'gray')
        ax.plot([0, xi], [yi, yi], lw=1, color=color, alpha=0.6)
        ax.scatter(xi, yi, s=size, color=color, edgecolor='k', zorder=3)

    ax.set_yticks([i / max(1, nL - 1) for i in range(nL)])
    ax.set_yticklabels([psy_names[i] for i in left_order], fontsize=9)
    ax.set_xticks([i / max(1, nR - 1) for i in range(nR)])
    ax.set_xticklabels([sud_names[i] for i in right_order], rotation=45, ha='right', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('SUD')
    ax.set_title(f'Top {int(top_fraction*100)}% PSY–SUD links')


def forest_panel(ax, mat, psy_names, sud_names):
    means = np.nanmean(mat, axis=1)
    order = np.argsort(-means)
    y = np.arange(mat.shape[0])
    mat_ord = mat[order, :]

    ax.hlines(
        y=y,
        xmin=np.nanmin(mat_ord, axis=1),
        xmax=np.nanmax(mat_ord, axis=1),
        color='lightgray',
        linewidth=2
    )

    for k, sud in enumerate(sud_names):
        ax.scatter(mat_ord[:, k], y, s=40,
                   color=SUD_COLORS[k], alpha=0.9, label=sud)

    ax.scatter(np.nanmean(mat_ord, axis=1), y,
               s=60, c='k', marker='D', label='Mean')

    ax.set_yticks(y)
    ax.set_yticklabels(np.array(psy_names)[order])
    ax.invert_yaxis()
    ax.set_xlabel('Euclidean similarity (permutation Z)')
    ax.legend(frameon=False, fontsize='small', loc='upper left')
    return order


# -----------------------
# Main
# -----------------------

def make_figure(group_dirs, outpath, topK=12):

    Zc_list, subctx_list = [], []

    for gd in group_dirs:
        cortex_csv = os.path.join(gd, 'Z_cortex_euclidean.csv')
        if os.path.exists(cortex_csv):
            Zc_list.append(pd.read_csv(cortex_csv, index_col=0))

            subctx_csv = os.path.join(gd, 'Z_subctx_euclidean.csv')
            if os.path.exists(subctx_csv):
                subctx_list.append(pd.read_csv(subctx_csv, index_col=0))

    if not Zc_list:
        raise FileNotFoundError("No euclidean cortex CSVs found.")

    Zc_combined = pd.concat(Zc_list, axis=0)
    mat = Zc_combined.to_numpy(float)

    psy_names = ['Schizotypy' if n.lower() == 'schizotypic' else n
                 for n in Zc_combined.index]
    sud_names = list(Zc_combined.columns)

    mat_z = mat  # no re-zscoring

    mat_sub = None
    if subctx_list:
        Zs = pd.concat(subctx_list, axis=0)
        Zs = Zs.reindex(index=Zc_combined.index,
                        columns=Zc_combined.columns)
        mat_sub = Zs.to_numpy(float)

    row_order, Zr = hierarchical_order(mat_z, axis=0)
    col_order, Zc = hierarchical_order(mat_z, axis=1)
    mat_ord = mat_z[np.ix_(row_order, col_order)]

    reordered_psy = [psy_names[i] for i in row_order]
    reordered_sud = [sud_names[i] for i in col_order]

    manual_clusters = {
        'Psychotic': ['SCZ','BD','CHR','Schizotypic'],
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

    psy_cluster_map = {}
    for cname, disorders in manual_clusters.items():
        for d in disorders:
            key = 'Schizotypy' if d.lower() == 'schizotypic' else d
            if key in psy_names:
                psy_cluster_map[key] = cluster_colors[cname]

    fig = plt.figure(figsize=(22,14))
    gs = GridSpec(4,4, figure=fig,
                  width_ratios=[0.2,1,1,1.2],
                  height_ratios=[0.4,1,1,1],
                  wspace=0.6, hspace=0.5)

    # ---- Panel A ----
    axA = fig.add_subplot(gs[0:2,1:])
    sns.heatmap(mat_ord, ax=axA, cmap='cividis',
                xticklabels=reordered_sud,
                yticklabels=reordered_psy,
                cbar_kws={'shrink':0.6})
    axA.set_title('A. PSY × SUD similarity (Euclidean, cortex)')

    ax_row = axA.inset_axes([-0.2,0,0.15,1])
    dendrogram(Zr, orientation='right', ax=ax_row, color_threshold=0)
    ax_row.invert_xaxis()
    ax_row.axis('off')

    ax_col = axA.inset_axes([0,1.02,1,0.15])
    dendrogram(Zc, orientation='top', ax=ax_col, color_threshold=0)
    ax_col.axis('off')

    # ---- Panel B ----
    axB = fig.add_subplot(gs[2,0:2])
    order_forest = forest_panel(axB, mat_z, psy_names, sud_names)
    axB.set_title('B. PSY disorder summary')

    # ---- Panel C ----
    axC = fig.add_subplot(gs[2,2:])
    for cname, disorders in manual_clusters.items():
        inds = [psy_names.index(d) for d in disorders if d in psy_names]
        prof = np.nanmean(mat_z[inds,:], axis=0)
        axC.plot(prof, marker='o', lw=2,
                 color=cluster_colors[cname],
                 label='-'.join(disorders))
    axC.set_xticks(range(len(sud_names)))
    axC.set_xticklabels(sud_names, rotation=45, ha='right')
    axC.legend(frameon=False)
    axC.set_title('C. Cluster fingerprints')

    # ---- Panel D ----
    axD = fig.add_subplot(gs[3,0:2])
    bipartite_lollipop_panel(
        axD, mat_z,
        order_forest, col_order,
        psy_names, sud_names,
        cluster_colors_map={
            i: psy_cluster_map.get(psy_names[i],'gray')
            for i in range(len(psy_names))
        }
    )
    axD.set_title('D. Top PSY–SUD links')

    # ---- Panel E ----
    axE = fig.add_subplot(gs[3,2:])
    if mat_sub is not None:
        mean_ctx = np.nanmean(mat, axis=1)
        mean_sub = np.nanmean(mat_sub, axis=1)
        colors = [psy_cluster_map.get(p,'gray') for p in psy_names]

        axE.scatter(mean_sub, mean_ctx, s=80,
                    c=colors, edgecolor='k')

        for i, p in enumerate(psy_names):
            axE.annotate(p, (mean_sub[i], mean_ctx[i]),
                         xytext=(3,3), textcoords='offset points',
                         fontsize=8)

        lims = [min(mean_sub.min(), mean_ctx.min()),
                max(mean_sub.max(), mean_ctx.max())]
        axE.plot(lims, lims, '--', color='gray')

        legend_handles = [
            Line2D([0],[0], marker='o', color='none',
                   markerfacecolor=c, markeredgecolor='k',
                   markersize=8, label=name)
            for name, c in cluster_colors.items()
        ]
        axE.legend(handles=legend_handles,
                   title='PSY cluster',
                   frameon=False,
                   fontsize='small',
                   loc='upper left')

        axE.set_xlabel('Subcortex')
        axE.set_ylabel('Cortex')
        axE.set_title('E. Cortex vs Subcortex')
    else:
        axE.axis('off')

    fig.suptitle('Figure 1 — PSY × SUD similarity (Euclidean)', fontsize=16)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outpath}")


# -----------------------
# CLI
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_dirs', nargs='+', required=True)
    parser.add_argument('--out', default='figure1_euclidean.png')
    parser.add_argument('--topK', type=int, default=12)
    args = parser.parse_args()
    make_figure(args.group_dirs, args.out, args.topK)
