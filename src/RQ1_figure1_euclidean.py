#!/usr/bin/env python3
"""
Figure 1 generator for PSY × SUD similarity panels (A–E)
Adults-only cortex data
EUCLIDEAN similarity version (permutation-calibrated Z-values)

Interpretation:
Higher Z = higher similarity (lower Euclidean distance than null)
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
# Globals
# -----------------------

SUD_COLORS = sns.color_palette("tab10", 10)

# -----------------------
# Helpers
# -----------------------

def hierarchical_order(mat, axis=0, method='ward', metric='euclidean'):
    data = mat if axis == 0 else mat.T

    if np.isnan(data).any():
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data = data.copy()
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
    """
    Shows top PSY–SUD links by highest similarity Z
    """
    ax.clear()
    nL, nR = mat.shape

    # select top links by Z (higher = more similar)
    vals = mat.flatten()
    n_top = max(1, int(len(vals) * top_fraction))
    top_inds = np.argsort(vals)[-n_top:]

    top_pairs = [(i // nR, i % nR) for i in top_inds]
    top_vals = np.array([mat[i, j] for i, j in top_pairs])

    min_v, max_v = top_vals.min(), top_vals.max()
    norm_vals = (top_vals - min_v) / (max_v - min_v + 1e-8)

    left_pos = {idx: pos for pos, idx in enumerate(left_order)}
    right_pos = {idx: pos for pos, idx in enumerate(right_order)}

    for k, (i, j) in enumerate(top_pairs):
        yi = left_pos[i] / max(1, len(left_order) - 1)
        xi = right_pos[j] / max(1, len(right_order) - 1)
        size = 50 + max_circle_size * norm_vals[k]
        color = cluster_colors_map.get(i, 'gray')

        ax.plot([0, xi], [yi, yi], lw=1, color=color, alpha=0.6)
        ax.scatter(xi, yi, s=size, color=color, edgecolor='k', zorder=3)

    ax.set_yticks([i / max(1, len(left_order) - 1) for i in range(len(left_order))])
    ax.set_yticklabels([psy_names[i] for i in left_order], fontsize=9)

    ax.set_xticks([i / max(1, len(right_order) - 1) for i in range(len(right_order))])
    ax.set_xticklabels([sud_names[i] for i in right_order],
                       rotation=45, ha='right', fontsize=9)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('SUD')
    ax.set_title('Top PSY–SUD similarity links')


def forest_panel(ax, mat, psy_names, sud_names):
    means = np.nanmean(mat, axis=1)
    order = np.argsort(-means)

    mat_ord = mat[order, :]
    y = np.arange(len(order))

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

def make_figure(group_dirs, outpath):

    cortex_list, subctx_list = [], []

    for gd in group_dirs:
        ctx = os.path.join(gd, 'Z_cortex_euclidean.csv')
        sub = os.path.join(gd, 'Z_subctx_euclidean.csv')

        if os.path.exists(ctx):
            cortex_list.append(pd.read_csv(ctx, index_col=0))
        if os.path.exists(sub):
            subctx_list.append(pd.read_csv(sub, index_col=0))

    if not cortex_list:
        raise FileNotFoundError("No cortex Euclidean CSVs found.")

    Zc = pd.concat(cortex_list, axis=0)
    mat = Zc.to_numpy(float)

    psy_names = ['Schizotypy' if p.lower() == 'schizotypic' else p
                 for p in Zc.index]
    sud_names = list(Zc.columns)

    # subcortex (optional, panel E)
    mat_sub = None
    if subctx_list:
        Zs = pd.concat(subctx_list, axis=0)
        Zs = Zs.reindex(index=Zc.index, columns=Zc.columns)
        mat_sub = Zs.to_numpy(float)

    row_order, Zr = hierarchical_order(mat, axis=0)
    col_order, Zc_link = hierarchical_order(mat, axis=1)

    mat_ord = mat[np.ix_(row_order, col_order)]
    psy_ord = [psy_names[i] for i in row_order]
    sud_ord = [sud_names[i] for i in col_order]

    # manual PSY clusters
    manual_clusters = {
        'Psychotic': ['SCZ','BD','CHR','Schizotypy'],
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
            if d in psy_names:
                psy_cluster_map[psy_names.index(d)] = cluster_colors[cname]

    # -----------------------
    # Figure layout
    # -----------------------

    fig = plt.figure(figsize=(22,14))
    gs = GridSpec(4,4, figure=fig,
                  width_ratios=[0.2,1,1,1.2],
                  height_ratios=[0.4,1,1,1],
                  wspace=0.6, hspace=0.5)

    # ---- Panel A ----
    axA = fig.add_subplot(gs[0:2,1:])
    sns.heatmap(
        mat_ord,
        ax=axA,
        cmap='cividis',
        center=0,
        xticklabels=sud_ord,
        yticklabels=psy_ord,
        cbar_kws={'label': 'Euclidean similarity (Z)'}
    )
    axA.set_title('A. PSY × SUD similarity (Euclidean, cortex)')

    ax_row = axA.inset_axes([-0.2,0,0.15,1])
    dendrogram(Zr, orientation='right', ax=ax_row, color_threshold=0)
    ax_row.invert_xaxis()
    ax_row.axis('off')

    ax_col = axA.inset_axes([0,1.02,1,0.15])
    dendrogram(Zc_link, orientation='top', ax=ax_col, color_threshold=0)
    ax_col.axis('off')

    # ---- Panel B ----
    axB = fig.add_subplot(gs[2,0:2])
    order_forest = forest_panel(axB, mat, psy_names, sud_names)
    axB.set_title('B. PSY disorder summary')

    # ---- Panel C ----
    axC = fig.add_subplot(gs[2,2:])
    for cname, disorders in manual_clusters.items():
        inds = [psy_names.index(d) for d in disorders if d in psy_names]
        prof = np.nanmean(mat[inds, :], axis=0)
        axC.plot(prof, marker='o', lw=2,
                 color=cluster_colors[cname], label=cname)

    axC.set_xticks(range(len(sud_names)))
    axC.set_xticklabels(sud_names, rotation=45, ha='right')
    axC.set_ylabel('Euclidean similarity (Z)')
    axC.legend(frameon=False)
    axC.set_title('C. PSY cluster fingerprints')

    # ---- Panel D ----
    axD = fig.add_subplot(gs[3,0:2])
    bipartite_lollipop_panel(
        axD, mat,
        order_forest, col_order,
        psy_names, sud_names,
        psy_cluster_map
    )
    axD.set_title('D. Top PSY–SUD links')

    # ---- Panel E ----
    axE = fig.add_subplot(gs[3,2:])
    if mat_sub is not None:
        mean_ctx = np.nanmean(mat, axis=1)
        mean_sub = np.nanmean(mat_sub, axis=1)
        colors = [psy_cluster_map.get(i, 'gray') for i in range(len(psy_names))]

        # scatter plot
        marker_size = 80  # già usato sopra
        axE.scatter(mean_sub, mean_ctx, s=marker_size,
                    c=colors, edgecolor='k', zorder=3)

        # calcola offset proporzionale alla dimensione del marker
        offset = max(4, marker_size ** 0.5 / 2)  # in punti

        for i, p in enumerate(psy_names):
            # alterna leggermente la direzione per evitare sovrapposizione tra label
            dx = offset if i % 2 == 0 else -offset
            dy = offset if (i // 2) % 2 == 0 else -offset

            ha = 'left' if dx > 0 else 'right'
            va = 'bottom' if dy > 0 else 'top'

            axE.annotate(p, (mean_sub[i], mean_ctx[i]),
                        xytext=(dx, dy),
                        textcoords='offset points',
                        fontsize=8,
                        ha=ha, va=va)

        # linea diagonale y=x
        lims = [min(mean_sub.min(), mean_ctx.min()),
                max(mean_sub.max(), mean_ctx.max())]
        axE.plot(lims, lims, '--', color='gray')

        # legenda cluster PSY
        legend_handles = [
            Line2D([0],[0], marker='o', color='none',
                markerfacecolor=c, markeredgecolor='k',
                markersize=8, label=name)
            for name, c in cluster_colors.items()
        ]
        axE.legend(handles=legend_handles, frameon=False,
                title='PSY cluster', fontsize='small')

        axE.set_xlabel('Subcortex similarity (Z)')
        axE.set_ylabel('Cortex similarity (Z)')
        axE.set_title('E. Cortex vs Subcortex')
    else:
        axE.axis('off')




    fig.suptitle('Figure 1 — PSY × SUD Euclidean similarity (adults)', fontsize=16)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outpath}")


# -----------------------
# CLI
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_dirs', nargs='+', required=True,
                        help='e.g. adults_ctx adults_all')
    parser.add_argument('--out', default='figure1_euclidean.png')
    args = parser.parse_args()

    make_figure(args.group_dirs, args.out)
