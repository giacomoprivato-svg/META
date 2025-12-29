#!/usr/bin/env python3
"""
Multipanel Figure Generator — PSY × SUD similarity (cortex only)

Creates a single figure with 4 heatmaps:
A. Cosine similarity
B. Spearman similarity
C. Euclidean distance
D. Z_combined (as in previous figures)

Supports both adults and adolescents. Data are loaded from multiple group directories
and concatenated along rows (PSY disorders), exactly as in Figure 1.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from matplotlib.gridspec import GridSpec

# -----------------------
# Helpers
# -----------------------
def hierarchical_order(mat, method='ward', metric='euclidean'):
    data = mat.copy()
    if np.isnan(data).any():
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])
    if data.shape[0] <= 1:
        return np.arange(data.shape[0]), None
    Z = linkage(pdist(data, metric=metric), method=method)
    order = leaves_list(Z)
    return order, Z


def symmetric_limits(mat):
    vmax = np.nanmax(np.abs(mat))
    return (-vmax, vmax)

# -----------------------
# Main figure function
# -----------------------
def make_figure(group_dirs, outpath, title_prefix=''):

    fnames = {
        'cosine': 'Z_cortex_cosine.csv',
        'spearman': 'Z_cortex_spearman.csv',
        'euclidean': 'Z_cortex_euclidean.csv',
        'z_combined': 'Z_cortex_combined.csv'
    }

    # Load and concatenate across group dirs
    data = {k: [] for k in fnames}

    for gd in group_dirs:
        for key, fn in fnames.items():
            path = os.path.join(gd, fn)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
            data[key].append(pd.read_csv(path, index_col=0))

    for key in data:
        data[key] = pd.concat(data[key], axis=0)

    base_cols = data['z_combined'].columns
    for key in data:
        data[key] = data[key].reindex(columns=base_cols)

    psy_names = list(data['z_combined'].index)
    sud_names = list(base_cols)

    mat_z = data['z_combined'].to_numpy(dtype=float)
    row_order, _ = hierarchical_order(mat_z)
    col_order, _ = hierarchical_order(mat_z.T)

    mats = {k: v.to_numpy(dtype=float)[np.ix_(row_order, col_order)] for k, v in data.items()}

    psy_ord = [psy_names[i] for i in row_order]
    sud_ord = [sud_names[i] for i in col_order]

    plt.close('all')
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.25)

    panels = [
        ('cosine', 'A. Cosine similarity', 'vlag', symmetric_limits(mats['cosine'])),
        ('spearman', 'B. Spearman similarity', 'vlag', symmetric_limits(mats['spearman'])),
        ('euclidean', 'C. Euclidean distance', 'vlag', symmetric_limits(mats['euclidean'])),
        ('z_combined', 'D. Z-combined similarity', 'vlag', symmetric_limits(mats['z_combined']))
    ]

    for i, (key, title, cmap, vlims) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        sns.heatmap(
            mats[key],
            ax=ax,
            cmap=cmap,
            xticklabels=sud_ord,
            yticklabels=psy_ord,
            linewidths=0.4,
            linecolor='lightgray',
            cbar_kws={'shrink': 0.7},
            vmin=vlims[0] if vlims else None,
            vmax=vlims[1] if vlims else None
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('SUD')
        ax.set_ylabel('PSY')
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(f'{title_prefix}PSY × SUD similarity (cortex)', fontsize=18)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved multipanel heatmap figure to {outpath}")

# -----------------------
# CLI
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 4-panel cortex-only PSY×SUD heatmap figure.')
    parser.add_argument('--out_adults', type=str, default='figure_cortex_adults.png', help='Output figure path for adults')
    parser.add_argument('--out_adolescents', type=str, default='figure_cortex_adolescents.png', help='Output figure path for adolescents')
    args = parser.parse_args()

    adults_dirs = [
        r'C:\Users\giaco\Desktop\Git_META\META\ALL_outputs_RQ1\adults_all',
        r'C:\Users\giaco\Desktop\Git_META\META\ALL_outputs_RQ1\adults_ctx'
    ]

    adolescents_dirs = [
        r'C:\Users\giaco\Desktop\Git_META\META\ALL_outputs_RQ1\adolescents_all',
        r'C:\Users\giaco\Desktop\Git_META\META\ALL_outputs_RQ1\adolescents_ctx'
    ]

    make_figure(adults_dirs, args.out_adults, title_prefix='Adults — ')
    make_figure(adolescents_dirs, args.out_adolescents, title_prefix='Adolescents — ')
