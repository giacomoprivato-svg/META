#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------
# Paths (repo-relative)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)


# ======================
# PATHS
# ======================
BASE = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "results", "RQ1_BARPLOTS_FINAL")
os.makedirs(OUTDIR, exist_ok=True)

PATH_ADULTS = os.path.join(BASE, "PSY_adults.xlsx")
PATH_SUD = os.path.join(BASE, "SUD.xlsx")

N_CORTEX = 68
N_SUBCORT = 14

# ======================
# CLUSTERS (AS YOU GAVE)
# ======================
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

GRAY = (0.65, 0.65, 0.65)

# ======================
# LIGHT PASTEL FUNCTION
# ======================
def pastel(color, alpha=0.55):
    r, g, b = color
    return (1 - (1 - r) * alpha,
            1 - (1 - g) * alpha,
            1 - (1 - b) * alpha)

# ======================
# LOAD
# ======================
def load(path):
    return pd.read_excel(path).select_dtypes(include=np.number)

# ======================
# MEAN EFFECT
# ======================
def mean_effect(df):
    cortex = df.iloc[:N_CORTEX].mean(axis=0)

    if df.shape[0] >= N_CORTEX + N_SUBCORT:
        sub = df.iloc[N_CORTEX:N_CORTEX+N_SUBCORT].mean(axis=0)
    else:
        sub = None

    return cortex, sub

# ======================
# CLUSTER ASSIGNMENT
# ======================
def get_cluster(label):
    for cl, members in manual_clusters.items():
        if label in members:
            return cl
    return None

# ======================
# SORT BY CORTEX
# ======================
def sort_by_cortex(cortex_vals, labels, sub_vals, is_sud):

    idx = np.argsort(cortex_vals)

    return (
        np.array(labels)[idx],
        cortex_vals[idx],
        sub_vals[idx],
        is_sud[idx]   # 🔥 FIX CRUCIALE
    )

# ======================
# MAIN PLOT
# ======================
def plot_adults(df):

    cortex, subcortex = mean_effect(df)

    labels = list(df.columns)

    cortex_vals = cortex.values
    sub_vals = subcortex.values

    # ----------------------
    # ADD SUD
    # ----------------------
    sud_df = load(PATH_SUD)
    sud_cortex, sud_sub = mean_effect(sud_df)

    labels += list(sud_df.columns)

    cortex_vals = np.concatenate([cortex_vals, sud_cortex.values])
    sub_vals = np.concatenate([sub_vals, sud_sub.values])

    is_sud = np.array([False]*len(df.columns) + [True]*len(sud_df.columns))

    # ----------------------
    # SORT (FIXED ALL VARIABLES)
    # ----------------------
    labels, cortex_vals, sub_vals, is_sud = sort_by_cortex(
        cortex_vals, labels, sub_vals, is_sud
    )

    # ======================
    # COLORS (NOW CORRECTLY ALIGNED)
    # ======================
    cortex_colors = []
    sub_colors = []

    for l, sud_flag in zip(labels, is_sud):

        if sud_flag:
            base = GRAY
        else:
            cl = get_cluster(l)
            base = cluster_colors.get(cl, (0.4,0.4,0.4))

        cortex_colors.append(base)
        sub_colors.append(pastel(base))

    # ======================
    # FIGURE
    # ======================
    fig, ax = plt.subplots(figsize=(26, 8))
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.38

    # ----------------------
    # BARS
    # ----------------------
    ax.bar(
        x - width/2,
        cortex_vals,
        width=width,
        color=cortex_colors,
        edgecolor="black",
        linewidth=1.2
    )

    ax.bar(
        x + width/2,
        sub_vals,
        width=width,
        color=sub_colors,
        edgecolor="black",
        linewidth=1.2
    )

    # ======================
    # AXES
    # ======================
    ax.axhline(0, color="black", lw=1.3, linestyle="--", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22)

    ax.set_ylabel("Cohen's d", fontsize=26)
    ax.set_title("Mean Cohen's d", fontsize=20)

    ax.invert_yaxis()

    # ======================
    # STYLE
    # ======================
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # ======================
    # LEGEND
    # ======================
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=cluster_colors['Psychotic'], label='Psychotic'),
        Patch(facecolor=cluster_colors['Neurodevelopmental'], label='Neurodevelopmental'),
        Patch(facecolor=cluster_colors['AN/OCD'], label='AN/OCD'),
        Patch(facecolor=cluster_colors['Mood/Anxiety'], label='Mood/Anxiety'),
        Patch(facecolor=GRAY, label='SUD'),

        # 🔥 struttura come voci normali
        Patch(facecolor="none", edgecolor="none", label="Cortex - darker"),
        Patch(facecolor="none", edgecolor="none", label="Subcortex - lighter"),
    ]

    ax.legend(
        handles=legend_patches,
        title="Clinical clusters",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="0.3",
        fontsize=20,
        title_fontsize=22,
        loc="upper right",
        labelspacing=0.6,
        borderpad=0.8
    )

    # 🔥 SOLO titolo bold e centrato
    leg = ax.get_legend()
    leg.get_title().set_fontweight('bold')
    leg.get_title().set_multialignment('center')

    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, "ADULTS_cortex_subcortex_grouped.png"),
                dpi=300, bbox_inches="tight")

    plt.close()

# ======================
# RUN
# ======================
print("Running...")

df_adults = load(PATH_ADULTS)
plot_adults(df_adults)

print("DONE ✔")