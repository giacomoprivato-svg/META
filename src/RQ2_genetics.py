#!/usr/bin/env python3
"""
Brain–genetics association by PSY cluster
Pooled Adults + Adolescents

• Brain metric: Z_combined (cortex)
• Genetic metric: PSY–SUD genetic correlation
• Stratified by PSY cluster
• Scatter + pooled regression line (visual aid)
• Spearman correlation for inference
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.gridspec import GridSpec

# -----------------------
# CONFIG
# -----------------------
BASE_DIR = r"C:\Users\giaco\Desktop\Git_META\META"
GENETIC_FILE = os.path.join(
    BASE_DIR, "data", "raw", "PSY_SUD_genetic_corr.xlsx"
)

GROUPS = {
    "Adults": [
        os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all"),
        os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_ctx"),
    ],
    "Adolescents": [
        os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all"),
        os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_ctx"),
    ],
}

OUTFIG = "brain_vs_genetics_clusters_pooled.png"

# -----------------------
# PSY CLUSTERS
# -----------------------
PSY_CLUSTERS = {
    "Psychotic": ["SCZ", "BD"],
    "Neurodevelopmental": ["ASD", "ADHD"],
    "AN/OCD": ["AN", "OCD"],
    "Mood/Anxiety": ["MDD", "PTSD"],
}

CLUSTER_ORDER = list(PSY_CLUSTERS.keys())

AGE_COLORS = {
    "Adults": "#1f77b4",
    "Adolescents": "#ff7f0e",
}

# -----------------------
# LOAD GENETICS
# -----------------------
genetic = pd.read_excel(GENETIC_FILE, index_col=0)

genetic_long = (
    genetic.reset_index()
    .melt(id_vars="index", var_name="SUD", value_name="genetic_corr")
    .rename(columns={"index": "PSY"})
)

# -----------------------
# LOAD BRAIN DATA
# -----------------------
records = []

for age, dirs in GROUPS.items():
    mats = []
    for d in dirs:
        path = os.path.join(d, "Z_cortex_combined.csv")
        mats.append(pd.read_csv(path, index_col=0))
    brain = pd.concat(mats, axis=0)

    brain_long = (
        brain.reset_index()
        .melt(id_vars="index", var_name="SUD", value_name="Z_combined")
        .rename(columns={"index": "PSY"})
    )
    brain_long["AgeGroup"] = age
    records.append(brain_long)

brain_all = pd.concat(records, axis=0)

# -----------------------
# MERGE BRAIN + GENETICS
# -----------------------
df = pd.merge(
    brain_all,
    genetic_long,
    on=["PSY", "SUD"],
    how="inner"
)

# assign PSY cluster
def assign_cluster(psy):
    for c, members in PSY_CLUSTERS.items():
        if psy in members:
            return c
    return None

df["Cluster"] = df["PSY"].apply(assign_cluster)
df = df.dropna(subset=["Cluster"])

print(f"Total PSY–SUD pairs (pooled): {len(df)}")

# -----------------------
# PLOT
# -----------------------
plt.close("all")
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

for i, cluster in enumerate(CLUSTER_ORDER):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    sub = df[df["Cluster"] == cluster]

    # scatter by age group
    for age in ["Adults", "Adolescents"]:
        s = sub[sub["AgeGroup"] == age]
        ax.scatter(
            s["genetic_corr"],
            s["Z_combined"],
            s=80,
            alpha=0.85,
            label=age,
            color=AGE_COLORS[age],
            edgecolor="k",
        )

    # ---- REGRESSION LINE (POOLED) ----
    x = sub["genetic_corr"].values
    y = sub["Z_combined"].values

    if len(x) >= 3:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="black", lw=2, alpha=0.8)

    # ---- SPEARMAN ----
    rho, p = spearmanr(x, y)

    ax.set_title(
        f"{cluster}\n"
        f"Spearman ρ = {rho:.2f}, p = {p:.3f}, n = {len(x)}",
        fontsize=12
    )

    ax.set_xlabel("Genetic correlation")
    ax.set_ylabel("Brain similarity (Z_combined)")
    ax.legend(frameon=False)

fig.suptitle(
    "Brain–genetic association by PSY cluster\nPooled adults + adolescents",
    fontsize=16
)

plt.savefig(OUTFIG, dpi=300, bbox_inches="tight")
print(f"Saved: {OUTFIG}")
