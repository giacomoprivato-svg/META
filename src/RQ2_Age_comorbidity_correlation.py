#!/usr/bin/env python3
"""
Brain–age of onset association by PSY cluster
Adults only

• Brain metric: Z_combined (cortex)
• Age of onset: mean age of onset for PSY–SUD comorbidity
• Stratified by PSY cluster
• Scatter + pooled regression line (visual aid)
• Spearman correlation for inference
• Additional plots for SCZ/MDD/BD only and all pairs together
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
AGE_ONSET_FILE = os.path.join(
    BASE_DIR, "data", "raw", "age_onset_prevalence_of_disorders.xlsx"
)

ADULT_DIRS = [
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_ctx"),
]

OUTFIG_CLUSTERS = "brain_vs_age_onset_clusters_adults.png"
OUTFIG_SCZ_MDD_BD = "brain_vs_age_onset_SCZ_MDD_BD.png"
OUTFIG_ALL_PAIRS = "brain_vs_age_onset_all_pairs.png"

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

# -----------------------
# LOAD AGE OF ONSET
# -----------------------
age_onset = pd.read_excel(AGE_ONSET_FILE, index_col=0)

def to_float(x):
    try:
        return float(str(x).replace(",", ".").strip())
    except:
        return np.nan

age_onset = age_onset.applymap(to_float)

# Convert to long format
age_onset_long = age_onset.reset_index().melt(
    id_vars="DISORDER", var_name="SUD", value_name="AgeOnset"
).rename(columns={"DISORDER": "PSY"})

# -----------------------
# LOAD BRAIN DATA (Adults only)
# -----------------------
records = []

for d in ADULT_DIRS:
    path = os.path.join(d, "Z_cortex_combined.csv")
    brain = pd.read_csv(path, index_col=0)

    brain_long = (
        brain.reset_index()
        .melt(id_vars="index", var_name="SUD", value_name="Z_combined")
        .rename(columns={"index": "PSY"})
    )
    brain_long["AgeGroup"] = "Adults"
    records.append(brain_long)

brain_all = pd.concat(records, axis=0)
brain_all["Z_combined"] = pd.to_numeric(brain_all["Z_combined"], errors="coerce")

# -----------------------
# MERGE BRAIN + AGE OF ONSET
# -----------------------
df = pd.merge(
    brain_all,
    age_onset_long,
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
df = df.dropna(subset=["Z_combined", "AgeOnset", "Cluster"])

print(f"Total PSY–SUD pairs (adults, valid): {len(df)}")

# -----------------------
# FUNCTION TO RUN SPEARMAN
# -----------------------
def run_spearman(sub_df):
    x = sub_df["AgeOnset"].values
    y = sub_df["Z_combined"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) >= 3:
        rho, p = spearmanr(x[mask], y[mask])
        return rho, p, np.sum(mask)
    else:
        return np.nan, np.nan, np.sum(mask)

# 1. Spearman for all adults
rho_all, p_all, n_all = run_spearman(df)
print(f"All adults: Spearman ρ = {rho_all:.2f}, p = {p_all:.3f}, n = {n_all}")

# 2. Spearman by cluster
for cluster in CLUSTER_ORDER:
    sub = df[df["Cluster"] == cluster]
    rho, p, n = run_spearman(sub)
    print(f"{cluster}: Spearman ρ = {rho:.2f}, p = {p:.3f}, n = {n}")

# 3. Spearman for SCZ, MDD, BD only
sub_scz = df[df["PSY"].isin(["SCZ", "MDD", "BD"])]
rho_sel, p_sel, n_sel = run_spearman(sub_scz)
print(f"SCZ/MDD/BD only: Spearman ρ = {rho_sel:.2f}, p = {p_sel:.3f}, n = {n_sel}")

# -----------------------
# COMBINED FIGURE: 4 clusters + SCZ/MDD/BD + All pairs
# -----------------------
plt.close("all")
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()
AGE_COLOR = "blue"

# 1. Cluster plots (first 4 subplots)
for i, cluster in enumerate(CLUSTER_ORDER):
    ax = axes[i]
    sub = df[df["Cluster"] == cluster]

    ax.scatter(
        sub["AgeOnset"],
        sub["Z_combined"],
        s=80,
        alpha=0.85,
        color=AGE_COLOR,
        edgecolor="k",
    )

    x = sub["AgeOnset"].values
    y = sub["Z_combined"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) >= 2 and np.std(x[mask]) > 0:
        m, b = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xs, m * xs + b, color="black", lw=2, alpha=0.8)

    rho, p = run_spearman(sub)[:2]
    ax.set_title(f"{cluster}\nSpearman ρ = {rho:.2f}, p = {p:.3f}, n = {len(sub)}", fontsize=12)
    ax.set_xlabel("Age of onset")
    ax.set_ylabel("Brain similarity (Z_combined)")

# 2. All PSY–SUD pairs (subplot 5)
ax = axes[4]
ax.scatter(
    df["AgeOnset"],
    df["Z_combined"],
    s=80,
    alpha=0.85,
    color=AGE_COLOR,
    edgecolor="k"
)
x = df["AgeOnset"].values
y = df["Z_combined"].values
mask = np.isfinite(x) & np.isfinite(y)
if np.sum(mask) >= 2 and np.std(x[mask]) > 0:
    m, b = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xs, m * xs + b, color="black", lw=2, alpha=0.8)

rho, p = run_spearman(df)[:2]
ax.set_title(f"All PSY–SUD pairs\nSpearman ρ = {rho:.2f}, p = {p:.3f}, n = {len(df)}", fontsize=12)
ax.set_xlabel("Age of onset")
ax.set_ylabel("Brain similarity (Z_combined)")

# 3. SCZ/MDD/BD only (subplot 6)
ax = axes[5]
ax.scatter(
    sub_scz["AgeOnset"],
    sub_scz["Z_combined"],
    s=80,
    alpha=0.85,
    color=AGE_COLOR,
    edgecolor="k"
)
x = sub_scz["AgeOnset"].values
y = sub_scz["Z_combined"].values
mask = np.isfinite(x) & np.isfinite(y)
if np.sum(mask) >= 2 and np.std(x[mask]) > 0:
    m, b = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xs, m * xs + b, color="black", lw=2, alpha=0.8)

rho, p = run_spearman(sub_scz)[:2]
ax.set_title(f"SCZ/MDD/BD only\nSpearman ρ = {rho:.2f}, p = {p:.3f}, n = {len(sub_scz)}", fontsize=12)
ax.set_xlabel("Age of onset")
ax.set_ylabel("Brain similarity (Z_combined)")

fig.suptitle("Brain–age of onset association (Adults)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("brain_vs_age_onset_combined_6plots.png", dpi=300, bbox_inches="tight")
print("Saved: brain_vs_age_onset_combined_6plots.png")
