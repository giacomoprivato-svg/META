#!/usr/bin/env python3
"""
SUBCORTICAL (Z_subctx_euclidean)
Adults + Adolescents

6-panel figure:
- Adults Comorbidity
- Adolescents Comorbidity
- Combined Comorbidity
- Adults Genetics
- Adolescents Genetics
- Combined Genetics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ----------------------------------------------------------
# ROBUST FUNCTIONS
# ----------------------------------------------------------

def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) >= 2:
        return spearmanr(x[mask], y[mask])[0]
    return np.nan

def bootstrap_spearman(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r = spearman_safe(x[idx], y[idx])
        if np.isfinite(r):
            boot.append(r)
    boot = np.array(boot)
    return np.nanmean(boot), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

def permutation_spearman(x, y, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    rho_obs = spearman_safe(x, y)
    null_dist = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r = spearman_safe(x, y_perm)
        if np.isfinite(r):
            null_dist.append(r)
    null_dist = np.array(null_dist)
    p_perm = np.mean(np.abs(null_dist) >= np.abs(rho_obs))
    return rho_obs, p_perm

def leave_one_out_spearman(x, y):
    loo_vals = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        r = spearman_safe(x[mask], y[mask])
        loo_vals.append(r)
    return np.array(loo_vals)

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

BASE_DIR = r"C:\Users\giaco\Desktop\Git_META\META"

ADULT_DIR = os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")
ADOLESCENT_DIR = os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all")

# ----------------------------------------------------------
# LOAD BRAIN DATA (SUBCORTICAL)
# ----------------------------------------------------------

def load_brain(dir_path):
    brain = pd.read_csv(
        os.path.join(dir_path, "Z_subctx_euclidean.csv"),
        index_col=0
    )
    brain_long = brain.reset_index().melt(
        id_vars="index",
        var_name="SUD",
        value_name="Z_euclidean"
    ).rename(columns={"index": "PSY"})
    return brain_long

brain_adults = load_brain(ADULT_DIR)
brain_adults["Group"] = "Adults"

brain_adol = load_brain(ADOLESCENT_DIR)

# FIX ADHD labels
brain_adol["PSY"] = brain_adol["PSY"].replace({
    "ADHD_ch": "ADHD",
    "ADHD_ado": "ADHD"
})

brain_adol["Group"] = "Adolescents"

brain_all = pd.concat([brain_adults, brain_adol])

# ----------------------------------------------------------
# LOAD AGE OF ONSET
# ----------------------------------------------------------

AGE_FILE = os.path.join(BASE_DIR, "data", "raw", "age_onset_prevalence_of_disorders.xlsx")
age_onset = pd.read_excel(AGE_FILE, index_col=0)
age_onset = age_onset.applymap(
    lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan
)

age_long = age_onset.reset_index().melt(
    id_vars="DISORDER",
    var_name="SUD",
    value_name="AgeOnset"
).rename(columns={"DISORDER": "PSY"})

# ----------------------------------------------------------
# LOAD GENETICS
# ----------------------------------------------------------

GEN_FILE = os.path.join(BASE_DIR, "data", "raw", "PSY_SUD_genetic_corr.xlsx")
genetic = pd.read_excel(GEN_FILE, index_col=0)

gen_long = genetic.reset_index().melt(
    id_vars="index",
    var_name="SUD",
    value_name="genetic_corr"
).rename(columns={"index": "PSY"})

# ----------------------------------------------------------
# MERGE DATA
# ----------------------------------------------------------

df_age = pd.merge(brain_all, age_long, on=["PSY", "SUD"], how="inner")
df_age = df_age.dropna(subset=["Z_euclidean", "AgeOnset"])

df_gen = pd.merge(brain_all, gen_long, on=["PSY", "SUD"], how="inner")
df_gen = df_gen.dropna(subset=["Z_euclidean", "genetic_corr"])

# Separate datasets
df_age_adults = df_age[df_age["Group"] == "Adults"]
df_age_adol = df_age[df_age["Group"] == "Adolescents"]

df_gen_adults = df_gen[df_gen["Group"] == "Adults"]
df_gen_adol = df_gen[df_gen["Group"] == "Adolescents"]

# ----------------------------------------------------------
# CREATE 6-PANEL FIGURE
# ----------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
results = []

def plot_panel(ax, df, x_col, title, label):
    for group, color in zip(["Adults", "Adolescents"], ["blue", "orange"]):
        sub = df[df["Group"] == group]
        ax.scatter(sub[x_col], sub["Z_euclidean"],
                   s=60, alpha=0.7,
                   color=color,
                   label=group)

    x = df[x_col].values
    y = df["Z_euclidean"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) >= 2:
        # Pearson
        r, p = pearsonr(x, y)

        # Spearman
        rho_s, ci_low, ci_high = bootstrap_spearman(x, y)
        rho_perm, p_perm = permutation_spearman(x, y)
        loo = leave_one_out_spearman(x, y)

        # Linear fit
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m*xs + b, color="black", linewidth=2)

        # Display stats
        ax.text(0.98, 0.02,
                f"r={r:.2f}, p={p:.3f}",
                transform=ax.transAxes,
                ha="right", va="bottom")

        results.append({
            "Analysis": label,
            "r_pearson": r,
            "p_pearson": p,
            "rho_spearman": rho_s,
            "spearman_CI_low": ci_low,
            "spearman_CI_high": ci_high,
            "p_spearman_perm": p_perm,
            "LOO_min": np.nanmin(loo),
            "LOO_max": np.nanmax(loo),
            "N": len(x)
        })

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Brain similarity (Z_subctx_euclidean)")

# ----- COMORBIDITY -----
plot_panel(axes[0], df_age_adults, "AgeOnset",
           "Adults: Comorbidity",
           "Adults_Comorbidity")

plot_panel(axes[1], df_age_adol, "AgeOnset",
           "Adolescents: Comorbidity",
           "Adolescents_Comorbidity")

plot_panel(axes[2], df_age, "AgeOnset",
           "Combined: Comorbidity",
           "Combined_Comorbidity")

# ----- GENETICS -----
plot_panel(axes[3], df_gen_adults, "genetic_corr",
           "Adults: Genetics",
           "Adults_Genetics")

plot_panel(axes[4], df_gen_adol, "genetic_corr",
           "Adolescents: Genetics",
           "Adolescents_Genetics")

plot_panel(axes[5], df_gen, "genetic_corr",
           "Combined: Genetics",
           "Combined_Genetics")

# Global legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor='blue', markersize=8, label='Adults'),
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor='orange', markersize=8, label='Adolescents')
]

fig.legend(handles=handles,
           loc='lower center',
           ncol=2,
           frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("SUBCTX_GLOBAL_6panel_stats_spearman.png", dpi=300)
plt.close()

print("Saved: SUBCTX_GLOBAL_6panel_stats_spearman.png")

# ----------------------------------------------------------
# SAVE STATS
# ----------------------------------------------------------

pd.DataFrame(results).to_csv(
    "SUBCTX_GLOBAL_6panel_stats_spearman.csv",
    index=False
)

print("Saved stats: SUBCTX_GLOBAL_6panel_stats_spearman.csv")