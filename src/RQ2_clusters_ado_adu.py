#!/usr/bin/env python3
"""
Adults + Adolescents (Cortex only): brain similarity vs comorbidity and genetics
Scatter plots per cluster, OLS regression line, Pearson r + p-value on figure,
all statistics (Pearson & Spearman) saved in CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# ---------------- ROBUST STAT FUNCTIONS ----------------

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
    boot_r = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r = spearman_safe(x[idx], y[idx])
        if np.isfinite(r):
            boot_r.append(r)
    boot_r = np.array(boot_r)
    return np.nanmean(boot_r), np.percentile(boot_r, 2.5), np.percentile(boot_r, 97.5)

def permutation_spearman(x, y, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    r_true = spearman_safe(x, y)
    perm_r = []
    for _ in range(n_perm):
        perm_r.append(spearman_safe(x, rng.permutation(y)))
    perm_r = np.array(perm_r)
    p = np.mean(np.abs(perm_r) >= np.abs(r_true))
    return r_true, p

def leave_one_out_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    return np.array([spearman_safe(np.delete(x, i), np.delete(y, i)) for i in range(len(x))])

# ---------------- CONFIG ----------------
BASE_DIR = r"C:\Users\giaco\Desktop\Git_META\META"

PSY_CLUSTERS = {
    "Psychotic": ["SCZ", "BD"],
    "Neurodevelopmental": ["ASD", "ADHD"],
    "AN/OCD": ["AN", "OCD"],
    "Mood/Anxiety": ["MDD", "PTSD"]
}
CLUSTER_ORDER = list(PSY_CLUSTERS.keys())
CLUSTER_COLORS = {
    "Psychotic": "#ff7f0e",
    "Neurodevelopmental": "#2ca02c",
    "AN/OCD": "#1f77b4",
    "Mood/Anxiety": "#9467bd"
}

def assign_cluster(psy):
    for c, members in PSY_CLUSTERS.items():
        if psy in members:
            return c
    return None

# ---------------- LOAD DATA ----------------
# Age of onset
AGE_FILE = os.path.join(BASE_DIR, "data", "raw", "age_onset_prevalence_of_disorders.xlsx")
age_onset = pd.read_excel(AGE_FILE, index_col=0)
age_onset = age_onset.applymap(lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan)
age_long = age_onset.reset_index().melt(id_vars="DISORDER", var_name="SUD", value_name="AgeOnset")
age_long = age_long.rename(columns={"DISORDER": "PSY"})

# Brain data: adults + adolescents (Cortex only)
ADULT_DIRS = [
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_ctx")
]

records = []
for d in ADULT_DIRS:
    brain_c = pd.read_csv(os.path.join(d, "Z_cortex_euclidean.csv"), index_col=0)
    rec_c = brain_c.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean").rename(columns={"index": "PSY"})
    records.append(rec_c)

brain_all = pd.concat(records)

# Harmonize ADHD labels
brain_all["PSY"] = brain_all["PSY"].replace({
    "ADHD_ch": "ADHD",
    "ADHD_ado": "ADHD"
})

# Merge with age
df_age = pd.merge(brain_all, age_long, on=["PSY", "SUD"], how="inner")
df_age["Cluster"] = df_age["PSY"].apply(assign_cluster)
df_age = df_age.dropna(subset=["Z_euclidean", "AgeOnset", "Cluster"])

# Genetics
GEN_FILE = os.path.join(BASE_DIR, "data", "raw", "PSY_SUD_genetic_corr.xlsx")
genetic = pd.read_excel(GEN_FILE, index_col=0)
gen_long = genetic.reset_index().melt(id_vars="index", var_name="SUD", value_name="genetic_corr").rename(columns={"index": "PSY"})

records_gen = []
for d in ADULT_DIRS:
    brain = pd.read_csv(os.path.join(d, "Z_cortex_euclidean.csv"), index_col=0)
    rec = brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean").rename(columns={"index": "PSY"})
    records_gen.append(rec)

brain_all_gen = pd.concat(records_gen)
brain_all_gen["PSY"] = brain_all_gen["PSY"].replace({
    "ADHD_ch": "ADHD",
    "ADHD_ado": "ADHD"
})
df_gen = pd.merge(brain_all_gen, gen_long, on=["PSY", "SUD"], how="inner")
df_gen["Cluster"] = df_gen["PSY"].apply(assign_cluster)
df_gen = df_gen.dropna(subset=["Z_euclidean", "genetic_corr", "Cluster"])

# ----------------- PLOTS & STATS -------------------------
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes = axes.flatten()

def add_margin(vmin, vmax, margin=0.05):
    delta = vmax - vmin
    return vmin - delta*margin, vmax + delta*margin

age_xmin, age_xmax = add_margin(df_age["AgeOnset"].min(), df_age["AgeOnset"].max())
age_ymin, age_ymax = add_margin(df_age["Z_euclidean"].min(), df_age["Z_euclidean"].max())
gen_xmin, gen_xmax = add_margin(df_gen["genetic_corr"].min(), df_gen["genetic_corr"].max())
gen_ymin, gen_ymax = add_margin(df_gen["Z_euclidean"].min(), df_gen["Z_euclidean"].max())

results = []

# ---------------- PANEL 1: Age of onset -----------------
ax = axes[0]
for i, cluster in enumerate(CLUSTER_ORDER):
    sub = df_age[df_age["Cluster"] == cluster]
    x, y = sub["AgeOnset"].values, sub["Z_euclidean"].values
    ax.scatter(x, y, s=80, alpha=0.6, color=CLUSTER_COLORS[cluster])
    
    if len(x) >= 2:
        r_pearson = np.corrcoef(x, y)[0,1]
        rho_s, ci_low, ci_high = bootstrap_spearman(x, y)
        rho_perm, p_perm = permutation_spearman(x, y)
        loo = leave_one_out_spearman(x, y)
        m, b = np.polyfit(x, y, 1)
        xs_data = np.linspace(x.min(), x.max(), 100)
        ys_data = m*xs_data + b
        ax.plot(xs_data, ys_data, color=CLUSTER_COLORS[cluster], lw=2)

        results.append({
            "Cluster": cluster,
            "Plot": "Brain vs Comorbidity",
            "r_pearson": r_pearson,
            "rho_spearman": rho_s,
            "spearman_CI_low": ci_low,
            "spearman_CI_high": ci_high,
            "p_spearman_perm": p_perm,
            "LOO_min": np.nanmin(loo),
            "LOO_max": np.nanmax(loo),
            "N": len(sub)
        })

# ---------------- PANEL 2: Genetics -----------------
ax = axes[1]
for i, cluster in enumerate(CLUSTER_ORDER):
    sub = df_gen[df_gen["Cluster"] == cluster]
    x, y = sub["genetic_corr"].values, sub["Z_euclidean"].values
    ax.scatter(x, y, s=80, alpha=0.6, color=CLUSTER_COLORS[cluster])
    
    if len(x) >= 2:
        r_pearson = np.corrcoef(x, y)[0,1]
        rho_s, ci_low, ci_high = bootstrap_spearman(x, y)
        rho_perm, p_perm = permutation_spearman(x, y)
        loo = leave_one_out_spearman(x, y)
        m, b = np.polyfit(x, y, 1)
        xs_data = np.linspace(x.min(), x.max(), 100)
        ys_data = m*xs_data + b
        ax.plot(xs_data, ys_data, color=CLUSTER_COLORS[cluster], lw=2)

        results.append({
            "Cluster": cluster,
            "Plot": "Brain vs Genetics",
            "r_pearson": r_pearson,
            "rho_spearman": rho_s,
            "spearman_CI_low": ci_low,
            "spearman_CI_high": ci_high,
            "p_spearman_perm": p_perm,
            "LOO_min": np.nanmin(loo),
            "LOO_max": np.nanmax(loo),
            "N": len(sub)
        })

# ---------------- LEGENDA -----------------
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', label=cl,
                  markerfacecolor=CLUSTER_COLORS[cl], markersize=10) for cl in CLUSTER_ORDER]

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.legend(
    handles=handles,
    labels=CLUSTER_ORDER,
    loc='lower center',
    bbox_to_anchor=(0.5, 0),
    ncol=len(CLUSTER_ORDER),
    frameon=True,
    fontsize=12
)

# ---------------- SALVA FIGURA -----------------
plt.savefig("adults_adolescents_cortex_brain_vs_comorbidity_genetics.png", dpi=300)
print("Saved: adults_adolescents_cortex_brain_vs_comorbidity_genetics.png")

# ---------------- SALVA CSV -----------------
results_df = pd.DataFrame(results)
results_df.to_csv("cluster_stats_adults_adolescents_cortex.csv", index=False)
print("Saved cluster stats to cluster_stats_adults_adolescents_cortex.csv")