#!/usr/bin/env python3
"""
Adults-only: brain similarity vs comorbidity and genetics
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

# Adults-only brain data
ADULT_DIRS = [os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")]
records = []
for d in ADULT_DIRS:
    brain = pd.read_csv(os.path.join(d, "Z_cortex_euclidean.csv"), index_col=0)
    records.append(brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean").rename(columns={"index": "PSY"}))
brain_all = pd.concat(records)
df_age = pd.merge(brain_all, age_long, on=["PSY", "SUD"], how="inner")
df_age["Cluster"] = df_age["PSY"].apply(assign_cluster)
df_age = df_age.dropna(subset=["Z_euclidean", "AgeOnset", "Cluster"])

# Genetics
GEN_FILE = os.path.join(BASE_DIR, "data", "raw", "PSY_SUD_genetic_corr.xlsx")
genetic = pd.read_excel(GEN_FILE, index_col=0)
gen_long = genetic.reset_index().melt(id_vars="index", var_name="SUD", value_name="genetic_corr").rename(columns={"index": "PSY"})
records_gen = []
for d in ADULT_DIRS:
    mats = [pd.read_csv(os.path.join(d, "Z_cortex_euclidean.csv"), index_col=0)]
    brain = pd.concat(mats)
    records_gen.append(brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean").rename(columns={"index": "PSY"}))
brain_all_gen = pd.concat(records_gen)
df_gen = pd.merge(brain_all_gen, gen_long, on=["PSY", "SUD"], how="inner")
df_gen["Cluster"] = df_gen["PSY"].apply(assign_cluster)

# ----------------- PLOTS & STATS -------------------------
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes = axes.flatten()

# Range globale con margini
def add_margin(vmin, vmax, margin=0.05):
    delta = vmax - vmin
    return vmin - delta*margin, vmax + delta*margin

age_xmin, age_xmax = add_margin(df_age["AgeOnset"].min(), df_age["AgeOnset"].max())
age_ymin, age_ymax = add_margin(df_age["Z_euclidean"].min(), df_age["Z_euclidean"].max())
gen_xmin, gen_xmax = add_margin(df_gen["genetic_corr"].min(), df_gen["genetic_corr"].max())
gen_ymin, gen_ymax = add_margin(df_gen["Z_euclidean"].min(), df_gen["Z_euclidean"].max())

results = []

from scipy.stats import spearmanr, pearsonr
# ---------------- PANEL 1: Age of onset -----------------
ax = axes[0]
age_texts = []
for i, cluster in enumerate(CLUSTER_ORDER):
    sub = df_age[df_age["Cluster"] == cluster]
    x, y = sub["AgeOnset"].values, sub["Z_euclidean"].values
    ax.scatter(x, y, s=80, alpha=0.6, color=CLUSTER_COLORS[cluster])
    
    if len(x) >= 2:
        # Pearson
        r_pearson = np.corrcoef(x, y)[0,1]
        
        # Spearman r bootstrap
        rho_s, ci_low, ci_high = bootstrap_spearman(x, y, n_boot=10000, seed=0)
        
        # Spearman p-value via permutation
        rho_perm, p_perm = permutation_spearman(x, y, n_perm=10000, seed=0)
        
        # Leave-one-out
        loo = leave_one_out_spearman(x, y)
        
        # Regressione per linea
        m, b = np.polyfit(x, y, 1)
        xs_data = np.linspace(x.min(), x.max(), 100)
        ys_data = m*xs_data + b
        ax.plot(xs_data, ys_data, color=CLUSTER_COLORS[cluster], lw=2)

        # Append risultati CSV
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

        # Testo in basso a dx (solo valori Pearson colorati cluster)
        age_texts.append((r_pearson, p_perm, CLUSTER_COLORS[cluster]))

ax.set_xlim(age_xmin, age_xmax)
ax.set_ylim(age_ymin, age_ymax)
ax.set_xlabel("Age of onset")
ax.set_ylabel("Brain similarity (Z_euclidean)")
ax.set_title("Adults: Brain vs Comorbidity")

# Testo in basso a dx (solo valori Pearson colorati cluster)
for j, cluster in enumerate(CLUSTER_ORDER):
    sub = df_age[df_age["Cluster"] == cluster]
    if len(sub) >= 2:
        x, y = sub["AgeOnset"].values, sub["Z_euclidean"].values
        r_pearson = np.corrcoef(x, y)[0,1]
        _, p_pearson = pearsonr(x, y)
        ax.text(
            0.98, 0.02 + j*0.05,
            f"r={r_pearson:.2f}, p={p_pearson:.3f}",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=10,
            color=CLUSTER_COLORS[cluster]
        )

# ---------------- PANEL 2: Genetics -----------------
ax = axes[1]
gen_texts = []
for i, cluster in enumerate(CLUSTER_ORDER):
    sub = df_gen[df_gen["Cluster"] == cluster]
    x, y = sub["genetic_corr"].values, sub["Z_euclidean"].values
    ax.scatter(x, y, s=80, alpha=0.6, color=CLUSTER_COLORS[cluster])
    
    if len(x) >= 2:
        r_pearson = np.corrcoef(x, y)[0,1]
        rho_s, ci_low, ci_high = bootstrap_spearman(x, y, n_boot=10000, seed=0)
        rho_perm, p_perm = permutation_spearman(x, y, n_perm=10000, seed=0)
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

        gen_texts.append((r_pearson, p_perm, CLUSTER_COLORS[cluster]))

ax.set_xlim(gen_xmin, gen_xmax)
ax.set_ylim(gen_ymin, gen_ymax)
ax.set_xlabel("Genetic correlation")
ax.set_ylabel("Brain similarity (Z_euclidean)")
ax.set_title("Adults: Brain vs Genetics")

for j, cluster in enumerate(CLUSTER_ORDER):
    sub = df_gen[df_gen["Cluster"] == cluster]
    if len(sub) >= 2:
        x, y = sub["genetic_corr"].values, sub["Z_euclidean"].values
        r_pearson = np.corrcoef(x, y)[0,1]
        _, p_pearson = pearsonr(x, y)
        ax.text(
            0.98, 0.02 + j*0.05,
            f"r={r_pearson:.2f}, p={p_pearson:.3f}",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=10,
            color=CLUSTER_COLORS[cluster]
        )

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
plt.savefig("adults_brain_vs_comorbidity_genetics_extlined.png", dpi=300)
print("Saved: adults_brain_vs_comorbidity_genetics_extlined.png")

# ---------------- SALVA CSV -----------------
results_df = pd.DataFrame(results)
results_df.to_csv("cluster_stats.csv", index=False)
print("Saved cluster stats to cluster_stats.csv")

# ==========================================================
#        GLOBAL FIGURE: 6 SUBPLOTS (ADULTS VS ADOLESCENTS)
# ==========================================================

print("\nGenerating combined 6-panel figure...")

# ----------------------------------------------------------
# LOAD ADOLESCENT DATA (ALL + CTX)
# ----------------------------------------------------------

ADOLESCENT_ALL_DIR = os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all")
ADOLESCENT_CTX_DIR = os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_ctx")

brain_adol_all = pd.read_csv(
    os.path.join(ADOLESCENT_ALL_DIR, "Z_cortex_euclidean.csv"),
    index_col=0
)

brain_adol_ctx = pd.read_csv(
    os.path.join(ADOLESCENT_CTX_DIR, "Z_cortex_euclidean.csv"),
    index_col=0
)

# Unisci adolescenti
brain_adol = pd.concat([brain_adol_all, brain_adol_ctx])

brain_adol_long = brain_adol.reset_index().melt(
    id_vars="index",
    var_name="SUD",
    value_name="Z_euclidean"
).rename(columns={"index": "PSY"})

# ----------------------------------------------------------
# FIX ADHD LABELS (ADHD_ch + ADHD_ado → ADHD)
# ----------------------------------------------------------

brain_adol_long["PSY"] = brain_adol_long["PSY"].replace({
    "ADHD_ch": "ADHD",
    "ADHD_ado": "ADHD"
})

# ----------------------------------------------------------
# DATASETS AGE
# ----------------------------------------------------------

df_age_adol = pd.merge(brain_adol_long, age_long, on=["PSY", "SUD"], how="inner")
df_age_adol = df_age_adol.dropna(subset=["Z_euclidean", "AgeOnset"])
df_age_adol["Group"] = "Adolescents"

df_age_adults = df_age.copy()
df_age_adults["Group"] = "Adults"

df_age_all = pd.concat([df_age_adults, df_age_adol])

# ----------------------------------------------------------
# DATASETS GENETICS
# ----------------------------------------------------------

df_gen_adol = pd.merge(brain_adol_long, gen_long, on=["PSY", "SUD"], how="inner")
df_gen_adol["Group"] = "Adolescents"

df_gen_adults = df_gen.copy()
df_gen_adults["Group"] = "Adults"

df_gen_all = pd.concat([df_gen_adults, df_gen_adol])

# ==========================================================
#                CREATE 6-SUBPLOT FIGURE
# ==========================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

global_results = []

def plot_panel(ax, df, x_col, title, label):

    # Scatter per group
    for group, color in zip(["Adults", "Adolescents"], ["blue", "orange"]):
        sub = df[df["Group"] == group]
        ax.scatter(sub[x_col], sub["Z_euclidean"],
                   s=60, alpha=0.7,
                   color=color,
                   label=group)

    # Global regression
    x = df[x_col].values
    y = df["Z_euclidean"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) >= 2:
        r, p = pearsonr(x, y)

        rho_s, ci_low, ci_high = bootstrap_spearman(x, y)
        rho_perm, p_perm = permutation_spearman(x, y)
        loo = leave_one_out_spearman(x, y)

        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m*xs + b, color="black", linewidth=2)

        ax.text(
            0.98, 0.02,
            f"r={r:.2f}, p={p:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10
        )

        global_results.append({
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
    ax.set_ylabel("Brain similarity (Z_euclidean)")


# ---------------- COMORBIDITY ----------------
plot_panel(axes[0], df_age_adults, "AgeOnset",
           "Adults: Comorbidity",
           "Adults_Comorbidity")

plot_panel(axes[1], df_age_adol, "AgeOnset",
           "Adolescents: Comorbidity",
           "Adolescents_Comorbidity")

plot_panel(axes[2], df_age_all, "AgeOnset",
           "Adults + Adolescents: Comorbidity",
           "Combined_Comorbidity")

# ---------------- GENETICS ----------------
plot_panel(axes[3], df_gen_adults, "genetic_corr",
           "Adults: Genetics",
           "Adults_Genetics")

plot_panel(axes[4], df_gen_adol, "genetic_corr",
           "Adolescents: Genetics",
           "Adolescents_Genetics")

plot_panel(axes[5], df_gen_all, "genetic_corr",
           "Adults + Adolescents: Genetics",
           "Combined_Genetics")

# Legend unica
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
plt.savefig("GLOBAL_6panel_brain_comorbidity_genetics.png", dpi=300)
plt.close()

print("Saved: GLOBAL_6panel_brain_comorbidity_genetics.png")

# ----------------------------------------------------------
# SAVE STATS
# ----------------------------------------------------------

pd.DataFrame(global_results).to_csv(
    "GLOBAL_6panel_correlation_stats.csv",
    index=False
)

print("Saved stats: GLOBAL_6panel_correlation_stats.csv")