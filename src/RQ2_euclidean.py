#!/usr/bin/env python3
"""
Brain–phenotype association analyses with robustness statistics

Produces:
1) brain_vs_age_onset_euclidean_6plots.png
2) brain_vs_genetics_clusters_pooled_euclidean.png

Plots are identical to previous versions, but titles now include:
• Bootstrap 95% CI for Spearman ρ
• Permutation p-value (10,000)
• Leave-one-out (LOO) stability range

Now uses Euclidean Z-scores from RQ1 similarity pipeline
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.gridspec import GridSpec

# ==========================================================
# ---------------- ROBUST STAT FUNCTIONS -------------------
# ==========================================================

def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) >= 3:
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
    return np.nanmean(boot_r), *np.percentile(boot_r, [2.5, 97.5])


def permutation_spearman(x, y, n_perm=10000, seed=0, one_sided=False):
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    r_true = spearman_safe(x, y)
    perm_r = []

    for _ in range(n_perm):
        perm_r.append(spearman_safe(x, rng.permutation(y)))

    perm_r = np.array(perm_r)

    if one_sided:
        p = np.mean(perm_r >= r_true)
    else:
        p = np.mean(np.abs(perm_r) >= np.abs(r_true))

    return r_true, p


def leave_one_out_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    return np.array([
        spearman_safe(np.delete(x, i), np.delete(y, i))
        for i in range(len(x))
    ])

# ==========================================================
# ---------------- CONFIG & CLUSTERS -----------------------
# ==========================================================

BASE_DIR = r"C:\Users\giaco\Desktop\Git_META\META"

PSY_CLUSTERS = {
    "Psychotic": ["SCZ", "BD"],
    "Neurodevelopmental": ["ASD", "ADHD", "ADHD_ch", "ADHD_ado"],
    "AN/OCD": ["AN", "OCD"],
    "Mood/Anxiety": ["MDD", "PTSD"],
}
CLUSTER_ORDER = list(PSY_CLUSTERS.keys())

def assign_cluster(psy):
    for c, members in PSY_CLUSTERS.items():
        if psy in members:
            return c
    return None

# ==========================================================
# ========== PART 1: AGE OF ONSET (ADULTS) =================
# ==========================================================

AGE_ONSET_FILE = os.path.join(
    BASE_DIR, "data", "raw", "age_onset_prevalence_of_disorders.xlsx"
)

ADULT_DIRS = [
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all"),
]

# ---- Load age of onset ----
age_onset = pd.read_excel(AGE_ONSET_FILE, index_col=0)
age_onset = age_onset.applymap(
    lambda x: float(str(x).replace(",", ".").strip()) if pd.notnull(x) else np.nan
)

age_long = (
    age_onset.reset_index()
    .melt(id_vars="DISORDER", var_name="SUD", value_name="AgeOnset")
    .rename(columns={"DISORDER": "PSY"})
)

# ---- Load brain Euclidean Z-scores ----
measure = "euclidean"
records = []
for d in ADULT_DIRS:
    brain = pd.read_csv(os.path.join(d, f"Z_cortex_{measure}.csv"), index_col=0)
    records.append(
        brain.reset_index()
        .melt(id_vars="index", var_name="SUD", value_name="Z_euclidean")
        .rename(columns={"index": "PSY"})
    )

brain_all = pd.concat(records)
df_age = pd.merge(brain_all, age_long, on=["PSY", "SUD"], how="inner")
df_age["Cluster"] = df_age["PSY"].apply(assign_cluster)
df_age = df_age.dropna(subset=["Z_euclidean", "AgeOnset", "Cluster"])

# ---- Plot (IDENTICAL layout except removed extra panels) ----
plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, cluster in enumerate(CLUSTER_ORDER):
    ax = axes[i]
    sub = df_age[df_age["Cluster"] == cluster]
    x, y = sub["AgeOnset"].values, sub["Z_euclidean"].values

    ax.scatter(x, y, s=80, alpha=0.85, color="blue", edgecolor="k")

    if np.sum(np.isfinite(x)) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, m * xs + b, color="black", lw=2)

    r = spearman_safe(x, y)
    _, ci_l, ci_u = bootstrap_spearman(x, y)
    _, p_perm = permutation_spearman(
        x, y, one_sided=(cluster == "Psychotic")
    )
    loo = leave_one_out_spearman(x, y)

    ax.set_title(
        f"{cluster}\n"
        f"ρ={r:.2f}, 95% CI [{ci_l:.2f},{ci_u:.2f}], "
        f"perm p={p_perm:.3f}\n"
        f"LOO Δρ [{np.nanmin(loo):.2f},{np.nanmax(loo):.2f}], n={len(sub)}",
        fontsize=11
    )
    ax.set_xlabel("Comorbidity")
    ax.set_ylabel("Brain similarity (Z_euclidean)")

fig.suptitle("Brain–Comorbidity association (Adults)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("brain_vs_age_onset_euclidean_6plots.png", dpi=300)
print("Saved: brain_vs_age_onset_euclidean_6plots.png")

# ==========================================================
# ========== PART 1B: AGE OF ONSET (POOLED) =================
# ==========================================================

# ---- Load brain Euclidean Z-scores (Adults + Adolescents) ----
ALL_DIRS = [
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_ctx"),
]

records = []

for d in ALL_DIRS:
    age_group = "Adults" if "adults" in d else "Adolescents"

    brain = pd.read_csv(os.path.join(d, f"Z_cortex_{measure}.csv"), index_col=0)
    records.append(
        brain.reset_index()
        .melt(id_vars="index", var_name="SUD", value_name="Z_euclidean")
        .rename(columns={"index": "PSY"})
        .assign(AgeGroup=age_group)
    )

brain_all_pooled = pd.concat(records)

# ---- Map ADHD_ch and ADHD_ado to ADHD for age-of-onset merge ----
brain_all_pooled["PSY_onset"] = brain_all_pooled["PSY"].replace({
    "ADHD_ch": "ADHD",
    "ADHD_ado": "ADHD"
})

age_long["PSY_onset"] = age_long["PSY"]

df_age_pooled = pd.merge(
    brain_all_pooled,
    age_long,
    left_on=["PSY_onset", "SUD"],
    right_on=["PSY_onset", "SUD"],
    how="inner"
)

# Restore original PSY from brain data
df_age_pooled["PSY"] = df_age_pooled["PSY_x"]

df_age_pooled["Cluster"] = df_age_pooled["PSY"].apply(assign_cluster)
df_age_pooled = df_age_pooled.dropna(
    subset=["Z_euclidean", "AgeOnset", "Cluster"]
)

# ---- Plot ----
plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, cluster in enumerate(CLUSTER_ORDER):
    ax = axes[i]
    sub = df_age_pooled[df_age_pooled["Cluster"] == cluster]

    # ---- scatter by age group ----
    for age in ["Adults", "Adolescents"]:
        s = sub[sub["AgeGroup"] == age]
        color = "#1f77b4" if age == "Adults" else "#ff7f0e"

        ax.scatter(
            s["AgeOnset"],
            s["Z_euclidean"],
            s=80,
            alpha=0.85,
            color=color,
            edgecolor="k",
            label=age
        )

    ax.legend(frameon=False)

    x = sub["AgeOnset"].values
    y = sub["Z_euclidean"].values

    if np.sum(np.isfinite(x)) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, m * xs + b, color="black", lw=2)

    r = spearman_safe(x, y)
    _, ci_l, ci_u = bootstrap_spearman(x, y)
    _, p_perm = permutation_spearman(
        x, y, one_sided=(cluster == "Psychotic")
    )
    loo = leave_one_out_spearman(x, y)

    ax.set_title(
        f"{cluster} (Adults + Adolescents)\n"
        f"ρ={r:.2f}, 95% CI [{ci_l:.2f},{ci_u:.2f}], "
        f"perm p={p_perm:.3f}\n"
        f"LOO Δρ [{np.nanmin(loo):.2f},{np.nanmax(loo):.2f}], n={len(sub)}",
        fontsize=11
    )
    ax.set_xlabel("Comorbidity")
    ax.set_ylabel("Brain similarity (Z_euclidean)")

fig.suptitle(
    "Brain–Comorbidity association (Adults + Adolescents)",
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("brain_vs_age_onset_euclidean_pooled.png", dpi=300)
print("Saved: brain_vs_age_onset_euclidean_pooled.png")

# ==========================================================
# ========== PART 2: GENETICS (POOLED) =====================
# ==========================================================

GENETIC_FILE = os.path.join(
    BASE_DIR, "data", "raw", "PSY_SUD_genetic_corr.xlsx"
)

GROUPS = {
    "Adults": [
        os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all"),
    ],
    "Adolescents": [
        os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all"),
        os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_ctx"),
    ],
}

genetic = pd.read_excel(GENETIC_FILE, index_col=0)
gen_long = genetic.reset_index().melt(
    id_vars="index", var_name="SUD", value_name="genetic_corr"
).rename(columns={"index": "PSY"})

records = []
for age, dirs in GROUPS.items():
    mats = [
        pd.read_csv(os.path.join(d, f"Z_cortex_{measure}.csv"), index_col=0)
        for d in dirs
    ]
    brain = pd.concat(mats)
    records.append(
        brain.reset_index()
        .melt(id_vars="index", var_name="SUD", value_name="Z_euclidean")
        .rename(columns={"index": "PSY"})
        .assign(AgeGroup=age)
    )

brain_all = pd.concat(records)

# ---- FIX: Map ADHD_ch and ADHD_ado to ADHD for genetics merge ----
brain_all["PSY_genetic"] = brain_all["PSY"].replace({
    "ADHD_ch": "ADHD",
    "ADHD_ado": "ADHD"
})
gen_long["PSY_genetic"] = gen_long["PSY"]

df_gen = pd.merge(
    brain_all,
    gen_long,
    left_on=["PSY_genetic", "SUD"],
    right_on=["PSY_genetic", "SUD"],
    how="inner"
)

# Restore original PSY column from brain data
df_gen["PSY"] = df_gen["PSY_x"]

df_gen["Cluster"] = df_gen["PSY"].apply(assign_cluster)


plt.close("all")
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, figure=fig)

AGE_COLORS = {
    "Adults": "#1f77b4",
    "Adolescents": "#ff7f0e",
}

for i, cluster in enumerate(CLUSTER_ORDER):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    sub = df_gen[df_gen["Cluster"] == cluster]
    x, y = sub["genetic_corr"].values, sub["Z_euclidean"].values

    for age in ["Adults", "Adolescents"]:
        s = sub[sub["AgeGroup"] == age]
        ax.scatter(
            s["genetic_corr"],
            s["Z_euclidean"],
            s=80,
            alpha=0.85,
            color=AGE_COLORS[age],
            edgecolor="k",
            label=age,
        )

    ax.legend(frameon=False)

    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b, color="black", lw=2)

    r = spearman_safe(x, y)
    _, ci_l, ci_u = bootstrap_spearman(x, y)
    _, p_perm = permutation_spearman(
        x, y, one_sided=(cluster == "Psychotic")
    )
    loo = leave_one_out_spearman(x, y)

    ax.set_title(
        f"{cluster}\n"
        f"ρ={r:.2f}, 95% CI [{ci_l:.2f},{ci_u:.2f}], perm p={p_perm:.3f}\n"
        f"LOO Δρ [{np.nanmin(loo):.2f},{np.nanmax(loo):.2f}], n={len(sub)}",
        fontsize=11
    )
    ax.set_xlabel("Genetic correlation")
    ax.set_ylabel("Brain similarity (Z_euclidean)")

fig.suptitle(
    "Brain–genetic association by PSY cluster\nPooled adults + adolescents",
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("brain_vs_genetics_clusters_pooled_euclidean.png", dpi=300)
print("Saved: brain_vs_genetics_clusters_pooled_euclidean.png")

# ==========================================================
# ========== PART 3: GENETICS (ADULTS ONLY) =================
# ==========================================================

df_gen_adults = df_gen[df_gen["AgeGroup"] == "Adults"].copy()

plt.close("all")
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, figure=fig)

for i, cluster in enumerate(CLUSTER_ORDER):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    sub = df_gen_adults[df_gen_adults["Cluster"] == cluster]
    x, y = sub["genetic_corr"].values, sub["Z_euclidean"].values

    # ---- scatter ----
    ax.scatter(
        x,
        y,
        s=80,
        alpha=0.85,
        color="#1f77b4",
        edgecolor="k",
    )

    # ---- regression line ----
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="black", lw=2)

    # ---- stats ----
    r = spearman_safe(x, y)
    _, ci_l, ci_u = bootstrap_spearman(x, y)
    _, p_perm = permutation_spearman(
        x, y, one_sided=(cluster == "Psychotic")
    )
    loo = leave_one_out_spearman(x, y)

    ax.set_title(
        f"{cluster} (Adults only)\n"
        f"ρ={r:.2f}, 95% CI [{ci_l:.2f},{ci_u:.2f}], perm p={p_perm:.3f}\n"
        f"LOO Δρ [{np.nanmin(loo):.2f},{np.nanmax(loo):.2f}], n={len(sub)}",
        fontsize=11
    )
    ax.set_xlabel("Genetic correlation")
    ax.set_ylabel("Brain similarity (Z_euclidean)")

fig.suptitle(
    "Brain–genetic association by PSY cluster\nAdults only",
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("brain_vs_genetics_clusters_adults_only_euclidean.png", dpi=300)
print("Saved: brain_vs_genetics_clusters_adults_only_euclidean.png")

