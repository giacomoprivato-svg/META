#!/usr/bin/env python3
"""
Adults-only: brain similarity vs comorbidity and genetics
Scatter plots per cluster, OLS regression line with interaction,
asterisk for interaction significance, lines thick if slope significant,
all statistics saved in a single CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D

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
# Comorbidity (sostituire con indice reale se disponibile)
AGE_FILE = os.path.join(BASE_DIR, "data", "raw", "age_onset_prevalence_of_disorders.xlsx")
age_onset = pd.read_excel(AGE_FILE, index_col=0)
age_onset = age_onset.applymap(lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan)
age_long = age_onset.reset_index().melt(id_vars="DISORDER", var_name="SUD", value_name="Comorbidity")
age_long = age_long.rename(columns={"DISORDER": "PSY"})

# Adults-only brain data
ADULT_DIRS = [os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")]
records = []
for d in ADULT_DIRS:
    brain = pd.read_csv(os.path.join(d, "Z_cortex_euclidean.csv"), index_col=0)
    records.append(
        brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean")
        .rename(columns={"index": "PSY"})
    )
brain_all = pd.concat(records)
df_com = pd.merge(brain_all, age_long, on=["PSY", "SUD"], how="inner")
df_com["Cluster"] = df_com["PSY"].apply(assign_cluster)
df_com = df_com.dropna(subset=["Z_euclidean", "Comorbidity", "Cluster"])

# Genetics
GEN_FILE = os.path.join(BASE_DIR, "data", "raw", "PSY_SUD_genetic_corr.xlsx")
genetic = pd.read_excel(GEN_FILE, index_col=0)
gen_long = genetic.reset_index().melt(id_vars="index", var_name="SUD", value_name="genetic_corr").rename(columns={"index": "PSY"})
records_gen = []
for d in ADULT_DIRS:
    mats = [pd.read_csv(os.path.join(d, "Z_cortex_euclidean.csv"), index_col=0)]
    brain = pd.concat(mats)
    records_gen.append(
        brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean")
        .rename(columns={"index": "PSY"})
    )
brain_all_gen = pd.concat(records_gen)
df_gen = pd.merge(brain_all_gen, gen_long, on=["PSY", "SUD"], how="inner")
df_gen["Cluster"] = df_gen["PSY"].apply(assign_cluster)

# ----------------- PLOTS & STATS -------------------------
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes = axes.flatten()

# Range con margini
def add_margin(vmin, vmax, margin=0.05):
    delta = vmax - vmin
    return vmin - delta*margin, vmax + delta*margin

com_xmin, com_xmax = add_margin(df_com["Comorbidity"].min(), df_com["Comorbidity"].max())
com_ymin, com_ymax = add_margin(df_com["Z_euclidean"].min(), df_com["Z_euclidean"].max())
gen_xmin, gen_xmax = add_margin(df_gen["genetic_corr"].min(), df_gen["genetic_corr"].max())
gen_ymin, gen_ymax = add_margin(df_gen["Z_euclidean"].min(), df_gen["Z_euclidean"].max())

results = []

# ---------------- PANEL 1: Comorbidity -----------------
ax = axes[0]
ax.set_xlabel("Comorbidity")
ax.set_ylabel("Brain similarity (Z_euclidean)")
ax.set_title("Adults: Brain vs Comorbidity")

df_com["Cluster"] = pd.Categorical(df_com["Cluster"], categories=CLUSTER_ORDER)
model_com = smf.ols("Z_euclidean ~ Comorbidity * Cluster", data=df_com).fit()

for cluster in CLUSTER_ORDER:
    sub = df_com[df_com["Cluster"] == cluster]
    x, y = sub["Comorbidity"].values, sub["Z_euclidean"].values
    ax.scatter(x, y, s=80, alpha=0.6, color=CLUSTER_COLORS[cluster])
    
    if len(sub) >= 2:
        if cluster == CLUSTER_ORDER[0]:  # reference
            slope_coef = model_com.params["Comorbidity"]
            slope_p = model_com.pvalues["Comorbidity"]
        else:
            slope_coef = model_com.params["Comorbidity"] + model_com.params.get(f"Comorbidity:Cluster[T.{cluster}]", 0)
            slope_p = model_com.pvalues.get(f"Comorbidity:Cluster[T.{cluster}]", 1)
        linestyle = "-" if slope_p < 0.05 else "--"
        linewidth = 3 if slope_p < 0.05 else 1.5
        interaction_p = model_com.pvalues.get(f"Comorbidity:Cluster[T.{cluster}]", 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = slope_coef*xs + model_com.params["Intercept"] + (model_com.params.get(f"Cluster[T.{cluster}]", 0) if cluster != CLUSTER_ORDER[0] else 0)
        ax.plot(xs, ys, color=CLUSTER_COLORS[cluster], linestyle=linestyle, linewidth=linewidth)
        if interaction_p < 0.05:
            ax.text(xs[-1], ys[-1], "*", fontsize=14, color=CLUSTER_COLORS[cluster])
        results.append({
            "Cluster": cluster,
            "Plot": "Brain vs Comorbidity",
            "Slope": slope_coef,
            "Slope_p": slope_p,
            "Interaction_p": interaction_p,
            "N": len(sub)
        })

ax.set_xlim(com_xmin, com_xmax)
ax.set_ylim(com_ymin, com_ymax)

# ---------------- PANEL 2: Genetics -----------------
ax = axes[1]
ax.set_xlabel("Genetic correlation")
ax.set_ylabel("Brain similarity (Z_euclidean)")
ax.set_title("Adults: Brain vs Genetics")

df_gen["Cluster"] = pd.Categorical(df_gen["Cluster"], categories=CLUSTER_ORDER)
model_gen = smf.ols("Z_euclidean ~ genetic_corr * Cluster", data=df_gen).fit()

for cluster in CLUSTER_ORDER:
    sub = df_gen[df_gen["Cluster"] == cluster]
    x, y = sub["genetic_corr"].values, sub["Z_euclidean"].values
    ax.scatter(x, y, s=80, alpha=0.6, color=CLUSTER_COLORS[cluster])
    
    if len(sub) >= 2:
        if cluster == CLUSTER_ORDER[0]:
            slope_coef = model_gen.params["genetic_corr"]
            slope_p = model_gen.pvalues["genetic_corr"]
        else:
            slope_coef = model_gen.params["genetic_corr"] + model_gen.params.get(f"genetic_corr:Cluster[T.{cluster}]", 0)
            slope_p = model_gen.pvalues.get(f"genetic_corr:Cluster[T.{cluster}]", 1)
        linestyle = "-" if slope_p < 0.05 else "--"
        linewidth = 3 if slope_p < 0.05 else 1.5
        interaction_p = model_gen.pvalues.get(f"genetic_corr:Cluster[T.{cluster}]", 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = slope_coef*xs + model_gen.params["Intercept"] + (model_gen.params.get(f"Cluster[T.{cluster}]", 0) if cluster != CLUSTER_ORDER[0] else 0)
        ax.plot(xs, ys, color=CLUSTER_COLORS[cluster], linestyle=linestyle, linewidth=linewidth)
        if interaction_p < 0.05:
            ax.text(xs[-1], ys[-1], "*", fontsize=14, color=CLUSTER_COLORS[cluster])
        results.append({
            "Cluster": cluster,
            "Plot": "Brain vs Genetics",
            "Slope": slope_coef,
            "Slope_p": slope_p,
            "Interaction_p": interaction_p,
            "N": len(sub)
        })

ax.set_xlim(gen_xmin, gen_xmax)
ax.set_ylim(gen_ymin, gen_ymax)

# ---------------- LEGENDA UNICA -----------------
cluster_handles = [Line2D([0], [0], marker='o', color='w', label=cl,
                          markerfacecolor=CLUSTER_COLORS[cl], markersize=10) 
                    for cl in CLUSTER_ORDER]

line_handles = [
    Line2D([0], [0], color='k', lw=3, linestyle='-', label='Slope significant'),
    Line2D([0], [0], color='k', lw=1.5, linestyle='--', label='Slope non significant'),
]

# Handle per l'asterisco nero nella legenda
asterisk_handle = Line2D([0], [0], marker=r"$\ast$", color='black',  # marker = asterisco vero
                          markersize=6,
                          label='Interaction significant',
                          linestyle='None')
all_handles = cluster_handles + line_handles + [asterisk_handle]

# Spazio sotto i pannelli per la legenda
fig.subplots_adjust(bottom=0.2)

fig.legend(handles=all_handles,
           labels=[h.get_label() for h in all_handles],
           loc='lower center',
           ncol=len(all_handles),
           frameon=True,
           fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("adults_brain_vs_comorbidity_genetics_interaction.png", dpi=300)
print("Saved figure: adults_brain_vs_comorbidity_genetics_interaction.png")

# ---------------- SALVA CSV -----------------
results_df = pd.DataFrame(results)
results_df.to_csv("cluster_stats_interaction.csv", index=False)
print("Saved CSV: cluster_stats_interaction.csv")