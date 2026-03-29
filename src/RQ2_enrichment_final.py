#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# ---------------- CONFIG ----------------
BASE_DIR = r"C:\Users\giaco\Desktop\Git_META\META"
ADULT_DIRS = [os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")]

K = 10
K_MAX = 32
N_PERM = 10000
SEED = 1
rng = np.random.default_rng(SEED)

# ---------------- LOAD DATA ----------------
def load_brain():
    rec = []
    for d in ADULT_DIRS:
        brain = pd.read_csv(os.path.join(d, "Z_cortex_euclidean.csv"), index_col=0)
        rec.append(
            brain.reset_index()
            .melt(id_vars="index", var_name="SUD", value_name="brain_similarity")
            .rename(columns={"index": "PSY"})
        )
    return pd.concat(rec, ignore_index=True)

def load_comorb():
    age = pd.read_excel(os.path.join(BASE_DIR, "data/raw/age_onset_prevalence_of_disorders.xlsx"), index_col=0)
    if "SUD" in age.columns:
        age = age.drop(columns=["SUD"])
    age = age.applymap(lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan)
    return (
        age.reset_index()
        .melt(id_vars="DISORDER", var_name="SUD", value_name="Comorbidity")
        .rename(columns={"DISORDER": "PSY"})
    )

def load_gen():
    gen = pd.read_excel(os.path.join(BASE_DIR, "data/raw/PSY_SUD_genetic_corr.xlsx"), index_col=0)
    return (
        gen.reset_index()
        .melt(id_vars="index", var_name="SUD", value_name="genetic_corr")
        .rename(columns={"index": "PSY"})
    )

# ---------------- MERGE DATA ----------------
df = load_brain()
df = df.merge(load_comorb(), on=["PSY","SUD"])
df = df.merge(load_gen(), on=["PSY","SUD"])
df["pair_id"] = df["PSY"] + "-" + df["SUD"]

# ---------------- COMORBIDITY FRACTION ----------------
df["Comorbidity_frac"] = df["Comorbidity"] / 100

# ---------------- EXPECTED COMORBIDITY ----------------
prev = pd.read_excel(os.path.join(BASE_DIR, "data/raw/SUD_general_prevalence.xlsx"))
prev_long = prev.melt(var_name="SUD", value_name="prev")
prev_long["prev"] = prev_long["prev"] / 100
prev_dict = dict(zip(prev_long["SUD"], prev_long["prev"]))
df["expected_comorbidity"] = df["SUD"].map(prev_dict)

# ---------------- Z-SCORE POISSON ----------------
df["comorbidity_z"] = (df["Comorbidity_frac"] - df["expected_comorbidity"]) / np.sqrt(df["expected_comorbidity"])

# ---------------- CLUSTERS ----------------
clusters = {
    'Psychotic': ['SCZ','BD','CHR'],
    'Neurodevelopmental': ['ASD','ADHD'],
    'AN/OCD': ['AN','OCD'],
    'Mood/Anxiety': ['MDD','PTSD']
}

def assign_cluster(psy):
    for k,v in clusters.items():
        if psy in v:
            return k
    return "Other"

df["cluster"] = df["PSY"].apply(assign_cluster)

colors = {
    'Psychotic': 'orange',
    'Neurodevelopmental': 'green',
    'AN/OCD': 'blue',
    'Mood/Anxiety': 'purple',
    'Other': 'gray'
}

# ---------------- COLORS PANEL C ----------------
pastel_colors = {
    "Comorbidity": "#336699",
    "Genetics": "#FF3333"
}

# ---------------- HELPERS ----------------
def get_topk(d, col, k):
    return set(d.nlargest(min(k,len(d)), col)["pair_id"])

def compute_null(d, col, k):
    ids = d["pair_id"].values
    brain_vals = d["brain_similarity"].values
    top_other = get_topk(d, col, k)
    null = []
    for _ in range(N_PERM):
        perm = rng.permutation(brain_vals)
        idx = np.argsort(-perm)[:min(k,len(d))]
        top_brain = set(ids[idx])
        null.append(len(top_brain & top_other))
    return np.array(null)

# ---------------- SCATTER PLOT ----------------
def plot_scatter(ax, d, x, y, title, K):
    d = d.dropna(subset=[x,y])
    topx = get_topk(d, x, K)
    topy = get_topk(d, y, K)
    overlap_ids = topx & topy

    # rettangolo celeste corretto solo top-K
    topx_vals = d[d["pair_id"].isin(topx)][x]
    topy_vals = d[d["pair_id"].isin(topy)][y]
    x_min, x_max = topx_vals.min(), topx_vals.max()
    y_min, y_max = topy_vals.min(), topy_vals.max()
    rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, color='lightblue', alpha=0.2, zorder=0)
    ax.add_patch(rect)

    # punti cluster
    for c in d["cluster"].unique():
        sub = d[d["cluster"] == c]
        ax.scatter(sub[x], sub[y], color=colors[c], alpha=0.6, label=c)

    # punti overlap
    overlap = d[d["pair_id"].isin(overlap_ids)]
    ax.scatter(overlap[x], overlap[y],
               c=[colors[c] for c in overlap["cluster"]],
               edgecolor='black', s=100, linewidth=1.5)

    # dashed line top-K
    ax.axvline(x_min, linestyle="--")
    ax.axhline(y_min, linestyle="--")

    ax.set_xlabel(x)
    ax.set_ylabel(f"{y} (Z-score Poisson)" if y=="comorbidity_z" else y)
    ax.set_title(title)

# ---------------- CURVES ----------------
def compute_curve(d, col):
    total_pairs = len(d)
    ks = np.arange(1, min(K_MAX,total_pairs)+1)
    obs, low, high = [], [], []
    for k in ks:
        top_b = get_topk(d, "brain_similarity", k)
        top_o = get_topk(d, col, k)
        obs.append(len(top_b & top_o))
        null = compute_null(d, col, k)
        low.append(np.percentile(null, 5))
        high.append(np.percentile(null, 95))
    return ks, np.array(obs), np.array(low), np.array(high)

# ---------------- DATASETS ----------------
d_com = df.dropna(subset=["brain_similarity","comorbidity_z"])
d_gen = df.dropna(subset=["brain_similarity","genetic_corr"])

# ---------------- FIGURE ----------------
fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(2,2, height_ratios=[1,1.2], hspace=0.4)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,:])

# Panel A: Brain vs Comorbidity
plot_scatter(ax1, d_com, "brain_similarity", "comorbidity_z", "A. Brain vs Comorbidity", K)

# Panel B: Brain vs Genetics
plot_scatter(ax2, d_gen, "brain_similarity", "genetic_corr", "B. Brain vs Genetics", K)

# Legend cluster condivisa sotto A/B
legend_elements = [Patch(facecolor=colors[c], label=c) for c in clusters.keys()]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))

# Panel C: rank overlap
ks_c, obs_c, low_c, high_c = compute_curve(d_com, "comorbidity_z")
ks_g, obs_g, low_g, high_g = compute_curve(d_gen, "genetic_corr")

ax3.plot(ks_c, obs_c, color=pastel_colors["Comorbidity"], label="Comorbidity", linewidth=2)
ax3.plot(ks_g, obs_g, color=pastel_colors["Genetics"], label="Genetics", linewidth=2)

ax3.fill_between(ks_c, low_c, high_c, color=pastel_colors["Comorbidity"], alpha=0.3)
ax3.fill_between(ks_g, low_g, high_g, color=pastel_colors["Genetics"], alpha=0.3)

ax3.set_xlabel("Top-K threshold")
ax3.set_ylabel("Overlap (absolute)")
ax3.set_title("C. Rank overlap")
ax3.legend()

plt.tight_layout()
plt.savefig("FINAL_FIGURE_ZSCORE_PANELCUTOFF_CORRECTED.png", dpi=300)
plt.show()