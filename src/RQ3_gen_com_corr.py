#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy.stats import linregress
import statsmodels.api as sm

# ---------------------------
# Paths (repo-relative)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)


# ---------------- CONFIG ----------------
BASE_DIR = repo_dir
ADULT_DIRS = [os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")]

K = 10
SEED = 1
rng = np.random.default_rng(SEED)

# ---------------- FORMAT P ----------------
def format_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3g}"

# ---------------- LOAD ----------------
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
    age = age.drop(columns=["SUD"], errors="ignore")
    age = age.applymap(lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan)
    return age.reset_index().melt(id_vars="DISORDER", var_name="SUD", value_name="Comorbidity") \
              .rename(columns={"DISORDER": "PSY"})

def load_gen():
    gen = pd.read_excel(os.path.join(BASE_DIR, "data/raw/PSY_SUD_genetic_corr.xlsx"), index_col=0)
    return gen.reset_index().melt(id_vars="index", var_name="SUD", value_name="genetic_corr") \
              .rename(columns={"index": "PSY"})

# ---------------- MERGE ----------------
df = load_brain()
df = df.merge(load_comorb(), on=["PSY","SUD"])
df = df.merge(load_gen(), on=["PSY","SUD"])

df["pair_id"] = df["PSY"] + "-" + df["SUD"]
df["Comorbidity_frac"] = df["Comorbidity"] / 100

# expected prevalence
prev = pd.read_excel(os.path.join(BASE_DIR, "data/raw/SUD_general_prevalence.xlsx"))
prev_long = prev.melt(var_name="SUD", value_name="prev")
prev_long["prev"] = prev_long["prev"] / 100
df["expected_comorbidity"] = df["SUD"].map(dict(zip(prev_long["SUD"], prev_long["prev"])))

df["ARD"] = df["Comorbidity_frac"] - df["expected_comorbidity"]

# ---------------- CLUSTERS ----------------
colors = {
    'Psychotic': 'orange',
    'Neurodevelopmental': 'green',
    'AN/OCD': 'blue',
    'Mood/Anxiety': 'purple',
}

def assign_cluster(x):
    if x in ["SCZ","BD","CHR"]: return "Psychotic"
    if x in ["ASD","ADHD"]: return "Neurodevelopmental"
    if x in ["AN","OCD"]: return "AN/OCD"
    if x in ["MDD","PTSD"]: return "Mood/Anxiety"
    return "Other"

df["cluster"] = df["PSY"].apply(assign_cluster)

# ---------------- HELPERS ----------------
def get_topk(d, col, k):
    return set(d.nlargest(min(k,len(d)), col)["pair_id"])

# ---------------- SCATTER OVERLAP ----------------
def plot_scatter(ax, d, x, y):

    d = d.dropna(subset=[x,y])

    topx = get_topk(d, x, K)
    topy = get_topk(d, y, K)
    overlap_ids = topx & topy

    topx_vals = d[d["pair_id"].isin(topx)][x]
    topy_vals = d[d["pair_id"].isin(topy)][y]

    x_min, x_max = topx_vals.min(), topx_vals.max()
    y_min, y_max = topy_vals.min(), topy_vals.max()

    rect = Rectangle((x_min-0.15, y_min),
                     x_max-x_min+0.30, y_max-y_min,
                     color='lightblue', alpha=0.2, zorder=0)
    ax.add_patch(rect)

    for c in d["cluster"].unique():
        sub = d[d["cluster"] == c]
        ax.scatter(sub[x], sub[y], color=colors[c], alpha=0.6)

    overlap = d[d["pair_id"].isin(overlap_ids)]
    ax.scatter(overlap[x], overlap[y],
               c=[colors[c] for c in overlap["cluster"]],
               edgecolor='black', s=100, linewidth=1.5)

    ax.plot([x_min-0.15, x_max+0.15], [y_min, y_min], color='black')
    ax.plot([x_min-0.15, x_max+0.15], [y_max, y_max], color='black')
    ax.plot([x_min-0.15, x_min-0.15], [y_min, y_max], color='black')
    ax.plot([x_max+0.15, x_max+0.15], [y_min, y_max], color='black')

    ax.axvline(x_min-0.15, linestyle='--', color='gray', alpha=0.5, zorder=-1)
    ax.axvline(x_max+0.15, linestyle='--', color='gray', alpha=0.5, zorder=-1)
    ax.axhline(y_min, linestyle='--', color='gray', alpha=0.5, zorder=-1)
    ax.axhline(y_max, linestyle='--', color='gray', alpha=0.5, zorder=-1)

    ax.set_xlabel("Euclidean Similarity", fontsize=16)

    if y == "ARD":
        ax.set_ylabel("Comorbidity (ARD)", fontsize=16)
    else:
        ax.set_ylabel("Genetic Correlation", fontsize=16)

    # TICKS BIGGER
    ax.tick_params(axis='both', labelsize=14)

# ---------------- SIMPLE SCATTER ----------------
def scatter(ax, d, x, y):

    for c in d["cluster"].unique():
        sub = d[d["cluster"] == c]
        ax.scatter(sub[x], sub[y], alpha=0.35, s=40, color=colors[c])

    ax.set_xlabel("Euclidean Similarity", fontsize=16)

    if y == "ARD":
        ax.set_ylabel("Comorbidity (ARD)", fontsize=16)
    else:
        ax.set_ylabel("Genetic Correlation", fontsize=16)

    # TICKS BIGGER
    ax.tick_params(axis='both', labelsize=14)

# =========================================================
# DATA
# =========================================================
d_com = df.dropna(subset=["brain_similarity","ARD"])
d_gen = df.dropna(subset=["brain_similarity","genetic_corr"])

# =========================================================
# FIGURE 2x2
# =========================================================
fig, axes = plt.subplots(2,2, figsize=(16,10))
ax2, ax1, ax4, ax3 = axes.flatten()

# A — overlap ARD
plot_scatter(ax1, d_com, "brain_similarity", "ARD")

# B — linear ARD
scatter(ax2, d_com, "brain_similarity", "ARD")

x = d_com["brain_similarity"].values
y = d_com["ARD"].values

slope, intercept, r, p_lin, _ = linregress(x, y)
xg = np.linspace(x.min(), x.max(), 200)

X_lin = sm.add_constant(x)
model_lin = sm.OLS(y, X_lin).fit()

pred_lin = model_lin.get_prediction(sm.add_constant(xg))
mean_lin = pred_lin.predicted_mean
ci_lin = pred_lin.conf_int()

ax2.plot(xg, mean_lin, color="black", lw=2)
ax2.fill_between(xg, ci_lin[:,0], ci_lin[:,1], color="black", alpha=0.15)

ax2.text(0.02, 0.98, f"r = {r:.2f}\n{format_p(p_lin)}",
         transform=ax2.transAxes, va="top", fontsize=16)

# C — overlap genetics
plot_scatter(ax3, d_gen, "brain_similarity", "genetic_corr")

# D — quadratic genetics
scatter(ax4, d_gen, "brain_similarity", "genetic_corr")

x = d_gen["brain_similarity"].values
y = d_gen["genetic_corr"].values

x_z = (x - x.mean()) / x.std()
xg = np.linspace(x.min(), x.max(), 200)
xg_z = (xg - x.mean()) / x.std()

X = sm.add_constant(np.column_stack([x_z, x_z**2]))
model = sm.OLS(y, X).fit()

pred = model.get_prediction(sm.add_constant(np.column_stack([xg_z, xg_z**2])))
mean = pred.predicted_mean
ci = pred.conf_int()

ax4.plot(xg, mean, color="black", lw=2)
ax4.fill_between(xg, ci[:,0], ci[:,1], color="black", alpha=0.15)

ax4.text(0.02, 0.98,
         f"R² = {model.rsquared:.2f}\n{format_p(model.pvalues[2])}",
         transform=ax4.transAxes, va="top", fontsize=16)

# legend
legend_elements = [Patch(facecolor=colors[c], label=c) for c in colors.keys()]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=5,
    bbox_to_anchor=(0.5, -0.005),
    fontsize=16,
    handlelength=1.5,
    handleheight=1.5
)

plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig("FINAL_FIGURE_4PANELS.png", dpi=300)
