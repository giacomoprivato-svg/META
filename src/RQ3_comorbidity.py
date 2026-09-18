#!/usr/bin/env python3
"""
RQ3 — standalone comorbidity scatter (single panel)

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, linregress
import statsmodels.api as sm

# ================= TOGGLES =================
SHOW_FIT   = True     # OLS line + CI band, for visual parity with genetics panel
SHOW_STATS = True     # print rho, 95% CI and n in the corner
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir   = os.path.dirname(script_dir)
BASE_DIR   = repo_dir
METRIC       = "spearman"
METRIC_LABEL = "Spearman ρ (morphometric similarity)"
ADULT_DIR = os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")
COMORB_FILE = os.path.join(BASE_DIR, "data", "raw", "PSY_SUD_comorbidity_prevalence.xlsx")
OUT_DIR   = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- clusters (Mood/Anxiety = MDD, PD; PTSD removed) ------------------------
colors = {"Psychotic": "orange", "Neurodevelopmental": "green",
          "AN/OCD": "blue", "Mood/Anxiety": "purple"}
def assign_cluster(x):
    if x in ["SCZ", "BD", "CHR"]: return "Psychotic"
    if x in ["ASD", "ADHD"]:      return "Neurodevelopmental"
    if x in ["AN", "OCD"]:        return "AN/OCD"
    if x in ["MDD", "PD"]:        return "Mood/Anxiety"
    return "Other"

def _key(s):  # normalise labels for a safe merge
    return s.astype(str).str.strip().str.upper()

# ---- brain similarity (Spearman), identical loader to the genetics script ---
def load_brain():
    fp = os.path.join(ADULT_DIR, f"RAW_cortex_{METRIC}.csv")
    brain = pd.read_csv(fp, index_col=0, sep=None, engine="python")
    brain.index.name = "PSY"
    return brain.reset_index().melt(id_vars="PSY", var_name="SUD",
                                    value_name="brain_similarity")

# ---- raw comorbidity prevalence (long format: PSY, SUD, Comorbidity_prevalence)
def load_comorb():
    com = pd.read_excel(COMORB_FILE)
    # tolerate a 'Comorbidity' header too
    val = "Comorbidity_prevalence" if "Comorbidity_prevalence" in com.columns else "Comorbidity"
    com = com.rename(columns={val: "prevalence"})
    return com[["PSY", "SUD", "prevalence"]]

# ---- build the per-pair table ----------------------------------------------
brain = load_brain()
com   = load_comorb()
brain["_P"], brain["_S"] = _key(brain["PSY"]), _key(brain["SUD"])
com["_P"],   com["_S"]   = _key(com["PSY"]),   _key(com["SUD"])
d = brain.merge(com[["_P", "_S", "prevalence"]], on=["_P", "_S"], how="inner")
d = d.dropna(subset=["brain_similarity", "prevalence"])
d["cluster"] = d["PSY"].apply(assign_cluster)
n = len(d)
print(f"[diag] comorbidity pairs merged: n = {n}")
print(d[["PSY", "SUD", "brain_similarity", "prevalence"]].to_string(index=False))

# ---- Spearman rho + Fisher-z 95% CI (no p, small n) ------------------------
rho, _p = spearmanr(d["brain_similarity"], d["prevalence"])
z  = np.arctanh(rho)
se = 1.0 / np.sqrt(n - 3)
ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
print(f"[result] Spearman rho = {rho:+.2f}  95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]  n = {n}")

# ---- plot (one panel, 8x5, genetics-panel style) ---------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for c in d["cluster"].unique():
    sub = d[d["cluster"] == c]
    ax.scatter(sub["brain_similarity"], sub["prevalence"], alpha=0.6, s=55,
               color=colors.get(c, "gray"))

if SHOW_FIT and n >= 3:
    x = d["brain_similarity"].values
    y = d["prevalence"].values
    xg = np.linspace(x.min(), x.max(), 200)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    pred = model.get_prediction(sm.add_constant(xg))
    ax.plot(xg, pred.predicted_mean, color="black", lw=2)
    ci = pred.conf_int()
    ax.fill_between(xg, ci[:, 0], ci[:, 1], color="black", alpha=0.15)

if SHOW_STATS:
    ax.text(0.02, 0.98,
            f"ρ = {rho:+.2f}\n95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]\nn = {n}",
            transform=ax.transAxes, va="top", fontsize=16)

ax.set_xlabel(METRIC_LABEL, fontsize=16)
ax.set_ylabel("Comorbidity prevalence (%)", fontsize=16)
ax.tick_params(axis="both", labelsize=14)

legend_elements = [Patch(facecolor=colors[c], label=c) for c in colors]
ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.20),
          ncol=4, fontsize=12, handlelength=1.2, handleheight=1.2, frameon=False)

plt.tight_layout()
stem = os.path.join(OUT_DIR, "RQ3_comorbidity_scatter")
fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
fig.savefig(stem + ".pdf", bbox_inches="tight")
print("Saved ->", stem + ".[png|pdf]")