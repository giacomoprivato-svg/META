#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

# -----------------------
# CONFIG
# -----------------------
BASE_DIR = r"C:\Users\giaco\Desktop\Git_META\META"
AGE_ONSET_FILE = os.path.join(BASE_DIR, "data", "raw", "PSY_age_of_onset.xlsx")

ADULT_DIRS = [
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_ctx"),
]

PED_ADOL_DIRS = [
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_ctx"),
]

OUTFIG_COMBINED = "overlap_vs_age_3plots.png"

PSY_COLORS = {
    "SCZ": "#1f77b4",
    "BD": "#ff7f0e",
    "MDD": "#2ca02c",
    "PTSD": "#d62728",
    "ASD": "#9467bd",
    "ADHD": "#8c564b",
    "AN": "#e377c2",
    "OCD": "#7f7f7f",
    "CD": "#bcbd22",
    "CHR": "#17becf"
}

# -----------------------
# LOAD AGE OF ONSET
# -----------------------
age_onset = pd.read_excel(AGE_ONSET_FILE)
age_onset.rename(columns={"disorder": "PSY", "Peak age onset (yrs)": "AgeOnset"}, inplace=True)
age_onset["AgeOnset"] = age_onset["AgeOnset"].apply(lambda x: float(str(x).replace(",", ".")) if pd.notna(x) else np.nan)

# -----------------------
# LOAD BRAIN DATA
# -----------------------
def load_brain_data(dirs, merge_adhd=False):
    records = []
    for d in dirs:
        path = os.path.join(d, "Z_cortex_combined.csv")
        brain = pd.read_csv(path, index_col=0)

        # Merge ADHD columns if requested
        if merge_adhd:
            adhd_cols = [c for c in brain.columns if c.startswith("ADHD")]
            for col in adhd_cols:
                temp = brain[[col]].copy()
                temp.rename(columns={col: "ADHD"}, inplace=True)
                records.append(temp.reset_index().melt(id_vars="index", var_name="SUD", value_name="Similarity").rename(columns={"index": "PSY"}))
            brain = brain.drop(columns=adhd_cols, errors="ignore")

        # Melt the rest
        brain_long = brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Similarity")
        brain_long.rename(columns={"index": "PSY"}, inplace=True)
        records.append(brain_long)
    df = pd.concat(records, axis=0)
    df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
    return df

brain_adult = load_brain_data(ADULT_DIRS)
brain_ped = load_brain_data(PED_ADOL_DIRS, merge_adhd=True)

# -----------------------
# MERGE AGE OF ONSET
# -----------------------
def merge_age(df_brain):
    df = pd.merge(df_brain, age_onset, on="PSY", how="inner")
    df = df.dropna(subset=["Similarity", "AgeOnset"])
    return df

df_adult = merge_age(brain_adult)
df_ped = merge_age(brain_ped)

df_combined = pd.concat([df_adult, df_ped])
df_combined = df_combined[~df_combined["PSY"].isin(["ADHD","ASD"])]

# -----------------------
# PLOTTING FUNCTION FOR ONE AXIS
# -----------------------
def plot_on_ax(ax, df, title):
    # Scatter points
    for psy in df["PSY"].unique():
        sub = df[df["PSY"] == psy]
        ax.scatter(sub["AgeOnset"], sub["Similarity"], s=80, alpha=0.85,
                   color=PSY_COLORS.get(psy, "grey"), edgecolor="k")

    # Regression
    x = df["AgeOnset"].values
    y = df["Similarity"].values
    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) >= 2 and np.ptp(x[mask]) > 0:
        # Linear
        m, b = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 100)
        ax.plot(xs, m*xs+b, color="black", lw=2, label="Linear fit")
        rho_lin, p_lin = spearmanr(x[mask], y[mask])

        # Quadratic
        if np.sum(mask) >= 3:
            p_quad = np.polyfit(x[mask], y[mask], 2)
            ax.plot(xs, np.polyval(p_quad, xs), color="black", lw=2, linestyle="--", alpha=0.3, label="Quadratic fit")
            y_quad = np.polyval(p_quad, x[mask])
            rho_quad, p_quad_val = spearmanr(x[mask], y_quad)
        else:
            rho_quad, p_quad_val = np.nan, np.nan

        # Print Spearman to console
        print(f"{title} — Linear: ρ={rho_lin:.2f}, p={p_lin:.3f}")
        print(f"{title} — Quadratic: ρ={rho_quad:.2f}, p={p_quad_val:.3f}")

        # Spearman text in right margin
        ax.text(1.05, 0.5, f"Linear: ρ={rho_lin:.2f}\np={p_lin:.3f}\nQuad: ρ={rho_quad:.2f}\np={p_quad_val:.3f}",
                transform=ax.transAxes, fontsize=10, va='center', ha='left', bbox=dict(facecolor='white', alpha=0.6))

    ax.set_xlabel("Age of onset")
    ax.set_ylabel("Neuroanatomical similarity")
    ax.set_title(title)

# -----------------------
# CREATE FIGURE WITH 3 SUBPLOTS
# -----------------------
fig, axes = plt.subplots(1,3, figsize=(22,6))

plot_on_ax(axes[0], df_adult, "Adults")
plot_on_ax(axes[1], df_ped, "Pediatric/Adolescent")
plot_on_ax(axes[2], df_combined, "Combined (no ADHD/ASD)")

# Legend for disorders and fits
handles = [Line2D([0], [0], marker='o', color='w', label=psy,
                  markerfacecolor=PSY_COLORS.get(psy,"grey"), markersize=8) for psy in PSY_COLORS.keys()]
handles += [Line2D([0], [0], color='black', lw=2, label='Linear fit'),
            Line2D([0], [0], color='black', lw=2, linestyle='--', alpha=0.3, label='Quadratic fit')]
fig.legend(handles=handles, title="PSY disorders", bbox_to_anchor=(0.5,-0.01), loc='upper center', ncol=6)

plt.tight_layout()
plt.savefig(OUTFIG_COMBINED, dpi=300, bbox_inches="tight")
print(f"Saved combined figure: {OUTFIG_COMBINED}")
