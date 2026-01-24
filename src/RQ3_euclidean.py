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

OUTFIG_COMBINED = os.path.join(BASE_DIR, "overlap_vs_age_all_points_euclidean.png")

# Colors for disorders
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
    "CHR": "#17becf",
    "Schizotypy": "#17a2b8"
}

# Map variants to canonical names
PSY_NAME_MAP = {
    "ADHD": "ADHD",
    "ADHD_ch": "ADHD",
    "ADHD_ado": "ADHD",
    "ASD": "ASD",
    "AN": "AN",
    "OCD": "OCD",
    "BD": "BD",
    "MDD": "MDD",
    "SCZ": "SCZ",
    "CHR": "CHR",
    "Schizotypic": "Schizotypy",
    "CD": "CD",
    "PTSD": "PTSD"
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
        path = os.path.join(d, "Z_cortex_euclidean.csv")  # <--- changed to Euclidean Z-scores
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        brain = pd.read_csv(path, index_col=0)

        # Merge ADHD variants if requested
        if merge_adhd:
            adhd_cols = [c for c in brain.columns if c.startswith("ADHD")]
            for col in adhd_cols:
                temp = brain[[col]].copy()
                temp.rename(columns={col: "ADHD"}, inplace=True)
                temp_long = temp.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean")
                temp_long.rename(columns={"index": "PSY"}, inplace=True)
                records.append(temp_long)
            brain = brain.drop(columns=adhd_cols, errors="ignore")

        # Melt remaining columns
        brain_long = brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean")
        brain_long.rename(columns={"index": "PSY"}, inplace=True)
        records.append(brain_long)

    if len(records) == 0:
        return pd.DataFrame(columns=["PSY", "SUD", "Z_euclidean"])

    df = pd.concat(records, axis=0, ignore_index=True)

    # Map to canonical PSY names
    df["PSY_main"] = df["PSY"].replace(PSY_NAME_MAP)

    # Ensure numeric
    df["Z_euclidean"] = pd.to_numeric(df["Z_euclidean"], errors="coerce")
    return df

# Load adult and pediatric data
df_adult = load_brain_data(ADULT_DIRS)
df_ped = load_brain_data(PED_ADOL_DIRS, merge_adhd=True)

# -----------------------
# MERGE AGE OF ONSET
# -----------------------
def merge_age(df_brain):
    df = pd.merge(df_brain, age_onset, left_on="PSY_main", right_on="PSY", how="left")
    # keep rows even if SUD/Region has NaN, just drop missing similarity or age
    df = df.dropna(subset=["Z_euclidean", "AgeOnset"])
    return df

df_adult = merge_age(df_adult)
df_ped = merge_age(df_ped)

# Combine both datasets
df_combined = pd.concat([df_adult, df_ped], axis=0, ignore_index=True)

# -----------------------
# PLOTTING FUNCTION
# -----------------------
def plot_on_ax(ax, df, title):
    for psy_label in df["PSY_main"].unique():
        sub = df[df["PSY_main"] == psy_label]
        ax.scatter(sub["AgeOnset"], sub["Z_euclidean"], s=80, alpha=0.85,
                   color=PSY_COLORS.get(psy_label, "grey"), edgecolor="k")

    # Regression
    x = df["AgeOnset"].values
    y = df["Z_euclidean"].values
    mask = np.isfinite(x) & np.isfinite(y)

    p_lin_text = "NA"
    p_quad_text = "NA"

    rho_lin, rho_quad = np.nan, np.nan

    if np.sum(mask) >= 2 and np.ptp(x[mask]) > 0:
        # Linear fit
        m, b = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 100)
        ax.plot(xs, m*xs+b, color="black", lw=2, label="Linear fit")
        rho_lin, p_lin = spearmanr(x[mask], y[mask])
        p_lin_text = f"p < 0.001" if p_lin < 0.001 else f"p={p_lin:.3f}"

        # Quadratic fit
        if np.sum(mask) >= 3:
            p_quad = np.polyfit(x[mask], y[mask], 2)
            ax.plot(xs, np.polyval(p_quad, xs), color="black", lw=2, linestyle="--", alpha=0.3, label="Quadratic fit")
            y_quad = np.polyval(p_quad, x[mask])
            rho_quad, p_quad_val = spearmanr(x[mask], y_quad)
            p_quad_text = f"p < 0.001" if p_quad_val < 0.001 else f"p={p_quad_val:.3f}"
        else:
            rho_quad, p_quad_val = np.nan, np.nan

        print(f"{title} — Linear: ρ={rho_lin:.2f}, {p_lin_text}")
        print(f"{title} — Quadratic: ρ={rho_quad:.2f}, {p_quad_text}")

    # Add text inside axes, top-left corner
    ax.text(0.02, 0.95, 
            f"Linear: ρ={rho_lin:.2f}, {p_lin_text}\nQuad: ρ={rho_quad:.2f}, {p_quad_text}",
            transform=ax.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel("Age of onset")
    ax.set_ylabel("Neuroanatomical similarity (Z_euclidean)")
    ax.set_title(title)

# -----------------------
# CREATE FIGURE
# -----------------------
fig, ax = plt.subplots(1,1, figsize=(10,7))
plot_on_ax(ax, df_combined, "Adults + Pediatric/Adolescent (All Points)")

# Legend
handles = [Line2D([0], [0], marker='o', color='w', label=psy,
                  markerfacecolor=PSY_COLORS.get(psy,"grey"), markersize=8, markeredgecolor='k') for psy in PSY_COLORS.keys()]
handles += [Line2D([0], [0], color='black', lw=2, label='Linear fit'),
            Line2D([0], [0], color='black', lw=2, linestyle='--', alpha=0.3, label='Quadratic fit')]
fig.legend(handles=handles, title="PSY disorders", bbox_to_anchor=(0.5,-0.02), loc='upper center', ncol=6)

plt.tight_layout()
plt.savefig(OUTFIG_COMBINED, dpi=300, bbox_inches="tight")
print(f"Saved combined figure: {OUTFIG_COMBINED}")
