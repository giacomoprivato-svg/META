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
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")
]

PED_ADOL_DIRS = [
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_ctx"),
]

OUTFIG_COMBINED = os.path.join(BASE_DIR, "overlap_vs_age_all_points.png")

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
    "CHR": "#17becf"
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
    "CD": "CD",
    "PTSD": "PTSD"
}

# -----------------------
# LOAD AGE OF ONSET
# -----------------------
age_onset = pd.read_excel(AGE_ONSET_FILE)

age_onset.rename(columns={
    "disorder": "PSY",
    "median": "AgeOnset",
    "p25": "P25",
    "p75": "P75",
    "Peak age onset (yrs)": "Peak"
}, inplace=True)

for col in ["AgeOnset", "P25", "P75", "Peak"]:
    age_onset[col] = age_onset[col].apply(
        lambda x: float(str(x).replace(",", ".")) if pd.notna(x) else np.nan
    )

# -----------------------
# LOAD BRAIN DATA
# -----------------------
def load_brain_data(dirs, merge_adhd=False):
    records = []
    for d in dirs:
        path = os.path.join(d, "Z_cortex_euclidean.csv")
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
                temp_long = temp.reset_index().melt(id_vars="index", var_name="SUD", value_name="Similarity")
                temp_long.rename(columns={"index": "PSY"}, inplace=True)
                records.append(temp_long)
            brain = brain.drop(columns=adhd_cols, errors="ignore")

        # Melt remaining columns
        brain_long = brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Similarity")
        brain_long.rename(columns={"index": "PSY"}, inplace=True)
        records.append(brain_long)

    if len(records) == 0:
        return pd.DataFrame(columns=["PSY", "SUD", "Similarity"])

    df = pd.concat(records, axis=0, ignore_index=True)

    # Map to canonical PSY names
    df["PSY_main"] = df["PSY"].replace(PSY_NAME_MAP)

    # Ensure numeric
    df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
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
    df = df.dropna(subset=["Similarity", "AgeOnset"])
    return df

df_adult = merge_age(df_adult)
df_ped = merge_age(df_ped)

# Combine both datasets
df_combined = pd.concat([df_adult, df_ped], axis=0, ignore_index=True)

# -----------------------
# PLOTTING FUNCTION WITH PVALUE TEXT TOP-LEFT
# -----------------------
def plot_on_ax(ax, df, title):
    for psy_label in df["PSY_main"].unique():
        sub = df[df["PSY_main"] == psy_label]
        ax.scatter(sub["AgeOnset"], sub["Similarity"], s=80, alpha=0.85,
                   color=PSY_COLORS.get(psy_label, "grey"), edgecolor="k")

    # Regression
    x = df["AgeOnset"].values
    y = df["Similarity"].values
    mask = np.isfinite(x) & np.isfinite(y)

    p_lin_text = "NA"
    p_quad_text = "NA"

    if np.sum(mask) >= 2 and np.ptp(x[mask]) > 0:
        # Linear fit
        m, b = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 100)
        ax.plot(xs, m*xs+b, color="black", lw=2, linestyle="--", alpha=0.3,label="Linear fit")
        rho_lin, p_lin = spearmanr(x[mask], y[mask])
        p_lin_text = f"p < 0.001" if p_lin < 0.001 else f"p={p_lin:.3f}"

        # Quadratic fit
        if np.sum(mask) >= 3:
            p_quad = np.polyfit(x[mask], y[mask], 2)
            ax.plot(xs, np.polyval(p_quad, xs), color="black", lw=2, label="Quadratic fit")
            y_quad = np.polyval(p_quad, x[mask])
            rho_quad, p_quad_val = spearmanr(x[mask], y_quad)
            p_quad_text = f"p < 0.001" if p_quad_val < 0.001 else f"p={p_quad_val:.3f}"
        else:
            rho_quad, p_quad_val = np.nan, np.nan

        # Print to console
        print(f"{title} — Linear: ρ={rho_lin:.2f}, {p_lin_text}")
        print(f"{title} — Quadratic: ρ={rho_quad:.2f}, {p_quad_text}")

    # Add text inside axes, top-left corner
    ax.text(0.98, 0.02, 
            f"Linear: ρ={rho_lin:.2f}, {p_lin_text}\nQuad: ρ={rho_quad:.2f}, {p_quad_text}",
            transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_ylabel("Neuroanatomical similarity")
    ax.set_title(title)

fig = plt.figure(figsize=(12,10))

# Scatter: 7-35
x_min_scatter, x_max_scatter = 5, 50

# IQR: 7-50
x_min_iqr, x_max_iqr = 5, 50

# Coordinata left della figura
left = 0.1

# Larghezza fisica della parte 7-35 (devono combaciare)
width_shared = 0.65  # scelta arbitraria, per scatter
# Larghezza totale asse IQR (proporzionale al range totale)
width_iqr = width_shared * (x_max_iqr - x_min_iqr) / (x_max_scatter - x_min_scatter)

# -----------------------
# SCATTER PANEL
# -----------------------
ax_scatter = fig.add_axes([left, 0.35, width_shared, 0.6])
plot_on_ax(ax_scatter, df_combined, "Neuroanatomical Similarity: Relation to Age of Onset")
ax_scatter.set_xlim(x_min_scatter, x_max_scatter)
ax_scatter.set_xticks([])
ax_scatter.set_xlabel('')
ax_scatter.tick_params(axis='x', which='both', length=0)

# Legenda aggiornata
handles = []

# PSY disorders: pallino colorato senza contorno
for psy in PSY_COLORS.keys():
    handles.append(
        Line2D([0], [0], marker='o', color='w', label=psy,
               markerfacecolor=PSY_COLORS[psy],
               markersize=8, markeredgecolor='none')
    )

# Median marker: cerchio con contorno nero
handles.append(
    Line2D([0], [0], marker='o', color='w', label='Median',
           markerfacecolor='white', markeredgecolor='black',
           markersize=8, markeredgewidth=1.5)
)

# Peak marker: triangolo con contorno nero
handles.append(
    Line2D([0], [0], marker='^', color='w', label='Peak',
           markerfacecolor='white', markeredgecolor='black',
           markersize=8, markeredgewidth=1.5)
)

# Linear & Quadratic fit
handles += [
    Line2D([0], [0], color='black', lw=2, linestyle='--', alpha=0.3, label='Linear fit'),
    Line2D([0], [0], color='black', lw=2, label='Quadratic fit')
]

# Legenda dentro l'asse, lato destro, centrata verticalmente
ax_scatter.legend(
    handles=handles,
    loc='center right',
    frameon=True,
    borderpad=1.0,
    labelspacing=0.5,
    handlelength=1.5,
    handletextpad=0.8,
    bbox_to_anchor=(0.98, 0.5)
)


# -----------------------
# IQR PANEL
# -----------------------
ax_iqr = fig.add_axes([left, 0.05, width_iqr, 0.25])  # inizia nello stesso left, width maggiore
ax_iqr.set_xlim(x_min_iqr, x_max_iqr)

age_plot = age_onset.sort_values("AgeOnset")
y_positions = np.arange(len(age_plot))

for i, (_, row) in enumerate(age_plot.iterrows()):
    psy = row["PSY"]
    color = PSY_COLORS.get(psy, "grey")
    ax_iqr.hlines(y=i, xmin=row["P25"], xmax=row["P75"], color=color, linewidth=6, alpha=0.8)
    ax_iqr.plot(
    row["Peak"], i,
    marker='^',                  # triangolino verso l’alto
    markersize=10,               # dimensione del triangolo
    markerfacecolor=color,       # colore interno uguale alla barra
    markeredgecolor='black',     # contorno nero
    markeredgewidth=1,         # spessore contorno
    zorder=5                     # sopra la barra IQR
)


ax_iqr.set_xlabel("Age of onset (years)")
ax_iqr.spines[['top','right', 'left']].set_visible(False)
ax_iqr.get_yaxis().set_visible(False)


# -----------------------
# Salvataggio
# -----------------------
plt.savefig(OUTFIG_COMBINED, dpi=300, bbox_inches='tight')
print(f"Saved combined figure: {OUTFIG_COMBINED}")
