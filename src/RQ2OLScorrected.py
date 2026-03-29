#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from matplotlib.ticker import NullLocator

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

OUTFIG_COMBINED = os.path.join(BASE_DIR, "overlap_vs_age_peak_OLS_cluster.png")

# Colors
PSY_COLORS = {
    "SCZ": "#1f77b4", "BD": "#ff7f0e", "MDD": "#2ca02c", "PTSD": "#d62728",
    "ASD": "#9467bd", "ADHD": "#8c564b", "AN": "#e377c2", "OCD": "#7f7f7f",
    "CD": "#bcbd22", "CHR": "#17becf"
}

PSY_NAME_MAP = {
    "ADHD": "ADHD","ADHD_ch": "ADHD_ch","ADHD_ado": "ADHD_ado","ASD": "ASD",
    "AN": "AN","OCD": "OCD","BD": "BD","MDD": "MDD","SCZ": "SCZ","CHR": "CHR",
    "CD": "CD","PTSD": "PTSD"
}

# -----------------------
# LOAD AGE
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
            continue
        brain = pd.read_csv(path, index_col=0)

        if merge_adhd:
            adhd_cols = [c for c in brain.columns if c.startswith("ADHD")]
            for col in adhd_cols:
                temp = brain[[col]].copy()
                temp_long = temp.reset_index().melt(
                    id_vars="index", var_name="SUD", value_name="Similarity"
                )
                temp_long.rename(columns={"index": "PSY"}, inplace=True)
                records.append(temp_long)
            brain = brain.drop(columns=adhd_cols, errors="ignore")

        brain_long = brain.reset_index().melt(
            id_vars="index", var_name="SUD", value_name="Similarity"
        )
        brain_long.rename(columns={"index": "PSY"}, inplace=True)
        records.append(brain_long)

    df = pd.concat(records, ignore_index=True)
    df["PSY_main"] = df["PSY"].replace(PSY_NAME_MAP)
    df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
    return df

df_adult = load_brain_data(ADULT_DIRS)
df_adult["Population"] = "Adult"
df_ped = load_brain_data(PED_ADOL_DIRS, merge_adhd=True)
df_ped["Population"] = "Pediatric"

# -----------------------
# MERGE AGE
# -----------------------
def merge_age(df):
    df["PSY_for_age"] = df["PSY_main"].replace({
        "ADHD_ch": "ADHD",
        "ADHD_ado": "ADHD"
    })
    df = pd.merge(df, age_onset, left_on="PSY_for_age", right_on="PSY", how="left")
    return df.dropna(subset=["Similarity", "Peak"])

df_adult = merge_age(df_adult)
df_ped = merge_age(df_ped)
df_combined = pd.concat([df_adult, df_ped], ignore_index=True)

# -----------------------
# PLOT FUNCTION (OLS CLUSTER)
# -----------------------
def plot_on_ax(ax, df, title):
    for psy in df["PSY_main"].unique():
        sub = df[df["PSY_main"] == psy]
        ax.scatter(
            sub["Peak"], sub["Similarity"], s=80, alpha=0.85,
            color=PSY_COLORS.get(psy.replace("_ch","").replace("_ado",""), "grey"),
            edgecolor='k'
        )

    df_valid = df.copy()
    df_valid["cluster"] = df_valid["PSY_main"] + "_" + df_valid["Population"]
    df_valid["logPeak"] = np.log(df_valid["Peak"])

    n_groups = df_valid["cluster"].nunique()
    print(f"{title} — Number of clusters: {n_groups}")

    # LINEAR MODEL
    X_lin = sm.add_constant(df_valid["logPeak"])
    model_lin = sm.OLS(df_valid["Similarity"], X_lin).fit(
        cov_type='cluster', cov_kwds={'groups': df_valid["cluster"]}
    )
    p_lin = model_lin.pvalues.get("logPeak", np.nan)
    p_lin_text = f"p < 0.001" if p_lin < 0.001 else f"p={p_lin:.3f}"

    # QUADRATIC MODEL
    df_valid["logPeak2"] = df_valid["logPeak"]**2
    X_quad = sm.add_constant(df_valid[["logPeak","logPeak2"]])
    model_quad = sm.OLS(df_valid["Similarity"], X_quad).fit(
        cov_type='cluster', cov_kwds={'groups': df_valid["cluster"]}
    )
    p_quad = model_quad.pvalues.get("logPeak2", np.nan)
    p_quad_text = f"p < 0.001" if p_quad < 0.001 else f"p={p_quad:.3f}"

    # PREDICTIONS
    xs = np.linspace(df_valid["Peak"].min(), df_valid["Peak"].max(), 100)
    ys_lin = model_lin.predict(sm.add_constant(np.log(xs)))
    ys_quad = model_quad.predict(sm.add_constant(pd.DataFrame({
        "logPeak": np.log(xs),
        "logPeak2": np.log(xs)**2
    })))
    ax.plot(xs, ys_lin, color="black", lw=2, linestyle="--", alpha=0.3)
    ax.plot(xs, ys_quad, color="black", lw=2)

    print(f"{title} — Linear: {p_lin_text}, Quadratic: {p_quad_text}")
    ax.text(0.98, 0.02, f"Linear: {p_lin_text}\nQuad: {p_quad_text}",
            transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_ylabel("Neuroanatomical similarity")
    ax.set_title(title)

# -----------------------
# FIGURE
# -----------------------
fig = plt.figure(figsize=(12,10))

x_min, x_max = 5, 50
left, width = 0.1, 0.65

ax_scatter = fig.add_axes([left, 0.35, width, 0.6])
plot_on_ax(ax_scatter, df_combined,
           "Neuroanatomical Similarity: Relation to Age of Onset")

ax_scatter.set_xscale('log')
ax_scatter.set_xlim(x_min, x_max)
ax_scatter.set_xticks([])              # Niente tick visibili
ax_scatter.xaxis.set_minor_locator(NullLocator())  # Rimuove tutti i minor ticks

# -----------------------
# LEGEND
# -----------------------
handles = [Line2D([0],[0], marker='o', color='w', label=psy,
                  markerfacecolor=PSY_COLORS[psy], markersize=8, markeredgecolor='none')
           for psy in PSY_COLORS.keys()]

handles.append(Line2D([0],[0], marker='o', color='w', label='Peak',
                      markerfacecolor='white', markeredgecolor='black'))
handles.append(Line2D([0],[0], marker='^', color='w', label='Median',
                      markerfacecolor='white', markeredgecolor='black'))
handles += [Line2D([0],[0], color='black', lw=2, linestyle='--', alpha=0.3, label='Linear fit'),
            Line2D([0],[0], color='black', lw=2, label='Quadratic fit')]

ax_scatter.legend(handles=handles, loc='center right', bbox_to_anchor=(0.98,0.5))

# -----------------------
# IQR PANEL
# -----------------------
ax_iqr = fig.add_axes([left, 0.05, width, 0.25])
ax_iqr.set_xscale('log')
ax_iqr.set_xlim(x_min, x_max)

ticks = [5,10,20,30,50]
ax_iqr.set_xticks(ticks)
ax_iqr.set_xticklabels([str(t) for t in ticks])
ax_iqr.xaxis.set_minor_locator(NullLocator())

age_plot = age_onset.sort_values("AgeOnset")
for i, (_, row) in enumerate(age_plot.iterrows()):
    color = PSY_COLORS.get(row["PSY"], "grey")
    ax_iqr.hlines(y=i, xmin=row["P25"], xmax=row["P75"], color=color, linewidth=6)
    ax_iqr.plot(row["AgeOnset"], i, marker='^', markersize=10,
                markerfacecolor=color, markeredgecolor='black')

ax_iqr.set_xlabel("Age of onset (years)")
ax_iqr.spines[['top','right','left']].set_visible(False)
ax_iqr.get_yaxis().set_visible(False)

# -----------------------
# SAVE
# -----------------------
plt.savefig(OUTFIG_COMBINED, dpi=300, bbox_inches='tight')
print(f"Saved: {OUTFIG_COMBINED}")