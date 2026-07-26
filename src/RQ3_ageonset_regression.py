#!/usr/bin/env python3
"""
RQ3 - Neuroanatomical similarity vs age of onset (SPEARMAN primary)
===================================================================


1. Primary metric = Spearman rho (reads RAW_cortex_spearman.csv).

2. Display fits are now functions of log10(Peak), so the LINEAR (dashed) fit
   renders as a STRAIGHT line on the log x-axis instead of a curve. This is
   the visualization the Methods already describe ("Peak age of onset was
   log-transformed in figures ... untransformed values were used in
   inferential models"): the axis and the drawn fits are in log space, while
   the reported Linear/Quad p-values still come from OLS on UNTRANSFORMED
   Peak. So the dashed line straightens without altering the inferential
   models.

3. A 95% confidence band is drawn around the linear (dashed) display fit.

4. The sample size of the inferential test (n = number of disorder-level
   points) is annotated on the panel.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from matplotlib.ticker import NullLocator

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

BASE_DIR = repo_dir
METRIC = "spearman"
AGE_ONSET_FILE = os.path.join(BASE_DIR, "data", "raw", "PSY_age_of_onset.xlsx")

ADULT_DIRS = [os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")]
PED_ADOL_DIRS = [
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_all"),
    os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adolescents_ctx"),
]
OUTFIG_COMBINED = os.path.join(BASE_DIR, "figures", "RQ3_ageonset_spearman.png")
os.makedirs(os.path.dirname(OUTFIG_COMBINED), exist_ok=True)

PSY_COLORS = {"SCZ": "#1f77b4", "BD": "#ff7f0e", "MDD": "#2ca02c", "PTSD": "#d62728",
              "ASD": "#9467bd", "ADHD": "#8c564b", "AN": "#e377c2", "OCD": "#7f7f7f",
              "CD": "#bcbd22", "CHR": "#17becf"}

# -----------------------
# LOAD AGE
# -----------------------
age_onset = pd.read_excel(AGE_ONSET_FILE)
age_onset.rename(columns={"disorder": "PSY", "median": "AgeOnset",
                          "p25": "P25", "p75": "P75",
                          "Peak age onset (yrs)": "Peak"}, inplace=True)
for col in ["AgeOnset", "P25", "P75", "Peak"]:
    age_onset[col] = age_onset[col].apply(
        lambda x: float(str(x).replace(",", ".")) if pd.notna(x) else np.nan)


# -----------------------
# LOAD BRAIN DATA
# -----------------------
def load_brain_data(dirs, population_label, merge_adhd=False):
    records = []
    for d in dirs:
        path = os.path.join(d, f"RAW_cortex_{METRIC}.csv")
        if not os.path.exists(path):
            continue
        brain = pd.read_csv(path, index_col=0)
        if merge_adhd:
            adhd_cols = [c for c in brain.columns if c.startswith("ADHD")]
            for col in adhd_cols:
                temp_long = brain[[col]].copy().reset_index().melt(
                    id_vars="index", var_name="SUD", value_name="Similarity")
                temp_long.rename(columns={"index": "PSY"}, inplace=True)
                records.append(temp_long)
            brain = brain.drop(columns=adhd_cols, errors="ignore")
        brain_long = brain.reset_index().melt(
            id_vars="index", var_name="SUD", value_name="Similarity")
        brain_long.rename(columns={"index": "PSY"}, inplace=True)
        records.append(brain_long)
    df = pd.concat(records, ignore_index=True)
    df["PSY_main"] = df["PSY"] + "_" + population_label
    df["Population"] = population_label
    df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
    return df


df_adult = load_brain_data(ADULT_DIRS, "adult")
df_ped = load_brain_data(PED_ADOL_DIRS, "ped", merge_adhd=True)


def merge_age(df):
    df["PSY_for_age"] = df["PSY"].replace({"ADHD_ch": "ADHD", "ADHD_ado": "ADHD"})
    df = pd.merge(df, age_onset, left_on="PSY_for_age", right_on="PSY", how="left")
    return df.dropna(subset=["Similarity", "Peak"])


df_adult = merge_age(df_adult)
df_ped = merge_age(df_ped)
df_combined = pd.concat([df_adult, df_ped], ignore_index=True)
# drop Schizotypy if present
df_combined = df_combined[~df_combined["PSY_main"].str.contains("Schizotyp", case=False, na=False)]


# -----------------------
# PLOT FUNCTION
# -----------------------
def plot_on_ax(ax, df, title):
    # small points
    for psy_label in df["PSY_main"].unique():
        sub = df[df["PSY_main"] == psy_label]
        base_psy = sub["PSY_main"].iloc[0].split("_")[0]
        ax.scatter(sub["Peak"], sub["Similarity"], s=60, alpha=0.6,
                   color=PSY_COLORS.get(base_psy, "grey"), edgecolor="none")

    # mean per PSY_main
    df_mean = df.groupby("PSY_main").agg({"Similarity": "mean", "Peak": "first"}).reset_index()
    df_mean["PSY_base"] = df_mean["PSY_main"].apply(lambda x: x.split("_")[0])
    for _, row in df_mean.iterrows():
        ax.scatter(row["Peak"], row["Similarity"], s=200, edgecolor="black",
                   linewidth=1.5, color=PSY_COLORS.get(row["PSY_base"], "grey"), zorder=3)

    p_lin_text = p_quad_text = "NA"
    n_test = df_mean.shape[0]
    model_stats = []
    band_df = None

    if n_test >= 3 and np.ptp(df_mean["Peak"]) > 0:
        # -------- INFERENCE: untransformed Peak (Methods) --------
        X_lin_raw = sm.add_constant(df_mean["Peak"])
        m_lin_raw = sm.OLS(df_mean["Similarity"], X_lin_raw).fit()
        p_lin = m_lin_raw.pvalues.get("Peak", np.nan)
        p_lin_text = "p < 0.001" if p_lin < 0.001 else f"p = {p_lin:.3f}"
        ci_lin = m_lin_raw.conf_int(alpha=0.05).loc["Peak"]

        df_mean["Peak2"] = df_mean["Peak"] ** 2
        X_quad_raw = sm.add_constant(df_mean[["Peak", "Peak2"]])
        m_quad_raw = sm.OLS(df_mean["Similarity"], X_quad_raw).fit()
        p_quad = m_quad_raw.pvalues.get("Peak2", np.nan)
        p_quad_text = "p < 0.001" if p_quad < 0.001 else f"p = {p_quad:.3f}"
        ci_quad = m_quad_raw.conf_int(alpha=0.05).loc["Peak2"]

        model_stats = [
            {"model": "linear", "term": "Peak", "estimate": m_lin_raw.params["Peak"],
             "ci_low_95": ci_lin[0], "ci_high_95": ci_lin[1], "p_value": p_lin,
             "r_squared": m_lin_raw.rsquared, "n": n_test},
            {"model": "quadratic", "term": "Peak2", "estimate": m_quad_raw.params["Peak2"],
             "ci_low_95": ci_quad[0], "ci_high_95": ci_quad[1], "p_value": p_quad,
             "r_squared": m_quad_raw.rsquared, "n": n_test},
        ]

        # -------- DISPLAY fits in log10(Peak) space --------
        # linear in log10(Peak) => STRAIGHT line on the log x-axis
        lx = np.log10(df_mean["Peak"].values)
        y = df_mean["Similarity"].values
        m_lin_log = sm.OLS(y, sm.add_constant(lx)).fit()
        m_quad_log = sm.OLS(y, sm.add_constant(np.column_stack([lx, lx ** 2]))).fit()

        xs = np.linspace(df_mean["Peak"].min(), df_mean["Peak"].max(), 200)
        lxs = np.log10(xs)

        # linear (dashed) + 95% CI band
        pred_lin = m_lin_log.get_prediction(sm.add_constant(lxs))
        ax.plot(xs, pred_lin.predicted_mean, color="black", lw=2, ls="--", alpha=0.55)
        ci = pred_lin.conf_int(alpha=0.05)
        ax.fill_between(xs, ci[:, 0], ci[:, 1], color="black", alpha=0.10, lw=0)

        # quadratic (solid)
        ys_quad = m_quad_log.predict(sm.add_constant(np.column_stack([lxs, lxs ** 2])))
        ax.plot(xs, ys_quad, color="black", lw=2)

        band_df = pd.DataFrame({
            "Peak_years": xs,
            "linear_fit_predicted_similarity": pred_lin.predicted_mean,
            "linear_fit_ci_low_95": ci[:, 0],
            "linear_fit_ci_high_95": ci[:, 1],
            "quadratic_fit_predicted_similarity": ys_quad,
        })

    ax.text(0.02, 0.98, f"Linear: {p_lin_text}\nQuad: {p_quad_text}\nn = {n_test}",
            transform=ax.transAxes, fontsize=16, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_ylabel("Neuroanatomical similarity (Spearman \u03c1)", fontsize=16)
    ax.set_title(title, fontsize=20)

    return model_stats, band_df


# -----------------------
# FIGURE
# -----------------------
fig = plt.figure(figsize=(16, 10))
x_min, x_max = 5, 50
left, width = 0.1, 0.65

ax_scatter = fig.add_axes([left, 0.35, width, 0.6])
model_stats, band_df = plot_on_ax(ax_scatter, df_combined, "Neuroanatomical Similarity: Relation to Age of Onset")
ax_scatter.set_xscale("log")
ax_scatter.set_xlim(x_min, x_max)
ax_scatter.set_xticks([])
ax_scatter.xaxis.set_minor_locator(NullLocator())

# legend
handles = [Line2D([0], [0], marker="o", color="w", label=psy,
                  markerfacecolor=PSY_COLORS[psy], markersize=8, markeredgecolor="none")
           for psy in PSY_COLORS]
handles.append(Line2D([0], [0], marker="o", color="w", label="Peak (mean)",
                      markerfacecolor="white", markeredgecolor="black", markersize=10))
handles.append(Line2D([0], [0], marker="^", color="w", label="Median",
                      markerfacecolor="white", markeredgecolor="black", markersize=8))
handles += [Line2D([0], [0], color="black", lw=2, ls="--", alpha=0.55, label="Linear fit"),
            Line2D([0], [0], color="black", lw=2, label="Quadratic fit")]
ax_scatter.legend(handles=handles, loc="center right", bbox_to_anchor=(0.98, 0.5),
                   fontsize=15, markerscale=1.6, labelspacing=0.7, handletextpad=0.8)

# IQR panel
ax_iqr = fig.add_axes([left, 0.05, width, 0.25])
ax_iqr.set_xscale("log")
ax_iqr.set_xlim(x_min, x_max)
ticks = [5, 10, 20, 30, 50]
ax_iqr.set_xticks(ticks)
ax_iqr.set_xticklabels([str(t) for t in ticks])
ax_iqr.xaxis.set_minor_locator(NullLocator())
age_plot = age_onset.sort_values("AgeOnset")
for i, (_, row) in enumerate(age_plot.iterrows()):
    color = PSY_COLORS.get(row["PSY"], "grey")
    ax_iqr.hlines(y=i, xmin=row["P25"], xmax=row["P75"], color=color, linewidth=6)
    ax_iqr.plot(row["AgeOnset"], i, marker="^", markersize=10,
                markerfacecolor=color, markeredgecolor="black")
ax_iqr.set_xlabel("Age of onset (years)", fontsize=16)
ax_iqr.spines[["top", "right", "left"]].set_visible(False)
ax_iqr.get_yaxis().set_visible(False)

plt.savefig(OUTFIG_COMBINED, dpi=300, bbox_inches="tight")
plt.savefig(OUTFIG_COMBINED.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {OUTFIG_COMBINED}")

# -----------------------
# EXPORT CI / MODEL RESULTS
# -----------------------
stats_path = os.path.join(BASE_DIR, "figures", "RQ3_ageonset_model_stats.csv")
band_path = os.path.join(BASE_DIR, "figures", "RQ3_ageonset_CI_band.csv")

pd.DataFrame(model_stats).round(6).to_csv(stats_path, index=False)
print(f"Saved: {stats_path}")

if band_df is not None:
    band_df.round(6).to_csv(band_path, index=False)
    print(f"Saved: {band_path}")