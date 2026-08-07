#!/usr/bin/env python3
"""
RQ3 — step 1: cortical similarity to SUD vs age of onset (SPEARMAN primary)
===========================================================================

Figure layout, fonts, panel geometry, log x-axis, IQR strip and output names
are UNCHANGED. What changed is upstream and inferential.

WHAT CHANGED
------------
1. PTSD -> PD everywhere, and PSY_age_of_onset.xlsx now carries panic disorder
   (peak 15.5, median 26, IQR 18-36) instead of PTSD (peak 15.5, median 30,
   IQR 17-48). The peak is identical, so the PRIMARY model — which regresses
   on Peak — barely moves; the IQR strip in the lower panel does move, and it
   is the median/IQR that changes the visual impression of where the
   Mood/Anxiety cluster sits.

2. THE MIXED MODEL IS FOLDED IN HERE, AS A NUMBER, NOT A FIGURE. The old
   RQ3_ageonset_mixedeffect.py was a separate script that read
   Z_cortex_euclidean.csv — a file the rewritten pipeline no longer produces,
   on a metric that stopped being primary, standardised by a z that no longer
   exists. It should be deleted from the repo, not just left unrun. Its
   result now appears in RQ3_ageonset_model_stats.csv with
   analysis = "mixed_effects_all_pairs" and is printed to the console.

   READ THIS BEFORE QUOTING THE MIXED-MODEL p. Peak age of onset is CONSTANT
   within each disorder-by-population group, and the model puts a random
   intercept on exactly that grouping. The predictor therefore lives entirely
   in the between-group variance that the random intercept is also absorbing,
   and the two compete for the same signal. The mixed model here is not a
   more powerful version of the group-mean OLS — it is the same contrast with
   an extra variance component fitted on top, and its standard error is
   sensitive to how that component converges. It is reported because the
   n = 15 group-mean model discards the within-group spread, and a reviewer
   may ask what happens if you keep it. The PRIMARY inference remains the
   n = 15 OLS. If the two disagree, say so rather than picking the smaller p.

3. QUADRATIC TERM still fitted and exported, still not drawn (PI decision).

4. Model fitting no longer silently returns empty on small n: it says which
   analysis was skipped and why.

UNCHANGED
---------
- Primary metric = Spearman rho, read from RAW_cortex_spearman.csv.
- Inference on UNTRANSFORMED Peak; the axis and the drawn fit are in log10
  space, so the linear fit renders straight.
- Primary model = n = 15 disorder-by-population group means. Sensitivity:
  unique disorders (n = 10) and adults only (n = 9).

Just press Run.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import statsmodels.api as sm

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
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

SHOW_QUADRATIC_FIT = False      # PI: keep it in the stats, off the panel
RUN_MIXED_MODEL = True          # result only, no figure

PSY_COLORS = {"SCZ": "#1f77b4", "BD": "#ff7f0e", "MDD": "#2ca02c", "PD": "#d62728",
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
age_onset["_join_key"] = age_onset["PSY"].astype(str).str.strip().str.upper()

missing_colors = [d for d in age_onset["PSY"] if d not in PSY_COLORS]
if missing_colors:
    raise KeyError(f"no colour defined for {missing_colors}. PSY_COLORS must cover "
                   f"every disorder in {os.path.basename(AGE_ONSET_FILE)} — otherwise "
                   f"they silently render grey and become indistinguishable.")


# -----------------------
# LOAD BRAIN DATA
# -----------------------
def load_brain_data(dirs, population_label, merge_adhd=False):
    records = []
    for d in dirs:
        path = os.path.join(d, f"RAW_cortex_{METRIC}.csv")
        if not os.path.exists(path):
            continue
        # sep=None + engine="python" auto-detects the delimiter: some exports
        # come back tab- or semicolon-delimited despite the .csv extension,
        # and with the default comma pandas reads each row as one field,
        # leaving 0 data columns after index_col=0 consumes it.
        brain = pd.read_csv(path, index_col=0, sep=None, engine="python")
        if brain.shape[1] == 0:
            raise ValueError(f"{path} read with 0 data columns — check its delimiter.")
        brain.index.name = "PSY"
        if merge_adhd:
            adhd_cols = [c for c in brain.columns if c.startswith("ADHD")]
            for col in adhd_cols:
                records.append(brain[[col]].reset_index().melt(
                    id_vars="PSY", var_name="SUD", value_name="Similarity"))
            brain = brain.drop(columns=adhd_cols, errors="ignore")
        records.append(brain.reset_index().melt(
            id_vars="PSY", var_name="SUD", value_name="Similarity"))
    df = pd.concat(records, ignore_index=True)
    df["PSY_main"] = df["PSY"] + "_" + population_label
    df["Population"] = population_label
    df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
    return df


def merge_age(df, label):
    df = df.copy()
    df["PSY_for_age"] = df["PSY"].replace({"ADHD_ch": "ADHD", "ADHD_ado": "ADHD"})
    df["_join_key"] = df["PSY_for_age"].astype(str).str.strip().str.upper()
    merged = pd.merge(df, age_onset.drop(columns=["PSY"]), on="_join_key", how="left")
    lost = sorted(set(merged.loc[merged["Peak"].isna(), "PSY_for_age"]))
    if lost:
        print(f"  [{label}] no age-of-onset row for {lost} — these pairs are dropped. "
              f"Available: {sorted(age_onset['PSY'])}")
    out = merged.drop(columns=["_join_key"]).dropna(subset=["Similarity", "Peak"])
    if len(out) == 0:
        raise ValueError(
            f"merge_age({label}): every row lost its Peak value. "
            f"brain labels = {sorted(set(df['PSY_for_age']))}; "
            f"age-of-onset labels = {sorted(age_onset['PSY'])}. "
            f"Check for a renamed disorder (PTSD -> PD) or a whitespace/case "
            f"difference between the two sources.")
    return out


print("Loading brain data:")
df_adult = merge_age(load_brain_data(ADULT_DIRS, "adult"), "adult")
df_ped = merge_age(load_brain_data(PED_ADOL_DIRS, "ped", merge_adhd=True), "ped")
df_combined = pd.concat([df_adult, df_ped], ignore_index=True)
df_combined = df_combined[~df_combined["PSY_main"].str.contains(
    "Schizotyp", case=False, na=False)]


# -----------------------
# MODELS
# -----------------------
def fit_models(df_points, analysis_label):
    """OLS on UNTRANSFORMED Peak. Needs columns Similarity, Peak."""
    n = len(df_points)
    if n < 3 or np.ptp(df_points["Peak"]) == 0:
        print(f"  [skip] {analysis_label}: n = {n}, "
              f"Peak range = {np.ptp(df_points['Peak']) if n else 0}")
        return [], np.nan, np.nan

    m_lin = sm.OLS(df_points["Similarity"], sm.add_constant(df_points["Peak"])).fit()
    p_lin, ci_lin = m_lin.pvalues.get("Peak", np.nan), m_lin.conf_int().loc["Peak"]

    d2 = df_points.copy()
    d2["Peak2"] = d2["Peak"] ** 2
    m_q = sm.OLS(d2["Similarity"], sm.add_constant(d2[["Peak", "Peak2"]])).fit()
    p_q, ci_q = m_q.pvalues.get("Peak2", np.nan), m_q.conf_int().loc["Peak2"]

    return ([{"analysis": analysis_label, "model": "linear", "term": "Peak",
              "estimate": m_lin.params["Peak"], "ci_low_95": ci_lin[0],
              "ci_high_95": ci_lin[1], "p_value": p_lin,
              "r_squared": m_lin.rsquared, "n": n},
             {"analysis": analysis_label, "model": "quadratic", "term": "Peak2",
              "estimate": m_q.params["Peak2"], "ci_low_95": ci_q[0],
              "ci_high_95": ci_q[1], "p_value": p_q,
              "r_squared": m_q.rsquared, "n": n}], p_lin, p_q)


def fit_mixed(df_pairs):
    """
    Random-intercept model over ALL pairwise observations, grouped by
    disorder-by-population. See the docstring: Peak is constant within group,
    so the fixed effect and the random intercept are competing for the same
    between-group variance. Reported, not primary.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    d = df_pairs.dropna(subset=["Similarity", "Peak"]).copy()
    d["logPeak"] = np.log(d["Peak"])
    out = []
    for term, cols in (("Peak", ["Peak"]), ("logPeak", ["logPeak"])):
        try:
            m = MixedLM(d["Similarity"], sm.add_constant(d[cols]),
                        groups=d["PSY_main"]).fit()
            ci = m.conf_int().loc[term]
            out.append({"analysis": "mixed_effects_all_pairs",
                        "model": f"random intercept ~ {term}", "term": term,
                        "estimate": m.params[term], "ci_low_95": ci[0],
                        "ci_high_95": ci[1], "p_value": m.pvalues[term],
                        "r_squared": np.nan, "n": len(d),
                        "n_groups": d["PSY_main"].nunique(),
                        "converged": bool(m.converged)})
        except Exception as e:
            print(f"  [mixed] {term} failed to fit: {type(e).__name__}: {e}")
    return out


def fmt_p(p):
    return "NA" if not np.isfinite(p) else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


# -----------------------
# PLOT + STATS
# -----------------------
def plot_on_ax(ax, df, title):
    for psy_label in df["PSY_main"].unique():
        sub = df[df["PSY_main"] == psy_label]
        base = psy_label.split("_")[0]
        ax.scatter(sub["Peak"], sub["Similarity"], s=60, alpha=0.6,
                   color=PSY_COLORS.get(base, "grey"), edgecolor="none")

    df_mean = df.groupby("PSY_main").agg(
        {"Similarity": "mean", "Peak": "first"}).reset_index()
    df_mean["PSY_base"] = df_mean["PSY_main"].apply(
        lambda x: "ADHD" if x.startswith("ADHD") else x.split("_")[0])
    for _, row in df_mean.iterrows():
        ax.scatter(row["Peak"], row["Similarity"], s=200, edgecolor="black",
                   linewidth=1.5, color=PSY_COLORS.get(row["PSY_base"], "grey"), zorder=3)

    stats_primary, p_lin, p_quad = fit_models(df_mean, "primary_disorder_by_population")
    n_test = len(df_mean)
    band_df = None

    if stats_primary:
        lx = np.log10(df_mean["Peak"].values)
        y = df_mean["Similarity"].values
        m_log = sm.OLS(y, sm.add_constant(lx)).fit()
        xs = np.linspace(df_mean["Peak"].min(), df_mean["Peak"].max(), 200)
        pred = m_log.get_prediction(sm.add_constant(np.log10(xs)))
        ax.plot(xs, pred.predicted_mean, color="black", lw=2)
        ci = pred.conf_int(alpha=0.05)
        ax.fill_between(xs, ci[:, 0], ci[:, 1], color="black", alpha=0.10, lw=0)
        band_df = pd.DataFrame({"Peak_years": xs,
                                "linear_fit_predicted_similarity": pred.predicted_mean,
                                "linear_fit_ci_low_95": ci[:, 0],
                                "linear_fit_ci_high_95": ci[:, 1]})
        if SHOW_QUADRATIC_FIT:
            lxs = np.log10(xs)
            m_q = sm.OLS(y, sm.add_constant(np.column_stack([lx, lx ** 2]))).fit()
            ys_q = m_q.predict(sm.add_constant(np.column_stack([lxs, lxs ** 2])))
            ax.plot(xs, ys_q, color="black", lw=2, ls="--", alpha=0.55)
            band_df["quadratic_fit_predicted_similarity"] = ys_q

    ax.text(0.02, 0.98, f"Linear: {fmt_p(p_lin)}\nn = {n_test}",
            transform=ax.transAxes, fontsize=16, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_ylabel("Neuroanatomical similarity (Spearman \u03c1)", fontsize=16)
    ax.set_title(title, fontsize=20)

    all_stats = list(stats_primary)
    df_unique = (df_mean.groupby("PSY_base")
                 .agg({"Similarity": "mean", "Peak": "first"}).reset_index())
    s_u, p_lin_u, p_q_u = fit_models(df_unique, "sensitivity_unique_disorder")
    df_ad = df_mean[df_mean["PSY_main"].str.endswith("_adult")]
    s_a, p_lin_a, p_q_a = fit_models(df_ad, "sensitivity_adults_only")
    all_stats += s_u + s_a

    print("\n--- Age of onset: linear model on Peak ---")
    print(f"  PRIMARY  disorder x population   n = {n_test:2d}  {fmt_p(p_lin)}"
          f"   [quadratic {fmt_p(p_quad)}]")
    print(f"  sens.    unique disorders        n = {len(df_unique):2d}  {fmt_p(p_lin_u)}"
          f"   [quadratic {fmt_p(p_q_u)}]")
    print(f"  sens.    adults only             n = {len(df_ad):2d}  {fmt_p(p_lin_a)}"
          f"   [quadratic {fmt_p(p_q_a)}]")

    if RUN_MIXED_MODEL:
        mixed = fit_mixed(df)
        all_stats += mixed
        print("\n--- Mixed effects, all pairwise observations (reported, NOT primary) ---")
        for m in mixed:
            print(f"  {m['term']:<8s} n = {m['n']:3d} in {m['n_groups']} groups  "
                  f"beta = {m['estimate']:+.5f}  {fmt_p(m['p_value'])}  "
                  f"converged = {m['converged']}")
        print("  NB: Peak is constant within group, so this shares its signal with "
              "the random intercept. Do not read it as a more powerful test.")

    figs = os.path.join(BASE_DIR, "figures")
    df_mean.assign(analysis="primary_disorder_by_population").to_csv(
        os.path.join(figs, "RQ3_ageonset_points_primary.csv"), index=False)
    df_unique.assign(analysis="sensitivity_unique_disorder").to_csv(
        os.path.join(figs, "RQ3_ageonset_points_unique_disorder.csv"), index=False)
    return all_stats, band_df


# -----------------------
# FIGURE (layout unchanged)
# -----------------------
fig = plt.figure(figsize=(16, 10))
x_min, x_max = 5, 50
left, width = 0.1, 0.65

ax_scatter = fig.add_axes([left, 0.35, width, 0.6])
model_stats, band_df = plot_on_ax(
    ax_scatter, df_combined, "Neuroanatomical Similarity: Relation to Age of Onset")
ax_scatter.set_xscale("log")
ax_scatter.set_xlim(x_min, x_max)
ax_scatter.set_xticks([])
ax_scatter.xaxis.set_minor_locator(NullLocator())

handles = [Line2D([0], [0], marker="o", color="w", label=psy,
                  markerfacecolor=PSY_COLORS[psy], markersize=8, markeredgecolor="none")
           for psy in PSY_COLORS]
handles += [Line2D([0], [0], marker="o", color="w", label="Peak (mean)",
                   markerfacecolor="white", markeredgecolor="black", markersize=10),
            Line2D([0], [0], marker="^", color="w", label="Median",
                   markerfacecolor="white", markeredgecolor="black", markersize=8),
            Line2D([0], [0], color="black", lw=2, label="Linear fit (95% CI)")]
ax_scatter.legend(handles=handles, loc="center right", bbox_to_anchor=(0.98, 0.5),
                  fontsize=15, markerscale=1.6, labelspacing=0.7, handletextpad=0.8)

ax_iqr = fig.add_axes([left, 0.05, width, 0.25])
ax_iqr.set_xscale("log")
ax_iqr.set_xlim(x_min, x_max)
ticks = [5, 10, 20, 30, 50]
ax_iqr.set_xticks(ticks)
ax_iqr.set_xticklabels([str(t) for t in ticks])
ax_iqr.xaxis.set_minor_locator(NullLocator())
for i, (_, row) in enumerate(age_onset.sort_values("AgeOnset").iterrows()):
    color = PSY_COLORS.get(row["PSY"], "grey")
    ax_iqr.hlines(y=i, xmin=row["P25"], xmax=row["P75"], color=color, linewidth=6)
    ax_iqr.plot(row["AgeOnset"], i, marker="^", markersize=10,
                markerfacecolor=color, markeredgecolor="black")
ax_iqr.set_xlabel("Age of onset (years)", fontsize=16)
ax_iqr.spines[["top", "right", "left"]].set_visible(False)
ax_iqr.get_yaxis().set_visible(False)

plt.savefig(OUTFIG_COMBINED, dpi=300, bbox_inches="tight")
plt.savefig(OUTFIG_COMBINED.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\nSaved: {OUTFIG_COMBINED}")

figs = os.path.join(BASE_DIR, "figures")
pd.DataFrame(model_stats).round(6).to_csv(
    os.path.join(figs, "RQ3_ageonset_model_stats.csv"), index=False)
print(f"Saved: {os.path.join(figs, 'RQ3_ageonset_model_stats.csv')}")
if band_df is not None:
    band_df.round(6).to_csv(os.path.join(figs, "RQ3_ageonset_CI_band.csv"), index=False)
    print(f"Saved: {os.path.join(figs, 'RQ3_ageonset_CI_band.csv')}")