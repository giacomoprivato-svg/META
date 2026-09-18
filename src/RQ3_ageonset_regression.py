#!/usr/bin/env python3
"""
RQ3 — step 1: cortical similarity to SUD vs age of onset (SPEARMAN primary)

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

# The aggregate all-SUD map. Set to False only to reproduce the old, inflated
# seven-category numbers for comparison — never for anything that goes in the
# manuscript. Matching is case-insensitive and whitespace-stripped; add spellings
# here if a future export renames the column.
EXCLUDE_AGGREGATE_SUD = True
AGGREGATE_SUD_NAMES = ["SUD", "ALL_SUD", "ALLSUD", "SUD_ALL", "ALL SUD"]
N_EXPECTED_SUD = 6              # substance-specific maps: ALC ATS CAN COC NIC OPI

RUN_AGE_AT_SCAN = True          # robustness (b); stats only, no figure

# Mean age at scan of the PATIENT group, keyed by PSY_main exactly as the script
# builds it (PSY label + "_" + population). Values from Table S1.
#   n_cases, mean age at scan, SD, % female
# ADHD_ado_ped and BD_ped are None: Table S1 does not report a separate
# adolescent ADHD sample or a separate pediatric BD sample, so those two groups
# drop out of the age-at-scan analysis (n = 13 instead of 15). Fill them in if
# the numbers become available — do not substitute the combined-sample age.
AGE_AT_SCAN = {
    # adults
    "ADHD_adult":   {"n": 2246, "age": 19.22, "sd": 11.31, "pct_f": 25.9},
    "AN_adult":     {"n":  685, "age": 21.00, "sd":  5.50, "pct_f": 100.0},
    "ASD_adult":    {"n": 1571, "age": 15.41, "sd":  8.64, "pct_f": 14.3},
    "BD_adult":     {"n": 2447, "age": 38.40, "sd": 10.90, "pct_f": 41.3},
    "CHR_adult":    {"n": 1792, "age": 20.80, "sd":  5.90, "pct_f": 46.4},
    "MDD_adult":    {"n": 1911, "age": 43.20, "sd": 12.60, "pct_f": 64.5},
    "OCD_adult":    {"n": 1498, "age": 31.31, "sd":  9.74, "pct_f": 49.9},
    "PD_adult":     {"n": 1146, "age": 33.80, "sd": 12.20, "pct_f": 64.0},
    "SCZ_adult":    {"n": 4474, "age": 32.30, "sd": 10.00, "pct_f": 34.0},
    # pediatric
    "ADHD_ch_ped":  {"n": 2707, "age": 10.11, "sd":  0.57, "pct_f": 50.6},
    "ADHD_ado_ped": None,
    "CD_ped":       {"n": 1185, "age": 13.71, "sd":  3.01, "pct_f": 28.6},
    "OCD_ped":      {"n":  407, "age": 13.69, "sd":  2.58, "pct_f": 47.2},
    "BD_ped":       None,
    "MDD_ped":      {"n":  237, "age": 19.10, "sd":  1.80, "pct_f": 68.4},
}

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
_aggregate_drop_log = []        # (file, column) for every aggregate column removed


def drop_aggregate_sud(brain, path):
    """
    Remove the all-SUD aggregate column from a PSY x SUD similarity matrix.

    Returns the matrix without it and appends to _aggregate_drop_log so the
    caller can prove, after all files are read, that the drop actually fired.
    Silence here is exactly the failure mode we are guarding against: an
    aggregate column that survives under an unexpected name would re-enter
    every mean without changing anything visible.
    """
    if not EXCLUDE_AGGREGATE_SUD:
        return brain
    wanted = {n.strip().upper() for n in AGGREGATE_SUD_NAMES}
    hits = [c for c in brain.columns if str(c).strip().upper() in wanted]
    for c in hits:
        _aggregate_drop_log.append((os.path.basename(os.path.dirname(path)), c))
    out = brain.drop(columns=hits)
    remaining = [c for c in out.columns if str(c).strip().upper() in wanted]
    assert not remaining, f"aggregate column survived the drop in {path}: {remaining}"
    return out


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
        # Drop the aggregate BEFORE any melt, so nothing downstream ever sees it.
        brain = drop_aggregate_sud(brain, path)
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
# AGGREGATE-EXCLUSION AUDIT
# -----------------------
# Everything above is unchanged logic; this block only proves the exclusion
# happened and that the pair count is what the manuscript will claim.
if EXCLUDE_AGGREGATE_SUD:
    if not _aggregate_drop_log:
        raise RuntimeError(
            "EXCLUDE_AGGREGATE_SUD is True but no aggregate column was found in any "
            f"RAW_cortex_{METRIC}.csv. Columns actually present: "
            f"{sorted(df_combined['SUD'].unique())}. Add the correct spelling to "
            "AGGREGATE_SUD_NAMES — do NOT just switch the flag off.")
    print("\n--- Aggregate all-SUD map excluded ---")
    for src, col in _aggregate_drop_log:
        print(f"  dropped column '{col}' from {src}")

sud_cats = sorted(df_combined["SUD"].unique())
n_groups = df_combined["PSY_main"].nunique()
print(f"  SUD categories retained ({len(sud_cats)}): {sud_cats}")
print(f"  {n_groups} disorder-by-population groups, {len(df_combined)} pairwise "
      f"observations")

if EXCLUDE_AGGREGATE_SUD and len(sud_cats) != N_EXPECTED_SUD:
    raise RuntimeError(
        f"expected {N_EXPECTED_SUD} substance-specific maps after excluding the "
        f"aggregate, got {len(sud_cats)}: {sud_cats}")

# Per-group balance check. An unbalanced group means a missing pair somewhere,
# which would quietly bias the group mean that the primary model regresses on.
counts = df_combined.groupby("PSY_main")["SUD"].nunique()
unbalanced = counts[counts != len(sud_cats)]
if len(unbalanced):
    print(f"  [warn] groups without all {len(sud_cats)} SUD categories:\n{unbalanced}")

# -----------------------
# AGE AT SCAN
# -----------------------
unknown_keys = sorted(set(df_combined["PSY_main"]) - set(AGE_AT_SCAN))
if unknown_keys:
    raise KeyError(
        f"AGE_AT_SCAN has no entry for {unknown_keys}. Every group must be listed "
        f"explicitly, as None if no age at scan is available — an absent key would "
        f"silently become a missing value and shrink the analysis without warning. "
        f"Keys defined: {sorted(AGE_AT_SCAN)}")

df_combined["AgeAtScan"] = df_combined["PSY_main"].map(
    lambda k: AGE_AT_SCAN[k]["age"] if AGE_AT_SCAN[k] else np.nan)
df_combined["N_cases"] = df_combined["PSY_main"].map(
    lambda k: AGE_AT_SCAN[k]["n"] if AGE_AT_SCAN[k] else np.nan)

no_scan_age = sorted({k for k in df_combined["PSY_main"].unique() if not AGE_AT_SCAN[k]})
if no_scan_age:
    print(f"  [note] no age at scan for {no_scan_age} — these groups enter the "
          f"age-of-onset models but drop out of the age-at-scan robustness check")

# -----------------------
# WHICH MAPS ENTER THE MAIN ANALYSIS
# -----------------------
inventory = (df_combined.groupby(["PSY_main", "Population"])
             .agg(n_pairs=("Similarity", "size"),
                  mean_similarity=("Similarity", "mean"),
                  peak_age_onset=("Peak", "first"),
                  age_at_scan=("AgeAtScan", "first"),
                  n_cases=("N_cases", "first"))
             .reset_index()
             .sort_values(["Population", "PSY_main"]))
print("\n--- Maps entering the PRIMARY age-of-onset analysis ---")
print(inventory.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"  {(inventory['Population'] == 'adult').sum()} adult + "
      f"{(inventory['Population'] == 'ped').sum()} pediatric = {len(inventory)} groups. "
      f"The primary analysis is NOT restricted to adults.")


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
              "r_squared": m_lin.rsquared, "n": n,
              "n_sud_categories": len(sud_cats)},
             {"analysis": analysis_label, "model": "quadratic", "term": "Peak2",
              "estimate": m_q.params["Peak2"], "ci_low_95": ci_q[0],
              "ci_high_95": ci_q[1], "p_value": p_q,
              "r_squared": m_q.rsquared, "n": n,
              "n_sud_categories": len(sud_cats)}], p_lin, p_q)


def fit_mixed(df_pairs, terms=(("Peak", ["Peak"]), ("logPeak", ["logPeak"])),
              analysis_label="mixed_effects_all_pairs"):
    """
    Random-intercept model over ALL pairwise observations, grouped by
    disorder-by-population. See the docstring: Peak is constant within group,
    so the fixed effect and the random intercept are competing for the same
    between-group variance. Reported, not primary.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    needed = sorted({c for _, cols in terms for c in cols if c != "logPeak"})
    d = df_pairs.dropna(subset=["Similarity"] + needed).copy()
    if "Peak" in d.columns:
        d["logPeak"] = np.log(d["Peak"])
    out = []
    for term, cols in terms:
        try:
            m = MixedLM(d["Similarity"], sm.add_constant(d[cols]),
                        groups=d["PSY_main"]).fit()
            ci = m.conf_int().loc[term]
            out.append({"analysis": analysis_label,
                        "model": f"random intercept ~ {term}", "term": term,
                        "estimate": m.params[term], "ci_low_95": ci[0],
                        "ci_high_95": ci[1], "p_value": m.pvalues[term],
                        "r_squared": np.nan, "n": len(d),
                        "n_groups": d["PSY_main"].nunique(),
                        "n_sud_categories": len(sud_cats),
                        "converged": bool(m.converged)})
        except Exception as e:
            print(f"  [mixed] {term} failed to fit: {type(e).__name__}: {e}")
    return out


def fit_ols_terms(df_points, xcols, analysis_label, model_label):
    """
    OLS of Similarity on one or more UNTRANSFORMED predictors, returning one
    stats row per predictor. Used for the age-at-scan robustness checks and for
    the adjusted model; the age-of-onset models keep going through fit_models()
    so their behaviour is untouched.
    """
    d = df_points.dropna(subset=["Similarity"] + list(xcols))
    n = len(d)
    if n < len(xcols) + 2 or any(np.ptp(d[c]) == 0 for c in xcols):
        print(f"  [skip] {analysis_label}: n = {n} for predictors {list(xcols)}")
        return []
    m = sm.OLS(d["Similarity"], sm.add_constant(d[list(xcols)])).fit()
    rows = []
    for c in xcols:
        ci = m.conf_int().loc[c]
        rows.append({"analysis": analysis_label, "model": model_label, "term": c,
                     "estimate": m.params[c], "ci_low_95": ci[0], "ci_high_95": ci[1],
                     "p_value": m.pvalues[c], "r_squared": m.rsquared, "n": n,
                     "n_sud_categories": len(sud_cats)})
    return rows


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
    print(f"  (similarity averaged over {len(sud_cats)} SUD categories: {sud_cats})")
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
    df_mean.assign(analysis="primary_disorder_by_population",
                   n_sud_categories=len(sud_cats)).to_csv(
        os.path.join(figs, "RQ3_ageonset_points_primary.csv"), index=False)
    df_unique.assign(analysis="sensitivity_unique_disorder",
                     n_sud_categories=len(sud_cats)).to_csv(
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

# -----------------------
# ROBUSTNESS (stats only, no figure)
# -----------------------
# Recomputed here rather than returned from plot_on_ax so the plotting path stays
# exactly as it was.
df_mean = (df_combined.groupby("PSY_main")
           .agg(Similarity=("Similarity", "mean"), Peak=("Peak", "first"),
                AgeAtScan=("AgeAtScan", "first"), Population=("Population", "first"))
           .reset_index())
df_mean_adult = df_mean[df_mean["Population"] == "adult"]

robust_stats = []

# (a) adults only, age of onset - already fitted inside plot_on_ax as
# "sensitivity_adults_only"; reported here in full rather than as a bare p.
print("\n--- ROBUSTNESS (a): age of onset, ADULT MAPS ONLY ---")
for r in [x for x in model_stats
          if x["analysis"] == "sensitivity_adults_only" and x["model"] == "linear"]:
    print(f"  n = {r['n']}  b = {r['estimate']:+.5f} per year  "
          f"95% CI [{r['ci_low_95']:+.5f}, {r['ci_high_95']:+.5f}]  "
          f"{fmt_p(r['p_value'])}  R2 = {r['r_squared']:.3f}")

if RUN_AGE_AT_SCAN:
    # (b) age at scan - expected to be null
    robust_stats += fit_ols_terms(df_mean, ["AgeAtScan"],
                                  "robustness_agescan_all_groups", "linear")
    robust_stats += fit_ols_terms(df_mean_adult, ["AgeAtScan"],
                                  "robustness_agescan_adults_only", "linear")
    # Peak alone on the SAME 13 groups that have an age at scan, so that any
    # change in the adjusted model is attributable to the adjustment and not to
    # the two groups lost for lack of an age-at-scan value.
    robust_stats += fit_ols_terms(df_mean.dropna(subset=["AgeAtScan"]), ["Peak"],
                                  "robustness_peak_alone_agescan_subset", "linear")
    robust_stats += fit_ols_terms(df_mean, ["Peak", "AgeAtScan"],
                                  "robustness_peak_adjusted_for_agescan",
                                  "linear, two predictors")
    robust_stats += fit_mixed(df_combined,
                              terms=(("AgeAtScan", ["AgeAtScan"]),),
                              analysis_label="mixed_effects_agescan_all_pairs")

    print("\n--- ROBUSTNESS (b): age at SCAN (expected: null) ---")
    for r in robust_stats:
        lab = f"{r['analysis']} [{r['term']}]"
        print(f"  {lab:<58s} n = {r['n']:3d}  b = {r['estimate']:+.5f}  "
              f"95% CI [{r['ci_low_95']:+.5f}, {r['ci_high_95']:+.5f}]  "
              f"{fmt_p(r['p_value'])}")

    both = df_mean.dropna(subset=["Peak", "AgeAtScan"])
    r_pear = np.corrcoef(both["Peak"], both["AgeAtScan"])[0, 1]
    print(f"\n  Peak vs age at scan across the {len(both)} groups with both: "
          f"r = {r_pear:+.3f}. The adjusted model is separating two correlated "
          f"predictors; read its Peak coefficient with that in mind.")
    robust_stats.append({"analysis": "diagnostic_peak_vs_agescan_correlation",
                         "model": "pearson", "term": "Peak~AgeAtScan",
                         "estimate": r_pear, "ci_low_95": np.nan,
                         "ci_high_95": np.nan, "p_value": np.nan,
                         "r_squared": np.nan, "n": len(both),
                         "n_sud_categories": len(sud_cats)})

model_stats = list(model_stats) + robust_stats

figs = os.path.join(BASE_DIR, "figures")
df_mean.assign(n_sud_categories=len(sud_cats)).round(6).to_csv(
    os.path.join(figs, "RQ3_ageonset_points_with_agescan.csv"), index=False)
inventory.round(6).to_csv(
    os.path.join(figs, "RQ3_ageonset_map_inventory.csv"), index=False)

pd.DataFrame(model_stats).round(6).to_csv(
    os.path.join(figs, "RQ3_ageonset_model_stats.csv"), index=False)
print(f"Saved: {os.path.join(figs, 'RQ3_ageonset_model_stats.csv')}")
print(f"Saved: {os.path.join(figs, 'RQ3_ageonset_points_with_agescan.csv')}")
print(f"Saved: {os.path.join(figs, 'RQ3_ageonset_map_inventory.csv')}")
if band_df is not None:
    band_df.round(6).to_csv(os.path.join(figs, "RQ3_ageonset_CI_band.csv"), index=False)
    print(f"Saved: {os.path.join(figs, 'RQ3_ageonset_CI_band.csv')}")