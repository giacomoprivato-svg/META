#!/usr/bin/env python3
"""
RQ3 - Neuroanatomical similarity vs comorbidity (ARD) and genetic correlation
=============================================================================
SPEARMAN primary version.

Changes vs the Pearson version
------------------------------
1. Primary metric = Spearman rho (reads RAW_cortex_spearman.csv). Axis labels
   updated to "Spearman rho".
2. Sample size of each test (n = number of pairs entering the regression) is
   annotated on the comorbidity and genetics linear panels.
3. Top-10 pairs are exported to CSV for BOTH comorbidity and genetics:
   the union of (top-10 by brain similarity) and (top-10 by the external
   measure), with a flag marking the pairs that populate the shaded overlap
   box in the figure.

BUGFIX (this version)
----------------------
Previously the comorbidity and genetics tables were merged onto the brain
dataframe SEQUENTIALLY (df.merge(comorb).merge(genetics)), both as inner
joins. Because the genetics table only covers 4 SUD categories (ALC, NIC,
COC, OPI; no CAN/ATS) and 8 PSY disorders (no CHR), that second inner join
silently dropped valid comorbidity rows for CAN, ATS, and CHR pairs as a
side effect -- shrinking the comorbidity analysis from its true n=35 down
to n=21. The two external measures are now merged onto the brain dataframe
INDEPENDENTLY (df_com_raw, df_gen_raw), each keeping all pairs for which
that specific external measure is available. Genetics is unaffected by the
fix (still n=32); comorbidity changes from n=21 to n=35, and its regression
result changes accordingly (see analysis notes / Results text).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy.stats import linregress
import statsmodels.api as sm

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

BASE_DIR = repo_dir
METRIC = "spearman"
METRIC_LABEL = "Spearman \u03c1"
ADULT_DIRS = [os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")]
OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

K = 10
SEED = 1
rng = np.random.default_rng(SEED)


def format_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3g}"


# ---------------- LOAD ----------------
def load_brain():
    rec = []
    for d in ADULT_DIRS:
        fp = os.path.join(d, f"RAW_cortex_{METRIC}.csv")
        print(f"[diag] load_brain(): looking for {fp}")
        print(f"[diag]   os.path.exists -> {os.path.exists(fp)}")
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"RAW_cortex_{METRIC}.csv not found at: {fp}\n"
                f"  BASE_DIR resolved to: {BASE_DIR}\n"
                f"  METRIC resolved to:   {METRIC!r}\n"
                "  Check that ALL_outputs_RQ1/adults_all exists relative to the repo root, "
                "and that this file wasn't left out of your local clone (e.g. .gitignore'd)."
            )
        brain = pd.read_csv(fp, index_col=0, sep=None, engine="python")
        print(f"[diag]   raw CSV shape (before melt): {brain.shape}")
        if brain.shape[1] == 0:
            raise ValueError(
                f"{fp} was read with 0 data columns. The file likely uses a different "
                "delimiter (e.g. ';' instead of ','). It has now been read with automatic "
                "separator detection (sep=None); if this error persists, open the file and "
                "check its delimiter and header row."
            )
        brain.index.name = "PSY"  # force a known name regardless of the CSV's header
        rec.append(brain.reset_index().melt(id_vars="PSY", var_name="SUD",
                   value_name="brain_similarity"))
    return pd.concat(rec, ignore_index=True)


def load_comorb():
    age = pd.read_excel(os.path.join(BASE_DIR, "data/raw/age_onset_prevalence_of_disorders.xlsx"), index_col=0)
    age = age.drop(columns=["SUD"], errors="ignore")
    _f = lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan
    age = age.apply(lambda col: col.map(_f))
    age.index.name = "PSY"
    return age.reset_index().melt(id_vars="PSY", var_name="SUD", value_name="Comorbidity")


def load_gen():
    gen = pd.read_excel(os.path.join(BASE_DIR, "data/raw/PSY_SUD_genetic_corr.xlsx"), index_col=0)
    gen.index.name = "PSY"
    return gen.reset_index().melt(id_vars="PSY", var_name="SUD", value_name="genetic_corr")


df_brain = load_brain()
print(f"[diag] df_brain: {df_brain.shape[0]} rows | PSY values: {sorted(df_brain['PSY'].unique())}")
assert len(df_brain) > 0, "load_brain() returned 0 rows -- check RAW_cortex_{METRIC}.csv path/METRIC"

_comorb = load_comorb()
print(f"[diag] load_comorb(): {_comorb.shape[0]} rows | PSY values: {sorted(_comorb['PSY'].unique())}")
assert len(_comorb) > 0, "load_comorb() returned 0 rows -- check age_onset_prevalence_of_disorders.xlsx path"

_gen = load_gen()
print(f"[diag] load_gen(): {_gen.shape[0]} rows | PSY values: {sorted(_gen['PSY'].unique())}")
assert len(_gen) > 0, "load_gen() returned 0 rows -- check PSY_SUD_genetic_corr.xlsx path"

df_com_raw = df_brain.merge(_comorb, on=["PSY", "SUD"])
print(f"[diag] df_brain x comorb merge: {df_com_raw.shape[0]} rows")
assert len(df_com_raw) > 0, (
    "Merge produced 0 rows: brain and comorbidity tables share no (PSY, SUD) keys. "
    f"brain PSY={sorted(df_brain['PSY'].unique())} vs comorb PSY={sorted(_comorb['PSY'].unique())} -- "
    "check for whitespace/case mismatches in disorder names."
)

df_gen_raw = df_brain.merge(_gen, on=["PSY", "SUD"])
print(f"[diag] df_brain x genetics merge: {df_gen_raw.shape[0]} rows")
assert len(df_gen_raw) > 0, "Merge produced 0 rows for genetics -- check PSY/SUD naming consistency."

df_com_raw["pair_id"] = df_com_raw["PSY"] + "-" + df_com_raw["SUD"]
df_gen_raw["pair_id"] = df_gen_raw["PSY"] + "-" + df_gen_raw["SUD"]

df_com_raw["Comorbidity_frac"] = df_com_raw["Comorbidity"] / 100

prev = pd.read_excel(os.path.join(BASE_DIR, "data/raw/SUD_general_prevalence.xlsx"))
prev_long = prev.melt(var_name="SUD", value_name="prev")
prev_long["prev"] = prev_long["prev"] / 100
df_com_raw["expected_comorbidity"] = df_com_raw["SUD"].map(dict(zip(prev_long["SUD"], prev_long["prev"])))
df_com_raw["ARD"] = df_com_raw["Comorbidity_frac"] - df_com_raw["expected_comorbidity"]

# drop Schizotypy if present, from both independently
df_com_raw = df_com_raw[~df_com_raw["PSY"].str.contains("Schizotyp", case=False, na=False)]
df_gen_raw = df_gen_raw[~df_gen_raw["PSY"].str.contains("Schizotyp", case=False, na=False)]

# ---------------- CLUSTERS ----------------
colors = {"Psychotic": "orange", "Neurodevelopmental": "green",
          "AN/OCD": "blue", "Mood/Anxiety": "purple"}


def assign_cluster(x):
    if x in ["SCZ", "BD", "CHR"]: return "Psychotic"
    if x in ["ASD", "ADHD"]: return "Neurodevelopmental"
    if x in ["AN", "OCD"]: return "AN/OCD"
    if x in ["MDD", "PTSD"]: return "Mood/Anxiety"
    return "Other"


df_com_raw["cluster"] = df_com_raw["PSY"].apply(assign_cluster)
df_gen_raw["cluster"] = df_gen_raw["PSY"].apply(assign_cluster)


# ---------------- HELPERS ----------------
def get_topk(d, col, k):
    return set(d.nlargest(min(k, len(d)), col)["pair_id"])


def export_top10(d, ext_col, out_csv):
    """Union of top-K by brain similarity and top-K by the external measure."""
    d = d.dropna(subset=["brain_similarity", ext_col]).copy()
    d["rank_brain"] = d["brain_similarity"].rank(ascending=False, method="min").astype(int)
    d["rank_external"] = d[ext_col].rank(ascending=False, method="min").astype(int)
    top_brain = get_topk(d, "brain_similarity", K)
    top_ext = get_topk(d, ext_col, K)
    d["top10_brain"] = d["pair_id"].isin(top_brain)
    d["top10_external"] = d["pair_id"].isin(top_ext)
    d["in_overlap_box"] = d["top10_brain"] & d["top10_external"]
    keep = d[d["top10_brain"] | d["top10_external"]].copy()
    keep = keep.sort_values(["in_overlap_box", ext_col], ascending=[False, False])
    cols = ["pair_id", "PSY", "SUD", "cluster", "brain_similarity", ext_col,
            "rank_brain", "rank_external", "top10_brain", "top10_external", "in_overlap_box"]
    keep = keep[cols].round({"brain_similarity": 4, ext_col: 4})
    keep.to_csv(out_csv, index=False)
    print(f"  saved {out_csv}  ({keep['in_overlap_box'].sum()} pairs in overlap box, "
          f"{len(keep)} rows total)")
    return keep


# ---------------- SCATTER OVERLAP ----------------
def plot_scatter(ax, d, x, y):
    d = d.dropna(subset=[x, y])
    top_x_ids = get_topk(d, x, K)
    top_y_ids = get_topk(d, y, K)
    overlap_ids = top_x_ids & top_y_ids
    overlap = d[d["pair_id"].isin(overlap_ids)]

    # Box position is defined by the top-K RANK THRESHOLD on each axis (the Kth-largest
    # value), not by the actual overlapping points. This way the box always marks the
    # "top-10 in both measures" corner of the plot, even when no point actually falls
    # inside it (as for genetics) -- it will simply render empty.
    thresh_x = d[d["pair_id"].isin(top_x_ids)][x].min()
    thresh_y = d[d["pair_id"].isin(top_y_ids)][y].min()
    x_min, x_max = thresh_x, d[x].max()
    y_min, y_max = thresh_y, d[y].max()
    pad_x = 0.05 * (d[x].max() - d[x].min())
    pad_y = 0.05 * (d[y].max() - d[y].min())
    x_min -= 0; x_max += pad_x; y_min -= 0; y_max += pad_y

    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           color="lightblue", alpha=0.2, zorder=0))
    for c in d["cluster"].unique():
        sub = d[d["cluster"] == c]
        ax.scatter(sub[x], sub[y], color=colors[c], alpha=0.6)
    if len(overlap) > 0:
        ax.scatter(overlap[x], overlap[y], c=[colors[c] for c in overlap["cluster"]],
                   edgecolor="black", s=100, linewidth=1.5)
    for xa, xb in [(x_min, x_max)]:
        ax.plot([xa, xb], [y_min, y_min], color="black")
        ax.plot([xa, xb], [y_max, y_max], color="black")
    ax.plot([x_min, x_min], [y_min, y_max], color="black")
    ax.plot([x_max, x_max], [y_min, y_max], color="black")
    for v in [x_min, x_max]:
        ax.axvline(v, linestyle="--", color="gray", alpha=0.5, zorder=-1)
    for h in [y_min, y_max]:
        ax.axhline(h, linestyle="--", color="gray", alpha=0.5, zorder=-1)
    ax.set_xlabel(METRIC_LABEL, fontsize=16)
    ax.set_ylabel("Comorbidity (ARD)" if y == "ARD" else "Genetic Correlation", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)


def scatter(ax, d, x, y):
    for c in d["cluster"].unique():
        sub = d[d["cluster"] == c]
        ax.scatter(sub[x], sub[y], alpha=0.35, s=40, color=colors[c])
    ax.set_xlabel(METRIC_LABEL, fontsize=16)
    ax.set_ylabel("Comorbidity (ARD)" if y == "ARD" else "Genetic Correlation", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)


def linfit_with_ci(ax, x, y):
    slope, intercept, r, p_lin, _ = linregress(x, y)
    xg = np.linspace(x.min(), x.max(), 200)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    pred = model.get_prediction(sm.add_constant(xg))
    ax.plot(xg, pred.predicted_mean, color="black", lw=2)
    ci = pred.conf_int()
    ax.fill_between(xg, ci[:, 0], ci[:, 1], color="black", alpha=0.15)
    return r, p_lin


# =========================================================
d_com = df_com_raw.dropna(subset=["brain_similarity", "ARD"])
d_gen = df_gen_raw.dropna(subset=["brain_similarity", "genetic_corr"])
n_com = len(d_com)
n_gen = len(d_gen)

# ---- export top-10 CSVs ----
print("Exporting top-10 pair CSVs:")
export_top10(df_com_raw, "ARD", os.path.join(OUT_DIR, "RQ3_top10_comorbidity.csv"))
export_top10(df_gen_raw, "genetic_corr", os.path.join(OUT_DIR, "RQ3_top10_genetics.csv"))

# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
ax2, ax1, ax4, ax3 = axes.flatten()

# A - overlap ARD
plot_scatter(ax1, d_com, "brain_similarity", "ARD")

# B - linear ARD
scatter(ax2, d_com, "brain_similarity", "ARD")
r, p_lin = linfit_with_ci(ax2, d_com["brain_similarity"].values, d_com["ARD"].values)
ax2.text(0.02, 0.98, f"r = {r:.2f}\n{format_p(p_lin)}\nn = {n_com}",
         transform=ax2.transAxes, va="top", fontsize=16)

# C - overlap genetics
plot_scatter(ax3, d_gen, "brain_similarity", "genetic_corr")

# D - linear genetics
scatter(ax4, d_gen, "brain_similarity", "genetic_corr")
r, p_lin = linfit_with_ci(ax4, d_gen["brain_similarity"].values, d_gen["genetic_corr"].values)
ax4.text(0.02, 0.98, f"r = {r:.2f}\n{format_p(p_lin)}\nn = {n_gen}",
         transform=ax4.transAxes, va="top", fontsize=16)

legend_elements = [Patch(facecolor=colors[c], label=c) for c in colors]
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.005), fontsize=16, handlelength=1.5, handleheight=1.5)
plt.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(OUT_DIR, "RQ3_gen_com_spearman.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {out}  (n_comorbidity={n_com}, n_genetics={n_gen})")