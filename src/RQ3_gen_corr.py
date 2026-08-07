#!/usr/bin/env python3
"""
RQ3 — step 2: genetic correlation (figure) and comorbidity (statistics only)
============================================================================

Figure: 1 x 2, genetics only. Left = PSY-SUD, right = PSY-PSY (the
within-domain reference). Panel geometry, figsize (16, 5), colours and output
names are UNCHANGED. Comorbidity remains descriptive: numbers and full ranking
exports, no scatter anywhere, per the PI's final call.

WHAT CHANGED
------------
1. THE CANNABIS/COCAINE MISPAIRING IS NOW FIXED IN THE DATA, AND IT MOVES THE
   RESULT. The column previously labelled COC in PSY_SUD_genetic_corr.xlsx
   held the CUD (cannabis) estimates — the Grotzinger panel has no cocaine
   phenotype at all — so cannabis genetic correlations were being regressed
   against COCAINE morphometric similarity. With the column relabelled CAN it
   now meets the cannabis map. On the updated brain maps this takes the
   PSY-SUD association from r = +0.026 to r = +0.248 (both n = 28, both
   non-significant). The guard below still refuses to run on a file
   containing a COC column, so the old pairing cannot come back.

2. PD IS ABSENT FROM THE GENETICS. Grotzinger reports no panic disorder
   phenotype, so PSY_SUD_genetic_corr.xlsx now has 7 disorders (AN, OCD, SCZ,
   BD, ASD, ADHD, MDD) and PSY_PSY_genetic_corr.xlsx is 7 x 7 -> 21 pairs
   (was 28). Consequence to state in the Limitations: the Mood/Anxiety cluster
   is represented in the genetic analyses by MDD alone, while RQ1 and RQ2
   have both MDD and PD. This is a coverage gap, not a null result.

3. COMORBIDITY COVERAGE IS REPORTED, NOT ASSUMED. The comorbidity workbook is
   still keyed on PTSD, so with PD in the brain maps the merge silently drops
   both. The script now prints exactly which disorders and which pairs were
   lost on each side instead of quietly shrinking n — the same class of bug as
   the earlier sequential-inner-join problem, which cut n from 35 to 21 and
   masked the association.

   THE COMORBIDITY QUANTITY MUST BE P(SUD | PSY). Any row that reports
   P(PSY | SUD) instead is measuring the reverse conditional and cannot be
   pooled with the others. Check any newly added source against this before
   putting it in the workbook.

4. CLUSTERS COME FROM RQ1_common (the local assign_cluster still had PTSD).

5. No statsmodels dependency for the fits: the OLS line and its 95% band are
   computed in closed form, matching the RQ1 figure scripts.

Just press Run.
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr, pearsonr, t as tdist

import RQ1_common as C

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = repo_dir
METRIC = "spearman"
METRIC_LABEL = "Spearman \u03c1"
ADULT_DIRS = [os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")]
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE_PSY_COMORBIDITY = ["CHR", "Schizotypic"]
COMORBIDITY_PAIR_WHITELIST = None

N_MANTEL_PERM = 10000
MANTEL_SEED = 42

COL_PSY_SUD = "#1f77b4"
COL_PSY_PSY = "#d62728"
# =========================================================


def format_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3g}"


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these exist:\n  " + "\n  ".join(paths))


def ols_band(x, y, xg, alpha=0.05):
    X = np.column_stack([np.ones(x.size), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    dof = x.size - 2
    s2 = float((y - X @ beta) @ (y - X @ beta)) / dof
    XtXi = np.linalg.inv(X.T @ X)
    Xg = np.column_stack([np.ones(xg.size), xg])
    fit = Xg @ beta
    se = np.sqrt(s2 * np.einsum("ij,jk,ik->i", Xg, XtXi, Xg))
    h = tdist.ppf(1 - alpha / 2, dof) * se
    return fit, fit - h, fit + h


def ols_fit(ax, x, y, color):
    r, p = linregress(x, y)[2:4]
    xg = np.linspace(x.min(), x.max(), 200)
    fit, lo, hi = ols_band(x, y, xg)
    ax.plot(xg, fit, color=color, lw=2)
    ax.fill_between(xg, lo, hi, color=color, alpha=0.15, lw=0)
    return r, p


# =========================================================
# LOADERS
# =========================================================
def load_brain():
    rec = []
    for d in ADULT_DIRS:
        fp = os.path.join(d, f"RAW_cortex_{METRIC}.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"{fp}\nRun RQ1_step1_cortex_similarity.py first.")
        brain = pd.read_csv(fp, index_col=0, sep=None, engine="python")
        if brain.shape[1] == 0:
            raise ValueError(f"{fp} read with 0 data columns — check its delimiter.")
        brain.index.name = "PSY"
        rec.append(brain.reset_index().melt(id_vars="PSY", var_name="SUD",
                                            value_name="brain_similarity"))
    return pd.concat(rec, ignore_index=True)


def load_brain_psy_psy():
    T, cols = C.read_maps(os.path.join(DATA_DIR, "PSY_adults.xlsx"), C.N_CORTEX)
    T = T[[c for c in cols if "Schizotyp" not in c]]
    rows = [dict(PSY=a, SUD=b, brain_similarity=spearmanr(T[a], T[b]).statistic)
            for a, b in itertools.combinations(sorted(T.columns), 2)]
    return pd.DataFrame(rows), T


def load_comorb():
    fp = first_existing(
        os.path.join(DATA_DIR, "PSY_SUD_comorbidity_prevalence.xlsx"),
        os.path.join(DATA_DIR, "age_onset_prevalence_of_disorders.xlsx"))
    tab = pd.read_excel(fp, index_col=0).drop(columns=["SUD"], errors="ignore")
    _f = lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan
    tab = tab.apply(lambda col: col.map(_f))
    tab.index.name = "PSY"
    print(f"[comorbidity] source: {os.path.basename(fp)}  disorders: {list(tab.index)}")
    return tab.reset_index().melt(id_vars="PSY", var_name="SUD", value_name="Comorbidity")


def load_gen_psy_sud():
    gen = pd.read_excel(os.path.join(DATA_DIR, "PSY_SUD_genetic_corr.xlsx"), index_col=0)
    if "COC" in gen.columns:
        raise ValueError(
            "PSY_SUD_genetic_corr.xlsx still has a 'COC' column. The Grotzinger panel "
            "contains no cocaine phenotype — those values are CUD (cannabis). "
            "Relabel the column CAN before running this.")
    gen.index.name = "PSY"
    return gen.reset_index().melt(id_vars="PSY", var_name="SUD", value_name="genetic_corr")


def load_gen_psy_psy():
    M = pd.read_excel(os.path.join(DATA_DIR, "PSY_PSY_genetic_corr.xlsx"), index_col=0)
    rows = [dict(PSY=a, SUD=b, genetic_corr=M.loc[a, b])
            for a, b in itertools.combinations(sorted(M.columns), 2)]
    return pd.DataFrame(rows), M


def report_coverage(left, right, keys, label):
    """Say out loud what each side contributes, instead of merging in silence."""
    lp, rp = set(left[keys[0]]), set(right[keys[0]])
    ls, rs = set(left[keys[1]]), set(right[keys[1]])
    print(f"[{label}] brain-only disorders: {sorted(lp - rp) or 'none'} | "
          f"{label}-only disorders: {sorted(rp - lp) or 'none'}")
    print(f"[{label}] brain-only substances: {sorted(ls - rs) or 'none'} | "
          f"{label}-only substances: {sorted(rs - ls) or 'none'}")


# =========================================================
# BUILD
# =========================================================
df_brain = load_brain()
print(f"[load] PSY x SUD similarity: {len(df_brain)} cells")

_comorb = load_comorb()
_gen_ps = load_gen_psy_sud()
report_coverage(df_brain, _comorb, ("PSY", "SUD"), "comorbidity")
report_coverage(df_brain, _gen_ps, ("PSY", "SUD"), "genetics")

df_com_raw = df_brain.merge(_comorb, on=["PSY", "SUD"])
df_gen_raw = df_brain.merge(_gen_ps, on=["PSY", "SUD"])
assert len(df_com_raw) and len(df_gen_raw), "Merge produced 0 rows — check PSY/SUD naming."

df_com_raw["Comorbidity_frac"] = df_com_raw["Comorbidity"] / 100
prev = pd.read_excel(os.path.join(DATA_DIR, "SUD_general_prevalence.xlsx"))
prev_long = prev.melt(var_name="SUD", value_name="prev")
prev_long["prev"] = prev_long["prev"] / 100
df_com_raw["expected_comorbidity"] = df_com_raw["SUD"].map(
    dict(zip(prev_long["SUD"], prev_long["prev"])))
df_com_raw["ARD"] = df_com_raw["Comorbidity_frac"] - df_com_raw["expected_comorbidity"]

for d in (df_com_raw, df_gen_raw):
    d["pair_id"] = d["PSY"] + "-" + d["SUD"]
    d.drop(d[d["PSY"].str.contains("Schizotyp", case=False, na=False)].index, inplace=True)
    d["cluster"] = d["PSY"].map(C.CLUSTERS).fillna("Other")

bpp, PSY_MAPS = load_brain_psy_psy()
gpp, GEN_MAT = load_gen_psy_psy()
df_genpp = bpp.merge(gpp, on=["PSY", "SUD"])
df_genpp["pair_id"] = df_genpp["PSY"] + "-" + df_genpp["SUD"]
n_dis_pp = len(set(df_genpp["PSY"]) | set(df_genpp["SUD"]))
assert len(df_genpp) == n_dis_pp * (n_dis_pp - 1) // 2, (
    f"Expected all unique PSY-PSY pairs, got {len(df_genpp)} from {n_dis_pp} disorders.")
print(f"[load] PSY x PSY pairs with genetics: {len(df_genpp)} ({n_dis_pp} disorders: "
      f"{sorted(set(df_genpp['PSY']) | set(df_genpp['SUD']))})")

d_com_all = df_com_raw.dropna(subset=["brain_similarity", "ARD"]).copy()
d_com = d_com_all[~d_com_all["PSY"].isin(EXCLUDE_PSY_COMORBIDITY)].copy()
if COMORBIDITY_PAIR_WHITELIST is not None:
    keep = {f"{p}-{s}" for p, s in COMORBIDITY_PAIR_WHITELIST}
    missing = keep - set(d_com["pair_id"])
    if missing:
        raise ValueError(f"Whitelisted pairs absent from the data: {sorted(missing)}")
    d_com = d_com[d_com["pair_id"].isin(keep)].copy()
d_gen = df_gen_raw.dropna(subset=["brain_similarity", "genetic_corr"]).copy()

print(f"\n[comorbidity] all available pairs   n = {len(d_com_all)} "
      f"({sorted(set(d_com_all['PSY']))})")
print(f"[comorbidity] after restriction     n = {len(d_com)}  "
      f"(excluded: {EXCLUDE_PSY_COMORBIDITY})")
print(f"[genetics]    PSY-SUD pairs         n = {len(d_gen)}  "
      f"({sorted(set(d_gen['PSY']))} x {sorted(set(d_gen['SUD']))})")


# =========================================================
# FULL RANKING EXPORTS (S22 / S23) — descriptive only
# =========================================================
def export_ranking(d, ext_col, ext_label, out_csv):
    d = d.dropna(subset=["brain_similarity", ext_col]).copy()
    d["rank_morphometric"] = d["brain_similarity"].rank(ascending=False, method="min").astype(int)
    d["rank_external"] = d[ext_col].rank(ascending=False, method="min").astype(int)
    d["rank_difference"] = d["rank_external"] - d["rank_morphometric"]
    out = d[["pair_id", "PSY", "SUD", "cluster", "brain_similarity", ext_col,
             "rank_morphometric", "rank_external", "rank_difference"]].copy()
    out.columns = ["PSY-SUD pair", "PSY", "SUD", "Cluster", "Morphometric similarity",
                   ext_label, "Rank morphometric",
                   f"Rank {ext_label.split(' (')[0].lower()}", "Rank difference"]
    out = out.sort_values("Rank morphometric").round(
        {"Morphometric similarity": 4, ext_label: 4})
    out.to_csv(out_csv, index=False)
    print(f"  saved {os.path.basename(out_csv)}  ({len(out)} pairs)")
    return out


print("\nExporting full ranking tables (Supplementary S22/S23):")
export_ranking(d_gen, "genetic_corr", "Genetic correlation",
               os.path.join(OUT_DIR, "RQ3_S22_ranking_genetics_all_pairs.csv"))
export_ranking(d_com, "ARD", "Comorbidity (ARD)",
               os.path.join(OUT_DIR, "RQ3_S23_ranking_comorbidity_all_pairs.csv"))
export_ranking(d_com_all, "ARD", "Comorbidity (ARD)",
               os.path.join(OUT_DIR, "RQ3_S23_ranking_comorbidity_unrestricted.csv"))


# =========================================================
# PERMUTATION TESTS
# =========================================================
def mantel_psy_psy(brain_maps, gen_matrix, n_perm=N_MANTEL_PERM, seed=MANTEL_SEED):
    disorders = [d for d in gen_matrix.columns if d in brain_maps.columns]
    n = len(disorders)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = spearmanr(brain_maps[disorders[i]],
                                          brain_maps[disorders[j]]).statistic
    G = gen_matrix.loc[disorders, disorders].values.astype(float)
    iu = np.triu_indices(n, 1)
    obs = pearsonr(M[iu], G[iu]).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        o = rng.permutation(n)
        null[k] = pearsonr(M[np.ix_(o, o)][iu], G[iu]).statistic
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1), n, len(iu[0])


def mantel_bipartite(df_pairs, val_a, val_b, n_perm=N_MANTEL_PERM, seed=MANTEL_SEED):
    """
    Bipartite analogue of the Mantel test for a rectangular PSY x SUD grid.
    The pairs are not independent: each PSY disorder appears once per SUD and
    each SUD once per PSY, so a naive per-pair Pearson overstates the effective
    n exactly as it would for PSY-PSY. Rows and columns are permuted within
    their own domain, preserving that structure.
    """
    A = df_pairs.pivot(index="PSY", columns="SUD", values=val_a)
    B = df_pairs.pivot(index="PSY", columns="SUD", values=val_b).loc[A.index, A.columns]
    A, B = A.values.astype(float), B.values.astype(float)
    if np.isnan(A).any() or np.isnan(B).any():
        raise ValueError("mantel_bipartite() needs a complete PSY x SUD grid; found NaNs. "
                         "Restrict to the disorders and substances present in both sources.")
    obs = pearsonr(A.ravel(), B.ravel()).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = pearsonr(A[rng.permutation(A.shape[0])][:, rng.permutation(A.shape[1])].ravel(),
                           B.ravel()).statistic
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1), A.shape[0], A.shape[1]


# =========================================================
# COMORBIDITY STATS (text/CSV only, no figure)
# =========================================================
r_c, p_c = linregress(d_com["brain_similarity"].values, d_com["ARD"].values)[2:4]
r_ca, p_ca = linregress(d_com_all["brain_similarity"].values, d_com_all["ARD"].values)[2:4]
print(f"\n[comorbidity] restricted   r = {r_c:+.3f}, {format_p(p_c)}, n = {len(d_com)}")
print(f"[comorbidity] unrestricted r = {r_ca:+.3f}, {format_p(p_ca)}, n = {len(d_com_all)}")


# =========================================================
# FIGURE: 1 x 2, genetics only
# =========================================================
fig, (ax_ps, ax_pp) = plt.subplots(1, 2, figsize=(16, 5))

ax_ps.scatter(d_gen["brain_similarity"], d_gen["genetic_corr"],
              s=55, alpha=0.65, color=COL_PSY_SUD, edgecolor="none")
r_gs, p_gs_naive = ols_fit(ax_ps, d_gen["brain_similarity"].values,
                           d_gen["genetic_corr"].values, COL_PSY_SUD)
r_gs_perm, p_gs_perm, n_psy, n_sud = mantel_bipartite(
    d_gen, "brain_similarity", "genetic_corr")
ax_ps.text(0.02, 0.98, f"r = {r_gs:.2f}\n{format_p(p_gs_perm)}\nn = {len(d_gen)}",
           transform=ax_ps.transAxes, va="top", fontsize=13)
ax_ps.set_xlabel(METRIC_LABEL, fontsize=15)
ax_ps.set_ylabel("Genetic correlation", fontsize=15)
ax_ps.set_title("PSY\u2013SUD", fontsize=15)
ax_ps.tick_params(axis="both", labelsize=13)
print(f"\n[genetics] PSY-SUD  r = {r_gs:+.3f}")
print(f"[genetics] PSY-SUD  naive OLS {format_p(p_gs_naive)}  <-- do NOT report")
print(f"[genetics] PSY-SUD  permutation {format_p(p_gs_perm)} "
      f"({n_psy} PSY x {n_sud} SUD)  <-- report this")

ax_pp.scatter(df_genpp["brain_similarity"], df_genpp["genetic_corr"],
              s=55, alpha=0.65, color=COL_PSY_PSY, edgecolor="none")
r_pp, p_pp_naive = ols_fit(ax_pp, df_genpp["brain_similarity"].values,
                           df_genpp["genetic_corr"].values, COL_PSY_PSY)
r_mantel, p_mantel, n_dis, n_pairs = mantel_psy_psy(PSY_MAPS, GEN_MAT)
ax_pp.text(0.02, 0.98, f"r = {r_pp:.2f}\n{format_p(p_mantel)}\nn = {len(df_genpp)}",
           transform=ax_pp.transAxes, va="top", fontsize=13)
ax_pp.set_xlabel(METRIC_LABEL, fontsize=15)
ax_pp.set_ylabel("Genetic correlation", fontsize=15)
ax_pp.set_title("PSY\u2013PSY", fontsize=15)
ax_pp.tick_params(axis="both", labelsize=13)
print(f"\n[genetics] PSY-PSY  r = {r_pp:+.3f}")
print(f"[genetics] PSY-PSY  naive OLS {format_p(p_pp_naive)}  <-- do NOT report")
print(f"[genetics] PSY-PSY  Mantel {format_p(p_mantel)} "
      f"({n_dis} disorders, {n_pairs} pairs)  <-- report this")

plt.tight_layout()
out = os.path.join(OUT_DIR, "RQ3_genetics_spearman.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\nSaved: {out}")

summary = [
    dict(analysis="genetics_PSY_SUD", test="OLS/Pearson (naive — do not report)",
         r=r_gs, p=p_gs_naive, n=len(d_gen)),
    dict(analysis="genetics_PSY_SUD", test="Bipartite permutation (report this)",
         r=r_gs_perm, p=p_gs_perm, n=len(d_gen)),
    dict(analysis="genetics_PSY_PSY", test="OLS/Pearson (naive — do not report)",
         r=r_pp, p=p_pp_naive, n=len(df_genpp)),
    dict(analysis="genetics_PSY_PSY", test="Mantel permutation (report this)",
         r=r_mantel, p=p_mantel, n=len(df_genpp)),
    dict(analysis="comorbidity_restricted", test="OLS/Pearson (descriptive, not plotted)",
         r=r_c, p=p_c, n=len(d_com)),
    dict(analysis="comorbidity_unrestricted", test="OLS/Pearson (descriptive, not plotted)",
         r=r_ca, p=p_ca, n=len(d_com_all)),
]
pd.DataFrame(summary).round(6).to_csv(
    os.path.join(OUT_DIR, "RQ3_gen_com_model_stats.csv"), index=False)
print(f"Saved: {os.path.join(OUT_DIR, 'RQ3_gen_com_model_stats.csv')}")