#!/usr/bin/env python3
"""
Supplementary Figures 5 & 6 — cluster-stratified association between cortical
morphometric similarity to SUD (PRIMARY METRIC: Spearman rho) and
    S5: epidemiological comorbidity, expressed as attributable risk difference (ARD)
    S6: cross-disorder genetic correlation

Each figure is a 2x2 grid, one panel per psychiatric cluster
(AN/OCD, Neurodevelopmental, Mood/Anxiety, Psychotic). Each point is a PSY-SUD
pair. A linear (OLS) fit is drawn per cluster; Pearson r and an FDR-corrected
p-value (BH across the four clusters, within each figure) are printed in-panel.

Fixes relative to the legacy RQ3_scatter_cluster.py:
  * Psychotic cluster now includes CHR (was silently dropped).
  * Similarity is the Spearman rho (RAW_cortex_spearman), not Euclidean Z.
  * Comorbidity is ARD (observed - expected from SUD base rates), not raw %.
  * Adults-only, matching the disorder-level comorbidity/genetic analyses.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
BASE = repo_dir
ADULT_DIR = os.path.join(BASE, "ALL_outputs_RQ1", "adults_all")
DATA = os.path.join(BASE, "data", "raw")

SIM_FILE = os.path.join(ADULT_DIR, "RAW_cortex_spearman.csv")   # observed rho

PSY_CLUSTERS = {
    "AN/OCD":             ["AN", "OCD"],
    "Neurodevelopmental": ["ASD", "ADHD"],
    "Mood/Anxiety":       ["MDD", "PTSD"],
    "Psychotic":          ["SCZ", "BD", "CHR"],
}
CLUSTER_ORDER = ["AN/OCD", "Neurodevelopmental", "Mood/Anxiety", "Psychotic"]
CLUSTER_COLORS = {
    "AN/OCD": "#1f77b4",
    "Neurodevelopmental": "#2ca02c",
    "Mood/Anxiety": "#9467bd",
    "Psychotic": "#ff7f0e",
}


def assign_cluster(psy):
    for c, members in PSY_CLUSTERS.items():
        if psy in members:
            return c
    return None


# ---------------------------------------------------------------------------
# Build the per-pair table (similarity + ARD + genetic correlation)
# ---------------------------------------------------------------------------
def _read_matrix_csv(path):
    """Robust reader for pipeline CSVs that may have been re-saved by Excel
    (Italian locale) with ';' delimiters and/or ',' decimals. The delimiter is
    sniffed from the header line."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, index_col=0, sep=sep)
    for c in df.columns:                        # coerce comma-decimal strings
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False),
                errors="coerce")
    return df


def load_similarity():
    sim = _read_matrix_csv(SIM_FILE)
    sim.index.name = "PSY"                      # robust to whatever the CSV header is
    return sim.reset_index().melt(id_vars="PSY", var_name="SUD",
                                  value_name="similarity")


def load_comorbidity_ARD():
    age = pd.read_excel(os.path.join(DATA, "age_onset_prevalence_of_disorders.xlsx"),
                        index_col=0)
    age = age.drop(columns=["SUD"], errors="ignore")
    def _num(x):
        return float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan
    age = age.apply(lambda col: col.map(_num))
    age.index.name = "PSY"
    com = age.reset_index().melt(id_vars="PSY", var_name="SUD",
                                 value_name="Comorbidity")
    com["Comorbidity_frac"] = com["Comorbidity"] / 100.0

    prev = pd.read_excel(os.path.join(DATA, "SUD_general_prevalence.xlsx"))
    prev_long = prev.melt(var_name="SUD", value_name="prev")
    prev_long["prev"] = prev_long["prev"] / 100.0
    exp = dict(zip(prev_long["SUD"], prev_long["prev"]))

    com["expected"] = com["SUD"].map(exp)
    com["ARD"] = com["Comorbidity_frac"] - com["expected"]
    return com[["PSY", "SUD", "ARD"]]


def load_genetic():
    gen = pd.read_excel(os.path.join(DATA, "PSY_SUD_genetic_corr.xlsx"), index_col=0)
    gen.index.name = "PSY"
    return gen.reset_index().melt(id_vars="PSY", var_name="SUD",
                                  value_name="genetic_corr")


def _key(s):
    """Normalise a label for merging: string, stripped, uppercased."""
    return s.astype(str).str.strip().str.upper()


def _merge_norm(left, right, value_col):
    """Merge on normalised PSY/SUD keys, keeping left's original labels."""
    L = left.copy()
    R = right.copy()
    L["_P"], L["_S"] = _key(L["PSY"]), _key(L["SUD"])
    R["_P"], R["_S"] = _key(R["PSY"]), _key(R["SUD"])
    R = R[["_P", "_S", value_col]]
    out = L.merge(R, on=["_P", "_S"], how="left")
    matched = out[value_col].notna().sum()
    if matched == 0:
        # tell the user exactly why nothing matched
        print(f"  [!] 0 matches for '{value_col}'.")
        print(f"      similarity SUD labels: {sorted(L['_S'].unique())}")
        print(f"      {value_col} SUD labels: {sorted(R['_S'].unique())}")
        print(f"      similarity PSY labels: {sorted(L['_P'].unique())}")
        print(f"      {value_col} PSY labels: {sorted(R['_P'].unique())}")
    else:
        print(f"  [ok] '{value_col}': {matched} pairs matched.")
    return out.drop(columns=["_P", "_S"])


def build_table():
    df = load_similarity()
    print("Merging external measures on normalised (stripped, upper-case) labels:")
    df = _merge_norm(df, load_comorbidity_ARD(), "ARD")
    df = _merge_norm(df, load_genetic(), "genetic_corr")
    df["Cluster"] = df["PSY"].apply(assign_cluster)
    df = df.dropna(subset=["Cluster"])
    print(f"Rows with a cluster: {len(df)} | "
          f"non-NaN ARD: {df['ARD'].notna().sum()} | "
          f"non-NaN genetic: {df['genetic_corr'].notna().sum()}")
    return df


# ---------------------------------------------------------------------------
# Per-cluster stats with BH-FDR across clusters (within one measure)
# ---------------------------------------------------------------------------
def cluster_stats(df, ycol):
    rows = []
    for cl in CLUSTER_ORDER:
        sub = df[(df["Cluster"] == cl)].dropna(subset=["similarity", ycol])
        if len(sub) >= 3:
            r, p = pearsonr(sub["similarity"], sub[ycol])
        else:
            r, p = np.nan, np.nan
        rows.append({"Cluster": cl, "r": r, "p": p, "n": len(sub)})
    res = pd.DataFrame(rows)
    valid = res["p"].notna()
    res["pFDR"] = np.nan
    if valid.sum() > 0:
        _, q = fdrcorrection(res.loc[valid, "p"].values)
        res.loc[valid, "pFDR"] = q
    return res


# ---------------------------------------------------------------------------
# Figure: 2x2, one panel per cluster
# ---------------------------------------------------------------------------
def make_figure(df, ycol, ylabel, title, outpath):
    stats = cluster_stats(df, ycol)
    stats_by_cl = stats.set_index("Cluster")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, cl in zip(axes, CLUSTER_ORDER):
        sub = df[(df["Cluster"] == cl)].dropna(subset=["similarity", ycol])
        color = CLUSTER_COLORS[cl]
        ax.scatter(sub["similarity"], sub[ycol], s=70, alpha=0.75,
                   color=color, edgecolor="white", linewidth=0.5)

        if len(sub) >= 2:
            m, b = np.polyfit(sub["similarity"], sub[ycol], 1)
            xs = np.linspace(sub["similarity"].min(), sub["similarity"].max(), 100)
            ax.plot(xs, m * xs + b, color="black", lw=1.5)

        row = stats_by_cl.loc[cl]
        r_txt = "r = n/a" if np.isnan(row["r"]) else f"r = {row['r']:.2f}"
        p_txt = "pFDR = n/a" if np.isnan(row["pFDR"]) else f"pFDR = {row['pFDR']:.3f}"
        ax.text(0.04, 0.96, f"{r_txt}\n{p_txt}", transform=ax.transAxes,
                va="top", ha="left", fontsize=12, fontweight="bold")

        ax.set_title(cl, fontsize=14)
        ax.set_xlabel("Spearman similarity (\u03c1)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(False)

    fig.suptitle(title, fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")
    return stats


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_table()

    s5 = make_figure(
        df, "ARD", "Comorbidity (ARD)",
        "Supplementary Figure 5 | Cluster-stratified similarity vs comorbidity",
        os.path.join(BASE, "SuppFig5_cluster_comorbidity_spearman.png"),
    )
    s6 = make_figure(
        df, "genetic_corr", "Genetic correlation",
        "Supplementary Figure 6 | Cluster-stratified similarity vs genetic correlation",
        os.path.join(BASE, "SuppFig6_cluster_genetics_spearman.png"),
    )

    print("\n--- S5 comorbidity (ARD) ---")
    print(s5.to_string(index=False))
    print("\n--- S6 genetic correlation ---")
    print(s6.to_string(index=False))

    out = pd.concat([s5.assign(measure="comorbidity_ARD"),
                     s6.assign(measure="genetic_corr")], ignore_index=True)
    out.to_csv(os.path.join(BASE, "SuppFig5_6_cluster_stats_spearman.csv"), index=False)
    print("\nSaved SuppFig5_6_cluster_stats_spearman.csv")