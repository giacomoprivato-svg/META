#!/usr/bin/env python3
"""
RQ3 — step 3: cluster-stratified similarity vs genetic correlation (Supplementary)

"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import RQ1_common as C

# ================= CONFIG — edit, then press Run =================
MIN_N = 6              # below this, no fit is drawn and no r is reported
N_PERM = 10000
SEED = 42
FIGSIZE = (12, 10)
FS_TITLE, FS_LABEL, FS_TEXT, FS_SUP = 14, 12, 12, 15
CLUSTER_ORDER = ["AN/OCD", "Neurodevelopmental", "Mood/Anxiety", "Psychotic"]
CLUSTER_COLORS = {"AN/OCD": "#1f77b4", "Neurodevelopmental": "#2ca02c",
                  "Mood/Anxiety": "#9467bd", "Psychotic": "#ff7f0e"}
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
ADULT_DIR = os.path.join(repo_dir, "ALL_outputs_RQ1", "adults_all")
DATA = os.path.join(repo_dir, "data", "raw")
OUT_DIR = os.path.join(repo_dir, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SIM_FILE = os.path.join(ADULT_DIR, f"RAW_cortex_{C.PRIMARY}.csv")


def _read_matrix_csv(path):
    """Reader tolerant of Excel re-saves (Italian locale: ';' and ',' decimals)."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, index_col=0, sep=sep)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False),
                                  errors="coerce")
    return df


def build_table():
    if not os.path.exists(SIM_FILE):
        raise FileNotFoundError(f"{SIM_FILE}\nRun RQ1_step1_cortex_similarity.py first.")
    sim = _read_matrix_csv(SIM_FILE)
    sim.index.name = "PSY"
    sim = sim.reset_index().melt(id_vars="PSY", var_name="SUD", value_name="similarity")

    gen = pd.read_excel(os.path.join(DATA, "PSY_SUD_genetic_corr.xlsx"), index_col=0)
    if "COC" in gen.columns:
        raise ValueError("PSY_SUD_genetic_corr.xlsx still has a 'COC' column — "
                         "those values are cannabis (CUD). Relabel it CAN.")
    gen.index.name = "PSY"
    gen = gen.reset_index().melt(id_vars="PSY", var_name="SUD", value_name="genetic_corr")

    df = sim.merge(gen, on=["PSY", "SUD"], how="inner")
    df["Cluster"] = df["PSY"].map(C.CLUSTERS)
    lost_psy = sorted(set(sim["PSY"]) - set(gen["PSY"]))
    lost_sud = sorted(set(sim["SUD"]) - set(gen["SUD"]))
    print(f"disorders in the brain maps with no genetic data: {lost_psy or 'none'}")
    print(f"substances in the brain maps with no genetic data: {lost_sud or 'none'}")
    df = df.dropna(subset=["Cluster", "similarity", "genetic_corr"])
    print(f"{len(df)} PSY-SUD pairs with both measures\n")
    return df


def bipartite_perm(sub, n_perm=N_PERM, seed=SEED):
    """
    Permutation p within one cluster, with its attainable floor.
    Rows (disorders) and columns (substances) are permuted independently, the
    same procedure used for the whole PSY-SUD grid in step 2.
    """
    A = sub.pivot(index="PSY", columns="SUD", values="similarity")
    B = sub.pivot(index="PSY", columns="SUD", values="genetic_corr").loc[A.index, A.columns]
    A, B = A.values.astype(float), B.values.astype(float)
    if np.isnan(A).any() or np.isnan(B).any():
        return np.nan, np.nan, np.nan
    nr, nc = A.shape
    obs = pearsonr(A.ravel(), B.ravel()).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = pearsonr(A[rng.permutation(nr)][:, rng.permutation(nc)].ravel(),
                           B.ravel()).statistic
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    floor = 1.0 / (__import__("math").factorial(nr) * __import__("math").factorial(nc) + 1)
    return obs, p, floor


def main():
    df = build_table()

    rows = []
    for cl in CLUSTER_ORDER:
        sub = df[df["Cluster"] == cl]
        members = sorted(set(sub["PSY"]))
        if len(sub) >= MIN_N:
            r, p_par = pearsonr(sub["similarity"], sub["genetic_corr"])
            r_perm, p_perm, floor = bipartite_perm(sub)
        else:
            r = p_par = r_perm = p_perm = floor = np.nan
        rows.append({"Cluster": cl, "n_pairs": len(sub), "n_disorders": len(members),
                     "disorders": ",".join(members), "r": r, "p_parametric": p_par,
                     "p_permutation": p_perm, "p_perm_floor": floor})
    stats = pd.DataFrame(rows)
    ok = stats["p_permutation"].notna()
    stats["pFDR_permutation"] = np.nan
    if ok.sum():
        stats.loc[ok, "pFDR_permutation"] = C.bh_fdr(stats.loc[ok, "p_permutation"].values)
    by_cl = stats.set_index("Cluster")

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    for ax, cl in zip(axes.flatten(), CLUSTER_ORDER):
        sub = df[df["Cluster"] == cl]
        color = CLUSTER_COLORS[cl]
        row = by_cl.loc[cl]
        ax.scatter(sub["similarity"], sub["genetic_corr"], s=70, alpha=0.75,
                   color=color, edgecolor="white", linewidth=0.5)
        if len(sub) >= MIN_N:
            m, b = np.polyfit(sub["similarity"], sub["genetic_corr"], 1)
            xs = np.linspace(sub["similarity"].min(), sub["similarity"].max(), 100)
            ax.plot(xs, m * xs + b, color="black", lw=1.5)
            txt = (f"r = {row['r']:+.2f}\n"
                   f"$p_{{perm}}$ = {row['p_permutation']:.3f} "
                   f"(floor {row['p_perm_floor']:.3f})\n"
                   f"n = {int(row['n_pairs'])} from {int(row['n_disorders'])} disorders")
        else:
            txt = (f"n = {int(row['n_pairs'])} from "
                   f"{int(row['n_disorders'])} disorder(s)\ntoo few to fit")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=FS_TEXT, fontweight="bold")
        ax.set_title(f"{cl} ({row['disorders']})", fontsize=FS_TITLE)
        ax.set_xlabel(f"Cortical similarity (Spearman \u03c1)", fontsize=FS_LABEL)
        ax.set_ylabel("Genetic correlation", fontsize=FS_LABEL)
        ax.grid(False)

    fig.suptitle("Supplementary Figure | Cluster-stratified similarity vs genetic correlation",
                 fontsize=FS_SUP, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "SuppFig_cluster_genetics_spearman.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")

    stats.round(4).to_csv(os.path.join(OUT_DIR, "SuppFig_cluster_genetics_stats.csv"),
                          index=False)
    pd.set_option("display.width", 200)
    print(stats.round(4).to_string(index=False))
    print(f"\nSaved {out}")
    print(f"Saved {os.path.join(OUT_DIR, 'SuppFig_cluster_genetics_stats.csv')}")


if __name__ == "__main__":
    main()