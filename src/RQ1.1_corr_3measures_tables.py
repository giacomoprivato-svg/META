#!/usr/bin/env python3
"""
RQ1 – Concordance between similarity metrics
Computed separately for Adults and Adolescents
Cortex only – z-scored similarity measures

Each table cell contains:
- Spearman rho
- 95% bootstrap confidence interval
- permutation p-value (10,000 permutations)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# -----------------------------
# PARAMETERS
# -----------------------------
N_BOOT = 10000
N_PERM = 10000
CI = 95
RNG = np.random.default_rng(42)

MEASURES = ["euclidean", "spearman", "cosine"]
PAIRS = [
    ("euclidean", "cosine"),
    ("euclidean", "spearman"),
    ("cosine", "spearman"),
]

# -----------------------------
# PATHS
# -----------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(REPO_DIR, "ALL_outputs_RQ1")
OUT_DIR = os.path.join(REPO_DIR, "RQ1_metric_concordance_tables")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = {
    "Adults": ["adults_all", "adults_ctx"],
    "Adolescents": ["adolescents_all", "adolescents_ctx"],
}

# -----------------------------
# STAT FUNCTIONS
# -----------------------------
def bootstrap_ci(x, y, n_boot=N_BOOT, ci=CI):
    r_obs = spearmanr(x, y, nan_policy="omit")[0]
    n = len(x)
    boot = np.empty(n_boot)

    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        boot[i] = spearmanr(x[idx], y[idx], nan_policy="omit")[0]

    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)

    return r_obs, lo, hi


def permutation_pval(x, y, n_perm=N_PERM):
    r_obs = spearmanr(x, y, nan_policy="omit")[0]
    perm = np.empty(n_perm)

    for i in range(n_perm):
        y_perm = RNG.permutation(y)
        perm[i] = spearmanr(x, y_perm, nan_policy="omit")[0]

    p = (np.sum(np.abs(perm) >= abs(r_obs)) + 1) / (n_perm + 1)
    return p


def format_cell(r, lo, hi, p):
    return f"{r:.2f} [{lo:.2f}–{hi:.2f}], p={p:.3g}"

# -----------------------------
# LOAD MATRICES
# -----------------------------
def load_group_matrices(folders):
    Z = {}
    psy_names, sud_names = None, None

    for m in MEASURES:
        mats = []
        for f in folders:
            path = os.path.join(DATA_DIR, f, f"Z_cortex_{m}.csv")
            df = pd.read_csv(path, index_col=0)

            if psy_names is None:
                psy_names = df.index.astype(str)
                sud_names = df.columns.astype(str)

            mats.append(df.values)

        Z[m] = np.vstack(mats)

    return Z, psy_names, sud_names

# -----------------------------
# MAIN
# -----------------------------
for label, folders in GROUPS.items():

    print(f"\nProcessing {label}")
    Z, psy_names, sud_names = load_group_matrices(folders)

    rows = []
    index = []

    # ---------- SUD ----------
    for j, sud in enumerate(sud_names):
        row = {}
        for m1, m2 in PAIRS:
            x = Z[m1][:, j]
            y = Z[m2][:, j]

            r, lo, hi = bootstrap_ci(x, y)
            p = permutation_pval(x, y)

            row[f"{m1}_{m2}"] = format_cell(r, lo, hi, p)

        rows.append(row)
        index.append(sud)

    # ---------- PSY ----------
    for i, psy in enumerate(psy_names):
        row = {}
        for m1, m2 in PAIRS:
            x = Z[m1][i, :]
            y = Z[m2][i, :]

            r, lo, hi = bootstrap_ci(x, y)
            p = permutation_pval(x, y)

            row[f"{m1}_{m2}"] = format_cell(r, lo, hi, p)

        rows.append(row)
        index.append(psy)

    df_out = pd.DataFrame(rows, index=index)
    df_out = df_out[
        ["euclidean_cosine", "euclidean_spearman", "cosine_spearman"]
    ]

    out_path = os.path.join(
        OUT_DIR, f"MetricConcordance_{label}_cortex.csv"
    )
    df_out.to_csv(out_path)

    print(f"Saved: {out_path}")

print("\n✅ All tables generated successfully")
