#!/usr/bin/env python3
"""
RQ1 similarity pipeline — SUBCORTEX ONLY
RAW values only (no permutations, no p-values, no z-scores)
Measures:
    - Spearman correlation
    - Cosine similarity
    - Negative Euclidean distance
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from tqdm import tqdm

# ---------------------------
# USER OPTIONS
# ---------------------------
N_CORTEX = 68   # subcortex starts after 68 regions
np.random.seed(42)

GROUPS = [
    ("adults_all", "PSY_adults.xlsx"),
    ("adolescents_all", "PSY_adolescents.xlsx"),
    ("adults_ctx", "PSY_adults_ctx.xlsx"),
    ("adolescents_ctx", "PSY_adolescents_ctx.xlsx"),
]

MEASURES = ["spearman", "cosine", "euclidean"]

# ---------------------------
# Paths
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
main_outdir = os.path.join(repo_dir, "ALL_outputs_RQ1")
os.makedirs(main_outdir, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def spearman_rankcorr_obs(X, Y):
    rx = rankdata(X, method="average")
    ry = rankdata(Y, method="average")
    mx, my = rx.mean(), ry.mean()
    num = np.sum((rx-mx)*(ry-my))
    den = np.sqrt(np.sum((rx-mx)**2)*np.sum((ry-my)**2))
    return 0.0 if den == 0 else float(num/den)

def read_excel_numeric_matrix(xlsx_path):
    df = pd.read_excel(xlsx_path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_cols].to_numpy()
    nonnum_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if nonnum_cols:
        region_names = df[nonnum_cols[0]].astype(str).to_numpy()
    else:
        region_names = np.array([f"R{i+1}" for i in range(df.shape[0])], dtype=str)
    return X, num_cols, region_names

# ---------------------------
# Load SUD
# ---------------------------
sud_file = os.path.join(data_dir, "SUD.xlsx")
if not os.path.exists(sud_file):
    raise FileNotFoundError("SUD.xlsx not found")

X_sud, sud_names, _ = read_excel_numeric_matrix(sud_file)

# ---------------------------
# Main loop
# ---------------------------
for group_label, psy_file_name in GROUPS:

    print(f"\n== Running group: {group_label} ==")

    psy_file = os.path.join(data_dir, psy_file_name)
    if not os.path.exists(psy_file):
        raise FileNotFoundError(f"{psy_file_name} not found")

    outdir = os.path.join(main_outdir, group_label)
    os.makedirs(outdir, exist_ok=True)

    # Load PSY
    X_psy, psy_names, _ = read_excel_numeric_matrix(psy_file)

    # ---------- Subcortex handling ----------
    subctx_idx = np.arange(N_CORTEX, min(X_psy.shape[0], X_sud.shape[0]))
    n_sub = len(subctx_idx)

    if n_sub == 0:
        print("No subcortical regions found — skipping.")
        continue

    X_psy_sub = X_psy[subctx_idx, :]
    X_sud_sub = X_sud[subctx_idx, :]

    n_psy = X_psy_sub.shape[1]
    n_sud = X_sud_sub.shape[1]

    # ---------- Containers ----------
    RAW = {m: np.zeros((n_psy, n_sud)) for m in MEASURES}

    # ---------- Compute similarity ----------
    print("Computing subcortex RAW similarity...")

    for i in tqdm(range(n_psy), desc="PSY maps"):
        psy_vec = X_psy_sub[:, i]

        for j in range(n_sud):
            sud_vec = X_sud_sub[:, j]

            # Spearman
            RAW["spearman"][i, j] = spearman_rankcorr_obs(psy_vec, sud_vec)

            # Cosine
            nx = np.linalg.norm(psy_vec)
            ny = np.linalg.norm(sud_vec)
            if nx == 0 or ny == 0:
                RAW["cosine"][i, j] = 0.0
            else:
                RAW["cosine"][i, j] = np.dot(psy_vec, sud_vec) / (nx * ny)

            # Negative Euclidean distance
            RAW["euclidean"][i, j] = -np.linalg.norm(psy_vec - sud_vec)

    # ---------- Save CSV ----------
    for m in MEASURES:
        pd.DataFrame(
            RAW[m],
            index=psy_names,
            columns=sud_names
        ).to_csv(os.path.join(outdir, f"RAW_subctx_{m}.csv"))

    # ---------- Ranking ----------
    for m in MEASURES:
        rank_idx = np.argsort(-RAW[m], axis=0)

        for j, sname in enumerate(sud_names):
            ord_psy = np.array(psy_names)[rank_idx[:, j]]
            ord_val = RAW[m][rank_idx[:, j], j]

            pd.DataFrame({
                "Psychiatric_Disorder": ord_psy,
                f"{m}_value": ord_val
            }).to_csv(
                os.path.join(outdir, f"RANK_{m}_by_{sname}.csv"),
                index=False
            )

        # Mean across SUD
        mean_val = np.nanmean(RAW[m], axis=1)
        mean_idx = np.argsort(-mean_val)

        pd.DataFrame({
            "Psychiatric_Disorder": np.array(psy_names)[mean_idx],
            f"Mean_{m}_over_SUD": mean_val[mean_idx]
        }).to_csv(
            os.path.join(outdir, f"RANK_{m}_mean_across_SUD.csv"),
            index=False
        )

print("\n✅ Subcortex-only RAW similarity analysis completed.")
