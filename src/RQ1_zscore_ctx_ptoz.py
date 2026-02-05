#!/usr/bin/env python3
"""
RQ1 similarity pipeline — CORTEX ONLY + p-value output

CORTEX:
- Similarity measures: Spearman, Cosine, (−)Euclidean
- Spatial permutation via spins (68 parcels)
- Signed z-score and associated p-value saved per PSY–SUD pair
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, norm
from tqdm import tqdm

# ---------------------------
# USER OPTIONS
# ---------------------------
N_PERM = 10000
N_CORTEX = 68
SAVE_NULLS = True

np.random.seed(42)

# ---------------------------
# Paths
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
main_outdir = os.path.join(repo_dir, "ALL_outputs_RQ1")
os.makedirs(main_outdir, exist_ok=True)

GROUPS = [
    ("adults_all", "PSY_adults.xlsx"),
    ("adolescents_all", "PSY_adolescents.xlsx"),
    ("adults_ctx", "PSY_adults_ctx.xlsx"),
    ("adolescents_ctx", "PSY_adolescents_ctx.xlsx"),
]

MEASURES = ["spearman", "cosine", "euclidean"]

# ---------------------------
# Helpers
# ---------------------------
def rand_rotation_matrix():
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q3*q4), 2*(q1*q3 + q2*q4)],
        [2*(q1*q2 + q3*q4), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q1*q4)],
        [2*(q1*q3 - q2*q4), 2*(q2*q3 + q1*q4), 1 - 2*(q1**2 + q2**2)]
    ])

def local_nn(A, B):
    return np.argmin(cdist(B, A), axis=1)

def spearman_rankcorr_obs(X, Y):
    rx = rankdata(X, method="average")
    ry = rankdata(Y, method="average")
    mx, my = rx.mean(), ry.mean()
    num = np.sum((rx - mx) * (ry - my))
    den = np.sqrt(np.sum((rx - mx)**2) * np.sum((ry - my)**2))
    return 0.0 if den == 0 else float(num / den)

def read_excel_numeric_matrix(xlsx_path):
    T = pd.read_excel(xlsx_path)
    num_cols = T.select_dtypes(include=[np.number]).columns.tolist()
    X = T[num_cols].to_numpy()
    nonnum_cols = T.select_dtypes(exclude=[np.number]).columns.tolist()
    if nonnum_cols:
        region_names = T[nonnum_cols[0]].astype(str).to_numpy()
    else:
        region_names = np.array([f"R{i+1}" for i in range(T.shape[0])], dtype=str)
    return X, num_cols, region_names

def vectorized_similarity(X, Y, perms):
    nperm = perms.shape[0]
    obs = np.zeros(3)
    nulls = np.zeros((nperm, 3))

    obs[0] = spearman_rankcorr_obs(X, Y)
    nx, ny = np.linalg.norm(X), np.linalg.norm(Y)
    obs[1] = 0.0 if nx == 0 or ny == 0 else float(np.dot(X, Y) / (nx * ny))
    obs[2] = -float(np.linalg.norm(X - Y))

    Xp = X[perms]

    rankY = rankdata(Y, method="average")
    meanY = rankY.mean()
    stdY = np.sqrt(np.sum((rankY - meanY) ** 2))
    rankXp = np.apply_along_axis(rankdata, 1, Xp)
    meanX = rankXp.mean(axis=1, keepdims=True)
    cov = np.sum((rankXp - meanX) * (rankY - meanY), axis=1)
    stdX = np.sqrt(np.sum((rankXp - meanX) ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        nulls[:, 0] = np.where((stdX == 0) | (stdY == 0), 0.0, cov / (stdX * stdY))

    Xn = np.linalg.norm(Xp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nulls[:, 1] = np.where((Xn == 0) | (ny == 0), 0.0, np.sum(Xp * Y, axis=1) / (Xn * ny))

    nulls[:, 2] = -np.linalg.norm(Xp - Y, axis=1)

    return obs, nulls

def signed_z_and_p_from_null(obs, null):
    null = np.asarray(null).ravel()
    n = null.size

    p_upper = (np.sum(null >= obs) + 1) / (n + 1)
    p_lower = (np.sum(null <= obs) + 1) / (n + 1)

    if p_upper <= p_lower:
        p = p_upper
        sign = 1.0
    else:
        p = p_lower
        sign = -1.0

    z = sign * norm.isf(p)
    return z, p  # <-- ritorna anche il p-value usato

# ---------------------------
# Load SUD
# ---------------------------
sud_file = os.path.join(data_dir, "SUD.xlsx")
X_sud, sud_names, _ = read_excel_numeric_matrix(sud_file)

# ---------------------------
# Main
# ---------------------------
for group_label, psy_file_name in GROUPS:
    print(f"\n== Running group: {group_label} ==")

    psy_file = os.path.join(data_dir, psy_file_name)
    outdir = os.path.join(main_outdir, group_label)
    os.makedirs(outdir, exist_ok=True)

    X_psy, psy_names, _ = read_excel_numeric_matrix(psy_file)

    # --------------------------------------------------
    # ENFORCE CORTEX-ONLY (discard subcortex if present)
    # --------------------------------------------------
    if X_psy.shape[0] < N_CORTEX or X_sud.shape[0] < N_CORTEX:
        raise ValueError("Input files must contain at least 68 cortical rows")

    X_psy = X_psy[:N_CORTEX, :]
    X_sud = X_sud[:N_CORTEX, :]

    cortex_idx = np.arange(N_CORTEX)

    # ---------------------------
    # Cortex spins
    # ---------------------------
    spinfile = os.path.join(outdir, "spins_ctx_68.mat")
    if os.path.exists(spinfile):
        spins_ctx = sio.loadmat(spinfile)["spins_ctx"] - 1
    else:
        centsfile = os.path.join(data_dir, "centroids_ctx_68.mat")
        with h5py.File(centsfile, "r") as f:
            LH = np.array(f["centroids_lh"]).T
            RH = np.array(f["centroids_rh"]).T

        LH /= np.linalg.norm(LH, axis=1, keepdims=True)
        RH /= np.linalg.norm(RH, axis=1, keepdims=True)

        spins = np.zeros((N_PERM, N_CORTEX), dtype=int)
        for k in range(N_PERM):
            R = rand_rotation_matrix()
            idxL = local_nn(LH, (R @ LH.T).T)
            idxR = local_nn(RH, (R @ RH.T).T)
            spins[k, :] = np.concatenate([idxL, idxR])

        spins_ctx = spins
        sio.savemat(spinfile, {"spins_ctx": spins_ctx + 1})

    num_psy = X_psy.shape[1]
    num_sud = X_sud.shape[1]

    RAW = {m: np.zeros((num_psy, num_sud)) for m in MEASURES}
    Z = {m: np.zeros((num_psy, num_sud)) for m in MEASURES}
    PVAL = {m: np.zeros((num_psy, num_sud)) for m in MEASURES}  # <-- matrice p-value
    NULLS = {m: np.zeros((num_psy, num_sud, N_PERM)) for m in MEASURES} if SAVE_NULLS else None

    print("Computing cortex similarity...")
    for i in tqdm(range(num_psy), desc="PSY maps"):
        for j in range(num_sud):
            obs, nulls = vectorized_similarity(
                X_psy[:, i],
                X_sud[:, j],
                spins_ctx
            )
            for k, m in enumerate(MEASURES):
                RAW[m][i, j] = obs[k]
                z, p = signed_z_and_p_from_null(obs[k], nulls[:, k])
                Z[m][i, j] = z
                PVAL[m][i, j] = p  # <-- salva il p-value usato per z
                if SAVE_NULLS:
                    NULLS[m][i, j, :] = nulls[:, k]

    # ---------------------------
    # Save outputs
    # ---------------------------
    for m in MEASURES:
        pd.DataFrame(RAW[m], index=psy_names, columns=sud_names)\
          .to_csv(os.path.join(outdir, f"RAW_cortex_{m}.csv"))
        pd.DataFrame(Z[m], index=psy_names, columns=sud_names)\
          .to_csv(os.path.join(outdir, f"Z_cortex_{m}.csv"))
        pd.DataFrame(PVAL[m], index=psy_names, columns=sud_names)\
          .to_csv(os.path.join(outdir, f"PVAL_cortex_{m}.csv"))  # <-- salvataggio p-value

        if SAVE_NULLS:
            for j, sname in enumerate(sud_names):
                pd.DataFrame(NULLS[m][:, j, :], index=psy_names)\
                  .to_csv(os.path.join(outdir, f"NULLS_cortex_{m}_{sname}.csv"))

    # ---------------------------
    # Ranking
    # ---------------------------
    for m in MEASURES:
        Zmat = Z[m]
        rank_idx = np.argsort(-Zmat, axis=0)

        for j, sname in enumerate(sud_names):
            pd.DataFrame({
                "Psychiatric_Disorder": np.array(psy_names)[rank_idx[:, j]],
                f"Z_{m}": Zmat[rank_idx[:, j], j]
            }).to_csv(os.path.join(outdir, f"RANK_{m}_by_{sname}.csv"), index=False)

        meanZ = np.nanmean(Zmat, axis=1)
        mean_idx = np.argsort(-meanZ)
        pd.DataFrame({
            "Psychiatric_Disorder": np.array(psy_names)[mean_idx],
            f"Mean_Z_{m}_over_SUD": meanZ[mean_idx]
        }).to_csv(os.path.join(outdir, f"RANK_{m}_mean_across_SUD.csv"), index=False)

print("\n✅ Cortex-only analysis with p-values completed successfully.")
