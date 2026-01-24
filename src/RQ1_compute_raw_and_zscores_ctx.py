#!/usr/bin/env python3
"""
RQ1 — Cortex-only similarity computation
Compatible with existing heatmap scripts
(68 cortical regions, ENIGMA-style spins)
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

# ---------------------------
# USER OPTIONS
# ---------------------------
N_PERM = 10000
N_CORTEX = 68
SAVE_NULLS = True
np.random.seed(42)

GROUPS = [
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
out_base = os.path.join(repo_dir, "ALL_outputs_RQ1")

# ---------------------------
# Helpers — Excel readers
# ---------------------------
def read_excel_ctx(xlsx_path, n_rows):
    """
    CTX Excel format:
    - first row: header
    - next n_rows rows: numeric values only
    - no region-label column
    """
    df = pd.read_excel(xlsx_path)

    X = df.iloc[:n_rows, :].to_numpy(dtype=float)
    col_names = df.columns.astype(str).tolist()
    region_names = [f"ROI_{i+1}" for i in range(n_rows)]

    return X, col_names, region_names


def read_excel_sud(xlsx_path, start_row, n_rows):
    """
    SUD Excel format:
    - first row: header
    - many rows; use only rows [start_row : start_row + n_rows]
    """
    df = pd.read_excel(xlsx_path)

    df_sel = df.iloc[start_row:start_row + n_rows, :]

    X = df_sel.to_numpy(dtype=float)
    col_names = df.columns.astype(str).tolist()
    region_names = [f"ROI_{i+1}" for i in range(n_rows)]

    return X, col_names, region_names


def rand_rotation_matrix():
    """Quaternion-based random rotation matrix."""
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1-u1) * np.sin(2*np.pi*u2)
    q2 = np.sqrt(1-u1) * np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1) * np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1) * np.cos(2*np.pi*u3)
    return np.array([
        [1-2*(q2*q2+q3*q3), 2*(q1*q2-q3*q4), 2*(q1*q3+q2*q4)],
        [2*(q1*q2+q3*q4), 1-2*(q1*q1+q3*q3), 2*(q2*q3-q1*q4)],
        [2*(q1*q3-q2*q4), 2*(q2*q3+q1*q4), 1-2*(q1*q1+q2*q2)]
    ])


def local_nn(A, B):
    """Nearest-neighbour mapping."""
    return np.argmin(cdist(B, A), axis=1)


def spearman_obs(x, y):
    return np.corrcoef(rankdata(x), rankdata(y))[0, 1]


def vectorized_similarity(X, Y, perms):
    """
    X, Y: (68,)
    perms: (N_PERM, 68)
    """
    nperm = perms.shape[0]
    obs = np.zeros(3)
    nulls = np.zeros((nperm, 3))

    # Observed
    obs[0] = spearman_obs(X, Y)
    obs[1] = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
    obs[2] = -np.linalg.norm(X - Y)

    # Permuted X
    Xp = X[perms]
    Ynorm = np.linalg.norm(Y)
    ry = rankdata(Y)

    for i in range(nperm):
        nulls[i, 0] = np.corrcoef(rankdata(Xp[i]), ry)[0, 1]

    nulls[:, 1] = np.sum(Xp * Y, axis=1) / (np.linalg.norm(Xp, axis=1) * Ynorm)
    nulls[:, 2] = -np.linalg.norm(Xp - Y, axis=1)

    return obs, nulls


# ---------------------------
# Load SUD (cortex only)
# ---------------------------
X_sud, sud_names, _ = read_excel_sud(
    os.path.join(data_dir, "SUD.xlsx"),
    start_row=1,          # Excel row 2
    n_rows=N_CORTEX
)

# ---------------------------
# Main
# ---------------------------
for label, psy_file in GROUPS:

    print(f"\n== Cortex-only computation: {label} ==")

    outdir = os.path.join(out_base, label)
    os.makedirs(outdir, exist_ok=True)

    X_psy, psy_names, _ = read_excel_ctx(
        os.path.join(data_dir, psy_file),
        n_rows=N_CORTEX
    )

    assert X_psy.shape[0] == X_sud.shape[0] == N_CORTEX

    # ---------------------------
    # Cortex spins (68)
    # ---------------------------
    spinfile = os.path.join(outdir, "spins_ctx_68.mat")

    if os.path.exists(spinfile):
        spins_ctx = sio.loadmat(spinfile)["spins_ctx"] - 1
    else:
        centsfile = os.path.join(data_dir, "centroids_ctx_68.mat")
        if not os.path.exists(centsfile):
            raise FileNotFoundError("Missing centroids_ctx_68.mat")

        with h5py.File(centsfile, "r") as f:
            LH = np.array(f["centroids_lh"]).T
            RH = np.array(f["centroids_rh"]).T

        LH = LH / np.linalg.norm(LH, axis=1, keepdims=True)
        RH = RH / np.linalg.norm(RH, axis=1, keepdims=True)

        spins_ctx = np.zeros((N_PERM, N_CORTEX), dtype=int)

        for k in range(N_PERM):
            R = rand_rotation_matrix()
            idxL = local_nn(LH, (R @ LH.T).T)
            idxR = local_nn(RH, (R @ RH.T).T)
            spins_ctx[k, :] = np.concatenate([idxL, idxR])

        sio.savemat(spinfile, {"spins_ctx": spins_ctx + 1})

    # ---------------------------
    # Containers
    # ---------------------------
    n_psy = X_psy.shape[1]
    n_sud = X_sud.shape[1]

    RAW = {m: np.zeros((n_psy, n_sud)) for m in MEASURES}
    Z = {m: np.zeros((n_psy, n_sud)) for m in MEASURES}
    NULLS = {
        m: np.zeros((n_psy, n_sud, N_PERM))
        for m in MEASURES
    }

    # ---------------------------
    # Compute similarities
    # ---------------------------
    for i in tqdm(range(n_psy), desc="PSY maps"):
        for j in range(n_sud):
            obs, nulls = vectorized_similarity(
                X_psy[:, i],
                X_sud[:, j],
                spins_ctx
            )

            for k, m in enumerate(MEASURES):
                RAW[m][i, j] = obs[k]
                Z[m][i, j] = (obs[k] - nulls[:, k].mean()) / nulls[:, k].std()
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

        if SAVE_NULLS:
            for j, s in enumerate(sud_names):
                pd.DataFrame(NULLS[m][:, j, :], index=psy_names)\
                    .to_csv(os.path.join(outdir, f"NULLS_cortex_{m}_{s}.csv"))

print("\n✅ Cortex-only RAW, Z, and NULLS computed successfully.")


