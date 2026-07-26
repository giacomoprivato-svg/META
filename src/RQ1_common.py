#!/usr/bin/env python3
"""
RQ1 — shared helpers for the corrected similarity pipeline
==========================================================

Single source of truth for both null frameworks, so the spin and BrainSMASH
pipelines cannot drift apart.

WHAT CHANGED vs the previous scripts
------------------------------------
1. PRIMARY METRIC = Pearson correlation between the two 68-region maps.
   Same statistic already used throughout RQ2, so RQ1/RQ2/RQ3 now share one
   similarity measure. Spearman / cosine / (-)Euclidean are retained as
   sensitivity metrics.

   Why not Euclidean as primary: raw Euclidean similarity is dominated by
   effect-size magnitude (||X|| ranges 0.34-4.66 across disorders, i.e. a
   14x spread driven partly by sample size, severity and chronicity rather
   than biology). A map of pure zeros ranks 8th of 9 disorders on raw
   Euclidean similarity to SUD, ahead of ADHD and AN, because for a weak map
   ||X - Y|| ~ ||Y||, a constant carrying no information about X.

2. SPIN BUG FIXED. Previously:
       spins[k] = concatenate([idxL, idxR])       # idxR in 0..33
   idxR indexes the 34-element RH centroid array, but the data map is
   ordered [LH 0..33, RH 34..67]. Without the +34 offset, every surrogate
   filled all 34 right-hemisphere positions with LEFT hemisphere values
   (100% of RH positions), leaving only ~30 of 68 unique source regions per
   surrogate. Now:
       spins[k] = concatenate([idxL, idxR + 34])

   NOTE: spins are cached to disk. The cache filename is versioned below so
   the corrected spins cannot collide with previously saved buggy ones.

3. Z IS NOW AN EFFECT SIZE, not a reparametrised p-value. Previously:
       z = sign * norm.isf(p_one)
   which is a deterministic function of p (z is recoverable from p to 4e-9),
   so the heatmap colour and the asterisks encoded the same quantity, and z
   saturated at norm.isf(1/10001) = 3.719 (7-21 of 63 pairs tied at the
   ceiling). Now:
       z = (obs - mean(null)) / sd(null)
   which matches what the RQ2 scripts already do.

   Ranking should use RAW Pearson r (a common scale across all pairs);
   z and p answer the separate question of per-pair significance.

4. The two-tailed p-value is UNCHANGED: p = min(2 * min(tail), 1) with +1
   smoothing. That part was correct.

NULL HYPOTHESIS (state this in the Methods)
-------------------------------------------
For each PSY-SUD pair the psychiatric map is spatially randomised while the
SUD map is held fixed. Both nulls preserve the psychiatric map's effect-size
distribution exactly and reproduce its spatial autocorrelation; only its
alignment with the SUD map is randomised. H0 is therefore that psychiatric
and SUD alterations are no more spatially co-localised than expected for maps
of the same magnitude and smoothness. Significance reflects spatial
correspondence and is by construction independent of effect-size magnitude.
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

N_CORTEX = 68
N_PER_HEMI = 34
SPIN_CACHE_NAME = "spins_ctx_68_fixed.mat"   # versioned: never collides with the old buggy cache

# pearson first = primary; the rest are sensitivity metrics
MEASURES = ["pearson", "spearman", "cosine", "euclidean"]


# ---------------------------------------------------------------
# I/O
# ---------------------------------------------------------------
def read_excel_numeric_matrix(xlsx_path):
    """Return (numeric matrix, column names, row labels)."""
    T = pd.read_excel(xlsx_path)
    num = T.select_dtypes(include=[np.number])
    non_num = T.select_dtypes(exclude=[np.number])
    if non_num.shape[1] > 0:
        row_labels = non_num.iloc[:, 0].astype(str).tolist()
    else:
        row_labels = [f"R{i+1}" for i in range(len(T))]
    return num.to_numpy(dtype=float), list(num.columns), row_labels


# ---------------------------------------------------------------
# Spins
# ---------------------------------------------------------------
def rand_rotation_matrix():
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q3*q4),     2*(q1*q3 + q2*q4)],
        [2*(q1*q2 + q3*q4),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q1*q4)],
        [2*(q1*q3 - q2*q4),     2*(q2*q3 + q1*q4),     1 - 2*(q1**2 + q2**2)]
    ])


def local_nn(A, B):
    """For each row of B, index of nearest row in A."""
    return np.argmin(cdist(B, A), axis=1)


def load_centroids(data_dir):
    with h5py.File(os.path.join(data_dir, "centroids_ctx_68.mat"), "r") as f:
        LH = np.array(f["centroids_lh"]).T
        RH = np.array(f["centroids_rh"]).T
    LH = LH / np.linalg.norm(LH, axis=1, keepdims=True)
    RH = RH / np.linalg.norm(RH, axis=1, keepdims=True)
    return LH, RH


def make_spins(data_dir, n_perm, cache_path=None, seed=42):
    """Corrected spin indices, shape (n_perm, 68), values 0..67."""
    if cache_path is not None and os.path.exists(cache_path):
        import scipy.io as sio
        spins = sio.loadmat(cache_path)["spins_ctx"] - 1
        _assert_spins_valid(spins)
        return spins

    rng_state = np.random.get_state()
    np.random.seed(seed)
    LH, RH = load_centroids(data_dir)
    spins = np.zeros((n_perm, N_CORTEX), dtype=int)
    for k in range(n_perm):
        R = rand_rotation_matrix()
        idxL = local_nn(LH, (R @ LH.T).T)              # 0..33  -> LH positions
        idxR = local_nn(RH, (R @ RH.T).T) + N_PER_HEMI  # 34..67 -> RH positions  <-- THE FIX
        spins[k, :] = np.concatenate([idxL, idxR])
    np.random.set_state(rng_state)

    _assert_spins_valid(spins)
    if cache_path is not None:
        import scipy.io as sio
        sio.savemat(cache_path, {"spins_ctx": spins + 1})
    return spins


def _assert_spins_valid(spins):
    """Guard against the hemisphere bug ever silently reappearing."""
    lh_block = spins[:, :N_PER_HEMI]
    rh_block = spins[:, N_PER_HEMI:]
    if lh_block.max() >= N_PER_HEMI:
        raise AssertionError("LH positions draw right-hemisphere values.")
    if rh_block.min() < N_PER_HEMI:
        raise AssertionError(
            "RH positions draw LEFT-hemisphere values: the +34 offset is missing. "
            "If you are loading a cached .mat, delete it and regenerate."
        )


# ---------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------
def _pearson_rows(Xp, y):
    Xc = Xp - Xp.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    denom = np.linalg.norm(Xc, axis=1) * np.linalg.norm(yc)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, 0.0, (Xc @ yc) / denom)


def similarity_and_nulls(X, Y, perms_or_surrogates, is_index=True):
    """
    Observed similarity and null distributions for all four metrics.

    is_index=True  -> perms_or_surrogates is an integer index array (spins)
    is_index=False -> it is an array of surrogate maps (BrainSMASH)

    All metrics are computed from the SAME permuted maps, so null index k is
    aligned across metrics.
    Returns obs (4,) and nulls (n_perm, 4), ordered as MEASURES.
    """
    Xp = X[perms_or_surrogates] if is_index else np.asarray(perms_or_surrogates)
    nperm = Xp.shape[0]
    obs = np.zeros(4)
    nulls = np.zeros((nperm, 4))

    # --- pearson (primary) ---
    Xc, Yc = X - X.mean(), Y - Y.mean()
    dn = np.linalg.norm(Xc) * np.linalg.norm(Yc)
    obs[0] = 0.0 if dn == 0 else float(np.dot(Xc, Yc) / dn)
    nulls[:, 0] = _pearson_rows(Xp, Y)

    # --- spearman ---
    obs[1] = float(np.corrcoef(rankdata(X), rankdata(Y))[0, 1])
    rankXp = np.apply_along_axis(rankdata, 1, Xp)
    nulls[:, 1] = _pearson_rows(rankXp, rankdata(Y))

    # --- cosine ---
    nx, ny = np.linalg.norm(X), np.linalg.norm(Y)
    obs[2] = 0.0 if nx == 0 or ny == 0 else float(np.dot(X, Y) / (nx * ny))
    Xn = np.linalg.norm(Xp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nulls[:, 2] = np.where((Xn == 0) | (ny == 0), 0.0, (Xp @ Y) / (Xn * ny))

    # --- (-)euclidean ---
    obs[3] = -float(np.linalg.norm(X - Y))
    nulls[:, 3] = -np.linalg.norm(Xp - Y, axis=1)

    return obs, nulls


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------
def z_and_p_from_null(obs, null):
    """
    Two-tailed permutation p (UNCHANGED from the original pipeline) and a
    standardized effect relative to the null (CHANGED: no longer norm.isf(p)).

    z > 0 : more similar than expected given the map's own magnitude and
            spatial autocorrelation.
    """
    null = np.asarray(null).ravel()
    null = null[np.isfinite(null)]
    n = null.size

    p_upper = (np.sum(null >= obs) + 1) / (n + 1)
    p_lower = (np.sum(null <= obs) + 1) / (n + 1)
    p = min(min(p_upper, p_lower) * 2.0, 1.0)

    sd = null.std(ddof=1)
    z = 0.0 if sd == 0 else float((obs - null.mean()) / sd)
    return z, p