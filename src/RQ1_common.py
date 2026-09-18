#!/usr/bin/env python3
"""
RQ1_common — the ONLY place RQ1 statistics are defined

"""

import os
import hashlib
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

N_CORTEX = 68
N_PER_HEMI = 34
SEED = 42

# MEASURES[0] is the primary metric. Do not reorder without checking every
# downstream `obs[0]` / `MEASURES[0]`.
MEASURES = ["spearman", "cosine", "euclidean"]
PRIMARY = MEASURES[0]


# =================================================================
# I/O
# =================================================================
def read_maps(xlsx_path, n_rows=None, drop_cols=()):
    """
    Numeric columns of an ENIGMA map workbook.

    n_rows=N_CORTEX  -> the 68 cortical rows
    n_rows=None      -> everything (82 = 68 cortical + 14 subcortical)

    Returns (DataFrame, column names).
    """
    T = pd.read_excel(xlsx_path).select_dtypes(include=[np.number])
    drop = {str(d).strip().upper() for d in drop_cols}
    T = T.drop(columns=[c for c in T.columns if str(c).strip().upper() in drop])
    if n_rows is not None:
        if T.shape[0] < n_rows:
            raise ValueError(f"{os.path.basename(xlsx_path)}: {T.shape[0]} rows, need {n_rows}")
        T = T.iloc[:n_rows, :]
    if not np.isfinite(T.to_numpy(float)).all():
        raise ValueError(f"{os.path.basename(xlsx_path)}: non-finite values")
    return T.reset_index(drop=True), list(T.columns)


def subcortical_rows(xlsx_path, drop_cols=()):
    """Rows 68.. of a map workbook (14 subcortical structures, ventricles already excluded)."""
    T = pd.read_excel(xlsx_path).select_dtypes(include=[np.number])
    drop = {str(d).strip().upper() for d in drop_cols}
    T = T.drop(columns=[c for c in T.columns if str(c).strip().upper() in drop])
    T = T.iloc[N_CORTEX:, :].reset_index(drop=True)
    return T, list(T.columns)


def map_hash(x):
    """SHA1 of a map's exact float64 bytes; the cache key for its surrogates."""
    return hashlib.sha1(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()[:12]


# =================================================================
# Geometry
# =================================================================
def load_centroids(data_dir, unit=False):
    """
    Cortical centroids, (34, 3) per hemisphere.

    unit=True  -> projected onto the unit sphere. ONLY for spins: a random
                  rotation is meaningful on a sphere, not on raw coordinates.
    unit=False -> raw coordinates. Use these for anything metric, i.e. the
                  BrainSMASH distance matrix.
    """
    with h5py.File(os.path.join(data_dir, "centroids_ctx_68.mat"), "r") as f:
        LH = np.array(f["centroids_lh"]).T
        RH = np.array(f["centroids_rh"]).T
    if unit:
        LH = LH / np.linalg.norm(LH, axis=1, keepdims=True)
        RH = RH / np.linalg.norm(RH, axis=1, keepdims=True)
    return LH, RH


def distance_matrix(data_dir):
    """(68, 68) Euclidean distances from RAW centroids, for BrainSMASH."""
    LH, RH = load_centroids(data_dir, unit=False)
    D = cdist(np.vstack([LH, RH]), np.vstack([LH, RH]))
    assert D.shape == (N_CORTEX, N_CORTEX)
    return D


# =================================================================
# Spin null
# =================================================================
def _rand_rotation(rng):
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q3*q4),     2*(q1*q3 + q2*q4)],
        [2*(q1*q2 + q3*q4),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q1*q4)],
        [2*(q1*q3 - q2*q4),     2*(q2*q3 + q1*q4),     1 - 2*(q1**2 + q2**2)],
    ])


def _assert_spins_valid(spins):
    """The hemisphere bug cannot come back silently."""
    if spins[:, :N_PER_HEMI].max() >= N_PER_HEMI:
        raise AssertionError("LH positions draw right-hemisphere values.")
    if spins[:, N_PER_HEMI:].min() < N_PER_HEMI:
        raise AssertionError(
            "RH positions draw LEFT-hemisphere values: the +34 offset is missing. "
            "This is the original bug. If a cached .mat produced this, delete it."
        )


def make_spins(data_dir, n_perm, cache_dir, seed=SEED):
    """
    (n_perm, 68) spin indices, generated once and shared by every group.

    The cache name carries n_perm AND the `_fixed` tag, so it can never
    collide with `spins_ctx_68.mat` from the buggy pipeline.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"spins_ctx_68_fixed_n{n_perm}_seed{seed}.npy")
    if os.path.exists(path):
        spins = np.load(path)
        _assert_spins_valid(spins)
        return spins

    LH, RH = load_centroids(data_dir, unit=True)      # unit sphere: spins only
    rng = np.random.default_rng(seed)
    spins = np.empty((n_perm, N_CORTEX), dtype=np.int16)
    for k in range(n_perm):
        R = _rand_rotation(rng)
        idxL = np.argmin(cdist((R @ LH.T).T, LH), axis=1)
        idxR = np.argmin(cdist((R @ RH.T).T, RH), axis=1) + N_PER_HEMI   # <-- THE FIX
        spins[k] = np.concatenate([idxL, idxR])
    _assert_spins_valid(spins)
    np.save(path, spins)
    return spins


# =================================================================
# BrainSMASH null
# =================================================================
def get_surrogates(x, D, n_perm, label, cache_dir, seed=SEED):
    """
    (n_perm, 68) SAC-preserving surrogates of map `x`, cached on disk.

    THE CACHE KEY IS THE MAP'S HASH ALONE — deliberately, and this was wrong in
    the first version. Naming the file surr_<label>_<hash>.npy meant the SAME
    map requested under two different labels was generated twice: RQ2 step 1
    asks for the adult PSY mean as "shared_Adults" and RQ2 step 2 asks for the
    identical vector as "rq2map_Adults_component". Identical bytes, identical
    surrogates, two BrainSMASH runs. Since BrainSMASH is the slowest thing in
    the pipeline, that alone doubled the wall time of the RQ2 pass.

    The label is now recorded in a sidecar index for human readability and has
    no effect on reuse. A map is generated once, ever, no matter who asks.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h = map_hash(x)
    path = os.path.join(cache_dir, f"surr_{h}_n{n_perm}.npy")

    idx = os.path.join(cache_dir, "surrogate_index.csv")
    seen = set()
    if os.path.exists(idx):
        with open(idx) as fh:
            seen = {ln.strip() for ln in fh}
    entry = f"{h},{n_perm},{label}"
    if entry not in seen:
        with open(idx, "a") as fh:
            fh.write(entry + "\n")

    if os.path.exists(path):
        surr = np.load(path)
        if surr.shape == (n_perm, len(x)):
            return surr.astype(float)
        os.remove(path)                                # wrong shape: rebuild

    from brainsmash.mapgen.base import Base
    gen = Base(x=np.asarray(x, float), D=D, resample=True, seed=seed)
    surr = gen(n=n_perm)
    tmp = path + ".part.npy"
    np.save(tmp, surr.astype(np.float32))
    os.replace(tmp, path)                              # atomic: no half files
    return surr.astype(float)


# =================================================================
# Similarity
# =================================================================
def _corr_rows_cols(A, B):
    """Pearson r between every ROW of A (n,p) and every COLUMN of B (p,k) -> (n,k)."""
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    na = np.linalg.norm(Ac, axis=1, keepdims=True)
    nb = np.linalg.norm(Bc, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((na == 0) | (nb == 0), 0.0, (Ac @ Bc) / (na * nb))


def observed_similarity(x, Y):
    """
    Observed similarity of one map x (p,) against every column of Y (p,k).
    Returns dict metric -> (k,).
    """
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)
    rx, rY = rankdata(x)[None, :], rankdata(Y, axis=0)
    ny = np.linalg.norm(Y, axis=0)
    nx = np.linalg.norm(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where((nx == 0) | (ny == 0), 0.0, (x @ Y) / (nx * ny))
    return {
        "spearman": _corr_rows_cols(rx, rY)[0],
        "cosine": cos,
        "euclidean": -np.linalg.norm(x[:, None] - Y, axis=0),
    }


def null_similarity(Xp, Y):
    """
    Null distributions for one map's surrogates Xp (n_perm, p) against every
    column of Y (p, k). Returns dict metric -> (n_perm, k).

    All k targets are done in one pass: the surrogates are ranked ONCE, not
    once per target. The old code called np.apply_along_axis(rankdata, 1, Xp)
    inside the loop over targets, i.e. it re-ranked a 10000 x 68 array seven
    times per psychiatric map for no reason.
    """
    Xp = np.asarray(Xp, float)
    Y = np.asarray(Y, float)
    rXp, rY = rankdata(Xp, axis=1), rankdata(Y, axis=0)
    nxp = np.linalg.norm(Xp, axis=1, keepdims=True)
    ny = np.linalg.norm(Y, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where((nxp == 0) | (ny == 0), 0.0, (Xp @ Y) / (nxp * ny))
    eu = -np.sqrt(np.maximum(
        (Xp**2).sum(1)[:, None] - 2 * (Xp @ Y) + (Y**2).sum(0)[None, :], 0.0))
    return {"spearman": _corr_rows_cols(rXp, rY), "cosine": cos, "euclidean": eu}


def spin_ranks(x, spins):
    """
    Ranked spin surrogates without ranking 10000 maps.

    rank(x[s]) == rank(x)[s]: ranking is invariant to reordering, so the map
    is ranked once and the spin indices are applied to the ranks. Used by the
    Delta pipeline; the pairwise script goes through null_similarity for
    metric consistency.
    """
    return rankdata(x)[spins]


# =================================================================
# Inference
# =================================================================
def perm_p(obs, null, axis=0):
    """
    Two-tailed permutation p with +1 smoothing, vectorised over targets.

    obs  : (k,)            null : (n_perm, k)
    Unchanged from the original pipeline — that part was always correct.
    """
    obs = np.asarray(obs, float)
    null = np.asarray(null, float)
    n = null.shape[axis]
    p_up = (np.sum(null >= obs[None, :], axis=axis) + 1) / (n + 1)
    p_lo = (np.sum(null <= obs[None, :], axis=axis) + 1) / (n + 1)
    return np.minimum(2.0 * np.minimum(p_up, p_lo), 1.0)


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (statsmodels is not a dependency)."""
    p = np.asarray(p, float)
    flat = p.ravel()
    n = flat.size
    order = np.argsort(flat)
    ranked = flat[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out.reshape(p.shape)


def bh_fdr_by_column(P):
    """BH within each column (each SUD map = one family), as in Figure 2."""
    P = np.asarray(P, float)
    return np.column_stack([bh_fdr(P[:, j]) for j in range(P.shape[1])])


# =================================================================
# Clinical clusters — one definition, used by every RQ1 script
# =================================================================
# These were previously re-declared in four places with THREE different
# spellings of the same four groups:
#     specificity_delta : "Mood/Anxiety", "Neurodevelopmental"
#     supp_block_table  : "Mood/Anx",     "Neurodev"
#     Figure2 generator : "Mood/Anxiety", "Neurodevelopmental"
#     JOINT16           : "Mood/Anxiety", "Neurodevelopmental"
# Joining two of those tables on the cluster column silently produced NaN.
# The Mood/Anxiety cluster now holds PD (panic disorder), not PTSD.
CLUSTER_MEMBERS = {
    "Psychotic":          ["SCZ", "BD", "CHR"],
    "AN/OCD":             ["AN", "OCD"],
    "Mood/Anxiety":       ["MDD", "PD"],
    "Neurodevelopmental": ["ADHD", "ASD"],
}
CLUSTERS = {d: c for c, members in CLUSTER_MEMBERS.items() for d in members}

CLUSTER_COLORS = {
    "Psychotic":          "#F5A623",
    "AN/OCD":             "#3B6EDB",
    "Mood/Anxiety":       "#B455C8",
    "Neurodevelopmental": "#2FB457",
}


def check_cluster_coverage(names):
    """Fail loudly if a map has no cluster (e.g. after a disorder is renamed)."""
    missing = [n for n in names if n not in CLUSTERS]
    if missing:
        raise KeyError(
            f"no clinical cluster defined for {missing}. "
            f"Known: {sorted(CLUSTERS)}. If a disorder was renamed (PTSD -> PD), "
            f"update CLUSTER_MEMBERS in RQ1_common — not in the calling script."
        )