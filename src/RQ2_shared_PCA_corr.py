#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import h5py

from scipy.spatial.distance import cdist
from scipy.stats import zscore, norm
from sklearn.decomposition import PCA

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical

from brainsmash.mapgen.base import Base

# ---------------------------
# Paths (repo-relative)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)


np.random.seed(42)

# ======================
# CONFIG
# ======================
BASE = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "results", "PC1_vs_SHARED")

os.makedirs(OUTDIR, exist_ok=True)

PATH_SUD = os.path.join(BASE, "SUD.xlsx")
PATH_PSY_ADULTS = os.path.join(BASE, "PSY_adults.xlsx")

N_CORTEX = 68
N_PERM = 10000

# ======================
# LOAD DATA
# ======================
def load(path):

    df = pd.read_excel(path).select_dtypes(include=np.number)

    return df.to_numpy(), df.columns.tolist()

PSY_A, _ = load(PATH_PSY_ADULTS)
SUD, _ = load(PATH_SUD)

PSY_A = PSY_A[:N_CORTEX]
SUD = SUD[:N_CORTEX]

# ======================
# CENTROIDS
# ======================
def nn(A, B):

    return np.argmin(cdist(B, A), axis=1)

centsfile = os.path.join(BASE, "centroids_ctx_68.mat")

with h5py.File(centsfile, "r") as f:

    LH = np.array(f["centroids_lh"]).T
    RH = np.array(f["centroids_rh"]).T

# normalized coords for spin
LH_spin = LH / np.linalg.norm(LH, axis=1, keepdims=True)
RH_spin = RH / np.linalg.norm(RH, axis=1, keepdims=True)

# coords for BrainSMASH
coords = np.vstack([LH, RH])

DISTMAT = cdist(coords, coords)

# ======================
# SPIN TEST
# ======================
def rand_rotation_matrix():

    u1, u2, u3 = np.random.rand(3)

    q1 = np.sqrt(1-u1)*np.sin(2*np.pi*u2)
    q2 = np.sqrt(1-u1)*np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1)*np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1)*np.cos(2*np.pi*u3)

    return np.array([

        [
            1-2*(q2**2+q3**2),
            2*(q1*q2-q3*q4),
            2*(q1*q3+q2*q4)
        ],

        [
            2*(q1*q2+q3*q4),
            1-2*(q1**2+q3**2),
            2*(q2*q3-q1*q4)
        ],

        [
            2*(q1*q3-q2*q4),
            2*(q2*q3+q1*q4),
            1-2*(q1**2+q2**2)
        ]
    ])

def build_spins():

    spins = np.zeros((N_PERM, N_CORTEX), dtype=int)

    for k in range(N_PERM):

        R = rand_rotation_matrix()

        idxL = nn(LH_spin, (R @ LH_spin.T).T)
        idxR = nn(RH_spin, (R @ RH_spin.T).T)

        spins[k] = np.concatenate([idxL, idxR])

    return spins

SPINS = build_spins()

def spin_test(x, y):

    obs = np.corrcoef(x, y)[0, 1]

    null = np.zeros(len(SPINS))

    for i in range(len(SPINS)):

        x_s = x[SPINS[i]]

        null[i] = np.corrcoef(x_s, y)[0, 1]

    p = (
        np.sum(np.abs(null) >= np.abs(obs)) + 1
    ) / (len(null) + 1)

    z = (
        obs - np.mean(null)
    ) / np.std(null)

    return obs, z, p, null

# ======================
# BRAINSMASH TEST
# ======================
def signed_z_and_p_from_null(obs, null):

    null = np.asarray(null).ravel()

    n = null.size

    p_upper = (
        np.sum(null >= obs) + 1
    ) / (n + 1)

    p_lower = (
        np.sum(null <= obs) + 1
    ) / (n + 1)

    if p_upper <= p_lower:

        p = p_upper
        sign = 1.0

    else:

        p = p_lower
        sign = -1.0

    z = sign * norm.isf(p)

    return z, p

def brainsmash_test(x, y):

    obs = np.corrcoef(x, y)[0, 1]

    gen = Base(
        x=x,
        D=DISTMAT,
        resample=True,
        seed=42
    )

    surrogates = gen(n=N_PERM)

    null = np.zeros(N_PERM)

    for i in range(N_PERM):

        null[i] = np.corrcoef(
            surrogates[i],
            y
        )[0, 1]

    z, p = signed_z_and_p_from_null(
        obs,
        null
    )

    return obs, z, p, null

# ======================
# SHARED MAP (ADULTS)
# ======================
def mean_effect(X):

    return np.nanmean(X, axis=1)

psy_mean = mean_effect(PSY_A)
sud_mean = mean_effect(SUD)

shared_map = (
    psy_mean + sud_mean
) / 2

# ======================
# PC1 (PSY + SUD)
# ======================
X = np.concatenate(
    [PSY_A, SUD],
    axis=1
)

# standardizzazione per regione
Xz = zscore(
    X,
    axis=0,
    nan_policy='omit'
)

pca = PCA(n_components=1)

pc1 = pca.fit_transform(Xz).flatten()

# ======================
# CORRELATIONS
# ======================

# ---------- SPIN ----------
r_spin, z_spin, p_spin, null_spin = spin_test(
    shared_map,
    pc1
)

print(
    f"\n[SPIN] Shared vs PC1 "
    f"→ r = {r_spin:.3f}, "
    f"z = {z_spin:.3f}, "
    f"p = {p_spin:.5f}"
)

# ---------- BRAINSMASH ----------
r_bs, z_bs, p_bs, null_bs = brainsmash_test(
    shared_map,
    pc1
)

print(
    f"\n[BRAINSMASH] Shared vs PC1 "
    f"→ r = {r_bs:.3f}, "
    f"z = {z_bs:.3f}, "
    f"p = {p_bs:.5f}"
)

# ======================
# SAVE DATA
# ======================
df = pd.DataFrame({

    "region": np.arange(N_CORTEX),

    "shared_map": shared_map,

    "PC1": pc1
})

df.to_csv(
    os.path.join(
        OUTDIR,
        "shared_vs_PC1.csv"
    ),
    index=False
)

# ======================
# SAVE STATS
# ======================
stats_df = pd.DataFrame({

    "method": [
        "spin",
        "brainsmash"
    ],

    "r": [
        r_spin,
        r_bs
    ],

    "z": [
        z_spin,
        z_bs
    ],

    "p": [
        p_spin,
        p_bs
    ]
})

stats_df.to_csv(
    os.path.join(
        OUTDIR,
        "shared_vs_PC1_stats.csv"
    ),
    index=False
)

# optional null distributions
pd.DataFrame({
    "spin_null": null_spin
}).to_csv(
    os.path.join(
        OUTDIR,
        "spin_null_distribution.csv"
    ),
    index=False
)

pd.DataFrame({
    "brainsmash_null": null_bs
}).to_csv(
    os.path.join(
        OUTDIR,
        "brainsmash_null_distribution.csv"
    ),
    index=False
)

# ======================
# PLOTTING FUNCTIONS
# ======================
def plot_brain(vals, name, vmin, vmax, cmap):

    CT = parcel_to_surface(
        vals,
        "aparc_fsa5"
    )

    plot_cortical(

        array_name=CT,

        surface_name="fsa5",

        cmap=cmap,

        color_range=(vmin, vmax),

        size=(800, 400),

        zoom=1.25,

        scale=(4,4),

        background=(1, 1, 1),

        color_bar="bottom",

        share="b",

        screenshot=True,

        filename=os.path.join(
            OUTDIR,
            f"{name}.png"
        )
    )

def plot_brain_sctx(vals14, name, vmin, vmax, cmap):

    vals16 = np.full(16, np.nan)

    vals16[:7] = vals14[:7]

    vals16[8:15] = vals14[7:]

    plot_subcortical(

        vals16,

        cmap=cmap,

        color_range=(vmin, vmax),

        size=(800, 400),

        zoom=1.25,

        scale=(4,4),

        background=(1, 1, 1),

        color_bar="bottom",

        share="b",

        screenshot=True,

        filename=os.path.join(
            OUTDIR,
            f"{name}_SCTX.png"
        )
    )

# ======================
# PLOT MAPS
# ======================

# Shared map
vmin_shared = np.nanmin(shared_map)

plot_brain(
    shared_map,
    "SHARED_MAP",
    vmin_shared,
    0,
    "YlOrRd_r"
)

plot_brain_sctx(
    shared_map[-14:],
    "SHARED_MAP",
    vmin_shared,
    0,
    "YlOrRd_r"
)

# PC1
vmax_pc1 = np.nanmax(
    np.abs(pc1)
)

plot_brain(
    pc1,
    "PC1_MAP",
    -vmax_pc1,
    vmax_pc1,
    "RdBu_r"
)

plot_brain_sctx(
    pc1[-14:],
    "PC1_MAP",
    -vmax_pc1,
    vmax_pc1,
    "RdBu_r"
)

print("\nDONE ✔")