#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import h5py

from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.decomposition import PCA

from enigmatoolbox.datasets import fetch_ahba
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical

from statsmodels.stats.multitest import multipletests
from brainsmash.mapgen.base import Base

import matplotlib.pyplot as plt

# ---------------------------
# Paths (repo-relative)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)


np.random.seed(42)

# =====================================================
# CONFIG
# =====================================================
BASE = os.path.join(repo_dir, "data", "raw")

OUTDIR = os.path.join(repo_dir, "results", "RQ1_GENE_CORR")
os.makedirs(OUTDIR, exist_ok=True)

PATH_SUD = os.path.join(BASE, "SUD.xlsx")
PATH_PSY_ADULTS = os.path.join(BASE, "PSY_adults.xlsx")

# NEW
PATH_DME = os.path.join(BASE, "ahba_dme_scores_in_dk.csv")

N_CORTEX = 68
N_PERM = 10000

# =====================================================
# OUTPUT SUBDIRS
# =====================================================
PLOT_DIR = os.path.join(OUTDIR, "brain_maps")
CMAP_DIR = os.path.join(OUTDIR, "colormaps")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CMAP_DIR, exist_ok=True)

# =====================================================
# PLOT HELPERS
# =====================================================
def save_colormap(cmap_name, vmin, vmax, filename):

    fig, ax = plt.subplots(figsize=(6, 1))

    gradient = np.linspace(vmin, vmax, 256)
    gradient = np.vstack((gradient, gradient))

    ax.imshow(
        gradient,
        aspect='auto',
        cmap=cmap_name,
        extent=[vmin, vmax, 0, 1]
    )

    ax.set_yticks([])
    ax.set_xlabel("Value", fontsize=12)

    plt.tight_layout()

    plt.savefig(
        os.path.join(CMAP_DIR, filename),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

def plot_brain(vals, name, vmin, vmax, cmap):

    CT = parcel_to_surface(vals, "aparc_fsa5")

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
        filename=os.path.join(PLOT_DIR, f"{name}.png")
    )

# =====================================================
# CENTROIDS (SPINS + BRAINSMASH)
# =====================================================
def nn(A, B):
    return np.argmin(cdist(B, A), axis=1)

centsfile = os.path.join(BASE, "centroids_ctx_68.mat")

with h5py.File(centsfile, "r") as f:

    LH = np.array(f["centroids_lh"]).T
    RH = np.array(f["centroids_rh"]).T

LH /= np.linalg.norm(LH, axis=1, keepdims=True)
RH /= np.linalg.norm(RH, axis=1, keepdims=True)

COORDS = np.vstack([LH, RH])

DISTMAT = cdist(COORDS, COORDS)

# =====================================================
# SPINS
# =====================================================
def rand_rotation_matrix():

    u1, u2, u3 = np.random.rand(3)

    q1 = np.sqrt(1-u1)*np.sin(2*np.pi*u2)
    q2 = np.sqrt(1-u1)*np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1)*np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1)*np.cos(2*np.pi*u3)

    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q3*q4), 2*(q1*q3+q2*q4)],
        [2*(q1*q2+q3*q4), 1-2*(q1**2+q3**2), 2*(q2*q3-q1*q4)],
        [2*(q1*q3-q2*q4), 2*(q2*q3+q1*q4), 1-2*(q1**2+q2**2)]
    ])

def build_spins():

    spins = np.zeros((N_PERM, N_CORTEX), dtype=int)

    for k in range(N_PERM):

        R = rand_rotation_matrix()

        idxL = nn(LH, (R @ LH.T).T)
        idxR = nn(RH, (R @ RH.T).T)

        spins[k] = np.concatenate([idxL, idxR])

    return spins

print("Building spins...")
SPINS = build_spins()

# =====================================================
# HELPERS
# =====================================================
def mean_effect(X):
    return np.nanmean(X, axis=1)

def load_excel(path):
    df = pd.read_excel(path).select_dtypes(include=np.number)
    return df.to_numpy()

# =====================================================
# SHARED MAP
# =====================================================
def shared_map(psy_mat, sud_vec):

    psy = mean_effect(psy_mat)
    sud = mean_effect(sud_vec)

    shared = (psy + sud) / 2

    return shared, psy, sud

# =====================================================
# SPIN TEST
# =====================================================
def spin_test(x, y):

    obs = np.corrcoef(x, y)[0, 1]

    null = np.zeros(len(SPINS))

    for i in range(len(SPINS)):

        x_s = x[SPINS[i]]

        null[i] = np.corrcoef(x_s, y)[0, 1]

    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)

    z = (obs - np.mean(null)) / np.std(null)

    return obs, z, p

# =====================================================
# BRAINSMASH TEST
# =====================================================
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

    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (N_PERM + 1)

    z = (obs - np.mean(null)) / np.std(null)

    return obs, z, p

# =====================================================
# LOAD DATA
# =====================================================
PSY = load_excel(PATH_PSY_ADULTS)[:N_CORTEX]
SUD = load_excel(PATH_SUD)[:N_CORTEX]

shared, psy_mean, sud_mean = shared_map(PSY, SUD)

# =====================================================
# LOAD DME GRADIENTS
# =====================================================
print("Loading DME gradients...")

dme = pd.read_csv(PATH_DME)

# Keep only C1/C2/C3
dme_lh = dme[["C1", "C2", "C3"]].to_numpy()

# Mirror LH -> RH
dme_full = np.vstack([dme_lh, dme_lh])

# 68 x 3
C1 = dme_full[:, 0]
C2 = dme_full[:, 1]
C3 = dme_full[:, 2]

print("DME shape:", dme_full.shape)

# =====================================================
# CORRELATIONS
# =====================================================
results = []

def compute_all(x, y, name_x, name_y):

    r_spin, z_spin, p_spin = spin_test(x, y)

    r_bs, z_bs, p_bs = brainsmash_test(x, y)

    results.append({

        "X": name_x,
        "Y": name_y,

        "r_spin": r_spin,
        "z_spin": z_spin,
        "p_spin": p_spin,

        "r_brainsmash": r_bs,
        "z_brainsmash": z_bs,
        "p_brainsmash": p_bs
    })

    print(
        f"{name_x} vs {name_y} | "
        f"r={r_spin:.3f} | "
        f"SPIN p={p_spin:.5f} | "
        f"BRAINSMASH p={p_bs:.5f}"
    )

# =====================================================
# RUN ALL
# =====================================================
compute_all(C1, shared, "C1", "ATROPHY")
compute_all(C2, shared, "C2", "ATROPHY")
compute_all(C3, shared, "C3", "ATROPHY")

# =====================================================
# FDR
# =====================================================
pvals_spin = [r["p_spin"] for r in results]
pvals_bs = [r["p_brainsmash"] for r in results]

_, p_fdr_spin, _, _ = multipletests(
    pvals_spin,
    method="fdr_bh"
)

_, p_fdr_bs, _, _ = multipletests(
    pvals_bs,
    method="fdr_bh"
)

for i in range(len(results)):

    results[i]["p_fdr_spin"] = p_fdr_spin[i]
    results[i]["p_fdr_brainsmash"] = p_fdr_bs[i]

# =====================================================
# SAVE CSV
# =====================================================
df_results = pd.DataFrame(results)

df_results.to_csv(
    os.path.join(OUTDIR, "all_correlations_spin_brainsmash.csv"),
    index=False
)

# =====================================================
# PLOTTING
# =====================================================
print("Plotting maps...")

# ATROPHY MAP
vmax = np.nanmax(np.abs(shared))

plot_brain(
    shared,
    "ATROPHY_map",
    -vmax,
    vmax,
    "RdBu_r"
)

save_colormap(
    "RdBu_r",
    -vmax,
    vmax,
    "ATROPHY_colormap.png"
)

# DME COMPONENTS
for vals, name in zip(
    [C1, C2, C3],
    ["C1_map", "C2_map", "C3_map"]
):

    vmax = np.nanmax(np.abs(vals))

    plot_brain(
        vals,
        name,
        -vmax,
        vmax,
        "RdBu_r"
    )

    save_colormap(
        "RdBu_r",
        -vmax,
        vmax,
        f"{name}_colormap.png"
    )

print("\nDONE ✔")