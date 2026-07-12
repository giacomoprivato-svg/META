#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import seaborn as sns

from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests
from brainsmash.mapgen.base import Base

# ---------------------------
# Paths (repo-relative)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)


# =====================================================
# CONFIG
# =====================================================
BASE = os.path.join(repo_dir, "data", "raw")

OUTDIR = os.path.join(repo_dir, "results", "RQ1_CLUSTER_HEATMAP_FINAL")
os.makedirs(OUTDIR, exist_ok=True)

PATH_PSY = os.path.join(BASE, "PSY_adults.xlsx")
PATH_SUD = os.path.join(BASE, "SUD.xlsx")
PATH_DME = os.path.join(BASE, "ahba_dme_scores_in_dk.csv")

N_CORTEX = 68
N_PERM = 10000

# =====================================================
# CLUSTERS
# =====================================================
CLUSTERS = {
    "Psychotic": ["SCZ", "BD", "CHR"],
    "AN/OCD": ["AN", "OCD"],
    "Mood/Anxiety": ["MDD", "PTSD"],
    "Neurodevelopmental": ["ASD", "ADHD"]
}

GRADIENTS = ["C2", "C3"]

# =====================================================
# LOAD DATA
# =====================================================
def load(path):
    df = pd.read_excel(path).select_dtypes(include=np.number)
    return df.to_numpy(), list(df.columns)

PSY, psy_names = load(PATH_PSY)
SUD, sud_names = load(PATH_SUD)
SUD = SUD[:, [n != "SUD" for n in sud_names]]
sud_names = [n for n in sud_names if n != "SUD"]

def mean_effect(X):
    return np.nanmean(X, axis=1)

def shared_map(mat):
    return (mean_effect(mat) + mean_effect(SUD)) / 2

# =====================================================
# CENTROIDS
# =====================================================
centsfile = os.path.join(BASE, "centroids_ctx_68.mat")

with h5py.File(centsfile, "r") as f:
    LH = np.array(f["centroids_lh"]).T
    RH = np.array(f["centroids_rh"]).T

LH /= np.linalg.norm(LH, axis=1, keepdims=True)
RH /= np.linalg.norm(RH, axis=1, keepdims=True)

def nn(A, B):
    return np.argmin(cdist(B, A), axis=1)

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

print("Building centroid-based spins...")
SPINS = build_spins()

# =====================================================
# PRECOMPUTE DIST MATRIX
# =====================================================
COORDS = np.vstack([LH, RH])
DISTMAT = cdist(COORDS, COORDS)

# =====================================================
# BrainSMASH CACHE
# =====================================================
BS_CACHE = {}

def brainsmash_surrogates(x):
    key = x.tobytes()
    if key in BS_CACHE:
        return BS_CACHE[key]

    gen = Base(x=x, D=DISTMAT, resample=True, seed=42)
    surrogates = gen(n=N_PERM)
    BS_CACHE[key] = surrogates
    return surrogates

def brainsmash_test(x, y):
    obs = np.corrcoef(x, y)[0, 1]
    surrogates = brainsmash_surrogates(x)

    null = np.array([
        np.corrcoef(s, y)[0, 1] for s in surrogates
    ])

    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (N_PERM + 1)
    return obs, p

def spin_test(x, y):
    obs = np.corrcoef(x, y)[0, 1]

    null = np.array([
        np.corrcoef(x[idx], y)[0, 1]
        for idx in SPINS
    ])

    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)
    return obs, p

# =====================================================
# DME
# =====================================================
dme = pd.read_csv(PATH_DME)
C2 = np.tile(dme["C2"].to_numpy(), 2)[:N_CORTEX]
C3 = np.tile(dme["C3"].to_numpy(), 2)[:N_CORTEX]

# =====================================================
# MAIN
# =====================================================
results = []

for cluster, disorders in CLUSTERS.items():

    cols = [psy_names.index(d) for d in disorders if d in psy_names]
    mat = PSY[:, cols]

    shared = shared_map(mat)
    shared_ctx = shared[:N_CORTEX]

    for grad_name, grad in zip(GRADIENTS, [C2, C3]):

        r_spin, p_spin = spin_test(shared_ctx, grad)
        r_bs, p_bs = brainsmash_test(shared_ctx, grad)

        results.append({
            "cluster": cluster,
            "gradient": grad_name,
            "r": r_spin,
            "p_spin": p_spin,
            "p_brainsmash": p_bs
        })

        print(cluster, grad_name, r_spin)

# =====================================================
# FDR
# =====================================================
df = pd.DataFrame(results)

_, df["p_fdr_spin"], _, _ = multipletests(df["p_spin"], method="fdr_bh")
_, df["p_fdr_brainsmash"], _, _ = multipletests(df["p_brainsmash"], method="fdr_bh")

df["sig"] = (df["p_fdr_spin"] < 0.05) & (df["p_fdr_brainsmash"] < 0.05)

df.to_csv(os.path.join(OUTDIR, "cluster_gradient_spin_brainsmash.csv"), index=False)

# =====================================================
# HEATMAP (IDENTICA allo script 2)
# =====================================================
clusters = list(CLUSTERS.keys())
gradients = GRADIENTS

heat = np.zeros((len(clusters), len(gradients)))
sig = np.zeros_like(heat, dtype=bool)

for i, c in enumerate(clusters):
    for j, g in enumerate(gradients):

        row = df[(df["cluster"] == c) & (df["gradient"] == g)]

        heat[i, j] = row["r"].values[0]
        sig[i, j] = row["sig"].values[0]

# ---- STYLE IDENTICO SCRIPT 2 ----
sns.set(style="white")

plt.figure(figsize=(4.2, 3.8))

blue_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "trunc_blue",
    plt.get_cmap("RdBu_r")(np.linspace(0.0, 0.5, 256))
)

ax = sns.heatmap(
    heat,
    cmap=blue_cmap,
    vmin=-1,
    vmax=0,
    linewidths=0.8,
    linecolor="white",
    cbar=False,
    square=True
)

ax.set_xticklabels(gradients, fontsize=16)
ax.set_yticklabels(clusters, fontsize=16, rotation=0)

ax.xaxis.tick_top()
ax.tick_params(axis='x', top=False, bottom=False,
               labeltop=True, labelbottom=False, pad=0)
ax.tick_params(axis='y', left=False, right=False, pad=0)

for i in range(len(clusters)):
    for j in range(len(gradients)):
        if sig[i, j]:
            ax.text(j + 0.5, i + 0.5, "*",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold")

norm = mpl.colors.Normalize(vmin=-1, vmax=0)
sm = mpl.cm.ScalarMappable(cmap=blue_cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(
    sm,
    ax=ax,
    orientation="vertical",
    fraction=0.04,
    pad=0.02
)

cbar.set_label("Correlation (r)", fontsize=14, labelpad=6)
cbar.set_ticks([-1, 0])
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "heatmap_centroid_spin_brainsmash.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("DONE ✔")