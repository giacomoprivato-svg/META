#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
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
OUTDIR = os.path.join(repo_dir, "results", "RQ1_BRAINSMASH_ONLY")
os.makedirs(OUTDIR, exist_ok=True)

PATH_SUD = os.path.join(BASE, "SUD.xlsx")
PATH_PSY_ADULTS = os.path.join(BASE, "PSY_adults.xlsx")
PATH_PSY_ADO = os.path.join(BASE, "PSY_adolescents.xlsx")

N_CORTEX = 68
N_PERM = 10000

# ======================
# CORTICAL CENTROID DISTANCE MATRIX
# ======================
centsfile = os.path.join(BASE, "centroids_ctx_68.mat")
with h5py.File(centsfile, "r") as f:
    LH = np.array(f["centroids_lh"]).T
    RH = np.array(f["centroids_rh"]).T

COORDS = np.vstack([LH, RH])
DISTMAT = cdist(COORDS, COORDS)

# ======================
# LOAD DATA
# ======================
def load_matrix(path):
    df = pd.read_excel(path).select_dtypes(include=np.number)
    return df.to_numpy(), df.columns.tolist()

def mean_effect(X):
    return np.nanmean(X, axis=1)

# ======================
# BRAINSMASH TEST
# ======================
def brainsmash_test(psy, sud, coords=None):

    obs = np.corrcoef(psy, sud)[0, 1]

    D = DISTMAT if coords is None else cdist(coords, coords)

    gen = Base(
        x=psy,
        D=D,
        resample=True,
        seed=42
    )

    null = gen(n=N_PERM)

    null_r = np.zeros(N_PERM)
    for i in range(N_PERM):
        null_r[i] = np.corrcoef(null[i], sud)[0, 1]

    p = (np.sum(np.abs(null_r) >= np.abs(obs)) + 1) / (N_PERM + 1)

    z = (obs - np.mean(null_r)) / np.std(null_r)

    return obs, z, p

# ======================
# RUN PIPELINE
# ======================
def run_group(name, psy_mat, sud_mat):

    psy_mat = psy_mat[:N_CORTEX]
    sud_mat = sud_mat[:N_CORTEX]

    psy = mean_effect(psy_mat)
    sud = mean_effect(sud_mat)

    r, z, p = brainsmash_test(psy, sud)

    return {
        "group": name,
        "r": r,
        "z": z,
        "p_brainsmash": p
    }

# ======================
# MAIN
# ======================
SUD, SUD_names = load_matrix(PATH_SUD)
SUD = SUD[:, [n != "SUD" for n in SUD_names]]
PSY_A, _ = load_matrix(PATH_PSY_ADULTS)
PSY_P, _ = load_matrix(PATH_PSY_ADO)

results = []

# Adults
results.append(run_group("ADULTS", PSY_A, SUD))

# Adolescents
results.append(run_group("ADOLESCENTS", PSY_P, SUD))

# Cluster analysis (optional)
CLUSTERS = {
    "Psychotic": ["SCZ", "BD", "CHR"],
    "Neurodevelopmental": ["ASD", "ADHD"],
    "AN_OCD": ["AN", "OCD"],
    "Mood_Anxiety": ["MDD", "PTSD"],
}

for cname, dis in CLUSTERS.items():
    psy_df = pd.read_excel(PATH_PSY_ADULTS)[dis]
    psy = psy_df.to_numpy()
    results.append(run_group(f"CLUSTER_{cname}", psy, SUD))

# ======================
# SAVE OUTPUT
# ======================
df = pd.DataFrame(results)

df["p_fdr"] = np.nan

from statsmodels.stats.multitest import multipletests

rej, p_fdr, _, _ = multipletests(df["p_brainsmash"], method="fdr_bh")

df["p_fdr"] = p_fdr
df["significant_fdr"] = rej

outpath = os.path.join(OUTDIR, "BRAINSMASH_ONLY_RESULTS.csv")
df.to_csv(outpath, index=False)

print("\nDONE ✔")
print(df)
print(f"\nSaved to: {outpath}")