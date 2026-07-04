#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from statsmodels.stats.multitest import multipletests

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
OUTDIR = os.path.join(repo_dir, "results", "RQ1_SHARED_ALTERATION")
os.makedirs(OUTDIR, exist_ok=True)

PATH_SUD = os.path.join(BASE, "SUD.xlsx")
PATH_PSY_ADULTS = os.path.join(BASE, "PSY_adults.xlsx")
PATH_PSY_ADO = os.path.join(BASE, "PSY_adolescents.xlsx")
PATH_PSY_ADO_CTX = os.path.join(BASE, "PSY_adolescents_ctx.xlsx")

N_CORTEX = 68
N_SUBCTX = 14
N_TOTAL = N_CORTEX + N_SUBCTX

N_PERM = 10000

# ======================
# CLUSTERS
# ======================
CLUSTERS = {
    "Psychotic": ["SCZ", "BD", "CHR"],
    "Neurodevelopmental": ["ASD", "ADHD"],
    "AN_OCD": ["AN", "OCD"],
    "Mood_Anxiety": ["MDD", "PTSD"],
}

# ======================
# CENTROIDS
# ======================
def nn(A, B):
    return np.argmin(cdist(B, A), axis=1)

centsfile = os.path.join(BASE, "centroids_ctx_68.mat")
with h5py.File(centsfile, "r") as f:
    LH = np.array(f["centroids_lh"]).T
    RH = np.array(f["centroids_rh"]).T

LH /= np.linalg.norm(LH, axis=1, keepdims=True)
RH /= np.linalg.norm(RH, axis=1, keepdims=True)

# ======================
# SPINS
# ======================
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

SPINS = build_spins()

# ======================
# HELPERS
# ======================
def mean_effect(X):
    return np.nanmean(X, axis=1)

def load(path):
    df = pd.read_excel(path).select_dtypes(include=np.number)
    return df.to_numpy(), df.columns.tolist()

# ======================
# SHARED MAP
# ======================
def shared_alteration_map(psy_mat, sud_vec):
    psy = mean_effect(psy_mat)
    sud = mean_effect(sud_vec)
    shared = (psy + sud) / 2
    return shared, psy, sud

# ======================
# CONTRIBUTION MAP
# ======================
def contribution_map(psy, sud):
    psy_abs = np.abs(psy)
    sud_abs = np.abs(sud)

    denom = psy_abs + sud_abs
    contrib = np.full_like(psy_abs, np.nan)

    valid = denom != 0
    contrib[valid] = (psy_abs[valid] / denom[valid]) * 100

    return contrib

# ======================
# SPIN TEST (cortex only)
# ======================
def spin_test_correlation(psy, sud):

    obs = np.corrcoef(psy, sud)[0, 1]
    null = np.zeros(len(SPINS))

    for i in range(len(SPINS)):
        psy_s = psy[SPINS[i]]
        null[i] = np.corrcoef(psy_s, sud)[0, 1]

    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)
    z = (obs - np.mean(null)) / np.std(null)

    return obs, z, p

# ======================
# LOO
# ======================
def leave_one_out_analysis(data_mat, colnames, label):

    results = []
    full = zscore(mean_effect(data_mat), nan_policy='omit')

    for i in range(data_mat.shape[1]):
        reduced = np.delete(data_mat, i, axis=1)
        loo = zscore(mean_effect(reduced), nan_policy='omit')
        r = np.corrcoef(full, loo)[0, 1]

        results.append({
            "removed_disorder": colnames[i],
            "correlation_with_full": r
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTDIR, f"LOO_{label}.csv"), index=False)

    print(f"\nLOO {label} → min r = {df['correlation_with_full'].min():.3f}")

def reorder_sctx_for_csv(vals14):
    """
    Input ENIGMA standard:
    0 L Accumbens
    1 L Amygdala
    2 L Caudate
    3 L Hippocampus
    4 L Pallidum
    5 L Putamen
    6 L Thalamus
    7 R Accumbens
    8 R Amygdala
    9 R Caudate
    10 R Hippocampus
    11 R Pallidum
    12 R Putamen
    13 R Thalamus
    """

    order = [
        6, 2, 5, 4, 3, 1, 0,   # LEFT: Thal → Caud → Put → Pall → Hip → Amy → Acc
        13, 9, 12, 11, 10, 8, 7 # RIGHT: Thal → Caud → Put → Pall → Hip → Amy → Acc
    ]

    return vals14[order]

# ======================
# PLOT CORTEX
# ======================
def plot_brain(vals, name, vmax, cmap="RdBu_r", center_zero=True):

    CT = parcel_to_surface(vals, "aparc_fsa5")

    crange = (-vmax, vmax) if center_zero else (np.nanmin(vals), np.nanmax(vals))

    plot_cortical(
        array_name=CT,
        surface_name="fsa5",
        cmap=cmap,
        color_range=crange,
        size=(800, 400),
        zoom=1.25,
        scale=(4,4),
        background=(1, 1, 1),
        color_bar="bottom",
        share="b",
        screenshot=True,
        filename=os.path.join(OUTDIR, f"{name}.png")
    )

# ======================
# SUBCORTEX PLOT (REAL SUBCORTEX)
# ======================
def plot_brain_sctx(vals14, name, vmax, cmap="RdBu_r"):

    vals16 = np.full(16, np.nan)
    vals16[:7] = vals14[:7]
    vals16[8:15] = vals14[7:]

    plot_subcortical(
        vals16,
        cmap=cmap,
        color_range=(-vmax, vmax),
        size=(800, 400),
        zoom=1.25,
        scale=(4,4),
        background=(1, 1, 1),
        color_bar="bottom",
        share="b",
        screenshot=True,
        filename=os.path.join(OUTDIR, f"{name}_SCTX.png")
    )

# ======================
# SHARED MAP PLOTS
# ======================
def plot_shared_brain(vals, name, vmin, vmax=0):

    CT = parcel_to_surface(vals, "aparc_fsa5")

    plot_cortical(
        array_name=CT,
        surface_name="fsa5",
        cmap="YlOrRd_r",
        color_range=(vmin, vmax),
        size=(800, 400),
        zoom=1.25,
        scale=(4,4),
        background=(1, 1, 1),
        color_bar="bottom",
        share="b",
        screenshot=True,
        filename=os.path.join(OUTDIR, f"{name}.png")
    )

def plot_shared_brain_sctx(vals14, name, vmin, vmax=0):

    vals16 = np.full(16, np.nan)
    vals16[:7] = vals14[:7]
    vals16[8:15] = vals14[7:]

    plot_subcortical(
        vals16,
        cmap="YlOrRd_r",
        color_range=(vmin, vmax),
        size=(800, 400),
        zoom=1.25,
        scale=(4,4),
        background=(1, 1, 1),
        color_bar="bottom",
        share="b",
        screenshot=True,
        filename=os.path.join(OUTDIR, f"{name}_SCTX.png")
    )

# ======================
# STORAGE
# ======================
ALL_SHARED = []
GLOBAL_STATS = []

# ======================
# RUN
# ======================
def run(name, psy_mat):

    sud_df = pd.read_excel(PATH_SUD).select_dtypes(include=np.number)
    sud = sud_df.to_numpy()

    psy_mat = psy_mat

    shared, psy_mean, sud_mean = shared_alteration_map(psy_mat, sud)

    ALL_SHARED.append(shared)

    # spin test ONLY cortex
    r, z, p = spin_test_correlation(
        psy_mean[:N_CORTEX],
        sud_mean[:N_CORTEX]
    )

    print(f"{name} → r={r:.3f}, z={z:.3f}, p={p:.5f}")

    GLOBAL_STATS.append({
        "group": name,
        "correlation_r": r,
        "z": z,
        "p_spin": p
    })

    contrib = contribution_map(psy_mean, sud_mean)

    # ======================
    # CSV (cortex + reordered subcortex ONLY for output)
    # ======================

    sctx_reordered = reorder_sctx_for_csv(psy_mean[N_CORTEX:])

    regions = [
    "CTX_" + str(i+1) for i in range(N_CORTEX)
    ] + [
        "L_Thalamus",
        "L_Caudate",
        "L_Putamen",
        "L_Pallidum",
        "L_Hippocampus",
        "L_Amygdala",
        "L_Accumbens",

        "R_Thalamus",
        "R_Caudate",
        "R_Putamen",
        "R_Pallidum",
        "R_Hippocampus",
        "R_Amygdala",
        "R_Accumbens"
    ]

    df = pd.DataFrame({
        "region": regions,
        "PSY_z": np.concatenate([psy_mean[:N_CORTEX], sctx_reordered]),
        "SUD_z": np.concatenate([sud_mean[:N_CORTEX], reorder_sctx_for_csv(sud_mean[N_CORTEX:])]),
        "shared_alteration": np.concatenate([shared[:N_CORTEX], reorder_sctx_for_csv(shared[N_CORTEX:])]),
        "PSY_contribution_%": np.concatenate([contrib[:N_CORTEX], reorder_sctx_for_csv(contrib[N_CORTEX:])])
    })

    df.to_csv(os.path.join(OUTDIR, f"{name}_shared_map.csv"), index=False)

    vmax = 50

    # cortex only
    plot_brain(contrib[:N_CORTEX] - 50,
               f"{name}_PSY_contribution",
               vmax,
               cmap="PiYG")

    # subcortex REAL
    plot_brain_sctx(contrib[N_CORTEX:] - 50,
                    f"{name}_PSY_contribution",
                    vmax,
                    cmap="PiYG")

    return shared

# ======================
# LOAD + RUN
# ======================
PSY_A, PSY_A_names = load(PATH_PSY_ADULTS)

PSY_P, PSY_P_names = load(PATH_PSY_ADO)
PSY_P_CTX, PSY_P_CTX_names = load(PATH_PSY_ADO_CTX)

SUD_FULL, SUD_names = load(PATH_SUD)

# =====================================================
# MERGE ADOLESCENT DISORDERS WITH CORTEX-ONLY DISORDERS
# =====================================================

extra_ctx = np.full((N_TOTAL, PSY_P_CTX.shape[1]), np.nan)

# fill cortex rows only (68 regions)
extra_ctx[:N_CORTEX, :] = PSY_P_CTX

# append new adolescent disorders
PSY_P_COMBINED = np.hstack([PSY_P, extra_ctx])

PSY_P_COMBINED_names = PSY_P_names + PSY_P_CTX_names

leave_one_out_analysis(PSY_A, PSY_A_names, "PSY_ADULTS")
leave_one_out_analysis(PSY_P_COMBINED, PSY_P_COMBINED_names, "PSY_ADOLESCENTS")
leave_one_out_analysis(SUD_FULL, SUD_names, "SUD")

run("ADULTS", PSY_A)
run("ADOLESCENTS", PSY_P_COMBINED)

for cname, dis in CLUSTERS.items():
    psy = pd.read_excel(PATH_PSY_ADULTS)[dis].to_numpy()
    run(f"CLUSTER_{cname}", psy)

names = ["ADULTS","ADOLESCENTS"] + [f"CLUSTER_{c}" for c in CLUSTERS]

shared_dict = dict(zip(names, ALL_SHARED))

adults_map = shared_dict["ADULTS"]

corr_results = []

for cname in CLUSTERS:
    cluster_name = f"CLUSTER_{cname}"
    cluster_map = shared_dict[cluster_name]

    r = np.corrcoef(adults_map, cluster_map)[0, 1]

    corr_results.append({
        "comparison": f"ADULTS_vs_{cluster_name}",
        "correlation_r": r
    })

    print(f"ADULTS vs {cluster_name} → r = {r:.3f}")

df_corr = pd.DataFrame(corr_results)
df_corr.to_csv(os.path.join(OUTDIR, "ADULTS_vs_CLUSTERS_correlations.csv"), index=False)

all_vals = np.concatenate(ALL_SHARED)
vmin_shared_global = np.nanmin(all_vals)

for name, vals in zip(names, ALL_SHARED):

    plot_shared_brain(vals[:N_CORTEX],
                      f"{name}_shared_map",
                      vmin_shared_global,
                      vmax=0)

    plot_shared_brain_sctx(vals[N_CORTEX:],
                           f"{name}_shared_map",
                           vmin_shared_global,
                           vmax=0)

df_stats = pd.DataFrame(GLOBAL_STATS)

rej, p_fdr, _, _ = multipletests(df_stats["p_spin"], method="fdr_bh")

df_stats["p_fdr"] = p_fdr
df_stats["significant_fdr"] = rej

df_stats.to_csv(os.path.join(OUTDIR, "GLOBAL_CORRELATION_SPIN.csv"), index=False)

print("\nDONE ✔")