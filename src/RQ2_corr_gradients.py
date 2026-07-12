#!/usr/bin/env python3
"""
RQ2_gradients_joint_fdr.py

MERGE di RQ2_corr_C1C2C3.py + RQ2_corr_MPCFC.py in un unico script.
Sostituisce entrambi (puoi eliminarli dopo aver verificato l'output).

Cosa fa:
  - costruisce la mappa shared PSY-SUD (media adulti);
  - la correla con 5 componenti di organizzazione corticale:
        C1, C2, C3  -> diffusion map embedding, trascrizione AHBA
        FC, MPC     -> gradiente funzionale e microstrutturale (MICA-MICs);
  - test spaziali spin (10.000) e BrainSMASH (10.000), two-tailed;
  - correzione FDR (Benjamini-Hochberg) JOINT su tutti e 5 i test insieme,
    separatamente per spin e BrainSMASH (invece dei due FDR separati 3+2
    degli script originali);
  - plotting completo: mappa "atrophy" shared, mappe dei 5 gradienti,
    e relative colormap;
  - un unico CSV di output.

Fix rispetto agli originali:
  1) colonna generica "SUD" (mappa transdiagnostica aggregata) esclusa dalla
     media SUD, per evitare double counting con le 6 substance-specific.
  2) FDR joint su 5 test (non 3+2).
  3) output CSV con nome dedicato (gli originali scrivevano entrambi sullo
     stesso file, sovrascrivendosi).

------------------------------------------------------------------------------
NOTA SUL SEGNO DI r  (importante, leggere prima di aggiornare il manoscritto)
------------------------------------------------------------------------------
Nei due script originali la convenzione di segno NON era la stessa:
  - RQ2_corr_C1C2C3.py correla i gradienti con la shared map NON invertita;
  - RQ2_corr_MPCFC.py inverte la shared map (shared = -shared) prima di
    correlarla con FC/MPC.
Poiche corr(x, -shared) = -corr(x, shared), questo cambia SOLO il segno di r
(non la magnitudine ne il p-value).

Il dizionario GRADIENTS qui sotto riproduce ESATTAMENTE le due convenzioni
originali (INVERT=False per C1/C2/C3, INVERT=True per FC/MPC), cosicche i
valori di r coincidano con quelli gia riportati nel manoscritto
(C2=-0.420, C3=-0.623, FC=-0.108, MPC=-0.041).

Se preferisci una convenzione UNICA e coerente per tutti e 5 (piu pulito dal
punto di vista metodologico), metti UNIFY_SIGN = True: verra usata la shared
map non invertita per tutti. In quel caso C1/C2/C3 restano identici, ma i
segni di FC e MPC si invertono (+0.108, +0.041) e vanno aggiornati nel testo
(entrambi comunque non significativi: la conclusione non cambia).
------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import h5py

from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests
from brainsmash.mapgen.base import Base

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical

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

OUTDIR = os.path.join(repo_dir, "results", "RQ2_GRADIENTS_JOINT_FDR")
PLOT_DIR = os.path.join(OUTDIR, "brain_maps")
CMAP_DIR = os.path.join(OUTDIR, "colormaps")
for d in (OUTDIR, PLOT_DIR, CMAP_DIR):
    os.makedirs(d, exist_ok=True)

PATH_SUD = os.path.join(BASE, "SUD.xlsx")
PATH_PSY_ADULTS = os.path.join(BASE, "PSY_adults.xlsx")
PATH_DME = os.path.join(BASE, "ahba_dme_scores_in_dk.csv")
PATH_FC_GRAD = os.path.join(BASE, "mica_hc100_gradient-FC.csv")
PATH_MPC_GRAD = os.path.join(BASE, "mica_hc100_gradient-MPC.csv")

N_CORTEX = 68
N_PERM = 10000

# Convenzione di segno: vedi nota in cima al file.
# False = riproduce gli script originali (segni come nel manoscritto).
# True  = convenzione unica per tutti e 5 (FC/MPC cambiano segno).
UNIFY_SIGN = False

# =====================================================
# PLOT HELPERS
# =====================================================
def save_colormap(cmap_name, vmin, vmax, filename):
    fig, ax = plt.subplots(figsize=(6, 1))
    gradient = np.linspace(vmin, vmax, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=cmap_name, extent=[vmin, vmax, 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("Value", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CMAP_DIR, filename), dpi=300, bbox_inches='tight')
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
        scale=(4, 4),
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

def rand_rotation_matrix():
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q3*q4), 2 * (q1*q3 + q2*q4)],
        [2 * (q1*q2 + q3*q4), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q1*q4)],
        [2 * (q1*q3 - q2*q4), 2 * (q2*q3 + q1*q4), 1 - 2 * (q1**2 + q2**2)]
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

def load_excel(path, drop_cols=None):
    df = pd.read_excel(path).select_dtypes(include=np.number)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df.to_numpy()

def shared_map(psy_mat, sud_mat):
    psy = mean_effect(psy_mat)
    sud = mean_effect(sud_mat)
    return (psy + sud) / 2, psy, sud

# =====================================================
# SPIN / BRAINSMASH (two-tailed)
# =====================================================
def spin_test(x, y):
    obs = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x[idx], y)[0, 1] for idx in SPINS])
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)
    z = (obs - np.mean(null)) / np.std(null)
    return obs, z, p

def brainsmash_test(x, y):
    obs = np.corrcoef(x, y)[0, 1]
    gen = Base(x=x, D=DISTMAT, resample=True, seed=42)
    surrogates = gen(n=N_PERM)
    null = np.array([np.corrcoef(s, y)[0, 1] for s in surrogates])
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (N_PERM + 1)
    z = (obs - np.mean(null)) / np.std(null)
    return obs, z, p

# =====================================================
# LOAD DATA
# =====================================================
PSY = load_excel(PATH_PSY_ADULTS)[:N_CORTEX]
# fix: escludi la colonna generica "SUD" dall'aggregato SUD
SUD = load_excel(PATH_SUD, drop_cols=["SUD"])[:N_CORTEX]

shared, psy_mean, sud_mean = shared_map(PSY, SUD)

# =====================================================
# LOAD GRADIENTS
# =====================================================
print("Loading DME transcriptional components (C1-C3)...")
dme = pd.read_csv(PATH_DME)
dme_lh = dme[["C1", "C2", "C3"]].to_numpy()   # 34 x 3 (solo LH)
dme_full = np.vstack([dme_lh, dme_lh])        # mirror LH -> RH -> 68 x 3
C1, C2, C3 = dme_full[:, 0], dme_full[:, 1], dme_full[:, 2]

print("Loading FC / MPC cortical gradients...")
fc_grad = pd.read_csv(PATH_FC_GRAD, header=None).iloc[:, 0].to_numpy()    # 68
mpc_grad = pd.read_csv(PATH_MPC_GRAD, header=None).iloc[:, 0].to_numpy()  # 68

# gradiente -> (mappa, inversione_shared_originale)
GRADIENTS = {
    "C1":  (C1,       False),
    "C2":  (C2,       False),
    "C3":  (C3,       False),
    "FC":  (fc_grad,  True),
    "MPC": (mpc_grad, True),
}

# =====================================================
# CORRELAZIONI + TEST
# =====================================================
results = []
for name, (grad, invert_orig) in GRADIENTS.items():

    invert = False if UNIFY_SIGN else invert_orig
    target = -shared if invert else shared

    r_spin, z_spin, p_spin = spin_test(grad, target)
    r_bs, z_bs, p_bs = brainsmash_test(grad, target)

    results.append({
        "gradient": name,
        "r": r_spin,
        "z_spin": z_spin,
        "p_spin": p_spin,
        "z_brainsmash": z_bs,
        "p_brainsmash": p_bs,
        "shared_inverted": invert,
    })

    print(f"{name} vs ATROPHY | r={r_spin:+.3f} | spin p={p_spin:.5f} | BrainSMASH p={p_bs:.5f}")

df = pd.DataFrame(results)

# =====================================================
# JOINT FDR SU TUTTI E 5 I GRADIENTI
# =====================================================
_, p_fdr_spin, _, _ = multipletests(df["p_spin"], method="fdr_bh")
_, p_fdr_bs, _, _ = multipletests(df["p_brainsmash"], method="fdr_bh")

df["p_fdr_spin"] = p_fdr_spin
df["p_fdr_brainsmash"] = p_fdr_bs
df["significant_joint_fdr"] = (df["p_fdr_spin"] < 0.05) & (df["p_fdr_brainsmash"] < 0.05)

outpath = os.path.join(OUTDIR, "all_gradients_joint_fdr.csv")
df.to_csv(outpath, index=False)

print("\n" + df.to_string(index=False))
print(f"\nSaved: {outpath}")

# =====================================================
# PLOTTING
# =====================================================
print("\nPlotting maps...")

# --- mappa shared "atrophy" (QC side-output; la figura del paper e' generata
#     altrove). Uso la shared NON invertita come mappa canonica. ---
vmax = np.nanmax(np.abs(shared))
plot_brain(shared, "ATROPHY_map", -vmax, vmax, "RdBu_r")
save_colormap("RdBu_r", -vmax, vmax, "ATROPHY_colormap.png")

# --- mappe dei 5 gradienti ---
GRAD_PLOTS = {
    "C1_map": C1, "C2_map": C2, "C3_map": C3,
    "FC_gradient_map": fc_grad, "MPC_gradient_map": mpc_grad,
}
for name, vals in GRAD_PLOTS.items():
    vmax = np.nanmax(np.abs(vals))
    plot_brain(vals, name, -vmax, vmax, "RdBu_r")
    save_colormap("RdBu_r", -vmax, vmax, f"{name}_colormap.png")

print("\nDONE \u2714")