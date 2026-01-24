#!/usr/bin/env python3
"""
RQ1 similarity pipeline (Python) — unified
Fix: saves NULLS cortex and works for cortex-only groups
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from tqdm import tqdm
import h5py
from enigmatoolbox.permutation_testing import shuf_test

# ---------------------------
# USER OPTIONS
# ---------------------------
N_PERM = 10000
N_CORTEX = 68
SAVE_NULLS = True  # <--- changed to allow saving NULLS cortex
np.random.seed(42)

GROUPS = [
    ("adults_all", "PSY_adults.xlsx"),
    ("adolescents_all", "PSY_adolescents.xlsx"),
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
main_outdir = os.path.join(repo_dir, "ALL_outputs_RQ1")
os.makedirs(main_outdir, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def rand_rotation_matrix():
    u1,u2,u3 = np.random.rand(3)
    q1 = np.sqrt(1-u1)*np.sin(2*np.pi*u2)
    q2 = np.sqrt(1-u1)*np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1)*np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1)*np.cos(2*np.pi*u3)
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q3*q4), 2*(q1*q3+q2*q4)],
        [2*(q1*q2+q3*q4), 1-2*(q1**2+q3**2), 2*(q2*q3-q1*q4)],
        [2*(q1*q3-q2*q4), 2*(q2*q3+q1*q4), 1-2*(q1**2+q2**2)]
    ])

def local_nn(A,B):
    return np.argmin(cdist(B,A), axis=1)

def spearman_rankcorr_obs(X,Y):
    rx = rankdata(X, method="average")
    ry = rankdata(Y, method="average")
    mx,my = rx.mean(), ry.mean()
    num = np.sum((rx-mx)*(ry-my))
    den = np.sqrt(np.sum((rx-mx)**2)*np.sum((ry-my)**2))
    return 0.0 if den==0 else float(num/den)

def read_excel_numeric_matrix(xlsx_path):
    df = pd.read_excel(xlsx_path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_cols].to_numpy()
    nonnum_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if nonnum_cols:
        region_names = df[nonnum_cols[0]].astype(str).to_numpy()
    else:
        region_names = np.array([f"R{i+1}" for i in range(df.shape[0])], dtype=str)
    return X, num_cols, region_names

def vectorized_similarity(X,Y,perms):
    nperm = perms.shape[0]
    obs = np.zeros(3, dtype=float)
    nulls = np.zeros((nperm,3), dtype=float)
    # Observed
    obs[0] = spearman_rankcorr_obs(X,Y)
    nx,ny = np.linalg.norm(X), np.linalg.norm(Y)
    obs[1] = 0.0 if nx==0 or ny==0 else float(np.dot(X,Y)/(nx*ny))
    obs[2] = -float(np.linalg.norm(X-Y))
    # Permuted X
    Xp = X[perms]
    rankY = rankdata(Y, method="average")
    meanY = rankY.mean()
    rankXp = np.apply_along_axis(rankdata,1,Xp)
    meanX = rankXp.mean(axis=1, keepdims=True)
    cov = np.sum((rankXp-meanX)*(rankY-meanY), axis=1)
    stdX = np.sqrt(np.sum((rankXp-meanX)**2, axis=1))
    stdY = np.sqrt(np.sum((rankY-meanY)**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        nulls[:,0] = np.where((stdX==0)|(stdY==0), 0.0, cov/(stdX*stdY))
    Xn = np.linalg.norm(Xp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nulls[:,1] = np.where((Xn==0)|(ny==0), 0.0, np.sum(Xp*Y, axis=1)/(Xn*ny))
    nulls[:,2] = -np.linalg.norm(Xp-Y, axis=1)
    return obs, nulls

# ---------------------------
# Load SUD
# ---------------------------
sud_file = os.path.join(data_dir,"SUD.xlsx")
if not os.path.exists(sud_file):
    raise FileNotFoundError(f"SUD.xlsx not found")
X_sud, sud_names, _ = read_excel_numeric_matrix(sud_file)

# ---------------------------
# Main loop
# ---------------------------
for group_label, psy_file_name in GROUPS:
    print(f"\n== Running group: {group_label} ==")
    psy_file = os.path.join(data_dir, psy_file_name)
    if not os.path.exists(psy_file):
        raise FileNotFoundError(f"{psy_file_name} not found")
    outdir = os.path.join(main_outdir, group_label)
    os.makedirs(outdir, exist_ok=True)

    # Load PSY
    X_psy, psy_names, _ = read_excel_numeric_matrix(psy_file)

    # ---------- Cortex handling ----------
    cortex_n = min(N_CORTEX, X_psy.shape[0], X_sud.shape[0])
    cortex_idx = np.arange(cortex_n)
    X_psy_cortex = X_psy[:cortex_n,:]
    X_sud_cortex = X_sud[:cortex_n,:]
    n_psy = X_psy_cortex.shape[1]
    n_sud = X_sud_cortex.shape[1]

    # Spins cortex
    spinfile = os.path.join(outdir,"spins_ctx_68.mat")
    if os.path.exists(spinfile):
        spins_ctx = sio.loadmat(spinfile)["spins_ctx"]-1
    else:
        centsfile = os.path.join(data_dir,"centroids_ctx_68.mat")
        if not os.path.exists(centsfile):
            raise FileNotFoundError("Missing centroids_ctx_68.mat")
        with h5py.File(centsfile,"r") as f:
            LH = np.array(f["centroids_lh"]).T
            RH = np.array(f["centroids_rh"]).T
        LH = LH/np.linalg.norm(LH, axis=1, keepdims=True)
        RH = RH/np.linalg.norm(RH, axis=1, keepdims=True)
        spins_ctx = np.zeros((N_PERM,cortex_n), dtype=int)
        for k in range(N_PERM):
            R = rand_rotation_matrix()
            idxL = local_nn(LH, (R@LH.T).T)
            idxR = local_nn(RH, (R@RH.T).T)
            spins_ctx[k,:] = np.concatenate([idxL, idxR])
        sio.savemat(spinfile, {"spins_ctx": spins_ctx+1})

    # ---------- Containers ----------
    RAW = {"cortex":{m: np.zeros((n_psy,n_sud)) for m in MEASURES}}
    Z = {"cortex":{m: np.zeros((n_psy,n_sud)) for m in MEASURES}}
    NULLS = {"cortex":{m: np.zeros((n_psy,n_sud,N_PERM)) for m in MEASURES}} if SAVE_NULLS else None

    # ---------- Cortex similarity ----------
    print("Computing cortex similarity...")
    for i in tqdm(range(n_psy), desc="PSY maps"):
        for j in range(n_sud):
            obs, nulls = vectorized_similarity(X_psy_cortex[:,i], X_sud_cortex[:,j], spins_ctx)
            for k,m in enumerate(MEASURES):
                RAW["cortex"][m][i,j] = obs[k]
                Z["cortex"][m][i,j] = (obs[k]-nulls[:,k].mean())/nulls[:,k].std()
                if SAVE_NULLS:
                    NULLS["cortex"][m][i,j,:] = nulls[:,k]

    # ---------- Subcortex similarity (unchanged) ----------
    subctx_idx = np.arange(N_CORTEX, min(X_psy.shape[0], X_sud.shape[0]))
    n_sub = len(subctx_idx)
    if n_sub>0:
        print("Computing subcortex similarity...")
        for i in tqdm(range(X_psy.shape[1]), desc="PSY subctx"):
            sub_psy = X_psy[subctx_idx,i]
            sub_sud = X_sud[subctx_idx,:]
            for j in range(sub_sud.shape[1]):
                sub_sud_vec = sub_sud[:,j]
                for m in MEASURES:
                    if m=="spearman":
                        obs = spearman_rankcorr_obs(sub_psy, sub_sud_vec)
                        _, r_dist = shuf_test(sub_psy, sub_sud_vec, n_rot=N_PERM, type="spearman", null_dist=True)
                    elif m=="cosine":
                        obs = np.dot(sub_psy, sub_sud_vec)/(np.linalg.norm(sub_psy)*np.linalg.norm(sub_sud_vec))
                        _, r_dist = shuf_test(sub_psy, sub_sud_vec, n_rot=N_PERM, type="pearson", null_dist=True)
                    else:  # euclidean
                        obs = -np.linalg.norm(sub_psy-sub_sud_vec)
                        _, r_dist = shuf_test(sub_psy, sub_sud_vec, n_rot=N_PERM, type="pearson", null_dist=True)
                    RAW.setdefault("subctx", {}).setdefault(m, np.zeros((X_psy.shape[1], sub_sud.shape[1])))[i,j] = obs
                    Z.setdefault("subctx", {}).setdefault(m, np.zeros((X_psy.shape[1], sub_sud.shape[1])))[i,j] = (obs-r_dist.mean())/r_dist.std()
                    if SAVE_NULLS:
                        NULLS.setdefault("subctx", {}).setdefault(m, np.zeros((X_psy.shape[1], sub_sud.shape[1], N_PERM*2)))[i,j,:] = r_dist

    # ---------- Save CSVs and ranking ----------
    for m in MEASURES:
        # Cortex
        pd.DataFrame(RAW["cortex"][m], index=psy_names, columns=sud_names).to_csv(os.path.join(outdir,f"RAW_cortex_{m}.csv"))
        pd.DataFrame(Z["cortex"][m], index=psy_names, columns=sud_names).to_csv(os.path.join(outdir,f"Z_cortex_{m}.csv"))
        if SAVE_NULLS:
            for j,sname in enumerate(sud_names):
                pd.DataFrame(NULLS["cortex"][m][:,j,:], index=psy_names).to_csv(os.path.join(outdir,f"NULLS_cortex_{m}_{sname}.csv"))

        # Subcortex (if present)
        if n_sub>0:
            pd.DataFrame(RAW["subctx"][m], index=psy_names, columns=sud_names).to_csv(os.path.join(outdir,f"RAW_subctx_{m}.csv"))
            pd.DataFrame(Z["subctx"][m], index=psy_names, columns=sud_names).to_csv(os.path.join(outdir,f"Z_subctx_{m}.csv"))
            if SAVE_NULLS:
                for j,sname in enumerate(sud_names):
                    pd.DataFrame(NULLS["subctx"][m][:,j,:], index=psy_names).to_csv(os.path.join(outdir,f"NULLS_subctx_{m}_{sname}.csv"))

        # Ranking per measure
        Zc = Z["cortex"][m]
        if n_sub>0:
            Ztot = (Zc + Z["subctx"][m])/np.sqrt(2)
        else:
            Ztot = Zc
        rank_idx = np.argsort(-Ztot, axis=0)
        for j,sname in enumerate(sud_names):
            ord_psy = np.array(psy_names)[rank_idx[:,j]]
            ord_z = Ztot[rank_idx[:,j],j]
            pd.DataFrame({"Psychiatric_Disorder": ord_psy, f"Z_{m}": ord_z}).to_csv(os.path.join(outdir,f"RANK_{m}_by_{sname}.csv"), index=False)
        meanZ = np.nanmean(Ztot, axis=1)
        mean_idx = np.argsort(-meanZ)
        pd.DataFrame({"Psychiatric_Disorder": np.array(psy_names)[mean_idx], f"Mean_Z_{m}_over_SUD": meanZ[mean_idx]}).to_csv(os.path.join(outdir,f"RANK_{m}_mean_across_SUD.csv"), index=False)

print("\n✅ All analyses done (cortex + subcortex if present, rankings included)")
