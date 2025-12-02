#!/usr/bin/env python3
"""
Replicated Python version of the MATLAB RQ1 similarity pipeline.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, norm
from tqdm import tqdm

# ---------------------------
# Determinism
# ---------------------------
np.random.seed(42)

# ---------------------------
# Paths (adjust if needed)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, 'data', 'raw')
main_outdir = os.path.join(repo_dir, 'ALL_outputs_RQ1')
os.makedirs(main_outdir, exist_ok=True)

# ---------------------------
# Load SUD data (common)
# ---------------------------
sud_file = os.path.join(data_dir, 'SUD.xlsx')
if not os.path.exists(sud_file):
    raise FileNotFoundError(f"SUD.xlsx not found at {sud_file}")
T_sud = pd.read_excel(sud_file)
X_sud = T_sud.select_dtypes(include=[np.number]).to_numpy()
sud_names = T_sud.columns.tolist()
nonnum_sud = T_sud.select_dtypes(exclude=[np.number]).columns.tolist()
region_names = T_sud[nonnum_sud[0]].astype(str).to_numpy() if nonnum_sud else np.array([f"R{i+1}" for i in range(T_sud.shape[0])])

# ---------------------------
# Groups (adjust names files as needed)
# ---------------------------
groups = [
    ('adults_all', 'PSY_adults.xlsx'),
    ('adolescents_all', 'PSY_adolescents.xlsx'),
    ('adults_ctx', 'PSY_adults_ctx.xlsx'),
    ('adolescents_ctx', 'PSY_adolescents_ctx.xlsx')
]

# ---------------------------
# Helpers
# ---------------------------
def rand_rotation_matrix():
    """Quaternion-based random rotation (same construction as MATLAB version)."""
    u1,u2,u3 = np.random.rand(3)
    q1 = np.sqrt(1-u1)*np.sin(2*np.pi*u2)
    q2 = np.sqrt(1-u1)*np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1)*np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1)*np.cos(2*np.pi*u3)
    R = np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q3*q4), 2*(q1*q3+q2*q4)],
        [2*(q1*q2+q3*q4), 1-2*(q1**2+q3**2), 2*(q2*q3-q1*q4)],
        [2*(q1*q3-q2*q4), 2*(q2*q3+q1*q4), 1-2*(q1**2+q2**2)]
    ])
    return R

def local_nn(A, B):
    """Nearest neighbor: for each row in B find argmin distance to rows in A.
       Equivalent to MATLAB local_nn implementation used to make spins."""
    # cdist returns distances B x A
    return np.argmin(cdist(B, A), axis=1)

def load_mat_variable(matfile, varname_guess):
    """Load a variable from a .mat file. If not found tries to return the only variable present."""
    mat = sio.loadmat(matfile)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if varname_guess in mat:
        return mat[varname_guess]
    elif len(keys) == 1:
        return mat[keys[0]]
    else:
        raise KeyError(f"Variable '{varname_guess}' not found in {matfile}. Available: {keys}")

def safe_perms_from_mat(perms, n_expected):
    """
    Ensure perms is shape (nperm, n_expected) and zero-based ints.
    Handles MATLAB 1-based indexing and transposed shapes.
    """
    perms = np.array(perms)
    # convert to 1D or 2D safely
    if perms.ndim == 1:
        perms = perms.reshape(1, -1)
    # If values look 1-based (>0 and min >=1) convert to zero-based:
    if perms.size > 0 and np.nanmin(perms) >= 1:
        perms = perms - 1
    # ensure integer dtype
    perms = perms.astype(int)
    # if perms are stored transposed (n_expected x nperm), transpose
    if perms.shape[1] != n_expected and perms.shape[0] == n_expected:
        perms = perms.T
    if perms.shape[1] != n_expected:
        raise ValueError(f"Permutations shape mismatch; expected second dim {n_expected}, got {perms.shape}")
    return perms

def make_subctx_centroids(n_sub):
    """Simple deterministic centroids for subcortex if needed."""
    rng = np.random.RandomState(42)
    return rng.rand(n_sub, 3)

def make_perms_from_centroids(centroids, nperm):
    """Create perms for subcortex via random rotation + NN (mirrors MATLAB approach used for cortex)."""
    n = centroids.shape[0]
    perms = np.zeros((nperm,n), dtype=int)
    for k in range(nperm):
        R = rand_rotation_matrix()
        rotated = (R @ centroids.T).T
        perms[k] = local_nn(centroids, rotated)
    return perms

def spearman_rankcorr_obs(X, Y):
    """Compute Spearman by ranking both vectors (average ties) and computing Pearson corr."""
    rx = rankdata(X, method='average')
    ry = rankdata(Y, method='average')
    # centralize
    mx = rx.mean()
    my = ry.mean()
    num = np.sum((rx - mx) * (ry - my))
    den = np.sqrt(np.sum((rx - mx)**2) * np.sum((ry - my)**2))
    if den == 0:
        return 0.0
    return num/den

def permutation_p_to_z(obs, nulls, tail='greater'):
    """Compute one-sided permutation p-value (counting nulls >= obs) and convert to z (upper-tail).
       tail='greater' means larger obs = more extreme (used for all three measures because we use -euclidean).
    """
    nperm = nulls.shape[0]
    if tail == 'greater':
        cnt = np.sum(nulls >= obs)
    elif tail == 'less':
        cnt = np.sum(nulls <= obs)
    else:
        raise ValueError('tail must be "greater" or "less"')
    p = (cnt + 1.0) / (nperm + 1.0)  # +1 correction
    # convert to z such that small p -> large positive z
    # z = norm.ppf(1 - p) (upper-tail)
    z = norm.ppf(1.0 - p)
    return p, z

def vectorized_similarity(X, Y, perms):
    """
    Vectorized computation of observed stats and null distributions.
    X: 1D array length n
    Y: 1D array length n
    perms: (nperm, n) integer index array such that X_perm[k] = X[perms[k]]
    Returns:
      obs: array([spearman, cosine, -euclidean])  (higher = more similar)
      nulls: (nperm, 3) same measures computed on permuted Xs
    """
    nperm = perms.shape[0]
    n = X.shape[0]
    obs = np.zeros(3)
    nulls = np.zeros((nperm, 3))

    # --- Observed ---
    # Make sure to compute spearman exactly via ranks -> Pearson
    obs[0] = spearman_rankcorr_obs(X, Y)
    # cosine
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    if normX == 0 or normY == 0:
        obs[1] = 0.0
    else:
        obs[1] = np.dot(X, Y) / (normX * normY)
    # euclidean: store negative distance so higher=more similar (consistent with your MATLAB)
    obs[2] = -np.linalg.norm(X - Y)

    # --- Permuted Xs ---
    # X_perm: shape (nperm, n)
    X_perm = X[perms]  # each row is the permuted X vector
    # Spearman nulls: rank each permuted X row and compute Pearson corr with rank(Y)
    # Rank Y once:
    rankY = rankdata(Y, method='average')
    meanY = rankY.mean()
    stdY = np.sqrt(np.sum((rankY - meanY)**2))
    # rank permuted Xs
    rank_X_perm = np.apply_along_axis(rankdata, 1, X_perm)  # shape (nperm, n)
    mean_X = np.mean(rank_X_perm, axis=1, keepdims=True)   # shape (nperm, 1)
    cov = np.sum((rank_X_perm - mean_X) * (rankY - meanY), axis=1)
    std_X = np.sqrt(np.sum((rank_X_perm - mean_X)**2, axis=1))
    # guard against zero division
    with np.errstate(divide='ignore', invalid='ignore'):
        spearman_nulls = np.where((std_X == 0) | (stdY == 0), 0.0, cov / (std_X * stdY))
    nulls[:, 0] = spearman_nulls

    # Cosine nulls
    X_norm = np.linalg.norm(X_perm, axis=1)
    Y_norm = normY
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_nulls = np.where((X_norm == 0) | (Y_norm == 0), 0.0, np.sum(X_perm * Y, axis=1) / (X_norm * Y_norm))
    nulls[:,1] = cos_nulls

    # Euclidean nulls (negative distances)
    nulls[:,2] = -np.linalg.norm(X_perm - Y, axis=1)

    return obs, nulls

# ---------------------------
# Main
# ---------------------------
nperm = 10000
n_cortex = 68

for group_label, psy_file_name in groups:
    print(f"\n== Running group: {group_label} ==")
    psy_file = os.path.join(data_dir, psy_file_name)
    if not os.path.exists(psy_file):
        raise FileNotFoundError(f"{psy_file_name} not found in {data_dir}")
    outdir = os.path.join(main_outdir, group_label)
    os.makedirs(outdir, exist_ok=True)

    # Load PSY
    T_psy = pd.read_excel(psy_file)
    X_psy = T_psy.select_dtypes(include=[np.number]).to_numpy()
    psy_names = T_psy.columns.tolist()
    
    cortex_idx = np.arange(min(n_cortex, X_psy.shape[0]))
    subctx_idx = np.arange(n_cortex, X_psy.shape[0]) if X_psy.shape[0] > n_cortex else np.array([], dtype=int)
    
    # --- Cortex spins: must exist (prefer MATLAB-generated file) ---
    spinfile = os.path.join(outdir,'spins_ctx_68.mat')
    if os.path.exists(spinfile):
        spins_ctx = load_mat_variable(spinfile, 'spins_ctx')
        spins_ctx = safe_perms_from_mat(spins_ctx, len(cortex_idx))
        print(f"Loaded cortex spins from {spinfile} (nperm={spins_ctx.shape[0]})")
    else:
        # If not present, try to construct from centroids (like MATLAB did)
        centsfile = os.path.join(repo_dir, 'centroids_ctx_68.mat')
        if os.path.exists(centsfile):
            C = sio.loadmat(centsfile)
            # Expect centroids_lh & centroids_rh
            LH = C.get('centroids_lh', None)
            RH = C.get('centroids_rh', None)
            if LH is None or RH is None:
                raise KeyError("centroids_ctx_68.mat must contain 'centroids_lh' and 'centroids_rh'")
            # normalize (MATLAB used normalization)
            LH = LH / np.linalg.norm(LH, axis=1, keepdims=True)
            RH = RH / np.linalg.norm(RH, axis=1, keepdims=True)
            spins = np.zeros((nperm, len(cortex_idx)), dtype=int)
            for k in range(nperm):
                R = rand_rotation_matrix()
                LH_rot = (R @ LH.T).T
                RH_rot = (R @ RH.T).T
                idxL = local_nn(LH, LH_rot)
                idxR = local_nn(RH, RH_rot)
                spins[k, :] = np.concatenate([idxL, idxR])
            spins_ctx = spins
            sio.savemat(spinfile, {'spins_ctx': spins_ctx + 1})  # save in MATLAB-friendly 1-based
            print("Constructed cortex spins and saved to disk.")
        else:
            raise FileNotFoundError("Cortex centroids required to compute spins (centroids_ctx_68.mat).")

    # --- Subcortex permutations (your choice) ---
    n_sub = len(subctx_idx)
    if n_sub > 0:
        permfile = os.path.join(outdir,'perms_subctx_14.mat')
        if os.path.exists(permfile):
            perms_subctx = load_mat_variable(permfile, 'perms_subctx')
            perms_subctx = safe_perms_from_mat(perms_subctx, n_sub)
            print("Loaded existing subcortex permutations")
        else:
            centroids_sub = make_subctx_centroids(n_sub)
            perms_subctx = make_perms_from_centroids(centroids_sub, nperm)
            sio.savemat(permfile, {'perms_subctx': perms_subctx + 1})  # save 1-based for MATLAB compatibility if desired
            print("Created subcortex permutations")
    else:
        perms_subctx = np.array([])

    # --- Similarity ---
    measures = ['spearman','cosine','euclidean','combined']
    num_psy, num_sud = X_psy.shape[1], X_sud.shape[1]
    Z = {'cortex':{m: np.zeros((num_psy,num_sud)) for m in measures},
         'subctx':{m: np.zeros((num_psy,num_sud)) for m in measures} if len(subctx_idx)>0 else {}}
    # Optionally keep p-values if you want to inspect them
    Pvals = {'cortex':{}, 'subctx':{}}
    print("Starting similarity computations...")
    for i in tqdm(range(num_psy), desc='PSY maps'):
        psy_vec = X_psy[:,i]
        for j in range(num_sud):
            sud_vec = X_sud[:,j]
            # Cortex
            obs_c, nulls_c = vectorized_similarity(psy_vec[cortex_idx], sud_vec[cortex_idx], spins_ctx)
            # For each of the three measures compute permutation p and z (one-sided greater)
            for k,m in enumerate(['spearman','cosine','euclidean']):
                p_c, z_c = permutation_p_to_z(obs_c[k], nulls_c[:,k], tail='greater')
                Z['cortex'][m][i,j] = z_c
                Pvals['cortex'].setdefault(m, np.full((num_psy,num_sud), np.nan))[i,j] = p_c
            # combined: Stouffer-like: sum z / sqrt(k)
            combined_c = (Z['cortex']['spearman'][i,j] + Z['cortex']['cosine'][i,j] + Z['cortex']['euclidean'][i,j]) / np.sqrt(3)
            Z['cortex']['combined'][i,j] = combined_c

            # Subcortex
            if len(subctx_idx)>0 and perms_subctx.size>0:
                obs_s, nulls_s = vectorized_similarity(psy_vec[subctx_idx], sud_vec[subctx_idx], perms_subctx)
                for k,m in enumerate(['spearman','cosine','euclidean']):
                    p_s, z_s = permutation_p_to_z(obs_s[k], nulls_s[:,k], tail='greater')
                    Z['subctx'][m][i,j] = z_s
                    Pvals['subctx'].setdefault(m, np.full((num_psy,num_sud), np.nan))[i,j] = p_s
                combined_s = (Z['subctx']['spearman'][i,j] + Z['subctx']['cosine'][i,j] + Z['subctx']['euclidean'][i,j]) / np.sqrt(3)
                Z['subctx']['combined'][i,j] = combined_s

    # --- Save tables ---
    for m in measures:
        # cortex
        df_c = pd.DataFrame(Z['cortex'][m], index=psy_names, columns=sud_names)
        df_c.to_csv(os.path.join(outdir,f'Z_cortex_{m}.csv'))
        # subctx if present
        if len(subctx_idx)>0:
            df_s = pd.DataFrame(Z['subctx'][m], index=psy_names, columns=sud_names)
            df_s.to_csv(os.path.join(outdir,f'Z_subctx_{m}.csv'))
    # Also save p-value files for the three measures if useful
    for m in ['spearman','cosine','euclidean']:
        df_p = pd.DataFrame(Pvals['cortex'][m], index=psy_names, columns=sud_names)
        df_p.to_csv(os.path.join(outdir,f'P_cortex_{m}.csv'))
        if len(subctx_idx)>0:
            df_ps = pd.DataFrame(Pvals['subctx'][m], index=psy_names, columns=sud_names)
            df_ps.to_csv(os.path.join(outdir,f'P_subctx_{m}.csv'))

    # --- Ranking ---
    Zc = Z['cortex']['combined']
    if len(subctx_idx)>0:
        Ztot = (Zc + Z['subctx']['combined'])/np.sqrt(2)
    else:
        Ztot = Zc
    rank_idx = np.argsort(-Ztot, axis=0)
    for j, sname in enumerate(sud_names):
        ord_psy = np.array(psy_names)[rank_idx[:,j]]
        ord_z = Ztot[rank_idx[:,j],j]
        df_rank = pd.DataFrame({'Psychiatric_Disorder': ord_psy, 'Z_Stouffer': ord_z})
        df_rank.to_csv(os.path.join(outdir,f'RANK_overall_by_{sname}.csv'), index=False)
    Zmean_sorted_idx = np.argsort(-np.nanmean(Ztot, axis=1))
    df_mean = pd.DataFrame({'Psychiatric_Disorder': np.array(psy_names)[Zmean_sorted_idx],
                            'Mean_Z_over_SUD': np.nanmean(Ztot, axis=1)[Zmean_sorted_idx]})
    df_mean.to_csv(os.path.join(outdir,'RANK_overall_mean_across_SUD.csv'), index=False)

print("\n✅ All analyses done.")
