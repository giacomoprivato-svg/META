"""
Replicated Python version of the MATLAB RQ1 similarity pipeline.

Key update (per Matthias/Amin):
- We compute permutation-based p-values for each similarity metric.
- We compute a permutation-based p-value for the *combined* metric by constructing the null distribution
  of the combined z from per-permutation z-transforms (accounts for dependence among measures).
- We also output p-values (useful for p<0.05 masking, FDR, etc.).
"""

import os
import shutil
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from scipy import stats
from tqdm import tqdm

# ---------------------------
# Determinism
# ---------------------------
np.random.seed(42)

# ---------------------------
# Paths
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")

OLD_OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ1")
main_outdir = os.path.join(repo_dir, "ALL_outputs_RQ1_permCombined")
os.makedirs(main_outdir, exist_ok=True)

# ---------------------------
# Groups
# ---------------------------
groups = [
    ("adults_all", "PSY_adults.xlsx"),
    ("adolescents_all", "PSY_adolescents.xlsx"),
    ("adults_ctx", "PSY_adults_ctx.xlsx"),
    ("adolescents_ctx", "PSY_adolescents_ctx.xlsx"),
]

# ---------------------------
# NiMARE-like transforms
# ---------------------------
def z_to_p(z, tail="two"):
    z = np.array(z)
    if tail == "two":
        p = stats.norm.sf(np.abs(z)) * 2
    elif tail == "one":
        p = stats.norm.sf(z)
    else:
        raise ValueError
    return p.item() if p.shape == () else p


def p_to_z(p, tail="two"):
    p = np.array(p)
    if tail == "two":
        z = stats.norm.isf(p / 2)
    elif tail == "one":
        z = stats.norm.isf(p)
        z = np.maximum(z, 0)
    else:
        raise ValueError
    return z.item() if z.shape == () else z


# ---------------------------
# Helpers
# ---------------------------
def rand_rotation_matrix():
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([
        [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q3*q4),     2*(q1*q3 + q2*q4)],
        [2*(q1*q2 + q3*q4),     1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q1*q4)],
        [2*(q1*q3 - q2*q4),     2*(q2*q3 + q1*q4),     1 - 2*(q1*q1 + q2*q2)]
    ])


def local_nn(A, B):
    return np.argmin(cdist(B, A), axis=1)


def load_mat_variable(matfile, varname_guess):
    mat = sio.loadmat(matfile)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if varname_guess in mat:
        return mat[varname_guess]
    if len(keys) == 1:
        return mat[keys[0]]
    raise KeyError(f"{varname_guess} not found in {matfile}")


def safe_perms_from_mat(perms, n_expected):
    perms = np.array(perms)
    if perms.ndim == 1:
        perms = perms.reshape(1, -1)
    if np.nanmin(perms) >= 1:
        perms = perms - 1
    perms = perms.astype(int)
    if perms.shape[1] != n_expected and perms.shape[0] == n_expected:
        perms = perms.T
    if perms.shape[1] != n_expected:
        raise ValueError("Permutation shape mismatch")
    return perms


def make_subctx_centroids(n):
    rng = np.random.RandomState(42)
    return rng.rand(n, 3)


def make_perms_from_centroids(centroids, nperm):
    n = centroids.shape[0]
    perms = np.zeros((nperm, n), dtype=int)
    for k in range(nperm):
        R = rand_rotation_matrix()
        rot = (R @ centroids.T).T
        perms[k] = local_nn(centroids, rot)
    return perms


def spearman_rankcorr_obs(X, Y):
    rx = rankdata(X)
    ry = rankdata(Y)
    return np.corrcoef(rx, ry)[0, 1]

def perm_p_and_z_with_znull(obs, nulls, tail="greater"):
    """
    Permutation p-value (one-sided) + z (one-sided, unsigned) AND null z distribution.
    """
    nulls = np.asarray(nulls)
    nperm = nulls.shape[0]

    if tail == "greater":
        cnt = np.sum(nulls >= obs)
        r = stats.rankdata(-nulls, method="average")
        p_null = r / (nperm + 1.0)
    elif tail == "less":
        cnt = np.sum(nulls <= obs)
        r = stats.rankdata(nulls, method="average")
        p_null = r / (nperm + 1.0)
    else:
        raise ValueError('tail must be "greater" or "less"')

    p_obs = (cnt + 1.0) / (nperm + 1.0)
    z_obs = p_to_z(p_obs, tail="one")
    z_null = p_to_z(p_null, tail="one")

    return float(p_obs), float(z_obs), np.asarray(z_null)


def vectorized_similarity(X, Y, perms):
    """
    Compute observed similarity stats and permutation nulls.
    Returns:
      obs   : (3,)  [spearman, cosine, -euclidean]
      nulls : (nperm, 3)
    """
    nperm = perms.shape[0]
    obs = np.zeros(3)
    nulls = np.zeros((nperm, 3))

    # --- Observed ---
    obs[0] = spearman_rankcorr_obs(X, Y)

    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    obs[1] = 0.0 if normX == 0 or normY == 0 else np.dot(X, Y) / (normX * normY)
    obs[2] = -np.linalg.norm(X - Y)

    # --- Permutations ---
    Xp = X[perms]

    # Spearman
    rankY = rankdata(Y)
    meanY = rankY.mean()
    stdY = np.sqrt(np.sum((rankY - meanY) ** 2))

    rankXp = np.apply_along_axis(rankdata, 1, Xp)
    meanX = rankXp.mean(axis=1, keepdims=True)
    cov = np.sum((rankXp - meanX) * (rankY - meanY), axis=1)
    stdX = np.sqrt(np.sum((rankXp - meanX) ** 2, axis=1))

    with np.errstate(divide="ignore", invalid="ignore"):
        nulls[:, 0] = np.where((stdX == 0) | (stdY == 0), 0.0, cov / (stdX * stdY))

    # Cosine
    Xn = np.linalg.norm(Xp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nulls[:, 1] = np.where((Xn == 0) | (normY == 0), 0.0, np.sum(Xp * Y, axis=1) / (Xn * normY))

    # Euclidean
    nulls[:, 2] = -np.linalg.norm(Xp - Y, axis=1)

    return obs, nulls

# ---------------------------
# Data loading
# ---------------------------
T_sud = pd.read_excel(os.path.join(data_dir, "SUD.xlsx"))
X_sud = T_sud.select_dtypes(include=[np.number]).to_numpy()
sud_names = T_sud.columns.tolist()

region_names = (
    T_sud.select_dtypes(exclude=[np.number]).iloc[:, 0].astype(str).to_numpy()
    if not T_sud.select_dtypes(exclude=[np.number]).empty
    else np.arange(T_sud.shape[0])
)

# ---------------------------
# Main
# ---------------------------
nperm = 10000
n_cortex = 68
measures = ["spearman", "cosine", "euclidean", "combined"]

for group_label, psy_file_name in groups:
    print(f"\n== Running group: {group_label} ==")

    outdir = os.path.join(main_outdir, group_label)
    os.makedirs(outdir, exist_ok=True)

    # ---------------------------
    # Load PSY
    # ---------------------------
    T_psy = pd.read_excel(os.path.join(data_dir, psy_file_name))
    X_psy = T_psy.select_dtypes(include=[np.number]).to_numpy()
    psy_names = T_psy.columns.tolist()

    cortex_idx = np.arange(min(n_cortex, X_psy.shape[0]))
    subctx_idx = np.arange(n_cortex, X_psy.shape[0])

    # ---------------------------
    # Cortex spins (REUSED ONLY)
    # ---------------------------
    spinfile = os.path.join(outdir, "spins_ctx_68.mat")
    old_spinfile = os.path.join(OLD_OUTDIR, group_label, "spins_ctx_68.mat")

    if not os.path.exists(spinfile):
        if not os.path.exists(old_spinfile):
            raise FileNotFoundError(
                f"Missing cortex spins. Expected at:\n{spinfile}\nor\n{old_spinfile}"
            )
        shutil.copy2(old_spinfile, spinfile)
        print("Copied cortex spins from old outputs")

    spins_ctx = load_mat_variable(spinfile, "spins_ctx")
    spins_ctx = safe_perms_from_mat(spins_ctx, len(cortex_idx))
    print(f"Loaded cortex spins (nperm={spins_ctx.shape[0]})")

    # ---------------------------
    # Subcortex perms (unchanged)
    # ---------------------------
    n_sub = len(subctx_idx)
    if n_sub > 0:
        permfile = os.path.join(outdir, "perms_subctx_14.mat")
        if os.path.exists(permfile):
            perms_subctx = safe_perms_from_mat(
                load_mat_variable(permfile, "perms_subctx"), n_sub
            )
        else:
            centroids = make_subctx_centroids(n_sub)
            perms_subctx = make_perms_from_centroids(centroids, nperm)
            sio.savemat(permfile, {"perms_subctx": perms_subctx + 1})
    else:
        perms_subctx = np.array([])

    # --- Outputs ---
    num_psy, num_sud = X_psy.shape[1], X_sud.shape[1]

    Z = {
        "cortex": {m: np.zeros((num_psy, num_sud), dtype=float) for m in measures},
        "subctx": {m: np.zeros((num_psy, num_sud), dtype=float) for m in measures} if n_sub > 0 else {},
    }
    Pvals = {
        "cortex": {m: np.full((num_psy, num_sud), np.nan, dtype=float) for m in measures},
        "subctx": {m: np.full((num_psy, num_sud), np.nan, dtype=float) for m in measures} if n_sub > 0 else {},
    }

    print("Starting similarity computations...")
    for i in tqdm(range(num_psy), desc="PSY maps"):
        psy_vec = X_psy[:, i]

        for j in range(num_sud):
            sud_vec = X_sud[:, j]

            # =========================
            # Cortex
            # =========================
            obs_c, nulls_c = vectorized_similarity(psy_vec[cortex_idx], sud_vec[cortex_idx], spins_ctx)

            znull_c = {}
            for k, m in enumerate(["spearman", "cosine", "euclidean"]):
                p_obs, z_obs, z_null = perm_p_and_z_with_znull(obs_c[k], nulls_c[:, k], tail="greater")
                Pvals["cortex"][m][i, j] = p_obs
                Z["cortex"][m][i, j] = z_obs
                znull_c[m] = z_null

            # Combined (permutation-calibrated)
            z_comb_obs = (Z["cortex"]["spearman"][i, j] + Z["cortex"]["cosine"][i, j] + Z["cortex"]["euclidean"][i, j]) / np.sqrt(3)
            z_comb_null = (znull_c["spearman"] + znull_c["cosine"] + znull_c["euclidean"]) / np.sqrt(3)

            p_comb = (np.sum(z_comb_null >= z_comb_obs) + 1.0) / (len(z_comb_null) + 1.0)
            z_comb = p_to_z(p_comb, tail="one")

            Pvals["cortex"]["combined"][i, j] = p_comb
            Z["cortex"]["combined"][i, j] = z_comb

            # =========================
            # Subcortex
            # =========================
            if n_sub > 0 and perms_subctx.size > 0:
                obs_s, nulls_s = vectorized_similarity(psy_vec[subctx_idx], sud_vec[subctx_idx], perms_subctx)

                znull_s = {}
                for k, m in enumerate(["spearman", "cosine", "euclidean"]):
                    p_obs, z_obs, z_null = perm_p_and_z_with_znull(obs_s[k], nulls_s[:, k], tail="greater")
                    Pvals["subctx"][m][i, j] = p_obs
                    Z["subctx"][m][i, j] = z_obs
                    znull_s[m] = z_null

                z_comb_obs_s = (Z["subctx"]["spearman"][i, j] + Z["subctx"]["cosine"][i, j] + Z["subctx"]["euclidean"][i, j]) / np.sqrt(3)
                z_comb_null_s = (znull_s["spearman"] + znull_s["cosine"] + znull_s["euclidean"]) / np.sqrt(3)

                p_comb_s = (np.sum(z_comb_null_s >= z_comb_obs_s) + 1.0) / (len(z_comb_null_s) + 1.0)
                z_comb_s = p_to_z(p_comb_s, tail="one")

                Pvals["subctx"]["combined"][i, j] = p_comb_s
                Z["subctx"]["combined"][i, j] = z_comb_s

    # --- Save Z tables ---
    for m in measures:
        df_c = pd.DataFrame(Z["cortex"][m], index=psy_names, columns=sud_names)
        df_c.to_csv(os.path.join(outdir, f"Z_cortex_{m}.csv"))

        if n_sub > 0:
            df_s = pd.DataFrame(Z["subctx"][m], index=psy_names, columns=sud_names)
            df_s.to_csv(os.path.join(outdir, f"Z_subctx_{m}.csv"))

    # --- Save P tables ---
    for m in measures:
        df_pc = pd.DataFrame(Pvals["cortex"][m], index=psy_names, columns=sud_names)
        df_pc.to_csv(os.path.join(outdir, f"P_cortex_{m}.csv"))

        if n_sub > 0:
            df_ps = pd.DataFrame(Pvals["subctx"][m], index=psy_names, columns=sud_names)
            df_ps.to_csv(os.path.join(outdir, f"P_subctx_{m}.csv"))

    # --- Ranking (overall combined; cortex+subctx Stouffer across compartments) ---
    Zc = Z["cortex"]["combined"]
    if n_sub > 0:
        Ztot = (Zc + Z["subctx"]["combined"]) / np.sqrt(2)
    else:
        Ztot = Zc

    rank_idx = np.argsort(-Ztot, axis=0)

    for j, sname in enumerate(sud_names):
        ord_psy = np.array(psy_names)[rank_idx[:, j]]
        ord_z = Ztot[rank_idx[:, j], j]
        df_rank = pd.DataFrame({"Psychiatric_Disorder": ord_psy, "Z_Combined": ord_z})
        df_rank.to_csv(os.path.join(outdir, f"RANK_overall_by_{sname}.csv"), index=False)

    Zmean_sorted_idx = np.argsort(-np.nanmean(Ztot, axis=1))
    df_mean = pd.DataFrame(
        {
            "Psychiatric_Disorder": np.array(psy_names)[Zmean_sorted_idx],
            "Mean_Z_over_SUD": np.nanmean(Ztot, axis=1)[Zmean_sorted_idx],
        }
    )
    df_mean.to_csv(os.path.join(outdir, "RANK_overall_mean_across_SUD.csv"), index=False)

print("\n✅ All analyses done.")