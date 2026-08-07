#!/usr/bin/env python3
"""
RQ1 — AIM1 step 1: SPECIFICITY OF THE PSY-SUD HIERARCHY
=======================================================

Question, design and caveats are unchanged from the previous version and are
reproduced at the bottom of this docstring. What changed is the plumbing.

WHAT CHANGED
------------
1. BRAINSMASH GEOMETRY CORRECTED — THIS MOVES NUMBERS. The old script did

       LH, RH = load_centroids(data_dir)      # <- returned UNIT-SPHERE centroids
       D = cdist(np.vstack([LH, RH]), ...)

   i.e. it fitted variograms against distances between points projected onto
   the unit sphere. Normalising centroids is not a uniform rescaling — regions
   sit at different distances from the origin — so the distance matrix was a
   distorted version of cortical geometry. Meanwhile the pairwise BrainSMASH
   script used RAW centroids. The two analyses in the same paper were using
   different brains.

   Now everything goes through C.distance_matrix(), which uses RAW centroids.
   The unit sphere is used only for spins, where a rotation requires it.
   EXPECT THE BRAINSMASH DELTA P-VALUES TO DIFFER from your previous run. The
   spin p-values and every observed quantity are unaffected.

2. SURROGATES CACHED. C.get_surrogates keys on a hash of the map, so the nine
   BrainSMASH runs happen once. Re-running with a different benchmark, a
   different FDR family or an extra sensitivity metric then costs seconds. If
   one disorder changes, only that disorder is regenerated.

3. SPINS SHARED. C.make_spins writes to ALL_outputs_RQ1/_cache, the same file
   the pairwise script uses, so AIM1 and the pairwise analysis are built on
   identical rotations. Previously AIM1 kept its own copy in a different
   folder.

4. CLUSTERS COME FROM RQ1_common. This script used to declare its own dict
   spelling the groups "Mood/Anxiety"/"Neurodevelopmental" while the
   supplementary-table script spelled them "Mood/Anx"/"Neurodev". Joining the
   two on cluster produced NaN. PTSD is now PD.

5. NO DUPLICATED STATISTICS. bh_fdr, the two-tailed p and the rank-correlation
   helper are imported, not redefined.

--- unchanged design notes -------------------------------------------------
For each adult psychiatric disorder i (n = 9):
    mSUD_i  = mean_j rho(PSY_i, SUD_j)     j = 6 substance-specific maps
    mPSY_i  = weighted mean over the other 8 PSY maps
    Delta_i = mSUD_i - mPSY_i              > 0 = preferential to SUD

Both nulls recompute BOTH means from the SAME surrogate, preserving the
dependency between the two targets, so the null is not centred on zero: it
absorbs baseline differences between the two target sets.

NOTE 1  SUD set = six substance maps only; the all-SUD column is dropped so
        neither side gains an aggregation advantage.
NOTE 2  The six substance maps share one control sample (n = 1951); the eight
        PSY maps do not. State this in the Methods.
NOTE 3  The two means run over 6 vs 8 targets; the null carries the same
        asymmetry.
NOTE 4  Parcel bootstrap ignores spatial autocorrelation -> mildly
        ANTI-conservative. Descriptive precision, not a second test.
NOTE 5  No pediatric Delta: 5 maps, two from one study, benchmark too unstable.
NOTE 6  Do not attach inference to r(mPSY, mSUD) across 9 points.
NOTE 7  TARGET-SET COHERENCE. The SUD side is internally uniform
        (within-set rho ~ .68), the PSY side heterogeneous (~ .15). Averaging
        over a heterogeneous set attenuates toward zero, so a positive Delta
        can come partly from coherence asymmetry. The spin null does not
        absorb this. `delta_proto` repeats the contrast with both sides as a
        single composite map; the gap between delta and delta_proto measures
        how much rides on coherence. Report both.

Just press Run.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import RQ1_common as C

# ================= CONFIG — edit, then press Run =================
NULL_MODE = "both"                    # "spin" = fast check | "both" = final numbers
BENCHMARKS = ["all", "outcluster"]    # "all" is primary; both are reported
N_PERM = 10000
N_BOOT = 10000
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ1", "specificity_adults")
CACHE = os.path.join(repo_dir, "ALL_outputs_RQ1", "_cache")
os.makedirs(OUTDIR, exist_ok=True)


def scalar_p(obs, null):
    """C.perm_p for a single (obs, null) pair, plus a genuine effect size."""
    null = np.asarray(null, float).ravel()
    null = null[np.isfinite(null)]
    p = float(C.perm_p(np.array([obs]), null[:, None])[0])
    sd = null.std(ddof=1)
    z = 0.0 if sd == 0 else float((obs - null.mean()) / sd)
    return z, p


def benchmark_weights(names, mode):
    """
    Row i = weights over the 9 PSY maps forming disorder i's benchmark.
    Rows sum to 1, diagonal 0.

      "all"        every other disorder equally (CONSERVATIVE: SCZ's benchmark
                   contains BD at rho .75 and CHR, the prodrome of the same
                   illness, so Delta is biased DOWN exactly for the disorders
                   at the top of the hierarchy).
      "outcluster" only disorders outside the own a priori cluster (LIBERAL:
                   the SUD side is itself one tight cluster, so trimming only
                   the PSY side biases Delta UP).
      "blocks"     equal weight per other cluster, then equal within it.

    "all" and "outcluster" bracket the answer; report both.
    """
    n = len(names)
    W = np.zeros((n, n))
    for i, ni in enumerate(names):
        if mode == "all":
            sel = [j for j in range(n) if j != i]
            W[i, sel] = 1.0 / len(sel)
        elif mode == "outcluster":
            sel = [j for j in range(n) if C.CLUSTERS[names[j]] != C.CLUSTERS[ni]]
            W[i, sel] = 1.0 / len(sel)
        elif mode == "blocks":
            blocks = {}
            for j, nj in enumerate(names):
                if j == i or C.CLUSTERS[nj] == C.CLUSTERS[ni]:
                    continue
                blocks.setdefault(C.CLUSTERS[nj], []).append(j)
            for b, mem in blocks.items():
                W[i, mem] = 1.0 / (len(blocks) * len(mem))
        else:
            raise ValueError(mode)
        assert abs(W[i].sum() - 1) < 1e-9 and W[i, i] == 0
    return W


def observed(P, S, names, sud_names, W):
    Rp, Rs = rankdata(P, axis=0), rankdata(S, axis=0)

    PP = C._corr_rows_cols(Rp.T, Rp)
    np.fill_diagonal(PP, np.nan)
    PP = pd.DataFrame(PP, index=names, columns=names)
    PS = pd.DataFrame(C._corr_rows_cols(Rp.T, Rs), index=names, columns=sud_names)

    m_psy = pd.Series((np.nan_to_num(PP.to_numpy()) * W).sum(axis=1), index=names)
    m_sud = PS.mean(axis=1)
    delta = m_sud - m_psy

    proto = []
    r_sud_proto = rankdata(S.mean(axis=1))
    for i in range(P.shape[1]):
        r_psy_proto = rankdata(P @ W[i])
        both = C._corr_rows_cols(Rp[:, i][None, :],
                                 np.column_stack([r_sud_proto, r_psy_proto]))[0]
        proto.append((both[0], both[1], both[0] - both[1]))
    proto = pd.DataFrame(proto, index=names,
                         columns=["rho_protoSUD", "rho_protoPSY", "delta_proto"])
    return PP, PS, m_psy, m_sud, delta, proto


def null_deltas(surr_ranks, Rp, w, Rs, r_sud_proto, r_psy_proto):
    """Both means recomputed from the SAME surrogate, preserving dependency."""
    m_sud = C._corr_rows_cols(surr_ranks, Rs).mean(axis=1)
    m_psy = C._corr_rows_cols(surr_ranks, Rp) @ w
    rp = C._corr_rows_cols(surr_ranks, np.column_stack([r_sud_proto, r_psy_proto]))
    return m_sud - m_psy, rp[:, 0] - rp[:, 1]


def run_null(kind, P, S, names, W, spins, D):
    Rp, Rs = rankdata(P, axis=0), rankdata(S, axis=0)
    out, out_proto = {}, {}
    for i, nm in enumerate(names):
        t0 = time.time()
        if kind == "spin":
            surr_ranks = Rp[:, i][spins]                      # rank(x[s]) == rank(x)[s]
        else:
            surr = C.get_surrogates(P[:, i], D, N_PERM, f"PSYadu_{nm}", CACHE)
            surr_ranks = rankdata(surr, axis=1)
        r_sud_proto = rankdata(S.mean(axis=1))
        r_psy_proto = rankdata(P @ W[i])
        out[nm], out_proto[nm] = null_deltas(surr_ranks, Rp, W[i], Rs,
                                             r_sud_proto, r_psy_proto)
        print(f"   {kind:<11s} {nm:>5s}  {time.time()-t0:5.1f}s", flush=True)
    return out, out_proto


def bootstrap_delta(P, S, names, W, n_boot, seed=C.SEED):
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, P.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, C.N_CORTEX, C.N_CORTEX)
        Rp, Rs = rankdata(P[idx, :], axis=0), rankdata(S[idx, :], axis=0)
        PP = C._corr_rows_cols(Rp.T, Rp)
        np.fill_diagonal(PP, 0.0)
        boot[b] = C._corr_rows_cols(Rp.T, Rs).mean(axis=1) - (PP * W).sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"delta_boot_lo": lo, "delta_boot_hi": hi,
                         "delta_boot_sd": boot.std(axis=0, ddof=1)}, index=names), boot


def main():
    psy, names = C.read_maps(os.path.join(data_dir, "PSY_adults.xlsx"), C.N_CORTEX)
    sud, sud_names = C.read_maps(os.path.join(data_dir, "SUD.xlsx"), C.N_CORTEX,
                                 drop_cols=["SUD"])          # NOTE 1
    C.check_cluster_coverage(names)
    assert len(sud_names) == 6, f"expected 6 substance maps, got {sud_names}"
    P, S = psy.to_numpy(float), sud.to_numpy(float)

    kinds = ["spin"] if NULL_MODE == "spin" else ["spin", "brainsmash"]
    spins = C.make_spins(data_dir, N_PERM, CACHE)
    D = C.distance_matrix(data_dir) if "brainsmash" in kinds else None   # RAW centroids

    for bench in BENCHMARKS:
        print("\n" + "=" * 70)
        print(f"BENCHMARK = {bench} | nulls = {NULL_MODE} | n_perm = {N_PERM}")
        print("=" * 70)
        W = benchmark_weights(names, bench)
        PP, PS, m_psy, m_sud, delta, proto = observed(P, S, names, sud_names, W)

        PP.to_csv(os.path.join(OUTDIR, "PSYxPSY_spearman_adults.csv"))
        PS.to_csv(os.path.join(OUTDIR, "PSYxSUD_spearman_adults_6substances.csv"))

        res = pd.DataFrame({
            "cluster": [C.CLUSTERS[n] for n in names],
            "mean_rho_PSY": m_psy.to_numpy(),
            "mean_rho_SUD": m_sud.to_numpy(),
            "delta": delta.to_numpy(),
        }, index=pd.Index(names, name="disorder")).join(proto)

        nulls, nulls_proto = {}, {}
        for kind in kinds:
            nulls[kind], nulls_proto[kind] = run_null(kind, P, S, names, W, spins, D)

        for tag, nd in nulls.items():
            stats = [scalar_p(res.loc[nm, "delta"], nd[nm]) for nm in names]
            pstats = [scalar_p(res.loc[nm, "delta_proto"], nulls_proto[tag][nm]) for nm in names]
            res[f"z_{tag}"] = [s[0] for s in stats]
            res[f"p_{tag}"] = [s[1] for s in stats]
            res[f"pFDR_{tag}"] = C.bh_fdr([s[1] for s in stats])
            res[f"null_lo_{tag}"] = [np.percentile(nd[nm], 2.5) for nm in names]
            res[f"null_hi_{tag}"] = [np.percentile(nd[nm], 97.5) for nm in names]
            res[f"z_proto_{tag}"] = [s[0] for s in pstats]
            res[f"p_proto_{tag}"] = [s[1] for s in pstats]
            res[f"pFDR_proto_{tag}"] = C.bh_fdr([s[1] for s in pstats])
            pd.DataFrame(nd).to_csv(
                os.path.join(OUTDIR, f"NULL_deltas_{tag}_{bench}.csv"), index=False)

        if len(nulls) == 2:
            res["robust_both_nulls"] = (res["pFDR_spin"] < .05) & (res["pFDR_brainsmash"] < .05)

        print("\n-- bootstrap over parcels --", flush=True)
        boot_ci, boot = bootstrap_delta(P, S, names, W, N_BOOT)
        res = res.join(boot_ci)
        pd.DataFrame(boot, columns=names).to_csv(
            os.path.join(OUTDIR, f"BOOT_deltas_{bench}.csv"), index=False)

        glob = {t: scalar_p(res["delta"].mean(), pd.DataFrame(nd).mean(axis=1).to_numpy())
                for t, nd in nulls.items()}
        g_ci = np.percentile(boot.mean(axis=1), [2.5, 97.5])

        res = res.sort_values("mean_rho_SUD", ascending=False)
        res.to_csv(os.path.join(OUTDIR, f"SPECIFICITY_delta_adults_{bench}.csv"))

        off = lambda M: M[~np.eye(M.shape[0], dtype=bool)].mean()
        with open(os.path.join(OUTDIR, f"README_specificity_{bench}.txt"), "w") as f:
            f.write(f"benchmark={bench}, nulls={NULL_MODE}, n_perm={N_PERM}, n_boot={N_BOOT}\n")
            f.write("SUD set = 6 substance-specific maps (all-SUD column dropped)\n")
            f.write("BrainSMASH distances from RAW centroids (was: unit-sphere; that was a bug)\n")
            f.write(f"global mean Delta = {res['delta'].mean():.4f}; "
                    + "; ".join(f"{t}: z={z:.2f} p={p:.4f}" for t, (z, p) in glob.items())
                    + f"; bootstrap 95% CI [{g_ci[0]:.3f}, {g_ci[1]:.3f}]\n")
            f.write(f"descriptive r(mean_rho_PSY, mean_rho_SUD) = "
                    f"{np.corrcoef(res['mean_rho_PSY'], res['mean_rho_SUD'])[0,1]:.3f} (n=9, no inference)\n")
            f.write(f"target-set coherence: SUD={off(np.corrcoef(rankdata(S, axis=0).T)):.3f}, "
                    f"PSY={off(np.corrcoef(rankdata(P, axis=0).T)):.3f}\n")

        pd.set_option("display.width", 220)
        cols = (["cluster", "mean_rho_PSY", "mean_rho_SUD", "delta",
                 "delta_boot_lo", "delta_boot_hi", "delta_proto"]
                + [c for c in res.columns if c.startswith("pFDR_") and "proto" not in c])
        print("\n" + res[cols].round(4).to_string())
        print(f"\nGLOBAL mean Delta = {res['delta'].mean():.3f}   "
              + "   ".join(f"[{t}] z={z:.2f} p={p:.4f}" for t, (z, p) in glob.items())
              + f"   bootstrap 95% CI [{g_ci[0]:.3f}, {g_ci[1]:.3f}]")

    print(f"\nDone. Outputs -> {OUTDIR}")


if __name__ == "__main__":
    main()