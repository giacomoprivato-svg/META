#!/usr/bin/env python3
"""
RQ2 — step 1: shared PSY-SUD alteration maps
============================================

Replaces RQ2_shared_maps_pspin.py, RQ2_shared_maps_brainsmash.py and
RQ2_shared_without_AN.py. Surface rendering moves to RQ2_surfaces.py; this
script produces the numbers and the map CSVs only, so it runs anywhere the
ENIGMA Toolbox is not installed.

WHAT IT DOES
------------
For adults, the pediatric sample and each clinical cluster:
  * r(PSY mean, SUD mean) with spin AND BrainSMASH nulls
  * the shared map (PSY mean + SUD mean)/2, cortex and subcortex
  * the PSY contribution index per region, |PSY| / (|PSY| + |SUD|)

Then one systematic sensitivity, replacing the old AN-only script.

WHAT CHANGED
------------
1. HEMISPHERE FIX AND SUD FIX BOTH INHERITED, NOT RETYPED. Spins come from
   RQ1_common. The old pspin script had the fix but as a local copy, and the
   BrainSMASH twin had neither the fix nor the SUD drop, so the two nulls for
   the same analysis were computed on different data. They now share one
   loader and one geometry.

2. SYSTEMATIC LEAVE-ONE-OUT, NOT JUST AN. RQ2_shared_without_AN.py existed
   because AN's effect sizes are ~3x the others and dominate the unweighted
   PSY mean. Singling out AN answers only the question you already suspected.
   The LOO here drops every psychiatric map in turn AND every substance map in
   turn, and reports two different things per drop:
       delta_r         how much r(PSY mean, SUD mean) moves
       r_map_vs_full   how much the SHARED MAP itself moves
   These can disagree: a map can barely shift the correlation while noticeably
   changing the shared map, or the reverse. AN appears in the table like
   everything else, and if it is the outlier that is now a result rather than
   an assumption.

3. THE CIRCULARITY DIAGNOSTIC IS FIXED — IT WAS MEASURING THE WRONG THING.
   Reporting r(shared, SUD mean) as evidence that the shared map is "mostly
   SUD" is not a diagnostic at all: for a mean of two maps it is determined
   entirely by their mutual correlation r and their sd ratio k,

       r(A+B, B) = (k*r + 1) / sqrt(k^2 + 2*k*r + 1)

   With the observed r = .720 and k = .680 that formula returns .953, which is
   exactly the observed value. Two maps correlating at .72 ALWAYS produce a
   mean correlating ~.9 with each of them. The script now prints the observed
   value next to the arithmetic prediction, so a deviation from prediction is
   visible and a match is recognised as uninformative. The quantity that
   actually carries information is the per-gradient comparison of the two
   components, which is step 2's job.

Just press Run.
"""

import os
import numpy as np
import pandas as pd

import RQ1_common as C
import RQ2_common as R2

# ================= CONFIG — edit, then press Run =================
NULLS = ["spin", "brainsmash"]   # ["spin"] alone for a fast check
N_PERM = 10000
RUN_LOO = True
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ2", "shared_maps")
CACHE = os.path.join(repo_dir, "ALL_outputs_RQ1", "_cache")   # shared with RQ1
os.makedirs(OUTDIR, exist_ok=True)


def subcortical(fname, drop=()):
    T, cols = C.subcortical_rows(os.path.join(data_dir, fname), drop_cols=drop)
    return T, cols


def arithmetic_prediction(A, B):
    """r(mean(A,B), B) and r(mean(A,B), A) implied by r(A,B) and sd(A)/sd(B)."""
    r = float(np.corrcoef(A, B)[0, 1])
    k = A.std() / B.std()
    d = np.sqrt(k ** 2 + 2 * k * r + 1)
    return (k * r + 1) / d, (k + r) / d, r, k


def test_pair(psy_vec, sud_vec, nulls_arrays):
    """r(PSY, SUD) with the PSY map permuted, under each available null."""
    obs = R2.corr(psy_vec, sud_vec)
    out = {"r": obs}
    for tag, surr in nulls_arrays.items():
        z, p = R2.z_p(obs, R2.corr_rows(surr, sud_vec))
        out[f"z_{tag}"], out[f"p_{tag}"] = z, p
    return out


def main():
    maps = R2.load_maps(data_dir)
    agg = R2.aggregates(maps)
    psy, sud = maps["psy"], maps["sud"]

    sud_sub, _ = subcortical("SUD.xlsx", drop=["SUD"])
    psy_sub, _ = subcortical("PSY_adults.xlsx")

    # ---------------- groups ----------------
    groups = {"Adults": agg["PSY mean"]}
    if "Pediatric" in agg:
        groups["Pediatric"] = agg["Pediatric"]
    for cl in C.CLUSTER_MEMBERS:
        groups[cl] = agg[cl]
    sud_mean = agg["SUD mean"]

    # ---------------- nulls (surrogates of each PSY group map) ----------------
    spins = C.make_spins(data_dir, N_PERM, CACHE) if "spin" in NULLS else None
    D = C.distance_matrix(data_dir) if "brainsmash" in NULLS else None

    rows, shared_cols, contrib_cols = [], {}, {}
    for name, pm in groups.items():
        na = {}
        if spins is not None:
            na["spin"] = pm[spins]
        if D is not None:
            na["brainsmash"] = C.get_surrogates(pm, D, N_PERM, f"shared_{name}", CACHE)
        rec = {"group": name}
        rec.update(test_pair(pm, sud_mean, na))
        pb, pa, r_ab, k = arithmetic_prediction(pm, sud_mean)
        rec.update({"r_shared_vs_SUD": R2.corr((pm + sud_mean) / 2, sud_mean),
                    "r_shared_vs_SUD_predicted": pb,
                    "r_shared_vs_PSY": R2.corr((pm + sud_mean) / 2, pm),
                    "r_shared_vs_PSY_predicted": pa,
                    "sd_ratio_PSY_over_SUD": k})
        rows.append(rec)
        shared_cols[name] = (pm + sud_mean) / 2
        contrib_cols[name] = np.abs(pm) / (np.abs(pm) + np.abs(sud_mean)) * 100

    res = pd.DataFrame(rows).set_index("group")
    for tag in NULLS:
        if f"p_{tag}" in res.columns:
            res[f"pFDR_{tag}"] = C.bh_fdr(res[f"p_{tag}"].values)
    if len(NULLS) == 2 and {"pFDR_spin", "pFDR_brainsmash"} <= set(res.columns):
        res["robust_both_nulls"] = (res.pFDR_spin < .05) & (res.pFDR_brainsmash < .05)
    res.to_csv(os.path.join(OUTDIR, "SHARED_psy_sud_correspondence.csv"))

    pd.DataFrame(shared_cols).to_csv(
        os.path.join(OUTDIR, "SHARED_maps_cortex.csv"), index=False)
    pd.DataFrame(contrib_cols).to_csv(
        os.path.join(OUTDIR, "SHARED_psy_contribution_index_cortex.csv"), index=False)

    # subcortex: adults only, descriptive (no spatial null for 14 structures)
    sub_shared = (psy_sub.mean(axis=1) + sud_sub.mean(axis=1)) / 2
    pd.DataFrame({"Adults": sub_shared}).to_csv(
        os.path.join(OUTDIR, "SHARED_maps_subcortex.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\n=== r(PSY mean, SUD mean) per group ===")
    show = [c for c in ("r", "p_spin", "pFDR_spin", "p_brainsmash",
                        "pFDR_brainsmash", "robust_both_nulls") if c in res.columns]
    print(res[show].round(4).to_string())

    print("\n=== shared-map composition: observed vs arithmetic prediction ===")
    print("    A mean of two maps correlating at r with sd ratio k has a FIXED")
    print("    correlation with each component. Match = uninformative.")
    print(res[["r", "sd_ratio_PSY_over_SUD", "r_shared_vs_SUD",
               "r_shared_vs_SUD_predicted", "r_shared_vs_PSY",
               "r_shared_vs_PSY_predicted"]].round(3).to_string())

    ci = contrib_cols["Adults"]
    print(f"\nPSY contribution index (adults): median {np.median(ci):.1f}%  "
          f"IQR {np.percentile(ci,25):.1f}-{np.percentile(ci,75):.1f}  "
          f"range {ci.min():.1f}-{ci.max():.1f}")

    # ---------------- systematic leave-one-out ----------------
    if RUN_LOO:
        P, S = psy.to_numpy(float), sud.to_numpy(float)
        full_psy, full_sud = P.mean(1), S.mean(1)
        r_full = R2.corr(full_psy, full_sud)
        shared_full = (full_psy + full_sud) / 2

        loo = []
        for side, M, names, other in (("PSY", P, list(psy.columns), full_sud),
                                      ("SUD", S, list(sud.columns), full_psy)):
            for j, nm in enumerate(names):
                red = np.delete(M, j, axis=1).mean(1)
                r = R2.corr(red, other) if side == "PSY" else R2.corr(other, red)
                sh = (red + other) / 2 if side == "PSY" else (other + red) / 2
                loo.append({"side": side, "dropped": nm, "r_LOO": r,
                            "delta_r": r - r_full,
                            "r_map_vs_full": R2.corr(sh, shared_full)})
        loo = pd.DataFrame(loo).sort_values("r_map_vs_full")
        loo["r_full"] = r_full
        loo.to_csv(os.path.join(OUTDIR, "SHARED_leave_one_out.csv"), index=False)

        print(f"\n=== systematic leave-one-out (full r = {r_full:.3f}) ===")
        print(loo.round(4).to_string(index=False))
        worst_r = loo.loc[loo.delta_r.abs().idxmax()]
        worst_m = loo.loc[loo.r_map_vs_full.idxmin()]
        print(f"\n  largest shift in the CORRELATION : {worst_r.dropped} "
              f"({worst_r.delta_r:+.3f})")
        print(f"  largest shift in the SHARED MAP  : {worst_m.dropped} "
              f"(r with full map {worst_m.r_map_vs_full:.3f})")
        print("  These need not be the same map; report both.")

    print(f"\nDone. Outputs -> {OUTDIR}")


if __name__ == "__main__":
    main()