#!/usr/bin/env python3
"""
RQ1 — step 1: cortical PSY x SUD similarity, both nulls, one script

"""

import os
import time
import numpy as np
import pandas as pd

import RQ1_common as C

# ================= CONFIG — edit, then press Run =================
N_PERM = 10000
NULLS = ["spin", "brainsmash"]   # ["spin"] alone for a fast check (~1 min)
SAVE_NULLS = False               # True writes 10000 values per pair per metric;
                                 # as .npz, not the CSV dump the old script made
GROUPS = [
    ("adults_all",      "PSY_adults.xlsx"),
    ("adolescents_all", "PSY_adolescents.xlsx"),
    ("adults_ctx",      "PSY_adults_ctx.xlsx"),
    ("adolescents_ctx", "PSY_adolescents_ctx.xlsx"),
]
SUD_FILE = "SUD.xlsx"

# --- the aggregate all-SUD column: display vs statistics (see docstring) ---
DROP_SUD_AGGREGATE = False       # False = RAW_cortex_* keeps the 7-column panel
                                 #         (Fig. 1C, Table S5: shown for completeness)
                                 # True  = the aggregate never loads at all
AGGREGATE_IN_STATS = False       # False = no p-value, no FDR, no mean uses it
                                 # True  = reproduces the old inflated numbers
AGGREGATE_IN_PVALUES = True      # True = the aggregate column keeps its own p and FDR.
                                 # Harmless: the BH family is one column, so it neither
                                 # borrows nor lends significance to the other six.
                                 # This is what Figure 1 reads.
AGGREGATE_COL = "SUD"            # column name of the transdiagnostic map
N_SUBSTANCE_MAPS = 6             # ALC ATS CAN COC NIC OPI
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUT = os.path.join(repo_dir, "ALL_outputs_RQ1")
CACHE = os.path.join(OUT, "_cache")
os.makedirs(CACHE, exist_ok=True)


def statistics_columns(sud_names):
    """
    Indices of the SUD columns that are allowed into inference.

    Returns (indices, names). With AGGREGATE_IN_STATS False this is the six
    substance-specific maps; the aggregate keeps its place in RAW but is
    excluded here. Raises rather than silently proceeding if the result is not
    the expected panel — a renamed column would otherwise slip back in.
    """
    if AGGREGATE_IN_STATS:
        return list(range(len(sud_names))), list(sud_names)

    target = AGGREGATE_COL.strip().upper()
    idx = [k for k, s in enumerate(sud_names) if str(s).strip().upper() != target]
    names = [sud_names[k] for k in idx]

    if len(names) == len(sud_names) and not DROP_SUD_AGGREGATE:
        raise RuntimeError(
            f"AGGREGATE_IN_STATS is False but no column named {AGGREGATE_COL!r} was "
            f"found in {sud_names}. Fix AGGREGATE_COL — do not switch the flag.")
    if len(names) != N_SUBSTANCE_MAPS:
        raise RuntimeError(
            f"expected {N_SUBSTANCE_MAPS} substance-specific maps for the statistics, "
            f"got {len(names)}: {names}")
    return idx, names


def main():
    drop = [AGGREGATE_COL] if DROP_SUD_AGGREGATE else []
    sud, sud_names = C.read_maps(os.path.join(data_dir, SUD_FILE), C.N_CORTEX, drop_cols=drop)
    Y = sud.to_numpy(float)
    stat_cols, stat_names = statistics_columns(sud_names)

    print(f"SUD panel loaded ({len(sud_names)}): {sud_names}")
    print(f"  RAW matrices written for all {len(sud_names)} columns")
    print(f"  p-values, FDR and means restricted to {len(stat_names)}: {stat_names}")
    if not AGGREGATE_IN_STATS and len(stat_names) < len(sud_names):
        print(f"  '{AGGREGATE_COL}' is displayed but excluded from inference "
              f"(its sample is the union of the six substance-specific samples)")

    spins = C.make_spins(data_dir, N_PERM, CACHE) if "spin" in NULLS else None
    D = C.distance_matrix(data_dir) if "brainsmash" in NULLS else None
    if D is not None:
        print(f"BrainSMASH distance matrix from RAW centroids: "
              f"mean {D[np.triu_indices_from(D, 1)].mean():.1f} mm")

    for group, fname in GROUPS:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  !! {fname} missing, skipping {group}")
            continue
        psy, psy_names = C.read_maps(path, C.N_CORTEX)
        X = psy.to_numpy(float)
        outdir = os.path.join(OUT, group)
        os.makedirs(outdir, exist_ok=True)
        print(f"\n== {group}: {len(psy_names)} maps {psy_names} ==")

        RAW = {m: np.zeros((X.shape[1], Y.shape[1])) for m in C.MEASURES}
        P = {null: {m: np.zeros_like(RAW[m]) for m in C.MEASURES} for null in NULLS}
        keep = {}

        for i, nm in enumerate(psy_names):
            x = X[:, i]
            obs = C.observed_similarity(x, Y)
            for m in C.MEASURES:
                RAW[m][i] = obs[m]

            for null in NULLS:
                t0 = time.time()
                if null == "spin":
                    Xp = x[spins]
                else:
                    Xp = C.get_surrogates(x, D, N_PERM, f"{group}_{nm}", CACHE)
                nul = C.null_similarity(Xp, Y)
                for m in C.MEASURES:
                    P[null][m][i] = C.perm_p(obs[m], nul[m])
                    if SAVE_NULLS:
                        keep[f"{null}_{m}_{nm}"] = nul[m].astype(np.float32)
                print(f"   {nm:>10s} {null:<11s} {time.time()-t0:5.1f}s", flush=True)

        # ---------------- save ----------------
        # Two writers, deliberately: `w_full` for the descriptive matrices,
        # `w_stats` for anything inferential. The column set is not a detail of
        # formatting, so it is visible at every call site.
        def w_full(mat, tag):
            pd.DataFrame(mat, index=psy_names, columns=sud_names).to_csv(
                os.path.join(outdir, f"{tag}.csv"))

        def w_stats(mat, tag):
            pd.DataFrame(mat[:, stat_cols], index=psy_names, columns=stat_names).to_csv(
                os.path.join(outdir, f"{tag}.csv"))

        for m in C.MEASURES:
            w_full(RAW[m], f"RAW_cortex_{m}")
            for null in NULLS:
                if AGGREGATE_IN_PVALUES:
                    w_full(P[null][m], f"PVAL_cortex_{m}_{null}")
                    w_full(C.bh_fdr_by_column(P[null][m]),
                           f"pFDR_cortex_{m}_{null}")
                else:
                    w_stats(P[null][m], f"PVAL_cortex_{m}_{null}")
                    w_stats(C.bh_fdr_by_column(P[null][m][:, stat_cols]),
                            f"pFDR_cortex_{m}_{null}")

        if SAVE_NULLS:
            np.savez_compressed(os.path.join(outdir, "NULLS_cortex.npz"), **keep)

        # ---------------- ranking on RAW, not on z ----------------
        prim = RAW[C.PRIMARY]
        prim_stats = prim[:, stat_cols]
        for j, s in zip(stat_cols, stat_names):
            o = np.argsort(-prim[:, j])
            pd.DataFrame({"PSY": np.array(psy_names)[o],
                          f"{C.PRIMARY}": prim[o, j]}).to_csv(
                os.path.join(outdir, f"RANK_cortex_{C.PRIMARY}_by_{s}.csv"), index=False)
        mo = np.argsort(-prim_stats.mean(axis=1))
        pd.DataFrame({"PSY": np.array(psy_names)[mo],
                      f"mean_{C.PRIMARY}_over_SUD": prim_stats.mean(axis=1)[mo],
                      "n_sud_categories": len(stat_names)}).to_csv(
            os.path.join(outdir, f"RANK_cortex_{C.PRIMARY}_mean_across_SUD.csv"), index=False)

        # ---------------- console summary ----------------
        print(f"\n   {C.PRIMARY} (primary), pairs surviving pFDR<.05 "
              f"over {len(stat_names)} SUD categories:")
        qs = {}
        for null in NULLS:
            qs[null] = C.bh_fdr_by_column(P[null][C.PRIMARY][:, stat_cols])
            print(f"     {null:<11s} {(qs[null] < .05).sum():3d} / {qs[null].size}")
        if len(NULLS) == 2:
            both = (qs["spin"] < .05) & (qs["brainsmash"] < .05)
            print(f"     intersection {both.sum():3d} / {both.size}"
                  f"  <- the criterion used in the paper")
            per_disorder = pd.Series(both.sum(axis=1), index=psy_names)
            print("     per disorder: "
                  + ", ".join(f"{d} {n}/{len(stat_names)}"
                              for d, n in per_disorder.sort_values(ascending=False).items()))

    print(f"\nDone. Outputs -> {OUT}/<group>/  Cache -> {CACHE}")


if __name__ == "__main__":
    main()