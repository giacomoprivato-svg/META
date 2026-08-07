#!/usr/bin/env python3
"""
RQ1 — step 1: cortical PSY x SUD similarity, both nulls, one script
===================================================================

Replaces RQ1_similarity_pspin.py AND RQ1_similarity_brainsmash.py. They were
two copies of the same pipeline that had drifted apart; everything statistical
now comes from RQ1_common.

WHAT IS DIFFERENT
-----------------
1. HEMISPHERE BUG GONE. Spins come from RQ1_common.make_spins, which applies
   the +34 offset and asserts it on every load. The old spin script had its
   own uncorrected copy and cached to `spins_ctx_68.mat` — the same filename
   as the buggy cache, so reruns silently reloaded broken spins. New cache
   name, new location, guard assertion.

2. ONE SET OF SPINS FOR ALL GROUPS. Previously spins were generated inside
   the loop over GROUPS, so adults_all and adults_ctx used different
   rotations and were then FDR-corrected together as one family.

3. NO Z. z = sign * norm.isf(p_one) was a deterministic function of p that
   saturated at 3.719, and RANK_* files were sorted on it — producing ties at
   the ceiling. Rankings now use RAW rho; significance is p and pFDR.

4. BRAINSMASH GEOMETRY FIXED. The distance matrix is built from RAW centroids
   (true inter-regional distances, which is what a variogram needs). The unit
   sphere projection is used only for spins.

5. NO FILENAME COLLISION. Everything carries the compartment (`cortex`) and
   the null (`spin` / `brainsmash`) in its name, so step 2 (subcortex) cannot
   overwrite it. Previously RANK_spearman_by_ALC.csv was written by both the
   cortex and the subcortex script into the same folder, and the survivor
   depended on which you pressed Run on last.

6. SURROGATES CACHED. Keyed by a hash of the map. Rerunning after changing
   one disorder regenerates one map, not nine.

7. FASTER, IDENTICALLY. Surrogates are ranked once per psychiatric map rather
   than once per (map, target) pair, and all 7 targets are evaluated in one
   matrix product. Same numbers, ~7x less rank sorting.

Just press Run.
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
DROP_SUD_AGGREGATE = False       # False = keep the 7-column SUD panel as published.
                                 # True  = drop the transdiagnostic all-SUD column
                                 #         (the de-double-counting fix; RQ2 and the
                                 #         Delta analysis always drop it).
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUT = os.path.join(repo_dir, "ALL_outputs_RQ1")
CACHE = os.path.join(OUT, "_cache")
os.makedirs(CACHE, exist_ok=True)


def main():
    drop = ["SUD"] if DROP_SUD_AGGREGATE else []
    sud, sud_names = C.read_maps(os.path.join(data_dir, SUD_FILE), C.N_CORTEX, drop_cols=drop)
    Y = sud.to_numpy(float)
    print(f"SUD targets ({len(sud_names)}): {sud_names}")

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
        def w(mat, tag):
            pd.DataFrame(mat, index=psy_names, columns=sud_names).to_csv(
                os.path.join(outdir, f"{tag}.csv"))

        for m in C.MEASURES:
            w(RAW[m], f"RAW_cortex_{m}")
            for null in NULLS:
                w(P[null][m], f"PVAL_cortex_{m}_{null}")
                w(C.bh_fdr_by_column(P[null][m]), f"pFDR_cortex_{m}_{null}")

        if SAVE_NULLS:
            np.savez_compressed(os.path.join(outdir, "NULLS_cortex.npz"), **keep)

        # ---------------- ranking on RAW, not on z ----------------
        prim = RAW[C.PRIMARY]
        for j, s in enumerate(sud_names):
            o = np.argsort(-prim[:, j])
            pd.DataFrame({"PSY": np.array(psy_names)[o],
                          f"{C.PRIMARY}": prim[o, j]}).to_csv(
                os.path.join(outdir, f"RANK_cortex_{C.PRIMARY}_by_{s}.csv"), index=False)
        mo = np.argsort(-prim.mean(axis=1))
        pd.DataFrame({"PSY": np.array(psy_names)[mo],
                      f"mean_{C.PRIMARY}_over_SUD": prim.mean(axis=1)[mo]}).to_csv(
            os.path.join(outdir, f"RANK_cortex_{C.PRIMARY}_mean_across_SUD.csv"), index=False)

        # ---------------- console summary ----------------
        print(f"\n   {C.PRIMARY} (primary), pairs surviving pFDR<.05:")
        for null in NULLS:
            q = C.bh_fdr_by_column(P[null][C.PRIMARY])
            print(f"     {null:<11s} {(q < .05).sum():3d} / {q.size}")
        if len(NULLS) == 2:
            both = (C.bh_fdr_by_column(P["spin"][C.PRIMARY]) < .05) & \
                   (C.bh_fdr_by_column(P["brainsmash"][C.PRIMARY]) < .05)
            print(f"     intersection {both.sum():3d} / {both.size}  <- the criterion used in the paper")

    print(f"\nDone. Outputs -> {OUT}/<group>/  Cache -> {CACHE}")


if __name__ == "__main__":
    main()