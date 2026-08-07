#!/usr/bin/env python3
"""
RQ1 — step 2: subcortical PSY x SUD similarity (raw, no null)
=============================================================

Replaces RQ1_raw_subctx.py.

WHY NO NULL HERE
----------------
There are 14 subcortical structures and no centroid file for them, so neither
a spin (needs a sphere) nor BrainSMASH (needs a distance matrix) can be built.
With 14 points a null would in any case be too coarse to interpret. These
values are descriptive and the Results should present them that way — the same
position the previous version took, just stated explicitly.

WHAT IS DIFFERENT
-----------------
1. FILENAMES NO LONGER COLLIDE. The old script wrote
       RANK_spearman_by_ALC.csv
   into ALL_outputs_RQ1/<group>/ — byte-for-byte the same path the CORTEX
   script wrote its own ranking to. Whichever you ran last won, so the content
   of those files depended on the order you pressed Run. Everything here now
   carries `subctx` in the name.

2. Statistics come from RQ1_common, so `spearman` means the same thing in
   step 1 and step 2. The old script had its own private copy of the Spearman
   function.

3. Ranking is on RAW rho (it always was here — this script never had the
   saturating z, unlike the cortex ones).

Just press Run. Seconds.
"""

import os
import numpy as np
import pandas as pd

import RQ1_common as C

# ================= CONFIG — edit, then press Run =================
GROUPS = [
    ("adults_all",      "PSY_adults.xlsx"),
    ("adolescents_all", "PSY_adolescents.xlsx"),
    ("adults_ctx",      "PSY_adults_ctx.xlsx"),
    ("adolescents_ctx", "PSY_adolescents_ctx.xlsx"),
]
SUD_FILE = "SUD.xlsx"
DROP_SUD_AGGREGATE = False       # keep consistent with step 1
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUT = os.path.join(repo_dir, "ALL_outputs_RQ1")


def main():
    drop = ["SUD"] if DROP_SUD_AGGREGATE else []
    sud_sub, sud_names = C.subcortical_rows(os.path.join(data_dir, SUD_FILE), drop_cols=drop)
    Y = sud_sub.to_numpy(float)
    if Y.shape[0] == 0:
        raise ValueError("SUD.xlsx has no rows past 68: no subcortical data.")
    print(f"{Y.shape[0]} subcortical structures; SUD targets: {sud_names}")

    for group, fname in GROUPS:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  !! {fname} missing, skipping {group}")
            continue
        psy_sub, psy_names = C.subcortical_rows(path)
        if psy_sub.shape[0] == 0:
            print(f"  {group}: cortex-only file, no subcortical rows — skipping")
            continue
        n = min(psy_sub.shape[0], Y.shape[0])
        X = psy_sub.to_numpy(float)[:n, :]
        Ys = Y[:n, :]

        outdir = os.path.join(OUT, group)
        os.makedirs(outdir, exist_ok=True)

        RAW = {m: np.zeros((X.shape[1], Ys.shape[1])) for m in C.MEASURES}
        for i in range(X.shape[1]):
            obs = C.observed_similarity(X[:, i], Ys)
            for m in C.MEASURES:
                RAW[m][i] = obs[m]

        for m in C.MEASURES:
            pd.DataFrame(RAW[m], index=psy_names, columns=sud_names).to_csv(
                os.path.join(outdir, f"RAW_subctx_{m}.csv"))

        prim = RAW[C.PRIMARY]
        for j, s in enumerate(sud_names):
            o = np.argsort(-prim[:, j])
            pd.DataFrame({"PSY": np.array(psy_names)[o],
                          C.PRIMARY: prim[o, j]}).to_csv(
                os.path.join(outdir, f"RANK_subctx_{C.PRIMARY}_by_{s}.csv"), index=False)
        mo = np.argsort(-prim.mean(axis=1))
        pd.DataFrame({"PSY": np.array(psy_names)[mo],
                      f"mean_{C.PRIMARY}_over_SUD": prim.mean(axis=1)[mo]}).to_csv(
            os.path.join(outdir, f"RANK_subctx_{C.PRIMARY}_mean_across_SUD.csv"), index=False)

        print(f"\n== {group} ({n} structures) — {C.PRIMARY} ==")
        print(pd.DataFrame(prim, index=psy_names, columns=sud_names).round(3).to_string())

    print(f"\nDone. Outputs -> {OUT}/<group>/  (all filenames carry `subctx`)")


if __name__ == "__main__":
    main()