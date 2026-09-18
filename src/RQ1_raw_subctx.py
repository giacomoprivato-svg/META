#!/usr/bin/env python3
"""
RQ1 — step 2: subcortical PSY x SUD similarity (raw, no null)

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

# --- keep these identical to step 1 ---
DROP_SUD_AGGREGATE = False       # False = RAW_subctx_* keeps the 7-column panel
AGGREGATE_IN_STATS = False       # False = the means use six substance maps only
AGGREGATE_COL = "SUD"
N_SUBSTANCE_MAPS = 6
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUT = os.path.join(repo_dir, "ALL_outputs_RQ1")


def statistics_columns(sud_names):
    """
    Indices of the SUD columns allowed into the means. Same contract as step 1:
    raises rather than silently proceeding if the aggregate cannot be found or
    the remaining panel is not the expected six.
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
            f"expected {N_SUBSTANCE_MAPS} substance-specific maps for the means, "
            f"got {len(names)}: {names}")
    return idx, names


def main():
    drop = [AGGREGATE_COL] if DROP_SUD_AGGREGATE else []
    sud_sub, sud_names = C.subcortical_rows(os.path.join(data_dir, SUD_FILE), drop_cols=drop)
    Y = sud_sub.to_numpy(float)
    if Y.shape[0] == 0:
        raise ValueError("SUD.xlsx has no rows past 68: no subcortical data.")
    stat_cols, stat_names = statistics_columns(sud_names)

    print(f"{Y.shape[0]} subcortical structures; SUD panel ({len(sud_names)}): {sud_names}")
    print(f"  RAW matrices written for all {len(sud_names)} columns; "
          f"means over {len(stat_names)}: {stat_names}")

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
        prim_stats = prim[:, stat_cols]
        for j, s in zip(stat_cols, stat_names):
            o = np.argsort(-prim[:, j])
            pd.DataFrame({"PSY": np.array(psy_names)[o],
                          C.PRIMARY: prim[o, j]}).to_csv(
                os.path.join(outdir, f"RANK_subctx_{C.PRIMARY}_by_{s}.csv"), index=False)
        mo = np.argsort(-prim_stats.mean(axis=1))
        pd.DataFrame({"PSY": np.array(psy_names)[mo],
                      f"mean_{C.PRIMARY}_over_SUD": prim_stats.mean(axis=1)[mo],
                      "n_sud_categories": len(stat_names)}).to_csv(
            os.path.join(outdir, f"RANK_subctx_{C.PRIMARY}_mean_across_SUD.csv"), index=False)

        print(f"\n== {group} ({n} structures) — {C.PRIMARY} ==")
        shown = pd.DataFrame(prim, index=psy_names, columns=sud_names)
        print(shown.round(3).to_string())
        if len(stat_names) < len(sud_names):
            print(f"   (mean_across_SUD excludes '{AGGREGATE_COL}')")

    print(f"\nDone. Outputs -> {OUT}/<group>/  (all filenames carry `subctx`)")


if __name__ == "__main__":
    main()