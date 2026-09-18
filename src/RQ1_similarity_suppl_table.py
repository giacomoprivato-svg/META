#!/usr/bin/env python3
"""
RQ1 — AIM1 step 2: supplementary block table

"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import RQ1_common as C

# ================= CONFIG — edit, then press Run =================
RECOMPUTE = False     # False = read step 1's matrices (default)
                      # True  = rebuild from the Excel files and ASSERT they
                      #         match step 1 to 1e-10. Use as an audit.
TOL = 1e-10
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ1", "specificity_adults")

PP_FILE = os.path.join(OUTDIR, "PSYxPSY_spearman_adults.csv")
PS_FILE = os.path.join(OUTDIR, "PSYxSUD_spearman_adults_6substances.csv")


def from_source():
    psy, names = C.read_maps(os.path.join(data_dir, "PSY_adults.xlsx"), C.N_CORTEX)
    sud, sud_names = C.read_maps(os.path.join(data_dir, "SUD.xlsx"), C.N_CORTEX,
                                 drop_cols=["SUD"])
    Rp, Rs = rankdata(psy.to_numpy(float), axis=0), rankdata(sud.to_numpy(float), axis=0)
    PP = C._corr_rows_cols(Rp.T, Rp)
    np.fill_diagonal(PP, np.nan)
    return (pd.DataFrame(PP, index=names, columns=names),
            pd.DataFrame(C._corr_rows_cols(Rp.T, Rs), index=names, columns=sud_names))


def load_matrices():
    if RECOMPUTE or not (os.path.exists(PP_FILE) and os.path.exists(PS_FILE)):
        PP, PS = from_source()
        if os.path.exists(PP_FILE) and os.path.exists(PS_FILE):
            PPr = pd.read_csv(PP_FILE, index_col=0)
            PSr = pd.read_csv(PS_FILE, index_col=0)
            d1 = np.nanmax(np.abs(PP.to_numpy() - PPr.reindex_like(PP).to_numpy()))
            d2 = np.nanmax(np.abs(PS.to_numpy() - PSr.reindex_like(PS).to_numpy()))
            if max(d1, d2) > TOL:
                raise AssertionError(
                    f"recomputed matrices disagree with step 1 (max diff PSYxPSY={d1:.2e}, "
                    f"PSYxSUD={d2:.2e}). The two code paths have drifted — do not "
                    f"publish either number until you know why.")
            print(f"audit OK: recomputed == step 1 (max diff {max(d1, d2):.1e})")
        return PP, PS
    return pd.read_csv(PP_FILE, index_col=0), pd.read_csv(PS_FILE, index_col=0)


def main():
    PP, PS = load_matrices()
    names = list(PP.index)
    C.check_cluster_coverage(names)

    rows = []
    for d in names:
        own = [m for m in C.CLUSTER_MEMBERS[C.CLUSTERS[d]] if m != d]
        rest = [m for m in names if m != d and C.CLUSTERS[m] != C.CLUSTERS[d]]
        rec = {"disorder": d,
               "cluster": C.CLUSTERS[d],
               "rho_SUD_block": PS.loc[d].mean(),
               "rho_own_cluster": PP.loc[d, own].mean() if own else np.nan,
               "rho_rest_of_panel": PP.loc[d, rest].mean()}
        for b, mem in C.CLUSTER_MEMBERS.items():
            mm = [m for m in mem if m != d and m in names]
            rec[f"rho_{b}"] = np.nan if C.CLUSTERS[d] == b or not mm else PP.loc[d, mm].mean()
        rows.append(rec)
    tab = pd.DataFrame(rows).set_index("disorder")

    for bench in ("all", "outcluster"):
        f = os.path.join(OUTDIR, f"SPECIFICITY_delta_adults_{bench}.csv")
        if not os.path.exists(f):
            print(f"  !! {os.path.basename(f)} not found — "
                  f"run RQ1_AIM1_step1_specificity_delta.py with "
                  f"BENCHMARKS including '{bench}'. Columns for this benchmark "
                  f"will be absent from the table.")
            continue
        d = pd.read_csv(f, index_col=0)
        keep = {"delta": f"delta_{bench}", "pFDR_spin": f"pFDR_spin_{bench}"}
        if "pFDR_brainsmash" in d.columns:
            keep["pFDR_bs"] = None                       # placeholder, set below
            keep.pop("pFDR_bs")
            keep["pFDR_brainsmash"] = f"pFDR_bs_{bench}"
        else:
            print(f"  note: {bench} has spin p-values only "
                  f"(NULL_MODE was not 'both' in step 1)")
        tab = tab.join(d[list(keep)].rename(columns=keep))

    tab = tab.sort_values("rho_SUD_block", ascending=False)
    out = os.path.join(OUTDIR, "SUPP_block_table.csv")
    tab.to_csv(out)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string())
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()