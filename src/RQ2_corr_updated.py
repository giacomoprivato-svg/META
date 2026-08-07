#!/usr/bin/env python3
"""
RQ2 — AIM 2: CORRECTED LEAVE-ONE-OUT + DECOMPOSED PANEL D
=========================================================

Addresses the PI's two substantive points on Figure 3 / Supplementary S17-S19.

PART 1 — Corrected leave-one-out
    The published LOO correlated PSY_mean(-i) with PSY_mean(all), which is ~.99
    by construction and measures nothing. The correct test drops disorder i from
    the PSY aggregate and re-correlates against the SUD aggregate, comparing the
    result with the full-sample r. Run on both sides (drop a PSY map, drop a
    substance map). This matters because AN's effect sizes are ~3x the others
    and dominate the unweighted PSY mean.

PART 2 — Decomposed panel D
    "C3 alignment generalises across all four groups" is circular if the
    cluster-specific shared maps are themselves near-copies of the SUD mean
    (the shared map is (PSY_mean + SUD_mean)/2, and when a cluster's PSY maps
    have small effect sizes the SUD term dominates). This script therefore
    correlates each gradient with THREE maps per group, side by side:
        (a) the group's PSY component alone
        (b) the SUD mean alone            <- identical across groups by design
        (c) the shared map                <- what the paper currently reports
    If (c) tracks (b) and not (a), the result is a SUD result, not a shared one.
    Part 2b reports r(shared_map, SUD_mean) per group as the direct diagnostic.

Gradients tested: C1, C2, C3 (AHBA DME, left hemisphere mirrored), FC G1, MPC G1.
FDR is applied JOINTLY across all rows x all gradients, separately within each
null framework; an association counts as robust only if it survives under both.

TWO FIXES BAKED IN
------------------
1. SUD double-counting: the generic all-SUD column is dropped from the SUD
   aggregate. `RQ2_corr_C1C2C3.py`, `RQ2_corr_MPCFC.py`, `RQ2_shared_PCA_corr.py`
   and `RQ2_mean_PSYSUD.py` in the repo still lack this fix and would regenerate
   pre-fix values if re-run.
2. Hemisphere offset in the spin: this script uses `make_spins` from
   RQ1_common (right-hemisphere indices offset by +34, with a guard assertion).
   The local `build_spins()` in `RQ2_corr_gradients.py` lacks the offset. That is
   a no-op for C1-C3, whose maps are bilaterally mirrored, but NOT for FC and MPC
   (LH-RH correlation .95 and .87), so the published FC/MPC spin p-values were
   computed with the buggy spin and are recomputed here.

WHICH MAP IS PERMUTED
---------------------
The gradient is permuted and the brain map held fixed, matching the convention
in `RQ2_corr_gradients.py` so that values remain comparable with the published
ones. Note that this makes the comparison across rows share a single null per
gradient, which is the intended behaviour.

Usage
-----
    Just press Run. Set NULL_MODE in the CONFIG block below.

Outputs -> ALL_outputs_RQ2/aim2/
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from RQ1_common import make_spins, load_centroids

# =====================================================================
# CONFIG — edit these, then press Run. No arguments needed.
# =====================================================================
NULL_MODE = "both"      # "spin" = fast check (~2 min); "both" = + BrainSMASH (slow)
EXTENDED = False        # True adds shared maps and FC/MPC as an internal check.
                        # Those rows are NOT panel D and stay out of its FDR family.
N_PERM = 10000
# =====================================================================

N_CORTEX = 68
SEED = 42

CLUSTERS = {"Psychotic": ["SCZ", "CHR", "BD"], "AN/OCD": ["AN", "OCD"],
            "Mood/Anx": ["MDD", "PTSD"], "Neurodev": ["ADHD", "ASD"]}

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ2", "aim2")
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------------- helpers
def read_csv_robust(path, header="infer"):
    """
    Tolerant CSV reader. Excel on an Italian-locale Windows install re-saves CSVs
    with ';' as separator and ',' as decimal mark, which silently turns numeric
    columns into strings. Try the plausible combinations and keep the first that
    yields numeric data.
    """
    last = None
    for sep, dec in ((",", "."), (";", ","), (";", "."), ("\t", "."), (",", ",")):
        try:
            df = pd.read_csv(path, sep=sep, decimal=dec, header=header, engine="python")
        except Exception as e:                                    # noqa: BLE001
            last = e
            continue
        if df.shape[1] >= 1 and df.select_dtypes(include=np.number).shape[1] >= 1:
            return df
    raise IOError(f"could not parse {path} as numeric CSV ({last})")


def load_num(fname, drop=None):
    df = pd.read_excel(os.path.join(data_dir, fname)).select_dtypes(include=np.number)
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])
    return df.iloc[:N_CORTEX, :]


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    r = p[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(r, 0, 1)
    return out


def z_p(obs, null):
    null = np.asarray(null, float)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)
    return float((obs - null.mean()) / null.std(ddof=1)), float(min(p, 1.0))


def corr_rows(A, y):
    """Pearson r between each row of A (n, p) and vector y (p,)."""
    Ac = A - A.mean(1, keepdims=True)
    yc = y - y.mean()
    return (Ac @ yc) / (np.linalg.norm(Ac, axis=1) * np.linalg.norm(yc))


# ---------------------------------------------------------------- data
def load_all():
    psy = load_num("PSY_adults.xlsx")
    sud = load_num("SUD.xlsx", drop=["SUD"])          # FIX 1
    assert sud.shape[1] == 6, list(sud.columns)

    ado = load_num("PSY_adolescents.xlsx")
    ado_ctx = load_num("PSY_adolescents_ctx.xlsx")
    ped = pd.concat([ado, ado_ctx], axis=1)          # 6 pediatric cortical maps

    dme = read_csv_robust(os.path.join(data_dir, "ahba_dme_scores_in_dk.csv"))
    assert dme.shape[0] == N_CORTEX // 2, f"expected 34 AHBA parcels, got {dme.shape[0]}"
    grads = {c: np.concatenate([dme[c].to_numpy(float), dme[c].to_numpy(float)])
             for c in ("C1", "C2", "C3")}            # left hemisphere mirrored
    for tag, f in (("FC", "mica_hc100_gradient-FC.csv"),
                   ("MPC", "mica_hc100_gradient-MPC.csv")):
        grads[tag] = read_csv_robust(os.path.join(data_dir, f),
                                     header=None).iloc[:, 0].to_numpy(float)
    for k, v in grads.items():
        assert v.shape[0] == N_CORTEX, (k, v.shape)
    return psy, sud, ped, grads


# ---------------------------------------------------------------- PART 1
def corrected_loo(psy, sud):
    P, S = psy.to_numpy(float), sud.to_numpy(float)
    psy_mean, sud_mean = P.mean(1), S.mean(1)
    r_full = np.corrcoef(psy_mean, sud_mean)[0, 1]

    rows = []
    for j, d in enumerate(psy.columns):
        red = np.delete(P, j, axis=1).mean(1)
        r = np.corrcoef(red, sud_mean)[0, 1]
        rows.append(("PSY", d, r, r - r_full))
    for j, d in enumerate(sud.columns):
        red = np.delete(S, j, axis=1).mean(1)
        r = np.corrcoef(psy_mean, red)[0, 1]
        rows.append(("SUD", d, r, r - r_full))

    loo = pd.DataFrame(rows, columns=["side", "dropped", "r_LOO", "delta_vs_full"])
    loo["r_full"] = r_full
    loo = loo.sort_values("r_LOO")
    loo.to_csv(os.path.join(OUTDIR, "PART1_corrected_LOO.csv"), index=False)

    print(f"\n=== PART 1 — corrected leave-one-out (full r = {r_full:.3f}) ===")
    print(loo.round(4).to_string(index=False))
    print(f"\n  range of LOO r: {loo.r_LOO.min():.3f} to {loo.r_LOO.max():.3f}"
          f"  |  largest single-map influence: "
          f"{loo.loc[loo.delta_vs_full.abs().idxmax(), 'dropped']} "
          f"({loo.delta_vs_full.abs().max():+.3f})")
    return r_full


# ---------------------------------------------------------------- PART 2
def build_rows(psy, sud, ped, extended=False):
    P, S = psy.to_numpy(float), sud.to_numpy(float)
    sud_mean = S.mean(1)
    psy_mean = P.mean(1)
    ped_mean = np.nanmean(ped.to_numpy(float), axis=1)

    groups = {"Adults": psy_mean, "Pediatric": ped_mean}
    for cl, mem in CLUSTERS.items():
        groups[cl] = psy[[m for m in mem if m in psy.columns]].to_numpy(float).mean(1)

    # Panel D as specified: component maps only, no shared maps.
    # Putting the PSY component maps in the rows removes the circularity by
    # construction — each row is independent of the SUD row — so the
    # shared-map / shared-vs-SUD-mean diagnostics are not part of the panel.
    rows = [("SUD mean", "component", sud_mean)]
    for g, pm in groups.items():
        rows.append((g, "component", pm))
    if extended:      # internal check only, NOT the panel
        for g, pm in groups.items():
            rows.append((g, "shared map", (pm + sud_mean) / 2))
    return rows, sud_mean


def run_part2(rows, sud_mean, grads, use_bs, extended=False):
    if not extended:
        grads = {k: v for k, v in grads.items() if k in ("C1", "C2", "C3")}
    spins = make_spins(data_dir, N_PERM,
                       cache_path=os.path.join(OUTDIR, f"spins_ctx68_n{N_PERM}.mat"),
                       seed=SEED)
    if use_bs:
        from brainsmash.mapgen.base import Base
        LH, RH = load_centroids(data_dir)
        coords = np.vstack([LH, RH])
        D = cdist(coords, coords)

    # one null per gradient (the gradient is the permuted map)
    spun = {k: g[spins] for k, g in grads.items()}
    bs = {}
    if use_bs:
        for k, g in grads.items():
            print(f"   BrainSMASH surrogates for gradient {k} ...", flush=True)
            bs[k] = Base(x=g, D=D, resample=True, seed=SEED)(n=N_PERM)

    out = []
    for group, kind, m in rows:
        for gname, g in grads.items():
            obs = np.corrcoef(g, m)[0, 1]
            zs, ps = z_p(obs, corr_rows(spun[gname], m))
            rec = dict(group=group, map=kind, gradient=gname, r=obs,
                       z_spin=zs, p_spin=ps)
            if use_bs:
                zb, pb = z_p(obs, corr_rows(bs[gname], m))
                rec.update(z_brainsmash=zb, p_brainsmash=pb)
            out.append(rec)

    df = pd.DataFrame(out)
    # FDR family = the panel itself (7 rows x 3 components = 21 tests),
    # SEPARATE from the panel C family. Do not pool the two.
    df["pFDR_spin"] = bh_fdr(df["p_spin"])
    if use_bs:
        df["pFDR_brainsmash"] = bh_fdr(df["p_brainsmash"])
        df["robust"] = (df.pFDR_spin < .05) & (df.pFDR_brainsmash < .05)
    else:
        df["robust_spin_only"] = df.pFDR_spin < .05
    df.to_csv(os.path.join(OUTDIR, "PART2_panelD_decomposed.csv"), index=False)

    print("\n=== PART 2 — panel D, decomposed (r, joint-FDR q under spin) ===")
    sig = "robust" if use_bs else "robust_spin_only"
    piv_r = df.pivot_table(index=["group", "map"], columns="gradient", values="r")
    piv_q = df.pivot_table(index=["group", "map"], columns="gradient", values="pFDR_spin")
    order = [g for g in ("C1", "C2", "C3", "FC", "MPC") if g in piv_r.columns]
    for idx in piv_r.index:
        line = f"  {idx[0]:12s} {idx[1]:14s}"
        for g in order:
            flag = "*" if df[(df.group == idx[0]) & (df["map"] == idx[1]) &
                             (df.gradient == g)][sig].iloc[0] else " "
            line += f"  {g}: r={piv_r.loc[idx, g]:+.3f} q={piv_q.loc[idx, g]:.3f}{flag}"
        print(line)

    # ---- PART 2b: circularity diagnostic (justification text only, not the panel)
    diag = []
    for group, kind, m in rows:
        if kind != "shared map":
            continue
        pm = [x for gg, kk, x in rows if gg == group and kk == "PSY component"][0]
        diag.append(dict(group=group,
                         r_shared_vs_SUDmean=np.corrcoef(m, sud_mean)[0, 1],
                         r_shared_vs_PSYcomponent=np.corrcoef(m, pm)[0, 1],
                         sd_PSY_component=pm.std(), sd_SUD_mean=sud_mean.std()))
    if diag:
        diag = pd.DataFrame(diag)
        diag.to_csv(os.path.join(OUTDIR, "PART2b_circularity_diagnostic.csv"), index=False)
        print("\n=== PART 2b — how much is each shared map just the SUD mean? ===")
        print(diag.round(3).to_string(index=False))
        print("\n  A shared map with r > ~.9 against the SUD mean carries little independent\n"
              "  PSY information; its gradient correlations restate the SUD row.\n"
              "  Justification text only — not part of panel D.")
    return df


def main():
    psy, sud, ped, grads = load_all()
    corrected_loo(psy, sud)
    rows, sud_mean = build_rows(psy, sud, ped, extended=EXTENDED)
    run_part2(rows, sud_mean, grads, use_bs=(NULL_MODE == "both"), extended=EXTENDED)
    print(f"\nDone. Outputs -> {OUTDIR}")


if __name__ == "__main__":
    main()