#!/usr/bin/env python3
"""
RQ2_common — shared machinery for RQ2, built on top of RQ1_common

"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import RQ1_common as C

N_CORTEX = C.N_CORTEX
SEED = C.SEED

# "pearson" (what every RQ2 script has always computed) or "spearman"
GRADIENT_METRIC = "pearson"

GRADIENTS = ["C1", "C2", "C3", "FC", "MPC"]
TRANSCRIPTIONAL = ["C1", "C2", "C3"]


# =================================================================
# I/O
# =================================================================
def read_csv_robust(path, header="infer"):
    """
    Excel on an Italian-locale Windows install re-saves CSVs with ';' as
    separator and ',' as decimal mark, which silently turns numeric columns
    into strings. Try the plausible combinations, keep the first that yields
    numeric data.
    """
    last = None
    for sep, dec in ((",", "."), (";", ","), (";", "."), ("\t", "."), (",", ",")):
        try:
            df = pd.read_csv(path, sep=sep, decimal=dec, header=header, engine="python")
        except Exception as e:                                   # noqa: BLE001
            last = e
            continue
        if df.shape[1] >= 1 and df.select_dtypes(include=np.number).shape[1] >= 1:
            return df
    raise IOError(f"could not parse {path} as a numeric CSV ({last})")


def load_maps(data_dir):
    """
    Every cortical map RQ2 uses, in one place.

    Returns a dict with:
      psy  (68 x 9)  adult psychiatric, PD in place of PTSD
      sud  (68 x 6)  substance-specific ONLY; the generic aggregate is dropped
      ped  (68 x 6)  pediatric/adolescent maps, both workbooks concatenated
    """
    psy, psy_cols = C.read_maps(os.path.join(data_dir, "PSY_adults.xlsx"), N_CORTEX)
    sud, sud_cols = C.read_maps(os.path.join(data_dir, "SUD.xlsx"), N_CORTEX,
                                drop_cols=["SUD"])
    if len(sud_cols) != 6:
        raise ValueError(f"expected 6 substance maps after dropping the aggregate, "
                         f"got {sud_cols}")
    C.check_cluster_coverage(psy_cols)

    peds = []
    for f in ("PSY_adolescents.xlsx", "PSY_adolescents_ctx.xlsx"):
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            peds.append(C.read_maps(p, N_CORTEX)[0])
    ped = pd.concat(peds, axis=1) if peds else None

    print(f"[maps] PSY adults {list(psy.columns)}")
    print(f"[maps] SUD        {list(sud.columns)}  (generic aggregate dropped)")
    if ped is not None:
        print(f"[maps] pediatric  {list(ped.columns)}")
    return {"psy": psy, "sud": sud, "ped": ped}


def load_gradients(data_dir):
    """
    C1/C2/C3 come from the AHBA DME scores, which exist for 34 LEFT-hemisphere
    parcels only and are mirrored onto the right. That mirroring is why the
    hemisphere spin bug had no practical effect on C1-C3: the surrogate maps
    were wrong, but the target was symmetric so the correlation barely moved.
    FC and MPC are genuinely 68-parcel and asymmetric, so for them it mattered.
    """
    dme = read_csv_robust(os.path.join(data_dir, "ahba_dme_scores_in_dk.csv"))
    if dme.shape[0] != N_CORTEX // 2:
        raise ValueError(f"expected 34 AHBA parcels, got {dme.shape[0]}")
    g = {c: np.concatenate([dme[c].to_numpy(float)] * 2) for c in TRANSCRIPTIONAL}
    for tag, f in (("FC", "mica_hc100_gradient-FC.csv"),
                   ("MPC", "mica_hc100_gradient-MPC.csv")):
        g[tag] = read_csv_robust(os.path.join(data_dir, f),
                                 header=None).iloc[:, 0].to_numpy(float)
    for k, v in g.items():
        if v.shape[0] != N_CORTEX:
            raise ValueError(f"gradient {k} has {v.shape[0]} parcels, expected {N_CORTEX}")
    lr = {k: np.corrcoef(v[:34], v[34:])[0, 1] for k, v in g.items()}
    print("[gradients] LH-RH correlation: "
          + ", ".join(f"{k} {r:+.2f}" for k, r in lr.items())
          + "   (C1-C3 are mirrored by construction)")
    return g


# =================================================================
# Aggregates
# =================================================================
def aggregates(maps):
    """
    The maps every RQ2 analysis is built from.

    NOTE on `shared`: it is the unweighted mean of the PSY mean and the SUD
    mean. When a group's psychiatric maps have small effect sizes the SUD term
    dominates it, which is why any claim about a cluster's shared map needs the
    r(shared, SUD mean) diagnostic alongside it.
    """
    P = maps["psy"].to_numpy(float)
    S = maps["sud"].to_numpy(float)
    out = {"PSY mean": P.mean(1), "SUD mean": S.mean(1)}
    out["shared"] = (out["PSY mean"] + out["SUD mean"]) / 2
    if maps["ped"] is not None:
        out["Pediatric"] = np.nanmean(maps["ped"].to_numpy(float), axis=1)
    for cl, members in C.CLUSTER_MEMBERS.items():
        present = [m for m in members if m in maps["psy"].columns]
        if not present:
            raise KeyError(f"cluster {cl} has no members in the adult maps")
        if len(present) < len(members):
            print(f"  [warn] cluster {cl}: {sorted(set(members) - set(present))} absent; "
                  f"the mean is over {present} only")
        out[cl] = maps["psy"][present].to_numpy(float).mean(1)
    return out


# =================================================================
# Correlation and inference
# =================================================================
def corr(x, y):
    """Scalar correlation under GRADIENT_METRIC."""
    if GRADIENT_METRIC == "spearman":
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def corr_rows(A, y):
    """Correlation between each ROW of A (n, p) and vector y (p,), same metric."""
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    if GRADIENT_METRIC == "spearman":
        A, y = rankdata(A, axis=1), rankdata(y)
    Ac = A - A.mean(1, keepdims=True)
    yc = y - y.mean()
    denom = np.linalg.norm(Ac, axis=1) * np.linalg.norm(yc)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, 0.0, (Ac @ yc) / denom)


def z_p(obs, null):
    """
    Two-tailed permutation p and a genuine standardized effect.

    z is (obs - mean(null)) / sd(null), NOT norm.isf(p) — the RQ1 scripts used
    to reparametrise p as z, which saturated and made rankings meaningless.
    """
    null = np.asarray(null, float).ravel()
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)
    sd = null.std(ddof=1)
    return (0.0 if sd == 0 else float((obs - null.mean()) / sd)), float(min(p, 1.0))


def gradient_nulls(grads, data_dir, cache_dir, n_perm, kinds=("spin",)):
    """
    Surrogates of each GRADIENT (the gradient is permuted, the brain map held
    fixed — the convention used throughout RQ2, so values stay comparable with
    the published ones).

    Spins and BrainSMASH surrogates both come from RQ1_common, so RQ1 and RQ2
    share one set of rotations and one distance matrix. BrainSMASH surrogates
    are cached by content hash: a gradient that has not changed is generated
    once, ever.
    """
    out = {}
    if "spin" in kinds:
        spins = C.make_spins(data_dir, n_perm, cache_dir)
        out["spin"] = {k: g[spins] for k, g in grads.items()}
    if "brainsmash" in kinds:
        D = C.distance_matrix(data_dir)          # RAW centroids, not unit sphere
        out["brainsmash"] = {}
        for k, g in grads.items():
            print(f"   BrainSMASH surrogates for gradient {k} ...", flush=True)
            out["brainsmash"][k] = C.get_surrogates(g, D, n_perm, f"grad_{k}", cache_dir)
    return out


bh_fdr = C.bh_fdr