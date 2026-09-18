#!/usr/bin/env python3
"""
RQ2 — step 2: gradient alignment, two declared analyses

"""

import os
import numpy as np
import pandas as pd

import RQ1_common as C
import RQ2_common as R2

# ================= CONFIG — edit, then press Run =================
NULLS = ["spin", "brainsmash"]      # ["spin"] alone for a fast check
N_PERM = 10000
PERMUTED_SIDE = "map"               # "map" (default, see note 1) | "gradient"
GRADIENTS_A = ["C1", "C2", "C3", "FC", "MPC"]     # analysis A
GRADIENTS_B = ["C1", "C2", "C3"]                  # analysis B (panel D)
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ2", "gradients")
CACHE = os.path.join(repo_dir, "ALL_outputs_RQ1", "_cache")
os.makedirs(OUTDIR, exist_ok=True)

ROW_ORDER = ["SUD mean", "Adults", "Pediatric", "Psychotic", "AN/OCD",
             "Mood/Anxiety", "Neurodevelopmental"]


def surrogates_for(m, label, spins, D):
    """Whichever nulls are switched on, for one map. BrainSMASH hits the cache."""
    s = {}
    if spins is not None:
        s["spin"] = m[spins]
    if D is not None:
        s["brainsmash"] = C.get_surrogates(m, D, N_PERM, label, CACHE)
    return s


def test(m, m_surr, grads, gradient_names, g_surr):
    recs = []
    for gn in gradient_names:
        g = grads[gn]
        rec = {"gradient": gn, "r": R2.corr(g, m)}
        for n in NULLS:
            null = (R2.corr_rows(g_surr[n][gn], m) if PERMUTED_SIDE == "gradient"
                    else R2.corr_rows(m_surr[n], g))
            rec[f"z_{n}"], rec[f"p_{n}"] = R2.z_p(rec["r"], null)
        recs.append(rec)
    return recs


def add_fdr(df):
    for n in NULLS:
        df[f"pFDR_{n}"] = C.bh_fdr(df[f"p_{n}"].values)
    df["robust"] = ((df.pFDR_spin < .05) & (df.pFDR_brainsmash < .05)
                    if len(NULLS) == 2 else df[f"pFDR_{NULLS[0]}"] < .05)
    return df


def main():
    maps = R2.load_maps(data_dir)
    agg = R2.aggregates(maps)
    grads = R2.load_gradients(data_dir)
    print(f"\npermuted side: {PERMUTED_SIDE}   nulls: {NULLS}   n_perm = {N_PERM}")

    spins = C.make_spins(data_dir, N_PERM, CACHE) if "spin" in NULLS else None
    D = C.distance_matrix(data_dir) if "brainsmash" in NULLS else None

    g_surr = {n: {} for n in NULLS}
    if PERMUTED_SIDE == "gradient":
        for gn, g in grads.items():
            for n, v in surrogates_for(g, f"grad_{gn}", spins, D).items():
                g_surr[n][gn] = v

    rows = [("SUD mean", agg["SUD mean"]), ("Adults", agg["PSY mean"])]
    if "Pediatric" in agg:
        rows.append(("Pediatric", agg["Pediatric"]))
    for cl in C.CLUSTER_MEMBERS:
        rows.append((cl, agg[cl]))
    shared_adult = agg["shared"]

    # ---------------- ANALYSIS A ----------------
    sa = surrogates_for(shared_adult, "sharedmap_Adults", spins, D)
    A = add_fdr(pd.DataFrame(test(shared_adult, sa, grads, GRADIENTS_A, g_surr)))
    A.insert(0, "map", "Adult shared map")
    A.to_csv(os.path.join(OUTDIR, "A_shared_map_vs_gradients.csv"), index=False)

    comp = pd.DataFrame([
        {"gradient": gn,
         "r_shared": R2.corr(grads[gn], shared_adult),
         "r_PSY_component": R2.corr(grads[gn], agg["PSY mean"]),
         "r_SUD_component": R2.corr(grads[gn], agg["SUD mean"])}
        for gn in GRADIENTS_A])
    comp["opposite_signs"] = (np.sign(comp.r_PSY_component)
                              != np.sign(comp.r_SUD_component))
    comp.to_csv(os.path.join(OUTDIR, "A_component_decomposition.csv"), index=False)

    # ---------------- ANALYSIS B ----------------
    recs = []
    for name, m in rows:
        ms = surrogates_for(m, f"row_{name.replace('/', '-')}", spins, D)
        for r in test(m, ms, grads, GRADIENTS_B, g_surr):
            r["group"] = name
            recs.append(r)
    B = add_fdr(pd.DataFrame(recs))
    B = B[["group"] + [c for c in B.columns if c != "group"]]
    B.to_csv(os.path.join(OUTDIR, "B_panelD_components.csv"), index=False)

    # ---------------- report ----------------
    pd.set_option("display.width", 240)
    q = f"pFDR_{NULLS[0]}"
    print(f"\n=== A — adult SHARED map vs 5 gradients (one BH family, {len(A)} tests) ===")
    for _, r in A.iterrows():
        print(f"  {r['gradient']:<4s} r = {r['r']:+.3f}   p = {r[f'p_{NULLS[0]}']:.4f}   "
              f"q = {r[q]:.4f}  {'*' if r['robust'] else ''}")

    print("\n  its two components (descriptive, outside the family):")
    for _, r in comp.iterrows():
        flag = ("   <-- OPPOSITE SIGNS: a null here is cancellation, not absence"
                if r["opposite_signs"] else "")
        print(f"  {r['gradient']:<4s} shared {r['r_shared']:+.3f}  =  PSY "
              f"{r['r_PSY_component']:+.3f} / SUD {r['r_SUD_component']:+.3f}{flag}")

    print(f"\n=== B — panel D: 7 components vs C1-C3 (one BH family, {len(B)} tests) ===")
    for name in [r for r in ROW_ORDER if r in set(B["group"])]:
        line = f"  {name:20s}"
        for gn in GRADIENTS_B:
            s = B[(B.group == name) & (B.gradient == gn)].iloc[0]
            line += f"  {gn}: {s['r']:+.3f}{'*' if s['robust'] else ' '}(q={s[q]:.3f})"
        print(line)

    print(f"\n  A: {int(A['robust'].sum())}/{len(A)} survive   "
          f"B: {int(B['robust'].sum())}/{len(B)} survive")
    print(f"\nDone. Outputs -> {OUTDIR}")


if __name__ == "__main__":
    main()