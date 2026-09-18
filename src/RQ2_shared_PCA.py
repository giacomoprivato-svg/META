#!/usr/bin/env python3
"""
RQ2 — step 5: PC1 vs the mean-based shared map (convergence check)

"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import RQ1_common as C
import RQ2_common as R2

# ================= CONFIG — edit, then press Run =================
NULLS = ["spin", "brainsmash"]          # add "brainsmash" for the second framework
N_PERM = 10000
N_TOP = 8                 # parcels to list where the two maps disagree most
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "ALL_outputs_RQ2", "pc1_vs_shared")
CACHE = os.path.join(repo_dir, "ALL_outputs_RQ1", "_cache")
os.makedirs(OUTDIR, exist_ok=True)

DK = ["bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
      "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
      "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
      "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
      "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
      "postcentral", "posteriorcingulate", "precentral", "precuneus",
      "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
      "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
      "temporalpole", "transversetemporal", "insula"]
PARCELS = [f"L_{r}" for r in DK] + [f"R_{r}" for r in DK]


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def main():
    maps = R2.load_maps(data_dir)
    agg = R2.aggregates(maps)
    psy, sud = maps["psy"], maps["sud"]
    names = list(psy.columns) + list(sud.columns)
    domain = ["PSY"] * psy.shape[1] + ["SUD"] * sud.shape[1]

    X = np.column_stack([psy.to_numpy(float), sud.to_numpy(float)])   # (68, 15)
    if X.shape != (C.N_CORTEX, len(names)):
        raise ValueError(f"expected (68, {len(names)}), got {X.shape}")

    # standardise each MAP so a disorder with 3x the effect sizes (AN) does not
    # dominate the component through scale alone
    Xz = (X - X.mean(0)) / X.std(0)

    pca = PCA(n_components=min(5, X.shape[1]))
    scores = pca.fit_transform(Xz)
    pc1 = scores[:, 0]
    load = pca.components_[0]

    shared = agg["shared"]
    flipped = np.corrcoef(pc1, shared)[0, 1] < 0
    if flipped:
        pc1, load = -pc1, -load          # PCA polarity is arbitrary
    r_pc1_shared = R2.corr(pc1, shared)

    # --- how even are the loadings? ---
    w2 = load ** 2
    w2 = w2 / w2.sum()
    participation = 1.0 / np.sum(w2 ** 2)          # 15 = perfectly even, 1 = one map
    cv = load.std(ddof=1) / np.abs(load).mean()
    psy_share = w2[np.array(domain) == "PSY"].sum() * 100

    print(f"\nPC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance "
          f"(PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)")
    print(f"sign of PC1 {'FLIPPED' if flipped else 'kept'} to align with the shared map")
    print(f"\nr(PC1, shared map) = {r_pc1_shared:+.3f}")
    print("  Expect this to be high: for positively intercorrelated standardised")
    print("  maps the leading eigenvector has near-equal loadings, so PC1 is")
    print("  approximately their mean. The numbers below say whether that is the")
    print("  case here, i.e. whether the agreement is informative or automatic.")

    print(f"\nloading evenness: participation ratio {participation:.1f} of "
          f"{len(names)} maps, CV of loadings {cv:.3f}")
    print(f"domain weighting: PC1 gives PSY {psy_share:.1f}% of the squared loadings "
          f"(9/15 = 60.0% if perfectly even);")
    print(f"                  the mean-based shared map gives PSY 50.0% by construction")

    ld = (pd.DataFrame({"map": names, "domain": domain, "loading": load,
                        "weight_pct": w2 * 100})
          .sort_values("loading", ascending=False))
    ld.to_csv(os.path.join(OUTDIR, "PC1_loadings.csv"), index=False)
    print("\nloadings:")
    print(ld.round(3).to_string(index=False))

    # --- where do the two maps disagree? ---
    d = z(pc1) - z(shared)
    dd = (pd.DataFrame({"parcel": PARCELS, "z_PC1": z(pc1), "z_shared": z(shared),
                        "difference": d})
          .reindex(np.argsort(-np.abs(d))))
    dd.to_csv(os.path.join(OUTDIR, "PC1_vs_shared_per_parcel.csv"), index=False)
    print(f"\nlargest disagreements (both maps z-scored, so shape not scale):")
    print(dd.head(N_TOP).round(3).to_string(index=False))

    # --- nulls, for completeness ---
    stats = {"r_PC1_shared": r_pc1_shared,
             "explained_variance_PC1": pca.explained_variance_ratio_[0],
             "participation_ratio": participation,
             "loading_CV": cv, "PSY_share_pct": psy_share,
             "sign_flipped": flipped}
    if NULLS:
        spins = C.make_spins(data_dir, N_PERM, CACHE) if "spin" in NULLS else None
        D = C.distance_matrix(data_dir) if "brainsmash" in NULLS else None
        for n in NULLS:
            surr = pc1[spins] if n == "spin" else C.get_surrogates(
                pc1, D, N_PERM, "pc1_adults", CACHE)
            zz, pp = R2.z_p(r_pc1_shared, R2.corr_rows(surr, shared))
            stats[f"z_{n}"], stats[f"p_{n}"] = zz, pp
            print(f"\n{n} null: z = {zz:+.2f}, p = {pp:.4f}")
        print("  (A spatial null tests alignment, not whether the two constructions")
        print("   are algebraically related. It is a floor, not the answer.)")

    pd.DataFrame([stats]).round(6).to_csv(
        os.path.join(OUTDIR, "PC1_vs_shared_summary.csv"), index=False)
    pd.DataFrame({"PC1": pc1, "shared": shared}).to_csv(
        os.path.join(OUTDIR, "PC1_and_shared_maps.csv"), index=False)
    print(f"\nDone. Outputs -> {OUTDIR}")


if __name__ == "__main__":
    main()