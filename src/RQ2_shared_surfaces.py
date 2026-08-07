#!/usr/bin/env python3
"""
RQ2 — surfaces: adult shared map and PSY contribution index
===========================================================

Four renders, nothing else:
    SHARED_adults           cortical  (68 parcels)
    SHARED_adults_SCTX      subcortical (14 structures)
    CONTRIBUTION_adults     cortical
    CONTRIBUTION_adults_SCTX subcortical

Extracted from RQ2_shared_maps_pspin.py and RQ2_shared_PCA_corr.py, which mixed
statistics and VTK rendering in the same file — so the whole analysis needed
the ENIGMA Toolbox installed just to produce numbers. The statistics now live
in RQ2_step1_shared_maps.py, which runs anywhere; this script is the only one
that imports enigmatoolbox.

Render settings (size, zoom, scale, background, colour bar placement) are
unchanged from the originals so the output drops into the existing figure.

VALUES ARE RECOMPUTED FROM THE RAW DATA, NOT READ FROM step 1
--------------------------------------------------------------
Deliberate: a surface is what a reader actually looks at, so it should not be
able to disagree with the analysis because someone regenerated one CSV and not
the other. Both this script and step 1 build the maps through RQ2_common, so
the SUD aggregate is dropped in both and PD is in both. If you would rather
plot exactly the CSV step 1 wrote, set FROM_STEP1 = True — the script then
reads it and asserts it matches what it would have computed.

COLOUR SCALES
-------------
The shared map is negative almost everywhere (thinning), so it keeps the
original one-sided YlOrRd_r from vmin to 0 rather than a diverging map centred
on zero, which would waste half the colour range.

The contribution index is a percentage that lives around 50, not around 0, so
it is drawn diverging and CENTRED ON 50. Plotting it centred on zero — which
is what a naive reuse of the cortical helper would do — would render every
region as the same saturated colour and show nothing at all.

Just press Run. Requires the ENIGMA Toolbox.
"""

import os
import numpy as np
import pandas as pd

import RQ1_common as C
import RQ2_common as R2

# ================= CONFIG — edit, then press Run =================
FROM_STEP1 = False        # True = read step 1's CSVs and cross-check
SHARED_CMAP = "YlOrRd_r"
CONTRIB_CMAP = "RdBu_r"
CONTRIB_HALF_RANGE = None  # None = symmetric around 50 at the observed extreme;
                           # set e.g. 25 to fix the scale at 25-75%
SIZE = (800, 400)
ZOOM = 1.25
SCALE = (4, 4)
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
STEP1DIR = os.path.join(repo_dir, "ALL_outputs_RQ2", "shared_maps")
OUTDIR = os.path.join(repo_dir, "figures", "surfaces")
os.makedirs(OUTDIR, exist_ok=True)


def sctx_to_16(vals14):
    """
    ENIGMA's plot_subcortical wants 16 values:
        0-6   L accumbens, amygdala, caudate, hippocampus, pallidum, putamen, thalamus
        7     L lateral ventricle
        8-14  R, same order
        15    R lateral ventricle
    Our workbooks hold the 14 structures in exactly that alphabetical order with
    the ventricles already removed, so the two ventricle slots are filled with
    NaN and everything else drops straight in.
    """
    v = np.full(16, np.nan)
    v[:7] = vals14[:7]
    v[8:15] = vals14[7:]
    return v


def plot_ctx(vals, name, cmap, crange):
    from enigmatoolbox.utils.parcellation import parcel_to_surface
    from enigmatoolbox.plotting import plot_cortical
    plot_cortical(array_name=parcel_to_surface(vals, "aparc_fsa5"),
                  surface_name="fsa5", cmap=cmap, color_range=crange,
                  size=SIZE, zoom=ZOOM, scale=SCALE, background=(1, 1, 1),
                  color_bar="bottom", share="b", screenshot=True,
                  filename=os.path.join(OUTDIR, f"{name}.png"))
    print(f"  saved {name}.png   range {crange[0]:.3f} to {crange[1]:.3f}")


def plot_sctx(vals14, name, cmap, crange):
    from enigmatoolbox.plotting import plot_subcortical
    plot_subcortical(sctx_to_16(np.asarray(vals14, float)), cmap=cmap,
                     color_range=crange, size=SIZE, zoom=ZOOM, scale=SCALE,
                     background=(1, 1, 1), color_bar="bottom", share="b",
                     screenshot=True,
                     filename=os.path.join(OUTDIR, f"{name}_SCTX.png"))
    print(f"  saved {name}_SCTX.png   range {crange[0]:.3f} to {crange[1]:.3f}")


def main():
    maps = R2.load_maps(data_dir)
    agg = R2.aggregates(maps)

    psy_ctx = agg["PSY mean"]
    sud_ctx = agg["SUD mean"]
    shared_ctx = agg["shared"]
    contrib_ctx = np.abs(psy_ctx) / (np.abs(psy_ctx) + np.abs(sud_ctx)) * 100

    psy_sub = C.subcortical_rows(os.path.join(data_dir, "PSY_adults.xlsx"))[0] \
        .mean(axis=1).to_numpy(float)
    sud_sub = C.subcortical_rows(os.path.join(data_dir, "SUD.xlsx"),
                                 drop_cols=["SUD"])[0].mean(axis=1).to_numpy(float)
    shared_sub = (psy_sub + sud_sub) / 2
    contrib_sub = np.abs(psy_sub) / (np.abs(psy_sub) + np.abs(sud_sub)) * 100

    if FROM_STEP1:
        f = os.path.join(STEP1DIR, "SHARED_maps_cortex.csv")
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f}\nRun RQ2_step1_shared_maps.py first, "
                                    f"or set FROM_STEP1 = False.")
        ref = pd.read_csv(f)["Adults"].to_numpy(float)
        d = np.max(np.abs(ref - shared_ctx))
        if d > 1e-9:
            raise AssertionError(
                f"the shared map on disk differs from the one computed here "
                f"(max |diff| = {d:.2e}). One of the two was generated from "
                f"different inputs — do not plot either until you know which.")
        print(f"  cross-check against step 1 OK (max diff {d:.1e})")

    print(f"\nshared map (cortex): {shared_ctx.min():+.3f} to {shared_ctx.max():+.3f}")
    print(f"shared map (subcortex): {shared_sub.min():+.3f} to {shared_sub.max():+.3f}")
    print(f"contribution index (cortex): {contrib_ctx.min():.1f}% to "
          f"{contrib_ctx.max():.1f}%, median {np.median(contrib_ctx):.1f}%")
    print(f"contribution index (subcortex): {contrib_sub.min():.1f}% to "
          f"{contrib_sub.max():.1f}%, median {np.median(contrib_sub):.1f}%")

    if shared_ctx.max() > 0:
        print("  note: the shared cortical map has POSITIVE values; the one-sided "
              "YlOrRd_r scale ends at 0 and those parcels will clip.")

    # ---- shared map: one-sided, vmin..0 ----
    v = float(np.floor(min(shared_ctx.min(), shared_sub.min()) * 100) / 100)
    print("\nrendering shared map ...")
    plot_ctx(shared_ctx, "SHARED_adults", SHARED_CMAP, (v, 0.0))
    plot_sctx(shared_sub, "SHARED_adults", SHARED_CMAP, (v, 0.0))

    # ---- contribution index: diverging, centred on 50 ----
    allc = np.concatenate([contrib_ctx, contrib_sub])
    half = CONTRIB_HALF_RANGE or float(np.ceil(np.max(np.abs(allc - 50))))
    crange = (50 - half, 50 + half)
    print("\nrendering contribution index (centred on 50%, not on 0) ...")
    plot_ctx(contrib_ctx, "CONTRIBUTION_adults", CONTRIB_CMAP, crange)
    plot_sctx(contrib_sub, "CONTRIBUTION_adults", CONTRIB_CMAP, crange)

    print(f"\nDone. Surfaces -> {OUTDIR}")


if __name__ == "__main__":
    main()