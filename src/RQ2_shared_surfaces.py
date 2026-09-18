#!/usr/bin/env python3
"""
RQ2 — surfaces: adult shared map and PSY contribution index

"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

import RQ1_common as C
import RQ2_common as R2

# ================= CONFIG — edit, then press Run =================
SHARED_CMAP = "YlOrRd_r"
CONTRIB_CMAP = "PiYG"
CONTRIB_HALF_RANGE = 50    # fixed, matches the original (full 0-100% span)
SIZE = (800, 400)
ZOOM = 1.25
SCALE = (4, 4)

CBAR_FIGSIZE = (18, 1)
CBAR_FONTSIZE = 84
CBAR_LINEWIDTH = 2.5
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "figures", "surfaces")
os.makedirs(OUTDIR, exist_ok=True)


def sctx_to_16(vals14):
    """
    ENIGMA's plot_subcortical wants 16 values:
        0-6   L accumbens, amygdala, caudate, hippocampus, pallidum, putamen, thalamus
        7     L lateral ventricle
        8-14  R, same order
        15    R lateral ventricle
    Our workbooks hold the 14 structures in exactly that alphabetical order
    (ventricles already removed), matching the original script's direct
    vals16[:7]/[8:15] assignment — no reordering needed here.
    """
    v = np.full(16, np.nan)
    v[:7] = vals14[:7]
    v[8:15] = vals14[7:]
    return v


def save_colorbar(cmap, vmin, vmax, path, label_min=None, label_max=None):
    """
    Standalone colour bar, saved next to the surfaces it belongs to: a plain
    frame (no ticks), the two endpoint values only, in large text below the
    corners. This is a matplotlib reproduction of the bar style used
    throughout the figures — the surfaces themselves are drawn by VTK inside
    the ENIGMA Toolbox and carry no matplotlib-editable bar of their own.

    label_min / label_max override what is printed WITHOUT changing the
    colour mapping — used for the contribution index, whose bar spans
    -50..+50 in colour space but should read "0" / "100" (the percentage the
    index actually represents).
    """
    label_min = label_min if label_min is not None else f"{vmin:g}"
    label_max = label_max if label_max is not None else f"{vmax:g}"
    fig, ax = plt.subplots(figsize=CBAR_FIGSIZE)
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(cmap),
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        orientation="horizontal")
    cb.set_ticks([])
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(CBAR_LINEWIDTH)
    cb.ax.text(0.0, -0.55, label_min, transform=cb.ax.transAxes,
              ha="center", va="top", fontsize=CBAR_FONTSIZE)
    cb.ax.text(1.0, -0.55, label_max, transform=cb.ax.transAxes,
              ha="center", va="top", fontsize=CBAR_FONTSIZE)
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  saved {os.path.basename(path)}")


def plot_ctx(vals, name, cmap, crange):
    from enigmatoolbox.utils.parcellation import parcel_to_surface
    from enigmatoolbox.plotting import plot_cortical
    plot_cortical(array_name=parcel_to_surface(vals, "aparc_fsa5"),
                  surface_name="fsa5", cmap=cmap, color_range=crange,
                  size=SIZE, zoom=ZOOM, scale=SCALE, background=(1, 1, 1),
                  color_bar="bottom", share="b", screenshot=True,
                  filename=os.path.join(OUTDIR, f"{name}.png"))
    print(f"  saved {name}.png   cmap={cmap}   range {crange[0]:.3f} to {crange[1]:.3f}")


def plot_sctx(vals14, name, cmap, crange):
    from enigmatoolbox.plotting import plot_subcortical
    plot_subcortical(sctx_to_16(np.asarray(vals14, float)), cmap=cmap,
                     color_range=crange, size=SIZE, zoom=ZOOM, scale=SCALE,
                     background=(1, 1, 1), color_bar="bottom", share="b",
                     screenshot=True,
                     filename=os.path.join(OUTDIR, f"{name}_SCTX.png"))
    print(f"  saved {name}_SCTX.png   cmap={cmap}   range {crange[0]:.3f} to {crange[1]:.3f}")


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

    print(f"shared map (cortex, adults): {shared_ctx.min():+.3f} to {shared_ctx.max():+.3f}")
    print(f"shared map (subcortex, adults): {shared_sub.min():+.3f} to {shared_sub.max():+.3f}")
    print(f"contribution index (cortex): {contrib_ctx.min():.1f}% to "
          f"{contrib_ctx.max():.1f}%, median {np.median(contrib_ctx):.1f}%")
    print(f"contribution index (subcortex): {contrib_sub.min():.1f}% to "
          f"{contrib_sub.max():.1f}%, median {np.median(contrib_sub):.1f}%")

    vmin = float(min(shared_ctx.min(), shared_sub.min()))
    if shared_ctx.max() > 0 or shared_sub.max() > 0:
        print("  note: the shared map has POSITIVE values; the (vmin, 0) scale "
              "clips them to the top colour.")

    print(f"\nrendering shared map (YlOrRd_r, {vmin:.3f}..0) ...")
    plot_ctx(shared_ctx, "SHARED_adults", SHARED_CMAP, (vmin, 0.0))
    plot_sctx(shared_sub, "SHARED_adults", SHARED_CMAP, (vmin, 0.0))
    save_colorbar(SHARED_CMAP, vmin, 0.0,
                  os.path.join(OUTDIR, "SHARED_adults_colorbar.png"),
                  label_min=f"{vmin:.3f}", label_max="0")

    print(f"\nrendering contribution index (PiYG, fixed +/-{CONTRIB_HALF_RANGE} "
          f"around 50%, plotted as contribution-50) ...")
    plot_ctx(contrib_ctx - 50, "CONTRIBUTION_adults", CONTRIB_CMAP,
             (-CONTRIB_HALF_RANGE, CONTRIB_HALF_RANGE))
    plot_sctx(contrib_sub - 50, "CONTRIBUTION_adults", CONTRIB_CMAP,
              (-CONTRIB_HALF_RANGE, CONTRIB_HALF_RANGE))
    # the bar spans -50..+50 in colour space but represents 0%..100%
    save_colorbar(CONTRIB_CMAP, -CONTRIB_HALF_RANGE, CONTRIB_HALF_RANGE,
                  os.path.join(OUTDIR, "CONTRIBUTION_adults_colorbar.png"),
                  label_min="0", label_max="100")

    print(f"\nDone. Surfaces -> {OUTDIR}")


if __name__ == "__main__":
    main()