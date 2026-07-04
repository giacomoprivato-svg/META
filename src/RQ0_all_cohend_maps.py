import os
import numpy as np
import pandas as pd

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical, plot_surf


# ---------------------------
# Paths
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
data_dir = os.path.join(repo_dir, "data", "raw")

main_outdir = os.path.join(repo_dir, "ALL_outputs_RQ1")
cortex_outdir = os.path.join(main_outdir, "cortex")
subcortex_outdir = os.path.join(main_outdir, "subcortex")
colorbar_outdir = os.path.join(main_outdir, "colorbars")  # 🔥 NEW

os.makedirs(main_outdir, exist_ok=True)
os.makedirs(cortex_outdir, exist_ok=True)
os.makedirs(subcortex_outdir, exist_ok=True)
os.makedirs(colorbar_outdir, exist_ok=True)


# ---------------------------
# Files
# ---------------------------
GROUPS = [
    ("adults_all", "PSY_adults.xlsx"),
    ("adolescents_all", "PSY_adolescents.xlsx"),
    ("adolescents_ctx", "PSY_adolescents_ctx.xlsx"),
]

SUD_FILE = os.path.join(data_dir, "SUD.xlsx")

# ---------------------------
# Constants
# ---------------------------
N_SUBCORT = 14


# ===========================
# FIXED: match second script
# ===========================
PLOT_KW = dict(
    cmap="RdBu_r",
    size=(800, 400),
    zoom=1.25,
    scale=(4, 4),
    background=(1, 1, 1),
    share="b",
    color_bar="bottom"
)

PLOT_KW_SCTX = dict(
    cmap="RdBu_r",
    size=(800, 400),
    zoom=1.25,
    scale=(4, 4),
    background=(1, 1, 1),
    share="b",
    color_bar="bottom"
)

# 🔥 NEW: colorbar-only settings (10:1 ratio)
PLOT_KW_CBAR = dict(
    cmap="RdBu_r",
    size=(1000, 100),   # <- orizzontale corta (≈10:1)
    zoom=1,
    scale=(1, 1),
    background=(1, 1, 1),
    color_bar="bottom",
    share="b",
    actor_kwds=dict(opacity=0),   # nasconde cervello
    cb_kwds=dict(labelFontSize=26, titleFontSize=28)
)


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter  # 🔥 NEW


def save_colorbar_only(vmin, vmax, outpath):

    fig, ax = plt.subplots(figsize=(3.3, 1))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal'
    )

    cb.set_ticks([vmin, vmax])

    # 🔥 max 2 decimali
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 🔥 numeri grandi
    cb.ax.tick_params(labelsize=28)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
# ---------------------------
# Helpers
# ---------------------------
def read_excel_numeric(path):
    df = pd.read_excel(path)
    return df.apply(pd.to_numeric, errors="coerce")


def split_ctx_sctx(df, group_name):
    if group_name == "adolescents_ctx":
        return df.copy(), None

    ctx = df.iloc[:-N_SUBCORT, :].copy()
    sctx = df.iloc[-N_SUBCORT:, :].copy()
    return ctx, sctx


def pad_subcortex(data14):
    padded = np.full(16, np.nan)
    padded[:7] = data14[:7]
    padded[8:15] = data14[7:]
    return padded


def compute_vmax(vals):
    vals = np.array(vals, dtype=float)
    vmax = np.nanmax(np.abs(vals)) if np.any(np.isfinite(vals)) else 0.1
    return np.round(vmax, 3)


# ---------------------------
# RUN GROUP
# ---------------------------
def run_group(group_name, df, out_prefix):

    ctx_df, sctx_df = split_ctx_sctx(df, group_name)

    print(f"\nProcessing: {group_name}")

    for col in df.columns:

        safe_name = f"{out_prefix}_{str(col).replace(' ', '_')}"

        vals_ctx = None
        vals_sctx = None

        # ---------------------------
        # CORTEX
        # ---------------------------
        if ctx_df is not None:

            vals_ctx = ctx_df[col].to_numpy(dtype=float)

            if len(vals_ctx) != 68:
                print(f"⚠️ Skip cortex {group_name} - {col}")
                continue

            surf_vals = parcel_to_surface(vals_ctx, "aparc_fsa5")

        # ---------------------------
        # SUBCORTEX
        # ---------------------------
        if sctx_df is not None:

            vals_sctx = sctx_df[col].to_numpy(dtype=float)

            if len(vals_sctx) == 14:
                vals_sctx = pad_subcortex(vals_sctx)

        # ---------------------------
        # UNIFIED SCALING
        # ---------------------------
        combined_vals = []

        if vals_ctx is not None:
            combined_vals.append(vals_ctx)

        if vals_sctx is not None:
            combined_vals.append(vals_sctx[~np.isnan(vals_sctx)])

        combined_vals = np.concatenate(combined_vals)
        vmax = compute_vmax(combined_vals)
        color_range = (-vmax, vmax)

        # 🔥 NEW: SAVE COLORBAR IMAGE
        save_colorbar_only(
            -vmax,
            vmax,
            os.path.join(colorbar_outdir, f"{safe_name}_colorbar.png")
        )

        # ---------------------------
        # PLOT CORTEX
        # ---------------------------
        if vals_ctx is not None:

            plot_cortical(
                array_name=surf_vals,
                surface_name="fsa5",
                color_range=color_range,
                screenshot=True,
                filename=os.path.join(cortex_outdir, f"{safe_name}.png"),
                **PLOT_KW
            )

        # ---------------------------
        # PLOT SUBCORTEX
        # ---------------------------
        if vals_sctx is not None:

            plot_subcortical(
                vals_sctx,
                color_range=color_range,
                screenshot=True,
                filename=os.path.join(subcortex_outdir, f"{safe_name}.png"),
                **PLOT_KW_SCTX
            )


# ---------------------------
# RUN MAIN GROUPS
# ---------------------------
print(f"Saving outputs to: {main_outdir}")

for group_name, filename in GROUPS:
    path = os.path.join(data_dir, filename)
    df = read_excel_numeric(path)
    run_group(group_name, df, group_name)


# ---------------------------
# RUN SUD
# ---------------------------
print("\nProcessing: SUD")

df_sud = read_excel_numeric(SUD_FILE)
run_group("SUD", df_sud, "SUD")

print("\n✅ Done: all groups + SUD processed.")