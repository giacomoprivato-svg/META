#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

# ---------------------------
# Paths (repo-relative)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)


# ======================
# PATHS
# ======================
BASE = os.path.join(repo_dir, "data", "raw")

OUTDIR = os.path.join(repo_dir, "results", "MEAN_MAPS_GLOBAL_CORRECT")
CORTEX_DIR = os.path.join(OUTDIR, "cortex")
SCTX_DIR = os.path.join(OUTDIR, "subcortex")
CB_DIR = os.path.join(OUTDIR, "colorbars")

os.makedirs(CORTEX_DIR, exist_ok=True)
os.makedirs(SCTX_DIR, exist_ok=True)
os.makedirs(CB_DIR, exist_ok=True)

# ======================
# FILES
# ======================
PATH_ADULTS = os.path.join(BASE, "PSY_adults.xlsx")
PATH_ADO = os.path.join(BASE, "PSY_adolescents.xlsx")
PATH_ADO_CTX = os.path.join(BASE, "PSY_adolescents_ctx.xlsx")
PATH_SUD = os.path.join(BASE, "SUD.xlsx")

# ======================
# CONSTANTS
# ======================
N_CORTEX = 68
N_SUBCORT = 14

# ======================
# STYLE
# ======================
PLOT_KW = dict(
    cmap="RdBu_r",
    size=(800, 400),
    zoom=1.25,
    scale=(4, 4),
    background=(1, 1, 1),
    share="b",
    color_bar="bottom"
)

# ======================
# LOAD
# ======================
def load(path):
    df = pd.read_excel(path).select_dtypes(include=np.number)
    return df.to_numpy(), df.columns.tolist()

def mean_map(X):
    return np.nanmean(X, axis=1)

# ======================
# COLORBAR
# ======================
def save_colorbar(vmin, vmax, path):
    fig, ax = plt.subplots(figsize=(3.3, 1))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal"
    )

    cb.set_ticks([vmin, vmax])
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cb.ax.tick_params(labelsize=28)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ======================
# LOAD DATA
# ======================
PSY_A, _ = load(PATH_ADULTS)
PSY_P, _ = load(PATH_ADO)
PSY_P_CTX, _ = load(PATH_ADO_CTX)
SUD, _ = load(PATH_SUD)

# ======================
# FIXED ADOLESCENT STRUCTURE
# ======================

# ---- cortex: merge 4 + 2 disorders ----
ado_cortex_4 = PSY_P[:N_CORTEX, :]
ado_cortex_2 = PSY_P_CTX  # already 68 × 2

ADO_CORTEX = np.concatenate([ado_cortex_4, ado_cortex_2], axis=1)  # 68 × 6

# ---- subcortex: ONLY original adolescents ----
ADO_SUBCORT = PSY_P[N_CORTEX:, :]  # 14 × 4

# ======================
# GLOBAL RANGE (COHERENT)
# ======================
print("Computing global range...")

all_maps = []

for mat in [PSY_A, PSY_P, SUD]:

    m = mean_map(mat)
    all_maps.append(m)

# cortex adolescents (only cortex part matters for range consistency)
all_maps.append(mean_map(np.vstack([ADO_CORTEX, np.zeros((N_SUBCORT, ADO_CORTEX.shape[1]))])))

all_maps = np.concatenate(all_maps)

vmax = np.nanmax(np.abs(all_maps))
vmin = -vmax

print(f"Global range: {vmin:.3f} to {vmax:.3f}")

# ======================
# RUN FUNCTION
# ======================
def run_group(name, cortex_mat, subcortex_mat):

    print(f"\nProcessing {name}")

    # mean maps
    m_cortex = mean_map(cortex_mat)
    m_subcortex = mean_map(subcortex_mat)

    # ------------------
    # COLORBAR
    # ------------------
    save_colorbar(
        vmin, vmax,
        os.path.join(CB_DIR, f"{name}_colorbar.png")
    )

    # ------------------
    # CORTEX
    # ------------------
    surf = parcel_to_surface(m_cortex, "aparc_fsa5")

    plot_cortical(
        array_name=surf,
        surface_name="fsa5",
        color_range=(vmin, vmax),
        screenshot=True,
        filename=os.path.join(CORTEX_DIR, f"{name}_CTX.png"),
        **PLOT_KW
    )

    # ------------------
    # SUBCORTEX
    # ------------------
    if len(m_subcortex) == N_SUBCORT:
        padded = np.full(16, np.nan)
        padded[:7] = m_subcortex[:7]
        padded[8:15] = m_subcortex[7:]

        plot_subcortical(
            padded,
            color_range=(vmin, vmax),
            screenshot=True,
            filename=os.path.join(SCTX_DIR, f"{name}_SCTX.png"),
            **PLOT_KW
        )

# ======================
# RUN
# ======================
run_group("PSY_adults", PSY_A[:N_CORTEX, :], PSY_A[N_CORTEX:, :])

run_group("PSY_adolescents", ADO_CORTEX, ADO_SUBCORT)

run_group("SUD", SUD[:N_CORTEX, :], SUD[N_CORTEX:, :])

print("\nDONE ✔")