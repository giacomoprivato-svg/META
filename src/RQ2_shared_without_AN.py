#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical

# ---------------------------
# Paths (repo-relative)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)


np.random.seed(42)

# ======================
# PATHS
# ======================
BASE = os.path.join(repo_dir, "data", "raw")
OUTDIR = os.path.join(repo_dir, "results", "SUPPLEMENT_AN_EFFECT")
os.makedirs(OUTDIR, exist_ok=True)

PATH_SUD = os.path.join(BASE, "SUD.xlsx")
PATH_PSY_ADULTS = os.path.join(BASE, "PSY_adults.xlsx")

N_CORTEX = 68

# ======================
# LOAD DATA
# ======================
def load(path):
    df = pd.read_excel(path).select_dtypes(include=np.number)
    return df.to_numpy(), df.columns.tolist()

# ======================
# MEAN EFFECT
# ======================
def mean_effect(X):
    return np.nanmean(X, axis=1)

# ======================
# SHARED MAP
# ======================
def shared_alteration_map(psy_mat, sud_vec):
    psy = mean_effect(psy_mat)
    sud = mean_effect(sud_vec)
    shared = (psy + sud) / 2
    return shared

# ======================
# PLOT (IDENTICAL SETTINGS)
# ======================
def plot_brain(vals, filename, vmax, cmap="YlOrRd_r", center_zero=True):

    CT = parcel_to_surface(vals, "aparc_fsa5")

    crange = (-vmax, 0) if center_zero else (np.nanmin(vals), np.nanmax(vals))

    plot_cortical(
        array_name=CT,
        surface_name="fsa5",
        cmap=cmap,
        color_range=crange,
        size=(800, 400),
        zoom=1.25,
        scale=(4,4),
        background=(1, 1, 1),
        color_bar="bottom",
        share="b",
        screenshot=True,
        filename=os.path.join(OUTDIR, filename)
    )

# ======================
# MAIN
# ======================
def main():

    # load
    psy_mat, psy_names = load(PATH_PSY_ADULTS)
    sud_mat, sud_names = load(PATH_SUD)

    psy_mat = psy_mat[:N_CORTEX]
    sud_mat = sud_mat[:N_CORTEX]

    # ======================
    # FULL PSY
    # ======================
    shared_all = shared_alteration_map(psy_mat, sud_mat)

    # ======================
    # REMOVE AN_OCD
    # ======================
    an_idx = psy_names.index("AN") if "AN" in psy_names else None
    ocd_idx = psy_names.index("OCD") if "OCD" in psy_names else None

    drop_idx = [i for i in [an_idx, ocd_idx] if i is not None]

    mask = np.ones(len(psy_names), dtype=bool)
    mask[drop_idx] = False

    psy_no_an = psy_mat[:, mask]

    shared_no_an = shared_alteration_map(psy_no_an, sud_mat)

    # ======================
    # SCALE (shared max across BOTH maps)
    # ======================
    # FIXED COLOR RANGE
    vmax = 0.385

    # ======================
    # PLOTS
    # ======================
    plot_brain(
        shared_all,
        "SUPP_ALL_PSY_SUD.png",
        vmax=vmax
    )

    plot_brain(
        shared_no_an,
        "SUPP_PSY_without_AN_SUD.png",
        vmax=vmax
    )

    print("DONE ✔ Supplementary AN comparison saved")

# ======================
if __name__ == "__main__":
    main()