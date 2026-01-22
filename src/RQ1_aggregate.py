#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

# =========================
# CONFIG
# =========================
BASE_DIR = "ALL_outputs_RQ1"
GROUPS = [
    "adults_all",
    "adolescents_all",
    "adults_ctx",
    "adolescents_ctx"
]

COMPARTMENTS = ["cortex", "subctx"]
MEASURE_FOR_NULLS = "euclidean"   # we only stack euclidean nulls

# =========================
# Helpers
# =========================
def parse_filename(fname):
    """
    Extract psy and sud names from filenames like:
    raw_psy_cortex_PSYNAME_SUDNAME.csv
    """
    parts = fname.replace(".csv", "").split("_")
    psy = parts[-2]
    sud = parts[-1]
    return psy, sud

# =========================
# Main
# =========================
for group in GROUPS:
    print(f"\n== Aggregating group: {group} ==")
    gdir = os.path.join(BASE_DIR, group)
    if not os.path.isdir(gdir):
        print(f"  Skipping {group} (not found)")
        continue

    for comp in COMPARTMENTS:
        print(f"  -> {comp}")

        # -------------------------
        # Discover files
        # -------------------------
        raw_psy_files = sorted([f for f in os.listdir(gdir) if f.startswith(f"raw_psy_{comp}_")])
        raw_sud_files = sorted([f for f in os.listdir(gdir) if f.startswith(f"raw_sud_{comp}_")])
        null_files    = sorted([f for f in os.listdir(gdir) if f.startswith(f"nulls_{comp}_")])

        if len(raw_psy_files) == 0:
            print(f"     No files found for {comp}")
            continue

        # -------------------------
        # Identify PSY & SUD names
        # -------------------------
        psy_names = sorted(set(parse_filename(f)[0] for f in raw_psy_files))
        sud_names = sorted(set(parse_filename(f)[1] for f in raw_psy_files))

        # Load one file to get n_regions / n_perm
        example_raw = pd.read_csv(os.path.join(gdir, raw_psy_files[0]))
        n_regions = example_raw.shape[0]

        example_nulls = pd.read_csv(os.path.join(gdir, null_files[0]))
        n_perm = example_nulls.shape[0]

        # -------------------------
        # Allocate arrays
        # -------------------------
        RAW_PSY = np.zeros((n_regions, len(psy_names)))
        RAW_SUD = np.zeros((n_regions, len(sud_names)))
        NULLS_EUC = np.zeros((len(psy_names), len(sud_names), n_perm))

        psy_idx = {p:i for i,p in enumerate(psy_names)}
        sud_idx = {s:i for i,s in enumerate(sud_names)}

        # -------------------------
        # Fill arrays
        # -------------------------
        for fname in tqdm(raw_psy_files, desc=f"     stacking {comp}"):
            psy, sud = parse_filename(fname)
            i = psy_idx[psy]
            j = sud_idx[sud]

            psy_vals = pd.read_csv(os.path.join(gdir, fname)).iloc[:,0].to_numpy()
            sud_vals = pd.read_csv(os.path.join(gdir, f"raw_sud_{comp}_{psy}_{sud}.csv")).iloc[:,0].to_numpy()
            nulls = pd.read_csv(os.path.join(gdir, f"nulls_{comp}_{psy}_{sud}.csv"))

            RAW_PSY[:, i] = psy_vals
            RAW_SUD[:, j] = sud_vals
            NULLS_EUC[i, j, :] = nulls[MEASURE_FOR_NULLS].to_numpy()

        # -------------------------
        # Save aggregated outputs
        # -------------------------
        pd.DataFrame(RAW_PSY, columns=psy_names).to_csv(
            os.path.join(gdir, f"RAW_PSY_{comp}_matrix.csv"),
            index=False
        )

        pd.DataFrame(RAW_SUD, columns=sud_names).to_csv(
            os.path.join(gdir, f"RAW_SUD_{comp}_matrix.csv"),
            index=False
        )

        np.save(
            os.path.join(gdir, f"NULLS_{comp}_{MEASURE_FOR_NULLS}.npy"),
            NULLS_EUC
        )

        # Optional: mean null (handy for sanity checks)
        mean_null = NULLS_EUC.mean(axis=2)
        pd.DataFrame(mean_null, index=psy_names, columns=sud_names).to_csv(
            os.path.join(gdir, f"NULLS_{comp}_{MEASURE_FOR_NULLS}_mean.csv")
        )

        print(f"     Saved aggregated matrices for {comp}")

print("\n✅ Aggregation complete.")
