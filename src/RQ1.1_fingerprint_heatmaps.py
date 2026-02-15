#!/usr/bin/env python3
"""
RQ1 – Metric profiles across PSY–SUD pairs
X-axis: PSY–SUD labels
Y-axis: Z-score similarity
Lines connect pairs belonging to the same metric
Cortex only
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PATHS
# -----------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)

MAIN_RESULTS = os.path.join(REPO_DIR, "ALL_outputs_RQ1")
OUT_DIR = os.path.join(REPO_DIR, "metric_profiles_RQ1")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = {
    "Adults": ["adults_all"],
    "Pediatric": ["adolescents_all", "adolescents_ctx"],
}

MEASURES = ["euclidean", "spearman", "cosine"]

# -----------------------------
# HELPERS
# -----------------------------
def load_pairs_with_labels(group_folders):
    Z = {}
    labels = None

    for m in MEASURES:
        mats = []
        labs = []

        for g in group_folders:
            f = os.path.join(MAIN_RESULTS, g, f"Z_cortex_{m}.csv")
            df = pd.read_csv(f, index_col=0)

            if labels is None:
                psy = df.index.astype(str)
                sud = df.columns.astype(str)
                labs.extend([f"{p}–{s}" for p in psy for s in sud])

            mats.append(df.values)

        Z[m] = np.vstack(mats).flatten()

    return Z, labels or labs

# -----------------------------
# MAIN
# -----------------------------
for label, folders in GROUPS.items():

    print(f"Plotting {label}")
    Z, pair_labels = load_pairs_with_labels(folders)

    # Sort by Euclidean similarity (recommended)
    order = np.argsort(Z["euclidean"])
    x = np.arange(len(order))

    plt.figure(figsize=(max(14, len(order)*0.25), 5))

    for m in MEASURES:
        y = Z[m][order]
        plt.plot(x, y, marker="o", linewidth=1, markersize=3, label=m)

    plt.xticks(
        x,
        np.array(pair_labels)[order],
        rotation=90,
        fontsize=6
    )

    plt.xlabel("Psychiatric disorder – SUD pairs")
    plt.ylabel("Z-score similarity")
    plt.title(f"RQ1 metric profiles across PSY–SUD pairs – {label} (cortex)")
    plt.legend(fontsize="small", ncol=3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT_DIR, f"MetricProfiles_RQ1_{label}_cortex.png"),
        dpi=300
    )
    plt.close()

print("✅ Done")

