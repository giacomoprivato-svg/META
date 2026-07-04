#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# DATA (Table S2)
# -----------------------------
data = [
    ["Control", 1951, 18, 75, 33.8, 12.0, 37.5],
    ["ALC", 1162, 18, 67, 37.7, 11.9, 29.1],
    ["COC", 404, 18, 60, 38.2, 9.0, 24.3],
    ["NIC", 652, 18, 55, 30.3, 9.7, 43.7],
    ["ATS", 184, 18, 54, 31.9, 10.0, 37.0],
    ["CAN", 288, 18, 57, 28.5, 9.2, 34.4],
    ["OPI", 92, 20, 58, 35.7, 9.5, 41.3],
]

df = pd.DataFrame(
    data,
    columns=["Group", "N", "Age_min", "Age_max", "Age_mean", "Age_SD", "Female_pct"]
)

# -----------------------------
# COLORS (aligned with PSY)
# -----------------------------
control_color = "#C9C9C9"
case_color = "#9FB7D5"
female_color = "#CBB6D6"
male_color = "#B7C7A3"
dark = "#111111"

x = np.arange(len(df))

# -----------------------------
# FIGURE
# -----------------------------
fig = plt.figure(figsize=(8.5, 7.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.1], hspace=0.45)

# =============================
# PANEL A — AGE
# =============================
ax1 = fig.add_subplot(gs[0])

for i, row in df.iterrows():

    color = control_color if row["Group"] == "Control" else case_color

    # range
    ax1.bar(
        i,
        row["Age_max"] - row["Age_min"],
        bottom=row["Age_min"],
        width=0.6,
        color=color,
        edgecolor=dark,
        linewidth=0.8
    )

    # mean
    ax1.scatter(
        i,
        row["Age_mean"],
        color=dark,
        s=30,
        zorder=3
    )

# labels
ax1.set_ylabel(
    "Age (years)",
    fontsize=16,
    color=dark
)

ax1.set_xticks([])

ax1.tick_params(
    axis="y",
    labelsize=13,
    colors=dark
)

# scale allineata a PSY
ax1.set_ylim(0, 80)
ax1.set_xlim(-0.6, len(df) - 0.4)

# style
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_color(dark)
ax1.spines["left"].set_color(dark)

# =============================
# PANEL B — SEX
# =============================
ax2 = fig.add_subplot(gs[1])

female = df["Female_pct"].values
male = 100 - female

ax2.bar(
    x,
    female,
    width=0.6,
    color=female_color,
    label="Female"
)

ax2.bar(
    x,
    male,
    bottom=female,
    width=0.6,
    color=male_color,
    label="Male"
)

ax2.set_ylabel(
    "Sex (%)",
    fontsize=16,
    color=dark
)

ax2.set_ylim(0, 100)
ax2.set_xlim(-0.6, len(df) - 0.4)

ax2.set_xticks(x)

# 🔥 label più grandi come nel codice precedente
ax2.set_xticklabels(
    df["Group"],
    fontsize=17,
    color=dark
)

ax2.tick_params(
    axis="y",
    labelsize=12,
    colors=dark
)

# style
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color(dark)
ax2.spines["left"].set_color(dark)

# legend
ax2.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=2,
    fontsize=13
)

# -----------------------------
# SAVE (PNG ONLY)
# -----------------------------
plt.savefig(
    "SUD_demographics.png",
    dpi=600,
    bbox_inches="tight"
)

