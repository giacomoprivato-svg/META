#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Recreate standardized table
# -----------------------------
data = [
    ["ADHD cortical", 2246, 19.22, 11.31, 25.9],
    ["ADHD child", 2707, 10.11, 0.57, 50.6],
    ["Conduct disorder", 1185, 13.71, 3.01, 339/1185*100],
    ["OCD cortical adult", 1498, 31.31, 9.74, 49.9],
    ["OCD cortical pediatric", 407, 13.69, 2.58, 47.2],
    ["BD cortical", 2447, 38.40, 10.90, 41.3],
    ["MDD cortical adult", 1911, 43.20, 12.60, 64.5],
    ["MDD cortical adolescent", 237, 19.10, 1.80, 68.4],
    ["Autism spectrum disorder", 1571, 15.41, 8.64, 14.3],
    ["Anorexia nervosa", 685, 21.00, 5.50, 100.0],
    ["Schizophrenia cortical", 4474, 32.30, 10.00, 34.0],
    ["PTSD", 1426, 39.60, np.nan, 28.7],
    ["Clinical high risk", 1792, 20.80, 5.90, 46.4],
]

df = pd.DataFrame(
    data,
    columns=["Group", "N", "Mean_Age", "SD_Age", "Female_pct"]
)

# -----------------------------
# Add aggregated controls
# -----------------------------
controls = pd.DataFrame([{
    "Group": "Aggregated controls",
    "N": 43291,
    "Mean_Age": 33.067847,
    "SD_Age": 14.604556,
    "Female_pct": 48.366762
}])

df = pd.concat([controls, df], ignore_index=True)

# -----------------------------
# Age ranges
# -----------------------------
range_map = {
    "Aggregated controls": (2, 83),
    "ADHD cortical": (4, 62),
    "ADHD child": (8.5, 11.9),
    "Conduct disorder": (7, 21),
    "OCD cortical adult": (np.nan, np.nan),
    "OCD cortical pediatric": (np.nan, np.nan),
    "BD cortical": (np.nan, np.nan),
    "MDD cortical adult": (np.nan, np.nan),
    "MDD cortical adolescent": (np.nan, np.nan),
    "Autism spectrum disorder": (2, 64),
    "Anorexia nervosa": (15, 27),
    "Schizophrenia cortical": (np.nan, np.nan),
    "PTSD": (17, 83),
    "Clinical high risk": (10, 39),
}

df["Age_min"] = df["Group"].map(lambda g: range_map[g][0])
df["Age_max"] = df["Group"].map(lambda g: range_map[g][1])

# -----------------------------
# Order
# -----------------------------
order = [
    "Aggregated controls",
    "Anorexia nervosa",
    "OCD cortical adult",
    "OCD cortical pediatric",
    "MDD cortical adult",
    "MDD cortical adolescent",
    "PTSD",
    "ADHD cortical",
    "ADHD child",
    "Autism spectrum disorder",
    "Conduct disorder",
    "BD cortical",
    "Clinical high risk",
    "Schizophrenia cortical",
]

df_ordered = df.set_index("Group").loc[order].reset_index()

labels = [
    "Controls",
    "AN",
    "OCD\nad",
    "OCD\nped",
    "MDD\nad",
    "MDD\nped",
    "PTSD",
    "ADHD",
    "ADHD\nchild",
    "ASD",
    "CD",
    "BD",
    "CHR",
    "SCZ"
]

x = np.arange(len(df_ordered))

# -----------------------------
# Colors
# -----------------------------
case_color = "#9FB7D5"
control_color = "#C9C9C9"
female_color = "#CBB6D6"
male_color = "#B7C7A3"
dark = "#111111"

# -----------------------------
# Figure
# -----------------------------
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.2], hspace=0.5)

# =============================
# PANEL A — AGE
# =============================
ax1 = fig.add_subplot(gs[0])

for i, row in df_ordered.iterrows():

    mean = row["Mean_Age"]
    sd = row["SD_Age"]
    amin = row["Age_min"]
    amax = row["Age_max"]

    color = (
        control_color
        if row["Group"] == "Aggregated controls"
        else case_color
    )

    if not np.isnan(amin) and not np.isnan(amax):

        ax1.bar(
            i,
            amax - amin,
            bottom=amin,
            width=0.6,
            color=color,
            edgecolor=dark,
            linewidth=0.8
        )

        ax1.scatter(
            i,
            mean,
            color=dark,
            s=25,
            zorder=3
        )

    elif not np.isnan(sd):

        lower = mean - sd
        upper = mean + sd

        ax1.bar(
            i,
            upper - lower,
            bottom=lower,
            width=0.6,
            color=color,
            edgecolor=dark,
            linewidth=0.8
        )

        ax1.scatter(
            i,
            mean,
            color=dark,
            s=25,
            zorder=3
        )

    else:

        ax1.scatter(
            i,
            mean,
            color=dark,
            s=25,
            zorder=3
        )

ax1.set_ylabel(
    "Age (years)",
    fontsize=17,
    color=dark
)

ax1.set_ylim(0, 86)

ax1.set_xticks([])

ax1.tick_params(
    axis="y",
    labelsize=14,
    colors=dark
)

ax1.set_title(
    "Demographic characteristics across ENIGMA samples",
    fontsize=20,
    color=dark
)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# =============================
# PANEL B — SEX
# =============================
ax2 = fig.add_subplot(gs[1])

female = df_ordered["Female_pct"].values
male = 100 - female

ax2.bar(
    x,
    female,
    width=0.7,
    color=female_color,
    label="Female"
)

ax2.bar(
    x,
    male,
    bottom=female,
    width=0.7,
    color=male_color,
    label="Male"
)

ax2.set_ylim(0, 100)

ax2.set_xticks(x)

ax2.set_xticklabels(
    labels,
    fontsize=17,
    color=dark
)

ax2.set_ylabel(
    "Sex (%)",
    fontsize=17,
    color=dark
)

ax2.tick_params(
    axis="y",
    labelsize=13,
    colors=dark
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# -----------------------------
# SAVE (PNG ONLY)
# -----------------------------
plt.savefig(
    "ENIGMA_demographics_final_cortexonly.png",
    dpi=600,
    bbox_inches="tight"
)
