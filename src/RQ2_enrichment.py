#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = r"C:\Users\giaco\Desktop\Git_META\META"

# ---------------- LOAD DATA ----------------
ADULT_DIRS = [os.path.join(BASE_DIR, "ALL_outputs_RQ1", "adults_all")]
records = []
for d in ADULT_DIRS:
    brain = pd.read_csv(os.path.join(d, "Z_cortex_euclidean.csv"), index_col=0)
    records.append(
        brain.reset_index().melt(id_vars="index", var_name="SUD", value_name="Z_euclidean")
        .rename(columns={"index": "PSY"})
    )
df = pd.concat(records)

AGE_FILE = os.path.join(BASE_DIR, "data", "raw", "age_onset_prevalence_of_disorders.xlsx")
age_onset = pd.read_excel(AGE_FILE, index_col=0)
age_onset = age_onset.applymap(lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else None)
age_long = age_onset.reset_index().melt(id_vars="DISORDER", var_name="SUD", value_name="Comorbidity")
age_long = age_long.rename(columns={"DISORDER": "PSY"})
df = pd.merge(df, age_long, on=["PSY","SUD"], how="inner")

GEN_FILE = os.path.join(BASE_DIR, "data", "raw", "PSY_SUD_genetic_corr.xlsx")
genetic = pd.read_excel(GEN_FILE, index_col=0)
gen_long = genetic.reset_index().melt(id_vars="index", var_name="SUD", value_name="genetic_corr").rename(columns={"index": "PSY"})
df = pd.merge(df, gen_long, on=["PSY","SUD"], how="inner")

# ---------------- TOP 10 -----------------
top_brain_df = df.nlargest(10, "Z_euclidean")[["PSY","SUD","Z_euclidean"]].copy()
top_gen_df = df.nlargest(10, "genetic_corr")[["PSY","SUD"]]
top_com_df = df.nlargest(10, "Comorbidity")[["PSY","SUD"]]

# ---------------- CHECK INTERSECTIONS -----------------
top_brain_df["In_Top_Genetics"] = top_brain_df.apply(
    lambda row: f"{row['PSY']}-{row['SUD']}" in set(top_gen_df.apply(lambda r: f"{r['PSY']}-{r['SUD']}", axis=1)), axis=1)
top_brain_df["In_Top_Comorbidity"] = top_brain_df.apply(
    lambda row: f"{row['PSY']}-{row['SUD']}" in set(top_com_df.apply(lambda r: f"{r['PSY']}-{r['SUD']}", axis=1)), axis=1)

# ---------------- PLOT -----------------
plt.figure(figsize=(10,6))
colors = []
for _, row in top_brain_df.iterrows():
    if row["In_Top_Genetics"] and row["In_Top_Comorbidity"]:
        colors.append("purple")  # presente in tutti e tre
    elif row["In_Top_Genetics"]:
        colors.append("blue")    # presente anche in genetics
    elif row["In_Top_Comorbidity"]:
        colors.append("orange")  # presente anche in comorbidity
    else:
        colors.append("gray")    # solo brain

bars = plt.barh(
    top_brain_df.apply(lambda r: f"{r['PSY']}-{r['SUD']}", axis=1),
    top_brain_df["Z_euclidean"],
    color=colors
)
plt.xlabel("Brain Similarity (Z_euclidean)")
plt.title("Top 10 Brain Similarity and Overlaps")
plt.gca().invert_yaxis()

# ---------------- LEGEND FUORI -----------------
legend_elements = [
    Patch(facecolor='gray', label='Only Brain Top 10'),
    Patch(facecolor='blue', label='Also in Genetics Top 10'),
    Patch(facecolor='orange', label='Also in Comorbidity Top 10'),
    Patch(facecolor='purple', label='In Brain, Genetics, and Comorbidity Top 10')
]

plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=2, frameon=True)

plt.tight_layout()
plt.savefig("top10_brain_overlap.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure: top10_brain_overlap.png")