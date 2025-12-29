import pandas as pd
import numpy as np
from scipy.stats import sem
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import sys

# Add ENIGMA Toolbox to path
sys.path.append(r"C:\Users\giaco\Desktop\Git_META\ENIGMA\enigmatoolbox")

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = r"C:\Users\giaco\Desktop\Git_META\META\data\raw"
RESULTS_DIR = r"C:\Users\giaco\Desktop\Git_META\META\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_enigma_file(fname):
    df = pd.read_excel(os.path.join(DATA_DIR, f"{fname}.xlsx"), decimal=",")
    return df.apply(pd.to_numeric, errors="coerce")

def merge_ctx(ctx_df, full_df):
    return pd.concat([full_df.iloc[0:68, :], ctx_df], axis=1)

def mean_across_disorders(df):
    return df.mean(axis=1)

def sem_across_disorders(df):
    return sem(df.values, axis=1, nan_policy='omit')

def pad_subcortex(data14):
    padded = np.zeros(16)
    padded[:7] = data14[:7]
    padded[7] = 0
    padded[8:15] = data14[7:14]
    padded[15] = 0
    return padded

def robust_sym_range(arrays, percentile=95):
    data = np.hstack([a[~np.isnan(a)] for a in arrays])
    v = np.percentile(np.abs(data), percentile)
    return (-v, v)

# -----------------------------
# LOAD DATA
# -----------------------------
psy_adult = load_enigma_file("PSY_adults")
psy_adult_ctx = load_enigma_file("PSY_adults_ctx")
psy_ped = load_enigma_file("PSY_adolescents")
psy_ped_ctx = load_enigma_file("PSY_adolescents_ctx")
sud = load_enigma_file("SUD")

psy_adult_ctx_all = merge_ctx(psy_adult_ctx, psy_adult)
psy_ped_ctx_all = merge_ctx(psy_ped_ctx, psy_ped)

psy_adult_sctx = psy_adult.iloc[68:82, :]
psy_ped_sctx = psy_ped.iloc[68:82, :]
sud_sctx = sud.iloc[68:82, :]

# -----------------------------
# PANEL A–B: Δ MAPS
# -----------------------------
adult_ctx_mean = mean_across_disorders(psy_adult_ctx_all)
ped_ctx_mean = mean_across_disorders(psy_ped_ctx_all)
sud_ctx_mean = mean_across_disorders(sud.iloc[0:68, :])

delta_adult_ctx = adult_ctx_mean - sud_ctx_mean
delta_ped_ctx = ped_ctx_mean - sud_ctx_mean

adult_sctx_mean = mean_across_disorders(psy_adult_sctx)
ped_sctx_mean = mean_across_disorders(psy_ped_sctx)
sud_sctx_mean = mean_across_disorders(sud_sctx)

delta_adult_sctx = adult_sctx_mean - sud_sctx_mean
delta_ped_sctx = ped_sctx_mean - sud_sctx_mean

delta_adult_sctx_pad = pad_subcortex(delta_adult_sctx.values)
delta_ped_sctx_pad = pad_subcortex(delta_ped_sctx.values)

# -------- COLOR RANGES --------
cortex_range_delta = robust_sym_range([
    delta_adult_ctx.values,
    delta_ped_ctx.values
])
subcortex_range_delta = robust_sym_range([
    delta_adult_sctx.values,
    delta_ped_sctx.values
])

# -----------------------------
# PANEL A
# -----------------------------
CT_d_fsa5 = parcel_to_surface(delta_adult_ctx.values, 'aparc_fsa5')
plot_cortical(
    array_name=CT_d_fsa5,
    surface_name="fsa5",
    cmap="RdBu_r",
    color_bar=True,
    color_range=cortex_range_delta,
    size=(1200,1200),
    scale=(3,3),
    background=(1,1,1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "Adult_PSYSUD_Delta_Cortex.png")
)

plot_subcortical(
    delta_adult_sctx_pad,
    cmap='RdBu_r',
    color_bar=True,
    color_range=subcortex_range_delta,
    size=(1200,1200),
    scale=(3,3),
    background=(1,1,1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "Adult_PSYSUD_Delta_Subcortex.png")
)

# -----------------------------
# PANEL B
# -----------------------------
CT_d_fsa5 = parcel_to_surface(delta_ped_ctx.values, 'aparc_fsa5')
plot_cortical(
    array_name=CT_d_fsa5,
    surface_name="fsa5",
    cmap="RdBu_r",
    color_bar=True,
    color_range=cortex_range_delta,
    size=(1200,1200),
    scale=(3,3),
    background=(1,1,1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "Pediatric_PSYSUD_Delta_Cortex.png")
)

plot_subcortical(
    delta_ped_sctx_pad,
    cmap='RdBu_r',
    color_bar=True,
    color_range=subcortex_range_delta,
    size=(1200,1200),
    scale=(3,3),
    background=(1,1,1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "Pediatric_PSYSUD_Delta_Subcortex.png")
)

# -----------------------------
# PANEL C: FINGERPRINT (UNCHANGED)
# -----------------------------
x_ctx = np.arange(68)
x_sctx = np.arange(68, 82)

ped_sctx_sem = sem_across_disorders(psy_ped_sctx)
adult_sctx_sem = sem_across_disorders(psy_adult_sctx)

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax_top.plot(x_ctx, ped_ctx_mean, color="orange", label="Pediatric PSY")
ax_top.plot(x_ctx, sud_ctx_mean, color="gray")
ax_top.fill_between(
    x_ctx,
    ped_ctx_mean - sem_across_disorders(psy_ped_ctx_all),
    ped_ctx_mean + sem_across_disorders(psy_ped_ctx_all),
    color="orange", alpha=0.3
)

ax_top.plot(x_sctx, ped_sctx_mean, color="orange")
ax_top.plot(x_sctx, sud_sctx_mean, color="gray")
ax_top.fill_between(
    x_sctx,
    ped_sctx_mean - ped_sctx_sem,
    ped_sctx_mean + ped_sctx_sem,
    color="orange", alpha=0.3
)

ax_top.axvline(67.5, color="black")
ax_top.set_title("Pediatric PSY vs SUD")
ax_top.legend()

ax_bot.plot(x_ctx, adult_ctx_mean, color="blue", label="Adult PSY")
ax_bot.plot(x_ctx, sud_ctx_mean, color="gray")
ax_bot.fill_between(
    x_ctx,
    adult_ctx_mean - sem_across_disorders(psy_adult_ctx_all),
    adult_ctx_mean + sem_across_disorders(psy_adult_ctx_all),
    color="blue", alpha=0.3
)

ax_bot.plot(x_sctx, adult_sctx_mean, color="blue")
ax_bot.plot(x_sctx, sud_sctx_mean, color="gray")
ax_bot.fill_between(
    x_sctx,
    adult_sctx_mean - adult_sctx_sem,
    adult_sctx_mean + adult_sctx_sem,
    color="blue", alpha=0.3
)

ax_bot.axvline(67.5, color="black")
ax_bot.set_title("Adult PSY vs SUD")
ax_bot.set_xlabel("Regions (Cortex → Subcortex)")
ax_bot.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Fingerprint_Pediatric_Top_Adult_Bottom.png"), dpi=300)
plt.close()

# -----------------------------
# PANEL D: PCA (ADULT & PEDIATRIC)
# -----------------------------
# ---------- ADULT PCA ----------
adult_ctx = pd.concat([psy_adult_ctx_all, sud.iloc[0:68, :]], axis=1)
adult_sctx = pd.concat([psy_adult_sctx, sud_sctx], axis=1)
adult_regions = pd.concat([adult_ctx, adult_sctx], axis=0, ignore_index=True)
X_adult = adult_regions.T

Xz_adult = X_adult.apply(lambda x: (x - x.mean()) / x.std(), axis=0).fillna(0)
pca_adult = PCA(n_components=5)
pca_adult.fit(Xz_adult)

pc1_adult = pca_adult.components_[0]
pc1_adult_ctx = pc1_adult[:68]
pc1_adult_sctx = pad_subcortex(pc1_adult[68:82])

pc1_range_adult = robust_sym_range([pc1_adult_ctx, pc1_adult_sctx])

# Cortical plot
CT_d_fsa5 = parcel_to_surface(pc1_adult_ctx, 'aparc_fsa5')
fn_cortex_adult = os.path.join(RESULTS_DIR, "PC1_loadings_cortex_adult.png")
plot_cortical(
    array_name=CT_d_fsa5,
    surface_name="fsa5",
    cmap='coolwarm',
    color_bar=True,
    color_range=pc1_range_adult,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_cortex_adult
)

# Subcortical plot
fn_subcortex_adult = os.path.join(RESULTS_DIR, "PC1_loadings_subcortex_adult.png")
plot_subcortical(
    pc1_adult_sctx,
    cmap='coolwarm',
    color_bar=True,
    color_range=pc1_range_adult,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_subcortex_adult
)

# ---------- PEDIATRIC PCA ----------
ped_ctx = pd.concat([psy_ped_ctx_all, sud.iloc[0:68, :]], axis=1)
ped_sctx = pd.concat([psy_ped_sctx, sud_sctx], axis=1)
ped_regions = pd.concat([ped_ctx, ped_sctx], axis=0, ignore_index=True)
X_ped = ped_regions.T

Xz_ped = X_ped.apply(lambda x: (x - x.mean()) / x.std(), axis=0).fillna(0)
pca_ped = PCA(n_components=5)
pca_ped.fit(Xz_ped)

pc1_ped = pca_ped.components_[0]
pc1_ped_ctx = pc1_ped[:68]
pc1_ped_sctx = pad_subcortex(pc1_ped[68:82])

pc1_range_ped = robust_sym_range([pc1_ped_ctx, pc1_ped_sctx])

# Cortical plot
CT_d_fsa5 = parcel_to_surface(pc1_ped_ctx, 'aparc_fsa5')
fn_cortex_ped = os.path.join(RESULTS_DIR, "PC1_loadings_cortex_ped.png")
plot_cortical(
    array_name=CT_d_fsa5,
    surface_name="fsa5",
    cmap='coolwarm',
    color_bar=True,
    color_range=pc1_range_ped,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_cortex_ped
)

# Subcortical plot
fn_subcortex_ped = os.path.join(RESULTS_DIR, "PC1_loadings_subcortex_ped.png")
plot_subcortical(
    pc1_ped_sctx,
    cmap='coolwarm',
    color_bar=True,
    color_range=pc1_range_ped,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_subcortex_ped
)

# -----------------------------
# CORRELATION BETWEEN PC1 AND DELTA MAPS
# -----------------------------
# Adult correlation (excluding ventricles)
pc1_adult_novent = np.concatenate([pc1_adult_ctx, pc1_adult_sctx[:7], pc1_adult_sctx[8:15]])
delta_adult_novent = np.concatenate([delta_adult_ctx.values, delta_adult_sctx.values])
corr_adult_delta = np.corrcoef(pc1_adult_novent, delta_adult_novent)[0, 1]

# Pediatric correlation (excluding ventricles)
pc1_ped_novent = np.concatenate([pc1_ped_ctx, pc1_ped_sctx[:7], pc1_ped_sctx[8:15]])
delta_ped_novent = np.concatenate([delta_ped_ctx.values, delta_ped_sctx.values])
corr_ped_delta = np.corrcoef(pc1_ped_novent, delta_ped_novent)[0, 1]

# Save results to CSV
corr_df = pd.DataFrame({
    "Population": ["Adult", "Pediatric"],
    "PC1_vs_Delta_map": [corr_adult_delta, corr_ped_delta]
})

corr_df.to_csv(os.path.join(RESULTS_DIR, "PC1_vs_Delta_map_correlation.csv"), index=False)
print("Saved PC1 vs Δ map correlations:\n", corr_df)


# -----------------------------
# PANEL E: DISORDER EXPRESSION OF PC1
# -----------------------------
# Select cortical regions only (first 68 rows)
adult_ctx_only = psy_adult_ctx_all.iloc[0:68, :]
ped_ctx_only = psy_ped_ctx_all.iloc[0:68, :]
sud_ctx_only = sud.iloc[0:68, :]

# -----------------------------
# Adult PC1 - Adult Disorders + SUD
# -----------------------------
all_disorders_adult = pd.concat([adult_ctx_only, sud_ctx_only], axis=1).T  # disorders x regions

# Compute correlation with adult cortical PC1
corrs_adult = [np.corrcoef(all_disorders_adult.loc[d].values, pc1_adult_ctx)[0, 1] 
               for d in all_disorders_adult.index]

df_corrs_adult = pd.DataFrame({
    "Disorder": all_disorders_adult.index,
    "PC1_corr": corrs_adult
})

# Color coding: adult PSY = dark blue, SUD = orange
colors_adult = []
for d in df_corrs_adult["Disorder"]:
    if d in adult_ctx_only.columns:
        colors_adult.append("darkblue")
    else:  # SUD
        colors_adult.append("gray")
df_corrs_adult["Color"] = colors_adult

df_corrs_adult = df_corrs_adult.sort_values("PC1_corr")

plt.figure(figsize=(10, 6))
plt.barh(df_corrs_adult["Disorder"], df_corrs_adult["PC1_corr"], color=df_corrs_adult["Color"])
plt.xlabel("Correlation with Adult PC1")
plt.title("Disorder Expression of Adult PC1")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Disorder_expression_PC1_adult.png"), dpi=300)
plt.close()


# -----------------------------
# Pediatric PC1 - Pediatric Disorders + SUD
# -----------------------------
all_disorders_ped = pd.concat([ped_ctx_only, sud_ctx_only], axis=1).T  # disorders x regions

# Compute correlation with pediatric cortical PC1
corrs_ped = [np.corrcoef(all_disorders_ped.loc[d].values, pc1_ped_ctx)[0, 1] 
             for d in all_disorders_ped.index]

df_corrs_ped = pd.DataFrame({
    "Disorder": all_disorders_ped.index,
    "PC1_corr": corrs_ped
})

# Color coding: pediatric PSY = turquoise, SUD = orange
colors_ped = []
for d in df_corrs_ped["Disorder"]:
    if d in ped_ctx_only.columns:
        colors_ped.append("orange")
    else:  # SUD
        colors_ped.append("gray")
df_corrs_ped["Color"] = colors_ped

df_corrs_ped = df_corrs_ped.sort_values("PC1_corr")

plt.figure(figsize=(10, 6))
plt.barh(df_corrs_ped["Disorder"], df_corrs_ped["PC1_corr"], color=df_corrs_ped["Color"])
plt.xlabel("Correlation with Pediatric PC1")
plt.title("Disorder Expression of Pediatric PC1")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Disorder_expression_PC1_pediatric.png"), dpi=300)
plt.close()
