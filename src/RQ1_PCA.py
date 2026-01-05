import pandas as pd
import numpy as np
from scipy.stats import sem
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

from lisbon import lisbon_map
import matplotlib
# Register the colormap directly from matplotlib, not matplotlib.cm
matplotlib.colormaps.register(lisbon_map)  # automatically uses lisbon_map.name



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

def remove_ventricles(arr16):
    arr_copy = arr16.copy()
    arr_copy[[7, 15]] = np.nan   # left and right ventricles
    return arr_copy


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

from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

# -----------------------------
# PANEL A–B: Δ MAPS (DARK-CENTERED)
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
# Symmetric around 0 with 95th percentile
cortex_range_delta = robust_sym_range([delta_adult_ctx.values, delta_ped_ctx.values])
subcortex_range_delta = robust_sym_range([delta_adult_sctx.values, delta_ped_sctx.values])

# Norm centered at 0 for better contrast
norm_cortex = TwoSlopeNorm(vmin=cortex_range_delta[0],
                           vcenter=0,
                           vmax=cortex_range_delta[1])
norm_subcortex = TwoSlopeNorm(vmin=subcortex_range_delta[0],
                              vcenter=0,
                              vmax=subcortex_range_delta[1])

# -----------------------------
# PANEL A–B: Δ MAPS (DARK-CENTERED)
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
# Symmetric around 0 for darker center
cortex_range_delta = robust_sym_range([delta_adult_ctx.values, delta_ped_ctx.values])
subcortex_range_delta = robust_sym_range([delta_adult_sctx.values, delta_ped_sctx.values])

# -----------------------------
# PANEL A: ADULT Δ MAPS
# -----------------------------
CT_d_fsa5 = parcel_to_surface(delta_adult_ctx.values, 'aparc_fsa5')
plot_cortical(
    array_name=CT_d_fsa5,
    surface_name="fsa5",
    cmap="lisbon",          # Berlin, Lisbon, Vanimo if available
    color_bar=True,
    color_range=cortex_range_delta,
    size=(1200,1200),
    scale=(3,3),
    background=(1,1,1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "Adult_PSYSUD_Delta_Cortex_darkcenter.png")
)

plot_subcortical(
    remove_ventricles(delta_adult_sctx_pad),
    cmap="lisbon",
    color_bar=True,
    color_range=subcortex_range_delta,
    size=(1200,1200),
    scale=(3,3),
    background=(1,1,1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "Adult_PSYSUD_Delta_Subcortex_darkcenter.png")
)

# -----------------------------
# PANEL B: PEDIATRIC Δ MAPS
# -----------------------------
CT_d_fsa5 = parcel_to_surface(delta_ped_ctx.values, 'aparc_fsa5')
plot_cortical(
    array_name=CT_d_fsa5,
    surface_name="fsa5",
    cmap="lisbon",
    color_bar=True,
    color_range=cortex_range_delta,
    size=(1200,1200),
    scale=(3,3),
    background=(1,1,1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "Pediatric_PSYSUD_Delta_Cortex_darkcenter.png")
)

plot_subcortical(
    remove_ventricles(delta_ped_sctx_pad),
    cmap="lisbon",
    color_bar=True,
    color_range=subcortex_range_delta,
    size=(1200,1200),
    scale=(3,3),
    background=(1,1,1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "Pediatric_PSYSUD_Delta_Subcortex_darkcenter.png")
)


# -----------------------------
# PANEL C: FINGERPRINT (UPDATED)
# -----------------------------

# -----------------------------
# REGION LABELS
# -----------------------------
ctx_labels = [
    # Left hemisphere
    "Left banks of superior temporal sulcus",
    "Left caudal anterior cingulate cortex",
    "Left caudal middle frontal gyrus",
    "Left cuneus",
    "Left entorhinal cortex",
    "Left fusiform gyrus",
    "Left inferior parietal cortex",
    "Left inferior temporal gyrus",
    "Left isthmus cingulate cortex",
    "Left lateral occipital cortex",
    "Left lateral orbitofrontal cortex",
    "Left lingual gyrus",
    "Left medial orbitofrontal cortex",
    "Left middle temporal gyrus",
    "Left parahippocampal gyrus",
    "Left paracentral lobule",
    "Left pars opercularis of inferior frontal gyrus",
    "Left pars orbitalis of inferior frontal gyrus",
    "Left pars triangularis of inferior frontal gyrus",
    "Left pericalcarine cortex",
    "Left postcentral gyrus",
    "Left posterior cingulate cortex",
    "Left precentral gyrus",
    "Left precuneus",
    "Left rostral anterior cingulate cortex",
    "Left rostral middle frontal gyrus",
    "Left superior frontal gyrus",
    "Left superior parietal cortex",
    "Left superior temporal gyrus",
    "Left supramarginal gyrus",
    "Left frontal pole",
    "Left temporal pole",
    "Left transverse temporal gyrus",
    "Left insula",
    # Right hemisphere
    "Right banks of superior temporal sulcus",
    "Right caudal anterior cingulate cortex",
    "Right caudal middle frontal gyrus",
    "Right cuneus",
    "Right entorhinal cortex",
    "Right fusiform gyrus",
    "Right inferior parietal cortex",
    "Right inferior temporal gyrus",
    "Right isthmus cingulate cortex",
    "Right lateral occipital cortex",
    "Right lateral orbitofrontal cortex",
    "Right lingual gyrus",
    "Right medial orbitofrontal cortex",
    "Right middle temporal gyrus",
    "Right parahippocampal gyrus",
    "Right paracentral lobule",
    "Right pars opercularis of inferior frontal gyrus",
    "Right pars orbitalis of inferior frontal gyrus",
    "Right pars triangularis of inferior frontal gyrus",
    "Right pericalcarine cortex",
    "Right postcentral gyrus",
    "Right posterior cingulate cortex",
    "Right precentral gyrus",
    "Right precuneus",
    "Right rostral anterior cingulate cortex",
    "Right rostral middle frontal gyrus",
    "Right superior frontal gyrus",
    "Right superior parietal cortex",
    "Right superior temporal gyrus",
    "Right supramarginal gyrus",
    "Right frontal pole",
    "Right temporal pole",
    "Right transverse temporal gyrus",
    "Right insula"
]

subctx_labels = [
    # Left hemisphere, alphabetical
    "Left Accumbens",
    "Left Amygdala",
    "Left Caudate",
    "Left Hippocampus",
    "Left Pallidum",
    "Left Putamen",
    "Left Thalamus",
    # Right hemisphere, alphabetical
    "Right Accumbens",
    "Right Amygdala",
    "Right Caudate",
    "Right Hippocampus",
    "Right Pallidum",
    "Right Putamen",
    "Right Thalamus"
]

region_labels = ctx_labels + subctx_labels

# -----------------------------
# PANEL C: FINGERPRINT
# -----------------------------

# -----------------------------
# SEM for cortex & subcortex
# -----------------------------
from scipy.stats import sem

adult_ctx_sem = sem_across_disorders(psy_adult_ctx_all)   # 68 cortical regions
adult_sctx_sem = sem_across_disorders(psy_adult_sctx)    # 14 subcortical regions

ped_ctx_sem = sem_across_disorders(psy_ped_ctx_all)
ped_sctx_sem = sem_across_disorders(psy_ped_sctx)

# -----------------------------
# X-axis positions
# -----------------------------
x_ctx = np.arange(len(ctx_labels))                   # 0 to 67
x_sctx = np.arange(len(ctx_labels), len(region_labels))  # 68 to 81

# -----------------------------
# Create figure
# -----------------------------
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

# -----------------------------
# Adult PSY (top plot)
# -----------------------------
ax_top.plot(x_ctx, adult_ctx_mean, color="blue", label="Adult PSY")
ax_top.plot(x_sctx, adult_sctx_mean, color="blue")
ax_top.plot(x_ctx, sud_ctx_mean, color="gray", label="SUD")
ax_top.plot(x_sctx, sud_sctx_mean, color="gray")
ax_top.fill_between(x_ctx, adult_ctx_mean - adult_ctx_sem, adult_ctx_mean + adult_ctx_sem, color="blue", alpha=0.3)
ax_top.fill_between(x_sctx, adult_sctx_mean - adult_sctx_sem, adult_sctx_mean + adult_sctx_sem, color="blue", alpha=0.3)
ax_top.axvline(len(ctx_labels) - 0.5, color="black")
ax_top.set_title("Adult PSY vs SUD")
ax_top.legend(loc='upper right', fontsize='small')

# -----------------------------
# Pediatric PSY (bottom plot)
# -----------------------------
ax_bot.plot(x_ctx, ped_ctx_mean, color="orange", label="Pediatric PSY")
ax_bot.plot(x_sctx, ped_sctx_mean, color="orange")
ax_bot.plot(x_ctx, sud_ctx_mean, color="gray", label="SUD")
ax_bot.plot(x_sctx, sud_sctx_mean, color="gray")
ax_bot.fill_between(x_ctx, ped_ctx_mean - ped_ctx_sem, ped_ctx_mean + ped_ctx_sem, color="orange", alpha=0.3)
ax_bot.fill_between(x_sctx, ped_sctx_mean - ped_sctx_sem, ped_sctx_mean + ped_sctx_sem, color="orange", alpha=0.3)
ax_bot.axvline(len(ctx_labels) - 0.5, color="black")
ax_bot.set_title("Pediatric PSY vs SUD")
ax_bot.set_xlabel("Regions (Cortex → Subcortex)")
ax_bot.legend(loc='upper right', fontsize='small')

# -----------------------------
# X-axis labels (applied only to bottom plot)
# -----------------------------
ax_bot.set_xticks(np.arange(len(region_labels)))
ax_bot.set_xticklabels(region_labels, rotation=45, ha='right', fontsize=8)

# -----------------------------
# Adjust layout
# -----------------------------
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Fingerprint_Adult_Top_Pediatric_Bottom.png"), dpi=300)
plt.close()


# -----------------------------
# PANEL D: PCA (ADULT & PEDIATRIC, SUD REMOVED)
# -----------------------------

# ---------- ADULT PCA ----------
adult_ctx_only = psy_adult_ctx_all.copy()  # only PSY disorders
adult_sctx_only = psy_adult_sctx.copy()
adult_regions_only = pd.concat([adult_ctx_only, adult_sctx_only], axis=0, ignore_index=True)
X_adult = adult_regions_only.T

# Z-score per region (already standard in your script)
Xz_adult = X_adult.apply(lambda x: (x - x.mean()) / x.std(), axis=0).fillna(0)

# PCA
pca_adult = PCA(n_components=5)
pca_adult.fit(Xz_adult)
print("Adult PC1 variance explained:", pca_adult.explained_variance_ratio_[0])

pc1_adult = pca_adult.components_[0]
pc1_adult_ctx = pc1_adult[:68]
pc1_adult_sctx = pad_subcortex(pc1_adult[68:82])

# Center PC1 for main figure
pc1_adult_ctx_centered = pc1_adult_ctx - np.mean(pc1_adult_ctx)
pc1_adult_sctx_centered = pc1_adult_sctx - np.mean(pc1_adult_sctx)

pc1_range_centered = robust_sym_range([pc1_adult_ctx_centered, pc1_adult_sctx_centered])
pc1_range_raw = robust_sym_range([pc1_adult_ctx, pc1_adult_sctx])

# Cortical plot - centered
CT_d_fsa5 = parcel_to_surface(pc1_adult_ctx_centered, 'aparc_fsa5')
fn_cortex_adult = os.path.join(RESULTS_DIR, "PC1_loadings_cortex_adult_centered.png")
plot_cortical(
    array_name=CT_d_fsa5,
    surface_name="fsa5",
    cmap='RdBu_r',
    color_bar=True,
    color_range=pc1_range_centered,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_cortex_adult
)

# Subcortical plot - centered
fn_subcortex_adult = os.path.join(RESULTS_DIR, "PC1_loadings_subcortex_adult_centered.png")
plot_subcortical(
    remove_ventricles(pc1_adult_sctx_centered),
    cmap='RdBu_r',
    color_bar=True,
    color_range=pc1_range_centered,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_subcortex_adult
)


# Optional: raw PC1 for supplement

pc1_adult_sctx_raw = remove_ventricles(pc1_adult_sctx)

CT_d_fsa5_raw = parcel_to_surface(pc1_adult_ctx, 'aparc_fsa5')
plot_cortical(
    array_name=CT_d_fsa5_raw,
    surface_name="fsa5",
    cmap='RdBu_r',
    color_bar=True,
    color_range=pc1_range_raw,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=os.path.join(RESULTS_DIR, "PC1_loadings_cortex_adult_raw.png")
)

# Adult raw subcortex
pc1_adult_sctx_raw = remove_ventricles(pc1_adult_sctx)

fn_subcortex_adult_raw = os.path.join(RESULTS_DIR, "PC1_loadings_subcortex_adult_raw.png")
plot_subcortical(
    pc1_adult_sctx_raw,       # raw, un-centered
    cmap='RdBu_r',
    color_bar=True,
    color_range=pc1_range_raw,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_subcortex_adult_raw
)


# ---------- PEDIATRIC PCA ----------
ped_ctx_only = psy_ped_ctx_all.copy()
ped_sctx_only = psy_ped_sctx.copy()
ped_regions_only = pd.concat([ped_ctx_only, ped_sctx_only], axis=0, ignore_index=True)
X_ped = ped_regions_only.T

Xz_ped = X_ped.apply(lambda x: (x - x.mean()) / x.std(), axis=0).fillna(0)

pca_ped = PCA(n_components=5)
pca_ped.fit(Xz_ped)
print("Pediatric PC1 variance explained:", pca_ped.explained_variance_ratio_[0])

pc1_ped = pca_ped.components_[0]
pc1_ped_ctx = pc1_ped[:68]
pc1_ped_sctx = pad_subcortex(pc1_ped[68:82])

# Center PC1
pc1_ped_ctx_centered = pc1_ped_ctx - np.mean(pc1_ped_ctx)
pc1_ped_sctx_centered = pc1_ped_sctx - np.mean(pc1_ped_sctx)

pc1_range_centered_ped = robust_sym_range([pc1_ped_ctx_centered, pc1_ped_sctx_centered])
pc1_range_raw_ped = robust_sym_range([pc1_ped_ctx, pc1_ped_sctx])

# Cortical plot - centered
CT_d_fsa5 = parcel_to_surface(pc1_ped_ctx_centered, 'aparc_fsa5')
fn_cortex_ped = os.path.join(RESULTS_DIR, "PC1_loadings_cortex_ped_centered.png")
plot_cortical(
    array_name=CT_d_fsa5,
    surface_name="fsa5",
    cmap='RdBu_r',
    color_bar=True,
    color_range=pc1_range_centered_ped,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_cortex_ped
)

# Subcortical plot - centered
fn_subcortex_ped = os.path.join(RESULTS_DIR, "PC1_loadings_subcortex_ped_centered.png")
plot_subcortical(
    remove_ventricles(pc1_ped_sctx_centered),
    cmap='RdBu_r',
    color_bar=True,
    color_range=pc1_range_centered_ped,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_subcortex_ped
)

# Pediatric raw cortex
CT_d_fsa5_raw_ped = parcel_to_surface(pc1_ped_ctx, 'aparc_fsa5')
fn_cortex_ped_raw = os.path.join(RESULTS_DIR, "PC1_loadings_cortex_ped_raw.png")
plot_cortical(
    array_name=CT_d_fsa5_raw_ped,
    surface_name="fsa5",
    cmap='RdBu_r',             # same colormap as adult raw
    color_bar=True,
    color_range=pc1_range_raw_ped,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_cortex_ped_raw
)

# Pediatric raw subcortex
pc1_ped_sctx_raw   = remove_ventricles(pc1_ped_sctx)

fn_subcortex_ped_raw = os.path.join(RESULTS_DIR, "PC1_loadings_subcortex_ped_raw.png")
plot_subcortical(
    pc1_ped_sctx_raw,         # raw, un-centered
    cmap='RdBu_r',
    color_bar=True,
    color_range=pc1_range_raw_ped,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
    screenshot=True,
    filename=fn_subcortex_ped_raw
)


# -----------------------------
# PANEL E: DISORDER EXPRESSION OF PC1 (ADULT & PEDIATRIC)
# -----------------------------

def plot_disorder_expression(pc_vals, ctx_df, population, color, filename):
    """Plot correlations of each disorder with PC1 (cortical regions only)."""
    corrs = [
        np.corrcoef(ctx_df[d].values, pc_vals)[0, 1]
        for d in ctx_df.columns
    ]
    df_corrs = pd.DataFrame({
        "Disorder": ctx_df.columns,
        "PC1_corr": corrs
    }).sort_values("PC1_corr")

    plt.figure(figsize=(10, 6))
    plt.barh(
        df_corrs["Disorder"],
        df_corrs["PC1_corr"],
        color=color
    )
    plt.xlabel(f"Correlation with {population} PC1")
    plt.title(f"Disorder Expression of {population} PC1")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()
    return corrs

# Cortical only (68 regions)
adult_ctx_only = psy_adult_ctx_all.iloc[:68, :]
ped_ctx_only = psy_ped_ctx_all.iloc[:68, :]

# Adult disorder loadings
corrs_adult = plot_disorder_expression(pc1_adult_ctx, adult_ctx_only, "Adult", "darkblue",
                                       "Disorder_expression_PC1_adult_PSY_only.png")

# Pediatric disorder loadings
corrs_ped = plot_disorder_expression(pc1_ped_ctx, ped_ctx_only, "Pediatric", "orange",
                                     "Disorder_expression_PC1_ped_PSY_only.png")


# -----------------------------
# CORRELATION BETWEEN PC1 AND DELTA MAPS (EXCLUDING VENTRICLES)
# -----------------------------
def exclude_ventricles(pc, delta):
    """
    Remove ventricles from subcortex arrays: idx 7 and 15 (0-indexed)  
    pc: concatenated PC1 or PC2 array (cortex + subcortex, length 82)
    delta: concatenated delta array (cortex + subcortex, length 82)
    """
    # cortex = first 68
    pc_cortex = pc[:68]
    delta_cortex = delta[:68]

    # subcortex = next 14
    pc_sub = pc[68:]
    delta_sub = delta[68:]

    # remove ventricles at subcortex indices 7 and 15 -> in sub array: indices 7 (left lat ventricle), 15 (right lat ventricle)
    # sub array length = 14, so indices are 7 (left ventricle) and 13 (right ventricle)
    vent_indices = [7, 13]  
    pc_sub_novent = np.delete(pc_sub, vent_indices)
    delta_sub_novent = np.delete(delta_sub, vent_indices)

    # concatenate back
    pc_novent = np.concatenate([pc_cortex, pc_sub_novent])
    delta_novent = np.concatenate([delta_cortex, delta_sub_novent])

    return pc_novent, delta_novent


# Adult
pc1_adult_novent, delta_adult_novent = exclude_ventricles(
    pc1_adult, np.concatenate([delta_adult_ctx.values, delta_adult_sctx.values])
)
corr_adult_delta = np.corrcoef(pc1_adult_novent, delta_adult_novent)[0, 1]

# Pediatric
pc1_ped_novent, delta_ped_novent = exclude_ventricles(
    pc1_ped, np.concatenate([delta_ped_ctx.values, delta_ped_sctx.values])
)
corr_ped_delta = np.corrcoef(pc1_ped_novent, delta_ped_novent)[0, 1]


# -----------------------------
# SAVE CSV: VARIANCE EXPLAINED + CORRELATION WITH DELTA
# -----------------------------
df_summary = pd.DataFrame({
    "Population": ["Adult", "Pediatric"],
    "PC1_variance_explained": [pca_adult.explained_variance_ratio_[0],
                               pca_ped.explained_variance_ratio_[0]],
    "PC2_variance_explained": [pca_adult.explained_variance_ratio_[1],
                               pca_ped.explained_variance_ratio_[1]],
    "PC1_vs_Delta_map_corr": [corr_adult_delta, corr_ped_delta]
})

csv_path = os.path.join(RESULTS_DIR, "PC1_PC2_variance_and_correlation.csv")
df_summary.to_csv(csv_path, index=False)
print(f"Saved PC1/PC2 variance + Δ map correlations:\n{df_summary}")
