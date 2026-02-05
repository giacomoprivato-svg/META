import os
import numpy as np
import pandas as pd

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical

# ======================
# CONFIG (WINDOWS)
# ======================
BASE = r"C:\Users\giaco\Desktop\Git_META\META\data\raw"
OUTDIR = r"C:\Users\giaco\Desktop\Git_META\META\results\RQ1_3_overlap"
os.makedirs(OUTDIR, exist_ok=True)

PATH_SUD = os.path.join(BASE, "SUD.xlsx")
PATH_PSY_ADULTS_ALL = os.path.join(BASE, "PSY_adults.xlsx")
PATH_PSY_ADULTS_CTX = os.path.join(BASE, "PSY_adults_ctx.xlsx")
PATH_PSY_ADO_ALL = os.path.join(BASE, "PSY_adolescents.xlsx")
PATH_PSY_ADO_CTX = os.path.join(BASE, "PSY_adolescents_ctx.xlsx")

TOP_PCTS = [0.10, 0.20]

CTX_SLICE = slice(0, 68)   # first 68 rows
SCTX_N = 14                # last 14 rows

PLOT_KW_OVERLAP = dict(
    cmap="Blues",
    color_bar=True,
    size=(1200, 1200),
    scale=(3, 3),
    background=(1, 1, 1),
)

# ======================
# REGION LABELS
# ======================
DKT68 = [
    "lh_bankssts","lh_caudalanteriorcingulate","lh_caudalmiddlefrontal","lh_cuneus",
    "lh_entorhinal","lh_fusiform","lh_inferiorparietal","lh_inferiortemporal",
    "lh_isthmuscingulate","lh_lateraloccipital","lh_lateralorbitofrontal","lh_lingual",
    "lh_medialorbitofrontal","lh_middletemporal","lh_parahippocampal","lh_paracentral",
    "lh_parsopercularis","lh_parsorbitalis","lh_parstriangularis","lh_pericalcarine",
    "lh_postcentral","lh_posteriorcingulate","lh_precentral","lh_precuneus",
    "lh_rostralanteriorcingulate","lh_rostralmiddlefrontal","lh_superiorfrontal","lh_superiorparietal",
    "lh_superiortemporal","lh_supramarginal","lh_frontalpole","lh_temporalpole",
    "lh_transversetemporal","lh_insula",
    "rh_bankssts","rh_caudalanteriorcingulate","rh_caudalmiddlefrontal","rh_cuneus",
    "rh_entorhinal","rh_fusiform","rh_inferiorparietal","rh_inferiortemporal",
    "rh_isthmuscingulate","rh_lateraloccipital","rh_lateralorbitofrontal","rh_lingual",
    "rh_medialorbitofrontal","rh_middletemporal","rh_parahippocampal","rh_paracentral",
    "rh_parsopercularis","rh_parsorbitalis","rh_parstriangularis","rh_pericalcarine",
    "rh_postcentral","rh_posteriorcingulate","rh_precentral","rh_precuneus",
    "rh_rostralanteriorcingulate","rh_rostralmiddlefrontal","rh_superiorfrontal","rh_superiorparietal",
    "rh_superiortemporal","rh_supramarginal","rh_frontalpole","rh_temporalpole",
    "rh_transversetemporal","rh_insula",
]

SV14 = [
    "Left-Thalamus","Right-Thalamus",
    "Left-Caudate","Right-Caudate",
    "Left-Putamen","Right-Putamen",
    "Left-Pallidum","Right-Pallidum",
    "Left-Hippocampus","Right-Hippocampus",
    "Left-Amygdala","Right-Amygdala",
    "Left-Accumbens","Right-Accumbens",
]

# ======================
# HELPERS
# ======================
def read_excel_numeric(path):
    df = pd.read_excel(path)
    return df.apply(pd.to_numeric, errors="coerce")

def split_ctx_sctx(df):
    ctx = df.iloc[CTX_SLICE, :].copy()
    sctx = df.iloc[-SCTX_N:, :].copy()
    return ctx, sctx

def ensure_unique_cols(df, prefix):
    df = df.copy()
    df.columns = [f"{prefix}{c}" for c in df.columns.astype(str)]
    return df

def pad_subcortex(data14):
    padded = np.zeros(16)
    padded[:7] = data14[:7]
    padded[7] = np.nan
    padded[8:15] = data14[7:14]
    padded[15] = np.nan
    return padded

# ======================
# OVERLAP CORE
# ======================
def compute_overlap_negative_tail_with_labels(X, top_pct, region_labels):
    """
    Compute overlap of the lowest X% of values across disorders, for each region.

    Returns a DataFrame with:
        - region
        - overlap_count
        - overlap_frac
        - overlap_disorders (all in ONE cell, separated by ';')
    """
    X = X.apply(pd.to_numeric, errors="coerce")
    n_regions, n_disorders = X.shape

    # 0/1 bin for whether each region-disorder value is in the negative tail
    bins = np.zeros((n_regions, n_disorders), dtype=int)
    for j, col in enumerate(X.columns):
        v = X[col].to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() == 0:
            continue
        thr = np.quantile(v[ok], top_pct)
        bins[ok, j] = (v[ok] <= thr).astype(int)

    # Count how many disorders each region is in the top_pct
    overlap_count = bins.sum(axis=1)
    overlap_frac = overlap_count / n_disorders

    # Collect overlapping disorders for each region in ONE cell
    overlap_disorders = []
    for i in range(n_regions):
        idx = np.where(bins[i] == 1)[0]
        if len(idx) == 0:
            overlap_disorders.append("")
        else:
            # list of disorder names that overlap for this region
            disorders_in_region = [str(X.columns[j]) for j in idx]
            # join all disorders with ';'
            overlap_disorders.append(";".join(disorders_in_region))

    # Build output DataFrame
    out = pd.DataFrame({
        "region": region_labels,
        "overlap_count": overlap_count,
        "overlap_frac": overlap_frac,
        "overlap_disorders": overlap_disorders,
    })

    meta = dict(
        n_regions=n_regions,
        n_disorders=n_disorders,
        disorders=list(X.columns),
        top_pct=top_pct,
        tail="negative",
    )
    return out, meta

# ======================
# PLOTTING
# ======================
def save_outputs(tag, out_df, meta):
    out_df.to_csv(
        os.path.join(OUTDIR, f"{tag}_overlap_top{int(meta['top_pct']*100)}.csv"),
        index=False,
    )

def plot_overlap_ctx(tag, out_df, top_pct):
    vals = out_df["overlap_frac"].to_numpy(float)
    CT = parcel_to_surface(vals, "aparc_fsa5")
    plot_cortical(
        array_name=CT,
        surface_name="fsa5",
        color_range=(0, 0.6),  # range fisso tra 0 e 0.6
        screenshot=True,
        filename=os.path.join(OUTDIR, f"{tag}_cortex_top{int(top_pct*100)}.png"),
        **PLOT_KW_OVERLAP,
    )

def plot_overlap_sctx(tag, out_df, top_pct):
    vals14 = out_df["overlap_frac"].to_numpy(float)
    vals16 = pad_subcortex(vals14)
    plot_subcortical(
        vals16,
        color_range=(0, 0.6),
        screenshot=True,
        filename=os.path.join(OUTDIR, f"{tag}_subcortex_top{int(top_pct*100)}.png"),
        **PLOT_KW_OVERLAP,
    )

# ======================
# BUILD MATRICES
# ======================
def build_adults_ctx():
    sud = read_excel_numeric(PATH_SUD)
    psyA = read_excel_numeric(PATH_PSY_ADULTS_ALL)
    psyA_ctx_only = read_excel_numeric(PATH_PSY_ADULTS_CTX)
    sud_ctx, _ = split_ctx_sctx(sud)
    psyA_ctx, _ = split_ctx_sctx(psyA)
    return pd.concat([
        ensure_unique_cols(sud_ctx, "SUD_"),
        ensure_unique_cols(psyA_ctx, "PSYad_"),
        ensure_unique_cols(psyA_ctx_only, "PSYadctx_"),
    ], axis=1)

def build_adults_sctx():
    sud = read_excel_numeric(PATH_SUD)
    psyA = read_excel_numeric(PATH_PSY_ADULTS_ALL)
    _, sud_s = split_ctx_sctx(sud)
    _, psyA_s = split_ctx_sctx(psyA)
    return pd.concat([
        ensure_unique_cols(sud_s, "SUD_"),
        ensure_unique_cols(psyA_s, "PSYad_"),
    ], axis=1)

def build_adolescents_ctx():
    sud = read_excel_numeric(PATH_SUD)
    psyP = read_excel_numeric(PATH_PSY_ADO_ALL)
    psyP_ctx_only = read_excel_numeric(PATH_PSY_ADO_CTX)
    sud_ctx, _ = split_ctx_sctx(sud)
    psyP_ctx, _ = split_ctx_sctx(psyP)
    return pd.concat([
        ensure_unique_cols(sud_ctx, "SUD_"),
        ensure_unique_cols(psyP_ctx, "PSYped_"),
        ensure_unique_cols(psyP_ctx_only, "PSYpedctx_"),
    ], axis=1)

def build_adolescents_sctx():
    sud = read_excel_numeric(PATH_SUD)
    psyP = read_excel_numeric(PATH_PSY_ADO_ALL)
    _, sud_s = split_ctx_sctx(sud)
    _, psyP_s = split_ctx_sctx(psyP)
    return pd.concat([
        ensure_unique_cols(sud_s, "SUD_"),
        ensure_unique_cols(psyP_s, "PSYped_"),
    ], axis=1)

# ======================
# RUN
# ======================
print(f"Saving outputs to: {OUTDIR}")

X_ad_ctx = build_adults_ctx()
X_ad_s = build_adults_sctx()
X_ped_ctx = build_adolescents_ctx()
X_ped_s = build_adolescents_sctx()

for top_pct in TOP_PCTS:
    out, meta = compute_overlap_negative_tail_with_labels(X_ad_ctx, top_pct, DKT68)
    save_outputs("ADULTS_ALLDISORDERS_CTX", out, meta)
    plot_overlap_ctx("ADULTS_ALLDISORDERS_CTX", out, top_pct)

    out, meta = compute_overlap_negative_tail_with_labels(X_ad_s, top_pct, SV14)
    save_outputs("ADULTS_ALLDISORDERS_SCTX", out, meta)
    plot_overlap_sctx("ADULTS_ALLDISORDERS_SCTX", out, top_pct)

    out, meta = compute_overlap_negative_tail_with_labels(X_ped_ctx, top_pct, DKT68)
    save_outputs("ADOLESCENTS_ALLDISORDERS_CTX", out, meta)
    plot_overlap_ctx("ADOLESCENTS_ALLDISORDERS_CTX", out, top_pct)

    out, meta = compute_overlap_negative_tail_with_labels(X_ped_s, top_pct, SV14)
    save_outputs("ADOLESCENTS_ALLDISORDERS_SCTX", out, meta)
    plot_overlap_sctx("ADOLESCENTS_ALLDISORDERS_SCTX", out, top_pct)

print("✅ Done: cortex + subcortex overlap maps (ventricles handled correctly, overlap_disorders in one cell).")