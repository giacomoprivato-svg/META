#!/usr/bin/env python3
"""
RQ1 — supplementary figure: metric concordance and cortex vs subcortex
======================================================================

Two robustness checks on the main heatmap, in one figure.

Panel A : does the choice of similarity metric change the picture? Primary
          (Spearman rho) against the two sensitivity metrics, cosine and
          -Euclidean. They live on very different scales, so cosine is on the
          left axis and -Euclidean on a twin right axis. OLS fit + 95% band.
Panel B : does the cortical result hold subcortically? Cortex rho against
          subcortex rho for the same 54 pairs, coloured by clinical cluster.

BOTH p-VALUES ARE MANTEL, NOT PARAMETRIC — and this matters. The 54 points are
not independent: each psychiatric map contributes 6 of them and each SUD map
contributes 9, so a parametric p on n = 54 is badly anti-conservative. The
Mantel p permutes the rows and columns of one matrix, preserving that
dependency structure. Report the Mantel p; the parametric one is available
only as a diagnostic.

WHAT CHANGED
------------
0. THE AGGREGATE all-SUD COLUMN IS DROPPED FROM BOTH PANELS. This script reads
   RAW_cortex_* and RAW_subctx_*, which keep all seven columns by design (they
   are the descriptive matrices the manuscript shows for completeness). Every
   panel here is an analysis, not a display, so the aggregate has no place in
   it: its 2312 cases are the union of the six substance-specific case groups
   and it contributes a whole redundant column to a 9 x 7 grid.

   This matters more here than elsewhere, because the Mantel null permutes
   COLUMNS. A column that is a composite of the other six is not exchangeable
   with them, so leaving it in violates the null the test is built on, not
   just the pair count. Both panels now run on 9 x 6 = 54 pairs.

   EXPECT THE MANTEL p-VALUES TO MOVE. Unlike the pairwise FDR — where the
   family is a single column and dropping one column changes nothing — this
   test is computed over the whole matrix, so removing a column changes both
   the observed r and the permutation distribution.

1. NO STATSMODELS. The OLS fit and its confidence band are computed directly
   (closed form, 12 lines). statsmodels is not installed in every environment
   this repo runs in, and this was the only script that needed it.

2. The Mantel loop is vectorised — the old one called pearsonr 10,000 times
   per panel, three panels, i.e. 30,000 scipy calls to produce two numbers.

3. Subcortex is read from RAW_subctx_*, which RQ1_step2 now writes under a
   name the cortex script cannot overwrite. Previously both compartments wrote
   RANK_spearman_by_<SUD>.csv into the same folder.

4. Clusters and colours from RQ1_common; PTSD -> PD.

Just press Run. Requires RQ1_step1 and RQ1_step2.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

import RQ1_common as C

# ================= CONFIG — edit, then press Run =================
GROUP = "adults_all"
EXCLUDE = []
METRIC = C.PRIMARY
METRIC_LABEL = r"Spearman $\rho$"
SENSITIVITY = ["cosine", "euclidean"]

# The transdiagnostic aggregate, excluded from both panels (see change 0).
# Set EXCLUDE_AGGREGATE_SUD = False only to reproduce the old 63-pair figure.
EXCLUDE_AGGREGATE_SUD = True
AGGREGATE_SUD_NAMES = ["SUD", "ALL_SUD", "ALLSUD", "SUD_ALL", "ALL SUD"]
N_SUBSTANCE_MAPS = 6

N_MANTEL = 10000
CI_ALPHA = 0.05
SEED = 42
FIGSIZE = (16, 5.5)          # matches RQ1_fig3_specificity exactly
FS_LABEL, FS_TITLE, FS_TICK = 14, 16, 13
HEADROOM = 0.30              # fraction of the y-range left empty at the top so the
                             # r/p box never sits on top of the point cloud
BOX = dict(boxstyle="round,pad=0.45", fc="white", ec="0.55", alpha=1.0)
SENS_COLORS = {"cosine": "#e07270", "euclidean": "#2f9e44"}
SENS_LABEL = {"cosine": "Cosine", "euclidean": "\u2212Euclidean"}
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
GDIR = os.path.join(repo_dir, "ALL_outputs_RQ1", GROUP)
FIGDIR = os.path.join(repo_dir, "figures")
os.makedirs(FIGDIR, exist_ok=True)

_dropped_aggregate = []


def drop_aggregate(df, tag):
    """
    Remove the aggregate all-SUD column from a PSY x SUD matrix.

    Records what it removed so the caller can prove, after all files are read,
    that the exclusion actually fired. A column surviving under an unexpected
    name would otherwise re-enter the Mantel test invisibly.
    """
    if not EXCLUDE_AGGREGATE_SUD:
        return df
    wanted = {n.strip().upper() for n in AGGREGATE_SUD_NAMES}
    hits = [c for c in df.columns if str(c).strip().upper() in wanted]
    _dropped_aggregate.extend((tag, c) for c in hits)
    return df.drop(columns=hits)


def read(tag):
    p = os.path.join(GDIR, f"{tag}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}\nRun RQ1_step1 / RQ1_step2 first.")
    df = pd.read_csv(p, index_col=0)
    df = df.drop(index=[i for i in df.index if i in EXCLUDE], errors="ignore")
    return drop_aggregate(df, tag)


def ols_band(x, y, xg, alpha=CI_ALPHA):
    """Least-squares fit and 95% confidence band on the mean, closed form."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    X = np.column_stack([np.ones(x.size), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    dof = x.size - 2
    s2 = float((y - X @ beta) @ (y - X @ beta)) / dof
    XtXi = np.linalg.inv(X.T @ X)
    Xg = np.column_stack([np.ones(xg.size), xg])
    fit = Xg @ beta
    se = np.sqrt(s2 * np.einsum("ij,jk,ik->i", Xg, XtXi, Xg))
    h = tdist.ppf(1 - alpha / 2, dof) * se
    return fit, fit - h, fit + h


def _r(a, B):
    """Pearson r between flat vector a and each row of flattened matrices B."""
    a = a - a.mean()
    B = B - B.mean(axis=1, keepdims=True)
    return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a))


def mantel(A, B, n=N_MANTEL, seed=SEED):
    """
    Label-permutation p for the correlation between two PSY x SUD matrices.
    Rows and columns of B are permuted independently, preserving the fact that
    each disorder contributes a whole row and each substance a whole column.

    The exchangeability this relies on is why the aggregate column has to go:
    a column that is a composite of the others is not exchangeable with them.
    """
    a = np.asarray(A, float).ravel()
    b = np.asarray(B, float)
    r_obs = float(np.corrcoef(a, b.ravel())[0, 1])
    rng = np.random.default_rng(seed)
    nr, nc = b.shape
    perm = np.empty((n, a.size))
    for k in range(n):
        perm[k] = b[np.ix_(rng.permutation(nr), rng.permutation(nc))].ravel()
    null = _r(a, perm)
    return r_obs, float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n + 1))


def headroom(ax, cap, frac=None):
    """
    Open empty space at the top of an axis for the annotation box, WITHOUT
    inventing tick labels for values the metric cannot take.

    The box used to be drawn over the data with alpha=0.9, which is why the
    numbers were hard to read: the scatter showed through it. Making it opaque
    alone would just hide points. Extending the y-limit instead means the box
    sits in genuinely empty space and hides nothing.

    But extending the limit made matplotlib add ticks past the end of the
    metric's range: Spearman rho and cosine similarity were labelled 1.25 and
    1.50, and -Euclidean was labelled above 0. Those values do not exist. The
    limit is still extended (that is what makes room for the box), but ticks
    are truncated at `cap` — the largest value the metric can actually take:
        Spearman rho, cosine ->  1.0
        -Euclidean           ->  0.0
    """
    frac = HEADROOM if frac is None else frac
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * (1 + frac))
    lo2, hi2 = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if lo2 <= t <= min(cap, hi2)])


def scatter(ax, x, y, color, label):
    ax.scatter(x, y, color=color, s=45, alpha=.6, label=label, edgecolor="none")
    xg = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    fit, lo, hi = ols_band(x, y, xg)
    ax.plot(xg, fit, color=color, lw=2, alpha=.9)
    ax.fill_between(xg, lo, hi, color=color, alpha=.15, lw=0)


def main():
    prim = read(f"RAW_cortex_{METRIC}")
    sens = {m: read(f"RAW_cortex_{m}").reindex_like(prim) for m in SENSITIVITY}
    sub = read(f"RAW_subctx_{METRIC}").reindex_like(prim)
    psy_names, sud_names = list(prim.index), list(prim.columns)
    C.check_cluster_coverage(psy_names)

    # ---- audit the exclusion before anything is computed ----
    if EXCLUDE_AGGREGATE_SUD:
        if not _dropped_aggregate:
            raise RuntimeError(
                f"EXCLUDE_AGGREGATE_SUD is True but no aggregate column was found in "
                f"any input. Columns present: {sud_names}. Add the correct spelling to "
                f"AGGREGATE_SUD_NAMES — do not switch the flag off.")
        if len(sud_names) != N_SUBSTANCE_MAPS:
            raise RuntimeError(
                f"expected {N_SUBSTANCE_MAPS} substance-specific maps after the drop, "
                f"got {len(sud_names)}: {sud_names}")
        for tag, col in _dropped_aggregate:
            print(f"  dropped aggregate column '{col}' from {tag}")

    print(f"{len(psy_names)} x {len(sud_names)} = {prim.size} pairs; SUD: {sud_names}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=FIGSIZE)
    x = prim.to_numpy(float).ravel()

    # ---------------- Panel A: metric concordance ----------------
    scatter(axA, x, sens["cosine"].to_numpy(float).ravel(),
            SENS_COLORS["cosine"], SENS_LABEL["cosine"])
    axA.set_ylabel("Cosine similarity", fontsize=FS_LABEL, color=SENS_COLORS["cosine"])
    axA.tick_params(axis="y", labelcolor=SENS_COLORS["cosine"], labelsize=FS_TICK)
    axA2 = axA.twinx()
    scatter(axA2, x, sens["euclidean"].to_numpy(float).ravel(),
            SENS_COLORS["euclidean"], SENS_LABEL["euclidean"])
    axA2.set_ylabel("\u2212Euclidean similarity", fontsize=FS_LABEL,
                    color=SENS_COLORS["euclidean"])
    axA2.tick_params(axis="y", labelcolor=SENS_COLORS["euclidean"], labelsize=FS_TICK)

    lines = []
    for m in SENSITIVITY:
        r, p = mantel(prim.to_numpy(float), sens[m].to_numpy(float))
        lines.append(f"{SENS_LABEL[m]:<12s} r = {r:+.3f}, $p_{{Mantel}}$ = {p:.4f}")
        print(f"  {METRIC} vs {m:<10s} r = {r:+.3f}  p_Mantel = {p:.4f}")
    headroom(axA, cap=1.0)      # cosine similarity is bounded above by 1
    headroom(axA2, cap=0.0)     # -Euclidean distance cannot be positive
    axA.text(.03, .97, "\n".join(lines), transform=axA.transAxes, ha="left", va="top",
             fontsize=11.5, bbox=BOX, zorder=30).set_clip_on(False)
    axA.set_xlabel(f"Cortex {METRIC_LABEL} (primary)", fontsize=FS_LABEL)
    axA.set_title("A. Concordance with sensitivity metrics", fontsize=FS_TITLE, loc="left")
    axA.tick_params(axis="x", labelsize=FS_TICK)

    # ---------------- Panel B: cortex vs subcortex ----------------
    xc, yc = prim.to_numpy(float).ravel(), sub.to_numpy(float).ravel()
    labs = np.repeat(np.array(psy_names), prim.shape[1])
    for cname, members in C.CLUSTER_MEMBERS.items():
        idx = [i for i, d in enumerate(labs) if d in members]
        if idx:
            axB.scatter(xc[idx], yc[idx], color=C.CLUSTER_COLORS[cname], s=45,
                        alpha=.75, label=cname, edgecolor="none")
    xg = np.linspace(np.nanmin(xc), np.nanmax(xc), 100)
    fit, lo, hi = ols_band(xc, yc, xg)
    axB.plot(xg, fit, color="0.35", lw=2)
    axB.fill_between(xg, lo, hi, color="0.5", alpha=.15, lw=0)
    r_cs, p_cs = mantel(prim.to_numpy(float), sub.to_numpy(float))
    print(f"  cortex vs subcortex     r = {r_cs:+.3f}  p_Mantel = {p_cs:.4f}")
    headroom(axB, cap=1.0)      # Spearman rho is bounded above by 1
    axB.text(.03, .97, f"r = {r_cs:+.3f}\n$p_{{Mantel}}$ = {p_cs:.4f}",
             transform=axB.transAxes, ha="left", va="top", fontsize=11.5,
             bbox=BOX, zorder=30).set_clip_on(False)
    axB.axhline(0, color="0.8", lw=.8, zorder=0)
    axB.axvline(0, color="0.8", lw=.8, zorder=0)
    axB.set_xlabel(f"Cortex {METRIC_LABEL}", fontsize=FS_LABEL)
    axB.set_ylabel(f"Subcortex {METRIC_LABEL}", fontsize=FS_LABEL)
    axB.set_title("B. Cortex vs subcortex", fontsize=FS_TITLE, loc="left")
    # legend moved OUT of the axes: in panel B the points fill the lower-right
    # corner, so an in-axes legend was being read through the scatter
    axB.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=len(C.CLUSTER_MEMBERS), fontsize=11, columnspacing=1.4,
               handletextpad=0.4)
    axB.tick_params(labelsize=FS_TICK)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.tight_layout()
    stem = os.path.join(FIGDIR, f"RQ1_FigS_concordance_{METRIC}")
    for ext, dpi in (("png", 300), ("pdf", 400), ("svg", 400)):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    print(f"\nSaved -> {stem}.[png|pdf|svg]")


if __name__ == "__main__":
    main()