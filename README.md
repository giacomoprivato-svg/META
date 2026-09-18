# META — PSY–SUD neuroanatomical similarity

Code and data for the study on macroscale neuroanatomical convergence between
psychiatric disorders (PSY) and substance use disorders (SUD), based on
ENIGMA-derived cortical and subcortical effect-size (Cohen's *d*) maps.

The pipeline quantifies pairwise PSY–SUD morphometric similarity (RQ1),
localises regional convergence and tests its alignment with cortical
transcriptional, functional and microstructural organisation (RQ2), and
relates the resulting hierarchy to developmental timing, genetic correlation
and clinical comorbidity (RQ3).

---

## Layout

```
data/raw/            input workbooks and surface geometry (tracked)
src/                 all analysis and figure scripts (26 files)
ALL_outputs_RQ1/     RQ1 matrices, rankings, specificity, surrogate cache
ALL_outputs_RQ2/     RQ2 shared maps, gradient statistics, PCA comparison
figures/             publication figures (+ figures/surfaces/)
results/             RQ0 barplot panels
```

Only `data/raw/` and `src/` are tracked. Everything the scripts write is
regenerated from them and is ignored by `.gitignore`.

---

## Run order

Scripts have no command-line arguments. Each has a `CONFIG` block at the top;
edit it and press Run. Paths are resolved relative to the script's own
location, so the repo can sit anywhere. Within a research question, run in the
order below — figures read the CSVs the analysis scripts write.

### RQ0 — descriptive

| # | script | produces |
|---|--------|----------|
| 1 | `RQ0_PSY_age_sex_barplots.py` | sample characteristics, psychiatric panel |
| 2 | `RQ0_SUD_agesex_barplots.py` | sample characteristics, substance panel |
| 3 | `RQ0_barplot_cohend.py` | effect-size barplots by cluster (`results/RQ1_BARPLOTS_FINAL/`) |
| 4 | `RQ0_all_cohend_maps.py` | per-disorder cortical surfaces |

### RQ1 — PSY × SUD similarity

| # | script | produces |
|---|--------|----------|
| 1 | `RQ1_cortex_similarity.py` | `RAW_`, `PVAL_`, `pFDR_` cortex, both nulls |
| 2 | `RQ1_raw_subctx.py` | `RAW_subctx_*` (descriptive, no null) |
| 3 | `RQ1_similarity_specificity.py` | Δ specificity, both benchmarks |
| 4 | `RQ1_similarity_suppl_table.py` | `SUPP_block_table.csv` |
| 5 | `RQ1_heatmap.py` | Panel A heatmap + pair ranking |
| 6 | `RQ1_panelfingerprints.py` | Panel F cluster fingerprints |
| 7 | `RQ1_similarity_spec_figure.py` | specificity figure |
| 8 | `RQ1_suppl_concordance_subctx.py` | metric concordance, cortex vs subcortex |
| 9 | `RQ1_joint_matrix_suppl.py` | 16 × 16 joint clustermap |

`RQ1_common.py` is imported by all of them and is the single definition of
spins, BrainSMASH surrogates, similarity, permutation *p*, BH-FDR and the
clinical clusters. **Do not copy its functions into a script.** Every bug
listed under "Corrections" below existed because someone did.

### RQ2 — shared maps and gradients

| # | script | produces |
|---|--------|----------|
| 1 | `RQ2_shared.py` | shared maps, contribution index, systematic leave-one-out |
| 2 | `RQ2_gradients.py` | analysis A (shared map × 5 gradients) and B (panel D) |
| 3 | `RQ2_panel_heatmap.py` | Figure 3 panel D |
| 4 | `RQ2_shared_gradients_figure.py` | Figure 3 panel C (shared map vs 5 gradients, components shown) |
| 5 | `RQ2_shared_PCA.py` | PC1 vs mean-based shared map |
| — | `RQ2_shared_surfaces.py` | cortical/subcortical renders (needs ENIGMA Toolbox) |

`RQ2_common.py` holds the gradient loaders (AHBA C1–C3, FC, MPC), the
hemisphere mirroring of the AHBA maps and `GRADIENT_METRIC`. It imports
`RQ1_common` for the nulls rather than redefining them.

### RQ3 — external validation

| # | script | produces |
|---|--------|----------|
| 1 | `RQ3_ageonset_regression.py` | age-of-onset models incl. mixed effects |
| 2 | `RQ3_gen_corr.py` | genetics figure, comorbidity statistics, S22/S23 rankings |
| 3 | `RQ3_gen_corr_cluster.py` | cluster-stratified genetics supplementary figure |
| 4 | `RQ3_comorbidity.py` | standalone comorbidity scatter (single panel, exploratory) |

`RQ3_comorbidity.py` reads `RAW_cortex_spearman.csv` from
`ALL_outputs_RQ1/adults_all/`, so RQ1 must have been run first. Its two
toggles (`SHOW_FIT`, `SHOW_STATS`) sit at the top of the file.

---

## Conventions

**Primary metric.** RQ1 uses Spearman ρ; cosine and −Euclidean are sensitivity
metrics. RQ2 gradient correlations use Pearson *r*, set explicitly in
`RQ2_common.GRADIENT_METRIC` — this was always what the code computed but was
never stated.

**Dual-null framework.** Every spatial test runs under a spin permutation and
under BrainSMASH surrogates. An effect is called robust only if it survives
BH-FDR under **both**. BrainSMASH is historically the more conservative of the
two.

**Which map is permuted.** The alteration map is randomised and the target
held fixed, in RQ1 and RQ2 alike. In RQ2 this reverses the earlier convention;
the reason is in `RQ2_gradients.py` and is not cosmetic — the AHBA C1–C3 maps
are mirrored across hemispheres (r(LH,RH) = 1.000 exactly), so building a
spatial null *from* them is unsound in both frameworks.

**FDR families** are declared in advance, one per inferential question, never
chosen after seeing results. RQ2 has two: the adult shared map against five
gradients (5 tests), and the seven component maps against C1–C3 (21 tests).

**Surrogate cache.** BrainSMASH surrogates are written to
`ALL_outputs_RQ1/_cache/` keyed by a SHA-1 of the map's bytes, so a map is
generated once ever, across scripts and across reruns. Change one region and
only that map regenerates. Cost is ≈ 140 s per map at n = 10 000. Spins are
cached in the same folder (`spins_ctx_68_fixed_n<N>_seed<S>.npy`) and shared by
every script; the `_fixed` in the name distinguishes them from the buggy
pre-correction spins, and the offset is re-asserted on every load, including
from cache.

**CSV reading.** All CSV loads use `pd.read_csv(..., sep=None, engine="python")`.
Excel under an Italian locale writes semicolon-delimited CSVs and this is the
only reliable way to read both.


---

## Input data (`data/raw/`)

| file | contents | read by |
|------|----------|---------|
| `PSY_adults.xlsx` | adult psychiatric maps, 68 cortical + 14 subcortical rows | RQ0, RQ1, RQ2 |
| `PSY_adults_ctx.xlsx` | additional adult cortical map(s) | RQ1 |
| `PSY_adolescents.xlsx`, `PSY_adolescents_ctx.xlsx` | pediatric maps | RQ1, RQ2 |
| `SUD.xlsx` | 6 substance maps + generic aggregate | RQ0, RQ1, RQ2 |
| `PSY_age_of_onset.xlsx` | peak, median and IQR age of onset | RQ3 |
| `PSY_SUD_genetic_corr.xlsx`, `PSY_PSY_genetic_corr.xlsx` | genetic correlations | RQ3 |
| `PSY_SUD_comorbidity_prevalence.xlsx`, `SUD_general_prevalence.xlsx` | comorbidity | RQ3 |
| `ahba_dme_scores_in_dk.csv` | AHBA C1–C3, 34 LH parcels, mirrored on load | RQ2 |
| `mica_hc100_gradient-FC.csv`, `-MPC.csv` | functional and microstructural gradients | RQ2 |
| `centroids_ctx_68.mat` | parcel centroids for spins and BrainSMASH | `RQ1_common` |
| `spins_ctx_10000.mat` | precomputed spin indices | `RQ1_common` |

`PSY_adults_pvalue.xlsx`, `SUD_pvalue.xlsx`, `PSY_adolescents_pvalue.xlsx` and
`PSY_adolescents_ctx_pvalue.xlsx` are tracked for provenance but are not read
by any current script.

Region order in all map workbooks is the standard ENIGMA/FreeSurfer DK order
(LH 1–34 then RH 35–68), **not** strictly alphabetical: `parahippocampal`
precedes `paracentral`, and `frontalpole`, `temporalpole`,
`transversetemporal` and `insula` come last. Subcortical rows are alphabetical
(accumbens, amygdala, caudate, hippocampus, pallidum, putamen, thalamus),
L then R, ventricles excluded.

Workbooks are read with `pd.read_excel(fp, sheet, header=1, index_col=0)`.

---

## Environment

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

`requirements.txt` covers everything except the ENIGMA Toolbox, which is not on
PyPI and is needed only by the surface-rendering scripts
(`RQ0_all_cohend_maps.py`, `RQ2_shared_surfaces.py`). Install it separately if
you need those figures:

```bash
git clone https://github.com/MICA-MNI/ENIGMA.git
cd ENIGMA && pip install .
```

Everything else runs without it.

To record the exact environment used for a given set of results:

```bash
pip freeze > requirements-lock.txt
```

---

## Citation

Data sources: ENIGMA consortium case-control effect-size maps; AHBA
transcriptional gradients (Dear et al., *Nat Neurosci* 2024); MICA-MNI
functional and microstructural gradients; Grotzinger et al. genetic
correlations.