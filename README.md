# META — PSY–SUD neuroanatomical similarity

Code and data for the study on macroscale neuroanatomical convergence between
psychiatric disorders (PSY) and substance use disorders (SUD), based on
ENIGMA-derived cortical and subcortical effect-size (Cohen's *d*) maps.

The pipeline quantifies pairwise PSY–SUD morphometric similarity (RQ1),
localises regional convergence and tests its alignment with cortical
transcriptional, functional and microstructural organisation (RQ2), and
relates the resulting hierarchy to developmental timing and genetic
correlation (RQ3).

---

## Run order

Scripts have no command-line arguments. Each has a `CONFIG` block at the top;
edit it and press Run. Within a research question, run in the order below —
figures read the CSVs the analysis scripts write.

### RQ0 — descriptive

| # | script | produces |
|---|--------|----------|
| 1 | `RQ0_PSY_age_sex_barplots.py` | sample characteristics, psychiatric panel |
| 2 | `RQ0_SUD_agesex_barplots.py` | sample characteristics, substance panel |
| 3 | `RQ0_barplot_cohend.py` | effect-size barplots by cluster |
| 4 | `RQ0_all_cohend_maps.py` | per-disorder cortical surfaces |

### RQ1 — PSY × SUD similarity

| # | script | produces |
|---|--------|----------|
| 1 | `RQ1_similarity_pspin_brainsmash.py` | `RAW_`, `PVAL_`, `pFDR_` cortex, both nulls |
| 2 | `RQ1_raw_subctx.py` | `RAW_subctx_*` (descriptive, no null) |
| 3 | `RQ1_similarity_specificity.py` | Δ specificity, both benchmarks |
| 4 | `RQ1_similarity_suppl_table.py` | `SUPP_block_table.csv` |
| 5 | `RQ1_figure1_cortex_updated_combined.py` | Panel A heatmap + pair ranking |
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
| 4 | *(missing — see below)* | Figure 3 panel C |
| 5 | `RQ2_shared_PCA.py` | PC1 vs mean-based shared map |
| — | `RQ2_shared_surfaces.py` | cortical/subcortical renders (needs ENIGMA Toolbox) |

### RQ3 — external validation

| # | script | produces |
|---|--------|----------|
| 1 | `RQ3_ageonset_regression.py` | age-of-onset models incl. mixed effects |
| 2 | `RQ3_gen_corr.py` | genetics figure, comorbidity statistics, S22/S23 rankings |
| 3 | *(missing — see below)* | cluster-stratified genetics supplementary |

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
cached in the same folder and shared by every script.

---

## Corrections carried by this version

These were found by auditing the committed scripts. Each one changed published
numbers.

1. **Hemisphere offset in the spin.** Right-hemisphere spin indices were not
   offset by +34, so every surrogate filled all 34 RH positions with LEFT
   hemisphere values. Present in 5 of the 6 spin-building RQ2 scripts and in
   both RQ1 similarity scripts. `RQ1_common.make_spins` applies the offset and
   asserts it on every load, including on cached files.

2. **SUD double-counting.** `SUD.xlsx` carries a generic aggregate `SUD`
   column alongside six substance-specific ones; averaging all seven
   overweights pooled patients. The aggregate is dropped wherever a SUD *mean*
   is formed. It is deliberately retained in the RQ1 pairwise panel, which
   displays all seven columns — note in the caption that it is a weighted
   composite of the other six and not independent evidence.

3. **BrainSMASH geometry.** The distance matrix is built from RAW centroids.
   One script used unit-sphere–projected centroids, which is not a uniform
   rescaling and distorts the variogram. The unit sphere is used only for
   spins, where a rotation requires it.

4. **`z` is an effect size.** The old pipeline reported
   `z = sign · norm.isf(p)`, a deterministic function of *p* saturating at
   3.719, and ranked results on it. `z` is now `(obs − mean(null)) / sd(null)`
   and rankings use raw ρ.

5. **Filename collisions.** Cortex and subcortex scripts wrote
   `RANK_spearman_by_<SUD>.csv` into the same folder, so the surviving file
   depended on run order. Every output now carries compartment and null in its
   name.

6. **Cluster definitions.** Declared in four places with three spellings of the
   same four groups. Now only in `RQ1_common.CLUSTER_MEMBERS`, with
   `check_cluster_coverage()` raising instead of silently dropping a renamed
   disorder.

---

## Known limitations to carry into the manuscript

- **PD replaces PTSD.** Panic disorder (ENIGMA-Anxiety mega-analysis) replaces
  PTSD in the cortical and subcortical maps. Its Cohen's *d* values are
  published to two decimals, giving 15 distinct values across 68 parcels; the
  attainable Spearman ceiling is 0.996, so the attenuation is negligible, but
  regional precision is lower than for the other maps. Its *n* differs by
  modality (≈ 936 cortical, ≈ 1132 subcortical of 1146) and its age range is
  10–66, i.e. the sample is not adult-only.

- **PD is absent from the genetic analyses.** The Grotzinger panel contains no
  panic disorder phenotype, so RQ3 genetics runs on 7 disorders and the
  Mood/Anxiety cluster is represented by MDD alone. CHR is likewise absent.
  This is a coverage gap, not a null result.

- **Comorbidity is exploratory.** The workbook is still keyed on PTSD, and no
  source supplies P(SUD | PD) per substance in a form comparable with the other
  rows. Reported descriptively, no figure.

- **Age of onset depends on the index.** The association holds on peak age
  (p = 0.004) but not on median age (p = 0.061). Report both, and justify the
  choice of index a priori.

- **The mixed-effects model is not a stronger test.** Peak age is constant
  within each disorder-by-population group and the random intercept sits on
  that same grouping, so the fixed effect and the variance component compete
  for the same signal. Primary inference remains the n = 15 group-mean OLS.

- **Pediatric results do not generalise.** The pediatric mean map shows no
  significant alignment with C3 (r = −0.060, q = 0.71).

- **Neurodevelopmental disorders do not participate in the shared pattern.**
  r(PSY, SUD) = 0.02 for that cluster; ASD's PC1 loading is ≈ 0.

- **A null on C1 is cancellation, not absence.** The adult PSY and SUD
  components correlate with C1 in opposite directions and the mean cancels
  them. Figure 3 panel C shows both components for this reason.

---

## Input data (`data/raw/`)

| file | contents |
|------|----------|
| `PSY_adults.xlsx` | 9 adult psychiatric maps, 68 cortical + 14 subcortical rows |
| `PSY_adults_ctx.xlsx` | additional adult cortical map(s) |
| `PSY_adolescents.xlsx`, `PSY_adolescents_ctx.xlsx` | pediatric maps |
| `SUD.xlsx` | 6 substance maps + generic aggregate |
| `PSY_age_of_onset.xlsx` | peak, median and IQR age of onset |
| `PSY_SUD_genetic_corr.xlsx`, `PSY_PSY_genetic_corr.xlsx` | genetic correlations |
| `PSY_SUD_comorbidity_prevalence.xlsx`, `SUD_general_prevalence.xlsx` | comorbidity |
| `ahba_dme_scores_in_dk.csv` | AHBA C1–C3, 34 LH parcels, mirrored on load |
| `mica_hc100_gradient-FC.csv`, `-MPC.csv` | functional and microstructural gradients |
| `centroids_ctx_68.mat` | parcel centroids for spins and BrainSMASH |

Region order in all map workbooks is the standard ENIGMA/FreeSurfer DK order
(LH 1–34 then RH 35–68), **not** strictly alphabetical: `parahippocampal`
precedes `paracentral`, and `frontalpole`, `temporalpole`, `transversetemporal`
and `insula` come last. Subcortical rows are alphabetical (accumbens,
amygdala, caudate, hippocampus, pallidum, putamen, thalamus), L then R,
ventricles excluded.

---

## Environment

```bash
pip install -r requirements.txt
```

`enigmatoolbox` is needed only by the surface-rendering scripts; everything
else runs without it.