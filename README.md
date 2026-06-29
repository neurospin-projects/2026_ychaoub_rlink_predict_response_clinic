# R-LiNK — Prédiction de la réponse au lithium à partir de variables cliniques

**Internship project — NeuroSpin / CEA / Université Paris-Saclay**  
Supervised by: Eduard Duchesnay & Cathy Philippe  
Author: Yassir Chaoub 

---

## Context

Bipolar disorder is a chronic psychiatric condition affecting 1–3% of the world population, characterised by alternating manic and depressive episodes. Lithium is the reference mood stabiliser, but response is highly variable: roughly one third of patients are good responders, one third partial responders, and one third non-responders.

**Research question:** Can we identify, *before* treatment starts, which bipolar patients will respond to lithium?

This repository addresses **Objective 1**: predict lithium response from **clinical variables collected at inclusion (M00)**.

---

## Study design (R-LiNK)

| Phase | Timepoint | Data collected |
|---|---|---|
| Controlled phase | M00 — Inclusion | Clinical (inclusion, baseline, family history), OMIC + DNA (genotyping, metabolomic, methylomic, miRNA, RNA), Imaging (MRI-anat, MRI-DWI, MRI-lithium, MRI-MRS) |
| Controlled phase | Lithium introduction | Titration |
| Controlled phase | M03 — 3-month evaluation | Clinical, OMIC, Imaging |
| Ecological phase | Follow-up to Month 24 | Lithium response & tolerance evaluations, enhanced adherence program, salivary self-monitoring |
| Endpoint | Month 24 | **Response label** |

---

## Target variable

Response is defined as a binary outcome at end of follow-up:

- **GR** (Good Responder): 49 subjects
- **No_GR** (Non-Responder): 89 subjects , partial responders (PaR) are merged into this class; subjects with uncertain classification (UC) and missing labels are excluded.

Total: **n = 138 subjects** from **16 clinical centres**.

---

## Clinical data (M00)

Starting from >1800 raw variables (inclusion questionnaires, baseline assessments, family history), a filtering step retains **p = 290 variables**, reduced to **p = 210** after removing features with >40% missing values.

Variables are classified by type for type-aware imputation inside sklearn Pipelines:

| Type | Imputation strategy |
|---|---|
| Binary / categorical / ordinal | Mode |
| Continuous | Median |

### Notable variable groups

Three questionnaire families showed strong within-group correlation and were studied both included and excluded:

| Group | Full name | Description |
|---|---|---|
| **MARS** | Medication Adherence Report Scale | 10-item scale measuring medication adherence, intentional skipping, and attitudes toward medication |
| **TRQ** | Tablets Routine Questionnaire | Assesses daily pill-taking habits and frequency of forgotten doses; behavioural complement to MARS |
| **BMQ** | Beliefs about Medicines Questionnaire | 18-item scale exploring patient beliefs about their medications; predictive of long-term adherence |

---

## Machine learning pipeline

- **Cross-validation:** `StratifiedKFold(n_splits=5)` over all 138 subjects
- **Imputation:** type-aware `SimpleImputer` inside `ColumnTransformer`, embedded in each sklearn `Pipeline` 
- **Class imbalance:** handled via `class_weight="balanced"` or equivalent per-model mechanism
- **Metrics:** Balanced Accuracy, ROC-AUC, Recall (Responders), Recall (Non-Responders)

### Models benchmarked

Logistic Regression · ElasticNet · SVM RBF · Random Forest · Gradient Boosting · XGBoost · Multilayer Perceptron · TabPFN

### Key results (summary across feature sets)

| Feature set | Best model | Balanced Acc. | ROC-AUC |
|---|---|---|---|
| All M00 variables (p=172) | Random Forest | 57.6% | 0.569 |
| All M00 (missing <0.4, p=150) | Random Forest | **57.8%** | **0.608** |
| Without TRQ, BMQ, MARS | SVM RBF | **61.8%** | 0.428 |
| TDupont variable list | Random Forest | **58.5%** | **0.601** |
| Thibut list without BMQ/MARS | SVM RBF | **62.2%** | 0.416 · RF: **0.612** |

> Overall performance is modest (balanced accuracy ~50–62%), consistent with the difficulty of predicting lithium response from clinical data alone. Random Forest and SVM RBF are the most competitive models across feature sets.

---

## Repository structure

```
├── 01_make_dataset.py                  <- Data loading, filtering, type mapping
├── 02_descriptives_statistics.py       <- Univariate statistics by response group
├── 02_eda.py                           <- Exploratory data analysis (distributions, correlations)
├── 03_Analysis_visualization.ipynb     <- Visualisations (correlation matrices, univariate volcano plot)
├── M00_variable_categories.json        <- Type map for M00 features (binary/continuous/ordinal/nominal)
├── M03_variable_categories.json        <- Type map for M03 features
├── config.py                           <- Paths and global settings
├── make_variables_mapping.py           <- Generates type-aware variable mapping
├── pixi.toml                           <- Environment definition (pixi)
├── utils/                              <- Shared utilities (ML helpers, imputation, metrics)
├── docs/                               <- Additional documentation
└── reports/                            <- Output figures and result tables
```

---

## Installation

Install [pixi](https://pixi.sh):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Set up the environment:

```bash
cd 2026_ychaoub_rlink_predict_response_clinic
pixi install
pixi shell
```

Main dependencies: `python`, `scikit-learn`, `pandas`, `xgboost`, `tabpfn`, `statsmodels`, `seaborn`, `openpyxl`, `ipykernel`

---

## `01_make_dataset.py` — Data preparation pipeline

This script is the **single entry point for all data preparation**. It reads the raw R-LiNK eCRF files, builds the feature matrix, encodes the outcome, and saves three ready-to-use datasets. It must be run before any analysis script.

### Inputs

| File | Content |
|---|---|
| `dataset-clinical_mod-inclusion_version-3.tsv` | Demographics: `participant_id`, `CENTERNUM`, `AGE`, `SEX` |
| `dataset-clinical_mod-baseline_version-3.tsv` | Baseline clinical variables at M00 (psychiatric history, biology, questionnaires) |
| `dataset-clinical_mod-visits_form-visit_version-3.tsv` | Longitudinal visit data; only the `M3` timepoint is used here |
| `dataset-clinical_mod-baseline_form-preLi_tab-med_version-3.tsv` | Pre-lithium medication table |
| `dataset-clinical_mod-baseline_form-postLi_tab-famhist_version-3.tsv` | Family psychiatric history (one row per relative) |
| `dataset-outcome_version-4.tsv` | Lithium response label at end of follow-up |
| `study-rlink_dataset-clinical_type-summary_version-20260220.xlsx` | Supplementary derived variables (F. Bellivier, 2026): BMI, illness duration, episode density, smoking, QIDS/BRMS total scores, delta medication variables |

All TSV files are read with `na_values=["ND"]` so that the eCRF "not determined" code is immediately converted to `NaN`.

### Outputs

Three CSV files are saved to `./data/`:

| File | Shape | Content |
|---|---|---|
| `Rlink_version3_type_Clinic_timepoint_M00_v20260602.csv` | 138 × ~172 | Baseline (M00) features only + response label |
| `Rlink_version3_type_Clinic_timepoint_M03_v20260602.csv` | 138 × (M03 cols + 2) | 3-month follow-up features only + response label |
| `Rlink_version3_type_Clinic_timepoint_M00_M03_v20260602.csv` | 138 × ~210 | Combined M00 + M03 + response label |

### Pipeline steps

The `main()` function runs eight sequential steps:

#### Step 1 — Column selection

Variables are selected from each source file using three explicit lists defined at the top of the script:

- **`BASELINE_VARS_CLINICAL`** (~110 columns): thymic state at inclusion, physical examination (weight, height, BMI, blood pressure), sociodemographics (age at onset, relationship status, education, employment), biology (renal panel, thyroid, lipids, haematology), and psychiatric/treatment history from the Post-Lithium Interview (PLI) covering two historical periods (2 years before inclusion, and lifetime).
- **`BASELINE_VARS_QUESTIONNAIRES`** (~55 columns): QIDS (depression severity), BRMS (mania severity), Columbia SSRS (suicide risk, 6 items), BPRS (global psychiatric severity), MARS (10-item medication adherence), TRQ (daily pill-taking routine, 9 items), BMQ (18 beliefs about medicines items), WHOA/lifestyle items.
- **`VISIT_M03_VARS`**: same clinical domains as M00 but collected 12 weeks after lithium introduction. Columns are renamed with a `_M03` suffix to distinguish them from baseline.

#### Step 2 — Outcome construction (`build_outcome`)

The raw outcome variable `Response.Status.at.end.of.follow.up` has four categories:

```
GR  (Good Responder)      → 49 subjects  → kept as GR  → encoded 0
PaR (Partial Responder)   → 33 subjects  → recoded to No_GR
NR  (Non Responder)       → 56 subjects  → recoded to No_GR
UC  (Unclassified)        → 21 subjects  → excluded
```

After recoding: **49 GR (0) vs 89 No_GR (1)**. Subjects with a missing outcome are also excluded at the merge step (step 6). A `LabelEncoder` is applied so the `response` column is always integer (0/1).

#### Step 3 — Feature engineering

Three derived feature blocks are added before the main merge:

**BMQ subscores** (`compute_bmq_subscores`): the 18 raw BMQ items are aggregated into four validated subscores:

| Subscore | Items | Interpretation |
|---|---|---|
| `BMQ_NECESSITY` | 1–5 | Perceived necessity of the medication |
| `BMQ_PREOCCUPATION` | 6–10 | Concerns about the medication |
| `BMQ_DIFFERENTIAL` | necessity − concerns | Net adherence predictor |
| `BMQ_GENERAL` | 11–18 | General beliefs about medicines |

**MARS total** (`compute_mars_total`): sums the 10 MARS items into `MARS_TOTAL`.

**Family history aggregation** (`compute_family_history_features`): the family history file has one row per relative, so it is aggregated to participant level via three features: `fhist_count` (total number of relatives with psychiatric history), `fhist_ratio_h` (proportion of male relatives), and `fhist_repli` (proportion of relatives with a positive lithium response). Subjects with no family history record receive 0 for all three.

#### Step 4 — Supplementary data merge

The supplementary Excel file (F. Bellivier, 2026) provides derived variables not directly computable from the raw eCRF: BMI at M00 and M03, delta-BMI, illness duration, episode density, hospitalisation density, smoking status, pre-processed QIDS and BRMS total scores, and delta variables for five drug classes (antipsychotics, antidepressants, mood stabilisers, neuroleptics, benzodiazepines). 

Before merging, a consistency check verifies that `AGE` and `SEX` match between the eCRF inclusion file and the supplementary file for every participant. Any mismatch raises an `AssertionError`.

#### Step 5 — Deduplication of redundant columns

After the merge, some columns exist in both the baseline file and the supplementary file (e.g. `QIDSTSC_PRELI` and `QIDS_TotalScore_M00` both represent the QIDS score at M00). The supplementary version is always preferred (more processed). The `REDUNDANT_BASELINE_COLS` dictionary maps old baseline column names to their canonical supplementary equivalents; baseline versions are dropped and supplementary versions are renamed accordingly.

#### Step 6 — Outcome merge and subject filtering

The binary response label is merged on `participant_id` using an inner join. This naturally excludes the 30 subjects who have no outcome label (UC + missing), leaving **n = 138 subjects**.

Shape at this point: `(138, 299)`.

#### Step 7 — Domain-specific cleaning (`clean_raw_values`)

Before imputation, a small number of eCRF-specific codes are cleaned:

| Column(s) | Raw code | Meaning | Action |
|---|---|---|---|
| `FHIST_PLI` | `9` | No family history (not "unknown") | Recoded to `0` |
| `MOOD_PLI`, `ANTIPSY_PLI`, `NEUROL_PLI`, `ANTIDEP_PLI`, `BENZOS_PLI` | `9` | Not applicable | Recoded to `NaN` |
| `RCY1_PLI`, `RCY2_PLI` | `2` | Not assessed | Recoded to `NaN` |
| All columns | `"ND"` string | Not determined | Recoded to `NaN` (already handled at load time, this step is a safety net) |

#### Step 8 — Missing-value filtering (`drop_high_missingness_columns`)

A curve of "number of features retained vs. missing-value threshold" is plotted (saved to `reports/`) to justify the threshold choice. Columns with more than **40% missing values** are dropped. This removes ~89 columns, leaving **p = 210 features**.

The per-row missing rate after filtering peaks at ~37%, which means no subject needs to be dropped — all 138 rows are retained. Imputation of remaining missing values is handled downstream, inside the sklearn `Pipeline` of each classifier (not in this script), to avoid data leakage.

#### Step 9 — Dataset splitting and export

The 210-feature matrix is split into three datasets based on column suffixes:

- **M00 only**: drop all columns ending in `_M03` or starting with `Delta`
- **M03 only**: keep `participant_id`, `response`, and all `_M03` / `Delta` columns
- **M00 + M03**: full 210-feature matrix

All three are exported as CSV files.

### Shape trace (assertions)

The script contains `assert` statements at every major step to catch unexpected changes in data shape:

```
inclusion_df      (168, 4)
baseline_df       (168, 162)
after inclusion+baseline merge   (168, 165)
after family history merge        (168, 173)
after M03 merge                   (168, 280)
after supplementary merge         (168, 303)
after outcome merge               (138, 299)   ← 30 subjects excluded
after missingness filter          (138, 210)
```

---
