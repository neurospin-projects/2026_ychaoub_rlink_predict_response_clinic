# R-LINK STUDY — Clinical Data Preparation Pipeline
# Purpose : Merge, clean, and save the clinical dataset (baseline M00 + visit
#           M03) used to predict lithium response in bipolar disorder.
#
# Input files (from R-Link eCRF repository):
#   - dataset-clinical_mod-inclusion_version-3.tsv      → demographics
#   - dataset-clinical_mod-baseline_version-3.tsv       → baseline clinical data (M00)
#   - dataset-clinical_mod-visits_form-visit_version-3.tsv → follow-up visits
#   - dataset-clinical_mod-baseline_form-preLi_tab-med_version-3.tsv → medications
#   - dataset-clinical_mod-baseline_form-postLi_tab-famhist_version-3.tsv → family history
#   - dataset-outcome_version-4.tsv                     → lithium response outcome
#   - study-rlink_dataset-clinical_type-summary_version-20260220.xlsx → supplementary data (F. Bellivier, 2026)
#
# Output files (saved to ./data/)
#
#   - Rlink_version3_type_Clinic_timepoint_M00_v20260602.xlsx
#       Clinical dataset containing baseline variables only (M00)
#
#   - Rlink_version3_type_Clinic_timepoint_M03_v20260602.xlsx
#       Clinical dataset containing follow-up variables only (M03)
#
#   - Rlink_version3_type_Clinic_timepoint_M00_M03_v20260602.xlsx
#       Combined clinical dataset containing both M00 and M03 variables
#
#
# Authors  : Yassir CHAOUB
# Date     : 2026


# 0. IMPORTS

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 1. CONFIGURATION — Paths
# Adjust these paths to match your local or server setup.

INPUT_DIR   = "/neurospin/rlink/PUBLICATION/rlink-ecrf/"
INPUT_SUPP  = "/neurospin/signatures/2026_ychaoub_rlink_predict_response_clinic/data/study-rlink_dataset-clinical_type-summary_version-20260220.xlsx"
OUTPUT_DIR  = "./data/"


# 2. VARIABLE LISTS
# These lists define which columns are loaded from each source file.
# Grouped by theme for readability.

# --- 2.1 Baseline (M00) — clinical and psychiatric variables ---

BASELINE_VARS_CLINICAL = [
    "participant_id",
    # Thymic state at inclusion
    "MOODYN_PRELI",     # Current mood episode (yes/no)
    "TYPEP_PRELI",      # Type of episode
    "PS_PRELI",         # With psychotic symptoms
    "MIX_PRELI",        # With mixed characteristics
    "DATSTEPD_PRELI",   # Date step depression
    "HOSP_PRELI",       # Currently hospitalized
    # Physical examination
    "WEIGHT_PRELI",
    "HEIGHT_PRELI",
    "WAIST_PRELI",
    "SBP_PRELI",        # Systolic blood pressure
    "DBP_PRELI",        # Diastolic blood pressure
    # Sociodemographic
    "RELSTAT_PRELI",    # Relationship status
    "ETHNICITY_PRELI",
    "CORG_PRELI",       # Country of origin
    "LIVSIT_PRELI",     # Living situation
    "RESIDENCE_PRELI",
    "SCHOOL_PRELI",     # Highest educational qualification
    "JOB_PRELI",        # Employment status
    "EVNT_PRELI",       # Recent stressful life event (<12 months)
    # Current treatment
    "CURRMED_PRELI",    # Current medications (yes/no)
    # Biology — renal / metabolic panel
    "NA_PRELI", "K_PRELI", "CL_PRELI", "CA_PRELI",
    "PROTEINS_PRELI", "UREA_PRELI", "CREAT_PRELI",
    "EGFR_PRELI", "MDRD_PRELI", "CKDEPI_PRELI",
    # Biology — thyroid
    "TSH_PRELI", "T3_PRELI", "T4_PRELI",
    # Biology — lipid / glycemic panel
    "GLY_PRELI", "TGC_PRELI", "HDL_PRELI", "LDL_PRELI",
    # Biology — haematology
    "BHCG_PRELI", "WBC_PRELI", "HB_PRELI", "HT_PRELI", "PLT_PRELI",
    "NP_PRELI", "EOS_PRELI", "LYMPH_PRELI", "MONO_PRELI",
    # Psychiatric and treatment history (Post-Lithium Interview = PLI)
    "FHIST_PLI", "MOOD_PLI", "ANTIPSY_PLI", "NEUROL_PLI", "ANTIDEP_PLI",
    "BENZOS_PLI", "PHCMBY_PLI",
    # Episode history — period 1 (2 years before inclusion)
    "RCY1_PLI",                                                 # Rapid cycling
    "MDE1_PLI", "MDEH1_PLI", "MDEPS1_PLI", "MDEMC1_PLI", "MDETD1_PLI",  # MDE
    "HYPOE1_PLI", "HYPOEH1_PLI", "HYPOEMC1_PLI", "HYPOETD1_PLI",         # Hypomanic
    "MANE1_PLI", "MANEH1_PLI", "MANEPS1_PLI", "MANEMC1_PLI", "MANETD1_PLI",  # Manic
    "NBH1_PLI", "TDH1_PLI", "OUTW1_PLI", "NBS1_PLI",
    "AD1_PLI", "SUD1_PLI", "MC1_PLI",
    # Episode history — lifetime (period 2)
    "RCY2_PLI", "AGESTBD2_PLI", "BDNOW2_PLI", "AGENDBD2_PLI",
    "MDE2_PLI", "MDEH2_PLI", "MDEPS2_PLI", "MDEMC2_PLI", "MDETD2_PLI",
    "AGEMDE2_PLI", "HYPOE2_PLI", "HYPOEH2_PLI", "HYPOEMC2_PLI",
    "HYPOETD2_PLI", "AGEHYPOE2_PLI",
    "MANE2_PLI", "MANEH2_PLI", "MANEPS2_PLI", "MANEMC2_PLI", "MANETD2_PLI",
    "AGEMANE2_PLI", "PATSEQM2_PLI", "NBH2_PLI", "TDH2_PLI",
    "AGESTBH2_PLI", "OUTW2_PLI", "NBS2_PLI", "AGES2_PLI",
    "AD2_PLI", "SUD2_PLI", "MC2_PLI",
]

BASELINE_VARS_QUESTIONNAIRES = [
    # Depression severity
    "QIDSTSC_PRELI",    # QIDS total score
    "BRMSTSC_PRELI",    # BRMS (Bech-Rafaelsen Mania Scale) total score
    # Suicide risk (Columbia SSRS)
    "SSRS1_PRELI", "SSRS2_PRELI", "SSRS3_PRELI", "SSRS4_PRELI",
    "SSRS5_PRELI", "SSRS6_PRELI", "SSRS6Y_PRELI",
    # General psychiatric symptom severity
    "BPRSTSC_PRELI",    # BPRS total score
    # Medication adherence — MARS (10-item)
    "MARS1_PRELI", "MARS2_PRELI", "MARS3_PRELI", "MARS4_PRELI", "MARS5_PRELI",
    "MARS6_PRELI", "MARS7_PRELI", "MARS8_PRELI", "MARS9_PRELI", "MARS10_PRELI",
    # Medication adherence — TRQ
    "TRQ_PRELI", "TRQ1_PRELI", "TRQ2_PRELI", "TRQ4_PRELI",
    "TRQ5_PRELI", "TRQ6_PRELI", "TRQ7_PRELI", "TRQ1B_PRELI", "TRQ2B_PRELI",
    # Beliefs about Medication Questionnaire — BMQ (18 items)
    "BMQ1_PRELI",  "BMQ2_PRELI",  "BMQ3_PRELI",  "BMQ4_PRELI",  "BMQ5_PRELI",
    "BMQ6_PRELI",  "BMQ7_PRELI",  "BMQ8_PRELI",  "BMQ9_PRELI",  "BMQ10_PRELI",
    "BMQ11_PRELI", "BMQ12_PRELI", "BMQ13_PRELI", "BMQ14_PRELI", "BMQ15_PRELI",
    "BMQ16_PRELI", "BMQ17_PRELI", "BMQ18_PRELI",
    # Lifestyle (WHOA)
    "WHOA1A_PLI", "WHOA1B_PLI", "WHOA1C_PLI", "WHOA1D_PLI",
    "WHOA1E_PLI", "WHOA1F_PLI", "WHOA1G_PLI", "WHOA1H_PLI",
]

# --- 2.2 Visit M03 variables (3-month follow-up) ---
# Column names differ from baseline; renamed later with suffix _M03.

VISIT_M03_VARS = [
    "participant_id",
    # Physical
    "WEIGHT", "WAIST", "SBP", "DBP",
    # Biology — renal panel
    "K", "CL", "CA", "PROTEINS", "UREA", "CREAT", "EGFR", "MDRD", "CKDEPI",
    # Biology — thyroid
    "TSH", "T3", "T4",
    # Biology — metabolic / haematology (suffixed with "2" in visits file)
    "NA2", "GLY2", "TGC2", "HDL2", "LDL2", "BHCG2",
    "WBC2", "HB2", "HT2", "PLT2", "NP2", "EOS2", "LYMPH2", "MONO2",
    # Treatment and sociodemographic (repeated at M03)
    "CURRMED", "RELSTAT", "ETHNICITY", "CORG", "LIVSIT",
    "RESIDENCE", "SCHOOL", "JOB", "EVNT", "PHCMBY",
    # Episode history
    "MDE2", "MDEH2", "MDEPS2", "MDEMC2", "HYPOE2", "HYPOEH2", "HYPOEMC2",
    "MANE2", "MANEH2", "MANEPS2", "MANEMC2",
    "NBH2", "TDH2", "RCY2", "AGESTBD2",
    # Suicide risk (SSRS)
    "SSRS1", "SSRS2", "SSRS3", "SSRS4", "SSRS5", "SSRS6", "SSRS6Y",
    # BPRS
    "BPRSTSC",
    # Medication adherence — MARS (suffixed with "V" at M03)
    "MARS1V", "MARS2V", "MARS3V", "MARS4V", "MARS5V",
    "MARS6V", "MARS7V", "MARS8V", "MARS9V", "MARS10V",
    # TRQ
    "TRQ", "TRQ1", "TRQ2", "TRQ3", "TRQ4", "TRQ5", "TRQ6", "TRQ7",
    # BMQ
    "BMQ1",  "BMQ2",  "BMQ3",  "BMQ4",  "BMQ5",  "BMQ6",  "BMQ7",  "BMQ8",
    "BMQ9",  "BMQ10", "BMQ11", "BMQ12", "BMQ13", "BMQ14", "BMQ15", "BMQ16",
    "BMQ17", "BMQ18",
    # WHOA
    "WHOA1A", "WHOA1B", "WHOA1C", "WHOA1D",
    "WHOA1E", "WHOA1F", "WHOA1G", "WHOA1H",
]

# --- 2.3 Supplementary file — variables to import ---
# Source: F. Bellivier (2026). Duplicate columns (AGE, SEX) already in
# inclusion file are excluded. Free-text disease category columns are also
# excluded (not used in modelling).

SUPP_VARS_TO_KEEP_RAW = [
    "participant_id",
    "BMI_M00", "BMI_M03", "DeltaBMI", "Delta_BMI_impute",
    "PHCMBY_PLI-ComorbiditeSomatique",
    "PSYHLTH_PLI-ComorbiditePsychiatrique",
    "AgeAtOnset", "NumberPreviousEpisodes", "DurationIllness",
    "DensityEpisodes", "NbHospitalizationsLifetime", "DensityHospit",
    "SmokingStatus-WHOA1A_PLI",
    "SuicideAttempts(Yes/No)",
    "QIDS_TotalScore_M00", "QIDS_W1_M03",
    "BRMS_TotalScore_M00", "BRMS_W1_M03",
    "DeltaAPA", "DeltaATD", "DeltaAC", "DeltaNLP", "DeltaBZD",
]

SUPP_VARS_TO_DROP = [
    # Already in inclusion file → avoid duplicate columns after merge
    "AGE", "SEX",
    # Free-text disease category columns — not useful for modelling
    "CTGY-CategorieComorbiditeSomatique_n°1",
    "CTGY-CategorieComorbiditeSomatique_n°2etplus",
    "DISEASE-PathologieComorbiditeSomatique_n°1",
    "DISEASE-PathologieComorbiditeSomatique_n°2etplus",
    "DISRDR-Trouble_n°1",
    "DISRDR-Trouble_n°2etplus",
]



# 3. DATA LOADING

def load_raw_data(input_dir: str, input_supp: str) -> dict:
    """
    Load all raw source files and return them as a dictionary of DataFrames.

    Parameters
    ----------
    input_dir  : str — path to the R-Link eCRF directory
    input_supp : str — path to the supplementary Excel file (F. Bellivier 2026)

    Returns
    -------
    dict with keys: 'inclusion', 'baseline', 'visits', 'medications',
                    'family_history', 'outcome', 'supplementary'
    """
    data = {}

    data["inclusion"] = pd.read_csv(
        input_dir + "dataset-clinical_mod-inclusion_version-3.tsv",
        sep="\t", na_values=["ND"]
    )
    data["baseline"] = pd.read_csv(
        input_dir + "dataset-clinical_mod-baseline_version-3.tsv",
        sep="\t", na_values=["ND"]
    )
    data["visits"] = pd.read_csv(
        input_dir + "dataset-clinical_mod-visits_form-visit_version-3.tsv",
        sep="\t", na_values=["ND"]
    )
    data["medications"] = pd.read_csv(
        input_dir + "dataset-clinical_mod-baseline_form-preLi_tab-med_version-3.tsv",
        sep="\t"
    )
    data["family_history"] = pd.read_csv(
        input_dir + "dataset-clinical_mod-baseline_form-postLi_tab-famhist_version-3.tsv",
        sep="\t", na_values=["ND"]
    )
    data["outcome"] = pd.read_csv(
        input_dir + "dataset-outcome_version-4.tsv",
        sep="\t"
    )

    # Supplementary file: first row is the real header
    supp = pd.read_excel(input_supp)
    assert supp.shape == (169, 32), f"Unexpected supp shape: {supp.shape}"
    supp.columns = supp.iloc[0]
    supp = supp.drop(supp.index[0])
    data["supplementary"] = supp

    return data


# 4. OUTCOME VARIABLE

def build_outcome(outcome_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the binary lithium response label from the raw outcome file.

    Classification:
      - GR  (Good Responder)     → kept as "GR"
      - PaR (Partial Responder)  → recoded as "No_GR"
      - NR  (Non Responder)      → recoded as "No_GR"
      - UC  (Unclassified)       → excluded

    The final column 'response' is label-encoded: 0 = GR, 1 = No_GR.

    Parameters
    ----------
    outcome_df : raw outcome DataFrame (must contain
                 'Response.Status.at.end.of.follow.up')

    Returns
    -------
    DataFrame with columns ['participant_id', 'response']
    """
    label_col = "Response.Status.at.end.of.follow.up"

    # Visualise raw distribution
    sns.countplot(x=label_col, data=outcome_df)
    plt.title("Lithium response status — raw")
    plt.ylabel("Count")
    plt.xlabel("Status")
    plt.tight_layout()
    plt.show()

    print("Raw distribution:\n", outcome_df[label_col].value_counts())

    # Keep only classified subjects
    classified = outcome_df[outcome_df[label_col].isin(["GR", "PaR", "NR"])].copy()
    classified = classified[["participant_id", label_col]].rename(
        columns={label_col: "response"}
    )

    # Recode to binary
    classified.loc[classified["response"].isin(["PaR", "NR"]), "response"] = "No_GR"
    assert set(classified["response"].unique()) == {"GR", "No_GR"}, \
        "Unexpected values after binary recoding"

    le = LabelEncoder()
    classified["response"] = le.fit_transform(classified["response"])
    # Label encoding result: 0 = GR, 1 = No_GR

    # Visualise binary distribution
    sns.countplot(x="response", data=classified)
    plt.title("Lithium response — binary (0=GR, 1=No_GR)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    return classified


# 5. FEATURE ENGINEERING

def compute_bmq_subscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the four BMQ (Beliefs about Medicines Questionnaire) subscores
    from the 18 individual items and add them as new columns.

    Subscores:
      BMQ_NECESSITY      : items 1–5  (specific necessity)
      BMQ_PREOCCUPATION  : items 6–10 (specific concerns)
      BMQ_DIFFERENTIAL   : necessity minus concerns (adherence predictor)
      BMQ_GENERAL        : items 11–18 (general beliefs)
    """
    bmq_items = {
        "BMQ_NECESSITY":     [f"BMQ{i}_PRELI" for i in range(1, 6)],
        "BMQ_PREOCCUPATION": [f"BMQ{i}_PRELI" for i in range(6, 11)],
        "BMQ_GENERAL":       [f"BMQ{i}_PRELI" for i in range(11, 19)],
    }
    df["BMQ_NECESSITY"]    = df[bmq_items["BMQ_NECESSITY"]].sum(axis=1)
    df["BMQ_PREOCCUPATION"]= df[bmq_items["BMQ_PREOCCUPATION"]].sum(axis=1)
    df["BMQ_DIFFERENTIAL"] = df["BMQ_NECESSITY"] - df["BMQ_PREOCCUPATION"]
    df["BMQ_GENERAL"]      = df[bmq_items["BMQ_GENERAL"]].sum(axis=1)
    return df


def compute_mars_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the MARS (Medication Adherence Rating Scale) total score
    by summing all 10 individual items.
    """
    mars_cols = [col for col in df.columns if col.startswith("MARS")]
    df["MARS_TOTAL"] = df[mars_cols].sum(axis=1)
    return df


def compute_family_history_features(fhist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the family history table into three participant-level features:

      fhist_count   : total number of relatives with psychiatric history
      fhist_ratio_h : proportion of male relatives (SEX_FH == 1)
      fhist_repli   : proportion of relatives with positive lithium response
                      (REPLI_FM == 1)

    Parameters
    ----------
    fhist_df : raw family history DataFrame (one row per relative)

    Returns
    -------
    DataFrame indexed by participant_id with the three new features
    """
    fhist_count   = fhist_df.groupby("participant_id").size()
    fhist_count.name = "fhist_count"

    fhist_ratio_h = fhist_df.groupby("participant_id")["SEX_FH"].apply(
        lambda x: (x == 1.0).mean()
    )
    fhist_ratio_h.name = "fhist_ratio_h"

    fhist_repli   = fhist_df.groupby("participant_id")["REPLI_FM"].apply(
        lambda x: (x == 1.0).mean()
    )
    fhist_repli.name = "fhist_repli"

    return pd.concat([fhist_count, fhist_ratio_h, fhist_repli], axis=1)


# 6. DATA CLEANING

def clean_raw_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific cleaning rules before imputation:

      - FHIST_PLI: recode 9 → 0 (no family history, not "unknown")
      - MOOD_PLI / ANTIPSY_PLI / NEUROL_PLI / ANTIDEP_PLI / BENZOS_PLI:
          recode 9 → NaN  (9 = "not applicable" in eCRF coding)
      - RCY1_PLI / RCY2_PLI:
          recode 2 → NaN  (2 = "not assessed")
      - Any remaining "ND" string → NaN
    """
    df = df.replace("ND", np.nan)

    df["FHIST_PLI"] = df["FHIST_PLI"].replace(9.0, 0.0)

    medication_history_cols = ["MOOD_PLI", "ANTIPSY_PLI", "NEUROL_PLI",
                                "ANTIDEP_PLI", "BENZOS_PLI"]
    df[medication_history_cols] = df[medication_history_cols].replace(9, np.nan)

    rapid_cycling_cols = ["RCY1_PLI", "RCY2_PLI"]
    df[rapid_cycling_cols] = df[rapid_cycling_cols].replace(2, np.nan)

    return df


def drop_high_missingness_columns(df: pd.DataFrame,
                                  threshold: float = 0.40) -> pd.DataFrame:
    """
    Drop any column where the proportion of missing values exceeds `threshold`.

    Default threshold is 40 %, chosen after visual inspection of the
    missingness curve (see plot_missingness_curve).
    After filtering, the maximum per-row missing rate is ~37 % → all rows retained.

    Parameters
    ----------
    df        : input DataFrame
    threshold : float in [0, 1] — columns with NaN rate > threshold are dropped

    Returns
    -------
    Filtered DataFrame
    """
    cols_to_drop = df.columns[df.isnull().mean() > threshold]
    print(f"Dropping {len(cols_to_drop)} columns with >{threshold:.0%} missing values.")
    return df.drop(columns=cols_to_drop)


def plot_missingness_curve(df: pd.DataFrame) -> None:
    """
    Plot the number of features retained as a function of the
    missing-value threshold. Useful for choosing the threshold
    passed to drop_high_missingness_columns().
    """
    thresholds   = np.arange(0.0, 1.0, 0.01)
    n_retained   = [
        (df.isnull().mean() <= t).sum() for t in thresholds
    ]
    plt.figure()
    plt.plot(1 - thresholds, n_retained)
    plt.xlabel("Maximum allowed NA rate (threshold)")
    plt.ylabel("Number of features retained")
    plt.title("Feature retention vs. missing-value threshold")
    plt.grid()
    plt.tight_layout()
    plt.show()




# 8. MAIN PIPELINE

def main():
    # ------------------------------------------------------------------
    # 8.1  Load raw data
    # ------------------------------------------------------------------
    raw = load_raw_data(INPUT_DIR, INPUT_SUPP)

    # ------------------------------------------------------------------
    # 8.2  Select columns from each source file
    # ------------------------------------------------------------------
    inclusion_df = raw["inclusion"][["participant_id", "CENTERNUM", "SEX", "AGE"]]
    assert inclusion_df.shape == (168, 4)

    baseline_df = raw["baseline"][
        BASELINE_VARS_CLINICAL + BASELINE_VARS_QUESTIONNAIRES
    ]
    assert baseline_df.shape == (168, 162)

    # M03 visit: keep only 3-month timepoint and rename columns with _M03 suffix
    m03_df = raw["visits"][raw["visits"]["VISCODE"] == "M3"][VISIT_M03_VARS].copy()
    m03_df = m03_df.rename(
        columns={col: f"{col}_M03" for col in m03_df.columns if col != "participant_id"}
    )

    # Outcome
    response_df = build_outcome(raw["outcome"])

    # ------------------------------------------------------------------
    # 8.3  Feature engineering
    # ------------------------------------------------------------------

    # Merge inclusion + baseline
    data = pd.merge(inclusion_df, baseline_df, on="participant_id", how="inner")
    assert data.shape == (168, 165)

    # BMQ subscoresdata_global
    data = compute_bmq_subscores(data)

    # MARS total
    data = compute_mars_total(data)

    # Family history aggregated features
    fhist_features = compute_family_history_features(raw["family_history"])
    data = data.merge(fhist_features, on="participant_id", how="left")
    data[["fhist_count", "fhist_ratio_h", "fhist_repli"]] = (
        data[["fhist_count", "fhist_ratio_h", "fhist_repli"]].fillna(0)
    )
    assert data.shape == (168, 173)

    # Add M03 visit data
    data = pd.merge(data, m03_df, on="participant_id", how="left")
    assert data.shape == (168, 280)

    # ------------------------------------------------------------------
    # 8.4  Merge supplementary data (F. Bellivier 2026)
    # ------------------------------------------------------------------
    supp_vars = list(set(SUPP_VARS_TO_KEEP_RAW) - set(SUPP_VARS_TO_DROP))
    supp_df   = raw["supplementary"]

    # Convert numeric supplementary columns before merge
    for col in supp_df.columns:
        supp_df[col] = pd.to_numeric(supp_df[col], errors="ignore")

    # Sanity check: AGE and SEX must be consistent between the two sources
    fd_idx   = data.set_index("participant_id").sort_index()
    supp_idx = supp_df.set_index("participant_id").sort_index()
    fd_idx, supp_idx = fd_idx.align(supp_idx)
    inconsistent = fd_idx["AGE"].ne(supp_idx["AGE"]) | fd_idx["SEX"].ne(supp_idx["SEX"])
    assert not inconsistent.any(), \
        f"AGE/SEX mismatch between sources for: {fd_idx.index[inconsistent].tolist()}"

    data = data.merge(supp_df[supp_vars], on="participant_id", how="inner")
    assert data.shape == (168, 303)

    # ------------------------------------------------------------------
    # 8.5  Deduplicate redundant columns
    # ------------------------------------------------------------------
    # The supplementary file contains derived versions of columns already in
    # the baseline. We keep the supplementary version (more processed) and
    # drop / rename accordingly.

    REDUNDANT_BASELINE_COLS = {
        # baseline column          → supplementary equivalent
        "PHCMBY_PLI":              "PHCMBY_PLI-ComorbiditeSomatique",
        "QIDSTSC_PRELI":           "QIDS_TotalScore_M00",
        "BRMSTSC_PRELI":           "BRMS_TotalScore_M00",
        "NBH2_PLI":                "NbHospitalizationsLifetime",
        "WHOA1A_PLI":              "SmokingStatus-WHOA1A_PLI",
    }

    # Drop the supplementary versions first (will be replaced by rename)
    data = data.drop(columns=list(REDUNDANT_BASELINE_COLS.values()), errors="ignore")
    # Rename baseline columns to the agreed canonical names
    data = data.rename(columns=REDUNDANT_BASELINE_COLS)
    data = data.rename(
        columns={"PSYHLTH_PLI-ComorbiditePsychiatrique": "Psychiatric_Comorbidity"}
    )

    # ------------------------------------------------------------------
    # 8.6  Merge outcome and filter to classified subjects
    # ------------------------------------------------------------------
    data = pd.merge(data, response_df[["participant_id", "response"]],
                    on="participant_id", how="inner")
    # 30 subjects have no outcome value → they are excluded here
    assert data.shape == (138, 299), f"Unexpected shape after outcome merge: {data.shape}"

    # ------------------------------------------------------------------
    # 8.7  Clean raw values
    # ------------------------------------------------------------------
    data = clean_raw_values(data)

    # ------------------------------------------------------------------
    # 8.8  Remove high-missingness columns
    # ------------------------------------------------------------------
    plot_missingness_curve(data)
    data = drop_high_missingness_columns(data, threshold=0.40)
    assert data.shape == (138, 210), f"Unexpected shape after missingness filter: {data.shape}"

    # ------------------------------------------------------------------
    # 8.9 Create datasets for analyses
    # ------------------------------------------------------------------

    m03_columns = [
        c for c in data.columns
        if c.endswith("_M03") or c.startswith("Delta")
    ]

    # M00 only
    data_m00 = data.drop(columns=m03_columns).copy()

    # M03 only
    m03_keep = ["participant_id", "response"] + m03_columns
    data_m03 = data[m03_keep].copy()

    # Global dataset
    data_global = data.copy()

    # ------------------------------------------------------------------
    # 8.10 Save datasets
    # ------------------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_m00.to_csv(
        OUTPUT_DIR + "Rlink_version3_type_Clinic_timepoint_M00_v20260602.csv",
        index=False
    )

    data_m03.to_csv(
        OUTPUT_DIR + "Rlink_version3_type_Clinic_timepoint_M03_v20260602.csv",
        index=False
    )

    data_global.to_csv(
        OUTPUT_DIR + "Rlink_version3_type_Clinic_timepoint_M00_M03_v20260602.csv",
        index=False
    )

    print(f"M00 dataset saved: {data_m00.shape}")
    print(f"M03 dataset saved: {data_m03.shape}")
    print(f"M00+M03 dataset saved: {data_global.shape}")



# ENTRY POINT

if __name__ == "__main__":
    main()







