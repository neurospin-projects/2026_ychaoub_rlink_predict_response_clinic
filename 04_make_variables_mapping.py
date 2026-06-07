import pandas as pd

OUTPUT_DIR = "./data/"

# ─────────────────────────────────────────────
# Source data
# ─────────────────────────────────────────────

COLUMN_LABELS = {
    "MOODYN_PRELI":   "Current mood episode",
    "TYPEP_PRELI":    "Type of episode",
    "PS_PRELI":       "With psychotic symptoms",
    "MIX_PRELI":      "With mixed characteristics",
    "HOSP_PRELI":     "Currently hospitalized",
    "WEIGHT_PRELI":   "Weight (kg)",
    "HEIGHT_PRELI":   "Height (cm)",
    "SBP_PRELI":      "Systolic BP (mmHg)",
    "DBP_PRELI":      "Diastolic BP (mmHg)",
    "RELSTAT_PRELI":  "Relationship status",
    "ETHNICITY_PRELI":"Ethnicity",
    "CORG_PRELI":     "Country of origin",
    "LIVSIT_PRELI":   "Living situation",
    "RESIDENCE_PRELI":"Place of residence",
    "SCHOOL_PRELI":   "Highest qualification",
    "JOB_PRELI":      "Employment status",
    "EVNT_PRELI":     "Recent stressful event (<12 months)",
    "CURRMED_PRELI":  "Current medications",
    "NA_PRELI":       "Sodium (mmol/L)",
    "K_PRELI":        "Potassium (mmol/L)",
    "CL_PRELI":       "Chloride (mmol/L)",
    "CA_PRELI":       "Calcium (mmol/L)",
    "PROTEINS_PRELI": "Proteins (g/L)",
    "UREA_PRELI":     "Urea (mmol/L)",
    "CREAT_PRELI":    "Creatinine (µmol/L)",
    "EGFR_PRELI":     "eGFR — CKD-EPI (mL/min/1.73m²)",
    "MDRD_PRELI":     "eGFR — MDRD (mL/min/1.73m²)",
    "CKDEPI_PRELI":   "eGFR — CKD-EPI alternative",
    "TSH_PRELI":      "TSH (mU/L)",
    "HB_PRELI":       "Hemoglobin (g/dL)",
    "PLT_PRELI":      "Platelets (×10⁹/L)",
    "QIDSTSC_PRELI":  "QIDS total score (M00)",
    "BRMSTSC_PRELI":  "BRMS total score (M00)",
    "BPRSTSC_PRELI":  "BPRS total score (M00)",
    "SEX":            "Male sex",
    "FHIST_PLI":      "Family history of BD",
    "fhist_count":    "Family history — number of relatives",
    "fhist_ratio_h":  "Family history — proportion of male relatives",
    "fhist_repli":    "Family history — proportion with positive Li response",
    "MOOD_PLI":       "Mood stabilizers (2 years before inclusion)",
    "ANTIPSY_PLI":    "Antipsychotics (2 years before inclusion)",
    "NEUROL_PLI":     "Neuroleptics (2 years before inclusion)",
    "ANTIDEP_PLI":    "Antidepressants (2 years before inclusion)",
    "BENZOS_PLI":     "Benzodiazepines (2 years before inclusion)",
    "RCY1_PLI":       "Rapid cycling (2 years before inclusion)",
    "MDE1_PLI":       "Depressive episode total duration (weeks) — 2y",
    "MDEH1_PLI":      "MDE hospitalized duration (weeks) — 2y",
    "MDEPS1_PLI":     "MDE with psychotic symptoms duration (weeks) — 2y",
    "MDEMC1_PLI":     "MDE mixed duration (weeks) — 2y",
    "PHCMBY_PLI":     "Physical comorbidity",
    "PSYHLTH_PLI-ComorbiditePsychiatrique": "Psychiatric comorbidity",
    "response":       "Lithium response (binary: GR vs No_GR)",
}

M00_Variables = {
    "Current mood disorder": [
        "Current mood episode", "Type of episode", "With mixed characteristics", "Currently hospitalized"
    ],
    "Clinical": [
        "Weight (kg)", "Height (cm)", "WAIST_PRELI", "BMI_M00",
        "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "Relationship status", "Ethnicity",
        "Country of origin", "Living situation", "Place of residence", "Highest qualification",
        "Employment status", "Recent stressful event (<12 months)", "Current medications"
    ],
    "Biological": [
        "Sodium (mmol/L)", "Potassium (mmol/L)", "Calcium (mmol/L)", "Urea (mmol/L)",
        "Creatinine (µmol/L)", "eGFR — CKD-EPI (mL/min/1.73m²)", "eGFR — MDRD (mL/min/1.73m²)",
        "eGFR — CKD-EPI alternative", "TSH (mU/L)"
    ],
    "Family history": [
        "Family history of BD", "Family history — number of relatives",
        "Family history — proportion of male relatives",
        "Family history — proportion with positive Li response",
        "CENTERNUM", "Male sex", "AGE"
    ],
    "Past medication < 2y": [
        "Mood stabilizers (2 years before inclusion)", "Antipsychotics (2 years before inclusion)",
        "Neuroleptics (2 years before inclusion)", "Antidepressants (2 years before inclusion)",
        "Benzodiazepines (2 years before inclusion)"
    ],
    "Any physical comorbidity": [
        "PHCMBY_PLI-ComorbiditeSomatique", "Psychiatric_Comorbidity"
    ],
    "Characteristics of BD < 2y": [
        "Rapid cycling (2 years before inclusion)", "RCY2_PLI", "MDE2_PLI", "MDEH2_PLI",
        "MDEPS2_PLI", "MDEMC2_PLI", "MDETD2_PLI", "AGEMDE2_PLI",
        "Depressive episode total duration (weeks) — 2y", "MDE hospitalized duration (weeks) — 2y",
        "MDE with psychotic symptoms duration (weeks) — 2y", "MDE mixed duration (weeks) — 2y",
        "HYPOE2_PLI", "HYPOEH2_PLI", "HYPOEMC2_PLI", "HYPOETD2_PLI", "AGEHYPOE2_PLI",
        "MANE2_PLI", "MANEH2_PLI", "MANEPS2_PLI", "MANEMC2_PLI", "MANETD2_PLI", "AGEMANE2_PLI",
        "NBH1_PLI", "TDH1_PLI", "OUTW1_PLI", "NBS1_PLI", "AD1_PLI", "SUD1_PLI", "MC1_PLI"
    ],
    "Characteristics of BD during lifetime": [
        "AgeAtOnset", "DurationIllness", "NumberPreviousEpisodes", "DensityEpisodes",
        "DensityHospit", "NbHospitalizationsLifetime", "SuicideAttempts(Yes/No)",
        "MDETD1_PLI", "HYPOE1_PLI", "HYPOEH1_PLI", "HYPOEMC1_PLI", "HYPOETD1_PLI",
        "MANE1_PLI", "MANEH1_PLI", "MANEPS1_PLI", "MANEMC1_PLI", "MANETD1_PLI",
        "TDH2_PLI", "AGESTBH2_PLI", "OUTW2_PLI", "NBS2_PLI", "AD2_PLI", "SUD2_PLI", "MC2_PLI"
    ],
    "QIDS / BRMS / SSRS / BPRS (psychiatric scales)": [
        "QIDS_TotalScore_M00", "BRMS_TotalScore_M00", "SSRS1_PRELI", "SSRS2_PRELI",
        "SSRS6_PRELI", "BPRS total score (M00)"
    ],
    "MARS (medication adherence)": [
        "MARS1_PRELI", "MARS2_PRELI", "MARS3_PRELI", "MARS4_PRELI", "MARS5_PRELI",
        "MARS6_PRELI", "MARS7_PRELI", "MARS8_PRELI", "MARS9_PRELI", "MARS10_PRELI", "MARS_TOTAL"
    ],
    "TRQ (medication adherence)": [
        "TRQ_PRELI", "TRQ1_PRELI", "TRQ2_PRELI", "TRQ4_PRELI",
        "TRQ5_PRELI", "TRQ6_PRELI", "TRQ7_PRELI"
    ],
    "BMQ (belief in medication)": [
        "BMQ1_PRELI", "BMQ2_PRELI", "BMQ3_PRELI", "BMQ4_PRELI", "BMQ5_PRELI",
        "BMQ6_PRELI", "BMQ7_PRELI", "BMQ8_PRELI", "BMQ9_PRELI", "BMQ10_PRELI",
        "BMQ11_PRELI", "BMQ12_PRELI", "BMQ13_PRELI", "BMQ14_PRELI", "BMQ15_PRELI",
        "BMQ16_PRELI", "BMQ17_PRELI", "BMQ18_PRELI",
        "BMQ_NECESSITY", "BMQ_PREOCCUPATION", "BMQ_DIFFERENTIAL", "BMQ_GENERAL"
    ],
    "WHOA (substance use)": [
        "SmokingStatus-WHOA1A_PLI", "WHOA1B_PLI", "WHOA1C_PLI", "WHOA1D_PLI",
        "WHOA1E_PLI", "WHOA1F_PLI", "WHOA1G_PLI", "WHOA1H_PLI"
    ],
}

M03_variables = {
    "Clinical — M03": [
        "WEIGHT_M03", "WAIST_M03", "BMI_M03", "SBP_M03", "DBP_M03", "CURRMED_M03"
    ],
    "Biological — M03": ["CREAT_M03", "MDRD_M03", "CKDEPI_M03"],
    "QIDS / BRMS / SSRS / BPRS — M03": [
        "QIDS_W1_M03", "BRMS_W1_M03", "SSRS1_M03", "SSRS2_M03", "SSRS6_M03", "BPRSTSC_M03"
    ],
    "MARS (medication adherence) — M03": [
        "MARS1V_M03", "MARS2V_M03", "MARS3V_M03", "MARS4V_M03", "MARS5V_M03",
        "MARS6V_M03", "MARS7V_M03", "MARS8V_M03", "MARS9V_M03", "MARS10V_M03"
    ],
    "TRQ (medication adherence) — M03": [
        "TRQ_M03", "TRQ1_M03", "TRQ2_M03", "TRQ3_M03",
        "TRQ4_M03", "TRQ5_M03", "TRQ6_M03", "TRQ7_M03"
    ],
    "BMQ (belief in medication) — M03": [
        "BMQ1_M03", "BMQ2_M03", "BMQ3_M03", "BMQ4_M03", "BMQ5_M03", "BMQ6_M03",
        "BMQ7_M03", "BMQ8_M03", "BMQ9_M03", "BMQ10_M03", "BMQ11_M03", "BMQ12_M03",
        "BMQ13_M03", "BMQ14_M03", "BMQ15_M03", "BMQ16_M03", "BMQ17_M03", "BMQ18_M03"
    ],
    "Delta — medication changes": ["DeltaATD", "DeltaAPA", "DeltaBZD", "DeltaNLP", "DeltaAC"],
    "Delta — anthropometric":     ["DeltaBMI", "Delta_BMI_impute"],
}

Type_of_M00_variables = {
    "binary": [
        "Male sex", "Current mood episode", "With mixed characteristics", "Currently hospitalized",
        "Recent stressful event (<12 months)", "Current medications", "Family history of BD",
        "Mood stabilizers (2 years before inclusion)", "Antipsychotics (2 years before inclusion)",
        "Neuroleptics (2 years before inclusion)", "Antidepressants (2 years before inclusion)",
        "Benzodiazepines (2 years before inclusion)", "PHCMBY_PLI-ComorbiditeSomatique",
        "Rapid cycling (2 years before inclusion)", "HYPOEH1_PLI", "RCY2_PLI",
        "SSRS1_PRELI", "SSRS2_PRELI", "SSRS6_PRELI",
        "MARS1_PRELI", "MARS2_PRELI", "MARS3_PRELI", "MARS4_PRELI", "MARS5_PRELI",
        "MARS6_PRELI", "MARS7_PRELI", "MARS8_PRELI", "MARS9_PRELI", "MARS10_PRELI",
        "TRQ_PRELI", "TRQ1_PRELI", "TRQ2_PRELI",
        "SmokingStatus-WHOA1A_PLI", "WHOA1B_PLI", "WHOA1C_PLI", "WHOA1D_PLI",
        "WHOA1E_PLI", "WHOA1F_PLI", "WHOA1G_PLI", "WHOA1H_PLI",
        "Psychiatric_Comorbidity", "SuicideAttempts(Yes/No)"
    ],
    "quantitative": [
        "CENTERNUM", "AGE", "Weight (kg)", "Height (cm)", "WAIST_PRELI",
        "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "Sodium (mmol/L)", "Potassium (mmol/L)",
        "Calcium (mmol/L)", "Urea (mmol/L)", "Creatinine (µmol/L)",
        "eGFR — CKD-EPI (mL/min/1.73m²)", "eGFR — MDRD (mL/min/1.73m²)",
        "eGFR — CKD-EPI alternative", "TSH (mU/L)",
        "Depressive episode total duration (weeks) — 2y",
        "MDETD1_PLI", "HYPOE1_PLI", "HYPOETD1_PLI", "MANETD1_PLI", "TDH1_PLI", "OUTW1_PLI",
        "MDE2_PLI", "MDETD2_PLI", "AGEMDE2_PLI", "HYPOE2_PLI", "HYPOETD2_PLI", "AGEHYPOE2_PLI",
        "MANE2_PLI", "MANETD2_PLI", "AGEMANE2_PLI", "NbHospitalizationsLifetime",
        "TDH2_PLI", "AGESTBH2_PLI", "OUTW2_PLI", "SUD2_PLI",
        "QIDS_TotalScore_M00", "BRMS_TotalScore_M00", "BPRS total score (M00)",
        "TRQ5_PRELI", "TRQ7_PRELI",
        "BMQ_NECESSITY", "BMQ_PREOCCUPATION", "BMQ_DIFFERENTIAL", "BMQ_GENERAL", "MARS_TOTAL",
        "Family history — number of relatives", "Family history — proportion of male relatives",
        "Family history — proportion with positive Li response",
        "BMI_M00", "DurationIllness", "AgeAtOnset", "DensityHospit", "DensityEpisodes",
        "NumberPreviousEpisodes"
    ],
    "multicategory": [
        "Type of episode", "Relationship status", "Ethnicity", "Country of origin",
        "Living situation", "Place of residence", "Highest qualification", "Employment status",
        "MDE hospitalized duration (weeks) — 2y", "MDE with psychotic symptoms duration (weeks) — 2y",
        "MDE mixed duration (weeks) — 2y", "HYPOEMC1_PLI",
        "MANE1_PLI", "MANEH1_PLI", "MANEPS1_PLI", "MANEMC1_PLI", "NBH1_PLI", "NBS1_PLI",
        "AD1_PLI", "SUD1_PLI", "MC1_PLI", "MDEH2_PLI", "MDEPS2_PLI", "MDEMC2_PLI",
        "HYPOEH2_PLI", "HYPOEMC2_PLI", "MANEH2_PLI", "MANEPS2_PLI", "MANEMC2_PLI",
        "NBS2_PLI", "AD2_PLI", "MC2_PLI",
        "TRQ4_PRELI", "TRQ6_PRELI",
        "BMQ1_PRELI", "BMQ2_PRELI", "BMQ3_PRELI", "BMQ4_PRELI", "BMQ5_PRELI",
        "BMQ6_PRELI", "BMQ7_PRELI", "BMQ8_PRELI", "BMQ9_PRELI", "BMQ10_PRELI",
        "BMQ11_PRELI", "BMQ12_PRELI", "BMQ13_PRELI", "BMQ14_PRELI", "BMQ15_PRELI",
        "BMQ16_PRELI", "BMQ17_PRELI", "BMQ18_PRELI"
    ]
}

Type_of_M03_variables = {
    "content":      ["CURRMED_M03"],
    "binary": [
        "SSRS1_M03", "SSRS2_M03", "SSRS6_M03",
        "MARS1V_M03", "MARS2V_M03", "MARS3V_M03", "MARS4V_M03", "MARS5V_M03",
        "MARS6V_M03", "MARS7V_M03", "MARS8V_M03", "MARS9V_M03", "MARS10V_M03",
        "TRQ_M03", "TRQ1_M03", "TRQ2_M03", "DeltaNLP"
    ],
    "quantitative": [
        "WEIGHT_M03", "WAIST_M03", "SBP_M03", "DBP_M03", "CREAT_M03", "MDRD_M03", "CKDEPI_M03",
        "BPRSTSC_M03", "TRQ3_M03", "BMI_M03", "DeltaBMI", "Delta_BMI_impute",
        "BRMS_W1_M03", "QIDS_W1_M03"
    ],
    "multicategory": [
        "TRQ4_M03", "TRQ5_M03", "TRQ6_M03", "TRQ7_M03",
        "BMQ1_M03", "BMQ2_M03", "BMQ3_M03", "BMQ4_M03", "BMQ5_M03", "BMQ6_M03",
        "BMQ7_M03", "BMQ8_M03", "BMQ9_M03", "BMQ10_M03", "BMQ11_M03", "BMQ12_M03",
        "BMQ13_M03", "BMQ14_M03", "BMQ15_M03", "BMQ16_M03", "BMQ17_M03", "BMQ18_M03",
        "DeltaATD", "DeltaAPA", "DeltaBZD", "DeltaAC"
    ]
}

# ─────────────────────────────────────────────
# Build lookup tables
# ─────────────────────────────────────────────

# code → label  and  label → code
label_to_code = {label: code for code, label in COLUMN_LABELS.items()}

# variable → type  (works on both codes and labels)
def invert_type_dict(type_dict):
    return {var: vtype for vtype, vars_list in type_dict.items() for var in vars_list}

# ─────────────────────────────────────────────
# Build a DataFrame from a category dict
# Entries can be original codes OR labels
# ─────────────────────────────────────────────
def build_dataframe(category_dict, type_dict):
    type_lookup = invert_type_dict(type_dict)
    rows = []
    seen = set()

    for category, entries in category_dict.items():
        for entry in entries:
            # Resolve code and label
            if entry in COLUMN_LABELS:          # entry is a code
                code, label = entry, COLUMN_LABELS[entry]
            elif entry in label_to_code:        # entry is a label
                code, label = label_to_code[entry], entry
            else:                               # raw variable, no rename
                code, label = entry, ""

            if code in seen:
                continue
            seen.add(code)

            vtype = type_lookup.get(code) or type_lookup.get(label) or ""
            rows.append({"variable": code, "label": label, "category": category, "type": vtype})

    return pd.DataFrame(rows, columns=["variable", "label", "category", "type"])

# ─────────────────────────────────────────────
# Generate the two CSV files
# ─────────────────────────────────────────────

df_m00 = build_dataframe(M00_Variables, Type_of_M00_variables)
df_m00.to_csv(OUTPUT_DIR + "Rlink_Clinical_variables_M00_mapping.csv", index=False, encoding="utf-8")
print(f"[OK] variables_M00.csv  ({len(df_m00)} variables)")

df_m03 = build_dataframe(M03_variables, Type_of_M03_variables)
df_m03.to_csv(OUTPUT_DIR + "Rlink_Clinical_variables_M03_mapping.csv", index=False, encoding="utf-8")
print(f"[OK] variables_M03_delta.csv  ({len(df_m03)} variables)")