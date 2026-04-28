# Read files
import pandas as pd
import os


# %% Set path to inout data and output results according to your local configuration

INPUT = "/neurospin/rlink/PUBLICATION/rlink-ecrf/"
INPUT_SUPP = "/neurospin/signatures/2026_ychaoub_rlink_predict_response_clinic/data/study-rlink_dataset-clinical_type-summary_version-20260220.xlsx"
#INPUT_SUPP = "./data/study-rlink_dataset-clinical_type-summary_version-20260220.xlsx"
OUTPUT = './data/'



# %%  Read and manipulate data
# ============================

# Supplemantary file provided by F. Bellivier in 2026
supp_df = pd.read_excel(INPUT_SUPP)
assert supp_df.shape == (169, 32)
supp_df.columns = supp_df.iloc[0] #set the first row as the header
supp_df = supp_df.drop(supp_df.index[0]) #remove the first row from the data



# Inclusion file
inclusion_df = pd.read_csv(INPUT + "dataset-clinical_mod-inclusion_version-3.tsv", sep='\t', na_values=['ND'])




# Baseline
baseline_df = pd.read_csv(INPUT + "dataset-clinical_mod-baseline_version-3.tsv", sep='\t', na_values=['ND'])

# %%  Select columns
# ==================


vars =["participant_id", "CENTERNUM", "SEX", "AGE"] # SEX: 1 = Male, 2 = Female , 3 = Other
inclusion_df = inclusion_df[vars]
assert inclusion_df.shape == (168, 4)



# %% 1. Selection from R-Link datasets
# ====================================
#
# Use inclusion_df & baseline_df to replicate selection from:
# /neurospin/signatures/temp_thibault/2024_rlink_predict_response/Clinical/RLINK-1-Clinical.ipynb

# %% 2. merge with supplementary data supp_df
# ===========================================


# %%  EXEMPLE: Read and manipulate data
# =====================================

# 1.1 Variables from file "dataset-clinical_mod-inclusion_version-2.tsv"



# 1.2 Variables from file "dataset-clinical_mod-baseline_version-2.tsv"
#Selected
vars = ["participant_id", "WHOA1A_PLI", "HEIGHT_PRELI", "WEIGHT_PRELI",
        "MARS1_PRELI", "MARS2_PRELI", "MARS3_PRELI", "MARS4_PRELI", "MARS5_PRELI",
        "MARS6_PRELI", "MARS7_PRELI", "MARS8_PRELI", "MARS9_PRELI", "MARS10_PRELI",
        "QIDSTSC_PRELI", "BRMSTSC_PRELI"]
baseline_df = pd.read_csv(INPUT + "dataset-clinical_mod-baseline_version-3.tsv", sep='\t', na_values=['ND'])
baseline_df = baseline_df[vars]
baseline_df["BMI"] = baseline_df["WEIGHT_PRELI"] / (baseline_df["HEIGHT_PRELI"] ** 2) * (100 ** 2)
assert df2.shape == (168, 17)

# # 1.3 Variables from file "dataset-clinical_mod-visits_form-visit_version-2.tsv
# vars = ["participant_id", "FORM_F_VISIT",
#         "WHOA2A", "PLIMRI", "ERYLIMRI", "PLI", "ERYLI", "PLI2",
#         "MARS1V", "MARS2V", "MARS3V", "MARS4V", "MARS5V", "MARS6V", "MARS7V", "MARS8V",
#         "MARS9V", "MARS10V"]
# m3_df = pd.read_csv(INPUT + "dataset-clinical_mod-visits_form-visit_version-3.tsv", sep='\t', na_values=['ND'])
# m3_df = m3_df[vars]
# m3_df.shape == (2449, 18)
# # Select visits at month 3 => FORM_F_VISIT == 'F_VISIT_1'
# m3_df = m3_df[m3_df.FORM_F_VISIT == 'F_VISIT_1']
# # drop FORM_F_VISIT column and suffix variable with "_M03"
# m3_df = m3_df.drop(['FORM_F_VISIT'], axis=1)
# m3_df = m3_df.rename(columns={"MARS1V":"MARS1_M03", "MARS2V":"MARS2_M03", "MARS3V":"MARS3_M03",
#             "MARS4V":"MARS4_M03", "MARS5V":"MARS5_M03", "MARS6V":"MARS6_M03",
#             "MARS7V":"MARS7_M03", "MARS8V":"MARS8_M03", "MARS9V":"MARS9_M03",
#             "MARS10V":"MARS10_M03"})
# assert m3_df.shape == (141, 17)

# 1.4 Variables from file "dataset-clinical_version-2_outcome.tsv"

vars = ["participant_id", "Response.Status.at.end.of.follow.up"]
outcome = pd.read_csv(INPUT + "dataset-outcome_version-4.tsv", sep='\t', na_values=['ND'])
outcome = outcome[vars]
#df4.participant_id = ["sub-"+ str(id) for id in df4.participant_id]
assert outcome.shape == (168, 2)


# %% 2. Merge tables using "participant_id" (used by default)


table = pd.merge(pd.merge(pd.merge(df1, df2), df3), df4)
table.shape == (141, 36)

# %% 3. Save to excel file
#

table.to_excel(OUTPUT + "data_demoSmokLiMarsResponse_python.xlsx", index=False)
table.to_csv(OUTPUT + "rlink-predict-response-clinic_v-20260416", index=False)
