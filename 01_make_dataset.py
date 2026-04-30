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




# %%  Select columns
# ==================


# %% 1. Selection from R-Link datasets
# ====================================

## Loading data from files in ecrf repo

meds = pd.read_csv(INPUT + "dataset-clinical_mod-baseline_form-preLi_tab-med_version-3.tsv", delimiter = "\t") # la variable UNIT est juste une unité  (page 14)
fhist = pd.read_csv(INPUT + "dataset-clinical_mod-baseline_form-postLi_tab-famhist_version-3.tsv", delimiter = "\t", na_values= "ND")

# Inclusion file
inclusion_df = pd.read_csv(INPUT + "dataset-clinical_mod-inclusion_version-3.tsv", sep='\t', na_values=['ND'])


# Baseline
baseline_df = pd.read_csv(INPUT + "dataset-clinical_mod-baseline_version-3.tsv", sep='\t', na_values=['ND'])




# %%  EXEMPLE: Read and manipulate data
# =====================================

# 1.1 Variables from file "dataset-clinical_mod-inclusion_version-3.tsv"

vars =["participant_id", "CENTERNUM", "SEX", "AGE"] # SEX: 1 = Male, 2 = Female , 3 = Other
inclusion_df = inclusion_df[vars]
assert inclusion_df.shape == (168, 4)

# 1.2 Variables from file "dataset-clinical_mod-baseline_version-3.tsv"
#Selected


VARS_TO_INCLUDE_PRELI = [
    "participant_id",
    #Thymic state at inclusion
    "MOODYN_PRELI", 
    "TYPEP_PRELI",
    "PS_PRELI",
    "MIX_PRELI",
    "DATSTEPD_PRELI",
    "HOSP_PRELI", 
   # "RSNINILI_PRELI", #NOT IN BASE
   # "OTHRSNINILI_PRELI", #NOT IN BASE
    #Physical preliminary exam
    "WEIGHT_PRELI",
    "HEIGHT_PRELI",
    "WAIST_PRELI",
    "SBP_PRELI",
    "DBP_PRELI",
    # Relationship status, school, job, recent traumatic event
    "RELSTAT_PRELI",
    "ETHNICITY_PRELI",
    "CORG_PRELI",
    #"COUNTRY_PRELI", #NOT IN BASE
    "LIVSIT_PRELI",
    "RESIDENCE_PRELI",
    "SCHOOL_PRELI",
    "JOB_PRELI",
    "EVNT_PRELI",
    # Medication history
    "CURRMED_PRELI",
    # Biological 
    "DATBIO_PRELI",
    "NA_PRELI",
    "K_PRELI",
    "CL_PRELI",
    "CA_PRELI",
    "PROTEINS_PRELI",
    "UREA_PRELI",
    "CREAT_PRELI",
    "EGFR_PRELI",
    "MDRD_PRELI",
    "CKDEPI_PRELI",
    "TSH_PRELI",
    "T3_PRELI",
    "T4_PRELI",
    "GLY_PRELI",
    "TGC_PRELI",
    "HDL_PRELI",
    "LDL_PRELI",
    "BHCG_PRELI",
    "WBC_PRELI",
    "HB_PRELI",
    "HT_PRELI",
    "PLT_PRELI",
    "NP_PRELI",
    "EOS_PRELI",
    "LYMPH_PRELI",
    "MONO_PRELI",
    "FHIST_PLI", "MOOD_PLI", "ANTIPSY_PLI", "NEUROL_PLI", "ANTIDEP_PLI", "BENZOS_PLI", "PHCMBY_PLI",
    "RCY1_PLI", "MDE1_PLI", "MDEH1_PLI", "MDEPS1_PLI", "MDEMC1_PLI", "MDETD1_PLI", 'HYPOE1_PLI', 'HYPOEH1_PLI', 'HYPOEMC1_PLI', 'HYPOETD1_PLI',
    'MANE1_PLI', 'MANEH1_PLI', 'MANEPS1_PLI', 'MANEMC1_PLI', 'MANETD1_PLI', 
    #"PATSEQ1_PLI", 
    "NBH1_PLI", 
    #"PATSEQM1_PLI", 
    "TDH1_PLI", "OUTW1_PLI", "NBS1_PLI", 
    "AD1_PLI", "SUD1_PLI", "MC1_PLI", "RCY2_PLI", "AGESTBD2_PLI", "BDNOW2_PLI", "AGENDBD2_PLI", 
    'MDE2_PLI', 'MDEH2_PLI', 'MDEPS2_PLI', 'MDEMC2_PLI', 'MDETD2_PLI', 'AGEMDE2_PLI', 'HYPOE2_PLI', 'HYPOEH2_PLI', 'HYPOEMC2_PLI', 
    'HYPOETD2_PLI', 'AGEHYPOE2_PLI', 'MANE2_PLI', 'MANEH2_PLI', 'MANEPS2_PLI', 'MANEMC2_PLI', 'MANETD2_PLI', 'AGEMANE2_PLI', 'PATSEQM2_PLI', 'NBH2_PLI', 'TDH2_PLI', 
    'AGESTBH2_PLI', 'OUTW2_PLI', 'NBS2_PLI', 'AGES2_PLI', 'AD2_PLI', 'SUD2_PLI', 'MC2_PLI',
]

VARS_TO_INCLUDE_QUESTIONNAIRES = [
    #DEPRESSION - QIDS 
    "QIDSTSC_PRELI", #total score
    "BRMSTSC_PRELI", #total score
    #suicide risk
    "SSRS1_PRELI", "SSRS2_PRELI", "SSRS3_PRELI", "SSRS4_PRELI", "SSRS5_PRELI", "SSRS6_PRELI", "SSRS6Y_PRELI",
    #BPRS - Psychiatric symptom severity
    "BPRSTSC_PRELI",
    #Medication adherence: MARS, TRQ, BMQ
    "MARS1_PRELI", "MARS2_PRELI", "MARS3_PRELI", "MARS4_PRELI", "MARS5_PRELI", "MARS6_PRELI", "MARS7_PRELI", "MARS8_PRELI", "MARS9_PRELI", "MARS10_PRELI",
    "TRQ_PRELI", "TRQ1_PRELI", "TRQ2_PRELI", 
    #"TRQ3_PRELI", 
    "TRQ4_PRELI", "TRQ5_PRELI", "TRQ6_PRELI", "TRQ7_PRELI", "TRQ1B_PRELI", "TRQ2B_PRELI",
    "BMQ1_PRELI", "BMQ2_PRELI", "BMQ3_PRELI", "BMQ4_PRELI", "BMQ5_PRELI", "BMQ6_PRELI", "BMQ7_PRELI", "BMQ8_PRELI", "BMQ9_PRELI", "BMQ10_PRELI", "BMQ11_PRELI", 
    "BMQ12_PRELI", "BMQ13_PRELI", "BMQ14_PRELI", "BMQ15_PRELI", "BMQ16_PRELI", "BMQ17_PRELI", "BMQ18_PRELI",
    'WHOA1A_PLI', 'WHOA1B_PLI', 'WHOA1C_PLI', 'WHOA1D_PLI', 'WHOA1E_PLI', 'WHOA1F_PLI', 'WHOA1G_PLI', 'WHOA1H_PLI'
]

baseline_df = pd.read_csv(INPUT + "dataset-clinical_mod-baseline_version-3.tsv", sep='\t', na_values=['ND'])
baseline_df = baseline_df[VARS_TO_INCLUDE_PRELI + VARS_TO_INCLUDE_QUESTIONNAIRES ]
baseline_df["BMI"] = baseline_df["WEIGHT_PRELI"] / (baseline_df["HEIGHT_PRELI"] ** 2) * (100 ** 2)
assert baseline_df.shape == (168, 164)




## All other relevant variables from base dataframe

# 1.4 Variables from file "dataset-clinical_version-2_outcome.tsv"

vars = ["participant_id", "Response.Status.at.end.of.follow.up"]
outcome = pd.read_csv(INPUT + "dataset-outcome_version-4.tsv", sep='\t', na_values=['ND'])
final_response = outcome[vars]
#df4.participant_id = ["sub-"+ str(id) for id in df4.participant_id]
assert final_response.shape == (168, 2)


###Response variables 

RESPONSEDIR = "/neurospin/rlink/workspace/code"
#final_response = pd.read_csv(os.path.join(RESPONSEDIR, "dataset-clinical_version-2_outcome.csv"), sep = ';')
print('Shape of the dataset: ', final_response.shape)
display(final_response["Response.Status.at.end.of.follow.up"].value_counts())
display(final_response['Status.ITT'].value_counts())
display(final_response['Status.per.protocol'].value_counts())
print("What the UCs in ITT have become after reclassification")
display(final_response.loc[final_response['Status.ITT'] == 'UC', 'Response.Status.at.end.of.follow.up'].value_counts())

response = final_response.rename(columns = {
    "SUBJECT_REF": "participant_id",
    "Response.Status.at.end.of.follow.up": "response"
})

response.loc[response["response"] == "UC", "response"] = np.NaN

## response_2 is a binary variable with GR and others (PR, NR, UC pooled)
response["response_2"] = np.nan
response.loc[((response["response"] == "PaR") | (response["response"] == "NR") | (response["response"] == "UC") ), "response_2"] = "No_GR"
response.loc[((response["response"] == "GR")), "response_2"] = "GR"

response["response_3"] = np.nan
response.loc[((response["response"] == "PaR") | (response["response"] == "GR") | (response["response"] == "UC") ), "response_3"] = "Others"
response.loc[((response["response"] == "NR")), "response_3"] = "NR"
# Plotting response_2 variable 

sns.countplot(x = "response", data = response)
plt.ylabel("Absolute counts")
plt.xlabel('Status')
plt.title("Response status at end of follow up (Binary variable)")
plt.show()


# %% 2. Merge tables using "participant_id" (used by default)
# ===========================================


Base = baseline_df.merge(
    inclusion_df,
    left_on = "participant_id",
    right_on = "participant_id",
    how = "left"
)

Base["id"] = baseline_df["participant_id"].str.split("-", expand = True).iloc[:, 1].astype(int)




table = pd.merge(pd.merge(inclusion_df, baseline_df), outcome)
table.shape == (141, 36)


# %% 2. merge with supplementary data supp_df
# ===========================================
# %% 3. Save to excel file
#
#table.to_excel(OUTPUT + "data_demoSmokLiMarsResponse_python.xlsx", index=False)
table.to_excel(OUTPUT + "Clinical_data.xlsx", index=False)
table.to_csv(OUTPUT + "rlink-predict-response-clinic_v-20260430", index=False)
