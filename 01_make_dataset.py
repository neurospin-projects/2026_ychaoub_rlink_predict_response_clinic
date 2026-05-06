# Read files
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
assert baseline_df.shape == (168, 164)




## All other relevant variables from base dataframe

# 1.4 Variables from file "dataset-clinical_version-2_outcome.tsv"

vars = ["participant_id", "Response.Status.at.end.of.follow.up"]
final_response = pd.read_csv(INPUT + "dataset-outcome_version-4.tsv", sep='\t')
#final_response= outcome[vars]
#df4.participant_id = ["sub-"+ str(id) for id in df4.participant_id]
#assert final_response.shape == (168, 2)


# Plotting response variable 

sns.countplot(x = "Response.Status.at.end.of.follow.up", data = final_response)
plt.ylabel("Absolute counts")
plt.xlabel('Status')
plt.title("Response status at end of follow up")
plt.show()

###Response variables 
# column of population dataframe that defines response to Li label
label = 'Response.Status.at.end.of.follow.up'

print(final_response[label].unique())

print("in population dataframe: \nnumber of Good Responders (GR) :",len(final_response[final_response[label]=="GR"].values))
print("number of Partial Responders (PaR) :",len(final_response[final_response[label]=="PaR"].values))
print("number of Non Responders (NR) :",len(final_response[final_response[label]=="NR"].values))
print("number of UnClassified (UC) :",len(final_response[final_response[label]=="UC"].values))

# we ignore the unclassified subjects, and keep only the good responders, partial responders, and non-responders
labels_to_keep= ["GR","PaR","NR"]
final_to_keep = final_response[final_response[label].isin(labels_to_keep)]
final_to_keep = final_to_keep[["participant_id",label]]
# keep the variable 'Response.Status.at.end.of.follow.up' as y (label / outcome) for classification
final_to_keep = final_to_keep.rename(columns={label: "response"})
assert set(final_to_keep['response'].unique()) == set(["NR","PaR","GR"])

# Plotting response variable 

sns.countplot(x = "response", data = final_to_keep)
plt.ylabel("Absolute counts")
plt.xlabel('Status')
plt.title("Response status at end of follow up")
plt.show()

# %% 2. Merge tables using "participant_id" (used by default)
# ===========================================

final_data = pd.merge(inclusion_df, 
                      baseline_df,
                      on='participant_id', 
                      how='inner'
)
assert baseline_df.shape == (168, 164)



### ADDING NEW VARIABLES ###
final_data['BMQ_NECESSITY'] = final_data['BMQ1_PRELI'] + final_data['BMQ2_PRELI'] +final_data['BMQ3_PRELI']+final_data['BMQ4_PRELI']+final_data['BMQ5_PRELI']
final_data['BMQ_PREOCCUPATION'] = final_data['BMQ6_PRELI'] + final_data['BMQ7_PRELI'] +final_data['BMQ8_PRELI']+final_data['BMQ9_PRELI']+final_data['BMQ10_PRELI']
final_data['BMQ_DIFFERENTIAL'] = final_data['BMQ_NECESSITY'] - final_data['BMQ_PREOCCUPATION']
final_data['BMQ_GENERAL'] = final_data['BMQ11_PRELI'] +final_data['BMQ12_PRELI']+final_data['BMQ13_PRELI']+final_data['BMQ14_PRELI']+final_data['BMQ15_PRELI']+final_data['BMQ16_PRELI']+final_data['BMQ17_PRELI']+final_data['BMQ18_PRELI']

final_data['MARS_TOTAL'] = np.sum(final_data[final_data.columns[final_data.columns.str.startswith('MARS')]], axis = 1)


#Data cleaning
final_data["FHIST_PLI"].replace(9.0, 0.0, inplace = True)

fhist_ratio_h = fhist.groupby('participant_id')['SEX_FH'].apply(lambda x: (x == 1.0).mean())
fhist_ratio_h.name = "fhist_ratio_h"
fhist_repli = fhist.groupby('participant_id')['REPLI_FM'].apply(lambda x: (x == 1.0).mean())
fhist_repli.name = "fhist_repli"
fhist_repli.value_counts()

### Counting relatives 
fhist_count = fhist.groupby("participant_id").size().sort_values()
fhist_count.name = "fhist_count"

final_data = final_data.merge(
    fhist_count, 
    on='participant_id', 
    how = "left"
)

final_data = final_data.merge(
    fhist_ratio_h, 
    on='participant_id', 
    how = "left"
)

final_data = final_data.merge(
    fhist_repli, 
    on='participant_id', 
    how = "left"
)
assert baseline_df.shape == (168, 164)



final_data['fhist_count'].fillna(0, inplace = True)
final_data['fhist_ratio_h'].fillna(0, inplace = True)
final_data['fhist_repli'].fillna(0, inplace = True)

final_data.shape


# %% 2. merge with supplementary data supp_df
# ===========================================
Variables_supp=['participant_id', 'AGE', 'SEX', 
                'BMI_M00', 'BMI_M03', 'DeltaBMI',
       'Delta_BMI_impute', 'PHCMBY_PLI-ComorbiditeSomatique',
       'CTGY-CategorieComorbiditeSomatique_n°1',
       'CTGY-CategorieComorbiditeSomatique_n°2etplus',
       'DISEASE-PathologieComorbiditeSomatique_n°1',
       'DISEASE-PathologieComorbiditeSomatique_n°2etplus',
       'PSYHLTH_PLI-ComorbiditePsychiatrique', 'DISRDR-Trouble_n°1',
       'DISRDR-Trouble_n°2etplus', 'AgeAtOnset', 'NumberPreviousEpisodes',
       'DurationIllness', 'DensityEpisodes', 'NbHospitalizationsLifetime',
       'DensityHospit', 'SmokingStatus-WHOA1A_PLI', 'SuicideAttempts(Yes/No)',
       'QIDS_TotalScore_M00', 'QIDS_W1_M03', 'BRMS_TotalScore_M00',
       'BRMS_W1_M03', 'DeltaAPA', 'DeltaATD', 'DeltaAC', 'DeltaNLP',
       'DeltaBZD']
Variables_to_drop = ['AGE', 'SEX', #variables_in_inclusion
                     'DeltaBMI','Delta_BMI_impute','DeltaAPA', 'DeltaATD', 'DeltaAC', 'DeltaNLP','DeltaBZD' #M03_in_Delta
                     'QIDS_W1_M03','BRMS_W1_M03', #M03
                     ]
Variables_might_be_in_final_data=['PHCMBY_PLI-ComorbiditeSomatique',
                                  'QIDS_TotalScore_M00','QIDS_W1_M03',
                                  'BRMS_TotalScore_M00']

Variables_idk = ['DISRDR-Trouble_n°1',
                 'DISRDR-Trouble_n°2etplus']

## Merging response variable and base

f_data = pd.merge(
    final_data, 
    final_to_keep[["participant_id", "response"]], 
    on='participant_id', 
    how='inner'
)
assert baseline_df.shape == (168, 164)


### Missing values assessment 
final_data.apply(lambda x: (x.isna().mean())*100, axis = 0)

## Removing response variables with absent values (for now) n = 30 responses 
print("Missing values : ", final_data["response_2"].isna().sum())
final_data_no_na = final_data.dropna(subset = ['response_2'], how = "any")

## Selecting an arbitrary threshold of missing values

threshold = 0.4
final_data_no_na.columns[final_data_no_na.apply(lambda x: x.isna().mean(), axis = 0) > threshold]

total = len(final_data_no_na.columns)

## plot of number of features remaining depending on the chosen thresold 
values_thresholds = np.arange(0.0, 1.0, 0.01)

dict_values = {key:0 for key in values_thresholds}

for key in dict_values.keys():
    n = len(final_data_no_na.columns[final_data_no_na.apply(lambda x: x.isna().mean(), axis = 0) > key])
    dict_values[key] = total - n


df = pd.DataFrame.from_dict(dict_values, orient = "index", dtype = float)

df.reset_index(inplace = True)
df['new_index'] = 1 - df['index']
df.set_index("new_index", inplace = True)
df.drop("index", axis =1, inplace = True )


assert baseline_df.shape == (168, 164)



sns.lineplot(df)
plt.ylabel("Features remaining")
plt.xlabel('Ratio of NAs (threshold)')
plt.legend('')
plt.grid()

#Choosing a threshold for missing values 
threshold = 0.4
cols_to_drop = final_data_no_na.columns[final_data_no_na.apply(lambda x: x.isna().mean(), axis = 0) > threshold]
df_final = final_data_no_na.drop(cols_to_drop, axis = 1)
df_final.info()




# %% 3. Save to excel file
#
#table.to_excel(OUTPUT + "data_demoSmokLiMarsResponse_python.xlsx", index=False)
#table.to_excel(OUTPUT + "Clinical_data.xlsx", index=False)
#table.to_csv(OUTPUT + "rlink-predict-response-clinic_v-20260430", index=False)

df_ = df_final.select_dtypes(exclude = "datetime")
df_.drop(["DATBIO_PRELI"], axis = 1, inplace = True)
#cols_to_select = df_.columns[~(df_.columns.str.startswith('BMQ') | df_.columns.str.startswith('MARS') | df_.columns.str.startswith('TRQ'))]
#df_ = df_[cols_to_select]
df_.to_csv(OUTPUT + 'CLINICAL_DATA.csv')
df_
