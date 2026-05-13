# Read files
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
assert baseline_df.shape == (168, 162)




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
Response = final_response[final_response[label].isin(labels_to_keep)]
Response = Response[["participant_id",label]]
# keep the variable 'Response.Status.at.end.of.follow.up' as y (label / outcome) for classification
Response = Response.rename(columns={label: "response"})
Response.loc[((Response["response"] == "PaR") | (Response["response"] == "NR") | (Response["response"] == "UC") ), "response"] = "No_GR"

assert set(Response['response'].unique()) == set(["No_GR","GR"])
le = LabelEncoder()
Response['response']=le.fit_transform(Response['response']) # 0 -> GR , 1-> No_GR
# Plotting response variable 

sns.countplot(x = "Response.Status.at.end.of.follow.up", data = final_response)
plt.ylabel("Absolute counts")
plt.xlabel('Status')
plt.title("Response status at end of follow up")
plt.show()
# Plotting response variable 

sns.countplot(x = "response", data = Response)
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
assert final_data.shape == (168, 165)



### ADDING NEW VARIABLES ###
final_data['BMQ_NECESSITY'] = final_data['BMQ1_PRELI'] + final_data['BMQ2_PRELI'] +final_data['BMQ3_PRELI']+final_data['BMQ4_PRELI']+final_data['BMQ5_PRELI']
final_data['BMQ_PREOCCUPATION'] = final_data['BMQ6_PRELI'] + final_data['BMQ7_PRELI'] +final_data['BMQ8_PRELI']+final_data['BMQ9_PRELI']+final_data['BMQ10_PRELI']
final_data['BMQ_DIFFERENTIAL'] = final_data['BMQ_NECESSITY'] - final_data['BMQ_PREOCCUPATION']
final_data['BMQ_GENERAL'] = final_data['BMQ11_PRELI'] +final_data['BMQ12_PRELI']+final_data['BMQ13_PRELI']+final_data['BMQ14_PRELI']+final_data['BMQ15_PRELI']+final_data['BMQ16_PRELI']+final_data['BMQ17_PRELI']+final_data['BMQ18_PRELI']

final_data['MARS_TOTAL'] = np.sum(final_data[final_data.columns[final_data.columns.str.startswith('MARS')]], axis = 1)


# Drop columns BMQ1 - BMQ18
bmq_cols = [f'BMQ{i}_PRELI' for i in range(1, 19)]
final_data = final_data.drop(columns=bmq_cols)

# Drop columns MARS except MARS_TOTAL
mars_cols = [
    col for col in final_data.columns
    if col.startswith('MARS') and col != 'MARS_TOTAL'
]

final_data = final_data.drop(columns=mars_cols)

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
assert final_data.shape == (168, 145)



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
                     'DeltaBMI','Delta_BMI_impute','DeltaAPA', 'DeltaATD', 'DeltaAC', 'DeltaNLP','DeltaBZD', #M03_in_Delta
                     'QIDS_W1_M03','BMI_M03','BRMS_W1_M03', #M03
                     'CTGY-CategorieComorbiditeSomatique_n°1','CTGY-CategorieComorbiditeSomatique_n°2etplus',
                     'DISEASE-PathologieComorbiditeSomatique_n°1','DISEASE-PathologieComorbiditeSomatique_n°2etplus','DISRDR-Trouble_n°1',
                     'DISRDR-Trouble_n°2etplus' # columns are not useful for our study

                     ]
Variables_supp_in_final_data=['PHCMBY_PLI-ComorbiditeSomatique','QIDS_TotalScore_M00'
                              ,'BRMS_TotalScore_M00','NbHospitalizationsLifetime',
                              'SmokingStatus-WHOA1A_PLI',
                              ]


## Differences between final_data and supp_df for AGE and SEX
# Align by participant_id
final_sorted = final_data.set_index("participant_id").sort_index()
supp_sorted = supp_df.set_index("participant_id").sort_index()

# Ensure same index order
final_sorted, supp_sorted = final_sorted.align(supp_sorted)

# Create mask safely
mask = (
    final_sorted["AGE"].ne(supp_sorted["AGE"]) |
    final_sorted["SEX"].ne(supp_sorted["SEX"])
)

# Get real differing indices
diff_indices = final_sorted.index[mask]

print("Differences found:", mask.any())
print("Differing participant_id:", diff_indices.tolist())

# There are no differences between final_data and supp_df for AGE and SEX when aligned by participant_id



variables_to_keep = list(set(Variables_supp) - set(Variables_to_drop))
final_data = final_data.merge(
    supp_df[variables_to_keep], 
    on='participant_id', 
    how = "inner"
)
assert final_data.shape == (168, 158)





###Coreelation (Variables_supp_in_final_data and variables of final_data)###

df_corr=final_data
for column in Variables_supp_in_final_data :
    df_corr[column]=pd.to_numeric(df_corr[column],errors='coerce')

cols_num= df_corr.select_dtypes(include=['number']).columns

corr_matrix = df_corr[cols_num].corr()
cols_num=cols_num.drop('PHCMBY_PLI-ComorbiditeSomatique')
cols_num=cols_num.drop('QIDS_TotalScore_M00')
cols_num=cols_num.drop('BRMS_TotalScore_M00')
cols_num=cols_num.drop('NbHospitalizationsLifetime')
cols_num=cols_num.drop('SmokingStatus-WHOA1A_PLI')

result = corr_matrix.loc[Variables_supp_in_final_data,cols_num]
constent=0.9
filtered_result=result[result.abs()>constent]
high_corr_pairs= filtered_result.stack().reset_index()
high_corr_pairs

# So the variables in Variables_supp_in_final_data already in final_data:
# PHCMBY_PLI-ComorbiditeSomatique = PHCMBY_PLI
# QIDS_TotalScore_M00 = QIDSTSC_PRELI
# BRMS_TotalScore_M00 = BRMSTSC_PRELI
# NbHospitalizationsLifetim = NBH2_PLI
# SmokingStatus-WHOA1A_PLI = WHOA1A_PLI

rename_dict = {
    'PHCMBY_PLI': 'PHCMBY_PLI-ComorbiditeSomatique',
    'QIDSTSC_PRELI': 'QIDS_TotalScore_M00',
    'BRMSTSC_PRELI': 'BRMS_TotalScore_M00',
    'NBH2_PLI': 'NbHospitalizationsLifetime',
    'WHOA1A_PLI': 'SmokingStatus-WHOA1A_PLI'
}

# Suppression des anciennes variables
final_data = final_data.drop(columns=Variables_supp_in_final_data)

# Renommage des variables restantes
final_data = final_data.rename(columns=rename_dict)

final_data = final_data.rename(columns={
    "PSYHLTH_PLI-ComorbiditePsychiatrique": "Psychiatric_Comorbidity"
})


## Merging response variable and base

final_data = pd.merge(
    final_data, 
    Response[["participant_id", "response"]], 
    on='participant_id', 
    how='inner'
)
assert final_data.shape == (138, 154)  #30 individuals have no value for the response variable.

final_data= final_data.replace('ND',np.nan) #Replace 'ND' values with NaN as they represent missing data
final_data[["MOOD_PLI", "ANTIPSY_PLI", "NEUROL_PLI", "ANTIDEP_PLI", "BENZOS_PLI"]] = (
    final_data[["MOOD_PLI", "ANTIPSY_PLI", "NEUROL_PLI", "ANTIDEP_PLI", "BENZOS_PLI"]]
    .replace(9, np.nan)
)
final_data[["RCY1_PLI","RCY2_PLI"]] = (
    final_data[["RCY1_PLI","RCY2_PLI"]]
    .replace(2, np.nan)
)

### Filter columns based on missing valeu percentage
final_data.apply(lambda x: (x.isna().mean())*100, axis = 0)

## Selecting an arbitrary threshold of missing values

threshold = 0.4
final_data.columns[final_data.apply(lambda x: x.isna().mean(), axis = 0) > threshold]

total = len(final_data.columns)

## plot of number of features remaining depending on the chosen thresold 
values_thresholds = np.arange(0.0, 1.0, 0.01)

dict_values = {key:0 for key in values_thresholds}

for key in dict_values.keys():
    n = len(final_data.columns[final_data.apply(lambda x: x.isna().mean(), axis = 0) > key])
    dict_values[key] = total - n


df = pd.DataFrame.from_dict(dict_values, orient = "index", dtype = float)

df.reset_index(inplace = True)
df['new_index'] = 1 - df['index']
df.set_index("new_index", inplace = True)
df.drop("index", axis =1, inplace = True )


sns.lineplot(df)
plt.ylabel("Features remaining")
plt.xlabel('Ratio of NAs (threshold)')
plt.legend('')
plt.grid()

#Choosing a threshold for missing values 
threshold = 0.4
cols_to_drop = final_data.columns[final_data.apply(lambda x: x.isna().mean(), axis = 0) > threshold]
df_final = final_data.drop(cols_to_drop, axis = 1)
df_final.info()

assert df_final.shape == (138, 124) 


###Missing_per row

missing_rate = df_final.isnull().mean(axis=1)
max_missin = missing_rate.max()

# Plot the distribution of missing rates per row
plt.figure(figsize=(10, 6))
plt.hist(missing_rate, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Missing Values per Row')
plt.xlabel('Missing Rate (Percentage)')
plt.ylabel('Number of Rows')
plt.legend()
plt.show()

# the maximum is 48.34% -> Keep all observations



statistic_df= df_final

'''
#test the correlation before impotation

def high_correlation_pairs(X, threshold=0.8):
    corr_matrix = X.corr(method="spearman")
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2", 0: "correlation"})
    )
    high_corr = high_corr[high_corr["correlation"].abs() > threshold]
    high_corr = high_corr.sort_values("correlation", ascending=False).reset_index(drop=True)
    
    return high_corr
df_final=df_final.drop("participant_id",axis=1)
pairs = high_correlation_pairs(df_final, threshold=0.85)
print(f"{len(pairs)} paires avec |r| > 0.85\n")
print(pairs.to_string())

'''

### Imputation

binary_cols = [
    "CENTERNUM", "SEX", "MIX_PRELI", "JOB_PRELI", "EVNT_PRELI", "TSH_PRELI",
    "MOOD_PLI", "ANTIPSY_PLI", "NEUROL_PLI", "ANTIDEP_PLI", "BENZOS_PLI","CURRMED_PRELI",
    "HYPOE1_PLI", "SSRS1_PRELI", "SSRS2_PRELI","RCY1_PLI","RCY2_PLI","Psychiatric_Comorbidity",
    "TRQ_PRELI", "TRQ1_PRELI","HOSP_PRELI","MOODYN_PRELI","FHIST_PLI", "SmokingStatus-WHOA1A_PLI",
    "WHOA1B_PLI", "WHOA1C_PLI", "WHOA1D_PLI","SSRS6_PRELI",
    "WHOA1E_PLI", "WHOA1F_PLI", "WHOA1G_PLI","PHCMBY_PLI-ComorbiditeSomatique","WHOA1H_PLI",
    "BMI_M00", "DensityHospit", "SuicideAttempts(Yes/No)","TRQ2_PRELI",
]

continuous_cols = [
    "AGE",  "WEIGHT_PRELI", "HEIGHT_PRELI", "WAIST_PRELI","SBP_PRELI", 
     "NA_PRELI", "K_PRELI", "CA_PRELI", "UREA_PRELI","BPRSTSC_PRELI",
    "CREAT_PRELI", "EGFR_PRELI", "MDRD_PRELI", "CKDEPI_PRELI",
     "MDE1_PLI", "MDEH1_PLI", "MDEPS1_PLI", "MDEMC1_PLI", "MDETD1_PLI",
    "MANE1_PLI", "MANEH1_PLI", "MANEPS1_PLI", "MANEMC1_PLI", "MANETD1_PLI",
    "NBH1_PLI", "TDH1_PLI", "AD1_PLI", "SUD1_PLI","AGEMANE2_PLI",
     "MDE2_PLI", "MDEH2_PLI", "MDEPS2_PLI", "MDEMC2_PLI", "MDETD2_PLI","AGEMDE2_PLI",
    "HYPOEH2_PLI", "HYPOEMC2_PLI", "HYPOETD2_PLI", "AGEHYPOE2_PLI",
    "MANE2_PLI", "MANEH2_PLI", "MANEPS2_PLI", "MANEMC2_PLI", "MANETD2_PLI",
    "NbHospitalizationsLifetime", "TDH2_PLI", "AGESTBH2_PLI", "AD2_PLI", "SUD2_PLI", "MC2_PLI",
    "QIDS_TotalScore_M00", "BRMS_TotalScore_M00","HYPOEH1_PLI", "HYPOEMC1_PLI", "HYPOETD1_PLI",
    "BMQ_NECESSITY", "BMQ_PREOCCUPATION", "BMQ_DIFFERENTIAL", "BMQ_GENERAL",
    "fhist_count", "fhist_ratio_h", "fhist_repli","DensityEpisodes",
    "AgeAtOnset", "NumberPreviousEpisodes", "DurationIllness",
]




ordinal_cols = [
    "DBP_PRELI",          # stades 1-7
    "SCHOOL_PRELI",       # niveau scolaire 1-8
    "OUTW1_PLI",          # résultat période 1
    "NBS1_PLI",           # nb épisodes sévères 1
    "MC1_PLI",            # comorbidité médicale 1
    "HYPOE2_PLI",         # 0-3
    "OUTW2_PLI",          # résultat période 2
    "NBS2_PLI",           # nb épisodes sévères 2
    "MARS_TOTAL",         # 1-5
    "TRQ5_PRELI",         # 1-3
    "TRQ7_PRELI",         # 1-5
    "TRQ4_PRELI",
    "TRQ6_PRELI",
]


nominal_cols = [
    "TYPEP_PRELI",      # type de trouble
    "RELSTAT_PRELI",    # statut relationnel
    "ETHNICITY_PRELI",  # ethnicité
    "CORG_PRELI",       # organisation des soins
    "LIVSIT_PRELI",     # situation de vie
    "RESIDENCE_PRELI",  # type de résidence
]




mode_imputer = SimpleImputer(strategy="most_frequent")
df_final[binary_cols] = mode_imputer.fit_transform(df_final[binary_cols])


df_final[nominal_cols] = mode_imputer.fit_transform(df_final[nominal_cols])

median_imputer = SimpleImputer(strategy="median")
df_final[continuous_cols] = median_imputer.fit_transform(df_final[continuous_cols])  

mode_imputer = SimpleImputer(strategy="most_frequent")
df_final[ordinal_cols] = mode_imputer.fit_transform(df_final[ordinal_cols])


print(df_final.isnull().sum().sum()) 




# Rename columns

labels={"MOODYN_PRELI": "Current mood episode (MOODYN_PRELI)",
        "TYPEP_PRELI": "Type of episode (TYPEP_PRELI)",
        "PS_PRELI": 'With psychotic sympt. (PS_PRELI)',
        "MIX_PRELI": "With mixed characteristics (MIX_PRELI)",
        "HOSP_PRELI": "Currently hospitalized (HOSP_PRELI)",
        "WEIGHT_PRELI":"Weight (WEIGHT_PRELI)",
        "HEIGHT_PRELI":"Height (HEIGHT_PRELI)",
        "SBP_PRELI":"Systolic BP (SBP_PRELI)",
        "DBP_PRELI":"Diastolic BP (DBP_PRELI)",
        "RELSTAT_PRELI": "Relationship status (RELSTAT_PRELI)",
        "ETHNICITY_PRELI": "Ethnicity (ETHNICITY_PRELI)",
        "CORG_PRELI": "Country of Origin (CORG_PRELI)",
        "LIVSIT_PRELI": "Living situation (LIVSIT_PRELI)",
        "RESIDENCE_PRELI": "Place of Residence (RESIDENCE_PRELI)",
        "SCHOOL_PRELI": "Highest qualification (SCHOOL_PRELI)",
        "JOB_PRELI": "Employment status (JOB_PRELI)",
        "EVNT_PRELI": "Recent stressful events inf. 12 mo (EVNT_PRELI)",
        "CURRMED_PRELI": "Current medications (CURRMED_PRELI)",
        "NA_PRELI": "Sodium (NA_PRELI)",
        'K_PRELI': 'Potassium (K_PRELI)',
        'CL_PRELI': "Chlore (CL_PRELI)",
        'CA_PRELI': "Calcium (CA_PRELI)",
        "PROTEINS_PRELI": "Proteins (PROTEINS_PRELI)",
        "UREA_PRELI": "Urea (UREA_PRELI)",
        "CREAT_PRELI": "Creatinine (CREAT_PRELI)",
        "EGFR_PRELI": "Glomerular filtration rate (EGFR)(EGFR_PRELI)",
        "MDRD_PRELI": "Glomerular filtration rate (EGFR) MDRD (MDRD_PRELI)",
        'CKDEPI_PRELI': "Glomerular filtration rate (EGFR) (CKDEPI_PRELI)",
        "TSH_PRELI": "TSH (TSH_PRELI)",
        'HB_PRELI': "Hemoglobin (HB_PRELI)",
        "PLT_PRELI": "PLatelets (PLT_PRELI)",
        "QIDSTSC_PRELI":"QIDS Total (QIDSTSC_PRELI)",
        "BRMSTSC_PRELI": 'BRMS Total (BRMSTSC_PRELI)',
        "BPRSTSC_PRELI": "BPRS Total (BPRSTSC_PRELI)",
        'SEX': "Male sex",
        "response": "Lithium Response (4 levels)",
        "FHIST_PLI": "Family history",
        "fhist_count":"F.hist absolute count",
        "fhist_ratio_h": "F.Hist male ratio",
        "fhist_repli": "F.Hist positive lithium response",
        "MOOD_PLI": "Mood stabilizers - 2y",
        "ANTIPSY_PLI": "Antipsychotics - 2y",
        "NEUROL_PLI": "Neuroleptics - 2y",
        "ANTIDEP_PLI": "Antidepressants - 2y",
        "BENZOS_PLI": "Benzodiazepines - 2y",
        "PHCMBY_PLI": "Physical comorbidity",
        "RCY1_PLI": "Rapid cycling - 2y",
        "MDE1_PLI":"Depressive. ep. total duration (w) -2y",
        "MDEH1_PLI": 'MDE hospitalized -duration (w)-2y',
        "MDEPS1_PLI": "MDE psychotic symptoms duration (w)- 2y",
        "MDEMC1_PLI": "MDE mixed duration (w) - 2y"   
        }

df_final= df_final.rename(columns = labels)

### X, y ###

y = df_final["Lithium Response (4 levels)"]

X = df_final.drop(columns=["Lithium Response (4 levels)", "participant_id"])



# final_data_dummy = pd.get_dummies(X_imputed_)

# Convertir toutes les colonnes object en float
X[X.select_dtypes(include='object').columns] = (
    X.select_dtypes(include='object').astype(float)
)

# Vérification
print(X.dtypes.value_counts())


### Correlation matrix

corr_matrix = X.corr()

plt.figure(figsize=(20, 16))
sns.heatmap(
    corr_matrix,
    annot=False,        
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.3,
    square=True,
)
plt.title("Matrice de corrélation", fontsize=14)
plt.tight_layout()
plt.show()


def high_correlation_pairs(X, threshold=0.8):
    corr_matrix = X.corr(method="spearman")
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2", 0: "correlation"})
    )
    high_corr = high_corr[high_corr["correlation"].abs() > threshold]
    high_corr = high_corr.sort_values("correlation", ascending=False).reset_index(drop=True)
    
    return high_corr

pairs = high_correlation_pairs(X, threshold=0.85)
print(f"{len(pairs)} paires avec |r| > 0.85\n")
print(pairs.to_string())

remaining_to_remove = [
    "NBS2_PLI",        # = SuicideAttempts(Yes/No) (r=0.99)     (Number of suicide attempts - SuicideAttempts(Yes/No) )
    "MDRD_PRELI",      # = CKDEPI_PRELI (r=0.95)                (CKDEPI_PRELI - MDRD_PRELI)
    "WHOA1A_PLI",      # = SmokingStatus-WHOA1A_PLI (r=0.94)    (Tobacco products (cigarettes, chewing tobacco, cigars, etc.) - SmokingStatus )
    "NBH1_PLI",        # = TDH1_PLI (r=0.88)                    (Number of hospitalizations - Total duration of hospitalizations )
    "AD2_PLI",         # = AD1_PLI (r=0.87)                     (Anxity disorders in the last in the life time - 2 yrs )
    "MDE2_PLI",        # = NumberPreviousEpisodes (r=0.85)      (Major Depressive episodes - NumberPreviousEpisodes)
]


remaining_to_remove = [c for c in remaining_to_remove if c in X.columns]

X_clean = X.drop(columns=remaining_to_remove)

print(f"{X.shape[1]} colonnes → {X_clean.shape[1]} colonnes conservées")

pairs_final = high_correlation_pairs(X_clean, threshold=0.85)
print(f"\nPaires restantes avec |r| > 0.85 : {len(pairs_final)}")
print(pairs_final.to_string())

### Multicolinerarity assessment: Matrix and VIF 

# ── Calcul du VIF ─────────────────────────────────────────────────────────────
def compute_vif(X):
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)

vif_df = compute_vif(X_clean)
print(vif_df)




# %% 3. Save to excel file
#
#table.to_excel(OUTPUT + "data_demoSmokLiMarsResponse_python.xlsx", index=False)
#table.to_excel(OUTPUT + "Clinical_data.xlsx", index=False)
#table.to_csv(OUTPUT + "rlink-predict-response-clinic_v-20260430", index=False)
'''
df_ = df_final.select_dtypes(exclude = "datetime")
df_.drop(["DATBIO_PRELI"], axis = 1, inplace = True)
#cols_to_select = df_.columns[~(df_.columns.str.startswith('BMQ') | df_.columns.str.startswith('MARS') | df_.columns.str.startswith('TRQ'))]
#df_ = df_[cols_to_select]
df_.to_csv(OUTPUT + 'CLINICAL_DATA.csv')
df_
'''