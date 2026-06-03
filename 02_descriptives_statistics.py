"""
Univariate statistics
"""


import pandas as pd
import numpy as np
import os

from utils.stats_pairwise import pairwise_stats
#from ml_utils import get_residualizer, create_print_log
from config import config


################################################################################
# %% Load Data
# ============

data = pd.read_csv(config['input_data'])

# Select Input = dataframe - (target + drop + residualization)
feature_columns = [c for c in data.columns if c not in [config['target']] + \
    config['drop'] + config['residualization']]

X = data[feature_columns].values
y = data[config['target']]


# %% Run Univariate statistics
# ============================

# Create DataFrame for stats
X_df = pd.DataFrame(X, columns=feature_columns)
X_df["response"] = y.values
X_df["AGE"] = data["AGE"]
X_df["Male sex"] = data["Male sex"]
X_df["CENTERNUM"] = data["CENTERNUM"]


stats = pairwise_stats(X_df, vars1=['response'],
                       vars2=feature_columns + ['AGE', 'Male sex', 'CENTERNUM'], cattest="prop_ztest")
stats.sort_values('pval', inplace=True)

excel_path = 'reports/statistics_univariate.xlsx'
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    stats.to_excel(writer, sheet_name="statistics_univariate", index=False)
print(f"Saved {excel_path}")


