################################################################################
# %% Imports
# ----------

# System
import sys
import os
import os.path
import time
import json
from datetime import datetime


# Scientific python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests

# Models
from sklearn.base import clone

# Metrics
import sklearn.metrics as metrics

# Resampling
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
#from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import sklearn.linear_model as lm
from sklearn.compose import ColumnTransformer

# Set pandas display options
pd.set_option('display.max_colwidth', None)  # No maximum width for columns
pd.set_option('display.width', 1000)  # Set the total width to a large number

################################################################################
# %% Config dictionary
# --------------------

config = {
    # Working directory
    "WD": "/neurospin/signatures/2026_ychaoub_rlink_predict_response_clinic/",
    # Input data
    "input_data" : "./data/Rlink_version3_type_Clinic_timepoint_M00_M03_v20260602.csv",
    "cv_test": "stratified-5cv.json",
    # Paths
    "LD_LIBRARY_PATH": ["/home/yc287630/.local/lib/python/site-packages/nitk/lib"],
    #"models_path": "./classification_models.py"
    # Output directories
    'output_models': "./models",
    'output_reports': "./reports",
    # Variables
    "target": "response",
    "target_remap": {"NR":0, "PaR":0, "GR":1},
    "residualization": ["age", "sex", "site"],
    "drop": ["participant_id"],
    "metrics": ["accuracy", "balanced_accuracy", "roc_auc"],
}

# Initialize WD
os.chdir(config["WD"])

    
# LD_LIBRARY_PATH
if "LD_LIBRARY_PATH" in config:
    for path in config["LD_LIBRARY_PATH"]:
        sys.path.append(path) # Add to system path

n_splits_val = 5
cv_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=42)
