"""
Supervised Classification: Clinical Data — Responder vs Non-Responder
======================================================================
Benchmark complet incluant les modèles du tableau "Models benchmark - 1st
model with all features " :
  - Logistic Regression
  - ElasticNet (Logistic Regression + pénalité elasticnet)
  - SVM RBF
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Multilayer Perceptron (MLP)

Pour chaque modèle, on calcule :
  - Accuracy CV (CV interne sur le train)
  - Balanced accuracy train / AUC train (performance sur le train, fit complet)
  - Balanced Accuracy test / AUC test (sur le set de test, ou en pooled OOF
    si CV externe prédéfinie)
  - Test de permutation (p-value, distribution nulle [IC 95%])

Label convention:
  0 = Responder (R)
  1 = Non-Responder (NR)
"""

################################################################################
# %% 1. Imports
# =============

#import osssh
import numpy as np
import pandas as pd
import warnings

from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    cross_validate, cross_val_score, cross_val_predict,
    StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone

import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    make_scorer, recall_score, balanced_accuracy_score, roc_auc_score,
    accuracy_score
)

warnings.filterwarnings("ignore")


################################################################################
# %% 2. Configuration
# ====================

CSV_PATH     = "data/Rlink_version3_type_Clinic_timepoint_M00_v20260602.csv"
TYPE_CSV     = "data/Rlink_Clinical_variables_M00_mapping.csv"
TARGET_COL   = "response"          # 0 = Responder, 1 = Non-Responder
ID_COL       = "participant_id"
N_SPLITS_OUTER = 5  # nombre de folds pour la CV externe (StratifiedKFold)
OUTPUT       = "reports/clinical_classification_reports.xlsx"

RANDOM_STATE = 42
N_JOBS       = 5          # jobs for GridSearchCV
N_PERMUTATIONS = 1000      # number of permutations for the permutation test

#os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)


data = pd.read_csv(CSV_PATH)
print(f"Data loaded: {data.shape[0]} subjects, {data.shape[1]} columns")


################################################################################
# %% 3. Build column type lists from variable mapping CSV
# =========================================================

Type_data = pd.read_csv(TYPE_CSV)
TYPE_MAP = dict(zip(Type_data["variable"], Type_data["type"]))
csv_cols = set(data.columns)

BINARY_COLS = [
    col for col, typ in TYPE_MAP.items()
    if typ == "binary" and col in csv_cols
]

CATEGORIAL_COLS = [
    col for col, typ in TYPE_MAP.items()
    if typ == "multicategory" and col in csv_cols
]

CONTINUOUS_COLS = [
    col for col, typ in TYPE_MAP.items()
    if typ == "quantitative" and col in csv_cols
]


################################################################################
# %% 4. Type-aware imputation
# ============================

def apply_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using column-type-appropriate strategies:
      - Binary, ordinal, nominal  → mode (most frequent value)
      - Continuous                → median

    Only imputes columns that are present in the DataFrame.

    Parameters
    ----------
    df : DataFrame after optional missingness filtering

    Returns
    -------
    Imputed DataFrame (same columns, no NaN in typed columns)
    """
    mode_imputer   = SimpleImputer(strategy="most_frequent")
    median_imputer = SimpleImputer(strategy="median")

    for col_group in [BINARY_COLS, CATEGORIAL_COLS]:
        present = [c for c in col_group if c in df.columns]
        if present:
            df[present] = mode_imputer.fit_transform(df[present])

    present_cont = [c for c in CONTINUOUS_COLS if c in df.columns]
    if present_cont:
        df[present_cont] = median_imputer.fit_transform(df[present_cont])

    remaining_na = df.isnull().sum().sum()
    print(f"Remaining missing values after imputation: {remaining_na}")
    return df


################################################################################
# %% 5. CV externe (StratifiedKFold 5 folds)
# ============================================
#
# La validation croisée externe utilise une StratifiedKFold à N_SPLITS_OUTER
# folds (5 par défaut), stratifiée sur la variable cible (response), au lieu
# d'une CV prédéfinie par participant_id chargée depuis un JSON.


################################################################################
# %% 6. Models
# =============
#
# Reproduit le tableau "Models benchmark - 1st model with all features (p=81)" :
#   - Logistic Regression
#   - ElasticNet (logistic regression, penalty='elasticnet')
#   - SVM RBF
#   - Random Forest
#   - Gradient Boosting
#   - XGBoost
#   - Multilayer Perceptron (MLP)
#
# Chaque modèle = StandardScaler + classifieur, classifieur potentiellement
# enveloppé dans un GridSearchCV pour la sélection d'hyperparamètres.

MLP_PARAM_GRID = {
    "hidden_layer_sizes": [
        (100,), (50,), (25,), (10,), (5,),          # 1 hidden layer
        (100, 50), (50, 25), (25, 10), (10, 5),     # 2 hidden layers
        (100, 50, 25), (50, 25, 10), (25, 10, 5),   # 3 hidden layers
    ],
    "activation": ["relu"],
    "solver": ["sgd"],
    "alpha": [0.0001],
}


def make_models(n_jobs_grid_search=N_JOBS, cv_val=None, scoring="balanced_accuracy"):
    """
    Construit le dictionnaire des modèles du benchmark.

    Parameters
    ----------
    n_jobs_grid_search : int
        Nombre de jobs parallèles pour les GridSearchCV internes.
    cv_val : CV splitter
        Schéma de validation croisée interne utilisé par GridSearchCV
        (sélection des hyperparamètres).
    scoring : str
        Métrique utilisée pour sélectionner les meilleurs hyperparamètres.

    Returns
    -------
    dict
        {nom_du_modèle: Pipeline}
    """
    if cv_val is None:
        cv_val = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    models = {}

    # 1. Logistic Regression (L2, sans tuning particulier de C)
    models["Logistic Regression"] = make_pipeline(
        StandardScaler(),
        lm.LogisticRegression(
            penalty="l2",
            fit_intercept=True,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )

    # 2. ElasticNet (Logistic Regression + pénalité elasticnet, GridSearch sur alpha/l1_ratio)
    models["ElasticNet"] = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            estimator=lm.SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                fit_intercept=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            param_grid={
                "alpha": 10.0 ** np.arange(-4, 1),
                "l1_ratio": [0.1, 0.5, 0.9],
            },
            cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring,
        ),
    )

    # 3. SVM RBF (probability=True pour pouvoir calculer l'AUC)
    models["SVM RBF"] = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            svm.SVC(kernel="rbf", class_weight="balanced", probability=True,
                    random_state=RANDOM_STATE),
            {"C": 10.0 ** np.arange(-1, 2), "gamma": ["scale", "auto"]},
            cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring,
        ),
    )

    # 4. Random Forest
    models["Random Forest"] = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"n_estimators": [10, 100, 300], "max_depth": [None, 5, 10]},
            cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring,
        ),
    )

    # 5. Gradient Boosting
    models["Gradient Boosting"] = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [10, 100, 300], "learning_rate": [0.01, 0.1]},
            cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring,
        ),
    )

    # 6. XGBoost
    models["XGBoost"] = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            {
                "n_estimators": [50, 100, 300],
                "max_depth": [2, 3, 5],
                "learning_rate": [0.01, 0.1],
            },
            cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring,
        ),
    )

    # 7. Multilayer Perceptron (MLP)
    models["Multilayer Perceptron"] = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            MLPClassifier(random_state=RANDOM_STATE, max_iter=200, tol=0.01),
            MLP_PARAM_GRID,
            cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring,
        ),
    )

    return models



################################################################################
# %% 9. CV externe (StratifiedKFold) + rapport "pooled OOF"
# ===========================================================
def evaluate_models_cv(models, X, y, cv_test, n_jobs=N_JOBS):
    scorers = {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "recall_R":  make_scorer(recall_score, labels=[0], average="macro"),
        "recall_NR": make_scorer(recall_score, labels=[1], average="macro"),
    }

    rows = []
    for name, model in models.items():
        print(f"\n{'─' * 70}")
        print(f"  Model: {name}")
        cv_res = cross_validate(
            estimator=model, X=X, y=y,
            cv=cv_test, scoring=scorers,
            return_estimator=False,
            return_train_score=True,   
            n_jobs=n_jobs, verbose=0,
        )
        row = {
            "model": name,
            # Train
            "balanced_accuracy_train": cv_res["train_balanced_accuracy"].mean(),
            "roc_auc_train":           cv_res["train_roc_auc"].mean(),
            "recall_R_train":          cv_res["train_recall_R"].mean(),
            "recall_NR_train":         cv_res["train_recall_NR"].mean(),
            # Test (OOF)
            "balanced_accuracy_test":  cv_res["test_balanced_accuracy"].mean(),
            "roc_auc_test":            cv_res["test_roc_auc"].mean(),
            "recall_R_test":           cv_res["test_recall_R"].mean(),
            "recall_NR_test":          cv_res["test_recall_NR"].mean(),
        }
        rows.append(row)
        print(f"  balanced_accuracy  train / test : {row['balanced_accuracy_train']:.3f} / {row['balanced_accuracy_test']:.3f}")
        print(f"  roc_auc            train / test : {row['roc_auc_train']:.3f} / {row['roc_auc_test']:.3f}")
        print(f"  recall R  (0)      train / test : {row['recall_R_train']:.3f} / {row['recall_R_test']:.3f}")
        print(f"  recall NR (1)      train / test : {row['recall_NR_train']:.3f} / {row['recall_NR_test']:.3f}")

    return pd.DataFrame(rows)


################################################################################
# %% 10. Prepare data
# ====================

participant_ids = data[ID_COL].values
y = data[TARGET_COL].values   # 0 = Responder, 1 = Non-Responder

drop_cols       = [TARGET_COL, ID_COL]
feature_df      = data[[c for c in data.columns if c not in drop_cols]].copy()
feature_columns = feature_df.columns.tolist()

print(f"Features : {len(feature_columns)} clinical variables")
print(f"Target   : {np.sum(y == 0)} Responders (0)  |  {np.sum(y == 1)} Non-Responders (1)")

feature_df = apply_imputation(feature_df)
X = feature_df.values

assert X.shape[1] == len(feature_columns), "Feature count mismatch after imputation"


################################################################################
# %% 11. Run benchmark
# =====================

if __name__ == "__main__":

    cv_val = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    models = make_models(n_jobs_grid_search=N_JOBS, cv_val=cv_val, scoring="balanced_accuracy")

    # ----- Option A: CV externe StratifiedKFold (pooled OOF)
    cv_test = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)
    print(f"CV splits: {cv_test.get_n_splits()} folds (StratifiedKFold)")
    summary_cv_df = evaluate_models_cv(models, X, y, cv_test, n_jobs=N_JOBS)

    print(f"\n{'=' * 70}")
    print(summary_cv_df.to_string(index=False))

"""
    # ----- Export Excel
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        summary_cv_df.to_excel(writer, sheet_name="cv_stratified5_oof", index=False)
        benchmark_df.to_excel(writer, sheet_name="benchmark_train_test", index=False)

    print(f"\n✔  Results saved to: {OUTPUT}")
"""