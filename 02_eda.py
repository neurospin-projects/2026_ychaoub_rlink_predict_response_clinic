"""
Exploratory Data Analysis (EDA) — Lithium Response
===============================================================
Runs the full EDA pipeline on the clinical feature matrix and saves
figures and a summary Excel workbook to reports/.

Steps
-----
1. Class balance — prints lithium-response group sizes.
2. Descriptive statistics — mean/SD/skewness for continuous variables;
   frequencies/proportions for binary/categorical variables.
3. Clustered correlation heatmap — hierarchical clustering of features
   based on absolute Pearson correlation; saved to eda_correlation_clustermap.png.
4. Pearson correlation matrix — plotted in the cluster order from step 3;
   saved to eda_correlation.png.
5. Variance Inflation Factors (VIF) — multicollinearity diagnosis;
   saved to eda_vif.png.
6. Feature dendrogram — Ward linkage on the VIF-standardised matrix;
   saved to eda_dendrogram.png.
7. PCA scree plot — explained variance vs. number of components with
   elbow detection; saved to eda_pca_components.png.
8. Feature–response associations — per-feature comparison between
   responders and non-responders; saved to eda_feature_response.png.

Outputs
-------
reports/eda_results.xlsx   — one sheet per analysis step, plus
                              ready-to-paste Methods / Results text.
reports/eda_*.png          — individual figures (one per step).

Inputs (from config.py)
-----------------------
data                    : pd.DataFrame — full patient dataset
config['clinical_vars'] : list[str]    — clinical feature names to analyse
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import utils
from utils.eda import (descriptive_stats, plot_correlation,
                       plot_feature_dendrogram, plot_pca_components)

from config import config


################################################################################
# %% Load Data
# ============

data_M00 = pd.read_csv("./data/Rlink_version3_type_Clinic_timepoint_M00_v20260602.csv")
data_M03 = pd.read_csv("./data/Rlink_version3_type_Clinic_timepoint_M03_v20260602.csv")



################################################################################
def eda(data, suffix):

    feature_columns = [
        c for c in data.columns
        if c not in [config['target']]
        + config['drop']
        + config['residualization']
    ]

    X_df = data[feature_columns].copy()
    y = data[config['target']]

    print("\n━━━ Lithium Response — Class Balance ━━━")
    counts = y.value_counts()

    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt} ({100*cnt/len(y):.1f} %)")

    quant_df, cat_df, pub_desc = descriptive_stats(
        data.drop(columns=['participant_id'], errors='ignore'),
        max_cat_unique=2
    )

    X_num = X_df.select_dtypes(include=np.number)

    corr_mat, pub_corr = plot_correlation(
        X_num,
        figscale=5.0,
        filename=f"reports/eda_correlation_{suffix}.png"
    )

    cluster_df, pub_dend = plot_feature_dendrogram(
        X_num,
        filename=f"reports/eda_dendrogram_{suffix}.png"
    )

    corr_mat_reordered, _ = plot_correlation(
        X_num[cluster_df["feature"]],
        figscale=5.0,
        filename=f"reports/eda_correlation_reordered_{suffix}.png"
    )

    scree_df, elbow_idx, thresh_results, pub_pca = plot_pca_components(
        X_num,
        filename=f"reports/eda_pca_components_{suffix}.png"
    )

    pub_df = pd.DataFrame([
        {"function": "descriptive_stats", **pub_desc},
        {"function": "plot_correlation", **pub_corr},
        {"function": "plot_feature_dendrogram", **pub_dend},
        {"function": "plot_pca_components", **pub_pca},
    ])

    excel_path = f"reports/eda_results_{suffix}.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        quant_df.to_excel(writer, sheet_name="desc_quantitative")
        cat_df.to_excel(writer, sheet_name="desc_categorical", index=False)
        corr_mat.to_excel(writer, sheet_name="corr_pearson")
        corr_mat_reordered.to_excel(writer, sheet_name="corr_pearson_reordered")
        cluster_df.to_excel(writer, sheet_name="feature_clusters", index=False)
        scree_df.to_excel(writer, sheet_name="pca_scree", index=False)
        pub_df.to_excel(writer, sheet_name="publication_text", index=False)

    print(f"\n✔ Saved {excel_path}")


eda(data_M00, "M00")
eda(data_M03, "M03")

