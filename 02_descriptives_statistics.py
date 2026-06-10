"""
Univariate statistics + Manhattan plot
=======================================
• Statistiques pairwise GR vs No_GR (M00 et M03)
• Pour chaque timepoint : Manhattan plot des -log10(p-value) par variable,
  colorées par catégorie, avec ligne de significativité et annotations
"""

import os
import colorsys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from utils.stats_pairwise import pairwise_stats, print_publication_table, grouped_stats_table, print_grouped_table

from make_variables_mapping import COLUMN_LABELS, M00_Variables, M03_variables
from config import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, "reports"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#E91E63", "#8BC34A", "#FF5722", "#3F51B5",
    "#009688", "#FFC107", "#673AB7", "#CDDC39", "#795548",
    "#607D8B", "#F06292", "#AED581", "#FFD54F", "#4DD0E1",
]


def _lighten(hex_color: str, amount: float = 0.3) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, l + amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))


def _aliases(name, code_to_label, label_to_code):
    s = {name}
    if name in code_to_label:
        s.add(code_to_label[name])
    if name in label_to_code:
        s.add(label_to_code[name])
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Manhattan plot
# ─────────────────────────────────────────────────────────────────────────────
def manhattan_plot(
    stats_df: pd.DataFrame,
    category_dict: dict,
    label_map: dict = None,
    var_col: str = "v2",
    pval_col: str = "pval",
    alpha: float = 0.05,
    title: str = "Univariate associations — GR vs No_GR",
    save_path: str = None,
    top_n_labels: int = 10,
) -> plt.Figure:

    code_to_label = dict(label_map) if label_map else {}
    label_to_code = {v: k for k, v in code_to_label.items()}

    # ------------------------------------------------------------------
    # Category assignment
    # ------------------------------------------------------------------
    df = stats_df.copy()

    if "category" not in df.columns:

        var_to_cat = {}

        for cat, vars_list in category_dict.items():
            for v in vars_list:
                for a in _aliases(v, code_to_label, label_to_code):
                    var_to_cat[a] = cat

        df["category"] = df[var_col].apply(
            lambda v: next(
                (
                    var_to_cat[a]
                    for a in _aliases(v, code_to_label, label_to_code)
                    if a in var_to_cat
                ),
                "Other",
            )
        )

    # ------------------------------------------------------------------
    # Category order
    # ------------------------------------------------------------------
    cat_order = list(
        dict.fromkeys(
            [c for c in category_dict.keys() if c in df["category"].values]
            + (["Other"] if "Other" in df["category"].values else [])
        )
    )

    cat_color = {
        cat: _PALETTE[i % len(_PALETTE)]
        for i, cat in enumerate(cat_order)
    }

    # ------------------------------------------------------------------
    # Sort variables
    # ------------------------------------------------------------------
    cat_rank = {c: i for i, c in enumerate(cat_order)}

    df["_cat_rank"] = (
        df["category"]
        .map(cat_rank)
        .fillna(len(cat_order))
    )

    df = (
        df.sort_values(["_cat_rank", pval_col])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # -log10(p)
    # ------------------------------------------------------------------
    df["_neglog"] = -np.log10(
        df[pval_col].clip(lower=1e-300)
    )

    # ------------------------------------------------------------------
    # X positions
    # ------------------------------------------------------------------
    gap = 1

    x = 0
    x_ticks = {}

    for cat in cat_order:

        sub_idx = df[df["category"] == cat].index.tolist()

        n = len(sub_idx)

        if n == 0:
            continue

        for rank, idx in enumerate(sub_idx):
            df.at[idx, "_xpos"] = x + rank

        x_ticks[cat] = x + (n - 1) / 2

        x += n + gap

    # ------------------------------------------------------------------
    # Figure (A4 landscape)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    # ------------------------------------------------------------------
    # Scatter points
    # ------------------------------------------------------------------
    for cat in cat_order:

        sub = df[df["category"] == cat]

        color = cat_color[cat]

        for i, (_, row) in enumerate(sub.iterrows()):

            shade = color 

            sig = row[pval_col] < alpha

            ax.scatter(
                row["_xpos"],
                row["_neglog"],
                color=shade,
                s=70 if sig else 35,
                zorder=3 if sig else 2,
                edgecolors="black" if sig else "none",
                linewidths=0.6,
                alpha=0.95 if sig else 0.70,
            )

    # ------------------------------------------------------------------
    # Standard threshold
    # ------------------------------------------------------------------
    thresh = -np.log10(alpha)

    ax.axhline(
        thresh,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"p < {alpha:.2f}",
        zorder=1,
    )

    # ------------------------------------------------------------------
    # Bonferroni threshold
    # ------------------------------------------------------------------
    n_tests = df[pval_col].notna().sum()

    bonf_alpha = max(alpha / n_tests, 1e-300)

    bonf_thresh = -np.log10(bonf_alpha)

    ax.axhline(
        bonf_thresh,
        color="navy",
        linestyle="-.",
        linewidth=1.5,
        label=f"p-adj ({bonf_alpha:.2e})",
        zorder=1,
    )

    # ------------------------------------------------------------------
    # Annotate top hits
    # ------------------------------------------------------------------
    sig_df = (
        df[df[pval_col] < alpha]
        .nsmallest(top_n_labels, pval_col)
        .sort_values("_neglog")
    )

    texts = []

    for i, (_, row) in enumerate(sig_df.iterrows()):

        raw = str(row[var_col])
        disp = code_to_label.get(raw, raw)


        # alternance haut / bas pour éviter overlaps
        y_offset = 10 + (i % 3) * 10
        x_offset = (-1) ** i * 8  # gauche/droite alterné

        ax.annotate(
            disp,
            xy=(row["_xpos"], row["_neglog"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=7,
            ha="center",
            va="bottom",
            rotation=0,
            color=cat_color[row["category"]],
            fontweight="bold",
        )
    # ------------------------------------------------------------------
    # Category labels
    # ------------------------------------------------------------------
    valid_cats = [c for c in cat_order if c in x_ticks]

    ax.set_xticks([x_ticks[c] for c in valid_cats])

    ax.set_xticklabels(
        valid_cats,
        rotation=35,
        ha="right",
        fontsize=10,
        fontweight="bold",
    )

    # colorer les labels selon les catégories
    for tick_label, cat in zip(ax.get_xticklabels(), valid_cats):
        tick_label.set_color(cat_color[cat])
    # ------------------------------------------------------------------
    # Vertical separators
    # ------------------------------------------------------------------
    sep = 0

    for cat in cat_order[:-1]:

        n = (df["category"] == cat).sum()

        sep += n + gap / 2

        ax.axvline(
            sep,
            color="grey",
            linewidth=0.4,
            linestyle=":",
            alpha=0.5,
        )

        sep += gap / 2

    # ------------------------------------------------------------------
    # Legend categories
    # ------------------------------------------------------------------
    patches = [
        mpatches.Patch(
            color=cat_color[c],
            label=c
        )
        for c in valid_cats
    ]

    leg1 = ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        framealpha=0.9,
        title="Category",
        title_fontsize=9,
    )

    ax.add_artist(leg1)

    ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    # ------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------
    ymax = max(
        df["_neglog"].max(),
        bonf_thresh,
    )

    ax.set_ylim(0, ymax * 1.15)

    ax.set_xlim(
        -1,
        df["_xpos"].max() + 1,
    )

    ax.set_ylabel(
        r"$-\log_{10}(p)$",
        fontsize=12,
    )

    ax.set_xlabel(
        "Variable categories",
        fontsize=11,
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(
        rect=[0, 0.08, 0.82, 1]
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "reports"
    )

    os.makedirs(base_dir, exist_ok=True)

    if save_path:

        base_name = os.path.splitext(os.path.basename(save_path))[0]

        png_path = os.path.join(base_dir, base_name + ".png")
        pdf_path = os.path.join(base_dir, base_name + ".pdf")
        svg_path = os.path.join(base_dir, base_name + ".svg")

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")

        print("Saved files:")
        print(png_path)
        print(pdf_path)
        print(svg_path)
# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_univariate(
    data_path: str,
    category_dict: dict,
    timepoint: str,
    drop_cols: list = None,
    extra_drop: list = None,
    cattest: str = "chi2",
    group_labels: dict = None,
):
    print(f"\n{'═'*70}\n  UNIVARIATE — {timepoint}\n{'═'*70}")

    data = pd.read_csv(data_path)

    all_drop = (drop_cols or []) + (extra_drop or [])
    feature_columns = [
        c for c in data.columns
        if c not in [config["target"]] + all_drop
    ]

    X_df = data[feature_columns].copy()
    X_df["response"] = data[config["target"]]

    # ── Pairwise stats ───────────────────────────────────────────────────
    stats = pairwise_stats(
        X_df,
        vars1        = ["response"],
        vars2        = feature_columns,
        cattest      = cattest,
        group_labels = group_labels,
    )
    print_publication_table(stats, label_gr="GR", label_nogr="No_GR", correction="fdr")

    # ── Grouped stats ────────────────────────────────────────────────────
    grouped = grouped_stats_table(
        stats_df      = stats,
        category_dict = category_dict,
        label_map     = COLUMN_LABELS,
        correction    = "fdr",
    )
    print_grouped_table(grouped, label_map=COLUMN_LABELS, correction="fdr")

    # ── Excel ────────────────────────────────────────────────────────────
    excel_path = os.path.join(SCRIPT_DIR, "reports", f"statistics_univariate_{timepoint}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        stats.to_excel(writer,   sheet_name="statistics_univariate",         index=False)
        grouped.to_excel(writer, sheet_name="statistics_univariate_grouped",  index=False)
    print(f"Saved → {excel_path}")

    # ── Manhattan plot ────────────────────────────────────────────────────
    plot_path = os.path.join(SCRIPT_DIR, "reports", f"statistics_manhattan_{timepoint}.png")
    manhattan_plot(
        stats_df      = grouped,
        category_dict = category_dict,
        label_map     = COLUMN_LABELS,
        pval_col      = "pval",
        alpha         = 0.05,
        title         = f"Univariate associations — {timepoint} ",
        save_path     = plot_path,
        top_n_labels  = 15,
    )

    return stats, grouped


# ─────────────────────────────────────────────────────────────────────────────
# M00
# ─────────────────────────────────────────────────────────────────────────────
run_univariate(
    data_path     = os.path.join(SCRIPT_DIR, "data",
                        "Rlink_version3_type_Clinic_timepoint_M00_v20260602.csv"),
    category_dict = M00_Variables,
    timepoint     = "M00",
    drop_cols     = config["drop"],
    group_labels  = None,
)

# ─────────────────────────────────────────────────────────────────────────────
# M03
# ─────────────────────────────────────────────────────────────────────────────
run_univariate(
    data_path     = os.path.join(SCRIPT_DIR, "data",
                        "Rlink_version3_type_Clinic_timepoint_M03_v20260602.csv"),
    category_dict = M03_variables,
    timepoint     = "M03",
    drop_cols     = config["drop"],
    extra_drop    = config.get("residualization", []),
    cattest       = "prop_ztest",
    group_labels  = {0: "GR", 1: "No_GR"},
)


