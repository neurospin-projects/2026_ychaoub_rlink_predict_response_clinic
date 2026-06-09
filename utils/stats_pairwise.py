import itertools
import os

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


# ─────────────────────────────────────────────────────────────────────────────
# Type map — built at import time from the two variable CSV files
# ─────────────────────────────────────────────────────────────────────────────

def _build_type_map() -> dict:
    """
    Read Rlink_Clinical_variables_M00_mapping.csv and Rlink_Clinical_variables_M03_mapping_delta.csv and return a {variable_name: type} dict.
    Both the original code AND its label are registered so lookups work
    regardless of which name appears in the DataFrame.
    """
    _here    = os.path.dirname(os.path.abspath(__file__))
    _csv_files = [
        os.path.join(_here, "../data/Rlink_Clinical_variables_M00_mapping.csv"),
        os.path.join(_here, "../data/Rlink_Clinical_variables_M03_mapping.csv"),
    ]
    type_map = {}
    for path in _csv_files:
        if not os.path.exists(path):
            import warnings
            warnings.warn(f"Variable CSV not found: {path}")
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            vtype = row["type"].strip()
            if not vtype:
                continue
            code = row["variable"].strip()
            if code:
                type_map[code] = vtype
            label = row["label"].strip()
            if label:
                type_map[label] = vtype
    return type_map

# Loaded once when the module is imported
TYPE_MAP: dict = _build_type_map()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_type(col_name, s):
    if col_name in TYPE_MAP:
        return TYPE_MAP[col_name]

    nunique = s.dropna().nunique()

    if nunique == 2:
        return "binary"

    
def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2)
    )
    if pooled_std == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_std


def _fmt_pval(p: float) -> str:
    if pd.isna(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_quant(s: pd.Series) -> str:
    g = s.dropna()
    if len(g) == 0:
        return "—"
    return f"{g.mean():.3f} ± {g.std():.3f} (N={len(g)})"


def _fmt_cat(s: pd.Series) -> str:
    s = s.dropna()
    total = len(s)
    if total == 0:
        return "—"
    unique_vals = sorted(s.unique())
    if len(unique_vals) == 2:
        pos_val = unique_vals[-1]
        cnt = (s == pos_val).sum()
        return f"{cnt}/{total} ({100*cnt/total:.1f}%)"
    parts = []
    for val, cnt in s.value_counts().sort_index().items():
        parts.append(f"{val}: {cnt} ({100*cnt/total:.1f}%)")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main pairwise function
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_stats(
    data: pd.DataFrame,
    vars1: list,
    vars2: list,
    cattest: str = "chi2",
    group_labels: dict = None,
) -> pd.DataFrame:
    """
    Pairwise statistical associations for each (v1, v2) in vars1 x vars2.

    Variable types are resolved from `type_map` (built from your CSV files
    via load_type_map()), with an automatic heuristic as fallback.

    Test selection:
      quant x quant         -> Pearson r
      quant x binary        -> Welch t  + Cohen's d
      quant x multicategory -> one-way ANOVA F
      categ x categ         -> chi2  (or prop_ztest when both binary)

    Parameters
    ----------
    data           : pd.DataFrame
    vars1          : grouping variable(s), e.g. ['response']
    vars2          : feature columns to test
    type_map       : {col_name: type_str} from load_type_map(); if None,
                     falls back to automatic detection
    cattest        : 'chi2' or 'prop_ztest'
    max_cat_unique : used only for the heuristic fallback
    group_labels   : dict to rename group levels, e.g. {0: 'GR', 1: 'No_GR'}

    Returns
    -------
    pd.DataFrame with columns:
        v1, v2, type_v1, type_v2, test, stat, dof,
        N_total, N_gr, N_nogr,
        desc_gr, desc_nogr,
        pval, pval_fmt,
        pval_bonf, pval_bonf_fmt,
        pval_fdr,  pval_fdr_fmt,
        cohens_d
    """
    rows = []

    for v1, v2 in itertools.product(vars1, vars2):
        if v1 == v2:
            continue

        df_ = data[[v1, v2]].dropna()
        n_total = len(df_)
        if n_total < 4:
            continue

        s1, s2 = df_[v1].copy(), df_[v2].copy()

        if group_labels:
            s1 = s1.map(lambda x: group_labels.get(x, x))

        t1 = _resolve_type(v1, s1)
        t2 = _resolve_type(v2, s2)


        if "constant" in (t1, t2):
            continue

        row = {
            "v1":       v1,
            "v2":       v2,
            "type_v1":  t1,
            "type_v2":  t2,
            "N_total":  n_total,
            "cohens_d": np.nan,
        }

        # ── Both quantitative -> Pearson r ───────────────────────────────
        if t1 == "quantitative" and t2 == "quantitative":
            r, pval = scipy.stats.pearsonr(s1, s2)
            row.update(
                test="pearson_r",
                stat=r,
                dof=n_total - 2,
                pval=pval,
                N_gr=n_total,
                N_nogr=np.nan,
                desc_gr=f"r={r:.3f}",
                desc_nogr="—",
            )

        # ── Quantitative x binary -> Welch t + Cohen's d ─────────────────
        elif {t1, t2} == {"quantitative", "binary"}:
            quant_s = s1 if t1 == "quantitative" else s2
            cat_s   = s2 if t1 == "quantitative" else s1
            cat_levels = sorted(cat_s.unique())
            g0 = quant_s[cat_s == cat_levels[0]].dropna()
            g1 = quant_s[cat_s == cat_levels[1]].dropna()
            if len(g0) < 2 or len(g1) < 2:
                continue

            t_res = scipy.stats.ttest_ind(g0, g1, equal_var=False)
            d     = _cohens_d(g0, g1)
            row.update(
                test="welch_t",
                stat=t_res.statistic,
                dof=round(t_res.df, 1),
                pval=t_res.pvalue,
                cohens_d=d,
                N_gr=len(g0),
                N_nogr=len(g1),
                desc_gr=_fmt_quant(g0),
                desc_nogr=_fmt_quant(g1),
            )

        # ── Quantitative x multicategory -> one-way ANOVA ─────────────────
        elif {t1, t2} >= {"quantitative"} and "multicategory" in (t1, t2):
            quant_s = s1 if t1 == "quantitative" else s2
            cat_s   = s2 if t1 == "quantitative" else s1
            groups  = [
                quant_s[cat_s == lev].dropna().values
                for lev in sorted(cat_s.unique())
                if (cat_s == lev).sum() >= 2
            ]
            if len(groups) < 2:
                continue

            f_stat, pval = scipy.stats.f_oneway(*groups)
            k = len(groups)
            n = sum(len(g) for g in groups)
            row.update(
                test="anova_F",
                stat=f_stat,
                dof=f"({k-1},{n-k})",
                pval=pval,
                N_gr=n_total,
                N_nogr=np.nan,
                desc_gr=_fmt_quant(quant_s[cat_s == sorted(cat_s.unique())[0]]),
                desc_nogr=_fmt_quant(quant_s[cat_s == sorted(cat_s.unique())[1]]),
            )

        # ── Both categorical -> chi2 / prop_ztest ────────────────────────
        else:
            crosstab = pd.crosstab(s1, s2)
            if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
                continue

            chi2_stat, pval, dof, _ = scipy.stats.chi2_contingency(crosstab)

            if cattest == "prop_ztest" and t1 == "binary" and t2 == "binary":
                levels_s1   = sorted(s1.unique())
                outcome_lev = sorted(s2.unique())[-1]
                counts = [
                    crosstab.at[lev, outcome_lev] if outcome_lev in crosstab.columns else 0
                    for lev in levels_s1
                ]
                nobs = [crosstab.loc[lev].sum() for lev in levels_s1]
                z_stat, pval = proportions_ztest(
                    count=counts, nobs=nobs, value=None, alternative="two-sided"
                )
                row.update(test="prop_ztest", stat=z_stat, dof=1, pval=pval)
            else:
                row.update(test="chi2", stat=chi2_stat, dof=dof, pval=pval)

            g0_mask = s1 == sorted(s1.unique())[0]
            g1_mask = s1 == sorted(s1.unique())[1] if len(s1.unique()) > 1 else ~g0_mask
            row.update(
                N_gr=g0_mask.sum(),
                N_nogr=g1_mask.sum(),
                desc_gr=_fmt_cat(s2[g0_mask]),
                desc_nogr=_fmt_cat(s2[g1_mask]),
            )

        rows.append(row)

    # ── Assemble ──────────────────────────────────────────────────────────
    cols = ["v1", "v2", "type_v1", "type_v2", "test", "stat", "dof",
            "N_total", "N_gr", "N_nogr", "desc_gr", "desc_nogr",
            "pval", "cohens_d"]
    df = (
        pd.DataFrame(rows, columns=cols)
        .sort_values("pval")
        .reset_index(drop=True)
    )

    df["pval_fmt"] = df["pval"].apply(_fmt_pval)

    if len(df) > 0:
        _, df["pval_bonf"], _, _ = multipletests(df["pval"], method="bonferroni")
        _, df["pval_fdr"],  _, _ = multipletests(df["pval"], method="fdr_bh")
        df["pval_bonf_fmt"] = df["pval_bonf"].apply(_fmt_pval)
        df["pval_fdr_fmt"]  = df["pval_fdr"].apply(_fmt_pval)

    final_cols = [
        "v1", "v2", "type_v1", "type_v2", "test", "stat", "dof",
        "N_total", "N_gr", "N_nogr",
        "desc_gr", "desc_nogr",
        "pval", "pval_fmt",
        "pval_bonf", "pval_bonf_fmt",
        "pval_fdr",  "pval_fdr_fmt",
        "cohens_d",
    ]
    return df[final_cols]


# ─────────────────────────────────────────────────────────────────────────────
# Publication-ready printer
# ─────────────────────────────────────────────────────────────────────────────

def print_publication_table(
    stats_df: pd.DataFrame,
    var_col: str = "v2",
    label_gr: str = "GR",
    label_nogr: str = "No_GR",
    alpha: float = 0.05,
    correction: str = "fdr",
) -> None:
    pval_col = {"none": "pval", "bonf": "pval_bonf", "fdr": "pval_fdr"}.get(
        correction, "pval_fdr"
    )
    fmt_col = pval_col + "_fmt"

    cw = {
        "Variable":   36,
        "N":           8,
        label_gr:     36,
        label_nogr:   36,
        "Test":       11,
        "Stat":        8,
        "p-value":     9,
        "p-adj":       9,
        "Cohen's d":  10,
        "Sig.":        5,
    }
    sep    = "─" * sum(cw.values())
    header = "".join(k.ljust(v) for k, v in cw.items())

    print("\n" + sep)
    print(header)
    print(sep)

    for _, r in stats_df.iterrows():
        p_adj    = r.get(pval_col, np.nan)
        sig_str  = "*" if (not pd.isna(p_adj) and p_adj < alpha) else ""
        d_str    = f"{r['cohens_d']:.2f}" if not pd.isna(r.get("cohens_d")) else "—"
        stat_str = f"{r['stat']:.3f}"     if not pd.isna(r.get("stat"))     else "—"
        desc_gr   = str(r.get("desc_gr",   ""))[:cw[label_gr]   - 1]
        desc_nogr = str(r.get("desc_nogr", ""))[:cw[label_nogr] - 1]

        row_vals = [
            str(r[var_col])[:cw["Variable"] - 1],
            f"N={int(r['N_total'])}",
            desc_gr, desc_nogr,
            str(r.get("test", "")),
            stat_str,
            str(r.get("pval_fmt", "")),
            str(r.get(fmt_col,    "")),
            d_str, sig_str,
        ]
        print("".join(str(v).ljust(w) for v, w in zip(row_vals, cw.values())))

    print(sep)
    correction_label = correction.upper() if correction != "none" else "None"
    print(f"* p_{correction} < {alpha}  |  Correction: {correction_label}")


# ─────────────────────────────────────────────────────────────────────────────
# Grouped stats: by category then sorted by p-value within each group
# ─────────────────────────────────────────────────────────────────────────────

def grouped_stats_table(
    stats_df: pd.DataFrame,
    category_dict: dict,
    label_map: dict = None,
    var_col: str = "v2",
    correction: str = "fdr",
) -> pd.DataFrame:
    """
    Re-organise pairwise_stats() output by category then sort by p-value
    within each category.

    Parameters
    ----------
    stats_df       : output of pairwise_stats()
    category_dict  : {category_name: [var1, var2, ...]}
                     Variables can be original codes or labels.
    label_map      : COLUMN_LABELS dict {code: label} for bidirectional matching
    var_col        : column in stats_df holding the variable name (default 'v2')
    correction     : 'none' | 'bonf' | 'fdr'

    Returns
    -------
    pd.DataFrame with an extra 'category' column, sorted by
    (category_order, pval_adj). Variables with no category go to "Other".
    """
    pval_col = {"none": "pval", "bonf": "pval_bonf", "fdr": "pval_fdr"}.get(
        correction, "pval_fdr"
    )

    code_to_label = dict(label_map) if label_map else {}
    label_to_code = {v: k for k, v in code_to_label.items()}

    def _aliases(name):
        candidates = {name}
        if name in code_to_label:
            candidates.add(code_to_label[name])
        if name in label_to_code:
            candidates.add(label_to_code[name])
        return candidates

    var_to_cat   = {}
    var_to_order = {}
    for cat_name, var_list in category_dict.items():
        for rank, var in enumerate(var_list):
            for alias in _aliases(var):
                var_to_cat[alias]   = cat_name
                var_to_order[alias] = rank

    cat_order_list = list(category_dict.keys())

    def _get_cat(v):
        for alias in _aliases(v):
            if alias in var_to_cat:
                return var_to_cat[alias]
        return "Other"

    def _cat_rank(cat):
        return cat_order_list.index(cat) if cat in cat_order_list else len(cat_order_list)

    df = stats_df.copy()
    df["category"] = df[var_col].apply(_get_cat)
    df["_cat_rank"] = df["category"].apply(_cat_rank)

    df = (
        df
        .sort_values(["_cat_rank", pval_col], ascending=[True, True])
        .drop(columns=["_cat_rank"])
        .reset_index(drop=True)
    )

    cols = ["category"] + [c for c in df.columns if c != "category"]
    return df[cols]


def print_grouped_table(
    grouped_df: pd.DataFrame,
    var_col: str = "v2",
    label_gr: str = "GR",
    label_nogr: str = "No_GR",
    alpha: float = 0.05,
    correction: str = "fdr",
    label_map: dict = None,
) -> None:
    """
    Print grouped_stats_table() output with category headers,
    rows sorted by p-value within each group.
    """
    pval_col = {"none": "pval", "bonf": "pval_bonf", "fdr": "pval_fdr"}.get(
        correction, "pval_fdr"
    )
    fmt_col       = pval_col + "_fmt"
    code_to_label = dict(label_map) if label_map else {}

    cw = {
        "Variable":   36,
        "N":           8,
        label_gr:     36,
        label_nogr:   36,
        "Test":       11,
        "Stat":        8,
        "p-value":     9,
        "p-adj":       9,
        "Cohen's d":  10,
        "Sig.":        5,
    }
    total_width = sum(cw.values())
    sep       = "─" * total_width
    thick_sep = "═" * total_width
    header    = "".join(k.ljust(v) for k, v in cw.items())

    print(f"\n{thick_sep}")
    print(header)
    print(thick_sep)

    current_cat = None
    for _, r in grouped_df.iterrows():
        cat = r.get("category", "")
        if cat != current_cat:
            if current_cat is not None:
                print(sep)
            print(f"\n  ▶  {cat}")
            print(sep)
            current_cat = cat

        raw_var   = str(r[var_col])
        disp_var  = code_to_label.get(raw_var, raw_var)
        p_adj     = r.get(pval_col, np.nan)
        sig_str   = "*" if (not pd.isna(p_adj) and p_adj < alpha) else ""
        d_str     = f"{r['cohens_d']:.2f}" if not pd.isna(r.get("cohens_d")) else "—"
        stat_str  = f"{r['stat']:.3f}"     if not pd.isna(r.get("stat"))     else "—"
        desc_gr   = str(r.get("desc_gr",   ""))[:cw[label_gr]   - 1]
        desc_nogr = str(r.get("desc_nogr", ""))[:cw[label_nogr] - 1]

        row_vals = [
            disp_var[:cw["Variable"] - 1],
            f"N={int(r['N_total'])}",
            desc_gr, desc_nogr,
            str(r.get("test", "")),
            stat_str,
            str(r.get("pval_fmt", "")),
            str(r.get(fmt_col,    "")),
            d_str, sig_str,
        ]
        print("".join(str(v).ljust(w) for v, w in zip(row_vals, cw.values())))

    print(thick_sep)
    correction_label = correction.upper() if correction != "none" else "None"
    print(f"* p_{correction} < {alpha}  |  Correction: {correction_label}")

