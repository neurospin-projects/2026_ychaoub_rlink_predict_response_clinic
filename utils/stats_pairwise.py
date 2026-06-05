import itertools
 
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _var_type(s: pd.Series, max_cat_unique: int = 10) -> str:
    n = s.nunique(dropna=True)
    if n <= 1:
        return "constant"
    if n == 2:
        return "binary"
    if (s.dtype.kind == "f") or (s.dtype.kind in ("i", "u") and n > max_cat_unique):
        return "quantitative"
    return "multicategory"
 
 
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
    """mean±SD (N=n)"""
    g = s.dropna()
    if len(g) == 0:
        return "—"
    return f"{g.mean():.3f} ± {g.std():.3f} (N={len(g)})"
 
 
def _fmt_cat(s: pd.Series) -> str:
    """
    For binary: show only the positive level -> "n/total (X%)"
    For multi:  "val: n (%) | val: n (%)"
    """
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
# Main function
# ─────────────────────────────────────────────────────────────────────────────
 
def pairwise_stats(
    data: pd.DataFrame,
    vars1: list,
    vars2: list,
    cattest: str = "chi2",
    max_cat_unique: int = 10,
    group_labels: dict = None,
) -> pd.DataFrame:
    """
    Pairwise statistical associations for each (v1, v2) in vars1 x vars2.
 
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
    cattest        : 'chi2' or 'prop_ztest'
    max_cat_unique : integer columns with <= this many unique values -> categorical
    group_labels   : dict to rename group levels, e.g. {0: 'GR', 1: 'No_GR'}
 
    Returns
    -------
    pd.DataFrame with columns:
        v1, v2, test, stat, dof,
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
 
        # Apply group labels if provided
        if group_labels:
            s1 = s1.map(lambda x: group_labels.get(x, x))
 
        t1 = _var_type(s1, max_cat_unique)
        t2 = _var_type(s2, max_cat_unique)
 
        if "constant" in (t1, t2):
            continue
 
        # Identify the two group levels for per-group descriptives
        levels = sorted(s1.unique()) if t1 == "binary" else sorted(s2.unique())
        lv0 = levels[0] if len(levels) > 0 else None
        lv1 = levels[1] if len(levels) > 1 else None
 
        row = {
            "v1":      v1,
            "v2":      v2,
            "N_total": n_total,
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
            d = _cohens_d(g0, g1)
 
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
        elif (t1 == "quantitative" and t2 == "multicategory") or \
             (t1 == "multicategory" and t2 == "quantitative"):
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
 
            # Per-group descriptive for categorical
            g0_mask = s1 == sorted(s1.unique())[0]
            g1_mask = s1 == sorted(s1.unique())[1]
            row.update(
                N_gr=g0_mask.sum(),
                N_nogr=g1_mask.sum(),
                desc_gr=_fmt_cat(s2[g0_mask]),
                desc_nogr=_fmt_cat(s2[g1_mask]),
            )
 
        rows.append(row)
 
    # ── Assemble ──────────────────────────────────────────────────────────
    cols = ["v1", "v2", "test", "stat", "dof",
            "N_total", "N_gr", "N_nogr",
            "desc_gr", "desc_nogr",
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
        "v1", "v2", "test", "stat", "dof",
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
    """
    Print a publication-ready aligned table from pairwise_stats() output.
 
    Columns:
        Variable | N_total | GR (stats) | No_GR (stats) | Test | Stat | p-value | p-adj | Cohen's d | Sig.
 
    Parameters
    ----------
    stats_df    : output of pairwise_stats()
    var_col     : column with feature names (default 'v2')
    label_gr    : display name for group 0 (default 'GR')
    label_nogr  : display name for group 1 (default 'No_GR')
    alpha       : significance threshold (default 0.05)
    correction  : 'none' | 'bonf' | 'fdr'
    """
    pval_col = {"none": "pval", "bonf": "pval_bonf", "fdr": "pval_fdr"}.get(
        correction, "pval_fdr"
    )
    fmt_col = pval_col + "_fmt"
 
    # Fixed column widths — tune these if needed
    cw = {
        "Variable":           36,
        "N":                   8,
        label_gr:             36,
        label_nogr:           36,
        "Test":               11,
        "Stat":                8,
        "p-value":             9,
        "p-adj":               9,
        "Cohen's d":          10,
        "Sig.":                5,
    }
 
    sep       = "─" * sum(cw.values())
    header    = "".join(k.ljust(v) for k, v in cw.items())
 
    print("\n" + sep)
    print(header)
    print(sep)
 
    for _, r in stats_df.iterrows():
        p_adj    = r.get(pval_col, np.nan)
        sig_str  = "*" if (not pd.isna(p_adj) and p_adj < alpha) else ""
        d_str    = f"{r['cohens_d']:.2f}" if not pd.isna(r.get("cohens_d")) else "—"
        stat_str = f"{r['stat']:.3f}"     if not pd.isna(r.get("stat"))     else "—"
 
        # Truncate descriptive columns to fit cleanly
        desc_gr    = str(r.get("desc_gr",    ""))[:cw[label_gr]   - 1]
        desc_nogr  = str(r.get("desc_nogr",  ""))[:cw[label_nogr] - 1]
 
        row_vals = [
            str(r[var_col])[:cw["Variable"] - 1],
            f"N={int(r['N_total'])}",
            desc_gr,
            desc_nogr,
            str(r.get("test", "")),
            stat_str,
            str(r.get("pval_fmt",  "")),
            str(r.get(fmt_col,     "")),
            d_str,
            sig_str,
        ]
        print("".join(str(v).ljust(w) for v, w in zip(row_vals, cw.values())))
 
    print(sep)
    correction_label = correction.upper() if correction != "none" else "None"
    print(f"* p_{correction} < {alpha}  |  Correction: {correction_label}")