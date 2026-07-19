from __future__ import annotations

import math
import warnings
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller


VIDEO_METRICS = [
    "view_count",
    "like_count",
    "comment_count",
    "views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "engagement_per_1000_views",
    "duration_minutes",
    "age_days",
    "channel_subscriber_count",
]

OUTCOMES = [
    "views_per_day",
    "engagement_per_1000_views",
    "view_count",
    "comment_count",
]

GROUP_COLUMNS = ["query_group", "topic_primary", "duration_band"]

METHOD_CATEGORIES = [
    "가정·데이터 진단",
    "두 집단 비교",
    "다집단·사후검정",
    "상관분석",
    "범주형 분석",
    "회귀·일반화선형모형",
    "시계열 추세",
    "차원축소·군집",
    "이상치·검정력",
]


def _empty(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _adjust_pvalues(frame: pd.DataFrame, p_column: str = "p_value") -> pd.DataFrame:
    result = frame.copy()
    result["p_holm"] = np.nan
    result["q_bh"] = np.nan
    if result.empty or p_column not in result:
        return result
    valid = pd.to_numeric(result[p_column], errors="coerce").notna()
    if valid.any():
        values = result.loc[valid, p_column].astype(float).clip(0, 1).to_numpy()
        result.loc[valid, "p_holm"] = multipletests(values, method="holm")[1]
        result.loc[valid, "q_bh"] = multipletests(values, method="fdr_bh")[1]
    return result


def _normality_interpretation(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "판정 불가"
    return "정규성 기각" if p_value < 0.05 else "정규성 기각 못함"


def normality_tests(videos: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for column in VIDEO_METRICS:
        if column not in videos:
            continue
        values = pd.to_numeric(videos[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(values) < 3 or values.nunique() < 2:
            continue
        sample = values.to_numpy(dtype=float)
        if len(sample) > 5000:
            sample = rng.choice(sample, 5000, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shapiro_stat, shapiro_p = stats.shapiro(sample)
        rows.append(
            {
                "variable": column,
                "test": "Shapiro-Wilk",
                "n_total": len(values),
                "n_tested": len(sample),
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "skewness": stats.skew(sample, bias=False),
                "excess_kurtosis": stats.kurtosis(sample, fisher=True, bias=False),
                "interpretation_0.05": _normality_interpretation(shapiro_p),
            }
        )
        if len(sample) >= 8:
            k2_stat, k2_p = stats.normaltest(sample)
            rows.append(
                {
                    "variable": column,
                    "test": "D'Agostino K²",
                    "n_total": len(values),
                    "n_tested": len(sample),
                    "statistic": k2_stat,
                    "p_value": k2_p,
                    "skewness": stats.skew(sample, bias=False),
                    "excess_kurtosis": stats.kurtosis(sample, fisher=True, bias=False),
                    "interpretation_0.05": _normality_interpretation(k2_p),
                }
            )
    return _adjust_pvalues(pd.DataFrame(rows))


def variance_homogeneity_tests(videos: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_column in GROUP_COLUMNS:
        if group_column not in videos:
            continue
        for outcome in OUTCOMES:
            if outcome not in videos:
                continue
            groups = []
            labels = []
            for label, subset in videos.groupby(group_column, dropna=True):
                values = pd.to_numeric(subset[outcome], errors="coerce").dropna()
                if len(values) >= 2:
                    groups.append(values.to_numpy(dtype=float))
                    labels.append(str(label))
            if len(groups) < 2:
                continue
            statistic, p_value = stats.levene(*groups, center="median")
            rows.append(
                {
                    "group_variable": group_column,
                    "outcome": outcome,
                    "test": "Brown-Forsythe (median-centered Levene)",
                    "groups": len(groups),
                    "n": sum(len(group) for group in groups),
                    "statistic": statistic,
                    "p_value": p_value,
                    "interpretation_0.05": "등분산성 기각"
                    if p_value < 0.05
                    else "등분산성 기각 못함",
                    "included_groups": "|".join(labels),
                }
            )
    return _adjust_pvalues(pd.DataFrame(rows))


def _cohen_effects(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    pooled_variance = (
        (n1 - 1) * np.var(first, ddof=1) + (n2 - 1) * np.var(second, ddof=1)
    ) / (n1 + n2 - 2)
    if pooled_variance <= 0:
        return np.nan, np.nan
    cohen_d = (np.mean(first) - np.mean(second)) / math.sqrt(pooled_variance)
    correction = 1 - 3 / (4 * (n1 + n2) - 9) if n1 + n2 > 2 else np.nan
    return float(cohen_d), float(cohen_d * correction)


def two_group_tests(videos: pd.DataFrame) -> pd.DataFrame:
    working = videos.copy()
    if "title_has_korean" in working:
        working["title_language_group"] = np.where(
            working["title_has_korean"].astype(bool), "한글 포함", "한글 미포함"
        )
    group_candidates = ["title_language_group"]
    if "query_group" in working:
        single_query = working[working["query_group"] != "복수 키워드"].copy()
    else:
        single_query = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    datasets = [("title_language_group", working)]
    if not single_query.empty and single_query["query_group"].nunique() == 2:
        datasets.append(("query_group", single_query))

    for group_column, data in datasets:
        if group_column not in data or data[group_column].nunique(dropna=True) != 2:
            continue
        labels = [str(value) for value in data[group_column].dropna().unique()]
        for outcome in OUTCOMES:
            if outcome not in data:
                continue
            first = pd.to_numeric(
                data.loc[data[group_column].astype(str) == labels[0], outcome],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            second = pd.to_numeric(
                data.loc[data[group_column].astype(str) == labels[1], outcome],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            if len(first) < 2 or len(second) < 2:
                continue
            cohen_d, hedges_g = _cohen_effects(first, second)
            welch = stats.ttest_ind(first, second, equal_var=False, nan_policy="omit")
            student = stats.ttest_ind(first, second, equal_var=True, nan_policy="omit")
            mann = stats.mannwhitneyu(first, second, alternative="two-sided")
            rank_biserial = 2 * mann.statistic / (len(first) * len(second)) - 1
            common = {
                "group_variable": group_column,
                "group_1": labels[0],
                "group_2": labels[1],
                "outcome": outcome,
                "n_1": len(first),
                "n_2": len(second),
                "mean_1": np.mean(first),
                "mean_2": np.mean(second),
                "median_1": np.median(first),
                "median_2": np.median(second),
                "cohen_d": cohen_d,
                "hedges_g": hedges_g,
                "rank_biserial": rank_biserial,
            }
            rows.extend(
                [
                    {
                        **common,
                        "test": "Welch t-test",
                        "statistic": welch.statistic,
                        "p_value": welch.pvalue,
                    },
                    {
                        **common,
                        "test": "Student t-test",
                        "statistic": student.statistic,
                        "p_value": student.pvalue,
                    },
                    {
                        **common,
                        "test": "Mann-Whitney U",
                        "statistic": mann.statistic,
                        "p_value": mann.pvalue,
                    },
                ]
            )
    return _adjust_pvalues(pd.DataFrame(rows))


def _eta_squared(groups: list[np.ndarray]) -> float:
    combined = np.concatenate(groups)
    grand_mean = np.mean(combined)
    total = np.sum((combined - grand_mean) ** 2)
    between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
    return float(between / total) if total > 0 else np.nan


def multi_group_tests(videos: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_column in GROUP_COLUMNS:
        if group_column not in videos:
            continue
        for outcome in OUTCOMES:
            if outcome not in videos:
                continue
            groups = []
            labels = []
            for label, subset in videos.groupby(group_column, dropna=True):
                values = pd.to_numeric(subset[outcome], errors="coerce").dropna()
                if len(values) >= 3:
                    groups.append(values.to_numpy(dtype=float))
                    labels.append(str(label))
            if len(groups) < 3:
                continue
            n_total = sum(len(group) for group in groups)
            anova = stats.f_oneway(*groups)
            welch = anova_oneway(groups, use_var="unequal", welch_correction=True)
            kruskal = stats.kruskal(*groups)
            eta_squared = _eta_squared(groups)
            epsilon_squared = max(
                0.0,
                float((kruskal.statistic - len(groups) + 1) / (n_total - len(groups))),
            )
            common = {
                "group_variable": group_column,
                "outcome": outcome,
                "groups": len(groups),
                "n": n_total,
                "included_groups": "|".join(labels),
                "eta_squared": eta_squared,
                "epsilon_squared": epsilon_squared,
            }
            rows.extend(
                [
                    {
                        **common,
                        "test": "One-way ANOVA",
                        "statistic": anova.statistic,
                        "p_value": anova.pvalue,
                    },
                    {
                        **common,
                        "test": "Welch ANOVA",
                        "statistic": welch.statistic,
                        "p_value": welch.pvalue,
                    },
                    {
                        **common,
                        "test": "Kruskal-Wallis",
                        "statistic": kruskal.statistic,
                        "p_value": kruskal.pvalue,
                    },
                ]
            )
    return _adjust_pvalues(pd.DataFrame(rows))


def posthoc_tests(videos: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_column in GROUP_COLUMNS:
        if group_column not in videos:
            continue
        for outcome in OUTCOMES[:2]:
            data = videos[[group_column, outcome]].copy()
            data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
            data = data.dropna()
            valid_labels = [
                label
                for label, count in data[group_column].value_counts().items()
                if count >= 3
            ]
            data = data[data[group_column].isin(valid_labels)]
            if data[group_column].nunique() < 3 or data[group_column].nunique() > 15:
                continue
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)

            try:
                tukey = pairwise_tukeyhsd(
                    endog=data[outcome].astype(float),
                    groups=data[group_column].astype(str),
                    alpha=0.05,
                )
                table = tukey._results_table.data
                headers = table[0]
                for values in table[1:]:
                    record = dict(zip(headers, values))
                    rows.append(
                        {
                            "group_variable": group_column,
                            "outcome": outcome,
                            "method": "Tukey HSD",
                            "group_1": record.get("group1"),
                            "group_2": record.get("group2"),
                            "effect": record.get("meandiff"),
                            "statistic": np.nan,
                            "p_value": record.get("p-adj"),
                            "ci_low": record.get("lower"),
                            "ci_high": record.get("upper"),
                            "significant_0.05": record.get("reject"),
                        }
                    )
            except (ValueError, TypeError):
                pass

            mann_rows: list[dict[str, Any]] = []
            for first_label, second_label in combinations(valid_labels, 2):
                first = data.loc[data[group_column] == first_label, outcome].to_numpy(dtype=float)
                second = data.loc[data[group_column] == second_label, outcome].to_numpy(dtype=float)
                result = stats.mannwhitneyu(first, second, alternative="two-sided")
                rank_biserial = 2 * result.statistic / (len(first) * len(second)) - 1
                mann_rows.append(
                    {
                        "group_variable": group_column,
                        "outcome": outcome,
                        "method": "Pairwise Mann-Whitney",
                        "group_1": str(first_label),
                        "group_2": str(second_label),
                        "effect": rank_biserial,
                        "statistic": result.statistic,
                        "p_value": result.pvalue,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }
                )
            if mann_rows:
                p_values = [row["p_value"] for row in mann_rows]
                adjusted = multipletests(p_values, method="holm")[1]
                for row, p_adjusted in zip(mann_rows, adjusted):
                    row["p_holm_within_family"] = p_adjusted
                    row["significant_0.05"] = bool(p_adjusted < 0.05)
                    rows.append(row)
    return _adjust_pvalues(pd.DataFrame(rows))


def full_correlations(videos: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in VIDEO_METRICS if column in videos]
    rows: list[dict[str, Any]] = []
    for first_column, second_column in combinations(available, 2):
        pair = videos[[first_column, second_column]].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 5 or pair[first_column].nunique() < 2 or pair[second_column].nunique() < 2:
            continue
        x = pair[first_column].to_numpy(dtype=float)
        y = pair[second_column].to_numpy(dtype=float)
        results = [
            ("Pearson", stats.pearsonr(x, y)),
            ("Spearman", stats.spearmanr(x, y)),
            ("Kendall tau-b", stats.kendalltau(x, y)),
        ]
        for method, result in results:
            rows.append(
                {
                    "variable_1": first_column,
                    "variable_2": second_column,
                    "method": method,
                    "n": len(pair),
                    "coefficient": result.statistic,
                    "p_value": result.pvalue,
                }
            )
    return _adjust_pvalues(pd.DataFrame(rows))


def _cramers_v(table: pd.DataFrame, chi2: float) -> float:
    observed = table.to_numpy()
    n = observed.sum()
    if n <= 1:
        return np.nan
    rows, columns = observed.shape
    phi2 = chi2 / n
    correction = ((columns - 1) * (rows - 1)) / (n - 1)
    phi2_corrected = max(0.0, phi2 - correction)
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    return math.sqrt(phi2_corrected / denominator) if denominator > 0 else np.nan


def contingency_tests(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    candidates: list[tuple[str, str, pd.DataFrame]] = []
    for first, second in [
        ("query_group", "topic_primary"),
        ("query_group", "duration_band"),
        ("title_has_korean", "topic_primary"),
    ]:
        if first in videos and second in videos:
            candidates.append((first, second, videos))
    if not comments.empty and {"topic_primary", "sentiment"}.issubset(comments.columns):
        candidates.append(("topic_primary", "sentiment", comments))

    rows: list[dict[str, Any]] = []
    for first, second, data in candidates:
        table = pd.crosstab(data[first], data[second])
        table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_value, dof, expected = stats.chi2_contingency(table)
        rows.append(
            {
                "variable_1": first,
                "variable_2": second,
                "test": "Chi-square independence",
                "rows": table.shape[0],
                "columns": table.shape[1],
                "n": int(table.to_numpy().sum()),
                "statistic": chi2,
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "cramers_v": _cramers_v(table, chi2),
                "expected_below_5_share": float((expected < 5).mean()),
            }
        )
        if table.shape == (2, 2):
            odds_ratio, fisher_p = stats.fisher_exact(table.to_numpy())
            rows.append(
                {
                    "variable_1": first,
                    "variable_2": second,
                    "test": "Fisher exact",
                    "rows": 2,
                    "columns": 2,
                    "n": int(table.to_numpy().sum()),
                    "statistic": odds_ratio,
                    "degrees_of_freedom": np.nan,
                    "p_value": fisher_p,
                    "cramers_v": np.nan,
                    "expected_below_5_share": float((expected < 5).mean()),
                }
            )
    return _adjust_pvalues(pd.DataFrame(rows))


def _regression_design(videos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame(index=videos.index)
    data["log_views"] = np.log1p(pd.to_numeric(videos["view_count"], errors="coerce"))
    data["comment_count"] = pd.to_numeric(videos["comment_count"], errors="coerce")
    data["engagement"] = pd.to_numeric(
        videos["engagement_per_1000_views"], errors="coerce"
    )
    data["log_age_days"] = np.log1p(pd.to_numeric(videos["age_days"], errors="coerce"))
    data["log_subscribers"] = np.log1p(
        pd.to_numeric(videos["channel_subscriber_count"], errors="coerce").fillna(0)
    )
    data["log_duration"] = np.log1p(
        pd.to_numeric(videos["duration_seconds"], errors="coerce").fillna(0)
    )
    data["title_keyword_mentions"] = pd.to_numeric(
        videos["title_keyword_mentions"], errors="coerce"
    ).fillna(0)
    data["title_has_korean"] = videos["title_has_korean"].astype(float)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    predictors = [
        "log_age_days",
        "log_subscribers",
        "log_duration",
        "title_keyword_mentions",
        "title_has_korean",
    ]
    nonconstant = [column for column in predictors if data[column].nunique() > 1]
    x = sm.add_constant(data[nonconstant], has_constant="add")
    return data, x


def _model_rows(model: Any, model_name: str, exponentiate: bool = False) -> list[dict[str, Any]]:
    conf = model.conf_int()
    rows = []
    for parameter in model.params.index:
        coefficient = float(model.params[parameter])
        rows.append(
            {
                "model": model_name,
                "outcome": getattr(model.model, "endog_names", ""),
                "parameter": parameter,
                "coefficient": coefficient,
                "standard_error": float(model.bse[parameter])
                if parameter in model.bse.index
                else np.nan,
                "statistic": float(model.tvalues[parameter])
                if parameter in model.tvalues.index
                else np.nan,
                "p_value": float(model.pvalues[parameter])
                if parameter in model.pvalues.index
                else np.nan,
                "ci_low": float(conf.loc[parameter, 0]),
                "ci_high": float(conf.loc[parameter, 1]),
                "exp_coefficient": math.exp(coefficient) if exponentiate else np.nan,
                "n": int(model.nobs),
                "aic": float(model.aic) if hasattr(model, "aic") else np.nan,
                "bic": float(model.bic) if hasattr(model, "bic") else np.nan,
                "pseudo_or_r_squared": float(model.rsquared)
                if hasattr(model, "rsquared")
                else (
                    float(model.prsquared) if hasattr(model, "prsquared") else np.nan
                ),
            }
        )
    return rows


def regression_suite(videos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, x = _regression_design(videos)
    if len(data) < max(30, x.shape[1] * 5):
        message = pd.DataFrame(
            [
                {
                    "model": "분석 불가",
                    "outcome": "",
                    "parameter": "",
                    "note": "회귀모형에는 예측변수당 약 5개 이상이며 총 30개 이상의 완전한 관측값이 필요합니다.",
                }
            ]
        )
        return message, _empty(["diagnostic", "statistic", "p_value", "interpretation"])

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sm.OLS(data["log_views"], x).fit(cov_type="HC3")
        rows.extend(_model_rows(ols, "OLS log_views (HC3 robust SE)"))

        try:
            quantile = sm.QuantReg(data["log_views"], x).fit(q=0.5, max_iter=2000)
            rows.extend(_model_rows(quantile, "Median quantile regression"))
        except (ValueError, np.linalg.LinAlgError):
            pass

        try:
            robust = sm.RLM(data["log_views"], x, M=sm.robust.norms.HuberT()).fit()
            robust_rows = _model_rows(robust, "Robust linear model (Huber)")
            rows.extend(robust_rows)
        except (ValueError, np.linalg.LinAlgError):
            pass

        count_x = x.assign(log_views=data["log_views"])
        try:
            poisson = sm.GLM(
                data["comment_count"], count_x, family=sm.families.Poisson()
            ).fit(cov_type="HC3")
            rows.extend(_model_rows(poisson, "Poisson GLM comments", exponentiate=True))
            dispersion = float(poisson.pearson_chi2 / poisson.df_resid)
            diagnostics.append(
                {
                    "model": "Poisson GLM comments",
                    "diagnostic": "Pearson dispersion",
                    "statistic": dispersion,
                    "p_value": np.nan,
                    "interpretation": "1보다 매우 크면 과산포로 음이항모형 우선",
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            pass

        try:
            negative_binomial = sm.GLM(
                data["comment_count"],
                count_x,
                family=sm.families.NegativeBinomial(alpha=1.0),
            ).fit(cov_type="HC3")
            rows.extend(
                _model_rows(
                    negative_binomial, "Negative-binomial GLM comments", exponentiate=True
                )
            )
        except (ValueError, np.linalg.LinAlgError):
            pass

        high_engagement = (data["engagement"] >= data["engagement"].median()).astype(int)
        if high_engagement.nunique() == 2:
            try:
                logistic = sm.Logit(high_engagement, x).fit(disp=False, maxiter=200)
                rows.extend(
                    _model_rows(logistic, "Logistic high engagement", exponentiate=True)
                )
            except (ValueError, np.linalg.LinAlgError):
                pass

    residuals = np.asarray(ols.resid, dtype=float)
    if len(residuals) > 5000:
        residuals_test = np.random.default_rng(42).choice(residuals, 5000, replace=False)
    else:
        residuals_test = residuals
    shapiro = stats.shapiro(residuals_test)
    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, ols.model.exog)
    diagnostics.extend(
        [
            {
                "model": "OLS log_views",
                "diagnostic": "Residual Shapiro-Wilk",
                "statistic": shapiro.statistic,
                "p_value": shapiro.pvalue,
                "interpretation": "p<.05이면 잔차 정규성 위배 가능",
            },
            {
                "model": "OLS log_views",
                "diagnostic": "Jarque-Bera",
                "statistic": jb_stat,
                "p_value": jb_p,
                "interpretation": f"잔차 왜도={skew:.3f}, 첨도={kurtosis:.3f}",
            },
            {
                "model": "OLS log_views",
                "diagnostic": "Breusch-Pagan",
                "statistic": bp_stat,
                "p_value": bp_p,
                "interpretation": "p<.05이면 이분산 가능; HC3 표준오차 사용됨",
            },
            {
                "model": "OLS log_views",
                "diagnostic": "Durbin-Watson",
                "statistic": durbin_watson(residuals),
                "p_value": np.nan,
                "interpretation": "2에 가까우면 1차 자기상관이 약함",
            },
        ]
    )
    for index, name in enumerate(x.columns):
        if name == "const":
            continue
        try:
            vif = variance_inflation_factor(x.to_numpy(dtype=float), index)
        except (ValueError, np.linalg.LinAlgError):
            vif = np.nan
        diagnostics.append(
            {
                "model": "OLS log_views",
                "diagnostic": f"VIF: {name}",
                "statistic": vif,
                "p_value": np.nan,
                "interpretation": "5 이상 주의, 10 이상 심각한 다중공선성 가능",
            }
        )
    return _adjust_pvalues(pd.DataFrame(rows)), pd.DataFrame(diagnostics)


def time_series_suite(videos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "publish_month" not in videos:
        return _empty(["outcome", "test", "statistic", "p_value"]), pd.DataFrame()
    monthly = (
        videos.dropna(subset=["publish_month"])
        .groupby("publish_month", as_index=False)
        .agg(
            video_count=("video_id", "nunique"),
            median_views_per_day=("views_per_day", "median"),
            median_engagement=("engagement_per_1000_views", "median"),
        )
        .sort_values("publish_month")
    )
    if monthly.empty:
        return _empty(["outcome", "test", "statistic", "p_value"]), monthly
    for column in ["video_count", "median_views_per_day", "median_engagement"]:
        monthly[f"{column}_rolling_3"] = monthly[column].rolling(3, min_periods=1).mean()
    x = np.arange(len(monthly), dtype=float)
    rows: list[dict[str, Any]] = []
    for outcome in ["video_count", "median_views_per_day", "median_engagement"]:
        values = pd.to_numeric(monthly[outcome], errors="coerce")
        valid = values.notna().to_numpy()
        if valid.sum() < 3:
            continue
        linear = stats.linregress(x[valid], values.to_numpy()[valid])
        kendall = stats.kendalltau(x[valid], values.to_numpy()[valid])
        rows.extend(
            [
                {
                    "outcome": outcome,
                    "test": "Linear time trend",
                    "n_months": int(valid.sum()),
                    "statistic": linear.slope,
                    "p_value": linear.pvalue,
                    "effect_or_slope": linear.slope,
                    "r_squared": linear.rvalue**2,
                },
                {
                    "outcome": outcome,
                    "test": "Mann-Kendall trend (Kendall tau)",
                    "n_months": int(valid.sum()),
                    "statistic": kendall.statistic,
                    "p_value": kendall.pvalue,
                    "effect_or_slope": kendall.statistic,
                    "r_squared": np.nan,
                },
            ]
        )
        if valid.sum() >= 12 and values[valid].nunique() > 1:
            try:
                adf = adfuller(values.to_numpy()[valid], autolag="AIC")
                rows.append(
                    {
                        "outcome": outcome,
                        "test": "Augmented Dickey-Fuller",
                        "n_months": int(valid.sum()),
                        "statistic": adf[0],
                        "p_value": adf[1],
                        "effect_or_slope": np.nan,
                        "r_squared": np.nan,
                    }
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
    return _adjust_pvalues(pd.DataFrame(rows)), monthly


def dimensionality_and_clusters(
    videos: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [
        column
        for column in [
            "view_count",
            "like_count",
            "comment_count",
            "views_per_day",
            "engagement_per_1000_views",
            "duration_minutes",
            "age_days",
            "channel_subscriber_count",
        ]
        if column in videos
    ]
    data = videos[["video_id", *features]].copy()
    for column in features:
        data[column] = np.log1p(pd.to_numeric(data[column], errors="coerce").clip(lower=0))
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 12 or len(features) < 3:
        empty = _empty(["feature", "PC1"])
        return empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    scaled = StandardScaler().fit_transform(data[features])
    component_count = min(5, len(features), len(data) - 1)
    pca = PCA(n_components=component_count, random_state=seed)
    scores = pca.fit_transform(scaled)
    component_names = [f"PC{index + 1}" for index in range(component_count)]
    loadings = pd.DataFrame(pca.components_.T, columns=component_names)
    loadings.insert(0, "feature", features)
    variance = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    score_table = pd.DataFrame(scores, columns=component_names)
    score_table.insert(0, "video_id", data["video_id"].to_numpy())

    best_score = -1.0
    best_labels: np.ndarray | None = None
    best_k = 0
    maximum_k = min(6, len(data) - 1)
    for cluster_count in range(2, maximum_k + 1):
        model = KMeans(n_clusters=cluster_count, random_state=seed, n_init=20)
        labels = model.fit_predict(scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_score = float(score)
            best_labels = labels
            best_k = cluster_count
    if best_labels is None:
        return loadings, variance, score_table, pd.DataFrame()

    score_table["cluster"] = best_labels + 1
    score_table["selected_k"] = best_k
    score_table["silhouette_score"] = best_score
    profile_source = videos.merge(
        score_table[["video_id", "cluster"]], on="video_id", how="inner"
    )
    profile = (
        profile_source.groupby("cluster", as_index=False)
        .agg(
            video_count=("video_id", "nunique"),
            median_views=("view_count", "median"),
            median_views_per_day=("views_per_day", "median"),
            median_engagement=("engagement_per_1000_views", "median"),
            median_duration=("duration_minutes", "median"),
            median_subscribers=("channel_subscriber_count", "median"),
        )
        .sort_values("cluster")
    )
    return loadings, variance, score_table, profile


def outlier_summary(videos: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in VIDEO_METRICS:
        if column not in videos:
            continue
        values = pd.to_numeric(videos[column], errors="coerce").dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_outliers = (values < lower) | (values > upper)
        median = values.median()
        mad = np.median(np.abs(values - median))
        if mad > 0:
            robust_z = 0.6745 * (values - median) / mad
            mad_outliers = np.abs(robust_z) > 3.5
        else:
            mad_outliers = pd.Series(False, index=values.index)
        rows.append(
            {
                "variable": column,
                "n": len(values),
                "iqr_lower": lower,
                "iqr_upper": upper,
                "iqr_outlier_count": int(iqr_outliers.sum()),
                "iqr_outlier_share": float(iqr_outliers.mean()),
                "mad_outlier_count": int(mad_outliers.sum()),
                "mad_outlier_share": float(mad_outliers.mean()),
                "note": "이상치는 자동 삭제하지 않고 표시만 함",
            }
        )
    return pd.DataFrame(rows)


def power_and_sample_size(videos: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    two_group_power = TTestIndPower()
    for effect in [0.2, 0.5, 0.8]:
        required = two_group_power.solve_power(
            effect_size=effect, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided"
        )
        rows.append(
            {
                "design": "Independent two-group t-test",
                "effect_size": effect,
                "alpha": 0.05,
                "target_power": 0.8,
                "groups": 2,
                "required_n_per_group": math.ceil(required),
                "required_total_n": math.ceil(required) * 2,
                "current_video_n": len(videos),
            }
        )
    anova_power = FTestAnovaPower()
    for effect in [0.1, 0.25, 0.4]:
        required_total = anova_power.solve_power(
            effect_size=effect, k_groups=3, alpha=0.05, power=0.8
        )
        rows.append(
            {
                "design": "One-way ANOVA (3 groups)",
                "effect_size": effect,
                "alpha": 0.05,
                "target_power": 0.8,
                "groups": 3,
                "required_n_per_group": math.ceil(required_total / 3),
                "required_total_n": math.ceil(required_total),
                "current_video_n": len(videos),
            }
        )
    return pd.DataFrame(rows)


def method_catalog(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    definitions = [
        ("가정·데이터 진단", "Shapiro-Wilk / D'Agostino K²", "연속형 변수의 정규성 탐색", "normality_tests"),
        ("가정·데이터 진단", "Brown-Forsythe", "집단별 분산 동질성", "variance_tests"),
        ("두 집단 비교", "Student / Welch t-test", "두 집단 평균 비교", "two_group_tests"),
        ("두 집단 비교", "Mann-Whitney U", "두 집단 순위·분포 비교", "two_group_tests"),
        ("두 집단 비교", "Cohen d / Hedges g / rank-biserial", "효과크기", "two_group_tests"),
        ("다집단·사후검정", "One-way / Welch ANOVA", "세 집단 이상 평균 비교", "multi_group_tests"),
        ("다집단·사후검정", "Kruskal-Wallis", "세 집단 이상 비모수 비교", "multi_group_tests"),
        ("다집단·사후검정", "Tukey HSD / pairwise Mann-Whitney", "유의한 전체 검정 후 집단쌍 비교", "posthoc_tests"),
        ("다집단·사후검정", "Holm / Benjamini-Hochberg", "다중검정 오류율 보정", "posthoc_tests"),
        ("상관분석", "Pearson / Spearman / Kendall", "연속형 변수 간 선형·순위 연관", "full_correlations"),
        ("범주형 분석", "Chi-square / Fisher exact", "범주형 변수 독립성", "contingency_tests"),
        ("범주형 분석", "Cramer's V", "범주형 연관 효과크기", "contingency_tests"),
        ("회귀·일반화선형모형", "OLS with HC3", "로그 조회수 다변량 연관", "regression_models"),
        ("회귀·일반화선형모형", "Quantile / robust regression", "중앙값·이상치 강건 회귀", "regression_models"),
        ("회귀·일반화선형모형", "Poisson / negative-binomial GLM", "댓글 수 같은 계수형 결과", "regression_models"),
        ("회귀·일반화선형모형", "Logistic regression", "고참여 영상 확률", "regression_models"),
        ("회귀·일반화선형모형", "VIF / BP / JB / DW", "회귀 가정과 적합성 진단", "regression_diagnostics"),
        ("시계열 추세", "Linear trend / Mann-Kendall", "월별 증가·감소 추세", "time_series_tests"),
        ("시계열 추세", "ADF stationarity", "12개월 이상 월별 정상성", "time_series_tests"),
        ("차원축소·군집", "PCA", "다변량 지표의 주요 축 요약", "pca_loadings"),
        ("차원축소·군집", "K-means / silhouette", "유사한 영상 유형 탐색", "cluster_profiles"),
        ("이상치·검정력", "IQR / robust MAD", "극단 관측값 진단", "outlier_summary"),
        ("이상치·검정력", "Power / sample-size analysis", "가정한 효과크기별 필요 표본", "power_analysis"),
    ]
    rows = []
    for category, method, purpose, result_key in definitions:
        frame = results.get(result_key, pd.DataFrame())
        status = "실행됨" if frame is not None and not frame.empty else "자료조건 미충족 또는 선택 안 함"
        rows.append(
            {
                "category": category,
                "method": method,
                "purpose": purpose,
                "result_table": result_key,
                "status": status,
                "interpretation_rule": "p값·보정 p값·효과크기·신뢰구간을 함께 해석",
            }
        )
    return pd.DataFrame(rows)


def run_advanced_statistics(
    videos: pd.DataFrame,
    comments: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    selected = set(config.get("advanced_methods") or METHOD_CATEGORIES)
    seed = int(config.get("random_seed", 42))
    results: dict[str, pd.DataFrame] = {}

    if "가정·데이터 진단" in selected:
        results["normality_tests"] = normality_tests(videos, seed)
        results["variance_tests"] = variance_homogeneity_tests(videos)
    if "두 집단 비교" in selected:
        results["two_group_tests"] = two_group_tests(videos)
    if "다집단·사후검정" in selected:
        results["multi_group_tests"] = multi_group_tests(videos)
        results["posthoc_tests"] = posthoc_tests(videos)
    if "상관분석" in selected:
        results["full_correlations"] = full_correlations(videos)
    if "범주형 분석" in selected:
        results["contingency_tests"] = contingency_tests(videos, comments)
    if "회귀·일반화선형모형" in selected:
        regression, diagnostics = regression_suite(videos)
        results["regression_models"] = regression
        results["regression_diagnostics"] = diagnostics
    if "시계열 추세" in selected:
        time_tests, monthly_extended = time_series_suite(videos)
        results["time_series_tests"] = time_tests
        results["monthly_extended"] = monthly_extended
    if "차원축소·군집" in selected:
        loadings, variance, scores, profiles = dimensionality_and_clusters(videos, seed)
        results["pca_loadings"] = loadings
        results["pca_variance"] = variance
        results["pca_cluster_scores"] = scores
        results["cluster_profiles"] = profiles
    if "이상치·검정력" in selected:
        results["outlier_summary"] = outlier_summary(videos)
        results["power_analysis"] = power_and_sample_size(videos)

    results["method_catalog"] = method_catalog(results)
    return results
