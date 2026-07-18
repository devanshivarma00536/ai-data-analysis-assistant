"""Execute analysis based on parsed intent."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.advanced_analytics import abc_analysis, anomaly_detection, cohort_analysis, rfm_analysis
from src.forecasting import forecast_series
from src.intent_parser import AnalysisIntent


def build_pandas_code(intent: AnalysisIntent, df_name: str = "df") -> str:
    metric = intent.metric_col
    group = intent.group_col

    if intent.intent_type == "summary":
        return f"{df_name}.describe(include='all').T"

    if intent.intent_type == "ranking" and metric and not group:
        return (
            f"{df_name}.sort_values('{metric}', ascending=False).head({intent.top_n})"
        )

    if intent.intent_type == "ranking" and metric and group:
        return (
            f"{df_name}.groupby('{group}', observed=True)['{metric}'].sum()"
            f".sort_values(ascending=False).head({intent.top_n}).reset_index()"
        )

    if intent.intent_type == "aggregation" and metric and group:
        return (
            f"{df_name}.groupby('{group}', observed=True)['{metric}'].agg(['sum', 'mean', 'count'])"
            f".sort_values('sum', ascending=False).reset_index()"
        )

    if intent.intent_type == "trend" and metric and group:
        grain = intent.time_grain or "month"
        freq_map = {"month": "M", "year": "Y", "week": "W", "day": "D"}
        freq = freq_map.get(grain, "M")
        return (
            f"(_t := {df_name}.copy(), "
            f"_t.__setitem__('{group}', pd.to_datetime(_t['{group}'], errors='coerce')), "
            f"_t := _t.dropna(subset=['{group}']), "
            f"_t.assign(_period=_t['{group}'].dt.to_period('{freq}'))"
            f".groupby('_period', observed=True)['{metric}'].sum().reset_index())[-1]"
        )

    if intent.intent_type == "compare" and group:
        if intent.compare_values and metric:
            mask = " | ".join(
                [
                    f"{df_name}['{group}'].astype(str).str.contains('{v}', case=False, na=False)"
                    for v in intent.compare_values
                ]
            )
            return f"{df_name}[{mask}].groupby('{group}', observed=True)['{metric}'].sum().reset_index()"
        if metric:
            return f"{df_name}.groupby('{group}', observed=True)['{metric}'].sum().reset_index()"
        return f"{df_name}['{group}'].value_counts().reset_index(name='count')"

    if intent.intent_type == "correlation":
        cols = [c for c in intent.relevant_columns if c]
        if len(cols) >= 2:
            return f"{df_name}[{cols!r}].corr()"
        return f"{df_name}.select_dtypes(include='number').corr()"

    if intent.intent_type == "distribution" and metric:
        return f"{df_name}['{metric}'].describe().to_frame().T"

    if intent.intent_type == "forecast" and metric and group:
        return f"# forecast: {metric} by {group} (3 periods)"

    if intent.intent_type == "abc" and metric and group:
        return f"# abc: {metric} by {group}"

    if intent.intent_type == "rfm":
        return "# rfm customer segmentation"

    if intent.intent_type == "cohort":
        return "# cohort retention analysis"

    if intent.intent_type == "anomaly" and metric:
        return f"# anomaly detection on {metric}"

    if metric and group:
        return f"{df_name}.groupby('{group}', observed=True)['{metric}'].sum().reset_index()"
    if group:
        return f"{df_name}['{group}'].value_counts().head({intent.top_n}).reset_index(name='count')"
    return f"{df_name}.head(20)"


def build_sql_code(intent: AnalysisIntent, table: str = "data") -> str:
    metric = intent.metric_col or "value"
    group = intent.group_col or "category"

    if intent.intent_type == "summary":
        return f"SELECT COUNT(*) AS row_count FROM {table};"

    if intent.intent_type == "ranking":
        return (
            f"SELECT {group}, SUM({metric}) AS total_{metric}\n"
            f"FROM {table}\n"
            f"GROUP BY {group}\n"
            f"ORDER BY total_{metric} DESC\n"
            f"LIMIT {intent.top_n};"
        )

    if intent.intent_type == "aggregation":
        return (
            f"SELECT {group},\n"
            f"       SUM({metric}) AS total,\n"
            f"       AVG({metric}) AS average,\n"
            f"       COUNT(*) AS count\n"
            f"FROM {table}\n"
            f"GROUP BY {group}\n"
            f"ORDER BY total DESC;"
        )

    if intent.intent_type == "compare" and intent.compare_values:
        conditions = " OR ".join([f"{group} LIKE '%{v}%'" for v in intent.compare_values])
        return (
            f"SELECT {group}, SUM({metric}) AS total_{metric}\n"
            f"FROM {table}\n"
            f"WHERE {conditions}\n"
            f"GROUP BY {group};"
        )

    return f"SELECT * FROM {table} LIMIT 20;"


def run_trend_analysis(df: pd.DataFrame, metric: str, group: str, grain: str) -> pd.DataFrame:
    freq_map = {"month": "M", "year": "Y", "week": "W", "day": "D"}
    freq = freq_map.get(grain, "M")
    tmp = df.copy()
    tmp[group] = pd.to_datetime(tmp[group], errors="coerce")
    tmp = tmp.dropna(subset=[group])
    tmp["_period"] = tmp[group].dt.to_period(freq)
    result = tmp.groupby("_period", observed=True)[metric].sum().reset_index()
    result["_period"] = result["_period"].astype(str)
    return result


def run_analysis(df: pd.DataFrame, intent: AnalysisIntent) -> tuple[Any, str, str]:
    code = build_pandas_code(intent)
    sql = build_sql_code(intent)

    if intent.intent_type == "forecast" and intent.metric_col and intent.group_col:
        try:
            result = forecast_series(
                df,
                intent.group_col,
                intent.metric_col,
                periods=3,
                grain=intent.time_grain or "month",
            )
            code = (
                f"forecast_series(df, '{intent.group_col}', '{intent.metric_col}', "
                f"periods=3, grain='{intent.time_grain or 'month'}')"
            )
            return result, code, sql
        except Exception:
            pass

    if intent.intent_type == "abc" and intent.metric_col and intent.group_col:
        result = abc_analysis(df, intent.group_col, intent.metric_col)
        code = f"abc_analysis(df, '{intent.group_col}', '{intent.metric_col}')"
        return result, code, sql

    if intent.intent_type == "rfm" and intent.group_col and intent.filter_col and intent.metric_col:
        result = rfm_analysis(df, intent.group_col, intent.filter_col, intent.metric_col)
        code = f"rfm_analysis(df, '{intent.group_col}', '{intent.filter_col}', '{intent.metric_col}')"
        return result, code, sql

    if intent.intent_type == "cohort" and intent.group_col and intent.filter_col:
        result = cohort_analysis(df, intent.group_col, intent.filter_col)
        code = f"cohort_analysis(df, '{intent.group_col}', '{intent.filter_col}')"
        return result, code, sql

    if intent.intent_type == "anomaly" and intent.metric_col:
        result = anomaly_detection(df, intent.metric_col)
        code = f"anomaly_detection(df, '{intent.metric_col}')"
        return result, code, sql

    if intent.intent_type == "trend" and intent.metric_col and intent.group_col:
        result = run_trend_analysis(
            df, intent.metric_col, intent.group_col, intent.time_grain or "month"
        )
        code = (
            f"df.assign({intent.group_col}=pd.to_datetime(df['{intent.group_col}'], errors='coerce'))"
            f".dropna(subset=['{intent.group_col}'])"
            f".groupby(df['{intent.group_col}'].dt.to_period('{intent.time_grain or 'month'}'))"
            f"['{intent.metric_col}'].sum().reset_index()"
        )
        return result, code, sql

    local_env = {"pd": pd, "df": df}
    try:
        result = eval(code, local_env)  # noqa: S307
    except Exception:
        result = df.head(20)

    return result, code, sql


def result_to_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.reset_index()
    return pd.DataFrame({"result": [result]})
