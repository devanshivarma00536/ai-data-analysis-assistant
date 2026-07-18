"""Structured analysis output formatting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.intent_parser import AnalysisIntent, format_intent_label


def generate_insights(
    intent: AnalysisIntent,
    result_df: pd.DataFrame,
    profile: dict[str, Any],
) -> tuple[list[str], list[str], int]:
    insights: list[str] = []
    recommendations: list[str] = []

    if result_df.empty:
        insights.append("The analysis returned no rows. Check filters or column names.")
        recommendations.append("Verify column names and try a broader question.")
        return insights, recommendations, 35

    cols = result_df.columns.tolist()
    numeric_col = None
    for col in reversed(cols):
        if pd.api.types.is_numeric_dtype(result_df[col]):
            numeric_col = col
            break

    if intent.intent_type == "ranking" and len(cols) >= 2 and numeric_col:
        top_row = result_df.iloc[0]
        insights.append(
            f"'{top_row[cols[0]]}' ranks first with {top_row[numeric_col]:,.2f} on {numeric_col}."
        )
        if len(result_df) > 1:
            gap = top_row[numeric_col] - result_df.iloc[1][numeric_col]
            insights.append(f"Lead over second place: {gap:,.2f}.")
        recommendations.append(f"Focus resources on top performer: {top_row[cols[0]]}.")
        recommendations.append("Investigate lower-ranked categories for improvement opportunities.")

    elif intent.intent_type == "trend" and len(cols) >= 2 and numeric_col:
        series = result_df[numeric_col].dropna()
        if len(series) >= 2:
            change = series.iloc[-1] - series.iloc[0]
            direction = "increased" if change >= 0 else "decreased"
            insights.append(
                f"{numeric_col} {direction} by {abs(change):,.2f} from first to last period."
            )
            recommendations.append(
                "Increase inventory/marketing if trend is positive; investigate root cause if negative."
            )

    elif intent.intent_type == "compare" and len(cols) >= 2 and numeric_col:
        best = result_df.loc[result_df[numeric_col].idxmax()]
        insights.append(f"Highest value: {best[cols[0]]} ({best[numeric_col]:,.2f}).")
        recommendations.append("Target marketing and retention in high-performing segments.")

    elif intent.intent_type == "correlation":
        insights.append("Review correlated pairs for drivers or redundant features.")
        recommendations.append("Use strong correlations for forecasting; check multicollinearity in models.")

    elif intent.intent_type == "forecast" and len(cols) >= 2:
        forecast_rows = result_df[result_df.get("type", "actual") == "forecast"] if "type" in result_df.columns else result_df.tail(3)
        if not forecast_rows.empty:
            val_col = [c for c in cols if c not in {"period", "type"}][-1]
            insights.append(f"Forecast next period: ~{forecast_rows.iloc[0][val_col]:,.2f}.")
            recommendations.append("Validate forecast with recent trends before business decisions.")

    elif intent.intent_type == "abc" and "abc_class" in result_df.columns:
        a_count = (result_df["abc_class"] == "A").sum()
        insights.append(f"{a_count} items in class A drive ~80% of value.")
        recommendations.append("Prioritize inventory and marketing for A-class items.")

    elif intent.intent_type == "anomaly":
        insights.append(f"Detected {len(result_df)} anomalous records (|z| > 2.5).")
        recommendations.append("Investigate flagged rows for data errors or genuine spikes.")

    elif intent.intent_type == "rfm":
        insights.append("Customers scored by Recency, Frequency, and Monetary value.")
        recommendations.append("Target high RFM segments for retention campaigns.")

    elif intent.intent_type == "cohort":
        insights.append("Cohort table shows customer retention by signup month.")
        recommendations.append("Focus on cohorts with steep early drop-off.")

    elif intent.intent_type == "summary":
        insights.append(f"Dataset has {profile.get('rows', 0):,} rows and {profile.get('columns', 0)} columns.")
        if profile.get("quality_warnings"):
            insights.append(profile["quality_warnings"][0])
        recommendations.append("Clean missing values before advanced modeling.")

    else:
        if numeric_col and len(result_df) >= 1:
            insights.append(f"Average {numeric_col}: {result_df[numeric_col].mean():,.2f}.")
        recommendations.append("Refine the question with specific column names for deeper analysis.")

    if profile.get("duplicates", 0) > 0:
        recommendations.append("Remove duplicate rows to improve accuracy.")

    confidence = int(min(95, max(40, intent.confidence * 100)))
    if result_df.shape[0] < 3:
        confidence = max(40, confidence - 15)

    return insights, recommendations, confidence


def format_analysis_report(
    question: str,
    intent: AnalysisIntent,
    pandas_code: str,
    sql_code: str,
    result_df: pd.DataFrame,
    chart_type: str,
    chart_reason: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    insights, recommendations, confidence = generate_insights(intent, result_df, profile)

    return {
        "user_question": question,
        "detected_intent": format_intent_label(intent),
        "intent_detail": intent.description,
        "relevant_columns": intent.relevant_columns or profile.get("column_names", [])[:5],
        "pandas_code": pandas_code,
        "sql_code": sql_code,
        "chart_type": chart_type,
        "chart_reason": chart_reason,
        "result_preview": result_df.head(20),
        "analysis": insights,
        "business_insights": insights,
        "recommendations": recommendations,
        "confidence_score": confidence,
    }
