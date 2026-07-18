"""Automatic chart selection and rendering."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.intent_parser import AnalysisIntent

CHART_REASONS = {
    "bar": "Best for comparing categories or top-N rankings.",
    "line": "Best for trends and changes over time.",
    "pie": "Best for showing composition or market share.",
    "scatter": "Best for exploring correlation between two numeric variables.",
    "histogram": "Best for viewing distribution and spread.",
    "heatmap": "Best for correlation matrices.",
    "box": "Best for outlier detection.",
    "area": "Best for cumulative growth over time.",
    "stacked_bar": "Best for multi-category comparisons.",
    "table": "Best when the result is small or textual.",
}


def select_chart(intent: AnalysisIntent, result_df: pd.DataFrame) -> tuple[str, str]:
    if intent.intent_type == "trend":
        return "line", CHART_REASONS["line"]
    if intent.intent_type == "forecast":
        return "line", CHART_REASONS["line"]
    if intent.intent_type == "abc":
        return "bar", CHART_REASONS["bar"]
    if intent.intent_type == "anomaly":
        return "box", CHART_REASONS["box"]
    if intent.intent_type == "cohort":
        return "heatmap", CHART_REASONS["heatmap"]
    if intent.intent_type == "correlation":
        if result_df.shape[0] == result_df.shape[1]:
            return "heatmap", CHART_REASONS["heatmap"]
        return "scatter", CHART_REASONS["scatter"]
    if intent.intent_type == "distribution":
        return "histogram", CHART_REASONS["histogram"]
    if intent.intent_type == "compare":
        return "bar", CHART_REASONS["bar"]
    if intent.intent_type == "ranking":
        return "bar", CHART_REASONS["bar"]
    if intent.intent_type == "aggregation":
        return "bar", CHART_REASONS["bar"]
    if result_df.shape[1] >= 2 and pd.api.types.is_numeric_dtype(result_df.iloc[:, -1]):
        return "bar", CHART_REASONS["bar"]
    return "table", CHART_REASONS["table"]


def create_plotly_chart(
    intent: AnalysisIntent,
    result_df: pd.DataFrame,
    chart_type: str,
) -> go.Figure | None:
    if result_df.empty or chart_type == "table":
        return None

    df = result_df.copy()
    cols = df.columns.tolist()

    try:
        if chart_type == "heatmap":
            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                return None
            return px.imshow(numeric, text_auto=True, title="Correlation Heatmap")

        if chart_type == "histogram" and intent.metric_col:
            return px.histogram(df, x=intent.metric_col, title=f"Distribution of {intent.metric_col}")

        if chart_type == "scatter" and len(cols) >= 2:
            return px.scatter(df, x=cols[0], y=cols[1], title=f"{cols[0]} vs {cols[1]}")

        if chart_type == "line":
            x_col, y_col = cols[0], cols[-1]
            if "type" in df.columns:
                fig = px.line(df, x=x_col, y=y_col, color="type", markers=True, title=f"{y_col} over {x_col}")
            else:
                fig = px.line(df, x=x_col, y=y_col, markers=True, title=f"{y_col} over {x_col}")
            return fig

        if chart_type == "pie" and len(cols) >= 2:
            return px.pie(df, names=cols[0], values=cols[1], title=f"{cols[1]} by {cols[0]}")

        if chart_type == "bar" and len(cols) >= 2:
            return px.bar(
                df.head(20),
                x=cols[0],
                y=cols[1],
                title=f"{cols[1]} by {cols[0]}",
                text=cols[1],
            )

        if len(cols) >= 2:
            return px.bar(df.head(20), x=cols[0], y=cols[1], title="Analysis Result")
    except Exception:
        return None
    return None


def create_matplotlib_chart(intent: AnalysisIntent, result_df: pd.DataFrame, chart_type: str) -> plt.Figure | None:
    if result_df.empty or chart_type == "table":
        return None

    df = result_df.head(20)
    cols = df.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        if chart_type == "line" and len(cols) >= 2:
            ax.plot(df[cols[0]], df[cols[-1]], marker="o")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[-1])
        elif chart_type == "histogram" and intent.metric_col and intent.metric_col in df.columns:
            ax.hist(df[intent.metric_col].dropna(), bins=20, edgecolor="black")
            ax.set_xlabel(intent.metric_col)
        elif len(cols) >= 2:
            ax.bar(df[cols[0]].astype(str), df[cols[1]])
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            plt.xticks(rotation=45, ha="right")
        else:
            plt.close(fig)
            return None
        ax.set_title(intent.description)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        return None
