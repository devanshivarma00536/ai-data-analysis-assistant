"""Natural language intent parsing (rule-based, no LLM required)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FILLER_WORDS = {
    "can", "you", "please", "i", "want", "could", "would", "show", "me", "the", "a", "an",
    "what", "which", "how", "is", "are", "do", "does", "did", "give", "tell", "find",
}


@dataclass
class AnalysisIntent:
    intent_type: str
    description: str
    metric_col: str | None = None
    group_col: str | None = None
    filter_col: str | None = None
    filter_value: str | None = None
    compare_values: list[str] = field(default_factory=list)
    top_n: int = 10
    time_grain: str | None = None
    confidence: float = 0.75
    relevant_columns: list[str] = field(default_factory=list)


def _tokens(question: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", question.lower())


def _match_column(token: str, columns: list[str]) -> str | None:
    token = token.lower()
    for col in columns:
        col_lower = col.lower()
        if token == col_lower or token in col_lower.replace("_", " ") or col_lower in token:
            return col
    return None


def _find_column_in_text(text: str, columns: list[str]) -> str | None:
    text_lower = text.lower()
    best: tuple[int, str] | None = None
    for col in columns:
        col_lower = col.lower()
        variants = {col_lower, col_lower.replace("_", " ")}
        for variant in variants:
            if variant in text_lower:
                score = len(variant)
                if best is None or score > best[0]:
                    best = (score, col)
    return best[1] if best else None


def parse_intent(question: str, roles: dict[str, list[str]], columns: list[str]) -> AnalysisIntent:
    q = question.strip()
    q_lower = q.lower()
    tokens = [t for t in _tokens(q) if t not in FILLER_WORDS]

    numeric = roles.get("numeric", [])
    categorical = roles.get("categorical", [])
    datetime_cols = roles.get("datetime", [])
    text_cols = roles.get("text", [])
    group_candidates = categorical + text_cols + datetime_cols

    metric = _find_column_in_text(q, numeric) or (numeric[0] if len(numeric) == 1 else None)
    group = _find_column_in_text(q, group_candidates) or (
        categorical[0] if len(categorical) == 1 else None
    )

    # "by {column}" pattern e.g. "top 10 state by population"
    by_match = re.search(r"by\s+([a-zA-Z0-9_ ]+)", q_lower)
    if by_match:
        by_phrase = by_match.group(1).strip()
        by_col = _find_column_in_text(by_phrase, numeric) or _find_column_in_text(by_phrase, group_candidates)
        if by_col in numeric:
            metric = by_col
        elif by_col:
            group = by_col

    if not group:
        for col in group_candidates:
            if col != metric and _find_column_in_text(col, [col]):
                if col.lower() in q_lower or col.lower().replace("_", " ") in q_lower:
                    group = col
                    break

    top_n = 10
    top_match = re.search(r"top\s+(\d+)", q_lower)
    if top_match:
        top_n = int(top_match.group(1))

    # Compare intent: "compare X and Y" or "X vs Y"
    compare_values: list[str] = []
    compare_match = re.search(r"compare\s+(.+?)\s+(?:and|vs|versus|with)\s+(.+)", q_lower)
    if compare_match:
        compare_values = [compare_match.group(1).strip().title(), compare_match.group(2).strip().title()]
        return AnalysisIntent(
            intent_type="compare",
            description="Compare values across a category",
            metric_col=metric,
            group_col=group or categorical[0] if categorical else None,
            compare_values=compare_values,
            top_n=top_n,
            confidence=0.85,
            relevant_columns=[c for c in [metric, group] if c],
        )

    # Trend / monthly / over time
    if any(kw in q_lower for kw in ["trend", "monthly", "over time", "time series", "by month", "by year"]):
        date_col = _find_column_in_text(q, datetime_cols) or (datetime_cols[0] if datetime_cols else None)
        grain = "month" if "month" in q_lower else "year" if "year" in q_lower else "month"
        return AnalysisIntent(
            intent_type="trend",
            description=f"Show trend over time ({grain}ly)",
            metric_col=metric,
            group_col=date_col,
            time_grain=grain,
            confidence=0.88 if date_col else 0.5,
            relevant_columns=[c for c in [metric, date_col] if c],
        )

    # Correlation
    if any(kw in q_lower for kw in ["correlation", "correlate", "relationship between"]):
        nums = numeric[:2] if len(numeric) >= 2 else numeric
        return AnalysisIntent(
            intent_type="correlation",
            description="Correlation between numeric columns",
            metric_col=nums[0] if nums else None,
            group_col=nums[1] if len(nums) > 1 else None,
            confidence=0.8 if len(nums) >= 2 else 0.4,
            relevant_columns=nums,
        )

    # Forecast / predict
    if any(kw in q_lower for kw in ["forecast", "predict", "prediction", "next month", "next year"]):
        date_col = _find_column_in_text(q, datetime_cols) or (datetime_cols[0] if datetime_cols else None)
        return AnalysisIntent(
            intent_type="forecast",
            description="Forecast future values",
            metric_col=metric,
            group_col=date_col,
            time_grain="month" if "month" in q_lower else "year" if "year" in q_lower else "month",
            confidence=0.82 if metric and date_col else 0.45,
            relevant_columns=[c for c in [metric, date_col] if c],
        )

    # RFM
    if any(kw in q_lower for kw in ["rfm", "recency frequency monetary", "customer segment"]):
        cust = _find_column_in_text(q, group_candidates) or group
        date_col = _find_column_in_text(q, datetime_cols) or (datetime_cols[0] if datetime_cols else None)
        return AnalysisIntent(
            intent_type="rfm",
            description="RFM customer segmentation",
            metric_col=metric,
            group_col=cust,
            filter_col=date_col,
            confidence=0.75 if cust and metric and date_col else 0.4,
            relevant_columns=[c for c in [cust, date_col, metric] if c],
        )

    # ABC
    if any(kw in q_lower for kw in ["abc analysis", "abc class", "pareto"]):
        return AnalysisIntent(
            intent_type="abc",
            description="ABC classification by revenue contribution",
            metric_col=metric,
            group_col=group or (group_candidates[0] if group_candidates else None),
            confidence=0.8 if metric and group else 0.5,
            relevant_columns=[c for c in [metric, group] if c],
        )

    # Cohort
    if any(kw in q_lower for kw in ["cohort", "retention", "churn"]):
        cust = _find_column_in_text(q, group_candidates) or group
        date_col = _find_column_in_text(q, datetime_cols) or (datetime_cols[0] if datetime_cols else None)
        return AnalysisIntent(
            intent_type="cohort",
            description="Cohort retention analysis",
            group_col=cust,
            filter_col=date_col,
            confidence=0.75 if cust and date_col else 0.4,
            relevant_columns=[c for c in [cust, date_col] if c],
        )

    # Anomaly
    if any(kw in q_lower for kw in ["anomaly", "anomalies", "outlier", "unusual", "spike"]):
        return AnalysisIntent(
            intent_type="anomaly",
            description="Anomaly / outlier detection",
            metric_col=metric or (numeric[0] if numeric else None),
            confidence=0.8 if metric else 0.5,
            relevant_columns=[metric] if metric else numeric[:1],
        )

    # Distribution
    if any(kw in q_lower for kw in ["distribution", "histogram", "spread"]) and "outlier" not in q_lower:
        return AnalysisIntent(
            intent_type="distribution",
            description="Distribution of a numeric column",
            metric_col=metric,
            confidence=0.82 if metric else 0.45,
            relevant_columns=[metric] if metric else numeric[:1],
        )

    # Summary / overview
    if any(kw in q_lower for kw in ["summary", "overview", "describe", "statistics", "stats"]):
        return AnalysisIntent(
            intent_type="summary",
            description="Descriptive statistics summary",
            confidence=0.9,
            relevant_columns=columns[:10],
        )

    # Top N / ranking / most / highest / best
    if any(kw in q_lower for kw in ["top", "most", "highest", "best", "largest", "maximum", "rank"]):
        return AnalysisIntent(
            intent_type="ranking",
            description=f"Top {top_n} by metric",
            metric_col=metric,
            group_col=group or (categorical[0] if categorical else None),
            top_n=top_n,
            confidence=0.85 if metric and group else 0.55,
            relevant_columns=[c for c in [metric, group] if c],
        )

    # Default: group aggregation
    if metric and group:
        return AnalysisIntent(
            intent_type="aggregation",
            description=f"Aggregate {metric} by {group}",
            metric_col=metric,
            group_col=group,
            confidence=0.7,
            relevant_columns=[metric, group],
        )

    if numeric:
        return AnalysisIntent(
            intent_type="summary",
            description="Dataset summary (fallback)",
            confidence=0.5,
            relevant_columns=numeric[:5],
        )

    return AnalysisIntent(
        intent_type="summary",
        description="General dataset overview",
        confidence=0.4,
        relevant_columns=columns[:5],
    )


def format_intent_label(intent: AnalysisIntent) -> str:
    labels = {
        "ranking": "Ranking / Top N Analysis",
        "trend": "Trend / Time Series Analysis",
        "compare": "Category Comparison",
        "correlation": "Correlation Analysis",
        "distribution": "Distribution Analysis",
        "aggregation": "Group Aggregation",
        "summary": "Descriptive Summary",
        "forecast": "Forecasting / Prediction",
        "rfm": "RFM Customer Segmentation",
        "abc": "ABC Analysis",
        "cohort": "Cohort / Retention Analysis",
        "anomaly": "Anomaly Detection",
    }
    return labels.get(intent.intent_type, intent.description)
