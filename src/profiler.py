"""Dataset profiling and schema inspection."""

from __future__ import annotations

from typing import Any

import pandas as pd


def infer_column_roles(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
            if parsed.notna().sum() >= max(3, int(len(df) * 0.5)):
                datetime_cols.append(col)

    categorical = [
        c
        for c in df.columns
        if c not in numeric and c not in datetime_cols and df[c].nunique(dropna=True) <= 50
    ]
    text = [c for c in df.columns if c not in numeric + datetime_cols + categorical]
    return {
        "numeric": numeric,
        "datetime": datetime_cols,
        "categorical": categorical,
        "text": text,
    }


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    roles = infer_column_roles(df)
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2) if len(df) else missing
    duplicates = int(df.duplicated().sum())

    columns = []
    for col in df.columns:
        columns.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "missing": int(missing[col]),
                "missing_pct": float(missing_pct[col]),
                "unique": int(df[col].nunique(dropna=True)),
                "sample": _sample_values(df[col]),
            }
        )

    quality_warnings = []
    if duplicates:
        quality_warnings.append(f"{duplicates} duplicate rows detected.")
    high_missing = [c["name"] for c in columns if c["missing_pct"] > 30]
    if high_missing:
        quality_warnings.append(f"High missing values (>30%): {', '.join(high_missing)}")
    if len(df) == 0:
        quality_warnings.append("Dataset is empty.")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "roles": roles,
        "column_details": columns,
        "duplicates": duplicates,
        "missing_total": int(missing.sum()),
        "quality_warnings": quality_warnings,
        "numeric_summary": df.describe(include="number").to_dict() if roles["numeric"] else {},
    }


def _sample_values(series: pd.Series, limit: int = 3) -> list[Any]:
    values = series.dropna().astype(str).unique()[:limit]
    return values.tolist()


def suggest_similar_columns(requested: str, available: list[str]) -> list[str]:
    requested_lower = requested.lower().replace("_", " ")
    scored: list[tuple[int, str]] = []
    for col in available:
        col_lower = col.lower().replace("_", " ")
        score = 0
        if requested_lower in col_lower or col_lower in requested_lower:
            score += 3
        if requested_lower.split()[0][:3] == col_lower[:3]:
            score += 1
        if score:
            scored.append((score, col))
    scored.sort(reverse=True)
    return [col for _, col in scored[:5]]
