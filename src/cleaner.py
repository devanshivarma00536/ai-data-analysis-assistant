"""Data cleaning utilities."""

from __future__ import annotations

import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cleaned = df.copy()
    actions: list[str] = []

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    removed = before - len(cleaned)
    if removed:
        actions.append(f"Removed {removed} duplicate rows.")

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            parsed = pd.to_datetime(cleaned[col], errors="coerce", utc=False)
            if parsed.notna().sum() >= max(3, int(len(cleaned) * 0.5)):
                cleaned[col] = parsed
                actions.append(f"Converted '{col}' to datetime.")

    for col in cleaned.select_dtypes(include="object").columns:
        numeric = pd.to_numeric(cleaned[col], errors="coerce")
        if numeric.notna().sum() >= max(3, int(len(cleaned) * 0.8)):
            cleaned[col] = numeric
            actions.append(f"Converted '{col}' to numeric.")

    for col in cleaned.select_dtypes(include="object").columns:
        if cleaned[col].nunique(dropna=True) <= 50:
            cleaned[col] = cleaned[col].astype("category")

    return cleaned, actions
