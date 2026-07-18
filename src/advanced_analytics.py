"""Advanced analytics: RFM, ABC, cohort, anomaly detection."""

from __future__ import annotations

import pandas as pd


def abc_analysis(df: pd.DataFrame, group_col: str, metric_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, observed=True)[metric_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total = grouped[metric_col].sum()
    grouped["share_pct"] = (grouped[metric_col] / total * 100).round(2) if total else 0
    grouped["cumulative_pct"] = grouped["share_pct"].cumsum().round(2)

    def classify(cum_pct: float) -> str:
        if cum_pct <= 80:
            return "A"
        if cum_pct <= 95:
            return "B"
        return "C"

    grouped["abc_class"] = grouped["cumulative_pct"].apply(classify)
    return grouped


def rfm_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    amount_col: str,
) -> pd.DataFrame:
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, amount_col, customer_col])
    snapshot = tmp[date_col].max() + pd.Timedelta(days=1)

    rfm = tmp.groupby(customer_col, observed=True).agg(
        recency=(date_col, lambda x: (snapshot - x.max()).days),
        frequency=(customer_col, "count"),
        monetary=(amount_col, "sum"),
    ).reset_index()

    for col, labels in [
        ("recency", [3, 2, 1]),
        ("frequency", [1, 2, 3]),
        ("monetary", [1, 2, 3]),
    ]:
        rfm[f"{col}_score"] = pd.qcut(rfm[col], 3, labels=labels, duplicates="drop")

    rfm["rfm_segment"] = (
        rfm["recency_score"].astype(str)
        + rfm["frequency_score"].astype(str)
        + rfm["monetary_score"].astype(str)
    )
    return rfm.sort_values("monetary", ascending=False)


def cohort_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
) -> pd.DataFrame:
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, customer_col])
    tmp["cohort"] = tmp.groupby(customer_col, observed=True)[date_col].transform(
        lambda x: x.dt.to_period("M")
    )
    tmp["period"] = tmp[date_col].dt.to_period("M")
    tmp["period_number"] = (tmp["period"] - tmp["cohort"]).apply(lambda x: x.n)

    cohort = (
        tmp.groupby(["cohort", "period_number"], observed=True)[customer_col]
        .nunique()
        .reset_index(name="customers")
    )
    cohort_pivot = cohort.pivot(index="cohort", columns="period_number", values="customers").fillna(0)
    cohort_pivot.index = cohort_pivot.index.astype(str)
    return cohort_pivot.reset_index()


def anomaly_detection(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    series = df[metric_col].dropna().astype(float)
    mean = series.mean()
    std = series.std() or 1.0
    z = ((series - mean) / std).abs()
    flagged = df.loc[series.index].copy()
    flagged["z_score"] = z.round(3)
    flagged["is_anomaly"] = z > 2.5
    return flagged[flagged["is_anomaly"]].sort_values("z_score", ascending=False)
