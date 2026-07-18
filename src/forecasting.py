"""Time series forecasting utilities."""

from __future__ import annotations

import pandas as pd


def forecast_series(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    periods: int = 3,
    grain: str = "month",
) -> pd.DataFrame:
    freq_map = {"month": "M", "year": "Y", "week": "W", "day": "D"}
    freq = freq_map.get(grain, "M")

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, metric_col])
    series = (
        tmp.groupby(tmp[date_col].dt.to_period(freq), observed=True)[metric_col]
        .sum()
        .sort_index()
    )

    if len(series) < 3:
        raise ValueError("Need at least 3 time periods for forecasting.")

    # Linear trend on index positions
    y = series.values.astype(float)
    x = list(range(len(y)))
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / max(
        sum((xi - x_mean) ** 2 for xi in x), 1e-9
    )
    intercept = y_mean - slope * x_mean

    last_period = series.index[-1]
    forecast_rows = []
    for step in range(1, periods + 1):
        pred = intercept + slope * (n - 1 + step)
        next_period = last_period + step
        forecast_rows.append(
            {
                "period": str(next_period),
                metric_col: round(max(pred, 0), 2),
                "type": "forecast",
            }
        )

    history = [
        {"period": str(idx), metric_col: round(float(val), 2), "type": "actual"}
        for idx, val in series.items()
    ]
    return pd.DataFrame(history + forecast_rows)
