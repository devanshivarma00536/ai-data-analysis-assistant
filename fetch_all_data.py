"""Fetch multiple open datasets into the data/ folder."""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TIMEOUT = 90


def fetch_json(url: str) -> tuple[object | None, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "DataAnalysisAssistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except Exception as exc:  # noqa: BLE001 - report all fetch failures
        return None, str(exc)


def fetch_text(url: str) -> tuple[str | None, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "DataAnalysisAssistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read().decode("utf-8"), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_products() -> tuple[str, str | None]:
    data, err = fetch_json("https://fakestoreapi.com/products")
    if err:
        return "failed", err
    rows = [
        {
            "id": item["id"],
            "title": item["title"],
            "price": item["price"],
            "category": item["category"],
            "rating_rate": item.get("rating", {}).get("rate"),
            "rating_count": item.get("rating", {}).get("count"),
        }
        for item in data
    ]
    write_csv(
        DATA_DIR / "ecommerce_products.csv",
        ["id", "title", "price", "category", "rating_rate", "rating_count"],
        rows,
    )
    return "ok", f"{len(rows)} products"


def save_countries() -> tuple[str, str | None]:
    url = (
        "https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv"
    )
    text, err = fetch_text(url)
    if err:
        return "failed", err
    lines = text.splitlines()
    if not lines:
        return "failed", "Empty country dataset"
    reader = csv.DictReader(lines)
    rows = []
    for item in reader:
        rows.append(
            {
                "country_code": item.get("ISO3166-1-Alpha-3"),
                "country_name": item.get("CLDR display name") or item.get("Official Name"),
                "capital": item.get("Capital"),
                "region": item.get("Region Name"),
                "subregion": item.get("Sub-region Name"),
                "population": item.get("Population"),
                "area_km2": item.get("Area"),
                "currency": item.get("ISO4217-currency_alphabetic_code"),
            }
        )
    write_csv(
        DATA_DIR / "world_countries.csv",
        [
            "country_code",
            "country_name",
            "capital",
            "region",
            "subregion",
            "population",
            "area_km2",
            "currency",
        ],
        rows,
    )
    return "ok", f"{len(rows)} countries"


def save_exchange_rates() -> tuple[str, str | None]:
    data, err = fetch_json("https://api.frankfurter.app/latest?from=USD")
    if err:
        return "failed", err
    as_of = data.get("date")
    rows = [
        {"base_currency": "USD", "target_currency": code, "rate": rate, "as_of": as_of}
        for code, rate in sorted((data.get("rates") or {}).items())
    ]
    write_csv(
        DATA_DIR / "currency_exchange_rates.csv",
        ["base_currency", "target_currency", "rate", "as_of"],
        rows,
    )
    return "ok", f"{len(rows)} rates as of {as_of}"


def save_crypto() -> tuple[str, str | None]:
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"
    )
    data, err = fetch_json(url)
    if err:
        return "failed", err
    rows = [
        {
            "rank": idx,
            "id": item["id"],
            "symbol": item["symbol"],
            "name": item["name"],
            "current_price_usd": item.get("current_price"),
            "market_cap_usd": item.get("market_cap"),
            "total_volume_usd": item.get("total_volume"),
            "price_change_24h_pct": item.get("price_change_percentage_24h"),
        }
        for idx, item in enumerate(data, start=1)
    ]
    write_csv(
        DATA_DIR / "crypto_top100.csv",
        [
            "rank",
            "id",
            "symbol",
            "name",
            "current_price_usd",
            "market_cap_usd",
            "total_volume_usd",
            "price_change_24h_pct",
        ],
        rows,
    )
    return "ok", f"{len(rows)} coins"


def save_earthquakes() -> tuple[str, str | None]:
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        "?format=geojson&starttime=2025-01-01&minmagnitude=5.0&limit=200"
    )
    data, err = fetch_json(url)
    if err:
        return "failed", err
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [None, None, None])
        rows.append(
            {
                "event_id": feature.get("id"),
                "magnitude": props.get("mag"),
                "place": props.get("place"),
                "time_utc": datetime.fromtimestamp(
                    (props.get("time") or 0) / 1000, tz=timezone.utc
                ).isoformat(),
                "longitude": coords[0] if len(coords) > 0 else None,
                "latitude": coords[1] if len(coords) > 1 else None,
                "depth_km": coords[2] if len(coords) > 2 else None,
                "alert": props.get("alert"),
                "tsunami": props.get("tsunami"),
            }
        )
    write_csv(
        DATA_DIR / "earthquakes_2025.csv",
        [
            "event_id",
            "magnitude",
            "place",
            "time_utc",
            "longitude",
            "latitude",
            "depth_km",
            "alert",
            "tsunami",
        ],
        rows,
    )
    return "ok", f"{len(rows)} earthquakes"


def save_weather() -> tuple[str, str | None]:
    # Daily weather for New York, London, Tokyo, Mumbai, Sydney (2024)
    locations = [
        ("New York", 40.7128, -74.0060),
        ("London", 51.5074, -0.1278),
        ("Tokyo", 35.6762, 139.6503),
        ("Mumbai", 19.0760, 72.8777),
        ("Sydney", -33.8688, 151.2093),
    ]
    rows = []
    for city, lat, lon in locations:
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}&start_date=2024-01-01&end_date=2024-12-31"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            "&timezone=UTC"
        )
        data, err = fetch_json(url)
        if err:
            return "failed", f"{city}: {err}"
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_t = daily.get("temperature_2m_max", [])
        min_t = daily.get("temperature_2m_min", [])
        rain = daily.get("precipitation_sum", [])
        for idx, day in enumerate(dates):
            rows.append(
                {
                    "city": city,
                    "date": day,
                    "temp_max_c": max_t[idx] if idx < len(max_t) else None,
                    "temp_min_c": min_t[idx] if idx < len(min_t) else None,
                    "precipitation_mm": rain[idx] if idx < len(rain) else None,
                }
            )
    write_csv(
        DATA_DIR / "weather_daily_2024.csv",
        ["city", "date", "temp_max_c", "temp_min_c", "precipitation_mm"],
        rows,
    )
    return "ok", f"{len(rows)} daily records across {len(locations)} cities"


def save_world_bank() -> tuple[str, str | None]:
    countries = "usa;chn;ind;deu;jpn;gbr;bra;fra;can;aus"
    url = (
        "https://api.worldbank.org/v2/country/"
        f"{countries}/indicator/NY.GDP.MKTP.KD.ZG"
        "?date=2015:2024&format=json&per_page=200"
    )
    data, err = fetch_json(url)
    if err:
        return "failed", err
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return "failed", "Unexpected World Bank response"
    rows = []
    for item in data[1]:
        rows.append(
            {
                "country_code": item.get("countryiso3code"),
                "country_name": (item.get("country") or {}).get("value"),
                "year": int(item["date"]),
                "gdp_growth_pct": item.get("value"),
            }
        )
    rows.sort(key=lambda row: (row["country_name"] or "", row["year"]))
    write_csv(
        DATA_DIR / "world_bank_gdp_growth.csv",
        ["country_code", "country_name", "year", "gdp_growth_pct"],
        rows,
    )
    return "ok", f"{len(rows)} country-year rows"


def save_github_csv(name: str, url: str) -> tuple[str, str | None]:
    text, err = fetch_text(url)
    if err:
        return "failed", err
    path = DATA_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    line_count = max(len(text.splitlines()) - 1, 0)
    return "ok", f"{line_count} rows"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("ecommerce_products", save_products),
        ("world_countries", save_countries),
        ("currency_exchange_rates", save_exchange_rates),
        ("crypto_top100", save_crypto),
        ("earthquakes_2025", save_earthquakes),
        ("weather_daily_2024", save_weather),
        ("world_bank_gdp_growth", save_world_bank),
        (
            "usa_states_2014",
            lambda: save_github_csv(
                "usa_states_2014.csv",
                "https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv",
            ),
        ),
        (
            "tips_dataset",
            lambda: save_github_csv(
                "tips_dataset.csv",
                "https://raw.githubusercontent.com/plotly/datasets/master/tips.csv",
            ),
        ),
        (
            "iris_dataset",
            lambda: save_github_csv(
                "iris_dataset.csv",
                "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
            ),
        ),
        (
            "gapminder_2007",
            lambda: save_github_csv(
                "gapminder_2007.csv",
                "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv",
            ),
        ),
        (
            "stock_prices_sample",
            lambda: save_github_csv(
                "stock_prices_sample.csv",
                "https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv",
            ),
        ),
    ]

    print(f"Saving datasets to: {DATA_DIR}\n")
    success = 0
    failed = 0
    for label, task in tasks:
        status, detail = task()
        if status == "ok":
            success += 1
            print(f"[OK]   {label}: {detail}")
        else:
            failed += 1
            print(f"[FAIL] {label}: {detail}")

    print(f"\nDone. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
