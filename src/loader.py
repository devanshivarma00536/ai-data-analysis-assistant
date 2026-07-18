"""Load datasets from CSV, Excel, SQLite, and MySQL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".sqlite", ".db"}


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def load_excel(path: str | Path, sheet_name: str | int = 0, **kwargs: Any) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def load_sqlite(path: str | Path, table: str | None = None, query: str | None = None) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{Path(path).resolve()}")
    try:
        if query:
            return pd.read_sql(text(query), engine)
        if table:
            return pd.read_sql(text(f'SELECT * FROM "{table}"'), engine)
        tables = list_sqlite_tables(engine)
        if not tables:
            raise ValueError("SQLite database has no tables.")
        if len(tables) == 1:
            return pd.read_sql(text(f'SELECT * FROM "{tables[0]}"'), engine)
        raise ValueError(f"Multiple tables found: {tables}. Specify a table name.")
    finally:
        engine.dispose()


def list_sqlite_tables(engine: Engine) -> list[str]:
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    tables = pd.read_sql(text(query), engine)["name"].tolist()
    return tables


def load_mysql(
    host: str,
    user: str,
    password: str,
    database: str,
    port: int = 3306,
    table: str | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(url)
    try:
        if query:
            return pd.read_sql(text(query), engine)
        if table:
            return pd.read_sql(text(f"SELECT * FROM `{table}`"), engine)
        tables = pd.read_sql(text("SHOW TABLES"), engine).iloc[:, 0].tolist()
        if not tables:
            raise ValueError("MySQL database has no tables.")
        if len(tables) == 1:
            return pd.read_sql(text(f"SELECT * FROM `{tables[0]}`"), engine)
        raise ValueError(f"Multiple tables found: {tables}. Specify a table name.")
    finally:
        engine.dispose()


def load_file(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return load_excel(path, **kwargs)
    if suffix in {".sqlite", ".db"}:
        return load_sqlite(path, **kwargs)
    raise ValueError(f"Unsupported file type: {suffix}")


def list_data_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(
        p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
