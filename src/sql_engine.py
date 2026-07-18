"""Read-only SQL execution with safety checks."""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

FORBIDDEN_PATTERN = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|CREATE|REPLACE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


def validate_read_only_sql(query: str) -> None:
    if FORBIDDEN_PATTERN.search(query):
        raise ValueError("Only read-only SELECT queries are allowed.")
    stripped = query.strip().lstrip("(").strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")


def run_sql(engine: Engine, query: str) -> pd.DataFrame:
    validate_read_only_sql(query)
    return pd.read_sql(text(query), engine)


def sqlite_engine(path: str) -> Engine:
    return create_engine(f"sqlite:///{path}")


def mysql_engine(host: str, user: str, password: str, database: str, port: int = 3306) -> Engine:
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)
