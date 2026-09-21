"""
db.py
------
SQLite persistence layer for the dashboard.

Why a database at all, instead of just re-fetching from the API every time?
  1. History accumulates over time. Every day the refresh job runs, one new
     row gets added. A year from now this repo has a genuine, growing
     price history it built itself -- not just whatever the last 2 years
     of the API happens to return.
  2. It's fast and resilient. The app reads from SQLite on every page load
     (milliseconds) instead of hitting an external API on every load. If
     the EIA API is down or rate-limited, the app still works off the last
     known-good data.
  3. It lets us cache the forecast itself, tagged with when it was
     generated, so "refresh the model" has a concrete meaning: a scheduled
     job re-fits the model on the latest data and stores the new forecast,
     rather than every visitor re-fitting ARIMA from scratch.

Schema:
  prices(commodity, date, price, source)      -- one row per commodity/day
  forecasts(commodity, generated_at, date,
            forecast, lower_ci, upper_ci)     -- latest cached forecast run
  refresh_log(commodity, last_refreshed_at)   -- bookkeeping
"""

import sqlite3
import datetime as dt
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "energy_data.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")  # safer for concurrent read/refresh
    return conn


def init_db() -> None:
    """Create tables if they don't already exist. Safe to call every startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                commodity TEXT NOT NULL,
                date TEXT NOT NULL,
                price REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'unknown',
                PRIMARY KEY (commodity, date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts (
                commodity TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                date TEXT NOT NULL,
                forecast REAL NOT NULL,
                lower_ci REAL NOT NULL,
                upper_ci REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refresh_log (
                commodity TEXT PRIMARY KEY,
                last_refreshed_at TEXT NOT NULL,
                rows_added INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()


def upsert_prices(commodity: str, df: pd.DataFrame, source: str) -> int:
    """
    Insert/replace rows for this commodity. df must have columns: date, price.
    Returns the number of NEW rows added (existing dates are updated, not counted as new).
    """
    with _connect() as conn:
        existing_dates = set(
            r[0] for r in conn.execute(
                "SELECT date FROM prices WHERE commodity = ?", (commodity,)
            ).fetchall()
        )
        rows = [
            (commodity, row.date.strftime("%Y-%m-%d"), float(row.price), source)
            for row in df.itertuples()
        ]
        conn.executemany(
            """
            INSERT INTO prices (commodity, date, price, source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(commodity, date) DO UPDATE SET price = excluded.price, source = excluded.source
            """,
            rows,
        )
        new_dates = set(r[1] for r in rows) - existing_dates
        conn.commit()
        return len(new_dates)


def get_prices(commodity: str, days: int | None = None) -> pd.DataFrame:
    """Read stored price history for a commodity, most recent `days` if specified."""
    with _connect() as conn:
        query = "SELECT date, price FROM prices WHERE commodity = ? ORDER BY date ASC"
        df = pd.read_sql_query(query, conn, params=(commodity,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    if days:
        df = df.tail(days).reset_index(drop=True)
    return df


def get_latest_source(commodity: str) -> str | None:
    """Return the `source` tag ('eia_live' or 'synthetic') of the most recent stored row."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT source FROM prices WHERE commodity = ? ORDER BY date DESC LIMIT 1",
            (commodity,),
        ).fetchone()
    return row[0] if row else None


def has_data(commodity: str) -> bool:
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM prices WHERE commodity = ?", (commodity,)
        ).fetchone()[0]
    return count > 0


def set_last_refresh(commodity: str, rows_added: int) -> None:
    now = dt.datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO refresh_log (commodity, last_refreshed_at, rows_added)
            VALUES (?, ?, ?)
            ON CONFLICT(commodity) DO UPDATE SET last_refreshed_at = excluded.last_refreshed_at,
                                                  rows_added = excluded.rows_added
            """,
            (commodity, now, rows_added),
        )
        conn.commit()


def get_last_refresh(commodity: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT last_refreshed_at, rows_added FROM refresh_log WHERE commodity = ?",
            (commodity,),
        ).fetchone()
    if row is None:
        return None
    return {"last_refreshed_at": row[0], "rows_added": row[1]}


def save_forecast(commodity: str, forecast_df: pd.DataFrame) -> None:
    """Overwrite the cached forecast for a commodity with a freshly generated one."""
    generated_at = dt.datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM forecasts WHERE commodity = ?", (commodity,))
        rows = [
            (commodity, generated_at, row.date.strftime("%Y-%m-%d"),
             float(row.forecast), float(row.lower_ci), float(row.upper_ci))
            for row in forecast_df.itertuples()
        ]
        conn.executemany(
            """
            INSERT INTO forecasts (commodity, generated_at, date, forecast, lower_ci, upper_ci)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def get_cached_forecast(commodity: str, max_age_hours: float = 24.0) -> tuple[pd.DataFrame | None, str | None]:
    """
    Return (forecast_df, generated_at) if a forecast exists and is fresh enough,
    else (None, None) so the caller knows to regenerate.
    """
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT generated_at, date, forecast, lower_ci, upper_ci FROM forecasts "
            "WHERE commodity = ? ORDER BY date ASC",
            conn,
            params=(commodity,),
        )
    if df.empty:
        return None, None

    generated_at = df["generated_at"].iloc[0]
    age_hours = (dt.datetime.utcnow() - dt.datetime.fromisoformat(generated_at)).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None, generated_at

    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "forecast", "lower_ci", "upper_ci"]], generated_at
