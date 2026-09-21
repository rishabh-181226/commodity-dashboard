"""
data_sources.py
----------------
Handles all external data ingestion for the Commodity Price & Forward Curve Dashboard:
  - EIA API (spot/historical prices for WTI, Brent, Henry Hub)
  - Yahoo Finance (futures curves via yfinance)
  - Baker Hughes rig count (public CSV)

Every fetch function is wrapped in try/except and falls back to realistic
synthetic sample data if the network call fails or no API key is configured.
This means the dashboard is fully demoable even without live credentials,
which matters a lot when showing it to recruiters on a machine with no
internet access or before you've set up your own EIA API key.
"""

import io
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EIA_BASE_URL = "https://api.eia.gov/v2"

# EIA legacy (v1-style) series IDs, queried via the v2 API's backward-compatible
# /v2/seriesid/{legacy_id} endpoint. This is deliberately simpler than the
# route + facets[] approach: legacy series IDs are stable, extremely well
# documented, and this endpoint doesn't require knowing the correct facet key
# name for each dataset (which varies and is easy to get wrong).
#   PET.RWTC.D    = Cushing, OK WTI Spot Price FOB, Daily
#   PET.RBRTE.D   = Europe Brent Spot Price FOB, Daily
#   NG.RNGWHHD.D  = Henry Hub Natural Gas Spot Price, Daily
EIA_SERIES = {
    "WTI Crude (Cushing)": "PET.RWTC.D",
    "Brent Crude": "PET.RBRTE.D",
    "Henry Hub Natural Gas": "NG.RNGWHHD.D",
}

# Yahoo Finance continuous futures tickers used as a forward-curve proxy
YFINANCE_TICKERS = {
    "WTI Crude (Cushing)": "CL=F",
    "Brent Crude": "BZ=F",
    "Henry Hub Natural Gas": "NG=F",
}

BAKER_HUGHES_URL = "https://rigcount.bakerhughes.com/na-rig-count"  # landing page; direct CSV varies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_price_series(start_price: float, days: int, vol: float, seed: int) -> pd.DataFrame:
    """
    Generate a realistic-looking synthetic price series for demo mode using a
    mean-reverting (Ornstein-Uhlenbeck style) process on log-price. This keeps
    prices oscillating in a believable band around `start_price` rather than
    drifting away indefinitely, which a pure random walk would do over long
    lookback windows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=dt.date.today(), periods=days, freq="D")

    log_start = np.log(start_price)
    theta = 0.02  # speed of mean reversion
    log_prices = np.empty(days)
    log_prices[0] = log_start
    for i in range(1, days):
        shock = rng.normal(0, vol)
        log_prices[i] = log_prices[i - 1] + theta * (log_start - log_prices[i - 1]) + shock

    prices = np.exp(log_prices)
    return pd.DataFrame({"date": dates, "price": prices})


def _mock_futures_curve(spot: float, contango: bool, seed: int) -> pd.DataFrame:
    """Generate a synthetic 12-month forward curve, either contango or backwardated."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(start=dt.date.today(), periods=12, freq="MS")
    slope = 0.004 if contango else -0.004
    noise = rng.normal(0, 0.006, size=12)
    curve = spot * np.exp(np.cumsum([slope] * 12) + noise)
    return pd.DataFrame({"contract_month": months, "price": curve})


# ---------------------------------------------------------------------------
# EIA historical / spot prices
# ---------------------------------------------------------------------------

def _eia_seriesid_request(legacy_id: str, api_key: str, length: int = 730,
                           start: str | None = None) -> pd.DataFrame | None:
    """
    Shared helper: query the EIA v2 API via its legacy-series-id-compatible
    endpoint. Returns a (date, price) dataframe sorted ascending, or None on
    any failure (bad key, rate limit, network error, empty response, etc).
    """
    try:
        params = {
            "api_key": api_key,
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": 0,
            "length": length,
        }
        if start:
            params["start"] = start

        url = f"{EIA_BASE_URL}/seriesid/{legacy_id}"
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("response", {}).get("data", [])
        if not records:
            return None

        df = pd.DataFrame(records).rename(columns={"period": "date", "value": "price"})
        df["date"] = pd.to_datetime(df["date"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)
        return df[["date", "price"]] if len(df) else None
    except Exception:
        return None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_eia_history(commodity: str, api_key: str | None, days: int = 730) -> tuple[pd.DataFrame, bool]:
    """
    Fetch daily historical prices from the EIA API for the given commodity.

    Returns (dataframe, is_live) where is_live indicates whether real data
    was retrieved (True) or synthetic fallback data was used (False).
    Dataframe columns: date, price
    """
    legacy_id = EIA_SERIES[commodity]

    if api_key:
        df = _eia_seriesid_request(legacy_id, api_key, length=days)
        if df is not None:
            return df, True

    # --- Fallback: synthetic demo data ---
    base_prices = {
        "WTI Crude (Cushing)": 68.0,
        "Brent Crude": 71.0,
        "Henry Hub Natural Gas": 3.6,
    }
    vol = {
        "WTI Crude (Cushing)": 0.018,
        "Brent Crude": 0.017,
        "Henry Hub Natural Gas": 0.035,
    }
    seed = abs(hash(commodity)) % (2**32)
    df = _mock_price_series(base_prices[commodity], days, vol[commodity], seed)
    return df, False


# ---------------------------------------------------------------------------
# Incremental refresh helpers (used by refresh_data.py for the daily job)
# NOTE: intentionally NOT decorated with @st.cache_data -- these are called
# from the standalone refresh_data.py script (no Streamlit runtime present)
# as well as from app.py's manual refresh button.
# ---------------------------------------------------------------------------

def fetch_full_history_live(commodity: str, api_key: str, days: int = 730) -> pd.DataFrame | None:
    """Fetch full historical series directly from the EIA API. Returns None on any failure."""
    legacy_id = EIA_SERIES[commodity]
    return _eia_seriesid_request(legacy_id, api_key, length=days)


def fetch_latest_live_points(commodity: str, api_key: str, since_date: pd.Timestamp) -> pd.DataFrame | None:
    """Fetch only rows newer than `since_date` from the EIA API. Returns None on failure or no new data."""
    legacy_id = EIA_SERIES[commodity]
    df = _eia_seriesid_request(legacy_id, api_key, length=10)  # a handful of recent points is plenty
    if df is None:
        return None
    df = df[df["date"] > since_date].reset_index(drop=True)
    return df if len(df) else None


def generate_synthetic_bootstrap(commodity: str, days: int = 730) -> pd.DataFrame:
    """Full synthetic history for first-time DB bootstrap when no live data is available."""
    base_prices = {"WTI Crude (Cushing)": 68.0, "Brent Crude": 71.0, "Henry Hub Natural Gas": 3.6}
    vol = {"WTI Crude (Cushing)": 0.018, "Brent Crude": 0.017, "Henry Hub Natural Gas": 0.035}
    seed = abs(hash(commodity)) % (2**32)
    return _mock_price_series(base_prices[commodity], days, vol[commodity], seed)


def generate_synthetic_next_days(commodity: str, last_price: float, last_date: pd.Timestamp,
                                  target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Continue the mean-reverting synthetic process forward from the last stored
    price/date up through target_date (normally 'today'). Used when no live
    API key is available, so the demo database still grows by one real new
    row every day instead of being regenerated from scratch each run.
    """
    base_prices = {"WTI Crude (Cushing)": 68.0, "Brent Crude": 71.0, "Henry Hub Natural Gas": 3.6}
    vol = {"WTI Crude (Cushing)": 0.018, "Brent Crude": 0.017, "Henry Hub Natural Gas": 0.035}
    log_start = np.log(base_prices[commodity])
    theta = 0.02

    n_days = (target_date - last_date).days
    if n_days <= 0:
        return pd.DataFrame(columns=["date", "price"])

    seed = abs(hash(f"{commodity}-{last_date}")) % (2**32)
    rng = np.random.default_rng(seed)

    log_price = np.log(last_price)
    dates, prices = [], []
    for i in range(1, n_days + 1):
        shock = rng.normal(0, vol[commodity])
        log_price = log_price + theta * (log_start - log_price) + shock
        dates.append(last_date + pd.Timedelta(days=i))
        prices.append(np.exp(log_price))

    return pd.DataFrame({"date": dates, "price": prices})


# ---------------------------------------------------------------------------
# Futures / forward curve (Yahoo Finance proxy)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_forward_curve(commodity: str, spot_price: float) -> tuple[pd.DataFrame, bool]:
    """
    Attempt to build a forward curve using yfinance continuous futures data.
    Yahoo Finance doesn't expose a clean multi-month curve for free, so in
    practice this mostly uses the front-month quote plus a modeled curve
    shape. Falls back to fully synthetic curve if the fetch fails.
    """
    ticker_symbol = YFINANCE_TICKERS[commodity]
    try:
        import yfinance as yf

        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            raise ValueError("no data returned")
        live_spot = float(hist["Close"].iloc[-1])
        contango = commodity != "Brent Crude"  # placeholder heuristic
        seed = abs(hash(commodity + "curve")) % (2**32)
        curve = _mock_futures_curve(live_spot, contango, seed)
        return curve, True
    except Exception:
        contango = True
        seed = abs(hash(commodity + "curve")) % (2**32)
        curve = _mock_futures_curve(spot_price, contango, seed)
        return curve, False


# ---------------------------------------------------------------------------
# Rig count (Baker Hughes) — used for correlation view
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_rig_count(days: int = 730) -> tuple[pd.DataFrame, bool]:
    """
    Baker Hughes doesn't publish a stable, scrape-friendly public CSV endpoint,
    so this always returns realistic synthetic weekly rig-count data unless
    you wire in your own licensed data source. Structure mirrors the real
    weekly series so swapping in real data later is a drop-in change.
    """
    rng = np.random.default_rng(42)
    weeks = pd.date_range(end=dt.date.today(), periods=days // 7, freq="W")
    base = 600
    trend = np.linspace(0, -40, len(weeks))
    noise = rng.normal(0, 8, len(weeks))
    rig_count = np.clip(base + trend + np.cumsum(noise) * 0.1, 300, 900)
    df = pd.DataFrame({"date": weeks, "rig_count": rig_count.round().astype(int)})
    return df, False


# ---------------------------------------------------------------------------
# USD Index (macro correlation feature) via FRED-style mock (no key required)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_usd_index(days: int = 730) -> pd.DataFrame:
    """Synthetic USD index series for correlation panel (swap for real FRED series easily)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=dt.date.today(), periods=days, freq="D")
    shocks = rng.normal(0, 0.002, days)
    idx = 100 * np.exp(np.cumsum(shocks))
    return pd.DataFrame({"date": dates, "usd_index": idx})
