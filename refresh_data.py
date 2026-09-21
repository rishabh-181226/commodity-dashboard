"""
refresh_data.py
------------------
The daily refresh job. This is the script that answers "how do we keep the
model accurate": run this once a day (via cron, GitHub Actions, or the
"Refresh now" button in the app) and it will:

  1. For each commodity, pull the newest price data:
       - if EIA_API_KEY is set: fetch the latest live point(s) from the EIA API
       - if not (or the API call fails): extend the synthetic demo series by
         exactly one more day, continuing from the last stored price, so the
         demo data keeps growing realistically instead of resetting
  2. Upsert those new rows into data/energy_data.db (idempotent -- running it
     twice in a day just re-writes today's row, it doesn't duplicate)
  3. Re-fit the ARIMA forecasting model on the *updated* price history and
     cache the new forecast, tagged with a timestamp, so the app always
     serves a forecast that was generated from yesterday's (or today's)
     closing price -- not a stale one from whenever the repo was cloned.

Run manually:
    python refresh_data.py

Run for a specific commodity only:
    python refresh_data.py --commodity "WTI Crude (Cushing)"

Environment variables:
    EIA_API_KEY   optional -- free key from https://www.eia.gov/opendata/register.php
"""

import os
import sys
import argparse
import datetime as dt

import pandas as pd

import db
from data_sources import (
    EIA_SERIES,
    fetch_full_history_live,
    fetch_latest_live_points,
    generate_synthetic_bootstrap,
    generate_synthetic_next_days,
)
from forecasting import arima_forecast

BOOTSTRAP_DAYS = 730  # ~2 years of history on first run


def refresh_commodity(commodity: str, api_key: str | None) -> dict:
    """Refresh a single commodity's price history + cached forecast. Returns a summary dict."""
    today = pd.Timestamp(dt.date.today())
    rows_added = 0
    source = "synthetic"

    if not db.has_data(commodity):
        # First run for this commodity: bootstrap full history.
        live_df = fetch_full_history_live(commodity, api_key, days=BOOTSTRAP_DAYS) if api_key else None
        if live_df is not None:
            source = "eia_live"
            bootstrap_df = live_df
        else:
            bootstrap_df = generate_synthetic_bootstrap(commodity, days=BOOTSTRAP_DAYS)
        rows_added = db.upsert_prices(commodity, bootstrap_df, source=source)

    else:
        # Incremental refresh: fetch only what's new since our last stored date.
        existing = db.get_prices(commodity)
        last_date = existing["date"].max()
        last_price = float(existing.sort_values("date")["price"].iloc[-1])

        new_df = fetch_latest_live_points(commodity, api_key, last_date) if api_key else None
        if new_df is not None and len(new_df) > 0:
            source = "eia_live"
        else:
            new_df = generate_synthetic_next_days(commodity, last_price, last_date, today)
            source = "synthetic"

        if len(new_df) > 0:
            rows_added = db.upsert_prices(commodity, new_df, source=source)

    db.set_last_refresh(commodity, rows_added)

    # Re-fit forecast on the freshest data available.
    forecast_status = "skipped"
    try:
        full_history = db.get_prices(commodity)
        if len(full_history) >= 30:  # ARIMA needs a reasonable history length
            forecast_df = arima_forecast(full_history, periods=30)
            db.save_forecast(commodity, forecast_df)
            forecast_status = "updated"
    except Exception as e:
        forecast_status = f"failed: {e}"

    return {
        "commodity": commodity,
        "source": source,
        "rows_added": rows_added,
        "forecast_status": forecast_status,
    }


def main():
    parser = argparse.ArgumentParser(description="Daily refresh job for the commodity dashboard.")
    parser.add_argument("--commodity", default=None, help="Refresh only this commodity (default: all)")
    args = parser.parse_args()

    api_key = os.environ.get("EIA_API_KEY") or None
    db.init_db()

    commodities = [args.commodity] if args.commodity else list(EIA_SERIES.keys())

    print(f"=== Daily refresh started: {dt.datetime.utcnow().isoformat()} UTC ===")
    print(f"EIA API key configured: {'yes' if api_key else 'no (using synthetic fallback)'}")

    all_ok = True
    for commodity in commodities:
        try:
            result = refresh_commodity(commodity, api_key)
            print(f"  [{result['commodity']}] source={result['source']} "
                  f"new_rows={result['rows_added']} forecast={result['forecast_status']}")
        except Exception as e:
            all_ok = False
            print(f"  [{commodity}] FAILED: {e}")

    print(f"=== Daily refresh finished: {dt.datetime.utcnow().isoformat()} UTC ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
