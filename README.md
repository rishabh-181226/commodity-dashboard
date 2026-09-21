# Energy Commodity Price & Forward Curve Dashboard


A live analytics dashboard for energy commodity markets — WTI crude, Brent
crude, and Henry Hub natural gas — covering historical pricing, forward
curves, volatility, macro correlations, short-term forecasting, and a
trading strategy backtest. Built to mirror the kind of market analytics
workflows sold by energy data platforms like **Enverus MarketView** and
**Enverus Sphere**.

![dashboard preview placeholder](https://via.placeholder.com/900x400?text=Dashboard+Screenshot)

## What it does

| Feature | What you see |
|---|---|
| 📈 **Historical prices** | Multi-year daily price history, pulled live from the U.S. Energy Information Administration (EIA) |
| 🧮 **Forward curve** | Modeled term structure showing whether the market is in contango or backwardation |
| 📊 **Volatility** | Rolling annualized volatility of daily returns, with an adjustable lookback window |
| 🔗 **Correlation** | Price plotted against rig count and USD index to surface macro/activity relationships |
| 🔮 **Forecast** | A 30-day ARIMA forecast with a 95% confidence band, regenerated automatically every day |
| 🧪 **Backtester** | A moving-average crossover strategy tested against buy-and-hold, with Sharpe ratio and trade count |

Every number on the page is computed live from the underlying price series —
nothing is hardcoded. Switching the commodity or lookback window in the
sidebar recalculates the entire dashboard.

## Why this project exists

Energy analytics platforms all sell some version of the same promise: pull
scattered market data into one place, fast, so a trader or analyst can act
on it with confidence. This project rebuilds that promise end-to-end using
public data, to demonstrate three things at once:

- **Data engineering** — ingesting and cleaning a live external data feed
  (the EIA API) into a clean, queryable schema, with graceful fallback
  behavior when the feed is unavailable
- **Applied analytics** — volatility measurement, time-series forecasting
  (ARIMA), and a transparent backtesting framework, not just charts
- **Product thinking** — a dashboard someone could actually open every
  morning, not a one-off notebook

## How it's kept current

The dashboard doesn't just query an API on page load — it runs on its own
small data pipeline:

- A **SQLite database** stores daily prices, so history accumulates over
  time instead of being re-fetched from scratch on every visit
- A **scheduled job** (GitHub Actions, running once a day) pulls the
  newest price, updates the database, and re-fits the forecasting model —
  the same "ingest separately from serve" pattern real trading/analytics
  platforms use
- The app reads from the database on every load, so it's fast and stays
  usable even if the live data source is briefly down

## Architecture

```
commodity-dashboard/
├── app.py                 # Streamlit UI — tabs, charts, layout
├── data_sources.py        # EIA API integration + synthetic fallback generators
├── db.py                  # SQLite persistence (prices, cached forecasts, refresh log)
├── refresh_data.py        # Daily job: pulls new data, re-fits the forecast
├── forecasting.py         # Volatility, ARIMA forecasting, backtest logic
├── test_eia_connection.py # Standalone script to verify an EIA API key
└── .github/workflows/     # GitHub Actions cron job for the daily refresh
```

## Tech stack

Python · Streamlit · pandas · Plotly · statsmodels (ARIMA) · SQLite ·
EIA Open Data API · GitHub Actions

## Data source & disclaimer

Live pricing data comes from the [U.S. Energy Information Administration
Open Data API](https://www.eia.gov/opendata/). Forecasts and the
backtested trading strategy are simplified statistical illustrations for
demonstration purposes — not investment advice.

---

## Setup

Full step-by-step instructions for running locally, deploying to Streamlit
Community Cloud, and configuring the EIA API key are in
[`SETUP.md`](SETUP.md).
