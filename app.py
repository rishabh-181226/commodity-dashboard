"""
app.py
-------
Commodity Price & Forward Curve Dashboard
A portfolio project demonstrating energy commodity market analytics
workflows (inspired by tools like Enverus MarketView / Sphere).

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud by pointing it at this repo.
"""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import db
from data_sources import get_eia_history, get_forward_curve, get_rig_count, get_usd_index
from forecasting import (
    rolling_volatility,
    price_percentile_bands,
    arima_forecast,
    moving_average_backtest,
)
from refresh_data import refresh_commodity

db.init_db()

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Energy Commodity Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2d3140;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .disclaimer {
        font-size: 0.82rem;
        color: #9aa0ac;
        border-left: 3px solid #4a5568;
        padding-left: 10px;
        margin-top: 6px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("🛢️ Controls")

commodity = st.sidebar.selectbox(
    "Commodity",
    ["WTI Crude (Cushing)", "Brent Crude", "Henry Hub Natural Gas"],
)

lookback_days = st.sidebar.slider("Lookback window (days)", 90, 1825, 730, step=30)

# The EIA API key is read from Streamlit secrets (set in Streamlit Cloud's
# "Secrets" panel when deployed, or in .streamlit/secrets.toml locally) so it
# never appears in the UI or in the source code. A manual override is
# available for local development/debugging only, tucked away in a collapsed
# expander so visitors never see an "enter your API key" prompt.
try:
    api_key = st.secrets.get("EIA_API_KEY", "") or None
except Exception:
    api_key = None

with st.sidebar.expander("⚙️ Developer options", expanded=False):
    _manual_key = st.text_input(
        "Override EIA API key (local testing only)",
        type="password",
        help="Leave blank in production — the deployed app reads the key from "
             "Streamlit secrets automatically. This field is only for testing "
             "a different key locally without editing secrets.toml.",
    )
    if _manual_key:
        api_key = _manual_key

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Refresh data now", use_container_width=True):
    with st.spinner(f"Refreshing {commodity}..."):
        result = refresh_commodity(commodity, api_key)
    st.session_state["_last_refresh_result"] = result
    st.cache_data.clear()  # drop cached reads so the app re-pulls from the DB immediately
    st.rerun()

if "_last_refresh_result" in st.session_state:
    r = st.session_state["_last_refresh_result"]
    st.sidebar.success(
        f"Refreshed {r['commodity']}: +{r['rows_added']} new row(s), "
        f"source={r['source']}, forecast={r['forecast_status']}"
    )

last_refresh = db.get_last_refresh(commodity)
if last_refresh:
    refreshed_at = dt.datetime.fromisoformat(last_refresh["last_refreshed_at"])
    st.sidebar.caption(f"🕒 Last refreshed: {refreshed_at.strftime('%Y-%m-%d %H:%M')} UTC")
else:
    st.sidebar.caption("🕒 Not refreshed yet — will bootstrap on first load.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built as a portfolio project demonstrating energy commodity analytics "
    "workflows — historical pricing, forward curves, volatility, and "
    "forecasting — similar in spirit to Enverus MarketView / Sphere."
)

# ---------------------------------------------------------------------------
# Data loading — reads from the local database, which the daily refresh job
# (refresh_data.py) keeps up to date. If the database is empty (first-ever
# run, e.g. right after cloning the repo), bootstrap it automatically so the
# app is never stuck showing nothing.
# ---------------------------------------------------------------------------

if not db.has_data(commodity):
    with st.spinner(f"First-time setup: building initial price history for {commodity}..."):
        refresh_commodity(commodity, api_key or None)

price_df = db.get_prices(commodity, days=lookback_days)
is_live = db.get_latest_source(commodity) == "eia_live"

if price_df.empty:
    st.error("No price data available yet. Click '🔄 Refresh data now' in the sidebar.")
    st.stop()

if not is_live:
    st.info(
        "📡 Running on **synthetic demo data**. This deployment doesn't have a live EIA API key "
        "configured yet — see the README's deployment guide for adding one via Streamlit secrets. "
        "The rest of the app works identically either way.",
        icon="ℹ️",
    )

spot_price = float(price_df["price"].iloc[-1])
curve_df, curve_is_live = get_forward_curve(commodity, spot_price)
rig_df, _ = get_rig_count(days=lookback_days)
usd_df = get_usd_index(days=lookback_days)

# ---------------------------------------------------------------------------
# Header + key metrics
# ---------------------------------------------------------------------------

st.title("Energy Commodity Price & Forward Curve Dashboard")
st.caption(f"Last updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · Data source: "
           f"{'EIA API (live)' if is_live else 'Synthetic demo data'}")

bands = price_percentile_bands(price_df, lookback_days=min(365, lookback_days))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${bands['current_price']:.2f}")
col2.metric("52-wk Percentile", f"{bands['percentile']:.0f}th")
col3.metric("52-wk Range", f"${bands['low']:.2f} – ${bands['high']:.2f}")
col4.metric("52-wk Median", f"${bands['median']:.2f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_hist, tab_curve, tab_vol, tab_corr, tab_forecast, tab_backtest = st.tabs(
    ["📈 Historical Prices", "🧮 Forward Curve", "📊 Volatility", "🔗 Correlation",
     "🔮 Forecast", "🧪 Backtester"]
)

# --- Historical Prices ---
with tab_hist:
    st.subheader(f"{commodity} — Historical Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df["date"], y=price_df["price"], mode="lines",
                              line=dict(color="#4dd0e1", width=2), name=commodity))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450, margin=dict(l=10, r=10, t=30, b=10),
                       yaxis_title="Price ($)", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw data"):
        st.dataframe(price_df.sort_values("date", ascending=False), use_container_width=True, height=300)

# --- Forward Curve ---
with tab_curve:
    st.subheader(f"{commodity} — Forward Curve")
    st.caption(
        "Illustrative forward curve. Yahoo Finance doesn't expose a free multi-month "
        "curve directly, so this blends the latest front-month quote with a modeled "
        "term structure. Swap in a licensed futures data feed for production use."
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve_df["contract_month"], y=curve_df["price"], mode="lines+markers",
                              line=dict(color="#ffb74d", width=2), name="Forward Curve"))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=30, b=10),
                       yaxis_title="Price ($)", xaxis_title="Contract Month")
    st.plotly_chart(fig, use_container_width=True)

    shape = "Contango (upward sloping)" if curve_df["price"].iloc[-1] > curve_df["price"].iloc[0] else \
            "Backwardation (downward sloping)"
    st.markdown(f"**Curve shape:** {shape}")

# --- Volatility ---
with tab_vol:
    st.subheader(f"{commodity} — Rolling Volatility")
    window = st.slider("Volatility window (days)", 10, 90, 30, key="vol_window")
    vol_df = rolling_volatility(price_df, window=window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_df["date"], y=vol_df["volatility"] * 100, mode="lines",
                              line=dict(color="#ef5350", width=2), name="Annualized Volatility"))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=30, b=10),
                       yaxis_title="Annualized Volatility (%)")
    st.plotly_chart(fig, use_container_width=True)

# --- Correlation ---
with tab_corr:
    st.subheader("Price vs. Macro / Activity Drivers")

    merged = price_df.merge(rig_df, on="date", how="inner")
    merged = merged.merge(usd_df, on="date", how="inner")

    metric_choice = st.radio("Compare price against:", ["Rig Count", "USD Index"], horizontal=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["price"], mode="lines",
                              name=commodity, line=dict(color="#4dd0e1"), yaxis="y1"))
    if metric_choice == "Rig Count":
        fig.add_trace(go.Scatter(x=merged["date"], y=merged["rig_count"], mode="lines",
                                  name="Rig Count", line=dict(color="#ba68c8"), yaxis="y2"))
        y2_title = "Rig Count"
    else:
        fig.add_trace(go.Scatter(x=merged["date"], y=merged["usd_index"], mode="lines",
                                  name="USD Index", line=dict(color="#ba68c8"), yaxis="y2"))
        y2_title = "USD Index"

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=450, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title=f"{commodity} ($)"),
        yaxis2=dict(title=y2_title, overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    corr_col = "rig_count" if metric_choice == "Rig Count" else "usd_index"
    corr_val = merged["price"].corr(merged[corr_col])
    st.metric(f"Correlation with {metric_choice}", f"{corr_val:.2f}")
    st.caption("Rig count data is synthetic (Baker Hughes has no stable free CSV endpoint — "
               "swap in a licensed feed for production). USD Index is also synthetic; "
               "replace with a live FRED series (DTWEXBGS) for real analysis.")

# --- Forecast ---
with tab_forecast:
    st.subheader(f"{commodity} — Short-Term Forecast (ARIMA)")
    st.markdown(
        '<div class="disclaimer">Illustrative statistical forecast only — not investment advice. '
        "ARIMA is a simple baseline model shown here to demonstrate a forecasting workflow "
        "end-to-end, not a production trading signal.</div>",
        unsafe_allow_html=True,
    )

    cached_forecast, generated_at = db.get_cached_forecast(commodity, max_age_hours=24)

    forecast_ok = False
    if cached_forecast is not None:
        st.success(
            f"✅ Using cached forecast from the daily refresh job — generated "
            f"{dt.datetime.fromisoformat(generated_at).strftime('%Y-%m-%d %H:%M')} UTC "
            f"(refreshed automatically once a day, so this reflects yesterday's or today's close)."
        )
        forecast_df = cached_forecast.rename(columns={"forecast": "forecast"})
        forecast_ok = True
    else:
        st.warning(
            "No fresh cached forecast found (either the daily refresh job hasn't run yet, or the "
            "cache is more than 24h old). Generating one live instead — click "
            "'🔄 Refresh data now' in the sidebar to also save it to the cache for next time."
        )
        horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
        with st.spinner("Fitting ARIMA model live..."):
            try:
                forecast_df = arima_forecast(price_df, periods=horizon)
                forecast_ok = True
            except Exception as e:
                st.error(f"Forecast failed: {e}")

    if forecast_ok:
        recent_hist = price_df.tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent_hist["date"], y=recent_hist["price"], mode="lines",
                                  name="Historical", line=dict(color="#4dd0e1")))
        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"], mode="lines",
                                  name="Forecast", line=dict(color="#ffd54f", dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
            y=list(forecast_df["upper_ci"]) + list(forecast_df["lower_ci"][::-1]),
            fill="toself", fillcolor="rgba(255,213,79,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True,
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=460, margin=dict(l=10, r=10, t=30, b=10),
                           yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

# --- Backtester ---
with tab_backtest:
    st.subheader("Moving Average Crossover — Illustrative Backtest")
    st.markdown(
        '<div class="disclaimer">Naive long/flat MA-crossover strategy for educational '
        "purposes only — not investment advice, no transaction costs modeled.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    short_w = c1.slider("Short MA window", 5, 50, 20)
    long_w = c2.slider("Long MA window", 20, 200, 50)

    if short_w >= long_w:
        st.warning("Short window should be smaller than long window.")
    else:
        result = moving_average_backtest(price_df, short_window=short_w, long_window=long_w)
        summary = result["summary"]
        equity = result["equity_curve"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy Return", f"{summary['total_return_strategy']*100:.1f}%")
        m2.metric("Buy & Hold Return", f"{summary['total_return_buy_hold']*100:.1f}%")
        m3.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
        m4.metric("# Trades", f"{summary['n_trades']}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["buy_hold_equity"], mode="lines",
                                  name="Buy & Hold", line=dict(color="#90a4ae")))
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["strategy_equity"], mode="lines",
                                  name="MA Crossover Strategy", line=dict(color="#66bb6a")))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=430, margin=dict(l=10, r=10, t=30, b=10),
                           yaxis_title="Growth of $1")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Built as a portfolio project demonstrating energy commodity analytics workflows. "
    "Data: EIA API (live when key provided), Yahoo Finance futures proxy, synthetic "
    "demo series otherwise. Not investment advice."
)
