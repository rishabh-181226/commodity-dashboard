"""
forecasting.py
----------------
Lightweight, dependency-light analytics for the dashboard:
  - rolling volatility
  - ARIMA short-term forecast with confidence bands
  - simple moving-average crossover backtester

Kept intentionally simple and well-commented -- the goal of this project is
to demonstrate clear understanding of the analytics, not to build a
production trading model. Every function returns a plain pandas DataFrame
so it's easy to plug into Plotly charts in app.py.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def rolling_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Compute annualized rolling volatility of daily log returns.
    df must have columns: date, price
    """
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["log_return"] = np.log(out["price"] / out["price"].shift(1))
    out[f"volatility_{window}d"] = out["log_return"].rolling(window).std() * np.sqrt(252)
    return out.dropna(subset=[f"volatility_{window}d"])[["date", f"volatility_{window}d"]].rename(
        columns={f"volatility_{window}d": "volatility"}
    )


def price_percentile_bands(df: pd.DataFrame, lookback_days: int = 365) -> dict:
    """Return current price percentile rank vs. trailing lookback window."""
    recent = df.sort_values("date").tail(lookback_days)
    current_price = recent["price"].iloc[-1]
    percentile = (recent["price"] < current_price).mean() * 100
    return {
        "current_price": current_price,
        "percentile": percentile,
        "low": recent["price"].min(),
        "high": recent["price"].max(),
        "median": recent["price"].median(),
    }


def arima_forecast(df: pd.DataFrame, periods: int = 30, order: tuple = (2, 1, 2)) -> pd.DataFrame:
    """
    Fit a simple ARIMA model and forecast `periods` days ahead with a 95% CI.
    df must have columns: date, price

    NOTE: This is intentionally a simple statistical baseline (not a trading
    signal). It's meant to demonstrate a forecasting workflow end-to-end,
    clearly labeled as illustrative in the UI.
    """
    from statsmodels.tsa.arima.model import ARIMA

    series = df.sort_values("date").set_index("date")["price"]
    series = series.asfreq("D").interpolate()

    model = ARIMA(series, order=order)
    fitted = model.fit()

    forecast_res = fitted.get_forecast(steps=periods)
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D")

    result = pd.DataFrame(
        {
            "date": future_dates,
            "forecast": mean_forecast.values,
            "lower_ci": conf_int.iloc[:, 0].values,
            "upper_ci": conf_int.iloc[:, 1].values,
        }
    )
    return result


def moving_average_backtest(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> dict:
    """
    Naive long-only moving-average crossover backtest:
      - Go long when short MA crosses above long MA
      - Go flat when short MA crosses below long MA

    Returns summary stats + an equity curve dataframe for charting.
    This is a simple, transparent baseline strategy for illustration --
    not a real trading recommendation.
    """
    data = df.sort_values("date").reset_index(drop=True).copy()
    data["short_ma"] = data["price"].rolling(short_window).mean()
    data["long_ma"] = data["price"].rolling(long_window).mean()
    data = data.dropna().reset_index(drop=True)

    data["signal"] = np.where(data["short_ma"] > data["long_ma"], 1, 0)
    data["position"] = data["signal"].shift(1).fillna(0)  # trade on next day's open

    data["daily_return"] = data["price"].pct_change().fillna(0)
    data["strategy_return"] = data["daily_return"] * data["position"]

    data["buy_hold_equity"] = (1 + data["daily_return"]).cumprod()
    data["strategy_equity"] = (1 + data["strategy_return"]).cumprod()

    total_return_strategy = data["strategy_equity"].iloc[-1] - 1
    total_return_buy_hold = data["buy_hold_equity"].iloc[-1] - 1

    n_days = len(data)
    ann_factor = 252 / n_days if n_days > 0 else 0
    cagr_strategy = (data["strategy_equity"].iloc[-1]) ** ann_factor - 1 if n_days > 0 else 0
    strategy_vol = data["strategy_return"].std() * np.sqrt(252)
    sharpe = (data["strategy_return"].mean() * 252) / strategy_vol if strategy_vol > 0 else 0

    n_trades = int((data["position"].diff().abs() > 0).sum())

    summary = {
        "total_return_strategy": total_return_strategy,
        "total_return_buy_hold": total_return_buy_hold,
        "annualized_return_strategy": cagr_strategy,
        "annualized_vol_strategy": strategy_vol,
        "sharpe_ratio": sharpe,
        "n_trades": n_trades,
    }
    equity_curve = data[["date", "buy_hold_equity", "strategy_equity"]]
    return {"summary": summary, "equity_curve": equity_curve}
