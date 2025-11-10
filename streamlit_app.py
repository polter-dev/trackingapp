"""Streamlit dashboard for tracking equities and macroeconomic indicators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as pdr

try:
    import yfinance as yf
    HAS_YFINANCE = True
    try:
        from yfinance import pdr_override as _yf_pdr_override
    except (ImportError, AttributeError):
        _yf_pdr_override = getattr(yf, "pdr_override", None)
except ModuleNotFoundError:
    yf = None
    _yf_pdr_override = None
    HAS_YFINANCE = False
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
import warnings

# Silence warnings from statsmodels when fitting simple ARIMA models
warnings.simplefilter("ignore", ValueWarning)

DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "VOO", "^GSPC"]
CACHE_TTL_SECONDS = 60 * 60  # refresh cached data every hour
MACRO_SERIES: Dict[str, str] = {
    "Personal Savings Rate": "PSAVERT",
    "Unemployment Rate": "UNRATE",
    "Federal Funds Rate": "FEDFUNDS",
    "Consumer Price Index (CPI)": "CPIAUCSL",
}


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_equity_prices(tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
    """Download adjusted close prices for the requested tickers."""
    tickers = list(dict.fromkeys(tickers))  # drop duplicates while preserving order
    if not tickers:
        return pd.DataFrame()

    if HAS_YFINANCE:
        raw = yf.download(
            tickers,
            start=start,
            end=end + timedelta(days=1),  # include end date
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )

        if len(tickers) == 1:
            data = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        else:
            data = raw.xs("Adj Close", axis=1, level=1)
    else:
        frames = []
        for ticker in tickers:
            try:
                # Stooq returns data in reverse chronological order
                stooq = pdr.DataReader(ticker, "stooq", start, end + timedelta(days=1)).sort_index()
            except Exception as exc:
                st.warning(f"Unable to load {ticker} from Stooq: {exc}")
                continue
            frames.append(stooq[["Close"]].rename(columns={"Close": ticker}))

        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, axis=1).sort_index()

    data.index.name = "Date"
    return data.dropna(how="all")


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_macro_series(symbol: str, start: date, end: date) -> pd.Series:
    """Fetch a macroeconomic time series from FRED."""
    try:
        series = pdr.DataReader(symbol, "fred", start, end)
    except Exception as exc:
        st.warning(f"Unable to load {symbol} from FRED: {exc}")
        return pd.Series(dtype=float)

    return series[symbol].rename(symbol)


def to_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily prices to monthly percentage returns."""
    if price_df.empty:
        return price_df

    monthly_prices = price_df.resample("M").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")
    return monthly_returns


def summarize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute average return, volatility, and cumulative return per ticker."""
    if returns.empty:
        return pd.DataFrame(columns=["Ticker", "Average Return", "Volatility", "Cumulative Return"])

    cumulative = (1 + returns).cumprod().iloc[-1] - 1
    summary = pd.DataFrame(
        {
            "Ticker": returns.columns,
            "Average Return": (returns.mean() * 252).values,
            "Volatility": (returns.std() * np.sqrt(252)).values,
            "Cumulative Return": cumulative.values,
        }
    )
    return summary.set_index("Ticker")


def describe_price_trend(series: pd.Series, lookback_days: int = 90) -> str:
    """Generate a human-readable description of a price series."""
    clean = series.dropna()
    if clean.empty:
        return f"{series.name}: Not enough data to describe the recent trend."

    latest_date = clean.index[-1]
    window_start = latest_date - timedelta(days=lookback_days)
    window = clean[clean.index >= window_start]
    if len(window) < 2:
        window = clean.tail(min(lookback_days, len(clean)))
    if len(window) < 2:
        return f"{series.name}: Not enough observations for a recent trend summary."

    days = (window.index[-1] - window.index[0]).days or 1
    change = window.iloc[-1] / window.iloc[0] - 1
    slope = np.polyfit(np.arange(len(window)), window.values, 1)[0]
    direction = "has been rising" if slope > 0 else "has been drifting lower"
    if abs(slope) < 1e-6:
        direction = "has been relatively flat"

    change_text = f"{change:+.2%}" if np.isfinite(change) else "an unclear"
    return (
        f"{series.name}: {direction} over the past {days} days with a {change_text} move. "
        f"Most recent close: {window.iloc[-1]:,.2f}."
    )


def describe_macro_trend(series_name: str, series: pd.Series) -> str:
    """Generate a human-readable description of a macro series."""
    clean = series.dropna()
    if clean.empty:
        return f"{series_name}: No recent data available."

    recent = clean.last("365D")
    if len(recent) < 2:
        recent = clean.tail(min(6, len(clean)))
    if len(recent) < 2:
        return f"{series_name}: Not enough observations for a recent trend summary."

    change = recent.iloc[-1] - recent.iloc[0]
    pct_change = recent.iloc[-1] / recent.iloc[0] - 1 if recent.iloc[0] != 0 else np.nan
    trend = "increased" if change > 0 else "decreased"
    if abs(change) < 1e-9:
        trend = "remained broadly unchanged"
    pct_text = f" ({pct_change:+.2%})" if np.isfinite(pct_change) else ""
    return (
        f"{series_name}: {trend} over the last {len(recent)} observations to {recent.iloc[-1]:,.2f}{pct_text}."
    )


@dataclass
class ForecastResult:
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def forecast_prices(series: pd.Series, steps: int = 30) -> Optional[ForecastResult]:
    """Use an ARIMA(1,1,1) model to forecast future prices."""
    clean_series = series.dropna()
    if clean_series.empty or len(clean_series) < 40:
        return None

    try:
        model = ARIMA(clean_series, order=(1, 1, 1))
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=steps)
        forecast_index = pd.date_range(clean_series.index[-1] + timedelta(days=1), periods=steps, freq="B")
        forecast = pd.Series(forecast_result.predicted_mean.values, index=forecast_index, name="Forecast")
        conf_int = forecast_result.conf_int(alpha=0.2)
        lower = pd.Series(conf_int.iloc[:, 0].values, index=forecast_index, name="Lower")
        upper = pd.Series(conf_int.iloc[:, 1].values, index=forecast_index, name="Upper")
        return ForecastResult(forecast=forecast, lower=lower, upper=upper)
    except Exception as exc:
        st.warning(f"Unable to build forecast for {series.name}: {exc}")
        return None


def plot_price_history(price_df: pd.DataFrame) -> None:
    if price_df.empty:
        st.info("No price data to display.")
        return

    chart_data = price_df.reset_index().melt("Date", var_name="Ticker", value_name="Price")
    chart = (
        alt.Chart(chart_data)
        .mark_line()
        .encode(x="Date:T", y="Price:Q", color="Ticker:N")
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def plot_returns(returns: pd.DataFrame) -> None:
    if returns.empty:
        st.info("No returns data to display.")
        return

    chart_data = returns.reset_index().melt("Date", var_name="Ticker", value_name="Return")
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(x="Date:T", y="Return:Q", color="Ticker:N")
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def plot_forecast(ticker: str, series: pd.Series, forecast: ForecastResult) -> None:
    history_df = series.dropna().reset_index()
    history_df.columns = ["Date", "Price"]

    forecast_df = pd.DataFrame(
        {
            "Date": forecast.forecast.index,
            "Price": forecast.forecast.values,
            "Lower": forecast.lower.values,
            "Upper": forecast.upper.values,
        }
    )

    base = alt.Chart(history_df).mark_line(color="#1f77b4").encode(x="Date:T", y="Price:Q")
    future_line = alt.Chart(forecast_df).mark_line(color="#ff7f0e").encode(x="Date:T", y="Price:Q")
    band = (
        alt.Chart(forecast_df)
        .mark_area(opacity=0.2)
        .encode(x="Date:T", y="Lower:Q", y2="Upper:Q")
    )

    st.altair_chart((base + band + future_line).properties(title=f"{ticker} Price Forecast"), use_container_width=True)


def plot_macro_series(series_name: str, series: pd.Series) -> None:
    if series.empty:
        st.info(f"No data available for {series_name}.")
        return

    df = series.reset_index()
    df.columns = ["Date", series_name]
    chart = (
        alt.Chart(df)
        .mark_line(color="#2ca02c")
        .encode(x="Date:T", y=f"{series_name}:Q")
        .properties(title=series_name, height=300)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def compute_macro_correlations(
    monthly_returns: pd.DataFrame, macro_data: Dict[str, pd.Series]
) -> pd.DataFrame:
    results = []
    for name, series in macro_data.items():
        if series.empty:
            continue
        macro_changes = series.resample("M").last().pct_change().dropna()
        aligned = monthly_returns.join(macro_changes.rename("Macro"), how="inner")
        if aligned.empty:
            continue
        correlations = aligned.corr()["Macro"].drop("Macro")
        for ticker, value in correlations.items():
            results.append({"Macro Indicator": name, "Ticker": ticker, "Correlation": value})

    if not results:
        return pd.DataFrame(columns=["Macro Indicator", "Ticker", "Correlation"])

    return pd.DataFrame(results).sort_values(by="Correlation", ascending=False)


def parse_custom_tickers(raw_text: str) -> List[str]:
    """Normalize custom ticker text input into uppercase symbols."""
    if not raw_text:
        return []

    parts = re.split(r"[\s,;]+", raw_text)
    cleaned: List[str] = []
    for token in parts:
        symbol = token.strip().upper()
        if symbol and symbol not in cleaned:
            cleaned.append(symbol)
    return cleaned


def summarize_market_snapshot(price_df: pd.DataFrame) -> List[str]:
    """Produce a small set of highlights for the latest trading session."""
    clean = price_df.dropna(how="all")
    if len(clean) < 2:
        return []

    latest = clean.iloc[-1]
    previous = clean.iloc[-2]
    daily_returns = (latest / previous - 1).replace([np.inf, -np.inf], np.nan).dropna()
    if daily_returns.empty:
        return []

    highlights: List[str] = []
    top_gainer = daily_returns.idxmax()
    top_loser = daily_returns.idxmin()
    highlights.append(
        f"Top gainer: {top_gainer} {daily_returns[top_gainer]:+.2%} since the prior close."
    )
    if top_loser != top_gainer:
        highlights.append(
            f"Biggest pullback: {top_loser} {daily_returns[top_loser]:+.2%} over the same window."
        )

    broad_move = daily_returns.mean()
    highlights.append(
        "Average move across tracked assets: "
        f"{broad_move:+.2%} (simple average of daily percentage changes)."
    )
    return highlights


def render_dashboard() -> None:
    st.set_page_config(page_title="Market & Macro Tracker", layout="wide")
    st.title("Market & Macro Tracker")
    st.caption(
        "Interactive dashboard that monitors equities, creates time-series forecasts, "
        "and relates performance to macroeconomic trends."
    )

    if not HAS_YFINANCE:
        st.info(
            "`yfinance` is not installed. Using Stooq prices via pandas-datareader instead. "
            "Install `yfinance` for broader ticker coverage."
        )

    with st.sidebar:
        st.header("Configuration")
        selected_tickers = st.multiselect(
            "Select tickers (stocks / ETFs)",
            options=DEFAULT_TICKERS,
            default=DEFAULT_TICKERS[:4],
        )

        custom_ticker_text = st.text_input(
            "Add custom tickers",
            placeholder="e.g. NVDA, TSLA",
            help="Enter comma or space separated ticker symbols to augment the default list.",
        )
        custom_tickers = parse_custom_tickers(custom_ticker_text)

        today = date.today()
        default_start = today - timedelta(days=5 * 365)
        start_date = st.date_input("Start date", value=default_start, max_value=today - timedelta(days=7))
        end_date = st.date_input("End date", value=today, max_value=today)
        forecast_horizon = st.slider("Forecast horizon (trading days)", min_value=5, max_value=90, value=30)
        selected_macro = st.multiselect(
            "Macroeconomic indicators",
            list(MACRO_SERIES.keys()),
            default=list(MACRO_SERIES.keys()),
        )

        if st.button("Refresh market & macro data"):
            st.cache_data.clear()
            st.experimental_rerun()

    all_tickers = list(dict.fromkeys([*selected_tickers, *custom_tickers]))

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    if not all_tickers:
        st.warning("Select at least one ticker to visualize market data.")
        return

    with st.spinner("Downloading price data..."):
        price_df = load_equity_prices(all_tickers, start_date, end_date)

    if price_df.empty:
        st.warning("No data retrieved for the selected tickers and date range.")
        return

    st.subheader("Adjusted Close Prices")
    last_available = price_df.dropna(how="all").index.max()
    if last_available:
        st.caption(
            "Prices refresh through the latest available trading session "
            f"(last close: {last_available.date()}). Cached data auto-refreshes every "
            f"{CACHE_TTL_SECONDS // 60} minutes or whenever you press the refresh button in the sidebar."
        )
    else:
        st.caption(
            "Prices refresh through the latest available trading session. Use the sidebar to update the date range."
        )
    plot_price_history(price_df)

    highlights = summarize_market_snapshot(price_df)
    if highlights:
        st.markdown("#### Session Highlights")
        for bullet in highlights:
            st.markdown(f"- {bullet}")

    st.markdown("#### Recent Price Trend Summaries")
    for ticker in all_tickers:
        if ticker not in price_df.columns:
            st.write(f"{ticker}: No price data available for the selected period.")
            continue
        st.write(describe_price_trend(price_df[ticker]))

    returns = price_df.pct_change().dropna(how="all")

    st.subheader("Daily Returns")
    plot_returns(returns)

    summary = summarize_returns(returns)
    if not summary.empty:
        st.subheader("Annualized Performance Summary")
        styled = summary.style.format({
            "Average Return": "{:+.2%}",
            "Volatility": "{:.2%}",
            "Cumulative Return": "{:+.2%}",
        })
        st.dataframe(styled, use_container_width=True)

    st.subheader("Price Forecasts")
    st.caption("ARIMA-based projections illustrate a plausible range for the next trading days.")
    for ticker in all_tickers:
        if ticker not in price_df.columns:
            continue
        series = price_df[ticker]
        forecast_result = forecast_prices(series, steps=forecast_horizon)
        if forecast_result:
            plot_forecast(ticker, series, forecast_result)
            latest_price = series.dropna().iloc[-1]
            projected = forecast_result.forecast.iloc[-1]
            delta = projected / latest_price - 1 if latest_price else np.nan
            delta_text = f"{delta:+.2%}" if np.isfinite(delta) else "an unclear"
            st.info(
                f"Projected {forecast_horizon}-day price for {ticker}: {projected:,.2f} "
                f"(vs. latest close {latest_price:,.2f}, change {delta_text})."
            )
        else:
            st.info(f"Not enough data to forecast {ticker} or model did not converge.")

    st.subheader("Macroeconomic Context")
    macro_data: Dict[str, pd.Series] = {}
    for name in selected_macro:
        symbol = MACRO_SERIES[name]
        series = load_macro_series(symbol, start_date, end_date)
        macro_data[name] = series
        with st.expander(name, expanded=False):
            plot_macro_series(name, series)
            st.caption(describe_macro_trend(name, series))

    monthly_returns = to_monthly_returns(price_df)
    correlation_table = compute_macro_correlations(monthly_returns, macro_data)
    if not correlation_table.empty:
        st.markdown("#### Correlation with Macro Indicators")
        st.dataframe(correlation_table.style.format({"Correlation": "{:+.2f}"}), use_container_width=True)
    else:
        st.info("Insufficient overlapping data to compute correlations with macro indicators.")


if __name__ == "__main__":
    if HAS_YFINANCE and callable(_yf_pdr_override):
        _yf_pdr_override()
    render_dashboard()