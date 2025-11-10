# Market & Macro Tracker

This Streamlit app tracks equities and ETFs using freely available market data and relates them to macroeconomic indicators from FRED. It includes time-series analysis, ARIMA-based price projections, and correlation insights between asset returns and macro data.

## Features
- Download historical adjusted close prices for selected stocks and ETFs using `yfinance`.
- Visualize price histories, daily returns, and annualized performance metrics alongside plain-language trend summaries.
- Generate ARIMA (1,1,1) forecasts for future prices with confidence intervals.
- Include the S&P 500 index (^GSPC) as a default market benchmark.
- Retrieve macroeconomic indicators (personal savings rate, unemployment rate, federal funds rate, CPI) from FRED via `pandas-datareader`.
- Explore correlations between monthly asset returns and macroeconomic changes.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Use the sidebar to choose tickers, date ranges, forecast horizons, and macro indicators. The app will display charts, forecasts, and correlation tables based on your selections.

## Notes
- Some macroeconomic series are released monthly; ensure the selected date range includes sufficient overlap with the equity data to compute correlations.
- Forecasts are generated using a simple ARIMA model and should be interpreted as illustrative projections rather than investment advice.
- If `yfinance` is unavailable, the app automatically falls back to Stooq prices via `pandas-datareader`; installing `yfinance` is recommended for broader ticker coverage.