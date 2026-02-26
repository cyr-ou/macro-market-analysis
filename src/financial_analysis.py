import yfinance as yf
import pandas as pd

# 1) Download daily data
start_date = "2020-01-01"

tickers = {
    "GOLD": "GC=F",       # Gold Futures
    "DXY": "DX-Y.NYB",    # US Dollar Index
    "US10Y": "^TNX"       # US 10Y Yield (in % * 10, e.g., 45.0 = 4.5%)
}

data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start=start_date, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {name} ({ticker}).")
    series = df["Close"].copy()
    series.name = name
    data[name] = series

# 2) Align into one DataFrame
prices = pd.concat(data.values(), axis=1, sort=False).dropna()
prices.columns = ["GOLD", "DXY", "US10Y"]  # Set column names to asset names

# 3) Compute daily returns
returns = prices.pct_change().dropna()

# 4) Summary
print("\n=== Price head ===")
print(prices.head())

print("\n=== Return head ===")
print(returns.head())

print("\n=== Correlation (returns) ===")
print(returns.corr())

# 5) Rolling correlation example: Gold vs DXY (60-day)
rolling_corr = returns["GOLD"].rolling(60).corr(returns["DXY"])
print("\n=== Rolling Corr (GOLD vs DXY) last 10 ===")
print(rolling_corr.dropna().tail(10))

# 6) Rolling volatility (30-day)
rolling_vol = returns.rolling(30).std()
print("\n=== Rolling Vol (30d) last 5 ===")
print(rolling_vol.tail(5))

# 7) Rolling Sharpe Ratio (30-day, risk-free rate = 0)
import numpy as np

mean_daily_return = returns.mean()
annual_return = mean_daily_return * 252

annual_vol = returns.std() * np.sqrt(252)

sharpe_ratio = annual_return / annual_vol

print("\n=== Annualized Return ===")
print(annual_return)

print("\n=== Annualized Volatility ===")
print(annual_vol)

print("\n=== Sharpe Ratio ===")
print(sharpe_ratio)

# 8) Maximum Drawdown
cumulative = (1 + returns).cumprod()
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max

max_drawdown = drawdown.min()

print("\n=== Maximum Drawdown ===")
print(max_drawdown)

# 9) Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(prices["GOLD"], label="Gold")
plt.title("Gold Price")
plt.legend()
plt.show()

# 10) Regime Analysis
high_vol = rolling_vol["GOLD"] > rolling_vol["GOLD"].median()
low_vol = rolling_vol["GOLD"] <= rolling_vol["GOLD"].median()

print("Gold avg return during high vol:", returns["GOLD"][high_vol].mean())
print("Gold avg return during low vol:", returns["GOLD"][low_vol].mean())