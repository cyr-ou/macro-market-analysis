import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

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

# --------- Save charts to <project_root>/visuals (robust) ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root
VIS_DIR = os.path.join(PROJECT_ROOT, "visuals")
os.makedirs(VIS_DIR, exist_ok=True)

# 1) Gold Price
plt.figure(figsize=(10, 6))
plt.plot(prices["GOLD"])
plt.title("Gold Price")
plt.savefig(os.path.join(VIS_DIR, "gold_price.png"), dpi=200, bbox_inches="tight")
plt.close()

# 2) Rolling Correlation
plt.figure(figsize=(10, 6))
plt.plot(rolling_corr)
plt.title("60-Day Rolling Correlation: Gold vs DXY")
plt.savefig(os.path.join(VIS_DIR, "rolling_correlation.png"), dpi=200, bbox_inches="tight")
plt.close()

# 3) Rolling Volatility
plt.figure(figsize=(10, 6))
plt.plot(rolling_vol["GOLD"], label="Gold")
plt.plot(rolling_vol["DXY"], label="DXY")
plt.plot(rolling_vol["US10Y"], label="US10Y")
plt.legend()
plt.title("30-Day Rolling Volatility")
plt.savefig(os.path.join(VIS_DIR, "rolling_volatility.png"), dpi=200, bbox_inches="tight")
plt.close()

print(f"\n✅ Charts saved to: {VIS_DIR}")

# Save cleaned dataset
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
returns.to_csv(os.path.join(DATA_DIR, "returns.csv"))