"""Download daily prices from Yahoo Finance, with a local CSV cache.

Rerunning with the same ticker and dates reads the CSV instead of calling
Yahoo again. That makes reruns fast and lets them work offline. For research
it also means two runs see exactly the same data, so a change in results
comes from a change in code, not from Yahoo silently revising its history.
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

# data_cache/ sits next to the package, in the repo root (it is gitignored).
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_prices(ticker, start, end=None, refresh=False):
    """Return a DataFrame of daily Open/High/Low/Close/Volume for `ticker`.

    - Prices are split- and dividend-adjusted (auto_adjust=True), so
      Close.pct_change() is close to the total return an investor received.
    - `end` is EXCLUSIVE (the yfinance convention). If omitted, today's date
      is used, which excludes today's still-forming bar. The cache file name
      includes the dates, so a new day means a fresh download.
    - refresh=True ignores the cache and downloads again.
    """
    ticker = ticker.upper()
    if end is None:
        end = dt.date.today().isoformat()

    path = CACHE_DIR / f"{ticker}_{start}_{end}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker!r} between {start} and {end}.")

    # Newer yfinance returns 2-level columns like ("Close", "MU"). Flatten them.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    prices = raw[COLUMNS].dropna(subset=["Close"])
    prices.index.name = "Date"

    CACHE_DIR.mkdir(exist_ok=True)
    prices.to_csv(path)
    return prices
