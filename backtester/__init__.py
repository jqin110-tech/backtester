"""A small, vectorized research backtester.

Modules
-------
data        download daily prices from Yahoo Finance, with a local CSV cache
strategies  turn prices into a signal (what we WANT to hold after today's close)
backtest    turn a signal into daily strategy returns and equity curves
ml          features, target and walk-forward validation for the volatility model

Run from the repo root:  python -m backtester --help
"""
