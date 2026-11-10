"""Performance metrics. YOUR implementation goes here.

All inputs are pandas Series indexed by date, with the warm-up already
removed (exactly what run_backtest returns):
    returns  = daily simple returns, e.g. frame["strategy_return"]
    position = daily held position, 0 or 1, e.g. frame["position"]

Rules: vectorized pandas/numpy only (no for-loops over rows), and every
function must pass tests/test_metrics.py. Replace each NotImplementedError.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252   # trading days per year, used to annualize


def total_return(returns):
    """prod(1 + r_t) - 1"""
    raise NotImplementedError


def annualized_return(returns):
    """Compound annual growth rate (CAGR): (1 + total_return) ** (252 / N) - 1,
    where N = number of daily returns."""
    raise NotImplementedError


def annualized_volatility(returns):
    """Sample standard deviation of daily returns (ddof=1, the pandas default)
    times sqrt(252). Scaling by sqrt(T) assumes returns are independent from
    day to day: variances add, so standard deviations scale with sqrt(T)."""
    raise NotImplementedError


def sharpe_ratio(returns, rf_annual=0.0):
    """Annualized Sharpe ratio of daily EXCESS returns:

        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        excess   = returns - rf_daily
        sharpe   = mean(excess) / std(excess) * sqrt(252)

    Default rf_annual = 0. Return np.nan if std(excess) is 0 (undefined).
    """
    raise NotImplementedError


def max_drawdown(returns):
    """Largest peak-to-trough fall of the equity curve.

    Returns (drawdown, peak_date, trough_date), with drawdown as a POSITIVE
    fraction (0.10 means a 10% fall).

        equity   = cumprod(1 + r)
        peak     = running max of equity, never below 1.0 (the starting capital)
        drawdown = 1 - equity / peak         -> the max of this is the answer
        trough_date = date where that max occurs
        peak_date   = date of the running peak before the trough, or None if
                      the peak was the starting capital (lost money from day 1)

    If there is no drawdown at all, return (0.0, None, None).
    """
    raise NotImplementedError


def number_of_trades(position):
    """Number of ROUND TRIPS = number of entries (cash -> long). Starting
    already long on the first day counts as an entry. A position still open
    on the last day counts too."""
    raise NotImplementedError


def win_rate(position, returns):
    """Fraction of round trips with a positive compounded return.

    A trade is one unbroken run of days with position == 1. Its return is
    prod(1 + r) over those days, minus 1. An open last trade is marked to
    market. Return np.nan if there are no trades.

    Hint (vectorized): mark each new entry, cumsum() the marks to give every
    day a trade id, keep only invested days, then groupby(trade_id).
    """
    raise NotImplementedError


def exposure(position):
    """Fraction of days invested: mean(position)."""
    raise NotImplementedError
