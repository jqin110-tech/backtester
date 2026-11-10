"""Performance metrics.

All inputs are pandas Series indexed by date, with the warm-up already
removed (exactly what run_backtest returns):
    returns  = daily simple returns, e.g. frame["strategy_return"]
    position = daily held position, 0 or 1, e.g. frame["position"]

Vectorized pandas/numpy only, no for-loops over rows. Each function is
checked against hand-computed answers in tests/test_metrics.py.

Risk-free assumption: sharpe_ratio defaults to rf = 0, which overstates
every Sharpe ratio a little (US T-bills averaged roughly 2-3% a year over
2018-2026). Pass rf_annual to change it. Related simplification: the
backtest earns 0% on cash days, not the T-bill rate, which understates a
strategy that sits in cash part of the time.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252   # trading days per year, used to annualize


def total_return(returns):
    """prod(1 + r_t) - 1"""
    return float((1 + returns).prod() - 1)


def annualized_return(returns):
    """Compound annual growth rate (CAGR): (1 + total_return) ** (252 / N) - 1,
    where N = number of daily returns."""
    n_days = len(returns)
    return float((1 + total_return(returns)) ** (TRADING_DAYS / n_days) - 1)


def annualized_volatility(returns):
    """Sample standard deviation of daily returns (ddof=1, the pandas default)
    times sqrt(252). Scaling by sqrt(T) assumes returns are independent from
    day to day: variances add, so standard deviations scale with sqrt(T)."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns, rf_annual=0.0):
    """Annualized Sharpe ratio of daily EXCESS returns:

        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        excess   = returns - rf_daily
        sharpe   = mean(excess) / std(excess) * sqrt(252)

    Default rf_annual = 0. Return np.nan if std(excess) is 0 (undefined).
    """
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    std = excess.std()
    # Floating-point noise can make a constant series' std ~1e-18, not 0.
    if not std > 1e-12:
        return np.nan
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


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
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    drawdown = 1 - equity / peak
    worst = float(drawdown.max())
    if worst <= 0:
        return 0.0, None, None

    trough_date = drawdown.idxmax()
    before_trough = equity.loc[:trough_date]
    # If equity never rose above the starting capital before the trough,
    # the peak was the starting capital itself (no date in the series).
    if before_trough.max() <= 1.0:
        peak_date = None
    else:
        peak_date = before_trough.idxmax()
    return worst, peak_date, trough_date


def _entries(position):
    """True on each day a new long position starts (cash -> long, or long
    on the very first day, since we assume we start in cash)."""
    previous = position.shift(1, fill_value=0)
    return (position == 1) & (previous == 0)


def number_of_trades(position):
    """Number of ROUND TRIPS = number of entries (cash -> long). Starting
    already long on the first day counts as an entry. A position still open
    on the last day counts too."""
    return int(_entries(position).sum())


def win_rate(position, returns):
    """Fraction of round trips with a positive compounded return.

    A trade is one unbroken run of days with position == 1. Its return is
    prod(1 + r) over those days, minus 1. An open last trade is marked to
    market. Return np.nan if there are no trades.

    Hint (vectorized): mark each new entry, cumsum() the marks to give every
    day a trade id, keep only invested days, then groupby(trade_id).
    """
    invested = position == 1
    if not invested.any():
        return np.nan
    trade_id = _entries(position).cumsum()
    growth = (1 + returns[invested]).groupby(trade_id[invested]).prod()
    trade_returns = growth - 1
    return float((trade_returns > 0).mean())


def exposure(position):
    """Fraction of days invested: mean(position)."""
    return float(position.mean())


def summarize(returns, position, rf_annual=0.0):
    """Every metric above for one return stream, as a dict (one table row)."""
    drawdown, peak_date, trough_date = max_drawdown(returns)
    return {
        "total_return": total_return(returns),
        "annual_return": annualized_return(returns),
        "annual_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, rf_annual),
        "max_drawdown": drawdown,
        "dd_peak": peak_date.date() if peak_date is not None else "start",
        "dd_trough": trough_date.date() if trough_date is not None else "-",
        "win_rate": win_rate(position, returns),
        "trades": number_of_trades(position),
        "exposure": exposure(position),
    }


def sharpe_standard_error(sharpe, n_days):
    """Approximate standard error of an ANNUALIZED Sharpe estimate from
    n_days of daily returns (Lo, 2002, assuming i.i.d. returns):

        SE(daily SR) = sqrt((1 + SR_daily**2 / 2) / n_days)
        SE(annual SR) = SE(daily SR) * sqrt(252)

    Rule of thumb it gives: 5 years of data -> SE about 0.45. So two
    strategies with Sharpe 0.9 and 1.2 over 5 years are NOT distinguishable.
    """
    sharpe_daily = sharpe / np.sqrt(TRADING_DAYS)
    return float(np.sqrt((1 + sharpe_daily ** 2 / 2) / n_days) * np.sqrt(TRADING_DAYS))
