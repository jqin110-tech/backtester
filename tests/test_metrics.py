"""Metrics on tiny series small enough to check on paper.

Each expected value has its arithmetic in a comment. Before running the
tests, verify one or two of them by hand.
"""

import math

import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (annualized_return, annualized_volatility, exposure,
                                max_drawdown, number_of_trades, sharpe_ratio,
                                total_return, win_rate)


def series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def returns_from_prices(prices):
    """Prices 100, 110, 99 -> returns +10%, -10% (dated on days 2 and 3)."""
    return series(prices).pct_change().dropna()


# ---- returns ------------------------------------------------------------

def test_total_return():
    # 1.1 * 0.9 - 1 = -0.01
    assert total_return(series([0.10, -0.10])) == pytest.approx(-0.01)


def test_annualized_return_two_years():
    # +21% over 504 days (2 years) -> 1.21 ** (252 / 504) - 1 = 1.1 - 1 = 10%
    daily = 1.21 ** (1 / 504) - 1
    assert annualized_return(series([daily] * 504)) == pytest.approx(0.10)


# ---- risk ---------------------------------------------------------------

def test_annualized_volatility():
    # +1%, -1%, +1%, -1%: mean 0, sum of squares 4e-4, sample var = 4e-4 / 3.
    # vol = 0.01 * sqrt(4/3) * sqrt(252) = 0.01 * sqrt(336)
    assert annualized_volatility(series([0.01, -0.01, 0.01, -0.01])) == pytest.approx(0.01 * math.sqrt(336))


def test_sharpe_with_zero_risk_free():
    # 2%, 0, 2%, 0: mean 0.01, sample std 0.01 * sqrt(4/3)
    # sharpe = sqrt(3/4) * sqrt(252) = sqrt(189)
    assert sharpe_ratio(series([0.02, 0.0, 0.02, 0.0])) == pytest.approx(math.sqrt(189))


def test_sharpe_subtracts_risk_free():
    # rf chosen so rf_daily is exactly 1%. Excess returns are 2%, 0, 2%, 0 -> same as above.
    rf_annual = 1.01 ** 252 - 1
    returns = series([0.03, 0.01, 0.03, 0.01])
    assert sharpe_ratio(returns, rf_annual=rf_annual) == pytest.approx(math.sqrt(189))


def test_sharpe_is_nan_when_volatility_is_zero():
    assert np.isnan(sharpe_ratio(series([0.01, 0.01, 0.01])))


# ---- drawdown -----------------------------------------------------------

def test_max_drawdown_simple():
    # 100 -> 110 -> 99: fall from 110 to 99 = 11 / 110 = 10%
    r = returns_from_prices([100, 110, 99])
    dd, peak, trough = max_drawdown(r)
    assert dd == pytest.approx(0.10)
    assert peak == r.index[0]       # the 110 day
    assert trough == r.index[1]     # the 99 day


def test_max_drawdown_picks_the_largest_fall():
    # 100, 90, 120, 60, 80. Falls: 100->90 (10%), 120->60 (50%). Answer: 50%.
    r = returns_from_prices([100, 90, 120, 60, 80])
    dd, peak, trough = max_drawdown(r)
    assert dd == pytest.approx(0.50)
    assert peak == r.index[1]       # the 120 day
    assert trough == r.index[2]     # the 60 day


def test_max_drawdown_from_starting_capital():
    # 100 -> 95 -> 98: loses money on day 1, so the peak is the starting capital.
    r = returns_from_prices([100, 95, 98])
    dd, peak, trough = max_drawdown(r)
    assert dd == pytest.approx(0.05)
    assert peak is None
    assert trough == r.index[0]


def test_max_drawdown_when_prices_only_rise():
    assert max_drawdown(returns_from_prices([100, 101, 102])) == (0.0, None, None)


# ---- trades and exposure -----------------------------------------------

POSITION = series([1, 1, 0, 1, 0, 1])
RETURNS = series([0.10, -0.05, 0.0, -0.02, 0.0, 0.03])
# Trade 1 (days 1-2): 1.10 * 0.95 - 1 = +4.5%  win
# Trade 2 (day 4):    -2%                       loss
# Trade 3 (day 6):    +3% (still open)          win


def test_number_of_trades_counts_round_trips():
    assert number_of_trades(POSITION) == 3


def test_win_rate():
    assert win_rate(POSITION, RETURNS) == pytest.approx(2 / 3)


def test_win_rate_is_nan_with_no_trades():
    assert np.isnan(win_rate(series([0, 0, 0]), series([0.01, -0.01, 0.02])))


def test_exposure():
    assert exposure(POSITION) == pytest.approx(4 / 6)
