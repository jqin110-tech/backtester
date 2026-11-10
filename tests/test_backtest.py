"""Backtest timing and bookkeeping, on tiny series you can check by hand."""

import numpy as np
import pandas as pd
import pytest

from backtester.backtest import run_backtest
from backtester.strategies import ma_crossover


def series(values):
    """Values on consecutive business days (dates don't matter for these tests)."""
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_position_is_previous_days_signal():
    # Returns: +10%, +10%, -10%. The signal turns to cash at the close of day 2,
    # so the strategy should still hold during day 2 and avoid only day 3's drop.
    close = series([100, 110, 121, 108.9])
    signal = series([1, 1, 0, 0])
    result = run_backtest(close, signal)
    assert result["strategy_return"].tolist() == pytest.approx([0.10, 0.10, 0.0])


def test_warmup_is_dropped_and_benchmark_uses_same_window():
    close = series(range(100, 110))            # 10 days
    signal = ma_crossover(close, fast=2, slow=3)
    result = run_backtest(close, signal)
    # First valid signal: row 2 (3-day MA needs 3 closes). First held position: row 3.
    assert result.index[0] == close.index[3]
    # Buy and hold is measured from the same starting close as the strategy (row 2).
    assert result["benchmark_equity"].iloc[-1] == pytest.approx(close.iloc[-1] / close.iloc[2])


def test_trade_count_has_no_phantom_first_trade():
    close = series([100] * 7)
    signal = series([np.nan, 1, 1, 0, 0, 1, 1])
    result = run_backtest(close, signal)
    # Held positions: 1, 1, 0, 0, 1 -> entry, exit, entry = 3 legs.
    assert result["trade"].sum() == 3


def test_ma_crossover_rejects_fast_not_less_than_slow():
    with pytest.raises(ValueError):
        ma_crossover(series(range(100)), fast=50, slow=20)


# ---- transaction costs ------------------------------------------------------

def test_costs_are_charged_per_leg():
    # Flat prices, so every bit of return is cost. Held positions 1, 1, 0, 1:
    # legs on days 1 (entry), 3 (exit), 4 (entry) -> 3 legs * 10 bp each.
    close = series([100] * 5)
    signal = series([1, 1, 0, 1, 1])
    result = run_backtest(close, signal, cost_bps=10)
    assert result["strategy_return"].tolist() == pytest.approx([-0.001, 0.0, -0.001, -0.001])


def test_buy_and_hold_pays_one_entry_cost():
    close = series([100] * 5)
    result = run_backtest(close, series([1] * 5), cost_bps=10)
    assert result["benchmark_return"].sum() == pytest.approx(-0.001)


def test_zero_cost_matches_frictionless():
    close = series([100, 105, 103, 108, 110, 107])
    signal = series([1, 0, 1, 1, 0, 1])
    assert run_backtest(close, signal, cost_bps=0)["strategy_return"].tolist() == \
        pytest.approx((signal.shift(1) * close.pct_change()).dropna().tolist())


def test_split_by_date_has_no_overlap_and_loses_no_rows():
    from backtester.backtest import split_by_date
    result = run_backtest(series(range(100, 120)), series([1] * 20))
    in_sample, out_of_sample = split_by_date(result, result.index[10])
    assert len(in_sample) + len(out_of_sample) == len(result)
    assert in_sample.index.max() < out_of_sample.index.min()
