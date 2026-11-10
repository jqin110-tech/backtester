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
