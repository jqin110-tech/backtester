"""Parameter robustness: a fast x slow grid of Sharpe ratios, in-sample vs
out-of-sample.

Why "pick the best cell" is overfitting
---------------------------------------
Each cell's Sharpe is an ESTIMATE with noise. Over ~5 years its standard
error is about 0.45 (see metrics.sharpe_standard_error). Searching 40 cells
and keeping the maximum selects whichever cell got the luckiest noise, so the
winner's in-sample Sharpe is biased upward: E[max of N noisy estimates] >
max of the true values. The bias grows with the number of cells tried, even
though nobody "fit" anything in the usual sense. The honest test: choose on
the in-sample period only, then look ONCE at the untouched out-of-sample
period. If the ranking of cells in-sample says nothing about their ranking
out-of-sample (rank correlation near 0), the grid's structure is noise.

What robustness looks like instead: a broad plateau of similar Sharpe
across neighbouring cells that survives out-of-sample, rather than one
bright isolated cell.

The loop below is over PARAMETERS (a few dozen backtests), not over rows.
Each backtest is fully vectorized.
"""

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .metrics import sharpe_ratio
from .strategies import ma_crossover

FAST_WINDOWS = [5, 10, 15, 20, 30, 40, 50]
SLOW_WINDOWS = [20, 30, 50, 75, 100, 150, 200]


def sharpe_grids(close, split_date, cost_bps, rf_annual=0.0, fast_windows=FAST_WINDOWS,
                 slow_windows=SLOW_WINDOWS):
    """Return (in_sample_grid, out_of_sample_grid, is_start).

    Each grid is a DataFrame of Sharpe ratios: rows = fast, columns = slow.
    Cells with fast >= slow are NaN (not a crossover).

    Every cell is measured over the SAME dates. A 200-day slow window needs
    200 days of warm-up and a 20-day one doesn't, so without a common start
    the cells would be scored on different periods and wouldn't be
    comparable. In-sample therefore starts on the first day the LARGEST slow
    window can hold a position (is_start) and ends the day before split_date.
    """
    if len(close) <= max(slow_windows):
        raise ValueError("Not enough history for the largest slow window.")
    is_start = close.index[max(slow_windows)]
    split = pd.Timestamp(split_date)

    in_sample = pd.DataFrame(np.nan, index=fast_windows, columns=slow_windows, dtype=float)
    out_of_sample = in_sample.copy()
    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue
            frame = run_backtest(close, ma_crossover(close, fast, slow), cost_bps)
            returns = frame["strategy_return"]
            in_sample.loc[fast, slow] = sharpe_ratio(returns[(returns.index >= is_start) & (returns.index < split)], rf_annual)
            out_of_sample.loc[fast, slow] = sharpe_ratio(returns[returns.index >= split], rf_annual)

    for grid in (in_sample, out_of_sample):
        grid.index.name, grid.columns.name = "fast", "slow"
    return in_sample, out_of_sample, is_start


def best_cell(grid):
    """(fast, slow) of the highest Sharpe in the grid."""
    return _cells(grid).idxmax()


def rank_correlation(in_sample, out_of_sample):
    """Spearman rank correlation between in-sample and out-of-sample cell
    Sharpes. Near +1: good IS cells stayed good. Near 0 or negative: the IS
    ranking carried no information about the future."""
    pairs = pd.DataFrame({"is": _cells(in_sample), "oos": _cells(out_of_sample)}).dropna()
    return float(pairs["is"].corr(pairs["oos"], method="spearman"))


def _cells(grid):
    """Grid -> Series indexed by (fast, slow), dropping the invalid NaN cells."""
    return grid.stack(future_stack=True).dropna()
