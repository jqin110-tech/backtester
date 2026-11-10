"""Strategies: functions that turn prices into a SIGNAL.

Convention used everywhere in this package:
    signal[t] = the position we want to hold AFTER the close of day t,
                computed only from data up to and including day t.
    1.0 = fully long, 0.0 = cash, NaN = not enough history to decide yet.

The backtest (not the strategy) is responsible for the one-day delay
between deciding and holding. Keeping that in one place means no strategy
can accidentally skip it.
"""

import numpy as np


def ma_crossover(close, fast=20, slow=50):
    """Long when the `fast`-day moving average is above the `slow`-day one.

    During the first `slow - 1` days the slow average doesn't exist yet, so
    the signal is NaN ("undecided"), not 0 ("cash"). Treating warm-up as
    cash would quietly make the strategy look worse than buy and hold over
    days on which it could never have traded.
    """
    if not (0 < fast < slow):
        raise ValueError(f"Need 0 < fast < slow, got fast={fast}, slow={slow}.")

    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    signal = (fast_ma > slow_ma).astype(float)
    signal[slow_ma.isna()] = np.nan
    return signal.rename("signal")
