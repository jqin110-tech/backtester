"""Turn a signal into daily strategy returns. Fully vectorized, no loops.

Timing (the most important rule in any backtest):
    signal[t] uses closes up to and including day t, so the earliest we can
    act on it is the close of day t. The position we HOLD over day t+1
    (from close t to close t+1) is therefore signal[t]:

        position = signal.shift(1)
        strategy_return[t] = position[t] * asset_return[t]

    Without the shift, the strategy would trade on a close it couldn't have
    seen yet (lookahead bias).
"""

import pandas as pd


def run_backtest(close, signal):
    """Return a DataFrame indexed by date with these columns:

    close, signal, position, asset_return, trade, strategy_return,
    strategy_equity, benchmark_equity

    - The warm-up period (no valid position yet) is DROPPED, and buy and hold
      ("benchmark") is measured over exactly the same days as the strategy.
      Both equity curves start from the same close.
    - trade = |change in position|. We assume we start in cash, so entering
      on the first day counts as a trade. A 0 -> 1 -> 0 round trip is 2 trades
      ("legs"). That is the unit transaction costs will be charged on.
    """
    frame = pd.DataFrame({
        "close": close,
        "signal": signal,
        "position": signal.shift(1),
        "asset_return": close.pct_change(),
    })
    frame = frame.dropna(subset=["position", "asset_return"]).copy()
    if frame.empty:
        raise ValueError("Not enough data: no day has a valid position. "
                         "Use a longer date range or a shorter slow window.")

    starting_position = frame["position"].iloc[0]
    frame["trade"] = frame["position"].diff().fillna(starting_position).abs()

    frame["strategy_return"] = frame["position"] * frame["asset_return"]
    frame["strategy_equity"] = (1 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1 + frame["asset_return"]).cumprod()
    return frame
