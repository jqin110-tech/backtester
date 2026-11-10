"""Turn a signal into daily strategy returns. Fully vectorized, no loops.

Timing (the most important rule in any backtest):
    signal[t] uses closes up to and including day t, so the earliest we can
    act on it is the close of day t. The position we HOLD over day t+1
    (from close t to close t+1) is therefore signal[t]:

        position = signal.shift(1)
        strategy_return[t] = position[t] * asset_return[t] - costs[t]

    Without the shift, the strategy would trade on a close it couldn't have
    seen yet (lookahead bias).

Costs: every unit of position change pays `cost_bps` basis points
(1 bp = 0.01%) of portfolio value. Think of it as commission + half the
bid-ask spread + slippage, all in one number. The cost is charged against
the return of the day the new position is first held. (Exactly, net value
would be V * (1 - c) * (1 + r); we use V * (1 + r - c), which differs only
by the tiny c * r term.)
"""

import pandas as pd


def run_backtest(close, signal, cost_bps=0.0):
    """Return a DataFrame indexed by date with these columns:

    close, signal, position, asset_return, trade, cost, strategy_return,
    benchmark_return, strategy_equity, benchmark_equity

    - The warm-up period (no valid position yet) is DROPPED, and buy and hold
      ("benchmark") is measured over exactly the same days as the strategy.
      Both equity curves start from the same close.
    - trade = |change in position|. We assume we start in cash, so entering
      on the first day counts as a trade. A 0 -> 1 -> 0 round trip is 2 trades
      ("legs"). Costs are charged per leg.
    - Buy and hold pays the cost once, for its single entry, so the comparison
      is like-for-like.
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

    cost_per_leg = cost_bps / 10_000
    starting_position = frame["position"].iloc[0]
    frame["trade"] = frame["position"].diff().fillna(starting_position).abs()
    frame["cost"] = frame["trade"] * cost_per_leg

    frame["strategy_return"] = frame["position"] * frame["asset_return"] - frame["cost"]

    frame["benchmark_return"] = frame["asset_return"]
    frame.iloc[0, frame.columns.get_loc("benchmark_return")] -= cost_per_leg

    frame["strategy_equity"] = (1 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1 + frame["benchmark_return"]).cumprod()
    return frame


def split_by_date(frame, split_date):
    """Split a backtest frame into (in_sample, out_of_sample).

    in_sample     = rows strictly before split_date
    out_of_sample = rows on or after split_date

    The signal was computed on the full history. That is safe: every input
    to the signal looks only backward, so OOS days use exactly the same
    information they would have had live. It also means the OOS period
    starts with a valid position instead of a fresh warm-up.
    """
    split = pd.Timestamp(split_date)
    return frame[frame.index < split], frame[frame.index >= split]
