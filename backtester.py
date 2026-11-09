# Part 1: a simple MA20/MA50 crossover strategy, backtested against buy &
# hold.
#
# Part 2: a machine learning model that predicts whether volatility will
# EXPAND over the next week -- a more learnable target than price direction
# (see comments below) -- validated with walk-forward testing across three
# semiconductor stocks (MU, AVGO, MRVL) to check the pattern isn't a fluke
# specific to one company.

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================================================
# Part 1: MA20/MA50 crossover strategy
# ==========================================================================
#
# Strategy: when the fast average (MA20) is above the slow average (MA50)
# we are "in" the stock (long); otherwise we are in cash.
# We then compare that to just buying and holding.

TICKER = "MU"

# Download plenty of history. MA50 needs 50 trading days before it produces
# any value, so a 2-week window would be all NaN. Grab ~2 years instead.
data = yf.download(TICKER, start="2024-01-01", end="2026-09-01")

# Newer yfinance versions return columns as a 2-level MultiIndex
# (e.g. ("Close", "MU")). Flatten it so data["Close"] is a plain Series.
if hasattr(data.columns, "get_level_values"):
    data.columns = data.columns.get_level_values(0)

# Moving averages of the closing price.
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# Signal = 1 when we want to be holding the stock, 0 when in cash.
data["Signal"] = (data["MA20"] > data["MA50"]).astype(int)

# Trade = +1 on the day we buy, -1 on the day we sell, 0 otherwise.
data["Trade"] = data["Signal"].diff()

# --- Backtest the returns ------------------------------------------------

# Daily percentage change of the stock.
data["DailyReturn"] = data["Close"].pct_change()

# Our strategy earns the daily return only when yesterday's signal said
# "hold" (shift(1) avoids look-ahead: we can't trade on today's close today).
data["StrategyReturn"] = data["DailyReturn"] * data["Signal"].shift(1)

# Turn daily returns into a growth curve starting at 1.0.
data["BuyHoldEquity"] = (1 + data["DailyReturn"]).cumprod()
data["StrategyEquity"] = (1 + data["StrategyReturn"]).cumprod()

# --- Results -----------------------------------------------------------

print("=== Part 1: MA20/MA50 crossover backtest ===")
print(data[["Close", "MA20", "MA50", "Signal", "Trade"]].tail(20))

buy_hold = data["BuyHoldEquity"].iloc[-1] - 1
strategy = data["StrategyEquity"].iloc[-1] - 1
trades = int((data["Trade"] != 0).sum())

print()
print(f"Buy & hold return: {buy_hold:.1%}")
print(f"Strategy return:   {strategy:.1%}")
print(f"Number of trades:  {trades}")


# ==========================================================================
# Part 2: Machine learning experiment -- predicting volatility expansion
# ==========================================================================
#
# An earlier version of this experiment tried predicting plain price
# direction ("will the price be up in 5 days?") with a few basic features.
# It came out dead even with the naive "always guess up" baseline -- the
# model hadn't learned anything real, just recovered the fact the stock
# mostly goes up. That's a very common, realistic outcome: price direction
# is close to random and hard to beat.
#
# This version predicts something more learnable instead: will volatility
# EXPAND over the next week? Volatility is known to "cluster" (calm periods
# stay calm, turbulent periods stay turbulent) -- a real, well-documented
# statistical pattern. It also uses more history, more features, and a
# RandomForest (which can pick up on feature combinations, not just
# straight-line relationships), and checks the result with walk-forward
# validation across three chip stocks instead of trusting a single result.

FOLD_COUNT = 5

FEATURES = [
    "Return_1d", "Return_5d", "Return_10d",
    "Volatility_10d", "RSI_14", "MACD_Hist",
    "BB_Width", "Volume_Change", "Relative_Volume", "MA_Spread",
]


def compute_rsi(prices, period=14):
    """Relative Strength Index: a 0-100 score of recent up-moves vs
    down-moves. Above ~70 is considered "overbought", below ~30
    "oversold". Widely used as a momentum feature."""
    delta = prices.diff()
    gains = delta.clip(lower=0)          # keep only up-days (losses -> 0)
    losses = -delta.clip(upper=0)        # keep only down-days, as positive numbers
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    relative_strength = avg_gain / avg_loss
    return 100 - (100 / (1 + relative_strength))


def build_features(ticker):
    """Download price history for `ticker` and compute the feature/target
    columns needed for the ML model. Returns a cleaned DataFrame."""
    prices = yf.download(ticker, start="2016-01-01", end="2026-09-01")

    if hasattr(prices.columns, "get_level_values"):
        prices.columns = prices.columns.get_level_values(0)

    prices["DailyReturn"] = prices["Close"].pct_change()

    # Returns over a few different lookback windows -- short and
    # medium-term momentum.
    prices["Return_1d"] = prices["Close"].pct_change(1)
    prices["Return_5d"] = prices["Close"].pct_change(5)
    prices["Return_10d"] = prices["Close"].pct_change(10)

    # Trailing volatility -- how choppy the last 10 days have been.
    prices["Volatility_10d"] = prices["DailyReturn"].rolling(10).std()

    prices["RSI_14"] = compute_rsi(prices["Close"])

    # MACD: difference between a fast (12-day) and slow (26-day)
    # exponential moving average, then a 9-day EMA of that ("signal
    # line"). The histogram (MACD - signal) is a common momentum feature.
    ema12 = prices["Close"].ewm(span=12, adjust=False).mean()
    ema26 = prices["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    prices["MACD_Hist"] = macd_line - macd_signal

    # Bollinger Band width: how wide the price's normal trading range is,
    # relative to price. Widens when the market gets volatile.
    rolling_mean_20 = prices["Close"].rolling(20).mean()
    rolling_std_20 = prices["Close"].rolling(20).std()
    prices["BB_Width"] = (2 * rolling_std_20) / rolling_mean_20

    # Volume features: is trading activity picking up?
    prices["Volume_Change"] = prices["Volume"].pct_change()
    prices["Relative_Volume"] = prices["Volume"] / prices["Volume"].rolling(20).mean()

    # The MA20/MA50 signal from Part 1, expressed as a continuous distance
    # instead of an on/off switch -- a more useful shape for a model.
    prices["MA_Spread"] = (
        prices["Close"].rolling(20).mean() - prices["Close"].rolling(50).mean()
    ) / prices["Close"].rolling(50).mean()

    # Target: will volatility be HIGHER over the next 5 days than it has
    # been over the trailing 10 days? shift(-5) pulls in a value computed
    # from the future so we can label today with what actually happened
    # next.
    future_volatility = prices["DailyReturn"].rolling(5).std().shift(-5)
    prices["Target"] = (future_volatility > prices["Volatility_10d"]).astype(int)

    return prices.dropna(subset=FEATURES + ["Target"])


def walk_forward_validate(ml_data, ticker):
    """Repeat the train/test process across FOLD_COUNT sequential time
    periods, always training only on data before the test period (never
    after -- that would leak the future). If the edge over the naive
    baseline shows up consistently across periods, that's real evidence of
    a pattern rather than one lucky test window."""
    chunk_size = len(ml_data) // (FOLD_COUNT + 1)
    chunks = [ml_data.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(FOLD_COUNT + 1)]

    print(f"=== {ticker} ===")
    print(f"{len(ml_data)} usable days of history ({ml_data.index[0].date()} to {ml_data.index[-1].date()})")
    print()

    fold_edges = []
    for fold_num in range(1, FOLD_COUNT + 1):
        fold_train = pd.concat(chunks[:fold_num])   # everything before this fold
        fold_test = chunks[fold_num]                # this fold's test period

        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        model.fit(fold_train[FEATURES], fold_train["Target"])

        predictions = model.predict(fold_test[FEATURES])
        accuracy = accuracy_score(fold_test["Target"], predictions)
        baseline = max(fold_test["Target"].mean(), 1 - fold_test["Target"].mean())
        edge = (accuracy - baseline) * 100
        fold_edges.append(edge)

        start_date = fold_test.index[0].date()
        end_date = fold_test.index[-1].date()
        print(f"Fold {fold_num} ({start_date} to {end_date}, {len(fold_test)} days): "
              f"model {accuracy:.1%}  baseline {baseline:.1%}  edge {edge:+.1f}pp")

    avg_edge = sum(fold_edges) / len(fold_edges)
    winning_folds = sum(1 for e in fold_edges if e > 0)
    print()
    print(f"Average edge across folds: {avg_edge:+.1f}pp")
    print(f"Beat baseline in {winning_folds}/{FOLD_COUNT} folds")
    print()

    return avg_edge, winning_folds


print()
print("=== Part 2: ML volatility-expansion model (walk-forward validated) ===")
print()

# Run the same experiment on a few chip stocks to see if the volatility-
# clustering pattern is specific to MU or holds up more broadly.
TICKERS = ["MU", "AVGO", "MRVL"]  # Micron, Broadcom, Marvell Technology

results = {}
for ticker in TICKERS:
    ml_data = build_features(ticker)
    avg_edge, winning_folds = walk_forward_validate(ml_data, ticker)
    results[ticker] = (avg_edge, winning_folds)

print("=== Summary ===")
for ticker, (avg_edge, winning_folds) in results.items():
    print(f"{ticker:<6} average edge {avg_edge:+.1f}pp, beat baseline in {winning_folds}/{FOLD_COUNT} folds")
