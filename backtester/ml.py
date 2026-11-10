"""ML experiment: predict whether volatility EXPANDS over the next week.

Target: is the volatility of the next HORIZON (5) daily returns higher than
the trailing 10-day volatility? Volatility clusters, which makes it more
predictable than price direction.

Why this is split into compute_features / compute_target / build_dataset:
features must use only the past, and the target is the one column that is
allowed (required, really) to look forward. Keeping them in separate
functions makes that contract easy to check, and lets the leakage test
recompute features on truncated data without the target getting in the way.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

HORIZON = 5     # days the target looks ahead
N_FOLDS = 5

FEATURES = [
    "Return_1d", "Return_5d", "Return_10d",
    "Volatility_10d", "RSI_14", "MACD_Hist",
    "BB_Width", "Volume_Change", "Relative_Volume", "MA_Spread",
]


def compute_rsi(close, period=14):
    """Relative Strength Index (simple-average version): a 0-100 score of
    recent average up-moves vs down-moves."""
    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(period).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def compute_features(prices):
    """Features for each day t, using ONLY data up to and including day t.
    Rows without a full set of features (the warm-up) are dropped."""
    close, volume = prices["Close"], prices["Volume"]
    daily_return = close.pct_change()
    features = pd.DataFrame(index=prices.index)

    # Momentum over a few lookbacks.
    features["Return_1d"] = daily_return
    features["Return_5d"] = close.pct_change(5)
    features["Return_10d"] = close.pct_change(10)

    # Trailing volatility. The target is defined relative to this.
    features["Volatility_10d"] = daily_return.rolling(10).std()

    features["RSI_14"] = compute_rsi(close)

    # MACD histogram: (EMA12 - EMA26) minus its own 9-day EMA.
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    features["MACD_Hist"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    # Bollinger Band width relative to price.
    features["BB_Width"] = 2 * close.rolling(20).std() / close.rolling(20).mean()

    # Volume activity.
    features["Volume_Change"] = volume.pct_change()
    features["Relative_Volume"] = volume / volume.rolling(20).mean()

    # The Part 1 crossover as a continuous distance instead of an on/off switch.
    ma20, ma50 = close.rolling(20).mean(), close.rolling(50).mean()
    features["MA_Spread"] = (ma20 - ma50) / ma50

    # A zero-volume day would give an infinite % change. Treat it as missing.
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna()


def compute_target(prices):
    """1 if the std of the next HORIZON daily returns exceeds the trailing
    10-day std, else 0. NaN for the last HORIZON days (the future isn't
    known yet).

    Bug fixed vs the original script: `NaN > x` is False, so those last days
    used to be silently labelled 0 instead of being dropped.
    """
    daily_return = prices["Close"].pct_change()
    trailing_vol = daily_return.rolling(10).std()
    # Value at row t = std of returns on days t+1 .. t+HORIZON.
    future_vol = daily_return.rolling(HORIZON).std().shift(-HORIZON)
    target = (future_vol > trailing_vol).astype(float)
    return target.where(future_vol.notna() & trailing_vol.notna()).rename("Target")


def build_dataset(prices):
    """Features joined with the target. Only rows where both exist are kept."""
    dataset = compute_features(prices).join(compute_target(prices)).dropna()
    return dataset.astype({"Target": int})


def fold_bounds(n_rows, n_folds=N_FOLDS, purge=HORIZON):
    """Row positions for expanding-window walk-forward with a purge gap.

    Splits the rows into n_folds + 1 equal chunks. Fold k tests on chunk k
    and trains on everything before it, minus the last `purge` rows.

    Why purge: the label of row t uses prices up to day t + HORIZON. If row t
    sits within HORIZON days of the test start, its label was computed from
    test-period prices. Dropping those rows from training removes that leak.

    Returns a list of (train_end, test_start, test_end). Python-slice style:
    train = rows[:train_end], test = rows[test_start:test_end].
    """
    chunk = n_rows // (n_folds + 1)
    return [(k * chunk - purge, k * chunk, (k + 1) * chunk) for k in range(1, n_folds + 1)]


def walk_forward(dataset, n_folds=N_FOLDS, purge=HORIZON, random_state=42):
    """Train and test a RandomForest on each walk-forward fold. Every number
    returned is OUT-OF-SAMPLE (measured on test rows the model never saw).

    Two baselines, because beating a weak baseline proves little:
    - majority: always guess the most common class OF THE TEST FOLD. It peeks
      at test labels, so it is an upper bound on any constant guess. That
      makes it a deliberately tough bar.
    - vol_rule: a one-line rule with no ML. Predict "expansion" when trailing
      10-day vol is below its TRAINING-period median. It captures pure mean
      reversion of volatility. If the forest can't beat this, the forest
      hasn't learned much beyond "low vol tends to rise".
    """
    rows = []
    for fold, (train_end, test_start, test_end) in enumerate(
            fold_bounds(len(dataset), n_folds, purge), start=1):
        train = dataset.iloc[:train_end]
        test = dataset.iloc[test_start:test_end]
        y_test = test["Target"]

        model = RandomForestClassifier(n_estimators=200, max_depth=5,
                                       random_state=random_state, n_jobs=-1)
        model.fit(train[FEATURES], train["Target"])
        predictions = model.predict(test[FEATURES])

        vol_threshold = train["Volatility_10d"].median()
        rule_predictions = (test["Volatility_10d"] < vol_threshold).astype(int)

        rows.append({
            "fold": fold,
            "test_start": test.index[0].date(),
            "test_end": test.index[-1].date(),
            "n_test": len(test),
            "model_acc": accuracy_score(y_test, predictions),
            "majority_acc": max(y_test.mean(), 1 - y_test.mean()),
            "vol_rule_acc": accuracy_score(y_test, rule_predictions),
        })
    return pd.DataFrame(rows).set_index("fold")
