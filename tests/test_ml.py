"""Target labelling and walk-forward purging."""

import numpy as np
import pandas as pd

from backtester.ml import HORIZON, build_dataset, compute_features, compute_target, fold_bounds


def fake_prices(n=400, seed=0):
    """A random-walk price series with random volume, reproducible via `seed`."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


def test_last_horizon_days_have_no_target():
    # Regression test: these used to be silently labelled 0.
    target = compute_target(fake_prices())
    assert target.iloc[-HORIZON:].isna().all()
    assert target.iloc[-HORIZON - 1:-HORIZON].notna().all()


def test_dataset_ends_before_unknown_future():
    prices = fake_prices()
    assert build_dataset(prices).index[-1] == prices.index[-HORIZON - 1]


def test_purge_keeps_training_labels_out_of_test_period():
    for train_end, test_start, _ in fold_bounds(n_rows=2631):
        last_train_row = train_end - 1
        # That row's label uses prices up to last_train_row + HORIZON.
        assert last_train_row + HORIZON < test_start


# ---- no feature may use future data ----------------------------------------

def changes_when_future_removed(feature_fn, prices, check_dates):
    """Names of columns whose value at date t changes when every row after t
    is deleted. A column that only uses the past can't change."""
    full = feature_fn(prices)
    if isinstance(full, pd.Series):
        full = full.to_frame()
    leaky = set()
    for t in check_dates:
        truncated = feature_fn(prices.loc[:t])
        if isinstance(truncated, pd.Series):
            truncated = truncated.to_frame()
        before = full.loc[t]
        after = truncated.loc[t] if t in truncated.index else pd.Series(np.nan, index=full.columns)
        same = np.isclose(before, after, rtol=1e-12, atol=0, equal_nan=True)
        leaky.update(full.columns[~same])
    return leaky


def check_dates(prices):
    """Dates spread through the series, after the 50-day warm-up."""
    return [prices.index[i] for i in (60, 150, 250, 330, len(prices) - 1)]


def test_no_feature_uses_future_data():
    prices = fake_prices()
    assert changes_when_future_removed(compute_features, prices, check_dates(prices)) == set()


def test_leak_check_catches_a_leaky_feature():
    # Sanity check on the test itself: a feature built with shift(-1) must be caught,
    # otherwise the test above could be passing because it checks nothing.
    def features_with_a_leak(prices):
        features = compute_features(prices)
        features["Tomorrow_Close"] = prices["Close"].shift(-1)
        return features

    prices = fake_prices()
    assert changes_when_future_removed(features_with_a_leak, prices,
                                       check_dates(prices)[:-1]) == {"Tomorrow_Close"}


def test_leak_check_flags_the_target():
    # The target is SUPPOSED to look forward. The check should see it.
    prices = fake_prices()
    assert changes_when_future_removed(compute_target, prices, check_dates(prices)[:-1]) == {"Target"}
