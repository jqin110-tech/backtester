"""Target labelling and walk-forward purging."""

import numpy as np
import pandas as pd

from backtester.ml import HORIZON, build_dataset, compute_target, fold_bounds


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
