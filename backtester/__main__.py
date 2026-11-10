"""Command-line entry point.

Examples (run from the repo root):
    python -m backtester --ticker AAPL --fast 20 --slow 50 --start 2018-01-01
    python -m backtester --ticker MU AVGO MRVL --start 2016-01-01 --ml
"""

import argparse
import warnings

# macOS system Python ships an old SSL library; urllib3 warns on import. Harmless here.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from .backtest import run_backtest
from .data import load_prices
from .ml import build_dataset, walk_forward
from .strategies import ma_crossover


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m backtester",
        description="MA crossover backtest vs buy and hold, plus an optional "
                    "walk-forward ML volatility experiment. Research only.")
    parser.add_argument("--ticker", nargs="+", default=["MU"],
                        help="one or more tickers (default: MU)")
    parser.add_argument("--fast", type=int, default=20, help="fast MA window in days (default: 20)")
    parser.add_argument("--slow", type=int, default=50, help="slow MA window in days (default: 50)")
    parser.add_argument("--start", default="2018-01-01", help="first date, YYYY-MM-DD (default: 2018-01-01)")
    parser.add_argument("--end", default=None, help="end date, exclusive (default: today)")
    parser.add_argument("--ml", action="store_true",
                        help="also run the walk-forward volatility-expansion model")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the local cache and re-download prices")
    return parser.parse_args(argv)


def report_backtest(ticker, prices, fast, slow):
    result = run_backtest(prices["Close"], ma_crossover(prices["Close"], fast, slow))
    first, last = result.index[0].date(), result.index[-1].date()

    print(f"=== {ticker}: MA{fast}/MA{slow} crossover vs buy and hold, {first} to {last} ===")
    print("    IN-SAMPLE: whole period, no held-out data. Describes the past, not expected performance.")
    print(f"    Strategy total return:      {result['strategy_equity'].iloc[-1] - 1:>9.1%}")
    print(f"    Buy and hold total return:  {result['benchmark_equity'].iloc[-1] - 1:>9.1%}")
    print(f"    Trades (entries + exits):   {int(result['trade'].sum()):>9d}")
    print()


def report_ml(ticker, prices):
    dataset = build_dataset(prices)
    folds = walk_forward(dataset)
    folds["edge_vs_majority_pp"] = (folds["model_acc"] - folds["majority_acc"]) * 100
    folds["edge_vs_vol_rule_pp"] = (folds["model_acc"] - folds["vol_rule_acc"]) * 100

    print(f"=== {ticker}: volatility-expansion model, {len(dataset)} usable days "
          f"({dataset.index[0].date()} to {dataset.index[-1].date()}) ===")
    print("    OUT-OF-SAMPLE: each row is a walk-forward test fold (5-day purge before each).")
    print(folds.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"    Mean edge vs majority: {folds['edge_vs_majority_pp'].mean():+.1f}pp "
          f"(beat it in {(folds['edge_vs_majority_pp'] > 0).sum()}/{len(folds)} folds)")
    print(f"    Mean edge vs vol rule: {folds['edge_vs_vol_rule_pp'].mean():+.1f}pp "
          f"(beat it in {(folds['edge_vs_vol_rule_pp'] > 0).sum()}/{len(folds)} folds)")
    print()


def main(argv=None):
    args = parse_args(argv)
    for ticker in args.ticker:
        prices = load_prices(ticker, args.start, args.end, refresh=args.refresh)
        report_backtest(ticker.upper(), prices, args.fast, args.slow)
        if args.ml:
            report_ml(ticker.upper(), prices)


if __name__ == "__main__":
    main()
