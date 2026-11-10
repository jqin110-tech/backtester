"""Command-line entry point.

Examples (run from the repo root):
    python -m backtester --ticker AAPL --fast 20 --slow 50 --start 2018-01-01
    python -m backtester --ticker MU AVGO MRVL --start 2016-01-01 --ml

For each ticker it prints, in order:
  1. metrics for the strategy and buy and hold, split IN-SAMPLE / OUT-OF-SAMPLE
  2. the same at 0, 5 and 20 bp transaction costs
  3. a fast x slow Sharpe grid: best cell chosen in-sample, then checked out-of-sample
  4. (with --ml) the walk-forward volatility model, all out-of-sample
and saves charts to results/.
"""

import argparse
import warnings

# macOS system Python ships an old SSL library; urllib3 warns on import. Harmless here.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import pandas as pd

from .backtest import run_backtest, split_by_date
from .data import load_prices
from .metrics import TRADING_DAYS, sharpe_ratio, sharpe_standard_error, summarize
from .ml import build_dataset, walk_forward
from .plots import plot_drawdown, plot_equity, plot_sharpe_heatmaps
from .robustness import best_cell, rank_correlation, sharpe_grids
from .strategies import ma_crossover

COST_LEVELS = [0, 5, 20]
PERCENT_COLUMNS = ["total_return", "annual_return", "annual_vol", "max_drawdown", "win_rate", "exposure"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m backtester",
        description="MA crossover backtest vs buy and hold with in-sample / "
                    "out-of-sample reporting, cost sensitivity, a parameter grid, "
                    "and an optional walk-forward ML volatility experiment. Research only.")
    parser.add_argument("--ticker", nargs="+", default=["MU"], help="one or more tickers (default: MU)")
    parser.add_argument("--fast", type=int, default=20, help="fast MA window in days (default: 20)")
    parser.add_argument("--slow", type=int, default=50, help="slow MA window in days (default: 50)")
    parser.add_argument("--start", default="2018-01-01", help="first date, YYYY-MM-DD (default: 2018-01-01)")
    parser.add_argument("--end", default=None, help="end date, exclusive (default: today)")
    parser.add_argument("--split-date", default="2023-01-01",
                        help="out-of-sample period starts here (default: 2023-01-01)")
    parser.add_argument("--cost-bps", type=float, default=5.0,
                        help="cost per trade leg in basis points (default: 5)")
    parser.add_argument("--rf", type=float, default=0.0,
                        help="annual risk-free rate for Sharpe, e.g. 0.03 (default: 0)")
    parser.add_argument("--ml", action="store_true", help="also run the walk-forward volatility model")
    parser.add_argument("--no-grid", action="store_true", help="skip the parameter grid")
    parser.add_argument("--no-plots", action="store_true", help="don't save charts")
    parser.add_argument("--refresh", action="store_true", help="ignore the local cache and re-download")
    return parser.parse_args(argv)


def heading(text):
    print()
    print(text)
    print("-" * len(text))


def metrics_table(rows):
    """{row label: summarize() dict} -> printable DataFrame."""
    table = pd.DataFrame(rows).T
    for column in PERCENT_COLUMNS:
        table[column] = table[column].map(lambda x: f"{x:.1%}" if pd.notna(x) else "n/a")
    table["sharpe"] = table["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    return table.to_string()


def period_text(frame):
    return f"{frame.index[0].date()} to {frame.index[-1].date()}, {len(frame)} days"


def report_metrics(ticker, frame, split_date, label, cost_bps, rf):
    in_sample, out_of_sample = split_by_date(frame, split_date)
    rows = {}
    for name, part in (("IS", in_sample), ("OOS", out_of_sample)):
        if len(part) < 2:
            continue
        rows[f"{label} [{name}]"] = summarize(part["strategy_return"], part["position"], rf)
        rows[f"Buy and hold [{name}]"] = summarize(part["benchmark_return"], part["position"] * 0 + 1, rf)

    heading(f"{ticker}: {label} vs buy and hold, {cost_bps:g} bp per trade leg, rf = {rf:.1%}")
    if len(in_sample):
        print(f"IS  = in-sample      {period_text(in_sample)}")
    if len(out_of_sample):
        print(f"OOS = out-of-sample  {period_text(out_of_sample)}")
    print("MA windows were fixed in advance (not tuned on this data), but the ticker and")
    print("the method were chosen by a person in 2026. Treat IS as description, not evidence.")
    print(metrics_table(rows))
    return in_sample, out_of_sample


def report_costs(ticker, close, signal, split_date, rf):
    rows = []
    for bps in COST_LEVELS:
        frame = run_backtest(close, signal, bps)
        in_sample, out_of_sample = split_by_date(frame, split_date)
        row = {"cost_bps": bps}
        for name, part in (("IS", in_sample), ("OOS", out_of_sample)):
            if len(part) < 2:
                continue
            row[f"{name} annual return"] = f"{(1 + part['strategy_return']).prod() ** (TRADING_DAYS / len(part)) - 1:.1%}"
            row[f"{name} Sharpe"] = f"{sharpe_ratio(part['strategy_return'], rf):.2f}"
        rows.append(row)

    frame = run_backtest(close, signal, 0)
    legs_per_year = frame["trade"].sum() / len(frame) * TRADING_DAYS
    heading(f"{ticker}: sensitivity to transaction costs (strategy only)")
    print(pd.DataFrame(rows).set_index("cost_bps").to_string())
    print(f"Turnover: {legs_per_year:.1f} trade legs per year, so each 1 bp of cost per leg "
          f"costs about {legs_per_year:.1f} bp of return per year.")


def report_grid(ticker, close, split_date, cost_bps, fast, slow, rf, make_plots):
    in_grid, out_grid, is_start = sharpe_grids(close, split_date, cost_bps, rf)
    chosen = best_cell(in_grid)
    n_cells = int(in_grid.notna().sum().sum())
    oos_rank = int(out_grid.stack(future_stack=True).dropna().rank(ascending=False)[chosen])
    is_days = int(((close.index >= is_start) & (close.index < pd.Timestamp(split_date))).sum())
    standard_error = sharpe_standard_error(in_grid.loc[chosen], is_days)

    heading(f"{ticker}: parameter grid, {n_cells} fast/slow pairs, {cost_bps:g} bp per leg")
    print(f"All cells scored on the same IS window: {is_start.date()} to day before {split_date}.")
    print("Sharpe, IN-SAMPLE (rows = fast, columns = slow):")
    print(in_grid.to_string(float_format=lambda x: f"{x:.2f}", na_rep="."))
    print("Sharpe, OUT-OF-SAMPLE:")
    print(out_grid.to_string(float_format=lambda x: f"{x:.2f}", na_rep="."))
    print(f"Best in-sample cell: fast={chosen[0]}, slow={chosen[1]}, IS Sharpe "
          f"{in_grid.loc[chosen]:.2f} (standard error about {standard_error:.2f}).")
    print(f"  Same cell OUT-OF-SAMPLE: Sharpe {out_grid.loc[chosen]:.2f}, "
          f"rank {oos_rank} of {n_cells} cells.")
    if (fast, slow) in out_grid.stack(future_stack=True).dropna().index:
        print(f"  Pre-chosen {fast}/{slow} for comparison: IS {in_grid.loc[fast, slow]:.2f}, "
              f"OOS {out_grid.loc[fast, slow]:.2f}.")
    print(f"  Rank correlation of cells, IS vs OOS (Spearman): {rank_correlation(in_grid, out_grid):+.2f}")
    print(f"  Median cell Sharpe: IS {in_grid.stack(future_stack=True).median():.2f}, OOS {out_grid.stack(future_stack=True).median():.2f}")

    if make_plots:
        plot_sharpe_heatmaps(in_grid, out_grid, ticker, chosen,
                             f"In-sample ({is_start.date()} to before {split_date})",
                             f"Out-of-sample ({split_date} onward)")


def report_ml(ticker, prices):
    dataset = build_dataset(prices)
    result = walk_forward(dataset)
    folds = result["folds"]

    heading(f"{ticker}: volatility-expansion model, {len(dataset)} usable days "
            f"({dataset.index[0].date()} to {dataset.index[-1].date()})")
    print("OUT-OF-SAMPLE: each row is a walk-forward test fold, with a 5-day purge before it.")
    print("precision / recall are for class 1 = 'volatility expands'.")
    print(folds.to_string(float_format=lambda x: f"{x:.3f}"))
    edge_majority = (folds["model_acc"] - folds["majority_acc"]) * 100
    edge_rule = (folds["model_acc"] - folds["vol_rule_acc"]) * 100
    print(f"Mean edge vs majority:  {edge_majority.mean():+.1f}pp (beat it in {(edge_majority > 0).sum()}/{len(folds)} folds)")
    print(f"Mean edge vs vol rule:  {edge_rule.mean():+.1f}pp (beat it in {(edge_rule > 0).sum()}/{len(folds)} folds)")

    print("Confusion matrices (rows = actual 0/1, columns = predicted 0/1):")
    for fold, matrix in result["confusion"].items():
        (tn, fp), (fn, tp) = matrix
        print(f"  fold {fold}:  TN {tn:>3}  FP {fp:>3}  |  FN {fn:>3}  TP {tp:>3}")

    print("Feature importance, averaged over folds:")
    print("  impurity = use in training splits; permutation = test-accuracy drop when shuffled")
    print(result["importance"].to_string(float_format=lambda x: f"{x:.3f}"))


def main(argv=None):
    args = parse_args(argv)
    label = f"MA{args.fast}/{args.slow}"
    for raw_ticker in args.ticker:
        ticker = raw_ticker.upper()
        prices = load_prices(ticker, args.start, args.end, refresh=args.refresh)
        close = prices["Close"]
        signal = ma_crossover(close, args.fast, args.slow)
        frame = run_backtest(close, signal, args.cost_bps)

        print()
        print("=" * 78)
        print(f"{ticker}   ({args.start} to {args.end or 'today'}; research output, not advice)")
        print("=" * 78)
        report_metrics(ticker, frame, args.split_date, label, args.cost_bps, args.rf)
        report_costs(ticker, close, signal, args.split_date, args.rf)
        if not args.no_grid:
            report_grid(ticker, close, args.split_date, args.cost_bps, args.fast, args.slow,
                        args.rf, not args.no_plots)
        if not args.no_plots:
            split = pd.Timestamp(args.split_date)
            plot_equity(frame, ticker, label, split, args.cost_bps)
            plot_drawdown(frame, ticker, label, split)
        if args.ml:
            report_ml(ticker, prices)


if __name__ == "__main__":
    main()
