"""Save charts to results/ as PNGs (embedded in the README).

Colours: strategy = blue, buy and hold = orange, the first two slots of a
colour-blind-checked palette. The Sharpe heatmap uses a diverging red-gray-
blue scale centred on 0, so "negative" and "positive" read as opposites and
gray means "about zero".
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

STRATEGY = "#2a78d6"
BENCHMARK = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SPLIT_LINE = "#8a8984"
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})


def _mark_split(ax, split_date):
    ax.axvline(split_date, color=SPLIT_LINE, linewidth=1, linestyle="--")
    ax.annotate("in-sample | out-of-sample", xy=(split_date, 1), xycoords=("data", "axes fraction"),
                xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                fontsize=8, color=TEXT_MUTED, backgroundcolor=SURFACE)


def _save(fig, name):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity(frame, ticker, label, split_date, cost_bps):
    """Growth of $1, strategy vs buy and hold, log scale (so equal % moves
    look equal at any level)."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(frame.index, frame["strategy_equity"], color=STRATEGY, linewidth=2, label=label)
    ax.plot(frame.index, frame["benchmark_equity"], color=BENCHMARK, linewidth=2, label="Buy and hold")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    for column, color in (("strategy_equity", STRATEGY), ("benchmark_equity", BENCHMARK)):
        ax.annotate(f"{frame[column].iloc[-1]:.1f}x", xy=(frame.index[-1], frame[column].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", color=TEXT, fontsize=9)
    _mark_split(ax, split_date)
    ax.legend(loc="upper left")
    ax.set_title(f"{ticker}: {label} vs buy and hold, after {cost_bps:g} bp costs per trade leg")
    return _save(fig, f"{ticker}_equity.png")


def plot_drawdown(frame, ticker, label, split_date):
    """Percent below the running peak, for both curves."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for column, color, name in (("strategy_equity", STRATEGY, label),
                                ("benchmark_equity", BENCHMARK, "Buy and hold")):
        equity = frame[column]
        drawdown = equity / equity.cummax().clip(lower=1.0) - 1
        ax.plot(frame.index, drawdown * 100, color=color, linewidth=1.5, label=name)
    ax.set_ylabel("Drawdown (%)")
    _mark_split(ax, split_date)
    ax.legend(loc="lower left")
    ax.set_title(f"{ticker}: drawdown from running peak")
    return _save(fig, f"{ticker}_drawdown.png")


def plot_sharpe_heatmaps(in_sample, out_of_sample, ticker, chosen, is_label, oos_label):
    """Side-by-side IS and OOS Sharpe grids on one shared colour scale. The
    cell chosen in-sample is outlined in both panels."""
    limit = np.nanmax(np.abs(np.concatenate([in_sample.values.ravel(), out_of_sample.values.ravel()])))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    for ax, grid, title in ((axes[0], in_sample, is_label), (axes[1], out_of_sample, oos_label)):
        image = ax.imshow(grid.values, cmap=DIVERGING, norm=norm, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set_xlabel("slow window (days)")
        ax.set_ylabel("fast window (days)")
        ax.set_title(title, fontsize=10)
        for i, fast in enumerate(grid.index):
            for j, slow in enumerate(grid.columns):
                value = grid.loc[fast, slow]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=TEXT)
        row, col = grid.index.get_loc(chosen[0]), grid.columns.get_loc(chosen[1])
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=TEXT, linewidth=2))

    fig.colorbar(image, ax=axes, shrink=0.85, label="Sharpe ratio (rf = 0)")
    fig.suptitle(f"{ticker}: Sharpe by MA windows. Outline = best in-sample cell",
                 x=0.0, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, f"{ticker}_sharpe_heatmap.png")
