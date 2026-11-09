# Backtester

A pandas script in two parts that downloads daily stock prices from Yahoo Finance with `yfinance`.

**Part 1: moving-average crossover.** It holds Micron (MU) whenever its 20-day moving average is above its 50-day average and holds cash otherwise. It compares that strategy with buy and hold from 2024 to 2026 and prints both returns and the number of trades. Signals are shifted by a day, so the strategy never trades on prices it couldn't have known yet.

**Part 2: volatility-expansion model.** A scikit-learn random forest predicts whether volatility over the next 5 days will be higher than over the last 10 days. Its features are returns, RSI, MACD, Bollinger Band width, volume and moving-average spread. It's checked with 5-fold walk-forward validation (it always trains on the past and tests on the next period) on MU, AVGO and MRVL from 2016 onward. For each fold it prints the model's accuracy against a naive baseline.

## Run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python backtester.py
```

It needs an internet connection to download prices. Results are printed to the terminal.

This is a learning project, not trading advice.
