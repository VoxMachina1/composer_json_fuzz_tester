# RSI Tester — Strategy Discovery & Path Analysis Engine

A modular, zero-lookahead bias, multi-asset Python backtesting engine built for discovering alpha across large parameter grids. Extended with a strategy path extractor and automated frontrunner analysis pipeline for Composer.Trade strategies.

## Project Structure

```
rsi_tester/
├── run_analysis.py          # Orchestration script — runs the full pipeline
├── setup_project.py         # Creates required folder structure
├── requirements.txt
│
├── pathfinder/
│   ├── strategy_paths.py    # Extracts all boolean paths from a strategy JSON
│   └── strategy.json        # Your Composer strategy export
│
└── strategy_engine/
    ├── main.py              # Manual backtest entry point
    ├── config/
    │   └── template.yaml    # Shared config for both manual and automated runs
    ├── data/                # Auto-managed price CSVs (gitignored)
    ├── results/             # Output CSVs (gitignored)
    └── src/
        ├── config_loader.py
        ├── data_alignment.py
        ├── data_loader.py
        ├── indicators.py
        ├── metrics.py
        ├── preconditions.py
        ├── range_tester.py
        ├── signals.py
        └── strategy_engine.py
```

## Setup & Installation

1. Clone the repository and navigate to the project root.
2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the project structure generator to create the necessary folders:
   ```
   python setup_project.py
   ```
5. Create a `.env` file inside `strategy_engine/` and add your Tiingo API keys (comma separated):
   ```
   TIINGO_API_KEYS=key1,key2,key3
   ```

---

## Automated Frontrunner Analysis Pipeline — `run_analysis.py`

The primary workflow. Takes a Composer strategy JSON, extracts every possible boolean path to every endpoint asset, and runs the RSI discovery engine against all combinations of target and signal assets automatically.

```powershell
python run_analysis.py pathfinder/strategy.json
```

**What it does:**
1. Parses the strategy JSON using `strategy_paths.py` — extracts all 50+ endpoint paths
2. For each path, translates the boolean conditions into engine-compatible precondition strings
3. Sets the path's endpoint as the benchmark asset for that run
4. Builds a candidate target asset list: all unique endpoint tickers from the strategy + `target_assets` from `template.yaml`, minus the benchmark
5. Builds a signal asset list: tickers referenced in the path's preconditions + `EXTRA_SIGNAL_ASSETS`, minus the benchmark
6. Runs the RSI threshold range tester across all path × target × signal × threshold combinations
7. Writes a combined, timestamped CSV to `strategy_engine/results/`

**Configuring the pipeline:**

All engine settings are read from `strategy_engine/config/template.yaml` — edit this file to change the threshold range, date range, slippage, indicator period, etc. The two lists hardcoded in `run_analysis.py` itself are:

- `EXTRA_SIGNAL_ASSETS` — signal tickers always included regardless of strategy content
- `RESULTS_DIR` — output path for CSVs

**Output CSV columns:**

| Column | Description |
|---|---|
| `sub_strategy` | Which sleeve of the strategy (Bond Compares, KMLM, etc.) |
| `conditions` | Full human-readable boolean condition chain |
| `engine_precondition` | pandas-compatible precondition string |
| `endpoint` | The strategy's endpoint asset (also serves as benchmark) |
| `benchmark_asset` | Same as endpoint — what the strategy would otherwise hold |
| `target_asset` | The candidate replacement asset being tested |
| `signal_asset` | The asset whose RSI is being used as the signal |
| `threshold` | The RSI threshold being tested |
| `slippage_bps` | Slippage assumption in basis points |
| `risk_free_rate` | Risk-free rate used for Sharpe/Sortino |
| `Total_Trades` | Number of days signal was active |
| `Win_Rate` | % of signal-active days where target beat benchmark |
| `Avg_Return` | Mean daily return of target on signal-active days |
| `Median_Return` | Median daily return of target on signal-active days |
| `Benchmark_Avg_Return` | Mean daily return of benchmark on signal-active days |
| `Benchmark_Median_Return` | Median daily return of benchmark on signal-active days |
| `Total_Return` | Cumulative return over the backtest window |
| `Annualized_Return` | Annualized return |
| `Sharpe_Ratio` | Sharpe ratio |
| `Sortino_Ratio` | Sortino ratio |
| `Calmar_Ratio` | Calmar ratio |
| `Max_Drawdown` | Maximum drawdown |
| `Final_Equity` | Final equity multiplier (1.0 = no change) |
| `Avg_Hold_Days` | Average consecutive days signal remains active |

**Suggested Excel filters for identifying frontrunner candidates:**
- `Win_Rate >= 0.75`
- `Benchmark_Avg_Return < 0` (benchmark is down on signal days)
- `Total_Trades >= 15` (minimum datapoint threshold)
- `Median_Return` as tiebreaker within close win rate bands

---

## Path Extractor — `pathfinder/strategy_paths.py`

Standalone script that parses a Composer/VOXPORT strategy JSON and prints every possible boolean path to a leaf endpoint, grouped by endpoint ticker.

```powershell
python pathfinder/strategy_paths.py pathfinder/strategy.json
python pathfinder/strategy_paths.py pathfinder/strategy.json > pathfinder/paths.txt
```

Each line shows the sub-strategy sleeve, the full condition chain, and the endpoint:
```
[Bond Compares] IF Price(SPY) > MA(SPY, 200) AND RSI(TLT, 20) > RSI(PSQ, 20) -> TQQQ
[FTLT] IF Price(SPY) > MA(SPY, 200) AND RSI(QQQ, 10) > 79 -> BIL
```

Each path also carries an `engine_precondition` field (used internally by `run_analysis.py`) — a pandas-compatible translation of the condition chain ready for use in `evaluate_preconditions()`.

Asset-selection filters (e.g. "Top 1 by RSI from SQQQ, TLT") are expanded into explicit pairwise conditions:
```
RSI(SQQQ, 10) > RSI(TLT, 10) -> SQQQ
RSI(TLT, 10) > RSI(SQQQ, 10) -> TLT
```

---

## Manual Backtest — `strategy_engine/main.py`

For running individual backtests against a YAML config file:

```powershell
python strategy_engine/main.py --config config/your_config.yaml
```

---

## template.yaml Reference

All settings shared between manual and automated runs live here:

```yaml
# Assets to test as frontrunner targets in the automated pipeline
target_assets: ["BIL", "UVXY"]

benchmark_asset: "SPY"       # Used in manual runs only
indicator: "RSI"
indicator_period: 10
signal_operator: ">"
threshold_start: 50.0
threshold_end: 80.0
threshold_step: 0.5
slippage_bps: 1.0
risk_free_rate: 0.0

date_range:
  start: "2015-01-01"
  end: "2026-03-15"
```

---

## How to Match Composer.Trade Output

| Setting | Value | Reason |
|---|---|---|
| `signal_operator` | `">"` | Composer uses strict greater-than |
| `slippage_bps` | `1.0` | Composer's default slippage assumption |
| `risk_free_rate` | `0.0` | Composer assumes 0% risk-free rate for Sharpe/Sortino |
| `date_range.start` | match Composer chart | Align the backtest window exactly |

---

## Supported Indicators

| Name | Key | Notes |
|---|---|---|
| RSI (Wilder's smoothing) | `RSI` | Primary signal indicator |
| Simple Moving Average | `SMA` | Used in preconditions |
| Exponential Moving Average | `EMA` | Available |
| Cumulative Return | `CumRet` | Rolling % return over N days |