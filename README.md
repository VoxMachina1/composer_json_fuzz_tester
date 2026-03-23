# RSI Tester — Strategy Discovery & Path Analysis Engine

A modular, zero-lookahead bias, multi-asset Python backtesting engine built for discovering alpha across large parameter grids. Extended with a strategy path extractor and automated analysis pipeline for Composer.Trade strategies.

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
    │   └── template.yaml    # Config template for manual runs
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
2. Create and activate a Python virtual environment.
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

## Automated Pipeline — `run_analysis.py`

The primary workflow. Takes a Composer strategy JSON, extracts every possible boolean path to every endpoint asset, and runs the RSI discovery engine against all of them automatically.

```
python run_analysis.py pathfinder/strategy.json
```

**What it does:**
1. Parses the strategy JSON and extracts all endpoint paths using `strategy_paths.py`
2. For each path, translates the boolean conditions into engine-compatible precondition strings
3. Determines signal assets to test: tickers already present in the path conditions + a hardcoded list of additional candidates
4. Runs the RSI threshold range tester across all signal/endpoint/threshold combinations
5. Writes a combined CSV to `strategy_engine/results/`, timestamped and named after the input file

**Output CSV columns:**
`sub_strategy`, `conditions`, `engine_precondition`, `endpoint`, `signal_asset`, `threshold`, `slippage_bps`, `risk_free_rate`, `Total_Trades`, `Win_Rate`, `Avg_Return`, `Total_Return`, `Annualized_Return`, `Sharpe_Ratio`, `Sortino_Ratio`, `Calmar_Ratio`, `Max_Drawdown`, `Final_Equity`, `Avg_Hold_Days`, `Benchmark_Avg_Return`, `Benchmark_Median_Return`

---

## Path Extractor — `pathfinder/strategy_paths.py`

Standalone script that parses a Composer/VOXPORT strategy JSON and prints every possible boolean path to a leaf endpoint.

```
python pathfinder/strategy_paths.py pathfinder/strategy.json
python pathfinder/strategy_paths.py pathfinder/strategy.json > pathfinder/paths.txt
```

Each line shows the sub-strategy sleeve, the full condition chain, and the endpoint asset:
```
[Bond Compares] IF Price(SPY) > MA(SPY, 200) AND RSI(TLT, 20) > RSI(PSQ, 20) -> TQQQ
[FTLT] IF Price(SPY) > MA(SPY, 200) AND RSI(QQQ, 10) > 79 -> BIL
```

---

## Manual Backtest — `strategy_engine/main.py`

For running individual backtests against a YAML config file:

```
python strategy_engine/main.py --config config/your_config.yaml
```

---

## How to Match Composer.Trade Output

Use these settings in your YAML config to align with Composer's platform assumptions:

| Setting | Value | Reason |
|---|---|---|
| `signal_operator` | `">"` | Composer uses strict greater-than |
| `slippage_bps` | `1.0` | Composer's default slippage assumption |
| `risk_free_rate` | `0.0` | Composer assumes 0% risk-free rate for Sharpe |
| `date_range.start` | match Composer chart | Align the backtest window exactly |

Example Composer-matched YAML:
```yaml
signal_assets: ["QQQ"]
target_assets: ["VIXY"]
benchmark_asset: "SPY"

indicator: "RSI"
indicator_period: 10

signal_operator: ">"
threshold_start: 50.0
threshold_end: 80.0
threshold_step: 0.5

slippage_bps: 1.0
risk_free_rate: 0.0

preconditions:
```

---

## Supported Indicators

| Name | YAML key | Notes |
|---|---|---|
| RSI (Wilder's smoothing) | `RSI` | Primary signal indicator |
| Simple Moving Average | `SMA` | Used in preconditions |
| Exponential Moving Average | `EMA` | Available |
| Cumulative Return | `CumRet` | Rolling % return over N days |