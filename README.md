# Strategy Discovery Engine

A modular, zero-lookahead bias, multi-asset Python backtesting engine built for discovering alpha across large parameter grids.

This engine downloads split-and-dividend adjusted price data directly from the Tiingo API, aligns market days across disparate assets, dynamically generates combinations of Signal and Target assets, and performs brute-force threshold parameter testing.

## Features
* **100% Transparent Math:** Row-by-row DataFrame evaluation means zero lookahead bias.
* **Binary Allocation:** Models state-based signals (e.g., Hold Target Asset when Signal is True, Hold Benchmark Asset when Signal is False).
* **Automatic Data Management:** Automatically syncs with Tiingo, determines the latest US market close, and maintains clean, cached `.csv` datasets.
* **Dynamic Preconditions:** Supports configurable `AND` filters (e.g., Only trade if `SPY close > SPY SMA 200`).

## Setup & Installation

1. Clone the repository and navigate to the project root.
2. Create and activate a Python virtual environment.
3. Install the dependencies:
`pip install -r requirements.txt`

Run the project structure generator to create the necessary folders:
`python setup_project.py`

Create a .env file inside the strategy_engine/ directory and add your Tiingo API keys (comma separated):
`TIINGO_API_KEYS=key1,key2,key3`

# Running a Backtest
Edit your YAML config file located in strategy_engine/config/, then run the engine via the terminal:

`python strategy_engine/main.py --config config/your_config.yaml`

# How to Match Composer.Trade Output
If you want to use this tool to validate or discover strategies that you plan to deploy on Composer, use the following specific settings in your strategy_config.yaml to ensure the math aligns with Composer's platform assumptions:
1. Signal Operator: Composer's Greater Than block is strictly >, not >=. Set signal_operator: ">".
2. Slippage: Composer's default backtests apply 1 basis point of slippage to each trade. Set slippage_bps: 1.0.
3. Risk-Free Rate: Composer assumes a 0% risk-free rate when calculating the Sharpe Ratio. Set risk_free_rate: 0.0.
4. Date Alignment: Ensure your start_date exactly matches the date on the far left of the Composer backtest chart.
Example Composer-Matched YAML
```
signal_assets: ["QQQ"]
target_assets:["VIXY"]
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