# You ever wonder how robust any given check in one of your strategies is?
https://github.com/VoxMachina1/composer_json_fuzz_tester

# Composer JSON Fuzz Tester

A robustness testing tool for [Composer](https://www.composer.trade) strategy JSON files.

Extracts every IF condition from your strategy, runs a 2D parameter sweep across each one (period × threshold), and measures how consistently each condition performs when its parameters are nudged/fuzzed. The output is a single self-contained HTML report with interactive heatmaps, a fragility-ranked sidebar, and a per-cell stat panel.

The goal is to separate robust edges from overfit noise — a condition that only works at one specific RSI threshold is fragile; one that works across a wide range is worth investigating.

---

## Requirements

- Python 3.10+
- A free [Tiingo](https://www.tiingo.com) API key

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/VoxMachina1/composer_json_fuzz_tester.git
cd composer_json_fuzz_tester
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Tiingo API key

Create the file `strategy_engine/.env` — this is gitignored and must never be committed:

```
TIINGO_API_KEYS=your_key_here
```

Multiple keys (for rotation) are supported as a comma-separated list:

```
TIINGO_API_KEYS=key_one,key_two,key_three
```

See `strategy_engine/.env.example` for the expected format. Get a free key at [tiingo.com](https://www.tiingo.com) — the free tier covers everything this tool needs.

### 5. Add your strategy JSON

Export your strategy from Composer or VOXPORT as a JSON file and place it at:

```
pathfinder/strategy.json
```

You can use a different path and enter it manually when the tool prompts you.

---

## Running

```bash
python fuzz_tester.py
```

The tool will prompt for configuration, download any missing price data automatically (internet required on first run), run the sweep, and write a timestamped HTML report to the project root.

### Configuration prompts

| Prompt | Default | Description |
|---|---|---|
| Strategy JSON path | `pathfinder/strategy.json` | Path to your exported strategy JSON |
| RSI fuzz range % | `30` | ±% swept around the base RSI period and threshold |
| MA fuzz range % | `30` | ±% swept around MA window lengths |
| CumRet fuzz range % | `30` | ±% swept around cumulative return windows |
| Price/MA cross fuzz % | `30` | ±% swept around MA cross window lengths |
| MaxDD fuzz range % | `30` | ±% swept around max drawdown windows |
| Threshold step size | `0.5` | Step between threshold values in the sweep |
| Period step size | `1` | Step between period values in the sweep |
| Primary strategy asset | `TQQQ` | Used for the "vs primary" beat-rate stat |
| Start date | `2015-01-01` | Backtest window start |
| End date | `2026-03-15` | Backtest window end |

Press Enter to accept any default.

---

## Reading the Report

Open the generated `fuzz_report_YYYYMMDD_HHMMSS.html` in any browser — no server needed, it's fully self-contained.

### Sidebar

Every IF condition in your strategy appears as a node in the left sidebar, grouped by sub-strategy sleeve. Click any node to open its charts.

The coloured dot and badge next to each node indicate **fragility** — how much the win rate varies when parameters are nudged away from the strategy's fitted values:

| Colour | Label | What it means |
|---|---|---|
| 🟢 Green | Robust | Win rate is consistent across the sweep — high confidence this edge is real |
| 🟡 Yellow | Stable | Minor variation — probably fine |
| 🟠 Orange | Moderate | The exact parameter values matter more than ideal |
| 🔴 Red | Fragile | Win rate spikes at the fitted value and drops off quickly |
| 🟣 Purple | Very Fragile | Likely overfit — treat with significant scepticism |

### Charts

**2D heatmap** — shown for RSI vs fixed threshold, CumRet vs fixed, MaxDD vs fixed, and MA/EMA cross conditions. Rows = indicator period, columns = threshold or second window. Each cell is coloured red → green by win rate. A healthy condition shows a broad green plateau. A lone spike of green surrounded by red is a warning sign.

**1D line chart** — shown for RSI vs RSI, MA vs MA, Price vs MA, and EMA cross conditions where there is no threshold to sweep, only window length. A flat line = robust. A sharp peak = fragile.

### Allocation tabs

When a condition gates multiple possible holdings (e.g. a top-1 filter choosing between SQQQ and TLT), each allocation gets its own tab. The sweep for each tab measures whether *that specific asset* outperformed BIL on days the condition fired.

### Right panel — cell stats

Hover any cell to preview its stats. Click to pin it (blue border). Click a second cell to pin a comparison (yellow border). The right panel shows:

| Stat | Description |
|---|---|
| Win Rate | % of signal days where the allocated asset beat BIL |
| Profit Factor | Gross gains ÷ gross losses on signal days. >1.5 is solid |
| Trades (n) | Days the condition fired. Low n = unreliable win rate |
| Score | Win Rate × log(n) — penalises high win rates from tiny samples |
| vs [primary] | % of signal days where the allocated asset beat your primary asset |

When two cells are pinned, a Δ row shows the difference between them.

---

## How It Works

### Condition extraction

`fuzz_tester.py` walks the strategy JSON tree recursively, extracting every IF condition it finds — including nested conditions, else branches, and asset-selection sort filters. Each condition is categorised by type (RSI_fixed, RSI_vs_RSI, Price_vs_MA, MA_vs_MA, CumRet_vs_CumRet, MaxDD_fixed, EMA_vs_MA, etc.) which determines how it's swept.

### The sweep

For each condition × allocation pair:

1. The base parameter values are read from the strategy JSON
2. A fuzz range is applied (default ±30%) to produce a grid of candidate values
3. For each point in the grid, the condition is evaluated against historical price data over the configured date range
4. On days the condition fires, the next-day return of the allocated asset is measured against BIL
5. Win rate, profit factor, and beat-rate vs the primary asset are recorded for that point

The resulting grid is rendered as a heatmap (2D) or line chart (1D) in the report.

### Fragility scoring

Fragility is computed as the coefficient of variation (std / mean) of the win rate across all sweep points, normalised to 0–1. Higher = more fragile.

### Price data

Price data is downloaded from Tiingo on first run and cached as CSVs in `strategy_engine/data/`. On subsequent runs, the tool checks freshness and only re-downloads stale tickers. The data directory is gitignored — each user downloads their own copy.

---

## Supported Condition Types

| Category | Description | Chart type |
|---|---|---|
| `RSI_fixed` | RSI compared to a fixed threshold | 2D heatmap |
| `RSI_vs_RSI` | RSI of one asset vs RSI of another | 1D line |
| `CumRet_fixed` | Cumulative return vs a fixed threshold | 2D heatmap |
| `CumRet_vs_CumRet` | Cumulative return of one asset vs another | 1D line |
| `MaxDD_fixed` | Max drawdown vs a fixed threshold | 2D heatmap |
| `MaxDD_vs_MaxDD` | Max drawdown of one asset vs another | 1D line |
| `Price_vs_MA` | Price vs its own moving average | 1D line |
| `MA_vs_MA` | MA of one asset vs MA of another (2D window sweep) | 2D heatmap |
| `Price_vs_EMA` | Price vs its own EMA | 1D line |
| `EMA_vs_MA` | EMA vs MA cross (2D window sweep) | 2D heatmap |
| `EMA_vs_EMA` | EMA of one asset vs EMA of another | 1D line |

---

## Project Structure

```
composer_json_fuzz_tester/
├── fuzz_tester.py              # Main script — run this
├── requirements.txt
├── README.md
├── .gitignore
│
├── pathfinder/
│   ├── strategy.json           # Your strategy export (gitignored, add your own)
│   └── strategy_paths.py       # Strategy tree walker (shared utility)
│
└── strategy_engine/
    ├── .env                    # Your API keys (gitignored, create from .env.example)
    ├── .env.example            # Key format reference
    ├── config/
    │   └── template.yaml       # Engine config
    ├── data/                   # Downloaded price CSVs (gitignored, auto-generated)
    └── src/
        ├── config_loader.py    # Loads .env and config
        ├── data_loader.py      # Tiingo download + freshness checks
        ├── data_alignment.py   # Multi-asset DataFrame builder
        └── indicators.py       # RSI, SMA, EMA, CumRet implementations
```

---

## Notes

- The HTML report is fully self-contained — share it with anyone, no dependencies required to view it.
- Conditions with fewer than 20 matching days in the backtest window are skipped and shown as errors in the report. This is expected for very restrictive conditions or short date ranges.
- `pathfinder/*.json` is gitignored — add your own strategy file, it won't be tracked.
- Price data in `strategy_engine/data/` is gitignored. Teammates download their own copy on first run.