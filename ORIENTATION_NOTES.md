# ORIENTATION_NOTES.md

## Purpose

This file is a durable "verified behavior" reference for future agent sessions.  
It supplements `HANDOFF.md` with code-validated facts and current assumptions.

## Verified Current Behavior (as of 2026-04-27)

- Entrypoint is `fuzz_tester.py` and remains intentionally monolithic.
- Strategy extraction now handles all three observed condition shapes:
  - direct `if-child` fields (`lhs-fn`, `rhs-fn` / `rhs-fixed-value?`)
  - nested `condition` payload with `condition-type: compound`
  - nested atomic payloads (`binary-compound`, `binary`)
- Condition extraction no longer emits malformed placeholder conditions for compound payloads.
- `MaxDD` and `MAReturn` are now separate indicator calculations:
  - `MaxDD`: strict rolling peak-to-trough drawdown (%)
  - `MAReturn`: rolling mean of daily returns (decimal)
- Sweeps still compare endpoint next-day return versus `BIL` as the baseline.
- Report output filename includes JSON stem and timestamp.

## Production/Runtime Fixes Applied

- `strategy_engine/src/data_loader.py`:
  - Fixed undefined variables in Tiingo-date probe and file write path.
  - Added HTTP 429 retry with exponential backoff in `download_ticker_data`.
  - Added small pacing delay after each successful rebuild in freshness updates.

## Sample Validation Results

Using local sample strategies in `pathfinder/`:

- `any_all_example.json` -> 6 extracted conditions, 0 malformed/unknown placeholders.
- `bestsignals3.json` -> 37 extracted conditions, 0 malformed/unknown placeholders.

These checks validate extraction shape handling, not strategy alpha quality.

## Known Remaining Direction

- Desired end-state: move condition families toward 2D sweeps where feasible.
- Some categories remain 1D by current implementation and should be evaluated for 2D expansion.
- If 2D conversion creates large runtime increases, consider bounded grids or adaptive sampling.

## Assumptions to Re-Verify Periodically

- `pathfinder/strategy_paths.py` is not required by `fuzz_tester.py` runtime path.
- Tiingo limits and free-tier behavior may change; retry and pacing should be tuned from real run logs.
- Composer JSON schema may evolve; extraction should be validated against new exports after platform updates.
