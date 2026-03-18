import pandas as pd
from pathlib import Path

# Try importing dependencies for testing purposes
try:
    from data_alignment import build_master_dataframe
    from indicators import add_indicator
except ImportError:
    from .data_alignment import build_master_dataframe
    from .indicators import add_indicator

def evaluate_preconditions(df, precondition_string):
    """
    Evaluates a boolean string logic statement (e.g. 'SPY_RSI_10 > 80 and QQQ_RSI_10 > 80')
    Creates a 'precondition_pass' column in the DataFrame (1=Pass, 0=Fail).
    """
    df = df.copy()
    
    # If no preconditions are defined, default to everything passing (1)
    if not precondition_string or str(precondition_string).strip().lower() in ["", "none", "[]"]:
        df['precondition_pass'] = 1
        return df

    try:
        # pandas.eval() magically interprets our string as math logic
        mask = df.eval(precondition_string)
        df['precondition_pass'] = mask.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to evaluate precondition string: '{precondition_string}'. Error: {e}")
        
    return df

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Boolean String Preconditions...")
        
        # 1. Build test dataframe with filter assets
        # Using XLF as signal, but if XLF.csv doesn't exist, fallback to QQQ
        test_signal = "XLF" if (data_directory / "XLF.csv").exists() else "QQQ"
        
        df = build_master_dataframe(test_signal, "VIXY", "SPY", data_directory, filter_assets=["SPY", "QQQ"])
        
        # 2. Add the indicators that our string will look for
        df = add_indicator(df, "SPY", "RSI", 10)
        df = add_indicator(df, "QQQ", "RSI", 10)
        df = df.dropna().reset_index(drop=True)
        
        # 3. Define our new awesome boolean string
        # We use 70 for the test to ensure we get a decent number of hits printed
        test_string = "SPY_RSI_10 > 70 and QQQ_RSI_10 > 70"
        
        # 4. Evaluate
        df = evaluate_preconditions(df, test_string)
        
        # 5. Output Verification
        print(f"\nEvaluated String: '{test_string}'")
        
        pass_count = df['precondition_pass'].sum()
        fail_count = len(df) - pass_count
        
        print(f"Total Rows: {len(df)}")
        print(f"Days Passing Preconditions: {pass_count}")
        print(f"Days Failing Preconditions: {fail_count}")
        
        print("\nSample Output (First 3 Passed Rows):")
        pd.set_option('display.max_columns', None)
        passed_sample = df[df['precondition_pass'] == 1].head(3)
        print(passed_sample[['date', 'SPY_RSI_10', 'QQQ_RSI_10', 'precondition_pass']])
        
        print("\nPASS")
        
    except Exception as e:
        print(f"FAIL: {e}")