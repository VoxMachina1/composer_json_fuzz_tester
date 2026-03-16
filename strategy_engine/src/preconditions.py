import operator
import pandas as pd
from pathlib import Path

# Try importing dependencies for testing purposes
try:
    from config_loader import load_config
    from data_alignment import build_master_dataframe
    from indicators import add_indicator
except ImportError:
    from .config_loader import load_config
    from .data_alignment import build_master_dataframe
    from .indicators import add_indicator

def evaluate_preconditions(df, preconditions):
    """
    Evaluates a list of precondition dictionaries using AND logic.
    Creates a 'precondition_pass' column in the DataFrame (1=Pass, 0=Fail).
    """
    df = df.copy()
    
    # If no preconditions are defined, default to everything passing (1)
    if not preconditions:
        df['precondition_pass'] = 1
        return df

    # Map string operators to actual Python operator functions
    ops = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne
    }
    
    # Initialize a master mask of all True
    master_mask = pd.Series(True, index=df.index)
    
    for cond in preconditions:
        left_col = cond['left']
        op_str = cond['operator']
        right_val = cond['right']
        
        if left_col not in df.columns:
            raise ValueError(f"Precondition column '{left_col}' not found in DataFrame.")
            
        if op_str not in ops:
            raise ValueError(f"Unsupported operator '{op_str}' in preconditions.")
            
        op_func = ops[op_str]
        
        # Check if 'right' is a column name or a static numeric value
        if isinstance(right_val, str) and right_val in df.columns:
            comparison_series = df[right_val]
        else:
            try:
                comparison_series = float(right_val)
            except ValueError:
                raise ValueError(f"Precondition right side '{right_val}' must be a column name or a number.")
        
        # Evaluate this single condition
        current_mask = op_func(df[left_col], comparison_series)
        
        # AND it with the master mask
        master_mask = master_mask & current_mask
        
    # Convert boolean mask to 1s and 0s
    df['precondition_pass'] = master_mask.astype(int)
    
    return df

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Preconditions Evaluation...")
        
        # Load the updated config
        cfg, _ = load_config()
        preconds = cfg.get('preconditions',[])
        
        # Build test dataframe
        df = build_master_dataframe("QQQ", "SPY", "SPY", data_directory)
        df = add_indicator(df, "benchmark", "SMA", 200)
        df = df.dropna().reset_index(drop=True)
        
        # Evaluate
        df = evaluate_preconditions(df, preconds)
        
        # Verify the results
        print("\nColumns after preconditions added:")
        print(df.columns.tolist())
        
        print("\nSample Output (First 5 Rows):")
        pd.set_option('display.max_columns', None)
        print(df[['date', 'benchmark_close', 'benchmark_SMA_200', 'precondition_pass']].head(5))
        
        # Verify both passed and failed states exist
        pass_count = df['precondition_pass'].sum()
        fail_count = len(df) - pass_count
        print(f"\nTotal Rows: {len(df)}")
        print(f"Days Passing Preconditions: {pass_count}")
        print(f"Days Failing Preconditions: {fail_count}")
        
        print("\nPASS")
        
    except Exception as e:
        print(f"FAIL: {e}")