import operator
import pandas as pd
from pathlib import Path

# Try importing dependencies for testing purposes
try:
    from config_loader import load_config
    from data_alignment import build_master_dataframe
    from indicators import add_indicator
    from preconditions import evaluate_preconditions
except ImportError:
    from .config_loader import load_config
    from .data_alignment import build_master_dataframe
    from .indicators import add_indicator
    from .preconditions import evaluate_preconditions

def generate_signals(df, indicator_col, operator_str, threshold):
    """
    Evaluates the primary signal condition and combines it with preconditions.
    Creates a 'signal_active' column (1=Active, 0=Inactive).
    """
    df = df.copy()
    
    # Map string operators to actual Python functions
    ops = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne
    }
    
    if operator_str not in ops:
        raise ValueError(f"Unsupported signal operator '{operator_str}'.")
        
    if indicator_col not in df.columns:
        raise ValueError(f"Indicator column '{indicator_col}' not found in DataFrame.")
        
    op_func = ops[operator_str]
    
    # 1. Evaluate primary signal
    primary_signal = op_func(df[indicator_col], threshold)
    
    # 2. Check preconditions (Default to True if column is missing)
    if 'precondition_pass' in df.columns:
        precond_mask = df['precondition_pass'] == 1
    else:
        precond_mask = pd.Series(True, index=df.index)
        
    # 3. Final signal requires BOTH to be True (AND logic)
    final_signal = primary_signal & precond_mask
    
    df['signal_active'] = final_signal.astype(int)
    
    return df

# --- TEST ---
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    
    try:
        print("Testing Signal Generation...")
        
        # 1. Load config
        cfg, _ = load_config()
        preconds = cfg.get('preconditions',[])
        sig_op = cfg.get('signal_operator', '>=')
        threshold = cfg.get('threshold_start', 50.0)
        indicator_period = cfg.get('indicator_period', 10)
        indicator_name = cfg.get('indicator', 'RSI')
        
        # Target column name (e.g., signal_RSI_10)
        sig_col = f"signal_{indicator_name}_{indicator_period}"
        
        # 2. Build DataFrame and Indicators
        df = build_master_dataframe("QQQ", "SPY", "SPY", data_directory)
        df = add_indicator(df, "signal", indicator_name, indicator_period)
        df = add_indicator(df, "benchmark", "SMA", 200)
        df = df.dropna().reset_index(drop=True)
        
        # 3. Apply Logic
        df = evaluate_preconditions(df, preconds)
        df = generate_signals(df, sig_col, sig_op, threshold)
        
        # 4. Results Output
        print("\nColumns after signals added:")
        print(df.columns.tolist())
        
        print(f"\nEvaluating: {sig_col} {sig_op} {threshold} AND Preconditions")
        
        active_count = df['signal_active'].sum()
        inactive_count = len(df) - active_count
        print(f"Total Rows: {len(df)}")
        print(f"Days Signal ACTIVE (Target Asset): {active_count}")
        print(f"Days Signal INACTIVE (Benchmark Asset): {inactive_count}")
        
        # Show a slice where the signal is active to verify
        print("\nSample Output (First 5 Active Rows):")
        pd.set_option('display.max_columns', None)
        active_sample = df[df['signal_active'] == 1].head(5)
        print(active_sample[['date', sig_col, 'benchmark_SMA_200', 'precondition_pass', 'signal_active']])
        
        print("\nPASS")
        
    except Exception as e:
        print(f"FAIL: {e}")