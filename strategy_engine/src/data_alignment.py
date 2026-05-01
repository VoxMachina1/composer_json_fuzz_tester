import pandas as pd
from pathlib import Path

def load_ticker_csv(ticker, data_dir):
    """
    Reads the price CSV for a given ticker.
    Converts 'date' to a pandas datetime object and sorts chronologically.
    """
    safe_ticker = ticker.replace('/', '-').replace('.', '-')
    file_path = data_dir / f"{safe_ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'close']]
