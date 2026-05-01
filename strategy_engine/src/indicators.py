import pandas as pd
import numpy as np

def calculate_sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def calculate_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period):
    """
    Relative Strength Index (Wilder's Smoothing)
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.replace(np.inf, 100)
    rsi.iloc[:period] = np.nan

    return rsi

def calculate_cumret(series, period):
    """Cumulative Return over a rolling window (percentage, not decimal)."""
    return series.pct_change(periods=period) * 100
