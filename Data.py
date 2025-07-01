import logging
import os
from typing import List, Dict, Optional

import pandas as pd

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def load_market_data(
    symbols: List[str],
    data_dir: str = '.'
) -> Dict[str, pd.DataFrame]:
    """
    Load historical OHLCV data from local CSV files for each symbol.
    - symbols: list of stock tickers, e.g. ['AAPL', 'MSFT', 'AMZN']
    - data_dir: directory containing <symbol>.csv files with columns: Date, Open, High, Low, Close, optionally Volume.

    Returns a dict mapping symbol to cleaned DataFrame indexed by datetime.
    """
    data: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        filepath = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.isfile(filepath):
            logger.warning(f"File not found for {sym}: {filepath}, skipping.")
            continue
        try:
            df = pd.read_csv(
                filepath,
                parse_dates=['Date'],
                index_col='Date'
            )
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            continue

        # Normalize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            # CSV might use 'Close', 'Adj Close', or 'Close/Last'
            'Close': 'close',
            'Adj Close': 'close',
            'Close/Last': 'close',
            'Volume': 'volume'
        })

                # Convert financial columns to numeric (strip any non-numeric characters)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True).astype(float)

        # Keep only required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns for {sym}: {missing}, skipping.")
            continue
        df = df[required]

        # Sort, fill, drop NaNs
        df = df.sort_index().ffill().bfill().dropna()
        if df.empty:
            logger.warning(f"Data for {sym} empty after cleaning, skipping.")
            continue

        data[sym] = df
        logger.info(f"Loaded {len(df)} rows for {sym} from {filepath}.")

    return data


def get_live_price(
    symbol: str,
    data: Dict[str, pd.DataFrame]
) -> Optional[float]:
    """
    Return the most recent closing price for the given symbol from preloaded data dict.
    """
    df = data.get(symbol)
    if df is None or df.empty:
        logger.warning(f"No data available for live price lookup: {symbol}")
        return None
    try:
        return float(df['close'].iloc[-1])
    except Exception as e:
        logger.error(f"Error retrieving live price for {symbol}: {e}")
        return None


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features: returns, 10/50-day moving averages, 10-day volatility.
    Returns DataFrame of features indexed by date.
    """
    feats = pd.DataFrame(index=df.index)
    feats['returns'] = df['close'].pct_change()
    feats['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
    feats['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
    feats['vol10'] = feats['returns'].rolling(window=10, min_periods=1).std()
    feats = feats.ffill().bfill().dropna()
    return feats

