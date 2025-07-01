import logging
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
import requests

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


def fetch_symbol_list(api_key: str, market: str = "stocks", limit: int = 1000) -> List[str]:
    """
    Fetch a listing of active stock symbols from Polygon.
    NOTE: Polygon paginates results; this fetches only the first `limit` tickers.
    """
    url = f"{POLYGON_BASE}/v3/reference/tickers"
    params = {
        "apiKey": api_key,
        "market": market,
        "active": "true",
        "limit": limit
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        symbols = [item["ticker"] for item in data.get("results", [])]
        logger.info(f"Fetched {len(symbols)} symbols from Polygon (market={market}).")
        return symbols
    except Exception as e:
        logger.error(f"Failed to fetch symbol list from Polygon: {e}")
        return []


def fetch_polygon_ohlcv(
    symbol: str,
    api_key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjusted: bool = True
) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV data for a symbol via Polygon's aggregates endpoint.
    - start_date/end_date as 'YYYY-MM-DD'. Defaults to last 1 year.
    """
    # Determine date window
    if end_date:
        to_dt = pd.to_datetime(end_date)
    else:
        to_dt = datetime.utcnow()
    if start_date:
        from_dt = pd.to_datetime(start_date)
    else:
        from_dt = to_dt - timedelta(days=365)

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/range/1/day/{from_dt.date()}/{to_dt.date()}"
    params = {
        "apiKey": api_key,
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        payload = r.json()
        if "results" not in payload:
            raise ValueError(f"No results in response: {payload}")
        df = pd.DataFrame(payload["results"])
        # rename & reindex
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        logger.info(f"Fetched {len(df)} rows for {symbol} via Polygon.")
        return df
    except Exception as e:
        logger.error(f"Polygon OHLCV fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def load_market_data_api(
    symbols: List[str],
    api_key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load historical data for given symbols via Polygon API.
    Optionally filter between ISO dates start_date and end_date (YYYY-MM-DD).
    Returns a dict mapping symbol -> cleaned DataFrame.
    """
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_polygon_ohlcv(sym, api_key, start_date, end_date)
        if df.empty:
            logger.warning(f"No data for {sym}, skipping.")
            continue
        # Clean missing
        df = df.ffill().bfill().dropna()
        if df.empty:
            logger.warning(f"Data for {sym} empty after cleaning, skipping.")
            continue
        data[sym] = df
        logger.info(f"Loaded {len(df)} rows for {sym} (Polygon).")
    return data


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features: returns, 10/50-day moving averages, 10-day volatility.
    (Unchanged.)
    """
    feats = pd.DataFrame(index=df.index)
    feats['returns'] = df['close'].pct_change()
    feats['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
    feats['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
    feats['vol10'] = feats['returns'].rolling(window=10, min_periods=1).std()
    feats = feats.ffill().bfill().dropna()
    return feats


def get_live_price_api(symbol: str, api_key: str) -> Optional[Dict[str, float]]:
    """
    Fetch yesterday's OHLCV (previous close) for a symbol via Polygon.
    Returns a dict with keys: open, high, low, close, volume, or None on error.
    """
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/prev"
    params = {
        "apiKey": api_key,
        "adjusted": "true"
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        if not results:
            raise ValueError(f"No previous data for {symbol}")
        agg = results[0]
        return {
            "open": agg["o"],
            "high": agg["h"],
            "low": agg["l"],
            "close": agg["c"],
            "volume": agg["v"]
        }
    except Exception as e:
        logger.error(f"Live (prev) price fetch failed for {symbol}: {e}")
        return None


