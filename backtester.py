"""
TREND SURFER - OANDA ENHANCED VERSION WITH PIN BAR ANALYSIS
==========================================================

Professional FX backtesting system with OANDA data integration.
Enhanced pin bar detection with comprehensive trade management and detailed rejection tracking.

USAGE:
Run with Streamlit: streamlit run trend_surfer_oanda.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta, time
import pytz
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import threading
import time as time_module
from functools import lru_cache


# ================================
# OANDA API CONFIGURATION
# ================================

class OandaAPI:
    """OANDA API client for fetching market data
    
    Required secrets.toml format:
    oanda_api_key = "your_api_key_here"
    oanda_account_id = "your_account_id_here"
    """
    
    def __init__(self):
        self.api_key = st.secrets["oanda_api_key"]
        self.account_id = st.secrets["oanda_account_id"]
        self.base_url = "https://api-fxtrade.oanda.com/v3"  # Live API
        # Use "https://api-fxpractice.oanda.com/v3" for practice
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # OANDA granularity mapping
        self.granularity_map = {
            '15m': 'M15',
            '30m': 'M30', 
            '1h': 'H1',
            '4h': 'H4',
            '1d': 'D'
        }
    
    def convert_symbol_to_oanda(self, symbol: str) -> str:
        """Convert Yahoo Finance symbol to OANDA instrument"""
        symbol_map = {
            'EURUSD=X': 'EUR_USD',
            'GBPUSD=X': 'GBP_USD',
            'USDJPY=X': 'USD_JPY',
            'AUDUSD=X': 'AUD_USD',
            'USDCAD=X': 'USD_CAD',
            'USDCHF=X': 'USD_CHF',
            'NZDUSD=X': 'NZD_USD',
            'EURJPY=X': 'EUR_JPY',
            'GBPJPY=X': 'GBP_JPY',
            'EURGBP=X': 'EUR_GBP',
            'AUDCAD=X': 'AUD_CAD',
            'AUDCHF=X': 'AUD_CHF',
            'AUDNZD=X': 'AUD_NZD',
            'CADJPY=X': 'CAD_JPY',
            'CHFJPY=X': 'CHF_JPY'
        }
        return symbol_map.get(symbol, symbol.replace('=X', '').replace('USD', '_USD'))
    
    @st.cache_data(ttl=36000)  # Cache for 30 minutes
    def fetch_candles(_self, instrument: str, granularity: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch candlestick data from OANDA with caching"""
        try:
            oanda_instrument = _self.convert_symbol_to_oanda(instrument)
            oanda_granularity = _self.granularity_map.get(granularity, 'H1')
            
            url = f"{_self.base_url}/instruments/{oanda_instrument}/candles"
            
            params = {
                'granularity': oanda_granularity,
                'from': start_time,
                'to': end_time,
                'price': 'M',  # Mid prices
                'includeFirst': 'true'
            }
            
            response = requests.get(url, headers=_self.headers, params=params)
            
            if response.status_code != 200:
                st.error(f"OANDA API Error: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            ohlcv_data = []
            for candle in candles:
                if candle['complete']:  # Only use complete candles
                    mid = candle['mid']
                    ohlcv_data.append({
                        'timestamp': pd.to_datetime(candle['time']),
                        'Open': float(mid['o']),
                        'High': float(mid['h']),
                        'Low': float(mid['l']),
                        'Close': float(mid['c']),
                        'Volume': float(candle.get('volume', 0))
                    })
            
            if not ohlcv_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv_data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Ensure timezone naive for compatibility
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching OANDA data: {str(e)}")
            return pd.DataFrame()


# ================================
# ENHANCED UTILITY FUNCTIONS
# ================================

def ensure_timezone_naive(dt):
    """Convert timezone-aware datetime to naive UTC datetime"""
    if dt is None:
        return None
    if hasattr(dt, 'tz_localize'):
        # pandas Timestamp
        if dt.tz is not None:
            return dt.tz_convert('UTC').tz_localize(None)
        return dt
    elif hasattr(dt, 'tzinfo'):
        # Python datetime
        if dt.tzinfo is not None:
            return dt.astimezone(pytz.UTC).replace(tzinfo=None)
        return dt
    return dt

def safe_datetime_subtract(dt1, dt2):
    """Safely subtract two datetime objects, handling timezone issues"""
    try:
        dt1_naive = ensure_timezone_naive(dt1)
        dt2_naive = ensure_timezone_naive(dt2)
        return dt1_naive - dt2_naive
    except Exception as e:
        st.warning(f"Datetime subtraction issue: {e}")
        return timedelta(0)

def normalize_datetime_index(df):
    """Normalize datetime index to be timezone-naive"""
    if df.empty:
        return df
    
    df = df.copy()
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df

def convert_utc_to_bst(utc_dt):
    """Convert UTC datetime to BST (British Summer Time)"""
    if utc_dt is None:
        return None
    
    utc_dt = ensure_timezone_naive(utc_dt)
    utc_tz = pytz.UTC
    bst_tz = pytz.timezone('Europe/London')
    
    # Make UTC timezone aware
    utc_aware = utc_tz.localize(utc_dt)
    # Convert to BST
    bst_aware = utc_aware.astimezone(bst_tz)
    
    return bst_aware

def is_valid_trading_time(utc_dt):
    """Check if time is within valid trading hours (3:00-17:00 BST)"""
    bst_dt = convert_utc_to_bst(utc_dt)
    if bst_dt is None:
        return False
    
    # Extract time component
    current_time = bst_dt.time()
    
    # Trading hours: 3:00 AM to 5:00 PM BST
    start_time = time(3, 0)  # 3:00 AM
    end_time = time(17, 0)   # 5:00 PM
    
    return start_time <= current_time <= end_time

def should_close_trade_time(utc_dt):
    """Check if trade should be closed due to time constraints (20:00 BST)"""
    bst_dt = convert_utc_to_bst(utc_dt)
    if bst_dt is None:
        return False
    
    # Extract time component
    current_time = bst_dt.time()
    
    # Force close time: 8:00 PM BST
    close_time = time(20, 0)  # 8:00 PM
    
    return current_time >= close_time


# ================================
# CORE DATA STRUCTURES
# ================================

class PinBarType(Enum):
    """Pin bar pattern types"""
    NONE = "none"
    BULLISH = "bullish"
    BEARISH = "bearish"


class TradeDirection(Enum):
    """Trade direction types"""
    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    """Trade execution status"""
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    CLOSED_BREAKEVEN = "closed_breakeven"
    CLOSED_TIME = "closed_time"
    NOT_TRIGGERED = "not_triggered"


@dataclass
class Candle:
    """Enhanced candlestick data structure"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    def is_bullish(self, doji_threshold: float = 0.0001) -> bool:
        """Check if candle is bullish (green) or doji"""
        body_size = abs(self.close - self.open)
        candle_range = self.high - self.low
        
        if candle_range == 0:
            return True  # Treat as doji
        
        # Check if it's a doji (very small body relative to range)
        if body_size / candle_range <= 0.1:  # Doji threshold
            return True
            
        return self.close >= self.open
    
    def is_bearish(self, doji_threshold: float = 0.0001) -> bool:
        """Check if candle is bearish (red) or doji"""
        body_size = abs(self.close - self.open)
        candle_range = self.high - self.low
        
        if candle_range == 0:
            return True  # Treat as doji
        
        # Check if it's a doji (very small body relative to range)
        if body_size / candle_range <= 0.1:  # Doji threshold
            return True
            
        return self.close < self.open


@dataclass
class Trade:
    """Enhanced trade execution data structure"""
    entry_time: pd.Timestamp
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_pips: float = 0.0
    risk_amount: float = 0.0
    pin_bar_data: Optional[Dict] = None
    lot_size: float = 0.0
    pnl_usd: float = 0.0
    forced_close_reason: Optional[str] = None

    def set_exit(self, exit_time: pd.Timestamp, exit_price: float, status: TradeStatus, reason: str = None):
        """Set exit details for the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = status
        if reason:
            self.forced_close_reason = reason


@dataclass
class PinBarResult:
    """Pin bar detection result with trade outcome and detailed rejection tracking"""
    timestamp: pd.Timestamp
    pin_bar_type: PinBarType
    strength: float
    open: float
    high: float
    low: float
    close: float
    is_bullish_candle: bool
    body_size: float
    ema6: float
    ema18: float
    ema50: float
    sma200: float
    bst_time: str
    in_trading_hours: bool
    trade_attempted: bool = False
    trade_success: bool = False
    rejection_reason: str = ""
    trade_id: Optional[int] = None


@dataclass
class BacktestResults:
    """Enhanced comprehensive backtest results"""
    trades: List[Trade] = field(default_factory=list)
    pin_bars: List[PinBarResult] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    symbol: str = ""
    start_date: datetime = None
    end_date: datetime = None
    risk_reward_ratio: float = 2.0
    total_pin_bars: int = 0
    valid_trades: int = 0
    data_1h: pd.DataFrame = field(default_factory=pd.DataFrame)
    debug_info: Dict = field(default_factory=dict)
    trading_hours_stats: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure backward compatibility"""
        if not hasattr(self, 'pin_bars'):
            self.pin_bars = []
        if not hasattr(self, 'trading_hours_stats'):
            self.trading_hours_stats = {}


# ================================
# ENHANCED DATA FETCHER (OANDA)
# ================================

class DataFetcher:
    """Enhanced data fetching with OANDA API"""
    
    def __init__(self):
        self.oanda_api = OandaAPI()
    
    @st.cache_data(ttl=300)
    def fetch_data(_self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data from OANDA with enhanced error handling"""
        try:
            # Convert datetime strings to ISO format for OANDA
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            start_iso = start_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            end_iso = end_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            
            data = _self.oanda_api.fetch_candles(symbol, interval, start_iso, end_iso)
            
            if data.empty:
                st.warning(f"No data available for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Standardize columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            standardized_data = pd.DataFrame(index=data.index)
            
            for col in required_columns:
                if col in data.columns:
                    standardized_data[col] = data[col]
                elif col == 'Volume':
                    standardized_data[col] = 0.0
                else:
                    st.error(f"Critical column {col} missing for {symbol}")
                    return pd.DataFrame()
            
            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                standardized_data[col] = pd.to_numeric(standardized_data[col], errors='coerce')
            
            # Remove NaN rows
            standardized_data = standardized_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            return standardized_data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_multi_timeframe_data(symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Enhanced multi-timeframe data fetching with caching"""
        fetcher = DataFetcher()
        
        # Ensure dates are timezone-naive
        start_date = ensure_timezone_naive(start_date)
        end_date = ensure_timezone_naive(end_date)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Flexible timeframe configuration - support up to 5 years
        timeframes_config = {
            '15m': {
                'start': start_str,  # Use full requested range
                'end': end_str
            },
            '30m': {
                'start': start_str,  # Use full requested range
                'end': end_str
            },
            '1h': {
                'start': start_str,
                'end': end_str
            },
            '4h': {
                'start': start_str,
                'end': end_str
            }
        }
        
        data = {}
        
        for tf, config in timeframes_config.items():
            try:
                print(f"Fetching {tf} data for {symbol}...")
                df = fetcher.fetch_data(symbol, tf, config['start'], config['end'])
                
                if not df.empty:
                    data[tf] = df
                    print(f"✓ {tf}: {len(df)} candles retrieved")
                else:
                    print(f"✗ {tf}: No data retrieved")
                    
            except Exception as e:
                print(f"✗ {tf}: Error during fetch - {str(e)}")
                continue
                
        return data


# ================================
# ENHANCED PIN BAR DETECTOR
# ================================

class PinBarDetector:
    """Enhanced pin bar detection with proper candle color validation"""
    
    def __init__(self, 
                 min_wick_ratio: float = 0.55,
                 max_body_ratio: float = 0.4,
                 max_opposite_wick: float = 0.3):
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        self.max_opposite_wick = max_opposite_wick
    
    def detect_pin_bar(self, candle: Candle, ema6: float, ema18: float, 
                      ema50: float, sma200: float) -> Tuple[PinBarType, float]:
        """Pin bar detection with proper candle color validation"""
        # Calculate candle metrics
        candle_range = candle.high - candle.low
        if candle_range == 0:
            return PinBarType.NONE, 0.0
        
        body_size = abs(candle.close - candle.open)
        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low
        
        # Calculate ratios
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range
        body_ratio = body_size / candle_range
        
        # Enhanced trend analysis
        uptrend_strong = (ema6 > ema18 > ema50 > sma200) and (candle.close > ema6)
        uptrend_moderate = (ema6 > ema18) and (ema6 > sma200) and (candle.close > ema18)
        uptrend = uptrend_strong or uptrend_moderate
        
        downtrend_strong = (ema6 < ema18 < ema50 < sma200) and (candle.close < ema6)
        downtrend_moderate = (ema6 < ema18) and (ema6 < sma200) and (candle.close < ema18)
        downtrend = downtrend_strong or downtrend_moderate
        
        # Bullish pin bar detection
        if (lower_wick_ratio >= self.min_wick_ratio and
            body_ratio <= self.max_body_ratio and
            upper_wick_ratio <= self.max_opposite_wick and
            uptrend):
            
            if candle.is_bullish():
                # Check EMA touch
                ema_touch = abs(candle.low - ema6) / ema6 <= 0.015
                
                if ema_touch:
                    strength = self._calculate_strength(lower_wick_ratio, body_ratio, upper_wick_ratio)
                    return PinBarType.BULLISH, strength
        
        # Bearish pin bar detection  
        elif (upper_wick_ratio >= self.min_wick_ratio and
              body_ratio <= self.max_body_ratio and
              lower_wick_ratio <= self.max_opposite_wick and
              downtrend):
            
            if candle.is_bearish():
                # Check EMA touch
                ema_touch = abs(candle.high - ema6) / ema6 <= 0.015
                
                if ema_touch:
                    strength = self._calculate_strength(upper_wick_ratio, body_ratio, lower_wick_ratio)
                    return PinBarType.BEARISH, strength
        
        return PinBarType.NONE, 0.0
    
    def _calculate_strength(self, dominant_wick: float, body_ratio: float, opposite_wick: float) -> float:
        """Calculate pin bar strength score (0-100)"""
        wick_score = min((dominant_wick - 0.55) / 0.35 * 50, 50)
        body_penalty = body_ratio * 25
        opposite_penalty = max(0, (opposite_wick - 0.1)) * 30
        
        strength = max(0, min(100, wick_score - body_penalty - opposite_penalty))
        return strength


# ================================
# TRADE MANAGER
# ================================

class TradeManager:
    """Manages active trades and enforces one-trade-per-pair constraint"""
    
    def __init__(self):
        self.active_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        
    def can_open_trade(self, symbol: str) -> bool:
        """Check if a new trade can be opened for this symbol"""
        return symbol not in self.active_trades
    
    def open_trade(self, trade: Trade) -> bool:
        """Open a new trade if allowed"""
        if self.can_open_trade(trade.symbol):
            self.active_trades[trade.symbol] = trade
            return True
        return False
    
    def close_trade(self, symbol: str, exit_time: pd.Timestamp, exit_price: float, 
                   status: TradeStatus, reason: str = None) -> Optional[Trade]:
        """Close an active trade"""
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            trade.set_exit(exit_time, exit_price, status, reason)
            
            # Calculate P&L
            self._calculate_trade_pnl(trade)
            
            # Move to closed trades
            self.closed_trades.append(trade)
            del self.active_trades[symbol]
            
            return trade
        
        return None
    
    def get_active_trade(self, symbol: str) -> Optional[Trade]:
        """Get active trade for symbol"""
        return self.active_trades.get(symbol)
    
    def get_all_trades(self) -> List[Trade]:
        """Get all trades (active + closed)"""
        return list(self.active_trades.values()) + self.closed_trades
    
    def force_close_time_expired_trades(self, current_time: pd.Timestamp) -> List[Trade]:
        """Force close trades due to time constraints"""
        closed_trades = []
        
        for symbol, trade in list(self.active_trades.items()):
            if should_close_trade_time(current_time):
                exit_price = trade.entry_price
                closed_trade = self.close_trade(
                    symbol, current_time, exit_price, 
                    TradeStatus.CLOSED_TIME, "Forced close at 20:00 BST"
                )
                if closed_trade:
                    closed_trades.append(closed_trade)
        
        return closed_trades
    
    def _calculate_trade_pnl(self, trade: Trade):
        """Calculate trade P&L"""
        if trade.exit_price is None:
            return
            
        pip_value = self._get_pip_value(trade.symbol)
        
        if trade.direction == TradeDirection.LONG:
            trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_value
        else:
            trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_value
        
        trade.pnl_usd = trade.pnl_pips * trade.lot_size * (10 if 'JPY' in trade.symbol else 1)
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        return 0.01 if 'JPY' in symbol else 0.0001


# ================================
# BACKTESTING ENGINE
# ================================

class TrendSurferBacktester:
    """Enhanced backtesting engine with detailed rejection tracking"""
    
    def __init__(self):
        self.detector = PinBarDetector()
        self.data_fetcher = DataFetcher()
        self.trade_manager = TradeManager()
    
    def run_backtest(self, symbol: str, start_date: datetime, end_date: datetime,
                    risk_reward_ratio: float = 2.0, account_balance: float = 10000.0,
                    risk_percentage: float = 0.01) -> BacktestResults:
        """Enhanced backtest with comprehensive analysis"""
        
        # Reset trade manager for new backtest
        self.trade_manager = TradeManager()
        
        # Ensure timezone-naive dates
        start_date = ensure_timezone_naive(start_date)
        end_date = ensure_timezone_naive(end_date)
        
        # Date optimization - allow up to 5 years of data
        current_date = ensure_timezone_naive(datetime.now())
        max_lookback_days = 5 * 365  # 5 years
        optimized_start = max(start_date, current_date - timedelta(days=max_lookback_days))
        optimized_end = min(end_date, current_date - timedelta(days=1))
        
        # Fetch data
        data = self.data_fetcher.fetch_multi_timeframe_data(symbol, optimized_start, optimized_end)
        
        if not data or '1h' not in data:
            return BacktestResults()
        
        # Debug data quality
        debug_info = {'data_quality': {}}
        for tf, df in data.items():
            debug_info['data_quality'][tf] = {
                'candles': len(df),
                'date_range': f"{df.index[0]} to {df.index[-1]}"
            }
        
        # Detect pin bars with caching
        pin_bars = self._detect_pin_bars_h1(data['1h'])
        
        debug_info['pin_bars_found'] = len(pin_bars)
        print(f"📍 Pin bars detected: {len(pin_bars)}")
        
        # Generate trades
        trades = self._generate_trades_with_constraints(
            pin_bars, data, symbol, risk_reward_ratio, 
            account_balance, risk_percentage, debug_info
        )
        
        print(f"🔄 Trades generated: {len(trades)}")
        
        # Calculate statistics
        statistics = self._calculate_enhanced_statistics(trades, symbol, account_balance, debug_info)
        
        # CRITICAL FIX: Ensure pin_bars are included in results
        results = BacktestResults(
            trades=trades,
            pin_bars=pin_bars,  # Make sure pin_bars are assigned
            statistics=statistics,
            symbol=symbol,
            start_date=optimized_start,
            end_date=optimized_end,
            risk_reward_ratio=risk_reward_ratio,
            total_pin_bars=len(pin_bars),
            valid_trades=len(trades),
            data_1h=data['1h'],
            debug_info=debug_info,
            trading_hours_stats=debug_info.get('trading_hours', {})
        )
        
        print(f"📊 BacktestResults created with {len(results.pin_bars)} pin bars")
        return results
    
    def _detect_pin_bars_h1(self, data_1h: pd.DataFrame) -> List[PinBarResult]:
        """Pin bar detection with comprehensive result tracking"""
        pin_bars = []
        
        if data_1h.empty or len(data_1h) < 50:
            return pin_bars
        
        # Normalize timezone
        data_1h = normalize_datetime_index(data_1h)
        
        # Calculate indicators
        data_1h = data_1h.copy()
        data_1h['EMA6'] = data_1h['Close'].ewm(span=6).mean()
        data_1h['EMA18'] = data_1h['Close'].ewm(span=18).mean()
        data_1h['EMA50'] = data_1h['Close'].ewm(span=50).mean()
        data_1h['SMA200'] = data_1h['Close'].rolling(window=200).mean()
        
        # Start detection after sufficient data for indicators
        start_idx = max(200, 0)  # Always start from index 200 for indicator stability
        
        for i in range(start_idx, len(data_1h)):
            row = data_1h.iloc[i]
            
            # Skip if indicators not available
            if pd.isna(row['EMA6']) or pd.isna(row['EMA18']) or pd.isna(row['SMA200']):
                continue
            
            # Get candle time and BST conversion
            candle_time = ensure_timezone_naive(row.name)
            bst_time_obj = convert_utc_to_bst(candle_time)
            bst_time_str = bst_time_obj.strftime('%m/%d %H:%M BST')
            in_trading_hours = is_valid_trading_time(candle_time)
            
            candle = Candle(
                timestamp=candle_time,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row.get('Volume', 0)
            )
            
            # Use fallback values for missing indicators
            ema50 = row['EMA50'] if not pd.isna(row['EMA50']) else row['EMA18']
            
            # Calculate basic pin bar metrics for analysis
            candle_range = candle.high - candle.low
            if candle_range == 0:
                continue
                
            body_size = abs(candle.close - candle.open)
            upper_wick = candle.high - max(candle.open, candle.close)
            lower_wick = min(candle.open, candle.close) - candle.low
            
            upper_wick_ratio = upper_wick / candle_range
            lower_wick_ratio = lower_wick / candle_range
            body_ratio = body_size / candle_range
            
            # Check if this looks like a potential pin bar
            potential_bullish_pin = (lower_wick_ratio >= 0.45 and body_ratio <= 0.5 and upper_wick_ratio <= 0.4)
            potential_bearish_pin = (upper_wick_ratio >= 0.45 and body_ratio <= 0.5 and lower_wick_ratio <= 0.4)
            
            if potential_bullish_pin or potential_bearish_pin:
                # Detect pin bar
                pin_bar_type, strength = self.detector.detect_pin_bar(
                    candle, row['EMA6'], row['EMA18'], ema50, row['SMA200']
                )
                
                # Create pin bar result
                pin_bar_result = PinBarResult(
                    timestamp=candle_time,
                    pin_bar_type=pin_bar_type,
                    strength=strength,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    is_bullish_candle=candle.is_bullish(),
                    body_size=body_size,
                    ema6=row['EMA6'],
                    ema18=row['EMA18'],
                    ema50=ema50,
                    sma200=row['SMA200'],
                    bst_time=bst_time_str,
                    in_trading_hours=in_trading_hours
                )
                
                pin_bars.append(pin_bar_result)
        
        return pin_bars
    
    def _generate_trades_with_constraints(self, pin_bars: List[PinBarResult], data: Dict[str, pd.DataFrame],
                                        symbol: str, risk_reward_ratio: float, account_balance: float,
                                        risk_percentage: float, debug_info: Dict) -> List[Trade]:
        """Generate trades with time constraints and detailed rejection tracking"""
        
        trading_hours_stats = {
            'total_opportunities': len([pb for pb in pin_bars if pb.pin_bar_type != PinBarType.NONE]),
            'outside_trading_hours': 0,
            'blocked_by_active_trade': 0,
            'successful_entries': 0,
            'time_forced_closes': 0,
            'invalid_candle_color': 0,
            'failed_sma_validation': 0,
            'invalid_trade_levels': 0
        }
        
        trade_id_counter = 1
        
        # Process only valid pin bars
        valid_pin_bars = [pb for pb in pin_bars if pb.pin_bar_type != PinBarType.NONE]
        
        for pin_bar in valid_pin_bars:
            timestamp = pin_bar.timestamp
            pin_type = pin_bar.pin_bar_type
            
            pin_bar.trade_attempted = True
            
            # Check 1: Trading hours constraint
            if not pin_bar.in_trading_hours:
                pin_bar.rejection_reason = "Outside trading hours (3:00-17:00 BST)"
                trading_hours_stats['outside_trading_hours'] += 1
                continue
            
            # Check 2: Candle color validation
            candle_color_valid = False
            if pin_type == PinBarType.BULLISH and pin_bar.is_bullish_candle:
                candle_color_valid = True
            elif pin_type == PinBarType.BEARISH and not pin_bar.is_bullish_candle:
                candle_color_valid = True
            
            if not candle_color_valid:
                color_desc = "green" if pin_bar.is_bullish_candle else "red"
                expected_desc = "green" if pin_type == PinBarType.BULLISH else "red"
                pin_bar.rejection_reason = f"Wrong candle color: {color_desc} candle for {pin_type.value} pin bar (expected {expected_desc})"
                trading_hours_stats['invalid_candle_color'] += 1
                continue
            
            # Check 3: One-trade-per-pair constraint
            if not self.trade_manager.can_open_trade(symbol):
                pin_bar.rejection_reason = "Blocked by existing active trade (one trade per pair limit)"
                trading_hours_stats['blocked_by_active_trade'] += 1
                continue
            
            # Determine trade direction
            if pin_type == PinBarType.BULLISH:
                direction = TradeDirection.LONG
            elif pin_type == PinBarType.BEARISH:
                direction = TradeDirection.SHORT
            else:
                pin_bar.rejection_reason = "Invalid pin bar type"
                continue
            
            # Check 4: Enhanced SMA validation
            if not self._check_sma_conditions_enhanced(pin_bar.close, timestamp, data, direction):
                pin_bar.rejection_reason = f"Failed trend/SMA validation: {direction.value} trade requires price {'above' if direction == TradeDirection.LONG else 'below'} SMA50 on multiple timeframes"
                trading_hours_stats['failed_sma_validation'] += 1
                continue
            
            # Calculate trade levels
            try:
                pin_bar_dict = {
                    'timestamp': pin_bar.timestamp,
                    'type': pin_bar.pin_bar_type,
                    'strength': pin_bar.strength,
                    'open': pin_bar.open,
                    'high': pin_bar.high,
                    'low': pin_bar.low,
                    'close': pin_bar.close,
                    'ema6': pin_bar.ema6
                }
                
                entry_price, stop_loss, take_profit = self._calculate_trade_levels_enhanced(
                    pin_bar_dict, direction, symbol, risk_reward_ratio
                )
            except Exception as e:
                pin_bar.rejection_reason = f"Failed to calculate trade levels: {str(e)}"
                trading_hours_stats['invalid_trade_levels'] += 1
                continue
            
            # Validate trade levels
            if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
                pin_bar.rejection_reason = "Invalid trade levels calculated (negative or zero values)"
                trading_hours_stats['invalid_trade_levels'] += 1
                continue
            
            # Additional validation: reasonable stop distance
            stop_distance_pips = abs(entry_price - stop_loss) / self._get_pip_value(symbol)
            if stop_distance_pips <= 0 or stop_distance_pips > 100:  # Max 100 pips stop
                pin_bar.rejection_reason = f"Unreasonable stop distance: {stop_distance_pips:.1f} pips (must be 1-100 pips)"
                trading_hours_stats['invalid_trade_levels'] += 1
                continue
            
            # Calculate position size
            try:
                lot_size = self._calculate_position_size_enhanced(
                    account_balance, risk_percentage, stop_distance_pips, symbol
                )
                
                if lot_size <= 0:
                    pin_bar.rejection_reason = "Unable to calculate valid position size"
                    trading_hours_stats['invalid_trade_levels'] += 1
                    continue
                    
            except Exception as e:
                pin_bar.rejection_reason = f"Position sizing error: {str(e)}"
                trading_hours_stats['invalid_trade_levels'] += 1
                continue
            
            # Create trade
            trade = Trade(
                entry_time=timestamp,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                symbol=symbol,
                pin_bar_data=pin_bar_dict,
                lot_size=lot_size
            )
            
            # Try to open trade
            if self.trade_manager.open_trade(trade):
                pin_bar.trade_success = True
                pin_bar.trade_id = trade_id_counter
                pin_bar.rejection_reason = ""  # Clear any previous rejection reason
                trade_id_counter += 1
                trading_hours_stats['successful_entries'] += 1
                
                # Simulate trade execution
                try:
                    self._simulate_trade_with_time_constraints(trade, data['1h'], symbol)
                except Exception as e:
                    pin_bar.rejection_reason = f"Trade simulation failed: {str(e)}"
                    self.trade_manager.close_trade(
                        symbol, timestamp, entry_price, 
                        TradeStatus.NOT_TRIGGERED, f"Simulation failed: {str(e)}"
                    )
                    continue
            else:
                pin_bar.rejection_reason = "Failed to open trade (trade manager error)"
                continue
        
        # Force close any remaining open trades
        if data['1h'] is not None and not data['1h'].empty:
            final_time = ensure_timezone_naive(data['1h'].index[-1])
            forced_closes = self.trade_manager.force_close_time_expired_trades(final_time)
            trading_hours_stats['time_forced_closes'] = len(forced_closes)
        
        debug_info['trading_hours'] = trading_hours_stats
        
        return self.trade_manager.get_all_trades()
    
    def _simulate_trade_with_time_constraints(self, trade: Trade, data_1h: pd.DataFrame, symbol: str):
        """Simulate trade with time constraint fixes"""
        data_1h = normalize_datetime_index(data_1h)
        trade_entry_time = ensure_timezone_naive(trade.entry_time)
        
        try:
            # Find entry index
            time_diffs = [(abs(safe_datetime_subtract(idx, trade_entry_time).total_seconds()), i) 
                         for i, idx in enumerate(data_1h.index)]
            if not time_diffs:
                self.trade_manager.close_trade(
                    symbol, trade_entry_time, trade.entry_price, 
                    TradeStatus.NOT_TRIGGERED, "No price data"
                )
                return
            
            _, entry_idx = min(time_diffs)
        except (IndexError, KeyError):
            self.trade_manager.close_trade(
                symbol, trade_entry_time, trade.entry_price, 
                TradeStatus.NOT_TRIGGERED, "Index error"
            )
            return
        
        if entry_idx + 1 >= len(data_1h):
            self.trade_manager.close_trade(
                symbol, trade_entry_time, trade.entry_price, 
                TradeStatus.NOT_TRIGGERED, "Insufficient future data"
            )
            return
        
        # Check entry trigger
        triggered = False
        trigger_idx = None
        
        for i in range(entry_idx + 1, min(entry_idx + 5, len(data_1h))):
            candle = data_1h.iloc[i]
            candle_time = ensure_timezone_naive(candle.name)
            
            # Check if we should force close due to time
            if should_close_trade_time(candle_time):
                self.trade_manager.close_trade(
                    symbol, candle_time, trade.entry_price, 
                    TradeStatus.NOT_TRIGGERED, "20:00 BST cutoff reached"
                )
                return
            
            if trade.direction == TradeDirection.LONG:
                if candle['High'] >= trade.entry_price:
                    triggered = True
                    trigger_idx = i
                    break
            else:  # SHORT
                if candle['Low'] <= trade.entry_price:
                    triggered = True
                    trigger_idx = i
                    break
        
        if not triggered:
            self.trade_manager.close_trade(
                symbol, trade_entry_time, trade.entry_price, 
                TradeStatus.NOT_TRIGGERED, "Entry level not reached"
            )
            return
        
        # Simulate execution from trigger point
        for i in range(trigger_idx, len(data_1h)):
            candle = data_1h.iloc[i]
            candle_time = ensure_timezone_naive(candle.name)
            
            # Force close at 20:00 BST
            if should_close_trade_time(candle_time):
                current_price = candle['Close']
                self.trade_manager.close_trade(
                    symbol, candle_time, current_price, 
                    TradeStatus.CLOSED_TIME, "Forced close at 20:00 BST"
                )
                return
            
            # Check normal exit conditions
            if trade.direction == TradeDirection.LONG:
                if candle['Low'] <= trade.stop_loss:
                    self.trade_manager.close_trade(
                        symbol, candle_time, trade.stop_loss, TradeStatus.CLOSED_LOSS
                    )
                    return
                elif candle['High'] >= trade.take_profit:
                    self.trade_manager.close_trade(
                        symbol, candle_time, trade.take_profit, TradeStatus.CLOSED_PROFIT
                    )
                    return
            else:  # SHORT
                if candle['High'] >= trade.stop_loss:
                    self.trade_manager.close_trade(
                        symbol, candle_time, trade.stop_loss, TradeStatus.CLOSED_LOSS
                    )
                    return
                elif candle['Low'] <= trade.take_profit:
                    self.trade_manager.close_trade(
                        symbol, candle_time, trade.take_profit, TradeStatus.CLOSED_PROFIT
                    )
                    return
        
        # Trade still open at end of data
        final_time = ensure_timezone_naive(data_1h.index[-1])
        final_price = data_1h.iloc[-1]['Close']
        self.trade_manager.close_trade(
            symbol, final_time, final_price, 
            TradeStatus.CLOSED_TIME, "End of backtest period"
        )
    
    def _check_sma_conditions_enhanced(self, price: float, timestamp: pd.Timestamp,
                                     data: Dict[str, pd.DataFrame], direction: TradeDirection) -> bool:
        """Enhanced SMA conditions with detailed validation"""
        timestamp = ensure_timezone_naive(timestamp)
        available_timeframes = ['15m', '30m', '4h']
        valid_timeframes = 0
        
        for tf in available_timeframes:
            if tf not in data:
                continue
            
            df = normalize_datetime_index(data[tf].copy())
            
            if len(df) >= 50:
                df['SMA50'] = df['Close'].rolling(window=min(50, len(df)//2)).mean()
            else:
                continue
            
            try:
                time_diffs = [(abs(safe_datetime_subtract(idx, timestamp).total_seconds()), i) 
                             for i, idx in enumerate(df.index)]
                if not time_diffs:
                    continue
                
                _, closest_idx = min(time_diffs)
                sma50 = df.iloc[closest_idx]['SMA50']
                
                if pd.isna(sma50):
                    continue
                
                # Stricter trend check with 2% margin
                margin = 0.02
                if direction == TradeDirection.LONG and price > sma50 * (1 + margin):
                    valid_timeframes += 1
                elif direction == TradeDirection.SHORT and price < sma50 * (1 - margin):
                    valid_timeframes += 1
                    
            except (IndexError, KeyError):
                continue
        
        # Require at least 2 timeframes to confirm trend
        return valid_timeframes >= 2
    
    def _calculate_trade_levels_enhanced(self, pin_bar: Dict, direction: TradeDirection,
                                       symbol: str, risk_reward_ratio: float) -> Tuple[float, float, float]:
        """Enhanced trade level calculation"""
        pip_value = self._get_pip_value(symbol)
        
        if direction == TradeDirection.LONG:
            entry_price = pin_bar['close'] + (1 * pip_value)
            stop_loss = pin_bar['low'] - (1 * pip_value)
            
            risk_distance = entry_price - stop_loss
            if risk_distance <= 0:
                raise ValueError("Invalid risk distance for long trade")
                
            take_profit = entry_price + (risk_distance * risk_reward_ratio)
            
        else:  # SHORT
            entry_price = pin_bar['close'] - (1 * pip_value)
            stop_loss = pin_bar['high'] + (1 * pip_value)
            
            risk_distance = stop_loss - entry_price
            if risk_distance <= 0:
                raise ValueError("Invalid risk distance for short trade")
                
            take_profit = entry_price - (risk_distance * risk_reward_ratio)
        
        return entry_price, stop_loss, take_profit
    
    def _calculate_position_size_enhanced(self, account_balance: float, risk_percentage: float,
                                        stop_loss_pips: float, symbol: str) -> float:
        """Enhanced position sizing with risk management"""
        if stop_loss_pips <= 0:
            return 0.0
        
        risk_amount = account_balance * risk_percentage
        pip_value_usd = 10 if 'JPY' in symbol else 1
        
        position_size = risk_amount / (stop_loss_pips * pip_value_usd)
        
        # Apply realistic constraints
        min_size = 0.01
        max_size = min(10, account_balance / 2000)
        
        return max(min_size, min(max_size, position_size))
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        return 0.01 if 'JPY' in symbol else 0.0001
    
    def _calculate_enhanced_statistics(self, trades: List[Trade], symbol: str, 
                                     account_balance: float, debug_info: Dict) -> Dict:
        """Enhanced statistics calculation"""
        if not trades:
            return {'total_trades': 0, 'account_balance': account_balance}
        
        triggered_trades = [t for t in trades if t.status != TradeStatus.NOT_TRIGGERED]
        untriggered_count = len(trades) - len(triggered_trades)
        
        if not triggered_trades:
            return {
                'total_trades': 0, 
                'untriggered_trades': untriggered_count,
                'account_balance': account_balance
            }
        
        # Basic metrics
        total_trades = len(triggered_trades)
        winning_trades = [t for t in triggered_trades if t.status == TradeStatus.CLOSED_PROFIT]
        losing_trades = [t for t in triggered_trades if t.status == TradeStatus.CLOSED_LOSS]
        time_closed_trades = [t for t in triggered_trades if t.status == TradeStatus.CLOSED_TIME]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        time_close_count = len(time_closed_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl_pips = sum(t.pnl_pips for t in triggered_trades if t.pnl_pips is not None)
        total_pnl_usd = sum(t.pnl_usd for t in triggered_trades if t.pnl_usd is not None)
        
        # Enhanced metrics
        avg_win = np.mean([t.pnl_pips for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pips for t in losing_trades]) if losing_trades else 0
        
        total_win_pips = sum(t.pnl_pips for t in winning_trades)
        total_loss_pips = abs(sum(t.pnl_pips for t in losing_trades))
        
        profit_factor = (total_win_pips / total_loss_pips) if total_loss_pips > 0 else float('inf')
        
        trading_hours_stats = debug_info.get('trading_hours', {})
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'time_closed_trades': time_close_count,
            'untriggered_trades': untriggered_count,
            'win_rate': win_rate,
            'total_pnl_pips': total_pnl_pips,
            'total_pnl_usd': total_pnl_usd,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'trigger_rate': (total_trades / len(trades) * 100) if trades else 0,
            'account_balance': account_balance,
            'ending_balance': account_balance + total_pnl_usd,
            'return_percent': (total_pnl_usd / account_balance * 100) if account_balance > 0 else 0,
            'trading_hours_efficiency': trading_hours_stats,
            'time_forced_closes_pct': (time_close_count / total_trades * 100) if total_trades > 0 else 0
        }


# ================================
# CHART BUILDER  
# ================================

class ChartBuilder:
    """Enhanced chart builder"""
    
    def __init__(self):
        self.colors = {
            'background': '#0d1421',
            'text': '#d1d4dc',
            'grid': '#2a2e39',
            'bullish': '#26a69a',
            'bearish': '#ef5350'
        }
    
    def create_tradingview_chart(self, df: pd.DataFrame, pin_bars: List, 
                               symbol: str, timeframe: str, 
                               show_ma: bool = True, highlight_trade=None,
                               show_trading_hours: bool = True) -> go.Figure:
        """Create enhanced TradingView-style chart"""
        df = normalize_datetime_index(df)
        
        # Calculate indicators if not present
        if 'EMA6' not in df.columns:
            df = self.calculate_moving_averages(df)
        
        # Create main chart
        fig = go.Figure()
        
        # Add trading hours background overlay
        if show_trading_hours:
            self._add_trading_hours_overlay(fig, df)
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        ))
        
        # Add moving averages
        if show_ma:
            self._add_moving_averages(fig, df)
        
        # Add pin bar highlights
        self._add_pin_bar_highlights(fig, pin_bars, highlight_trade=highlight_trade)
        
        # Apply styling
        self._apply_enhanced_styling(fig, symbol, timeframe)
        
        return fig
    
    def _add_trading_hours_overlay(self, fig: go.Figure, df: pd.DataFrame):
        """Add trading hours background overlay"""
        for idx, row in df.iterrows():
            utc_time = ensure_timezone_naive(idx)
            
            if is_valid_trading_time(utc_time):
                fig.add_vrect(
                    x0=idx, x1=idx + timedelta(hours=1),
                    fillcolor="rgba(0, 255, 0, 0.05)",
                    layer="below", line_width=0
                )
            elif should_close_trade_time(utc_time):
                fig.add_vrect(
                    x0=idx, x1=idx + timedelta(hours=1),
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    layer="below", line_width=0
                )
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def calculate_moving_averages(_self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs and SMA for the chart with caching"""
        df = df.copy()
        df['EMA6'] = df['Close'].ewm(span=6).mean()
        df['EMA18'] = df['Close'].ewm(span=18).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        return df
    
    def _add_moving_averages(self, fig: go.Figure, df: pd.DataFrame):
        """Add moving average lines to chart"""
        # EMA6 (Light Red)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA6'], name='EMA6',
            line=dict(color='#ff7f7f', width=1.5),
            opacity=0.8, showlegend=True
        ))
        
        # EMA18 (Light Blue)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA18'], name='EMA18',
            line=dict(color='#7fc7ff', width=1.5),
            opacity=0.8, showlegend=True
        ))
        
        # EMA50 (Dark Blue)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA50'], name='EMA50',
            line=dict(color='#1f77b4', width=2),
            opacity=0.9, showlegend=True
        ))
        
        # SMA200 (Dark Red)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA200'], name='SMA200',
            line=dict(color='#d62728', width=2.5),
            opacity=0.9, showlegend=True
        ))
    
    def _add_pin_bar_highlights(self, fig: go.Figure, pin_bars: List, highlight_trade=None):
        """Add enhanced pin bar highlights"""
        for pin_bar in pin_bars:
            if hasattr(pin_bar, 'timestamp'):
                # New PinBarResult format
                timestamp = ensure_timezone_naive(pin_bar.timestamp)
                pin_type = pin_bar.pin_bar_type
                is_bullish_candle = pin_bar.is_bullish_candle
                strength = pin_bar.strength
                low_price = pin_bar.low
                high_price = pin_bar.high
                close_price = pin_bar.close
            else:
                # Old dict format
                timestamp = ensure_timezone_naive(pin_bar['timestamp'])
                pin_type = pin_bar['type']
                is_bullish_candle = pin_bar.get('is_bullish_candle', True)
                strength = pin_bar.get('strength', 0)
                low_price = pin_bar['low']
                high_price = pin_bar['high']
                close_price = pin_bar['close']
            
            # Color code by candle type
            if pin_type == PinBarType.BULLISH:
                color = 'green' if is_bullish_candle else 'orange'
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[low_price * 0.999],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color=color),
                    name=f'Bullish Pin ({color})',
                    showlegend=False,
                    hovertemplate=f'<b>BULLISH PIN BAR</b><br>' +
                                f'Time: {timestamp}<br>' +
                                f'Candle: {"GREEN" if is_bullish_candle else "DOJI"}<br>' +
                                f'Strength: {strength:.1f}%<br>' +
                                f'Close: {close_price:.5f}<extra></extra>'
                ))
            elif pin_type == PinBarType.BEARISH:
                color = 'red' if not is_bullish_candle else 'orange'
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[high_price * 1.001],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color=color),
                    name=f'Bearish Pin ({color})',
                    showlegend=False,
                    hovertemplate=f'<b>BEARISH PIN BAR</b><br>' +
                                f'Time: {timestamp}<br>' +
                                f'Candle: {"RED" if not is_bullish_candle else "DOJI"}<br>' +
                                f'Strength: {strength:.1f}%<br>' +
                                f'Close: {close_price:.5f}<extra></extra>'
                ))
        
        # Add trade highlights if specified
        if highlight_trade:
            self._add_enhanced_trade_markers(fig, highlight_trade)
    
    def _add_enhanced_trade_markers(self, fig: go.Figure, trade: Trade):
        """Add enhanced trade visualization"""
        entry_time = ensure_timezone_naive(trade.entry_time)
        
        # Entry marker
        entry_color = 'gold'
        if trade.status == TradeStatus.CLOSED_TIME:
            entry_color = 'orange'
        
        fig.add_trace(go.Scatter(
            x=[entry_time],
            y=[trade.entry_price],
            mode='markers',
            marker=dict(symbol='star', size=15, color=entry_color, line=dict(width=2, color='white')),
            name='Trade Entry',
            hovertemplate=f'<b>TRADE ENTRY</b><br>' +
                        f'Direction: {trade.direction.value.title()}<br>' +
                        f'Entry: {trade.entry_price:.5f}<br>' +
                        f'BST Time: {convert_utc_to_bst(entry_time).strftime("%H:%M")}<br>' +
                        f'Stop Loss: {trade.stop_loss:.5f}<br>' +
                        f'Take Profit: {trade.take_profit:.5f}<extra></extra>'
        ))
        
        # Exit marker
        if trade.exit_time and trade.exit_price:
            exit_time = ensure_timezone_naive(trade.exit_time)
            
            if trade.status == TradeStatus.CLOSED_TIME:
                exit_color = 'orange'
                outcome = 'TIME CLOSE'
            elif trade.pnl_pips > 0:
                exit_color = 'green'
                outcome = 'PROFIT'
            elif trade.pnl_pips < 0:
                exit_color = 'red'
                outcome = 'LOSS'
            else:
                exit_color = 'gray'
                outcome = 'BREAKEVEN'
            
            fig.add_trace(go.Scatter(
                x=[exit_time],
                y=[trade.exit_price],
                mode='markers',
                marker=dict(symbol='circle', size=12, color=exit_color, line=dict(width=2, color='white')),
                name='Trade Exit',
                hovertemplate=f'<b>TRADE EXIT - {outcome}</b><br>' +
                            f'Exit: {trade.exit_price:.5f}<br>' +
                            f'BST Time: {convert_utc_to_bst(exit_time).strftime("%H:%M")}<br>' +
                            f'P&L: {trade.pnl_pips:.1f} pips<br>' +
                            f'P&L USD: ${trade.pnl_usd:.2f}<br>' +
                            f'Reason: {trade.forced_close_reason or "Normal exit"}<extra></extra>'
            ))
        
        # Price level lines
        fig.add_hline(y=trade.entry_price, line_dash="solid", line_color="blue", line_width=2,
                     annotation_text=f"Entry: {trade.entry_price:.5f}")
        fig.add_hline(y=trade.stop_loss, line_dash="dash", line_color="red", line_width=2,
                     annotation_text=f"Stop: {trade.stop_loss:.5f}")
        fig.add_hline(y=trade.take_profit, line_dash="dash", line_color="green", line_width=2,
                     annotation_text=f"Target: {trade.take_profit:.5f}")
    
    def _apply_enhanced_styling(self, fig: go.Figure, symbol: str, timeframe: str):
        """Apply enhanced styling"""
        is_jpy_pair = 'JPY' in symbol
        y_tick_format = '.2f' if is_jpy_pair else '.5f'
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font_color=self.colors['text'],
            title=f"{symbol} - {timeframe} Chart",
            title_font_size=18,
            xaxis=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                rangeslider=dict(visible=False),
                type='date',
                rangebreaks=[
                    dict(bounds=["sat", "mon"], pattern="day of week"),
                ]
            ),
            yaxis=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                side='right',
                tickformat=y_tick_format
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=60, t=80, b=20),
            height=600
        )


# ================================
# COMPATIBILITY FIXES
# ================================

def safe_get_pin_bars(results):
    """Safely get pin bars from results with backward compatibility and debugging"""
    print(f"🔍 safe_get_pin_bars called with results type: {type(results)}")
    
    if hasattr(results, 'pin_bars'):
        pin_bars = results.pin_bars
        print(f"🔍 Found pin_bars attribute: {type(pin_bars)}, length: {len(pin_bars) if pin_bars else 'None'}")
        
        if pin_bars:
            print(f"🔍 First pin bar type: {type(pin_bars[0]) if pin_bars else 'No pin bars'}")
            return pin_bars
        else:
            print("🔍 pin_bars attribute exists but is empty or None")
            return []
    else:
        print("🔍 No pin_bars attribute found in results")
        return []

def ensure_backtest_results_compatibility(results):
    """Ensure BacktestResults object has all required attributes"""
    if not hasattr(results, 'pin_bars'):
        results.pin_bars = []
    if not hasattr(results, 'trading_hours_stats'):
        results.trading_hours_stats = {}
    return results


# ================================
# STREAMLIT UI WITH PIN BAR ANALYSIS
# ================================

class TrendSurferUI:
    """Enhanced Streamlit UI with comprehensive pin bar analysis"""
    
    def __init__(self):
        self.backtester = TrendSurferBacktester()
        self.chart_builder = ChartBuilder()
        self.detector = PinBarDetector()
        
        # Initialize session state
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        if 'show_backtest_chart' not in st.session_state:
            st.session_state.show_backtest_chart = False
        if 'selected_trade_index' not in st.session_state:
            st.session_state.selected_trade_index = 0
    
    def render_sidebar(self):
        """Configuration sidebar"""
        st.sidebar.title("🏄‍♂️ Trend Surfer")
        
        # Symbol selection
        forex_pairs = [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
            "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X",
            "AUDCAD=X", "AUDCHF=X", "AUDNZD=X", "CADJPY=X", "CHFJPY=X"
        ]
        
        selected_symbol = st.sidebar.selectbox(
            "🎯 Select Trading Pair",
            forex_pairs,
            index=1,
            help="Choose the currency pair for analysis"
        )
        
        # Backtest parameters
        st.sidebar.subheader("⚙️ Backtest Parameters")
        st.sidebar.caption("📈 Now supports up to 5 years of historical data")
        st.sidebar.caption("⚡ Intelligent caching for faster performance")
        
        # Date range
        current_date = datetime.now().date()
        end_date = st.sidebar.date_input(
            "📅 End Date",
            value=current_date - timedelta(days=1),
            max_value=current_date - timedelta(days=1)
        )
        
        # Duration selection - now supports up to 5 years
        duration_options = {
            "1 Week": 7,
            "2 Weeks": 14,
            "1 Month": 30,
            "2 Months": 60,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825,
            "Custom": None
        }
        
        selected_duration = st.sidebar.selectbox(
            "⏱️ Backtest Duration",
            list(duration_options.keys()),
            index=2
        )
        
        # Calculate start date
        if selected_duration == "Custom":
            start_date = st.sidebar.date_input(
                "📅 Custom Start Date",
                value=end_date - timedelta(days=30),
                max_value=end_date - timedelta(days=1)
            )
        else:
            days_back = duration_options[selected_duration]
            start_date = end_date - timedelta(days=days_back)
            st.sidebar.text_input(
                "📅 Start Date (Auto)",
                value=start_date.strftime("%Y-%m-%d"),
                disabled=True
            )
        
        # Risk management
        st.sidebar.subheader("💰 Risk Management")
        
        account_size = st.sidebar.selectbox(
            "💵 Account Size",
            [1000, 2500, 5000, 10000, 25000, 50000, 100000],
            index=3,
            format_func=lambda x: f"${x:,}"
        )
        
        risk_percentage = st.sidebar.selectbox(
            "⚠️ Risk Per Trade",
            [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
            index=1,
            format_func=lambda x: f"{x*100:.1f}%"
        )
        
        risk_reward = st.sidebar.selectbox(
            "🎯 Risk:Reward Ratio",
            [1.5, 2.0, 2.5, 3.0],
            index=1,
            format_func=lambda x: f"1:{x}"
        )
        
        # Detector settings
        st.sidebar.subheader("🔍 Detection Settings")
        
        min_wick = st.sidebar.slider(
            "Min Wick Ratio",
            min_value=0.5,
            max_value=0.8,
            value=0.55,
            step=0.05
        )
        
        max_body = st.sidebar.slider(
            "Max Body Ratio", 
            min_value=0.2,
            max_value=0.5,
            value=0.4,
            step=0.05
        )
        
        max_opposite_wick = st.sidebar.slider(
            "Max Opposite Wick",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05
        )
        
        return {
            'symbol': selected_symbol,
            'start_date': start_date,
            'end_date': end_date,
            'duration': selected_duration,
            'account_size': account_size,
            'risk_percentage': risk_percentage,
            'risk_reward': risk_reward,
            'min_wick_ratio': min_wick,
            'max_body_ratio': max_body,
            'max_opposite_wick': max_opposite_wick
        }
    
    def render_backtest_tab(self, config: Dict):
        """Backtesting interface with enhanced pin bar analysis"""
        st.header("🔬 Trend Surfer Backtesting")
        
        # Configuration display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Trading Setup**
            - Symbol: {config['symbol']}
            - Duration: {config['duration']}
            - Period: {config['start_date']} to {config['end_date']}
            """)
        
        with col2:
            st.info(f"""
            **Risk Management**
            - Account: ${config['account_size']:,}
            - Risk/Trade: {config['risk_percentage']*100:.1f}%
            - R:R Ratio: 1:{config['risk_reward']}
            """)
        
        with col3:
            st.info(f"""
            **Trading Constraints**
            - Scan: 3:00-17:00 BST
            - Close: 20:00 BST
            - Max: 1 trade per pair
            - Valid candle colors only
            """)
        
        # Run backtest button with cache info
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
                # Performance warning for large datasets
                if config['duration'] in ["2 Years", "5 Years"] or (config['duration'] == "Custom" and (config['end_date'] - config['start_date']).days > 365):
                    st.info("⚡ Large dataset detected. Using intelligent caching to optimize performance.")
                
                # Update detector settings
                self.backtester.detector = PinBarDetector(
                    min_wick_ratio=config['min_wick_ratio'],
                    max_body_ratio=config['max_body_ratio'],
                    max_opposite_wick=config['max_opposite_wick']
                )
                
                # Performance tracking
                import time
                start_time = time.time()
                
                with st.spinner(f"🔄 Running {config['duration']} backtest..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📊 Fetching market data (cached if available)...")
                    progress_bar.progress(20)
                    
                    # Convert dates to datetime
                    start_datetime = datetime.combine(config['start_date'], datetime.min.time())
                    end_datetime = datetime.combine(config['end_date'], datetime.min.time())
                    
                    status_text.text("🔍 Detecting pin bars with rejection tracking...")
                    progress_bar.progress(40)
                    
                    results = self.backtester.run_backtest(
                        symbol=config['symbol'],
                        start_date=start_datetime,
                        end_date=end_datetime,
                        risk_reward_ratio=config['risk_reward'],
                        account_balance=config['account_size'],
                        risk_percentage=config['risk_percentage']
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("📈 Analyzing results and rejection reasons...")
                    
                    # Ensure compatibility
                    results = ensure_backtest_results_compatibility(results)
                    
                    progress_bar.progress(100)
                    
                    # Calculate performance time
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    status_text.text(f"✅ Backtest completed in {execution_time:.1f} seconds!")
                    
                    if results.trades or results.statistics:
                        st.session_state.backtest_results = results
                        st.success(f"🎉 {config['duration']} backtest completed in {execution_time:.1f}s!")
                        
                        # Enhanced summary with pin bar stats
                        pin_bars = safe_get_pin_bars(results)
                        total_pin_bars = len([pb for pb in pin_bars if pb.pin_bar_type != PinBarType.NONE]) if pin_bars else 0
                        trades_taken = results.statistics.get('total_trades', 0)
                        
                        if trades_taken > 0:
                            win_rate = results.statistics.get('win_rate', 0)
                            total_pips = results.statistics.get('total_pnl_pips', 0)
                            conversion_rate = (trades_taken / total_pin_bars * 100) if total_pin_bars > 0 else 0
                            st.info(f"📊 **Quick Summary:** {total_pin_bars} pin bars detected → {trades_taken} trades taken ({conversion_rate:.1f}% conversion) → {win_rate:.1f}% win rate → {total_pips:.1f} pips")
                        else:
                            st.warning(f"⚠️ {total_pin_bars} pin bars detected but no trades taken. Check rejection reasons in analysis table below.")
                    else:
                        st.error("❌ No valid data found. Try different parameters or date range.")
        
        with col_btn2:
            if st.button("🧹 Clear Results", use_container_width=True):
                st.session_state.backtest_results = None
                st.session_state.selected_trade_index = 0
                st.success("Results cleared!")
        
        with col_btn3:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.info("Next backtest will fetch fresh data.")
        
        # Display results with enhanced pin bar analysis
        if st.session_state.backtest_results:
            try:
                results = ensure_backtest_results_compatibility(st.session_state.backtest_results)
                
                # Quick stats at the top
                pin_bars = safe_get_pin_bars(results)
                if pin_bars:
                    total_pins = len([pb for pb in pin_bars if pb.pin_bar_type != PinBarType.NONE])
                    successful_trades = len([pb for pb in pin_bars if getattr(pb, 'trade_success', False)])
                    outside_hours = len([pb for pb in pin_bars if not getattr(pb, 'in_trading_hours', True)])
                    wrong_color = len([pb for pb in pin_bars if getattr(pb, 'rejection_reason', '').startswith('Wrong candle color')])
                    
                    st.info(f"""
                    **🎯 Pin Bar Conversion Summary:**
                    • {total_pins} pin bars detected
                    • {successful_trades} trades taken ({(successful_trades/total_pins*100):.1f}% conversion rate)
                    • {outside_hours} rejected for wrong timing
                    • {wrong_color} rejected for wrong candle color
                    """)
                
                self.display_enhanced_results(results)
                
            except Exception as e:
                st.error(f"Error displaying results: {str(e)}")
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    def display_enhanced_results(self, results: BacktestResults):
        """Display enhanced backtest results with comprehensive pin bar analysis"""
        stats = results.statistics
        
        if not stats:
            st.warning("No statistics available")
            return
        
        # Performance dashboard
        st.subheader("📊 Performance Dashboard")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_trades = stats.get('total_trades', 0)
            st.metric("Total Trades", total_trades)
            
            trigger_rate = stats.get('trigger_rate', 0)
            st.metric("Trigger Rate", f"{trigger_rate:.1f}%")
        
        with col2:
            win_rate = stats.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            
            profit_factor = stats.get('profit_factor', 0)
            pf_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
            st.metric("Profit Factor", pf_display)
        
        with col3:
            total_pips = stats.get('total_pnl_pips', 0)
            st.metric("Total Pips", f"{total_pips:.1f}", delta=f"{total_pips:.1f}" if total_pips != 0 else None)
            
            avg_win = stats.get('average_win', 0)
            avg_loss = stats.get('average_loss', 0)
            if avg_loss != 0:
                win_loss_ratio = abs(avg_win / avg_loss)
                st.metric("Avg Win/Loss", f"{win_loss_ratio:.2f}")
            else:
                st.metric("Avg Win/Loss", "∞")
        
        with col4:
            total_usd = stats.get('total_pnl_usd', 0)
            st.metric("P&L (USD)", f"${total_usd:.2f}", delta=f"${total_usd:.2f}" if total_usd != 0 else None)
            
            return_pct = stats.get('return_percent', 0)
            st.metric("Return %", f"{return_pct:.2f}%", delta=f"{return_pct:.2f}%" if return_pct != 0 else None)
        
        with col5:
            time_closes = stats.get('time_closed_trades', 0)
            time_close_pct = stats.get('time_forced_closes_pct', 0)
            st.metric("Time Closes", time_closes)
            st.metric("Time Close %", f"{time_close_pct:.1f}%")
        
        # PIN BAR ANALYSIS TABLE - NEW SECTION
        st.subheader("🎯 Pin Bar Analysis")
        
        # Get pin bars from results
        pin_bars = safe_get_pin_bars(results)
        
        if pin_bars:
            st.info(f"📍 **Total Pin Bars Detected:** {len(pin_bars)} during backtest period")
            
            # Create pin bar analysis data
            pin_bar_data = []
            
            for i, pin_bar in enumerate(pin_bars):
                # Handle both PinBarResult objects and dict formats
                if hasattr(pin_bar, 'timestamp'):
                    # PinBarResult object
                    timestamp = ensure_timezone_naive(pin_bar.timestamp)
                    pin_type = pin_bar.pin_bar_type
                    strength = pin_bar.strength
                    is_bullish_candle = pin_bar.is_bullish_candle
                    in_trading_hours = pin_bar.in_trading_hours
                    trade_attempted = getattr(pin_bar, 'trade_attempted', False)
                    trade_success = getattr(pin_bar, 'trade_success', False)
                    rejection_reason = getattr(pin_bar, 'rejection_reason', "")
                    bst_time = pin_bar.bst_time
                    close_price = pin_bar.close
                else:
                    # Dict format (fallback)
                    timestamp = ensure_timezone_naive(pin_bar['timestamp'])
                    pin_type = pin_bar['type']
                    strength = pin_bar.get('strength', 0)
                    is_bullish_candle = pin_bar.get('is_bullish_candle', True)
                    in_trading_hours = is_valid_trading_time(timestamp)
                    trade_attempted = pin_bar.get('trade_attempted', False)
                    trade_success = pin_bar.get('trade_success', False)
                    rejection_reason = pin_bar.get('rejection_reason', "")
                    bst_time = convert_utc_to_bst(timestamp).strftime('%m/%d %H:%M BST')
                    close_price = pin_bar['close']
                
                # Skip non-pin bars
                if pin_type == PinBarType.NONE:
                    continue
                
                # Determine pin bar type emoji and direction
                if pin_type == PinBarType.BULLISH:
                    type_emoji = "📈"
                    type_text = "Bullish"
                elif pin_type == PinBarType.BEARISH:
                    type_emoji = "📉"
                    type_text = "Bearish"
                else:
                    type_emoji = "⚪"
                    type_text = "None"
                
                # Determine candle color and validation
                if is_bullish_candle:
                    candle_color = "🟢 Green"
                    color_valid = pin_type == PinBarType.BULLISH
                else:
                    candle_color = "🔴 Red"
                    color_valid = pin_type == PinBarType.BEARISH
                
                # Check if candle color matches pin bar type
                color_validation = "✅ Valid" if color_valid else "❌ Invalid Color"
                
                # Trading hours status
                trading_hours_status = "✅ Valid" if in_trading_hours else "❌ Outside Hours"
                
                # Trade outcome
                if trade_success:
                    outcome = "✅ Trade Taken"
                    reason = "Successfully entered trade"
                elif trade_attempted:
                    outcome = "❌ Trade Rejected"
                    # Use the stored rejection reason
                    reason = rejection_reason if rejection_reason else "Failed validation checks"
                else:
                    outcome = "⏭️ Not Attempted"
                    if not color_valid:
                        reason = "Wrong candle color for pin bar type"
                    elif not in_trading_hours:
                        reason = "Outside trading hours (3:00-17:00 BST)"
                    else:
                        reason = "Failed initial validation"
                
                pin_bar_data.append({
                    '#': i + 1,
                    'BST Time': bst_time,
                    'Type': f"{type_emoji} {type_text}",
                    'Strength': f"{strength:.1f}%",
                    'Candle': candle_color,
                    'Color Valid': color_validation,
                    'Trading Hours': trading_hours_status,
                    'Price': f"{close_price:.5f}",
                    'Outcome': outcome,
                    'Rejection Reason': reason
                })
            
            if pin_bar_data:
                # Create DataFrame
                pin_df = pd.DataFrame(pin_bar_data)
                
                # Display summary stats first
                total_pin_bars = len(pin_df)
                valid_color = len([d for d in pin_bar_data if "✅ Valid" in d['Color Valid']])
                valid_hours = len([d for d in pin_bar_data if "✅ Valid" in d['Trading Hours']])
                trades_taken = len([d for d in pin_bar_data if "✅ Trade Taken" in d['Outcome']])
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Total Pin Bars", total_pin_bars)
                
                with col_stat2:
                    valid_color_pct = (valid_color / total_pin_bars * 100) if total_pin_bars > 0 else 0
                    st.metric("Valid Color", valid_color, delta=f"{valid_color_pct:.1f}%")
                
                with col_stat3:
                    valid_hours_pct = (valid_hours / total_pin_bars * 100) if total_pin_bars > 0 else 0
                    st.metric("Valid Hours", valid_hours, delta=f"{valid_hours_pct:.1f}%")
                
                with col_stat4:
                    trades_pct = (trades_taken / total_pin_bars * 100) if total_pin_bars > 0 else 0
                    st.metric("Trades Taken", trades_taken, delta=f"{trades_pct:.1f}%")
                
                # Filter options
                st.subheader("🔍 Filter Pin Bars")
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                
                with col_filter1:
                    filter_type = st.selectbox(
                        "Pin Bar Type",
                        ["All", "Bullish Only", "Bearish Only"],
                        index=0
                    )
                
                with col_filter2:
                    filter_outcome = st.selectbox(
                        "Trade Outcome",
                        ["All", "Trades Taken", "Trades Rejected", "Not Attempted"],
                        index=0
                    )
                
                with col_filter3:
                    filter_hours = st.selectbox(
                        "Trading Hours",
                        ["All", "Valid Hours Only", "Outside Hours Only"],
                        index=0
                    )
                
                # Apply filters
                filtered_df = pin_df.copy()
                
                if filter_type == "Bullish Only":
                    filtered_df = filtered_df[filtered_df['Type'].str.contains('📈')]
                elif filter_type == "Bearish Only":
                    filtered_df = filtered_df[filtered_df['Type'].str.contains('📉')]
                
                if filter_outcome == "Trades Taken":
                    filtered_df = filtered_df[filtered_df['Outcome'].str.contains('✅ Trade Taken')]
                elif filter_outcome == "Trades Rejected":
                    filtered_df = filtered_df[filtered_df['Outcome'].str.contains('❌ Trade Rejected')]
                elif filter_outcome == "Not Attempted":
                    filtered_df = filtered_df[filtered_df['Outcome'].str.contains('⏭️ Not Attempted')]
                
                if filter_hours == "Valid Hours Only":
                    filtered_df = filtered_df[filtered_df['Trading Hours'].str.contains('✅')]
                elif filter_hours == "Outside Hours Only":
                    filtered_df = filtered_df[filtered_df['Trading Hours'].str.contains('❌')]
                
                # Display filtered results
                st.write(f"**Showing {len(filtered_df)} of {len(pin_df)} pin bars**")
                
                # Display the table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(600, len(filtered_df) * 40 + 40),
                    column_config={
                        "Strength": st.column_config.ProgressColumn(
                            "Strength",
                            help="Pin bar strength percentage",
                            min_value=0,
                            max_value=100,
                            format="%.1f%%"
                        ),
                        "Rejection Reason": st.column_config.TextColumn(
                            "Rejection Reason",
                            help="Detailed explanation of why trade was not taken",
                            width="large"
                        )
                    }
                )
                
                # Export option
                if st.button("📥 Export Pin Bar Analysis to CSV"):
                    csv = pin_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"pin_bar_analysis_{results.symbol}_{results.start_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                # Common rejection reasons summary
                st.subheader("📋 Rejection Reasons Summary")
                
                rejection_counts = {}
                for data in pin_bar_data:
                    if "❌" in data['Outcome'] or "⏭️" in data['Outcome']:
                        reason = data['Rejection Reason']
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                
                if rejection_counts:
                    reason_data = []
                    total_rejections = sum(rejection_counts.values())
                    
                    for reason, count in sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total_rejections * 100) if total_rejections > 0 else 0
                        reason_data.append({
                            'Rejection Reason': reason,
                            'Count': count,
                            'Percentage': f"{percentage:.1f}%"
                        })
                    
                    reason_df = pd.DataFrame(reason_data)
                    st.dataframe(reason_df, use_container_width=True, hide_index=True)
                else:
                    st.info("✅ All pin bars resulted in successful trades!")
            
            else:
                st.info("ℹ️ No valid pin bars detected during the backtest period")
        
        else:
            st.warning("⚠️ No pin bar data available from backtest results")
        
        # Trade breakdown (existing code continues...)
        if results.trades:
            st.subheader("📋 Trade Analysis")
            
            # Trade filtering
            triggered_trades = [t for t in results.trades if t.status != TradeStatus.NOT_TRIGGERED]
            
            if triggered_trades:
                # Create trade data
                trade_data = []
                for i, trade in enumerate(triggered_trades):
                    # Color coding by exit reason
                    if trade.status == TradeStatus.CLOSED_PROFIT:
                        outcome_emoji = "🟢"
                    elif trade.status == TradeStatus.CLOSED_LOSS:
                        outcome_emoji = "🔴"
                    elif trade.status == TradeStatus.CLOSED_TIME:
                        outcome_emoji = "⏰"
                    else:
                        outcome_emoji = "⚪"
                    
                    direction_emoji = "📈" if trade.direction == TradeDirection.LONG else "📉"
                    
                    # Calculate duration
                    duration_str = "Open"
                    if trade.exit_time:
                        entry_time = ensure_timezone_naive(trade.entry_time)
                        exit_time = ensure_timezone_naive(trade.exit_time)
                        duration = safe_datetime_subtract(exit_time, entry_time)
                        hours = duration.total_seconds() / 3600
                        if hours < 24:
                            duration_str = f"{hours:.1f}h"
                        else:
                            duration_str = f"{hours/24:.1f}d"
                    
                    # BST times
                    entry_bst = convert_utc_to_bst(trade.entry_time)
                    exit_bst = convert_utc_to_bst(trade.exit_time) if trade.exit_time else None
                    
                    trade_data.append({
                        '#': i + 1,
                        'Entry (BST)': entry_bst.strftime('%m/%d %H:%M'),
                        'Exit (BST)': exit_bst.strftime('%m/%d %H:%M') if exit_bst else "Open",
                        'Duration': duration_str,
                        'Dir': f"{direction_emoji} {trade.direction.value.title()}",
                        'Entry': f"{trade.entry_price:.5f}",
                        'Exit': f"{trade.exit_price:.5f}" if trade.exit_price else "Open",
                        'Pips': f"{outcome_emoji} {trade.pnl_pips:.1f}",
                        'USD': f"${trade.pnl_usd:.2f}",
                        'Status': trade.status.value.replace('_', ' ').title()
                    })
                
                if trade_data:
                    trade_df = pd.DataFrame(trade_data)
                    
                    st.dataframe(
                        trade_df, 
                        use_container_width=True, 
                        hide_index=True,
                        height=min(400, len(trade_data) * 40 + 40)
                    )
                    
                    # Trade chart selection
                    st.subheader("🔍 Individual Trade Analysis")
                    
                    selected_trade_num = st.number_input(
                        f"Select trade number (1-{len(triggered_trades)}):",
                        min_value=1,
                        max_value=len(triggered_trades),
                        value=min(st.session_state.selected_trade_index + 1, len(triggered_trades)),
                        step=1
                    )
                    
                    selected_trade_idx = selected_trade_num - 1
                    st.session_state.selected_trade_index = selected_trade_idx
                    selected_trade = triggered_trades[selected_trade_idx]
                    
                    # Chart buttons
                    col_btn1, col_btn2 = st.columns([1, 1])
                    with col_btn1:
                        if st.button(f"📊 View Trade #{selected_trade_num} Chart", 
                                   type="primary", 
                                   use_container_width=True):
                            self._display_enhanced_trade_chart(results, selected_trade, selected_trade_num)
                    
                    with col_btn2:
                        if st.button("📈 View All Trades Chart", 
                                   type="secondary", 
                                   use_container_width=True):
                            self.display_all_trades_chart(results, triggered_trades)
    
    def _display_enhanced_trade_chart(self, results: BacktestResults, trade: Trade, trade_number: int):
        """Display enhanced chart for individual trade"""
        st.subheader(f"📊 Trade #{trade_number} - Chart Analysis")
        
        if results.data_1h.empty:
            st.error("No chart data available")
            return
        
        # Get pin bars from results
        trade_time = ensure_timezone_naive(trade.entry_time)
        time_window = timedelta(hours=48)
        
        relevant_pin_bars = []
        pin_bars = safe_get_pin_bars(results)
        if pin_bars:
            for pb in pin_bars:
                pb_time = ensure_timezone_naive(pb.timestamp)
                if abs(safe_datetime_subtract(pb_time, trade_time).total_seconds()) <= time_window.total_seconds():
                    relevant_pin_bars.append(pb)
        else:
            if trade.pin_bar_data:
                relevant_pin_bars = [trade.pin_bar_data]
        
        try:
            fig = self.chart_builder.create_tradingview_chart(
                results.data_1h,
                relevant_pin_bars,
                results.symbol,
                "1H",
                show_ma=True,
                highlight_trade=trade,
                show_trading_hours=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade summary
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            with col_summary1:
                if trade.status == TradeStatus.CLOSED_TIME:
                    outcome = "⏰ TIME CLOSE"
                elif trade.pnl_pips > 0:
                    outcome = "🟢 PROFIT"
                elif trade.pnl_pips < 0:
                    outcome = "🔴 LOSS"
                else:
                    outcome = "⚪ BREAKEVEN"
                
                st.markdown(f"""
                **Trade Outcome**
                {outcome}
                P&L: {trade.pnl_pips:.1f} pips
                USD: ${trade.pnl_usd:.2f}
                """)
            
            with col_summary2:
                direction_icon = "📈" if trade.direction == TradeDirection.LONG else "📉"
                entry_bst = convert_utc_to_bst(trade.entry_time)
                
                st.markdown(f"""
                **Trade Setup**
                {direction_icon} {trade.direction.value.title()}
                Entry: {trade.entry_price:.5f}
                BST: {entry_bst.strftime('%H:%M')}
                """)
            
            with col_summary3:
                if trade.exit_time:
                    entry_time = ensure_timezone_naive(trade.entry_time)
                    exit_time = ensure_timezone_naive(trade.exit_time)
                    duration = safe_datetime_subtract(exit_time, entry_time)
                    hours = duration.total_seconds() / 3600
                    duration_display = f"{hours:.1f}h" if hours < 24 else f"{hours/24:.1f}d"
                    
                    exit_bst = convert_utc_to_bst(trade.exit_time)
                    exit_display = exit_bst.strftime('%m/%d %H:%M')
                else:
                    duration_display = "Still Open"
                    exit_display = "Open"
                
                st.markdown(f"""
                **Trade Timing**
                Entry: {entry_bst.strftime('%m/%d %H:%M')}
                Exit: {exit_display}
                Duration: {duration_display}
                """)
        
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
    
    def display_all_trades_chart(self, results: BacktestResults, trades: List[Trade]):
        """Display chart with all trades"""
        st.subheader("📈 All Trades Overview")
        
        if results.data_1h.empty:
            st.error("No chart data available")
            return
        
        # Use pin bars from results
        pin_bars_to_show = []
        pin_bars = safe_get_pin_bars(results)
        if pin_bars:
            pin_bars_to_show = [pb for pb in pin_bars if pb.trade_success]
        else:
            for trade in trades:
                if trade.pin_bar_data:
                    pin_bars_to_show.append(trade.pin_bar_data)
        
        try:
            # Create base chart
            fig = self.chart_builder.create_tradingview_chart(
                results.data_1h,
                pin_bars_to_show,
                results.symbol,
                "1H",
                show_ma=True,
                show_trading_hours=True
            )
            
            # Add all trade markers
            for i, trade in enumerate(trades):
                entry_time = ensure_timezone_naive(trade.entry_time)
                
                # Color coding
                if trade.status == TradeStatus.CLOSED_PROFIT:
                    marker_color = 'green'
                    marker_symbol = 'triangle-up' if trade.direction == TradeDirection.LONG else 'triangle-down'
                elif trade.status == TradeStatus.CLOSED_LOSS:
                    marker_color = 'red'
                    marker_symbol = 'triangle-up' if trade.direction == TradeDirection.LONG else 'triangle-down'
                elif trade.status == TradeStatus.CLOSED_TIME:
                    marker_color = 'orange'
                    marker_symbol = 'square'
                else:
                    marker_color = 'gray'
                    marker_symbol = 'circle'
                
                # Entry markers
                fig.add_trace(go.Scatter(
                    x=[entry_time],
                    y=[trade.entry_price],
                    mode='markers',
                    marker=dict(symbol=marker_symbol, size=10, color=marker_color),
                    name=f'Trade {i+1}',
                    hovertemplate=f'<b>Trade {i+1}</b><br>' +
                                f'Entry: {trade.entry_price:.5f}<br>' +
                                f'BST: {convert_utc_to_bst(entry_time).strftime("%H:%M")}<br>' +
                                f'P&L: {trade.pnl_pips:.1f} pips<br>' +
                                f'Status: {trade.status.value}<extra></extra>',
                    showlegend=False
                ))
                
                # Exit markers
                if trade.exit_time and trade.exit_price:
                    exit_time = ensure_timezone_naive(trade.exit_time)
                    
                    fig.add_trace(go.Scatter(
                        x=[exit_time],
                        y=[trade.exit_price],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color=marker_color, 
                                  line=dict(width=2, color='white')),
                        name=f'Exit {i+1}',
                        hovertemplate=f'<b>Trade {i+1} Exit</b><br>' +
                                    f'Exit: {trade.exit_price:.5f}<br>' +
                                    f'BST: {convert_utc_to_bst(exit_time).strftime("%H:%M")}<br>' +
                                    f'P&L: {trade.pnl_pips:.1f} pips<extra></extra>',
                        showlegend=False
                    ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            profit_trades = [t for t in trades if t.pnl_pips > 0]
            loss_trades = [t for t in trades if t.pnl_pips < 0]
            time_trades = [t for t in trades if t.status == TradeStatus.CLOSED_TIME]
            
            col_overview1, col_overview2, col_overview3, col_overview4 = st.columns(4)
            
            with col_overview1:
                st.metric("Total Trades", len(trades))
                st.metric("Profitable", len(profit_trades), delta=f"{len(profit_trades)/len(trades)*100:.1f}%")
            
            with col_overview2:
                total_pips = sum(t.pnl_pips for t in trades)
                st.metric("Total Pips", f"{total_pips:.1f}")
                avg_pips = total_pips / len(trades) if trades else 0
                st.metric("Avg per Trade", f"{avg_pips:.1f} pips")
            
            with col_overview3:
                total_usd = sum(t.pnl_usd for t in trades)
                st.metric("Total P&L", f"${total_usd:.2f}")
                win_rate = len(profit_trades) / len(trades) * 100 if trades else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col_overview4:
                st.metric("Time-Closed", len(time_trades))
                time_close_pct = len(time_trades) / len(trades) * 100 if trades else 0
                st.metric("Time Close %", f"{time_close_pct:.1f}%")
        
        except Exception as e:
            st.error(f"Error creating overview chart: {str(e)}")
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        return 0.01 if 'JPY' in symbol else 0.0001


# ================================
# LIVE ANALYSIS
# ================================

def render_live_analysis_tab(config: Dict):
    """Live market analysis"""
    st.header("📊 Live Market Analysis")
    
    col_live1, col_live2 = st.columns([3, 1])
    
    with col_live2:
        st.subheader("⚙️ Analysis Settings")
        
        timeframe = st.selectbox("Timeframe", ["1h", "4h"], index=0)
        lookback_days = st.selectbox("Lookback Period", [7, 14, 30, 60, 90, 180, 365], index=4)  # Default to 90 days
        show_pin_bars = st.checkbox("Show Pin Bars", value=True)
        show_emas = st.checkbox("Show EMAs", value=True)
        show_trading_hours = st.checkbox("Show Trading Hours", value=True)
        
        # Current BST time display
        current_utc = datetime.now()
        current_bst = convert_utc_to_bst(current_utc)
        st.info(f"""
        **Current BST Time:**
        {current_bst.strftime('%H:%M:%S %Z')}
        
        **Trading Status:**
        {'🟢 SCAN HOURS' if is_valid_trading_time(current_utc) else '🔴 OUTSIDE HOURS'}
        """)
        
        if st.button("🔄 Refresh Data", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared! Fresh data will be fetched.")
            st.rerun()
    
    with col_live1:
        # Enhanced data fetching with time awareness
        end_date = ensure_timezone_naive(datetime.now())
        start_date = end_date - timedelta(days=lookback_days)
        
        with st.spinner(f"📡 Fetching {timeframe} data for {config['symbol']} (cached if available)..."):
            try:
                fetcher = DataFetcher()
                data = fetcher.fetch_data(
                    config['symbol'], 
                    timeframe,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            except Exception as e:
                st.error(f"Data fetch error: {e}")
                data = pd.DataFrame()
        
        if not data.empty:
            # Calculate indicators
            chart_builder = ChartBuilder()
            data_with_indicators = chart_builder.calculate_moving_averages(data)
            
            # Detect pin bars
            pin_bars = []
            if show_pin_bars:
                pin_bars = detect_recent_pin_bars(data_with_indicators, config)
            
            # Create enhanced chart
            try:
                fig = chart_builder.create_tradingview_chart(
                    data_with_indicators,
                    pin_bars,
                    config['symbol'],
                    timeframe.upper(),
                    show_ma=show_emas,
                    show_trading_hours=show_trading_hours
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart creation error: {e}")
            
            # Pin bar analysis
            if pin_bars:
                st.subheader(f"🎯 Recent Pin Bars ({len(pin_bars)} found)")
                
                # Filter pin bars by trading hours
                valid_time_pins = [pb for pb in pin_bars if is_valid_trading_time(pb['timestamp'])]
                invalid_time_pins = [pb for pb in pin_bars if not is_valid_trading_time(pb['timestamp'])]
                
                col_pins1, col_pins2 = st.columns(2)
                
                with col_pins1:
                    st.success(f"""
                    **✅ Valid Time Pin Bars**
                    Count: {len(valid_time_pins)}
                    (During 3:00-17:00 BST)
                    """)
                
                with col_pins2:
                    st.warning(f"""
                    **⏰ Outside Hours Pin Bars**
                    Count: {len(invalid_time_pins)}
                    (Outside trading hours)
                    """)
                
                # Display recent valid pin bars
                if valid_time_pins:
                    recent_pin_data = []
                    current_time = ensure_timezone_naive(datetime.now())
                    
                    for pb in valid_time_pins[-5:]:  # Show last 5 valid ones
                        pb_time = ensure_timezone_naive(pb['timestamp'])
                        pb_bst = convert_utc_to_bst(pb_time)
                        time_ago = safe_datetime_subtract(current_time, pb_time)
                        hours_ago = max(1, int(time_ago.total_seconds() / 3600))
                        
                        # Color validation info
                        candle_color = "GREEN" if pb.get('is_bullish_candle', True) else "RED"
                        color_valid = "✅" if (
                            (pb['type'] == PinBarType.BULLISH and pb.get('is_bullish_candle', True)) or
                            (pb['type'] == PinBarType.BEARISH and not pb.get('is_bullish_candle', True))
                        ) else "❌"
                        
                        recent_pin_data.append({
                            'Time Ago': f"{hours_ago}h ago" if hours_ago < 24 else f"{hours_ago//24}d ago",
                            'BST Time': pb_bst.strftime('%m/%d %H:%M'),
                            'Type': f"{'📈' if pb['type'] == PinBarType.BULLISH else '📉'} {pb['type'].value.title()}",
                            'Candle Color': f"{color_valid} {candle_color}",
                            'Strength': f"{pb['strength']:.1f}%",
                            'Price': f"{pb['close']:.5f}"
                        })
                    
                    if recent_pin_data:
                        pin_df = pd.DataFrame(recent_pin_data)
                        st.dataframe(pin_df, use_container_width=True, hide_index=True)
                        
                        # Trading opportunity alert
                        latest_pin = valid_time_pins[-1]
                        latest_pin_time = ensure_timezone_naive(latest_pin['timestamp'])
                        time_since = safe_datetime_subtract(current_time, latest_pin_time)
                        
                        if time_since.total_seconds() < 7200:  # Less than 2 hours
                            hours_since = max(1, int(time_since.total_seconds()/3600))
                            color_check = "✅ VALID" if latest_pin.get('is_bullish_candle', True) == (latest_pin['type'] == PinBarType.BULLISH) else "❌ INVALID"
                            
                            if color_check == "✅ VALID":
                                st.success(f"🚨 **VALID Trading Opportunity:** {latest_pin['type'].value.title()} pin bar detected {hours_since}h ago with correct candle color!")
                            else:
                                st.error(f"⚠️ **INVALID Pin Bar:** {latest_pin['type'].value.title()} pin bar detected but wrong candle color - would be rejected")
                else:
                    st.info("ℹ️ No valid pin bars detected during trading hours (3:00-17:00 BST)")
            else:
                st.info("ℹ️ No pin bars detected in recent data")
            
            # Market summary
            st.subheader("📋 Market Summary")
            
            latest_price = data_with_indicators['Close'].iloc[-1]
            ema6 = data_with_indicators['EMA6'].iloc[-1]
            ema18 = data_with_indicators['EMA18'].iloc[-1]
            ema50 = data_with_indicators['EMA50'].iloc[-1]
            
            # Trend analysis
            if ema6 > ema18 > ema50:
                trend = "🟢 Strong Uptrend"
                trend_detail = "Bullish pin bars only"
            elif ema6 > ema18:
                trend = "🔵 Weak Uptrend"
                trend_detail = "Bullish pin bars preferred"
            elif ema6 < ema18 < ema50:
                trend = "🔴 Strong Downtrend"
                trend_detail = "Bearish pin bars only"
            elif ema6 < ema18:
                trend = "🟠 Weak Downtrend"
                trend_detail = "Bearish pin bars preferred"
            else:
                trend = "⚪ Sideways/Mixed"
                trend_detail = "No clear bias"
            
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            with col_summary1:
                st.metric("Current Price", f"{latest_price:.5f}")
                st.metric("Trend Direction", trend)
                st.caption(trend_detail)
            
            with col_summary2:
                st.metric("EMA6", f"{ema6:.5f}")
                st.metric("EMA18", f"{ema18:.5f}")
                
                # Trading hours status
                current_utc = datetime.now()
                if is_valid_trading_time(current_utc):
                    st.success("🟢 SCAN HOURS ACTIVE")
                elif should_close_trade_time(current_utc):
                    st.error("🔴 FORCE CLOSE TIME")
                else:
                    st.warning("🟡 OUTSIDE TRADING HOURS")
            
            with col_summary3:
                distance_to_ema6 = abs(latest_price - ema6) / ema6 * 100
                st.metric("Distance to EMA6", f"{distance_to_ema6:.2f}%")
                
                if not pd.isna(data_with_indicators['SMA200'].iloc[-1]):
                    sma200 = data_with_indicators['SMA200'].iloc[-1]
                    st.metric("SMA200", f"{sma200:.5f}")
                else:
                    st.metric("SMA200", "Calculating...")
                
                # Next trading window
                current_bst = convert_utc_to_bst(current_utc)
                next_scan = current_bst.replace(hour=3, minute=0, second=0, microsecond=0)
                if current_bst.hour >= 17 or current_bst.hour < 3:  # After 5 PM or before 3 AM
                    if current_bst.hour < 3:  # Early morning before 3 AM
                        # Next scan is today at 3 AM
                        pass
                    else:  # After 5 PM
                        # Next scan is tomorrow at 3 AM
                        next_scan += timedelta(days=1)
                st.caption(f"Next scan: {next_scan.strftime('%m/%d %H:%M BST')}")
        
        else:
            st.error("❌ Unable to fetch chart data. Please try again or select a different symbol.")


@st.cache_data(ttl=900)  # Cache for 15 minutes
def detect_recent_pin_bars(data: pd.DataFrame, config: Dict) -> List:
    """Pin bar detection for live analysis returning compatible format with caching"""
    pin_bars = []
    
    if len(data) < 20:
        return pin_bars
    
    # Normalize timezone
    data = normalize_datetime_index(data)
    
    # Use detector settings
    detector = PinBarDetector(
        min_wick_ratio=config.get('min_wick_ratio', 0.55),
        max_body_ratio=config.get('max_body_ratio', 0.4),
        max_opposite_wick=config.get('max_opposite_wick', 0.3)
    )
    
    # Start detection after sufficient data - analyze more recent data for live analysis
    start_idx = max(200, len(data) - 500)  # Analyze last 500 candles for better performance
    
    for i in range(start_idx, len(data)):
        row = data.iloc[i]
        
        # Skip if EMAs not available
        if pd.isna(row['EMA6']) or pd.isna(row['EMA18']):
            continue
        
        candle = Candle(
            timestamp=ensure_timezone_naive(row.name),
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close']
        )
        
        # Use available indicators with fallbacks
        ema50 = row['EMA50'] if not pd.isna(row['EMA50']) else row['EMA18']
        sma200 = row['SMA200'] if not pd.isna(row['SMA200']) else row['Close']
        
        # Detection
        pin_bar_type, strength = detector.detect_pin_bar(
            candle, row['EMA6'], row['EMA18'], ema50, sma200
        )
        
        if pin_bar_type != PinBarType.NONE and strength > 30:  # Quality threshold
            # Return in compatible dict format
            pin_bars.append({
                'timestamp': candle.timestamp,
                'type': pin_bar_type,
                'strength': strength,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'is_bullish_candle': candle.is_bullish(),
                'body_size': abs(candle.close - candle.open)
            })
    
    return pin_bars


# ================================
# MAIN SYSTEM CLASS
# ================================

class TrendSurferSystem:
    """Enhanced Trend Surfer trading system"""
    
    def __init__(self):
        self.ui = TrendSurferUI()
        self.backtester = TrendSurferBacktester()
        self.chart_builder = ChartBuilder()
    
    def run_streamlit_app(self):
        """Enhanced Streamlit application"""
        st.set_page_config(
            page_title="Trend Surfer",
            page_icon="🏄‍♂️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # CSS styling
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #2e8b57, #32cd32);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stAlert > div {
            padding: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🏄‍♂️ Trend Surfer Strategy</h1>
            <p>Professional FX Backtester with OANDA Data Integration</p>
            <p style="font-size: 14px; opacity: 0.8;">⚡ Intelligent Caching • 5-10x Faster Performance • Up to 5 Years Data • Complete Pin Bar Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Render sidebar configuration
        config = self.ui.render_sidebar()
        
        # Main content tabs
        tab1, tab2 = st.tabs([
            "📊 Live Analysis",
            "🔬 Backtesting"
        ])
        
        with tab1:
            render_live_analysis_tab(config)
        
        with tab2:
            self.ui.render_backtest_tab(config)


# ================================
# MAIN APPLICATION ENTRY POINT
# ================================

def main():
    """Main application entry point"""
    try:
        system = TrendSurferSystem()
        system.run_streamlit_app()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        
        # Error details
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()