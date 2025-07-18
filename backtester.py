"""
TREND SURFER - OANDA API INTEGRATION
===================================
Professional pin bar trading system with Oanda API integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import pytz
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error
import json
import time as time_module

# ================================
# CONFIGURATION
# ================================

# Oanda API Configuration
OANDA_API_KEY = st.secrets.get("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = st.secrets.get("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT = st.secrets.get("OANDA_ENVIRONMENT", "practice")  # practice or live

# ================================
# UTILITY FUNCTIONS
# ================================

def ensure_timezone_naive(dt):
    """Convert timezone-aware datetime to naive UTC datetime"""
    if dt is None:
        return None
    if hasattr(dt, 'tz_localize'):
        if dt.tz is not None:
            return dt.tz_convert('UTC').tz_localize(None)
        return dt
    elif hasattr(dt, 'tzinfo'):
        if dt.tzinfo is not None:
            return dt.astimezone(pytz.UTC).replace(tzinfo=None)
        return dt
    return dt

def convert_utc_to_bst(utc_dt):
    """Convert UTC datetime to BST"""
    if utc_dt is None:
        return None
    utc_dt = ensure_timezone_naive(utc_dt)
    utc_tz = pytz.UTC
    bst_tz = pytz.timezone('Europe/London')
    utc_aware = utc_tz.localize(utc_dt)
    return utc_aware.astimezone(bst_tz)

def is_valid_trading_time(utc_dt):
    """Check if time is within valid trading hours (3:00-16:00 BST)"""
    bst_dt = convert_utc_to_bst(utc_dt)
    if bst_dt is None:
        return False
    current_time = bst_dt.time()
    return time(3, 0) <= current_time <= time(16, 0)

def should_close_trade_time(utc_dt):
    """Check if trade should be closed due to time constraints (20:00 BST)"""
    bst_dt = convert_utc_to_bst(utc_dt)
    if bst_dt is None:
        return False
    current_time = bst_dt.time()
    return current_time >= time(20, 0)

# ================================
# DATA STRUCTURES
# ================================

class PinBarType(Enum):
    NONE = "none"
    BULLISH = "bullish"
    BEARISH = "bearish"

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"

class TradeStatus(Enum):
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    CLOSED_BREAKEVEN = "closed_breakeven"
    CLOSED_TIME = "closed_time"
    NOT_TRIGGERED = "not_triggered"

@dataclass
class Candle:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    def is_bullish(self) -> bool:
        body_size = abs(self.close - self.open)
        candle_range = self.high - self.low
        if candle_range == 0:
            return True
        if body_size / candle_range <= 0.1:
            return True
        return self.close >= self.open
    
    def is_bearish(self) -> bool:
        body_size = abs(self.close - self.open)
        candle_range = self.high - self.low
        if candle_range == 0:
            return True
        if body_size / candle_range <= 0.1:
            return True
        return self.close < self.open

@dataclass
class Trade:
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
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = status
        if reason:
            self.forced_close_reason = reason

@dataclass
class PinBarResult:
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

# ================================
# OANDA DATA FETCHER
# ================================

class OandaDataFetcher:
    def __init__(self, api_key: str, account_id: str, environment: str = "practice"):
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        
        if environment == "live":
            self.client = oandapyV20.API(access_token=api_key)
        else:
            self.client = oandapyV20.API(access_token=api_key, environment="practice")
    
    @st.cache_data(ttl=300)
    def fetch_data(_self, symbol: str, granularity: str, start_date: str, end_date: str, count: int = 5000) -> pd.DataFrame:
        """Fetch OHLCV data from Oanda API"""
        try:
            # Convert symbol format (GBPUSD=X -> GBP_USD)
            oanda_symbol = _self._convert_symbol_to_oanda(symbol)
            
            # Convert granularity
            oanda_granularity = _self._convert_granularity_to_oanda(granularity)
            
            params = {
                "granularity": oanda_granularity,
                "from": start_date + "T00:00:00Z",
                "to": end_date + "T23:59:59Z",
                "count": count
            }
            
            request = instruments.InstrumentsCandles(instrument=oanda_symbol, params=params)
            response = _self.client.request(request)
            
            if not response or 'candles' not in response:
                st.error(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for candle in response['candles']:
                if candle['complete']:
                    data.append({
                        'timestamp': pd.to_datetime(candle['time']),
                        'Open': float(candle['mid']['o']),
                        'High': float(candle['mid']['h']),
                        'Low': float(candle['mid']['l']),
                        'Close': float(candle['mid']['c']),
                        'Volume': int(candle['volume'])
                    })
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            return df
            
        except V20Error as e:
            st.error(f"Oanda API Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _convert_symbol_to_oanda(self, symbol: str) -> str:
        """Convert Yahoo Finance symbol to Oanda format"""
        symbol_map = {
            "EURUSD=X": "EUR_USD",
            "GBPUSD=X": "GBP_USD",
            "USDJPY=X": "USD_JPY",
            "AUDUSD=X": "AUD_USD",
            "USDCAD=X": "USD_CAD",
            "USDCHF=X": "USD_CHF",
            "NZDUSD=X": "NZD_USD",
            "EURJPY=X": "EUR_JPY",
            "GBPJPY=X": "GBP_JPY",
            "EURGBP=X": "EUR_GBP",
            "AUDCAD=X": "AUD_CAD",
            "AUDCHF=X": "AUD_CHF",
            "AUDNZD=X": "AUD_NZD",
            "CADJPY=X": "CAD_JPY",
            "CHFJPY=X": "CHF_JPY"
        }
        return symbol_map.get(symbol, symbol.replace("=X", "").replace("USD", "_USD"))
    
    def _convert_granularity_to_oanda(self, granularity: str) -> str:
        """Convert granularity to Oanda format"""
        granularity_map = {
            "1m": "M1",
            "5m": "M5",
            "15m": "M15",
            "30m": "M30",
            "1h": "H1",
            "4h": "H4",
            "1d": "D"
        }
        return granularity_map.get(granularity, "H1")
    
    def fetch_multi_timeframe_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        start_date = ensure_timezone_naive(start_date)
        end_date = ensure_timezone_naive(end_date)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        timeframes = ['15m', '30m', '1h', '4h']
        data = {}
        
        for tf in timeframes:
            df = self.fetch_data(symbol, tf, start_str, end_str)
            if not df.empty:
                data[tf] = df
        
        return data

# ================================
# PIN BAR DETECTOR
# ================================

class PinBarDetector:
    def __init__(self, min_wick_ratio: float = 0.55, max_body_ratio: float = 0.4, max_opposite_wick: float = 0.3):
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        self.max_opposite_wick = max_opposite_wick
    
    def detect_pin_bar(self, candle: Candle, ema6: float, ema18: float, ema50: float, sma200: float) -> Tuple[PinBarType, float]:
        candle_range = candle.high - candle.low
        if candle_range == 0:
            return PinBarType.NONE, 0.0
        
        body_size = abs(candle.close - candle.open)
        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low
        
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range
        body_ratio = body_size / candle_range
        
        uptrend_strong = (ema6 > ema18 > ema50 > sma200) and (candle.close > ema6)
        uptrend_moderate = (ema6 > ema18) and (ema6 > sma200) and (candle.close > ema18)
        uptrend = uptrend_strong or uptrend_moderate
        
        downtrend_strong = (ema6 < ema18 < ema50 < sma200) and (candle.close < ema6)
        downtrend_moderate = (ema6 < ema18) and (ema6 < sma200) and (candle.close < ema18)
        downtrend = downtrend_strong or downtrend_moderate
        
        if (lower_wick_ratio >= self.min_wick_ratio and
            body_ratio <= self.max_body_ratio and
            upper_wick_ratio <= self.max_opposite_wick and
            uptrend):
            
            if candle.is_bullish():
                ema_touch = abs(candle.low - ema6) / ema6 <= 0.015
                if ema_touch:
                    strength = self._calculate_strength(lower_wick_ratio, body_ratio, upper_wick_ratio)
                    return PinBarType.BULLISH, strength
        
        elif (upper_wick_ratio >= self.min_wick_ratio and
              body_ratio <= self.max_body_ratio and
              lower_wick_ratio <= self.max_opposite_wick and
              downtrend):
            
            if candle.is_bearish():
                ema_touch = abs(candle.high - ema6) / ema6 <= 0.015
                if ema_touch:
                    strength = self._calculate_strength(upper_wick_ratio, body_ratio, lower_wick_ratio)
                    return PinBarType.BEARISH, strength
        
        return PinBarType.NONE, 0.0
    
    def _calculate_strength(self, dominant_wick: float, body_ratio: float, opposite_wick: float) -> float:
        wick_score = min((dominant_wick - 0.55) / 0.35 * 50, 50)
        body_penalty = body_ratio * 25
        opposite_penalty = max(0, (opposite_wick - 0.1)) * 30
        strength = max(0, min(100, wick_score - body_penalty - opposite_penalty))
        return strength

# ================================
# TRADE MANAGER
# ================================

class TradeManager:
    def __init__(self):
        self.active_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        
    def can_open_trade(self, symbol: str) -> bool:
        return symbol not in self.active_trades
    
    def open_trade(self, trade: Trade) -> bool:
        if self.can_open_trade(trade.symbol):
            self.active_trades[trade.symbol] = trade
            return True
        return False
    
    def close_trade(self, symbol: str, exit_time: pd.Timestamp, exit_price: float, 
                   status: TradeStatus, reason: str = None) -> Optional[Trade]:
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            trade.set_exit(exit_time, exit_price, status, reason)
            self._calculate_trade_pnl(trade)
            self.closed_trades.append(trade)
            del self.active_trades[symbol]
            return trade
        return None
    
    def get_active_trade(self, symbol: str) -> Optional[Trade]:
        return self.active_trades.get(symbol)
    
    def get_all_trades(self) -> List[Trade]:
        return list(self.active_trades.values()) + self.closed_trades
    
    def _calculate_trade_pnl(self, trade: Trade):
        if trade.exit_price is None:
            return
            
        pip_value = self._get_pip_value(trade.symbol)
        
        if trade.direction == TradeDirection.LONG:
            trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_value
        else:
            trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_value
        
        trade.pnl_usd = trade.pnl_pips * trade.lot_size * (10 if 'JPY' in trade.symbol else 1)
    
    def _get_pip_value(self, symbol: str) -> float:
        return 0.01 if 'JPY' in symbol else 0.0001

# ================================
# BACKTESTING ENGINE
# ================================

class TrendSurferBacktester:
    def __init__(self, data_fetcher: OandaDataFetcher):
        self.detector = PinBarDetector()
        self.data_fetcher = data_fetcher
        self.trade_manager = TradeManager()
    
    def run_backtest(self, symbol: str, start_date: datetime, end_date: datetime,
                    risk_reward_ratio: float = 2.0, account_balance: float = 10000.0,
                    risk_percentage: float = 0.01) -> BacktestResults:
        
        self.trade_manager = TradeManager()
        
        start_date = ensure_timezone_naive(start_date)
        end_date = ensure_timezone_naive(end_date)
        
        current_date = ensure_timezone_naive(datetime.now())
        optimized_start = max(start_date, current_date - timedelta(days=59))
        optimized_end = min(end_date, current_date - timedelta(days=1))
        
        data = self.data_fetcher.fetch_multi_timeframe_data(symbol, optimized_start, optimized_end)
        
        if not data or '1h' not in data:
            return BacktestResults()
        
        pin_bars = self._detect_pin_bars_h1(data['1h'])
        trades = self._generate_trades(pin_bars, data, symbol, risk_reward_ratio, 
                                     account_balance, risk_percentage)
        statistics = self._calculate_statistics(trades, symbol, account_balance)
        
        return BacktestResults(
            trades=trades,
            pin_bars=pin_bars,
            statistics=statistics,
            symbol=symbol,
            start_date=optimized_start,
            end_date=optimized_end,
            risk_reward_ratio=risk_reward_ratio,
            total_pin_bars=len(pin_bars),
            valid_trades=len(trades),
            data_1h=data['1h']
        )
    
    def _detect_pin_bars_h1(self, data_1h: pd.DataFrame) -> List[PinBarResult]:
        pin_bars = []
        
        if data_1h.empty or len(data_1h) < 50:
            return pin_bars
        
        data_1h = data_1h.copy()
        data_1h['EMA6'] = data_1h['Close'].ewm(span=6).mean()
        data_1h['EMA18'] = data_1h['Close'].ewm(span=18).mean()
        data_1h['EMA50'] = data_1h['Close'].ewm(span=50).mean()
        data_1h['SMA200'] = data_1h['Close'].rolling(window=200).mean()
        
        start_idx = max(200, len(data_1h) - len(data_1h) + 200)
        
        for i in range(start_idx, len(data_1h)):
            row = data_1h.iloc[i]
            
            if pd.isna(row['EMA6']) or pd.isna(row['EMA18']) or pd.isna(row['SMA200']):
                continue
            
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
            
            ema50 = row['EMA50'] if not pd.isna(row['EMA50']) else row['EMA18']
            
            pin_bar_type, strength = self.detector.detect_pin_bar(
                candle, row['EMA6'], row['EMA18'], ema50, row['SMA200']
            )
            
            if pin_bar_type != PinBarType.NONE:
                pin_bar_result = PinBarResult(
                    timestamp=candle_time,
                    pin_bar_type=pin_bar_type,
                    strength=strength,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    is_bullish_candle=candle.is_bullish(),
                    body_size=abs(candle.close - candle.open),
                    ema6=row['EMA6'],
                    ema18=row['EMA18'],
                    ema50=ema50,
                    sma200=row['SMA200'],
                    bst_time=bst_time_str,
                    in_trading_hours=in_trading_hours
                )
                
                pin_bars.append(pin_bar_result)
        
        return pin_bars
    
    def _generate_trades(self, pin_bars: List[PinBarResult], data: Dict[str, pd.DataFrame],
                        symbol: str, risk_reward_ratio: float, account_balance: float,
                        risk_percentage: float) -> List[Trade]:
        
        valid_pin_bars = [pb for pb in pin_bars if pb.pin_bar_type != PinBarType.NONE]
        
        for pin_bar in valid_pin_bars:
            pin_bar.trade_attempted = True
            
            if not pin_bar.in_trading_hours:
                pin_bar.rejection_reason = "Outside trading hours"
                continue
            
            if not self.trade_manager.can_open_trade(symbol):
                pin_bar.rejection_reason = "Active trade exists"
                continue
            
            if pin_bar.pin_bar_type == PinBarType.BULLISH:
                direction = TradeDirection.LONG
            elif pin_bar.pin_bar_type == PinBarType.BEARISH:
                direction = TradeDirection.SHORT
            else:
                continue
            
            try:
                entry_price, stop_loss, take_profit = self._calculate_trade_levels(
                    pin_bar, direction, symbol, risk_reward_ratio
                )
                
                stop_distance_pips = abs(entry_price - stop_loss) / self._get_pip_value(symbol)
                lot_size = self._calculate_position_size(
                    account_balance, risk_percentage, stop_distance_pips, symbol
                )
                
                trade = Trade(
                    entry_time=pin_bar.timestamp,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    symbol=symbol,
                    lot_size=lot_size
                )
                
                if self.trade_manager.open_trade(trade):
                    pin_bar.trade_success = True
                    self._simulate_trade(trade, data['1h'], symbol)
                
            except Exception as e:
                pin_bar.rejection_reason = f"Trade setup failed: {str(e)[:50]}"
        
        return self.trade_manager.get_all_trades()
    
    def _calculate_trade_levels(self, pin_bar: PinBarResult, direction: TradeDirection,
                               symbol: str, risk_reward_ratio: float) -> Tuple[float, float, float]:
        pip_value = self._get_pip_value(symbol)
        
        if direction == TradeDirection.LONG:
            entry_price = pin_bar.close + (1 * pip_value)
            stop_loss = pin_bar.low - (1 * pip_value)
            risk_distance = entry_price - stop_loss
            take_profit = entry_price + (risk_distance * risk_reward_ratio)
        else:
            entry_price = pin_bar.close - (1 * pip_value)
            stop_loss = pin_bar.high + (1 * pip_value)
            risk_distance = stop_loss - entry_price
            take_profit = entry_price - (risk_distance * risk_reward_ratio)
        
        return entry_price, stop_loss, take_profit
    
    def _simulate_trade(self, trade: Trade, data_1h: pd.DataFrame, symbol: str):
        try:
            entry_idx = data_1h.index.get_indexer([trade.entry_time], method='nearest')[0]
        except:
            self.trade_manager.close_trade(
                symbol, trade.entry_time, trade.entry_price, 
                TradeStatus.NOT_TRIGGERED, "Index error"
            )
            return
        
        if entry_idx + 1 >= len(data_1h):
            self.trade_manager.close_trade(
                symbol, trade.entry_time, trade.entry_price, 
                TradeStatus.NOT_TRIGGERED, "Insufficient data"
            )
            return
        
        triggered = False
        for i in range(entry_idx + 1, min(entry_idx + 5, len(data_1h))):
            candle = data_1h.iloc[i]
            candle_time = ensure_timezone_naive(candle.name)
            
            if should_close_trade_time(candle_time):
                self.trade_manager.close_trade(
                    symbol, candle_time, trade.entry_price, 
                    TradeStatus.NOT_TRIGGERED, "Time cutoff"
                )
                return
            
            if trade.direction == TradeDirection.LONG:
                if candle['High'] >= trade.entry_price:
                    triggered = True
                    break
            else:
                if candle['Low'] <= trade.entry_price:
                    triggered = True
                    break
        
        if not triggered:
            self.trade_manager.close_trade(
                symbol, trade.entry_time, trade.entry_price, 
                TradeStatus.NOT_TRIGGERED, "Entry not reached"
            )
            return
        
        for i in range(entry_idx + 1, len(data_1h)):
            candle = data_1h.iloc[i]
            candle_time = ensure_timezone_naive(candle.name)
            
            if should_close_trade_time(candle_time):
                self.trade_manager.close_trade(
                    symbol, candle_time, candle['Close'], 
                    TradeStatus.CLOSED_TIME, "20:00 BST cutoff"
                )
                return
            
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
            else:
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
    
    def _get_pip_value(self, symbol: str) -> float:
        return 0.01 if 'JPY' in symbol else 0.0001
    
    def _calculate_position_size(self, account_balance: float, risk_percentage: float,
                                stop_loss_pips: float, symbol: str) -> float:
        if stop_loss_pips <= 0:
            return 0.0
        
        risk_amount = account_balance * risk_percentage
        pip_value_usd = 10 if 'JPY' in symbol else 1
        
        position_size = risk_amount / (stop_loss_pips * pip_value_usd)
        return max(0.01, min(10, position_size))
    
    def _calculate_statistics(self, trades: List[Trade], symbol: str, account_balance: float) -> Dict:
        if not trades:
            return {'total_trades': 0, 'account_balance': account_balance}
        
        triggered_trades = [t for t in trades if t.status != TradeStatus.NOT_TRIGGERED]
        
        if not triggered_trades:
            return {'total_trades': 0, 'untriggered_trades': len(trades), 'account_balance': account_balance}
        
        total_trades = len(triggered_trades)
        winning_trades = [t for t in triggered_trades if t.status == TradeStatus.CLOSED_PROFIT]
        losing_trades = [t for t in triggered_trades if t.status == TradeStatus.CLOSED_LOSS]
        time_closed_trades = [t for t in triggered_trades if t.status == TradeStatus.CLOSED_TIME]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        time_close_count = len(time_closed_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl_pips = sum(t.pnl_pips for t in triggered_trades if t.pnl_pips is not None)
        total_pnl_usd = sum(t.pnl_usd for t in triggered_trades if t.pnl_usd is not None)
        
        avg_win = np.mean([t.pnl_pips for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pips for t in losing_trades]) if losing_trades else 0
        
        total_win_pips = sum(t.pnl_pips for t in winning_trades)
        total_loss_pips = abs(sum(t.pnl_pips for t in losing_trades))
        
        profit_factor = (total_win_pips / total_loss_pips) if total_loss_pips > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'time_closed_trades': time_close_count,
            'untriggered_trades': len(trades) - total_trades,
            'win_rate': win_rate,
            'total_pnl_pips': total_pnl_pips,
            'total_pnl_usd': total_pnl_usd,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'trigger_rate': (total_trades / len(trades) * 100) if trades else 0,
            'account_balance': account_balance,
            'ending_balance': account_balance + total_pnl_usd,
            'return_percent': (total_pnl_usd / account_balance * 100) if account_balance > 0 else 0
        }

# ================================
# CHART BUILDER
# ================================

class ChartBuilder:
    def __init__(self):
        self.colors = {
            'background': '#0d1421',
            'text': '#d1d4dc',
            'grid': '#2a2e39',
            'bullish': '#26a69a',
            'bearish': '#ef5350'
        }
    
    def create_chart(self, df: pd.DataFrame, pin_bars: List, symbol: str, timeframe: str, 
                    show_ma: bool = True, highlight_trade=None) -> go.Figure:
        
        if 'EMA6' not in df.columns:
            df = self.calculate_moving_averages(df)
        
        fig = go.Figure()
        
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
        
        if show_ma:
            self._add_moving_averages(fig, df)
        
        self._add_pin_bar_highlights(fig, pin_bars)
        
        if highlight_trade:
            self._add_trade_markers(fig, highlight_trade)
        
        self._apply_styling(fig, symbol, timeframe)
        
        return fig
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['EMA6'] = df['Close'].ewm(span=6).mean()
        df['EMA18'] = df['Close'].ewm(span=18).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        return df
    
    def _add_moving_averages(self, fig: go.Figure, df: pd.DataFrame):
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA6'], name='EMA6',
            line=dict(color='#ff7f7f', width=1.5),
            opacity=0.8
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA18'], name='EMA18',
            line=dict(color='#7fc7ff', width=1.5),
            opacity=0.8
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA50'], name='EMA50',
            line=dict(color='#1f77b4', width=2),
            opacity=0.9
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA200'], name='SMA200',
            line=dict(color='#d62728', width=2.5),
            opacity=0.9
        ))
    
    def _add_pin_bar_highlights(self, fig: go.Figure, pin_bars: List):
        for pin_bar in pin_bars:
            if hasattr(pin_bar, 'timestamp'):
                timestamp = pin_bar.timestamp
                pin_type = pin_bar.pin_bar_type
                low_price = pin_bar.low
                high_price = pin_bar.high
            else:
                timestamp = pin_bar['timestamp']
                pin_type = pin_bar['type']
                low_price = pin_bar['low']
                high_price = pin_bar['high']
            
            if pin_type == PinBarType.BULLISH:
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[low_price * 0.999],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Bullish Pin',
                    showlegend=False
                ))
            elif pin_type == PinBarType.BEARISH:
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[high_price * 1.001],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Bearish Pin',
                    showlegend=False
                ))
    
    def _add_trade_markers(self, fig: go.Figure, trade: Trade):
        fig.add_trace(go.Scatter(
            x=[trade.entry_time],
            y=[trade.entry_price],
            mode='markers',
            marker=dict(symbol='star', size=15, color='gold'),
            name='Entry',
            showlegend=False
        ))
        
        if trade.exit_time and trade.exit_price:
            color = 'green' if trade.pnl_pips > 0 else 'red'
            fig.add_trace(go.Scatter(
                x=[trade.exit_time],
                y=[trade.exit_price],
                mode='markers',
                marker=dict(symbol='circle', size=12, color=color),
                name='Exit',
                showlegend=False
            ))
        
        fig.add_hline(y=trade.entry_price, line_dash="solid", line_color="blue")
        fig.add_hline(y=trade.stop_loss, line_dash="dash", line_color="red")
        fig.add_hline(y=trade.take_profit, line_dash="dash", line_color="green")
    
    def _apply_styling(self, fig: go.Figure, symbol: str, timeframe: str):
        is_jpy_pair = 'JPY' in symbol
        y_tick_format = '.2f' if is_jpy_pair else '.5f'
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font_color=self.colors['text'],
            title=f"{symbol} - {timeframe}",
            title_font_size=18,
            xaxis=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                rangeslider=dict(visible=False)
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
            height=600,
            margin=dict(l=20, r=60, t=60, b=20)
        )

# ================================
# STREAMLIT UI
# ================================

class TrendSurferUI:
    def __init__(self, data_fetcher: OandaDataFetcher):
        self.backtester = TrendSurferBacktester(data_fetcher)
        self.chart_builder = ChartBuilder()
        self.data_fetcher = data_fetcher
        
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
    
    def render_sidebar(self):
        st.sidebar.header("⚙️ Configuration")
        
        forex_pairs = [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
            "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"
        ]
        
        selected_symbol = st.sidebar.selectbox("Trading Pair", forex_pairs, index=1)
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=datetime.now().date() - timedelta(days=1)
        )
        
        duration = st.sidebar.selectbox("Duration", ["1 Week", "2 Weeks", "1 Month"], index=2)
        duration_days = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}[duration]
        start_date = end_date - timedelta(days=duration_days)
        
        account_size = st.sidebar.selectbox(
            "Account Size",
            [1000, 5000, 10000, 25000, 50000],
            index=2,
            format_func=lambda x: f"${x:,}"
        )
        
        risk_percentage = st.sidebar.selectbox(
            "Risk Per Trade",
            [0.005, 0.01, 0.015, 0.02],
            index=1,
            format_func=lambda x: f"{x*100:.1f}%"
        )
        
        risk_reward = st.sidebar.selectbox(
            "Risk:Reward",
            [1.5, 2.0, 2.5, 3.0],
            index=1,
            format_func=lambda x: f"1:{x}"
        )
        
        return {
            'symbol': selected_symbol,
            'start_date': start_date,
            'end_date': end_date,
            'account_size': account_size,
            'risk_percentage': risk_percentage,
            'risk_reward': risk_reward
        }
    
    def render_backtest_tab(self, config: Dict):
        st.header("📊 Backtesting")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric("Symbol", config['symbol'])
            st.metric("Period", f"{config['start_date']} to {config['end_date']}")
            st.metric("Account Size", f"${config['account_size']:,}")
        
        with col2:
            if st.button("Run Backtest", type="primary", use_container_width=True):
                with st.spinner("Running backtest..."):
                    results = self.backtester.run_backtest(
                        symbol=config['symbol'],
                        start_date=datetime.combine(config['start_date'], datetime.min.time()),
                        end_date=datetime.combine(config['end_date'], datetime.min.time()),
                        risk_reward_ratio=config['risk_reward'],
                        account_balance=config['account_size'],
                        risk_percentage=config['risk_percentage']
                    )
                    
                    if results.trades or results.statistics:
                        st.session_state.backtest_results = results
                        st.success("Backtest completed!")
                    else:
                        st.error("No trades found.")
        
        if st.session_state.backtest_results:
            self.display_results(st.session_state.backtest_results)
    
    def display_results(self, results: BacktestResults):
        stats = results.statistics
        
        if not stats:
            return
        
        st.subheader("Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", stats.get('total_trades', 0))
            st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
        
        with col2:
            st.metric("P&L (Pips)", f"{stats.get('total_pnl_pips', 0):.1f}")
            st.metric("P&L (USD)", f"${stats.get('total_pnl_usd', 0):.2f}")
        
        with col3:
            st.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
            st.metric("Return %", f"{stats.get('return_percent', 0):.2f}%")
        
        with col4:
            st.metric("Time Closes", stats.get('time_closed_trades', 0))
            st.metric("Trigger Rate", f"{stats.get('trigger_rate', 0):.1f}%")
        
        if results.trades:
            triggered_trades = [t for t in results.trades if t.status != TradeStatus.NOT_TRIGGERED]
            
            if triggered_trades:
                st.subheader("Trades")
                
                trade_data = []
                for i, trade in enumerate(triggered_trades):
                    trade_data.append({
                        '#': i + 1,
                        'Date': trade.entry_time.strftime('%Y-%m-%d %H:%M'),
                        'Direction': trade.direction.value.title(),
                        'Entry': f"{trade.entry_price:.5f}",
                        'Exit': f"{trade.exit_price:.5f}" if trade.exit_price else "Open",
                        'P&L (Pips)': f"{trade.pnl_pips:.1f}",
                        'P&L (USD)': f"${trade.pnl_usd:.2f}",
                        'Status': trade.status.value.replace('_', ' ').title()
                    })
                
                df = pd.DataFrame(trade_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                if st.button("View Chart"):
                    self.display_chart(results, triggered_trades[0])
    
    def display_chart(self, results: BacktestResults, trade: Trade):
        st.subheader("Chart Analysis")
        
        if results.data_1h.empty:
            st.error("No chart data available")
            return
        
        chart_df = self.chart_builder.calculate_moving_averages(results.data_1h)
        
        fig = self.chart_builder.create_chart(
            chart_df,
            results.pin_bars,
            results.symbol,
            "1H",
            show_ma=True,
            highlight_trade=trade
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ================================
# MAIN SYSTEM
# ================================

class TrendSurferSystem:
    def __init__(self):
        if not OANDA_API_KEY:
            st.error("Please set OANDA_API_KEY in Streamlit secrets")
            st.stop()
        
        self.data_fetcher = OandaDataFetcher(OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENVIRONMENT)
        self.ui = TrendSurferUI(self.data_fetcher)
    
    def run_app(self):
        st.set_page_config(
            page_title="Trend Surfer Pro",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📈 Trend Surfer Pro")
        st.caption("Professional FX Pin Bar Trading System")
        
        config = self.ui.render_sidebar()
        
        tab1, tab2 = st.tabs(["Live Analysis", "Backtesting"])
        
        with tab1:
            self.render_live_analysis(config)
        
        with tab2:
            self.ui.render_backtest_tab(config)
    
    def render_live_analysis(self, config: Dict):
        st.header("Live Market Analysis")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        with st.spinner("Fetching live data..."):
            data = self.data_fetcher.fetch_data(
                config['symbol'], 
                '1h',
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        if not data.empty:
            data_with_indicators = self.ui.chart_builder.calculate_moving_averages(data)
            pin_bars = self._detect_live_pin_bars(data_with_indicators)
            
            fig = self.ui.chart_builder.create_chart(
                data_with_indicators,
                pin_bars,
                config['symbol'],
                "1H"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if pin_bars:
                st.subheader(f"Recent Pin Bars ({len(pin_bars)})")
                
                pin_data = []
                for pb in pin_bars[-5:]:
                    pin_data.append({
                        'Time': pb['timestamp'].strftime('%m/%d %H:%M'),
                        'Type': pb['type'].value.title(),
                        'Strength': f"{pb['strength']:.1f}%",
                        'Price': f"{pb['close']:.5f}"
                    })
                
                if pin_data:
                    df = pd.DataFrame(pin_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error("Unable to fetch data")
    
    def _detect_live_pin_bars(self, data: pd.DataFrame) -> List:
        pin_bars = []
        
        if len(data) < 20:
            return pin_bars
        
        detector = PinBarDetector()
        start_idx = max(200, len(data) - 50)
        
        for i in range(start_idx, len(data)):
            row = data.iloc[i]
            
            if pd.isna(row['EMA6']) or pd.isna(row['EMA18']):
                continue
            
            candle = Candle(
                timestamp=ensure_timezone_naive(row.name),
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close']
            )
            
            ema50 = row['EMA50'] if not pd.isna(row['EMA50']) else row['EMA18']
            sma200 = row['SMA200'] if not pd.isna(row['SMA200']) else row['Close']
            
            pin_bar_type, strength = detector.detect_pin_bar(
                candle, row['EMA6'], row['EMA18'], ema50, sma200
            )
            
            if pin_bar_type != PinBarType.NONE and strength > 30:
                pin_bars.append({
                    'timestamp': candle.timestamp,
                    'type': pin_bar_type,
                    'strength': strength,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close
                })
        
        return pin_bars

def main():
    system = TrendSurferSystem()
    system.run_app()

if __name__ == "__main__":
    main()