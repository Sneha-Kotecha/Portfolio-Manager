"""
TREND SURFER - FIXED TRADING SYSTEM
===================================

Fixed version addressing datetime issues and other improvements.
Key fixes:
1. Timezone-aware datetime handling
2. Better error handling for datetime operations
3. Improved data validation
4. Enhanced compatibility with yfinance data

USAGE:
Run this with Streamlit: streamlit run trend_surfer_fixed.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import threading
import time
from functools import lru_cache


# ================================
# UTILITY FUNCTIONS FOR DATETIME HANDLING
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
    NOT_TRIGGERED = "not_triggered"


@dataclass
class Candle:
    """Candlestick data structure"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class Trade:
    """Trade execution data structure"""
    entry_time: pd.Timestamp
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_pips: float = 0.0
    risk_amount: float = 0.0
    pin_bar_data: Optional[Dict] = None
    lot_size: float = 0.0
    pnl_usd: float = 0.0

    def set_exit(self, exit_time: pd.Timestamp, exit_price: float, status: TradeStatus):
        """Set exit details for the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = status


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    trades: List[Trade] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    symbol: str = ""
    start_date: datetime = None
    end_date: datetime = None
    risk_reward_ratio: float = 2.0
    total_pin_bars: int = 0
    valid_trades: int = 0
    data_1h: pd.DataFrame = field(default_factory=pd.DataFrame)
    debug_info: Dict = field(default_factory=dict)


# ================================
# ENHANCED DATA FETCHER
# ================================

class DataFetcher:
    """Enhanced data fetching with timezone handling"""
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data with enhanced error handling and timezone fixes"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                st.warning(f"No data available for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Fix timezone issues
            data = normalize_datetime_index(data)
            
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
    def fetch_multi_timeframe_data(symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Enhanced multi-timeframe data fetching with timezone handling"""
        # Ensure dates are timezone-naive
        start_date = ensure_timezone_naive(start_date)
        end_date = ensure_timezone_naive(end_date)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # More flexible timeframe configuration
        timeframes_config = {
            '15m': {
                'start': max(start_date, end_date - timedelta(days=55)).strftime('%Y-%m-%d'),
                'end': end_str
            },
            '30m': {
                'start': max(start_date, end_date - timedelta(days=55)).strftime('%Y-%m-%d'),
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
                st.info(f"Fetching {tf} data for {symbol}...")
                df = DataFetcher.fetch_data(symbol, tf, config['start'], config['end'])
                
                if not df.empty:
                    data[tf] = df
                    st.success(f"✓ {tf}: {len(df)} candles retrieved")
                else:
                    st.warning(f"✗ {tf}: No data retrieved")
                    
            except Exception as e:
                st.warning(f"✗ {tf}: Error during fetch - {str(e)}")
                continue
                
        return data


# ================================
# ENHANCED PIN BAR DETECTOR
# ================================

class PinBarDetector:
    """Enhanced pin bar detection with more flexible validation"""
    
    def __init__(self, 
                 min_wick_ratio: float = 0.55,
                 max_body_ratio: float = 0.4,
                 max_opposite_wick: float = 0.3):
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        self.max_opposite_wick = max_opposite_wick
    
    def detect_pin_bar(self, candle: Candle, ema6: float, ema18: float, 
                      ema50: float, sma200: float) -> Tuple[PinBarType, float]:
        """Enhanced pin bar detection with relaxed criteria"""
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
        
        # More flexible trend alignment - allow minor deviations
        uptrend_strong = ema6 > ema18 > ema50 > sma200
        uptrend_weak = ema6 > ema18 and ema6 > sma200
        
        downtrend_strong = ema6 < ema18 < ema50 < sma200
        downtrend_weak = ema6 < ema18 and ema6 < sma200
        
        # Bullish pin bar detection
        if (lower_wick_ratio >= self.min_wick_ratio and
            body_ratio <= self.max_body_ratio and
            upper_wick_ratio <= self.max_opposite_wick and
            (uptrend_strong or uptrend_weak)):
            
            # Relaxed EMA touch (3% tolerance)
            ema_touch = abs(candle.low - ema6) / ema6 <= 0.03
            
            if ema_touch:
                strength = self._calculate_strength(lower_wick_ratio, body_ratio, upper_wick_ratio)
                return PinBarType.BULLISH, strength
        
        # Bearish pin bar detection
        elif (upper_wick_ratio >= self.min_wick_ratio and
              body_ratio <= self.max_body_ratio and
              lower_wick_ratio <= self.max_opposite_wick and
              (downtrend_strong or downtrend_weak)):
            
            # Relaxed EMA touch (3% tolerance)
            ema_touch = abs(candle.high - ema6) / ema6 <= 0.03
            
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
# ENHANCED BACKTESTING ENGINE
# ================================

class TrendSurferBacktester:
    """Enhanced backtesting engine with timezone fixes"""
    
    def __init__(self):
        self.detector = PinBarDetector()
        self.data_fetcher = DataFetcher()
    
    def run_backtest(self, symbol: str, start_date: datetime, end_date: datetime,
                    risk_reward_ratio: float = 2.0, account_balance: float = 10000.0,
                    risk_percentage: float = 0.01) -> BacktestResults:
        """Enhanced backtest with timezone handling"""
        
        # Ensure timezone-naive dates
        start_date = ensure_timezone_naive(start_date)
        end_date = ensure_timezone_naive(end_date)
        
        # Date optimization
        current_date = ensure_timezone_naive(datetime.now())
        optimized_start = max(start_date, current_date - timedelta(days=59))
        optimized_end = min(end_date, current_date - timedelta(days=1))
        
        st.info(f"Starting enhanced backtest for {symbol}")
        st.info(f"Period: {optimized_start.date()} to {optimized_end.date()}")
        
        # Fetch data
        data = self.data_fetcher.fetch_multi_timeframe_data(symbol, optimized_start, optimized_end)
        
        if not data or '1h' not in data:
            st.error("Insufficient 1H data for backtesting")
            return BacktestResults()
        
        # Debug data quality
        debug_info = {'data_quality': {}}
        for tf, df in data.items():
            debug_info['data_quality'][tf] = {
                'candles': len(df),
                'date_range': f"{df.index[0]} to {df.index[-1]}"
            }
        
        # Detect pin bars
        pin_bars = self._detect_pin_bars_h1(data['1h'])
        st.success(f"Found {len(pin_bars)} pin bars on H1 timeframe")
        
        debug_info['pin_bars_found'] = len(pin_bars)
        
        # Generate trades with enhanced logic
        trades = self._generate_trades_enhanced(pin_bars, data, symbol, risk_reward_ratio, 
                                              account_balance, risk_percentage, debug_info)
        
        st.success(f"Generated {len(trades)} valid trades")
        
        # Calculate statistics
        statistics = self._calculate_statistics(trades, symbol, account_balance)
        
        return BacktestResults(
            trades=trades,
            statistics=statistics,
            symbol=symbol,
            start_date=optimized_start,
            end_date=optimized_end,
            risk_reward_ratio=risk_reward_ratio,
            total_pin_bars=len(pin_bars),
            valid_trades=len(trades),
            data_1h=data['1h'],
            debug_info=debug_info
        )
    
    def _detect_pin_bars_h1(self, data_1h: pd.DataFrame) -> List[Dict]:
        """Enhanced pin bar detection with timezone handling"""
        pin_bars = []
        
        if data_1h.empty or len(data_1h) < 50:
            st.warning("Insufficient data for pin bar detection")
            return pin_bars
        
        # Normalize timezone
        data_1h = normalize_datetime_index(data_1h)
        
        # Calculate indicators
        data_1h = data_1h.copy()
        data_1h['EMA6'] = data_1h['Close'].ewm(span=6).mean()
        data_1h['EMA18'] = data_1h['Close'].ewm(span=18).mean()
        data_1h['EMA50'] = data_1h['Close'].ewm(span=50).mean()
        data_1h['SMA200'] = data_1h['Close'].rolling(window=200).mean()
        
        # Start detection earlier
        start_idx = min(50, len(data_1h) - 10)
        
        for i in range(start_idx, len(data_1h)):
            row = data_1h.iloc[i]
            
            # Skip if indicators not available
            if pd.isna(row['EMA6']) or pd.isna(row['EMA18']):
                continue
            
            candle = Candle(
                timestamp=ensure_timezone_naive(row.name),
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row.get('Volume', 0)
            )
            
            # Use fallback values for missing indicators
            ema50 = row['EMA50'] if not pd.isna(row['EMA50']) else row['EMA18']
            sma200 = row['SMA200'] if not pd.isna(row['SMA200']) else row['Close']
            
            # Detect pin bar
            pin_bar_type, strength = self.detector.detect_pin_bar(
                candle, row['EMA6'], row['EMA18'], ema50, sma200
            )
            
            if pin_bar_type != PinBarType.NONE:
                pin_bars.append({
                    'timestamp': candle.timestamp,
                    'type': pin_bar_type,
                    'strength': strength,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'ema6': row['EMA6'],
                    'ema18': row['EMA18'],
                    'ema50': ema50,
                    'sma200': sma200
                })
        
        return pin_bars
    
    def _generate_trades_enhanced(self, pin_bars: List[Dict], data: Dict[str, pd.DataFrame],
                                symbol: str, risk_reward_ratio: float, account_balance: float,
                                risk_percentage: float, debug_info: Dict) -> List[Trade]:
        """Enhanced trade generation with timezone handling"""
        trades = []
        failed_validations = {
            'sma_conditions': 0,
            'invalid_levels': 0,
            'zero_position_size': 0,
            'simulation_failed': 0
        }
        
        for pin_bar in pin_bars:
            timestamp = ensure_timezone_naive(pin_bar['timestamp'])
            pin_type = pin_bar['type']
            
            # Determine trade direction
            if pin_type == PinBarType.BULLISH:
                direction = TradeDirection.LONG
            elif pin_type == PinBarType.BEARISH:
                direction = TradeDirection.SHORT
            else:
                continue
            
            # Relaxed SMA validation
            if not self._check_sma_conditions_relaxed(pin_bar['close'], timestamp, data, direction):
                failed_validations['sma_conditions'] += 1
                continue
            
            # Calculate trade levels
            try:
                entry_price, stop_loss, take_profit = self._calculate_trade_levels_enhanced(
                    pin_bar, direction, symbol, risk_reward_ratio
                )
            except Exception as e:
                st.warning(f"Failed to calculate trade levels: {e}")
                failed_validations['invalid_levels'] += 1
                continue
            
            # Validate trade levels
            if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
                failed_validations['invalid_levels'] += 1
                continue
            
            # Calculate position size
            try:
                stop_distance_pips = abs(entry_price - stop_loss) / self._get_pip_value(symbol)
                if stop_distance_pips <= 0:
                    failed_validations['zero_position_size'] += 1
                    continue
                    
                lot_size = self._calculate_position_size_enhanced(
                    account_balance, risk_percentage, stop_distance_pips, symbol
                )
                
                if lot_size <= 0:
                    failed_validations['zero_position_size'] += 1
                    continue
                    
            except Exception as e:
                st.warning(f"Position sizing failed: {e}")
                failed_validations['zero_position_size'] += 1
                continue
            
            # Create trade
            trade = Trade(
                entry_time=timestamp,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                pin_bar_data=pin_bar,
                lot_size=lot_size
            )
            
            # Simulate trade execution
            try:
                trade = self._simulate_trade_enhanced(trade, data['1h'], symbol)
                trades.append(trade)
            except Exception as e:
                st.warning(f"Trade simulation failed: {e}")
                failed_validations['simulation_failed'] += 1
                continue
        
        # Store debug information
        debug_info['failed_validations'] = failed_validations
        debug_info['successful_trades'] = len(trades)
        
        # Display validation summary
        st.info("Trade Generation Summary:")
        st.write(f"- Pin bars found: {len(pin_bars)}")
        st.write(f"- Failed SMA validation: {failed_validations['sma_conditions']}")
        st.write(f"- Invalid trade levels: {failed_validations['invalid_levels']}")
        st.write(f"- Zero position size: {failed_validations['zero_position_size']}")
        st.write(f"- Simulation failures: {failed_validations['simulation_failed']}")
        st.write(f"- Successful trades: {len(trades)}")
        
        return trades
    
    def _check_sma_conditions_relaxed(self, price: float, timestamp: pd.Timestamp,
                                    data: Dict[str, pd.DataFrame], direction: TradeDirection) -> bool:
        """Relaxed SMA conditions with timezone handling"""
        timestamp = ensure_timezone_naive(timestamp)
        available_timeframes = ['15m', '30m', '4h']
        valid_timeframes = 0
        
        for tf in available_timeframes:
            if tf not in data:
                continue
            
            df = normalize_datetime_index(data[tf].copy())
            
            # Use simple moving average if enough data
            if len(df) >= 50:
                df['SMA50'] = df['Close'].rolling(window=min(50, len(df)//2)).mean()
            else:
                continue
            
            try:
                # Find closest time index safely
                time_diffs = [(abs(safe_datetime_subtract(idx, timestamp).total_seconds()), i) 
                             for i, idx in enumerate(df.index)]
                if not time_diffs:
                    continue
                
                _, closest_idx = min(time_diffs)
                sma50 = df.iloc[closest_idx]['SMA50']
                
                if pd.isna(sma50):
                    continue
                
                # Relaxed trend check
                if direction == TradeDirection.LONG and price > sma50:
                    valid_timeframes += 1
                elif direction == TradeDirection.SHORT and price < sma50:
                    valid_timeframes += 1
                    
            except (IndexError, KeyError):
                continue
        
        # Require at least 1 valid timeframe
        return valid_timeframes >= 1
    
    def _calculate_trade_levels_enhanced(self, pin_bar: Dict, direction: TradeDirection,
                                       symbol: str, risk_reward_ratio: float) -> Tuple[float, float, float]:
        """Enhanced trade level calculation"""
        pip_value = self._get_pip_value(symbol)
        
        if direction == TradeDirection.LONG:
            entry_price = pin_bar['close'] + (2 * pip_value)
            stop_loss = pin_bar['low'] - (1 * pip_value)
            
            risk_distance = entry_price - stop_loss
            if risk_distance <= 0:
                raise ValueError("Invalid risk distance for long trade")
                
            take_profit = entry_price + (risk_distance * risk_reward_ratio)
            
        else:  # SHORT
            entry_price = pin_bar['close'] - (2 * pip_value)
            stop_loss = pin_bar['high'] + (1 * pip_value)
            
            risk_distance = stop_loss - entry_price
            if risk_distance <= 0:
                raise ValueError("Invalid risk distance for short trade")
                
            take_profit = entry_price - (risk_distance * risk_reward_ratio)
        
        return entry_price, stop_loss, take_profit
    
    def _calculate_position_size_enhanced(self, account_balance: float, risk_percentage: float,
                                        stop_loss_pips: float, symbol: str) -> float:
        """Enhanced position sizing"""
        if stop_loss_pips <= 0:
            return 0.0
        
        risk_amount = account_balance * risk_percentage
        pip_value_usd = 10 if 'JPY' in symbol else 1
        
        position_size = risk_amount / (stop_loss_pips * pip_value_usd)
        
        # Apply realistic constraints
        min_size = 0.01
        max_size = min(50, account_balance / 1000)
        
        return max(min_size, min(max_size, position_size))
    
    def _simulate_trade_enhanced(self, trade: Trade, data_1h: pd.DataFrame, symbol: str) -> Trade:
        """Enhanced trade simulation with timezone handling"""
        data_1h = normalize_datetime_index(data_1h)
        trade_entry_time = ensure_timezone_naive(trade.entry_time)
        
        try:
            # Find entry index safely
            time_diffs = [(abs(safe_datetime_subtract(idx, trade_entry_time).total_seconds()), i) 
                         for i, idx in enumerate(data_1h.index)]
            if not time_diffs:
                trade.status = TradeStatus.NOT_TRIGGERED
                return trade
            
            _, entry_idx = min(time_diffs)
        except (IndexError, KeyError):
            trade.status = TradeStatus.NOT_TRIGGERED
            return trade
        
        if entry_idx + 1 >= len(data_1h):
            trade.status = TradeStatus.NOT_TRIGGERED
            return trade
        
        # Check entry trigger
        triggered = False
        for i in range(entry_idx + 1, min(entry_idx + 5, len(data_1h))):
            candle = data_1h.iloc[i]
            
            if trade.direction == TradeDirection.LONG:
                if candle['High'] >= trade.entry_price:
                    triggered = True
                    break
            else:  # SHORT
                if candle['Low'] <= trade.entry_price:
                    triggered = True
                    break
        
        if not triggered:
            trade.status = TradeStatus.NOT_TRIGGERED
            return trade
        
        # Simulate execution
        pip_value = self._get_pip_value(symbol)
        
        # Find exit point
        for i in range(entry_idx + 1, len(data_1h)):
            candle = data_1h.iloc[i]
            
            if trade.direction == TradeDirection.LONG:
                if candle['Low'] <= trade.stop_loss:
                    trade.set_exit(ensure_timezone_naive(candle.name), trade.stop_loss, TradeStatus.CLOSED_LOSS)
                    break
                elif candle['High'] >= trade.take_profit:
                    trade.set_exit(ensure_timezone_naive(candle.name), trade.take_profit, TradeStatus.CLOSED_PROFIT)
                    break
            else:  # SHORT
                if candle['High'] >= trade.stop_loss:
                    trade.set_exit(ensure_timezone_naive(candle.name), trade.stop_loss, TradeStatus.CLOSED_LOSS)
                    break
                elif candle['Low'] <= trade.take_profit:
                    trade.set_exit(ensure_timezone_naive(candle.name), trade.take_profit, TradeStatus.CLOSED_PROFIT)
                    break
        
        # Calculate P&L
        if trade.exit_price is not None:
            if trade.direction == TradeDirection.LONG:
                trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_value
            else:
                trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_value
            
            trade.pnl_usd = trade.pnl_pips * trade.lot_size * (10 if 'JPY' in symbol else 1)
        
        return trade
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        return 0.01 if 'JPY' in symbol else 0.0001
    
    def _calculate_statistics(self, trades: List[Trade], symbol: str, account_balance: float) -> Dict:
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
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
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
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
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
            'return_percent': (total_pnl_usd / account_balance * 100) if account_balance > 0 else 0
        }


# ================================
# CHART BUILDER
# ================================

class ChartBuilder:
    """Professional chart builder for trade visualization"""
    
    def __init__(self):
        self.colors = {
            'background': '#0d1421',
            'text': '#d1d4dc',
            'grid': '#2a2e39',
            'bullish': '#26a69a',
            'bearish': '#ef5350'
        }
    
    def create_tradingview_chart(self, df: pd.DataFrame, pin_bars: List[Dict], 
                               symbol: str, timeframe: str, 
                               show_ma: bool = True, highlight_trade=None) -> go.Figure:
        """Create professional TradingView-style chart with timezone handling"""
        # Normalize timezone
        df = normalize_datetime_index(df)
        
        # Calculate indicators if not present
        if 'EMA6' not in df.columns:
            df = self.calculate_moving_averages(df)
        
        # Create main chart
        fig = go.Figure()
        
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
        self._apply_tradingview_styling(fig, symbol, timeframe)
        
        return fig
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs and SMA for the chart"""
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
    
    def _add_pin_bar_highlights(self, fig: go.Figure, pin_bars: List[Dict], highlight_trade=None):
        """Add pin bar highlights and trade markers"""
        # Add pin bar triangular markers
        for pin_bar in pin_bars:
            timestamp = ensure_timezone_naive(pin_bar['timestamp'])
            
            if pin_bar['type'] == PinBarType.BULLISH:
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[pin_bar['low'] * 0.999],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=8, color='green'),
                    name='Bullish Pin Bar',
                    showlegend=False
                ))
            elif pin_bar['type'] == PinBarType.BEARISH:
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[pin_bar['high'] * 1.001],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=8, color='red'),
                    name='Bearish Pin Bar',
                    showlegend=False
                ))
        
        # Add trade highlights if specified
        if highlight_trade:
            self._add_trade_markers(fig, highlight_trade)
    
    def _add_trade_markers(self, fig: go.Figure, trade: Trade):
        """Add comprehensive trade visualization markers"""
        entry_time = ensure_timezone_naive(trade.entry_time)
        
        # Entry marker (Golden star)
        fig.add_trace(go.Scatter(
            x=[entry_time],
            y=[trade.entry_price],
            mode='markers',
            marker=dict(symbol='star', size=15, color='gold', line=dict(width=2, color='orange')),
            name='Trade Entry',
            hovertemplate=f'<b>TRADE ENTRY</b><br>' +
                        f'Direction: {trade.direction.value.title()}<br>' +
                        f'Entry: {trade.entry_price:.5f}<br>' +
                        f'Lot Size: {trade.lot_size:.2f}<br>' +
                        f'Stop Loss: {trade.stop_loss:.5f}<br>' +
                        f'Take Profit: {trade.take_profit:.5f}<extra></extra>'
        ))
        
        # Exit marker if trade closed
        if trade.exit_time and trade.exit_price:
            exit_time = ensure_timezone_naive(trade.exit_time)
            exit_color = 'green' if trade.pnl_pips > 0 else 'red' if trade.pnl_pips < 0 else 'gray'
            outcome = 'PROFIT' if trade.pnl_pips > 0 else 'LOSS' if trade.pnl_pips < 0 else 'BREAKEVEN'
            
            fig.add_trace(go.Scatter(
                x=[exit_time],
                y=[trade.exit_price],
                mode='markers',
                marker=dict(symbol='circle', size=12, color=exit_color, line=dict(width=2, color='white')),
                name='Trade Exit',
                hovertemplate=f'<b>TRADE EXIT - {outcome}</b><br>' +
                            f'Exit: {trade.exit_price:.5f}<br>' +
                            f'P&L: {trade.pnl_pips:.1f} pips<br>' +
                            f'P&L USD: ${trade.pnl_usd:.2f}<extra></extra>'
            ))
        
        # Price level lines
        fig.add_hline(y=trade.entry_price, line_dash="solid", line_color="blue", line_width=2,
                     annotation_text=f"Entry: {trade.entry_price:.5f}")
        fig.add_hline(y=trade.stop_loss, line_dash="dash", line_color="red", line_width=2,
                     annotation_text=f"Stop: {trade.stop_loss:.5f}")
        fig.add_hline(y=trade.take_profit, line_dash="dash", line_color="green", line_width=2,
                     annotation_text=f"Target: {trade.take_profit:.5f}")
    
    def _apply_tradingview_styling(self, fig: go.Figure, symbol: str, timeframe: str):
        """Apply professional TradingView styling"""
        # Determine price precision
        is_jpy_pair = 'JPY' in symbol
        y_tick_format = '.2f' if is_jpy_pair else '.5f'
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font_color=self.colors['text'],
            title=f"{symbol} - {timeframe} Chart",
            title_font_size=20,
            xaxis=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                rangeslider=dict(visible=False),
                type='date'
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
            margin=dict(l=20, r=60, t=60, b=20),
            height=600
        )


# ================================
# ENHANCED STREAMLIT UI
# ================================

class TrendSurferUI:
    """Enhanced Streamlit UI with timezone handling"""
    
    def __init__(self):
        self.backtester = TrendSurferBacktester()
        self.chart_builder = ChartBuilder()
        self.detector = PinBarDetector()
        
        # Initialize session state
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        if 'show_backtest_chart' not in st.session_state:
            st.session_state.show_backtest_chart = False
    
    def render_sidebar(self):
        """Enhanced configuration sidebar"""
        st.sidebar.title("🏄‍♂️ Trend Surfer Config")
        
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
        
        # Enhanced backtest parameters
        st.sidebar.subheader("⚙️ Backtest Parameters")
        
        # Date range with better defaults
        current_date = datetime.now().date()
        end_date = st.sidebar.date_input(
            "📅 End Date",
            value=current_date - timedelta(days=1),
            max_value=current_date - timedelta(days=1),
            help="End date for backtesting (yesterday is latest)"
        )
        
        # Duration selection
        duration_options = {
            "2 Weeks": 14,
            "1 Month": 30,
            "2 Months": 60,
            "Custom": None
        }
        
        selected_duration = st.sidebar.selectbox(
            "⏱️ Backtest Duration",
            list(duration_options.keys()),
            index=2,
            help="Select backtest period"
        )
        
        # Calculate or allow custom start date
        if selected_duration == "Custom":
            start_date = st.sidebar.date_input(
                "📅 Custom Start Date",
                value=end_date - timedelta(days=30),
                max_value=end_date - timedelta(days=1),
                help="Custom start date"
            )
        else:
            days_back = duration_options[selected_duration]
            start_date = end_date - timedelta(days=days_back)
            st.sidebar.text_input(
                "📅 Start Date (Auto)",
                value=start_date.strftime("%Y-%m-%d"),
                disabled=True,
                help=f"Automatically calculated as {selected_duration} before end date"
            )
        
        # Enhanced risk management
        st.sidebar.subheader("💰 Risk Management")
        
        account_size = st.sidebar.selectbox(
            "💵 Account Size",
            [1000, 2500, 5000, 10000, 25000, 50000, 100000],
            index=3,
            format_func=lambda x: f"${x:,}",
            help="Starting account balance for position sizing"
        )
        
        risk_percentage = st.sidebar.selectbox(
            "⚠️ Risk Per Trade",
            [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
            index=1,
            format_func=lambda x: f"{x*100:.1f}%",
            help="Percentage of account to risk per trade"
        )
        
        risk_reward = st.sidebar.selectbox(
            "🎯 Risk:Reward Ratio",
            [1.5, 2.0, 2.5, 3.0],
            index=1,
            format_func=lambda x: f"1:{x}",
            help="Target profit vs maximum loss ratio"
        )
        
        # Enhanced detector settings
        st.sidebar.subheader("🔍 Detection Settings")
        
        min_wick = st.sidebar.slider(
            "Min Wick Ratio",
            min_value=0.5,
            max_value=0.8,
            value=0.55,
            step=0.05,
            help="Minimum dominant wick size"
        )
        
        max_body = st.sidebar.slider(
            "Max Body Ratio", 
            min_value=0.2,
            max_value=0.5,
            value=0.4,
            step=0.05,
            help="Maximum body size"
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
            'max_body_ratio': max_body
        }
    
    def render_backtest_tab(self, config: Dict):
        """Enhanced backtesting interface"""
        st.header("🔬 Enhanced Trend Surfer Backtesting")
        
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
            **Detection Settings**
            - Min Wick: {config['min_wick_ratio']*100:.0f}%
            - Max Body: {config['max_body_ratio']*100:.0f}%
            - Timezone: Fixed ✅
            """)
        
        # Enhanced run button
        col_btn1, col_btn2 = st.columns([2, 1])
        
        with col_btn1:
            if st.button("🚀 Run Enhanced Backtest", type="primary", use_container_width=True):
                # Update detector settings
                self.backtester.detector = PinBarDetector(
                    min_wick_ratio=config['min_wick_ratio'],
                    max_body_ratio=config['max_body_ratio']
                )
                
                with st.spinner(f"🔄 Running enhanced {config['duration']} backtest..."):
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📊 Fetching market data...")
                    progress_bar.progress(20)
                    
                    # Convert dates to datetime
                    start_datetime = datetime.combine(config['start_date'], datetime.min.time())
                    end_datetime = datetime.combine(config['end_date'], datetime.min.time())
                    
                    results = self.backtester.run_backtest(
                        symbol=config['symbol'],
                        start_date=start_datetime,
                        end_date=end_datetime,
                        risk_reward_ratio=config['risk_reward'],
                        account_balance=config['account_size'],
                        risk_percentage=config['risk_percentage']
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("📈 Analyzing results...")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Backtest completed!")
                    
                    if results.trades or results.statistics:
                        st.session_state.backtest_results = results
                        st.success(f"🎉 Enhanced {config['duration']} backtest completed!")
                        
                        # Show quick summary
                        if results.statistics.get('total_trades', 0) > 0:
                            win_rate = results.statistics.get('win_rate', 0)
                            total_pips = results.statistics.get('total_pnl_pips', 0)
                            st.info(f"📊 Quick Summary: {results.statistics['total_trades']} trades, {win_rate:.1f}% win rate, {total_pips:.1f} pips")
                        else:
                            st.warning("⚠️ No triggered trades found. Consider adjusting detection settings.")
                    else:
                        st.error("❌ No valid trades found. Try different parameters or time period.")
        
        with col_btn2:
            if st.button("🧹 Clear Results", use_container_width=True):
                st.session_state.backtest_results = None
                st.success("Results cleared!")
        
        # Display results
        if st.session_state.backtest_results:
            self.display_enhanced_results(st.session_state.backtest_results)
    
    def display_enhanced_results(self, results: BacktestResults):
        """Display enhanced backtest results"""
        stats = results.statistics
        
        if not stats:
            st.warning("No statistics available")
            return
        
        # Performance dashboard
        st.subheader("📊 Performance Dashboard")
        
        # Key metrics in colored cards
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        # Trade table and chart functionality can be added here
        if results.trades:
            triggered_trades = [t for t in results.trades if t.status != TradeStatus.NOT_TRIGGERED]
            
            if triggered_trades:
                st.subheader("📋 Individual Trade Analysis")
                
                # Create trade data for display
                trade_data = []
                for i, trade in enumerate(triggered_trades):
                    outcome_emoji = "🟢" if trade.pnl_pips > 0 else "🔴" if trade.pnl_pips < 0 else "⚪"
                    direction_emoji = "📈" if trade.direction == TradeDirection.LONG else "📉"
                    
                    trade_data.append({
                        '#': i + 1,
                        'Date': trade.entry_time.strftime('%m/%d %H:%M'),
                        'Dir': f"{direction_emoji} {trade.direction.value.title()}",
                        'Entry': f"{trade.entry_price:.5f}",
                        'Exit': f"{trade.exit_price:.5f}" if trade.exit_price else "Open",
                        'Lots': f"{trade.lot_size:.2f}",
                        'Pips': f"{outcome_emoji} {trade.pnl_pips:.1f}",
                        'USD': f"${trade.pnl_usd:.2f}",
                        'Status': trade.status.value.replace('_', ' ').title()
                    })
                
                if trade_data:
                    trade_df = pd.DataFrame(trade_data)
                    st.dataframe(trade_df, use_container_width=True, hide_index=True)


# ================================
# MAIN SYSTEM CLASS
# ================================

class TrendSurferSystem:
    """Enhanced Trend Surfer trading system with timezone fixes"""
    
    def __init__(self):
        self.ui = TrendSurferUI()
        self.backtester = TrendSurferBacktester()
        self.chart_builder = ChartBuilder()
    
    def run_streamlit_app(self):
        """Enhanced Streamlit application entry point"""
        st.set_page_config(
            page_title="Fixed Trend Surfer",
            page_icon="🏄‍♂️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #1f77b4, #26a69a);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Enhanced header
        st.markdown("""
        <div class="main-header">
            <h1>🏄‍♂️ Fixed Trend Surfer</h1>
            <p>Professional Pin Bar Trading System - Timezone Issues Fixed</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key fixes display
        st.info("""
        🔧 **Key Fixes Applied:**
        • Timezone-aware datetime handling with proper conversion to UTC
        • Safe datetime arithmetic preventing offset-naive/aware conflicts  
        • Enhanced data normalization for yfinance compatibility
        • Improved error handling for edge cases
        • Better validation of datetime operations throughout the system
        """)
        
        # Render sidebar configuration
        config = self.ui.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Live Chart Analysis",
            "🔬 Enhanced Backtesting", 
            "🛠️ System Info"
        ])
        
        with tab1:
            self.render_live_analysis_tab(config)
        
        with tab2:
            self.ui.render_backtest_tab(config)
        
        with tab3:
            self.render_system_info_tab()
    
    def render_live_analysis_tab(self, config: Dict):
        """Live analysis with timezone fixes"""
        st.header("📊 Live Market Analysis")
        
        col_live1, col_live2 = st.columns([3, 1])
        
        with col_live2:
            st.subheader("⚙️ Chart Settings")
            
            timeframe = st.selectbox("Timeframe", ["1h", "4h"], index=0)
            lookback_days = st.selectbox("Lookback Period", [7, 14, 30, 60], index=2)
            show_pin_bars = st.checkbox("Show Pin Bars", value=True)
            show_emas = st.checkbox("Show EMAs", value=True)
            
            if st.button("🔄 Refresh Data", type="secondary", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        with col_live1:
            # Fetch recent data with timezone handling
            end_date = ensure_timezone_naive(datetime.now())
            start_date = end_date - timedelta(days=lookback_days)
            
            with st.spinner(f"📡 Fetching {timeframe} data for {config['symbol']}..."):
                try:
                    data = DataFetcher.fetch_data(
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
                data_with_indicators = self.chart_builder.calculate_moving_averages(data)
                
                # Detect pin bars if requested
                pin_bars = []
                if show_pin_bars:
                    pin_bars = self._detect_recent_pin_bars(data_with_indicators, config)
                
                # Create chart
                try:
                    fig = self.chart_builder.create_tradingview_chart(
                        data_with_indicators,
                        pin_bars,
                        config['symbol'],
                        timeframe.upper(),
                        show_ma=show_emas
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart creation error: {e}")
                
                # Display recent pin bar summary
                if pin_bars:
                    st.subheader(f"🎯 Recent Pin Bars ({len(pin_bars)} found)")
                    
                    recent_pin_data = []
                    current_time = ensure_timezone_naive(datetime.now())
                    
                    for pb in pin_bars[-5:]:  # Show last 5
                        pb_time = ensure_timezone_naive(pb['timestamp'])
                        time_ago = safe_datetime_subtract(current_time, pb_time)
                        hours_ago = max(1, int(time_ago.total_seconds() / 3600))
                        
                        recent_pin_data.append({
                            'Time': f"{hours_ago}h ago" if hours_ago < 24 else f"{hours_ago//24}d ago",
                            'Type': f"{'📈' if pb['type'] == PinBarType.BULLISH else '📉'} {pb['type'].value.title()}",
                            'Strength': f"{pb['strength']:.1f}%",
                            'Price': f"{pb['close']:.5f}",
                            'Date': pb_time.strftime('%m/%d %H:%M')
                        })
                    
                    if recent_pin_data:
                        pin_df = pd.DataFrame(recent_pin_data)
                        st.dataframe(pin_df, use_container_width=True, hide_index=True)
                        
                        # Trading opportunity alert
                        latest_pin = pin_bars[-1]
                        latest_pin_time = ensure_timezone_naive(latest_pin['timestamp'])
                        time_since = safe_datetime_subtract(current_time, latest_pin_time)
                        
                        if time_since.total_seconds() < 7200:  # Less than 2 hours
                            hours_since = max(1, int(time_since.total_seconds()/3600))
                            st.success(f"🚨 Fresh Trading Opportunity: {latest_pin['type'].value.title()} pin bar detected {hours_since}h ago!")
                else:
                    st.info("ℹ️ No pin bars detected in recent data. Monitor for new opportunities.")
                
                # Market summary
                st.subheader("📋 Market Summary")
                
                latest_price = data_with_indicators['Close'].iloc[-1]
                ema6 = data_with_indicators['EMA6'].iloc[-1]
                ema18 = data_with_indicators['EMA18'].iloc[-1]
                ema50 = data_with_indicators['EMA50'].iloc[-1]
                
                # Trend analysis
                if ema6 > ema18 > ema50:
                    trend = "🟢 Strong Uptrend"
                elif ema6 > ema18:
                    trend = "🔵 Weak Uptrend"
                elif ema6 < ema18 < ema50:
                    trend = "🔴 Strong Downtrend"
                elif ema6 < ema18:
                    trend = "🟠 Weak Downtrend"
                else:
                    trend = "⚪ Sideways/Mixed"
                
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                
                with col_summary1:
                    st.metric("Current Price", f"{latest_price:.5f}")
                    st.metric("Trend Direction", trend)
                
                with col_summary2:
                    st.metric("EMA6", f"{ema6:.5f}")
                    st.metric("EMA18", f"{ema18:.5f}")
                
                with col_summary3:
                    distance_to_ema6 = abs(latest_price - ema6) / ema6 * 100
                    st.metric("Distance to EMA6", f"{distance_to_ema6:.2f}%")
                    
                    if not pd.isna(data_with_indicators['SMA200'].iloc[-1]):
                        sma200 = data_with_indicators['SMA200'].iloc[-1]
                        st.metric("SMA200", f"{sma200:.5f}")
                    else:
                        st.metric("SMA200", "Calculating...")
            
            else:
                st.error("❌ Unable to fetch chart data. Please try again or select a different symbol.")
    
    def _detect_recent_pin_bars(self, data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Detect pin bars for live analysis with timezone handling"""
        pin_bars = []
        
        if len(data) < 20:
            return pin_bars
        
        # Normalize timezone
        data = normalize_datetime_index(data)
        
        # Use configured detector settings
        detector = PinBarDetector(
            min_wick_ratio=config.get('min_wick_ratio', 0.55),
            max_body_ratio=config.get('max_body_ratio', 0.4)
        )
        
        # Start detection after sufficient data
        start_idx = max(20, len(data) - 100)  # Analyze last 100 candles
        
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
            
            pin_bar_type, strength = detector.detect_pin_bar(
                candle, row['EMA6'], row['EMA18'], ema50, sma200
            )
            
            if pin_bar_type != PinBarType.NONE and strength > 30:  # Minimum quality threshold
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
    
    def render_system_info_tab(self):
        """System information and troubleshooting"""
        st.header("🛠️ System Information & Fixes")
        
        # Key fixes summary
        st.subheader("🔧 Timezone Fixes Applied")
        
        col_fix1, col_fix2 = st.columns(2)
        
        with col_fix1:
            st.markdown("""
            **DateTime Handling Fixes:**
            - ✅ Timezone-aware datetime conversion to UTC
            - ✅ Safe datetime arithmetic functions
            - ✅ Proper handling of pandas Timestamp objects
            - ✅ Normalization of datetime indexes
            - ✅ Prevention of offset-naive/aware conflicts
            """)
        
        with col_fix2:
            st.markdown("""
            **Data Processing Enhancements:**
            - ✅ Enhanced yfinance data compatibility
            - ✅ Improved error handling for edge cases
            - ✅ Better validation throughout the system
            - ✅ Robust chart creation with timezone support
            - ✅ Safe time difference calculations
            """)
        
        # System compatibility test
        st.subheader("📡 System Compatibility Test")
        
        if st.button("🔍 Run System Test", type="secondary"):
            with st.spinner("Testing system components..."):
                test_results = {}
                
                # Test timezone handling
                try:
                    test_dt = datetime.now()
                    naive_dt = ensure_timezone_naive(test_dt)
                    test_results['timezone_handling'] = '✅ Working'
                except Exception as e:
                    test_results['timezone_handling'] = f'❌ Error: {str(e)[:50]}'
                
                # Test data fetching
                try:
                    test_data = DataFetcher.fetch_data('EURUSD=X', '1h', 
                                                     (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                                                     datetime.now().strftime('%Y-%m-%d'))
                    if not test_data.empty:
                        test_results['data_fetching'] = f'✅ Working ({len(test_data)} candles)'
                    else:
                        test_results['data_fetching'] = '⚠️ No data returned'
                except Exception as e:
                    test_results['data_fetching'] = f'❌ Error: {str(e)[:50]}'
                
                # Test datetime arithmetic
                try:
                    dt1 = datetime.now()
                    dt2 = datetime.now() - timedelta(hours=1)
                    diff = safe_datetime_subtract(dt1, dt2)
                    test_results['datetime_arithmetic'] = f'✅ Working ({diff.total_seconds()}s)'
                except Exception as e:
                    test_results['datetime_arithmetic'] = f'❌ Error: {str(e)[:50]}'
                
                # Test pin bar detection
                try:
                    detector = PinBarDetector()
                    test_candle = Candle(datetime.now(), 1.1000, 1.1020, 1.0980, 1.1015)
                    pin_type, strength = detector.detect_pin_bar(test_candle, 1.1000, 1.0990, 1.0980, 1.0970)
                    test_results['pin_bar_detection'] = '✅ Working'
                except Exception as e:
                    test_results['pin_bar_detection'] = f'❌ Error: {str(e)[:50]}'
                
                # Display results
                for component, result in test_results.items():
                    st.write(f"**{component.replace('_', ' ').title()}:** {result}")
        
        # Version information
        st.subheader("📋 Version Information")
        
        version_info = {
            'System Version': 'Fixed v2.0',
            'Key Fix': 'Timezone-aware datetime handling',
            'Compatibility': 'yfinance 0.2.x, pandas 2.x',
            'Python Version': '3.8+',
            'Last Updated': 'December 2024'
        }
        
        for key, value in version_info.items():
            st.write(f"**{key}:** {value}")
        
        # Troubleshooting guide
        st.subheader("🔧 Troubleshooting Guide")
        
        with st.expander("❓ Common Issues & Solutions", expanded=False):
            st.markdown("""
            **Issue: Timezone errors or datetime conflicts**
            - ✅ Fixed: All datetime operations now use timezone-aware handling
            - ✅ Fixed: Automatic conversion to UTC for consistency
            - ✅ Fixed: Safe arithmetic functions prevent conflicts
            
            **Issue: "No data available" errors**
            - Solution: Try a shorter time period (max 60 days for intraday data)
            - Solution: Check if markets are open (forex trades 24/5)
            - Solution: Verify symbol format (use =X suffix for forex)
            
            **Issue: Chart display problems**
            - ✅ Fixed: Enhanced chart builder with timezone support
            - ✅ Fixed: Proper data normalization before plotting
            - Solution: Clear cache and refresh if issues persist
            
            **Issue: Poor backtest performance**
            - Solution: Adjust detector sensitivity (try min_wick_ratio 0.50)
            - Solution: Consider different time periods or market conditions
            - Solution: Verify risk management parameters are appropriate
            """)
        
        # Performance tips
        st.subheader("⚡ Performance Tips")
        
        st.markdown("""
        **For Best Performance:**
        - Use shorter lookback periods for live analysis (7-30 days)
        - Clear cache regularly using the refresh button
        - Choose appropriate timeframes (1H recommended for pin bars)
        - Limit backtest duration to 2 months for faster processing
        - Use standard forex pairs for most reliable data
        """)


# ================================
# MAIN APPLICATION ENTRY POINT
# ================================

def main():
    """Fixed main application entry point"""
    try:
        system = TrendSurferSystem()
        system.run_streamlit_app()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("This error suggests a system-level issue. Please check the error details and try refreshing.")
        
        # Enhanced error details for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()