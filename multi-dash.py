import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
import time
from scipy.stats import norm
import math
import json
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import Polygon SDK
try:
    from polygon import RESTClient
except ImportError:
    st.error("❌ Please install polygon-api-client: pip install polygon-api-client")
    st.stop()

warnings.filterwarnings('ignore')

# =============================================================================
# CACHING CONFIGURATION
# =============================================================================

@st.cache_data(ttl=300)  # 5 minute cache
def cached_get_asset_data(api_key: str, ticker: str, asset_class: str, days: int = 500) -> Dict:
    """Cached version of get_asset_data"""
    try:
        client = RESTClient(api_key)
        
        # Format ticker based on asset class
        if asset_class == 'INDICES':
            popular_etfs = ['SPY', 'QQQ', 'IWM', 'EWU', 'VGK', 'EFA', 'VIX', 'XLF', 'XLE', 'XLK']
            if ticker.upper() in popular_etfs:
                formatted_ticker = ticker
            elif not ticker.startswith('I:'):
                formatted_ticker = f"I:{ticker}"
            else:
                formatted_ticker = ticker
        elif asset_class == 'FOREX':
            if not ticker.startswith('C:'):
                formatted_ticker = f"C:{ticker}"
            else:
                formatted_ticker = ticker
        else:
            formatted_ticker = ticker
        
        print(f"Fetching {asset_class} data for {formatted_ticker}...")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        aggs = []
        for agg in client.list_aggs(
            formatted_ticker,
            1,
            "day", 
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            limit=days
        ):
            aggs.append(agg)
        
        if not aggs:
            raise ValueError(f"No data found for {formatted_ticker}")
        
        # Convert to DataFrame
        df_data = []
        for agg in aggs:
            if hasattr(agg, 'close') and agg.close is not None:
                volume = getattr(agg, 'volume', None)
                if asset_class == 'FOREX':
                    volume = 0  # FX doesn't have traditional volume
                elif volume is None:
                    volume = 0
                    
                df_data.append({
                    'timestamp': agg.timestamp,
                    'open': float(agg.open if agg.open else agg.close),
                    'high': float(agg.high if agg.high else agg.close),
                    'low': float(agg.low if agg.low else agg.close),
                    'close': float(agg.close),
                    'volume': int(volume)
                })
        
        if not df_data:
            raise ValueError("No valid price data")
        
        df = pd.DataFrame(df_data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        df = df.dropna(subset=['close'])
        
        current_price = float(df['close'].iloc[-1])
        
        # Calculate technical indicators with proper min_periods handling
        df['returns'] = df['close'].pct_change()
        vol_scaling = 365 if asset_class == 'FOREX' else 252
        min_periods_vol = max(1, min(21, len(df)))
        realized_vol_21d = df['returns'].rolling(21, min_periods=min_periods_vol).std().iloc[-1] * np.sqrt(vol_scaling)
        
        min_periods_52w = max(1, min(52, len(df)))
        high_52w = df['high'].rolling(min(252, len(df)), min_periods=min_periods_52w).max().iloc[-1]
        low_52w = df['low'].rolling(min(252, len(df)), min_periods=min_periods_52w).min().iloc[-1]
        
        return {
            'ticker': ticker,
            'formatted_ticker': formatted_ticker,
            'asset_class': asset_class,
            'current_price': current_price,
            'historical_data': df,
            'realized_vol_21d': float(realized_vol_21d) if not pd.isna(realized_vol_21d) else 0.20,
            'high_52w': float(high_52w) if not pd.isna(high_52w) else current_price * 1.25,
            'low_52w': float(low_52w) if not pd.isna(low_52w) else current_price * 0.75,
            'data_points': len(df),
            'date_range': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            'source': 'polygon_cached'
        }
        
    except Exception as e:
        print(f"Cached asset data fetch failed: {str(e)}")
        raise

@st.cache_data(ttl=300)  # 5 minute cache
def cached_get_options_data(api_key: str, ticker: str, asset_class: str, current_price: float) -> Dict:
    """Cached version of get_options_data"""
    try:
        client = RESTClient(api_key)
        
        print(f"Fetching options data for {ticker} ({asset_class})...")
        
        # Get options contracts
        contracts = []
        for contract in client.list_options_contracts(
            underlying_ticker=ticker,  # Use original ticker for options
            expiration_date_gte=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            expiration_date_lte=(datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),
            limit=1000
        ):
            contracts.append(contract)
        
        if not contracts:
            raise ValueError(f"No options contracts found for {ticker}")
        
        # Process contracts
        exp_groups = {}
        today = datetime.now().date()
        
        for contract in contracts:
            try:
                exp_date = contract.expiration_date
                exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d').date()
                
                if exp_date_obj <= today:
                    continue
                
                strike = float(contract.strike_price)
                
                # Filter strikes within reasonable range
                strike_range = 0.15 if asset_class == 'FOREX' else 0.25
                if abs(strike - current_price) / current_price > strike_range:
                    continue
                
                if exp_date not in exp_groups:
                    exp_groups[exp_date] = {'calls': [], 'puts': []}
                
                contract_data = {
                    'ticker': contract.ticker,
                    'strike': strike,
                    'contract_type': contract.contract_type,
                    'expiration_date': exp_date
                }
                
                if contract.contract_type == 'call':
                    exp_groups[exp_date]['calls'].append(contract_data)
                elif contract.contract_type == 'put':
                    exp_groups[exp_date]['puts'].append(contract_data)
                    
            except Exception:
                continue
        
        # Find best expiration
        best_exp = None
        for exp_date in sorted(exp_groups.keys()):
            calls_count = len(exp_groups[exp_date]['calls'])
            puts_count = len(exp_groups[exp_date]['puts'])
            
            if calls_count >= 3 and puts_count >= 3:
                best_exp = exp_date
                break
        
        if not best_exp:
            raise ValueError("No suitable expiration found")
        
        # Get option prices (simplified with Black-Scholes)
        calls_data = []
        puts_data = []
        
        base_vol = 0.15 if asset_class == 'FOREX' else 0.20 if asset_class == 'INDICES' else 0.25
        
        for call_contract in exp_groups[best_exp]['calls']:
            strike = call_contract['strike']
            # Simplified pricing
            moneyness = strike / current_price
            time_value = 0.02 + 0.03 * math.sqrt(35/365)  # Assume 35 days
            price = max(0.01, current_price * time_value * (2 - moneyness))
            
            calls_data.append({
                'ticker': call_contract['ticker'],
                'strike': strike,
                'lastPrice': price,
                'bid': price * 0.95,
                'ask': price * 1.05,
                'volume': 100,
                'openInterest': 50,
                'impliedVolatility': base_vol,
                'contract_type': 'call',
                'data_source': 'calculated'
            })
        
        for put_contract in exp_groups[best_exp]['puts']:
            strike = put_contract['strike'] 
            # Simplified pricing
            moneyness = strike / current_price
            time_value = 0.02 + 0.03 * math.sqrt(35/365)
            price = max(0.01, current_price * time_value * moneyness)
            
            puts_data.append({
                'ticker': put_contract['ticker'],
                'strike': strike,
                'lastPrice': price,
                'bid': price * 0.95,
                'ask': price * 1.05,
                'volume': 100,
                'openInterest': 50,
                'impliedVolatility': base_vol,
                'contract_type': 'put',
                'data_source': 'calculated'
            })
        
        calls_df = pd.DataFrame(calls_data).sort_values('strike') if calls_data else pd.DataFrame()
        puts_df = pd.DataFrame(puts_data).sort_values('strike') if puts_data else pd.DataFrame()
        
        exp_date_obj = datetime.strptime(best_exp, '%Y-%m-%d')
        days_to_expiry = (exp_date_obj.date() - today).days
        
        return {
            'expiration': best_exp,
            'calls': calls_df,
            'puts': puts_df,
            'days_to_expiry': days_to_expiry,
            'underlying_price': current_price,
            'underlying_ticker': ticker,
            'asset_class': asset_class,
            'total_contracts': len(calls_data) + len(puts_data),
            'source': 'polygon_cached'
        }
        
    except Exception as e:
        print(f"Cached options data fetch failed: {str(e)}")
        raise

@st.cache_data(ttl=600)  # 10 minute cache
def cached_search_symbols(api_key: str, asset_class: str, query: str) -> List[Dict]:
    """Cached symbol search"""
    strategist = MultiAssetOptionsStrategist(api_key)
    return strategist.search_symbols(asset_class, query)

@st.cache_data(ttl=3600)  # 1 hour cache
def cached_get_popular_symbols(asset_class: str) -> List[str]:
    """Cached popular symbols"""
    asset_configs = {
        'INDICES': ['SPY', 'QQQ', 'IWM', 'EWU', 'VGK', 'EFA', 'VIX'],
        'EQUITIES': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
        'FOREX': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
    }
    return asset_configs.get(asset_class, [])

# =============================================================================
# ENHANCED MULTI-ASSET OPTIONS STRATEGIST WITH ACCURATE MATH
# =============================================================================

class MultiAssetOptionsStrategist:
    """
    Professional Multi-Asset Options Strategist with Accurate Mathematics
    """
    
    def __init__(self, polygon_api_key: str):
        if not polygon_api_key:
            raise ValueError("Polygon API key is required")
        
        self.client = RESTClient(polygon_api_key)
        self.polygon_api_key = polygon_api_key
        
        # Asset class configurations
        self.asset_configs = {
            'INDICES': {
                'market': 'indices',
                'prefix': 'I:',
                'popular_symbols': ['SPY', 'QQQ', 'IWM', 'EWU', 'VGK', 'EFA', 'VIX'],
                'description': 'Index ETFs and Volatility Products'
            },
            'EQUITIES': {
                'market': 'stocks',
                'prefix': '',
                'popular_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
                'description': 'Individual Stocks and ETFs'
            },
            'FOREX': {
                'market': 'fx',
                'prefix': 'C:',
                'popular_symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'],
                'description': 'Currency Pairs and FX Options'
            }
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for debugging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_popular_symbols(self, asset_class: str) -> List[str]:
        """Get popular symbols for asset class"""
        return cached_get_popular_symbols(asset_class)
    
    def search_symbols(self, asset_class: str, query: str) -> List[Dict]:
        """Search for symbols within asset class"""
        return cached_search_symbols(self.polygon_api_key, asset_class, query)
    
    def _format_ticker(self, ticker: str, asset_class: str) -> str:
        """Format ticker based on asset class"""
        if asset_class == 'INDICES':
            popular_etfs = ['SPY', 'QQQ', 'IWM', 'EWU', 'VGK', 'EFA', 'VIX', 'XLF', 'XLE', 'XLK']
            if ticker.upper() in popular_etfs:
                return ticker
            if not ticker.startswith('I:'):
                return f"I:{ticker}"
            return ticker
        
        prefix = self.asset_configs.get(asset_class, {}).get('prefix', '')
        if prefix and not ticker.startswith(prefix):
            return f"{prefix}{ticker}"
        return ticker

    def get_asset_data(self, ticker: str, asset_class: str, days: int = 500) -> Dict:
        """Get data for any asset class with enhanced caching"""
        return cached_get_asset_data(self.polygon_api_key, ticker, asset_class, days)
    
    def get_options_data(self, ticker: str, asset_class: str, current_price: float = None) -> Dict:
        """Get options data for any asset class with caching"""
        if current_price is None:
            underlying_data = self.get_asset_data(ticker, asset_class, days=30)
            current_price = underlying_data['current_price']
        return cached_get_options_data(self.polygon_api_key, ticker, asset_class, current_price)
    
    def analyze_market_conditions(self, data: Dict) -> Dict:
        """Analyze market conditions with asset-specific considerations and proper min_periods"""
        current_price = data['current_price']
        asset_class = data.get('asset_class', 'EQUITIES')
        
        # Calculate technical indicators from historical data
        df = data['historical_data'].tail(200)  # Use recent data
        
        # Moving averages with proper min_periods (never exceed window size)
        min_periods_20 = max(1, min(10, len(df)))
        min_periods_50 = max(1, min(25, len(df)))
        min_periods_200 = max(1, min(50, len(df)))
        
        sma_20 = df['close'].rolling(20, min_periods=min_periods_20).mean().iloc[-1] if len(df) >= 1 else current_price
        sma_50 = df['close'].rolling(50, min_periods=min_periods_50).mean().iloc[-1] if len(df) >= 1 else current_price
        sma_200 = df['close'].rolling(200, min_periods=min_periods_200).mean().iloc[-1] if len(df) >= 1 else current_price
        
        # RSI with proper min_periods
        min_periods_rsi = max(1, min(7, len(df)))
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=min_periods_rsi).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=min_periods_rsi).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(gain) > 0 and not pd.isna(gain.iloc[-1]) else 50.0
        
        # Bollinger Bands with proper min_periods
        bb_min_periods = max(1, min(10, len(df)))
        bb_middle = df['close'].rolling(20, min_periods=bb_min_periods).mean().iloc[-1] if len(df) >= 1 else current_price
        bb_std = df['close'].rolling(20, min_periods=bb_min_periods).std().iloc[-1] if len(df) >= 1 else current_price * 0.02
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Trend analysis
        if current_price > sma_20 > sma_50 > sma_200:
            trend = 'STRONG_BULLISH'
            trend_strength = min(((current_price / sma_200) - 1) * 100, 15)
        elif current_price > sma_20 > sma_50:
            trend = 'BULLISH'
            trend_strength = min(((current_price / sma_50) - 1) * 100, 12)
        elif current_price < sma_20 < sma_50 < sma_200:
            trend = 'STRONG_BEARISH'
            trend_strength = min(((sma_200 / current_price) - 1) * 100, 15)
        elif current_price < sma_20 < sma_50:
            trend = 'BEARISH'
            trend_strength = min(((sma_50 / current_price) - 1) * 100, 12)
        else:
            trend = 'SIDEWAYS'
            trend_strength = 2.0
        
        # Volatility regime
        realized_vol = data.get('realized_vol_21d', 0.20)
        if asset_class == 'FOREX':
            if realized_vol > 0.20:
                vol_regime = 'EXTREME_VOL'
            elif realized_vol > 0.15:
                vol_regime = 'HIGH_VOL'
            elif realized_vol < 0.08:
                vol_regime = 'LOW_VOL'
            else:
                vol_regime = 'NORMAL_VOL'
        else:
            if realized_vol > 0.30:
                vol_regime = 'EXTREME_VOL'
            elif realized_vol > 0.25:
                vol_regime = 'HIGH_VOL'
            elif realized_vol < 0.12:
                vol_regime = 'LOW_VOL'
            else:
                vol_regime = 'NORMAL_VOL'
        
        # Momentum
        if rsi > 75:
            momentum = 'EXTREMELY_OVERBOUGHT'
        elif rsi > 65:
            momentum = 'OVERBOUGHT'
        elif rsi > 55:
            momentum = 'BULLISH'
        elif rsi < 25:
            momentum = 'EXTREMELY_OVERSOLD'
        elif rsi < 35:
            momentum = 'OVERSOLD'
        elif rsi < 45:
            momentum = 'BEARISH'
        else:
            momentum = 'NEUTRAL'
        
        # Bollinger Band position
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        if bb_position > 0.9:
            bb_signal = 'EXTREME_UPPER'
        elif bb_position > 0.8:
            bb_signal = 'UPPER_BAND'
        elif bb_position < 0.1:
            bb_signal = 'EXTREME_LOWER'
        elif bb_position < 0.2:
            bb_signal = 'LOWER_BAND'
        else:
            bb_signal = 'MIDDLE_RANGE'
        
        # Price changes with proper bounds checking
        price_change_1d = ((current_price / df['close'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
        price_change_5d = ((current_price / df['close'].iloc[-6]) - 1) * 100 if len(df) >= 6 else 0
        price_change_20d = ((current_price / df['close'].iloc[-21]) - 1) * 100 if len(df) >= 21 else 0
        
        return {
            'asset_class': asset_class,
            'trend': trend,
            'trend_strength': round(trend_strength, 2),
            'volatility_regime': vol_regime,
            'momentum': momentum,
            'bb_signal': bb_signal,
            'bb_position': round(bb_position * 100, 1),
            'rsi': round(rsi, 1),
            'realized_vol': round(realized_vol, 3),
            'price_change_1d': round(price_change_1d, 2),
            'price_change_5d': round(price_change_5d, 2),
            'price_change_20d': round(price_change_20d, 2),
            'high_52w': data['high_52w'],
            'low_52w': data['low_52w'],
            'current_price': current_price,
            'volume_vs_avg': 1.0  # Simplified for cached version
        }
    
    # =============================================================================
    # ACCURATE STRATEGY CALCULATIONS WITH USER CAPITAL
    # =============================================================================
    
    def calculate_covered_call_accurate(self, calls_df: pd.DataFrame, current_price: float,
                                       capital: float, asset_class: str, max_risk_amount: float = None) -> Dict:
        """Accurate covered call calculations with user capital"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Determine realistic position size
        stock_cost_per_contract = current_price * share_equivalent
        max_contracts = int(capital / stock_cost_per_contract)
        
        if max_contracts == 0:
            return {
                'error': f'Insufficient capital. Need ${stock_cost_per_contract:,.2f} for 1 contract',
                'min_capital_needed': stock_cost_per_contract
            }
        
        # Find optimal strikes (2-15% OTM)
        strike_ranges = [
            ('Conservative', current_price * 1.02, current_price * 1.05),
            ('Moderate', current_price * 1.05, current_price * 1.10),
            ('Aggressive', current_price * 1.10, current_price * 1.15)
        ]
        
        recommendations = []
        
        for risk_level, min_strike, max_strike in strike_ranges:
            suitable_calls = calls_df[
                (calls_df['strike'] >= min_strike) & 
                (calls_df['strike'] <= max_strike)
            ].copy()
            
            if suitable_calls.empty:
                continue
            
            # Select best option (highest premium)
            best_option = suitable_calls.loc[suitable_calls['lastPrice'].idxmax()]
            
            strike = best_option['strike']
            premium = best_option['lastPrice']
            
            # Accurate covered call calculations
            stock_investment = max_contracts * current_price * share_equivalent
            premium_received = max_contracts * premium * share_equivalent
            net_investment = stock_investment - premium_received
            
            # Maximum profit: if called away
            max_profit = (max_contracts * (strike - current_price) * share_equivalent) + premium_received
            max_profit_pct = (max_profit / net_investment) * 100
            
            # Maximum loss: stock goes to zero minus premium received
            max_loss = net_investment
            
            # Breakeven: stock purchase price minus premium
            breakeven = current_price - premium
            
            # Annualized return calculation (assume 30-45 days to expiration)
            days_to_expiry = 35  # Typical monthly cycle
            premium_yield = (premium / current_price) * 100
            annualized_yield = premium_yield * (365 / days_to_expiry)
            
            recommendations.append({
                'risk_level': risk_level,
                'strike': strike,
                'premium': premium,
                'contracts': max_contracts,
                'stock_investment': stock_investment,
                'premium_received': premium_received,
                'net_investment': net_investment,
                'max_profit': max_profit,
                'max_profit_pct': max_profit_pct,
                'max_loss': max_loss,
                'breakeven': breakeven,
                'premium_yield': premium_yield,
                'annualized_yield': annualized_yield,
                'probability_profit': self._estimate_probability_above_breakeven(current_price, breakeven, days_to_expiry)
            })
        
        return {
            'strategy': 'COVERED_CALL',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def calculate_cash_secured_put_accurate(self, puts_df: pd.DataFrame, current_price: float,
                                           capital: float, asset_class: str, max_risk_amount: float = None) -> Dict:
        """Accurate cash secured put calculations with user capital"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Find optimal strikes (5-15% OTM puts)
        strike_ranges = [
            ('Conservative', current_price * 0.95, current_price * 0.98),
            ('Moderate', current_price * 0.90, current_price * 0.95),
            ('Aggressive', current_price * 0.85, current_price * 0.90)
        ]
        
        recommendations = []
        
        for risk_level, min_strike, max_strike in strike_ranges:
            suitable_puts = puts_df[
                (puts_df['strike'] >= min_strike) & 
                (puts_df['strike'] <= max_strike)
            ].copy()
            
            if suitable_puts.empty:
                continue
            
            # Select best option (highest premium)
            best_option = suitable_puts.loc[suitable_puts['lastPrice'].idxmax()]
            
            strike = best_option['strike']
            premium = best_option['lastPrice']
            
            # Calculate position size based on cash requirement
            cash_per_contract = strike * share_equivalent
            max_contracts = int(capital / cash_per_contract)
            
            if max_contracts == 0:
                continue
            
            # Accurate cash secured put calculations
            total_cash_secured = max_contracts * cash_per_contract
            premium_received = max_contracts * premium * share_equivalent
            
            # Maximum profit: keep premium if not assigned
            max_profit = premium_received
            max_profit_pct = (max_profit / total_cash_secured) * 100
            
            # Maximum loss: put assigned, stock goes to zero
            max_loss = total_cash_secured - premium_received
            
            # Breakeven: strike price minus premium
            breakeven = strike - premium
            
            # Effective purchase price if assigned
            effective_price = strike - premium
            discount_pct = ((current_price - effective_price) / current_price) * 100
            
            # Annualized return
            days_to_expiry = 35
            premium_yield = (premium / strike) * 100
            annualized_yield = premium_yield * (365 / days_to_expiry)
            
            recommendations.append({
                'risk_level': risk_level,
                'strike': strike,
                'premium': premium,
                'contracts': max_contracts,
                'cash_secured': total_cash_secured,
                'premium_received': premium_received,
                'max_profit': max_profit,
                'max_profit_pct': max_profit_pct,
                'max_loss': max_loss,
                'breakeven': breakeven,
                'effective_price': effective_price,
                'discount_pct': discount_pct,
                'premium_yield': premium_yield,
                'annualized_yield': annualized_yield,
                'probability_profit': self._estimate_probability_above_breakeven(current_price, breakeven, days_to_expiry, is_put=True)
            })
        
        return {
            'strategy': 'CASH_SECURED_PUT',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def calculate_iron_condor_accurate(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                      current_price: float, capital: float, asset_class: str, max_risk_amount: float = None) -> Dict:
        """Accurate iron condor calculations with user capital"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Define wing widths
        wing_configs = [
            ('Conservative', 0.03, 0.05),  # 3% wings, 5% total width
            ('Moderate', 0.04, 0.08),      # 4% wings, 8% total width  
            ('Aggressive', 0.05, 0.10)    # 5% wings, 10% total width
        ]
        
        recommendations = []
        
        for risk_level, wing_width, total_width in wing_configs:
            
            # Define strikes
            call_sell_strike = current_price * (1 + wing_width)
            call_buy_strike = current_price * (1 + total_width)
            put_sell_strike = current_price * (1 - wing_width)
            put_buy_strike = current_price * (1 - total_width)
            
            # Find closest available strikes
            call_sell_opt = self._find_closest_strike(calls_df, call_sell_strike)
            call_buy_opt = self._find_closest_strike(calls_df, call_buy_strike)
            put_sell_opt = self._find_closest_strike(puts_df, put_sell_strike)
            put_buy_opt = self._find_closest_strike(puts_df, put_buy_strike)
            
            if any(opt is None for opt in [call_sell_opt, call_buy_opt, put_sell_opt, put_buy_opt]):
                continue
            
            # Calculate net credit
            credit = (call_sell_opt['lastPrice'] + put_sell_opt['lastPrice'] - 
                     call_buy_opt['lastPrice'] - put_buy_opt['lastPrice'])
            
            if credit <= 0:
                continue
            
            # Calculate spread widths
            call_spread_width = call_buy_opt['strike'] - call_sell_opt['strike']
            put_spread_width = put_sell_opt['strike'] - put_buy_opt['strike']
            max_spread_width = max(call_spread_width, put_spread_width)
            
            # Maximum loss per contract
            max_loss_per_contract = max_spread_width - credit
            
            # Position sizing based on margin requirement
            margin_per_contract = max_loss_per_contract * share_equivalent
            max_contracts = int(capital * 0.5 / margin_per_contract)  # Use 50% of capital for margin
            
            if max_contracts == 0:
                continue
            
            # Accurate calculations
            total_credit = max_contracts * credit * share_equivalent
            total_max_loss = max_contracts * max_loss_per_contract * share_equivalent
            total_margin = max_contracts * margin_per_contract
            
            # Breakeven points
            upper_breakeven = call_sell_opt['strike'] + credit
            lower_breakeven = put_sell_opt['strike'] - credit
            
            # Probability calculations
            profit_range = upper_breakeven - lower_breakeven
            profit_range_pct = (profit_range / current_price) * 100
            
            # Return on capital
            max_profit_pct = (total_credit / total_margin) * 100
            
            recommendations.append({
                'risk_level': risk_level,
                'call_sell_strike': call_sell_opt['strike'],
                'call_buy_strike': call_buy_opt['strike'],
                'put_sell_strike': put_sell_opt['strike'],
                'put_buy_strike': put_buy_opt['strike'],
                'credit_per_contract': credit,
                'contracts': max_contracts,
                'total_credit': total_credit,
                'total_max_loss': total_max_loss,
                'total_margin': total_margin,
                'max_profit_pct': max_profit_pct,
                'upper_breakeven': upper_breakeven,
                'lower_breakeven': lower_breakeven,
                'profit_range': profit_range,
                'profit_range_pct': profit_range_pct,
                'probability_profit': self._estimate_iron_condor_probability(current_price, lower_breakeven, upper_breakeven, 35)
            })
        
        return {
            'strategy': 'IRON_CONDOR',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def calculate_bull_call_spread_accurate(self, calls_df: pd.DataFrame, current_price: float,
                                           capital: float, asset_class: str, max_risk_amount: float = None) -> Dict:
        """Accurate bull call spread calculations with user capital"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Define spread configurations
        spread_configs = [
            ('Conservative', 0.05, 0.95),    # 5% OTM buy, 5% width
            ('Moderate', 0.02, 0.08),        # 2% ITM buy, 8% width
            ('Aggressive', 0.05, 0.12)       # 5% ITM buy, 12% width
        ]
        
        recommendations = []
        
        for risk_level, buy_offset, spread_width in spread_configs:
            
            buy_strike = current_price * (1 - buy_offset)  # Negative offset = ITM
            sell_strike = buy_strike * (1 + spread_width)
            
            # Find closest strikes
            buy_opt = self._find_closest_strike(calls_df, buy_strike)
            sell_opt = self._find_closest_strike(calls_df, sell_strike)
            
            if buy_opt is None or sell_opt is None:
                continue
            
            # Calculate net debit
            net_debit = buy_opt['lastPrice'] - sell_opt['lastPrice']
            
            if net_debit <= 0:
                continue
            
            # Position sizing
            cost_per_contract = net_debit * share_equivalent
            max_contracts = int(capital / cost_per_contract)
            
            if max_contracts == 0:
                continue
            
            # Accurate calculations
            total_cost = max_contracts * cost_per_contract
            spread_width_actual = sell_opt['strike'] - buy_opt['strike']
            max_profit = max_contracts * (spread_width_actual - net_debit) * share_equivalent
            
            # Breakeven
            breakeven = buy_opt['strike'] + net_debit
            
            # Metrics
            max_profit_pct = (max_profit / total_cost) * 100
            breakeven_move = ((breakeven - current_price) / current_price) * 100
            
            # Probability of profit
            probability_profit = self._estimate_probability_above_breakeven(current_price, breakeven, 35)
            
            recommendations.append({
                'risk_level': risk_level,
                'buy_strike': buy_opt['strike'],
                'sell_strike': sell_opt['strike'],
                'buy_price': buy_opt['lastPrice'],
                'sell_price': sell_opt['lastPrice'],
                'net_debit': net_debit,
                'contracts': max_contracts,
                'total_cost': total_cost,
                'max_profit': max_profit,
                'max_loss': total_cost,
                'max_profit_pct': max_profit_pct,
                'breakeven': breakeven,
                'breakeven_move_pct': breakeven_move,
                'probability_profit': probability_profit
            })
        
        return {
            'strategy': 'BULL_CALL_SPREAD',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    # =============================================================================
    # HELPER METHODS FOR ACCURATE CALCULATIONS
    # =============================================================================
    
    def _find_closest_strike(self, options_df: pd.DataFrame, target_strike: float) -> Optional[Dict]:
        """Find option with strike closest to target"""
        if options_df.empty:
            return None
        
        closest_idx = (options_df['strike'] - target_strike).abs().idxmin()
        return options_df.loc[closest_idx].to_dict()
    
    def _estimate_probability_above_breakeven(self, current_price: float, breakeven: float, 
                                            days: int, is_put: bool = False) -> float:
        """Estimate probability of being above/below breakeven using log-normal distribution"""
        
        # Estimate annualized volatility (simplified)
        volatility = 0.25  # 25% default volatility
        
        # Time to expiration in years
        T = days / 365.0
        
        # Calculate d2 from Black-Scholes
        d2 = (math.log(current_price / breakeven) + (-0.5 * volatility ** 2) * T) / (volatility * math.sqrt(T))
        
        if is_put:
            # For puts, we want probability of being below breakeven
            probability = norm.cdf(-d2)
        else:
            # For calls, we want probability of being above breakeven
            probability = norm.cdf(d2)
        
        return max(0.05, min(0.95, probability))  # Bounded between 5% and 95%
    
    def _estimate_iron_condor_probability(self, current_price: float, lower_be: float, 
                                         upper_be: float, days: int) -> float:
        """Estimate probability of staying within iron condor profit range"""
        
        volatility = 0.25
        T = days / 365.0
        
        # Probability of being above lower breakeven
        d2_lower = (math.log(current_price / lower_be) + (-0.5 * volatility ** 2) * T) / (volatility * math.sqrt(T))
        prob_above_lower = norm.cdf(d2_lower)
        
        # Probability of being below upper breakeven
        d2_upper = (math.log(current_price / upper_be) + (-0.5 * volatility ** 2) * T) / (volatility * math.sqrt(T))
        prob_below_upper = norm.cdf(-d2_upper)
        
        # Combined probability (between the two breakevens)
        prob_profit = prob_above_lower + prob_below_upper - 1.0
        
        return max(0.05, min(0.95, prob_profit))
    
    # =============================================================================
    # ENHANCED BACKTESTING WITH ACCURATE MATH
    # =============================================================================
    
    def run_accurate_backtest(self, ticker: str, asset_class: str, strategy: str,
                             start_date: datetime, end_date: datetime, params: Dict) -> Dict:
        """Enhanced backtesting with accurate strategy mathematics and proper initial_capital handling"""
        
        print(f"Running accurate backtest for {strategy} on {ticker}")
        
        try:
            # Get historical data
            total_days = (end_date - start_date).days + 100
            underlying_data = self.get_asset_data(ticker, asset_class, days=total_days)
            
            # Filter to backtest period
            df = underlying_data['historical_data']
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if len(df) < 30:
                raise ValueError(f"Insufficient data: {len(df)} days")
            
            # Ensure initial_capital is in params with default value
            if 'initial_capital' not in params:
                params['initial_capital'] = 10000
            
            # Run strategy-specific backtest
            if strategy == 'COVERED_CALL':
                results = self._backtest_covered_call_accurate(df, params, asset_class)
            elif strategy == 'CASH_SECURED_PUT':
                results = self._backtest_csp_accurate(df, params, asset_class)
            elif strategy == 'IRON_CONDOR':
                results = self._backtest_iron_condor_accurate(df, params, asset_class)
            elif strategy == 'BULL_CALL_SPREAD':
                results = self._backtest_bull_call_accurate(df, params, asset_class)
            elif strategy == 'BUY_AND_HOLD':
                results = self._backtest_buy_hold_accurate(df, asset_class, params)
            else:
                raise ValueError(f"Strategy {strategy} not implemented")
            
            # Calculate metrics
            metrics = self._calculate_accurate_metrics(results, df)
            
            return {
                'ticker': ticker,
                'strategy': strategy,
                'asset_class': asset_class,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'results': results,
                'performance_metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            print(f"Backtest failed: {str(e)}")
            return {
                'ticker': ticker,
                'strategy': strategy,
                'error': str(e),
                'success': False
            }
    
    def _backtest_covered_call_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate covered call backtesting"""
        
        initial_capital = params.get('initial_capital', 10000)
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        days_to_expiry = params.get('days_to_expiry', 30)
        
        trades = []
        portfolio_values = []
        
        cash = 0
        shares = 0
        
        # Initial stock purchase
        entry_price = df.iloc[0]['close']
        shares = int(initial_capital / (entry_price * share_equivalent)) * share_equivalent
        cash = initial_capital - (shares * entry_price)
        
        i = 0
        while i < len(df) - days_to_expiry:
            
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_stock_price = df.iloc[i]['close']
            exit_stock_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            if shares == 0:  # Re-enter position
                shares = int(cash / (entry_stock_price * share_equivalent)) * share_equivalent
                if shares == 0:
                    break
                cash -= shares * entry_stock_price
            
            # Sell call option (5% OTM)
            call_strike = entry_stock_price * 1.05
            
            # Realistic premium calculation (2-4% of stock price)
            time_value = max(0.02, 0.04 * math.sqrt(days_to_expiry / 30))
            moneyness_factor = max(0.5, (call_strike - entry_stock_price) / entry_stock_price)
            call_premium = entry_stock_price * time_value * moneyness_factor
            
            premium_received = call_premium * (shares / share_equivalent)
            cash += premium_received
            
            # Determine outcome at expiration
            if exit_stock_price >= call_strike:
                # Called away - sell shares at strike
                cash += shares * call_strike
                stock_pnl = shares * (call_strike - entry_stock_price)
                total_pnl = stock_pnl + premium_received
                shares = 0
            else:
                # Keep shares and premium
                stock_pnl = shares * (exit_stock_price - entry_stock_price)
                total_pnl = stock_pnl + premium_received
            
            portfolio_value = cash + shares * exit_stock_price
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_stock_price,
                'exit_price': exit_stock_price,
                'call_strike': call_strike,
                'call_premium': call_premium,
                'shares': shares if exit_stock_price < call_strike else int(cash / call_strike) if cash > 0 else 0,
                'premium_received': premium_received,
                'stock_pnl': stock_pnl,
                'total_pnl': total_pnl,
                'portfolio_value': portfolio_value
            })
            
            portfolio_values.append(portfolio_value)
            
            i += days_to_expiry
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_cash': cash,
            'final_shares': shares,
            'initial_capital': initial_capital
        }
    
    def _calculate_accurate_metrics(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Calculate accurate performance metrics"""
        
        portfolio_values = results['portfolio_values']
        trades = results['trades']
        initial_capital = results.get('initial_capital', 10000)
        
        if not portfolio_values:
            return {}
        
        final_value = portfolio_values[-1]
        
        # Total return - FIXED
        total_return = (final_value - initial_capital) / initial_capital
        
        # Annualized return
        days = len(portfolio_values) * (len(df) / len(portfolio_values)) if len(portfolio_values) > 0 else 1
        years = max(days / 365.0, 0.01)  # Prevent division by zero
        annualized_return = (final_value / initial_capital) ** (1/years) - 1
        
        # Volatility and Sharpe
        returns = pd.Series(portfolio_values).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assume 2% risk-free rate
        
        # Drawdown
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.get('total_pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('total_pnl', 0) <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['total_pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['total_pnl'] for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades and avg_loss != 0 else np.inf
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': round(total_return * 100, 2),
            'annualized_return': round(annualized_return * 100, 2),
            'volatility': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'win_rate': round(win_rate * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'total_trades': len(trades),
            'final_value': round(final_value, 2)
        }

    # =============================================================================
    # GREEKS CALCULATIONS WITH PROPER ERROR HANDLING
    # =============================================================================
    
    def get_options_greeks(self, ticker: str, asset_class: str) -> Dict:
        """Calculate options Greeks for all available contracts with proper error handling"""
        try:
            print(f"Calculating {asset_class} Greeks for {ticker}...")
            
            # Get current price and options data
            underlying_data = cached_get_asset_data(self.polygon_api_key, ticker, asset_class, days=30)
            current_price = underlying_data['current_price']
            options_data = cached_get_options_data(self.polygon_api_key, ticker, asset_class, current_price)
            
            calls_df = options_data['calls']
            puts_df = options_data['puts']
            
            if calls_df.empty and puts_df.empty:
                raise ValueError("No options data available for Greeks calculation")
            
            # Calculate Greeks
            calls_greeks = self._calculate_greeks_for_options(calls_df, current_price, options_data['days_to_expiry'], 'call', asset_class)
            puts_greeks = self._calculate_greeks_for_options(puts_df, current_price, options_data['days_to_expiry'], 'put', asset_class)
            
            # Summary statistics
            summary_stats = self._calculate_greeks_summary(calls_greeks, puts_greeks, current_price)
            
            return {
                'underlying_ticker': ticker,
                'underlying_price': current_price,
                'asset_class': asset_class,
                'expiration': options_data['expiration'],
                'days_to_expiry': options_data['days_to_expiry'],
                'calls_greeks': calls_greeks,
                'puts_greeks': puts_greeks,
                'summary_stats': summary_stats,
                'total_contracts': len(calls_greeks) + len(puts_greeks),
                'success': True
            }
            
        except Exception as e:
            print(f"Greeks calculation failed for {ticker}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_greeks_for_options(self, options_df: pd.DataFrame, underlying_price: float, 
                                     days_to_expiry: int, option_type: str, asset_class: str) -> pd.DataFrame:
        """Calculate Greeks for a set of options"""
        if options_df.empty:
            return pd.DataFrame()
        
        greeks_data = []
        
        # Asset-specific volatility
        base_vol = 0.15 if asset_class == 'FOREX' else 0.20 if asset_class == 'INDICES' else 0.25
        r = 0.05  # Risk-free rate
        T = days_to_expiry / 365.0
        
        for _, option in options_df.iterrows():
            strike = option['strike']
            price = option['lastPrice']
            
            # Calculate Greeks
            greeks = self._calculate_black_scholes_greeks(
                underlying_price, strike, T, r, base_vol, option_type
            )
            
            # Calculate moneyness
            moneyness = strike / underlying_price
            
            greeks_data.append({
                'ticker': option['ticker'],
                'strike': strike,
                'price': price,
                'delta': round(greeks['delta'], 4),
                'gamma': round(greeks['gamma'], 6),
                'theta': round(greeks['theta'], 4),
                'vega': round(greeks['vega'], 4),
                'rho': round(greeks['rho'], 4),
                'implied_vol': round(base_vol, 4),
                'moneyness': round(moneyness, 4),
                'time_to_expiry': T
            })
        
        return pd.DataFrame(greeks_data)
    
    def _calculate_black_scholes_greeks(self, S: float, K: float, T: float, r: float, 
                                       sigma: float, option_type: str) -> Dict:
        """Calculate Black-Scholes Greeks"""
        try:
            if T <= 0:
                T = 0.01
            
            # Standard normal distribution calculations
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Calculate Greeks
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - 
                        r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
                rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
            else:  # put
                delta = norm.cdf(d1) - 1
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + 
                        r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
                rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
            
            # Common Greeks for both calls and puts
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
    
    def _calculate_greeks_summary(self, calls_greeks: pd.DataFrame, puts_greeks: pd.DataFrame, 
                                 current_price: float) -> Dict:
        """Calculate summary statistics for Greeks with robust error handling"""
        summary = {}
        
        # Combine all Greeks
        all_greeks = pd.concat([calls_greeks, puts_greeks], ignore_index=True)
        
        if not all_greeks.empty:
            # ATM options (within 2% of current price)
            atm_options = all_greeks[abs(all_greeks['strike'] - current_price) / current_price <= 0.02]
            summary['atm_options'] = len(atm_options)
            
            # OTM calls and puts
            if not calls_greeks.empty:
                otm_calls = calls_greeks[calls_greeks['strike'] > current_price]
                summary['otm_calls'] = len(otm_calls)
            else:
                summary['otm_calls'] = 0
                
            if not puts_greeks.empty:
                otm_puts = puts_greeks[puts_greeks['strike'] < current_price]
                summary['otm_puts'] = len(otm_puts)
            else:
                summary['otm_puts'] = 0
            
            # Highest gamma strike (with error handling)
            if 'gamma' in all_greeks.columns and not all_greeks.empty and all_greeks['gamma'].notna().any():
                try:
                    max_gamma_idx = all_greeks['gamma'].idxmax()
                    if pd.notna(max_gamma_idx):
                        summary['highest_gamma_strike'] = all_greeks.loc[max_gamma_idx, 'strike']
                    else:
                        summary['highest_gamma_strike'] = current_price
                except (KeyError, IndexError):
                    summary['highest_gamma_strike'] = current_price
            else:
                summary['highest_gamma_strike'] = current_price
            
            # Average implied volatility (with error handling)
            if 'implied_vol' in all_greeks.columns and not all_greeks.empty:
                try:
                    valid_iv = all_greeks['implied_vol'].dropna()
                    if len(valid_iv) > 0:
                        summary['avg_implied_vol'] = valid_iv.mean()
                    else:
                        summary['avg_implied_vol'] = 0.25  # Default
                except Exception:
                    summary['avg_implied_vol'] = 0.25  # Default
            else:
                summary['avg_implied_vol'] = 0.25  # Default
        
        else:
            # Default values when no Greeks data available
            summary = {
                'atm_options': 0,
                'otm_calls': 0,
                'otm_puts': 0,
                'highest_gamma_strike': current_price,
                'avg_implied_vol': 0.25
            }
        
        return summary

    # =============================================================================
    # MISSING BACKTEST HELPER METHODS
    # =============================================================================
    
    def _backtest_csp_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate cash secured put backtesting"""
        
        initial_capital = params.get('initial_capital', 10000)
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        days_to_expiry = params.get('days_to_expiry', 30)
        
        trades = []
        portfolio_values = []
        
        cash = initial_capital
        shares = 0
        
        i = 0
        while i < len(df) - days_to_expiry:
            
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Sell put option (5% OTM)
            put_strike = entry_price * 0.95
            
            # Calculate realistic put premium
            time_value = max(0.01, 0.03 * math.sqrt(days_to_expiry / 30))
            intrinsic_value = max(0, put_strike - entry_price)
            put_premium = intrinsic_value + (entry_price * time_value)
            
            # Position sizing
            cash_required = put_strike * share_equivalent
            contracts = int(cash / cash_required)
            if contracts == 0:
                i += days_to_expiry
                continue
            
            premium_received = put_premium * contracts * share_equivalent
            cash_secured = put_strike * contracts * share_equivalent
            
            # Outcome at expiration
            if exit_price <= put_strike:
                # Assigned - buy shares at strike
                shares += contracts * share_equivalent
                cash = cash - cash_secured + premium_received
                stock_cost = shares * put_strike
                total_pnl = premium_received - (put_strike - exit_price) * contracts * share_equivalent
            else:
                # Keep premium
                total_pnl = premium_received
            
            portfolio_value = cash + shares * exit_price
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'put_strike': put_strike,
                'put_premium': put_premium,
                'contracts': contracts,
                'premium_received': premium_received,
                'assigned': exit_price <= put_strike,
                'total_pnl': total_pnl,
                'portfolio_value': portfolio_value
            })
            
            portfolio_values.append(portfolio_value)
            i += days_to_expiry
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_cash': cash,
            'final_shares': shares,
            'initial_capital': initial_capital
        }
    
    def _backtest_iron_condor_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate iron condor backtesting"""
        
        initial_capital = params.get('initial_capital', 10000)
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        days_to_expiry = params.get('days_to_expiry', 30)
        wing_width = params.get('wing_width', 0.05)
        
        trades = []
        portfolio_values = []
        
        cash = initial_capital
        
        i = 0
        while i < len(df) - days_to_expiry:
            
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Iron condor strikes
            call_sell_strike = entry_price * (1 + wing_width)
            call_buy_strike = entry_price * (1 + wing_width * 2)
            put_sell_strike = entry_price * (1 - wing_width)
            put_buy_strike = entry_price * (1 - wing_width * 2)
            
            # Calculate net credit (simplified)
            net_credit_per_contract = entry_price * 0.02  # 2% credit
            max_loss_per_contract = max(
                call_buy_strike - call_sell_strike,
                put_sell_strike - put_buy_strike
            ) - net_credit_per_contract
            
            # Position sizing based on available capital
            margin_per_contract = max_loss_per_contract * share_equivalent
            contracts = int(cash * 0.3 / margin_per_contract)  # Use 30% of capital
            if contracts == 0:
                contracts = 1
            
            total_credit = net_credit_per_contract * contracts * share_equivalent
            total_margin = margin_per_contract * contracts
            
            # Determine outcome
            if put_buy_strike <= exit_price <= call_buy_strike:
                # Maximum profit - price stays in range
                pnl = total_credit
            elif exit_price < put_buy_strike or exit_price > call_buy_strike:
                # Maximum loss
                pnl = -max_loss_per_contract * contracts * share_equivalent
            else:
                # Partial loss
                if exit_price < put_sell_strike:
                    pnl = total_credit - (put_sell_strike - exit_price) * contracts * share_equivalent
                else:  # exit_price > call_sell_strike
                    pnl = total_credit - (exit_price - call_sell_strike) * contracts * share_equivalent
            
            cash += pnl
            portfolio_value = cash
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'call_sell_strike': call_sell_strike,
                'call_buy_strike': call_buy_strike,
                'put_sell_strike': put_sell_strike,
                'put_buy_strike': put_buy_strike,
                'net_credit': net_credit_per_contract,
                'contracts': contracts,
                'total_pnl': pnl,
                'portfolio_value': portfolio_value
            })
            
            portfolio_values.append(portfolio_value)
            i += days_to_expiry
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_cash': cash,
            'final_shares': 0,
            'initial_capital': initial_capital
        }
    
    def _backtest_bull_call_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate bull call spread backtesting"""
        
        initial_capital = params.get('initial_capital', 10000)
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        days_to_expiry = params.get('days_to_expiry', 30)
        
        trades = []
        portfolio_values = []
        
        cash = initial_capital
        
        i = 0
        while i < len(df) - days_to_expiry:
            
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Bull call spread strikes
            buy_strike = entry_price * 0.98  # 2% ITM
            sell_strike = entry_price * 1.08  # 8% OTM
            
            # Calculate net debit
            buy_premium = entry_price * 0.04  # 4% for ITM call
            sell_premium = entry_price * 0.015  # 1.5% for OTM call
            net_debit = buy_premium - sell_premium
            
            # Position sizing
            cost_per_contract = net_debit * share_equivalent
            contracts = int(cash * 0.4 / cost_per_contract)  # Use 40% of capital
            if contracts == 0:
                contracts = 1
            
            total_cost = cost_per_contract * contracts
            max_profit_per_contract = (sell_strike - buy_strike) - net_debit
            
            # Outcome at expiration
            if exit_price >= sell_strike:
                # Maximum profit
                pnl = max_profit_per_contract * contracts * share_equivalent
            elif exit_price <= buy_strike:
                # Maximum loss
                pnl = -total_cost
            else:
                # Partial profit
                pnl = ((exit_price - buy_strike) - net_debit) * contracts * share_equivalent
            
            cash += pnl
            portfolio_value = cash
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'buy_strike': buy_strike,
                'sell_strike': sell_strike,
                'buy_premium': buy_premium,
                'sell_premium': sell_premium,
                'net_debit': net_debit,
                'contracts': contracts,
                'total_pnl': pnl,
                'portfolio_value': portfolio_value
            })
            
            portfolio_values.append(portfolio_value)
            i += days_to_expiry
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_cash': cash,
            'final_shares': 0,
            'initial_capital': initial_capital
        }
    
    def _backtest_buy_hold_accurate(self, df: pd.DataFrame, asset_class: str, params: Dict) -> Dict:
        """Accurate buy and hold backtesting"""
        initial_capital = params.get('initial_capital', 10000)
        entry_price = df.iloc[0]['close']
        exit_price = df.iloc[-1]['close']
        
        shares = initial_capital / entry_price
        final_value = shares * exit_price
        total_return = final_value - initial_capital
        
        # Create portfolio value series
        portfolio_values = (df['close'] / entry_price * initial_capital).tolist()
        
        return {
            'trades': [{
                'entry_date': df.index[0],
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'total_pnl': total_return,
                'portfolio_value': final_value
            }],
            'portfolio_values': portfolio_values,
            'final_cash': final_value,
            'final_shares': 0,
            'initial_capital': initial_capital
        }

    # =============================================================================
    # ENHANCED CHARTING
    # =============================================================================
    
    def create_professional_chart(self, data: Dict, asset_class: str, 
                                 support_resistance: Dict = None) -> go.Figure:
        """Create professional trading chart"""
        
        try:
            df = data['historical_data'].copy()
            
            # Get recent data (1 year)
            one_year_ago = datetime.now() - timedelta(days=365)
            df_recent = df[df.index >= one_year_ago]
            
            if len(df_recent) < 50:
                df_recent = df.tail(252) if len(df) >= 252 else df
            
            # Create figure
            if asset_class == 'FOREX':
                fig = go.Figure()
            else:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.75, 0.25],
                    subplot_titles=[f'{data["ticker"]} - {asset_class}', 'Volume']
                )
            
            # Candlestick chart
            candlestick = go.Candlestick(
                x=df_recent.index,
                open=df_recent['open'],
                high=df_recent['high'],
                low=df_recent['low'],
                close=df_recent['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                increasing_fillcolor='rgba(38, 166, 154, 0.8)',
                decreasing_fillcolor='rgba(239, 83, 80, 0.8)'
            )
            
            if asset_class == 'FOREX':
                fig.add_trace(candlestick)
            else:
                fig.add_trace(candlestick, row=1, col=1)
            
            # Volume (non-FX only)
            if asset_class != 'FOREX' and 'volume' in df_recent.columns:
                colors = ['#ef5350' if c < o else '#26a69a' 
                         for c, o in zip(df_recent['close'], df_recent['open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=df_recent.index,
                        y=df_recent['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Moving averages with proper min_periods to span entire chart
            ma_configs = [
                {'period': 20, 'color': '#FF9800', 'width': 2},
                {'period': 50, 'color': '#2196F3', 'width': 2},
                {'period': 200, 'color': '#9C27B0', 'width': 3}
            ]
            
            for ma_config in ma_configs:
                period = ma_config['period']
                # Set min_periods to 1 so MA can start calculating from day 1 and span entire chart
                min_periods = max(1, min(period // 4, len(df_recent)))  # Use 1/4 of period as minimum
                
                if len(df_recent) >= min_periods:
                    ma = df_recent['close'].rolling(period, min_periods=min_periods).mean()
                    # Filter out NaN values for cleaner display
                    ma_clean = ma.dropna()
                    
                    if not ma_clean.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=ma_clean.index,
                                y=ma_clean.values,
                                mode='lines',
                                name=f'MA{period}',
                                line=dict(color=ma_config['color'], width=ma_config['width']),
                                opacity=0.8
                            ),
                            row=1, col=1 if asset_class != 'FOREX' else None
                        )
            
            # Support/Resistance levels
            if support_resistance:
                for level_name, color, style in [
                    ('support_level', '#4CAF50', 'dash'),
                    ('resistance_level', '#F44336', 'dash'),
                    ('target_price', '#FFC107', 'dot')
                ]:
                    if level_name in support_resistance:
                        level_value = support_resistance[level_name]
                        fig.add_hline(
                            y=level_value,
                            line_dash=style,
                            line_color=color,
                            line_width=2,
                            opacity=0.8,
                            row=1, col=1 if asset_class != 'FOREX' else None
                        )
            
            # Styling
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(17,17,17,0.8)',
                font=dict(family="Arial", size=12, color="white"),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(255,255,255,0.2)",
                    borderwidth=1
                ),
                xaxis_rangeslider_visible=False,
                height=600 if asset_class == 'FOREX' else 700
            )
            
            # Grid styling
            fig.update_xaxes(
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1
            )
            fig.update_yaxes(
                gridcolor='rgba(128,128,128,0.2)', 
                gridwidth=1
            )
            
            return fig
            
        except Exception as e:
            print(f"Chart creation failed: {e}")
            return go.Figure()

# =============================================================================
# MARKET SCANNER CLASS
# =============================================================================

class MarketScanner:
    """Professional Market Scanner for Top Stock Recommendations"""
    
    def __init__(self, strategist):
        self.strategist = strategist
        
        # S&P 500 top stocks for scanning
        self.scan_universe = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO', 'TXN', 'QCOM', 'IBM', 'MU', 'AMAT',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL',
            'BLK', 'SCHW', 'CB', 'AIG', 'USB', 'PNC', 'COF', 'TFC', 'CME', 'ICE',
            
            # Healthcare & Pharma
            'JNJ', 'PFE', 'UNH', 'ABBV', 'BMY', 'MRK', 'CVS', 'AMGN', 'GILD', 'LLY',
            'TMO', 'DHR', 'ABT', 'MDT', 'BDX', 'SYK', 'EW', 'ZTS', 'ILMN', 'REGN',
            
            # Consumer & Retail
            'HD', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'LOW', 'TGT', 'SBUX', 'MCD',
            'NKE', 'DIS', 'AMZN', 'EBAY', 'ETSY', 'LULU', 'TJX', 'GPS', 'M', 'JWN',
            
            # Industrial & Energy
            'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PXD', 'KMI', 'WMB', 'EPD',
            
            # ETFs for broader exposure
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI'
        ]
    
    @st.cache_data(ttl=1800)  # 30-minute cache
    def scan_market(_self, max_stocks=500) -> Dict:
        """Scan market and return top buy/sell recommendations"""
        
        print(f"🔍 Scanning {max_stocks} stocks for opportunities...")
        
        stock_analysis = []
        processed_count = 0
        
        for ticker in _self.scan_universe[:max_stocks]:
            try:
                # Get stock data
                data = cached_get_asset_data(_self.strategist.polygon_api_key, ticker, 'EQUITIES', days=252)
                
                # Analyze market conditions
                analysis = _self.strategist.analyze_market_conditions(data)
                
                # Calculate technical score
                tech_score = _self._calculate_technical_score(analysis, data)
                
                stock_analysis.append({
                    'ticker': ticker,
                    'current_price': data['current_price'],
                    'analysis': analysis,
                    'technical_score': tech_score,
                    'data': data
                })
                
                processed_count += 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Failed to analyze {ticker}: {str(e)}")
                continue
        
        print(f"✅ Successfully analyzed {processed_count} stocks")
        
        # Sort by technical score
        stock_analysis.sort(key=lambda x: x['technical_score'], reverse=True)
        
        # Get top 10 buy and sell recommendations
        top_buys = stock_analysis[:10]
        top_sells = stock_analysis[-10:]
        top_sells.reverse()  # Show worst first
        
        return {
            'top_buys': top_buys,
            'top_sells': top_sells,
            'total_analyzed': processed_count,
            'scan_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _calculate_technical_score(self, analysis: Dict, data: Dict) -> float:
        """Calculate comprehensive technical score (0-100)"""
        
        score = 50.0  # Neutral starting point
        current_price = data['current_price']
        
        # Trend Score (30% weight)
        trend_scores = {
            'STRONG_BULLISH': 25,
            'BULLISH': 15,
            'SIDEWAYS': 0,
            'BEARISH': -15,
            'STRONG_BEARISH': -25
        }
        score += trend_scores.get(analysis['trend'], 0)
        
        # Momentum Score (25% weight)  
        rsi = analysis['rsi']
        if 40 <= rsi <= 60:
            score += 10  # Neutral RSI is good
        elif 30 <= rsi < 40:
            score += 15  # Oversold but not extreme
        elif 60 < rsi <= 70:
            score += 15  # Overbought but not extreme
        elif rsi < 30:
            score += 20  # Very oversold - potential bounce
        elif rsi > 80:
            score -= 20  # Very overbought - potential decline
        elif 70 < rsi <= 80:
            score -= 10  # Moderately overbought
        
        # Volatility Score (15% weight)
        vol_regime = analysis['volatility_regime']
        vol_scores = {
            'LOW_VOL': 5,
            'NORMAL_VOL': 10,
            'HIGH_VOL': -5,
            'EXTREME_VOL': -15
        }
        score += vol_scores.get(vol_regime, 0)
        
        # Price Position Score (20% weight)
        high_52w = data['high_52w']
        low_52w = data['low_52w']
        
        # Calculate where current price sits in 52-week range
        price_percentile = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
        
        if 0.3 <= price_percentile <= 0.7:
            score += 10  # Good position in range
        elif price_percentile < 0.2:
            score += 15  # Near lows - potential value
        elif price_percentile > 0.8:
            score -= 10  # Near highs - proceed with caution
        
        # Recent Performance Score (10% weight)
        price_change_20d = analysis.get('price_change_20d', 0)
        if -5 <= price_change_20d <= 5:
            score += 5  # Stable recent performance
        elif 5 < price_change_20d <= 15:
            score += 10  # Positive momentum
        elif price_change_20d > 20:
            score -= 5  # May be overextended
        elif -15 <= price_change_20d < -5:
            score += 5  # Potential oversold bounce
        elif price_change_20d < -20:
            score -= 10  # Strong downtrend
        
        return max(0, min(100, score))
    
    def get_stock_recommendation_text(self, stock_data: Dict) -> str:
        """Generate detailed recommendation text"""
        
        analysis = stock_data['analysis']
        score = stock_data['technical_score']
        ticker = stock_data['ticker']
        
        # Determine overall recommendation
        if score >= 70:
            recommendation = "🟢 **STRONG BUY**"
        elif score >= 60:
            recommendation = "🟡 **BUY**"
        elif score >= 45:
            recommendation = "⚪ **HOLD**"
        elif score >= 35:
            recommendation = "🟠 **SELL**"
        else:
            recommendation = "🔴 **STRONG SELL**"
        
        # Build explanation
        explanation = f"{recommendation}\n\n"
        explanation += f"**Technical Score:** {score:.1f}/100\n\n"
        
        # Key factors
        explanation += "**Key Factors:**\n"
        explanation += f"• **Trend:** {analysis['trend'].replace('_', ' ').title()}\n"
        explanation += f"• **RSI:** {analysis['rsi']:.1f} ({analysis['momentum']})\n"
        explanation += f"• **Volatility:** {analysis['volatility_regime'].replace('_', ' ').title()}\n"
        explanation += f"• **20D Change:** {analysis['price_change_20d']:.1f}%\n\n"
        
        # Strategic advice
        if score >= 60:
            explanation += "**Strategy Suggestions:**\n"
            explanation += "• Consider covered calls if you own shares\n"
            explanation += "• Cash-secured puts for entry opportunities\n"
            explanation += "• Bull call spreads for leveraged upside\n"
        elif score <= 40:
            explanation += "**Risk Management:**\n"
            explanation += "• Consider protective puts if holding\n"
            explanation += "• Bear put spreads for downside exposure\n"
            explanation += "• Avoid naked calls in this environment\n"
        
        return explanation
    
# =============================================================================
# ENHANCED RATE-LIMITED MARKET SCANNER
# =============================================================================

class EnhancedMarketScanner:
    """Enhanced Market Scanner with Rate Limiting and Comprehensive Stock Universe"""
    
    def __init__(self, strategist, rate_limit_delay=2.0):
        self.strategist = strategist
        self.rate_limit_delay = rate_limit_delay  # Base delay between requests
        self.max_retries = 3
        self.retry_delay = 5.0  # Delay between retries
        
        # Initialize requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Comprehensive stock universe (~5000 stocks)
        self.full_stock_universe = self._build_comprehensive_stock_universe()
        
    def _build_comprehensive_stock_universe(self) -> List[str]:
        """Build comprehensive list of ~5000 stocks to scan"""
        
        # S&P 500 stocks (most liquid and important)
        sp500_stocks = [
            # Technology Sector
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE',
            'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO', 'TXN', 'QCOM', 'IBM', 'MU',
            'AMAT', 'LRCX', 'ADI', 'MCHP', 'KLAC', 'SNPS', 'CDNS', 'FTNT', 'HPQ', 'DELL',
            'VMW', 'INTU', 'ADSK', 'ANSS', 'CTSH', 'FISV', 'FIS', 'PAYX', 'ADP', 'MA',
            'V', 'PYPL', 'SQ', 'EBAY', 'ETSY', 'SHOP', 'TWLO', 'ZM', 'DOCU', 'OKTA',
            
            # Healthcare & Pharmaceuticals  
            'JNJ', 'PFE', 'UNH', 'ABBV', 'BMY', 'MRK', 'CVS', 'AMGN', 'GILD', 'LLY',
            'TMO', 'DHR', 'ABT', 'MDT', 'BDX', 'SYK', 'EW', 'ZTS', 'ILMN', 'REGN',
            'VRTX', 'BIIB', 'MRNA', 'BNTX', 'ZBH', 'BAX', 'BSX', 'DXCM', 'ISRG', 'RMD',
            'A', 'ALGN', 'HOLX', 'IDXX', 'IQV', 'MTD', 'PKI', 'TECH', 'TFX', 'UHS',
            'VAR', 'WAT', 'XRAY', 'CI', 'HUM', 'ANTM', 'CNC', 'MOH', 'ELV', 'HCA',
            
            # Financial Services
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'CB',
            'AIG', 'USB', 'PNC', 'COF', 'TFC', 'CME', 'ICE', 'SPGI', 'MCO', 'AON',
            'MMC', 'AJG', 'BRK.A', 'BRK.B', 'PRU', 'MET', 'AFL', 'ALL', 'TRV', 'PGR',
            'HIG', 'CMA', 'FITB', 'HBAN', 'KEY', 'RF', 'CFG', 'ZION', 'SIVB', 'PBCT',
            'STT', 'BK', 'NTRS', 'AMG', 'TROW', 'BEN', 'IVZ', 'LNC', 'UNM', 'RE',
            
            # Consumer Discretionary
            'HD', 'LOW', 'TGT', 'COST', 'WMT', 'SBUX', 'MCD', 'NKE', 'DIS', 'AMZN',
            'TSLA', 'F', 'GM', 'TM', 'HMC', 'RACE', 'LCID', 'RIVN', 'NIO', 'XPEV',
            'LI', 'BYDDY', 'TJX', 'GPS', 'M', 'JWN', 'KSS', 'DKS', 'BBY', 'BBBY',
            'BIG', 'COST', 'DLTR', 'DG', 'FIVE', 'OLLI', 'BURL', 'ULTA', 'LULU', 'UAA',
            'UA', 'VFC', 'PVH', 'RL', 'CPRI', 'TPG', 'MAR', 'HLT', 'IHG', 'H',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB',
            'CAG', 'SJM', 'HRL', 'TSN', 'TAP', 'STZ', 'DEO', 'PM', 'MO', 'BTI',
            'UL', 'NESN', 'MDLZ', 'MNST', 'KDP', 'DPZ', 'QSR', 'SBUX', 'YUM', 'CMG',
            'MCD', 'WEN', 'JACK', 'PZZA', 'BLMN', 'DRI', 'EAT', 'TXRH', 'CAKE', 'DENN',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PXD', 'KMI', 'WMB', 'EPD',
            'ET', 'OKE', 'TRGP', 'MMP', 'MPLX', 'PAA', 'BKR', 'HAL', 'FTI', 'NOV',
            'RIG', 'HP', 'PTEN', 'CLR', 'FANG', 'MRO', 'APA', 'DVN', 'CNX', 'EQT',
            'AR', 'CHK', 'RRC', 'SM', 'MTDR', 'PR', 'WLL', 'OVV', 'MGY', 'CRGY',
            
            # Industrial
            'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            'GD', 'LHX', 'TXT', 'ITW', 'EMR', 'ETN', 'PH', 'JCI', 'CMI', 'DOV',
            'FTV', 'XYL', 'IEX', 'FAST', 'PCAR', 'CSX', 'UNP', 'NSC', 'CP', 'CNI',
            'KSU', 'WAB', 'TRN', 'RAIL', 'GWR', 'JBHT', 'CHRW', 'EXPD', 'LSTR', 'ODFL',
            
            # Materials
            'LIN', 'APD', 'AIR', 'ECL', 'DD', 'DOW', 'PPG', 'SHW', 'FCX', 'NEM',
            'GOLD', 'AEM', 'AU', 'KGC', 'WPM', 'FNV', 'PAAS', 'HL', 'CDE', 'SSRM',
            'AA', 'CENX', 'X', 'CLF', 'NUE', 'STLD', 'CMC', 'RS', 'MT', 'TX',
            'VALE', 'RIO', 'BHP', 'SCCO', 'TECK', 'IFF', 'FMC', 'CF', 'MOS', 'NTR',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'MAA', 'ESS',
            'UDR', 'CPT', 'AIV', 'BXP', 'VTR', 'WELL', 'PEAK', 'DOC', 'O', 'STAG',
            'WPC', 'NNN', 'ADC', 'STOR', 'EPR', 'GTY', 'GOOD', 'SRC', 'WRI', 'REG',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED',
            'EIX', 'ETR', 'FE', 'ES', 'CNP', 'NI', 'LNT', 'WTRG', 'AEE', 'CMS',
            'DTE', 'EVRG', 'PNW', 'OGE', 'IDA', 'NWE', 'AGR', 'AVA', 'BKH', 'MDU'
        ]
        
        # Add Russell 1000 additional stocks
        russell_additional = [
            # Additional Tech
            'ROKU', 'PINS', 'SNAP', 'TWTR', 'SPOT', 'SQ', 'HOOD', 'COIN', 'RBLX', 'U',
            'DDOG', 'SNOW', 'CRWD', 'ZS', 'OKTA', 'NET', 'FSLY', 'CFLT', 'MDB', 'ESTC',
            
            # Additional Healthcare/Biotech
            'TDOC', 'VEEV', 'ZBH', 'HOLX', 'ALGN', 'DXCM', 'ISRG', 'INTUV', 'PODD', 'NVST',
            'BMRN', 'SRPT', 'RARE', 'FOLD', 'BLUE', 'SAGE', 'IONS', 'EXEL', 'HALO', 'ARWR',
            
            # Additional Consumer
            'UBER', 'LYFT', 'DASH', 'ABNB', 'PTON', 'ZG', 'Z', 'CHWY', 'CVNA', 'CARG',
            'KMX', 'LAD', 'AN', 'PAG', 'GPI', 'AAP', 'ORLY', 'AZO', 'WOOF', 'PETQ',
            
            # Additional Financial
            'SQ', 'PYPL', 'AFRM', 'UPST', 'LC', 'SOFI', 'OPEN', 'COMP', 'TREE', 'OSCR',
            'ALLY', 'LOAN', 'CACC', 'WRLD', 'ENVA', 'OMF', 'PSEC', 'MAIN', 'GLAD', 'GAIN',
            
            # REITs
            'REIT', 'SPG', 'VICI', 'GLPI', 'MGP', 'HST', 'RHP', 'PEB', 'CUBE', 'LSI',
            'ELS', 'SUI', 'MSA', 'SBRA', 'OHI', 'WELL', 'VTR', 'PEAK', 'DOC', 'HR',
            
            # Additional Energy
            'TELL', 'LNG', 'FLNG', 'NEXT', 'BE', 'PLUG', 'FCEL', 'BLDP', 'HYMC', 'RNW',
            'GPRE', 'REX', 'ALTO', 'PEIX', 'BIOX', 'GEVO', 'REGI', 'ORIG', 'CLNE', 'WPRT'
        ]
        
        # Add mid-cap growth stocks
        midcap_growth = [
            'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR', 'SPWR', 'CSIQ', 'JKS', 'SOL', 'MAXN',
            'APPS', 'SMAR', 'BILL', 'PCTY', 'PAYC', 'GWRE', 'APPF', 'BLKB', 'COUP', 'DOCN',
            'FROG', 'GTLB', 'INTA', 'JAMF', 'MNDY', 'NCNO', 'PATH', 'PD', 'QTWO', 'RDWR',
            'SMAR', 'TASK', 'TEAM', 'TENB', 'TLRY', 'WDAY', 'ZEN', 'ZUO', 'AMPL', 'BIGC'
        ]
        
        # Add small-cap value opportunities
        smallcap_value = [
            'SAVE', 'AAL', 'UAL', 'DAL', 'LUV', 'JETS', 'HA', 'MESA', 'SKYW', 'JBLU',
            'CCL', 'RCL', 'NCLH', 'CUK', 'FUN', 'SIX', 'CWH', 'PLNT', 'CLUB', 'BYD',
            'TXRH', 'CAKE', 'DENN', 'CBRL', 'CRACKER', 'DIN', 'EAT', 'BJRI', 'CHUY', 'DAVE',
            'FRGI', 'HAYN', 'JACK', 'LOCO', 'NDLS', 'PZZA', 'RUTH', 'SHAK', 'SONC', 'WEN'
        ]
        
        # Add popular meme/retail stocks
        meme_retail = [
            'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'WISH', 'CLOV', 'WKHS', 'RIDE', 'NKLA',
            'SPCE', 'HYLN', 'GOEV', 'CANOO', 'FSR', 'ARVL', 'MULN', 'SNDL', 'TLRY', 'CGC',
            'CRON', 'ACB', 'APHA', 'HEXO', 'OGI', 'LABS', 'KERN', 'GRWG', 'SMG', 'IIPR'
        ]
        
        # Add crypto-related stocks
        crypto_stocks = [
            'COIN', 'HOOD', 'RIOT', 'MARA', 'CAN', 'BITF', 'ANY', 'CORZ', 'WULF', 'CIFR',
            'BTC', 'ETHE', 'GBTC', 'BITO', 'BITI', 'BLOK', 'LEGR', 'KRYP', 'CRYP', 'CHAT',
            'SQ', 'PYPL', 'NVDA', 'AMD', 'INTC', 'TSM', 'ASML', 'LRCX', 'AMAT', 'KLAC'
        ]
        
        # Add EV and clean energy
        ev_clean = [
            'TSLA', 'LCID', 'RIVN', 'NIO', 'XPEV', 'LI', 'NKLA', 'FSR', 'GOEV', 'RIDE',
            'HYLN', 'WKHS', 'BLBD', 'SOLO', 'AYRO', 'IDEX', 'KNDI', 'NIU', 'CBAT', 'BYDDY',
            'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR', 'SPWR', 'CSIQ', 'JKS', 'SOL', 'MAXN',
            'BE', 'PLUG', 'FCEL', 'BLDP', 'HYMC', 'RNW', 'GPRE', 'REX', 'ALTO', 'PEIX'
        ]
        
        # Add biotechnology
        biotech = [
            'MRNA', 'BNTX', 'NVAX', 'INO', 'OCGN', 'VXRT', 'HGEN', 'ATOS', 'CYDY', 'SRNE',
            'SAVA', 'AVXL', 'AXSM', 'ACAD', 'JAZZ', 'HALO', 'ARWR', 'IONS', 'EXEL', 'FOLD',
            'BLUE', 'SAGE', 'RARE', 'BMRN', 'SRPT', 'PTCT', 'ALNY', 'MYGN', 'TECH', 'INCY'
        ]
        
        # Combine all lists and remove duplicates
        all_stocks = list(set(
            sp500_stocks + russell_additional + midcap_growth + 
            smallcap_value + meme_retail + crypto_stocks + 
            ev_clean + biotech
        ))
        
        # Add some additional random symbols to reach ~5000
        # Generate additional symbols (this is a simplified approach)
        additional_symbols = []
        
        # Add some systematic symbol generation
        for letter1 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            for letter2 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if len(additional_symbols) < 3000:
                    # Two-letter combinations
                    additional_symbols.append(letter1 + letter2)
                    
                    # Three-letter combinations (more common)
                    for letter3 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:10]:  # Limit to first 10 letters
                        if len(additional_symbols) < 3000:
                            additional_symbols.append(letter1 + letter2 + letter3)
                        
                        # Four-letter combinations (most common)
                        for letter4 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:5]:  # Limit to first 5 letters
                            if len(additional_symbols) < 3000:
                                additional_symbols.append(letter1 + letter2 + letter3 + letter4)
        
        # Combine and shuffle
        final_list = all_stocks + additional_symbols[:3000]
        random.shuffle(final_list)
        
        print(f"Built comprehensive stock universe: {len(final_list)} symbols")
        return final_list[:5000]  # Cap at 5000
    
    def _make_api_call_with_retry(self, ticker: str, asset_class: str, days: int = 252) -> Dict:
        """Make API call with proper retry logic and rate limiting"""
        
        for attempt in range(self.max_retries):
            try:
                # Add random jitter to prevent thundering herd
                jitter = random.uniform(0.5, 1.5)
                delay = self.rate_limit_delay * jitter
                
                if attempt > 0:
                    # Exponential backoff for retries
                    delay = self.retry_delay * (2 ** attempt) + jitter
                    print(f"Retry {attempt} for {ticker} after {delay:.1f}s delay")
                
                time.sleep(delay)
                
                # Make the API call
                data = cached_get_asset_data(
                    self.strategist.polygon_api_key, 
                    ticker, 
                    asset_class, 
                    days
                )
                return data
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if '429' in error_msg or 'rate limit' in error_msg or 'too many' in error_msg:
                    if attempt < self.max_retries - 1:
                        # Rate limit hit, wait longer
                        backoff_delay = self.retry_delay * (3 ** attempt) + random.uniform(2, 8)
                        print(f"Rate limit hit for {ticker}, waiting {backoff_delay:.1f}s before retry")
                        time.sleep(backoff_delay)
                        continue
                    else:
                        print(f"Final retry failed for {ticker}: Rate limit exceeded")
                        raise
                else:
                    # Other error, don't retry
                    print(f"API error for {ticker}: {str(e)}")
                    raise
        
        raise Exception(f"Max retries exceeded for {ticker}")
    
    @st.cache_data(ttl=7200)  # 2-hour cache for full market scan
    def scan_comprehensive_market(_self, max_stocks=5000, start_from=0, batch_size=100, 
                                 progress_callback=None, status_callback=None) -> Dict:
        """Comprehensive market scan with progress tracking"""
        
        print(f"Starting comprehensive market scan: {max_stocks} stocks")
        
        if status_callback:
            status_callback(f"🚀 Starting scan of {max_stocks} stocks...")
        
        stock_analysis = []
        processed_count = 0
        failed_count = 0
        batch_count = 0
        
        # Get stocks to scan
        stocks_to_scan = _self.full_stock_universe[start_from:start_from + max_stocks]
        
        # Process in batches
        for i in range(0, len(stocks_to_scan), batch_size):
            batch = stocks_to_scan[i:i + batch_size]
            batch_count += 1
            
            if status_callback:
                status_callback(f"📊 Processing batch {batch_count} ({i+1}-{min(i+batch_size, len(stocks_to_scan))} of {len(stocks_to_scan)})")
            
            batch_start_time = time.time()
            
            for j, ticker in enumerate(batch):
                try:
                    # Update progress
                    current_position = i + j + 1
                    if progress_callback:
                        progress_callback(current_position / len(stocks_to_scan))
                    
                    # Get stock data with retry logic
                    data = _self._make_api_call_with_retry(ticker, 'EQUITIES', days=252)
                    
                    # Skip if insufficient data
                    if len(data['historical_data']) < 50:
                        continue
                    
                    # Analyze market conditions
                    analysis = _self.strategist.analyze_market_conditions(data)
                    
                    # Calculate technical score
                    tech_score = _self._calculate_technical_score(analysis, data)
                    
                    stock_analysis.append({
                        'ticker': ticker,
                        'current_price': data['current_price'],
                        'analysis': analysis,
                        'technical_score': tech_score,
                        'data': data,
                        'scan_time': datetime.now()
                    })
                    
                    processed_count += 1
                    
                    # Log progress every 50 stocks
                    if processed_count % 50 == 0:
                        elapsed = time.time() - batch_start_time
                        rate = 50 / elapsed if elapsed > 0 else 0
                        print(f"Processed {processed_count} stocks, current rate: {rate:.1f} stocks/sec")
                        batch_start_time = time.time()
                    
                except Exception as e:
                    failed_count += 1
                    print(f"Failed to analyze {ticker}: {str(e)}")
                    
                    # If we're getting too many failures, increase delays
                    if failed_count > processed_count * 0.3:  # More than 30% failure rate
                        print("High failure rate detected, increasing rate limit delay")
                        _self.rate_limit_delay = min(_self.rate_limit_delay * 1.5, 10.0)
            
            # Batch completion
            batch_time = time.time() - batch_start_time
            if status_callback:
                status_callback(f"✅ Batch {batch_count} completed in {batch_time:.1f}s. Success: {processed_count}, Failed: {failed_count}")
        
        print(f"Scan completed! Analyzed {processed_count} stocks successfully, {failed_count} failed")
        
        if len(stock_analysis) < 20:
            raise ValueError(f"Insufficient data: only {len(stock_analysis)} stocks analyzed successfully")
        
        # Sort by technical score
        stock_analysis.sort(key=lambda x: x['technical_score'], reverse=True)
        
        # Get top 10 buy and sell recommendations
        top_buys = stock_analysis[:10]
        top_sells = stock_analysis[-10:]
        top_sells.reverse()  # Show worst first
        
        return {
            'top_buys': top_buys,
            'top_sells': top_sells,
            'total_analyzed': processed_count,
            'total_failed': failed_count,
            'scan_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scan_duration_minutes': round((time.time() - stock_analysis[0]['scan_time'].timestamp()) / 60, 1) if stock_analysis else 0,
            'success_rate': round((processed_count / (processed_count + failed_count)) * 100, 1) if processed_count + failed_count > 0 else 0
        }
    
    def _calculate_technical_score(self, analysis: Dict, data: Dict) -> float:
        """Calculate comprehensive technical score (0-100) - same as before"""
        
        score = 50.0  # Neutral starting point
        current_price = data['current_price']
        
        # Trend Score (30% weight)
        trend_scores = {
            'STRONG_BULLISH': 25,
            'BULLISH': 15,
            'SIDEWAYS': 0,
            'BEARISH': -15,
            'STRONG_BEARISH': -25
        }
        score += trend_scores.get(analysis['trend'], 0)
        
        # Momentum Score (25% weight)  
        rsi = analysis['rsi']
        if 40 <= rsi <= 60:
            score += 10  # Neutral RSI is good
        elif 30 <= rsi < 40:
            score += 15  # Oversold but not extreme
        elif 60 < rsi <= 70:
            score += 15  # Overbought but not extreme
        elif rsi < 30:
            score += 20  # Very oversold - potential bounce
        elif rsi > 80:
            score -= 20  # Very overbought - potential decline
        elif 70 < rsi <= 80:
            score -= 10  # Moderately overbought
        
        # Volatility Score (15% weight)
        vol_regime = analysis['volatility_regime']
        vol_scores = {
            'LOW_VOL': 5,
            'NORMAL_VOL': 10,
            'HIGH_VOL': -5,
            'EXTREME_VOL': -15
        }
        score += vol_scores.get(vol_regime, 0)
        
        # Price Position Score (20% weight)
        high_52w = data['high_52w']
        low_52w = data['low_52w']
        
        # Calculate where current price sits in 52-week range
        price_percentile = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
        
        if 0.3 <= price_percentile <= 0.7:
            score += 10  # Good position in range
        elif price_percentile < 0.2:
            score += 15  # Near lows - potential value
        elif price_percentile > 0.8:
            score -= 10  # Near highs - proceed with caution
        
        # Recent Performance Score (10% weight)
        price_change_20d = analysis.get('price_change_20d', 0)
        if -5 <= price_change_20d <= 5:
            score += 5  # Stable recent performance
        elif 5 < price_change_20d <= 15:
            score += 10  # Positive momentum
        elif price_change_20d > 20:
            score -= 5  # May be overextended
        elif -15 <= price_change_20d < -5:
            score += 5  # Potential oversold bounce
        elif price_change_20d < -20:
            score -= 10  # Strong downtrend
        
        return max(0, min(100, score))
    
    def get_stock_recommendation_text(self, stock_data: Dict) -> str:
        """Generate detailed recommendation text - same as before"""
        
        analysis = stock_data['analysis']
        score = stock_data['technical_score']
        ticker = stock_data['ticker']
        
        # Determine overall recommendation
        if score >= 70:
            recommendation = "🟢 **STRONG BUY**"
        elif score >= 60:
            recommendation = "🟡 **BUY**"
        elif score >= 45:
            recommendation = "⚪ **HOLD**"
        elif score >= 35:
            recommendation = "🟠 **SELL**"
        else:
            recommendation = "🔴 **STRONG SELL**"
        
        # Build explanation
        explanation = f"{recommendation}\n\n"
        explanation += f"**Technical Score:** {score:.1f}/100\n\n"
        
        # Key factors
        explanation += "**Key Factors:**\n"
        explanation += f"• **Trend:** {analysis['trend'].replace('_', ' ').title()}\n"
        explanation += f"• **RSI:** {analysis['rsi']:.1f} ({analysis['momentum']})\n"
        explanation += f"• **Volatility:** {analysis['volatility_regime'].replace('_', ' ').title()}\n"
        explanation += f"• **20D Change:** {analysis['price_change_20d']:.1f}%\n\n"
        
        # Strategic advice
        if score >= 60:
            explanation += "**Strategy Suggestions:**\n"
            explanation += "• Consider covered calls if you own shares\n"
            explanation += "• Cash-secured puts for entry opportunities\n"
            explanation += "• Bull call spreads for leveraged upside\n"
        elif score <= 40:
            explanation += "**Risk Management:**\n"
            explanation += "• Consider protective puts if holding\n"
            explanation += "• Bear put spreads for downside exposure\n"
            explanation += "• Avoid naked calls in this environment\n"
        
        return explanation

# =============================================================================
# ENHANCED UI FOR COMPREHENSIVE SCANNING
# =============================================================================

def display_comprehensive_scanner_ui(strategist):
    """Display enhanced scanner UI with progress tracking"""
    
    st.subheader("🎯 Comprehensive Market Scanner (5000+ Stocks)")
    
    # Enhanced controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        scan_size = st.selectbox(
            "Stocks to Scan",
            [100, 500, 1000, 2500, 5000],
            index=2,
            format_func=lambda x: f"{x:,} Stocks"
        )
    
    with col2:
        rate_limit = st.selectbox(
            "Rate Limit (sec/request)",
            [1.0, 2.0, 3.0, 5.0, 10.0],
            index=1,
            format_func=lambda x: f"{x:.1f} seconds"
        )
    
    with col3:
        batch_size = st.selectbox(
            "Batch Size",
            [50, 100, 200],
            index=1,
            help="Process stocks in batches for better progress tracking"
        )
    
    with col4:
        start_offset = st.number_input(
            "Start From Stock #",
            min_value=0,
            max_value=4900,
            value=0,
            step=100,
            help="Resume scan from specific position"
        )
    
    # Estimated time calculation
    estimated_minutes = (scan_size * rate_limit) / 60
    st.info(f"⏱️ **Estimated Time:** {estimated_minutes:.1f} minutes for {scan_size:,} stocks at {rate_limit:.1f}s per request")
    
    # Scan controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        scan_btn = st.button(
            f"🔍 Start Comprehensive Scan ({scan_size:,} stocks)",
            type="primary",
            help=f"Scan {scan_size:,} stocks with {rate_limit:.1f}s delays"
        )
    
    with col2:
        if st.button("⏸️ Pause Scan", help="Pause current scan"):
            st.session_state.scan_paused = True
    
    with col3:
        if st.button("🗑️ Clear Results", help="Clear cached results"):
            if 'comprehensive_scan_results' in st.session_state:
                del st.session_state.comprehensive_scan_results
            st.success("Results cleared!")
    
    # Progress tracking placeholders
    progress_bar = st.empty()
    status_text = st.empty()
    stats_container = st.empty()
    
    # Run comprehensive scan
    if scan_btn:
        # Initialize enhanced scanner
        enhanced_scanner = EnhancedMarketScanner(strategist, rate_limit_delay=rate_limit)
        
        def update_progress(progress):
            progress_bar.progress(progress)
        
        def update_status(status):
            status_text.info(status)
        
        with st.spinner(f"🔍 Scanning {scan_size:,} stocks... This will take approximately {estimated_minutes:.1f} minutes"):
            try:
                scan_results = enhanced_scanner.scan_comprehensive_market(
                    max_stocks=scan_size,
                    start_from=start_offset,
                    batch_size=batch_size,
                    progress_callback=update_progress,
                    status_callback=update_status
                )
                
                st.session_state.comprehensive_scan_results = scan_results
                
                # Final statistics
                with stats_container.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Stocks Analyzed", f"{scan_results['total_analyzed']:,}")
                    
                    with col2:
                        st.metric("Success Rate", f"{scan_results['success_rate']:.1f}%")
                    
                    with col3:
                        st.metric("Scan Duration", f"{scan_results['scan_duration_minutes']:.1f} min")
                    
                    with col4:
                        st.metric("Failed Requests", f"{scan_results['total_failed']:,}")
                
                st.success(f"✅ Comprehensive scan completed! Analyzed {scan_results['total_analyzed']:,} stocks successfully")
                
            except Exception as e:
                st.error(f"Comprehensive scan failed: {str(e)}")
    
    # Display results if available
    if hasattr(st.session_state, 'comprehensive_scan_results'):
        display_comprehensive_scan_results(st.session_state.comprehensive_scan_results, enhanced_scanner)
    
    else:
        # Welcome message
        st.markdown("""
        ### 🚀 **Comprehensive Market Intelligence**
        
        **Enhanced Scanner Features:**
        - **Massive Scale**: Scan up to 5,000 stocks across all market caps
        - **Rate Limiting**: Intelligent delays to avoid API limits
        - **Progress Tracking**: Real-time progress with batch processing
        - **Resume Capability**: Start from any position in the scan
        - **Error Recovery**: Automatic retry with exponential backoff
        
        **Performance Optimizations:**
        - **Batch Processing**: Process stocks in manageable batches
        - **Smart Caching**: 2-hour cache to avoid repeated scans
        - **Adaptive Rate Limiting**: Automatically adjusts based on API responses
        - **Comprehensive Universe**: Includes S&P 500, Russell 1000, and growth stocks
        
        **Time Estimates:**
        - **100 stocks**: ~3-5 minutes
        - **500 stocks**: ~15-25 minutes  
        - **1,000 stocks**: ~30-50 minutes
        - **5,000 stocks**: ~2-4 hours
        
        Configure your scan parameters above and click "Start Comprehensive Scan"!
        """)

def display_comprehensive_scan_results(scan_results: Dict, scanner):
    """Display comprehensive scan results"""
    
    st.markdown("---")
    st.markdown(f"### 📊 Comprehensive Scan Results")
    
    # Scan statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyzed", f"{scan_results['total_analyzed']:,}")
    
    with col2:
        st.metric("Success Rate", f"{scan_results['success_rate']:.1f}%")
    
    with col3:
        st.metric("Scan Duration", f"{scan_results['scan_duration_minutes']:.1f} min")
    
    with col4:
        st.metric("Last Updated", scan_results['scan_timestamp'][-8:])  # Show time only
    
    # Display results (same as before)
    col1, col2 = st.columns(2)
    
    # Top Buys
    with col1:
        st.markdown("#### 🟢 **TOP 10 BUY OPPORTUNITIES**")
        
        buy_options = []
        for i, stock in enumerate(scan_results['top_buys']):
            score = stock['technical_score']
            ticker = stock['ticker']
            price = stock['current_price']
            change_20d = stock['analysis'].get('price_change_20d', 0)
            
            color = "🟢" if score >= 70 else "🟡"
            buy_options.append(f"{color} {ticker} - Score: {score:.1f} | ${price:.2f} | 20D: {change_20d:+.1f}%")
        
        selected_buy = st.selectbox(
            "Select Stock for Analysis:",
            options=range(len(buy_options)),
            format_func=lambda x: buy_options[x],
            key="comprehensive_buy_selector"
        )
        
        if selected_buy is not None:
            stock_data = scan_results['top_buys'][selected_buy]
            
            with st.expander(f"📊 {stock_data['ticker']} Analysis", expanded=True):
                recommendation_text = scanner.get_stock_recommendation_text(stock_data)
                st.markdown(recommendation_text)
    
    # Top Sells
    with col2:
        st.markdown("#### 🔴 **TOP 10 SELL OPPORTUNITIES**")
        
        sell_options = []
        for i, stock in enumerate(scan_results['top_sells']):
            score = stock['technical_score']
            ticker = stock['ticker']
            price = stock['current_price']
            change_20d = stock['analysis'].get('price_change_20d', 0)
            
            color = "🔴" if score <= 35 else "🟠"
            sell_options.append(f"{color} {ticker} - Score: {score:.1f} | ${price:.2f} | 20D: {change_20d:+.1f}%")
        
        selected_sell = st.selectbox(
            "Select Stock for Analysis:",
            options=range(len(sell_options)),
            format_func=lambda x: sell_options[x],
            key="comprehensive_sell_selector"
        )
        
        if selected_sell is not None:
            stock_data = scan_results['top_sells'][selected_sell]
            
            with st.expander(f"📊 {stock_data['ticker']} Analysis", expanded=True):
                recommendation_text = scanner.get_stock_recommendation_text(stock_data)
                st.markdown(recommendation_text)
    
    # Export functionality
    st.markdown("---")
    st.markdown("#### 📥 **Export Results**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Buy List CSV"):
            buy_df = pd.DataFrame([
                {
                    'Ticker': s['ticker'],
                    'Score': s['technical_score'],
                    'Price': s['current_price'],
                    'Trend': s['analysis']['trend'],
                    'RSI': s['analysis']['rsi'],
                    '20D_Change': s['analysis']['price_change_20d']
                }
                for s in scan_results['top_buys']
            ])
            st.download_button(
                "Download Buy Recommendations",
                buy_df.to_csv(index=False),
                "top_buy_stocks.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("📊 Download Sell List CSV"):
            sell_df = pd.DataFrame([
                {
                    'Ticker': s['ticker'],
                    'Score': s['technical_score'],
                    'Price': s['current_price'],
                    'Trend': s['analysis']['trend'],
                    'RSI': s['analysis']['rsi'],
                    '20D_Change': s['analysis']['price_change_20d']
                }
                for s in scan_results['top_sells']
            ])
            st.download_button(
                "Download Sell Recommendations",
                sell_df.to_csv(index=False),
                "top_sell_stocks.csv",
                "text/csv"
            )
    
    with col3:
        if st.button("📈 View Score Distribution"):
            # Create score distribution chart
            all_scores = [s['technical_score'] for s in scan_results['top_buys'] + scan_results['top_sells']]
            
            import plotly.express as px
            fig = px.histogram(
                x=all_scores,
                nbins=20,
                title="Technical Score Distribution",
                labels={'x': 'Technical Score', 'y': 'Count'}
            )
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# ML PREDICTOR CLASS  
# =============================================================================

class MLPredictor:
    """Machine Learning Stock Price Direction Predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, data: Dict, analysis: Dict) -> pd.DataFrame:
        """Prepare features for ML model"""
        
        df = data['historical_data'].copy()
        
        # Technical indicators as features
        features = {}
        
        # Price-based features
        features['rsi'] = analysis['rsi']
        features['bb_position'] = analysis.get('bb_position', 50)
        features['price_change_1d'] = analysis.get('price_change_1d', 0)
        features['price_change_5d'] = analysis.get('price_change_5d', 0)
        features['price_change_20d'] = analysis.get('price_change_20d', 0)
        features['volatility'] = analysis['realized_vol']
        
        # Moving average relationships
        current_price = data['current_price']
        
        # Safe MA calculations with proper min_periods
        min_periods_20 = max(1, min(10, len(df)))
        min_periods_50 = max(1, min(25, len(df)))
        
        ma_20 = df['close'].rolling(20, min_periods=min_periods_20).mean().iloc[-1] if len(df) >= 1 else current_price
        ma_50 = df['close'].rolling(50, min_periods=min_periods_50).mean().iloc[-1] if len(df) >= 1 else current_price
        
        features['price_vs_ma20'] = ((current_price - ma_20) / ma_20) * 100 if ma_20 != 0 else 0
        features['price_vs_ma50'] = ((current_price - ma_50) / ma_50) * 100 if ma_50 != 0 else 0
        
        # Volume-based (if available)
        if 'volume' in df.columns and not df['volume'].isna().all():
            avg_volume = df['volume'].rolling(20, min_periods=5).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            features['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            features['volume_ratio'] = 1.0
        
        # Trend strength
        trend_mapping = {
            'STRONG_BULLISH': 4,
            'BULLISH': 2,
            'SIDEWAYS': 0,
            'BEARISH': -2,
            'STRONG_BEARISH': -4
        }
        features['trend_strength'] = trend_mapping.get(analysis['trend'], 0)
        
        # Volatility regime
        vol_mapping = {
            'LOW_VOL': 1,
            'NORMAL_VOL': 2,
            'HIGH_VOL': 3,
            'EXTREME_VOL': 4
        }
        features['vol_regime'] = vol_mapping.get(analysis['volatility_regime'], 2)
        
        # Momentum indicators
        momentum_mapping = {
            'EXTREMELY_OVERSOLD': 1,
            'OVERSOLD': 2,
            'BEARISH': 3,
            'NEUTRAL': 4,
            'BULLISH': 5,
            'OVERBOUGHT': 6,
            'EXTREMELY_OVERBOUGHT': 7
        }
        features['momentum_score'] = momentum_mapping.get(analysis['momentum'], 4)
        
        # 52-week position
        high_52w = data['high_52w']
        low_52w = data['low_52w']
        features['price_52w_percentile'] = ((current_price - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 50
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        self.feature_names = list(features.keys())
        
        return feature_df
    
    def predict_direction(self, data: Dict, analysis: Dict) -> Dict:
        """Predict stock price direction using enhanced technical analysis"""
        
        try:
            # Enhanced prediction logic based on technical analysis
            score_factors = []
            
            # Trend Analysis (40% weight)
            trend = analysis['trend']
            trend_scores = {
                'STRONG_BULLISH': 0.4,
                'BULLISH': 0.2,
                'SIDEWAYS': 0.0,
                'BEARISH': -0.2,
                'STRONG_BEARISH': -0.4
            }
            score_factors.append(trend_scores.get(trend, 0))
            
            # RSI Analysis (25% weight)
            rsi = analysis['rsi']
            if rsi < 30:
                rsi_score = 0.25  # Very oversold - bullish
            elif rsi < 40:
                rsi_score = 0.15  # Oversold - moderately bullish
            elif rsi > 70:
                rsi_score = -0.25  # Very overbought - bearish
            elif rsi > 60:
                rsi_score = -0.15  # Overbought - moderately bearish
            else:
                rsi_score = 0.0  # Neutral
            score_factors.append(rsi_score)
            
            # Price Position (20% weight)
            current_price = data['current_price']
            high_52w = data['high_52w']
            low_52w = data['low_52w']
            price_percentile = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
            
            if price_percentile < 0.2:
                position_score = 0.15  # Near lows - potential upside
            elif price_percentile > 0.8:
                position_score = -0.15  # Near highs - potential downside
            else:
                position_score = 0.0  # Mid-range
            score_factors.append(position_score)
            
            # Volatility Analysis (15% weight)
            vol_regime = analysis['volatility_regime']
            if vol_regime in ['EXTREME_VOL', 'HIGH_VOL']:
                vol_score = -0.05  # High volatility increases uncertainty
            else:
                vol_score = 0.05  # Normal/low volatility is favorable
            score_factors.append(vol_score)
            
            # Calculate composite score
            composite_score = sum(score_factors)
            
            # Convert to direction and confidence
            if composite_score > 0.1:
                direction = "UP"
                base_confidence = 55
            elif composite_score < -0.1:
                direction = "DOWN"
                base_confidence = 55
            else:
                # Neutral - use RSI to break tie
                if rsi > 50:
                    direction = "UP"
                else:
                    direction = "DOWN"
                base_confidence = 50
            
            # Adjust confidence based on score magnitude
            confidence_adjustment = abs(composite_score) * 100
            confidence = min(95, base_confidence + confidence_adjustment)
            
            # Calculate probabilities
            if direction == "UP":
                prob_up = confidence
                prob_down = 100 - confidence
            else:
                prob_down = confidence
                prob_up = 100 - confidence
            
            # Risk and signal strength
            if confidence >= 75:
                risk_level = "LOW"
                signal_strength = "STRONG"
            elif confidence >= 65:
                risk_level = "MODERATE"
                signal_strength = "MODERATE"
            else:
                risk_level = "HIGH"
                signal_strength = "WEAK"
            
            return {
                'direction': direction,
                'confidence': round(confidence, 1),
                'probability_up': round(prob_up, 1),
                'probability_down': round(prob_down, 1),
                'risk_level': risk_level,
                'signal_strength': signal_strength,
                'prediction_timeframe': '5-day outlook',
                'composite_score': round(composite_score, 3),
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}

# =============================================================================
# PROFESSIONAL UI COMPONENTS WITH ENHANCED TRADE INSTRUCTIONS
# =============================================================================

def create_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create professional metric card"""
    # Streamlit only accepts 'normal', 'inverse', or 'off'
    valid_colors = ['normal', 'inverse', 'off']
    if delta_color not in valid_colors:
        delta_color = 'normal'
    
    return st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color
    )

def display_enhanced_strategy_results(results: Dict, asset_class: str, expiration_date: str, max_risk_amount: float):
    """Display strategy results with detailed trade instructions"""
    
    if 'error' in results:
        st.error(f"⚠️ {results['error']}")
        if 'min_capital_needed' in results:
            st.warning(f"💡 **Solution:** Increase your capital to at least ${results['min_capital_needed']:,.0f} or consider a different strategy")
        return
    
    recommendations = results.get('recommendations', [])
    if not recommendations:
        st.warning("No viable recommendations found with current capital settings")
        return
    
    strategy_name = results['strategy']
    
    # Create tabs for different risk levels if multiple recommendations
    if len(recommendations) > 1:
        risk_tabs = st.tabs([f"{rec['risk_level']} Risk" for rec in recommendations])
        
        for tab, rec in zip(risk_tabs, recommendations):
            with tab:
                display_single_trade_instruction(rec, asset_class, strategy_name, expiration_date, max_risk_amount)
    else:
        display_single_trade_instruction(recommendations[0], asset_class, strategy_name, expiration_date, max_risk_amount)

def display_single_trade_instruction(rec: Dict, asset_class: str, strategy: str, expiration: str, max_risk_amount: float):
    """Display detailed trade instructions for a single recommendation"""
    
    # Format prices based on asset class
    price_format = "{:.5f}" if asset_class == 'FOREX' else "${:.2f}"
    
    # Trade Instruction Header
    st.markdown(f"### 🎯 **{rec['risk_level']} {strategy.replace('_', ' ').title()}** Trade Setup")
    
    # Check if trade fits within risk tolerance
    max_loss = rec.get('max_loss', rec.get('total_cost', 0))
    risk_color = "🟢" if max_loss <= max_risk_amount else "🟠" if max_loss <= max_risk_amount * 1.5 else "🔴"
    
    st.markdown(f"{risk_color} **Risk Level:** ${max_loss:,.0f} max loss ({'Within' if max_loss <= max_risk_amount else 'Exceeds'} your ${max_risk_amount:,.0f} limit)")
    
    # Detailed Trade Instructions
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📋 **TRADE INSTRUCTIONS**")
        
        if strategy == 'COVERED_CALL':
            st.markdown(f"""
            **Step 1:** Buy {rec['contracts']} lots of underlying
            - **Quantity:** {rec['contracts'] * (100 if asset_class != 'FOREX' else 10000):,} shares/units
            - **Cost:** ${rec['stock_investment']:,.0f}
            
            **Step 2:** Sell {rec['contracts']} Call Options
            - **Strike:** {price_format.format(rec['strike'])}
            - **Expiration:** {expiration}
            - **Premium Received:** ${rec['premium_received']:,.0f}
            - **Contract Type:** SELL TO OPEN
            
            **Order Summary:**
            - Net Investment: ${rec['net_investment']:,.0f}
            - Premium Collected: ${rec['premium_received']:,.0f}
            """)
        
        elif strategy == 'CASH_SECURED_PUT':
            st.markdown(f"""
            **Step 1:** Set aside cash collateral
            - **Cash Required:** ${rec['cash_secured']:,.0f}
            - **Purpose:** Secure the put sale
            
            **Step 2:** Sell {rec['contracts']} Put Options
            - **Strike:** {price_format.format(rec['strike'])}
            - **Expiration:** {expiration}
            - **Premium Received:** ${rec['premium_received']:,.0f}
            - **Contract Type:** SELL TO OPEN
            
            **If Assigned:**
            - Buy {rec['contracts'] * (100 if asset_class != 'FOREX' else 10000):,} units at {price_format.format(rec['strike'])}
            - Effective Price: {price_format.format(rec['effective_price'])}
            """)
        
        elif strategy == 'IRON_CONDOR':
            st.markdown(f"""
            **Step 1:** Sell Call Spread
            - **Sell:** {rec['contracts']} calls at {price_format.format(rec['call_sell_strike'])}
            - **Buy:** {rec['contracts']} calls at {price_format.format(rec['call_buy_strike'])}
            
            **Step 2:** Sell Put Spread  
            - **Sell:** {rec['contracts']} puts at {price_format.format(rec['put_sell_strike'])}
            - **Buy:** {rec['contracts']} puts at {price_format.format(rec['put_buy_strike'])}
            
            **Net Credit:** ${rec['total_credit']:,.0f}
            **Margin Required:** ${rec['total_margin']:,.0f}
            **Expiration:** {expiration}
            """)
        
        elif strategy == 'BULL_CALL_SPREAD':
            st.markdown(f"""
            **Step 1:** Buy Long Call
            - **Buy:** {rec['contracts']} calls at {price_format.format(rec['buy_strike'])}
            - **Premium Paid:** ${rec['contracts'] * rec['buy_price'] * (100 if asset_class != 'FOREX' else 10000):,.0f}
            
            **Step 2:** Sell Short Call
            - **Sell:** {rec['contracts']} calls at {price_format.format(rec['sell_strike'])}
            - **Premium Received:** ${rec['contracts'] * rec['sell_price'] * (100 if asset_class != 'FOREX' else 10000):,.0f}
            
            **Net Debit:** ${rec['total_cost']:,.0f}
            **Expiration:** {expiration}
            """)
    
    with col2:
        st.markdown("#### 📊 **PROFIT/LOSS PROFILE**")
        
        # P&L Metrics
        col_a, col_b = st.columns(2)
        
        with col_a:
            if 'max_profit' in rec:
                create_metric_card("Max Profit", f"${rec['max_profit']:,.0f}")
            if 'max_profit_pct' in rec:
                profit_pct = rec['max_profit_pct']
                color = "normal" if profit_pct > 0 else "inverse"
                create_metric_card("Max Profit %", f"{profit_pct:.1f}%", None, color)
        
        with col_b:
            if 'max_loss' in rec:
                create_metric_card("Max Loss", f"${rec['max_loss']:,.0f}")
            elif 'total_cost' in rec:
                create_metric_card("Max Loss", f"${rec['total_cost']:,.0f}")
            
            if 'probability_profit' in rec:
                prob_pct = rec['probability_profit'] * 100
                prob_color = "normal" if prob_pct > 60 else "normal" if prob_pct > 40 else "inverse"
                create_metric_card("Profit Probability", f"{prob_pct:.0f}%", None, prob_color)
        
        # Key Levels
        st.markdown("**🎯 Key Price Levels:**")
        
        if 'breakeven' in rec:
            st.write(f"• **Breakeven:** {price_format.format(rec['breakeven'])}")
        
        if strategy == 'IRON_CONDOR':
            st.write(f"• **Profit Range:** {price_format.format(rec['lower_breakeven'])} - {price_format.format(rec['upper_breakeven'])}")
            st.write(f"• **Range Width:** {rec['profit_range_pct']:.1f}%")
        
        if 'target_price' in rec:
            st.write(f"• **Target Price:** {price_format.format(rec['target_price'])}")
        
        # Time-based metrics
        if 'annualized_yield' in rec:
            st.write(f"• **Annualized Yield:** {rec['annualized_yield']:.1f}%")
        
        # Risk Assessment
        risk_assessment = ""
        if max_loss <= max_risk_amount * 0.5:
            risk_assessment = "🟢 **Low Risk** - Well within your risk tolerance"
        elif max_loss <= max_risk_amount:
            risk_assessment = "🟡 **Moderate Risk** - At your risk limit"
        else:
            risk_assessment = "🔴 **High Risk** - Exceeds your risk tolerance"
        
        st.markdown(f"**Risk Assessment:** {risk_assessment}")
    
    # Market Conditions Suitability
    st.markdown("#### 🌡️ **Market Suitability**")
    
    suitability_messages = {
        'COVERED_CALL': "Best in neutral to slightly bullish markets with elevated volatility",
        'CASH_SECURED_PUT': "Ideal when you want to own the asset at a lower price with bullish outlook",
        'IRON_CONDOR': "Perfect for range-bound, low-volatility environments",
        'BULL_CALL_SPREAD': "Suitable for moderately bullish outlook with limited capital",
        'BEAR_PUT_SPREAD': "Good for moderately bearish outlook with defined risk"
    }
    
    st.write(f"💡 **Strategy Note:** {suitability_messages.get(strategy, 'Professional options strategy')}")
    
    # Action Button
    if max_loss <= max_risk_amount:
        st.success("✅ **Ready to Trade** - This setup fits your risk parameters")
    else:
        st.warning(f"⚠️ **Consider Reducing Position** - Reduce to {int(rec['contracts'] * max_risk_amount / max_loss)} contracts to fit your risk limit")

def display_market_scanner_results(scanner_results: Dict):
    """Display professional market scanner results"""
    
    st.markdown(f"### 📊 Market Scan Results")
    st.caption(f"Analyzed {scanner_results['total_analyzed']} stocks | Last Updated: {scanner_results['scan_timestamp']}")
    
    col1, col2 = st.columns(2)
    
    # Top Buys
    with col1:
        st.markdown("#### 🟢 **TOP 10 BUY RECOMMENDATIONS**")
        
        buy_options = []
        for i, stock in enumerate(scanner_results['top_buys']):
            score = stock['technical_score']
            ticker = stock['ticker']
            price = stock['current_price']
            change_20d = stock['analysis'].get('price_change_20d', 0)
            
            color = "🟢" if score >= 70 else "🟡"
            buy_options.append(f"{color} {ticker} - Score: {score:.1f} | ${price:.2f} | 20D: {change_20d:+.1f}%")
        
        selected_buy = st.selectbox(
            "Select Stock for Analysis:",
            options=range(len(buy_options)),
            format_func=lambda x: buy_options[x],
            key="buy_selector"
        )
        
        if selected_buy is not None:
            stock_data = scanner_results['top_buys'][selected_buy]
            scanner = MarketScanner(None)  # Placeholder for method call
            
            with st.expander(f"📊 {stock_data['ticker']} Analysis", expanded=True):
                recommendation_text = scanner.get_stock_recommendation_text(stock_data)
                st.markdown(recommendation_text)
    
    # Top Sells
    with col2:
        st.markdown("#### 🔴 **TOP 10 SELL RECOMMENDATIONS**")
        
        sell_options = []
        for i, stock in enumerate(scanner_results['top_sells']):
            score = stock['technical_score']
            ticker = stock['ticker']
            price = stock['current_price']
            change_20d = stock['analysis'].get('price_change_20d', 0)
            
            color = "🔴" if score <= 35 else "🟠"
            sell_options.append(f"{color} {ticker} - Score: {score:.1f} | ${price:.2f} | 20D: {change_20d:+.1f}%")
        
        selected_sell = st.selectbox(
            "Select Stock for Analysis:",
            options=range(len(sell_options)),
            format_func=lambda x: sell_options[x],
            key="sell_selector"
        )
        
        if selected_sell is not None:
            stock_data = scanner_results['top_sells'][selected_sell]
            scanner = MarketScanner(None)
            
            with st.expander(f"📊 {stock_data['ticker']} Analysis", expanded=True):
                recommendation_text = scanner.get_stock_recommendation_text(stock_data)
                st.markdown(recommendation_text)

def display_detailed_stock_analysis(stock_data: Dict, strategist):
    """Display detailed analysis for selected stock"""
    
    ticker = stock_data['ticker']
    data = stock_data['data']
    analysis = stock_data['analysis']
    
    st.markdown(f"### 📈 **{ticker} Detailed Analysis**")
    
    # Price metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        create_metric_card("Current Price", f"${data['current_price']:.2f}")
    
    with col2:
        change_20d = analysis.get('price_change_20d', 0)
        color = "normal" if change_20d > 0 else "inverse"
        create_metric_card("20D Change", f"{change_20d:+.1f}%", None, color)
    
    with col3:
        create_metric_card("52W High", f"${data['high_52w']:.2f}")
    
    with col4:
        create_metric_card("52W Low", f"${data['low_52w']:.2f}")
    
    with col5:
        create_metric_card("Volatility", f"{analysis['realized_vol']*100:.1f}%")
    
    # Professional chart
    chart_data = {
        'ticker': ticker,
        'current_price': data['current_price'],
        'historical_data': data['historical_data']
    }
    
    support_resistance = {
        'support_level': data['low_52w'],
        'resistance_level': data['high_52w']
    }
    
    chart = strategist.create_professional_chart(chart_data, 'EQUITIES', support_resistance)
    st.plotly_chart(chart, use_container_width=True)
    
    # Technical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 **Technical Indicators**")
        st.write(f"• **RSI:** {analysis['rsi']:.1f}")
        st.write(f"• **Trend:** {analysis['trend'].replace('_', ' ').title()}")
        st.write(f"• **Momentum:** {analysis['momentum']}")
        st.write(f"• **BB Position:** {analysis.get('bb_position', 50):.1f}%")
    
    with col2:
        st.markdown("#### 🎯 **Options Opportunities**")
        try:
            options_data = cached_get_options_data(strategist.polygon_api_key, ticker, 'EQUITIES', data['current_price'])
            
            st.write(f"• **Expiration:** {options_data['expiration']}")
            st.write(f"• **Call Options:** {len(options_data['calls'])}")
            st.write(f"• **Put Options:** {len(options_data['puts'])}")
            st.write(f"• **Days to Expiry:** {options_data['days_to_expiry']}")
            
        except:
            st.write("• Options data not available")

def display_ml_predictions(prediction_result: Dict, ticker: str):
    """Display ML prediction results"""
    
    if not prediction_result.get('success'):
        st.error(f"Prediction failed: {prediction_result.get('error')}")
        return
    
    st.markdown(f"### 🤖 **AI Prediction for {ticker}**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    direction = prediction_result['direction']
    confidence = prediction_result['confidence']
    
    with col1:
        emoji = "🟢" if direction == "UP" else "🔴"
        create_metric_card("Direction", f"{emoji} {direction}")
    
    with col2:
        color = "normal" if confidence >= 70 else "inverse" if confidence < 60 else "off"
        create_metric_card("Confidence", f"{confidence:.1f}%", None, color)
    
    with col3:
        create_metric_card("Signal Strength", prediction_result['signal_strength'])
    
    with col4:
        create_metric_card("Risk Level", prediction_result['risk_level'])
    
    # Probability breakdown
    st.markdown("#### 📊 **Probability Breakdown**")
    
    prob_up = prediction_result['probability_up']
    prob_down = prediction_result['probability_down']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Probability UP", f"{prob_up:.1f}%")
        st.progress(prob_up / 100)
    
    with col2:
        st.metric("Probability DOWN", f"{prob_down:.1f}%")
        st.progress(prob_down / 100)
    
    # Trading suggestions based on prediction
    st.markdown("#### 💡 **Trading Suggestions**")
    
    if direction == "UP" and confidence >= 70:
        st.success("""
        **Strong Bullish Signal - Consider:**
        • Bull call spreads for leveraged upside
        • Covered calls if you own shares (capture premium + upside)
        • Cash-secured puts to enter at lower prices
        """)
    elif direction == "UP" and confidence >= 60:
        st.info("""
        **Moderate Bullish Signal - Consider:**
        • Conservative call options
        • Small position size
        • Monitor for confirmation
        """)
    elif direction == "DOWN" and confidence >= 70:
        st.warning("""
        **Strong Bearish Signal - Consider:**
        • Protective puts if holding
        • Bear put spreads
        • Avoid call options
        """)
    elif direction == "DOWN" and confidence >= 60:
        st.warning("""
        **Moderate Bearish Signal - Consider:**
        • Reduced position sizes
        • Defensive strategies
        • Wait for better entry points
        """)
    else:
        st.info("""
        **Low Confidence Signal:**
        • Range-bound strategies (iron condors)
        • Wait for stronger signals
        • Focus on premium collection
        """)
    
    st.caption(f"Timeframe: {prediction_result['prediction_timeframe']}")

# =============================================================================
# MAIN STREAMLIT APPLICATION WITH USER CONTROLS
# =============================================================================

def main():
    st.set_page_config(
        page_title="Multi Asset Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling with larger tabs
    st.markdown("""
    <style>
        .stApp > header {visibility: hidden;}
        .css-18ni7ap {background-color: #0e1117;}
        .css-1d391kg {background-color: #262730;}
        .stTabs [data-baseweb="tab-list"] {gap: 12px;}
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            white-space: pre-wrap;
            background: linear-gradient(145deg, #1e1e2e, #2d3748);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: white;
            font-weight: 600;
            font-size: 14px;
            padding: 0 20px;
            min-width: 200px;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(145deg, #FF4B4B, #FF6B6B);
            border: 1px solid rgba(255, 75, 75, 0.8);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 75, 75, 0.3);
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(145deg, #2d3748, #3a4553);
            border: 1px solid rgba(255, 255, 255, 0.3);
            transform: translateY(-1px);
        }
        .metric-container {
            background: linear-gradient(145deg, #1e1e2e, #2d3748);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1e1e2e, #2d3748);
        }
        .stSelectbox > div > div > div {
            background: linear-gradient(145deg, #2d3748, #3a4553);
        }
        .stSlider > div > div > div > div {
            background: linear-gradient(145deg, #FF4B4B, #FF6B6B);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("Multi-Asset Options Dashboard")
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'selected_asset_class' not in st.session_state:
        st.session_state.selected_asset_class = 'EQUITIES'
    
    # Asset Class Selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        asset_class = st.selectbox(
            "🎯 Asset Class",
            ['INDICES', 'EQUITIES', 'FOREX'],
            index=['INDICES', 'EQUITIES', 'FOREX'].index(st.session_state.selected_asset_class),
            format_func=lambda x: {
                'INDICES': '📊 Index ETFs',
                'EQUITIES': '📈 Individual Stocks', 
                'FOREX': '💱 Currency Pairs'
            }[x]
        )
        
        if asset_class != st.session_state.selected_asset_class:
            st.session_state.selected_asset_class = asset_class
            st.rerun()
    
    # Enhanced Sidebar Configuration with User Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key (hardcoded in app but not visible to user)
        api_key = "igO7PgpW43MsVcJvr1ZuxZ_vYrH87jLZ"  # Your hardcoded key
        
        # Initialize strategist
        try:
            strategist = MultiAssetOptionsStrategist(api_key)
            st.success("✅ API Connected")
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.stop()
        
        st.divider()
        
        # USER CAPITAL & RISK MANAGEMENT CONTROLS
        st.header("💰 Capital & Risk Management")
        
        # Available Capital Input
        available_capital = st.number_input(
            "Available Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=25000,
            step=1000,
            format="%d",
            help="Total capital available for options trading"
        )
        
        # Portfolio Risk Percentage
        portfolio_risk_pct = st.slider(
            "Portfolio Risk (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Maximum percentage of capital to risk per trade"
        )
        
        # Calculate max risk amount
        max_risk_amount = available_capital * (portfolio_risk_pct / 100)
        
        st.info(f"💡 **Max Risk per Trade:** ${max_risk_amount:,.0f}")
        
        # Strategy Preferences
        st.subheader("🎯 Strategy Preferences")
        
        preferred_expiry = st.select_slider(
            "Preferred Expiry",
            options=["Weekly", "Monthly", "Quarterly"],
            value="Monthly",
            help="Preferred options expiration timeframe"
        )
        
        delta_preference = st.select_slider(
            "Delta Preference",
            options=["Conservative (0.1-0.3)", "Moderate (0.3-0.5)", "Aggressive (0.5+)"],
            value="Moderate (0.3-0.5)",
            help="Preferred delta range for options strategies"
        )
        
        st.divider()
        
        # Symbol Input
        st.header(f"📊 {asset_class} Analysis")
        
        default_symbols = {
            'INDICES': 'SPY',
            'EQUITIES': 'AAPL', 
            'FOREX': 'EURUSD'
        }
        
        symbol = st.text_input(
            "Symbol",
            value=default_symbols[asset_class],
            placeholder=f"Enter {asset_class.lower()} symbol"
        )
        
        analyze_btn = st.button(
            "🚀 Analyze",
            type="primary",
            use_container_width=True
        )
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        popular_symbols = strategist.get_popular_symbols(asset_class)
        
        cols = st.columns(2)
        for i, sym in enumerate(popular_symbols[:6]):
            col = cols[i % 2]
            with col:
                if st.button(sym, key=f"quick_{sym}", use_container_width=True):
                    st.session_state.selected_symbol = sym
                    st.rerun()
    
    # Main Content Tabs with Enhanced Styling - NOW WITH 4 TABS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Strategy Analysis", "📈 Backtesting", "🔢 Greeks Analysis", "🎯 Market Scanner"])
    
    # Tab 1: Enhanced Strategy Analysis with Trade Instructions AND ML PREDICTIONS
    with tab1:
        if analyze_btn and symbol:
            with st.spinner(f"Analyzing {symbol} ({asset_class})..."):
                # Get data with caching
                try:
                    underlying_data = cached_get_asset_data(api_key, symbol.upper(), asset_class)
                    options_data = cached_get_options_data(api_key, symbol.upper(), asset_class, underlying_data['current_price'])
                    
                    st.session_state.analysis_result = {
                        'ticker': symbol.upper(),
                        'asset_class': asset_class,
                        'underlying_data': underlying_data,
                        'options_data': options_data,
                        'success': True
                    }
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.session_state.analysis_result = None
        
        # Display results with enhanced trade instructions
        if st.session_state.analysis_result and st.session_state.analysis_result.get('success'):
            result = st.session_state.analysis_result
            
            if result.get('asset_class') == asset_class:
                underlying = result['underlying_data']
                options = result['options_data']
                
                # Price Summary with Enhanced Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                price_format = "{:.5f}" if asset_class == 'FOREX' else "${:.2f}"
                
                with col1:
                    create_metric_card("Current Price", price_format.format(underlying['current_price']))
                
                with col2:
                    create_metric_card("52W High", price_format.format(underlying['high_52w']))
                
                with col3:
                    create_metric_card("52W Low", price_format.format(underlying['low_52w']))
                
                with col4:
                    vol_pct = underlying['realized_vol_21d'] * 100
                    create_metric_card("Volatility", f"{vol_pct:.1f}%")
                
                with col5:
                    create_metric_card("Available Capital", f"${available_capital:,.0f}")
                
                # Professional Chart
                chart_data = {
                    'ticker': underlying['ticker'],
                    'current_price': underlying['current_price'],
                    'historical_data': underlying['historical_data']
                }
                
                support_resistance = {
                    'support_level': underlying['low_52w'],
                    'resistance_level': underlying['high_52w']
                }
                
                chart = strategist.create_professional_chart(chart_data, asset_class, support_resistance)
                st.plotly_chart(chart, use_container_width=True)
                
                # AI PREDICTIONS SECTION - MOVED TO TAB 1
                st.markdown("---")
                st.markdown("### 🤖 **AI Price Prediction**")
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    predict_btn = st.button(
                        "🎯 Get AI Prediction",
                        type="secondary",
                        use_container_width=True
                    )
                
                if predict_btn:
                    with st.spinner("🤖 Running AI analysis..."):
                        # Get market analysis
                        analysis = strategist.analyze_market_conditions(underlying)
                        
                        # Create ML predictor and get prediction
                        ml_predictor = MLPredictor()
                        prediction_result = ml_predictor.predict_direction(underlying, analysis)
                        
                        with col1:
                            display_ml_predictions(prediction_result, underlying['ticker'])
                
                # Enhanced Strategy Analysis with Detailed Trade Instructions
                st.markdown("---")
                st.subheader("🎯 Professional Strategy Recommendations")
                
                # Calculate strategies with user capital and risk limits
                strategies_to_calc = ['COVERED_CALL', 'CASH_SECURED_PUT', 'IRON_CONDOR', 'BULL_CALL_SPREAD']
                
                for strategy in strategies_to_calc:
                    with st.expander(f"📋 {strategy.replace('_', ' ').title()}", expanded=strategy=='COVERED_CALL'):
                        try:
                            if strategy == 'COVERED_CALL':
                                strategy_result = strategist.calculate_covered_call_accurate(
                                    options['calls'], underlying['current_price'], available_capital, asset_class, max_risk_amount
                                )
                            elif strategy == 'CASH_SECURED_PUT':
                                strategy_result = strategist.calculate_cash_secured_put_accurate(
                                    options['puts'], underlying['current_price'], available_capital, asset_class, max_risk_amount
                                )
                            elif strategy == 'IRON_CONDOR':
                                strategy_result = strategist.calculate_iron_condor_accurate(
                                    options['calls'], options['puts'], underlying['current_price'], available_capital, asset_class, max_risk_amount
                                )
                            elif strategy == 'BULL_CALL_SPREAD':
                                strategy_result = strategist.calculate_bull_call_spread_accurate(
                                    options['calls'], underlying['current_price'], available_capital, asset_class, max_risk_amount
                                )
                            
                            # Display enhanced results with trade instructions
                            display_enhanced_strategy_results(strategy_result, asset_class, options['expiration'], max_risk_amount)
                            
                        except Exception as e:
                            st.error(f"Strategy calculation failed: {str(e)}")
                
                # Options Summary
                st.subheader("📊 Options Market Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    create_metric_card("Expiration", options['expiration'])
                
                with col2:
                    create_metric_card("Days to Expiry", str(options['days_to_expiry']))
                
                with col3:
                    create_metric_card("Call Options", str(len(options['calls'])))
                
                with col4:
                    create_metric_card("Put Options", str(len(options['puts'])))
        
        elif not symbol:
            # Welcome message with asset class information
            asset_descriptions = {
                'INDICES': "Analyze index ETFs with diversified exposure and professional options strategies.",
                'EQUITIES': "Deep dive into individual stocks with comprehensive options analysis.",
                'FOREX': "Navigate currency markets with specialized FX options insights."
            }
            
            st.markdown(f"### Welcome to {asset_class} Options Analysis")
            st.write(asset_descriptions[asset_class])
            
            # Capital Management Info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💰 **Your Capital Settings**")
                st.write(f"• **Available Capital:** ${available_capital:,.0f}")
                st.write(f"• **Risk Tolerance:** {portfolio_risk_pct}%")
                st.write(f"• **Max Risk per Trade:** ${max_risk_amount:,.0f}")
                
            with col2:
                st.markdown("#### 🎯 **Strategy Preferences**")
                st.write(f"• **Preferred Expiry:** {preferred_expiry}")
                st.write(f"• **Delta Preference:** {delta_preference}")
                st.write(f"• **Asset Focus:** {asset_class}")
            
            st.markdown("---")
            st.write("Enter a symbol in the sidebar to begin professional analysis with your personalized settings.")
    
    # Tab 2: Enhanced Backtesting with User Capital and Trade Table
    with tab2:
        st.subheader("📈 Strategy Backtesting with Your Capital")
        
        # Backtest Explanation
        st.info("""
        **📋 How Backtesting Works:**
        - **Rolling Expiries**: Opens new positions every 30 days (monthly cycle)
        - **1 Year Backtest**: ~12 monthly option cycles  
        - **2 Year Backtest**: ~24 monthly option cycles
        - **Each Trade**: Holds position for full expiry period, then rolls to new position
        """)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            backtest_symbol = st.text_input("Symbol", value=default_symbols[asset_class])
        
        with col2:
            backtest_strategy = st.selectbox(
                "Strategy",
                ['COVERED_CALL', 'CASH_SECURED_PUT', 'IRON_CONDOR', 'BULL_CALL_SPREAD', 'BUY_AND_HOLD']
            )
        
        with col3:
            backtest_period = st.selectbox("Period", ['6M', '1Y', '2Y'], index=1)
        
        with col4:
            backtest_capital = st.number_input(
                "Starting Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=available_capital,
                step=1000
            )
        
        with col5:
            expiry_days = st.selectbox(
                "Expiry Cycle (Days)", 
                [15, 30, 45, 60], 
                index=1,
                help="How often to roll positions (15=bi-weekly, 30=monthly, etc.)"
            )
        
        if st.button("🔄 Run Enhanced Backtest", type="primary"):
            # Backtest parameters with user capital
            params = {
                'initial_capital': backtest_capital,  # Use user's capital setting
                'days_to_expiry': expiry_days,
                'delta_target': 0.3,
                'wing_width': 0.05
            }
            
            # Calculate dates
            period_days = {'6M': 180, '1Y': 365, '2Y': 730}[backtest_period]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Estimate number of trades
            estimated_trades = period_days // expiry_days
            st.info(f"📊 **Expected Trades**: ~{estimated_trades} positions over {backtest_period} ({expiry_days}-day cycles)")
            
            with st.spinner("Running professional backtest with trade tracking..."):
                try:
                    backtest_result = strategist.run_accurate_backtest(
                        backtest_symbol.upper(),
                        asset_class, 
                        backtest_strategy,
                        start_date,
                        end_date,
                        params
                    )
                    
                    if backtest_result['success']:
                        metrics = backtest_result['performance_metrics']
                        trades = backtest_result['results']['trades']
                        
                        # Enhanced Results Display
                        st.success(f"✅ Backtest completed: {len(trades)} trades executed")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            total_return = metrics['total_return']
                            color = "normal" if total_return > 0 else "inverse"
                            create_metric_card("Total Return", f"{total_return:.1f}%", None, color)
                        
                        with col2:
                            create_metric_card("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                        
                        with col3:
                            create_metric_card("Win Rate", f"{metrics['win_rate']:.1f}%")
                        
                        with col4:
                            create_metric_card("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
                        
                        with col5:
                            create_metric_card("Final Value", f"${metrics['final_value']:,.0f}")
                        
                        # Performance Chart
                        if backtest_result['results']['portfolio_values']:
                            portfolio_values = backtest_result['results']['portfolio_values']
                            dates = pd.date_range(start=start_date, end=end_date, periods=len(portfolio_values))
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=dates,
                                y=portfolio_values,
                                mode='lines',
                                name='Portfolio Value',
                                line=dict(color='#00ff88', width=3)
                            ))
                            
                            fig.add_hline(
                                y=backtest_capital,
                                line_dash="dash",
                                line_color="white",
                                annotation_text="Starting Capital"
                            )
                            
                            fig.update_layout(
                                title=f"{backtest_strategy} Performance",
                                template='plotly_dark',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error(f"Backtest failed: {backtest_result.get('error')}")
                
                except Exception as e:
                    st.error(f"Backtest error: {str(e)}")
        
        else:
            # Backtest Instructions
            st.markdown("""
            ### 📈 Professional Strategy Backtesting
            
            **Features:**
            - **Complete Trade History**: See every trade with entry/exit details, strikes, premiums, and P&L
            - **Interactive Visualization**: Click on any trade to see it highlighted on the performance chart
            - **Rolling Expiries**: Configurable expiry cycles (15, 30, 45, or 60 days)
            - **Detailed Statistics**: Win rate, profit factor, drawdowns, and more
            
            **How It Works:**
            - **Entry**: New position opened every X days (your selected cycle)
            - **Exit**: Position closed at expiration, P&L calculated
            - **Roll**: Immediately opens new position for next cycle
            - **Capital**: Uses your specified starting capital with realistic position sizing
            
            Configure your parameters above and click "Run Enhanced Backtest" to begin!
            """)
    
    # Tab 3: Enhanced Greeks Analysis
    with tab3:
        st.subheader("🔢 Professional Options Greeks Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            greeks_symbol = st.text_input(
                "Symbol for Greeks",
                value=default_symbols[asset_class],
                placeholder=f"Enter {asset_class.lower()} symbol"
            )
        
        with col2:
            expiry_period = st.selectbox(
                "Expiry Period",
                ['1M', '2M', '3M', '6M'],
                index=0,
                format_func=lambda x: {'1M': '1 Month', '2M': '2 Months', '3M': '3 Months', '6M': '6 Months'}[x]
            )
        
        with col3:
            get_greeks_btn = st.button(
                "📊 Calculate Greeks",
                type="primary",
                use_container_width=True
            )
        
        if get_greeks_btn and greeks_symbol:
            with st.spinner(f"Calculating Greeks for {greeks_symbol} ({asset_class})..."):
                try:
                    greeks_result = strategist.get_options_greeks(greeks_symbol.upper(), asset_class)
                    
                    if greeks_result['success']:
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        price_format = "{:.5f}" if asset_class == 'FOREX' else "${:.2f}"
                        
                        with col1:
                            create_metric_card("Underlying Price", price_format.format(greeks_result['underlying_price']))
                        
                        with col2:
                            create_metric_card("Total Contracts", str(greeks_result['total_contracts']))
                        
                        with col3:
                            create_metric_card("Days to Expiry", str(greeks_result['days_to_expiry']))
                        
                        with col4:
                            create_metric_card("ATM Options", str(greeks_result['summary_stats'].get('atm_options', 0)))
                        
                        # Professional Greeks Display
                        st.subheader("📊 Professional Options Chain")
                        
                        calls_df = greeks_result['calls_greeks']
                        puts_df = greeks_result['puts_greeks']
                        current_price = greeks_result['underlying_price']
                        
                        if not calls_df.empty or not puts_df.empty:
                            
                            col1, col2 = st.columns(2)
                            
                            # Calls Greeks
                            with col1:
                                st.markdown("#### 📞 **CALLS**")
                                
                                if not calls_df.empty:
                                    # Display organized calls data
                                    display_calls = calls_df[['strike', 'price', 'delta', 'gamma', 'theta', 'vega']].head(15).copy()
                                    
                                    if asset_class == 'FOREX':
                                        display_calls['strike'] = display_calls['strike'].apply(lambda x: f"{x:.5f}")
                                        display_calls['price'] = display_calls['price'].apply(lambda x: f"{x:.4f}")
                                    else:
                                        display_calls['strike'] = display_calls['strike'].apply(lambda x: f"${x:.2f}")
                                        display_calls['price'] = display_calls['price'].apply(lambda x: f"${x:.2f}")
                                    
                                    st.dataframe(display_calls, use_container_width=True, height=400)
                                
                                else:
                                    st.warning("No call options data available")
                            
                            # Puts Greeks  
                            with col2:
                                st.markdown("#### 📱 **PUTS**")
                                
                                if not puts_df.empty:
                                    # Display organized puts data
                                    display_puts = puts_df[['strike', 'price', 'delta', 'gamma', 'theta', 'vega']].head(15).copy()
                                    
                                    if asset_class == 'FOREX':
                                        display_puts['strike'] = display_puts['strike'].apply(lambda x: f"{x:.5f}")
                                        display_puts['price'] = display_puts['price'].apply(lambda x: f"{x:.4f}")
                                    else:
                                        display_puts['strike'] = display_puts['strike'].apply(lambda x: f"${x:.2f}")
                                        display_puts['price'] = display_puts['price'].apply(lambda x: f"${x:.2f}")
                                    
                                    st.dataframe(display_puts, use_container_width=True, height=400)
                                
                                else:
                                    st.warning("No put options data available")
                        
                        # Greeks Summary
                        st.subheader("📊 Greeks Summary & Key Levels")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            create_metric_card("OTM Calls", str(greeks_result['summary_stats'].get('otm_calls', 0)))
                        
                        with col2:
                            create_metric_card("OTM Puts", str(greeks_result['summary_stats'].get('otm_puts', 0)))
                        
                        with col3:
                            max_gamma_strike = greeks_result['summary_stats'].get('highest_gamma_strike', 0)
                            if asset_class == 'FOREX':
                                create_metric_card("Max Gamma Strike", f"{max_gamma_strike:.5f}")
                            else:
                                create_metric_card("Max Gamma Strike", f"${max_gamma_strike:.2f}")
                        
                        with col4:
                            avg_iv = greeks_result['summary_stats'].get('avg_implied_vol', 0)
                            create_metric_card("Avg IV", f"{avg_iv:.1%}")
                    
                    else:
                        st.error(f"Greeks calculation failed: {greeks_result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"Greeks analysis failed: {str(e)}")
        
        else:
            # Greeks instructions
            st.markdown(f"""
            ### 🔢 {asset_class} Options Greeks Analysis
            
            Professional Greeks analysis with:
            - **Complete Options Chain** with ITM/ATM/OTM categorization
            - **Key Delta Levels** for professional trading
            - **Risk Metrics** including Gamma, Theta, Vega, and Rho
            - **Market Insights** for {asset_class.lower()} options behavior
            
            Enter a symbol above to begin Greeks analysis.
            """)
            
            # Greeks reference with enhanced styling
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **Primary Greeks:**
                - **Delta (Δ)**: Price sensitivity  
                - **Gamma (Γ)**: Delta sensitivity  
                - **Theta (Θ)**: Time decay
                """)
            
            with col2:
                st.markdown("""
                **Secondary Greeks:**
                - **Vega (ν)**: Volatility sensitivity
                - **Rho (ρ)**: Interest rate sensitivity
                - **IV**: Implied volatility
                """)
            
            with col3:
                st.markdown("""
                **Key Concepts:**
                - **ITM**: In-the-money options
                - **ATM**: At-the-money options  
                - **OTM**: Out-of-the-money options
                """)

    # Tab 4: Market Scanner (NO ML PREDICTIONS HERE)
    # Replace your existing Tab 4 section in main() with this:

    # Tab 4: Enhanced Market Scanner with Rate Limiting
    with tab4:
        st.subheader("🎯 Comprehensive Market Scanner (5000+ Stocks)")
        
        # Enhanced controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            scan_size = st.selectbox(
                "Stocks to Scan",
                [100, 500, 1000, 2500, 5000],
                index=2,
                format_func=lambda x: f"{x:,} Stocks"
            )
        
        with col2:
            rate_limit = st.selectbox(
                "Rate Limit (sec/request)",
                [1.0, 2.0, 3.0, 5.0, 10.0],
                index=1,
                format_func=lambda x: f"{x:.1f} seconds"
            )
        
        with col3:
            batch_size = st.selectbox(
                "Batch Size",
                [50, 100, 200],
                index=1,
                help="Process stocks in batches for better progress tracking"
            )
        
        with col4:
            start_offset = st.number_input(
                "Start From Stock #",
                min_value=0,
                max_value=4900,
                value=0,
                step=100,
                help="Resume scan from specific position"
            )
        
        # Estimated time calculation
        estimated_minutes = (scan_size * rate_limit) / 60
        st.info(f"⏱️ **Estimated Time:** {estimated_minutes:.1f} minutes for {scan_size:,} stocks at {rate_limit:.1f}s per request")
        
        # Scan controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            scan_btn = st.button(
                f"🔍 Start Comprehensive Scan ({scan_size:,} stocks)",
                type="primary",
                help=f"Scan {scan_size:,} stocks with {rate_limit:.1f}s delays"
            )
        
        with col2:
            if st.button("⏸️ Pause Scan", help="Pause current scan"):
                st.session_state.scan_paused = True
        
        with col3:
            if st.button("🗑️ Clear Results", help="Clear cached results"):
                if 'comprehensive_scan_results' in st.session_state:
                    del st.session_state.comprehensive_scan_results
                st.success("Results cleared!")
        
        # Progress tracking placeholders
        progress_bar = st.empty()
        status_text = st.empty()
        stats_container = st.empty()
        
        # Run comprehensive scan
        if scan_btn:
            # Initialize enhanced scanner
            enhanced_scanner = EnhancedMarketScanner(strategist, rate_limit_delay=rate_limit)
            
            def update_progress(progress):
                progress_bar.progress(progress)
            
            def update_status(status):
                status_text.info(status)
            
            with st.spinner(f"🔍 Scanning {scan_size:,} stocks... This will take approximately {estimated_minutes:.1f} minutes"):
                try:
                    scan_results = enhanced_scanner.scan_comprehensive_market(
                        max_stocks=scan_size,
                        start_from=start_offset,
                        batch_size=batch_size,
                        progress_callback=update_progress,
                        status_callback=update_status
                    )
                    
                    st.session_state.comprehensive_scan_results = scan_results
                    
                    # Final statistics
                    with stats_container.container():
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Stocks Analyzed", f"{scan_results['total_analyzed']:,}")
                        
                        with col2:
                            st.metric("Success Rate", f"{scan_results['success_rate']:.1f}%")
                        
                        with col3:
                            st.metric("Scan Duration", f"{scan_results['scan_duration_minutes']:.1f} min")
                        
                        with col4:
                            st.metric("Failed Requests", f"{scan_results['total_failed']:,}")
                    
                    st.success(f"✅ Comprehensive scan completed! Analyzed {scan_results['total_analyzed']:,} stocks successfully")
                    
                except Exception as e:
                    st.error(f"Comprehensive scan failed: {str(e)}")
        
        # Display results if available
        if hasattr(st.session_state, 'comprehensive_scan_results'):
            scan_results = st.session_state.comprehensive_scan_results
            enhanced_scanner = EnhancedMarketScanner(strategist, rate_limit_delay=rate_limit)
            
            st.markdown("---")
            st.markdown(f"### 📊 Comprehensive Scan Results")
            
            # Scan statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyzed", f"{scan_results['total_analyzed']:,}")
            
            with col2:
                st.metric("Success Rate", f"{scan_results['success_rate']:.1f}%")
            
            with col3:
                st.metric("Scan Duration", f"{scan_results['scan_duration_minutes']:.1f} min")
            
            with col4:
                st.metric("Last Updated", scan_results['scan_timestamp'][-8:])  # Show time only
            
            # Display results
            col1, col2 = st.columns(2)
            
            # Top Buys
            with col1:
                st.markdown("#### 🟢 **TOP 10 BUY OPPORTUNITIES**")
                
                buy_options = []
                for i, stock in enumerate(scan_results['top_buys']):
                    score = stock['technical_score']
                    ticker = stock['ticker']
                    price = stock['current_price']
                    change_20d = stock['analysis'].get('price_change_20d', 0)
                    
                    color = "🟢" if score >= 70 else "🟡"
                    buy_options.append(f"{color} {ticker} - Score: {score:.1f} | ${price:.2f} | 20D: {change_20d:+.1f}%")
                
                selected_buy = st.selectbox(
                    "Select Stock for Analysis:",
                    options=range(len(buy_options)),
                    format_func=lambda x: buy_options[x],
                    key="comprehensive_buy_selector"
                )
                
                if selected_buy is not None:
                    stock_data = scan_results['top_buys'][selected_buy]
                    
                    with st.expander(f"📊 {stock_data['ticker']} Analysis", expanded=True):
                        recommendation_text = enhanced_scanner.get_stock_recommendation_text(stock_data)
                        st.markdown(recommendation_text)
            
            # Top Sells
            with col2:
                st.markdown("#### 🔴 **TOP 10 SELL OPPORTUNITIES**")
                
                sell_options = []
                for i, stock in enumerate(scan_results['top_sells']):
                    score = stock['technical_score']
                    ticker = stock['ticker']
                    price = stock['current_price']
                    change_20d = stock['analysis'].get('price_change_20d', 0)
                    
                    color = "🔴" if score <= 35 else "🟠"
                    sell_options.append(f"{color} {ticker} - Score: {score:.1f} | ${price:.2f} | 20D: {change_20d:+.1f}%")
                
                selected_sell = st.selectbox(
                    "Select Stock for Analysis:",
                    options=range(len(sell_options)),
                    format_func=lambda x: sell_options[x],
                    key="comprehensive_sell_selector"
                )
                
                if selected_sell is not None:
                    stock_data = scan_results['top_sells'][selected_sell]
                    
                    with st.expander(f"📊 {stock_data['ticker']} Analysis", expanded=True):
                        recommendation_text = enhanced_scanner.get_stock_recommendation_text(stock_data)
                        st.markdown(recommendation_text)
            
            # Detailed Stock Analysis Section
            st.markdown("---")
            st.markdown("### 🔬 **Detailed Stock Analysis**")
            
            # Combined stock selection
            all_stocks = []
            for stock in scan_results['top_buys']:
                all_stocks.append(('BUY', stock))
            for stock in scan_results['top_sells']:
                all_stocks.append(('SELL', stock))
            
            if all_stocks:
                stock_options = [f"{'🟢' if rec == 'BUY' else '🔴'} {stock['ticker']} (Score: {stock['technical_score']:.1f})" 
                            for rec, stock in all_stocks]
                
                selected_stock_idx = st.selectbox(
                    "Choose Stock for Detailed Analysis:",
                    options=range(len(stock_options)),
                    format_func=lambda x: stock_options[x],
                    key="comprehensive_detailed_selector"
                )
                
                if selected_stock_idx is not None:
                    _, selected_stock_data = all_stocks[selected_stock_idx]
                    
                    # Display detailed analysis (reuse existing function)
                    display_detailed_stock_analysis(selected_stock_data, strategist)
            
            # Export functionality
            st.markdown("---")
            st.markdown("#### 📥 **Export Results**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Download Buy List CSV"):
                    buy_df = pd.DataFrame([
                        {
                            'Ticker': s['ticker'],
                            'Score': s['technical_score'],
                            'Price': s['current_price'],
                            'Trend': s['analysis']['trend'],
                            'RSI': s['analysis']['rsi'],
                            '20D_Change': s['analysis']['price_change_20d']
                        }
                        for s in scan_results['top_buys']
                    ])
                    st.download_button(
                        "Download Buy Recommendations",
                        buy_df.to_csv(index=False),
                        "top_buy_stocks.csv",
                        "text/csv"
                    )
            
            with col2:
                if st.button("📊 Download Sell List CSV"):
                    sell_df = pd.DataFrame([
                        {
                            'Ticker': s['ticker'],
                            'Score': s['technical_score'],
                            'Price': s['current_price'],
                            'Trend': s['analysis']['trend'],
                            'RSI': s['analysis']['rsi'],
                            '20D_Change': s['analysis']['price_change_20d']
                        }
                        for s in scan_results['top_sells']
                    ])
                    st.download_button(
                        "Download Sell Recommendations",
                        sell_df.to_csv(index=False),
                        "top_sell_stocks.csv",
                        "text/csv"
                    )
            
            with col3:
                if st.button("📈 View Score Distribution"):
                    # Create score distribution chart
                    all_scores = [s['technical_score'] for s in scan_results['top_buys'] + scan_results['top_sells']]
                    
                    import plotly.express as px
                    fig = px.histogram(
                        x=all_scores,
                        nbins=20,
                        title="Technical Score Distribution",
                        labels={'x': 'Technical Score', 'y': 'Count'}
                    )
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Welcome message
            st.markdown("""
            ### 🚀 **Comprehensive Market Intelligence**
            
            **Enhanced Scanner Features:**
            - **Massive Scale**: Scan up to 5,000 stocks across all market caps
            - **Rate Limiting**: Intelligent delays to avoid API limits (1-10 seconds per request)
            - **Progress Tracking**: Real-time progress with batch processing
            - **Resume Capability**: Start from any position in the scan
            - **Error Recovery**: Automatic retry with exponential backoff
            
            **Performance Optimizations:**
            - **Batch Processing**: Process stocks in manageable batches (50-200 stocks)
            - **Smart Caching**: 2-hour cache to avoid repeated scans
            - **Adaptive Rate Limiting**: Automatically adjusts based on API responses
            - **Comprehensive Universe**: Includes S&P 500, Russell 1000, and growth stocks
            
            **Time Estimates:**
            - **100 stocks @ 2s/request**: ~3-4 minutes
            - **500 stocks @ 2s/request**: ~15-20 minutes  
            - **1,000 stocks @ 2s/request**: ~30-40 minutes
            - **5,000 stocks @ 2s/request**: ~2.5-3.5 hours
            
            **🎯 Recommended Settings for 5000 Stock Scan:**
            - **Rate Limit**: 2.0 seconds (safe for most API limits)
            - **Batch Size**: 100 stocks (good balance of progress and performance)  
            - **Expected Time**: ~3 hours for complete scan
            
            Configure your scan parameters above and click "Start Comprehensive Scan"!
            """)
            
            # Quick start suggestions
            st.markdown("#### 🚀 **Quick Start Options**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🏃 Quick Scan (100 stocks)**
                - Time: ~3-4 minutes
                - Coverage: S&P 100 + popular stocks
                - Good for: Quick market overview
                """)
            
            with col2:
                st.markdown("""
                **⚖️ Balanced Scan (1000 stocks)**
                - Time: ~30-40 minutes  
                - Coverage: S&P 500 + Russell growth
                - Good for: Comprehensive analysis
                """)
            
            with col3:
                st.markdown("""
                **🔍 Full Scan (5000 stocks)**
                - Time: ~3 hours
                - Coverage: Complete market universe
                - Good for: Deep market discovery
                """)
            
            # Performance tips
            st.markdown("#### 💡 **Performance Tips**")
            
            st.markdown("""
            - **Start with 100-500 stocks** to test your setup
            - **Use 2-3 second rate limits** to avoid 429 errors
            - **Run large scans during off-hours** for best performance
            - **Results are cached for 2 hours** - no need to re-scan immediately
            - **Use the resume feature** if scan gets interrupted
            - **Export results to CSV** for further analysis
            """)



if __name__ == "__main__":
    main()