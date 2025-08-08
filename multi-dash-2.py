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
# ENHANCED CACHING WITH COMPREHENSIVE FX SUPPORT
# =============================================================================

@st.cache_data(ttl=300)  # 5 minute cache
def cached_get_asset_data(api_key: str, ticker: str, asset_class: str, days: int = 500) -> Dict:
    """Enhanced cached version of get_asset_data with comprehensive FX support"""
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
    """Enhanced cached version with comprehensive FX options handling"""
    try:
        client = RESTClient(api_key)
        
        print(f"Fetching options data for {ticker} ({asset_class})...")
        
        # First, try to get actual options contracts
        contracts = []
        options_ticker = ticker  # Use original ticker for options search
        
        try:
            for contract in client.list_options_contracts(
                underlying_ticker=options_ticker,
                expiration_date_gte=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                expiration_date_lte=(datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),
                limit=1000
            ):
                contracts.append(contract)
        except Exception as options_error:
            print(f"Options contracts fetch failed: {str(options_error)}")
            contracts = []
        
        # If no options found and it's FOREX, provide helpful alternatives
        if not contracts and asset_class == 'FOREX':
            # Enhanced Currency ETF mapping for options trading
            forex_to_etf_mapping = {
                'EURUSD': {'etf': 'FXE', 'name': 'Euro ETF', 'liquidity': 'High'},
                'GBPUSD': {'etf': 'FXB', 'name': 'British Pound ETF', 'liquidity': 'High'},
                'USDJPY': {'etf': 'FXY', 'name': 'Japanese Yen ETF', 'liquidity': 'High'},
                'USDCHF': {'etf': 'FXF', 'name': 'Swiss Franc ETF', 'liquidity': 'Medium'},
                'AUDUSD': {'etf': 'FXA', 'name': 'Australian Dollar ETF', 'liquidity': 'High'},
                'USDCAD': {'etf': 'FXC', 'name': 'Canadian Dollar ETF', 'liquidity': 'High'},
                'EURGBP': {'etf': 'EUFX', 'name': 'EUR/GBP ETF', 'liquidity': 'Low'},
                'EURJPY': {'etf': 'EURJPY', 'name': 'EUR/JPY ETF', 'liquidity': 'Low'},
                'NZDUSD': {'etf': 'BNZ', 'name': 'New Zealand Dollar ETF', 'liquidity': 'Low'},
                'USDSGD': {'etf': 'UUP', 'name': 'USD ETF (SGD proxy)', 'liquidity': 'Medium'}
            }
            
            suggested_info = forex_to_etf_mapping.get(ticker.upper(), {})
            suggested_etf = suggested_info.get('etf')
            
            return {
                'expiration': 'N/A',
                'calls': pd.DataFrame(),
                'puts': pd.DataFrame(), 
                'days_to_expiry': 0,
                'underlying_price': current_price,
                'underlying_ticker': ticker,
                'asset_class': asset_class,
                'total_contracts': 0,
                'source': 'forex_no_options',
                'forex_limitation': True,
                'suggested_etf': suggested_etf,
                'suggested_info': suggested_info,
                'error_message': f"FOREX options not available for {ticker}. Consider {suggested_etf} ETF for options strategies." if suggested_etf else f"FOREX options not available for {ticker}.",
                'forex_etf_mapping': forex_to_etf_mapping
            }
        
        # If no contracts found for any asset class
        if not contracts:
            raise ValueError(f"No options contracts found for {ticker}")
        
        # Process contracts normally (existing code)
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
        
        # Enhanced fallback for FOREX
        if asset_class == 'FOREX':
            forex_to_etf_mapping = {
                'EURUSD': {'etf': 'FXE', 'name': 'Euro ETF', 'liquidity': 'High'},
                'GBPUSD': {'etf': 'FXB', 'name': 'British Pound ETF', 'liquidity': 'High'},
                'USDJPY': {'etf': 'FXY', 'name': 'Japanese Yen ETF', 'liquidity': 'High'},
                'USDCHF': {'etf': 'FXF', 'name': 'Swiss Franc ETF', 'liquidity': 'Medium'},
                'AUDUSD': {'etf': 'FXA', 'name': 'Australian Dollar ETF', 'liquidity': 'High'},
                'USDCAD': {'etf': 'FXC', 'name': 'Canadian Dollar ETF', 'liquidity': 'High'},
                'EURGBP': {'etf': 'EUFX', 'name': 'EUR/GBP ETF', 'liquidity': 'Low'},
                'NZDUSD': {'etf': 'BNZ', 'name': 'New Zealand Dollar ETF', 'liquidity': 'Low'}
            }
            
            suggested_info = forex_to_etf_mapping.get(ticker.upper(), {})
            suggested_etf = suggested_info.get('etf')
            
            return {
                'expiration': 'N/A',
                'calls': pd.DataFrame(),
                'puts': pd.DataFrame(),
                'days_to_expiry': 0,
                'underlying_price': current_price,
                'underlying_ticker': ticker,
                'asset_class': asset_class,
                'total_contracts': 0,
                'source': 'forex_fallback',
                'forex_limitation': True,
                'suggested_etf': suggested_etf,
                'suggested_info': suggested_info,
                'error_message': f"FOREX options not available for {ticker}. Try {suggested_etf} ETF instead." if suggested_etf else "FOREX options not available.",
                'forex_etf_mapping': forex_to_etf_mapping
            }
        
        raise

@st.cache_data(ttl=300)  # 5 minute cache
def cached_search_symbols(api_key: str, asset_class: str, query: str) -> List[Dict]:
    """Enhanced cached symbol search with comprehensive FX support"""
    try:
        client = RESTClient(api_key)
        results = []
        
        if asset_class == 'FOREX':
            # Get FX tickers using your approach
            for ticker in client.list_tickers(market="fx", active="true", limit=100):
                if query.upper() in ticker.ticker.upper():
                    results.append({
                        'symbol': ticker.ticker.replace('C:', ''),
                        'name': f"{ticker.ticker} - Forex Pair",
                        'market': 'fx'
                    })
        elif asset_class == 'EQUITIES':
            # Get stock tickers
            for ticker in client.list_tickers(market="stocks", active="true", limit=100):
                if query.upper() in ticker.ticker.upper():
                    results.append({
                        'symbol': ticker.ticker,
                        'name': getattr(ticker, 'name', ticker.ticker),
                        'market': 'stocks'
                    })
        
        return results[:20]  # Limit results
        
    except Exception as e:
        print(f"Symbol search failed: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # 1 hour cache
def cached_get_popular_symbols(asset_class: str) -> List[str]:
    """Enhanced cached popular symbols with comprehensive FX support"""
    asset_configs = {
        'INDICES': ['SPY', 'QQQ', 'IWM', 'EWU', 'VGK', 'EFA', 'VIX'],
        'EQUITIES': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
        'FOREX': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'EURGBP', 'EURJPY']
    }
    return asset_configs.get(asset_class, [])

# =============================================================================
# ENHANCED MULTI-ASSET OPTIONS STRATEGIST WITH COMPREHENSIVE FX SUPPORT
# =============================================================================

class MultiAssetOptionsStrategist:
    """Professional Multi-Asset Options Strategist with Comprehensive FX Support"""
    
    def __init__(self, polygon_api_key: str):
        if not polygon_api_key:
            raise ValueError("Polygon API key is required")
        
        self.client = RESTClient(polygon_api_key)
        self.polygon_api_key = polygon_api_key
        
        # Enhanced asset class configurations with comprehensive FX
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
                'popular_symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'EURGBP', 'EURJPY'],
                'description': 'Currency Pairs (Options via ETF Alternatives)'
            }
        }
    
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
    
    def get_asset_data(self, ticker: str, asset_class: str, days: int = 500) -> Dict:
        """Get data for any asset class with enhanced caching"""
        return cached_get_asset_data(self.polygon_api_key, ticker, asset_class, days)
    
    def get_options_data(self, ticker: str, asset_class: str, current_price: float = None) -> Dict:
        """Get options data for any asset class with comprehensive FX support"""
        if current_price is None:
            underlying_data = self.get_asset_data(ticker, asset_class, days=30)
            current_price = underlying_data['current_price']
        return cached_get_options_data(self.polygon_api_key, ticker, asset_class, current_price)
    
    def analyze_market_conditions(self, data: Dict) -> Dict:
        """Enhanced market analysis with comprehensive FX-specific considerations"""
        current_price = data['current_price']
        asset_class = data.get('asset_class', 'EQUITIES')
        
        # Calculate technical indicators from historical data
        df = data['historical_data'].tail(200)  # Use recent data
        
        # Moving averages with proper min_periods
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
        
        # Enhanced volatility regime with FX-specific thresholds
        realized_vol = data.get('realized_vol_21d', 0.20)
        if asset_class == 'FOREX':
            # FX-specific volatility thresholds
            if realized_vol > 0.25:
                vol_regime = 'EXTREME_VOL'
            elif realized_vol > 0.18:
                vol_regime = 'HIGH_VOL'
            elif realized_vol < 0.10:
                vol_regime = 'LOW_VOL'
            else:
                vol_regime = 'NORMAL_VOL'
        else:
            # Standard volatility thresholds
            if realized_vol > 0.30:
                vol_regime = 'EXTREME_VOL'
            elif realized_vol > 0.25:
                vol_regime = 'HIGH_VOL'
            elif realized_vol < 0.12:
                vol_regime = 'LOW_VOL'
            else:
                vol_regime = 'NORMAL_VOL'
        
        # Enhanced momentum with FX-specific thresholds
        if asset_class == 'FOREX':
            # FX has different RSI thresholds
            if rsi > 80:
                momentum = 'EXTREMELY_OVERBOUGHT'
            elif rsi > 70:
                momentum = 'OVERBOUGHT'
            elif rsi > 55:
                momentum = 'BULLISH'
            elif rsi < 20:
                momentum = 'EXTREMELY_OVERSOLD'
            elif rsi < 30:
                momentum = 'OVERSOLD'
            elif rsi < 45:
                momentum = 'BEARISH'
            else:
                momentum = 'NEUTRAL'
        else:
            # Standard momentum thresholds
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
    
    def _assess_market_suitability(self, strategy_name: str, market_analysis: Dict, asset_class: str) -> Dict:
        """Assess how suitable current market conditions are for the strategy"""
        
        trend = market_analysis.get('trend', 'SIDEWAYS')
        volatility = market_analysis.get('volatility_regime', 'NORMAL_VOL')
        momentum = market_analysis.get('momentum', 'NEUTRAL')
        
        suitability_map = {
            'COVERED_CALL': {
                'ideal_conditions': ['BULLISH', 'STRONG_BULLISH'],
                'ideal_volatility': ['NORMAL_VOL', 'HIGH_VOL'],
                'description': 'Best when moderately bullish with elevated IV'
            },
            'CASH_SECURED_PUT': {
                'ideal_conditions': ['BULLISH', 'SIDEWAYS'],
                'ideal_volatility': ['NORMAL_VOL', 'HIGH_VOL'],
                'description': 'Excellent for acquiring stocks at discount'
            },
            'BULL_CALL_SPREAD': {
                'ideal_conditions': ['BULLISH', 'STRONG_BULLISH'],
                'ideal_volatility': ['NORMAL_VOL'],
                'description': 'Perfect for directional bullish plays'
            },
            'BEAR_PUT_SPREAD': {
                'ideal_conditions': ['BEARISH', 'STRONG_BEARISH'],
                'ideal_volatility': ['NORMAL_VOL'],
                'description': 'Ideal for directional bearish outlook'
            },
            'BEAR_CALL_SPREAD': {
                'ideal_conditions': ['BEARISH', 'STRONG_BEARISH'],
                'ideal_volatility': ['HIGH_VOL'],
                'description': 'Great for bearish outlook with high IV'
            },
            'IRON_CONDOR': {
                'ideal_conditions': ['SIDEWAYS'],
                'ideal_volatility': ['LOW_VOL', 'NORMAL_VOL'],
                'description': 'Perfect for range-bound, low volatility markets'
            },
            'LONG_STRADDLE': {
                'ideal_conditions': ['SIDEWAYS'],  # Before big moves
                'ideal_volatility': ['LOW_VOL'],   # Buy low IV, sell high
                'description': 'Best before expected volatility expansion'
            },
            'SHORT_STRADDLE': {
                'ideal_conditions': ['SIDEWAYS'],
                'ideal_volatility': ['HIGH_VOL'],  # Sell high IV
                'description': 'Ideal for range-bound markets with high IV'
            },
            'LONG_STRANGLE': {
                'ideal_conditions': ['SIDEWAYS'],  # Before breakouts
                'ideal_volatility': ['LOW_VOL'],
                'description': 'Good for expecting big moves in either direction'
            },
            'SHORT_STRANGLE': {
                'ideal_conditions': ['SIDEWAYS'],
                'ideal_volatility': ['HIGH_VOL'],
                'description': 'Perfect for stable markets with high IV'
            },
            'PROTECTIVE_PUT': {
                'ideal_conditions': ['BULLISH'],   # Bullish but want protection
                'ideal_volatility': ['HIGH_VOL'],
                'description': 'Insurance for long stock positions'
            },
            'COLLAR': {
                'ideal_conditions': ['SIDEWAYS', 'BULLISH'],
                'ideal_volatility': ['NORMAL_VOL'],
                'description': 'Conservative protection with limited upside'
            }
        }
        
        strategy_info = suitability_map.get(strategy_name, {})
        
        # Calculate suitability score
        suitability_score = 50  # Base score
        
        if trend in strategy_info.get('ideal_conditions', []):
            suitability_score += 25
        
        if volatility in strategy_info.get('ideal_volatility', []):
            suitability_score += 25
        
        # Determine suitability level
        if suitability_score >= 85:
            suitability_level = "EXCELLENT"
            suitability_color = "🟢"
        elif suitability_score >= 65:
            suitability_level = "GOOD"
            suitability_color = "🟡"
        elif suitability_score >= 45:
            suitability_level = "FAIR"
            suitability_color = "🟠"
        else:
            suitability_level = "POOR"
            suitability_color = "🔴"
        
        return {
            'score': suitability_score,
            'level': suitability_level,
            'color': suitability_color,
            'description': strategy_info.get('description', ''),
            'current_trend': trend,
            'current_volatility': volatility
        }
    
    def _get_strategy_explanation(self, strategy_name: str) -> Dict:
        """Get comprehensive strategy explanation"""
        
        explanations = {
            'COVERED_CALL': {
                'summary': 'Own stock + sell call option for income',
                'mechanics': 'Buy 100 shares, sell 1 call option against them',
                'profit_source': 'Stock appreciation (up to strike) + option premium',
                'best_when': 'Neutral to moderately bullish, elevated IV',
                'risk_level': 'LOW-MODERATE',
                'complexity': 'BEGINNER'
            },
            'CASH_SECURED_PUT': {
                'summary': 'Sell put option while holding cash to buy shares if assigned',
                'mechanics': 'Sell put option, keep cash equal to 100 shares at strike',
                'profit_source': 'Option premium if not assigned, stock ownership if assigned',
                'best_when': 'Want to own stock at lower price, bullish long-term',
                'risk_level': 'LOW-MODERATE',
                'complexity': 'BEGINNER'
            },
            'BULL_CALL_SPREAD': {
                'summary': 'Buy lower strike call + sell higher strike call',
                'mechanics': 'Buy ITM/ATM call, sell OTM call, same expiration',
                'profit_source': 'Stock price appreciation between strikes',
                'best_when': 'Moderately bullish outlook, limited capital',
                'risk_level': 'MODERATE',
                'complexity': 'INTERMEDIATE'
            },
            'BEAR_PUT_SPREAD': {
                'summary': 'Buy higher strike put + sell lower strike put',
                'mechanics': 'Buy ITM/ATM put, sell OTM put, same expiration',
                'profit_source': 'Stock price decline between strikes',
                'best_when': 'Moderately bearish outlook, limited capital',
                'risk_level': 'MODERATE',
                'complexity': 'INTERMEDIATE'
            },
            'BEAR_CALL_SPREAD': {
                'summary': 'Sell lower strike call + buy higher strike call',
                'mechanics': 'Sell ITM/ATM call, buy OTM call, collect net credit',
                'profit_source': 'Net credit if stock stays below short strike',
                'best_when': 'Bearish outlook, high IV environment',
                'risk_level': 'MODERATE',
                'complexity': 'INTERMEDIATE'
            },
            'IRON_CONDOR': {
                'summary': 'Sell call spread + sell put spread for range-bound profits',
                'mechanics': 'Sell OTM call & put, buy further OTM call & put',
                'profit_source': 'Net credit if stock stays within breakeven range',
                'best_when': 'Low volatility, range-bound market expected',
                'risk_level': 'MODERATE',
                'complexity': 'ADVANCED'
            },
            'LONG_STRANGLE': {
                'summary': 'Buy OTM call + buy OTM put for big move profits',
                'mechanics': 'Buy OTM call and OTM put, different strikes',
                'profit_source': 'Large price movement beyond breakeven points',
                'best_when': 'Expecting volatility, cheaper than straddle',
                'risk_level': 'HIGH',
                'complexity': 'INTERMEDIATE'
            },
            'SHORT_STRANGLE': {
                'summary': 'Sell OTM call + sell OTM put for range income',
                'mechanics': 'Sell OTM call and put, collect net credit',
                'profit_source': 'Net credit if stock stays between strikes',
                'best_when': 'High IV, stable market expected',
                'risk_level': 'HIGH',
                'complexity': 'ADVANCED'
            },
            'PROTECTIVE_PUT': {
                'summary': 'Own stock + buy put option for downside protection',
                'mechanics': 'Buy 100 shares + buy 1 put option as insurance',
                'profit_source': 'Stock appreciation minus put premium cost',
                'best_when': 'Bullish long-term but want downside protection',
                'risk_level': 'LOW',
                'complexity': 'BEGINNER'
            },
            'COLLAR': {
                'summary': 'Own stock + buy put + sell call for protected income',
                'mechanics': 'Buy protective put + sell covered call simultaneously',
                'profit_source': 'Limited stock appreciation + net option premium',
                'best_when': 'Want protection with limited upside, reduce cost',
                'risk_level': 'LOW',
                'complexity': 'INTERMEDIATE'
            }
        }
        
        return explanations.get(strategy_name, {
            'summary': 'Professional options strategy',
            'mechanics': 'Complex multi-leg options position',
            'profit_source': 'Market movement and time decay',
            'best_when': 'Specific market conditions align',
            'risk_level': 'MODERATE',
            'complexity': 'INTERMEDIATE'
        })

    # =============================================================================
    # ENHANCED BACKTESTING WITH ACCURATE MATH (ALL PRESERVED FROM MAIN)
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
            elif strategy == 'BEAR_PUT_SPREAD':
                results = self._backtest_bear_put_accurate(df, params, asset_class)
            elif strategy == 'BEAR_CALL_SPREAD':
                results = self._backtest_bear_call_accurate(df, params, asset_class)
            elif strategy == 'LONG_STRADDLE':
                results = self._backtest_long_straddle_accurate(df, params, asset_class)
            elif strategy == 'SHORT_STRADDLE':
                results = self._backtest_short_straddle_accurate(df, params, asset_class)
            elif strategy == 'LONG_STRANGLE':
                results = self._backtest_long_strangle_accurate(df, params, asset_class)
            elif strategy == 'SHORT_STRANGLE':
                results = self._backtest_short_strangle_accurate(df, params, asset_class)
            elif strategy == 'PROTECTIVE_PUT':
                results = self._backtest_protective_put_accurate(df, params, asset_class)
            elif strategy == 'COLLAR':
                results = self._backtest_collar_accurate(df, params, asset_class)
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
    
    def _backtest_bear_put_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate bear put spread backtesting"""
        
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
            
            # Bear put spread strikes
            buy_strike = entry_price * 0.98   # Buy higher strike (ITM)
            sell_strike = entry_price * 0.90  # Sell lower strike (OTM)
            
            # Calculate net debit
            buy_premium = entry_price * 0.05  # 5% for ITM put
            sell_premium = entry_price * 0.02  # 2% for OTM put
            net_debit = buy_premium - sell_premium
            
            # Position sizing
            cost_per_contract = net_debit * share_equivalent
            contracts = int(cash * 0.4 / cost_per_contract)
            if contracts == 0:
                contracts = 1
            
            total_cost = contracts * cost_per_contract
            spread_width = buy_strike - sell_strike
            max_profit_per_contract = spread_width - net_debit
            
            # Outcome at expiration
            if exit_price <= sell_strike:
                # Maximum profit
                pnl = contracts * max_profit_per_contract * share_equivalent
            elif exit_price >= buy_strike:
                # Maximum loss
                pnl = -total_cost
            else:
                # Partial profit
                pnl = contracts * ((buy_strike - exit_price) - net_debit) * share_equivalent
            
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
    
    def _backtest_bear_call_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate bear call spread backtesting"""
        
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
            
            # Bear call spread strikes
            sell_strike = entry_price * 1.02  # Sell lower strike
            buy_strike = entry_price * 1.10   # Buy higher strike
            
            # Calculate net credit
            sell_premium = entry_price * 0.03  # 3% for short call
            buy_premium = entry_price * 0.015  # 1.5% for long call
            net_credit = sell_premium - buy_premium
            
            # Margin requirement
            spread_width = buy_strike - sell_strike
            margin_per_contract = (spread_width - net_credit) * share_equivalent
            contracts = int(cash * 0.5 / margin_per_contract)
            if contracts == 0:
                contracts = 1
            
            total_credit = contracts * net_credit * share_equivalent
            total_margin = contracts * margin_per_contract
            
            # Outcome at expiration
            if exit_price <= sell_strike:
                # Maximum profit
                pnl = total_credit
            elif exit_price >= buy_strike:
                # Maximum loss
                pnl = total_credit - contracts * spread_width * share_equivalent
            else:
                # Partial loss
                pnl = total_credit - contracts * (exit_price - sell_strike) * share_equivalent
            
            cash += pnl
            portfolio_value = cash
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'sell_strike': sell_strike,
                'buy_strike': buy_strike,
                'sell_premium': sell_premium,
                'buy_premium': buy_premium,
                'net_credit': net_credit,
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
    
    def _backtest_long_straddle_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate long straddle backtesting"""
        
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
            
            # ATM straddle strikes
            strike = entry_price
            
            # Calculate premiums (simplified)
            call_premium = entry_price * 0.04  # 4% for ATM call
            put_premium = entry_price * 0.04   # 4% for ATM put
            total_premium = call_premium + put_premium
            
            # Position sizing
            cost_per_contract = total_premium * share_equivalent
            contracts = int(cash * 0.6 / cost_per_contract)  # Use 60% of capital
            if contracts == 0:
                contracts = 1
            
            total_cost = contracts * cost_per_contract
            
            # Breakeven points
            upper_be = strike + total_premium
            lower_be = strike - total_premium
            
            # P&L calculation
            if exit_price >= upper_be:
                # Profit on call side
                pnl = contracts * (exit_price - upper_be) * share_equivalent
            elif exit_price <= lower_be:
                # Profit on put side  
                pnl = contracts * (lower_be - exit_price) * share_equivalent
            else:
                # Loss (between breakevens)
                pnl = -total_cost + contracts * min(max(exit_price - strike, 0), max(strike - exit_price, 0)) * share_equivalent
            
            cash += pnl
            portfolio_value = cash
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'strike': strike,
                'call_premium': call_premium,
                'put_premium': put_premium,
                'total_premium': total_premium,
                'contracts': contracts,
                'upper_breakeven': upper_be,
                'lower_breakeven': lower_be,
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
    
    def _backtest_short_straddle_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate short straddle backtesting"""
        
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
            
            # ATM straddle strikes
            strike = entry_price
            
            # Calculate premiums
            call_premium = entry_price * 0.04  # 4% for ATM call
            put_premium = entry_price * 0.04   # 4% for ATM put
            total_premium = call_premium + put_premium
            
            # Margin requirement
            margin_per_contract = entry_price * 0.25 * share_equivalent  # 25% margin
            contracts = int(cash * 0.4 / margin_per_contract)  # Use 40% of capital
            if contracts == 0:
                contracts = 1
            
            total_credit = contracts * total_premium * share_equivalent
            total_margin = contracts * margin_per_contract
            
            # P&L calculation (short straddle)
            intrinsic_value = max(abs(exit_price - strike), 0)
            
            if exit_price == strike:
                # Maximum profit
                pnl = total_credit
            else:
                # Profit/loss based on how far from strike
                pnl = total_credit - contracts * max(intrinsic_value - total_premium, 0) * share_equivalent
            
            cash += pnl
            portfolio_value = cash
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'strike': strike,
                'call_premium': call_premium,
                'put_premium': put_premium,
                'total_premium': total_premium,
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
    
    def _backtest_long_strangle_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate long strangle backtesting"""
        
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
            
            # Long strangle strikes (OTM options)
            call_strike = entry_price * 1.05  # 5% OTM call
            put_strike = entry_price * 0.95   # 5% OTM put
            
            # Calculate premiums
            call_premium = entry_price * 0.025  # 2.5% for OTM call
            put_premium = entry_price * 0.025   # 2.5% for OTM put
            total_premium = call_premium + put_premium
            
            # Position sizing
            cost_per_contract = total_premium * share_equivalent
            contracts = int(cash * 0.6 / cost_per_contract)
            if contracts == 0:
                contracts = 1
            
            total_cost = contracts * cost_per_contract
            
            # Breakeven points
            upper_be = call_strike + total_premium
            lower_be = put_strike - total_premium
            
            # P&L calculation
            if exit_price >= upper_be:
                # Profit on call side
                pnl = contracts * (exit_price - upper_be) * share_equivalent
            elif exit_price <= lower_be:
                # Profit on put side
                pnl = contracts * (lower_be - exit_price) * share_equivalent
            else:
                # Loss (between breakevens)
                call_value = max(exit_price - call_strike, 0)
                put_value = max(put_strike - exit_price, 0)
                total_value = call_value + put_value
                pnl = contracts * (total_value - total_premium) * share_equivalent
            
            cash += pnl
            portfolio_value = cash
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'call_strike': call_strike,
                'put_strike': put_strike,
                'call_premium': call_premium,
                'put_premium': put_premium,
                'total_premium': total_premium,
                'contracts': contracts,
                'upper_breakeven': upper_be,
                'lower_breakeven': lower_be,
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
    
    def _backtest_short_strangle_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate short strangle backtesting"""
        
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
            
            # Short strangle strikes
            call_strike = entry_price * 1.08  # 8% OTM call
            put_strike = entry_price * 0.92   # 8% OTM put
            
            # Calculate premiums
            call_premium = entry_price * 0.02  # 2% for OTM call
            put_premium = entry_price * 0.02   # 2% for OTM put
            total_premium = call_premium + put_premium
            
            # Margin requirement
            margin_per_contract = entry_price * 0.2 * share_equivalent  # 20% margin
            contracts = int(cash * 0.4 / margin_per_contract)
            if contracts == 0:
                contracts = 1
            
            total_credit = contracts * total_premium * share_equivalent
            total_margin = contracts * margin_per_contract
            
            # P&L calculation (short strangle)
            call_value = max(exit_price - call_strike, 0)
            put_value = max(put_strike - exit_price, 0)
            total_intrinsic = call_value + put_value
            
            pnl = total_credit - contracts * total_intrinsic * share_equivalent
            
            cash += pnl
            portfolio_value = cash
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'call_strike': call_strike,
                'put_strike': put_strike,
                'call_premium': call_premium,
                'put_premium': put_premium,
                'total_premium': total_premium,
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
    
    def _backtest_protective_put_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate protective put backtesting"""
        
        initial_capital = params.get('initial_capital', 10000)
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        days_to_expiry = params.get('days_to_expiry', 90)  # Longer for protection
        
        trades = []
        portfolio_values = []
        
        cash = 0
        shares = 0
        
        # Initial position
        entry_price = df.iloc[0]['close']
        put_strike = entry_price * 0.95  # 5% OTM protection
        put_premium = entry_price * 0.02  # 2% put cost
        
        # Calculate position size
        total_cost_per_contract = (entry_price + put_premium) * share_equivalent
        contracts = int(initial_capital / total_cost_per_contract)
        
        shares = contracts * share_equivalent
        stock_investment = shares * entry_price
        put_investment = contracts * put_premium * share_equivalent
        cash = initial_capital - stock_investment - put_investment
        
        i = 0
        while i < len(df) - days_to_expiry:
            
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            exit_stock_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Calculate position value
            if exit_stock_price >= put_strike:
                # Put expires worthless, keep stock gains
                position_value = shares * exit_stock_price
                put_value = 0
            else:
                # Put provides protection
                position_value = shares * put_strike  # Protected value
                put_value = shares * (put_strike - exit_stock_price)
            
            total_portfolio_value = cash + position_value
            period_pnl = total_portfolio_value - (stock_investment + put_investment)
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_stock_price,
                'put_strike': put_strike,
                'put_premium': put_premium,
                'shares': shares,
                'stock_investment': stock_investment,
                'put_investment': put_investment,
                'position_value': position_value,
                'total_pnl': period_pnl,
                'portfolio_value': total_portfolio_value
            })
            
            portfolio_values.append(total_portfolio_value)
            
            # Roll position (buy new puts)
            entry_price = exit_stock_price
            put_strike = entry_price * 0.95
            put_premium = entry_price * 0.02
            stock_investment = shares * entry_price
            put_investment = contracts * put_premium * share_equivalent
            cash = total_portfolio_value - stock_investment - put_investment
            
            i += days_to_expiry
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_cash': cash,
            'final_shares': shares,
            'initial_capital': initial_capital
        }
    
    def _backtest_collar_accurate(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Accurate collar backtesting"""
        
        initial_capital = params.get('initial_capital', 10000)
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        days_to_expiry = params.get('days_to_expiry', 60)  # Longer for collar
        
        trades = []
        portfolio_values = []
        
        cash = 0
        shares = 0
        
        # Initial position setup
        entry_price = df.iloc[0]['close']
        put_strike = entry_price * 0.95   # 5% protection
        call_strike = entry_price * 1.05  # 5% upside cap
        
        put_premium = entry_price * 0.025  # 2.5% put cost
        call_premium = entry_price * 0.025  # 2.5% call income
        net_option_cost = put_premium - call_premium  # Often near zero
        
        # Position sizing
        total_cost_per_contract = (entry_price + net_option_cost) * share_equivalent
        contracts = int(initial_capital / total_cost_per_contract)
        
        shares = contracts * share_equivalent
        stock_cost = shares * entry_price
        option_cost = contracts * net_option_cost * share_equivalent if net_option_cost > 0 else 0
        cash = initial_capital - stock_cost - option_cost
        
        i = 0
        while i < len(df) - days_to_expiry:
            
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Collar outcome
            if exit_price <= put_strike:
                # Put provides protection
                position_value = shares * put_strike
            elif exit_price >= call_strike:
                # Called away at call strike
                position_value = shares * call_strike
            else:
                # Normal stock ownership
                position_value = shares * exit_price
            
            total_portfolio_value = cash + position_value
            period_pnl = total_portfolio_value - initial_capital
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'put_strike': put_strike,
                'call_strike': call_strike,
                'put_premium': put_premium,
                'call_premium': call_premium,
                'net_option_cost': net_option_cost,
                'shares': shares,
                'position_value': position_value,
                'total_pnl': period_pnl,
                'portfolio_value': total_portfolio_value
            })
            
            portfolio_values.append(total_portfolio_value)
            
            # Roll position
            entry_price = exit_price
            put_strike = entry_price * 0.95
            call_strike = entry_price * 1.05
            stock_cost = shares * entry_price
            cash = total_portfolio_value - stock_cost
            
            i += days_to_expiry
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_cash': cash,
            'final_shares': shares,
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
    
    def create_professional_chart(self, data: Dict, asset_class: str, 
                             support_resistance: Dict = None) -> go.Figure:
        """Enhanced professional trading chart with comprehensive FX support"""
        
        try:
            df = data['historical_data'].copy()
            
            # Get recent data (1 year)
            one_year_ago = datetime.now() - timedelta(days=365)
            df_recent = df[df.index >= one_year_ago]
            
            if len(df_recent) < 50:
                df_recent = df.tail(252) if len(df) >= 252 else df
            
            # Create figure - FX doesn't need volume subplot
            is_forex = asset_class == 'FOREX'
            
            if is_forex:
                fig = go.Figure()
                chart_title = f'{data["ticker"]} - Currency Pair'
            else:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.75, 0.25],
                    subplot_titles=[f'{data["ticker"]} - {asset_class}', 'Volume']
                )
                chart_title = f'{data["ticker"]} - {asset_class}'
            
            # Candlestick chart with FX-appropriate colors
            if is_forex:
                # FX-specific colors (more neutral)
                increasing_color = '#2E8B57'  # Sea Green
                decreasing_color = '#DC143C'  # Crimson
            else:
                # Standard colors
                increasing_color = '#26a69a'
                decreasing_color = '#ef5350'
            
            candlestick = go.Candlestick(
                x=df_recent.index,
                open=df_recent['open'],
                high=df_recent['high'],
                low=df_recent['low'],
                close=df_recent['close'],
                name='Price',
                increasing_line_color=increasing_color,
                decreasing_line_color=decreasing_color,
                increasing_fillcolor=f'rgba(46, 139, 87, 0.8)' if is_forex else 'rgba(38, 166, 154, 0.8)',
                decreasing_fillcolor=f'rgba(220, 20, 60, 0.8)' if is_forex else 'rgba(239, 83, 80, 0.8)'
            )
            
            # Add candlestick trace - FIXED: proper row/col handling
            if is_forex:
                fig.add_trace(candlestick)
            else:
                fig.add_trace(candlestick, row=1, col=1)
            
            # Volume (non-FX only)
            if not is_forex and 'volume' in df_recent.columns:
                colors = [decreasing_color if c < o else increasing_color 
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
            
            # Moving averages with enhanced styling for FX
            ma_configs = [
                {'period': 20, 'color': '#FF9800', 'width': 2, 'name': 'MA20'},
                {'period': 50, 'color': '#2196F3', 'width': 2, 'name': 'MA50'},
                {'period': 200, 'color': '#9C27B0', 'width': 3, 'name': 'MA200'}
            ]
            
            for ma_config in ma_configs:
                period = ma_config['period']
                min_periods = max(1, min(period // 4, len(df_recent)))
                
                if len(df_recent) >= min_periods:
                    ma = df_recent['close'].rolling(period, min_periods=min_periods).mean()
                    ma_clean = ma.dropna()
                    
                    if not ma_clean.empty:
                        ma_trace = go.Scatter(
                            x=ma_clean.index,
                            y=ma_clean.values,
                            mode='lines',
                            name=ma_config['name'],
                            line=dict(color=ma_config['color'], width=ma_config['width']),
                            opacity=0.8
                        )
                        
                        # FIXED: proper row/col handling for moving averages
                        if is_forex:
                            fig.add_trace(ma_trace)
                        else:
                            fig.add_trace(ma_trace, row=1, col=1)
            
            # Support/Resistance levels with enhanced FX styling
            if support_resistance:
                for level_name, color, style in [
                    ('support_level', '#4CAF50', 'dash'),
                    ('resistance_level', '#F44336', 'dash'),
                    ('target_price', '#FFC107', 'dot')
                ]:
                    if level_name in support_resistance:
                        level_value = support_resistance[level_name]
                        
                        # FIXED: proper row/col handling for horizontal lines
                        hline_kwargs = {
                            'y': level_value,
                            'line_dash': style,
                            'line_color': color,
                            'line_width': 2,
                            'opacity': 0.8,
                            'annotation_text': level_name.replace('_', ' ').title(),
                            'annotation_position': "bottom right"
                        }
                        
                        if not is_forex:
                            hline_kwargs['row'] = 1
                            hline_kwargs['col'] = 1
                        
                        fig.add_hline(**hline_kwargs)
            
            # Enhanced styling with FX considerations
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
                height=600 if is_forex else 700,
                title=dict(
                    text=chart_title,
                    x=0.5,
                    font=dict(size=16, color="white")
                )
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
            
            # FX-specific y-axis formatting
            if is_forex:
                fig.update_yaxes(tickformat='.5f')
            
            return fig
            
        except Exception as e:
            print(f"Chart creation failed: {e}")
            # Return empty figure as fallback
            return go.Figure().add_annotation(
                text=f"Chart creation failed: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
        

    # =============================================================================
    # ALL ACCURATE STRATEGY CALCULATIONS WITH USER CAPITAL
    # =============================================================================
    
    def calculate_long_straddle_accurate(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                        current_price: float, capital: float, asset_class: str, max_risk_amount: float = None) -> Dict:
        """Accurate long straddle calculations with user capital"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Find ATM options (closest to current price)
        if calls_df.empty or puts_df.empty:
            return {'error': 'Insufficient options data for straddle'}
        
        # Find closest strikes to current price
        atm_call = self._find_closest_strike(calls_df, current_price)
        atm_put = self._find_closest_strike(puts_df, current_price)
        
        if not atm_call or not atm_put:
            return {'error': 'Unable to find suitable ATM options'}
        
        # Calculate total cost
        call_premium = atm_call['lastPrice']
        put_premium = atm_put['lastPrice']
        total_premium = call_premium + put_premium
        
        # Position sizing
        cost_per_contract = total_premium * share_equivalent
        max_contracts = int(capital / cost_per_contract)
        
        if max_contracts == 0:
            return {
                'error': f'Insufficient capital. Need ${cost_per_contract:,.2f} for 1 contract',
                'min_capital_needed': cost_per_contract
            }
        
        # Accurate calculations
        total_cost = max_contracts * cost_per_contract
        
        # Breakeven points
        strike = atm_call['strike']  # Should be same for both
        upper_breakeven = strike + total_premium
        lower_breakeven = strike - total_premium
        
        # Max loss is the premium paid
        max_loss = total_cost
        
        # Theoretical max profit is unlimited
        # For practical purposes, estimate profit at 2x volatility move
        volatility_estimate = 0.25  # 25% annual volatility
        days_to_expiry = 30  # Assume monthly
        expected_move = current_price * volatility_estimate * math.sqrt(days_to_expiry / 365)
        
        if expected_move > total_premium:
            estimated_profit = max_contracts * (expected_move - total_premium) * share_equivalent
        else:
            estimated_profit = -total_cost * 0.5  # Partial loss
        
        profit_probability = self._estimate_straddle_probability(current_price, lower_breakeven, upper_breakeven, days_to_expiry)
        
        return {
            'strategy': 'LONG_STRADDLE',
            'strategy_name': 'LONG_STRADDLE',
            'strike': strike,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'contracts': max_contracts,
            'total_cost': total_cost,
            'max_loss': max_loss,
            'estimated_profit': estimated_profit,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'breakeven_range': upper_breakeven - lower_breakeven,
            'breakeven_range_pct': ((upper_breakeven - lower_breakeven) / current_price) * 100,
            'probability_profit': profit_probability,
            'confidence_level': self._calculate_straddle_confidence(asset_class, profit_probability)
        }
    
    def calculate_short_straddle_accurate(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                         current_price: float, capital: float, asset_class: str, max_risk_amount: float = None) -> Dict:
        """Accurate short straddle calculations with user capital"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Find ATM options
        atm_call = self._find_closest_strike(calls_df, current_price)
        atm_put = self._find_closest_strike(puts_df, current_price)
        
        if not atm_call or not atm_put:
            return {'error': 'Unable to find suitable ATM options'}
        
        call_premium = atm_call['lastPrice']
        put_premium = atm_put['lastPrice']
        total_premium = call_premium + put_premium
        
        # Margin requirement (simplified)
        margin_per_contract = current_price * 0.2 * share_equivalent  # 20% margin
        max_contracts = int(capital * 0.3 / margin_per_contract)  # Use 30% of capital
        
        if max_contracts == 0:
            return {'error': 'Insufficient capital for margin requirements'}
        
        # Accurate calculations
        total_credit = max_contracts * total_premium * share_equivalent
        total_margin = max_contracts * margin_per_contract
        
        # Breakeven points
        strike = atm_call['strike']
        upper_breakeven = strike + total_premium
        lower_breakeven = strike - total_premium
        
        # Max profit is premium collected
        max_profit = total_credit
        
        # Max loss is theoretically unlimited, but estimate reasonable loss
        max_reasonable_loss = total_margin  # Use margin as max loss estimate
        
        profit_probability = 1 - self._estimate_straddle_probability(current_price, lower_breakeven, upper_breakeven, 30)
        
        return {
            'strategy': 'SHORT_STRADDLE',
            'strategy_name': 'SHORT_STRADDLE',
            'strike': strike,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'contracts': max_contracts,
            'total_credit': total_credit,
            'total_margin': total_margin,
            'max_profit': max_profit,
            'max_loss': max_reasonable_loss,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'profit_range': upper_breakeven - lower_breakeven,
            'profit_range_pct': ((upper_breakeven - lower_breakeven) / current_price) * 100,
            'probability_profit': profit_probability,
            'confidence_level': self._calculate_straddle_confidence(asset_class, profit_probability, is_short=True)
        }
    
    def calculate_long_strangle_accurate(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                        current_price: float, capital: float, asset_class: str, max_risk_amount: float = None) -> Dict:
        """Accurate long strangle calculations with user capital"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Find OTM options (5-10% OTM)
        call_strike = current_price * 1.05  # 5% OTM call
        put_strike = current_price * 0.95   # 5% OTM put
        
        otm_call = self._find_closest_strike(calls_df, call_strike)
        otm_put = self._find_closest_strike(puts_df, put_strike)
        
        if not otm_call or not otm_put:
            return {'error': 'Unable to find suitable OTM options'}
        
        call_premium = otm_call['lastPrice']
        put_premium = otm_put['lastPrice']
        total_premium = call_premium + put_premium
        
        # Position sizing
        cost_per_contract = total_premium * share_equivalent
        max_contracts = int(capital / cost_per_contract)
        
        if max_contracts == 0:
            return {'error': f'Insufficient capital. Need ${cost_per_contract:,.2f} for 1 contract'}
        
        total_cost = max_contracts * cost_per_contract
        
        # Breakeven points
        upper_breakeven = otm_call['strike'] + total_premium
        lower_breakeven = otm_put['strike'] - total_premium
        
        # Max loss is premium paid
        max_loss = total_cost
        
        # Estimate profit potential
        breakeven_range = upper_breakeven - lower_breakeven
        volatility_estimate = 0.25
        days_to_expiry = 30
        expected_move = current_price * volatility_estimate * math.sqrt(days_to_expiry / 365)
        
        if expected_move > breakeven_range / 2:
            estimated_profit = max_contracts * (expected_move - total_premium) * share_equivalent * 0.7
        else:
            estimated_profit = -total_cost * 0.6
        
        profit_probability = self._estimate_straddle_probability(current_price, lower_breakeven, upper_breakeven, days_to_expiry)
        
        return {
            'strategy': 'LONG_STRANGLE',
            'strategy_name': 'LONG_STRANGLE',
            'call_strike': otm_call['strike'],
            'put_strike': otm_put['strike'],
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'contracts': max_contracts,
            'total_cost': total_cost,
            'max_loss': max_loss,
            'estimated_profit': estimated_profit,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'breakeven_range': breakeven_range,
            'breakeven_range_pct': (breakeven_range / current_price) * 100,
            'probability_profit': profit_probability,
            'confidence_level': self._calculate_strangle_confidence(asset_class, profit_probability)
        }
    
    def calculate_short_strangle_accurate(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                         current_price: float, capital: float, asset_class: str, max_risk_amount: float = None) -> Dict:
        """Accurate short strangle calculations with user capital"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Find OTM options
        call_strike = current_price * 1.08  # 8% OTM call
        put_strike = current_price * 0.92   # 8% OTM put
        
        otm_call = self._find_closest_strike(calls_df, call_strike)
        otm_put = self._find_closest_strike(puts_df, put_strike)
        
        if not otm_call or not otm_put:
            return {'error': 'Unable to find suitable OTM options'}
        
        call_premium = otm_call['lastPrice']
        put_premium = otm_put['lastPrice']
        total_premium = call_premium + put_premium
        
        # Margin requirement
        margin_per_contract = current_price * 0.15 * share_equivalent  # 15% margin
        max_contracts = int(capital * 0.4 / margin_per_contract)  # Use 40% of capital
        
        if max_contracts == 0:
            return {'error': 'Insufficient capital for margin requirements'}
        
        total_credit = max_contracts * total_premium * share_equivalent
        total_margin = max_contracts * margin_per_contract
        
        # Breakeven points
        upper_breakeven = otm_call['strike'] + total_premium
        lower_breakeven = otm_put['strike'] - total_premium
        
        max_profit = total_credit
        max_reasonable_loss = total_margin
        
        profit_probability = 1 - self._estimate_straddle_probability(current_price, lower_breakeven, upper_breakeven, 30)
        
        return {
            'strategy': 'SHORT_STRANGLE',
            'strategy_name': 'SHORT_STRANGLE',
            'call_strike': otm_call['strike'],
            'put_strike': otm_put['strike'],
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'contracts': max_contracts,
            'total_credit': total_credit,
            'total_margin': total_margin,
            'max_profit': max_profit,
            'max_loss': max_reasonable_loss,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'profit_range': upper_breakeven - lower_breakeven,
            'profit_range_pct': ((upper_breakeven - lower_breakeven) / current_price) * 100,
            'probability_profit': profit_probability,
            'confidence_level': self._calculate_strangle_confidence(asset_class, profit_probability, is_short=True)
        }

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
            
            # Calculate probability of profit
            probability_profit = self._estimate_probability_above_breakeven(current_price, breakeven, days_to_expiry)
            
            # Calculate confidence level
            confidence_level = self._calculate_covered_call_confidence(asset_class, probability_profit, premium_yield)
            
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
                'probability_profit': probability_profit,
                'confidence_level': confidence_level
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
            
            # Calculate probability of profit (not being assigned)
            probability_profit = self._estimate_probability_above_breakeven(current_price, strike, days_to_expiry)
            
            # Calculate confidence level
            confidence_level = self._calculate_csp_confidence(asset_class, probability_profit, discount_pct)
            
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
                'probability_profit': probability_profit,
                'confidence_level': confidence_level
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
            
            # Calculate probability of profit
            probability_profit = self._estimate_iron_condor_probability(current_price, lower_breakeven, upper_breakeven, 35)
            
            # Calculate confidence level
            confidence_level = self._calculate_iron_condor_confidence(asset_class, probability_profit, profit_range_pct)
            
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
                'max_profit': total_credit,
                'max_profit_pct': max_profit_pct,
                'max_loss': total_max_loss,
                'upper_breakeven': upper_breakeven,
                'lower_breakeven': lower_breakeven,
                'profit_range': profit_range,
                'profit_range_pct': profit_range_pct,
                'probability_profit': probability_profit,
                'confidence_level': confidence_level
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
            ('Conservative', -0.02, 0.05),    # 2% ITM buy, 5% width
            ('Moderate', 0.00, 0.08),         # ATM buy, 8% width
            ('Aggressive', 0.02, 0.12)       # 2% OTM buy, 12% width
        ]
        
        recommendations = []
        
        for risk_level, buy_offset, spread_width in spread_configs:
            
            buy_strike = current_price * (1 + buy_offset)
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
            
            # Calculate confidence level
            confidence_level = self._calculate_directional_confidence(asset_class, probability_profit, 'bullish')
            
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
                'probability_profit': probability_profit,
                'confidence_level': confidence_level
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
    
    def _estimate_straddle_probability(self, current_price: float, lower_be: float, 
                                     upper_be: float, days: int) -> float:
        """Estimate probability of big move for straddle/strangle strategies"""
        
        volatility = 0.25
        T = days / 365.0
        
        # Probability of being below lower breakeven OR above upper breakeven
        d2_lower = (math.log(current_price / lower_be) + (-0.5 * volatility ** 2) * T) / (volatility * math.sqrt(T))
        prob_below_lower = norm.cdf(-d2_lower)
        
        d2_upper = (math.log(current_price / upper_be) + (-0.5 * volatility ** 2) * T) / (volatility * math.sqrt(T))
        prob_above_upper = norm.cdf(d2_upper)
        
        # Combined probability of big move
        prob_big_move = prob_below_lower + prob_above_upper
        
        return max(0.05, min(0.95, prob_big_move))
    
    # Enhanced confidence calculation methods with FX support
    def _calculate_straddle_confidence(self, asset_class: str, profit_probability: float, is_short: bool = False) -> str:
        """Enhanced straddle confidence with FX adjustments"""
        
        # Adjust base confidence by asset class
        base_confidence = {
            'FOREX': 0.20,      # FX has frequent big moves
            'INDICES': 0.25,    # Indices moderate volatility  
            'EQUITIES': 0.20    # Individual stocks variable
        }.get(asset_class, 0.20)
        
        if is_short:
            # Short straddles work better in low volatility
            if profit_probability > 0.7:
                return "HIGH"
            elif profit_probability > 0.55:
                return "MODERATE"
            else:
                return "LOW"
        else:
            # Long straddles need big moves - FX gets bonus
            threshold_adjustment = 0.05 if asset_class == 'FOREX' else 0.0
            if profit_probability > (0.4 - threshold_adjustment):
                return "HIGH"
            elif profit_probability > (0.25 - threshold_adjustment):
                return "MODERATE"
            else:
                return "LOW"
    
    def _calculate_strangle_confidence(self, asset_class: str, profit_probability: float, is_short: bool = False) -> str:
        """Calculate confidence level for strangle strategies"""
        
        if is_short:
            # Short strangles benefit from range-bound markets
            if profit_probability > 0.65:
                return "HIGH"
            elif profit_probability > 0.5:
                return "MODERATE"
            else:
                return "LOW"
        else:
            # Long strangles need volatility but less than straddles
            threshold_adjustment = 0.05 if asset_class == 'FOREX' else 0.0
            if profit_probability > (0.35 - threshold_adjustment):
                return "HIGH"
            elif profit_probability > (0.2 - threshold_adjustment):
                return "MODERATE"
            else:
                return "LOW"
    
    def _calculate_directional_confidence(self, asset_class: str, profit_probability: float, direction: str) -> str:
        """Calculate confidence level for directional strategies"""
        
        # Base confidence adjustments
        base_adjustments = {
            'FOREX': -0.05,     # FX trends can reverse quickly
            'INDICES': 0.05,    # Indices have cleaner trends
            'EQUITIES': 0.0     # Individual stocks neutral
        }
        
        adjusted_prob = profit_probability + base_adjustments.get(asset_class, 0)
        
        if adjusted_prob > 0.65:
            return "HIGH"
        elif adjusted_prob > 0.5:
            return "MODERATE"
        else:
            return "LOW"
    
    def _calculate_covered_call_confidence(self, asset_class: str, probability_profit: float, premium_yield: float) -> str:
        """Calculate confidence level for covered call strategy"""
        
        # Base confidence from probability
        base_confidence = probability_profit
        
        # Bonus for good premium yield
        if premium_yield > 3.0:  # > 3% monthly premium
            base_confidence += 0.1
        elif premium_yield > 2.0:  # > 2% monthly premium
            base_confidence += 0.05
        
        # Asset class adjustment
        if asset_class == 'INDICES':
            base_confidence += 0.05  # Indices more predictable
        elif asset_class == 'FOREX':
            base_confidence -= 0.05  # FX more volatile
        
        # Convert to confidence level
        if base_confidence > 0.7:
            return "HIGH"
        elif base_confidence > 0.55:
            return "MODERATE"
        else:
            return "LOW"

    def _calculate_csp_confidence(self, asset_class: str, probability_profit: float, discount_pct: float) -> str:
        """Calculate confidence level for cash secured put strategy"""
        
        base_confidence = probability_profit
        
        # Bonus for good discount opportunity
        if discount_pct > 10:  # Getting stock at >10% discount
            base_confidence += 0.15
        elif discount_pct > 5:  # Getting stock at >5% discount
            base_confidence += 0.08
        
        # Asset class adjustment
        if asset_class == 'EQUITIES':
            base_confidence += 0.03  # Individual stocks better for CSP
        
        if base_confidence > 0.68:
            return "HIGH"
        elif base_confidence > 0.52:
            return "MODERATE"
        else:
            return "LOW"

    def _calculate_iron_condor_confidence(self, asset_class: str, probability_profit: float, profit_range_pct: float) -> str:
        """Calculate confidence level for iron condor strategy"""
        
        base_confidence = probability_profit
        
        # Bonus for wider profit range
        if profit_range_pct > 12:  # Wide profit range
            base_confidence += 0.1
        elif profit_range_pct > 8:  # Moderate profit range
            base_confidence += 0.05
        
        # Asset class adjustment - indices better for iron condors
        if asset_class == 'INDICES':
            base_confidence += 0.08
        elif asset_class == 'FOREX':
            base_confidence -= 0.05  # FX more volatile
        
        if base_confidence > 0.72:
            return "HIGH"
        elif base_confidence > 0.58:
            return "MODERATE"
        else:
            return "LOW"
    
    # =============================================================================
    # COMPREHENSIVE STRATEGY ANALYZER WITH RANKING
    # =============================================================================
    
    def analyze_all_strategies(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                              current_price: float, capital: float, asset_class: str, 
                              max_risk_amount: float, market_analysis: Dict) -> Dict:
        """Analyze all strategies and rank by profitability with confidence levels"""
        
        strategies_to_analyze = [
            ('COVERED_CALL', self.calculate_covered_call_accurate),
            ('CASH_SECURED_PUT', self.calculate_cash_secured_put_accurate),
            ('IRON_CONDOR', self.calculate_iron_condor_accurate),
            ('BULL_CALL_SPREAD', self.calculate_bull_call_spread_accurate),
            ('LONG_STRADDLE', self.calculate_long_straddle_accurate),
            ('SHORT_STRADDLE', self.calculate_short_straddle_accurate),
            ('LONG_STRANGLE', self.calculate_long_strangle_accurate),
            ('SHORT_STRANGLE', self.calculate_short_strangle_accurate),
        ]
        
        strategy_results = []
        
        for strategy_name, strategy_func in strategies_to_analyze:
            try:
                if strategy_name in ['COVERED_CALL', 'BULL_CALL_SPREAD']:
                    result = strategy_func(calls_df, current_price, capital, asset_class, max_risk_amount)
                elif strategy_name in ['CASH_SECURED_PUT']:
                    result = strategy_func(puts_df, current_price, capital, asset_class, max_risk_amount)
                else:  # Strategies requiring both calls and puts
                    result = strategy_func(calls_df, puts_df, current_price, capital, asset_class, max_risk_amount)
                
                if 'error' not in result:
                    # Calculate profitability score and market suitability
                    profitability_score = self._calculate_profitability_score(result, strategy_name, market_analysis)
                    market_suitability = self._assess_market_suitability(strategy_name, market_analysis, asset_class)
                    strategy_explanation = self._get_strategy_explanation(strategy_name)
                    
                    # Add metadata to result
                    result['strategy_name'] = strategy_name
                    result['profitability_score'] = profitability_score
                    result['market_suitability'] = market_suitability
                    result['strategy_explanation'] = strategy_explanation
                    result['success'] = True
                    
                    strategy_results.append(result)
            
            except Exception as e:
                print(f"Strategy {strategy_name} failed: {str(e)}")
                continue
        
        # Sort by profitability score (highest first)
        strategy_results.sort(key=lambda x: x['profitability_score'], reverse=True)
        
        return {
            'strategies': strategy_results,
            'total_analyzed': len(strategy_results),
            'market_conditions': market_analysis,
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital
        }
    
    def _calculate_profitability_score(self, strategy_result: Dict, strategy_name: str, market_analysis: Dict) -> float:
        """Calculate comprehensive profitability score (0-100)"""
        
        score = 50.0  # Base score
        
        # Expected return component (40% weight)
        if 'max_profit_pct' in strategy_result:
            profit_pct = strategy_result['max_profit_pct']
            score += min(profit_pct * 0.8, 25)  # Cap at 25 points
        elif 'estimated_profit' in strategy_result and 'total_cost' in strategy_result:
            if strategy_result['total_cost'] > 0:
                estimated_return = (strategy_result['estimated_profit'] / strategy_result['total_cost']) * 100
                score += min(estimated_return * 0.4, 20)  # Cap at 20 points
        
        # Probability of profit component (30% weight)
        if 'probability_profit' in strategy_result:
            prob = strategy_result['probability_profit']
            score += (prob * 30)  # 0-30 points based on probability
        
        # Risk-adjusted return (20% weight)
        if 'max_loss' in strategy_result and strategy_result['max_loss'] > 0:
            max_profit = strategy_result.get('max_profit', strategy_result.get('estimated_profit', 0))
            if max_profit > 0:
                risk_reward_ratio = max_profit / strategy_result['max_loss']
                score += min(risk_reward_ratio * 10, 15)  # Cap at 15 points
        
        # Market condition alignment (10% weight)
        trend = market_analysis.get('trend', 'SIDEWAYS')
        volatility = market_analysis.get('volatility_regime', 'NORMAL_VOL')
        
        # Strategy-specific market alignment
        alignment_bonus = self._get_market_alignment_bonus(strategy_name, trend, volatility)
        score += alignment_bonus
        
        return max(0, min(100, score))
    
    def _get_market_alignment_bonus(self, strategy_name: str, trend: str, volatility: str) -> float:
        """Get bonus points for market condition alignment"""
        
        bonus = 0.0
        
        # Trend alignment
        bullish_strategies = ['COVERED_CALL', 'BULL_CALL_SPREAD', 'CASH_SECURED_PUT']
        neutral_strategies = ['IRON_CONDOR', 'SHORT_STRADDLE', 'SHORT_STRANGLE']
        volatile_strategies = ['LONG_STRADDLE', 'LONG_STRANGLE']
        
        if strategy_name in bullish_strategies and 'BULLISH' in trend:
            bonus += 8
        elif strategy_name in neutral_strategies and trend == 'SIDEWAYS':
            bonus += 10
        elif strategy_name in volatile_strategies and volatility in ['HIGH_VOL', 'EXTREME_VOL']:
            bonus += 12
        
        # Volatility alignment
        if strategy_name in ['SHORT_STRADDLE', 'SHORT_STRANGLE', 'IRON_CONDOR'] and volatility == 'LOW_VOL':
            bonus += 5
        elif strategy_name in ['LONG_STRADDLE', 'LONG_STRANGLE'] and volatility in ['HIGH_VOL', 'EXTREME_VOL']:
            bonus += 5
        
        return bonus


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
# ENHANCED ML PREDICTOR CLASS WITH COMPREHENSIVE FX SUPPORT
# =============================================================================

class MLPredictor:
    """Enhanced Machine Learning Stock Price Direction Predictor with FX support"""
    
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
        """Enhanced prediction with comprehensive FX-specific adjustments"""
        
        try:
            asset_class = data.get('asset_class', 'EQUITIES')
            
            # Enhanced prediction logic with comprehensive FX considerations
            score_factors = []
            
            # Trend Analysis (40% weight) - Enhanced FX adjustments
            trend = analysis['trend']
            trend_scores = {
                'STRONG_BULLISH': 0.4,
                'BULLISH': 0.2,
                'SIDEWAYS': 0.0,
                'BEARISH': -0.2,
                'STRONG_BEARISH': -0.4
            }
            trend_score = trend_scores.get(trend, 0)
            
            # FX trends can be stronger and more persistent
            if asset_class == 'FOREX':
                trend_score *= 1.2  # Increased multiplier for FX
            
            score_factors.append(trend_score)
            
            # RSI Analysis (25% weight) - Enhanced FX-specific thresholds
            rsi = analysis['rsi']
            if asset_class == 'FOREX':
                # FX markets are more volatile, adjust thresholds
                if rsi < 20:
                    rsi_score = 0.3  # Very oversold - strongly bullish
                elif rsi < 30:
                    rsi_score = 0.2  # Oversold - bullish
                elif rsi < 40:
                    rsi_score = 0.1  # Moderately oversold
                elif rsi > 80:
                    rsi_score = -0.3  # Very overbought - strongly bearish
                elif rsi > 70:
                    rsi_score = -0.2  # Overbought - bearish
                elif rsi > 60:
                    rsi_score = -0.1  # Moderately overbought
                else:
                    rsi_score = 0.0  # Neutral
            else:
                # Standard equity/index thresholds
                if rsi < 30:
                    rsi_score = 0.25
                elif rsi < 40:
                    rsi_score = 0.15
                elif rsi > 70:
                    rsi_score = -0.25
                elif rsi > 60:
                    rsi_score = -0.15
                else:
                    rsi_score = 0.0
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
            
            # Volatility Analysis (15% weight) - Enhanced FX-specific interpretation
            vol_regime = analysis['volatility_regime']
            if asset_class == 'FOREX':
                # FX volatility interpretation
                if vol_regime == 'EXTREME_VOL':
                    vol_score = -0.15  # Extreme volatility increases uncertainty significantly
                elif vol_regime == 'HIGH_VOL':
                    vol_score = -0.08  # High volatility moderate uncertainty
                elif vol_regime == 'LOW_VOL':
                    vol_score = 0.1   # Low volatility favorable for trend continuation
                else:
                    vol_score = 0.05  # Normal volatility slightly favorable
            else:
                if vol_regime in ['EXTREME_VOL', 'HIGH_VOL']:
                    vol_score = -0.05  # High volatility increases uncertainty
                else:
                    vol_score = 0.05  # Normal/low volatility is favorable
            score_factors.append(vol_score)
            
            # Calculate composite score
            composite_score = sum(score_factors)
            
            # Convert to direction and confidence with FX-specific thresholds
            if asset_class == 'FOREX':
                # FX markets need slightly different thresholds
                if composite_score > 0.08:
                    direction = "UP"
                    base_confidence = 60
                elif composite_score < -0.08:
                    direction = "DOWN"
                    base_confidence = 60
                else:
                    # Neutral - use RSI to break tie
                    if rsi > 50:
                        direction = "UP"
                    else:
                        direction = "DOWN"
                    base_confidence = 45
            else:
                # Standard thresholds for equities/indices
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
            
            # Risk and signal strength with FX considerations
            if asset_class == 'FOREX':
                # FX has different confidence thresholds
                if confidence >= 70:
                    risk_level = "LOW"
                    signal_strength = "STRONG"
                elif confidence >= 60:
                    risk_level = "MODERATE"
                    signal_strength = "MODERATE"
                else:
                    risk_level = "HIGH"
                    signal_strength = "WEAK"
            else:
                # Standard thresholds
                if confidence >= 75:
                    risk_level = "LOW"
                    signal_strength = "STRONG"
                elif confidence >= 65:
                    risk_level = "MODERATE"
                    signal_strength = "MODERATE"
                else:
                    risk_level = "HIGH"
                    signal_strength = "WEAK"
            
            # Timeframe based on asset class
            if asset_class == 'FOREX':
                timeframe = '3-day outlook (FX moves faster)'
            else:
                timeframe = '5-day outlook'
            
            return {
                'direction': direction,
                'confidence': round(confidence, 1),
                'probability_up': round(prob_up, 1),
                'probability_down': round(prob_down, 1),
                'risk_level': risk_level,
                'signal_strength': signal_strength,
                'prediction_timeframe': timeframe,
                'composite_score': round(composite_score, 3),
                'asset_class': asset_class,
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}


# =============================================================================
# ENHANCED UI HELPER FUNCTIONS WITH COMPREHENSIVE FX SUPPORT
# =============================================================================

def create_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create professional metric card"""
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
    """Display strategy results with detailed trade instructions - ENHANCED WITH FX SUPPORT"""
    
    # Handle error cases
    if 'error' in results:
        st.error(f"⚠️ {results['error']}")
        if 'min_capital_needed' in results:
            st.warning(f"💡 **Solution:** Increase your capital to at least ${results['min_capital_needed']:,.0f} or consider a different strategy")
        return
    
    # Handle multi-recommendation format (like COVERED_CALL, CASH_SECURED_PUT, etc.)
    if 'recommendations' in results:
        recommendations = results.get('recommendations', [])
        if not recommendations:
            st.warning("No viable recommendations found with current capital settings")
            return
        strategy_name = results['strategy']
        
        # Create tabs for different risk levels if multiple recommendations
        if len(recommendations) > 1:
            risk_tabs = st.tabs([f"{rec.get('risk_level', 'Standard')} Risk" for rec in recommendations])
            
            for tab, rec in zip(risk_tabs, recommendations):
                with tab:
                    display_single_trade_instruction(rec, asset_class, strategy_name, expiration_date, max_risk_amount)
        else:
            display_single_trade_instruction(recommendations[0], asset_class, strategy_name, expiration_date, max_risk_amount)
    
    # Handle single strategy format (like LONG_STRADDLE, etc.)
    else:
        strategy_name = results.get('strategy_name', results.get('strategy', 'UNKNOWN'))
        display_single_trade_instruction(results, asset_class, strategy_name, expiration_date, max_risk_amount)

def display_single_trade_instruction(rec: Dict, asset_class: str, strategy: str, expiration: str, max_risk_amount: float):
    """Display detailed trade instructions for a single recommendation - Enhanced with FX support"""
    
    # Enhanced format prices based on asset class
    if asset_class == 'FOREX':
        price_format = "{:.5f}"
        unit_description = "FX units"
        share_equivalent = 10000
    else:
        price_format = "${:.2f}"
        unit_description = "shares" if asset_class == 'EQUITIES' else "units"
        share_equivalent = 100
    
    # Trade Instruction Header
    risk_level = rec.get('risk_level', 'Standard')
    st.markdown(f"### 🎯 **{risk_level} {strategy.replace('_', ' ').title()}** Trade Setup")
    
    # Enhanced risk check with FX considerations
    max_loss = rec.get('max_loss', rec.get('total_cost', rec.get('total_margin', 0)))
    if asset_class == 'FOREX':
        risk_color = "🟢" if max_loss <= max_risk_amount else "🟠" if max_loss <= max_risk_amount * 1.2 else "🔴"  # More lenient for FX
    else:
        risk_color = "🟢" if max_loss <= max_risk_amount else "🟠" if max_loss <= max_risk_amount * 1.5 else "🔴"
    
    st.markdown(f"{risk_color} **Risk Level:** ${max_loss:,.0f} max loss ({'Within' if max_loss <= max_risk_amount else 'Exceeds'} your ${max_risk_amount:,.0f} limit)")
    
    # Detailed Trade Instructions
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📋 **TRADE INSTRUCTIONS**")
        
        if strategy == 'COVERED_CALL':
            st.markdown(f"""
            **Step 1:** Buy {rec['contracts']} lots of underlying
            - **Quantity:** {rec['contracts'] * share_equivalent:,} {unit_description}
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
            - Buy {rec['contracts'] * share_equivalent:,} {unit_description} at {price_format.format(rec['strike'])}
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
            - **Premium Paid:** ${rec['contracts'] * rec['buy_price'] * share_equivalent:,.0f}
            
            **Step 2:** Sell Short Call
            - **Sell:** {rec['contracts']} calls at {price_format.format(rec['sell_strike'])}
            - **Premium Received:** ${rec['contracts'] * rec['sell_price'] * share_equivalent:,.0f}
            
            **Net Debit:** ${rec['total_cost']:,.0f}
            **Expiration:** {expiration}
            """)
        
        elif strategy == 'LONG_STRADDLE':
            st.markdown(f"""
            **Step 1:** Buy ATM Call
            - **Buy:** {rec['contracts']} calls at {price_format.format(rec['strike'])}
            - **Premium Paid:** ${rec['contracts'] * rec['call_premium'] * share_equivalent:,.0f}
            
            **Step 2:** Buy ATM Put
            - **Buy:** {rec['contracts']} puts at {price_format.format(rec['strike'])}
            - **Premium Paid:** ${rec['contracts'] * rec['put_premium'] * share_equivalent:,.0f}
            
            **Total Cost:** ${rec['total_cost']:,.0f}
            **Expiration:** {expiration}
            """)
        
        elif strategy == 'SHORT_STRADDLE':
            st.markdown(f"""
            **Step 1:** Sell ATM Call
            - **Sell:** {rec['contracts']} calls at {price_format.format(rec['strike'])}
            - **Premium Received:** ${rec['contracts'] * rec['call_premium'] * share_equivalent:,.0f}
            
            **Step 2:** Sell ATM Put
            - **Sell:** {rec['contracts']} puts at {price_format.format(rec['strike'])}
            - **Premium Received:** ${rec['contracts'] * rec['put_premium'] * share_equivalent:,.0f}
            
            **Total Credit:** ${rec['total_credit']:,.0f}
            **Margin Required:** ${rec['total_margin']:,.0f}
            **Expiration:** {expiration}
            """)
        
        elif strategy == 'LONG_STRANGLE':
            st.markdown(f"""
            **Step 1:** Buy OTM Call
            - **Buy:** {rec['contracts']} calls at {price_format.format(rec['call_strike'])}
            - **Premium Paid:** ${rec['contracts'] * rec['call_premium'] * share_equivalent:,.0f}
            
            **Step 2:** Buy OTM Put
            - **Buy:** {rec['contracts']} puts at {price_format.format(rec['put_strike'])}
            - **Premium Paid:** ${rec['contracts'] * rec['put_premium'] * share_equivalent:,.0f}
            
            **Total Cost:** ${rec['total_cost']:,.0f}
            **Expiration:** {expiration}
            """)
        
        elif strategy == 'SHORT_STRANGLE':
            st.markdown(f"""
            **Step 1:** Sell OTM Call
            - **Sell:** {rec['contracts']} calls at {price_format.format(rec['call_strike'])}
            - **Premium Received:** ${rec['contracts'] * rec['call_premium'] * share_equivalent:,.0f}
            
            **Step 2:** Sell OTM Put
            - **Sell:** {rec['contracts']} puts at {price_format.format(rec['put_strike'])}
            - **Premium Received:** ${rec['contracts'] * rec['put_premium'] * share_equivalent:,.0f}
            
            **Total Credit:** ${rec['total_credit']:,.0f}
            **Margin Required:** ${rec['total_margin']:,.0f}
            **Expiration:** {expiration}
            """)
    
    with col2:
        st.markdown("#### 📊 **PROFIT/LOSS PROFILE**")
        
        # P&L Metrics
        col_a, col_b = st.columns(2)
        
        with col_a:
            if 'max_profit' in rec:
                create_metric_card("Max Profit", f"${rec['max_profit']:,.0f}")
            elif 'estimated_profit' in rec:
                create_metric_card("Est. Profit", f"${rec['estimated_profit']:,.0f}")
            
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
        
        # Key Levels with FX formatting
        st.markdown("**🎯 Key Price Levels:**")
        
        if 'breakeven' in rec:
            st.write(f"• **Breakeven:** {price_format.format(rec['breakeven'])}")
        
        if 'upper_breakeven' in rec and 'lower_breakeven' in rec:
            st.write(f"• **Profit Range:** {price_format.format(rec['lower_breakeven'])} - {price_format.format(rec['upper_breakeven'])}")
            if 'profit_range_pct' in rec:
                st.write(f"• **Range Width:** {rec['profit_range_pct']:.1f}%")
            elif 'breakeven_range_pct' in rec:
                st.write(f"• **Range Width:** {rec['breakeven_range_pct']:.1f}%")
        
        # Time-based metrics
        if 'annualized_yield' in rec:
            st.write(f"• **Annualized Yield:** {rec['annualized_yield']:.1f}%")
        
        # Enhanced risk assessment with FX considerations
        if asset_class == 'FOREX':
            if max_loss <= max_risk_amount * 0.8:
                risk_assessment = "🟢 **Low Risk** - Well within FX risk tolerance"
            elif max_loss <= max_risk_amount * 1.2:
                risk_assessment = "🟡 **Moderate Risk** - Acceptable for FX volatility"
            else:
                risk_assessment = "🔴 **High Risk** - Exceeds recommended FX risk tolerance"
        else:
            if max_loss <= max_risk_amount * 0.5:
                risk_assessment = "🟢 **Low Risk** - Well within your risk tolerance"
            elif max_loss <= max_risk_amount:
                risk_assessment = "🟡 **Moderate Risk** - At your risk limit"
            else:
                risk_assessment = "🔴 **High Risk** - Exceeds your risk tolerance"
        
        st.markdown(f"**Risk Assessment:** {risk_assessment}")
    
    # Enhanced Market Conditions Suitability with FX context
    st.markdown("#### 🌡️ **Market Suitability**")
    
    suitability_messages = {
        'COVERED_CALL': f"Best in neutral to slightly bullish {asset_class.lower()} markets with elevated volatility",
        'CASH_SECURED_PUT': f"Ideal when you want to own the {asset_class.lower()} asset at a lower price with bullish outlook",
        'IRON_CONDOR': f"Perfect for range-bound, low-volatility {asset_class.lower()} environments",
        'BULL_CALL_SPREAD': f"Suitable for moderately bullish {asset_class.lower()} outlook with limited capital",
        'LONG_STRADDLE': f"Best before expected volatility expansion or major {asset_class.lower()} events",
        'SHORT_STRADDLE': f"Ideal for range-bound {asset_class.lower()} markets with high implied volatility",
        'LONG_STRANGLE': f"Perfect for expecting big {asset_class.lower()} moves but cheaper than straddle",
        'SHORT_STRANGLE': f"Great for stable {asset_class.lower()} markets with elevated implied volatility",
    }
    
    if asset_class == 'FOREX':
        fx_note = " 💱 **FX Note:** Currency markets move faster and with higher frequency - consider shorter timeframes and tighter risk management."
        suitability_messages[strategy] += fx_note
    
    st.write(f"💡 **Strategy Note:** {suitability_messages.get(strategy, 'Professional options strategy')}")
    
    # Enhanced Action Button with FX considerations
    if asset_class == 'FOREX':
        if max_loss <= max_risk_amount * 1.2:  # More lenient for FX
            st.success("✅ **Ready for FX Trading** - This setup fits your FX risk parameters")
        else:
            adjusted_contracts = int(rec['contracts'] * max_risk_amount / max_loss)
            st.warning(f"⚠️ **Consider Reducing Position** - Reduce to {adjusted_contracts} contracts for FX risk management")
    else:
        if max_loss <= max_risk_amount:
            st.success("✅ **Ready to Trade** - This setup fits your risk parameters")
        else:
            st.warning(f"⚠️ **Consider Reducing Position** - Reduce to {int(rec['contracts'] * max_risk_amount / max_loss)} contracts to fit your risk limit")

def display_ml_predictions(prediction_result: Dict, ticker: str):
    """Enhanced ML prediction display with comprehensive FX support"""
    
    if not prediction_result.get('success'):
        st.error(f"Prediction failed: {prediction_result.get('error')}")
        return
    
    asset_class = prediction_result.get('asset_class', 'EQUITIES')
    asset_emoji = "💱" if asset_class == 'FOREX' else "📊" if asset_class == 'INDICES' else "📈"
    
    st.markdown(f"### 🤖 **AI Prediction for {ticker}** {asset_emoji}")
    
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
    
    # Enhanced trading suggestions based on asset class and prediction
    st.markdown("#### 💡 **Trading Suggestions**")
    
    if asset_class == 'FOREX':
        # FX-specific suggestions
        if direction == "UP" and confidence >= 70:
            st.success(f"""
            **Strong Bullish Signal for {ticker} - Consider:**
            • Long positions in the base currency
            • Consider FX CFDs or spot trading
            • For options: Use {ticker} ETF alternatives (FXE, FXB, etc.)
            • Watch for central bank announcements
            • Monitor economic data releases
            • Set tight stops due to FX volatility
            """)
        elif direction == "UP" and confidence >= 60:
            st.info(f"""
            **Moderate Bullish Signal for {ticker} - Consider:**
            • Small long positions with tight stops
            • Monitor economic data releases
            • Wait for pullbacks to key support levels
            • Consider correlation with other major pairs
            • Use proper FX position sizing
            """)
        elif direction == "DOWN" and confidence >= 70:
            st.warning(f"""
            **Strong Bearish Signal for {ticker} - Consider:**
            • Short positions in the base currency
            • Hedge existing FX exposure
            • Watch for support level breaks
            • Monitor risk-off sentiment in markets
            • Consider safe-haven currency flows
            """)
        elif direction == "DOWN" and confidence >= 60:
            st.warning(f"""
            **Moderate Bearish Signal for {ticker} - Consider:**
            • Reduced position sizes
            • Defensive positioning
            • Wait for better entry points
            • Watch for oversold bounces
            • Monitor central bank interventions
            """)
        else:
            st.info(f"""
            **Low Confidence Signal for {ticker}:**
            • Range trading opportunities
            • Wait for stronger directional signals
            • Focus on key support/resistance levels
            • Consider mean reversion strategies
            • Monitor breakout attempts
            """)
    else:
        # Standard equity/index suggestions
        if direction == "UP" and confidence >= 70:
            st.success("""
            **Strong Bullish Signal - Consider:**
            • Bull call spreads for leveraged upside
            • Covered calls if you own shares (capture premium + upside)
            • Cash-secured puts to enter at lower prices
            • Long call options for maximum leverage
            """)
        elif direction == "UP" and confidence >= 60:
            st.info("""
            **Moderate Bullish Signal - Consider:**
            • Conservative call options
            • Small position sizes
            • Monitor for trend confirmation
            • Consider protective stops
            """)
        elif direction == "DOWN" and confidence >= 70:
            st.warning("""
            **Strong Bearish Signal - Consider:**
            • Protective puts if holding positions
            • Bear put spreads for defined risk
            • Avoid call options
            • Consider short positions (with caution)
            """)
        elif direction == "DOWN" and confidence >= 60:
            st.warning("""
            **Moderate Bearish Signal - Consider:**
            • Reduced position sizes
            • Defensive strategies
            • Wait for better entry points
            • Consider hedging existing positions
            """)
        else:
            st.info("""
            **Low Confidence Signal:**
            • Range-bound strategies (iron condors)
            • Wait for stronger signals
            • Focus on premium collection strategies
            • Avoid directional bets
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
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    
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
    
    # Main Content Tabs with Enhanced Styling - 4 TABS
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
                
                # Enhanced Strategy Analysis with Profitability Ranking
                st.markdown("---")
                st.subheader("🎯 Professional Strategy Recommendations (Ranked by Profitability)")
                
                with st.spinner("🔍 Analyzing all strategies and ranking by profitability..."):
                    # Get market analysis
                    market_analysis = strategist.analyze_market_conditions(underlying)
                    
                    # Analyze all strategies comprehensively
                    comprehensive_analysis = strategist.analyze_all_strategies(
                        options['calls'], options['puts'], underlying['current_price'], 
                        available_capital, asset_class, max_risk_amount, market_analysis
                    )
                
                if comprehensive_analysis['strategies']:
                    # Display market conditions summary
                    st.markdown("#### 🌡️ **Current Market Conditions**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        trend = market_analysis['trend'].replace('_', ' ').title()
                        trend_color = "🟢" if 'BULLISH' in market_analysis['trend'] else "🔴" if 'BEARISH' in market_analysis['trend'] else "🟡"
                        create_metric_card("Market Trend", f"{trend_color} {trend}")
                    
                    with col2:
                        vol_regime = market_analysis['volatility_regime'].replace('_', ' ').title()
                        vol_color = "🔴" if 'HIGH' in vol_regime or 'EXTREME' in vol_regime else "🟡" if 'NORMAL' in vol_regime else "🟢"
                        create_metric_card("Volatility", f"{vol_color} {vol_regime}")
                    
                    with col3:
                        momentum = market_analysis['momentum'].replace('_', ' ').title()
                        mom_color = "🟢" if 'BULLISH' in momentum or 'OVERBOUGHT' in momentum else "🔴" if 'BEARISH' in momentum or 'OVERSOLD' in momentum else "🟡"
                        create_metric_card("Momentum", f"{mom_color} {momentum}")
                    
                    with col4:
                        rsi = market_analysis['rsi']
                        rsi_color = "🔴" if rsi > 70 or rsi < 30 else "🟡" if rsi > 60 or rsi < 40 else "🟢"
                        create_metric_card("RSI", f"{rsi_color} {rsi:.1f}")
                    
                    st.markdown("---")
                    
                    # Display strategies ranked by profitability
                    st.markdown("#### 🏆 **Strategy Rankings** (Most Profitable to Least)")
                    
                    # Create tabs for top strategies vs all strategies
                    strategy_tabs = st.tabs(["🥇 Top 5 Strategies", "📊 All Strategies"])
                    
                    with strategy_tabs[0]:  # Top 5 Strategies
                        st.markdown("##### 🎯 **Highest Ranked Strategies for Current Market**")
                        
                        for i, strategy in enumerate(comprehensive_analysis['strategies'][:5]):
                            rank = i + 1
                            strategy_name = strategy['strategy_name']
                            profitability_score = strategy['profitability_score']
                            market_suitability = strategy['market_suitability']
                            explanation = strategy['strategy_explanation']
                            
                            # Strategy ranking header with medals
                            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                            medal = medals[i] if i < 5 else f"{rank}️⃣"
                            
                            with st.expander(
                                f"{medal} **{strategy_name.replace('_', ' ').title()}** "
                                f"(Score: {profitability_score:.1f}/100 | "
                                f"Market Fit: {market_suitability['color']} {market_suitability['level']})",
                                expanded=(i == 0)  # Expand top strategy
                            ):
                                # Strategy overview
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"**📋 Strategy Overview**")
                                    st.write(f"• **Summary:** {explanation['summary']}")
                                    st.write(f"• **Mechanics:** {explanation['mechanics']}")
                                    st.write(f"• **Profit Source:** {explanation['profit_source']}")
                                    st.write(f"• **Best When:** {explanation['best_when']}")
                                    
                                    st.markdown(f"**🎯 Market Analysis**")
                                    st.write(f"• **Current Suitability:** {market_suitability['color']} {market_suitability['level']}")
                                    st.write(f"• **Reason:** {market_suitability['description']}")
                                
                                with col2:
                                    # Key metrics
                                    st.markdown(f"**📊 Key Metrics**")
                                    
                                    if 'probability_profit' in strategy:
                                        prob = strategy['probability_profit'] * 100
                                        prob_color = "🟢" if prob > 60 else "🟡" if prob > 40 else "🔴"
                                        st.write(f"• **Win Probability:** {prob_color} {prob:.0f}%")
                                    
                                    if 'confidence_level' in strategy:
                                        conf = strategy['confidence_level']
                                        conf_color = "🟢" if conf == "HIGH" else "🟡" if conf == "MODERATE" else "🔴"
                                        st.write(f"• **Confidence:** {conf_color} {conf}")
                                    
                                    st.write(f"• **Risk Level:** {explanation['risk_level']}")
                                    st.write(f"• **Complexity:** {explanation['complexity']}")
                                    
                                    if 'max_profit_pct' in strategy:
                                        st.write(f"• **Max Return:** {strategy['max_profit_pct']:.1f}%")
                                    elif 'estimated_profit' in strategy and 'total_cost' in strategy and strategy['total_cost'] > 0:
                                        est_return = (strategy['estimated_profit'] / strategy['total_cost']) * 100
                                        st.write(f"• **Est. Return:** {est_return:.1f}%")
                                
                                # Detailed trade instructions
                                st.markdown("---")
                                display_enhanced_strategy_results(
                                    strategy, 
                                    asset_class, options['expiration'], max_risk_amount
                                )
                    
                    with strategy_tabs[1]:  # All Strategies
                        st.markdown("##### 📊 **Complete Strategy Analysis**")
                        
                        # Create a dataframe for better display
                        strategy_data = []
                        for i, strategy in enumerate(comprehensive_analysis['strategies']):
                            prob = strategy.get('probability_profit', 0) * 100
                            conf = strategy.get('confidence_level', 'N/A')
                            market_fit = strategy['market_suitability']['level']
                            
                            max_profit = 'N/A'
                            if 'max_profit_pct' in strategy:
                                max_profit = f"{strategy['max_profit_pct']:.1f}%"
                            elif 'estimated_profit' in strategy and 'total_cost' in strategy and strategy['total_cost'] > 0:
                                est_return = (strategy['estimated_profit'] / strategy['total_cost']) * 100
                                max_profit = f"{est_return:.1f}%"
                            
                            strategy_data.append({
                                'Rank': i + 1,
                                'Strategy': strategy['strategy_name'].replace('_', ' ').title(),
                                'Score': f"{strategy['profitability_score']:.1f}",
                                'Win Rate': f"{prob:.0f}%",
                                'Confidence': conf,
                                'Market Fit': market_fit,
                                'Max Return': max_profit,
                                'Risk Level': strategy['strategy_explanation']['risk_level']
                            })
                        
                        strategy_df = pd.DataFrame(strategy_data)
                        st.dataframe(strategy_df, use_container_width=True, height=400)
                        
                        # Strategy selection for detailed view
                        st.markdown("---")
                        st.markdown("**🔍 Select Strategy for Detailed Analysis:**")
                        
                        strategy_names = [s['strategy_name'] for s in comprehensive_analysis['strategies']]
                        strategy_display_names = [s.replace('_', ' ').title() for s in strategy_names]
                        
                        selected_strategy_idx = st.selectbox(
                            "Choose Strategy:",
                            options=range(len(strategy_display_names)),
                            format_func=lambda x: f"{x+1}. {strategy_display_names[x]} (Score: {comprehensive_analysis['strategies'][x]['profitability_score']:.1f})",
                            key="detailed_strategy_selector"
                        )
                        
                        if selected_strategy_idx is not None:
                            selected_strategy = comprehensive_analysis['strategies'][selected_strategy_idx]
                            
                            st.markdown(f"#### 📋 **{selected_strategy['strategy_name'].replace('_', ' ').title()} - Detailed Analysis**")
                            
                            # Display enhanced results
                            display_enhanced_strategy_results(
                                selected_strategy, 
                                asset_class, options['expiration'], max_risk_amount
                            )
                
                else:
                    st.warning("No suitable strategies found for current capital and market conditions.")
                    
                    # Suggestions for improvement
                    st.markdown("#### 💡 **Suggestions**")
                    
                    if available_capital < 10000:
                        st.write("• **Increase Capital**: Consider strategies like spreads which require less capital")
                    
                    if max_risk_amount < 1000:
                        st.write("• **Increase Risk Tolerance**: Higher risk tolerance allows for more strategies")
                    
                    st.write("• **Try Different Asset Class**: Some asset classes may have better opportunities")
                    st.write("• **Wait for Better Market Conditions**: Current conditions may not favor options strategies")

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
    
    # Tab 2: Enhanced Backtesting with User Capital
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
            backtest_symbol = st.text_input("Symbol", value=default_symbols[asset_class], key="backtest_symbol")
        
        with col2:
            backtest_strategy = st.selectbox(
                "Strategy",
                ['COVERED_CALL', 'CASH_SECURED_PUT', 'IRON_CONDOR', 'BULL_CALL_SPREAD', 
                 'BEAR_PUT_SPREAD', 'BEAR_CALL_SPREAD', 'LONG_STRADDLE', 'SHORT_STRADDLE',
                 'LONG_STRANGLE', 'SHORT_STRANGLE', 'PROTECTIVE_PUT', 'COLLAR', 'BUY_AND_HOLD'],
                format_func=lambda x: x.replace('_', ' ').title(),
                key="backtest_strategy"
            )
        
        with col3:
            backtest_period = st.selectbox("Period", ['6M', '1Y', '2Y'], index=1, key="backtest_period")
        
        with col4:
            backtest_capital = st.number_input(
                "Starting Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=available_capital,
                step=1000,
                key="backtest_capital"
            )
        
        with col5:
            expiry_days = st.selectbox(
                "Expiry Cycle (Days)", 
                [15, 30, 45, 60], 
                index=1,
                help="How often to roll positions (15=bi-weekly, 30=monthly, etc.)",
                key="expiry_days"
            )
        
        # Clear results button
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            run_backtest = st.button("🔄 Run Enhanced Backtest", type="primary", key="run_backtest_btn")
        
        with col_btn2:
            if st.button("🗑️ Clear Results", key="clear_backtest_btn") and st.session_state.backtest_results:
                st.session_state.backtest_results = None
                st.rerun()
        
        # Run backtest
        if run_backtest:
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
                    
                    # Store results in session state
                    if backtest_result['success']:
                        st.session_state.backtest_results = {
                            'result': backtest_result,
                            'symbol': backtest_symbol.upper(),
                            'strategy': backtest_strategy,
                            'period': backtest_period,
                            'capital': backtest_capital,
                            'expiry_days': expiry_days,
                            'start_date': start_date,
                            'end_date': end_date
                        }
                        st.success(f"✅ Backtest completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Backtest failed: {backtest_result.get('error')}")
                
                except Exception as e:
                    st.error(f"Backtest error: {str(e)}")
        
        # Display results if available
        if st.session_state.backtest_results:
            backtest_data = st.session_state.backtest_results
            backtest_result = backtest_data['result']
            metrics = backtest_result['performance_metrics']
            trades = backtest_result['results']['trades']
            
            # Enhanced Results Display
            st.success(f"✅ Backtest Results for {backtest_data['symbol']} ({backtest_data['strategy']}) - {len(trades)} trades executed")
            
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
                dates = pd.date_range(start=backtest_data['start_date'], end=backtest_data['end_date'], periods=len(portfolio_values))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00ff88', width=3)
                ))
                
                fig.add_hline(
                    y=backtest_data['capital'],
                    line_dash="dash",
                    line_color="white",
                    annotation_text="Starting Capital"
                )
                
                fig.update_layout(
                    title=f"{backtest_data['strategy']} Performance",
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual Trades Analysis
            st.markdown("---")
            st.subheader("📋 Individual Trades Analysis")
            
            if trades:
                # Trades summary stats
                winning_trades = [t for t in trades if t.get('total_pnl', 0) > 0]
                losing_trades = [t for t in trades if t.get('total_pnl', 0) <= 0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    create_metric_card("Total Trades", str(len(trades)))
                
                with col2:
                    create_metric_card("Winning Trades", str(len(winning_trades)))
                
                with col3:
                    create_metric_card("Losing Trades", str(len(losing_trades)))
                
                with col4:
                    avg_trade_pnl = sum(t.get('total_pnl', 0) for t in trades) / len(trades)
                    color = "normal" if avg_trade_pnl > 0 else "inverse"
                    create_metric_card("Avg Trade P&L", f"${avg_trade_pnl:.0f}", None, color)
                
                # Detailed Trades Table
                st.markdown("#### 📊 **Complete Trade History**")
                
                # Create trades DataFrame
                trades_data = []
                
                for i, trade in enumerate(trades, 1):
                    # Format data based on strategy
                    if backtest_data['strategy'] == 'COVERED_CALL':
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Call Strike': f"${trade['call_strike']:.2f}",
                            'Premium': f"${trade['premium_received']:.0f}",
                            'Stock P&L': f"${trade['stock_pnl']:.0f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    elif backtest_data['strategy'] == 'CASH_SECURED_PUT':
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Put Strike': f"${trade['put_strike']:.2f}",
                            'Premium': f"${trade['premium_received']:.0f}",
                            'Assigned': '✅' if trade['assigned'] else '❌',
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    elif backtest_data['strategy'] == 'IRON_CONDOR':
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Call Strikes': f"${trade['call_sell_strike']:.2f}/{trade['call_buy_strike']:.2f}",
                            'Put Strikes': f"${trade['put_sell_strike']:.2f}/{trade['put_buy_strike']:.2f}",
                            'Net Credit': f"${trade['net_credit']:.2f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    elif backtest_data['strategy'] in ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']:
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Buy Strike': f"${trade['buy_strike']:.2f}",
                            'Sell Strike': f"${trade['sell_strike']:.2f}",
                            'Net Debit': f"${trade['net_debit']:.2f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    elif backtest_data['strategy'] == 'BEAR_CALL_SPREAD':
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Sell Strike': f"${trade['sell_strike']:.2f}",
                            'Buy Strike': f"${trade['buy_strike']:.2f}",
                            'Net Credit': f"${trade['net_credit']:.2f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    elif backtest_data['strategy'] in ['LONG_STRADDLE', 'SHORT_STRADDLE']:
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Strike': f"${trade['strike']:.2f}",
                            'Call Premium': f"${trade['call_premium']:.2f}",
                            'Put Premium': f"${trade['put_premium']:.2f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    elif backtest_data['strategy'] in ['LONG_STRANGLE', 'SHORT_STRANGLE']:
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Call Strike': f"${trade['call_strike']:.2f}",
                            'Put Strike': f"${trade['put_strike']:.2f}",
                            'Total Premium': f"${trade['total_premium']:.2f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    elif backtest_data['strategy'] == 'PROTECTIVE_PUT':
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Put Strike': f"${trade['put_strike']:.2f}",
                            'Stock Investment': f"${trade['stock_investment']:.0f}",
                            'Put Investment': f"${trade['put_investment']:.0f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    elif backtest_data['strategy'] == 'COLLAR':
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Put Strike': f"${trade['put_strike']:.2f}",
                            'Call Strike': f"${trade['call_strike']:.2f}",
                            'Net Option Cost': f"${trade['net_option_cost']:.2f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                    
                    else:  # BUY_AND_HOLD or other
                        trades_data.append({
                            'Trade #': i,
                            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                            'Entry Price': f"${trade['entry_price']:.2f}",
                            'Exit Price': f"${trade['exit_price']:.2f}",
                            'Shares': f"{trade.get('shares', 0):.0f}",
                            'Total P&L': f"${trade['total_pnl']:.0f}",
                            'Portfolio Value': f"${trade['portfolio_value']:.0f}"
                        })
                
                # Display trades table
                trades_df = pd.DataFrame(trades_data)
                st.dataframe(trades_df, use_container_width=True, height=400)
                
                # Trade Analysis Charts
                st.markdown("#### 📈 **Trade Performance Charts**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # P&L Distribution
                    pnl_values = [t.get('total_pnl', 0) for t in trades]
                    
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Histogram(
                        x=pnl_values,
                        nbinsx=20,
                        name='P&L Distribution',
                        marker_color='rgba(255, 75, 75, 0.7)'
                    ))
                    
                    fig_pnl.update_layout(
                        title='Trade P&L Distribution',
                        xaxis_title='P&L ($)',
                        yaxis_title='Number of Trades',
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig_pnl, use_container_width=True)
                
                with col2:
                    # Cumulative P&L
                    cumulative_pnl = []
                    running_total = 0
                    
                    for trade in trades:
                        running_total += trade.get('total_pnl', 0)
                        cumulative_pnl.append(running_total)
                    
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=list(range(1, len(cumulative_pnl) + 1)),
                        y=cumulative_pnl,
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='#00ff88', width=3),
                        marker=dict(size=6)
                    ))
                    
                    fig_cum.update_layout(
                        title='Cumulative P&L by Trade',
                        xaxis_title='Trade Number',
                        yaxis_title='Cumulative P&L ($)',
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig_cum, use_container_width=True)
                
                # Interactive Trade Explorer
                st.markdown("#### 🔍 **Interactive Trade Explorer**")
                
                selected_trade_idx = st.selectbox(
                    "Select Trade to Analyze:",
                    options=range(len(trades)),
                    format_func=lambda x: f"Trade #{x+1} - {trades[x]['entry_date'].strftime('%Y-%m-%d')} - P&L: ${trades[x].get('total_pnl', 0):.0f}",
                    key="trade_selector_backtest"
                )
                
                if selected_trade_idx is not None:
                    selected_trade = trades[selected_trade_idx]
                    
                    # Trade Details
                    st.markdown(f"##### 📋 **Trade #{selected_trade_idx + 1} Details**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Entry Date:** {selected_trade['entry_date'].strftime('%Y-%m-%d')}")
                        st.write(f"**Entry Price:** ${selected_trade['entry_price']:.2f}")
                        st.write(f"**Duration:** {(selected_trade['exit_date'] - selected_trade['entry_date']).days} days")
                    
                    with col2:
                        st.write(f"**Exit Date:** {selected_trade['exit_date'].strftime('%Y-%m-%d')}")
                        st.write(f"**Exit Price:** ${selected_trade['exit_price']:.2f}")
                        price_change = ((selected_trade['exit_price'] / selected_trade['entry_price']) - 1) * 100
                        st.write(f"**Price Change:** {price_change:.1f}%")
                    
                    with col3:
                        pnl = selected_trade.get('total_pnl', 0)
                        pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                        st.write(f"**Total P&L:** {pnl_color} ${pnl:.0f}")
                        st.write(f"**Portfolio Value:** ${selected_trade['portfolio_value']:.0f}")
                        
                        if 'contracts' in selected_trade:
                            st.write(f"**Contracts:** {selected_trade['contracts']}")
                    
                    # Strategy-specific details
                    if backtest_data['strategy'] == 'COVERED_CALL':
                        st.markdown("**Options Details:**")
                        st.write(f"• Call Strike: ${selected_trade['call_strike']:.2f}")
                        st.write(f"• Premium Received: ${selected_trade['premium_received']:.0f}")
                        st.write(f"• Stock P&L: ${selected_trade['stock_pnl']:.0f}")
                    
                    elif backtest_data['strategy'] == 'CASH_SECURED_PUT':
                        st.markdown("**Options Details:**")
                        st.write(f"• Put Strike: ${selected_trade['put_strike']:.2f}")
                        st.write(f"• Premium Received: ${selected_trade['premium_received']:.0f}")
                        st.write(f"• Assigned: {'Yes' if selected_trade['assigned'] else 'No'}")
                    
                    # Add more strategy-specific details as needed
                
                # Export trades data and summary
                st.markdown("#### 💾 **Export Trade Data**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Trade History CSV",
                        data=csv,
                        file_name=f"{backtest_data['strategy']}_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="download_trades_csv"
                    )
                
                with col2:
                    # Create trade summary for display
                    summary_data = {
                        'Strategy': backtest_data['strategy'],
                        'Symbol': backtest_data['symbol'],
                        'Period': backtest_data['period'],
                        'Starting Capital': f"${backtest_data['capital']:,.0f}",
                        'Final Value': f"${metrics['final_value']:,.0f}",
                        'Total Return': f"{metrics['total_return']:.1f}%",
                        'Total Trades': len(trades),
                        'Winning Trades': len(winning_trades),
                        'Losing Trades': len(losing_trades),
                        'Win Rate': f"{metrics['win_rate']:.1f}%",
                        'Average P&L': f"${avg_trade_pnl:.0f}",
                        'Total P&L': f"${sum(t.get('total_pnl', 0) for t in trades):,.0f}",
                        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                        'Max Drawdown': f"{metrics['max_drawdown']:.1f}%"
                    }
                    
                    if st.button("📊 View Trade Summary", key="view_trade_summary_btn"):
                        st.markdown("##### 📊 **Complete Backtest Summary**")
                        
                        # Display summary in organized way
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.markdown("**📋 Basic Info:**")
                            st.write(f"• Strategy: {summary_data['Strategy']}")
                            st.write(f"• Symbol: {summary_data['Symbol']}")
                            st.write(f"• Period: {summary_data['Period']}")
                            st.write(f"• Starting Capital: {summary_data['Starting Capital']}")
                            st.write(f"• Final Value: {summary_data['Final Value']}")
                            st.write(f"• Total Return: {summary_data['Total Return']}")
                        
                        with summary_col2:
                            st.markdown("**📊 Trade Statistics:**")
                            st.write(f"• Total Trades: {summary_data['Total Trades']}")
                            st.write(f"• Winning Trades: {summary_data['Winning Trades']}")
                            st.write(f"• Losing Trades: {summary_data['Losing Trades']}")
                            st.write(f"• Win Rate: {summary_data['Win Rate']}")
                            st.write(f"• Average P&L: {summary_data['Average P&L']}")
                            st.write(f"• Sharpe Ratio: {summary_data['Sharpe Ratio']}")
                            st.write(f"• Max Drawdown: {summary_data['Max Drawdown']}")
            
            else:
                st.warning("No individual trade data available for this backtest.")
        
        else:
            # Show instructions only if no results are stored
            if not st.session_state.backtest_results:
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

    # Tab 4: Market Scanner
    with tab4:
        st.subheader("🎯 Professional Market Scanner")
        
        # Scanner controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scan_size = st.selectbox(
                "Stocks to Scan",
                [50, 100, 200, 500],
                index=1,
                format_func=lambda x: f"{x} Stocks"
            )
        
        with col2:
            scanner_asset_class = st.selectbox(
                "Asset Class",
                ['EQUITIES'],
                index=0,
                format_func=lambda x: "📈 Individual Stocks"
            )
        
        with col3:
            scan_btn = st.button(
                f"🔍 Scan Market ({scan_size} stocks)",
                type="primary",
                use_container_width=True
            )
        
        # Estimated time
        estimated_minutes = (scan_size * 0.1) / 60
        st.info(f"⏱️ **Estimated Time:** {estimated_minutes:.1f} minutes for {scan_size} stocks")
        
        # Run market scan
        if scan_btn:
            # Initialize scanner
            scanner = MarketScanner(strategist)
            
            with st.spinner(f"🔍 Scanning {scan_size} stocks... This may take a few minutes"):
                try:
                    scan_results = scanner.scan_market(max_stocks=scan_size)
                    
                    # Store results in session state
                    st.session_state.market_scan_results = scan_results
                    
                    st.success(f"✅ Scan completed! Analyzed {scan_results['total_analyzed']} stocks")
                    
                except Exception as e:
                    st.error(f"Market scan failed: {str(e)}")
        
        # Display scan results
        if 'market_scan_results' in st.session_state:
            scan_results = st.session_state.market_scan_results
            
            st.markdown("---")
            st.subheader("🏆 Market Scanner Results")
            
            # Results summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_metric_card("Total Scanned", f"{scan_results['total_analyzed']}")
            
            with col2:
                create_metric_card("Last Updated", scan_results.get('scan_timestamp', 'Unknown'))
            
            with col3:
                if st.button("🔄 Clear Results"):
                    del st.session_state.market_scan_results
                    st.rerun()
            
            # Top recommendations in tabs
            results_tabs = st.tabs(["🟢 TOP BUYS", "🔴 TOP SELLS", "📊 DETAILED VIEW"])
            
            with results_tabs[0]:  # Top Buys
                st.markdown("#### 🟢 **Top Buy Recommendations**")
                
                if scan_results.get('top_buys'):
                    for i, stock in enumerate(scan_results['top_buys'], 1):
                        with st.expander(
                            f"{i}. **{stock['ticker']}** - Score: {stock['technical_score']:.1f}/100 "
                            f"(${stock['current_price']:.2f})",
                            expanded=(i <= 3)
                        ):
                            recommendation_text = scanner.get_stock_recommendation_text(stock)
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(recommendation_text)
                            
                            with col2:
                                analysis = stock['analysis']
                                st.markdown("**📊 Key Metrics:**")
                                st.write(f"• **Price:** ${stock['current_price']:.2f}")
                                st.write(f"• **Trend:** {analysis['trend'].replace('_', ' ').title()}")
                                st.write(f"• **RSI:** {analysis['rsi']:.1f}")
                                st.write(f"• **Volatility:** {analysis['volatility_regime'].replace('_', ' ').title()}")
                                st.write(f"• **20D Change:** {analysis['price_change_20d']:.1f}%")
                else:
                    st.warning("No buy recommendations found. Try running a scan first.")
            
            with results_tabs[1]:  # Top Sells
                st.markdown("#### 🔴 **Top Sell Recommendations**")
                
                if scan_results.get('top_sells'):
                    for i, stock in enumerate(scan_results['top_sells'], 1):
                        with st.expander(
                            f"{i}. **{stock['ticker']}** - Score: {stock['technical_score']:.1f}/100 "
                            f"(${stock['current_price']:.2f})",
                            expanded=(i <= 3)
                        ):
                            recommendation_text = scanner.get_stock_recommendation_text(stock)
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(recommendation_text)
                            
                            with col2:
                                analysis = stock['analysis']
                                st.markdown("**📊 Key Metrics:**")
                                st.write(f"• **Price:** ${stock['current_price']:.2f}")
                                st.write(f"• **Trend:** {analysis['trend'].replace('_', ' ').title()}")
                                st.write(f"• **RSI:** {analysis['rsi']:.1f}")
                                st.write(f"• **Volatility:** {analysis['volatility_regime'].replace('_', ' ').title()}")
                                st.write(f"• **20D Change:** {analysis['price_change_20d']:.1f}%")
                else:
                    st.warning("No sell recommendations found. Try running a scan first.")
            
            with results_tabs[2]:  # Detailed View
                st.markdown("#### 📊 **Detailed Market Analysis**")
                
                all_stocks = scan_results.get('top_buys', []) + scan_results.get('top_sells', [])
                
                if all_stocks:
                    # Create analysis DataFrame
                    analysis_data = []
                    
                    for stock in all_stocks:
                        analysis = stock['analysis']
                        
                        analysis_data.append({
                            'Symbol': stock['ticker'],
                            'Price': f"${stock['current_price']:.2f}",
                            'Score': f"{stock['technical_score']:.1f}",
                            'Trend': analysis['trend'].replace('_', ' ').title(),
                            'RSI': f"{analysis['rsi']:.1f}",
                            'Volatility': analysis['volatility_regime'].replace('_', ' ').title(),
                            '1D Change': f"{analysis.get('price_change_1d', 0):.1f}%",
                            '5D Change': f"{analysis.get('price_change_5d', 0):.1f}%",
                            '20D Change': f"{analysis['price_change_20d']:.1f}%",
                            'Momentum': analysis['momentum'].replace('_', ' ').title()
                        })
                    
                    analysis_df = pd.DataFrame(analysis_data)
                    st.dataframe(analysis_df, use_container_width=True, height=400)
                    
                    # Download functionality
                    if st.button("📥 Download Analysis CSV"):
                        csv = analysis_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Market Analysis",
                            data=csv,
                            file_name=f"market_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("No analysis data available. Run a market scan first.")
        
        else:
            # Scanner instructions
            st.markdown("""
            ### 🎯 **Professional Market Scanner Features**
            
            **📊 Technical Analysis Scoring:**
            - **Trend Analysis** (30%): Moving averages and trend strength
            - **Momentum Indicators** (25%): RSI and overbought/oversold conditions
            - **Volatility Assessment** (15%): Current vs historical volatility
            - **Price Position** (20%): 52-week range and relative strength
            - **Recent Performance** (10%): Short-term momentum and stability
            
            **🚀 Getting Started:**
            1. **Select Scan Size**: Choose how many stocks to analyze (50-500)
            2. **Run Scan**: Click "Scan Market" and wait for analysis to complete
            3. **Review Results**: Get top buy/sell recommendations with detailed reasoning
            4. **Export Data**: Download results for further analysis
            
            **💡 How It Works:**
            - Scans S&P 500 and popular stocks using real-time market data
            - Applies professional technical analysis to each stock
            - Ranks stocks by overall technical score (0-100)
            - Provides actionable trading recommendations
            
            **⚡ Results Include:**
            - Top 10 buy recommendations with reasoning
            - Top 10 sell recommendations with risk warnings  
            - Complete technical analysis breakdown
            - Options strategy suggestions for each stock
            """)
            
            # Sample results preview
            st.markdown("#### 🔍 **Sample Analysis Preview**")
            
            sample_data = {
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                'Score': ['78.5', '82.1', '75.3', '65.7', '88.2'],
                'Trend': ['Bullish', 'Strong Bullish', 'Sideways', 'Bearish', 'Strong Bullish'],
                'RSI': ['65.2', '58.4', '72.1', '35.8', '42.3'],
                'Recommendation': ['🟢 BUY', '🟢 STRONG BUY', '⚪ HOLD', '🟠 SELL', '🟢 STRONG BUY']
            }
            
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)
            
            st.caption("This is sample data showing the format of scanner results. Run a real scan to get current market data.")

    # Footer with Enhanced Information
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **💰 Your Settings**
        - Capital: ${:,}
        - Risk: {}%
        - Max Risk: ${:,}
        """.format(available_capital, portfolio_risk_pct, int(max_risk_amount)))
    
    with col2:
        st.markdown("""
        **🎯 Current Focus**
        - Asset: {}
        - Expiry: {}  
        - Delta: {}
        """.format(asset_class, preferred_expiry, delta_preference))
    
    with col3:
        st.markdown("""
        **📊 Capabilities**
        - 12+ Strategies
        - Multi-Asset Support
        - Professional Greeks
        """)
    
    with col4:
        st.markdown("""
        **⚡ Features**
        - Real-time Analysis
        - Risk Management
        - AI Predictions
        """)

if __name__ == "__main__":
    # Initialize Streamlit app
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page and try again. If the error persists, check your API connection.")