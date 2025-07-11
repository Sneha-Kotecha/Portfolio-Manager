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

# Import Polygon SDK
try:
    from polygon import RESTClient
except ImportError:
    st.error("❌ Please install polygon-api-client: pip install polygon-api-client")
    st.stop()

warnings.filterwarnings('ignore')

# =============================================================================
# MULTI-ASSET OPTIONS DASHBOARD - Professional Trading Platform
# =============================================================================

class MultiAssetOptionsStrategist:
    """
    Professional Multi-Asset Options Strategist
    Supports Indices, Equities, and FX Options
    """
    
    def __init__(self, polygon_api_key: str):
        if not polygon_api_key:
            raise ValueError("Polygon API key is required")
        
        self.client = RESTClient(polygon_api_key)
        self.polygon_api_key = polygon_api_key
        
        # Cache for different asset classes
        self._available_indices = None
        self._available_stocks = None
        self._available_fx = None
        self._available_crypto = None
        
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
    
    def get_asset_class_instruments(self, asset_class: str) -> List[Dict]:
        """Get available instruments for specific asset class"""
        config = self.asset_configs.get(asset_class, {})
        market = config.get('market', 'stocks')
        
        try:
            print(f"🔍 Fetching {asset_class} instruments from Polygon...")
            instruments = []
            
            if asset_class == 'INDICES':
                # Get ETFs and index products
                for ticker in self.client.list_tickers(
                    market="stocks",
                    active=True,
                    limit=1000,
                    sort="ticker"
                ):
                    name = getattr(ticker, 'name', '') or ''
                    ticker_symbol = getattr(ticker, 'ticker', '') or ''
                    
                    # Filter for index-related ETFs
                    if any(term in name.lower() for term in [
                        'index', 'etf', 'ishares', 'vanguard', 'spdr', 'invesco'
                    ]) or ticker_symbol in config['popular_symbols']:
                        instruments.append({
                            'ticker': ticker_symbol,
                            'name': name,
                            'type': 'INDEX_ETF',
                            'market': 'indices'
                        })
            
            elif asset_class == 'EQUITIES':
                # Get individual stocks
                for ticker in self.client.list_tickers(
                    market="stocks",
                    active=True,
                    limit=1000,
                    sort="ticker"
                ):
                    ticker_type = getattr(ticker, 'type', '')
                    if ticker_type == 'CS':  # Common Stock
                        instruments.append({
                            'ticker': ticker.ticker,
                            'name': getattr(ticker, 'name', 'Unknown'),
                            'type': 'COMMON_STOCK',
                            'market': 'stocks'
                        })
            
            elif asset_class == 'FOREX':
                # Get FX pairs
                for ticker in self.client.list_tickers(
                    market="fx",
                    active=True,
                    limit=100,
                    sort="ticker"
                ):
                    instruments.append({
                        'ticker': ticker.ticker,
                        'name': getattr(ticker, 'name', ticker.ticker),
                        'type': 'CURRENCY_PAIR',
                        'market': 'fx'
                    })
            
            print(f"✅ Found {len(instruments)} {asset_class} instruments")
            return instruments[:500]  # Limit for performance
            
        except Exception as e:
            print(f"❌ Failed to get {asset_class} instruments: {str(e)}")
            return []
    
    def search_symbols(self, asset_class: str, query: str) -> List[Dict]:
        """Search for symbols within asset class"""
        instruments = self.get_asset_class_instruments(asset_class)
        query_lower = query.lower()
        
        matches = []
        for instrument in instruments:
            ticker = instrument.get('ticker', '').lower()
            name = instrument.get('name', '').lower()
            
            if query_lower in ticker or query_lower in name:
                matches.append(instrument)
        
        return matches[:20]  # Return top 20 matches
    
    def get_popular_symbols(self, asset_class: str) -> List[str]:
        """Get popular symbols for asset class"""
        return self.asset_configs.get(asset_class, {}).get('popular_symbols', [])
    
    def quick_data_check(self, ticker: str, asset_class: str) -> Dict:
        """Quick check of data availability for any asset class"""
        try:
            st.info(f"🔍 Quick data check for {ticker} ({asset_class})...")
            
            # Adjust ticker format based on asset class
            formatted_ticker = self._format_ticker(ticker, asset_class)
            
            # Check recent data availability
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            aggs = []
            try:
                for agg in self.client.list_aggs(
                    formatted_ticker,
                    1,
                    "day",
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    limit=30
                ):
                    aggs.append(agg)
            except Exception as e:
                return {'available': False, 'error': str(e)}
            
            if not aggs:
                return {'available': False, 'reason': 'No data returned from API'}
            
            # Check data quality
            valid_records = 0
            for agg in aggs:
                if hasattr(agg, 'close') and agg.close is not None and not pd.isna(agg.close):
                    valid_records += 1
            
            return {
                'available': valid_records > 10,
                'total_records': len(aggs),
                'valid_records': valid_records,
                'latest_price': float(aggs[-1].close) if aggs and hasattr(aggs[-1], 'close') and aggs[-1].close else None,
                'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'formatted_ticker': formatted_ticker
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _format_ticker(self, ticker: str, asset_class: str) -> str:
            """Format ticker based on asset class"""
            # Special handling for INDICES - popular ETFs trade as regular stocks
            if asset_class == 'INDICES':
                # Popular index ETFs don't need prefix (they trade as regular stocks)
                popular_etfs = ['SPY', 'QQQ', 'IWM', 'EWU', 'VGK', 'EFA', 'VIX', 'XLF', 'XLE', 'XLK']
                if ticker.upper() in popular_etfs:
                    return ticker  # No prefix for ETFs
                # For actual index tickers, use prefix
                if not ticker.startswith('I:'):
                    return f"I:{ticker}"
                return ticker
            
            # For other asset classes, use original logic
            prefix = self.asset_configs.get(asset_class, {}).get('prefix', '')
            
            if prefix and not ticker.startswith(prefix):
                return f"{prefix}{ticker}"
            return ticker
    def create_trading_chart(self, data: Dict, asset_class: str) -> go.Figure:
        """Create trading chart adapted for different asset classes"""
        try:
            df = data['historical_data'].copy()
            
            # Asset-specific chart customizations
            if asset_class == 'FOREX':
                # FX markets trade 24/5, no weekend filtering
                chart_title = f'{data["ticker"]} - FX Pair (Last Year)'
                price_annotation = f"Current: {data['current_price']:.5f}"
            else:
                # Filter for weekdays only for stocks/indices
                df = df[df.index.dayofweek < 5]
                chart_title = f'{data["ticker"]} - {asset_class} (Last Year)'
                price_annotation = f"Current: ${data['current_price']:.2f}"
            
            # Get last year of data
            one_year_ago = datetime.now() - timedelta(days=365)
            df_last_year = df[df.index >= one_year_ago]
            
            if len(df_last_year) < 50:
                df_last_year = df
            
            # Create subplot structure
            if asset_class == 'FOREX':
                # FX charts don't typically show volume
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df_last_year.index,
                        open=df_last_year['open'],
                        high=df_last_year['high'],
                        low=df_last_year['low'],
                        close=df_last_year['close'],
                        name='Price',
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff4444'
                    )
                )
                
                fig.update_layout(height=500)
                
            else:
                # Stocks/Indices with volume
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=(chart_title, 'Volume'),
                    row_heights=[0.7, 0.3]
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df_last_year.index,
                        open=df_last_year['open'],
                        high=df_last_year['high'],
                        low=df_last_year['low'],
                        close=df_last_year['close'],
                        name='Price',
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff4444'
                    ),
                    row=1, col=1
                )
                
                # Volume bars
                if 'volume' in df_last_year.columns:
                    colors = ['#ff4444' if close < open else '#00ff88' 
                             for open, close in zip(df_last_year['open'], df_last_year['close'])]
                    
                    fig.add_trace(
                        go.Bar(
                            x=df_last_year.index,
                            y=df_last_year['volume'],
                            name='Volume',
                            marker_color=colors,
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
                
                fig.update_layout(height=600)
            
            # Moving averages for all asset classes
            colors = ['#ff6b35', '#004e89', '#7209b7']
            ma_periods = [20, 50, 200]
            ma_names = ['SMA 20', 'SMA 50', 'SMA 200']
            
            for period, color, name in zip(ma_periods, colors, ma_names):
                if len(df_last_year) >= period:
                    ma = df_last_year['close'].rolling(window=period).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df_last_year.index,
                            y=ma,
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=2)
                        ),
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
            
            # Update layout
            fig.update_layout(
                title=chart_title,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Add current price annotation
            current_price = data['current_price']
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="white",
                annotation_text=price_annotation,
                annotation_position="bottom right",
                row=1, col=1 if asset_class != 'FOREX' else None
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating {asset_class} chart: {e}")
            return go.Figure().add_annotation(
                text=f"{asset_class} chart could not be generated",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def check_options_availability(self, ticker: str, asset_class: str) -> Dict:
        """Check options availability for any asset class"""
        try:
            formatted_ticker = self._format_ticker(ticker, asset_class)
            st.info(f"🎯 Checking options availability for {formatted_ticker} ({asset_class})...")
            
            contracts = list(self.client.list_options_contracts(
                underlying_ticker=ticker,  # Use original ticker for options
                limit=10
            ))
            
            if contracts:
                return {
                    'has_options': True,
                    'contract_count': len(contracts),
                    'sample_expiration': getattr(contracts[0], 'expiration_date', 'unknown'),
                    'status': 'Options Available',
                    'asset_class': asset_class
                }
            else:
                return {
                    'has_options': False,
                    'contract_count': 0,
                    'status': f'No Options Found for {asset_class}',
                    'asset_class': asset_class
                }
                
        except Exception as e:
            return {
                'has_options': False,
                'error': str(e),
                'status': f'Error Checking {asset_class} Options',
                'asset_class': asset_class
            }
    
    def get_asset_data(self, ticker: str, asset_class: str, days: int = 500) -> Dict:
        """Get data for any asset class"""
        try:
            formatted_ticker = self._format_ticker(ticker, asset_class)
            print(f"📊 Fetching {asset_class} data for {formatted_ticker}...")
            
            # Get more historical data to account for weekends/holidays
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            print(f"🔍 Requesting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            aggs = []
            try:
                for agg in self.client.list_aggs(
                    formatted_ticker,
                    1,
                    "day",
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    limit=days
                ):
                    aggs.append(agg)
            except Exception as e:
                print(f"Failed to fetch aggregates: {e}")
                raise
            
            if not aggs:
                raise ValueError(f"No historical data found for {formatted_ticker}")
            
            print(f"📈 Received {len(aggs)} raw data points")
            
            # Convert to DataFrame with asset-specific handling
            df_data = []
            skipped_records = 0
            
            for agg in aggs:
                # Handle volume data based on asset class
                volume = getattr(agg, 'volume', None)
                if asset_class == 'FOREX':
                    volume = 0  # FX doesn't have traditional volume
                elif volume is None or pd.isna(volume):
                    volume = 0
                
                # Get price data
                open_price = getattr(agg, 'open', None)
                high_price = getattr(agg, 'high', None) 
                low_price = getattr(agg, 'low', None)
                close_price = getattr(agg, 'close', None)
                
                # Skip if ALL price data is missing
                if all(x is None or pd.isna(x) for x in [open_price, high_price, low_price, close_price]):
                    skipped_records += 1
                    continue
                
                # Fill missing price values with close price if available
                if close_price is not None and not pd.isna(close_price):
                    if open_price is None or pd.isna(open_price):
                        open_price = close_price
                    if high_price is None or pd.isna(high_price):
                        high_price = close_price
                    if low_price is None or pd.isna(low_price):
                        low_price = close_price
                else:
                    skipped_records += 1
                    continue
                
                df_data.append({
                    'timestamp': agg.timestamp,
                    'open': float(open_price),
                    'high': float(high_price),
                    'low': float(low_price),
                    'close': float(close_price),
                    'volume': int(volume)
                })
            
            if skipped_records > 0:
                print(f"⚠️ Skipped {skipped_records} incomplete records")
            
            if not df_data:
                raise ValueError(f"No valid price data found for {formatted_ticker}")
            
            print(f"✅ Processing {len(df_data)} valid records")
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date').sort_index()
            
            # Remove any remaining NaN values but be less aggressive
            initial_length = len(df)
            df = df.dropna(subset=['close'])
            final_length = len(df)
            
            if initial_length != final_length:
                print(f"🧹 Cleaned {initial_length - final_length} NaN records, {final_length} remaining")
            
            if len(df) < 21:
                if days < 1000:
                    print(f"Only {len(df)} days available, trying longer time range...")
                    return self.get_asset_data(ticker, asset_class, days=1000)
                else:
                    raise ValueError(f"Insufficient clean data for {formatted_ticker}: only {len(df)} valid days after trying extended range")
            
            # Calculate technical indicators
            current_price = float(df['close'].iloc[-1])
            tech_data = self._calculate_technical_indicators(df, current_price, asset_class)
            
            print(f"✅ Successfully processed {len(df)} days of data for {formatted_ticker}")
            
            return {
                'ticker': ticker,
                'formatted_ticker': formatted_ticker,
                'asset_class': asset_class,
                'current_price': current_price,
                'historical_data': df,
                **tech_data,
                'data_points': len(df),
                'date_range': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                'source': 'polygon_real'
            }
            
        except Exception as e:
            print(f"❌ Failed to get {asset_class} data for {ticker}: {str(e)}")
            raise
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, current_price: float, asset_class: str) -> Dict:
        """Calculate technical indicators with asset-specific adjustments"""
        df = df.copy()
        
        # Fill NaN values in volume
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        if len(df) < 21:
            raise ValueError(f"Insufficient data: only {len(df)} days available, need at least 21")
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20, min_periods=10).mean()
        df['sma_50'] = df['close'].rolling(50, min_periods=25).mean()
        df['sma_200'] = df['close'].rolling(200, min_periods=100).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period, min_periods=10).mean()
        bb_std_dev = df['close'].rolling(bb_period, min_periods=10).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=7).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility (adjusted for asset class)
        df['returns'] = df['close'].pct_change()
        
        # Asset-specific volatility scaling
        if asset_class == 'FOREX':
            vol_scaling = 365  # FX trades more days
        else:
            vol_scaling = 252  # Traditional equity market days
        
        realized_vol_21d = df['returns'].rolling(21, min_periods=10).std() * np.sqrt(vol_scaling)
        realized_vol_63d = df['returns'].rolling(63, min_periods=30).std() * np.sqrt(vol_scaling)
        
        # Get latest values with NaN handling
        latest = df.iloc[-1]
        
        def safe_float(value, default=0.0):
            if pd.isna(value):
                return default
            return float(value)
        
        def safe_int(value, default=0):
            if pd.isna(value):
                return default
            return int(value)
        
        def safe_price_change(days_back, default=0.0):
            try:
                if len(df) > days_back:
                    current = df['close'].iloc[-1]
                    past = df['close'].iloc[-days_back-1]
                    if pd.notna(current) and pd.notna(past) and past != 0:
                        return float((current / past - 1) * 100)
            except (IndexError, ZeroDivisionError):
                pass
            return default
        
        # Volume calculations (adjusted for asset class)
        if asset_class == 'FOREX':
            avg_volume_20d = 0  # FX doesn't have traditional volume
            volume = 0
        else:
            avg_volume_20d = df['volume'].rolling(20, min_periods=5).mean().iloc[-1]
            volume = latest['volume']
        
        return {
            'sma_20': safe_float(latest['sma_20'], current_price),
            'sma_50': safe_float(latest['sma_50'], current_price),
            'sma_200': safe_float(latest['sma_200'], current_price),
            'bb_upper': safe_float(latest['bb_upper'], current_price * 1.02),
            'bb_lower': safe_float(latest['bb_lower'], current_price * 0.98),
            'bb_middle': safe_float(latest['bb_middle'], current_price),
            'rsi': safe_float(latest['rsi'], 50.0),
            'realized_vol_21d': safe_float(realized_vol_21d.iloc[-1], 0.20),
            'realized_vol_63d': safe_float(realized_vol_63d.iloc[-1], 0.20),
            'volume': safe_int(volume, 1000000 if asset_class != 'FOREX' else 0),
            'avg_volume_20d': safe_int(avg_volume_20d, 1000000 if asset_class != 'FOREX' else 0),
            'high_52w': safe_float(df['high'].rolling(min(252, len(df)), min_periods=50).max().iloc[-1], current_price * 1.25),
            'low_52w': safe_float(df['low'].rolling(min(252, len(df)), min_periods=50).min().iloc[-1], current_price * 0.75),
            'returns_series': df['returns'].dropna(),
            'price_change_1d': safe_price_change(1),
            'price_change_5d': safe_price_change(5),
            'price_change_20d': safe_price_change(20)
        }
    
    def get_options_data(self, underlying_ticker: str, asset_class: str, current_price: float = None) -> Dict:
        """Get options data for any asset class"""
        try:
            print(f"🎯 Fetching options data for {underlying_ticker} ({asset_class})...")
            
            # Options contracts use original ticker (not formatted)
            contracts = []
            for contract in self.client.list_options_contracts(
                underlying_ticker=underlying_ticker,
                expiration_date_gte=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                expiration_date_lte=(datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),
                limit=1000
            ):
                contracts.append(contract)
            
            if not contracts:
                raise ValueError(f"No options contracts found for {underlying_ticker}")
            
            print(f"Found {len(contracts)} options contracts")
            
            # Get current underlying price if not provided
            if current_price is None:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=5)
                    formatted_ticker = self._format_ticker(underlying_ticker, asset_class)
                    
                    recent_aggs = []
                    for agg in self.client.list_aggs(
                        formatted_ticker,
                        1,
                        "day",
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        limit=5
                    ):
                        if hasattr(agg, 'close') and agg.close is not None:
                            recent_aggs.append(agg)
                    
                    if recent_aggs:
                        current_price = float(recent_aggs[-1].close)
                        print(f"💰 Current price for options: {current_price}")
                    else:
                        raise ValueError("Could not get current price")
                        
                except Exception as e:
                    raise ValueError(f"Could not get current price for {underlying_ticker}: {e}")
            
            # Process contracts
            options_data = self._process_real_options_contracts(contracts, current_price, underlying_ticker, asset_class)
            
            return options_data
            
        except Exception as e:
            print(f"❌ Failed to get options data for {underlying_ticker}: {str(e)}")
            raise
    
    def _process_real_options_contracts(self, contracts: List, current_price: float, underlying_ticker: str, asset_class: str) -> Dict:
        """Process options contracts with asset-specific considerations"""
        exp_groups = {}
        today = datetime.now().date()
        
        for contract in contracts:
            try:
                exp_date = contract.expiration_date
                exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d').date()
                
                if exp_date_obj <= today:
                    continue
                
                strike = float(contract.strike_price)
                
                # Asset-specific strike filtering
                if asset_class == 'FOREX':
                    # FX options have tighter strike ranges
                    strike_range = 0.15  # 15%
                else:
                    strike_range = 0.25  # 25%
                
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
                    
            except Exception as e:
                continue
        
        if not exp_groups:
            raise ValueError("No valid option contracts found in reasonable strike range")
        
        # Find best expiration
        best_exp = None
        best_score = 0
        
        for exp_date in sorted(exp_groups.keys()):
            calls_count = len(exp_groups[exp_date]['calls'])
            puts_count = len(exp_groups[exp_date]['puts'])
            
            if calls_count >= 3 and puts_count >= 3:
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_datetime.date() - today).days
                
                if 14 <= days_to_exp <= 60:
                    time_score = 100 - abs(30 - days_to_exp)
                    option_score = min(calls_count + puts_count, 100)
                    total_score = time_score + option_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_exp = exp_date
        
        if not best_exp:
            for exp_date in sorted(exp_groups.keys()):
                calls_count = len(exp_groups[exp_date]['calls'])
                puts_count = len(exp_groups[exp_date]['puts'])
                if calls_count >= 3 and puts_count >= 3:
                    best_exp = exp_date
                    break
        
        if not best_exp:
            raise ValueError("No expiration found with sufficient options")
        
        # Get option prices
        self._current_underlying_price = current_price
        
        calls_data = self._get_real_option_prices(exp_groups[best_exp]['calls'], underlying_ticker, asset_class)
        puts_data = self._get_real_option_prices(exp_groups[best_exp]['puts'], underlying_ticker, asset_class)
        
        if hasattr(self, '_current_underlying_price'):
            delattr(self, '_current_underlying_price')
        
        calls_df = pd.DataFrame(calls_data).sort_values('strike') if calls_data else pd.DataFrame()
        puts_df = pd.DataFrame(puts_data).sort_values('strike') if puts_data else pd.DataFrame()
        
        if calls_df.empty or puts_df.empty:
            raise ValueError(f"Insufficient options data: {len(calls_data)} calls, {len(puts_data)} puts")
        
        if len(calls_df) < 3 or len(puts_df) < 3:
            st.warning(f"⚠️ Limited options data: {len(calls_df)} calls, {len(puts_df)} puts")
        
        exp_date_obj = datetime.strptime(best_exp, '%Y-%m-%d')
        days_to_expiry = (exp_date_obj.date() - today).days
        
        # Count data sources
        real_price_calls = len([c for c in calls_data if c.get('data_source') in ['real_trade', 'real_quote']])
        real_price_puts = len([p for p in puts_data if p.get('data_source') in ['real_trade', 'real_quote']])
        calc_price_calls = len([c for c in calls_data if c.get('data_source') == 'calculated'])
        calc_price_puts = len([p for p in puts_data if p.get('data_source') == 'calculated'])
        
        return {
            'expiration': best_exp,
            'calls': calls_df,
            'puts': puts_df,
            'days_to_expiry': days_to_expiry,
            'underlying_price': current_price,
            'underlying_ticker': underlying_ticker,
            'asset_class': asset_class,
            'total_contracts': len(calls_data) + len(puts_data),
            'pricing_breakdown': {
                'real_price_calls': real_price_calls,
                'real_price_puts': real_price_puts,
                'calculated_calls': calc_price_calls,
                'calculated_puts': calc_price_puts,
                'total_real': real_price_calls + real_price_puts,
                'total_calculated': calc_price_calls + calc_price_puts
            },
            'source': 'polygon_hybrid'
        }
    
    def _get_real_option_prices(self, contracts: List[Dict], underlying_ticker: str, asset_class: str) -> List[Dict]:
        """Get option prices with asset-specific volatility adjustments"""
        options_data = []
        real_price_count = 0
        calculated_price_count = 0
        
        # Asset-specific base volatility
        if asset_class == 'FOREX':
            base_vol = 0.15  # FX typically lower vol
        elif asset_class == 'INDICES':
            base_vol = 0.20  # Index ETFs moderate vol
        else:
            base_vol = 0.25  # Individual stocks higher vol
        
        for contract in contracts:
            try:
                ticker = contract['ticker']
                strike = contract['strike']
                contract_type = contract['contract_type']
                expiration = contract['expiration_date']
                
                # Try to get real option price data
                last_price = None
                bid = None
                ask = None
                volume = 0
                data_source = 'calculated'
                
                try:
                    trades = list(self.client.list_trades(
                        ticker,
                        timestamp_gte=(datetime.now() - timedelta(days=5)),
                        limit=5
                    ))
                    
                    if trades:
                        for trade in trades:
                            price = getattr(trade, 'price', None)
                            if price is not None and not pd.isna(price) and price > 0:
                                last_price = float(price)
                                volume = int(getattr(trade, 'size', 0))
                                data_source = 'real_trade'
                                real_price_count += 1
                                break
                        
                except Exception as e:
                    if "NOT_AUTHORIZED" in str(e):
                        pass
                    else:
                        self.logger.warning(f"Trade data error for {ticker}: {e}")
                
                try:
                    if last_price is None:
                        quotes = list(self.client.list_quotes(
                            ticker,
                            timestamp_gte=(datetime.now() - timedelta(days=5)),
                            limit=5
                        ))
                        
                        if quotes:
                            for quote in quotes:
                                bid_price = getattr(quote, 'bid', None)
                                ask_price = getattr(quote, 'ask', None)
                                
                                if (bid_price is not None and not pd.isna(bid_price) and bid_price > 0 and
                                    ask_price is not None and not pd.isna(ask_price) and ask_price > 0):
                                    bid = float(bid_price)
                                    ask = float(ask_price)
                                    last_price = (bid + ask) / 2
                                    data_source = 'real_quote'
                                    real_price_count += 1
                                    break
                        
                except Exception as e:
                    if "NOT_AUTHORIZED" in str(e):
                        pass
                    else:
                        self.logger.warning(f"Quote data error for {ticker}: {e}")
                
                # Calculate using Black-Scholes if no real price
                if last_price is None:
                    try:
                        underlying_price = getattr(self, '_current_underlying_price', 50.0)
                        
                        calculated_price = self._black_scholes_price(
                            underlying_price, strike, expiration, contract_type, base_vol
                        )
                        
                        if calculated_price > 0.01:  # Lower threshold for FX
                            last_price = calculated_price
                            bid = last_price * 0.95
                            ask = last_price * 1.05
                            data_source = 'calculated'
                            calculated_price_count += 1
                            
                            # Estimate volume based on moneyness and asset class
                            moneyness = strike / underlying_price
                            if asset_class == 'FOREX':
                                # FX options typically have lower volume
                                if 0.98 <= moneyness <= 1.02:
                                    volume = 50
                                elif 0.95 <= moneyness <= 1.05:
                                    volume = 25
                                else:
                                    volume = 10
                            else:
                                if 0.95 <= moneyness <= 1.05:
                                    volume = 100
                                elif 0.90 <= moneyness <= 1.10:
                                    volume = 50
                                else:
                                    volume = 25
                        
                    except Exception as e:
                        self.logger.warning(f"Could not calculate price for {ticker}: {e}")
                        continue
                
                # Include this option if we have a price
                min_price = 0.01 if asset_class == 'FOREX' else 0.05
                if last_price and last_price > min_price:
                    options_data.append({
                        'ticker': ticker,
                        'strike': strike,
                        'lastPrice': round(last_price, 4 if asset_class == 'FOREX' else 2),
                        'bid': round(bid, 4 if asset_class == 'FOREX' else 2) if bid else round(last_price * 0.95, 4 if asset_class == 'FOREX' else 2),
                        'ask': round(ask, 4 if asset_class == 'FOREX' else 2) if ask else round(last_price * 1.05, 4 if asset_class == 'FOREX' else 2),
                        'volume': volume,
                        'openInterest': 0,
                        'impliedVolatility': base_vol,
                        'contract_type': contract_type,
                        'data_source': data_source
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error processing option contract {contract.get('ticker', 'unknown')}: {e}")
                continue
        
        if real_price_count > 0:
            print(f"✅ Got real prices for {real_price_count} options")
        if calculated_price_count > 0:
            print(f"🧮 Calculated prices for {calculated_price_count} options")
        
        if len(options_data) < 3:
            raise ValueError(f"Insufficient options data: only {len(options_data)} valid contracts found")
        
        return options_data
    
    def _black_scholes_price(self, S: float, K: float, exp_date: str, option_type: str,
                           volatility: float, r: float = 0.05) -> float:
        """Calculate Black-Scholes option price with currency considerations"""
        try:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            T = max((exp_datetime - datetime.now()).days / 365.0, 0.01)
            
            # Adjust volatility based on moneyness
            moneyness = K / S
            if option_type.lower() == 'put' and moneyness > 1.0:
                vol_adjust = 1 + (moneyness - 1) * 0.5
            elif option_type.lower() == 'call' and moneyness < 1.0:
                vol_adjust = 1 + (1 - moneyness) * 0.2
            else:
                vol_adjust = 1.0
            
            sigma = volatility * vol_adjust
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(0.01, price)
            
        except Exception as e:
            self.logger.warning(f"Black-Scholes calculation error: {e}")
            return 0.01
    
    def analyze_market_conditions(self, data: Dict) -> Dict:
        """Analyze market conditions with asset-specific considerations"""
        current_price = data['current_price']
        asset_class = data.get('asset_class', 'EQUITIES')
        sma_20 = data['sma_20']
        sma_50 = data['sma_50']
        sma_200 = data['sma_200']
        rsi = data['rsi']
        realized_vol = data['realized_vol_21d']
        bb_upper = data['bb_upper']
        bb_lower = data['bb_lower']
        
        # Asset-specific trend analysis
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
        elif current_price > sma_50:
            trend = 'SIDEWAYS_BULLISH'
            trend_strength = abs((current_price / sma_50 - 1) * 100)
        elif current_price < sma_50:
            trend = 'SIDEWAYS_BEARISH'
            trend_strength = abs((sma_50 / current_price - 1) * 100)
        else:
            trend = 'SIDEWAYS'
            trend_strength = 2.0
        
        # Asset-specific volatility regimes
        if asset_class == 'FOREX':
            # FX has different vol thresholds
            if realized_vol > 0.20:
                vol_regime = 'EXTREME_VOL'
            elif realized_vol > 0.15:
                vol_regime = 'HIGH_VOL'
            elif realized_vol < 0.08:
                vol_regime = 'LOW_VOL'
            else:
                vol_regime = 'NORMAL_VOL'
        else:
            # Equities/Indices
            if realized_vol > 0.30:
                vol_regime = 'EXTREME_VOL'
            elif realized_vol > 0.25:
                vol_regime = 'HIGH_VOL'
            elif realized_vol < 0.12:
                vol_regime = 'LOW_VOL'
            else:
                vol_regime = 'NORMAL_VOL'
        
        # Momentum (universal across asset classes)
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
        
        # Volume analysis (asset-specific)
        if asset_class == 'FOREX':
            volume_vs_avg = 1.0  # FX doesn't have traditional volume
        else:
            volume_vs_avg = data['volume'] / data['avg_volume_20d'] if data['avg_volume_20d'] > 0 else 1.0
        
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
            'price_change_1d': round(data['price_change_1d'], 2),
            'price_change_5d': round(data['price_change_5d'], 2),
            'price_change_20d': round(data['price_change_20d'], 2),
            'high_52w': data['high_52w'],
            'low_52w': data['low_52w'],
            'current_price': current_price,
            'volume_vs_avg': round(volume_vs_avg, 2)
        }
    
    def select_strategy(self, market_analysis: Dict, underlying_data: Dict, options_data: Dict) -> Dict[str, float]:
        """Select optimal strategy with asset-specific considerations"""
        calls = options_data.get('calls', pd.DataFrame())
        puts = options_data.get('puts', pd.DataFrame())
        asset_class = underlying_data.get('asset_class', 'EQUITIES')
        
        if calls.empty or puts.empty:
            raise ValueError("Insufficient options data for strategy analysis")
        
        liquid_calls = calls[calls['lastPrice'] > (0.01 if asset_class == 'FOREX' else 0.05)]
        liquid_puts = puts[puts['lastPrice'] > (0.01 if asset_class == 'FOREX' else 0.05)]
        
        if liquid_calls.empty or liquid_puts.empty:
            raise ValueError("No liquid options found")
        
        # Get market conditions
        trend = market_analysis['trend']
        vol_regime = market_analysis['volatility_regime']
        momentum = market_analysis['momentum']
        bb_signal = market_analysis['bb_signal']
        
        scores = {}
        
        # Asset-specific strategy scoring adjustments
        vol_multiplier = 1.5 if asset_class == 'FOREX' else 1.0  # FX options strategies
        
        # Covered Call
        if len(liquid_calls) >= 1:
            base_score = 7.0
            if trend in ['SIDEWAYS', 'SIDEWAYS_BULLISH']:
                base_score += 1.5 * vol_multiplier
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                base_score += 1.0
            if momentum in ['OVERBOUGHT', 'EXTREMELY_OVERBOUGHT']:
                base_score += 0.8
            scores['COVERED_CALL'] = base_score
        
        # Cash Secured Put
        if len(liquid_puts) >= 1:
            base_score = 7.0
            if trend in ['BULLISH', 'STRONG_BULLISH', 'SIDEWAYS_BULLISH']:
                base_score += 1.5 * vol_multiplier
            if momentum in ['OVERSOLD', 'EXTREMELY_OVERSOLD']:
                base_score += 1.2
            if bb_signal in ['LOWER_BAND', 'EXTREME_LOWER']:
                base_score += 1.0
            scores['CASH_SECURED_PUT'] = base_score
        
        # Iron Condor (especially good for FX)
        if len(liquid_calls) >= 2 and len(liquid_puts) >= 2:
            base_score = 6.5
            if asset_class == 'FOREX':
                base_score += 1.0  # FX tends to range-trade
            if trend in ['SIDEWAYS', 'SIDEWAYS_BULLISH', 'SIDEWAYS_BEARISH']:
                base_score += 2.0
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                base_score += 1.5
            if bb_signal == 'MIDDLE_RANGE':
                base_score += 1.0
            scores['IRON_CONDOR'] = base_score
        
        # Bull Call Spread
        if len(liquid_calls) >= 2:
            base_score = 6.0
            if trend in ['BULLISH', 'STRONG_BULLISH']:
                base_score += 2.0
            if momentum in ['BULLISH', 'OVERSOLD']:
                base_score += 1.0
            if vol_regime in ['NORMAL_VOL', 'LOW_VOL']:
                base_score += 0.8
            scores['BULL_CALL_SPREAD'] = base_score
        
        # Bear Put Spread
        if len(liquid_puts) >= 2:
            base_score = 6.0
            if trend in ['BEARISH', 'STRONG_BEARISH']:
                base_score += 2.0
            if momentum in ['BEARISH', 'OVERBOUGHT']:
                base_score += 1.0
            if vol_regime in ['NORMAL_VOL', 'LOW_VOL']:
                base_score += 0.8
            scores['BEAR_PUT_SPREAD'] = base_score
        
        # Long Straddle (good for volatile markets)
        if len(liquid_calls) >= 1 and len(liquid_puts) >= 1:
            base_score = 5.5
            if vol_regime == 'LOW_VOL':
                base_score += 2.0
            if trend == 'SIDEWAYS':
                base_score += 1.0
            if bb_signal == 'MIDDLE_RANGE':
                base_score += 0.8
            if asset_class == 'FOREX' and vol_regime == 'LOW_VOL':
                base_score += 0.5  # FX volatility plays
            scores['LONG_STRADDLE'] = base_score
        
        # Protective Put
        if len(liquid_puts) >= 1:
            base_score = 5.0
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                base_score += 1.5
            if trend in ['BEARISH', 'STRONG_BEARISH']:
                base_score += 1.5
            if momentum in ['BEARISH', 'EXTREMELY_OVERBOUGHT']:
                base_score += 1.0
            scores['PROTECTIVE_PUT'] = base_score
        
        if not scores:
            raise ValueError("No viable strategies found for current market conditions")
        
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def analyze_symbol(self, ticker: str, asset_class: str, debug: bool = False) -> Dict:
        """Analyze symbol for any asset class"""
        try:
            print(f"🔍 Starting {asset_class} analysis for {ticker}")
            
            if debug:
                print(f"**Debug:** Fetching {asset_class} data...")
            
            # Get underlying data
            underlying_data = self.get_asset_data(ticker, asset_class)
            
            if debug:
                print(f"**Debug:** Got {underlying_data['data_points']} data points")
                print(f"**Debug:** Current price: {underlying_data['current_price']}")
            
            # Validate underlying data
            if underlying_data['current_price'] <= 0:
                raise ValueError(f"Invalid current price: {underlying_data['current_price']}")
            
            if underlying_data['data_points'] < 21:
                raise ValueError(f"Insufficient data points: {underlying_data['data_points']}")
            
            # Get options data
            if debug:
                print("**Debug:** Fetching options data...")
            
            options_data = self.get_options_data(ticker, asset_class, underlying_data['current_price'])
            
            if debug:
                print(f"**Debug:** Found {len(options_data['calls'])} calls, {len(options_data['puts'])} puts")
                print(f"**Debug:** Expiration: {options_data['expiration']}")
            
            # Validate options data
            if options_data['calls'].empty or options_data['puts'].empty:
                raise ValueError("No options data available")
            
            if len(options_data['calls']) < 3 or len(options_data['puts']) < 3:
                raise ValueError(f"Insufficient options: {len(options_data['calls'])} calls, {len(options_data['puts'])} puts")
            
            # Market analysis
            if debug:
                print("**Debug:** Analyzing market conditions...")
            
            market_analysis = self.analyze_market_conditions(underlying_data)
            
            if debug:
                print(f"**Debug:** Trend: {market_analysis['trend']}")
                print(f"**Debug:** Volatility: {market_analysis['volatility_regime']}")
                print(f"**Debug:** Momentum: {market_analysis['momentum']}")
            
            # Strategy selection
            if debug:
                print("**Debug:** Selecting strategies...")
            
            strategy_scores = self.select_strategy(market_analysis, underlying_data, options_data)
            
            if debug:
                print(f"**Debug:** Found {len(strategy_scores)} viable strategies")
            
            # Get best strategy
            best_strategy_name = max(strategy_scores.items(), key=lambda x: x[1])[0]
            best_confidence = strategy_scores[best_strategy_name]
            
            result = {
                'ticker': ticker,
                'asset_class': asset_class,
                'underlying_data': underlying_data,
                'options_data': options_data,
                'market_analysis': market_analysis,
                'strategy_scores': strategy_scores,
                'best_strategy': best_strategy_name,
                'confidence': best_confidence,
                'success': True,
                'debug_info': {
                    'data_points': underlying_data['data_points'],
                    'options_contracts': options_data['total_contracts'],
                    'calls_count': len(options_data['calls']),
                    'puts_count': len(options_data['puts'])
                }
            }
            
            if debug:
                print("**Debug:** Analysis completed successfully")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {asset_class} analysis failed for {ticker}: {error_msg}")
            
            if debug:
                import traceback
                print("**Debug - Full Error Traceback:**")
                print(traceback.format_exc())
            
            return {
                'ticker': ticker,
                'asset_class': asset_class,
                'error': error_msg,
                'success': False
            }
    
    def get_options_greeks(self, ticker: str, asset_class: str, current_price: float = None) -> Dict:
        """Get options Greeks for any asset class"""
        try:
            print(f"🔢 Fetching options Greeks for {ticker} ({asset_class})...")
            
            # Get options contracts first
            contracts = []
            for contract in self.client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date_gte=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                expiration_date_lte=(datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
                limit=500
            ):
                contracts.append(contract)
            
            if not contracts:
                raise ValueError(f"No options contracts found for {ticker}")
            
            print(f"Found {len(contracts)} options contracts for Greeks analysis")
            
            # Get current underlying price if not provided
            if current_price is None:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=5)
                    formatted_ticker = self._format_ticker(ticker, asset_class)
                    
                    recent_aggs = []
                    for agg in self.client.list_aggs(
                        formatted_ticker,
                        1,
                        "day",
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        limit=5
                    ):
                        if hasattr(agg, 'close') and agg.close is not None:
                            recent_aggs.append(agg)
                    
                    if recent_aggs:
                        current_price = float(recent_aggs[-1].close)
                    else:
                        raise ValueError("Could not get current price")
                        
                except Exception as e:
                    raise ValueError(f"Could not get current price for {ticker}: {e}")
            
            # Process contracts and calculate/fetch Greeks
            greeks_data = self._process_options_greeks(contracts, current_price, ticker, asset_class)
            
            return greeks_data
            
        except Exception as e:
            print(f"❌ Failed to get Greeks for {ticker}: {str(e)}")
            raise
    
    def _process_options_greeks(self, contracts: List, current_price: float, underlying_ticker: str, asset_class: str) -> Dict:
        """Process options contracts and calculate Greeks with asset-specific considerations"""
        
        # Group by expiration and filter
        exp_groups = {}
        today = datetime.now().date()
        
        for contract in contracts:
            try:
                exp_date = contract.expiration_date
                exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d').date()
                
                if exp_date_obj <= today:
                    continue
                
                strike = float(contract.strike_price)
                
                # Asset-specific strike filtering
                if asset_class == 'FOREX':
                    strike_range = 0.20  # 20% for FX
                elif asset_class == 'INDICES':
                    strike_range = 0.25  # 25% for indices
                else:
                    strike_range = 0.30  # 30% for individual stocks
                
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
                    
            except Exception as e:
                continue
        
        if not exp_groups:
            raise ValueError("No valid option contracts found for Greeks analysis")
        
        # Find best expiration for Greeks analysis (prefer 30-45 days)
        best_exp = None
        best_score = 0
        
        for exp_date in sorted(exp_groups.keys()):
            calls_count = len(exp_groups[exp_date]['calls'])
            puts_count = len(exp_groups[exp_date]['puts'])
            
            if calls_count >= 3 and puts_count >= 3:
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_datetime.date() - today).days
                
                if 14 <= days_to_exp <= 60:
                    time_score = 100 - abs(30 - days_to_exp)
                    option_score = min(calls_count + puts_count, 100)
                    total_score = time_score + option_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_exp = exp_date
        
        if not best_exp:
            # Fallback to any viable expiration
            for exp_date in sorted(exp_groups.keys()):
                calls_count = len(exp_groups[exp_date]['calls'])
                puts_count = len(exp_groups[exp_date]['puts'])
                if calls_count >= 3 and puts_count >= 3:
                    best_exp = exp_date
                    break
        
        if not best_exp:
            raise ValueError("No expiration found with sufficient options for Greeks analysis")
        
        # Calculate Greeks for the best expiration
        calls_greeks = self._calculate_option_greeks(exp_groups[best_exp]['calls'], current_price, 'call', asset_class)
        puts_greeks = self._calculate_option_greeks(exp_groups[best_exp]['puts'], current_price, 'put', asset_class)
        
        # Combine results
        all_greeks = calls_greeks + puts_greeks
        
        # Create summary statistics
        summary_stats = self._calculate_greeks_summary(all_greeks, current_price)
        
        exp_date_obj = datetime.strptime(best_exp, '%Y-%m-%d')
        days_to_expiry = (exp_date_obj.date() - today).days
        
        return {
            'expiration': best_exp,
            'days_to_expiry': days_to_expiry,
            'underlying_price': current_price,
            'underlying_ticker': underlying_ticker,
            'asset_class': asset_class,
            'calls_greeks': pd.DataFrame(calls_greeks),
            'puts_greeks': pd.DataFrame(puts_greeks),
            'all_greeks': pd.DataFrame(all_greeks),
            'summary_stats': summary_stats,
            'total_contracts': len(all_greeks)
        }
    
    def _calculate_option_greeks(self, contracts: List[Dict], underlying_price: float, option_type: str, asset_class: str) -> List[Dict]:
        """Calculate Greeks for option contracts with asset-specific considerations"""
        greeks_data = []
        
        # Risk-free rate (approximate)
        r = 0.05
        
        # Asset-specific base volatility
        if asset_class == 'FOREX':
            base_vol = 0.15
        elif asset_class == 'INDICES':
            base_vol = 0.20
        else:  # EQUITIES
            base_vol = 0.25
        
        for contract in contracts:
            try:
                strike = contract['strike']
                expiration = contract['expiration_date']
                
                # Calculate time to expiration
                exp_datetime = datetime.strptime(expiration, '%Y-%m-%d')
                T = max((exp_datetime - datetime.now()).days / 365.0, 0.01)
                
                # Adjust volatility based on moneyness and asset class
                moneyness = strike / underlying_price
                if option_type == 'put' and moneyness > 1.0:
                    # Put skew - higher vol for OTM puts
                    if asset_class == 'FOREX':
                        vol_adj = base_vol * (1 + (moneyness - 1) * 0.2)  # Less skew for FX
                    else:
                        vol_adj = base_vol * (1 + (moneyness - 1) * 0.3)
                elif option_type == 'call' and moneyness < 1.0:
                    # Call skew - slightly higher vol for OTM calls  
                    vol_adj = base_vol * (1 + (1 - moneyness) * 0.2)
                else:
                    vol_adj = base_vol
                
                # Calculate Greeks using Black-Scholes
                greeks = self._black_scholes_greeks(underlying_price, strike, T, r, vol_adj, option_type)
                
                # Add contract information
                greeks.update({
                    'ticker': contract['ticker'],
                    'strike': strike,
                    'expiration': expiration,
                    'contract_type': option_type,
                    'moneyness': round(moneyness, 3),
                    'time_to_expiry': round(T, 3),
                    'implied_vol': round(vol_adj, 3),
                    'asset_class': asset_class
                })
                
                greeks_data.append(greeks)
                
            except Exception as e:
                self.logger.warning(f"Error calculating Greeks for {contract.get('ticker', 'unknown')}: {e}")
                continue
        
        return greeks_data
    
    def _black_scholes_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict:
        """Calculate Black-Scholes Greeks"""
        try:
            import math
            from scipy.stats import norm
            
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Calculate Greeks
            if option_type.lower() == 'call':
                # Call option Greeks
                delta = norm.cdf(d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                        - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
                rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
            else:
                # Put option Greeks
                delta = norm.cdf(d1) - 1
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                        + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
                rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
            
            # Common Greeks for both calls and puts
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            
            # Calculate option price
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return {
                'price': round(max(0.01, price), 4),
                'delta': round(delta, 4),
                'gamma': round(gamma, 6),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'rho': round(rho, 4)
            }
            
        except Exception as e:
            self.logger.warning(f"Greeks calculation error: {e}")
            return {
                'price': 0.01,
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
    
    def _calculate_greeks_summary(self, all_greeks: List[Dict], underlying_price: float) -> Dict:
        """Calculate summary statistics for Greeks"""
        if not all_greeks:
            return {}
        
        df = pd.DataFrame(all_greeks)
        
        # ATM options (within 2% of current price)
        atm_mask = abs(df['strike'] - underlying_price) / underlying_price <= 0.02
        atm_options = df[atm_mask]
        
        # OTM calls and puts
        otm_calls = df[(df['contract_type'] == 'call') & (df['strike'] > underlying_price)]
        otm_puts = df[(df['contract_type'] == 'put') & (df['strike'] < underlying_price)]
        
        summary = {
            'total_options': len(df),
            'atm_options': len(atm_options),
            'otm_calls': len(otm_calls),
            'otm_puts': len(otm_puts),
            'avg_delta_calls': df[df['contract_type'] == 'call']['delta'].mean() if len(df[df['contract_type'] == 'call']) > 0 else 0,
            'avg_delta_puts': df[df['contract_type'] == 'put']['delta'].mean() if len(df[df['contract_type'] == 'put']) > 0 else 0,
            'max_gamma': df['gamma'].max(),
            'avg_theta': df['theta'].mean(),
            'avg_vega': df['vega'].mean(),
            'highest_gamma_strike': df.loc[df['gamma'].idxmax(), 'strike'] if len(df) > 0 else 0
        }
        
        # Round values
        for key, value in summary.items():
            if isinstance(value, float):
                summary[key] = round(value, 4)
        
        return summary
    
    def backtest_strategy(self, ticker: str, asset_class: str, strategy_name: str, 
                         start_date: str, end_date: str, parameters: Dict = None) -> Dict:
        """Backtest an options strategy over a specified period for any asset class"""
        try:
            print(f"🔄 Starting backtest for {strategy_name} on {ticker} ({asset_class})")
            
            # Get historical data for backtesting period
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Fetch extended historical data
            total_days = (end_dt - start_dt).days + 100  # Extra days for indicators
            historical_data = self.get_asset_data(ticker, asset_class, days=total_days)
            
            # Filter data to backtest period
            hist_df = historical_data['historical_data']
            mask = (hist_df.index >= start_dt) & (hist_df.index <= end_dt)
            backtest_df = hist_df[mask].copy()
            
            if len(backtest_df) < 30:
                raise ValueError(f"Insufficient data for backtesting period: {len(backtest_df)} days")
            
            # Run strategy backtest
            if strategy_name == 'COVERED_CALL':
                results = self._backtest_covered_call(backtest_df, parameters or {}, asset_class)
            elif strategy_name == 'CASH_SECURED_PUT':
                results = self._backtest_cash_secured_put(backtest_df, parameters or {}, asset_class)
            elif strategy_name == 'IRON_CONDOR':
                results = self._backtest_iron_condor(backtest_df, parameters or {}, asset_class)
            elif strategy_name == 'BULL_CALL_SPREAD':
                results = self._backtest_bull_call_spread(backtest_df, parameters or {}, asset_class)
            elif strategy_name == 'BEAR_PUT_SPREAD':
                results = self._backtest_bear_put_spread(backtest_df, parameters or {}, asset_class)
            else:
                # Default buy and hold strategy
                results = self._backtest_buy_and_hold(backtest_df, asset_class)
            
            # Calculate performance metrics
            performance = self._calculate_backtest_performance(results, backtest_df)
            
            return {
                'ticker': ticker,
                'asset_class': asset_class,
                'strategy': strategy_name,
                'start_date': start_date,
                'end_date': end_date,
                'parameters': parameters or {},
                'results': results,
                'performance': performance,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Backtest failed: {str(e)}")
            return {
                'ticker': ticker,
                'asset_class': asset_class,
                'strategy': strategy_name,
                'error': str(e),
                'success': False
            }
    
    def _backtest_covered_call(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest covered call strategy with asset-specific considerations"""
        # Parameters
        dte_target = params.get('days_to_expiry', 30)
        delta_target = params.get('delta_target', 0.3)
        
        trades = []
        equity_curve = []
        current_position = None
        total_pnl = 0
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            
            # If no position, enter new covered call
            if current_position is None:
                # Buy 100 shares (or equivalent for FX)
                share_equivalent = 100 if asset_class != 'FOREX' else 10000  # FX in units
                stock_cost = current_price * share_equivalent
                
                # Sell call option (estimate premium)
                call_strike = current_price * (1 + delta_target)
                call_premium = self._estimate_option_premium(current_price, call_strike, dte_target, 'call', asset_class) * share_equivalent
                
                current_position = {
                    'entry_date': current_date,
                    'stock_price': current_price,
                    'stock_cost': stock_cost,
                    'call_strike': call_strike,
                    'call_premium': call_premium,
                    'expiry_date': current_date + timedelta(days=dte_target),
                    'share_equivalent': share_equivalent
                }
                
                total_pnl -= stock_cost  # Buy stock
                total_pnl += call_premium  # Sell call
            
            # Check if position should be closed
            if current_position and current_date >= current_position['expiry_date']:
                # Close position
                share_equiv = current_position['share_equivalent']
                
                if current_price > current_position['call_strike']:
                    # Called away
                    stock_sale = current_position['call_strike'] * share_equiv
                else:
                    # Keep stock
                    stock_sale = current_price * share_equiv
                
                total_pnl += stock_sale
                
                trade_pnl = (stock_sale + current_position['call_premium'] - current_position['stock_cost'])
                
                trades.append({
                    'entry_date': current_position['entry_date'],
                    'exit_date': current_date,
                    'strategy': 'COVERED_CALL',
                    'pnl': trade_pnl,
                    'return_pct': (trade_pnl / current_position['stock_cost']) * 100
                })
                
                current_position = None
            
            # Calculate current portfolio value
            if current_position:
                portfolio_value = total_pnl + (current_price * current_position['share_equivalent'])
            else:
                portfolio_value = total_pnl
            
            equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'underlying_price': current_price
            })
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_pnl': total_pnl
        }
    
    def _backtest_cash_secured_put(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest cash secured put strategy"""
        dte_target = params.get('days_to_expiry', 30)
        delta_target = params.get('delta_target', -0.3)
        
        trades = []
        equity_curve = []
        current_position = None
        
        # Asset-specific starting capital
        if asset_class == 'FOREX':
            cash_balance = 100000  # Larger for FX
        else:
            cash_balance = 10000
        
        total_pnl = 0
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            
            share_equivalent = 100 if asset_class != 'FOREX' else 10000
            
            if current_position is None and cash_balance >= current_price * share_equivalent:
                # Sell put option
                put_strike = current_price * (1 + delta_target)  # OTM put
                put_premium = self._estimate_option_premium(current_price, put_strike, dte_target, 'put', asset_class) * share_equivalent
                
                current_position = {
                    'entry_date': current_date,
                    'put_strike': put_strike,
                    'put_premium': put_premium,
                    'expiry_date': current_date + timedelta(days=dte_target),
                    'share_equivalent': share_equivalent
                }
                
                cash_balance += put_premium
                total_pnl += put_premium
            
            if current_position and current_date >= current_position['expiry_date']:
                share_equiv = current_position['share_equivalent']
                
                if current_price < current_position['put_strike']:
                    # Assigned - buy stock
                    stock_cost = current_position['put_strike'] * share_equiv
                    cash_balance -= stock_cost
                    total_pnl -= stock_cost
                    
                    trade_pnl = current_position['put_premium'] - (current_position['put_strike'] - current_price) * share_equiv
                else:
                    # Keep premium
                    trade_pnl = current_position['put_premium']
                
                trades.append({
                    'entry_date': current_position['entry_date'],
                    'exit_date': current_date,
                    'strategy': 'CASH_SECURED_PUT',
                    'pnl': trade_pnl,
                    'return_pct': (trade_pnl / (current_position['put_strike'] * share_equiv)) * 100
                })
                
                current_position = None
            
            portfolio_value = cash_balance + total_pnl
            equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'underlying_price': current_price
            })
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_pnl': total_pnl
        }
    
    def _backtest_iron_condor(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest iron condor strategy"""
        dte_target = params.get('days_to_expiry', 30)
        wing_width = params.get('wing_width', 0.05)
        
        # Asset-specific adjustments
        if asset_class == 'FOREX':
            wing_width *= 0.7  # Tighter for FX
            
        trades = []
        equity_curve = []
        total_pnl = 0
        
        i = 0
        while i < len(df) - dte_target:
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            
            # Set up iron condor strikes
            call_sell_strike = current_price * (1 + wing_width)
            call_buy_strike = current_price * (1 + wing_width * 2)
            put_sell_strike = current_price * (1 - wing_width)
            put_buy_strike = current_price * (1 - wing_width * 2)
            
            # Calculate premiums
            call_sell_premium = self._estimate_option_premium(current_price, call_sell_strike, dte_target, 'call', asset_class)
            call_buy_premium = self._estimate_option_premium(current_price, call_buy_strike, dte_target, 'call', asset_class)
            put_sell_premium = self._estimate_option_premium(current_price, put_sell_strike, dte_target, 'put', asset_class)
            put_buy_premium = self._estimate_option_premium(current_price, put_buy_strike, dte_target, 'put', asset_class)
            
            share_equivalent = 100 if asset_class != 'FOREX' else 10000
            net_credit = (call_sell_premium + put_sell_premium - call_buy_premium - put_buy_premium) * share_equivalent
            
            # Jump to expiry
            expiry_idx = min(i + dte_target, len(df) - 1)
            expiry_date = df.index[expiry_idx]
            expiry_price = df.iloc[expiry_idx]['close']
            
            # Calculate P&L at expiry
            if put_buy_strike <= expiry_price <= call_buy_strike:
                # Max profit - all options expire worthless
                trade_pnl = net_credit
            elif expiry_price < put_sell_strike:
                # Loss on put side
                put_loss = (put_sell_strike - expiry_price) * share_equivalent
                max_loss = (put_sell_strike - put_buy_strike) * share_equivalent
                trade_pnl = net_credit - min(put_loss, max_loss)
            elif expiry_price > call_sell_strike:
                # Loss on call side
                call_loss = (expiry_price - call_sell_strike) * share_equivalent
                max_loss = (call_buy_strike - call_sell_strike) * share_equivalent
                trade_pnl = net_credit - min(call_loss, max_loss)
            else:
                # In profit zone
                trade_pnl = net_credit
            
            trades.append({
                'entry_date': current_date,
                'exit_date': expiry_date,
                'strategy': 'IRON_CONDOR',
                'pnl': trade_pnl,
                'return_pct': (trade_pnl / (abs(net_credit) + 1000)) * 100
            })
            
            total_pnl += trade_pnl
            
            # Add equity curve points
            for j in range(i, expiry_idx + 1):
                equity_curve.append({
                    'date': df.index[j],
                    'portfolio_value': total_pnl,
                    'underlying_price': df.iloc[j]['close']
                })
            
            i = expiry_idx + 1
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_pnl': total_pnl
        }
    
    def _backtest_bull_call_spread(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest bull call spread strategy"""
        dte_target = params.get('days_to_expiry', 30)
        
        trades = []
        equity_curve = []
        total_pnl = 0
        
        i = 0
        while i < len(df) - dte_target:
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            
            # Set up bull call spread
            buy_strike = current_price  # ATM
            sell_strike = current_price * 1.05  # 5% OTM
            
            buy_premium = self._estimate_option_premium(current_price, buy_strike, dte_target, 'call', asset_class)
            sell_premium = self._estimate_option_premium(current_price, sell_strike, dte_target, 'call', asset_class)
            
            share_equivalent = 100 if asset_class != 'FOREX' else 10000
            net_debit = (buy_premium - sell_premium) * share_equivalent
            
            # Jump to expiry
            expiry_idx = min(i + dte_target, len(df) - 1)
            expiry_date = df.index[expiry_idx]
            expiry_price = df.iloc[expiry_idx]['close']
            
            # Calculate P&L at expiry
            if expiry_price <= buy_strike:
                trade_pnl = -net_debit  # Max loss
            elif expiry_price >= sell_strike:
                trade_pnl = (sell_strike - buy_strike) * share_equivalent - net_debit  # Max profit
            else:
                trade_pnl = (expiry_price - buy_strike) * share_equivalent - net_debit
            
            trades.append({
                'entry_date': current_date,
                'exit_date': expiry_date,
                'strategy': 'BULL_CALL_SPREAD',
                'pnl': trade_pnl,
                'return_pct': (trade_pnl / abs(net_debit)) * 100 if net_debit != 0 else 0
            })
            
            total_pnl += trade_pnl
            
            # Add equity curve points
            for j in range(i, expiry_idx + 1):
                equity_curve.append({
                    'date': df.index[j],
                    'portfolio_value': total_pnl,
                    'underlying_price': df.iloc[j]['close']
                })
            
            i = expiry_idx + 5  # Wait 5 days before next trade
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_pnl': total_pnl
        }
    
    def _backtest_bear_put_spread(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest bear put spread strategy"""
        dte_target = params.get('days_to_expiry', 30)
        
        trades = []
        equity_curve = []
        total_pnl = 0
        
        i = 0
        while i < len(df) - dte_target:
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            
            # Set up bear put spread
            buy_strike = current_price  # ATM
            sell_strike = current_price * 0.95  # 5% OTM
            
            buy_premium = self._estimate_option_premium(current_price, buy_strike, dte_target, 'put', asset_class)
            sell_premium = self._estimate_option_premium(current_price, sell_strike, dte_target, 'put', asset_class)
            
            share_equivalent = 100 if asset_class != 'FOREX' else 10000
            net_debit = (buy_premium - sell_premium) * share_equivalent
            
            # Jump to expiry
            expiry_idx = min(i + dte_target, len(df) - 1)
            expiry_date = df.index[expiry_idx]
            expiry_price = df.iloc[expiry_idx]['close']
            
            # Calculate P&L at expiry
            if expiry_price >= buy_strike:
                trade_pnl = -net_debit  # Max loss
            elif expiry_price <= sell_strike:
                trade_pnl = (buy_strike - sell_strike) * share_equivalent - net_debit  # Max profit
            else:
                trade_pnl = (buy_strike - expiry_price) * share_equivalent - net_debit
            
            trades.append({
                'entry_date': current_date,
                'exit_date': expiry_date,
                'strategy': 'BEAR_PUT_SPREAD',
                'pnl': trade_pnl,
                'return_pct': (trade_pnl / abs(net_debit)) * 100 if net_debit != 0 else 0
            })
            
            total_pnl += trade_pnl
            
            # Add equity curve points
            for j in range(i, expiry_idx + 1):
                equity_curve.append({
                    'date': df.index[j],
                    'portfolio_value': total_pnl,
                    'underlying_price': df.iloc[j]['close']
                })
            
            i = expiry_idx + 5  # Wait 5 days before next trade
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_pnl': total_pnl
        }
    
    def _backtest_buy_and_hold(self, df: pd.DataFrame, asset_class: str) -> Dict:
        """Backtest simple buy and hold strategy"""
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        start_value = start_price * share_equivalent
        end_value = end_price * share_equivalent
        total_pnl = end_value - start_value
        
        equity_curve = []
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            portfolio_value = (current_price * share_equivalent) - start_value
            equity_curve.append({
                'date': df.index[i],
                'portfolio_value': portfolio_value,
                'underlying_price': current_price
            })
        
        trades = [{
            'entry_date': df.index[0],
            'exit_date': df.index[-1],
            'strategy': 'BUY_AND_HOLD',
            'pnl': total_pnl,
            'return_pct': ((end_price / start_price) - 1) * 100
        }]
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_pnl': total_pnl
        }
    
    def _estimate_option_premium(self, spot: float, strike: float, dte: int, option_type: str, asset_class: str) -> float:
        """Estimate option premium with asset-specific adjustments"""
        T = dte / 365.0
        r = 0.05
        
        # Asset-specific volatility
        if asset_class == 'FOREX':
            vol = 0.15
        elif asset_class == 'INDICES':
            vol = 0.20
        else:
            vol = 0.25
        
        # Adjust vol for moneyness
        moneyness = strike / spot
        if option_type == 'put' and moneyness > 1.0:
            vol *= (1 + (moneyness - 1) * 0.3)
        elif option_type == 'call' and moneyness < 1.0:
            vol *= (1 + (1 - moneyness) * 0.2)
        
        return self._black_scholes_price(spot, strike, 
                                       (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d'), 
                                       option_type, vol, r)
    
    def _calculate_backtest_performance(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive backtest performance metrics"""
        trades = results['trades']
        equity_curve = results['equity_curve']
        
        if not trades:
            return {'error': 'No trades to analyze'}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        # Returns analysis
        returns = [t['return_pct'] for t in trades]
        avg_return = np.mean(returns) if returns else 0
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio
        excess_returns = [r - 5 for r in returns]
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if len(excess_returns) > 1 and np.std(excess_returns) != 0 else 0
        
        # Drawdown analysis
        equity_values = [point['portfolio_value'] for point in equity_curve]
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = [(eq - max_val) / max_val * 100 if max_val != 0 else 0 
                    for eq, max_val in zip(equity_values, running_max)]
        max_drawdown = min(drawdowns) if drawdowns else 0
        
        # Time in market
        start_date = trades[0]['entry_date'] if trades else df.index[0]
        end_date = trades[-1]['exit_date'] if trades else df.index[-1]
        total_days = (end_date - start_date).days
        
        # Benchmark comparison
        benchmark_return = ((df.iloc[-1]['close'] / df.iloc[0]['close']) - 1) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_return_per_trade': round(avg_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'total_days': total_days,
            'benchmark_return': round(benchmark_return, 2),
            'alpha': round(avg_return - (benchmark_return / total_trades if total_trades > 0 else 0), 2)
        }
    
    def predict_market_direction(self, ticker: str, asset_class: str, prediction_days: int = 30) -> Dict:
        """Predict market direction using technical analysis and volatility for any asset class"""
        try:
            print(f"🔮 Generating market predictions for {ticker} ({asset_class})")
            
            # Get extended historical data for better predictions
            data = self.get_asset_data(ticker, asset_class, days=500)
            df = data['historical_data']
            
            current_price = data['current_price']
            
            # Technical analysis predictions
            technical_signals = self._analyze_technical_signals(df, current_price, asset_class)
            
            # Volatility forecasting
            volatility_forecast = self._forecast_volatility(df, prediction_days, asset_class)
            
            # Support and resistance levels
            support_resistance = self._calculate_support_resistance(df, current_price, asset_class)
            
            # Price targets
            price_targets = self._calculate_price_targets(df, current_price, technical_signals, asset_class)
            
            # Momentum analysis
            momentum_analysis = self._analyze_momentum(df, current_price, asset_class)
            
            # Overall prediction
            overall_prediction = self._generate_overall_prediction(
                technical_signals, volatility_forecast, support_resistance, 
                price_targets, momentum_analysis, asset_class
            )
            
            return {
                'ticker': ticker,
                'asset_class': asset_class,
                'current_price': current_price,
                'prediction_period': prediction_days,
                'technical_signals': technical_signals,
                'volatility_forecast': volatility_forecast,
                'support_resistance': support_resistance,
                'price_targets': price_targets,
                'momentum_analysis': momentum_analysis,
                'overall_prediction': overall_prediction,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return {
                'ticker': ticker,
                'asset_class': asset_class,
                'error': str(e),
                'success': False
            }
    
    def _analyze_technical_signals(self, df: pd.DataFrame, current_price: float, asset_class: str) -> Dict:
        """Analyze technical indicators for prediction signals with asset-specific considerations"""
        latest = df.iloc[-1]
        
        # Calculate additional indicators if not present
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
        
        # RSI signal
        rsi_values = []
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
        
        # Asset-specific RSI thresholds
        if asset_class == 'FOREX':
            overbought_threshold = 80  # FX less extreme
            oversold_threshold = 20
        else:
            overbought_threshold = 70
            oversold_threshold = 30
        
        if rsi > overbought_threshold:
            rsi_signal = 'BEARISH'
            rsi_strength = min((rsi - overbought_threshold) / 10, 1.0)
        elif rsi < oversold_threshold:
            rsi_signal = 'BULLISH'
            rsi_strength = min((oversold_threshold - rsi) / 10, 1.0)
        else:
            rsi_signal = 'NEUTRAL'
            rsi_strength = 0.5
        
        # Moving average signals
        sma_20 = latest.get('sma_20', current_price)
        sma_50 = latest.get('sma_50', current_price)
        sma_200 = latest.get('sma_200', current_price)
        
        ma_signals = []
        if current_price > sma_20:
            ma_signals.append('SHORT_TERM_BULLISH')
        if current_price > sma_50:
            ma_signals.append('MEDIUM_TERM_BULLISH')
        if current_price > sma_200:
            ma_signals.append('LONG_TERM_BULLISH')
        
        # MACD signal
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal
        
        current_macd = macd_histogram.iloc[-1]
        prev_macd = macd_histogram.iloc[-2] if len(macd_histogram) > 1 else 0
        
        if current_macd > 0 and current_macd > prev_macd:
            macd_signal_direction = 'BULLISH'
        elif current_macd < 0 and current_macd < prev_macd:
            macd_signal_direction = 'BEARISH'
        else:
            macd_signal_direction = 'NEUTRAL'
        
        return {
            'rsi_signal': rsi_signal,
            'rsi_value': round(rsi, 2),
            'rsi_strength': round(rsi_strength, 2),
            'moving_average_signals': ma_signals,
            'macd_signal': macd_signal_direction,
            'macd_value': round(current_macd, 4),
            'price_vs_sma20': round(((current_price / sma_20) - 1) * 100, 2),
            'price_vs_sma50': round(((current_price / sma_50) - 1) * 100, 2),
            'price_vs_sma200': round(((current_price / sma_200) - 1) * 100, 2)
        }
    
    def _forecast_volatility(self, df: pd.DataFrame, days: int, asset_class: str) -> Dict:
        """Forecast volatility with asset-specific considerations"""
        returns = df['close'].pct_change().dropna()
        
        # Asset-specific volatility scaling
        if asset_class == 'FOREX':
            vol_scaling = 365  # FX trades more days
        else:
            vol_scaling = 252
        
        # Current realized volatility
        vol_10d = returns.tail(10).std() * np.sqrt(vol_scaling)
        vol_21d = returns.tail(21).std() * np.sqrt(vol_scaling)
        vol_63d = returns.tail(63).std() * np.sqrt(vol_scaling)
        
        # Simple volatility forecast (mean reversion model)
        long_term_vol = returns.std() * np.sqrt(vol_scaling)
        current_vol = vol_21d
        
        # Asset-specific mean reversion parameter
        if asset_class == 'FOREX':
            alpha = 0.15  # Faster mean reversion for FX
        else:
            alpha = 0.1
        
        forecast_vol = current_vol * (1 - alpha) + long_term_vol * alpha
        
        # Asset-specific volatility regime thresholds
        if asset_class == 'FOREX':
            high_vol_threshold = long_term_vol * 1.3
            low_vol_threshold = long_term_vol * 0.8
        else:
            high_vol_threshold = long_term_vol * 1.5
            low_vol_threshold = long_term_vol * 0.7
        
        # Volatility regime
        if current_vol > high_vol_threshold:
            regime = 'HIGH_VOLATILITY'
            regime_confidence = min((current_vol / long_term_vol - 1), 1.0)
        elif current_vol < low_vol_threshold:
            regime = 'LOW_VOLATILITY'
            regime_confidence = min((1 - current_vol / long_term_vol), 1.0)
        else:
            regime = 'NORMAL_VOLATILITY'
            regime_confidence = 0.5
        
        return {
            'current_vol_10d': round(vol_10d, 3),
            'current_vol_21d': round(vol_21d, 3),
            'current_vol_63d': round(vol_63d, 3),
            'long_term_vol': round(long_term_vol, 3),
            'forecast_vol': round(forecast_vol, 3),
            'volatility_regime': regime,
            'regime_confidence': round(regime_confidence, 2),
            'vol_trend': 'INCREASING' if vol_10d > vol_21d else 'DECREASING'
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame, current_price: float, asset_class: str) -> Dict:
        """Calculate support and resistance levels with asset-specific considerations"""
        # Use recent highs and lows
        recent_data = df.tail(252)  # Last year
        
        # Resistance levels (recent highs)
        resistance_levels = []
        for i in range(5, len(recent_data) - 5):
            if recent_data.iloc[i]['high'] == recent_data.iloc[i-5:i+6]['high'].max():
                resistance_levels.append(recent_data.iloc[i]['high'])
        
        # Support levels (recent lows)
        support_levels = []
        for i in range(5, len(recent_data) - 5):
            if recent_data.iloc[i]['low'] == recent_data.iloc[i-5:i+6]['low'].min():
                support_levels.append(recent_data.iloc[i]['low'])
        
        # Filter and sort levels
        resistance_levels = sorted([r for r in resistance_levels if r > current_price])[:3]
        support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)[:3]
        
        # Fibonacci retracements
        high_52w = recent_data['high'].max()
        low_52w = recent_data['low'].min()
        
        fib_levels = {
            '23.6%': low_52w + (high_52w - low_52w) * 0.236,
            '38.2%': low_52w + (high_52w - low_52w) * 0.382,
            '50.0%': low_52w + (high_52w - low_52w) * 0.500,
            '61.8%': low_52w + (high_52w - low_52w) * 0.618,
            '78.6%': low_52w + (high_52w - low_52w) * 0.786
        }
        
        # Asset-specific precision
        precision = 5 if asset_class == 'FOREX' else 2
        
        return {
            'resistance_levels': [round(r, precision) for r in resistance_levels],
            'support_levels': [round(s, precision) for s in support_levels],
            'fibonacci_levels': {k: round(v, precision) for k, v in fib_levels.items()},
            '52_week_high': round(high_52w, precision),
            '52_week_low': round(low_52w, precision),
            'distance_to_52w_high': round(((high_52w / current_price) - 1) * 100, 2),
            'distance_to_52w_low': round(((current_price / low_52w) - 1) * 100, 2)
        }
    
    def _calculate_price_targets(self, df: pd.DataFrame, current_price: float, 
                                technical_signals: Dict, asset_class: str) -> Dict:
        """Calculate price targets with asset-specific considerations"""
        
        # Average True Range for volatility-based targets
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        current_atr = atr.iloc[-1]
        
        # Asset-specific ATR multipliers
        if asset_class == 'FOREX':
            atr_multipliers = [1.0, 2.0, 3.0]  # More conservative for FX
        else:
            atr_multipliers = [1.0, 2.0, 3.0]
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = df['close'].rolling(bb_period).mean().iloc[-1]
        bb_std_dev = df['close'].rolling(bb_period).std().iloc[-1]
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        
        # Calculate targets based on trend
        if any('BULLISH' in signal for signal in technical_signals['moving_average_signals']):
            # Bullish targets
            target_1 = current_price + current_atr * atr_multipliers[0]
            target_2 = current_price + current_atr * atr_multipliers[1]
            target_3 = current_price + current_atr * atr_multipliers[2]
            
            stop_loss = current_price - current_atr
        else:
            # Bearish targets
            target_1 = current_price - current_atr * atr_multipliers[0]
            target_2 = current_price - current_atr * atr_multipliers[1]
            target_3 = current_price - current_atr * atr_multipliers[2]
            
            stop_loss = current_price + current_atr
        
        # Probability estimates (simplified)
        target_1_prob = 0.7
        target_2_prob = 0.4
        target_3_prob = 0.2
        
        precision = 5 if asset_class == 'FOREX' else 2
        
        return {
            'bullish_targets': {
                'target_1': round(max(target_1, current_price), precision),
                'target_2': round(max(target_2, current_price), precision),
                'target_3': round(max(target_3, current_price), precision),
                'probabilities': [target_1_prob, target_2_prob, target_3_prob]
            },
            'bearish_targets': {
                'target_1': round(min(target_1, current_price), precision),
                'target_2': round(min(target_2, current_price), precision),
                'target_3': round(min(target_3, current_price), precision),
                'probabilities': [target_1_prob, target_2_prob, target_3_prob]
            },
            'bollinger_bands': {
                'upper': round(bb_upper, precision),
                'middle': round(bb_middle, precision),
                'lower': round(bb_lower, precision)
            },
            'atr_value': round(current_atr, precision),
            'suggested_stop_loss': round(stop_loss, precision)
        }
    
    def _analyze_momentum(self, df: pd.DataFrame, current_price: float, asset_class: str) -> Dict:
        """Analyze price momentum indicators with asset-specific considerations"""
        
        # Rate of Change (ROC)
        roc_5 = ((current_price / df['close'].iloc[-6]) - 1) * 100 if len(df) > 5 else 0
        roc_10 = ((current_price / df['close'].iloc[-11]) - 1) * 100 if len(df) > 10 else 0
        roc_20 = ((current_price / df['close'].iloc[-21]) - 1) * 100 if len(df) > 20 else 0
        
        # Volume analysis (not applicable for FX)
        if asset_class == 'FOREX':
            volume_ratio = 1.0
            volume_signal = 'N/A'
        else:
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                volume_signal = 'HIGH'
            elif volume_ratio < 0.7:
                volume_signal = 'LOW'
            else:
                volume_signal = 'NORMAL'
        
        # Price momentum
        recent_highs = (df['close'].tail(5) == df['close'].tail(5).max()).sum()
        recent_lows = (df['close'].tail(5) == df['close'].tail(5).min()).sum()
        
        # Asset-specific momentum thresholds
        momentum_threshold = 1.5 if asset_class == 'FOREX' else 2.0
        
        if recent_highs >= 3:
            momentum_direction = 'STRONG_BULLISH'
        elif recent_lows >= 3:
            momentum_direction = 'STRONG_BEARISH'
        elif roc_5 > momentum_threshold:
            momentum_direction = 'BULLISH'
        elif roc_5 < -momentum_threshold:
            momentum_direction = 'BEARISH'
        else:
            momentum_direction = 'NEUTRAL'
        
        # Momentum strength
        momentum_strength = abs(roc_5) / 5.0  # Normalize to 0-1 scale
        momentum_strength = min(momentum_strength, 1.0)
        
        return {
            'momentum_direction': momentum_direction,
            'momentum_strength': round(momentum_strength, 2),
            'roc_5_day': round(roc_5, 2),
            'roc_10_day': round(roc_10, 2),
            'roc_20_day': round(roc_20, 2),
            'volume_ratio': round(volume_ratio, 2),
            'volume_signal': volume_signal
        }
    
    def _generate_overall_prediction(self, technical_signals: Dict, volatility_forecast: Dict,
                                   support_resistance: Dict, price_targets: Dict,
                                   momentum_analysis: Dict, asset_class: str) -> Dict:
        """Generate overall market prediction with asset-specific considerations"""
        
        # Scoring system
        bullish_score = 0
        bearish_score = 0
        
        # Technical signals scoring
        if technical_signals['rsi_signal'] == 'BULLISH':
            bullish_score += 2
        elif technical_signals['rsi_signal'] == 'BEARISH':
            bearish_score += 2
        
        bullish_ma_signals = len(technical_signals['moving_average_signals'])
        bullish_score += bullish_ma_signals
        
        if technical_signals['macd_signal'] == 'BULLISH':
            bullish_score += 1
        elif technical_signals['macd_signal'] == 'BEARISH':
            bearish_score += 1
        
        # Momentum scoring
        if momentum_analysis['momentum_direction'] in ['BULLISH', 'STRONG_BULLISH']:
            bullish_score += 2
        elif momentum_analysis['momentum_direction'] in ['BEARISH', 'STRONG_BEARISH']:
            bearish_score += 2
        
        # Asset-specific volatility scoring
        if volatility_forecast['volatility_regime'] == 'HIGH_VOLATILITY':
            if asset_class == 'FOREX':
                bearish_score += 0.5  # FX less affected by high vol
            else:
                bearish_score += 1
        elif volatility_forecast['volatility_regime'] == 'LOW_VOLATILITY':
            bullish_score += 1
        
        # Overall direction
        total_score = bullish_score + bearish_score
        if total_score == 0:
            direction = 'NEUTRAL'
            confidence = 0.5
        else:
            if bullish_score > bearish_score:
                direction = 'BULLISH'
                confidence = bullish_score / total_score
            elif bearish_score > bullish_score:
                direction = 'BEARISH'
                confidence = bearish_score / total_score
            else:
                direction = 'NEUTRAL'
                confidence = 0.5
        
        # Strength classification
        if confidence >= 0.8:
            strength = 'VERY_HIGH'
        elif confidence >= 0.7:
            strength = 'HIGH'
        elif confidence >= 0.6:
            strength = 'MODERATE'
        else:
            strength = 'LOW'
        
        # Asset-specific time horizon
        if asset_class == 'FOREX':
            if momentum_analysis['momentum_strength'] > 0.7:
                time_horizon = 'SHORT_TERM'  # 1-2 weeks
            else:
                time_horizon = 'MEDIUM_TERM'  # 1 month
        else:
            if momentum_analysis['momentum_strength'] > 0.7:
                time_horizon = 'SHORT_TERM'
            elif any('LONG_TERM' in signal for signal in technical_signals['moving_average_signals']):
                time_horizon = 'LONG_TERM'  # 2-3 months
            else:
                time_horizon = 'MEDIUM_TERM'
        
        # Key risks (asset-specific)
        risks = []
        if volatility_forecast['volatility_regime'] == 'HIGH_VOLATILITY':
            risks.append('High volatility environment - expect larger price swings')
        
        if asset_class == 'FOREX':
            risks.append('Central bank intervention risk')
            if momentum_analysis['volume_signal'] == 'N/A':
                pass  # No volume risk for FX
        else:
            if momentum_analysis['volume_ratio'] < 0.7:
                risks.append('Low volume - moves may not be sustainable')
        
        if technical_signals['rsi_value'] > 70:
            risks.append('Overbought conditions - potential pullback risk')
        elif technical_signals['rsi_value'] < 30:
            risks.append('Oversold conditions - potential bounce risk')
        
        return {
            'direction': direction,
            'confidence': round(confidence, 2),
            'strength': strength,
            'time_horizon': time_horizon,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'key_risks': risks,
            'summary': f"{strength} {direction} prediction with {confidence:.0%} confidence for {time_horizon.lower().replace('_', ' ')} horizon"
        }


# =============================================================================
# STRATEGY EXPLANATIONS (Updated for Multi-Asset)
# =============================================================================

def get_strategy_explanations() -> Dict[str, Dict]:
    """Get detailed explanations for all option strategies across asset classes"""
    return {
        'COVERED_CALL': {
            'name': 'Covered Call',
            'description': 'Income-generating strategy where you own the underlying and sell call options',
            'market_outlook': 'Neutral to slightly bullish',
            'asset_classes': ['INDICES', 'EQUITIES'],
            'max_profit': 'Strike price - underlying purchase price + premium received',
            'max_loss': 'Underlying purchase price - premium received',
            'breakeven': 'Underlying purchase price - premium received',
            'when_to_use': [
                'You own the underlying asset',
                'Expecting sideways to slightly bullish movement',
                'Want to generate additional income',
                'Willing to sell if called away'
            ],
            'pros': [
                'Generates additional income from premiums',
                'Reduces cost basis of position',
                'Limited downside protection from premium'
            ],
            'cons': [
                'Caps upside potential if asset rises significantly',
                'Asset can still decline below breakeven',
                'May be forced to sell at strike price'
            ],
            'examples': {
                'EQUITIES': 'Own 100 shares of AAPL at $150. Sell 1 call option with $155 strike for $2.50 premium.',
                'INDICES': 'Own 100 shares of SPY at $420. Sell 1 call option with $425 strike for $3.00 premium.',
                'FOREX': 'Not typically applicable for FX options due to settlement differences.'
            }
        },
        
        'CASH_SECURED_PUT': {
            'name': 'Cash Secured Put',
            'description': 'Strategy to acquire assets at a discount by selling put options while holding cash',
            'market_outlook': 'Neutral to bullish',
            'asset_classes': ['INDICES', 'EQUITIES', 'FOREX'],
            'max_profit': 'Premium received',
            'max_loss': 'Strike price - premium received',
            'breakeven': 'Strike price - premium received',
            'when_to_use': [
                'Want to buy asset at a lower price',
                'Have cash available for purchase',
                'Expecting neutral to bullish movement',
                'Comfortable owning the underlying'
            ],
            'pros': [
                'Earns premium while waiting to buy',
                'Acquires asset at effective discount if assigned',
                'Limited risk if you want to own anyway'
            ],
            'cons': [
                'Miss out if asset rises significantly',
                'May be forced to buy in declining market',
                'Ties up capital as collateral'
            ],
            'examples': {
                'EQUITIES': 'Want to buy TSLA. Sell put with $200 strike for $8.00 while holding $20,000 cash.',
                'INDICES': 'Want to buy QQQ. Sell put with $350 strike for $5.00 while holding cash.',
                'FOREX': 'Sell EUR/USD put at 1.0800 for 0.0050 premium expecting euro strength.'
            }
        },
        
        'IRON_CONDOR': {
            'name': 'Iron Condor',
            'description': 'Range-bound strategy selling both call and put spreads for premium collection',
            'market_outlook': 'Neutral (sideways movement)',
            'asset_classes': ['INDICES', 'EQUITIES', 'FOREX'],
            'max_profit': 'Net premium received',
            'max_loss': 'Strike width - net premium received',
            'breakeven': 'Two breakeven points: Lower strike + premium and Upper strike - premium',
            'when_to_use': [
                'Expecting low volatility and sideways movement',
                'High implied volatility environment',
                'Want to profit from time decay',
                'Limited directional bias'
            ],
            'pros': [
                'Profits from time decay and volatility contraction',
                'Defined risk and reward',
                'High probability of profit in range',
                'Excellent for range-bound FX pairs'
            ],
            'cons': [
                'Limited profit potential',
                'Loses if asset moves significantly either direction',
                'Multiple commissions and bid/ask spreads'
            ],
            'examples': {
                'EQUITIES': 'AAPL at $150. Sell $145 put, buy $140 put, sell $155 call, buy $160 call.',
                'INDICES': 'SPY at $420. Sell $415 put, buy $410 put, sell $425 call, buy $430 call.',
                'FOREX': 'EUR/USD at 1.1000. Sell 1.0950 put, buy 1.0900 put, sell 1.1050 call, buy 1.1100 call.'
            }
        },
        
        'BULL_CALL_SPREAD': {
            'name': 'Bull Call Spread',
            'description': 'Bullish strategy buying lower strike call and selling higher strike call',
            'market_outlook': 'Moderately bullish',
            'asset_classes': ['INDICES', 'EQUITIES', 'FOREX'],
            'max_profit': 'Strike width - net premium paid',
            'max_loss': 'Net premium paid',
            'breakeven': 'Lower strike + net premium paid',
            'when_to_use': [
                'Moderately bullish on underlying',
                'Want to reduce cost of long call',
                'Expecting move to specific price level',
                'Limited capital available'
            ],
            'pros': [
                'Lower cost than buying call outright',
                'Defined risk and reward',
                'Benefits from upward price movement'
            ],
            'cons': [
                'Limited upside profit potential',
                'Both options can expire worthless',
                'Time decay affects long option'
            ],
            'examples': {
                'EQUITIES': 'NVDA at $400. Buy $400 call for $15, sell $420 call for $8. Net cost: $7.',
                'INDICES': 'QQQ at $350. Buy $350 call for $8, sell $360 call for $4. Net cost: $4.',
                'FOREX': 'GBP/USD at 1.2500. Buy 1.2500 call, sell 1.2600 call for net debit.'
            }
        },
        
        'BEAR_PUT_SPREAD': {
            'name': 'Bear Put Spread',
            'description': 'Bearish strategy buying higher strike put and selling lower strike put',
            'market_outlook': 'Moderately bearish',
            'asset_classes': ['INDICES', 'EQUITIES', 'FOREX'],
            'max_profit': 'Strike width - net premium paid',
            'max_loss': 'Net premium paid',
            'breakeven': 'Higher strike - net premium paid',
            'when_to_use': [
                'Moderately bearish on underlying',
                'Want to reduce cost of long put',
                'Expecting decline to specific price level',
                'Limited capital available'
            ],
            'pros': [
                'Lower cost than buying put outright',
                'Defined risk and reward',
                'Benefits from downward price movement'
            ],
            'cons': [
                'Limited profit potential',
                'Both options can expire worthless',
                'Time decay affects long option'
            ],
            'examples': {
                'EQUITIES': 'AAPL at $150. Buy $150 put for $6, sell $140 put for $3. Net cost: $3.',
                'INDICES': 'SPY at $420. Buy $420 put for $8, sell $410 put for $4. Net cost: $4.',
                'FOREX': 'USD/JPY at 150.00. Buy 150.00 put, sell 148.00 put for net debit.'
            }
        },
        
        'LONG_STRADDLE': {
            'name': 'Long Straddle',
            'description': 'Volatility strategy buying both call and put at same strike',
            'market_outlook': 'Neutral direction, expecting high volatility',
            'asset_classes': ['INDICES', 'EQUITIES', 'FOREX'],
            'max_profit': 'Unlimited (theoretically)',
            'max_loss': 'Total premium paid',
            'breakeven': 'Two points: Strike ± total premium paid',
            'when_to_use': [
                'Expecting significant price movement',
                'Uncertain about direction',
                'Before earnings or major announcements',
                'Low implied volatility environment',
                'Before central bank decisions (FX)'
            ],
            'pros': [
                'Profits from large moves in either direction',
                'Unlimited upside potential',
                'Benefits from volatility expansion',
                'Excellent for event-driven trades'
            ],
            'cons': [
                'High premium cost',
                'Needs significant movement to be profitable',
                'Time decay hurts both options'
            ],
            'examples': {
                'EQUITIES': 'TSLA at $200 before earnings. Buy $200 call and $200 put for total cost of $20.',
                'INDICES': 'SPY at $420 before Fed meeting. Buy $420 call and put for total of $16.',
                'FOREX': 'EUR/USD at 1.1000 before ECB meeting. Buy 1.1000 call and put for 0.0120 total.'
            }
        },
        
        'PROTECTIVE_PUT': {
            'name': 'Protective Put',
            'description': 'Insurance strategy buying put options while owning the underlying',
            'market_outlook': 'Bullish but want downside protection',
            'asset_classes': ['INDICES', 'EQUITIES', 'FOREX'],
            'max_profit': 'Unlimited (appreciation - put premium)',
            'max_loss': 'Underlying price - strike price + put premium',
            'breakeven': 'Underlying price + put premium paid',
            'when_to_use': [
                'Own asset and want downside protection',
                'Expecting volatility or uncertainty',
                'Protecting gains in profitable position',
                'Cannot afford significant losses'
            ],
            'pros': [
                'Provides downside protection',
                'Maintains upside potential',
                'Peace of mind during volatile periods',
                'Good for FX exposure hedging'
            ],
            'cons': [
                'Cost of insurance reduces returns',
                'Premium lost if asset doesn\'t decline',
                'Time decay reduces put value'
            ],
            'examples': {
                'EQUITIES': 'Own 100 AAPL shares at $150. Buy $145 put for $4 as insurance.',
                'INDICES': 'Own 100 SPY shares at $420. Buy $410 put for $6 as portfolio hedge.',
                'FOREX': 'Long EUR/USD at 1.1000. Buy 1.0900 put for 0.0080 to limit downside.'
            }
        }
    }


# =============================================================================
# ENHANCED STREAMLIT INTERFACE - MULTI-ASSET DASHBOARD
# =============================================================================

def main():
    st.set_page_config(
        page_title="Multi-Asset Options Dashboard", 
        page_icon="🌍", 
        layout="wide"
    )
    
    st.title("🌍 Multi-Asset Options Dashboard")
    st.markdown("**Professional Trading Platform** • Indices • Equities • FX Options")
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'greeks_result' not in st.session_state:
        st.session_state.greeks_result = None
    if 'backtest_result' not in st.session_state:
        st.session_state.backtest_result = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'selected_asset_class' not in st.session_state:
        st.session_state.selected_asset_class = 'EQUITIES'
    
    # Asset Class Selector (Top Level)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎯 Select Asset Class")
        asset_class = st.selectbox(
            "Choose your trading focus:",
            ['INDICES', 'EQUITIES', 'FOREX'],
            index=['INDICES', 'EQUITIES', 'FOREX'].index(st.session_state.selected_asset_class),
            format_func=lambda x: {
                'INDICES': '📊 Indices & ETFs',
                'EQUITIES': '📈 Individual Stocks', 
                'FOREX': '💱 Currency Pairs'
            }[x],
            help="Select the asset class you want to analyze and trade options on"
        )
        
        if asset_class != st.session_state.selected_asset_class:
            st.session_state.selected_asset_class = asset_class
            st.rerun()
    
    # Asset Class Description
    asset_config = {
        'INDICES': {
            'description': '📊 **Index ETFs & Volatility Products** - Diversified exposure to market segments',
            'examples': 'SPY, QQQ, IWM, EWU, VGK, VIX',
            'characteristics': 'Lower volatility, broad market exposure, high liquidity'
        },
        'EQUITIES': {
            'description': '📈 **Individual Stocks** - Direct company exposure with higher alpha potential',
            'examples': 'AAPL, MSFT, GOOGL, TSLA, NVDA, META',
            'characteristics': 'Higher volatility, company-specific risk, earnings-driven moves'
        },
        'FOREX': {
            'description': '💱 **Currency Pairs** - Global forex markets with 24/5 trading',
            'examples': 'EUR/USD, GBP/USD, USD/JPY, AUD/USD',
            'characteristics': 'Central bank driven, global macro exposure, different settlement'
        }
    }
    
    st.info(f"{asset_config[asset_class]['description']}\n\n"
           f"**Popular Instruments:** {asset_config[asset_class]['examples']}\n\n"
           f"**Characteristics:** {asset_config[asset_class]['characteristics']}")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analysis", 
        "📚 Strategy Guide", 
        "🔢 Options Greeks", 
        "📈 Backtester", 
        "🔮 Market Predictions"
    ])
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 Configuration")
        
        # API Key
        polygon_key = st.text_input(
            "Polygon API Key (Required)", 
            value="igO7PgpW43MsVcJvr1ZuxZ_vYrH87jLZ", 
            type="password",
            help="Real Polygon API key required - supports all asset classes"
        )
        
        if not polygon_key:
            st.error("❌ Polygon API key required")
            st.stop()
        
        st.success("✅ API key provided")
        
        # Initialize strategist
        try:
            strategist = MultiAssetOptionsStrategist(polygon_key)
        except Exception as e:
            st.error(f"❌ Failed to initialize: {str(e)}")
            st.stop()
        
        st.markdown("---")
        
        # Asset-Specific Discovery
        st.header(f"🔍 Discover {asset_class}")
        
        # Popular symbols for current asset class
        popular_symbols = strategist.get_popular_symbols(asset_class)
        if popular_symbols:
            st.markdown("**🌟 Popular Symbols:**")
            for symbol in popular_symbols[:5]:  # Show top 5
                if st.button(f"📊 {symbol}", key=f"pop_{symbol}"):
                    st.session_state.selected_symbol = symbol
            
            if len(popular_symbols) > 5:
                with st.expander("Show more popular symbols"):
                    for symbol in popular_symbols[5:]:
                        if st.button(f"📊 {symbol}", key=f"exp_{symbol}"):
                            st.session_state.selected_symbol = symbol
        
        # Search functionality
        st.markdown("**🔍 Search Symbols:**")
        search_query = st.text_input(
            "Search",
            placeholder=f"Search {asset_class.lower()}...",
            help=f"Search for {asset_class.lower()} symbols"
        )
        
        if search_query and len(search_query) > 1:
            with st.spinner(f"Searching {asset_class}..."):
                try:
                    search_results = strategist.search_symbols(asset_class, search_query)
                    if search_results:
                        st.success(f"Found {len(search_results)} matches:")
                        for result in search_results[:10]:  # Show top 10
                            ticker = result['ticker']
                            name = result.get('name', 'Unknown')
                            if st.button(f"{ticker}: {name[:30]}{'...' if len(name) > 30 else ''}", 
                                       key=f"search_{ticker}"):
                                st.session_state.selected_symbol = ticker
                    else:
                        st.warning("No matches found")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
        
        # Quick data check
        if st.button("📊 Test Data Quality"):
            if hasattr(st.session_state, 'selected_symbol'):
                test_symbol = st.session_state.selected_symbol
            else:
                test_symbol = popular_symbols[0] if popular_symbols else 'SPY'
            
            with st.spinner(f"Testing {test_symbol} data..."):
                try:
                    data_check = strategist.quick_data_check(test_symbol, asset_class)
                    if data_check['available']:
                        st.success(f"✅ {test_symbol} data looks good!")
                        st.write(f"• **Valid records:** {data_check['valid_records']}/{data_check['total_records']}")
                        if data_check['latest_price']:
                            st.write(f"• **Latest price:** {data_check['latest_price']:.4f}")
                        st.write(f"• **Date range:** {data_check['date_range']}")
                    else:
                        st.error(f"❌ {test_symbol} data issues:")
                        if 'reason' in data_check:
                            st.write(f"• **Reason:** {data_check['reason']}")
                        if 'error' in data_check:
                            st.write(f"• **Error:** {data_check['error']}")
                except Exception as e:
                    st.error(f"Data check failed: {str(e)}")
        
        # Options availability check
        if st.button("🎯 Check Options"):
            if hasattr(st.session_state, 'selected_symbol'):
                test_symbol = st.session_state.selected_symbol
            else:
                test_symbol = popular_symbols[0] if popular_symbols else 'SPY'
            
            with st.spinner(f"Checking {test_symbol} options..."):
                try:
                    options_check = strategist.check_options_availability(test_symbol, asset_class)
                    if options_check['has_options']:
                        st.success(f"✅ {test_symbol} has options!")
                        st.write(f"• **Contracts found:** {options_check.get('contract_count', 0)}")
                        st.write(f"• **Sample expiration:** {options_check.get('sample_expiration', 'N/A')}")
                    else:
                        st.warning(f"❌ {options_check['status']}")
                except Exception as e:
                    st.error(f"Options check failed: {str(e)}")
        
        st.markdown("---")
        
        # Analysis section
        st.header("📊 Analysis")
        
        # Default symbol based on asset class
        default_symbols = {
            'INDICES': 'SPY',
            'EQUITIES': 'AAPL', 
            'FOREX': 'EURUSD'
        }
        
        # Use selected symbol if available, otherwise default
        default_symbol = getattr(st.session_state, 'selected_symbol', default_symbols[asset_class])
        
        symbol_input = st.text_input(
            f"Symbol to Analyze ({asset_class})",
            value=default_symbol,
            help=f"Enter {asset_class.lower()} ticker symbol"
        )
        
        debug_mode = st.checkbox(
            "🐛 Debug Mode",
            value=False,
            help="Show detailed analysis information"
        )
        
        # Store debug mode in session state
        st.session_state.debug_mode = debug_mode
        
        analyze_button = st.button(
            "🚀 Analyze Real Data",
            type="primary",
            disabled=not symbol_input
        )
    
    # Tab 1: Analysis
    with tab1:
        if analyze_button and symbol_input:
            with st.spinner(f"Analyzing {symbol_input} ({asset_class}) with real data..."):
                result = strategist.analyze_symbol(symbol_input.upper(), asset_class, debug=debug_mode)
            
            if result['success']:
                # Store result in session state
                st.session_state.analysis_result = result
                
                # Display results
                st.success(f"✅ {asset_class} analysis complete for {result['ticker']}")
                
                # Asset-specific success message
                if asset_class == 'FOREX':
                    st.info("💱 FX analysis includes 24/5 market considerations and currency-specific volatility modeling")
                elif asset_class == 'INDICES':
                    st.info("📊 Index analysis includes diversification benefits and sector rotation insights")
                else:
                    st.info("📈 Equity analysis includes company-specific risk factors and earnings considerations")
                
                # Market Data Summary
                st.subheader(f"📊 {asset_class} Market Data Summary")
                underlying = result['underlying_data']
                analysis = result['market_analysis']
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Asset-specific price formatting
                if asset_class == 'FOREX':
                    price_display = f"{underlying['current_price']:.5f}"
                    price_label = "Current Rate"
                else:
                    price_display = f"${underlying['current_price']:.2f}"
                    price_label = "Current Price"
                
                with col1:
                    st.metric(price_label, price_display)
                    st.metric("1-Day Change", f"{analysis['price_change_1d']:.2f}%")
                with col2:
                    st.metric("RSI", f"{analysis['rsi']:.1f}")
                    st.metric("5-Day Change", f"{analysis['price_change_5d']:.2f}%")
                with col3:
                    st.metric("Realized Vol (21d)", f"{analysis['realized_vol']:.1%}")
                    st.metric("20-Day Change", f"{analysis['price_change_20d']:.2f}%")
                with col4:
                    st.metric("Data Points", underlying['data_points'])
                    if asset_class != 'FOREX':
                        st.metric("Volume vs Avg", f"{analysis['volume_vs_avg']:.2f}x")
                    else:
                        st.metric("Market", "24/5 Trading")
                
                # Trading Chart
                st.subheader(f"📈 {asset_class} Trading Chart")
                try:
                    chart_data = {
                        'ticker': underlying['ticker'],
                        'current_price': underlying['current_price'],
                        'historical_data': underlying['historical_data']
                    }
                    chart = strategist.create_trading_chart(chart_data, asset_class)
                    st.plotly_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create {asset_class} chart: {str(e)}")
                    if debug_mode:
                        st.write("**Chart Debug Info:**")
                        st.write(f"- Asset Class: {asset_class}")
                        st.write(f"- Data shape: {underlying.get('historical_data', pd.DataFrame()).shape}")
                
                # Market Analysis
                st.subheader("📈 Market Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trend_color = "🟢" if "BULLISH" in analysis['trend'] else "🔴" if "BEARISH" in analysis['trend'] else "🟡"
                    st.metric("Trend", f"{trend_color} {analysis['trend']}")
                    st.metric("Trend Strength", f"{analysis['trend_strength']:.2f}")
                
                with col2:
                    vol_color = "🔴" if analysis['volatility_regime'] in ['HIGH_VOL', 'EXTREME_VOL'] else "🟢"
                    st.metric("Volatility", f"{vol_color} {analysis['volatility_regime']}")
                    st.metric("BB Position", f"{analysis['bb_position']:.1f}%")
                
                with col3:
                    momentum_color = "🔴" if "OVERBOUGHT" in analysis['momentum'] else "🟢" if "OVERSOLD" in analysis['momentum'] else "🟡"
                    st.metric("Momentum", f"{momentum_color} {analysis['momentum']}")
                    st.metric("BB Signal", analysis['bb_signal'])
                
                # Options Data Summary
                st.subheader("🎯 Options Data Summary")
                options = result['options_data']
                pricing = options.get('pricing_breakdown', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Expiration", options['expiration'])
                with col2:
                    st.metric("Available Calls", len(options['calls']))
                with col3:
                    st.metric("Available Puts", len(options['puts']))
                with col4:
                    st.metric("Days to Expiry", options['days_to_expiry'])
                
                # Asset-specific options insights
                if asset_class == 'FOREX':
                    st.info("💱 **FX Options Note:** Settlement differences and 24/5 trading affect strategy selection")
                elif asset_class == 'INDICES':
                    st.info("📊 **Index Options Note:** European-style settlement and cash settlement for some products")
                
                # Strategy Recommendations
                st.subheader("💡 Strategy Recommendations")
                
                st.success(f"**Best Strategy for {asset_class}:** {result['best_strategy']} (Confidence: {result['confidence']:.1f}/10)")
                
                # Asset-specific strategy note
                asset_strategy_notes = {
                    'FOREX': "FX options strategies focus on volatility and central bank events",
                    'INDICES': "Index strategies benefit from diversification and lower single-name risk",
                    'EQUITIES': "Equity strategies can capture company-specific moves and earnings"
                }
                
                st.info(f"📝 **{asset_class} Strategy Note:** {asset_strategy_notes[asset_class]}")
                
                st.markdown("**All Strategy Scores:**")
                for strategy, score in result['strategy_scores'].items():
                    st.write(f"• **{strategy}:** {score:.1f}/10")
                
                # Export data
                st.subheader("📤 Export Analysis")
                
                export_data = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'asset_class': asset_class,
                    'ticker': result['ticker'],
                    'market_analysis': result['market_analysis'],
                    'strategy_scores': result['strategy_scores'],
                    'best_strategy': result['best_strategy'],
                    'confidence': result['confidence']
                }
                
                st.download_button(
                    f"📋 Download {asset_class} Analysis",
                    json.dumps(export_data, indent=2),
                    f"{result['ticker']}_{asset_class}_analysis.json",
                    "application/json"
                )
            
            else:
                st.error(f"❌ {asset_class} analysis failed: {result['error']}")
                
                # Asset-specific troubleshooting
                if asset_class == 'FOREX' and 'No historical data' in result['error']:
                    st.info("💡 **FX Troubleshooting:** Try major pairs like EURUSD, GBPUSD, or USDJPY")
                elif asset_class == 'INDICES' and 'options' in result['error'].lower():
                    st.info("💡 **Index Troubleshooting:** Try liquid ETFs like SPY, QQQ, or IWM")
        
        else:
            # Instructions for current asset class
            asset_instructions = {
                'INDICES': """
                ## 📊 Index & ETF Options Analysis
                
                **Analyze diversified market exposure** with professional-grade tools for index options trading.
                
                ### 🎯 **Popular Index Products:**
                - **SPY**: S&P 500 ETF (most liquid options)
                - **QQQ**: NASDAQ 100 ETF (tech focus)
                - **IWM**: Russell 2000 ETF (small caps)
                - **EWU**: UK market exposure (FTSE focus)
                - **VGK**: European markets
                - **VIX**: Volatility index products
                
                ### 💡 **Index Options Advantages:**
                - **Diversification**: Reduced single-name risk
                - **Liquidity**: Tight bid-ask spreads
                - **Variety**: Sector, region, and style exposure
                - **Settlement**: Cash settlement for many products
                
                ### 🚀 **Getting Started:**
                1. **Select a popular symbol** from the sidebar
                2. **Click "Analyze Real Data"** for comprehensive analysis
                3. **Review strategy recommendations** optimized for index products
                4. **Use backtesting** to validate historical performance
                """,
                
                'EQUITIES': """
                ## 📈 Individual Stock Options Analysis
                
                **Target specific companies** with higher alpha potential and company-specific catalysts.
                
                ### 🎯 **Popular Equity Options:**
                - **AAPL**: Apple (earnings-driven moves)
                - **MSFT**: Microsoft (enterprise focus)
                - **GOOGL**: Alphabet (search/AI exposure)
                - **TSLA**: Tesla (high volatility)
                - **NVDA**: NVIDIA (AI/semiconductors)
                - **META**: Meta Platforms (social media)
                
                ### 💡 **Equity Options Characteristics:**
                - **Higher Volatility**: Greater profit potential
                - **Earnings Events**: Quarterly catalysts
                - **Company-Specific**: Fundamental analysis matters
                - **Sector Rotation**: Style considerations
                
                ### 🚀 **Getting Started:**
                1. **Search for your target company** using the sidebar
                2. **Analyze before earnings** for volatility opportunities
                3. **Consider sector trends** in strategy selection
                4. **Use Greeks analysis** for risk management
                """,
                
                'FOREX': """
                ## 💱 Currency Options Analysis
                
                **Trade global macro themes** with 24/5 markets and central bank-driven moves.
                
                ### 🎯 **Major Currency Pairs:**
                - **EURUSD**: Euro/US Dollar (most liquid)
                - **GBPUSD**: British Pound/US Dollar
                - **USDJPY**: US Dollar/Japanese Yen
                - **USDCHF**: US Dollar/Swiss Franc
                - **AUDUSD**: Australian Dollar/US Dollar
                - **USDCAD**: US Dollar/Canadian Dollar
                
                ### 💡 **FX Options Characteristics:**
                - **24/5 Trading**: Global market access
                - **Central Bank Events**: Policy-driven volatility
                - **Macro Themes**: Economic data sensitivity
                - **Different Settlement**: Delivery considerations
                
                ### 🚀 **Getting Started:**
                1. **Start with major pairs** for best liquidity
                2. **Monitor central bank calendars** for volatility
                3. **Use iron condors** for range-bound trading
                4. **Consider time zones** for optimal trading windows
                """
            }
            
            st.markdown(asset_instructions[asset_class])
    
    # Tab 2: Strategy Guide
    with tab2:
        st.header("📚 Multi-Asset Strategy Guide")
        
        st.markdown(f"""
        Comprehensive guide for **{asset_class} options strategies** with asset-specific considerations 
        and risk management techniques.
        """)
        
        # Get strategy explanations
        strategies = get_strategy_explanations()
        
        # Filter strategies by asset class
        for strategy_key, strategy_info in strategies.items():
            if asset_class in strategy_info.get('asset_classes', []):
                with st.expander(f"📋 {strategy_info['name']} (Optimized for {asset_class})", expanded=False):
                    
                    # Strategy header
                    st.markdown(f"**{strategy_info['description']}**")
                    
                    # Asset class applicability
                    st.markdown(f"**✅ Applicable to:** {', '.join(strategy_info['asset_classes'])}")
                    
                    # Market outlook and basics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Strategy Overview")
                        st.markdown(f"**Market Outlook:** {strategy_info['market_outlook']}")
                        st.markdown(f"**Max Profit:** {strategy_info['max_profit']}")
                        st.markdown(f"**Max Loss:** {strategy_info['max_loss']}")
                        st.markdown(f"**Breakeven:** {strategy_info['breakeven']}")
                    
                    with col2:
                        st.markdown("### 💡 When to Use")
                        for condition in strategy_info['when_to_use']:
                            st.markdown(f"• {condition}")
                    
                    # Pros and Cons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### ✅ Pros")
                        for pro in strategy_info['pros']:
                            st.markdown(f"• {pro}")
                    
                    with col2:
                        st.markdown("### ❌ Cons")
                        for con in strategy_info['cons']:
                            st.markdown(f"• {con}")
                    
                    # Asset-specific examples
                    st.markdown("### 📝 Examples by Asset Class")
                    examples = strategy_info.get('examples', {})
                    
                    if asset_class in examples:
                        st.success(f"**{asset_class} Example:** {examples[asset_class]}")
                    
                    # Show other asset class examples if available
                    other_examples = {k: v for k, v in examples.items() if k != asset_class}
                    if other_examples:
                        st.markdown("**Other Asset Classes:**")
                        for ac, example in other_examples.items():
                            st.info(f"**{ac}:** {example}")
                    
                    st.markdown("---")
        
        # Asset-specific strategy recommendations
        st.markdown(f"## 🎯 {asset_class}-Specific Strategy Recommendations")
        
        if asset_class == 'FOREX':
            st.markdown("""
            ### 💱 FX Options Strategy Considerations:
            
            **🎯 Best Strategies for FX:**
            - **Iron Condors**: Excellent for range-bound currency pairs
            - **Straddles**: Perfect for central bank meetings and economic releases
            - **Bull/Bear Spreads**: For directional macro themes
            
            **⚠️ FX-Specific Risks:**
            - **Settlement Differences**: Physical delivery vs cash settlement
            - **Time Zones**: 24/5 trading affects option pricing
            - **Central Bank Risk**: Policy changes create large moves
            - **Correlation Risk**: Currency pairs often move together
            
            **💡 FX Strategy Tips:**
            - Monitor economic calendars for volatility events
            - Use lower delta targets due to different volatility patterns
            - Consider carry trade implications
            - Watch for intervention risk in major pairs
            """)
        
        elif asset_class == 'INDICES':
            st.markdown("""
            ### 📊 Index Options Strategy Considerations:
            
            **🎯 Best Strategies for Indices:**
            - **Covered Calls**: Generate income on diversified holdings
            - **Protective Puts**: Portfolio insurance
            - **Iron Condors**: Benefit from market stability
            - **Bull Call Spreads**: Participate in market uptrends
            
            **⚠️ Index-Specific Considerations:**
            - **European Settlement**: Many index options are European-style
            - **Cash Settlement**: No physical delivery of shares
            - **Diversification**: Lower single-name risk
            - **Correlation**: Sector rotation affects performance
            
            **💡 Index Strategy Tips:**
            - Use for portfolio-level hedging
            - Monitor sector rotation and style preferences
            - Consider volatility index products for pure vol plays
            - Align with broader market outlook
            """)
        
        else:  # EQUITIES
            st.markdown("""
            ### 📈 Equity Options Strategy Considerations:
            
            **🎯 Best Strategies for Equities:**
            - **Covered Calls**: Income on stock holdings
            - **Protective Puts**: Insurance on concentrated positions
            - **Straddles**: Earnings and event-driven plays
            - **Bull/Bear Spreads**: Moderate directional exposure
            
            **⚠️ Equity-Specific Risks:**
            - **Earnings Risk**: Quarterly volatility spikes
            - **Single-Name Risk**: Company-specific events
            - **Liquidity Risk**: Some options have wide spreads
            - **Assignment Risk**: Early exercise considerations
            
            **💡 Equity Strategy Tips:**
            - Time strategies around earnings announcements
            - Monitor analyst upgrades/downgrades
            - Consider sector and style factors
            - Use fundamental analysis alongside technical
            """)
    
    # Tab 3: Options Greeks (Multi-Asset)
    with tab3:
        st.header(f"🔢 {asset_class} Options Greeks")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Auto-populate with symbol from analysis tab if available
            default_symbol = default_symbols = {
                'INDICES': 'SPY',
                'EQUITIES': 'AAPL', 
                'FOREX': 'EURUSD'
            }[asset_class]
            
            if (st.session_state.analysis_result and 
                st.session_state.analysis_result.get('success') and
                st.session_state.analysis_result.get('asset_class') == asset_class):
                default_symbol = st.session_state.analysis_result['ticker']
            
            greeks_symbol = st.text_input(
                f"Symbol for Greeks Analysis ({asset_class})",
                value=default_symbol,
                help=f"Enter {asset_class.lower()} ticker symbol"
            )
            
            if (st.session_state.analysis_result and 
                st.session_state.analysis_result.get('success') and
                st.session_state.analysis_result['ticker'] == greeks_symbol.upper() and
                st.session_state.analysis_result.get('asset_class') == asset_class):
                st.info("🔗 This matches your symbol from the Analysis tab!")
        
        with col2:
            get_greeks_button = st.button(
                f"📊 Get {asset_class} Greeks",
                type="primary",
                disabled=not greeks_symbol or not polygon_key
            )
        
        if get_greeks_button and greeks_symbol:
            with st.spinner(f"Fetching {asset_class} Options Greeks for {greeks_symbol}..."):
                try:
                    greeks_result = strategist.get_options_greeks(greeks_symbol.upper(), asset_class)
                    
                    # Store result in session state
                    st.session_state.greeks_result = greeks_result
                    
                    st.success(f"✅ {asset_class} Greeks analysis complete for {greeks_result['underlying_ticker']}")
                    
                    # Show connection to analysis tab if same symbol
                    if (st.session_state.analysis_result and 
                        st.session_state.analysis_result.get('success') and
                        st.session_state.analysis_result['ticker'] == greeks_result['underlying_ticker'] and
                        st.session_state.analysis_result.get('asset_class') == asset_class):
                        st.info(f"🔗 This {asset_class} Greeks analysis matches your symbol from the Analysis tab!")
                    
                    # Asset-specific insights
                    if asset_class == 'FOREX':
                        st.info("💱 FX Greeks reflect 24/5 trading and central bank policy impacts")
                    elif asset_class == 'INDICES':
                        st.info("📊 Index Greeks show diversified exposure with lower single-name risk")
                    else:
                        st.info("📈 Equity Greeks capture company-specific volatility and earnings effects")
                    
                    # Summary metrics
                    st.subheader(f"📊 {asset_class} Greeks Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Asset-specific price formatting
                    if asset_class == 'FOREX':
                        price_display = f"{greeks_result['underlying_price']:.5f}"
                    else:
                        price_display = f"${greeks_result['underlying_price']:.2f}"
                    
                    with col1:
                        st.metric("Underlying Price", price_display)
                        st.metric("Total Contracts", greeks_result['total_contracts'])
                    
                    with col2:
                        st.metric("Expiration", greeks_result['expiration'])
                        st.metric("Days to Expiry", greeks_result['days_to_expiry'])
                    
                    with col3:
                        st.metric("ATM Options", greeks_result['summary_stats'].get('atm_options', 0))
                        st.metric("OTM Calls", greeks_result['summary_stats'].get('otm_calls', 0))
                    
                    with col4:
                        st.metric("OTM Puts", greeks_result['summary_stats'].get('otm_puts', 0))
                        max_gamma_strike = greeks_result['summary_stats'].get('highest_gamma_strike', 0)
                        if asset_class == 'FOREX':
                            st.metric("Max Gamma Strike", f"{max_gamma_strike:.5f}")
                        else:
                            st.metric("Max Gamma Strike", f"${max_gamma_strike:.2f}")
                    
                    # Greeks Tables
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📞 Call Options Greeks")
                        calls_df = greeks_result['calls_greeks'].copy()
                        if not calls_df.empty:
                            # Select and format columns for display
                            display_calls = calls_df[['strike', 'price', 'delta', 'gamma', 'theta', 'vega', 'moneyness']].copy()
                            display_calls = display_calls.sort_values('strike')
                            
                            # Asset-specific formatting
                            if asset_class == 'FOREX':
                                display_calls['price'] = display_calls['price'].apply(lambda x: f"{x:.4f}")
                                display_calls['strike'] = display_calls['strike'].apply(lambda x: f"{x:.5f}")
                            else:
                                display_calls['price'] = display_calls['price'].apply(lambda x: f"${x:.2f}")
                                display_calls['strike'] = display_calls['strike'].apply(lambda x: f"${x:.2f}")
                            
                            display_calls['moneyness'] = display_calls['moneyness'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(display_calls, use_container_width=True)
                        else:
                            st.info("No call options data available")
                    
                    with col2:
                        st.subheader("📱 Put Options Greeks")
                        puts_df = greeks_result['puts_greeks'].copy()
                        if not puts_df.empty:
                            # Select and format columns for display
                            display_puts = puts_df[['strike', 'price', 'delta', 'gamma', 'theta', 'vega', 'moneyness']].copy()
                            display_puts = display_puts.sort_values('strike', ascending=False)
                            
                            # Asset-specific formatting
                            if asset_class == 'FOREX':
                                display_puts['price'] = display_puts['price'].apply(lambda x: f"{x:.4f}")
                                display_puts['strike'] = display_puts['strike'].apply(lambda x: f"{x:.5f}")
                            else:
                                display_puts['price'] = display_puts['price'].apply(lambda x: f"${x:.2f}")
                                display_puts['strike'] = display_puts['strike'].apply(lambda x: f"${x:.2f}")
                            
                            display_puts['moneyness'] = display_puts['moneyness'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(display_puts, use_container_width=True)
                        else:
                            st.info("No put options data available")
                    
                    # Greeks Visualization
                    st.subheader(f"📈 {asset_class} Greeks Visualization")
                    
                    all_greeks_df = greeks_result['all_greeks']
                    
                    if not all_greeks_df.empty:
                        # Create visualizations
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Delta by Strike', 'Gamma by Strike', 'Theta by Strike', 'Vega by Strike'),
                            vertical_spacing=0.12,
                            horizontal_spacing=0.1
                        )
                        
                        # Separate calls and puts
                        calls_data = all_greeks_df[all_greeks_df['contract_type'] == 'call'].sort_values('strike')
                        puts_data = all_greeks_df[all_greeks_df['contract_type'] == 'put'].sort_values('strike')
                        
                        # Color scheme based on asset class
                        if asset_class == 'FOREX':
                            call_color = '#00ff88'
                            put_color = '#ff6b35'
                        elif asset_class == 'INDICES':
                            call_color = '#004e89'
                            put_color = '#7209b7'
                        else:  # EQUITIES
                            call_color = '#00ff88'
                            put_color = '#ff4444'
                        
                        # Delta plot
                        if not calls_data.empty:
                            fig.add_trace(go.Scatter(x=calls_data['strike'], y=calls_data['delta'], 
                                                   mode='lines+markers', name=f'{asset_class} Calls Delta', 
                                                   line=dict(color=call_color)), row=1, col=1)
                        if not puts_data.empty:
                            fig.add_trace(go.Scatter(x=puts_data['strike'], y=puts_data['delta'], 
                                                   mode='lines+markers', name=f'{asset_class} Puts Delta', 
                                                   line=dict(color=put_color)), row=1, col=1)
                        
                        # Gamma plot
                        if not calls_data.empty:
                            fig.add_trace(go.Scatter(x=calls_data['strike'], y=calls_data['gamma'], 
                                                   mode='lines+markers', name=f'{asset_class} Calls Gamma', 
                                                   line=dict(color=call_color), showlegend=False), row=1, col=2)
                        if not puts_data.empty:
                            fig.add_trace(go.Scatter(x=puts_data['strike'], y=puts_data['gamma'], 
                                                   mode='lines+markers', name=f'{asset_class} Puts Gamma', 
                                                   line=dict(color=put_color), showlegend=False), row=1, col=2)
                        
                        # Theta plot
                        if not calls_data.empty:
                            fig.add_trace(go.Scatter(x=calls_data['strike'], y=calls_data['theta'], 
                                                   mode='lines+markers', name=f'{asset_class} Calls Theta', 
                                                   line=dict(color=call_color), showlegend=False), row=2, col=1)
                        if not puts_data.empty:
                            fig.add_trace(go.Scatter(x=puts_data['strike'], y=puts_data['theta'], 
                                                   mode='lines+markers', name=f'{asset_class} Puts Theta', 
                                                   line=dict(color=put_color), showlegend=False), row=2, col=1)
                        
                        # Vega plot
                        if not calls_data.empty:
                            fig.add_trace(go.Scatter(x=calls_data['strike'], y=calls_data['vega'], 
                                                   mode='lines+markers', name=f'{asset_class} Calls Vega', 
                                                   line=dict(color=call_color), showlegend=False), row=2, col=2)
                        if not puts_data.empty:
                            fig.add_trace(go.Scatter(x=puts_data['strike'], y=puts_data['vega'], 
                                                   mode='lines+markers', name=f'{asset_class} Puts Vega', 
                                                   line=dict(color=put_color), showlegend=False), row=2, col=2)
                        
                        # Add current price line to all subplots
                        current_price = greeks_result['underlying_price']
                        for row in [1, 2]:
                            for col in [1, 2]:
                                fig.add_vline(x=current_price, line_dash="dash", line_color="white", 
                                            row=row, col=col)
                        
                        fig.update_layout(
                            height=600,
                            title=f"{asset_class} Options Greeks for {greeks_result['underlying_ticker']}",
                            template='plotly_dark'
                        )
                        
                        fig.update_xaxes(title_text="Strike Price")
                        fig.update_yaxes(title_text="Delta", row=1, col=1)
                        fig.update_yaxes(title_text="Gamma", row=1, col=2)
                        fig.update_yaxes(title_text="Theta", row=2, col=1)
                        fig.update_yaxes(title_text="Vega", row=2, col=2)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Asset-Specific Greeks Analysis
                    st.subheader(f"📚 Understanding {asset_class} Greeks")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if asset_class == 'FOREX':
                            st.markdown("""
                            **💱 FX Greeks Characteristics:**
                            - **Lower Gamma**: Central bank intervention stabilizes
                            - **24/5 Theta**: Continuous time decay
                            - **Policy Vega**: Central bank meetings drive volatility
                            - **Carry Delta**: Interest rate differential effects
                            """)
                        elif asset_class == 'INDICES':
                            st.markdown("""
                            **📊 Index Greeks Characteristics:**
                            - **Diversified Delta**: Broad market exposure
                            - **Stable Gamma**: Lower single-name risk
                            - **Regime Vega**: Market volatility clustering
                            - **Sector Theta**: Time decay across holdings
                            """)
                        else:
                            st.markdown("""
                            **📈 Equity Greeks Characteristics:**
                            - **Higher Gamma**: More sensitive to moves
                            - **Event Vega**: Earnings volatility spikes
                            - **News Delta**: Company-specific reactions
                            - **Earnings Theta**: Time decay acceleration
                            """)
                    
                    with col2:
                        st.markdown("""
                        **🔺 Delta (Δ)**
                        - Measures price sensitivity to $1 move in underlying
                        - Calls: 0 to 1 | Puts: -1 to 0
                        - Higher absolute delta = more sensitive to price moves
                        
                        **🔄 Gamma (Γ)**
                        - Rate of change of Delta
                        - Highest for ATM options
                        - Shows how Delta will change as price moves
                        """)
                    
                    # Export Greeks data
                    st.subheader("📤 Export Greeks Data")
                    
                    greeks_export = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'asset_class': asset_class,
                        'underlying_ticker': greeks_result['underlying_ticker'],
                        'underlying_price': greeks_result['underlying_price'],
                        'expiration': greeks_result['expiration'],
                        'days_to_expiry': greeks_result['days_to_expiry'],
                        'summary_stats': greeks_result['summary_stats'],
                        'total_contracts': greeks_result['total_contracts']
                    }
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            f"📊 Download {asset_class} Greeks Summary",
                            json.dumps(greeks_export, indent=2),
                            f"{greeks_result['underlying_ticker']}_{asset_class}_greeks_summary.json",
                            "application/json"
                        )
                    
                    with col2:
                        if not all_greeks_df.empty:
                            csv_data = all_greeks_df.to_csv(index=False)
                            st.download_button(
                                f"📋 Download {asset_class} Greeks CSV",
                                csv_data,
                                f"{greeks_result['underlying_ticker']}_{asset_class}_greeks_full.csv",
                                "text/csv"
                            )
                
                except Exception as e:
                    st.error(f"❌ {asset_class} Greeks analysis failed: {str(e)}")
                    # Show error details in expander
                    with st.expander("Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
        
        else:
            # Asset-specific Greeks instructions
            greeks_instructions = {
                'FOREX': """
                ## 💱 FX Options Greeks
                
                **Currency options Greeks** have unique characteristics due to 24/5 trading and different volatility patterns.
                
                ### 🔍 **FX Greeks Considerations:**
                - **Delta**: Currency pair sensitivity to rate moves
                - **Gamma**: More stable due to central bank intervention
                - **Theta**: 24/5 decay vs traditional market hours
                - **Vega**: Central bank policy affects volatility
                
                ### 💡 **FX Greeks Applications:**
                - **Delta hedging**: Currency exposure management
                - **Vega trading**: Volatility around central bank meetings
                - **Theta strategies**: Time decay in range-bound pairs
                """,
                
                'INDICES': """
                ## 📊 Index Options Greeks
                
                **Index options Greeks** reflect broader market dynamics and diversification benefits.
                
                ### 🔍 **Index Greeks Characteristics:**
                - **Lower Gamma**: More stable due to diversification
                - **Predictable Theta**: Time decay patterns
                - **Correlation Effects**: Sector rotation impacts
                - **Volatility Clustering**: Market regime changes
                
                ### 💡 **Index Greeks Applications:**
                - **Portfolio hedging**: Protective put Greeks
                - **Income strategies**: Covered call Greeks management
                - **Market timing**: Volatility regime analysis
                """,
                
                'EQUITIES': """
                ## 📈 Equity Options Greeks
                
                **Individual stock Greeks** capture company-specific risk and earnings volatility.
                
                ### 🔍 **Equity Greeks Features:**
                - **Higher Gamma**: More sensitive to moves
                - **Earnings Impact**: Volatility spikes affect all Greeks
                - **Single-Name Risk**: Concentrated exposure
                - **Event-Driven**: News and announcements matter
                
                ### 💡 **Equity Greeks Applications:**
                - **Earnings plays**: Gamma and Vega strategies
                - **Risk management**: Position sizing with Greeks
                - **Volatility trading**: Event-driven opportunities
                """
            }
            
            st.markdown(greeks_instructions[asset_class])
    
    # Tab 4: Backtester (Multi-Asset)
    with tab4:
        st.header(f"📈 {asset_class} Strategy Backtester")
        
        # Get debug mode from sidebar
        debug_mode = st.session_state.get('debug_mode', False)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Auto-populate with symbol from analysis tab
            default_symbol = {
                'INDICES': 'SPY',
                'EQUITIES': 'AAPL', 
                'FOREX': 'EURUSD'
            }[asset_class]
            
            if (st.session_state.analysis_result and 
                st.session_state.analysis_result.get('success') and
                st.session_state.analysis_result.get('asset_class') == asset_class):
                default_symbol = st.session_state.analysis_result['ticker']
            
            backtest_symbol = st.text_input(
                f"Symbol for Backtesting ({asset_class})",
                value=default_symbol,
                help=f"Enter {asset_class.lower()} ticker symbol"
            )
            
            # Strategy selection with asset-specific defaults
            available_strategies = [
                'COVERED_CALL',
                'CASH_SECURED_PUT', 
                'IRON_CONDOR',
                'BULL_CALL_SPREAD',
                'BEAR_PUT_SPREAD',
                'BUY_AND_HOLD'
            ]
            
            # Asset-specific strategy recommendations
            if asset_class == 'FOREX':
                recommended_strategy = 'IRON_CONDOR'
            elif asset_class == 'INDICES':
                recommended_strategy = 'COVERED_CALL'
            else:
                recommended_strategy = 'BULL_CALL_SPREAD'
            
            # Auto-select best strategy if available
            if (st.session_state.analysis_result and 
                st.session_state.analysis_result.get('success') and
                st.session_state.analysis_result.get('asset_class') == asset_class):
                recommended_strategy = st.session_state.analysis_result.get('best_strategy', recommended_strategy)
                if st.session_state.analysis_result['ticker'] == backtest_symbol:
                    st.info(f"🔗 Auto-selected best {asset_class} strategy: '{recommended_strategy}'")
            
            selected_strategy = st.selectbox(
                f"Strategy to Backtest ({asset_class})",
                available_strategies,
                index=available_strategies.index(recommended_strategy) if recommended_strategy in available_strategies else 0
            )
            
            # Date range
            col1a, col1b = st.columns(2)
            with col1a:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365),
                    max_value=datetime.now() - timedelta(days=30)
                )
            with col1b:
                end_date = st.date_input(
                    "End Date", 
                    value=datetime.now() - timedelta(days=1),
                    max_value=datetime.now()
                )
        
        with col2:
            st.markdown(f"### {asset_class} Parameters")
            
            # Asset-specific parameters
            params = {}
            
            if asset_class == 'FOREX':
                st.info("💱 FX-optimized parameters")
                params['days_to_expiry'] = st.slider("Days to Expiry", 15, 60, 21)  # Shorter for FX
                params['delta_target'] = st.slider("Delta Target", 0.15, 0.35, 0.25, 0.05)  # Lower for FX
            else:
                params['days_to_expiry'] = st.slider("Days to Expiry", 15, 60, 30)
                params['delta_target'] = st.slider("Delta Target", 0.1, 0.5, 0.3, 0.05)
            
            if selected_strategy == 'IRON_CONDOR':
                if asset_class == 'FOREX':
                    params['wing_width'] = st.slider("Wing Width %", 2, 8, 4) / 100  # Tighter for FX
                else:
                    params['wing_width'] = st.slider("Wing Width %", 3, 10, 5) / 100
            
            run_backtest_button = st.button(
                f"🚀 Run {asset_class} Backtest",
                type="primary",
                disabled=not backtest_symbol or not polygon_key
            )
        
        if run_backtest_button and backtest_symbol:
            with st.spinner(f"Running {selected_strategy} backtest on {backtest_symbol} ({asset_class})..."):
                try:
                    backtest_result = strategist.backtest_strategy(
                        backtest_symbol.upper(),
                        asset_class,
                        selected_strategy,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        params
                    )
                    
                    # Debug information
                    if debug_mode:
                        st.write("**Debug - Backtest Result:**")
                        st.write(f"Type: {type(backtest_result)}")
                        st.write(f"Content: {backtest_result}")
                    
                    # Check if result is valid
                    if backtest_result is None:
                        st.error("❌ Backtest returned None - method execution failed")
                        return
                    
                    if not isinstance(backtest_result, dict):
                        st.error(f"❌ Backtest returned unexpected type: {type(backtest_result)}")
                        return
                    
                    # Handle the response
                    if backtest_result.get('success'):
                        # Store result in session state
                        st.session_state.backtest_result = backtest_result
                        
                        st.success(f"✅ {asset_class} backtest completed for {backtest_result['strategy']} on {backtest_result['ticker']}")
                        
                        # Show connection to analysis tab if same symbol
                        if (st.session_state.analysis_result and 
                            st.session_state.analysis_result.get('success') and
                            st.session_state.analysis_result['ticker'] == backtest_result['ticker'] and
                            st.session_state.analysis_result.get('asset_class') == asset_class):
                            st.info(f"🔗 This {asset_class} backtest matches your symbol from the Analysis tab!")
                        
                        # Performance Summary
                        st.subheader("📊 Performance Summary")
                        
                        perf = backtest_result['performance']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total P&L", f"${perf['total_pnl']:.2f}")
                            st.metric("Total Trades", perf['total_trades'])
                        
                        with col2:
                            st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
                            st.metric("Avg Return/Trade", f"{perf['avg_return_per_trade']:.2f}%")
                        
                        with col3:
                            st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
                            st.metric("Max Drawdown", f"{perf['max_drawdown']:.2f}%")
                        
                        with col4:
                            st.metric("vs Buy & Hold", f"{perf['alpha']:.2f}%")
                            st.metric("Volatility", f"{perf['volatility']:.2f}%")
                        
                        # Detailed Performance
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📈 Risk/Return Metrics")
                            st.write(f"**Winning Trades:** {perf['winning_trades']}")
                            st.write(f"**Losing Trades:** {perf['losing_trades']}")
                            st.write(f"**Average Win:** ${perf['avg_win']:.2f}")
                            st.write(f"**Average Loss:** ${perf['avg_loss']:.2f}")
                            profit_factor = abs(perf['avg_win'] / perf['avg_loss']) if perf['avg_loss'] != 0 else "N/A"
                            st.write(f"**Profit Factor:** {profit_factor:.2f}" if isinstance(profit_factor, float) else f"**Profit Factor:** {profit_factor}")
                        
                        with col2:
                            st.markdown("### 📅 Time Analysis")
                            st.write(f"**Backtest Period:** {perf['total_days']} days")
                            st.write(f"**Benchmark Return:** {perf['benchmark_return']:.2f}%")
                            st.write(f"**Strategy Alpha:** {perf['alpha']:.2f}%")
                            
                            # Performance classification
                            if perf['sharpe_ratio'] > 1.5:
                                perf_rating = "🏆 Excellent"
                            elif perf['sharpe_ratio'] > 1.0:
                                perf_rating = "✅ Good" 
                            elif perf['sharpe_ratio'] > 0.5:
                                perf_rating = "⚠️ Fair"
                            else:
                                perf_rating = "❌ Poor"
                            
                            st.write(f"**Performance Rating:** {perf_rating}")
                        
                        # Equity Curve Chart
                        st.subheader("📈 Equity Curve")
                        
                        equity_data = backtest_result['results']['equity_curve']
                        if equity_data:
                            equity_df = pd.DataFrame(equity_data)
                            
                            fig = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=(f'{backtest_result["strategy"]} Performance vs Underlying', 'Underlying Price'),
                                row_heights=[0.7, 0.3]
                            )
                            
                            # Strategy equity curve
                            fig.add_trace(
                                go.Scatter(
                                    x=equity_df['date'],
                                    y=equity_df['portfolio_value'],
                                    mode='lines',
                                    name=f'{backtest_result["strategy"]} P&L',
                                    line=dict(color='#00ff88', width=2)
                                ),
                                row=1, col=1
                            )
                            
                            # Underlying price
                            fig.add_trace(
                                go.Scatter(
                                    x=equity_df['date'],
                                    y=equity_df['underlying_price'],
                                    mode='lines',
                                    name='Underlying Price',
                                    line=dict(color='#ff6b35', width=2)
                                ),
                                row=2, col=1
                            )
                            
                            fig.update_layout(
                                height=600,
                                title=f'{backtest_result["strategy"]} Backtest Results - {backtest_result["ticker"]} ({asset_class})',
                                template='plotly_dark'
                            )
                            
                            fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
                            fig.update_yaxes(title_text="Price", row=2, col=1)
                            fig.update_xaxes(title_text="Date", row=2, col=1)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Trade Analysis
                        trades = backtest_result['results']['trades']
                        if trades:
                            st.subheader("📋 Trade Analysis")
                            
                            trades_df = pd.DataFrame(trades)
                            
                            # Format for display
                            display_trades = trades_df.copy()
                            display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
                            display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
                            display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f"${x:.2f}")
                            display_trades['return_pct'] = display_trades['return_pct'].apply(lambda x: f"{x:.2f}%")
                            
                            # Show last 10 trades
                            st.write("**Recent Trades:**")
                            st.dataframe(display_trades.tail(10), use_container_width=True)
                            
                            # Trade distribution
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # P&L distribution
                                fig_hist = go.Figure(data=[
                                    go.Histogram(
                                        x=trades_df['return_pct'],
                                        nbinsx=20,
                                        name='Return Distribution',
                                        marker_color='lightblue'
                                    )
                                ])
                                fig_hist.update_layout(
                                    title='Trade Return Distribution',
                                    xaxis_title='Return %',
                                    yaxis_title='Frequency',
                                    template='plotly_dark'
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            with col2:
                                # Monthly returns
                                trades_df['entry_month'] = pd.to_datetime(trades_df['entry_date']).dt.to_period('M')
                                monthly_returns = trades_df.groupby('entry_month')['return_pct'].mean()
                                
                                fig_monthly = go.Figure(data=[
                                    go.Bar(
                                        x=[str(m) for m in monthly_returns.index],
                                        y=monthly_returns.values,
                                        name='Monthly Avg Returns',
                                        marker_color='lightgreen'
                                    )
                                ])
                                fig_monthly.update_layout(
                                    title='Average Monthly Returns',
                                    xaxis_title='Month',
                                    yaxis_title='Return %',
                                    template='plotly_dark'
                                )
                                st.plotly_chart(fig_monthly, use_container_width=True)
                        
                        # Export Results
                        st.subheader("📤 Export Results")
                        
                        backtest_export = {
                            'backtest_timestamp': datetime.now().isoformat(),
                            'asset_class': asset_class,
                            'strategy': backtest_result['strategy'],
                            'ticker': backtest_result['ticker'],
                            'parameters': backtest_result['parameters'],
                            'performance_metrics': backtest_result['performance'],
                            'trade_count': len(trades) if trades else 0
                        }
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                f"📊 Download {asset_class} Summary",
                                json.dumps(backtest_export, indent=2),
                                f"{backtest_result['ticker']}_{asset_class}_{backtest_result['strategy']}_backtest.json",
                                "application/json"
                            )
                        
                        with col2:
                            if trades:
                                trades_csv = pd.DataFrame(trades).to_csv(index=False)
                                st.download_button(
                                    f"📋 Download {asset_class} Trades CSV",
                                    trades_csv,
                                    f"{backtest_result['ticker']}_{asset_class}_{backtest_result['strategy']}_trades.csv",
                                    "text/csv"
                                )
                    else:
                        st.error(f"❌ {asset_class} backtest failed: {backtest_result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    st.error(f"❌ {asset_class} backtest execution failed: {str(e)}")
                    with st.expander("Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
        
        else:
            # Asset-specific backtesting instructions
            backtest_instructions = {
                'FOREX': """
                ## 💱 FX Options Backtesting
                
                **Test currency strategies** with 24/5 market considerations and central bank event impacts.
                
                ### 🎯 **FX Backtesting Features:**
                - **24/5 Time Decay**: Continuous theta calculations
                - **Central Bank Events**: Volatility spike modeling
                - **Correlation Analysis**: Multi-pair portfolio effects
                - **Intervention Risk**: Central bank policy impacts
                
                ### 💡 **Best FX Strategies to Test:**
                - **Iron Condors**: Range-bound currency pairs
                - **Straddles**: Around central bank meetings
                - **Bull/Bear Spreads**: Directional macro trades
                """,
                
                'INDICES': """
                ## 📊 Index Options Backtesting
                
                **Validate diversified strategies** with broad market exposure and reduced single-name risk.
                
                ### 🎯 **Index Backtesting Features:**
                - **Market Regime Analysis**: Bull/bear/sideways periods
                - **Volatility Clustering**: VIX relationship modeling
                - **Sector Rotation**: Style factor considerations
                - **Portfolio Effects**: Correlation with holdings
                
                ### 💡 **Best Index Strategies to Test:**
                - **Covered Calls**: Income on diversified holdings
                - **Protective Puts**: Portfolio insurance
                - **Iron Condors**: Market stability plays
                """,
                
                'EQUITIES': """
                ## 📈 Equity Options Backtesting
                
                **Test company-specific strategies** with earnings events and fundamental catalysts.
                
                ### 🎯 **Equity Backtesting Features:**
                - **Earnings Impact**: Quarterly volatility modeling
                - **Event-Driven**: News and announcement effects
                - **Single-Name Risk**: Concentrated exposure analysis
                - **Sector Analysis**: Industry-specific patterns
                
                ### 💡 **Best Equity Strategies to Test:**
                - **Straddles**: Earnings and event plays
                - **Bull/Bear Spreads**: Directional company bets
                - **Covered Calls**: Income on stock holdings
                """
            }
            
            st.markdown(backtest_instructions[asset_class])
    
    # Tab 5: Market Predictions (Multi-Asset) - COMPLETE IMPLEMENTATION
    with tab5:
        st.header(f"🔮 {asset_class} Market Predictions")
        
        # Get debug mode from sidebar
        debug_mode = st.session_state.get('debug_mode', False)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Auto-populate with symbol from analysis tab
            default_symbol = {
                'INDICES': 'SPY',
                'EQUITIES': 'AAPL', 
                'FOREX': 'EURUSD'
            }[asset_class]
            
            analysis_match = False
            if (st.session_state.analysis_result and 
                st.session_state.analysis_result.get('success') and
                st.session_state.analysis_result.get('asset_class') == asset_class):
                default_symbol = st.session_state.analysis_result['ticker']
                analysis_match = True
            
            prediction_symbol = st.text_input(
                f"Symbol for {asset_class} Prediction",
                value=default_symbol,
                help=f"Enter {asset_class.lower()} ticker symbol"
            )
            
            if analysis_match and prediction_symbol.upper() == default_symbol:
                st.info(f"🔗 Using same {asset_class} symbol from Analysis tab for enhanced integration!")
        
        with col2:
            # Asset-specific prediction horizons
            if asset_class == 'FOREX':
                default_days = 21  # Shorter for FX
                max_days = 60
            else:
                default_days = 30
                max_days = 90
            
            prediction_days = st.slider(
                f"{asset_class} Prediction Horizon (Days)",
                7, max_days, default_days,
                help=f"Prediction period optimized for {asset_class.lower()}"
            )
            
            get_prediction_button = st.button(
                f"🔮 Generate {asset_class} Prediction",
                type="primary",
                disabled=not prediction_symbol or not polygon_key
            )
        
        if get_prediction_button and prediction_symbol:
            with st.spinner(f"Generating {asset_class} market predictions for {prediction_symbol}..."):
                try:
                    prediction_result = strategist.predict_market_direction(
                        prediction_symbol.upper(), 
                        asset_class,
                        prediction_days
                    )
                    
                    # Debug information
                    if debug_mode:
                        st.write("**Debug - Prediction Result:**")
                        st.write(f"Type: {type(prediction_result)}")
                        st.write(f"Keys: {list(prediction_result.keys()) if isinstance(prediction_result, dict) else 'Not a dict'}")
                    
                    # Check if result is valid
                    if prediction_result is None:
                        st.error("❌ Prediction returned None - method execution failed")
                    elif not isinstance(prediction_result, dict):
                        st.error(f"❌ Prediction returned unexpected type: {type(prediction_result)}")
                    elif not prediction_result.get('success'):
                        st.error(f"❌ {asset_class} prediction failed: {prediction_result.get('error', 'Unknown error')}")
                    else:
                        # SUCCESS - Display the complete prediction results
                        st.session_state.prediction_result = prediction_result
                        
                        st.success(f"✅ {asset_class} prediction analysis completed for {prediction_result['ticker']}")
                        
                        # Show connection to analysis tab if same symbol
                        if (st.session_state.analysis_result and 
                            st.session_state.analysis_result.get('success') and
                            st.session_state.analysis_result['ticker'] == prediction_result['ticker'] and
                            st.session_state.analysis_result.get('asset_class') == asset_class):
                            st.info(f"🔗 This {asset_class} prediction matches your symbol from the Analysis tab!")
                        
                        # Asset-specific insights
                        if asset_class == 'FOREX':
                            st.info("💱 FX predictions include central bank policy impacts and 24/5 market dynamics")
                        elif asset_class == 'INDICES':
                            st.info("📊 Index predictions reflect broad market trends and sector rotation patterns")
                        else:
                            st.info("📈 Equity predictions capture company-specific catalysts and earnings expectations")
                        
                        # MAIN PREDICTION SUMMARY
                        st.subheader("🎯 Overall Market Prediction")
                        
                        overall = prediction_result['overall_prediction']
                        current_price = prediction_result['current_price']
                        
                        # Asset-specific price formatting
                        if asset_class == 'FOREX':
                            price_display = f"{current_price:.5f}"
                            price_label = "Current Rate"
                        else:
                            price_display = f"${current_price:.2f}"
                            price_label = "Current Price"
                        
                        # Main prediction display
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(price_label, price_display)
                            st.metric("Prediction Period", f"{prediction_result['prediction_period']} days")
                        
                        with col2:
                            direction_emoji = {
                                'BULLISH': '🟢',
                                'BEARISH': '🔴', 
                                'NEUTRAL': '🟡'
                            }.get(overall['direction'], '🟡')
                            st.metric("Direction", f"{direction_emoji} {overall['direction']}")
                            st.metric("Confidence", f"{overall['confidence']:.0%}")
                        
                        with col3:
                            st.metric("Strength", overall['strength'])
                            st.metric("Time Horizon", overall['time_horizon'].replace('_', ' ').title())
                        
                        with col4:
                            st.metric("Bullish Score", f"{overall['bullish_score']}/10")
                            st.metric("Bearish Score", f"{overall['bearish_score']}/10")
                        
                        # Prediction Summary
                        st.markdown(f"**📝 Summary:** {overall['summary']}")
                        
                        # Key Risks
                        if overall.get('key_risks'):
                            st.markdown("**⚠️ Key Risks:**")
                            for risk in overall['key_risks']:
                                st.markdown(f"• {risk}")
                        
                        # TECHNICAL ANALYSIS SECTION
                        st.subheader("📊 Technical Analysis Signals")
                        
                        technical = prediction_result['technical_signals']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**RSI Analysis**")
                            rsi_color = {
                                'BULLISH': '🟢',
                                'BEARISH': '🔴',
                                'NEUTRAL': '🟡'
                            }.get(technical['rsi_signal'], '🟡')
                            st.write(f"{rsi_color} **Signal:** {technical['rsi_signal']}")
                            st.write(f"📊 **Value:** {technical['rsi_value']}")
                            st.write(f"💪 **Strength:** {technical['rsi_strength']:.2f}")
                        
                        with col2:
                            st.markdown("**Moving Averages**")
                            ma_signals = technical['moving_average_signals']
                            if ma_signals:
                                for signal in ma_signals:
                                    st.write(f"✅ {signal.replace('_', ' ').title()}")
                            else:
                                st.write("🔴 Below Key Moving Averages")
                            
                            st.write(f"📈 **vs SMA20:** {technical['price_vs_sma20']:.2f}%")
                            st.write(f"📈 **vs SMA50:** {technical['price_vs_sma50']:.2f}%")
                        
                        with col3:
                            st.markdown("**MACD Signal**")
                            macd_color = {
                                'BULLISH': '🟢',
                                'BEARISH': '🔴',
                                'NEUTRAL': '🟡'
                            }.get(technical['macd_signal'], '🟡')
                            st.write(f"{macd_color} **Signal:** {technical['macd_signal']}")
                            st.write(f"📊 **Value:** {technical['macd_value']:.4f}")
                            st.write(f"📈 **vs SMA200:** {technical['price_vs_sma200']:.2f}%")
                        
                        # VOLATILITY FORECAST SECTION
                        st.subheader("📈 Volatility Forecast")
                        
                        vol_forecast = prediction_result['volatility_forecast']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Vol (21d)", f"{vol_forecast['current_vol_21d']:.1%}")
                            st.metric("Long-term Vol", f"{vol_forecast['long_term_vol']:.1%}")
                        
                        with col2:
                            st.metric("Forecast Vol", f"{vol_forecast['forecast_vol']:.1%}")
                            st.metric("Vol Trend", vol_forecast['vol_trend'])
                        
                        with col3:
                            regime_color = {
                                'HIGH_VOLATILITY': '🔴',
                                'LOW_VOLATILITY': '🟢',
                                'NORMAL_VOLATILITY': '🟡'
                            }.get(vol_forecast['volatility_regime'], '🟡')
                            st.metric("Vol Regime", f"{regime_color} {vol_forecast['volatility_regime']}")
                            st.metric("Regime Confidence", f"{vol_forecast['regime_confidence']:.0%}")
                        
                        with col4:
                            st.metric("Vol (10d)", f"{vol_forecast['current_vol_10d']:.1%}")
                            st.metric("Vol (63d)", f"{vol_forecast['current_vol_63d']:.1%}")
                        
                        # SUPPORT & RESISTANCE SECTION
                        st.subheader("🎯 Support & Resistance Levels")
                        
                        support_resistance = prediction_result['support_resistance']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Resistance Levels**")
                            resistance_levels = support_resistance['resistance_levels']
                            if resistance_levels:
                                for i, level in enumerate(resistance_levels[:3], 1):
                                    if asset_class == 'FOREX':
                                        st.write(f"🔴 **R{i}:** {level:.5f}")
                                    else:
                                        st.write(f"🔴 **R{i}:** ${level:.2f}")
                            else:
                                st.write("No clear resistance levels")
                        
                        with col2:
                            st.markdown("**Support Levels**")
                            support_levels = support_resistance['support_levels']
                            if support_levels:
                                for i, level in enumerate(support_levels[:3], 1):
                                    if asset_class == 'FOREX':
                                        st.write(f"🟢 **S{i}:** {level:.5f}")
                                    else:
                                        st.write(f"🟢 **S{i}:** ${level:.2f}")
                            else:
                                st.write("No clear support levels")
                        
                        with col3:
                            st.markdown("**52-Week Range**")
                            high_52w = support_resistance['52_week_high']
                            low_52w = support_resistance['52_week_low']
                            
                            if asset_class == 'FOREX':
                                st.write(f"📈 **High:** {high_52w:.5f}")
                                st.write(f"📉 **Low:** {low_52w:.5f}")
                            else:
                                st.write(f"📈 **High:** ${high_52w:.2f}")
                                st.write(f"📉 **Low:** ${low_52w:.2f}")
                            
                            st.write(f"⬆️ **To High:** {support_resistance['distance_to_52w_high']:.1f}%")
                            st.write(f"⬇️ **From Low:** {support_resistance['distance_to_52w_low']:.1f}%")
                        
                        # FIBONACCI RETRACEMENTS
                        with st.expander("📐 Fibonacci Retracement Levels"):
                            fib_levels = support_resistance['fibonacci_levels']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                for level, price in list(fib_levels.items())[:3]:
                                    if asset_class == 'FOREX':
                                        st.write(f"**{level}:** {price:.5f}")
                                    else:
                                        st.write(f"**{level}:** ${price:.2f}")
                            
                            with col2:
                                for level, price in list(fib_levels.items())[3:]:
                                    if asset_class == 'FOREX':
                                        st.write(f"**{level}:** {price:.5f}")
                                    else:
                                        st.write(f"**{level}:** ${price:.2f}")
                        
                        # PRICE TARGETS SECTION
                        st.subheader("🎯 Price Targets & Projections")
                        
                        price_targets = prediction_result['price_targets']
                        
                        # Bullish vs Bearish targets based on overall prediction
                        if overall['direction'] == 'BULLISH':
                            targets = price_targets['bullish_targets']
                            target_type = "Bullish"
                            target_color = "🟢"
                        elif overall['direction'] == 'BEARISH':
                            targets = price_targets['bearish_targets']
                            target_type = "Bearish"
                            target_color = "🔴"
                        else:
                            targets = price_targets['bullish_targets']  # Default to bullish
                            target_type = "Neutral"
                            target_color = "🟡"
                        
                        st.markdown(f"**{target_color} {target_type} Price Targets:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if asset_class == 'FOREX':
                                st.metric("Target 1", f"{targets['target_1']:.5f}")
                            else:
                                st.metric("Target 1", f"${targets['target_1']:.2f}")
                            st.write(f"📊 Probability: {targets['probabilities'][0]:.0%}")
                        
                        with col2:
                            if asset_class == 'FOREX':
                                st.metric("Target 2", f"{targets['target_2']:.5f}")
                            else:
                                st.metric("Target 2", f"${targets['target_2']:.2f}")
                            st.write(f"📊 Probability: {targets['probabilities'][1]:.0%}")
                        
                        with col3:
                            if asset_class == 'FOREX':
                                st.metric("Target 3", f"{targets['target_3']:.5f}")
                            else:
                                st.metric("Target 3", f"${targets['target_3']:.2f}")
                            st.write(f"📊 Probability: {targets['probabilities'][2]:.0%}")
                        
                        with col4:
                            if asset_class == 'FOREX':
                                st.metric("Stop Loss", f"{price_targets['suggested_stop_loss']:.5f}")
                            else:
                                st.metric("Stop Loss", f"${price_targets['suggested_stop_loss']:.2f}")
                            st.write(f"📊 ATR: {price_targets['atr_value']:.4f}")
                        
                        # Bollinger Bands
                        st.markdown("**📊 Bollinger Bands:**")
                        bb = price_targets['bollinger_bands']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if asset_class == 'FOREX':
                                st.metric("BB Upper", f"{bb['upper']:.5f}")
                            else:
                                st.metric("BB Upper", f"${bb['upper']:.2f}")
                        
                        with col2:
                            if asset_class == 'FOREX':
                                st.metric("BB Middle", f"{bb['middle']:.5f}")
                            else:
                                st.metric("BB Middle", f"${bb['middle']:.2f}")
                        
                        with col3:
                            if asset_class == 'FOREX':
                                st.metric("BB Lower", f"{bb['lower']:.5f}")
                            else:
                                st.metric("BB Lower", f"${bb['lower']:.2f}")
                        
                        # MOMENTUM ANALYSIS SECTION
                        st.subheader("⚡ Momentum Analysis")
                        
                        momentum = prediction_result['momentum_analysis']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            momentum_color = {
                                'STRONG_BULLISH': '🟢',
                                'BULLISH': '🟢',
                                'STRONG_BEARISH': '🔴',
                                'BEARISH': '🔴',
                                'NEUTRAL': '🟡'
                            }.get(momentum['momentum_direction'], '🟡')
                            
                            st.metric("Direction", f"{momentum_color} {momentum['momentum_direction']}")
                            st.metric("Strength", f"{momentum['momentum_strength']:.2f}")
                        
                        with col2:
                            st.metric("5-Day ROC", f"{momentum['roc_5_day']:.2f}%")
                            st.metric("10-Day ROC", f"{momentum['roc_10_day']:.2f}%")
                            st.metric("20-Day ROC", f"{momentum['roc_20_day']:.2f}%")
                        
                        with col3:
                            if asset_class != 'FOREX':
                                volume_color = {
                                    'HIGH': '🟢',
                                    'LOW': '🔴',
                                    'NORMAL': '🟡'
                                }.get(momentum['volume_signal'], '🟡')
                                
                                st.metric("Volume Signal", f"{volume_color} {momentum['volume_signal']}")
                                st.metric("Volume Ratio", f"{momentum['volume_ratio']:.2f}x")
                            else:
                                st.metric("Market Type", "💱 24/5 FX")
                                st.metric("Volume Analysis", "N/A for FX")
                        
                        # EXPORT PREDICTIONS
                        st.subheader("📤 Export Prediction Data")
                        
                        prediction_export = {
                            'prediction_timestamp': datetime.now().isoformat(),
                            'asset_class': asset_class,
                            'ticker': prediction_result['ticker'],
                            'prediction_period': prediction_result['prediction_period'],
                            'current_price': prediction_result['current_price'],
                            'overall_prediction': prediction_result['overall_prediction'],
                            'technical_signals': prediction_result['technical_signals'],
                            'volatility_forecast': prediction_result['volatility_forecast'],
                            'price_targets': prediction_result['price_targets'],
                            'momentum_analysis': prediction_result['momentum_analysis']
                        }
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                f"📊 Download {asset_class} Prediction",
                                json.dumps(prediction_export, indent=2),
                                f"{prediction_result['ticker']}_{asset_class}_prediction.json",
                                "application/json"
                            )
                        
                        with col2:
                            # Create a summary CSV
                            summary_data = {
                                'Ticker': [prediction_result['ticker']],
                                'Asset_Class': [asset_class],
                                'Current_Price': [prediction_result['current_price']],
                                'Direction': [overall['direction']],
                                'Confidence': [overall['confidence']],
                                'Strength': [overall['strength']],
                                'Time_Horizon': [overall['time_horizon']],
                                'Target_1': [targets['target_1']],
                                'Target_2': [targets['target_2']],
                                'Target_3': [targets['target_3']],
                                'Stop_Loss': [price_targets['suggested_stop_loss']],
                                'RSI': [technical['rsi_value']],
                                'Vol_Regime': [vol_forecast['volatility_regime']],
                                'Momentum': [momentum['momentum_direction']]
                            }
                            
                            summary_df = pd.DataFrame(summary_data)
                            summary_csv = summary_df.to_csv(index=False)
                            
                            st.download_button(
                                f"📋 Download {asset_class} Summary CSV",
                                summary_csv,
                                f"{prediction_result['ticker']}_{asset_class}_prediction_summary.csv",
                                "text/csv"
                            )
                    
                except Exception as e:
                    st.error(f"❌ {asset_class} prediction execution failed: {str(e)}")
                    with st.expander("Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
        
        else:
            # Asset-specific prediction instructions when no prediction is generated
            prediction_instructions = {
                'FOREX': """
                ## 💱 FX Market Predictions
                
                **AI-powered currency forecasting** using central bank analysis, economic data, and technical patterns.
                
                ### 🎯 **FX Prediction Components:**
                - **Central Bank Analysis**: Policy expectations and intervention risk
                - **Economic Data**: GDP, inflation, employment impacts
                - **Technical Levels**: Key support/resistance in major pairs
                - **Carry Trade Analysis**: Interest rate differential trends
                
                ### 📊 **FX-Specific Indicators:**
                - **Interest Rate Differentials**: Carry trade opportunities
                - **Economic Calendar**: High-impact data releases
                - **Central Bank Communications**: Hawkish/dovish signals
                - **Risk Sentiment**: Safe haven vs risk currency flows
                
                ### 💡 **FX Prediction Applications:**
                - **Directional Trades**: Major trend identification
                - **Event Trading**: Central bank meeting strategies
                - **Range Trading**: Support/resistance levels
                - **Volatility Plays**: Event-driven straddles
                """,
                
                'INDICES': """
                ## 📊 Index Market Predictions
                
                **Broad market forecasting** using economic cycles, sector rotation, and volatility analysis.
                
                ### 🎯 **Index Prediction Components:**
                - **Economic Cycle Analysis**: Growth, inflation, policy phases
                - **Sector Rotation**: Style and factor analysis
                - **Volatility Forecasting**: VIX and regime changes
                - **Technical Analysis**: Major support/resistance levels
                
                ### 📊 **Index-Specific Indicators:**
                - **Market Breadth**: Advance/decline patterns
                - **Volatility Regime**: Low/normal/high vol environments
                - **Sector Leadership**: Growth vs value rotations
                - **Economic Indicators**: Leading vs lagging signals
                
                ### 💡 **Index Prediction Applications:**
                - **Portfolio Allocation**: Sector and style timing
                - **Hedging Strategies**: Portfolio protection timing
                - **Income Strategies**: Covered call timing
                - **Volatility Trading**: VIX-based strategies
                """,
                
                'EQUITIES': """
                ## 📈 Equity Market Predictions
                
                **Company-specific forecasting** using fundamental analysis, earnings expectations, and technical patterns.
                
                ### 🎯 **Equity Prediction Components:**
                - **Fundamental Analysis**: Valuation and growth metrics
                - **Earnings Expectations**: Consensus vs reality analysis
                - **Technical Patterns**: Company-specific chart analysis
                - **Sector Analysis**: Industry trends and comparisons
                
                ### 📊 **Equity-Specific Indicators:**
                - **Earnings Revisions**: Analyst estimate changes
                - **Valuation Metrics**: P/E, PEG, price-to-sales ratios
                - **Institutional Flow**: Smart money positioning
                - **Event Calendar**: Earnings, product launches, FDA approvals
                
                ### 💡 **Equity Prediction Applications:**
                - **Earnings Plays**: Volatility and direction strategies
                - **Event Trading**: Catalyst-driven opportunities
                - **Value Identification**: Undervalued company targeting
                - **Growth Momentum**: Trend continuation strategies
                """
            }
            
            st.markdown(prediction_instructions[asset_class])
        


if __name__ == "__main__":
    main()