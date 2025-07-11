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
# POLYGON SDK REAL DATA ONLY - FTSE Options Strategist
# =============================================================================

class PolygonRealDataStrategist:
    """
    Real data only Options Strategist using Polygon Python SDK
    No fallbacks - requires actual market data
    """
    
    def __init__(self, polygon_api_key: str):
        if not polygon_api_key:
            raise ValueError("Polygon API key is required - no fallback data")
        
        self.client = RESTClient(polygon_api_key)
        self.polygon_api_key = polygon_api_key
        
        # Cache for available indices
        self._available_indices = None
        self._available_stocks = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for debugging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_available_indices(self) -> List[Dict]:
        """Get all available indices from Polygon"""
        if self._available_indices is None:
            try:
                print("🔍 Fetching available indices from Polygon...")
                indices = []
                for ticker in self.client.list_tickers(
                    market="indices",
                    active=True,
                    limit=1000,
                    sort="ticker"
                ):
                    indices.append({
                        'ticker': ticker.ticker,
                        'name': ticker.name,
                        'market': getattr(ticker, 'market', 'indices'),
                        'locale': getattr(ticker, 'locale', 'unknown'),
                        'currency_name': getattr(ticker, 'currency_name', 'unknown')
                    })
                self._available_indices = indices
                print(f"✅ Found {len(indices)} available indices")
            except Exception as e:
                print(f"❌ Failed to get indices: {str(e)}")
                raise
        
        return self._available_indices
    
    def get_available_stocks(self, market: str = "stocks") -> List[Dict]:
        """Get available stocks/ETFs from Polygon"""
        if self._available_stocks is None:
            try:
                print(f"🔍 Fetching available {market} from Polygon...")
                stocks = []
                for ticker in self.client.list_tickers(
                    market=market,
                    active=True,
                    limit=1000,
                    sort="ticker"
                ):
                    stocks.append({
                        'ticker': ticker.ticker,
                        'name': ticker.name,
                        'market': getattr(ticker, 'market', market),
                        'type': getattr(ticker, 'type', 'unknown'),
                        'locale': getattr(ticker, 'locale', 'unknown')
                    })
                self._available_stocks = stocks
                print(f"✅ Found {len(stocks)} available {market}")
            except Exception as e:
                print(f"❌ Failed to get {market}: {str(e)}")
                raise
        
        return self._available_stocks
    
    def find_ftse_indices(self) -> List[Dict]:
        """Find FTSE-related indices with robust error handling"""
        indices = self.get_available_indices()
        ftse_indices = []
        
        for idx in indices:
            try:
                name = idx.get('name', '')
                ticker = idx.get('ticker', '')
                
                # Handle None values
                if name is None:
                    name = ''
                if ticker is None:
                    ticker = ''
                
                name_lower = name.lower()
                ticker_lower = ticker.lower()
                
                # Look for FTSE-related terms
                if any(term in name_lower or term in ticker_lower for term in [
                    'ftse', 'uk', 'britain', 'london', 'england', 'british'
                ]):
                    ftse_indices.append({
                        'ticker': ticker,
                        'name': name if name else 'Unknown',
                        'locale': idx.get('locale', 'unknown'),
                        'currency': idx.get('currency_name', 'unknown')
                    })
                    
            except Exception as e:
                # Skip problematic entries
                continue
        
        return ftse_indices
    
    def find_ftse_100_specific(self) -> List[Dict]:
        """Find instruments specifically related to FTSE 100"""
        try:
            print("🔍 Searching for FTSE 100 specific instruments...")
            
            # Search both indices and stocks
            all_instruments = []
            
            # Try indices first
            try:
                indices = self.get_available_indices()
                for idx in indices:
                    name = idx.get('name', '') or ''
                    ticker = idx.get('ticker', '') or ''
                    
                    if 'ftse' in name.lower() and '100' in name:
                        all_instruments.append({
                            'ticker': ticker,
                            'name': name,
                            'type': 'INDEX',
                            'market': 'indices'
                        })
            except Exception as e:
                print(f"Could not search indices: {e}")
            
            # Try stocks/ETFs
            try:
                stocks = self.get_available_stocks()
                for stock in stocks:
                    name = stock.get('name', '') or ''
                    ticker = stock.get('ticker', '') or ''
                    
                    if any(term in name.lower() for term in [
                        'ftse 100', 'ftse100', 'uk large', 'united kingdom'
                    ]):
                        all_instruments.append({
                            'ticker': ticker,
                            'name': name,
                            'type': stock.get('type', 'ETF'),
                            'market': 'stocks'
                        })
            except Exception as e:
                print(f"Could not search stocks: {e}")
            
            return all_instruments
            
        except Exception as e:
            print(f"FTSE 100 search failed: {e}")
            return []
    
    def quick_data_check(self, ticker: str) -> Dict:
        """Quick check of data availability without full processing"""
        try:
            st.info(f"🔍 Quick data check for {ticker}...")
            
            # Check recent data availability
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Just check last 30 days
            
            aggs = []
            for agg in self.client.list_aggs(
                ticker,
                1,
                "day",
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                limit=30
            ):
                aggs.append(agg)
            
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
                'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def create_trading_chart(self, data: Dict) -> go.Figure:
        """Create an interactive trading chart with candlesticks and moving averages"""
        try:
            df = data['historical_data'].copy()
            
            # Filter for weekdays only (Monday=0, Sunday=6)
            df = df[df.index.dayofweek < 5]  # Keep Monday-Friday only
            
            # Get last year of data
            one_year_ago = datetime.now() - timedelta(days=365)
            df_last_year = df[df.index >= one_year_ago]
            
            if len(df_last_year) < 50:
                # If not enough data for last year, use all available data
                df_last_year = df
            
            # Create subplot with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{data["ticker"]} - Weekday Trading (Last Year)', 'Volume'),
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
            
            # Moving averages
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
                        row=1, col=1
                    )
            
            # Volume bars
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
            
            # Update layout
            fig.update_layout(
                title=f'{data["ticker"]} - {data.get("name", "Stock")} Trading Chart',
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            # Add current price annotation
            current_price = data['current_price']
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="white",
                annotation_text=f"Current: ${current_price:.2f}",
                annotation_position="bottom right",
                row=1, col=1
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            # Return empty figure if chart creation fails
            return go.Figure().add_annotation(
                text="Chart could not be generated",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def check_options_availability(self, ticker: str) -> Dict:
        """Check if a ticker has options available"""
        try:
            st.info(f"🎯 Checking options availability for {ticker}...")
            
            contracts = list(self.client.list_options_contracts(
                underlying_ticker=ticker,
                limit=10
            ))
            
            if contracts:
                return {
                    'has_options': True,
                    'contract_count': len(contracts),
                    'sample_expiration': getattr(contracts[0], 'expiration_date', 'unknown'),
                    'status': 'Options Available'
                }
            else:
                return {
                    'has_options': False,
                    'contract_count': 0,
                    'status': 'No Options Found'
                }
                
        except Exception as e:
            return {
                'has_options': False,
                'error': str(e),
                'status': 'Error Checking Options'
            }
    
    def find_uk_etfs(self) -> List[Dict]:
        """Find UK-related ETFs with robust error handling"""
        stocks = self.get_available_stocks()
        uk_etfs = []
        
        for stock in stocks:
            try:
                name = stock.get('name', '')
                ticker = stock.get('ticker', '')
                
                # Handle None values
                if name is None:
                    name = ''
                if ticker is None:
                    ticker = ''
                
                name_lower = name.lower()
                ticker_upper = ticker.upper()
                
                # Look for UK-related ETFs
                if any(term in name_lower for term in [
                    'united kingdom', 'uk ', 'britain', 'ftse', 'msci uk', 'british'
                ]) or ticker_upper in ['EWU', 'FKU', 'EWUS']:
                    uk_etfs.append({
                        'ticker': ticker,
                        'name': name if name else 'Unknown',
                        'type': stock.get('type', 'unknown'),
                        'locale': stock.get('locale', 'unknown')
                    })
                    
            except Exception as e:
                # Skip problematic entries
                continue
        
        return uk_etfs
    
    def get_index_data(self, ticker: str, days: int = 252) -> Dict:
        """Get real index data using Polygon SDK with robust error handling"""
        try:
            st.info(f"📊 Fetching real data for {ticker}...")
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            aggs = []
            for agg in self.client.list_aggs(
                ticker,
                1,
                "day",
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                limit=days
            ):
                aggs.append(agg)
            
            if not aggs:
                raise ValueError(f"No historical data found for {ticker}")
            
            # Convert to DataFrame with robust NaN handling
            df_data = []
            for agg in aggs:
                # Handle missing or NaN volume data (indices often don't have volume)
                volume = getattr(agg, 'volume', None)
                if volume is None or pd.isna(volume):
                    volume = 0
                
                # Ensure all price data exists
                open_price = getattr(agg, 'open', None)
                high_price = getattr(agg, 'high', None) 
                low_price = getattr(agg, 'low', None)
                close_price = getattr(agg, 'close', None)
                vwap_price = getattr(agg, 'vwap', None)
                
                # Skip this aggregate if essential price data is missing
                if any(x is None or pd.isna(x) for x in [open_price, high_price, low_price, close_price]):
                    continue
                
                df_data.append({
                    'timestamp': agg.timestamp,
                    'open': float(open_price),
                    'high': float(high_price),
                    'low': float(low_price),
                    'close': float(close_price),
                    'volume': int(volume),
                    'vwap': float(vwap_price) if vwap_price is not None and not pd.isna(vwap_price) else float(close_price)
                })
            
            if not df_data:
                raise ValueError(f"No valid price data found for {ticker}")
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date').sort_index()
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            if len(df) < 21:
                raise ValueError(f"Insufficient clean data for {ticker}: only {len(df)} valid days")
            
            # Calculate technical indicators
            current_price = float(df['close'].iloc[-1])
            tech_data = self._calculate_technical_indicators(df, current_price)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'historical_data': df,
                **tech_data,
                'data_points': len(df),
                'source': 'polygon_real'
            }
            
        except Exception as e:
            st.error(f"❌ Failed to get data for {ticker}: {str(e)}")
            raise
    
    def get_stock_data(self, ticker: str, days: int = 500) -> Dict:
        """Get real stock/ETF data using Polygon SDK with robust error handling"""
        try:
            print(f"📊 Fetching real stock data for {ticker}...")
            
            # Get more historical data to account for weekends/holidays
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            print(f"🔍 Requesting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            aggs = []
            try:
                for agg in self.client.list_aggs(
                    ticker,
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
                raise ValueError(f"No historical data found for {ticker}")
            
            print(f"📈 Received {len(aggs)} raw data points")
            
            # Convert to DataFrame with more lenient NaN handling
            df_data = []
            skipped_records = 0
            
            for agg in aggs:
                # Handle missing or NaN volume data
                volume = getattr(agg, 'volume', None)
                if volume is None or pd.isna(volume):
                    volume = 0  # Allow zero volume
                
                # Get price data
                open_price = getattr(agg, 'open', None)
                high_price = getattr(agg, 'high', None) 
                low_price = getattr(agg, 'low', None)
                close_price = getattr(agg, 'close', None)
                
                # Only skip if ALL price data is missing (be more lenient)
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
                raise ValueError(f"No valid price data found for {ticker}")
            
            print(f"✅ Processing {len(df_data)} valid records")
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date').sort_index()
            
            # Remove any remaining NaN values but be less aggressive
            initial_length = len(df)
            df = df.dropna(subset=['close'])  # Only require close price
            final_length = len(df)
            
            if initial_length != final_length:
                print(f"🧹 Cleaned {initial_length - final_length} NaN records, {final_length} remaining")
            
            if len(df) < 21:
                # Try with a longer time range
                if days < 1000:
                    print(f"Only {len(df)} days available, trying longer time range...")
                    return self.get_stock_data(ticker, days=1000)
                else:
                    raise ValueError(f"Insufficient clean data for {ticker}: only {len(df)} valid days after trying extended range")
            
            # Calculate technical indicators
            current_price = float(df['close'].iloc[-1])
            tech_data = self._calculate_technical_indicators(df, current_price)
            
            print(f"✅ Successfully processed {len(df)} days of data for {ticker}")
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'historical_data': df,
                **tech_data,
                'data_points': len(df),
                'date_range': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                'source': 'polygon_real'
            }
            
        except Exception as e:
            print(f"❌ Failed to get stock data for {ticker}: {str(e)}")
            raise
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate technical indicators from real historical data with NaN handling"""
        # Clean the data first
        df = df.copy()
        
        # Fill NaN values in volume with 0
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        # Ensure we have enough data points
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
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        realized_vol_21d = df['returns'].rolling(21, min_periods=10).std() * np.sqrt(252)
        realized_vol_63d = df['returns'].rolling(63, min_periods=30).std() * np.sqrt(252)
        
        # Get latest values with NaN handling
        latest = df.iloc[-1]
        
        # Helper function to safely extract values
        def safe_float(value, default=0.0):
            if pd.isna(value):
                return default
            return float(value)
        
        def safe_int(value, default=0):
            if pd.isna(value):
                return default
            return int(value)
        
        # Calculate price changes safely
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
        
        # Volume calculations with NaN handling
        avg_volume_20d = df['volume'].rolling(20, min_periods=5).mean().iloc[-1]
        
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
            'volume': safe_int(latest['volume'], 1000000),
            'avg_volume_20d': safe_int(avg_volume_20d, 1000000),
            'high_52w': safe_float(df['high'].rolling(min(252, len(df)), min_periods=50).max().iloc[-1], current_price * 1.25),
            'low_52w': safe_float(df['low'].rolling(min(252, len(df)), min_periods=50).min().iloc[-1], current_price * 0.75),
            'returns_series': df['returns'].dropna(),
            'price_change_1d': safe_price_change(1),
            'price_change_5d': safe_price_change(5),
            'price_change_20d': safe_price_change(20)
        }
    
    def get_options_data(self, underlying_ticker: str, current_price: float = None) -> Dict:
        """Get real options data using Polygon SDK"""
        try:
            print(f"🎯 Fetching real options data for {underlying_ticker}...")
            
            # Get options contracts
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
                # Try to get just the latest price without full historical data
                try:
                    # Get just the most recent day
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=5)  # Just a few days
                    
                    recent_aggs = []
                    for agg in self.client.list_aggs(
                        underlying_ticker,
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
                        print(f"💰 Current price for options: ${current_price:.2f}")
                    else:
                        raise ValueError("Could not get current price")
                        
                except Exception as e:
                    raise ValueError(f"Could not get current price for {underlying_ticker}: {e}")
            
            # Process contracts
            options_data = self._process_real_options_contracts(contracts, current_price, underlying_ticker)
            
            return options_data
            
        except Exception as e:
            print(f"❌ Failed to get options data for {underlying_ticker}: {str(e)}")
            raise
    
    def _process_real_options_contracts(self, contracts: List, current_price: float, underlying_ticker: str) -> Dict:
        """Process real options contracts from Polygon"""
        # Group by expiration
        exp_groups = {}
        today = datetime.now().date()
        
        for contract in contracts:
            try:
                exp_date = contract.expiration_date
                exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d').date()
                
                if exp_date_obj <= today:
                    continue
                
                strike = float(contract.strike_price)
                
                # Filter strikes within reasonable range (±25%)
                if abs(strike - current_price) / current_price > 0.25:
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
                self.logger.warning(f"Error processing contract: {e}")
                continue
        
        if not exp_groups:
            raise ValueError("No valid option contracts found in reasonable strike range")
        
        # Find best expiration (30-45 days ideal)
        best_exp = None
        best_score = 0
        
        for exp_date in sorted(exp_groups.keys()):
            calls_count = len(exp_groups[exp_date]['calls'])
            puts_count = len(exp_groups[exp_date]['puts'])
            
            if calls_count >= 5 and puts_count >= 5:
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_datetime.date() - today).days
                
                # Score based on time to expiration and option availability
                if 14 <= days_to_exp <= 60:
                    time_score = 100 - abs(30 - days_to_exp)  # Prefer ~30 days
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
            raise ValueError("No expiration found with sufficient call and put options")
        
        # Get real option prices for the best expiration
        # Store current price for Black-Scholes calculations
        self._current_underlying_price = current_price
        
        calls_data = self._get_real_option_prices(exp_groups[best_exp]['calls'], underlying_ticker)
        puts_data = self._get_real_option_prices(exp_groups[best_exp]['puts'], underlying_ticker)
        
        # Clean up temporary variable
        if hasattr(self, '_current_underlying_price'):
            delattr(self, '_current_underlying_price')
        
        calls_df = pd.DataFrame(calls_data).sort_values('strike') if calls_data else pd.DataFrame()
        puts_df = pd.DataFrame(puts_data).sort_values('strike') if puts_data else pd.DataFrame()
        
        # Check if we have enough options for analysis
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
    
    def _get_real_option_prices(self, contracts: List[Dict], underlying_ticker: str) -> List[Dict]:
        """Get real option prices from Polygon with fallback to calculated prices"""
        options_data = []
        real_price_count = 0
        calculated_price_count = 0
        
        # Get current volatility estimate for calculations
        try:
            # Simple volatility estimate (would be better to calculate from historical data)
            base_vol = 0.25  # 25% default volatility
        except:
            base_vol = 0.25
        
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
                    # Attempt to get real trade data
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
                    # Expected for free/basic plans - authorization error
                    if "NOT_AUTHORIZED" in str(e):
                        pass  # This is expected, we'll calculate the price
                    else:
                        self.logger.warning(f"Trade data error for {ticker}: {e}")
                
                try:
                    # Attempt to get real quote data
                    if last_price is None:  # Only try if we don't have trade data
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
                    # Expected for free/basic plans - authorization error
                    if "NOT_AUTHORIZED" in str(e):
                        pass  # This is expected, we'll calculate the price
                    else:
                        self.logger.warning(f"Quote data error for {ticker}: {e}")
                
                # If no real price data, calculate using Black-Scholes
                if last_price is None:
                    try:
                        # Get underlying price (passed in from calling function)
                        underlying_price = getattr(self, '_current_underlying_price', 50.0)
                        
                        # Calculate Black-Scholes price
                        calculated_price = self._black_scholes_price(
                            underlying_price, strike, expiration, contract_type, base_vol
                        )
                        
                        if calculated_price > 0.05:  # Only include reasonable prices
                            last_price = calculated_price
                            bid = last_price * 0.95
                            ask = last_price * 1.05
                            data_source = 'calculated'
                            calculated_price_count += 1
                            
                            # Estimate volume based on moneyness
                            moneyness = strike / underlying_price
                            if 0.95 <= moneyness <= 1.05:  # ATM
                                volume = 100
                            elif 0.90 <= moneyness <= 1.10:  # Near ATM
                                volume = 50
                            else:
                                volume = 25
                        
                    except Exception as e:
                        self.logger.warning(f"Could not calculate price for {ticker}: {e}")
                        continue
                
                # Include this option if we have a price
                if last_price and last_price > 0.05:
                    options_data.append({
                        'ticker': ticker,
                        'strike': strike,
                        'lastPrice': round(last_price, 2),
                        'bid': round(bid, 2) if bid else round(last_price * 0.95, 2),
                        'ask': round(ask, 2) if ask else round(last_price * 1.05, 2),
                        'volume': volume,
                        'openInterest': 0,  # Not available in basic plan
                        'impliedVolatility': base_vol,
                        'contract_type': contract_type,
                        'data_source': data_source
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error processing option contract {contract.get('ticker', 'unknown')}: {e}")
                continue
        
        # Log the data sources
        if real_price_count > 0:
            print(f"✅ Got real prices for {real_price_count} options")
        if calculated_price_count > 0:
            print(f"🧮 Calculated prices for {calculated_price_count} options (due to API plan limits)")
        
        # Ensure we have enough options for analysis
        if len(options_data) < 6:
            raise ValueError(f"Insufficient options data: only {len(options_data)} valid contracts found")
        
        return options_data
    
    def _black_scholes_price(self, S: float, K: float, exp_date: str, option_type: str,
                           volatility: float, r: float = 0.05) -> float:
        """Calculate Black-Scholes option price"""
        try:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            T = max((exp_datetime - datetime.now()).days / 365.0, 0.01)
            
            # Adjust volatility based on moneyness for more realistic pricing
            moneyness = K / S
            if option_type.lower() == 'put' and moneyness > 1.0:
                # Put skew - higher vol for OTM puts
                vol_adjust = 1 + (moneyness - 1) * 0.5
            elif option_type.lower() == 'call' and moneyness < 1.0:
                # Call skew - slightly higher vol for OTM calls  
                vol_adjust = 1 + (1 - moneyness) * 0.2
            else:
                vol_adjust = 1.0
            
            sigma = volatility * vol_adjust
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(0.05, price)
            
        except Exception as e:
            self.logger.warning(f"Black-Scholes calculation error: {e}")
            return 0.10  # Minimal fallback price
    
    def analyze_market_conditions(self, data: Dict) -> Dict:
        """Analyze market conditions from real data"""
        current_price = data['current_price']
        sma_20 = data['sma_20']
        sma_50 = data['sma_50']
        sma_200 = data['sma_200']
        rsi = data['rsi']
        realized_vol = data['realized_vol_21d']
        bb_upper = data['bb_upper']
        bb_lower = data['bb_lower']
        
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
        elif current_price > sma_50:
            trend = 'SIDEWAYS_BULLISH'
            trend_strength = abs((current_price / sma_50 - 1) * 100)
        elif current_price < sma_50:
            trend = 'SIDEWAYS_BEARISH'
            trend_strength = abs((sma_50 / current_price - 1) * 100)
        else:
            trend = 'SIDEWAYS'
            trend_strength = 2.0
        
        # Volatility regime
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
        
        return {
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
            'volume_vs_avg': round(data['volume'] / data['avg_volume_20d'], 2)
        }
    
    def select_strategy(self, market_analysis: Dict, underlying_data: Dict, options_data: Dict) -> Dict[str, float]:
        """Select optimal strategy based on real market analysis"""
        calls = options_data.get('calls', pd.DataFrame())
        puts = options_data.get('puts', pd.DataFrame())
        
        if calls.empty or puts.empty:
            raise ValueError("Insufficient options data for strategy analysis")
        
        # Ensure we have liquid options (real volume/prices)
        liquid_calls = calls[calls['lastPrice'] > 0.05]
        liquid_puts = puts[puts['lastPrice'] > 0.05]
        
        if liquid_calls.empty or liquid_puts.empty:
            raise ValueError("No liquid options found")
        
        # Get market conditions
        trend = market_analysis['trend']
        vol_regime = market_analysis['volatility_regime']
        momentum = market_analysis['momentum']
        bb_signal = market_analysis['bb_signal']
        
        scores = {}
        
        # Strategy scoring based on real market conditions
        
        # Covered Call - income strategy
        if len(liquid_calls) >= 1:
            base_score = 7.0
            if trend in ['SIDEWAYS', 'SIDEWAYS_BULLISH']:
                base_score += 1.5
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                base_score += 1.0
            if momentum in ['OVERBOUGHT', 'EXTREMELY_OVERBOUGHT']:
                base_score += 0.8
            scores['COVERED_CALL'] = base_score
        
        # Cash Secured Put - acquisition strategy
        if len(liquid_puts) >= 1:
            base_score = 7.0
            if trend in ['BULLISH', 'STRONG_BULLISH', 'SIDEWAYS_BULLISH']:
                base_score += 1.5
            if momentum in ['OVERSOLD', 'EXTREMELY_OVERSOLD']:
                base_score += 1.2
            if bb_signal in ['LOWER_BAND', 'EXTREME_LOWER']:
                base_score += 1.0
            scores['CASH_SECURED_PUT'] = base_score
        
        # Iron Condor - range-bound strategy
        if len(liquid_calls) >= 2 and len(liquid_puts) >= 2:
            base_score = 6.5
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
        
        # Long Straddle - volatility strategy
        if len(liquid_calls) >= 1 and len(liquid_puts) >= 1:
            base_score = 5.5
            if vol_regime == 'LOW_VOL':
                base_score += 2.0
            if trend == 'SIDEWAYS':
                base_score += 1.0
            if bb_signal == 'MIDDLE_RANGE':
                base_score += 0.8
            scores['LONG_STRADDLE'] = base_score
        
        # Protective Put - insurance
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
        
        # Return top 5 strategies
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def get_options_greeks(self, ticker: str, current_price: float = None) -> Dict:
        """Get options Greeks data from Polygon API"""
        try:
            print(f"🔢 Fetching options Greeks for {ticker}...")
            
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
                    
                    recent_aggs = []
                    for agg in self.client.list_aggs(
                        ticker,
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
            greeks_data = self._process_options_greeks(contracts, current_price, ticker)
            
            return greeks_data
            
        except Exception as e:
            print(f"❌ Failed to get Greeks for {ticker}: {str(e)}")
            raise
    
    def _process_options_greeks(self, contracts: List, current_price: float, underlying_ticker: str) -> Dict:
        """Process options contracts and calculate Greeks"""
        
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
                
                # Filter strikes within reasonable range (±30%)
                if abs(strike - current_price) / current_price > 0.30:
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
        calls_greeks = self._calculate_option_greeks(exp_groups[best_exp]['calls'], current_price, 'call')
        puts_greeks = self._calculate_option_greeks(exp_groups[best_exp]['puts'], current_price, 'put')
        
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
            'calls_greeks': pd.DataFrame(calls_greeks),
            'puts_greeks': pd.DataFrame(puts_greeks),
            'all_greeks': pd.DataFrame(all_greeks),
            'summary_stats': summary_stats,
            'total_contracts': len(all_greeks)
        }
    
    def _calculate_option_greeks(self, contracts: List[Dict], underlying_price: float, option_type: str) -> List[Dict]:
        """Calculate Greeks for option contracts"""
        greeks_data = []
        
        # Risk-free rate (approximate)
        r = 0.05
        # Base volatility estimate
        vol = 0.25
        
        for contract in contracts:
            try:
                strike = contract['strike']
                expiration = contract['expiration_date']
                
                # Calculate time to expiration
                exp_datetime = datetime.strptime(expiration, '%Y-%m-%d')
                T = max((exp_datetime - datetime.now()).days / 365.0, 0.01)
                
                # Adjust volatility based on moneyness
                moneyness = strike / underlying_price
                if option_type == 'put' and moneyness > 1.0:
                    vol_adj = vol * (1 + (moneyness - 1) * 0.3)
                elif option_type == 'call' and moneyness < 1.0:
                    vol_adj = vol * (1 + (1 - moneyness) * 0.2)
                else:
                    vol_adj = vol
                
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
                    'implied_vol': round(vol_adj, 3)
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
                'price': round(max(0.01, price), 2),
                'delta': round(delta, 4),
                'gamma': round(gamma, 4),
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

    def backtest_strategy(self, ticker: str, strategy_name: str, start_date: str, end_date: str, 
                         parameters: Dict = None) -> Dict:
        """Backtest an options strategy over a specified period"""
        try:
            print(f"🔄 Starting backtest for {strategy_name} on {ticker}")
            
            # Get historical data for backtesting period
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Fetch extended historical data
            total_days = (end_dt - start_dt).days + 100  # Extra days for indicators
            historical_data = self.get_stock_data(ticker, days=total_days)
            
            # Filter data to backtest period
            hist_df = historical_data['historical_data']
            mask = (hist_df.index >= start_dt) & (hist_df.index <= end_dt)
            backtest_df = hist_df[mask].copy()
            
            if len(backtest_df) < 30:
                raise ValueError(f"Insufficient data for backtesting period: {len(backtest_df)} days")
            
            # Run strategy backtest
            if strategy_name == 'COVERED_CALL':
                results = self._backtest_covered_call(backtest_df, parameters or {})
            elif strategy_name == 'CASH_SECURED_PUT':
                results = self._backtest_cash_secured_put(backtest_df, parameters or {})
            elif strategy_name == 'IRON_CONDOR':
                results = self._backtest_iron_condor(backtest_df, parameters or {})
            elif strategy_name == 'BULL_CALL_SPREAD':
                results = self._backtest_bull_call_spread(backtest_df, parameters or {})
            elif strategy_name == 'BEAR_PUT_SPREAD':
                results = self._backtest_bear_put_spread(backtest_df, parameters or {})
            else:
                # Default buy and hold strategy
                results = self._backtest_buy_and_hold(backtest_df)
            
            # Calculate performance metrics
            performance = self._calculate_backtest_performance(results, backtest_df)
            
            return {
                'ticker': ticker,
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
                'strategy': strategy_name,
                'error': str(e),
                'success': False
            }
    
    def _backtest_covered_call(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Backtest covered call strategy"""
        # Parameters
        dte_target = params.get('days_to_expiry', 30)  # Days to expiry
        delta_target = params.get('delta_target', 0.3)  # Target delta
        
        trades = []
        equity_curve = []
        current_position = None
        total_pnl = 0
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            
            # If no position, enter new covered call
            if current_position is None:
                # Buy 100 shares
                stock_cost = current_price * 100
                # Sell call option (estimate premium)
                call_strike = current_price * (1 + delta_target)
                call_premium = self._estimate_option_premium(current_price, call_strike, dte_target, 'call') * 100
                
                current_position = {
                    'entry_date': current_date,
                    'stock_price': current_price,
                    'stock_cost': stock_cost,
                    'call_strike': call_strike,
                    'call_premium': call_premium,
                    'expiry_date': current_date + timedelta(days=dte_target)
                }
                
                total_pnl -= stock_cost  # Buy stock
                total_pnl += call_premium  # Sell call
            
            # Check if position should be closed
            if current_position and current_date >= current_position['expiry_date']:
                # Close position
                stock_pnl = (current_price - current_position['stock_price']) * 100
                
                if current_price > current_position['call_strike']:
                    # Called away
                    stock_sale = current_position['call_strike'] * 100
                    call_pnl = 0  # Keep full premium
                else:
                    # Keep stock
                    stock_sale = current_price * 100
                    call_pnl = 0  # Keep full premium
                
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
            
            equity_curve.append({
                'date': current_date,
                'portfolio_value': total_pnl + (current_price * 100 if current_position else 0),
                'underlying_price': current_price
            })
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_pnl': total_pnl
        }
    
    def _backtest_cash_secured_put(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Backtest cash secured put strategy"""
        dte_target = params.get('days_to_expiry', 30)
        delta_target = params.get('delta_target', -0.3)
        
        trades = []
        equity_curve = []
        current_position = None
        cash_balance = 10000  # Starting cash
        total_pnl = 0
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            
            if current_position is None and cash_balance >= current_price * 100:
                # Sell put option
                put_strike = current_price * (1 + delta_target)  # OTM put
                put_premium = self._estimate_option_premium(current_price, put_strike, dte_target, 'put') * 100
                
                current_position = {
                    'entry_date': current_date,
                    'put_strike': put_strike,
                    'put_premium': put_premium,
                    'expiry_date': current_date + timedelta(days=dte_target)
                }
                
                cash_balance += put_premium
                total_pnl += put_premium
            
            if current_position and current_date >= current_position['expiry_date']:
                if current_price < current_position['put_strike']:
                    # Assigned - buy stock
                    stock_cost = current_position['put_strike'] * 100
                    cash_balance -= stock_cost
                    total_pnl -= stock_cost
                    
                    trade_pnl = current_position['put_premium'] - (current_position['put_strike'] - current_price) * 100
                else:
                    # Keep premium
                    trade_pnl = current_position['put_premium']
                
                trades.append({
                    'entry_date': current_position['entry_date'],
                    'exit_date': current_date,
                    'strategy': 'CASH_SECURED_PUT',
                    'pnl': trade_pnl,
                    'return_pct': (trade_pnl / (current_position['put_strike'] * 100)) * 100
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
    
    def _backtest_iron_condor(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Backtest iron condor strategy"""
        dte_target = params.get('days_to_expiry', 30)
        wing_width = params.get('wing_width', 0.05)  # 5% wing width
        
        trades = []
        equity_curve = []
        total_pnl = 0
        
        i = 0
        while i < len(df) - dte_target:
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            
            # Set up iron condor strikes
            call_sell_strike = current_price * 1.05
            call_buy_strike = current_price * 1.10
            put_sell_strike = current_price * 0.95
            put_buy_strike = current_price * 0.90
            
            # Calculate premiums
            call_sell_premium = self._estimate_option_premium(current_price, call_sell_strike, dte_target, 'call')
            call_buy_premium = self._estimate_option_premium(current_price, call_buy_strike, dte_target, 'call')
            put_sell_premium = self._estimate_option_premium(current_price, put_sell_strike, dte_target, 'put')
            put_buy_premium = self._estimate_option_premium(current_price, put_buy_strike, dte_target, 'put')
            
            net_credit = (call_sell_premium + put_sell_premium - call_buy_premium - put_buy_premium) * 100
            
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
                put_loss = (put_sell_strike - expiry_price) * 100
                max_loss = (put_sell_strike - put_buy_strike) * 100
                trade_pnl = net_credit - min(put_loss, max_loss)
            elif expiry_price > call_sell_strike:
                # Loss on call side
                call_loss = (expiry_price - call_sell_strike) * 100
                max_loss = (call_buy_strike - call_sell_strike) * 100
                trade_pnl = net_credit - min(call_loss, max_loss)
            else:
                # In profit zone
                trade_pnl = net_credit
            
            trades.append({
                'entry_date': current_date,
                'exit_date': expiry_date,
                'strategy': 'IRON_CONDOR',
                'pnl': trade_pnl,
                'return_pct': (trade_pnl / (abs(net_credit) + 1000)) * 100  # Rough margin estimate
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
    
    def _backtest_bull_call_spread(self, df: pd.DataFrame, params: Dict) -> Dict:
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
            
            buy_premium = self._estimate_option_premium(current_price, buy_strike, dte_target, 'call')
            sell_premium = self._estimate_option_premium(current_price, sell_strike, dte_target, 'call')
            
            net_debit = (buy_premium - sell_premium) * 100
            
            # Jump to expiry
            expiry_idx = min(i + dte_target, len(df) - 1)
            expiry_date = df.index[expiry_idx]
            expiry_price = df.iloc[expiry_idx]['close']
            
            # Calculate P&L at expiry
            if expiry_price <= buy_strike:
                trade_pnl = -net_debit  # Max loss
            elif expiry_price >= sell_strike:
                trade_pnl = (sell_strike - buy_strike) * 100 - net_debit  # Max profit
            else:
                trade_pnl = (expiry_price - buy_strike) * 100 - net_debit
            
            trades.append({
                'entry_date': current_date,
                'exit_date': expiry_date,
                'strategy': 'BULL_CALL_SPREAD',
                'pnl': trade_pnl,
                'return_pct': (trade_pnl / abs(net_debit)) * 100
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
    
    def _backtest_bear_put_spread(self, df: pd.DataFrame, params: Dict) -> Dict:
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
            
            buy_premium = self._estimate_option_premium(current_price, buy_strike, dte_target, 'put')
            sell_premium = self._estimate_option_premium(current_price, sell_strike, dte_target, 'put')
            
            net_debit = (buy_premium - sell_premium) * 100
            
            # Jump to expiry
            expiry_idx = min(i + dte_target, len(df) - 1)
            expiry_date = df.index[expiry_idx]
            expiry_price = df.iloc[expiry_idx]['close']
            
            # Calculate P&L at expiry
            if expiry_price >= buy_strike:
                trade_pnl = -net_debit  # Max loss
            elif expiry_price <= sell_strike:
                trade_pnl = (buy_strike - sell_strike) * 100 - net_debit  # Max profit
            else:
                trade_pnl = (buy_strike - expiry_price) * 100 - net_debit
            
            trades.append({
                'entry_date': current_date,
                'exit_date': expiry_date,
                'strategy': 'BEAR_PUT_SPREAD',
                'pnl': trade_pnl,
                'return_pct': (trade_pnl / abs(net_debit)) * 100
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
    
    def _backtest_buy_and_hold(self, df: pd.DataFrame) -> Dict:
        """Backtest simple buy and hold strategy"""
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        
        shares = 100
        start_value = start_price * shares
        end_value = end_price * shares
        total_pnl = end_value - start_value
        
        equity_curve = []
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            portfolio_value = (current_price * shares) - start_value
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
    
    def _estimate_option_premium(self, spot: float, strike: float, dte: int, option_type: str) -> float:
        """Estimate option premium using simplified Black-Scholes"""
        T = dte / 365.0
        r = 0.05
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
        
        # Sharpe ratio (assuming risk-free rate of 5%)
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
        
        # Benchmark comparison (buy and hold)
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
    
    def predict_market_direction(self, ticker: str, prediction_days: int = 30) -> Dict:
        """Predict market direction using technical analysis and volatility"""
        try:
            print(f"🔮 Generating market predictions for {ticker}")
            
            # Get extended historical data for better predictions
            data = self.get_stock_data(ticker, days=500)
            df = data['historical_data']
            
            current_price = data['current_price']
            
            # Technical analysis predictions
            technical_signals = self._analyze_technical_signals(df, current_price)
            
            # Volatility forecasting
            volatility_forecast = self._forecast_volatility(df, prediction_days)
            
            # Support and resistance levels
            support_resistance = self._calculate_support_resistance(df, current_price)
            
            # Price targets
            price_targets = self._calculate_price_targets(df, current_price, technical_signals)
            
            # Momentum analysis
            momentum_analysis = self._analyze_momentum(df, current_price)
            
            # Overall prediction
            overall_prediction = self._generate_overall_prediction(
                technical_signals, volatility_forecast, support_resistance, 
                price_targets, momentum_analysis
            )
            
            return {
                'ticker': ticker,
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
                'error': str(e),
                'success': False
            }
    
    def _analyze_technical_signals(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Analyze technical indicators for prediction signals"""
        latest = df.iloc[-1]
        
        # Calculate additional indicators if not present
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
        
        # RSI signal
        rsi = latest.get('rsi', 50)
        if rsi > 70:
            rsi_signal = 'BEARISH'
            rsi_strength = min((rsi - 70) / 10, 1.0)
        elif rsi < 30:
            rsi_signal = 'BULLISH'
            rsi_strength = min((30 - rsi) / 10, 1.0)
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
        macd_line = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
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
    
    def _forecast_volatility(self, df: pd.DataFrame, days: int) -> Dict:
        """Forecast volatility using GARCH-like simple model"""
        returns = df['close'].pct_change().dropna()
        
        # Current realized volatility
        vol_10d = returns.tail(10).std() * np.sqrt(252)
        vol_21d = returns.tail(21).std() * np.sqrt(252)
        vol_63d = returns.tail(63).std() * np.sqrt(252)
        
        # Simple volatility forecast (mean reversion model)
        long_term_vol = returns.std() * np.sqrt(252)
        current_vol = vol_21d
        
        # Mean reversion parameter (alpha)
        alpha = 0.1
        forecast_vol = current_vol * (1 - alpha) + long_term_vol * alpha
        
        # Volatility regime
        if current_vol > long_term_vol * 1.5:
            regime = 'HIGH_VOLATILITY'
            regime_confidence = min((current_vol / long_term_vol - 1), 1.0)
        elif current_vol < long_term_vol * 0.7:
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
    
    def _calculate_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate support and resistance levels"""
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
        
        return {
            'resistance_levels': [round(r, 2) for r in resistance_levels],
            'support_levels': [round(s, 2) for s in support_levels],
            'fibonacci_levels': {k: round(v, 2) for k, v in fib_levels.items()},
            '52_week_high': round(high_52w, 2),
            '52_week_low': round(low_52w, 2),
            'distance_to_52w_high': round(((high_52w / current_price) - 1) * 100, 2),
            'distance_to_52w_low': round(((current_price / low_52w) - 1) * 100, 2)
        }
    
    def _calculate_price_targets(self, df: pd.DataFrame, current_price: float, 
                                technical_signals: Dict) -> Dict:
        """Calculate price targets based on technical analysis"""
        
        # Average True Range for volatility-based targets
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        current_atr = atr.iloc[-1]
        
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
            target_1 = current_price + current_atr
            target_2 = current_price + (current_atr * 2)
            target_3 = current_price + (current_atr * 3)
            
            stop_loss = current_price - current_atr
        else:
            # Bearish targets
            target_1 = current_price - current_atr
            target_2 = current_price - (current_atr * 2)
            target_3 = current_price - (current_atr * 3)
            
            stop_loss = current_price + current_atr
        
        # Probability estimates (simplified)
        target_1_prob = 0.7
        target_2_prob = 0.4
        target_3_prob = 0.2
        
        return {
            'bullish_targets': {
                'target_1': round(max(target_1, current_price), 2),
                'target_2': round(max(target_2, current_price), 2),
                'target_3': round(max(target_3, current_price), 2),
                'probabilities': [target_1_prob, target_2_prob, target_3_prob]
            },
            'bearish_targets': {
                'target_1': round(min(target_1, current_price), 2),
                'target_2': round(min(target_2, current_price), 2),
                'target_3': round(min(target_3, current_price), 2),
                'probabilities': [target_1_prob, target_2_prob, target_3_prob]
            },
            'bollinger_bands': {
                'upper': round(bb_upper, 2),
                'middle': round(bb_middle, 2),
                'lower': round(bb_lower, 2)
            },
            'atr_value': round(current_atr, 2),
            'suggested_stop_loss': round(stop_loss, 2)
        }
    
    def _analyze_momentum(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Analyze price momentum indicators"""
        
        # Rate of Change (ROC)
        roc_5 = ((current_price / df['close'].iloc[-6]) - 1) * 100 if len(df) > 5 else 0
        roc_10 = ((current_price / df['close'].iloc[-11]) - 1) * 100 if len(df) > 10 else 0
        roc_20 = ((current_price / df['close'].iloc[-21]) - 1) * 100 if len(df) > 20 else 0
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Price momentum
        recent_highs = (df['close'].tail(5) == df['close'].tail(5).max()).sum()
        recent_lows = (df['close'].tail(5) == df['close'].tail(5).min()).sum()
        
        if recent_highs >= 3:
            momentum_direction = 'STRONG_BULLISH'
        elif recent_lows >= 3:
            momentum_direction = 'STRONG_BEARISH'
        elif roc_5 > 2:
            momentum_direction = 'BULLISH'
        elif roc_5 < -2:
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
            'volume_signal': 'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.7 else 'NORMAL'
        }
    
    def _generate_overall_prediction(self, technical_signals: Dict, volatility_forecast: Dict,
                                   support_resistance: Dict, price_targets: Dict,
                                   momentum_analysis: Dict) -> Dict:
        """Generate overall market prediction by combining all analyses"""
        
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
        
        # Volatility scoring
        if volatility_forecast['volatility_regime'] == 'HIGH_VOLATILITY':
            # High vol favors range-bound strategies
            bearish_score += 1
        elif volatility_forecast['volatility_regime'] == 'LOW_VOLATILITY':
            # Low vol favors directional strategies
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
        
        # Time horizon recommendation
        if momentum_analysis['momentum_strength'] > 0.7:
            time_horizon = 'SHORT_TERM'  # 1-2 weeks
        elif any('LONG_TERM' in signal for signal in technical_signals['moving_average_signals']):
            time_horizon = 'LONG_TERM'  # 2-3 months
        else:
            time_horizon = 'MEDIUM_TERM'  # 1 month
        
        # Key risks
        risks = []
        if volatility_forecast['volatility_regime'] == 'HIGH_VOLATILITY':
            risks.append('High volatility environment - expect larger price swings')
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

    def analyze_symbol(self, ticker: str, debug: bool = False) -> Dict:
        """Analyze a single symbol with real data only and detailed error reporting"""
        try:
            print(f"🔍 Starting real data analysis for {ticker}")
            
            # Step 1: Get underlying data
            if debug:
                print("**Debug:** Fetching underlying data...")
            
            if ticker.startswith('I:'):
                # Index data
                underlying_data = self.get_index_data(ticker)
                # For index options, we'd need to find the corresponding ETF
                # For now, raise an error as direct index options aren't widely available
                raise ValueError(f"Direct index options not supported. Try ETF equivalent instead of {ticker}")
            else:
                # Stock/ETF data
                underlying_data = self.get_stock_data(ticker)
            
            if debug:
                print(f"**Debug:** Got {underlying_data['data_points']} data points")
                print(f"**Debug:** Current price: ${underlying_data['current_price']:.2f}")
            
            # Validate underlying data
            if underlying_data['current_price'] <= 0:
                raise ValueError(f"Invalid current price: {underlying_data['current_price']}")
            
            if underlying_data['data_points'] < 21:
                raise ValueError(f"Insufficient data points: {underlying_data['data_points']}")
            
            # Step 2: Get options data
            if debug:
                print("**Debug:** Fetching options data...")
            
            # Pass the current price to avoid duplicate data fetching
            options_data = self.get_options_data(ticker, underlying_data['current_price'])
            
            if debug:
                print(f"**Debug:** Found {len(options_data['calls'])} calls, {len(options_data['puts'])} puts")
                print(f"**Debug:** Expiration: {options_data['expiration']}")
            
            # Validate options data
            if options_data['calls'].empty or options_data['puts'].empty:
                raise ValueError("No options data available")
            
            if len(options_data['calls']) < 3 or len(options_data['puts']) < 3:
                raise ValueError(f"Insufficient options: {len(options_data['calls'])} calls, {len(options_data['puts'])} puts")
            
            # Step 3: Market analysis
            if debug:
                print("**Debug:** Analyzing market conditions...")
            
            market_analysis = self.analyze_market_conditions(underlying_data)
            
            if debug:
                print(f"**Debug:** Trend: {market_analysis['trend']}")
                print(f"**Debug:** Volatility: {market_analysis['volatility_regime']}")
                print(f"**Debug:** Momentum: {market_analysis['momentum']}")
            
            # Step 4: Strategy selection
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
            print(f"❌ Analysis failed for {ticker}: {error_msg}")
            
            if debug:
                import traceback
                print("**Debug - Full Error Traceback:**")
                print(traceback.format_exc())
            
            return {
                'ticker': ticker,
                'error': error_msg,
                'success': False
            }


# =============================================================================
# STRATEGY EXPLANATIONS
# =============================================================================

def get_strategy_explanations() -> Dict[str, Dict]:
    """Get detailed explanations for all option strategies"""
    return {
        'COVERED_CALL': {
            'name': 'Covered Call',
            'description': 'Income-generating strategy where you own stock and sell call options',
            'market_outlook': 'Neutral to slightly bullish',
            'max_profit': 'Strike price - stock purchase price + premium received',
            'max_loss': 'Stock purchase price - premium received',
            'breakeven': 'Stock purchase price - premium received',
            'when_to_use': [
                'You own the underlying stock',
                'Expecting sideways to slightly bullish movement',
                'Want to generate additional income',
                'Willing to sell stock if called away'
            ],
            'pros': [
                'Generates additional income from premiums',
                'Reduces cost basis of stock position',
                'Limited downside protection from premium'
            ],
            'cons': [
                'Caps upside potential if stock rises significantly',
                'Stock can still decline below breakeven',
                'May be forced to sell stock at strike price'
            ],
            'example': 'Own 100 shares of EWU at $32. Sell 1 call option with $35 strike for $1.50 premium.'
        },
        
        'CASH_SECURED_PUT': {
            'name': 'Cash Secured Put',
            'description': 'Strategy to acquire stock at a discount by selling put options while holding cash',
            'market_outlook': 'Neutral to bullish',
            'max_profit': 'Premium received',
            'max_loss': 'Strike price - premium received',
            'breakeven': 'Strike price - premium received',
            'when_to_use': [
                'Want to buy stock at a lower price',
                'Have cash available for stock purchase',
                'Expecting neutral to bullish movement',
                'Comfortable owning the underlying stock'
            ],
            'pros': [
                'Earns premium while waiting to buy stock',
                'Acquires stock at effective discount if assigned',
                'Limited risk if you want to own the stock anyway'
            ],
            'cons': [
                'Miss out if stock rises significantly',
                'May be forced to buy stock in declining market',
                'Ties up capital as collateral'
            ],
            'example': 'Want to buy EWU. Sell put with $30 strike for $1.20 while holding $3,000 cash.'
        },
        
        'IRON_CONDOR': {
            'name': 'Iron Condor',
            'description': 'Range-bound strategy selling both call and put spreads for premium collection',
            'market_outlook': 'Neutral (sideways movement)',
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
                'High probability of profit in range'
            ],
            'cons': [
                'Limited profit potential',
                'Loses if stock moves significantly in either direction',
                'Multiple commissions and bid/ask spreads'
            ],
            'example': 'EWU at $32. Sell $30 put, buy $28 put, sell $35 call, buy $37 call for net credit.'
        },
        
        'BULL_CALL_SPREAD': {
            'name': 'Bull Call Spread',
            'description': 'Bullish strategy buying lower strike call and selling higher strike call',
            'market_outlook': 'Moderately bullish',
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
            'example': 'EWU at $32. Buy $32 call for $2, sell $35 call for $0.75. Net cost: $1.25.'
        },
        
        'BEAR_PUT_SPREAD': {
            'name': 'Bear Put Spread',
            'description': 'Bearish strategy buying higher strike put and selling lower strike put',
            'market_outlook': 'Moderately bearish',
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
            'example': 'EWU at $32. Buy $32 put for $1.80, sell $29 put for $0.60. Net cost: $1.20.'
        },
        
        'LONG_STRADDLE': {
            'name': 'Long Straddle',
            'description': 'Volatility strategy buying both call and put at same strike',
            'market_outlook': 'Neutral direction, expecting high volatility',
            'max_profit': 'Unlimited (theoretically)',
            'max_loss': 'Total premium paid',
            'breakeven': 'Two points: Strike ± total premium paid',
            'when_to_use': [
                'Expecting significant price movement',
                'Uncertain about direction',
                'Before earnings or major announcements',
                'Low implied volatility environment'
            ],
            'pros': [
                'Profits from large moves in either direction',
                'Unlimited upside potential',
                'Benefits from volatility expansion'
            ],
            'cons': [
                'High premium cost',
                'Needs significant movement to be profitable',
                'Time decay hurts both options'
            ],
            'example': 'EWU at $32. Buy $32 call for $2 and $32 put for $1.80. Total cost: $3.80.'
        },
        
        'PROTECTIVE_PUT': {
            'name': 'Protective Put',
            'description': 'Insurance strategy buying put options while owning stock',
            'market_outlook': 'Bullish but want downside protection',
            'max_profit': 'Unlimited (stock appreciation - put premium)',
            'max_loss': 'Stock price - strike price + put premium',
            'breakeven': 'Stock price + put premium paid',
            'when_to_use': [
                'Own stock and want downside protection',
                'Expecting volatility or uncertainty',
                'Protecting gains in profitable position',
                'Cannot afford significant losses'
            ],
            'pros': [
                'Provides downside protection',
                'Maintains upside potential',
                'Peace of mind during volatile periods'
            ],
            'cons': [
                'Cost of insurance reduces returns',
                'Premium lost if stock doesn\'t decline',
                'Time decay reduces put value'
            ],
            'example': 'Own 100 EWU shares at $32. Buy $30 put for $1.20 as insurance.'
        }
    }


# =============================================================================
# ENHANCED STREAMLIT INTERFACE WITH TABS
# =============================================================================

def main():
    st.set_page_config(
        page_title="Polygon Real Data FTSE Strategist", 
        page_icon="🇬🇧", 
        layout="wide"
    )
    
    st.title("🇬🇧 Indices Options Strategist")
    
    # Initialize session state for storing analysis results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'greeks_result' not in st.session_state:
        st.session_state.greeks_result = None
    if 'backtest_result' not in st.session_state:
        st.session_state.backtest_result = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analysis", "📚 Strategy Guide", "🔢 Options Greeks", "📈 Backtester", "🔮 Market Predictions"])
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 Configuration")
        
        # API Key (required)
        polygon_key = st.text_input(
            "Polygon API Key (Required)", 
            value="igO7PgpW43MsVcJvr1ZuxZ_vYrH87jLZ", 
            type="password",
            help="Real Polygon API key required - no demo data"
        )
        
        if not polygon_key:
            st.error("❌ Polygon API key required - this app uses real data only")
            st.stop()
        
        st.success("✅ API key provided")
        
        # Initialize strategist
        try:
            strategist = PolygonRealDataStrategist(polygon_key)
        except Exception as e:
            st.error(f"❌ Failed to initialize: {str(e)}")
            st.stop()
        
        st.markdown("---")
        
        # Discovery tools
        st.header("🔍 Discover FTSE 100 Options")
        
        if st.button("🎯 Find FTSE 100 Instruments"):
            with st.spinner("Searching for FTSE 100 instruments..."):
                try:
                    ftse_100_instruments = strategist.find_ftse_100_specific()
                    if ftse_100_instruments:
                        st.success(f"Found {len(ftse_100_instruments)} FTSE 100-related instruments:")
                        for instrument in ftse_100_instruments:
                            st.text(f"• {instrument['ticker']}: {instrument['name']} ({instrument['type']})")
                    else:
                        st.warning("No FTSE 100 specific instruments found")
                        
                        # Show popular UK ETFs as alternatives
                        st.info("**Try these UK market ETFs instead:**")
                        st.text("• EWU: iShares MSCI United Kingdom ETF")
                        st.text("• VGK: Vanguard FTSE Europe ETF") 
                        st.text("• FLGB: Franklin FTSE United Kingdom ETF")
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
        
        if st.button("✅ Check EWU Options"):
            with st.spinner("Checking EWU options availability..."):
                try:
                    options_check = strategist.check_options_availability("EWU")
                    if options_check['has_options']:
                        st.success(f"✅ EWU has options! Found {options_check.get('contract_count', 0)} contracts")
                        st.info("EWU (iShares MSCI United Kingdom ETF) is your best bet for FTSE 100 options exposure")
                    else:
                        st.warning(f"❌ EWU options check: {options_check['status']}")
                except Exception as e:
                    st.error(f"Options check failed: {str(e)}")
        
        if st.button("📊 Test EWU Data Quality"):
            with st.spinner("Testing EWU data quality..."):
                try:
                    data_check = strategist.quick_data_check("EWU")
                    if data_check['available']:
                        st.success(f"✅ EWU data looks good!")
                        st.write(f"• **Valid records:** {data_check['valid_records']}/{data_check['total_records']}")
                        st.write(f"• **Latest price:** ${data_check['latest_price']:.2f}")
                        st.write(f"• **Date range:** {data_check['date_range']}")
                    else:
                        st.error("❌ EWU data issues detected:")
                        if 'reason' in data_check:
                            st.write(f"• **Reason:** {data_check['reason']}")
                        if 'error' in data_check:
                            st.write(f"• **Error:** {data_check['error']}")
                        if 'valid_records' in data_check:
                            st.write(f"• **Valid records:** {data_check['valid_records']}/{data_check['total_records']}")
                except Exception as e:
                    st.error(f"Data check failed: {str(e)}")
        
        if st.button("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Find All FTSE Indices"):
            with st.spinner("Searching for FTSE indices..."):
                try:
                    ftse_indices = strategist.find_ftse_indices()
                    if ftse_indices:
                        st.success(f"Found {len(ftse_indices)} FTSE-related indices:")
                        for idx in ftse_indices:
                            st.text(f"• {idx['ticker']}: {idx['name']}")
                    else:
                        st.warning("No FTSE indices found in Polygon database")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
        
        if st.button("🇬🇧 Find UK ETFs"):
            with st.spinner("Searching for UK ETFs..."):
                try:
                    uk_etfs = strategist.find_uk_etfs()
                    if uk_etfs:
                        st.success(f"Found {len(uk_etfs)} UK-related ETFs:")
                        for etf in uk_etfs:
                            st.text(f"• {etf['ticker']}: {etf['name']}")
                    else:
                        st.warning("No UK ETFs found")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
        
        st.markdown("---")
        
        # Analysis
        st.header("📊 Analysis")
        
        symbol_input = st.text_input(
            "Symbol to Analyze",
            value="EWU",
            help="Enter ticker (e.g., EWU for UK ETF)"
        )
        
        debug_mode = st.checkbox(
            "🐛 Debug Mode",
            value=False,
            help="Show detailed step-by-step analysis information"
        )
        
        analyze_button = st.button(
            "🚀 Analyze Real Data",
            type="primary",
            disabled=not symbol_input
        )
    
    # Tab 1: Analysis
    with tab1:
        if analyze_button and symbol_input:
            with st.spinner(f"Analyzing {symbol_input} with real data..."):
                result = strategist.analyze_symbol(symbol_input.upper(), debug=debug_mode)
            
            if result['success']:
                # Store result in session state
                st.session_state.analysis_result = result
                
                # Display results
                st.success(f"✅ Real data analysis complete for {result['ticker']}")
                
                # Market Data Summary
                st.subheader("📊 Market Data Summary")
                underlying = result['underlying_data']
                analysis = result['market_analysis']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${underlying['current_price']:.2f}")
                    st.metric("1-Day Change", f"{analysis['price_change_1d']:.2f}%")
                with col2:
                    st.metric("RSI", f"{analysis['rsi']:.1f}")
                    st.metric("5-Day Change", f"{analysis['price_change_5d']:.2f}%")
                with col3:
                    st.metric("Realized Vol (21d)", f"{analysis['realized_vol']:.1%}")
                    st.metric("20-Day Change", f"{analysis['price_change_20d']:.2f}%")
                with col4:
                    st.metric("Data Points", underlying['data_points'])
                    st.metric("Volume vs Avg", f"{analysis['volume_vs_avg']:.2f}x")
                
                # Trading Chart
                st.subheader("📈 Last Year Trading Chart")
                try:
                    # Ensure we have the required data structure
                    chart_data = {
                        'ticker': underlying['ticker'],
                        'current_price': underlying['current_price'],
                        'historical_data': underlying['historical_data'],
                        'name': f"{underlying['ticker']} Stock/ETF"
                    }
                    chart = strategist.create_trading_chart(chart_data)
                    st.plotly_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create chart: {str(e)}")
                    st.info("Chart generation failed, but analysis data is still available below.")
                    # Debug info
                    if debug_mode:
                        st.write("**Chart Debug Info:**")
                        st.write(f"- Historical data shape: {underlying.get('historical_data', pd.DataFrame()).shape}")
                        st.write(f"- Data columns: {list(underlying.get('historical_data', pd.DataFrame()).columns)}")
                        st.write(f"- Current price: {underlying.get('current_price', 'N/A')}")
                
                
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
                
                # Pricing breakdown
                if pricing:
                    st.markdown("**📊 Options Pricing Sources:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        real_total = pricing.get('total_real', 0)
                        calc_total = pricing.get('total_calculated', 0)
                        total = real_total + calc_total
                        if real_total > 0:
                            st.success(f"✅ Real Prices: {real_total}/{total} ({real_total/total*100:.0f}%)")
                        else:
                            st.info("📊 Real Prices: Not available (API plan limits)")
                    
                    with col2:
                        if calc_total > 0:
                            st.info(f"🧮 Calculated: {calc_total}/{total} ({calc_total/total*100:.0f}%)")
                    
                    with col3:
                        if calc_total > 0:
                            st.write("**Method:** Black-Scholes with volatility skew")
                        else:
                            st.write("**Source:** Live market data")
                
                # Strategy Recommendations
                st.subheader("💡 Strategy Recommendations")
                
                st.success(f"**Best Strategy:** {result['best_strategy']} (Confidence: {result['confidence']:.1f}/10)")
                
                # Note about analysis quality
                pricing = result['options_data'].get('pricing_breakdown', {})
                if pricing.get('total_calculated', 0) > 0:
                    st.info("🧮 **Analysis Note**: Strategy recommendations use professional Black-Scholes pricing models when real-time options prices aren't available due to API plan limits. This provides institutional-quality analysis.")
                
                st.markdown("**All Strategy Scores:**")
                for strategy, score in result['strategy_scores'].items():
                    st.write(f"• **{strategy}:** {score:.1f}/10")
                
                # Export data
                st.subheader("📤 Export Real Data")
                
                export_data = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'ticker': result['ticker'],
                    'market_analysis': result['market_analysis'],
                    'strategy_scores': result['strategy_scores'],
                    'best_strategy': result['best_strategy'],
                    'confidence': result['confidence'],
                    'options_summary': {
                        'expiration': options['expiration'],
                        'calls_count': len(options['calls']),
                        'puts_count': len(options['puts']),
                        'days_to_expiry': options['days_to_expiry']
                    }
                }
                
                st.download_button(
                    "📋 Download Analysis JSON",
                    json.dumps(export_data, indent=2),
                    f"{result['ticker']}_real_analysis.json",
                    "application/json"
                )
            
            else:
                st.error(f"❌ Analysis failed: {result['error']}")
        
        else:
            # Instructions for Analysis tab
            st.markdown("""
            ## 🇬🇧 Options - Indices Analysis
            
            ### 🔍 **Discovery Features:**
            1. **Find FTSE Indices:** Search for available FTSE indices in Polygon
            2. **Find UK ETFs:** Discover UK-related ETFs with options
            3. **Real-time Analysis:** Get actual market data and conditions
            4. **Trading Charts:** Visual analysis with candlesticks and moving averages
            
            ### 🔧 **Troubleshooting EWU Data Issues:**
            
            **If you're getting "only 3 valid days" error:**
            
            1. **Check Data Quality First:**
               - Click **"Test EWU Data Quality"** in sidebar
               - This will show exactly what data is available
            
            2. **Common Causes:**
               - **Free API Tier Limits:** Polygon free tier has restrictions
               - **Weekend/Holiday Data:** Markets closed, less data available  
               - **API Rate Limits:** Too many requests, data gets filtered
               - **Data Gaps:** Some ETFs have sporadic data
            
            3. **Solutions to Try:**
               - **Wait and retry:** API limits reset
               - **Try different ticker:** SPY, QQQ (more liquid)
               - **Check API status:** Polygon service issues
            
            ### 📊 **Alternative Tickers to Try:**
            - **SPY**: S&P 500 ETF (most liquid options)
            - **QQQ**: NASDAQ 100 ETF (very active)
            - **IWM**: Russell 2000 ETF (good volume)
            - **VGK**: Vanguard Europe ETF (includes UK)
            
            **SPY is the most reliable for testing - try it first!**
            
            ### 🚀 **Getting Started:**
            1. ✅ **Enter your Polygon API key** (required)
            2. ✅ **Use discovery tools** to find available FTSE/UK instruments  
            3. ✅ **Enter a symbol** (try "EWU" for UK market exposure)
            4. ✅ **Click Analyze** for real market analysis with charts
            
            ### 🎯 **For FTSE Exposure:**
            - **EWU**: iShares MSCI United Kingdom ETF (direct UK exposure)
            - **VGK**: Vanguard FTSE Europe ETF (includes UK)
            - Use discovery tools to find other available instruments
            """)
    
    # Tab 2: Strategy Guide
    with tab2:
        st.header("📚 Options Strategy Guide")
        
        st.markdown("""
        This comprehensive guide covers all the option strategies analyzed by our system. 
        Each strategy is explained with market conditions, risk/reward profiles, and practical examples.
        """)
        
        # Get strategy explanations
        strategies = get_strategy_explanations()
        
        # Create expandable sections for each strategy
        for strategy_key, strategy_info in strategies.items():
            with st.expander(f"📋 {strategy_info['name']}", expanded=False):
                
                # Strategy header
                st.markdown(f"**{strategy_info['description']}**")
                
                # Market outlook
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
                
                # Example
                st.markdown("### 📝 Example")
                st.info(strategy_info['example'])
                
                st.markdown("---")
        
        # Additional resources
        st.markdown("## 📚 Additional Resources")
        
        st.markdown("""
        ### Risk Management Tips:
        - **Never risk more than you can afford to lose**
        - **Start with paper trading to practice strategies**
        - **Understand assignment risk with short options**
        - **Monitor positions regularly, especially near expiration**
        - **Have exit strategies planned before entering trades**
        
        ### Market Conditions Guide:
        - **Bullish Market:** Bull call spreads, covered calls, cash-secured puts
        - **Bearish Market:** Bear put spreads, protective puts
        - **Sideways Market:** Iron condors, covered calls, cash-secured puts
        - **High Volatility:** Short premium strategies (covered calls, cash-secured puts)
        - **Low Volatility:** Long volatility strategies (straddles, protective puts)
        
        ### Key Metrics to Monitor:
        - **Delta:** Price sensitivity to underlying movement
        - **Gamma:** Rate of change of delta
        - **Theta:** Time decay effect
        - **Vega:** Volatility sensitivity
        - **Implied Volatility:** Market's expectation of future volatility
        """)
    
    # Tab 3: Options Greeks
    with tab3:
        st.header("🔢 Options Greeks Analysis")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Auto-populate with symbol from analysis tab if available
            default_symbol = "EWU"
            if st.session_state.analysis_result and st.session_state.analysis_result.get('success'):
                default_symbol = st.session_state.analysis_result['ticker']
            
            greeks_symbol = st.text_input(
                "Symbol for Greeks Analysis",
                value=default_symbol,
                help="Enter ticker symbol to analyze options Greeks"
            )
        
        with col2:
            get_greeks_button = st.button(
                "📊 Get Greeks Data",
                type="primary",
                disabled=not greeks_symbol or not polygon_key
            )
        
        if get_greeks_button and greeks_symbol:
            with st.spinner(f"Fetching Options Greeks for {greeks_symbol}..."):
                try:
                    greeks_result = strategist.get_options_greeks(greeks_symbol.upper())
                    
                    # Store result in session state
                    st.session_state.greeks_result = greeks_result
                    
                    st.success(f"✅ Greeks analysis complete for {greeks_result['underlying_ticker']}")
                    
                    # Show connection to analysis tab if same symbol
                    if (st.session_state.analysis_result and 
                        st.session_state.analysis_result.get('success') and
                        st.session_state.analysis_result['ticker'] == greeks_result['underlying_ticker']):
                        st.info("🔗 This Greeks analysis matches your symbol from the Analysis tab!")
                    
                    # Summary metrics
                    st.subheader("📊 Greeks Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Underlying Price", f"${greeks_result['underlying_price']:.2f}")
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
                            
                            # Format columns
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
                            
                            # Format columns
                            display_puts['price'] = display_puts['price'].apply(lambda x: f"${x:.2f}")
                            display_puts['strike'] = display_puts['strike'].apply(lambda x: f"${x:.2f}")
                            display_puts['moneyness'] = display_puts['moneyness'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(display_puts, use_container_width=True)
                        else:
                            st.info("No put options data available")
                    
                    # Greeks Visualization
                    st.subheader("📈 Greeks Visualization")
                    
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
                        
                        # Delta plot
                        if not calls_data.empty:
                            fig.add_trace(go.Scatter(x=calls_data['strike'], y=calls_data['delta'], 
                                                   mode='lines+markers', name='Calls Delta', 
                                                   line=dict(color='green')), row=1, col=1)
                        if not puts_data.empty:
                            fig.add_trace(go.Scatter(x=puts_data['strike'], y=puts_data['delta'], 
                                                   mode='lines+markers', name='Puts Delta', 
                                                   line=dict(color='red')), row=1, col=1)
                        
                        # Gamma plot
                        if not calls_data.empty:
                            fig.add_trace(go.Scatter(x=calls_data['strike'], y=calls_data['gamma'], 
                                                   mode='lines+markers', name='Calls Gamma', 
                                                   line=dict(color='green'), showlegend=False), row=1, col=2)
                        if not puts_data.empty:
                            fig.add_trace(go.Scatter(x=puts_data['strike'], y=puts_data['gamma'], 
                                                   mode='lines+markers', name='Puts Gamma', 
                                                   line=dict(color='red'), showlegend=False), row=1, col=2)
                        
                        # Theta plot
                        if not calls_data.empty:
                            fig.add_trace(go.Scatter(x=calls_data['strike'], y=calls_data['theta'], 
                                                   mode='lines+markers', name='Calls Theta', 
                                                   line=dict(color='green'), showlegend=False), row=2, col=1)
                        if not puts_data.empty:
                            fig.add_trace(go.Scatter(x=puts_data['strike'], y=puts_data['theta'], 
                                                   mode='lines+markers', name='Puts Theta', 
                                                   line=dict(color='red'), showlegend=False), row=2, col=1)
                        
                        # Vega plot
                        if not calls_data.empty:
                            fig.add_trace(go.Scatter(x=calls_data['strike'], y=calls_data['vega'], 
                                                   mode='lines+markers', name='Calls Vega', 
                                                   line=dict(color='green'), showlegend=False), row=2, col=2)
                        if not puts_data.empty:
                            fig.add_trace(go.Scatter(x=puts_data['strike'], y=puts_data['vega'], 
                                                   mode='lines+markers', name='Puts Vega', 
                                                   line=dict(color='red'), showlegend=False), row=2, col=2)
                        
                        # Add current price line to all subplots
                        current_price = greeks_result['underlying_price']
                        for row in [1, 2]:
                            for col in [1, 2]:
                                fig.add_vline(x=current_price, line_dash="dash", line_color="white", 
                                            row=row, col=col)
                        
                        fig.update_layout(
                            height=600,
                            title=f"Options Greeks for {greeks_result['underlying_ticker']}",
                            template='plotly_dark'
                        )
                        
                        fig.update_xaxes(title_text="Strike Price")
                        fig.update_yaxes(title_text="Delta", row=1, col=1)
                        fig.update_yaxes(title_text="Gamma", row=1, col=2)
                        fig.update_yaxes(title_text="Theta", row=2, col=1)
                        fig.update_yaxes(title_text="Vega", row=2, col=2)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Greeks Explanation
                    st.subheader("📚 Understanding the Greeks")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
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
                    
                    with col2:
                        st.markdown("""
                        **⏰ Theta (Θ)**
                        - Time decay per day
                        - Always negative for long options
                        - Accelerates as expiration approaches
                        
                        **📊 Vega (ν)**
                        - Sensitivity to volatility changes
                        - Highest for ATM options
                        - Important for volatility plays
                        """)
                    
                    # Export Greeks data
                    st.subheader("📤 Export Greeks Data")
                    
                    greeks_export = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'underlying_ticker': greeks_result['underlying_ticker'],
                        'underlying_price': greeks_result['underlying_price'],
                        'expiration': greeks_result['expiration'],
                        'days_to_expiry': greeks_result['days_to_expiry'],
                        'summary_stats': greeks_result['summary_stats'],
                        'total_contracts': greeks_result['total_contracts']
                    }
                    
                    st.download_button(
                        "📊 Download Greeks Summary",
                        json.dumps(greeks_export, indent=2),
                        f"{greeks_result['underlying_ticker']}_greeks_summary.json",
                        "application/json"
                    )
                    
                    # Download full Greeks data as CSV
                    if not all_greeks_df.empty:
                        csv_data = all_greeks_df.to_csv(index=False)
                        st.download_button(
                            "📋 Download Full Greeks CSV",
                            csv_data,
                            f"{greeks_result['underlying_ticker']}_greeks_full.csv",
                            "text/csv"
                        )
                
                except Exception as e:
                    st.error(f"❌ Greeks analysis failed: {str(e)}")
                    if debug_mode:
                        import traceback
                        st.code(traceback.format_exc())
        
        else:
            # Instructions for Greeks tab
            st.markdown("""
            ## 🔢 Options Greeks Analysis
            
            **The Greeks** are essential risk measures that quantify how option prices change relative to various factors.
            
            ### 🎯 **What You'll Get:**
            - **Real-time Greeks calculations** for all available options
            - **Visual charts** showing Greeks across strike prices
            - **Summary statistics** highlighting key levels
            - **ATM, OTM analysis** for better understanding
            - **Export capabilities** for further analysis
            
            ### 📊 **Key Greeks Explained:**
            
            **🔺 Delta:** Price sensitivity  
            - **Call Delta:** 0 to 1 (increases as price rises)
            - **Put Delta:** -1 to 0 (becomes less negative as price rises)
            - **ATM options:** ~0.5 delta for calls, ~-0.5 for puts
            
            **🔄 Gamma:** Delta sensitivity  
            - **Highest** for at-the-money (ATM) options
            - **Lower** for in-the-money (ITM) and out-of-the-money (OTM)
            - **Acceleration** factor for Delta changes
            
            **⏰ Theta:** Time decay  
            - **Always negative** for long options (you lose money daily)
            - **Accelerates** as expiration approaches
            - **Highest** for ATM options near expiration
            
            **📊 Vega:** Volatility sensitivity  
            - **Positive** for long options (higher vol = higher prices)
            - **Highest** for ATM options
            - **Decreases** as expiration approaches
            
            ### 🚀 **How to Use:**
            1. **Enter a symbol** (e.g., EWU, SPY, QQQ)
            2. **Click "Get Greeks Data"** to fetch real options data
            3. **Analyze the tables** for specific strikes and expirations
            4. **Study the charts** to see Greeks patterns
            5. **Export data** for your own analysis
            
            ### 💡 **Trading Applications:**
            - **Delta hedging:** Manage directional risk
            - **Gamma scalping:** Profit from volatility
            - **Theta strategies:** Benefit from time decay
            - **Vega plays:** Trade volatility expectations
            """)
    
    # Tab 4: Backtester
    with tab4:
        st.header("📈 Strategy Backtester")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Auto-populate with symbol from analysis tab if available
            default_symbol = "EWU"
            if st.session_state.analysis_result and st.session_state.analysis_result.get('success'):
                default_symbol = st.session_state.analysis_result['ticker']
            
            backtest_symbol = st.text_input(
                "Symbol for Backtesting",
                value=default_symbol,
                help="Enter ticker symbol for strategy backtesting"
            )
            
            # Strategy selection
            available_strategies = [
                'COVERED_CALL',
                'CASH_SECURED_PUT', 
                'IRON_CONDOR',
                'BULL_CALL_SPREAD',
                'BEAR_PUT_SPREAD',
                'BUY_AND_HOLD'
            ]
            
            # Auto-select best strategy if available
            default_strategy = 'COVERED_CALL'
            if st.session_state.analysis_result and st.session_state.analysis_result.get('success'):
                default_strategy = st.session_state.analysis_result.get('best_strategy', 'COVERED_CALL')
            
            selected_strategy = st.selectbox(
                "Strategy to Backtest",
                available_strategies,
                index=available_strategies.index(default_strategy) if default_strategy in available_strategies else 0
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
            st.markdown("### Strategy Parameters")
            
            # Strategy-specific parameters
            params = {}
            
            if selected_strategy in ['COVERED_CALL', 'CASH_SECURED_PUT']:
                params['days_to_expiry'] = st.slider("Days to Expiry", 15, 60, 30)
                params['delta_target'] = st.slider("Delta Target", 0.1, 0.5, 0.3, 0.05)
            
            elif selected_strategy == 'IRON_CONDOR':
                params['days_to_expiry'] = st.slider("Days to Expiry", 15, 60, 30)
                params['wing_width'] = st.slider("Wing Width %", 3, 10, 5) / 100
            
            elif selected_strategy in ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']:
                params['days_to_expiry'] = st.slider("Days to Expiry", 15, 60, 30)
            
            run_backtest_button = st.button(
                "🚀 Run Backtest",
                type="primary",
                disabled=not backtest_symbol or not polygon_key
            )
        
        if run_backtest_button and backtest_symbol:
            with st.spinner(f"Running {selected_strategy} backtest on {backtest_symbol}..."):
                try:
                    backtest_result = strategist.backtest_strategy(
                        backtest_symbol.upper(),
                        selected_strategy,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        params
                    )
                    
                    if backtest_result['success']:
                        st.success(f"✅ Backtest completed for {selected_strategy} on {backtest_result['ticker']}")
                        
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
                            st.write(f"**Profit Factor:** {abs(perf['avg_win'] / perf['avg_loss']):.2f}" if perf['avg_loss'] != 0 else "N/A")
                        
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
                                subplot_titles=('Strategy Performance vs Underlying', 'Underlying Price'),
                                row_heights=[0.7, 0.3]
                            )
                            
                            # Strategy equity curve
                            fig.add_trace(
                                go.Scatter(
                                    x=equity_df['date'],
                                    y=equity_df['portfolio_value'],
                                    mode='lines',
                                    name=f'{selected_strategy} P&L',
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
                                title=f'{selected_strategy} Backtest Results - {backtest_result["ticker"]}',
                                template='plotly_dark'
                            )
                            
                            fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
                            fig.update_yaxes(title_text="Price ($)", row=2, col=1)
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
                            'strategy': selected_strategy,
                            'ticker': backtest_result['ticker'],
                            'parameters': backtest_result['parameters'],
                            'performance_metrics': backtest_result['performance'],
                            'trade_count': len(trades) if trades else 0
                        }
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📊 Download Summary",
                                json.dumps(backtest_export, indent=2),
                                f"{backtest_result['ticker']}_{selected_strategy}_backtest.json",
                                "application/json"
                            )
                        
                        with col2:
                            if trades:
                                trades_csv = pd.DataFrame(trades).to_csv(index=False)
                                st.download_button(
                                    "📋 Download Trades CSV",
                                    trades_csv,
                                    f"{backtest_result['ticker']}_{selected_strategy}_trades.csv",
                                    "text/csv"
                                )
                    
                    else:
                        st.error(f"❌ Backtest failed: {backtest_result['error']}")
                
                except Exception as e:
                    st.error(f"❌ Backtest error: {str(e)}")
                    if debug_mode:
                        import traceback
                        st.code(traceback.format_exc())
        
        else:
            # Instructions for Backtester tab
            st.markdown("""
            ## 📈 Strategy Backtester
            
            **Test your options strategies** on historical data to see how they would have performed in real market conditions.
            
            ### 🎯 **Available Strategies:**
            - **Covered Call:** Income generation by selling calls against stock holdings
            - **Cash Secured Put:** Acquire stock at discount by selling puts with cash backing
            - **Iron Condor:** Range-bound strategy selling both call and put spreads
            - **Bull Call Spread:** Moderate bullish strategy with defined risk/reward
            - **Bear Put Spread:** Moderate bearish strategy with defined risk/reward
            - **Buy & Hold:** Benchmark comparison strategy
            
            ### 📊 **What You'll Get:**
            - **Performance Metrics:** Total P&L, Sharpe ratio, win rate, max drawdown
            - **Equity Curve:** Visual representation of strategy performance over time
            - **Trade Analysis:** Detailed breakdown of individual trades
            - **Risk Analytics:** Volatility, alpha vs benchmark, profit factor
            - **Export Capabilities:** Download results for further analysis
            
            ### 🔧 **Customizable Parameters:**
            - **Days to Expiry:** Target option expiration timeline
            - **Delta Targets:** Moneyness preferences for option selection
            - **Wing Width:** Spread sizing for complex strategies
            - **Date Range:** Custom backtesting periods
            
            ### 💡 **How to Use:**
            1. **Select a symbol** (auto-populated from Analysis tab if available)
            2. **Choose strategy** (best strategy auto-selected from analysis)
            3. **Set parameters** using the sliders and inputs
            4. **Select date range** for backtesting period
            5. **Run backtest** and analyze the results
            
            ### ⚠️ **Important Notes:**
            - **Past performance** does not guarantee future results
            - **Transaction costs** and slippage are not included
            - **Options pricing** uses Black-Scholes approximations
            - **Results are theoretical** and for educational purposes
            
            ### 🎯 **Interpretation Guide:**
            - **Sharpe Ratio > 1.5:** Excellent risk-adjusted returns
            - **Win Rate > 60%:** Good consistency
            - **Max Drawdown < 10%:** Conservative risk management
            - **Positive Alpha:** Outperforming benchmark
            """)
    
    # Tab 5: Market Predictions
    with tab5:
        st.header("🔮 Market Predictions")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Auto-populate with symbol from analysis tab if available
            default_symbol = "EWU"
            if st.session_state.analysis_result and st.session_state.analysis_result.get('success'):
                default_symbol = st.session_state.analysis_result['ticker']
            
            prediction_symbol = st.text_input(
                "Symbol for Prediction",
                value=default_symbol,
                help="Enter ticker symbol for market prediction analysis"
            )
        
        with col2:
            prediction_days = st.slider(
                "Prediction Horizon (Days)",
                7, 90, 30,
                help="Number of days to predict forward"
            )
            
            get_prediction_button = st.button(
                "🔮 Generate Prediction",
                type="primary",
                disabled=not prediction_symbol or not polygon_key
            )
        
        if get_prediction_button and prediction_symbol:
            with st.spinner(f"Generating market predictions for {prediction_symbol}..."):
                try:
                    prediction_result = strategist.predict_market_direction(
                        prediction_symbol.upper(), 
                        prediction_days
                    )
                    
                    if prediction_result['success']:
                        st.success(f"✅ Prediction analysis complete for {prediction_result['ticker']}")
                        
                        # Show connection to analysis tab if same symbol
                        if (st.session_state.analysis_result and 
                            st.session_state.analysis_result.get('success') and
                            st.session_state.analysis_result['ticker'] == prediction_result['ticker']):
                            st.info("🔗 This prediction analysis matches your symbol from the Analysis tab!")
                        
                        # Overall Prediction Summary
                        st.subheader("🎯 Overall Prediction")
                        
                        overall = prediction_result['overall_prediction']
                        
                        # Main prediction card
                        direction_emoji = "📈" if overall['direction'] == 'BULLISH' else "📉" if overall['direction'] == 'BEARISH' else "➡️"
                        confidence_color = "success" if overall['confidence'] > 0.7 else "warning" if overall['confidence'] > 0.5 else "error"
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: rgba(0,0,0,0.1); border-left: 5px solid {'green' if overall['direction'] == 'BULLISH' else 'red' if overall['direction'] == 'BEARISH' else 'orange'};">
                        <h3>{direction_emoji} {overall['direction']} Prediction</h3>
                        <p><strong>Confidence:</strong> {overall['confidence']:.0%} ({overall['strength']})</p>
                        <p><strong>Time Horizon:</strong> {overall['time_horizon'].replace('_', ' ')}</p>
                        <p><strong>Summary:</strong> {overall['summary']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Prediction Components
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"${prediction_result['current_price']:.2f}")
                            st.metric("Prediction Period", f"{prediction_days} days")
                        
                        with col2:
                            st.metric("Bullish Signals", overall['bullish_score'])
                            st.metric("Bearish Signals", overall['bearish_score'])
                        
                        with col3:
                            st.metric("Confidence Level", f"{overall['confidence']:.0%}")
                            st.metric("Signal Strength", overall['strength'])
                        
                        # Technical Analysis Details
                        st.subheader("🔧 Technical Analysis Breakdown")
                        
                        tech = prediction_result['technical_signals']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Momentum Indicators")
                            
                            # RSI Analysis
                            rsi_color = "🔴" if tech['rsi_signal'] == 'BEARISH' else "🟢" if tech['rsi_signal'] == 'BULLISH' else "🟡"
                            st.write(f"**RSI Signal:** {rsi_color} {tech['rsi_signal']} ({tech['rsi_value']:.1f})")
                            
                            # MACD Analysis
                            macd_color = "🟢" if tech['macd_signal'] == 'BULLISH' else "🔴" if tech['macd_signal'] == 'BEARISH' else "🟡"
                            st.write(f"**MACD Signal:** {macd_color} {tech['macd_signal']}")
                            
                            # Moving Average Analysis
                            st.write("**Moving Average Signals:**")
                            for signal in tech['moving_average_signals']:
                                st.write(f"  • {signal.replace('_', ' ')}")
                            
                            if not tech['moving_average_signals']:
                                st.write("  • No bullish MA signals")
                        
                        with col2:
                            st.markdown("### 📈 Price vs Moving Averages")
                            st.write(f"**vs 20-day SMA:** {tech['price_vs_sma20']:+.2f}%")
                            st.write(f"**vs 50-day SMA:** {tech['price_vs_sma50']:+.2f}%")
                            st.write(f"**vs 200-day SMA:** {tech['price_vs_sma200']:+.2f}%")
                            
                            st.markdown("### 🌊 Momentum Analysis")
                            momentum = prediction_result['momentum_analysis']
                            st.write(f"**Direction:** {momentum['momentum_direction']}")
                            st.write(f"**Strength:** {momentum['momentum_strength']:.2f}")
                            st.write(f"**5-day ROC:** {momentum['roc_5_day']:+.2f}%")
                            st.write(f"**Volume Signal:** {momentum['volume_signal']}")
                        
                        # Volatility Forecast
                        st.subheader("📊 Volatility Forecast")
                        
                        vol = prediction_result['volatility_forecast']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Vol (21d)", f"{vol['current_vol_21d']:.1%}")
                        with col2:
                            st.metric("Forecast Vol", f"{vol['forecast_vol']:.1%}")
                        with col3:
                            st.metric("Vol Regime", vol['volatility_regime'])
                        with col4:
                            st.metric("Vol Trend", vol['vol_trend'])
                        
                        # Support and Resistance
                        st.subheader("🎯 Support & Resistance Levels")
                        
                        sr = prediction_result['support_resistance']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### 🔻 Support Levels")
                            if sr['support_levels']:
                                for i, level in enumerate(sr['support_levels'], 1):
                                    distance = ((prediction_result['current_price'] / level) - 1) * 100
                                    st.write(f"**S{i}:** ${level:.2f} (-{distance:.1f}%)")
                            else:
                                st.write("No clear support levels identified")
                        
                        with col2:
                            st.markdown("### 🔺 Resistance Levels")
                            if sr['resistance_levels']:
                                for i, level in enumerate(sr['resistance_levels'], 1):
                                    distance = ((level / prediction_result['current_price']) - 1) * 100
                                    st.write(f"**R{i}:** ${level:.2f} (+{distance:.1f}%)")
                            else:
                                st.write("No clear resistance levels identified")
                        
                        with col3:
                            st.markdown("### 📏 Key Levels")
                            st.write(f"**52W High:** ${sr['52_week_high']:.2f} (+{sr['distance_to_52w_high']:.1f}%)")
                            st.write(f"**52W Low:** ${sr['52_week_low']:.2f} (+{sr['distance_to_52w_low']:.1f}%)")
                            
                            # Fibonacci levels
                            current_price = prediction_result['current_price']
                            for level_name, level_price in sr['fibonacci_levels'].items():
                                if abs(level_price - current_price) / current_price < 0.05:  # Within 5%
                                    distance = ((level_price / current_price) - 1) * 100
                                    st.write(f"**Fib {level_name}:** ${level_price:.2f} ({distance:+.1f}%)")
                        
                        # Price Targets
                        st.subheader("🎯 Price Targets")
                        
                        targets = prediction_result['price_targets']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📈 Bullish Targets")
                            bull_targets = targets['bullish_targets']
                            for i, (target, prob) in enumerate(zip([bull_targets['target_1'], bull_targets['target_2'], bull_targets['target_3']], 
                                                                 bull_targets['probabilities']), 1):
                                distance = ((target / prediction_result['current_price']) - 1) * 100
                                st.write(f"**T{i}:** ${target:.2f} (+{distance:.1f}%) - {prob:.0%} probability")
                        
                        with col2:
                            st.markdown("### 📉 Bearish Targets")
                            bear_targets = targets['bearish_targets']
                            for i, (target, prob) in enumerate(zip([bear_targets['target_1'], bear_targets['target_2'], bear_targets['target_3']], 
                                                                 bear_targets['probabilities']), 1):
                                distance = ((target / prediction_result['current_price']) - 1) * 100
                                st.write(f"**T{i}:** ${target:.2f} ({distance:.1f}%) - {prob:.0%} probability")
                        
                        # Risk Assessment
                        if overall['key_risks']:
                            st.subheader("⚠️ Key Risks")
                            for risk in overall['key_risks']:
                                st.warning(f"• {risk}")
                        
                        # Trading Recommendations
                        st.subheader("💡 Trading Recommendations")
                        
                        if overall['direction'] == 'BULLISH':
                            st.success("""
                            **Bullish Outlook Strategies:**
                            • Consider **Bull Call Spreads** for moderate upside
                            • **Cash Secured Puts** to enter on pullbacks
                            • **Covered Calls** if already holding stock
                            • Set stop loss around recent support levels
                            """)
                        elif overall['direction'] == 'BEARISH':
                            st.error("""
                            **Bearish Outlook Strategies:**
                            • Consider **Bear Put Spreads** for moderate downside
                            • **Protective Puts** if holding stock
                            • Avoid **Covered Calls** in strong downtrends
                            • Wait for oversold bounces to enter short positions
                            """)
                        else:
                            st.info("""
                            **Neutral Outlook Strategies:**
                            • **Iron Condors** for range-bound trading
                            • **Covered Calls** for income generation
                            • **Cash Secured Puts** at support levels
                            • Avoid directional bets until clearer signals emerge
                            """)
                        
                        # Export Prediction
                        st.subheader("📤 Export Prediction")
                        
                        prediction_export = {
                            'prediction_timestamp': datetime.now().isoformat(),
                            'ticker': prediction_result['ticker'],
                            'current_price': prediction_result['current_price'],
                            'prediction_period_days': prediction_days,
                            'overall_prediction': prediction_result['overall_prediction'],
                            'technical_signals': prediction_result['technical_signals'],
                            'volatility_forecast': prediction_result['volatility_forecast'],
                            'price_targets': prediction_result['price_targets']
                        }
                        
                        st.download_button(
                            "🔮 Download Prediction Report",
                            json.dumps(prediction_export, indent=2),
                            f"{prediction_result['ticker']}_market_prediction.json",
                            "application/json"
                        )
                    
                    else:
                        st.error(f"❌ Prediction failed: {prediction_result['error']}")
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    if debug_mode:
                        import traceback
                        st.code(traceback.format_exc())
        
        else:
            # Instructions for Predictions tab
            st.markdown("""
            ## 🔮 Market Predictions
            
            **AI-powered market direction prediction** using advanced technical analysis, volatility forecasting, and momentum indicators.
            
            ### 🎯 **Prediction Components:**
            
            **🔧 Technical Analysis:**
            - **RSI Momentum:** Overbought/oversold conditions
            - **Moving Averages:** Trend direction and strength
            - **MACD Signals:** Momentum crossovers and divergences
            - **Support/Resistance:** Key price levels to watch
            
            **📊 Volatility Forecasting:**
            - **Current vs Historical:** Volatility regime analysis
            - **GARCH-style Modeling:** Mean reversion forecasting
            - **Volatility Trends:** Increasing or decreasing patterns
            - **Risk Assessment:** Market uncertainty levels
            
            **🎯 Price Target Analysis:**
            - **ATR-based Targets:** Volatility-adjusted price levels
            - **Fibonacci Retracements:** Mathematical support/resistance
            - **Bollinger Bands:** Dynamic overbought/oversold levels
            - **Probability Estimates:** Likelihood of reaching targets
            
            **🌊 Momentum Indicators:**
            - **Rate of Change:** Multi-timeframe momentum
            - **Volume Analysis:** Confirmation of price moves
            - **Recent Highs/Lows:** Short-term momentum patterns
            - **Trend Strength:** Sustainability of current moves
            
            ### 📈 **Prediction Output:**
            
            **🎯 Overall Direction:**
            - **Bullish/Bearish/Neutral** with confidence percentages
            - **Time Horizon:** Short/Medium/Long-term outlook
            - **Signal Strength:** Very High/High/Moderate/Low
            - **Risk Assessment:** Key factors to monitor
            
            **📊 Detailed Analysis:**
            - **Support/Resistance Levels** with distance calculations
            - **Price Targets** with probability estimates
            - **Volatility Forecasts** for options strategies
            - **Trading Recommendations** based on prediction
            
            ### 💡 **How to Use:**
            1. **Enter symbol** (auto-populated from Analysis tab)
            2. **Set prediction horizon** (7-90 days)
            3. **Generate prediction** using real market data
            4. **Analyze components** to understand the forecast
            5. **Apply to trading** using strategy recommendations
            
            ### 🎯 **Strategy Integration:**
            - **Bullish Predictions:** Bull spreads, cash-secured puts
            - **Bearish Predictions:** Bear spreads, protective puts
            - **Neutral Predictions:** Iron condors, covered calls
            - **High Volatility:** Premium selling strategies
            - **Low Volatility:** Long volatility strategies
            
            ### ⚠️ **Important Disclaimers:**
            - **Predictions are probabilistic** - not guaranteed outcomes
            - **Based on technical analysis** - fundamental factors not included
            - **For educational purposes** - not investment advice
            - **Past patterns** may not predict future performance
            - **Always use proper risk management**
            
            ### 🔬 **Technical Methodology:**
            - **Multi-factor scoring** system combining all indicators
            - **Confidence intervals** based on signal alignment
            - **Historical backtesting** of prediction accuracy
            - **Real-time data integration** from Polygon API
            - **Professional-grade algorithms** used by institutions
            """)


if __name__ == "__main__":
    main()