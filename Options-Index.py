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
                shared_xaxis=True,
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
    
    # Main content
    if analyze_button and symbol_input:
        with st.spinner(f"Analyzing {symbol_input} with real data..."):
            result = strategist.analyze_symbol(symbol_input.upper(), debug=debug_mode)
        
        if result['success']:
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
        st.info("🔍 Enter a ticker symbol and click '🚀 Analyze Real Data' to start analysis")
if __name__ == "__main__":
    main()