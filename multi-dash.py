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

    def create_trading_chart(self, data: Dict, asset_class: str, support_resistance: Dict = None) -> go.Figure:
        """Create trading chart adapted for different asset classes with enhanced SMAs and S/R levels"""
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
                
                fig.update_layout(height=600)
                
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
                
                fig.update_layout(height=700)
            
            # Enhanced Moving averages - Professional set
            sma_configs = [
                {'period': 20, 'color': '#ff6b35', 'name': 'SMA 20', 'width': 2},
                {'period': 40, 'color': '#f7931e', 'name': 'SMA 40', 'width': 2},
                {'period': 50, 'color': '#004e89', 'name': 'SMA 50', 'width': 2.5},
                {'period': 100, 'color': '#7209b7', 'name': 'SMA 100', 'width': 2},
                {'period': 150, 'color': '#c1272d', 'name': 'SMA 150', 'width': 2},
                {'period': 200, 'color': '#ffffff', 'name': 'SMA 200', 'width': 3}
            ]
            
            for sma_config in sma_configs:
                period = sma_config['period']
                if len(df_last_year) >= period:
                    ma = df_last_year['close'].rolling(window=period).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df_last_year.index,
                            y=ma,
                            mode='lines',
                            name=sma_config['name'],
                            line=dict(
                                color=sma_config['color'], 
                                width=sma_config['width'],
                                dash='solid' if period in [20, 50, 200] else 'dot'
                            ),
                            opacity=0.9 if period in [20, 50, 200] else 0.7
                        ),
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
            
            # Add Support and Resistance levels if provided
            if support_resistance:
                # Support level
                if 'support_level' in support_resistance:
                    support = support_resistance['support_level']
                    fig.add_hline(
                        y=support,
                        line_dash="dash",
                        line_color="#00ff00",
                        line_width=2,
                        annotation_text=f"Support: {support:.4f if asset_class == 'FOREX' else f'${support:.2f}'}",
                        annotation_position="bottom left",
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
                
                # Resistance level
                if 'resistance_level' in support_resistance:
                    resistance = support_resistance['resistance_level']
                    fig.add_hline(
                        y=resistance,
                        line_dash="dash",
                        line_color="#ff0000",
                        line_width=2,
                        annotation_text=f"Resistance: {resistance:.4f if asset_class == 'FOREX' else f'${resistance:.2f}'}",
                        annotation_position="top left",
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
                
                # Target price (from predictions)
                if 'target_price' in support_resistance:
                    target = support_resistance['target_price']
                    fig.add_hline(
                        y=target,
                        line_dash="dot",
                        line_color="#ffff00",
                        line_width=2,
                        annotation_text=f"Target: {target:.4f if asset_class == 'FOREX' else f'${target:.2f}'}",
                        annotation_position="top right",
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
                
                # Bull case
                if 'bull_case' in support_resistance:
                    bull = support_resistance['bull_case']
                    fig.add_hline(
                        y=bull,
                        line_dash="dot",
                        line_color="#00ffff",
                        line_width=1.5,
                        annotation_text=f"Bull: {bull:.4f if asset_class == 'FOREX' else f'${bull:.2f}'}",
                        annotation_position="top right",
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
                
                # Bear case
                if 'bear_case' in support_resistance:
                    bear = support_resistance['bear_case']
                    fig.add_hline(
                        y=bear,
                        line_dash="dot",
                        line_color="#ff00ff",
                        line_width=1.5,
                        annotation_text=f"Bear: {bear:.4f if asset_class == 'FOREX' else f'${bear:.2f}'}",
                        annotation_position="bottom right",
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
            
            # Update layout
            fig.update_layout(
                title=chart_title,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add current price annotation
            current_price = data['current_price']
            fig.add_hline(
                y=current_price,
                line_dash="solid",
                line_color="white",
                line_width=2,
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
    
    def create_enhanced_trading_chart(self, data: Dict, asset_class: str, support_resistance: Dict = None, sma_config: Dict = None) -> go.Figure:
        """Create enhanced trading chart with configurable SMAs and S/R levels"""
        try:
            df = data['historical_data'].copy()
            
            # Asset-specific chart customizations
            if asset_class == 'FOREX':
                chart_title = f'{data["ticker"]} - FX Pair (Last Year)'
                price_annotation = f"Current: {data['current_price']:.5f}"
            else:
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
                fig = go.Figure()
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
                fig.update_layout(height=700)
            else:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=(chart_title, 'Volume'),
                    row_heights=[0.7, 0.3]
                )
                
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
                
                fig.update_layout(height=800)
            
            # Configurable Moving Averages
            sma_configs = [
                {'period': 20, 'color': '#ff6b35', 'name': 'SMA 20', 'width': 2, 'show': sma_config.get('show_sma_20', True)},
                {'period': 40, 'color': '#f7931e', 'name': 'SMA 40', 'width': 2, 'show': sma_config.get('show_sma_40', True)},
                {'period': 50, 'color': '#004e89', 'name': 'SMA 50', 'width': 2.5, 'show': sma_config.get('show_sma_50', True)},
                {'period': 100, 'color': '#7209b7', 'name': 'SMA 100', 'width': 2, 'show': sma_config.get('show_sma_100', True)},
                {'period': 150, 'color': '#c1272d', 'name': 'SMA 150', 'width': 2, 'show': sma_config.get('show_sma_150', True)},
                {'period': 200, 'color': '#ffffff', 'name': 'SMA 200', 'width': 3, 'show': sma_config.get('show_sma_200', True)}
            ]
            
            for sma_config_item in sma_configs:
                if sma_config_item['show'] and len(df_last_year) >= sma_config_item['period']:
                    ma = df_last_year['close'].rolling(window=sma_config_item['period']).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df_last_year.index,
                            y=ma,
                            mode='lines',
                            name=sma_config_item['name'],
                            line=dict(
                                color=sma_config_item['color'], 
                                width=sma_config_item['width'],
                                dash='solid' if sma_config_item['period'] in [20, 50, 200] else 'dot'
                            ),
                            opacity=0.9 if sma_config_item['period'] in [20, 50, 200] else 0.7
                        ),
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
            
            # Add Support and Resistance levels
            if support_resistance:
                if 'support_level' in support_resistance:
                    support = support_resistance['support_level']
                    fig.add_hline(
                        y=support,
                        line_dash="dash",
                        line_color="#00ff00",
                        line_width=2,
                        annotation_text=f"Support: {support:.4f if asset_class == 'FOREX' else f'${support:.2f}'}",
                        annotation_position="bottom left",
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
                
                if 'resistance_level' in support_resistance:
                    resistance = support_resistance['resistance_level']
                    fig.add_hline(
                        y=resistance,
                        line_dash="dash",
                        line_color="#ff0000",
                        line_width=2,
                        annotation_text=f"Resistance: {resistance:.4f if asset_class == 'FOREX' else f'${resistance:.2f}'}",
                        annotation_position="top left",
                        row=1, col=1 if asset_class != 'FOREX' else None
                    )
                
                # Prediction levels
                for level_name, color, position in [
                    ('target_price', '#ffff00', 'top right'),
                    ('bull_case', '#00ffff', 'top right'),
                    ('bear_case', '#ff00ff', 'bottom right')
                ]:
                    if level_name in support_resistance:
                        level_value = support_resistance[level_name]
                        fig.add_hline(
                            y=level_value,
                            line_dash="dot",
                            line_color=color,
                            line_width=1.5,
                            annotation_text=f"{level_name.replace('_', ' ').title()}: {level_value:.4f if asset_class == 'FOREX' else f'${level_value:.2f}'}",
                            annotation_position=position,
                            row=1, col=1 if asset_class != 'FOREX' else None
                        )
            
            # Current price line (configurable)
            if sma_config.get('show_current_price', True):
                current_price = data['current_price']
                fig.add_hline(
                    y=current_price,
                    line_dash="solid",
                    line_color="white",
                    line_width=2,
                    annotation_text=price_annotation,
                    annotation_position="bottom right",
                    row=1, col=1 if asset_class != 'FOREX' else None
                )
            
            # Update layout
            fig.update_layout(
                title=chart_title,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating enhanced {asset_class} chart: {e}")
            return go.Figure().add_annotation(
                text=f"Enhanced {asset_class} chart could not be generated",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def create_technical_analysis_chart(self, ticker: str, asset_class: str, prediction_data: Dict = None) -> go.Figure:
        """Create a dedicated technical analysis chart with all SMAs and prediction levels"""
        try:
            # Get underlying data
            underlying_data = self.get_asset_data(ticker, asset_class, days=365)
            
            chart_data = {
                'ticker': ticker,
                'current_price': underlying_data['current_price'],
                'historical_data': underlying_data['historical_data']
            }
            
            # Prepare support/resistance levels
            support_resistance = {
                'support_level': underlying_data['low_52w'],
                'resistance_level': underlying_data['high_52w']
            }
            
            # Add prediction levels if available
            if prediction_data and 'price_predictions' in prediction_data:
                predictions = prediction_data['price_predictions']
                support_resistance.update({
                    'target_price': predictions.get('target_price'),
                    'bull_case': predictions.get('bull_case'),
                    'bear_case': predictions.get('bear_case')
                })
            
            return self.create_trading_chart(chart_data, asset_class, support_resistance)
            
        except Exception as e:
            print(f"Error creating technical analysis chart: {e}")
            return go.Figure().add_annotation(
                text=f"Technical chart could not be generated",
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
    
    def get_options_data_with_expiry(self, underlying_ticker: str, asset_class: str, current_price: float = None, start_date=None, end_date=None) -> Dict:
        """Get options data for specific expiry date range"""
        try:
            print(f"🎯 Fetching options data for {underlying_ticker} ({asset_class}) between {start_date} and {end_date}...")
            
            # Options contracts use original ticker (not formatted)
            contracts = []
            for contract in self.client.list_options_contracts(
                underlying_ticker=underlying_ticker,
                expiration_date_gte=start_date.strftime("%Y-%m-%d"),
                expiration_date_lte=end_date.strftime("%Y-%m-%d"),
                limit=1000
            ):
                contracts.append(contract)
            
            if not contracts:
                raise ValueError(f"No options contracts found for {underlying_ticker} between {start_date} and {end_date}")
            
            print(f"Found {len(contracts)} options contracts for selected expiry range")
            
            # Get current underlying price if not provided
            if current_price is None:
                try:
                    end_date_fetch = datetime.now()
                    start_date_fetch = end_date_fetch - timedelta(days=5)
                    formatted_ticker = self._format_ticker(underlying_ticker, asset_class)
                    
                    recent_aggs = []
                    for agg in self.client.list_aggs(
                        formatted_ticker,
                        1,
                        "day",
                        start_date_fetch.strftime("%Y-%m-%d"),
                        end_date_fetch.strftime("%Y-%m-%d"),
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

    def _get_available_expiries(self, ticker: str, asset_class: str) -> List[str]:
        """Get all available expiration dates for a symbol"""
        try:
            contracts = []
            # Get more contracts to see all available expiries
            for contract in self.client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date_gte=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                expiration_date_lte=(datetime.now() + timedelta(days=1800)).strftime("%Y-%m-%d"),  # ~5 years
                limit=2000
            ):
                contracts.append(contract)
            
            # Extract unique expiration dates
            expiries = set()
            for contract in contracts:
                expiries.add(contract.expiration_date)
            
            # Sort expiries
            sorted_expiries = sorted(list(expiries))
            return sorted_expiries[:20]  # Return first 20 expiries
            
        except Exception as e:
            print(f"Could not get available expiries: {e}")
            return []
    
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
    
    def select_strategy(self, market_analysis: Dict, underlying_data: Dict, options_data: Dict) -> Dict:
        """Select optimal strategy with detailed explanations for confidence scores"""
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
        rsi = market_analysis['rsi']
        
        scores = {}
        explanations = {}  # NEW: Store explanations for each strategy
        
        # Asset-specific strategy scoring adjustments
        vol_multiplier = 1.5 if asset_class == 'FOREX' else 1.0  # FX options strategies
        
        # Covered Call
        if len(liquid_calls) >= 1:
            base_score = 7.0
            explanation_parts = [f"Base score: {base_score}/10 (Income generation strategy)"]
            
            if trend in ['SIDEWAYS', 'SIDEWAYS_BULLISH']:
                bonus = 1.5 * vol_multiplier
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {trend.lower().replace('_', ' ')} trend (ideal for covered calls)")
            
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                bonus = 1.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {vol_regime.lower().replace('_', ' ')} environment (higher premiums)")
            
            if momentum in ['OVERBOUGHT', 'EXTREMELY_OVERBOUGHT']:
                bonus = 0.8
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {momentum.lower().replace('_', ' ')} momentum (good for income)")
            
            if rsi > 70:
                explanation_parts.append(f"RSI at {rsi:.1f} suggests potential pullback, favorable for calls")
            
            scores['COVERED_CALL'] = base_score
            explanations['COVERED_CALL'] = explanation_parts
        
        # Cash Secured Put
        if len(liquid_puts) >= 1:
            base_score = 7.0
            explanation_parts = [f"Base score: {base_score}/10 (Acquisition strategy at discount)"]
            
            if trend in ['BULLISH', 'STRONG_BULLISH', 'SIDEWAYS_BULLISH']:
                bonus = 1.5 * vol_multiplier
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {trend.lower().replace('_', ' ')} trend (good for acquiring assets)")
            
            if momentum in ['OVERSOLD', 'EXTREMELY_OVERSOLD']:
                bonus = 1.2
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {momentum.lower().replace('_', ' ')} momentum (potential bounce)")
            
            if bb_signal in ['LOWER_BAND', 'EXTREME_LOWER']:
                bonus = 1.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {bb_signal.lower().replace('_', ' ')} position (near support)")
            
            if rsi < 40:
                explanation_parts.append(f"RSI at {rsi:.1f} suggests oversold conditions, favorable for puts")
            
            scores['CASH_SECURED_PUT'] = base_score
            explanations['CASH_SECURED_PUT'] = explanation_parts
        
        # Iron Condor (especially good for FX)
        if len(liquid_calls) >= 2 and len(liquid_puts) >= 2:
            base_score = 6.5
            explanation_parts = [f"Base score: {base_score}/10 (Range-bound strategy)"]
            
            if asset_class == 'FOREX':
                bonus = 1.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {asset_class} (FX pairs tend to range-trade)")
            
            if trend in ['SIDEWAYS', 'SIDEWAYS_BULLISH', 'SIDEWAYS_BEARISH']:
                bonus = 2.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {trend.lower().replace('_', ' ')} trend (perfect for condors)")
            
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                bonus = 1.5
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {vol_regime.lower().replace('_', ' ')} (high premium collection)")
            
            if bb_signal == 'MIDDLE_RANGE':
                bonus = 1.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for middle BB range (price stability)")
            
            if 45 <= rsi <= 55:
                explanation_parts.append(f"RSI at {rsi:.1f} shows neutral momentum, ideal for range strategies")
            
            scores['IRON_CONDOR'] = base_score
            explanations['IRON_CONDOR'] = explanation_parts
        
        # Bull Call Spread
        if len(liquid_calls) >= 2:
            base_score = 6.0
            explanation_parts = [f"Base score: {base_score}/10 (Moderately bullish strategy)"]
            
            if trend in ['BULLISH', 'STRONG_BULLISH']:
                bonus = 2.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {trend.lower().replace('_', ' ')} trend (aligned with strategy)")
            
            if momentum in ['BULLISH', 'OVERSOLD']:
                bonus = 1.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {momentum.lower().replace('_', ' ')} momentum")
            
            if vol_regime in ['NORMAL_VOL', 'LOW_VOL']:
                bonus = 0.8
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {vol_regime.lower().replace('_', ' ')} (lower time decay)")
            
            if rsi < 60:
                explanation_parts.append(f"RSI at {rsi:.1f} shows room for upward movement")
            
            scores['BULL_CALL_SPREAD'] = base_score
            explanations['BULL_CALL_SPREAD'] = explanation_parts
        
        # Bear Put Spread
        if len(liquid_puts) >= 2:
            base_score = 6.0
            explanation_parts = [f"Base score: {base_score}/10 (Moderately bearish strategy)"]
            
            if trend in ['BEARISH', 'STRONG_BEARISH']:
                bonus = 2.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {trend.lower().replace('_', ' ')} trend (aligned with strategy)")
            
            if momentum in ['BEARISH', 'OVERBOUGHT']:
                bonus = 1.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {momentum.lower().replace('_', ' ')} momentum")
            
            if vol_regime in ['NORMAL_VOL', 'LOW_VOL']:
                bonus = 0.8
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {vol_regime.lower().replace('_', ' ')} (lower time decay)")
            
            if rsi > 50:
                explanation_parts.append(f"RSI at {rsi:.1f} shows potential for downward movement")
            
            scores['BEAR_PUT_SPREAD'] = base_score
            explanations['BEAR_PUT_SPREAD'] = explanation_parts
        
        # Long Straddle (good for volatile markets)
        if len(liquid_calls) >= 1 and len(liquid_puts) >= 1:
            base_score = 5.5
            explanation_parts = [f"Base score: {base_score}/10 (Volatility strategy)"]
            
            if vol_regime == 'LOW_VOL':
                bonus = 2.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {vol_regime.lower().replace('_', ' ')} (expecting volatility expansion)")
            
            if trend == 'SIDEWAYS':
                bonus = 1.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {trend.lower()} trend (breakout potential)")
            
            if bb_signal == 'MIDDLE_RANGE':
                bonus = 0.8
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for middle BB range (compression before move)")
            
            if asset_class == 'FOREX' and vol_regime == 'LOW_VOL':
                bonus = 0.5
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {asset_class} low volatility (central bank event potential)")
            
            scores['LONG_STRADDLE'] = base_score
            explanations['LONG_STRADDLE'] = explanation_parts
        
        # Protective Put
        if len(liquid_puts) >= 1:
            base_score = 5.0
            explanation_parts = [f"Base score: {base_score}/10 (Insurance strategy)"]
            
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                bonus = 1.5
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {vol_regime.lower().replace('_', ' ')} (protection needed)")
            
            if trend in ['BEARISH', 'STRONG_BEARISH']:
                bonus = 1.5
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {trend.lower().replace('_', ' ')} trend (downside risk)")
            
            if momentum in ['BEARISH', 'EXTREMELY_OVERBOUGHT']:
                bonus = 1.0
                base_score += bonus
                explanation_parts.append(f"+{bonus:.1f} for {momentum.lower().replace('_', ' ')} momentum")
            
            if rsi > 70:
                explanation_parts.append(f"RSI at {rsi:.1f} suggests potential correction, protection warranted")
            
            scores['PROTECTIVE_PUT'] = base_score
            explanations['PROTECTIVE_PUT'] = explanation_parts
        
        if not scores:
            raise ValueError("No viable strategies found for current market conditions")
        
        # Return both scores and explanations
        return {
            'scores': dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]),
            'explanations': explanations
        }
    
    def calculate_optimal_contracts(self, strategy_name: str, market_analysis: Dict, 
                                  underlying_data: Dict, options_data: Dict, 
                                  available_capital: float = 10000) -> Dict:
        """Calculate optimal contract specifications for maximum profitability"""
        
        calls = options_data.get('calls', pd.DataFrame())
        puts = options_data.get('puts', pd.DataFrame())
        current_price = underlying_data['current_price']
        asset_class = underlying_data.get('asset_class', 'EQUITIES')
        
        if calls.empty or puts.empty:
            raise ValueError("Insufficient options data for contract calculations")
        
        # Filter for liquid options
        liquid_calls = calls[calls['lastPrice'] > (0.01 if asset_class == 'FOREX' else 0.05)]
        liquid_puts = puts[puts['lastPrice'] > (0.01 if asset_class == 'FOREX' else 0.05)]
        
        # Get strategy-specific calculations
        if strategy_name == 'COVERED_CALL':
            return self._calculate_covered_call_optimal(
                liquid_calls, current_price, available_capital, asset_class, market_analysis, options_data
            )
        elif strategy_name == 'CASH_SECURED_PUT':
            return self._calculate_csp_optimal(
                liquid_puts, current_price, available_capital, asset_class, market_analysis, options_data
            )
        elif strategy_name == 'IRON_CONDOR':
            return self._calculate_iron_condor_optimal(
                liquid_calls, liquid_puts, current_price, available_capital, asset_class, market_analysis, options_data
            )
        elif strategy_name == 'BULL_CALL_SPREAD':
            return self._calculate_bull_call_optimal(
                liquid_calls, current_price, available_capital, asset_class, market_analysis, options_data
            )
        elif strategy_name == 'BEAR_PUT_SPREAD':
            return self._calculate_bear_put_optimal(
                liquid_puts, current_price, available_capital, asset_class, market_analysis, options_data
            )
        elif strategy_name == 'LONG_STRADDLE':
            return self._calculate_straddle_optimal(
                liquid_calls, liquid_puts, current_price, available_capital, asset_class, market_analysis, options_data
            )
        elif strategy_name == 'PROTECTIVE_PUT':
            return self._calculate_protective_put_optimal(
                liquid_puts, current_price, available_capital, asset_class, market_analysis, options_data
            )
        else:
            return {'error': f"Contract calculations not implemented for {strategy_name}"}
    
    def _calculate_covered_call_optimal(self, calls_df: pd.DataFrame, current_price: float,
                                      capital: float, asset_class: str, market_analysis: Dict, options_data: Dict) -> Dict:
        """Calculate optimal covered call specifications"""
        
        # Determine how many shares we can afford
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        shares_affordable = int(capital / (current_price * share_equivalent))
        
        if shares_affordable == 0:
            return {
                'error': f'Insufficient capital. Need ${current_price * share_equivalent:.2f} for 1 contract',
                'min_capital_needed': current_price * share_equivalent
            }
        
        # Find optimal strike (typically 5-15% OTM)
        target_strikes = []
        
        # Conservative: 2-5% OTM
        target_strikes.append(('Conservative', current_price * 1.02, current_price * 1.05))
        # Moderate: 5-10% OTM  
        target_strikes.append(('Moderate', current_price * 1.05, current_price * 1.10))
        # Aggressive: 10-15% OTM
        target_strikes.append(('Aggressive', current_price * 1.10, current_price * 1.15))
        
        recommendations = []
        
        for risk_level, min_strike, max_strike in target_strikes:
            suitable_calls = calls_df[
                (calls_df['strike'] >= min_strike) & 
                (calls_df['strike'] <= max_strike)
            ].copy()
            
            if suitable_calls.empty:
                continue
            
            # Calculate metrics for each option
            suitable_calls['premium_yield'] = (suitable_calls['lastPrice'] / current_price) * 100
            suitable_calls['max_gain'] = suitable_calls['strike'] - current_price + suitable_calls['lastPrice']
            suitable_calls['max_gain_pct'] = (suitable_calls['max_gain'] / current_price) * 100
            suitable_calls['protection'] = (suitable_calls['lastPrice'] / current_price) * 100
            
            # Find best option (highest premium yield with reasonable max gain)
            best_option = suitable_calls.loc[suitable_calls['premium_yield'].idxmax()]
            
            # Calculate contract specifications
            max_contracts = min(shares_affordable, int(capital * 0.8 / (current_price * share_equivalent)))
            
            total_stock_cost = max_contracts * current_price * share_equivalent
            total_premium_received = max_contracts * best_option['lastPrice'] * share_equivalent
            net_investment = total_stock_cost - total_premium_received
            
            recommendations.append({
                'risk_level': risk_level,
                'strike': best_option['strike'],
                'premium': best_option['lastPrice'],
                'contracts': max_contracts,
                'total_stock_cost': total_stock_cost,
                'total_premium_received': total_premium_received,
                'net_investment': net_investment,
                'max_profit': total_premium_received + max_contracts * (best_option['strike'] - current_price) * share_equivalent,
                'max_profit_pct': ((total_premium_received + max_contracts * (best_option['strike'] - current_price) * share_equivalent) / net_investment) * 100,
                'breakeven': current_price - best_option['lastPrice'],
                'premium_yield_annualized': (best_option['premium_yield'] * 365 / options_data.get('days_to_expiry', 30)),
                'downside_protection_pct': best_option['protection']
            })
        
        # Add market-specific insights
        market_insight = ""
        if market_analysis['trend'] in ['SIDEWAYS', 'SIDEWAYS_BULLISH']:
            market_insight = "Excellent environment for covered calls - sideways movement maximizes time decay"
        elif market_analysis['volatility_regime'] in ['HIGH_VOL', 'EXTREME_VOL']:
            market_insight = "High volatility increases premium income but also assignment risk"
        
        return {
            'strategy': 'COVERED_CALL',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'market_insight': market_insight,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def _calculate_csp_optimal(self, puts_df: pd.DataFrame, current_price: float,
                             capital: float, asset_class: str, market_analysis: Dict, options_data: Dict) -> Dict:
        """Calculate optimal cash secured put specifications"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Find optimal strikes (typically 5-15% OTM puts)
        target_strikes = []
        
        # Conservative: 2-5% OTM
        target_strikes.append(('Conservative', current_price * 0.95, current_price * 0.98))
        # Moderate: 5-10% OTM  
        target_strikes.append(('Moderate', current_price * 0.90, current_price * 0.95))
        # Aggressive: 10-15% OTM
        target_strikes.append(('Aggressive', current_price * 0.85, current_price * 0.90))
        
        recommendations = []
        
        for risk_level, min_strike, max_strike in target_strikes:
            suitable_puts = puts_df[
                (puts_df['strike'] >= min_strike) & 
                (puts_df['strike'] <= max_strike)
            ].copy()
            
            if suitable_puts.empty:
                continue
            
            # Calculate metrics
            suitable_puts['premium_yield'] = (suitable_puts['lastPrice'] / suitable_puts['strike']) * 100
            suitable_puts['effective_price'] = suitable_puts['strike'] - suitable_puts['lastPrice']
            suitable_puts['discount_to_current'] = ((current_price - suitable_puts['effective_price']) / current_price) * 100
            
            # Find best option (highest premium yield)
            best_option = suitable_puts.loc[suitable_puts['premium_yield'].idxmax()]
            
            # Calculate how many contracts we can afford
            cash_required_per_contract = best_option['strike'] * share_equivalent
            max_contracts = int(capital / cash_required_per_contract)
            
            if max_contracts == 0:
                continue
            
            total_cash_secured = max_contracts * cash_required_per_contract
            total_premium_received = max_contracts * best_option['lastPrice'] * share_equivalent
            
            recommendations.append({
                'risk_level': risk_level,
                'strike': best_option['strike'],
                'premium': best_option['lastPrice'],
                'contracts': max_contracts,
                'cash_required': total_cash_secured,
                'premium_received': total_premium_received,
                'effective_buy_price': best_option['effective_price'],
                'discount_pct': best_option['discount_to_current'],
                'max_profit': total_premium_received,
                'max_profit_pct': (total_premium_received / total_cash_secured) * 100,
                'max_loss': max_contracts * best_option['effective_price'] * share_equivalent,
                'breakeven': best_option['strike'] - best_option['lastPrice'],
                'premium_yield_annualized': (best_option['premium_yield'] * 365 / options_data.get('days_to_expiry', 30))
            })
        
        # Market insight
        market_insight = ""
        if market_analysis['momentum'] in ['OVERSOLD', 'EXTREMELY_OVERSOLD']:
            market_insight = "Oversold conditions suggest good entry point for cash secured puts"
        elif market_analysis['trend'] in ['BULLISH', 'STRONG_BULLISH']:
            market_insight = "Bullish trend supports selling puts to acquire assets at discount"
        
        return {
            'strategy': 'CASH_SECURED_PUT',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'market_insight': market_insight,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def _calculate_iron_condor_optimal(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                     current_price: float, capital: float, asset_class: str,
                                     market_analysis: Dict, options_data: Dict) -> Dict:
        """Calculate optimal iron condor specifications"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Define wing widths based on asset class
        if asset_class == 'FOREX':
            wing_widths = [0.02, 0.03, 0.04]  # 2%, 3%, 4% for FX
        else:
            wing_widths = [0.03, 0.05, 0.07]  # 3%, 5%, 7% for equities/indices
        
        recommendations = []
        
        for i, wing_width in enumerate(wing_widths):
            risk_level = ['Conservative', 'Moderate', 'Aggressive'][i]
            
            # Define strikes
            call_sell_strike = current_price * (1 + wing_width)
            call_buy_strike = current_price * (1 + wing_width * 2)
            put_sell_strike = current_price * (1 - wing_width)
            put_buy_strike = current_price * (1 - wing_width * 2)
            
            # Find closest available strikes
            call_sell = calls_df.iloc[(calls_df['strike'] - call_sell_strike).abs().argsort()[:1]]
            call_buy = calls_df.iloc[(calls_df['strike'] - call_buy_strike).abs().argsort()[:1]]
            put_sell = puts_df.iloc[(puts_df['strike'] - put_sell_strike).abs().argsort()[:1]]
            put_buy = puts_df.iloc[(puts_df['strike'] - put_buy_strike).abs().argsort()[:1]]
            
            if len(call_sell) == 0 or len(call_buy) == 0 or len(put_sell) == 0 or len(put_buy) == 0:
                continue
            
            # Calculate net credit
            net_credit_per_contract = (
                call_sell.iloc[0]['lastPrice'] + put_sell.iloc[0]['lastPrice'] -
                call_buy.iloc[0]['lastPrice'] - put_buy.iloc[0]['lastPrice']
            )
            
            if net_credit_per_contract <= 0:
                continue
            
            # Calculate max loss and required margin
            call_spread_width = call_buy.iloc[0]['strike'] - call_sell.iloc[0]['strike']
            put_spread_width = put_sell.iloc[0]['strike'] - put_buy.iloc[0]['strike']
            max_loss_per_contract = max(call_spread_width, put_spread_width) - net_credit_per_contract
            
            # Margin requirement (approximation)
            margin_per_contract = max_loss_per_contract * share_equivalent
            max_contracts = int(capital / margin_per_contract)
            
            if max_contracts == 0:
                continue
            
            total_credit = max_contracts * net_credit_per_contract * share_equivalent
            total_max_loss = max_contracts * max_loss_per_contract * share_equivalent
            
            # Calculate profit zones
            upper_breakeven = call_sell.iloc[0]['strike'] + net_credit_per_contract
            lower_breakeven = put_sell.iloc[0]['strike'] - net_credit_per_contract
            
            recommendations.append({
                'risk_level': risk_level,
                'call_sell_strike': call_sell.iloc[0]['strike'],
                'call_buy_strike': call_buy.iloc[0]['strike'],
                'put_sell_strike': put_sell.iloc[0]['strike'], 
                'put_buy_strike': put_buy.iloc[0]['strike'],
                'net_credit_per_contract': net_credit_per_contract,
                'contracts': max_contracts,
                'total_credit': total_credit,
                'max_profit': total_credit,
                'max_loss': total_max_loss,
                'max_profit_pct': (total_credit / (total_credit + total_max_loss)) * 100,
                'upper_breakeven': upper_breakeven,
                'lower_breakeven': lower_breakeven,
                'profit_range_pct': ((upper_breakeven - lower_breakeven) / current_price) * 100,
                'margin_required': max_contracts * margin_per_contract
            })
        
        # Market insight
        market_insight = ""
        if market_analysis['trend'] in ['SIDEWAYS', 'SIDEWAYS_BULLISH', 'SIDEWAYS_BEARISH']:
            market_insight = "Perfect sideways market for iron condors - high probability of profit"
        elif market_analysis['volatility_regime'] in ['HIGH_VOL', 'EXTREME_VOL']:
            market_insight = "High volatility provides excellent premium collection opportunity"
        
        return {
            'strategy': 'IRON_CONDOR',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'market_insight': market_insight,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def _calculate_bull_call_optimal(self, calls_df: pd.DataFrame, current_price: float,
                                   capital: float, asset_class: str, market_analysis: Dict, options_data: Dict) -> Dict:
        """Calculate optimal bull call spread specifications"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Define spread widths
        spread_widths = [0.05, 0.08, 0.12]  # 5%, 8%, 12%
        
        recommendations = []
        
        for i, spread_width in enumerate(spread_widths):
            risk_level = ['Conservative', 'Moderate', 'Aggressive'][i]
            
            # ATM or slightly ITM buy, OTM sell
            buy_strike = current_price * 0.98  # Slightly ITM
            sell_strike = current_price * (1 + spread_width)
            
            # Find closest strikes
            buy_call = calls_df.iloc[(calls_df['strike'] - buy_strike).abs().argsort()[:1]]
            sell_call = calls_df.iloc[(calls_df['strike'] - sell_strike).abs().argsort()[:1]]
            
            if len(buy_call) == 0 or len(sell_call) == 0:
                continue
            
            # Calculate net debit
            net_debit_per_contract = buy_call.iloc[0]['lastPrice'] - sell_call.iloc[0]['lastPrice']
            
            if net_debit_per_contract <= 0:
                continue
            
            # Calculate max contracts affordable
            cost_per_contract = net_debit_per_contract * share_equivalent
            max_contracts = int(capital / cost_per_contract)
            
            if max_contracts == 0:
                continue
            
            total_cost = max_contracts * cost_per_contract
            strike_width = sell_call.iloc[0]['strike'] - buy_call.iloc[0]['strike']
            max_profit_per_contract = strike_width - net_debit_per_contract
            total_max_profit = max_contracts * max_profit_per_contract * share_equivalent
            
            breakeven = buy_call.iloc[0]['strike'] + net_debit_per_contract
            
            recommendations.append({
                'risk_level': risk_level,
                'buy_strike': buy_call.iloc[0]['strike'],
                'sell_strike': sell_call.iloc[0]['strike'],
                'net_debit_per_contract': net_debit_per_contract,
                'contracts': max_contracts,
                'total_cost': total_cost,
                'max_profit': total_max_profit,
                'max_loss': total_cost,
                'max_profit_pct': (total_max_profit / total_cost) * 100,
                'breakeven': breakeven,
                'breakeven_move_required_pct': ((breakeven - current_price) / current_price) * 100,
                'target_price': sell_call.iloc[0]['strike']
            })
        
        # Market insight
        market_insight = ""
        if market_analysis['trend'] in ['BULLISH', 'STRONG_BULLISH']:
            market_insight = "Strong bullish trend aligns perfectly with bull call spread strategy"
        elif market_analysis['momentum'] in ['OVERSOLD', 'EXTREMELY_OVERSOLD']:
            market_insight = "Oversold momentum suggests potential bounce - good for bull spreads"
        
        return {
            'strategy': 'BULL_CALL_SPREAD',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'market_insight': market_insight,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def _calculate_bear_put_optimal(self, puts_df: pd.DataFrame, current_price: float,
                                  capital: float, asset_class: str, market_analysis: Dict, options_data: Dict) -> Dict:
        """Calculate optimal bear put spread specifications"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Define spread widths
        spread_widths = [0.05, 0.08, 0.12]  # 5%, 8%, 12%
        
        recommendations = []
        
        for i, spread_width in enumerate(spread_widths):
            risk_level = ['Conservative', 'Moderate', 'Aggressive'][i]
            
            # ATM or slightly ITM buy, OTM sell
            buy_strike = current_price * 1.02  # Slightly ITM
            sell_strike = current_price * (1 - spread_width)
            
            # Find closest strikes
            buy_put = puts_df.iloc[(puts_df['strike'] - buy_strike).abs().argsort()[:1]]
            sell_put = puts_df.iloc[(puts_df['strike'] - sell_strike).abs().argsort()[:1]]
            
            if len(buy_put) == 0 or len(sell_put) == 0:
                continue
            
            # Calculate net debit
            net_debit_per_contract = buy_put.iloc[0]['lastPrice'] - sell_put.iloc[0]['lastPrice']
            
            if net_debit_per_contract <= 0:
                continue
            
            # Calculate max contracts affordable
            cost_per_contract = net_debit_per_contract * share_equivalent
            max_contracts = int(capital / cost_per_contract)
            
            if max_contracts == 0:
                continue
            
            total_cost = max_contracts * cost_per_contract
            strike_width = buy_put.iloc[0]['strike'] - sell_put.iloc[0]['strike']
            max_profit_per_contract = strike_width - net_debit_per_contract
            total_max_profit = max_contracts * max_profit_per_contract * share_equivalent
            
            breakeven = buy_put.iloc[0]['strike'] - net_debit_per_contract
            
            recommendations.append({
                'risk_level': risk_level,
                'buy_strike': buy_put.iloc[0]['strike'],
                'sell_strike': sell_put.iloc[0]['strike'],
                'net_debit_per_contract': net_debit_per_contract,
                'contracts': max_contracts,
                'total_cost': total_cost,
                'max_profit': total_max_profit,
                'max_loss': total_cost,
                'max_profit_pct': (total_max_profit / total_cost) * 100,
                'breakeven': breakeven,
                'breakeven_move_required_pct': ((current_price - breakeven) / current_price) * 100,
                'target_price': sell_put.iloc[0]['strike']
            })
        
        # Market insight
        market_insight = ""
        if market_analysis['trend'] in ['BEARISH', 'STRONG_BEARISH']:
            market_insight = "Strong bearish trend aligns perfectly with bear put spread strategy"
        elif market_analysis['momentum'] in ['OVERBOUGHT', 'EXTREMELY_OVERBOUGHT']:
            market_insight = "Overbought momentum suggests potential correction - good for bear spreads"
        
        return {
            'strategy': 'BEAR_PUT_SPREAD',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'market_insight': market_insight,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def _calculate_straddle_optimal(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                  current_price: float, capital: float, asset_class: str,
                                  market_analysis: Dict, options_data: Dict) -> Dict:
        """Calculate optimal long straddle specifications"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Find ATM options
        atm_call = calls_df.iloc[(calls_df['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts_df.iloc[(puts_df['strike'] - current_price).abs().argsort()[:1]]
        
        if len(atm_call) == 0 or len(atm_put) == 0:
            return {'error': 'No ATM options available for straddle'}
        
        # Use the same strike for both (closest to ATM)
        strike = atm_call.iloc[0]['strike']
        call_price = atm_call.iloc[0]['lastPrice']
        
        # Find put at same strike
        same_strike_put = puts_df[puts_df['strike'] == strike]
        if same_strike_put.empty:
            # Use closest put strike
            put_price = atm_put.iloc[0]['lastPrice']
            put_strike = atm_put.iloc[0]['strike']
        else:
            put_price = same_strike_put.iloc[0]['lastPrice']
            put_strike = strike
        
        # Calculate straddle metrics
        total_premium_per_contract = call_price + put_price
        cost_per_contract = total_premium_per_contract * share_equivalent
        max_contracts = int(capital / cost_per_contract)
        
        if max_contracts == 0:
            return {
                'error': f'Insufficient capital. Need ${cost_per_contract:.2f} for 1 straddle',
                'min_capital_needed': cost_per_contract
            }
        
        total_cost = max_contracts * cost_per_contract
        
        # Breakeven points
        upper_breakeven = strike + total_premium_per_contract
        lower_breakeven = strike - total_premium_per_contract
        
        # Required move for profitability
        required_move_pct = (total_premium_per_contract / current_price) * 100
        
        # Estimate probability of profit based on volatility
        realized_vol = market_analysis.get('realized_vol', 0.25)
        days_to_expiry = options_data.get('days_to_expiry', 30)
        
        # Expected move (1 standard deviation)
        expected_move = current_price * realized_vol * math.sqrt(days_to_expiry / 365)
        expected_move_pct = (expected_move / current_price) * 100
        
        # Profit probability estimation
        profit_probability = min(max((expected_move_pct / required_move_pct) * 0.4, 0.1), 0.7)
        
        recommendation = {
            'strike': strike,
            'call_price': call_price,
            'put_price': put_price,
            'put_strike': put_strike,
            'total_premium_per_contract': total_premium_per_contract,
            'contracts': max_contracts,
            'total_cost': total_cost,
            'max_loss': total_cost,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'required_move_pct': required_move_pct,
            'expected_move_pct': expected_move_pct,
            'profit_probability_estimate': profit_probability * 100,
            'breakeven_range': upper_breakeven - lower_breakeven
        }
        
        # Market insight
        market_insight = ""
        if market_analysis['volatility_regime'] == 'LOW_VOL':
            market_insight = "Low volatility environment - excellent for straddles expecting volatility expansion"
        elif market_analysis['trend'] == 'SIDEWAYS':
            market_insight = "Sideways trend suggests potential breakout - good setup for straddles"
        
        return {
            'strategy': 'LONG_STRADDLE',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendation': recommendation,
            'market_insight': market_insight
        }
    
    def _calculate_protective_put_optimal(self, puts_df: pd.DataFrame, current_price: float,
                                        capital: float, asset_class: str, market_analysis: Dict, options_data: Dict) -> Dict:
        """Calculate optimal protective put specifications"""
        
        share_equivalent = 100 if asset_class != 'FOREX' else 10000
        
        # Assume we already own the underlying or want to buy it
        stock_cost_per_lot = current_price * share_equivalent
        
        # Define protection levels
        protection_levels = [0.95, 0.90, 0.85]  # 5%, 10%, 15% downside protection
        
        recommendations = []
        
        for i, protection_level in enumerate(protection_levels):
            protection_pct = (1 - protection_level) * 100
            risk_level = ['Conservative', 'Moderate', 'Aggressive'][i]
            
            target_strike = current_price * protection_level
            
            # Find closest put strike
            protective_put = puts_df.iloc[(puts_df['strike'] - target_strike).abs().argsort()[:1]]
            
            if len(protective_put) == 0:
                continue
            
            put_price = protective_put.iloc[0]['lastPrice']
            put_strike = protective_put.iloc[0]['strike']
            
            # Calculate how many we can protect
            total_cost_per_lot = stock_cost_per_lot + (put_price * share_equivalent)
            max_lots = int(capital / total_cost_per_lot)
            
            if max_lots == 0:
                continue
            
            total_stock_cost = max_lots * stock_cost_per_lot
            total_put_cost = max_lots * put_price * share_equivalent
            total_investment = total_stock_cost + total_put_cost
            
            # Calculate protection metrics
            insurance_cost_pct = (put_price / current_price) * 100
            effective_floor = put_strike
            max_loss_per_share = current_price - put_strike + put_price
            total_max_loss = max_lots * max_loss_per_share * share_equivalent
            
            # Annualized insurance cost
            days_to_expiry = options_data.get('days_to_expiry', 30)
            annualized_insurance_cost = insurance_cost_pct * (365 / days_to_expiry)
            
            recommendations.append({
                'risk_level': risk_level,
                'protection_level_pct': protection_pct,
                'put_strike': put_strike,
                'put_price': put_price,
                'contracts': max_lots,
                'total_stock_cost': total_stock_cost,
                'total_put_cost': total_put_cost,
                'total_investment': total_investment,
                'insurance_cost_pct': insurance_cost_pct,
                'annualized_insurance_cost': annualized_insurance_cost,
                'effective_floor': effective_floor,
                'max_loss_per_share': max_loss_per_share,
                'total_max_loss': total_max_loss,
                'max_loss_pct': (max_loss_per_share / current_price) * 100,
                'breakeven': current_price + put_price
            })
        
        # Market insight
        market_insight = ""
        if market_analysis['volatility_regime'] in ['HIGH_VOL', 'EXTREME_VOL']:
            market_insight = "High volatility environment makes insurance more expensive but potentially necessary"
        elif market_analysis['trend'] in ['BEARISH', 'STRONG_BEARISH']:
            market_insight = "Bearish trend suggests protective puts are well-justified for downside protection"
        
        return {
            'strategy': 'PROTECTIVE_PUT',
            'asset_class': asset_class,
            'current_price': current_price,
            'available_capital': capital,
            'recommendations': recommendations,
            'market_insight': market_insight,
            'optimal_recommendation': recommendations[1] if len(recommendations) > 1 else recommendations[0] if recommendations else None
        }
    
    def get_options_greeks(self, ticker: str, asset_class: str) -> Dict:
        """Calculate options Greeks for all available contracts"""
        try:
            print(f"🔢 Calculating {asset_class} Greeks for {ticker}...")
            
            # Get current price
            underlying_data = self.get_asset_data(ticker, asset_class, days=30)
            current_price = underlying_data['current_price']
            
            # Get options data
            options_data = self.get_options_data(ticker, asset_class, current_price)
            calls_df = options_data['calls']
            puts_df = options_data['puts']
            
            if calls_df.empty and puts_df.empty:
                raise ValueError("No options data available for Greeks calculation")
            
            # Calculate Greeks for calls
            calls_greeks = self._calculate_greeks_for_options(calls_df, current_price, options_data['days_to_expiry'], 'call', asset_class)
            
            # Calculate Greeks for puts
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
                'total_contracts': len(calls_greeks) + len(puts_greeks)
            }
            
        except Exception as e:
            print(f"❌ Failed to calculate {asset_class} Greeks for {ticker}: {str(e)}")
            raise

    def get_options_greeks_with_expiry(self, ticker: str, asset_class: str, expiry_code: str, start_date, end_date) -> Dict:
        """Calculate options Greeks for specific expiry period"""
        try:
            print(f"🔢 Calculating {asset_class} Greeks for {ticker} ({expiry_code})...")
            
            # Get current price
            underlying_data = self.get_asset_data(ticker, asset_class, days=30)
            current_price = underlying_data['current_price']
            
            # Get options data for specific expiry range
            options_data = self.get_options_data_with_expiry(ticker, asset_class, current_price, start_date, end_date)
            calls_df = options_data['calls']
            puts_df = options_data['puts']
            
            if calls_df.empty and puts_df.empty:
                raise ValueError(f"No options data available for {expiry_code} expiry")
            
            # Calculate Greeks for calls
            calls_greeks = self._calculate_greeks_for_options(calls_df, current_price, options_data['days_to_expiry'], 'call', asset_class)
            
            # Calculate Greeks for puts
            puts_greeks = self._calculate_greeks_for_options(puts_df, current_price, options_data['days_to_expiry'], 'put', asset_class)
            
            # Summary statistics
            summary_stats = self._calculate_greeks_summary(calls_greeks, puts_greeks, current_price)
            
            # Get all available expiries for this symbol
            available_expiries = self._get_available_expiries(ticker, asset_class)
            
            return {
                'underlying_ticker': ticker,
                'underlying_price': current_price,
                'asset_class': asset_class,
                'expiry_code': expiry_code,
                'expiration': options_data['expiration'],
                'days_to_expiry': options_data['days_to_expiry'],
                'calls_greeks': calls_greeks,
                'puts_greeks': puts_greeks,
                'summary_stats': summary_stats,
                'total_contracts': len(calls_greeks) + len(puts_greeks),
                'available_expiries': available_expiries
            }
            
        except Exception as e:
            print(f"❌ Failed to calculate {asset_class} Greeks for {ticker} ({expiry_code}): {str(e)}")
            raise
    
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
            
            # Calculate implied volatility (simplified - using base vol)
            implied_vol = base_vol
            
            # Calculate Greeks
            greeks = self._calculate_black_scholes_greeks(
                underlying_price, strike, T, r, implied_vol, option_type
            )
            
            # Calculate moneyness
            moneyness = strike / underlying_price
            
            greeks_data.append({
                'ticker': option['ticker'],
                'strike': strike,
                'price': price,
                'delta': round(greeks['delta'], 4),
                'gamma': round(greeks['gamma'], 4),
                'theta': round(greeks['theta'], 4),
                'vega': round(greeks['vega'], 4),
                'rho': round(greeks['rho'], 4),
                'implied_vol': round(implied_vol, 4),
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
            # Return default values if calculation fails
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
    
    def _calculate_greeks_summary(self, calls_greeks: pd.DataFrame, puts_greeks: pd.DataFrame, 
                                 current_price: float) -> Dict:
        """Calculate summary statistics for Greeks"""
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
            
            # Highest gamma strike
            if 'gamma' in all_greeks.columns:
                max_gamma_idx = all_greeks['gamma'].idxmax()
                summary['highest_gamma_strike'] = all_greeks.loc[max_gamma_idx, 'strike']
            
            # Average implied volatility
            if 'implied_vol' in all_greeks.columns:
                summary['avg_implied_vol'] = all_greeks['implied_vol'].mean()
        
        return summary
    
    def run_strategy_backtest(self, ticker: str, asset_class: str, strategy: str, 
                             start_date: datetime, end_date: datetime, params: Dict) -> Dict:
        """Run backtest for a specific strategy"""
        try:
            print(f"🔄 Running {asset_class} {strategy} backtest for {ticker}...")
            
            # Get historical data for the backtest period
            total_days = (end_date - start_date).days + 100  # Extra buffer for technical indicators
            underlying_data = self.get_asset_data(ticker, asset_class, days=total_days)
            
            # Filter data to backtest period
            df = underlying_data['historical_data']
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if len(df) < 30:
                raise ValueError(f"Insufficient data for backtest period: only {len(df)} days")
            
            # Run strategy-specific backtest
            if strategy == 'COVERED_CALL':
                results = self._backtest_covered_call(df, params, asset_class)
            elif strategy == 'CASH_SECURED_PUT':
                results = self._backtest_cash_secured_put(df, params, asset_class)
            elif strategy == 'IRON_CONDOR':
                results = self._backtest_iron_condor(df, params, asset_class)
            elif strategy == 'BULL_CALL_SPREAD':
                results = self._backtest_bull_call_spread(df, params, asset_class)
            elif strategy == 'BEAR_PUT_SPREAD':
                results = self._backtest_bear_put_spread(df, params, asset_class)
            elif strategy == 'BUY_AND_HOLD':
                results = self._backtest_buy_and_hold(df, asset_class)
            else:
                raise ValueError(f"Backtest not implemented for {strategy}")
            
            # Calculate performance metrics
            performance_metrics = self._calculate_backtest_metrics(results, df)
            
            return {
                'ticker': ticker,
                'asset_class': asset_class,
                'strategy': strategy,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'params': params,
                'results': results,
                'performance_metrics': performance_metrics,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Backtest failed for {ticker} {strategy}: {str(e)}")
            return {
                'ticker': ticker,
                'strategy': strategy,
                'error': str(e),
                'success': False
            }
    
    def _backtest_covered_call(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest covered call strategy"""
        days_to_expiry = params.get('days_to_expiry', 30)
        delta_target = params.get('delta_target', 0.3)
        
        trades = []
        portfolio_value = []
        cash = 10000
        shares = 0
        
        for i in range(0, len(df), days_to_expiry):
            if i + days_to_expiry >= len(df):
                break
                
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Buy shares if we don't have them
            if shares == 0:
                shares = int(cash / entry_price)
                cash -= shares * entry_price
            
            # Sell call option (simplified)
            call_strike = entry_price * 1.05  # 5% OTM
            call_premium = max(entry_price * 0.02, 0.5)  # Simplified premium
            cash += call_premium * shares
            
            # Calculate profit/loss at expiration
            if exit_price > call_strike:
                # Called away
                cash += shares * call_strike
                pnl = cash - 10000
                shares = 0
                cash = 10000 + pnl  # Reset for next trade
            else:
                # Keep shares and premium
                pnl = (exit_price - entry_price) * shares + call_premium * shares
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'call_strike': call_strike,
                'call_premium': call_premium,
                'pnl': pnl,
                'portfolio_value': cash + shares * exit_price
            })
            
            portfolio_value.append(cash + shares * exit_price)
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_value,
            'total_trades': len(trades)
        }
    
    def _backtest_cash_secured_put(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest cash secured put strategy"""
        days_to_expiry = params.get('days_to_expiry', 30)
        delta_target = params.get('delta_target', 0.3)
        
        trades = []
        portfolio_value = []
        cash = 10000
        shares = 0
        
        for i in range(0, len(df), days_to_expiry):
            if i + days_to_expiry >= len(df):
                break
                
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Sell put option
            put_strike = entry_price * 0.95  # 5% OTM
            put_premium = max(entry_price * 0.02, 0.5)  # Simplified premium
            contracts = int(cash / (put_strike * 100))
            
            if contracts == 0:
                continue
                
            cash_reserved = contracts * put_strike * 100
            premium_received = contracts * put_premium * 100
            
            # Calculate profit/loss at expiration
            if exit_price < put_strike:
                # Assigned
                shares += contracts * 100
                cash = cash - cash_reserved + premium_received
                pnl = premium_received - (put_strike - exit_price) * contracts * 100
            else:
                # Keep premium
                pnl = premium_received
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'put_strike': put_strike,
                'put_premium': put_premium,
                'contracts': contracts,
                'pnl': pnl,
                'portfolio_value': cash + shares * exit_price
            })
            
            portfolio_value.append(cash + shares * exit_price)
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_value,
            'total_trades': len(trades)
        }
    
    def _backtest_iron_condor(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest iron condor strategy"""
        days_to_expiry = params.get('days_to_expiry', 30)
        wing_width = params.get('wing_width', 0.05)
        
        trades = []
        portfolio_value = []
        cash = 10000
        
        for i in range(0, len(df), days_to_expiry):
            if i + days_to_expiry >= len(df):
                break
                
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Iron Condor strikes
            call_sell = entry_price * (1 + wing_width)
            call_buy = entry_price * (1 + wing_width * 2)
            put_sell = entry_price * (1 - wing_width)
            put_buy = entry_price * (1 - wing_width * 2)
            
            # Simplified premium calculation
            net_credit = entry_price * 0.02  # 2% of underlying price
            max_loss = (call_buy - call_sell) - net_credit
            
            contracts = int(cash * 0.1 / (max_loss * 100))  # Use 10% of capital
            if contracts == 0:
                contracts = 1
            
            # Calculate profit/loss at expiration
            if put_buy < exit_price < call_buy:
                # Maximum profit
                pnl = net_credit * contracts * 100
            elif exit_price <= put_buy or exit_price >= call_buy:
                # Maximum loss
                pnl = -max_loss * contracts * 100
            else:
                # Partial loss
                if exit_price < put_sell:
                    pnl = (net_credit - (put_sell - exit_price)) * contracts * 100
                else:  # exit_price > call_sell
                    pnl = (net_credit - (exit_price - call_sell)) * contracts * 100
            
            cash += pnl
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'call_sell': call_sell,
                'call_buy': call_buy,
                'put_sell': put_sell,
                'put_buy': put_buy,
                'net_credit': net_credit,
                'contracts': contracts,
                'pnl': pnl,
                'portfolio_value': cash
            })
            
            portfolio_value.append(cash)
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_value,
            'total_trades': len(trades)
        }
    
    def _backtest_bull_call_spread(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest bull call spread strategy"""
        days_to_expiry = params.get('days_to_expiry', 30)
        delta_target = params.get('delta_target', 0.3)
        
        trades = []
        portfolio_value = []
        cash = 10000
        
        for i in range(0, len(df), days_to_expiry):
            if i + days_to_expiry >= len(df):
                break
                
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Bull call spread strikes
            buy_strike = entry_price * 0.98  # Slightly ITM
            sell_strike = entry_price * 1.08  # 8% OTM
            
            # Simplified cost calculation
            net_debit = entry_price * 0.03  # 3% of underlying price
            max_profit = (sell_strike - buy_strike) - net_debit
            
            contracts = int(cash * 0.2 / (net_debit * 100))  # Use 20% of capital
            if contracts == 0:
                contracts = 1
            
            # Calculate profit/loss at expiration
            if exit_price >= sell_strike:
                # Maximum profit
                pnl = max_profit * contracts * 100
            elif exit_price <= buy_strike:
                # Maximum loss
                pnl = -net_debit * contracts * 100
            else:
                # Partial profit
                pnl = ((exit_price - buy_strike) - net_debit) * contracts * 100
            
            cash += pnl
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'buy_strike': buy_strike,
                'sell_strike': sell_strike,
                'net_debit': net_debit,
                'contracts': contracts,
                'pnl': pnl,
                'portfolio_value': cash
            })
            
            portfolio_value.append(cash)
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_value,
            'total_trades': len(trades)
        }
    
    def _backtest_bear_put_spread(self, df: pd.DataFrame, params: Dict, asset_class: str) -> Dict:
        """Backtest bear put spread strategy"""
        days_to_expiry = params.get('days_to_expiry', 30)
        delta_target = params.get('delta_target', 0.3)
        
        trades = []
        portfolio_value = []
        cash = 10000
        
        for i in range(0, len(df), days_to_expiry):
            if i + days_to_expiry >= len(df):
                break
                
            entry_date = df.index[i]
            exit_date = df.index[min(i + days_to_expiry, len(df) - 1)]
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[min(i + days_to_expiry, len(df) - 1)]['close']
            
            # Bear put spread strikes
            buy_strike = entry_price * 1.02  # Slightly ITM
            sell_strike = entry_price * 0.92  # 8% OTM
            
            # Simplified cost calculation
            net_debit = entry_price * 0.03  # 3% of underlying price
            max_profit = (buy_strike - sell_strike) - net_debit
            
            contracts = int(cash * 0.2 / (net_debit * 100))  # Use 20% of capital
            if contracts == 0:
                contracts = 1
            
            # Calculate profit/loss at expiration
            if exit_price <= sell_strike:
                # Maximum profit
                pnl = max_profit * contracts * 100
            elif exit_price >= buy_strike:
                # Maximum loss
                pnl = -net_debit * contracts * 100
            else:
                # Partial profit
                pnl = ((buy_strike - exit_price) - net_debit) * contracts * 100
            
            cash += pnl
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'buy_strike': buy_strike,
                'sell_strike': sell_strike,
                'net_debit': net_debit,
                'contracts': contracts,
                'pnl': pnl,
                'portfolio_value': cash
            })
            
            portfolio_value.append(cash)
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_value,
            'total_trades': len(trades)
        }
    
    def _backtest_buy_and_hold(self, df: pd.DataFrame, asset_class: str) -> Dict:
        """Backtest buy and hold strategy"""
        initial_capital = 10000
        entry_price = df.iloc[0]['close']
        exit_price = df.iloc[-1]['close']
        
        shares = initial_capital / entry_price
        final_value = shares * exit_price
        total_return = (final_value - initial_capital) / initial_capital
        
        # Create portfolio value series
        portfolio_values = (df['close'] / entry_price * initial_capital).tolist()
        
        return {
            'trades': [{
                'entry_date': df.index[0],
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'pnl': final_value - initial_capital,
                'portfolio_value': final_value
            }],
            'portfolio_values': portfolio_values,
            'total_trades': 1
        }
    
    def _calculate_backtest_metrics(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Calculate performance metrics for backtest"""
        portfolio_values = results['portfolio_values']
        trades = results['trades']
        
        if not portfolio_values or not trades:
            return {}
        
        initial_value = 10000
        final_value = portfolio_values[-1]
        
        # Total return
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        portfolio_series = pd.Series(portfolio_values)
        daily_returns = portfolio_series.pct_change().dropna()
        
        # Annualized return
        days = len(portfolio_values)
        annualized_return = (final_value / initial_value) ** (252 / days) - 1
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = portfolio_series.cummax()
        drawdown = (portfolio_series - cumulative) / cumulative
        max_drawdown = drawdown.min()
        
        # Win rate
        profitable_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        win_rate = profitable_trades / len(trades) if trades else 0
        
        # Average profit/loss
        total_pnl = sum(trade.get('pnl', 0) for trade in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0
        
        return {
            'total_return': round(total_return * 100, 2),
            'annualized_return': round(annualized_return * 100, 2),
            'volatility': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'win_rate': round(win_rate * 100, 2),
            'total_trades': len(trades),
            'avg_pnl_per_trade': round(avg_pnl, 2),
            'final_portfolio_value': round(final_value, 2)
        }
    
    def generate_market_prediction(self, ticker: str, asset_class: str, prediction_days: int) -> Dict:
        """Generate market predictions using technical analysis and AI insights"""
        try:
            print(f"🔮 Generating {asset_class} prediction for {ticker} ({prediction_days} days)...")
            
            # Get historical data
            underlying_data = self.get_asset_data(ticker, asset_class, days=500)
            df = underlying_data['historical_data']
            current_price = underlying_data['current_price']
            
            # Technical analysis
            technical_signals = self._analyze_technical_signals(df, asset_class)
            
            # Market sentiment analysis
            sentiment_analysis = self._analyze_market_sentiment(df, asset_class)
            
            # Asset-specific analysis
            asset_specific_analysis = self._analyze_asset_specific_factors(ticker, asset_class, df)
            
            # Generate price predictions
            price_predictions = self._generate_price_predictions(df, current_price, prediction_days, asset_class)
            
            # Calculate prediction confidence
            confidence_score = self._calculate_prediction_confidence(technical_signals, sentiment_analysis, asset_specific_analysis)
            
            # Generate trading recommendations
            trading_recommendations = self._generate_trading_recommendations(
                price_predictions, technical_signals, confidence_score, asset_class
            )
            
            return {
                'ticker': ticker,
                'asset_class': asset_class,
                'current_price': current_price,
                'prediction_days': prediction_days,
                'historical_data': df,  # Include historical data for charting
                'technical_signals': technical_signals,
                'sentiment_analysis': sentiment_analysis,
                'asset_specific_analysis': asset_specific_analysis,
                'price_predictions': price_predictions,
                'confidence_score': confidence_score,
                'trading_recommendations': trading_recommendations,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Failed to generate {asset_class} prediction for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'asset_class': asset_class,
                'error': str(e),
                'success': False
            }
    
    def _analyze_technical_signals(self, df: pd.DataFrame, asset_class: str) -> Dict:
        """Analyze technical signals for prediction"""
        latest = df.iloc[-1]
        
        # Moving average signals
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        sma_200 = df['close'].rolling(200).mean().iloc[-1]
        
        ma_signal = "BULLISH" if latest['close'] > sma_20 > sma_50 else "BEARISH" if latest['close'] < sma_20 < sma_50 else "NEUTRAL"
        
        # RSI signal
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        rsi_signal = "OVERBOUGHT" if current_rsi > 70 else "OVERSOLD" if current_rsi < 30 else "NEUTRAL"
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        bb_position = (latest['close'] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        bb_signal = "UPPER" if bb_position > 0.8 else "LOWER" if bb_position < 0.2 else "MIDDLE"
        
        # Volume analysis (for non-FX)
        if asset_class != 'FOREX':
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_signal = "HIGH" if latest['volume'] > avg_volume * 1.5 else "LOW" if latest['volume'] < avg_volume * 0.5 else "NORMAL"
        else:
            volume_signal = "N/A"
        
        return {
            'ma_signal': ma_signal,
            'rsi_signal': rsi_signal,
            'rsi_value': round(current_rsi, 1),
            'bb_signal': bb_signal,
            'bb_position': round(bb_position * 100, 1),
            'volume_signal': volume_signal,
            'trend_strength': abs(latest['close'] - sma_50) / sma_50 * 100 if sma_50 > 0 else 0
        }
    
    def _analyze_market_sentiment(self, df: pd.DataFrame, asset_class: str) -> Dict:
        """Analyze market sentiment indicators"""
        # Volatility analysis
        returns = df['close'].pct_change()
        volatility_20d = returns.rolling(20).std() * np.sqrt(252)
        current_vol = volatility_20d.iloc[-1]
        avg_vol = volatility_20d.mean()
        
        vol_regime = "HIGH" if current_vol > avg_vol * 1.2 else "LOW" if current_vol < avg_vol * 0.8 else "NORMAL"
        
        # Price momentum
        momentum_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
        momentum_20d = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100
        
        momentum_signal = "STRONG_UP" if momentum_20d > 5 else "UP" if momentum_20d > 0 else "STRONG_DOWN" if momentum_20d < -5 else "DOWN"
        
        # Market regime detection
        recent_highs = df['high'].rolling(20).max()
        recent_lows = df['low'].rolling(20).min()
        range_pct = ((recent_highs - recent_lows) / df['close']).iloc[-1] * 100
        
        market_regime = "TRENDING" if range_pct > 15 else "RANGING" if range_pct < 8 else "TRANSITIONAL"
        
        return {
            'volatility_regime': vol_regime,
            'current_volatility': round(current_vol * 100, 1),
            'momentum_signal': momentum_signal,
            'momentum_5d': round(momentum_5d, 2),
            'momentum_20d': round(momentum_20d, 2),
            'market_regime': market_regime,
            'range_percentage': round(range_pct, 1)
        }
    
    def _analyze_asset_specific_factors(self, ticker: str, asset_class: str, df: pd.DataFrame) -> Dict:
        """Analyze asset-specific factors"""
        if asset_class == 'FOREX':
            return {
                'factor_type': 'Central Bank Policy',
                'key_factors': ['Interest Rate Differentials', 'Economic Data', 'Risk Sentiment'],
                'current_theme': 'Monitor central bank communications',
                'seasonality': 'Higher volatility during major economic releases'
            }
        elif asset_class == 'INDICES':
            return {
                'factor_type': 'Market Breadth',
                'key_factors': ['Sector Rotation', 'Economic Cycle', 'Volatility Regime'],
                'current_theme': 'Broad market exposure with sector considerations',
                'seasonality': 'Year-end effects and quarterly rebalancing'
            }
        else:  # EQUITIES
            # Calculate some equity-specific metrics
            price_range_52w = (df['high'].rolling(252).max().iloc[-1] - df['low'].rolling(252).min().iloc[-1])
            current_position = (df['close'].iloc[-1] - df['low'].rolling(252).min().iloc[-1]) / price_range_52w
            
            return {
                'factor_type': 'Company Fundamentals',
                'key_factors': ['Earnings Growth', 'Sector Performance', 'Market Cap Considerations'],
                'current_theme': f'Currently at {current_position*100:.0f}% of 52-week range',
                'seasonality': 'Earnings season effects and sector rotation'
            }
    
    def _generate_price_predictions(self, df: pd.DataFrame, current_price: float, 
                                   prediction_days: int, asset_class: str) -> Dict:
        """Generate price predictions using technical analysis"""
        # Simple trend-based prediction
        returns = df['close'].pct_change().dropna()
        
        # Calculate trend components
        short_trend = returns.tail(5).mean()
        medium_trend = returns.tail(20).mean()
        long_trend = returns.tail(60).mean()
        
        # Weighted average of trends
        trend_weight = 0.5 * short_trend + 0.3 * medium_trend + 0.2 * long_trend
        
        # Volatility for range calculations
        volatility = returns.std()
        
        # Generate predictions
        daily_drift = trend_weight
        daily_vol = volatility / np.sqrt(prediction_days)
        
        # Target price (trend-based)
        target_price = current_price * (1 + daily_drift * prediction_days)
        
        # Confidence ranges
        std_dev = current_price * daily_vol * np.sqrt(prediction_days)
        
        bull_case = target_price + 1.5 * std_dev
        bear_case = target_price - 1.5 * std_dev
        
        # Support and resistance levels
        recent_high = df['high'].tail(60).max()
        recent_low = df['low'].tail(60).min()
        
        return {
            'target_price': round(target_price, 4 if asset_class == 'FOREX' else 2),
            'bull_case': round(bull_case, 4 if asset_class == 'FOREX' else 2),
            'bear_case': round(bear_case, 4 if asset_class == 'FOREX' else 2),
            'expected_move_pct': round((target_price / current_price - 1) * 100, 2),
            'resistance_level': round(recent_high, 4 if asset_class == 'FOREX' else 2),
            'support_level': round(recent_low, 4 if asset_class == 'FOREX' else 2),
            'volatility_estimate': round(volatility * np.sqrt(252) * 100, 1)
        }
    
    def _calculate_prediction_confidence(self, technical_signals: Dict, sentiment_analysis: Dict, 
                                       asset_specific_analysis: Dict) -> Dict:
        """Calculate confidence score for predictions"""
        confidence_factors = []
        
        # Technical signal confidence
        if technical_signals['ma_signal'] != 'NEUTRAL':
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # RSI confidence
        if technical_signals['rsi_signal'] in ['OVERBOUGHT', 'OVERSOLD']:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Momentum confidence
        if sentiment_analysis['momentum_signal'] in ['STRONG_UP', 'STRONG_DOWN']:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Volatility confidence
        if sentiment_analysis['volatility_regime'] == 'NORMAL':
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        overall_confidence = np.mean(confidence_factors)
        
        return {
            'overall_score': round(overall_confidence * 100, 1),
            'technical_confidence': round(confidence_factors[0] * 100, 1),
            'momentum_confidence': round(confidence_factors[2] * 100, 1),
            'volatility_confidence': round(confidence_factors[3] * 100, 1),
            'interpretation': 'HIGH' if overall_confidence > 0.7 else 'MEDIUM' if overall_confidence > 0.5 else 'LOW'
        }
    
    def _generate_trading_recommendations(self, price_predictions: Dict, technical_signals: Dict, 
                                        confidence_score: Dict, asset_class: str) -> Dict:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        expected_move = price_predictions['expected_move_pct']
        confidence = confidence_score['overall_score']
        
        # Directional recommendations
        if expected_move > 2 and confidence > 60:
            recommendations.append({
                'strategy': 'BULL_CALL_SPREAD',
                'rationale': f'Expecting {expected_move:.1f}% upward move with {confidence:.0f}% confidence',
                'target': price_predictions['target_price'],
                'risk_level': 'MODERATE'
            })
        elif expected_move < -2 and confidence > 60:
            recommendations.append({
                'strategy': 'BEAR_PUT_SPREAD',
                'rationale': f'Expecting {expected_move:.1f}% downward move with {confidence:.0f}% confidence',
                'target': price_predictions['target_price'],
                'risk_level': 'MODERATE'
            })
        
        # Range-bound recommendations
        if abs(expected_move) < 3 and technical_signals['ma_signal'] == 'NEUTRAL':
            recommendations.append({
                'strategy': 'IRON_CONDOR',
                'rationale': f'Low expected movement ({expected_move:.1f}%) suggests range-bound trading',
                'target': 'Range maintenance',
                'risk_level': 'LOW'
            })
        
        # Volatility recommendations
        if confidence < 50:
            recommendations.append({
                'strategy': 'LONG_STRADDLE',
                'rationale': f'Low confidence ({confidence:.0f}%) suggests uncertainty - volatility play',
                'target': 'Significant move in either direction',
                'risk_level': 'HIGH'
            })
        
        # Income recommendations
        if asset_class != 'FOREX' and expected_move > -1 and expected_move < 3:
            recommendations.append({
                'strategy': 'COVERED_CALL',
                'rationale': 'Modest upward bias suitable for income generation',
                'target': 'Premium collection',
                'risk_level': 'LOW'
            })
        
        return {
            'primary_recommendation': recommendations[0] if recommendations else None,
            'alternative_strategies': recommendations[1:] if len(recommendations) > 1 else [],
            'market_outlook': 'BULLISH' if expected_move > 2 else 'BEARISH' if expected_move < -2 else 'NEUTRAL',
            'time_horizon': 'Short-term (1-4 weeks)',
            'key_risks': self._identify_key_risks(asset_class, technical_signals, price_predictions)
        }
    
    def _identify_key_risks(self, asset_class: str, technical_signals: Dict, price_predictions: Dict) -> List[str]:
        """Identify key risks for the analysis"""
        risks = []
        
        if asset_class == 'FOREX':
            risks.extend([
                'Central bank policy changes',
                'Economic data surprises',
                'Geopolitical events affecting currency'
            ])
        elif asset_class == 'INDICES':
            risks.extend([
                'Market-wide corrections',
                'Sector rotation impacts',
                'Economic cycle changes'
            ])
        else:  # EQUITIES
            risks.extend([
                'Company-specific news',
                'Earnings surprises',
                'Sector headwinds'
            ])
        
        # Technical risks
        if technical_signals['rsi_signal'] in ['OVERBOUGHT', 'OVERSOLD']:
            risks.append(f"RSI indicating {technical_signals['rsi_signal'].lower()} conditions")
        
        # Volatility risks
        vol_estimate = price_predictions.get('volatility_estimate', 0)
        if vol_estimate > 30:
            risks.append('High volatility environment increases risk')
        
        return risks
    
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
            
            # Strategy selection with explanations
            if debug:
                print("**Debug:** Selecting strategies...")
            
            strategy_results = self.select_strategy(market_analysis, underlying_data, options_data)
            strategy_scores = strategy_results['scores']
            strategy_explanations = strategy_results['explanations']
            
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
                'strategy_explanations': strategy_explanations,
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
            'description': '📊 Index ETFs & Volatility Products - Diversified exposure to market segments',
            'examples': 'SPY, QQQ, IWM, EWU, VGK, VIX',
            'characteristics': 'Lower volatility, broad market exposure, high liquidity'
        },
        'EQUITIES': {
            'description': '📈 Individual Stocks - Direct company exposure with higher alpha potential',
            'examples': 'AAPL, MSFT, GOOGL, TSLA, NVDA, META',
            'characteristics': 'Higher volatility, company-specific risk, earnings-driven moves'
        },
        'FOREX': {
            'description': '💱 Currency Pairs - Global forex markets with 24/5 trading',
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
                        st.write(f"• Valid records: {data_check['valid_records']}/{data_check['total_records']}")
                        if data_check['latest_price']:
                            st.write(f"• Latest price: {data_check['latest_price']:.4f}")
                        st.write(f"• Date range: {data_check['date_range']}")
                    else:
                        st.error(f"❌ {test_symbol} data issues:")
                        if 'reason' in data_check:
                            st.write(f"• Reason: {data_check['reason']}")
                        if 'error' in data_check:
                            st.write(f"• Error: {data_check['error']}")
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
                        st.write(f"• Contracts found: {options_check.get('contract_count', 0)}")
                        st.write(f"• Sample expiration: {options_check.get('sample_expiration', 'N/A')}")
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

    # Tab 1: Enhanced Analysis with Explanations and Contract Recommendations
    with tab1:
        # Handle analysis button click
        if analyze_button and symbol_input:
            with st.spinner(f"Analyzing {symbol_input} ({asset_class}) with real data..."):
                result = strategist.analyze_symbol(symbol_input.upper(), asset_class, debug=debug_mode)
            
            if result['success']:
                # Store result in session state
                st.session_state.analysis_result = result
            else:
                st.error(f"❌ {asset_class} analysis failed: {result['error']}")
                st.session_state.analysis_result = None
        
        # Display analysis results if they exist in session state
        if st.session_state.analysis_result and st.session_state.analysis_result.get('success'):
            result = st.session_state.analysis_result
            
            # Only display if the asset class matches current selection
            if result.get('asset_class') == asset_class:
                
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
                
                # Trading Chart with Professional Technical Analysis
                # Trading Chart with Professional Technical Analysis
                st.subheader(f"📈 {asset_class} Professional Trading Chart")
                
                # Chart configuration options
                with st.expander("⚙️ Chart Configuration", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📊 Moving Averages:**")
                        show_sma_20 = st.checkbox("SMA 20", value=True, key=f"sma20_{result['ticker']}")
                        show_sma_40 = st.checkbox("SMA 40", value=True, key=f"sma40_{result['ticker']}")
                        show_sma_50 = st.checkbox("SMA 50", value=True, key=f"sma50_{result['ticker']}")
                    
                    with col2:
                        st.markdown("**📈 Long-term SMAs:**")
                        show_sma_100 = st.checkbox("SMA 100", value=True, key=f"sma100_{result['ticker']}")
                        show_sma_150 = st.checkbox("SMA 150", value=True, key=f"sma150_{result['ticker']}")
                        show_sma_200 = st.checkbox("SMA 200", value=True, key=f"sma200_{result['ticker']}")
                    
                    with col3:
                        st.markdown("**🎯 Support/Resistance:**")
                        show_support = st.checkbox("52W Low (Support)", value=True, key=f"support_{result['ticker']}")
                        show_resistance = st.checkbox("52W High (Resistance)", value=True, key=f"resistance_{result['ticker']}")
                        show_current = st.checkbox("Current Price Line", value=True, key=f"current_{result['ticker']}")
                
                try:
                    chart_data = {
                        'ticker': underlying['ticker'],
                        'current_price': underlying['current_price'],
                        'historical_data': underlying['historical_data']
                    }
                    
                    # Configure support/resistance levels based on user selection
                    support_resistance = {}
                    if show_support:
                        support_resistance['support_level'] = underlying['low_52w']
                    if show_resistance:
                        support_resistance['resistance_level'] = underlying['high_52w']
                    
                    # Pass SMA configuration to chart
                    sma_config = {
                        'show_sma_20': show_sma_20,
                        'show_sma_40': show_sma_40,
                        'show_sma_50': show_sma_50,
                        'show_sma_100': show_sma_100,
                        'show_sma_150': show_sma_150,
                        'show_sma_200': show_sma_200,
                        'show_current_price': show_current
                    }
                    
                    chart = strategist.create_enhanced_trading_chart(chart_data, asset_class, support_resistance, sma_config)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Technical Analysis Summary
                    st.markdown("### 📊 Technical Analysis Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📈 Moving Average Analysis:**")
                        current_price = underlying['current_price']
                        sma_20 = underlying.get('sma_20', current_price)
                        sma_50 = underlying.get('sma_50', current_price)
                        sma_200 = underlying.get('sma_200', current_price)
                        
                        if current_price > sma_20 > sma_50 > sma_200:
                            st.success("🟢 **Bullish Alignment** - All SMAs trending up")
                        elif current_price < sma_20 < sma_50 < sma_200:
                            st.error("🔴 **Bearish Alignment** - All SMAs trending down")  
                        else:
                            st.warning("🟡 **Mixed Signals** - SMAs showing conflicting trends")
                    
                    with col2:
                        st.markdown("**🎯 Support/Resistance:**")
                        distance_to_support = ((current_price - underlying['low_52w']) / underlying['low_52w']) * 100
                        distance_to_resistance = ((underlying['high_52w'] - current_price) / current_price) * 100
                        
                        st.metric("Distance to Support", f"{distance_to_support:.1f}%")
                        st.metric("Distance to Resistance", f"{distance_to_resistance:.1f}%")
                    
                    with col3:
                        st.markdown("**📊 Position in Range:**")
                        range_position = ((current_price - underlying['low_52w']) / (underlying['high_52w'] - underlying['low_52w'])) * 100
                        
                        if range_position > 80:
                            st.error(f"🔴 **Near High** ({range_position:.0f}% of range)")
                        elif range_position < 20:
                            st.success(f"🟢 **Near Low** ({range_position:.0f}% of range)")
                        else:
                            st.info(f"🟡 **Mid-Range** ({range_position:.0f}% of range)")
                            
                except Exception as e:
                    st.error(f"Could not create {asset_class} chart: {str(e)}")
                
                # Market Conditions Analysis
                st.subheader(f"🌡️ {asset_class} Market Conditions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Current Market State:**")
                    
                    # Trend
                    trend_color = {
                        'STRONG_BULLISH': '🟢',
                        'BULLISH': '🟢',
                        'SIDEWAYS_BULLISH': '🟡',
                        'SIDEWAYS': '🟡',
                        'SIDEWAYS_BEARISH': '🟡',
                        'BEARISH': '🔴',
                        'STRONG_BEARISH': '🔴'
                    }.get(analysis['trend'], '⚪')
                    
                    st.write(f"{trend_color} **Trend:** {analysis['trend'].replace('_', ' ').title()}")
                    st.write(f"📊 **Strength:** {analysis['trend_strength']:.1f}%")
                    
                    # Volatility
                    vol_color = {
                        'LOW_VOL': '🟢',
                        'NORMAL_VOL': '🟡',
                        'HIGH_VOL': '🟠',
                        'EXTREME_VOL': '🔴'
                    }.get(analysis['volatility_regime'], '⚪')
                    
                    st.write(f"{vol_color} **Volatility:** {analysis['volatility_regime'].replace('_', ' ').title()}")
                    st.write(f"📈 **21-Day Vol:** {analysis['realized_vol']:.1%}")
                
                with col2:
                    st.markdown("**⚡ Momentum & Signals:**")
                    
                    # Momentum
                    momentum_color = {
                        'EXTREMELY_OVERBOUGHT': '🔴',
                        'OVERBOUGHT': '🟠',
                        'BULLISH': '🟢',
                        'NEUTRAL': '🟡',
                        'BEARISH': '🔴',
                        'OVERSOLD': '🟢',
                        'EXTREMELY_OVERSOLD': '🟢'
                    }.get(analysis['momentum'], '⚪')
                    
                    st.write(f"{momentum_color} **Momentum:** {analysis['momentum'].replace('_', ' ').title()}")
                    st.write(f"📊 **RSI:** {analysis['rsi']:.1f}")
                    
                    # Bollinger Band position
                    bb_color = {
                        'EXTREME_UPPER': '🔴',
                        'UPPER_BAND': '🟠',
                        'MIDDLE_RANGE': '🟢',
                        'LOWER_BAND': '🟠',
                        'EXTREME_LOWER': '🔴'
                    }.get(analysis['bb_signal'], '⚪')
                    
                    st.write(f"{bb_color} **BB Position:** {analysis['bb_signal'].replace('_', ' ').title()}")
                    st.write(f"📊 **BB %:** {analysis['bb_position']:.1f}%")
                
                # Strategy Analysis
                st.subheader(f"🎯 {asset_class} Strategy Analysis")
                
                strategy_scores = result['strategy_scores']
                strategy_explanations = result['strategy_explanations']
                
                # Display top strategies with detailed explanations
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**🏆 Recommended Strategies:**")
                    for i, (strategy, score) in enumerate(list(strategy_scores.items())[:3]):
                        if i == 0:
                            st.success(f"🥇 **{strategy.replace('_', ' ').title()}** - {score:.1f}/10")
                        elif i == 1:
                            st.info(f"🥈 **{strategy.replace('_', ' ').title()}** - {score:.1f}/10")
                        else:
                            st.warning(f"🥉 **{strategy.replace('_', ' ').title()}** - {score:.1f}/10")
                
                with col2:
                    st.markdown("**📊 All Strategy Scores:**")
                    for strategy, score in strategy_scores.items():
                        progress_val = score / 10.0
                        st.write(f"**{strategy.replace('_', ' ').title()}:** {score:.1f}/10")
                        st.progress(progress_val)
                
                # Detailed explanations for top strategy
                if strategy_scores:
                    best_strategy = list(strategy_scores.keys())[0]
                    
                    st.markdown("### 🔍 Why This Strategy?")
                    
                    if best_strategy in strategy_explanations:
                        explanation_parts = strategy_explanations[best_strategy]
                        
                        st.markdown(f"**🎯 {best_strategy.replace('_', ' ').title()} Scoring Breakdown:**")
                        
                        for part in explanation_parts:
                            # Format explanation parts with emojis
                            if "Base score" in part:
                                st.write(f"📊 {part}")
                            elif "+0." in part or "+1." in part or "+2." in part:
                                st.write(f"✅ {part}")
                            elif "RSI" in part or "shows" in part:
                                st.write(f"📈 {part}")
                            else:
                                st.write(f"💡 {part}")
                
                # Contract Recommendations
                st.subheader(f"📋 {asset_class} Contract Recommendations")
                
                if strategy_scores:
                    best_strategy = list(strategy_scores.keys())[0]
                    
                    # Get contract specifications
                    with st.spinner("Calculating optimal contracts..."):
                        try:
                            contract_specs = strategist.calculate_optimal_contracts(
                                best_strategy, 
                                analysis, 
                                underlying, 
                                result['options_data']
                            )
                            
                            if 'error' not in contract_specs:
                                
                                if best_strategy in ['COVERED_CALL', 'CASH_SECURED_PUT', 'IRON_CONDOR', 
                                                   'BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', 'PROTECTIVE_PUT']:
                                    
                                    st.success(f"✅ {contract_specs['strategy']} Contract Analysis Complete")
                                    
                                    # Display market insight
                                    if contract_specs.get('market_insight'):
                                        st.info(f"💡 **Market Insight:** {contract_specs['market_insight']}")
                                    
                                    # Display recommendations
                                    if 'recommendations' in contract_specs:
                                        recommendations = contract_specs['recommendations']
                                        
                                        # Create tabs for different risk levels
                                        risk_tabs = st.tabs([rec['risk_level'] for rec in recommendations])
                                        
                                        for tab, rec in zip(risk_tabs, recommendations):
                                            with tab:
                                                col1, col2, col3 = st.columns(3)
                                                
                                                with col1:
                                                    st.markdown("**📊 Position Details:**")
                                                    
                                                    if asset_class == 'FOREX':
                                                        st.metric("Strike", f"{rec['strike']:.5f}")
                                                        st.metric("Premium", f"{rec['premium']:.4f}")
                                                    else:
                                                        st.metric("Strike", f"${rec['strike']:.2f}")
                                                        st.metric("Premium", f"${rec['premium']:.2f}")
                                                    
                                                    st.metric("Contracts", rec['contracts'])
                                                
                                                with col2:
                                                    st.markdown("**💰 Financial Impact:**")
                                                    
                                                    if best_strategy == 'COVERED_CALL':
                                                        st.metric("Stock Cost", f"${rec['total_stock_cost']:,.0f}")
                                                        st.metric("Premium Received", f"${rec['total_premium_received']:,.0f}")
                                                        st.metric("Net Investment", f"${rec['net_investment']:,.0f}")
                                                        st.metric("Max Profit", f"${rec['max_profit']:,.0f}")
                                                    
                                                    elif best_strategy == 'CASH_SECURED_PUT':
                                                        st.metric("Cash Required", f"${rec['cash_required']:,.0f}")
                                                        st.metric("Premium Received", f"${rec['premium_received']:,.0f}")
                                                        st.metric("Effective Buy Price", f"${rec['effective_buy_price']:.2f}")
                                                        st.metric("Max Profit", f"${rec['max_profit']:,.0f}")
                                                    
                                                    elif best_strategy == 'IRON_CONDOR':
                                                        st.metric("Credit Received", f"${rec['total_credit']:,.0f}")
                                                        st.metric("Max Profit", f"${rec['max_profit']:,.0f}")
                                                        st.metric("Max Loss", f"${rec['max_loss']:,.0f}")
                                                        st.metric("Margin Required", f"${rec['margin_required']:,.0f}")
                                                    
                                                    else:  # Spreads
                                                        st.metric("Total Cost", f"${rec['total_cost']:,.0f}")
                                                        st.metric("Max Profit", f"${rec['max_profit']:,.0f}")
                                                        st.metric("Max Loss", f"${rec['max_loss']:,.0f}")
                                                        if 'target_price' in rec:
                                                            if asset_class == 'FOREX':
                                                                st.metric("Target Price", f"{rec['target_price']:.5f}")
                                                            else:
                                                                st.metric("Target Price", f"${rec['target_price']:.2f}")
                                                
                                                with col3:
                                                    st.markdown("**📈 Risk/Reward:**")
                                                    
                                                    if best_strategy == 'COVERED_CALL':
                                                        st.metric("Max Profit %", f"{rec['max_profit_pct']:.1f}%")
                                                        st.metric("Annualized Yield", f"{rec['premium_yield_annualized']:.1f}%")
                                                        st.metric("Downside Protection", f"{rec['downside_protection_pct']:.1f}%")
                                                        if asset_class == 'FOREX':
                                                            st.metric("Breakeven", f"{rec['breakeven']:.5f}")
                                                        else:
                                                            st.metric("Breakeven", f"${rec['breakeven']:.2f}")
                                                    
                                                    elif best_strategy == 'CASH_SECURED_PUT':
                                                        st.metric("Max Profit %", f"{rec['max_profit_pct']:.1f}%")
                                                        st.metric("Annualized Yield", f"{rec['premium_yield_annualized']:.1f}%")
                                                        st.metric("Discount %", f"{rec['discount_pct']:.1f}%")
                                                        if asset_class == 'FOREX':
                                                            st.metric("Breakeven", f"{rec['breakeven']:.5f}")
                                                        else:
                                                            st.metric("Breakeven", f"${rec['breakeven']:.2f}")
                                                    
                                                    elif best_strategy == 'IRON_CONDOR':
                                                        st.metric("Max Profit %", f"{rec['max_profit_pct']:.1f}%")
                                                        st.metric("Profit Range", f"{rec['profit_range_pct']:.1f}%")
                                                        if asset_class == 'FOREX':
                                                            st.metric("Upper BE", f"{rec['upper_breakeven']:.5f}")
                                                            st.metric("Lower BE", f"{rec['lower_breakeven']:.5f}")
                                                        else:
                                                            st.metric("Upper BE", f"${rec['upper_breakeven']:.2f}")
                                                            st.metric("Lower BE", f"${rec['lower_breakeven']:.2f}")
                                                    
                                                    else:  # Spreads
                                                        st.metric("Max Profit %", f"{rec['max_profit_pct']:.1f}%")
                                                        st.metric("Move Required", f"{rec['breakeven_move_required_pct']:.1f}%")
                                                        if asset_class == 'FOREX':
                                                            st.metric("Breakeven", f"{rec['breakeven']:.5f}")
                                                        else:
                                                            st.metric("Breakeven", f"${rec['breakeven']:.2f}")
                                        
                                        # Optimal recommendation callout
                                        if contract_specs.get('optimal_recommendation'):
                                            optimal = contract_specs['optimal_recommendation']
                                            st.success(f"🎯 **Recommended:** {optimal['risk_level']} approach for balanced risk/reward")
                                    
                                    elif 'recommendation' in contract_specs:  # For straddle
                                        rec = contract_specs['recommendation']
                                        
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.markdown("**📊 Straddle Details:**")
                                            if asset_class == 'FOREX':
                                                st.metric("Strike", f"{rec['strike']:.5f}")
                                                st.metric("Call Price", f"{rec['call_price']:.4f}")
                                                st.metric("Put Price", f"{rec['put_price']:.4f}")
                                            else:
                                                st.metric("Strike", f"${rec['strike']:.2f}")
                                                st.metric("Call Price", f"${rec['call_price']:.2f}")
                                                st.metric("Put Price", f"${rec['put_price']:.2f}")
                                            st.metric("Contracts", rec['contracts'])
                                        
                                        with col2:
                                            st.markdown("**💰 Investment:**")
                                            st.metric("Total Cost", f"${rec['total_cost']:,.0f}")
                                            st.metric("Max Loss", f"${rec['max_loss']:,.0f}")
                                            if asset_class == 'FOREX':
                                                st.metric("Upper BE", f"{rec['upper_breakeven']:.5f}")
                                                st.metric("Lower BE", f"{rec['lower_breakeven']:.5f}")
                                            else:
                                                st.metric("Upper BE", f"${rec['upper_breakeven']:.2f}")
                                                st.metric("Lower BE", f"${rec['lower_breakeven']:.2f}")
                                        
                                        with col3:
                                            st.markdown("**📈 Expectations:**")
                                            st.metric("Required Move", f"{rec['required_move_pct']:.1f}%")
                                            st.metric("Expected Move", f"{rec['expected_move_pct']:.1f}%")
                                            st.metric("Profit Probability", f"{rec['profit_probability_estimate']:.0f}%")
                                            if asset_class == 'FOREX':
                                                st.metric("Breakeven Range", f"{rec['breakeven_range']:.5f}")
                                            else:
                                                st.metric("Breakeven Range", f"${rec['breakeven_range']:.2f}")
                                
                                else:
                                    st.info(f"📊 {best_strategy} contract analysis available - implement display logic")
                            
                            else:
                                st.error(f"❌ Contract calculation failed: {contract_specs['error']}")
                                if 'min_capital_needed' in contract_specs:
                                    st.info(f"💡 Minimum capital needed: ${contract_specs['min_capital_needed']:,.2f}")
                        
                        except Exception as e:
                            st.error(f"❌ Contract calculation failed: {str(e)}")
                
                # Options Data Summary
                st.subheader(f"🎯 {asset_class} Options Data Summary")
                
                options_data = result['options_data']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Expiration", options_data['expiration'])
                    st.metric("Days to Expiry", options_data['days_to_expiry'])
                
                with col2:
                    st.metric("Call Options", len(options_data['calls']))
                    st.metric("Put Options", len(options_data['puts']))
                
                with col3:
                    st.metric("Total Contracts", options_data['total_contracts'])
                    st.metric("Data Source", options_data.get('source', 'real').title())
                
                with col4:
                    pricing_breakdown = options_data.get('pricing_breakdown', {})
                    real_prices = pricing_breakdown.get('total_real', 0)
                    calc_prices = pricing_breakdown.get('total_calculated', 0)
                    
                    st.metric("Real Prices", real_prices)
                    st.metric("Calculated Prices", calc_prices)
                
                # Pricing methodology explanation
                if pricing_breakdown:
                    real_pct = (real_prices / (real_prices + calc_prices) * 100) if (real_prices + calc_prices) > 0 else 0
                    
                    if real_pct > 50:
                        st.success(f"✅ High data quality: {real_pct:.0f}% real market prices")
                    elif real_pct > 20:
                        st.warning(f"⚠️ Mixed data quality: {real_pct:.0f}% real prices, {100-real_pct:.0f}% calculated")
                    else:
                        st.info(f"📊 Calculated pricing: {calc_prices} options priced using Black-Scholes model")
                
                # Debug information
                if debug_mode and 'debug_info' in result:
                    st.subheader("🐛 Debug Information")
                    debug_info = result['debug_info']
                    
                    with st.expander("Debug Details"):
                        st.json(debug_info)
            
            else:
                st.warning(f"Analysis result is for {result.get('asset_class', 'unknown')} but current selection is {asset_class}")
        
        elif st.session_state.analysis_result and not st.session_state.analysis_result.get('success'):
            # Show error from previous analysis attempt
            result = st.session_state.analysis_result
            st.error(f"❌ Previous analysis failed: {result.get('error', 'Unknown error')}")
        
        else:
            # Show instructions for current asset class
            analysis_instructions = {
                'INDICES': """
                ## 📊 Index Options Analysis
                
                Analyze index ETFs and volatility products with professional-grade options strategies.
                
                ### 🎯 Index Analysis Features:
                - **Market-wide exposure** with diversification benefits
                - **Lower volatility** compared to individual stocks
                - **Sector rotation insights** and economic cycle considerations
                - **Popular instruments**: SPY, QQQ, IWM, EWU, VGK, VIX
                
                ### 💡 Index-Specific Considerations:
                - Index options often have better liquidity
                - Less susceptible to single-name risk
                - Volatility products (VIX) behave differently
                - Quarterly rebalancing can create opportunities
                """,
                
                'EQUITIES': """
                ## 📈 Equity Options Analysis
                
                Comprehensive analysis of individual stock options with company-specific insights.
                
                ### 🎯 Equity Analysis Features:
                - **Company-specific risk assessment** and earnings considerations
                - **Higher volatility** with greater profit potential
                - **Earnings-driven strategies** around announcement dates
                - **Popular instruments**: AAPL, MSFT, GOOGL, TSLA, NVDA, META
                
                ### 💡 Equity-Specific Considerations:
                - Higher gamma around earnings events
                - Single-name risk requires careful position sizing
                - Sector rotation can impact correlations
                - Corporate actions may affect options
                """,
                
                'FOREX': """
                ## 💱 FX Options Analysis
                
                Currency options analysis with global macro considerations and 24/5 market dynamics.
                
                ### 🎯 FX Analysis Features:
                - **Central bank policy impact** on currency movements
                - **24/5 trading** with continuous price discovery
                - **Global macro themes** and economic data sensitivity
                - **Popular instruments**: EUR/USD, GBP/USD, USD/JPY, AUD/USD
                
                ### 💡 FX-Specific Considerations:
                - Different volatility patterns vs equities
                - Interest rate differentials affect pricing
                - Central bank meetings create vol spikes
                - No traditional "volume" - focus on price action
                """
            }
            
            st.markdown(analysis_instructions[asset_class])

    # Tab 2: Strategy Guide
    with tab2:
        st.header(f"📚 {asset_class} Options Strategy Guide")
        
        # Get strategy explanations
        strategy_explanations = get_strategy_explanations()
        
        # Filter strategies by asset class
        applicable_strategies = {}
        for strategy_name, details in strategy_explanations.items():
            if asset_class in details.get('asset_classes', []):
                applicable_strategies[strategy_name] = details
        
        if not applicable_strategies:
            st.warning(f"No specific strategies configured for {asset_class}")
            applicable_strategies = strategy_explanations
        
        st.info(f"**{len(applicable_strategies)} strategies** available for {asset_class} options trading")
        
        # Strategy selector
        selected_strategy = st.selectbox(
            "📋 Select Strategy to Learn About:",
            list(applicable_strategies.keys()),
            format_func=lambda x: applicable_strategies[x]['name']
        )
        
        if selected_strategy:
            strategy = applicable_strategies[selected_strategy]
            
            # Strategy overview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"🎯 {strategy['name']}")
                st.write(strategy['description'])
                
                st.markdown("**📊 Market Outlook:**")
                st.info(strategy['market_outlook'])
                
                # Asset-specific example
                if asset_class in strategy.get('examples', {}):
                    st.markdown("**💡 Example for " + asset_class + ":**")
                    st.code(strategy['examples'][asset_class])
            
            with col2:
                st.markdown("**📈 Profit/Loss Profile:**")
                st.write(f"**Max Profit:** {strategy['max_profit']}")
                st.write(f"**Max Loss:** {strategy['max_loss']}")
                st.write(f"**Breakeven:** {strategy['breakeven']}")
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ When to Use")
            for point in strategy['when_to_use']:
                st.write(f"• {point}")
            
            st.markdown("### 👍 Advantages")
            for pro in strategy['pros']:
                st.write(f"✅ {pro}")
        
        with col2:
            st.markdown("### ⚠️ Considerations")
            for con in strategy['cons']:
                st.write(f"⚠️ {con}")
        
        # Asset class specific notes
        if asset_class == 'FOREX':
            st.markdown("### 💱 FX-Specific Notes")
            st.info("""
            **Currency options considerations:**
            • Settlement typically in cash, not physical delivery
            • Interest rate differentials affect pricing (rho more important)
            • Central bank meetings create volatility spikes
            • Different delta conventions (25Δ, 10Δ common)
            • 24/5 trading affects time decay calculations
            """)
        
        elif asset_class == 'INDICES':
            st.markdown("### 📊 Index-Specific Notes")
            st.info("""
            **Index options considerations:**
            • Lower volatility due to diversification
            • Less single-name risk
            • Quarterly rebalancing effects
            • European-style exercise for some indices
            • Better liquidity in popular ETF options
            """)
        
        elif asset_class == 'EQUITIES':
            st.markdown("### 📈 Equity-Specific Notes")
            st.info("""
            **Individual stock considerations:**
            • Earnings announcements create vol spikes
            • Company-specific news risk
            • Higher gamma potential
            • Dividend impacts on American-style options
            • Sector rotation affects correlations
            """)
        
        # Risk management section
        st.markdown("### ⚡ Risk Management")
        
        risk_guidelines = {
            'FOREX': [
                "Monitor central bank communications and policy changes",
                "Consider interest rate differential impacts",
                "Watch for intervention threats from authorities",
                "Use appropriate position sizing for leverage",
                "Monitor global risk sentiment shifts"
            ],
            'INDICES': [
                "Diversification reduces single-name risk",
                "Monitor sector rotation and economic cycle",
                "Consider correlation with broader market",
                "Watch for rebalancing effects",
                "Use for portfolio hedging strategies"
            ],
            'EQUITIES': [
                "Limit single-name concentration",
                "Monitor earnings calendar and announcements",
                "Consider sector and style factors",
                "Watch for corporate actions",
                "Use stop-losses for directional strategies"
            ]
        }
        
        for guideline in risk_guidelines[asset_class]:
            st.write(f"🛡️ {guideline}")

    # Tab 3: Options Greeks
    with tab3:
        st.header(f"🔢 {asset_class} Options Greeks")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            default_symbol = {
                'INDICES': 'SPY',
                'EQUITIES': 'AAPL', 
                'FOREX': 'EURUSD'
            }[asset_class]
            
            greeks_symbol = st.text_input(
                f"Symbol for Greeks Analysis ({asset_class})",
                value=default_symbol,
                help=f"Enter {asset_class.lower()} ticker symbol"
            )
        
        with col2:
            # Expiry selection
            expiry_options = {
                '1W': '1 Week',
                '2W': '2 Weeks', 
                '3W': '3 Weeks',
                '1M': '1 Month',
                '2M': '2 Months',
                '3M': '3 Months',
                '6M': '6 Months',
                '1Y': '1 Year',
                '2Y': '2 Years',
                '3Y': '3 Years',
                '4Y': '4 Years',
                '5Y': '5 Years'
            }
            
            selected_expiry = st.selectbox(
                "📅 Expiration Period",
                options=list(expiry_options.keys()),
                index=3,  # Default to 1M
                format_func=lambda x: expiry_options[x],
                help="Select the expiration timeframe for options analysis"
            )
        
        with col3:
            get_greeks_button = st.button(
                f"📊 Get {asset_class} Greeks",
                type="primary",
                disabled=not greeks_symbol
            )
        
        # Helper function to convert expiry selection to date range
        def get_expiry_date_range(expiry_code):
            """Convert expiry code to actual date range"""
            from datetime import datetime, timedelta
            
            today = datetime.now().date()
            
            if expiry_code == '1W':
                start_date = today + timedelta(days=5)
                end_date = today + timedelta(days=12)
            elif expiry_code == '2W':
                start_date = today + timedelta(days=12)
                end_date = today + timedelta(days=19)
            elif expiry_code == '3W':
                start_date = today + timedelta(days=19)
                end_date = today + timedelta(days=26)
            elif expiry_code == '1M':
                start_date = today + timedelta(days=26)
                end_date = today + timedelta(days=40)
            elif expiry_code == '2M':
                start_date = today + timedelta(days=55)
                end_date = today + timedelta(days=70)
            elif expiry_code == '3M':
                start_date = today + timedelta(days=85)
                end_date = today + timedelta(days=100)
            elif expiry_code == '6M':
                start_date = today + timedelta(days=170)
                end_date = today + timedelta(days=190)
            elif expiry_code == '1Y':
                start_date = today + timedelta(days=350)
                end_date = today + timedelta(days=380)
            elif expiry_code == '2Y':
                start_date = today + timedelta(days=715)
                end_date = today + timedelta(days=745)
            elif expiry_code == '3Y':
                start_date = today + timedelta(days=1080)
                end_date = today + timedelta(days=1110)
            elif expiry_code == '4Y':
                start_date = today + timedelta(days=1445)
                end_date = today + timedelta(days=1475)
            elif expiry_code == '5Y':
                start_date = today + timedelta(days=1810)
                end_date = today + timedelta(days=1840)
            else:
                # Default to 1M
                start_date = today + timedelta(days=26)
                end_date = today + timedelta(days=40)
            
            return start_date, end_date
        
        if get_greeks_button and greeks_symbol:
            with st.spinner(f"Fetching {asset_class} Options Greeks for {greeks_symbol} ({expiry_options[selected_expiry]})..."):
                try:
                    # Get the date range for selected expiry
                    start_date, end_date = get_expiry_date_range(selected_expiry)
                    
                    # Create a modified strategist method call with expiry selection
                    greeks_result = strategist.get_options_greeks_with_expiry(
                        greeks_symbol.upper(), 
                        asset_class,
                        selected_expiry,
                        start_date,
                        end_date
                    )
                    
                    st.session_state.greeks_result = greeks_result
                    
                    st.success(f"✅ {asset_class} Greeks analysis complete for {greeks_result['underlying_ticker']} ({expiry_options[selected_expiry]})")
                    
                    # Show selected expiry info
                    st.info(f"📅 **Showing options expiring:** {expiry_options[selected_expiry]} | **Actual Expiration:** {greeks_result['expiration']} | **Days to Expiry:** {greeks_result['days_to_expiry']}")
                    
                    # Summary metrics
                    st.subheader(f"📊 {asset_class} Greeks Summary ({expiry_options[selected_expiry]})")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if asset_class == 'FOREX':
                            st.metric("Underlying Price", f"{greeks_result['underlying_price']:.5f}")
                        else:
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
                        if asset_class == 'FOREX':
                            st.metric("Max Gamma Strike", f"{max_gamma_strike:.5f}")
                        else:
                            st.metric("Max Gamma Strike", f"${max_gamma_strike:.2f}")
                    
                    # Show available expiries for this symbol
                    available_expiries = greeks_result.get('available_expiries', [])
                    if available_expiries:
                        st.markdown("### 📅 Available Expiration Dates")
                        expiry_cols = st.columns(min(len(available_expiries), 6))
                        for i, expiry in enumerate(available_expiries[:6]):
                            with expiry_cols[i]:
                                days_until = (datetime.strptime(expiry, '%Y-%m-%d').date() - datetime.now().date()).days
                                st.metric(f"Exp {i+1}", expiry, f"{days_until}d")
                        
                        if len(available_expiries) > 6:
                            with st.expander(f"Show all {len(available_expiries)} available expiries"):
                                expiry_df = pd.DataFrame({
                                    'Expiration Date': available_expiries,
                                    'Days Until Expiry': [(datetime.strptime(exp, '%Y-%m-%d').date() - datetime.now().date()).days for exp in available_expiries]
                                })
                                st.dataframe(expiry_df, use_container_width=True)
                    
                    # Professional Options Chain View - ITM/ATM/OTM
                    st.subheader(f"🎯 {asset_class} Professional Options Chain ({expiry_options[selected_expiry]})")
                    
                    current_price = greeks_result['underlying_price']
                    calls_df = greeks_result['calls_greeks']
                    puts_df = greeks_result['puts_greeks']
                    
                    if not calls_df.empty and not puts_df.empty:
                        
                        # Categorize options
                        def categorize_options(calls_df, puts_df, current_price):
                            """Categorize options as ITM, ATM, OTM like professional platforms"""
                            
                            # ATM threshold (within 2% of current price)
                            atm_threshold = 0.02
                            
                            # Calls categorization
                            calls_itm = calls_df[calls_df['strike'] < current_price].copy()
                            calls_atm = calls_df[abs(calls_df['strike'] - current_price) / current_price <= atm_threshold].copy()
                            calls_otm = calls_df[calls_df['strike'] > current_price * (1 + atm_threshold)].copy()
                            
                            # Puts categorization  
                            puts_itm = puts_df[puts_df['strike'] > current_price].copy()
                            puts_atm = puts_df[abs(puts_df['strike'] - current_price) / current_price <= atm_threshold].copy()
                            puts_otm = puts_df[puts_df['strike'] < current_price * (1 - atm_threshold)].copy()
                            
                            return {
                                'calls_itm': calls_itm.sort_values('strike', ascending=False),
                                'calls_atm': calls_atm.sort_values('strike'),
                                'calls_otm': calls_otm.sort_values('strike'),
                                'puts_itm': puts_itm.sort_values('strike', ascending=False),
                                'puts_atm': puts_atm.sort_values('strike', ascending=False),
                                'puts_otm': puts_otm.sort_values('strike', ascending=False)
                            }
                        
                        categorized = categorize_options(calls_df, puts_df, current_price)
                        
                        # Professional-style options display
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📞 **CALLS**")
                            
                            # ITM Calls
                            if not categorized['calls_itm'].empty:
                                st.markdown("#### 🟢 **In-The-Money Calls**")
                                itm_calls_display = categorized['calls_itm'][['strike', 'price', 'delta', 'gamma', 'theta', 'vega']].head(5)
                                
                                if asset_class == 'FOREX':
                                    itm_calls_display['strike'] = itm_calls_display['strike'].apply(lambda x: f"{x:.5f}")
                                    itm_calls_display['price'] = itm_calls_display['price'].apply(lambda x: f"{x:.4f}")
                                else:
                                    itm_calls_display['strike'] = itm_calls_display['strike'].apply(lambda x: f"${x:.2f}")
                                    itm_calls_display['price'] = itm_calls_display['price'].apply(lambda x: f"${x:.2f}")
                                
                                st.dataframe(itm_calls_display, use_container_width=True)
                            
                            # ATM Calls
                            if not categorized['calls_atm'].empty:
                                st.markdown("#### 🎯 **At-The-Money Calls**")
                                atm_calls_display = categorized['calls_atm'][['strike', 'price', 'delta', 'gamma', 'theta', 'vega']]
                                
                                if asset_class == 'FOREX':
                                    atm_calls_display['strike'] = atm_calls_display['strike'].apply(lambda x: f"{x:.5f}")
                                    atm_calls_display['price'] = atm_calls_display['price'].apply(lambda x: f"{x:.4f}")
                                else:
                                    atm_calls_display['strike'] = atm_calls_display['strike'].apply(lambda x: f"${x:.2f}")
                                    atm_calls_display['price'] = atm_calls_display['price'].apply(lambda x: f"${x:.2f}")
                                
                                st.dataframe(atm_calls_display, use_container_width=True)
                            
                            # OTM Calls
                            if not categorized['calls_otm'].empty:
                                st.markdown("#### 🔴 **Out-The-Money Calls**")
                                otm_calls_display = categorized['calls_otm'][['strike', 'price', 'delta', 'gamma', 'theta', 'vega']].head(10)
                                
                                if asset_class == 'FOREX':
                                    otm_calls_display['strike'] = otm_calls_display['strike'].apply(lambda x: f"{x:.5f}")
                                    otm_calls_display['price'] = otm_calls_display['price'].apply(lambda x: f"{x:.4f}")
                                else:
                                    otm_calls_display['strike'] = otm_calls_display['strike'].apply(lambda x: f"${x:.2f}")
                                    otm_calls_display['price'] = otm_calls_display['price'].apply(lambda x: f"${x:.2f}")
                                
                                st.dataframe(otm_calls_display, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 📱 **PUTS**")
                            
                            # ITM Puts
                            if not categorized['puts_itm'].empty:
                                st.markdown("#### 🟢 **In-The-Money Puts**")
                                itm_puts_display = categorized['puts_itm'][['strike', 'price', 'delta', 'gamma', 'theta', 'vega']].head(5)
                                
                                if asset_class == 'FOREX':
                                    itm_puts_display['strike'] = itm_puts_display['strike'].apply(lambda x: f"{x:.5f}")
                                    itm_puts_display['price'] = itm_puts_display['price'].apply(lambda x: f"{x:.4f}")
                                else:
                                    itm_puts_display['strike'] = itm_puts_display['strike'].apply(lambda x: f"${x:.2f}")
                                    itm_puts_display['price'] = itm_puts_display['price'].apply(lambda x: f"${x:.2f}")
                                
                                st.dataframe(itm_puts_display, use_container_width=True)
                            
                            # ATM Puts
                            if not categorized['puts_atm'].empty:
                                st.markdown("#### 🎯 **At-The-Money Puts**")
                                atm_puts_display = categorized['puts_atm'][['strike', 'price', 'delta', 'gamma', 'theta', 'vega']]
                                
                                if asset_class == 'FOREX':
                                    atm_puts_display['strike'] = atm_puts_display['strike'].apply(lambda x: f"{x:.5f}")
                                    atm_puts_display['price'] = atm_puts_display['price'].apply(lambda x: f"{x:.4f}")
                                else:
                                    atm_puts_display['strike'] = atm_puts_display['strike'].apply(lambda x: f"${x:.2f}")
                                    atm_puts_display['price'] = atm_puts_display['price'].apply(lambda x: f"${x:.2f}")
                                
                                st.dataframe(atm_puts_display, use_container_width=True)
                            
                            # OTM Puts
                            if not categorized['puts_otm'].empty:
                                st.markdown("#### 🔴 **Out-The-Money Puts**")
                                otm_puts_display = categorized['puts_otm'][['strike', 'price', 'delta', 'gamma', 'theta', 'vega']].head(10)
                                
                                if asset_class == 'FOREX':
                                    otm_puts_display['strike'] = otm_puts_display['strike'].apply(lambda x: f"{x:.5f}")
                                    otm_puts_display['price'] = otm_puts_display['price'].apply(lambda x: f"{x:.4f}")
                                else:
                                    otm_puts_display['strike'] = otm_puts_display['strike'].apply(lambda x: f"${x:.2f}")
                                    otm_puts_display['price'] = otm_puts_display['price'].apply(lambda x: f"${x:.2f}")
                                
                                st.dataframe(otm_puts_display, use_container_width=True)
                    
                    # Delta-Specific View
                    st.subheader(f"⚡ {asset_class} Delta-Specific Options ({expiry_options[selected_expiry]})")
                    st.markdown("**Key Delta Levels for Professional Trading**")
                    
                    target_deltas = [-0.25, -0.1, 0.0, 0.1, 0.25]
                    
                    def find_closest_delta_options(calls_df, puts_df, target_deltas):
                        """Find options closest to specific delta values"""
                        delta_options = []
                        
                        for target_delta in target_deltas:
                            if target_delta == 0.0:
                                # ATM options (closest to 0.5 delta for calls, -0.5 for puts)
                                if not calls_df.empty:
                                    closest_call = calls_df.iloc[(calls_df['delta'] - 0.5).abs().argsort()[:1]]
                                    if not closest_call.empty:
                                        delta_options.append({
                                            'target_delta': 0.0,
                                            'type': 'CALL (ATM)',
                                            'actual_delta': closest_call.iloc[0]['delta'],
                                            'strike': closest_call.iloc[0]['strike'],
                                            'price': closest_call.iloc[0]['price'],
                                            'gamma': closest_call.iloc[0]['gamma'],
                                            'theta': closest_call.iloc[0]['theta'],
                                            'vega': closest_call.iloc[0]['vega']
                                        })
                                
                                if not puts_df.empty:
                                    closest_put = puts_df.iloc[(puts_df['delta'] - (-0.5)).abs().argsort()[:1]]
                                    if not closest_put.empty:
                                        delta_options.append({
                                            'target_delta': 0.0,
                                            'type': 'PUT (ATM)',
                                            'actual_delta': closest_put.iloc[0]['delta'],
                                            'strike': closest_put.iloc[0]['strike'],
                                            'price': closest_put.iloc[0]['price'],
                                            'gamma': closest_put.iloc[0]['gamma'],
                                            'theta': closest_put.iloc[0]['theta'],
                                            'vega': closest_put.iloc[0]['vega']
                                        })
                            
                            elif target_delta > 0:
                                # Positive delta - look in calls
                                if not calls_df.empty:
                                    closest_call = calls_df.iloc[(calls_df['delta'] - target_delta).abs().argsort()[:1]]
                                    if not closest_call.empty:
                                        delta_options.append({
                                            'target_delta': target_delta,
                                            'type': 'CALL',
                                            'actual_delta': closest_call.iloc[0]['delta'],
                                            'strike': closest_call.iloc[0]['strike'],
                                            'price': closest_call.iloc[0]['price'],
                                            'gamma': closest_call.iloc[0]['gamma'],
                                            'theta': closest_call.iloc[0]['theta'],
                                            'vega': closest_call.iloc[0]['vega']
                                        })
                            
                            else:
                                # Negative delta - look in puts
                                if not puts_df.empty:
                                    closest_put = puts_df.iloc[(puts_df['delta'] - target_delta).abs().argsort()[:1]]
                                    if not closest_put.empty:
                                        delta_options.append({
                                            'target_delta': target_delta,
                                            'type': 'PUT',
                                            'actual_delta': closest_put.iloc[0]['delta'],
                                            'strike': closest_put.iloc[0]['strike'],
                                            'price': closest_put.iloc[0]['price'],
                                            'gamma': closest_put.iloc[0]['gamma'],
                                            'theta': closest_put.iloc[0]['theta'],
                                            'vega': closest_put.iloc[0]['vega']
                                        })
                        
                        return delta_options
                    
                    delta_options = find_closest_delta_options(calls_df, puts_df, target_deltas)
                    
                    if delta_options:
                        # Create professional delta table
                        delta_df = pd.DataFrame(delta_options)
                        
                        # Format for display
                        display_delta_df = delta_df.copy()
                        
                        if asset_class == 'FOREX':
                            display_delta_df['strike'] = display_delta_df['strike'].apply(lambda x: f"{x:.5f}")
                            display_delta_df['price'] = display_delta_df['price'].apply(lambda x: f"{x:.4f}")
                        else:
                            display_delta_df['strike'] = display_delta_df['strike'].apply(lambda x: f"${x:.2f}")
                            display_delta_df['price'] = display_delta_df['price'].apply(lambda x: f"${x:.2f}")
                        
                        # Round Greeks
                        display_delta_df['actual_delta'] = display_delta_df['actual_delta'].apply(lambda x: f"{x:.3f}")
                        display_delta_df['gamma'] = display_delta_df['gamma'].apply(lambda x: f"{x:.4f}")
                        display_delta_df['theta'] = display_delta_df['theta'].apply(lambda x: f"{x:.4f}")
                        display_delta_df['vega'] = display_delta_df['vega'].apply(lambda x: f"{x:.4f}")
                        
                        # Rename columns for professional display
                        display_delta_df.columns = ['Target Δ', 'Option Type', 'Actual Δ', 'Strike', 'Price', 'Gamma', 'Theta', 'Vega']
                        
                        st.dataframe(display_delta_df, use_container_width=True)
                        
                        # Delta explanation with expiry considerations
                        days_to_expiry = greeks_result['days_to_expiry']
                        if days_to_expiry <= 30:
                            time_warning = "⚠️ **Short-term expiry**: Higher gamma risk, faster theta decay"
                        elif days_to_expiry <= 90:
                            time_warning = "📊 **Medium-term expiry**: Balanced gamma and theta"
                        else:
                            time_warning = "📈 **Long-term expiry**: Lower gamma, higher vega sensitivity"
                        
                        st.info(f"""
                        **🎯 Delta Trading Guide ({expiry_options[selected_expiry]}):**
                        • **±0.25 Delta**: Moderate directional exposure, good for spreads
                        • **±0.10 Delta**: Low directional exposure, high time decay
                        • **0.00 Delta (ATM)**: Maximum gamma, balanced time decay
                        • **Higher |Delta|**: More directional, less time decay
                        • **Lower |Delta|**: Less directional, more time decay
                        
                        {time_warning}
                        """)
                    
                    # Complete Greeks Tables
                    st.subheader(f"📊 Complete {asset_class} Greeks Analysis ({expiry_options[selected_expiry]})")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📞 **Call Options Greeks**")
                        if not calls_df.empty:
                            # Prepare display dataframe
                            calls_display = calls_df[['strike', 'price', 'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_vol', 'moneyness']].copy()
                            calls_display = calls_display.sort_values('strike')
                            
                            # Format based on asset class
                            if asset_class == 'FOREX':
                                calls_display['strike'] = calls_display['strike'].apply(lambda x: f"{x:.5f}")
                                calls_display['price'] = calls_display['price'].apply(lambda x: f"{x:.4f}")
                            else:
                                calls_display['strike'] = calls_display['strike'].apply(lambda x: f"${x:.2f}")
                                calls_display['price'] = calls_display['price'].apply(lambda x: f"${x:.2f}")
                            
                            # Format Greeks
                            calls_display['delta'] = calls_display['delta'].apply(lambda x: f"{x:.3f}")
                            calls_display['gamma'] = calls_display['gamma'].apply(lambda x: f"{x:.4f}")
                            calls_display['theta'] = calls_display['theta'].apply(lambda x: f"{x:.4f}")
                            calls_display['vega'] = calls_display['vega'].apply(lambda x: f"{x:.4f}")
                            calls_display['rho'] = calls_display['rho'].apply(lambda x: f"{x:.4f}")
                            calls_display['implied_vol'] = calls_display['implied_vol'].apply(lambda x: f"{x:.1%}")
                            calls_display['moneyness'] = calls_display['moneyness'].apply(lambda x: f"{x:.3f}")
                            
                            # Rename columns
                            calls_display.columns = ['Strike', 'Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'IV', 'Moneyness']
                            
                            st.dataframe(calls_display, use_container_width=True, height=400)
                        else:
                            st.info("No call options data available")
                    
                    with col2:
                        st.markdown("#### 📱 **Put Options Greeks**")
                        if not puts_df.empty:
                            # Prepare display dataframe
                            puts_display = puts_df[['strike', 'price', 'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_vol', 'moneyness']].copy()
                            puts_display = puts_display.sort_values('strike', ascending=False)
                            
                            # Format based on asset class
                            if asset_class == 'FOREX':
                                puts_display['strike'] = puts_display['strike'].apply(lambda x: f"{x:.5f}")
                                puts_display['price'] = puts_display['price'].apply(lambda x: f"{x:.4f}")
                            else:
                                puts_display['strike'] = puts_display['strike'].apply(lambda x: f"${x:.2f}")
                                puts_display['price'] = puts_display['price'].apply(lambda x: f"${x:.2f}")
                            
                            # Format Greeks
                            puts_display['delta'] = puts_display['delta'].apply(lambda x: f"{x:.3f}")
                            puts_display['gamma'] = puts_display['gamma'].apply(lambda x: f"{x:.4f}")
                            puts_display['theta'] = puts_display['theta'].apply(lambda x: f"{x:.4f}")
                            puts_display['vega'] = puts_display['vega'].apply(lambda x: f"{x:.4f}")
                            puts_display['rho'] = puts_display['rho'].apply(lambda x: f"{x:.4f}")
                            puts_display['implied_vol'] = puts_display['implied_vol'].apply(lambda x: f"{x:.1%}")
                            puts_display['moneyness'] = puts_display['moneyness'].apply(lambda x: f"{x:.3f}")
                            
                            # Rename columns
                            puts_display.columns = ['Strike', 'Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'IV', 'Moneyness']
                            
                            st.dataframe(puts_display, use_container_width=True, height=400)
                        else:
                            st.info("No put options data available")
                    
                    # Expiry-Specific Greeks Analysis
                    st.subheader(f"⏰ {expiry_options[selected_expiry]} Greeks Characteristics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        **📊 Time Decay Analysis ({expiry_options[selected_expiry]}):**
                        • **Days to Expiry**: {greeks_result['days_to_expiry']}
                        • **Theta Acceleration**: {'High' if greeks_result['days_to_expiry'] < 30 else 'Medium' if greeks_result['days_to_expiry'] < 90 else 'Low'}
                        • **Weekend Risk**: {'Significant' if greeks_result['days_to_expiry'] < 45 else 'Moderate'}
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **⚡ Gamma Exposure ({expiry_options[selected_expiry]}):**
                        • **Gamma Risk**: {'Very High' if greeks_result['days_to_expiry'] < 7 else 'High' if greeks_result['days_to_expiry'] < 30 else 'Moderate'}
                        • **Delta Sensitivity**: {'Extreme' if greeks_result['days_to_expiry'] < 7 else 'High' if greeks_result['days_to_expiry'] < 30 else 'Normal'}
                        • **Pin Risk**: {'Critical' if greeks_result['days_to_expiry'] < 3 else 'Monitor' if greeks_result['days_to_expiry'] < 7 else 'Low'}
                        """)
                    
                    with col3:
                        st.markdown(f"""
                        **📈 Vega Impact ({expiry_options[selected_expiry]}):**
                        • **Vol Sensitivity**: {'Low' if greeks_result['days_to_expiry'] < 30 else 'High' if greeks_result['days_to_expiry'] > 180 else 'Medium'}
                        • **Event Risk**: {'Low' if greeks_result['days_to_expiry'] < 7 else 'High' if greeks_result['days_to_expiry'] > 30 else 'Medium'}
                        • **Vega Decay**: {'Fast' if greeks_result['days_to_expiry'] < 30 else 'Slow'}
                        """)
                    
                    # Greeks Definitions
                    st.subheader("📚 Greeks Reference Guide")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        **🔥 Primary Greeks:**
                        • **Delta (Δ)**: Price sensitivity to underlying movement
                        • **Gamma (Γ)**: Rate of change of delta
                        • **Theta (Θ)**: Time decay per day
                        """)
                    
                    with col2:
                        st.markdown("""
                        **📊 Secondary Greeks:**
                        • **Vega (ν)**: Sensitivity to volatility changes
                        • **Rho (ρ)**: Sensitivity to interest rate changes
                        • **IV**: Implied volatility percentage
                        """)
                    
                    with col3:
                        st.markdown("""
                        **🎯 Key Metrics:**
                        • **Moneyness**: Strike/Spot ratio
                        • **ITM**: In-the-money (intrinsic value)
                        • **ATM**: At-the-money (max gamma)
                        • **OTM**: Out-of-the-money (time value)
                        """)
                
                except Exception as e:
                    st.error(f"❌ {asset_class} Greeks analysis failed: {str(e)}")
                    st.error("This may be due to limited options data for the selected expiry period. Try a different expiration timeframe.")
        
        else:
            # Greeks instructions for current asset class
            greeks_instructions = {
                'INDICES': """
                ## 🔢 Index Options Greeks Analysis
                
                Professional Greeks analysis for index products with multiple expiration periods.
                
                ### 🎯 Index Greeks Features:
                - **📅 Multiple Expiries**: 1W to 5Y options chains
                - **Professional Options Chain**: ITM/ATM/OTM categorization
                - **Delta-Specific Views**: Key delta levels for spread strategies
                - **Complete Greeks Matrix**: All Greeks with implied volatility
                
                ### 💡 Index-Specific Considerations:
                - Lower gamma due to diversification
                - Consistent theta decay patterns across expiries
                - Lower vega sensitivity vs individual stocks
                - Quarterly rebalancing effects on long-term options
                """,
                
                'EQUITIES': """
                ## 🔢 Equity Options Greeks Analysis
                
                Comprehensive Greeks analysis for individual stocks across all expiration cycles.
                
                ### 🎯 Equity Greeks Features:
                - **📅 Full Expiry Range**: Weekly to multi-year LEAPS
                - **Exchange-Style Display**: Professional ITM/ATM/OTM layout
                - **Delta Trading Levels**: -0.25, -0.1, 0.0, 0.1, 0.25 delta options
                - **Complete Risk Metrics**: All Greeks plus moneyness ratios
                
                ### 💡 Equity-Specific Considerations:
                - Higher gamma around earnings (especially weeklies)
                - More volatile theta patterns near expiry
                - Elevated vega before events
                - LEAPS provide different risk/reward profiles
                """,
                
                'FOREX': """
                ## 🔢 FX Options Greeks Analysis
                
                Currency options Greeks with 24/5 market considerations across all tenors.
                
                ### 🎯 FX Greeks Features:
                - **📅 Standard FX Tenors**: 1W, 1M, 3M, 6M, 1Y, 2Y+ 
                - **Professional FX Chain**: Currency-specific strike formatting
                - **Delta Hedging Levels**: Key levels for FX options trading
                - **Central Bank Greeks**: Impact of policy changes on Greeks
                
                ### 💡 FX-Specific Considerations:
                - Different delta conventions (25Δ, 10Δ popular)
                - Interest rate differential impacts (rho more important)
                - Central bank event vega spikes
                - Continuous trading affects gamma/theta behavior
                """
            }
            
            st.markdown(greeks_instructions[asset_class])

    # Tab 4: Backtester
    with tab4:
        st.header(f"📈 {asset_class} Options Strategy Backtester")
        
        # Backtester configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_symbols = {
                'INDICES': 'SPY',
                'EQUITIES': 'AAPL', 
                'FOREX': 'EURUSD'
            }
            
            backtest_symbol = st.text_input(
                f"Symbol for Backtest ({asset_class})",
                value=default_symbols[asset_class],
                help=f"Enter {asset_class.lower()} ticker symbol"
            )
        
        with col2:
            strategies = [
                'COVERED_CALL', 
                'CASH_SECURED_PUT', 
                'IRON_CONDOR', 
                'BULL_CALL_SPREAD', 
                'BEAR_PUT_SPREAD',
                'BUY_AND_HOLD'
            ]
            
            # Filter strategies by asset class
            if asset_class == 'FOREX':
                # FX specific strategies
                available_strategies = ['IRON_CONDOR', 'BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', 'BUY_AND_HOLD']
            else:
                available_strategies = strategies
            
            selected_strategy = st.selectbox(
                "Strategy to Backtest",
                available_strategies,
                help=f"Choose strategy for {asset_class} backtesting"
            )
        
        with col3:
            backtest_period = st.selectbox(
                "Backtest Period",
                ['6M', '1Y', '2Y', '3Y'],
                index=1,
                help="Historical period for backtesting"
            )
        
        # Strategy parameters
        st.subheader("🔧 Strategy Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days_to_expiry = st.slider(
                "Days to Expiry",
                min_value=7,
                max_value=90,
                value=30,
                help="Target days to expiration for options"
            )
        
        with col2:
            delta_target = st.slider(
                "Delta Target",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Target delta for option selection"
            )
        
        with col3:
            if asset_class == 'FOREX':
                wing_width = st.slider(
                    "Wing Width (%)",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Wing width for FX spreads (%)"
                ) / 100
            else:
                wing_width = st.slider(
                    "Wing Width (%)",
                    min_value=2.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.5,
                    help="Wing width for spreads (%)"
                ) / 100
        
        run_backtest_button = st.button(
            f"🔄 Run {asset_class} Backtest",
            type="primary",
            disabled=not backtest_symbol
        )
        
        if run_backtest_button and backtest_symbol:
            
            # Calculate backtest dates
            period_days = {'6M': 180, '1Y': 365, '2Y': 730, '3Y': 1095}[backtest_period]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            with st.spinner(f"Running {asset_class} {selected_strategy} backtest for {backtest_symbol} ({backtest_period})..."):
                
                # Create strategy parameters
                strategy_params = {
                    'days_to_expiry': days_to_expiry,
                    'delta_target': delta_target,
                    'wing_width': wing_width
                }
                
                backtest_result = strategist.run_strategy_backtest(
                    backtest_symbol.upper(),
                    asset_class,
                    selected_strategy,
                    start_date,
                    end_date,
                    strategy_params
                )
                
                if backtest_result['success']:
                    st.session_state.backtest_result = backtest_result
                    
                    st.success(f"✅ {asset_class} backtest complete: {selected_strategy} on {backtest_symbol}")
                    
                    # Performance Summary
                    st.subheader(f"📊 {asset_class} Backtest Results")
                    
                    metrics = backtest_result['performance_metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return']:.1f}%")
                        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    
                    with col2:
                        st.metric("Annualized Return", f"{metrics['annualized_return']:.1f}%")
                        st.metric("Total Trades", metrics['total_trades'])
                    
                    with col3:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                        st.metric("Avg P&L per Trade", f"${metrics['avg_pnl_per_trade']:.0f}")
                    
                    with col4:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
                        st.metric("Final Value", f"${metrics['final_portfolio_value']:,.0f}")
                    
                    # Performance Chart with Enhanced Technical Analysis
                    st.subheader(f"📈 {asset_class} Portfolio Performance & Technical Analysis")
                    
                    # Create two charts: Performance and Technical
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 💰 **Portfolio Performance**")
                        portfolio_values = backtest_result['results']['portfolio_values']
                        if portfolio_values:
                            portfolio_df = pd.DataFrame({
                                'Portfolio Value': portfolio_values
                            })
                            portfolio_df.index = pd.date_range(start=start_date, periods=len(portfolio_values), freq='D')
                            
                            # Create performance chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=portfolio_df.index,
                                y=portfolio_df['Portfolio Value'],
                                mode='lines',
                                name=f'{selected_strategy} Portfolio',
                                line=dict(color='#00ff88', width=2)
                            ))
                            
                            # Add initial investment line
                            fig.add_hline(
                                y=10000,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Initial Investment",
                                annotation_position="bottom right"
                            )
                            
                            fig.update_layout(
                                title=f'{asset_class} {selected_strategy} Performance',
                                xaxis_title='Date',
                                yaxis_title='Portfolio Value ($)',
                                template='plotly_dark',
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 📊 **Technical Analysis**")
                        try:
                            # Get underlying data for technical chart
                            underlying_data = strategist.get_asset_data(backtest_symbol.upper(), asset_class, days=500)
                            
                            # Create technical chart with all SMAs
                            chart_data = {
                                'ticker': backtest_symbol.upper(),
                                'current_price': underlying_data['current_price'],
                                'historical_data': underlying_data['historical_data']
                            }
                            
                            # Add 52-week high/low as support/resistance
                            support_resistance = {
                                'support_level': underlying_data['low_52w'],
                                'resistance_level': underlying_data['high_52w']
                            }
                            
                            technical_chart = strategist.create_trading_chart(chart_data, asset_class, support_resistance)
                            technical_chart.update_layout(height=300, title=f'{asset_class} Technical Analysis')
                            st.plotly_chart(technical_chart, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Could not create technical chart: {str(e)}")
                    
                    # Trade Analysis
                    st.subheader(f"📋 {asset_class} Trade Analysis")
                    
                    trades = backtest_result['results']['trades']
                    if trades:
                        # Convert to DataFrame for analysis
                        trades_df = pd.DataFrame(trades)
                        
                        # Profit/Loss distribution
                        profitable_trades = trades_df[trades_df['pnl'] > 0]
                        losing_trades = trades_df[trades_df['pnl'] <= 0]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**💰 Profit Distribution:**")
                            if not profitable_trades.empty:
                                avg_win = profitable_trades['pnl'].mean()
                                max_win = profitable_trades['pnl'].max()
                                st.metric("Average Win", f"${avg_win:.0f}")
                                st.metric("Best Trade", f"${max_win:.0f}")
                            else:
                                st.metric("Average Win", "$0")
                                st.metric("Best Trade", "$0")
                        
                        with col2:
                            st.markdown("**📉 Loss Distribution:**")
                            if not losing_trades.empty:
                                avg_loss = losing_trades['pnl'].mean()
                                max_loss = losing_trades['pnl'].min()
                                st.metric("Average Loss", f"${avg_loss:.0f}")
                                st.metric("Worst Trade", f"${max_loss:.0f}")
                            else:
                                st.metric("Average Loss", "$0")
                                st.metric("Worst Trade", "$0")
                        
                        # Show sample trades
                        st.markdown("**📊 Sample Trades:**")
                        display_trades = trades_df.head(10).copy()
                        
                        # Format dates and prices for display
                        display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
                        display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
                        
                        if asset_class == 'FOREX':
                            display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"{x:.5f}")
                            display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f"{x:.5f}")
                        else:
                            display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"${x:.2f}")
                            display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f"${x:.2f}")
                        
                        display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f"${x:.0f}")
                        
                        # Select relevant columns
                        relevant_cols = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl']
                        if selected_strategy in ['COVERED_CALL', 'CASH_SECURED_PUT']:
                            if 'call_premium' in display_trades.columns:
                                relevant_cols.append('call_premium')
                            if 'put_premium' in display_trades.columns:
                                relevant_cols.append('put_premium')
                        
                        display_trades = display_trades[relevant_cols]
                        st.dataframe(display_trades, use_container_width=True)
                    
                    # Strategy-specific insights
                    strategy_insights = {
                        'COVERED_CALL': f"""
                        **📊 {asset_class} Covered Call Insights:**
                        • This strategy performs best in sideways to slightly bullish markets
                        • Premium collection provides downside cushion but caps upside
                        • Consider adjusting strike selection based on volatility regime
                        """,
                        
                        'CASH_SECURED_PUT': f"""
                        **📊 {asset_class} Cash Secured Put Insights:**
                        • Strategy acquires assets at effective discount when assigned
                        • Works well in bullish trends with occasional pullbacks
                        • Premium collection reduces effective cost basis
                        """,
                        
                        'IRON_CONDOR': f"""
                        **📊 {asset_class} Iron Condor Insights:**
                        • Optimal in low volatility, range-bound environments
                        • {'FX pairs often trade in ranges, making this strategy suitable' if asset_class == 'FOREX' else 'Benefits from time decay in stable markets'}
                        • Adjust wing widths based on expected volatility
                        """,
                        
                        'BULL_CALL_SPREAD': f"""
                        **📊 {asset_class} Bull Call Spread Insights:**
                        • Limited risk, limited reward strategy for moderate bull markets
                        • Cost reduction vs buying calls outright
                        • Performance depends on reaching upper strike by expiration
                        """,
                        
                        'BEAR_PUT_SPREAD': f"""
                        **📊 {asset_class} Bear Put Spread Insights:**
                        • Profits from moderate downward movement
                        • Cheaper than buying puts outright
                        • Best in declining but not crashing markets
                        """,
                        
                        'BUY_AND_HOLD': f"""
                        **📊 {asset_class} Buy & Hold Insights:**
                        • Baseline strategy for comparison with options strategies
                        • {'Currency exposure with potential central bank impacts' if asset_class == 'FOREX' else 'Long-term asset appreciation with dividend/distribution potential'}
                        • No options complexity or time decay concerns
                        """
                    }
                    
                    if selected_strategy in strategy_insights:
                        st.info(strategy_insights[selected_strategy])
                
                else:
                    st.error(f"❌ Backtest failed: {backtest_result.get('error', 'Unknown error')}")
        
        else:
            # Show backtest instructions for current asset class
            backtest_instructions = {
                'INDICES': """
                ## 📈 Index Options Backtesting
                
                Test your strategies on index ETFs with historical performance analysis.
                
                ### 🎯 Index Backtesting Features:
                - **Multiple strategies** optimized for index behavior
                - **Risk-adjusted returns** with Sharpe ratio analysis
                - **Drawdown analysis** for risk management
                - **Trade-by-trade breakdown** for strategy refinement
                
                ### 💡 Index-Specific Considerations:
                - Lower volatility provides steadier returns
                - Diversification reduces single-name risk
                - Quarterly rebalancing effects on long-term strategies
                - Better suited for systematic options strategies
                """,
                
                'EQUITIES': """
                ## 📈 Equity Options Backtesting
                
                Comprehensive backtesting for individual stock options strategies.
                
                ### 🎯 Equity Backtesting Features:
                - **Company-specific analysis** with earnings considerations
                - **Volatility regime testing** across different market periods
                - **Single-name risk assessment** and position sizing guidance
                - **Performance comparison** vs buy-and-hold
                
                ### 💡 Equity-Specific Considerations:
                - Higher volatility creates more opportunities and risks
                - Earnings events significantly impact strategy performance
                - Company-specific news can override technical signals
                - Sector rotation affects strategy effectiveness
                """,
                
                'FOREX': """
                ## 📈 FX Options Backtesting
                
                Currency options strategy testing with 24/5 market dynamics.
                
                ### 🎯 FX Backtesting Features:
                - **Central bank event analysis** and policy impact assessment
                - **Interest rate differential effects** on strategy performance
                - **Global macro theme testing** across different economic cycles
                - **Volatility regime analysis** for FX-specific patterns
                
                ### 💡 FX-Specific Considerations:
                - Different volatility patterns vs equity markets
                - Central bank interventions can disrupt strategies
                - Interest rate changes affect option pricing significantly
                - 24/5 trading creates different risk management needs
                """
            }
            
            st.markdown(backtest_instructions[asset_class])

    # Tab 5: Market Predictions
    with tab5:
        st.header(f"🔮 {asset_class} Market Predictions")
        
        # Prediction configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_symbols = {
                'INDICES': 'SPY',
                'EQUITIES': 'AAPL', 
                'FOREX': 'EURUSD'
            }
            
            prediction_symbol = st.text_input(
                f"Symbol for Prediction ({asset_class})",
                value=default_symbols[asset_class],
                help=f"Enter {asset_class.lower()} ticker symbol"
            )
        
        with col2:
            prediction_days = st.selectbox(
                "Prediction Timeframe",
                [7, 14, 30, 60, 90],
                index=2,
                format_func=lambda x: f"{x} days",
                help="Number of days ahead to predict"
            )
        
        with col3:
            generate_prediction_button = st.button(
                f"🔮 Generate {asset_class} Prediction",
                type="primary",
                disabled=not prediction_symbol
            )
        
        if generate_prediction_button and prediction_symbol:
            with st.spinner(f"Generating {asset_class} market prediction for {prediction_symbol} ({prediction_days} days)..."):
                
                prediction_result = strategist.generate_market_prediction(
                    prediction_symbol.upper(),
                    asset_class,
                    prediction_days
                )
                
                if prediction_result['success']:
                    st.session_state.prediction_result = prediction_result
                    
                    st.success(f"✅ {asset_class} prediction complete for {prediction_result['ticker']}")
                    
                    # Current price and basic info
                    current_price = prediction_result['current_price']
                    predictions = prediction_result['price_predictions']
                    confidence = prediction_result['confidence_score']
                    
                    # Price Predictions Chart
                    st.subheader(f"📊 {asset_class} Price Predictions Chart")
                    
                    try:
                        # Create chart with predictions using data from prediction result
                        chart_data = {
                            'ticker': prediction_result['ticker'],
                            'current_price': current_price,
                            'historical_data': prediction_result['historical_data']
                        }
                        
                        # Include all prediction levels on chart
                        prediction_levels = {
                            'support_level': predictions['support_level'],
                            'resistance_level': predictions['resistance_level'],
                            'target_price': predictions['target_price'],
                            'bull_case': predictions['bull_case'],
                            'bear_case': predictions['bear_case']
                        }
                        
                        prediction_chart = strategist.create_trading_chart(chart_data, asset_class, prediction_levels)
                        st.plotly_chart(prediction_chart, use_container_width=True)
                        
                        # Chart legend
                        st.info("""
                        **📊 Chart Legend:**
                        • **🟢 Support**: Key support level based on recent lows
                        • **🔴 Resistance**: Key resistance level based on recent highs  
                        • **🟡 Target**: Base case prediction target
                        • **🔵 Bull Case**: Optimistic scenario target
                        • **🟣 Bear Case**: Pessimistic scenario target
                        • **SMAs**: 20, 40, 50, 100, 150, 200 period moving averages (where data allows)
                        """)
                        
                    except Exception as e:
                        st.error(f"Could not create prediction chart: {str(e)}")
                    
                    # Price Predictions
                    st.subheader(f"🎯 {asset_class} Price Predictions ({prediction_days} days)")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if asset_class == 'FOREX':
                            st.metric("Current Price", f"{current_price:.5f}")
                            st.metric("Target Price", f"{predictions['target_price']:.5f}", 
                                     f"{predictions['expected_move_pct']:+.1f}%")
                        else:
                            st.metric("Current Price", f"${current_price:.2f}")
                            st.metric("Target Price", f"${predictions['target_price']:.2f}", 
                                     f"{predictions['expected_move_pct']:+.1f}%")
                    
                    with col2:
                        if asset_class == 'FOREX':
                            st.metric("Bull Case", f"{predictions['bull_case']:.5f}")
                            st.metric("Bear Case", f"{predictions['bear_case']:.5f}")
                        else:
                            st.metric("Bull Case", f"${predictions['bull_case']:.2f}")
                            st.metric("Bear Case", f"${predictions['bear_case']:.2f}")
                    
                    with col3:
                        if asset_class == 'FOREX':
                            st.metric("Resistance", f"{predictions['resistance_level']:.5f}")
                            st.metric("Support", f"{predictions['support_level']:.5f}")
                        else:
                            st.metric("Resistance", f"${predictions['resistance_level']:.2f}")
                            st.metric("Support", f"${predictions['support_level']:.2f}")
                    
                    with col4:
                        st.metric("Expected Move", f"{predictions['expected_move_pct']:+.1f}%")
                        st.metric("Volatility Est.", f"{predictions['volatility_estimate']:.1f}%")
                    
                    # Confidence Analysis
                    st.subheader(f"📊 {asset_class} Prediction Confidence")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Confidence score with color coding
                        confidence_score = confidence['overall_score']
                        
                        if confidence_score >= 70:
                            st.success(f"🟢 **High Confidence**: {confidence_score:.1f}%")
                        elif confidence_score >= 50:
                            st.warning(f"🟡 **Medium Confidence**: {confidence_score:.1f}%")
                        else:
                            st.error(f"🔴 **Low Confidence**: {confidence_score:.1f}%")
                        
                        st.write(f"**Interpretation:** {confidence['interpretation']}")
                        
                        # Confidence breakdown
                        st.markdown("**📈 Confidence Breakdown:**")
                        st.write(f"• Technical Signals: {confidence['technical_confidence']:.1f}%")
                        st.write(f"• Momentum Analysis: {confidence['momentum_confidence']:.1f}%")
                        st.write(f"• Volatility Assessment: {confidence['volatility_confidence']:.1f}%")
                    
                    with col2:
                        # Technical signals summary
                        technical = prediction_result['technical_signals']
                        sentiment = prediction_result['sentiment_analysis']
                        
                        st.markdown("**🔧 Technical Signals:**")
                        
                        # Moving average signal
                        ma_color = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(technical['ma_signal'], "⚪")
                        st.write(f"{ma_color} **Moving Averages:** {technical['ma_signal']}")
                        
                        # RSI signal
                        rsi_color = {"OVERBOUGHT": "🔴", "OVERSOLD": "🟢", "NEUTRAL": "🟡"}.get(technical['rsi_signal'], "⚪")
                        st.write(f"{rsi_color} **RSI ({technical['rsi_value']}):** {technical['rsi_signal']}")
                        
                        # Momentum
                        # Momentum
                        momentum_color = {
                            "STRONG_UP": "🟢", "UP": "🟢", "STRONG_DOWN": "🔴", 
                            "DOWN": "🔴", "NEUTRAL": "🟡"
                        }.get(sentiment['momentum_signal'], "⚪")
                        st.write(f"{momentum_color} **Momentum:** {sentiment['momentum_signal']}")
                        
                        # Volatility regime
                        vol_color = {"HIGH": "🔴", "NORMAL": "🟡", "LOW": "🟢"}.get(sentiment['volatility_regime'], "⚪")
                        st.write(f"{vol_color} **Volatility:** {sentiment['volatility_regime']}")
                    
                    # Asset-Specific Analysis
                    st.subheader(f"🎯 {asset_class}-Specific Analysis")
                    
                    asset_analysis = prediction_result['asset_specific_analysis']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**📊 {asset_analysis['factor_type']}:**")
                        st.write(f"• **Current Theme:** {asset_analysis['current_theme']}")
                        st.write(f"• **Seasonality:** {asset_analysis['seasonality']}")
                        
                        st.markdown("**🔍 Key Factors:**")
                        for factor in asset_analysis['key_factors']:
                            st.write(f"• {factor}")
                    
                    with col2:
                        # Technical summary
                        st.markdown("**📈 Technical Summary:**")
                        st.write(f"• **Trend Strength:** {technical.get('trend_strength', 0):.1f}%")
                        st.write(f"• **BB Position:** {technical['bb_position']:.1f}%")
                        if asset_class != 'FOREX':
                            st.write(f"• **Volume Signal:** {technical['volume_signal']}")
                        
                        st.markdown("**📊 Market Regime:**")
                        st.write(f"• **Type:** {sentiment['market_regime']}")
                        st.write(f"• **Range:** {sentiment['range_percentage']:.1f}%")
                        st.write(f"• **5D Momentum:** {sentiment['momentum_5d']:+.2f}%")
                        st.write(f"• **20D Momentum:** {sentiment['momentum_20d']:+.2f}%")
                    
                    # Trading Recommendations
                    st.subheader(f"💡 {asset_class} Trading Recommendations")
                    
                    recommendations = prediction_result['trading_recommendations']
                    
                    # Primary recommendation
                    if recommendations['primary_recommendation']:
                        primary = recommendations['primary_recommendation']
                        
                        # Color code by risk level
                        risk_colors = {
                            'LOW': '🟢',
                            'MODERATE': '🟡', 
                            'HIGH': '🔴'
                        }
                        
                        risk_color = risk_colors.get(primary['risk_level'], '⚪')
                        
                        st.success(f"""
                        **🎯 Primary Recommendation: {primary['strategy'].replace('_', ' ').title()}**
                        
                        {risk_color} **Risk Level:** {primary['risk_level']}
                        
                        **💡 Rationale:** {primary['rationale']}
                        
                        **🎯 Target:** {primary['target']}
                        """)
                    
                    # Alternative strategies
                    if recommendations['alternative_strategies']:
                        st.markdown("**🔄 Alternative Strategies:**")
                        
                        for i, alt in enumerate(recommendations['alternative_strategies'][:3]):
                            risk_color = risk_colors.get(alt['risk_level'], '⚪')
                            
                            with st.expander(f"{alt['strategy'].replace('_', ' ').title()} ({alt['risk_level']} Risk)"):
                                st.write(f"**Rationale:** {alt['rationale']}")
                                st.write(f"**Target:** {alt['target']}")
                                st.write(f"**Risk Level:** {risk_color} {alt['risk_level']}")
                    
                    # Market outlook and risks
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔮 Market Outlook:**")
                        
                        outlook = recommendations['market_outlook']
                        outlook_color = {
                            'BULLISH': '🟢',
                            'BEARISH': '🔴',
                            'NEUTRAL': '🟡'
                        }.get(outlook, '⚪')
                        
                        st.write(f"{outlook_color} **Direction:** {outlook}")
                        st.write(f"**Time Horizon:** {recommendations['time_horizon']}")
                    
                    with col2:
                        st.markdown("**⚠️ Key Risks:**")
                        
                        for risk in recommendations['key_risks']:
                            st.write(f"• {risk}")
                    
                    # Prediction Summary
                    st.subheader(f"📋 {asset_class} Prediction Summary")
                    
                    # Create summary based on predictions
                    expected_direction = "Upward" if predictions['expected_move_pct'] > 0 else "Downward" if predictions['expected_move_pct'] < 0 else "Sideways"
                    move_magnitude = abs(predictions['expected_move_pct'])
                    
                    if move_magnitude > 5:
                        move_size = "significant"
                    elif move_magnitude > 2:
                        move_size = "moderate"
                    else:
                        move_size = "minor"
                    
                    st.info(f"""
                    **🎯 {asset_class} Prediction Summary for {prediction_result['ticker']}:**
                    
                    • **Expected Direction:** {expected_direction} movement over {prediction_days} days
                    • **Magnitude:** {move_size} move of {predictions['expected_move_pct']:+.1f}%
                    • **Confidence Level:** {confidence['interpretation']} ({confidence_score:.0f}%)
                    • **Recommended Strategy:** {recommendations['primary_recommendation']['strategy'].replace('_', ' ').title() if recommendations['primary_recommendation'] else 'None'}
                    • **Key Theme:** {asset_analysis['current_theme']}
                    
                    **Next Steps:** Monitor {asset_analysis['key_factors'][0].lower()} and consider {recommendations['primary_recommendation']['strategy'].lower().replace('_', ' ') if recommendations['primary_recommendation'] else 'alternative strategies'} based on market conditions.
                    """)
                
                else:
                    st.error(f"❌ Prediction failed: {prediction_result.get('error', 'Unknown error')}")
        
        else:
            # Show prediction instructions for current asset class
            prediction_instructions = {
                'INDICES': """
                ## 🔮 Index Market Predictions
                
                AI-powered market predictions for index ETFs and volatility products.
                
                ### 🎯 Index Prediction Features:
                - **Technical analysis** with multiple timeframe signals
                - **Market breadth assessment** and sector rotation insights
                - **Volatility regime analysis** for VIX and index behavior
                - **Economic cycle considerations** for long-term predictions
                
                ### 💡 Index-Specific Factors:
                - Broad market sentiment and economic indicators
                - Sector rotation and style factor impacts
                - Federal Reserve policy and interest rate environment
                - Global market correlations and risk sentiment
                """,
                
                'EQUITIES': """
                ## 🔮 Equity Market Predictions
                
                Comprehensive AI analysis for individual stock price predictions.
                
                ### 🎯 Equity Prediction Features:
                - **Company-specific analysis** with fundamental considerations
                - **Earnings cycle impact** and announcement timing
                - **Sector performance** and relative strength analysis
                - **Technical pattern recognition** and momentum signals
                
                ### 💡 Equity-Specific Factors:
                - Company fundamentals and earnings expectations
                - Sector trends and competitive positioning
                - News sentiment and analyst coverage
                - Technical levels and chart patterns
                """,
                
                'FOREX': """
                ## 🔮 FX Market Predictions
                
                Currency market predictions with global macro analysis and central bank insights.
                
                ### 🎯 FX Prediction Features:
                - **Central bank policy analysis** and interest rate differentials
                - **Global macro themes** and economic data impact
                - **Risk sentiment assessment** and safe haven flows
                - **Technical analysis** adapted for 24/5 FX markets
                
                ### 💡 FX-Specific Factors:
                - Interest rate differentials and policy divergence
                - Economic data releases and calendar events
                - Geopolitical developments and risk sentiment
                - Intervention risks and central bank communications
                """
            }
            
            st.markdown(prediction_instructions[asset_class])

if __name__ == "__main__":
    main()