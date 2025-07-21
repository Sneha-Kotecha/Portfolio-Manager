import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
from enum import Enum
from typing import Tuple, NamedTuple, Optional
from dataclasses import dataclass

OANDA_API_KEY = "1400757678007e080b3b2a49a1c08e66-44740147c10d16adcc5b66b6b33f6e47"


# Your PinBarDetector classes (copied from your file)
class PinBarType(Enum):
    NONE = "none"
    BULLISH = "bullish"
    BEARISH = "bearish"

@dataclass
class Candle:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

class PinBarDetector:
    """
    Pin bar detection system using the provided algorithm with configurable parameters.
    """
    
    def __init__(self, min_wick_ratio: float = 0.6, max_body_ratio: float = 0.3, max_opposite_wick: float = 0.2):
        """
        Initialize the pin bar detector with configurable parameters.
        
        Args:
            min_wick_ratio: Minimum ratio of main wick to total range
            max_body_ratio: Maximum ratio of body to total range
            max_opposite_wick: Maximum ratio of opposite wick to total range
        """
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        self.max_opposite_wick = max_opposite_wick
    
    def update_parameters(self, min_wick_ratio: float, max_body_ratio: float, max_opposite_wick: float):
        """Update detection parameters."""
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        self.max_opposite_wick = max_opposite_wick
    
    def detect_pin_bar(self, candle: Candle, ema6: Optional[float] = None, ema18: Optional[float] = None, 
                       ema50: Optional[float] = None, sma200: Optional[float] = None) -> Tuple[PinBarType, float]:
        """
        Detect pin bar pattern using the original algorithm with configurable parameters.
        Includes trend filtering and candle color validation.
        
        Args:
            candle: Candle data to analyze
            ema6: 6-period EMA value for trend analysis
            ema18: 18-period EMA value for trend analysis
            ema50: 50-period EMA value for trend analysis
            sma200: 200-period SMA value for trend analysis
            
        Returns:
            Tuple of (PinBarType, strength_score)
        """
        # Calculate basic measurements
        rng = candle.high - candle.low
        body = abs(candle.close - candle.open)
        
        # Handle zero range case
        if rng == 0:
            return PinBarType.NONE, 0.0
        
        # Calculate wick sizes
        upper = candle.high - max(candle.open, candle.close)
        lower = min(candle.open, candle.close) - candle.low
        
        # Convert to ratios
        upper_ratio = upper / rng
        lower_ratio = lower / rng
        body_ratio = body / rng
        
        # Check candle color
        is_green_candle = candle.close > candle.open
        is_red_candle = candle.close < candle.open
        
        # Determine trend direction if indicators are provided
        uptrend = False
        downtrend = False
        
        if all(x is not None for x in [ema6, ema18, ema50, sma200]):
            # Uptrend: 6EMA > 18EMA > 50EMA > 200SMA
            uptrend = ema6 > ema18 > ema50 > sma200
            # Downtrend: 6EMA < 18EMA < 50EMA < 200SMA
            downtrend = ema6 < ema18 < ema50 < sma200
        else:
            # If EMAs are not provided, do not detect any pin bars
            # This ensures trend filtering is always applied
            return PinBarType.NONE, 0.0
        
        # Check for bullish pin bar (hammer pattern)
        if (lower_ratio >= self.min_wick_ratio and 
            body_ratio <= self.max_body_ratio and 
            upper_ratio <= self.max_opposite_wick):
            
            # Check if lower wick touches the 6EMA (within small tolerance)
            wick_touches_ema6 = False
            if ema6 is not None:
                # Lower wick range: from low to body bottom
                body_bottom = min(candle.open, candle.close)
                # Check if 6EMA is within the lower wick range (with small tolerance)
                tolerance = rng * 0.02  # 2% of candle range as tolerance
                wick_touches_ema6 = (candle.low - tolerance <= ema6 <= body_bottom + tolerance)
            
            # Only show bullish pin bars in uptrends with green/doji candles AND lower wick touches 6EMA
            # Require all trend indicators to be present for valid detection
            if (uptrend and (is_green_candle or body_ratio <= 0.05) and wick_touches_ema6 and
                all(x is not None for x in [ema6, ema18, ema50, sma200])):
                strength = min(100, 
                    (lower_ratio / 0.75) * 40 + 
                    max(0, 30 - (body_ratio / 0.15) * 30) + 
                    max(0, 30 - max(0, (upper_ratio - 0.05) / 0.05) * 30)
                )
                return PinBarType.BULLISH, round(strength, 1)
        
        # Check for bearish pin bar (shooting star pattern)
        if (upper_ratio >= self.min_wick_ratio and 
            body_ratio <= self.max_body_ratio and 
            lower_ratio <= self.max_opposite_wick):
            
            # Check if upper wick touches the 6EMA (within small tolerance)
            wick_touches_ema6 = False
            if ema6 is not None:
                # Upper wick range: from body top to high
                body_top = max(candle.open, candle.close)
                # Check if 6EMA is within the upper wick range (with small tolerance)
                tolerance = rng * 0.02  # 2% of candle range as tolerance
                wick_touches_ema6 = (body_top - tolerance <= ema6 <= candle.high + tolerance)
            
            # Only show bearish pin bars in downtrends with red candles AND upper wick touches 6EMA
            # Require all trend indicators to be present for valid detection
            if (downtrend and is_red_candle and wick_touches_ema6 and
                all(x is not None for x in [ema6, ema18, ema50, sma200])):
                strength = min(100, 
                    (upper_ratio / 0.75) * 40 + 
                    max(0, 30 - (body_ratio / 0.15) * 30) + 
                    max(0, 30 - max(0, (lower_ratio - 0.05) / 0.05) * 30)
                )
                return PinBarType.BEARISH, round(strength, 1)
        
        return PinBarType.NONE, 0.0
    
    def get_pin_bar_info(self, candle: Candle) -> dict:
        """
        Get detailed information about a candle's pin bar characteristics.
        
        Args:
            candle: Candle data to analyze
            
        Returns:
            Dictionary with detailed analysis
        """
        rng = candle.high - candle.low
        
        if rng == 0:
            return {
                'range': 0,
                'body': 0,
                'upper_wick': 0,
                'lower_wick': 0,
                'body_ratio': 0,
                'upper_ratio': 0,
                'lower_ratio': 0,
                'pin_bar_type': PinBarType.NONE,
                'strength': 0.0
            }
        
        body = abs(candle.close - candle.open)
        upper = candle.high - max(candle.open, candle.close)
        lower = min(candle.open, candle.close) - candle.low
        
        pin_bar_type, strength = self.detect_pin_bar(candle)
        
        return {
            'range': rng,
            'body': body,
            'upper_wick': upper,
            'lower_wick': lower,
            'body_ratio': body / rng,
            'upper_ratio': upper / rng,
            'lower_ratio': lower / rng,
            'pin_bar_type': pin_bar_type,
            'strength': strength
        }

# OANDA Data Fetcher
class OANDADataFetcher:
    def __init__(self, api_key: str, account_type: str = "practice"):
        self.api_key = api_key
        self.base_url = f"https://api-fx{'practice' if account_type == 'practice' else 'trade'}.oanda.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_instruments(self):
        """Get available instruments from OANDA"""
        url = f"{self.base_url}/v3/accounts"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                accounts = response.json()["accounts"]
                if accounts:
                    account_id = accounts[0]["id"]
                    url = f"{self.base_url}/v3/accounts/{account_id}/instruments"
                    response = requests.get(url, headers=self.headers)
                    if response.status_code == 200:
                        instruments = response.json()["instruments"]
                        return [inst["name"] for inst in instruments if inst["type"] == "CURRENCY"]
            return []
        except:
            return []
    
    def get_candles(self, instrument: str, count: int = 500, granularity: str = "D"):
        """Fetch candlestick data from OANDA"""
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M"  # Mid prices
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                candles_data = response.json()["candles"]
                
                data = []
                for candle in candles_data:
                    if candle["complete"]:
                        mid = candle["mid"]
                        data.append({
                            "timestamp": pd.to_datetime(candle["time"]),
                            "open": float(mid["o"]),
                            "high": float(mid["h"]),
                            "low": float(mid["l"]),
                            "close": float(mid["c"]),
                            "volume": float(candle["volume"])
                        })
                
                return pd.DataFrame(data)
            else:
                st.error(f"Error fetching data: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return pd.DataFrame()

# Technical Indicators
def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

# Main Streamlit App
def main():
    st.set_page_config(page_title="Pin Bar Detector", layout="wide")
    
    st.title("🎯 Pin Bar Detector with OANDA Data")
    st.markdown("Detect pin bar patterns in forex data using your custom algorithm")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # OANDA API Configuration
    st.sidebar.subheader("OANDA API Settings")
    account_type = st.sidebar.selectbox("Account Type", ["practice", "live"])
    
    if OANDA_API_KEY == "YOUR_API_KEY_HERE":
        st.error("Please replace 'YOUR_API_KEY_HERE' with your actual OANDA API key in the code.")
        st.info("You can get a free practice account at https://www.oanda.com/")
        return
    
    # Initialize data fetcher
    fetcher = OANDADataFetcher(OANDA_API_KEY, account_type)
    
    # Currency pair selection
    st.sidebar.subheader("Data Selection")
    
    # Try to get instruments from OANDA
    instruments = fetcher.get_instruments()
    if not instruments:
        # Fallback to common forex pairs
        instruments = [
            "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", 
            "USD_CAD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY"
        ]
        st.sidebar.warning("Could not fetch instruments from OANDA. Using default list.")
    
    selected_pair = st.sidebar.selectbox("Currency Pair", instruments)
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox("Timeframe", [
        ("Daily", "D"), ("4 Hour", "H4"), ("1 Hour", "H1"), 
        ("30 Min", "M30"), ("15 Min", "M15"), ("5 Min", "M5")
    ], format_func=lambda x: x[0])
    
    candle_count = st.sidebar.slider("Number of Candles", 100, 1000, 500)
    
    # Pin Bar Detector Parameters
    st.sidebar.subheader("Pin Bar Parameters")
    min_wick_ratio = st.sidebar.slider("Min Wick Ratio", 0.3, 0.8, 0.6, 0.05)
    max_body_ratio = st.sidebar.slider("Max Body Ratio", 0.1, 0.5, 0.3, 0.05)
    max_opposite_wick = st.sidebar.slider("Max Opposite Wick", 0.1, 0.4, 0.2, 0.05)
    
    # Fetch and process data
    if st.sidebar.button("Fetch Data & Detect Pin Bars"):
        with st.spinner("Fetching data from OANDA..."):
            df = fetcher.get_candles(selected_pair, candle_count, timeframe[1])
        
        if df.empty:
            st.error("No data received. Please check your API key and connection.")
            return
        
        # Calculate technical indicators
        with st.spinner("Calculating indicators..."):
            df['ema6'] = calculate_ema(df['close'], 6)
            df['ema18'] = calculate_ema(df['close'], 18)
            df['ema50'] = calculate_ema(df['close'], 50)
            df['sma200'] = calculate_sma(df['close'], 200)
        
        # Filter to weekdays only (Monday=0 to Friday=4)
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df_weekdays = df[df['dayofweek'] < 5].copy()  # Keep only Monday to Friday
        df_weekdays = df_weekdays.drop('dayofweek', axis=1)  # Remove helper column
        
        if df_weekdays.empty:
            st.error("No weekday data available for the selected period.")
            return
        
        # Reset index and create sequential numbering for continuous x-axis
        df_weekdays = df_weekdays.reset_index(drop=True)
        df_weekdays['x_axis'] = range(len(df_weekdays))
        
        # Initialize pin bar detector
        detector = PinBarDetector(min_wick_ratio, max_body_ratio, max_opposite_wick)
        
        # Detect pin bars
        with st.spinner("Detecting pin bars..."):
            pin_bars = []
            for idx, row in df_weekdays.iterrows():
                candle = Candle(
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                
                pin_type, strength = detector.detect_pin_bar(
                    candle, row['ema6'], row['ema18'], row['ema50'], row['sma200']
                )
                
                if pin_type != PinBarType.NONE:
                    pin_bars.append({
                        'timestamp': row['timestamp'],
                        'type': pin_type.value,
                        'strength': strength,
                        'price': row['close']
                    })
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Weekday Candles", len(df_weekdays))
        with col2:
            st.metric("Pin Bars Found", len(pin_bars))
        with col3:
            if len(pin_bars) > 0:
                avg_strength = np.mean([pb['strength'] for pb in pin_bars])
                st.metric("Avg Strength", f"{avg_strength:.1f}")
            else:
                st.metric("Avg Strength", "N/A")
        
        # Create the chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f"{selected_pair} - {timeframe[0]} (Weekdays Only)", "Volume"],
            vertical_spacing=0.1,
            row_heights=[0.8, 0.2]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_weekdays['x_axis'],
                open=df_weekdays['open'],
                high=df_weekdays['high'],
                low=df_weekdays['low'],
                close=df_weekdays['close'],
                name="Price",
                increasing_line_color='green',
                decreasing_line_color='red',
                text=df_weekdays['timestamp'].dt.strftime('%Y-%m-%d'),
                hovertext='<b>%{text}</b><br>Open: %{open}<br>High: %{high}<br>Low: %{low}<br>Close: %{close}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        fig.add_trace(go.Scatter(x=df_weekdays['x_axis'], y=df_weekdays['ema6'], name='EMA 6', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_weekdays['x_axis'], y=df_weekdays['ema18'], name='EMA 18', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_weekdays['x_axis'], y=df_weekdays['ema50'], name='EMA 50', line=dict(color='purple', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_weekdays['x_axis'], y=df_weekdays['sma200'], name='SMA 200', line=dict(color='red', width=2)), row=1, col=1)
        
        # Add pin bar markers
        for pin_bar in pin_bars:
            # Find the x_axis position for this timestamp
            matching_row = df_weekdays[df_weekdays['timestamp'] == pin_bar['timestamp']]
            if not matching_row.empty:
                x_pos = matching_row['x_axis'].iloc[0]
                color = 'lime' if pin_bar['type'] == 'bullish' else 'red'
                symbol = 'triangle-up' if pin_bar['type'] == 'bullish' else 'triangle-down'
                
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos],
                        y=[pin_bar['price']],
                        mode='markers',
                        marker=dict(size=15, color=color, symbol=symbol),
                        name=f"{pin_bar['type'].title()} Pin Bar",
                        text=f"Strength: {pin_bar['strength']}",
                        hovertemplate=f"<b>{pin_bar['type'].title()} Pin Bar</b><br>Date: {pin_bar['timestamp'].strftime('%Y-%m-%d')}<br>Strength: {pin_bar['strength']}<br>Price: {pin_bar['price']}<extra></extra>",
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Add volume
        colors = ['green' if close >= open else 'red' for close, open in zip(df_weekdays['close'], df_weekdays['open'])]
        fig.add_trace(
            go.Bar(x=df_weekdays['x_axis'], y=df_weekdays['volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_pair} Pin Bar Analysis",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        # Configure x-axis to show date labels without gaps
        # Show every 10th date label to avoid overcrowding
        tick_spacing = max(1, len(df_weekdays) // 10)
        tick_indices = list(range(0, len(df_weekdays), tick_spacing))
        tick_labels = [df_weekdays.iloc[i]['timestamp'].strftime('%Y-%m-%d') for i in tick_indices]
        
        fig.update_xaxes(
            tickmode='array',
            tickvals=tick_indices,
            ticktext=tick_labels,
            title_text="Date",
            row=1, col=1
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=tick_indices,
            ticktext=tick_labels,
            title_text="Date",
            row=2, col=1
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pin Bar Details Table
        if pin_bars:
            st.subheader("Pin Bar Details")
            pin_df = pd.DataFrame(pin_bars)
            pin_df['timestamp'] = pd.to_datetime(pin_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(pin_df, use_container_width=True)
        
        # Download data option
        if st.button("Download Weekday Data as CSV"):
            csv = df_weekdays.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_pair}_{timeframe[1]}_weekdays_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()