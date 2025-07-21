import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
from enum import Enum
from typing import Optional
from dataclasses import dataclass

# Replace with your actual key
OANDA_API_KEY = "1400757678007e080b3b2a49a1c08e66-44740147c10d16adcc5b66b6b33f6e47"

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
    Pin bar detection with combined bullish/bearish explanations for every candle.
    Trend and EMA-touch filters have been relaxed to capture bars like #292 and #494.
    """
    def __init__(self, min_wick_ratio: float = 0.6, max_body_ratio: float = 0.3, max_opposite_wick: float = 0.2):
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        self.max_opposite_wick = max_opposite_wick

    def update_parameters(self, min_wick_ratio: float, max_body_ratio: float, max_opposite_wick: float):
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        self.max_opposite_wick = max_opposite_wick

    def explain_pin_bar(self,
                        candle: Candle,
                        ema6: Optional[float],
                        ema18: Optional[float],
                        ema50: Optional[float],
                        sma200: Optional[float]) -> dict:
        """
        Returns a dict with:
          - pin_bar_type: PinBarType
          - strength: float
          - explanation: detailed reasons for bullish & bearish checks
        EMA-touch and strict trend requirements have been removed for broader detection.
        """
        rng = candle.high - candle.low
        if rng == 0:
            return {
                'pin_bar_type': PinBarType.NONE,
                'strength': 0.0,
                'explanation': 'Zero range (high == low), cannot form a pin bar.'
            }

        body = abs(candle.close - candle.open)
        upper = candle.high - max(candle.open, candle.close)
        lower = min(candle.open, candle.close) - candle.low
        upper_ratio = upper / rng
        lower_ratio = lower / rng
        body_ratio = body / rng

        # Build bullish reasoning
        bullish_reasons = [
            f"range={rng:.5f}",
            f"lower_ratio={lower_ratio:.2f}",
            f"body_ratio={body_ratio:.2f}",
            f"upper_ratio={upper_ratio:.2f}"
        ]
        bullish_reasons.append(
            f"lower_ratio {'>=' if lower_ratio >= self.min_wick_ratio else '<'} min_wick_ratio {self.min_wick_ratio}"
        )
        bullish_reasons.append(
            f"body_ratio {'<=' if body_ratio <= self.max_body_ratio else '>'} max_body_ratio {self.max_body_ratio}"
        )
        bullish_reasons.append(
            f"upper_ratio {'<=' if upper_ratio <= self.max_opposite_wick else '>'} max_opposite_wick {self.max_opposite_wick}"
        )

        # Bullish condition (trend & EMA-touch removed)
        bullish_condition = (
            lower_ratio >= self.min_wick_ratio and
            body_ratio <= self.max_body_ratio and
            upper_ratio <= self.max_opposite_wick and
            (candle.close > candle.open or body_ratio <= 0.05)
        )

        if bullish_condition:
            strength = min(
                100,
                (lower_ratio / 0.75) * 40 +
                max(0, 30 - (body_ratio / 0.15) * 30) +
                max(0, 30 - max(0, (upper_ratio - 0.05) / 0.05) * 30)
            )
            return {
                'pin_bar_type': PinBarType.BULLISH,
                'strength': round(strength, 1),
                'explanation': ' | '.join(bullish_reasons)
            }

        # Build bearish reasoning
        bearish_reasons = [
            f"range={rng:.5f}",
            f"upper_ratio={upper_ratio:.2f}",
            f"body_ratio={body_ratio:.2f}",
            f"lower_ratio={lower_ratio:.2f}"
        ]
        bearish_reasons.append(
            f"upper_ratio {'>=' if upper_ratio >= self.min_wick_ratio else '<'} min_wick_ratio {self.min_wick_ratio}"
        )
        bearish_reasons.append(
            f"body_ratio {'<=' if body_ratio <= self.max_body_ratio else '>'} max_body_ratio {self.max_body_ratio}"
        )
        bearish_reasons.append(
            f"lower_ratio {'<=' if lower_ratio <= self.max_opposite_wick else '>'} max_opposite_wick {self.max_opposite_wick}"
        )

        # Bearish condition (trend & EMA-touch removed)
        bearish_condition = (
            upper_ratio >= self.min_wick_ratio and
            body_ratio <= self.max_body_ratio and
            lower_ratio <= self.max_opposite_wick and
            (candle.close < candle.open)
        )

        if bearish_condition:
            strength = min(
                100,
                (upper_ratio / 0.75) * 40 +
                max(0, 30 - (body_ratio / 0.15) * 30) +
                max(0, 30 - max(0, (lower_ratio - 0.05) / 0.05) * 30)
            )
            return {
                'pin_bar_type': PinBarType.BEARISH,
                'strength': round(strength, 1),
                'explanation': ' | '.join(bearish_reasons)
            }

        # Neither
        full_explanation = (
            f"Bullish: {'; '.join(bullish_reasons)} | "
            f"Bearish: {'; '.join(bearish_reasons)}"
        )
        return {
            'pin_bar_type': PinBarType.NONE,
            'strength': 0.0,
            'explanation': full_explanation
        }


class OANDADataFetcher:
    def __init__(self, api_key: str, account_type: str = "practice"):
        self.api_key = api_key
        self.base_url = (
            f"https://api-fx{'practice' if account_type=='practice' else 'trade'}.oanda.com"
        )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def get_instruments(self) -> list:
        url = f"{self.base_url}/v3/accounts"
        try:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            accounts = r.json().get("accounts", [])
            if not accounts:
                return []
            aid = accounts[0]["id"]
            inst_url = f"{self.base_url}/v3/accounts/{aid}/instruments"
            r2 = requests.get(inst_url, headers=self.headers)
            r2.raise_for_status()
            return [i["name"] for i in r2.json().get("instruments", []) if i.get("type") == "CURRENCY"]
        except Exception:
            return []

    def get_candles(
        self, instrument: str, count: int = 500, granularity: str = "D"
    ) -> pd.DataFrame:
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {"count": count, "granularity": granularity, "price": "M"}
        try:
            r = requests.get(url, headers=self.headers, params=params)
            r.raise_for_status()
            data = []
            for c in r.json().get("candles", []):
                if c.get("complete"):
                    mid = c.get("mid", {})
                    data.append({
                        "timestamp": pd.to_datetime(c["time"]),
                        "open": float(mid.get("o", 0)),
                        "high": float(mid.get("h", 0)),
                        "low": float(mid.get("l", 0)),
                        "close": float(mid.get("c", 0)),
                        "volume": float(c.get("volume", 0))
                    })
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

# Indicators
def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period).mean()


def main():
    st.set_page_config(page_title="Pin Bar Detector", layout="wide")
    st.title("🎯 Pin Bar Detector with Explanations")
    st.markdown("Detect pin bars and see why each candle passed or failed both bullish & bearish criteria.")

    # Sidebar
    st.sidebar.header("Configuration")
    account_type = st.sidebar.selectbox("Account Type", ["practice", "live"])
    if OANDA_API_KEY == "YOUR_API_KEY_HERE":
        st.error("Please set your OANDA API key in the code.")
        return
    fetcher = OANDADataFetcher(OANDA_API_KEY, account_type)

    instruments = fetcher.get_instruments()
    if not instruments:
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY"]
        st.sidebar.warning("Could not fetch instruments. Using defaults.")
    pair = st.sidebar.selectbox("Currency Pair", instruments)
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        [("Daily", "D"), ("4 Hour", "H4"), ("1 Hour", "H1"),
         ("30 Min", "M30"), ("15 Min", "M15"), ("5 Min", "M5")],
        format_func=lambda x: x[0]
    )
    count = st.sidebar.slider("Number of Candles", 100, 1000, 500)
    min_wick = st.sidebar.slider("Min Wick Ratio", 0.3, 0.8, 0.6, 0.05)
    max_body = st.sidebar.slider("Max Body Ratio", 0.1, 0.5, 0.3, 0.05)
    max_opp = st.sidebar.slider("Max Opposite Wick", 0.1, 0.4, 0.2, 0.05)

    if st.sidebar.button("Fetch Data & Analyze"):
        df = fetcher.get_candles(pair, count, timeframe[1])
        if df.empty:
            st.error("No data received. Check API key or connection.")
            return
        df['ema6'] = calculate_ema(df['close'], 6)
        df['ema18'] = calculate_ema(df['close'], 18)
        df['ema50'] = calculate_ema(df['close'], 50)
        df['sma200'] = calculate_sma(df['close'], 200)
        df['dow'] = df['timestamp'].dt.dayofweek
        df = df[df['dow'] < 5].reset_index(drop=True)
        df['x'] = df.index

        detector = PinBarDetector(min_wick, max_body, max_opp)
        entries = []
        for _, row in df.iterrows():
            candle = Candle(
                row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']
            )
            info = detector.explain_pin_bar(
                candle, row.get('ema6'), row.get('ema18'), row.get('ema50'), row.get('sma200')
            )
            entries.append({
                'timestamp': row['timestamp'],
                'type': info['pin_bar_type'].value,
                'strength': info['strength'],
                'price': row['close'],
                'explanation': info['explanation']
            })

        # Metrics
        total = len(entries)
        pins = [e for e in entries if e['type'] != 'none']
        pin_count = len(pins)
        avg_strength = np.mean([p['strength'] for p in pins]) if pins else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Candles", total)
        c2.metric("Pin Bars", pin_count)
        c3.metric("Avg Strength", f"{avg_strength:.1f}" if pins else "N/A")

        # Chart
        fig = make_subplots(rows=2, cols=1, subplot_titles=[f"{pair} ({timeframe[0]})", "Volume"],
                            vertical_spacing=0.1, row_heights=[0.7, 0.3])
        fig.add_trace(
            go.Candlestick(
                x=df['x'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                name='Price', increasing_line_color='green', decreasing_line_color='red'
            ), row=1, col=1
        )
        for ema, name in [('ema6', 'EMA6'), ('ema18', 'EMA18'), ('ema50', 'EMA50')]:
            fig.add_trace(
                go.Scatter(x=df['x'], y=df[ema], name=name, line=dict(width=1)), row=1, col=1
            )
        fig.add_trace(
            go.Scatter(x=df['x'], y=df['sma200'], name='SMA200', line=dict(width=2)), row=1, col=1
        )
        for e in entries:
            if e['type'] != 'none':
                idx = int(df[df['timestamp'] == e['timestamp']]['x'])
                color = 'lime' if e['type'] == 'bullish' else 'red'
                symbol = 'triangle-up' if e['type'] == 'bullish' else 'triangle-down'
                fig.add_trace(
                    go.Scatter(x=[idx], y=[e['price']], mode='markers',
                               marker=dict(size=12, color=color, symbol=symbol),
                               hovertext=e['explanation'], showlegend=False),
                    row=1, col=1
                )
        vol_colors = ['green' if r['close'] >= r['open'] else 'red' for _, r in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df['x'], y=df['volume'], marker_color=vol_colors, name='Volume'), row=2, col=1
        )
        step = max(1, total // 10)
        ticks = list(range(0, total, step))
        labels = [df.iloc[i]['timestamp'].strftime('%Y-%m-%d') for i in ticks]
        fig.update_xaxes(tickmode='array', tickvals=ticks, ticktext=labels, row='all')
        fig.update_layout(height=800, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Explanation table
        st.subheader("Candle Explanations")
        pd.options.display.max_colwidth = 200
        expl_df = pd.DataFrame(entries)
        expl_df['timestamp'] = expl_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(expl_df[['timestamp', 'type', 'strength', 'explanation']], use_container_width=True)

        # Download CSV
        if st.button("Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data",
                data=csv,
                file_name=f"{pair}_{timeframe[1]}_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
