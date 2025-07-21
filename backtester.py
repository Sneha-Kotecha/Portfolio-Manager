import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Suppress numpy “mean of empty slice” warnings globally
warnings.simplefilter("ignore", category=RuntimeWarning)

# ========== CONFIGURATION ==========
OANDA_API_KEY = "1400757678007e080b3b2a49a1c08e66-44740147c10d16adcc5b66b6b33f6e47"
MAX_OANDA_COUNT = 5000  # OANDA API maximum candles per request

# ========== DATA MODELS ==========
@dataclass
class Candle:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

class PinBarType(Enum):
    NONE = "none"
    BULLISH = "bullish"
    BEARISH = "bearish"

# ========== PIN BAR DETECTOR ==========
class PinBarDetector:
    def __init__(self,
                 min_wick_ratio: float = 0.6,
                 max_body_ratio: float = 0.3,
                 max_opposite_wick: float = 0.2):
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        self.max_opposite_wick = max_opposite_wick

    def detect_pin_bar(self, candle: Candle) -> Tuple[PinBarType, float]:
        rng = candle.high - candle.low
        if rng <= 0:
            return PinBarType.NONE, 0.0
        body = abs(candle.close - candle.open)
        upper = candle.high - max(candle.open, candle.close)
        lower = min(candle.open, candle.close) - candle.low
        upper_ratio = upper / rng
        lower_ratio = lower / rng
        body_ratio = body / rng

        # Bullish pin bar
        if (lower_ratio >= self.min_wick_ratio and
            body_ratio <= self.max_body_ratio and
            upper_ratio <= self.max_opposite_wick):
            strength = min(100,
                           (lower_ratio/0.75)*40 +
                           max(0, 30 - (body_ratio/0.15)*30) +
                           max(0, 30 - max(0,(upper_ratio-0.05)/0.05)*30))
            return PinBarType.BULLISH, round(strength,1)
        # Bearish pin bar
        if (upper_ratio >= self.min_wick_ratio and
            body_ratio <= self.max_body_ratio and
            lower_ratio <= self.max_opposite_wick):
            strength = min(100,
                           (upper_ratio/0.75)*40 +
                           max(0, 30 - (body_ratio/0.15)*30) +
                           max(0, 30 - max(0,(lower_ratio-0.05)/0.05)*30))
            return PinBarType.BEARISH, round(strength,1)
        return PinBarType.NONE, 0.0

# ========== OANDA DATA FETCHER ==========
class OANDADataFetcher:
    def __init__(self, api_key: str, account_type: str = "practice"):
        prefix = 'practice' if account_type=='practice' else 'trade'
        self.base_url = f"https://api-fx{prefix}.oanda.com/v3/instruments"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_candles(self,
                    instrument: str,
                    count: int,
                    granularity: str) -> pd.DataFrame:
        from requests.exceptions import HTTPError
        count = min(count, MAX_OANDA_COUNT)
        url = f"{self.base_url}/{instrument}/candles"
        params = {"count": count,
                  "granularity": granularity,
                  "price": "M"}
        try:
            r = requests.get(url, headers=self.headers, params=params)
            r.raise_for_status()
        except HTTPError as e:
            if r.status_code == 401:
                st.error("Unauthorized: please check your OANDA API key and account type.")
            else:
                st.error(f"Error fetching candles: {e}")
            return pd.DataFrame()
        data = r.json().get('candles', [])
        rec = []
        for c in data:
            if not c.get('complete', False):
                continue
            mid = c['mid']
            rec.append({
                'timestamp': pd.to_datetime(c['time']),
                'open': float(mid['o']),
                'high': float(mid['h']),
                'low': float(mid['l']),
                'close': float(mid['c']),
                'volume': c.get('volume', 0)
            })
        df = pd.DataFrame(rec)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df

# ========== BACKTESTER ==========
class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"

class TradeStatus(Enum):
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    NOT_TRIGGERED = "not_triggered"

@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    pb: Dict                     # store the originating pin-bar here
    status: TradeStatus = TradeStatus.OPEN
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0

    def set_exit(self, t, p, s: TradeStatus):
        self.exit_time, self.exit_price, self.status = t, p, s

class Backtester:
    def __init__(self, fetcher: OANDADataFetcher, detector: PinBarDetector):
        self.fetcher = fetcher
        self.detector = detector

    def get_pip_value(self, s: str) -> float:
        return 0.01 if 'JPY' in s else 0.0001

    def calc_pos_size(self, balance, risk_pct, stop_pips):
        risk_usd = balance * risk_pct
        pip_usd = 10 if 'JPY' not in 'USD' else 9.09
        size = risk_usd / (stop_pips * pip_usd)
        return max(0.01, round(size, 2))

    def fetch_multi(self, symbol, start, end):
        days = (end - start).days + 1
        counts = {'M15': days*96, 'M30': days*48, 'H1': days*24, 'H4': days*6}
        return {tf: self.fetcher.get_candles(symbol, counts[tf] + 50, tf) for tf in counts}

    def detect_pbs(self, df1h: pd.DataFrame) -> List[Dict]:
        out = []
        for ts, row in df1h.iterrows():
            if row[['ema6','ema18','ema50','sma200']].isnull().any():
                continue
            uptrend_strong = row['ema6'] > row['ema18'] > row['ema50'] > row['sma200']
            uptrend_weak   = row['ema6'] > row['ema18'] and row['ema6'] > row['sma200']
            downtrend_strong = row['ema6'] < row['ema18'] < row['ema50'] < row['sma200']
            downtrend_weak   = row['ema6'] < row['ema18'] and row['ema6'] < row['sma200']

            candle = Candle(ts, row['open'], row['high'], row['low'], row['close'], row['volume'])
            pin_type, strength = self.detector.detect_pin_bar(candle)
            if pin_type == PinBarType.NONE:
                continue

            if pin_type == PinBarType.BULLISH:
                if not (uptrend_strong or uptrend_weak): continue
                if abs(row['low'] - row['ema6'])/row['ema6'] > 0.03: continue
            else:
                if not (downtrend_strong or downtrend_weak): continue
                if abs(row['high'] - row['ema6'])/row['ema6'] > 0.03: continue

            out.append({
                'timestamp': ts, 'type': pin_type, 'strength': strength,
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close']
            })
        return out

    def simulate(self, symbol, pbs, d1, d15, d30, d4, rr, balance, risk_pct):
        trades, history = [], []
        equity = balance
        pip = self.get_pip_value(symbol)

        # Prepare multi-TF SMA
        for df in [d15, d30, d4]:
            if not df.empty:
                df['SMA200'] = df['close'].rolling(200, min_periods=200).mean()

        for b in pbs:
            ts, pr = b['timestamp'], b['close']
            valid = True
            for df in [d15, d30, d4]:
                sma_series = df.loc[df.index <= ts, 'SMA200'].dropna()
                if sma_series.empty:
                    valid = False
                    break
                last_sma = sma_series.iloc[-1]
                if not ((b['type'] == PinBarType.BULLISH and pr > last_sma) or
                        (b['type'] == PinBarType.BEARISH and pr < last_sma)):
                    valid = False
                    break
            if not valid:
                continue

            entry = pr + 2*pip if b['type'] == PinBarType.BULLISH else pr - 2*pip
            sl    = b['low'] - 2*pip if b['type'] == PinBarType.BULLISH else b['high'] + 2*pip
            dist_pips = abs(entry - sl) / pip
            tp    = entry + dist_pips*rr if b['type'] == PinBarType.BULLISH else entry - dist_pips*rr
            lot   = self.calc_pos_size(equity, risk_pct, dist_pips)

            trade = Trade(
                entry_time  = pd.Timestamp(ts),
                direction   = TradeDirection.LONG if b['type']==PinBarType.BULLISH else TradeDirection.SHORT,
                entry_price = entry,
                stop_loss   = sl,
                take_profit = tp,
                lot_size    = lot,
                pb          = b
            )

            fut = d1.loc[d1.index > ts]
            triggered = False
            for ti, row in fut.iterrows():
                if not triggered and (
                   (trade.direction == TradeDirection.LONG and row['high'] >= entry) or
                   (trade.direction == TradeDirection.SHORT and row['low']  <= entry)
                ):
                    triggered = True
                    trade.entry_time = ti

                if triggered:
                    if trade.direction == TradeDirection.LONG:
                        if row['low'] <= sl:
                            trade.set_exit(ti, sl, TradeStatus.CLOSED_LOSS)
                            break
                        if row['high'] >= tp:
                            trade.set_exit(ti, tp, TradeStatus.CLOSED_PROFIT)
                            break
                    else:
                        if row['high'] >= sl:
                            trade.set_exit(ti, sl, TradeStatus.CLOSED_LOSS)
                            break
                        if row['low'] <= tp:
                            trade.set_exit(ti, tp, TradeStatus.CLOSED_PROFIT)
                            break

            if trade.exit_price:
                delta = ((trade.exit_price - trade.entry_price)
                         if trade.direction == TradeDirection.LONG
                         else (trade.entry_price - trade.exit_price))
                trade.pnl_pips = delta / pip
                trade.pnl_usd  = trade.pnl_pips * lot * (10 if pip < 0.01 else 1)
                equity += trade.pnl_usd
                history.append({'timestamp': trade.exit_time, 'equity': equity})
                trades.append(trade)

        return trades, history

    def calc_stats(self, trades: List[Trade], balance: float) -> Dict:
        if not trades:
            return {}
        wins   = [t for t in trades if t.status == TradeStatus.CLOSED_PROFIT]
        losses = [t for t in trades if t.status == TradeStatus.CLOSED_LOSS]
        total  = len(trades)
        win_rate = len(wins) / total * 100

        profit_factor = (
            sum(t.pnl_pips for t in wins)
            / abs(sum(t.pnl_pips for t in losses))
            if losses else float('inf')
        )
        net = sum(t.pnl_usd for t in trades)

        avg_win  = np.mean([t.pnl_usd for t in wins])   if wins   else 0.0
        avg_loss = np.mean([t.pnl_usd for t in losses]) if losses else 0.0
        expectancy = (win_rate / 100) * avg_win + ((100 - win_rate) / 100) * avg_loss

        return {
            'total_trades':  total,
            'win_rate':      win_rate,
            'profit_factor': profit_factor,
            'net_usd':       net,
            'expectancy':    expectancy
        }

# ========== STREAMLIT UI ==========
def main():
    st.set_page_config(page_title="Trend Surfer", layout="wide")
    account = st.sidebar.selectbox("Account", ["practice", "live"])
    symbol  = st.sidebar.selectbox("Pair", [
        "EUR_USD", "GBP_USD", "USD_JPY",
        "AUD_USD", "USD_CAD", "NZD_USD"
    ])
    start, end = st.sidebar.date_input(
        "Period",
        [datetime.now().date() - timedelta(days=60),
         datetime.now().date() - timedelta(days=1)]
    )
    rr   = st.sidebar.selectbox("R:R", [1.5, 2.0, 2.5, 3.0], index=1)
    bal  = st.sidebar.number_input("Capital", 1000.0)
    risk = st.sidebar.slider("Risk%", 0.1, 5.0, 1.0) / 100

    fetcher = OANDADataFetcher(OANDA_API_KEY, account)
    detector = PinBarDetector()
    tester  = Backtester(fetcher, detector)

    data = tester.fetch_multi(
        symbol,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end,   datetime.min.time())
    )
    df1 = data['H1']
    df1 = df1[df1.index.dayofweek < 5]
    data['H1'] = df1

    # MAs on H1
    df1['ema6']   = df1['close'].ewm(span=6,  adjust=False).mean()
    df1['ema18']  = df1['close'].ewm(span=18, adjust=False).mean()
    df1['ema50']  = df1['close'].ewm(span=50, adjust=False).mean()
    df1['sma200'] = df1['close'].rolling(200, min_periods=200).mean()

    pbs = tester.detect_pbs(df1)
    trades, hist = tester.simulate(
        symbol, pbs, df1,
        data['M15'], data['M30'], data['H4'],
        rr, bal, risk
    )

    tab1, tab2 = st.tabs(["Analyzer", "Backtester"])
    with tab1:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05
        )
        fig.add_trace(
            go.Candlestick(
                x=df1.index,
                open=df1['open'],
                high=df1['high'],
                low=df1['low'],
                close=df1['close'],
                name='Price'
            ), row=1, col=1
        )
        for col, name in [
            ('ema6','EMA6'), ('ema18','EMA18'),
            ('ema50','EMA50'), ('sma200','SMA200')
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df1.index,
                    y=df1[col],
                    mode='lines',
                    name=name,
                    line=dict(width=1)
                ), row=1, col=1
            )
        for pb in pbs:
            fig.add_trace(
                go.Scatter(
                    x=[pb['timestamp']],
                    y=[pb['close']],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='lime' if pb['type']==PinBarType.BULLISH else 'red',
                        symbol='triangle-up' if pb['type']==PinBarType.BULLISH else 'triangle-down'
                    ),
                    showlegend=False
                ), row=1, col=1
            )
        fig.add_trace(
            go.Bar(
                x=df1.index,
                y=df1['volume'],
                marker_color=[
                    'green' if o<=c else 'red'
                    for o, c in zip(df1['open'], df1['close'])
                ],
                name='Volume'
            ), row=2, col=1
        )
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            xaxis=dict(rangebreaks=[dict(bounds=["sat","mon"])], type='date')
        )
        st.plotly_chart(fig, use_container_width=True)

        if trades:
            st.subheader("View Trades on Chart")
            idx = st.number_input("Trade #", 1, len(trades), 1)
            tr  = trades[idx-1]
            fig2 = fig
            fig2.add_trace(
                go.Scatter(
                    x=[tr.entry_time],
                    y=[tr.entry_price],
                    mode='markers',
                    marker=dict(symbol='star', size=12, color='gold'),
                    name='Entry'
                ), row=1, col=1
            )
            if tr.exit_time:
                fig2.add_trace(
                    go.Scatter(
                        x=[tr.exit_time],
                        y=[tr.exit_price],
                        mode='markers',
                        marker=dict(symbol='circle', size=12, color='blue'),
                        name='Exit'
                    ), row=1, col=1
                )
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader(f"Trade {idx} Explanation")
            st.write(f"A {tr.direction.value} trade taken because:")
            st.write(f"- Pin bar type: {tr.pb['type'].value.title()} with strength {tr.pb['strength']:.1f}%")
            st.write("- H1 trend: EMA6 > EMA18 > EMA50 > SMA200 and price above SMA200")
            st.write("- Pin bar wick touched EMA6")
        else:
            st.write("No trades to display.")

    with tab2:
        st.subheader("Backtest Results")
        s = tester.calc_stats(trades, bal)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", s.get('total_trades', 0))
        c2.metric("Win Rate", f"{s.get('win_rate', 0):.1f}%")
        c3.metric("Profit Factor", f"{s.get('profit_factor', 0):.2f}")
        c4.metric("Net P&L", f"${s.get('net_usd', 0):.2f}")
        st.metric("Expectancy", f"${s.get('expectancy', 0):.2f}")

        if hist:
            st.line_chart(pd.DataFrame(hist).set_index('timestamp')['equity'])

        st.subheader("Trades")
        if trades:
            st.dataframe(pd.DataFrame([
                {
                    'entry':  t.entry_time,
                    'exit':   t.exit_time,
                    'pnl':    t.pnl_usd,
                    'status': t.status.value
                } for t in trades
            ]))
        else:
            st.write("No trades executed.")

if __name__ == "__main__":
    main()
