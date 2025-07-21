"""
RSI-SMA Mean Reversion Trading Strategy - OANDA Integration
==========================================================

A comprehensive algorithmic trading strategy that combines RSI and SMA indicators
for mean reversion trading. Includes backtesting framework and live trading capabilities
with OANDA's forex and CFD platform.

Strategy Logic:
- Buy Signal: RSI < 30 AND Price > SMA(200) (oversold in uptrend)
- Sell Signal: RSI > 70 OR Price < SMA(200) (overbought or trend reversal)
- Risk Management: Stop-loss and take-profit levels in pips

Supported Instruments:
- Forex pairs (EUR_USD, GBP_USD, USD_JPY, etc.)
- Precious metals (XAU_USD, XAG_USD)
- Commodities (BCO_USD, NATGAS_USD)
- Stock indices (SPX500_USD, NAS100_USD, etc.)

Dependencies:
pip install oandapyV20 pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import oandapyV20
from oandapyV20 import API
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.trades as trades
from oandapyV20.exceptions import V20Error

warnings.filterwarnings('ignore')

# Set matplotlib style with fallback
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
        print("Using default matplotlib style")

class MeanReversionStrategy:
    """
    RSI-SMA Mean Reversion Trading Strategy
    
    This strategy uses RSI oversold/overbought conditions combined with 
    trend confirmation via SMA to identify mean reversion opportunities.
    """
    
    def __init__(self, 
                 symbol: str = 'EUR_USD',
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 sma_period: int = 200,
                 stop_loss_pips: float = 50,
                 take_profit_pips: float = 100,
                 position_size: int = 10000):
        """
        Initialize the strategy parameters.
        
        Args:
            symbol: Trading instrument (e.g., 'EUR_USD', 'GBP_JPY', 'XAU_USD')
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            sma_period: Simple Moving Average period
            stop_loss_pips: Stop loss in pips (50 = 50 pips)
            take_profit_pips: Take profit in pips (100 = 100 pips)
            position_size: Position size in units (10000 = 1 mini lot for forex)
        """
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sma_period = sma_period
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.position_size = position_size
        
        # Trading state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        return prices.rolling(window=period).mean()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI and SMA conditions.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals and indicators
        """
        df = data.copy()
        
        # Calculate indicators
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        df['SMA'] = self.calculate_sma(df['Close'], self.sma_period)
        
        # Generate signals
        df['Signal'] = 0
        
        # Buy signal: RSI oversold AND price above SMA (uptrend)
        buy_condition = (df['RSI'] < self.rsi_oversold) & (df['Close'] > df['SMA'])
        df.loc[buy_condition, 'Signal'] = 1
        
        # Sell signal: RSI overbought OR price below SMA (trend reversal)
        sell_condition = (df['RSI'] > self.rsi_overbought) | (df['Close'] < df['SMA'])
        df.loc[sell_condition, 'Signal'] = -1
        
        # Position (1: long, 0: no position, -1: short)
        # Forward fill signals, replacing 0s with previous non-zero values
        position_series = df['Signal'].replace(0, np.nan)
        df['Position'] = position_series.ffill().fillna(0)
        
        return df
    
    def get_oanda_historical_data(self,
                                 access_token: str,
                                 account_id: str,
                                 instrument: str,
                                 start_date: str,
                                 end_date: str,
                                 granularity: str = 'D') -> pd.DataFrame:
        """
        Get historical data from OANDA API.
        
        Args:
            access_token: OANDA access token
            account_id: OANDA account ID
            instrument: Trading instrument (e.g., 'EUR_USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            granularity: Data granularity ('M1', 'M5', 'H1', 'H4', 'D')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            api = API(access_token=access_token, environment='practice')
            
            # First, test account access to verify authorization
            print("Testing account access...")
            try:
                account_req = accounts.AccountDetails(account_id)
                account_info = api.request(account_req)
                print(f"✅ Account access verified: {account_info['account']['id']}")
            except V20Error as acc_error:
                print(f"❌ Account access failed: {acc_error}")
                raise ValueError(f"Cannot access account {account_id}: {acc_error}")
            
            # Convert dates to OANDA format
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # OANDA requires RFC3339 format
            start_rfc = start_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            end_rfc = end_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            
            print(f"Fetching data from {start_rfc} to {end_rfc}")
            
            params = {
                'from': start_rfc,
                'to': end_rfc,
                'granularity': granularity,
                'price': 'M'  # Mid prices
            }
            
            # Get historical candles
            candles_req = instruments.InstrumentsCandles(
                instrument=instrument,
                params=params
            )
            
            print(f"Making API request for {instrument}...")
            candles_data = api.request(candles_req)
            print(f"Received {len(candles_data.get('candles', []))} candles")
            
            # Convert to DataFrame
            data = []
            for candle in candles_data['candles']:
                if candle['complete']:
                    data.append({
                        'Open': float(candle['mid']['o']),
                        'High': float(candle['mid']['h']),
                        'Low': float(candle['mid']['l']),
                        'Close': float(candle['mid']['c']),
                        'Volume': int(candle['volume']),
                        'Time': candle['time']
                    })
            
            if not data:
                raise ValueError(f"No historical data found for {instrument}")
            
            df = pd.DataFrame(data)
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            
            # Sort by time to ensure chronological order
            df.sort_index(inplace=True)
            
            print(f"Successfully processed {len(df)} data points")
            return df
            
        except V20Error as e:
            print(f"OANDA API Error: {e}")
            raise ValueError(f"Error fetching OANDA data: {e}")
        except Exception as e:
            print(f"General Error: {e}")
            raise ValueError(f"Error processing OANDA data: {e}")
    
    def backtest(self, 
                 access_token: str,
                 account_id: str,
                 start_date: str = '2020-01-01', 
                 end_date: str = '2024-01-01',
                 initial_capital: float = 100000,
                 granularity: str = 'D') -> Dict:
        """
        Backtest the strategy using OANDA historical data.
        
        Args:
            access_token: OANDA access token for data access
            account_id: OANDA account ID
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            initial_capital: Initial portfolio value
            granularity: Data granularity ('M1', 'M5', 'H1', 'H4', 'D')
            
        Returns:
            Dictionary with backtest results and performance metrics
        """
        print(f"Backtesting {self.symbol} from {start_date} to {end_date}")
        print(f"Using OANDA data with {granularity} granularity")
        
        # Get historical data from OANDA
        data = self.get_oanda_historical_data(
            access_token=access_token,
            account_id=account_id,
            instrument=self.symbol,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity
        )
        
        if data.empty:
            raise ValueError(f"No data found for instrument {self.symbol}")
        
        # Generate signals
        df = self.generate_signals(data)
        
        # Initialize portfolio
        df['Portfolio_Value'] = initial_capital
        df['Holdings'] = 0
        df['Cash'] = initial_capital
        df['Returns'] = 0
        
        cash = initial_capital
        holdings = 0
        position = 0
        entry_price = None
        trades_list = []
        
        # Calculate pip value for the instrument
        pip_value = 0.01 if 'JPY' in self.symbol else 0.0001
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            
            # Check for new signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size based on risk
                # For simplicity, use 10% of portfolio value
                position_value = cash * 0.1
                
                # Convert to units (for forex, this is typically in base currency units)
                units = int(position_value / current_price) * 1000  # Convert to standard units
                
                if units > 0:
                    cost = (units / 1000) * current_price  # Cost in account currency
                    cash -= cost
                    holdings = units
                    position = 1
                    entry_price = current_price
                    
                    trades_list.append({
                        'Date': df.index[i],
                        'Type': 'BUY',
                        'Price': current_price,
                        'Units': units,
                        'Value': cost
                    })
            
            elif signal == -1 and position == 1:  # Sell signal
                if holdings > 0:
                    proceeds = (holdings / 1000) * current_price
                    cash += proceeds
                    
                    # Calculate trade return
                    trade_return = (current_price - entry_price) / entry_price
                    
                    trades_list.append({
                        'Date': df.index[i],
                        'Type': 'SELL',
                        'Price': current_price,
                        'Units': holdings,
                        'Value': proceeds,
                        'Return': trade_return
                    })
                    
                    holdings = 0
                    position = 0
                    entry_price = None
            
            # Update portfolio values
            if holdings > 0:
                portfolio_value = cash + ((holdings / 1000) * current_price)
            else:
                portfolio_value = cash
                
            df.loc[df.index[i], 'Portfolio_Value'] = portfolio_value
            df.loc[df.index[i], 'Holdings'] = holdings
            df.loc[df.index[i], 'Cash'] = cash
            
            # Calculate returns
            if i > 0:
                df.loc[df.index[i], 'Returns'] = (portfolio_value - df['Portfolio_Value'].iloc[i-1]) / df['Portfolio_Value'].iloc[i-1]
        
        # Calculate performance metrics
        total_return = (df['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital
        
        # Calculate buy-and-hold return for comparison
        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        
        # Calculate Sharpe ratio
        returns = df['Returns'].dropna()
        if granularity == 'D':
            periods_per_year = 252
        elif granularity == 'H4':
            periods_per_year = 252 * 6  # 6 four-hour periods per day
        elif granularity == 'H1':
            periods_per_year = 252 * 24
        else:
            periods_per_year = 252  # Default to daily
            
        sharpe_ratio = np.sqrt(periods_per_year) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        rolling_max = df['Portfolio_Value'].expanding().max()
        drawdown = (df['Portfolio_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate calculation
        profitable_trades = [t for t in trades_list if 'Return' in t and t['Return'] > 0]
        total_trades = len([t for t in trades_list if 'Return' in t])
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        results = {
            'data': df,
            'trades': trades_list,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'initial_capital': initial_capital,
            'final_value': df['Portfolio_Value'].iloc[-1]
        }
        
        return results
    
    def plot_backtest_results(self, results: Dict):
        """Plot comprehensive backtest results."""
        df = results['data']
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Plot 1: Price and Moving Average
        axes[0].plot(df.index, df['Close'], label=f'{self.symbol} Price', linewidth=1)
        axes[0].plot(df.index, df['SMA'], label=f'SMA({self.sma_period})', alpha=0.7)
        
        # Mark buy/sell points
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        axes[0].scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', s=60, label='Buy Signal', alpha=0.7)
        axes[0].scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', s=60, label='Sell Signal', alpha=0.7)
        
        axes[0].set_title(f'{self.symbol} Price and Trading Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: RSI
        axes[1].plot(df.index, df['RSI'], label='RSI', color='purple')
        axes[1].axhline(y=self.rsi_overbought, color='red', linestyle='--', alpha=0.7, label='Overbought')
        axes[1].axhline(y=self.rsi_oversold, color='green', linestyle='--', alpha=0.7, label='Oversold')
        axes[1].set_title('Relative Strength Index (RSI)')
        axes[1].set_ylabel('RSI')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Portfolio Value vs Buy & Hold
        axes[2].plot(df.index, df['Portfolio_Value'], label='Strategy', linewidth=2)
        
        # Calculate buy & hold portfolio value
        buy_hold_portfolio = results['initial_capital'] * (df['Close'] / df['Close'].iloc[0])
        axes[2].plot(df.index, buy_hold_portfolio, label='Buy & Hold', alpha=0.7)
        
        axes[2].set_title('Portfolio Value Comparison')
        axes[2].set_ylabel('Portfolio Value ($)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Drawdown
        rolling_max = df['Portfolio_Value'].expanding().max()
        drawdown = (df['Portfolio_Value'] - rolling_max) / rolling_max * 100
        
        axes[3].fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
        axes[3].plot(df.index, drawdown, color='red', linewidth=1)
        axes[3].set_title('Portfolio Drawdown')
        axes[3].set_ylabel('Drawdown (%)')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        print("\n" + "="*50)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Buy & Hold Return: {results['buy_hold_return']:.2%}")
        print(f"Excess Return: {results['total_return'] - results['buy_hold_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print("="*50)


class OandaLiveTrader:
    """
    Live trading implementation using OANDA API.
    
    This class handles live trading execution of the mean reversion strategy
    using OANDA's forex and CFD trading platform.
    """
    
    def __init__(self, 
                 access_token: str,
                 account_id: str,
                 environment: str = 'practice',  # 'practice' or 'live'
                 strategy: MeanReversionStrategy = None):
        """
        Initialize OANDA API connection.
        
        Args:
            access_token: OANDA access token
            account_id: OANDA account ID
            environment: 'practice' for demo trading, 'live' for real trading
            strategy: MeanReversionStrategy instance
        """
        self.access_token = access_token
        self.account_id = account_id
        self.environment = environment
        self.strategy = strategy or MeanReversionStrategy()
        
        # Initialize OANDA API client
        self.api = API(access_token=access_token, environment=environment)
        
        # Verify account connection
        try:
            account_req = accounts.AccountDetails(self.account_id)
            account_info = self.api.request(account_req)
            print(f"Connected to OANDA account: {account_info['account']['id']}")
            print(f"Account Currency: {account_info['account']['currency']}")
            print(f"Balance: {account_info['account']['balance']}")
            print(f"Margin Available: {account_info['account']['marginAvailable']}")
        except V20Error as e:
            print(f"Error connecting to OANDA: {e}")
    
    def get_pip_value(self, instrument: str) -> float:
        """Calculate pip value for the instrument."""
        # For most major pairs, pip value is 0.0001
        # For JPY pairs, pip value is 0.01
        if 'JPY' in instrument:
            return 0.01
        else:
            return 0.0001
    
    def get_latest_data(self, instrument: str, count: int = 500, granularity: str = 'H1') -> pd.DataFrame:
        """
        Get latest market data for the instrument.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            count: Number of candles to retrieve
            granularity: Data granularity ('M1', 'M5', 'H1', 'H4', 'D')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            params = {
                'count': count,
                'granularity': granularity
            }
            
            candles_req = instruments.InstrumentsCandles(
                instrument=instrument, 
                params=params
            )
            candles_data = self.api.request(candles_req)
            
            # Convert to DataFrame
            data = []
            for candle in candles_data['candles']:
                if candle['complete']:
                    data.append({
                        'Open': float(candle['mid']['o']),
                        'High': float(candle['mid']['h']),
                        'Low': float(candle['mid']['l']),
                        'Close': float(candle['mid']['c']),
                        'Volume': int(candle['volume']),
                        'Time': candle['time']
                    })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['Time'] = pd.to_datetime(df['Time'])
                df.set_index('Time', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except V20Error as e:
            print(f"Error fetching data for {instrument}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, instrument: str) -> Optional[float]:
        """Get current bid/ask prices for an instrument."""
        try:
            params = {'instruments': instrument}
            pricing_req = pricing.PricingInfo(self.account_id, params=params)
            pricing_data = self.api.request(pricing_req)
            
            if pricing_data['prices']:
                price_info = pricing_data['prices'][0]
                # Use mid price for simplicity
                bid = float(price_info['bids'][0]['price'])
                ask = float(price_info['asks'][0]['price'])
                return (bid + ask) / 2
            return None
            
        except V20Error as e:
            print(f"Error getting price for {instrument}: {e}")
            return None
    
    def place_market_order(self, 
                          instrument: str, 
                          units: int, 
                          stop_loss_price: float = None,
                          take_profit_price: float = None) -> Optional[dict]:
        """
        Place a market order with optional stop loss and take profit.
        
        Args:
            instrument: Trading instrument
            units: Number of units (positive for buy, negative for sell)
            stop_loss_price: Stop loss price level
            take_profit_price: Take profit price level
            
        Returns:
            Order response if successful, None otherwise
        """
        try:
            # Prepare order request
            order_data = {
                'order': {
                    'type': 'MARKET',
                    'instrument': instrument,
                    'units': str(units),
                    'timeInForce': 'FOK',  # Fill or Kill
                    'positionFill': 'DEFAULT'
                }
            }
            
            # Add stop loss if provided
            if stop_loss_price:
                order_data['order']['stopLossOnFill'] = {
                    'price': str(round(stop_loss_price, 5))
                }
            
            # Add take profit if provided
            if take_profit_price:
                order_data['order']['takeProfitOnFill'] = {
                    'price': str(round(take_profit_price, 5))
                }
            
            # Place order
            order_req = orders.OrderCreate(self.account_id, data=order_data)
            response = self.api.request(order_req)
            
            action = "BUY" if units > 0 else "SELL"
            print(f"Order placed: {action} {abs(units)} units of {instrument}")
            
            return response
            
        except V20Error as e:
            print(f"Error placing order: {e}")
            return None
    
    def get_open_positions(self) -> dict:
        """Get current open positions."""
        try:
            positions_req = accounts.AccountDetails(self.account_id)
            account_data = self.api.request(positions_req)
            
            positions = {}
            for position in account_data['account']['positions']:
                if float(position['long']['units']) != 0 or float(position['short']['units']) != 0:
                    net_units = float(position['long']['units']) + float(position['short']['units'])
                    positions[position['instrument']] = net_units
            
            return positions
            
        except V20Error as e:
            print(f"Error getting positions: {e}")
            return {}
    
    def close_position(self, instrument: str, side: str = 'ALL') -> Optional[dict]:
        """
        Close position for an instrument.
        
        Args:
            instrument: Trading instrument
            side: 'LONG', 'SHORT', or 'ALL'
        """
        try:
            if side == 'ALL':
                # Close both long and short positions
                close_data = {
                    'longUnits': 'ALL',
                    'shortUnits': 'ALL'
                }
            elif side == 'LONG':
                close_data = {'longUnits': 'ALL'}
            elif side == 'SHORT':
                close_data = {'shortUnits': 'ALL'}
            
            close_req = orders.PositionClose(
                accountID=self.account_id,
                instrument=instrument,
                data=close_data
            )
            response = self.api.request(close_req)
            
            print(f"Closed {side} position for {instrument}")
            return response
            
        except V20Error as e:
            print(f"Error closing position: {e}")
            return None
    
    def calculate_position_size(self, 
                              instrument: str, 
                              risk_amount: float,
                              stop_loss_pips: float) -> int:
        """
        Calculate position size based on risk amount and stop loss.
        
        Args:
            instrument: Trading instrument
            risk_amount: Amount to risk in account currency
            stop_loss_pips: Stop loss distance in pips
            
        Returns:
            Position size in units
        """
        pip_value = self.get_pip_value(instrument)
        
        # Calculate pip value per unit based on instrument
        # For major pairs: 1 pip = 0.0001, for JPY pairs: 1 pip = 0.01
        # For a mini lot (10,000 units), pip value is typically $1 for USD pairs
        
        if 'USD' in instrument:
            # For USD pairs, pip value is roughly $1 per 10,000 units
            pip_value_per_unit = pip_value
        else:
            # For other pairs, estimate pip value
            pip_value_per_unit = pip_value
        
        # Position size = Risk Amount / (Stop Loss Pips * Pip Value)
        if stop_loss_pips > 0:
            position_size = int(risk_amount / (stop_loss_pips * pip_value_per_unit))
        else:
            position_size = 1000  # Default minimum
            
        # Ensure minimum position size
        return max(position_size, 1000)
    
    def run_strategy(self, 
                    instrument: str, 
                    check_interval: int = 300,
                    risk_per_trade: float = 100):
        """
        Run the live trading strategy.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            check_interval: How often to check for signals (seconds)
            risk_per_trade: Amount to risk per trade in account currency
        """
        print(f"Starting live OANDA trading strategy for {instrument}")
        print(f"Check interval: {check_interval} seconds")
        print(f"Risk per trade: {risk_per_trade}")
        
        while True:
            try:
                # Get latest data
                data = self.get_latest_data(instrument, count=max(500, self.strategy.sma_period + 50))
                
                if data.empty:
                    print(f"No data available for {instrument}")
                    continue
                
                # Generate signals
                df_with_signals = self.strategy.generate_signals(data)
                latest_signal = df_with_signals['Signal'].iloc[-1]
                current_price = df_with_signals['Close'].iloc[-1]
                current_rsi = df_with_signals['RSI'].iloc[-1]
                
                # Get current positions
                positions = self.get_open_positions()
                current_position = positions.get(instrument, 0)
                
                print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Price: {current_price:.5f}, RSI: {current_rsi:.2f}")
                print(f"Current Position: {current_position} units")
                print(f"Signal: {latest_signal}")
                
                # Calculate pip value and position size
                pip_value = self.get_pip_value(instrument)
                position_size = self.calculate_position_size(
                    instrument, 
                    risk_per_trade, 
                    self.strategy.stop_loss_pips
                )
                
                # Execute trades based on signals
                if latest_signal == 1 and current_position == 0:  # Buy signal
                    stop_loss_price = current_price - (self.strategy.stop_loss_pips * pip_value)
                    take_profit_price = current_price + (self.strategy.take_profit_pips * pip_value)
                    
                    self.place_market_order(
                        instrument=instrument,
                        units=position_size,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price
                    )
                
                elif latest_signal == -1 and current_position > 0:  # Sell signal
                    self.close_position(instrument, 'ALL')
                
                # Wait for next check
                import time
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\nStopping live trading strategy...")
                break
            except Exception as e:
                print(f"Error in strategy execution: {e}")
                import time
                time.sleep(check_interval)


# Example usage and testing
if __name__ == "__main__":
    
    # Initialize strategy for EUR/USD forex pair
    strategy = MeanReversionStrategy(
        symbol='EUR_USD',
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        sma_period=200,
        stop_loss_pips=50,      # 50 pips stop loss
        take_profit_pips=100,   # 100 pips take profit
        position_size=10000     # 10,000 units (mini lot)
    )
    
    print("RSI-SMA Mean Reversion Trading Strategy - OANDA Edition")
    print("==========================================================")
    print(f"Strategy initialized for {strategy.symbol}")
    print(f"RSI Period: {strategy.rsi_period}, Oversold: {strategy.rsi_oversold}, Overbought: {strategy.rsi_overbought}")
    print(f"SMA Period: {strategy.sma_period}")
    print(f"Risk Management: {strategy.stop_loss_pips} pips SL, {strategy.take_profit_pips} pips TP")
    
    # OANDA credentials for both backtesting and live trading
    ACCESS_TOKEN = "1400757678007e080b3b2a49a1c08e66-44740147c10d16adcc5b66b6b33f6e47"
    ACCOUNT_ID = "001-004-17531327-003"
    
    # First, test credentials
    print("\n1. Testing OANDA Credentials...")
    
    try:
        api = API(access_token=ACCESS_TOKEN, environment='practice')
        account_req = accounts.AccountDetails(ACCOUNT_ID)
        account_info = api.request(account_req)
        
        print(f"✅ Credentials verified!")
        print(f"   Account ID: {account_info['account']['id']}")
        print(f"   Currency: {account_info['account']['currency']}")
        print(f"   Balance: {account_info['account']['balance']}")
        
        # Now run backtest with verified credentials
        print("\n2. Running Backtest with OANDA Data...")
        print("This may take a moment to fetch historical data from OANDA...")
        
        results = strategy.backtest(
            access_token=ACCESS_TOKEN,
            account_id=ACCOUNT_ID,  # Now including account ID
            start_date='2022-01-01',  # Using more recent data
            end_date='2024-01-01',
            initial_capital=100000,
            granularity='D'  # Daily data
        )
        
        print("✅ Backtest completed successfully!")
        
        # Plot results
        strategy.plot_backtest_results(results)
        
    except V20Error as v20_error:
        print(f"❌ OANDA API Error: {v20_error}")
        print("\nPossible issues:")
        print("1. Access token is invalid or expired")
        print("2. Account ID is incorrect")
        print("3. Account doesn't have API access enabled")
        print("4. Token doesn't have permission for this account")
        print("\nSolution:")
        print("1. Go to https://www.oanda.com/")
        print("2. Log into your account")
        print("3. Go to Account Management → My Services → Manage API Access")
        print("4. Revoke current token and generate a new one")
        print("5. Make sure to copy the FULL token (it's quite long)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nGeneral troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify all credentials are correct")
        print("3. Try regenerating your OANDA access token")
    
    # Example for live trading (uncomment and add your OANDA credentials)
    """
    print("\n2. Setting up Live Trading with OANDA...")
    
    # Replace with your actual OANDA credentials
    ACCESS_TOKEN = "your_oanda_access_token_here"
    ACCOUNT_ID = "your_oanda_account_id_here"
    
    # Initialize OANDA live trader (practice environment by default)
    live_trader = OandaLiveTrader(
        access_token=ACCESS_TOKEN,
        account_id=ACCOUNT_ID,
        environment='practice',  # Use 'practice' for demo, 'live' for real trading
        strategy=strategy
    )
    
    # Run live strategy (uncomment to start live trading)
    # live_trader.run_strategy('EUR_USD', check_interval=300, risk_per_trade=100)
    """
    
    # print("\n3. Testing Library Installation...")
    
    # # Test required libraries
    # required_libs = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'oandapyV20']
    # missing_libs = []
    
    # for lib in required_libs:
    #     try:
    #         if lib == 'oandapyV20':
    #             import oandapyV20
    #             print(f"✅ {lib} - OK")
    #         elif lib == 'pandas':
    #             import pandas
    #             print(f"✅ {lib} - OK")
    #         elif lib == 'numpy':
    #             import numpy
    #             print(f"✅ {lib} - OK")
    #         elif lib == 'matplotlib':
    #             import matplotlib
    #             print(f"✅ {lib} - OK")
    #         elif lib == 'seaborn':
    #             import seaborn
    #             print(f"✅ {lib} - OK")
    #     except ImportError:
    #         missing_libs.append(lib)
    #         print(f"❌ {lib} - MISSING")
    
    # if missing_libs:
    #     print(f"\n⚠️  Missing libraries: {', '.join(missing_libs)}")
    #     print(f"Install with: pip install {' '.join(missing_libs)}")
    # else:
    #     print("\n✅ All required libraries are installed!")
    
    # print("\nComplete OANDA-Only Strategy Implementation!")
    # print("\nTo use this strategy:")
    # print("1. Sign up for OANDA at https://www.oanda.com/")
    # print("2. Create a practice (demo) or live account")
    # print("3. Get your access token from: Account Management → My Services → Manage API Access")
    # print("4. Get your account ID from your account details")
    # print("5. Replace ACCESS_TOKEN and ACCOUNT_ID in the code")
    # print("6. Install dependencies: pip install oandapyV20 pandas numpy matplotlib seaborn")
    # print("7. Run backtests and live trading!")
    
    # print("\nAdvantages of OANDA-only approach:")
    # print("✓ Consistent data between backtesting and live trading")
    # print("✓ Access to 30+ years of forex historical data")
    # print("✓ Real-time data without additional subscriptions")
    # print("✓ Support for 90+ currency pairs, metals, and CFDs")
    # print("✓ Professional-grade API with low latency")
    
    # print("\nSupported instruments:")
    # print("- Forex pairs: EUR_USD, GBP_USD, USD_JPY, AUD_USD, etc.")
    # print("- Precious metals: XAU_USD (Gold), XAG_USD (Silver)")
    # print("- Commodities: BCO_USD (Oil), NATGAS_USD (Natural Gas)")
    # print("- Indices: SPX500_USD, NAS100_USD, UK100_GBP, etc.")