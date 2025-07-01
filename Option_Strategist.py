import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Advanced Options Strategist Expert with MarketStack
# =============================================================================

class OptionsStrategist(nn.Module):
    """
    Advanced Options Strategist: Comprehensive options strategy recommendation system
    that analyzes market conditions and recommends specific option trades with 
    detailed risk/reward profiles using MarketStack API.
    """
    
    def __init__(self, seq_len: int = 30, output_dim: int = 10, risk_tolerance: float = 0.02, 
                 marketstack_api_key: str = None):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.risk_tolerance = risk_tolerance  # Max % of portfolio to risk per trade
        self.api_key = "9ad0d4f85e1a72dd7b3d19b8617b25f9"
        self.base_url = "https://api.marketstack.com/v2"
        
        # Strategy mappings
        self.strategies = {
            'BULL_CALL_SPREAD': self._bull_call_spread,
            'BEAR_PUT_SPREAD': self._bear_put_spread,
            'IRON_CONDOR': self._iron_condor,
            'STRADDLE': self._long_straddle,
            'STRANGLE': self._long_strangle,
            'COVERED_CALL': self._covered_call,
            'PROTECTIVE_PUT': self._protective_put,
            'CASH_SECURED_PUT': self._cash_secured_put,
            'COLLAR': self._collar,
            'BUTTERFLY': self._butterfly_spread
        }
        
        # Market regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Bull, Bear, Sideways, High_Vol
        )
        
        # Strategy selector network
        self.strategy_selector = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.strategies))
        )
    
    def update_market_data(self, data_manager):
        """Update market data using the provided data manager"""
        self.data_manager = data_manager
        
    def forward(self, price_series: torch.Tensor, symbols: List[str], 
                portfolio_value: float = 100000) -> Dict:
        """
        Main forward pass that analyzes market conditions and recommends strategies
        """
        recommendations = {}
        
        for i, symbol in enumerate(symbols):
            try:
                # Get market data from MarketStack
                stock_data = self._get_stock_data(symbol)
                
                # Since MarketStack doesn't provide options data, we'll simulate it
                # In production, you'd integrate with an options data provider
                options_data = self._simulate_options_data(symbol, stock_data)
                
                if stock_data is None or options_data is None:
                    continue
                
                # Analyze market conditions
                market_analysis = self._analyze_market_conditions(stock_data, price_series[i])
                
                # Select optimal strategy
                strategy_scores = self._select_strategy(market_analysis, stock_data)
                best_strategy = max(strategy_scores, key=strategy_scores.get)
                
                # Generate specific trade recommendation
                trade_rec = self._generate_trade_recommendation(
                    symbol, best_strategy, stock_data, options_data, 
                    market_analysis, portfolio_value
                )
                
                recommendations[symbol] = {
                    'strategy': best_strategy,
                    'market_analysis': market_analysis,
                    'trade_details': trade_rec,
                    'confidence': strategy_scores[best_strategy]
                }
                
            except Exception as e:
                st.warning(f"Error analyzing {symbol}: {str(e)}")
                continue
                
        return recommendations
    
    def _get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch comprehensive stock data from MarketStack"""
        try:
            if not self.api_key:
                st.error("MarketStack API key not provided")
                return None
            
            # Get historical data (3 months)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            params = {
                'access_key': self.api_key,
                'symbols': symbol,
                'date_from': start_date.strftime('%Y-%m-%d'),
                'date_to': end_date.strftime('%Y-%m-%d'),
                'limit': 1000
            }
            
            response = requests.get(f"{self.base_url}/eod", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return None
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Get current price (latest close)
            current_price = float(df['close'].iloc[-1])
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            sma_20 = df['sma_20'].iloc[-1] if len(df) >= 20 else current_price
            sma_50 = df['sma_50'].iloc[-1] if len(df) >= 50 else current_price
            
            # Volatility metrics
            df['returns'] = df['close'].pct_change()
            returns = df['returns'].dropna()
            realized_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.2
            
            # RSI calculation
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-8)
                rsi_series = 100 - (100 / (1 + rs))
                rsi = rsi_series.iloc[-1]
            else:
                rsi = 50.0  # Neutral RSI if insufficient data
            
            # Get company info for additional metrics
            company_info = self._get_company_info(symbol)
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'realized_vol': realized_vol,
                'rsi': rsi,
                'hist': df,
                'returns': returns,
                'implied_vol': realized_vol * 1.2,  # Estimate IV as 120% of RV
                'beta': company_info.get('beta', 1.0),
                'market_cap': company_info.get('market_cap', 0),
                'pe_ratio': company_info.get('pe_ratio', 0)
            }
            
        except Exception as e:
            st.warning(f"Error fetching stock data for {symbol}: {str(e)}")
            return None
    
    def _get_company_info(self, symbol: str) -> Dict:
        """Get company information from MarketStack"""
        try:
            params = {
                'access_key': self.api_key,
                'ticker': symbol
            }
            
            response = requests.get(f"{self.base_url}/tickerinfo", params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    info = data['data']
                    return {
                        'beta': 1.0,  # MarketStack doesn't provide beta, use default
                        'market_cap': info.get('full_time_employees', 0) * 100000,  # Rough estimate
                        'pe_ratio': 15.0,  # Default P/E ratio
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown')
                    }
            
            return {'beta': 1.0, 'market_cap': 0, 'pe_ratio': 15.0}
            
        except Exception:
            return {'beta': 1.0, 'market_cap': 0, 'pe_ratio': 15.0}
    
    def _simulate_options_data(self, symbol: str, stock_data: Dict) -> Optional[Dict]:
        """
        Simulate options data since MarketStack doesn't provide options chains.
        In production, you would integrate with an options data provider like:
        - Alpha Query, TradingView, IEX Cloud, Polygon.io, etc.
        """
        try:
            current_price = stock_data['current_price']
            volatility = stock_data['realized_vol']
            
            # Generate expiration date (30 days from now)
            expiration = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            days_to_expiry = 30
            
            # Generate strike prices around current price
            strikes = []
            base_strikes = np.arange(0.8, 1.25, 0.05)  # 80% to 125% of current price
            for multiplier in base_strikes:
                strike = round(current_price * multiplier, 2)
                # Round to nearest $0.50 or $1.00 for realism
                if strike < 50:
                    strike = round(strike * 2) / 2  # Round to nearest $0.50
                else:
                    strike = round(strike)  # Round to nearest $1.00
                strikes.append(strike)
            
            strikes = sorted(list(set(strikes)))  # Remove duplicates and sort
            
            # Simulate calls and puts
            calls_data = []
            puts_data = []
            
            for strike in strikes:
                # Simple Black-Scholes approximation for option pricing
                # This is a simplified simulation - use proper options pricing in production
                
                time_to_expiry = days_to_expiry / 365.0
                moneyness = current_price / strike
                
                # Simulate call price
                if moneyness > 1:  # ITM call
                    intrinsic = current_price - strike
                    time_value = volatility * np.sqrt(time_to_expiry) * current_price * 0.1
                    call_price = max(intrinsic + time_value, intrinsic)
                else:  # OTM call
                    time_value = volatility * np.sqrt(time_to_expiry) * current_price * 0.1 * moneyness
                    call_price = max(time_value, 0.01)
                
                # Simulate put price
                if moneyness < 1:  # ITM put
                    intrinsic = strike - current_price
                    time_value = volatility * np.sqrt(time_to_expiry) * current_price * 0.1
                    put_price = max(intrinsic + time_value, intrinsic)
                else:  # OTM put
                    time_value = volatility * np.sqrt(time_to_expiry) * current_price * 0.1 / moneyness
                    put_price = max(time_value, 0.01)
                
                # Simulate volume and open interest (higher for ATM options)
                distance_from_atm = abs(strike - current_price) / current_price
                volume_multiplier = max(0.1, 1 - distance_from_atm * 3)
                
                base_volume = int(100 * volume_multiplier * np.random.uniform(0.5, 2.0))
                base_oi = int(500 * volume_multiplier * np.random.uniform(0.5, 2.0))
                
                calls_data.append({
                    'strike': strike,
                    'lastPrice': round(call_price, 2),
                    'bid': round(call_price * 0.95, 2),
                    'ask': round(call_price * 1.05, 2),
                    'volume': base_volume,
                    'openInterest': base_oi,
                    'impliedVolatility': volatility
                })
                
                puts_data.append({
                    'strike': strike,
                    'lastPrice': round(put_price, 2),
                    'bid': round(put_price * 0.95, 2),
                    'ask': round(put_price * 1.05, 2),
                    'volume': base_volume,
                    'openInterest': base_oi,
                    'impliedVolatility': volatility
                })
            
            calls_df = pd.DataFrame(calls_data)
            puts_df = pd.DataFrame(puts_data)
            
            return {
                'expiration': expiration,
                'calls': calls_df,
                'puts': puts_df,
                'days_to_expiry': days_to_expiry
            }
            
        except Exception as e:
            st.warning(f"Error simulating options data for {symbol}: {str(e)}")
            return None
    
    def _analyze_market_conditions(self, stock_data: Dict, price_tensor: torch.Tensor) -> Dict:
        """Comprehensive market condition analysis"""
        current_price = stock_data['current_price']
        sma_20 = stock_data['sma_20']
        sma_50 = stock_data['sma_50']
        rsi = stock_data['rsi']
        realized_vol = stock_data['realized_vol']
        implied_vol = stock_data['implied_vol']
        
        # Trend analysis
        if current_price > sma_20 > sma_50:
            trend = 'BULLISH'
            trend_strength = min((current_price - sma_50) / sma_50, 0.1) * 10
        elif current_price < sma_20 < sma_50:
            trend = 'BEARISH'
            trend_strength = min((sma_50 - current_price) / sma_50, 0.1) * 10
        else:
            trend = 'SIDEWAYS'
            trend_strength = 5.0
        
        # Volatility regime
        vol_ratio = implied_vol / realized_vol if realized_vol > 0 else 1.0
        if vol_ratio > 1.2:
            vol_regime = 'HIGH_IV'  # Options expensive
        elif vol_ratio < 0.8:
            vol_regime = 'LOW_IV'   # Options cheap
        else:
            vol_regime = 'NORMAL_IV'
        
        # Momentum
        if rsi > 70:
            momentum = 'OVERBOUGHT'
        elif rsi < 30:
            momentum = 'OVERSOLD'
        else:
            momentum = 'NEUTRAL'
        
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'volatility_regime': vol_regime,
            'momentum': momentum,
            'iv_rank': min(max((vol_ratio - 0.5) * 100, 0), 100),
            'rsi': rsi,
            'vol_ratio': vol_ratio
        }
    
    def _select_strategy(self, market_analysis: Dict, stock_data: Dict) -> Dict[str, float]:
        """Score different strategies based on market conditions"""
        scores = {}
        
        trend = market_analysis['trend']
        vol_regime = market_analysis['volatility_regime']
        momentum = market_analysis['momentum']
        iv_rank = market_analysis['iv_rank']
        
        # Directional strategies
        if trend == 'BULLISH':
            scores['BULL_CALL_SPREAD'] = 8.0 + (market_analysis['trend_strength'] / 10)
            scores['COVERED_CALL'] = 7.0 if vol_regime == 'HIGH_IV' else 5.0
            scores['CASH_SECURED_PUT'] = 7.5 if momentum == 'OVERSOLD' else 6.0
        
        if trend == 'BEARISH':
            scores['BEAR_PUT_SPREAD'] = 8.0 + (market_analysis['trend_strength'] / 10)
            scores['PROTECTIVE_PUT'] = 8.5
        
        # Neutral strategies
        if trend == 'SIDEWAYS':
            scores['IRON_CONDOR'] = 8.5 if vol_regime == 'HIGH_IV' else 6.0
            scores['BUTTERFLY'] = 7.5
            scores['COVERED_CALL'] = 8.0 if vol_regime == 'HIGH_IV' else 6.0
        
        # Volatility strategies
        if vol_regime == 'HIGH_IV':
            scores['IRON_CONDOR'] = scores.get('IRON_CONDOR', 0) + 2.0
            scores['STRANGLE'] = 6.0  # Sell premium
        
        if vol_regime == 'LOW_IV':
            scores['STRADDLE'] = 8.0  # Buy cheap options
            scores['STRANGLE'] = 8.0
        
        # Default scoring for all strategies
        for strategy in self.strategies.keys():
            if strategy not in scores:
                scores[strategy] = 5.0  # Neutral score
        
        return scores
    
    def _generate_trade_recommendation(self, symbol: str, strategy: str, 
                                     stock_data: Dict, options_data: Dict,
                                     market_analysis: Dict, portfolio_value: float) -> Dict:
        """Generate specific trade recommendation with risk/reward analysis"""
        
        if strategy not in self.strategies:
            return {'error': f'Unknown strategy: {strategy}'}
        
        try:
            return self.strategies[strategy](symbol, stock_data, options_data, 
                                           market_analysis, portfolio_value)
        except Exception as e:
            return {'error': f'Failed to generate {strategy} recommendation: {str(e)}'}
    
    def _bull_call_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                         market_analysis: Dict, portfolio_value: float) -> Dict:
        """Bull Call Spread: Buy lower strike call, sell higher strike call"""
        current_price = stock_data['current_price']
        calls = options_data['calls']
        days_to_expiry = options_data['days_to_expiry']
        
        # Filter for liquid options (open interest > 10)
        liquid_calls = calls[calls['openInterest'] > 10].copy()
        if liquid_calls.empty:
            liquid_calls = calls.copy()
        
        # Find ATM call to buy
        liquid_calls['strike_diff'] = abs(liquid_calls['strike'] - current_price)
        atm_call = liquid_calls.loc[liquid_calls['strike_diff'].idxmin()]
        
        # Find OTM call to sell (5-10% above current price)
        target_strike = current_price * 1.07
        otm_calls = liquid_calls[liquid_calls['strike'] > current_price]
        if not otm_calls.empty:
            otm_calls['strike_diff'] = abs(otm_calls['strike'] - target_strike)
            otm_call = otm_calls.loc[otm_calls['strike_diff'].idxmin()]
        else:
            return {'error': 'No suitable OTM calls found'}
        
        # Calculate trade metrics
        net_debit = atm_call['lastPrice'] - otm_call['lastPrice']
        max_profit = (otm_call['strike'] - atm_call['strike']) - net_debit
        max_loss = net_debit
        breakeven = atm_call['strike'] + net_debit
        
        # Position sizing
        risk_amount = portfolio_value * self.risk_tolerance
        max_contracts = int(risk_amount / (max_loss * 100))
        contracts = max(1, min(max_contracts, 10))  # Limit to 10 contracts
        
        return {
            'strategy_name': 'Bull Call Spread',
            'legs': [
                {
                    'action': 'BUY',
                    'option_type': 'CALL',
                    'strike': atm_call['strike'],
                    'expiration': options_data['expiration'],
                    'price': atm_call['lastPrice'],
                    'contracts': contracts
                },
                {
                    'action': 'SELL',
                    'option_type': 'CALL',
                    'strike': otm_call['strike'],
                    'expiration': options_data['expiration'],
                    'price': otm_call['lastPrice'],
                    'contracts': contracts
                }
            ],
            'net_debit': net_debit * contracts * 100,
            'max_profit': max_profit * contracts * 100,
            'max_loss': max_loss * contracts * 100,
            'breakeven': breakeven,
            'days_to_expiry': days_to_expiry,
            'profit_probability': self._calculate_profit_probability(current_price, breakeven, stock_data['realized_vol'], days_to_expiry),
            'risk_reward_ratio': max_profit / max_loss if max_loss > 0 else 0,
            'rationale': f"Bullish outlook with limited risk. Profit if {symbol} closes above ${breakeven:.2f} at expiration."
        }
    
    def _bear_put_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                        market_analysis: Dict, portfolio_value: float) -> Dict:
        """Bear Put Spread: Buy higher strike put, sell lower strike put"""
        current_price = stock_data['current_price']
        puts = options_data['puts']
        days_to_expiry = options_data['days_to_expiry']
        
        liquid_puts = puts[puts['openInterest'] > 10].copy()
        if liquid_puts.empty:
            liquid_puts = puts.copy()
        
        # Find ATM put to buy
        liquid_puts['strike_diff'] = abs(liquid_puts['strike'] - current_price)
        atm_put = liquid_puts.loc[liquid_puts['strike_diff'].idxmin()]
        
        # Find OTM put to sell (5-10% below current price)
        target_strike = current_price * 0.93
        otm_puts = liquid_puts[liquid_puts['strike'] < current_price]
        if not otm_puts.empty:
            otm_puts['strike_diff'] = abs(otm_puts['strike'] - target_strike)
            otm_put = otm_puts.loc[otm_puts['strike_diff'].idxmin()]
        else:
            return {'error': 'No suitable OTM puts found'}
        
        net_debit = atm_put['lastPrice'] - otm_put['lastPrice']
        max_profit = (atm_put['strike'] - otm_put['strike']) - net_debit
        max_loss = net_debit
        breakeven = atm_put['strike'] - net_debit
        
        risk_amount = portfolio_value * self.risk_tolerance
        max_contracts = int(risk_amount / (max_loss * 100))
        contracts = max(1, min(max_contracts, 10))
        
        return {
            'strategy_name': 'Bear Put Spread',
            'legs': [
                {
                    'action': 'BUY',
                    'option_type': 'PUT',
                    'strike': atm_put['strike'],
                    'expiration': options_data['expiration'],
                    'price': atm_put['lastPrice'],
                    'contracts': contracts
                },
                {
                    'action': 'SELL',
                    'option_type': 'PUT',
                    'strike': otm_put['strike'],
                    'expiration': options_data['expiration'],
                    'price': otm_put['lastPrice'],
                    'contracts': contracts
                }
            ],
            'net_debit': net_debit * contracts * 100,
            'max_profit': max_profit * contracts * 100,
            'max_loss': max_loss * contracts * 100,
            'breakeven': breakeven,
            'days_to_expiry': days_to_expiry,
            'profit_probability': self._calculate_profit_probability(current_price, breakeven, stock_data['realized_vol'], days_to_expiry, direction='down'),
            'risk_reward_ratio': max_profit / max_loss if max_loss > 0 else 0,
            'rationale': f"Bearish outlook with limited risk. Profit if {symbol} closes below ${breakeven:.2f} at expiration."
        }
    
    def _iron_condor(self, symbol: str, stock_data: Dict, options_data: Dict,
                    market_analysis: Dict, portfolio_value: float) -> Dict:
        """Iron Condor: Sell call spread + sell put spread"""
        current_price = stock_data['current_price']
        calls = options_data['calls']
        puts = options_data['puts']
        
        # Find OTM options to sell
        call_strikes = calls[calls['strike'] > current_price * 1.05]['strike'].values
        put_strikes = puts[puts['strike'] < current_price * 0.95]['strike'].values
        
        if len(call_strikes) < 2 or len(put_strikes) < 2:
            return {'error': 'Insufficient strikes for Iron Condor'}
        
        # Select strikes
        short_call_strike = call_strikes[0]
        long_call_strike = call_strikes[1] if len(call_strikes) > 1 else call_strikes[0] + 5
        short_put_strike = put_strikes[-1]
        long_put_strike = put_strikes[-2] if len(put_strikes) > 1 else put_strikes[-1] - 5
        
        # Get option prices
        short_call = calls[calls['strike'] == short_call_strike].iloc[0]
        long_call = calls[calls['strike'] == long_call_strike].iloc[0]
        short_put = puts[puts['strike'] == short_put_strike].iloc[0]
        long_put = puts[puts['strike'] == long_put_strike].iloc[0]
        
        net_credit = (short_call['lastPrice'] + short_put['lastPrice']) - (long_call['lastPrice'] + long_put['lastPrice'])
        max_profit = net_credit
        max_loss = min(long_call_strike - short_call_strike, short_put_strike - long_put_strike) - net_credit
        
        risk_amount = portfolio_value * self.risk_tolerance
        max_contracts = int(risk_amount / (abs(max_loss) * 100))
        contracts = max(1, min(max_contracts, 5))
        
        return {
            'strategy_name': 'Iron Condor',
            'legs': [
                {'action': 'SELL', 'option_type': 'CALL', 'strike': short_call_strike, 
                 'expiration': options_data['expiration'], 'price': short_call['lastPrice'], 'contracts': contracts},
                {'action': 'BUY', 'option_type': 'CALL', 'strike': long_call_strike, 
                 'expiration': options_data['expiration'], 'price': long_call['lastPrice'], 'contracts': contracts},
                {'action': 'SELL', 'option_type': 'PUT', 'strike': short_put_strike, 
                 'expiration': options_data['expiration'], 'price': short_put['lastPrice'], 'contracts': contracts},
                {'action': 'BUY', 'option_type': 'PUT', 'strike': long_put_strike, 
                 'expiration': options_data['expiration'], 'price': long_put['lastPrice'], 'contracts': contracts}
            ],
            'net_credit': net_credit * contracts * 100,
            'max_profit': max_profit * contracts * 100,
            'max_loss': max_loss * contracts * 100,
            'profit_range': (short_put_strike + net_credit, short_call_strike - net_credit),
            'days_to_expiry': options_data['days_to_expiry'],
            'rationale': f"Profit if {symbol} stays between ${short_put_strike + net_credit:.2f} and ${short_call_strike - net_credit:.2f}"
        }
    
    def _long_straddle(self, symbol: str, stock_data: Dict, options_data: Dict,
                      market_analysis: Dict, portfolio_value: float) -> Dict:
        """Long Straddle: Buy ATM call and put"""
        current_price = stock_data['current_price']
        calls = options_data['calls']
        puts = options_data['puts']
        
        # Find ATM options
        calls['strike_diff'] = abs(calls['strike'] - current_price)
        puts['strike_diff'] = abs(puts['strike'] - current_price)
        
        atm_call = calls.loc[calls['strike_diff'].idxmin()]
        atm_put = puts.loc[puts['strike_diff'].idxmin()]
        
        net_debit = atm_call['lastPrice'] + atm_put['lastPrice']
        breakeven_up = atm_call['strike'] + net_debit
        breakeven_down = atm_put['strike'] - net_debit
        
        risk_amount = portfolio_value * self.risk_tolerance
        max_contracts = int(risk_amount / (net_debit * 100))
        contracts = max(1, min(max_contracts, 10))
        
        return {
            'strategy_name': 'Long Straddle',
            'legs': [
                {'action': 'BUY', 'option_type': 'CALL', 'strike': atm_call['strike'],
                 'expiration': options_data['expiration'], 'price': atm_call['lastPrice'], 'contracts': contracts},
                {'action': 'BUY', 'option_type': 'PUT', 'strike': atm_put['strike'],
                 'expiration': options_data['expiration'], 'price': atm_put['lastPrice'], 'contracts': contracts}
            ],
            'net_debit': net_debit * contracts * 100,
            'max_loss': net_debit * contracts * 100,
            'breakeven_up': breakeven_up,
            'breakeven_down': breakeven_down,
            'days_to_expiry': options_data['days_to_expiry'],
            'rationale': f"Profit if {symbol} moves beyond ${breakeven_down:.2f} or ${breakeven_up:.2f}"
        }
    
    # Implement remaining strategies with similar detailed analysis...
    def _long_strangle(self, symbol: str, stock_data: Dict, options_data: Dict,
                      market_analysis: Dict, portfolio_value: float) -> Dict:
        """Long Strangle: Buy OTM call and put"""
        current_price = stock_data['current_price']
        calls = options_data['calls']
        puts = options_data['puts']
        
        # Find OTM options
        otm_calls = calls[calls['strike'] > current_price * 1.03]
        otm_puts = puts[puts['strike'] < current_price * 0.97]
        
        if otm_calls.empty or otm_puts.empty:
            return {'error': 'No suitable OTM options for strangle'}
        
        call_option = otm_calls.iloc[0]
        put_option = otm_puts.iloc[-1]
        
        net_debit = call_option['lastPrice'] + put_option['lastPrice']
        breakeven_up = call_option['strike'] + net_debit
        breakeven_down = put_option['strike'] - net_debit
        
        risk_amount = portfolio_value * self.risk_tolerance
        max_contracts = int(risk_amount / (net_debit * 100))
        contracts = max(1, min(max_contracts, 10))
        
        return {
            'strategy_name': 'Long Strangle',
            'legs': [
                {'action': 'BUY', 'option_type': 'CALL', 'strike': call_option['strike'],
                 'expiration': options_data['expiration'], 'price': call_option['lastPrice'], 'contracts': contracts},
                {'action': 'BUY', 'option_type': 'PUT', 'strike': put_option['strike'],
                 'expiration': options_data['expiration'], 'price': put_option['lastPrice'], 'contracts': contracts}
            ],
            'net_debit': net_debit * contracts * 100,
            'max_loss': net_debit * contracts * 100,
            'breakeven_up': breakeven_up,
            'breakeven_down': breakeven_down,
            'days_to_expiry': options_data['days_to_expiry'],
            'rationale': f"Lower cost volatility play. Profit if {symbol} moves beyond ${breakeven_down:.2f} or ${breakeven_up:.2f}"
        }
    
    def _covered_call(self, symbol: str, stock_data: Dict, options_data: Dict,
                     market_analysis: Dict, portfolio_value: float) -> Dict:
        """Covered Call: Own stock + sell call"""
        current_price = stock_data['current_price']
        calls = options_data['calls']
        
        # Find slightly OTM call to sell
        otm_calls = calls[calls['strike'] > current_price * 1.02]
        if otm_calls.empty:
            return {'error': 'No suitable calls for covered call'}
        
        call_to_sell = otm_calls.iloc[0]
        shares_needed = 100
        cost_basis = current_price * shares_needed
        premium_received = call_to_sell['lastPrice'] * 100
        
        return {
            'strategy_name': 'Covered Call',
            'legs': [
                {'action': 'BUY', 'instrument': 'STOCK', 'quantity': shares_needed, 'price': current_price},
                {'action': 'SELL', 'option_type': 'CALL', 'strike': call_to_sell['strike'],
                 'expiration': options_data['expiration'], 'price': call_to_sell['lastPrice'], 'contracts': 1}
            ],
            'initial_cost': cost_basis - premium_received,
            'premium_received': premium_received,
            'max_profit': (call_to_sell['strike'] * shares_needed) - cost_basis + premium_received,
            'breakeven': current_price - call_to_sell['lastPrice'],
            'assignment_risk': call_to_sell['strike'],
            'days_to_expiry': options_data['days_to_expiry'],
            'rationale': f"Generate income from stock holdings. Called away if {symbol} > ${call_to_sell['strike']:.2f}"
        }
    
    def _protective_put(self, symbol: str, stock_data: Dict, options_data: Dict,
                       market_analysis: Dict, portfolio_value: float) -> Dict:
        """Protective Put: Own stock + buy put"""
        current_price = stock_data['current_price']
        puts = options_data['puts']
        
        # Find put 5-10% OTM for protection
        protective_puts = puts[puts['strike'] < current_price * 0.95]
        if protective_puts.empty:
            return {'error': 'No suitable puts for protection'}
        
        put_to_buy = protective_puts.iloc[-1]  # Closest to target
        shares_owned = 100
        insurance_cost = put_to_buy['lastPrice'] * 100
        
        return {
            'strategy_name': 'Protective Put',
            'legs': [
                {'action': 'OWN', 'instrument': 'STOCK', 'quantity': shares_owned, 'price': current_price},
                {'action': 'BUY', 'option_type': 'PUT', 'strike': put_to_buy['strike'],
                 'expiration': options_data['expiration'], 'price': put_to_buy['lastPrice'], 'contracts': 1}
            ],
            'insurance_cost': insurance_cost,
            'protected_value': put_to_buy['strike'] * shares_owned,
            'max_loss': (current_price - put_to_buy['strike']) * shares_owned + insurance_cost,
            'days_to_expiry': options_data['days_to_expiry'],
            'rationale': f"Downside protection. Limits loss if {symbol} falls below ${put_to_buy['strike']:.2f}"
        }
    
    def _cash_secured_put(self, symbol: str, stock_data: Dict, options_data: Dict,
                         market_analysis: Dict, portfolio_value: float) -> Dict:
        """Cash Secured Put: Sell put with cash backing"""
        current_price = stock_data['current_price']
        puts = options_data['puts']
        
        # Find put 5-10% OTM to sell
        target_strike = current_price * 0.95
        otm_puts = puts[puts['strike'] < current_price]
        if otm_puts.empty:
            return {'error': 'No suitable puts for cash secured put'}
        
        otm_puts['strike_diff'] = abs(otm_puts['strike'] - target_strike)
        put_to_sell = otm_puts.loc[otm_puts['strike_diff'].idxmin()]
        
        cash_required = put_to_sell['strike'] * 100
        premium_received = put_to_sell['lastPrice'] * 100
        
        return {
            'strategy_name': 'Cash Secured Put',
            'legs': [
                {'action': 'SELL', 'option_type': 'PUT', 'strike': put_to_sell['strike'],
                 'expiration': options_data['expiration'], 'price': put_to_sell['lastPrice'], 'contracts': 1}
            ],
            'cash_required': cash_required,
            'premium_received': premium_received,
            'net_cost_if_assigned': cash_required - premium_received,
            'breakeven': put_to_sell['strike'] - put_to_sell['lastPrice'],
            'assignment_probability': self._calculate_assignment_probability(current_price, put_to_sell['strike'], stock_data['realized_vol'], options_data['days_to_expiry']),
            'days_to_expiry': options_data['days_to_expiry'],
            'rationale': f"Generate income with willingness to buy {symbol} at ${put_to_sell['strike']:.2f}"
        }
    
    def _collar(self, symbol: str, stock_data: Dict, options_data: Dict,
                market_analysis: Dict, portfolio_value: float) -> Dict:
        """Collar: Own stock + buy put + sell call"""
        current_price = stock_data['current_price']
        calls = options_data['calls']
        puts = options_data['puts']
        
        # Find protective put
        protective_puts = puts[puts['strike'] < current_price * 0.95]
        if protective_puts.empty:
            return {'error': 'No suitable puts for collar'}
        
        # Find call to sell
        calls_to_sell = calls[calls['strike'] > current_price * 1.05]
        if calls_to_sell.empty:
            return {'error': 'No suitable calls for collar'}
        
        put_to_buy = protective_puts.iloc[-1]
        call_to_sell = calls_to_sell.iloc[0]
        
        net_premium = call_to_sell['lastPrice'] - put_to_buy['lastPrice']
        shares_owned = 100
        
        return {
            'strategy_name': 'Collar',
            'legs': [
                {'action': 'OWN', 'instrument': 'STOCK', 'quantity': shares_owned, 'price': current_price},
                {'action': 'BUY', 'option_type': 'PUT', 'strike': put_to_buy['strike'],
                 'expiration': options_data['expiration'], 'price': put_to_buy['lastPrice'], 'contracts': 1},
                {'action': 'SELL', 'option_type': 'CALL', 'strike': call_to_sell['strike'],
                 'expiration': options_data['expiration'], 'price': call_to_sell['lastPrice'], 'contracts': 1}
            ],
            'net_premium': net_premium * 100,
            'protected_floor': put_to_buy['strike'],
            'upside_cap': call_to_sell['strike'],
            'max_profit': (call_to_sell['strike'] - current_price) * shares_owned + (net_premium * 100),
            'max_loss': (current_price - put_to_buy['strike']) * shares_owned - (net_premium * 100),
            'days_to_expiry': options_data['days_to_expiry'],
            'rationale': f"Limited risk/reward. Profit range: ${put_to_buy['strike']:.2f} - ${call_to_sell['strike']:.2f}"
        }
    
    def _butterfly_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                         market_analysis: Dict, portfolio_value: float) -> Dict:
        """Butterfly Spread: Buy 1 ITM, Sell 2 ATM, Buy 1 OTM (calls or puts)"""
        current_price = stock_data['current_price']
        calls = options_data['calls']
        
        # Use calls for butterfly
        itm_calls = calls[calls['strike'] < current_price]
        atm_calls = calls[abs(calls['strike'] - current_price) < current_price * 0.02]
        otm_calls = calls[calls['strike'] > current_price]
        
        if itm_calls.empty or atm_calls.empty or otm_calls.empty:
            return {'error': 'Insufficient strikes for butterfly spread'}
        
        itm_call = itm_calls.iloc[-1]  # Closest ITM
        atm_call = atm_calls.iloc[0]   # Closest ATM
        otm_call = otm_calls.iloc[0]   # Closest OTM
        
        net_debit = itm_call['lastPrice'] + otm_call['lastPrice'] - (2 * atm_call['lastPrice'])
        max_profit = (atm_call['strike'] - itm_call['strike']) - net_debit
        max_loss = net_debit
        
        risk_amount = portfolio_value * self.risk_tolerance
        max_contracts = int(risk_amount / (max_loss * 100))
        contracts = max(1, min(max_contracts, 5))
        
        return {
            'strategy_name': 'Butterfly Spread',
            'legs': [
                {'action': 'BUY', 'option_type': 'CALL', 'strike': itm_call['strike'],
                 'expiration': options_data['expiration'], 'price': itm_call['lastPrice'], 'contracts': contracts},
                {'action': 'SELL', 'option_type': 'CALL', 'strike': atm_call['strike'],
                 'expiration': options_data['expiration'], 'price': atm_call['lastPrice'], 'contracts': contracts * 2},
                {'action': 'BUY', 'option_type': 'CALL', 'strike': otm_call['strike'],
                 'expiration': options_data['expiration'], 'price': otm_call['lastPrice'], 'contracts': contracts}
            ],
            'net_debit': net_debit * contracts * 100,
            'max_profit': max_profit * contracts * 100,
            'max_loss': max_loss * contracts * 100,
            'optimal_price': atm_call['strike'],
            'days_to_expiry': options_data['days_to_expiry'],
            'rationale': f"Max profit if {symbol} closes at ${atm_call['strike']:.2f} at expiration"
        }
    
    def _calculate_profit_probability(self, current_price: float, target_price: float, 
                                    volatility: float, days_to_expiry: int, direction: str = 'up') -> float:
        """Calculate probability of reaching target price using Black-Scholes assumptions"""
        if days_to_expiry <= 0:
            return 0.0
        
        time_to_expiry = days_to_expiry / 365.0
        
        # Log-normal distribution assumptions
        d = (np.log(target_price / current_price)) / (volatility * np.sqrt(time_to_expiry))
        
        if direction == 'up':
            # Probability of price being above target (simplified normal approximation)
            probability = 0.5 * (1 + np.tanh(d))
        else:
            # Probability of price being below target
            probability = 0.5 * (1 - np.tanh(d))
        
        return max(0.0, min(1.0, probability)) * 100  # Return as percentage
    
    def _calculate_assignment_probability(self, current_price: float, strike_price: float,
                                        volatility: float, days_to_expiry: int) -> float:
        """Calculate probability of option assignment"""
        return self._calculate_profit_probability(current_price, strike_price, volatility, days_to_expiry, direction='down')
    
    def display_recommendations(self, recommendations: Dict) -> None:
        """Display formatted recommendations in Streamlit"""
        st.title("🎯 Options Strategy Recommendations")
        
        if not recommendations:
            st.warning("No recommendations available. Please check your symbols and try again.")
            return
        
        for symbol, rec in recommendations.items():
            if 'error' in rec.get('trade_details', {}):
                st.error(f"{symbol}: {rec['trade_details']['error']}")
                continue
                
            with st.expander(f"📊 {symbol} - {rec['strategy']} (Confidence: {rec['confidence']:.1f}/10)", expanded=True):
                trade = rec['trade_details']
                
                # Market Analysis Summary
                st.subheader("Market Analysis")
                analysis = rec['market_analysis']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Trend", analysis['trend'])
                    st.metric("RSI", f"{analysis['rsi']:.1f}")
                
                with col2:
                    st.metric("Volatility Regime", analysis['volatility_regime'])
                    st.metric("IV Rank", f"{analysis['iv_rank']:.1f}%")
                
                with col3:
                    st.metric("Momentum", analysis['momentum'])
                    st.metric("Vol Ratio", f"{analysis['vol_ratio']:.2f}")
                
                # Strategy Details
                st.subheader(f"Strategy: {trade['strategy_name']}")
                st.info(trade['rationale'])
                
                # Trade Legs
                st.subheader("Trade Details")
                legs_df = pd.DataFrame(trade['legs'])
                st.dataframe(legs_df, use_container_width=True)
                
                # Risk/Reward Metrics
                st.subheader("Risk/Reward Analysis")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    if 'max_profit' in trade:
                        st.metric("Max Profit", f"${trade['max_profit']:,.2f}")
                    if 'max_loss' in trade:
                        st.metric("Max Loss", f"${trade['max_loss']:,.2f}")
                    if 'net_debit' in trade:
                        st.metric("Net Debit", f"${trade['net_debit']:,.2f}")
                    elif 'net_credit' in trade:
                        st.metric("Net Credit", f"${trade['net_credit']:,.2f}")
                
                with metrics_col2:
                    if 'breakeven' in trade:
                        st.metric("Breakeven", f"${trade['breakeven']:.2f}")
                    elif 'breakeven_up' in trade and 'breakeven_down' in trade:
                        st.metric("Breakeven Range", f"${trade['breakeven_down']:.2f} - ${trade['breakeven_up']:.2f}")
                    if 'risk_reward_ratio' in trade:
                        st.metric("Risk/Reward Ratio", f"{trade['risk_reward_ratio']:.2f}")
                    if 'profit_probability' in trade:
                        st.metric("Profit Probability", f"{trade['profit_probability']:.1f}%")
                
                # Additional Strategy-Specific Metrics
                if 'profit_range' in trade:
                    st.metric("Profit Range", f"${trade['profit_range'][0]:.2f} - ${trade['profit_range'][1]:.2f}")
                
                if 'assignment_probability' in trade:
                    st.metric("Assignment Probability", f"{trade['assignment_probability']:.1f}%")
                
                st.metric("Days to Expiry", trade['days_to_expiry'])

# =============================================================================
# Streamlit Interface for Options Strategist
# =============================================================================

def main():
    st.set_page_config(page_title="Options Strategist Expert", page_icon="🎯", layout="wide")

    api_key = "9ad0d4f85e1a72dd7b3d19b8617b25f9"
    
    st.title("🎯 Advanced Options Strategist")
    st.markdown("*AI-powered options strategy recommendations based on comprehensive market analysis*")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        
        # # API Key input
        # api_key = st.text_input("MarketStack API Key", type="password", 
        #                        help="Enter your MarketStack API key")
        
        if not api_key:
            st.warning("Please enter your MarketStack API key to continue")
            st.stop()
        
        # Portfolio settings
        portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=100000, step=1000)
        risk_tolerance = st.slider("Risk Tolerance (%)", min_value=1, max_value=10, value=2) / 100
        
        # Symbol input
        symbols_input = st.text_input("Enter stock symbols (comma-separated)", value="AAPL,TSLA,SPY")
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        
        analyze_button = st.button("🔍 Analyze Options Strategies", type="primary")
    
    if analyze_button and symbols:
        with st.spinner("Analyzing market conditions and generating recommendations..."):
            try:
                # Initialize the strategist with API key
                strategist = OptionsStrategist(risk_tolerance=risk_tolerance, 
                                             marketstack_api_key=api_key)
                
                # Create dummy price series for the forward pass
                price_series = torch.randn(len(symbols), 30) * 0.02 + 1  # Simulate 30-day returns
                
                # Get recommendations
                recommendations = strategist.forward(price_series, symbols, portfolio_value)
                
                # Display results
                strategist.display_recommendations(recommendations)
                
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.exception(e)
    
    # Educational content
    with st.expander("📚 Options Strategies Guide"):
        st.markdown("""
        ### Available Strategies:
        
        **Directional Strategies:**
        - **Bull Call Spread**: Limited risk/reward bullish strategy
        - **Bear Put Spread**: Limited risk/reward bearish strategy
        - **Covered Call**: Income generation from stock holdings
        - **Cash Secured Put**: Income generation with willingness to buy stock
        
        **Neutral/Income Strategies:**
        - **Iron Condor**: Profit from low volatility/sideways movement
        - **Butterfly Spread**: High probability, limited profit neutral strategy
        - **Collar**: Protective strategy with limited upside
        
        **Volatility Strategies:**
        - **Long Straddle**: Profit from large price movements in either direction
        - **Long Strangle**: Lower cost volatility play with wider breakeven points
        - **Protective Put**: Insurance for stock positions
        
        ### Important Notes:
        - **Options Data**: Since MarketStack doesn't provide options chains, this system simulates realistic options data for demonstration purposes
        - **Production Use**: For live trading, integrate with a dedicated options data provider like Alpha Query, TradingView, or IEX Cloud
        - **Risk Management**: Position sizing based on portfolio value and risk tolerance
        - **Educational Purpose**: This tool is for educational and analysis purposes only
        
        ### Risk Management:
        - Position sizing based on portfolio value and risk tolerance
        - Maximum risk per trade limited by settings
        - Comprehensive profit probability calculations
        - Realistic options pricing simulation based on volatility
        """)

if __name__ == "__main__":
    main()