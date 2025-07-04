import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import time
from scipy.stats import norm
import math
import json

warnings.filterwarnings('ignore')

# =============================================================================
# Enhanced Options Strategist with Fixed Strategy Logic
# =============================================================================

class OptionsStrategist:
    """
    Enhanced Options Strategist with robust error handling and fixed strategy implementations
    """
    
    def __init__(self, risk_tolerance: float = 0.02, 
                 marketstack_api_key: str = None, 
                 polygon_api_key: str = None):
        self.risk_tolerance = risk_tolerance
        self.marketstack_api_key = marketstack_api_key
        self.polygon_api_key = polygon_api_key
        self.marketstack_url = "https://api.marketstack.com/v2"
        self.polygon_url = "https://api.polygon.io"
        
        # Rate limiting
        self.last_api_call = {}
        self.min_interval = 0.5  # Minimum seconds between API calls
        
        # Strategy mappings with improved fallback logic
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
    
    def _rate_limit(self, api_name: str):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        if api_name in self.last_api_call:
            time_since_last = current_time - self.last_api_call[api_name]
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
        self.last_api_call[api_name] = time.time()
    
    def _make_api_request(self, url: str, params: Dict, api_name: str, timeout: int = 10) -> Optional[Dict]:
        """Make API request with error handling and rate limiting"""
        try:
            self._rate_limit(api_name)
            
            response = requests.get(url, params=params, timeout=timeout)
            
            # Log the request for debugging
            # st.write(f"🔍 API Request: {api_name}")
            # st.write(f"📍 URL: {url}")
            # st.write(f"📋 Status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                st.error(f"❌ {api_name} API: Access forbidden (403). Check your API key and permissions.")
                return None
            elif response.status_code == 429:
                st.warning(f"⚠️ {api_name} API: Rate limit exceeded. Waiting...")
                time.sleep(2)
                return None
            else:
                st.warning(f"⚠️ {api_name} API returned status {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            st.error(f"⏰ {api_name} API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"🔌 {api_name} API connection error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error with {api_name} API: {str(e)}")
            return None

    def analyze_symbols(self, symbols: List[str], portfolio_value: float = 100000) -> Dict:
        """Main analysis function with enhanced error handling and strategy fallback"""
        recommendations = {}
        
        # Validate API keys
        if not self.marketstack_api_key:
            st.warning("⚠️ MarketStack API key missing - using synthetic data")
        
        if not self.polygon_api_key:
            st.warning("⚠️ Polygon.io API key missing - using synthetic data")
        
        for symbol in symbols:
            # st.info(f"🔍 Analyzing {symbol}...")
            
            try:
                # Get stock data with fallback
                stock_data = self._get_marketstack_data(symbol)
                if not stock_data:
                    st.warning(f"⚠️ Skipping {symbol} - no stock data available")
                    continue
                
                # Try to get options data, use synthetic if unavailable
                options_data = self._get_polygon_options_data(symbol)
                if not options_data:
                    st.info(f"📊 Using synthetic options data for {symbol}")
                    options_data = self._generate_synthetic_options_data(stock_data)
                
                # Analyze market conditions
                market_analysis = self._analyze_market_conditions(stock_data)
                
                # Select optimal strategy with fallback logic
                strategy_scores = self._select_strategy(market_analysis, stock_data)
                
                # Try strategies in order of preference with fallbacks
                successful_strategy = None
                trade_rec = None
                
                # Sort strategies by score and try each one
                sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
                
                for strategy_name, confidence in sorted_strategies:
                    try:
                        trade_rec = self._generate_trade_recommendation(
                            symbol, strategy_name, stock_data, options_data, 
                            market_analysis, portfolio_value
                        )
                        
                        if 'error' not in trade_rec:
                            successful_strategy = strategy_name
                            break
                        else:
                            st.warning(f"⚠️ {strategy_name} failed: {trade_rec['error']}")
                    except Exception as e:
                        st.warning(f"⚠️ {strategy_name} error: {str(e)}")
                        continue
                
                # If no complex strategy worked, try simple strategies
                if not successful_strategy:
                    simple_strategies = ['COVERED_CALL', 'CASH_SECURED_PUT']
                    for simple_strategy in simple_strategies:
                        try:
                            trade_rec = self._generate_trade_recommendation(
                                symbol, simple_strategy, stock_data, options_data, 
                                market_analysis, portfolio_value
                            )
                            if 'error' not in trade_rec:
                                successful_strategy = simple_strategy
                                break
                        except:
                            continue
                
                if successful_strategy and trade_rec and 'error' not in trade_rec:
                    recommendations[symbol] = {
                        'strategy': successful_strategy,
                        'market_analysis': market_analysis,
                        'trade_details': trade_rec,
                        'confidence': strategy_scores.get(successful_strategy, 5.0),
                        'data_sources': {
                            'stock_data': 'MarketStack' if stock_data.get('source') != 'synthetic' else 'Synthetic',
                            'options_data': 'Polygon.io' if options_data.get('source') != 'synthetic' else 'Synthetic'
                        }
                    }
                    st.success(f"✅ Analysis complete for {symbol} - {successful_strategy}")
                else:
                    st.warning(f"⚠️ Could not generate any viable strategy for {symbol}")
                
            except Exception as e:
                st.error(f"❌ Error analyzing {symbol}: {str(e)}")
                continue
        
        return recommendations
    
    def _generate_synthetic_options_data(self, stock_data: Dict) -> Dict:
        """Generate comprehensive synthetic options data when Polygon fails"""
        st.info(f"📊 Generating synthetic options data for {stock_data['symbol']}")
        
        current_price = stock_data['current_price']
        symbol = stock_data['symbol']
        realized_vol = stock_data.get('realized_vol', 0.25)
        
        # Generate expiration 30 days out
        exp_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Generate comprehensive strike prices (more strikes for better strategies)
        strikes = []
        for i in range(-10, 11):  # 21 strikes from -50% to +50%
            strike = current_price * (1 + (i * 0.05))  # 5% intervals
            strikes.append(max(1, round(strike, 2)))
        
        # Generate calls and puts with better pricing
        calls_data = []
        puts_data = []
        
        for strike in strikes:
            call_price = self._black_scholes_price(current_price, strike, exp_date, 'call', sigma=realized_vol)
            put_price = self._black_scholes_price(current_price, strike, exp_date, 'put', sigma=realized_vol)
            
            # Include all options with any meaningful value
            if call_price > 0.01:
                calls_data.append({
                    'strike': strike,
                    'lastPrice': call_price,
                    'bid': max(0.01, call_price * 0.95),
                    'ask': call_price * 1.05,
                    'volume': max(10, int(100 * np.random.exponential(0.3))),
                    'openInterest': max(50, int(500 * np.random.exponential(0.5))),
                    'impliedVolatility': realized_vol * (0.9 + 0.2 * np.random.random()),
                    'ticker': f"C{strike}"
                })
            
            if put_price > 0.01:
                puts_data.append({
                    'strike': strike,
                    'lastPrice': put_price,
                    'bid': max(0.01, put_price * 0.95),
                    'ask': put_price * 1.05,
                    'volume': max(10, int(100 * np.random.exponential(0.3))),
                    'openInterest': max(50, int(500 * np.random.exponential(0.5))),
                    'impliedVolatility': realized_vol * (0.9 + 0.2 * np.random.random()),
                    'ticker': f"P{strike}"
                })
        
        calls_df = pd.DataFrame(calls_data).sort_values('strike').reset_index(drop=True)
        puts_df = pd.DataFrame(puts_data).sort_values('strike').reset_index(drop=True)
        
        st.success(f"✅ Generated {len(calls_df)} calls and {len(puts_df)} puts for {symbol}")
        
        return {
            'expiration': exp_date,
            'calls': calls_df,
            'puts': puts_df,
            'days_to_expiry': 30,
            'underlying_price': current_price,
            'source': 'synthetic'
        }
    
    # [Previous API methods remain the same...]
    def _get_marketstack_data(self, symbol: str) -> Optional[Dict]:
        """Enhanced MarketStack data fetching with better error handling"""
        if not self.marketstack_api_key:
            return self._generate_synthetic_stock_data(symbol)
            
        try:
            # st.info(f"📈 Fetching stock data for {symbol} from MarketStack...")
            
            # Get latest price first
            latest_url = f"{self.marketstack_url}/eod/latest"
            latest_params = {
                'access_key': self.marketstack_api_key,
                'symbols': symbol,
                'limit': 1
            }
            
            latest_data = self._make_api_request(latest_url, latest_params, "MarketStack Latest")
            
            if not latest_data or 'data' not in latest_data or not latest_data['data']:
                st.warning(f"No current data for {symbol} from MarketStack")
                return self._generate_synthetic_stock_data(symbol)
            
            current_price = float(latest_data['data'][0]['close'])
            latest_volume = float(latest_data['data'][0].get('volume', 1000000))
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            hist_url = f"{self.marketstack_url}/eod"
            hist_params = {
                'access_key': self.marketstack_api_key,
                'symbols': symbol,
                'date_from': start_date.strftime('%Y-%m-%d'),
                'date_to': end_date.strftime('%Y-%m-%d'),
                'limit': 100
            }
            
            hist_data = self._make_api_request(hist_url, hist_params, "MarketStack Historical")
            
            if not hist_data or 'data' not in hist_data or not hist_data['data']:
                st.warning(f"Limited historical data for {symbol}")
                return self._generate_basic_stock_data(symbol, current_price, latest_volume)
            
            # Process historical data
            df = pd.DataFrame(hist_data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Convert to numeric
            for col in ['close', 'high', 'low', 'open', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(20, min_periods=5).mean()
            df['sma_50'] = df['close'].rolling(50, min_periods=10).mean()
            
            # Get most recent values with fallbacks
            sma_20 = df['sma_20'].dropna().iloc[-1] if len(df['sma_20'].dropna()) > 0 else current_price
            sma_50 = df['sma_50'].dropna().iloc[-1] if len(df['sma_50'].dropna()) > 0 else current_price
            
            # Calculate returns and volatility
            df['returns'] = df['close'].pct_change()
            returns = df['returns'].dropna()
            realized_vol = returns.std() * np.sqrt(252) if len(returns) > 5 else 0.25
            
            # RSI calculation
            rsi = self._calculate_rsi(df['close'])
            
            stock_data = {
                'symbol': symbol,
                'current_price': current_price,
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'realized_vol': realized_vol,
                'rsi': rsi,
                'returns': returns,
                'hist_data': df,
                'volume': latest_volume,
                'avg_volume': df['volume'].mean() if not df['volume'].isna().all() else latest_volume,
                'high_52w': df['high'].max() if not df['high'].isna().all() else current_price * 1.2,
                'low_52w': df['low'].min() if not df['low'].isna().all() else current_price * 0.8,
                'market_cap': None,
                'sector': 'Unknown',
                'beta': 1.0,
                'source': 'marketstack'
            }
            
            # st.success(f"✅ Stock data retrieved for {symbol}: ${current_price:.2f}")
            return stock_data
            
        except Exception as e:
            st.error(f"Error fetching MarketStack data for {symbol}: {str(e)}")
            return self._generate_synthetic_stock_data(symbol)
    
    def _generate_synthetic_stock_data(self, symbol: str, base_price: float = 100.0) -> Dict:
        """Generate synthetic stock data when API fails"""
        # st.info(f"📊 Generating synthetic stock data for {symbol}")
        
        # Use symbol hash to get consistent "random" data
        import hashlib
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % 1000
        np.random.seed(seed)
        
        current_price = base_price + np.random.normal(0, 20)
        current_price = max(10, current_price)  # Minimum $10
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'sma_20': current_price * (0.95 + np.random.random() * 0.1),
            'sma_50': current_price * (0.9 + np.random.random() * 0.2),
            'realized_vol': 0.15 + np.random.random() * 0.3,
            'rsi': 30 + np.random.random() * 40,
            'returns': pd.Series(np.random.normal(0.001, 0.02, 20)),
            'volume': 1000000 + np.random.randint(0, 5000000),
            'avg_volume': 1200000,
            'high_52w': current_price * (1.1 + np.random.random() * 0.3),
            'low_52w': current_price * (0.7 + np.random.random() * 0.2),
            'market_cap': None,
            'sector': 'Technology',
            'beta': 0.8 + np.random.random() * 0.8,
            'source': 'synthetic'
        }
    
    def _generate_basic_stock_data(self, symbol: str, current_price: float, volume: float) -> Dict:
        """Generate basic stock data with limited info"""
        return {
            'symbol': symbol,
            'current_price': current_price,
            'sma_20': current_price,
            'sma_50': current_price,
            'realized_vol': 0.25,
            'rsi': 50.0,
            'returns': pd.Series([0.01] * 20),
            'volume': volume,
            'avg_volume': volume,
            'high_52w': current_price * 1.2,
            'low_52w': current_price * 0.8,
            'market_cap': None,
            'sector': 'Unknown',
            'beta': 1.0,
            'source': 'basic'
        }
    
    def _get_polygon_options_data(self, symbol: str) -> Optional[Dict]:
        """Enhanced Polygon options data with better error handling and longer expiry search"""
        if not self.polygon_api_key:
            return None
            
        try:
            # st.info(f"🎯 Fetching options data for {symbol} from Polygon...")
            
            # Test API access first with a simple call
            test_url = f"{self.polygon_url}/v1/meta/symbols/{symbol}/company"
            test_params = {'apikey': self.polygon_api_key}
            
            test_result = self._make_api_request(test_url, test_params, "Polygon Test", timeout=5)
            if not test_result:
                st.warning(f"Cannot access Polygon API for {symbol}")
                return None
            
            # Get current stock price
            current_price = self._get_current_price_polygon(symbol)
            if not current_price:
                st.warning(f"Could not get current price for {symbol} from Polygon")
                return None
            
            # Get options contracts with retry logic and extended date range
            max_retries = 2
            contracts_data = None
            
            for attempt in range(max_retries):
                contracts_url = f"{self.polygon_url}/v3/reference/options/contracts"
                
                # Calculate date ranges - look for contracts 1 week to 2 months out
                start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                end_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
                
                contracts_params = {
                    'underlying_ticker': symbol,
                    'apikey': self.polygon_api_key,
                    'limit': 1000,  # Increased limit to get more contracts
                    'order': 'asc',
                    'sort': 'expiration_date',
                    'expiration_date.gte': start_date,  # Greater than or equal to tomorrow
                    'expiration_date.lte': end_date     # Less than or equal to 60 days out
                }
                
                contracts_data = self._make_api_request(contracts_url, contracts_params, f"Polygon Contracts (attempt {attempt+1})")
                
                if contracts_data and 'results' in contracts_data and contracts_data['results']:
                    print(f"Found {len(contracts_data['results'])} contracts")
                    print(contracts_data['results'][:5])  # Debugging output
                    break
                
                if attempt < max_retries - 1:
                    st.info(f"Retrying options data fetch for {symbol}...")
                    time.sleep(2)
            
            if not contracts_data or 'results' not in contracts_data or not contracts_data['results']:
                st.warning(f"No future options contracts found for {symbol}")
                return None
            
            # Process contracts more conservatively
            future_contracts = self._filter_options_contracts(contracts_data['results'], current_price)

            print(f"Filtered to {len(future_contracts)} contracts")
            print(future_contracts[:5])  # Debugging output
            
            if not future_contracts:
                st.warning(f"No suitable options contracts for {symbol}")
                return None
            
            # Create simplified options data
            return self._create_options_dataframe(future_contracts, current_price)
            
        except Exception as e:
            st.error(f"Error getting Polygon options data for {symbol}: {str(e)}")
            return None

    def _filter_options_contracts(self, contracts: List[Dict], current_price: float) -> List[Dict]:
        """Filter options contracts to reasonable strikes and dates with improved logic"""
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        filtered = []
        
        print(f"Filtering {len(contracts)} contracts. Today: {today}")
        
        for contract in contracts:
            try:
                exp_date_str = contract.get('expiration_date')
                if not exp_date_str:
                    continue
                
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                
                # Only include contracts that expire tomorrow or later (at least 1 day remaining)
                if exp_date <= today:
                    continue
                
                strike = contract.get('strike_price', 0)
                if strike <= 0:
                    continue
                
                # Filter to reasonable strikes (within 30% of current price for more options)
                strike_ratio = abs(strike - current_price) / current_price
                if strike_ratio <= 0.30:
                    # Include contracts expiring within 90 days for better selection
                    days_to_exp = (exp_date - today).days
                    if 1 <= days_to_exp <= 90:  # At least 1 day, max 90 days
                        contract['days_to_expiry'] = days_to_exp
                        filtered.append(contract)
                        
            except (ValueError, TypeError) as e:
                print(f"Error processing contract: {e}")
                continue
        
        print(f"After filtering: {len(filtered)} contracts remain")
        
        # If we have very few contracts, try with wider strike range
        if len(filtered) < 10:
            print("Too few contracts, expanding strike range...")
            for contract in contracts:
                try:
                    exp_date_str = contract.get('expiration_date')
                    if not exp_date_str:
                        continue
                    
                    exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                    if exp_date <= today:
                        continue
                    
                    strike = contract.get('strike_price', 0)
                    if strike <= 0:
                        continue
                    
                    # Wider strike range (50% of current price)
                    strike_ratio = abs(strike - current_price) / current_price
                    if strike_ratio <= 0.50:
                        days_to_exp = (exp_date - today).days
                        if 1 <= days_to_exp <= 90:
                            contract['days_to_expiry'] = days_to_exp
                            if contract not in filtered:  # Avoid duplicates
                                filtered.append(contract)
                            
                except (ValueError, TypeError):
                    continue
        
        print(f"Final filtered count: {len(filtered)} contracts")
        return filtered

    def _create_options_dataframe(self, contracts: List[Dict], current_price: float) -> Dict:
        """Create simplified options dataframe with better expiration selection"""
        # Group by expiration and find the best one
        exp_groups = {}
        for contract in contracts:
            exp_date = contract['expiration_date']
            if exp_date not in exp_groups:
                exp_groups[exp_date] = {'calls': [], 'puts': []}
            
            contract_type = contract.get('contract_type')
            if contract_type == 'call':
                exp_groups[exp_date]['calls'].append(contract)
            elif contract_type == 'put':
                exp_groups[exp_date]['puts'].append(contract)
        
        print(f"Found {len(exp_groups)} expiration dates")
        for exp_date, data in exp_groups.items():
            print(f"  {exp_date}: {len(data['calls'])} calls, {len(data['puts'])} puts")
        
        # Find best expiration (prefer ~30 days out with good option count)
        best_exp = None
        best_score = 0
        
        for exp_date in sorted(exp_groups.keys()):
            calls_count = len(exp_groups[exp_date]['calls'])
            puts_count = len(exp_groups[exp_date]['puts'])
            
            # Must have at least 3 of each type
            if calls_count >= 3 and puts_count >= 3:
                # Calculate days to expiry
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_datetime.date() - datetime.now().date()).days
                
                # Score based on days to expiry (prefer 20-40 days) and option count
                if 7 <= days_to_exp <= 60:
                    time_score = 100 - abs(30 - days_to_exp)  # Prefer 30 days
                    option_score = min(calls_count + puts_count, 50)  # Cap at 50
                    total_score = time_score + option_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_exp = exp_date
        
        # Fallback to any expiration with sufficient options
        if not best_exp:
            for exp_date in sorted(exp_groups.keys()):
                calls_count = len(exp_groups[exp_date]['calls'])
                puts_count = len(exp_groups[exp_date]['puts'])
                if calls_count >= 2 and puts_count >= 2:
                    best_exp = exp_date
                    break
        
        if not best_exp:
            print("No suitable expiration found")
            return None
        
        print(f"Selected expiration: {best_exp}")
        
        # Generate option prices using Black-Scholes for selected expiration
        calls_data = self._generate_option_prices(
            exp_groups[best_exp]['calls'][:15], current_price, best_exp, 'call'
        )
        puts_data = self._generate_option_prices(
            exp_groups[best_exp]['puts'][:15], current_price, best_exp, 'put'
        )
        
        calls_df = pd.DataFrame(calls_data).sort_values('strike') if calls_data else pd.DataFrame()
        puts_df = pd.DataFrame(puts_data).sort_values('strike') if puts_data else pd.DataFrame()
        
        exp_date_obj = datetime.strptime(best_exp, '%Y-%m-%d')
        days_to_expiry = max(1, (exp_date_obj.date() - datetime.now().date()).days)
        
        st.success(f"✅ Options data created: {len(calls_df)} calls, {len(puts_df)} puts, {days_to_expiry} days to expiry")
        
        return {
            'expiration': best_exp,
            'calls': calls_df,
            'puts': puts_df,
            'days_to_expiry': days_to_expiry,
            'underlying_price': current_price,
            'source': 'polygon'
        }
    
    def _generate_option_prices(self, contracts: List[Dict], underlying_price: float, 
                               exp_date: str, option_type: str) -> List[Dict]:
        """Generate option prices using Black-Scholes"""
        options_data = []
        
        for contract in contracts:
            strike = contract.get('strike_price', 0)
            if strike <= 0:
                continue
            
            # Calculate theoretical price
            price = self._black_scholes_price(underlying_price, strike, exp_date, option_type)
            
            if price > 0.01:  # Only include options with meaningful value
                options_data.append({
                    'strike': strike,
                    'lastPrice': price,
                    'bid': price * 0.95,
                    'ask': price * 1.05,
                    'volume': 100,
                    'openInterest': 500,
                    'impliedVolatility': 0.25,
                    'ticker': contract.get('ticker', f"{option_type.upper()[:1]}{strike}")
                })
        
        return options_data
    
    def _get_current_price_polygon(self, symbol: str) -> Optional[float]:
        """Get current stock price from Polygon with fallback"""
        try:
            url = f"{self.polygon_url}/v2/aggs/ticker/{symbol}/prev"
            params = {'apikey': self.polygon_api_key}
            
            data = self._make_api_request(url, params, "Polygon Price")
            
            if data and 'results' in data and data['results']:
                return float(data['results'][0]['c'])
            return None
        except Exception:
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI with error handling"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / (loss + 1e-8)
            rsi_series = 100 - (100 / (1 + rs))
            
            return float(rsi_series.dropna().iloc[-1]) if len(rsi_series.dropna()) > 0 else 50.0
        except Exception:
            return 50.0
    
    def _black_scholes_price(self, S: float, K: float, exp_date: str, option_type: str,
                            r: float = 0.05, sigma: float = 0.25) -> float:
        """Calculate option price using Black-Scholes formula"""
        try:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            T = max((exp_datetime - datetime.now()).days / 365.0, 0.01)
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(0.01, price)
        except Exception:
            return 0.01
    
    def _analyze_market_conditions(self, stock_data: Dict) -> Dict:
        """Market condition analysis"""
        current_price = stock_data['current_price']
        sma_20 = stock_data['sma_20']
        sma_50 = stock_data['sma_50']
        rsi = stock_data['rsi']
        realized_vol = stock_data['realized_vol']
        
        # Trend analysis
        if current_price > sma_20 > sma_50:
            trend = 'BULLISH'
            trend_strength = min((current_price - sma_50) / sma_50 * 100, 10)
        elif current_price < sma_20 < sma_50:
            trend = 'BEARISH'
            trend_strength = min((sma_50 - current_price) / sma_50 * 100, 10)
        else:
            trend = 'SIDEWAYS'
            trend_strength = 5.0
        
        # Volatility regime
        if realized_vol > 0.35:
            vol_regime = 'HIGH_VOL'
        elif realized_vol < 0.15:
            vol_regime = 'LOW_VOL'
        else:
            vol_regime = 'NORMAL_VOL'
        
        # Momentum
        if rsi > 70:
            momentum = 'OVERBOUGHT'
        elif rsi < 30:
            momentum = 'OVERSOLD'
        else:
            momentum = 'NEUTRAL'
        
        volume_ratio = stock_data['volume'] / stock_data.get('avg_volume', stock_data['volume'])
        if volume_ratio > 1.5:
            volume_trend = 'HIGH_VOLUME'
        elif volume_ratio < 0.5:
            volume_trend = 'LOW_VOLUME'
        else:
            volume_trend = 'NORMAL_VOLUME'
        
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'volatility_regime': vol_regime,
            'momentum': momentum,
            'volume_trend': volume_trend,
            'rsi': rsi,
            'realized_vol': realized_vol,
            'volume_ratio': volume_ratio,
            'price_vs_52w_high': (current_price / stock_data['high_52w']) * 100,
            'price_vs_52w_low': (current_price / stock_data['low_52w']) * 100
        }
    
    def _select_strategy(self, market_analysis: Dict, stock_data: Dict) -> Dict[str, float]:
        """Strategy selection logic with improved scoring"""
        scores = {}
        
        trend = market_analysis['trend']
        vol_regime = market_analysis['volatility_regime']
        momentum = market_analysis['momentum']
        trend_strength = market_analysis['trend_strength']
        volume_trend = market_analysis['volume_trend']
        
        # Score strategies based on market conditions
        if trend == 'BULLISH':
            scores['BULL_CALL_SPREAD'] = 8.0 + (trend_strength / 10)
            scores['COVERED_CALL'] = 7.0 if vol_regime == 'HIGH_VOL' else 5.0
            scores['CASH_SECURED_PUT'] = 7.5 if momentum == 'OVERSOLD' else 6.0
        
        if trend == 'BEARISH':
            scores['BEAR_PUT_SPREAD'] = 8.0 + (trend_strength / 10)
            scores['PROTECTIVE_PUT'] = 8.5
        
        if trend == 'SIDEWAYS':
            scores['IRON_CONDOR'] = 8.5 if vol_regime == 'HIGH_VOL' else 6.0
            scores['BUTTERFLY'] = 7.5
            scores['COVERED_CALL'] = 8.0 if vol_regime == 'HIGH_VOL' else 6.0
        
        if vol_regime == 'LOW_VOL':
            scores['STRADDLE'] = 8.0
            scores['STRANGLE'] = 8.0
        
        # Ensure all strategies have scores (fallback strategies get higher scores)
        fallback_strategies = {
            'COVERED_CALL': 6.0,
            'CASH_SECURED_PUT': 6.0,
            'BULL_CALL_SPREAD': 5.5,
            'BEAR_PUT_SPREAD': 5.5,
            'STRADDLE': 5.0,
            'STRANGLE': 5.0,
            'PROTECTIVE_PUT': 4.5,
            'COLLAR': 4.0,
            'IRON_CONDOR': 3.5,
            'BUTTERFLY': 3.0
        }
        
        for strategy, fallback_score in fallback_strategies.items():
            if strategy not in scores:
                scores[strategy] = fallback_score
        
        return scores
    
    def _generate_trade_recommendation(self, symbol: str, strategy: str,
                                     stock_data: Dict, options_data: Dict,
                                     market_analysis: Dict, portfolio_value: float) -> Dict:
        """Generate trade recommendation with enhanced error handling"""
        if strategy not in self.strategies:
            return {'error': f'Unknown strategy: {strategy}'}
        
        try:
            return self.strategies[strategy](
                symbol, stock_data, options_data, market_analysis, portfolio_value
            )
        except Exception as e:
            return {'error': f'Failed to generate {strategy}: {str(e)}'}
    
    def _bull_call_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                        market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Bull Call Spread with better validation"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            
            if calls.empty or len(calls) < 2:
                return {'error': 'Insufficient calls for bull call spread'}
            
            # Filter calls with meaningful prices
            viable_calls = calls[(calls['lastPrice'] > 0.05) & (calls['lastPrice'] < current_price * 0.5)].copy()
            if len(viable_calls) < 2:
                return {'error': 'No viable call options found'}
            
            viable_calls = viable_calls.sort_values('strike').reset_index(drop=True)
            viable_calls['moneyness'] = viable_calls['strike'] / current_price
            
            # Buy call close to ATM (90-105% of current price)
            atm_calls = viable_calls[(viable_calls['moneyness'] >= 0.90) & (viable_calls['moneyness'] <= 1.05)]
            if atm_calls.empty:
                # Fallback to closest to ATM
                viable_calls['distance'] = abs(viable_calls['strike'] - current_price)
                buy_call = viable_calls.loc[viable_calls['distance'].idxmin()]
            else:
                buy_call = atm_calls.iloc[0]
            
            # Sell call OTM (at least 3% higher strike)
            min_sell_strike = buy_call['strike'] * 1.03
            otm_calls = viable_calls[viable_calls['strike'] >= min_sell_strike]
            if otm_calls.empty:
                return {'error': 'No suitable OTM calls found for spread'}
            
            sell_call = otm_calls.iloc[0]
            
            # Validate spread economics
            net_debit = buy_call['lastPrice'] - sell_call['lastPrice']
            max_profit = (sell_call['strike'] - buy_call['strike']) - net_debit
            max_loss = net_debit
            breakeven = buy_call['strike'] + net_debit
            
            if net_debit <= 0:
                return {'error': 'Spread results in net credit - not optimal for bull call'}
            if max_profit <= 0:
                return {'error': 'No profit potential in spread'}
            if max_loss > current_price * 0.1:  # Sanity check
                return {'error': 'Risk too high relative to stock price'}
            
            # Position sizing
            risk_amount = portfolio_value * self.risk_tolerance
            vol_adjustment = 0.8 if stock_data['realized_vol'] > 0.3 else 1.2 if stock_data['realized_vol'] < 0.15 else 1.0
            adjusted_risk = risk_amount * vol_adjustment
            contracts = max(1, min(int(adjusted_risk / (max_loss * 100)), 10))
            
            # Calculate profit potential percentage
            profit_potential = (max_profit / max_loss) * 100 if max_loss > 0 else 0
            
            return {
                'strategy_name': 'Bull Call Spread',
                'legs': [
                    {
                        'action': 'BUY',
                        'option_type': 'CALL',
                        'strike': buy_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': buy_call['lastPrice'],
                        'contracts': contracts
                    },
                    {
                        'action': 'SELL',
                        'option_type': 'CALL',
                        'strike': sell_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': sell_call['lastPrice'],
                        'contracts': contracts
                    }
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_profit': round(max_profit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'breakeven': round(breakeven, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'rationale': f"Bullish strategy for {symbol}. Max profit: ${round(max_profit * contracts * 100, 2)} ({round(profit_potential, 1)}% return) if {symbol} rises above ${sell_call['strike']:.2f} by expiration. Breakeven at ${breakeven:.2f}"
            }
        except Exception as e:
            return {'error': f'Bull call spread calculation failed: {str(e)}'}
    
    def _collar(self, symbol: str, stock_data: Dict, options_data: Dict,
                market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Collar strategy"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            puts = options_data['puts']
            
            # Find protective puts
            protective_puts = puts[(puts['strike'] < current_price * 0.97) & (puts['lastPrice'] > 0.02)]
            if protective_puts.empty:
                return {'error': 'No suitable puts for collar'}
            
            # Find calls to sell
            calls_to_sell = calls[(calls['strike'] > current_price * 1.03) & (calls['lastPrice'] > 0.02)]
            if calls_to_sell.empty:
                return {'error': 'No suitable calls for collar'}
            
            # Select reasonable strikes
            put_to_buy = protective_puts.iloc[-1]  # Highest strike protective put
            call_to_sell = calls_to_sell.iloc[0]   # Lowest strike call to sell
            
            net_premium = call_to_sell['lastPrice'] - put_to_buy['lastPrice']
            shares_owned = 100
            
            # Calculate collar metrics
            max_profit = (call_to_sell['strike'] - current_price) * shares_owned + (net_premium * 100)
            max_loss = (current_price - put_to_buy['strike']) * shares_owned - (net_premium * 100)
            protection_level = (1 - put_to_buy['strike']/current_price) * 100
            
            return {
                'strategy_name': 'Collar',
                'legs': [
                    {'action': 'OWN', 'instrument': 'STOCK', 'quantity': shares_owned, 'price': current_price},
                    {'action': 'BUY', 'option_type': 'PUT', 'strike': put_to_buy['strike'],
                     'expiration': options_data['expiration'], 'price': put_to_buy['lastPrice'], 'contracts': 1},
                    {'action': 'SELL', 'option_type': 'CALL', 'strike': call_to_sell['strike'],
                     'expiration': options_data['expiration'], 'price': call_to_sell['lastPrice'], 'contracts': 1}
                ],
                'net_premium': round(net_premium * 100, 2),
                'protected_floor': put_to_buy['strike'],
                'upside_cap': call_to_sell['strike'],
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'cost_basis_protection': round(protection_level, 2),
                'rationale': f"Protected position for {symbol}. Range: ${put_to_buy['strike']:.2f} - ${call_to_sell['strike']:.2f}"
            }
        except Exception as e:
            return {'error': f'Collar calculation failed: {str(e)}'}
    
    def _butterfly_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                         market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Butterfly Spread with robust strike selection"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            
            if calls.empty or len(calls) < 3:
                return {'error': 'Insufficient calls for butterfly spread'}
            
            # Filter viable calls
            viable_calls = calls[(calls['lastPrice'] > 0.02) & (calls['lastPrice'] < current_price * 0.3)].copy()
            if len(viable_calls) < 3:
                return {'error': 'Need at least 3 viable strikes for butterfly'}
            
            viable_calls = viable_calls.sort_values('strike').reset_index(drop=True)
            n_strikes = len(viable_calls)
            
            # Select strikes with equal spacing if possible
            # Find ATM or close to ATM for center strike
            viable_calls['distance'] = abs(viable_calls['strike'] - current_price)
            center_idx = viable_calls['distance'].idxmin()
            
            # Try to find equidistant strikes
            center_strike = viable_calls.loc[center_idx, 'strike']
            
            # Look for strikes above and below center
            lower_calls = viable_calls[viable_calls['strike'] < center_strike]
            higher_calls = viable_calls[viable_calls['strike'] > center_strike]
            
            if lower_calls.empty or higher_calls.empty:
                return {'error': 'Cannot create symmetric butterfly'}
            
            # Select strikes
            low_call = lower_calls.iloc[-1]  # Highest strike below center
            mid_call = viable_calls.loc[center_idx]
            high_call = higher_calls.iloc[0]  # Lowest strike above center
            
            # Calculate butterfly economics
            net_debit = low_call['lastPrice'] + high_call['lastPrice'] - (2 * mid_call['lastPrice'])
            
            if net_debit <= 0:
                return {'error': 'Butterfly results in net credit - not optimal'}
            
            max_profit = (mid_call['strike'] - low_call['strike']) - net_debit
            max_loss = net_debit
            
            if max_profit <= 0:
                return {'error': 'No profit potential in butterfly'}
            
            # Position sizing
            risk_amount = portfolio_value * self.risk_tolerance
            contracts = max(1, min(int(risk_amount / (max_loss * 100)), 5))
            
            # Breakeven points
            lower_breakeven = low_call['strike'] + net_debit
            upper_breakeven = high_call['strike'] - net_debit
            
            return {
                'strategy_name': 'Butterfly Spread',
                'legs': [
                    {'action': 'BUY', 'option_type': 'CALL', 'strike': low_call['strike'],
                     'expiration': options_data['expiration'], 'price': low_call['lastPrice'], 'contracts': contracts},
                    {'action': 'SELL', 'option_type': 'CALL', 'strike': mid_call['strike'],
                     'expiration': options_data['expiration'], 'price': mid_call['lastPrice'], 'contracts': contracts * 2},
                    {'action': 'BUY', 'option_type': 'CALL', 'strike': high_call['strike'],
                     'expiration': options_data['expiration'], 'price': high_call['lastPrice'], 'contracts': contracts}
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_profit': round(max_profit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'optimal_price': mid_call['strike'],
                'days_to_expiry': options_data['days_to_expiry'],
                'risk_reward_ratio': round(max_profit / max_loss, 2),
                'profit_range': (round(lower_breakeven, 2), round(upper_breakeven, 2)),
                'rationale': f"Neutral strategy for {symbol}. Max profit if closes at ${mid_call['strike']:.2f}"
            }
        except Exception as e:
            return {'error': f'Butterfly spread calculation failed: {str(e)}'}
        
    def _bear_put_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                        market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Bear Put Spread with better validation"""
        try:
            current_price = stock_data['current_price']
            puts = options_data['puts']
            
            if puts.empty or len(puts) < 2:
                return {'error': 'Insufficient puts for bear put spread'}
            
            # Filter puts with meaningful prices
            viable_puts = puts[(puts['lastPrice'] > 0.05) & (puts['lastPrice'] < current_price * 0.5)].copy()
            if len(viable_puts) < 2:
                return {'error': 'No viable put options found'}
            
            viable_puts = viable_puts.sort_values('strike', ascending=False).reset_index(drop=True)
            viable_puts['moneyness'] = viable_puts['strike'] / current_price
            
            # Buy put close to ATM (95-110% of current price)
            atm_puts = viable_puts[(viable_puts['moneyness'] >= 0.95) & (viable_puts['moneyness'] <= 1.10)]
            if atm_puts.empty:
                # Fallback to closest to ATM
                viable_puts['distance'] = abs(viable_puts['strike'] - current_price)
                buy_put = viable_puts.loc[viable_puts['distance'].idxmin()]
            else:
                buy_put = atm_puts.iloc[0]
            
            # Sell put OTM (at least 3% lower strike)
            max_sell_strike = buy_put['strike'] * 0.97
            otm_puts = viable_puts[viable_puts['strike'] <= max_sell_strike]
            if otm_puts.empty:
                return {'error': 'No suitable OTM puts found for spread'}
            
            sell_put = otm_puts.iloc[0]
            
            # Validate spread economics
            net_debit = buy_put['lastPrice'] - sell_put['lastPrice']
            max_profit = (buy_put['strike'] - sell_put['strike']) - net_debit
            max_loss = net_debit
            breakeven = buy_put['strike'] - net_debit
            
            if net_debit <= 0:
                return {'error': 'Spread results in net credit - not optimal for bear put'}
            if max_profit <= 0:
                return {'error': 'No profit potential in spread'}
            if max_loss > current_price * 0.1:
                return {'error': 'Risk too high relative to stock price'}
            
            # Position sizing
            risk_amount = portfolio_value * self.risk_tolerance
            vol_adjustment = 0.8 if stock_data['realized_vol'] > 0.3 else 1.2 if stock_data['realized_vol'] < 0.15 else 1.0
            adjusted_risk = risk_amount * vol_adjustment
            contracts = max(1, min(int(adjusted_risk / (max_loss * 100)), 10))
            
            return {
                'strategy_name': 'Bear Put Spread',
                'legs': [
                    {
                        'action': 'BUY',
                        'option_type': 'PUT',
                        'strike': buy_put['strike'],
                        'expiration': options_data['expiration'],
                        'price': buy_put['lastPrice'],
                        'contracts': contracts
                    },
                    {
                        'action': 'SELL',
                        'option_type': 'PUT',
                        'strike': sell_put['strike'],
                        'expiration': options_data['expiration'],
                        'price': sell_put['lastPrice'],
                        'contracts': contracts
                    }
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_profit': round(max_profit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'breakeven': round(breakeven, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'risk_reward_ratio': round(max_profit / max_loss, 2),
                'rationale': f"Bearish strategy for {symbol}. Target: ${sell_put['strike']:.2f}. Profit if below ${breakeven:.2f}"
            }
        except Exception as e:
            return {'error': f'Bear put spread calculation failed: {str(e)}'}
    
    def _iron_condor(self, symbol: str, stock_data: Dict, options_data: Dict,
                    market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Iron Condor with comprehensive validation"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            puts = options_data['puts']
            
            if calls.empty or puts.empty or len(calls) < 2 or len(puts) < 2:
                return {'error': 'Insufficient options for iron condor'}
            
            # Filter viable options
            viable_calls = calls[(calls['lastPrice'] > 0.02) & (calls['strike'] > current_price * 1.02)].copy()
            viable_puts = puts[(puts['lastPrice'] > 0.02) & (puts['strike'] < current_price * 0.98)].copy()

            if len(viable_calls) < 2 or len(viable_puts) < 2:
                return {'error': 'Not enough viable OTM options for condor'}
            
            viable_calls = viable_calls.sort_values('strike').reset_index(drop=True)
            viable_puts = viable_puts.sort_values('strike', ascending=False).reset_index(drop=True)
            
            # Select short strikes (sell first OTM options)
            short_call = viable_calls.iloc[0]  # First OTM call
            short_put = viable_puts.iloc[0]   # First OTM put
            
            # Select long strikes (buy further OTM for protection)
            long_call_candidates = viable_calls[viable_calls['strike'] > short_call['strike']]
            long_put_candidates = viable_puts[viable_puts['strike'] < short_put['strike']]
            
            if long_call_candidates.empty or long_put_candidates.empty:
                return {'error': 'No suitable long options for condor protection'}
            
            long_call = long_call_candidates.iloc[0]
            long_put = long_put_candidates.iloc[0]
            
            # Calculate economics
            premium_collected = short_call['lastPrice'] + short_put['lastPrice']
            premium_paid = long_call['lastPrice'] + long_put['lastPrice']
            net_credit = premium_collected - premium_paid
            
            if net_credit <= 0:
                return {'error': 'Iron condor results in net debit'}
            
            call_width = long_call['strike'] - short_call['strike']
            put_width = short_put['strike'] - long_put['strike']
            max_loss = min(call_width, put_width) - net_credit
            max_profit = net_credit
            
            if max_loss <= 0:
                return {'error': 'Invalid condor structure'}
            
            # Profit zone
            lower_breakeven = short_put['strike'] + net_credit
            upper_breakeven = short_call['strike'] - net_credit
            
            # Position sizing
            risk_amount = portfolio_value * self.risk_tolerance
            risk_adjustment = 1.2 if market_analysis['trend'] == 'SIDEWAYS' else 0.8
            adjusted_risk = risk_amount * risk_adjustment
            contracts = max(1, min(int(adjusted_risk / (abs(max_loss) * 100)), 5))
            
            return {
                'strategy_name': 'Iron Condor',
                'legs': [
                    {'action': 'SELL', 'option_type': 'CALL', 'strike': short_call['strike'],
                     'expiration': options_data['expiration'], 'price': short_call['lastPrice'], 'contracts': contracts},
                    {'action': 'BUY', 'option_type': 'CALL', 'strike': long_call['strike'],
                     'expiration': options_data['expiration'], 'price': long_call['lastPrice'], 'contracts': contracts},
                    {'action': 'SELL', 'option_type': 'PUT', 'strike': short_put['strike'],
                     'expiration': options_data['expiration'], 'price': short_put['lastPrice'], 'contracts': contracts},
                    {'action': 'BUY', 'option_type': 'PUT', 'strike': long_put['strike'],
                     'expiration': options_data['expiration'], 'price': long_put['lastPrice'], 'contracts': contracts}
                ],
                'net_credit': round(net_credit * contracts * 100, 2),
                'max_profit': round(max_profit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'profit_range': (round(lower_breakeven, 2), round(upper_breakeven, 2)),
                'days_to_expiry': options_data['days_to_expiry'],
                'rationale': f"Neutral strategy. Profit if {symbol} stays between ${lower_breakeven:.2f} and ${upper_breakeven:.2f}"
            }
        except Exception as e:
            return {'error': f'Iron condor calculation failed: {str(e)}'}
    
    def _long_straddle(self, symbol: str, stock_data: Dict, options_data: Dict,
                      market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Long Straddle with better ATM selection"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            puts = options_data['puts']
            
            if calls.empty or puts.empty:
                return {'error': 'No options available for straddle'}
            
            # Find closest to ATM options
            calls_copy = calls.copy()
            puts_copy = puts.copy()
            
            calls_copy['distance'] = abs(calls_copy['strike'] - current_price)
            puts_copy['distance'] = abs(puts_copy['strike'] - current_price)
            
            # Filter out very cheap options
            calls_filtered = calls_copy[calls_copy['lastPrice'] > 0.05]
            puts_filtered = puts_copy[puts_copy['lastPrice'] > 0.05]
            
            if calls_filtered.empty or puts_filtered.empty:
                return {'error': 'No viable options for straddle'}
            
            atm_call = calls_filtered.loc[calls_filtered['distance'].idxmin()]
            atm_put = puts_filtered.loc[puts_filtered['distance'].idxmin()]
            
            # Calculate metrics
            net_debit = atm_call['lastPrice'] + atm_put['lastPrice']
            breakeven_up = atm_call['strike'] + net_debit
            breakeven_down = atm_put['strike'] - net_debit
            
            if net_debit <= 0:
                return {'error': 'Invalid straddle pricing'}
            
            # Position sizing
            risk_amount = portfolio_value * self.risk_tolerance
            vol_adjustment = 1.3 if stock_data['realized_vol'] < 0.2 else 0.7 if stock_data['realized_vol'] > 0.4 else 1.0
            adjusted_risk = risk_amount * vol_adjustment
            contracts = max(1, min(int(adjusted_risk / (net_debit * 100)), 10))
            
            return {
                'strategy_name': 'Long Straddle',
                'legs': [
                    {'action': 'BUY', 'option_type': 'CALL', 'strike': atm_call['strike'],
                     'expiration': options_data['expiration'], 'price': atm_call['lastPrice'], 'contracts': contracts},
                    {'action': 'BUY', 'option_type': 'PUT', 'strike': atm_put['strike'],
                     'expiration': options_data['expiration'], 'price': atm_put['lastPrice'], 'contracts': contracts}
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_loss': round(net_debit * contracts * 100, 2),
                'breakeven_up': round(breakeven_up, 2),
                'breakeven_down': round(breakeven_down, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'rationale': f"Volatility play for {symbol}. Profit if moves beyond ${breakeven_down:.2f} or ${breakeven_up:.2f}"
            }
        except Exception as e:
            return {'error': f'Long straddle calculation failed: {str(e)}'}
    
    def _long_strangle(self, symbol: str, stock_data: Dict, options_data: Dict,
                      market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Long Strangle with better OTM selection"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            puts = options_data['puts']
            
            # Find OTM options with reasonable distance
            otm_calls = calls[(calls['strike'] > current_price * 1.02) & (calls['lastPrice'] > 0.05)]
            otm_puts = puts[(puts['strike'] < current_price * 0.98) & (puts['lastPrice'] > 0.05)]
            
            if otm_calls.empty or otm_puts.empty:
                return {'error': 'No suitable OTM options for strangle'}
            
            # Select first viable OTM options
            call_option = otm_calls.iloc[0]
            put_option = otm_puts.iloc[-1]  # Highest strike put below current price
            
            net_debit = call_option['lastPrice'] + put_option['lastPrice']
            breakeven_up = call_option['strike'] + net_debit
            breakeven_down = put_option['strike'] - net_debit
            
            if net_debit <= 0:
                return {'error': 'Invalid strangle pricing'}
            
            # Position sizing
            risk_amount = portfolio_value * self.risk_tolerance
            contracts = max(1, min(int(risk_amount / (net_debit * 100)), 10))
            
            return {
                'strategy_name': 'Long Strangle',
                'legs': [
                    {'action': 'BUY', 'option_type': 'CALL', 'strike': call_option['strike'],
                     'expiration': options_data['expiration'], 'price': call_option['lastPrice'], 'contracts': contracts},
                    {'action': 'BUY', 'option_type': 'PUT', 'strike': put_option['strike'],
                     'expiration': options_data['expiration'], 'price': put_option['lastPrice'], 'contracts': contracts}
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_loss': round(net_debit * contracts * 100, 2),
                'breakeven_up': round(breakeven_up, 2),
                'breakeven_down': round(breakeven_down, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'rationale': f"Lower cost volatility play for {symbol}. Profit if moves beyond ${breakeven_down:.2f} or ${breakeven_up:.2f}"
            }
        except Exception as e:
            return {'error': f'Long strangle calculation failed: {str(e)}'}
    
    def _covered_call(self, symbol: str, stock_data: Dict, options_data: Dict,
                     market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Covered Call - Most reliable strategy"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            
            # Find OTM calls to sell
            otm_calls = calls[(calls['strike'] > current_price * 1.01) & (calls['lastPrice'] > 0.05)]
            if otm_calls.empty:
                # Fallback to any calls above current price
                otm_calls = calls[calls['strike'] > current_price]
                if otm_calls.empty:
                    return {'error': 'No suitable calls for covered call'}
            
            call_to_sell = otm_calls.iloc[0]
            shares_needed = 100
            cost_basis = current_price * shares_needed
            premium_received = call_to_sell['lastPrice'] * 100
            
            # Validate economics
            if premium_received <= 0:
                return {'error': 'No premium available from call'}
            
            # Calculate metrics
            max_profit_if_called = (call_to_sell['strike'] * shares_needed) - cost_basis + premium_received
            breakeven = current_price - call_to_sell['lastPrice']
            yield_if_called = ((call_to_sell['strike'] - current_price + call_to_sell['lastPrice']) / current_price) * 100
            
            return {
                'strategy_name': 'Covered Call',
                'legs': [
                    {'action': 'BUY', 'instrument': 'STOCK', 'quantity': shares_needed, 'price': current_price},
                    {'action': 'SELL', 'option_type': 'CALL', 'strike': call_to_sell['strike'],
                     'expiration': options_data['expiration'], 'price': call_to_sell['lastPrice'], 'contracts': 1}
                ],
                'initial_cost': round(cost_basis - premium_received, 2),
                'premium_received': round(premium_received, 2),
                'max_profit': round(max_profit_if_called, 2),
                'breakeven': round(breakeven, 2),
                'assignment_risk': call_to_sell['strike'],
                'days_to_expiry': options_data['days_to_expiry'],
                'yield_if_called': round(yield_if_called, 2),
                'rationale': f"Income strategy for {symbol}. {round(premium_received/cost_basis*100, 2)}% yield. Called away if > ${call_to_sell['strike']:.2f}"
            }
        except Exception as e:
            return {'error': f'Covered call calculation failed: {str(e)}'}
    
    def _cash_secured_put(self, symbol: str, stock_data: Dict, options_data: Dict,
                         market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Cash Secured Put - Another reliable strategy"""
        try:
            current_price = stock_data['current_price']
            puts = options_data['puts']
            
            # Find OTM puts to sell
            otm_puts = puts[(puts['strike'] < current_price * 0.99) & (puts['lastPrice'] > 0.05)]
            if otm_puts.empty:
                # Fallback to any puts below current price
                otm_puts = puts[puts['strike'] < current_price]
                if otm_puts.empty:
                    return {'error': 'No suitable puts for cash secured put'}
            
            # Select put closest to desired target (5% below current price)
            target_strike = current_price * 0.95
            otm_puts['distance'] = abs(otm_puts['strike'] - target_strike)
            put_to_sell = otm_puts.loc[otm_puts['distance'].idxmin()]
            
            cash_required = put_to_sell['strike'] * 100
            premium_received = put_to_sell['lastPrice'] * 100
            
            # Validate economics
            if premium_received <= 0:
                return {'error': 'No premium available from put'}
            
            # Calculate metrics
            net_cost_if_assigned = cash_required - premium_received
            breakeven = put_to_sell['strike'] - put_to_sell['lastPrice']
            annual_yield = (premium_received / cash_required) * (365 / options_data['days_to_expiry']) * 100
            
            return {
                'strategy_name': 'Cash Secured Put',
                'legs': [
                    {'action': 'SELL', 'option_type': 'PUT', 'strike': put_to_sell['strike'],
                     'expiration': options_data['expiration'], 'price': put_to_sell['lastPrice'], 'contracts': 1}
                ],
                'cash_required': round(cash_required, 2),
                'premium_received': round(premium_received, 2),
                'net_cost_if_assigned': round(net_cost_if_assigned, 2),
                'breakeven': round(breakeven, 2),
                'annual_yield': round(annual_yield, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'rationale': f"Income strategy for {symbol}. {round(premium_received/cash_required*100, 2)}% yield. Buy obligation at ${put_to_sell['strike']:.2f}"
            }
        except Exception as e:
            return {'error': f'Cash secured put calculation failed: {str(e)}'}
    
    def _protective_put(self, symbol: str, stock_data: Dict, options_data: Dict,
                    market_analysis: Dict, portfolio_value: float) -> Dict:
        """Fixed Protective Put"""
        try:
            current_price = stock_data['current_price']
            puts = options_data['puts']
            
            # Find OTM puts for protection
            protective_puts = puts[(puts['strike'] < current_price * 0.98) & (puts['lastPrice'] > 0.02)]
            if protective_puts.empty:
                return {'error': 'No suitable puts for protection'}
            
            # Select put that provides reasonable protection (around 90-95% of current price)
            target_protection = current_price * 0.92
            protective_puts['distance'] = abs(protective_puts['strike'] - target_protection)
            put_to_buy = protective_puts.loc[protective_puts['distance'].idxmin()]
            
            shares_owned = 100
            insurance_cost = put_to_buy['lastPrice'] * 100
            
            # Calculate metrics
            protected_value = put_to_buy['strike'] * shares_owned
            max_loss = (current_price - put_to_buy['strike']) * shares_owned + insurance_cost
            insurance_percentage = (insurance_cost / (current_price * shares_owned)) * 100
            
            # Calculate upside potential (unlimited, but show cost impact)
            breakeven_price = current_price + put_to_buy['lastPrice']  # Stock needs to rise by premium amount
            protection_level = (put_to_buy['strike'] / current_price) * 100
            
            # Position sizing based on portfolio
            position_size = min(portfolio_value * 0.1, current_price * shares_owned)  # Max 10% of portfolio
            actual_shares = int(position_size / current_price)
            actual_contracts = max(1, actual_shares // 100)  # At least 1 contract
            actual_shares = actual_contracts * 100  # Adjust to match contracts
            
            # Recalculate with actual position size
            actual_insurance_cost = put_to_buy['lastPrice'] * 100 * actual_contracts
            actual_protected_value = put_to_buy['strike'] * actual_shares
            actual_max_loss = (current_price - put_to_buy['strike']) * actual_shares + actual_insurance_cost
            actual_insurance_percentage = (actual_insurance_cost / (current_price * actual_shares)) * 100
            
            return {
                'strategy_name': 'Protective Put',
                'legs': [
                    {
                        'action': 'OWN',
                        'instrument': 'STOCK',
                        'quantity': actual_shares,
                        'price': current_price
                    },
                    {
                        'action': 'BUY',
                        'option_type': 'PUT',
                        'strike': put_to_buy['strike'],
                        'expiration': options_data['expiration'],
                        'price': put_to_buy['lastPrice'],
                        'contracts': actual_contracts
                    }
                ],
                'insurance_cost': round(actual_insurance_cost, 2),
                'protected_value': round(actual_protected_value, 2),
                'max_loss': round(actual_max_loss, 2),
                'insurance_percentage': round(actual_insurance_percentage, 2),
                'protection_level': round(protection_level, 1),
                'breakeven_price': round(breakeven_price, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'rationale': f"Downside protection for {symbol}. Protected below ${put_to_buy['strike']:.2f} ({protection_level:.1}% of current price). Insurance cost: {actual_insurance_percentage:.1f}% of position value. Stock needs to reach ${breakeven_price:.2f} to overcome premium cost."
            }
        except Exception as e:
            return {'error': f'Protective put calculation failed: {str(e)}'}
    
    def display_recommendations(self, recommendations: Dict) -> None:
        """Enhanced display with better error handling and data source indicators"""
        st.title("🎯 Enhanced Options Strategy Recommendations")
        st.markdown("*Powered by MarketStack + Polygon.io with fallback synthetic data*")
        
        if not recommendations:
            st.warning("⚠️ No recommendations available. Please check your symbols and API keys.")
            return
        
        # Summary metrics
        st.subheader("📊 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Symbols Analyzed", len(recommendations))
        
        with col2:
            strategies = [rec['strategy'] for rec in recommendations.values()]
            most_common = max(set(strategies), key=strategies.count) if strategies else "N/A"
            st.metric("Most Common Strategy", most_common)
        
        with col3:
            avg_confidence = np.mean([rec['confidence'] for rec in recommendations.values()])
            st.metric("Avg Confidence", f"{avg_confidence:.1f}/10")
        
        with col4:
            synthetic_count = sum(1 for rec in recommendations.values() 
                                if 'synthetic' in rec['data_sources']['stock_data'].lower() 
                                or 'synthetic' in rec['data_sources']['options_data'].lower())
            st.metric("Using Synthetic Data", f"{synthetic_count}/{len(recommendations)}")
        
        # Individual recommendations
        for symbol, rec in recommendations.items():
            data_source_color = "🟢" if rec['data_sources']['stock_data'] == 'MarketStack' else "🟡"
            options_source_color = "🟢" if rec['data_sources']['options_data'] == 'Polygon.io' else "🟡"
            
            with st.expander(f"{data_source_color}{options_source_color} {symbol} - {rec['strategy']} (Confidence: {rec['confidence']:.1f}/10)", expanded=True):
                trade = rec['trade_details']
                
                # Data source indicators
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📈 Stock Data: {rec['data_sources']['stock_data']}")
                with col2:
                    st.info(f"🎯 Options Data: {rec['data_sources']['options_data']}")
                
                # Market Analysis
                st.subheader("📊 Market Analysis")
                analysis = rec['market_analysis']
                
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    trend_color = "🟢" if analysis['trend'] == 'BULLISH' else "🔴" if analysis['trend'] == 'BEARISH' else "🟡"
                    st.metric("Trend", f"{trend_color} {analysis['trend']}")
                    st.metric("RSI", f"{analysis['rsi']:.1f}")
                
                with met_col2:
                    vol_color = "🔴" if analysis['volatility_regime'] == 'HIGH_VOL' else "🟢" if analysis['volatility_regime'] == 'LOW_VOL' else "🟡"
                    st.metric("Volatility", f"{vol_color} {analysis['volatility_regime']}")
                    st.metric("Realized Vol", f"{analysis['realized_vol']:.1%}")
                
                with met_col3:
                    momentum_color = "🔴" if analysis['momentum'] == 'OVERBOUGHT' else "🟢" if analysis['momentum'] == 'OVERSOLD' else "🟡"
                    st.metric("Momentum", f"{momentum_color} {analysis['momentum']}")
                    st.metric("Volume", analysis['volume_trend'])
                
                with met_col4:
                    st.metric("52W High", f"{analysis['price_vs_52w_high']:.1f}%")
                    st.metric("52W Low", f"{analysis['price_vs_52w_low']:.1f}%")
                
                # Strategy Details
                st.subheader(f"💡 Strategy: {trade['strategy_name']}")
                
                # Show rationale prominently
                if 'rationale' in trade:
                    st.success(f"**Rationale:** {trade['rationale']}")
                
                # Trade legs in a nice table
                if 'legs' in trade:
                    st.subheader("📋 Trade Details")
                    legs_df = pd.DataFrame(trade['legs'])
                    
                    # Format the dataframe for better display
                    if not legs_df.empty:
                        # Add formatting
                        for col in ['price', 'strike']:
                            if col in legs_df.columns:
                                legs_df[col] = legs_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                        
                        st.dataframe(legs_df, use_container_width=True, hide_index=True)
                
                # Risk/Reward Analysis
                st.subheader("⚖️ Risk/Reward Analysis")
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    if 'max_profit' in trade:
                        profit_color = "🟢" if trade['max_profit'] > 0 else "🔴"
                        st.metric("Max Profit", f"{profit_color} ${trade['max_profit']:,.2f}")
                    
                    if 'max_loss' in trade:
                        loss_color = "🔴" if trade['max_loss'] > 0 else "🟢"
                        st.metric("Max Loss", f"{loss_color} ${trade['max_loss']:,.2f}")
                    
                    if 'net_debit' in trade:
                        st.metric("Net Debit", f"💰 ${trade['net_debit']:,.2f}")
                    elif 'net_credit' in trade:
                        st.metric("Net Credit", f"💰 ${trade['net_credit']:,.2f}")
                
                with risk_col2:
                    if 'breakeven' in trade:
                        st.metric("Breakeven", f"🎯 ${trade['breakeven']:.2f}")
                    elif 'breakeven_up' in trade and 'breakeven_down' in trade:
                        st.metric("Breakeven Up", f"📈 ${trade['breakeven_up']:.2f}")
                        st.metric("Breakeven Down", f"📉 ${trade['breakeven_down']:.2f}")
                    
                    if 'risk_reward_ratio' in trade:
                        ratio_color = "🟢" if trade['risk_reward_ratio'] > 1.5 else "🟡" if trade['risk_reward_ratio'] > 1.0 else "🔴"
                        st.metric("Risk/Reward", f"{ratio_color} {trade['risk_reward_ratio']:.2f}")
                    
                    if 'days_to_expiry' in trade:
                        days_color = "🔴" if trade['days_to_expiry'] < 7 else "🟡" if trade['days_to_expiry'] < 30 else "🟢"
                        st.metric("Days to Expiry", f"{days_color} {trade['days_to_expiry']}")
                
                # Strategy-specific metrics
                if 'profit_range' in trade:
                    st.info(f"🎯 **Profit Range:** ${trade['profit_range'][0]:.2f} - ${trade['profit_range'][1]:.2f}")
                
                if 'annual_yield' in trade:
                    yield_color = "🟢" if trade['annual_yield'] > 10 else "🟡" if trade['annual_yield'] > 5 else "🔴"
                    st.metric("Annualized Yield", f"{yield_color} {trade['annual_yield']:.2f}%")
                
                if 'yield_if_called' in trade:
                    st.metric("Yield if Called", f"📞 {trade['yield_if_called']:.2f}%")
                
                if 'insurance_percentage' in trade:
                    st.metric("Insurance Cost", f"🛡️ {trade['insurance_percentage']:.2f}%")
                
                if 'assignment_risk' in trade:
                    st.warning(f"⚠️ **Assignment Risk:** Stock may be called away at ${trade['assignment_risk']:.2f}")
                
                # Add a separator
                st.markdown("---")

# =============================================================================
# Enhanced Streamlit Interface
# =============================================================================

def main():
    st.set_page_config(
        page_title="Enhanced Options Strategist", 
        page_icon="🎯", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Enhanced Options Strategist")
    st.markdown("*Dual-API powered with intelligent fallbacks: MarketStack + Polygon.io*")
    
    # Add status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **Stock Data:** MarketStack API")
    with col2:
        st.info("🎯 **Options Data:** Polygon.io API")
    with col3:
        st.info("🔄 **Fallback:** Synthetic Data Available")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 API Configuration")
        
        # API Keys with help text
        st.markdown("### 🔑 API Keys")
        marketstack_key = st.text_input(
            "Market Stack API Key", 
            value="9ad0d4f85e1a72dd7b3d19b8617b25f9",
            type="password",
            help="Get your free API key at https://marketstack.com"
        )
        
        polygon_key = st.text_input(
            "Polygon.io API Key", 
            value="igO7PgpW43MsVcJvr1ZuxZ_vYrH87jLZ", 
            type="password",
            help="Get your API key at https://polygon.io"
        )
        
        # API Key validation
        if marketstack_key:
            st.success("✅ MarketStack key provided")
        else:
            st.warning("⚠️ MarketStack key needed for real stock data")
        
        if polygon_key:
            st.success("✅ Polygon key provided")
        else:
            st.warning("⚠️ Polygon key needed for real options data")
        
        st.markdown("---")
        
        # Portfolio settings
        st.header("💼 Portfolio Settings")
        
        portfolio_value = st.number_input(
            "Portfolio Value ($)", 
            min_value=1000, 
            value=100000, 
            step=1000,
            help="Total portfolio value for position sizing"
        )
        
        risk_tolerance = st.slider(
            "Risk Tolerance (%)", 
            min_value=1, 
            max_value=10, 
            value=2,
            help="Percentage of portfolio to risk per trade"
        ) / 100
        
        st.info(f"💰 **Max Risk per Trade:** ${portfolio_value * risk_tolerance:,.2f}")
        
        st.markdown("---")
        
        # Analysis settings
        st.header("🔍 Analysis Settings")
        
        symbols_input = st.text_area(
            "Stock Symbols", 
            value="AAPL,MSFT,TSLA,GOOGL,AMZN",
            height=100,
            help="Enter stock symbols separated by commas or new lines"
        )
        
        # Parse symbols more flexibly
        symbols = []
        for line in symbols_input.replace(',', '\n').split('\n'):
            for symbol in line.split(','):
                clean_symbol = symbol.strip().upper()
                if clean_symbol and len(clean_symbol) <= 5:  # Basic validation
                    symbols.append(clean_symbol)
        
        st.info(f"📊 **Symbols to analyze:** {len(symbols)}")
        if symbols:
            st.write(", ".join(symbols))
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            use_fallback = st.checkbox(
                "Enable Synthetic Data Fallback", 
                value=True,
                help="Use synthetic data when API calls fail"
            )
            
            max_symbols = st.slider(
                "Max Symbols to Process", 
                min_value=1, 
                max_value=20, 
                value=10,
                help="Limit processing to avoid API rate limits"
            )
            
            symbols = symbols[:max_symbols]  # Limit symbols
        
        st.markdown("---")
        
        # Action button
        analyze_button = st.button(
            "🚀 Analyze Options Strategies", 
            type="primary",
            use_container_width=True,
            disabled=not symbols
        )
        
        if not symbols:
            st.error("❌ Please enter at least one valid symbol")
    
    # Main analysis
    if analyze_button and symbols:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize strategist
            status_text.text("🔧 Initializing Options Strategist...")
            progress_bar.progress(10)
            
            strategist = OptionsStrategist(
                risk_tolerance=risk_tolerance,
                marketstack_api_key=marketstack_key,
                polygon_api_key=polygon_key
            )
            
            # Run analysis
            status_text.text("📊 Analyzing symbols...")
            progress_bar.progress(30)
            
            recommendations = strategist.analyze_symbols(symbols, portfolio_value)
            progress_bar.progress(90)
            
            # Display results
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            if recommendations:
                strategist.display_recommendations(recommendations)
                
                # Export functionality
                st.subheader("📤 Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create summary for export
                    summary_data = []
                    for symbol, rec in recommendations.items():
                        trade = rec['trade_details']
                        summary_data.append({
                            'Symbol': symbol,
                            'Strategy': rec['strategy'],
                            'Confidence': rec['confidence'],
                            'Max Profit': trade.get('max_profit', 'N/A'),
                            'Max Loss': trade.get('max_loss', 'N/A'),
                            'Breakeven': trade.get('breakeven', trade.get('breakeven_up', 'N/A')),
                            'Days to Expiry': trade.get('days_to_expiry', 'N/A'),
                            'Stock Data Source': rec['data_sources']['stock_data'],
                            'Options Data Source': rec['data_sources']['options_data']
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download Summary CSV",
                        csv,
                        "options_analysis_summary.csv",
                        "text/csv"
                    )
                
                with col2:
                    # JSON export for detailed data
                    json_data = json.dumps(recommendations, indent=2, default=str)
                    st.download_button(
                        "📋 Download Detailed JSON",
                        json_data,
                        "options_analysis_detailed.json",
                        "application/json"
                    )
            
            else:
                st.error("❌ No recommendations generated. Please check your API keys and symbols.")
                
                # Troubleshooting help
                with st.expander("🔧 Troubleshooting Tips"):
                    st.markdown("""
                    **Common Issues:**
                    
                    1. **403 Errors from Polygon:** 
                       - Check your API key is correct
                       - Verify your subscription level supports the endpoints
                       - Try fewer symbols to avoid rate limits
                    
                    2. **No Data from MarketStack:**
                       - Verify the stock symbols are correct
                       - Check if symbols are available on the exchange
                       - Ensure your API key has sufficient credits
                    
                    3. **Synthetic Data Fallback:**
                       - When real APIs fail, synthetic data is used
                       - Results are for educational purposes only
                       - Consider upgrading API subscriptions for better data
                    
                    **Next Steps:**
                    - Enable synthetic data fallback for testing
                    - Try with fewer symbols (1-3)
                    - Check API documentation for proper usage
                    """)
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Analysis failed: {str(e)}")
            
            # Show detailed error in expander
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Educational content
    with st.expander("📚 Strategy Guide & Documentation"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Available Strategies
            
            **📈 Bullish Strategies:**
            - **Bull Call Spread:** Limited risk, limited reward
            - **Covered Call:** Income from stock ownership
            - **Cash Secured Put:** Income with purchase obligation
            
            **📉 Bearish Strategies:**
            - **Bear Put Spread:** Limited risk bearish play
            - **Protective Put:** Portfolio insurance
            
            **⚖️ Neutral Strategies:**
            - **Iron Condor:** Profit from low volatility
            - **Butterfly Spread:** High probability neutral play
            - **Collar:** Protection with limited upside
            
            **🌊 Volatility Strategies:**
            - **Long Straddle:** Profit from big moves
            - **Long Strangle:** Lower cost volatility play
            """)
        
        with col2:
            st.markdown("""
            ### 🔧 Technical Features
            
            **📊 Data Sources:**
            - MarketStack: Historical prices, technical indicators
            - Polygon.io: Real-time options chains
            - Synthetic: Fallback Black-Scholes pricing
            
            **🧠 Analysis Features:**
            - RSI momentum analysis
            - Moving average trends
            - Volatility regime detection
            - Volume analysis
            - Risk-adjusted position sizing
            
            **🛡️ Risk Management:**
            - Portfolio percentage risk limits
            - Volatility-based position sizing
            - Strategy confidence scoring
            - Multiple breakeven calculations
            """)
    

if __name__ == "__main__":
    main()
    

    
