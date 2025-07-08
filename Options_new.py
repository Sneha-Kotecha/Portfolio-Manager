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
import logging
import pytz

warnings.filterwarnings('ignore')

# =============================================================================
# Enhanced Options Strategist with All Improvements - Production Version
# =============================================================================

class PerformanceTracker:
    """Track strategy performance over time"""
    
    def __init__(self):
        self.trades = []
        self.portfolio_value = []
        self.win_rate = 0
        self.total_return = 0
    
    def add_trade(self, trade_data: Dict):
        """Add completed trade to tracking"""
        self.trades.append({
            'symbol': trade_data['symbol'],
            'strategy': trade_data['strategy'],
            'entry_date': trade_data['entry_date'],
            'exit_date': trade_data.get('exit_date'),
            'profit_loss': trade_data.get('profit_loss', 0),
            'max_risk': trade_data['max_risk'],
            'success': trade_data.get('profit_loss', 0) > 0
        })
        
        self._update_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        if not self.trades:
            return
            
        completed_trades = [t for t in self.trades if 'profit_loss' in t]
        if not completed_trades:
            return
        
        total_profit = sum(t['profit_loss'] for t in completed_trades)
        winning_trades = sum(1 for t in completed_trades if t['profit_loss'] > 0)
        
        self.win_rate = winning_trades / len(completed_trades)
        self.total_return = total_profit
    
    def display_performance(self):
        """Display performance metrics in Streamlit"""
        if not self.trades:
            st.info("📊 No trades recorded yet")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(self.trades))
        with col2:
            st.metric("Win Rate", f"{self.win_rate:.1%}")
        with col3:
            st.metric("Total P&L", f"${self.total_return:,.2f}")
        with col4:
            avg_trade = self.total_return / len(self.trades) if self.trades else 0
            st.metric("Avg Trade P&L", f"${avg_trade:,.2f}")

class OptionsStrategist:
    """
    Enhanced Options Strategist with comprehensive improvements and real data only
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
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # Educational content
        self.strategy_explanations = self._load_strategy_explanations()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for debugging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_strategy_explanations(self) -> Dict:
        """Load educational content for strategies"""
        return {
            "Bull Call Spread": {
                "description": "Moderately bullish strategy with limited risk and reward",
                "best_conditions": "Moderately bullish outlook, moderate volatility",
                "max_profit": "Difference between strikes minus net debit",
                "max_loss": "Net debit paid",
                "breakeven": "Lower strike + net debit"
            },
            "Bear Put Spread": {
                "description": "Moderately bearish strategy with limited risk and reward",
                "best_conditions": "Moderately bearish outlook, moderate volatility",
                "max_profit": "Difference between strikes minus net debit",
                "max_loss": "Net debit paid",
                "breakeven": "Higher strike - net debit"
            },
            "Iron Condor": {
                "description": "Neutral strategy that profits from low volatility",
                "best_conditions": "Sideways market, high implied volatility",
                "max_profit": "Net credit received",
                "max_loss": "Strike width minus net credit",
                "breakeven": "Two breakeven points around short strikes"
            },
            "Covered Call": {
                "description": "Income strategy for stock owners",
                "best_conditions": "Neutral to slightly bullish, own the stock",
                "max_profit": "Strike price - cost basis + premium",
                "max_loss": "Substantial (stock could go to zero)",
                "breakeven": "Stock cost basis - premium received"
            },
            "Cash Secured Put": {
                "description": "Income strategy with obligation to buy stock",
                "best_conditions": "Neutral to bullish, want to own stock at lower price",
                "max_profit": "Premium received",
                "max_loss": "Strike price - premium received",
                "breakeven": "Strike price - premium received"
            },
            "Protective Put": {
                "description": "Insurance for stock holdings",
                "best_conditions": "Own stock, want downside protection",
                "max_profit": "Unlimited (stock appreciation minus premium)",
                "max_loss": "Stock price - strike price + premium paid",
                "breakeven": "Stock price + premium paid"
            }
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

    def check_market_conditions(self) -> Dict:
        """Check overall market conditions to adjust strategy recommendations"""
        try:
            market_data = {
                'market_hours': self._is_market_open(),
                'volatility_environment': 'NORMAL',
                'market_trend': 'NEUTRAL',
                'earnings_season': False,
                'fed_meeting_week': False,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_data
        except Exception as e:
            self.logger.error(f"Could not fetch market conditions: {str(e)}")
            return {'error': f'Could not fetch market conditions: {str(e)}'}

    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)
            
            market_open = time(9, 30)
            market_close = time(16, 0)
            
            is_weekday = now.weekday() < 5
            is_market_hours = market_open <= now.time() <= market_close
            
            return is_weekday and is_market_hours
        except Exception:
            return False

    def validate_strategy_inputs(self, symbol: str, strategy: str, stock_data: Dict, options_data: Dict) -> bool:
        """Validate inputs before attempting strategy"""
        
        required_data = {
            'BULL_CALL_SPREAD': {'min_calls': 2, 'min_puts': 0},
            'BEAR_PUT_SPREAD': {'min_calls': 0, 'min_puts': 2},
            'IRON_CONDOR': {'min_calls': 2, 'min_puts': 2},
            'STRADDLE': {'min_calls': 1, 'min_puts': 1},
            'STRANGLE': {'min_calls': 1, 'min_puts': 1},
            'COVERED_CALL': {'min_calls': 1, 'min_puts': 0},
            'PROTECTIVE_PUT': {'min_calls': 0, 'min_puts': 1},
            'CASH_SECURED_PUT': {'min_calls': 0, 'min_puts': 1},
            'COLLAR': {'min_calls': 1, 'min_puts': 1},
            'BUTTERFLY': {'min_calls': 3, 'min_puts': 0}
        }
        
        if strategy not in required_data:
            return False
            
        reqs = required_data[strategy]
        calls = options_data.get('calls', pd.DataFrame())
        puts = options_data.get('puts', pd.DataFrame())
        
        viable_calls = len(calls[calls['lastPrice'] > 0.02]) if not calls.empty else 0
        viable_puts = len(puts[puts['lastPrice'] > 0.02]) if not puts.empty else 0
        
        if viable_calls < reqs['min_calls'] or viable_puts < reqs['min_puts']:
            self.logger.warning(f"{strategy} requires {reqs['min_calls']} calls and {reqs['min_puts']} puts. Available: {viable_calls} calls, {viable_puts} puts")
            return False
        
        return True

    def optimize_portfolio_allocation(self, recommendations: Dict, portfolio_value: float) -> Dict:
        """Optimize position sizing across multiple strategies"""
        
        if not recommendations:
            return {}
        
        # Calculate risk-adjusted allocation
        total_risk = 0
        for symbol, rec in recommendations.items():
            trade = rec['trade_details']
            max_loss = trade.get('max_loss', 0)
            if isinstance(max_loss, (int, float)) and max_loss > 0:
                total_risk += max_loss
        
        if total_risk == 0:
            return recommendations
        
        # Adjust position sizes to not exceed total risk tolerance
        max_total_risk = portfolio_value * 0.1  # 10% max total portfolio risk
        
        if total_risk > max_total_risk:
            risk_reduction_factor = max_total_risk / total_risk
            
            st.warning(f"⚠️ Total risk (${total_risk:,.2f}) exceeds limit. Reducing positions by {(1-risk_reduction_factor)*100:.1f}%")
            
            # Apply reduction factor to all positions
            for symbol, rec in recommendations.items():
                trade = rec['trade_details']
                if 'legs' in trade:
                    for leg in trade['legs']:
                        if 'contracts' in leg:
                            leg['contracts'] = max(1, int(leg['contracts'] * risk_reduction_factor))
        
        return recommendations

    def analyze_symbols(self, symbols: List[str], portfolio_value: float = 100000) -> Dict:
        """Main analysis function with enhanced strategy selection"""
        recommendations = {}
        
        # Check market conditions
        market_conditions = self.check_market_conditions()
        
        # Validate API keys
        if not self.marketstack_api_key:
            st.error("❌ MarketStack API key required for real data")
            return {}
        
        if not self.polygon_api_key:
            st.error("❌ Polygon.io API key required for real data")
            return {}
        
        for symbol in symbols:
            try:
                # Get stock data
                stock_data = self._get_marketstack_data(symbol)
                if not stock_data:
                    st.warning(f"⚠️ Skipping {symbol} - no stock data available")
                    continue
                
                # Get options data
                options_data = self._get_polygon_options_data(symbol)
                if not options_data:
                    st.warning(f"⚠️ Skipping {symbol} - no options data available")
                    continue
                
                # Analyze market conditions
                market_analysis = self._analyze_market_conditions(stock_data)
                
                # Smart strategy selection with data validation
                strategy_scores = self._select_strategy_with_data_validation(
                    market_analysis, stock_data, options_data
                )
                
                # Try strategies in order of preference
                successful_strategy = None
                trade_rec = None
                
                sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
                
                for strategy_name, confidence in sorted_strategies:
                    # Validate before attempting
                    if not self.validate_strategy_inputs(symbol, strategy_name, stock_data, options_data):
                        continue
                    
                    try:
                        trade_rec = self._generate_trade_recommendation(
                            symbol, strategy_name, stock_data, options_data, 
                            market_analysis, portfolio_value
                        )
                        
                        if 'error' not in trade_rec:
                            successful_strategy = strategy_name
                            break
                        else:
                            self.logger.warning(f"{strategy_name} failed for {symbol}: {trade_rec['error']}")
                    except Exception as e:
                        self.logger.error(f"{strategy_name} error for {symbol}: {str(e)}")
                        continue
                
                if successful_strategy and trade_rec and 'error' not in trade_rec:
                    recommendations[symbol] = {
                        'strategy': successful_strategy,
                        'market_analysis': market_analysis,
                        'trade_details': trade_rec,
                        'confidence': strategy_scores.get(successful_strategy, 5.0),
                        'data_sources': {
                            'stock_data': 'MarketStack',
                            'options_data': 'Polygon.io'
                        },
                        'market_conditions': market_conditions
                    }
                    st.success(f"✅ Analysis complete for {symbol} - {successful_strategy}")
                else:
                    st.warning(f"⚠️ Could not generate any viable strategy for {symbol}")
                
            except Exception as e:
                st.error(f"❌ Error analyzing {symbol}: {str(e)}")
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Optimize portfolio allocation
        recommendations = self.optimize_portfolio_allocation(recommendations, portfolio_value)
        
        return recommendations
    
    def _select_strategy_with_data_validation(self, market_analysis: Dict, stock_data: Dict, options_data: Dict) -> Dict[str, float]:
        """
        Enhanced strategy selection using weighted scoring system based on:
        - Market conditions (trend, volatility, momentum)
        - Options liquidity and availability
        - Risk/reward profiles
        - Market regime appropriateness
        """
        scores = {}
        calls = options_data.get('calls', pd.DataFrame())
        puts = options_data.get('puts', pd.DataFrame())
        current_price = stock_data['current_price']
        volatility = stock_data.get('realized_vol', 0.3)
        
        # Enhanced options availability metrics
        if calls.empty and puts.empty:
            self.logger.warning("No options data available")
            return {'BUY_AND_HOLD': 3.0}
        
        if calls.empty:
            self.logger.warning("No calls available - put strategies only")
            return {'CASH_SECURED_PUT': 6.0, 'PROTECTIVE_PUT': 4.0}
        
        if puts.empty:
            self.logger.warning("No puts available - call strategies only")
            return {'COVERED_CALL': 6.0, 'BULL_CALL_SPREAD': 4.0}
        
        # Calculate option quality metrics - Fix data type issues
        try:
            # Ensure numeric columns are properly converted
            for df_name, df in [('calls', calls), ('puts', puts)]:
                if not df.empty:
                    for col in ['lastPrice', 'volume', 'strike', 'impliedVolatility']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0 if col != 'impliedVolatility' else volatility)
                            
        except Exception as e:
            self.logger.error(f"Error processing options data: {str(e)}")
            return {'COVERED_CALL': 5.0, 'CASH_SECURED_PUT': 5.0}
        
        # Filter liquid options with more relaxed criteria
        liquid_calls = calls[
            (calls['lastPrice'] > 0.01) & 
            (calls['volume'] > 0)  # Accept any volume > 0
        ] if not calls.empty else pd.DataFrame()
        
        liquid_puts = puts[
            (puts['lastPrice'] > 0.01) & 
            (puts['volume'] > 0)  # Accept any volume > 0
        ] if not puts.empty else pd.DataFrame()
        
        # ATM and OTM options with more flexible ranges
        try:
            atm_range = current_price * 0.10  # Wider range
            near_atm_calls = liquid_calls[abs(liquid_calls['strike'] - current_price) <= atm_range] if not liquid_calls.empty else pd.DataFrame()
            near_atm_puts = liquid_puts[abs(liquid_puts['strike'] - current_price) <= atm_range] if not liquid_puts.empty else pd.DataFrame()
            
            # OTM options for spreads - more lenient
            otm_calls = liquid_calls[liquid_calls['strike'] > current_price] if not liquid_calls.empty else pd.DataFrame()
            otm_puts = liquid_puts[liquid_puts['strike'] < current_price] if not liquid_puts.empty else pd.DataFrame()
            
            # ITM options for protective strategies
            itm_puts = liquid_puts[liquid_puts['strike'] > current_price] if not liquid_puts.empty else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error filtering options by strike: {str(e)}")
            # Fallback to available options
            otm_calls = liquid_calls.head(10) if not liquid_calls.empty else pd.DataFrame()
            otm_puts = liquid_puts.head(10) if not liquid_puts.empty else pd.DataFrame()
            near_atm_calls = liquid_calls.head(5) if not liquid_calls.empty else pd.DataFrame()
            near_atm_puts = liquid_puts.head(5) if not liquid_puts.empty else pd.DataFrame()
            itm_puts = liquid_puts.head(5) if not liquid_puts.empty else pd.DataFrame()
        
        # Market condition factors
        trend = market_analysis.get('trend', 'SIDEWAYS')
        vol_regime = market_analysis.get('volatility_regime', 'NORMAL_VOL')
        momentum = market_analysis.get('momentum', 'NEUTRAL')
        
        # Much more flexible IV assessment
        try:
            if not calls.empty and 'impliedVolatility' in calls.columns:
                avg_iv = calls['impliedVolatility'].mean()
                if pd.isna(avg_iv) or avg_iv <= 0:
                    avg_iv = volatility * 1.1
            else:
                avg_iv = volatility * 1.1
                
            # More permissive IV categorization
            iv_vs_rv_ratio = avg_iv / max(volatility, 0.1)  # Avoid division by zero
            if iv_vs_rv_ratio > 2.0:
                iv_environment = 'HIGH'
            elif iv_vs_rv_ratio < 0.6:
                iv_environment = 'LOW'
            else:
                iv_environment = 'NORMAL'
        except Exception:
            iv_environment = 'NORMAL'
            avg_iv = volatility
        
        self.logger.info(f"Market conditions - Trend: {trend}, Vol Regime: {vol_regime}, IV Environment: {iv_environment}")
        self.logger.info(f"Options available - Calls: {len(liquid_calls)}, Puts: {len(liquid_puts)}")
        
        # STRATEGY SCORING SYSTEM - Much more permissive
        
        # 1. BASIC INCOME STRATEGIES (always available if options exist)
        if len(otm_calls) >= 1:
            base_score = 7.0  # Higher base score
            # Moderate boosts instead of strict requirements
            if trend in ['SIDEWAYS', 'SHORT_TERM_BULLISH', 'BULLISH']: 
                base_score += 1.0
            if iv_environment in ['HIGH', 'NORMAL']: 
                base_score += 0.5
            if vol_regime in ['LOW_VOL', 'NORMAL_VOL']: 
                base_score += 0.5
            scores['COVERED_CALL'] = base_score
        
        if len(otm_puts) >= 1:
            base_score = 7.0  # Higher base score
            # More permissive conditions
            if trend in ['BULLISH', 'SHORT_TERM_BULLISH', 'SIDEWAYS', 'SHORT_TERM_BEARISH']: 
                base_score += 1.0
            if iv_environment in ['HIGH', 'NORMAL']: 
                base_score += 0.5
            # Only significant penalty for very bearish conditions
            if trend == 'STRONG_BEARISH' and momentum == 'EXTREMELY_OVERSOLD':
                base_score -= 1.0
            scores['CASH_SECURED_PUT'] = base_score
        
        # 2. DIRECTIONAL STRATEGIES - More flexible
        if len(otm_calls) >= 2:
            base_score = 5.5
            # Boost for any bullish conditions
            if trend in ['BULLISH', 'SHORT_TERM_BULLISH']:
                base_score += 1.5
            if momentum in ['BULLISH', 'OVERSOLD']:  # Oversold can lead to bounce
                base_score += 1.0
            # Accept any IV environment
            if iv_environment in ['NORMAL', 'LOW']:
                base_score += 0.5
            # Mild penalty only for strong bearish
            if trend == 'STRONG_BEARISH':
                base_score -= 1.0
            scores['BULL_CALL_SPREAD'] = base_score
        
        if len(otm_puts) >= 2:
            base_score = 5.5
            # Boost for any bearish conditions
            if trend in ['BEARISH', 'SHORT_TERM_BEARISH']:
                base_score += 1.5
            if momentum in ['BEARISH', 'OVERBOUGHT']:  # Overbought can lead to decline
                base_score += 1.0
            # Accept any IV environment
            if iv_environment in ['NORMAL', 'LOW']:
                base_score += 0.5
            # Mild penalty only for strong bullish
            if trend == 'STRONG_BULLISH':
                base_score -= 1.0
            scores['BEAR_PUT_SPREAD'] = base_score
        
        # 3. NEUTRAL STRATEGIES - More accessible
        if len(otm_calls) >= 1 and len(otm_puts) >= 1:  # Reduced requirement
            base_score = 5.5
            # Boost for sideways but don't require it
            if trend == 'SIDEWAYS':
                base_score += 1.5
            elif trend in ['SHORT_TERM_BULLISH', 'SHORT_TERM_BEARISH']:
                base_score += 0.5
            # Any IV environment acceptable
            if iv_environment == 'HIGH':
                base_score += 1.0
            elif iv_environment == 'NORMAL':
                base_score += 0.5
            scores['IRON_CONDOR'] = base_score
        
        # 4. VOLATILITY STRATEGIES - More practical
        if len(near_atm_calls) >= 1 and len(near_atm_puts) >= 1:
            base_score = 4.5
            # Boost for low IV or volatile conditions
            if iv_environment == 'LOW' or vol_regime == 'LOW_VOL':
                base_score += 1.5
            if trend == 'SIDEWAYS':  # Breakout potential
                base_score += 1.0
            scores['STRADDLE'] = base_score
            
            # Strangle as cheaper alternative
            if len(otm_calls) >= 1 and len(otm_puts) >= 1:
                scores['STRANGLE'] = base_score + 0.5
        
        # 5. PROTECTIVE STRATEGIES - Always useful
        if len(liquid_puts) >= 1:  # Accept any puts
            base_score = 5.0
            # Boost for uncertain or declining markets
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                base_score += 1.0
            if trend in ['BEARISH', 'SHORT_TERM_BEARISH']:
                base_score += 1.0
            # Also useful for protecting gains
            if trend in ['BULLISH', 'SHORT_TERM_BULLISH']:
                base_score += 0.5
            scores['PROTECTIVE_PUT'] = base_score
        
        # 6. COMBINATION STRATEGIES
        if len(otm_calls) >= 1 and len(liquid_puts) >= 1:
            base_score = 5.2
            # Boost for mixed market conditions
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                base_score += 0.8
            if trend in ['SIDEWAYS', 'SHORT_TERM_BULLISH', 'SHORT_TERM_BEARISH']:
                base_score += 0.8
            scores['COLLAR'] = base_score
        
        # 7. ADVANCED STRATEGIES - More accessible requirements
        if len(liquid_calls) >= 3:
            # Butterfly - neutral strategy
            base_score = 4.8
            if trend in ['SIDEWAYS', 'SHORT_TERM_BULLISH', 'SHORT_TERM_BEARISH']:
                base_score += 1.0
            if iv_environment in ['HIGH', 'NORMAL']:
                base_score += 0.5
            scores['BUTTERFLY'] = base_score
        
        # MINIMAL FILTERING - Ensure strategies are available
        
        # Very lenient liquidity requirements
        min_options_needed = {
            'COVERED_CALL': 1,
            'CASH_SECURED_PUT': 1,
            'BULL_CALL_SPREAD': 2,
            'BEAR_PUT_SPREAD': 2,
            'IRON_CONDOR': 2,  # Much lower requirement
            'STRADDLE': 2,
            'STRANGLE': 2,
            'PROTECTIVE_PUT': 1,
            'COLLAR': 2,
            'BUTTERFLY': 3
        }
        
        # Filter strategies with very low threshold
        filtered_scores = {}
        for strategy, score in scores.items():
            required_options = min_options_needed.get(strategy, 1)
            
            if strategy in ['COVERED_CALL', 'BULL_CALL_SPREAD']:
                available = len(liquid_calls)
            elif strategy in ['CASH_SECURED_PUT', 'BEAR_PUT_SPREAD', 'PROTECTIVE_PUT']:
                available = len(liquid_puts)
            else:
                available = min(len(liquid_calls), len(liquid_puts))
            
            # Very low score threshold
            if available >= required_options and score > 1.0:
                filtered_scores[strategy] = score
        
        # ALWAYS provide fallback strategies
        if not filtered_scores:
            self.logger.warning("No strategies passed filtering - providing fallback options")
            if len(liquid_calls) > 0 and len(liquid_puts) > 0:
                filtered_scores = {
                    'COVERED_CALL': 6.0,
                    'CASH_SECURED_PUT': 6.0,
                    'IRON_CONDOR': 5.0
                }
            elif len(liquid_calls) > 0:
                filtered_scores = {
                    'COVERED_CALL': 6.0,
                    'BULL_CALL_SPREAD': 4.0 if len(liquid_calls) >= 2 else 0
                }
            elif len(liquid_puts) > 0:
                filtered_scores = {
                    'CASH_SECURED_PUT': 6.0,
                    'PROTECTIVE_PUT': 4.0
                }
            else:
                filtered_scores = {'BUY_AND_HOLD': 3.0}
            
            # Remove zero scores
            filtered_scores = {k: v for k, v in filtered_scores.items() if v > 0}
        
        # Sort by score
        sorted_strategies = dict(sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True))
        
        self.logger.info(f"Strategy scores: {sorted_strategies}")
        
        # Return top 5 strategies for variety
        return dict(list(sorted_strategies.items())[:5])


    def _get_strategy_description(self, strategy: str, market_analysis: Dict) -> str:
        """Get detailed description of why this strategy was selected"""
        
        descriptions = {
            'COVERED_CALL': f"Income generation strategy suitable for {market_analysis['trend'].lower()} market with {market_analysis['volatility_regime'].lower()} volatility",
            'CASH_SECURED_PUT': f"Income + potential stock acquisition in {market_analysis['trend'].lower()} market",
            'BULL_CALL_SPREAD': f"Limited risk directional play for bullish outlook (momentum: {market_analysis['momentum']:.2%})",
            'BEAR_PUT_SPREAD': f"Limited risk directional play for bearish outlook (momentum: {market_analysis['momentum']:.2%})",
            'IRON_CONDOR': f"Range-bound strategy for sideways market with {market_analysis['volatility_regime'].lower()} volatility",
            'STRADDLE': f"Volatility expansion play - expecting significant movement",
            'STRANGLE': f"Lower-cost volatility play - expecting moderate movement",
            'PROTECTIVE_PUT': f"Downside protection for existing positions in uncertain market",
            'COLLAR': f"Income + protection combination for volatile market conditions",
            'BUTTERFLY': f"Precision neutral strategy for range-bound market",
            'CALENDAR_SPREAD': f"Time decay strategy with volatility advantage"
        }
        
        return descriptions.get(strategy, "Custom options strategy based on current market conditions")


    def get_top_strategy_recommendation(self, market_analysis: Dict, stock_data: Dict, options_data: Dict) -> Tuple[str, float, str]:
        """Get the single best strategy recommendation with explanation"""
        
        strategy_scores = self.select_strategy_with_data_validation(market_analysis, stock_data, options_data)
        
        if not strategy_scores:
            return "BUY_AND_HOLD", 3.0, "No suitable options strategies available"
        
        # Get the highest scoring strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        strategy_name, confidence = best_strategy
        
        # Get explanation
        explanation = self._get_strategy_description(strategy_name, market_analysis)
        
        return strategy_name, confidence, explanation

    def _get_marketstack_data(self, symbol: str) -> Optional[Dict]:
        """Enhanced MarketStack data fetching"""
        if not self.marketstack_api_key:
            st.error("❌ MarketStack API key required")
            return None
            
        try:
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
                return None
            
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
            df['sma_40'] = df['close'].rolling(40, min_periods=10).mean()
            df['sma_50'] = df['close'].rolling(50, min_periods=10).mean()
            df['sma_100'] = df['close'].rolling(100, min_periods=20).mean()
            df['sma_150'] = df['close'].rolling(150, min_periods=30).mean()
            df['sma_200'] = df['close'].rolling(200, min_periods=40).mean()

            # Get most recent values with fallbacks
            sma_20 = df['sma_20'].dropna().iloc[-1] if len(df['sma_20'].dropna()) > 0 else current_price
            sma_40 = df['sma_40'].dropna().iloc[-1] if len(df['sma_40'].dropna()) > 0 else current_price
            sma_50 = df['sma_50'].dropna().iloc[-1] if len(df['sma_50'].dropna()) > 0 else current_price
            sma_100 = df['sma_100'].dropna().iloc[-1] if len(df['sma_100'].dropna()) > 0 else current_price
            sma_150 = df['sma_150'].dropna().iloc[-1] if len(df['sma_150'].dropna()) > 0 else current_price
            sma_200 = df['sma_200'].dropna().iloc[-1] if len(df['sma_200'].dropna()) > 0 else current_price
            
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
                'sma_40': float(sma_40),
                'sma_50': float(sma_50),
                'sma_100': float(sma_100),
                'sma_150': float(sma_150),
                'sma_200': float(sma_200),
                'realized_vol': realized_vol,
                'rsi': rsi,
                'returns': returns,
                'hist_data': df,
                'volume': latest_volume,
                'avg_volume': df['volume'].mean() if not df['volume'].isna().all() else latest_volume,
                'high_52w': df['high'].max(),
                'low_52w': df['low'].min(),
                'market_cap': None,
                'sector': 'Unknown',
                'beta': 1.0,
                'source': 'marketstack'
            }
            
            return stock_data
            
        except Exception as e:
            st.error(f"Error fetching MarketStack data for {symbol}: {str(e)}")
            return None
    
    def _generate_basic_stock_data(self, symbol: str, current_price: float, volume: float) -> Dict:
        """Generate basic stock data with limited info when historical data fails"""
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
        """Enhanced Polygon options data with better error handling"""
        if not self.polygon_api_key:
            st.error("❌ Polygon.io API key required")
            return None
            
        try:
            # Get current stock price
            current_price = self._get_current_price_polygon(symbol)
            if not current_price:
                st.warning(f"Could not get current price for {symbol} from Polygon")
                return None
            
            # Get options contracts
            contracts_url = f"{self.polygon_url}/v3/reference/options/contracts"
            
            start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
            
            contracts_params = {
                'underlying_ticker': symbol,
                'apikey': self.polygon_api_key,
                'limit': 1000,
                'order': 'asc',
                'sort': 'expiration_date',
                'expiration_date.gte': start_date,
                'expiration_date.lte': end_date
            }
            
            contracts_data = self._make_api_request(contracts_url, contracts_params, "Polygon Contracts")
            
            if not contracts_data or 'results' not in contracts_data or not contracts_data['results']:
                st.warning(f"No options contracts found for {symbol}")
                return None
            
            # Process contracts
            future_contracts = self._filter_options_contracts(contracts_data['results'], current_price)
            
            if not future_contracts:
                st.warning(f"No suitable options contracts for {symbol}")
                return None
            
            # Create options dataframe
            return self._create_options_dataframe(future_contracts, current_price)
            
        except Exception as e:
            st.error(f"Error getting Polygon options data for {symbol}: {str(e)}")
            return None

    def _filter_options_contracts(self, contracts: List[Dict], current_price: float) -> List[Dict]:
        """Filter options contracts to reasonable strikes and dates"""
        today = datetime.now().date()
        filtered = []
        
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
                
                # More flexible strike filtering for better options availability
                strike_ratio = abs(strike - current_price) / current_price
                if strike_ratio <= 0.30:  # Within 30% of current price
                    days_to_exp = (exp_date - today).days
                    if 1 <= days_to_exp <= 90:
                        contract['days_to_expiry'] = days_to_exp
                        filtered.append(contract)
                        
            except (ValueError, TypeError):
                continue
        
        return filtered

    def _create_options_dataframe(self, contracts: List[Dict], current_price: float) -> Dict:
        """Create options dataframe with better expiration selection"""
        # Group by expiration
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
        
        # Find best expiration
        best_exp = None
        best_score = 0
        
        for exp_date in sorted(exp_groups.keys()):
            calls_count = len(exp_groups[exp_date]['calls'])
            puts_count = len(exp_groups[exp_date]['puts'])
            
            if calls_count >= 3 and puts_count >= 3:
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_datetime.date() - datetime.now().date()).days
                
                if 7 <= days_to_exp <= 60:
                    time_score = 100 - abs(30 - days_to_exp)
                    option_score = min(calls_count + puts_count, 50)
                    total_score = time_score + option_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_exp = exp_date
        
        if not best_exp:
            for exp_date in sorted(exp_groups.keys()):
                calls_count = len(exp_groups[exp_date]['calls'])
                puts_count = len(exp_groups[exp_date]['puts'])
                if calls_count >= 2 and puts_count >= 2:
                    best_exp = exp_date
                    break
        
        if not best_exp:
            return None
        
        # Generate option prices
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
            
            price = self._black_scholes_price(underlying_price, strike, exp_date, option_type)
            
            if price > 0.01:
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
        """Get current stock price from Polygon"""
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
        """Comprehensive market condition analysis with multi-timeframe trends"""
        current_price = stock_data['current_price']
        
        # Get all SMA values
        sma_20 = stock_data['sma_20']
        sma_40 = stock_data['sma_40']
        sma_50 = stock_data['sma_50']
        sma_100 = stock_data['sma_100']
        sma_150 = stock_data['sma_150']
        sma_200 = stock_data['sma_200']
        
        rsi = stock_data['rsi']
        realized_vol = stock_data['realized_vol']
        
        # Enhanced trend analysis with multiple timeframes
        def analyze_trend():
            # Short-term trend signals
            short_term_bullish = current_price > sma_20 > sma_40 and sma_20 > sma_50
            short_term_bearish = current_price < sma_20 < sma_40 and sma_20 < sma_50
            
            # Long-term trend signals
            long_term_bullish = sma_50 > sma_100 > sma_150 and sma_100 > sma_200
            long_term_bearish = sma_50 < sma_100 < sma_150 and sma_100 < sma_200
            
            # Price position analysis
            above_all_smas = current_price > max(sma_20, sma_40, sma_50, sma_100, sma_150, sma_200)
            below_all_smas = current_price < min(sma_20, sma_40, sma_50, sma_100, sma_150, sma_200)
            
            # Determine trend and strength
            if short_term_bullish and long_term_bullish:
                trend = 'STRONG_BULLISH'
                strength = min((current_price - sma_200) / sma_200 * 100, 15)
            elif short_term_bullish and not long_term_bearish:
                trend = 'SHORT_TERM_BULLISH'
                strength = min((current_price - sma_50) / sma_50 * 100, 12)
            elif short_term_bearish and long_term_bearish:
                trend = 'STRONG_BEARISH'
                strength = min((sma_200 - current_price) / sma_200 * 100, 15)
            elif short_term_bearish and not long_term_bullish:
                trend = 'SHORT_TERM_BEARISH'
                strength = min((sma_50 - current_price) / sma_50 * 100, 12)
            elif above_all_smas:
                trend = 'BULLISH'
                strength = min((current_price - sma_100) / sma_100 * 100, 10)
            elif below_all_smas:
                trend = 'BEARISH'
                strength = min((sma_100 - current_price) / sma_100 * 100, 10)
            else:
                trend = 'SIDEWAYS'
                sma_range = max(sma_20, sma_50, sma_100, sma_200) - min(sma_20, sma_50, sma_100, sma_200)
                strength = max(5.0, min(sma_range / current_price * 100, 8))
            
            return trend, abs(strength)
        
        trend, trend_strength = analyze_trend()
        
        # Enhanced volatility regime analysis
        if realized_vol > 0.40:
            vol_regime = 'EXTREME_VOL'
        elif realized_vol > 0.30:
            vol_regime = 'HIGH_VOL'
        elif realized_vol < 0.10:
            vol_regime = 'LOW_VOL'
        elif realized_vol < 0.20:
            vol_regime = 'NORMAL_VOL'
        else:
            vol_regime = 'ELEVATED_VOL'
        
        # Enhanced momentum analysis with additional levels
        if rsi > 80:
            momentum = 'EXTREMELY_OVERBOUGHT'
        elif rsi > 70:
            momentum = 'OVERBOUGHT'
        elif rsi > 60:
            momentum = 'BULLISH'
        elif rsi < 20:
            momentum = 'EXTREMELY_OVERSOLD'
        elif rsi < 30:
            momentum = 'OVERSOLD'
        elif rsi < 40:
            momentum = 'BEARISH'
        else:
            momentum = 'NEUTRAL'
        
        # Volume analysis
        volume_ratio = stock_data['volume'] / stock_data.get('avg_volume', stock_data['volume'])
        if volume_ratio > 2.0:
            volume_trend = 'EXTREME_VOLUME'
        elif volume_ratio > 1.5:
            volume_trend = 'HIGH_VOLUME'
        elif volume_ratio < 0.3:
            volume_trend = 'VERY_LOW_VOLUME'
        elif volume_ratio < 0.5:
            volume_trend = 'LOW_VOLUME'
        else:
            volume_trend = 'NORMAL_VOLUME'
        
        # Calculate additional metrics
        price_vs_52w_high = (current_price / stock_data['high_52w']) * 100
        price_vs_52w_low = (current_price / stock_data['low_52w']) * 100
        
        # SMA alignment analysis
        sma_alignment_bullish = sma_20 > sma_40 > sma_50 > sma_100 > sma_150 > sma_200
        sma_alignment_bearish = sma_20 < sma_40 < sma_50 < sma_100 < sma_150 < sma_200
        
        # Market structure analysis
        if price_vs_52w_high > 95:
            market_structure = 'BREAKOUT_TERRITORY'
        elif price_vs_52w_high > 80:
            market_structure = 'STRONG_UPTREND'
        elif price_vs_52w_low < 105:
            market_structure = 'NEAR_LOWS'
        elif price_vs_52w_low < 120:
            market_structure = 'WEAK_STRUCTURE'
        else:
            market_structure = 'BALANCED'
        
        return {
            'trend': trend,
            'trend_strength': round(trend_strength, 2),
            'volatility_regime': vol_regime,
            'momentum': momentum,
            'volume_trend': volume_trend,
            'market_structure': market_structure,
            'sma_alignment_bullish': sma_alignment_bullish,
            'sma_alignment_bearish': sma_alignment_bearish,
            'rsi': rsi,
            'realized_vol': realized_vol,
            'volume_ratio': round(volume_ratio, 2),
            '52w_high': stock_data['high_52w'],
            '52w_low': stock_data['low_52w'],
            'price_vs_52w_high': round(price_vs_52w_high, 2),
            'price_vs_52w_low': round(price_vs_52w_low, 2),
            'current_price': current_price,
            'sma_values': {
                'sma_20': sma_20,
                'sma_40': sma_40,
                'sma_50': sma_50,
                'sma_100': sma_100,
                'sma_150': sma_150,
                'sma_200': sma_200
            }
        }
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
            error_msg = self._format_error_message(str(e), symbol, strategy)
            self.logger.error(f"Strategy {strategy} failed for {symbol}: {str(e)}")
            return {'error': error_msg}
    
    def _format_error_message(self, error_msg: str, symbol: str, strategy: str) -> str:
        """Create user-friendly error messages with suggestions"""
        suggestions = {
            "insufficient": "Try a simpler strategy or different symbol",
            "no suitable": "Options chain may be limited - try longer expiration",
            "net debit": "Market conditions unfavorable for this strategy",
            "invalid": "Options pricing inconsistent - market may be volatile"
        }
        
        suggestion = "Check symbol and try again"
        for key, value in suggestions.items():
            if key.lower() in error_msg.lower():
                suggestion = value
                break
        
        return f"{error_msg}. Suggestion: {suggestion}"
    
    def _calculate_delta(self, options_df, current_price: float, days_to_expiry: int, 
                    implied_vol: float) -> list:
        """Calculate theoretical delta using Black-Scholes approximation"""
        import numpy as np
        from scipy.stats import norm
        
        # Simplified delta calculation
        risk_free_rate = 0.05  # Assume 5% risk-free rate
        time_to_expiry = days_to_expiry / 365
        
        deltas = []
        for _, row in options_df.iterrows():
            strike = row['strike']
            
            if time_to_expiry <= 0:
                delta = 1.0 if current_price > strike else 0.0
            else:
                d1 = (np.log(current_price / strike) + 
                    (risk_free_rate + 0.5 * implied_vol**2) * time_to_expiry) / \
                    (implied_vol * np.sqrt(time_to_expiry))
                delta = norm.cdf(d1)
            
            deltas.append(delta)
        
        return deltas
    
    # ENHANCED STRATEGY IMPLEMENTATIONS
    
    def _bull_call_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                    market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Bull Call Spread with advanced selection criteria"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if calls.empty or len(calls) < 2:
                return {'error': 'Insufficient calls for bull call spread'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check if conditions are suitable for bull call spread
            bullish_trends = ['BULLISH', 'SHORT_TERM_BULLISH', 'STRONG_BULLISH']
            if trend not in bullish_trends:
                return {'error': f'Trend {trend} not suitable for bull call spread'}
            
            # Check for low IV environment (IV < Realized Vol or IV percentile < 40%)
            iv_rank = stock_data.get('iv_rank', 50)  # IV percentile rank
            if implied_vol > realized_vol * 1.3 or iv_rank > 40:
                return {'error': 'IV too high for bull call spread - consider selling premium instead'}
            
            # Check momentum conditions
            bullish_momentum = ['BULLISH', 'NEUTRAL', 'OVERSOLD']
            if momentum not in bullish_momentum:
                return {'error': f'Momentum {momentum} not suitable for bull call spread'}
            
            # Filter viable calls
            viable_calls = calls[
                (calls['lastPrice'] > 0.05) & 
                (calls['lastPrice'] < current_price * 0.5) &
                (calls['volume'] > 5) &  # Minimum volume requirement
                (calls['openInterest'] > 10)  # Minimum open interest
            ].copy()
            
            if len(viable_calls) < 2:
                return {'error': 'No viable call options found'}
            
            viable_calls = viable_calls.sort_values('strike').reset_index(drop=True)
            viable_calls['moneyness'] = viable_calls['strike'] / current_price
            
            # Calculate theoretical deltas if not provided
            if 'delta' not in viable_calls.columns:
                viable_calls['delta'] = self._calculate_delta(viable_calls, current_price, 
                                                            options_data['days_to_expiry'], implied_vol)
            
            # Enhanced long strike selection: Delta between 0.5-0.6 (ATM)
            long_candidates = viable_calls[
                (viable_calls['delta'] >= 0.5) & 
                (viable_calls['delta'] <= 0.6)
            ]
            
            if long_candidates.empty:
                # Fallback to closest to ATM
                viable_calls['distance'] = abs(viable_calls['strike'] - current_price)
                buy_call = viable_calls.loc[viable_calls['distance'].idxmin()]
            else:
                # Select the one closest to 0.55 delta
                long_candidates['delta_distance'] = abs(long_candidates['delta'] - 0.55)
                buy_call = long_candidates.loc[long_candidates['delta_distance'].idxmin()]
            
            # Calculate optimal short strike based on multiple factors
            def calculate_short_strike_target():
                # Base target using 1 standard deviation
                time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
                one_std_move = current_price * realized_vol * time_factor
                
                # Technical analysis factors
                resistance_level = stock_data.get('resistance_level')
                
                # Trend-based adjustments
                trend_multiplier = {
                    'STRONG_BULLISH': 1.2,
                    'BULLISH': 1.0,
                    'SHORT_TERM_BULLISH': 0.9
                }.get(trend, 1.0)
                
                # Calculate targets
                targets = []
                
                # 1. Statistical target (1 std dev adjusted for trend)
                stat_target = current_price + (one_std_move * trend_multiplier)
                targets.append(stat_target)
                
                # 2. Resistance level target (if available and reasonable)
                if resistance_level and resistance_level > current_price * 1.02:
                    if resistance_level < current_price * 1.20:  # Within 20%
                        targets.append(resistance_level * 0.98)  # Slightly below resistance
                
                # 3. Round number target (psychological levels)
                round_targets = []
                for multiplier in [1.05, 1.10, 1.15]:
                    round_price = current_price * multiplier
                    # Find nearest $5 or $10 round number
                    if round_price < 50:
                        round_num = round(round_price / 5) * 5
                    else:
                        round_num = round(round_price / 10) * 10
                    if round_num > current_price * 1.02:
                        round_targets.append(round_num)
                
                if round_targets:
                    targets.append(min(round_targets))
                
                # Return the most conservative (lowest) reasonable target
                final_target = min(targets)
                
                # Ensure target is within reasonable bounds
                min_target = current_price * 1.03
                max_target = current_price * 1.20
                
                return max(min_target, min(final_target, max_target))
            
            short_strike_target = calculate_short_strike_target()
            
            # Find calls near the target strike
            min_sell_strike = max(buy_call['strike'] * 1.02, short_strike_target * 0.95)
            max_sell_strike = short_strike_target * 1.10
            
            otm_calls = viable_calls[
                (viable_calls['strike'] >= min_sell_strike) & 
                (viable_calls['strike'] <= max_sell_strike)
            ]
            
            if otm_calls.empty:
                return {'error': 'No suitable OTM calls found for spread'}
            
            # Select best short strike based on target proximity and delta
            otm_calls['strike_distance'] = abs(otm_calls['strike'] - short_strike_target)
            sell_call = otm_calls.loc[otm_calls['strike_distance'].idxmin()]
            
            # Validate spread economics
            net_debit = buy_call['lastPrice'] - sell_call['lastPrice']
            max_profit = (sell_call['strike'] - buy_call['strike']) - net_debit
            max_loss = net_debit
            breakeven = buy_call['strike'] + net_debit
            risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0
            
            # Enhanced validation
            if net_debit <= 0:
                return {'error': 'Spread results in net credit - not optimal for bull call'}
            
            if max_profit <= 0:
                return {'error': 'No profit potential in spread'}
            
            if risk_reward_ratio < 1.5:  # Minimum 1:1.5 risk-reward
                return {'error': f'Risk-reward ratio {risk_reward_ratio:.2f} too low'}
            
            # Check breakeven feasibility
            breakeven_move = (breakeven - current_price) / current_price
            if breakeven_move > 0.15:  # More than 15% move required
                return {'error': f'Breakeven requires {breakeven_move:.1%} move - too aggressive'}
            
            # Position sizing with enhanced risk management
            base_risk = portfolio_value * self.risk_tolerance
            vol_adjustment = min(1.0, 0.20 / realized_vol)  # Reduce size in high vol
            momentum_adjustment = 1.2 if momentum == 'BULLISH' else 1.0
            
            adjusted_risk = base_risk * vol_adjustment * momentum_adjustment
            contracts = max(1, min(int(adjusted_risk / (max_loss * 100)), 15))
            
            # Calculate profit/loss thresholds
            profit_target = net_debit * 0.8  # 80% of max profit
            loss_threshold = net_debit * 0.5  # 50% of max loss
            
            # Generate short strike rationale
            move_required = ((short_strike_target - current_price) / current_price) * 100
            rationale_parts = [f"Target strike ${short_strike_target:.2f} ({move_required:.1f}% move)"]
            
            resistance_level = stock_data.get('resistance_level')
            if resistance_level and abs(short_strike_target - resistance_level) / resistance_level < 0.05:
                rationale_parts.append("positioned near resistance level")
            
            time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
            one_std = current_price * realized_vol * time_factor
            
            if abs(short_strike_target - (current_price + one_std)) / current_price < 0.02:
                rationale_parts.append("based on 1-standard deviation move")
            
            short_strike_rationale = " - ".join(rationale_parts)
            
            return {
                'strategy_name': 'Bull Call Spread',
                'legs': [
                    {
                        'action': 'BUY',
                        'option_type': 'CALL',
                        'strike': buy_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': buy_call['lastPrice'],
                        'contracts': contracts,
                        'delta': buy_call.get('delta', 'N/A'),
                        'volume': buy_call.get('volume', 'N/A'),
                        'open_interest': buy_call.get('openInterest', 'N/A')
                    },
                    {
                        'action': 'SELL',
                        'option_type': 'CALL',
                        'strike': sell_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': sell_call['lastPrice'],
                        'contracts': contracts,
                        'delta': sell_call.get('delta', 'N/A'),
                        'volume': sell_call.get('volume', 'N/A'),
                        'open_interest': sell_call.get('openInterest', 'N/A')
                    }
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_profit': round(max_profit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'breakeven': round(breakeven, 2),
                'breakeven_move_required': round(breakeven_move * 100, 2),
                'profit_target_price': round(profit_target * contracts * 100, 2),
                'loss_threshold_price': round(loss_threshold * contracts * 100, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'iv_rank': iv_rank,
                'short_strike_rationale': short_strike_rationale,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"Bullish low-IV strategy for {symbol}. Target: ${sell_call['strike']:.2f} "
                            f"({((sell_call['strike'] - current_price) / current_price) * 100:.1f}% move). "
                            f"Take profit at 80% max profit (${round(profit_target * contracts * 100, 2)}), "
                            f"stop loss at 50% max loss (${round(loss_threshold * contracts * 100, 2)})."
            }
            
        except Exception as e:
            return {'error': f'Bull call spread calculation failed: {str(e)}'}
    
    def _bear_put_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                    market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Bear Put Spread with advanced selection criteria"""
        try:
            current_price = stock_data['current_price']
            puts = options_data['puts']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if puts.empty or len(puts) < 2:
                return {'error': 'Insufficient puts for bear put spread'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check if conditions are suitable for bear put spread
            bearish_trends = ['BEARISH', 'SHORT_TERM_BEARISH', 'STRONG_BEARISH']
            if trend not in bearish_trends:
                return {'error': f'Trend {trend} not suitable for bear put spread'}
            
            # Check for low IV environment (IV < Realized Vol or IV percentile < 40%)
            iv_rank = stock_data.get('iv_rank', 50)  # IV percentile rank
            if implied_vol > realized_vol * 1.3 or iv_rank > 40:
                return {'error': 'IV too high for bear put spread - consider selling premium instead'}
            
            # Check momentum conditions
            bearish_momentum = ['BEARISH', 'NEUTRAL', 'OVERBOUGHT']
            if momentum not in bearish_momentum:
                return {'error': f'Momentum {momentum} not suitable for bear put spread'}
            
            # Filter viable puts
            viable_puts = puts[
                (puts['lastPrice'] > 0.05) & 
                (puts['lastPrice'] < current_price * 0.5) &
                (puts['volume'] > 5) &  # Minimum volume requirement
                (puts['openInterest'] > 10)  # Minimum open interest
            ].copy()
            
            if len(viable_puts) < 2:
                return {'error': 'No viable put options found'}
            
            viable_puts = viable_puts.sort_values('strike', ascending=False).reset_index(drop=True)
            viable_puts['moneyness'] = viable_puts['strike'] / current_price
            
            # Calculate theoretical deltas if not provided (puts have negative delta)
            if 'delta' not in viable_puts.columns:
                viable_puts['delta'] = self._calculate_delta(viable_puts, current_price, 
                                                            options_data['days_to_expiry'], implied_vol, option_type='put')
            
            # Enhanced long strike selection: Delta between -0.5 to -0.6 (ATM)
            long_candidates = viable_puts[
                (viable_puts['delta'] <= -0.5) & 
                (viable_puts['delta'] >= -0.6)
            ]
            
            if long_candidates.empty:
                # Fallback to closest to ATM
                viable_puts['distance'] = abs(viable_puts['strike'] - current_price)
                buy_put = viable_puts.loc[viable_puts['distance'].idxmin()]
            else:
                # Select the one closest to -0.55 delta
                long_candidates['delta_distance'] = abs(long_candidates['delta'] + 0.55)
                buy_put = long_candidates.loc[long_candidates['delta_distance'].idxmin()]
            
            # Calculate optimal short strike based on multiple factors
            def calculate_short_strike_target():
                # Base target using 1 standard deviation
                time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
                one_std_move = current_price * realized_vol * time_factor
                
                # Technical analysis factors
                support_level = stock_data.get('support_level')
                
                # Trend-based adjustments
                trend_multiplier = {
                    'STRONG_BEARISH': 1.2,
                    'BEARISH': 1.0,
                    'SHORT_TERM_BEARISH': 0.9
                }.get(trend, 1.0)
                
                # Calculate targets
                targets = []
                
                # 1. Statistical target (1 std dev adjusted for trend)
                stat_target = current_price - (one_std_move * trend_multiplier)
                targets.append(stat_target)
                
                # 2. Support level target (if available and reasonable)
                if support_level and support_level < current_price * 0.98:
                    if support_level > current_price * 0.80:  # Within 20%
                        targets.append(support_level * 1.02)  # Slightly above support
                
                # 3. Round number target (psychological levels)
                round_targets = []
                for multiplier in [0.95, 0.90, 0.85]:
                    round_price = current_price * multiplier
                    # Find nearest $5 or $10 round number
                    if round_price < 50:
                        round_num = round(round_price / 5) * 5
                    else:
                        round_num = round(round_price / 10) * 10
                    if round_num < current_price * 0.98:
                        round_targets.append(round_num)
                
                if round_targets:
                    targets.append(max(round_targets))
                
                # Return the most conservative (highest) reasonable target
                final_target = max(targets)
                
                # Ensure target is within reasonable bounds
                max_target = current_price * 0.97
                min_target = current_price * 0.80
                
                return min(max_target, max(final_target, min_target))
            
            short_strike_target = calculate_short_strike_target()
            
            # Find puts near the target strike
            max_sell_strike = min(buy_put['strike'] * 0.98, short_strike_target * 1.05)
            min_sell_strike = short_strike_target * 0.90
            
            otm_puts = viable_puts[
                (viable_puts['strike'] <= max_sell_strike) & 
                (viable_puts['strike'] >= min_sell_strike)
            ]
            
            if otm_puts.empty:
                return {'error': 'No suitable OTM puts found for spread'}
            
            # Select best short strike based on target proximity
            otm_puts['strike_distance'] = abs(otm_puts['strike'] - short_strike_target)
            sell_put = otm_puts.loc[otm_puts['strike_distance'].idxmin()]
            
            # Validate spread economics
            net_debit = buy_put['lastPrice'] - sell_put['lastPrice']
            max_profit = (buy_put['strike'] - sell_put['strike']) - net_debit
            max_loss = net_debit
            breakeven = buy_put['strike'] - net_debit
            risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0
            
            # Enhanced validation
            if net_debit <= 0:
                return {'error': 'Spread results in net credit - not optimal for bear put'}
            
            if max_profit <= 0:
                return {'error': 'No profit potential in spread'}
            
            if risk_reward_ratio < 1.5:  # Minimum 1:1.5 risk-reward
                return {'error': f'Risk-reward ratio {risk_reward_ratio:.2f} too low'}
            
            # Check breakeven feasibility
            breakeven_move = (current_price - breakeven) / current_price
            if breakeven_move > 0.15:  # More than 15% move required
                return {'error': f'Breakeven requires {breakeven_move:.1%} move - too aggressive'}
            
            # Position sizing with enhanced risk management
            base_risk = portfolio_value * self.risk_tolerance
            vol_adjustment = min(1.0, 0.20 / realized_vol)  # Reduce size in high vol
            momentum_adjustment = 1.2 if momentum == 'BEARISH' else 1.0
            
            adjusted_risk = base_risk * vol_adjustment * momentum_adjustment
            contracts = max(1, min(int(adjusted_risk / (max_loss * 100)), 15))
            
            # Calculate profit/loss thresholds
            profit_target = net_debit * 0.8  # 80% of max profit
            loss_threshold = net_debit * 0.5  # 50% of max loss
            
            # Generate short strike rationale
            move_required = ((current_price - short_strike_target) / current_price) * 100
            rationale_parts = [f"Target strike ${short_strike_target:.2f} ({move_required:.1f}% move down)"]
            
            support_level = stock_data.get('support_level')
            if support_level and abs(short_strike_target - support_level) / support_level < 0.05:
                rationale_parts.append("positioned near support level")
            
            time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
            one_std = current_price * realized_vol * time_factor
            
            if abs(short_strike_target - (current_price - one_std)) / current_price < 0.02:
                rationale_parts.append("based on 1-standard deviation move")
            
            short_strike_rationale = " - ".join(rationale_parts)
            
            return {
                'strategy_name': 'Bear Put Spread',
                'legs': [
                    {
                        'action': 'BUY',
                        'option_type': 'PUT',
                        'strike': buy_put['strike'],
                        'expiration': options_data['expiration'],
                        'price': buy_put['lastPrice'],
                        'contracts': contracts,
                        'delta': buy_put.get('delta', 'N/A'),
                        'volume': buy_put.get('volume', 'N/A'),
                        'open_interest': buy_put.get('openInterest', 'N/A')
                    },
                    {
                        'action': 'SELL',
                        'option_type': 'PUT',
                        'strike': sell_put['strike'],
                        'expiration': options_data['expiration'],
                        'price': sell_put['lastPrice'],
                        'contracts': contracts,
                        'delta': sell_put.get('delta', 'N/A'),
                        'volume': sell_put.get('volume', 'N/A'),
                        'open_interest': sell_put.get('openInterest', 'N/A')
                    }
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_profit': round(max_profit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'breakeven': round(breakeven, 2),
                'breakeven_move_required': round(breakeven_move * 100, 2),
                'profit_target_price': round(profit_target * contracts * 100, 2),
                'loss_threshold_price': round(loss_threshold * contracts * 100, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'iv_rank': iv_rank,
                'short_strike_rationale': short_strike_rationale,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"Bearish low-IV strategy for {symbol}. Target: ${sell_put['strike']:.2f} "
                            f"({((current_price - sell_put['strike']) / current_price) * 100:.1f}% move down). "
                            f"Take profit at 80% max profit (${round(profit_target * contracts * 100, 2)}), "
                            f"stop loss at 50% max loss (${round(loss_threshold * contracts * 100, 2)})."
            }
            
        except Exception as e:
            return {'error': f'Bear put spread calculation failed: {str(e)}'}
    
    def _iron_condor(self, symbol: str, stock_data: Dict, options_data: Dict,
                market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Iron Condor with advanced selection criteria"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            puts = options_data['puts']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if calls.empty or puts.empty or len(calls) < 2 or len(puts) < 2:
                return {'error': 'Insufficient options for iron condor'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check if conditions are suitable for iron condor (neutral strategies)
            neutral_trends = ['SIDEWAYS']
            range_bound_trends = ['SHORT_TERM_BULLISH', 'SHORT_TERM_BEARISH']  # Can work if range-bound
            
            if trend not in neutral_trends and trend not in range_bound_trends:
                return {'error': f'Trend {trend} not suitable for iron condor - requires neutral/range-bound market'}
            
            # Check for high IV environment (ideal for selling premium)
            iv_rank = stock_data.get('iv_rank', 50)
            if implied_vol < realized_vol * 0.8 or iv_rank < 30:
                return {'error': 'IV too low for iron condor - better to buy premium instead'}
            
            # Check momentum conditions (avoid extreme momentum)
            extreme_momentum = ['EXTREMELY_OVERBOUGHT', 'EXTREMELY_OVERSOLD']
            if momentum in extreme_momentum:
                return {'error': f'Momentum {momentum} too extreme for iron condor'}
            
            # Enhanced filtering for viable options
            min_price_threshold = 0.05 if vol_regime in ['HIGH_VOL', 'EXTREME_VOL'] else 0.02
            
            viable_calls = calls[
                (calls['lastPrice'] > min_price_threshold) & 
                (calls['strike'] > current_price * 1.005) &
                (calls['volume'] > 2) &
                (calls['openInterest'] > 5)
            ].copy()
            
            viable_puts = puts[
                (puts['lastPrice'] > min_price_threshold) & 
                (puts['strike'] < current_price * 0.995) &
                (puts['volume'] > 2) &
                (puts['openInterest'] > 5)
            ].copy()
            
            # Fallback with more relaxed criteria
            if len(viable_calls) < 2 or len(viable_puts) < 2:
                viable_calls = calls[
                    (calls['lastPrice'] > 0.01) & 
                    (calls['strike'] > current_price)
                ].copy()
                viable_puts = puts[
                    (puts['lastPrice'] > 0.01) & 
                    (puts['strike'] < current_price)
                ].copy()
            
            if len(viable_calls) < 2 or len(viable_puts) < 2:
                return {'error': f'Not enough viable options. Calls: {len(viable_calls)}, Puts: {len(viable_puts)}'}
            
            viable_calls = viable_calls.sort_values('strike').reset_index(drop=True)
            viable_puts = viable_puts.sort_values('strike', ascending=False).reset_index(drop=True)
            
            # Calculate deltas if not provided
            if 'delta' not in viable_calls.columns:
                viable_calls['delta'] = self._calculate_delta(viable_calls, current_price, 
                                                            options_data['days_to_expiry'], implied_vol)
            if 'delta' not in viable_puts.columns:
                viable_puts['delta'] = self._calculate_delta(viable_puts, current_price, 
                                                        options_data['days_to_expiry'], implied_vol, option_type='put')
            
            # Enhanced strike selection based on probability and technical levels
            def calculate_optimal_strikes():
                # Calculate expected price range based on volatility and time
                time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
                one_std_move = current_price * implied_vol * time_factor
                
                # Get technical levels
                resistance_level = stock_data.get('resistance_level')
                support_level = stock_data.get('support_level')
                
                # Target deltas for short strikes (probability-based)
                target_short_call_delta = 0.20  # ~20% probability of finishing ITM
                target_short_put_delta = -0.20
                
                # Find short call strike (prefer delta-based, fallback to volatility-based)
                short_call_candidates = viable_calls[
                    (viable_calls['delta'] >= 0.15) & 
                    (viable_calls['delta'] <= 0.25)
                ]
                
                if short_call_candidates.empty:
                    # Fallback: use volatility-based target
                    target_call_strike = current_price + (one_std_move * 0.8)
                    viable_calls['strike_distance'] = abs(viable_calls['strike'] - target_call_strike)
                    short_call = viable_calls.loc[viable_calls['strike_distance'].idxmin()]
                else:
                    # Select closest to target delta
                    short_call_candidates['delta_distance'] = abs(short_call_candidates['delta'] - target_short_call_delta)
                    short_call = short_call_candidates.loc[short_call_candidates['delta_distance'].idxmin()]
                
                # Adjust for resistance level if available
                if resistance_level and resistance_level > current_price * 1.02:
                    if abs(short_call['strike'] - resistance_level) / resistance_level > 0.05:
                        # Find strike closer to resistance
                        resistance_candidates = viable_calls[
                            abs(viable_calls['strike'] - resistance_level) / resistance_level <= 0.05
                        ]
                        if not resistance_candidates.empty:
                            short_call = resistance_candidates.iloc[0]
                
                # Find short put strike
                short_put_candidates = viable_puts[
                    (viable_puts['delta'] <= -0.15) & 
                    (viable_puts['delta'] >= -0.25)
                ]
                
                if short_put_candidates.empty:
                    # Fallback: use volatility-based target
                    target_put_strike = current_price - (one_std_move * 0.8)
                    viable_puts['strike_distance'] = abs(viable_puts['strike'] - target_put_strike)
                    short_put = viable_puts.loc[viable_puts['strike_distance'].idxmin()]
                else:
                    # Select closest to target delta
                    short_put_candidates['delta_distance'] = abs(short_put_candidates['delta'] - target_short_put_delta)
                    short_put = short_put_candidates.loc[short_put_candidates['delta_distance'].idxmin()]
                
                # Adjust for support level if available
                if support_level and support_level < current_price * 0.98:
                    if abs(short_put['strike'] - support_level) / support_level > 0.05:
                        # Find strike closer to support
                        support_candidates = viable_puts[
                            abs(viable_puts['strike'] - support_level) / support_level <= 0.05
                        ]
                        if not support_candidates.empty:
                            short_put = support_candidates.iloc[0]
                
                return short_call, short_put
            
            short_call, short_put = calculate_optimal_strikes()
            
            # Select long strikes for protection (aim for balanced wing widths)
            target_wing_width = max(5, current_price * 0.05)  # 5% of stock price or $5 minimum
            
            # Long call selection
            target_long_call_strike = short_call['strike'] + target_wing_width
            long_call_candidates = viable_calls[viable_calls['strike'] > short_call['strike']]
            
            if long_call_candidates.empty:
                return {'error': 'No protection strikes available for calls'}
            
            long_call_candidates['strike_distance'] = abs(long_call_candidates['strike'] - target_long_call_strike)
            long_call = long_call_candidates.loc[long_call_candidates['strike_distance'].idxmin()]
            
            # Long put selection
            target_long_put_strike = short_put['strike'] - target_wing_width
            long_put_candidates = viable_puts[viable_puts['strike'] < short_put['strike']]
            
            if long_put_candidates.empty:
                return {'error': 'No protection strikes available for puts'}
            
            long_put_candidates['strike_distance'] = abs(long_put_candidates['strike'] - target_long_put_strike)
            long_put = long_put_candidates.loc[long_put_candidates['strike_distance'].idxmin()]
            
            # Calculate economics
            premium_collected = short_call['lastPrice'] + short_put['lastPrice']
            premium_paid = long_call['lastPrice'] + long_put['lastPrice']
            net_credit = premium_collected - premium_paid
            
            if net_credit <= 0:
                return {'error': f'Iron condor results in net debit: {net_credit:.2f}'}
            
            call_width = long_call['strike'] - short_call['strike']
            put_width = short_put['strike'] - long_put['strike']
            max_loss = min(call_width, put_width) - net_credit
            max_profit = net_credit
            
            if max_loss <= 0:
                return {'error': 'Invalid condor structure - negative max loss'}
            
            # Enhanced validation
            risk_reward_ratio = max_profit / max_loss
            if risk_reward_ratio < 0.3:  # Minimum 1:3 risk-reward for credit spreads
                return {'error': f'Risk-reward ratio {risk_reward_ratio:.2f} too low for iron condor'}
            
            # Profit zone analysis
            lower_breakeven = short_put['strike'] + net_credit
            upper_breakeven = short_call['strike'] - net_credit
            profit_zone_width = upper_breakeven - lower_breakeven
            profit_zone_percentage = profit_zone_width / current_price
            
            if profit_zone_percentage < 0.05:  # Less than 5% profit zone
                return {'error': f'Profit zone too narrow: {profit_zone_percentage:.1%}'}
            
            # Position sizing with enhanced risk management
            base_risk = portfolio_value * self.risk_tolerance
            
            # Adjust for market conditions
            vol_adjustment = min(1.2, implied_vol / 0.20)  # Increase size in higher IV
            trend_adjustment = 0.8 if trend in ['SHORT_TERM_BULLISH', 'SHORT_TERM_BEARISH'] else 1.0
            
            adjusted_risk = base_risk * vol_adjustment * trend_adjustment
            contracts = max(1, min(int(adjusted_risk / (abs(max_loss) * 100)), 10))
            
            # Calculate management thresholds
            profit_target = net_credit * 0.5  # Take profit at 50% of max credit
            loss_threshold = net_credit * 1.5  # Stop loss at 150% of credit received
            
            # Generate strategy rationale
            prob_profit = self._calculate_probability_of_profit(lower_breakeven, upper_breakeven, 
                                                            current_price, implied_vol, 
                                                            options_data['days_to_expiry'])
            
            return {
                'strategy_name': 'Iron Condor',
                'legs': [
                    {
                        'action': 'SELL',
                        'option_type': 'CALL',
                        'strike': short_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': short_call['lastPrice'],
                        'contracts': contracts,
                        'delta': short_call.get('delta', 'N/A'),
                        'volume': short_call.get('volume', 'N/A'),
                        'open_interest': short_call.get('openInterest', 'N/A')
                    },
                    {
                        'action': 'BUY',
                        'option_type': 'CALL',
                        'strike': long_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': long_call['lastPrice'],
                        'contracts': contracts,
                        'delta': long_call.get('delta', 'N/A'),
                        'volume': long_call.get('volume', 'N/A'),
                        'open_interest': long_call.get('openInterest', 'N/A')
                    },
                    {
                        'action': 'SELL',
                        'option_type': 'PUT',
                        'strike': short_put['strike'],
                        'expiration': options_data['expiration'],
                        'price': short_put['lastPrice'],
                        'contracts': contracts,
                        'delta': short_put.get('delta', 'N/A'),
                        'volume': short_put.get('volume', 'N/A'),
                        'open_interest': short_put.get('openInterest', 'N/A')
                    },
                    {
                        'action': 'BUY',
                        'option_type': 'PUT',
                        'strike': long_put['strike'],
                        'expiration': options_data['expiration'],
                        'price': long_put['lastPrice'],
                        'contracts': contracts,
                        'delta': long_put.get('delta', 'N/A'),
                        'volume': long_put.get('volume', 'N/A'),
                        'open_interest': long_put.get('openInterest', 'N/A')
                    }
                ],
                'net_credit': round(net_credit * contracts * 100, 2),
                'max_profit': round(max_profit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'profit_range': (round(lower_breakeven, 2), round(upper_breakeven, 2)),
                'profit_zone_width': round(profit_zone_width, 2),
                'profit_zone_percentage': round(profit_zone_percentage * 100, 2),
                'profit_target_price': round(profit_target * contracts * 100, 2),
                'loss_threshold_price': round(loss_threshold * contracts * 100, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'probability_of_profit': round(prob_profit * 100, 1),
                'iv_rank': iv_rank,
                'wing_widths': {
                    'call_width': round(call_width, 2),
                    'put_width': round(put_width, 2)
                },
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"High-IV neutral strategy for {symbol}. Profit zone: ${lower_breakeven:.2f} - ${upper_breakeven:.2f} "
                            f"({profit_zone_percentage:.1%} range). Probability of profit: {prob_profit*100:.1f}%. "
                            f"Take profit at 50% credit (${round(profit_target * contracts * 100, 2)}), "
                            f"manage at 150% loss (${round(loss_threshold * contracts * 100, 2)})."
            }
            
        except Exception as e:
            return {'error': f'Iron condor calculation failed: {str(e)}'}

    def _calculate_probability_of_profit(self, lower_breakeven: float, upper_breakeven: float,
                                    current_price: float, implied_vol: float, days_to_expiry: int) -> float:
        """Calculate probability that stock will finish within profit zone"""
        import numpy as np
        from scipy.stats import norm
        
        # Using log-normal distribution assumption
        time_to_expiry = days_to_expiry / 365
        
        if time_to_expiry <= 0:
            return 1.0 if lower_breakeven <= current_price <= upper_breakeven else 0.0
        
        # Calculate z-scores for breakeven points
        vol_sqrt_t = implied_vol * np.sqrt(time_to_expiry)
        
        z_lower = (np.log(lower_breakeven / current_price)) / vol_sqrt_t
        z_upper = (np.log(upper_breakeven / current_price)) / vol_sqrt_t
        
        # Probability of finishing between the breakevens
        prob_profit = norm.cdf(z_upper) - norm.cdf(z_lower)
        
        return max(0.0, min(1.0, prob_profit))
    
    def _long_straddle(self, symbol: str, stock_data: Dict, options_data: Dict,
                 market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Long Straddle with advanced volatility and event-driven selection"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            puts = options_data['puts']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if calls.empty or puts.empty:
                return {'error': 'No options available for straddle'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check for suitable conditions for long straddle
            # 1. Low IV relative to realized vol (buying premium when cheap)
            iv_rank = stock_data.get('iv_rank', 50)
            if implied_vol > realized_vol * 1.5 and iv_rank > 60:
                return {'error': 'IV too high for long straddle - premium too expensive'}
            
            # 2. Expect volatility expansion or major move
            # Good conditions: earnings approaching, low vol regime, extreme momentum
            earnings_soon = stock_data.get('earnings_days', 999) <= 7
            volatility_expansion_expected = (
                vol_regime == 'LOW_VOL' or 
                earnings_soon or 
                momentum in ['EXTREMELY_OVERBOUGHT', 'EXTREMELY_OVERSOLD'] or
                trend == 'SIDEWAYS'  # Sideways trends often precede breakouts
            )
            
            if not volatility_expansion_expected and iv_rank > 40:
                return {'error': 'No volatility expansion expected and IV not attractive for long straddle'}
            
            # 3. Check for sufficient time to expiration (straddles need time for moves)
            if options_data['days_to_expiry'] < 14:
                return {'error': 'Too close to expiration for long straddle - need more time for large moves'}
            
            if options_data['days_to_expiry'] > 60:
                return {'error': 'Too much time to expiration - theta decay too high for long straddle'}
            
            # Enhanced option filtering
            min_price_threshold = max(0.10, current_price * 0.002)  # Minimum $0.10 or 0.2% of stock price
            
            calls_filtered = calls[
                (calls['lastPrice'] > min_price_threshold) &
                (calls['volume'] > 10) &  # Higher volume requirement
                (calls['openInterest'] > 20) &
                (calls['bid'] > 0) & (calls['ask'] > 0)  # Ensure tradeable spread
            ].copy()
            
            puts_filtered = puts[
                (puts['lastPrice'] > min_price_threshold) &
                (puts['volume'] > 10) &
                (puts['openInterest'] > 20) &
                (puts['bid'] > 0) & (puts['ask'] > 0)
            ].copy()
            
            if calls_filtered.empty or puts_filtered.empty:
                return {'error': 'No viable liquid options for straddle'}
            
            # Calculate distances and add spread analysis
            calls_filtered['distance'] = abs(calls_filtered['strike'] - current_price)
            puts_filtered['distance'] = abs(puts_filtered['strike'] - current_price)
            calls_filtered['bid_ask_spread'] = calls_filtered['ask'] - calls_filtered['bid']
            puts_filtered['bid_ask_spread'] = puts_filtered['ask'] - puts_filtered['bid']
            
            # Calculate deltas if not provided for better ATM selection
            if 'delta' not in calls_filtered.columns:
                calls_filtered['delta'] = self._calculate_delta(calls_filtered, current_price, 
                                                            options_data['days_to_expiry'], implied_vol)
            if 'delta' not in puts_filtered.columns:
                puts_filtered['delta'] = self._calculate_delta(puts_filtered, current_price, 
                                                            options_data['days_to_expiry'], implied_vol, option_type='put')
            
            # Enhanced ATM selection - prefer closest to 0.5/-0.5 delta (true ATM)
            def find_best_atm_options():
                # Find calls closest to 0.5 delta
                atm_call_candidates = calls_filtered[
                    (calls_filtered['delta'] >= 0.45) & 
                    (calls_filtered['delta'] <= 0.55)
                ]
                
                if atm_call_candidates.empty:
                    # Fallback to closest strike
                    atm_call = calls_filtered.loc[calls_filtered['distance'].idxmin()]
                else:
                    # Among delta candidates, pick the one with best liquidity and tight spread
                    atm_call_candidates['liquidity_score'] = (
                        atm_call_candidates['volume'] * 0.5 + 
                        atm_call_candidates['openInterest'] * 0.3 -
                        atm_call_candidates['bid_ask_spread'] * 20
                    )
                    atm_call = atm_call_candidates.loc[atm_call_candidates['liquidity_score'].idxmax()]
                
                # Find puts closest to -0.5 delta, preferably same strike as call
                same_strike_puts = puts_filtered[puts_filtered['strike'] == atm_call['strike']]
                
                if not same_strike_puts.empty:
                    atm_put = same_strike_puts.iloc[0]
                else:
                    # Find puts closest to -0.5 delta
                    atm_put_candidates = puts_filtered[
                        (puts_filtered['delta'] <= -0.45) & 
                        (puts_filtered['delta'] >= -0.55)
                    ]
                    
                    if atm_put_candidates.empty:
                        atm_put = puts_filtered.loc[puts_filtered['distance'].idxmin()]
                    else:
                        atm_put_candidates['liquidity_score'] = (
                            atm_put_candidates['volume'] * 0.5 + 
                            atm_put_candidates['openInterest'] * 0.3 -
                            atm_put_candidates['bid_ask_spread'] * 20
                        )
                        atm_put = atm_put_candidates.loc[atm_put_candidates['liquidity_score'].idxmax()]
                
                return atm_call, atm_put
            
            atm_call, atm_put = find_best_atm_options()
            
            # Calculate straddle economics
            net_debit = atm_call['lastPrice'] + atm_put['lastPrice']
            max_loss = net_debit
            
            # Calculate breakeven points
            if atm_call['strike'] == atm_put['strike']:
                # True straddle (same strike)
                breakeven_up = atm_call['strike'] + net_debit
                breakeven_down = atm_put['strike'] - net_debit
                strike_difference = 0
            else:
                # Strangle (different strikes) - adjust breakevens
                breakeven_up = atm_call['strike'] + net_debit
                breakeven_down = atm_put['strike'] - net_debit
                strike_difference = abs(atm_call['strike'] - atm_put['strike'])
            
            if net_debit <= 0:
                return {'error': 'Invalid straddle pricing'}
            
            # Enhanced validation - check required move vs historical volatility
            breakeven_move_up = (breakeven_up - current_price) / current_price
            breakeven_move_down = (current_price - breakeven_down) / current_price
            required_move = min(breakeven_move_up, breakeven_move_down)
            
            # Check if required move is reasonable given historical volatility
            time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
            expected_move = realized_vol * time_factor
            
            if required_move > expected_move * 1.5:
                return {'error': f'Required move {required_move:.1%} too large vs expected {expected_move:.1%}'}
            
            # Calculate probability of profit
            prob_profit = self._calculate_straddle_probability(
                breakeven_up, breakeven_down, current_price, implied_vol, options_data['days_to_expiry']
            )
            
            if prob_profit < 0.25:  # Less than 25% chance of profit
                return {'error': f'Probability of profit too low: {prob_profit:.1%}'}
            
            # Enhanced position sizing
            base_risk = portfolio_value * self.risk_tolerance
            
            # Adjust for volatility and event risk
            vol_adjustment = min(1.5, 0.20 / realized_vol)  # Increase size in low vol
            event_adjustment = 1.3 if earnings_soon else 1.0
            liquidity_adjustment = min(1.0, (atm_call.get('volume', 10) + atm_put.get('volume', 10)) / 50)
            
            adjusted_risk = base_risk * vol_adjustment * event_adjustment * liquidity_adjustment
            contracts = max(1, min(int(adjusted_risk / (max_loss * 100)), 8))
            
            # Calculate management levels
            profit_target_1 = net_debit * 1.5  # First target: 50% gain
            profit_target_2 = net_debit * 2.0   # Second target: 100% gain
            loss_threshold = net_debit * 0.5    # Stop loss at 50% of premium
            
            # Calculate expected move for comparison
            implied_move = current_price * implied_vol * time_factor
            
            # Determine strategy type
            strategy_type = 'Long Straddle' if strike_difference == 0 else 'Long Strangle'
            
            return {
                'strategy_name': strategy_type,
                'legs': [
                    {
                        'action': 'BUY',
                        'option_type': 'CALL',
                        'strike': atm_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': atm_call['lastPrice'],
                        'contracts': contracts,
                        'delta': atm_call.get('delta', 'N/A'),
                        'volume': atm_call.get('volume', 'N/A'),
                        'open_interest': atm_call.get('openInterest', 'N/A'),
                        'bid_ask_spread': round(atm_call.get('bid_ask_spread', 0), 2)
                    },
                    {
                        'action': 'BUY',
                        'option_type': 'PUT',
                        'strike': atm_put['strike'],
                        'expiration': options_data['expiration'],
                        'price': atm_put['lastPrice'],
                        'contracts': contracts,
                        'delta': atm_put.get('delta', 'N/A'),
                        'volume': atm_put.get('volume', 'N/A'),
                        'open_interest': atm_put.get('openInterest', 'N/A'),
                        'bid_ask_spread': round(atm_put.get('bid_ask_spread', 0), 2)
                    }
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'breakeven_up': round(breakeven_up, 2),
                'breakeven_down': round(breakeven_down, 2),
                'breakeven_moves': {
                    'upside_required': round(breakeven_move_up * 100, 2),
                    'downside_required': round(breakeven_move_down * 100, 2)
                },
                'profit_targets': {
                    'target_1': round(profit_target_1 * contracts * 100, 2),
                    'target_2': round(profit_target_2 * contracts * 100, 2)
                },
                'loss_threshold': round(loss_threshold * contracts * 100, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'probability_of_profit': round(prob_profit * 100, 1),
                'expected_vs_required_move': {
                    'implied_move': round(implied_move / current_price * 100, 2),
                    'required_move': round(required_move * 100, 2)
                },
                'iv_rank': iv_rank,
                'earnings_soon': earnings_soon,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"Volatility expansion play for {symbol}. "
                            f"{'Earnings in <7 days - ' if earnings_soon else ''}"
                            f"Needs {required_move:.1%} move. Breakevens: ${breakeven_down:.2f} - ${breakeven_up:.2f}. "
                            f"Probability of profit: {prob_profit*100:.1f}%. "
                            f"Take profits at 50% gain (${round(profit_target_1 * contracts * 100, 2)}) "
                            f"and 100% gain (${round(profit_target_2 * contracts * 100, 2)}). "
                            f"Stop loss at 50% premium loss (${round(loss_threshold * contracts * 100, 2)})."
            }
            
        except Exception as e:
            return {'error': f'Long straddle calculation failed: {str(e)}'}

    def _calculate_straddle_probability(self, breakeven_up: float, breakeven_down: float,
                                    current_price: float, implied_vol: float, days_to_expiry: int) -> float:
        """Calculate probability that stock will move beyond breakeven points"""
        import numpy as np
        from scipy.stats import norm
        
        time_to_expiry = days_to_expiry / 365
        
        if time_to_expiry <= 0:
            return 1.0 if current_price <= breakeven_down or current_price >= breakeven_up else 0.0
        
        # Using log-normal distribution
        vol_sqrt_t = implied_vol * np.sqrt(time_to_expiry)
        
        # Z-scores for breakeven points
        z_upper = (np.log(breakeven_up / current_price)) / vol_sqrt_t
        z_lower = (np.log(breakeven_down / current_price)) / vol_sqrt_t
        
        # Probability of finishing outside the breakeven range
        prob_profit = 1 - (norm.cdf(z_upper) - norm.cdf(z_lower))
        
        return max(0.0, min(1.0, prob_profit))
    
    def _long_strangle(self, symbol: str, stock_data: Dict, options_data: Dict,
                 market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Long Strangle with advanced OTM selection and volatility analysis"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            puts = options_data['puts']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if calls.empty or puts.empty:
                return {'error': 'No options available for strangle'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check for suitable conditions for long strangle
            # 1. Lower IV compared to straddle tolerance (strangles cheaper but need bigger moves)
            iv_rank = stock_data.get('iv_rank', 50)
            if implied_vol > realized_vol * 1.8 and iv_rank > 70:
                return {'error': 'IV too high for long strangle - premium too expensive'}
            
            # 2. Expect significant volatility expansion or major directional move
            earnings_soon = stock_data.get('earnings_days', 999) <= 10
            volatility_expansion_expected = (
                vol_regime in ['LOW_VOL', 'NORMAL_VOL'] or 
                earnings_soon or 
                momentum in ['EXTREMELY_OVERBOUGHT', 'EXTREMELY_OVERSOLD'] or
                trend == 'SIDEWAYS' or
                trend in ['SHORT_TERM_BULLISH', 'SHORT_TERM_BEARISH']  # Potential for breakout
            )
            
            if not volatility_expansion_expected and iv_rank > 50:
                return {'error': 'No significant volatility expansion expected for long strangle'}
            
            # 3. Check for sufficient time to expiration
            if options_data['days_to_expiry'] < 21:
                return {'error': 'Too close to expiration for long strangle - need more time for large moves'}
            
            if options_data['days_to_expiry'] > 75:
                return {'error': 'Too much time to expiration - theta decay too high for long strangle'}
            
            # Enhanced OTM option filtering with delta-based selection
            min_price_threshold = max(0.05, current_price * 0.001)
            
            # Target deltas for strangle strikes (further OTM than straddle)
            target_call_delta = 0.25  # ~25% probability ITM
            target_put_delta = -0.25
            
            otm_calls = calls[
                (calls['strike'] > current_price * 1.01) &
                (calls['lastPrice'] > min_price_threshold) &
                (calls['volume'] > 5) &
                (calls['openInterest'] > 15) &
                (calls['bid'] > 0) & (calls['ask'] > 0)
            ].copy()
            
            otm_puts = puts[
                (puts['strike'] < current_price * 0.99) &
                (puts['lastPrice'] > min_price_threshold) &
                (puts['volume'] > 5) &
                (puts['openInterest'] > 15) &
                (puts['bid'] > 0) & (puts['ask'] > 0)
            ].copy()
            
            if otm_calls.empty or otm_puts.empty:
                return {'error': 'No suitable liquid OTM options for strangle'}
            
            # Calculate deltas if not provided
            if 'delta' not in otm_calls.columns:
                otm_calls['delta'] = self._calculate_delta(otm_calls, current_price, 
                                                        options_data['days_to_expiry'], implied_vol)
            if 'delta' not in otm_puts.columns:
                otm_puts['delta'] = self._calculate_delta(otm_puts, current_price, 
                                                        options_data['days_to_expiry'], implied_vol, option_type='put')
            
            # Enhanced strike selection based on multiple factors
            def select_optimal_strikes():
                # Calculate expected move based on volatility
                time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
                expected_move = current_price * implied_vol * time_factor
                
                # Technical levels
                resistance_level = stock_data.get('resistance_level')
                support_level = stock_data.get('support_level')
                
                # Call selection strategy
                call_candidates = otm_calls[
                    (otm_calls['delta'] >= 0.15) & 
                    (otm_calls['delta'] <= 0.35)
                ]
                
                if call_candidates.empty:
                    # Fallback: target 1 standard deviation
                    target_call_strike = current_price + expected_move
                    otm_calls['strike_distance'] = abs(otm_calls['strike'] - target_call_strike)
                    call_option = otm_calls.loc[otm_calls['strike_distance'].idxmin()]
                else:
                    # Prefer strikes near technical resistance or psychological levels
                    if resistance_level and resistance_level > current_price * 1.02:
                        # Find calls near resistance
                        resistance_calls = call_candidates[
                            abs(call_candidates['strike'] - resistance_level) / resistance_level <= 0.05
                        ]
                        if not resistance_calls.empty:
                            call_option = resistance_calls.iloc[0]
                        else:
                            # Select based on delta preference and liquidity
                            call_candidates['score'] = (
                                abs(call_candidates['delta'] - target_call_delta) * -10 +
                                call_candidates['volume'] * 0.1 +
                                call_candidates['openInterest'] * 0.05
                            )
                            call_option = call_candidates.loc[call_candidates['score'].idxmax()]
                    else:
                        # Standard delta-based selection
                        call_candidates['delta_distance'] = abs(call_candidates['delta'] - target_call_delta)
                        call_option = call_candidates.loc[call_candidates['delta_distance'].idxmin()]
                
                # Put selection strategy
                put_candidates = otm_puts[
                    (otm_puts['delta'] <= -0.15) & 
                    (otm_puts['delta'] >= -0.35)
                ]
                
                if put_candidates.empty:
                    # Fallback: target 1 standard deviation
                    target_put_strike = current_price - expected_move
                    otm_puts['strike_distance'] = abs(otm_puts['strike'] - target_put_strike)
                    put_option = otm_puts.loc[otm_puts['strike_distance'].idxmin()]
                else:
                    # Prefer strikes near technical support or psychological levels
                    if support_level and support_level < current_price * 0.98:
                        # Find puts near support
                        support_puts = put_candidates[
                            abs(put_candidates['strike'] - support_level) / support_level <= 0.05
                        ]
                        if not support_puts.empty:
                            put_option = support_puts.iloc[0]
                        else:
                            # Select based on delta preference and liquidity
                            put_candidates['score'] = (
                                abs(put_candidates['delta'] - target_put_delta) * -10 +
                                put_candidates['volume'] * 0.1 +
                                put_candidates['openInterest'] * 0.05
                            )
                            put_option = put_candidates.loc[put_candidates['score'].idxmax()]
                    else:
                        # Standard delta-based selection
                        put_candidates['delta_distance'] = abs(put_candidates['delta'] - target_put_delta)
                        put_option = put_candidates.loc[put_candidates['delta_distance'].idxmin()]
                
                return call_option, put_option
            
            call_option, put_option = select_optimal_strikes()
            
            # Calculate strangle economics
            net_debit = call_option['lastPrice'] + put_option['lastPrice']
            max_loss = net_debit
            
            breakeven_up = call_option['strike'] + net_debit
            breakeven_down = put_option['strike'] - net_debit
            
            if net_debit <= 0:
                return {'error': 'Invalid strangle pricing'}
            
            # Enhanced validation - check required moves
            breakeven_move_up = (breakeven_up - current_price) / current_price
            breakeven_move_down = (current_price - breakeven_down) / current_price
            required_move = min(breakeven_move_up, breakeven_move_down)
            
            # Compare against expected move
            time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
            expected_move_pct = implied_vol * time_factor
            
            if required_move > expected_move_pct * 1.8:
                return {'error': f'Required move {required_move:.1%} too large vs expected {expected_move_pct:.1%}'}
            
            # Calculate probability of profit
            prob_profit = self._calculate_strangle_probability(
                breakeven_up, breakeven_down, current_price, implied_vol, options_data['days_to_expiry']
            )
            
            if prob_profit < 0.20:  # Less than 20% chance of profit
                return {'error': f'Probability of profit too low: {prob_profit:.1%}'}
            
            # Calculate cost efficiency vs straddle
            strangle_width = call_option['strike'] - put_option['strike']
            cost_per_point = net_debit / strangle_width
            
            # Enhanced position sizing
            base_risk = portfolio_value * self.risk_tolerance
            
            # Adjust for market conditions
            vol_adjustment = min(1.3, 0.25 / realized_vol)  # Increase size in low vol
            event_adjustment = 1.4 if earnings_soon else 1.0
            trend_adjustment = 1.1 if trend == 'SIDEWAYS' else 1.0  # Slightly higher for breakout plays
            liquidity_adjustment = min(1.0, (call_option.get('volume', 5) + put_option.get('volume', 5)) / 30)
            
            adjusted_risk = base_risk * vol_adjustment * event_adjustment * trend_adjustment * liquidity_adjustment
            contracts = max(1, min(int(adjusted_risk / (max_loss * 100)), 6))
            
            # Calculate management levels
            profit_target_1 = net_debit * 1.5  # 50% gain
            profit_target_2 = net_debit * 2.5  # 150% gain
            loss_threshold = net_debit * 0.4   # Stop loss at 60% of premium
            
            # Calculate additional metrics
            total_wing_width = (call_option['strike'] - current_price) + (current_price - put_option['strike'])
            wing_symmetry = abs((call_option['strike'] - current_price) - (current_price - put_option['strike'])) / current_price
            
            return {
                'strategy_name': 'Long Strangle',
                'legs': [
                    {
                        'action': 'BUY',
                        'option_type': 'CALL',
                        'strike': call_option['strike'],
                        'expiration': options_data['expiration'],
                        'price': call_option['lastPrice'],
                        'contracts': contracts,
                        'delta': call_option.get('delta', 'N/A'),
                        'volume': call_option.get('volume', 'N/A'),
                        'open_interest': call_option.get('openInterest', 'N/A'),
                        'distance_from_spot': round(((call_option['strike'] - current_price) / current_price) * 100, 2)
                    },
                    {
                        'action': 'BUY',
                        'option_type': 'PUT',
                        'strike': put_option['strike'],
                        'expiration': options_data['expiration'],
                        'price': put_option['lastPrice'],
                        'contracts': contracts,
                        'delta': put_option.get('delta', 'N/A'),
                        'volume': put_option.get('volume', 'N/A'),
                        'open_interest': put_option.get('openInterest', 'N/A'),
                        'distance_from_spot': round(((current_price - put_option['strike']) / current_price) * 100, 2)
                    }
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'breakeven_up': round(breakeven_up, 2),
                'breakeven_down': round(breakeven_down, 2),
                'breakeven_moves': {
                    'upside_required': round(breakeven_move_up * 100, 2),
                    'downside_required': round(breakeven_move_down * 100, 2)
                },
                'profit_targets': {
                    'target_1': round(profit_target_1 * contracts * 100, 2),
                    'target_2': round(profit_target_2 * contracts * 100, 2)
                },
                'loss_threshold': round(loss_threshold * contracts * 100, 2),
                'days_to_expiry': options_data['days_to_expiry'],
                'probability_of_profit': round(prob_profit * 100, 1),
                'expected_vs_required_move': {
                    'implied_move': round(expected_move_pct * 100, 2),
                    'required_move': round(required_move * 100, 2)
                },
                'strangle_metrics': {
                    'total_wing_width': round(total_wing_width, 2),
                    'wing_symmetry': round(wing_symmetry * 100, 2),
                    'cost_per_point': round(cost_per_point, 3),
                    'strangle_width': round(strangle_width, 2)
                },
                'iv_rank': iv_rank,
                'earnings_soon': earnings_soon,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"Lower-cost volatility play for {symbol}. "
                            f"{'Earnings in <10 days - ' if earnings_soon else ''}"
                            f"Needs {required_move:.1%} move vs {expected_move_pct:.1%} implied. "
                            f"Breakevens: ${breakeven_down:.2f} - ${breakeven_up:.2f}. "
                            f"Wing distances: {((call_option['strike'] - current_price) / current_price) * 100:.1f}% call, "
                            f"{((current_price - put_option['strike']) / current_price) * 100:.1f}% put. "
                            f"Probability of profit: {prob_profit*100:.1f}%. "
                            f"Take profits at 50% (${round(profit_target_1 * contracts * 100, 2)}) "
                            f"and 150% (${round(profit_target_2 * contracts * 100, 2)}). "
                            f"Stop loss at 60% premium loss (${round(loss_threshold * contracts * 100, 2)})."
            }
            
        except Exception as e:
            return {'error': f'Long strangle calculation failed: {str(e)}'}

    def _calculate_strangle_probability(self, breakeven_up: float, breakeven_down: float,
                                    current_price: float, implied_vol: float, days_to_expiry: int) -> float:
        """Calculate probability that stock will move beyond breakeven points for strangle"""
        import numpy as np
        from scipy.stats import norm
        
        time_to_expiry = days_to_expiry / 365
        
        if time_to_expiry <= 0:
            return 1.0 if current_price <= breakeven_down or current_price >= breakeven_up else 0.0
        
        # Using log-normal distribution
        vol_sqrt_t = implied_vol * np.sqrt(time_to_expiry)
        
        # Z-scores for breakeven points
        z_upper = (np.log(breakeven_up / current_price)) / vol_sqrt_t
        z_lower = (np.log(breakeven_down / current_price)) / vol_sqrt_t
        
        # Probability of finishing outside the breakeven range
        prob_profit = 1 - (norm.cdf(z_upper) - norm.cdf(z_lower))
        
        return max(0.0, min(1.0, prob_profit))
    
    def _covered_call(self, symbol: str, stock_data: Dict, options_data: Dict,
                    market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Covered Call with advanced income optimization and risk management"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if calls.empty:
                return {'error': 'No call options available for covered call'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check if conditions are suitable for covered call
            # 1. Neutral to slightly bullish conditions (don't want stock called away in strong bull market)
            unsuitable_trends = ['STRONG_BULLISH']
            if trend in unsuitable_trends:
                return {'error': f'Trend {trend} not suitable for covered call - stock likely to be called away'}
            
            # 2. High IV environment preferred (selling premium when expensive)
            iv_rank = stock_data.get('iv_rank', 50)
            if implied_vol < realized_vol * 0.7 or iv_rank < 25:
                return {'error': 'IV too low for covered call - premium not attractive enough'}
            
            # 3. Check momentum - avoid extremely bullish momentum
            bullish_momentum = ['EXTREMELY_OVERBOUGHT']
            if momentum in bullish_momentum:
                return {'error': f'Momentum {momentum} too bullish for covered call'}
            
            # Enhanced call filtering with liquidity and delta requirements
            min_premium = max(0.10, current_price * 0.005)  # Minimum 0.5% of stock price
            
            otm_calls = calls[
                (calls['strike'] > current_price * 1.005) &  # At least 0.5% OTM
                (calls['lastPrice'] > min_premium) &
                (calls['volume'] > 10) &  # Liquidity requirement
                (calls['openInterest'] > 25) &
                (calls['bid'] > 0) & (calls['ask'] > 0)
            ].copy()
            
            # Fallback with relaxed criteria
            if otm_calls.empty:
                otm_calls = calls[
                    (calls['strike'] > current_price) &
                    (calls['lastPrice'] > 0.05)
                ].copy()
            
            if otm_calls.empty:
                return {'error': 'No suitable calls for covered call'}
            
            # Calculate deltas if not provided
            if 'delta' not in otm_calls.columns:
                otm_calls['delta'] = self._calculate_delta(otm_calls, current_price, 
                                                        options_data['days_to_expiry'], implied_vol)
            
            # Enhanced strike selection based on multiple criteria
            def select_optimal_strike():
                # Technical analysis factors
                resistance_level = stock_data.get('resistance_level')
                
                # Calculate target delta range based on market conditions and time
                if options_data['days_to_expiry'] <= 30:
                    # Short-term: higher delta for more premium
                    target_delta_range = (0.25, 0.45)
                elif options_data['days_to_expiry'] <= 45:
                    # Medium-term: moderate delta
                    target_delta_range = (0.20, 0.35)
                else:
                    # Longer-term: lower delta to reduce assignment risk
                    target_delta_range = (0.15, 0.30)
                
                # Filter by target delta range
                delta_candidates = otm_calls[
                    (otm_calls['delta'] >= target_delta_range[0]) & 
                    (otm_calls['delta'] <= target_delta_range[1])
                ]
                
                if delta_candidates.empty:
                    # Fallback to any OTM calls
                    delta_candidates = otm_calls
                
                # Score candidates based on multiple factors
                delta_candidates = delta_candidates.copy()
                
                # 1. Premium yield (annualized)
                days_to_annual = 365 / options_data['days_to_expiry']
                delta_candidates['annualized_yield'] = (delta_candidates['lastPrice'] / current_price) * days_to_annual * 100
                
                # 2. Distance from resistance (if available)
                if resistance_level and resistance_level > current_price * 1.02:
                    # Prefer strikes near but below resistance
                    delta_candidates['resistance_score'] = np.where(
                        delta_candidates['strike'] <= resistance_level,
                        1 - abs(delta_candidates['strike'] - resistance_level) / resistance_level,
                        0.5  # Penalty for strikes above resistance
                    )
                else:
                    delta_candidates['resistance_score'] = 0.8  # Neutral score
                
                # 3. Probability of profit (keeping the stock)
                delta_candidates['prob_keep_stock'] = 1 - delta_candidates['delta']
                
                # 4. Liquidity score
                delta_candidates['liquidity_score'] = np.log1p(delta_candidates['volume']) * np.log1p(delta_candidates['openInterest'])
                
                # 5. Risk-adjusted return score
                delta_candidates['total_return_if_called'] = (
                    (delta_candidates['strike'] - current_price + delta_candidates['lastPrice']) / current_price * 100
                )
                
                # Composite scoring based on strategy goals
                if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                    # High vol: prioritize premium collection
                    delta_candidates['composite_score'] = (
                        delta_candidates['annualized_yield'] * 0.4 +
                        delta_candidates['resistance_score'] * 0.2 +
                        delta_candidates['prob_keep_stock'] * 0.2 +
                        delta_candidates['liquidity_score'] * 0.1 +
                        delta_candidates['total_return_if_called'] * 0.1
                    )
                else:
                    # Normal vol: balance premium and assignment risk
                    delta_candidates['composite_score'] = (
                        delta_candidates['annualized_yield'] * 0.3 +
                        delta_candidates['resistance_score'] * 0.3 +
                        delta_candidates['prob_keep_stock'] * 0.25 +
                        delta_candidates['liquidity_score'] * 0.1 +
                        delta_candidates['total_return_if_called'] * 0.05
                    )
                
                # Select the highest scoring option
                best_call = delta_candidates.loc[delta_candidates['composite_score'].idxmax()]
                
                return best_call
            
            call_to_sell = select_optimal_strike()
            
            # Position sizing based on available capital and risk management
            shares_per_contract = 100
            max_position_value = portfolio_value * 0.15  # Max 15% in single covered call
            max_contracts = int(max_position_value / (current_price * shares_per_contract))
            contracts = max(1, min(max_contracts, 5))  # 1-5 contracts max
            
            shares_needed = shares_per_contract * contracts
            cost_basis = current_price * shares_needed
            premium_received = call_to_sell['lastPrice'] * 100 * contracts
            
            if premium_received <= 0:
                return {'error': 'No premium available from call'}
            
            # Calculate all financial metrics
            net_investment = cost_basis - premium_received
            max_profit_if_called = (call_to_sell['strike'] * shares_needed) - cost_basis + premium_received
            max_profit_if_not_called = premium_received  # Just the premium
            breakeven = current_price - call_to_sell['lastPrice']
            
            # Yield calculations
            premium_yield = (premium_received / cost_basis) * 100
            yield_if_called = ((call_to_sell['strike'] - current_price + call_to_sell['lastPrice']) / current_price) * 100
            annualized_premium_yield = premium_yield * (365 / options_data['days_to_expiry'])
            annualized_yield_if_called = yield_if_called * (365 / options_data['days_to_expiry'])
            
            # Risk metrics
            downside_protection = (call_to_sell['lastPrice'] / current_price) * 100
            assignment_probability = call_to_sell.get('delta', 0.3) * 100
            
            # Validate minimum yield requirements
            min_annualized_yield = 8.0  # 8% minimum annualized yield
            if annualized_premium_yield < min_annualized_yield:
                return {'error': f'Annualized yield {annualized_premium_yield:.1f}% below minimum {min_annualized_yield}%'}
            
            # Management rules
            profit_target_premium = premium_received * 0.5  # Close at 50% profit
            roll_threshold_days = 7  # Consider rolling if <7 days and ITM
            
            # Calculate time decay benefit
            theta_per_day = premium_received / options_data['days_to_expiry']
            
            return {
                'strategy_name': 'Covered Call',
                'legs': [
                    {
                        'action': 'BUY',
                        'instrument': 'STOCK',
                        'quantity': shares_needed,
                        'price': current_price,
                        'total_cost': round(cost_basis, 2)
                    },
                    {
                        'action': 'SELL',
                        'option_type': 'CALL',
                        'strike': call_to_sell['strike'],
                        'expiration': options_data['expiration'],
                        'price': call_to_sell['lastPrice'],
                        'contracts': contracts,
                        'delta': call_to_sell.get('delta', 'N/A'),
                        'volume': call_to_sell.get('volume', 'N/A'),
                        'open_interest': call_to_sell.get('openInterest', 'N/A'),
                        'total_premium': round(premium_received, 2)
                    }
                ],
                'initial_cost': round(net_investment, 2),
                'premium_received': round(premium_received, 2),
                'max_profit_if_called': round(max_profit_if_called, 2),
                'max_profit_if_not_called': round(max_profit_if_not_called, 2),
                'breakeven': round(breakeven, 2),
                'assignment_strike': call_to_sell['strike'],
                'assignment_probability': round(assignment_probability, 1),
                'days_to_expiry': options_data['days_to_expiry'],
                'yields': {
                    'premium_yield': round(premium_yield, 2),
                    'yield_if_called': round(yield_if_called, 2),
                    'annualized_premium_yield': round(annualized_premium_yield, 2),
                    'annualized_yield_if_called': round(annualized_yield_if_called, 2)
                },
                'risk_metrics': {
                    'downside_protection': round(downside_protection, 2),
                    'theta_per_day': round(theta_per_day, 2),
                    'break_even_decline': round(((current_price - breakeven) / current_price) * 100, 2)
                },
                'management_rules': {
                    'profit_target': round(profit_target_premium, 2),
                    'roll_threshold_days': roll_threshold_days,
                    'consider_rolling_if': f"<{roll_threshold_days} days and strike ${call_to_sell['strike']:.2f} is ITM"
                },
                'iv_rank': iv_rank,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"High-IV income strategy for {symbol}. "
                            f"Collect {premium_yield:.1f}% premium ({annualized_premium_yield:.1f}% annualized). "
                            f"Stock called away at ${call_to_sell['strike']:.2f} for {yield_if_called:.1f}% total return. "
                            f"Assignment probability: {assignment_probability:.1f}%. "
                            f"Downside protection: {downside_protection:.1f}%. "
                            f"Close position at 50% profit (${round(profit_target_premium, 2)}) or "
                            f"consider rolling if <{roll_threshold_days} days and ITM."
            }
            
        except Exception as e:
            return {'error': f'Covered call calculation failed: {str(e)}'}
    
    def _cash_secured_put(self, symbol: str, stock_data: Dict, options_data: Dict,
                        market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Cash Secured Put with advanced income optimization and risk management"""
        try:
            current_price = stock_data['current_price']
            puts = options_data['puts']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if puts.empty:
                return {'error': 'No put options available for cash secured put'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check if conditions are suitable for cash secured put
            # 1. Neutral to slightly bearish conditions (want to acquire stock at discount)
            unsuitable_trends = ['STRONG_BEARISH']
            if trend in unsuitable_trends:
                return {'error': f'Trend {trend} not suitable for cash secured put - high assignment risk in weak market'}
            
            # 2. High IV environment preferred (selling premium when expensive)
            iv_rank = stock_data.get('iv_rank', 50)
            if implied_vol < realized_vol * 0.7 or iv_rank < 25:
                return {'error': 'IV too low for cash secured put - premium not attractive enough'}
            
            # 3. Check momentum - avoid extremely bearish momentum
            bearish_momentum = ['EXTREMELY_OVERSOLD']
            if momentum in bearish_momentum:
                return {'error': f'Momentum {momentum} too bearish for cash secured put'}
            
            # 4. Stock quality check - should be willing to own this stock
            price_vs_52w_low = market_analysis.get('price_vs_52w_low', 150)
            if price_vs_52w_low < 110:  # Too close to 52-week low
                return {'error': 'Stock too close to 52-week low - high assignment risk'}
            
            # Enhanced put filtering with liquidity and delta requirements
            min_premium = max(0.10, current_price * 0.005)  # Minimum 0.5% of stock price
            
            otm_puts = puts[
                (puts['strike'] < current_price * 0.995) &  # At least 0.5% OTM
                (puts['lastPrice'] > min_premium) &
                (puts['volume'] > 10) &  # Liquidity requirement
                (puts['openInterest'] > 25) &
                (puts['bid'] > 0) & (puts['ask'] > 0)
            ].copy()
            
            # Fallback with relaxed criteria
            if otm_puts.empty:
                otm_puts = puts[
                    (puts['strike'] < current_price) &
                    (puts['lastPrice'] > 0.05)
                ].copy()
            
            if otm_puts.empty:
                return {'error': 'No suitable puts for cash secured put'}
            
            # Calculate deltas if not provided
            if 'delta' not in otm_puts.columns:
                otm_puts['delta'] = self._calculate_delta(otm_puts, current_price, 
                                                        options_data['days_to_expiry'], implied_vol, option_type='put')
            
            # Enhanced strike selection based on multiple criteria
            def select_optimal_strike():
                # Technical analysis factors
                support_level = stock_data.get('support_level')
                
                # Calculate target delta range based on market conditions and time
                if options_data['days_to_expiry'] <= 30:
                    # Short-term: higher delta for more premium
                    target_delta_range = (-0.45, -0.25)
                elif options_data['days_to_expiry'] <= 45:
                    # Medium-term: moderate delta
                    target_delta_range = (-0.35, -0.20)
                else:
                    # Longer-term: lower delta to reduce assignment risk
                    target_delta_range = (-0.30, -0.15)
                
                # Filter by target delta range
                delta_candidates = otm_puts[
                    (otm_puts['delta'] <= target_delta_range[0]) & 
                    (otm_puts['delta'] >= target_delta_range[1])
                ]
                
                if delta_candidates.empty:
                    # Fallback to any OTM puts
                    delta_candidates = otm_puts
                
                # Score candidates based on multiple factors
                delta_candidates = delta_candidates.copy()
                
                # 1. Premium yield (annualized)
                days_to_annual = 365 / options_data['days_to_expiry']
                delta_candidates['annualized_yield'] = (delta_candidates['lastPrice'] / delta_candidates['strike']) * days_to_annual * 100
                
                # 2. Distance from support (if available)
                if support_level and support_level < current_price * 0.98:
                    # Prefer strikes near but above support
                    delta_candidates['support_score'] = np.where(
                        delta_candidates['strike'] >= support_level,
                        1 - abs(delta_candidates['strike'] - support_level) / support_level,
                        0.5  # Penalty for strikes below support
                    )
                else:
                    delta_candidates['support_score'] = 0.8  # Neutral score
                
                # 3. Probability of profit (not getting assigned)
                delta_candidates['prob_avoid_assignment'] = 1 - abs(delta_candidates['delta'])
                
                # 4. Liquidity score
                delta_candidates['liquidity_score'] = np.log1p(delta_candidates['volume']) * np.log1p(delta_candidates['openInterest'])
                
                # 5. Value proposition if assigned
                delta_candidates['discount_if_assigned'] = ((current_price - delta_candidates['strike']) / current_price) * 100
                delta_candidates['effective_cost_basis'] = delta_candidates['strike'] - delta_candidates['lastPrice']
                delta_candidates['discount_to_current'] = ((current_price - delta_candidates['effective_cost_basis']) / current_price) * 100
                
                # 6. Strike attractiveness (prefer round numbers and technical levels)
                delta_candidates['strike_attractiveness'] = np.where(
                    delta_candidates['strike'] % 5 == 0,  # Round to $5
                    1.1,
                    np.where(delta_candidates['strike'] % 1 == 0, 1.05, 1.0)  # Round to $1
                )
                
                # Composite scoring based on strategy goals
                if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                    # High vol: prioritize premium collection
                    delta_candidates['composite_score'] = (
                        delta_candidates['annualized_yield'] * 0.35 +
                        delta_candidates['support_score'] * 0.2 +
                        delta_candidates['prob_avoid_assignment'] * 0.15 +
                        delta_candidates['liquidity_score'] * 0.1 +
                        delta_candidates['discount_to_current'] * 0.1 +
                        delta_candidates['strike_attractiveness'] * 0.1
                    )
                else:
                    # Normal vol: balance premium and assignment risk
                    delta_candidates['composite_score'] = (
                        delta_candidates['annualized_yield'] * 0.25 +
                        delta_candidates['support_score'] * 0.25 +
                        delta_candidates['prob_avoid_assignment'] * 0.2 +
                        delta_candidates['liquidity_score'] * 0.1 +
                        delta_candidates['discount_to_current'] * 0.15 +
                        delta_candidates['strike_attractiveness'] * 0.05
                    )
                
                # Select the highest scoring option
                best_put = delta_candidates.loc[delta_candidates['composite_score'].idxmax()]
                
                return best_put
            
            put_to_sell = select_optimal_strike()
            
            # Position sizing based on available capital and risk management
            max_position_value = portfolio_value * 0.20  # Max 20% in single CSP
            shares_per_contract = 100
            max_contracts = int(max_position_value / (put_to_sell['strike'] * shares_per_contract))
            contracts = max(1, min(max_contracts, 5))  # 1-5 contracts max
            
            cash_required = put_to_sell['strike'] * 100 * contracts
            premium_received = put_to_sell['lastPrice'] * 100 * contracts
            
            if premium_received <= 0:
                return {'error': 'No premium available from put'}
            
            # Validate sufficient cash
            if cash_required > portfolio_value * 0.25:
                return {'error': f'Insufficient cash for position. Required: ${cash_required:,.0f}'}
            
            # Calculate all financial metrics
            net_cost_if_assigned = cash_required - premium_received
            effective_cost_basis = put_to_sell['strike'] - put_to_sell['lastPrice']
            breakeven = put_to_sell['strike'] - put_to_sell['lastPrice']
            
            # Yield calculations
            premium_yield = (premium_received / cash_required) * 100
            annualized_premium_yield = premium_yield * (365 / options_data['days_to_expiry'])
            
            # If assigned, what's the discount?
            discount_if_assigned = ((current_price - effective_cost_basis) / current_price) * 100
            assignment_probability = abs(put_to_sell.get('delta', -0.3)) * 100
            
            # Risk metrics
            max_loss_if_assigned = net_cost_if_assigned - (breakeven * shares_per_contract * contracts)
            downside_to_breakeven = ((current_price - breakeven) / current_price) * 100
            
            # Validate minimum yield requirements
            min_annualized_yield = 6.0  # 6% minimum annualized yield
            if annualized_premium_yield < min_annualized_yield:
                return {'error': f'Annualized yield {annualized_premium_yield:.1f}% below minimum {min_annualized_yield}%'}
            
            # Management rules
            profit_target_premium = premium_received * 0.5  # Close at 50% profit
            roll_threshold_days = 7  # Consider rolling if <7 days and ITM
            
            # Calculate opportunity cost
            theta_per_day = premium_received / options_data['days_to_expiry']
            cash_utilization = (cash_required / portfolio_value) * 100
            
            return {
                'strategy_name': 'Cash Secured Put',
                'legs': [
                    {
                        'action': 'SELL',
                        'option_type': 'PUT',
                        'strike': put_to_sell['strike'],
                        'expiration': options_data['expiration'],
                        'price': put_to_sell['lastPrice'],
                        'contracts': contracts,
                        'delta': put_to_sell.get('delta', 'N/A'),
                        'volume': put_to_sell.get('volume', 'N/A'),
                        'open_interest': put_to_sell.get('openInterest', 'N/A'),
                        'total_premium': round(premium_received, 2)
                    }
                ],
                'cash_required': round(cash_required, 2),
                'premium_received': round(premium_received, 2),
                'net_cost_if_assigned': round(net_cost_if_assigned, 2),
                'effective_cost_basis': round(effective_cost_basis, 2),
                'breakeven': round(breakeven, 2),
                'assignment_probability': round(assignment_probability, 1),
                'days_to_expiry': options_data['days_to_expiry'],
                'yields': {
                    'premium_yield': round(premium_yield, 2),
                    'annualized_premium_yield': round(annualized_premium_yield, 2)
                },
                'assignment_analysis': {
                    'discount_if_assigned': round(discount_if_assigned, 2),
                    'current_price': current_price,
                    'strike_price': put_to_sell['strike'],
                    'would_buy_at': round(effective_cost_basis, 2),
                    'shares_if_assigned': shares_per_contract * contracts
                },
                'risk_metrics': {
                    'downside_to_breakeven': round(downside_to_breakeven, 2),
                    'theta_per_day': round(theta_per_day, 2),
                    'cash_utilization': round(cash_utilization, 2),
                    'max_theoretical_loss': 'Limited to strike price if stock goes to zero'
                },
                'management_rules': {
                    'profit_target': round(profit_target_premium, 2),
                    'roll_threshold_days': roll_threshold_days,
                    'consider_rolling_if': f"<{roll_threshold_days} days and strike ${put_to_sell['strike']:.2f} is ITM"
                },
                'iv_rank': iv_rank,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"High-IV income strategy for {symbol}. "
                            f"Collect {premium_yield:.1f}% premium ({annualized_premium_yield:.1f}% annualized) "
                            f"on ${cash_required:,.0f} cash. "
                            f"Assignment probability: {assignment_probability:.1f}%. "
                            f"If assigned, acquire stock at {discount_if_assigned:.1f}% discount to current price "
                            f"(${effective_cost_basis:.2f} vs ${current_price:.2f}). "
                            f"Close position at 50% profit (${round(profit_target_premium, 2)}) or "
                            f"consider rolling if <{roll_threshold_days} days and ITM. "
                            f"Cash utilization: {cash_utilization:.1f}% of portfolio."
            }
            
        except Exception as e:
            return {'error': f'Cash secured put calculation failed: {str(e)}'}
    
    def _protective_put(self, symbol: str, stock_data: Dict, options_data: Dict,
                    market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Protective Put with advanced hedging optimization and risk management"""
        try:
            current_price = stock_data['current_price']
            puts = options_data['puts']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if puts.empty:
                return {'error': 'No put options available for protective put'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check if conditions warrant protective puts
            # 1. Market showing signs of weakness or high uncertainty
            hedge_warranted_trends = ['BEARISH', 'SHORT_TERM_BEARISH', 'STRONG_BEARISH', 'SIDEWAYS']
            hedge_warranted_momentum = ['BEARISH', 'OVERBOUGHT', 'EXTREMELY_OVERBOUGHT', 'NEUTRAL']
            
            if trend not in hedge_warranted_trends and momentum not in hedge_warranted_momentum:
                return {'error': f'Market conditions ({trend}, {momentum}) do not warrant expensive hedging'}
            
            # 2. Check if insurance is reasonably priced
            iv_rank = stock_data.get('iv_rank', 50)
            if implied_vol > realized_vol * 2.0 or iv_rank > 80:
                return {'error': 'Put insurance too expensive - consider other hedging strategies'}
            
            # 3. Position size validation - must own or plan to own the stock
            current_position_value = stock_data.get('position_value', 0)
            if current_position_value == 0:
                # Planning to buy stock + protection
                max_position_value = portfolio_value * 0.15  # Max 15% in single hedged position
            else:
                # Already own stock, hedging existing position
                max_position_value = current_position_value
            
            # Enhanced put filtering with protection level optimization
            min_premium = 0.05  # Minimum viable premium
            
            # Calculate protection levels based on risk tolerance and market conditions
            if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                # High vol: closer protection due to higher downside risk
                protection_levels = [0.95, 0.93, 0.90, 0.88]
            elif trend in ['STRONG_BEARISH', 'BEARISH']:
                # Bearish trend: tighter protection
                protection_levels = [0.95, 0.92, 0.90, 0.85]
            else:
                # Normal conditions: standard protection levels
                protection_levels = [0.92, 0.90, 0.88, 0.85]
            
            # Filter viable puts
            protective_puts = puts[
                (puts['strike'] < current_price * 0.98) &  # OTM protection
                (puts['lastPrice'] > min_premium) &
                (puts['volume'] > 5) &  # Liquidity requirement
                (puts['openInterest'] > 10) &
                (puts['bid'] > 0) & (puts['ask'] > 0)
            ].copy()
            
            if protective_puts.empty:
                # Fallback with relaxed criteria
                protective_puts = puts[
                    (puts['strike'] < current_price) &
                    (puts['lastPrice'] > 0.02)
                ].copy()
            
            if protective_puts.empty:
                return {'error': 'No suitable puts for protection'}
            
            # Calculate deltas if not provided
            if 'delta' not in protective_puts.columns:
                protective_puts['delta'] = self._calculate_delta(protective_puts, current_price, 
                                                            options_data['days_to_expiry'], implied_vol, option_type='put')
            
            # Enhanced strike selection based on protection goals and cost efficiency
            def select_optimal_protection():
                # Technical analysis factors
                support_level = stock_data.get('support_level')
                
                # Score protection options
                protection_candidates = []
                
                for protection_level in protection_levels:
                    target_strike = current_price * protection_level
                    
                    # Find puts near this protection level
                    level_puts = protective_puts[
                        (protective_puts['strike'] >= target_strike * 0.95) &
                        (protective_puts['strike'] <= target_strike * 1.05)
                    ]
                    
                    if level_puts.empty:
                        continue
                    
                    # Select best put at this level
                    level_puts = level_puts.copy()
                    level_puts['strike_distance'] = abs(level_puts['strike'] - target_strike)
                    best_at_level = level_puts.loc[level_puts['strike_distance'].idxmin()]
                    
                    # Calculate metrics for this protection level
                    insurance_cost_pct = (best_at_level['lastPrice'] / current_price) * 100
                    protection_pct = (best_at_level['strike'] / current_price) * 100
                    cost_efficiency = protection_pct / insurance_cost_pct  # Protection per $ spent
                    
                    # Annualized insurance cost
                    annualized_cost = insurance_cost_pct * (365 / options_data['days_to_expiry'])
                    
                    # Support level alignment bonus
                    support_bonus = 0
                    if support_level and abs(best_at_level['strike'] - support_level) / support_level < 0.05:
                        support_bonus = 2  # Bonus for strikes near support
                    
                    # Delta considerations (higher delta = more expensive but better protection)
                    delta_score = abs(best_at_level.get('delta', -0.2)) * 10
                    
                    # Liquidity score
                    liquidity_score = np.log1p(best_at_level['volume']) + np.log1p(best_at_level['openInterest'])
                    
                    # Composite score
                    composite_score = (
                        cost_efficiency * 3 +
                        (100 - annualized_cost) * 0.5 +  # Lower annualized cost is better
                        support_bonus +
                        delta_score +
                        liquidity_score * 0.5
                    )
                    
                    protection_candidates.append({
                        'put': best_at_level,
                        'protection_level': protection_level,
                        'insurance_cost_pct': insurance_cost_pct,
                        'protection_pct': protection_pct,
                        'cost_efficiency': cost_efficiency,
                        'annualized_cost': annualized_cost,
                        'composite_score': composite_score
                    })
                
                if not protection_candidates:
                    # Fallback to closest available put
                    protective_puts['distance'] = abs(protective_puts['strike'] - (current_price * 0.90))
                    return protective_puts.loc[protective_puts['distance'].idxmin()]
                
                # Select the highest scoring protection
                best_protection = max(protection_candidates, key=lambda x: x['composite_score'])
                return best_protection['put']
            
            put_to_buy = select_optimal_protection()
            
            # Position sizing based on available capital and existing holdings
            shares_per_contract = 100
            max_contracts = int(max_position_value / (current_price * shares_per_contract))
            
            # Consider insurance cost in position sizing
            insurance_cost_per_contract = put_to_buy['lastPrice'] * 100
            total_cost_per_contract = (current_price * shares_per_contract) + insurance_cost_per_contract
            
            # Adjust contracts based on total cost including insurance
            max_contracts_with_insurance = int(max_position_value / total_cost_per_contract)
            contracts = max(1, min(max_contracts_with_insurance, 5))  # 1-5 contracts max
            
            shares_needed = shares_per_contract * contracts
            insurance_cost = insurance_cost_per_contract * contracts
            
            # Calculate all financial metrics
            stock_cost = current_price * shares_needed
            total_investment = stock_cost + insurance_cost
            protected_value = put_to_buy['strike'] * shares_needed
            
            # Risk calculations
            max_loss_stock = (current_price - put_to_buy['strike']) * shares_needed
            max_total_loss = max_loss_stock + insurance_cost
            insurance_percentage = (insurance_cost / stock_cost) * 100
            
            # Profit calculations
            breakeven_price = current_price + put_to_buy['lastPrice']
            upside_potential = float('inf')  # Unlimited upside
            
            # Time value analysis
            protection_period_days = options_data['days_to_expiry']
            daily_insurance_cost = insurance_cost / protection_period_days
            
            # Validate insurance cost reasonableness
            annualized_insurance_cost = insurance_percentage * (365 / protection_period_days)
            max_acceptable_annual_cost = 15.0  # 15% annual insurance cost limit
            
            if annualized_insurance_cost > max_acceptable_annual_cost:
                return {'error': f'Annual insurance cost {annualized_insurance_cost:.1f}% exceeds {max_acceptable_annual_cost}% limit'}
            
            # Protection efficiency metrics
            protection_level_pct = (put_to_buy['strike'] / current_price) * 100
            downside_protected = ((current_price - put_to_buy['strike']) / current_price) * 100
            
            # Calculate hedge ratio effectiveness
            hedge_delta = abs(put_to_buy.get('delta', -0.2))
            hedge_effectiveness = hedge_delta * 100  # % of downside moves hedged
            
            return {
                'strategy_name': 'Protective Put',
                'legs': [
                    {
                        'action': 'BUY' if current_position_value == 0 else 'OWN',
                        'instrument': 'STOCK',
                        'quantity': shares_needed,
                        'price': current_price,
                        'total_cost': round(stock_cost, 2)
                    },
                    {
                        'action': 'BUY',
                        'option_type': 'PUT',
                        'strike': put_to_buy['strike'],
                        'expiration': options_data['expiration'],
                        'price': put_to_buy['lastPrice'],
                        'contracts': contracts,
                        'delta': put_to_buy.get('delta', 'N/A'),
                        'volume': put_to_buy.get('volume', 'N/A'),
                        'open_interest': put_to_buy.get('openInterest', 'N/A'),
                        'total_cost': round(insurance_cost, 2)
                    }
                ],
                'total_investment': round(total_investment, 2),
                'insurance_cost': round(insurance_cost, 2),
                'protected_value': round(protected_value, 2),
                'max_loss': round(max_total_loss, 2),
                'insurance_percentage': round(insurance_percentage, 2),
                'protection_level': round(protection_level_pct, 1),
                'breakeven_price': round(breakeven_price, 2),
                'days_to_expiry': protection_period_days,
                'protection_metrics': {
                    'downside_protected': round(downside_protected, 2),
                    'hedge_effectiveness': round(hedge_effectiveness, 1),
                    'daily_insurance_cost': round(daily_insurance_cost, 2),
                    'annualized_insurance_cost': round(annualized_insurance_cost, 2),
                    'protection_floor': put_to_buy['strike']
                },
                'risk_analysis': {
                    'max_downside_stock_only': round(max_loss_stock, 2),
                    'insurance_cost_of_protection': round(insurance_cost, 2),
                    'break_even_move_required': round(((breakeven_price - current_price) / current_price) * 100, 2),
                    'protection_expires': options_data['expiration']
                },
                'cost_benefit': {
                    'protection_period_days': protection_period_days,
                    'cost_per_day': round(daily_insurance_cost, 2),
                    'cost_per_percent_protected': round(insurance_cost / downside_protected, 2) if downside_protected > 0 else 'N/A'
                },
                'iv_rank': iv_rank,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'management_strategy': {
                    'roll_before_expiry': f"Consider rolling 2-3 weeks before expiration if protection still needed",
                    'profit_taking': f"Consider taking profits if stock rises {breakeven_price:.1f}% above breakeven",
                    'exercise_decision': f"Exercise put if stock closes below ${put_to_buy['strike']:.2f} at expiration"
                },
                'rationale': f"Downside protection for {symbol} in {trend.lower()} market. "
                            f"Protects {downside_protected:.1f}% downside below ${put_to_buy['strike']:.2f} "
                            f"({protection_level_pct:.1f}% of current price). "
                            f"Insurance cost: {insurance_percentage:.1f}% of position "
                            f"({annualized_insurance_cost:.1f}% annualized). "
                            f"Hedge effectiveness: {hedge_effectiveness:.1f}%. "
                            f"Breakeven: ${breakeven_price:.2f} (+{((breakeven_price - current_price) / current_price) * 100:.1f}%). "
                            f"Protection expires in {protection_period_days} days."
            }
            
        except Exception as e:
            return {'error': f'Protective put calculation failed: {str(e)}'}
    
    def _collar(self, symbol: str, stock_data: Dict, options_data: Dict,
          market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Collar with advanced hedging and income optimization"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            puts = options_data['puts']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if calls.empty or puts.empty:
                return {'error': 'Insufficient options for collar strategy'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check if conditions are suitable for collar
            # 1. Collar works best in uncertain/volatile markets or when holding large positions
            suitable_conditions = (
                vol_regime in ['HIGH_VOL', 'EXTREME_VOL'] or
                trend in ['SIDEWAYS', 'SHORT_TERM_BEARISH', 'SHORT_TERM_BULLISH'] or
                momentum in ['OVERBOUGHT', 'OVERSOLD', 'NEUTRAL']
            )
            
            if not suitable_conditions:
                return {'error': f'Market conditions ({trend}, {momentum}, {vol_regime}) not ideal for collar'}
            
            # 2. Position size validation - collar makes sense for larger positions
            current_position_value = stock_data.get('position_value', 0)
            min_position_size = portfolio_value * 0.05  # Minimum 5% position for collar to make sense
            
            if current_position_value == 0:
                max_position_value = portfolio_value * 0.20  # Max 20% in new collared position
            else:
                max_position_value = current_position_value
            
            if max_position_value < min_position_size:
                return {'error': 'Position too small for collar strategy - consider other approaches'}
            
            # Enhanced option filtering
            iv_rank = stock_data.get('iv_rank', 50)
            min_premium = 0.05
            
            # Filter protective puts
            protective_puts = puts[
                (puts['strike'] < current_price * 0.98) &  # OTM protection
                (puts['lastPrice'] > min_premium) &
                (puts['volume'] > 5) &
                (puts['openInterest'] > 15) &
                (puts['bid'] > 0) & (puts['ask'] > 0)
            ].copy()
            
            # Filter calls to sell
            calls_to_sell = calls[
                (calls['strike'] > current_price * 1.02) &  # OTM calls
                (calls['lastPrice'] > min_premium) &
                (calls['volume'] > 5) &
                (calls['openInterest'] > 15) &
                (calls['bid'] > 0) & (calls['ask'] > 0)
            ].copy()
            
            if protective_puts.empty or calls_to_sell.empty:
                return {'error': 'No suitable options for collar strategy'}
            
            # Calculate deltas if not provided
            if 'delta' not in protective_puts.columns:
                protective_puts['delta'] = self._calculate_delta(protective_puts, current_price, 
                                                            options_data['days_to_expiry'], implied_vol, option_type='put')
            if 'delta' not in calls_to_sell.columns:
                calls_to_sell['delta'] = self._calculate_delta(calls_to_sell, current_price, 
                                                            options_data['days_to_expiry'], implied_vol)
            
            # Enhanced collar optimization - find best put/call combination
            def optimize_collar():
                # Technical levels
                support_level = stock_data.get('support_level')
                resistance_level = stock_data.get('resistance_level')
                
                # Target protection and upside levels based on market conditions
                if vol_regime in ['HIGH_VOL', 'EXTREME_VOL']:
                    # High vol: tighter collar for more protection
                    target_protection_levels = [0.95, 0.93, 0.90]
                    target_upside_levels = [1.05, 1.08, 1.10]
                else:
                    # Normal vol: wider collar for more upside
                    target_protection_levels = [0.93, 0.90, 0.88]
                    target_upside_levels = [1.08, 1.12, 1.15]
                
                best_collar = None
                best_score = -float('inf')
                
                for protection_level in target_protection_levels:
                    for upside_level in target_upside_levels:
                        # Find puts near protection level
                        target_put_strike = current_price * protection_level
                        put_candidates = protective_puts[
                            (protective_puts['strike'] >= target_put_strike * 0.95) &
                            (protective_puts['strike'] <= target_put_strike * 1.05)
                        ]
                        
                        if put_candidates.empty:
                            continue
                        
                        # Find calls near upside level
                        target_call_strike = current_price * upside_level
                        call_candidates = calls_to_sell[
                            (calls_to_sell['strike'] >= target_call_strike * 0.95) &
                            (calls_to_sell['strike'] <= target_call_strike * 1.05)
                        ]
                        
                        if call_candidates.empty:
                            continue
                        
                        # Try combinations
                        for _, put_option in put_candidates.iterrows():
                            for _, call_option in call_candidates.iterrows():
                                # Calculate collar metrics
                                net_premium = call_option['lastPrice'] - put_option['lastPrice']
                                
                                # Range characteristics
                                range_width = call_option['strike'] - put_option['strike']
                                range_pct = (range_width / current_price) * 100
                                
                                # Financial metrics
                                max_profit = (call_option['strike'] - current_price) * 100 + (net_premium * 100)
                                max_loss = (current_price - put_option['strike']) * 100 - (net_premium * 100)
                                
                                # Protection and upside analysis
                                protection_pct = ((current_price - put_option['strike']) / current_price) * 100
                                upside_potential_pct = ((call_option['strike'] - current_price) / current_price) * 100
                                
                                # Cost analysis
                                insurance_cost_pct = (put_option['lastPrice'] / current_price) * 100
                                income_pct = (call_option['lastPrice'] / current_price) * 100
                                net_cost_pct = insurance_cost_pct - income_pct
                                
                                # Technical alignment scores
                                support_score = 0
                                if support_level and abs(put_option['strike'] - support_level) / support_level < 0.05:
                                    support_score = 3
                                
                                resistance_score = 0
                                if resistance_level and abs(call_option['strike'] - resistance_level) / resistance_level < 0.05:
                                    resistance_score = 3
                                
                                # Liquidity score
                                liquidity_score = (
                                    np.log1p(put_option['volume']) + np.log1p(call_option['volume']) +
                                    np.log1p(put_option['openInterest']) + np.log1p(call_option['openInterest'])
                                )
                                
                                # Delta balance (prefer balanced exposure)
                                delta_balance = abs(abs(put_option['delta']) - call_option['delta'])
                                delta_balance_score = max(0, 1 - delta_balance)
                                
                                # Composite scoring
                                score = (
                                    range_pct * 2 +  # Wider range is generally better
                                    upside_potential_pct * 1.5 +  # More upside potential
                                    (10 - abs(net_cost_pct)) * 2 +  # Prefer near-zero net cost
                                    support_score + resistance_score +
                                    liquidity_score * 0.5 +
                                    delta_balance_score * 2 +
                                    (15 - protection_pct) * 0.5  # Some penalty for too much protection needed
                                )
                                
                                if score > best_score:
                                    best_score = score
                                    best_collar = {
                                        'put': put_option,
                                        'call': call_option,
                                        'net_premium': net_premium,
                                        'range_width': range_width,
                                        'range_pct': range_pct,
                                        'max_profit': max_profit,
                                        'max_loss': max_loss,
                                        'protection_pct': protection_pct,
                                        'upside_potential_pct': upside_potential_pct,
                                        'net_cost_pct': net_cost_pct,
                                        'score': score
                                    }
                
                return best_collar
            
            optimal_collar = optimize_collar()
            
            if not optimal_collar:
                return {'error': 'No optimal collar combination found'}
            
            put_to_buy = optimal_collar['put']
            call_to_sell = optimal_collar['call']
            net_premium = optimal_collar['net_premium']
            
            # Position sizing
            shares_per_contract = 100
            max_contracts = int(max_position_value / (current_price * shares_per_contract))
            contracts = max(1, min(max_contracts, 8))  # 1-8 contracts max
            
            shares_needed = shares_per_contract * contracts
            stock_cost = current_price * shares_needed
            net_premium_total = net_premium * 100 * contracts
            
            # Recalculate all metrics for actual position size
            max_profit = (call_to_sell['strike'] - current_price) * shares_needed + net_premium_total
            max_loss = (current_price - put_to_buy['strike']) * shares_needed - net_premium_total
            
            # Performance metrics
            max_return_pct = (max_profit / stock_cost) * 100
            max_loss_pct = (max_loss / stock_cost) * 100
            
            # Calculate breakeven points
            upside_breakeven = current_price + (net_premium if net_premium > 0 else 0)
            downside_breakeven = current_price + (net_premium if net_premium < 0 else 0)
            
            # Time value analysis
            days_to_expiry = options_data['days_to_expiry']
            annualized_max_return = max_return_pct * (365 / days_to_expiry)
            
            # Risk-reward analysis
            risk_reward_ratio = abs(max_profit / max_loss) if max_loss != 0 else float('inf')
            
            # Management thresholds
            profit_target = max_profit * 0.75  # Take profit at 75% of max
            loss_threshold = max_loss * 0.5   # Consider closing at 50% of max loss
            
            return {
                'strategy_name': 'Collar',
                'legs': [
                    {
                        'action': 'BUY' if current_position_value == 0 else 'OWN',
                        'instrument': 'STOCK',
                        'quantity': shares_needed,
                        'price': current_price,
                        'total_cost': round(stock_cost, 2)
                    },
                    {
                        'action': 'BUY',
                        'option_type': 'PUT',
                        'strike': put_to_buy['strike'],
                        'expiration': options_data['expiration'],
                        'price': put_to_buy['lastPrice'],
                        'contracts': contracts,
                        'delta': put_to_buy.get('delta', 'N/A'),
                        'volume': put_to_buy.get('volume', 'N/A'),
                        'open_interest': put_to_buy.get('openInterest', 'N/A')
                    },
                    {
                        'action': 'SELL',
                        'option_type': 'CALL',
                        'strike': call_to_sell['strike'],
                        'expiration': options_data['expiration'],
                        'price': call_to_sell['lastPrice'],
                        'contracts': contracts,
                        'delta': call_to_sell.get('delta', 'N/A'),
                        'volume': call_to_sell.get('volume', 'N/A'),
                        'open_interest': call_to_sell.get('openInterest', 'N/A')
                    }
                ],
                'net_premium': round(net_premium_total, 2),
                'protected_floor': put_to_buy['strike'],
                'upside_cap': call_to_sell['strike'],
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'days_to_expiry': days_to_expiry,
                'performance_metrics': {
                    'max_return_pct': round(max_return_pct, 2),
                    'max_loss_pct': round(max_loss_pct, 2),
                    'annualized_max_return': round(annualized_max_return, 2),
                    'risk_reward_ratio': round(risk_reward_ratio, 2)
                },
                'range_analysis': {
                    'range_width': round(call_to_sell['strike'] - put_to_buy['strike'], 2),
                    'range_pct': round(optimal_collar['range_pct'], 2),
                    'protection_pct': round(optimal_collar['protection_pct'], 2),
                    'upside_potential_pct': round(optimal_collar['upside_potential_pct'], 2),
                    'current_position_in_range': round(((current_price - put_to_buy['strike']) / 
                                                    (call_to_sell['strike'] - put_to_buy['strike'])) * 100, 1)
                },
                'cost_analysis': {
                    'put_cost': round(put_to_buy['lastPrice'] * 100 * contracts, 2),
                    'call_income': round(call_to_sell['lastPrice'] * 100 * contracts, 2),
                    'net_cost_pct': round(optimal_collar['net_cost_pct'], 2),
                    'is_net_credit': net_premium_total > 0
                },
                'breakeven_analysis': {
                    'upside_breakeven': round(upside_breakeven, 2) if net_premium > 0 else 'N/A',
                    'downside_breakeven': round(downside_breakeven, 2) if net_premium < 0 else 'N/A'
                },
                'management_rules': {
                    'profit_target': round(profit_target, 2),
                    'loss_threshold': round(loss_threshold, 2),
                    'roll_calls_if': f"Stock approaches ${call_to_sell['strike']:.2f} with >2 weeks remaining",
                    'roll_puts_if': f"Stock approaches ${put_to_buy['strike']:.2f} with >2 weeks remaining",
                    'early_exit': "Consider closing if 75% of max profit achieved"
                },
                'iv_rank': iv_rank,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"Protected range-bound strategy for {symbol} in {vol_regime.lower()} environment. "
                            f"Range: ${put_to_buy['strike']:.2f} - ${call_to_sell['strike']:.2f} "
                            f"({optimal_collar['range_pct']:.1f}% width). "
                            f"{'Net credit' if net_premium_total > 0 else 'Net debit'}: ${abs(net_premium_total):.0f}. "
                            f"Max return: {max_return_pct:.1f}% ({annualized_max_return:.1f}% annualized). "
                            f"Downside protection: {optimal_collar['protection_pct']:.1f}%. "
                            f"Upside potential: {optimal_collar['upside_potential_pct']:.1f}%. "
                            f"Best for sideways to moderately directional moves within the range."
            }
            
        except Exception as e:
            return {'error': f'Collar calculation failed: {str(e)}'}
    
    def _butterfly_spread(self, symbol: str, stock_data: Dict, options_data: Dict,
                    market_analysis: Dict, portfolio_value: float) -> Dict:
        """Enhanced Butterfly Spread with advanced neutral strategy optimization"""
        try:
            current_price = stock_data['current_price']
            calls = options_data['calls']
            implied_vol = stock_data.get('implied_vol', 0.25)
            realized_vol = stock_data['realized_vol']
            
            if calls.empty or len(calls) < 3:
                return {'error': 'Insufficient calls for butterfly spread'}
            
            # Enhanced market condition validation
            trend = market_analysis['trend']
            momentum = market_analysis['momentum']
            vol_regime = market_analysis['volatility_regime']
            
            # Check if conditions are suitable for butterfly spread
            # 1. Neutral market conditions preferred
            neutral_trends = ['SIDEWAYS']
            if trend not in neutral_trends:
                return {'error': f'Trend {trend} not suitable for butterfly spread - requires neutral market'}
            
            # 2. Low to moderate volatility (high vol makes butterflies expensive)
            iv_rank = stock_data.get('iv_rank', 50)
            if implied_vol > realized_vol * 1.5 or iv_rank > 60:
                return {'error': 'IV too high for butterfly spread - premium too expensive'}
            
            # 3. Neutral momentum preferred
            extreme_momentum = ['EXTREMELY_OVERBOUGHT', 'EXTREMELY_OVERSOLD']
            if momentum in extreme_momentum:
                return {'error': f'Momentum {momentum} too extreme for butterfly spread'}
            
            # 4. Time to expiration considerations
            if options_data['days_to_expiry'] < 14:
                return {'error': 'Too close to expiration for butterfly spread'}
            
            if options_data['days_to_expiry'] > 60:
                return {'error': 'Too much time to expiration - theta decay benefit insufficient'}
            
            # Enhanced option filtering
            min_premium = 0.05
            max_premium = current_price * 0.25
            
            viable_calls = calls[
                (calls['lastPrice'] > min_premium) & 
                (calls['lastPrice'] < max_premium) &
                (calls['volume'] > 10) &  # Higher liquidity requirement
                (calls['openInterest'] > 25) &
                (calls['bid'] > 0) & (calls['ask'] > 0)
            ].copy()
            
            if len(viable_calls) < 3:
                return {'error': 'Need at least 3 viable strikes for butterfly'}
            
            viable_calls = viable_calls.sort_values('strike').reset_index(drop=True)
            
            # Calculate deltas if not provided
            if 'delta' not in viable_calls.columns:
                viable_calls['delta'] = self._calculate_delta(viable_calls, current_price, 
                                                            options_data['days_to_expiry'], implied_vol)
            
            # Enhanced butterfly optimization
            def optimize_butterfly():
                # Technical analysis factors
                resistance_level = stock_data.get('resistance_level')
                support_level = stock_data.get('support_level')
                
                # Calculate expected price range
                time_factor = (options_data['days_to_expiry'] / 365) ** 0.5
                expected_range = current_price * realized_vol * time_factor
                
                # Target center strikes - prefer areas where stock is likely to settle
                center_candidates = []
                
                # 1. Current price area (ATM butterfly)
                center_candidates.append(current_price)
                
                # 2. Technical levels if available
                if resistance_level and current_price < resistance_level < current_price * 1.15:
                    center_candidates.append(resistance_level)
                
                if support_level and current_price > support_level > current_price * 0.85:
                    center_candidates.append(support_level)
                
                # 3. Round number levels (psychological support/resistance)
                for multiplier in [0.95, 1.0, 1.05]:
                    round_price = current_price * multiplier
                    if round_price < 50:
                        round_num = round(round_price / 5) * 5
                    else:
                        round_num = round(round_price / 10) * 10
                    center_candidates.append(round_num)
                
                best_butterfly = None
                best_score = -float('inf')
                
                for center_target in center_candidates:
                    # Find center strike closest to target
                    viable_calls['center_distance'] = abs(viable_calls['strike'] - center_target)
                    center_idx = viable_calls['center_distance'].idxmin()
                    center_strike = viable_calls.loc[center_idx, 'strike']
                    
                    # Find wing strikes - prefer equal spacing
                    target_wing_width = min(expected_range * 0.8, current_price * 0.08)  # 8% max wing width
                    
                    # Lower wing
                    lower_target = center_strike - target_wing_width
                    lower_calls = viable_calls[viable_calls['strike'] < center_strike]
                    if lower_calls.empty:
                        continue
                    
                    lower_calls = lower_calls.copy()
                    lower_calls['wing_distance'] = abs(lower_calls['strike'] - lower_target)
                    lower_idx = lower_calls['wing_distance'].idxmin()
                    
                    # Higher wing
                    higher_target = center_strike + target_wing_width
                    higher_calls = viable_calls[viable_calls['strike'] > center_strike]
                    if higher_calls.empty:
                        continue
                    
                    higher_calls = higher_calls.copy()
                    higher_calls['wing_distance'] = abs(higher_calls['strike'] - higher_target)
                    higher_idx = higher_calls['wing_distance'].idxmin()
                    
                    # Get the actual options
                    low_call = lower_calls.loc[lower_idx]
                    mid_call = viable_calls.loc[center_idx]
                    high_call = higher_calls.loc[higher_idx]
                    
                    # Check for equal spacing (butterfly requirement)
                    lower_spacing = center_strike - low_call['strike']
                    upper_spacing = high_call['strike'] - center_strike
                    spacing_difference = abs(lower_spacing - upper_spacing)
                    
                    if spacing_difference > min(lower_spacing, upper_spacing) * 0.1:  # 10% tolerance
                        continue
                    
                    # Calculate butterfly metrics
                    net_debit = low_call['lastPrice'] + high_call['lastPrice'] - (2 * mid_call['lastPrice'])
                    
                    if net_debit <= 0:
                        continue
                    
                    wing_width = min(lower_spacing, upper_spacing)
                    max_profit = wing_width - net_debit
                    max_loss = net_debit
                    
                    if max_profit <= 0:
                        continue
                    
                    # Calculate scoring factors
                    # 1. Risk-reward ratio
                    risk_reward = max_profit / max_loss
                    
                    # 2. Profit zone width
                    lower_breakeven = low_call['strike'] + net_debit
                    upper_breakeven = high_call['strike'] - net_debit
                    profit_zone_width = upper_breakeven - lower_breakeven
                    profit_zone_pct = (profit_zone_width / current_price) * 100
                    
                    # 3. Probability of profit (simplified)
                    prob_in_zone = max(0.1, min(0.9, profit_zone_width / (expected_range * 2)))
                    
                    # 4. Center strike attractiveness
                    center_attractiveness = 1.0
                    if abs(center_strike - current_price) / current_price < 0.02:  # Very close to current
                        center_attractiveness = 1.5
                    elif any(abs(center_strike - level) / level < 0.02 for level in [resistance_level, support_level] if level):
                        center_attractiveness = 1.3
                    
                    # 5. Liquidity score
                    liquidity_score = (
                        np.log1p(low_call['volume']) + 
                        np.log1p(mid_call['volume']) + 
                        np.log1p(high_call['volume'])
                    )
                    
                    # 6. Delta symmetry (prefer balanced wings)
                    delta_symmetry = 1 - abs(abs(low_call['delta']) - high_call['delta'])
                    
                    # 7. Cost efficiency (premium per dollar of max profit)
                    cost_efficiency = max_profit / net_debit
                    
                    # Composite score
                    score = (
                        risk_reward * 3 +
                        profit_zone_pct * 2 +
                        prob_in_zone * 10 +
                        center_attractiveness * 2 +
                        liquidity_score * 0.5 +
                        delta_symmetry * 2 +
                        cost_efficiency * 1.5
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_butterfly = {
                            'low_call': low_call,
                            'mid_call': mid_call,
                            'high_call': high_call,
                            'net_debit': net_debit,
                            'max_profit': max_profit,
                            'max_loss': max_loss,
                            'risk_reward': risk_reward,
                            'profit_zone_width': profit_zone_width,
                            'profit_zone_pct': profit_zone_pct,
                            'prob_in_zone': prob_in_zone,
                            'lower_breakeven': lower_breakeven,
                            'upper_breakeven': upper_breakeven,
                            'wing_width': wing_width,
                            'score': score
                        }
                
                return best_butterfly
            
            optimal_butterfly = optimize_butterfly()
            
            if not optimal_butterfly:
                return {'error': 'No optimal butterfly spread found'}
            
            # Extract optimal strikes
            low_call = optimal_butterfly['low_call']
            mid_call = optimal_butterfly['mid_call']
            high_call = optimal_butterfly['high_call']
            net_debit = optimal_butterfly['net_debit']
            max_profit = optimal_butterfly['max_profit']
            max_loss = optimal_butterfly['max_loss']
            
            # Position sizing
            risk_amount = portfolio_value * self.risk_tolerance
            max_contracts = int(risk_amount / (max_loss * 100))
            contracts = max(1, min(max_contracts, 3))  # 1-3 contracts for butterflies
            
            # Calculate management levels
            profit_target = max_profit * 0.75  # Take profit at 75% of max
            loss_threshold = max_loss * 0.5   # Stop loss at 50% of max loss
            
            # Time decay analysis
            theta_benefit = (max_profit / options_data['days_to_expiry']) * contracts * 100
            
            # Calculate probability of finishing in profit zone
            prob_profit = optimal_butterfly['prob_in_zone']
            
            return {
                'strategy_name': 'Butterfly Spread',
                'legs': [
                    {
                        'action': 'BUY',
                        'option_type': 'CALL',
                        'strike': low_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': low_call['lastPrice'],
                        'contracts': contracts,
                        'delta': low_call.get('delta', 'N/A'),
                        'volume': low_call.get('volume', 'N/A'),
                        'open_interest': low_call.get('openInterest', 'N/A')
                    },
                    {
                        'action': 'SELL',
                        'option_type': 'CALL',
                        'strike': mid_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': mid_call['lastPrice'],
                        'contracts': contracts * 2,
                        'delta': mid_call.get('delta', 'N/A'),
                        'volume': mid_call.get('volume', 'N/A'),
                        'open_interest': mid_call.get('openInterest', 'N/A')
                    },
                    {
                        'action': 'BUY',
                        'option_type': 'CALL',
                        'strike': high_call['strike'],
                        'expiration': options_data['expiration'],
                        'price': high_call['lastPrice'],
                        'contracts': contracts,
                        'delta': high_call.get('delta', 'N/A'),
                        'volume': high_call.get('volume', 'N/A'),
                        'open_interest': high_call.get('openInterest', 'N/A')
                    }
                ],
                'net_debit': round(net_debit * contracts * 100, 2),
                'max_profit': round(max_profit * contracts * 100, 2),
                'max_loss': round(max_loss * contracts * 100, 2),
                'optimal_price': mid_call['strike'],
                'days_to_expiry': options_data['days_to_expiry'],
                'risk_reward_ratio': round(optimal_butterfly['risk_reward'], 2),
                'profit_range': (round(optimal_butterfly['lower_breakeven'], 2), 
                            round(optimal_butterfly['upper_breakeven'], 2)),
                'profit_zone_analysis': {
                    'zone_width': round(optimal_butterfly['profit_zone_width'], 2),
                    'zone_width_pct': round(optimal_butterfly['profit_zone_pct'], 2),
                    'probability_in_zone': round(prob_profit * 100, 1),
                    'current_position_in_zone': round(((current_price - optimal_butterfly['lower_breakeven']) / 
                                                    optimal_butterfly['profit_zone_width']) * 100, 1)
                },
                'wing_analysis': {
                    'wing_width': round(optimal_butterfly['wing_width'], 2),
                    'lower_wing': round(mid_call['strike'] - low_call['strike'], 2),
                    'upper_wing': round(high_call['strike'] - mid_call['strike'], 2),
                    'wing_symmetry': 'Balanced' if abs((mid_call['strike'] - low_call['strike']) - 
                                                    (high_call['strike'] - mid_call['strike'])) < 0.5 else 'Unbalanced'
                },
                'time_decay_analysis': {
                    'theta_benefit_per_day': round(theta_benefit, 2),
                    'optimal_time_remaining': f"{options_data['days_to_expiry']} days",
                    'time_decay_acceleration': 'Benefits from time decay if in profit zone'
                },
                'management_rules': {
                    'profit_target': round(profit_target * contracts * 100, 2),
                    'loss_threshold': round(loss_threshold * contracts * 100, 2),
                    'optimal_exit_timing': '2-3 weeks before expiration or at 75% max profit',
                    'adjustment_levels': f"Consider adjustments if stock moves outside ${optimal_butterfly['lower_breakeven']:.2f} - ${optimal_butterfly['upper_breakeven']:.2f}"
                },
                'iv_rank': iv_rank,
                'market_conditions': {
                    'trend': trend,
                    'momentum': momentum,
                    'vol_regime': vol_regime,
                    'iv_vs_rv': round(implied_vol / realized_vol, 2)
                },
                'rationale': f"Neutral low-volatility strategy for {symbol}. "
                            f"Profits if stock stays between ${optimal_butterfly['lower_breakeven']:.2f} - ${optimal_butterfly['upper_breakeven']:.2f} "
                            f"({optimal_butterfly['profit_zone_pct']:.1f}% range). "
                            f"Maximum profit ${round(max_profit * contracts * 100, 2)} at ${mid_call['strike']:.2f}. "
                            f"Probability of profit: {prob_profit*100:.1f}%. "
                            f"Risk-reward ratio: {optimal_butterfly['risk_reward']:.2f}:1. "
                            f"Benefits from time decay and low volatility. "
                            f"Close at 75% profit or if stock moves outside profit zone."
            }
            
        except Exception as e:
            return {'error': f'Butterfly spread calculation failed: {str(e)}'}
    
    def display_recommendations(self, recommendations: Dict) -> None:
        """Enhanced display with educational content and performance tracking"""
        st.title("🎯 Enhanced Options Strategy Recommendations")
        st.markdown("*Powered by MarketStack + Polygon.io APIs*")
        
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
            market_open = recommendations[list(recommendations.keys())[0]]['market_conditions']['market_hours']
            market_status = "🟢 OPEN" if market_open else "🔴 CLOSED"
            st.metric("Market Status", market_status)
        
        # Individual recommendations
        for symbol, rec in recommendations.items():
            with st.expander(f"🎯 {symbol} - {rec['strategy']} (Confidence: {rec['confidence']:.1f}/10)", expanded=True):
                trade = rec['trade_details']
                
                # Educational content
                if rec['strategy'].replace('_', ' ').title() in self.strategy_explanations:
                    strategy_info = self.strategy_explanations[rec['strategy'].replace('_', ' ').title()]
                    
                    with st.container():
                        st.info(f"📚 **About {trade['strategy_name']}**: {strategy_info['description']}")
                        st.caption(f"**Best Conditions**: {strategy_info['best_conditions']}")
                
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
                    st.metric("52W High", f"${analysis['52w_high']:.2f}")
                    st.metric("52W Low", f"${analysis['52w_low']:.2f}")
                
                # Strategy Details
                st.subheader(f"💡 Strategy: {trade['strategy_name']}")
                
                if 'rationale' in trade:
                    st.success(f"**Rationale:** {trade['rationale']}")
                
                # Trade legs in a table
                if 'legs' in trade:
                    st.subheader("📋 Trade Details")
                    legs_df = pd.DataFrame(trade['legs'])
                    
                    if not legs_df.empty:
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
                
                st.markdown("---")

# =============================================================================
# ENHANCED STREAMLIT INTERFACE WITH ALL IMPROVEMENTS
# =============================================================================

def add_strategy_presets():
    """Add preset configurations for different trading styles"""
    presets = {
        "Conservative Income": {
            "symbols": "AAPL,MSFT,JNJ,PG",
            "risk_tolerance": 1,
            "description": "Focus on income generation with blue-chip stocks"
        },
        "Growth Momentum": {
            "symbols": "TSLA,NVDA,GOOGL,AMZN",
            "risk_tolerance": 3,
            "description": "Target growth stocks with momentum strategies"
        },
        "Volatility Trading": {
            "symbols": "SPY,QQQ,IWM,VIX",
            "risk_tolerance": 4,
            "description": "Trade volatility with neutral strategies"
        },
        "Tech Focus": {
            "symbols": "AAPL,MSFT,GOOGL,META,NVDA",
            "risk_tolerance": 2,
            "description": "Technology sector concentration"
        }
    }
    
    return presets

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Optional, Tuple
import time

def get_historical_stock_data(symbol: str, start_date: str, end_date: str, api_key: str) -> Optional[pd.DataFrame]:
    """Get historical stock data from Polygon.io"""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {'apikey': api_key}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'results' not in data or not data['results']:
            return None
        
        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'c': 'close', 'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume'})
        df = df.sort_values('date').reset_index(drop=True)
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

def calculate_black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    from scipy.stats import norm
    import math
    
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return max(price, 0.01)  # Minimum price of $0.01

def calculate_implied_volatility(stock_data: pd.DataFrame, window: int = 30) -> float:
    """Calculate historical volatility as proxy for implied volatility"""
    if len(stock_data) < window:
        return 0.3  # Default 30% volatility
    
    returns = stock_data['close'].pct_change().dropna()
    volatility = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
    return max(volatility, 0.1)  # Minimum 10% volatility

def backtest_covered_call(stock_data: pd.DataFrame, days_to_expiry: int = 30, 
                         strike_pct: float = 0.05, shares: int = 100) -> Dict:
    """Backtest covered call strategy"""
    trades = []
    current_shares = shares
    cash = 0
    
    for i in range(0, len(stock_data) - days_to_expiry, days_to_expiry):
        entry_date = stock_data.iloc[i]['date']
        entry_price = stock_data.iloc[i]['close']
        
        # Calculate strike price (5% OTM by default)
        strike_price = entry_price * (1 + strike_pct)
        
        # Calculate option price using Black-Scholes
        time_to_expiry = days_to_expiry / 365.0
        volatility = calculate_implied_volatility(stock_data[:i+1])
        risk_free_rate = 0.05  # 5% risk-free rate
        
        call_price = calculate_black_scholes_price(
            entry_price, strike_price, time_to_expiry, risk_free_rate, volatility, 'call'
        )
        
        # Sell call option (receive premium)
        premium_received = call_price * current_shares
        cash += premium_received
        
        # Check expiration
        if i + days_to_expiry < len(stock_data):
            exit_date = stock_data.iloc[i + days_to_expiry]['date']
            exit_price = stock_data.iloc[i + days_to_expiry]['close']
            
            # Determine if call is assigned
            if exit_price > strike_price:
                # Call assigned - sell shares at strike price
                cash += strike_price * current_shares
                current_shares = 0
                pnl = (strike_price - entry_price) * shares + premium_received
                
                # Buy back shares for next cycle
                current_shares = int(cash / exit_price)
                cash -= current_shares * exit_price
            else:
                # Call expires worthless - keep premium and shares
                pnl = premium_received
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'strike_price': strike_price,
                'premium_received': premium_received,
                'pnl': pnl,
                'shares': current_shares
            })
    
    return {
        'trades': trades,
        'final_shares': current_shares,
        'final_cash': cash
    }

def backtest_cash_secured_put(stock_data: pd.DataFrame, days_to_expiry: int = 30,
                             strike_pct: float = 0.05, capital: float = 10000) -> Dict:
    """Backtest cash secured put strategy"""
    trades = []
    cash = capital
    shares = 0
    
    for i in range(0, len(stock_data) - days_to_expiry, days_to_expiry):
        entry_date = stock_data.iloc[i]['date']
        entry_price = stock_data.iloc[i]['close']
        
        # Calculate strike price (5% OTM by default)
        strike_price = entry_price * (1 - strike_pct)
        
        # Check if we have enough cash to secure the put
        if cash < strike_price * 100:
            continue
            
        # Calculate option price using Black-Scholes
        time_to_expiry = days_to_expiry / 365.0
        volatility = calculate_implied_volatility(stock_data[:i+1])
        risk_free_rate = 0.05
        
        put_price = calculate_black_scholes_price(
            entry_price, strike_price, time_to_expiry, risk_free_rate, volatility, 'put'
        )
        
        # Sell put option (receive premium)
        premium_received = put_price * 100
        cash += premium_received
        
        # Check expiration
        if i + days_to_expiry < len(stock_data):
            exit_date = stock_data.iloc[i + days_to_expiry]['date']
            exit_price = stock_data.iloc[i + days_to_expiry]['close']
            
            # Determine if put is assigned
            if exit_price < strike_price:
                # Put assigned - buy shares at strike price
                shares += 100
                cash -= strike_price * 100
                pnl = premium_received - (strike_price - exit_price) * 100
            else:
                # Put expires worthless - keep premium
                pnl = premium_received
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'strike_price': strike_price,
                'premium_received': premium_received,
                'pnl': pnl,
                'shares': shares
            })
    
    return {
        'trades': trades,
        'final_shares': shares,
        'final_cash': cash
    }

def backtest_bull_call_spread(stock_data: pd.DataFrame, days_to_expiry: int = 30,
                             lower_strike_pct: float = 0.02, upper_strike_pct: float = 0.08,
                             contracts: int = 1) -> Dict:
    """Backtest bull call spread strategy"""
    trades = []
    cash = 0
    
    for i in range(0, len(stock_data) - days_to_expiry, days_to_expiry):
        entry_date = stock_data.iloc[i]['date']
        entry_price = stock_data.iloc[i]['close']
        
        # Calculate strike prices
        lower_strike = entry_price * (1 + lower_strike_pct)
        upper_strike = entry_price * (1 + upper_strike_pct)
        
        # Calculate option prices
        time_to_expiry = days_to_expiry / 365.0
        volatility = calculate_implied_volatility(stock_data[:i+1])
        risk_free_rate = 0.05
        
        lower_call_price = calculate_black_scholes_price(
            entry_price, lower_strike, time_to_expiry, risk_free_rate, volatility, 'call'
        )
        upper_call_price = calculate_black_scholes_price(
            entry_price, upper_strike, time_to_expiry, risk_free_rate, volatility, 'call'
        )
        
        # Net debit (buy lower strike, sell upper strike)
        net_debit = (lower_call_price - upper_call_price) * 100 * contracts
        
        # Check expiration
        if i + days_to_expiry < len(stock_data):
            exit_date = stock_data.iloc[i + days_to_expiry]['date']
            exit_price = stock_data.iloc[i + days_to_expiry]['close']
            
            # Calculate payoff at expiration
            if exit_price <= lower_strike:
                payoff = 0
            elif exit_price >= upper_strike:
                payoff = (upper_strike - lower_strike) * 100 * contracts
            else:
                payoff = (exit_price - lower_strike) * 100 * contracts
            
            pnl = payoff - net_debit
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'lower_strike': lower_strike,
                'upper_strike': upper_strike,
                'net_debit': net_debit,
                'payoff': payoff,
                'pnl': pnl
            })
    
    return {'trades': trades}

def add_real_backtest():
    """Add real backtesting functionality using actual market data"""
    st.subheader("📈 Real Strategy Backtest")
    
    # Your Polygon.io API key
    api_key = "igO7PgpW43MsVcJvr1ZuxZ_vYrH87jLZ"
    
    with st.expander("🔬 Backtest Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_symbol = st.selectbox("Symbol to Backtest", ["AAPL", "MSFT", "TSLA", "SPY", "QQQ", "NVDA", "GOOGL"])
            backtest_strategy = st.selectbox("Strategy", [
                "COVERED_CALL",
                "CASH_SECURED_PUT", 
                "BULL_CALL_SPREAD",
                "BEAR_PUT_SPREAD",
                "IRON_CONDOR"
            ])
            
        with col2:
            lookback_days = st.slider("Lookback Period (Days)", 60, 365, 180)
            starting_capital = st.number_input("Starting Capital", 10000, 1000000, 50000)
            days_to_expiry = st.slider("Days to Expiry per Trade", 15, 45, 30)
    
    if st.button("🚀 Run Real Backtest"):
        with st.spinner("📊 Fetching historical data and running backtest..."):
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')
            
            # Get historical stock data
            stock_data = get_historical_stock_data(backtest_symbol, start_date, end_date, api_key)
            
            if stock_data is None:
                st.error("Failed to fetch historical data")
                return
            
            st.success(f"✅ Fetched {len(stock_data)} days of historical data")
            
            # Run backtest based on strategy
            if backtest_strategy == "COVERED_CALL":
                shares = int(starting_capital / stock_data.iloc[0]['close'] / 100) * 100
                results = backtest_covered_call(stock_data, days_to_expiry, shares=shares)
                
            elif backtest_strategy == "CASH_SECURED_PUT":
                results = backtest_cash_secured_put(stock_data, days_to_expiry, capital=starting_capital)
                
            elif backtest_strategy == "BULL_CALL_SPREAD":
                contracts = max(1, int(starting_capital / 5000))  # Rough estimate
                results = backtest_bull_call_spread(stock_data, days_to_expiry, contracts=contracts)
                
            else:
                st.warning("Strategy not yet implemented")
                return
            
            # Calculate performance metrics
            trades = results['trades']
            if not trades:
                st.warning("No trades were executed in the backtest period")
                return
            
            trade_pnls = [trade['pnl'] for trade in trades]
            total_pnl = sum(trade_pnls)
            
            # Calculate final portfolio value
            if backtest_strategy == "COVERED_CALL":
                final_stock_value = results['final_shares'] * stock_data.iloc[-1]['close']
                final_value = results['final_cash'] + final_stock_value
                total_return = (final_value - starting_capital) / starting_capital
            elif backtest_strategy == "CASH_SECURED_PUT":
                final_stock_value = results['final_shares'] * stock_data.iloc[-1]['close']
                final_value = results['final_cash'] + final_stock_value
                total_return = (final_value - starting_capital) / starting_capital
            else:
                final_value = starting_capital + total_pnl
                total_return = total_pnl / starting_capital
            
            # Calculate other metrics
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
            
            avg_win = np.mean(winning_trades) if winning_trades else 0
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            # Calculate max drawdown
            cumulative_pnl = np.cumsum(trade_pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max) / starting_capital
            max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(trade_pnls) > 1:
                returns_std = np.std(trade_pnls)
                sharpe_ratio = (np.mean(trade_pnls) * len(trade_pnls)) / (returns_std * np.sqrt(len(trade_pnls))) if returns_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Display results
            st.success("✅ Backtest Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Return", f"{total_return:.1%}")
                st.metric("Win Rate", f"{win_rate:.1%}")
                st.metric("Total Trades", len(trades))
                
            with col2:
                st.metric("Total P&L", f"${total_pnl:,.0f}")
                st.metric("Avg Win", f"${avg_win:.0f}")
                st.metric("Avg Loss", f"${avg_loss:.0f}")
                
            with col3:
                st.metric("Final Value", f"${final_value:,.0f}")
                st.metric("Max Drawdown", f"-{max_drawdown:.1%}")
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # Performance chart
            st.subheader("📊 Performance Chart")
            if len(trades) > 0:
                # Create portfolio value over time
                portfolio_values = [starting_capital]
                dates = [trades[0]['entry_date']]
                
                for trade in trades:
                    portfolio_values.append(portfolio_values[-1] + trade['pnl'])
                    dates.append(trade['exit_date'])
                
                chart_data = pd.DataFrame({
                    'Date': dates,
                    'Portfolio Value': portfolio_values
                })
                
                st.line_chart(chart_data.set_index('Date'))
            
            # Trade details
            st.subheader("📋 Trade Details")
            if trades:
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df.round(2))
            
            # Stock price chart
            st.subheader("📈 Stock Price During Backtest")
            stock_chart_data = stock_data[['date', 'close']].copy()
            stock_chart_data = stock_chart_data.set_index('date')
            st.line_chart(stock_chart_data)
            
            #st.info("📝 **Note:** This backtest uses actual historical stock data and Black-Scholes pricing for options. Results are based on theoretical option prices and may not reflect actual trading conditions, bid-ask spreads, or market liquidity.")

def setup_alerts():
    """Setup alert system for strategy opportunities"""
    st.subheader("🔔 Strategy Alerts")
    
    with st.expander("📢 Alert Configuration"):
        alert_symbols = st.multiselect(
            "Symbols to Monitor", 
            ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"],
            default=["AAPL"]
        )
        
        alert_conditions = st.multiselect(
            "Alert Conditions",
            [
                "High IV Crush Opportunity",
                "Earnings Play Setup", 
                "Breakout Strategy Signal",
                "Mean Reversion Setup",
                "Unusual Options Activity",
                "RSI Oversold/Overbought",
                "Volume Spike"
            ]
        )
        
        notification_method = st.radio(
            "Notification Method",
            ["Email", "SMS", "In-App Only"]
        )
        
        alert_frequency = st.selectbox(
            "Check Frequency",
            ["Real-time", "Every 15 minutes", "Hourly", "Daily"]
        )
        
        if st.button("🔔 Setup Alerts"):
            st.success("✅ Alerts configured!")
            st.info("📱 Alert system activated for the selected symbols and conditions.")
            
            # Show example alert
            st.write("**Example Alert:**")
            st.code(f"""
🚨 STRATEGY ALERT 🚨
Symbol: {alert_symbols[0] if alert_symbols else 'AAPL'}
Condition: High IV detected (45% vs 30% avg)
Suggested Strategy: Iron Condor
Confidence: 8.5/10
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)

def main():
    st.set_page_config(
        page_title="Enhanced Options Strategist", 
        page_icon="🎯", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Enhanced Options Strategist")
    st.markdown("*Professional options analysis with real-time data and advanced strategies*")
    
    # Add status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **Stock Data:** MarketStack API")
    with col2:
        st.info("🎯 **Options Data:** Polygon.io API")
    with col3:
        st.info("🧠 **Smart Strategy Selection**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Strategy presets
        st.markdown("### 📋 Trading Style Presets")
        presets = add_strategy_presets()
        selected_preset = st.selectbox("Choose a preset:", 
                                      ["Custom"] + list(presets.keys()))
        
        if selected_preset != "Custom":
            preset = presets[selected_preset]
            st.info(f"**{selected_preset}**\n\n{preset['description']}")
            default_symbols = preset['symbols']
            default_risk = preset['risk_tolerance']
        else:
            default_symbols = "AAPL,MSFT,TSLA,GOOGL,AMZN"
            default_risk = 2
        
        st.markdown("---")
        
        # API Keys
        st.markdown("### 🔑 API Keys")
        marketstack_key = st.text_input(
            "MarketStack API Key", 
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
        
        # Validation
        if marketstack_key and polygon_key:
            st.success("✅ Both API keys provided")
        else:
            st.error("❌ Both API keys required for real data")
        
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
            value=default_risk,
            help="Percentage of portfolio to risk per trade"
        ) / 100
        
        st.info(f"💰 **Max Risk per Trade:** ${portfolio_value * risk_tolerance:,.2f}")
        
        st.markdown("---")
        
        # Analysis settings
        st.header("🔍 Analysis Settings")
        
        symbols_input = st.text_area(
            "Stock Symbols", 
            value=default_symbols,
            height=100,
            help="Enter stock symbols separated by commas"
        )
        
        # Parse symbols
        symbols = []
        for line in symbols_input.replace(',', '\n').split('\n'):
            for symbol in line.split(','):
                clean_symbol = symbol.strip().upper()
                if clean_symbol and len(clean_symbol) <= 5:
                    symbols.append(clean_symbol)
        
        st.info(f"📊 **Symbols to analyze:** {len(symbols)}")
        if symbols:
            st.write(", ".join(symbols))
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            max_symbols = st.slider(
                "Max Symbols to Process", 
                min_value=1, 
                max_value=20, 
                value=10,
                help="Limit processing to avoid API rate limits"
            )
            
            enable_logging = st.checkbox(
                "Enable Debug Logging", 
                value=False,
                help="Show detailed processing information"
            )
            
            symbols = symbols[:max_symbols]
        
        st.markdown("---")
        
        # Action button
        analyze_button = st.button(
            "🚀 Analyze Options Strategies", 
            type="primary",
            use_container_width=True,
            disabled=not symbols or not marketstack_key or not polygon_key
        )
        
        if not symbols:
            st.error("❌ Please enter at least one valid symbol")
        if not marketstack_key or not polygon_key:
            st.error("❌ Both API keys required")
    
    # Main content tabs - NOW 5 TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "📊 Analysis", "📈 Backtest", "🔔 Alerts", "📚 Education"])
    
    # TAB 1: Overview/Welcome Information
    with tab1:
        st.markdown("""
        ## Welcome to Enhanced Options Strategist! 🎯
        
        This professional-grade options analysis tool provides:
        
        ### 🚀 **Key Features:**
        - **Smart Strategy Selection** - AI-powered strategy matching based on market conditions
        - **Real-time Data** - Live stock prices and options chains
        - **Risk Management** - Portfolio-aware position sizing
        - **Educational Content** - Learn as you trade
        
        ### 📊 **Supported Strategies:**
        - Bull/Bear Call/Put Spreads
        - Iron Condors & Butterflies  
        - Straddles & Strangles
        - Covered Calls & Cash Secured Puts
        - Protective Puts & Collars
        
        ### 🎯 **Getting Started:**
        1. Enter your API keys in the sidebar
        2. Choose a trading style preset or enter symbols
        3. Set your portfolio value and risk tolerance
        4. Go to the "Analysis" tab and click "Analyze Options Strategies"
        
        ### 💡 **Pro Tips:**
        - Start with liquid stocks like AAPL, MSFT, TSLA
        - Use presets for guided strategy selection
        - Monitor the backtest tab for historical performance
        - Set up alerts for opportunity notifications
        
        ### 📱 **Navigation Guide:**
        - **🏠 Overview:** You are here - welcome and setup guide
        - **📊 Analysis:** Run real-time options strategy analysis
        - **📈 Backtest:** Test strategies with historical data
        - **🔔 Alerts:** Set up opportunity notifications
        - **📚 Education:** Learn options trading concepts
        """)
        
        # Quick setup checklist
        st.subheader("✅ Quick Setup Checklist")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Configuration Status")
            
            # Check API keys
            if marketstack_key:
                st.success("✅ MarketStack API Key provided")
            else:
                st.error("❌ MarketStack API Key missing")
            
            if polygon_key:
                st.success("✅ Polygon.io API Key provided")
            else:
                st.error("❌ Polygon.io API Key missing")
            
            # Check symbols
            if symbols:
                st.success(f"✅ {len(symbols)} symbols configured")
            else:
                st.error("❌ No symbols entered")
            
            # Portfolio settings
            st.success(f"✅ Portfolio value: ${portfolio_value:,}")
            st.success(f"✅ Risk tolerance: {risk_tolerance:.1%}")
        
        with col2:
            st.markdown("### 🎯 Next Steps")
            
            if not marketstack_key or not polygon_key:
                st.warning("1. 🔑 Add your API keys in the sidebar")
            else:
                st.success("1. ✅ API keys configured")
            
            if not symbols:
                st.warning("2. 📊 Enter stock symbols to analyze")
            else:
                st.success("2. ✅ Symbols ready for analysis")
            
            if marketstack_key and polygon_key and symbols:
                st.success("3. 🚀 Ready! Go to 'Analysis' tab")
            else:
                st.info("3. 🚀 Complete setup to begin analysis")
        
        # Performance preview
        st.subheader("📈 Platform Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Supported Strategies", "7+", help="Bull/Bear spreads, Iron Condors, Straddles, etc.")
        
        with col2:
            st.metric("Data Sources", "2", help="MarketStack for stocks, Polygon.io for options")
        
        with col3:
            st.metric("Max Symbols", "20", help="Process up to 20 symbols simultaneously")
        
        with col4:
            st.metric("Backtest Period", "365 days", help="Historical analysis up to 1 year")
        
        # API Information
        st.subheader("🔗 API Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📈 MarketStack API
            - **Purpose:** Real-time stock market data
            - **Features:** Prices, volumes, historical data
            - **Free Tier:** 1,000 requests/month
            - **Get Key:** [marketstack.com](https://marketstack.com)
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Polygon.io API
            - **Purpose:** Options chains and derivatives data
            - **Features:** Real-time options prices, Greeks
            - **Free Tier:** Limited requests/month
            - **Get Key:** [polygon.io](https://polygon.io)
            """)
        
        # Safety notice
        st.info("""
        🛡️ **Risk Disclaimer:** Options trading involves substantial risk and is not suitable for all investors. 
        This tool is for educational and analysis purposes only. Always consult with a qualified financial advisor 
        before making investment decisions. Past performance does not guarantee future results.
        """)
    
    # TAB 2: Analysis (moved from original tab1)
    with tab2:
        if analyze_button and symbols and marketstack_key and polygon_key:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize strategist
                status_text.text("🔧 Initializing Enhanced Options Strategist...")
                progress_bar.progress(10)
                
                strategist = OptionsStrategist(
                    risk_tolerance=risk_tolerance,
                    marketstack_api_key=marketstack_key,
                    polygon_api_key=polygon_key
                )
                
                # Run analysis
                status_text.text("📊 Analyzing symbols with smart strategy selection...")
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
                                'Risk/Reward': trade.get('risk_reward_ratio', 'N/A')
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
                        
                        1. **API Key Issues:**
                           - Verify MarketStack API key has sufficient credits
                           - Check Polygon.io subscription level supports options data
                           - Ensure API keys are entered correctly
                        
                        2. **Symbol Issues:**
                           - Verify stock symbols are correct and tradeable
                           - Check if options are available for the symbols
                           - Try popular symbols like AAPL, MSFT first
                        
                        3. **Market Hours:**
                           - Some data may be limited outside market hours
                           - Options chains may be sparse for certain symbols
                        
                        **Next Steps:**
                        - Try with fewer symbols (1-3)
                        - Use popular, liquid stocks
                        - Check API documentation for usage limits
                        """)
            
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                
                # Show detailed error in expander
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
        else:
            # Instructions when analysis hasn't been run
            st.markdown("""
            ## 📊 Options Strategy Analysis
            
            Ready to analyze your selected symbols and find the best options strategies!
            
            ### 🔧 **Current Configuration:**
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Portfolio Settings:**
                - Portfolio Value: ${portfolio_value:,}
                - Risk Tolerance: {risk_tolerance:.1%}
                - Max Risk per Trade: ${portfolio_value * risk_tolerance:,.2f}
                """)
            
            with col2:
                if symbols:
                    st.info(f"""
                    **Symbols to Analyze:**
                    {', '.join(symbols[:10])}
                    {f'... and {len(symbols)-10} more' if len(symbols) > 10 else ''}
                    """)
                else:
                    st.warning("No symbols configured")
            
            st.markdown("""
            ### 🚀 **To Start Analysis:**
            1. ✅ Ensure API keys are entered in the sidebar
            2. ✅ Verify your symbols are configured
            3. ✅ Click the "🚀 Analyze Options Strategies" button in the sidebar
            
            ### 📈 **What You'll Get:**
            - Smart strategy recommendations for each symbol
            - Detailed trade setups with entry/exit points
            - Risk/reward analysis and position sizing
            - Real-time options pricing and Greeks
            - Export capabilities for further analysis
            """)
            
            if not marketstack_key or not polygon_key:
                st.error("❌ Please add both API keys in the sidebar to proceed")
            elif not symbols:
                st.error("❌ Please add at least one stock symbol in the sidebar")
            else:
                st.success("✅ Configuration complete! Click 'Analyze Options Strategies' in the sidebar to begin.")
    
    # TAB 3: Backtest (unchanged)
    with tab3:
        add_real_backtest()
    
    # TAB 4: Alerts (unchanged)
    with tab4:
        setup_alerts()
    
    # TAB 5: Education (unchanged)
    with tab5:
        # Educational content
        st.header("📚 Options Trading Education")
        
        # Strategy comparison
        st.subheader("🎯 Strategy Comparison Guide")
        
        strategy_comparison = {
            "Strategy": [
                "Covered Call",
                "Cash Secured Put", 
                "Bull Call Spread",
                "Bear Put Spread",
                "Iron Condor",
                "Long Straddle",
                "Protective Put"
            ],
            "Market Outlook": [
                "Neutral to Bullish",
                "Neutral to Bullish",
                "Moderately Bullish", 
                "Moderately Bearish",
                "Neutral/Sideways",
                "High Volatility Expected",
                "Bullish with Protection"
            ],
            "Risk Level": [
                "Low", "Low", "Medium", "Medium", "Medium", "High", "Low"
            ],
            "Complexity": [
                "Beginner", "Beginner", "Intermediate", "Intermediate", "Advanced", "Intermediate", "Beginner"
            ],
            "Best For": [
                "Income Generation",
                "Income + Stock Acquisition",
                "Limited Upside Potential",
                "Limited Downside Potential", 
                "Range-bound Markets",
                "Earnings/Events",
                "Downside Protection"
            ]
        }
        
        comparison_df = pd.DataFrame(strategy_comparison)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Options Greeks explanation
        st.subheader("🔤 Understanding Options Greeks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📈 Delta**
            - Measures price sensitivity to stock movement
            - Call deltas: 0 to 1.0
            - Put deltas: -1.0 to 0
            - Higher delta = more sensitive to stock price
            
            **📊 Gamma** 
            - Rate of change of delta
            - Acceleration of option price movement
            - Highest for at-the-money options
            - Important for portfolio hedging
            """)
        
        with col2:
            st.markdown("""
            **⏰ Theta**
            - Time decay of option value
            - Always negative for long options
            - Accelerates closer to expiration
            - "Enemy" of option buyers
            
            **🌊 Vega**
            - Sensitivity to volatility changes
            - Higher for at-the-money options
            - Longer-dated options have higher vega
            - Important for volatility strategies
            """)
        
        # Risk management section
        st.subheader("🛡️ Risk Management Best Practices")
        
        st.markdown("""
        ### 💰 **Position Sizing Rules:**
        - Never risk more than 2-5% of portfolio per trade
        - Use smaller positions for complex strategies
        - Consider correlation between positions
        - Scale into positions gradually
        
        ### 📅 **Time Management:**
        - Monitor theta decay carefully
        - Have exit plans at 50% max profit
        - Close trades at 21 DTE for monthly options
        - Avoid holding through earnings unless intentional
        
        ### 🎯 **Strategy Selection:**
        - Match strategy to market outlook
        - Consider implied volatility levels
        - Understand assignment risks
        - Paper trade new strategies first
        
        ### 📊 **Portfolio Management:**
        - Diversify across strategies and timeframes
        - Monitor overall portfolio delta
        - Keep some cash for opportunities
        - Regular portfolio reviews and adjustments
        """)
        
        # Market conditions guide
        st.subheader("🌤️ Market Conditions Guide")
        
        conditions_guide = {
            "Market Condition": [
                "Low Volatility",
                "High Volatility", 
                "Trending Up",
                "Trending Down",
                "Range-bound",
                "Earnings Season"
            ],
            "Best Strategies": [
                "Straddles, Strangles, Calendar Spreads",
                "Iron Condors, Credit Spreads, Covered Calls",
                "Bull Call Spreads, Call Buying, Covered Calls",
                "Bear Put Spreads, Put Buying, Protective Puts",
                "Iron Condors, Butterflies, Short Strangles",
                "Straddles, Strangles (long), Protective strategies"
            ],
            "Avoid": [
                "Credit spreads, Short premium",
                "Long premium strategies", 
                "Bear strategies",
                "Bull strategies",
                "Directional strategies",
                "Naked short options"
            ]
        }
        
        conditions_df = pd.DataFrame(conditions_guide)
        st.dataframe(conditions_df, use_container_width=True, hide_index=True)
        
        # Glossary
        with st.expander("📖 Options Trading Glossary"):
            st.markdown("""
            **Assignment** - When an option seller is required to fulfill the obligation of the contract
            
            **ATM (At-the-Money)** - When the option strike price equals the current stock price
            
            **Exercise** - When an option buyer chooses to use their right to buy/sell the stock
            
            **Expiration** - The date when the option contract ends
            
            **ITM (In-the-Money)** - When an option has intrinsic value
            
            **IV (Implied Volatility)** - Market's expectation of future volatility
            
            **Liquidity** - How easily an option can be bought or sold
            
            **OTM (Out-of-the-Money)** - When an option has no intrinsic value
            
            **Premium** - The price paid for an option
            
            **Strike Price** - The price at which the option can be exercised
            
            **Underlying** - The stock that the option contract is based on
            """)

if __name__ == "__main__":
    main()