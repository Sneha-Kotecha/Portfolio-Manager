import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import warnings
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Import your existing components
try:
    from Expert import EnhancedExpert
    from Option_Strategist import OptionsStrategist
    from MOE import MixtureOfExperts, FinancialMixtureOfExperts
    from Gate import FinancialDenseGate, FinancialNoisyTopKGate, AdaptiveFinancialGate
    from PortfolioManager import EnhancedPortfolioManager
    from RiskManager import RiskManager
    from Reporting import ReportGenerator
    
    # Strategy mapping for options strategies
    STRATEGY_MAPPING = {
        0: 'BULL_CALL_SPREAD',
        1: 'BEAR_PUT_SPREAD', 
        2: 'IRON_CONDOR',
        3: 'STRADDLE',
        4: 'STRANGLE', 
        5: 'COVERED_CALL',
        6: 'PROTECTIVE_PUT',
        7: 'CASH_SECURED_PUT',
        8: 'COLLAR',
        9: 'BUTTERFLY'
    }
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all your MOE framework files are in the same directory as this main.py file")
    st.error("Required files: Expert.py, Option_Strategist.py, MOE.py, Gate.py, PortfolioManager.py, RiskManager.py, Reporting.py")
    st.stop()

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(name)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="MOE Trading Framework",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Data Manager with Yahoo Finance (Rate Limited)
# =============================================================================

class YahooFinanceDataManager:
    """Enhanced data manager with aggressive rate limiting to avoid 429 errors"""
    
    def __init__(self):
        self.cache = {}
        self.last_fetch_time = {}
        self.rate_limit_delay = 1.0  # 1 second between requests to avoid rate limits
        self.session_cache_duration = 1800  # 30 minutes cache
        
    def fetch_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock data with aggressive caching and rate limiting"""
        cache_key = f"{symbol}_{period}"
        current_time = time.time()
        
        # Check cache first (30 minute cache)
        if (cache_key in self.cache and 
            cache_key in self.last_fetch_time and 
            current_time - self.last_fetch_time[cache_key] < self.session_cache_duration):
            logger.info(f"Using cached data for {symbol}")
            return self.cache[cache_key]
        
        # Rate limiting - ensure minimum delay between requests
        if self.last_fetch_time:
            last_request = max(self.last_fetch_time.values()) if self.last_fetch_time.values() else 0
            time_since_last = current_time - last_request
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                time.sleep(sleep_time)
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.error(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache the data
            self.cache[cache_key] = data.copy()
            self.last_fetch_time[cache_key] = current_time
            
            logger.info(f"Fetched {len(data)} rows for {symbol} via Yahoo Finance")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        if data.empty or len(data) < 50:
            return data
            
        try:
            # Price-based indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / (data['Volume_SMA'] + 1e-8)
            
            # Volatility and momentum
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Price position indicators
            data['Price_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-8)
            data['High_Low_Pct'] = (data['High'] - data['Low']) / (data['Close'] + 1e-8)
            
            # Advanced indicators
            data['ATR'] = self._calculate_atr(data)
            data['Williams_R'] = self._calculate_williams_r(data)
            
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(0)
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except:
            return pd.Series(0, index=data.index)
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()
            
            williams_r = -100 * (highest_high - data['Close']) / (highest_high - lowest_low + 1e-8)
            
            return williams_r
        except:
            return pd.Series(-50, index=data.index)
    
    def get_live_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current live prices with rate limiting"""
        live_prices = {}
        
        for i, symbol in enumerate(symbols):
            if i > 0:  # Add delay between requests
                time.sleep(self.rate_limit_delay)
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get('currentPrice', info.get('regularMarketPrice', 0.0))
                
                if price == 0.0 or price is None:
                    # Fallback to recent data
                    recent_data = self.fetch_stock_data(symbol, period="1d")
                    if not recent_data.empty:
                        price = recent_data['Close'].iloc[-1]
                    else:
                        price = 0.0
                
                live_prices[symbol] = float(price)
                logger.info(f"Live price for {symbol}: ${price:.2f}")
                
            except Exception as e:
                logger.error(f"Error fetching live price for {symbol}: {e}")
                live_prices[symbol] = 0.0
        
        return live_prices

# =============================================================================
# Session State Management
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data_manager': YahooFinanceDataManager(),
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'moe_framework': None,
        'portfolio_manager': None,
        'risk_manager': None,
        'report_generator': None,
        'options_strategist': None,
        'training_complete': False,
        'signals_dict': {},
        'portfolio_value': 1000000.0,
        'live_prices': {},
        'expert_predictions': {},
        'market_regime': {},
        'volatility_regime': {},
        'training_progress': 0.0,
        'training_status': 'Not Started',
        'portfolio_history': [],
        'risk_metrics': {},
        'strategy_recommendations': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# =============================================================================
# Expert Management
# =============================================================================

def create_moe_framework(symbols: List[str], config: Dict) -> FinancialMixtureOfExperts:
    """Create the complete MOE framework with all components"""
    try:
        # Create individual experts
        experts = {}
        
        # Create enhanced experts for each symbol
        for symbol in symbols:
            expert = EnhancedExpert(
                input_dim=20,
                model_dim=config.get('model_dim', 256),
                num_heads=config.get('num_heads', 8),
                num_layers=4,
                seq_len=config.get('seq_length', 30),
                output_dim=10,
                num_strategies=10,
                dropout=0.1
            )
            experts[f"expert_{symbol}"] = expert
        
        # Create options experts
        options_expert = OptionsStrategist(
            hidden_dim=128,
            num_strategies=10
        )
        experts["options_expert"] = options_expert
        
        # Create gating network
        gating_network = FinancialDenseGate(
            input_dim=20,
            num_experts=len(experts),
            hidden_dim=128,
            dropout=0.1,
            market_aware=True
        )
        
        # Create MOE framework
        moe_framework = FinancialMixtureOfExperts(
            experts=experts,
            gating_network=gating_network,
            input_dim=20
        )
        
        logger.info(f"Created MOE framework with {len(experts)} experts")
        return moe_framework
        
    except Exception as e:
        logger.error(f"Error creating MOE framework: {e}")
        return None

def create_supporting_components(config: Dict):
    """Create portfolio manager, risk manager, and other components"""
    try:
        # Portfolio Manager
        portfolio_manager = EnhancedPortfolioManager(
            initial_capital=config['initial_capital'],
            max_position_size=config['max_position_size'],
            transaction_cost=0.001  # 0.1% transaction cost
        )
        
        # Risk Manager
        risk_manager = RiskManager(
            max_portfolio_risk=config.get('max_portfolio_risk', 0.15),
            max_position_size=config['max_position_size'],
            stop_loss_pct=config['stop_loss'],
            var_confidence=0.95
        )
        
        # Options Strategist
        options_strategist = OptionsStrategist(
            risk_free_rate=0.05,
            max_expiry_days=60
        )
        
        # Report Generator
        report_generator = ReportGenerator(
            output_format='markdown'
        )
        
        return portfolio_manager, risk_manager, options_strategist, report_generator
        
    except Exception as e:
        logger.error(f"Error creating supporting components: {e}")
        return None, None, None, None

def prepare_training_data(data: pd.DataFrame, seq_len: int = 30) -> torch.Tensor:
    """Prepare data for neural network training"""
    try:
        if data.empty or len(data) < seq_len + 10:
            return torch.empty(0)
        
        # Select features for training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Width', 'Volume_Ratio', 
            'Volatility', 'Returns', 'Price_Position', 'ATR', 'Williams_R'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in data.columns]
        
        if len(available_features) < 5:
            logger.warning(f"Insufficient features available: {available_features}")
            return torch.empty(0)
        
        # Extract feature data
        feature_data = data[available_features].fillna(0).values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(feature_data)
        
        # Create sequences
        sequences = []
        for i in range(len(normalized_data) - seq_len + 1):
            seq = normalized_data[i:i+seq_len]
            sequences.append(seq)
        
        if not sequences:
            return torch.empty(0)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(np.array(sequences))
        
        # Pad features to expected input_dim if necessary
        input_dim = 20  # Expected by the expert
        current_features = tensor_data.size(-1)
        
        if current_features < input_dim:
            padding = torch.zeros(tensor_data.size(0), tensor_data.size(1), input_dim - current_features)
            tensor_data = torch.cat([tensor_data, padding], dim=-1)
        elif current_features > input_dim:
            tensor_data = tensor_data[:, :, :input_dim]
        
        logger.info(f"Prepared training data: {tensor_data.shape}")
        return tensor_data
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return torch.empty(0)

# =============================================================================
# Training Functions
# =============================================================================

def train_moe_framework(symbols: List[str], config: Dict, progress_placeholder, status_placeholder):
    """Train the complete MOE framework"""
    try:
        status_placeholder.text("🔄 Initializing MOE Framework...")
        
        # Create MOE framework
        moe_framework = create_moe_framework(symbols, config)
        if not moe_framework:
            st.error("Failed to create MOE framework")
            return False
        
        # Create supporting components
        portfolio_manager, risk_manager, options_strategist, report_generator = create_supporting_components(config)
        
        if not all([portfolio_manager, risk_manager, options_strategist, report_generator]):
            st.error("Failed to create supporting components")
            return False
        
        total_steps = len(symbols) * 2  # Training + validation for each symbol
        current_step = 0
        
        # Training data collection
        training_data = {}
        
        for symbol in symbols:
            status_placeholder.text(f"🔄 Fetching data for {symbol}...")
            
            # Fetch comprehensive training data
            data = st.session_state.data_manager.fetch_stock_data(symbol, period="2y")
            
            if not data.empty:
                # Prepare training data
                training_tensor = prepare_training_data(data, seq_len=config.get('seq_length', 30))
                
                if training_tensor.numel() > 0:
                    training_data[symbol] = {
                        'data': training_tensor,
                        'raw_data': data
                    }
                    
                    current_step += 1
                    progress_placeholder.progress(current_step / total_steps)
                else:
                    logger.warning(f"No valid training data for {symbol}")
            else:
                logger.warning(f"No data fetched for {symbol}")
        
        if not training_data:
            st.error("No valid training data available")
            return False
        
        # Train the MOE framework
        status_placeholder.text("🧠 Training MOE Framework...")
        
        # Prepare combined training data
        all_training_data = []
        all_symbols = []
        
        for symbol, data_dict in training_data.items():
            tensor_data = data_dict['data']
            for i in range(tensor_data.size(0)):
                all_training_data.append(tensor_data[i])
                all_symbols.append(symbol)
        
        if all_training_data:
            combined_data = torch.stack(all_training_data)
            
            # Train MOE framework
            success = train_moe_with_data(moe_framework, combined_data, all_symbols, progress_placeholder, status_placeholder)
            
            if success:
                # Train options strategist
                status_placeholder.text("📊 Training Options Strategist...")
                train_options_strategist(options_strategist, training_data)
                
                # Store components in session state
                st.session_state.moe_framework = moe_framework
                st.session_state.portfolio_manager = portfolio_manager
                st.session_state.risk_manager = risk_manager
                st.session_state.options_strategist = options_strategist
                st.session_state.report_generator = report_generator
                st.session_state.training_complete = True
                st.session_state.training_status = 'Completed'
                
                status_placeholder.text("✅ MOE Framework training completed successfully!")
                progress_placeholder.progress(1.0)
                return True
            else:
                status_placeholder.text("❌ MOE Framework training failed")
                return False
        else:
            st.error("No training data prepared")
            return False
        
    except Exception as e:
        logger.error(f"MOE training failed: {e}")
        status_placeholder.text(f"❌ Training failed: {e}")
        return False

def train_moe_with_data(moe_framework, training_data, symbols, progress_placeholder, status_placeholder):
    """Train the MOE framework with prepared data"""
    try:
        # Set framework to training mode
        moe_framework.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(moe_framework.parameters(), lr=0.001)
        
        # Training parameters
        epochs = 10
        batch_size = 32
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Create batches
            num_samples = training_data.size(0)
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch = training_data[i:end_idx]
                
                if batch.size(0) > 0:
                    # Forward pass
                    outputs = moe_framework(batch)
                    
                    # Calculate loss (you can customize this based on your MOE implementation)
                    if isinstance(outputs, dict):
                        # Handle dictionary output
                        signals = outputs.get('signals', torch.zeros(batch.size(0), 10))
                        loss = torch.mean(torch.abs(signals))  # Simple L1 loss
                        
                        # Add gating loss if available
                        if 'gating_loss' in outputs:
                            loss += outputs['gating_loss']
                    else:
                        # Handle tensor output
                        loss = torch.mean(torch.abs(outputs))
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected at epoch {epoch}, batch {i}")
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(moe_framework.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            # Update progress
            progress = 0.5 + (epoch + 1) / epochs * 0.5  # Second half of progress bar
            progress_placeholder.progress(progress)
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                status_placeholder.text(f"🧠 Training MOE - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                logger.info(f"MOE Training - Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
        
        # Set to evaluation mode
        moe_framework.eval()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in MOE training: {e}")
        return False

def train_options_strategist(options_strategist, training_data):
    """Train the options strategist with market data"""
    try:
        for symbol, data_dict in training_data.items():
            raw_data = data_dict['raw_data']
            
            # Calculate implied volatility and other options metrics
            if len(raw_data) > 30:
                returns = raw_data['Returns'].dropna()
                current_price = raw_data['Close'].iloc[-1]
                volatility = returns.std() * np.sqrt(252)
                
                # Train options strategist (this would depend on your implementation)
                options_strategist.update_market_data(symbol, {
                    'price': current_price,
                    'volatility': volatility,
                    'returns': returns.tolist()[-30:]  # Last 30 returns
                })
        
        logger.info("Options strategist training completed")
        
    except Exception as e:
        logger.error(f"Error training options strategist: {e}")

# =============================================================================
# Signal Generation
# =============================================================================

def generate_comprehensive_signals():
    """Generate comprehensive trading signals using the complete MOE framework"""
    if not st.session_state.training_complete or not st.session_state.moe_framework:
        st.warning("Please train the MOE framework first")
        return {}
    
    signals_dict = {}
    expert_predictions = {}
    market_regimes = {}
    volatility_regimes = {}
    strategy_recommendations = {}
    risk_metrics = {}
    
    try:
        moe_framework = st.session_state.moe_framework
        options_strategist = st.session_state.options_strategist
        risk_manager = st.session_state.risk_manager
        
        for symbol in st.session_state.symbols:
            # Fetch recent data
            data = st.session_state.data_manager.fetch_stock_data(symbol, period="6mo")
            
            if not data.empty:
                # Prepare input data
                input_tensor = prepare_training_data(data, seq_len=30)
                
                if input_tensor.numel() > 0:
                    # Set to evaluation mode
                    moe_framework.eval()
                    
                    with torch.no_grad():
                        # Use the last sequence for prediction
                        latest_input = input_tensor[-1:] if len(input_tensor) > 0 else input_tensor
                        
                        if latest_input.numel() > 0:
                            # Generate MOE prediction
                            moe_outputs = moe_framework(latest_input)
                            
                            # Extract MOE signals
                            if isinstance(moe_outputs, dict):
                                signals = moe_outputs.get('signals', torch.zeros(1, 10))[0].numpy()
                                expert_weights = moe_outputs.get('expert_weights', torch.ones(1, len(moe_framework.experts))/len(moe_framework.experts))[0].numpy()
                                gating_info = moe_outputs.get('gating_info', {})
                            else:
                                signals = moe_outputs[0].numpy()
                                expert_weights = np.ones(len(moe_framework.experts)) / len(moe_framework.experts)
                                gating_info = {}
                            
                            # Calculate overall signal strength
                            signal_strength = float(np.mean(signals))
                            confidence = float(np.std(signals))  # Use std as confidence measure
                            
                            # Get options strategy recommendations
                            current_price = data['Close'].iloc[-1]
                            volatility = data['Volatility'].iloc[-1] if 'Volatility' in data.columns else 0.2
                            
                            options_rec = options_strategist.recommend_strategy(
                                symbol=symbol,
                                current_price=current_price,
                                volatility=volatility,
                                market_outlook=signal_strength
                            )
                            
                            # Risk assessment
                            position_risk = risk_manager.assess_position_risk(
                                symbol=symbol,
                                signal=signal_strength,
                                volatility=volatility,
                                current_price=current_price
                            )
                            
                            # Combine all information
                            signals_dict[symbol] = {
                                'signal': signal_strength,
                                'confidence': confidence,
                                'moe_signals': signals.tolist(),
                                'expert_weights': expert_weights.tolist(),
                                'reasoning': f"MOE prediction: {signal_strength:.3f}, Confidence: {confidence:.3f}"
                            }
                            
                            strategy_recommendations[symbol] = options_rec
                            risk_metrics[symbol] = position_risk
                            
                            # Store detailed predictions
                            expert_predictions[symbol] = {
                                'signals': signals.tolist(),
                                'expert_weights': expert_weights.tolist(),
                                'gating_info': gating_info
                            }
                            
                            logger.info(f"Generated comprehensive signals for {symbol}: {signal_strength:.3f}")
        
        # Store results in session state
        st.session_state.signals_dict = signals_dict
        st.session_state.expert_predictions = expert_predictions
        st.session_state.strategy_recommendations = strategy_recommendations
        st.session_state.risk_metrics = risk_metrics
        st.session_state.market_regime = market_regimes
        st.session_state.volatility_regime = volatility_regimes
        
        return signals_dict
        
    except Exception as e:
        logger.error(f"Error generating comprehensive signals: {e}")
        st.error(f"Signal generation failed: {e}")
        return {}

def execute_portfolio_rebalancing():
    """Execute portfolio rebalancing using the portfolio manager"""
    if not st.session_state.signals_dict or not st.session_state.portfolio_manager:
        st.warning("Please generate signals first and ensure portfolio manager is initialized")
        return False
    
    try:
        portfolio_manager = st.session_state.portfolio_manager
        risk_manager = st.session_state.risk_manager
        
        # Get live prices
        live_prices = st.session_state.data_manager.get_live_prices(st.session_state.symbols)
        st.session_state.live_prices = live_prices
        
        # Apply risk management filters
        filtered_signals = risk_manager.filter_signals(
            st.session_state.signals_dict,
            live_prices,
            st.session_state.risk_metrics
        )
        
        # Execute rebalancing
        rebalance_result = portfolio_manager.rebalance(filtered_signals, live_prices)
        
        if rebalance_result:
            # Update portfolio history
            portfolio_value = portfolio_manager.get_total_value(live_prices)
            st.session_state.portfolio_value = portfolio_value
            
            # Store portfolio history
            if 'portfolio_history' not in st.session_state:
                st.session_state.portfolio_history = []
            
            st.session_state.portfolio_history.append({
                'timestamp': datetime.now(),
                'total_value': portfolio_value,
                'positions': portfolio_manager.get_positions(),
                'signals': st.session_state.signals_dict.copy()
            })
            
            logger.info(f"Portfolio rebalanced successfully. New value: ${portfolio_value:,.2f}")
            return True
        else:
            logger.error("Portfolio rebalancing failed")
            return False
        
    except Exception as e:
        logger.error(f"Error in portfolio rebalancing: {e}")
        st.error(f"Rebalancing failed: {e}")
        return False
    

# --- UI Components ---

def display_sidebar():
    st.sidebar.title("🚀 MOE Framework")
    st.sidebar.markdown("---")
    
    # Model Configuration
    st.sidebar.header("Model Configuration")
    symbols_input = st.sidebar.text_area(
        "Enter symbols (comma-separated)",
        value=",".join(st.session_state.symbols),
        help="Enter stock symbols separated by commas"
    )
    if st.sidebar.button("Update Symbols"):
        new_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if new_symbols != st.session_state.symbols:
            st.session_state.symbols = new_symbols
            st.session_state.training_complete = False
            st.session_state.training_status = 'Not Started'
            st.rerun()

    st.sidebar.subheader("🧠 Training Parameters")
    seq_length = st.sidebar.slider("Sequence Length", 10, 60, 30)
    model_dim = st.sidebar.slider("Model Dimension", 128, 512, 256)
    num_heads = st.sidebar.slider("Attention Heads", 4, 16, 8)

    st.sidebar.subheader("💰 Portfolio Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=int(st.session_state.portfolio_value),
        step=10000
    )

    st.sidebar.subheader("⚠️ Risk Management")
    max_position_size = st.sidebar.slider("Max Position Size (%)", 5, 50, 20)
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 20, 5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 System Status")
    status_color = "🟢" if st.session_state.training_complete else "🔴"
    st.sidebar.write(f"{status_color} Training Status: {st.session_state.training_status}")
    if st.session_state.signals_dict:
        st.sidebar.write(f"📊 Active Signals: {len(st.session_state.signals_dict)}")

    return {
        'seq_length': seq_length,
        'model_dim': model_dim,
        'num_heads': num_heads,
        'initial_capital': initial_capital,
        'max_position_size': max_position_size / 100,
        'stop_loss': stop_loss / 100
    }


def display_training_tab():
    st.header("🧠 Expert Training")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Training Configuration")
        st.write(f"**Selected Symbols:** {', '.join(st.session_state.symbols)}")
        st.write(f"**Number of Experts:** {len(st.session_state.symbols)}")
        if st.session_state.training_complete:
            st.success("✅ Training completed! All experts are ready.")
        else:
            st.warning("⚠️ Training required before generating signals.")
    with col2:
        st.subheader("Quick Actions")
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            progress_placeholder = st.progress(0)
            status_placeholder = st.empty()
            success = train_moe_framework(
                st.session_state.symbols,
                {'seq_length': st.session_state.training_parameters['seq_length'],
                 'model_dim': st.session_state.training_parameters['model_dim'],
                 'num_heads': st.session_state.training_parameters['num_heads'],
                 'initial_capital': st.session_state.training_parameters['initial_capital'],
                 'max_position_size': st.session_state.training_parameters['max_position_size'],
                 'stop_loss': st.session_state.training_parameters['stop_loss']},
                progress_placeholder,
                status_placeholder
            )
            if success:
                st.success("Training completed successfully!")
                st.balloons()
            else:
                st.error("Training failed. Check logs for details.")
        if st.button("🔄 Reset Training", use_container_width=True):
            st.session_state.training_complete = False
            st.session_state.training_status = 'Not Started'
            st.session_state.signals_dict = {}
            st.rerun()


def display_signals_tab():
    st.header("📊 Trading Signals")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Generate Signals", type="primary", use_container_width=True):
            with st.spinner("Generating signals..."):
                signals = generate_comprehensive_signals()
                if signals:
                    st.success(f"Generated signals for {len(signals)} symbols!")
                else:
                    st.error("Failed to generate signals")
    with col2:
        if st.button("💰 Update Live Prices", use_container_width=True):
            with st.spinner("Fetching live prices..."):
                live_prices = st.session_state.data_manager.get_live_prices(st.session_state.symbols)
                st.session_state.live_prices = live_prices
                st.success("Live prices updated!")
    if st.session_state.signals_dict:
        st.subheader("🎯 Current Signals")
        signals_data = []
        for symbol, info in st.session_state.signals_dict.items():
            action = "🟢 BUY" if info['signal'] > 0.1 else "🔴 SELL" if info['signal'] < -0.1 else "🟡 HOLD"
            signals_data.append({
                'Symbol': symbol,
                'Signal': f"{info['signal']:.3f}",
                'Confidence': f"{info['confidence']:.3f}",
                'Action': action,
                'Reasoning': info.get('reasoning', 'N/A')
            })
        st.dataframe(signals_data, use_container_width=True)
        symbols = [row['Symbol'] for row in signals_data]
        values = [float(row['Signal']) for row in signals_data]
        fig = go.Figure(data=go.Bar(x=symbols, y=values,
                                     marker_color=['green' if v>0 else 'red' if v<0 else 'gray' for v in values],
                                     text=[f"{v:.3f}" for v in values], textposition='auto'))
        fig.update_layout(title="Trading Signal Strength", xaxis_title="Symbol", yaxis_title="Signal Strength", yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig, use_container_width=True)
    if st.session_state.live_prices:
        st.subheader("💰 Live Prices")
        cols = st.columns(len(st.session_state.live_prices))
        for idx, (sym, price) in enumerate(st.session_state.live_prices.items()):
            with cols[idx]: st.metric(sym, f"${price:.2f}")


def display_analysis_tab():
    st.header("📈 Market Analysis")
    if not st.session_state.expert_predictions:
        st.warning("Generate signals first to see detailed analysis")
        return
    symbol = st.selectbox("Select Symbol", st.session_state.symbols)
    if symbol in st.session_state.expert_predictions:
        data = st.session_state.expert_predictions[symbol]
        # Signal breakdown radar
        signals = data['signals']
        if len(signals) >= 5:
            categories = [f"Signal {i+1}" for i in range(len(signals))]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=signals, theta=categories, fill='toself'))
            fig.update_layout(title=f"{symbol} Signal Breakdown", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)
        # Expert weights bar
        weights = data.get('expert_weights', [])
        if weights:
            exp_names = list(st.session_state.moe_framework.experts.keys())
            fig2 = go.Figure(data=go.Bar(x=exp_names, y=weights, text=[f"{w:.2f}" for w in weights], textposition='auto'))
            fig2.update_layout(title=f"{symbol} Expert Weights", xaxis_title="Expert", yaxis_title="Weight")
            st.plotly_chart(fig2, use_container_width=True)
        # Gating info table
        gating = data.get('gating_info', {})
        if gating:
            df_gating = pd.DataFrame.from_dict(gating)
            st.subheader("Gating Network Info")
            st.dataframe(df_gating, use_container_width=True)


def display_portfolio_tab():
    st.header("💼 Portfolio Overview")
    if 'portfolio_history' not in st.session_state or not st.session_state.portfolio_history:
        st.warning("Run a rebalance to view portfolio history")
        return
    history = pd.DataFrame([{
        'Time': entry['timestamp'],
        'Value': entry['total_value']
    } for entry in st.session_state.portfolio_history])
    fig = px.line(history, x='Time', y='Value', title='Portfolio Value Over Time')
    st.plotly_chart(fig, use_container_width=True)
    latest = st.session_state.portfolio_history[-1]
    st.subheader("Current Positions")
    positions = latest['positions']
    pos_df = pd.DataFrame([{'Symbol': s, 'Quantity': q, 'Market Value': v} for s, (q,v) in positions.items()])
    st.dataframe(pos_df, use_container_width=True)


def display_reports_tab():
    st.header("📋 Reports")
    if st.session_state.training_complete:
        report_md = st.session_state.report_generator.generate()
        st.markdown(report_md)
    else:
        st.warning("Generate training and signals to enable reports")


def main():
    initialize_session_state()
    st.session_state.training_parameters = display_sidebar()
    tabs = st.tabs(["Training", "Signals", "Analysis", "Portfolio", "Reports"])
    with tabs[0]: display_training_tab()
    with tabs[1]: display_signals_tab()
    with tabs[2]: display_analysis_tab()
    with tabs[3]: display_portfolio_tab()
    with tabs[4]: display_reports_tab()

if __name__ == '__main__':
    main()
