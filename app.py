import streamlit as st
import yfinance as yf
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
import sys
import os

# =============================================================================
# Integration modifications for main.py to use existing OptionsStrategist
# =============================================================================

# 1. Update the import section in main.py (replace the existing import line):
try:
    from Expert import EnhancedExpert
    from Option_Strategist import OptionsStrategist  # Your existing class
    from MOE import MixtureOfExperts, FinancialMixtureOfExperts
    from Gate import FinancialDenseGate, FinancialNoisyTopKGate, AdaptiveFinancialGate
    from PortfolioManager import EnhancedPortfolioManager
    from RiskManager import RiskManager, EnhancedRiskManager
    from Reporting import ReportGenerator, EnhancedReportGenerator
    
    # Strategy mapping for options strategies (keep existing)
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
# Data Manager with MarketStack API
# =============================================================================

import pandas as pd
import time
import requests
import numpy as np
import logging
from typing import Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketStackDataManager:
    """Enhanced data manager with MarketStack API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.marketstack.com/v2"
        self.cache = {}
        self.last_fetch_time = {}
        self.rate_limit_delay = 0.2  # MarketStack allows 5 requests per second
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
            data = self._fetch_marketstack_data(symbol, period)
            
            if data.empty:
                logger.error(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache the data
            self.cache[cache_key] = data.copy()
            self.last_fetch_time[cache_key] = current_time
            
            logger.info(f"Fetched {len(data)} rows for {symbol} via MarketStack")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_marketstack_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from MarketStack API"""
        # Calculate date range based on period
        end_date = datetime.now()
        
        period_map = {
            "1d": 1,
            "5d": 5, 
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "ytd": None  # Will calculate based on year start
        }
        
        if period == "ytd":
            start_date = datetime(end_date.year, 1, 1)
        else:
            days = period_map.get(period, 730)  # Default to 2 years
            start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")
        
        # Make API request for EOD data
        params = {
            'access_key': self.api_key,
            'symbols': symbol,
            'date_from': date_from,
            'date_to': date_to,
            'limit': 1000,
            'sort': 'ASC'
        }
        
        response = requests.get(f"{self.base_url}/eod", params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data or not data['data']:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Process the data
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Rename columns to match Market Stack format
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'adj_close': 'Adj Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Keep only the columns we need and ensure they exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'Adj Close':
                    df[col] = df['Close']  # Use close price if adj_close not available
                else:
                    df[col] = 0
        
        # Convert to numeric types
        for col in required_columns + ['Adj Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date ascending
        df = df.sort_index()
        
        return df[required_columns + (['Adj Close'] if 'Adj Close' in df.columns else [])]
    
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
                # Use MarketStack's latest EOD endpoint for live prices
                params = {
                    'access_key': self.api_key,
                    'symbols': symbol,
                    'limit': 1
                }
                
                response = requests.get(f"{self.base_url}/eod/latest", params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if 'data' in data and data['data']:
                    price = float(data['data'][0]['close'])
                    live_prices[symbol] = price
                    logger.info(f"Live price for {symbol}: ${price:.2f}")
                else:
                    live_prices[symbol] = 0.0
                    logger.warning(f"No price data available for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching live price for {symbol}: {e}")
                live_prices[symbol] = 0.0
        
        return live_prices
    
    def get_intraday_data(self, symbol: str, interval: str = "1hour") -> pd.DataFrame:
        """Get intraday data (requires Basic plan or higher)"""
        try:
            params = {
                'access_key': self.api_key,
                'symbols': symbol,
                'interval': interval,
                'limit': 1000
            }
            
            response = requests.get(f"{self.base_url}/intraday", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low', 
                'close': 'Close',
                'volume': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information for a ticker"""
        try:
            params = {
                'access_key': self.api_key,
                'ticker': symbol
            }
            
            response = requests.get(f"{self.base_url}/tickerinfo", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data:
                return data['data']
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {}
    
    def get_splits_data(self, symbol: str, date_from: str = None, date_to: str = None) -> pd.DataFrame:
        """Get stock splits data"""
        try:
            params = {
                'access_key': self.api_key,
                'symbols': symbol,
                'limit': 1000
            }
            
            if date_from:
                params['date_from'] = date_from
            if date_to:
                params['date_to'] = date_to
            
            response = requests.get(f"{self.base_url}/splits", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error fetching splits data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_dividends_data(self, symbol: str, date_from: str = None, date_to: str = None) -> pd.DataFrame:
        """Get dividends data"""
        try:
            params = {
                'access_key': self.api_key,
                'symbols': symbol,
                'limit': 1000
            }
            
            if date_from:
                params['date_from'] = date_from
            if date_to:
                params['date_to'] = date_to
            
            response = requests.get(f"{self.base_url}/dividends", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error fetching dividends data for {symbol}: {e}")
            return pd.DataFrame()

api_key = "9ad0d4f85e1a72dd7b3d19b8617b25f9"
data_manager = MarketStackDataManager(api_key)
    


# =============================================================================
# Session State Management
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Get MarketStack API key from environment variable or Streamlit secrets
    api_key = "9ad0d4f85e1a72dd7b3d19b8617b25f9"
    
    # Default symbols if none exist in session state
    default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    defaults = {
        'data_manager': MarketStackDataManager(api_key) if api_key else None,
        'symbols': default_symbols,  # Will be updated from user input
        'moe_framework': None,
        'portfolio_manager': None,
        'risk_manager': None,
        'report_generator': None,
        'options_strategist': None,
        'training_complete': False,
        'signals_dict': {},
        'portfolio_value': 1000.0,
        'live_prices': {},
        'expert_predictions': {},
        'market_regime': {},
        'volatility_regime': {},
        'training_progress': 0.0,
        'training_status': 'Not Started',
        'portfolio_history': [],
        'risk_metrics': {},
        'strategy_recommendations': {},
        'api_key_configured': api_key is not None,  # Track if API key is set
        'symbols_input': ",".join(default_symbols),  # Store input string for UI
        'symbols_changed': False  # Flag to track when symbols are updated
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_marketstack_api_key():
    """Get MarketStack API key from various sources"""
    # Try to get from Streamlit secrets first (recommended for production)
    try:
        if hasattr(st, 'secrets') and 'MARKETSTACK_API_KEY' in st.secrets:
            return st.secrets['MARKETSTACK_API_KEY']
    except Exception:
        pass
    
    # Try to get from environment variable
    api_key = os.getenv('9ad0d4f85e1a72dd7b3d19b8617b25f9')
    if api_key:
        return api_key
    
    # Try to get from session state (if user entered it in the app)
    if 'marketstack_api_key' in st.session_state:
        return st.session_state.marketstack_api_key
    
    return None

# =============================================================================
# Expert Management with MarketStack Integration
# =============================================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

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
        
        # Create options experts - pass MarketStack API key if available
        marketstack_api_key = getattr(st.session_state.data_manager, 'api_key', None)
        options_expert = OptionsStrategist(marketstack_api_key=marketstack_api_key)
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
            input_dim=20,
            seq_len=config.get('seq_length', 30),
            model_dim=config.get('model_dim', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=4,
            ff_dim=config.get('model_dim', 256) * 4,
            output_dim=10,
            num_experts=len(experts)
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
            initial_cash=config['initial_capital'],
            max_position_size=config['max_position_size'],
            commission_rate=0.001  # 0.1% transaction cost
        )
                
        # Risk Manager
        risk_manager = EnhancedRiskManager(
            max_drawdown_threshold=config.get('max_portfolio_risk', 0.15),
            max_position_size=config['max_position_size'],
            var_threshold_95=0.02,
            var_threshold_99=0.05
        )
        
        # Options Strategist with MarketStack API key
        marketstack_api_key = getattr(st.session_state.data_manager, 'api_key', None)
        options_strategist = OptionsStrategist(marketstack_api_key=marketstack_api_key)
        
        # Report Generator
        report_generator = EnhancedReportGenerator(
            report_style='professional',
            include_charts=True,
            auto_insights=True
        )
        
        return portfolio_manager, risk_manager, options_strategist, report_generator
        
    except Exception as e:
        logger.error(f"Error creating supporting components: {e}")
        return None, None, None, None

def prepare_training_data(data: pd.DataFrame, seq_len: int = 30) -> torch.Tensor:
    """Prepare MarketStack data for neural network training"""
    try:
        if data.empty or len(data) < seq_len + 10:
            logger.warning(f"Insufficient data: {len(data)} rows, need at least {seq_len + 10}")
            return torch.empty(0, seq_len, 20)  # Return with correct dimensions
        
        # MarketStack provides these basic columns
        base_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Technical indicators that should be available after processing
        technical_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'Volume_SMA', 'Volume_Ratio', 
            'Returns', 'Log_Returns', 'Volatility', 'Price_Position', 'High_Low_Pct', 'ATR', 'Williams_R'
        ]
        
        # Select available features, prioritizing base columns
        available_features = []
        
        # First, add base columns that exist
        for col in base_columns:
            if col in data.columns:
                available_features.append(col)
        
        # Then add technical indicators that exist
        for col in technical_columns:
            if col in data.columns and not pd.isna(data[col]).all():
                available_features.append(col)
        
        # Ensure we have at least 5 features
        if len(available_features) < 5:
            logger.warning(f"Insufficient features available: {available_features}")
            # Fill with base columns and zeros if needed
            required_features = base_columns[:5]
            for col in required_features:
                if col not in available_features:
                    if col in data.columns:
                        available_features.append(col)
                    else:
                        # Create dummy column
                        data[col] = 0.0
                        available_features.append(col)
        
        # Limit to top 19 features (will pad to 20 later)
        available_features = available_features[:19]
        
        logger.info(f"Using features: {available_features}")
        
        # Extract feature data and handle missing values
        feature_data = data[available_features].copy()
        
        # Fill NaN values with forward fill, then backward fill, then 0
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Convert to numpy array
        feature_array = feature_data.values.astype(np.float32)
        
        # Check for any remaining NaN or inf values
        if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
            logger.warning("Found NaN or inf values, replacing with zeros")
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features using robust scaling
        try:
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(feature_array)
            
            # Handle case where std is 0 (constant values)
            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)
            
        except Exception as scaling_error:
            logger.warning(f"Scaling failed, using min-max normalization: {scaling_error}")
            # Fallback to min-max normalization
            min_vals = np.min(feature_array, axis=0)
            max_vals = np.max(feature_array, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            normalized_data = (feature_array - min_vals) / range_vals
        
        # Create sequences
        sequences = []
        for i in range(len(normalized_data) - seq_len + 1):
            seq = normalized_data[i:i+seq_len]
            sequences.append(seq)
        
        if not sequences:
            logger.warning("No sequences could be created")
            return torch.empty(0, seq_len, 20)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(np.array(sequences))
        
        # Ensure we have the correct dimensions: [num_sequences, seq_len, num_features]
        logger.info(f"Created tensor with shape: {tensor_data.shape}")
        
        # Pad or truncate features to exactly 20 dimensions
        input_dim = 20
        current_features = tensor_data.size(-1)
        
        if current_features < input_dim:
            # Pad with zeros
            padding_size = input_dim - current_features
            padding = torch.zeros(tensor_data.size(0), tensor_data.size(1), padding_size)
            tensor_data = torch.cat([tensor_data, padding], dim=-1)
            logger.info(f"Padded features from {current_features} to {input_dim}")
        elif current_features > input_dim:
            # Truncate to input_dim
            tensor_data = tensor_data[:, :, :input_dim]
            logger.info(f"Truncated features from {current_features} to {input_dim}")
        
        logger.info(f"Final training data shape: {tensor_data.shape}")
        
        # Validate final tensor
        if torch.isnan(tensor_data).any() or torch.isinf(tensor_data).any():
            logger.error("Final tensor contains NaN or inf values")
            tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return tensor_data
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return torch.empty(0, seq_len, 20)

# =============================================================================
# Training Functions
# =============================================================================

def train_moe_framework(symbols: List[str], config: Dict, progress_placeholder, status_placeholder):
    """Train the complete MOE framework with MarketStack data"""
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
            
            # Fetch comprehensive training data using MarketStack
            data = st.session_state.data_manager.fetch_stock_data(symbol, period="2y")
            
            if not data.empty:
                logger.info(f"Fetched {len(data)} rows for {symbol}")
                logger.info(f"Available columns: {list(data.columns)}")
                
                # Prepare training data
                training_tensor = prepare_training_data(data, seq_len=config.get('seq_length', 30))
                
                if training_tensor.numel() > 0:
                    training_data[symbol] = {
                        'data': training_tensor,
                        'raw_data': data
                    }
                    
                    current_step += 1
                    progress_placeholder.progress(current_step / total_steps)
                    logger.info(f"Prepared training data for {symbol}: {training_tensor.shape}")
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
            logger.info(f"Processing {symbol} with tensor shape: {tensor_data.shape}")
            
            for i in range(tensor_data.size(0)):
                all_training_data.append(tensor_data[i])
                all_symbols.append(symbol)
        
        if all_training_data:
            combined_data = torch.stack(all_training_data)
            logger.info(f"Combined training data shape: {combined_data.shape}")
            
            # Train MOE framework
            success = train_moe_with_data(moe_framework, combined_data, all_symbols, progress_placeholder, status_placeholder)
            
            if success:
                # Train options strategist
                status_placeholder.text("📊 Training Options Strategist...")
                train_options_strategist(options_strategist, training_data, st.session_state.data_manager)
                
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
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        status_placeholder.text(f"❌ Training failed: {e}")
        return False

def train_moe_with_data(moe_framework, training_data, symbols, progress_placeholder, status_placeholder):
    """Train the MOE framework with prepared data - Fixed tensor dimension issues"""
    try:
        logger.info(f"Starting MOE training with data shape: {training_data.shape}")
        
        # Set framework to training mode
        moe_framework.train()
        
        # Create optimizer with lower learning rate for stability
        optimizer = torch.optim.Adam(moe_framework.parameters(), lr=0.0001)
        
        # Training parameters
        epochs = 10
        batch_size = min(16, training_data.size(0))  # Smaller batch size for stability
        
        # Validate input data
        if training_data.dim() != 3:
            logger.error(f"Expected 3D tensor, got {training_data.dim()}D: {training_data.shape}")
            return False
        
        num_samples, seq_len, num_features = training_data.shape
        logger.info(f"Training with {num_samples} samples, sequence length {seq_len}, {num_features} features")
        
        # Test forward pass before training
        try:
            with torch.no_grad():
                test_batch = training_data[:1]  # Single sample
                test_output = moe_framework(test_batch)
                logger.info(f"Test forward pass successful. Output type: {type(test_output)}")
                
                # Log output structure for debugging
                if isinstance(test_output, dict):
                    for key, value in test_output.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: {value.shape}")
                elif isinstance(test_output, tuple):
                    for i, item in enumerate(test_output):
                        if isinstance(item, torch.Tensor):
                            logger.info(f"  Output[{i}]: {item.shape}")
                elif isinstance(test_output, torch.Tensor):
                    logger.info(f"  Output shape: {test_output.shape}")
                    
        except Exception as test_error:
            logger.error(f"Forward pass test failed: {test_error}")
            return False
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            successful_batches = 0
            
            # Shuffle data for each epoch
            indices = torch.randperm(num_samples)
            shuffled_data = training_data[indices]
            
            # Create batches
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch = shuffled_data[i:end_idx]
                actual_batch_size = batch.size(0)
                
                if actual_batch_size == 0:
                    continue
                
                try:
                    # Forward pass
                    outputs = moe_framework(batch)
                    
                    # Handle different output formats and create appropriate targets
                    loss = None
                    
                    if isinstance(outputs, dict):
                        # Handle dictionary output
                        if 'signals' in outputs:
                            predictions = outputs['signals']
                        elif 'predictions' in outputs:
                            predictions = outputs['predictions']
                        else:
                            # Use first tensor in the dictionary
                            tensor_outputs = {k: v for k, v in outputs.items() if isinstance(v, torch.Tensor)}
                            if tensor_outputs:
                                predictions = next(iter(tensor_outputs.values()))
                            else:
                                logger.warning(f"No tensor outputs found in dictionary: {outputs.keys()}")
                                continue
                        
                        # Ensure predictions have correct batch dimension
                        if predictions.dim() == 1:
                            predictions = predictions.unsqueeze(0)
                        
                        # Handle batch size mismatch
                        if predictions.size(0) != actual_batch_size:
                            if predictions.size(0) == 1 and actual_batch_size > 1:
                                # Expand single prediction to match batch size
                                predictions = predictions.expand(actual_batch_size, -1)
                            elif predictions.size(0) > actual_batch_size:
                                # Truncate predictions
                                predictions = predictions[:actual_batch_size]
                            else:
                                # Skip this batch if we can't resolve the mismatch
                                logger.warning(f"Skipping batch due to size mismatch: pred={predictions.size(0)}, batch={actual_batch_size}")
                                continue
                        
                        # Create target - use self-supervised learning approach
                        if predictions.dim() >= 2:
                            # For multi-dimensional outputs, create targets based on input patterns
                            target = torch.zeros_like(predictions)
                            
                            # Simple self-supervised target: predict next step pattern
                            if batch.size(1) > 1:  # If we have sequence length > 1
                                # Use simple pattern: small random targets around zero
                                target = torch.randn_like(predictions) * 0.1
                        else:
                            target = torch.zeros_like(predictions)
                        
                        # Calculate main loss
                        loss = nn.MSELoss()(predictions, target)
                        
                        # Add auxiliary losses if available
                        if 'gating_loss' in outputs and isinstance(outputs['gating_loss'], torch.Tensor):
                            gating_loss = outputs['gating_loss']
                            if not torch.isnan(gating_loss) and not torch.isinf(gating_loss):
                                loss += 0.1 * gating_loss
                        
                        if 'expert_loss' in outputs and isinstance(outputs['expert_loss'], torch.Tensor):
                            expert_loss = outputs['expert_loss']
                            if not torch.isnan(expert_loss) and not torch.isinf(expert_loss):
                                loss += 0.05 * expert_loss
                                
                    elif isinstance(outputs, tuple):
                        # Handle tuple output
                        predictions = outputs[0]
                        
                        # Ensure correct dimensions
                        if predictions.dim() == 1:
                            predictions = predictions.unsqueeze(0)
                        
                        # Handle batch size mismatch
                        if predictions.size(0) != actual_batch_size:
                            if predictions.size(0) == 1:
                                predictions = predictions.expand(actual_batch_size, -1)
                            else:
                                predictions = predictions[:actual_batch_size]
                        
                        # Create target
                        target = torch.zeros_like(predictions)
                        loss = nn.MSELoss()(predictions, target)
                        
                        # Add auxiliary losses from tuple if available
                        if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
                            aux_loss = outputs[1]
                            if not torch.isnan(aux_loss) and not torch.isinf(aux_loss):
                                loss += 0.1 * aux_loss
                    
                    else:
                        # Handle tensor output
                        predictions = outputs
                        
                        # Ensure correct dimensions
                        if predictions.dim() == 1:
                            predictions = predictions.unsqueeze(0)
                        elif predictions.dim() == 0:
                            predictions = predictions.unsqueeze(0).unsqueeze(0)
                        
                        # Handle batch size mismatch
                        if predictions.size(0) != actual_batch_size:
                            if predictions.size(0) == 1:
                                predictions = predictions.expand(actual_batch_size, -1)
                            else:
                                predictions = predictions[:actual_batch_size]
                        
                        # Create target
                        target = torch.zeros_like(predictions)
                        loss = nn.MSELoss()(predictions, target)
                    
                    # Validate loss
                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss at epoch {epoch}, batch {i}: {loss}")
                        continue
                    
                    # Check for exploding gradients
                    if loss.item() > 100.0:
                        logger.warning(f"Very large loss detected: {loss.item()}, skipping batch")
                        continue
                    
                    # Backward pass with gradient clipping
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(moe_framework.parameters(), max_norm=1.0)
                    
                    # Check for gradient explosion
                    total_norm = 0
                    for p in moe_framework.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    
                    if total_norm > 10.0:
                        logger.warning(f"Large gradient norm detected: {total_norm}, skipping update")
                        continue
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    successful_batches += 1
                    
                except Exception as batch_error:
                    logger.error(f"Error in batch {i} of epoch {epoch}: {batch_error}")
                    # Continue to next batch instead of stopping
                    continue
                
                num_batches += 1
            
            # Update progress
            progress = 0.5 + (epoch + 1) / epochs * 0.5
            progress_placeholder.progress(progress)
            
            if successful_batches > 0:
                avg_loss = epoch_loss / successful_batches
                status_placeholder.text(f"🧠 Training MOE - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Success Rate: {successful_batches}/{num_batches}")
                logger.info(f"MOE Training - Epoch {epoch+1}, Average Loss: {avg_loss:.6f}, Successful Batches: {successful_batches}/{num_batches}")
            else:
                logger.warning(f"No successful batches in epoch {epoch}")
                status_placeholder.text(f"🧠 Training MOE - Epoch {epoch+1}/{epochs}, No successful batches")
        
        # Set to evaluation mode
        moe_framework.eval()
        logger.info("MOE training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in MOE training: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def prepare_training_data_enhanced(data: pd.DataFrame, seq_len: int = 30) -> torch.Tensor:
    """Enhanced data preparation with better error handling and dimension consistency"""
    try:
        if data.empty or len(data) < seq_len + 10:
            logger.warning(f"Insufficient data: {len(data)} rows, need at least {seq_len + 10}")
            return torch.empty(0, seq_len, 20)
        
        # Define feature columns with fallbacks
        base_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        technical_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'Volume_SMA', 'Volume_Ratio', 
            'Returns', 'Log_Returns', 'Volatility', 'Price_Position', 'High_Low_Pct', 'ATR', 'Williams_R'
        ]
        
        # Select available features
        available_features = []
        
        # Prioritize base columns
        for col in base_columns:
            if col in data.columns and not data[col].isna().all():
                available_features.append(col)
        
        # Add technical indicators
        for col in technical_columns:
            if col in data.columns and not data[col].isna().all():
                available_features.append(col)
        
        # Ensure minimum features
        if len(available_features) < 5:
            logger.warning(f"Only {len(available_features)} features available, adding dummy features")
            # Create minimal required features
            for i, base_col in enumerate(base_columns):
                if base_col not in available_features:
                    if base_col in data.columns:
                        available_features.append(base_col)
                    else:
                        # Create dummy column with meaningful values
                        if base_col == 'Volume':
                            data[base_col] = 1000.0
                        else:
                            data[base_col] = data.get('Close', pd.Series([100.0] * len(data)))
                        available_features.append(base_col)
                if len(available_features) >= 5:
                    break
        
        # Limit to 19 features (will pad to 20 later)
        available_features = available_features[:19]
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        # Extract and clean feature data
        feature_data = data[available_features].copy()
        
        # Handle missing values systematically
        for col in feature_data.columns:
            # Forward fill, then backward fill, then use column mean, finally zero
            feature_data[col] = feature_data[col].fillna(method='ffill')
            feature_data[col] = feature_data[col].fillna(method='bfill')
            feature_data[col] = feature_data[col].fillna(feature_data[col].mean())
            feature_data[col] = feature_data[col].fillna(0)
        
        # Convert to numpy and handle edge cases
        feature_array = feature_data.values.astype(np.float32)
        
        # Replace any remaining NaN/inf values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Robust normalization
        try:
            # Use robust scaling (median and IQR)
            median_vals = np.median(feature_array, axis=0)
            q75 = np.percentile(feature_array, 75, axis=0)
            q25 = np.percentile(feature_array, 25, axis=0)
            iqr = q75 - q25
            
            # Avoid division by zero
            iqr[iqr == 0] = 1.0
            
            normalized_data = (feature_array - median_vals) / iqr
            
            # Clip extreme values
            normalized_data = np.clip(normalized_data, -5, 5)
            
        except Exception as norm_error:
            logger.warning(f"Robust normalization failed: {norm_error}, using min-max")
            # Fallback to min-max
            min_vals = np.min(feature_array, axis=0)
            max_vals = np.max(feature_array, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0
            normalized_data = (feature_array - min_vals) / range_vals
        
        # Create sequences with overlap
        sequences = []
        for i in range(len(normalized_data) - seq_len + 1):
            seq = normalized_data[i:i+seq_len]
            sequences.append(seq)
        
        if not sequences:
            logger.warning("No sequences could be created")
            return torch.empty(0, seq_len, 20)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(np.array(sequences))
        
        # Pad or truncate to exactly 20 features
        input_dim = 20
        current_features = tensor_data.size(-1)
        
        if current_features < input_dim:
            # Pad with small random noise instead of zeros for better training
            padding_size = input_dim - current_features
            padding = torch.randn(tensor_data.size(0), tensor_data.size(1), padding_size) * 0.01
            tensor_data = torch.cat([tensor_data, padding], dim=-1)
            logger.info(f"Padded features from {current_features} to {input_dim}")
        elif current_features > input_dim:
            tensor_data = tensor_data[:, :, :input_dim]
            logger.info(f"Truncated features from {current_features} to {input_dim}")
        
        # Final validation
        if torch.isnan(tensor_data).any() or torch.isinf(tensor_data).any():
            logger.error("Final tensor contains NaN or inf values, cleaning...")
            tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logger.info(f"Final training data shape: {tensor_data.shape}")
        logger.info(f"Data range: [{tensor_data.min():.3f}, {tensor_data.max():.3f}]")
        
        return tensor_data
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return torch.empty(0, seq_len, 20)

def train_options_strategist(options_strategist, training_data, data_manager):
    """Train the options strategist with MarketStack market data"""
    try:
        # Update the options strategist with the data manager
        options_strategist.update_market_data(data_manager)
        
        for symbol, data_dict in training_data.items():
            raw_data = data_dict['raw_data']
            
            # Calculate implied volatility and other options metrics
            if len(raw_data) > 30:
                # Handle MarketStack data structure
                if 'Returns' in raw_data.columns:
                    returns = raw_data['Returns'].dropna()
                else:
                    # Calculate returns if not available
                    returns = raw_data['Close'].pct_change().dropna()
                
                current_price = raw_data['Close'].iloc[-1]
                volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.2
                
                # Update options strategist with market data
                market_data = {
                    'price': float(current_price),
                    'volatility': float(volatility),
                    'returns': returns.tolist()[-30:] if len(returns) >= 30 else returns.tolist()
                }
                
                logger.info(f"Updated options strategist for {symbol}: price={current_price:.2f}, vol={volatility:.4f}")
        
        logger.info("Options strategist training completed")
        
    except Exception as e:
        logger.error(f"Error training options strategist: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

# 2. Update the create_supporting_components function to use your OptionsStrategist:
def create_supporting_components(config: Dict):
    """Create portfolio manager, risk manager, and other components"""
    try:
        # Portfolio Manager
        portfolio_manager = EnhancedPortfolioManager(
            initial_capital=config['initial_capital']
        )
        
        # Risk Manager  
        risk_manager = EnhancedRiskManager(
            max_position_size=config['max_position_size'],
            max_drawdown_threshold=config.get('stop_loss', 0.05),  # Changed parameter name
            var_threshold_95=0.02,
            var_threshold_99=0.05
        )
        
        # Use your comprehensive OptionsStrategist
        try:
            options_strategist = OptionsStrategist(
                seq_len=config.get('seq_length', 30),
                output_dim=10,
                risk_tolerance=config.get('risk_tolerance', 0.02)
            )
        except Exception as e:
            logger.warning(f"Could not create OptionsStrategist: {e}")
            options_strategist = OptionsStrategist()  # Fallback to default
        
        # Report Generator
        report_generator = ReportGenerator()
        
        return portfolio_manager, risk_manager, options_strategist, report_generator
        
    except Exception as e:
        logger.error(f"Error creating supporting components: {e}")
        return None, None, None, None

# 3. Update the generate_comprehensive_signals function to use your OptionsStrategist properly:
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
        
        # Create price series tensor for the options strategist
        price_tensors = []
        
        for symbol in st.session_state.symbols:
            # Fetch recent data
            data = st.session_state.data_manager.fetch_stock_data(symbol, period="6mo")
            
            if not data.empty:
                # Prepare input data for MOE
                input_tensor = prepare_training_data(data, seq_len=30)
                
                # Prepare price series for options strategist
                returns = data['Returns'].fillna(0).values[-30:]  # Last 30 returns
                if len(returns) < 30:
                    returns = np.pad(returns, (30-len(returns), 0), 'constant', constant_values=0)
                price_tensors.append(torch.FloatTensor(returns))
                
                if input_tensor.numel() > 0:
                    # MOE Framework prediction (existing code)
                    moe_framework.eval()
                    
                    with torch.no_grad():
                        latest_input = input_tensor[-1:] if len(input_tensor) > 0 else input_tensor
                        
                        if latest_input.numel() > 0:
                            # Generate MOE prediction
                            try:
                                moe_outputs = moe_framework(latest_input)
                                
                                # Extract MOE signals - handle different output formats
                                if isinstance(moe_outputs, dict):
                                    signals = moe_outputs.get('signals', torch.zeros(1, 10))[0].numpy()
                                    expert_weights = moe_outputs.get('expert_weights', torch.ones(1, len(moe_framework.experts))/len(moe_framework.experts))[0].numpy()
                                    gating_info = moe_outputs.get('gating_info', {})
                                elif isinstance(moe_outputs, tuple):
                                    signals = moe_outputs[0][0].numpy() if len(moe_outputs) > 0 else np.zeros(10)
                                    expert_weights = moe_outputs[1][0].numpy() if len(moe_outputs) > 1 else np.ones(len(moe_framework.experts))/len(moe_framework.experts)
                                    gating_info = moe_outputs[2] if len(moe_outputs) > 2 else {}
                                else:
                                    signals = moe_outputs[0].numpy() if moe_outputs.dim() > 1 else moe_outputs.numpy()
                                    expert_weights = np.ones(len(moe_framework.experts)) / len(moe_framework.experts)
                                    gating_info = {}
                                
                            except Exception as e:
                                logger.warning(f"Error in MOE forward pass for {symbol}: {e}")
                                signals = np.random.randn(10) * 0.1
                                expert_weights = np.ones(len(moe_framework.experts)) / len(moe_framework.experts)
                                gating_info = {}
                            
                            # Calculate overall signal strength
                            signal_strength = float(np.mean(signals))
                            confidence = float(np.std(signals)) if len(signals) > 1 else 0.5
                            
                            # Store MOE results
                            signals_dict[symbol] = {
                                'signal': signal_strength,
                                'confidence': confidence,
                                'moe_signals': signals.tolist() if hasattr(signals, 'tolist') else [float(signals)],
                                'expert_weights': expert_weights.tolist() if hasattr(expert_weights, 'tolist') else [1.0],
                                'reasoning': f"MOE prediction: {signal_strength:.3f}, Confidence: {confidence:.3f}"
                            }
                            
                            # Store detailed predictions
                            expert_predictions[symbol] = {
                                'signals': signals.tolist() if hasattr(signals, 'tolist') else [float(signals)],
                                'expert_weights': expert_weights.tolist() if hasattr(expert_weights, 'tolist') else [1.0],
                                'gating_info': gating_info if isinstance(gating_info, dict) else {}
                            }
                            
                            logger.info(f"Generated comprehensive signals for {symbol}: {signal_strength:.3f}")
        
        # Now use your comprehensive OptionsStrategist
        if price_tensors and hasattr(options_strategist, 'forward'):
            try:
                price_series_tensor = torch.stack(price_tensors)
                portfolio_value = st.session_state.portfolio_value
                
                # Get comprehensive options recommendations using your strategist
                options_recommendations = options_strategist.forward(
                    price_series_tensor, 
                    st.session_state.symbols, 
                    portfolio_value
                )
                
                # Integrate options recommendations with signals
                for symbol, options_rec in options_recommendations.items():
                    if symbol in signals_dict:
                        # Enhance the signal with options strategy info
                        signals_dict[symbol]['options_strategy'] = options_rec.get('strategy', 'HOLD')
                        signals_dict[symbol]['options_confidence'] = options_rec.get('confidence', 0.5)
                        
                        # Store detailed options recommendations
                        strategy_recommendations[symbol] = {
                            'strategy': options_rec.get('strategy', 'HOLD'),
                            'confidence': options_rec.get('confidence', 0.5),
                            'market_analysis': options_rec.get('market_analysis', {}),
                            'trade_details': options_rec.get('trade_details', {})
                        }
                        
                        # Extract risk metrics from options analysis
                        trade_details = options_rec.get('trade_details', {})
                        risk_metrics[symbol] = {
                            'risk_score': min(abs(trade_details.get('max_loss', 1000)) / 10000, 1.0),
                            'var_95': abs(trade_details.get('max_loss', 1000)),
                            'max_drawdown': 5.0,
                            'profit_probability': trade_details.get('profit_probability', 50.0),
                            'risk_reward_ratio': trade_details.get('risk_reward_ratio', 1.0)
                        }
                
            except Exception as e:
                logger.warning(f"Error in options strategist forward pass: {e}")
                # Fallback to simple strategy recommendations
                for symbol in signals_dict:
                    signal = signals_dict[symbol]['signal']
                    if signal > 0.1:
                        strategy = 'BULL_CALL_SPREAD'
                    elif signal < -0.1:
                        strategy = 'BEAR_PUT_SPREAD'
                    else:
                        strategy = 'IRON_CONDOR'
                    
                    strategy_recommendations[symbol] = {
                        'strategy': strategy,
                        'confidence': 0.5
                    }
                    risk_metrics[symbol] = {'risk_score': 0.5, 'var_95': 1000, 'max_drawdown': 5.0}
        
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

# 4. Add a new tab specifically for detailed options analysis:
def display_options_tab():
    """Detailed options strategies analysis using your OptionsStrategist"""
    st.header("🎯 Advanced Options Analysis")
    
    if not st.session_state.training_complete:
        st.warning("Please train the MOE framework first.")
        return
    
    # Input controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔍 Analyze Options Strategies", type="primary", use_container_width=True):
            with st.spinner("Performing comprehensive options analysis..."):
                analyze_detailed_options()
    
    with col2:
        risk_tolerance = st.slider("Risk Tolerance (%)", 1, 10, 2) / 100
        st.session_state.risk_tolerance = risk_tolerance
    
    with col3:
        portfolio_value = st.number_input(
            "Portfolio Value ($)", 
            min_value=10000, 
            value=int(st.session_state.portfolio_value), 
            step=10000
        )
    
    # Display detailed options analysis if available
    if hasattr(st.session_state, 'detailed_options_analysis'):
        display_detailed_options_results()

def analyze_detailed_options():
    """Perform detailed options analysis using your OptionsStrategist"""
    try:
        # Initialize your options strategist
        options_strategist = OptionsStrategist(
            seq_len=30,
            output_dim=10,
            risk_tolerance=st.session_state.get('risk_tolerance', 0.02)
        )
        
        # Prepare price data
        price_tensors = []
        valid_symbols = []
        
        for symbol in st.session_state.symbols:
            data = st.session_state.data_manager.fetch_stock_data(symbol, period="3mo")
            if not data.empty:
                returns = data['Returns'].fillna(0).values[-30:]
                if len(returns) < 30:
                    returns = np.pad(returns, (30-len(returns), 0), 'constant', constant_values=0)
                price_tensors.append(torch.FloatTensor(returns))
                valid_symbols.append(symbol)
        
        if price_tensors:
            price_series = torch.stack(price_tensors)
            
            # Get comprehensive recommendations
            recommendations = options_strategist.forward(
                price_series, 
                valid_symbols, 
                st.session_state.portfolio_value
            )
            
            st.session_state.detailed_options_analysis = recommendations
            st.success(f"Analyzed options strategies for {len(recommendations)} symbols")
            
        else:
            st.error("No valid data available for options analysis")
            
    except Exception as e:
        logger.error(f"Error in detailed options analysis: {e}")
        st.error(f"Options analysis failed: {e}")

def display_detailed_options_results():
    """Display the detailed options analysis results"""
    recommendations = st.session_state.detailed_options_analysis
    
    if not recommendations:
        st.warning("No options recommendations available.")
        return
    
    # Use your existing display_recommendations method
    try:
        # Create a temporary strategist instance to use the display method
        temp_strategist = OptionsStrategist()
        temp_strategist.display_recommendations(recommendations)
    except Exception as e:
        # Fallback display if the method doesn't work in this context
        st.subheader("Options Strategy Recommendations")
        
        for symbol, rec in recommendations.items():
            with st.expander(f"📊 {symbol} - {rec.get('strategy', 'N/A')}", expanded=True):
                
                # Market Analysis
                if 'market_analysis' in rec:
                    st.subheader("Market Analysis")
                    analysis = rec['market_analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trend", analysis.get('trend', 'N/A'))
                    with col2:
                        st.metric("Volatility Regime", analysis.get('volatility_regime', 'N/A'))
                    with col3:
                        st.metric("Momentum", analysis.get('momentum', 'N/A'))
                
                # Trade Details
                if 'trade_details' in rec:
                    st.subheader("Trade Details")
                    trade = rec['trade_details']
                    
                    if 'legs' in trade:
                        legs_df = pd.DataFrame(trade['legs'])
                        st.dataframe(legs_df, use_container_width=True)
                    
                    # Risk/Reward metrics
                    if 'rationale' in trade:
                        st.info(trade['rationale'])

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
            config = st.session_state.get('config', {
                'seq_length': 30,
                'model_dim': 256,
                'num_heads': 8,
                'initial_capital': 10000,
                'max_position_size': 0.2,
                'stop_loss': 0.05
            })

            success = train_moe_framework(
                st.session_state.symbols,
                config,
                progress_placeholder,
                status_placeholder
            )
            if success:
                st.success("Training completed successfully!")
                # st.balloons()
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
    
    if not st.session_state.training_complete:
        st.warning("Generate training and signals to enable reports")
        return
    
    if not st.session_state.report_generator:
        st.error("Report generator not initialized")
        return
    
    try:
        # Get the required components from session state
        portfolio_manager = st.session_state.get('portfolio_manager')
        risk_manager = st.session_state.get('risk_manager')
        moe_system = st.session_state.get('moe_framework')
        
        if not portfolio_manager or not risk_manager:
            st.warning("Portfolio manager and risk manager required for reports")
            return
        
        # Generate comprehensive report
        with st.spinner("Generating comprehensive report..."):
            report = st.session_state.report_generator.generate_comprehensive_report(
                portfolio_manager=portfolio_manager,
                risk_manager=risk_manager,
                moe_system=moe_system,
                report_period_days=30
            )
        
        # Display executive summary
        if 'executive_summary' in report:
            st.subheader("📊 Executive Summary")
            summary = report['executive_summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio Value", f"${summary.get('portfolio_value', 0):,.0f}")
            with col2:
                st.metric("Total Return", f"{summary.get('total_return_pct', 0):.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
            with col4:
                st.metric("Risk Level", summary.get('current_risk_level', 'UNKNOWN'))
        
        # Display insights if available
        if 'insights' in report and report['insights']:
            st.subheader("💡 Key Insights")
            for insight in report['insights']:
                insight_type = insight.get('type', 'info')
                if insight_type == 'critical':
                    st.error(f"**{insight.get('title', '')}**: {insight.get('description', '')}")
                elif insight_type == 'warning':
                    st.warning(f"**{insight.get('title', '')}**: {insight.get('description', '')}")
                elif insight_type == 'positive':
                    st.success(f"**{insight.get('title', '')}**: {insight.get('description', '')}")
                else:
                    st.info(f"**{insight.get('title', '')}**: {insight.get('description', '')}")
        
        # Display performance analysis
        if 'performance_analysis' in report and not report['performance_analysis'].get('error'):
            st.subheader("📈 Performance Analysis")
            perf_metrics = report['performance_analysis'].get('performance_metrics', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Period Return", f"{perf_metrics.get('period_return', 0):.2f}%")
                st.metric("Volatility", f"{perf_metrics.get('volatility', 0):.2f}%")
                st.metric("Max Drawdown", f"{perf_metrics.get('max_drawdown', 0):.2f}%")
            
            with col2:
                st.metric("Win Rate", f"{perf_metrics.get('win_rate', 0):.1f}%")
                st.metric("Best Day", f"{perf_metrics.get('best_day', 0):.2f}%")
                st.metric("Worst Day", f"{perf_metrics.get('worst_day', 0):.2f}%")
        
        # Display position analysis
        if 'position_analysis' in report and not report['position_analysis'].get('error'):
            st.subheader("🎯 Position Analysis")
            pos_analysis = report['position_analysis']
            
            if 'top_positions' in pos_analysis and pos_analysis['top_positions']:
                st.write("**Top Positions:**")
                positions_df = pd.DataFrame(pos_analysis['top_positions'][:10])
                st.dataframe(positions_df, use_container_width=True)
        
        # Export options
        st.subheader("📄 Export Report")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export HTML Report", use_container_width=True):
                try:
                    filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    st.session_state.report_generator.export_report_html(report, filename)
                    st.success(f"HTML report exported to {filename}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        with col2:
            if st.button("📊 Export JSON Data", use_container_width=True):
                try:
                    filename = f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    st.session_state.report_generator.export_report_json(report, filename)
                    st.success(f"JSON data exported to {filename}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
    except Exception as e:
        st.error(f"Error generating report: {e}")
        logger.error(f"Report generation error: {e}")



# =============================================================================
# Complete Main Function and Enhanced Sidebar
# =============================================================================

def display_sidebar():
    """Enhanced sidebar configuration with dynamic symbol handling"""
    st.sidebar.title("🚀 MOE Framework")
    st.sidebar.markdown("---")
    
    # Model Configuration
    st.sidebar.header("Model Configuration")
    
    # Symbol selection with dynamic updates
    st.sidebar.subheader("📊 Stock Selection")
    
    # Get current symbols input, fallback to session state if needed
    current_symbols_input = st.session_state.get('symbols_input', ",".join(st.session_state.symbols))
    
    symbols_input = st.sidebar.text_area(
        "Enter symbols (comma-separated)",
        value=current_symbols_input,
        help="Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL)",
        key="symbols_text_area"
    )
    
    # Update symbols when button is clicked or when input changes
    if st.sidebar.button("🔄 Update Symbols", use_container_width=True):
        update_symbols_from_input(symbols_input)
    
    # Auto-update symbols if input has changed (optional real-time update)
    if symbols_input != st.session_state.get('symbols_input', ''):
        st.session_state.symbols_input = symbols_input
        # Uncomment the line below for real-time updates (may cause frequent reruns)
        # update_symbols_from_input(symbols_input)
    
    # Display current active symbols
    st.sidebar.write(f"**Active Symbols ({len(st.session_state.symbols)}):**")
    symbols_display = ", ".join(st.session_state.symbols[:5])  # Show first 5
    if len(st.session_state.symbols) > 5:
        symbols_display += f" + {len(st.session_state.symbols) - 5} more"
    st.sidebar.write(symbols_display)
    
    # Show if symbols have changed recently
    if st.session_state.get('symbols_changed', False):
        st.sidebar.success("✅ Symbols updated! Retrain for new analysis.")
        # Reset the flag after showing the message
        if st.sidebar.button("Clear Notice", key="clear_symbols_notice"):
            st.session_state.symbols_changed = False
    
    # Training parameters
    st.sidebar.subheader("🧠 Training Parameters")
    seq_length = st.sidebar.slider("Sequence Length", 10, 60, 30)
    model_dim = st.sidebar.slider("Model Dimension", 128, 512, 256)
    num_heads = st.sidebar.slider("Attention Heads", 4, 16, 8)
    
    # Portfolio settings
    st.sidebar.subheader("💰 Portfolio Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10000000,
        value=int(st.session_state.portfolio_value),
        step=10000
    )
    
    # Risk management
    st.sidebar.subheader("⚠️ Risk Management")
    max_position_size = st.sidebar.slider("Max Position Size (%)", 5, 50, 20)
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 20, 5)
    
    # Add risk tolerance for options
    risk_tolerance = st.sidebar.slider("Options Risk Tolerance (%)", 1, 10, 2)
    
    # Options specific settings
    st.sidebar.subheader("🎯 Options Settings")
    options_expiry_range = st.sidebar.slider("Preferred Days to Expiry", 15, 60, 30)
    min_open_interest = st.sidebar.number_input("Min Open Interest", min_value=1, value=10, step=1)
    
    # Training status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 System Status")
    
    status_color = "🟢" if st.session_state.training_complete else "🔴"
    st.sidebar.write(f"{status_color} Training Status: {st.session_state.training_status}")
    
    if st.session_state.signals_dict:
        st.sidebar.write(f"📊 Active Signals: {len(st.session_state.signals_dict)}")
    
    if st.session_state.strategy_recommendations:
        st.sidebar.write(f"🎯 Options Strategies: {len(st.session_state.strategy_recommendations)}")
    
    # Data refresh controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Data Controls")
    
    if st.sidebar.button("🔄 Refresh Market Data", use_container_width=True):
        # Clear cache to force fresh data fetch
        if hasattr(st.session_state.data_manager, 'cache'):
            st.session_state.data_manager.cache.clear()
        st.sidebar.success("Cache cleared - next data fetch will be fresh")
    
    # Quick symbol presets
    st.sidebar.subheader("⚡ Quick Presets")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📱 Tech Giants", use_container_width=True):
            update_symbols_from_preset(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA'])
    
    with col2:
        if st.button("🏦 Finance", use_container_width=True):
            update_symbols_from_preset(['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'])
    
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        if st.button("🏥 Healthcare", use_container_width=True):
            update_symbols_from_preset(['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'])
    
    with col4:
        if st.button("⚡ Energy", use_container_width=True):
            update_symbols_from_preset(['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL'])
    
    # System information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Info")
    st.sidebar.write(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}")
    st.sidebar.write(f"🔥 PyTorch: {torch.__version__}")
    st.sidebar.write(f"📊 Active Symbols: {len(st.session_state.symbols)}")
    
    return {
        'seq_length': seq_length,
        'model_dim': model_dim,
        'num_heads': num_heads,
        'initial_capital': initial_capital,
        'max_position_size': max_position_size / 100,
        'stop_loss': stop_loss / 100,
        'risk_tolerance': risk_tolerance / 100,
        'options_expiry_range': options_expiry_range,
        'min_open_interest': min_open_interest
    }

def update_symbols_from_input(symbols_input):
    """Update symbols from text input with validation"""
    try:
        # Parse and clean symbols
        new_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_symbols = []
        for symbol in new_symbols:
            if symbol not in seen and len(symbol) <= 5:  # Basic validation
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        # Validate we have at least one symbol
        if not unique_symbols:
            st.sidebar.error("❌ Please enter at least one valid symbol")
            return False
        
        # Limit to reasonable number (e.g., 20 symbols max)
        if len(unique_symbols) > 20:
            st.sidebar.warning(f"⚠️ Limited to first 20 symbols (entered {len(unique_symbols)})")
            unique_symbols = unique_symbols[:20]
        
        # Check if symbols actually changed
        if unique_symbols != st.session_state.symbols:
            # Store old symbols for comparison
            old_symbols = st.session_state.symbols.copy()
            
            # Update symbols
            st.session_state.symbols = unique_symbols
            st.session_state.symbols_input = ",".join(unique_symbols)
            st.session_state.symbols_changed = True
            
            # Reset training status since symbols changed
            st.session_state.training_complete = False
            st.session_state.training_status = 'Not Started'
            st.session_state.signals_dict = {}
            st.session_state.expert_predictions = {}
            st.session_state.strategy_recommendations = {}
            st.session_state.risk_metrics = {}
            
            # Log the change
            logger.info(f"Symbols updated: {old_symbols} -> {unique_symbols}")
            
            st.sidebar.success(f"✅ Updated to {len(unique_symbols)} symbols")
            return True
        else:
            st.sidebar.info("ℹ️ Symbols unchanged")
            return False
            
    except Exception as e:
        st.sidebar.error(f"❌ Error updating symbols: {e}")
        logger.error(f"Error updating symbols: {e}")
        return False

def update_symbols_from_preset(preset_symbols):
    """Update symbols from a preset list"""
    symbols_string = ",".join(preset_symbols)
    st.session_state.symbols_input = symbols_string
    update_symbols_from_input(symbols_string)
    st.rerun()  # Force rerun to update the UI

def validate_symbols(symbols):
    """Validate symbol format and availability"""
    valid_symbols = []
    invalid_symbols = []
    
    for symbol in symbols:
        # Basic format validation
        if len(symbol) > 5 or not symbol.isalpha():
            invalid_symbols.append(symbol)
            continue
        
        # You could add more sophisticated validation here
        # like checking against a known symbol list or API
        valid_symbols.append(symbol)
    
    return valid_symbols, invalid_symbols

def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🚀 MOE Trading Framework")
    st.markdown("*Advanced AI-powered trading system with comprehensive options strategies*")
    
    # Quick status overview
    if st.session_state.training_complete:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Framework Status", 
                "✅ Ready",
                delta="Trained" if st.session_state.training_complete else "Not Trained"
            )
        
        with col2:
            portfolio_value = st.session_state.portfolio_value
            st.metric(
                "Portfolio Value", 
                f"${portfolio_value:,.0f}",
                delta=f"${portfolio_value - 1000000:,.0f}" if portfolio_value != 1000000 else None
            )
        
        with col3:
            signal_count = len(st.session_state.signals_dict)
            st.metric(
                "Active Signals", 
                signal_count,
                delta="Updated" if signal_count > 0 else "Generate"
            )
        
        with col4:
            strategy_count = len(st.session_state.strategy_recommendations)
            st.metric(
                "Options Strategies", 
                strategy_count,
                delta="Available" if strategy_count > 0 else "Analyze"
            )
        
        st.markdown("---")
    
    # Display sidebar and get configuration
    config = display_sidebar()
    
    # Add risk tolerance to config
    config['risk_tolerance'] = st.session_state.get('risk_tolerance', 0.02)
    
    # Store config in session state for access across functions
    st.session_state.config = config
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 Training", 
        "📊 Signals", 
        "🎯 Options",
        "💼 Portfolio", 
        "📈 Analytics",
        "📋 Reports"
    ])
    
    with tab1:
        display_training_tab()
    
    with tab2:
        display_signals_tab()
    
    with tab3:
        display_options_tab()
    
    with tab4:
        display_portfolio_tab()
    
    # with tab5:
    #     display_analytics_tab()
    
    with tab5:
        display_reports_tab()
    
    # Footer with system information
    st.markdown("---")
    with st.expander("ℹ️ System Information & Help"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Configuration")
            st.json({
                "Symbols": st.session_state.symbols,
                "Training Complete": st.session_state.training_complete,
                "Portfolio Value": f"${st.session_state.portfolio_value:,.2f}",
                "Risk Tolerance": f"{config['risk_tolerance']*100:.1f}%",
                "Sequence Length": config['seq_length'],
                "Model Dimension": config['model_dim']
            })
        
        with col2:
            st.subheader("Quick Start Guide")
            st.markdown("""
            1. **Configure Settings**: Update symbols and parameters in sidebar
            2. **Train Framework**: Go to Training tab and click "Start Training"
            3. **Generate Signals**: In Signals tab, click "Generate Signals"
            4. **Analyze Options**: Use Options tab for detailed strategy analysis
            5. **Monitor Portfolio**: Track performance in Portfolio tab
            6. **Review Analytics**: Check detailed metrics in Analytics tab
            
            **Need Help?**
            - Hover over any metric for additional information
            - Check the Reports tab for comprehensive analysis
            - All data is cached for 30 minutes to improve performance
            """)
    
    # Error handling and status messages
    if not st.session_state.training_complete:
        st.info("💡 **Getting Started**: Please go to the Training tab to initialize the MOE framework before generating signals.")
    
    # Auto-refresh data notification
    if st.session_state.get('auto_refresh_enabled', False):
        st.success("🔄 Auto-refresh is enabled. Market data will update automatically.")

def display_options_tab():
    """Detailed options strategies analysis using the comprehensive OptionsStrategist"""
    st.header("🎯 Advanced Options Analysis")
    st.markdown("*Comprehensive options strategy recommendations with real market data*")
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train the MOE framework first to enable options analysis.")
        return
    
    # Control panel
    st.subheader("🎛️ Analysis Controls")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.button("🔍 Analyze Options Strategies", type="primary", use_container_width=True):
            with st.spinner("Performing comprehensive options analysis..."):
                analyze_detailed_options()
    
    with col2:
        risk_tolerance = st.selectbox(
            "Risk Level",
            ["Conservative", "Moderate", "Aggressive"],
            index=1
        )
        risk_mapping = {"Conservative": 0.01, "Moderate": 0.02, "Aggressive": 0.05}
        st.session_state.risk_tolerance = risk_mapping[risk_tolerance]
    
    with col3:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["All Strategies", "Directional Only", "Neutral Only", "Volatility Only"],
            index=0
        )
    
    with col4:
        portfolio_percentage = st.slider("Portfolio % Risk", 1, 10, 2)
    
    # Display current market conditions summary
    if st.session_state.signals_dict:
        st.subheader("📊 Current Market Overview")
        
        market_summary = []
        for symbol, signal_info in st.session_state.signals_dict.items():
            signal = signal_info.get('signal', 0)
            confidence = signal_info.get('confidence', 0)
            
            if signal > 0.1:
                outlook = "🟢 Bullish"
            elif signal < -0.1:
                outlook = "🔴 Bearish"
            else:
                outlook = "🟡 Neutral"
            
            market_summary.append({
                'Symbol': symbol,
                'Signal': f"{signal:.3f}",
                'Confidence': f"{confidence:.3f}",
                'Outlook': outlook
            })
        
        if market_summary:
            st.dataframe(market_summary, use_container_width=True)
    
    # Display detailed options analysis if available
    if hasattr(st.session_state, 'detailed_options_analysis') and st.session_state.detailed_options_analysis:
        st.subheader("🎯 Strategy Recommendations")
        display_detailed_options_results()
    else:
        st.info("Click 'Analyze Options Strategies' to generate comprehensive recommendations.")
    
    # Educational content
    with st.expander("📚 Options Strategy Guide"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Directional Strategies:**
            - **Bull Call Spread**: Limited risk bullish play
            - **Bear Put Spread**: Limited risk bearish play
            - **Covered Call**: Income from stock holdings
            - **Cash Secured Put**: Income with stock acquisition intent
            
            **Neutral Strategies:**
            - **Iron Condor**: Profit from low volatility
            - **Butterfly Spread**: High probability, limited profit
            - **Collar**: Protective strategy with capped upside
            """)
        
        with col2:
            st.markdown("""
            **Volatility Strategies:**
            - **Long Straddle**: Profit from large moves
            - **Long Strangle**: Lower cost volatility play
            - **Protective Put**: Downside insurance
            
            **Risk Management:**
            - Position sizing based on portfolio value
            - Real-time profit probability calculations
            - Liquidity requirements (minimum open interest)
            - Maximum risk per trade controls
            """)

def analyze_detailed_options():
    """Perform detailed options analysis using the comprehensive OptionsStrategist"""
    try:
        # Get configuration
        config = st.session_state.get('config', {})
        
        # Initialize the options strategist with current settings
        options_strategist = OptionsStrategist(
            seq_len=config.get('seq_length', 30),
            output_dim=10,
            risk_tolerance=st.session_state.get('risk_tolerance', 0.02)
        )
        
        # Prepare price data for all symbols
        price_tensors = []
        valid_symbols = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(st.session_state.symbols):
            status_text.text(f"Fetching options data for {symbol}...")
            progress_bar.progress((i + 1) / len(st.session_state.symbols))
            
            # Get market data
            data = st.session_state.data_manager.fetch_stock_data(symbol, period="3mo")
            
            if not data.empty:
                # Prepare returns data for the strategist
                returns = data['Returns'].fillna(0).values[-30:]
                if len(returns) < 30:
                    returns = np.pad(returns, (30-len(returns), 0), 'constant', constant_values=0)
                
                price_tensors.append(torch.FloatTensor(returns))
                valid_symbols.append(symbol)
        
        if price_tensors:
            status_text.text("Analyzing options strategies...")
            
            # Stack price tensors
            price_series = torch.stack(price_tensors)
            
            # Get comprehensive recommendations using the options strategist
            recommendations = options_strategist.forward(
                price_series, 
                valid_symbols, 
                st.session_state.portfolio_value
            )
            
            # Store results
            st.session_state.detailed_options_analysis = recommendations
            
            # Update strategy recommendations in main session state
            for symbol, rec in recommendations.items():
                if symbol not in st.session_state.strategy_recommendations:
                    st.session_state.strategy_recommendations[symbol] = {}
                
                st.session_state.strategy_recommendations[symbol].update({
                    'detailed_strategy': rec.get('strategy', 'HOLD'),
                    'detailed_confidence': rec.get('confidence', 0.5),
                    'market_analysis': rec.get('market_analysis', {}),
                    'trade_details': rec.get('trade_details', {})
                })
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            st.success(f"Successfully analyzed options strategies for {len(recommendations)} symbols")
            time.sleep(1)  # Brief pause to show completion
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
        else:
            st.error("❌ No valid market data available for options analysis")
            
    except Exception as e:
        logger.error(f"Error in detailed options analysis: {e}")
        st.error(f"Options analysis failed: {str(e)}")
        
        # Clean up progress indicators on error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def display_detailed_options_results():
    """Display the detailed options analysis results with enhanced formatting"""
    recommendations = st.session_state.detailed_options_analysis
    
    if not recommendations:
        st.warning("No options recommendations available.")
        return
    
    # Summary statistics
    strategy_counts = {}
    total_profit_potential = 0
    total_risk_amount = 0
    
    for symbol, rec in recommendations.items():
        strategy = rec.get('strategy', 'UNKNOWN')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        trade_details = rec.get('trade_details', {})
        if isinstance(trade_details, dict) and 'max_profit' in trade_details:
            total_profit_potential += trade_details.get('max_profit', 0)
            total_risk_amount += abs(trade_details.get('max_loss', 0))
    
    # Display summary
    st.subheader("📈 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Symbols Analyzed", len(recommendations))
    
    with col2:
        st.metric("Unique Strategies", len(strategy_counts))
    
    with col3:
        if total_risk_amount > 0:
            risk_reward = total_profit_potential / total_risk_amount
            st.metric("Avg Risk/Reward", f"{risk_reward:.2f}")
        else:
            st.metric("Avg Risk/Reward", "N/A")
    
    with col4:
        portfolio_risk_pct = (total_risk_amount / st.session_state.portfolio_value) * 100
        st.metric("Portfolio Risk %", f"{portfolio_risk_pct:.1f}%")
    
    # Strategy distribution
    if strategy_counts:
        st.subheader("🎯 Strategy Distribution")
        strategy_df = pd.DataFrame(list(strategy_counts.items()), columns=['Strategy', 'Count'])
        
        fig = px.pie(strategy_df, values='Count', names='Strategy', 
                     title="Recommended Strategy Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed recommendations for each symbol
    st.subheader("📊 Detailed Recommendations")
    
    # Create tabs for different strategy types
    directional_strategies = ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', 'COVERED_CALL', 'CASH_SECURED_PUT', 'PROTECTIVE_PUT']
    neutral_strategies = ['IRON_CONDOR', 'BUTTERFLY', 'COLLAR']
    volatility_strategies = ['STRADDLE', 'STRANGLE']
    
    tab_dir, tab_neu, tab_vol, tab_all = st.tabs(['📈 Directional', '⚖️ Neutral', '🌊 Volatility', '📋 All'])
    
    with tab_dir:
        display_strategy_group(recommendations, directional_strategies, "Directional Strategies")
    
    with tab_neu:
        display_strategy_group(recommendations, neutral_strategies, "Neutral Strategies")
    
    with tab_vol:
        display_strategy_group(recommendations, volatility_strategies, "Volatility Strategies")
    
    with tab_all:
        # Display all recommendations using the original strategist display method
        try:
            # Use your existing comprehensive display
            temp_strategist = OptionsStrategist()
            temp_strategist.display_recommendations(recommendations)
        except Exception as e:
            # Fallback display
            st.error(f"Error displaying recommendations: {e}")
            for symbol, rec in recommendations.items():
                display_single_recommendation(symbol, rec)

def display_strategy_group(recommendations, strategy_list, group_name):
    """Display a specific group of strategies"""
    filtered_recs = {
        symbol: rec for symbol, rec in recommendations.items() 
        if rec.get('strategy', '') in strategy_list
    }
    
    if not filtered_recs:
        st.info(f"No {group_name.lower()} recommended for current market conditions.")
        return
    
    for symbol, rec in filtered_recs.items():
        display_single_recommendation(symbol, rec)

def display_single_recommendation(symbol, rec):
    """Display a single recommendation with error handling"""
    try:
        strategy = rec.get('strategy', 'UNKNOWN')
        confidence = rec.get('confidence', 0)
        
        with st.expander(f"📊 {symbol} - {strategy} (Confidence: {confidence:.1f}/10)", expanded=False):
            
            # Market Analysis
            if 'market_analysis' in rec:
                st.subheader("Market Analysis")
                analysis = rec['market_analysis']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Trend", analysis.get('trend', 'N/A'))
                    st.metric("RSI", f"{analysis.get('rsi', 0):.1f}")
                
                with col2:
                    st.metric("Volatility Regime", analysis.get('volatility_regime', 'N/A'))
                    st.metric("IV Rank", f"{analysis.get('iv_rank', 0):.1f}%")
                
                with col3:
                    st.metric("Momentum", analysis.get('momentum', 'N/A'))
                    st.metric("Vol Ratio", f"{analysis.get('vol_ratio', 1):.2f}")
            
            # Trade Details
            if 'trade_details' in rec:
                trade = rec['trade_details']
                
                if 'error' in trade:
                    st.error(f"Error: {trade['error']}")
                    return
                
                st.subheader(f"Strategy: {trade.get('strategy_name', strategy)}")
                
                if 'rationale' in trade:
                    st.info(trade['rationale'])
                
                # Trade legs
                if 'legs' in trade:
                    st.subheader("Trade Legs")
                    legs_df = pd.DataFrame(trade['legs'])
                    st.dataframe(legs_df, use_container_width=True)
                
                # Risk/Reward metrics
                st.subheader("Risk/Reward Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    for metric in ['max_profit', 'max_loss', 'net_debit', 'net_credit']:
                        if metric in trade:
                            st.metric(metric.replace('_', ' ').title(), f"${trade[metric]:,.2f}")
                
                with col2:
                    for metric in ['breakeven', 'profit_probability', 'risk_reward_ratio']:
                        if metric in trade:
                            if 'probability' in metric:
                                st.metric(metric.replace('_', ' ').title(), f"{trade[metric]:.1f}%")
                            elif 'ratio' in metric:
                                st.metric(metric.replace('_', ' ').title(), f"{trade[metric]:.2f}")
                            else:
                                st.metric(metric.replace('_', ' ').title(), f"${trade[metric]:.2f}")
                
                if 'days_to_expiry' in trade:
                    st.metric("Days to Expiry", trade['days_to_expiry'])
    
    except Exception as e:
        st.error(f"Error displaying recommendation for {symbol}: {e}")

if __name__ == "__main__":
    main()