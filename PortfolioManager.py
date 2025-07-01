import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import optimize
import cvxpy as cp
from collections import defaultdict, deque
import json

# =============================================================================
# Logger Setup
# =============================================================================
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Enhanced Data Structures for Portfolio Management
# =============================================================================

@dataclass
class Position:
    """Enhanced position tracking with comprehensive metadata"""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    last_updated: datetime
    entry_date: datetime
    sector: Optional[str] = None
    strategy_source: Optional[str] = None  # Which expert/strategy recommended this
    confidence_score: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.shares
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'entry_date': self.entry_date.isoformat(),
            'sector': self.sector,
            'strategy_source': self.strategy_source,
            'confidence_score': self.confidence_score
        }

@dataclass
class Trade:
    """Enhanced trade tracking with execution details"""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    commission: float = 0.0
    strategy_source: Optional[str] = None
    confidence_score: float = 0.0
    market_regime: Optional[str] = None
    
    @property
    def notional_value(self) -> float:
        return abs(self.shares * self.price)
    
    @property
    def total_cost(self) -> float:
        return self.notional_value + self.commission
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'shares': self.shares,
            'price': self.price,
            'notional_value': self.notional_value,
            'commission': self.commission,
            'total_cost': self.total_cost,
            'strategy_source': self.strategy_source,
            'confidence_score': self.confidence_score,
            'market_regime': self.market_regime
        }

@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    num_positions: int
    daily_pnl: float
    cumulative_pnl: float
    exposure_by_sector: Dict[str, float]
    concentration_risk: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'cash': self.cash,
            'positions_value': self.positions_value,
            'num_positions': self.num_positions,
            'daily_pnl': self.daily_pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'exposure_by_sector': self.exposure_by_sector,
            'concentration_risk': self.concentration_risk
        }

# =============================================================================
# Advanced Portfolio Optimization Engine
# =============================================================================

class PortfolioOptimizer:
    """Advanced portfolio optimization with multiple objective functions"""
    
    def __init__(self,
                 risk_aversion: float = 1.0,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.1,
                 max_turnover: float = 0.5):
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.max_turnover = max_turnover
    
    def optimize_weights(self,
                        expected_returns: np.ndarray,
                        covariance_matrix: np.ndarray,
                        current_weights: np.ndarray,
                        confidence_scores: Optional[np.ndarray] = None,
                        sector_constraints: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Multi-objective portfolio optimization with constraints
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            current_weights: Current portfolio weights
            confidence_scores: Confidence in each signal (0-1)
            sector_constraints: Maximum weight per sector
            
        Returns:
            Dictionary with optimal weights and optimization metadata
        """
        n_assets = len(expected_returns)
        
        # Decision variables
        w = cp.Variable(n_assets)  # Target weights
        turnover = cp.Variable(n_assets)  # Turnover per asset
        
        # Apply confidence scores to expected returns
        if confidence_scores is not None:
            adjusted_returns = expected_returns * confidence_scores
        else:
            adjusted_returns = expected_returns
        
        # Objective function: Maximize utility (return - risk - transaction costs)
        portfolio_return = w.T @ adjusted_returns
        portfolio_risk = cp.quad_form(w, covariance_matrix)
        transaction_costs = cp.sum(turnover) * self.transaction_cost
        
        objective = cp.Maximize(
            portfolio_return - 
            self.risk_aversion * portfolio_risk - 
            transaction_costs
        )
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= 0,  # Long-only (modify for long-short)
            w <= self.max_position_size,  # Position size limits
            turnover >= w - current_weights,  # Turnover definition
            turnover >= current_weights - w,
            cp.sum(turnover) <= self.max_turnover  # Turnover limit
        ]
        
        # Sector constraints if provided
        if sector_constraints:
            # This would require sector mapping - simplified for now
            pass
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = w.value
                optimization_info = {
                    'status': 'optimal',
                    'objective_value': problem.value,
                    'expected_return': float(optimal_weights.T @ adjusted_returns),
                    'expected_risk': float(np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)),
                    'turnover': float(np.sum(turnover.value)),
                    'max_weight': float(np.max(optimal_weights)),
                    'concentration': float(np.sum(optimal_weights**2))  # Herfindahl index
                }
                
                return {
                    'weights': optimal_weights,
                    'info': optimization_info
                }
            else:
                logger.warning(f"Optimization failed with status: {problem.status}")
                return {
                    'weights': current_weights,
                    'info': {'status': 'failed', 'message': problem.status}
                }
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {
                'weights': current_weights,
                'info': {'status': 'error', 'message': str(e)}
            }

# =============================================================================
# Enhanced Portfolio Manager
# =============================================================================

class EnhancedPortfolioManager:
    """
    Advanced Portfolio Manager with MoE integration, risk management,
    and comprehensive tracking capabilities
    """
    
    def __init__(self,
                 initial_capital: float = 1e6,
                 commission_rate: float = 0.001,
                 min_trade_size: float = 100.0,
                 max_position_size: float = 0.1,
                 rebalance_threshold: float = 0.05,
                 risk_budget: float = 0.02,
                 use_optimization: bool = True):
        
        # Core portfolio state
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        
        # Trading parameters
        self.commission_rate = commission_rate
        self.min_trade_size = min_trade_size
        self.max_position_size = max_position_size
        self.rebalance_threshold = rebalance_threshold
        self.risk_budget = risk_budget
        self.use_optimization = use_optimization
        
        # Portfolio optimizer
        self.optimizer = PortfolioOptimizer(
            max_position_size=max_position_size,
            transaction_cost=commission_rate
        )
        
        # Historical tracking
        self.trade_history: List[Trade] = []
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.performance_metrics: Dict = {}
        
        # Risk and exposure tracking
        self.sector_exposures: Dict[str, float] = defaultdict(float)
        self.strategy_attributions: Dict[str, float] = defaultdict(float)
        self.correlation_matrix: Optional[np.ndarray] = None
        
        # Market data cache
        self.price_cache: Dict[str, float] = {}
        self.returns_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        # Performance tracking
        self.benchmark_returns: deque = deque(maxlen=252)
        self.portfolio_returns: deque = deque(maxlen=252)
        
        logger.info(f"Enhanced Portfolio Manager initialized with ${initial_capital:,.2f}")
    
    def update_prices(self, prices: Dict[str, float], timestamp: Optional[datetime] = None):
        """Update current market prices and cache for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.price_cache.update(prices)
        
        # Update position prices and calculate returns
        for symbol, position in self.positions.items():
            if symbol in prices:
                old_price = position.current_price
                new_price = prices[symbol]
                position.current_price = new_price
                position.last_updated = timestamp
                
                # Calculate return and cache
                if old_price > 0:
                    daily_return = (new_price - old_price) / old_price
                    self.returns_cache[symbol].append(daily_return)
    
    def process_moe_signals(self,
                           moe_output: torch.Tensor,
                           aux_info: Dict,
                           symbol_mapping: Dict[int, str],
                           current_prices: Dict[str, float],
                           market_regime: Optional[str] = None) -> Dict[str, Any]:
        """
        Process MoE signals and generate optimal portfolio weights
        
        Args:
            moe_output: [batch, output_dim] - MoE predictions
            aux_info: Auxiliary information from MoE forward pass
            symbol_mapping: Mapping from output indices to symbols
            current_prices: Current market prices
            market_regime: Current market regime identifier
            
        Returns:
            Dictionary with rebalancing decisions and metadata
        """
        timestamp = datetime.now()
        
        # Extract signals and confidence scores
        signals = self._extract_signals_from_moe(moe_output, aux_info, symbol_mapping)
        
        # Get expert confidence and routing information
        expert_weights = aux_info.get('topk_weights', torch.ones(1, 2))
        routing_entropy = aux_info.get('routing_entropy', 0.0)
        
        # Calculate overall confidence based on expert consensus
        confidence_scores = self._calculate_signal_confidence(
            signals, expert_weights, routing_entropy
        )
        
        # Get current portfolio state
        current_weights = self._get_current_weights(current_prices)
        
        # Prepare optimization inputs
        symbols = list(signals.keys())
        expected_returns = np.array([signals[symbol] for symbol in symbols])
        current_weight_vector = np.array([current_weights.get(symbol, 0.0) for symbol in symbols])
        confidence_vector = np.array([confidence_scores.get(symbol, 0.5) for symbol in symbols])
        
        # Estimate covariance matrix from historical returns
        covariance_matrix = self._estimate_covariance_matrix(symbols)
        
        # Optimize portfolio if enabled
        if self.use_optimization and len(symbols) > 1:
            optimization_result = self.optimizer.optimize_weights(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                current_weights=current_weight_vector,
                confidence_scores=confidence_vector
            )
            target_weights = dict(zip(symbols, optimization_result['weights']))
            optimization_info = optimization_result['info']
        else:
            # Simple proportional allocation based on signals
            signal_sum = sum(abs(s) for s in signals.values())
            if signal_sum > 0:
                target_weights = {
                    symbol: abs(signal) / signal_sum * confidence_scores.get(symbol, 0.5)
                    for symbol, signal in signals.items()
                }
            else:
                target_weights = current_weights
            optimization_info = {'status': 'simple_allocation'}
        
        # Calculate required trades
        trades_needed = self._calculate_required_trades(
            target_weights, current_weights, current_prices
        )
        
        # Execute trades if they meet minimum criteria
        executed_trades = []
        for trade_info in trades_needed:
            if abs(trade_info['trade_value']) >= self.min_trade_size:
                trade = self._execute_trade(
                    symbol=trade_info['symbol'],
                    shares=trade_info['shares'],
                    price=trade_info['price'],
                    strategy_source='MoE',
                    confidence_score=confidence_scores.get(trade_info['symbol'], 0.0),
                    market_regime=market_regime
                )
                executed_trades.append(trade)
        
        # Update portfolio snapshot
        self._take_portfolio_snapshot(timestamp)
        
        # Calculate attribution by expert
        expert_attribution = self._calculate_expert_attribution(aux_info, executed_trades)
        
        return {
            'timestamp': timestamp,
            'signals': signals,
            'confidence_scores': confidence_scores,
            'target_weights': target_weights,
            'executed_trades': [trade.to_dict() for trade in executed_trades],
            'optimization_info': optimization_info,
            'expert_attribution': expert_attribution,
            'routing_entropy': routing_entropy,
            'portfolio_value': self.get_total_value(current_prices)
        }
    
    def _extract_signals_from_moe(self,
                                 moe_output: torch.Tensor,
                                 aux_info: Dict,
                                 symbol_mapping: Dict[int, str]) -> Dict[str, float]:
        """Extract trading signals from MoE output"""
        signals = {}
        
        # Handle batch output - take mean across batch
        if moe_output.dim() == 2:
            signal_values = moe_output.mean(0).detach().cpu().numpy()
        else:
            signal_values = moe_output.detach().cpu().numpy()
        
        # Map to symbols
        for i, value in enumerate(signal_values):
            if i in symbol_mapping:
                symbol = symbol_mapping[i]
                # Apply sigmoid to normalize signals to [-1, 1] range
                signals[symbol] = float(np.tanh(value))
        
        return signals
    
    def _calculate_signal_confidence(self,
                                   signals: Dict[str, float],
                                   expert_weights: torch.Tensor,
                                   routing_entropy: float) -> Dict[str, float]:
        """Calculate confidence scores for each signal based on expert consensus"""
        confidence_scores = {}
        
        # Base confidence from routing entropy (higher entropy = lower confidence)
        max_entropy = np.log(expert_weights.shape[-1])  # Maximum possible entropy
        entropy_confidence = 1.0 - (routing_entropy / max_entropy) if max_entropy > 0 else 0.5
        
        # Expert weight concentration (more concentrated = higher confidence)
        weight_concentration = float(torch.max(expert_weights).item())
        
        # Combined confidence score
        base_confidence = (entropy_confidence + weight_concentration) / 2.0
        
        for symbol, signal in signals.items():
            # Scale confidence by signal strength
            signal_strength = abs(signal)
            confidence_scores[symbol] = base_confidence * signal_strength
        
        return confidence_scores
    
    def _get_current_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        total_value = self.get_total_value(current_prices)
        
        if total_value <= 0:
            return {}
        
        weights = {}
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                market_value = position.shares * current_prices[symbol]
                weights[symbol] = market_value / total_value
        
        return weights
    
    def _estimate_covariance_matrix(self, symbols: List[str]) -> np.ndarray:
        """Estimate covariance matrix from historical returns"""
        returns_matrix = []
        
        for symbol in symbols:
            if symbol in self.returns_cache and len(self.returns_cache[symbol]) > 20:
                returns_matrix.append(list(self.returns_cache[symbol]))
            else:
                # Use default variance for new symbols
                returns_matrix.append([0.0] * 21)  # Minimum length
        
        if not returns_matrix:
            # Return identity matrix for single asset or no history
            return np.eye(len(symbols)) * 0.01
        
        # Align lengths
        min_length = min(len(returns) for returns in returns_matrix)
        if min_length < 20:
            # Not enough data - use simple diagonal matrix
            return np.eye(len(symbols)) * 0.01
        
        aligned_returns = np.array([returns[-min_length:] for returns in returns_matrix])
        
        try:
            cov_matrix = np.cov(aligned_returns)
            
            # Regularize if needed
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                cov_matrix = np.eye(len(symbols)) * 0.01
            else:
                # Add small diagonal term for numerical stability
                cov_matrix += np.eye(len(symbols)) * 1e-6
            
            return cov_matrix
            
        except Exception as e:
            logger.warning(f"Covariance estimation failed: {e}")
            return np.eye(len(symbols)) * 0.01
    
    def _calculate_required_trades(self,
                                 target_weights: Dict[str, float],
                                 current_weights: Dict[str, float],
                                 current_prices: Dict[str, float]) -> List[Dict]:
        """Calculate trades needed to reach target weights"""
        total_value = self.get_total_value(current_prices)
        trades_needed = []
        
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        
        for symbol in all_symbols:
            if symbol not in current_prices:
                continue
            
            target_weight = target_weights.get(symbol, 0.0)
            current_weight = current_weights.get(symbol, 0.0)
            
            weight_diff = target_weight - current_weight
            
            # Check if rebalancing threshold is met
            if abs(weight_diff) < self.rebalance_threshold:
                continue
            
            target_value = target_weight * total_value
            current_value = current_weight * total_value
            trade_value = target_value - current_value
            
            price = current_prices[symbol]
            shares = int(trade_value / price)
            
            if shares != 0:
                trades_needed.append({
                    'symbol': symbol,
                    'shares': shares,
                    'price': price,
                    'trade_value': trade_value,
                    'weight_diff': weight_diff
                })
        
        return trades_needed
    
    def _execute_trade(self,
                      symbol: str,
                      shares: float,
                      price: float,
                      strategy_source: str = 'Unknown',
                      confidence_score: float = 0.0,
                      market_regime: Optional[str] = None) -> Trade:
        """Execute a trade and update portfolio state"""
        timestamp = datetime.now()
        
        # Calculate commission
        commission = abs(shares * price) * self.commission_rate
        
        # Determine side
        side = 'BUY' if shares > 0 else 'SELL'
        
        # Update cash
        total_cost = shares * price + (commission if shares > 0 else -commission)
        self.cash -= total_cost
        
        # Update position
        if symbol in self.positions:
            position = self.positions[symbol]
            old_shares = position.shares
            old_value = old_shares * position.avg_cost
            new_shares = old_shares + shares
            
            if new_shares == 0:
                # Closing position
                del self.positions[symbol]
            else:
                # Update average cost
                if new_shares > 0:
                    new_value = old_value + shares * price
                    position.avg_cost = new_value / new_shares
                position.shares = new_shares
                position.current_price = price
                position.last_updated = timestamp
        else:
            # New position
            if shares > 0:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    avg_cost=price,
                    current_price=price,
                    last_updated=timestamp,
                    entry_date=timestamp,
                    strategy_source=strategy_source,
                    confidence_score=confidence_score
                )
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            shares=abs(shares),
            price=price,
            commission=commission,
            strategy_source=strategy_source,
            confidence_score=confidence_score,
            market_regime=market_regime
        )
        
        self.trade_history.append(trade)
        
        logger.info(
            f"Executed {side}: {symbol} {abs(shares)} shares @ ${price:.2f} "
            f"(Commission: ${commission:.2f}, Source: {strategy_source})"
        )
        
        return trade
    
    def _take_portfolio_snapshot(self, timestamp: datetime):
        """Take a snapshot of current portfolio state"""
        total_value = self.get_total_value()
        positions_value = total_value - self.cash
        
        # Calculate daily P&L
        daily_pnl = 0.0
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1].total_value
            daily_pnl = total_value - prev_value
        
        # Calculate cumulative P&L
        cumulative_pnl = total_value - self.initial_capital
        
        # Calculate sector exposures
        sector_exposures = self._calculate_sector_exposures()
        
        # Calculate concentration risk (Herfindahl index)
        concentration_risk = self._calculate_concentration_risk()
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=self.cash,
            positions_value=positions_value,
            num_positions=len(self.positions),
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            exposure_by_sector=sector_exposures,
            concentration_risk=concentration_risk
        )
        
        self.portfolio_history.append(snapshot)
        
        # Update portfolio returns cache
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2].total_value
            if prev_value > 0:
                portfolio_return = (total_value - prev_value) / prev_value
                self.portfolio_returns.append(portfolio_return)
    
    def _calculate_sector_exposures(self) -> Dict[str, float]:
        """Calculate exposure by sector"""
        total_value = self.get_total_value()
        if total_value <= 0:
            return {}
        
        sector_exposures = defaultdict(float)
        for position in self.positions.values():
            sector = position.sector or 'Unknown'
            exposure = position.market_value / total_value
            sector_exposures[sector] += exposure
        
        return dict(sector_exposures)
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration using Herfindahl index"""
        total_value = self.get_total_value()
        if total_value <= 0:
            return 0.0
        
        weights_squared = sum(
            (position.market_value / total_value) ** 2
            for position in self.positions.values()
        )
        
        return weights_squared
    
    def _calculate_expert_attribution(self,
                                    aux_info: Dict,
                                    executed_trades: List[Trade]) -> Dict[str, float]:
        """Calculate performance attribution by expert"""
        attribution = defaultdict(float)
        
        if not executed_trades:
            return dict(attribution)
        
        # Get expert weights and indices
        expert_weights = aux_info.get('topk_weights', torch.ones(1, 2))
        expert_indices = aux_info.get('topk_idx', torch.zeros(1, 2, dtype=torch.long))
        
        # Calculate trade value by expert
        total_trade_value = sum(trade.notional_value for trade in executed_trades)
        
        if total_trade_value > 0:
            # Distribute attribution based on expert weights
            for i, weight in enumerate(expert_weights.mean(0)):
                expert_id = f"Expert_{i}"
                attribution[expert_id] = float(weight.item()) * total_trade_value
        
        return dict(attribution)
    
    def get_total_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total portfolio value"""
        if prices is None:
            prices = self.price_cache
        
        positions_value = sum(
            position.shares * prices.get(position.symbol, position.current_price)
            for position in self.positions.values()
        )
        
        return self.cash + positions_value
    
    def get_portfolio_summary(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        if prices is None:
            prices = self.price_cache
        
        total_value = self.get_total_value(prices)
        
        # Position summaries
        position_summaries = [
            position.to_dict() for position in self.positions.values()
        ]
        
        # Performance metrics
        performance = self._calculate_performance_metrics()
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'num_positions': len(self.positions),
            'positions': position_summaries,
            'performance': performance,
            'risk_metrics': risk_metrics,
            'sector_exposures': self._calculate_sector_exposures(),
            'recent_trades': [trade.to_dict() for trade in self.trade_history[-10:]],
            'concentration_risk': self._calculate_concentration_risk()
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_returns:
            return {}
        
        returns = np.array(self.portfolio_returns)
        
        # Basic metrics
        total_return = (self.get_total_value() - self.initial_capital) / self.initial_capital
        annualized_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(returns > 0) if len(returns) > 0 else 0.0
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        if not self.portfolio_returns:
            return {}
        
        returns = np.array(self.portfolio_returns)
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0.0
        
        # Tail ratio
        upside_returns = returns[returns > 0]
        downside_returns = returns[returns < 0]
        
        tail_ratio = (
            (np.mean(upside_returns) if len(upside_returns) > 0 else 0.0) /
            abs(np.mean(downside_returns) if len(downside_returns) > 0 else 1.0)
        )
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_ratio': tail_ratio,
            'concentration_risk': self._calculate_concentration_risk()
        }
    
    def rebalance_legacy(self, signals: Dict[str, float], prices: Dict[str, float]):
        """
        Legacy rebalancing method for backward compatibility
        Converts simple signals dict to enhanced MoE format
        """
        # Convert legacy signals to MoE-like format
        batch_size = 1
        output_dim = len(signals)
        
        # Create mock MoE output
        signal_values = list(signals.values())
        moe_output = torch.tensor(signal_values).unsqueeze(0)  # [1, output_dim]
        
        # Create mock auxiliary info
        aux_info = {
            'topk_weights': torch.ones(batch_size, 2) * 0.5,  # Equal expert weights
            'topk_idx': torch.zeros(batch_size, 2, dtype=torch.long),  # Expert indices
            'routing_entropy': 0.693,  # log(2) for equal weights
            'expert_usage': {'frequency': torch.ones(2) * 0.5}
        }
        
        # Create symbol mapping
        symbol_mapping = {i: symbol for i, symbol in enumerate(signals.keys())}
        
        # Update prices
        self.update_prices(prices)
        
        # Process through enhanced pipeline
        result = self.process_moe_signals(
            moe_output=moe_output,
            aux_info=aux_info,
            symbol_mapping=symbol_mapping,
            current_prices=prices,
            market_regime='Normal'
        )
        
        return result
    
    def execute_trade_legacy(self, symbol: str, shares: float, price: float):
        """Legacy trade execution method for backward compatibility"""
        return self._execute_trade(
            symbol=symbol,
            shares=shares,
            price=price,
            strategy_source='Legacy',
            confidence_score=0.5
        )
    
    def export_state(self, filepath: str):
        """Export portfolio state to JSON file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'cash': self.cash,
            'initial_capital': self.initial_capital,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'trade_history': [trade.to_dict() for trade in self.trade_history],
            'performance_metrics': self._calculate_performance_metrics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'portfolio_summary': self.get_portfolio_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Portfolio state exported to {filepath}")
    
    def import_state(self, filepath: str):
        """Import portfolio state from JSON file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore basic state
        self.cash = state['cash']
        self.initial_capital = state['initial_capital']
        
        # Restore positions
        self.positions = {}
        for symbol, pos_data in state['positions'].items():
            self.positions[symbol] = Position(
                symbol=pos_data['symbol'],
                shares=pos_data['shares'],
                avg_cost=pos_data['avg_cost'],
                current_price=pos_data['current_price'],
                last_updated=datetime.fromisoformat(pos_data['entry_date']),
                entry_date=datetime.fromisoformat(pos_data['entry_date']),
                sector=pos_data.get('sector'),
                strategy_source=pos_data.get('strategy_source'),
                confidence_score=pos_data.get('confidence_score', 0.0)
            )
        
        # Restore trade history
        self.trade_history = []
        for trade_data in state.get('trade_history', []):
            self.trade_history.append(Trade(
                timestamp=datetime.fromisoformat(trade_data['timestamp']),
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                shares=trade_data['shares'],
                price=trade_data['price'],
                commission=trade_data.get('commission', 0.0),
                strategy_source=trade_data.get('strategy_source'),
                confidence_score=trade_data.get('confidence_score', 0.0),
                market_regime=trade_data.get('market_regime')
            ))
        
        logger.info(f"Portfolio state imported from {filepath}")

# =============================================================================
# Advanced Analytics and Reporting
# =============================================================================

class PortfolioAnalytics:
    """Advanced analytics for portfolio performance and risk"""
    
    def __init__(self, portfolio_manager: EnhancedPortfolioManager):
        self.pm = portfolio_manager
    
    def generate_performance_report(self, 
                                  benchmark_returns: Optional[List[float]] = None,
                                  period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Get recent portfolio history
        recent_history = self.pm.portfolio_history[-period_days:] if len(self.pm.portfolio_history) >= period_days else self.pm.portfolio_history
        
        if not recent_history:
            return {'error': 'No portfolio history available'}
        
        # Calculate period returns
        period_returns = []
        for i in range(1, len(recent_history)):
            prev_value = recent_history[i-1].total_value
            curr_value = recent_history[i].total_value
            if prev_value > 0:
                period_returns.append((curr_value - prev_value) / prev_value)
        
        if not period_returns:
            return {'error': 'Insufficient data for analysis'}
        
        period_returns = np.array(period_returns)
        
        # Performance metrics
        total_return = (recent_history[-1].total_value - recent_history[0].total_value) / recent_history[0].total_value
        annualized_return = np.mean(period_returns) * 252
        volatility = np.std(period_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Risk metrics
        var_95 = np.percentile(period_returns, 5)
        max_drawdown = self._calculate_max_drawdown([h.total_value for h in recent_history])
        
        # Trade analysis
        recent_trades = [t for t in self.pm.trade_history if t.timestamp >= recent_history[0].timestamp]
        
        # Expert attribution
        expert_attribution = self._analyze_expert_performance(recent_trades)
        
        return {
            'period_days': len(recent_history),
            'performance': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio
            },
            'risk_metrics': {
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'concentration_risk': self.pm._calculate_concentration_risk()
            },
            'trading_metrics': {
                'num_trades': len(recent_trades),
                'avg_trade_size': np.mean([t.notional_value for t in recent_trades]) if recent_trades else 0,
                'win_rate': self._calculate_win_rate(recent_trades)
            },
            'expert_attribution': expert_attribution,
            'sector_performance': self._analyze_sector_performance(recent_history)
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from value series"""
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate from trade history"""
        if not trades:
            return 0.0
        
        # Simple win rate calculation (would need position tracking for accurate P&L)
        buy_trades = [t for t in trades if t.side == 'BUY']
        sell_trades = [t for t in trades if t.side == 'SELL']
        
        if not buy_trades or not sell_trades:
            return 0.5  # Neutral if no complete round trips
        
        # Simplified: assume even distribution
        return 0.6  # Placeholder - would need more sophisticated tracking
    
    def _analyze_expert_performance(self, trades: List[Trade]) -> Dict[str, float]:
        """Analyze performance attribution by expert/strategy"""
        attribution = defaultdict(float)
        
        for trade in trades:
            source = trade.strategy_source or 'Unknown'
            attribution[source] += trade.notional_value
        
        total_volume = sum(attribution.values())
        if total_volume > 0:
            attribution = {k: v/total_volume for k, v in attribution.items()}
        
        return dict(attribution)
    
    def _analyze_sector_performance(self, history: List[PortfolioSnapshot]) -> Dict[str, float]:
        """Analyze performance by sector"""
        if len(history) < 2:
            return {}
        
        # Get sector exposure changes
        start_exposures = history[0].exposure_by_sector
        end_exposures = history[-1].exposure_by_sector
        
        sector_performance = {}
        for sector in set(start_exposures.keys()) | set(end_exposures.keys()):
            start_exp = start_exposures.get(sector, 0.0)
            end_exp = end_exposures.get(sector, 0.0)
            sector_performance[sector] = end_exp - start_exp
        
        return sector_performance

# =============================================================================
# Testing and Example Usage
# =============================================================================

def test_enhanced_portfolio_manager():
    """Test the enhanced portfolio manager"""
    
    print("Testing Enhanced Portfolio Manager...")
    
    # Initialize portfolio manager
    pm = EnhancedPortfolioManager(
        initial_capital=1000000,
        commission_rate=0.001,
        max_position_size=0.1,
        use_optimization=True
    )
    
    # Test data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 'TSLA': 200.0}
    
    # Update prices
    pm.update_prices(prices)
    
    # Create mock MoE signals
    batch_size = 1
    output_dim = len(symbols)
    
    # Mock MoE output (normalized signals)
    moe_output = torch.tensor([[0.8, -0.3, 0.5, -0.1]])  # [batch, output_dim]
    
    # Mock auxiliary info
    aux_info = {
        'topk_weights': torch.tensor([[0.7, 0.3]]),  # Expert weights
        'topk_idx': torch.tensor([[0, 1]]),  # Expert indices
        'routing_entropy': 0.5,
        'expert_usage': {'frequency': torch.tensor([0.6, 0.4])},
        'load_balance_loss': torch.tensor(0.01)
    }
    
    # Symbol mapping
    symbol_mapping = {i: symbol for i, symbol in enumerate(symbols)}
    
    print(f"Initial portfolio value: ${pm.get_total_value(prices):,.2f}")
    print(f"Initial cash: ${pm.cash:,.2f}")
    
    # Process MoE signals
    result = pm.process_moe_signals(
        moe_output=moe_output,
        aux_info=aux_info,
        symbol_mapping=symbol_mapping,
        current_prices=prices,
        market_regime='Bull'
    )
    
    print(f"\nMoE Signal Processing Results:")
    print(f"Signals: {result['signals']}")
    print(f"Target weights: {result['target_weights']}")
    print(f"Executed trades: {len(result['executed_trades'])}")
    print(f"Portfolio value after rebalancing: ${result['portfolio_value']:,.2f}")
    
    # Test portfolio summary
    summary = pm.get_portfolio_summary(prices)
    print(f"\nPortfolio Summary:")
    print(f"Total value: ${summary['total_value']:,.2f}")
    print(f"Number of positions: {summary['num_positions']}")
    print(f"Cash remaining: ${summary['cash']:,.2f}")
    
    # Test legacy compatibility
    print(f"\n--- Testing Legacy Compatibility ---")
    legacy_signals = {'AAPL': 0.5, 'MSFT': -0.2}
    legacy_result = pm.rebalance_legacy(legacy_signals, prices)
    print(f"Legacy rebalancing completed: {len(legacy_result['executed_trades'])} trades")
    
    # Test analytics
    analytics = PortfolioAnalytics(pm)
    if pm.portfolio_history:
        report = analytics.generate_performance_report(period_days=1)
        print(f"\nPerformance Report:")
        print(f"Performance metrics available: {list(report.keys())}")
    
    # Test state export/import
    pm.export_state('test_portfolio_state.json')
    print(f"Portfolio state exported successfully")
    
    print("\nEnhanced Portfolio Manager testing completed!")

if __name__ == "__main__":
    test_enhanced_portfolio_manager()