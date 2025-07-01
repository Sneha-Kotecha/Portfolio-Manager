import numpy as np
import pandas as pd
import logging
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
from scipy import stats
from scipy.optimize import minimize
import warnings
import json
from enum import Enum

try:
    from Data import get_live_price  # Assuming Data.py is in the same directory
except ImportError:
    def get_live_price(symbol, data=None):
        """Fallback function if Data.py is not available"""
        logger.warning(f"get_live_price not available, using mock price for {symbol}")
        return 100.0  # Mock price

# =============================================================================
# Logger Setup
# =============================================================================
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Enhanced Risk Management Data Structures
# =============================================================================

class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Types of risk alerts"""
    VAR_BREACH = "VAR_BREACH"
    CONCENTRATION = "CONCENTRATION"
    CORRELATION = "CORRELATION"
    DRAWDOWN = "DRAWDOWN"
    VOLATILITY = "VOLATILITY"
    LIQUIDITY = "LIQUIDITY"
    POSITION_SIZE = "POSITION_SIZE"
    SECTOR_EXPOSURE = "SECTOR_EXPOSURE"
    EXPERT_DIVERGENCE = "EXPERT_DIVERGENCE"

@dataclass
class RiskAlert:
    """Risk alert with detailed information"""
    timestamp: datetime
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    current_value: float
    threshold: float
    affected_positions: List[str]
    recommended_actions: List[str]
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'affected_positions': self.affected_positions,
            'recommended_actions': self.recommended_actions,
            'confidence_score': self.confidence_score
        }

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics snapshot"""
    timestamp: datetime
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float
    current_drawdown: float
    volatility_annual: float
    sharpe_ratio: float
    beta: float
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: float
    expert_consensus_risk: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'volatility_annual': self.volatility_annual,
            'sharpe_ratio': self.sharpe_ratio,
            'beta': self.beta,
            'concentration_risk': self.concentration_risk,
            'correlation_risk': self.correlation_risk,
            'liquidity_risk': self.liquidity_risk,
            'expert_consensus_risk': self.expert_consensus_risk
        }

# =============================================================================
# Advanced Risk Calculation Engine
# =============================================================================

class RiskCalculationEngine:
    """Advanced risk calculation methods"""
    
    @staticmethod
    def calculate_var(returns: np.ndarray, 
                     confidence: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk using multiple methods
        
        Args:
            returns: Array of portfolio returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'
        """
        if len(returns) == 0:
            return 0.0
        
        if method == 'historical':
            # Historical simulation
            sorted_returns = np.sort(returns)
            idx = int((1 - confidence) * len(sorted_returns))
            return abs(sorted_returns[idx]) if idx < len(sorted_returns) else 0.0
        
        elif method == 'parametric':
            # Parametric VaR (assumes normal distribution)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence)
            return abs(mean_return + z_score * std_return)
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            simulated_returns = np.random.normal(mean_return, std_return, 10000)
            sorted_sim = np.sort(simulated_returns)
            idx = int((1 - confidence) * len(sorted_sim))
            return abs(sorted_sim[idx])
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var_threshold = RiskCalculationEngine.calculate_var(returns, confidence)
        tail_losses = returns[returns <= -var_threshold]
        
        return abs(np.mean(tail_losses)) if len(tail_losses) > 0 else var_threshold
    
    @staticmethod
    def calculate_maximum_drawdown(portfolio_values: np.ndarray) -> Tuple[float, float]:
        """
        Calculate maximum drawdown and current drawdown
        
        Returns:
            Tuple of (max_drawdown, current_drawdown)
        """
        if len(portfolio_values) < 2:
            return 0.0, 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdowns
        drawdowns = (portfolio_values - running_max) / running_max
        
        max_drawdown = abs(np.min(drawdowns))
        current_drawdown = abs(drawdowns[-1])
        
        return max_drawdown, current_drawdown
    
    @staticmethod
    def calculate_correlation_risk(correlation_matrix: np.ndarray) -> float:
        """
        Calculate correlation risk based on eigenvalue concentration
        """
        if correlation_matrix.size == 0:
            return 0.0
        
        try:
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Remove non-positive eigenvalues
            
            if len(eigenvalues) == 0:
                return 0.0
            
            # Concentration ratio (largest eigenvalue / sum of eigenvalues)
            concentration = np.max(eigenvalues) / np.sum(eigenvalues)
            
            return concentration
        except:
            return 0.0
    
    @staticmethod
    def calculate_liquidity_risk(positions: Dict, daily_volumes: Dict[str, float]) -> float:
        """
        Calculate liquidity risk based on position sizes vs daily volumes
        """
        if not positions or not daily_volumes:
            return 0.0
        
        liquidity_ratios = []
        for symbol, position in positions.items():
            volume = daily_volumes.get(symbol, 1e6)  # Default volume if not available
            position_value = abs(position.get('market_value', 0))
            
            if volume > 0:
                # Position size as fraction of daily volume
                liquidity_ratio = position_value / (volume * position.get('current_price', 1))
                liquidity_ratios.append(liquidity_ratio)
        
        return np.mean(liquidity_ratios) if liquidity_ratios else 0.0

# =============================================================================
# Enhanced Risk Manager
# =============================================================================

class EnhancedRiskManager:
    """
    Advanced Risk Manager with multi-layered risk monitoring,
    MoE integration, and sophisticated risk controls
    """
    
    def __init__(self,
                 var_threshold_95: float = 0.02,
                 var_threshold_99: float = 0.05,
                 max_drawdown_threshold: float = 0.15,
                 max_position_size: float = 0.1,
                 max_sector_exposure: float = 0.3,
                 max_concentration: float = 0.5,
                 volatility_threshold: float = 0.25,
                 correlation_threshold: float = 0.8,
                 liquidity_threshold: float = 0.1,
                 expert_divergence_threshold: float = 0.7,
                 lookback_window: int = 252,
                 enable_auto_hedge: bool = True):
        
        # Risk thresholds
        self.var_threshold_95 = var_threshold_95
        self.var_threshold_99 = var_threshold_99
        self.max_drawdown_threshold = max_drawdown_threshold
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_concentration = max_concentration
        self.volatility_threshold = volatility_threshold
        self.correlation_threshold = correlation_threshold
        self.liquidity_threshold = liquidity_threshold
        self.expert_divergence_threshold = expert_divergence_threshold
        
        # Analysis parameters
        self.lookback_window = lookback_window
        self.enable_auto_hedge = enable_auto_hedge
        
        # Risk calculation engine
        self.calc_engine = RiskCalculationEngine()
        
        # Historical tracking
        self.risk_metrics_history: List[RiskMetrics] = []
        self.alerts_history: List[RiskAlert] = []
        self.returns_cache: deque = deque(maxlen=lookback_window)
        self.portfolio_values_cache: deque = deque(maxlen=lookback_window)
        
        # Risk state
        self.current_risk_level: RiskLevel = RiskLevel.LOW
        self.active_alerts: List[RiskAlert] = []
        self.risk_override: bool = False
        
        # Market data for risk calculations
        self.benchmark_returns: deque = deque(maxlen=lookback_window)
        self.correlation_matrix: Optional[np.ndarray] = None
        self.volatility_estimates: Dict[str, float] = {}
        
        logger.info("Enhanced Risk Manager initialized with comprehensive risk monitoring")
    
    def update_portfolio_state(self,
                             portfolio_value: float,
                             portfolio_return: Optional[float] = None,
                             positions: Optional[Dict] = None,
                             timestamp: Optional[datetime] = None):
        """Update portfolio state for risk monitoring"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update caches
        self.portfolio_values_cache.append(portfolio_value)
        
        if portfolio_return is not None:
            self.returns_cache.append(portfolio_return)
        elif len(self.portfolio_values_cache) >= 2:
            # Calculate return from values
            prev_value = self.portfolio_values_cache[-2]
            current_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            self.returns_cache.append(current_return)
        
        # Calculate comprehensive risk metrics
        risk_metrics = self._calculate_comprehensive_risk_metrics(positions, timestamp)
        self.risk_metrics_history.append(risk_metrics)
        
        # Check for risk violations
        new_alerts = self._check_risk_violations(risk_metrics, positions)
        
        # Update active alerts
        self._update_active_alerts(new_alerts)
        
        # Update overall risk level
        self._update_risk_level()
    
    def assess_moe_signals(self,
                          moe_output: torch.Tensor,
                          aux_info: Dict,
                          current_positions: Dict,
                          proposed_trades: List[Dict]) -> Dict[str, Any]:
        """
        Assess risk implications of MoE signals and proposed trades
        
        Args:
            moe_output: MoE model output
            aux_info: Auxiliary information from MoE
            current_positions: Current portfolio positions
            proposed_trades: Proposed trades from portfolio manager
            
        Returns:
            Risk assessment with recommendations
        """
        timestamp = datetime.now()
        
        # Analyze expert consensus and divergence
        expert_risk = self._analyze_expert_consensus_risk(aux_info)
        
        # Assess proposed trades impact
        trade_risk_assessment = self._assess_trade_risks(proposed_trades, current_positions)
        
        # Calculate post-trade portfolio metrics
        projected_portfolio = self._project_portfolio_after_trades(
            current_positions, proposed_trades
        )
        
        # Check for constraint violations
        constraint_violations = self._check_constraint_violations(projected_portfolio)
        
        # Generate risk-adjusted recommendations
        recommendations = self._generate_risk_recommendations(
            expert_risk, trade_risk_assessment, constraint_violations
        )
        
        return {
            'timestamp': timestamp.isoformat(),
            'overall_risk_level': self.current_risk_level.value,
            'expert_consensus_risk': expert_risk,
            'trade_risk_assessment': trade_risk_assessment,
            'constraint_violations': constraint_violations,
            'recommendations': recommendations,
            'approved_trades': self._filter_approved_trades(proposed_trades, recommendations),
            'risk_override_required': len(constraint_violations) > 0 and not self.risk_override
        }
    
    def _calculate_comprehensive_risk_metrics(self,
                                            positions: Optional[Dict],
                                            timestamp: datetime) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        returns_array = np.array(self.returns_cache) if self.returns_cache else np.array([])
        portfolio_values = np.array(self.portfolio_values_cache) if self.portfolio_values_cache else np.array([])
        
        # VaR calculations
        var_95 = self.calc_engine.calculate_var(returns_array, 0.95) if len(returns_array) > 0 else 0.0
        var_99 = self.calc_engine.calculate_var(returns_array, 0.99) if len(returns_array) > 0 else 0.0
        cvar_95 = self.calc_engine.calculate_cvar(returns_array, 0.95) if len(returns_array) > 0 else 0.0
        
        # Drawdown calculations
        max_drawdown, current_drawdown = self.calc_engine.calculate_maximum_drawdown(portfolio_values)
        
        # Volatility and Sharpe ratio
        volatility_annual = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 0 else 0.0
        mean_return = np.mean(returns_array) if len(returns_array) > 0 else 0.0
        sharpe_ratio = (mean_return * 252) / volatility_annual if volatility_annual > 0 else 0.0
        
        # Beta calculation (vs benchmark)
        beta = self._calculate_beta(returns_array)
        
        # Concentration and correlation risk
        concentration_risk = self._calculate_concentration_risk(positions)
        correlation_risk = self.calc_engine.calculate_correlation_risk(
            self.correlation_matrix if self.correlation_matrix is not None else np.array([])
        )
        
        # Liquidity risk (simplified)
        liquidity_risk = 0.0  # Would need volume data
        
        # Expert consensus risk
        expert_consensus_risk = 0.0  # Calculated separately in MoE assessment
        
        return RiskMetrics(
            timestamp=timestamp,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility_annual=volatility_annual,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            liquidity_risk=liquidity_risk,
            expert_consensus_risk=expert_consensus_risk
        )
    
    def _check_risk_violations(self,
                             risk_metrics: RiskMetrics,
                             positions: Optional[Dict]) -> List[RiskAlert]:
        """Check for risk threshold violations"""
        alerts = []
        timestamp = risk_metrics.timestamp
        
        # VaR violations
        if risk_metrics.var_95 > self.var_threshold_95:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type=AlertType.VAR_BREACH,
                risk_level=RiskLevel.HIGH if risk_metrics.var_95 > self.var_threshold_95 * 1.5 else RiskLevel.MEDIUM,
                message=f"95% VaR ({risk_metrics.var_95:.2%}) exceeds threshold ({self.var_threshold_95:.2%})",
                current_value=risk_metrics.var_95,
                threshold=self.var_threshold_95,
                affected_positions=list(positions.keys()) if positions else [],
                recommended_actions=["Reduce position sizes", "Increase diversification", "Consider hedging"]
            ))
        
        if risk_metrics.var_99 > self.var_threshold_99:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type=AlertType.VAR_BREACH,
                risk_level=RiskLevel.CRITICAL,
                message=f"99% VaR ({risk_metrics.var_99:.2%}) exceeds threshold ({self.var_threshold_99:.2%})",
                current_value=risk_metrics.var_99,
                threshold=self.var_threshold_99,
                affected_positions=list(positions.keys()) if positions else [],
                recommended_actions=["Emergency position reduction", "Implement stop-losses", "Review risk model"]
            ))
        
        # Drawdown violations
        if risk_metrics.max_drawdown > self.max_drawdown_threshold:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type=AlertType.DRAWDOWN,
                risk_level=RiskLevel.HIGH,
                message=f"Maximum drawdown ({risk_metrics.max_drawdown:.2%}) exceeds threshold ({self.max_drawdown_threshold:.2%})",
                current_value=risk_metrics.max_drawdown,
                threshold=self.max_drawdown_threshold,
                affected_positions=list(positions.keys()) if positions else [],
                recommended_actions=["Review strategy performance", "Consider position rebalancing", "Implement stricter stops"]
            ))
        
        # Concentration risk
        if risk_metrics.concentration_risk > self.max_concentration:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type=AlertType.CONCENTRATION,
                risk_level=RiskLevel.MEDIUM,
                message=f"Portfolio concentration ({risk_metrics.concentration_risk:.2%}) exceeds threshold ({self.max_concentration:.2%})",
                current_value=risk_metrics.concentration_risk,
                threshold=self.max_concentration,
                affected_positions=self._get_concentrated_positions(positions),
                recommended_actions=["Diversify holdings", "Reduce large positions", "Add uncorrelated assets"]
            ))
        
        # Volatility violations
        if risk_metrics.volatility_annual > self.volatility_threshold:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type=AlertType.VOLATILITY,
                risk_level=RiskLevel.MEDIUM,
                message=f"Portfolio volatility ({risk_metrics.volatility_annual:.2%}) exceeds threshold ({self.volatility_threshold:.2%})",
                current_value=risk_metrics.volatility_annual,
                threshold=self.volatility_threshold,
                affected_positions=list(positions.keys()) if positions else [],
                recommended_actions=["Reduce volatile positions", "Add stable assets", "Consider volatility hedging"]
            ))
        
        return alerts
    
    def _analyze_expert_consensus_risk(self, aux_info: Dict) -> Dict[str, float]:
        """Analyze risk from expert consensus/divergence"""
        expert_weights = aux_info.get('topk_weights', torch.ones(1, 2))
        routing_entropy = aux_info.get('routing_entropy', 0.0)
        expert_usage = aux_info.get('expert_usage', {})
        
        # Calculate consensus metrics
        weight_concentration = float(torch.max(expert_weights).item()) if expert_weights.numel() > 0 else 0.5
        weight_variance = float(torch.var(expert_weights).item()) if expert_weights.numel() > 1 else 0.0
        
        # Entropy-based consensus (lower entropy = higher consensus)
        max_entropy = np.log(expert_weights.shape[-1]) if expert_weights.numel() > 0 else 1.0
        consensus_score = 1.0 - (routing_entropy / max_entropy) if max_entropy > 0 else 0.5
        
        # Expert usage variance (higher variance = less balanced usage)
        usage_freq = expert_usage.get('frequency', torch.ones(2) * 0.5)
        usage_variance = float(torch.var(usage_freq).item()) if usage_freq.numel() > 1 else 0.0
        
        return {
            'consensus_score': consensus_score,
            'weight_concentration': weight_concentration,
            'weight_variance': weight_variance,
            'routing_entropy': routing_entropy,
            'usage_variance': usage_variance,
            'divergence_risk': 1.0 - consensus_score
        }
    
    def _assess_trade_risks(self,
                          proposed_trades: List[Dict],
                          current_positions: Dict) -> Dict[str, Any]:
        """Assess risk implications of proposed trades"""
        if not proposed_trades:
            return {'total_risk_score': 0.0, 'trade_risks': []}
        
        trade_risks = []
        total_exposure_change = 0.0
        
        for trade in proposed_trades:
            symbol = trade.get('symbol', '')
            trade_value = abs(trade.get('trade_value', 0.0))
            weight_change = abs(trade.get('weight_diff', 0.0))
            
            # Calculate individual trade risk
            position_risk = min(weight_change / self.max_position_size, 1.0)
            volatility_risk = self.volatility_estimates.get(symbol, 0.2) / self.volatility_threshold
            
            trade_risk_score = (position_risk + volatility_risk) / 2.0
            
            trade_risks.append({
                'symbol': symbol,
                'trade_value': trade_value,
                'weight_change': weight_change,
                'risk_score': trade_risk_score,
                'risk_level': self._score_to_risk_level(trade_risk_score).value
            })
            
            total_exposure_change += trade_value
        
        return {
            'total_risk_score': np.mean([tr['risk_score'] for tr in trade_risks]),
            'total_exposure_change': total_exposure_change,
            'trade_risks': trade_risks,
            'high_risk_trades': [tr for tr in trade_risks if tr['risk_score'] > 0.7]
        }
    
    def _project_portfolio_after_trades(self,
                                      current_positions: Dict,
                                      proposed_trades: List[Dict]) -> Dict:
        """Project portfolio state after executing proposed trades"""
        projected_portfolio = current_positions.copy()
        
        for trade in proposed_trades:
            symbol = trade.get('symbol', '')
            shares = trade.get('shares', 0)
            price = trade.get('price', 0)
            
            if symbol in projected_portfolio:
                current_shares = projected_portfolio[symbol].get('shares', 0)
                projected_portfolio[symbol]['shares'] = current_shares + shares
                projected_portfolio[symbol]['market_value'] = projected_portfolio[symbol]['shares'] * price
            else:
                projected_portfolio[symbol] = {
                    'shares': shares,
                    'current_price': price,
                    'market_value': shares * price
                }
        
        return projected_portfolio
    
    def _check_constraint_violations(self, projected_portfolio: Dict) -> List[Dict]:
        """Check for constraint violations in projected portfolio"""
        violations = []
        
        if not projected_portfolio:
            return violations
        
        total_value = sum(pos.get('market_value', 0) for pos in projected_portfolio.values())
        
        # Position size constraints
        for symbol, position in projected_portfolio.items():
            weight = position.get('market_value', 0) / total_value if total_value > 0 else 0
            if weight > self.max_position_size:
                violations.append({
                    'type': 'position_size',
                    'symbol': symbol,
                    'current_weight': weight,
                    'threshold': self.max_position_size,
                    'severity': 'HIGH' if weight > self.max_position_size * 1.5 else 'MEDIUM'
                })
        
        # Concentration constraint
        concentration = self._calculate_concentration_risk(projected_portfolio)
        if concentration > self.max_concentration:
            violations.append({
                'type': 'concentration',
                'current_value': concentration,
                'threshold': self.max_concentration,
                'severity': 'MEDIUM'
            })
        
        return violations
    
    def _generate_risk_recommendations(self,
                                     expert_risk: Dict,
                                     trade_risk: Dict,
                                     violations: List[Dict]) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        # Expert consensus recommendations
        if expert_risk.get('divergence_risk', 0) > self.expert_divergence_threshold:
            recommendations.append("High expert divergence detected - consider reducing position sizes")
        
        # Trade risk recommendations
        if trade_risk.get('total_risk_score', 0) > 0.7:
            recommendations.append("High-risk trades detected - consider phased execution")
        
        # Violation-based recommendations
        for violation in violations:
            if violation['type'] == 'position_size':
                recommendations.append(f"Reduce position size in {violation['symbol']}")
            elif violation['type'] == 'concentration':
                recommendations.append("Increase portfolio diversification")
        
        # General risk level recommendations
        if self.current_risk_level == RiskLevel.HIGH:
            recommendations.append("Consider overall risk reduction measures")
        elif self.current_risk_level == RiskLevel.CRITICAL:
            recommendations.append("Immediate risk reduction required")
        
        return recommendations
    
    def _filter_approved_trades(self,
                              proposed_trades: List[Dict],
                              recommendations: List[str]) -> List[Dict]:
        """Filter trades based on risk assessment"""
        if self.risk_override:
            return proposed_trades
        
        approved_trades = []
        
        for trade in proposed_trades:
            # Simple filtering logic - can be enhanced
            trade_value = abs(trade.get('trade_value', 0))
            weight_change = abs(trade.get('weight_diff', 0))
            
            # Reject trades that are too large
            if weight_change > self.max_position_size:
                logger.warning(f"Rejecting trade for {trade.get('symbol')} due to position size limit")
                continue
            
            # In critical risk mode, reject all large trades
            if self.current_risk_level == RiskLevel.CRITICAL and trade_value > 10000:
                logger.warning(f"Rejecting large trade for {trade.get('symbol')} due to critical risk level")
                continue
            
            approved_trades.append(trade)
        
        return approved_trades
    
    def enforce_risk_limits(self, portfolio_manager) -> Dict[str, Any]:
        """
        Enhanced risk limit enforcement with intelligent position management
        
        Args:
            portfolio_manager: Instance of EnhancedPortfolioManager
            
        Returns:
            Dictionary with enforcement actions taken
        """
        timestamp = datetime.now()
        actions_taken = []
        
        if not self.risk_metrics_history:
            return {'actions_taken': [], 'message': 'No risk metrics available'}
        
        current_metrics = self.risk_metrics_history[-1]
        
        # Get current portfolio state
        try:
            portfolio_summary = portfolio_manager.get_portfolio_summary()
            current_positions = {
                pos['symbol']: pos for pos in portfolio_summary.get('positions', [])
            }
            current_prices = {
                pos['symbol']: pos['current_price'] for pos in portfolio_summary.get('positions', [])
            }
        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            # Fallback for legacy portfolio manager
            try:
                current_value = portfolio_manager.cash + sum(
                    portfolio_manager.positions.get(sym, 0) * get_live_price(sym, None)
                    for sym in portfolio_manager.positions.keys()
                )
                current_positions = {}
                current_prices = {}
                portfolio_summary = {'total_value': current_value, 'positions': []}
            except Exception as fallback_error:
                logger.error(f"Fallback portfolio state calculation failed: {fallback_error}")
                return {'actions_taken': [], 'error': f'Cannot determine portfolio state: {fallback_error}'}
        
        # Critical VaR breach - emergency reduction
        if current_metrics.var_99 > self.var_threshold_99:
            logger.critical(f"Critical VaR breach: {current_metrics.var_99:.2%} > {self.var_threshold_99:.2%}")
            
            # Emergency position reduction
            for symbol, position in current_positions.items():
                current_weight = position['market_value'] / portfolio_summary['total_value']
                if current_weight > self.max_position_size * 0.5:  # Reduce large positions first
                    reduction_shares = int(position['shares'] * 0.3)  # Reduce by 30%
                    if reduction_shares > 0:
                        try:
                            price = current_prices.get(symbol, get_live_price(symbol, None))
                            portfolio_manager.execute_trade_legacy(symbol, -reduction_shares, price)
                            actions_taken.append(f"Emergency reduction: {symbol} -{reduction_shares} shares")
                        except Exception as e:
                            logger.error(f"Failed to execute emergency trade for {symbol}: {e}")
        
        # High VaR breach - scaled reduction
        elif current_metrics.var_95 > self.var_threshold_95:
            logger.warning(f"VaR breach: {current_metrics.var_95:.2%} > {self.var_threshold_95:.2%}")
            
            # Scale down positions proportionally
            scale_factor = self.var_threshold_95 / current_metrics.var_95
            
            for symbol, position in current_positions.items():
                current_shares = position['shares']
                target_shares = int(current_shares * scale_factor)
                reduction_shares = current_shares - target_shares
                
                if reduction_shares > 0:
                    try:
                        price = current_prices.get(symbol, get_live_price(symbol, None))
                        portfolio_manager.execute_trade_legacy(symbol, -reduction_shares, price)
                        actions_taken.append(f"VaR scaling: {symbol} -{reduction_shares} shares")
                    except Exception as e:
                        logger.error(f"Failed to execute scaling trade for {symbol}: {e}")
        
        # Concentration risk - reduce largest positions
        if current_metrics.concentration_risk > self.max_concentration:
            logger.warning(f"Concentration risk: {current_metrics.concentration_risk:.2%} > {self.max_concentration:.2%}")
            
            # Sort positions by weight (largest first)
            total_value = portfolio_summary['total_value']
            sorted_positions = sorted(
                current_positions.items(),
                key=lambda x: x[1]['market_value'] / total_value,
                reverse=True
            )
            
            for symbol, position in sorted_positions[:3]:  # Address top 3 positions
                current_weight = position['market_value'] / total_value
                if current_weight > self.max_position_size:
                    target_weight = self.max_position_size * 0.9  # 10% buffer
                    target_value = target_weight * total_value
                    current_value = position['market_value']
                    reduction_value = current_value - target_value
                    
                    if reduction_value > 0:
                        price = current_prices.get(symbol, get_live_price(symbol, None))
                        reduction_shares = int(reduction_value / price)
                        
                        if reduction_shares > 0:
                            try:
                                portfolio_manager.execute_trade_legacy(symbol, -reduction_shares, price)
                                actions_taken.append(f"Concentration reduction: {symbol} -{reduction_shares} shares")
                            except Exception as e:
                                logger.error(f"Failed to execute concentration trade for {symbol}: {e}")
        
        # Drawdown protection - implement stops
        if current_metrics.current_drawdown > self.max_drawdown_threshold * 0.8:
            logger.warning(f"Approaching drawdown limit: {current_metrics.current_drawdown:.2%}")
            
            # Implement protective stops on losing positions
            for symbol, position in current_positions.items():
                unrealized_pnl_pct = position.get('unrealized_pnl_pct', 0)
                if unrealized_pnl_pct < -0.1:  # 10% loss
                    stop_shares = int(position['shares'] * 0.5)  # Reduce by 50%
                    if stop_shares > 0:
                        try:
                            price = current_prices.get(symbol, get_live_price(symbol, None))
                            portfolio_manager.execute_trade_legacy(symbol, -stop_shares, price)
                            actions_taken.append(f"Stop loss: {symbol} -{stop_shares} shares")
                        except Exception as e:
                            logger.error(f"Failed to execute stop loss for {symbol}: {e}")
        
        return {
            'timestamp': timestamp.isoformat(),
            'actions_taken': actions_taken,
            'risk_level': self.current_risk_level.value,
            'metrics': current_metrics.to_dict()
        }
    
    def _calculate_beta(self, portfolio_returns: np.ndarray) -> float:
        """Calculate portfolio beta vs benchmark"""
        if len(portfolio_returns) == 0 or len(self.benchmark_returns) == 0:
            return 1.0
        
        # Align lengths
        min_length = min(len(portfolio_returns), len(self.benchmark_returns))
        if min_length < 20:
            return 1.0
        
        port_rets = portfolio_returns[-min_length:]
        bench_rets = np.array(list(self.benchmark_returns)[-min_length:])
        
        try:
            covariance = np.cov(port_rets, bench_rets)[0, 1]
            benchmark_variance = np.var(bench_rets)
            
            if benchmark_variance > 0:
                return covariance / benchmark_variance
            else:
                return 1.0
        except:
            return 1.0
    
    def _calculate_concentration_risk(self, positions: Optional[Dict]) -> float:
        """Calculate portfolio concentration risk (Herfindahl index)"""
        if not positions:
            return 0.0
        
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        if total_value <= 0:
            return 0.0
        
        weights_squared = sum(
            (pos.get('market_value', 0) / total_value) ** 2
            for pos in positions.values()
        )
        
        return weights_squared
    
    def _get_concentrated_positions(self, positions: Optional[Dict]) -> List[str]:
        """Get list of positions that contribute to concentration risk"""
        if not positions:
            return []
        
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        if total_value <= 0:
            return []
        
        concentrated = []
        for symbol, position in positions.items():
            weight = position.get('market_value', 0) / total_value
            if weight > self.max_position_size * 0.7:  # 70% of max position size
                concentrated.append(symbol)
        
        return concentrated
    
    def _update_active_alerts(self, new_alerts: List[RiskAlert]):
        """Update active alerts list"""
        # Add new alerts
        self.alerts_history.extend(new_alerts)
        
        # Update active alerts (keep only recent high-priority alerts)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [
            alert for alert in self.alerts_history
            if (alert.timestamp > cutoff_time and 
                alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        ]
    
    def _update_risk_level(self):
        """Update overall portfolio risk level"""
        if not self.risk_metrics_history:
            self.current_risk_level = RiskLevel.LOW
            return
        
        current_metrics = self.risk_metrics_history[-1]
        
        # Count critical violations
        critical_count = 0
        high_count = 0
        
        if current_metrics.var_99 > self.var_threshold_99:
            critical_count += 1
        if current_metrics.var_95 > self.var_threshold_95:
            high_count += 1
        if current_metrics.max_drawdown > self.max_drawdown_threshold:
            high_count += 1
        if current_metrics.concentration_risk > self.max_concentration:
            high_count += 1
        if current_metrics.volatility_annual > self.volatility_threshold:
            high_count += 1
        
        # Determine overall risk level
        if critical_count > 0:
            self.current_risk_level = RiskLevel.CRITICAL
        elif high_count >= 2:
            self.current_risk_level = RiskLevel.HIGH
        elif high_count == 1:
            self.current_risk_level = RiskLevel.MEDIUM
        else:
            self.current_risk_level = RiskLevel.LOW
    
    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert numerical risk score to risk level"""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard"""
        if not self.risk_metrics_history:
            return {'error': 'No risk data available'}
        
        current_metrics = self.risk_metrics_history[-1]
        
        # Recent alerts
        recent_alerts = [
            alert.to_dict() for alert in self.alerts_history[-10:]
        ]
        
        # Risk trends (last 30 periods)
        recent_metrics = self.risk_metrics_history[-30:]
        
        trends = {
            'var_95_trend': [m.var_95 for m in recent_metrics],
            'drawdown_trend': [m.current_drawdown for m in recent_metrics],
            'volatility_trend': [m.volatility_annual for m in recent_metrics],
            'concentration_trend': [m.concentration_risk for m in recent_metrics]
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_risk_level': self.current_risk_level.value,
            'current_metrics': current_metrics.to_dict(),
            'active_alerts': [alert.to_dict() for alert in self.active_alerts],
            'recent_alerts': recent_alerts,
            'risk_trends': trends,
            'thresholds': {
                'var_95_threshold': self.var_threshold_95,
                'var_99_threshold': self.var_threshold_99,
                'max_drawdown_threshold': self.max_drawdown_threshold,
                'max_position_size': self.max_position_size,
                'max_concentration': self.max_concentration,
                'volatility_threshold': self.volatility_threshold
            },
            'risk_override_active': self.risk_override
        }
    
    def set_risk_override(self, override: bool, reason: str = ""):
        """Set risk override to bypass certain controls"""
        self.risk_override = override
        logger.warning(f"Risk override set to {override}. Reason: {reason}")
        
        if override:
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.VAR_BREACH,  # Generic type
                risk_level=RiskLevel.HIGH,
                message=f"Risk override activated: {reason}",
                current_value=1.0,
                threshold=0.0,
                affected_positions=[],
                recommended_actions=["Review override necessity", "Monitor positions closely"]
            )
            self.alerts_history.append(alert)
    
    def update_benchmark_returns(self, benchmark_return: float):
        """Update benchmark returns for beta calculation"""
        self.benchmark_returns.append(benchmark_return)
    
    def update_correlation_matrix(self, correlation_matrix: np.ndarray):
        """Update correlation matrix for risk calculations"""
        self.correlation_matrix = correlation_matrix
    
    def export_risk_report(self, filepath: str, period_days: int = 30):
        """Export comprehensive risk report"""
        dashboard = self.get_risk_dashboard()
        
        # Add detailed analysis
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': period_days,
            'risk_dashboard': dashboard,
            'detailed_metrics': [
                m.to_dict() for m in self.risk_metrics_history[-period_days:]
            ],
            'all_alerts': [
                alert.to_dict() for alert in self.alerts_history
                if alert.timestamp > datetime.now() - timedelta(days=period_days)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Risk report exported to {filepath}")

# =============================================================================
# Legacy Compatibility
# =============================================================================

class RiskManager(EnhancedRiskManager):
    """
    Legacy compatibility wrapper for the original RiskManager interface
    """
    
    def __init__(self, var_threshold: float = 0.05):
        super().__init__(
            var_threshold_95=var_threshold,
            var_threshold_99=var_threshold * 2,
            enable_auto_hedge=True
        )
        self.var_threshold = var_threshold  # Store for legacy compatibility
    
    def compute_var(self, returns: Union[np.ndarray, List[float]], confidence: float = 0.95) -> float:
        """Legacy VaR computation method"""
        if isinstance(returns, list):
            returns = np.array(returns)
        
        return self.calc_engine.calculate_var(returns, confidence, method='historical')
    
    def enforce_limits(self, portfolio_manager, returns_history: Union[np.ndarray, List[float]]):
        """
        Legacy limit enforcement method
        
        Args:
            portfolio_manager: Portfolio manager instance (legacy or enhanced)
            returns_history: Historical returns array
        """
        if isinstance(returns_history, list):
            returns_history = np.array(returns_history)
        
        # Update our internal state with the returns
        for ret in returns_history[-min(len(returns_history), 50):]:  # Only use recent returns
            self.returns_cache.append(ret)
        
        # Calculate current portfolio value (estimate if not available)
        try:
            current_value = portfolio_manager.get_total_value()
        except (AttributeError, TypeError):
            # Fallback for legacy portfolio manager
            try:
                current_value = portfolio_manager.cash + sum(
                    portfolio_manager.positions.get(sym, 0) * get_live_price(sym, None)
                    for sym in portfolio_manager.positions.keys()
                )
            except Exception as e:
                logger.error(f"Cannot calculate portfolio value: {e}")
                current_value = getattr(portfolio_manager, 'cash', 100000)  # Fallback
        
        # Update portfolio state
        self.update_portfolio_state(
            portfolio_value=current_value,
            portfolio_return=returns_history[-1] if len(returns_history) > 0 else 0.0
        )
        
        # Use enhanced enforcement
        enforcement_result = self.enforce_risk_limits(portfolio_manager)
        
        # Log results in legacy format
        var = self.compute_var(returns_history)
        if var > self.var_threshold:
            logger.warning(f"VaR {var:.2%} exceeds threshold; actions taken: {len(enforcement_result['actions_taken'])}")

# =============================================================================
# Testing and Example Usage
# =============================================================================

def test_enhanced_risk_manager():
    """Test the enhanced risk manager"""
    
    print("Testing Enhanced Risk Manager...")
    
    # Initialize risk manager
    risk_manager = EnhancedRiskManager(
        var_threshold_95=0.02,
        max_drawdown_threshold=0.15,
        max_position_size=0.1,
        enable_auto_hedge=True
    )
    
    # Simulate portfolio returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
    portfolio_values = [1000000]
    
    for ret in returns:
        new_value = portfolio_values[-1] * (1 + ret)
        portfolio_values.append(new_value)
        
        # Update risk manager
        risk_manager.update_portfolio_state(
            portfolio_value=new_value,
            portfolio_return=ret
        )
    
    print(f"Simulated {len(returns)} days of trading")
    print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
    print(f"Current risk level: {risk_manager.current_risk_level.value}")
    
    # Test MoE signal assessment
    print(f"\n--- Testing MoE Signal Assessment ---")
    
    # Create mock MoE output
    moe_output = torch.tensor([[0.5, -0.3, 0.2]])
    aux_info = {
        'topk_weights': torch.tensor([[0.8, 0.2]]),
        'topk_idx': torch.tensor([[0, 1]]),
        'routing_entropy': 0.3,
        'expert_usage': {'frequency': torch.tensor([0.7, 0.3])}
    }
    
    current_positions = {
        'AAPL': {'shares': 100, 'current_price': 150, 'market_value': 15000},
        'GOOGL': {'shares': 10, 'current_price': 2500, 'market_value': 25000}
    }
    
    proposed_trades = [
        {'symbol': 'AAPL', 'shares': 50, 'price': 150, 'trade_value': 7500, 'weight_diff': 0.05},
        {'symbol': 'TSLA', 'shares': 25, 'price': 200, 'trade_value': 5000, 'weight_diff': 0.03}
    ]
    
    assessment = risk_manager.assess_moe_signals(
        moe_output=moe_output,
        aux_info=aux_info,
        current_positions=current_positions,
        proposed_trades=proposed_trades
    )
    
    print(f"MoE Risk Assessment:")
    print(f"  Overall risk level: {assessment['overall_risk_level']}")
    print(f"  Expert consensus risk: {assessment['expert_consensus_risk']['consensus_score']:.3f}")
    print(f"  Approved trades: {len(assessment['approved_trades'])}/{len(proposed_trades)}")
    
    # Test risk dashboard
    dashboard = risk_manager.get_risk_dashboard()
    print(f"\n--- Risk Dashboard ---")
    print(f"Current risk level: {dashboard['current_risk_level']}")
    print(f"Active alerts: {len(dashboard['active_alerts'])}")
    print(f"VaR 95%: {dashboard['current_metrics']['var_95']:.3f}")
    print(f"Max drawdown: {dashboard['current_metrics']['max_drawdown']:.3f}")
    
    # Test legacy compatibility
    print(f"\n--- Testing Legacy Compatibility ---")
    legacy_risk_manager = RiskManager(var_threshold=0.05)
    
    # Mock legacy portfolio manager
    class MockPortfolioManager:
        def __init__(self):
            self.cash = 50000
            self.positions = {'AAPL': 100, 'GOOGL': 10}
        
        def execute_trade(self, symbol, shares, price):
            """Legacy execute_trade method"""
            print(f"Legacy trade executed: {symbol} {shares} @ {price}")
            # Update positions
            if symbol in self.positions:
                self.positions[symbol] += shares
            else:
                self.positions[symbol] = shares
        
        def execute_trade_legacy(self, symbol, shares, price):
            """Wrapper for legacy compatibility"""
            return self.execute_trade(symbol, shares, price)
    
    mock_pm = MockPortfolioManager()
    legacy_risk_manager.enforce_limits(mock_pm, returns[-30:])
    
    print("Legacy compatibility test completed")
    
    # Export risk report
    risk_manager.export_risk_report('test_risk_report.json')
    print("Risk report exported to test_risk_report.json")
    
    print("\nEnhanced Risk Manager testing completed!")

if __name__ == "__main__":
    test_enhanced_risk_manager()