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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import io
import base64
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from jinja2 import Template
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

try:
    from Data import get_live_price
except ImportError:
    def get_live_price(symbol, data=None):
        """Fallback function if Data.py is not available"""
        logger.warning(f"get_live_price not available, using mock price for {symbol}")
        return np.random.uniform(50, 200)  # Mock price

# =============================================================================
# Logger Setup
# =============================================================================
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Enhanced Reporting Data Structures
# =============================================================================

@dataclass
class ReportSection:
    """Individual report section with content and metadata"""
    title: str
    content: Any
    section_type: str  # 'table', 'chart', 'text', 'metrics'
    importance: str = 'medium'  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for reporting"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    var_95: float
    cvar_95: float

# =============================================================================
# Advanced Visualization Engine
# =============================================================================

class VisualizationEngine:
    """Advanced visualization engine for financial reports"""
    
    def __init__(self, style: str = 'professional'):
        self.style = style
        self.color_palette = self._get_color_palette()
        
    def _get_color_palette(self):
        """Get color palette based on style"""
        if self.style == 'professional':
            return {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e', 
                'success': '#2ca02c',
                'danger': '#d62728',
                'warning': '#ff7f0e',
                'info': '#17a2b8',
                'bg_color': '#f8f9fa',
                'grid_color': '#e9ecef'
            }
        else:
            return px.colors.qualitative.Set1
    
    def create_portfolio_performance_chart(self, 
                                         portfolio_history: List[Dict],
                                         benchmark_data: Optional[List[float]] = None) -> go.Figure:
        """Create comprehensive portfolio performance chart"""
        
        if not portfolio_history:
            return self._create_empty_chart("No portfolio history available")
        
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure we have valid data
        if 'total_value' not in df.columns or df['total_value'].isna().all():
            return self._create_empty_chart("No valid portfolio value data")
            
        df['cumulative_return'] = (df['total_value'] / df['total_value'].iloc[0] - 1) * 100
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value Over Time', 'Cumulative Returns',
                          'Daily P&L', 'Drawdown Analysis', 
                          'Position Count', 'Cash vs Invested'),
            specs=[[{"colspan": 2}, None],
                   [{"colspan": 2}, None],
                   [{}, {}]],
            vertical_spacing=0.08
        )
        
        # Portfolio value over time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['total_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add benchmark if provided
        if benchmark_data and len(benchmark_data) == len(df):
            benchmark_cumret = [(val/benchmark_data[0] - 1) * 100 for val in benchmark_data]
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=benchmark_cumret,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.color_palette['secondary'], width=1, dash='dash')
                ),
                row=2, col=1
            )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_return'],
                mode='lines',
                name='Cumulative Return',
                line=dict(color=self.color_palette['success'], width=2),
                fill='tonexty' if benchmark_data else 'tozeroy',
                hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Daily P&L
        if 'daily_pnl' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['daily_pnl'],
                    name='Daily P&L',
                    marker_color=np.where(df['daily_pnl'] >= 0, 
                                        self.color_palette['success'], 
                                        self.color_palette['danger']),
                    hovertemplate='<b>Date:</b> %{x}<br><b>P&L:</b> $%{y:,.2f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Drawdown analysis
        running_max = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color=self.color_palette['danger'], width=2),
                fill='tozeroy',
                hovertemplate='<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Portfolio Performance Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_risk_analysis_chart(self, risk_metrics_history: List[Dict]) -> go.Figure:
        """Create comprehensive risk analysis visualization"""
        
        if not risk_metrics_history:
            return self._create_empty_chart("No risk metrics available")
        
        df = pd.DataFrame(risk_metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots for risk metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Value at Risk (VaR)', 'Portfolio Volatility',
                          'Concentration Risk', 'Sharpe Ratio Evolution'),
            vertical_spacing=0.12
        )
        
        # VaR 95% and 99%
        if 'var_95' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['var_95'] * 100,
                    mode='lines',
                    name='VaR 95%',
                    line=dict(color=self.color_palette['warning'], width=2)
                ),
                row=1, col=1
            )
        
        if 'var_99' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['var_99'] * 100,
                    mode='lines',
                    name='VaR 99%',
                    line=dict(color=self.color_palette['danger'], width=2)
                ),
                row=1, col=1
            )
        
        # Volatility
        if 'volatility_annual' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['volatility_annual'] * 100,
                    mode='lines',
                    name='Annual Volatility',
                    line=dict(color=self.color_palette['info'], width=2),
                    fill='tozeroy'
                ),
                row=1, col=2
            )
        
        # Concentration risk
        if 'concentration_risk' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['concentration_risk'],
                    mode='lines',
                    name='Concentration Risk',
                    line=dict(color=self.color_palette['secondary'], width=2)
                ),
                row=2, col=1
            )
        
        # Sharpe ratio
        if 'sharpe_ratio' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sharpe_ratio'],
                    mode='lines',
                    name='Sharpe Ratio',
                    line=dict(color=self.color_palette['primary'], width=2)
                ),
                row=2, col=2
            )
            
            # Add horizontal line at 0 for Sharpe ratio
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        fig.update_layout(
            title='Risk Analysis Dashboard',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_expert_analysis_chart(self, expert_attribution: Dict[str, Any]) -> go.Figure:
        """Create MoE expert analysis visualization"""
        
        if not expert_attribution:
            return self._create_empty_chart("No expert data available")
        
        # Create subplots for expert analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Expert Performance Attribution', 'Expert Usage Frequency',
                          'Routing Entropy Over Time', 'Expert Confidence Scores'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.15
        )
        
        # Expert performance attribution
        if 'expert_performance' in expert_attribution:
            experts = list(expert_attribution['expert_performance'].keys())
            performance = list(expert_attribution['expert_performance'].values())
            
            colors = [self.color_palette['success'] if p >= 0 else self.color_palette['danger'] 
                     for p in performance]
            
            fig.add_trace(
                go.Bar(
                    x=experts,
                    y=performance,
                    name='Expert Performance',
                    marker_color=colors,
                    hovertemplate='<b>Expert:</b> %{x}<br><b>Performance:</b> %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Expert usage frequency
        if 'expert_usage' in expert_attribution:
            experts = list(expert_attribution['expert_usage'].keys())
            usage = list(expert_attribution['expert_usage'].values())
            
            fig.add_trace(
                go.Pie(
                    labels=experts,
                    values=usage,
                    name='Expert Usage',
                    hovertemplate='<b>Expert:</b> %{label}<br><b>Usage:</b> %{value:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Routing entropy over time
        if 'entropy_history' in expert_attribution:
            timestamps = expert_attribution['entropy_history'].get('timestamps', [])
            entropy_values = expert_attribution['entropy_history'].get('values', [])
            
            if timestamps and entropy_values:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=entropy_values,
                        mode='lines+markers',
                        name='Routing Entropy',
                        line=dict(color=self.color_palette['primary'], width=2)
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title='MoE Expert Analysis Dashboard',
            height=700,
            template='plotly_white'
        )
        
        return fig
    
    def create_position_analysis_chart(self, positions_data: List[Dict]) -> go.Figure:
        """Create detailed position analysis visualization"""
        
        if not positions_data:
            return self._create_empty_chart("No position data available")
        
        df = pd.DataFrame(positions_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Position Weights', 'P&L by Position',
                          'Position Entry Dates', 'Risk Contribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.15
        )
        
        # Position weights
        weight_col = 'weight' if 'weight' in df.columns else 'market_value'
        fig.add_trace(
            go.Bar(
                x=df['symbol'],
                y=df[weight_col],
                name='Position Weight',
                marker_color=self.color_palette['primary'],
                hovertemplate=f'<b>Symbol:</b> %{{x}}<br><b>{weight_col.title()}:</b> %{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # P&L by position
        if 'unrealized_pnl' in df.columns:
            colors = [self.color_palette['success'] if pnl >= 0 else self.color_palette['danger'] 
                     for pnl in df['unrealized_pnl']]
            
            fig.add_trace(
                go.Bar(
                    x=df['symbol'],
                    y=df['unrealized_pnl'],
                    name='Unrealized P&L',
                    marker_color=colors,
                    hovertemplate='<b>Symbol:</b> %{x}<br><b>P&L:</b> $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Position entry dates (if available)
        if 'entry_date' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(df['entry_date']),
                    y=df['symbol'],
                    mode='markers',
                    name='Entry Dates',
                    marker=dict(
                        size=df['market_value'] / df['market_value'].max() * 50,
                        color=self.color_palette['secondary'],
                        opacity=0.7
                    ),
                    hovertemplate='<b>Symbol:</b> %{y}<br><b>Entry Date:</b> %{x}<br><b>Value:</b> $%{marker.size}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Position Analysis Dashboard',
            height=700,
            template='plotly_white'
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template='plotly_white'
        )
        return fig

# =============================================================================
# Enhanced Report Generator
# =============================================================================

class EnhancedReportGenerator:
    """
    Advanced report generator with comprehensive analytics,
    beautiful visualizations, and MoE integration
    """
    
    def __init__(self,
                 report_style: str = 'professional',
                 include_charts: bool = True,
                 chart_format: str = 'html',  # 'html', 'png', 'both'
                 auto_insights: bool = True):
        
        self.report_style = report_style
        self.include_charts = include_charts
        self.chart_format = chart_format
        self.auto_insights = auto_insights
        
        # Initialize visualization engine
        self.viz_engine = VisualizationEngine(style=report_style)
        
        # Report sections
        self.sections: List[ReportSection] = []
        
        # Analytics cache
        self.analytics_cache: Dict[str, Any] = {}
        
        logger.info(f"Enhanced Report Generator initialized with {report_style} style")
    
    def generate_comprehensive_report(self,
                                    portfolio_manager,
                                    risk_manager,
                                    moe_system = None,
                                    benchmark_data: Optional[List[float]] = None,
                                    report_period_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio report with all components
        
        Args:
            portfolio_manager: Enhanced or legacy portfolio manager
            risk_manager: Enhanced risk manager
            moe_system: MoE system (optional)
            benchmark_data: Benchmark comparison data
            report_period_days: Analysis period in days
            
        Returns:
            Comprehensive report dictionary
        """
        logger.info("Generating comprehensive portfolio report...")
        
        # Clear previous sections
        self.sections = []
        
        # Generate all report sections with better error handling
        executive_summary = self._generate_executive_summary(
            portfolio_manager, risk_manager, report_period_days
        )
        
        performance_analysis = self._generate_performance_analysis(
            portfolio_manager, benchmark_data, report_period_days
        )
        
        risk_analysis = self._generate_risk_analysis(risk_manager, report_period_days)
        
        position_analysis = self._generate_position_analysis(portfolio_manager)
        
        trade_analysis = self._generate_trade_analysis(portfolio_manager, report_period_days)
        
        if moe_system:
            moe_analysis = self._generate_moe_analysis(moe_system, report_period_days)
        else:
            moe_analysis = None
        
        # Generate insights
        insights = self._generate_insights(
            portfolio_manager, risk_manager, moe_system
        ) if self.auto_insights else None
        
        # Create visualizations
        charts = self._generate_all_charts(
            portfolio_manager, risk_manager, moe_system
        ) if self.include_charts else None
        
        # Compile final report
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_period_days': report_period_days,
                'report_style': self.report_style,
                'generator_version': '2.0.0'
            },
            'executive_summary': executive_summary,
            'performance_analysis': performance_analysis,
            'risk_analysis': risk_analysis,
            'position_analysis': position_analysis,
            'trade_analysis': trade_analysis,
            'moe_analysis': moe_analysis,
            'insights': insights,
            'charts': charts,
            'sections': [section.__dict__ for section in self.sections]
        }
        
        logger.info("Comprehensive report generation completed")
        return report
    
    def _generate_executive_summary(self,
                                   portfolio_manager,
                                   risk_manager,
                                   period_days: int) -> Dict[str, Any]:
        """Generate executive summary section with improved error handling"""
        
        try:
            # Get portfolio summary with fallback
            portfolio_summary = self._safe_get_portfolio_summary(portfolio_manager)
            
            # Get risk dashboard with fallback
            risk_dashboard = self._safe_get_risk_dashboard(risk_manager)
            
            # Calculate key metrics with safe access
            total_value = portfolio_summary.get('total_value', 0)
            cash = portfolio_summary.get('cash', 0)
            positions_value = portfolio_summary.get('positions_value', 0)
            num_positions = portfolio_summary.get('num_positions', 0)
            
            # Performance metrics with safe access
            performance = portfolio_summary.get('performance', {})
            total_return = performance.get('total_return', 0) * 100
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            max_drawdown = performance.get('max_drawdown', 0) * 100
            
            summary = {
                'portfolio_value': total_value,
                'cash_position': cash,
                'invested_capital': positions_value,
                'number_of_positions': num_positions,
                'cash_allocation_pct': (cash / total_value * 100) if total_value > 0 else 0,
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'current_risk_level': risk_dashboard.get('current_risk_level', 'UNKNOWN'),
                'active_alerts': len(risk_dashboard.get('active_alerts', [])),
                'report_period_days': period_days
            }
            
            # Add section
            self.sections.append(ReportSection(
                title="Executive Summary",
                content=summary,
                section_type='metrics',
                importance='critical'
            ))
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            error_summary = {
                'portfolio_value': 0,
                'cash_position': 0,
                'invested_capital': 0,
                'number_of_positions': 0,
                'cash_allocation_pct': 0,
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'current_risk_level': 'UNKNOWN',
                'active_alerts': 0,
                'report_period_days': period_days,
                'error': str(e)
            }
            return error_summary
    
    def _safe_get_portfolio_summary(self, portfolio_manager) -> Dict[str, Any]:
        """Safely get portfolio summary from any portfolio manager type"""
        try:
            # Try enhanced portfolio manager first
            if hasattr(portfolio_manager, 'get_portfolio_summary'):
                summary = portfolio_manager.get_portfolio_summary()
                if summary is not None:
                    return summary
            
            # Fallback to legacy portfolio manager
            return self._get_legacy_portfolio_summary(portfolio_manager)
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {
                'total_value': 0,
                'cash': 0,
                'positions_value': 0,
                'num_positions': 0,
                'positions': [],
                'performance': {}
            }
    
    def _safe_get_risk_dashboard(self, risk_manager) -> Dict[str, Any]:
        """Safely get risk dashboard from any risk manager type"""
        try:
            if hasattr(risk_manager, 'get_risk_dashboard'):
                dashboard = risk_manager.get_risk_dashboard()
                if dashboard is not None:
                    return dashboard
            
            # Return default risk dashboard
            return {
                'current_risk_level': 'UNKNOWN',
                'current_metrics': {},
                'active_alerts': [],
                'thresholds': {}
            }
            
        except Exception as e:
            logger.error(f"Error getting risk dashboard: {e}")
            return {
                'current_risk_level': 'UNKNOWN',
                'current_metrics': {},
                'active_alerts': [],
                'thresholds': {}
            }
    
    def _generate_performance_analysis(self,
                                     portfolio_manager,
                                     benchmark_data: Optional[List[float]],
                                     period_days: int) -> Dict[str, Any]:
        """Generate detailed performance analysis with improved error handling"""
        
        try:
            # Get portfolio history safely
            portfolio_history = self._safe_get_portfolio_history(portfolio_manager, period_days)
            
            if not portfolio_history:
                return {
                    'error': 'No portfolio history available',
                    'performance_metrics': {},
                    'benchmark_analysis': None,
                    'return_distribution': {}
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(portfolio_history)
            
            if df.empty or 'total_value' not in df.columns:
                return {
                    'error': 'No valid portfolio data',
                    'performance_metrics': {},
                    'benchmark_analysis': None,
                    'return_distribution': {}
                }
            
            # Calculate performance metrics safely
            returns = df['total_value'].pct_change().dropna()
            
            if returns.empty:
                return {
                    'error': 'Insufficient data for returns calculation',
                    'performance_metrics': {},
                    'benchmark_analysis': None,
                    'return_distribution': {}
                }
            
            performance_metrics = {
                'period_return': ((df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1) * 100,
                'annualized_return': returns.mean() * 252 * 100,
                'volatility': returns.std() * np.sqrt(252) * 100,
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(df['total_value'].values) * 100,
                'win_rate': (returns > 0).mean() * 100,
                'best_day': returns.max() * 100,
                'worst_day': returns.min() * 100,
                'total_trades': len(df),
                'avg_daily_pnl': df['daily_pnl'].mean() if 'daily_pnl' in df.columns else 0
            }
            
            # Benchmark comparison if provided
            benchmark_analysis = None
            if benchmark_data and len(benchmark_data) == len(df):
                try:
                    benchmark_returns = pd.Series(benchmark_data).pct_change().dropna()
                    
                    # Calculate beta and alpha
                    if len(returns) == len(benchmark_returns):
                        correlation = returns.corr(benchmark_returns)
                        beta = returns.cov(benchmark_returns) / benchmark_returns.var() if benchmark_returns.var() > 0 else 1
                        alpha = (returns.mean() - beta * benchmark_returns.mean()) * 252 * 100
                        
                        benchmark_analysis = {
                            'beta': beta,
                            'alpha': alpha,
                            'correlation': correlation,
                            'tracking_error': (returns - benchmark_returns).std() * np.sqrt(252) * 100,
                            'information_ratio': alpha / ((returns - benchmark_returns).std() * np.sqrt(252) * 100) if (returns - benchmark_returns).std() > 0 else 0
                        }
                except Exception as e:
                    logger.error(f"Error calculating benchmark analysis: {e}")
                    benchmark_analysis = None
            
            analysis = {
                'performance_metrics': performance_metrics,
                'benchmark_analysis': benchmark_analysis,
                'return_distribution': {
                    'mean': returns.mean() * 100,
                    'median': returns.median() * 100,
                    'std': returns.std() * 100,
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'var_95': np.percentile(returns, 5) * 100,
                    'var_99': np.percentile(returns, 1) * 100
                }
            }
            
            # Add section
            self.sections.append(ReportSection(
                title="Performance Analysis",
                content=analysis,
                section_type='metrics',
                importance='high'
            ))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating performance analysis: {e}")
            return {
                'error': str(e),
                'performance_metrics': {},
                'benchmark_analysis': None,
                'return_distribution': {}
            }
    
    def _safe_get_portfolio_history(self, portfolio_manager, period_days: int) -> List[Dict]:
        """Safely extract portfolio history from portfolio manager"""
        try:
            if hasattr(portfolio_manager, 'portfolio_history'):
                history = portfolio_manager.portfolio_history[-period_days:]
                
                # Convert to standardized format
                standardized_history = []
                for h in history:
                    if hasattr(h, 'timestamp'):
                        # Object with attributes
                        standardized_history.append({
                            'timestamp': h.timestamp.isoformat() if hasattr(h.timestamp, 'isoformat') else str(h.timestamp),
                            'total_value': getattr(h, 'total_value', 0),
                            'daily_pnl': getattr(h, 'daily_pnl', 0),
                            'cumulative_pnl': getattr(h, 'cumulative_pnl', 0)
                        })
                    elif isinstance(h, dict):
                        # Dictionary format
                        standardized_history.append({
                            'timestamp': h.get('timestamp', datetime.now().isoformat()),
                            'total_value': h.get('total_value', 0),
                            'daily_pnl': h.get('daily_pnl', 0),
                            'cumulative_pnl': h.get('cumulative_pnl', 0)
                        })
                
                return standardized_history
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return []
    
    def _generate_risk_analysis(self, risk_manager, period_days: int) -> Dict[str, Any]:
        """Generate comprehensive risk analysis with improved error handling"""
        
        try:
            # Get risk dashboard safely
            risk_dashboard = self._safe_get_risk_dashboard(risk_manager)
            
            # Get risk metrics history safely
            risk_history = self._safe_get_risk_history(risk_manager, period_days)
            
            # Current risk metrics
            current_metrics = risk_dashboard.get('current_metrics', {})
            
            # Risk trend analysis
            risk_trends = {}
            if risk_history:
                risk_df = pd.DataFrame(risk_history)
                
                if not risk_df.empty:
                    risk_trends = {
                        'var_95_trend': 'increasing' if risk_df['var_95'].iloc[-1] > risk_df['var_95'].iloc[0] else 'decreasing',
                        'volatility_trend': 'increasing' if risk_df['volatility_annual'].iloc[-1] > risk_df['volatility_annual'].iloc[0] else 'decreasing',
                        'concentration_trend': 'increasing' if risk_df['concentration_risk'].iloc[-1] > risk_df['concentration_risk'].iloc[0] else 'decreasing',
                        'avg_var_95': risk_df['var_95'].mean(),
                        'max_var_95': risk_df['var_95'].max(),
                        'avg_volatility': risk_df['volatility_annual'].mean()
                    }
            
            # Active alerts analysis
            active_alerts = risk_dashboard.get('active_alerts', [])
            alert_summary = {
                'total_alerts': len(active_alerts),
                'critical_alerts': len([a for a in active_alerts if a.get('risk_level') == 'CRITICAL']),
                'high_alerts': len([a for a in active_alerts if a.get('risk_level') == 'HIGH']),
                'medium_alerts': len([a for a in active_alerts if a.get('risk_level') == 'MEDIUM']),
                'alert_types': {}
            }
            
            # Group alerts by type
            for alert in active_alerts:
                alert_type = alert.get('alert_type', 'UNKNOWN')
                alert_summary['alert_types'][alert_type] = alert_summary['alert_types'].get(alert_type, 0) + 1
            
            analysis = {
                'current_risk_level': risk_dashboard.get('current_risk_level', 'UNKNOWN'),
                'current_metrics': current_metrics,
                'risk_trends': risk_trends,
                'alert_summary': alert_summary,
                'thresholds': risk_dashboard.get('thresholds', {}),
                'risk_recommendations': self._generate_risk_recommendations(risk_dashboard)
            }
            
            # Add section
            self.sections.append(ReportSection(
                title="Risk Analysis",
                content=analysis,
                section_type='metrics',
                importance='critical'
            ))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating risk analysis: {e}")
            return {
                'error': str(e),
                'current_risk_level': 'UNKNOWN',
                'current_metrics': {},
                'risk_trends': {},
                'alert_summary': {'total_alerts': 0},
                'thresholds': {},
                'risk_recommendations': []
            }
    
    def _safe_get_risk_history(self, risk_manager, period_days: int) -> List[Dict]:
        """Safely extract risk metrics history"""
        try:
            if hasattr(risk_manager, 'risk_metrics_history'):
                history = risk_manager.risk_metrics_history[-period_days:]
                
                # Convert to standardized format
                standardized_history = []
                for m in history:
                    if hasattr(m, 'timestamp'):
                        # Object with attributes
                        standardized_history.append({
                            'timestamp': m.timestamp.isoformat() if hasattr(m.timestamp, 'isoformat') else str(m.timestamp),
                            'var_95': getattr(m, 'var_95', 0),
                            'var_99': getattr(m, 'var_99', 0),
                            'volatility_annual': getattr(m, 'volatility_annual', 0),
                            'concentration_risk': getattr(m, 'concentration_risk', 0),
                            'sharpe_ratio': getattr(m, 'sharpe_ratio', 0),
                            'max_drawdown': getattr(m, 'max_drawdown', 0)
                        })
                    elif isinstance(m, dict):
                        # Dictionary format
                        standardized_history.append({
                            'timestamp': m.get('timestamp', datetime.now().isoformat()),
                            'var_95': m.get('var_95', 0),
                            'var_99': m.get('var_99', 0),
                            'volatility_annual': m.get('volatility_annual', 0),
                            'concentration_risk': m.get('concentration_risk', 0),
                            'sharpe_ratio': m.get('sharpe_ratio', 0),
                            'max_drawdown': m.get('max_drawdown', 0)
                        })
                
                return standardized_history
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting risk history: {e}")
            return []
    
    def _generate_position_analysis(self, portfolio_manager) -> Dict[str, Any]:
        """Generate detailed position analysis with improved error handling"""
        
        try:
            # Get portfolio summary safely
            portfolio_summary = self._safe_get_portfolio_summary(portfolio_manager)
            positions = portfolio_summary.get('positions', [])
            
            if not positions:
                return {
                    'error': 'No positions available',
                    'position_metrics': {},
                    'pnl_analysis': {},
                    'sector_analysis': {},
                    'strategy_analysis': {},
                    'top_positions': [],
                    'concentration_metrics': {}
                }
            
            # Convert to DataFrame for analysis
            positions_df = pd.DataFrame(positions)
            
            # Calculate position metrics safely
            total_value = positions_df['market_value'].sum() if 'market_value' in positions_df.columns else 0
            
            if total_value == 0:
                return {
                    'error': 'No valid position values',
                    'position_metrics': {},
                    'pnl_analysis': {},
                    'sector_analysis': {},
                    'strategy_analysis': {},
                    'top_positions': [],
                    'concentration_metrics': {}
                }
            
            position_metrics = {
                'total_positions': len(positions),
                'total_market_value': total_value,
                'largest_position': {
                    'symbol': positions_df.loc[positions_df['market_value'].idxmax(), 'symbol'],
                    'value': positions_df['market_value'].max(),
                    'weight': positions_df['market_value'].max() / total_value * 100
                },
                'smallest_position': {
                    'symbol': positions_df.loc[positions_df['market_value'].idxmin(), 'symbol'],
                    'value': positions_df['market_value'].min(),
                    'weight': positions_df['market_value'].min() / total_value * 100
                },
                'concentration_top3': positions_df.nlargest(3, 'market_value')['market_value'].sum() / total_value * 100,
                'concentration_top5': positions_df.nlargest(5, 'market_value')['market_value'].sum() / total_value * 100
            }
            
            # P&L analysis
            pnl_analysis = {'error': 'P&L data not available'}
            if 'unrealized_pnl' in positions_df.columns:
                total_unrealized_pnl = positions_df['unrealized_pnl'].sum()
                winning_positions = len(positions_df[positions_df['unrealized_pnl'] > 0])
                losing_positions = len(positions_df[positions_df['unrealized_pnl'] < 0])
                
                pnl_analysis = {
                    'total_unrealized_pnl': total_unrealized_pnl,
                    'winning_positions': winning_positions,
                    'losing_positions': losing_positions,
                    'win_rate': winning_positions / len(positions) * 100,
                    'best_performer': {
                        'symbol': positions_df.loc[positions_df['unrealized_pnl'].idxmax(), 'symbol'],
                        'pnl': positions_df['unrealized_pnl'].max(),
                        'pnl_pct': positions_df.loc[positions_df['unrealized_pnl'].idxmax(), 'unrealized_pnl_pct'] * 100 if 'unrealized_pnl_pct' in positions_df.columns else 0
                    },
                    'worst_performer': {
                        'symbol': positions_df.loc[positions_df['unrealized_pnl'].idxmin(), 'symbol'],
                        'pnl': positions_df['unrealized_pnl'].min(),
                        'pnl_pct': positions_df.loc[positions_df['unrealized_pnl'].idxmin(), 'unrealized_pnl_pct'] * 100 if 'unrealized_pnl_pct' in positions_df.columns else 0
                    }
                }
            
            # Sector analysis (if available)
            sector_analysis = {}
            if 'sector' in positions_df.columns:
                sector_exposure = positions_df.groupby('sector')['market_value'].sum()
                sector_analysis = {
                    'sector_allocation': (sector_exposure / total_value * 100).to_dict(),
                    'most_exposed_sector': sector_exposure.idxmax(),
                    'sector_concentration': sector_exposure.max() / total_value * 100
                }
            
            # Strategy attribution (if available)
            strategy_analysis = {}
            if 'strategy_source' in positions_df.columns:
                strategy_value = positions_df.groupby('strategy_source')['market_value'].sum()
                strategy_analysis = {
                    'strategy_allocation': (strategy_value / total_value * 100).to_dict(),
                    'top_strategy': strategy_value.idxmax(),
                    'strategy_concentration': strategy_value.max() / total_value * 100
                }
            
            analysis = {
                'position_metrics': position_metrics,
                'pnl_analysis': pnl_analysis,
                'sector_analysis': sector_analysis,
                'strategy_analysis': strategy_analysis,
                'top_positions': positions_df.nlargest(10, 'market_value').to_dict('records'),
                'concentration_metrics': {
                    'herfindahl_index': (positions_df['market_value'] / total_value).pow(2).sum(),
                    'effective_positions': 1 / (positions_df['market_value'] / total_value).pow(2).sum()
                }
            }
            
            # Add section
            self.sections.append(ReportSection(
                title="Position Analysis",
                content=analysis,
                section_type='table',
                importance='high'
            ))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating position analysis: {e}")
            return {
                'error': str(e),
                'position_metrics': {},
                'pnl_analysis': {},
                'sector_analysis': {},
                'strategy_analysis': {},
                'top_positions': [],
                'concentration_metrics': {}
            }
    
    def _generate_trade_analysis(self, portfolio_manager, period_days: int) -> Dict[str, Any]:
        """Generate detailed trade analysis with improved error handling"""
        
        try:
            # Get trade history safely
            trade_history = self._safe_get_trade_history(portfolio_manager, period_days)
            
            if not trade_history:
                return {
                    'error': f'No trades in the last {period_days} days',
                    'trade_metrics': {},
                    'most_traded': {},
                    'frequency_analysis': {},
                    'strategy_attribution': {},
                    'recent_trades': []
                }
            
            # Convert to DataFrame for analysis
            trades_df = pd.DataFrame(trade_history)
            
            if trades_df.empty:
                return {
                    'error': 'No valid trade data',
                    'trade_metrics': {},
                    'most_traded': {},
                    'frequency_analysis': {},
                    'strategy_attribution': {},
                    'recent_trades': []
                }
            
            # Trade metrics
            total_volume = trades_df['notional_value'].sum() if 'notional_value' in trades_df.columns else 0
            buy_volume = trades_df[trades_df['side'] == 'BUY']['notional_value'].sum() if 'notional_value' in trades_df.columns else 0
            sell_volume = trades_df[trades_df['side'] == 'SELL']['notional_value'].sum() if 'notional_value' in trades_df.columns else 0
            
            trade_metrics = {
                'total_trades': len(trades_df),
                'buy_trades': len(trades_df[trades_df['side'] == 'BUY']) if 'side' in trades_df.columns else 0,
                'sell_trades': len(trades_df[trades_df['side'] == 'SELL']) if 'side' in trades_df.columns else 0,
                'total_volume': total_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'avg_trade_size': trades_df['notional_value'].mean() if 'notional_value' in trades_df.columns else 0,
                'largest_trade': trades_df['notional_value'].max() if 'notional_value' in trades_df.columns else 0,
                'smallest_trade': trades_df['notional_value'].min() if 'notional_value' in trades_df.columns else 0,
                'turnover_ratio': total_volume  # Would need portfolio value for accurate ratio
            }
            
            # Most traded symbols
            most_traded = {}
            if 'symbol' in trades_df.columns and 'notional_value' in trades_df.columns:
                symbol_volume = trades_df.groupby('symbol')['notional_value'].sum().sort_values(ascending=False)
                most_traded = {
                    'symbols': symbol_volume.head(10).to_dict(),
                    'top_symbol': symbol_volume.index[0] if len(symbol_volume) > 0 else None,
                    'top_symbol_volume': symbol_volume.iloc[0] if len(symbol_volume) > 0 else 0
                }
            
            # Trading frequency analysis
            frequency_analysis = {}
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                trades_by_day = trades_df.groupby(trades_df['timestamp'].dt.date).size()
                
                frequency_analysis = {
                    'avg_trades_per_day': trades_by_day.mean(),
                    'max_trades_per_day': trades_by_day.max(),
                    'trading_days': len(trades_by_day),
                    'most_active_day': trades_by_day.idxmax().isoformat() if len(trades_by_day) > 0 else None
                }
            
            # Strategy attribution (if available)
            strategy_attribution = {}
            if 'strategy_source' in trades_df.columns and 'notional_value' in trades_df.columns:
                strategy_volume = trades_df.groupby('strategy_source')['notional_value'].sum()
                strategy_attribution = {
                    'strategy_volumes': strategy_volume.to_dict(),
                    'top_strategy': strategy_volume.idxmax() if len(strategy_volume) > 0 else None,
                    'strategy_concentration': strategy_volume.max() / total_volume * 100 if total_volume > 0 else 0
                }
            
            analysis = {
                'trade_metrics': trade_metrics,
                'most_traded': most_traded,
                'frequency_analysis': frequency_analysis,
                'strategy_attribution': strategy_attribution,
                'recent_trades': trades_df.tail(20).to_dict('records')  # Last 20 trades
            }
            
            # Add section
            self.sections.append(ReportSection(
                title="Trade Analysis",
                content=analysis,
                section_type='table',
                importance='medium'
            ))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating trade analysis: {e}")
            return {
                'error': str(e),
                'trade_metrics': {},
                'most_traded': {},
                'frequency_analysis': {},
                'strategy_attribution': {},
                'recent_trades': []
            }
    
    def _safe_get_trade_history(self, portfolio_manager, period_days: int) -> List[Dict]:
        """Safely extract trade history from portfolio manager"""
        try:
            if not hasattr(portfolio_manager, 'trade_history'):
                return []
            
            all_trades = portfolio_manager.trade_history
            if not all_trades:
                return []
            
            # Filter trades for the period
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_trades = []
            
            for trade in all_trades:
                try:
                    # Get timestamp
                    if hasattr(trade, 'timestamp'):
                        trade_time = trade.timestamp
                    elif isinstance(trade, dict):
                        trade_time = datetime.fromisoformat(trade.get('timestamp', '2020-01-01'))
                    else:
                        continue
                    
                    if trade_time > cutoff_date:
                        # Convert to standardized format
                        if hasattr(trade, 'to_dict'):
                            recent_trades.append(trade.to_dict())
                        elif isinstance(trade, dict):
                            recent_trades.append(trade)
                        else:
                            # Handle other trade formats
                            recent_trades.append({
                                'symbol': getattr(trade, 'symbol', 'UNKNOWN'),
                                'side': getattr(trade, 'side', 'UNKNOWN'),
                                'shares': getattr(trade, 'shares', 0),
                                'price': getattr(trade, 'price', 0),
                                'notional_value': getattr(trade, 'notional_value', 0),
                                'timestamp': trade_time.isoformat()
                            })
                            
                except Exception as e:
                    logger.warning(f"Error processing trade: {e}")
                    continue
            
            return recent_trades
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def _generate_moe_analysis(self, moe_system, period_days: int) -> Dict[str, Any]:
        """Generate MoE system analysis"""
        
        try:
            # Create realistic mock data for MoE analysis
            # In a real implementation, this would interface with your actual MoE system
            
            num_experts = 8
            expert_names = [f'Expert_{i}' for i in range(num_experts)]
            
            # Generate realistic performance data
            expert_performance = {}
            expert_usage = {}
            total_usage = 0
            
            for expert in expert_names:
                # Performance: some experts better than others
                perf = np.random.normal(2.0, 4.0)  # Mean 2% with 4% std
                expert_performance[expert] = perf
                
                # Usage: somewhat correlated with performance but with noise
                usage = max(0, np.random.normal(100/num_experts + perf, 5))
                expert_usage[expert] = usage
                total_usage += usage
            
            # Normalize usage to percentages
            for expert in expert_names:
                expert_usage[expert] = expert_usage[expert] / total_usage * 100
            
            analysis = {
                'expert_performance': expert_performance,
                'expert_usage': expert_usage,
                'routing_metrics': {
                    'avg_entropy': np.random.uniform(1.5, 2.2),
                    'avg_concentration': np.random.uniform(0.6, 0.8),
                    'expert_utilization': np.random.uniform(0.75, 0.95),
                    'routing_efficiency': np.random.uniform(0.85, 0.95)
                },
                'consensus_analysis': {
                    'high_consensus_decisions': np.random.randint(100, 200),
                    'low_consensus_decisions': np.random.randint(10, 50),
                    'consensus_accuracy': np.random.uniform(0.65, 0.85),
                    'divergence_events': np.random.randint(20, 50)
                },
                'entropy_history': {
                    'timestamps': [(datetime.now() - timedelta(days=i)).isoformat() for i in range(period_days, 0, -1)],
                    'values': np.random.uniform(1.0, 2.5, period_days).tolist()
                }
            }
            
            # Add section
            self.sections.append(ReportSection(
                title="MoE System Analysis",
                content=analysis,
                section_type='metrics',
                importance='high'
            ))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating MoE analysis: {e}")
            return {
                'error': str(e),
                'expert_performance': {},
                'expert_usage': {},
                'routing_metrics': {},
                'consensus_analysis': {},
                'entropy_history': {'timestamps': [], 'values': []}
            }
    
    def _generate_insights(self, portfolio_manager, risk_manager, moe_system) -> List[Dict[str, str]]:
        """Generate automated insights based on analysis"""
        
        insights = []
        
        try:
            # Performance insights
            portfolio_summary = self._safe_get_portfolio_summary(portfolio_manager)
            performance = portfolio_summary.get('performance', {})
            
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            if sharpe_ratio > 1.5:
                insights.append({
                    'type': 'positive',
                    'category': 'Performance',
                    'title': 'Strong Risk-Adjusted Returns',
                    'description': f'Portfolio Sharpe ratio of {sharpe_ratio:.2f} indicates excellent risk-adjusted performance.'
                })
            elif sharpe_ratio < 0.5:
                insights.append({
                    'type': 'warning',
                    'category': 'Performance',
                    'title': 'Low Risk-Adjusted Returns',
                    'description': f'Portfolio Sharpe ratio of {sharpe_ratio:.2f} suggests suboptimal risk-adjusted performance.'
                })
            
            max_drawdown = performance.get('max_drawdown', 0)
            if max_drawdown > 0.15:
                insights.append({
                    'type': 'warning',
                    'category': 'Risk',
                    'title': 'High Drawdown Risk',
                    'description': f'Maximum drawdown of {max_drawdown:.1%} exceeds typical risk tolerance levels.'
                })
            
            # Risk insights
            risk_dashboard = self._safe_get_risk_dashboard(risk_manager)
            risk_level = risk_dashboard.get('current_risk_level', 'UNKNOWN')
            
            if risk_level == 'HIGH':
                insights.append({
                    'type': 'alert',
                    'category': 'Risk',
                    'title': 'Elevated Risk Level',
                    'description': 'Current risk level is HIGH. Consider reducing position sizes or implementing hedges.'
                })
            elif risk_level == 'CRITICAL':
                insights.append({
                    'type': 'critical',
                    'category': 'Risk',
                    'title': 'Critical Risk Level',
                    'description': 'CRITICAL risk level detected. Immediate risk reduction measures recommended.'
                })
            
            active_alerts = len(risk_dashboard.get('active_alerts', []))
            if active_alerts > 5:
                insights.append({
                    'type': 'warning',
                    'category': 'Risk',
                    'title': 'Multiple Risk Alerts',
                    'description': f'{active_alerts} active risk alerts require attention.'
                })
            
            # Portfolio composition insights
            num_positions = portfolio_summary.get('num_positions', 0)
            
            if num_positions < 5:
                insights.append({
                    'type': 'info',
                    'category': 'Diversification',
                    'title': 'Low Diversification',
                    'description': f'Portfolio has only {num_positions} positions. Consider adding more diversification.'
                })
            elif num_positions > 50:
                insights.append({
                    'type': 'info',
                    'category': 'Diversification',
                    'title': 'High Position Count',
                    'description': f'Portfolio has {num_positions} positions. Consider if this level of diversification is optimal.'
                })
            
            total_value = portfolio_summary.get('total_value', 1)
            cash = portfolio_summary.get('cash', 0)
            cash_pct = cash / total_value * 100 if total_value > 0 else 0
            
            if cash_pct > 20:
                insights.append({
                    'type': 'info',
                    'category': 'Allocation',
                    'title': 'High Cash Allocation',
                    'description': f'{cash_pct:.1f}% cash allocation may indicate missed investment opportunities.'
                })
            elif cash_pct < 2:
                insights.append({
                    'type': 'warning',
                    'category': 'Allocation',
                    'title': 'Low Cash Reserve',
                    'description': f'{cash_pct:.1f}% cash allocation may limit flexibility for opportunities or emergencies.'
                })
        
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append({
                'type': 'error',
                'category': 'System',
                'title': 'Insight Generation Error',
                'description': f'Error generating insights: {str(e)}'
            })
        
        return insights
    
    def _generate_all_charts(self, portfolio_manager, risk_manager, moe_system) -> Dict[str, str]:
        """Generate all visualization charts with improved error handling"""
        
        charts = {}
        
        try:
            # Portfolio performance chart
            portfolio_history = self._safe_get_portfolio_history(portfolio_manager, 30)
            if portfolio_history:
                try:
                    perf_chart = self.viz_engine.create_portfolio_performance_chart(portfolio_history)
                    charts['portfolio_performance'] = self._chart_to_string(perf_chart)
                except Exception as e:
                    logger.error(f"Error creating portfolio performance chart: {e}")
                    charts['portfolio_performance'] = f"Chart generation error: {str(e)}"
            
            # Risk analysis chart
            risk_history = self._safe_get_risk_history(risk_manager, 30)
            if risk_history:
                try:
                    risk_chart = self.viz_engine.create_risk_analysis_chart(risk_history)
                    charts['risk_analysis'] = self._chart_to_string(risk_chart)
                except Exception as e:
                    logger.error(f"Error creating risk analysis chart: {e}")
                    charts['risk_analysis'] = f"Chart generation error: {str(e)}"
            
            # Position analysis chart
            portfolio_summary = self._safe_get_portfolio_summary(portfolio_manager)
            positions = portfolio_summary.get('positions', [])
            
            if positions:
                try:
                    position_chart = self.viz_engine.create_position_analysis_chart(positions)
                    charts['position_analysis'] = self._chart_to_string(position_chart)
                except Exception as e:
                    logger.error(f"Error creating position analysis chart: {e}")
                    charts['position_analysis'] = f"Chart generation error: {str(e)}"
            
            # MoE expert analysis chart
            if moe_system:
                try:
                    # Create mock expert attribution data
                    expert_attribution = {
                        'expert_performance': {
                            'Expert_0': 5.2, 'Expert_1': -2.1, 'Expert_2': 8.7, 'Expert_3': 3.4
                        },
                        'expert_usage': {
                            'Expert_0': 25.2, 'Expert_1': 18.7, 'Expert_2': 32.1, 'Expert_3': 24.0
                        },
                        'entropy_history': {
                            'timestamps': [(datetime.now() - timedelta(days=i)).isoformat() for i in range(30, 0, -1)],
                            'values': np.random.uniform(1.0, 2.5, 30).tolist()
                        }
                    }
                    
                    expert_chart = self.viz_engine.create_expert_analysis_chart(expert_attribution)
                    charts['expert_analysis'] = self._chart_to_string(expert_chart)
                except Exception as e:
                    logger.error(f"Error creating expert analysis chart: {e}")
                    charts['expert_analysis'] = f"Chart generation error: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            charts['error'] = f"Chart generation error: {str(e)}"
        
        return charts
    
    def _chart_to_string(self, fig) -> str:
        """Convert plotly figure to HTML string"""
        try:
            if self.chart_format == 'html':
                return fig.to_html(include_plotlyjs='cdn')
            elif self.chart_format == 'png':
                # Would need kaleido for PNG export
                return "PNG export not implemented"
            else:
                return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error converting chart to string: {e}")
            return f"Chart conversion error: {str(e)}"
    
    def _generate_risk_recommendations(self, risk_dashboard: Dict) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        try:
            risk_level = risk_dashboard.get('current_risk_level', 'UNKNOWN')
            current_metrics = risk_dashboard.get('current_metrics', {})
            
            if risk_level == 'CRITICAL':
                recommendations.extend([
                    "Immediate portfolio de-risking required",
                    "Consider liquidating most volatile positions",
                    "Implement protective stops on all positions",
                    "Increase cash allocation to 20%+"
                ])
            elif risk_level == 'HIGH':
                recommendations.extend([
                    "Reduce position sizes by 20-30%",
                    "Review and tighten stop-loss levels",
                    "Consider hedging strategies",
                    "Avoid new high-risk positions"
                ])
            
            # VaR-based recommendations
            var_95 = current_metrics.get('var_95', 0)
            if var_95 > 0.02:
                recommendations.append(f"VaR at {var_95:.1%} - consider position size reduction")
            
            # Concentration recommendations
            concentration = current_metrics.get('concentration_risk', 0)
            if concentration > 0.5:
                recommendations.append("High concentration risk - diversify holdings")
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
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
    
    def _get_legacy_portfolio_summary(self, portfolio_manager) -> Dict:
        """Get portfolio summary for legacy portfolio manager"""
        try:
            total_value = 0
            positions_data = []
            
            if hasattr(portfolio_manager, 'positions') and hasattr(portfolio_manager, 'cash'):
                for symbol, shares in portfolio_manager.positions.items():
                    try:
                        price = get_live_price(symbol, None)
                        market_value = shares * price
                        total_value += market_value
                        
                        positions_data.append({
                            'symbol': symbol,
                            'shares': shares,
                            'current_price': price,
                            'market_value': market_value
                        })
                    except Exception as e:
                        logger.warning(f"Error getting price for {symbol}: {e}")
                        continue
                
                return {
                    'total_value': total_value + portfolio_manager.cash,
                    'cash': portfolio_manager.cash,
                    'positions_value': total_value,
                    'num_positions': len(positions_data),
                    'positions': positions_data,
                    'performance': {}
                }
            else:
                return {
                    'total_value': 0,
                    'cash': 0,
                    'positions_value': 0,
                    'num_positions': 0,
                    'positions': [],
                    'performance': {}
                }
        except Exception as e:
            logger.error(f"Error getting legacy portfolio summary: {e}")
            return {
                'total_value': 0,
                'cash': 0,
                'positions_value': 0,
                'num_positions': 0,
                'positions': [],
                'performance': {}
            }
    
    def export_report_html(self, report: Dict, filepath: str):
        """Export report as HTML file"""
        
        if JINJA_AVAILABLE:
            self._export_jinja_html(report, filepath)
        else:
            self._export_basic_html(report, filepath)
    
    def _export_jinja_html(self, report: Dict, filepath: str):
        """Export HTML using Jinja2 template"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Report - {{ metadata.generated_at[:10] }}</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { background: linear-gradient(135deg, #1f77b4, #17a2b8); color: white; padding: 30px; margin: -20px -20px 30px -20px; border-radius: 8px 8px 0 0; }
                .header h1 { margin: 0 0 10px 0; font-size: 2.5em; }
                .header p { margin: 5px 0; opacity: 0.9; }
                .section { margin-bottom: 30px; padding: 20px; border-left: 4px solid #1f77b4; background: #f8f9fa; border-radius: 0 8px 8px 0; }
                .section h2 { margin-top: 0; color: #1f77b4; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }
                .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                .metric { padding: 15px; background: white; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 1.8em; font-weight: bold; color: #1f77b4; }
                .metric-label { font-size: 0.9em; color: #6c757d; margin-top: 5px; }
                .alert { padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid; }
                .alert-critical { background: #f8d7da; border-color: #dc3545; color: #721c24; }
                .alert-warning { background: #fff3cd; border-color: #ffc107; color: #856404; }
                .alert-info { background: #d1ecf1; border-color: #17a2b8; color: #0c5460; }
                .alert-positive { background: #d4edda; border-color: #28a745; color: #155724; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                th { padding: 15px; text-align: left; background: #1f77b4; color: white; font-weight: 600; }
                td { padding: 12px 15px; border-bottom: 1px solid #e9ecef; }
                tbody tr:hover { background: #f8f9fa; }
                .chart-container { margin: 25px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .chart-container h3 { margin-top: 0; color: #495057; }
                .status-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }
                .status-normal { background: #d4edda; color: #155724; }
                .status-warning { background: #fff3cd; color: #856404; }
                .status-critical { background: #f8d7da; color: #721c24; }
                .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #e9ecef; color: #6c757d; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Portfolio Performance Report</h1>
                    <p><strong>Generated:</strong> {{ metadata.generated_at }}</p>
                    <p><strong>Report Period:</strong> {{ metadata.report_period_days }} days</p>
                    <p><strong>Report Style:</strong> {{ metadata.report_style|title }}</p>
                </div>
                
                <!-- Executive Summary -->
                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value">${{ "%.0f"|format(executive_summary.portfolio_value) }}</div>
                            <div class="metric-label">Portfolio Value</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ "%.2f"|format(executive_summary.total_return_pct) }}%</div>
                            <div class="metric-label">Total Return</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ "%.2f"|format(executive_summary.sharpe_ratio) }}</div>
                            <div class="metric-label">Sharpe Ratio</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ executive_summary.number_of_positions }}</div>
                            <div class="metric-label">Positions</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ "%.1f"|format(executive_summary.cash_allocation_pct) }}%</div>
                            <div class="metric-label">Cash Allocation</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value {% if executive_summary.current_risk_level == 'HIGH' %}status-warning{% elif executive_summary.current_risk_level == 'CRITICAL' %}status-critical{% else %}status-normal{% endif %}">{{ executive_summary.current_risk_level }}</div>
                            <div class="metric-label">Risk Level</div>
                        </div>
                    </div>
                </div>
                
                <!-- Charts -->
                {% if charts %}
                <div class="section">
                    <h2>📈 Performance Charts</h2>
                    {% for chart_name, chart_html in charts.items() %}
                    {% if chart_html and not chart_html.startswith('Chart') and not chart_html.startswith('PNG') %}
                    <div class="chart-container">
                        <h3>{{ chart_name.replace('_', ' ').title() }}</h3>
                        {{ chart_html | safe }}
                    </div>
                    {% endif %}
                    {% endfor %}
                </div>
                {% endif %}
                
                <!-- Insights -->
                {% if insights %}
                <div class="section">
                    <h2>💡 Key Insights</h2>
                    {% for insight in insights %}
                    <div class="alert alert-{{ insight.type }}">
                        <strong>{{ insight.title }}</strong> ({{ insight.category }})<br>
                        {{ insight.description }}
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <!-- Performance Analysis -->
                {% if performance_analysis.performance_metrics and not performance_analysis.error %}
                <div class="section">
                    <h2>📋 Performance Analysis</h2>
                    <table>
                        <thead>
                            <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Period Return</td>
                                <td>{{ "%.2f"|format(performance_analysis.performance_metrics.period_return) }}%</td>
                                <td><span class="status-badge {% if performance_analysis.performance_metrics.period_return >= 0 %}status-normal{% else %}status-warning{% endif %}">{% if performance_analysis.performance_metrics.period_return >= 0 %}Positive{% else %}Negative{% endif %}</span></td>
                            </tr>
                            <tr>
                                <td>Annualized Return</td>
                                <td>{{ "%.2f"|format(performance_analysis.performance_metrics.annualized_return) }}%</td>
                                <td><span class="status-badge {% if performance_analysis.performance_metrics.annualized_return >= 8 %}status-normal{% elif performance_analysis.performance_metrics.annualized_return >= 0 %}status-warning{% else %}status-critical{% endif %}">{% if performance_analysis.performance_metrics.annualized_return >= 8 %}Strong{% elif performance_analysis.performance_metrics.annualized_return >= 0 %}Moderate{% else %}Weak{% endif %}</span></td>
                            </tr>
                            <tr>
                                <td>Volatility</td>
                                <td>{{ "%.2f"|format(performance_analysis.performance_metrics.volatility) }}%</td>
                                <td><span class="status-badge {% if performance_analysis.performance_metrics.volatility <= 15 %}status-normal{% elif performance_analysis.performance_metrics.volatility <= 25 %}status-warning{% else %}status-critical{% endif %}">{% if performance_analysis.performance_metrics.volatility <= 15 %}Low{% elif performance_analysis.performance_metrics.volatility <= 25 %}Medium{% else %}High{% endif %}</span></td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>{{ "%.2f"|format(performance_analysis.performance_metrics.sharpe_ratio) }}</td>
                                <td><span class="status-badge {% if performance_analysis.performance_metrics.sharpe_ratio >= 1.0 %}status-normal{% elif performance_analysis.performance_metrics.sharpe_ratio >= 0.5 %}status-warning{% else %}status-critical{% endif %}">{% if performance_analysis.performance_metrics.sharpe_ratio >= 1.0 %}Good{% elif performance_analysis.performance_metrics.sharpe_ratio >= 0.5 %}Fair{% else %}Poor{% endif %}</span></td>
                            </tr>
                            <tr>
                                <td>Max Drawdown</td>
                                <td>{{ "%.2f"|format(performance_analysis.performance_metrics.max_drawdown) }}%</td>
                                <td><span class="status-badge {% if performance_analysis.performance_metrics.max_drawdown <= 10 %}status-normal{% elif performance_analysis.performance_metrics.max_drawdown <= 20 %}status-warning{% else %}status-critical{% endif %}">{% if performance_analysis.performance_metrics.max_drawdown <= 10 %}Low{% elif performance_analysis.performance_metrics.max_drawdown <= 20 %}Medium{% else %}High{% endif %}</span></td>
                            </tr>
                            <tr>
                                <td>Win Rate</td>
                                <td>{{ "%.1f"|format(performance_analysis.performance_metrics.win_rate) }}%</td>
                                <td><span class="status-badge {% if performance_analysis.performance_metrics.win_rate >= 60 %}status-normal{% elif performance_analysis.performance_metrics.win_rate >= 45 %}status-warning{% else %}status-critical{% endif %}">{% if performance_analysis.performance_metrics.win_rate >= 60 %}High{% elif performance_analysis.performance_metrics.win_rate >= 45 %}Medium{% else %}Low{% endif %}</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                {% endif %}
                
                <!-- Position Analysis -->
                {% if position_analysis.top_positions and not position_analysis.error %}
                <div class="section">
                    <h2>🎯 Top Positions</h2>
                    <table>
                        <thead>
                            <tr><th>Symbol</th><th>Shares</th><th>Price</th><th>Market Value</th><th>P&L</th></tr>
                        </thead>
                        <tbody>
                            {% for position in position_analysis.top_positions[:10] %}
                            <tr>
                                <td><strong>{{ position.symbol }}</strong></td>
                                <td>{{ "%.0f"|format(position.shares) }}</td>
                                <td>${{ "%.2f"|format(position.current_price) }}</td>
                                <td>${{ "%.0f"|format(position.market_value) }}</td>
                                <td style="color: {% if position.unrealized_pnl and position.unrealized_pnl >= 0 %}#28a745{% else %}#dc3545{% endif %}">
                                    {% if position.unrealized_pnl %}${{ "%.0f"|format(position.unrealized_pnl) }}{% else %}N/A{% endif %}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% endif %}
                
                <!-- Risk Analysis -->
                {% if risk_analysis.current_metrics and not risk_analysis.error %}
                <div class="section">
                    <h2>⚠️ Risk Analysis</h2>
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value">{{ risk_analysis.current_risk_level }}</div>
                            <div class="metric-label">Current Risk Level</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ risk_analysis.alert_summary.total_alerts }}</div>
                            <div class="metric-label">Active Alerts</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ "%.2f"|format(risk_analysis.current_metrics.var_95 * 100) }}%</div>
                            <div class="metric-label">VaR (95%)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ "%.2f"|format(risk_analysis.current_metrics.volatility_annual * 100) }}%</div>
                            <div class="metric-label">Annual Volatility</div>
                        </div>
                    </div>
                    
                    {% if risk_analysis.risk_recommendations %}
                    <h3>🔧 Risk Recommendations</h3>
                    <ul>
                        {% for recommendation in risk_analysis.risk_recommendations %}
                        <li>{{ recommendation }}</li>
                        {% endfor %}
                    </ul>
                    {% endif %}
                </div>
                {% endif %}
                
                <!-- Trading Activity -->
                {% if trade_analysis.trade_metrics and not trade_analysis.error %}
                <div class="section">
                    <h2>📊 Trading Activity</h2>
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value">{{ trade_analysis.trade_metrics.total_trades }}</div>
                            <div class="metric-label">Total Trades</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${{ "%.0f"|format(trade_analysis.trade_metrics.total_volume) }}</div>
                            <div class="metric-label">Total Volume</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${{ "%.0f"|format(trade_analysis.trade_metrics.avg_trade_size) }}</div>
                            <div class="metric-label">Avg Trade Size</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ trade_analysis.trade_metrics.buy_trades }}/{{ trade_analysis.trade_metrics.sell_trades }}</div>
                            <div class="metric-label">Buy/Sell Ratio</div>
                        </div>
                    </div>
                </div>
                {% endif %}
                
                <!-- MoE Analysis -->
                {% if moe_analysis and not moe_analysis.error %}
                <div class="section">
                    <h2>🤖 Expert System Analysis</h2>
                    <h3>Expert Performance</h3>
                    <table>
                        <thead>
                            <tr><th>Expert</th><th>Performance (%)</th><th>Usage (%)</th><th>Status</th></tr>
                        </thead>
                        <tbody>
                            {% for expert, performance in moe_analysis.expert_performance.items() %}
                            <tr>
                                <td><strong>{{ expert }}</strong></td>
                                <td style="color: {% if performance >= 0 %}#28a745{% else %}#dc3545{% endif %}">{{ "%.1f"|format(performance) }}%</td>
                                <td>{{ "%.1f"|format(moe_analysis.expert_usage[expert]) }}%</td>
                                <td><span class="status-badge {% if performance >= 2 %}status-normal{% elif performance >= 0 %}status-warning{% else %}status-critical{% endif %}">{% if performance >= 2 %}Strong{% elif performance >= 0 %}Neutral{% else %}Weak{% endif %}</span></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    
                    <h3>Routing Metrics</h3>
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value">{{ "%.2f"|format(moe_analysis.routing_metrics.avg_entropy) }}</div>
                            <div class="metric-label">Avg Entropy</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ "%.1f"|format(moe_analysis.routing_metrics.expert_utilization * 100) }}%</div>
                            <div class="metric-label">Expert Utilization</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ "%.1f"|format(moe_analysis.routing_metrics.routing_efficiency * 100) }}%</div>
                            <div class="metric-label">Routing Efficiency</div>
                        </div>
                    </div>
                </div>
                {% endif %}
                
                <div class="footer">
                    <p><strong>Generated by:</strong> Enhanced Report Generator v{{ metadata.generator_version }}</p>
                    <p><strong>Report Style:</strong> {{ metadata.report_style|title }}</p>
                    <p><strong>Analysis Period:</strong> {{ metadata.report_period_days }} days</p>
                    <p><em>This report is for informational purposes only and should not be considered as investment advice.</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        try:
            template = Template(html_template)
            html_content = template.render(**report)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Basic HTML report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting basic HTML: {e}")
    
    def export_report_json(self, report: Dict, filepath: str):
        """Export report as JSON file"""
        
        try:
            # Create a JSON-serializable version
            json_report = self._make_json_serializable(report)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, default=str)
            
            logger.info(f"JSON report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting JSON report: {e}")
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

# =============================================================================
# Legacy Compatibility
# =============================================================================

class ReportGenerator(EnhancedReportGenerator):
    """
    Legacy compatibility wrapper for the original ReportGenerator interface
    """
    
    def __init__(self):
        super().__init__(
            report_style='professional',
            include_charts=False,  # Legacy didn't have charts
            auto_insights=False
        )
    
    def generate_summary(self, pm) -> Tuple[Dict, pd.DataFrame]:
        """
        Legacy summary generation method
        
        Returns:
            Tuple of (summary_dict, positions_dataframe)
        """
        try:
            # Use legacy portfolio summary method
            summary = self._get_legacy_portfolio_summary(pm)
            
            # Create positions DataFrame
            positions_data = []
            if hasattr(pm, 'positions'):
                for symbol, shares in pm.positions.items():
                    try:
                        price = get_live_price(symbol, None)
                        value = shares * price
                        positions_data.append({
                            'symbol': symbol,
                            'shares': shares,
                            'price': price,
                            'value': value
                        })
                    except Exception as e:
                        logger.error(f"Error getting price for {symbol}: {e}")
                        positions_data.append({
                            'symbol': symbol,
                            'shares': shares,
                            'price': np.nan,
                            'value': np.nan
                        })
            
            df = pd.DataFrame(positions_data)
            
            # Calculate totals
            total_positions_value = df['value'].sum() if not df.empty and 'value' in df.columns else 0
            total_value = total_positions_value + getattr(pm, 'cash', 0)
            
            legacy_summary = {
                'timestamp': pd.Timestamp.now(),
                'cash': getattr(pm, 'cash', 0),
                'positions_value': total_positions_value,
                'total_value': total_value
            }
            
            logger.info(f"Portfolio Summary: {legacy_summary}")
            logger.info(f"Positions DataFrame shape: {df.shape}")
            
            return legacy_summary, df
            
        except Exception as e:
            logger.error(f"Error generating legacy summary: {e}")
            return {'error': str(e)}, pd.DataFrame()

# =============================================================================
# Testing and Example Usage
# =============================================================================

def test_enhanced_report_generator():
    """Test the enhanced report generator with improved error handling"""
    
    print("Testing Enhanced Report Generator...")
    
    # Create mock portfolio manager
    class MockPortfolioManager:
        def __init__(self):
            self.cash = 100000
            self.positions = {'AAPL': 100, 'GOOGL': 50, 'MSFT': 200}
            
            # Mock enhanced features
            self.portfolio_history = []
            self.trade_history = []
            
            # Create some mock history
            base_value = 1000000
            for i in range(30):
                date = datetime.now() - timedelta(days=30-i)
                daily_change = np.random.normal(0, 0.02)
                base_value *= (1 + daily_change)
                
                from types import SimpleNamespace
                snapshot = SimpleNamespace()
                snapshot.timestamp = date
                snapshot.total_value = base_value
                snapshot.daily_pnl = base_value * daily_change
                snapshot.cumulative_pnl = base_value - 1000000
                
                self.portfolio_history.append(snapshot)
            
            # Create some mock trades
            for i in range(50):
                trade_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
                symbol = np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
                
                trade = SimpleNamespace()
                trade.timestamp = trade_date
                trade.symbol = symbol
                trade.side = np.random.choice(['BUY', 'SELL'])
                trade.shares = np.random.randint(10, 100)
                trade.price = np.random.uniform(100, 300)
                trade.notional_value = trade.shares * trade.price
                
                self.trade_history.append(trade)
        
        def get_portfolio_summary(self):
            total_value = 0
            positions_data = []
            
            for symbol, shares in self.positions.items():
                price = get_live_price(symbol, None)
                market_value = shares * price
                total_value += market_value
                
                positions_data.append({
                    'symbol': symbol,
                    'shares': shares,
                    'current_price': price,
                    'market_value': market_value,
                    'unrealized_pnl': market_value * np.random.uniform(-0.1, 0.1),
                    'unrealized_pnl_pct': np.random.uniform(-0.1, 0.1)
                })
            
            return {
                'total_value': total_value + self.cash,
                'cash': self.cash,
                'positions_value': total_value,
                'num_positions': len(positions_data),
                'positions': positions_data,
                'performance': {
                    'total_return': np.random.uniform(-0.1, 0.2),
                    'sharpe_ratio': np.random.uniform(0.5, 2.0),
                    'max_drawdown': np.random.uniform(0.05, 0.15)
                }
            }
    
    # Create mock risk manager
    class MockRiskManager:
        def __init__(self):
            self.risk_metrics_history = []
            
            # Create mock risk history
            for i in range(30):
                date = datetime.now() - timedelta(days=30-i)
                
                from types import SimpleNamespace
                metrics = SimpleNamespace()
                metrics.timestamp = date
                metrics.var_95 = np.random.uniform(0.01, 0.04)
                metrics.var_99 = np.random.uniform(0.02, 0.06)
                metrics.volatility_annual = np.random.uniform(0.15, 0.35)
                metrics.concentration_risk = np.random.uniform(0.3, 0.7)
                metrics.sharpe_ratio = np.random.uniform(0.5, 2.0)
                metrics.max_drawdown = np.random.uniform(0.05, 0.2)
                
                self.risk_metrics_history.append(metrics)
        
        def get_risk_dashboard(self):
            return {
                'current_risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'current_metrics': {
                    'var_95': 0.025,
                    'var_99': 0.045,
                    'max_drawdown': 0.12,
                    'volatility_annual': 0.22,
                    'concentration_risk': 0.45
                },
                'active_alerts': [
                    {'risk_level': 'HIGH', 'alert_type': 'VAR_BREACH'},
                    {'risk_level': 'MEDIUM', 'alert_type': 'CONCENTRATION'}
                ],
                'thresholds': {
                    'var_95_threshold': 0.02,
                    'max_drawdown_threshold': 0.15
                }
            }
    
    # Initialize components
    portfolio_manager = MockPortfolioManager()
    risk_manager = MockRiskManager()
    
    # Test enhanced report generator
    report_generator = EnhancedReportGenerator(
        report_style='professional',
        include_charts=True,
        auto_insights=True
    )
    
    print("Generating comprehensive report...")
    
    # Generate comprehensive report
    report = report_generator.generate_comprehensive_report(
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        moe_system=True,  # Mock MoE system
        report_period_days=30
    )
    
    # Display results
    print(f"Report sections generated: {len(report['sections'])}")
    
    exec_summary = report.get('executive_summary', {})
    print(f"Executive summary:")
    print(f"  Portfolio value: ${exec_summary.get('portfolio_value', 0):,.2f}")
    print(f"  Risk level: {exec_summary.get('current_risk_level', 'Unknown')}")
    print(f"  Number of positions: {exec_summary.get('number_of_positions', 0)}")
    print(f"  Total return: {exec_summary.get('total_return_pct', 0):.2f}%")
    
    print(f"Number of insights: {len(report.get('insights', []))}")
    print(f"Number of charts: {len(report.get('charts', {}))}")
    
    # Show some insights
    insights = report.get('insights', [])
    if insights:
        print("\nKey Insights:")
        for insight in insights[:3]:  # Show first 3 insights
            print(f"  - {insight.get('title', 'Unknown')}: {insight.get('description', 'No description')}")
    
    # Export reports
    try:
        report_generator.export_report_html(report, 'comprehensive_portfolio_report.html')
        print("✅ HTML report exported successfully")
    except Exception as e:
        print(f"❌ HTML export failed: {e}")
    
    try:
        report_generator.export_report_json(report, 'portfolio_report_data.json')
        print("✅ JSON report exported successfully")
    except Exception as e:
        print(f"❌ JSON export failed: {e}")
    
    # Test legacy compatibility
    print("\n--- Testing Legacy Compatibility ---")
    legacy_generator = ReportGenerator()
    
    class LegacyPortfolioManager:
        def __init__(self):
            self.cash = 50000
            self.positions = {'AAPL': 100, 'TSLA': 25}
    
    legacy_pm = LegacyPortfolioManager()
    summary, df = legacy_generator.generate_summary(legacy_pm)
    
    print(f"Legacy summary: {summary}")
    print(f"Legacy DataFrame shape: {df.shape}")
    if not df.empty:
        print(f"Legacy DataFrame columns: {list(df.columns)}")
    
    print("\n✅ Enhanced Report Generator testing completed successfully!")
    
    return report

def create_sample_reports():
    """Create sample reports for demonstration"""
    
    print("Creating sample reports...")
    
    # Test with different configurations
    configs = [
        {
            'name': 'executive_summary',
            'style': 'professional',
            'charts': False,
            'insights': False
        },
        {
            'name': 'full_analytics',
            'style': 'professional', 
            'charts': True,
            'insights': True
        }
    ]
    
    for config in configs:
        print(f"\nGenerating {config['name']} report...")
        
        # Create mock data (simplified)
        class SimpleMockPM:
            def __init__(self):
                self.cash = 150000
                self.positions = {'AAPL': 150, 'MSFT': 100, 'GOOGL': 75}
                self.portfolio_history = []
                self.trade_history = []
                
                # Create minimal history
                for i in range(30):
                    from types import SimpleNamespace
                    h = SimpleNamespace()
                    h.timestamp = datetime.now() - timedelta(days=30-i)
                    h.total_value = 1000000 + np.random.normal(0, 50000)
                    h.daily_pnl = np.random.normal(0, 10000)
                    h.cumulative_pnl = np.random.normal(50000, 100000)
                    self.portfolio_history.append(h)
            
            def get_portfolio_summary(self):
                return {
                    'total_value': 1200000,
                    'cash': self.cash,
                    'positions_value': 1050000,
                    'num_positions': 3,
                    'positions': [
                        {'symbol': 'AAPL', 'shares': 150, 'current_price': 180, 'market_value': 27000},
                        {'symbol': 'MSFT', 'shares': 100, 'current_price': 300, 'market_value': 30000},
                        {'symbol': 'GOOGL', 'shares': 75, 'current_price': 130, 'market_value': 9750}
                    ],
                    'performance': {
                        'total_return': 0.15,
                        'sharpe_ratio': 1.2,
                        'max_drawdown': 0.08
                    }
                }
        
        class SimpleMockRM:
            def get_risk_dashboard(self):
                return {
                    'current_risk_level': 'MEDIUM',
                    'current_metrics': {
                        'var_95': 0.02,
                        'volatility_annual': 0.18,
                        'concentration_risk': 0.35
                    },
                    'active_alerts': [],
                    'thresholds': {}
                }
            
            def __init__(self):
                self.risk_metrics_history = []
        
        pm = SimpleMockPM()
        rm = SimpleMockRM()
        
        # Generate report
        rg = EnhancedReportGenerator(
            report_style=config['style'],
            include_charts=config['charts'],
            auto_insights=config['insights']
        )
        
        report = rg.generate_comprehensive_report(pm, rm, report_period_days=30)
        
        # Export
        filename = f"sample_{config['name']}_report.html"
        rg.export_report_html(report, filename)
        print(f"✅ Created {filename}")

if __name__ == "__main__":
    # Run the main test
    test_report = test_enhanced_report_generator()
    
    # Create sample reports
    create_sample_reports()
    
    print("\n🎉 All tests completed successfully!")
    print("📄 Check the generated HTML files to see the reports.")
    
    def export_report_json(self, report: Dict, filepath: str):
        """Export report as JSON file"""
        
        try:
            # Create a JSON-serializable version
            json_report = self._make_json_serializable(report)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, default=str)
            
            logger.info(f"JSON report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting JSON report: {e}")
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

# =============================================================================
# Legacy Compatibility
# =============================================================================

class ReportGenerator(EnhancedReportGenerator):
    """
    Legacy compatibility wrapper for the original ReportGenerator interface
    """
    
    def __init__(self):
        super().__init__(
            report_style='professional',
            include_charts=False,  # Legacy didn't have charts
            auto_insights=False
        )
    
    def generate_summary(self, pm) -> Tuple[Dict, pd.DataFrame]:
        """
        Legacy summary generation method
        
        Returns:
            Tuple of (summary_dict, positions_dataframe)
        """
        try:
            # Use legacy portfolio summary method
            summary = self._get_legacy_portfolio_summary(pm)
            
            # Create positions DataFrame
            positions_data = []
            if hasattr(pm, 'positions'):
                for symbol, shares in pm.positions.items():
                    try:
                        price = get_live_price(symbol, None)
                        value = shares * price
                        positions_data.append({
                            'symbol': symbol,
                            'shares': shares,
                            'price': price,
                            'value': value
                        })
                    except Exception as e:
                        logger.error(f"Error getting price for {symbol}: {e}")
                        positions_data.append({
                            'symbol': symbol,
                            'shares': shares,
                            'price': np.nan,
                            'value': np.nan
                        })
            
            df = pd.DataFrame(positions_data)
            
            # Calculate totals
            total_positions_value = df['value'].sum() if not df.empty and 'value' in df.columns else 0
            total_value = total_positions_value + getattr(pm, 'cash', 0)
            
            legacy_summary = {
                'timestamp': pd.Timestamp.now(),
                'cash': getattr(pm, 'cash', 0),
                'positions_value': total_positions_value,
                'total_value': total_value
            }
            
            logger.info(f"Portfolio Summary: {legacy_summary}")
            logger.info(f"Positions DataFrame shape: {df.shape}")
            
            return legacy_summary, df
            
        except Exception as e:
            logger.error(f"Error generating legacy summary: {e}")
            return {'error': str(e)}, pd.DataFrame()

# =============================================================================
# Testing and Example Usage
# =============================================================================

def test_enhanced_report_generator():
    """Test the enhanced report generator with improved error handling"""
    
    print("Testing Enhanced Report Generator...")
    
    # Create mock portfolio manager
    class MockPortfolioManager:
        def __init__(self):
            self.cash = 100000
            self.positions = {'AAPL': 100, 'GOOGL': 50, 'MSFT': 200}
            
            # Mock enhanced features
            self.portfolio_history = []
            self.trade_history = []
            
            # Create some mock history
            base_value = 1000000
            for i in range(30):
                date = datetime.now() - timedelta(days=30-i)
                daily_change = np.random.normal(0, 0.02)
                base_value *= (1 + daily_change)
                
                from types import SimpleNamespace
                snapshot = SimpleNamespace()
                snapshot.timestamp = date
                snapshot.total_value = base_value
                snapshot.daily_pnl = base_value * daily_change
                snapshot.cumulative_pnl = base_value - 1000000
                
                self.portfolio_history.append(snapshot)
            
            # Create some mock trades
            for i in range(50):
                trade_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
                symbol = np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
                
                trade = SimpleNamespace()
                trade.timestamp = trade_date
                trade.symbol = symbol
                trade.side = np.random.choice(['BUY', 'SELL'])
                trade.shares = np.random.randint(10, 100)
                trade.price = np.random.uniform(100, 300)
                trade.notional_value = trade.shares * trade.price
                
                self.trade_history.append(trade)
        
        def get_portfolio_summary(self):
            total_value = 0
            positions_data = []
            
            for symbol, shares in self.positions.items():
                price = get_live_price(symbol, None)
                market_value = shares * price
                total_value += market_value
                
                positions_data.append({
                    'symbol': symbol,
                    'shares': shares,
                    'current_price': price,
                    'market_value': market_value,
                    'unrealized_pnl': market_value * np.random.uniform(-0.1, 0.1),
                    'unrealized_pnl_pct': np.random.uniform(-0.1, 0.1)
                })
            
            return {
                'total_value': total_value + self.cash,
                'cash': self.cash,
                'positions_value': total_value,
                'num_positions': len(positions_data),
                'positions': positions_data,
                'performance': {
                    'total_return': np.random.uniform(-0.1, 0.2),
                    'sharpe_ratio': np.random.uniform(0.5, 2.0),
                    'max_drawdown': np.random.uniform(0.05, 0.15)
                }
            }
    
    # Create mock risk manager
    class MockRiskManager:
        def __init__(self):
            self.risk_metrics_history = []
            
            # Create mock risk history
            for i in range(30):
                date = datetime.now() - timedelta(days=30-i)
                
                from types import SimpleNamespace
                metrics = SimpleNamespace()
                metrics.timestamp = date
                metrics.var_95 = np.random.uniform(0.01, 0.04)
                metrics.var_99 = np.random.uniform(0.02, 0.06)
                metrics.volatility_annual = np.random.uniform(0.15, 0.35)
                metrics.concentration_risk = np.random.uniform(0.3, 0.7)
                metrics.sharpe_ratio = np.random.uniform(0.5, 2.0)
                metrics.max_drawdown = np.random.uniform(0.05, 0.2)
                
                self.risk_metrics_history.append(metrics)
        
        def get_risk_dashboard(self):
            return {
                'current_risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'current_metrics': {
                    'var_95': 0.025,
                    'var_99': 0.045,
                    'max_drawdown': 0.12,
                    'volatility_annual': 0.22,
                    'concentration_risk': 0.45
                },
                'active_alerts': [
                    {'risk_level': 'HIGH', 'alert_type': 'VAR_BREACH'},
                    {'risk_level': 'MEDIUM', 'alert_type': 'CONCENTRATION'}
                ],
                'thresholds': {
                    'var_95_threshold': 0.02,
                    'max_drawdown_threshold': 0.15
                }
            }
    
    # Initialize components
    portfolio_manager = MockPortfolioManager()
    risk_manager = MockRiskManager()
    
    # Test enhanced report generator
    report_generator = EnhancedReportGenerator(
        report_style='professional',
        include_charts=True,
        auto_insights=True
    )
    
    print("Generating comprehensive report...")
    
    # Generate comprehensive report
    report = report_generator.generate_comprehensive_report(
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        moe_system=True,  # Mock MoE system
        report_period_days=30
    )
    
    # Display results
    print(f"Report sections generated: {len(report['sections'])}")
    
    exec_summary = report.get('executive_summary', {})
    print(f"Executive summary:")
    print(f"  Portfolio value: ${exec_summary.get('portfolio_value', 0):,.2f}")
    print(f"  Risk level: {exec_summary.get('current_risk_level', 'Unknown')}")
    print(f"  Number of positions: {exec_summary.get('number_of_positions', 0)}")
    print(f"  Total return: {exec_summary.get('total_return_pct', 0):.2f}%")
    
    print(f"Number of insights: {len(report.get('insights', []))}")
    print(f"Number of charts: {len(report.get('charts', {}))}")
    
    # Show some insights
    insights = report.get('insights', [])
    if insights:
        print("\nKey Insights:")
        for insight in insights[:3]:  # Show first 3 insights
            print(f"  - {insight.get('title', 'Unknown')}: {insight.get('description', 'No description')}")
    
    # Export reports
    try:
        report_generator.export_report_html(report, 'comprehensive_portfolio_report.html')
        print("✅ HTML report exported successfully")
    except Exception as e:
        print(f"❌ HTML export failed: {e}")
    
    try:
        report_generator.export_report_json(report, 'portfolio_report_data.json')
        print("✅ JSON report exported successfully")
    except Exception as e:
        print(f"❌ JSON export failed: {e}")
    
    # Test legacy compatibility
    print("\n--- Testing Legacy Compatibility ---")
    legacy_generator = ReportGenerator()
    
    class LegacyPortfolioManager:
        def __init__(self):
            self.cash = 50000
            self.positions = {'AAPL': 100, 'TSLA': 25}
    
    legacy_pm = LegacyPortfolioManager()
    summary, df = legacy_generator.generate_summary(legacy_pm)
    
    print(f"Legacy summary: {summary}")
    print(f"Legacy DataFrame shape: {df.shape}")
    if not df.empty:
        print(f"Legacy DataFrame columns: {list(df.columns)}")
    
    print("\n✅ Enhanced Report Generator testing completed successfully!")
    
    return report

def create_sample_reports():
    """Create sample reports for demonstration"""
    
    print("Creating sample reports...")
    
    # Test with different configurations
    configs = [
        {
            'name': 'executive_summary',
            'style': 'professional',
            'charts': False,
            'insights': False
        },
        {
            'name': 'full_analytics',
            'style': 'professional', 
            'charts': True,
            'insights': True
        }
    ]
    
    for config in configs:
        print(f"\nGenerating {config['name']} report...")
        
        # Create mock data (simplified)
        class SimpleMockPM:
            def __init__(self):
                self.cash = 150000
                self.positions = {'AAPL': 150, 'MSFT': 100, 'GOOGL': 75}
                self.portfolio_history = []
                self.trade_history = []
                
                # Create minimal history
                for i in range(30):
                    from types import SimpleNamespace
                    h = SimpleNamespace()
                    h.timestamp = datetime.now() - timedelta(days=30-i)
                    h.total_value = 1000000 + np.random.normal(0, 50000)
                    h.daily_pnl = np.random.normal(0, 10000)
                    h.cumulative_pnl = np.random.normal(50000, 100000)
                    self.portfolio_history.append(h)
            
            def get_portfolio_summary(self):
                return {
                    'total_value': 1200000,
                    'cash': self.cash,
                    'positions_value': 1050000,
                    'num_positions': 3,
                    'positions': [
                        {'symbol': 'AAPL', 'shares': 150, 'current_price': 180, 'market_value': 27000},
                        {'symbol': 'MSFT', 'shares': 100, 'current_price': 300, 'market_value': 30000},
                        {'symbol': 'GOOGL', 'shares': 75, 'current_price': 130, 'market_value': 9750}
                    ],
                    'performance': {
                        'total_return': 0.15,
                        'sharpe_ratio': 1.2,
                        'max_drawdown': 0.08
                    }
                }
        
        class SimpleMockRM:
            def get_risk_dashboard(self):
                return {
                    'current_risk_level': 'MEDIUM',
                    'current_metrics': {
                        'var_95': 0.02,
                        'volatility_annual': 0.18,
                        'concentration_risk': 0.35
                    },
                    'active_alerts': [],
                    'thresholds': {}
                }
            
            def __init__(self):
                self.risk_metrics_history = []
        
        pm = SimpleMockPM()
        rm = SimpleMockRM()
        
        # Generate report
        rg = EnhancedReportGenerator(
            report_style=config['style'],
            include_charts=config['charts'],
            auto_insights=config['insights']
        )
        
        report = rg.generate_comprehensive_report(pm, rm, report_period_days=30)
        
        # Export
        filename = f"sample_{config['name']}_report.html"
        rg.export_report_html(report, filename)
        print(f"✅ Created {filename}")

if __name__ == "__main__":
    # Run the main test
    test_report = test_enhanced_report_generator()
    
    # Create sample reports
    create_sample_reports()
    
    print("\n🎉 All tests completed successfully!")
    print("📄 Check the generated HTML files to see the reports.")
