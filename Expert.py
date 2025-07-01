import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class MultiHeadCrossAttention(nn.Module):
    """Cross-attention between price series and market regime features"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        B, L, D = query.shape
        
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class VolatilityEncodingModule(nn.Module):
    """Specialized module for encoding volatility patterns"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Volatility feature extractors
        self.realized_vol_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        self.vol_surface_encoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Volatility regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)  # Low, Normal, High, Extreme volatility
        )
        
    def _calculate_rolling_volatility(self, returns: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate rolling volatility using a simple approach"""
        batch_size, seq_len = returns.shape
        
        if seq_len < window:
            return torch.std(returns, dim=1, keepdim=True)
        
        # Simple rolling calculation
        volatilities = []
        for i in range(seq_len - window + 1):
            window_returns = returns[:, i:i+window]
            vol = torch.std(window_returns, dim=1)
            volatilities.append(vol)
        
        if volatilities:
            vol_tensor = torch.stack(volatilities, dim=1)
            return torch.mean(vol_tensor, dim=1, keepdim=True)
        else:
            return torch.std(returns, dim=1, keepdim=True)
    
    def forward(self, price_series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            price_series: [batch, seq_len, features]
        Returns:
            vol_features: [batch, hidden_dim]
            vol_regime: [batch, 4] - volatility regime probabilities
        """
        try:
            # Calculate realized volatility features
            if price_series.size(-1) == 1:  # Just prices
                # Add small epsilon to avoid log(0)
                safe_prices = torch.clamp(price_series, min=1e-8)
                log_prices = torch.log(safe_prices)
                returns = torch.diff(log_prices, dim=1).squeeze(-1)  # [batch, seq_len-1]
            else:  # Multiple features including returns
                returns = price_series[:, 1:, 0]  # Assume first feature is returns
            
            # Handle edge case where returns might be empty
            if returns.size(1) == 0:
                # Fallback to using price differences
                if price_series.size(-1) == 1:
                    prices = price_series.squeeze(-1)
                else:
                    prices = price_series[:, :, 0]
                returns = torch.diff(prices, dim=1)
            
            # Rolling volatility calculations with different windows
            window_sizes = [5, 10, 20]
            vol_features = []
            
            for window in window_sizes:
                vol_feature = self._calculate_rolling_volatility(returns, window)
                vol_features.append(vol_feature)
            
            vol_tensor = torch.cat(vol_features, dim=1)  # [batch, 3]
            
            # Pad to match input_dim if necessary
            if vol_tensor.size(1) < self.input_dim:
                padding = torch.zeros(vol_tensor.size(0), self.input_dim - vol_tensor.size(1), 
                                    device=vol_tensor.device)
                vol_tensor = torch.cat([vol_tensor, padding], dim=1)
            elif vol_tensor.size(1) > self.input_dim:
                vol_tensor = vol_tensor[:, :self.input_dim]
            
            # Encode volatility features
            realized_vol_feats = self.realized_vol_encoder(vol_tensor)
            vol_surface_feats = self.vol_surface_encoder(realized_vol_feats)
            
            # Combine features
            combined_vol_feats = torch.cat([realized_vol_feats, vol_surface_feats], dim=1)
            
            # Classify volatility regime
            vol_regime = F.softmax(self.regime_classifier(combined_vol_feats), dim=1)
            
            return combined_vol_feats, vol_regime
            
        except Exception as e:
            print(f"Error in VolatilityEncodingModule: {e}")
            # Fallback: return simple features
            batch_size = price_series.size(0)
            device = price_series.device
            
            # Simple fallback features
            fallback_features = torch.randn(batch_size, self.hidden_dim, device=device) * 0.1
            fallback_regime = F.softmax(torch.randn(batch_size, 4, device=device), dim=1)
            
            return fallback_features, fallback_regime

class MarketRegimeEncoder(nn.Module):
    """Encode market regime information (trend, momentum, etc.)"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        # Store input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Technical indicator encoders
        self.trend_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        self.momentum_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        self.mean_reversion_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        self.cycle_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        # Market regime classifier
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)  # Bull, Bear, Sideways, Transition
        )
        
    def forward(self, price_series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract market regime features from price series
        """
        try:
            batch_size, seq_len = price_series.shape[:2]
            
            if price_series.size(-1) == 1:
                prices = price_series.squeeze(-1)
            else:
                prices = price_series[:, :, 0]  # Assume first feature is price
            
            # Calculate technical indicators
            features = []
            
            # Trend features (moving averages)
            if seq_len >= 20:
                # Manual moving average calculation for better control
                sma_5_vals = []
                sma_20_vals = []
                
                for i in range(seq_len):
                    # SMA-5
                    start_5 = max(0, i - 4)
                    sma_5 = torch.mean(prices[:, start_5:i+1], dim=1)
                    sma_5_vals.append(sma_5)
                    
                    # SMA-20  
                    start_20 = max(0, i - 19)
                    sma_20 = torch.mean(prices[:, start_20:i+1], dim=1)
                    sma_20_vals.append(sma_20)
                
                sma_5_tensor = torch.stack(sma_5_vals, dim=1)  # [batch, seq_len]
                sma_20_tensor = torch.stack(sma_20_vals, dim=1)  # [batch, seq_len]
                
                trend_signal = (prices - sma_20_tensor) / (sma_20_tensor + 1e-8)
                features.append(torch.mean(trend_signal, dim=1, keepdim=True))
            else:
                features.append(torch.zeros(batch_size, 1, device=price_series.device))
            
            # Momentum features (rate of change)
            if seq_len >= 10:
                momentum = (prices[:, -1:] - prices[:, -10:-9]) / (prices[:, -10:-9] + 1e-8)
                features.append(momentum)
            else:
                features.append(torch.zeros(batch_size, 1, device=price_series.device))
            
            # Mean reversion features (deviation from mean)
            price_mean = torch.mean(prices, dim=1, keepdim=True)
            mean_reversion = (prices[:, -1:] - price_mean) / (price_mean + 1e-8)
            features.append(mean_reversion)
            
            # Cycle features (periodicity detection)
            if seq_len >= 5:
                cycle_feature = torch.std(prices, dim=1, keepdim=True) / (torch.mean(prices, dim=1, keepdim=True) + 1e-8)
                features.append(cycle_feature)
            else:
                features.append(torch.zeros(batch_size, 1, device=price_series.device))
            
            # Combine all features
            combined_features = torch.cat(features, dim=1)  # [batch, 4]
            
            # Pad if necessary to match input_dim
            if combined_features.size(1) < self.input_dim:
                padding = torch.zeros(batch_size, self.input_dim - combined_features.size(1), 
                                    device=price_series.device)
                combined_features = torch.cat([combined_features, padding], dim=1)
            elif combined_features.size(1) > self.input_dim:
                combined_features = combined_features[:, :self.input_dim]
            
            # Encode different aspects
            trend_feats = self.trend_encoder(combined_features)
            momentum_feats = self.momentum_encoder(combined_features)
            mean_rev_feats = self.mean_reversion_encoder(combined_features)
            cycle_feats = self.cycle_encoder(combined_features)
            
            # Combine all regime features
            regime_features = torch.cat([trend_feats, momentum_feats, mean_rev_feats, cycle_feats], dim=1)
            
            # Classify market regime
            regime_probs = F.softmax(self.regime_head(regime_features), dim=1)
            
            return regime_features, regime_probs
            
        except Exception as e:
            print(f"Error in MarketRegimeEncoder: {e}")
            # Fallback: return simple features
            batch_size = price_series.size(0)
            device = price_series.device
            
            # Simple fallback features
            fallback_features = torch.randn(batch_size, self.hidden_dim, device=device) * 0.1
            fallback_regime = F.softmax(torch.randn(batch_size, 4, device=device), dim=1)
            
            return fallback_features, fallback_regime

class OptionsStrategyEncoder(nn.Module):
    """Encode strategy-specific features and constraints"""
    def __init__(self, num_strategies: int, feature_dim: int):
        super().__init__()
        self.num_strategies = num_strategies
        self.feature_dim = feature_dim
        
        # Strategy embeddings
        self.strategy_embeddings = nn.Embedding(num_strategies, feature_dim)
        
        # We'll create the compatibility network dynamically based on actual input sizes
        self.compatibility_network = None
        
    def _create_compatibility_network(self, input_dim: int, device):
        """Create compatibility network with correct input dimension"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        
    def forward(self, market_features: torch.Tensor, vol_features: torch.Tensor) -> torch.Tensor:
        """
        Score compatibility of each strategy with current market conditions
        """
        batch_size = market_features.size(0)
        strategy_scores = []
        
        # Get first strategy embedding to determine dimensions
        strategy_embed_sample = self.strategy_embeddings(
            torch.full((batch_size,), 0, device=market_features.device)
        )
        
        # Combine features to determine actual input size
        combined_sample = torch.cat([market_features, vol_features, strategy_embed_sample], dim=1)
        actual_input_dim = combined_sample.size(1)
        
        # Create compatibility network if not exists or wrong size
        if (self.compatibility_network is None or 
            self.compatibility_network[0].in_features != actual_input_dim):
            self.compatibility_network = self._create_compatibility_network(
                actual_input_dim, market_features.device
            )
        
        # Score each strategy
        for strategy_id in range(self.num_strategies):
            strategy_embed = self.strategy_embeddings(
                torch.full((batch_size,), strategy_id, device=market_features.device)
            )
            
            # Combine all features
            combined = torch.cat([market_features, vol_features, strategy_embed], dim=1)
            score = self.compatibility_network(combined)
            strategy_scores.append(score)
        
        return torch.cat(strategy_scores, dim=1)  # [batch, num_strategies]

class EnhancedExpert(nn.Module):
    """
    Enhanced Expert for Options Trading with specialized financial modules
    """
    def __init__(self,
                 input_dim: int,
                 model_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_dim: int = 512,
                 seq_len: int = 30,
                 output_dim: int = 10,
                 num_strategies: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_strategies = num_strategies
        
        # Specialized financial modules
        self.volatility_module = VolatilityEncodingModule(input_dim, model_dim // 2)
        self.market_regime_module = MarketRegimeEncoder(input_dim, model_dim // 2)
        self.strategy_encoder = OptionsStrategyEncoder(num_strategies, model_dim // 4)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        
        # Input projection with financial feature fusion
        self.input_proj = nn.Linear(input_dim, model_dim // 2)
        
        # Calculate fusion input dimension:
        # h_input: model_dim // 2
        # vol_features: model_dim // 2 (from VolatilityEncodingModule.hidden_dim) 
        # market_features: model_dim // 2 (from MarketRegimeEncoder.hidden_dim)
        fusion_input_dim = (model_dim // 2) * 3  # Three components of size model_dim // 2
        
        self.financial_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced Transformer with cross-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',  # Better for financial data
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-level attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, model_dim))
        
        # Multi-task output heads
        self.strategy_head = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_strategies)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # max_profit, max_loss, probability
        )
        
        self.signal_head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
        # Regime-aware gating
        self.regime_gate = nn.Sequential(
            nn.Linear(4, model_dim),  # 4 market regimes
            nn.Sigmoid()
        )
        
        self.vol_gate = nn.Sequential(
            nn.Linear(4, model_dim),  # 4 volatility regimes
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, return_auxiliary: bool = False) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with multi-task outputs
        
        Args:
            x: [batch, seq_len, input_dim] - price series and features
            return_auxiliary: whether to return auxiliary outputs
            
        Returns:
            Dictionary containing main signals and auxiliary outputs
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract specialized financial features
        vol_features, vol_regime = self.volatility_module(x)  # [batch, model_dim//2], [batch, 4]
        market_features, market_regime = self.market_regime_module(x)  # [batch, model_dim//2], [batch, 4]
        
        # Project input features
        h_input = self.input_proj(x.view(-1, self.input_dim)).view(batch_size, seq_len, -1)  # [batch, seq_len, model_dim//2]
        
        # Expand features to sequence length for fusion
        vol_expanded = vol_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, model_dim//2]
        market_expanded = market_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, model_dim//2]
        
        # Debug: Print shapes to understand the issue
        # print(f"h_input shape: {h_input.shape}")
        # print(f"vol_expanded shape: {vol_expanded.shape}")
        # print(f"market_expanded shape: {market_expanded.shape}")
        
        # Fuse all features - this should be [batch, seq_len, model_dim//2 * 3]
        h_combined = torch.cat([h_input, vol_expanded, market_expanded], dim=-1)
        # print(f"h_combined shape: {h_combined.shape}")
        
        # Calculate the actual input size for financial_fusion
        actual_fusion_input_dim = h_combined.size(-1)
        
        # If dimensions don't match, adjust the fusion layer
        if actual_fusion_input_dim != self.model_dim * 2:
            # Create a new fusion layer with correct dimensions
            self.financial_fusion = nn.Sequential(
                nn.Linear(actual_fusion_input_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).to(h_combined.device)
        
        h = self.financial_fusion(h_combined)
        
        # Add positional embeddings
        h = h + self.pos_embed
        
        # Regime-aware gating
        market_gate = self.regime_gate(market_regime).unsqueeze(1)
        vol_gate = self.vol_gate(vol_regime).unsqueeze(1)
        h = h * market_gate * vol_gate
        
        # Transformer encoding
        h = self.transformer(h)
        
        # Attention-based pooling
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        pooled_h, attention_weights = self.attention_pool(pool_query, h, h, need_weights=True)
        pooled_h = pooled_h.squeeze(1)  # [batch, model_dim]
        
        # Multi-task outputs
        strategy_scores = self.strategy_head(pooled_h)
        risk_metrics = self.risk_head(pooled_h)
        trading_signals = self.signal_head(pooled_h)
        
        # Get strategy compatibility scores
        strategy_compatibility = self.strategy_encoder(market_features, vol_features)
        
        # Combine strategy scores with compatibility
        final_strategy_scores = strategy_scores + strategy_compatibility
        
        outputs = {
            'signals': trading_signals,
            'strategy_scores': final_strategy_scores,
            'risk_metrics': risk_metrics,
        }
        
        if return_auxiliary:
            outputs.update({
                'market_regime': market_regime,
                'volatility_regime': vol_regime,
                'attention_weights': attention_weights,
                'volatility_features': vol_features,
                'market_features': market_features
            })
        
        return outputs
    
    def get_strategy_recommendation(self, x: torch.Tensor, symbols: List[str]) -> Dict[str, any]:
        """
        Get specific strategy recommendation for given inputs
        """
        with torch.no_grad():
            outputs = self.forward(x, return_auxiliary=True)
            
            # Get best strategy for each symbol
            strategy_scores = outputs['strategy_scores']
            best_strategies = torch.argmax(strategy_scores, dim=1)
            
            recommendations = {}
            for i, symbol in enumerate(symbols):
                recommendations[symbol] = {
                    'best_strategy_id': best_strategies[i].item(),
                    'strategy_confidence': torch.softmax(strategy_scores[i], dim=0)[best_strategies[i]].item(),
                    'market_regime': torch.argmax(outputs['market_regime'][i]).item(),
                    'volatility_regime': torch.argmax(outputs['volatility_regime'][i]).item(),
                    'risk_metrics': outputs['risk_metrics'][i].tolist(),
                    'trading_signals': outputs['signals'][i].tolist()
                }
            
            return recommendations

# Strategy mapping for integration with OptionsStrategist
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

# Example usage and testing
def test_enhanced_expert():
    """Test the enhanced expert"""
    print("Testing Enhanced Expert...")
    
    # Create dummy data
    batch_size, seq_len, input_dim = 4, 30, 5
    x = torch.randn(batch_size, seq_len, input_dim)
    symbols = ['AAPL', 'TSLA', 'SPY', 'QQQ']
    
    print(f"Input tensor shape: {x.shape}")
    print(f"Input tensor stats - min: {x.min():.3f}, max: {x.max():.3f}, mean: {x.mean():.3f}")
    
    try:
        # Initialize expert
        model_dim = 256
        expert = EnhancedExpert(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=8,
            num_layers=4,
            seq_len=seq_len,
            output_dim=10,
            num_strategies=10
        )
        print("✓ Expert initialized successfully")
        
        # Test individual modules first
        print("\nTesting individual modules...")
        
        vol_module = expert.volatility_module
        vol_features, vol_regime = vol_module(x)
        print(f"✓ Volatility module - features: {vol_features.shape}, regime: {vol_regime.shape}")
        
        market_module = expert.market_regime_module  
        market_features, market_regime = market_module(x)
        print(f"✓ Market regime module - features: {market_features.shape}, regime: {market_regime.shape}")
        
        # Test input projection
        h_input = expert.input_proj(x.view(-1, input_dim)).view(batch_size, seq_len, -1)
        print(f"✓ Input projection - shape: {h_input.shape}")
        
        # Test feature expansion and fusion
        vol_expanded = vol_features.unsqueeze(1).expand(-1, seq_len, -1)
        market_expanded = market_features.unsqueeze(1).expand(-1, seq_len, -1)
        h_combined = torch.cat([h_input, vol_expanded, market_expanded], dim=-1)
        print(f"✓ Feature fusion - combined shape: {h_combined.shape}")
        print(f"  Expected fusion input dim: {expert.financial_fusion[0].in_features}")
        print(f"  Actual fusion input dim: {h_combined.size(-1)}")
        
        # Test strategy encoder specifically
        strategy_encoder = expert.strategy_encoder
        print(f"✓ Strategy encoder feature_dim: {strategy_encoder.feature_dim}")
        
        # Test strategy compatibility calculation
        try:
            strategy_embed = strategy_encoder.strategy_embeddings(torch.tensor([0], device=vol_features.device))
            print(f"✓ Strategy embedding shape: {strategy_embed.shape}")
            
            combined_test = torch.cat([market_features[:1], vol_features[:1], strategy_embed], dim=1)
            print(f"✓ Combined features for strategy test: {combined_test.shape}")
            print(f"  Expected compatibility input: {strategy_encoder.compatibility_network[0].in_features}")
        except Exception as e:
            print(f"✗ Strategy encoder dimension issue: {e}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        outputs = expert(x, return_auxiliary=True)
        print("✓ Forward pass successful")
        
        print("\nOutput shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Test strategy recommendations
        print("\nTesting strategy recommendations...")
        recommendations = expert.get_strategy_recommendation(x, symbols)
        print("✓ Strategy recommendations successful")
        
        print("\nStrategy Recommendations:")
        for symbol, rec in recommendations.items():
            strategy_name = STRATEGY_MAPPING[rec['best_strategy_id']]
            print(f"  {symbol}: {strategy_name} (confidence: {rec['strategy_confidence']:.3f})")
            
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_expert()