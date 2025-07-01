import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import math

# =============================================================================
# Financial Market Condition Analyzer
# =============================================================================

class MarketConditionEncoder(nn.Module):
    """
    Encodes market conditions to help gate make informed expert selection
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Technical indicator extractors
        self.volatility_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
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
        
        self.regime_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        # Market regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 4)  # Bull, Bear, Sideways, High Volatility
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract market condition features
        
        Args:
            x: [batch, seq_len, features] or [batch, features]
            
        Returns:
            market_features: [batch, hidden_dim]
            regime_probs: [batch, 4]
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 3:
            # For sequence data, use the last timestep
            x = x[:, -1, :]
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Ensure we have the right input dimension
        if x.size(-1) != self.input_dim:
            # Pad or truncate to match expected input_dim
            if x.size(-1) < self.input_dim:
                padding = torch.zeros(x.size(0), self.input_dim - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.input_dim]
        
        # Extract different market condition features
        vol_features = self.volatility_encoder(x)
        trend_features = self.trend_encoder(x)
        momentum_features = self.momentum_encoder(x)
        regime_features = self.regime_encoder(x)
        
        # Combine all features
        market_features = torch.cat([vol_features, trend_features, 
                                   momentum_features, regime_features], dim=1)
        
        # Classify market regime
        regime_probs = F.softmax(self.regime_classifier(market_features), dim=1)
        
        return market_features, regime_probs

# =============================================================================
# Enhanced Noisy Top-K Gating with Financial Awareness - FIXED
# =============================================================================

class FinancialNoisyTopKGate(nn.Module):
    """
    Enhanced Noisy Top-K gating specifically designed for financial data
    with market regime awareness and adaptive noise scaling
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, 
                 base_noise_std: float = 1.0, market_aware: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.base_noise_std = base_noise_std
        self.market_aware = market_aware
        
        # Market condition encoder
        if market_aware:
            self.market_encoder = MarketConditionEncoder(input_dim, 64)
            gating_input_dim = input_dim + 64 + 4  # original + market features + regime probs
        else:
            gating_input_dim = input_dim
        
        # Main gating network with financial-specific architecture
        self.gating_network = nn.Sequential(
            nn.Linear(gating_input_dim, gating_input_dim * 2),
            nn.LayerNorm(gating_input_dim * 2),
            nn.GELU(),  # Better for financial data than ReLU
            nn.Dropout(0.1),
            
            nn.Linear(gating_input_dim * 2, gating_input_dim),
            nn.LayerNorm(gating_input_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            
            nn.Linear(gating_input_dim, num_experts)
        )
        
        # Adaptive noise scaling based on market volatility
        if market_aware:
            self.noise_scaler = nn.Sequential(
                nn.Linear(4, 16),  # 4 regime probabilities
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        # Expert specialization matrix (learnable)
        self.expert_specialization = nn.Parameter(
            torch.randn(num_experts, 4) * 0.1  # [experts, market_regimes]
        )
        
        # Load balancing parameters
        self.load_balance_weight = 0.01
        self.importance_loss_weight = 0.01
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with enhanced financial awareness
        
        Returns:
            topk_weights: [batch, k] - weights for top-k experts
            topk_idx: [batch, k] - indices of top-k experts  
            loss_aux: auxiliary loss for load balancing
            aux_info: additional information for analysis
        """
        # Ensure x is 2D [batch, features]
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  # Flatten sequence dimension
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        
        # Market condition analysis
        aux_info = {}
        if self.market_aware:
            market_features, regime_probs = self.market_encoder(x)
            aux_info['regime_probs'] = regime_probs
            aux_info['market_features'] = market_features
            
            # Combine original features with market analysis
            gating_input = torch.cat([x, market_features, regime_probs], dim=1)
            
            # Adaptive noise scaling based on market volatility
            # Higher volatility -> more noise (more exploration)
            noise_scale = self.noise_scaler(regime_probs) * self.base_noise_std
        else:
            gating_input = x
            noise_scale = self.base_noise_std
        
        # Compute clean logits
        clean_logits = self.gating_network(gating_input)  # [batch, num_experts]
        
        # Market regime bias: experts specialized for current regime get boost
        if self.market_aware:
            regime_bias = torch.matmul(regime_probs, self.expert_specialization.T)  # [batch, num_experts]
            clean_logits = clean_logits + regime_bias
        
        # Add adaptive noise for exploration
        if isinstance(noise_scale, torch.Tensor):
            noise = torch.randn_like(clean_logits) * noise_scale.unsqueeze(1)
        else:
            noise = torch.randn_like(clean_logits) * noise_scale
        
        noisy_logits = clean_logits + noise
        
        # Top-k selection with temperature scaling during training
        if self.training:
            temperature = 1.0 + 0.5 * torch.rand(1).item()  # Random temperature 1.0-1.5
            scaled_logits = noisy_logits / temperature
        else:
            scaled_logits = noisy_logits
        
        topk_logits, topk_idx = torch.topk(scaled_logits, self.k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        
        # Enhanced load balancing with importance weighting
        gates_softmax = F.softmax(clean_logits, dim=-1)  # [batch, num_experts]
        
        # Compute expert usage statistics
        prob_expert = gates_softmax.mean(0)  # [num_experts] - average gate probability
        
        # FIXED: Compute actual load (how often each expert is selected)
        one_hot = torch.zeros_like(clean_logits)  # [batch, num_experts]
        
        # Method 1: Use advanced indexing (most reliable)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.k)
        one_hot[batch_indices, topk_idx] = 1.0
        
        load_expert = one_hot.mean(0)  # [num_experts]
        
        # Load balancing loss: penalize imbalance
        load_loss = self.num_experts * torch.sum(prob_expert * load_expert)
        
        # Importance loss: encourage diverse expert usage
        importance = gates_softmax.sum(0)  # [num_experts]
        importance_loss = torch.var(importance) / torch.mean(importance)
        
        # Combined auxiliary loss
        loss_aux = (self.load_balance_weight * load_loss + 
                   self.importance_loss_weight * importance_loss)
        
        # Store additional info for monitoring
        aux_info.update({
            'expert_usage': load_expert,
            'expert_importance': prob_expert,
            'load_balance_loss': load_loss,
            'importance_loss': importance_loss,
            'noise_scale': noise_scale if isinstance(noise_scale, torch.Tensor) else torch.tensor(noise_scale)
        })
        
        return topk_weights, topk_idx, loss_aux, aux_info

# =============================================================================
# Enhanced Dense Gating Network
# =============================================================================

class FinancialDenseGate(nn.Module):
    """
    Enhanced dense gating network with financial market awareness
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: Optional[int] = None, 
                 dropout: float = 0.1, market_aware: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.market_aware = market_aware
        
        # Default hidden dimension
        hidden_dim = hidden_dim or max(input_dim, 128)
        
        # Market condition encoder
        if market_aware:
            self.market_encoder = MarketConditionEncoder(input_dim, 64)
            total_input_dim = input_dim + 64 + 4  # original + market features + regime probs
        else:
            total_input_dim = input_dim
        
        # Enhanced gating architecture
        self.gating_network = nn.Sequential(
            # First layer with residual connection capability
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second layer with attention-like mechanism
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            # Expert selection head
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Market regime specialization (similar to sparse gate)
        if market_aware:
            self.regime_expert_affinity = nn.Parameter(
                torch.randn(4, num_experts) * 0.1  # [regimes, experts]
            )
        
        # Temperature parameter for controlling gate sharpness
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass returning expert weights and auxiliary information
        
        Returns:
            weights: [batch, num_experts] - mixture weights
            aux_info: additional information for monitoring
        """
        # Ensure x is 2D [batch, features]
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  # Flatten sequence dimension
        elif x.dim() == 1:
            x = x.unsqueeze(0)
            
        batch_size = x.size(0)
        aux_info = {}
        
        # Market condition analysis
        if self.market_aware:
            market_features, regime_probs = self.market_encoder(x)
            aux_info['regime_probs'] = regime_probs
            aux_info['market_features'] = market_features
            
            # Combine features
            gating_input = torch.cat([x, market_features, regime_probs], dim=1)
            
            # Market regime bias for experts
            regime_bias = torch.matmul(regime_probs, self.regime_expert_affinity)  # [batch, num_experts]
        else:
            gating_input = x
            regime_bias = 0
        
        # Compute gating logits
        logits = self.gating_network(gating_input) + regime_bias
        
        # Apply temperature scaling
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=5.0)
        
        # Compute mixture weights
        weights = F.softmax(scaled_logits, dim=-1)
        
        # Store monitoring information
        aux_info.update({
            'expert_weights': weights,
            'gating_entropy': -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean(),
            'temperature': self.temperature.item(),
            'max_weight': weights.max(dim=1)[0].mean(),
            'effective_experts': (weights > 0.01).sum(dim=1).float().mean()
        })
        
        return weights, aux_info

# =============================================================================
# Adaptive Gating Strategy
# =============================================================================

class AdaptiveFinancialGate(nn.Module):
    """
    Adaptive gating that switches between sparse and dense based on market conditions
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, 
                 switch_threshold: float = 0.7):
        super().__init__()
        self.switch_threshold = switch_threshold
        
        # Both gating strategies
        self.sparse_gate = FinancialNoisyTopKGate(input_dim, num_experts, k, market_aware=True)
        self.dense_gate = FinancialDenseGate(input_dim, num_experts, market_aware=True)
        
        # Switching network
        self.switch_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Adaptive forward pass
        """
        # Ensure x is 2D [batch, features]
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  # Flatten sequence dimension
        elif x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Decide which gating strategy to use
        switch_prob = self.switch_network(x)
        use_sparse = switch_prob.squeeze() > self.switch_threshold
        
        if use_sparse.any():
            # Use sparse gating
            sparse_weights, sparse_idx, sparse_loss, sparse_aux = self.sparse_gate(x)
            
            # Convert sparse to dense format
            batch_size = x.size(0)
            dense_weights = torch.zeros(batch_size, self.sparse_gate.num_experts, device=x.device)
            
            # FIXED: Use advanced indexing for conversion
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.sparse_gate.k)
            dense_weights[batch_indices, sparse_idx] = sparse_weights
            
            return dense_weights, sparse_idx, sparse_loss, {
                'gating_type': 'sparse',
                'switch_prob': switch_prob,
                **sparse_aux
            }
        else:
            # Use dense gating
            dense_weights, dense_aux = self.dense_gate(x)
            
            # Get top-k for compatibility
            topk_weights, topk_idx = torch.topk(dense_weights, self.sparse_gate.k, dim=-1)
            
            return dense_weights, topk_idx, torch.tensor(0.0), {
                'gating_type': 'dense', 
                'switch_prob': switch_prob,
                **dense_aux
            }

# =============================================================================
# Backward Compatibility Wrapper
# =============================================================================

class Gate(nn.Module):
    """
    Enhanced Gate that maintains backward compatibility while adding financial intelligence
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: Optional[int] = None, 
                 dropout: float = 0.1, gating_type: str = 'dense', k: int = 2):
        super().__init__()
        self.gating_type = gating_type
        
        if gating_type == 'sparse':
            self.gate = FinancialNoisyTopKGate(input_dim, num_experts, k)
        elif gating_type == 'adaptive':
            self.gate = AdaptiveFinancialGate(input_dim, num_experts, k)
        else:  # dense
            self.gate = FinancialDenseGate(input_dim, num_experts, hidden_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Backward compatible forward pass
        """
        # Ensure x is 2D [batch, features]
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  # Flatten sequence dimension
        elif x.dim() == 1:
            x = x.unsqueeze(0)
            
        if self.gating_type == 'dense':
            weights, aux_info = self.gate(x)
            return weights
        else:
            weights, topk_idx, loss_aux, aux_info = self.gate(x)
            
            # Ensure weights and topk_idx have correct shapes
            batch_size = x.size(0)
            
            # For backward compatibility, return dense weights
            if hasattr(self.gate, 'num_experts'):
                num_experts = self.gate.num_experts
            elif hasattr(self.gate, 'sparse_gate'):
                num_experts = self.gate.sparse_gate.num_experts
            else:
                # If weights is already dense, return as-is
                if weights.dim() == 2 and weights.size(1) > topk_idx.size(1):
                    return weights
                num_experts = 8  # fallback
                
            # Check if conversion is needed
            if weights.size(1) != num_experts:
                # Convert sparse to dense using advanced indexing
                k = topk_idx.size(1)
                dense_weights = torch.zeros(batch_size, num_experts, device=x.device)
                
                # Ensure both tensors have correct shapes for indexing
                if weights.dim() == 2 and weights.size(1) == k and topk_idx.dim() == 2:
                    batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)
                    dense_weights[batch_indices, topk_idx] = weights
                else:
                    # Fallback: if shapes don't match expected pattern, return what we have
                    print(f"Warning: Unexpected tensor shapes - weights: {weights.shape}, topk_idx: {topk_idx.shape}")
                    return weights
                    
                return dense_weights
            return weights

# Legacy compatibility
class NoisyTopKGate(FinancialNoisyTopKGate):
    """Backward compatibility for existing code"""
    def __init__(self, input_dim, num_experts, k=2, noise_std=1.0):
        super().__init__(input_dim, num_experts, k, noise_std, market_aware=False)
    
    def forward(self, x):
        topk_weights, topk_idx, loss_aux, _ = super().forward(x)
        return topk_weights, topk_idx, loss_aux

# =============================================================================
# Testing and Example Usage
# =============================================================================

def test_scatter_operation():
    """Test the scatter operation separately to debug the issue"""
    print("Testing scatter operation...")
    
    batch_size, num_experts, k = 4, 8, 2
    
    # Create test tensors
    clean_logits = torch.randn(batch_size, num_experts)
    topk_logits, topk_idx = torch.topk(clean_logits, k, dim=-1)
    
    print(f"clean_logits shape: {clean_logits.shape}")
    print(f"topk_idx shape: {topk_idx.shape}")
    print(f"topk_logits shape: {topk_logits.shape}")
    
    # Test NEW scatter operation using advanced indexing
    one_hot = torch.zeros_like(clean_logits)
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
    one_hot[batch_indices, topk_idx] = 1.0
    
    print("✓ Advanced indexing scatter successful!")
    print(f"Result one_hot shape: {one_hot.shape}")
    print(f"Sum check (should be {k} per row): {one_hot.sum(dim=1)}")
    
    return one_hot

def test_enhanced_gates():
    """Test all gate implementations with better error handling"""
    
    # First test the scatter operation in isolation
    test_scatter_operation()
    
    batch_size, seq_len, input_dim, num_experts = 4, 30, 10, 8
    
    # Test data
    x_2d = torch.randn(batch_size, input_dim)
    x_3d = torch.randn(batch_size, seq_len, input_dim)
    
    print("\nTesting Enhanced Gates...")
    print(f"Input shapes - 2D: {x_2d.shape}, 3D: {x_3d.shape}")
    
    try:
        # Test dense gate
        print("\n1. Testing Dense Gate...")
        dense_gate = FinancialDenseGate(input_dim, num_experts)
        weights, aux = dense_gate(x_2d)
        print(f"✓ Dense Gate - Weights shape: {weights.shape}, Entropy: {aux['gating_entropy']:.3f}")
        print(f"  Effective experts: {aux['effective_experts']:.2f}, Max weight: {aux['max_weight']:.3f}")
    except Exception as e:
        print(f"✗ Dense Gate failed: {e}")
    
    try:
        # Test sparse gate
        print("\n2. Testing Sparse Gate...")
        sparse_gate = FinancialNoisyTopKGate(input_dim, num_experts, k=2)
        sparse_weights, sparse_idx, loss, aux = sparse_gate(x_2d)
        print(f"✓ Sparse Gate - Weights shape: {sparse_weights.shape}, Indices shape: {sparse_idx.shape}")
        print(f"  Load balance loss: {loss:.4f}")
        print(f"  Expert usage range: {aux['expert_usage'].min():.3f} - {aux['expert_usage'].max():.3f}")
    except Exception as e:
        print(f"✗ Sparse Gate failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test adaptive gate
        print("\n3. Testing Adaptive Gate...")
        adaptive_gate = AdaptiveFinancialGate(input_dim, num_experts)
        adaptive_weights, adaptive_idx, adaptive_loss, aux = adaptive_gate(x_2d)
        print(f"✓ Adaptive Gate - Type: {aux['gating_type']}, Weights shape: {adaptive_weights.shape}")
        print(f"  Switch probability: {aux['switch_prob'].mean():.3f}")
    except Exception as e:
        print(f"✗ Adaptive Gate failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test backward compatibility
        print("\n4. Testing Backward Compatibility...")
        compat_gate = Gate(input_dim, num_experts, gating_type='dense')
        compat_weights = compat_gate(x_2d)
        print(f"✓ Compatible Gate (dense) - Weights shape: {compat_weights.shape}")
        
        compat_gate_sparse = Gate(input_dim, num_experts, gating_type='sparse', k=2)
        compat_weights_sparse = compat_gate_sparse(x_2d)
        print(f"✓ Compatible Gate (sparse) - Weights shape: {compat_weights_sparse.shape}")
    except Exception as e:
        print(f"✗ Compatible Gate failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting completed!")

if __name__ == "__main__":
    test_enhanced_gates()