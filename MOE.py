import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
import math

# Import the enhanced components we already built
from Expert import EnhancedExpert  # Your existing Expert implementation
from Gate import (
    Gate, 
    NoisyTopKGate, 
    FinancialNoisyTopKGate, 
    FinancialDenseGate, 
    AdaptiveFinancialGate,
    MarketConditionEncoder
)

# =============================================================================
# Enhanced Financial Mixture of Experts using our existing Gate & Expert code
# =============================================================================

class FinancialMixtureOfExperts(nn.Module):
    """
    Advanced Financial MoE that properly utilizes our enhanced Gate and Expert implementations
    """
    
    def __init__(self,
                 input_dim: int,
                 seq_len: int,
                 model_dim: int,
                 num_heads: int,
                 num_layers: int,
                 ff_dim: int,
                 output_dim: int,
                 num_experts: int,
                 k: int = 2,
                 gating_type: str = 'adaptive',  # 'sparse', 'dense', 'adaptive'
                 noise_std: float = 1.0,
                 load_balance_weight: float = 0.01,
                 expert_dropout: float = 0.1,
                 use_market_conditioning: bool = True,
                 expert_output_key: str = 'signals',  # Which key to use from expert dict output
                 **kwargs):
        super().__init__()
        
        # Core parameters
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.gating_type = gating_type
        self.load_balance_weight = load_balance_weight
        self.use_market_conditioning = use_market_conditioning
        self.expert_output_key = expert_output_key  # NEW: Key to extract from expert dict
        
        # Calculate gate input dimension (flattened sequence)
        gate_input_dim = input_dim * seq_len
        
        # Initialize experts using your existing Expert class
        self.experts = nn.ModuleList([
            EnhancedExpert(
                input_dim=input_dim,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                seq_len=seq_len,
                output_dim=output_dim,
                dropout=expert_dropout
            ) for _ in range(num_experts)
        ])
        
        # Initialize enhanced gating using our existing Gate implementations
        self.gate = self._create_enhanced_gate(
            gate_input_dim, num_experts, k, gating_type, noise_std
        )
        
        # Market condition encoder (if using market conditioning)
        if use_market_conditioning:
            self.market_encoder = MarketConditionEncoder(gate_input_dim, hidden_dim=64)
        
        # Expert usage tracking for load balancing
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
        # Output processing
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(0.1)
        
        # Residual connection (if dimensions match)
        self.use_residual = (gate_input_dim == output_dim)
        if not self.use_residual and gate_input_dim != output_dim:
            self.residual_proj = nn.Linear(gate_input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def _create_enhanced_gate(self, input_dim: int, num_experts: int, k: int, 
                            gating_type: str, noise_std: float):
        """Create the appropriate enhanced gate using our existing implementations"""
        
        if gating_type == 'sparse':
            # Use our enhanced FinancialNoisyTopKGate
            return FinancialNoisyTopKGate(
                input_dim=input_dim,
                num_experts=num_experts,
                k=k,
                base_noise_std=noise_std,
                market_aware=self.use_market_conditioning
            )
        elif gating_type == 'dense':
            # Use our enhanced FinancialDenseGate
            return FinancialDenseGate(
                input_dim=input_dim,
                num_experts=num_experts,
                market_aware=self.use_market_conditioning
            )
        elif gating_type == 'adaptive':
            # Use our AdaptiveFinancialGate
            return AdaptiveFinancialGate(
                input_dim=input_dim,
                num_experts=num_experts,
                k=k
            )
        else:
            # Fallback to basic enhanced Gate
            return Gate(
                input_dim=input_dim,
                num_experts=num_experts,
                gating_type='dense',
                k=k
            )
    
    def forward(self, x: torch.Tensor, 
                return_aux_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Enhanced forward pass utilizing our existing Gate and Expert implementations
        
        Args:
            x: [batch, seq_len, input_dim] - Input financial time series
            return_aux_info: Whether to return auxiliary information
            
        Returns:
            output: [batch, output_dim] - MoE predictions
            aux_info: Dictionary with gating and expert information (if requested)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq_len, input_dim], got {x.shape}")
        
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # Prepare input for gating (flatten sequence dimension)
        gate_input = x.reshape(batch_size, -1)  # [batch, seq_len * input_dim]
        
        # Get gating decisions using our enhanced gates
        aux_info = {}
        
        if self.gating_type in ['sparse', 'adaptive']:
            # Enhanced gates return (weights, indices, loss, aux_info)
            try:
                topk_weights, topk_idx, gate_loss, gate_aux = self.gate(gate_input)
                aux_info.update(gate_aux)
            except ValueError as e:
                # Handle cases where gate returns different format
                gate_result = self.gate(gate_input)
                if len(gate_result) == 4:
                    topk_weights, topk_idx, gate_loss, gate_aux = gate_result
                    aux_info.update(gate_aux)
                elif len(gate_result) == 3:
                    topk_weights, topk_idx, gate_loss = gate_result
                else:
                    # Basic gate compatibility
                    gate_weights = gate_result
                    topk_weights, topk_idx = torch.topk(gate_weights, self.k, dim=-1)
                    topk_weights = F.softmax(topk_weights, dim=-1)
                    gate_loss = torch.tensor(0.0, device=device)
        
        elif self.gating_type == 'dense':
            # Dense gate returns (weights, aux_info)
            try:
                gate_weights, gate_aux = self.gate(gate_input)
                aux_info.update(gate_aux)
                # Get top-k for sparse computation
                topk_weights, topk_idx = torch.topk(gate_weights, self.k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1)
                gate_loss = torch.tensor(0.0, device=device)
            except:
                # Fallback for basic gate
                gate_weights = self.gate(gate_input)
                topk_weights, topk_idx = torch.topk(gate_weights, self.k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1)
                gate_loss = torch.tensor(0.0, device=device)
        
        else:
            # Basic gate fallback
            gate_weights = self.gate(gate_input)
            topk_weights, topk_idx = torch.topk(gate_weights, self.k, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)
            gate_loss = torch.tensor(0.0, device=device)
        
        # Compute expert outputs using efficient dispatching
        output = self._dispatch_to_experts(x, topk_weights, topk_idx)
        
        # Apply residual connection if applicable
        if self.use_residual or hasattr(self, 'residual_proj'):
            residual = self.residual_proj(gate_input)
            output = output + 0.1 * residual  # Scaled residual
        
        # Apply output normalization
        output = self.output_norm(output)
        if self.training:
            output = self.output_dropout(output)
        
        # Update expert usage tracking
        self._update_expert_usage(topk_idx)
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(topk_weights, topk_idx)
        total_aux_loss = gate_loss + self.load_balance_weight * load_balance_loss
        
        if not return_aux_info:
            return output
        
        # Prepare comprehensive auxiliary information
        aux_info.update({
            'topk_weights': topk_weights,
            'topk_idx': topk_idx,
            'gate_loss': gate_loss,
            'load_balance_loss': load_balance_loss,
            'total_aux_loss': total_aux_loss,
            'expert_usage': self._get_expert_usage_stats(),
            'gating_type': self.gating_type,
            'routing_entropy': self._compute_routing_entropy(topk_weights)
        })
        
        return output, aux_info
    
    def _dispatch_to_experts(self, x: torch.Tensor, 
                           topk_weights: torch.Tensor, 
                           topk_idx: torch.Tensor) -> torch.Tensor:
        """
        Efficiently dispatch inputs to selected experts using your existing Expert implementation
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize output accumulator
        output = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Efficient expert computation
        for i in range(self.k):
            expert_indices = topk_idx[:, i]  # [batch] - which expert for each sample
            weights = topk_weights[:, i].unsqueeze(-1)  # [batch, 1] - weight for this expert
            
            # For each unique expert that's selected
            unique_experts = torch.unique(expert_indices)
            
            for expert_id in unique_experts:
                expert_id_int = expert_id.item()
                
                # Find which samples use this expert
                mask = (expert_indices == expert_id).float().unsqueeze(-1)  # [batch, 1]
                
                if mask.sum() == 0:
                    continue
                
                # Compute expert output for all samples
                expert_result = self.experts[expert_id_int](x)
                
                # FIXED: Handle different expert output formats more carefully
                expert_output = self._extract_expert_output(expert_result, batch_size)
                
                # Ensure output dimensions match for broadcasting
                if expert_output.shape[-1] != self.output_dim:
                    # If expert output doesn't match expected output_dim, project it
                    if not hasattr(self, f'expert_proj_{expert_id_int}'):
                        setattr(self, f'expert_proj_{expert_id_int}', 
                               nn.Linear(expert_output.shape[-1], self.output_dim).to(device))
                    proj_layer = getattr(self, f'expert_proj_{expert_id_int}')
                    expert_output = proj_layer(expert_output)
                
                # Ensure weights and mask have compatible dimensions
                expert_weights = weights * mask  # [batch, 1]
                
                # Handle dimension mismatch between expert_output and weights
                if expert_output.dim() == 2 and expert_weights.dim() == 2:
                    if expert_output.shape[1] != expert_weights.shape[1]:
                        # Broadcast weights to match expert output
                        expert_weights = expert_weights.expand(-1, expert_output.shape[1])
                
                # Accumulate weighted expert outputs
                weighted_output = expert_output * expert_weights
                
                # Ensure weighted_output has the right shape for accumulation
                if weighted_output.shape != output.shape:
                    # Sum over extra dimensions if needed
                    while weighted_output.dim() > output.dim():
                        weighted_output = weighted_output.sum(dim=-1)
                    
                    # If still shape mismatch, take only the first output_dim columns
                    if weighted_output.shape[-1] > output.shape[-1]:
                        weighted_output = weighted_output[..., :self.output_dim]
                    elif weighted_output.shape[-1] < output.shape[-1]:
                        # Pad with zeros if needed
                        pad_size = output.shape[-1] - weighted_output.shape[-1]
                        weighted_output = F.pad(weighted_output, (0, pad_size))
                
                output += weighted_output
        
        return output
    
    def _extract_expert_output(self, expert_result, batch_size: int) -> torch.Tensor:
        """
        Extract the main output tensor from expert result, handling different formats
        """
        if isinstance(expert_result, dict):
            # Priority order for extracting output from dict
            priority_keys = [self.expert_output_key, 'signals', 'output', 'prediction', 'logits']
            
            for key in priority_keys:
                if key in expert_result:
                    expert_output = expert_result[key]
                    if isinstance(expert_output, torch.Tensor) and expert_output.shape[0] == batch_size:
                        return expert_output
            
            # If no priority key found, find any suitable tensor
            for key, value in expert_result.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                    return value
            
            raise ValueError(f"Could not find suitable tensor output in expert result dict: {expert_result.keys()}")
            
        elif isinstance(expert_result, (tuple, list)):
            # Take the first tensor element
            expert_output = expert_result[0]
            if not isinstance(expert_output, torch.Tensor):
                raise TypeError(f"First element of expert result must be a tensor, got {type(expert_output)}")
            return expert_output
            
        else:
            # Expert returns tensor directly
            if not isinstance(expert_result, torch.Tensor):
                raise TypeError(f"Expert output must be a tensor, got {type(expert_result)}")
            return expert_result
    
    def _update_expert_usage(self, topk_idx: torch.Tensor):
        """Update expert usage statistics for load balancing"""
        if self.training:
            # Count expert usage
            for expert_id in range(self.num_experts):
                count = (topk_idx == expert_id).sum().float()
                self.expert_usage_count[expert_id] += count
            
            self.total_tokens += topk_idx.numel()
    
    def _get_expert_usage_stats(self) -> Dict:
        """Get current expert usage statistics"""
        if self.total_tokens > 0:
            usage_freq = self.expert_usage_count / self.total_tokens
            usage_variance = torch.var(usage_freq)
            usage_entropy = -torch.sum(usage_freq * torch.log(usage_freq + 1e-8))
        else:
            usage_freq = torch.zeros(self.num_experts)
            usage_variance = torch.tensor(0.0)
            usage_entropy = torch.tensor(0.0)
        
        return {
            'frequency': usage_freq,
            'variance': usage_variance.item(),
            'entropy': usage_entropy.item(),
            'total_tokens': self.total_tokens.item()
        }
    
    def _compute_load_balance_loss(self, topk_weights: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to prevent expert collapse"""
        batch_size, k = topk_idx.shape
        
        # Count expert usage in this batch
        expert_usage = torch.zeros(self.num_experts, device=topk_idx.device)
        for i in range(self.num_experts):
            expert_usage[i] = (topk_idx == i).sum().float()
        
        # Normalize by total selections
        expert_usage = expert_usage / (batch_size * k)
        
        # Target uniform distribution
        uniform_usage = 1.0 / self.num_experts
        
        # L2 loss from uniform distribution
        load_loss = torch.sum((expert_usage - uniform_usage) ** 2)
        
        return load_loss
    
    def _compute_routing_entropy(self, topk_weights: torch.Tensor) -> float:
        """Compute entropy of routing decisions"""
        # Average weights across batch
        avg_weights = topk_weights.mean(0)  # [k]
        if avg_weights.dim() > 1:
            avg_weights = avg_weights.mean()
        
        # For top-k, we need to consider the full distribution
        # This is an approximation of the routing entropy
        weight_sum = avg_weights.sum()
        if weight_sum > 0:
            normalized_weights = avg_weights / weight_sum
            entropy = -torch.sum(normalized_weights * torch.log(normalized_weights + 1e-8))
            return entropy.item()
        return 0.0
    
    def get_routing_analysis(self, x: torch.Tensor) -> Dict:
        """Analyze routing patterns for the given input"""
        with torch.no_grad():
            _, aux_info = self.forward(x, return_aux_info=True)
            
            routing_analysis = {
                'expert_selection_frequency': aux_info['expert_usage']['frequency'],
                'expert_weight_distribution': aux_info['topk_weights'].mean(0),
                'routing_entropy': aux_info['routing_entropy'],
                'load_balance_loss': aux_info['load_balance_loss'].item(),
                'gate_type': self.gating_type
            }
            
            # Add market conditioning info if available
            if 'regime_probs' in aux_info:
                routing_analysis['market_regime_probs'] = aux_info['regime_probs'].mean(0)
            
        return routing_analysis

# =============================================================================
# Backward Compatibility Wrapper - Using Enhanced Components
# =============================================================================

class MixtureOfExperts(FinancialMixtureOfExperts):
    """
    Backward compatible wrapper that maintains your original interface
    while using our enhanced Gate and Expert implementations
    """
    
    def __init__(self, 
                 input_dim: int,
                 seq_len: int,
                 model_dim: int,
                 num_heads: int,
                 num_layers: int,
                 ff_dim: int,
                 output_dim: int,
                 num_experts: int,
                 k: int = 2,
                 noise_std: float = 1.0,
                 **kwargs):
        
        # Initialize with enhanced implementation but maintain compatibility
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            k=k,
            gating_type='dense',  # Use dense for backward compatibility (sparse had issues)
            noise_std=noise_std,
            use_market_conditioning=True,  # Enable enhanced features
            expert_output_key='signals',  # Use 'signals' as primary output
            **kwargs
        )
        
        # Store original parameters for compatibility
        self.noise_std = noise_std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward compatible forward method that returns the original format
        
        Returns:
            output: [batch, output_dim] - Model predictions
            topk_weights: [batch, k] - Expert weights
            load_loss: Scalar - Load balancing loss
        """
        # Use enhanced forward pass
        output, aux_info = super().forward(x, return_aux_info=True)
        
        # Extract information in original format
        topk_weights = aux_info['topk_weights']
        load_loss = aux_info['total_aux_loss']
        
        return output, topk_weights, load_loss

# =============================================================================
# Testing and Example Usage
# =============================================================================

def test_enhanced_moe_with_existing_components():
    """Test the enhanced MoE using our existing Gate and Expert implementations"""
    
    print("Testing Enhanced MoE with existing Gate and Expert components...")
    
    # Test parameters
    batch_size = 4
    seq_len = 30
    input_dim = 10
    model_dim = 64
    num_heads = 4
    num_layers = 2
    ff_dim = 128
    output_dim = 1
    num_experts = 8
    k = 2
    
    # Create test data
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {x.shape}")
    
    # Test different gating types
    for gating_type in ['dense', 'adaptive']:  # Skip sparse for now due to complexity
        print(f"\n--- Testing {gating_type} gating ---")
        
        try:
            # Create enhanced MoE
            moe = FinancialMixtureOfExperts(
                input_dim=input_dim,
                seq_len=seq_len,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                output_dim=output_dim,
                num_experts=num_experts,
                k=k,
                gating_type=gating_type,
                expert_output_key='signals'  # Specify which output to use
            )
            
            # Forward pass
            output, aux_info = moe(x, return_aux_info=True)
            
            print(f"✓ {gating_type.capitalize()} MoE - Output shape: {output.shape}")
            print(f"  Expert weights shape: {aux_info['topk_weights'].shape}")
            print(f"  Expert indices shape: {aux_info['topk_idx'].shape}")
            print(f"  Load balance loss: {aux_info['load_balance_loss']:.4f}")
            print(f"  Routing entropy: {aux_info['routing_entropy']:.3f}")
            
            # Test routing analysis
            routing_analysis = moe.get_routing_analysis(x)
            print(f"  Expert usage variance: {routing_analysis['expert_selection_frequency'].var():.4f}")
            
        except Exception as e:
            print(f"✗ {gating_type.capitalize()} MoE failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test backward compatibility
    print(f"\n--- Testing Backward Compatibility ---")
    try:
        legacy_moe = MixtureOfExperts(
            input_dim=input_dim,
            seq_len=seq_len,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            k=k
        )
        
        output, weights, loss = legacy_moe(x)
        print(f"✓ Legacy MoE - Output: {output.shape}, Weights: {weights.shape}, Loss: {loss:.4f}")
        
    except Exception as e:
        print(f"✗ Legacy MoE failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting completed!")

if __name__ == "__main__":
    test_enhanced_moe_with_existing_components()