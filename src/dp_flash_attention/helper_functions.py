"""
Helper functions for DP-Flash-Attention operations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, Union
import warnings
import math


def estimate_memory_usage(
    batch_size: int, 
    seq_len: int, 
    num_heads: int, 
    head_dim: int,
    dtype: torch.dtype = torch.float16
) -> Dict[str, float]:
    """
    Estimate memory usage for attention computation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Tensor data type
        
    Returns:
        Dictionary with memory estimates in MB
    """
    element_size = torch.tensor(0, dtype=dtype).element_size()
    
    # Input tensors (Q, K, V)
    input_size = 3 * batch_size * seq_len * num_heads * head_dim * element_size
    
    # Attention scores
    scores_size = batch_size * num_heads * seq_len * seq_len * element_size
    
    # Output tensor
    output_size = batch_size * seq_len * num_heads * head_dim * element_size
    
    # Noise tensors for DP
    noise_size = scores_size  # Noise same size as scores
    
    # Gradient storage (rough estimate)
    grad_size = input_size + output_size
    
    total_bytes = input_size + scores_size + output_size + noise_size + grad_size
    total_mb = total_bytes / (1024 * 1024)
    
    return {
        'input_mb': input_size / (1024 * 1024),
        'scores_mb': scores_size / (1024 * 1024), 
        'output_mb': output_size / (1024 * 1024),
        'noise_mb': noise_size / (1024 * 1024),
        'gradients_mb': grad_size / (1024 * 1024),
        'total_estimated_mb': total_mb,
    }


def _extract_attention_dims(module: nn.Module) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract embed_dim and num_heads from various attention module types.
    
    Args:
        module: Attention module to analyze
        
    Returns:
        Tuple of (embed_dim, num_heads) or (None, None) if extraction fails
    """
    try:
        # PyTorch MultiheadAttention
        if hasattr(module, 'embed_dim') and hasattr(module, 'num_heads'):
            return module.embed_dim, module.num_heads
        
        # Transformers library attention modules
        if hasattr(module, 'config'):
            config = module.config
            if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
                return config.hidden_size, config.num_attention_heads
        
        # HuggingFace attention patterns
        if hasattr(module, 'attention'):
            attn = module.attention
            if hasattr(attn, 'config'):
                config = attn.config
                if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
                    return config.hidden_size, config.num_attention_heads
        
        # BERT/RoBERTa style
        if hasattr(module, 'num_attention_heads') and hasattr(module, 'attention_head_size'):
            num_heads = module.num_attention_heads
            embed_dim = num_heads * module.attention_head_size
            return embed_dim, num_heads
        
        # GPT/T5 style
        if hasattr(module, 'n_head') and hasattr(module, 'd_model'):
            return module.d_model, module.n_head
            
        # Common dimension attributes
        if hasattr(module, 'hidden_size') and hasattr(module, 'num_heads'):
            return module.hidden_size, module.num_heads
            
        # Try to infer from linear layer dimensions
        for name, submodule in module.named_modules():
            if isinstance(submodule, nn.Linear):
                if 'q_proj' in name or 'query' in name:
                    embed_dim = submodule.in_features
                    out_features = submodule.out_features
                    if out_features == embed_dim:
                        # Assume standard head dimensions
                        if embed_dim % 64 == 0:
                            return embed_dim, embed_dim // 64
                        elif embed_dim % 32 == 0:
                            return embed_dim, embed_dim // 32
                elif 'in_proj' in name:
                    # For combined QKV projection
                    embed_dim = submodule.in_features
                    out_features = submodule.out_features
                    if out_features == 3 * embed_dim:
                        # Standard transformer attention
                        if embed_dim % 64 == 0:
                            return embed_dim, embed_dim // 64
                        elif embed_dim % 32 == 0:
                            return embed_dim, embed_dim // 32
                        
    except Exception as e:
        warnings.warn(f"Error extracting attention dimensions: {e}")
    
    return None, None


def _copy_attention_weights(source_module: nn.Module, target_module: nn.Module) -> bool:
    """
    Copy weights from source attention module to target DP attention module.
    
    Args:
        source_module: Original attention module
        target_module: Target DP attention module
        
    Returns:
        True if weights were successfully copied, False otherwise
    """
    try:
        # Handle PyTorch MultiheadAttention
        if hasattr(source_module, 'in_proj_weight'):
            # Combined QKV projection
            weight = source_module.in_proj_weight
            embed_dim = weight.size(1)
            
            # Split into Q, K, V projections
            q_weight = weight[:embed_dim, :]
            k_weight = weight[embed_dim:2*embed_dim, :]
            v_weight = weight[2*embed_dim:, :]
            
            target_module.q_proj.weight.data.copy_(q_weight)
            target_module.k_proj.weight.data.copy_(k_weight)
            target_module.v_proj.weight.data.copy_(v_weight)
            
            if hasattr(source_module, 'in_proj_bias') and source_module.in_proj_bias is not None:
                bias = source_module.in_proj_bias
                q_bias = bias[:embed_dim]
                k_bias = bias[embed_dim:2*embed_dim]
                v_bias = bias[2*embed_dim:]
                
                target_module.q_proj.bias.data.copy_(q_bias)
                target_module.k_proj.bias.data.copy_(k_bias)
                target_module.v_proj.bias.data.copy_(v_bias)
            
            # Output projection
            if hasattr(source_module, 'out_proj'):
                target_module.out_proj.weight.data.copy_(source_module.out_proj.weight)
                if source_module.out_proj.bias is not None:
                    target_module.out_proj.bias.data.copy_(source_module.out_proj.bias)
                    
            return True
            
        # Handle separate Q, K, V projections
        elif hasattr(source_module, 'q_proj') and hasattr(source_module, 'k_proj') and hasattr(source_module, 'v_proj'):
            target_module.q_proj.weight.data.copy_(source_module.q_proj.weight)
            target_module.k_proj.weight.data.copy_(source_module.k_proj.weight)
            target_module.v_proj.weight.data.copy_(source_module.v_proj.weight)
            
            if hasattr(source_module.q_proj, 'bias') and source_module.q_proj.bias is not None:
                target_module.q_proj.bias.data.copy_(source_module.q_proj.bias)
                target_module.k_proj.bias.data.copy_(source_module.k_proj.bias)
                target_module.v_proj.bias.data.copy_(source_module.v_proj.bias)
            
            if hasattr(source_module, 'out_proj'):
                target_module.out_proj.weight.data.copy_(source_module.out_proj.weight)
                if source_module.out_proj.bias is not None:
                    target_module.out_proj.bias.data.copy_(source_module.out_proj.bias)
                    
            return True
            
    except Exception as e:
        warnings.warn(f"Failed to copy attention weights: {e}")
        
    return False


def _replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> bool:
    """
    Replace a module in a PyTorch model with a new module.
    
    Args:
        model: The model containing the module to replace
        module_name: Dot-separated path to the module (e.g., 'encoder.layer.0.attention')
        new_module: The new module to insert
        
    Returns:
        True if replacement was successful, False otherwise
    """
    try:
        # Split the module name into components
        components = module_name.split('.')
        
        # Navigate to the parent module
        parent = model
        for component in components[:-1]:
            parent = getattr(parent, component)
        
        # Replace the final module
        setattr(parent, components[-1], new_module)
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to replace module {module_name}: {e}")
        return False


def create_attention_mask(
    seq_len: int, 
    causal: bool = False, 
    window_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Create attention mask for various attention patterns.
    
    Args:
        seq_len: Sequence length
        causal: Whether to create causal (lower-triangular) mask
        window_size: Local attention window size (None for full attention)
        device: Device to place mask on
        dtype: Data type for mask
        
    Returns:
        Attention mask tensor
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.bool
        
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    if causal:
        # Causal mask: upper triangular is True (masked out)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    
    if window_size is not None and window_size > 0:
        # Local attention mask
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, :start] = True
            mask[i, end:] = True
    
    return mask


def estimate_privacy_cost(
    epsilon: float,
    delta: float,
    num_steps: int,
    batch_size: int,
    dataset_size: int,
    noise_multiplier: float = 1.0
) -> Dict[str, float]:
    """
    Estimate privacy cost for a training run.
    
    Args:
        epsilon: Privacy budget parameter
        delta: Privacy parameter
        num_steps: Number of training steps
        batch_size: Batch size
        dataset_size: Size of dataset
        noise_multiplier: Noise multiplier for DP-SGD
        
    Returns:
        Dictionary with privacy cost estimates
    """
    sampling_rate = batch_size / dataset_size
    
    # Simple privacy accounting (would use more sophisticated methods in practice)
    per_step_epsilon = epsilon / num_steps
    total_privacy_cost = per_step_epsilon * num_steps
    
    # Privacy amplification by subsampling (simplified)
    amplification_factor = 1.0
    if sampling_rate < 1.0:
        amplification_factor = sampling_rate
    
    effective_epsilon = total_privacy_cost * amplification_factor
    
    return {
        'total_epsilon': epsilon,
        'per_step_epsilon': per_step_epsilon,
        'effective_epsilon': effective_epsilon,
        'delta': delta,
        'sampling_rate': sampling_rate,
        'amplification_factor': amplification_factor,
        'num_steps': num_steps,
        'noise_multiplier': noise_multiplier,
    }


def validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    epsilon: float,
    delta: float,
    max_grad_norm: float
) -> None:
    """
    Validate inputs for attention computation.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        epsilon: Privacy budget parameter
        delta: Privacy parameter
        max_grad_norm: Maximum gradient norm
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Shape validation
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Q, K, V must have same shape. Got Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    
    if q.dim() != 4:
        raise ValueError(f"Expected 4D tensors [batch, seq_len, num_heads, head_dim], got {q.dim()}D")
    
    # Device consistency
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, V tensors must be on the same device")
    
    # Data type consistency
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, V tensors must have the same dtype")
    
    # Privacy parameter validation
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    
    if delta < 0 or delta >= 1:
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
    
    # Tensor value validation
    if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
        raise ValueError("Input tensors contain NaN values")
    
    if torch.isinf(q).any() or torch.isinf(k).any() or torch.isinf(v).any():
        raise ValueError("Input tensors contain infinite values")


def compute_attention_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> Dict[str, float]:
    """
    Compute statistics about attention inputs.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        
    Returns:
        Dictionary with input statistics
    """
    with torch.no_grad():
        stats = {
            'q_mean': q.mean().item(),
            'q_std': q.std().item(),
            'q_min': q.min().item(),
            'q_max': q.max().item(),
            'k_mean': k.mean().item(),
            'k_std': k.std().item(),
            'k_min': k.min().item(),
            'k_max': k.max().item(),
            'v_mean': v.mean().item(),
            'v_std': v.std().item(),
            'v_min': v.min().item(),
            'v_max': v.max().item(),
        }
        
        # Compute attention score statistics
        batch_size, seq_len, num_heads, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)
        
        # Sample a few attention scores for stats
        sample_scores = torch.matmul(q[:1, :min(16, seq_len)], k[:1, :min(16, seq_len)].transpose(-2, -1)) * scale
        
        stats.update({
            'sample_scores_mean': sample_scores.mean().item(),
            'sample_scores_std': sample_scores.std().item(),
            'sample_scores_min': sample_scores.min().item(),
            'sample_scores_max': sample_scores.max().item(),
        })
        
    return stats