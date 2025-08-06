"""
Helper functions for DP-Flash-Attention model conversion.
"""

import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor


def _extract_attention_dims(module):
    """Extract embed_dim and num_heads from attention module."""
    # BERT/RoBERTa style
    if hasattr(module, 'num_attention_heads') and hasattr(module, 'attention_head_size'):
        num_heads = module.num_attention_heads
        embed_dim = num_heads * module.attention_head_size
        return embed_dim, num_heads
    
    # GPT/T5 style
    if hasattr(module, 'n_head') and hasattr(module, 'd_model'):
        return module.d_model, module.n_head
    
    # PyTorch MultiheadAttention
    if hasattr(module, 'embed_dim') and hasattr(module, 'num_heads'):
        return module.embed_dim, module.num_heads
    
    # Try to infer from linear layers
    if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
        q_weight = module.q_proj.weight if hasattr(module.q_proj, 'weight') else None
        if q_weight is not None:
            embed_dim = q_weight.shape[1]
            # Assume standard head dimensions
            if embed_dim % 64 == 0:
                num_heads = embed_dim // 64
                return embed_dim, num_heads
            elif embed_dim % 32 == 0:
                num_heads = embed_dim // 32
                return embed_dim, num_heads
    
    return None, None


def _copy_attention_weights(source_module, target_module):
    """Copy weights from source attention to DP attention."""
    try:
        # Copy projection weights if they exist
        if hasattr(source_module, 'q_proj') and hasattr(target_module, 'q_proj'):
            target_module.q_proj.weight.data.copy_(source_module.q_proj.weight.data)
            if source_module.q_proj.bias is not None and target_module.q_proj.bias is not None:
                target_module.q_proj.bias.data.copy_(source_module.q_proj.bias.data)
        
        if hasattr(source_module, 'k_proj') and hasattr(target_module, 'k_proj'):
            target_module.k_proj.weight.data.copy_(source_module.k_proj.weight.data)
            if source_module.k_proj.bias is not None and target_module.k_proj.bias is not None:
                target_module.k_proj.bias.data.copy_(source_module.k_proj.bias.data)
        
        if hasattr(source_module, 'v_proj') and hasattr(target_module, 'v_proj'):
            target_module.v_proj.weight.data.copy_(source_module.v_proj.weight.data)
            if source_module.v_proj.bias is not None and target_module.v_proj.bias is not None:
                target_module.v_proj.bias.data.copy_(source_module.v_proj.bias.data)
        
        if hasattr(source_module, 'out_proj') and hasattr(target_module, 'out_proj'):
            target_module.out_proj.weight.data.copy_(source_module.out_proj.weight.data)
            if source_module.out_proj.bias is not None and target_module.out_proj.bias is not None:
                target_module.out_proj.bias.data.copy_(source_module.out_proj.bias.data)
                
    except Exception as e:
        warnings.warn(f"Could not copy all weights: {e}")


def _replace_module(model, module_name, new_module):
    """Replace a module in the model by name."""
    parts = module_name.split('.')
    parent = model
    
    # Navigate to parent module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    # Replace the target module
    setattr(parent, parts[-1], new_module)


def create_attention_mask(seq_len: int, causal: bool = False, device: torch.device = None) -> Optional[Tensor]:
    """Create attention mask for DP-Flash-Attention."""
    if not causal:
        return None
    
    device = device or torch.device('cpu')
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def estimate_privacy_cost(
    epsilon_per_step: float,
    num_steps: int, 
    num_layers: int,
    composition_type: str = 'basic'
) -> float:
    """Estimate total privacy cost for training."""
    if composition_type == 'basic':
        return epsilon_per_step * num_steps * num_layers
    elif composition_type == 'advanced':
        # Use RDP composition (simplified)
        return epsilon_per_step * num_layers * (num_steps ** 0.5)
    else:
        raise ValueError(f"Unknown composition type: {composition_type}")


def validate_attention_inputs(q: Tensor, k: Tensor, v: Tensor) -> None:
    """Validate attention input tensors."""
    if not all(isinstance(t, torch.Tensor) for t in [q, k, v]):
        raise TypeError("All inputs must be torch.Tensor")
    
    if q.device != k.device or q.device != v.device:
        raise ValueError("All tensors must be on the same device")
    
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("All tensors must have the same dtype")
    
    if len(q.shape) != 4:
        raise ValueError(f"Expected 4D tensors, got {len(q.shape)}D")
    
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, V tensors must have the same shape")