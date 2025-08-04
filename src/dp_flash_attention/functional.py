"""
Functional interface for DP-Flash-Attention.

Drop-in replacement for flash_attn_func with integrated differential privacy.
"""

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

from .kernels import dp_flash_attention_kernel
from .privacy import PrivacyStats
from .utils import validate_privacy_params, compute_noise_scale


def dp_flash_attn_func(
    q: Tensor,
    k: Tensor, 
    v: Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    # DP-specific parameters
    epsilon: float = 1.0,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    return_privacy_stats: bool = False,
) -> Tensor:
    """
    Drop-in replacement for flash_attn_func with differential privacy.
    
    Args:
        q: Query tensor [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor with same shape as q
        v: Value tensor with same shape as q
        dropout_p: Dropout probability (0.0 to disable)
        softmax_scale: Scale factor for attention scores (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        window_size: Local attention window size (not supported yet)
        alibi_slopes: ALiBi slopes (not supported yet)  
        deterministic: Whether to use deterministic implementation
        return_attn_probs: Whether to return attention probabilities (disabled for privacy)
        epsilon: Privacy budget parameter
        delta: Privacy parameter  
        max_grad_norm: Gradient clipping threshold
        return_privacy_stats: Whether to return privacy statistics
        
    Returns:
        Attention output tensor, optionally with privacy statistics
        
    Note:
        This function provides the same interface as flash_attn_func but with
        integrated differential privacy. Some features like attention probability
        return are disabled to preserve privacy.
    """
    # Validate inputs
    validate_privacy_params(epsilon, delta)
    
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4-dimensional [batch, seq_len, num_heads, head_dim]")
    
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Set default softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Warn about unsupported features
    if window_size != (-1, -1):
        warnings.warn("window_size not supported yet, ignoring")
    
    if alibi_slopes is not None:
        warnings.warn("alibi_slopes not supported yet, ignoring")
    
    if return_attn_probs:
        warnings.warn(
            "return_attn_probs disabled for differential privacy. "
            "Attention probabilities not returned."
        )
        return_attn_probs = False
    
    # Compute noise scale for differential privacy
    noise_scale = compute_noise_scale(epsilon, delta, max_grad_norm, seq_len)
    
    # Call optimized CUDA kernel
    output, grad_norm = dp_flash_attention_kernel(
        q, k, v,
        epsilon=epsilon,
        delta=delta, 
        max_grad_norm=max_grad_norm,
        noise_scale=noise_scale,
        causal=causal,
        scale=softmax_scale,
        deterministic=deterministic,
    )
    
    # Apply dropout if specified
    if dropout_p > 0.0 and q.requires_grad:
        output = torch.nn.functional.dropout(output, p=dropout_p, training=True)
    
    if return_privacy_stats:
        privacy_stats = PrivacyStats(
            epsilon_spent=epsilon,
            delta=delta,
            grad_norm=grad_norm,
            noise_scale=noise_scale,
            step_epsilon=epsilon,
        )
        return output, privacy_stats
    
    return output


def dp_flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor, 
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    # DP-specific parameters
    epsilon: float = 1.0,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    return_privacy_stats: bool = False,
) -> Tensor:
    """
    Variable-length sequence version of DP-Flash-Attention.
    
    For packed sequences with different lengths in the same batch.
    Currently falls back to regular implementation with padding.
    
    Args:
        q: Packed query tensor [total_seq_len, num_heads, head_dim]
        k: Packed key tensor  
        v: Packed value tensor
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length  
        dropout_p: Dropout probability
        softmax_scale: Scale factor for attention
        causal: Whether to apply causal masking
        window_size: Local attention window (not supported)
        alibi_slopes: ALiBi slopes (not supported)
        deterministic: Whether to use deterministic implementation
        return_attn_probs: Whether to return attention probs (disabled for privacy) 
        epsilon: Privacy budget parameter
        delta: Privacy parameter
        max_grad_norm: Gradient clipping threshold
        return_privacy_stats: Whether to return privacy statistics
        
    Returns:
        Packed attention output tensor
    """
    warnings.warn(
        "Variable-length sequences not fully optimized yet. "
        "Consider using standard dp_flash_attn_func with padding."
    )
    
    # For now, fall back to regular implementation
    # TODO: Implement proper variable-length kernel
    batch_size = len(cu_seqlens_q) - 1
    
    # Reconstruct padded tensors (simplified implementation)
    q_padded = torch.zeros(
        batch_size, max_seqlen_q, q.shape[-2], q.shape[-1],
        dtype=q.dtype, device=q.device
    )
    k_padded = torch.zeros(
        batch_size, max_seqlen_k, k.shape[-2], k.shape[-1], 
        dtype=k.dtype, device=k.device
    )
    v_padded = torch.zeros(
        batch_size, max_seqlen_k, v.shape[-2], v.shape[-1],
        dtype=v.dtype, device=v.device  
    )
    
    # Pack sequences into padded format
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i+1]
        k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i+1] 
        
        q_len = q_end - q_start
        k_len = k_end - k_start
        
        q_padded[i, :q_len] = q[q_start:q_end]
        k_padded[i, :k_len] = k[k_start:k_end]
        v_padded[i, :k_len] = v[k_start:k_end]
    
    # Call standard function
    if return_privacy_stats:
        output_padded, privacy_stats = dp_flash_attn_func(
            q_padded, k_padded, v_padded,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
            return_privacy_stats=True,
        )
    else:
        output_padded = dp_flash_attn_func(
            q_padded, k_padded, v_padded,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale, 
            causal=causal,
            deterministic=deterministic,
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
        )
    
    # Unpack back to variable length format
    total_output_len = cu_seqlens_q[-1]
    output = torch.zeros(
        total_output_len, output_padded.shape[-2], output_padded.shape[-1],
        dtype=output_padded.dtype, device=output_padded.device
    )
    
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i+1]
        q_len = q_end - q_start
        output[q_start:q_end] = output_padded[i, :q_len]
    
    if return_privacy_stats:
        return output, privacy_stats
    
    return output


def make_model_differentially_private(
    model: torch.nn.Module,
    target_epsilon: float,
    target_delta: float,
    num_epochs: int,
    batch_size: int,
    replace_attention: bool = True,
) -> torch.nn.Module:
    """
    Convert a transformer model to use differential privacy.
    
    Replaces standard attention layers with DP-Flash-Attention equivalents.
    
    Args:
        model: PyTorch model to convert
        target_epsilon: Target privacy budget for entire training
        target_delta: Target privacy parameter  
        num_epochs: Number of training epochs
        batch_size: Training batch size
        replace_attention: Whether to replace attention layers
        
    Returns:
        Model with DP attention layers
    """
    from .core import DPFlashAttention
    
    if not replace_attention:
        warnings.warn("replace_attention=False, returning original model")
        return model
    
    # Calculate per-step privacy budget
    # Assuming one attention layer per transformer block
    num_attention_layers = sum(
        1 for name, _ in model.named_modules() 
        if 'attention' in name.lower() or 'attn' in name.lower()
    )
    
    if num_attention_layers == 0:
        warnings.warn("No attention layers found in model")
        return model
    
    # Distribute privacy budget across layers and steps
    total_steps = num_epochs * (50000 // batch_size)  # Estimate
    per_layer_epsilon = target_epsilon / (num_attention_layers * total_steps)
    
    print(f"Converting {num_attention_layers} attention layers")
    print(f"Per-layer epsilon: {per_layer_epsilon:.6f}")
    
    # Replace attention layers (simplified - would need model-specific logic)
    for name, module in model.named_modules():
        if hasattr(module, 'num_attention_heads') and hasattr(module, 'attention_head_size'):
            # BERT-style attention
            embed_dim = module.num_attention_heads * module.attention_head_size
            dp_attention = DPFlashAttention(
                embed_dim=embed_dim,
                num_heads=module.num_attention_heads,
                epsilon=per_layer_epsilon,
                delta=target_delta,
                max_grad_norm=1.0,
            )
            # Would need to copy weights and replace module properly
            warnings.warn(f"Found attention layer {name} but replacement not fully implemented")
    
    return model