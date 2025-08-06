"""
Core DP-Flash-Attention implementation with integrated differential privacy.
"""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .privacy import RenyiAccountant, PrivacyStats
from .kernels import dp_flash_attention_kernel
from .utils import validate_privacy_params, compute_noise_scale
from .error_handling import (handle_errors, validate_privacy_parameters, 
                            validate_tensor_inputs, check_memory_usage, 
                            PrivacyParameterError, TensorShapeError)
from .logging_utils import get_logger, PerformanceMonitor
from .security import get_input_validator, get_privacy_auditor


class DPFlashAttention(nn.Module):
    """
    Differentially Private Flash-Attention implementation.
    
    Integrates Rényi differential privacy directly into Flash-Attention 3 kernels
    with zero overhead compared to post-hoc gradient clipping approaches.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads  
        epsilon: Privacy budget parameter (ε)
        delta: Privacy parameter (δ) 
        max_grad_norm: Clipping threshold for per-sample gradients
        head_epsilons: Optional per-head privacy budgets
        dropout: Dropout probability
        bias: Whether to use bias in projections
        batch_first: If True, batch dimension is first
        device: Device to place parameters on
        dtype: Data type for parameters
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        head_epsilons: Optional[list] = None,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Validate inputs
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        
        validate_privacy_parameters(epsilon, delta, strict=True)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.dropout = dropout
        self.bias = bias
        self.batch_first = batch_first
        
        # Per-head privacy budgets (if specified)
        if head_epsilons is not None:
            if len(head_epsilons) != num_heads:
                raise ValueError(f"head_epsilons length {len(head_epsilons)} != num_heads {num_heads}")
            self.head_epsilons = torch.tensor(head_epsilons, dtype=torch.float32)
        else:
            self.head_epsilons = torch.full((num_heads,), epsilon, dtype=torch.float32)
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        
        # Privacy accounting
        self.privacy_accountant = RenyiAccountant()
        self.privacy_spent = 0.0
        
        # Dropout layer
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = nn.Identity()
            
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)  
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.bias:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    @handle_errors(reraise=True, log_errors=True)
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        return_privacy_stats: bool = False,
        causal: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]], Tuple[Tensor, PrivacyStats]]:
        """
        Forward pass with integrated differential privacy.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim] or [seq_len, batch_size, embed_dim]
            key: Key tensor with same shape as query
            value: Value tensor with same shape as query  
            key_padding_mask: Optional mask for padding tokens
            need_weights: Whether to return attention weights
            attn_mask: Optional attention mask
            return_privacy_stats: Whether to return privacy statistics
            causal: Whether to apply causal masking
            
        Returns:
            Output tensor and optionally attention weights or privacy stats
        """
        # Enhanced input validation
        validate_tensor_inputs([query, key, value], ['query', 'key', 'value'])
        
        # Check tensor compatibility
        if query.shape != key.shape or query.shape != value.shape:
            raise TensorShapeError(
                f"Q, K, V tensors must have same shape. Got Q: {query.shape}, K: {key.shape}, V: {value.shape}",
                expected_shape=query.shape,
                actual_shape=(key.shape, value.shape)
            )
        
        # Get logger for performance monitoring
        logger = get_logger()
        
        with PerformanceMonitor("dp_attention_forward", logger, log_memory=True) as monitor:
            if not self.batch_first:
                # Convert seq_first to batch_first
                query = query.transpose(0, 1)
                key = key.transpose(0, 1) 
                value = value.transpose(0, 1)
            
            batch_size, seq_len, embed_dim = query.shape
            
            # Memory usage check
            from .utils import estimate_memory_usage
            memory_est = estimate_memory_usage(batch_size, seq_len, self.num_heads, self.head_dim)
            check_memory_usage(memory_est['total_estimated_mb'], device=str(query.device))
            
            # Apply projections
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Compute noise scale for differential privacy
            noise_scale = compute_noise_scale(
                self.epsilon, self.delta, self.max_grad_norm, seq_len
            )
            
            # Call optimized CUDA kernel with integrated DP
            attn_output, grad_norm = dp_flash_attention_kernel(
                q, k, v,
                epsilon=self.epsilon,
                delta=self.delta,
                max_grad_norm=self.max_grad_norm,
                noise_scale=noise_scale,
                causal=causal,
                scale=self.scale,
            )
            
            # Reshape output
            attn_output = attn_output.view(batch_size, seq_len, embed_dim)
            
            # Apply output projection
            output = self.out_proj(attn_output)
            
            # Apply dropout
            output = self.dropout_layer(output)
            
            # Update privacy accounting
            step_epsilon = self.privacy_accountant.add_step(
                noise_scale, self.delta, batch_size, seq_len
            )
            self.privacy_spent += step_epsilon
            
            # Log privacy metrics
            logger.log_privacy_step(
                epsilon_spent=step_epsilon,
                delta=self.delta,
                noise_scale=noise_scale,
                gradient_norm=grad_norm,
                clipping_bound=self.max_grad_norm,
                additional_info={
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'num_heads': self.num_heads
                }
            )
            
            # Privacy auditing
            auditor = get_privacy_auditor()
            audit_result = auditor.audit_privacy_step(
                epsilon_spent=step_epsilon,
                delta=self.delta,
                noise_scale=noise_scale,
                gradient_norm=grad_norm,
                clipping_bound=self.max_grad_norm
            )
            
            if audit_result['issues']:
                logger.log_security_event(
                    event_type='privacy_audit_warning',
                    severity='medium',
                    description=f"Privacy audit found {len(audit_result['issues'])} issues",
                    additional_data={'issues': audit_result['issues']}
                )
            
            # Convert back to seq_first if needed
            if not self.batch_first:
                output = output.transpose(0, 1)
            
            # Prepare return values
            if return_privacy_stats:
                privacy_stats = PrivacyStats(
                    epsilon_spent=self.privacy_spent,
                    delta=self.delta,
                    grad_norm=grad_norm,
                    noise_scale=noise_scale,
                    step_epsilon=step_epsilon,
                )
                return output, privacy_stats
            
            if need_weights:
                # Note: attention weights not returned for privacy reasons
                warnings.warn(
                    "Attention weights not returned to preserve differential privacy. "
                    "Returning None for attention weights."
                )
                return output, None
            
            return output
    
    def get_privacy_spent(self) -> float:
        """Get total privacy budget spent so far."""
        return self.privacy_spent
    
    def reset_privacy_accounting(self):
        """Reset privacy accounting to start fresh."""
        self.privacy_accountant.reset()
        self.privacy_spent = 0.0
    
    def set_privacy_params(self, epsilon: float = None, delta: float = None, 
                          max_grad_norm: float = None):
        """Update privacy parameters."""
        if epsilon is not None:
            validate_privacy_parameters(epsilon, self.delta, strict=True)
            self.epsilon = epsilon
        if delta is not None:
            validate_privacy_parameters(self.epsilon, delta, strict=True)
            self.delta = delta
        if max_grad_norm is not None:
            if max_grad_norm <= 0:
                raise PrivacyParameterError(
                    f"max_grad_norm must be positive, got {max_grad_norm}"
                )
            self.max_grad_norm = max_grad_norm
    
    def extra_repr(self) -> str:
        """String representation of module."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"epsilon={self.epsilon}, delta={self.delta}, "
            f"max_grad_norm={self.max_grad_norm}, dropout={self.dropout}"
        )


class DPCheckpointedAttention(DPFlashAttention):
    """
    Memory-efficient version with gradient checkpointing.
    
    Trades computation for memory while preserving differential privacy guarantees.
    """
    
    def __init__(self, segments: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.segments = segments
    
    def forward(self, *args, **kwargs):
        """Forward pass with gradient checkpointing."""
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                super().forward, *args, **kwargs
            )
        else:
            return super().forward(*args, **kwargs)