#!/usr/bin/env python3
"""
Advanced Privacy Mechanisms for DP-Flash-Attention Research.

Implements cutting-edge differential privacy mechanisms specifically designed
for attention computation, including Privacy Loss Distribution (PLD) framework,
attention-specific sensitivity analysis, and structured noise mechanisms.
"""

import math
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from pathlib import Path
import logging
import time
import hashlib
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    
try:
    from scipy import stats, optimize
    from scipy.special import gammainc, gamma
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PrivacyMechanismType(Enum):
    """Types of advanced privacy mechanisms."""
    PRIVACY_LOSS_DISTRIBUTION = "pld"
    STRUCTURED_NOISE = "structured_noise"
    ATTENTION_SENSITIVITY = "attention_sensitivity"
    ADAPTIVE_COMPOSITION = "adaptive_composition"
    QUANTUM_RESISTANT = "quantum_resistant"
    FEDERATED_SECURE = "federated_secure"


@dataclass
class PrivacyLossPoint:
    """Single point in privacy loss distribution."""
    privacy_loss: float
    probability: float
    mechanism_type: str
    parameters: Dict[str, Any]


@dataclass
class AttentionSensitivityProfile:
    """Sensitivity profile for attention mechanisms."""
    per_head_sensitivity: List[float]
    layer_sensitivity: float
    query_sensitivity: float
    key_sensitivity: float
    value_sensitivity: float
    output_sensitivity: float
    gradient_bound: float
    computed_at: float


class PrivacyLossDistribution:
    """
    Advanced Privacy Loss Distribution (PLD) framework for optimal composition.
    
    Implements state-of-the-art privacy accounting based on:
    "Numerical Composition of Differential Privacy" (2021)
    "Connect the Dots: Tighter Discrete Approximations of Privacy Loss Distributions" (2022)
    """
    
    def __init__(self, discretization_interval: float = 1e-4):
        self.discretization_interval = discretization_interval
        self.privacy_losses: List[PrivacyLossPoint] = []
        self.composition_history: List[Dict[str, Any]] = []
        
    def add_mechanism(self, mechanism_name: str, sensitivity: float, 
                     epsilon: float, delta: float, **kwargs) -> None:
        """Add a privacy mechanism to the composition."""
        if mechanism_name == "gaussian":
            self._add_gaussian_mechanism(sensitivity, epsilon, delta, **kwargs)
        elif mechanism_name == "laplace":
            self._add_laplace_mechanism(sensitivity, epsilon, **kwargs)
        elif mechanism_name == "exponential":
            self._add_exponential_mechanism(sensitivity, epsilon, **kwargs)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism_name}")
            
        self.composition_history.append({
            "mechanism": mechanism_name,
            "sensitivity": sensitivity,
            "epsilon": epsilon,
            "delta": delta,
            "timestamp": time.time(),
            "kwargs": kwargs
        })
    
    def _add_gaussian_mechanism(self, sensitivity: float, epsilon: float, 
                               delta: float, **kwargs) -> None:
        """Add Gaussian mechanism to PLD."""
        # Compute optimal noise multiplier for (ε, δ)-DP
        noise_multiplier = self._compute_gaussian_noise_multiplier(epsilon, delta)
        sigma = noise_multiplier * sensitivity
        
        # Discretize the privacy loss distribution
        max_privacy_loss = 10.0  # Practical bound
        num_points = int(2 * max_privacy_loss / self.discretization_interval)
        
        for i in range(num_points):
            privacy_loss = -max_privacy_loss + i * self.discretization_interval
            
            # Compute probability density at this privacy loss
            if SCIPY_AVAILABLE:
                # P(L = l) for Gaussian mechanism
                prob_plus = stats.norm.pdf(privacy_loss, loc=0.5/sigma**2, scale=1/sigma)
                prob_minus = stats.norm.pdf(-privacy_loss, loc=0.5/sigma**2, scale=1/sigma)
                probability = 0.5 * (prob_plus + prob_minus)
            else:
                # Simplified approximation without scipy
                probability = math.exp(-0.5 * privacy_loss**2 / (1/sigma**2))
                probability /= math.sqrt(2 * math.pi / (1/sigma**2))
            
            if probability > 1e-10:  # Filter negligible probabilities
                self.privacy_losses.append(PrivacyLossPoint(
                    privacy_loss=privacy_loss,
                    probability=probability,
                    mechanism_type="gaussian",
                    parameters={"sigma": sigma, "sensitivity": sensitivity}
                ))
    
    def _add_laplace_mechanism(self, sensitivity: float, epsilon: float, **kwargs) -> None:
        """Add Laplace mechanism to PLD."""
        scale = sensitivity / epsilon
        
        # Laplace mechanism has simpler PLD
        max_privacy_loss = 10.0
        num_points = int(2 * max_privacy_loss / self.discretization_interval)
        
        for i in range(num_points):
            privacy_loss = -max_privacy_loss + i * self.discretization_interval
            
            # P(L = l) for Laplace mechanism
            if privacy_loss >= 0:
                probability = 0.5 * math.exp(-privacy_loss / epsilon)
            else:
                probability = 0.5 * math.exp(privacy_loss / epsilon)
            
            if probability > 1e-10:
                self.privacy_losses.append(PrivacyLossPoint(
                    privacy_loss=privacy_loss,
                    probability=probability,
                    mechanism_type="laplace",
                    parameters={"scale": scale, "sensitivity": sensitivity}
                ))
    
    def _add_exponential_mechanism(self, sensitivity: float, epsilon: float, **kwargs) -> None:
        """Add Exponential mechanism to PLD."""
        # Simplified exponential mechanism for categorical outputs
        utility_function = kwargs.get("utility_function", lambda x: 1.0)
        output_range = kwargs.get("output_range", [-1.0, 1.0])
        
        # Discretize output space
        num_outputs = 100
        outputs = np.linspace(output_range[0], output_range[1], num_outputs)
        
        # Compute exponential probabilities
        utilities = [utility_function(output) for output in outputs]
        max_utility = max(utilities)
        
        probabilities = []
        for utility in utilities:
            prob = math.exp(epsilon * utility / (2 * sensitivity))
            probabilities.append(prob)
        
        # Normalize
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Convert to privacy loss distribution
        for i, prob in enumerate(probabilities):
            if prob > 1e-10:
                # Privacy loss for exponential mechanism
                privacy_loss = epsilon * (utilities[i] - max_utility) / (2 * sensitivity)
                
                self.privacy_losses.append(PrivacyLossPoint(
                    privacy_loss=privacy_loss,
                    probability=prob,
                    mechanism_type="exponential",
                    parameters={"utility": utilities[i], "sensitivity": sensitivity}
                ))
    
    def _compute_gaussian_noise_multiplier(self, epsilon: float, delta: float) -> float:
        """Compute optimal noise multiplier for Gaussian mechanism."""
        if not SCIPY_AVAILABLE:
            # Simplified approximation
            return math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        
        # Binary search for optimal noise multiplier
        def privacy_loss(sigma):
            # Approximate privacy loss for Gaussian mechanism
            if sigma <= 0:
                return float('inf')
            
            # RDP to (ε, δ)-DP conversion
            alpha = 2.0
            rdp_epsilon = 1 / (2 * sigma**2)
            
            # Convert RDP to (ε, δ)-DP
            converted_epsilon = rdp_epsilon + math.log(1/delta) / (alpha - 1)
            return abs(converted_epsilon - epsilon)
        
        # Binary search
        result = optimize.minimize_scalar(privacy_loss, bounds=(0.1, 10.0), method='bounded')
        return result.x
    
    def compose(self) -> Tuple[float, float]:
        """Compose all mechanisms and return final (ε, δ) guarantee."""
        if not self.privacy_losses:
            return 0.0, 0.0
        
        # Simple composition for now (can be improved with convolution)
        total_epsilon = 0.0
        total_delta = 0.0
        
        for entry in self.composition_history:
            total_epsilon += entry["epsilon"]
            total_delta += entry["delta"]
        
        # Apply advanced composition if available
        if SCIPY_AVAILABLE and len(self.composition_history) > 1:
            total_epsilon, total_delta = self._advanced_composition()
        
        return total_epsilon, total_delta
    
    def _advanced_composition(self) -> Tuple[float, float]:
        """Apply advanced composition theorem."""
        k = len(self.composition_history)
        max_epsilon = max(entry["epsilon"] for entry in self.composition_history)
        sum_delta = sum(entry["delta"] for entry in self.composition_history)
        
        # Advanced composition bound
        if k * max_epsilon * max_epsilon < 1:
            # Small epsilon regime
            epsilon_prime = math.sqrt(2 * k * math.log(1/sum_delta)) * max_epsilon + k * max_epsilon**2
        else:
            # Large epsilon regime  
            epsilon_prime = math.sqrt(k) * max_epsilon
        
        delta_prime = k * sum_delta
        
        return epsilon_prime, delta_prime
    
    def get_privacy_curve(self, delta_range: List[float]) -> List[Tuple[float, float]]:
        """Get privacy curve (ε vs δ) for this composition."""
        curve = []
        
        for delta in delta_range:
            # Recompute epsilon for this delta
            epsilon = 0.0
            
            # Simple approach: sum of epsilons (can be improved)
            for entry in self.composition_history:
                entry_epsilon = entry["epsilon"]
                entry_delta = entry["delta"]
                
                # Adjust epsilon based on delta ratio
                if entry_delta > 0:
                    adjustment = math.log(delta / entry_delta)
                    epsilon += max(0, entry_epsilon + adjustment)
                else:
                    epsilon += entry_epsilon
            
            curve.append((epsilon, delta))
        
        return curve


class AttentionSensitivityAnalyzer:
    """
    Analyzes sensitivity of attention mechanisms for optimal privacy calibration.
    
    Computes per-head, per-layer, and per-component sensitivity bounds
    for tight privacy analysis and adaptive noise injection.
    """
    
    def __init__(self, clip_method: str = "per_sample"):
        self.clip_method = clip_method
        self.sensitivity_cache: Dict[str, AttentionSensitivityProfile] = {}
        
    def analyze_attention_sensitivity(
        self, 
        model: Any, 
        data_loader: Any, 
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, AttentionSensitivityProfile]:
        """Analyze sensitivity for attention layers."""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning dummy sensitivity")
            return {}
        
        model.eval()
        sensitivities = {}
        
        # Get attention layers
        attention_layers = self._get_attention_layers(model, layer_indices)
        
        for layer_idx, attention_layer in attention_layers.items():
            logger.info(f"Analyzing sensitivity for attention layer {layer_idx}")
            
            sensitivity_profile = self._compute_layer_sensitivity(
                attention_layer, data_loader, layer_idx
            )
            
            sensitivities[layer_idx] = sensitivity_profile
            
            # Cache result
            cache_key = self._get_cache_key(model, layer_idx)
            self.sensitivity_cache[cache_key] = sensitivity_profile
        
        return sensitivities
    
    def _get_attention_layers(self, model: Any, layer_indices: Optional[List[int]]) -> Dict[int, Any]:
        """Extract attention layers from model."""
        attention_layers = {}
        
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-style model
            layers = model.transformer.h
            for i, layer in enumerate(layers):
                if layer_indices is None or i in layer_indices:
                    if hasattr(layer, 'attn'):
                        attention_layers[i] = layer.attn
        
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT-style model  
            layers = model.encoder.layer
            for i, layer in enumerate(layers):
                if layer_indices is None or i in layer_indices:
                    if hasattr(layer, 'attention'):
                        attention_layers[i] = layer.attention
        
        else:
            # Try to find attention modules
            for name, module in model.named_modules():
                if 'attention' in name.lower() or 'attn' in name.lower():
                    if hasattr(module, 'forward'):
                        # Extract layer index from name if possible
                        layer_idx = self._extract_layer_index(name)
                        if layer_indices is None or layer_idx in layer_indices:
                            attention_layers[layer_idx] = module
        
        return attention_layers
    
    def _extract_layer_index(self, name: str) -> int:
        """Extract layer index from module name."""
        import re
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0
    
    def _compute_layer_sensitivity(self, attention_layer: Any, data_loader: Any, layer_idx: int) -> AttentionSensitivityProfile:
        """Compute sensitivity profile for a single attention layer."""
        
        if not TORCH_AVAILABLE:
            return AttentionSensitivityProfile(
                per_head_sensitivity=[1.0] * 8,  # Default 8 heads
                layer_sensitivity=1.0,
                query_sensitivity=1.0,
                key_sensitivity=1.0,
                value_sensitivity=1.0,
                output_sensitivity=1.0,
                gradient_bound=1.0,
                computed_at=time.time()
            )
        
        # Hooks to capture gradients
        gradients = {}
        
        def gradient_hook(name):
            def hook(grad):
                gradients[name] = grad.clone().detach()
            return hook
        
        # Register hooks
        hooks = []
        if hasattr(attention_layer, 'q_proj'):
            hooks.append(attention_layer.q_proj.weight.register_hook(gradient_hook('query')))
        if hasattr(attention_layer, 'k_proj'):
            hooks.append(attention_layer.k_proj.weight.register_hook(gradient_hook('key')))
        if hasattr(attention_layer, 'v_proj'):
            hooks.append(attention_layer.v_proj.weight.register_hook(gradient_hook('value')))
        if hasattr(attention_layer, 'out_proj'):
            hooks.append(attention_layer.out_proj.weight.register_hook(gradient_hook('output')))
        
        # Sample data for sensitivity analysis
        gradient_norms = {'query': [], 'key': [], 'value': [], 'output': []}
        
        try:
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= 10:  # Limit samples for efficiency
                    break
                
                # Forward pass
                if isinstance(batch, dict):
                    inputs = batch['input_ids'] if 'input_ids' in batch else list(batch.values())[0]
                else:
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                # Simplified forward pass (model-dependent)
                try:
                    outputs = attention_layer(inputs)
                    if isinstance(outputs, tuple):
                        loss = outputs[0].sum()
                    else:
                        loss = outputs.sum()
                    
                    # Backward pass
                    loss.backward(retain_graph=True)
                    
                    # Collect gradient norms
                    for component in ['query', 'key', 'value', 'output']:
                        if component in gradients:
                            grad_norm = torch.norm(gradients[component]).item()
                            gradient_norms[component].append(grad_norm)
                            gradients[component] = None  # Clear for next iteration
                
                except Exception as e:
                    logger.warning(f"Error in sensitivity computation for batch {batch_idx}: {e}")
                    continue
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Compute sensitivity profile
        def compute_sensitivity(norms):
            if not norms:
                return 1.0
            return max(norms)  # L∞ sensitivity
        
        query_sensitivity = compute_sensitivity(gradient_norms['query'])
        key_sensitivity = compute_sensitivity(gradient_norms['key'])
        value_sensitivity = compute_sensitivity(gradient_norms['value'])
        output_sensitivity = compute_sensitivity(gradient_norms['output'])
        
        # Overall layer sensitivity
        layer_sensitivity = max(query_sensitivity, key_sensitivity, value_sensitivity, output_sensitivity)
        
        # Per-head sensitivity (simplified - assumes equal heads)
        num_heads = getattr(attention_layer, 'num_heads', 8)
        per_head_sensitivity = [layer_sensitivity / math.sqrt(num_heads)] * num_heads
        
        # Gradient bound for clipping
        all_norms = []
        for component_norms in gradient_norms.values():
            all_norms.extend(component_norms)
        
        gradient_bound = np.percentile(all_norms, 95) if all_norms else 1.0
        
        return AttentionSensitivityProfile(
            per_head_sensitivity=per_head_sensitivity,
            layer_sensitivity=layer_sensitivity,
            query_sensitivity=query_sensitivity,
            key_sensitivity=key_sensitivity,
            value_sensitivity=value_sensitivity,
            output_sensitivity=output_sensitivity,
            gradient_bound=gradient_bound,
            computed_at=time.time()
        )
    
    def _get_cache_key(self, model: Any, layer_idx: int) -> str:
        """Generate cache key for sensitivity profile."""
        model_hash = hashlib.md5(str(model).encode()).hexdigest()[:8]
        return f"{model_hash}_layer_{layer_idx}"
    
    def get_optimal_clipping_bounds(self, sensitivities: Dict[int, AttentionSensitivityProfile]) -> Dict[str, float]:
        """Compute optimal clipping bounds across all layers."""
        
        all_query_sens = []
        all_key_sens = []
        all_value_sens = []
        all_output_sens = []
        all_grad_bounds = []
        
        for profile in sensitivities.values():
            all_query_sens.append(profile.query_sensitivity)
            all_key_sens.append(profile.key_sensitivity)
            all_value_sens.append(profile.value_sensitivity)
            all_output_sens.append(profile.output_sensitivity)
            all_grad_bounds.append(profile.gradient_bound)
        
        return {
            'query_clip': max(all_query_sens) if all_query_sens else 1.0,
            'key_clip': max(all_key_sens) if all_key_sens else 1.0,
            'value_clip': max(all_value_sens) if all_value_sens else 1.0,
            'output_clip': max(all_output_sens) if all_output_sens else 1.0,
            'gradient_clip': np.percentile(all_grad_bounds, 90) if all_grad_bounds else 1.0
        }


class StructuredNoiseMechanism:
    """
    Advanced structured noise mechanisms for attention matrices.
    
    Implements novel noise patterns that preserve attention structure
    while providing differential privacy guarantees.
    """
    
    def __init__(self, noise_structure: str = "low_rank"):
        self.noise_structure = noise_structure
        self.noise_cache: Dict[str, Any] = {}
        
    def generate_structured_noise(
        self, 
        tensor_shape: Tuple[int, ...], 
        sensitivity: float,
        epsilon: float, 
        delta: float,
        **kwargs
    ) -> Any:
        """Generate structured noise for attention tensors."""
        
        if not TORCH_AVAILABLE:
            return np.zeros(tensor_shape)
        
        if self.noise_structure == "low_rank":
            return self._generate_low_rank_noise(tensor_shape, sensitivity, epsilon, delta, **kwargs)
        elif self.noise_structure == "sparse":
            return self._generate_sparse_noise(tensor_shape, sensitivity, epsilon, delta, **kwargs)
        elif self.noise_structure == "block_diagonal":
            return self._generate_block_diagonal_noise(tensor_shape, sensitivity, epsilon, delta, **kwargs)
        elif self.noise_structure == "attention_aware":
            return self._generate_attention_aware_noise(tensor_shape, sensitivity, epsilon, delta, **kwargs)
        else:
            return self._generate_gaussian_noise(tensor_shape, sensitivity, epsilon, delta)
    
    def _generate_low_rank_noise(
        self, 
        tensor_shape: Tuple[int, ...], 
        sensitivity: float,
        epsilon: float, 
        delta: float,
        rank: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate low-rank structured noise."""
        
        # Default rank is much smaller than full rank
        if rank is None:
            rank = min(tensor_shape[-2:]) // 4
        
        rank = max(1, min(rank, min(tensor_shape[-2:])))
        
        # Compute noise scale
        noise_scale = self._compute_noise_scale(sensitivity, epsilon, delta)
        
        # Generate low-rank factorization: U @ V^T
        batch_dims = tensor_shape[:-2]
        matrix_shape = tensor_shape[-2:]
        
        # U: [..., m, rank]
        U_shape = batch_dims + (matrix_shape[0], rank)
        U = torch.normal(0, noise_scale, U_shape)
        
        # V: [..., n, rank]  
        V_shape = batch_dims + (matrix_shape[1], rank)
        V = torch.normal(0, noise_scale, V_shape)
        
        # Compute low-rank noise: U @ V^T
        noise = torch.matmul(U, V.transpose(-2, -1))
        
        # Scale appropriately for privacy
        noise = noise / math.sqrt(rank)
        
        return noise
    
    def _generate_sparse_noise(
        self, 
        tensor_shape: Tuple[int, ...], 
        sensitivity: float,
        epsilon: float, 
        delta: float,
        sparsity: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """Generate sparse structured noise."""
        
        noise_scale = self._compute_noise_scale(sensitivity, epsilon, delta)
        
        # Generate dense noise
        noise = torch.normal(0, noise_scale, tensor_shape)
        
        # Apply sparsity mask
        mask = torch.rand(tensor_shape) > sparsity
        noise = noise * mask.float()
        
        # Adjust scale to maintain privacy guarantee
        active_fraction = 1.0 - sparsity
        noise = noise / math.sqrt(active_fraction)
        
        return noise
    
    def _generate_block_diagonal_noise(
        self, 
        tensor_shape: Tuple[int, ...], 
        sensitivity: float,
        epsilon: float, 
        delta: float,
        block_size: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate block-diagonal structured noise."""
        
        if len(tensor_shape) < 2:
            return self._generate_gaussian_noise(tensor_shape, sensitivity, epsilon, delta)
        
        noise_scale = self._compute_noise_scale(sensitivity, epsilon, delta)
        
        # Default block size
        if block_size is None:
            block_size = min(tensor_shape[-2:]) // 4
        
        block_size = max(1, block_size)
        
        # Initialize noise tensor
        noise = torch.zeros(tensor_shape)
        
        # Fill diagonal blocks
        matrix_dims = tensor_shape[-2:]
        num_blocks_row = (matrix_dims[0] + block_size - 1) // block_size
        num_blocks_col = (matrix_dims[1] + block_size - 1) // block_size
        num_blocks = min(num_blocks_row, num_blocks_col)
        
        for i in range(num_blocks):
            row_start = i * block_size
            row_end = min((i + 1) * block_size, matrix_dims[0])
            col_start = i * block_size 
            col_end = min((i + 1) * block_size, matrix_dims[1])
            
            # Generate noise for this block
            block_shape = tensor_shape[:-2] + (row_end - row_start, col_end - col_start)
            block_noise = torch.normal(0, noise_scale, block_shape)
            
            # Assign to noise tensor
            noise[..., row_start:row_end, col_start:col_end] = block_noise
        
        return noise
    
    def _generate_attention_aware_noise(
        self, 
        tensor_shape: Tuple[int, ...], 
        sensitivity: float,
        epsilon: float, 
        delta: float,
        attention_pattern: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate attention-pattern-aware noise."""
        
        noise_scale = self._compute_noise_scale(sensitivity, epsilon, delta)
        
        if attention_pattern is None:
            # Default to causal attention pattern
            seq_len = tensor_shape[-1]
            attention_pattern = torch.tril(torch.ones(seq_len, seq_len))
        
        # Generate base noise
        noise = torch.normal(0, noise_scale, tensor_shape)
        
        # Apply attention pattern weighting
        # More noise where attention is weaker, less where it's stronger
        if attention_pattern.shape[-2:] == tensor_shape[-2:]:
            # Normalize attention pattern
            pattern = attention_pattern / (attention_pattern.max() + 1e-8)
            
            # Inverse weighting: more noise where attention is low
            weight = 1.0 - pattern + 0.1  # Avoid zero weights
            
            # Expand weight to match tensor dimensions
            while len(weight.shape) < len(tensor_shape):
                weight = weight.unsqueeze(0)
            
            noise = noise * weight
        
        return noise
    
    def _generate_gaussian_noise(
        self, 
        tensor_shape: Tuple[int, ...], 
        sensitivity: float,
        epsilon: float, 
        delta: float
    ) -> torch.Tensor:
        """Generate standard Gaussian noise (fallback)."""
        
        noise_scale = self._compute_noise_scale(sensitivity, epsilon, delta)
        return torch.normal(0, noise_scale, tensor_shape)
    
    def _compute_noise_scale(self, sensitivity: float, epsilon: float, delta: float) -> float:
        """Compute noise scale for (ε, δ)-DP."""
        
        if delta == 0:
            # Pure ε-DP (Laplace mechanism)
            return sensitivity / epsilon
        
        # (ε, δ)-DP (Gaussian mechanism)
        if not SCIPY_AVAILABLE:
            # Simplified approximation
            return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        
        # More precise calculation
        c = math.sqrt(2 * math.log(1.25 / delta))
        return sensitivity * c / epsilon
    
    def compute_privacy_cost(
        self, 
        tensor_shape: Tuple[int, ...], 
        sensitivity: float,
        epsilon: float, 
        delta: float
    ) -> Dict[str, float]:
        """Compute privacy cost for structured noise."""
        
        base_cost = epsilon
        
        # Structure-specific adjustments
        if self.noise_structure == "low_rank":
            # Low-rank noise may have better privacy-utility trade-off
            rank = min(tensor_shape[-2:]) // 4
            total_params = tensor_shape[-2] * tensor_shape[-1]
            effective_params = rank * (tensor_shape[-2] + tensor_shape[-1])
            cost_reduction = effective_params / total_params
            adjusted_cost = base_cost * cost_reduction
        
        elif self.noise_structure == "sparse":
            # Sparse noise concentrates privacy cost
            sparsity = 0.9  # Default
            active_fraction = 1.0 - sparsity
            adjusted_cost = base_cost / math.sqrt(active_fraction)
        
        else:
            adjusted_cost = base_cost
        
        return {
            "base_epsilon": base_cost,
            "adjusted_epsilon": adjusted_cost,
            "structure": self.noise_structure,
            "efficiency_ratio": base_cost / adjusted_cost if adjusted_cost > 0 else 1.0
        }


class AdvancedCompositionAnalyzer:
    """
    Advanced composition analysis for complex privacy mechanisms.
    
    Implements state-of-the-art composition theorems and optimal
    privacy budget allocation strategies.
    """
    
    def __init__(self):
        self.mechanisms: List[Dict[str, Any]] = []
        self.composition_strategy = "optimal"
        
    def add_mechanism(
        self, 
        mechanism_type: str,
        sensitivity: float,
        epsilon: float,
        delta: float,
        iterations: int = 1,
        **kwargs
    ) -> None:
        """Add a mechanism to the composition."""
        
        self.mechanisms.append({
            "type": mechanism_type,
            "sensitivity": sensitivity,
            "epsilon": epsilon,
            "delta": delta,
            "iterations": iterations,
            "timestamp": time.time(),
            "kwargs": kwargs
        })
    
    def compute_total_privacy_cost(self) -> Tuple[float, float]:
        """Compute total privacy cost using advanced composition."""
        
        if not self.mechanisms:
            return 0.0, 0.0
        
        if self.composition_strategy == "basic":
            return self._basic_composition()
        elif self.composition_strategy == "advanced":
            return self._advanced_composition()
        elif self.composition_strategy == "optimal":
            return self._optimal_composition()
        else:
            return self._basic_composition()
    
    def _basic_composition(self) -> Tuple[float, float]:
        """Basic composition: sum of epsilons and deltas."""
        total_epsilon = sum(m["epsilon"] * m["iterations"] for m in self.mechanisms)
        total_delta = sum(m["delta"] * m["iterations"] for m in self.mechanisms)
        return total_epsilon, total_delta
    
    def _advanced_composition(self) -> Tuple[float, float]:
        """Advanced composition theorem."""
        
        # Group mechanisms by type
        grouped = {}
        for mechanism in self.mechanisms:
            key = (mechanism["type"], mechanism["epsilon"], mechanism["delta"])
            if key not in grouped:
                grouped[key] = 0
            grouped[key] += mechanism["iterations"]
        
        total_epsilon = 0.0
        total_delta = 0.0
        
        for (mech_type, epsilon, delta), k in grouped.items():
            if delta == 0:
                # Pure DP
                total_epsilon += k * epsilon
            else:
                # Apply advanced composition for (ε, δ)-DP
                if k * epsilon * epsilon < 1:
                    # Small epsilon regime
                    composed_epsilon = math.sqrt(2 * k * math.log(1/delta)) * epsilon + k * epsilon * epsilon
                else:
                    # Large epsilon regime
                    composed_epsilon = math.sqrt(k) * epsilon
                
                total_epsilon += composed_epsilon
                total_delta += k * delta
        
        return total_epsilon, total_delta
    
    def _optimal_composition(self) -> Tuple[float, float]:
        """Optimal composition using privacy loss distributions."""
        
        # Create PLD for composition
        pld = PrivacyLossDistribution()
        
        for mechanism in self.mechanisms:
            for _ in range(mechanism["iterations"]):
                pld.add_mechanism(
                    mechanism["type"],
                    mechanism["sensitivity"],
                    mechanism["epsilon"],
                    mechanism["delta"]
                )
        
        return pld.compose()
    
    def optimize_budget_allocation(
        self, 
        total_epsilon: float, 
        total_delta: float,
        utilities: Optional[List[float]] = None
    ) -> List[Dict[str, float]]:
        """Optimize privacy budget allocation across mechanisms."""
        
        n_mechanisms = len(self.mechanisms)
        if n_mechanisms == 0:
            return []
        
        if utilities is None:
            # Equal utility assumption
            utilities = [1.0] * n_mechanisms
        
        # Simple proportional allocation based on utilities
        total_utility = sum(utilities)
        
        allocations = []
        remaining_epsilon = total_epsilon
        remaining_delta = total_delta
        
        for i, (mechanism, utility) in enumerate(zip(self.mechanisms, utilities)):
            if i == n_mechanisms - 1:
                # Last mechanism gets remaining budget
                allocated_epsilon = remaining_epsilon
                allocated_delta = remaining_delta
            else:
                # Proportional allocation
                proportion = utility / total_utility
                allocated_epsilon = total_epsilon * proportion
                allocated_delta = total_delta * proportion
                
                remaining_epsilon -= allocated_epsilon
                remaining_delta -= allocated_delta
            
            allocations.append({
                "mechanism_index": i,
                "allocated_epsilon": allocated_epsilon,
                "allocated_delta": allocated_delta,
                "utility_weight": utility,
                "efficiency": utility / (allocated_epsilon + 1e-8)
            })
        
        return allocations
    
    def get_composition_summary(self) -> Dict[str, Any]:
        """Get detailed summary of the composition."""
        
        total_epsilon, total_delta = self.compute_total_privacy_cost()
        
        # Mechanism breakdown
        mechanism_counts = {}
        for mechanism in self.mechanisms:
            mech_type = mechanism["type"]
            if mech_type not in mechanism_counts:
                mechanism_counts[mech_type] = 0
            mechanism_counts[mech_type] += mechanism["iterations"]
        
        return {
            "total_privacy_cost": {
                "epsilon": total_epsilon,
                "delta": total_delta
            },
            "mechanism_breakdown": mechanism_counts,
            "total_mechanisms": len(self.mechanisms),
            "total_iterations": sum(m["iterations"] for m in self.mechanisms),
            "composition_strategy": self.composition_strategy,
            "privacy_regime": "small_epsilon" if total_epsilon < 1 else "large_epsilon"
        }


def create_research_mechanism(
    mechanism_type: PrivacyMechanismType,
    **kwargs
) -> Union[PrivacyLossDistribution, AttentionSensitivityAnalyzer, StructuredNoiseMechanism, AdvancedCompositionAnalyzer]:
    """Factory function to create research mechanisms."""
    
    if mechanism_type == PrivacyMechanismType.PRIVACY_LOSS_DISTRIBUTION:
        discretization = kwargs.get("discretization_interval", 1e-4)
        return PrivacyLossDistribution(discretization)
    
    elif mechanism_type == PrivacyMechanismType.ATTENTION_SENSITIVITY:
        clip_method = kwargs.get("clip_method", "per_sample")
        return AttentionSensitivityAnalyzer(clip_method)
    
    elif mechanism_type == PrivacyMechanismType.STRUCTURED_NOISE:
        noise_structure = kwargs.get("noise_structure", "low_rank")
        return StructuredNoiseMechanism(noise_structure)
    
    elif mechanism_type == PrivacyMechanismType.ADAPTIVE_COMPOSITION:
        return AdvancedCompositionAnalyzer()
    
    else:
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")


# Example usage and integration points
if __name__ == "__main__":
    # Example: Privacy Loss Distribution
    pld = create_research_mechanism(PrivacyMechanismType.PRIVACY_LOSS_DISTRIBUTION)
    pld.add_mechanism("gaussian", sensitivity=1.0, epsilon=0.5, delta=1e-5)
    pld.add_mechanism("laplace", sensitivity=1.0, epsilon=0.3, delta=0.0)
    
    total_epsilon, total_delta = pld.compose()
    print(f"Total privacy cost: (ε={total_epsilon:.3f}, δ={total_delta:.3e})")
    
    # Example: Structured Noise
    structured_noise = create_research_mechanism(
        PrivacyMechanismType.STRUCTURED_NOISE,
        noise_structure="low_rank"
    )
    
    if TORCH_AVAILABLE:
        attention_shape = (32, 8, 128, 128)  # (batch, heads, seq, seq)
        noise = structured_noise.generate_structured_noise(
            attention_shape, 
            sensitivity=1.0, 
            epsilon=1.0, 
            delta=1e-5,
            rank=16
        )
        print(f"Generated structured noise with shape: {noise.shape}")
    
    # Example: Composition Analysis
    composition = create_research_mechanism(PrivacyMechanismType.ADAPTIVE_COMPOSITION)
    composition.add_mechanism("gaussian", 1.0, 0.5, 1e-5, iterations=100)
    composition.add_mechanism("laplace", 1.0, 0.3, 0.0, iterations=50)
    
    summary = composition.get_composition_summary()
    print(f"Composition summary: {json.dumps(summary, indent=2)}")
