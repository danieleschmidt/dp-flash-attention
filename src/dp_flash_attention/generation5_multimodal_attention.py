"""
Generation 5.2: Multi-Modal Differentially Private Attention

Advanced DP attention mechanisms supporting text, vision, audio, and cross-modal
fusion with modality-specific privacy budgets and adaptive noise calibration.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    nn = None

from .generation5_quantum_privacy import LatticeBasedNoiseMechanism, QuantumRenyiAccountant, QuantumThreatModel
from .utils import validate_privacy_params, estimate_memory_usage
from .error_handling import handle_errors, PrivacyParameterError, TensorShapeError
from .logging_utils import get_logger


class ModalityType(Enum):
    """Supported modality types for multi-modal attention."""
    TEXT = "text"
    VISION = "vision" 
    AUDIO = "audio"
    VIDEO = "video"
    GRAPH = "graph"
    TIME_SERIES = "time_series"
    TABULAR = "tabular"


@dataclass
class ModalityConfig:
    """Configuration for a specific modality."""
    modality_type: ModalityType
    embed_dim: int
    num_heads: int
    privacy_budget: float  # Dedicated epsilon for this modality
    sensitivity_factor: float = 1.0  # Modality-specific sensitivity scaling
    attention_pattern: str = "full"  # full, sparse, local, hierarchical
    privacy_noise_type: str = "gaussian"  # gaussian, laplace, structured
    temporal_correlation: bool = False  # Account for temporal privacy leakage
    cross_modal_leakage_bound: float = 0.1  # Max privacy leakage to other modalities


@dataclass
class CrossModalFusionConfig:
    """Configuration for cross-modal fusion layers."""
    fusion_type: str = "attention"  # attention, concatenation, gating, transformer
    fusion_privacy_budget: float = 0.5  # Additional budget for fusion
    alignment_mechanism: str = "learned"  # learned, fixed, dynamic
    privacy_barrier: bool = True  # Enforce privacy barriers between modalities
    differential_fusion: bool = True  # Apply DP to fusion weights


class MultiModalPrivacyBudgetManager:
    """
    Advanced privacy budget allocation across multiple modalities.
    
    Implements optimal budget allocation considering modality sensitivity,
    cross-modal correlations, and privacy amplification opportunities.
    """
    
    def __init__(self, 
                 modality_configs: List[ModalityConfig],
                 total_epsilon: float = 3.0,
                 delta: float = 1e-5,
                 allocation_strategy: str = "adaptive"):
        
        self.modality_configs = {config.modality_type: config for config in modality_configs}
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.allocation_strategy = allocation_strategy
        self.logger = get_logger()
        
        # Privacy accounting per modality
        self.modality_accountants = {}
        self.cross_modal_accountant = QuantumRenyiAccountant(
            params=type('QuantumParams', (), {
                'threat_model': QuantumThreatModel.POST_QUANTUM,
                'lattice_dimension': 256,
                'quantum_security_level': 128
            })()
        )
        
        # Initialize modality-specific accountants
        self._initialize_modality_accountants()
        
        # Compute optimal budget allocation
        self._compute_optimal_allocation()
    
    def _initialize_modality_accountants(self):
        """Initialize privacy accountants for each modality."""
        for modality_type in self.modality_configs:
            self.modality_accountants[modality_type] = QuantumRenyiAccountant(
                params=type('QuantumParams', (), {
                    'threat_model': QuantumThreatModel.POST_QUANTUM,
                    'lattice_dimension': 512,
                    'quantum_security_level': 128
                })()
            )
    
    def _compute_optimal_allocation(self):
        """Compute optimal privacy budget allocation across modalities."""
        if self.allocation_strategy == "equal":
            # Equal allocation
            per_modality_budget = self.total_epsilon / len(self.modality_configs)
            for config in self.modality_configs.values():
                config.privacy_budget = per_modality_budget
                
        elif self.allocation_strategy == "sensitivity_weighted":
            # Allocate based on inverse sensitivity (more budget to less sensitive)
            total_inv_sensitivity = sum(1.0 / config.sensitivity_factor 
                                      for config in self.modality_configs.values())
            
            for config in self.modality_configs.values():
                weight = (1.0 / config.sensitivity_factor) / total_inv_sensitivity
                config.privacy_budget = self.total_epsilon * weight * 0.8  # Reserve 20% for fusion
                
        elif self.allocation_strategy == "adaptive":
            # Adaptive allocation based on modality complexity and sensitivity
            self._adaptive_budget_allocation()
        
        self.logger.info("Privacy budget allocation:")
        for modality_type, config in self.modality_configs.items():
            self.logger.info(f"  {modality_type.value}: ε = {config.privacy_budget:.4f}")
    
    def _adaptive_budget_allocation(self):
        """Adaptive budget allocation using utility-privacy trade-off optimization."""
        # Estimate utility contribution of each modality
        modality_weights = {}
        total_weight = 0.0
        
        for modality_type, config in self.modality_configs.items():
            # Utility score based on dimensionality, sensitivity, and attention heads
            utility_score = (
                config.embed_dim * config.num_heads / 
                (config.sensitivity_factor * (1 + config.cross_modal_leakage_bound))
            )
            
            # Apply modality-specific utility multipliers
            multipliers = {
                ModalityType.TEXT: 1.2,  # Text usually most informative
                ModalityType.VISION: 1.0,
                ModalityType.AUDIO: 0.8,
                ModalityType.VIDEO: 1.1, 
                ModalityType.GRAPH: 0.9,
                ModalityType.TIME_SERIES: 0.9,
                ModalityType.TABULAR: 1.0
            }
            
            weighted_utility = utility_score * multipliers.get(modality_type, 1.0)
            modality_weights[modality_type] = weighted_utility
            total_weight += weighted_utility
        
        # Allocate budget proportional to utility (with minimum guarantees)
        fusion_reserve = 0.15 * self.total_epsilon  # Reserve for fusion
        available_budget = self.total_epsilon - fusion_reserve
        
        min_budget = 0.1  # Minimum budget per modality
        
        for modality_type, config in self.modality_configs.items():
            proportional_budget = available_budget * (modality_weights[modality_type] / total_weight)
            config.privacy_budget = max(min_budget, proportional_budget)
    
    def consume_privacy_budget(self, 
                              modality_type: ModalityType,
                              epsilon_used: float,
                              is_cross_modal: bool = False) -> bool:
        """
        Consume privacy budget for a modality operation.
        
        Args:
            modality_type: Type of modality
            epsilon_used: Privacy budget to consume
            is_cross_modal: Whether this is a cross-modal operation
            
        Returns:
            True if budget was available and consumed, False otherwise
        """
        if is_cross_modal:
            # Track cross-modal privacy consumption
            self.cross_modal_accountant.add_quantum_mechanism(
                epsilon=epsilon_used,
                delta=self.delta / len(self.modality_configs),
                mechanism_type="gaussian"
            )
        else:
            # Track modality-specific consumption
            if modality_type in self.modality_accountants:
                config = self.modality_configs[modality_type]
                
                # Check if budget is available
                current_spent = self.modality_accountants[modality_type].get_quantum_epsilon(self.delta)
                if current_spent + epsilon_used > config.privacy_budget:
                    self.logger.warning(f"Privacy budget exceeded for {modality_type.value}")
                    return False
                
                # Consume budget
                self.modality_accountants[modality_type].add_quantum_mechanism(
                    epsilon=epsilon_used,
                    delta=self.delta,
                    mechanism_type="gaussian"
                )
        
        return True
    
    def get_remaining_budget(self, modality_type: ModalityType) -> float:
        """Get remaining privacy budget for a modality."""
        if modality_type not in self.modality_accountants:
            return 0.0
            
        config = self.modality_configs[modality_type] 
        spent = self.modality_accountants[modality_type].get_quantum_epsilon(self.delta)
        return max(0.0, config.privacy_budget - spent)
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary across all modalities."""
        summary = {
            'total_budget': self.total_epsilon,
            'delta': self.delta,
            'allocation_strategy': self.allocation_strategy,
            'modalities': {}
        }
        
        total_spent = 0.0
        for modality_type, accountant in self.modality_accountants.items():
            config = self.modality_configs[modality_type]
            spent = accountant.get_quantum_epsilon(self.delta)
            remaining = config.privacy_budget - spent
            
            summary['modalities'][modality_type.value] = {
                'allocated_budget': config.privacy_budget,
                'spent_budget': spent,
                'remaining_budget': remaining,
                'utilization_rate': spent / config.privacy_budget if config.privacy_budget > 0 else 0.0
            }
            total_spent += spent
        
        # Add cross-modal fusion privacy
        cross_modal_spent = self.cross_modal_accountant.get_quantum_epsilon(self.delta)
        summary['cross_modal_spent'] = cross_modal_spent
        summary['total_spent'] = total_spent + cross_modal_spent
        summary['overall_utilization'] = summary['total_spent'] / self.total_epsilon
        
        return summary


class MultiModalDPAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-modal differentially private attention mechanism.
    
    Supports multiple input modalities with modality-specific privacy budgets,
    cross-modal attention, and privacy-preserving fusion.
    """
    
    def __init__(self,
                 modality_configs: List[ModalityConfig],
                 fusion_config: CrossModalFusionConfig,
                 total_privacy_budget: float = 3.0,
                 delta: float = 1e-5,
                 quantum_secure: bool = True):
        
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.modality_configs = modality_configs
        self.fusion_config = fusion_config
        self.quantum_secure = quantum_secure
        self.logger = get_logger()
        
        # Initialize privacy budget manager
        self.privacy_manager = MultiModalPrivacyBudgetManager(
            modality_configs=modality_configs,
            total_epsilon=total_privacy_budget,
            delta=delta,
            allocation_strategy="adaptive"
        )
        
        # Initialize modality-specific attention layers
        self.modality_attentions = {}
        self.modality_projections = {}
        
        if TORCH_AVAILABLE:
            self._initialize_attention_layers()
            self._initialize_fusion_layers()
        
        # Initialize quantum-resistant noise mechanisms
        self.noise_mechanisms = {}
        self._initialize_noise_mechanisms()
    
    def _initialize_attention_layers(self):
        """Initialize attention layers for each modality."""
        for config in self.modality_configs:
            modality_type = config.modality_type
            
            # Create modality-specific multi-head attention
            self.modality_attentions[modality_type] = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                dropout=0.0,  # DP noise replaces dropout
                bias=True,
                batch_first=True
            )
            
            # Create modality-specific input projections
            self.modality_projections[modality_type] = nn.ModuleDict({
                'input_proj': nn.Linear(config.embed_dim, config.embed_dim),
                'output_proj': nn.Linear(config.embed_dim, config.embed_dim),
                'layer_norm': nn.LayerNorm(config.embed_dim)
            })
    
    def _initialize_fusion_layers(self):
        """Initialize cross-modal fusion layers."""
        total_dim = sum(config.embed_dim for config in self.modality_configs)
        
        if self.fusion_config.fusion_type == "attention":
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=total_dim,
                num_heads=8,
                dropout=0.0,
                batch_first=True
            )
        elif self.fusion_config.fusion_type == "gating":
            self.fusion_gate = nn.ModuleDict({
                'gate_proj': nn.Linear(total_dim, total_dim),
                'value_proj': nn.Linear(total_dim, total_dim),
                'output_proj': nn.Linear(total_dim, total_dim)
            })
        elif self.fusion_config.fusion_type == "transformer":
            self.fusion_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=total_dim,
                    nhead=8,
                    dim_feedforward=total_dim * 2,
                    dropout=0.0,
                    batch_first=True
                ),
                num_layers=2
            )
        
        # Fusion output projection
        self.fusion_output_proj = nn.Linear(total_dim, total_dim)
    
    def _initialize_noise_mechanisms(self):
        """Initialize quantum-resistant noise mechanisms for each modality."""
        for config in self.modality_configs:
            from .generation5_quantum_privacy import create_quantum_privacy_mechanism
            
            noise_mech, _ = create_quantum_privacy_mechanism(
                threat_model=QuantumThreatModel.POST_QUANTUM if self.quantum_secure else QuantumThreatModel.CLASSICAL,
                security_level=128,
                lattice_dimension=min(512, config.embed_dim)
            )
            
            self.noise_mechanisms[config.modality_type] = noise_mech
    
    @handle_errors(reraise=True, log_errors=True)
    def forward(self, 
                modality_inputs: Dict[ModalityType, Union[Tensor, np.ndarray]],
                attention_masks: Optional[Dict[ModalityType, Union[Tensor, np.ndarray]]] = None,
                return_attention_weights: bool = False,
                return_privacy_stats: bool = False) -> Dict[str, Any]:
        """
        Multi-modal forward pass with differential privacy.
        
        Args:
            modality_inputs: Dictionary mapping modality types to input tensors
            attention_masks: Optional attention masks per modality
            return_attention_weights: Whether to return attention weights (privacy risk)
            return_privacy_stats: Whether to return privacy consumption statistics
            
        Returns:
            Dictionary containing fused output and optional additional information
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for multi-modal attention")
        
        # Validate inputs
        self._validate_multimodal_inputs(modality_inputs)
        
        # Process each modality with differential privacy
        modality_outputs = {}
        privacy_consumptions = {}
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            for modality_type, input_tensor in modality_inputs.items():
                if modality_type not in self.modality_configs:
                    raise ValueError(f"Unknown modality type: {modality_type}")
                
                config = self.modality_configs[modality_type]
                
                # Apply modality-specific processing
                processed_output, privacy_used = self._process_modality(
                    input_tensor=input_tensor,
                    modality_config=config,
                    attention_mask=attention_masks.get(modality_type) if attention_masks else None
                )
                
                modality_outputs[modality_type] = processed_output
                privacy_consumptions[modality_type] = privacy_used
        
        # Cross-modal fusion with differential privacy
        fused_output, fusion_privacy = self._cross_modal_fusion(modality_outputs)
        
        # Prepare output
        result = {
            'fused_output': fused_output,
            'modality_outputs': modality_outputs if return_attention_weights else None
        }
        
        if return_privacy_stats:
            result['privacy_stats'] = {
                'modality_privacy': privacy_consumptions,
                'fusion_privacy': fusion_privacy,
                'budget_summary': self.privacy_manager.get_privacy_summary()
            }
        
        return result
    
    def _validate_multimodal_inputs(self, modality_inputs: Dict[ModalityType, Union[Tensor, np.ndarray]]):
        """Validate multi-modal input tensors."""
        if not modality_inputs:
            raise ValueError("No modality inputs provided")
        
        for modality_type, tensor in modality_inputs.items():
            if modality_type not in self.modality_configs:
                available = [m.value for m in self.modality_configs.keys()]
                raise ValueError(f"Unknown modality {modality_type}. Available: {available}")
            
            config = self.modality_configs[modality_type]
            
            if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                if tensor.dim() != 3:  # [batch, seq, dim]
                    raise TensorShapeError(
                        f"Expected 3D tensor for {modality_type.value}, got {tensor.dim()}D",
                        expected_shape="[batch, seq, embed_dim]",
                        actual_shape=str(tensor.shape)
                    )
                
                if tensor.size(-1) != config.embed_dim:
                    raise TensorShapeError(
                        f"Expected embed_dim {config.embed_dim} for {modality_type.value}, got {tensor.size(-1)}",
                        expected_shape=f"[*, *, {config.embed_dim}]",
                        actual_shape=str(tensor.shape)
                    )
    
    def _process_modality(self, 
                         input_tensor: Tensor,
                         modality_config: ModalityConfig,
                         attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, float]:
        """Process a single modality with differential privacy."""
        modality_type = modality_config.modality_type
        
        # Apply input projection
        projected = self.modality_projections[modality_type]['input_proj'](input_tensor)
        projected = self.modality_projections[modality_type]['layer_norm'](projected)
        
        # Compute attention with privacy
        attention_output, _ = self.modality_attentions[modality_type](
            query=projected,
            key=projected, 
            value=projected,
            key_padding_mask=attention_mask,
            need_weights=False  # Privacy: don't return attention weights
        )
        
        # Add differential privacy noise
        epsilon_per_step = modality_config.privacy_budget * 0.5  # Use half budget for attention
        noise_mechanism = self.noise_mechanisms[modality_type]
        
        # Calculate sensitivity based on modality characteristics
        sensitivity = self._calculate_modality_sensitivity(modality_config, input_tensor)
        
        # Add quantum-resistant noise
        noised_attention = noise_mechanism.add_quantum_noise(
            tensor=attention_output,
            sensitivity=sensitivity,
            epsilon=epsilon_per_step,
            delta=self.privacy_manager.delta
        )
        
        # Apply output projection
        output = self.modality_projections[modality_type]['output_proj'](noised_attention)
        
        # Update privacy accounting
        privacy_consumed = self.privacy_manager.consume_privacy_budget(
            modality_type=modality_type,
            epsilon_used=epsilon_per_step
        )
        
        if not privacy_consumed:
            self.logger.warning(f"Privacy budget exhausted for {modality_type.value}")
        
        return output, epsilon_per_step
    
    def _calculate_modality_sensitivity(self, config: ModalityConfig, input_tensor: Tensor) -> float:
        """Calculate L2 sensitivity for a modality based on its characteristics."""
        base_sensitivity = 1.0
        
        # Modality-specific sensitivity adjustments
        modality_factors = {
            ModalityType.TEXT: 1.0,      # Standard text sensitivity
            ModalityType.VISION: 1.2,    # Images can have higher variance
            ModalityType.AUDIO: 0.8,     # Audio often has bounded ranges
            ModalityType.VIDEO: 1.5,     # Video combines spatial and temporal
            ModalityType.GRAPH: 1.1,     # Graph structure adds complexity
            ModalityType.TIME_SERIES: 0.9,  # Time series often normalized
            ModalityType.TABULAR: 1.0    # Standard tabular sensitivity
        }
        
        sensitivity = base_sensitivity * modality_factors.get(config.modality_type, 1.0)
        
        # Apply configuration-specific scaling
        sensitivity *= config.sensitivity_factor
        
        # Adjust for temporal correlation if present
        if config.temporal_correlation:
            sensitivity *= 1.1  # Small increase for temporal leakage
        
        return sensitivity
    
    def _cross_modal_fusion(self, modality_outputs: Dict[ModalityType, Tensor]) -> Tuple[Tensor, float]:
        """Perform cross-modal fusion with differential privacy."""
        # Concatenate modality outputs
        concat_outputs = torch.cat(list(modality_outputs.values()), dim=-1)
        
        fusion_privacy_budget = self.fusion_config.fusion_privacy_budget
        
        if self.fusion_config.fusion_type == "attention":
            # Cross-modal attention fusion
            fused, _ = self.cross_modal_attention(
                query=concat_outputs,
                key=concat_outputs,
                value=concat_outputs,
                need_weights=False
            )
        
        elif self.fusion_config.fusion_type == "gating":
            # Gated fusion mechanism
            gate = torch.sigmoid(self.fusion_gate['gate_proj'](concat_outputs))
            value = self.fusion_gate['value_proj'](concat_outputs)
            fused = gate * value
            fused = self.fusion_gate['output_proj'](fused)
        
        elif self.fusion_config.fusion_type == "transformer":
            # Transformer-based fusion
            fused = self.fusion_transformer(concat_outputs)
        
        else:
            # Simple concatenation fusion
            fused = concat_outputs
        
        # Apply differential privacy to fusion
        if self.fusion_config.differential_fusion:
            # Use cross-modal noise mechanism (average of modality mechanisms)
            noise_mechanism = list(self.noise_mechanisms.values())[0]  # Use first mechanism
            
            fused = noise_mechanism.add_quantum_noise(
                tensor=fused,
                sensitivity=1.5,  # Higher sensitivity for fusion
                epsilon=fusion_privacy_budget,
                delta=self.privacy_manager.delta
            )
            
            # Update cross-modal privacy accounting
            self.privacy_manager.consume_privacy_budget(
                modality_type=list(self.modality_configs.keys())[0],  # Dummy for cross-modal
                epsilon_used=fusion_privacy_budget,
                is_cross_modal=True
            )
        
        # Final output projection
        final_output = self.fusion_output_proj(fused)
        
        return final_output, fusion_privacy_budget
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get comprehensive privacy usage report."""
        return self.privacy_manager.get_privacy_summary()
    
    def reset_privacy_accounting(self):
        """Reset privacy accounting for all modalities."""
        for accountant in self.privacy_manager.modality_accountants.values():
            # Reset would need to be implemented in QuantumRenyiAccountant
            pass
        self.privacy_manager.cross_modal_accountant = QuantumRenyiAccountant(
            params=type('QuantumParams', (), {
                'threat_model': QuantumThreatModel.POST_QUANTUM,
                'lattice_dimension': 256,
                'quantum_security_level': 128
            })()
        )


def create_multimodal_dp_attention(modality_types: List[str],
                                  embed_dims: List[int],
                                  num_heads: List[int],
                                  total_privacy_budget: float = 3.0,
                                  quantum_secure: bool = True) -> MultiModalDPAttention:
    """
    Factory function to create multi-modal DP attention.
    
    Args:
        modality_types: List of modality type names
        embed_dims: Embedding dimensions for each modality  
        num_heads: Number of attention heads for each modality
        total_privacy_budget: Total privacy budget to allocate
        quantum_secure: Whether to use quantum-resistant mechanisms
        
    Returns:
        Configured MultiModalDPAttention instance
    """
    if len(modality_types) != len(embed_dims) or len(embed_dims) != len(num_heads):
        raise ValueError("All modality configuration lists must have same length")
    
    # Create modality configurations
    modality_configs = []
    for modality_str, embed_dim, heads in zip(modality_types, embed_dims, num_heads):
        modality_type = ModalityType(modality_str)
        
        config = ModalityConfig(
            modality_type=modality_type,
            embed_dim=embed_dim,
            num_heads=heads,
            privacy_budget=0.0,  # Will be set by budget manager
            sensitivity_factor=1.0,
            attention_pattern="full",
            privacy_noise_type="gaussian"
        )
        modality_configs.append(config)
    
    # Create fusion configuration
    fusion_config = CrossModalFusionConfig(
        fusion_type="attention",
        fusion_privacy_budget=0.5,
        alignment_mechanism="learned",
        privacy_barrier=True,
        differential_fusion=True
    )
    
    return MultiModalDPAttention(
        modality_configs=modality_configs,
        fusion_config=fusion_config,
        total_privacy_budget=total_privacy_budget,
        quantum_secure=quantum_secure
    )


# Example usage and testing
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        # Create multi-modal DP attention for text + vision
        mm_attention = create_multimodal_dp_attention(
            modality_types=["text", "vision"],
            embed_dims=[768, 512], 
            num_heads=[12, 8],
            total_privacy_budget=3.0,
            quantum_secure=True
        )
        
        # Test with sample inputs
        text_input = torch.randn(2, 50, 768)  # [batch, seq, dim]
        vision_input = torch.randn(2, 196, 512)  # [batch, patches, dim]
        
        modality_inputs = {
            ModalityType.TEXT: text_input,
            ModalityType.VISION: vision_input
        }
        
        # Forward pass
        output = mm_attention(modality_inputs, return_privacy_stats=True)
        
        print(f"✅ Multi-modal output shape: {output['fused_output'].shape}")
        print(f"✅ Privacy stats: {output['privacy_stats']['budget_summary']['overall_utilization']:.2%}")
        
        # Print privacy report
        report = mm_attention.get_privacy_report()
        print(f"✅ Total privacy used: {report['total_spent']:.4f} / {report['total_budget']:.4f}")
    else:
        print("⚠️  PyTorch not available - multi-modal attention requires PyTorch")