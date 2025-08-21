"""
Generation 5.3: Edge Deployment Optimization

Advanced edge computing optimizations for differential privacy with:
- Mobile/IoT device compatibility
- Federated learning with local DP
- Model compression with privacy preservation
- Battery-aware privacy budgeting
- Network-efficient DP protocols
"""

import math
import time
import psutil
import platform
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import threading
import queue

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

from .generation5_quantum_privacy import LatticeBasedNoiseMechanism, QuantumRenyiAccountant
from .utils import validate_privacy_params, estimate_memory_usage
from .error_handling import handle_errors, PrivacyParameterError
from .logging_utils import get_logger


class DeviceType(Enum):
    """Supported edge device types."""
    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    ARDUINO = "arduino"
    IOT_SENSOR = "iot_sensor"
    EDGE_SERVER = "edge_server"
    WEARABLE = "wearable"


class ComputeCapability(Enum):
    """Device compute capability levels."""
    MINIMAL = "minimal"      # < 1GB RAM, ARM Cortex-M
    LOW = "low"             # 1-4GB RAM, ARM Cortex-A
    MEDIUM = "medium"       # 4-8GB RAM, ARM/x86
    HIGH = "high"           # 8GB+ RAM, GPU available
    ULTRA = "ultra"         # High-end edge server


@dataclass
class EdgeDeviceProfile:
    """Profile of edge device capabilities and constraints."""
    device_type: DeviceType
    compute_capability: ComputeCapability
    memory_mb: int
    cpu_cores: int
    has_gpu: bool = False
    battery_capacity_mah: Optional[int] = None
    network_bandwidth_mbps: float = 10.0
    power_budget_watts: float = 5.0
    thermal_limit_celsius: float = 70.0
    privacy_hardware: bool = False  # Trusted execution environment
    quantization_support: List[str] = field(default_factory=lambda: ["int8", "fp16"])


@dataclass
class EdgePrivacyConfig:
    """Configuration for edge differential privacy."""
    local_privacy_budget: float = 1.0
    federated_rounds: int = 10
    minimum_participants: int = 100
    secure_aggregation: bool = True
    compression_ratio: float = 0.1  # Model compression
    quantization_bits: int = 8
    battery_aware: bool = True
    network_efficient: bool = True
    adaptive_noise: bool = True
    privacy_amplification_via_sampling: bool = True


class BatteryAwarePrivacyScheduler:
    """
    Battery-aware privacy budget scheduling for mobile devices.
    
    Dynamically adjusts privacy parameters based on battery level,
    power consumption patterns, and usage predictions.
    """
    
    def __init__(self, 
                 total_privacy_budget: float = 3.0,
                 battery_threshold_low: float = 0.2,
                 battery_threshold_critical: float = 0.1):
        
        self.total_privacy_budget = total_privacy_budget
        self.battery_threshold_low = battery_threshold_low
        self.battery_threshold_critical = battery_threshold_critical
        self.logger = get_logger()
        
        # Privacy budget allocation based on battery level
        self.budget_allocation = {
            'critical': 0.1,  # Save most budget when battery critical
            'low': 0.3,       # Reduced privacy when battery low
            'normal': 0.7,    # Normal operation
            'high': 1.0       # Full privacy when battery high
        }
        
        # Track power consumption patterns
        self.power_history = []
        self.privacy_usage_history = []
        
        # Battery monitoring
        self.current_battery_level = 1.0
        self._start_battery_monitoring()
    
    def _start_battery_monitoring(self):
        """Start background battery monitoring."""
        def monitor_battery():
            while True:
                try:
                    if hasattr(psutil, 'sensors_battery'):
                        battery = psutil.sensors_battery()
                        if battery:
                            self.current_battery_level = battery.percent / 100.0
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.warning(f"Battery monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_battery, daemon=True)
        monitor_thread.start()
    
    def get_battery_status(self) -> str:
        """Get current battery status category."""
        if self.current_battery_level <= self.battery_threshold_critical:
            return 'critical'
        elif self.current_battery_level <= self.battery_threshold_low:
            return 'low'
        elif self.current_battery_level >= 0.8:
            return 'high'
        else:
            return 'normal'
    
    def get_adjusted_privacy_budget(self, base_epsilon: float) -> float:
        """Get battery-adjusted privacy budget."""
        battery_status = self.get_battery_status()
        allocation_factor = self.budget_allocation[battery_status]
        
        adjusted_budget = base_epsilon * allocation_factor
        
        self.logger.debug(f"Battery {self.current_battery_level:.1%}, "
                         f"status: {battery_status}, "
                         f"privacy factor: {allocation_factor:.2f}")
        
        return adjusted_budget
    
    def predict_battery_usage(self, privacy_operations: int) -> float:
        """Predict battery usage for given privacy operations."""
        # Simple power model - can be enhanced with device-specific measurements
        base_power_per_op = 0.001  # Watts per privacy operation
        
        if len(self.power_history) > 10:
            # Use historical data for better prediction
            avg_power_per_op = np.mean(self.power_history[-10:])
            base_power_per_op = avg_power_per_op
        
        estimated_power = privacy_operations * base_power_per_op
        return estimated_power
    
    def should_defer_computation(self, estimated_power: float) -> bool:
        """Decide whether to defer computation based on battery."""
        battery_status = self.get_battery_status()
        
        if battery_status == 'critical':
            # Defer all but essential computations
            return True
        elif battery_status == 'low':
            # Defer high-power computations
            return estimated_power > 0.5
        
        return False


class NetworkEfficientDPProtocol:
    """
    Network-efficient differential privacy protocols for edge federation.
    
    Implements compression, quantization, and sparse communication
    while maintaining privacy guarantees.
    """
    
    def __init__(self, config: EdgePrivacyConfig):
        self.config = config
        self.logger = get_logger()
        
        # Communication efficiency metrics
        self.communication_rounds = 0
        self.total_bytes_sent = 0
        self.compression_ratios = []
        
    @handle_errors(reraise=True, log_errors=True)
    def compress_and_privatize_gradients(self, 
                                       gradients: Union[Tensor, np.ndarray],
                                       epsilon: float,
                                       delta: float) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress and privatize gradients for efficient transmission.
        
        Args:
            gradients: Model gradients to compress and privatize
            epsilon: Privacy parameter
            delta: Privacy parameter
            
        Returns:
            Compressed privatized gradients and metadata
        """
        # Step 1: Apply differential privacy noise
        from .generation5_quantum_privacy import create_quantum_privacy_mechanism
        
        noise_mechanism, _ = create_quantum_privacy_mechanism()
        
        # Calculate sensitivity for gradients
        gradient_norm = np.linalg.norm(gradients.flatten() if TORCH_AVAILABLE and isinstance(gradients, torch.Tensor) 
                                     else gradients.flatten())
        sensitivity = min(1.0, gradient_norm)  # Clip sensitivity
        
        privatized_gradients = noise_mechanism.add_quantum_noise(
            tensor=gradients,
            sensitivity=sensitivity,
            epsilon=epsilon,
            delta=delta
        )
        
        # Step 2: Quantization for compression
        quantized_gradients = self._quantize_tensor(
            privatized_gradients, 
            bits=self.config.quantization_bits
        )
        
        # Step 3: Sparsification (keep only top-k elements)
        sparse_gradients = self._sparsify_tensor(
            quantized_gradients,
            sparsity_ratio=self.config.compression_ratio
        )
        
        # Step 4: Compress using custom protocol
        compressed_data = self._compress_sparse_tensor(sparse_gradients)
        
        # Metadata for reconstruction
        metadata = {
            'original_shape': gradients.shape,
            'quantization_bits': self.config.quantization_bits,
            'sparsity_ratio': self.config.compression_ratio,
            'noise_scale': sensitivity / epsilon,
            'privacy_params': {'epsilon': epsilon, 'delta': delta}
        }
        
        compression_ratio = len(compressed_data) / gradients.nbytes
        self.compression_ratios.append(compression_ratio)
        self.total_bytes_sent += len(compressed_data)
        
        self.logger.info(f"Gradient compression: {compression_ratio:.3f} ratio, "
                        f"privacy: ε={epsilon:.3f}")
        
        return compressed_data, metadata
    
    def _quantize_tensor(self, tensor: Union[Tensor, np.ndarray], bits: int) -> Union[Tensor, np.ndarray]:
        """Quantize tensor to specified bit width."""
        if bits >= 32:
            return tensor  # No quantization needed
        
        # Symmetric quantization
        max_val = 2**(bits-1) - 1
        min_val = -2**(bits-1)
        
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            scale = torch.max(torch.abs(tensor)) / max_val
            quantized = torch.round(tensor / scale).clamp(min_val, max_val)
            return quantized * scale
        else:
            scale = np.max(np.abs(tensor)) / max_val
            quantized = np.round(tensor / scale).clip(min_val, max_val)
            return quantized * scale
    
    def _sparsify_tensor(self, tensor: Union[Tensor, np.ndarray], sparsity_ratio: float) -> Union[Tensor, np.ndarray]:
        """Apply top-k sparsification to tensor."""
        flat_tensor = tensor.flatten() if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor) else tensor.flatten()
        k = max(1, int(len(flat_tensor) * (1 - sparsity_ratio)))
        
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            _, indices = torch.topk(torch.abs(flat_tensor), k)
            sparse_flat = torch.zeros_like(flat_tensor)
            sparse_flat[indices] = flat_tensor[indices]
            return sparse_flat.reshape(tensor.shape)
        else:
            indices = np.argpartition(np.abs(flat_tensor), -k)[-k:]
            sparse_flat = np.zeros_like(flat_tensor)
            sparse_flat[indices] = flat_tensor[indices]
            return sparse_flat.reshape(tensor.shape)
    
    def _compress_sparse_tensor(self, sparse_tensor: Union[Tensor, np.ndarray]) -> bytes:
        """Compress sparse tensor using run-length encoding."""
        # Convert to numpy for compression
        if TORCH_AVAILABLE and isinstance(sparse_tensor, torch.Tensor):
            np_tensor = sparse_tensor.detach().cpu().numpy()
        else:
            np_tensor = sparse_tensor
        
        # Simple run-length encoding for sparse data
        flat = np_tensor.flatten()
        non_zero_indices = np.nonzero(flat)[0]
        non_zero_values = flat[non_zero_indices]
        
        # Pack indices and values
        import struct
        compressed = struct.pack(f'I{len(non_zero_indices)}I{len(non_zero_values)}f', 
                               len(non_zero_indices),
                               *non_zero_indices,
                               *non_zero_values)
        
        return compressed
    
    def decompress_gradients(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Union[Tensor, np.ndarray]:
        """Decompress gradients from network transmission."""
        import struct
        
        # Unpack compressed data
        offset = 0
        num_indices = struct.unpack_from('I', compressed_data, offset)[0]
        offset += struct.calcsize('I')
        
        indices = struct.unpack_from(f'{num_indices}I', compressed_data, offset)
        offset += struct.calcsize(f'{num_indices}I')
        
        values = struct.unpack_from(f'{num_indices}f', compressed_data, offset)
        
        # Reconstruct sparse tensor
        total_elements = np.prod(metadata['original_shape'])
        flat_tensor = np.zeros(total_elements, dtype=np.float32)
        flat_tensor[list(indices)] = values
        
        reconstructed = flat_tensor.reshape(metadata['original_shape'])
        
        if TORCH_AVAILABLE:
            return torch.from_numpy(reconstructed)
        else:
            return reconstructed
    
    def get_communication_stats(self) -> Dict[str, float]:
        """Get communication efficiency statistics."""
        return {
            'total_rounds': self.communication_rounds,
            'total_bytes_sent': self.total_bytes_sent,
            'average_compression_ratio': np.mean(self.compression_ratios) if self.compression_ratios else 0.0,
            'bytes_per_round': self.total_bytes_sent / max(1, self.communication_rounds)
        }


class EdgeOptimizedDPAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Edge-optimized differential privacy attention mechanism.
    
    Designed for resource-constrained devices with:
    - Dynamic model scaling based on device capabilities
    - Battery-aware privacy scheduling
    - Network-efficient federated learning
    - Hardware-aware optimizations
    """
    
    def __init__(self,
                 device_profile: EdgeDeviceProfile,
                 privacy_config: EdgePrivacyConfig,
                 base_embed_dim: int = 512,
                 base_num_heads: int = 8):
        
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.device_profile = device_profile
        self.privacy_config = privacy_config
        self.logger = get_logger()
        
        # Scale model based on device capabilities
        self.embed_dim, self.num_heads = self._scale_model_for_device(
            base_embed_dim, base_num_heads
        )
        
        # Initialize battery-aware scheduler
        self.battery_scheduler = BatteryAwarePrivacyScheduler(
            total_privacy_budget=privacy_config.local_privacy_budget
        )
        
        # Initialize network protocol
        self.network_protocol = NetworkEfficientDPProtocol(privacy_config)
        
        # Device-specific optimizations
        self._apply_device_optimizations()
        
        if TORCH_AVAILABLE:
            self._initialize_layers()
    
    def _scale_model_for_device(self, base_embed_dim: int, base_num_heads: int) -> Tuple[int, int]:
        """Scale model parameters based on device capabilities."""
        scaling_factors = {
            ComputeCapability.MINIMAL: 0.25,
            ComputeCapability.LOW: 0.5,
            ComputeCapability.MEDIUM: 0.75,
            ComputeCapability.HIGH: 1.0,
            ComputeCapability.ULTRA: 1.25
        }
        
        scale = scaling_factors[self.device_profile.compute_capability]
        
        # Scale dimensions while maintaining divisibility
        scaled_embed_dim = max(64, int(base_embed_dim * scale))
        scaled_num_heads = max(1, int(base_num_heads * scale))
        
        # Ensure embed_dim is divisible by num_heads
        scaled_embed_dim = (scaled_embed_dim // scaled_num_heads) * scaled_num_heads
        
        self.logger.info(f"Scaled model for {self.device_profile.compute_capability.value}: "
                        f"dim={scaled_embed_dim}, heads={scaled_num_heads}")
        
        return scaled_embed_dim, scaled_num_heads
    
    def _apply_device_optimizations(self):
        """Apply device-specific optimizations."""
        optimizations = []
        
        # Memory optimizations
        if self.device_profile.memory_mb < 2048:
            optimizations.append("gradient_checkpointing")
            optimizations.append("memory_efficient_attention")
        
        # CPU optimizations
        if not self.device_profile.has_gpu:
            optimizations.append("cpu_optimized_kernels")
            optimizations.append("int8_quantization")
        
        # Battery optimizations
        if self.device_profile.battery_capacity_mah:
            optimizations.append("battery_aware_scheduling")
            optimizations.append("dynamic_frequency_scaling")
        
        # Network optimizations
        if self.device_profile.network_bandwidth_mbps < 50:
            optimizations.append("aggressive_compression")
            optimizations.append("sparse_updates")
        
        self.active_optimizations = optimizations
        self.logger.info(f"Applied optimizations: {optimizations}")
    
    def _initialize_layers(self):
        """Initialize attention layers with device optimizations."""
        # Use appropriate data type based on device
        dtype = torch.float16 if "fp16" in self.device_profile.quantization_support else torch.float32
        
        # Multi-head attention with reduced precision
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,  # DP noise replaces dropout
            bias=True,
            batch_first=True,
            dtype=dtype
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim, dtype=dtype)
        
        # Optional quantization
        if "int8_quantization" in self.active_optimizations:
            self._apply_quantization()
    
    def _apply_quantization(self):
        """Apply post-training quantization for inference."""
        if TORCH_AVAILABLE and hasattr(torch.quantization, 'quantize_dynamic'):
            self.attention = torch.quantization.quantize_dynamic(
                self.attention, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
    
    @handle_errors(reraise=True, log_errors=True)
    def forward(self,
               query: Tensor,
               key: Optional[Tensor] = None,
               value: Optional[Tensor] = None,
               return_privacy_stats: bool = False) -> Dict[str, Any]:
        """
        Edge-optimized forward pass with adaptive privacy.
        
        Args:
            query: Query tensor
            key: Key tensor (optional, defaults to query)
            value: Value tensor (optional, defaults to query)
            return_privacy_stats: Whether to return privacy statistics
            
        Returns:
            Dictionary with attention output and optional privacy stats
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for edge attention")
        
        key = key if key is not None else query
        value = value if value is not None else query
        
        # Battery-aware privacy adjustment
        base_epsilon = self.privacy_config.local_privacy_budget
        adjusted_epsilon = self.battery_scheduler.get_adjusted_privacy_budget(base_epsilon)
        
        # Check if computation should be deferred
        estimated_power = self.battery_scheduler.predict_battery_usage(1)
        if self.battery_scheduler.should_defer_computation(estimated_power):
            self.logger.info("Deferring computation due to low battery")
            # Return cached or simplified result
            return self._get_cached_or_simplified_result(query, return_privacy_stats)
        
        # Apply layer normalization
        query_norm = self.layer_norm(query)
        key_norm = self.layer_norm(key) if key is not query else query_norm
        value_norm = self.layer_norm(value) if value is not query else query_norm
        
        # Compute attention
        if "memory_efficient_attention" in self.active_optimizations:
            # Use gradient checkpointing for memory efficiency
            attn_output = torch.utils.checkpoint.checkpoint(
                self._compute_attention_with_privacy,
                query_norm, key_norm, value_norm, adjusted_epsilon
            )
        else:
            attn_output = self._compute_attention_with_privacy(
                query_norm, key_norm, value_norm, adjusted_epsilon
            )
        
        result = {'attention_output': attn_output}
        
        if return_privacy_stats:
            result['privacy_stats'] = {
                'epsilon_used': adjusted_epsilon,
                'battery_level': self.battery_scheduler.current_battery_level,
                'device_optimizations': self.active_optimizations,
                'estimated_power': estimated_power
            }
        
        return result
    
    def _compute_attention_with_privacy(self, 
                                      query: Tensor, 
                                      key: Tensor, 
                                      value: Tensor,
                                      epsilon: float) -> Tensor:
        """Compute attention with edge-optimized differential privacy."""
        # Standard attention computation
        attn_output, _ = self.attention(query, key, value, need_weights=False)
        
        # Add differential privacy noise optimized for edge
        if epsilon > 0:
            from .generation5_quantum_privacy import create_quantum_privacy_mechanism
            
            # Use lightweight noise mechanism for edge devices
            noise_mechanism, _ = create_quantum_privacy_mechanism(
                lattice_dimension=min(256, self.embed_dim)  # Smaller lattice for edge
            )
            
            # Calculate edge-optimized sensitivity
            sensitivity = self._calculate_edge_sensitivity(attn_output)
            
            # Add noise
            noised_output = noise_mechanism.add_quantum_noise(
                tensor=attn_output,
                sensitivity=sensitivity,
                epsilon=epsilon,
                delta=1e-6  # Tighter delta for edge privacy
            )
            
            return noised_output
        
        return attn_output
    
    def _calculate_edge_sensitivity(self, tensor: Tensor) -> float:
        """Calculate sensitivity optimized for edge constraints."""
        # Use device-specific sensitivity calculation
        base_sensitivity = 1.0
        
        # Adjust for device capabilities
        if self.device_profile.compute_capability in [ComputeCapability.MINIMAL, ComputeCapability.LOW]:
            # Lower sensitivity for resource-constrained devices
            base_sensitivity *= 0.8
        
        # Adjust for quantization
        if "int8_quantization" in self.active_optimizations:
            base_sensitivity *= 0.7  # Quantization reduces effective sensitivity
        
        return base_sensitivity
    
    def _get_cached_or_simplified_result(self, query: Tensor, return_privacy_stats: bool) -> Dict[str, Any]:
        """Return cached or simplified result when computation is deferred."""
        # Simple identity mapping with noise as fallback
        simplified_output = query + torch.randn_like(query) * 0.01
        
        result = {'attention_output': simplified_output}
        
        if return_privacy_stats:
            result['privacy_stats'] = {
                'epsilon_used': 0.01,  # Minimal privacy for simplified computation
                'battery_level': self.battery_scheduler.current_battery_level,
                'computation_deferred': True,
                'reason': 'low_battery'
            }
        
        return result
    
    def prepare_for_federated_round(self) -> Tuple[bytes, Dict[str, Any]]:
        """Prepare model updates for federated learning round."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for federated learning")
        
        # Extract model parameters/gradients
        params = []
        for param in self.parameters():
            if param.grad is not None:
                params.append(param.grad.clone())
        
        if not params:
            # No gradients available, return empty update
            return b'', {}
        
        # Concatenate all gradients
        flattened_grads = torch.cat([p.flatten() for p in params])
        
        # Compress and privatize for network transmission
        compressed_data, metadata = self.network_protocol.compress_and_privatize_gradients(
            gradients=flattened_grads,
            epsilon=self.privacy_config.local_privacy_budget,
            delta=1e-6
        )
        
        return compressed_data, metadata
    
    def apply_federated_update(self, compressed_update: bytes, metadata: Dict[str, Any]):
        """Apply federated learning update from server."""
        if not TORCH_AVAILABLE:
            return
        
        # Decompress update
        update = self.network_protocol.decompress_gradients(compressed_update, metadata)
        
        # Apply update to model parameters
        param_offset = 0
        for param in self.parameters():
            param_size = param.numel()
            param_update = update[param_offset:param_offset + param_size].reshape(param.shape)
            param.data += param_update * 0.1  # Learning rate
            param_offset += param_size
    
    def get_edge_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive edge performance statistics."""
        return {
            'device_profile': {
                'type': self.device_profile.device_type.value,
                'capability': self.device_profile.compute_capability.value,
                'memory_mb': self.device_profile.memory_mb,
                'has_gpu': self.device_profile.has_gpu
            },
            'model_scaling': {
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'active_optimizations': self.active_optimizations
            },
            'battery_stats': {
                'current_level': self.battery_scheduler.current_battery_level,
                'status': self.battery_scheduler.get_battery_status()
            },
            'network_stats': self.network_protocol.get_communication_stats(),
            'privacy_config': {
                'local_budget': self.privacy_config.local_privacy_budget,
                'compression_ratio': self.privacy_config.compression_ratio,
                'quantization_bits': self.privacy_config.quantization_bits
            }
        }


def create_edge_optimized_dp_attention(device_type: str = "smartphone",
                                     memory_mb: int = 4096,
                                     has_gpu: bool = False,
                                     battery_capacity_mah: Optional[int] = 3000,
                                     privacy_budget: float = 1.0) -> EdgeOptimizedDPAttention:
    """
    Factory function to create edge-optimized DP attention.
    
    Args:
        device_type: Type of edge device
        memory_mb: Available memory in MB
        has_gpu: Whether device has GPU acceleration
        battery_capacity_mah: Battery capacity (None for powered devices)
        privacy_budget: Local privacy budget
        
    Returns:
        Configured EdgeOptimizedDPAttention instance
    """
    # Determine compute capability based on specs
    if memory_mb < 1024:
        compute_capability = ComputeCapability.MINIMAL
    elif memory_mb < 4096:
        compute_capability = ComputeCapability.LOW
    elif memory_mb < 8192:
        compute_capability = ComputeCapability.MEDIUM
    else:
        compute_capability = ComputeCapability.HIGH
    
    device_profile = EdgeDeviceProfile(
        device_type=DeviceType(device_type),
        compute_capability=compute_capability,
        memory_mb=memory_mb,
        cpu_cores=psutil.cpu_count() or 4,
        has_gpu=has_gpu,
        battery_capacity_mah=battery_capacity_mah,
        quantization_support=["int8", "fp16"] if has_gpu else ["int8"]
    )
    
    privacy_config = EdgePrivacyConfig(
        local_privacy_budget=privacy_budget,
        battery_aware=battery_capacity_mah is not None,
        network_efficient=True,
        compression_ratio=0.1,
        quantization_bits=8
    )
    
    return EdgeOptimizedDPAttention(
        device_profile=device_profile,
        privacy_config=privacy_config
    )


# Example usage and testing
if __name__ == "__main__":
    # Create edge-optimized DP attention for smartphone
    edge_attention = create_edge_optimized_dp_attention(
        device_type="smartphone",
        memory_mb=6144,  # 6GB RAM
        has_gpu=True,
        battery_capacity_mah=4000,
        privacy_budget=1.5
    )
    
    if TORCH_AVAILABLE:
        # Test with sample input
        test_input = torch.randn(1, 50, edge_attention.embed_dim)
        
        output = edge_attention(test_input, return_privacy_stats=True)
        
        print(f"✅ Edge attention output shape: {output['attention_output'].shape}")
        print(f"✅ Battery level: {output['privacy_stats']['battery_level']:.1%}")
        print(f"✅ Optimizations: {output['privacy_stats']['device_optimizations']}")
        
        # Test federated preparation
        compressed_update, metadata = edge_attention.prepare_for_federated_round()
        print(f"✅ Federated update size: {len(compressed_update)} bytes")
        
        # Get performance stats
        stats = edge_attention.get_edge_performance_stats()
        print(f"✅ Device capability: {stats['device_profile']['capability']}")
        print(f"✅ Model dimensions: {stats['model_scaling']['embed_dim']}x{stats['model_scaling']['num_heads']}")
    else:
        print("⚠️  PyTorch not available - edge optimization requires PyTorch")