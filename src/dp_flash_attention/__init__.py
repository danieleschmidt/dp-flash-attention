"""
DP-Flash-Attention: Hardware-accelerated differentially private Flash-Attention 3.

This library provides CUDA-optimized implementations of differential privacy
mechanisms integrated directly into Flash-Attention 3 kernels.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@dp-flash-attention.org"

from .core import DPFlashAttention, DPCheckpointedAttention
from .functional import dp_flash_attn_func, dp_flash_attn_varlen_func, make_model_differentially_private
from .privacy import RenyiAccountant, GaussianMechanism, AdaptiveNoiseCalibrator, PrivacyStats
from .utils import cuda_version, privacy_check, validate_privacy_params, compute_noise_scale
from .kernels import dp_flash_attention_kernel, get_kernel_info
from .error_handling import (DPFlashAttentionError, PrivacyParameterError, 
                            CUDACompatibilityError, TensorShapeError,
                            PrivacyBudgetExceededError, handle_errors)
from .logging_utils import get_logger, setup_logging, PerformanceMonitor
from .security import get_secure_rng, get_input_validator, get_privacy_auditor
from . import cli

# Import scaling and optimization components (with error handling)
try:
    from .optimization import get_global_optimizer, optimize_attention_globally
    from .concurrent import get_global_processor, parallel_attention_batch
    from .autoscaling import AutoScaler, ScalingPolicy
    from .performance_tuning import auto_tune_for_hardware, OptimizationLevel
    from .distributed import DistributedStrategy, create_distributed_config
    _SCALING_AVAILABLE = True
except ImportError as e:
    # Graceful degradation if scaling modules have issues
    _SCALING_AVAILABLE = False
    import warnings
    warnings.warn(f"Scaling features not available: {e}")

__all__ = [
    "DPFlashAttention",
    "DPCheckpointedAttention",
    "dp_flash_attn_func", 
    "dp_flash_attn_varlen_func",
    "make_model_differentially_private",
    "RenyiAccountant",
    "GaussianMechanism", 
    "AdaptiveNoiseCalibrator",
    "PrivacyStats",
    "cuda_version",
    "privacy_check",
    "validate_privacy_params",
    "compute_noise_scale",
    "dp_flash_attention_kernel",
    "get_kernel_info",
    # Error handling
    "DPFlashAttentionError",
    "PrivacyParameterError", 
    "CUDACompatibilityError",
    "TensorShapeError",
    "PrivacyBudgetExceededError",
    "handle_errors",
    # Logging and monitoring
    "get_logger",
    "setup_logging",
    "PerformanceMonitor",
    # Security
    "get_secure_rng",
    "get_input_validator", 
    "get_privacy_auditor",
]

# Add scaling features if available
if _SCALING_AVAILABLE:
    __all__.extend([
        # Performance optimization
        "get_global_optimizer",
        "optimize_attention_globally",
        # Concurrency and scaling
        "get_global_processor",
        "parallel_attention_batch", 
        "AutoScaler",
        "ScalingPolicy",
        # Performance tuning
        "auto_tune_for_hardware",
        "OptimizationLevel",
        # Distributed processing
        "DistributedStrategy",
        "create_distributed_config",
    ])