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
]