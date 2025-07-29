"""
DP-Flash-Attention: Hardware-accelerated differentially private Flash-Attention 3.

This library provides CUDA-optimized implementations of differential privacy
mechanisms integrated directly into Flash-Attention 3 kernels.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@dp-flash-attention.org"

from .core import DPFlashAttention
from .functional import dp_flash_attn_func
from .privacy import RenyiAccountant
from .utils import cuda_version, privacy_check

__all__ = [
    "DPFlashAttention",
    "dp_flash_attn_func", 
    "RenyiAccountant",
    "cuda_version",
    "privacy_check",
]