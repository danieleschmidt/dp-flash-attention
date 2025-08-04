"""
Utility functions for DP-Flash-Attention.

Includes CUDA version detection, privacy parameter validation, and helper functions.
"""

import math
import platform
import subprocess
import warnings
from typing import Optional, Tuple, Union

import torch
import numpy as np


def cuda_version() -> str:
    """
    Get CUDA version information.
    
    Returns:
        CUDA version string or error message
    """
    try:
        # Check PyTorch CUDA version
        if torch.cuda.is_available():
            pytorch_cuda = torch.version.cuda
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # Try to get CUDA runtime version
            try:
                result = subprocess.run(
                    ['nvcc', '--version'], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'release' in line.lower():
                            nvcc_version = line.strip()
                            break
                    else:
                        nvcc_version = "NVCC version not found"
                else:
                    nvcc_version = "NVCC not available"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                nvcc_version = "NVCC not found"
            
            return (
                f"PyTorch CUDA: {pytorch_cuda}\n"
                f"Devices: {device_count}\n"
                f"Current: {current_device} ({device_name})\n"
                f"NVCC: {nvcc_version}"
            )
        else:
            return "CUDA not available in PyTorch"
            
    except Exception as e:
        return f"Error checking CUDA version: {e}"


def privacy_check() -> str:
    """
    Run basic privacy implementation checks.
    
    Returns:
        Status message about privacy implementation
    """
    try:
        # Check differential privacy libraries
        status = []
        
        try:
            import opacus
            status.append(f"✓ Opacus: {opacus.__version__}")
        except ImportError:
            status.append("✗ Opacus: Not installed")
        
        try:
            import dp_accounting
            status.append("✓ DP-Accounting: Available") 
        except ImportError:
            status.append("✗ DP-Accounting: Not installed")
        
        try:
            import prv_accountant
            status.append("✓ PRV-Accountant: Available")
        except ImportError:
            status.append("✗ PRV-Accountant: Not installed")
        
        # Check CUDA availability for kernels
        if torch.cuda.is_available():
            # Test basic CUDA operations
            try:
                x = torch.randn(100, 100, device='cuda')
                y = torch.randn(100, 100, device='cuda')
                z = torch.mm(x, y)
                status.append("✓ CUDA operations: Working")
            except Exception as e:
                status.append(f"✗ CUDA operations: Error - {e}")
        else:
            status.append("✗ CUDA: Not available")
        
        # Test noise generation
        try:
            noise = torch.normal(0, 1.0, (1000,))
            std_dev = torch.std(noise).item()
            if 0.9 < std_dev < 1.1:
                status.append("✓ Noise generation: Working")
            else:
                status.append(f"⚠ Noise generation: Unusual std {std_dev:.3f}")
        except Exception as e:
            status.append(f"✗ Noise generation: Error - {e}")
        
        return "\n".join(status)
        
    except Exception as e:
        return f"Error running privacy checks: {e}"


def validate_privacy_params(epsilon: float, delta: float) -> None:
    """
    Validate differential privacy parameters.
    
    Args:
        epsilon: Privacy budget (ε)
        delta: Privacy parameter (δ)
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(epsilon, (int, float)):
        raise ValueError(f"epsilon must be numeric, got {type(epsilon)}")
    
    if not isinstance(delta, (int, float)):
        raise ValueError(f"delta must be numeric, got {type(delta)}")
    
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    
    # Warn about weak privacy parameters
    if epsilon > 10.0:
        warnings.warn(
            f"Large epsilon value {epsilon} provides weak privacy guarantees. "
            "Consider using epsilon < 10 for meaningful privacy."
        )
    
    if delta > 1e-3:
        warnings.warn(
            f"Large delta value {delta} provides weak privacy guarantees. "
            "Consider using delta < 1e-5 for strong privacy."
        )


def compute_noise_scale(
    epsilon: float, 
    delta: float, 
    max_grad_norm: float,
    sequence_length: int,
    composition_steps: int = 1,
) -> float:
    """
    Compute noise scale for Gaussian mechanism.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter
        max_grad_norm: L2 sensitivity (clipping bound)
        sequence_length: Length of input sequence
        composition_steps: Number of composition steps
        
    Returns:
        Noise standard deviation
    """
    validate_privacy_params(epsilon, delta)
    
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
    
    # Account for composition across steps
    effective_epsilon = epsilon / composition_steps
    
    # Gaussian mechanism noise scale: σ = √(2 ln(1.25/δ)) * Δ / ε
    # where Δ is the L2 sensitivity
    noise_scale = math.sqrt(2 * math.log(1.25 / delta)) * max_grad_norm / effective_epsilon
    
    # Account for sequence length (simplified - actual sensitivity analysis more complex)
    if sequence_length > 512:
        # Longer sequences may have higher sensitivity
        length_factor = math.sqrt(sequence_length / 512)
        noise_scale *= length_factor
    
    return noise_scale


def clip_gradients(
    gradients: torch.Tensor,
    max_norm: float,
    norm_type: float = 2.0,
) -> Tuple[torch.Tensor, float]:
    """
    Clip gradients to bound sensitivity.
    
    Args:
        gradients: Gradient tensor
        max_norm: Maximum allowed norm
        norm_type: Type of norm (2.0 for L2)
        
    Returns:
        Tuple of (clipped_gradients, actual_norm)
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    
    # Compute gradient norm
    if norm_type == 2.0:
        grad_norm = torch.norm(gradients, p=2).item()
    elif norm_type == 1.0:
        grad_norm = torch.norm(gradients, p=1).item()
    else:
        grad_norm = torch.norm(gradients, p=norm_type).item()
    
    # Clip if necessary
    if grad_norm > max_norm:
        clip_factor = max_norm / grad_norm
        clipped_gradients = gradients * clip_factor
    else:
        clipped_gradients = gradients
    
    return clipped_gradients, grad_norm


def estimate_memory_usage(
    batch_size: int,
    sequence_length: int, 
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Estimate memory usage for DP-Flash-Attention.
    
    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Bytes per element
    if dtype == torch.float32:
        bytes_per_element = 4
    elif dtype == torch.float16 or dtype == torch.bfloat16:
        bytes_per_element = 2
    else:
        bytes_per_element = 4  # Default assumption
    
    # Input tensors (Q, K, V)
    input_size = batch_size * sequence_length * num_heads * head_dim * bytes_per_element
    total_input = 3 * input_size  # Q, K, V
    
    # Attention scores (before softmax)
    scores_size = batch_size * num_heads * sequence_length * sequence_length * bytes_per_element
    
    # Output tensor
    output_size = input_size
    
    # Noise buffer (for DP)
    noise_size = scores_size  # Noise added to attention scores
    
    # Additional working memory (approximation)
    working_memory = max(scores_size, output_size) * 2
    
    total_mb = (total_input + scores_size + output_size + noise_size + working_memory) / (1024 ** 2)
    
    return {
        'input_tensors_mb': total_input / (1024 ** 2),
        'attention_scores_mb': scores_size / (1024 ** 2),
        'output_tensor_mb': output_size / (1024 ** 2),
        'noise_buffer_mb': noise_size / (1024 ** 2),
        'working_memory_mb': working_memory / (1024 ** 2),
        'total_estimated_mb': total_mb,
    }


def check_system_requirements() -> dict:
    """
    Check system requirements for DP-Flash-Attention.
    
    Returns:
        Dictionary with requirement check results
    """
    results = {}
    
    # Python version
    python_version = platform.python_version()
    results['python_version'] = python_version
    results['python_ok'] = python_version >= '3.10'
    
    # PyTorch version
    torch_version = torch.__version__
    results['torch_version'] = torch_version
    results['torch_ok'] = torch_version >= '2.3.0'
    
    # CUDA availability
    results['cuda_available'] = torch.cuda.is_available()
    if results['cuda_available']:
        results['cuda_version'] = torch.version.cuda
        results['cuda_devices'] = torch.cuda.device_count()
        results['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        # Check for compute capability
        if hasattr(torch.cuda, 'get_device_capability'):
            major, minor = torch.cuda.get_device_capability(0)
            results['compute_capability'] = f"{major}.{minor}"
            results['compute_ok'] = major >= 8  # Ampere or newer for best performance
        else:
            results['compute_capability'] = "Unknown"
            results['compute_ok'] = None
    else:
        results['cuda_ok'] = False
    
    # Memory check
    if results['cuda_available']:
        try:
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            results['gpu_memory_gb'] = memory_gb
            results['memory_ok'] = memory_gb >= 16  # Minimum 16GB recommended
        except:
            results['gpu_memory_gb'] = "Unknown"
            results['memory_ok'] = None
    
    # Package availability
    packages = ['triton', 'einops', 'numpy', 'ninja', 'pybind11']
    for pkg in packages:
        try:
            __import__(pkg)
            results[f'{pkg}_available'] = True
        except ImportError:
            results[f'{pkg}_available'] = False
    
    return results


def benchmark_attention_kernel(
    batch_size: int = 32,
    sequence_length: int = 512,
    num_heads: int = 12,
    head_dim: int = 64,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> dict:
    """
    Benchmark DP-Flash-Attention kernel performance.
    
    Args:
        batch_size: Batch size for benchmark
        sequence_length: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        num_iterations: Number of timing iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Benchmark results dictionary
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    device = torch.cuda.current_device()
    dtype = torch.float16
    
    # Create test tensors
    q = torch.randn(batch_size, sequence_length, num_heads, head_dim, 
                   device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, sequence_length, num_heads, head_dim,
                   device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, sequence_length, num_heads, head_dim,
                   device=device, dtype=dtype, requires_grad=True)
    
    # Warmup
    for _ in range(warmup_iterations):
        try:
            # Use standard attention for benchmarking (DP kernel not fully implemented)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)
            torch.cuda.synchronize()
        except:
            return {'error': 'Warmup failed'}
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    for _ in range(num_iterations):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
    
    end_time.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_time.elapsed_time(end_time)
    avg_time_ms = total_time_ms / num_iterations
    
    # Memory usage
    memory_stats = estimate_memory_usage(batch_size, sequence_length, num_heads, head_dim, dtype)
    
    return {
        'avg_time_ms': avg_time_ms,
        'total_time_ms': total_time_ms,
        'iterations': num_iterations,
        'throughput_samples_per_sec': batch_size * 1000 / avg_time_ms,
        'memory_usage_mb': memory_stats['total_estimated_mb'],
        'configuration': {
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'dtype': str(dtype),
        }
    }