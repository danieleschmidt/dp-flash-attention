"""
Comprehensive input validation and error handling for DP-Flash-Attention.

Provides robust validation of tensors, privacy parameters, and system requirements
with detailed error messages and recovery suggestions.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
from torch import Tensor
import numpy as np


class DPFlashAttentionError(Exception):
    """Base exception for DP-Flash-Attention errors."""
    pass


class PrivacyParameterError(DPFlashAttentionError):
    """Exception for invalid privacy parameters."""
    pass


class TensorValidationError(DPFlashAttentionError):
    """Exception for tensor validation failures."""
    pass


class SystemRequirementError(DPFlashAttentionError):
    """Exception for system requirement failures."""
    pass


class ConfigurationError(DPFlashAttentionError):
    """Exception for configuration errors."""
    pass


def validate_tensor_shapes(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    expected_dims: int = 4
) -> None:
    """
    Validate tensor shapes for attention computation.
    
    Args:
        q: Query tensor
        k: Key tensor  
        v: Value tensor
        expected_dims: Expected number of dimensions
        
    Raises:
        TensorValidationError: If shapes are invalid
    """
    if q.dim() != expected_dims:
        raise TensorValidationError(
            f"Query tensor must have {expected_dims} dimensions, got {q.dim()}. "
            f"Expected shape: [batch, seq_len, num_heads, head_dim]"
        )
    
    if k.dim() != expected_dims:
        raise TensorValidationError(
            f"Key tensor must have {expected_dims} dimensions, got {k.dim()}. "
            f"Expected shape: [batch, seq_len, num_heads, head_dim]"
        )
    
    if v.dim() != expected_dims:
        raise TensorValidationError(
            f"Value tensor must have {expected_dims} dimensions, got {v.dim()}. "
            f"Expected shape: [batch, seq_len, num_heads, head_dim]"
        )
    
    if q.shape != k.shape:
        raise TensorValidationError(
            f"Query and key tensors must have same shape. "
            f"Query: {q.shape}, Key: {k.shape}"
        )
    
    if q.shape != v.shape:
        raise TensorValidationError(
            f"Query and value tensors must have same shape. "
            f"Query: {q.shape}, Value: {v.shape}"
        )
    
    # Check for reasonable dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    if batch_size <= 0:
        raise TensorValidationError(f"Batch size must be positive, got {batch_size}")
    
    if seq_len <= 0:
        raise TensorValidationError(f"Sequence length must be positive, got {seq_len}")
    
    if num_heads <= 0:
        raise TensorValidationError(f"Number of heads must be positive, got {num_heads}")
    
    if head_dim <= 0:
        raise TensorValidationError(f"Head dimension must be positive, got {head_dim}")
    
    # Warn about potentially problematic dimensions
    if seq_len > 16384:
        warnings.warn(
            f"Long sequence length {seq_len} may cause memory issues. "
            f"Consider using sequence length <= 16384."
        )
    
    if num_heads > 64:
        warnings.warn(
            f"Large number of heads {num_heads} may impact performance. "
            f"Consider using fewer heads or ensure sufficient GPU memory."
        )
    
    if head_dim > 256:
        warnings.warn(
            f"Large head dimension {head_dim} may impact performance."
        )


def validate_tensor_dtypes(
    q: Tensor,
    k: Tensor, 
    v: Tensor,
    supported_dtypes: Optional[List[torch.dtype]] = None
) -> None:
    """
    Validate tensor data types.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        supported_dtypes: List of supported data types
        
    Raises:
        TensorValidationError: If dtypes are invalid
    """
    if supported_dtypes is None:
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TensorValidationError(
            f"All tensors must have same dtype. "
            f"Query: {q.dtype}, Key: {k.dtype}, Value: {v.dtype}"
        )
    
    if q.dtype not in supported_dtypes:
        raise TensorValidationError(
            f"Unsupported dtype {q.dtype}. "
            f"Supported dtypes: {supported_dtypes}"
        )
    
    # Performance warnings
    if q.dtype == torch.float64:
        warnings.warn(
            "Float64 not optimized for attention kernels. "
            "Consider using float16 or bfloat16 for better performance."
        )


def validate_tensor_devices(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    require_cuda: bool = True
) -> None:
    """
    Validate tensor devices and placement.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        require_cuda: Whether to require CUDA tensors
        
    Raises:
        TensorValidationError: If device placement is invalid
    """
    if q.device != k.device or q.device != v.device:
        raise TensorValidationError(
            f"All tensors must be on same device. "
            f"Query: {q.device}, Key: {k.device}, Value: {v.device}"
        )
    
    if require_cuda and not q.device.type == 'cuda':
        raise TensorValidationError(
            f"Tensors must be on CUDA device for optimal performance. "
            f"Current device: {q.device}. "
            f"Use tensor.cuda() to move to GPU."
        )
    
    if q.device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemRequirementError(
            "CUDA tensors provided but CUDA is not available. "
            "Check CUDA installation."
        )


def validate_tensor_memory(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    check_gpu_memory: bool = True
) -> Dict[str, Any]:
    """
    Validate tensor memory usage and availability.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        check_gpu_memory: Whether to check GPU memory
        
    Returns:
        Dictionary with memory information
        
    Raises:
        TensorValidationError: If memory requirements cannot be met
    """
    # Calculate tensor sizes
    q_size_mb = q.numel() * q.element_size() / (1024 ** 2)
    k_size_mb = k.numel() * k.element_size() / (1024 ** 2)
    v_size_mb = v.numel() * v.element_size() / (1024 ** 2)
    total_input_mb = q_size_mb + k_size_mb + v_size_mb
    
    # Estimate additional memory for attention computation
    batch_size, seq_len, num_heads, head_dim = q.shape
    attention_scores_size = batch_size * num_heads * seq_len * seq_len * q.element_size()
    attention_scores_mb = attention_scores_size / (1024 ** 2)
    
    # Estimate total memory requirement
    estimated_total_mb = total_input_mb + attention_scores_mb * 2  # Factor for working memory
    
    memory_info = {
        'input_tensors_mb': total_input_mb,
        'attention_scores_mb': attention_scores_mb,
        'estimated_total_mb': estimated_total_mb,
    }
    
    if check_gpu_memory and q.device.type == 'cuda':
        device_idx = q.device.index or 0
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        available_memory = total_memory - torch.cuda.memory_allocated(device_idx)
        
        total_memory_mb = total_memory / (1024 ** 2)
        available_memory_mb = available_memory / (1024 ** 2)
        
        memory_info.update({
            'gpu_total_mb': total_memory_mb,
            'gpu_available_mb': available_memory_mb,
            'gpu_utilization': torch.cuda.memory_allocated(device_idx) / total_memory,
        })
        
        if estimated_total_mb > available_memory_mb * 0.8:  # Leave 20% buffer
            raise TensorValidationError(
                f"Insufficient GPU memory. "
                f"Required: ~{estimated_total_mb:.1f}MB, "
                f"Available: {available_memory_mb:.1f}MB. "
                f"Consider reducing batch size or sequence length."
            )
        
        if estimated_total_mb > available_memory_mb * 0.5:
            warnings.warn(
                f"High memory usage detected. "
                f"Required: ~{estimated_total_mb:.1f}MB, "
                f"Available: {available_memory_mb:.1f}MB. "
                f"Monitor memory usage carefully."
            )
    
    return memory_info


def validate_privacy_parameters_comprehensive(
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    additional_checks: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive validation of privacy parameters.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter
        max_grad_norm: Gradient clipping norm
        additional_checks: Whether to perform additional validation
        
    Returns:
        Dictionary with validation results and recommendations
        
    Raises:
        PrivacyParameterError: If parameters are invalid
    """
    validation_results = {
        'epsilon_valid': True,
        'delta_valid': True,
        'max_grad_norm_valid': True,
        'privacy_level': 'unknown',
        'recommendations': [],
    }
    
    # Basic epsilon validation
    if not isinstance(epsilon, (int, float)):
        raise PrivacyParameterError(f"Epsilon must be numeric, got {type(epsilon)}")
    
    if not math.isfinite(epsilon):
        raise PrivacyParameterError(f"Epsilon must be finite, got {epsilon}")
    
    if epsilon <= 0:
        raise PrivacyParameterError(f"Epsilon must be positive, got {epsilon}")
    
    # Basic delta validation
    if not isinstance(delta, (int, float)):
        raise PrivacyParameterError(f"Delta must be numeric, got {type(delta)}")
    
    if not math.isfinite(delta):
        raise PrivacyParameterError(f"Delta must be finite, got {delta}")
    
    if delta <= 0 or delta >= 1:
        raise PrivacyParameterError(f"Delta must be in (0, 1), got {delta}")
    
    # Basic max_grad_norm validation
    if not isinstance(max_grad_norm, (int, float)):
        raise PrivacyParameterError(f"max_grad_norm must be numeric, got {type(max_grad_norm)}")
    
    if not math.isfinite(max_grad_norm):
        raise PrivacyParameterError(f"max_grad_norm must be finite, got {max_grad_norm}")
    
    if max_grad_norm <= 0:
        raise PrivacyParameterError(f"max_grad_norm must be positive, got {max_grad_norm}")
    
    if additional_checks:
        # Privacy level assessment
        if epsilon < 0.1:
            validation_results['privacy_level'] = 'very_strong'
        elif epsilon < 1.0:
            validation_results['privacy_level'] = 'strong'
        elif epsilon < 3.0:
            validation_results['privacy_level'] = 'moderate'
        elif epsilon < 10.0:
            validation_results['privacy_level'] = 'weak'
        else:
            validation_results['privacy_level'] = 'very_weak'
            validation_results['recommendations'].append(
                f"Epsilon {epsilon} provides very weak privacy. Consider epsilon < 10."
            )
        
        # Delta recommendations
        if delta > 1e-3:
            validation_results['recommendations'].append(
                f"Delta {delta} is relatively large. Consider delta < 1e-5 for stronger guarantees."
            )
        
        # Grad norm recommendations
        if max_grad_norm > 10.0:
            validation_results['recommendations'].append(
                f"Large gradient clipping norm {max_grad_norm} may impact utility. "
                f"Consider smaller values like 1.0-2.0."
            )
        
        if max_grad_norm < 0.1:
            validation_results['recommendations'].append(
                f"Very small gradient clipping norm {max_grad_norm} may severely impact utility."
            )
        
        # Combined parameter analysis
        noise_scale = math.sqrt(2 * math.log(1.25 / delta)) * max_grad_norm / epsilon
        validation_results['noise_scale'] = noise_scale
        
        if noise_scale > 5.0:
            validation_results['recommendations'].append(
                f"High noise scale {noise_scale:.2f} may significantly impact model utility. "
                f"Consider increasing epsilon or decreasing max_grad_norm."
            )
    
    return validation_results


def validate_system_requirements_comprehensive() -> Dict[str, Any]:
    """
    Comprehensive system requirements validation.
    
    Returns:
        Dictionary with detailed system validation results
        
    Raises:
        SystemRequirementError: If critical requirements are not met
    """
    requirements = {
        'python_version': None,
        'pytorch_version': None,
        'cuda_available': False,
        'cuda_version': None,
        'gpu_memory_gb': 0,
        'compute_capability': None,
        'requirements_met': True,
        'warnings': [],
        'errors': [],
    }
    
    try:
        # Python version check
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        requirements['python_version'] = python_version
        
        if sys.version_info < (3, 10):
            requirements['errors'].append(
                f"Python {python_version} not supported. Requires Python >= 3.10."
            )
            requirements['requirements_met'] = False
        
        # PyTorch version check
        torch_version = torch.__version__
        requirements['pytorch_version'] = torch_version
        
        major_minor = '.'.join(torch_version.split('.')[:2])
        if float(major_minor) < 2.3:
            requirements['errors'].append(
                f"PyTorch {torch_version} not supported. Requires PyTorch >= 2.3.0."
            )
            requirements['requirements_met'] = False
        
        # CUDA availability
        cuda_available = torch.cuda.is_available()
        requirements['cuda_available'] = cuda_available
        
        if cuda_available:
            requirements['cuda_version'] = torch.version.cuda
            
            # GPU memory check
            try:
                device_props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = device_props.total_memory / (1024**3)
                requirements['gpu_memory_gb'] = gpu_memory_gb
                requirements['compute_capability'] = f"{device_props.major}.{device_props.minor}"
                
                if gpu_memory_gb < 8:
                    requirements['warnings'].append(
                        f"GPU memory {gpu_memory_gb:.1f}GB is low. "
                        f"Recommended: >= 16GB for optimal performance."
                    )
                
                if device_props.major < 7:
                    requirements['warnings'].append(
                        f"GPU compute capability {device_props.major}.{device_props.minor} "
                        f"may not support all optimizations. Recommended: >= 7.0 (Volta)."
                    )
                
            except Exception as e:
                requirements['warnings'].append(f"Could not query GPU properties: {e}")
        else:
            requirements['warnings'].append(
                "CUDA not available. Performance will be significantly reduced."
            )
        
        # Package dependencies
        required_packages = {
            'triton': '2.3.0',
            'einops': '0.7.0', 
            'numpy': '1.24.0',
            'ninja': '1.11.0',
        }
        
        for package, min_version in required_packages.items():
            try:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    version = module.__version__
                    requirements[f'{package}_version'] = version
                else:
                    requirements[f'{package}_available'] = True
            except ImportError:
                requirements['errors'].append(f"Required package '{package}' not installed.")
                requirements['requirements_met'] = False
        
        # Optional privacy packages
        optional_packages = ['opacus', 'dp_accounting', 'prv_accountant']
        for package in optional_packages:
            try:
                __import__(package)
                requirements[f'{package}_available'] = True
            except ImportError:
                requirements['warnings'].append(
                    f"Optional package '{package}' not available. "
                    f"Some privacy features may be limited."
                )
        
    except Exception as e:
        requirements['errors'].append(f"Error during system validation: {e}")
        requirements['requirements_met'] = False
    
    if not requirements['requirements_met']:
        error_msg = "System requirements not met:\n" + "\n".join(requirements['errors'])
        raise SystemRequirementError(error_msg)
    
    return requirements


def validate_attention_configuration(
    embed_dim: int,
    num_heads: int,
    sequence_length: int,
    batch_size: int,
    dropout: float = 0.0,
) -> Dict[str, Any]:
    """
    Validate attention layer configuration.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        sequence_length: Input sequence length
        batch_size: Batch size
        dropout: Dropout probability
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    config_info = {
        'head_dim': embed_dim // num_heads,
        'total_parameters': 0,
        'memory_estimate_mb': 0,
        'warnings': [],
        'valid': True,
    }
    
    # Basic parameter validation
    if embed_dim <= 0:
        raise ConfigurationError(f"embed_dim must be positive, got {embed_dim}")
    
    if num_heads <= 0:
        raise ConfigurationError(f"num_heads must be positive, got {num_heads}")
    
    if embed_dim % num_heads != 0:
        raise ConfigurationError(
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        )
    
    if sequence_length <= 0:
        raise ConfigurationError(f"sequence_length must be positive, got {sequence_length}")
    
    if batch_size <= 0:
        raise ConfigurationError(f"batch_size must be positive, got {batch_size}")
    
    if not 0.0 <= dropout <= 1.0:
        raise ConfigurationError(f"dropout must be in [0, 1], got {dropout}")
    
    head_dim = embed_dim // num_heads
    config_info['head_dim'] = head_dim
    
    # Parameter count (4 linear layers: Q, K, V, output)
    config_info['total_parameters'] = 4 * embed_dim * embed_dim
    
    # Memory estimation
    bytes_per_param = 4  # float32
    model_memory_mb = config_info['total_parameters'] * bytes_per_param / (1024**2)
    
    # Activation memory (rough estimate)
    activation_memory_mb = (
        batch_size * sequence_length * embed_dim * 4 * bytes_per_param / (1024**2)
    )
    
    config_info['memory_estimate_mb'] = model_memory_mb + activation_memory_mb
    
    # Configuration warnings
    if head_dim < 32:
        config_info['warnings'].append(
            f"Small head dimension {head_dim} may limit model expressiveness. "
            f"Consider head_dim >= 32."
        )
    
    if head_dim > 256:
        config_info['warnings'].append(
            f"Large head dimension {head_dim} may impact performance."
        )
    
    if num_heads > 32:
        config_info['warnings'].append(
            f"Large number of heads {num_heads} may impact performance."
        )
    
    if sequence_length > 8192:
        config_info['warnings'].append(
            f"Long sequence {sequence_length} will require significant memory. "
            f"Quadratic scaling: O(seq_len²)"
        )
    
    if config_info['memory_estimate_mb'] > 1000:
        config_info['warnings'].append(
            f"High memory usage estimated: {config_info['memory_estimate_mb']:.1f}MB"
        )
    
    return config_info