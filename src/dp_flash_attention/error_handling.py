"""
Enhanced error handling for DP-Flash-Attention.

Provides comprehensive error handling, recovery strategies, and user-friendly
error messages for differential privacy operations.
"""

import traceback
import warnings
from typing import Optional, Dict, Any, Callable, Union, Type
from functools import wraps
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DPFlashAttentionError(Exception):
    """Base exception class for DP-Flash-Attention errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        suggestions: Optional[list] = None,
        context: Optional[dict] = None
    ):
        """
        Initialize DP-Flash-Attention error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            suggestions: List of suggested solutions
            context: Additional context about the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.suggestions = suggestions or []
        self.context = context or {}
    
    def __str__(self):
        """Format error message with suggestions."""
        error_str = f"[{self.error_code}] {self.message}"
        
        if self.suggestions:
            error_str += "\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                error_str += f"\n  {i}. {suggestion}"
        
        if self.context:
            error_str += f"\n\nContext: {self.context}"
        
        return error_str


class PrivacyParameterError(DPFlashAttentionError):
    """Error in differential privacy parameters."""
    
    def __init__(self, message: str, epsilon: float = None, delta: float = None):
        suggestions = [
            "Ensure epsilon > 0 (typically 0.1 to 10 for meaningful privacy)",
            "Ensure 0 < delta < 1 (typically 1e-5 to 1e-3)",
            "Consider the privacy-utility tradeoff for your use case",
            "Check privacy accounting literature for parameter guidance"
        ]
        
        context = {}
        if epsilon is not None:
            context["epsilon"] = epsilon
        if delta is not None:
            context["delta"] = delta
        
        super().__init__(
            message=message,
            error_code="PRIVACY_PARAM_ERROR",
            suggestions=suggestions,
            context=context
        )


class CUDACompatibilityError(DPFlashAttentionError):
    """Error related to CUDA availability or compatibility."""
    
    def __init__(self, message: str, cuda_available: bool = None):
        suggestions = [
            "Install PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118",
            "Check CUDA installation: nvcc --version",
            "Verify GPU is available and accessible",
            "Consider using CPU fallback with cpu_fallback=True"
        ]
        
        if not cuda_available:
            suggestions.extend([
                "For CPU-only usage, expect slower performance",
                "Consider using smaller batch sizes for CPU execution"
            ])
        
        context = {"cuda_available": cuda_available}
        
        super().__init__(
            message=message,
            error_code="CUDA_ERROR",
            suggestions=suggestions,
            context=context
        )


class TensorShapeError(DPFlashAttentionError):
    """Error in tensor shapes or dimensions."""
    
    def __init__(self, message: str, expected_shape: tuple = None, actual_shape: tuple = None):
        suggestions = [
            "Check input tensor dimensions match expected format",
            "Ensure batch_first parameter is set correctly",
            "Verify sequence length and embedding dimensions",
            "Consider reshaping inputs to match expected format"
        ]
        
        context = {}
        if expected_shape is not None:
            context["expected_shape"] = expected_shape
        if actual_shape is not None:
            context["actual_shape"] = actual_shape
        
        super().__init__(
            message=message,
            error_code="TENSOR_SHAPE_ERROR",
            suggestions=suggestions,
            context=context
        )


class PrivacyBudgetExceededError(DPFlashAttentionError):
    """Error when privacy budget is exceeded."""
    
    def __init__(
        self,
        message: str,
        spent_epsilon: float = None,
        target_epsilon: float = None,
        remaining_epsilon: float = None
    ):
        suggestions = [
            "Reduce number of training steps or increase privacy budget",
            "Use privacy amplification techniques (subsampling, etc.)",
            "Consider using advanced composition (RDP, GDP) for tighter bounds",
            "Reset privacy accounting if starting new experiment"
        ]
        
        context = {}
        if spent_epsilon is not None:
            context["spent_epsilon"] = spent_epsilon
        if target_epsilon is not None:
            context["target_epsilon"] = target_epsilon
        if remaining_epsilon is not None:
            context["remaining_epsilon"] = remaining_epsilon
        
        super().__init__(
            message=message,
            error_code="PRIVACY_BUDGET_EXCEEDED",
            suggestions=suggestions,
            context=context
        )


class KernelCompilationError(DPFlashAttentionError):
    """Error in CUDA kernel compilation."""
    
    def __init__(self, message: str, kernel_name: str = None):
        suggestions = [
            "Check CUDA toolkit installation and version compatibility",
            "Ensure nvcc is available in PATH",
            "Verify PyTorch CUDA version matches system CUDA",
            "Try using fallback implementation with use_fallback=True",
            "Check compiler compatibility (GCC version, etc.)"
        ]
        
        context = {"kernel_name": kernel_name} if kernel_name else {}
        
        super().__init__(
            message=message,
            error_code="KERNEL_COMPILATION_ERROR",
            suggestions=suggestions,
            context=context
        )


class SecurityValidationError(DPFlashAttentionError):
    """Error in security validation."""
    
    def __init__(self, message: str, validation_details: dict = None):
        suggestions = [
            "Review input data for potential security issues",
            "Check for NaN or infinite values in tensors",
            "Verify model parameters are within expected ranges",
            "Consider using input sanitization functions"
        ]
        
        super().__init__(
            message=message,
            error_code="SECURITY_VALIDATION_ERROR",
            suggestions=suggestions,
            context=validation_details or {}
        )


def handle_errors(
    fallback_value: Any = None,
    reraise: bool = True,
    log_errors: bool = True
) -> Callable:
    """
    Decorator for enhanced error handling.
    
    Args:
        fallback_value: Value to return on error (if not reraising)
        reraise: Whether to reraise exceptions after handling
        log_errors: Whether to log errors
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except DPFlashAttentionError:
                # Our custom errors - just reraise
                if log_errors:
                    logging.getLogger('dp_flash_attention.errors').error(
                        f"DP-Flash-Attention error in {func.__name__}: {traceback.format_exc()}"
                    )
                if reraise:
                    raise
                return fallback_value
            
            except ValueError as e:
                # Convert common ValueError cases to our custom exceptions
                error_msg = str(e).lower()
                
                if "epsilon" in error_msg or "delta" in error_msg:
                    raise PrivacyParameterError(f"Invalid privacy parameters: {e}")
                elif "shape" in error_msg or "dimension" in error_msg:
                    raise TensorShapeError(f"Tensor shape error: {e}")
                else:
                    # Generic ValueError
                    raise DPFlashAttentionError(
                        f"Value error in {func.__name__}: {e}",
                        error_code="VALUE_ERROR",
                        suggestions=[
                            "Check input parameters and their types",
                            "Verify tensor shapes and dimensions",
                            "Review function documentation for parameter requirements"
                        ]
                    )
            
            except RuntimeError as e:
                error_msg = str(e).lower()
                
                if "cuda" in error_msg:
                    cuda_available = HAS_TORCH and torch.cuda.is_available()
                    raise CUDACompatibilityError(
                        f"CUDA runtime error: {e}",
                        cuda_available=cuda_available
                    )
                else:
                    raise DPFlashAttentionError(
                        f"Runtime error in {func.__name__}: {e}",
                        error_code="RUNTIME_ERROR",
                        suggestions=[
                            "Check system resources (memory, disk space)",
                            "Verify CUDA/GPU availability if using GPU",
                            "Try reducing batch size or model complexity"
                        ]
                    )
            
            except ImportError as e:
                raise DPFlashAttentionError(
                    f"Missing dependency: {e}",
                    error_code="IMPORT_ERROR",
                    suggestions=[
                        "Install missing dependencies with pip install",
                        "Check requirements.txt for complete dependency list",
                        "Verify Python environment is activated",
                        "Use pip list to see installed packages"
                    ]
                )
            
            except Exception as e:
                # Unexpected error
                if log_errors:
                    logging.getLogger('dp_flash_attention.errors').error(
                        f"Unexpected error in {func.__name__}: {traceback.format_exc()}"
                    )
                
                raise DPFlashAttentionError(
                    f"Unexpected error in {func.__name__}: {e}",
                    error_code="UNEXPECTED_ERROR",
                    suggestions=[
                        "Check the full traceback for more details",
                        "Verify all inputs are correct and properly formatted",
                        "Try a minimal example to isolate the issue",
                        "Consider reporting this as a bug if the error persists"
                    ],
                    context={"function": func.__name__, "exception_type": type(e).__name__}
                )
        
        return wrapper
    return decorator


def validate_privacy_parameters(epsilon: float, delta: float, strict: bool = True) -> None:
    """
    Validate differential privacy parameters with enhanced error messages.
    
    Args:
        epsilon: Privacy budget parameter
        delta: Privacy parameter
        strict: Whether to use strict validation
        
    Raises:
        PrivacyParameterError: If parameters are invalid
    """
    errors = []
    
    # Type validation
    if not isinstance(epsilon, (int, float)):
        errors.append(f"epsilon must be numeric, got {type(epsilon).__name__}")
    
    if not isinstance(delta, (int, float)):
        errors.append(f"delta must be numeric, got {type(delta).__name__}")
    
    if errors:
        raise PrivacyParameterError(
            "Invalid parameter types: " + "; ".join(errors),
            epsilon=epsilon,
            delta=delta
        )
    
    # Range validation
    if epsilon <= 0:
        raise PrivacyParameterError(
            f"epsilon must be positive, got {epsilon}. "
            f"Epsilon represents privacy budget - smaller values = stronger privacy.",
            epsilon=epsilon,
            delta=delta
        )
    
    if delta <= 0:
        raise PrivacyParameterError(
            f"delta must be positive, got {delta}. "
            f"Delta represents failure probability - should be very small (e.g., 1e-5).",
            epsilon=epsilon,
            delta=delta
        )
    
    if delta >= 1:
        raise PrivacyParameterError(
            f"delta must be less than 1, got {delta}. "
            f"Delta represents failure probability - values ≥ 1 provide no privacy.",
            epsilon=epsilon,
            delta=delta
        )
    
    # Practical range warnings
    if strict:
        if epsilon > 50:
            raise PrivacyParameterError(
                f"epsilon = {epsilon} is extremely large and provides minimal privacy. "
                f"Consider epsilon < 10 for meaningful privacy protection.",
                epsilon=epsilon,
                delta=delta
            )
        
        if delta > 0.01:  # 1%
            raise PrivacyParameterError(
                f"delta = {delta} is very large for differential privacy. "
                f"Consider delta < 1e-3 (0.001) for strong privacy guarantees.",
                epsilon=epsilon,
                delta=delta
            )
    else:
        # Just warn for questionable values
        if epsilon > 10:
            warnings.warn(
                f"Large epsilon ({epsilon}) provides weak privacy. "
                f"Consider smaller values for stronger privacy.",
                UserWarning
            )
        
        if delta > 1e-3:
            warnings.warn(
                f"Large delta ({delta}) may compromise privacy guarantees. "
                f"Consider smaller values (e.g., 1e-5).",
                UserWarning
            )


def validate_tensor_inputs(tensors: list, names: list = None, allow_none: bool = False) -> None:
    """
    Validate tensor inputs with detailed error messages.
    
    Args:
        tensors: List of tensors to validate
        names: Optional names for tensors (for error messages)
        allow_none: Whether to allow None values
        
    Raises:
        TensorShapeError: If tensors have invalid shapes
        SecurityValidationError: If tensors contain invalid values
    """
    if not HAS_TORCH:
        warnings.warn("PyTorch not available, skipping tensor validation")
        return
    
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]
    
    for i, (tensor, name) in enumerate(zip(tensors, names)):
        if tensor is None:
            if not allow_none:
                raise TensorShapeError(
                    f"{name} cannot be None",
                    context={"tensor_index": i, "tensor_name": name}
                )
            continue
        
        # Type check
        if not isinstance(tensor, torch.Tensor):
            raise TensorShapeError(
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}",
                context={"tensor_index": i, "tensor_name": name, "actual_type": type(tensor).__name__}
            )
        
        # Check for invalid values
        if torch.isnan(tensor).any():
            raise SecurityValidationError(
                f"{name} contains NaN values",
                validation_details={
                    "tensor_name": name,
                    "nan_count": torch.isnan(tensor).sum().item(),
                    "total_elements": tensor.numel()
                }
            )
        
        if torch.isinf(tensor).any():
            raise SecurityValidationError(
                f"{name} contains infinite values",
                validation_details={
                    "tensor_name": name,
                    "inf_count": torch.isinf(tensor).sum().item(),
                    "total_elements": tensor.numel()
                }
            )
        
        # Check for suspiciously large values
        if tensor.numel() > 0:
            max_val = torch.max(torch.abs(tensor)).item()
            if max_val > 1e6:
                warnings.warn(
                    f"{name} contains very large values (max={max_val:.2e}). "
                    f"This might indicate a problem with the input data.",
                    UserWarning
                )


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with enhanced error handling.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Result of division or default value
    """
    try:
        if abs(denominator) < 1e-15:
            warnings.warn(f"Division by very small number ({denominator}), returning default")
            return default
        return numerator / denominator
    except (ZeroDivisionError, OverflowError):
        warnings.warn(f"Division failed ({numerator}/{denominator}), returning default")
        return default


def check_memory_usage(
    required_mb: float,
    device: str = 'cuda',
    safety_factor: float = 1.2
) -> None:
    """
    Check if sufficient memory is available.
    
    Args:
        required_mb: Required memory in MB
        device: Device to check ('cuda' or 'cpu')
        safety_factor: Safety factor for memory estimation
        
    Raises:
        DPFlashAttentionError: If insufficient memory
    """
    if not HAS_TORCH:
        warnings.warn("Cannot check memory without PyTorch")
        return
    
    required_bytes = required_mb * 1024 * 1024 * safety_factor
    
    try:
        if device == 'cuda' and torch.cuda.is_available():
            available_bytes = torch.cuda.get_device_properties(0).total_memory
            allocated_bytes = torch.cuda.memory_allocated()
            free_bytes = available_bytes - allocated_bytes
            
            if required_bytes > free_bytes:
                raise DPFlashAttentionError(
                    f"Insufficient GPU memory. Required: {required_mb:.1f}MB "
                    f"(+{safety_factor:.1f}x safety factor), "
                    f"Available: {free_bytes/(1024*1024):.1f}MB",
                    error_code="INSUFFICIENT_MEMORY",
                    suggestions=[
                        "Reduce batch size",
                        "Reduce sequence length",
                        "Use gradient checkpointing to trade compute for memory",
                        "Consider using CPU fallback for smaller workloads"
                    ],
                    context={
                        "required_mb": required_mb,
                        "available_mb": free_bytes / (1024 * 1024),
                        "device": device
                    }
                )
        
        # For CPU, we can't easily check available memory, so just warn
        elif device == 'cpu' and required_mb > 8000:  # 8GB threshold
            warnings.warn(
                f"Large memory requirement ({required_mb:.1f}MB) for CPU operation. "
                f"Consider reducing batch size if you encounter memory issues."
            )
    
    except Exception as e:
        warnings.warn(f"Could not check memory usage: {e}")


class ErrorRecovery:
    """Utility class for error recovery strategies."""
    
    @staticmethod
    def retry_with_smaller_batch(
        func: Callable,
        initial_batch_size: int,
        min_batch_size: int = 1,
        *args,
        **kwargs
    ):
        """
        Retry function with progressively smaller batch sizes.
        
        Args:
            func: Function to retry
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size to try
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        batch_size = initial_batch_size
        
        while batch_size >= min_batch_size:
            try:
                # Update batch size in kwargs if present
                if 'batch_size' in kwargs:
                    kwargs['batch_size'] = batch_size
                
                return func(*args, **kwargs)
            
            except (RuntimeError, CUDACompatibilityError) as e:
                if "memory" in str(e).lower() and batch_size > min_batch_size:
                    batch_size = max(batch_size // 2, min_batch_size)
                    warnings.warn(
                        f"Memory error encountered, retrying with batch_size={batch_size}"
                    )
                    continue
                else:
                    raise
        
        raise DPFlashAttentionError(
            f"Could not complete operation even with minimum batch size {min_batch_size}",
            error_code="BATCH_SIZE_TOO_LARGE",
            suggestions=[
                "Try an even smaller batch size manually",
                "Use gradient checkpointing",
                "Switch to CPU execution",
                "Reduce model size or sequence length"
            ]
        )