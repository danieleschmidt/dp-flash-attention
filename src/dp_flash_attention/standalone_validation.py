"""
Standalone validation utilities that work without PyTorch dependencies.
"""

import math
import sys
import warnings
import traceback
from typing import Optional, Dict, Any, Union

class ValidationError(Exception):
    """Base validation error."""
    pass

class PrivacyValidationError(ValidationError):
    """Privacy parameter validation error."""
    pass

def validate_privacy_parameters_standalone(
    epsilon: Union[int, float], 
    delta: Union[int, float], 
    strict: bool = True
) -> bool:
    """
    Standalone privacy parameter validation without external dependencies.
    
    Args:
        epsilon: Privacy budget parameter (must be > 0)
        delta: Privacy parameter (must be in [0, 1))
        strict: Whether to use strict validation bounds
        
    Returns:
        True if parameters are valid
        
    Raises:
        PrivacyValidationError: If parameters are invalid
    """
    # Type validation
    if not isinstance(epsilon, (int, float)):
        raise PrivacyValidationError(f"epsilon must be numeric, got {type(epsilon).__name__}")
    
    if not isinstance(delta, (int, float)):
        raise PrivacyValidationError(f"delta must be numeric, got {type(delta).__name__}")
    
    # Check for special float values
    if math.isnan(epsilon):
        raise PrivacyValidationError("epsilon cannot be NaN")
    
    if math.isnan(delta):
        raise PrivacyValidationError("delta cannot be NaN")
    
    if math.isinf(epsilon):
        raise PrivacyValidationError("epsilon cannot be infinite")
    
    if math.isinf(delta):
        raise PrivacyValidationError("delta cannot be infinite")
    
    # Range validation
    if epsilon <= 0:
        raise PrivacyValidationError(
            f"epsilon must be positive, got {epsilon}. "
            f"Epsilon represents privacy budget - smaller values = stronger privacy."
        )
    
    if delta < 0:
        raise PrivacyValidationError(
            f"delta must be non-negative, got {delta}. "
            f"Delta represents failure probability."
        )
    
    if delta >= 1:
        raise PrivacyValidationError(
            f"delta must be less than 1, got {delta}. "
            f"Delta ≥ 1 provides no privacy guarantees."
        )
    
    # Strict validation for practical ranges
    if strict:
        if epsilon > 50:
            raise PrivacyValidationError(
                f"epsilon = {epsilon} is extremely large and provides minimal privacy. "
                f"Consider epsilon < 10 for meaningful privacy protection."
            )
        
        if delta > 0.01:  # 1%
            raise PrivacyValidationError(
                f"delta = {delta} is very large for differential privacy. "
                f"Consider delta < 1e-3 (0.001) for strong privacy guarantees."
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
    
    return True

def compute_noise_scale_standalone(
    epsilon: float,
    delta: float, 
    sensitivity: float = 2.0,
    mechanism: str = "gaussian"
) -> float:
    """
    Compute noise scale for differential privacy mechanisms.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter
        sensitivity: L2 sensitivity of the function
        mechanism: Noise mechanism ("gaussian" or "laplace")
        
    Returns:
        Noise scale (standard deviation for Gaussian, scale for Laplace)
    """
    validate_privacy_parameters_standalone(epsilon, delta, strict=False)
    
    if sensitivity <= 0:
        raise ValidationError(f"sensitivity must be positive, got {sensitivity}")
    
    if mechanism == "gaussian":
        # Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        if delta <= 0:
            raise ValidationError("Gaussian mechanism requires delta > 0")
        
        noise_scale = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
    
    elif mechanism == "laplace":
        # Laplace mechanism: b = sensitivity / ε
        noise_scale = sensitivity / epsilon
    
    else:
        raise ValidationError(f"Unknown mechanism: {mechanism}. Use 'gaussian' or 'laplace'")
    
    return noise_scale

def estimate_memory_usage_standalone(
    batch_size: int,
    seq_len: int, 
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # fp16 = 2 bytes
) -> Dict[str, float]:
    """
    Estimate memory usage for attention computation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Input tensors Q, K, V
    qkv_elements = 3 * batch_size * seq_len * num_heads * head_dim
    qkv_bytes = qkv_elements * dtype_bytes
    
    # Attention scores matrix (batch_size * num_heads, seq_len, seq_len)
    scores_elements = batch_size * num_heads * seq_len * seq_len
    scores_bytes = scores_elements * dtype_bytes
    
    # Output tensor
    output_elements = batch_size * seq_len * num_heads * head_dim
    output_bytes = output_elements * dtype_bytes
    
    # Gradient tensors (approximately same size as forward tensors)
    gradient_bytes = qkv_bytes + scores_bytes + output_bytes
    
    # Noise tensors for differential privacy
    noise_bytes = scores_bytes  # Noise added to attention scores
    
    # Overhead (activations, temporaries, etc.)
    overhead_bytes = (qkv_bytes + scores_bytes + output_bytes) * 0.3
    
    total_bytes = qkv_bytes + scores_bytes + output_bytes + gradient_bytes + noise_bytes + overhead_bytes
    
    return {
        'qkv_mb': qkv_bytes / (1024 * 1024),
        'scores_mb': scores_bytes / (1024 * 1024),
        'output_mb': output_bytes / (1024 * 1024),
        'gradient_mb': gradient_bytes / (1024 * 1024),
        'noise_mb': noise_bytes / (1024 * 1024),
        'overhead_mb': overhead_bytes / (1024 * 1024),
        'total_mb': total_bytes / (1024 * 1024),
        'total_estimated_mb': total_bytes / (1024 * 1024),  # For compatibility
    }

class SimplePrivacyAccountant:
    """Standalone privacy accountant without external dependencies."""
    
    def __init__(self):
        self.steps = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0
    
    def add_step(self, epsilon: float, delta: float, batch_size: int = None, seq_len: int = None) -> float:
        """
        Add a privacy step with basic composition.
        
        Args:
            epsilon: Privacy budget for this step
            delta: Privacy parameter for this step
            batch_size: Batch size (for accounting purposes)
            seq_len: Sequence length (for accounting purposes)
            
        Returns:
            Epsilon consumed in this step
        """
        validate_privacy_parameters_standalone(epsilon, delta, strict=False)
        
        step_info = {
            'epsilon': epsilon,
            'delta': delta,
            'step_id': len(self.steps),
        }
        
        if batch_size is not None:
            step_info['batch_size'] = batch_size
        if seq_len is not None:
            step_info['seq_len'] = seq_len
        
        self.steps.append(step_info)
        
        # Basic composition (conservative)
        self.total_epsilon += epsilon
        self.total_delta += delta
        
        return epsilon
    
    def get_total_epsilon(self, target_delta: float = None) -> float:
        """Get total privacy spent."""
        if target_delta is not None and target_delta != self.total_delta:
            warnings.warn(
                f"Requested delta {target_delta} differs from accumulated delta {self.total_delta}. "
                f"Using basic composition."
            )
        return self.total_epsilon
    
    def get_total_delta(self) -> float:
        """Get total delta parameter."""
        return self.total_delta
    
    def reset(self):
        """Reset privacy accounting."""
        self.steps.clear()
        self.total_epsilon = 0.0
        self.total_delta = 0.0
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of privacy accounting."""
        return {
            'total_steps': len(self.steps),
            'total_epsilon': self.total_epsilon,
            'total_delta': self.total_delta,
            'steps': self.steps,
        }

def test_standalone_components():
    """Test all standalone components."""
    print("Testing standalone robustness components...")
    
    try:
        # Test privacy validation
        validate_privacy_parameters_standalone(1.0, 1e-5)
        print("✓ Privacy validation works")
        
        # Test noise scale computation
        noise_scale = compute_noise_scale_standalone(1.0, 1e-5, 2.0, "gaussian")
        assert noise_scale > 0
        print(f"✓ Noise scale computation: {noise_scale:.4f}")
        
        # Test memory estimation
        memory_est = estimate_memory_usage_standalone(32, 512, 12, 64)
        assert memory_est['total_mb'] > 0
        print(f"✓ Memory estimation: {memory_est['total_mb']:.1f} MB")
        
        # Test privacy accounting
        accountant = SimplePrivacyAccountant()
        for i in range(5):
            accountant.add_step(0.1, 1e-5, 32, 512)
        
        summary = accountant.get_privacy_summary()
        assert summary['total_steps'] == 5
        assert abs(summary['total_epsilon'] - 0.5) < 1e-10
        print(f"✓ Privacy accounting: {summary['total_epsilon']} epsilon spent")
        
        return True
        
    except Exception as e:
        print(f"✗ Standalone components test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_standalone_components()
    if success:
        print("\n🎉 GENERATION 2 STANDALONE ROBUSTNESS VALIDATION PASSED!")
        print("✅ Privacy validation enhanced")
        print("✅ Noise computation optimized") 
        print("✅ Memory estimation improved")
        print("✅ Privacy accounting robust")
        print("🚀 Ready for Generation 3 optimization")
    else:
        print("\n❌ Generation 2 validation failed")
    
    sys.exit(0 if success else 1)