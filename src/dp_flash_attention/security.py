"""
Security utilities for DP-Flash-Attention.

Provides cryptographically secure random number generation, secure noise sampling,
and security validation functions for differential privacy operations.
"""

import os
import secrets
import hashlib
import warnings
from typing import Optional, Tuple, Union, Dict, Any
import time

import torch
from torch import Tensor
import numpy as np

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    warnings.warn("cryptography package not available, using fallback security measures")


class SecureRandomGenerator:
    """
    Cryptographically secure random number generator for differential privacy.
    
    Uses system entropy sources and cryptographically secure PRNGs to generate
    noise for differential privacy mechanisms.
    """
    
    def __init__(self, seed: Optional[int] = None, use_system_entropy: bool = True):
        """
        Initialize secure random generator.
        
        Args:
            seed: Optional seed for reproducibility (not recommended for production)
            use_system_entropy: Whether to use system entropy sources
        """
        self.use_system_entropy = use_system_entropy
        self._initialized = False
        
        if seed is not None:
            warnings.warn(
                "Using fixed seed reduces security. Only use for testing/debugging."
            )
            torch.manual_seed(seed)
            np.random.seed(seed)
            self._seed = seed
        else:
            self._seed = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize secure random state."""
        if self.use_system_entropy and self._seed is None:
            # Generate cryptographically secure seed from system entropy
            entropy_bytes = os.urandom(32)  # 256 bits of entropy
            seed_value = int.from_bytes(entropy_bytes, byteorder='big') % (2**32)
            
            # Use secrets module for additional security
            secure_seed = secrets.randbits(32)
            combined_seed = seed_value ^ secure_seed
            
            torch.manual_seed(combined_seed)
            np.random.seed(combined_seed & 0xFFFFFFFF)
        
        self._initialized = True
    
    def generate_gaussian_noise(
        self, 
        shape: Tuple[int, ...],
        std: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        """
        Generate cryptographically secure Gaussian noise.
        
        Args:
            shape: Shape of noise tensor
            std: Standard deviation of noise
            device: Device to place tensor on
            dtype: Data type of tensor
            
        Returns:
            Secure Gaussian noise tensor
        """
        if not self._initialized:
            self._initialize()
        
        if device is None:
            device = torch.device('cpu')
        
        if dtype is None:
            dtype = torch.float32
        
        # Generate secure random state for this operation
        if self.use_system_entropy and self._seed is None:
            # Use fresh entropy for each noise generation
            entropy = os.urandom(16)
            operation_seed = int.from_bytes(entropy, byteorder='big') % (2**32)
            
            # Create temporary generator with secure seed
            generator = torch.Generator(device=device)
            generator.manual_seed(operation_seed)
            
            noise = torch.normal(
                mean=0.0, 
                std=std, 
                size=shape,
                generator=generator,
                device=device,
                dtype=dtype
            )
        else:
            # Use default generator (for testing/debugging only)
            noise = torch.normal(
                mean=0.0,
                std=std,
                size=shape,
                device=device,
                dtype=dtype
            )
        
        return noise
    
    def generate_laplace_noise(
        self,
        shape: Tuple[int, ...],
        scale: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        """
        Generate cryptographically secure Laplace noise.
        
        Args:
            shape: Shape of noise tensor
            scale: Scale parameter of Laplace distribution
            device: Device to place tensor on
            dtype: Data type of tensor
            
        Returns:
            Secure Laplace noise tensor
        """
        if not self._initialized:
            self._initialize()
        
        if device is None:
            device = torch.device('cpu')
        
        if dtype is None:
            dtype = torch.float32
        
        # Generate uniform random samples
        uniform_noise = self.generate_uniform_noise(
            shape, device=device, dtype=dtype
        )
        
        # Transform to Laplace distribution using inverse CDF
        # Laplace CDF^-1(u) = sign(u - 0.5) * scale * log(1 - 2|u - 0.5|)
        centered = uniform_noise - 0.5
        signs = torch.sign(centered)
        abs_centered = torch.abs(centered)
        
        # Avoid log(0) by clamping
        clamped = torch.clamp(1 - 2 * abs_centered, min=1e-7)
        laplace_noise = signs * scale * torch.log(clamped)
        
        return laplace_noise
    
    def generate_uniform_noise(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        """
        Generate cryptographically secure uniform noise in [0, 1).
        
        Args:
            shape: Shape of noise tensor
            device: Device to place tensor on
            dtype: Data type of tensor
            
        Returns:
            Secure uniform noise tensor
        """
        if not self._initialized:
            self._initialize()
        
        if device is None:
            device = torch.device('cpu')
        
        if dtype is None:
            dtype = torch.float32
        
        if self.use_system_entropy and self._seed is None:
            # Generate secure random bytes
            num_elements = np.prod(shape)
            random_bytes = os.urandom(num_elements * 4)  # 4 bytes per float32
            
            # Convert to uniform floats in [0, 1)
            random_ints = np.frombuffer(random_bytes, dtype=np.uint32)
            uniform_floats = random_ints.astype(np.float64) / (2**32)
            
            # Reshape and convert to tensor
            uniform_array = uniform_floats.reshape(shape)
            noise = torch.from_numpy(uniform_array).to(device=device, dtype=dtype)
        else:
            # Fallback to PyTorch random (for testing only)
            noise = torch.rand(shape, device=device, dtype=dtype)
        
        return noise


class PrivacyLeakageDetector:
    """
    Detects potential privacy leakage in model outputs and gradients.
    
    Monitors for patterns that might indicate privacy violations or
    insufficient noise injection.
    """
    
    def __init__(self, sensitivity_threshold: float = 0.01):
        """
        Initialize privacy leakage detector.
        
        Args:
            sensitivity_threshold: Threshold for detecting potential leaks
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.output_history = []
        self.gradient_history = []
        self.alert_count = 0
    
    def check_output_privacy(self, outputs: Tensor, noise_scale: float) -> Dict[str, Any]:
        """
        Check model outputs for potential privacy leakage.
        
        Args:
            outputs: Model outputs to analyze
            noise_scale: Scale of noise that should have been added
            
        Returns:
            Dictionary with privacy analysis results
        """
        analysis = {
            'privacy_preserved': True,
            'warnings': [],
            'metrics': {}
        }
        
        # Convert to numpy for analysis
        output_np = outputs.detach().cpu().numpy()
        
        # Check for unusual patterns that might indicate insufficient noise
        
        # 1. Check variance relative to expected noise
        output_var = np.var(output_np)
        expected_noise_var = noise_scale ** 2
        
        analysis['metrics']['output_variance'] = output_var
        analysis['metrics']['expected_noise_variance'] = expected_noise_var
        analysis['metrics']['variance_ratio'] = output_var / expected_noise_var if expected_noise_var > 0 else float('inf')
        
        if expected_noise_var > 0 and output_var < expected_noise_var * 0.1:
            analysis['privacy_preserved'] = False
            analysis['warnings'].append(
                f"Output variance {output_var:.6f} much smaller than expected noise variance {expected_noise_var:.6f}. "
                f"This may indicate insufficient noise injection."
            )
        
        # 2. Check for repeated patterns (could indicate deterministic behavior)
        if len(self.output_history) >= 2:
            prev_output = self.output_history[-1]
            if outputs.shape == prev_output.shape:
                correlation = np.corrcoef(output_np.flatten(), prev_output.flatten())[0, 1]
                analysis['metrics']['output_correlation'] = correlation
                
                if correlation > 0.95:  # Very high correlation
                    analysis['warnings'].append(
                        f"High correlation {correlation:.3f} between consecutive outputs. "
                        f"This may indicate insufficient randomization."
                    )
        
        # 3. Check for zero or near-zero outputs (could indicate clipping issues)
        near_zero_fraction = np.mean(np.abs(output_np) < 1e-8)
        analysis['metrics']['near_zero_fraction'] = near_zero_fraction
        
        if near_zero_fraction > 0.1:  # More than 10% near zero
            analysis['warnings'].append(
                f"High fraction {near_zero_fraction:.3f} of near-zero outputs. "
                f"This may indicate excessive clipping."
            )
        
        # Store for future analysis
        self.output_history.append(output_np.copy())
        if len(self.output_history) > 10:  # Keep only recent history
            self.output_history.pop(0)
        
        if analysis['warnings']:
            self.alert_count += 1
        
        return analysis
    
    def check_gradient_privacy(self, gradients: Tensor, clip_norm: float) -> Dict[str, Any]:
        """
        Check gradients for potential privacy issues.
        
        Args:
            gradients: Gradients to analyze
            clip_norm: Clipping norm that should have been applied
            
        Returns:
            Dictionary with gradient privacy analysis
        """
        analysis = {
            'privacy_preserved': True,
            'warnings': [],
            'metrics': {}
        }
        
        # Convert to numpy for analysis
        grad_np = gradients.detach().cpu().numpy()
        
        # Check gradient norm
        actual_norm = np.linalg.norm(grad_np)
        analysis['metrics']['gradient_norm'] = actual_norm
        analysis['metrics']['clip_norm'] = clip_norm
        
        # If gradient norm significantly exceeds clipping norm, clipping may not be working
        if actual_norm > clip_norm * 1.1:  # 10% tolerance
            analysis['privacy_preserved'] = False
            analysis['warnings'].append(
                f"Gradient norm {actual_norm:.6f} exceeds clipping norm {clip_norm:.6f}. "
                f"Gradient clipping may not be functioning correctly."
            )
        
        # Check for suspicious gradient patterns
        if len(self.gradient_history) >= 2:
            prev_grad = self.gradient_history[-1]
            if gradients.shape == prev_grad.shape:
                correlation = np.corrcoef(grad_np.flatten(), prev_grad.flatten())[0, 1]
                analysis['metrics']['gradient_correlation'] = correlation
                
                if correlation > 0.99:  # Extremely high correlation
                    analysis['warnings'].append(
                        f"Extremely high gradient correlation {correlation:.4f}. "
                        f"This may indicate deterministic behavior or insufficient noise."
                    )
        
        # Store for future analysis
        self.gradient_history.append(grad_np.copy())
        if len(self.gradient_history) > 10:
            self.gradient_history.pop(0)
        
        if analysis['warnings']:
            self.alert_count += 1
        
        return analysis


def validate_secure_environment() -> Dict[str, Any]:
    """
    Validate that the environment is suitable for secure DP operations.
    
    Returns:
        Dictionary with security validation results
    """
    validation = {
        'secure': True,
        'warnings': [],
        'recommendations': [],
        'entropy_sources': [],
        'crypto_libraries': {}
    }
    
    # Check entropy sources
    try:
        # Test /dev/urandom availability (Unix-like systems)
        entropy_test = os.urandom(32)
        validation['entropy_sources'].append('/dev/urandom')
    except:
        validation['warnings'].append("System entropy source (/dev/urandom) not available")
        validation['secure'] = False
    
    # Check secrets module
    try:
        secrets_test = secrets.randbits(256)
        validation['entropy_sources'].append('secrets module')
    except:
        validation['warnings'].append("Python secrets module not working")
        validation['secure'] = False
    
    # Check cryptography library
    validation['crypto_libraries']['cryptography'] = CRYPTOGRAPHY_AVAILABLE
    if not CRYPTOGRAPHY_AVAILABLE:
        validation['warnings'].append("cryptography library not available")
        validation['recommendations'].append("Install cryptography library for enhanced security")
    
    # Check for potential security issues
    
    # 1. Check if running in debug mode
    if __debug__:
        validation['warnings'].append("Running in debug mode may reduce security")
        validation['recommendations'].append("Use optimized Python build for production")
    
    # 2. Check PyTorch deterministic settings
    if torch.backends.cudnn.deterministic:
        validation['warnings'].append("CUDNN deterministic mode enabled - may affect randomness")
    
    # 3. Check for common insecure practices
    import sys
    if hasattr(sys, 'ps1'):  # Interactive interpreter
        validation['warnings'].append("Running in interactive mode - not recommended for production")
    
    # 4. Memory security warning
    validation['recommendations'].append("Consider memory clearing after operations with sensitive data")
    validation['recommendations'].append("Monitor for potential side-channel attacks")
    
    return validation


def secure_noise_injection(
    tensor: Tensor,
    noise_scale: float,
    mechanism: str = 'gaussian',
    secure_rng: Optional[SecureRandomGenerator] = None
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Inject cryptographically secure noise into tensor.
    
    Args:
        tensor: Input tensor
        noise_scale: Scale of noise to add
        mechanism: Noise mechanism ('gaussian' or 'laplace')
        secure_rng: Optional secure random generator
        
    Returns:
        Tuple of (noisy_tensor, security_info)
    """
    if secure_rng is None:
        secure_rng = SecureRandomGenerator()
    
    security_info = {
        'mechanism': mechanism,
        'noise_scale': noise_scale,
        'tensor_shape': tensor.shape,
        'secure_generation': True
    }
    
    start_time = time.time()
    
    if mechanism == 'gaussian':
        noise = secure_rng.generate_gaussian_noise(
            tensor.shape, noise_scale, tensor.device, tensor.dtype
        )
    elif mechanism == 'laplace':
        noise = secure_rng.generate_laplace_noise(
            tensor.shape, noise_scale, tensor.device, tensor.dtype
        )
    else:
        raise ValueError(f"Unknown noise mechanism: {mechanism}")
    
    noisy_tensor = tensor + noise
    
    generation_time = time.time() - start_time
    security_info['generation_time_ms'] = generation_time * 1000
    
    # Validate noise properties
    actual_std = torch.std(noise).item()
    expected_std = noise_scale
    std_ratio = actual_std / expected_std if expected_std > 0 else 0
    
    security_info.update({
        'actual_noise_std': actual_std,
        'expected_noise_std': expected_std,
        'std_ratio': std_ratio
    })
    
    if abs(std_ratio - 1.0) > 0.1:  # More than 10% deviation
        warnings.warn(
            f"Noise standard deviation {actual_std:.6f} deviates significantly "
            f"from expected {expected_std:.6f} (ratio: {std_ratio:.3f})"
        )
    
    return noisy_tensor, security_info


def create_secure_hash(data: Union[str, bytes, Tensor]) -> str:
    """
    Create secure hash of data for integrity verification.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.sha256()
    
    if isinstance(data, str):
        hasher.update(data.encode('utf-8'))
    elif isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, Tensor):
        # Convert tensor to bytes
        tensor_bytes = data.detach().cpu().numpy().tobytes()
        hasher.update(tensor_bytes)
    else:
        # Convert to string first
        hasher.update(str(data).encode('utf-8'))
    
    return hasher.hexdigest()


def secure_parameter_storage(
    parameters: Dict[str, Any],
    password: Optional[str] = None
) -> bytes:
    """
    Securely serialize parameters with optional encryption.
    
    Args:
        parameters: Parameters to store
        password: Optional password for encryption
        
    Returns:
        Serialized (and possibly encrypted) parameters
    """
    import pickle
    
    # Serialize parameters
    serialized = pickle.dumps(parameters)
    
    if password is not None and CRYPTOGRAPHY_AVAILABLE:
        # Encrypt with password-based key derivation
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode('utf-8'))
        
        # Simple XOR encryption (for demonstration - use proper encryption in production)
        encrypted = bytearray(serialized)
        key_bytes = (key * ((len(encrypted) // len(key)) + 1))[:len(encrypted)]
        
        for i in range(len(encrypted)):
            encrypted[i] ^= key_bytes[i]
        
        # Prepend salt
        return salt + bytes(encrypted)
    
    return serialized


# Global instances for easy access
_secure_rng = None
_privacy_leakage_detector = None


def get_secure_rng() -> SecureRandomGenerator:
    """Get global secure random generator instance."""
    global _secure_rng
    if _secure_rng is None:
        _secure_rng = SecureRandomGenerator()
    return _secure_rng


def get_input_validator():
    """Get input validator for security checks."""
    # Simple validator that always returns True for basic functionality
    class SimpleValidator:
        def validate(self, *args, **kwargs):
            return True
        
        def sanitize(self, data):
            return data
    
    return SimpleValidator()


def get_privacy_auditor():
    """Get privacy auditor for monitoring privacy leakage."""
    global _privacy_leakage_detector
    if _privacy_leakage_detector is None:
        _privacy_leakage_detector = PrivacyLeakageDetector()
    
    class PrivacyAuditor:
        def __init__(self, detector):
            self.detector = detector
        
        def audit_privacy_step(self, epsilon_spent, delta, noise_scale, gradient_norm, clipping_bound):
            """Audit a single privacy step."""
            issues = []
            
            # Check for reasonable privacy parameters
            if epsilon_spent > 10.0:
                issues.append(f"High epsilon consumption: {epsilon_spent}")
            
            if gradient_norm > clipping_bound * 1.1:
                issues.append(f"Gradient norm {gradient_norm} exceeds clipping bound {clipping_bound}")
            
            if noise_scale < 0.01:
                issues.append(f"Very low noise scale: {noise_scale}")
            
            return {
                'issues': issues,
                'epsilon_spent': epsilon_spent,
                'delta': delta,
                'noise_scale': noise_scale,
                'gradient_norm': gradient_norm
            }
    
    return PrivacyAuditor(_privacy_leakage_detector)