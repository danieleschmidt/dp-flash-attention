"""
Configuration management and validation for DP-Flash-Attention.

Provides safe configuration loading, validation, and management with
built-in security checks and reasonable defaults.
"""

import os
import json
import warnings
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading

import torch

from .validation import (
    validate_privacy_parameters_comprehensive,
    validate_attention_configuration,
    ConfigurationError
)


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy parameters."""
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_mechanism: str = 'gaussian'  # 'gaussian' or 'laplace'
    secure_rng: bool = True
    
    def __post_init__(self):
        """Validate privacy configuration after initialization."""
        validate_privacy_parameters_comprehensive(
            self.epsilon, self.delta, self.max_grad_norm
        )
        
        if self.noise_mechanism not in ['gaussian', 'laplace']:
            raise ConfigurationError(
                f"Invalid noise mechanism: {self.noise_mechanism}. "
                f"Must be 'gaussian' or 'laplace'"
            )


@dataclass  
class AttentionConfig:
    """Configuration for attention layer parameters."""
    embed_dim: int = 768
    num_heads: int = 12
    dropout: float = 0.0
    bias: bool = True
    batch_first: bool = True
    
    def __post_init__(self):
        """Validate attention configuration after initialization."""
        if self.embed_dim <= 0:
            raise ConfigurationError(f"embed_dim must be positive, got {self.embed_dim}")
        
        if self.num_heads <= 0:
            raise ConfigurationError(f"num_heads must be positive, got {self.num_heads}")
        
        if self.embed_dim % self.num_heads != 0:
            raise ConfigurationError(
                f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}"
            )
        
        if not 0.0 <= self.dropout <= 1.0:
            raise ConfigurationError(f"dropout must be in [0, 1], got {self.dropout}")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    use_flash_attention: bool = True
    use_checkpointing: bool = False
    checkpoint_segments: int = 4
    max_sequence_length: int = 16384
    memory_efficient: bool = True
    compile_kernels: bool = False
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.checkpoint_segments <= 0:
            raise ConfigurationError(f"checkpoint_segments must be positive, got {self.checkpoint_segments}")
        
        if self.max_sequence_length <= 0:
            raise ConfigurationError(f"max_sequence_length must be positive, got {self.max_sequence_length}")


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    enable_privacy_tracking: bool = True
    enable_performance_tracking: bool = True
    enable_prometheus: bool = False
    enable_opentelemetry: bool = False
    prometheus_port: int = 8000
    log_level: str = 'INFO'
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_epsilon_per_hour': 10.0,
        'max_duration_ms': 5000.0,
        'max_memory_mb': 8192.0,
        'max_gpu_utilization': 0.95
    })
    
    def __post_init__(self):
        """Validate monitoring configuration."""
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}"
            )
        
        if not 1024 <= self.prometheus_port <= 65535:
            raise ConfigurationError(f"prometheus_port must be in [1024, 65535], got {self.prometheus_port}")


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    validate_inputs: bool = True
    use_secure_rng: bool = True
    enable_privacy_audit: bool = True
    memory_clearing: bool = False
    strict_mode: bool = False
    
    def __post_init__(self):
        """Validate security configuration."""
        if self.strict_mode and not self.use_secure_rng:
            warnings.warn("strict_mode enabled but secure RNG disabled - this reduces security")


@dataclass
class DPFlashAttentionConfig:
    """Main configuration class for DP-Flash-Attention."""
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Global settings
    device: Optional[str] = None
    dtype: Optional[str] = None
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate complete configuration."""
        # Validate device setting
        if self.device is not None:
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                warnings.warn(f"CUDA device '{self.device}' requested but CUDA not available")
        
        # Validate dtype setting
        if self.dtype is not None:
            valid_dtypes = ['float16', 'bfloat16', 'float32', 'float64']
            if self.dtype not in valid_dtypes:
                raise ConfigurationError(f"Invalid dtype: {self.dtype}. Must be one of {valid_dtypes}")
        
        # Security warnings
        if self.seed is not None:
            warnings.warn("Fixed seed specified - this reduces randomness and may impact privacy")
        
        # Consistency checks
        if self.security.strict_mode:
            if self.privacy.epsilon > 1.0:
                warnings.warn(f"Strict mode enabled but epsilon={self.privacy.epsilon} > 1.0")
            
            if not self.security.use_secure_rng:
                raise ConfigurationError("Strict mode requires secure RNG to be enabled")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DPFlashAttentionConfig':
        """Create configuration from dictionary."""
        # Extract nested configurations
        privacy_dict = config_dict.get('privacy', {})
        attention_dict = config_dict.get('attention', {})
        performance_dict = config_dict.get('performance', {})
        monitoring_dict = config_dict.get('monitoring', {})
        security_dict = config_dict.get('security', {})
        
        return cls(
            privacy=PrivacyConfig(**privacy_dict),
            attention=AttentionConfig(**attention_dict),
            performance=PerformanceConfig(**performance_dict),
            monitoring=MonitoringConfig(**monitoring_dict),
            security=SecurityConfig(**security_dict),
            device=config_dict.get('device'),
            dtype=config_dict.get('dtype'),
            seed=config_dict.get('seed')
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'DPFlashAttentionConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ConfigurationError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")


class ConfigManager:
    """
    Thread-safe configuration manager with environment variable support.
    
    Manages configuration loading, validation, and runtime updates.
    """
    
    def __init__(self, config: Optional[DPFlashAttentionConfig] = None):
        """
        Initialize configuration manager.
        
        Args:
            config: Initial configuration (uses defaults if None)
        """
        self._config = config or DPFlashAttentionConfig()
        self._lock = threading.RLock()
        self._observers: List[callable] = []
    
    def get_config(self) -> DPFlashAttentionConfig:
        """Get current configuration (thread-safe)."""
        with self._lock:
            return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        with self._lock:
            # Create new config with updates
            current_dict = self._config.to_dict()
            self._deep_update(current_dict, updates)
            
            # Validate new configuration
            new_config = DPFlashAttentionConfig.from_dict(current_dict)
            
            # Update if validation passes
            old_config = self._config
            self._config = new_config
            
            # Notify observers
            for observer in self._observers:
                try:
                    observer(old_config, new_config)
                except Exception as e:
                    warnings.warn(f"Configuration observer failed: {e}")
    
    def _deep_update(self, base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep update dictionary with nested updates."""
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def add_observer(self, observer: callable) -> None:
        """Add configuration change observer."""
        with self._lock:
            self._observers.append(observer)
    
    def remove_observer(self, observer: callable) -> None:
        """Remove configuration change observer."""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
    
    def load_from_env(self, prefix: str = 'DP_FLASH_') -> None:
        """
        Load configuration overrides from environment variables.
        
        Args:
            prefix: Environment variable prefix
        """
        env_overrides = {}
        
        # Map environment variables to config paths
        env_mappings = {
            f'{prefix}EPSILON': 'privacy.epsilon',
            f'{prefix}DELTA': 'privacy.delta',
            f'{prefix}MAX_GRAD_NORM': 'privacy.max_grad_norm',
            f'{prefix}EMBED_DIM': 'attention.embed_dim',
            f'{prefix}NUM_HEADS': 'attention.num_heads',
            f'{prefix}DROPOUT': 'attention.dropout',
            f'{prefix}DEVICE': 'device',
            f'{prefix}DTYPE': 'dtype',
            f'{prefix}LOG_LEVEL': 'monitoring.log_level',
            f'{prefix}SECURE_RNG': 'security.use_secure_rng',
            f'{prefix}STRICT_MODE': 'security.strict_mode',
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Parse value based on type
                parsed_value = self._parse_env_value(value)
                self._set_nested_value(env_overrides, config_path, parsed_value)
        
        if env_overrides:
            self.update_config(env_overrides)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String value
        return value
    
    def _set_nested_value(self, dictionary: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation path."""
        keys = path.split('.')
        current = dictionary
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def validate_runtime_constraints(self) -> List[str]:
        """
        Validate runtime constraints and return any violations.
        
        Returns:
            List of constraint violation messages
        """
        violations = []
        config = self.get_config()
        
        # Check privacy budget constraints
        if config.privacy.epsilon > 10.0:
            violations.append(f"Privacy epsilon {config.privacy.epsilon} exceeds recommended maximum of 10.0")
        
        # Check memory constraints for configuration
        if torch.cuda.is_available():
            try:
                device_props = torch.cuda.get_device_properties(0)
                total_memory_gb = device_props.total_memory / (1024**3)
                
                # Estimate memory requirements
                estimated_memory = self._estimate_memory_usage(config)
                
                if estimated_memory > total_memory_gb * 0.8:  # 80% of GPU memory
                    violations.append(
                        f"Estimated memory usage {estimated_memory:.1f}GB exceeds "
                        f"80% of available GPU memory {total_memory_gb:.1f}GB"
                    )
            except Exception:
                pass  # Skip if GPU properties can't be checked
        
        # Check sequence length constraints
        if config.performance.max_sequence_length > 32768:
            violations.append(
                f"Maximum sequence length {config.performance.max_sequence_length} "
                f"may cause memory issues"
            )
        
        return violations
    
    def _estimate_memory_usage(self, config: DPFlashAttentionConfig) -> float:
        """Estimate memory usage for given configuration (rough approximation)."""
        # Simplified memory estimation
        embed_dim = config.attention.embed_dim
        max_seq_len = config.performance.max_sequence_length
        
        # Model parameters (4 linear layers)
        param_memory = 4 * embed_dim * embed_dim * 4 / (1024**3)  # 4 bytes per parameter
        
        # Activation memory (rough estimate for batch_size=32)
        activation_memory = 32 * max_seq_len * embed_dim * 8 / (1024**3)  # Factor for intermediate tensors
        
        return param_memory + activation_memory


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None
_config_lock = threading.Lock()


def get_global_config() -> DPFlashAttentionConfig:
    """Get global configuration instance."""
    global _global_config_manager
    
    if _global_config_manager is None:
        with _config_lock:
            if _global_config_manager is None:
                _global_config_manager = ConfigManager()
                # Load environment variables
                _global_config_manager.load_from_env()
    
    return _global_config_manager.get_config()


def update_global_config(updates: Dict[str, Any]) -> None:
    """Update global configuration."""
    global _global_config_manager
    
    if _global_config_manager is None:
        with _config_lock:
            if _global_config_manager is None:
                _global_config_manager = ConfigManager()
    
    _global_config_manager.update_config(updates)


def load_config_from_file(filepath: Union[str, Path]) -> DPFlashAttentionConfig:
    """Load configuration from file."""
    return DPFlashAttentionConfig.load(filepath)


def create_default_config(
    privacy_level: str = 'moderate',
    performance_mode: str = 'balanced'
) -> DPFlashAttentionConfig:
    """
    Create configuration with predefined presets.
    
    Args:
        privacy_level: 'strong', 'moderate', or 'weak'
        performance_mode: 'fast', 'balanced', or 'memory_efficient'
        
    Returns:
        Configured DPFlashAttentionConfig
    """
    # Privacy presets
    privacy_presets = {
        'strong': {'epsilon': 0.5, 'delta': 1e-6, 'max_grad_norm': 0.5},
        'moderate': {'epsilon': 1.0, 'delta': 1e-5, 'max_grad_norm': 1.0},
        'weak': {'epsilon': 3.0, 'delta': 1e-4, 'max_grad_norm': 2.0}
    }
    
    # Performance presets
    performance_presets = {
        'fast': {
            'use_flash_attention': True,
            'use_checkpointing': False,
            'memory_efficient': False,
            'compile_kernels': True
        },
        'balanced': {
            'use_flash_attention': True,
            'use_checkpointing': False,
            'memory_efficient': True,
            'compile_kernels': False
        },
        'memory_efficient': {
            'use_flash_attention': True,
            'use_checkpointing': True,
            'memory_efficient': True,
            'checkpoint_segments': 8
        }
    }
    
    if privacy_level not in privacy_presets:
        raise ValueError(f"Invalid privacy_level: {privacy_level}")
    
    if performance_mode not in performance_presets:
        raise ValueError(f"Invalid performance_mode: {performance_mode}")
    
    return DPFlashAttentionConfig(
        privacy=PrivacyConfig(**privacy_presets[privacy_level]),
        performance=PerformanceConfig(**performance_presets[performance_mode])
    )