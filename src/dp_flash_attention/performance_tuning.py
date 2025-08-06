"""
Advanced performance tuning and optimization for DP-Flash-Attention.

Provides adaptive performance tuning, kernel selection, memory optimization,
and hardware-specific optimizations.
"""

import time
import threading
import math
import os
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import json

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class OptimizationLevel(Enum):
    """Optimization levels for performance tuning."""
    CONSERVATIVE = "conservative"  # Safe, minimal optimizations
    BALANCED = "balanced"         # Good balance of safety and performance
    AGGRESSIVE = "aggressive"     # Maximum performance, some risk
    EXPERIMENTAL = "experimental" # Cutting-edge optimizations


@dataclass
class HardwareProfile:
    """Hardware profile for optimization targeting."""
    device_type: str  # cuda, cpu, mps
    device_name: str
    compute_capability: Optional[str] = None
    total_memory_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    tensor_cores_available: bool = False
    fp16_support: bool = False
    bf16_support: bool = False
    int8_support: bool = False


@dataclass
class PerformanceTarget:
    """Performance targets for optimization."""
    target_latency_ms: Optional[float] = None
    target_throughput_ops_per_sec: Optional[float] = None
    target_memory_efficiency: Optional[float] = None  # % of theoretical peak
    max_memory_usage_gb: Optional[float] = None
    priority_metric: str = "latency"  # latency, throughput, memory, balanced


class KernelSelector:
    """
    Intelligent kernel selection based on input characteristics and hardware.
    
    Selects optimal kernel variants for different scenarios.
    """
    
    def __init__(self):
        """Initialize kernel selector."""
        self.kernel_benchmarks: Dict[str, Dict[str, float]] = {}
        self.selection_cache: Dict[tuple, str] = {}
        self.hardware_profile: Optional[HardwareProfile] = None
        self._lock = threading.Lock()
        
        if HAS_TORCH:
            self._detect_hardware()
    
    def _detect_hardware(self) -> None:
        """Detect and profile hardware capabilities."""
        if not HAS_TORCH:
            return
            
        device_type = "cpu"
        device_name = "Unknown CPU"
        compute_capability = None
        total_memory_gb = 0.0
        tensor_cores = False
        fp16_support = True  # Most modern CPUs support fp16
        bf16_support = False
        
        if torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            
            try:
                props = torch.cuda.get_device_properties(0)
                major, minor = props.major, props.minor
                compute_capability = f"{major}.{minor}"
                total_memory_gb = props.total_memory / (1024**3)
                
                # Tensor Cores available on compute capability >= 7.0 (Volta+)
                tensor_cores = major >= 7
                
                # Modern GPUs support various precisions
                fp16_support = major >= 5  # Maxwell and later
                bf16_support = major >= 8  # Ampere and later
                
            except Exception as e:
                warnings.warn(f"Could not get detailed GPU info: {e}")
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple Silicon"
            fp16_support = True
        
        self.hardware_profile = HardwareProfile(
            device_type=device_type,
            device_name=device_name,
            compute_capability=compute_capability,
            total_memory_gb=total_memory_gb,
            tensor_cores_available=tensor_cores,
            fp16_support=fp16_support,
            bf16_support=bf16_support
        )
    
    def select_optimal_kernel(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        causal: bool = False,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    ) -> Dict[str, Any]:
        """
        Select optimal kernel configuration for given parameters.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of attention heads
            head_dim: Head dimension
            causal: Whether using causal attention
            optimization_level: Optimization aggressiveness
            
        Returns:
            Optimal kernel configuration
        """
        # Create cache key
        cache_key = (batch_size, seq_len, num_heads, head_dim, causal, optimization_level.value)
        
        with self._lock:
            if cache_key in self.selection_cache:
                cached_config = self.selection_cache[cache_key]
                return self._get_kernel_config(cached_config, optimization_level)
        
        # Analyze input characteristics
        total_ops = batch_size * seq_len * seq_len * num_heads * head_dim
        memory_requirement_gb = self._estimate_memory_requirement(
            batch_size, seq_len, num_heads, head_dim
        )
        
        # Select kernel based on characteristics
        kernel_type = self._select_kernel_type(
            total_ops, memory_requirement_gb, seq_len, optimization_level
        )
        
        # Cache selection
        with self._lock:
            self.selection_cache[cache_key] = kernel_type
        
        return self._get_kernel_config(kernel_type, optimization_level)
    
    def _select_kernel_type(
        self,
        total_ops: int,
        memory_requirement_gb: float,
        seq_len: int,
        optimization_level: OptimizationLevel
    ) -> str:
        """Select kernel type based on workload characteristics."""
        if not self.hardware_profile:
            return "fallback"
        
        # Memory-bound vs compute-bound analysis
        is_memory_bound = memory_requirement_gb > (self.hardware_profile.total_memory_gb * 0.7)
        is_large_sequence = seq_len > 2048
        is_small_batch = total_ops < 1e9
        
        # Device-specific selection
        if self.hardware_profile.device_type == "cuda":
            if is_memory_bound:
                return "memory_efficient_cuda"
            elif is_large_sequence:
                return "long_sequence_cuda" 
            elif optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.EXPERIMENTAL]:
                return "high_performance_cuda"
            else:
                return "standard_cuda"
        
        elif self.hardware_profile.device_type == "cpu":
            if is_small_batch:
                return "small_batch_cpu"
            else:
                return "standard_cpu"
        
        elif self.hardware_profile.device_type == "mps":
            return "mps_optimized"
        
        return "fallback"
    
    def _get_kernel_config(self, kernel_type: str, optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Get configuration for selected kernel type."""
        configs = {
            "standard_cuda": {
                "kernel_name": "dp_flash_attention_standard",
                "block_size": 64,
                "num_warps": 4,
                "num_stages": 2,
                "use_tensor_cores": self.hardware_profile.tensor_cores_available if self.hardware_profile else False,
                "precision": "fp16" if self.hardware_profile and self.hardware_profile.fp16_support else "fp32",
                "memory_coalescing": True,
                "shared_memory_kb": 48
            },
            
            "high_performance_cuda": {
                "kernel_name": "dp_flash_attention_optimized",
                "block_size": 128,
                "num_warps": 8,
                "num_stages": 3,
                "use_tensor_cores": True,
                "precision": "bf16" if self.hardware_profile and self.hardware_profile.bf16_support else "fp16",
                "memory_coalescing": True,
                "shared_memory_kb": 96,
                "prefetch_enabled": True,
                "async_copy": True
            },
            
            "memory_efficient_cuda": {
                "kernel_name": "dp_flash_attention_memory_efficient",
                "block_size": 32,
                "num_warps": 2,
                "num_stages": 1,
                "use_tensor_cores": False,
                "precision": "fp16",
                "memory_coalescing": True,
                "shared_memory_kb": 16,
                "gradient_checkpointing": True
            },
            
            "long_sequence_cuda": {
                "kernel_name": "dp_flash_attention_long_sequence",
                "block_size": 256,
                "num_warps": 16,
                "num_stages": 4,
                "use_tensor_cores": True,
                "precision": "fp16",
                "sequence_tiling": True,
                "shared_memory_kb": 128,
                "overlap_compute_communication": True
            },
            
            "standard_cpu": {
                "kernel_name": "dp_flash_attention_cpu",
                "num_threads": min(os.cpu_count() or 4, 16),
                "vectorization": True,
                "precision": "fp32",
                "blocking_factor": 64,
                "cache_friendly": True
            },
            
            "small_batch_cpu": {
                "kernel_name": "dp_flash_attention_cpu_small",
                "num_threads": min(os.cpu_count() or 4, 8),
                "vectorization": True,
                "precision": "fp32",
                "blocking_factor": 32,
                "cache_friendly": True,
                "loop_unrolling": True
            },
            
            "mps_optimized": {
                "kernel_name": "dp_flash_attention_mps",
                "precision": "fp16",
                "metal_performance_shaders": True,
                "unified_memory_optimization": True
            },
            
            "fallback": {
                "kernel_name": "dp_flash_attention_fallback",
                "precision": "fp32",
                "safe_mode": True,
                "validation_enabled": True
            }
        }
        
        base_config = configs.get(kernel_type, configs["fallback"])
        
        # Apply optimization level adjustments
        if optimization_level == OptimizationLevel.CONSERVATIVE:
            base_config = base_config.copy()
            base_config["safe_mode"] = True
            base_config["validation_enabled"] = True
            if "num_stages" in base_config:
                base_config["num_stages"] = min(base_config.get("num_stages", 2), 2)
        
        elif optimization_level == OptimizationLevel.AGGRESSIVE:
            base_config = base_config.copy()
            base_config["safe_mode"] = False
            base_config["validation_enabled"] = False
            if "num_stages" in base_config:
                base_config["num_stages"] = base_config.get("num_stages", 2) + 1
        
        elif optimization_level == OptimizationLevel.EXPERIMENTAL:
            base_config = base_config.copy()
            base_config["safe_mode"] = False
            base_config["validation_enabled"] = False
            base_config["experimental_optimizations"] = True
            if "async_copy" in base_config:
                base_config["async_copy"] = True
        
        base_config["optimization_level"] = optimization_level.value
        return base_config
    
    def _estimate_memory_requirement(
        self, batch_size: int, seq_len: int, num_heads: int, head_dim: int
    ) -> float:
        """Estimate memory requirement in GB."""
        # Simplified memory estimation
        element_size = 2  # fp16
        
        # Q, K, V tensors
        qkv_memory = 3 * batch_size * seq_len * num_heads * head_dim * element_size
        
        # Attention scores
        scores_memory = batch_size * num_heads * seq_len * seq_len * element_size
        
        # Output and intermediates
        output_memory = batch_size * seq_len * num_heads * head_dim * element_size
        intermediate_memory = scores_memory * 1.5  # Working memory
        
        total_bytes = qkv_memory + scores_memory + output_memory + intermediate_memory
        return total_bytes / (1024**3)


class AdaptiveOptimizer:
    """
    Adaptive optimizer that learns from performance patterns and adjusts accordingly.
    """
    
    def __init__(self, performance_target: Optional[PerformanceTarget] = None):
        """Initialize adaptive optimizer."""
        self.performance_target = performance_target or PerformanceTarget()
        self.kernel_selector = KernelSelector()
        
        # Learning state
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_parameters: Dict[str, Any] = {}
        self.learning_rate = 0.1
        self.exploration_rate = 0.1  # For exploration vs exploitation
        
        # Optimization state
        self.current_config: Optional[Dict[str, Any]] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_performance: float = float('inf')
        
        self._lock = threading.Lock()
    
    def optimize_for_workload(
        self,
        workload_samples: List[Tuple[int, int, int, int]],  # (batch_size, seq_len, num_heads, head_dim)
        max_iterations: int = 50,
        convergence_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Optimize configuration for specific workload patterns.
        
        Args:
            workload_samples: Sample workload configurations
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for improvement
            
        Returns:
            Optimized configuration
        """
        print(f"🔧 Starting adaptive optimization for {len(workload_samples)} workload samples...")
        
        # Analyze workload patterns
        workload_stats = self._analyze_workload_patterns(workload_samples)
        print(f"📊 Workload analysis: {workload_stats}")
        
        # Initialize with base configuration
        representative_sample = workload_samples[0] if workload_samples else (32, 512, 12, 64)
        batch_size, seq_len, num_heads, head_dim = representative_sample
        
        current_config = self.kernel_selector.select_optimal_kernel(
            batch_size, seq_len, num_heads, head_dim
        )
        
        best_config = current_config.copy()
        best_score = float('inf')
        
        improvement_history = []
        
        for iteration in range(max_iterations):
            # Test current configuration
            score = self._evaluate_configuration(current_config, workload_samples[:5])  # Sample for speed
            
            print(f"  Iteration {iteration + 1}: score = {score:.4f}")
            
            # Update best if improved
            if score < best_score:
                best_score = score
                best_config = current_config.copy()
                print(f"  🎯 New best score: {best_score:.4f}")
            
            improvement_history.append(score)
            
            # Check convergence
            if len(improvement_history) >= 5:
                recent_improvement = (max(improvement_history[-5:]) - min(improvement_history[-5:])) / min(improvement_history[-5:])
                if recent_improvement < convergence_threshold:
                    print(f"  ✅ Converged after {iteration + 1} iterations")
                    break
            
            # Generate next configuration to try
            current_config = self._mutate_configuration(current_config, workload_stats)
        
        # Store results
        with self._lock:
            self.best_config = best_config
            self.best_performance = best_score
        
        print(f"🎉 Optimization complete! Best score: {best_score:.4f}")
        return best_config
    
    def _analyze_workload_patterns(self, workload_samples: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Analyze patterns in workload samples."""
        if not workload_samples:
            return {}
        
        batch_sizes = [s[0] for s in workload_samples]
        seq_lens = [s[1] for s in workload_samples]
        num_heads = [s[2] for s in workload_samples]
        head_dims = [s[3] for s in workload_samples]
        
        return {
            "avg_batch_size": sum(batch_sizes) / len(batch_sizes),
            "avg_seq_len": sum(seq_lens) / len(seq_lens),
            "avg_num_heads": sum(num_heads) / len(num_heads),
            "avg_head_dim": sum(head_dims) / len(head_dims),
            "batch_size_range": (min(batch_sizes), max(batch_sizes)),
            "seq_len_range": (min(seq_lens), max(seq_lens)),
            "total_samples": len(workload_samples),
            "complexity_estimate": sum(b * s * s * h * d for b, s, h, d in workload_samples) / len(workload_samples)
        }
    
    def _evaluate_configuration(
        self, config: Dict[str, Any], workload_samples: List[Tuple[int, int, int, int]]
    ) -> float:
        """Evaluate configuration performance on workload samples."""
        # Simulate performance evaluation
        # In practice, would run actual benchmarks
        
        total_score = 0.0
        for batch_size, seq_len, num_heads, head_dim in workload_samples:
            # Estimate performance based on configuration
            base_score = math.log(batch_size * seq_len * seq_len * num_heads * head_dim)
            
            # Configuration-specific adjustments
            if config.get("use_tensor_cores", False):
                base_score *= 0.7  # Tensor cores speedup
            
            if config.get("precision") == "fp16":
                base_score *= 0.8  # FP16 speedup
            elif config.get("precision") == "bf16":
                base_score *= 0.75  # BF16 speedup
            
            block_size = config.get("block_size", 64)
            if block_size == 128:
                base_score *= 0.9  # Larger blocks can be more efficient
            elif block_size == 32:
                base_score *= 1.1  # Smaller blocks may be less efficient
            
            # Memory efficiency factor
            if config.get("gradient_checkpointing", False):
                base_score *= 1.2  # Slower due to recomputation
            
            # Add some controlled randomness to simulate measurement noise
            noise = (hash(str(config)) % 1000) / 10000.0 - 0.05  # ±5% noise
            base_score *= (1 + noise)
            
            total_score += base_score
        
        return total_score / len(workload_samples)
    
    def _mutate_configuration(
        self, config: Dict[str, Any], workload_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate mutated configuration for exploration."""
        new_config = config.copy()
        
        # Decide whether to explore or exploit
        if (hash(str(config)) % 100) / 100.0 < self.exploration_rate:
            # Exploration: make random changes
            mutations = [
                lambda c: c.update({"block_size": [32, 64, 128, 256][hash(str(c)) % 4]}),
                lambda c: c.update({"num_warps": [2, 4, 8, 16][hash(str(c)) % 4]}),
                lambda c: c.update({"num_stages": [1, 2, 3, 4][hash(str(c)) % 4]}),
                lambda c: c.update({"precision": ["fp16", "fp32", "bf16"][hash(str(c)) % 3]}),
                lambda c: c.update({"use_tensor_cores": not c.get("use_tensor_cores", False)}),
            ]
            
            # Apply random mutation
            mutation = mutations[hash(str(config)) % len(mutations)]
            mutation(new_config)
        else:
            # Exploitation: make small improvements based on workload
            if workload_stats.get("avg_seq_len", 512) > 1024:
                new_config["block_size"] = min(new_config.get("block_size", 64) * 2, 256)
            
            if workload_stats.get("complexity_estimate", 0) > 1e12:
                new_config["num_stages"] = min(new_config.get("num_stages", 2) + 1, 4)
        
        return new_config
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        with self._lock:
            return {
                "best_performance": self.best_performance,
                "best_config": self.best_config,
                "current_config": self.current_config,
                "optimization_history_length": len(self.performance_history),
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate
            }


def auto_tune_for_hardware(
    sample_workloads: List[Tuple[int, int, int, int]],
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
    max_tuning_time_minutes: int = 30
) -> Dict[str, Any]:
    """
    Automatically tune DP-Flash-Attention for current hardware.
    
    Args:
        sample_workloads: Representative workload samples
        optimization_level: Optimization aggressiveness
        max_tuning_time_minutes: Maximum time to spend tuning
        
    Returns:
        Optimized configuration
    """
    print("🚀 Starting automatic hardware tuning...")
    
    start_time = time.time()
    
    # Initialize components
    optimizer = AdaptiveOptimizer()
    
    # Calculate tuning iterations based on time budget
    max_iterations = min(max_tuning_time_minutes * 2, 100)  # Rough estimate
    
    # Perform optimization
    optimal_config = optimizer.optimize_for_workload(
        sample_workloads,
        max_iterations=max_iterations,
        convergence_threshold=0.005
    )
    
    tuning_time = time.time() - start_time
    
    print(f"⏱️  Tuning completed in {tuning_time:.1f} seconds")
    print(f"🎯 Optimal configuration: {optimal_config['kernel_name']}")
    
    # Add metadata
    optimal_config["tuning_metadata"] = {
        "tuning_time_seconds": tuning_time,
        "optimization_level": optimization_level.value,
        "sample_workloads_count": len(sample_workloads),
        "tuning_timestamp": time.time(),
        "hardware_profile": optimizer.kernel_selector.hardware_profile.__dict__ if optimizer.kernel_selector.hardware_profile else None
    }
    
    return optimal_config


def save_optimization_profile(config: Dict[str, Any], filepath: str) -> None:
    """Save optimization configuration to file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"💾 Optimization profile saved to {filepath}")
    except Exception as e:
        warnings.warn(f"Could not save optimization profile: {e}")


def load_optimization_profile(filepath: str) -> Optional[Dict[str, Any]]:
    """Load optimization configuration from file."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        print(f"📂 Optimization profile loaded from {filepath}")
        return config
    except Exception as e:
        warnings.warn(f"Could not load optimization profile: {e}")
        return None


# Global optimization cache
_optimization_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()


def get_cached_optimization(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int
) -> Optional[Dict[str, Any]]:
    """Get cached optimization configuration."""
    cache_key = f"{batch_size}_{seq_len}_{num_heads}_{head_dim}"
    
    with _cache_lock:
        return _optimization_cache.get(cache_key)


def cache_optimization(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int, config: Dict[str, Any]
) -> None:
    """Cache optimization configuration."""
    cache_key = f"{batch_size}_{seq_len}_{num_heads}_{head_dim}"
    
    with _cache_lock:
        _optimization_cache[cache_key] = config.copy()
        
        # Limit cache size
        if len(_optimization_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(_optimization_cache.keys())[:100]
            for key in keys_to_remove:
                del _optimization_cache[key]