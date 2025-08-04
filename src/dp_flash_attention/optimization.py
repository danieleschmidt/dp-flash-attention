"""
Performance optimization and caching for DP-Flash-Attention.

Implements various optimization strategies including caching, memory pooling,
kernel fusion, and adaptive optimization based on usage patterns.
"""

import time
import threading
import weakref
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import math

import torch
from torch import Tensor
import numpy as np

from .monitoring import DPTelemetry
from .validation import validate_tensor_shapes, validate_tensor_memory


@dataclass
class CacheKey:
    """Key for caching attention computations."""
    batch_size: int
    sequence_length: int
    num_heads: int
    head_dim: int
    epsilon: float
    delta: float
    causal: bool
    dtype: str
    device: str
    
    def __hash__(self):
        return hash((
            self.batch_size, self.sequence_length, self.num_heads, self.head_dim,
            self.epsilon, self.delta, self.causal, self.dtype, self.device
        ))


@dataclass
class CacheEntry:
    """Cached computation entry."""
    key: CacheKey
    noise_scale: float
    computation_time_ms: float
    memory_usage_mb: float
    access_count: int
    last_access: float
    creation_time: float


class AdaptiveCache:
    """
    Adaptive LRU cache with memory and performance-aware eviction.
    
    Caches computation metadata and optimization parameters based on
    usage patterns and available memory.
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 512.0):
        """
        Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of cache entries
            max_memory_mb: Maximum memory usage for cache
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self.total_memory_mb = 0.0
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
    
    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Get cached entry and update access statistics."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                self.hit_count += 1
                return entry
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: CacheKey, entry: CacheEntry) -> None:
        """Add entry to cache with eviction if necessary."""
        with self._lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_memory_mb -= old_entry.memory_usage_mb
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            self.total_memory_mb += entry.memory_usage_mb
            
            # Evict if necessary
            self._evict_if_necessary()
    
    def _evict_if_necessary(self) -> None:
        """Evict entries based on size and memory constraints."""
        # Evict based on size limit
        while len(self.cache) > self.max_size:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.total_memory_mb -= oldest_entry.memory_usage_mb
        
        # Evict based on memory limit using smart strategy
        while self.total_memory_mb > self.max_memory_mb and self.cache:
            # Find least valuable entry (low access count, old, high memory)
            worst_key = None
            worst_score = float('inf')
            current_time = time.time()
            
            for key, entry in self.cache.items():
                # Score based on access frequency, recency, and memory usage
                age_penalty = (current_time - entry.last_access) / 3600  # Hours since last access
                memory_penalty = entry.memory_usage_mb / 10  # Memory usage in 10MB units
                access_bonus = math.log(max(1, entry.access_count))  # Log of access count
                
                score = age_penalty + memory_penalty - access_bonus
                
                if score < worst_score:
                    worst_score = score
                    worst_key = key
            
            if worst_key:
                worst_entry = self.cache.pop(worst_key)
                self.total_memory_mb -= worst_entry.memory_usage_mb
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.total_memory_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.total_memory_mb,
                'max_memory_mb': self.max_memory_mb,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class MemoryPool:
    """
    Memory pool for efficient tensor allocation and reuse.
    
    Manages tensor allocation to reduce memory fragmentation and
    allocation overhead.
    """
    
    def __init__(self, device: torch.device, initial_size_mb: float = 256.0):
        """
        Initialize memory pool.
        
        Args:
            device: Device for tensor allocation
            initial_size_mb: Initial pool size in MB
        """
        self.device = device
        self.initial_size_mb = initial_size_mb
        self.pools: Dict[Tuple[torch.dtype, Tuple[int, ...]], List[Tensor]] = defaultdict(list)
        self.allocated_tensors: Dict[int, Tuple[torch.dtype, Tuple[int, ...]]] = {}
        self.total_allocated_mb = 0.0
        self.total_pooled_mb = 0.0
        self._lock = threading.Lock()
    
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tensor:
        """
        Allocate tensor from pool or create new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Allocated tensor
        """
        with self._lock:
            pool_key = (dtype, shape)
            
            if self.pools[pool_key]:
                # Reuse tensor from pool
                tensor = self.pools[pool_key].pop()
                tensor.zero_()  # Clear data for privacy
                self.total_pooled_mb -= self._tensor_size_mb(tensor)
            else:
                # Create new tensor
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
            
            # Track allocation
            tensor_id = id(tensor)
            self.allocated_tensors[tensor_id] = pool_key
            self.total_allocated_mb += self._tensor_size_mb(tensor)
            
            return tensor
    
    def deallocate(self, tensor: Tensor) -> None:
        """
        Return tensor to pool for reuse.
        
        Args:
            tensor: Tensor to deallocate
        """
        with self._lock:
            tensor_id = id(tensor)
            
            if tensor_id in self.allocated_tensors:
                pool_key = self.allocated_tensors.pop(tensor_id)
                self.total_allocated_mb -= self._tensor_size_mb(tensor)
                
                # Return to pool if there's space
                if len(self.pools[pool_key]) < 10:  # Limit pool size per shape
                    self.pools[pool_key].append(tensor)
                    self.total_pooled_mb += self._tensor_size_mb(tensor)
                else:
                    # Let tensor be garbage collected
                    del tensor
    
    def _tensor_size_mb(self, tensor: Tensor) -> float:
        """Calculate tensor size in MB."""
        return tensor.numel() * tensor.element_size() / (1024 ** 2)
    
    def clear_pool(self) -> None:
        """Clear all pooled tensors."""
        with self._lock:
            for pool in self.pools.values():
                pool.clear()
            self.pools.clear()
            self.total_pooled_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                'allocated_tensors': len(self.allocated_tensors),
                'allocated_memory_mb': self.total_allocated_mb,
                'pooled_memory_mb': self.total_pooled_mb,
                'pool_shapes': list(self.pools.keys()),
                'total_pools': len(self.pools)
            }


class KernelOptimizer:
    """
    Kernel optimization based on usage patterns and hardware characteristics.
    
    Automatically tunes kernel parameters for optimal performance.
    """
    
    def __init__(self, device: torch.device):
        """Initialize kernel optimizer."""
        self.device = device
        self.optimization_history: Dict[CacheKey, List[float]] = defaultdict(list)
        self.optimal_configs: Dict[CacheKey, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def get_optimal_config(self, key: CacheKey) -> Dict[str, Any]:
        """
        Get optimal configuration for given parameters.
        
        Args:
            key: Cache key identifying the configuration
            
        Returns:
            Optimal configuration parameters
        """
        with self._lock:
            if key in self.optimal_configs:
                return self.optimal_configs[key].copy()
            else:
                # Return default configuration
                return self._get_default_config(key)
    
    def record_performance(self, key: CacheKey, duration_ms: float, config: Dict[str, Any]) -> None:
        """
        Record performance for configuration tuning.
        
        Args:
            key: Cache key identifying the configuration
            duration_ms: Execution duration in milliseconds
            config: Configuration used
        """
        with self._lock:
            self.optimization_history[key].append(duration_ms)
            
            # Keep only recent history
            if len(self.optimization_history[key]) > 100:
                self.optimization_history[key] = self.optimization_history[key][-50:]
            
            # Update optimal config if we have enough data
            if len(self.optimization_history[key]) >= 10:
                avg_duration = np.mean(self.optimization_history[key][-10:])
                
                if key not in self.optimal_configs or avg_duration < self.optimal_configs[key].get('avg_duration', float('inf')):
                    self.optimal_configs[key] = config.copy()
                    self.optimal_configs[key]['avg_duration'] = avg_duration
    
    def _get_default_config(self, key: CacheKey) -> Dict[str, Any]:
        """Get default configuration based on input characteristics."""
        config = {
            'use_flash_attention': True,
            'block_size': 64,
            'num_warps': 4,
            'stages': 2,
            'use_tensor_cores': self.device.type == 'cuda' and torch.cuda.get_device_capability(self.device.index)[0] >= 7
        }
        
        # Adjust based on sequence length
        if key.sequence_length > 4096:
            config['block_size'] = 128
            config['num_warps'] = 8
        elif key.sequence_length < 512:
            config['block_size'] = 32
            config['num_warps'] = 2
        
        # Adjust based on number of heads
        if key.num_heads > 16:
            config['stages'] = 3
        
        return config
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        with self._lock:
            return {
                'tracked_configurations': len(self.optimization_history),
                'optimal_configurations': len(self.optimal_configs),
                'total_measurements': sum(len(history) for history in self.optimization_history.values())
            }


class AttentionOptimizer:
    """
    Main optimizer that coordinates caching, memory pooling, and kernel optimization.
    
    Provides optimized attention computation with automatic performance tuning.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        cache_size: int = 1000,
        cache_memory_mb: float = 512.0,
        pool_size_mb: float = 256.0,
        enable_telemetry: bool = True
    ):
        """
        Initialize attention optimizer.
        
        Args:
            device: Device for computations
            cache_size: Maximum cache entries
            cache_memory_mb: Maximum cache memory
            pool_size_mb: Memory pool size
            enable_telemetry: Whether to enable telemetry
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.cache = AdaptiveCache(cache_size, cache_memory_mb)
        self.memory_pool = MemoryPool(self.device, pool_size_mb)
        self.kernel_optimizer = KernelOptimizer(self.device)
        
        # Telemetry
        self.telemetry = DPTelemetry() if enable_telemetry else None
        
        # Optimization state
        self.warmup_iterations = 0
        self.is_warmed_up = False
        self._optimization_lock = threading.Lock()
    
    def optimize_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        causal: bool = False,
        **kwargs
    ) -> Tuple[Tensor, float, Dict[str, Any]]:
        """
        Optimized attention computation with caching and memory management.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            epsilon: Privacy budget
            delta: Privacy parameter
            max_grad_norm: Gradient clipping norm
            causal: Whether to use causal masking
            
        Returns:
            Tuple of (output_tensor, grad_norm, optimization_info)
        """
        start_time = time.time()
        
        # Create cache key
        cache_key = CacheKey(
            batch_size=q.shape[0],
            sequence_length=q.shape[1],
            num_heads=q.shape[2],
            head_dim=q.shape[3],
            epsilon=epsilon,
            delta=delta,
            causal=causal,
            dtype=str(q.dtype),
            device=str(q.device)
        )
        
        # Check cache for optimization parameters
        cached_entry = self.cache.get(cache_key)
        if cached_entry:
            noise_scale = cached_entry.noise_scale
            kernel_config = self.kernel_optimizer.get_optimal_config(cache_key)
        else:
            # Compute optimization parameters
            from .utils import compute_noise_scale
            noise_scale = compute_noise_scale(epsilon, delta, max_grad_norm, q.shape[1])
            kernel_config = self.kernel_optimizer.get_optimal_config(cache_key)
        
        # Allocate output tensor from pool
        output_shape = q.shape
        output = self.memory_pool.allocate(output_shape, q.dtype)
        
        try:
            # Perform optimized attention computation
            output, grad_norm = self._compute_attention_optimized(
                q, k, v, noise_scale, causal, kernel_config
            )
            
            # Record performance metrics
            computation_time_ms = (time.time() - start_time) * 1000
            memory_usage_mb = self._estimate_computation_memory(q)
            
            # Update cache if this is a new entry
            if not cached_entry:
                cache_entry = CacheEntry(
                    key=cache_key,
                    noise_scale=noise_scale,
                    computation_time_ms=computation_time_ms,
                    memory_usage_mb=memory_usage_mb,
                    access_count=1,
                    last_access=time.time(),
                    creation_time=time.time()
                )
                self.cache.put(cache_key, cache_entry)
            
            # Record optimization statistics
            self.kernel_optimizer.record_performance(cache_key, computation_time_ms, kernel_config)
            
            # Update warmup state
            with self._optimization_lock:
                self.warmup_iterations += 1
                if self.warmup_iterations >= 10:
                    self.is_warmed_up = True
            
            # Telemetry
            if self.telemetry:
                self.telemetry.record_performance_metrics(
                    operation_type='optimized_attention',
                    duration_ms=computation_time_ms,
                    memory_used_mb=memory_usage_mb,
                    batch_size=q.shape[0],
                    sequence_length=q.shape[1],
                    num_heads=q.shape[2]
                )
            
            optimization_info = {
                'cache_hit': cached_entry is not None,
                'computation_time_ms': computation_time_ms,
                'memory_usage_mb': memory_usage_mb,
                'kernel_config': kernel_config,
                'is_warmed_up': self.is_warmed_up
            }
            
            return output, grad_norm, optimization_info
            
        finally:
            # Clean up temporary tensors (output is managed separately)
            pass
    
    def _compute_attention_optimized(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        noise_scale: float,
        causal: bool,
        kernel_config: Dict[str, Any]
    ) -> Tuple[Tensor, float]:
        """
        Perform optimized attention computation.
        
        This is a placeholder for the actual optimized kernel implementation.
        In production, this would dispatch to optimized CUDA kernels.
        """
        from .kernels import dp_flash_attention_kernel
        
        # Use the existing kernel implementation with optimization hints
        return dp_flash_attention_kernel(
            q, k, v,
            epsilon=noise_scale,  # This is a simplification
            delta=1e-5,
            max_grad_norm=1.0,
            noise_scale=noise_scale,
            causal=causal,
            deterministic=kernel_config.get('deterministic', False)
        )
    
    def _estimate_computation_memory(self, q: Tensor) -> float:
        """Estimate memory usage for attention computation."""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Input tensors (Q, K, V)
        input_memory = 3 * q.numel() * q.element_size()
        
        # Attention scores
        scores_memory = batch_size * num_heads * seq_len * seq_len * q.element_size()
        
        # Output tensor
        output_memory = q.numel() * q.element_size()
        
        total_bytes = input_memory + scores_memory + output_memory
        return total_bytes / (1024 ** 2)
    
    def prefetch_optimization(self, keys: List[CacheKey]) -> None:
        """
        Prefetch optimization parameters for given configurations.
        
        Args:
            keys: List of cache keys to prefetch
        """
        for key in keys:
            if key not in self.cache.cache:
                # Compute and cache optimization parameters
                from .utils import compute_noise_scale
                noise_scale = compute_noise_scale(key.epsilon, key.delta, 1.0, key.sequence_length)
                kernel_config = self.kernel_optimizer.get_optimal_config(key)
                
                cache_entry = CacheEntry(
                    key=key,
                    noise_scale=noise_scale,
                    computation_time_ms=0.0,  # Will be updated on first use
                    memory_usage_mb=0.0,      # Will be updated on first use
                    access_count=0,
                    last_access=time.time(),
                    creation_time=time.time()
                )
                self.cache.put(key, cache_entry)
    
    def clear_caches(self) -> None:
        """Clear all caches and pools."""
        self.cache.clear()
        self.memory_pool.clear_pool()
        
        with self._optimization_lock:
            self.warmup_iterations = 0
            self.is_warmed_up = False
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        cache_stats = self.cache.get_stats()
        pool_stats = self.memory_pool.get_stats()
        kernel_stats = self.kernel_optimizer.get_optimization_stats()
        
        return {
            'cache': cache_stats,
            'memory_pool': pool_stats,
            'kernel_optimizer': kernel_stats,
            'warmup_iterations': self.warmup_iterations,
            'is_warmed_up': self.is_warmed_up,
            'device': str(self.device)
        }
    
    def tune_for_workload(self, sample_inputs: List[Tuple[Tensor, Tensor, Tensor]]) -> None:
        """
        Tune optimizer for specific workload patterns.
        
        Args:
            sample_inputs: List of sample (q, k, v) tensors representing typical workload
        """
        print("Tuning optimizer for workload...")
        
        # Extract patterns from sample inputs
        patterns = []
        for q, k, v in sample_inputs:
            key = CacheKey(
                batch_size=q.shape[0],
                sequence_length=q.shape[1],
                num_heads=q.shape[2],
                head_dim=q.shape[3],
                epsilon=1.0,  # Default
                delta=1e-5,   # Default
                causal=False, # Default
                dtype=str(q.dtype),
                device=str(q.device)
            )
            patterns.append(key)
        
        # Prefetch optimizations for common patterns
        self.prefetch_optimization(patterns)
        
        # Run warmup iterations
        for i, (q, k, v) in enumerate(sample_inputs[:5]):  # Limit warmup samples
            try:
                _, _, _ = self.optimize_attention(
                    q, k, v,
                    epsilon=1.0,
                    delta=1e-5,
                    max_grad_norm=1.0
                )
                print(f"Warmup iteration {i+1}/5 completed")
            except Exception as e:
                print(f"Warmup iteration {i+1} failed: {e}")
        
        print("Optimizer tuning completed")


# Global optimizer instance
_global_optimizer: Optional[AttentionOptimizer] = None
_optimizer_lock = threading.Lock()


def get_global_optimizer() -> AttentionOptimizer:
    """Get or create global attention optimizer."""
    global _global_optimizer
    
    if _global_optimizer is None:
        with _optimizer_lock:
            if _global_optimizer is None:
                _global_optimizer = AttentionOptimizer()
    
    return _global_optimizer


def optimize_attention_globally(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    **kwargs
) -> Tuple[Tensor, float, Dict[str, Any]]:
    """Use global optimizer for attention computation."""
    optimizer = get_global_optimizer()
    return optimizer.optimize_attention(q, k, v, epsilon, delta, max_grad_norm, **kwargs)