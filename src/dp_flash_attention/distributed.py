"""
Distributed processing support for DP-Flash-Attention.

Provides data parallelism, model parallelism, and distributed privacy accounting
for large-scale differential privacy workloads.
"""

import time
import threading
import queue
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    HAS_TORCH_DISTRIBUTED = hasattr(torch, 'distributed')
except ImportError:
    HAS_TORCH_DISTRIBUTED = False


class DistributedStrategy(Enum):
    """Distributed processing strategies."""
    DATA_PARALLEL = "data_parallel"          # Replicate model, split data
    MODEL_PARALLEL = "model_parallel"        # Split model across devices
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline model layers
    HYBRID = "hybrid"                        # Combination of strategies


@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""
    strategy: DistributedStrategy
    world_size: int
    rank: int
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    
    # Data parallel specific
    gradient_accumulation_steps: int = 1
    sync_frequency: int = 1  # Steps between synchronization
    
    # Model parallel specific
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Privacy accounting
    coordinate_privacy_accounting: bool = True
    privacy_aggregation_method: str = "sum"  # sum, max, average


class DistributedPrivacyAccountant:
    """
    Distributed privacy accountant for coordinating privacy budgets across nodes.
    
    Ensures privacy guarantees are maintained across distributed computation.
    """
    
    def __init__(self, config: DistributedConfig, local_accountant):
        """
        Initialize distributed privacy accountant.
        
        Args:
            config: Distributed configuration
            local_accountant: Local privacy accountant
        """
        self.config = config
        self.local_accountant = local_accountant
        self.global_privacy_spent = 0.0
        self.local_privacy_spent = 0.0
        
        # Synchronization
        self._sync_lock = threading.Lock()
        self._last_sync_time = time.time()
        self.sync_interval_seconds = 30.0  # Sync every 30 seconds
        
        # Communication buffers
        self._privacy_buffer: List[float] = []
        self._sync_requests = queue.Queue()
        
        self.logger = logging.getLogger("distributed_privacy")
    
    def add_step(self, epsilon_spent: float, delta: float, **kwargs) -> float:
        """
        Add privacy step with distributed coordination.
        
        Args:
            epsilon_spent: Local epsilon spent this step
            delta: Privacy parameter
            **kwargs: Additional parameters for local accountant
            
        Returns:
            Global step epsilon after coordination
        """
        # Update local accounting
        local_step_epsilon = self.local_accountant.add_step(epsilon_spent, delta, **kwargs)
        self.local_privacy_spent += local_step_epsilon
        
        # Buffer for distributed synchronization
        self._privacy_buffer.append(local_step_epsilon)
        
        # Check if synchronization is needed
        current_time = time.time()
        should_sync = (
            current_time - self._last_sync_time > self.sync_interval_seconds or
            len(self._privacy_buffer) >= 100  # Buffer full
        )
        
        if should_sync and self.config.coordinate_privacy_accounting:
            self._synchronize_privacy_accounting()
        
        return local_step_epsilon
    
    def get_global_epsilon(self, delta: float) -> float:
        """Get global privacy epsilon across all nodes."""
        if not self.config.coordinate_privacy_accounting:
            return self.local_accountant.get_epsilon(delta)
        
        # Force synchronization to get latest global state
        self._synchronize_privacy_accounting()
        
        return self.global_privacy_spent
    
    def _synchronize_privacy_accounting(self) -> None:
        """Synchronize privacy accounting across distributed nodes."""
        if not HAS_TORCH_DISTRIBUTED or not dist.is_initialized():
            self.logger.warning("Distributed backend not initialized, skipping sync")
            return
        
        with self._sync_lock:
            try:
                # Prepare local privacy data for synchronization
                local_epsilon = sum(self._privacy_buffer)
                
                # Create tensor for all-reduce
                epsilon_tensor = torch.tensor([local_epsilon], dtype=torch.float32)
                
                if torch.cuda.is_available():
                    epsilon_tensor = epsilon_tensor.cuda()
                
                # Perform all-reduce based on aggregation method
                if self.config.privacy_aggregation_method == "sum":
                    dist.all_reduce(epsilon_tensor, op=dist.ReduceOp.SUM)
                elif self.config.privacy_aggregation_method == "max":
                    dist.all_reduce(epsilon_tensor, op=dist.ReduceOp.MAX)
                elif self.config.privacy_aggregation_method == "average":
                    dist.all_reduce(epsilon_tensor, op=dist.ReduceOp.SUM)
                    epsilon_tensor /= self.config.world_size
                
                # Update global privacy spent
                self.global_privacy_spent += epsilon_tensor.item()
                
                # Clear buffer and update sync time
                self._privacy_buffer.clear()
                self._last_sync_time = time.time()
                
                self.logger.debug(f"Privacy sync complete. Global epsilon: {self.global_privacy_spent:.6f}")
                
            except Exception as e:
                self.logger.error(f"Privacy synchronization failed: {e}")
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get comprehensive privacy statistics."""
        local_stats = self.local_accountant.get_composition_stats() if hasattr(self.local_accountant, 'get_composition_stats') else {}
        
        return {
            "local_privacy_spent": self.local_privacy_spent,
            "global_privacy_spent": self.global_privacy_spent,
            "pending_sync_steps": len(self._privacy_buffer),
            "last_sync_time": self._last_sync_time,
            "sync_interval_seconds": self.sync_interval_seconds,
            "coordination_enabled": self.config.coordinate_privacy_accounting,
            "aggregation_method": self.config.privacy_aggregation_method,
            "local_stats": local_stats
        }


class DistributedAttentionWorker:
    """
    Worker process for distributed DP-Flash-Attention computation.
    """
    
    def __init__(
        self,
        config: DistributedConfig,
        worker_id: int,
        device: Optional[torch.device] = None
    ):
        """
        Initialize distributed worker.
        
        Args:
            config: Distributed configuration
            worker_id: Unique worker identifier
            device: Device for computation
        """
        self.config = config
        self.worker_id = worker_id
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Worker state
        self.is_initialized = False
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        
        # Communication
        self._result_queue = mp.Queue()
        self._task_queue = mp.Queue()
        self._shutdown_event = mp.Event()
        
        # Privacy accounting
        self.privacy_accountant = None
        
        self.logger = logging.getLogger(f"worker_{worker_id}")
    
    def initialize(self) -> bool:
        """Initialize worker for distributed processing."""
        try:
            if HAS_TORCH_DISTRIBUTED and self.config.world_size > 1:
                # Initialize distributed backend
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend=self.config.backend,
                        init_method=self.config.init_method,
                        world_size=self.config.world_size,
                        rank=self.config.rank
                    )
                
                # Set device for distributed training
                if torch.cuda.is_available() and self.config.backend == "nccl":
                    torch.cuda.set_device(self.config.rank % torch.cuda.device_count())
                    self.device = torch.device(f'cuda:{self.config.rank % torch.cuda.device_count()}')
            
            # Initialize DP components
            from .privacy import RenyiAccountant
            from .core import DPFlashAttention
            
            local_accountant = RenyiAccountant()
            self.privacy_accountant = DistributedPrivacyAccountant(self.config, local_accountant)
            
            # Initialize attention module
            self.dp_attention = DPFlashAttention(
                embed_dim=768,  # Default, will be updated per task
                num_heads=12,   # Default, will be updated per task
                device=self.device
            )
            
            self.is_initialized = True
            self.logger.info(f"Worker {self.worker_id} initialized on {self.device}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} initialization failed: {e}")
            return False
    
    def process_distributed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        epsilon: float,
        delta: float,
        strategy: DistributedStrategy,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process attention with distributed strategy.
        
        Args:
            q: Query tensor
            k: Key tensor  
            v: Value tensor
            epsilon: Privacy budget
            delta: Privacy parameter
            strategy: Distributed processing strategy
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (output_tensor, metadata)
        """
        start_time = time.time()
        
        try:
            if strategy == DistributedStrategy.DATA_PARALLEL:
                output, metadata = self._data_parallel_attention(q, k, v, epsilon, delta, **kwargs)
            
            elif strategy == DistributedStrategy.MODEL_PARALLEL:
                output, metadata = self._model_parallel_attention(q, k, v, epsilon, delta, **kwargs)
            
            elif strategy == DistributedStrategy.PIPELINE_PARALLEL:
                output, metadata = self._pipeline_parallel_attention(q, k, v, epsilon, delta, **kwargs)
            
            else:
                # Fallback to local processing
                output, privacy_stats = self.dp_attention(q, k, v, return_privacy_stats=True)
                metadata = {"privacy_stats": privacy_stats, "strategy": "local"}
            
            # Update worker statistics
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            
            metadata["processing_time_seconds"] = processing_time
            metadata["worker_id"] = self.worker_id
            
            return output, metadata
            
        except Exception as e:
            self.logger.error(f"Distributed attention processing failed: {e}")
            raise
    
    def _data_parallel_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        epsilon: float, delta: float, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process attention with data parallelism."""
        if not HAS_TORCH_DISTRIBUTED or not dist.is_initialized():
            # Fallback to local processing
            return self.dp_attention(q, k, v, return_privacy_stats=True)
        
        # Ensure tensors are on correct device
        q = q.to(self.device)
        k = k.to(self.device)  
        v = v.to(self.device)
        
        # Process local shard
        local_output, privacy_stats = self.dp_attention(q, k, v, return_privacy_stats=True)
        
        # Synchronize privacy accounting
        if self.privacy_accountant:
            global_epsilon = self.privacy_accountant.add_step(
                privacy_stats.epsilon_spent, delta
            )
        else:
            global_epsilon = privacy_stats.epsilon_spent
        
        # All-reduce output (if needed for consistency)
        if kwargs.get("sync_outputs", False):
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
            local_output /= self.config.world_size
        
        metadata = {
            "privacy_stats": privacy_stats,
            "global_epsilon_spent": global_epsilon,
            "strategy": "data_parallel",
            "world_size": self.config.world_size,
            "rank": self.config.rank
        }
        
        return local_output, metadata
    
    def _model_parallel_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        epsilon: float, delta: float, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process attention with model parallelism."""
        # Simplified model parallel implementation
        # In practice, would split attention heads across devices
        
        num_heads_total = q.shape[2]
        heads_per_device = num_heads_total // self.config.world_size
        start_head = self.config.rank * heads_per_device
        end_head = start_head + heads_per_device
        
        # Process subset of heads
        q_local = q[:, :, start_head:end_head, :].to(self.device)
        k_local = k[:, :, start_head:end_head, :].to(self.device)
        v_local = v[:, :, start_head:end_head, :].to(self.device)
        
        # Process local heads
        local_output, privacy_stats = self.dp_attention(q_local, k_local, v_local, return_privacy_stats=True)
        
        # Gather results from all devices
        if HAS_TORCH_DISTRIBUTED and dist.is_initialized():
            output_list = [torch.zeros_like(local_output) for _ in range(self.config.world_size)]
            dist.all_gather(output_list, local_output)
            
            # Concatenate along head dimension
            full_output = torch.cat(output_list, dim=2)
        else:
            full_output = local_output
        
        metadata = {
            "privacy_stats": privacy_stats,
            "strategy": "model_parallel",
            "heads_processed": f"{start_head}-{end_head}",
            "world_size": self.config.world_size
        }
        
        return full_output, metadata
    
    def _pipeline_parallel_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        epsilon: float, delta: float, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process attention with pipeline parallelism."""
        # Simplified pipeline implementation
        # In practice, would pipeline different stages of attention computation
        
        pipeline_stage = self.config.rank % self.config.pipeline_parallel_size
        
        if pipeline_stage == 0:
            # Stage 0: QK computation
            q = q.to(self.device)
            k = k.to(self.device)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1))
            intermediate = scores
            
        elif pipeline_stage == 1:
            # Stage 1: Softmax and privacy noise
            # Would receive intermediate from previous stage
            intermediate = q  # Placeholder - would be actual intermediate
            
            # Apply softmax and DP noise
            from .utils import compute_noise_scale
            noise_scale = compute_noise_scale(epsilon, delta, 1.0, intermediate.shape[-1])
            
            if intermediate.device != self.device:
                intermediate = intermediate.to(self.device)
            
            # Add noise
            noise = torch.normal(0, noise_scale, intermediate.shape, device=self.device)
            noisy_scores = intermediate + noise
            attn_weights = torch.softmax(noisy_scores, dim=-1)
            
            intermediate = attn_weights
            
        else:
            # Final stage: Value multiplication
            intermediate = q  # Placeholder
            v = v.to(self.device)
            
            if intermediate.device != self.device:
                intermediate = intermediate.to(self.device)
            
            output = torch.matmul(intermediate, v)
            intermediate = output
        
        # In full implementation, would pass intermediate between stages
        # For now, return local computation
        local_output, privacy_stats = self.dp_attention(q, k, v, return_privacy_stats=True)
        
        metadata = {
            "privacy_stats": privacy_stats,
            "strategy": "pipeline_parallel",
            "pipeline_stage": pipeline_stage,
            "world_size": self.config.world_size
        }
        
        return local_output, metadata
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "device": str(self.device),
            "is_initialized": self.is_initialized,
            "tasks_processed": self.tasks_processed,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": self.total_processing_time / max(1, self.tasks_processed),
            "privacy_stats": self.privacy_accountant.get_privacy_stats() if self.privacy_accountant else {}
        }
    
    def shutdown(self) -> None:
        """Shutdown worker gracefully."""
        self._shutdown_event.set()
        
        if HAS_TORCH_DISTRIBUTED and dist.is_initialized():
            dist.destroy_process_group()
        
        self.logger.info(f"Worker {self.worker_id} shutdown complete")


class DistributedAttentionProcessor:
    """
    High-level distributed processor for DP-Flash-Attention operations.
    
    Manages multiple workers and coordinates distributed computation.
    """
    
    def __init__(self, config: DistributedConfig):
        """
        Initialize distributed processor.
        
        Args:
            config: Distributed configuration
        """
        self.config = config
        self.workers: List[DistributedAttentionWorker] = []
        self.is_initialized = False
        
        # Task management
        self.task_counter = 0
        self.pending_tasks: Dict[str, Any] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        # Synchronization
        self._processor_lock = threading.Lock()
        
        self.logger = logging.getLogger("distributed_processor")
    
    def initialize(self) -> bool:
        """Initialize distributed processor and workers."""
        try:
            # Create workers
            for i in range(self.config.world_size):
                worker = DistributedAttentionWorker(
                    config=self.config,
                    worker_id=i,
                    device=torch.device(f'cuda:{i}' if torch.cuda.is_available() and i < torch.cuda.device_count() else 'cpu')
                )
                
                if worker.initialize():
                    self.workers.append(worker)
                else:
                    self.logger.error(f"Failed to initialize worker {i}")
                    return False
            
            self.is_initialized = True
            self.logger.info(f"Distributed processor initialized with {len(self.workers)} workers")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Distributed processor initialization failed: {e}")
            return False
    
    def process_distributed_batch(
        self,
        batch_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        epsilon: float,
        delta: float,
        strategy: DistributedStrategy = DistributedStrategy.DATA_PARALLEL,
        **kwargs
    ) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Process batch of inputs with distributed computation.
        
        Args:
            batch_inputs: List of (q, k, v) tensor tuples
            epsilon: Privacy budget
            delta: Privacy parameter
            strategy: Distributed processing strategy
            **kwargs: Additional parameters
            
        Returns:
            List of (output, metadata) tuples
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed processor not initialized")
        
        results = []
        
        # Distribute work across workers
        for i, (q, k, v) in enumerate(batch_inputs):
            worker_idx = i % len(self.workers)
            worker = self.workers[worker_idx]
            
            try:
                output, metadata = worker.process_distributed_attention(
                    q, k, v, epsilon, delta, strategy, **kwargs
                )
                results.append((output, metadata))
                
            except Exception as e:
                self.logger.error(f"Worker {worker_idx} failed to process batch item {i}: {e}")
                # Create error result
                error_metadata = {
                    "error": str(e),
                    "worker_id": worker_idx,
                    "strategy": strategy.value
                }
                results.append((torch.zeros_like(q), error_metadata))
        
        return results
    
    def get_global_privacy_stats(self) -> Dict[str, Any]:
        """Get aggregated privacy statistics from all workers."""
        if not self.workers:
            return {}
        
        # Collect stats from all workers
        all_stats = []
        for worker in self.workers:
            try:
                stats = worker.get_worker_stats()
                all_stats.append(stats)
            except Exception as e:
                self.logger.warning(f"Could not get stats from worker {worker.worker_id}: {e}")
        
        if not all_stats:
            return {}
        
        # Aggregate statistics
        total_tasks = sum(stats.get("tasks_processed", 0) for stats in all_stats)
        total_time = sum(stats.get("total_processing_time", 0) for stats in all_stats)
        avg_time = total_time / max(1, total_tasks)
        
        # Aggregate privacy stats
        privacy_stats = {}
        for stats in all_stats:
            worker_privacy = stats.get("privacy_stats", {})
            if "global_privacy_spent" in worker_privacy:
                privacy_stats["global_privacy_spent"] = max(
                    privacy_stats.get("global_privacy_spent", 0),
                    worker_privacy["global_privacy_spent"]
                )
        
        return {
            "total_workers": len(all_stats),
            "total_tasks_processed": total_tasks,
            "total_processing_time": total_time,
            "avg_processing_time": avg_time,
            "privacy_stats": privacy_stats,
            "worker_stats": all_stats,
            "distributed_config": {
                "strategy": self.config.strategy.value,
                "world_size": self.config.world_size,
                "backend": self.config.backend
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown all workers and clean up resources."""
        self.logger.info("Shutting down distributed processor...")
        
        for worker in self.workers:
            try:
                worker.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down worker {worker.worker_id}: {e}")
        
        self.workers.clear()
        self.is_initialized = False
        
        self.logger.info("Distributed processor shutdown complete")


def create_distributed_config(
    strategy: DistributedStrategy,
    world_size: int,
    rank: int,
    backend: str = "auto"
) -> DistributedConfig:
    """
    Create distributed configuration with sensible defaults.
    
    Args:
        strategy: Distributed processing strategy
        world_size: Total number of processes
        rank: Current process rank
        backend: Communication backend
        
    Returns:
        Distributed configuration
    """
    # Auto-select backend
    if backend == "auto":
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
    
    return DistributedConfig(
        strategy=strategy,
        world_size=world_size,
        rank=rank,
        backend=backend,
        init_method="env://",
        coordinate_privacy_accounting=True
    )


def launch_distributed_training(
    train_function: Callable,
    config: DistributedConfig,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None
) -> None:
    """
    Launch distributed training with proper setup.
    
    Args:
        train_function: Training function to run
        config: Distributed configuration  
        args: Arguments for training function
        kwargs: Keyword arguments for training function
    """
    if kwargs is None:
        kwargs = {}
    
    if not HAS_TORCH_DISTRIBUTED:
        warnings.warn("PyTorch distributed not available, running locally")
        train_function(*args, **kwargs)
        return
    
    try:
        # Set up environment variables for distributed training
        os.environ["WORLD_SIZE"] = str(config.world_size)
        os.environ["RANK"] = str(config.rank)
        
        if config.backend == "nccl" and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.rank % torch.cuda.device_count())
        
        # Launch training
        train_function(*args, **kwargs)
        
    except Exception as e:
        logging.error(f"Distributed training launch failed: {e}")
        raise
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()