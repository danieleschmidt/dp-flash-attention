"""
Concurrent processing and resource pooling for DP-Flash-Attention.

Provides thread-safe parallel processing, resource management, and
load balancing for high-throughput differential privacy operations.
"""

import time
import threading
import concurrent.futures
import queue
import weakref
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

import torch
from torch import Tensor
import torch.multiprocessing as mp

from .monitoring import DPTelemetry
from .optimization import AttentionOptimizer
from .security import SecureRandomGenerator


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2  
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Task for concurrent processing."""
    id: str
    priority: TaskPriority
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    submit_time: float
    timeout: Optional[float] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any
    error: Optional[Exception]
    start_time: float
    end_time: float
    worker_id: str


class ResourcePool:
    """
    Thread-safe resource pool for GPU devices and computation resources.
    
    Manages allocation and deallocation of computational resources
    with automatic load balancing.
    """
    
    def __init__(self, max_concurrent_tasks: int = 4):
        """
        Initialize resource pool.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks per device
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.device_pools: Dict[torch.device, List[bool]] = {}
        self.device_locks: Dict[torch.device, threading.Semaphore] = {}
        self.active_tasks: Dict[torch.device, int] = {}
        self._pool_lock = threading.Lock()
        
        # Initialize GPU devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                self.device_pools[device] = [True] * max_concurrent_tasks
                self.device_locks[device] = threading.Semaphore(max_concurrent_tasks)
                self.active_tasks[device] = 0
        
        # CPU fallback
        cpu_device = torch.device('cpu')
        self.device_pools[cpu_device] = [True] * max_concurrent_tasks
        self.device_locks[cpu_device] = threading.Semaphore(max_concurrent_tasks)
        self.active_tasks[cpu_device] = 0
    
    def acquire_resource(self, preferred_device: Optional[torch.device] = None, timeout: float = 30.0) -> Optional[torch.device]:
        """
        Acquire computational resource.
        
        Args:
            preferred_device: Preferred device (None for auto-selection)
            timeout: Acquisition timeout in seconds
            
        Returns:
            Acquired device or None if timeout
        """
        start_time = time.time()
        
        # Determine candidate devices
        if preferred_device and preferred_device in self.device_pools:
            candidates = [preferred_device]
        else:
            # Sort devices by availability
            candidates = sorted(
                self.device_pools.keys(),
                key=lambda d: self.active_tasks[d]
            )
        
        for device in candidates:
            # Try to acquire with timeout
            remaining_timeout = timeout - (time.time() - start_time)
            if remaining_timeout <= 0:
                break
            
            if self.device_locks[device].acquire(timeout=remaining_timeout):
                with self._pool_lock:
                    self.active_tasks[device] += 1
                return device
        
        return None
    
    def release_resource(self, device: torch.device) -> None:
        """
        Release computational resource.
        
        Args:
            device: Device to release
        """
        if device in self.device_locks:
            with self._pool_lock:
                self.active_tasks[device] = max(0, self.active_tasks[device] - 1)
            self.device_locks[device].release()
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        with self._pool_lock:
            return {
                'devices': {
                    str(device): {
                        'max_concurrent': self.max_concurrent_tasks,
                        'active_tasks': self.active_tasks[device],
                        'utilization': self.active_tasks[device] / self.max_concurrent_tasks
                    }
                    for device in self.device_pools.keys()
                },
                'total_devices': len(self.device_pools),
                'total_active_tasks': sum(self.active_tasks.values())
            }


class WorkerThread:
    """
    Worker thread for processing attention computation tasks.
    
    Each worker maintains its own optimizer and secure RNG for
    thread-safe operation.
    """
    
    def __init__(self, worker_id: str, device: torch.device, telemetry: Optional[DPTelemetry] = None):
        """
        Initialize worker thread.
        
        Args:
            worker_id: Unique worker identifier
            device: Device for computations
            telemetry: Optional telemetry instance
        """
        self.worker_id = worker_id
        self.device = device
        self.telemetry = telemetry
        
        # Initialize worker-specific resources
        self.optimizer = AttentionOptimizer(device=device, enable_telemetry=False)
        self.secure_rng = SecureRandomGenerator()
        
        # Worker statistics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        self.last_activity = time.time()
        
        # Logger
        self.logger = logging.getLogger(f"worker.{worker_id}")
    
    def process_task(self, task: Task) -> TaskResult:
        """
        Process a single task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        start_time = time.time()
        self.last_activity = start_time
        
        try:
            # Set device context if needed
            if hasattr(torch.cuda, 'set_device') and self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
            
            # Execute task
            result = task.function(*task.args, **task.kwargs)
            
            # Update statistics
            end_time = time.time()
            processing_time = end_time - start_time
            self.tasks_completed += 1
            self.total_processing_time += processing_time
            
            # Record telemetry
            if self.telemetry:
                self.telemetry.record_performance_metrics(
                    operation_type='worker_task',
                    duration_ms=processing_time * 1000,
                    batch_size=getattr(task.args[0], 'shape', [0])[0] if task.args else 0
                )
            
            return TaskResult(
                task_id=task.id,
                success=True,
                result=result,
                error=None,
                start_time=start_time,
                end_time=end_time,
                worker_id=self.worker_id
            )
            
        except Exception as e:
            end_time = time.time()
            self.tasks_failed += 1
            
            self.logger.error(f"Task {task.id} failed: {e}")
            
            return TaskResult(
                task_id=task.id,
                success=False,
                result=None,
                error=e,
                start_time=start_time,
                end_time=end_time,
                worker_id=self.worker_id
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'device': str(self.device),
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.total_processing_time / max(1, self.tasks_completed),
            'last_activity': self.last_activity
        }


class ConcurrentAttentionProcessor:
    """
    High-level concurrent processor for DP-Flash-Attention operations.
    
    Manages task queuing, worker allocation, and result aggregation
    with automatic load balancing and error handling.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        enable_telemetry: bool = True
    ):
        """
        Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum task queue size
            enable_telemetry: Whether to enable telemetry
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Initialize components
        self.resource_pool = ResourcePool(max_concurrent_tasks=2)
        self.telemetry = DPTelemetry() if enable_telemetry else None
        
        # Task management
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.pending_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
        # Worker management
        self.workers: Dict[str, WorkerThread] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.worker_futures: Dict[str, concurrent.futures.Future] = {}
        
        # Synchronization
        self._processor_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Statistics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        # Start processing
        self._start_processing()
    
    def submit_task(
        self,
        function: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        timeout: Optional[float] = None,
        dependencies: List[str] = None,
        task_id: Optional[str] = None
    ) -> str:
        """
        Submit task for concurrent processing.
        
        Args:
            function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority
            timeout: Task timeout
            dependencies: List of task IDs this task depends on
            task_id: Optional custom task ID
            
        Returns:
            Task ID
        """
        if kwargs is None:
            kwargs = {}
        
        if dependencies is None:
            dependencies = []
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        task = Task(
            id=task_id,
            priority=priority,
            function=function,
            args=args,
            kwargs=kwargs,
            submit_time=time.time(),
            timeout=timeout,
            dependencies=dependencies
        )
        
        with self._processor_lock:
            # Check if dependencies are satisfied
            if self._check_dependencies(dependencies):
                # Add to queue immediately
                self.task_queue.put((priority.value, time.time(), task))
            else:
                # Store as pending
                self.pending_tasks[task_id] = task
                self.task_dependencies[task_id] = dependencies
            
            self.total_tasks_submitted += 1
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Get task result.
        
        Args:
            task_id: Task ID to get result for
            timeout: Maximum wait time
            
        Returns:
            Task result or None if timeout
        """
        start_time = time.time()
        
        while True:
            with self._processor_lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return None
            
            time.sleep(0.01)  # Small sleep to avoid busy waiting
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Maximum wait time
            
        Returns:
            Dictionary of task results
        """
        start_time = time.time()
        results = {}
        
        while len(results) < len(task_ids):
            with self._processor_lock:
                for task_id in task_ids:
                    if task_id in self.completed_tasks and task_id not in results:
                        results[task_id] = self.completed_tasks[task_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                break
            
            time.sleep(0.01)
        
        return results
    
    def _start_processing(self) -> None:
        """Start background processing threads."""
        for i in range(self.max_workers):
            worker_id = f"worker_{i}"
            future = self.executor.submit(self._worker_loop, worker_id)
            self.worker_futures[worker_id] = future
    
    def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop."""
        worker = None
        current_device = None
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get task from queue
                    priority, submit_time, task = self.task_queue.get(timeout=1.0)
                    
                    # Acquire resource
                    device = self.resource_pool.acquire_resource(timeout=30.0)
                    if device is None:
                        # Put task back in queue if no resources available
                        self.task_queue.put((priority, submit_time, task))
                        continue
                    
                    # Create worker if needed or device changed
                    if worker is None or current_device != device:
                        worker = WorkerThread(worker_id, device, self.telemetry)
                        current_device = device
                        self.workers[worker_id] = worker
                    
                    # Process task
                    result = worker.process_task(task)
                    
                    # Store result
                    with self._processor_lock:
                        self.completed_tasks[task.id] = result
                        if result.success:
                            self.total_tasks_completed += 1
                        else:
                            self.total_tasks_failed += 1
                        
                        # Check for dependent tasks
                        self._check_pending_tasks(task.id)
                    
                    # Release resource
                    self.resource_pool.release_resource(device)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Worker {worker_id} error: {e}")
                    if current_device:
                        self.resource_pool.release_resource(current_device)
                    continue
                
        except Exception as e:
            logging.error(f"Worker {worker_id} crashed: {e}")
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep_id in self.completed_tasks for dep_id in dependencies)
    
    def _check_pending_tasks(self, completed_task_id: str) -> None:
        """Check if any pending tasks can now be queued."""
        to_queue = []
        
        for task_id, task in list(self.pending_tasks.items()):
            if completed_task_id in task.dependencies:
                # Update dependencies
                task.dependencies.remove(completed_task_id)
            
            # Check if all dependencies are now satisfied
            if self._check_dependencies(task.dependencies):
                to_queue.append(task_id)
        
        # Queue ready tasks
        for task_id in to_queue:
            task = self.pending_tasks.pop(task_id)
            self.task_queue.put((task.priority.value, task.submit_time, task))
            del self.task_dependencies[task_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        with self._processor_lock:
            worker_stats = {wid: worker.get_stats() for wid, worker in self.workers.items()}
            
            return {
                'total_tasks_submitted': self.total_tasks_submitted,
                'total_tasks_completed': self.total_tasks_completed,
                'total_tasks_failed': self.total_tasks_failed,
                'pending_tasks': len(self.pending_tasks),
                'queue_size': self.task_queue.qsize(),
                'max_queue_size': self.max_queue_size,
                'workers': worker_stats,
                'resource_pool': self.resource_pool.get_resource_stats()
            }
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Shutdown processor gracefully.
        
        Args:
            wait: Whether to wait for completion
            timeout: Shutdown timeout
        """
        self._shutdown_event.set()
        
        if wait:
            self.executor.shutdown(wait=True, timeout=timeout)


def parallel_attention_batch(
    batch_inputs: List[Tuple[Tensor, Tensor, Tensor]],
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    max_workers: int = 4,
    **kwargs
) -> List[Tuple[Tensor, float]]:
    """
    Process batch of attention computations in parallel.
    
    Args:
        batch_inputs: List of (q, k, v) tensor tuples
        epsilon: Privacy budget
        delta: Privacy parameter
        max_grad_norm: Gradient clipping norm
        max_workers: Maximum parallel workers
        
    Returns:
        List of (output, grad_norm) tuples
    """
    processor = ConcurrentAttentionProcessor(max_workers=max_workers)
    
    try:
        # Submit all tasks
        task_ids = []
        for i, (q, k, v) in enumerate(batch_inputs):
            from .functional import dp_flash_attn_func
            
            task_id = processor.submit_task(
                function=dp_flash_attn_func,
                args=(q, k, v),
                kwargs={
                    'epsilon': epsilon,
                    'delta': delta,
                    'max_grad_norm': max_grad_norm,
                    **kwargs
                },
                task_id=f"batch_attention_{i}"
            )
            task_ids.append(task_id)
        
        # Wait for completion
        results = processor.wait_for_completion(task_ids, timeout=300.0)
        
        # Extract outputs
        outputs = []
        for task_id in task_ids:
            if task_id in results and results[task_id].success:
                outputs.append(results[task_id].result)
            else:
                # Handle failed tasks
                error = results.get(task_id, TaskResult(task_id, False, None, Exception("Unknown error"), 0, 0, "")).error
                raise RuntimeError(f"Task {task_id} failed: {error}")
        
        return outputs
        
    finally:
        processor.shutdown(wait=True, timeout=10.0)


# Global processor instance
_global_processor: Optional[ConcurrentAttentionProcessor] = None
_processor_lock = threading.Lock()


def get_global_processor() -> ConcurrentAttentionProcessor:
    """Get or create global concurrent processor."""
    global _global_processor
    
    if _global_processor is None:
        with _processor_lock:
            if _global_processor is None:
                _global_processor = ConcurrentAttentionProcessor()
    
    return _global_processor


def shutdown_global_processor() -> None:
    """Shutdown global processor."""
    global _global_processor
    
    if _global_processor is not None:
        with _processor_lock:
            if _global_processor is not None:
                _global_processor.shutdown()
                _global_processor = None