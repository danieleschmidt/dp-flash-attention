"""
Auto-scaling and load balancing for DP-Flash-Attention.

Provides dynamic resource allocation, load balancing, and auto-scaling
based on workload patterns and system metrics.
"""

import time
import threading
import math
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import torch
import psutil

from .monitoring import DPTelemetry
from .concurrent import ConcurrentAttentionProcessor, TaskPriority
from .validation import validate_tensor_memory


class ScalingAction(Enum):
    """Auto-scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    REBALANCE = "rebalance"


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    queue_depth: int
    avg_response_time_ms: float
    throughput_ops_per_sec: float
    error_rate: float
    timestamp: float


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    # Scale-up thresholds
    cpu_scale_up_threshold: float = 80.0
    memory_scale_up_threshold: float = 75.0  
    gpu_scale_up_threshold: float = 85.0
    queue_depth_scale_up_threshold: int = 50
    response_time_scale_up_threshold_ms: float = 1000.0
    
    # Scale-down thresholds
    cpu_scale_down_threshold: float = 30.0
    memory_scale_down_threshold: float = 40.0
    gpu_scale_down_threshold: float = 25.0
    queue_depth_scale_down_threshold: int = 5
    response_time_scale_down_threshold_ms: float = 200.0
    
    # Scaling limits
    min_workers: int = 1
    max_workers: int = 16
    
    # Scaling behavior
    scale_up_cooldown_seconds: float = 300.0  # 5 minutes
    scale_down_cooldown_seconds: float = 600.0  # 10 minutes
    evaluation_interval_seconds: float = 60.0  # 1 minute
    consecutive_evaluations_required: int = 3
    
    # Load balancing
    enable_load_balancing: bool = True
    rebalance_threshold: float = 0.3  # 30% imbalance triggers rebalancing


class WorkloadPredictor:
    """
    Predicts workload patterns for proactive scaling.
    
    Uses historical data to predict future resource needs.
    """
    
    def __init__(self, history_length: int = 1440):  # 24 hours of minute-by-minute data
        """Initialize workload predictor."""
        self.history_length = history_length
        self.metrics_history: List[ScalingMetrics] = []
        self.prediction_cache: Dict[str, Tuple[float, float]] = {}  # key -> (prediction, timestamp)
        self._lock = threading.Lock()
    
    def record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record metrics for prediction."""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Keep only recent history
            if len(self.metrics_history) > self.history_length:
                self.metrics_history = self.metrics_history[-self.history_length:]
    
    def predict_workload(self, horizon_minutes: int = 15) -> Dict[str, float]:
        """
        Predict workload metrics for given time horizon.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Dictionary with predicted metrics
        """
        with self._lock:
            if len(self.metrics_history) < 10:
                # Not enough data for prediction, return current metrics
                if self.metrics_history:
                    latest = self.metrics_history[-1]
                    return {
                        'cpu_utilization': latest.cpu_utilization,
                        'memory_utilization': latest.memory_utilization,
                        'gpu_utilization': latest.gpu_utilization,
                        'queue_depth': latest.queue_depth,
                        'throughput_ops_per_sec': latest.throughput_ops_per_sec
                    }
                else:
                    # No data, return conservative estimates
                    return {
                        'cpu_utilization': 50.0,
                        'memory_utilization': 50.0,
                        'gpu_utilization': 50.0,
                        'queue_depth': 10,
                        'throughput_ops_per_sec': 10.0
                    }
            
            # Simple moving average prediction with trend analysis
            recent_metrics = self.metrics_history[-min(60, len(self.metrics_history)):]  # Last hour
            
            predictions = {}
            
            # Predict each metric
            for metric_name in ['cpu_utilization', 'memory_utilization', 'gpu_utilization', 'queue_depth', 'throughput_ops_per_sec']:
                values = [getattr(m, metric_name) for m in recent_metrics]
                
                # Calculate moving average
                avg = sum(values) / len(values)
                
                # Calculate trend (simple linear regression slope)
                n = len(values)
                sum_x = sum(range(n))
                sum_y = sum(values)
                sum_xy = sum(i * values[i] for i in range(n))
                sum_x2 = sum(i * i for i in range(n))
                
                if n * sum_x2 - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                else:
                    slope = 0
                
                # Project forward
                prediction = avg + slope * horizon_minutes
                
                # Apply bounds based on metric type
                if metric_name.endswith('_utilization'):
                    prediction = max(0, min(100, prediction))
                elif metric_name == 'queue_depth':
                    prediction = max(0, prediction)
                elif metric_name == 'throughput_ops_per_sec':
                    prediction = max(0, prediction)
                
                predictions[metric_name] = prediction
            
            return predictions


class LoadBalancer:
    """
    Load balancer for distributing work across multiple workers and devices.
    
    Implements various load balancing strategies and monitors worker health.
    """
    
    def __init__(self):
        """Initialize load balancer."""
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.device_assignments: Dict[str, torch.device] = {}
        self.task_distribution: Dict[str, int] = {}  # worker_id -> task_count
        self._lock = threading.Lock()
    
    def register_worker(self, worker_id: str, device: torch.device) -> None:
        """Register a worker with the load balancer."""
        with self._lock:
            self.worker_stats[worker_id] = {
                'device': device,
                'tasks_assigned': 0,
                'tasks_completed': 0,
                'avg_processing_time': 0.0,
                'last_activity': time.time(),
                'health_score': 1.0
            }
            self.device_assignments[worker_id] = device
            self.task_distribution[worker_id] = 0
    
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker from the load balancer."""
        with self._lock:
            self.worker_stats.pop(worker_id, None)
            self.device_assignments.pop(worker_id, None)
            self.task_distribution.pop(worker_id, None)
    
    def select_worker(self, task_size_estimate: float = 1.0) -> Optional[str]:
        """
        Select best worker for task assignment.
        
        Args:
            task_size_estimate: Relative size estimate of the task
            
        Returns:
            Selected worker ID or None if no workers available
        """
        with self._lock:
            if not self.worker_stats:
                return None
            
            # Calculate scores for each worker
            worker_scores = {}
            
            for worker_id, stats in self.worker_stats.items():
                # Base score from health
                score = stats['health_score']
                
                # Penalize high current load
                current_load = self.task_distribution[worker_id]
                score *= 1.0 / (1.0 + current_load * 0.1)
                
                # Prefer faster workers
                if stats['avg_processing_time'] > 0:
                    score *= 1.0 / (1.0 + stats['avg_processing_time'] * 0.001)
                
                # Penalize inactive workers (may be stuck)
                time_since_activity = time.time() - stats['last_activity']
                if time_since_activity > 300:  # 5 minutes
                    score *= 0.5
                
                worker_scores[worker_id] = score
            
            # Select worker with highest score
            best_worker = max(worker_scores.keys(), key=lambda w: worker_scores[w])
            
            # Update assignment count
            self.task_distribution[best_worker] += 1
            self.worker_stats[best_worker]['tasks_assigned'] += 1
            
            return best_worker
    
    def report_task_completion(self, worker_id: str, processing_time: float, success: bool) -> None:
        """Report task completion for load balancing metrics."""
        with self._lock:
            if worker_id in self.worker_stats:
                stats = self.worker_stats[worker_id]
                
                # Update task count
                self.task_distribution[worker_id] = max(0, self.task_distribution[worker_id] - 1)
                stats['last_activity'] = time.time()
                
                if success:
                    stats['tasks_completed'] += 1
                    
                    # Update average processing time (exponential moving average)
                    alpha = 0.1
                    if stats['avg_processing_time'] == 0:
                        stats['avg_processing_time'] = processing_time
                    else:
                        stats['avg_processing_time'] = (1 - alpha) * stats['avg_processing_time'] + alpha * processing_time
                    
                    # Update health score (success increases health)
                    stats['health_score'] = min(1.0, stats['health_score'] * 1.01)
                else:
                    # Failure decreases health score
                    stats['health_score'] *= 0.95
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            if not self.worker_stats:
                return {'workers': 0, 'balance_score': 1.0}
            
            # Calculate load imbalance
            task_counts = list(self.task_distribution.values())
            if task_counts:
                avg_load = sum(task_counts) / len(task_counts)
                max_load = max(task_counts)
                min_load = min(task_counts)
                
                if avg_load > 0:
                    imbalance = (max_load - min_load) / avg_load
                else:
                    imbalance = 0.0
                
                balance_score = max(0.0, 1.0 - imbalance)
            else:
                balance_score = 1.0
            
            return {
                'workers': len(self.worker_stats),
                'balance_score': balance_score,
                'task_distribution': self.task_distribution.copy(),
                'worker_health': {wid: stats['health_score'] for wid, stats in self.worker_stats.items()}
            }


class AutoScaler:
    """
    Auto-scaler for DP-Flash-Attention processing.
    
    Monitors system metrics and automatically adjusts resource allocation
    based on workload patterns and scaling policies.
    """
    
    def __init__(
        self,
        processor: ConcurrentAttentionProcessor,
        policy: Optional[ScalingPolicy] = None,
        telemetry: Optional[DPTelemetry] = None
    ):
        """
        Initialize auto-scaler.
        
        Args:
            processor: Concurrent processor to scale
            policy: Scaling policy (uses default if None)
            telemetry: Optional telemetry instance
        """
        self.processor = processor
        self.policy = policy or ScalingPolicy()
        self.telemetry = telemetry
        
        # Components
        self.predictor = WorkloadPredictor()
        self.load_balancer = LoadBalancer()
        
        # State
        self.current_workers = processor.max_workers
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.consecutive_scale_up_signals = 0
        self.consecutive_scale_down_signals = 0
        
        # Monitoring
        self.scaling_history: List[Tuple[float, ScalingAction, int]] = []  # (timestamp, action, worker_count)
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger("autoscaler")
    
    def start(self) -> None:
        """Start auto-scaling monitoring."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            
            self.logger.info("Auto-scaler started")
    
    def stop(self) -> None:
        """Stop auto-scaling monitoring."""
        with self._lock:
            self._running = False
            
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
            
        self.logger.info("Auto-scaler stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Record for prediction
                self.predictor.record_metrics(metrics)
                
                # Evaluate scaling decision
                action = self._evaluate_scaling(metrics)
                
                # Execute scaling action
                if action != ScalingAction.MAINTAIN:
                    self._execute_scaling_action(action, metrics)
                
                # Record telemetry
                if self.telemetry:
                    self.telemetry.record_performance_metrics(
                        operation_type='autoscaling_evaluation',
                        duration_ms=0,  # Instantaneous
                        memory_used_mb=metrics.memory_utilization,
                        gpu_utilization=metrics.gpu_utilization / 100.0,
                        batch_size=self.current_workers
                    )
                
                # Sleep until next evaluation
                time.sleep(self.policy.evaluation_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Auto-scaling monitoring error: {e}")
                time.sleep(30)  # Back off on error
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU metrics
        gpu_percent = 0.0
        if torch.cuda.is_available():
            try:
                # Simplified GPU utilization (would need nvidia-ml-py for accurate metrics)
                gpu_percent = torch.cuda.memory_allocated(0) / torch.cuda.max_memory_allocated(0) * 100
            except:
                gpu_percent = 50.0  # Default estimate
        
        # Processor metrics
        processor_stats = self.processor.get_stats()
        queue_depth = processor_stats.get('queue_size', 0)
        
        # Calculate average response time from recent completions
        avg_response_time = 0.0
        total_completed = processor_stats.get('total_tasks_completed', 0)
        if total_completed > 0 and 'workers' in processor_stats:
            worker_stats = processor_stats['workers'].values()
            processing_times = [w.get('avg_processing_time', 0) for w in worker_stats if w.get('avg_processing_time', 0) > 0]
            if processing_times:
                avg_response_time = sum(processing_times) / len(processing_times) * 1000  # Convert to ms
        
        # Calculate throughput
        throughput = 0.0
        if hasattr(self, '_last_completed_count') and hasattr(self, '_last_metrics_time'):
            time_delta = time.time() - self._last_metrics_time
            completed_delta = total_completed - self._last_completed_count
            if time_delta > 0:
                throughput = completed_delta / time_delta
        
        self._last_completed_count = total_completed
        self._last_metrics_time = time.time()
        
        # Error rate
        total_failed = processor_stats.get('total_tasks_failed', 0)
        total_tasks = total_completed + total_failed
        error_rate = (total_failed / total_tasks * 100) if total_tasks > 0 else 0.0
        
        return ScalingMetrics(
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            gpu_utilization=gpu_percent,
            queue_depth=queue_depth,
            avg_response_time_ms=avg_response_time,
            throughput_ops_per_sec=throughput,
            error_rate=error_rate,
            timestamp=time.time()
        )
    
    def _evaluate_scaling(self, metrics: ScalingMetrics) -> ScalingAction:
        """Evaluate whether scaling action is needed."""
        current_time = time.time()
        
        # Check cooldown periods
        scale_up_ready = (current_time - self.last_scale_up) >= self.policy.scale_up_cooldown_seconds
        scale_down_ready = (current_time - self.last_scale_down) >= self.policy.scale_down_cooldown_seconds
        
        # Evaluate scale-up conditions
        scale_up_signals = 0
        if metrics.cpu_utilization > self.policy.cpu_scale_up_threshold:
            scale_up_signals += 1
        if metrics.memory_utilization > self.policy.memory_scale_up_threshold:
            scale_up_signals += 1
        if metrics.gpu_utilization > self.policy.gpu_scale_up_threshold:
            scale_up_signals += 1
        if metrics.queue_depth > self.policy.queue_depth_scale_up_threshold:
            scale_up_signals += 1
        if metrics.avg_response_time_ms > self.policy.response_time_scale_up_threshold_ms:
            scale_up_signals += 1
        
        # Evaluate scale-down conditions
        scale_down_signals = 0
        if metrics.cpu_utilization < self.policy.cpu_scale_down_threshold:
            scale_down_signals += 1
        if metrics.memory_utilization < self.policy.memory_scale_down_threshold:
            scale_down_signals += 1
        if metrics.gpu_utilization < self.policy.gpu_scale_down_threshold:
            scale_down_signals += 1
        if metrics.queue_depth < self.policy.queue_depth_scale_down_threshold:
            scale_down_signals += 1
        if metrics.avg_response_time_ms < self.policy.response_time_scale_down_threshold_ms:
            scale_down_signals += 1
        
        # Update consecutive signal counters
        if scale_up_signals >= 2:  # At least 2 signals needed
            self.consecutive_scale_up_signals += 1
            self.consecutive_scale_down_signals = 0
        elif scale_down_signals >= 3:  # At least 3 signals needed (more conservative)
            self.consecutive_scale_down_signals += 1
            self.consecutive_scale_up_signals = 0
        else:
            self.consecutive_scale_up_signals = 0
            self.consecutive_scale_down_signals = 0
        
        # Make scaling decision
        action = ScalingAction.MAINTAIN
        
        if (scale_up_ready and 
            self.consecutive_scale_up_signals >= self.policy.consecutive_evaluations_required and
            self.current_workers < self.policy.max_workers):
            action = ScalingAction.SCALE_UP
        elif (scale_down_ready and
              self.consecutive_scale_down_signals >= self.policy.consecutive_evaluations_required and
              self.current_workers > self.policy.min_workers):
            action = ScalingAction.SCALE_DOWN
        elif self.policy.enable_load_balancing:
            # Check if rebalancing is needed
            balance_stats = self.load_balancer.get_load_balance_stats()
            if balance_stats['balance_score'] < (1.0 - self.policy.rebalance_threshold):
                action = ScalingAction.REBALANCE
        
        return action
    
    def _execute_scaling_action(self, action: ScalingAction, metrics: ScalingMetrics) -> None:
        """Execute scaling action."""
        current_time = time.time()
        
        if action == ScalingAction.SCALE_UP:
            new_worker_count = min(self.policy.max_workers, self.current_workers + 1)
            if new_worker_count > self.current_workers:
                self._scale_workers(new_worker_count)
                self.last_scale_up = current_time
                self.consecutive_scale_up_signals = 0
                
                self.logger.info(f"Scaled up to {new_worker_count} workers")
        
        elif action == ScalingAction.SCALE_DOWN:
            new_worker_count = max(self.policy.min_workers, self.current_workers - 1)
            if new_worker_count < self.current_workers:
                self._scale_workers(new_worker_count)
                self.last_scale_down = current_time
                self.consecutive_scale_down_signals = 0
                
                self.logger.info(f"Scaled down to {new_worker_count} workers")
        
        elif action == ScalingAction.REBALANCE:
            self._rebalance_workers()
            self.logger.info("Rebalanced worker load distribution")
        
        # Record scaling history
        self.scaling_history.append((current_time, action, self.current_workers))
        
        # Keep only recent history
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]
    
    def _scale_workers(self, target_count: int) -> None:
        """Scale worker count to target."""
        # This is a simplified implementation
        # In practice, would need to coordinate with the processor
        self.current_workers = target_count
        
        # Update processor configuration (conceptual)
        # processor.update_worker_count(target_count)
    
    def _rebalance_workers(self) -> None:
        """Rebalance work distribution across workers."""
        # Implementation would redistribute pending tasks
        # and update load balancer assignments
        pass
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            'current_workers': self.current_workers,
            'scaling_policy': {
                'min_workers': self.policy.min_workers,
                'max_workers': self.policy.max_workers,
                'scale_up_cooldown': self.policy.scale_up_cooldown_seconds,
                'scale_down_cooldown': self.policy.scale_down_cooldown_seconds
            },
            'consecutive_signals': {
                'scale_up': self.consecutive_scale_up_signals,
                'scale_down': self.consecutive_scale_down_signals
            },
            'last_scaling': {
                'scale_up': self.last_scale_up,
                'scale_down': self.last_scale_down
            },
            'scaling_history_count': len(self.scaling_history),
            'load_balancer': self.load_balancer.get_load_balance_stats(),
            'is_running': self._running
        }
    
    def get_predictions(self, horizon_minutes: int = 15) -> Dict[str, Any]:
        """Get workload predictions."""
        predictions = self.predictor.predict_workload(horizon_minutes)
        return {
            'horizon_minutes': horizon_minutes,
            'predictions': predictions,
            'history_length': len(self.predictor.metrics_history)
        }