"""
Monitoring and observability utilities for DP-Flash-Attention.

This module provides telemetry, metrics collection, and privacy-aware monitoring
capabilities for differential privacy operations.
"""

from typing import Dict, Any, Optional, Union, List
import time
import logging
import json
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from functools import wraps
import threading
import weakref

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


@dataclass
class PrivacyMetrics:
    """Privacy-related metrics for a single operation."""
    epsilon: float
    delta: float
    noise_scale: float
    gradient_norm: float
    clipping_applied: bool
    timestamp: float
    operation_id: str


@dataclass
class PerformanceMetrics:
    """Performance metrics for attention operations."""
    duration_ms: float
    memory_used_mb: float
    gpu_utilization: float
    batch_size: int
    sequence_length: int
    num_heads: int
    timestamp: float
    operation_id: str


class PrivacyBudgetTracker:
    """Tracks privacy budget consumption across operations."""
    
    def __init__(self, initial_epsilon: float = 0.0, initial_delta: float = 0.0):
        self._epsilon_consumed = initial_epsilon
        self._delta_consumed = initial_delta
        self._operations: List[PrivacyMetrics] = []
        self._lock = threading.Lock()
    
    def consume_budget(self, epsilon: float, delta: float, operation_id: str) -> None:
        """Record privacy budget consumption."""
        with self._lock:
            self._epsilon_consumed += epsilon
            self._delta_consumed += delta
            
            metrics = PrivacyMetrics(
                epsilon=epsilon,
                delta=delta,
                noise_scale=0.0,  # Will be updated by calling code
                gradient_norm=0.0,  # Will be updated by calling code
                clipping_applied=False,  # Will be updated by calling code
                timestamp=time.time(),
                operation_id=operation_id
            )
            self._operations.append(metrics)
    
    def get_consumed_budget(self) -> tuple[float, float]:
        """Get total consumed privacy budget."""
        with self._lock:
            return self._epsilon_consumed, self._delta_consumed
    
    def get_operation_history(self) -> List[PrivacyMetrics]:
        """Get history of privacy operations."""
        with self._lock:
            return self._operations.copy()
    
    def reset_budget(self) -> None:
        """Reset privacy budget tracking."""
        with self._lock:
            self._epsilon_consumed = 0.0
            self._delta_consumed = 0.0
            self._operations.clear()


class DPTelemetry:
    """Main telemetry class for DP-Flash-Attention."""
    
    def __init__(
        self,
        service_name: str = "dp-flash-attention",
        privacy_tracking: bool = True,
        performance_tracking: bool = True,
        prometheus_registry: Optional[CollectorRegistry] = None
    ):
        self.service_name = service_name
        self.privacy_tracking = privacy_tracking
        self.performance_tracking = performance_tracking
        
        # Privacy budget tracking
        self.privacy_tracker = PrivacyBudgetTracker()
        
        # Performance metrics storage
        self._performance_metrics: List[PerformanceMetrics] = []
        self._perf_lock = threading.Lock()
        
        # Initialize telemetry backends
        self._init_prometheus(prometheus_registry)
        self._init_opentelemetry()
        
        # Set up logging
        self.logger = logging.getLogger(f"dp_telemetry.{service_name}")
    
    def _init_prometheus(self, registry: Optional[CollectorRegistry]) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available, skipping Prometheus setup")
            return
            
        self.prometheus_registry = registry or CollectorRegistry()
        
        # Privacy metrics
        self.privacy_epsilon_consumed = Counter(
            'dp_privacy_epsilon_consumed_total',
            'Total epsilon privacy budget consumed',
            ['operation_type', 'model_id'],
            registry=self.prometheus_registry
        )
        
        self.privacy_delta_consumed = Counter(
            'dp_privacy_delta_consumed_total', 
            'Total delta privacy budget consumed',
            ['operation_type', 'model_id'],
            registry=self.prometheus_registry
        )
        
        # Performance metrics
        self.attention_duration = Histogram(
            'dp_attention_duration_seconds',
            'Time spent in DP attention operations',
            ['operation_type', 'batch_size_bucket'],
            registry=self.prometheus_registry
        )
        
        self.gpu_memory_usage = Gauge(
            'dp_gpu_memory_used_bytes',
            'GPU memory used by DP operations',
            ['gpu_id'],
            registry=self.prometheus_registry
        )
    
    def _init_opentelemetry(self) -> None:
        """Initialize OpenTelemetry tracing."""
        if not OPENTELEMETRY_AVAILABLE:
            self.logger.warning("OpenTelemetry not available, skipping OpenTelemetry setup")
            return
            
        self.tracer = trace.get_tracer(self.service_name)
    
    @contextmanager
    def privacy_context(
        self,
        operation_type: str,
        epsilon: float,
        delta: float,
        model_id: str = "default"
    ):
        """Context manager for privacy-aware operations."""
        operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Start OpenTelemetry span if available
        span = None
        if hasattr(self, 'tracer'):
            span = self.tracer.start_span(f"dp_{operation_type}")
            span.set_attributes({
                "privacy.epsilon": epsilon,
                "privacy.delta": delta,
                "privacy.operation_id": operation_id,
                "model.id": model_id
            })
        
        try:
            yield operation_id
            
            # Record successful operation
            if self.privacy_tracking:
                self.privacy_tracker.consume_budget(epsilon, delta, operation_id)
                
                # Update Prometheus metrics
                if hasattr(self, 'privacy_epsilon_consumed'):
                    self.privacy_epsilon_consumed.labels(
                        operation_type=operation_type,
                        model_id=model_id
                    ).inc(epsilon)
                    
                    self.privacy_delta_consumed.labels(
                        operation_type=operation_type,
                        model_id=model_id
                    ).inc(delta)
            
            if span:
                span.set_status(Status(StatusCode.OK))
                
        except Exception as e:
            self.logger.error(f"Error in privacy operation {operation_id}: {e}")
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
            
        finally:
            if span:
                span.end()
    
    def record_performance_metrics(
        self,
        operation_type: str,
        duration_ms: float,
        memory_used_mb: float = 0.0,
        gpu_utilization: float = 0.0,
        batch_size: int = 0,
        sequence_length: int = 0,
        num_heads: int = 0
    ) -> None:
        """Record performance metrics for an operation."""
        if not self.performance_tracking:
            return
            
        operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        
        metrics = PerformanceMetrics(
            duration_ms=duration_ms,
            memory_used_mb=memory_used_mb,
            gpu_utilization=gpu_utilization,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_heads=num_heads,
            timestamp=time.time(),
            operation_id=operation_id
        )
        
        with self._perf_lock:
            self._performance_metrics.append(metrics)
        
        # Update Prometheus metrics
        if hasattr(self, 'attention_duration'):
            batch_bucket = self._get_batch_size_bucket(batch_size)
            self.attention_duration.labels(
                operation_type=operation_type,
                batch_size_bucket=batch_bucket
            ).observe(duration_ms / 1000.0)  # Convert to seconds
        
        if hasattr(self, 'gpu_memory_usage') and memory_used_mb > 0:
            self.gpu_memory_usage.labels(gpu_id="0").set(memory_used_mb * 1024 * 1024)
    
    def _get_batch_size_bucket(self, batch_size: int) -> str:
        """Get batch size bucket for metrics."""
        if batch_size <= 8:
            return "small"
        elif batch_size <= 32:
            return "medium"
        elif batch_size <= 128:
            return "large"
        else:
            return "xlarge"
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of privacy budget consumption."""
        epsilon_total, delta_total = self.privacy_tracker.get_consumed_budget()
        operations = self.privacy_tracker.get_operation_history()
        
        return {
            "total_epsilon": epsilon_total,
            "total_delta": delta_total,
            "operation_count": len(operations),
            "operations": [asdict(op) for op in operations[-10:]]  # Last 10 operations
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        with self._perf_lock:
            metrics = self._performance_metrics.copy()
        
        if not metrics:
            return {"message": "No performance metrics recorded"}
        
        # Calculate summary statistics
        durations = [m.duration_ms for m in metrics]
        memory_usage = [m.memory_used_mb for m in metrics if m.memory_used_mb > 0]
        
        return {
            "operation_count": len(metrics),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "avg_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "max_memory_mb": max(memory_usage) if memory_usage else 0,
            "recent_operations": [asdict(m) for m in metrics[-5:]]  # Last 5 operations
        }


def trace_privacy_operation(
    operation_type: str = "dp_operation",
    epsilon: float = 1.0,
    delta: float = 1e-5
):
    """Decorator to automatically trace privacy operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract epsilon and delta from kwargs if provided
            actual_epsilon = kwargs.get('epsilon', epsilon)
            actual_delta = kwargs.get('delta', delta)
            
            # Get or create global telemetry instance
            telemetry = _get_global_telemetry()
            
            with telemetry.privacy_context(operation_type, actual_epsilon, actual_delta):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Record performance metrics
                    telemetry.record_performance_metrics(
                        operation_type=operation_type,
                        duration_ms=duration_ms
                    )
                    
                    return result
                except Exception:
                    raise
        
        return wrapper
    return decorator


# Global telemetry instance
_global_telemetry: Optional[DPTelemetry] = None
_telemetry_lock = threading.Lock()

def _get_global_telemetry() -> DPTelemetry:
    """Get or create global telemetry instance."""
    global _global_telemetry
    
    if _global_telemetry is None:
        with _telemetry_lock:
            if _global_telemetry is None:
                _global_telemetry = DPTelemetry()
    
    return _global_telemetry


def configure_monitoring(
    service_name: str = "dp-flash-attention",
    privacy_tracking: bool = True,
    performance_tracking: bool = True,
    export_to_prometheus: bool = False,
    export_to_jaeger: bool = False
) -> DPTelemetry:
    """Configure global monitoring for DP-Flash-Attention."""
    global _global_telemetry
    
    with _telemetry_lock:
        _global_telemetry = DPTelemetry(
            service_name=service_name,
            privacy_tracking=privacy_tracking,
            performance_tracking=performance_tracking
        )
        
        # Set up exporters if requested
        if export_to_prometheus and PROMETHEUS_AVAILABLE:
            from prometheus_client import start_http_server
            start_http_server(8000)
            
        if export_to_jaeger and OPENTELEMETRY_AVAILABLE:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            
            trace.set_tracer_provider(TracerProvider())
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
    
    return _global_telemetry


def get_monitoring_status() -> Dict[str, Any]:
    """Get current monitoring system status."""
    telemetry = _get_global_telemetry()
    
    status = {
        "service_name": telemetry.service_name,
        "privacy_tracking": telemetry.privacy_tracking,
        "performance_tracking": telemetry.performance_tracking,
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
    }
    
    # Add privacy and performance summaries
    status.update({
        "privacy_summary": telemetry.get_privacy_summary(),
        "performance_summary": telemetry.get_performance_summary()
    })
    
    return status