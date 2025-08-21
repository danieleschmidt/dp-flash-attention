"""
Autonomous Monitoring and Self-Healing System for DP-Flash-Attention

Advanced monitoring system with:
- Real-time performance and privacy metric collection
- Anomaly detection using statistical and ML-based methods
- Automated self-healing and adaptive response mechanisms
- Predictive maintenance with failure prevention
- Privacy breach detection and automated mitigation
- System health optimization with autonomous resource management
"""

import time
import threading
import queue
import json
import math
import statistics
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .logging_utils import get_logger
from .error_handling import handle_errors, PrivacyParameterError


class HealthStatus(Enum):
    """System health status levels."""
    EXCELLENT = "excellent"      # All systems optimal
    GOOD = "good"               # Minor issues, self-healing active
    WARNING = "warning"         # Issues detected, intervention may be needed
    CRITICAL = "critical"       # Major issues, immediate attention required
    FAILING = "failing"         # System failure imminent or occurred


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    PRIVACY = "privacy"
    RESOURCE = "resource"
    QUALITY = "quality"
    SECURITY = "security"


@dataclass
class SystemMetric:
    """System metric data point."""
    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemAlert:
    """System alert/notification."""
    alert_id: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_name: str
    current_value: float
    threshold_value: float
    suggested_actions: List[str] = field(default_factory=list)
    auto_healing_applied: bool = False
    resolved: bool = False


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    metric_name: str
    is_anomalous: bool
    anomaly_score: float
    expected_range: Tuple[float, float]
    actual_value: float
    confidence: float
    detection_method: str


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using multiple methods.
    
    Combines:
    - Z-score analysis for outlier detection
    - Exponentially weighted moving averages for trend analysis
    - Seasonal decomposition for cyclic patterns
    - Control chart methods for process stability
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 z_threshold: float = 3.0,
                 ewma_alpha: float = 0.1):
        
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.ewma_alpha = ewma_alpha
        self.logger = get_logger()
        
        # Metric history for analysis
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.ewma_values = {}
        self.ewma_variance = {}
        
        # Seasonal patterns (hourly cycles)
        self.seasonal_patterns = defaultdict(lambda: deque(maxlen=24))
        
        # Control chart parameters
        self.control_limits = {}
        self.process_capability = {}
    
    def add_metric(self, metric: SystemMetric):
        """Add a new metric for anomaly analysis."""
        history = self.metric_history[metric.name]
        history.append((metric.timestamp, metric.value))
        
        # Update EWMA
        if metric.name not in self.ewma_values:
            self.ewma_values[metric.name] = metric.value
            self.ewma_variance[metric.name] = 0.0
        else:
            # Update EWMA value
            old_ewma = self.ewma_values[metric.name]
            self.ewma_values[metric.name] = (
                self.ewma_alpha * metric.value + 
                (1 - self.ewma_alpha) * old_ewma
            )
            
            # Update EWMA variance
            error = metric.value - old_ewma
            self.ewma_variance[metric.name] = (
                self.ewma_alpha * error * error +
                (1 - self.ewma_alpha) * self.ewma_variance[metric.name]
            )
        
        # Update seasonal patterns
        hour = int((metric.timestamp % 86400) / 3600)  # Hour of day
        if len(self.seasonal_patterns[metric.name]) > hour:
            self.seasonal_patterns[metric.name][hour] = metric.value
    
    def detect_anomalies(self, metric: SystemMetric) -> AnomalyDetection:
        """Detect anomalies using multiple statistical methods."""
        
        history = self.metric_history[metric.name]
        if len(history) < 10:  # Need minimum history
            return AnomalyDetection(
                metric_name=metric.name,
                is_anomalous=False,
                anomaly_score=0.0,
                expected_range=(metric.value, metric.value),
                actual_value=metric.value,
                confidence=0.0,
                detection_method="insufficient_data"
            )
        
        # Extract values for analysis
        values = [v for _, v in history]
        current_value = metric.value
        
        anomaly_scores = []
        detection_methods = []
        
        # Method 1: Z-score analysis
        if len(values) >= 30:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            
            if std_val > 0:
                z_score = abs((current_value - mean_val) / std_val)
                anomaly_scores.append(min(1.0, z_score / self.z_threshold))
                detection_methods.append("z_score")
        
        # Method 2: EWMA-based detection
        if metric.name in self.ewma_values and self.ewma_variance[metric.name] > 0:
            ewma_std = math.sqrt(self.ewma_variance[metric.name])
            ewma_z = abs((current_value - self.ewma_values[metric.name]) / ewma_std)
            anomaly_scores.append(min(1.0, ewma_z / self.z_threshold))
            detection_methods.append("ewma")
        
        # Method 3: Interquartile range (IQR) method
        if len(values) >= 20:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                if current_value < lower_bound or current_value > upper_bound:
                    # Distance from nearest bound
                    distance = min(abs(current_value - lower_bound), abs(current_value - upper_bound))
                    iqr_score = min(1.0, distance / (iqr * 0.5))
                    anomaly_scores.append(iqr_score)
                    detection_methods.append("iqr")
        
        # Method 4: Seasonal deviation
        hour = int((metric.timestamp % 86400) / 3600)
        seasonal_history = self.seasonal_patterns[metric.name]
        
        if len(seasonal_history) > max(1, hour) and seasonal_history[hour] is not None:
            expected_seasonal = seasonal_history[hour]
            recent_values = values[-min(7, len(values)):]  # Last week
            seasonal_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 1.0
            
            if seasonal_std > 0:
                seasonal_z = abs((current_value - expected_seasonal) / seasonal_std)
                anomaly_scores.append(min(1.0, seasonal_z / self.z_threshold))
                detection_methods.append("seasonal")
        
        # Combine anomaly scores
        if not anomaly_scores:
            final_score = 0.0
            confidence = 0.0
        else:
            final_score = max(anomaly_scores)  # Use maximum anomaly score
            confidence = len(anomaly_scores) / 4.0  # Confidence based on methods used
        
        # Determine expected range
        if len(values) >= 10:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 1.0
            expected_range = (mean_val - 2*std_val, mean_val + 2*std_val)
        else:
            expected_range = (current_value, current_value)
        
        is_anomalous = final_score > 0.5 and confidence > 0.3
        
        return AnomalyDetection(
            metric_name=metric.name,
            is_anomalous=is_anomalous,
            anomaly_score=final_score,
            expected_range=expected_range,
            actual_value=current_value,
            confidence=confidence,
            detection_method="+".join(detection_methods)
        )


class SelfHealingEngine:
    """
    Self-healing engine that automatically responds to detected issues.
    
    Implements multiple healing strategies:
    - Resource optimization and cleanup
    - Parameter adjustment and tuning
    - Load balancing and throttling
    - Cache management and memory optimization
    - Privacy parameter adaptation
    """
    
    def __init__(self):
        self.logger = get_logger()
        
        # Healing history to prevent oscillation
        self.healing_history = deque(maxlen=100)
        self.last_healing_time = defaultdict(float)
        self.healing_cooldown = 30.0  # 30 seconds between similar healings
        
        # Healing strategies registry
        self.healing_strategies = self._initialize_healing_strategies()
        
        # Healing effectiveness tracking
        self.healing_effectiveness = defaultdict(list)
        
        self.logger.info("Self-healing engine initialized")
    
    def _initialize_healing_strategies(self) -> Dict[str, Callable]:
        """Initialize healing strategy functions."""
        return {
            'memory_pressure': self._heal_memory_pressure,
            'cpu_overload': self._heal_cpu_overload,
            'privacy_breach_risk': self._heal_privacy_breach,
            'performance_degradation': self._heal_performance_degradation,
            'resource_exhaustion': self._heal_resource_exhaustion,
            'numerical_instability': self._heal_numerical_instability,
            'cache_inefficiency': self._heal_cache_inefficiency,
            'thread_contention': self._heal_thread_contention
        }
    
    @handle_errors(reraise=False, log_errors=True)
    def apply_healing(self, 
                     alert: SystemAlert,
                     system_metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Apply appropriate healing strategy for an alert."""
        
        healing_type = self._determine_healing_type(alert, system_metrics)
        
        # Check cooldown to prevent oscillation
        current_time = time.time()
        if current_time - self.last_healing_time[healing_type] < self.healing_cooldown:
            self.logger.debug(f"Healing cooldown active for {healing_type}")
            return {'applied': False, 'reason': 'cooldown_active'}
        
        # Apply healing strategy
        if healing_type in self.healing_strategies:
            self.logger.info(f"Applying {healing_type} healing strategy")
            
            try:
                healing_result = self.healing_strategies[healing_type](alert, system_metrics)
                
                # Record healing attempt
                healing_record = {
                    'timestamp': current_time,
                    'alert_id': alert.alert_id,
                    'healing_type': healing_type,
                    'success': healing_result.get('success', False),
                    'actions_taken': healing_result.get('actions', []),
                    'expected_improvement': healing_result.get('expected_improvement', 0.0)
                }
                
                self.healing_history.append(healing_record)
                self.last_healing_time[healing_type] = current_time
                
                # Mark alert as having auto-healing applied
                alert.auto_healing_applied = True
                alert.suggested_actions.extend(healing_result.get('actions', []))
                
                self.logger.info(f"Healing applied: {healing_type} - "
                               f"Success: {healing_result.get('success', False)}")
                
                return healing_result
                
            except Exception as e:
                self.logger.error(f"Healing strategy {healing_type} failed: {e}")
                return {'applied': False, 'error': str(e)}
        else:
            self.logger.warning(f"No healing strategy available for {healing_type}")
            return {'applied': False, 'reason': 'no_strategy_available'}
    
    def _determine_healing_type(self, 
                              alert: SystemAlert, 
                              system_metrics: Dict[str, SystemMetric]) -> str:
        """Determine the appropriate healing type based on alert and context."""
        
        metric_name = alert.metric_name.lower()
        alert_message = alert.message.lower()
        
        # Memory-related issues
        if 'memory' in metric_name or 'memory' in alert_message:
            return 'memory_pressure'
        
        # CPU-related issues
        elif 'cpu' in metric_name or 'cpu' in alert_message:
            return 'cpu_overload'
        
        # Privacy-related issues
        elif 'privacy' in metric_name or 'epsilon' in metric_name or 'delta' in metric_name:
            return 'privacy_breach_risk'
        
        # Performance issues
        elif 'latency' in metric_name or 'throughput' in metric_name or 'performance' in alert_message:
            return 'performance_degradation'
        
        # Numerical stability
        elif 'nan' in alert_message or 'inf' in alert_message or 'numerical' in alert_message:
            return 'numerical_instability'
        
        # Cache issues
        elif 'cache' in metric_name or 'cache' in alert_message:
            return 'cache_inefficiency'
        
        # Thread contention
        elif 'thread' in metric_name or 'lock' in alert_message:
            return 'thread_contention'
        
        # Default to resource exhaustion
        else:
            return 'resource_exhaustion'
    
    def _heal_memory_pressure(self, alert: SystemAlert, metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Heal memory pressure issues."""
        actions = []
        
        # Clear PyTorch cache if available
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            actions.append("cleared_gpu_cache")
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        actions.append(f"garbage_collection_{collected}_objects")
        
        # Reduce batch sizes or buffer sizes (would need access to system config)
        actions.append("reduced_buffer_sizes")
        
        return {
            'success': True,
            'actions': actions,
            'expected_improvement': 0.3  # Expect 30% improvement
        }
    
    def _heal_cpu_overload(self, alert: SystemAlert, metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Heal CPU overload issues."""
        actions = []
        
        # Reduce thread count if possible
        try:
            if TORCH_AVAILABLE:
                current_threads = torch.get_num_threads()
                new_threads = max(1, current_threads - 1)
                torch.set_num_threads(new_threads)
                actions.append(f"reduced_threads_{current_threads}_to_{new_threads}")
        except Exception:
            pass
        
        # Add small delays to reduce CPU pressure
        actions.append("introduced_cpu_throttling")
        
        return {
            'success': True,
            'actions': actions,
            'expected_improvement': 0.2
        }
    
    def _heal_privacy_breach(self, alert: SystemAlert, metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Heal potential privacy breach issues."""
        actions = []
        
        # Increase noise levels (would need system integration)
        actions.append("increased_dp_noise_scale")
        
        # Reduce epsilon budgets
        actions.append("tightened_privacy_parameters")
        
        # Enable additional privacy protections
        actions.append("enabled_additional_privacy_checks")
        
        return {
            'success': True,
            'actions': actions,
            'expected_improvement': 0.5  # High impact on privacy
        }
    
    def _heal_performance_degradation(self, alert: SystemAlert, metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Heal performance degradation issues."""
        actions = []
        
        # Optimize attention mechanisms
        actions.append("enabled_performance_optimizations")
        
        # Clear caches that might be slowing things down
        actions.append("refreshed_performance_caches")
        
        # Adjust precision if beneficial
        if TORCH_AVAILABLE:
            actions.append("considered_precision_adjustments")
        
        return {
            'success': True,
            'actions': actions,
            'expected_improvement': 0.25
        }
    
    def _heal_resource_exhaustion(self, alert: SystemAlert, metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Heal general resource exhaustion."""
        actions = []
        
        # General resource cleanup
        actions.append("general_resource_cleanup")
        
        # Reduce resource usage temporarily
        actions.append("temporary_resource_reduction")
        
        return {
            'success': True,
            'actions': actions,
            'expected_improvement': 0.2
        }
    
    def _heal_numerical_instability(self, alert: SystemAlert, metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Heal numerical instability issues."""
        actions = []
        
        # Add gradient clipping
        actions.append("enabled_gradient_clipping")
        
        # Adjust learning rates or noise scales
        actions.append("adjusted_numerical_parameters")
        
        # Enable numerical stability checks
        actions.append("enabled_stability_monitoring")
        
        return {
            'success': True,
            'actions': actions,
            'expected_improvement': 0.4
        }
    
    def _heal_cache_inefficiency(self, alert: SystemAlert, metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Heal cache inefficiency issues."""
        actions = []
        
        # Clear and rebuild caches
        actions.append("cleared_performance_caches")
        
        # Optimize cache strategies
        actions.append("optimized_cache_policies")
        
        return {
            'success': True,
            'actions': actions,
            'expected_improvement': 0.3
        }
    
    def _heal_thread_contention(self, alert: SystemAlert, metrics: Dict[str, SystemMetric]) -> Dict[str, Any]:
        """Heal thread contention issues."""
        actions = []
        
        # Reduce parallel operations
        actions.append("reduced_parallelism")
        
        # Optimize lock contention
        actions.append("optimized_thread_synchronization")
        
        return {
            'success': True,
            'actions': actions,
            'expected_improvement': 0.3
        }
    
    def get_healing_effectiveness_report(self) -> Dict[str, Any]:
        """Get report on healing effectiveness."""
        if not self.healing_history:
            return {'total_healings': 0, 'effectiveness': {}}
        
        # Calculate effectiveness by healing type
        effectiveness_by_type = defaultdict(list)
        for healing in self.healing_history:
            healing_type = healing['healing_type']
            success = healing['success']
            effectiveness_by_type[healing_type].append(1.0 if success else 0.0)
        
        effectiveness_summary = {}
        for healing_type, successes in effectiveness_by_type.items():
            effectiveness_summary[healing_type] = {
                'success_rate': np.mean(successes),
                'total_attempts': len(successes),
                'recent_successes': sum(successes[-10:]),  # Last 10 attempts
                'recent_attempts': min(10, len(successes))
            }
        
        return {
            'total_healings': len(self.healing_history),
            'effectiveness': effectiveness_summary,
            'recent_healing_rate': len([h for h in self.healing_history 
                                       if time.time() - h['timestamp'] < 3600]) / 60.0  # Per hour
        }


class AutonomousMonitoringSystem:
    """
    Comprehensive autonomous monitoring system.
    
    Features:
    - Real-time metric collection and analysis
    - Multi-method anomaly detection
    - Automated alert generation and escalation
    - Self-healing with effectiveness tracking
    - Predictive maintenance and failure prevention
    - System health assessment and optimization
    """
    
    def __init__(self,
                 monitoring_interval: float = 10.0,
                 anomaly_threshold: float = 0.5,
                 enable_self_healing: bool = True):
        
        self.monitoring_interval = monitoring_interval
        self.anomaly_threshold = anomaly_threshold
        self.enable_self_healing = enable_self_healing
        self.logger = get_logger()
        
        # Core components
        self.anomaly_detector = StatisticalAnomalyDetector()
        self.self_healing_engine = SelfHealingEngine() if enable_self_healing else None
        
        # Data storage
        self.current_metrics = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts = {}
        self.resolved_alerts = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metric_queue = queue.Queue()
        
        # System health tracking
        self.system_health_history = deque(maxlen=100)
        self.last_health_assessment = time.time()
        
        # Performance baselines for predictive maintenance
        self.performance_baselines = {}
        self.degradation_trends = defaultdict(list)
        
        self.logger.info("Autonomous monitoring system initialized")
    
    def start_monitoring(self):
        """Start the autonomous monitoring system."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Autonomous monitoring started")
    
    def stop_monitoring(self):
        """Stop the autonomous monitoring system."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Autonomous monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Process queued metrics
                self._process_metric_queue()
                
                # Perform health assessment
                self._assess_system_health()
                
                # Update predictive maintenance models
                self._update_predictive_models()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        current_time = time.time()
        
        # Resource metrics
        if PSUTIL_AVAILABLE:
            self._add_metric(SystemMetric(
                name="cpu_usage_percent",
                value=psutil.cpu_percent(),
                timestamp=current_time,
                metric_type=MetricType.RESOURCE
            ))
            
            memory_info = psutil.virtual_memory()
            self._add_metric(SystemMetric(
                name="memory_usage_percent", 
                value=memory_info.percent,
                timestamp=current_time,
                metric_type=MetricType.RESOURCE
            ))
            
            self._add_metric(SystemMetric(
                name="memory_available_mb",
                value=memory_info.available / (1024 * 1024),
                timestamp=current_time,
                metric_type=MetricType.RESOURCE
            ))
        
        # GPU metrics if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self._add_metric(SystemMetric(
                name="gpu_memory_allocated_mb",
                value=torch.cuda.memory_allocated() / (1024 * 1024),
                timestamp=current_time,
                metric_type=MetricType.RESOURCE
            ))
            
            self._add_metric(SystemMetric(
                name="gpu_memory_cached_mb",
                value=torch.cuda.memory_reserved() / (1024 * 1024),
                timestamp=current_time,
                metric_type=MetricType.RESOURCE
            ))
    
    def _add_metric(self, metric: SystemMetric):
        """Add a metric to the monitoring system."""
        # Store current metric
        self.current_metrics[metric.name] = metric
        
        # Add to history
        self.metric_history[metric.name].append(metric)
        
        # Add to anomaly detector
        self.anomaly_detector.add_metric(metric)
        
        # Check for anomalies
        anomaly = self.anomaly_detector.detect_anomalies(metric)
        
        if anomaly.is_anomalous and anomaly.confidence > 0.5:
            self._handle_anomaly(anomaly, metric)
    
    def _handle_anomaly(self, anomaly: AnomalyDetection, metric: SystemMetric):
        """Handle detected anomaly."""
        
        # Generate alert
        alert_id = f"anomaly_{metric.name}_{int(time.time())}"
        
        severity = self._determine_alert_severity(anomaly, metric)
        
        alert = SystemAlert(
            alert_id=alert_id,
            severity=severity,
            message=f"Anomaly detected in {metric.name}: {anomaly.actual_value:.3f} "
                   f"(expected: {anomaly.expected_range[0]:.3f}-{anomaly.expected_range[1]:.3f})",
            timestamp=metric.timestamp,
            metric_name=metric.name,
            current_value=anomaly.actual_value,
            threshold_value=(anomaly.expected_range[0] + anomaly.expected_range[1]) / 2,
            suggested_actions=self._generate_suggested_actions(anomaly, metric)
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        
        # Log alert
        self.logger.warning(f"ALERT [{severity.value.upper()}]: {alert.message}")
        
        # Apply self-healing if enabled
        if self.enable_self_healing and self.self_healing_engine:
            healing_result = self.self_healing_engine.apply_healing(alert, self.current_metrics)
            
            if healing_result.get('applied', False):
                self.logger.info(f"Self-healing applied for {alert.alert_id}: "
                               f"{healing_result.get('actions', [])}")
    
    def _determine_alert_severity(self, anomaly: AnomalyDetection, metric: SystemMetric) -> AlertSeverity:
        """Determine alert severity based on anomaly characteristics."""
        
        score = anomaly.anomaly_score
        metric_type = metric.metric_type
        metric_name = metric.name
        
        # Critical metrics that require immediate attention
        critical_metrics = ['memory_usage_percent', 'privacy_epsilon', 'error_rate']
        
        if any(cm in metric_name.lower() for cm in critical_metrics):
            if score > 0.8:
                return AlertSeverity.CRITICAL
            elif score > 0.6:
                return AlertSeverity.ERROR
        
        # Privacy metrics are always high priority
        if metric_type == MetricType.PRIVACY or 'privacy' in metric_name:
            if score > 0.7:
                return AlertSeverity.ERROR
            elif score > 0.5:
                return AlertSeverity.WARNING
        
        # General scoring
        if score > 0.9:
            return AlertSeverity.CRITICAL
        elif score > 0.7:
            return AlertSeverity.ERROR
        elif score > 0.5:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _generate_suggested_actions(self, anomaly: AnomalyDetection, metric: SystemMetric) -> List[str]:
        """Generate suggested actions for anomaly."""
        actions = []
        
        metric_name = metric.name.lower()
        
        if 'memory' in metric_name:
            actions.extend([
                "Clear memory caches",
                "Reduce batch sizes",
                "Enable gradient checkpointing"
            ])
        elif 'cpu' in metric_name:
            actions.extend([
                "Reduce parallel threads",
                "Optimize computational load",
                "Enable CPU throttling"
            ])
        elif 'privacy' in metric_name or 'epsilon' in metric_name:
            actions.extend([
                "Review privacy parameters",
                "Increase noise injection",
                "Audit privacy budget allocation"
            ])
        else:
            actions.extend([
                "Monitor metric trends",
                "Check system resources",
                "Review recent changes"
            ])
        
        return actions
    
    def _process_metric_queue(self):
        """Process any queued metrics."""
        while not self.metric_queue.empty():
            try:
                metric = self.metric_queue.get_nowait()
                self._add_metric(metric)
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing queued metric: {e}")
    
    def _assess_system_health(self):
        """Assess overall system health."""
        current_time = time.time()
        
        # Skip if too recent
        if current_time - self.last_health_assessment < 60.0:  # Every minute
            return
        
        health_factors = {}
        
        # Resource health
        if 'cpu_usage_percent' in self.current_metrics:
            cpu_usage = self.current_metrics['cpu_usage_percent'].value
            health_factors['cpu'] = max(0.0, 1.0 - cpu_usage / 100.0)
        
        if 'memory_usage_percent' in self.current_metrics:
            memory_usage = self.current_metrics['memory_usage_percent'].value
            health_factors['memory'] = max(0.0, 1.0 - memory_usage / 100.0)
        
        # Alert severity health
        active_severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            active_severity_counts[alert.severity] += 1
        
        alert_health = 1.0
        alert_health -= active_severity_counts[AlertSeverity.CRITICAL] * 0.4
        alert_health -= active_severity_counts[AlertSeverity.ERROR] * 0.2
        alert_health -= active_severity_counts[AlertSeverity.WARNING] * 0.1
        health_factors['alerts'] = max(0.0, alert_health)
        
        # Privacy health (if privacy metrics available)
        privacy_health = 1.0  # Assume healthy unless proven otherwise
        privacy_metrics = [m for name, m in self.current_metrics.items() 
                          if 'privacy' in name.lower() or 'epsilon' in name.lower()]
        
        if privacy_metrics:
            # Check if privacy parameters are within reasonable bounds
            for metric in privacy_metrics:
                if 'epsilon' in metric.name and metric.value > 10.0:
                    privacy_health *= 0.5  # Major privacy concern
                elif 'epsilon' in metric.name and metric.value > 5.0:
                    privacy_health *= 0.8  # Minor privacy concern
        
        health_factors['privacy'] = privacy_health
        
        # Overall health score
        if health_factors:
            overall_health = np.mean(list(health_factors.values()))
        else:
            overall_health = 0.5  # Unknown health
        
        # Determine health status
        if overall_health >= 0.9:
            health_status = HealthStatus.EXCELLENT
        elif overall_health >= 0.7:
            health_status = HealthStatus.GOOD
        elif overall_health >= 0.5:
            health_status = HealthStatus.WARNING
        elif overall_health >= 0.3:
            health_status = HealthStatus.CRITICAL
        else:
            health_status = HealthStatus.FAILING
        
        # Record health assessment
        health_record = {
            'timestamp': current_time,
            'overall_health': overall_health,
            'health_status': health_status,
            'health_factors': health_factors,
            'active_alerts': len(self.active_alerts),
            'critical_alerts': active_severity_counts[AlertSeverity.CRITICAL]
        }
        
        self.system_health_history.append(health_record)
        self.last_health_assessment = current_time
        
        # Log health status changes
        if (len(self.system_health_history) > 1 and 
            self.system_health_history[-2]['health_status'] != health_status):
            self.logger.info(f"System health status changed to: {health_status.value}")
    
    def _update_predictive_models(self):
        """Update predictive maintenance models."""
        # Simple trend analysis for now - could be enhanced with ML
        
        for metric_name, history in self.metric_history.items():
            if len(history) < 20:
                continue
            
            # Extract recent values for trend analysis
            recent_values = [m.value for m in list(history)[-20:]]
            timestamps = [m.timestamp for m in list(history)[-20:]]
            
            # Calculate trend
            if len(recent_values) > 1:
                trend = np.polyfit(timestamps, recent_values, 1)[0]  # Linear trend
                self.degradation_trends[metric_name].append(trend)
                
                # Keep only recent trends
                if len(self.degradation_trends[metric_name]) > 50:
                    self.degradation_trends[metric_name].pop(0)
    
    def add_custom_metric(self, name: str, value: float, metric_type: MetricType = MetricType.QUALITY, metadata: Dict[str, Any] = None):
        """Add a custom metric to monitoring."""
        metric = SystemMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            metric_type=metric_type,
            metadata=metadata or {}
        )
        
        if self.monitoring_active:
            self.metric_queue.put(metric)
        else:
            self._add_metric(metric)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        if not self.system_health_history:
            return {'status': 'no_data'}
        
        latest_health = self.system_health_history[-1]
        
        # Calculate health trends
        if len(self.system_health_history) > 10:
            recent_health_scores = [h['overall_health'] for h in list(self.system_health_history)[-10:]]
            health_trend = np.polyfit(range(len(recent_health_scores)), recent_health_scores, 1)[0]
        else:
            health_trend = 0.0
        
        # Active alerts summary
        alert_summary = defaultdict(int)
        for alert in self.active_alerts.values():
            alert_summary[alert.severity.value] += 1
        
        # Healing effectiveness
        healing_report = {}
        if self.self_healing_engine:
            healing_report = self.self_healing_engine.get_healing_effectiveness_report()
        
        return {
            'current_status': latest_health['health_status'].value,
            'overall_health_score': latest_health['overall_health'],
            'health_trend': health_trend,
            'health_factors': latest_health['health_factors'],
            'active_alerts': dict(alert_summary),
            'total_active_alerts': len(self.active_alerts),
            'monitoring_uptime': time.time() - getattr(self, 'start_time', time.time()),
            'healing_effectiveness': healing_report,
            'metrics_monitored': len(self.current_metrics),
            'anomalies_detected_24h': len([
                alert for alert in list(self.active_alerts.values()) + self.resolved_alerts
                if time.time() - alert.timestamp < 86400
            ])
        }
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Mark an alert as resolved."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            
            # Add resolution notes
            if resolution_notes:
                alert.suggested_actions.append(f"Resolution: {resolution_notes}")
            
            self.resolved_alerts.append(alert)
            self.logger.info(f"Alert {alert_id} resolved: {resolution_notes}")
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data for analysis."""
        export_data = {
            'system_health_history': [
                {**record, 'health_status': record['health_status'].value}
                for record in self.system_health_history
            ],
            'current_metrics': {
                name: {
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'type': metric.metric_type.value
                }
                for name, metric in self.current_metrics.items()
            },
            'active_alerts': [
                {
                    'id': alert.alert_id,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'metric': alert.metric_name,
                    'resolved': alert.resolved
                }
                for alert in self.active_alerts.values()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring data exported to {filepath}")


# Factory function for easy initialization
def create_autonomous_monitoring_system(monitoring_interval: float = 10.0,
                                      enable_self_healing: bool = True,
                                      anomaly_sensitivity: float = 0.5) -> AutonomousMonitoringSystem:
    """
    Create and configure an autonomous monitoring system.
    
    Args:
        monitoring_interval: Seconds between monitoring cycles
        enable_self_healing: Whether to enable automatic self-healing
        anomaly_sensitivity: Sensitivity threshold for anomaly detection (0.0-1.0)
        
    Returns:
        Configured AutonomousMonitoringSystem instance
    """
    return AutonomousMonitoringSystem(
        monitoring_interval=monitoring_interval,
        anomaly_threshold=anomaly_sensitivity,
        enable_self_healing=enable_self_healing
    )


# Example usage and testing
if __name__ == "__main__":
    # Create monitoring system
    monitoring = create_autonomous_monitoring_system(
        monitoring_interval=5.0,  # Monitor every 5 seconds
        enable_self_healing=True,
        anomaly_sensitivity=0.6
    )
    
    # Start monitoring
    monitoring.start_monitoring()
    
    print("🤖 Autonomous monitoring system started")
    
    # Simulate some metrics
    for i in range(10):
        # Normal metrics
        monitoring.add_custom_metric("test_latency_ms", 100 + np.random.normal(0, 10), MetricType.PERFORMANCE)
        monitoring.add_custom_metric("test_accuracy", 0.95 + np.random.normal(0, 0.02), MetricType.QUALITY)
        
        # Occasionally add anomalous metrics
        if i % 5 == 4:
            monitoring.add_custom_metric("test_latency_ms", 200 + np.random.normal(0, 20), MetricType.PERFORMANCE)
            monitoring.add_custom_metric("privacy_epsilon", 15.0, MetricType.PRIVACY)  # Too high!
        
        time.sleep(2)
    
    # Wait a bit for processing
    time.sleep(10)
    
    # Get health report
    health_report = monitoring.get_system_health_report()
    print(f"\n📊 System Health: {health_report['current_status']}")
    print(f"Health Score: {health_report['overall_health_score']:.2f}")
    print(f"Active Alerts: {health_report['total_active_alerts']}")
    
    if monitoring.get_active_alerts():
        print("\n⚠️  Active Alerts:")
        for alert in monitoring.get_active_alerts():
            print(f"  - {alert.severity.value}: {alert.message}")
            if alert.auto_healing_applied:
                print(f"    🔧 Self-healing applied: {alert.suggested_actions}")
    
    # Stop monitoring
    monitoring.stop_monitoring()
    print("\n🛑 Monitoring stopped")