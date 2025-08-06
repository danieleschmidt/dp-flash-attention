"""
Logging utilities for DP-Flash-Attention.

Provides structured logging for privacy operations, performance monitoring,
and security auditing with privacy-preserving log management.
"""

import logging
import json
import time
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import warnings

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PrivacyAwareFormatter(logging.Formatter):
    """
    Log formatter that sanitizes sensitive information.
    
    Ensures that logs don't accidentally leak private information
    while still providing useful debugging information.
    """
    
    SENSITIVE_KEYS = {
        'data', 'input', 'output', 'tensor', 'gradients', 'weights',
        'bias', 'activations', 'embeddings', 'features'
    }
    
    def format(self, record):
        """Format log record while sanitizing sensitive data."""
        # Make a copy to avoid modifying original record
        safe_record = logging.makeLogRecord(record.__dict__)
        
        # Sanitize message if it contains sensitive data
        if hasattr(safe_record, 'msg') and isinstance(safe_record.msg, str):
            safe_record.msg = self._sanitize_message(safe_record.msg)
        
        # Sanitize args
        if hasattr(safe_record, 'args') and safe_record.args:
            safe_record.args = self._sanitize_args(safe_record.args)
        
        return super().format(safe_record)
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize log message to remove sensitive information."""
        # Simple pattern matching for tensor-like objects
        import re
        
        # Replace tensor representations with sanitized versions
        message = re.sub(
            r'tensor\([^)]*\)', 
            'tensor([SANITIZED])', 
            message, 
            flags=re.IGNORECASE
        )
        
        # Replace numpy array representations
        message = re.sub(
            r'array\([^)]*\)',
            'array([SANITIZED])',
            message,
            flags=re.IGNORECASE
        )
        
        return message
    
    def _sanitize_args(self, args):
        """Sanitize log arguments."""
        sanitized = []
        
        for arg in args:
            if HAS_TORCH and isinstance(arg, torch.Tensor):
                # Replace tensor with metadata only
                sanitized.append(f"Tensor(shape={list(arg.shape)}, dtype={arg.dtype})")
            elif isinstance(arg, (list, tuple)) and len(arg) > 100:
                # Truncate large lists/tuples
                sanitized.append(f"{type(arg).__name__}([TRUNCATED {len(arg)} items])")
            elif isinstance(arg, dict):
                # Sanitize dictionary
                sanitized_dict = {}
                for key, value in arg.items():
                    if key.lower() in self.SENSITIVE_KEYS:
                        if HAS_TORCH and isinstance(value, torch.Tensor):
                            sanitized_dict[key] = f"Tensor(shape={list(value.shape)})"
                        else:
                            sanitized_dict[key] = "[SANITIZED]"
                    else:
                        sanitized_dict[key] = value
                sanitized.append(sanitized_dict)
            else:
                sanitized.append(arg)
        
        return tuple(sanitized)


class PrivacyMetricsLogger:
    """
    Logger specifically for differential privacy metrics and auditing.
    
    Tracks privacy budget consumption, noise injection, and security events
    without logging sensitive data.
    """
    
    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO):
        """
        Initialize privacy metrics logger.
        
        Args:
            log_file: Optional file to write logs to
            level: Logging level
        """
        self.logger = logging.getLogger('dp_flash_attention.privacy')
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = PrivacyAwareFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                warnings.warn(f"Could not create file handler: {e}")
        
        # Privacy metrics storage
        self.privacy_log = []
        self.performance_log = []
        self.security_events = []
    
    def log_privacy_step(
        self, 
        epsilon_spent: float,
        delta: float,
        noise_scale: float,
        gradient_norm: float,
        clipping_bound: float,
        step: int = None,
        additional_info: Dict[str, Any] = None
    ):
        """Log privacy step information."""
        privacy_info = {
            'timestamp': time.time(),
            'step': step,
            'epsilon_spent': epsilon_spent,
            'delta': delta,
            'noise_scale': noise_scale,
            'gradient_norm': gradient_norm,
            'clipping_bound': clipping_bound,
            'clipping_applied': gradient_norm > clipping_bound
        }
        
        if additional_info:
            privacy_info.update(additional_info)
        
        # Store in memory
        self.privacy_log.append(privacy_info)
        
        # Log to file/console
        self.logger.info(
            f"Privacy step - ε={epsilon_spent:.6f}, δ={delta:.2e}, "
            f"noise_scale={noise_scale:.4f}, grad_norm={gradient_norm:.4f}, "
            f"clipped={gradient_norm > clipping_bound}"
        )
        
        # Check for concerning patterns
        if epsilon_spent > 10:
            self.logger.warning(f"High epsilon value: {epsilon_spent:.6f}")
        
        if noise_scale < 0.1:
            self.logger.warning(f"Low noise scale: {noise_scale:.4f}")
        
        if gradient_norm > clipping_bound * 1.5:
            self.logger.warning(
                f"Gradient norm {gradient_norm:.4f} significantly exceeds "
                f"clipping bound {clipping_bound:.4f}"
            )
    
    def log_performance_metrics(
        self,
        operation: str,
        duration_ms: float,
        memory_usage_mb: Optional[float] = None,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        additional_metrics: Dict[str, Any] = None
    ):
        """Log performance metrics."""
        perf_info = {
            'timestamp': time.time(),
            'operation': operation,
            'duration_ms': duration_ms,
            'memory_usage_mb': memory_usage_mb,
            'batch_size': batch_size,
            'sequence_length': sequence_length
        }
        
        if additional_metrics:
            perf_info.update(additional_metrics)
        
        # Store in memory
        self.performance_log.append(perf_info)
        
        # Log to file/console
        self.logger.info(
            f"Performance - {operation}: {duration_ms:.2f}ms"
            f"{f', memory: {memory_usage_mb:.1f}MB' if memory_usage_mb else ''}"
            f"{f', batch: {batch_size}' if batch_size else ''}"
            f"{f', seq_len: {sequence_length}' if sequence_length else ''}"
        )
        
        # Check for performance issues
        if duration_ms > 1000:  # More than 1 second
            self.logger.warning(f"Slow operation {operation}: {duration_ms:.2f}ms")
        
        if memory_usage_mb and memory_usage_mb > 8000:  # More than 8GB
            self.logger.warning(f"High memory usage in {operation}: {memory_usage_mb:.1f}MB")
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        additional_data: Dict[str, Any] = None
    ):
        """Log security-related events."""
        event_info = {
            'timestamp': time.time(),
            'event_type': event_type,
            'severity': severity,
            'description': description
        }
        
        if additional_data:
            # Sanitize additional data
            sanitized_data = {}
            for key, value in additional_data.items():
                if key.lower() in PrivacyAwareFormatter.SENSITIVE_KEYS:
                    sanitized_data[key] = "[SANITIZED]"
                elif HAS_TORCH and isinstance(value, torch.Tensor):
                    sanitized_data[key] = f"Tensor(shape={list(value.shape)})"
                else:
                    sanitized_data[key] = value
            event_info['additional_data'] = sanitized_data
        
        # Store in memory
        self.security_events.append(event_info)
        
        # Log with appropriate level
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity.lower(), logging.WARNING)
        
        self.logger.log(
            log_level,
            f"Security event [{event_type}] - {description}"
        )
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of privacy metrics."""
        if not self.privacy_log:
            return {'status': 'no_data', 'total_steps': 0}
        
        total_epsilon = sum(entry['epsilon_spent'] for entry in self.privacy_log)
        avg_noise_scale = sum(entry['noise_scale'] for entry in self.privacy_log) / len(self.privacy_log)
        clipping_rate = sum(1 for entry in self.privacy_log if entry['clipping_applied']) / len(self.privacy_log)
        
        return {
            'total_steps': len(self.privacy_log),
            'total_epsilon_consumed': total_epsilon,
            'average_noise_scale': avg_noise_scale,
            'clipping_rate': clipping_rate,
            'latest_delta': self.privacy_log[-1]['delta'] if self.privacy_log else None,
            'time_span_hours': (self.privacy_log[-1]['timestamp'] - self.privacy_log[0]['timestamp']) / 3600
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.performance_log:
            return {'status': 'no_data', 'total_operations': 0}
        
        operations = {}
        for entry in self.performance_log:
            op = entry['operation']
            if op not in operations:
                operations[op] = {'count': 0, 'total_time': 0, 'max_time': 0}
            
            operations[op]['count'] += 1
            operations[op]['total_time'] += entry['duration_ms']
            operations[op]['max_time'] = max(operations[op]['max_time'], entry['duration_ms'])
        
        # Calculate averages
        for op_stats in operations.values():
            op_stats['avg_time'] = op_stats['total_time'] / op_stats['count']
        
        return {
            'total_operations': len(self.performance_log),
            'operations': operations,
            'time_span_hours': (self.performance_log[-1]['timestamp'] - self.performance_log[0]['timestamp']) / 3600
        }
    
    def export_logs(self, filepath: str, format: str = 'json'):
        """Export logs to file."""
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'dp_flash_attention_version': '0.1.0',
                'privacy_steps': len(self.privacy_log),
                'performance_entries': len(self.performance_log),
                'security_events': len(self.security_events)
            },
            'privacy_log': self.privacy_log,
            'performance_log': self.performance_log,
            'security_events': self.security_events,
            'summaries': {
                'privacy': self.get_privacy_summary(),
                'performance': self.get_performance_summary()
            }
        }
        
        try:
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Logs exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
    
    def clear_logs(self, keep_recent: int = 0):
        """Clear stored logs, optionally keeping recent entries."""
        if keep_recent > 0:
            self.privacy_log = self.privacy_log[-keep_recent:]
            self.performance_log = self.performance_log[-keep_recent:]
            self.security_events = self.security_events[-keep_recent:]
        else:
            self.privacy_log.clear()
            self.performance_log.clear()
            self.security_events.clear()
        
        self.logger.info(f"Logs cleared (kept {keep_recent} recent entries)")


class PerformanceMonitor:
    """
    Context manager for monitoring performance of DP operations.
    
    Usage:
        with PerformanceMonitor('forward_pass') as monitor:
            output = model(input)
        
        print(f"Operation took {monitor.duration_ms:.2f}ms")
    """
    
    def __init__(
        self,
        operation_name: str,
        logger: Optional[PrivacyMetricsLogger] = None,
        log_memory: bool = False
    ):
        """
        Initialize performance monitor.
        
        Args:
            operation_name: Name of operation being monitored
            logger: Optional logger to record results
            log_memory: Whether to monitor memory usage
        """
        self.operation_name = operation_name
        self.logger = logger
        self.log_memory = log_memory and HAS_TORCH and torch.cuda.is_available()
        
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.duration_ms = None
        self.memory_used_mb = None
    
    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        
        if self.log_memory:
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring and log results."""
        if self.log_memory:
            torch.cuda.synchronize()
        
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if self.log_memory:
            self.end_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            self.memory_used_mb = self.end_memory - self.start_memory
        
        # Log to privacy logger if provided
        if self.logger:
            self.logger.log_performance_metrics(
                operation=self.operation_name,
                duration_ms=self.duration_ms,
                memory_usage_mb=self.memory_used_mb
            )


# Global logger instance
_global_logger = None


def get_logger(log_file: Optional[str] = None) -> PrivacyMetricsLogger:
    """Get global privacy metrics logger."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = PrivacyMetricsLogger(log_file)
    
    return _global_logger


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> PrivacyMetricsLogger:
    """
    Set up logging for DP-Flash-Attention.
    
    Args:
        level: Logging level
        log_file: Optional file to write logs to
        console_output: Whether to output to console
        
    Returns:
        Configured privacy metrics logger
    """
    global _global_logger
    
    _global_logger = PrivacyMetricsLogger(log_file, level)
    
    if not console_output:
        # Remove console handler
        _global_logger.logger.handlers = [
            h for h in _global_logger.logger.handlers 
            if not isinstance(h, logging.StreamHandler)
        ]
    
    return _global_logger