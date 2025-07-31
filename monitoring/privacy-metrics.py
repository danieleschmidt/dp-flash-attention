#!/usr/bin/env python3
"""
Privacy Metrics Collection and Monitoring for DP-Flash-Attention

This module provides comprehensive privacy budget tracking, 
violation detection, and compliance monitoring.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
import requests


@dataclass
class PrivacyMetrics:
    """Privacy metrics data structure"""
    timestamp: datetime
    epsilon_spent: float
    delta_spent: float
    epsilon_budget: float
    delta_budget: float
    mechanism: str
    session_id: str
    model_id: Optional[str] = None
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None


class PrivacyMonitor:
    """
    Privacy metrics collector and monitor
    
    Tracks privacy budget consumption, detects violations,
    and generates alerts for compliance monitoring.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Prometheus metrics
        self.privacy_budget_used = Gauge(
            'dp_privacy_budget_used',
            'Current privacy budget usage',
            ['mechanism', 'session_id']
        )
        
        self.privacy_violations = Counter(
            'dp_privacy_violations_total',
            'Total privacy budget violations',
            ['violation_type', 'mechanism']
        )
        
        self.privacy_queries = Counter(
            'dp_privacy_queries_total',
            'Total privacy queries processed',
            ['mechanism', 'status']
        )
        
        self.epsilon_histogram = Histogram(
            'dp_epsilon_distribution',
            'Distribution of epsilon values used',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        )
        
        self.privacy_latency = Summary(
            'dp_privacy_computation_seconds',
            'Time spent on privacy computations',
            ['mechanism']
        )
        
        # Internal tracking
        self.active_sessions: Dict[str, List[PrivacyMetrics]] = {}
        self.violation_threshold = self.config.get('violation_threshold', 0.95)
        self.alert_endpoint = self.config.get('alert_endpoint')
        
        # Background monitoring
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or environment"""
        default_config = {
            'log_level': 'INFO',
            'metrics_port': 8000,
            'violation_threshold': 0.95,
            'alert_endpoint': None,
            'retention_days': 30,
            'export_interval': 60
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
        
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config['log_level']))
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def record_privacy_usage(
        self,
        session_id: str,
        epsilon_spent: float,
        delta_spent: float,
        epsilon_budget: float,
        delta_budget: float,
        mechanism: str = "dp_flash_attention",
        **kwargs
    ) -> None:
        """Record privacy budget usage"""
        
        metrics = PrivacyMetrics(
            timestamp=datetime.utcnow(),
            epsilon_spent=epsilon_spent,
            delta_spent=delta_spent,
            epsilon_budget=epsilon_budget,
            delta_budget=delta_budget,
            mechanism=mechanism,
            session_id=session_id,
            **kwargs
        )
        
        # Store metrics
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        self.active_sessions[session_id].append(metrics)
        
        # Update Prometheus metrics
        budget_ratio = epsilon_spent / epsilon_budget if epsilon_budget > 0 else 1.0
        self.privacy_budget_used.labels(
            mechanism=mechanism,
            session_id=session_id
        ).set(budget_ratio)
        
        self.epsilon_histogram.observe(epsilon_spent)
        self.privacy_queries.labels(mechanism=mechanism, status='success').inc()
        
        # Check for violations
        self._check_violations(metrics)
        
        self.logger.info(
            f"Privacy usage recorded: session={session_id}, "
            f"epsilon={epsilon_spent:.4f}/{epsilon_budget:.4f}, "
            f"delta={delta_spent:.2e}/{delta_budget:.2e}"
        )
        
    def _check_violations(self, metrics: PrivacyMetrics) -> None:
        """Check for privacy budget violations"""
        
        epsilon_ratio = metrics.epsilon_spent / metrics.epsilon_budget if metrics.epsilon_budget > 0 else 1.0
        delta_ratio = metrics.delta_spent / metrics.delta_budget if metrics.delta_budget > 0 else 1.0
        
        violations = []
        
        if epsilon_ratio > 1.0:
            violations.append(('epsilon_exceeded', epsilon_ratio))
            self.privacy_violations.labels(
                violation_type='epsilon_exceeded',
                mechanism=metrics.mechanism
            ).inc()
            
        if delta_ratio > 1.0:
            violations.append(('delta_exceeded', delta_ratio))
            self.privacy_violations.labels(
                violation_type='delta_exceeded',
                mechanism=metrics.mechanism
            ).inc()
            
        if epsilon_ratio > self.violation_threshold:
            violations.append(('epsilon_warning', epsilon_ratio))
            self.privacy_violations.labels(
                violation_type='epsilon_warning',
                mechanism=metrics.mechanism
            ).inc()
            
        # Send alerts for violations
        for violation_type, ratio in violations:
            self._send_alert(violation_type, metrics, ratio)
            
    def _send_alert(
        self,
        violation_type: str,
        metrics: PrivacyMetrics,
        ratio: float
    ) -> None:
        """Send privacy violation alert"""
        
        alert_data = {
            'timestamp': metrics.timestamp.isoformat(),
            'violation_type': violation_type,
            'session_id': metrics.session_id,
            'mechanism': metrics.mechanism,
            'ratio': ratio,
            'severity': 'critical' if ratio > 1.0 else 'warning',
            'details': asdict(metrics)
        }
        
        self.logger.error(f"Privacy violation detected: {alert_data}")
        
        if self.alert_endpoint:
            try:
                response = requests.post(
                    self.alert_endpoint,
                    json=alert_data,
                    timeout=10
                )
                response.raise_for_status()
            except Exception as e:
                self.logger.error(f"Failed to send alert: {e}")
                
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get privacy usage summary for a session"""
        
        if session_id not in self.active_sessions:
            return None
            
        metrics_list = self.active_sessions[session_id]
        if not metrics_list:
            return None
            
        # Calculate totals and statistics
        total_epsilon = sum(m.epsilon_spent for m in metrics_list)
        total_delta = sum(m.delta_spent for m in metrics_list)
        
        latest_metrics = metrics_list[-1]
        
        return {
            'session_id': session_id,
            'total_queries': len(metrics_list),
            'total_epsilon_spent': total_epsilon,
            'total_delta_spent': total_delta,
            'epsilon_budget': latest_metrics.epsilon_budget,
            'delta_budget': latest_metrics.delta_budget,
            'epsilon_remaining': max(0, latest_metrics.epsilon_budget - total_epsilon),
            'delta_remaining': max(0, latest_metrics.delta_budget - total_delta),
            'start_time': metrics_list[0].timestamp.isoformat(),
            'last_activity': latest_metrics.timestamp.isoformat(),
            'mechanisms_used': list(set(m.mechanism for m in metrics_list))
        }
        
    def start_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
            
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start Prometheus metrics server
        prometheus_client.start_http_server(self.config['metrics_port'])
        
        self.logger.info(f"Privacy monitoring started on port {self.config['metrics_port']}")
        
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        
        while not self.shutdown_event.is_set():
            try:
                # Clean up old sessions
                self._cleanup_old_sessions()
                
                # Export metrics
                self._export_metrics()
                
                # Health check
                self._health_check()
                
                time.sleep(self.config['export_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
                
    def _cleanup_old_sessions(self) -> None:
        """Remove old session data"""
        
        cutoff_time = datetime.utcnow() - timedelta(days=self.config['retention_days'])
        sessions_to_remove = []
        
        for session_id, metrics_list in self.active_sessions.items():
            # Remove old metrics within session
            self.active_sessions[session_id] = [
                m for m in metrics_list if m.timestamp > cutoff_time
            ]
            
            # Mark empty sessions for removal
            if not self.active_sessions[session_id]:
                sessions_to_remove.append(session_id)
                
        # Remove empty sessions
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            self.logger.info(f"Cleaned up old session: {session_id}")
            
    def _export_metrics(self) -> None:
        """Export current metrics state"""
        
        metrics_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'active_sessions': len(self.active_sessions),
            'total_queries': sum(len(metrics) for metrics in self.active_sessions.values()),
            'sessions': {
                session_id: self.get_session_summary(session_id)
                for session_id in self.active_sessions.keys()
            }
        }
        
        # Write to file for persistence
        metrics_file = Path('metrics/privacy_summary.json')
        metrics_file.parent.mkdir(exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
            
    def _health_check(self) -> None:
        """Perform health check"""
        
        # Check for sessions approaching budget limits
        for session_id in self.active_sessions:
            summary = self.get_session_summary(session_id)
            if summary:
                epsilon_ratio = (
                    summary['total_epsilon_spent'] / summary['epsilon_budget']
                    if summary['epsilon_budget'] > 0 else 1.0
                )
                
                if epsilon_ratio > self.violation_threshold:
                    self.logger.warning(
                        f"Session {session_id} approaching epsilon budget limit: "
                        f"{epsilon_ratio:.2%}"
                    )
                    
    def shutdown(self) -> None:
        """Graceful shutdown"""
        self.logger.info("Shutting down privacy monitor...")
        self.shutdown_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        # Final metrics export
        self._export_metrics()


def main():
    """Main entry point for privacy monitoring service"""
    
    monitor = PrivacyMonitor()
    
    try:
        monitor.start_monitoring()
        
        # Keep the service running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        monitor.shutdown()
        
    except Exception as e:
        monitor.logger.error(f"Fatal error: {e}")
        monitor.shutdown()
        raise


if __name__ == '__main__':
    main()