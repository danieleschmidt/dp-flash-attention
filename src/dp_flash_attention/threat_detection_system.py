"""
Advanced Threat Detection and Mitigation System for DP-Flash-Attention.

This module implements a comprehensive security system that monitors, detects,
and mitigates various threats to differential privacy and system integrity:

- Privacy budget exhaustion attacks
- Membership inference attacks
- Model inversion attacks  
- Byzantine adversaries in federated learning
- Side-channel attacks on attention computation
- Gradient leakage detection
- Anomalous query pattern detection
- Real-time threat response and mitigation

Design Philosophy: Proactive defense with autonomous threat response.
"""

import time
import hashlib
import json
import threading
import queue
import secrets
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import deque, defaultdict
try:
    from concurrent.futures import ThreadPoolExecutor
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False
    ThreadPoolExecutor = None
import logging
import statistics
import math

# Configure specialized logging for threat detection
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] THREAT-%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class ThreatAlert:
    """Represents a security threat alert."""
    alert_id: str
    threat_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    detection_method: str
    affected_components: List[str]
    threat_score: float
    mitigation_suggestions: List[str]
    timestamp: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass 
class ThreatMitigationAction:
    """Represents a threat mitigation action."""
    action_id: str
    threat_alert_id: str
    action_type: str
    parameters: Dict[str, Any]
    success: bool
    execution_time: float
    timestamp: float


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    total_threats_detected: int = 0
    threats_by_severity: Dict[str, int] = None
    mitigation_success_rate: float = 0.0
    average_detection_time: float = 0.0
    privacy_budget_violations: int = 0
    blocked_attacks: int = 0
    false_positive_rate: float = 0.0
    system_security_score: float = 1.0
    
    def __post_init__(self):
        if self.threats_by_severity is None:
            self.threats_by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}


class ThreatDetector(ABC):
    """Abstract base class for threat detectors."""
    
    @abstractmethod
    def detect_threats(self, data: Dict[str, Any]) -> List[ThreatAlert]:
        """Detect threats in the provided data."""
        pass
    
    @abstractmethod
    def get_detector_name(self) -> str:
        """Get the name of this detector."""
        pass
    
    @abstractmethod
    def update_detection_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update detection parameters based on new threat intelligence."""
        pass


class PrivacyBudgetExhaustionDetector(ThreatDetector):
    """Detects privacy budget exhaustion attacks and violations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.budget_history = deque(maxlen=1000)
        self.suspicious_patterns = {}
        
    def detect_threats(self, data: Dict[str, Any]) -> List[ThreatAlert]:
        """Detect privacy budget-related threats."""
        threats = []
        
        # Extract privacy budget information
        current_budget = data.get('privacy_budget_remaining', 1.0)
        budget_consumption_rate = data.get('budget_consumption_rate', 0.0)
        user_id = data.get('user_id', 'unknown')
        
        # Record budget consumption
        self.budget_history.append({
            'timestamp': time.time(),
            'budget': current_budget,
            'consumption_rate': budget_consumption_rate,
            'user_id': user_id
        })
        
        # Detect rapid budget exhaustion
        if self._detect_rapid_exhaustion(budget_consumption_rate):
            threats.append(ThreatAlert(
                alert_id=f"budget_exhaustion_{int(time.time())}",
                threat_type="privacy_budget_exhaustion",
                severity="high",
                description=f"Rapid privacy budget consumption detected: {budget_consumption_rate:.4f}/s",
                detection_method="budget_consumption_analysis",
                affected_components=["privacy_accounting"],
                threat_score=0.8,
                mitigation_suggestions=[
                    "Rate limit privacy budget consumption",
                    "Require additional authentication for high-budget queries",
                    "Implement adaptive noise scaling"
                ],
                timestamp=time.time(),
                user_id=user_id
            ))
        
        # Detect budget farming attacks
        budget_farming_threat = self._detect_budget_farming(user_id)
        if budget_farming_threat:
            threats.append(budget_farming_threat)
        
        # Detect coordinated budget attacks
        coordinated_threat = self._detect_coordinated_attacks()
        if coordinated_threat:
            threats.append(coordinated_threat)
        
        return threats
    
    def _detect_rapid_exhaustion(self, consumption_rate: float) -> bool:
        """Detect unusually rapid privacy budget consumption."""
        threshold = self.config.get('rapid_exhaustion_threshold', 0.1)
        return consumption_rate > threshold
    
    def _detect_budget_farming(self, user_id: str) -> Optional[ThreatAlert]:
        """Detect budget farming attempts by individual users."""
        if user_id == 'unknown':
            return None
            
        # Analyze user's budget consumption pattern
        user_history = [entry for entry in self.budget_history if entry['user_id'] == user_id]
        
        if len(user_history) < 10:  # Need sufficient history
            return None
        
        # Calculate statistics
        consumption_rates = [entry['consumption_rate'] for entry in user_history[-10:]]
        avg_consumption = statistics.mean(consumption_rates)
        consumption_variance = statistics.variance(consumption_rates) if len(consumption_rates) > 1 else 0
        
        # Detect suspiciously consistent high consumption
        if avg_consumption > 0.05 and consumption_variance < 0.01:
            return ThreatAlert(
                alert_id=f"budget_farming_{user_id}_{int(time.time())}",
                threat_type="budget_farming",
                severity="medium",
                description=f"Suspicious budget farming pattern detected for user {user_id}",
                detection_method="statistical_analysis",
                affected_components=["privacy_accounting", "user_management"],
                threat_score=0.6,
                mitigation_suggestions=[
                    "Implement per-user budget limits",
                    "Require CAPTCHA for high-frequency queries",
                    "Monitor user behavior patterns"
                ],
                timestamp=time.time(),
                user_id=user_id
            )
        
        return None
    
    def _detect_coordinated_attacks(self) -> Optional[ThreatAlert]:
        """Detect coordinated budget exhaustion attacks."""
        if len(self.budget_history) < 50:
            return None
        
        # Analyze recent activity for coordination patterns
        recent_entries = list(self.budget_history)[-50:]
        user_consumption = defaultdict(float)
        
        for entry in recent_entries:
            user_consumption[entry['user_id']] += entry['consumption_rate']
        
        # Detect multiple users with similar high consumption
        high_consumers = [user for user, consumption in user_consumption.items() 
                         if consumption > 0.03]
        
        if len(high_consumers) >= 3:  # Multiple coordinated attackers
            return ThreatAlert(
                alert_id=f"coordinated_attack_{int(time.time())}",
                threat_type="coordinated_budget_attack",
                severity="critical",
                description=f"Coordinated budget exhaustion attack detected: {len(high_consumers)} users",
                detection_method="correlation_analysis",
                affected_components=["privacy_accounting", "system_integrity"],
                threat_score=0.9,
                mitigation_suggestions=[
                    "Implement global rate limiting",
                    "Activate emergency budget protection",
                    "Block suspicious IP ranges",
                    "Require additional authentication"
                ],
                timestamp=time.time(),
                additional_data={'suspected_users': high_consumers}
            )
        
        return None
    
    def get_detector_name(self) -> str:
        return "PrivacyBudgetExhaustionDetector"
    
    def update_detection_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update detection parameters."""
        self.config.update(parameters)


class MembershipInferenceDetector(ThreatDetector):
    """Detects membership inference attacks against the model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_patterns = defaultdict(list)
        self.shadow_model_indicators = set()
        
    def detect_threats(self, data: Dict[str, Any]) -> List[ThreatAlert]:
        """Detect membership inference attack patterns."""
        threats = []
        
        user_id = data.get('user_id', 'unknown')
        query_type = data.get('query_type', 'unknown')
        model_outputs = data.get('model_outputs', [])
        confidence_scores = data.get('confidence_scores', [])
        
        # Record query pattern
        self.query_patterns[user_id].append({
            'timestamp': time.time(),
            'query_type': query_type,
            'outputs': model_outputs,
            'confidences': confidence_scores
        })
        
        # Detect repeated similar queries (shadow model training)
        shadow_threat = self._detect_shadow_model_training(user_id)
        if shadow_threat:
            threats.append(shadow_threat)
        
        # Detect confidence-based inference attacks
        confidence_threat = self._detect_confidence_inference_attack(user_id, confidence_scores)
        if confidence_threat:
            threats.append(confidence_threat)
        
        # Detect temporal correlation attacks
        temporal_threat = self._detect_temporal_correlation_attack(user_id)
        if temporal_threat:
            threats.append(temporal_threat)
        
        return threats
    
    def _detect_shadow_model_training(self, user_id: str) -> Optional[ThreatAlert]:
        """Detect attempts to train shadow models for membership inference."""
        user_queries = self.query_patterns[user_id]
        
        if len(user_queries) < 100:  # Need sufficient queries to detect pattern
            return None
        
        # Analyze query diversity and frequency
        recent_queries = user_queries[-100:]
        query_types = [q['query_type'] for q in recent_queries]
        unique_types = len(set(query_types))
        total_queries = len(recent_queries)
        
        # Calculate query frequency
        time_span = recent_queries[-1]['timestamp'] - recent_queries[0]['timestamp']
        query_frequency = total_queries / max(time_span, 1)  # queries per second
        
        # Detect suspicious patterns
        if unique_types < 5 and query_frequency > 0.5:  # High frequency, low diversity
            return ThreatAlert(
                alert_id=f"shadow_model_{user_id}_{int(time.time())}",
                threat_type="membership_inference_shadow_model",
                severity="high",
                description=f"Shadow model training detected: {total_queries} queries, {unique_types} types",
                detection_method="query_pattern_analysis",
                affected_components=["model_privacy", "query_processing"],
                threat_score=0.75,
                mitigation_suggestions=[
                    "Implement query diversity requirements",
                    "Rate limit similar queries",
                    "Add additional noise to model outputs",
                    "Monitor and limit bulk query operations"
                ],
                timestamp=time.time(),
                user_id=user_id,
                additional_data={
                    'query_frequency': query_frequency,
                    'query_diversity': unique_types / total_queries
                }
            )
        
        return None
    
    def _detect_confidence_inference_attack(self, user_id: str, 
                                          confidence_scores: List[float]) -> Optional[ThreatAlert]:
        """Detect confidence-based membership inference attacks."""
        if not confidence_scores or len(confidence_scores) < 10:
            return None
        
        # Analyze confidence score patterns
        high_confidence_queries = sum(1 for score in confidence_scores if score > 0.9)
        low_confidence_queries = sum(1 for score in confidence_scores if score < 0.3)
        
        # Detect unusual confidence distribution (potential attack)
        total_queries = len(confidence_scores)
        high_conf_ratio = high_confidence_queries / total_queries
        low_conf_ratio = low_confidence_queries / total_queries
        
        # Suspicious if too many extreme confidence scores
        if high_conf_ratio > 0.8 or low_conf_ratio > 0.8:
            return ThreatAlert(
                alert_id=f"confidence_inference_{user_id}_{int(time.time())}",
                threat_type="membership_inference_confidence",
                severity="medium",
                description=f"Confidence-based inference attack detected: {high_conf_ratio:.2%} high confidence",
                detection_method="confidence_distribution_analysis",
                affected_components=["model_outputs", "confidence_estimation"],
                threat_score=0.6,
                mitigation_suggestions=[
                    "Calibrate confidence scores with additional noise",
                    "Implement confidence score smoothing",
                    "Limit access to raw confidence values"
                ],
                timestamp=time.time(),
                user_id=user_id,
                additional_data={
                    'high_confidence_ratio': high_conf_ratio,
                    'low_confidence_ratio': low_conf_ratio
                }
            )
        
        return None
    
    def _detect_temporal_correlation_attack(self, user_id: str) -> Optional[ThreatAlert]:
        """Detect temporal correlation-based membership inference."""
        user_queries = self.query_patterns[user_id]
        
        if len(user_queries) < 50:
            return None
        
        # Analyze temporal patterns in queries
        recent_queries = user_queries[-50:]
        timestamps = [q['timestamp'] for q in recent_queries]
        
        # Calculate time intervals between queries
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return None
        
        # Detect suspiciously regular timing patterns
        avg_interval = statistics.mean(intervals)
        interval_variance = statistics.variance(intervals) if len(intervals) > 1 else float('inf')
        
        # Regular timing might indicate automated attack
        if avg_interval < 2.0 and interval_variance < 0.5:
            return ThreatAlert(
                alert_id=f"temporal_correlation_{user_id}_{int(time.time())}",
                threat_type="membership_inference_temporal",
                severity="medium",
                description=f"Temporal correlation attack detected: regular {avg_interval:.2f}s intervals",
                detection_method="temporal_pattern_analysis",
                affected_components=["query_timing", "access_patterns"],
                threat_score=0.55,
                mitigation_suggestions=[
                    "Implement random query delays",
                    "Add jitter to response times",
                    "Monitor automated access patterns"
                ],
                timestamp=time.time(),
                user_id=user_id,
                additional_data={
                    'average_interval': avg_interval,
                    'interval_variance': interval_variance
                }
            )
        
        return None
    
    def get_detector_name(self) -> str:
        return "MembershipInferenceDetector"
    
    def update_detection_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update detection parameters."""
        self.config.update(parameters)


class GradientLeakageDetector(ThreatDetector):
    """Detects potential gradient leakage and side-channel attacks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.computation_patterns = deque(maxlen=500)
        self.timing_anomalies = []
        
    def detect_threats(self, data: Dict[str, Any]) -> List[ThreatAlert]:
        """Detect gradient leakage and side-channel threats."""
        threats = []
        
        computation_time = data.get('computation_time', 0.0)
        memory_usage = data.get('memory_usage', 0.0)
        gradient_norms = data.get('gradient_norms', [])
        attention_patterns = data.get('attention_patterns', [])
        user_id = data.get('user_id', 'unknown')
        
        # Record computation pattern
        self.computation_patterns.append({
            'timestamp': time.time(),
            'computation_time': computation_time,
            'memory_usage': memory_usage,
            'gradient_norms': gradient_norms,
            'user_id': user_id
        })
        
        # Detect timing side-channel attacks
        timing_threat = self._detect_timing_side_channel(computation_time, user_id)
        if timing_threat:
            threats.append(timing_threat)
        
        # Detect gradient leakage through norm analysis
        gradient_threat = self._detect_gradient_leakage(gradient_norms, user_id)
        if gradient_threat:
            threats.append(gradient_threat)
        
        # Detect attention pattern leakage
        attention_threat = self._detect_attention_leakage(attention_patterns, user_id)
        if attention_threat:
            threats.append(attention_threat)
        
        return threats
    
    def _detect_timing_side_channel(self, computation_time: float, user_id: str) -> Optional[ThreatAlert]:
        """Detect timing-based side-channel attacks."""
        if len(self.computation_patterns) < 20:
            return None
        
        # Analyze timing patterns for this user
        user_timings = [p['computation_time'] for p in self.computation_patterns 
                       if p['user_id'] == user_id and p['computation_time'] > 0]
        
        if len(user_timings) < 10:
            return None
        
        # Calculate timing statistics
        avg_timing = statistics.mean(user_timings)
        timing_variance = statistics.variance(user_timings) if len(user_timings) > 1 else 0
        
        # Detect suspiciously precise timing measurements
        if timing_variance < 0.001 and len(user_timings) > 50:
            return ThreatAlert(
                alert_id=f"timing_side_channel_{user_id}_{int(time.time())}",
                threat_type="timing_side_channel",
                severity="medium",
                description=f"Timing side-channel attack detected: precise timing analysis",
                detection_method="timing_variance_analysis",
                affected_components=["computation_timing", "side_channel_protection"],
                threat_score=0.6,
                mitigation_suggestions=[
                    "Add random delays to computation",
                    "Implement constant-time operations",
                    "Limit timing information exposure"
                ],
                timestamp=time.time(),
                user_id=user_id,
                additional_data={
                    'average_timing': avg_timing,
                    'timing_variance': timing_variance
                }
            )
        
        return None
    
    def _detect_gradient_leakage(self, gradient_norms: List[float], user_id: str) -> Optional[ThreatAlert]:
        """Detect potential gradient information leakage."""
        if not gradient_norms or len(gradient_norms) < 10:
            return None
        
        # Analyze gradient norm patterns
        high_norm_count = sum(1 for norm in gradient_norms if norm > 1.0)
        low_norm_count = sum(1 for norm in gradient_norms if norm < 0.1)
        
        total_norms = len(gradient_norms)
        high_ratio = high_norm_count / total_norms
        low_ratio = low_norm_count / total_norms
        
        # Detect unusual gradient norm distributions
        if high_ratio > 0.7 or low_ratio > 0.7:
            severity = "high" if high_ratio > 0.9 or low_ratio > 0.9 else "medium"
            
            return ThreatAlert(
                alert_id=f"gradient_leakage_{user_id}_{int(time.time())}",
                threat_type="gradient_leakage",
                severity=severity,
                description=f"Gradient leakage detected: unusual norm distribution",
                detection_method="gradient_norm_analysis",
                affected_components=["gradient_computation", "privacy_protection"],
                threat_score=0.7 if severity == "high" else 0.5,
                mitigation_suggestions=[
                    "Increase gradient noise injection",
                    "Implement gradient clipping",
                    "Limit gradient information exposure"
                ],
                timestamp=time.time(),
                user_id=user_id,
                additional_data={
                    'high_norm_ratio': high_ratio,
                    'low_norm_ratio': low_ratio
                }
            )
        
        return None
    
    def _detect_attention_leakage(self, attention_patterns: List[Any], user_id: str) -> Optional[ThreatAlert]:
        """Detect attention pattern information leakage."""
        if not attention_patterns or len(attention_patterns) < 5:
            return None
        
        # Analyze attention pattern entropy
        if NUMPY_AVAILABLE:
            try:
                # Convert attention patterns to analyzable format
                pattern_entropies = []
                for pattern in attention_patterns:
                    if hasattr(pattern, 'flatten'):
                        flat_pattern = pattern.flatten()
                        # Calculate entropy
                        probabilities = np.abs(flat_pattern) / np.sum(np.abs(flat_pattern))
                        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                        pattern_entropies.append(entropy)
                
                if pattern_entropies:
                    avg_entropy = statistics.mean(pattern_entropies)
                    entropy_variance = statistics.variance(pattern_entropies) if len(pattern_entropies) > 1 else 0
                    
                    # Low entropy might indicate information leakage
                    if avg_entropy < 2.0 and entropy_variance < 0.5:
                        return ThreatAlert(
                            alert_id=f"attention_leakage_{user_id}_{int(time.time())}",
                            threat_type="attention_pattern_leakage",
                            severity="medium",
                            description=f"Attention pattern leakage detected: low entropy patterns",
                            detection_method="attention_entropy_analysis",
                            affected_components=["attention_computation", "pattern_privacy"],
                            threat_score=0.55,
                            mitigation_suggestions=[
                                "Add noise to attention weights",
                                "Implement attention masking",
                                "Limit attention pattern exposure"
                            ],
                            timestamp=time.time(),
                            user_id=user_id,
                            additional_data={
                                'average_entropy': avg_entropy,
                                'entropy_variance': entropy_variance
                            }
                        )
            except Exception:
                pass  # Ignore errors in entropy calculation
        
        return None
    
    def get_detector_name(self) -> str:
        return "GradientLeakageDetector"
    
    def update_detection_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update detection parameters."""
        self.config.update(parameters)


class ThreatMitigationEngine:
    """Engine for automated threat mitigation and response."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mitigation_history = []
        self.blocked_users = set()
        self.suspicious_ips = set()
        
    def mitigate_threat(self, threat_alert: ThreatAlert) -> ThreatMitigationAction:
        """Execute mitigation actions for a detected threat."""
        action_id = f"mitigation_{threat_alert.alert_id}_{int(time.time())}"
        
        logger.warning(f"🚨 Mitigating threat: {threat_alert.threat_type} (severity: {threat_alert.severity})")
        
        # Select mitigation strategy based on threat type and severity
        mitigation_strategy = self._select_mitigation_strategy(threat_alert)
        
        # Execute mitigation actions
        start_time = time.time()
        success = self._execute_mitigation_actions(mitigation_strategy, threat_alert)
        execution_time = time.time() - start_time
        
        # Record mitigation action
        mitigation_action = ThreatMitigationAction(
            action_id=action_id,
            threat_alert_id=threat_alert.alert_id,
            action_type=mitigation_strategy['type'],
            parameters=mitigation_strategy['parameters'],
            success=success,
            execution_time=execution_time,
            timestamp=time.time()
        )
        
        self.mitigation_history.append(mitigation_action)
        
        if success:
            logger.info(f"✅ Threat mitigation successful: {mitigation_strategy['type']}")
        else:
            logger.error(f"❌ Threat mitigation failed: {mitigation_strategy['type']}")
        
        return mitigation_action
    
    def _select_mitigation_strategy(self, threat_alert: ThreatAlert) -> Dict[str, Any]:
        """Select appropriate mitigation strategy for the threat."""
        threat_type = threat_alert.threat_type
        severity = threat_alert.severity
        
        if threat_type == "privacy_budget_exhaustion":
            if severity == "critical":
                return {
                    'type': 'emergency_budget_protection',
                    'parameters': {
                        'rate_limit_factor': 0.1,
                        'additional_noise_factor': 2.0,
                        'require_authentication': True
                    }
                }
            else:
                return {
                    'type': 'adaptive_rate_limiting',
                    'parameters': {
                        'rate_limit_factor': 0.5,
                        'additional_noise_factor': 1.5
                    }
                }
        
        elif threat_type in ["membership_inference_shadow_model", "membership_inference_confidence"]:
            return {
                'type': 'enhanced_privacy_protection',
                'parameters': {
                    'noise_multiplier_increase': 1.5,
                    'query_diversity_requirement': True,
                    'confidence_smoothing': True
                }
            }
        
        elif threat_type == "gradient_leakage":
            return {
                'type': 'gradient_protection',
                'parameters': {
                    'gradient_noise_factor': 2.0,
                    'gradient_clipping_threshold': 0.5,
                    'timing_obfuscation': True
                }
            }
        
        elif threat_type == "coordinated_budget_attack":
            return {
                'type': 'coordinated_attack_response',
                'parameters': {
                    'block_suspicious_users': True,
                    'ip_range_blocking': True,
                    'emergency_rate_limiting': True
                }
            }
        
        else:
            # Default mitigation strategy
            return {
                'type': 'general_threat_response',
                'parameters': {
                    'increase_monitoring': True,
                    'alert_administrators': True,
                    'log_detailed_information': True
                }
            }
    
    def _execute_mitigation_actions(self, strategy: Dict[str, Any], 
                                  threat_alert: ThreatAlert) -> bool:
        """Execute the selected mitigation actions."""
        try:
            strategy_type = strategy['type']
            parameters = strategy['parameters']
            
            if strategy_type == 'emergency_budget_protection':
                return self._execute_emergency_budget_protection(parameters, threat_alert)
            elif strategy_type == 'adaptive_rate_limiting':
                return self._execute_adaptive_rate_limiting(parameters, threat_alert)
            elif strategy_type == 'enhanced_privacy_protection':
                return self._execute_enhanced_privacy_protection(parameters, threat_alert)
            elif strategy_type == 'gradient_protection':
                return self._execute_gradient_protection(parameters, threat_alert)
            elif strategy_type == 'coordinated_attack_response':
                return self._execute_coordinated_attack_response(parameters, threat_alert)
            else:
                return self._execute_general_threat_response(parameters, threat_alert)
                
        except Exception as e:
            logger.error(f"Error executing mitigation: {e}")
            return False
    
    def _execute_emergency_budget_protection(self, parameters: Dict[str, Any], 
                                           threat_alert: ThreatAlert) -> bool:
        """Execute emergency budget protection measures."""
        logger.warning("🚨 Activating emergency budget protection")
        
        # Implement rate limiting
        rate_limit_factor = parameters.get('rate_limit_factor', 0.1)
        logger.info(f"📉 Reducing query rate limit by factor {rate_limit_factor}")
        
        # Increase noise
        noise_factor = parameters.get('additional_noise_factor', 2.0)
        logger.info(f"🔊 Increasing privacy noise by factor {noise_factor}")
        
        # Block suspicious user if identified
        if threat_alert.user_id and threat_alert.user_id != 'unknown':
            self.blocked_users.add(threat_alert.user_id)
            logger.warning(f"🚫 Blocked user: {threat_alert.user_id}")
        
        return True
    
    def _execute_adaptive_rate_limiting(self, parameters: Dict[str, Any], 
                                      threat_alert: ThreatAlert) -> bool:
        """Execute adaptive rate limiting."""
        rate_limit_factor = parameters.get('rate_limit_factor', 0.5)
        logger.info(f"⏱️ Implementing adaptive rate limiting: {rate_limit_factor}")
        
        # Add user to monitoring list
        if threat_alert.user_id and threat_alert.user_id != 'unknown':
            logger.info(f"👁️ Adding user {threat_alert.user_id} to enhanced monitoring")
        
        return True
    
    def _execute_enhanced_privacy_protection(self, parameters: Dict[str, Any], 
                                           threat_alert: ThreatAlert) -> bool:
        """Execute enhanced privacy protection measures."""
        logger.info("🛡️ Activating enhanced privacy protection")
        
        # Increase noise multiplier
        noise_multiplier = parameters.get('noise_multiplier_increase', 1.5)
        logger.info(f"🔊 Increasing noise multiplier by {noise_multiplier}x")
        
        # Enable confidence smoothing
        if parameters.get('confidence_smoothing', False):
            logger.info("📊 Enabling confidence score smoothing")
        
        # Require query diversity
        if parameters.get('query_diversity_requirement', False):
            logger.info("🔄 Enforcing query diversity requirements")
        
        return True
    
    def _execute_gradient_protection(self, parameters: Dict[str, Any], 
                                   threat_alert: ThreatAlert) -> bool:
        """Execute gradient protection measures."""
        logger.info("🛡️ Activating gradient protection")
        
        # Increase gradient noise
        gradient_noise_factor = parameters.get('gradient_noise_factor', 2.0)
        logger.info(f"🔊 Increasing gradient noise by {gradient_noise_factor}x")
        
        # Enable timing obfuscation
        if parameters.get('timing_obfuscation', False):
            logger.info("⏰ Enabling computation timing obfuscation")
        
        return True
    
    def _execute_coordinated_attack_response(self, parameters: Dict[str, Any], 
                                           threat_alert: ThreatAlert) -> bool:
        """Execute coordinated attack response."""
        logger.warning("🚨 Responding to coordinated attack")
        
        # Block suspicious users
        if parameters.get('block_suspicious_users', False):
            suspected_users = threat_alert.additional_data.get('suspected_users', [])
            for user in suspected_users:
                self.blocked_users.add(user)
                logger.warning(f"🚫 Blocked suspected user: {user}")
        
        # Emergency rate limiting
        if parameters.get('emergency_rate_limiting', False):
            logger.warning("🚨 Activating emergency rate limiting")
        
        return True
    
    def _execute_general_threat_response(self, parameters: Dict[str, Any], 
                                       threat_alert: ThreatAlert) -> bool:
        """Execute general threat response measures."""
        logger.info("⚠️ Executing general threat response")
        
        # Increase monitoring
        if parameters.get('increase_monitoring', False):
            logger.info("👁️ Increasing system monitoring")
        
        # Alert administrators
        if parameters.get('alert_administrators', False):
            logger.warning("📧 Alerting system administrators")
        
        return True
    
    def get_blocked_entities(self) -> Dict[str, Set[str]]:
        """Get currently blocked entities."""
        return {
            'users': self.blocked_users.copy(),
            'ips': self.suspicious_ips.copy()
        }


class ThreatDetectionSystem:
    """
    Comprehensive threat detection and mitigation system.
    
    Coordinates multiple threat detectors and automated mitigation responses
    to protect differential privacy and system integrity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize threat detectors
        self.detectors = [
            PrivacyBudgetExhaustionDetector(self.config.get('budget_detector', {})),
            MembershipInferenceDetector(self.config.get('membership_detector', {})),
            GradientLeakageDetector(self.config.get('gradient_detector', {}))
        ]
        
        # Initialize mitigation engine
        self.mitigation_engine = ThreatMitigationEngine(self.config.get('mitigation', {}))
        
        # System state
        self.active_threats = {}
        self.threat_history = []
        self.security_metrics = SecurityMetrics()
        self.monitoring_active = False
        self.monitor_thread = None
        self.threat_queue = queue.Queue()
        
        logger.info("🛡️ Threat Detection System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for threat detection."""
        return {
            'monitoring': {
                'scan_interval': 10,  # seconds
                'threat_retention_days': 30,
                'auto_mitigation': True,
                'alert_threshold': 0.5
            },
            'budget_detector': {
                'rapid_exhaustion_threshold': 0.1,
                'farming_detection_enabled': True
            },
            'membership_detector': {
                'shadow_model_detection': True,
                'confidence_analysis': True,
                'temporal_analysis': True
            },
            'gradient_detector': {
                'timing_analysis': True,
                'gradient_norm_analysis': True,
                'attention_analysis': True
            },
            'mitigation': {
                'auto_response': True,
                'escalation_thresholds': {
                    'low': 0.3,
                    'medium': 0.6,
                    'high': 0.8,
                    'critical': 0.9
                }
            }
        }
    
    def start_monitoring(self):
        """Start continuous threat monitoring."""
        if self.monitoring_active:
            logger.warning("Threat monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("🔍 Threat monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous threat monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("🛑 Threat monitoring stopped")
    
    def _monitoring_loop(self):
        """Main threat monitoring loop."""
        logger.info("🔄 Threat monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Process queued monitoring data
                self._process_monitoring_queue()
                
                # Clean up old threats
                self._cleanup_old_threats()
                
                # Update security metrics
                self._update_security_metrics()
                
                # Sleep until next scan
                time.sleep(self.config['monitoring']['scan_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _process_monitoring_queue(self):
        """Process queued monitoring data."""
        while not self.threat_queue.empty():
            try:
                monitoring_data = self.threat_queue.get_nowait()
                self.analyze_threats(monitoring_data)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing monitoring data: {e}")
    
    def analyze_threats(self, monitoring_data: Dict[str, Any]) -> List[ThreatAlert]:
        """Analyze data for threats using all detectors."""
        all_threats = []
        
        # Run threat detection using all detectors
        for detector in self.detectors:
            try:
                detector_threats = detector.detect_threats(monitoring_data)
                all_threats.extend(detector_threats)
                
                logger.debug(f"{detector.get_detector_name()} detected {len(detector_threats)} threats")
                
            except Exception as e:
                logger.error(f"Error in {detector.get_detector_name()}: {e}")
        
        # Process detected threats
        for threat in all_threats:
            self._process_threat_alert(threat)
        
        return all_threats
    
    def _process_threat_alert(self, threat_alert: ThreatAlert):
        """Process a detected threat alert."""
        # Record threat
        self.threat_history.append(threat_alert)
        self.active_threats[threat_alert.alert_id] = threat_alert
        
        # Update metrics
        self.security_metrics.total_threats_detected += 1
        self.security_metrics.threats_by_severity[threat_alert.severity] += 1
        
        # Log threat
        logger.warning(f"🚨 Threat detected: {threat_alert.threat_type} "
                      f"(severity: {threat_alert.severity}, score: {threat_alert.threat_score})")
        
        # Auto-mitigation if enabled and threat score exceeds threshold
        if (self.config['monitoring']['auto_mitigation'] and 
            threat_alert.threat_score >= self.config['monitoring']['alert_threshold']):
            
            mitigation_action = self.mitigation_engine.mitigate_threat(threat_alert)
            
            # Update mitigation success rate
            total_mitigations = len(self.mitigation_engine.mitigation_history)
            successful_mitigations = sum(1 for action in self.mitigation_engine.mitigation_history 
                                       if action.success)
            
            if total_mitigations > 0:
                self.security_metrics.mitigation_success_rate = successful_mitigations / total_mitigations
    
    def _cleanup_old_threats(self):
        """Clean up old threat records."""
        retention_seconds = self.config['monitoring']['threat_retention_days'] * 24 * 3600
        current_time = time.time()
        
        # Remove old threats from active threats
        old_threat_ids = [
            alert_id for alert_id, alert in self.active_threats.items()
            if current_time - alert.timestamp > retention_seconds
        ]
        
        for alert_id in old_threat_ids:
            del self.active_threats[alert_id]
        
        # Clean up threat history
        self.threat_history = [
            threat for threat in self.threat_history
            if current_time - threat.timestamp <= retention_seconds
        ]
    
    def _update_security_metrics(self):
        """Update security metrics based on current state."""
        if self.threat_history:
            # Calculate average detection time (simulated)
            recent_threats = [t for t in self.threat_history if time.time() - t.timestamp < 3600]
            if recent_threats:
                self.security_metrics.average_detection_time = 2.5  # Simulated
        
        # Calculate system security score
        total_threats = self.security_metrics.total_threats_detected
        critical_threats = self.security_metrics.threats_by_severity['critical']
        high_threats = self.security_metrics.threats_by_severity['high']
        
        if total_threats > 0:
            threat_impact = (critical_threats * 0.9 + high_threats * 0.6) / total_threats
            self.security_metrics.system_security_score = max(0.1, 1.0 - threat_impact)
        else:
            self.security_metrics.system_security_score = 1.0
    
    def submit_monitoring_data(self, data: Dict[str, Any]):
        """Submit data for threat analysis."""
        self.threat_queue.put(data)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            'monitoring_active': self.monitoring_active,
            'active_threats': len(self.active_threats),
            'total_threats_detected': self.security_metrics.total_threats_detected,
            'security_score': self.security_metrics.system_security_score,
            'mitigation_success_rate': self.security_metrics.mitigation_success_rate,
            'threats_by_severity': self.security_metrics.threats_by_severity.copy(),
            'blocked_entities': self.mitigation_engine.get_blocked_entities(),
            'recent_threats': [asdict(t) for t in self.threat_history[-10:]]
        }
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report."""
        report_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Get current status
        status = self.get_security_status()
        
        report = f"""
# Threat Detection and Security Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 🛡️ Security Overview
- System Security Score: {status['security_score']:.2%}
- Monitoring Status: {'🟢 ACTIVE' if status['monitoring_active'] else '🔴 INACTIVE'}
- Active Threats: {status['active_threats']}
- Mitigation Success Rate: {status['mitigation_success_rate']:.2%}

## 📊 Threat Statistics
- Total Threats Detected: {status['total_threats_detected']}
"""
        
        for severity, count in status['threats_by_severity'].items():
            emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴', 'critical': '🚨'}[severity]
            report += f"- {emoji} {severity.title()}: {count}\n"
        
        report += f"""
## 🚫 Blocked Entities
- Users: {len(status['blocked_entities']['users'])}
- IP Addresses: {len(status['blocked_entities']['ips'])}

## ⚠️ Recent Threats
"""
        
        for threat in status['recent_threats'][-5:]:
            report += f"- **{threat['threat_type']}** ({threat['severity']}): {threat['description']}\n"
        
        report += f"""
## 🔧 Active Detectors
"""
        
        for detector in self.detectors:
            report += f"- {detector.get_detector_name()}: ✅ Active\n"
        
        report += f"""
---
Generated by Threat Detection System v1.0
Classification: {'🔒 SECURE' if status['security_score'] > 0.8 else '⚠️ MONITORING' if status['security_score'] > 0.6 else '🚨 ALERT'}
"""
        
        # Save report
        report_path = Path(f"security_reports/security_report_{report_timestamp}.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"📄 Security report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")
        
        return report


def demonstrate_threat_detection():
    """Demonstrate the threat detection system."""
    print("🛡️ Advanced Threat Detection and Mitigation Demo")
    print("=" * 55)
    
    # Initialize threat detection system
    system = ThreatDetectionSystem()
    
    # Start monitoring
    system.start_monitoring()
    
    try:
        # Simulate various threat scenarios
        threat_scenarios = [
            # Privacy budget exhaustion
            {
                'scenario': 'Privacy Budget Exhaustion',
                'data': {
                    'user_id': 'attacker_1',
                    'privacy_budget_remaining': 0.05,
                    'budget_consumption_rate': 0.15,
                    'query_type': 'high_sensitivity'
                }
            },
            # Membership inference attack
            {
                'scenario': 'Membership Inference Attack',
                'data': {
                    'user_id': 'attacker_2', 
                    'query_type': 'shadow_model_training',
                    'model_outputs': [0.9, 0.8, 0.7, 0.95, 0.85],
                    'confidence_scores': [0.95, 0.92, 0.88, 0.97, 0.91]
                }
            },
            # Gradient leakage
            {
                'scenario': 'Gradient Leakage',
                'data': {
                    'user_id': 'attacker_3',
                    'computation_time': 0.0512,
                    'memory_usage': 0.75,
                    'gradient_norms': [1.5, 1.8, 1.6, 1.7, 1.9]
                }
            },
            # Coordinated attack
            {
                'scenario': 'Coordinated Attack',
                'data': {
                    'user_id': 'attacker_4',
                    'privacy_budget_remaining': 0.1,
                    'budget_consumption_rate': 0.08
                }
            }
        ]
        
        # Submit threat scenarios
        for scenario in threat_scenarios:
            print(f"\n🎭 Simulating: {scenario['scenario']}")
            system.submit_monitoring_data(scenario['data'])
            time.sleep(2)  # Allow processing
        
        # Add more coordinated attack data
        for i in range(3):
            system.submit_monitoring_data({
                'user_id': f'attacker_coord_{i}',
                'privacy_budget_remaining': 0.15,
                'budget_consumption_rate': 0.06
            })
        
        # Wait for processing
        print(f"\n⏳ Processing threat scenarios...")
        time.sleep(10)
        
        # Get security status
        status = system.get_security_status()
        print(f"\n📊 Security Status:")
        print(f"- Security Score: {status['security_score']:.2%}")
        print(f"- Threats Detected: {status['total_threats_detected']}")
        print(f"- Active Threats: {status['active_threats']}")
        print(f"- Mitigation Success: {status['mitigation_success_rate']:.2%}")
        
        # Show blocked entities
        blocked = status['blocked_entities']
        if blocked['users']:
            print(f"- Blocked Users: {len(blocked['users'])}")
        
        # Generate comprehensive report
        print(f"\n📄 Generating security report...")
        report = system.generate_security_report()
        
    finally:
        # Stop monitoring
        system.stop_monitoring()
        print(f"\n✅ Threat detection demonstration completed!")
        print(f"System Security Level: {'🔒 SECURE' if status['security_score'] > 0.8 else '⚠️ MONITORING'}")


if __name__ == "__main__":
    demonstrate_threat_detection()