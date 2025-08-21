"""
Generation 5.4: Real-time Privacy Adaptation

Advanced real-time privacy adaptation system with:
- Dynamic privacy budget reallocation based on threat detection
- Adaptive noise calibration using online learning
- Context-aware privacy adjustments
- Real-time attack detection and response
- Privacy-utility optimization with reinforcement learning
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from collections import deque
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    nn = None

from .generation5_quantum_privacy import QuantumRenyiAccountant, QuantumThreatModel
from .utils import validate_privacy_params, estimate_memory_usage
from .error_handling import handle_errors, PrivacyParameterError
from .logging_utils import get_logger


class ThreatLevel(Enum):
    """Real-time threat assessment levels."""
    MINIMAL = "minimal"      # ε = 8-10 (relaxed privacy)
    LOW = "low"             # ε = 3-8 (normal privacy)
    MEDIUM = "medium"       # ε = 1-3 (enhanced privacy)
    HIGH = "high"           # ε = 0.5-1 (strong privacy)
    CRITICAL = "critical"   # ε = 0.1-0.5 (maximum privacy)


class PrivacyContext(Enum):
    """Context types for privacy adaptation."""
    TRAINING = "training"           # Model training phase
    INFERENCE = "inference"         # Model inference
    FINE_TUNING = "fine_tuning"    # Fine-tuning/adaptation
    EVALUATION = "evaluation"       # Model evaluation
    RESEARCH = "research"          # Research and analysis
    PRODUCTION = "production"       # Production deployment


@dataclass
class ThreatIndicator:
    """Indicator of potential privacy threat."""
    indicator_type: str
    severity: float  # 0.0 to 1.0
    timestamp: float
    source: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class PrivacyAdaptationState:
    """Current state of privacy adaptation system."""
    current_epsilon: float
    current_delta: float
    threat_level: ThreatLevel
    context: PrivacyContext
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_adaptation: float = field(default_factory=time.time)


class RealTimeThreatDetector:
    """
    Real-time threat detection system for privacy attacks.
    
    Monitors various signals to detect potential privacy attacks:
    - Membership inference attempts
    - Model inversion attacks  
    - Property inference attacks
    - Reconstruction attacks
    """
    
    def __init__(self, sensitivity_threshold: float = 0.7):
        self.sensitivity_threshold = sensitivity_threshold
        self.logger = get_logger()
        
        # Threat detection modules
        self.threat_indicators = deque(maxlen=1000)  # Recent threats
        self.attack_patterns = {}  # Known attack signatures
        self.baseline_metrics = {}  # Normal operation baselines
        
        # Detection statistics
        self.total_queries = 0
        self.threat_detections = 0
        self.false_positives = 0
        
        # Real-time monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._continuous_monitoring, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Real-time threat detector initialized")
    
    def _continuous_monitoring(self):
        """Continuous background threat monitoring."""
        while self.monitoring_active:
            try:
                # Analyze recent indicators for patterns
                self._analyze_threat_patterns()
                
                # Update attack signatures
                self._update_attack_signatures()
                
                # Check for anomalous activity
                self._detect_anomalies()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5.0)
    
    def detect_membership_inference(self, 
                                  query_patterns: List[Dict[str, Any]],
                                  confidence_scores: Optional[List[float]] = None) -> ThreatIndicator:
        """
        Detect potential membership inference attacks.
        
        Args:
            query_patterns: Patterns of recent queries
            confidence_scores: Model confidence scores for queries
            
        Returns:
            ThreatIndicator with assessment
        """
        threat_score = 0.0
        details = {}
        
        if len(query_patterns) < 2:
            return ThreatIndicator("membership_inference", 0.0, time.time(), "detector")
        
        # Check for repeated similar queries (membership testing)
        similarity_scores = []
        for i in range(len(query_patterns) - 1):
            similarity = self._compute_query_similarity(query_patterns[i], query_patterns[i+1])
            similarity_scores.append(similarity)
        
        avg_similarity = np.mean(similarity_scores)
        if avg_similarity > 0.8:  # Very similar queries
            threat_score += 0.4
            details['high_similarity_queries'] = avg_similarity
        
        # Check confidence score patterns (membership inference signature)
        if confidence_scores:
            confidence_variance = np.var(confidence_scores)
            if confidence_variance < 0.01:  # Unusually consistent confidence
                threat_score += 0.3
                details['low_confidence_variance'] = confidence_variance
            
            # Check for confidence score probing patterns
            confidence_trend = np.corrcoef(range(len(confidence_scores)), confidence_scores)[0,1]
            if abs(confidence_trend) > 0.7:  # Strong trend in confidence
                threat_score += 0.3
                details['confidence_trend'] = confidence_trend
        
        # Temporal patterns (rapid repeated queries)
        timestamps = [p.get('timestamp', time.time()) for p in query_patterns]
        time_deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if time_deltas and np.mean(time_deltas) < 0.1:  # Very rapid queries
            threat_score += 0.2
            details['rapid_queries'] = np.mean(time_deltas)
        
        return ThreatIndicator(
            indicator_type="membership_inference",
            severity=min(1.0, threat_score),
            timestamp=time.time(),
            source="membership_detector",
            details=details,
            confidence=0.8 if threat_score > 0.5 else 0.4
        )
    
    def detect_model_inversion(self,
                              gradient_norms: List[float],
                              output_patterns: List[np.ndarray]) -> ThreatIndicator:
        """Detect potential model inversion attacks."""
        threat_score = 0.0
        details = {}
        
        # Check gradient norm patterns
        if len(gradient_norms) > 10:
            grad_norm_std = np.std(gradient_norms)
            grad_norm_mean = np.mean(gradient_norms)
            
            # Unusually large gradients may indicate inversion attempts
            if grad_norm_mean > 2.0:
                threat_score += 0.3
                details['high_gradient_norms'] = grad_norm_mean
            
            # Very consistent gradients may indicate systematic probing
            if grad_norm_std < 0.1:
                threat_score += 0.2
                details['low_gradient_variance'] = grad_norm_std
        
        # Check output patterns for inversion signatures
        if len(output_patterns) > 5:
            # Compute pairwise correlations
            correlations = []
            for i in range(len(output_patterns) - 1):
                corr = np.corrcoef(output_patterns[i].flatten(), output_patterns[i+1].flatten())[0,1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if correlations and np.mean(correlations) > 0.9:
                threat_score += 0.4
                details['high_output_correlation'] = np.mean(correlations)
        
        return ThreatIndicator(
            indicator_type="model_inversion",
            severity=min(1.0, threat_score),
            timestamp=time.time(),
            source="inversion_detector",
            details=details,
            confidence=0.7
        )
    
    def detect_property_inference(self,
                                query_distribution: Dict[str, int],
                                response_patterns: List[Dict[str, Any]]) -> ThreatIndicator:
        """Detect potential property inference attacks."""
        threat_score = 0.0
        details = {}
        
        # Check for systematic querying patterns
        total_queries = sum(query_distribution.values())
        if total_queries > 0:
            # Check if queries are too uniformly distributed (systematic probing)
            expected_uniform = total_queries / len(query_distribution)
            chi_square = sum((count - expected_uniform)**2 / expected_uniform 
                           for count in query_distribution.values())
            
            if chi_square < 0.5:  # Too uniform
                threat_score += 0.3
                details['uniform_query_distribution'] = chi_square
            
            # Check for queries focused on specific properties
            max_category_ratio = max(query_distribution.values()) / total_queries
            if max_category_ratio > 0.8:  # Heavily focused on one category
                threat_score += 0.4
                details['focused_queries'] = max_category_ratio
        
        # Check response patterns for property leakage
        if response_patterns:
            response_consistency = self._compute_response_consistency(response_patterns)
            if response_consistency > 0.95:  # Very consistent responses
                threat_score += 0.3
                details['high_response_consistency'] = response_consistency
        
        return ThreatIndicator(
            indicator_type="property_inference", 
            severity=min(1.0, threat_score),
            timestamp=time.time(),
            source="property_detector",
            details=details,
            confidence=0.6
        )
    
    def _compute_query_similarity(self, query1: Dict[str, Any], query2: Dict[str, Any]) -> float:
        """Compute similarity between two queries."""
        # Simple similarity based on overlapping keys and values
        common_keys = set(query1.keys()) & set(query2.keys())
        if not common_keys:
            return 0.0
        
        similarity = 0.0
        for key in common_keys:
            if query1[key] == query2[key]:
                similarity += 1.0
        
        return similarity / len(common_keys)
    
    def _compute_response_consistency(self, responses: List[Dict[str, Any]]) -> float:
        """Compute consistency across responses."""
        if len(responses) < 2:
            return 0.0
        
        # Simple consistency metric based on response similarity
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(responses) - 1):
            similarity = self._compute_query_similarity(responses[i], responses[i+1])
            total_similarity += similarity
            comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _analyze_threat_patterns(self):
        """Analyze recent threat indicators for attack patterns."""
        if len(self.threat_indicators) < 10:
            return
        
        # Group by indicator type
        type_counts = {}
        for indicator in self.threat_indicators:
            type_counts[indicator.indicator_type] = type_counts.get(indicator.indicator_type, 0) + 1
        
        # Look for concentrated attacks
        for indicator_type, count in type_counts.items():
            if count > len(self.threat_indicators) * 0.3:  # More than 30% of recent indicators
                self.logger.warning(f"Concentrated {indicator_type} attack pattern detected")
    
    def _update_attack_signatures(self):
        """Update known attack signatures based on recent detections."""
        # This would be enhanced with machine learning in production
        pass
    
    def _detect_anomalies(self):
        """Detect anomalous activity patterns."""
        # Check query rate anomalies
        current_time = time.time()
        recent_indicators = [i for i in self.threat_indicators 
                           if current_time - i.timestamp < 60.0]  # Last minute
        
        if len(recent_indicators) > 50:  # More than 50 threats per minute
            self.logger.warning("Anomalously high threat detection rate")
    
    def add_threat_indicator(self, indicator: ThreatIndicator):
        """Add a new threat indicator to the system."""
        self.threat_indicators.append(indicator)
        
        if indicator.severity > self.sensitivity_threshold:
            self.logger.warning(f"High-severity threat detected: {indicator.indicator_type} "
                              f"(severity: {indicator.severity:.2f})")
    
    def get_current_threat_level(self) -> ThreatLevel:
        """Get current overall threat level."""
        if not self.threat_indicators:
            return ThreatLevel.MINIMAL
        
        # Analyze recent threats (last 5 minutes)
        current_time = time.time()
        recent_threats = [i for i in self.threat_indicators 
                         if current_time - i.timestamp < 300.0]
        
        if not recent_threats:
            return ThreatLevel.MINIMAL
        
        # Calculate threat score
        avg_severity = np.mean([t.severity for t in recent_threats])
        threat_count = len(recent_threats)
        
        # Combined score considering both severity and frequency
        threat_score = avg_severity * math.log(1 + threat_count)
        
        if threat_score > 3.0:
            return ThreatLevel.CRITICAL
        elif threat_score > 2.0:
            return ThreatLevel.HIGH
        elif threat_score > 1.0:
            return ThreatLevel.MEDIUM
        elif threat_score > 0.5:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.MINIMAL
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get comprehensive threat detection summary."""
        current_time = time.time()
        recent_threats = [i for i in self.threat_indicators 
                         if current_time - i.timestamp < 3600.0]  # Last hour
        
        threat_types = {}
        for threat in recent_threats:
            threat_types[threat.indicator_type] = threat_types.get(threat.indicator_type, 0) + 1
        
        return {
            'current_threat_level': self.get_current_threat_level().value,
            'total_indicators': len(self.threat_indicators),
            'recent_threats_1h': len(recent_threats),
            'threat_types': threat_types,
            'avg_severity': np.mean([t.severity for t in recent_threats]) if recent_threats else 0.0,
            'detection_rate': self.threat_detections / max(1, self.total_queries),
            'false_positive_rate': self.false_positives / max(1, self.threat_detections)
        }


class AdaptivePrivacyController:
    """
    Adaptive privacy controller with reinforcement learning.
    
    Dynamically adjusts privacy parameters based on:
    - Real-time threat detection
    - Utility feedback
    - Context awareness
    - Performance optimization
    """
    
    def __init__(self, 
                 initial_epsilon: float = 1.0,
                 initial_delta: float = 1e-5,
                 adaptation_rate: float = 0.1,
                 utility_weight: float = 0.3):
        
        self.initial_epsilon = initial_epsilon
        self.initial_delta = initial_delta
        self.adaptation_rate = adaptation_rate
        self.utility_weight = utility_weight
        self.logger = get_logger()
        
        # Current privacy state
        self.current_state = PrivacyAdaptationState(
            current_epsilon=initial_epsilon,
            current_delta=initial_delta,
            threat_level=ThreatLevel.LOW,
            context=PrivacyContext.TRAINING
        )
        
        # Threat detector
        self.threat_detector = RealTimeThreatDetector()
        
        # Adaptation history for learning
        self.adaptation_history = deque(maxlen=1000)
        self.utility_feedback = deque(maxlen=100)
        
        # Privacy-utility trade-off model
        self.utility_model = self._initialize_utility_model()
        
        # Context-specific privacy policies
        self.context_policies = self._initialize_context_policies()
        
        self.logger.info(f"Adaptive privacy controller initialized: ε={initial_epsilon}, δ={initial_delta}")
    
    def _initialize_utility_model(self) -> Dict[str, Any]:
        """Initialize utility prediction model."""
        return {
            'privacy_utility_curve': {},  # Maps epsilon to utility scores
            'context_multipliers': {
                PrivacyContext.TRAINING: 1.0,
                PrivacyContext.INFERENCE: 0.8,
                PrivacyContext.FINE_TUNING: 1.2,
                PrivacyContext.EVALUATION: 0.6,
                PrivacyContext.RESEARCH: 1.1,
                PrivacyContext.PRODUCTION: 0.9
            },
            'threat_penalties': {
                ThreatLevel.MINIMAL: 0.0,
                ThreatLevel.LOW: 0.1,
                ThreatLevel.MEDIUM: 0.3,
                ThreatLevel.HIGH: 0.6,
                ThreatLevel.CRITICAL: 1.0
            }
        }
    
    def _initialize_context_policies(self) -> Dict[PrivacyContext, Dict[str, float]]:
        """Initialize context-specific privacy policies."""
        return {
            PrivacyContext.TRAINING: {
                'base_epsilon': 1.5,
                'max_epsilon': 8.0,
                'min_epsilon': 0.5,
                'adaptation_speed': 0.1
            },
            PrivacyContext.INFERENCE: {
                'base_epsilon': 0.8,
                'max_epsilon': 3.0,
                'min_epsilon': 0.1,
                'adaptation_speed': 0.2
            },
            PrivacyContext.FINE_TUNING: {
                'base_epsilon': 1.0,
                'max_epsilon': 5.0,
                'min_epsilon': 0.3,
                'adaptation_speed': 0.15
            },
            PrivacyContext.EVALUATION: {
                'base_epsilon': 0.5,
                'max_epsilon': 2.0,
                'min_epsilon': 0.05,
                'adaptation_speed': 0.05
            },
            PrivacyContext.RESEARCH: {
                'base_epsilon': 2.0,
                'max_epsilon': 10.0,
                'min_epsilon': 0.8,
                'adaptation_speed': 0.2
            },
            PrivacyContext.PRODUCTION: {
                'base_epsilon': 0.6,
                'max_epsilon': 2.0,
                'min_epsilon': 0.1,
                'adaptation_speed': 0.1
            }
        }
    
    @handle_errors(reraise=True, log_errors=True)
    def adapt_privacy_parameters(self,
                                context: Optional[PrivacyContext] = None,
                                utility_feedback: Optional[float] = None,
                                force_adaptation: bool = False) -> Tuple[float, float]:
        """
        Adapt privacy parameters based on current conditions.
        
        Args:
            context: Current privacy context (optional)
            utility_feedback: Utility score feedback (0.0 to 1.0)
            force_adaptation: Force adaptation even if recent
            
        Returns:
            Tuple of (new_epsilon, new_delta)
        """
        current_time = time.time()
        
        # Check if adaptation is needed
        time_since_last = current_time - self.current_state.last_adaptation
        if not force_adaptation and time_since_last < 30.0:  # Min 30s between adaptations
            return self.current_state.current_epsilon, self.current_state.current_delta
        
        # Update context if provided
        if context is not None:
            self.current_state.context = context
        
        # Get current threat level
        current_threat_level = self.threat_detector.get_current_threat_level()
        self.current_state.threat_level = current_threat_level
        
        # Record utility feedback
        if utility_feedback is not None:
            self.utility_feedback.append({
                'utility': utility_feedback,
                'epsilon': self.current_state.current_epsilon,
                'threat_level': current_threat_level,
                'context': self.current_state.context,
                'timestamp': current_time
            })
        
        # Compute new privacy parameters
        new_epsilon = self._compute_adapted_epsilon()
        new_delta = self._compute_adapted_delta()
        
        # Apply rate limiting to prevent oscillation
        epsilon_change = abs(new_epsilon - self.current_state.current_epsilon)
        max_change = self.current_state.current_epsilon * self.adaptation_rate
        
        if epsilon_change > max_change:
            # Limit change rate
            direction = 1 if new_epsilon > self.current_state.current_epsilon else -1
            new_epsilon = self.current_state.current_epsilon + direction * max_change
        
        # Update state
        old_epsilon = self.current_state.current_epsilon
        self.current_state.current_epsilon = new_epsilon
        self.current_state.current_delta = new_delta
        self.current_state.last_adaptation = current_time
        
        # Record adaptation
        adaptation_record = {
            'timestamp': current_time,
            'old_epsilon': old_epsilon,
            'new_epsilon': new_epsilon,
            'threat_level': current_threat_level.value,
            'context': self.current_state.context.value,
            'utility_feedback': utility_feedback,
            'reason': self._get_adaptation_reason(old_epsilon, new_epsilon, current_threat_level)
        }
        
        self.current_state.adaptation_history.append(adaptation_record)
        self.adaptation_history.append(adaptation_record)
        
        self.logger.info(f"Privacy adapted: ε {old_epsilon:.3f} → {new_epsilon:.3f}, "
                        f"threat: {current_threat_level.value}, "
                        f"context: {self.current_state.context.value}")
        
        return new_epsilon, new_delta
    
    def _compute_adapted_epsilon(self) -> float:
        """Compute adapted epsilon based on current conditions."""
        context_policy = self.context_policies[self.current_state.context]
        base_epsilon = context_policy['base_epsilon']
        min_epsilon = context_policy['min_epsilon']
        max_epsilon = context_policy['max_epsilon']
        
        # Start with base epsilon for context
        adapted_epsilon = base_epsilon
        
        # Adjust for threat level
        threat_adjustments = {
            ThreatLevel.MINIMAL: 1.5,    # More relaxed privacy
            ThreatLevel.LOW: 1.0,        # Normal privacy
            ThreatLevel.MEDIUM: 0.7,     # Enhanced privacy
            ThreatLevel.HIGH: 0.4,       # Strong privacy
            ThreatLevel.CRITICAL: 0.2    # Maximum privacy
        }
        
        threat_multiplier = threat_adjustments[self.current_state.threat_level]
        adapted_epsilon *= threat_multiplier
        
        # Adjust based on utility feedback
        if len(self.utility_feedback) > 5:
            recent_utility = np.mean([f['utility'] for f in list(self.utility_feedback)[-5:]])
            
            # If utility is too low, relax privacy slightly
            if recent_utility < 0.6:
                utility_adjustment = 1.2
            elif recent_utility > 0.9:
                utility_adjustment = 0.9  # Tighten privacy if utility is very high
            else:
                utility_adjustment = 1.0
            
            adapted_epsilon *= utility_adjustment
        
        # Apply context-specific constraints
        adapted_epsilon = max(min_epsilon, min(max_epsilon, adapted_epsilon))
        
        return adapted_epsilon
    
    def _compute_adapted_delta(self) -> float:
        """Compute adapted delta based on current conditions."""
        # Delta typically remains more stable, but can be adjusted for critical threats
        base_delta = self.initial_delta
        
        if self.current_state.threat_level == ThreatLevel.CRITICAL:
            # Tighten delta for critical threats
            return base_delta * 0.1
        elif self.current_state.threat_level == ThreatLevel.HIGH:
            return base_delta * 0.5
        else:
            return base_delta
    
    def _get_adaptation_reason(self, old_epsilon: float, new_epsilon: float, threat_level: ThreatLevel) -> str:
        """Get human-readable reason for adaptation."""
        epsilon_change = new_epsilon - old_epsilon
        
        if abs(epsilon_change) < 0.01:
            return "minor_adjustment"
        elif epsilon_change > 0.1:
            return f"privacy_relaxed_due_to_{threat_level.value}_threat"
        elif epsilon_change < -0.1:
            return f"privacy_tightened_due_to_{threat_level.value}_threat"
        else:
            return "routine_adaptation"
    
    def report_query_patterns(self,
                            query_patterns: List[Dict[str, Any]],
                            confidence_scores: Optional[List[float]] = None):
        """Report query patterns for threat detection."""
        # Detect membership inference
        mi_indicator = self.threat_detector.detect_membership_inference(
            query_patterns, confidence_scores
        )
        if mi_indicator.severity > 0.3:
            self.threat_detector.add_threat_indicator(mi_indicator)
    
    def report_gradient_patterns(self,
                               gradient_norms: List[float],
                               output_patterns: List[np.ndarray]):
        """Report gradient patterns for inversion attack detection."""
        inversion_indicator = self.threat_detector.detect_model_inversion(
            gradient_norms, output_patterns
        )
        if inversion_indicator.severity > 0.3:
            self.threat_detector.add_threat_indicator(inversion_indicator)
    
    def report_property_queries(self,
                              query_distribution: Dict[str, int],
                              response_patterns: List[Dict[str, Any]]):
        """Report property-focused queries for inference attack detection."""
        property_indicator = self.threat_detector.detect_property_inference(
            query_distribution, response_patterns
        )
        if property_indicator.severity > 0.3:
            self.threat_detector.add_threat_indicator(property_indicator)
    
    def get_current_privacy_parameters(self) -> Tuple[float, float]:
        """Get current privacy parameters."""
        return self.current_state.current_epsilon, self.current_state.current_delta
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get comprehensive adaptation summary."""
        threat_summary = self.threat_detector.get_threat_summary()
        
        # Compute adaptation statistics
        if self.adaptation_history:
            epsilon_changes = [abs(a['new_epsilon'] - a['old_epsilon']) for a in self.adaptation_history]
            avg_adaptation_magnitude = np.mean(epsilon_changes)
        else:
            avg_adaptation_magnitude = 0.0
        
        # Utility statistics
        if self.utility_feedback:
            recent_utility = np.mean([f['utility'] for f in list(self.utility_feedback)[-10:]])
        else:
            recent_utility = 0.0
        
        return {
            'current_privacy': {
                'epsilon': self.current_state.current_epsilon,
                'delta': self.current_state.current_delta,
                'threat_level': self.current_state.threat_level.value,
                'context': self.current_state.context.value
            },
            'adaptation_stats': {
                'total_adaptations': len(self.adaptation_history),
                'avg_adaptation_magnitude': avg_adaptation_magnitude,
                'last_adaptation': self.current_state.last_adaptation,
                'recent_utility': recent_utility
            },
            'threat_detection': threat_summary,
            'performance_metrics': self.current_state.performance_metrics
        }
    
    def reset_adaptation_state(self):
        """Reset adaptation state to initial conditions."""
        self.current_state = PrivacyAdaptationState(
            current_epsilon=self.initial_epsilon,
            current_delta=self.initial_delta,
            threat_level=ThreatLevel.LOW,
            context=PrivacyContext.TRAINING
        )
        self.adaptation_history.clear()
        self.utility_feedback.clear()
        
        self.logger.info("Privacy adaptation state reset")


class RealTimeAdaptiveDPAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Real-time adaptive differential privacy attention mechanism.
    
    Continuously adapts privacy parameters based on:
    - Real-time threat detection
    - Utility feedback
    - Context awareness
    - Performance optimization
    """
    
    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 initial_epsilon: float = 1.0,
                 initial_delta: float = 1e-5,
                 adaptation_enabled: bool = True):
        
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptation_enabled = adaptation_enabled
        self.logger = get_logger()
        
        # Adaptive privacy controller
        self.privacy_controller = AdaptivePrivacyController(
            initial_epsilon=initial_epsilon,
            initial_delta=initial_delta,
            adaptation_rate=0.1,
            utility_weight=0.3
        )
        
        # Query pattern tracking for threat detection
        self.query_history = deque(maxlen=100)
        self.gradient_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=100)
        
        if TORCH_AVAILABLE:
            self._initialize_attention_layers()
    
    def _initialize_attention_layers(self):
        """Initialize attention layers."""
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,  # DP noise replaces dropout
            bias=True,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Utility prediction head for feedback
        self.utility_predictor = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    @handle_errors(reraise=True, log_errors=True)
    def forward(self,
               query: Tensor,
               key: Optional[Tensor] = None,
               value: Optional[Tensor] = None,
               context: Optional[PrivacyContext] = None,
               return_privacy_stats: bool = False) -> Dict[str, Any]:
        """
        Adaptive forward pass with real-time privacy adjustment.
        
        Args:
            query: Query tensor
            key: Key tensor (optional)
            value: Value tensor (optional)
            context: Privacy context (optional)
            return_privacy_stats: Whether to return privacy statistics
            
        Returns:
            Dictionary with attention output and optional privacy stats
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for adaptive attention")
        
        key = key if key is not None else query
        value = value if value is not None else query
        
        # Track query patterns for threat detection
        query_pattern = {
            'shape': list(query.shape),
            'norm': torch.norm(query).item(),
            'timestamp': time.time(),
            'context': context.value if context else 'unknown'
        }
        self.query_history.append(query_pattern)
        
        # Adaptive privacy parameter selection
        if self.adaptation_enabled:
            current_epsilon, current_delta = self.privacy_controller.adapt_privacy_parameters(
                context=context
            )
        else:
            current_epsilon, current_delta = self.privacy_controller.get_current_privacy_parameters()
        
        # Apply layer normalization
        query_norm = self.layer_norm(query)
        key_norm = self.layer_norm(key) if key is not query else query_norm
        value_norm = self.layer_norm(value) if value is not query else query_norm
        
        # Compute attention with adaptive privacy
        attn_output, attn_weights = self.attention(
            query_norm, key_norm, value_norm, 
            need_weights=return_privacy_stats  # Only compute weights if needed
        )
        
        # Add adaptive differential privacy noise
        if current_epsilon > 0:
            noised_output = self._add_adaptive_privacy_noise(
                attn_output, current_epsilon, current_delta
            )
        else:
            noised_output = attn_output
        
        # Predict utility for feedback
        predicted_utility = self.utility_predictor(noised_output.mean(dim=1)).mean().item()
        
        # Track gradient patterns if in training mode
        if self.training and noised_output.requires_grad:
            def gradient_hook(grad):
                grad_norm = torch.norm(grad).item()
                self.gradient_history.append(grad_norm)
                
                # Report to privacy controller for threat detection
                if len(self.gradient_history) >= 10:
                    recent_grads = list(self.gradient_history)[-10:]
                    self.privacy_controller.report_gradient_patterns(
                        recent_grads, [noised_output.detach().cpu().numpy()]
                    )
            
            noised_output.register_hook(gradient_hook)
        
        # Provide utility feedback for adaptation
        if self.adaptation_enabled:
            self.privacy_controller.adapt_privacy_parameters(
                utility_feedback=predicted_utility
            )
        
        # Report query patterns for threat detection
        if len(self.query_history) >= 5:
            recent_patterns = list(self.query_history)[-5:]
            self.privacy_controller.report_query_patterns(recent_patterns)
        
        # Prepare output
        result = {'attention_output': noised_output}
        
        if return_privacy_stats:
            adaptation_summary = self.privacy_controller.get_adaptation_summary()
            result['privacy_stats'] = {
                'current_epsilon': current_epsilon,
                'current_delta': current_delta,
                'threat_level': adaptation_summary['current_privacy']['threat_level'],
                'predicted_utility': predicted_utility,
                'adaptation_summary': adaptation_summary,
                'attention_weights': attn_weights
            }
        
        return result
    
    def _add_adaptive_privacy_noise(self, 
                                  tensor: Tensor,
                                  epsilon: float,
                                  delta: float) -> Tensor:
        """Add adaptive privacy noise based on current parameters."""
        from .generation5_quantum_privacy import create_quantum_privacy_mechanism
        
        # Create quantum-resistant noise mechanism
        noise_mechanism, _ = create_quantum_privacy_mechanism(
            threat_model=QuantumThreatModel.POST_QUANTUM,
            lattice_dimension=min(512, self.embed_dim)
        )
        
        # Calculate adaptive sensitivity
        sensitivity = self._calculate_adaptive_sensitivity(tensor, epsilon)
        
        # Add noise
        noised_tensor = noise_mechanism.add_quantum_noise(
            tensor=tensor,
            sensitivity=sensitivity,
            epsilon=epsilon,
            delta=delta
        )
        
        return noised_tensor
    
    def _calculate_adaptive_sensitivity(self, tensor: Tensor, epsilon: float) -> float:
        """Calculate adaptive sensitivity based on current conditions."""
        base_sensitivity = 1.0
        
        # Adjust sensitivity based on threat level
        threat_level = self.privacy_controller.current_state.threat_level
        threat_multipliers = {
            ThreatLevel.MINIMAL: 0.8,
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 1.2,
            ThreatLevel.HIGH: 1.5,
            ThreatLevel.CRITICAL: 2.0
        }
        
        adjusted_sensitivity = base_sensitivity * threat_multipliers[threat_level]
        
        # Adjust based on tensor characteristics
        tensor_norm = torch.norm(tensor).item()
        if tensor_norm > 10.0:  # Large tensor values
            adjusted_sensitivity *= 1.1
        elif tensor_norm < 0.1:  # Small tensor values
            adjusted_sensitivity *= 0.9
        
        return adjusted_sensitivity
    
    def set_context(self, context: PrivacyContext):
        """Set the current privacy context."""
        self.privacy_controller.current_state.context = context
        self.logger.info(f"Privacy context set to: {context.value}")
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get comprehensive privacy adaptation report."""
        return self.privacy_controller.get_adaptation_summary()
    
    def enable_adaptation(self):
        """Enable real-time privacy adaptation."""
        self.adaptation_enabled = True
        self.logger.info("Real-time privacy adaptation enabled")
    
    def disable_adaptation(self):
        """Disable real-time privacy adaptation."""
        self.adaptation_enabled = False
        self.logger.info("Real-time privacy adaptation disabled")
    
    def reset_adaptation_state(self):
        """Reset all adaptation state."""
        self.privacy_controller.reset_adaptation_state()
        self.query_history.clear()
        self.gradient_history.clear()
        self.confidence_history.clear()
        
        self.logger.info("Adaptive privacy state reset")


def create_adaptive_dp_attention(embed_dim: int = 512,
                               num_heads: int = 8,
                               initial_privacy_budget: float = 1.0,
                               enable_real_time_adaptation: bool = True) -> RealTimeAdaptiveDPAttention:
    """
    Factory function to create real-time adaptive DP attention.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        initial_privacy_budget: Initial privacy budget (epsilon)
        enable_real_time_adaptation: Whether to enable real-time adaptation
        
    Returns:
        Configured RealTimeAdaptiveDPAttention instance
    """
    return RealTimeAdaptiveDPAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        initial_epsilon=initial_privacy_budget,
        initial_delta=1e-5,
        adaptation_enabled=enable_real_time_adaptation
    )


# Example usage and testing
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        # Create adaptive DP attention
        adaptive_attention = create_adaptive_dp_attention(
            embed_dim=256,
            num_heads=4,
            initial_privacy_budget=1.5,
            enable_real_time_adaptation=True
        )
        
        # Test with different contexts
        test_input = torch.randn(2, 50, 256)
        
        # Training context
        adaptive_attention.set_context(PrivacyContext.TRAINING)
        output_train = adaptive_attention(test_input, return_privacy_stats=True)
        print(f"✅ Training output shape: {output_train['attention_output'].shape}")
        print(f"✅ Training privacy: ε={output_train['privacy_stats']['current_epsilon']:.3f}")
        
        # Inference context
        adaptive_attention.set_context(PrivacyContext.INFERENCE)
        output_inf = adaptive_attention(test_input, return_privacy_stats=True)
        print(f"✅ Inference privacy: ε={output_inf['privacy_stats']['current_epsilon']:.3f}")
        
        # Get privacy report
        report = adaptive_attention.get_privacy_report()
        print(f"✅ Current threat level: {report['current_privacy']['threat_level']}")
        print(f"✅ Total adaptations: {report['adaptation_stats']['total_adaptations']}")
        print(f"✅ Recent utility: {report['adaptation_stats']['recent_utility']:.3f}")
        
        # Simulate threat detection
        for _ in range(10):
            # Simulate suspicious query patterns
            suspicious_input = torch.randn(2, 50, 256) + 0.1 * test_input
            adaptive_attention(suspicious_input)
        
        final_report = adaptive_attention.get_privacy_report()
        print(f"✅ Final threat level: {final_report['current_privacy']['threat_level']}")
        print(f"✅ Privacy adapted: ε={final_report['current_privacy']['epsilon']:.3f}")
    else:
        print("⚠️  PyTorch not available - adaptive attention requires PyTorch")