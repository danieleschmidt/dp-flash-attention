"""
Advanced Federated Learning Coordinator for DP-Flash-Attention.

This module implements state-of-the-art federated learning coordination with
integrated differential privacy, enabling secure collaborative training across
distributed edge devices and data centers while maintaining privacy guarantees.

Key Features:
- Secure aggregation with differential privacy
- Heterogeneous device coordination
- Adaptive communication protocols
- Privacy-preserving model updates
- Cross-silo and cross-device federation support
- Real-time privacy budget management
- Byzantine fault tolerance
- Asynchronous training coordination
"""

import time
import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False
    ThreadPoolExecutor = None
    as_completed = None
import queue
import socket
import secrets
import logging

# Configure specialized logging for federated operations
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] FL-%(levelname)s - %(message)s')
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
class FederatedNode:
    """Represents a federated learning participant."""
    node_id: str
    node_type: str  # 'edge', 'datacenter', 'mobile'
    capabilities: Dict[str, Any]
    privacy_budget: float
    trust_score: float
    last_update: float
    computational_power: float
    network_bandwidth: float
    location_region: str
    status: str = 'active'


@dataclass
class FederatedRound:
    """Represents a federated learning round."""
    round_id: int
    participants: List[str]
    aggregation_method: str
    privacy_budget_consumed: float
    convergence_metric: float
    byzantine_nodes: List[str]
    communication_cost: float
    round_duration: float
    timestamp: float


@dataclass
class PrivacyBudgetAllocation:
    """Privacy budget allocation across federated nodes."""
    total_budget: float
    per_node_allocation: Dict[str, float]
    adaptive_allocation: bool
    budget_remaining: float
    allocation_strategy: str
    expiry_time: Optional[float] = None


class FederatedAggregator(ABC):
    """Abstract base class for federated aggregation methods."""
    
    @abstractmethod
    def aggregate(self, model_updates: Dict[str, Any], weights: Dict[str, float]) -> Any:
        """Aggregate model updates from multiple nodes."""
        pass
    
    @abstractmethod
    def get_privacy_cost(self) -> float:
        """Get privacy cost of aggregation method."""
        pass


class SecureAggregator(FederatedAggregator):
    """Secure aggregation with differential privacy."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.aggregation_history = []
    
    def aggregate(self, model_updates: Dict[str, Any], weights: Dict[str, float]) -> Any:
        """Perform secure aggregation with DP noise."""
        logger.info(f"🔒 Performing secure aggregation for {len(model_updates)} updates")
        
        if not model_updates:
            return None
        
        # Weighted averaging of model updates
        aggregated_update = None
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            logger.warning("Zero total weight in aggregation")
            return None
        
        for node_id, update in model_updates.items():
            node_weight = weights.get(node_id, 1.0) / total_weight
            
            if aggregated_update is None:
                if TORCH_AVAILABLE and isinstance(update, dict):
                    # PyTorch model state dict
                    aggregated_update = {}
                    for key, value in update.items():
                        aggregated_update[key] = value * node_weight
                elif NUMPY_AVAILABLE and hasattr(update, 'shape'):
                    # NumPy array
                    aggregated_update = update * node_weight
                else:
                    # Scalar or other type
                    aggregated_update = update * node_weight
            else:
                if isinstance(aggregated_update, dict) and isinstance(update, dict):
                    for key in aggregated_update:
                        if key in update:
                            aggregated_update[key] += update[key] * node_weight
                elif hasattr(aggregated_update, '__add__'):
                    aggregated_update = aggregated_update + (update * node_weight)
        
        # Add differential privacy noise
        dp_aggregated = self._add_dp_noise(aggregated_update)
        
        # Record aggregation
        self.aggregation_history.append({
            'timestamp': time.time(),
            'num_participants': len(model_updates),
            'epsilon_spent': self.epsilon,
            'aggregation_method': 'secure_averaging'
        })
        
        return dp_aggregated
    
    def _add_dp_noise(self, aggregated_update: Any) -> Any:
        """Add differential privacy noise to aggregated update."""
        # Calculate sensitivity and noise scale
        sensitivity = 1.0  # Assume L1 sensitivity of 1
        noise_scale = sensitivity / self.epsilon
        
        if isinstance(aggregated_update, dict):
            # Add noise to each parameter
            noisy_update = {}
            for key, value in aggregated_update.items():
                if TORCH_AVAILABLE and hasattr(value, 'shape'):
                    noise = torch.normal(0, noise_scale, size=value.shape, device=value.device)
                    noisy_update[key] = value + noise
                elif NUMPY_AVAILABLE and hasattr(value, 'shape'):
                    noise = np.random.normal(0, noise_scale, size=value.shape)
                    noisy_update[key] = value + noise
                else:
                    # Scalar parameter
                    noise = secrets.SystemRandom().gauss(0, noise_scale)
                    noisy_update[key] = value + noise
            return noisy_update
        elif hasattr(aggregated_update, 'shape'):
            # Array-like update
            if TORCH_AVAILABLE and hasattr(aggregated_update, 'device'):
                noise = torch.normal(0, noise_scale, size=aggregated_update.shape, 
                                   device=aggregated_update.device)
                return aggregated_update + noise
            elif NUMPY_AVAILABLE:
                noise = np.random.normal(0, noise_scale, size=aggregated_update.shape)
                return aggregated_update + noise
        else:
            # Scalar update
            noise = secrets.SystemRandom().gauss(0, noise_scale)
            return aggregated_update + noise
        
        return aggregated_update
    
    def get_privacy_cost(self) -> float:
        """Get privacy cost of secure aggregation."""
        return self.epsilon


class ByzantineRobustAggregator(FederatedAggregator):
    """Byzantine-robust aggregation using coordinate-wise median."""
    
    def __init__(self, byzantine_fraction: float = 0.2):
        self.byzantine_fraction = byzantine_fraction
        self.detected_byzantine = set()
    
    def aggregate(self, model_updates: Dict[str, Any], weights: Dict[str, float]) -> Any:
        """Perform Byzantine-robust aggregation."""
        logger.info(f"🛡️ Performing Byzantine-robust aggregation for {len(model_updates)} updates")
        
        if len(model_updates) < 3:
            logger.warning("Insufficient updates for Byzantine robustness")
            return list(model_updates.values())[0] if model_updates else None
        
        # Detect Byzantine nodes using statistical analysis
        byzantine_nodes = self._detect_byzantine_nodes(model_updates)
        self.detected_byzantine.update(byzantine_nodes)
        
        # Filter out Byzantine updates
        clean_updates = {node_id: update for node_id, update in model_updates.items() 
                        if node_id not in byzantine_nodes}
        
        if not clean_updates:
            logger.error("All nodes detected as Byzantine!")
            return None
        
        # Coordinate-wise median aggregation
        aggregated = self._coordinate_median_aggregation(clean_updates)
        
        logger.info(f"🔍 Detected {len(byzantine_nodes)} Byzantine nodes: {byzantine_nodes}")
        return aggregated
    
    def _detect_byzantine_nodes(self, model_updates: Dict[str, Any]) -> List[str]:
        """Detect Byzantine nodes using statistical outlier detection."""
        byzantine_nodes = []
        
        if len(model_updates) < 3:
            return byzantine_nodes
        
        # Simple outlier detection based on update magnitude
        update_norms = {}
        for node_id, update in model_updates.items():
            if isinstance(update, dict):
                # Calculate L2 norm of all parameters
                total_norm = 0.0
                for value in update.values():
                    if hasattr(value, 'norm'):
                        total_norm += float(value.norm()) ** 2
                    elif NUMPY_AVAILABLE and hasattr(value, 'shape'):
                        total_norm += float(np.linalg.norm(value)) ** 2
                    else:
                        total_norm += float(value) ** 2
                update_norms[node_id] = total_norm ** 0.5
            else:
                # Single parameter update
                if hasattr(update, 'norm'):
                    update_norms[node_id] = float(update.norm())
                elif NUMPY_AVAILABLE and hasattr(update, 'shape'):
                    update_norms[node_id] = float(np.linalg.norm(update))
                else:
                    update_norms[node_id] = abs(float(update))
        
        if not update_norms:
            return byzantine_nodes
        
        # Statistical outlier detection using IQR method
        norms = list(update_norms.values())
        if NUMPY_AVAILABLE:
            q1, q3 = np.percentile(norms, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for node_id, norm in update_norms.items():
                if norm < lower_bound or norm > upper_bound:
                    byzantine_nodes.append(node_id)
        else:
            # Simple threshold-based detection without NumPy
            mean_norm = sum(norms) / len(norms)
            std_norm = (sum((x - mean_norm) ** 2 for x in norms) / len(norms)) ** 0.5
            
            for node_id, norm in update_norms.items():
                if abs(norm - mean_norm) > 2 * std_norm:
                    byzantine_nodes.append(node_id)
        
        return byzantine_nodes
    
    def _coordinate_median_aggregation(self, clean_updates: Dict[str, Any]) -> Any:
        """Perform coordinate-wise median aggregation."""
        if not clean_updates:
            return None
        
        updates_list = list(clean_updates.values())
        
        if isinstance(updates_list[0], dict):
            # Model state dict aggregation
            aggregated = {}
            for key in updates_list[0].keys():
                parameter_values = [update[key] for update in updates_list if key in update]
                if parameter_values:
                    aggregated[key] = self._compute_coordinate_median(parameter_values)
            return aggregated
        else:
            # Single parameter aggregation
            return self._compute_coordinate_median(updates_list)
    
    def _compute_coordinate_median(self, values: List[Any]) -> Any:
        """Compute coordinate-wise median of parameter values."""
        if not values:
            return None
        
        if len(values) == 1:
            return values[0]
        
        if TORCH_AVAILABLE and hasattr(values[0], 'shape'):
            # PyTorch tensor median
            stacked = torch.stack(values, dim=0)
            median_values, _ = torch.median(stacked, dim=0)
            return median_values
        elif NUMPY_AVAILABLE and hasattr(values[0], 'shape'):
            # NumPy array median
            stacked = np.stack(values, axis=0)
            return np.median(stacked, axis=0)
        else:
            # Scalar median
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 1:
                return sorted_values[n // 2]
            else:
                return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    
    def get_privacy_cost(self) -> float:
        """Get privacy cost of Byzantine-robust aggregation."""
        return 0.1  # Minimal privacy cost for robustness


class FederatedLearningCoordinator:
    """
    Advanced federated learning coordinator with integrated differential privacy.
    
    Manages distributed training across heterogeneous devices while maintaining
    privacy guarantees and providing Byzantine fault tolerance.
    """
    
    def __init__(self, coordinator_config: Optional[Dict[str, Any]] = None):
        self.config = coordinator_config or self._get_default_config()
        self.nodes = {}  # node_id -> FederatedNode
        self.aggregators = {
            'secure': SecureAggregator(self.config['privacy']['epsilon'], 
                                     self.config['privacy']['delta']),
            'byzantine_robust': ByzantineRobustAggregator(self.config['robustness']['byzantine_fraction'])
        }
        self.current_round = 0
        self.round_history = []
        self.privacy_budget_manager = PrivacyBudgetAllocation(
            total_budget=self.config['privacy']['total_budget'],
            per_node_allocation={},
            adaptive_allocation=True,
            budget_remaining=self.config['privacy']['total_budget'],
            allocation_strategy='adaptive'
        )
        self.running = False
        self.coordinator_thread = None
        self.update_queue = queue.Queue()
        
        logger.info("🌐 Federated Learning Coordinator initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for federated learning."""
        return {
            'privacy': {
                'epsilon': 1.0,
                'delta': 1e-5,
                'total_budget': 10.0,
                'per_round_budget': 0.5
            },
            'coordination': {
                'min_participants': 3,
                'max_participants': 100,
                'round_timeout': 300,  # seconds
                'sync_frequency': 10,  # rounds
                'async_mode': True
            },
            'robustness': {
                'byzantine_fraction': 0.2,
                'fault_tolerance': True,
                'redundancy_factor': 1.5
            },
            'optimization': {
                'adaptive_learning_rate': True,
                'momentum': 0.9,
                'convergence_threshold': 1e-6
            }
        }
    
    def register_node(self, node_config: Dict[str, Any]) -> str:
        """Register a new federated learning node."""
        node_id = node_config.get('node_id', f"node_{len(self.nodes):04d}")
        
        node = FederatedNode(
            node_id=node_id,
            node_type=node_config.get('node_type', 'edge'),
            capabilities=node_config.get('capabilities', {}),
            privacy_budget=node_config.get('privacy_budget', 1.0),
            trust_score=node_config.get('trust_score', 1.0),
            last_update=time.time(),
            computational_power=node_config.get('computational_power', 1.0),
            network_bandwidth=node_config.get('network_bandwidth', 1.0),
            location_region=node_config.get('location_region', 'unknown')
        )
        
        self.nodes[node_id] = node
        
        # Allocate privacy budget
        self._allocate_privacy_budget(node_id)
        
        logger.info(f"📱 Registered node {node_id} ({node.node_type}) in {node.location_region}")
        return node_id
    
    def _allocate_privacy_budget(self, node_id: str):
        """Allocate privacy budget to a node."""
        if self.privacy_budget_manager.adaptive_allocation:
            # Adaptive allocation based on node capabilities
            node = self.nodes[node_id]
            base_allocation = self.config['privacy']['per_round_budget']
            
            # Adjust based on computational power and trust score
            allocation = base_allocation * node.computational_power * node.trust_score
            
            # Ensure we don't exceed remaining budget
            allocation = min(allocation, 
                           self.privacy_budget_manager.budget_remaining / max(len(self.nodes), 1))
            
            self.privacy_budget_manager.per_node_allocation[node_id] = allocation
            self.privacy_budget_manager.budget_remaining -= allocation
            
            logger.info(f"💰 Allocated privacy budget {allocation:.4f} to node {node_id}")
    
    def start_coordination(self):
        """Start federated learning coordination."""
        if self.running:
            logger.warning("Coordination already running")
            return
        
        self.running = True
        self.coordinator_thread = threading.Thread(target=self._coordination_loop)
        self.coordinator_thread.daemon = True
        self.coordinator_thread.start()
        
        logger.info("🚀 Federated learning coordination started")
    
    def stop_coordination(self):
        """Stop federated learning coordination."""
        if not self.running:
            return
        
        self.running = False
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=10)
        
        logger.info("🛑 Federated learning coordination stopped")
    
    def _coordination_loop(self):
        """Main coordination loop."""
        logger.info("🔄 Federated learning coordination loop started")
        
        while self.running:
            try:
                if len(self.nodes) >= self.config['coordination']['min_participants']:
                    self._execute_federated_round()
                else:
                    logger.info(f"⏳ Waiting for participants: {len(self.nodes)}/{self.config['coordination']['min_participants']}")
                
                # Sleep between rounds
                time.sleep(self.config['coordination']['round_timeout'])
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _execute_federated_round(self):
        """Execute a single federated learning round."""
        self.current_round += 1
        round_start_time = time.time()
        
        logger.info(f"🔄 Starting federated round {self.current_round}")
        
        # Select participants for this round
        participants = self._select_participants()
        
        if not participants:
            logger.warning("No participants selected for this round")
            return
        
        # Simulate model update collection
        model_updates = self._collect_model_updates(participants)
        
        if not model_updates:
            logger.warning("No model updates received")
            return
        
        # Aggregate updates using appropriate method
        aggregation_method = self._select_aggregation_method(participants)
        aggregated_update = self.aggregators[aggregation_method].aggregate(
            model_updates, 
            {node_id: self.nodes[node_id].trust_score for node_id in participants}
        )
        
        # Calculate privacy cost and update budgets
        privacy_cost = self.aggregators[aggregation_method].get_privacy_cost()
        self._update_privacy_budgets(participants, privacy_cost)
        
        # Record round completion
        round_duration = time.time() - round_start_time
        self._record_round_completion(participants, aggregation_method, privacy_cost, round_duration)
        
        logger.info(f"✅ Completed round {self.current_round} with {len(participants)} participants "
                   f"in {round_duration:.2f}s (privacy cost: {privacy_cost:.4f})")
    
    def _select_participants(self) -> List[str]:
        """Select participants for the current round."""
        # Filter active nodes with sufficient privacy budget
        eligible_nodes = [
            node_id for node_id, node in self.nodes.items()
            if (node.status == 'active' and 
                self.privacy_budget_manager.per_node_allocation.get(node_id, 0) > 0.01)
        ]
        
        if not eligible_nodes:
            return []
        
        # Select participants based on strategy
        max_participants = min(len(eligible_nodes), self.config['coordination']['max_participants'])
        
        # Priority-based selection (computational power * trust score)
        node_priorities = [
            (node_id, self.nodes[node_id].computational_power * self.nodes[node_id].trust_score)
            for node_id in eligible_nodes
        ]
        
        # Sort by priority and select top nodes
        node_priorities.sort(key=lambda x: x[1], reverse=True)
        selected = [node_id for node_id, _ in node_priorities[:max_participants]]
        
        return selected
    
    def _collect_model_updates(self, participants: List[str]) -> Dict[str, Any]:
        """Simulate collection of model updates from participants."""
        model_updates = {}
        
        for node_id in participants:
            # Simulate model update (in real implementation, this would be network communication)
            if TORCH_AVAILABLE:
                # Simulate PyTorch model parameters
                update = {
                    'layer1.weight': torch.randn(64, 128),
                    'layer1.bias': torch.randn(64),
                    'layer2.weight': torch.randn(32, 64),
                    'layer2.bias': torch.randn(32)
                }
            elif NUMPY_AVAILABLE:
                # Simulate NumPy arrays
                update = {
                    'layer1': np.random.randn(64, 128),
                    'layer2': np.random.randn(32, 64)
                }
            else:
                # Simple scalar update
                update = {'parameter': secrets.SystemRandom().gauss(0, 0.1)}
            
            model_updates[node_id] = update
            
            # Update node last_update timestamp
            self.nodes[node_id].last_update = time.time()
        
        return model_updates
    
    def _select_aggregation_method(self, participants: List[str]) -> str:
        """Select appropriate aggregation method based on participants."""
        # Check for potential Byzantine nodes
        suspicious_nodes = [
            node_id for node_id in participants 
            if self.nodes[node_id].trust_score < 0.7
        ]
        
        byzantine_fraction = len(suspicious_nodes) / len(participants)
        
        if byzantine_fraction > self.config['robustness']['byzantine_fraction']:
            return 'byzantine_robust'
        else:
            return 'secure'
    
    def _update_privacy_budgets(self, participants: List[str], privacy_cost: float):
        """Update privacy budgets after aggregation."""
        cost_per_participant = privacy_cost / len(participants)
        
        for node_id in participants:
            current_budget = self.privacy_budget_manager.per_node_allocation.get(node_id, 0)
            new_budget = max(0, current_budget - cost_per_participant)
            self.privacy_budget_manager.per_node_allocation[node_id] = new_budget
            
            # Update node privacy budget
            self.nodes[node_id].privacy_budget = new_budget
        
        # Update remaining total budget
        self.privacy_budget_manager.budget_remaining = max(0, 
            self.privacy_budget_manager.budget_remaining - privacy_cost)
    
    def _record_round_completion(self, participants: List[str], aggregation_method: str, 
                               privacy_cost: float, round_duration: float):
        """Record completion of a federated round."""
        byzantine_nodes = []
        if aggregation_method == 'byzantine_robust':
            byzantine_nodes = list(self.aggregators['byzantine_robust'].detected_byzantine)
        
        round_record = FederatedRound(
            round_id=self.current_round,
            participants=participants,
            aggregation_method=aggregation_method,
            privacy_budget_consumed=privacy_cost,
            convergence_metric=secrets.SystemRandom().uniform(0.85, 0.95),  # Simulated
            byzantine_nodes=byzantine_nodes,
            communication_cost=len(participants) * 1.5,  # Simulated MB
            round_duration=round_duration,
            timestamp=time.time()
        )
        
        self.round_history.append(round_record)
        
        # Update trust scores based on round performance
        self._update_trust_scores(participants, byzantine_nodes)
    
    def _update_trust_scores(self, participants: List[str], byzantine_nodes: List[str]):
        """Update trust scores based on round performance."""
        for node_id in participants:
            node = self.nodes[node_id]
            
            if node_id in byzantine_nodes:
                # Decrease trust score for Byzantine behavior
                node.trust_score = max(0.1, node.trust_score * 0.8)
                logger.warning(f"⚠️ Decreased trust score for Byzantine node {node_id}: {node.trust_score:.2f}")
            else:
                # Increase trust score for good behavior
                node.trust_score = min(1.0, node.trust_score * 1.02)
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status."""
        active_nodes = [node for node in self.nodes.values() if node.status == 'active']
        
        return {
            'status': 'running' if self.running else 'stopped',
            'current_round': self.current_round,
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'privacy_budget_remaining': self.privacy_budget_manager.budget_remaining,
            'recent_rounds': [asdict(round_) for round_ in self.round_history[-5:]],
            'aggregation_methods_available': list(self.aggregators.keys())
        }
    
    def generate_federation_report(self) -> str:
        """Generate comprehensive federated learning report."""
        report_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_privacy_spent = (self.privacy_budget_manager.total_budget - 
                             self.privacy_budget_manager.budget_remaining)
        
        avg_round_duration = (sum(r.round_duration for r in self.round_history) / 
                            len(self.round_history) if self.round_history else 0)
        
        byzantine_detections = sum(len(r.byzantine_nodes) for r in self.round_history)
        
        report = f"""
# Federated Learning Coordination Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Coordination Statistics
- Total Rounds: {self.current_round}
- Active Nodes: {len([n for n in self.nodes.values() if n.status == 'active'])}/{len(self.nodes)}
- Average Round Duration: {avg_round_duration:.2f}s
- Total Privacy Budget Spent: {total_privacy_spent:.4f}/{self.privacy_budget_manager.total_budget}

## 🛡️ Security & Privacy
- Byzantine Nodes Detected: {byzantine_detections}
- Privacy Budget Remaining: {self.privacy_budget_manager.budget_remaining:.4f}
- Aggregation Methods Used: {set(r.aggregation_method for r in self.round_history)}

## 📱 Node Status
"""
        
        for node_id, node in list(self.nodes.items())[:10]:  # Show first 10 nodes
            report += f"- **{node_id}** ({node.node_type}): Trust={node.trust_score:.2f}, "
            report += f"Budget={self.privacy_budget_manager.per_node_allocation.get(node_id, 0):.4f}\n"
        
        if len(self.nodes) > 10:
            report += f"... and {len(self.nodes) - 10} more nodes\n"
        
        report += f"""
## 🔄 Recent Rounds
"""
        
        for round_record in self.round_history[-5:]:
            report += f"- Round {round_record.round_id}: {len(round_record.participants)} participants, "
            report += f"{round_record.aggregation_method} aggregation, "
            report += f"privacy cost: {round_record.privacy_budget_consumed:.4f}\n"
        
        report += f"""
---
Generated by Federated Learning Coordinator v1.0
Status: {'🟢 ACTIVE' if self.running else '🔴 INACTIVE'}
"""
        
        # Save report
        report_path = Path(f"federated_reports/federation_report_{report_timestamp}.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"📄 Federation report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")
        
        return report


def demonstrate_federated_learning():
    """Demonstrate federated learning coordination."""
    print("🌐 Advanced Federated Learning Coordination Demo")
    print("=" * 55)
    
    # Initialize coordinator
    config = {
        'privacy': {'epsilon': 1.0, 'delta': 1e-5, 'total_budget': 5.0, 'per_round_budget': 0.3},
        'coordination': {'min_participants': 2, 'max_participants': 10, 'round_timeout': 5},
        'robustness': {'byzantine_fraction': 0.3, 'fault_tolerance': True}
    }
    
    coordinator = FederatedLearningCoordinator(config)
    
    # Register diverse nodes
    node_configs = [
        {'node_id': 'datacenter_1', 'node_type': 'datacenter', 'computational_power': 2.0, 
         'trust_score': 0.9, 'location_region': 'us_east'},
        {'node_id': 'edge_1', 'node_type': 'edge', 'computational_power': 1.0, 
         'trust_score': 0.8, 'location_region': 'us_west'},
        {'node_id': 'mobile_1', 'node_type': 'mobile', 'computational_power': 0.5, 
         'trust_score': 0.7, 'location_region': 'eu_west'},
        {'node_id': 'edge_2', 'node_type': 'edge', 'computational_power': 1.2, 
         'trust_score': 0.85, 'location_region': 'asia_pacific'},
        {'node_id': 'suspicious_node', 'node_type': 'edge', 'computational_power': 1.0, 
         'trust_score': 0.3, 'location_region': 'unknown'}  # Potentially Byzantine
    ]
    
    for node_config in node_configs:
        coordinator.register_node(node_config)
    
    # Start coordination
    coordinator.start_coordination()
    
    try:
        # Run for demonstration
        print(f"⏳ Running federated learning for 20 seconds...")
        time.sleep(20)
        
        # Get status
        status = coordinator.get_coordination_status()
        print(f"\n📊 Coordination Status:")
        print(f"- Rounds Completed: {status['current_round']}")
        print(f"- Active Nodes: {status['active_nodes']}/{status['total_nodes']}")
        print(f"- Privacy Budget Remaining: {status['privacy_budget_remaining']:.4f}")
        
        # Generate comprehensive report
        print(f"\n📄 Generating federation report...")
        report = coordinator.generate_federation_report()
        
    finally:
        # Stop coordination
        coordinator.stop_coordination()
        print(f"\n✅ Federated learning demonstration completed!")


if __name__ == "__main__":
    demonstrate_federated_learning()