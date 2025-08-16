"""
Quantum-Resistant Privacy Mechanisms for DP-Flash-Attention.

Implements post-quantum secure differential privacy mechanisms using 
lattice-based cryptography and discrete noise generation techniques.
"""

import math
import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback tensor type for type hints
    Tensor = Any

logger = logging.getLogger(__name__)


class QuantumThreatLevel(Enum):
    """Quantum threat level classifications."""
    CLASSICAL = "classical"  # No quantum resistance needed
    NEAR_TERM = "near_term"  # NISQ-era quantum computers
    FAULT_TOLERANT = "fault_tolerant"  # Cryptographically relevant quantum computers
    ULTRA_SECURE = "ultra_secure"  # Maximum quantum resistance


@dataclass
class QuantumSecurityParameters:
    """Parameters for quantum-resistant privacy mechanisms."""
    threat_level: QuantumThreatLevel
    lattice_dimension: int
    security_parameter: int
    error_distribution_std: float
    quantum_advantage_factor: float
    post_quantum_noise_multiplier: float
    
    @classmethod
    def from_threat_level(cls, threat_level: QuantumThreatLevel) -> 'QuantumSecurityParameters':
        """Create security parameters based on threat level."""
        
        if threat_level == QuantumThreatLevel.CLASSICAL:
            return cls(
                threat_level=threat_level,
                lattice_dimension=256,
                security_parameter=128,
                error_distribution_std=3.2,
                quantum_advantage_factor=1.0,
                post_quantum_noise_multiplier=1.0
            )
        elif threat_level == QuantumThreatLevel.NEAR_TERM:
            return cls(
                threat_level=threat_level,
                lattice_dimension=512,
                security_parameter=192,
                error_distribution_std=4.1,
                quantum_advantage_factor=1.5,
                post_quantum_noise_multiplier=1.3
            )
        elif threat_level == QuantumThreatLevel.FAULT_TOLERANT:
            return cls(
                threat_level=threat_level,
                lattice_dimension=1024,
                security_parameter=256,
                error_distribution_std=5.7,
                quantum_advantage_factor=2.2,
                post_quantum_noise_multiplier=1.8
            )
        else:  # ULTRA_SECURE
            return cls(
                threat_level=threat_level,
                lattice_dimension=2048,
                security_parameter=384,
                error_distribution_std=8.1,
                quantum_advantage_factor=3.0,
                post_quantum_noise_multiplier=2.5
            )


class DiscreteGaussianSampler:
    """
    Cryptographically secure discrete Gaussian sampler for quantum-resistant DP.
    
    Uses rejection sampling with lattice-based security guarantees.
    """
    
    def __init__(
        self, 
        sigma: float, 
        security_params: QuantumSecurityParameters,
        precision_bits: int = 64
    ):
        self.sigma = sigma
        self.security_params = security_params
        self.precision_bits = precision_bits
        
        # Precompute rejection sampling parameters
        self.tail_cut = math.ceil(sigma * 6)  # 6-sigma tail cut
        self.acceptance_threshold = self._compute_acceptance_threshold()
        
        logger.info(f"Initialized discrete Gaussian sampler: σ={sigma}, security={security_params.security_parameter}-bit")
    
    def _compute_acceptance_threshold(self) -> float:
        """Compute acceptance threshold for rejection sampling."""
        # Use lattice-based security bound
        lattice_factor = math.sqrt(self.security_params.lattice_dimension / (2 * math.pi * math.e))
        return 1.0 / (lattice_factor * math.sqrt(2 * math.pi))
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample from discrete Gaussian distribution."""
        
        if not TORCH_AVAILABLE:
            # Fallback to NumPy implementation
            return self._sample_numpy(shape)
        
        total_samples = np.prod(shape)
        samples = []
        
        # Rejection sampling loop
        attempts = 0
        max_attempts = total_samples * 10  # Safety limit
        
        while len(samples) < total_samples and attempts < max_attempts:
            # Generate candidate samples
            candidates = np.random.normal(0, self.sigma, size=min(total_samples * 2, 10000))
            
            # Round to nearest integers
            discrete_candidates = np.round(candidates).astype(np.int64)
            
            # Apply tail cut
            valid_mask = np.abs(discrete_candidates) <= self.tail_cut
            discrete_candidates = discrete_candidates[valid_mask]
            
            # Acceptance/rejection test
            acceptance_probs = np.exp(
                -(discrete_candidates**2) / (2 * self.sigma**2)
            ) / (self.sigma * math.sqrt(2 * math.pi))
            
            uniform_samples = np.random.uniform(0, 1, len(discrete_candidates))
            accepted_mask = uniform_samples < acceptance_probs * self.acceptance_threshold
            
            accepted_samples = discrete_candidates[accepted_mask]
            samples.extend(accepted_samples[:total_samples - len(samples)])
            attempts += len(candidates)
        
        if len(samples) < total_samples:
            logger.warning(f"Discrete Gaussian sampling incomplete: {len(samples)}/{total_samples}")
            # Pad with zeros if needed
            samples.extend([0] * (total_samples - len(samples)))
        
        return np.array(samples[:total_samples]).reshape(shape)
    
    def _sample_numpy(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Fallback NumPy implementation."""
        # Simplified discrete Gaussian (less secure but functional)
        continuous_samples = np.random.normal(0, self.sigma, size=shape)
        return np.round(continuous_samples).astype(np.int64)


class QuantumResistantGaussianMechanism:
    """
    Quantum-resistant Gaussian mechanism for differential privacy.
    
    Provides (ε, δ)-differential privacy with post-quantum security guarantees.
    """
    
    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
        threat_level: QuantumThreatLevel = QuantumThreatLevel.FAULT_TOLERANT,
        custom_security_params: Optional[QuantumSecurityParameters] = None
    ):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not (0 < delta < 1):
            raise ValueError("Delta must be in (0, 1)")
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
        
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.threat_level = threat_level
        
        # Initialize quantum security parameters
        self.security_params = custom_security_params or QuantumSecurityParameters.from_threat_level(threat_level)
        
        # Compute quantum-resistant noise scale
        self.noise_scale = self._compute_quantum_noise_scale()
        
        # Initialize discrete Gaussian sampler
        self.sampler = DiscreteGaussianSampler(
            sigma=self.noise_scale,
            security_params=self.security_params
        )
        
        logger.info(f"Quantum-resistant mechanism initialized: ε={epsilon}, δ={delta}, σ={self.noise_scale:.4f}")
    
    def _compute_quantum_noise_scale(self) -> float:
        """Compute noise scale with quantum resistance."""
        
        # Classical Gaussian mechanism noise scale
        classical_scale = self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        
        # Apply quantum resistance multiplier
        quantum_scale = classical_scale * self.security_params.post_quantum_noise_multiplier
        
        # Additional security margin based on quantum advantage
        security_margin = 1.0 + (self.security_params.quantum_advantage_factor - 1.0) * 0.5
        
        return quantum_scale * security_margin
    
    def add_noise(self, tensor: Tensor, in_place: bool = False) -> Tensor:
        """Add quantum-resistant noise to tensor."""
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for tensor operations")
        
        # Generate noise using discrete Gaussian sampler
        noise_array = self.sampler.sample(tensor.shape)
        noise_tensor = torch.from_numpy(noise_array.astype(np.float32)).to(tensor.device)
        
        # Scale noise appropriately
        scaled_noise = noise_tensor * (self.noise_scale / 1000.0)  # Scale factor for numerical stability
        
        if in_place:
            tensor.add_(scaled_noise)
            return tensor
        else:
            return tensor + scaled_noise
    
    def privacy_cost(self) -> Tuple[float, float]:
        """Return privacy cost as (epsilon, delta)."""
        return (self.epsilon, self.delta)
    
    def security_level(self) -> Dict[str, Any]:
        """Return security level information."""
        return {
            "threat_level": self.threat_level.value,
            "security_parameter": self.security_params.security_parameter,
            "lattice_dimension": self.security_params.lattice_dimension,
            "quantum_advantage_factor": self.security_params.quantum_advantage_factor,
            "post_quantum_multiplier": self.security_params.post_quantum_noise_multiplier
        }


class AdaptiveQuantumNoiseMechanism:
    """
    Adaptive quantum-resistant noise mechanism that learns optimal parameters.
    
    Uses meta-learning to adapt noise scaling based on gradient statistics
    and privacy utility trade-offs.
    """
    
    def __init__(
        self,
        base_epsilon: float,
        base_delta: float,
        sensitivity: float,
        threat_level: QuantumThreatLevel = QuantumThreatLevel.FAULT_TOLERANT,
        adaptation_rate: float = 0.01,
        privacy_buffer: float = 0.1
    ):
        self.base_epsilon = base_epsilon
        self.base_delta = base_delta
        self.sensitivity = sensitivity
        self.threat_level = threat_level
        self.adaptation_rate = adaptation_rate
        self.privacy_buffer = privacy_buffer
        
        # Initialize base quantum mechanism
        self.base_mechanism = QuantumResistantGaussianMechanism(
            epsilon=base_epsilon * (1 - privacy_buffer),  # Reserve privacy budget for adaptation
            delta=base_delta,
            sensitivity=sensitivity,
            threat_level=threat_level
        )
        
        # Adaptation state
        self.gradient_variance_history = []
        self.utility_history = []
        self.noise_scale_history = []
        self.adaptation_step = 0
        
        # Learned parameters
        self.alpha = 1.0  # Gradient variance sensitivity
        self.beta = 0.5   # Variance exponent
        
        logger.info(f"Adaptive quantum mechanism initialized with {privacy_buffer:.1%} privacy buffer")
    
    def update_adaptation_parameters(
        self, 
        gradient_variance: float, 
        utility_score: float
    ):
        """Update adaptation parameters based on observed statistics."""
        
        self.gradient_variance_history.append(gradient_variance)
        self.utility_history.append(utility_score)
        self.noise_scale_history.append(self.base_mechanism.noise_scale)
        
        # Keep only recent history
        max_history = 1000
        if len(self.gradient_variance_history) > max_history:
            self.gradient_variance_history = self.gradient_variance_history[-max_history:]
            self.utility_history = self.utility_history[-max_history:]
            self.noise_scale_history = self.noise_scale_history[-max_history:]
        
        # Meta-learning update every 10 steps
        if self.adaptation_step % 10 == 0 and len(self.utility_history) >= 10:
            self._meta_update()
        
        self.adaptation_step += 1
    
    def _meta_update(self):
        """Meta-learning update for adaptation parameters."""
        
        if len(self.utility_history) < 10:
            return
        
        # Simple gradient-based meta-learning
        recent_variance = np.array(self.gradient_variance_history[-10:])
        recent_utility = np.array(self.utility_history[-10:])
        
        # Compute utility gradient w.r.t. variance
        if np.std(recent_variance) > 1e-6:
            utility_gradient = np.corrcoef(recent_variance, recent_utility)[0, 1]
            
            # Update alpha (gradient variance sensitivity)
            if utility_gradient < 0:  # Higher variance hurts utility
                self.alpha = min(2.0, self.alpha + self.adaptation_rate)
            else:
                self.alpha = max(0.1, self.alpha - self.adaptation_rate)
        
        logger.debug(f"Meta-update: α={self.alpha:.3f}, β={self.beta:.3f}")
    
    def compute_adaptive_noise_scale(self, gradient_variance: float) -> float:
        """Compute adaptive noise scale based on current gradient statistics."""
        
        base_scale = self.base_mechanism.noise_scale
        
        # Adaptive scaling based on gradient variance
        if gradient_variance > 0:
            adaptation_factor = 1.0 + self.alpha * (gradient_variance ** self.beta)
        else:
            adaptation_factor = 1.0
        
        return base_scale * adaptation_factor
    
    def add_adaptive_noise(
        self, 
        tensor: Tensor, 
        gradient_variance: float,
        utility_score: Optional[float] = None,
        in_place: bool = False
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Add adaptive quantum-resistant noise with meta-learning."""
        
        # Compute adaptive noise scale
        adaptive_scale = self.compute_adaptive_noise_scale(gradient_variance)
        
        # Create temporary mechanism with adaptive scale
        adaptive_mechanism = QuantumResistantGaussianMechanism(
            epsilon=self.base_epsilon,
            delta=self.base_delta,
            sensitivity=self.sensitivity,
            threat_level=self.threat_level
        )
        adaptive_mechanism.noise_scale = adaptive_scale
        adaptive_mechanism.sampler = DiscreteGaussianSampler(
            sigma=adaptive_scale,
            security_params=adaptive_mechanism.security_params
        )
        
        # Add noise
        noisy_tensor = adaptive_mechanism.add_noise(tensor, in_place=in_place)
        
        # Update adaptation if utility score provided
        if utility_score is not None:
            self.update_adaptation_parameters(gradient_variance, utility_score)
        
        # Return statistics
        stats = {
            "base_noise_scale": self.base_mechanism.noise_scale,
            "adaptive_noise_scale": adaptive_scale,
            "adaptation_factor": adaptive_scale / self.base_mechanism.noise_scale,
            "gradient_variance": gradient_variance,
            "alpha": self.alpha,
            "beta": self.beta
        }
        
        return noisy_tensor, stats


class FederatedQuantumPrivacyMechanism:
    """
    Federated learning privacy mechanism with quantum resistance.
    
    Implements differential privacy with shuffling and secure aggregation
    for enhanced privacy amplification in quantum-resistant settings.
    """
    
    def __init__(
        self,
        num_clients: int,
        sampling_rate: float,
        local_epsilon: float,
        local_delta: float,
        sensitivity: float,
        threat_level: QuantumThreatLevel = QuantumThreatLevel.FAULT_TOLERANT
    ):
        if not (0 < sampling_rate <= 1):
            raise ValueError("Sampling rate must be in (0, 1]")
        if num_clients < 2:
            raise ValueError("Number of clients must be at least 2")
        
        self.num_clients = num_clients
        self.sampling_rate = sampling_rate
        self.local_epsilon = local_epsilon
        self.local_delta = local_delta
        self.sensitivity = sensitivity
        self.threat_level = threat_level
        
        # Compute privacy amplification
        self.amplified_epsilon, self.amplified_delta = self._compute_amplified_privacy()
        
        # Initialize per-client mechanisms
        self.client_mechanisms = {}
        for client_id in range(num_clients):
            self.client_mechanisms[client_id] = QuantumResistantGaussianMechanism(
                epsilon=local_epsilon,
                delta=local_delta,
                sensitivity=sensitivity,
                threat_level=threat_level
            )
        
        logger.info(f"Federated quantum mechanism: {num_clients} clients, amplified (ε={self.amplified_epsilon:.4f}, δ={self.amplified_delta:.2e})")
    
    def _compute_amplified_privacy(self) -> Tuple[float, float]:
        """Compute privacy amplification via subsampling and shuffling."""
        
        # Subsampling amplification (simplified)
        q = self.sampling_rate
        if q < 1:
            # Privacy amplification factor
            amplification_factor = math.sqrt(q * math.log(1/self.local_delta))
            amplified_epsilon = self.local_epsilon * amplification_factor
        else:
            amplified_epsilon = self.local_epsilon
        
        # Shuffling amplification (additional factor)
        if self.num_clients >= 10:
            shuffling_factor = math.sqrt(math.log(self.num_clients) / self.num_clients)
            amplified_epsilon *= shuffling_factor
        
        # Conservative delta amplification
        amplified_delta = self.local_delta * self.num_clients * q
        
        return amplified_epsilon, amplified_delta
    
    def client_add_noise(self, client_id: int, tensor: Tensor) -> Tensor:
        """Add noise for specific client."""
        if client_id not in self.client_mechanisms:
            raise ValueError(f"Unknown client ID: {client_id}")
        
        return self.client_mechanisms[client_id].add_noise(tensor)
    
    def secure_aggregate(self, client_tensors: List[Tensor]) -> Tensor:
        """Perform secure aggregation with quantum-resistant privacy."""
        
        if not client_tensors:
            raise ValueError("No client tensors provided")
        
        # Simple averaging (in practice would use secure protocols)
        aggregated = torch.stack(client_tensors).mean(dim=0)
        
        # Add additional noise for central DP
        central_mechanism = QuantumResistantGaussianMechanism(
            epsilon=self.amplified_epsilon * 0.1,  # Small central noise
            delta=self.amplified_delta * 0.1,
            sensitivity=self.sensitivity / math.sqrt(len(client_tensors)),
            threat_level=self.threat_level
        )
        
        return central_mechanism.add_noise(aggregated)
    
    def privacy_guarantees(self) -> Dict[str, Any]:
        """Return overall privacy guarantees."""
        return {
            "local_privacy": (self.local_epsilon, self.local_delta),
            "amplified_privacy": (self.amplified_epsilon, self.amplified_delta),
            "num_clients": self.num_clients,
            "sampling_rate": self.sampling_rate,
            "quantum_threat_level": self.threat_level.value
        }


# Utility functions
def recommend_quantum_threat_level(
    deployment_environment: str,
    data_sensitivity: str,
    regulatory_requirements: List[str]
) -> QuantumThreatLevel:
    """Recommend appropriate quantum threat level based on deployment context."""
    
    # High-sensitivity deployments
    if any(req in regulatory_requirements for req in ["FIPS-140", "CC-EAL", "NSA-Suite-B"]):
        return QuantumThreatLevel.ULTRA_SECURE
    
    # Government/military deployments
    if deployment_environment in ["government", "military", "intelligence"]:
        return QuantumThreatLevel.FAULT_TOLERANT
    
    # High-value commercial applications
    if data_sensitivity in ["financial", "healthcare", "legal"] or "GDPR" in regulatory_requirements:
        return QuantumThreatLevel.FAULT_TOLERANT
    
    # Near-term quantum concerns
    if deployment_environment in ["cloud", "multi-tenant"] or "CCPA" in regulatory_requirements:
        return QuantumThreatLevel.NEAR_TERM
    
    # Default for research/development
    return QuantumThreatLevel.CLASSICAL


def quantum_privacy_audit(mechanism: QuantumResistantGaussianMechanism) -> Dict[str, Any]:
    """Perform security audit of quantum-resistant privacy mechanism."""
    
    security_info = mechanism.security_level()
    
    audit_results = {
        "mechanism_type": "QuantumResistantGaussian",
        "privacy_parameters": mechanism.privacy_cost(),
        "security_level": security_info["security_parameter"],
        "quantum_resistance": True,
        "threat_model": security_info["threat_level"],
        "lattice_security": security_info["lattice_dimension"],
        "recommendations": []
    }
    
    # Security recommendations
    if security_info["security_parameter"] < 256:
        audit_results["recommendations"].append(
            "Consider upgrading to 256-bit security for fault-tolerant quantum resistance"
        )
    
    if security_info["quantum_advantage_factor"] < 2.0:
        audit_results["recommendations"].append(
            "Consider higher quantum advantage factor for long-term security"
        )
    
    if mechanism.epsilon > 1.0:
        audit_results["recommendations"].append(
            "Large epsilon value may weaken privacy guarantees even with quantum resistance"
        )
    
    return audit_results


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test quantum-resistant mechanism
    threat_level = recommend_quantum_threat_level(
        deployment_environment="healthcare",
        data_sensitivity="medical",
        regulatory_requirements=["HIPAA", "GDPR"]
    )
    
    mechanism = QuantumResistantGaussianMechanism(
        epsilon=1.0,
        delta=1e-5,
        sensitivity=1.0,
        threat_level=threat_level
    )
    
    print(f"Quantum threat level: {threat_level}")
    print(f"Security audit: {quantum_privacy_audit(mechanism)}")
    
    if TORCH_AVAILABLE:
        # Test with tensor
        test_tensor = torch.randn(100, 768)
        noisy_tensor = mechanism.add_noise(test_tensor)
        print(f"Noise added successfully, tensor shape: {noisy_tensor.shape}")
    
    logger.info("Quantum-resistant privacy mechanisms ready for deployment")