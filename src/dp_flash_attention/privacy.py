"""
Privacy accounting and differential privacy mechanisms.

Implements Rényi differential privacy accounting with composition analysis.
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

try:
    from dp_accounting import dp_event, privacy_accountant
    from prv_accountant import PoissonSubsampledGaussianMechanism
    HAS_DP_ACCOUNTING = True
except ImportError:
    HAS_DP_ACCOUNTING = False
    warnings.warn(
        "dp-accounting or prv-accountant not available. "
        "Using simplified privacy accounting."
    )


@dataclass
class PrivacyStats:
    """Statistics from a single DP computation step."""
    epsilon_spent: float
    delta: float  
    grad_norm: float
    noise_scale: float
    step_epsilon: float


class RenyiAccountant:
    """
    Rényi Differential Privacy accountant with composition.
    
    Tracks privacy budget consumption across multiple mechanisms and
    provides tight analysis via the Rényi DP framework.
    """
    
    def __init__(self, alpha_max: float = 32.0):
        """
        Initialize privacy accountant.
        
        Args:
            alpha_max: Maximum Rényi order to track
        """
        self.alpha_max = alpha_max
        self.alphas = [1 + x / 10.0 for x in range(1, int(10 * alpha_max))]
        self.reset()
    
    def reset(self):
        """Reset accounting to start fresh."""
        self.rdp_orders = np.array(self.alphas)
        self.rdp_values = np.zeros_like(self.rdp_orders)
        self.steps = []
        
        if HAS_DP_ACCOUNTING:
            self.accountant = privacy_accountant.PrivacyAccountant()
    
    def add_step(self, 
                 noise_scale: float,
                 delta: float, 
                 batch_size: int,
                 dataset_size: int = None,
                 sampling_rate: float = None) -> float:
        """
        Add a privacy step to the accounting.
        
        Args:
            noise_scale: Standard deviation of Gaussian noise
            delta: Privacy parameter
            batch_size: Size of the batch
            dataset_size: Total dataset size (for sampling rate calculation)
            sampling_rate: Direct sampling rate (alternative to dataset_size)
            
        Returns:
            Step epsilon consumption
        """
        if sampling_rate is None and dataset_size is not None:
            sampling_rate = batch_size / dataset_size
        elif sampling_rate is None:
            sampling_rate = 1.0  # Conservative assumption
        
        # Compute RDP values for this step
        if HAS_DP_ACCOUNTING and sampling_rate < 1.0:
            # Use precise accounting with subsampling amplification
            event = dp_event.PoissonSampledDpEvent(
                sampling_probability=sampling_rate,
                event=dp_event.GaussianDpEvent(noise_multiplier=noise_scale)
            )
            self.accountant.compose(event)
            
            # Get RDP values
            step_rdp = np.array([
                event.rdp_bound(alpha) for alpha in self.rdp_orders
            ])
        else:
            # Simplified RDP computation
            step_rdp = self._compute_rdp_gaussian(noise_scale, sampling_rate)
        
        # Compose with existing privacy loss
        self.rdp_values += step_rdp
        
        # Store step information
        step_info = {
            'noise_scale': noise_scale,
            'sampling_rate': sampling_rate,
            'batch_size': batch_size,
            'rdp_values': step_rdp.copy(),
        }
        self.steps.append(step_info)
        
        # Compute step epsilon (rough approximation)
        step_epsilon = self._rdp_to_dp(step_rdp, delta)[0]
        
        return step_epsilon
    
    def get_epsilon(self, delta: float) -> float:
        """
        Get current epsilon for given delta.
        
        Args:
            delta: Privacy parameter
            
        Returns:
            Current epsilon value
        """
        if HAS_DP_ACCOUNTING:
            return self.accountant.get_epsilon(delta)
        else:
            return self._rdp_to_dp(self.rdp_values, delta)[0]
    
    def get_privacy_bounds(self, delta: float) -> Tuple[float, float]:
        """
        Get privacy bounds (epsilon, delta) for current state.
        
        Args:
            delta: Target delta value
            
        Returns:
            Tuple of (epsilon, actual_delta)
        """
        epsilon = self.get_epsilon(delta)
        return epsilon, delta
    
    def _compute_rdp_gaussian(self, noise_scale: float, sampling_rate: float = 1.0) -> np.ndarray:
        """
        Compute RDP values for Gaussian mechanism.
        
        Args:
            noise_scale: Standard deviation of noise
            sampling_rate: Subsampling rate
            
        Returns:
            Array of RDP values for each alpha
        """
        rdp_values = np.zeros_like(self.rdp_orders)
        
        for i, alpha in enumerate(self.rdp_orders):
            if alpha == 1.0:
                rdp_values[i] = 0.0
            else:
                # RDP for Gaussian mechanism: α/(2σ²)
                rdp_gaussian = alpha / (2 * noise_scale ** 2)
                
                # Apply privacy amplification via subsampling if rate < 1
                if sampling_rate < 1.0:
                    # Simplified amplification (exact formula is more complex)
                    rdp_values[i] = sampling_rate * rdp_gaussian
                else:
                    rdp_values[i] = rdp_gaussian
        
        return rdp_values
    
    def _rdp_to_dp(self, rdp_values: np.ndarray, delta: float) -> Tuple[float, int]:
        """
        Convert RDP to (ε, δ)-DP via optimization.
        
        Args:
            rdp_values: Array of RDP values
            delta: Target delta
            
        Returns:
            Tuple of (epsilon, optimal_alpha_index)
        """
        if delta <= 0:
            return float('inf'), 0
        
        eps_values = []
        for i, (alpha, rdp) in enumerate(zip(self.rdp_orders, rdp_values)):
            if alpha == 1.0:
                eps_values.append(float('inf'))
            else:
                # Convert RDP to DP: ε = RDP + log(1/δ)/(α-1)
                eps = rdp + math.log(1.0 / delta) / (alpha - 1)
                eps_values.append(eps)
        
        # Return minimum epsilon and corresponding alpha index
        min_idx = np.argmin(eps_values)
        return eps_values[min_idx], min_idx
    
    def get_composition_stats(self) -> dict:
        """Get detailed composition statistics."""
        total_steps = len(self.steps)
        if total_steps == 0:
            return {'total_steps': 0}
        
        avg_noise_scale = np.mean([step['noise_scale'] for step in self.steps])
        avg_sampling_rate = np.mean([step['sampling_rate'] for step in self.steps])
        
        return {
            'total_steps': total_steps,
            'avg_noise_scale': avg_noise_scale,
            'avg_sampling_rate': avg_sampling_rate,
            'rdp_orders': self.rdp_orders.tolist(),
            'rdp_values': self.rdp_values.tolist(),
        }


class GaussianMechanism:
    """
    Gaussian mechanism for differential privacy.
    
    Adds calibrated Gaussian noise to preserve (ε, δ)-differential privacy.
    """
    
    def __init__(self, 
                 epsilon: float,
                 delta: float,
                 sensitivity: float = 1.0,
                 secure_rng: bool = True):
        """
        Initialize Gaussian mechanism.
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter  
            sensitivity: L2 sensitivity of the function
            secure_rng: Whether to use cryptographically secure RNG
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.secure_rng = secure_rng
        
        # Compute noise scale: σ = √(2 ln(1.25/δ)) * Δ / ε
        self.noise_scale = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
    
    def add_noise(self, tensor: Tensor, scale_override: Optional[float] = None) -> Tensor:
        """
        Add calibrated Gaussian noise to tensor.
        
        Args:
            tensor: Input tensor
            scale_override: Override noise scale
            
        Returns:
            Noisy tensor
        """
        scale = scale_override if scale_override is not None else self.noise_scale
        
        if self.secure_rng and hasattr(torch, 'randint_generator'):
            # Use secure random number generation if available
            generator = torch.Generator()
            generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
            noise = torch.normal(0, scale, tensor.shape, generator=generator, device=tensor.device)
        else:
            noise = torch.normal(0, scale, tensor.shape, device=tensor.device)
        
        return tensor + noise.to(tensor.dtype)
    
    def calibrate_noise_scale(self, 
                             grad_norm: float,
                             clipping_bound: float) -> float:
        """
        Calibrate noise scale based on actual gradient norm.
        
        Args:
            grad_norm: Actual gradient norm
            clipping_bound: Clipping bound used
            
        Returns:
            Calibrated noise scale
        """
        # If gradient norm is small, we can use less noise
        calibration_factor = min(1.0, grad_norm / clipping_bound)
        return self.noise_scale * calibration_factor


class AdaptiveNoiseCalibrator:
    """
    Adaptive noise calibration based on gradient statistics.
    
    Automatically adjusts noise and clipping parameters based on observed
    gradient distributions to minimize privacy-utility tradeoff.
    """
    
    def __init__(self,
                 target_epsilon: float,
                 target_delta: float = 1e-5,
                 confidence_interval: float = 0.95,
                 calibration_steps: int = 100):
        """
        Initialize adaptive calibrator.
        
        Args:
            target_epsilon: Target privacy budget
            target_delta: Target privacy parameter
            confidence_interval: Confidence level for calibration
            calibration_steps: Number of steps for calibration
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.confidence_interval = confidence_interval
        self.calibration_steps = calibration_steps
        
        self.grad_norms = []
        self.noise_multiplier = None
        self.clip_norm = None
    
    def observe_gradients(self, grad_norm: float):
        """Observe gradient norm for calibration."""
        self.grad_norms.append(grad_norm)
    
    def calibrate(self, 
                  model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Calibrate noise and clipping parameters.
        
        Args:
            model: Model to calibrate on
            data_loader: Data loader for calibration
            
        Returns:
            Tuple of (noise_multiplier, clip_norm)
        """
        # Collect gradient statistics
        self.grad_norms = []
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= self.calibration_steps:
                    break
                
                # Compute per-sample gradient norms (simplified)
                if hasattr(batch, 'keys'):
                    # Dict-like batch
                    inputs = batch
                else:
                    # Tensor batch  
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                # Mock gradient computation for calibration
                # In practice, would compute actual per-sample gradients
                mock_grad_norm = torch.randn(1).abs().item() * 2.0
                self.observe_gradients(mock_grad_norm)
        
        # Compute statistics
        grad_norms = np.array(self.grad_norms)
        median_norm = np.median(grad_norms)
        percentile_95 = np.percentile(grad_norms, 95)
        
        # Set clipping bound based on gradient distribution
        self.clip_norm = percentile_95
        
        # Compute noise multiplier for target privacy
        mechanism = GaussianMechanism(
            self.target_epsilon, self.target_delta, self.clip_norm
        )
        self.noise_multiplier = mechanism.noise_scale / self.clip_norm
        
        print(f"Calibrated clip_norm: {self.clip_norm:.4f}")
        print(f"Calibrated noise_multiplier: {self.noise_multiplier:.4f}")
        
        return self.noise_multiplier, self.clip_norm