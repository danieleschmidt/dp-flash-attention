"""
Privacy accounting for differentially private attention.

Implements Rényi Differential Privacy (RDP) accounting with composition,
converting to (ε, δ)-DP via the standard moment bound.

Key math:
  - Gaussian mechanism: M(x) = f(x) + N(0, σ²·I)
  - Sensitivity Δf = global L2 sensitivity of f
  - Noise scale σ = Δf · c / ε_rdp, where c calibrates to target (ε, δ)
  - RDP composition: add RDP values across steps, then convert to (ε, δ)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PrivacyBudget:
    """Cumulative privacy budget."""
    epsilon: float
    delta: float
    steps: int = 0

    def __repr__(self):
        return f"PrivacyBudget(ε={self.epsilon:.4f}, δ={self.delta:.2e}, steps={self.steps})"


@dataclass
class StepRecord:
    """Record of a single privacy step."""
    noise_scale: float
    sensitivity: float
    step: int


def _rdp_gaussian(alpha: float, noise_scale: float, sensitivity: float = 1.0) -> float:
    """
    RDP of the Gaussian mechanism at order alpha.

    M(x) = x + N(0, sigma^2), sensitivity Delta.
    RDP_alpha = alpha * Delta^2 / (2 * sigma^2)
    """
    return alpha * (sensitivity ** 2) / (2.0 * noise_scale ** 2)


def _rdp_to_dp(rdp_value: float, alpha: float, delta: float) -> float:
    """
    Convert RDP guarantee to (ε, δ)-DP via the standard bound.

    ε = rdp_value + log(1 - 1/alpha) - log(delta * (1 - 1/alpha) / alpha) / (alpha - 1)
    Simplified practical form (Balle et al. 2020, Proposition 3):
      ε = rdp_value - (log(delta) + log(1 - 1/alpha)) / (alpha - 1) + log(1 - 1/alpha)
    We use the tightest known conversion (Canonne et al. / Wang et al.):
      ε = min over alpha of: rdp(alpha) + log((alpha-1)/alpha) - (log(delta) + log(1-1/alpha))/(alpha-1)
    """
    if alpha <= 1.0:
        return float('inf')
    # Standard conversion: epsilon = rdp - log(delta*(alpha-1)/alpha) / (alpha-1)
    # = rdp + (log(alpha/(alpha-1)) - log(delta)) / (alpha-1)
    epsilon = rdp_value + (math.log(alpha / (alpha - 1)) - math.log(delta)) / (alpha - 1)
    return max(0.0, epsilon)


def calibrate_noise(
    target_epsilon: float,
    target_delta: float,
    sensitivity: float = 1.0,
    steps: int = 1,
    alpha_range: Optional[List[float]] = None,
) -> float:
    """
    Calibrate Gaussian noise scale σ to achieve (ε, δ)-DP over `steps` steps.

    Uses bisection on σ. For each σ, computes composed RDP over all steps,
    then converts to (ε, δ) and checks against target.

    Args:
        target_epsilon: Desired ε privacy budget
        target_delta: Desired δ
        sensitivity: L2 sensitivity of the mechanism
        steps: Number of composition steps (calls to the mechanism)
        alpha_range: Rényi orders to optimize over

    Returns:
        Minimum σ achieving (target_epsilon, target_delta)-DP
    """
    if target_epsilon == float('inf'):
        return 0.0  # No noise needed for infinite privacy budget

    if alpha_range is None:
        # Good set of orders for tight analysis
        alpha_range = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0, 32.0, 64.0]

    def achieves_target(sigma: float) -> bool:
        if sigma <= 0:
            return False
        # Compose RDP over steps
        best_epsilon = float('inf')
        for alpha in alpha_range:
            rdp_per_step = _rdp_gaussian(alpha, sigma, sensitivity)
            rdp_total = rdp_per_step * steps
            eps = _rdp_to_dp(rdp_total, alpha, target_delta)
            best_epsilon = min(best_epsilon, eps)
        return best_epsilon <= target_epsilon

    # Bisection search for minimum sigma
    lo, hi = 1e-6, 1e4
    if not achieves_target(hi):
        # Even with massive noise, can't achieve — shouldn't happen
        return hi

    for _ in range(64):  # 64 iterations → ~1e-19 precision
        mid = (lo + hi) / 2
        if achieves_target(mid):
            hi = mid
        else:
            lo = mid

    return hi


def compute_epsilon(
    noise_scale: float,
    delta: float,
    steps: int,
    sensitivity: float = 1.0,
    alpha_range: Optional[List[float]] = None,
) -> float:
    """
    Compute (ε, δ)-DP guarantee for Gaussian mechanism run `steps` times.

    Args:
        noise_scale: σ of the Gaussian noise
        delta: Target δ
        steps: Number of composition steps
        sensitivity: L2 sensitivity
        alpha_range: Rényi orders to optimize over

    Returns:
        Achieved ε (tightest bound)
    """
    if noise_scale <= 0:
        return float('inf')

    if alpha_range is None:
        alpha_range = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0, 32.0, 64.0]

    best_epsilon = float('inf')
    for alpha in alpha_range:
        rdp_per_step = _rdp_gaussian(alpha, noise_scale, sensitivity)
        rdp_total = rdp_per_step * steps
        eps = _rdp_to_dp(rdp_total, alpha, delta)
        best_epsilon = min(best_epsilon, eps)
    return best_epsilon


class PrivacyAccountant:
    """
    Tracks cumulative (ε, δ)-DP privacy budget across training steps.

    Uses Rényi DP composition: RDP values add linearly across independent
    mechanisms, then get converted to (ε, δ) at the end.

    Example::

        accountant = PrivacyAccountant(delta=1e-5)
        for batch in dataloader:
            # ... training step with DP attention ...
            accountant.record_step(noise_scale=sigma, sensitivity=1.0)

        budget = accountant.privacy_spent()
        print(f"Spent: ε={budget.epsilon:.2f}, δ={budget.delta:.2e}")
    """

    def __init__(
        self,
        delta: float = 1e-5,
        alpha_range: Optional[List[float]] = None,
    ):
        """
        Args:
            delta: Privacy parameter δ for (ε, δ)-DP reporting
            alpha_range: Rényi orders to track (default: good practical set)
        """
        self.delta = delta
        if alpha_range is None:
            self.alpha_range = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0, 32.0, 64.0]
        else:
            self.alpha_range = alpha_range

        # Accumulated RDP values per order
        self._rdp_accumulated = {alpha: 0.0 for alpha in self.alpha_range}
        self._steps: List[StepRecord] = []

    def record_step(self, noise_scale: float, sensitivity: float = 1.0) -> float:
        """
        Record one mechanism application and return current ε.

        Args:
            noise_scale: σ of Gaussian noise applied
            sensitivity: L2 sensitivity of the function being privatized

        Returns:
            Current cumulative ε
        """
        step_num = len(self._steps)
        self._steps.append(StepRecord(noise_scale, sensitivity, step_num))

        for alpha in self.alpha_range:
            rdp = 0.0 if noise_scale <= 0 else _rdp_gaussian(alpha, noise_scale, sensitivity)
            self._rdp_accumulated[alpha] += rdp

        return self.privacy_spent().epsilon

    def privacy_spent(self) -> PrivacyBudget:
        """
        Return current cumulative (ε, δ) privacy expenditure.

        Returns:
            PrivacyBudget with current epsilon, delta, and step count
        """
        if not self._steps:
            return PrivacyBudget(epsilon=0.0, delta=self.delta, steps=0)

        best_epsilon = float('inf')
        for alpha in self.alpha_range:
            rdp_total = self._rdp_accumulated[alpha]
            eps = _rdp_to_dp(rdp_total, alpha, self.delta)
            best_epsilon = min(best_epsilon, eps)

        return PrivacyBudget(
            epsilon=max(0.0, best_epsilon),
            delta=self.delta,
            steps=len(self._steps),
        )

    def reset(self):
        """Reset all accumulated privacy cost."""
        self._rdp_accumulated = {alpha: 0.0 for alpha in self.alpha_range}
        self._steps = []

    @property
    def num_steps(self) -> int:
        return len(self._steps)

    def __repr__(self):
        budget = self.privacy_spent()
        return f"PrivacyAccountant(steps={budget.steps}, {budget})"
