#!/usr/bin/env python3
"""
Comprehensive Experimental Framework for DP-Flash-Attention Research.

Implements reproducible experimental designs, statistical analysis, and 
benchmark evaluation for privacy-preserving attention mechanism research.
"""

import os
import json
import logging
import time
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import numpy as np
from pathlib import Path

# Optional dependencies for advanced functionality
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy import stats
    import scipy.stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of experiments in the research framework."""
    PRIVACY_MECHANISM_COMPARISON = "privacy_mechanism_comparison"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    UTILITY_PRIVACY_TRADEOFF = "utility_privacy_tradeoff"
    HARDWARE_OPTIMIZATION = "hardware_optimization"
    REAL_WORLD_EVALUATION = "real_world_evaluation"
    THEORETICAL_VALIDATION = "theoretical_validation"


class ModelArchitecture(Enum):
    """Supported model architectures for experiments."""
    BERT_BASE = "bert-base"
    BERT_LARGE = "bert-large"
    GPT2_SMALL = "gpt2-small"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    VIT_BASE = "vit-base"
    VIT_LARGE = "vit-large"
    T5_SMALL = "t5-small"
    T5_BASE = "t5-base"


class Dataset(Enum):
    """Supported datasets for experiments."""
    GLUE_COLA = "glue-cola"
    GLUE_SST2 = "glue-sst2"
    GLUE_MRPC = "glue-mrpc"
    GLUE_QNLI = "glue-qnli"
    WIKITEXT_103 = "wikitext-103"
    IMAGENET_1K = "imagenet-1k"
    CIFAR_10 = "cifar-10"
    SQUAD_V1 = "squad-v1"
    MEDICAL_NER = "medical-ner"
    FINANCIAL_SENTIMENT = "financial-sentiment"


@dataclass
class ExperimentConfiguration:
    """Configuration for a single experiment."""
    experiment_id: str
    experiment_type: ExperimentType
    name: str
    description: str
    
    # Model and data configuration
    model_architecture: ModelArchitecture
    dataset: Dataset
    sequence_length: int
    batch_size: int
    
    # Privacy configuration
    privacy_mechanisms: List[str]
    epsilon_values: List[float]
    delta_values: List[float]
    sensitivity: float
    
    # Training configuration
    num_epochs: int
    learning_rate: float
    warmup_steps: int
    
    # Experimental design
    num_trials: int
    random_seeds: List[int]
    statistical_power: float
    confidence_level: float
    
    # Resource constraints
    max_memory_gb: Optional[int] = None
    max_time_hours: Optional[int] = None
    gpu_type: Optional[str] = None
    
    # Output configuration
    save_intermediate_results: bool = True
    generate_plots: bool = True
    export_data: bool = True


@dataclass
class ExperimentResult:
    """Results from a single experimental trial."""
    trial_id: str
    experiment_id: str
    configuration: Dict[str, Any]
    
    # Performance metrics
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    
    # Privacy metrics
    epsilon_spent: Optional[float] = None
    delta_spent: Optional[float] = None
    privacy_cost: Optional[float] = None
    
    # Computational metrics
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    
    # Statistical metrics
    gradient_norm: Optional[float] = None
    noise_scale: Optional[float] = None
    convergence_epoch: Optional[int] = None
    
    # Metadata
    timestamp: str = ""
    random_seed: int = 0
    hardware_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.hardware_info is None:
            self.hardware_info = {}


class StatisticalAnalyzer:
    """Statistical analysis tools for experimental results."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def descriptive_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute descriptive statistics for a list of values."""
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "iqr": np.percentile(values, 75) - np.percentile(values, 25)
        }
    
    def confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Compute confidence interval for mean."""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean = statistics.mean(values)
        std_err = statistics.stdev(values) / np.sqrt(len(values))
        
        if SCIPY_AVAILABLE:
            # Use t-distribution for small samples
            t_critical = scipy_stats.t.ppf(1 - self.alpha/2, len(values) - 1)
            margin_error = t_critical * std_err
        else:
            # Approximate with normal distribution
            z_critical = 1.96  # 95% confidence
            margin_error = z_critical * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def two_sample_ttest(
        self, 
        group1: List[float], 
        group2: List[float],
        equal_var: bool = False
    ) -> Dict[str, Any]:
        """Perform two-sample t-test."""
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available for statistical testing"}
        
        if len(group1) < 2 or len(group2) < 2:
            return {"error": "Insufficient data for t-test"}
        
        statistic, p_value = scipy_stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) + 
                             (len(group2) - 1) * np.var(group2)) / 
                            (len(group1) + len(group2) - 2))
        
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0
        
        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "cohens_d": cohens_d,
            "effect_size_interpretation": self._interpret_effect_size(abs(cohens_d))
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def anova_test(self, groups: List[List[float]]) -> Dict[str, Any]:
        """Perform one-way ANOVA test."""
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available for ANOVA"}
        
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for ANOVA"}
        
        # Filter out empty groups
        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) < 2:
            return {"error": "Need at least 2 non-empty groups"}
        
        statistic, p_value = scipy_stats.f_oneway(*valid_groups)
        
        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "num_groups": len(valid_groups)
        }
    
    def multiple_comparison_correction(
        self, 
        p_values: List[float], 
        method: str = "bonferroni"
    ) -> List[float]:
        """Apply multiple comparison correction."""
        if method == "bonferroni":
            return [min(1.0, p * len(p_values)) for p in p_values]
        elif method == "holm":
            # Holm-Bonferroni method (step-down)
            sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
            corrected = [0.0] * len(p_values)
            
            for rank, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) - rank
                corrected[idx] = min(1.0, p_values[idx] * correction_factor)
                
                # Step-down: if this test is not significant, 
                # all subsequent tests are also not significant
                if corrected[idx] >= self.alpha:
                    for remaining_idx in sorted_indices[rank+1:]:
                        corrected[remaining_idx] = 1.0
                    break
            
            return corrected
        else:
            return p_values  # No correction


class ExperimentRunner:
    """Orchestrates and executes experimental research."""
    
    def __init__(self, output_dir: str = "experimental_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.results_db = []
        self.experiment_metadata = {}
        self.statistical_analyzer = StatisticalAnalyzer()
        
        logger.info(f"Experiment runner initialized, output: {self.output_dir}")
    
    def create_experiment_configuration(
        self,
        name: str,
        experiment_type: ExperimentType,
        **kwargs
    ) -> ExperimentConfiguration:
        """Create experiment configuration with defaults."""
        
        # Generate unique experiment ID
        experiment_id = hashlib.md5(
            (name + str(time.time())).encode()
        ).hexdigest()[:12]
        
        # Default configuration
        defaults = {
            "experiment_id": experiment_id,
            "experiment_type": experiment_type,
            "name": name,
            "description": f"Automated {experiment_type.value} experiment",
            "model_architecture": ModelArchitecture.BERT_BASE,
            "dataset": Dataset.GLUE_SST2,
            "sequence_length": 512,
            "batch_size": 16,
            "privacy_mechanisms": ["gaussian", "discrete_gaussian", "quantum_resistant"],
            "epsilon_values": [0.1, 0.5, 1.0, 2.0, 5.0],
            "delta_values": [1e-5],
            "sensitivity": 1.0,
            "num_epochs": 3,
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "num_trials": 5,
            "random_seeds": [42, 123, 456, 789, 101112],
            "statistical_power": 0.8,
            "confidence_level": 0.95
        }
        
        # Update with provided kwargs
        defaults.update(kwargs)
        
        return ExperimentConfiguration(**defaults)
    
    def run_experiment(self, config: ExperimentConfiguration) -> List[ExperimentResult]:
        """Run a complete experiment with multiple trials."""
        
        logger.info(f"Starting experiment: {config.name} ({config.experiment_id})")
        
        # Store experiment metadata
        self.experiment_metadata[config.experiment_id] = {
            "config": asdict(config),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "running"
        }
        
        all_results = []
        
        try:
            # Run trials for each privacy mechanism and epsilon value
            for mechanism in config.privacy_mechanisms:
                for epsilon in config.epsilon_values:
                    for delta in config.delta_values:
                        for trial_idx in range(config.num_trials):
                            
                            # Create trial configuration
                            trial_config = {
                                "privacy_mechanism": mechanism,
                                "epsilon": epsilon,
                                "delta": delta,
                                "trial_index": trial_idx,
                                "random_seed": config.random_seeds[trial_idx % len(config.random_seeds)]
                            }
                            
                            # Run single trial
                            trial_result = self._run_single_trial(config, trial_config)
                            all_results.append(trial_result)
                            
                            # Save intermediate results if requested
                            if config.save_intermediate_results:
                                self._save_intermediate_result(trial_result)
            
            # Update experiment status
            self.experiment_metadata[config.experiment_id]["status"] = "completed"
            self.experiment_metadata[config.experiment_id]["end_time"] = datetime.now(timezone.utc).isoformat()
            self.experiment_metadata[config.experiment_id]["total_trials"] = len(all_results)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.experiment_metadata[config.experiment_id]["status"] = "failed"
            self.experiment_metadata[config.experiment_id]["error"] = str(e)
            raise
        
        # Store results
        self.results_db.extend(all_results)
        
        # Generate analysis and reports
        self._generate_experiment_analysis(config, all_results)
        
        logger.info(f"Experiment completed: {config.name}, {len(all_results)} trials")
        return all_results
    
    def _run_single_trial(
        self, 
        config: ExperimentConfiguration, 
        trial_config: Dict[str, Any]
    ) -> ExperimentResult:
        """Run a single experimental trial."""
        
        trial_id = f"{config.experiment_id}_trial_{trial_config['trial_index']}"
        
        logger.debug(f"Running trial: {trial_id}")
        
        start_time = time.time()
        
        # Create result object
        result = ExperimentResult(
            trial_id=trial_id,
            experiment_id=config.experiment_id,
            configuration=trial_config,
            random_seed=trial_config["random_seed"]
        )
        
        try:
            # Simulate model training and evaluation
            # In a real implementation, this would run actual models
            result = self._simulate_trial_execution(config, trial_config, result)
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            result.accuracy = 0.0
            result.training_time_seconds = time.time() - start_time
        
        result.training_time_seconds = time.time() - start_time
        
        return result
    
    def _simulate_trial_execution(
        self,
        config: ExperimentConfiguration,
        trial_config: Dict[str, Any],
        result: ExperimentResult
    ) -> ExperimentResult:
        """Simulate trial execution with realistic results."""
        
        # Set random seed for reproducibility
        np.random.seed(trial_config["random_seed"])
        
        # Simulate privacy mechanism effects
        mechanism = trial_config["privacy_mechanism"]
        epsilon = trial_config["epsilon"]
        
        # Base performance (varies by model and dataset)
        base_accuracy = self._get_base_accuracy(config.model_architecture, config.dataset)
        
        # Privacy degradation (higher privacy = lower utility)
        privacy_penalty = self._compute_privacy_penalty(mechanism, epsilon)
        simulated_accuracy = max(0.1, base_accuracy * (1 - privacy_penalty))
        
        # Add realistic noise
        accuracy_noise = np.random.normal(0, 0.02)  # ±2% noise
        result.accuracy = max(0.0, min(1.0, simulated_accuracy + accuracy_noise))
        
        # Simulate other metrics
        result.f1_score = result.accuracy * 0.95 + np.random.normal(0, 0.01)
        result.epsilon_spent = epsilon * (0.9 + np.random.uniform(0, 0.2))
        result.delta_spent = trial_config["delta"]
        
        # Computational metrics
        base_time = self._get_base_training_time(config.model_architecture)
        privacy_overhead = self._compute_privacy_overhead(mechanism)
        result.training_time_seconds = base_time * (1 + privacy_overhead) * (0.8 + np.random.uniform(0, 0.4))
        
        result.inference_time_ms = np.random.uniform(50, 200)
        result.memory_usage_mb = np.random.uniform(2000, 8000)
        result.throughput_samples_per_sec = np.random.uniform(100, 500)
        
        # Privacy-specific metrics
        result.gradient_norm = np.random.uniform(0.5, 2.0)
        result.noise_scale = self._compute_noise_scale(epsilon, trial_config["delta"])
        result.convergence_epoch = np.random.randint(1, config.num_epochs + 1)
        
        return result
    
    def _get_base_accuracy(self, model: ModelArchitecture, dataset: Dataset) -> float:
        """Get baseline accuracy for model-dataset combination."""
        # Simplified lookup table
        baseline_map = {
            (ModelArchitecture.BERT_BASE, Dataset.GLUE_SST2): 0.92,
            (ModelArchitecture.BERT_BASE, Dataset.GLUE_COLA): 0.85,
            (ModelArchitecture.BERT_LARGE, Dataset.GLUE_SST2): 0.94,
            (ModelArchitecture.GPT2_SMALL, Dataset.WIKITEXT_103): 0.87,
            (ModelArchitecture.VIT_BASE, Dataset.CIFAR_10): 0.89,
        }
        return baseline_map.get((model, dataset), 0.80)  # Default
    
    def _compute_privacy_penalty(self, mechanism: str, epsilon: float) -> float:
        """Compute utility penalty from privacy mechanism."""
        # Higher privacy (lower epsilon) = higher penalty
        base_penalty = min(0.3, 1.0 / (epsilon + 0.1))
        
        # Mechanism-specific factors
        mechanism_factors = {
            "gaussian": 1.0,
            "discrete_gaussian": 0.95,
            "quantum_resistant": 1.1,
            "federated": 0.9,
            "adaptive": 0.85
        }
        
        return base_penalty * mechanism_factors.get(mechanism, 1.0)
    
    def _get_base_training_time(self, model: ModelArchitecture) -> float:
        """Get baseline training time in seconds."""
        time_map = {
            ModelArchitecture.BERT_BASE: 1800,  # 30 minutes
            ModelArchitecture.BERT_LARGE: 3600,  # 1 hour
            ModelArchitecture.GPT2_SMALL: 2400,  # 40 minutes
            ModelArchitecture.VIT_BASE: 1200,   # 20 minutes
        }
        return time_map.get(model, 1800)
    
    def _compute_privacy_overhead(self, mechanism: str) -> float:
        """Compute computational overhead from privacy mechanism."""
        overhead_map = {
            "gaussian": 0.1,
            "discrete_gaussian": 0.15,
            "quantum_resistant": 0.25,
            "federated": 0.3,
            "adaptive": 0.2
        }
        return overhead_map.get(mechanism, 0.1)
    
    def _compute_noise_scale(self, epsilon: float, delta: float) -> float:
        """Compute noise scale for privacy mechanism."""
        return np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    def _save_intermediate_result(self, result: ExperimentResult):
        """Save intermediate trial result."""
        trial_file = self.output_dir / f"trial_{result.trial_id}.json"
        
        with open(trial_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    def _generate_experiment_analysis(
        self, 
        config: ExperimentConfiguration, 
        results: List[ExperimentResult]
    ):
        """Generate comprehensive analysis of experiment results."""
        
        logger.info(f"Generating analysis for experiment: {config.experiment_id}")
        
        # Group results by privacy mechanism and epsilon
        grouped_results = {}
        for result in results:
            key = (result.configuration["privacy_mechanism"], result.configuration["epsilon"])
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Statistical analysis
        analysis = {
            "experiment_id": config.experiment_id,
            "experiment_name": config.name,
            "total_trials": len(results),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "grouped_analysis": {},
            "statistical_comparisons": {},
            "summary_statistics": {}
        }
        
        # Analyze each group
        for (mechanism, epsilon), group_results in grouped_results.items():
            group_key = f"{mechanism}_eps_{epsilon}"
            
            # Extract metrics
            accuracies = [r.accuracy for r in group_results if r.accuracy is not None]
            training_times = [r.training_time_seconds for r in group_results if r.training_time_seconds is not None]
            epsilon_spent = [r.epsilon_spent for r in group_results if r.epsilon_spent is not None]
            
            group_analysis = {
                "mechanism": mechanism,
                "epsilon": epsilon,
                "num_trials": len(group_results),
                "accuracy_stats": self.statistical_analyzer.descriptive_statistics(accuracies),
                "training_time_stats": self.statistical_analyzer.descriptive_statistics(training_times),
                "epsilon_spent_stats": self.statistical_analyzer.descriptive_statistics(epsilon_spent),
                "accuracy_ci": self.statistical_analyzer.confidence_interval(accuracies),
                "training_time_ci": self.statistical_analyzer.confidence_interval(training_times)
            }
            
            analysis["grouped_analysis"][group_key] = group_analysis
        
        # Pairwise statistical comparisons
        mechanisms = list(set(r.configuration["privacy_mechanism"] for r in results))
        
        for i, mech1 in enumerate(mechanisms):
            for mech2 in mechanisms[i+1:]:
                # Compare mechanisms at same epsilon values
                for epsilon in config.epsilon_values:
                    group1 = [r for r in results if r.configuration["privacy_mechanism"] == mech1 
                             and r.configuration["epsilon"] == epsilon]
                    group2 = [r for r in results if r.configuration["privacy_mechanism"] == mech2 
                             and r.configuration["epsilon"] == epsilon]
                    
                    if len(group1) >= 2 and len(group2) >= 2:
                        acc1 = [r.accuracy for r in group1 if r.accuracy is not None]
                        acc2 = [r.accuracy for r in group2 if r.accuracy is not None]
                        
                        comparison_key = f"{mech1}_vs_{mech2}_eps_{epsilon}"
                        analysis["statistical_comparisons"][comparison_key] = self.statistical_analyzer.two_sample_ttest(acc1, acc2)
        
        # Overall summary statistics
        all_accuracies = [r.accuracy for r in results if r.accuracy is not None]
        all_training_times = [r.training_time_seconds for r in results if r.training_time_seconds is not None]
        
        analysis["summary_statistics"] = {
            "overall_accuracy": self.statistical_analyzer.descriptive_statistics(all_accuracies),
            "overall_training_time": self.statistical_analyzer.descriptive_statistics(all_training_times),
            "accuracy_range": (min(all_accuracies), max(all_accuracies)) if all_accuracies else (0, 0),
            "best_mechanism": self._find_best_mechanism(grouped_results),
            "efficiency_analysis": self._analyze_efficiency(grouped_results)
        }
        
        # Save analysis
        analysis_file = self.output_dir / f"analysis_{config.experiment_id}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate plots if requested
        if config.generate_plots and PLOTTING_AVAILABLE:
            self._generate_plots(config, results, grouped_results)
        
        # Export raw data if requested
        if config.export_data:
            self._export_data(config, results)
        
        logger.info(f"Analysis saved to: {analysis_file}")
        return analysis
    
    def _find_best_mechanism(self, grouped_results: Dict) -> Dict[str, Any]:
        """Find the best performing mechanism across conditions."""
        best_accuracy = 0
        best_mechanism = None
        best_epsilon = None
        
        for (mechanism, epsilon), group_results in grouped_results.items():
            accuracies = [r.accuracy for r in group_results if r.accuracy is not None]
            if accuracies:
                mean_accuracy = statistics.mean(accuracies)
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_mechanism = mechanism
                    best_epsilon = epsilon
        
        return {
            "mechanism": best_mechanism,
            "epsilon": best_epsilon,
            "accuracy": best_accuracy
        }
    
    def _analyze_efficiency(self, grouped_results: Dict) -> Dict[str, Any]:
        """Analyze efficiency (accuracy per unit privacy cost)."""
        efficiency_scores = {}
        
        for (mechanism, epsilon), group_results in grouped_results.items():
            accuracies = [r.accuracy for r in group_results if r.accuracy is not None]
            if accuracies:
                mean_accuracy = statistics.mean(accuracies)
                # Efficiency = accuracy / privacy_cost (higher epsilon = lower privacy)
                efficiency = mean_accuracy / epsilon if epsilon > 0 else 0
                efficiency_scores[f"{mechanism}_eps_{epsilon}"] = efficiency
        
        if efficiency_scores:
            best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
            return {
                "efficiency_scores": efficiency_scores,
                "most_efficient": best_efficiency[0],
                "efficiency_score": best_efficiency[1]
            }
        
        return {"efficiency_scores": {}}
    
    def _generate_plots(
        self, 
        config: ExperimentConfiguration, 
        results: List[ExperimentResult],
        grouped_results: Dict
    ):
        """Generate visualization plots for experiment results."""
        
        plot_dir = self.output_dir / f"plots_{config.experiment_id}"
        plot_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        
        # Plot 1: Privacy-Utility Tradeoff
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mechanism in config.privacy_mechanisms:
            epsilons = []
            accuracies = []
            
            for epsilon in config.epsilon_values:
                group = grouped_results.get((mechanism, epsilon), [])
                if group:
                    acc_values = [r.accuracy for r in group if r.accuracy is not None]
                    if acc_values:
                        epsilons.append(epsilon)
                        accuracies.append(statistics.mean(acc_values))
            
            if epsilons and accuracies:
                ax.plot(epsilons, accuracies, marker='o', label=mechanism)
        
        ax.set_xlabel('Privacy Budget (ε)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Privacy-Utility Tradeoff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / "privacy_utility_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Training Time vs Privacy
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mechanism in config.privacy_mechanisms:
            epsilons = []
            training_times = []
            
            for epsilon in config.epsilon_values:
                group = grouped_results.get((mechanism, epsilon), [])
                if group:
                    time_values = [r.training_time_seconds for r in group if r.training_time_seconds is not None]
                    if time_values:
                        epsilons.append(epsilon)
                        training_times.append(statistics.mean(time_values))
            
            if epsilons and training_times:
                ax.plot(epsilons, training_times, marker='s', label=mechanism)
        
        ax.set_xlabel('Privacy Budget (ε)')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Computational Overhead vs Privacy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / "computational_overhead.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to: {plot_dir}")
    
    def _export_data(self, config: ExperimentConfiguration, results: List[ExperimentResult]):
        """Export raw experimental data."""
        
        # Export as JSON
        data_file = self.output_dir / f"raw_data_{config.experiment_id}.json"
        export_data = {
            "configuration": asdict(config),
            "results": [asdict(r) for r in results],
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        with open(data_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Export as CSV for analysis in external tools
        csv_file = self.output_dir / f"results_{config.experiment_id}.csv"
        
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            if results:
                fieldnames = ['trial_id', 'experiment_id', 'privacy_mechanism', 'epsilon', 'delta', 
                             'accuracy', 'f1_score', 'training_time_seconds', 'inference_time_ms',
                             'memory_usage_mb', 'epsilon_spent', 'random_seed']
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'trial_id': result.trial_id,
                        'experiment_id': result.experiment_id,
                        'privacy_mechanism': result.configuration.get('privacy_mechanism', ''),
                        'epsilon': result.configuration.get('epsilon', ''),
                        'delta': result.configuration.get('delta', ''),
                        'accuracy': result.accuracy,
                        'f1_score': result.f1_score,
                        'training_time_seconds': result.training_time_seconds,
                        'inference_time_ms': result.inference_time_ms,
                        'memory_usage_mb': result.memory_usage_mb,
                        'epsilon_spent': result.epsilon_spent,
                        'random_seed': result.random_seed
                    }
                    writer.writerow(row)
        
        logger.info(f"Data exported to: {data_file} and {csv_file}")


def main():
    """Example usage of the experimental framework."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize experiment runner
    runner = ExperimentRunner(output_dir="dp_flash_attention_experiments")
    
    # Create experiment configuration
    config = runner.create_experiment_configuration(
        name="Privacy Mechanism Comparison Study",
        experiment_type=ExperimentType.PRIVACY_MECHANISM_COMPARISON,
        model_architecture=ModelArchitecture.BERT_BASE,
        dataset=Dataset.GLUE_SST2,
        privacy_mechanisms=["gaussian", "discrete_gaussian", "quantum_resistant"],
        epsilon_values=[0.1, 0.5, 1.0, 2.0],
        delta_values=[1e-5],
        num_trials=10,
        num_epochs=3
    )
    
    # Run experiment
    results = runner.run_experiment(config)
    
    print(f"Experiment completed with {len(results)} trials")
    print(f"Results saved to: {runner.output_dir}")
    
    # Example of accessing results
    best_result = max(results, key=lambda r: r.accuracy or 0)
    print(f"Best result: {best_result.accuracy:.3f} accuracy with {best_result.configuration['privacy_mechanism']}")


if __name__ == "__main__":
    main()