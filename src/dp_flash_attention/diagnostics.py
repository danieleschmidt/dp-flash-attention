"""
System diagnostics and health checks for DP-Flash-Attention.

Provides comprehensive system validation, health monitoring, and diagnostic
capabilities for differential privacy operations.
"""

import time
import psutil
import warnings
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

import torch
import numpy as np

from .validation import (
    validate_system_requirements_comprehensive,
    validate_privacy_parameters_comprehensive
)
from .security import validate_secure_environment, SecureRandomGenerator
from .utils import cuda_version, privacy_check, benchmark_attention_kernel


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical" 
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    remediation: Optional[str] = None


class SystemDiagnostics:
    """
    Comprehensive system diagnostics for DP-Flash-Attention.
    
    Performs health checks, system validation, and performance diagnostics.
    """
    
    def __init__(self, include_performance_tests: bool = True):
        """
        Initialize system diagnostics.
        
        Args:
            include_performance_tests: Whether to include performance benchmarks
        """
        self.include_performance_tests = include_performance_tests
        self.last_check_time = None
        self.check_history: List[Dict[str, Any]] = []
    
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """
        Run complete system diagnostics.
        
        Returns:
            Comprehensive diagnostic report
        """
        start_time = time.time()
        
        diagnostic_report = {
            'timestamp': start_time,
            'version': '0.1.0',
            'diagnostics': {
                'system_requirements': self.check_system_requirements(),
                'security_environment': self.check_security_environment(),
                'hardware_resources': self.check_hardware_resources(),
                'privacy_capabilities': self.check_privacy_capabilities(),
                'performance_baseline': self.check_performance_baseline() if self.include_performance_tests else None,
            },
            'overall_status': HealthStatus.UNKNOWN.value,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'execution_time_ms': 0
        }
        
        # Analyze results and determine overall status
        all_checks = []
        for category, checks in diagnostic_report['diagnostics'].items():
            if checks is not None:
                if isinstance(checks, list):
                    all_checks.extend(checks)
                elif isinstance(checks, dict) and 'checks' in checks:
                    all_checks.extend(checks['checks'])
        
        # Determine overall status
        critical_count = sum(1 for check in all_checks if check.get('status') == HealthStatus.CRITICAL.value)
        warning_count = sum(1 for check in all_checks if check.get('status') == HealthStatus.WARNING.value)
        
        if critical_count > 0:
            diagnostic_report['overall_status'] = HealthStatus.CRITICAL.value
        elif warning_count > 0:
            diagnostic_report['overall_status'] = HealthStatus.WARNING.value
        else:
            diagnostic_report['overall_status'] = HealthStatus.HEALTHY.value
        
        # Collect issues and recommendations
        for check in all_checks:
            if check.get('status') == HealthStatus.CRITICAL.value:
                diagnostic_report['critical_issues'].append(check.get('message', ''))
            elif check.get('status') == HealthStatus.WARNING.value:
                diagnostic_report['warnings'].append(check.get('message', ''))
            
            if check.get('remediation'):
                diagnostic_report['recommendations'].append(check['remediation'])
        
        execution_time = (time.time() - start_time) * 1000
        diagnostic_report['execution_time_ms'] = execution_time
        
        # Store in history
        self.last_check_time = start_time
        self.check_history.append(diagnostic_report)
        
        # Keep only recent history
        if len(self.check_history) > 50:  
            self.check_history = self.check_history[-50:]
        
        return diagnostic_report
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements and dependencies."""
        checks = []
        
        try:
            requirements = validate_system_requirements_comprehensive()
            
            # Python version check
            if requirements.get('python_ok', False):
                checks.append(self._create_check(
                    "python_version", 
                    HealthStatus.HEALTHY,
                    f"Python {requirements['python_version']} supported",
                    {'version': requirements['python_version']}
                ))
            else:
                checks.append(self._create_check(
                    "python_version",
                    HealthStatus.CRITICAL,
                    f"Python {requirements.get('python_version', 'unknown')} not supported",
                    {'version': requirements.get('python_version', 'unknown')},
                    "Upgrade to Python >= 3.10"
                ))
            
            # PyTorch version check
            if requirements.get('torch_ok', False):
                checks.append(self._create_check(
                    "pytorch_version",
                    HealthStatus.HEALTHY,
                    f"PyTorch {requirements['torch_version']} supported",
                    {'version': requirements['torch_version']}
                ))
            else:
                checks.append(self._create_check(
                    "pytorch_version",
                    HealthStatus.CRITICAL,
                    f"PyTorch {requirements.get('torch_version', 'unknown')} not supported",
                    {'version': requirements.get('torch_version', 'unknown')},
                    "Upgrade to PyTorch >= 2.3.0"
                ))
            
            # CUDA availability
            if requirements.get('cuda_available', False):
                compute_ok = requirements.get('compute_ok')
                if compute_ok is True:
                    status = HealthStatus.HEALTHY
                    message = f"CUDA available with good compute capability"
                elif compute_ok is False:
                    status = HealthStatus.WARNING
                    message = f"CUDA available but compute capability may be limited"
                else:
                    status = HealthStatus.WARNING
                    message = f"CUDA available, compute capability unknown"
                
                checks.append(self._create_check(
                    "cuda_availability",
                    status,
                    message,
                    {
                        'cuda_version': requirements.get('cuda_version', 'unknown'),
                        'compute_capability': requirements.get('compute_capability', 'unknown'),
                        'device_count': requirements.get('cuda_devices', 0)
                    }
                ))
            else:
                checks.append(self._create_check(
                    "cuda_availability",
                    HealthStatus.WARNING,
                    "CUDA not available - performance will be limited",
                    {},
                    "Install CUDA-compatible PyTorch for optimal performance"
                ))
            
            # Required packages
            required_packages = ['triton', 'einops', 'numpy', 'ninja']
            for pkg in required_packages:
                if requirements.get(f'{pkg}_available', False):
                    checks.append(self._create_check(
                        f"package_{pkg}",
                        HealthStatus.HEALTHY,
                        f"Package {pkg} available",
                        {'package': pkg}
                    ))
                else:
                    checks.append(self._create_check(
                        f"package_{pkg}",
                        HealthStatus.CRITICAL,
                        f"Required package {pkg} not available",
                        {'package': pkg},
                        f"Install {pkg}: pip install {pkg}"
                    ))
            
        except Exception as e:
            checks.append(self._create_check(
                "system_requirements_check",
                HealthStatus.CRITICAL,
                f"Failed to check system requirements: {e}",
                {'error': str(e)},
                "Investigate system configuration"
            ))
        
        return {
            'category': 'system_requirements',
            'checks': checks,
            'summary': f"{len([c for c in checks if c['status'] == HealthStatus.HEALTHY.value])} of {len(checks)} checks passed"
        }
    
    def check_security_environment(self) -> Dict[str, Any]:
        """Check security environment and cryptographic capabilities."""
        checks = []
        
        try:
            security_validation = validate_secure_environment()
            
            # Overall security status
            if security_validation.get('secure', False):
                checks.append(self._create_check(
                    "security_environment",
                    HealthStatus.HEALTHY,
                    "Security environment validated",
                    security_validation
                ))
            else:
                checks.append(self._create_check(
                    "security_environment",
                    HealthStatus.WARNING,
                    "Security environment has issues",
                    security_validation,
                    "Review security warnings and recommendations"
                ))
            
            # Entropy sources
            entropy_count = len(security_validation.get('entropy_sources', []))
            if entropy_count >= 2:
                status = HealthStatus.HEALTHY
                message = f"Multiple entropy sources available ({entropy_count})"
            elif entropy_count == 1:
                status = HealthStatus.WARNING
                message = f"Single entropy source available"
            else:
                status = HealthStatus.CRITICAL
                message = f"No secure entropy sources available"
            
            checks.append(self._create_check(
                "entropy_sources",
                status,
                message,
                {'sources': security_validation.get('entropy_sources', [])},
                "Ensure system has access to /dev/urandom and Python secrets module" if entropy_count < 2 else None
            ))
            
            # Cryptography library
            crypto_available = security_validation.get('crypto_libraries', {}).get('cryptography', False)
            if crypto_available:
                checks.append(self._create_check(
                    "cryptography_library",
                    HealthStatus.HEALTHY,
                    "Cryptography library available",
                    {'available': True}
                ))
            else:
                checks.append(self._create_check(
                    "cryptography_library",
                    HealthStatus.WARNING,
                    "Cryptography library not available - using fallback security",
                    {'available': False},
                    "Install cryptography library: pip install cryptography"
                ))
            
            # Test secure random generation
            try:
                rng = SecureRandomGenerator()
                test_noise = rng.generate_gaussian_noise((100,), 1.0)
                noise_std = torch.std(test_noise).item()
                
                if 0.8 < noise_std < 1.2:  # Within reasonable range for std=1.0
                    checks.append(self._create_check(
                        "secure_rng_test",
                        HealthStatus.HEALTHY,
                        f"Secure RNG functioning correctly (std={noise_std:.3f})",
                        {'test_std': noise_std}
                    ))
                else:
                    checks.append(self._create_check(
                        "secure_rng_test",
                        HealthStatus.WARNING,
                        f"Secure RNG may have issues (std={noise_std:.3f})",
                        {'test_std': noise_std}
                    ))
            except Exception as e:
                checks.append(self._create_check(
                    "secure_rng_test",
                    HealthStatus.CRITICAL,
                    f"Secure RNG test failed: {e}",
                    {'error': str(e)}
                ))
            
        except Exception as e:
            checks.append(self._create_check(
                "security_check",
                HealthStatus.CRITICAL,
                f"Security environment check failed: {e}",
                {'error': str(e)}
            ))
        
        return {
            'category': 'security_environment',
            'checks': checks,
            'summary': f"Security validation completed with {len([c for c in checks if c['status'] == HealthStatus.CRITICAL.value])} critical issues"
        }
    
    def check_hardware_resources(self) -> Dict[str, Any]:
        """Check hardware resources and availability."""
        checks = []
        
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            checks.append(self._create_check(
                "cpu_resources",
                HealthStatus.HEALTHY if cpu_percent < 80 else HealthStatus.WARNING,
                f"CPU: {cpu_count} cores, {cpu_percent:.1f}% usage",
                {
                    'cpu_count': cpu_count,
                    'cpu_percent': cpu_percent,
                    'cpu_freq_mhz': cpu_freq.current if cpu_freq else None
                }
            ))
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_percent = memory.percent
            
            if memory_gb < 8:
                status = HealthStatus.WARNING
                message = f"Low system memory: {memory_gb:.1f}GB"
                remediation = "Consider upgrading system memory for large models"
            elif memory_percent > 90:
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_percent:.1f}% of {memory_gb:.1f}GB"
                remediation = "Close unnecessary applications"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory: {memory_gb:.1f}GB, {memory_percent:.1f}% used"
                remediation = None
            
            checks.append(self._create_check(
                "system_memory",
                status,
                message,
                {
                    'total_gb': memory_gb,
                    'percent_used': memory_percent,
                    'available_gb': memory.available / (1024**3)
                },
                remediation
            ))
            
            # GPU information
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
                        memory_total = props.total_memory / (1024**3)
                        memory_percent = (memory_allocated / memory_total) * 100
                        
                        if memory_percent > 90:
                            status = HealthStatus.WARNING
                            message = f"GPU {i} memory high: {memory_percent:.1f}%"
                            remediation = "Free GPU memory or use smaller batch sizes"
                        elif memory_total < 8:
                            status = HealthStatus.WARNING  
                            message = f"GPU {i} low memory: {memory_total:.1f}GB"
                            remediation = "Consider using GPU with more memory"
                        else:
                            status = HealthStatus.HEALTHY
                            message = f"GPU {i}: {props.name}, {memory_total:.1f}GB"
                            remediation = None
                        
                        checks.append(self._create_check(
                            f"gpu_{i}_resources",
                            status,
                            message,
                            {
                                'device_id': i,
                                'name': props.name,
                                'memory_total_gb': memory_total,
                                'memory_allocated_gb': memory_allocated,
                                'memory_percent': memory_percent,
                                'compute_capability': f"{props.major}.{props.minor}"
                            },
                            remediation
                        ))
                        
                    except Exception as e:
                        checks.append(self._create_check(
                            f"gpu_{i}_check",
                            HealthStatus.WARNING,
                            f"GPU {i} check failed: {e}",
                            {'device_id': i, 'error': str(e)}
                        ))
            else:
                checks.append(self._create_check(
                    "gpu_resources",
                    HealthStatus.WARNING,
                    "No CUDA GPUs available",
                    {},
                    "Install CUDA-compatible GPU for optimal performance"
                ))
            
        except Exception as e:
            checks.append(self._create_check(
                "hardware_check",
                HealthStatus.CRITICAL,
                f"Hardware resource check failed: {e}",
                {'error': str(e)}
            ))
        
        return {
            'category': 'hardware_resources',
            'checks': checks,
            'summary': f"Hardware resources checked: {len(checks)} components"
        }
    
    def check_privacy_capabilities(self) -> Dict[str, Any]:
        """Check differential privacy capabilities."""
        checks = []
        
        try:
            # Test privacy parameter validation
            test_cases = [
                (1.0, 1e-5, True, "Standard parameters"),
                (0.1, 1e-5, True, "Strong privacy parameters"),
                (10.0, 1e-3, False, "Weak privacy parameters"),
            ]
            
            for epsilon, delta, should_pass, description in test_cases:
                try:
                    validation = validate_privacy_parameters_comprehensive(epsilon, delta, 1.0)
                    
                    if should_pass:
                        checks.append(self._create_check(
                            f"privacy_validation_{epsilon}_{delta}",
                            HealthStatus.HEALTHY,
                            f"Privacy validation passed: {description}",
                            validation
                        ))
                    else:
                        # Check if warnings were generated as expected
                        warning_count = len(validation.get('recommendations', []))
                        if warning_count > 0:
                            checks.append(self._create_check(
                                f"privacy_validation_{epsilon}_{delta}",
                                HealthStatus.HEALTHY,
                                f"Privacy validation correctly identified weak parameters: {description}",
                                validation
                            ))
                        else:
                            checks.append(self._create_check(
                                f"privacy_validation_{epsilon}_{delta}",
                                HealthStatus.WARNING,
                                f"Privacy validation may be too permissive: {description}",
                                validation
                            ))
                
                except Exception as e:
                    checks.append(self._create_check(
                        f"privacy_validation_{epsilon}_{delta}",
                        HealthStatus.CRITICAL,
                        f"Privacy validation failed for {description}: {e}",
                        {'error': str(e)}
                    ))
            
            # Test privacy accounting
            try:
                from .privacy import RenyiAccountant
                
                accountant = RenyiAccountant()
                accountant.add_step(1.0, 1e-5, 32, 1000)  # Test step
                epsilon_spent = accountant.get_epsilon(1e-5)
                
                if 0.5 < epsilon_spent < 2.0:  # Reasonable range
                    checks.append(self._create_check(
                        "privacy_accounting",
                        HealthStatus.HEALTHY,
                        f"Privacy accounting functional (epsilon={epsilon_spent:.3f})",
                        {'epsilon_spent': epsilon_spent}
                    ))
                else:
                    checks.append(self._create_check(
                        "privacy_accounting",
                        HealthStatus.WARNING,
                        f"Privacy accounting may have issues (epsilon={epsilon_spent:.3f})",
                        {'epsilon_spent': epsilon_spent}
                    ))
                    
            except Exception as e:
                checks.append(self._create_check(
                    "privacy_accounting",
                    HealthStatus.CRITICAL,
                    f"Privacy accounting test failed: {e}",
                    {'error': str(e)}
                ))
            
        except Exception as e:
            checks.append(self._create_check(
                "privacy_capabilities",
                HealthStatus.CRITICAL,
                f"Privacy capabilities check failed: {e}",
                {'error': str(e)}
            ))
        
        return {
            'category': 'privacy_capabilities',
            'checks': checks,
            'summary': f"Privacy capabilities tested: {len(checks)} tests"
        }
    
    def check_performance_baseline(self) -> Dict[str, Any]:
        """Check performance baseline and benchmarks."""
        checks = []
        
        try:
            # Run basic performance benchmark
            benchmark_result = benchmark_attention_kernel(
                batch_size=16,
                sequence_length=512,
                num_heads=8,
                head_dim=64,
                num_iterations=10,
                warmup_iterations=2
            )
            
            if 'error' in benchmark_result:
                checks.append(self._create_check(
                    "performance_benchmark",
                    HealthStatus.WARNING,
                    f"Performance benchmark failed: {benchmark_result['error']}",
                    benchmark_result
                ))
            else:
                avg_time = benchmark_result.get('avg_time_ms', 0)
                throughput = benchmark_result.get('throughput_samples_per_sec', 0)
                
                # Evaluate performance
                if avg_time < 10:  # Very fast
                    perf_status = HealthStatus.HEALTHY
                    perf_message = f"Excellent performance: {avg_time:.2f}ms avg"
                elif avg_time < 50:  # Acceptable
                    perf_status = HealthStatus.HEALTHY
                    perf_message = f"Good performance: {avg_time:.2f}ms avg"
                elif avg_time < 200:  # Slow but usable
                    perf_status = HealthStatus.WARNING
                    perf_message = f"Moderate performance: {avg_time:.2f}ms avg"
                else:  # Very slow
                    perf_status = HealthStatus.WARNING
                    perf_message = f"Slow performance: {avg_time:.2f}ms avg"
                
                checks.append(self._create_check(
                    "performance_benchmark",
                    perf_status,
                    perf_message,
                    benchmark_result,
                    "Consider GPU acceleration or smaller batch sizes" if perf_status == HealthStatus.WARNING else None
                ))
            
        except Exception as e:
            checks.append(self._create_check(
                "performance_baseline",
                HealthStatus.WARNING,
                f"Performance baseline check failed: {e}",
                {'error': str(e)}
            ))
        
        return {
            'category': 'performance_baseline',
            'checks': checks,
            'summary': f"Performance baseline completed"
        }
    
    def _create_check(
        self,
        name: str,
        status: HealthStatus,
        message: str,
        details: Dict[str, Any],
        remediation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a health check result dictionary."""
        return {
            'name': name,
            'status': status.value,
            'message': message,
            'details': details,
            'timestamp': time.time(),
            'remediation': remediation
        }
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get summary of recent diagnostic results."""
        if not self.check_history:
            return {'message': 'No diagnostic history available'}
        
        latest = self.check_history[-1]
        
        return {
            'last_check_time': latest['timestamp'],
            'overall_status': latest['overall_status'],
            'critical_issues_count': len(latest['critical_issues']),
            'warnings_count': len(latest['warnings']),
            'recommendations_count': len(latest['recommendations']),
            'execution_time_ms': latest['execution_time_ms'],
            'check_categories': list(latest['diagnostics'].keys())
        }


def run_quick_health_check() -> Dict[str, Any]:
    """Run a quick health check for immediate status."""
    diagnostics = SystemDiagnostics(include_performance_tests=False)
    return diagnostics.run_full_diagnostics()


def run_comprehensive_diagnostics() -> Dict[str, Any]:
    """Run comprehensive diagnostics including performance tests."""
    diagnostics = SystemDiagnostics(include_performance_tests=True)
    return diagnostics.run_full_diagnostics()


def export_diagnostic_report(report: Dict[str, Any], filepath: str) -> None:
    """Export diagnostic report to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)