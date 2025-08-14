#!/usr/bin/env python3
"""
Autonomous SDLC Quality Gates for DP-Flash-Attention.

This module implements comprehensive quality validation for the autonomous SDLC
implementation, checking code quality, architecture completeness, and deployment readiness.
"""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class QualityMetric:
    """Quality metric with score and details."""
    name: str
    score: float  # 0.0 to 1.0
    weight: float  # Importance weight
    details: str
    status: str  # "pass", "fail", "warning"
    recommendations: List[str]


@dataclass
class QualityGateResult:
    """Overall quality gate result."""
    gate_name: str
    overall_score: float
    weighted_score: float
    metrics: List[QualityMetric]
    passed: bool
    timestamp: float


class AutonomousSDLCQualityGates:
    """
    Comprehensive quality gates for autonomous SDLC implementation.
    
    Validates:
    - Code architecture and structure
    - Implementation completeness
    - Documentation quality
    - Deployment readiness
    - Research capabilities
    - Global scaling features
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.results: Dict[str, QualityGateResult] = {}
        
        # Quality gate thresholds
        self.pass_threshold = 0.80  # 80% overall score to pass
        self.warning_threshold = 0.60  # Below 60% is a warning
        
        # Expected file patterns and structures
        self.expected_structure = {
            "core_modules": [
                "src/dp_flash_attention/__init__.py",
                "src/dp_flash_attention/core.py",
                "src/dp_flash_attention/privacy.py",
                "src/dp_flash_attention/utils.py",
                "src/dp_flash_attention/kernels.py"
            ],
            "advanced_modules": [
                "src/dp_flash_attention/autonomous_improvements.py",
                "src/dp_flash_attention/advanced_research_engine.py",
                "src/dp_flash_attention/global_deployment_engine.py"
            ],
            "scaling_modules": [
                "src/dp_flash_attention/optimization.py",
                "src/dp_flash_attention/autoscaling.py",
                "src/dp_flash_attention/distributed.py",
                "src/dp_flash_attention/performance_tuning.py"
            ],
            "deployment_files": [
                "Dockerfile",
                "docker-compose.yml",
                "deployment/kubernetes.yaml",
                "deployment/deployment.yaml"
            ],
            "documentation": [
                "README.md",
                "ARCHITECTURE.md",
                "DEPLOYMENT.md",
                "SECURITY.md"
            ],
            "test_files": [
                "test_basic.py",
                "test_generation1_basic.py",
                "test_generation2_robustness.py",
                "test_generation3_scaling.py"
            ]
        }
    
    def run_all_quality_gates(self) -> Dict[str, QualityGateResult]:
        """Run all quality gates and return comprehensive results."""
        print("🛡️ Running Autonomous SDLC Quality Gates")
        print("=" * 50)
        
        gates = [
            ("Architecture Validation", self._validate_architecture),
            ("Implementation Completeness", self._validate_implementation),
            ("Documentation Quality", self._validate_documentation),
            ("Deployment Readiness", self._validate_deployment),
            ("Research Capabilities", self._validate_research),
            ("Global Scaling Features", self._validate_global_scaling),
            ("Quality Standards", self._validate_quality_standards),
            ("Security Compliance", self._validate_security)
        ]
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Running {gate_name}...")
            try:
                result = gate_func()
                self.results[gate_name] = result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} {gate_name}: {result.overall_score:.2f}/1.00")
            except Exception as e:
                print(f"❌ {gate_name} crashed: {e}")
                self.results[gate_name] = QualityGateResult(
                    gate_name=gate_name,
                    overall_score=0.0,
                    weighted_score=0.0,
                    metrics=[],
                    passed=False,
                    timestamp=time.time()
                )
        
        return self.results
    
    def _validate_architecture(self) -> QualityGateResult:
        """Validate system architecture and design."""
        metrics = []
        
        # Check core module structure
        core_score = self._check_file_structure("core_modules")
        metrics.append(QualityMetric(
            name="Core Module Structure",
            score=core_score,
            weight=0.3,
            details=f"Core modules completeness: {core_score:.1%}",
            status="pass" if core_score > 0.8 else "fail",
            recommendations=["Ensure all core modules are implemented"] if core_score < 1.0 else []
        ))
        
        # Check advanced module structure
        advanced_score = self._check_file_structure("advanced_modules")
        metrics.append(QualityMetric(
            name="Advanced Module Structure",
            score=advanced_score,
            weight=0.25,
            details=f"Advanced modules completeness: {advanced_score:.1%}",
            status="pass" if advanced_score > 0.8 else "fail",
            recommendations=["Implement missing advanced modules"] if advanced_score < 1.0 else []
        ))
        
        # Check scaling module structure
        scaling_score = self._check_file_structure("scaling_modules")
        metrics.append(QualityMetric(
            name="Scaling Module Structure",
            score=scaling_score,
            weight=0.25,
            details=f"Scaling modules completeness: {scaling_score:.1%}",
            status="pass" if scaling_score > 0.8 else "fail",
            recommendations=["Implement missing scaling modules"] if scaling_score < 1.0 else []
        ))
        
        # Check architecture documentation
        arch_doc_score = 1.0 if (self.repo_path / "ARCHITECTURE.md").exists() else 0.0
        metrics.append(QualityMetric(
            name="Architecture Documentation",
            score=arch_doc_score,
            weight=0.2,
            details="Architecture documentation exists" if arch_doc_score > 0 else "Missing ARCHITECTURE.md",
            status="pass" if arch_doc_score > 0 else "fail",
            recommendations=["Create comprehensive architecture documentation"] if arch_doc_score == 0 else []
        ))
        
        return self._calculate_gate_result("Architecture Validation", metrics)
    
    def _validate_implementation(self) -> QualityGateResult:
        """Validate implementation completeness across all generations."""
        metrics = []
        
        # Check Generation 1-3 implementation (existing)
        gen_files = [
            "test_generation1_basic.py",
            "test_generation2_robustness.py", 
            "test_generation3_scaling.py"
        ]
        
        gen_score = sum(1.0 for f in gen_files if (self.repo_path / f).exists()) / len(gen_files)
        metrics.append(QualityMetric(
            name="Core Generations Implementation",
            score=gen_score,
            weight=0.3,
            details=f"Generations 1-3 implementation: {gen_score:.1%}",
            status="pass" if gen_score > 0.8 else "fail",
            recommendations=["Complete missing generation implementations"] if gen_score < 1.0 else []
        ))
        
        # Check Generation 4 advanced features
        gen4_files = [
            "src/dp_flash_attention/autonomous_improvements.py",
            "src/dp_flash_attention/advanced_research_engine.py",
            "src/dp_flash_attention/global_deployment_engine.py"
        ]
        
        gen4_score = sum(1.0 for f in gen4_files if (self.repo_path / f).exists()) / len(gen4_files)
        metrics.append(QualityMetric(
            name="Generation 4 Advanced Features",
            score=gen4_score,
            weight=0.4,
            details=f"Advanced features implementation: {gen4_score:.1%}",
            status="pass" if gen4_score > 0.8 else "fail",
            recommendations=["Implement missing Generation 4 features"] if gen4_score < 1.0 else []
        ))
        
        # Check research validation
        research_files = ["advanced_research_validation.py", "research_outputs/"]
        research_score = sum(0.5 for f in research_files if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Research Validation",
            score=research_score,
            weight=0.15,
            details=f"Research validation completeness: {research_score:.1%}",
            status="pass" if research_score > 0.8 else "warning",
            recommendations=["Add comprehensive research validation"] if research_score < 1.0 else []
        ))
        
        # Check deployment automation
        deploy_files = ["deployment_config.py", "scaling_optimization.py"]
        deploy_score = sum(0.5 for f in deploy_files if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Deployment Automation",
            score=deploy_score,
            weight=0.15,
            details=f"Deployment automation: {deploy_score:.1%}",
            status="pass" if deploy_score > 0.8 else "warning",
            recommendations=["Enhance deployment automation"] if deploy_score < 1.0 else []
        ))
        
        return self._calculate_gate_result("Implementation Completeness", metrics)
    
    def _validate_documentation(self) -> QualityGateResult:
        """Validate documentation quality and completeness."""
        metrics = []
        
        # Check core documentation files
        doc_score = self._check_file_structure("documentation")
        metrics.append(QualityMetric(
            name="Core Documentation",
            score=doc_score,
            weight=0.3,
            details=f"Core documentation completeness: {doc_score:.1%}",
            status="pass" if doc_score > 0.8 else "fail",
            recommendations=["Create missing documentation files"] if doc_score < 1.0 else []
        ))
        
        # Check README quality
        readme_score = self._evaluate_readme_quality()
        metrics.append(QualityMetric(
            name="README Quality",
            score=readme_score,
            weight=0.25,
            details=f"README comprehensiveness: {readme_score:.1%}",
            status="pass" if readme_score > 0.8 else "warning",
            recommendations=["Enhance README with more examples and details"] if readme_score < 1.0 else []
        ))
        
        # Check research documentation
        research_docs = [
            "research_outputs/research_publication_1755030901.md",
            "TERRAGON_SDLC_COMPLETION_REPORT.md"
        ]
        research_doc_score = sum(1.0 for f in research_docs if (self.repo_path / f).exists()) / len(research_docs)
        metrics.append(QualityMetric(
            name="Research Documentation",
            score=research_doc_score,
            weight=0.2,
            details=f"Research documentation: {research_doc_score:.1%}",
            status="pass" if research_doc_score > 0.5 else "warning",
            recommendations=["Add research documentation and publications"] if research_doc_score < 1.0 else []
        ))
        
        # Check deployment documentation
        deploy_docs = ["DEPLOYMENT.md", "deployment/production_checklist.md"]
        deploy_doc_score = sum(0.5 for f in deploy_docs if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Deployment Documentation",
            score=deploy_doc_score,
            weight=0.25,
            details=f"Deployment documentation: {deploy_doc_score:.1%}",
            status="pass" if deploy_doc_score > 0.8 else "warning",
            recommendations=["Enhance deployment documentation"] if deploy_doc_score < 1.0 else []
        ))
        
        return self._calculate_gate_result("Documentation Quality", metrics)
    
    def _validate_deployment(self) -> QualityGateResult:
        """Validate deployment readiness and infrastructure."""
        metrics = []
        
        # Check containerization
        container_files = ["Dockerfile", "docker-compose.yml", "Dockerfile.prod"]
        container_score = sum(1.0 for f in container_files if (self.repo_path / f).exists()) / len(container_files)
        metrics.append(QualityMetric(
            name="Containerization",
            score=container_score,
            weight=0.25,
            details=f"Container configuration: {container_score:.1%}",
            status="pass" if container_score > 0.6 else "fail",
            recommendations=["Complete Docker configuration"] if container_score < 1.0 else []
        ))
        
        # Check Kubernetes deployment
        k8s_files = [
            "deployment/kubernetes.yaml",
            "deployment/deployment.yaml", 
            "deployment/hpa.yaml"
        ]
        k8s_score = sum(1.0 for f in k8s_files if (self.repo_path / f).exists()) / len(k8s_files)
        metrics.append(QualityMetric(
            name="Kubernetes Configuration",
            score=k8s_score,
            weight=0.25,
            details=f"Kubernetes setup: {k8s_score:.1%}",
            status="pass" if k8s_score > 0.6 else "warning",
            recommendations=["Complete Kubernetes deployment configuration"] if k8s_score < 1.0 else []
        ))
        
        # Check monitoring and observability
        monitor_files = [
            "monitoring/prometheus.yml",
            "monitoring/grafana/dashboards/privacy-dashboard.json",
            "monitoring/privacy-metrics.py"
        ]
        monitor_score = sum(1.0 for f in monitor_files if (self.repo_path / f).exists()) / len(monitor_files)
        metrics.append(QualityMetric(
            name="Monitoring Setup",
            score=monitor_score,
            weight=0.2,
            details=f"Monitoring configuration: {monitor_score:.1%}",
            status="pass" if monitor_score > 0.6 else "warning",
            recommendations=["Enhance monitoring and observability"] if monitor_score < 1.0 else []
        ))
        
        # Check deployment automation
        deploy_scripts = [
            "deployment/deploy_docker.sh",
            "deployment/deploy_k8s.sh",
            "scripts/entrypoint.sh"
        ]
        script_score = sum(1.0 for f in deploy_scripts if (self.repo_path / f).exists()) / len(deploy_scripts)
        metrics.append(QualityMetric(
            name="Deployment Automation",
            score=script_score,
            weight=0.3,
            details=f"Deployment scripts: {script_score:.1%}",
            status="pass" if script_score > 0.6 else "warning",
            recommendations=["Add deployment automation scripts"] if script_score < 1.0 else []
        ))
        
        return self._calculate_gate_result("Deployment Readiness", metrics)
    
    def _validate_research(self) -> QualityGateResult:
        """Validate research capabilities and validation."""
        metrics = []
        
        # Check research engine implementation
        research_impl_score = 1.0 if (self.repo_path / "src/dp_flash_attention/advanced_research_engine.py").exists() else 0.0
        metrics.append(QualityMetric(
            name="Research Engine Implementation",
            score=research_impl_score,
            weight=0.3,
            details="Advanced research engine implemented" if research_impl_score > 0 else "Missing research engine",
            status="pass" if research_impl_score > 0 else "fail",
            recommendations=["Implement advanced research engine"] if research_impl_score == 0 else []
        ))
        
        # Check research validation scripts
        validation_files = [
            "advanced_research_validation.py",
            "minimal_research_validation.py"
        ]
        validation_score = sum(0.5 for f in validation_files if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Research Validation",
            score=validation_score,
            weight=0.25,
            details=f"Research validation scripts: {validation_score:.1%}",
            status="pass" if validation_score > 0.8 else "warning",
            recommendations=["Add comprehensive research validation"] if validation_score < 1.0 else []
        ))
        
        # Check research outputs and results
        output_dirs = ["research_outputs/", "robustness_outputs/", "scaling_outputs/"]
        output_score = sum(1.0 for d in output_dirs if (self.repo_path / d).exists()) / len(output_dirs)
        metrics.append(QualityMetric(
            name="Research Outputs",
            score=output_score,
            weight=0.2,
            details=f"Research output directories: {output_score:.1%}",
            status="pass" if output_score > 0.6 else "warning",
            recommendations=["Generate research outputs and results"] if output_score < 1.0 else []
        ))
        
        # Check benchmarking capabilities
        benchmark_files = [
            "src/dp_flash_attention/benchmarking.py",
            "tests/benchmarks/test_performance.py"
        ]
        benchmark_score = sum(0.5 for f in benchmark_files if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Benchmarking Capabilities",
            score=benchmark_score,
            weight=0.25,
            details=f"Benchmarking implementation: {benchmark_score:.1%}",
            status="pass" if benchmark_score > 0.8 else "warning",
            recommendations=["Enhance benchmarking capabilities"] if benchmark_score < 1.0 else []
        ))
        
        return self._calculate_gate_result("Research Capabilities", metrics)
    
    def _validate_global_scaling(self) -> QualityGateResult:
        """Validate global scaling and deployment features."""
        metrics = []
        
        # Check global deployment engine
        global_engine_score = 1.0 if (self.repo_path / "src/dp_flash_attention/global_deployment_engine.py").exists() else 0.0
        metrics.append(QualityMetric(
            name="Global Deployment Engine",
            score=global_engine_score,
            weight=0.3,
            details="Global deployment engine implemented" if global_engine_score > 0 else "Missing global deployment",
            status="pass" if global_engine_score > 0 else "fail",
            recommendations=["Implement global deployment engine"] if global_engine_score == 0 else []
        ))
        
        # Check autoscaling implementation
        autoscaling_files = [
            "src/dp_flash_attention/autoscaling.py",
            "deployment/hpa.yaml"
        ]
        autoscaling_score = sum(0.5 for f in autoscaling_files if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Autoscaling Implementation",
            score=autoscaling_score,
            weight=0.25,
            details=f"Autoscaling features: {autoscaling_score:.1%}",
            status="pass" if autoscaling_score > 0.8 else "warning",
            recommendations=["Complete autoscaling implementation"] if autoscaling_score < 1.0 else []
        ))
        
        # Check distributed processing
        distributed_files = [
            "src/dp_flash_attention/distributed.py",
            "src/dp_flash_attention/concurrent.py"
        ]
        distributed_score = sum(0.5 for f in distributed_files if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Distributed Processing",
            score=distributed_score,
            weight=0.25,
            details=f"Distributed processing: {distributed_score:.1%}",
            status="pass" if distributed_score > 0.8 else "warning",
            recommendations=["Enhance distributed processing capabilities"] if distributed_score < 1.0 else []
        ))
        
        # Check globalization features
        globalization_files = [
            "src/dp_flash_attention/globalization.py",
            "config/production.json"
        ]
        globalization_score = sum(0.5 for f in globalization_files if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Globalization Features",
            score=globalization_score,
            weight=0.2,
            details=f"Globalization support: {globalization_score:.1%}",
            status="pass" if globalization_score > 0.5 else "warning",
            recommendations=["Add i18n and global compliance features"] if globalization_score < 1.0 else []
        ))
        
        return self._calculate_gate_result("Global Scaling Features", metrics)
    
    def _validate_quality_standards(self) -> QualityGateResult:
        """Validate code quality standards and best practices."""
        metrics = []
        
        # Check project configuration
        config_files = ["pyproject.toml", "requirements.txt", "requirements-dev.txt"]
        config_score = sum(1.0 for f in config_files if (self.repo_path / f).exists()) / len(config_files)
        metrics.append(QualityMetric(
            name="Project Configuration",
            score=config_score,
            weight=0.2,
            details=f"Project configuration files: {config_score:.1%}",
            status="pass" if config_score > 0.6 else "warning",
            recommendations=["Complete project configuration"] if config_score < 1.0 else []
        ))
        
        # Check testing infrastructure
        test_score = self._check_file_structure("test_files")
        metrics.append(QualityMetric(
            name="Testing Infrastructure",
            score=test_score,
            weight=0.3,
            details=f"Test coverage: {test_score:.1%}",
            status="pass" if test_score > 0.7 else "fail",
            recommendations=["Expand test coverage"] if test_score < 1.0 else []
        ))
        
        # Check code organization
        org_score = self._evaluate_code_organization()
        metrics.append(QualityMetric(
            name="Code Organization",
            score=org_score,
            weight=0.25,
            details=f"Code organization quality: {org_score:.1%}",
            status="pass" if org_score > 0.8 else "warning",
            recommendations=["Improve code structure and organization"] if org_score < 1.0 else []
        ))
        
        # Check quality gates implementation
        qg_score = 1.0 if (self.repo_path / "quality_gates.py").exists() else 0.0
        metrics.append(QualityMetric(
            name="Quality Gates",
            score=qg_score,
            weight=0.25,
            details="Quality gates implemented" if qg_score > 0 else "Missing quality gates",
            status="pass" if qg_score > 0 else "warning",
            recommendations=["Implement comprehensive quality gates"] if qg_score == 0 else []
        ))
        
        return self._calculate_gate_result("Quality Standards", metrics)
    
    def _validate_security(self) -> QualityGateResult:
        """Validate security compliance and privacy features."""
        metrics = []
        
        # Check security implementation
        security_files = [
            "src/dp_flash_attention/security.py",
            "SECURITY.md",
            "scripts/cuda_security_check.py"
        ]
        security_score = sum(1.0 for f in security_files if (self.repo_path / f).exists()) / len(security_files)
        metrics.append(QualityMetric(
            name="Security Implementation",
            score=security_score,
            weight=0.3,
            details=f"Security features: {security_score:.1%}",
            status="pass" if security_score > 0.6 else "fail",
            recommendations=["Enhance security implementation"] if security_score < 1.0 else []
        ))
        
        # Check privacy implementation
        privacy_files = [
            "src/dp_flash_attention/privacy.py",
            "scripts/check_privacy_params.py",
            "tests/privacy/test_privacy_guarantees.py"
        ]
        privacy_score = sum(1.0 for f in privacy_files if (self.repo_path / f).exists()) / len(privacy_files)
        metrics.append(QualityMetric(
            name="Privacy Implementation",
            score=privacy_score,
            weight=0.35,
            details=f"Privacy features: {privacy_score:.1%}",
            status="pass" if privacy_score > 0.8 else "fail",
            recommendations=["Complete privacy implementation"] if privacy_score < 1.0 else []
        ))
        
        # Check compliance documentation
        compliance_files = [
            "docs/compliance/SBOM-policy.md",
            "docs/compliance/SOC2-compliance.md",
            "deployment/security_policy.md"
        ]
        compliance_score = sum(1.0 for f in compliance_files if (self.repo_path / f).exists()) / len(compliance_files)
        metrics.append(QualityMetric(
            name="Compliance Documentation",
            score=compliance_score,
            weight=0.2,
            details=f"Compliance documentation: {compliance_score:.1%}",
            status="pass" if compliance_score > 0.6 else "warning",
            recommendations=["Add compliance documentation"] if compliance_score < 1.0 else []
        ))
        
        # Check security policies
        policy_files = [
            "deployment/network-policy.yaml",
            "deployment/security_policy.md"
        ]
        policy_score = sum(0.5 for f in policy_files if (self.repo_path / f).exists())
        metrics.append(QualityMetric(
            name="Security Policies",
            score=policy_score,
            weight=0.15,
            details=f"Security policies: {policy_score:.1%}",
            status="pass" if policy_score > 0.5 else "warning",
            recommendations=["Define comprehensive security policies"] if policy_score < 1.0 else []
        ))
        
        return self._calculate_gate_result("Security Compliance", metrics)
    
    def _check_file_structure(self, category: str) -> float:
        """Check completeness of file structure for a category."""
        files = self.expected_structure.get(category, [])
        if not files:
            return 1.0
        
        existing_files = sum(1.0 for f in files if (self.repo_path / f).exists())
        return existing_files / len(files)
    
    def _evaluate_readme_quality(self) -> float:
        """Evaluate README.md quality and completeness."""
        readme_path = self.repo_path / "README.md"
        if not readme_path.exists():
            return 0.0
        
        try:
            content = readme_path.read_text()
            
            # Check for key sections
            required_sections = [
                "overview", "installation", "usage", "example", 
                "requirements", "performance", "privacy", "license"
            ]
            
            score = 0.0
            for section in required_sections:
                if section.lower() in content.lower():
                    score += 1.0
            
            # Bonus for comprehensive content
            if len(content) > 5000:  # Substantial README
                score += 1.0
            if "```" in content:  # Code examples
                score += 1.0
            
            return min(1.0, score / len(required_sections))
            
        except Exception:
            return 0.5  # Partial credit for existing but unreadable README
    
    def _evaluate_code_organization(self) -> float:
        """Evaluate overall code organization quality."""
        score = 0.0
        
        # Check source structure
        src_path = self.repo_path / "src"
        if src_path.exists():
            score += 0.2
        
        # Check package structure
        pkg_path = src_path / "dp_flash_attention" if src_path.exists() else None
        if pkg_path and pkg_path.exists():
            score += 0.2
        
        # Check init file
        if pkg_path and (pkg_path / "__init__.py").exists():
            score += 0.2
        
        # Check modular organization
        if pkg_path:
            module_count = len([f for f in pkg_path.iterdir() if f.is_file() and f.suffix == '.py'])
            if module_count >= 10:  # Well-modularized
                score += 0.2
        
        # Check test organization
        test_dirs = ["tests/", "test_*.py"]
        if any((self.repo_path / pattern).exists() or 
               list(self.repo_path.glob(pattern)) for pattern in test_dirs):
            score += 0.2
        
        return score
    
    def _calculate_gate_result(self, gate_name: str, metrics: List[QualityMetric]) -> QualityGateResult:
        """Calculate overall result for a quality gate."""
        if not metrics:
            return QualityGateResult(
                gate_name=gate_name,
                overall_score=0.0,
                weighted_score=0.0,
                metrics=[],
                passed=False,
                timestamp=time.time()
            )
        
        # Calculate weighted score
        total_weight = sum(m.weight for m in metrics)
        weighted_score = sum(m.score * m.weight for m in metrics) / total_weight if total_weight > 0 else 0.0
        
        # Calculate simple average
        overall_score = sum(m.score for m in metrics) / len(metrics)
        
        # Gate passes if weighted score meets threshold
        passed = weighted_score >= self.pass_threshold
        
        return QualityGateResult(
            gate_name=gate_name,
            overall_score=overall_score,
            weighted_score=weighted_score,
            metrics=metrics,
            passed=passed,
            timestamp=time.time()
        )
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive quality gate report."""
        if not self.results:
            return "No quality gate results available."
        
        # Calculate overall statistics
        total_gates = len(self.results)
        passed_gates = sum(1 for r in self.results.values() if r.passed)
        overall_score = sum(r.weighted_score for r in self.results.values()) / total_gates
        
        # Generate report
        report = f"""
# Autonomous SDLC Quality Gates Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Overall Score**: {overall_score:.2f}/1.00 ({overall_score*100:.1f}%)
- **Gates Passed**: {passed_gates}/{total_gates}
- **Status**: {'✅ PASS' if overall_score >= self.pass_threshold else '❌ FAIL'}

## Quality Gate Results

"""
        
        for gate_name, result in self.results.items():
            status_emoji = "✅" if result.passed else "❌"
            report += f"### {status_emoji} {gate_name}\n"
            report += f"**Score**: {result.weighted_score:.2f}/1.00 ({result.weighted_score*100:.1f}%)\n\n"
            
            for metric in result.metrics:
                metric_status = {"pass": "✅", "fail": "❌", "warning": "⚠️"}.get(metric.status, "❓")
                report += f"- {metric_status} **{metric.name}**: {metric.score:.2f} ({metric.details})\n"
                
                if metric.recommendations:
                    for rec in metric.recommendations:
                        report += f"  - 💡 {rec}\n"
            
            report += "\n"
        
        # Add recommendations section
        all_recommendations = []
        for result in self.results.values():
            for metric in result.metrics:
                all_recommendations.extend(metric.recommendations)
        
        if all_recommendations:
            report += "## Priority Recommendations\n\n"
            for i, rec in enumerate(set(all_recommendations), 1):
                report += f"{i}. {rec}\n"
        
        report += f"""
## Quality Gates Summary

| Gate | Score | Status | Key Issues |
|------|-------|--------|------------|
"""
        
        for gate_name, result in self.results.items():
            status = "PASS" if result.passed else "FAIL"
            key_issues = len([m for m in result.metrics if m.status == "fail"])
            report += f"| {gate_name} | {result.weighted_score:.2f} | {status} | {key_issues} |\n"
        
        return report
    
    def save_report(self, filepath: str):
        """Save comprehensive report to file."""
        report = self.generate_comprehensive_report()
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        # Also save JSON data
        json_filepath = filepath.replace('.md', '.json')
        json_data = {
            "timestamp": time.time(),
            "overall_score": sum(r.weighted_score for r in self.results.values()) / len(self.results) if self.results else 0.0,
            "total_gates": len(self.results),
            "passed_gates": sum(1 for r in self.results.values() if r.passed),
            "results": {name: asdict(result) for name, result in self.results.items()}
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)


def main():
    """Main execution function."""
    print("🛡️ Autonomous SDLC Quality Gates Validation")
    print("=" * 55)
    
    # Initialize quality gates
    quality_gates = AutonomousSDLCQualityGates()
    
    # Run all quality gates
    results = quality_gates.run_all_quality_gates()
    
    # Generate and display summary
    print("\n" + "=" * 55)
    print("📊 Quality Gates Summary")
    print("=" * 55)
    
    total_gates = len(results)
    passed_gates = sum(1 for r in results.values() if r.passed)
    overall_score = sum(r.weighted_score for r in results.values()) / total_gates if total_gates > 0 else 0.0
    
    for gate_name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} {gate_name}: {result.weighted_score:.2f}/1.00")
    
    print(f"\n🎯 Overall Result: {passed_gates}/{total_gates} gates passed")
    print(f"📊 Overall Score: {overall_score:.2f}/1.00 ({overall_score*100:.1f}%)")
    
    if overall_score >= quality_gates.pass_threshold:
        print("🎉 Autonomous SDLC Quality Gates: PASSED")
        final_status = True
    else:
        print("⚠️  Autonomous SDLC Quality Gates: FAILED")
        final_status = False
    
    # Save comprehensive report
    report_path = "AUTONOMOUS_SDLC_QUALITY_REPORT.md"
    quality_gates.save_report(report_path)
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return final_status


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)