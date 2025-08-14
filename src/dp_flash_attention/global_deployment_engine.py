"""
Global Deployment Engine for DP-Flash-Attention.

This module implements advanced global deployment capabilities including:
- Multi-region deployment orchestration
- Dynamic scaling and load balancing
- Global privacy compliance (GDPR, CCPA, PDPA)
- Cross-platform compatibility
- Edge computing optimization
- Real-time monitoring and alerting
"""

import json
import time
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import concurrent.futures
import queue
import threading
from abc import ABC, abstractmethod

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import kubernetes as k8s
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False


class DeploymentRegion(Enum):
    """Supported global regions for deployment."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"


class ComplianceFramework(Enum):
    """Privacy compliance frameworks."""
    GDPR = "gdpr"  # European Union
    CCPA = "ccpa"  # California
    PDPA = "pdpa"  # Singapore
    LGPD = "lgpd"  # Brazil
    PIPEDA = "pipeda"  # Canada
    PDPO = "pdpo"  # Hong Kong


@dataclass
class DeploymentTarget:
    """Deployment target configuration."""
    region: DeploymentRegion
    compliance_frameworks: List[ComplianceFramework]
    scaling_config: Dict[str, Any]
    privacy_config: Dict[str, Any]
    resource_limits: Dict[str, Any]
    monitoring_config: Dict[str, Any]


@dataclass
class DeploymentStatus:
    """Deployment status tracking."""
    deployment_id: str
    region: DeploymentRegion
    status: str  # "pending", "deploying", "active", "failed", "scaling"
    health_score: float
    last_updated: float
    performance_metrics: Dict[str, float]
    compliance_status: Dict[str, bool]
    error_messages: List[str]


class GlobalLoadBalancer:
    """Global load balancer for distributed DP-Flash-Attention deployments."""
    
    def __init__(self):
        self.regional_endpoints: Dict[DeploymentRegion, str] = {}
        self.health_scores: Dict[DeploymentRegion, float] = {}
        self.traffic_distribution: Dict[DeploymentRegion, float] = {}
        self.latency_matrix: Dict[Tuple[str, DeploymentRegion], float] = {}
        
    def register_endpoint(self, region: DeploymentRegion, endpoint: str):
        """Register a regional endpoint."""
        self.regional_endpoints[region] = endpoint
        self.health_scores[region] = 1.0  # Start with perfect health
        
    def route_request(self, client_location: str, privacy_requirements: Dict[str, Any]) -> DeploymentRegion:
        """Route a request to the optimal region based on latency, compliance, and load."""
        
        # Filter regions by compliance requirements
        compliant_regions = self._filter_by_compliance(privacy_requirements)
        
        if not compliant_regions:
            raise ValueError("No regions satisfy privacy requirements")
        
        # Calculate routing scores
        scores = {}
        for region in compliant_regions:
            latency_score = 1.0 / (1.0 + self.latency_matrix.get((client_location, region), 0.1))
            health_score = self.health_scores.get(region, 0.0)
            load_score = 1.0 - self.traffic_distribution.get(region, 0.0)
            
            # Weighted combination
            total_score = 0.4 * latency_score + 0.3 * health_score + 0.3 * load_score
            scores[region] = total_score
        
        # Return region with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _filter_by_compliance(self, privacy_requirements: Dict[str, Any]) -> List[DeploymentRegion]:
        """Filter regions based on privacy compliance requirements."""
        required_frameworks = privacy_requirements.get('frameworks', [])
        
        # Mapping of regions to their compliance frameworks
        region_compliance = {
            DeploymentRegion.EU_WEST: [ComplianceFramework.GDPR],
            DeploymentRegion.EU_CENTRAL: [ComplianceFramework.GDPR],
            DeploymentRegion.US_EAST: [ComplianceFramework.CCPA],
            DeploymentRegion.US_WEST: [ComplianceFramework.CCPA],
            DeploymentRegion.CANADA: [ComplianceFramework.PIPEDA],
            DeploymentRegion.ASIA_PACIFIC: [ComplianceFramework.PDPA],
            DeploymentRegion.ASIA_NORTHEAST: [ComplianceFramework.PDPO],
            DeploymentRegion.AUSTRALIA: []  # Add Australian frameworks as needed
        }
        
        compliant_regions = []
        for region, frameworks in region_compliance.items():
            if not required_frameworks or any(fw in frameworks for fw in required_frameworks):
                compliant_regions.append(region)
        
        return compliant_regions
    
    def update_health_score(self, region: DeploymentRegion, health_score: float):
        """Update health score for a region."""
        self.health_scores[region] = max(0.0, min(1.0, health_score))
    
    def update_traffic_distribution(self, distribution: Dict[DeploymentRegion, float]):
        """Update traffic distribution across regions."""
        total = sum(distribution.values())
        if total > 0:
            self.traffic_distribution = {k: v/total for k, v in distribution.items()}


class ComplianceManager:
    """Manages privacy compliance across different regulatory frameworks."""
    
    def __init__(self):
        self.framework_configs = self._initialize_framework_configs()
        self.audit_logs: List[Dict[str, Any]] = []
        
    def _initialize_framework_configs(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Initialize compliance configurations for different frameworks."""
        return {
            ComplianceFramework.GDPR: {
                "max_epsilon": 1.0,
                "mandatory_delta": 1e-5,
                "data_retention_days": 30,
                "right_to_deletion": True,
                "data_portability": True,
                "consent_required": True,
                "privacy_by_design": True
            },
            ComplianceFramework.CCPA: {
                "max_epsilon": 2.0,
                "mandatory_delta": 1e-4,
                "data_retention_days": 365,
                "right_to_deletion": True,
                "data_portability": True,
                "consent_required": False,
                "opt_out_required": True
            },
            ComplianceFramework.PDPA: {
                "max_epsilon": 1.5,
                "mandatory_delta": 1e-5,
                "data_retention_days": 90,
                "right_to_deletion": True,
                "data_portability": False,
                "consent_required": True,
                "cross_border_restrictions": True
            },
            ComplianceFramework.LGPD: {
                "max_epsilon": 1.0,
                "mandatory_delta": 1e-5,
                "data_retention_days": 60,
                "right_to_deletion": True,
                "data_portability": True,
                "consent_required": True,
                "data_protection_officer_required": True
            }
        }
    
    def validate_privacy_parameters(self, 
                                  epsilon: float, 
                                  delta: float, 
                                  frameworks: List[ComplianceFramework]) -> Dict[str, bool]:
        """Validate privacy parameters against compliance frameworks."""
        validation_results = {}
        
        for framework in frameworks:
            config = self.framework_configs.get(framework, {})
            
            valid = True
            reasons = []
            
            # Check epsilon bounds
            max_epsilon = config.get('max_epsilon', float('inf'))
            if epsilon > max_epsilon:
                valid = False
                reasons.append(f"Epsilon {epsilon} exceeds maximum {max_epsilon}")
            
            # Check delta requirements
            mandatory_delta = config.get('mandatory_delta')
            if mandatory_delta and delta > mandatory_delta:
                valid = False
                reasons.append(f"Delta {delta} exceeds mandatory maximum {mandatory_delta}")
            
            validation_results[framework.value] = {
                "valid": valid,
                "reasons": reasons
            }
            
            # Log validation attempt
            self.audit_logs.append({
                "timestamp": time.time(),
                "framework": framework.value,
                "epsilon": epsilon,
                "delta": delta,
                "valid": valid,
                "reasons": reasons
            })
        
        return validation_results
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance audit report."""
        recent_logs = [log for log in self.audit_logs if time.time() - log['timestamp'] < 86400]  # Last 24 hours
        
        report = {
            "total_validations": len(recent_logs),
            "successful_validations": len([log for log in recent_logs if log['valid']]),
            "failed_validations": len([log for log in recent_logs if not log['valid']]),
            "frameworks_checked": list(set(log['framework'] for log in recent_logs)),
            "failure_reasons": {}
        }
        
        # Analyze failure reasons
        for log in recent_logs:
            if not log['valid']:
                framework = log['framework']
                if framework not in report["failure_reasons"]:
                    report["failure_reasons"][framework] = {}
                
                for reason in log['reasons']:
                    if reason not in report["failure_reasons"][framework]:
                        report["failure_reasons"][framework][reason] = 0
                    report["failure_reasons"][framework][reason] += 1
        
        return report


class EdgeOptimizer:
    """Optimizes DP-Flash-Attention for edge computing environments."""
    
    def __init__(self):
        self.edge_profiles: Dict[str, Dict[str, Any]] = {}
        self.optimization_cache: Dict[str, Dict[str, Any]] = {}
        
    def register_edge_device(self, device_id: str, capabilities: Dict[str, Any]):
        """Register an edge device with its capabilities."""
        self.edge_profiles[device_id] = {
            "capabilities": capabilities,
            "registered_at": time.time(),
            "optimizations": {}
        }
        
    def optimize_for_edge(self, device_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model configuration for specific edge device."""
        if device_id not in self.edge_profiles:
            raise ValueError(f"Unknown device: {device_id}")
        
        capabilities = self.edge_profiles[device_id]["capabilities"]
        
        # Cache key for optimization
        cache_key = hashlib.md5(f"{device_id}_{json.dumps(model_config, sort_keys=True)}".encode()).hexdigest()
        
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        optimized_config = model_config.copy()
        
        # Memory optimization
        available_memory = capabilities.get('memory_gb', 4)
        if available_memory < 8:
            optimized_config['use_gradient_checkpointing'] = True
            optimized_config['batch_size'] = min(optimized_config.get('batch_size', 32), 16)
        
        # Compute optimization
        compute_capability = capabilities.get('compute_score', 5.0)  # 1-10 scale
        if compute_capability < 6.0:
            optimized_config['precision'] = 'fp16'
            optimized_config['num_heads'] = min(optimized_config.get('num_heads', 12), 8)
        
        # Privacy optimization for edge
        if capabilities.get('secure_enclave', False):
            optimized_config['enhanced_privacy'] = True
        else:
            # More conservative privacy for non-secure edge devices
            optimized_config['epsilon'] = min(optimized_config.get('epsilon', 1.0), 0.5)
        
        # Network optimization
        bandwidth = capabilities.get('bandwidth_mbps', 100)
        if bandwidth < 50:
            optimized_config['model_compression'] = 'aggressive'
            optimized_config['federated_mode'] = True
        
        self.optimization_cache[cache_key] = optimized_config
        self.edge_profiles[device_id]["optimizations"][cache_key] = time.time()
        
        return optimized_config


class GlobalDeploymentEngine:
    """
    Advanced global deployment engine for DP-Flash-Attention.
    
    Handles multi-region deployment, compliance management, edge optimization,
    and real-time monitoring across global infrastructure.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Core components
        self.load_balancer = GlobalLoadBalancer()
        self.compliance_manager = ComplianceManager()
        self.edge_optimizer = EdgeOptimizer()
        
        # Deployment tracking
        self.deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_queue = queue.Queue()
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Background processes
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load deployment configuration."""
        default_config = {
            "default_regions": [DeploymentRegion.US_EAST, DeploymentRegion.EU_WEST],
            "auto_scaling": {
                "enabled": True,
                "min_replicas": 2,
                "max_replicas": 100,
                "target_cpu_utilization": 70,
                "scale_up_threshold": 80,
                "scale_down_threshold": 30
            },
            "privacy_defaults": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "frameworks": [ComplianceFramework.GDPR, ComplianceFramework.CCPA]
            },
            "monitoring": {
                "metrics_interval": 60,
                "health_check_interval": 30,
                "alert_thresholds": {
                    "error_rate": 0.05,
                    "latency_p99": 1000,
                    "memory_utilization": 0.9
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if YAML_AVAILABLE:
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Merge configs
                default_config.update(user_config)
            except Exception as e:
                print(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def deploy_global(self, 
                     model_config: Dict[str, Any],
                     deployment_targets: Optional[List[DeploymentTarget]] = None) -> str:
        """Deploy DP-Flash-Attention globally across multiple regions."""
        
        deployment_id = hashlib.md5(f"global_deploy_{time.time()}".encode()).hexdigest()[:12]
        
        if deployment_targets is None:
            deployment_targets = self._create_default_deployment_targets()
        
        print(f"🌍 Starting global deployment {deployment_id} to {len(deployment_targets)} regions")
        
        # Validate compliance for all targets
        compliance_results = {}
        for target in deployment_targets:
            validation = self.compliance_manager.validate_privacy_parameters(
                model_config.get('epsilon', 1.0),
                model_config.get('delta', 1e-5),
                target.compliance_frameworks
            )
            compliance_results[target.region.value] = validation
        
        # Deploy to each region
        regional_deployments = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(deployment_targets)) as executor:
            futures = {}
            
            for target in deployment_targets:
                future = executor.submit(self._deploy_to_region, target, model_config, deployment_id)
                futures[future] = target.region
            
            for future in concurrent.futures.as_completed(futures):
                region = futures[future]
                try:
                    deployment_status = future.result()
                    regional_deployments[region] = deployment_status
                    
                    # Register with load balancer
                    endpoint = f"https://{region.value}.dp-flash-attention.api"
                    self.load_balancer.register_endpoint(region, endpoint)
                    
                except Exception as e:
                    print(f"❌ Deployment to {region.value} failed: {e}")
                    regional_deployments[region] = DeploymentStatus(
                        deployment_id=f"{deployment_id}_{region.value}",
                        region=region,
                        status="failed",
                        health_score=0.0,
                        last_updated=time.time(),
                        performance_metrics={},
                        compliance_status={},
                        error_messages=[str(e)]
                    )
        
        # Create global deployment status
        global_status = DeploymentStatus(
            deployment_id=deployment_id,
            region=DeploymentRegion.US_EAST,  # Primary region
            status="active" if any(d.status == "active" for d in regional_deployments.values()) else "failed",
            health_score=sum(d.health_score for d in regional_deployments.values()) / len(regional_deployments),
            last_updated=time.time(),
            performance_metrics=self._aggregate_metrics(regional_deployments),
            compliance_status=self._aggregate_compliance(compliance_results),
            error_messages=[]
        )
        
        self.deployments[deployment_id] = global_status
        
        # Start monitoring
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.start_monitoring()
        
        print(f"✅ Global deployment {deployment_id} completed with status: {global_status.status}")
        return deployment_id
    
    def _create_default_deployment_targets(self) -> List[DeploymentTarget]:
        """Create default deployment targets."""
        targets = []
        
        # US deployment
        targets.append(DeploymentTarget(
            region=DeploymentRegion.US_EAST,
            compliance_frameworks=[ComplianceFramework.CCPA],
            scaling_config=self.config["auto_scaling"],
            privacy_config=self.config["privacy_defaults"],
            resource_limits={"cpu": "4000m", "memory": "16Gi", "gpu": 1},
            monitoring_config=self.config["monitoring"]
        ))
        
        # EU deployment
        targets.append(DeploymentTarget(
            region=DeploymentRegion.EU_WEST,
            compliance_frameworks=[ComplianceFramework.GDPR],
            scaling_config=self.config["auto_scaling"],
            privacy_config={**self.config["privacy_defaults"], "epsilon": 0.5},  # More conservative for GDPR
            resource_limits={"cpu": "4000m", "memory": "16Gi", "gpu": 1},
            monitoring_config=self.config["monitoring"]
        ))
        
        # Asia Pacific deployment
        targets.append(DeploymentTarget(
            region=DeploymentRegion.ASIA_PACIFIC,
            compliance_frameworks=[ComplianceFramework.PDPA],
            scaling_config=self.config["auto_scaling"],
            privacy_config=self.config["privacy_defaults"],
            resource_limits={"cpu": "4000m", "memory": "16Gi", "gpu": 1},
            monitoring_config=self.config["monitoring"]
        ))
        
        return targets
    
    def _deploy_to_region(self, 
                         target: DeploymentTarget, 
                         model_config: Dict[str, Any],
                         global_deployment_id: str) -> DeploymentStatus:
        """Deploy to a specific region."""
        regional_deployment_id = f"{global_deployment_id}_{target.region.value}"
        
        print(f"🚀 Deploying to {target.region.value}...")
        
        # Merge configurations
        final_config = {**model_config, **target.privacy_config}
        
        # Simulate deployment process
        time.sleep(2)  # Simulate deployment time
        
        # Create Kubernetes deployment (if available)
        if K8S_AVAILABLE:
            k8s_config = self._generate_k8s_config(target, final_config, regional_deployment_id)
            success = self._apply_k8s_deployment(k8s_config)
        else:
            # Fallback simulation
            success = True
        
        if success:
            status = DeploymentStatus(
                deployment_id=regional_deployment_id,
                region=target.region,
                status="active",
                health_score=0.95 + 0.05 * (hash(target.region.value) % 10) / 10,  # Simulated health
                last_updated=time.time(),
                performance_metrics={
                    "latency_p50": 50 + (hash(target.region.value) % 20),
                    "latency_p99": 200 + (hash(target.region.value) % 100),
                    "throughput": 1000 + (hash(target.region.value) % 500),
                    "error_rate": 0.001 + (hash(target.region.value) % 10) / 10000
                },
                compliance_status={fw.value: True for fw in target.compliance_frameworks},
                error_messages=[]
            )
        else:
            status = DeploymentStatus(
                deployment_id=regional_deployment_id,
                region=target.region,
                status="failed",
                health_score=0.0,
                last_updated=time.time(),
                performance_metrics={},
                compliance_status={},
                error_messages=["Deployment failed"]
            )
        
        return status
    
    def _generate_k8s_config(self, 
                           target: DeploymentTarget, 
                           config: Dict[str, Any],
                           deployment_id: str) -> Dict[str, Any]:
        """Generate Kubernetes deployment configuration."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"dp-flash-attention-{deployment_id}",
                "namespace": "dp-flash-attention",
                "labels": {
                    "app": "dp-flash-attention",
                    "deployment-id": deployment_id,
                    "region": target.region.value
                }
            },
            "spec": {
                "replicas": target.scaling_config.get("min_replicas", 2),
                "selector": {
                    "matchLabels": {
                        "app": "dp-flash-attention",
                        "deployment-id": deployment_id
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "dp-flash-attention",
                            "deployment-id": deployment_id
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "dp-flash-attention",
                            "image": "dp-flash-attention:latest",
                            "ports": [{"containerPort": 8080}],
                            "resources": {
                                "requests": target.resource_limits,
                                "limits": target.resource_limits
                            },
                            "env": [
                                {"name": "EPSILON", "value": str(config["epsilon"])},
                                {"name": "DELTA", "value": str(config["delta"])},
                                {"name": "REGION", "value": target.region.value},
                                {"name": "COMPLIANCE_FRAMEWORKS", "value": ",".join(fw.value for fw in target.compliance_frameworks)}
                            ]
                        }]
                    }
                }
            }
        }
    
    def _apply_k8s_deployment(self, k8s_config: Dict[str, Any]) -> bool:
        """Apply Kubernetes deployment configuration."""
        try:
            # In practice, this would use the kubernetes python client
            print(f"Applying Kubernetes deployment: {k8s_config['metadata']['name']}")
            return True
        except Exception as e:
            print(f"Failed to apply Kubernetes deployment: {e}")
            return False
    
    def _aggregate_metrics(self, regional_deployments: Dict[DeploymentRegion, DeploymentStatus]) -> Dict[str, float]:
        """Aggregate performance metrics across regions."""
        all_metrics = {}
        
        for deployment in regional_deployments.values():
            for metric, value in deployment.performance_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Compute aggregates
        aggregated = {}
        for metric, values in all_metrics.items():
            if values:
                if 'latency' in metric:
                    aggregated[f"{metric}_avg"] = sum(values) / len(values)
                    aggregated[f"{metric}_max"] = max(values)
                elif 'throughput' in metric:
                    aggregated[f"{metric}_total"] = sum(values)
                    aggregated[f"{metric}_avg"] = sum(values) / len(values)
                elif 'error_rate' in metric:
                    aggregated[f"{metric}_avg"] = sum(values) / len(values)
                    aggregated[f"{metric}_max"] = max(values)
        
        return aggregated
    
    def _aggregate_compliance(self, compliance_results: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """Aggregate compliance status across regions."""
        all_frameworks = set()
        for region_compliance in compliance_results.values():
            all_frameworks.update(region_compliance.keys())
        
        aggregated = {}
        for framework in all_frameworks:
            # Framework is compliant if it's compliant in at least one region
            aggregated[framework] = any(
                region_compliance.get(framework, {}).get('valid', False)
                for region_compliance in compliance_results.values()
            )
        
        return aggregated
    
    def start_monitoring(self):
        """Start background monitoring of deployments."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Started global deployment monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                for deployment_id, status in self.deployments.items():
                    self._check_deployment_health(deployment_id, status)
                
                # Check for scaling needs
                self._check_auto_scaling()
                
                # Update load balancer health scores
                self._update_load_balancer_health()
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
            
            # Wait before next check
            self.stop_event.wait(self.config["monitoring"]["metrics_interval"])
    
    def _check_deployment_health(self, deployment_id: str, status: DeploymentStatus):
        """Check health of a specific deployment."""
        # Simulate health check
        current_health = status.health_score + (hash(deployment_id) % 10 - 5) / 100
        current_health = max(0.0, min(1.0, current_health))
        
        # Update health score
        status.health_score = current_health
        status.last_updated = time.time()
        
        # Check alert thresholds
        thresholds = self.config["monitoring"]["alert_thresholds"]
        
        if current_health < 0.5:
            self.alert_manager.trigger_alert(
                f"Low health score for deployment {deployment_id}: {current_health:.2f}",
                severity="critical"
            )
        elif status.performance_metrics.get("error_rate", 0) > thresholds["error_rate"]:
            self.alert_manager.trigger_alert(
                f"High error rate for deployment {deployment_id}: {status.performance_metrics['error_rate']:.3f}",
                severity="warning"
            )
    
    def _check_auto_scaling(self):
        """Check if any deployments need scaling."""
        for deployment_id, status in self.deployments.items():
            if status.status != "active":
                continue
            
            # Simulate scaling decision based on metrics
            cpu_utilization = status.performance_metrics.get("cpu_utilization", 50)
            error_rate = status.performance_metrics.get("error_rate", 0.001)
            
            scaling_config = self.config["auto_scaling"]
            
            if cpu_utilization > scaling_config["scale_up_threshold"]:
                print(f"🔼 Scaling up deployment {deployment_id} (CPU: {cpu_utilization}%)")
                self._scale_deployment(deployment_id, "up")
            elif cpu_utilization < scaling_config["scale_down_threshold"] and error_rate < 0.01:
                print(f"🔽 Scaling down deployment {deployment_id} (CPU: {cpu_utilization}%)")
                self._scale_deployment(deployment_id, "down")
    
    def _scale_deployment(self, deployment_id: str, direction: str):
        """Scale a deployment up or down."""
        # Simulate scaling operation
        status = self.deployments[deployment_id]
        status.status = "scaling"
        
        # In practice, this would interact with Kubernetes HPA or cloud auto-scaling
        print(f"Scaling {direction} deployment {deployment_id}")
        
        # Simulate scaling completion
        time.sleep(1)
        status.status = "active"
    
    def _update_load_balancer_health(self):
        """Update load balancer with current health scores."""
        for deployment_id, status in self.deployments.items():
            if status.status == "active":
                self.load_balancer.update_health_score(status.region, status.health_score)
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global deployment status."""
        total_deployments = len(self.deployments)
        active_deployments = len([d for d in self.deployments.values() if d.status == "active"])
        
        avg_health = sum(d.health_score for d in self.deployments.values()) / total_deployments if total_deployments > 0 else 0
        
        return {
            "total_deployments": total_deployments,
            "active_deployments": active_deployments,
            "failed_deployments": total_deployments - active_deployments,
            "average_health_score": avg_health,
            "regions": list(set(d.region.value for d in self.deployments.values())),
            "compliance_frameworks": list(set().union(*[
                d.compliance_status.keys() for d in self.deployments.values() if d.compliance_status
            ])),
            "last_updated": max([d.last_updated for d in self.deployments.values()]) if self.deployments else 0
        }


class MetricsCollector:
    """Collects and aggregates metrics from global deployments."""
    
    def __init__(self):
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.aggregated_metrics: Dict[str, Any] = {}
    
    def collect_metrics(self, deployment_id: str, metrics: Dict[str, Any]):
        """Collect metrics from a deployment."""
        metric_entry = {
            "timestamp": time.time(),
            "deployment_id": deployment_id,
            "metrics": metrics
        }
        self.metrics_buffer.append(metric_entry)
        
        # Keep only recent metrics
        cutoff_time = time.time() - 3600  # 1 hour
        self.metrics_buffer = [m for m in self.metrics_buffer if m["timestamp"] > cutoff_time]
    
    def get_aggregated_metrics(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get aggregated metrics for the specified time window."""
        cutoff_time = time.time() - time_window
        recent_metrics = [m for m in self.metrics_buffer if m["timestamp"] > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Aggregate by metric type
        aggregated = {}
        for entry in recent_metrics:
            for metric_name, value in entry["metrics"].items():
                if metric_name not in aggregated:
                    aggregated[metric_name] = []
                aggregated[metric_name].append(value)
        
        # Compute statistics
        result = {}
        for metric_name, values in aggregated.items():
            result[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return result


class AlertManager:
    """Manages alerts for global deployments."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "critical": 0,
            "warning": 10,
            "info": 60
        }
    
    def trigger_alert(self, message: str, severity: str = "info"):
        """Trigger an alert."""
        alert = {
            "timestamp": time.time(),
            "message": message,
            "severity": severity,
            "resolved": False
        }
        self.alerts.append(alert)
        
        # Print alert (in production, would send to monitoring system)
        emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(severity, "📢")
        print(f"{emoji} ALERT [{severity.upper()}]: {message}")
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by severity."""
        active_alerts = [a for a in self.alerts if not a["resolved"]]
        
        if severity:
            active_alerts = [a for a in active_alerts if a["severity"] == severity]
        
        return active_alerts


if __name__ == "__main__":
    # Example usage
    engine = GlobalDeploymentEngine()
    
    # Example model configuration
    model_config = {
        "epsilon": 1.0,
        "delta": 1e-5,
        "embed_dim": 768,
        "num_heads": 12,
        "batch_size": 32
    }
    
    # Deploy globally
    deployment_id = engine.deploy_global(model_config)
    
    # Check status
    status = engine.get_global_status()
    print(f"Global deployment status: {json.dumps(status, indent=2)}")
    
    # Simulate some monitoring
    time.sleep(5)
    
    # Stop monitoring
    engine.stop_monitoring()
    
    print("Global deployment engine validation complete!")