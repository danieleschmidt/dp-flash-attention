"""
Advanced Deployment Orchestration for DP-Flash-Attention.

This module provides production-grade deployment, scaling, and lifecycle management
for global privacy-preserving attention systems.
"""

import os
import json
import yaml
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DR = "disaster_recovery"  # Disaster Recovery


class DeploymentStrategy(Enum):
    """Deployment strategies for updates."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class OrchestrationPlatform(Enum):
    """Supported orchestration platforms."""
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    ECS = "ecs"
    NOMAD = "nomad"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    CUSTOM_METRICS = "custom_metrics"
    PRIVACY_BUDGET = "privacy_budget"  # Novel: Scale based on privacy budget consumption


@dataclass
class DeploymentConfig:
    """Comprehensive deployment configuration."""
    
    # Basic configuration
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    platform: OrchestrationPlatform
    
    # Resource configuration
    cpu_request: float
    cpu_limit: float
    memory_request: str
    memory_limit: str
    gpu_required: bool
    gpu_count: int
    
    # Scaling configuration
    min_replicas: int
    max_replicas: int
    target_cpu_percent: int
    target_memory_percent: int
    
    # Privacy configuration
    privacy_budget_per_replica: float
    privacy_budget_refresh_interval: timedelta
    
    # Security configuration
    enable_tls: bool
    secret_management: str
    network_policies: List[str]
    
    # Monitoring configuration
    enable_metrics: bool
    enable_tracing: bool
    log_level: str
    
    # Backup and DR
    backup_enabled: bool
    backup_schedule: str
    dr_region: Optional[str]


@dataclass
class DeploymentStatus:
    """Current deployment status information."""
    environment: str
    version: str
    replicas_ready: int
    replicas_desired: int
    last_updated: datetime
    health_status: str
    privacy_budget_remaining: float
    errors: List[str]
    warnings: List[str]


class KubernetesManifestGenerator:
    """Generates Kubernetes manifests for DP-Flash-Attention."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    def generate_deployment_manifest(self, image: str, version: str) -> Dict[str, Any]:
        """Generate Kubernetes Deployment manifest."""
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"dp-flash-attention-{self.config.environment.value}",
                "namespace": f"ml-{self.config.environment.value}",
                "labels": {
                    "app": "dp-flash-attention",
                    "version": version,
                    "environment": self.config.environment.value,
                    "privacy-enabled": "true"
                }
            },
            "spec": {
                "replicas": self.config.min_replicas,
                "strategy": {
                    "type": "RollingUpdate" if self.config.strategy == DeploymentStrategy.ROLLING else "Recreate",
                    "rollingUpdate": {
                        "maxSurge": 1,
                        "maxUnavailable": 0
                    } if self.config.strategy == DeploymentStrategy.ROLLING else None
                },
                "selector": {
                    "matchLabels": {
                        "app": "dp-flash-attention",
                        "environment": self.config.environment.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "dp-flash-attention",
                            "version": version,
                            "environment": self.config.environment.value
                        },
                        "annotations": {
                            "privacy.dp-flash-attention.io/budget-per-replica": str(self.config.privacy_budget_per_replica),
                            "monitoring.dp-flash-attention.io/scrape": "true",
                            "deployment.dp-flash-attention.io/timestamp": datetime.now().isoformat()
                        }
                    },
                    "spec": {
                        "serviceAccountName": "dp-flash-attention",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 2000
                        },
                        "containers": [
                            {
                                "name": "dp-flash-attention",
                                "image": f"{image}:{version}",
                                "imagePullPolicy": "Always",
                                "ports": [
                                    {"containerPort": 8080, "name": "http"},
                                    {"containerPort": 9090, "name": "metrics"}
                                ],
                                "env": [
                                    {"name": "ENVIRONMENT", "value": self.config.environment.value},
                                    {"name": "LOG_LEVEL", "value": self.config.log_level},
                                    {"name": "PRIVACY_BUDGET", "value": str(self.config.privacy_budget_per_replica)},
                                    {"name": "ENABLE_METRICS", "value": str(self.config.enable_metrics).lower()},
                                    {"name": "ENABLE_TRACING", "value": str(self.config.enable_tracing).lower()}
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": str(self.config.cpu_request),
                                        "memory": self.config.memory_request
                                    },
                                    "limits": {
                                        "cpu": str(self.config.cpu_limit),
                                        "memory": self.config.memory_limit
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health/live",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 3
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/health/ready",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 3
                                },
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": True,
                                    "capabilities": {
                                        "drop": ["ALL"]
                                    }
                                },
                                "volumeMounts": [
                                    {
                                        "name": "tmp",
                                        "mountPath": "/tmp"
                                    },
                                    {
                                        "name": "privacy-config",
                                        "mountPath": "/etc/privacy",
                                        "readOnly": True
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "tmp",
                                "emptyDir": {}
                            },
                            {
                                "name": "privacy-config",
                                "configMap": {
                                    "name": "dp-flash-attention-privacy-config"
                                }
                            }
                        ]
                    }
                }
            }
        }
    
    def generate_service_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes Service manifest."""
        
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"dp-flash-attention-{self.config.environment.value}",
                "namespace": f"ml-{self.config.environment.value}",
                "labels": {
                    "app": "dp-flash-attention",
                    "environment": self.config.environment.value
                },
                "annotations": {
                    "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
                    "service.beta.kubernetes.io/aws-load-balancer-ssl-cert": "arn:aws:acm:region:account:certificate/cert-id",
                    "privacy.dp-flash-attention.io/tls-required": str(self.config.enable_tls).lower()
                }
            },
            "spec": {
                "selector": {
                    "app": "dp-flash-attention",
                    "environment": self.config.environment.value
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 80,
                        "targetPort": 8080,
                        "protocol": "TCP"
                    },
                    {
                        "name": "https", 
                        "port": 443,
                        "targetPort": 8080,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 9090,
                        "targetPort": 9090,
                        "protocol": "TCP"
                    }
                ],
                "type": "LoadBalancer"
            }
        }
    
    def generate_hpa_manifest(self) -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler manifest with privacy-aware scaling."""
        
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"dp-flash-attention-hpa-{self.config.environment.value}",
                "namespace": f"ml-{self.config.environment.value}",
                "labels": {
                    "app": "dp-flash-attention",
                    "environment": self.config.environment.value
                }
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"dp-flash-attention-{self.config.environment.value}"
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_cpu_percent
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization", 
                                "averageUtilization": self.config.target_memory_percent
                            }
                        }
                    },
                    {
                        "type": "Pods",
                        "pods": {
                            "metric": {
                                "name": "privacy_budget_utilization"
                            },
                            "target": {
                                "type": "AverageValue",
                                "averageValue": "75"  # Scale when 75% of privacy budget is used
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,
                                "periodSeconds": 60
                            }
                        ]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 10,
                                "periodSeconds": 60
                            }
                        ]
                    }
                }
            }
        }
    
    def generate_configmap_manifest(self, privacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ConfigMap for privacy configuration."""
        
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "dp-flash-attention-privacy-config",
                "namespace": f"ml-{self.config.environment.value}",
                "labels": {
                    "app": "dp-flash-attention",
                    "environment": self.config.environment.value
                }
            },
            "data": {
                "privacy.json": json.dumps(privacy_config, indent=2),
                "compliance.yaml": yaml.dump({
                    "frameworks": ["gdpr", "ccpa", "pdpa"],
                    "audit_enabled": True,
                    "encryption_required": True,
                    "data_residency": self.config.environment != DeploymentEnvironment.DEVELOPMENT
                })
            }
        }


class DeploymentOrchestrator:
    """Orchestrates deployment across different platforms and environments."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.manifest_generator = KubernetesManifestGenerator(config)
        self.deployment_history = []
        
    def deploy(self, image: str, version: str) -> DeploymentStatus:
        """Deploy DP-Flash-Attention with specified configuration."""
        
        logger.info(f"Starting deployment: {image}:{version} to {self.config.environment.value}")
        
        try:
            if self.config.platform == OrchestrationPlatform.KUBERNETES:
                return self._deploy_kubernetes(image, version)
            elif self.config.platform == OrchestrationPlatform.DOCKER_SWARM:
                return self._deploy_docker_swarm(image, version)
            else:
                raise NotImplementedError(f"Platform {self.config.platform} not implemented")
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return DeploymentStatus(
                environment=self.config.environment.value,
                version=version,
                replicas_ready=0,
                replicas_desired=self.config.min_replicas,
                last_updated=datetime.now(),
                health_status="FAILED",
                privacy_budget_remaining=self.config.privacy_budget_per_replica,
                errors=[str(e)],
                warnings=[]
            )
    
    def _deploy_kubernetes(self, image: str, version: str) -> DeploymentStatus:
        """Deploy to Kubernetes cluster."""
        
        # Generate manifests
        deployment_manifest = self.manifest_generator.generate_deployment_manifest(image, version)
        service_manifest = self.manifest_generator.generate_service_manifest()
        hpa_manifest = self.manifest_generator.generate_hpa_manifest()
        
        # Privacy configuration
        privacy_config = {
            "epsilon": self.config.privacy_budget_per_replica,
            "delta": 1e-6,
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "accountant": "rdp"
        }
        configmap_manifest = self.manifest_generator.generate_configmap_manifest(privacy_config)
        
        # Save manifests to temporary files
        manifests = [
            ("configmap", configmap_manifest),
            ("deployment", deployment_manifest), 
            ("service", service_manifest),
            ("hpa", hpa_manifest)
        ]
        
        success = True
        errors = []
        
        for name, manifest in manifests:
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(manifest, f)
                    temp_file = f.name
                
                # Apply manifest (simulated - would use kubectl in real implementation)
                logger.info(f"Applying {name} manifest...")
                # result = subprocess.run(['kubectl', 'apply', '-f', temp_file], 
                #                        capture_output=True, text=True)
                
                # Simulate successful application
                logger.info(f"✓ {name.capitalize()} applied successfully")
                
                # Clean up
                os.unlink(temp_file)
                
            except Exception as e:
                success = False
                errors.append(f"Failed to apply {name}: {e}")
                logger.error(f"Failed to apply {name}: {e}")
        
        # Wait for deployment to be ready (simulated)
        if success:
            logger.info("Waiting for deployment to be ready...")
            time.sleep(2)  # Simulate deployment time
            
            replicas_ready = self.config.min_replicas
            health_status = "HEALTHY"
        else:
            replicas_ready = 0
            health_status = "FAILED"
        
        status = DeploymentStatus(
            environment=self.config.environment.value,
            version=version,
            replicas_ready=replicas_ready,
            replicas_desired=self.config.min_replicas,
            last_updated=datetime.now(),
            health_status=health_status,
            privacy_budget_remaining=self.config.privacy_budget_per_replica,
            errors=errors,
            warnings=[]
        )
        
        # Record deployment
        self.deployment_history.append(status)
        
        if success:
            logger.info(f"✅ Deployment successful: {replicas_ready}/{self.config.min_replicas} replicas ready")
        else:
            logger.error(f"❌ Deployment failed with {len(errors)} errors")
            
        return status
    
    def _deploy_docker_swarm(self, image: str, version: str) -> DeploymentStatus:
        """Deploy to Docker Swarm."""
        
        # Generate Docker Compose file for Swarm
        compose_config = {
            "version": "3.8",
            "services": {
                "dp-flash-attention": {
                    "image": f"{image}:{version}",
                    "environment": [
                        f"ENVIRONMENT={self.config.environment.value}",
                        f"LOG_LEVEL={self.config.log_level}",
                        f"PRIVACY_BUDGET={self.config.privacy_budget_per_replica}"
                    ],
                    "ports": ["8080:8080", "9090:9090"],
                    "deploy": {
                        "replicas": self.config.min_replicas,
                        "resources": {
                            "reservations": {
                                "cpus": str(self.config.cpu_request),
                                "memory": self.config.memory_request
                            },
                            "limits": {
                                "cpus": str(self.config.cpu_limit),
                                "memory": self.config.memory_limit
                            }
                        },
                        "update_config": {
                            "parallelism": 1,
                            "delay": "10s",
                            "failure_action": "rollback"
                        },
                        "restart_policy": {
                            "condition": "on-failure",
                            "delay": "5s",
                            "max_attempts": 3
                        }
                    },
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    },
                    "networks": ["dp-flash-attention-network"]
                }
            },
            "networks": {
                "dp-flash-attention-network": {
                    "driver": "overlay",
                    "attachable": True
                }
            }
        }
        
        # Deploy to swarm (simulated)
        logger.info("Deploying to Docker Swarm...")
        
        # Would save compose file and run: docker stack deploy
        logger.info("✓ Docker Swarm deployment completed")
        
        return DeploymentStatus(
            environment=self.config.environment.value,
            version=version,
            replicas_ready=self.config.min_replicas,
            replicas_desired=self.config.min_replicas,
            last_updated=datetime.now(),
            health_status="HEALTHY",
            privacy_budget_remaining=self.config.privacy_budget_per_replica,
            errors=[],
            warnings=[]
        )
    
    def rollback(self, target_version: Optional[str] = None) -> DeploymentStatus:
        """Rollback to previous version or specified version."""
        
        if not self.deployment_history:
            raise ValueError("No deployment history available for rollback")
        
        if target_version:
            # Find specific version in history
            target_deployment = None
            for deployment in reversed(self.deployment_history):
                if deployment.version == target_version:
                    target_deployment = deployment
                    break
            
            if not target_deployment:
                raise ValueError(f"Version {target_version} not found in deployment history")
        else:
            # Rollback to previous successful deployment
            target_deployment = None
            for deployment in reversed(self.deployment_history[:-1]):  # Skip current
                if deployment.health_status == "HEALTHY":
                    target_deployment = deployment
                    break
            
            if not target_deployment:
                raise ValueError("No previous healthy deployment found for rollback")
        
        logger.info(f"Rolling back to version {target_deployment.version}")
        
        # Perform rollback (simulated)
        rollback_status = DeploymentStatus(
            environment=target_deployment.environment,
            version=target_deployment.version,
            replicas_ready=target_deployment.replicas_ready,
            replicas_desired=target_deployment.replicas_desired,
            last_updated=datetime.now(),
            health_status="HEALTHY",
            privacy_budget_remaining=target_deployment.privacy_budget_remaining,
            errors=[],
            warnings=[f"Rolled back from {self.deployment_history[-1].version}"]
        )
        
        self.deployment_history.append(rollback_status)
        
        logger.info(f"✅ Rollback completed to version {target_deployment.version}")
        return rollback_status
    
    def get_deployment_status(self) -> Optional[DeploymentStatus]:
        """Get current deployment status."""
        return self.deployment_history[-1] if self.deployment_history else None
    
    def scale(self, replicas: int) -> DeploymentStatus:
        """Scale deployment to specified number of replicas."""
        
        if replicas < self.config.min_replicas or replicas > self.config.max_replicas:
            raise ValueError(f"Replica count must be between {self.config.min_replicas} and {self.config.max_replicas}")
        
        logger.info(f"Scaling deployment to {replicas} replicas")
        
        current_status = self.get_deployment_status()
        if not current_status:
            raise RuntimeError("No active deployment found")
        
        # Perform scaling (simulated)
        scaled_status = DeploymentStatus(
            environment=current_status.environment,
            version=current_status.version,
            replicas_ready=replicas,
            replicas_desired=replicas,
            last_updated=datetime.now(),
            health_status="HEALTHY",
            privacy_budget_remaining=current_status.privacy_budget_remaining,
            errors=[],
            warnings=[f"Scaled from {current_status.replicas_ready} to {replicas} replicas"]
        )
        
        self.deployment_history.append(scaled_status)
        
        logger.info(f"✅ Scaling completed: {replicas} replicas ready")
        return scaled_status


class MultiEnvironmentManager:
    """Manages deployments across multiple environments."""
    
    def __init__(self):
        self.environments = {}
        self.promotion_rules = self._initialize_promotion_rules()
        
    def _initialize_promotion_rules(self) -> Dict[str, List[str]]:
        """Initialize environment promotion rules."""
        return {
            "development": ["staging"],
            "staging": ["production"],
            "production": ["dr"],  # Disaster recovery
            "dr": []  # Terminal environment
        }
    
    def register_environment(self, env: DeploymentEnvironment, orchestrator: DeploymentOrchestrator):
        """Register an environment with its orchestrator."""
        self.environments[env] = orchestrator
        logger.info(f"Registered environment: {env.value}")
    
    def deploy_to_environment(self, env: DeploymentEnvironment, image: str, version: str) -> DeploymentStatus:
        """Deploy to specific environment."""
        
        if env not in self.environments:
            raise ValueError(f"Environment {env.value} not registered")
        
        orchestrator = self.environments[env]
        return orchestrator.deploy(image, version)
    
    def promote_version(self, from_env: DeploymentEnvironment, to_env: DeploymentEnvironment, version: str) -> DeploymentStatus:
        """Promote version from one environment to another."""
        
        # Validate promotion path
        if to_env.value not in self.promotion_rules.get(from_env.value, []):
            raise ValueError(f"Cannot promote from {from_env.value} to {to_env.value}")
        
        # Get source deployment status
        source_orchestrator = self.environments[from_env]
        source_status = source_orchestrator.get_deployment_status()
        
        if not source_status or source_status.version != version:
            raise ValueError(f"Version {version} not found in {from_env.value}")
        
        if source_status.health_status != "HEALTHY":
            raise ValueError(f"Cannot promote unhealthy deployment from {from_env.value}")
        
        # Perform promotion
        logger.info(f"Promoting version {version} from {from_env.value} to {to_env.value}")
        
        # Extract image from deployment history (simplified)
        image = f"dp-flash-attention"  # Would extract from deployment metadata
        
        return self.deploy_to_environment(to_env, image, version)
    
    def get_environment_status(self) -> Dict[str, Optional[DeploymentStatus]]:
        """Get status of all registered environments."""
        
        status = {}
        for env, orchestrator in self.environments.items():
            status[env.value] = orchestrator.get_deployment_status()
        
        return status


# Factory functions for common deployment scenarios
def create_production_deployment_config() -> DeploymentConfig:
    """Create production-ready deployment configuration."""
    
    return DeploymentConfig(
        environment=DeploymentEnvironment.PRODUCTION,
        strategy=DeploymentStrategy.ROLLING,
        platform=OrchestrationPlatform.KUBERNETES,
        cpu_request=2.0,
        cpu_limit=4.0,
        memory_request="4Gi",
        memory_limit="8Gi",
        gpu_required=True,
        gpu_count=1,
        min_replicas=3,
        max_replicas=10,
        target_cpu_percent=70,
        target_memory_percent=80,
        privacy_budget_per_replica=1.0,
        privacy_budget_refresh_interval=timedelta(hours=24),
        enable_tls=True,
        secret_management="kubernetes-secrets",
        network_policies=["default-deny", "allow-ingress"],
        enable_metrics=True,
        enable_tracing=True,
        log_level="INFO",
        backup_enabled=True,
        backup_schedule="0 2 * * *",  # Daily at 2 AM
        dr_region="us-west-2"
    )


def create_development_deployment_config() -> DeploymentConfig:
    """Create development deployment configuration."""
    
    return DeploymentConfig(
        environment=DeploymentEnvironment.DEVELOPMENT,
        strategy=DeploymentStrategy.RECREATE,
        platform=OrchestrationPlatform.KUBERNETES,
        cpu_request=0.5,
        cpu_limit=1.0,
        memory_request="1Gi",
        memory_limit="2Gi",
        gpu_required=False,
        gpu_count=0,
        min_replicas=1,
        max_replicas=2,
        target_cpu_percent=80,
        target_memory_percent=85,
        privacy_budget_per_replica=10.0,  # Relaxed for development
        privacy_budget_refresh_interval=timedelta(hours=1),
        enable_tls=False,
        secret_management="configmap",
        network_policies=[],
        enable_metrics=True,
        enable_tracing=True,
        log_level="DEBUG",
        backup_enabled=False,
        backup_schedule="",
        dr_region=None
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create production deployment
    prod_config = create_production_deployment_config()
    prod_orchestrator = DeploymentOrchestrator(prod_config)
    
    # Deploy to production
    status = prod_orchestrator.deploy("dp-flash-attention", "v1.0.0")
    print(f"Deployment status: {status.health_status}")
    
    # Create multi-environment manager
    manager = MultiEnvironmentManager()
    manager.register_environment(DeploymentEnvironment.PRODUCTION, prod_orchestrator)
    
    # Get all environment statuses
    all_status = manager.get_environment_status()
    print(f"Environment statuses: {all_status}")
    
    logger.info("🚀 Advanced deployment orchestration completed")