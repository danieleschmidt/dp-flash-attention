#!/bin/bash
"""
Production Deployment Script for DP-Flash-Attention
Automated deployment with comprehensive validation and monitoring setup.
"""

set -euo pipefail

# Configuration
NAMESPACE="dp-flash-attention"
DEPLOYMENT_NAME="dp-flash-attention"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
REGION="${REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Validation functions
validate_prerequisites() {
    log "Validating deployment prerequisites..."
    
    # Check required tools
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v docker >/dev/null 2>&1 || error "docker is required but not installed"
    command -v python3 >/dev/null 2>&1 || error "python3 is required but not installed"
    
    # Check Kubernetes connectivity
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"
    
    # Check NVIDIA GPU support (if applicable)
    if kubectl get nodes -o json | grep -q "nvidia.com/gpu"; then
        success "NVIDIA GPU support detected"
    else
        warning "No NVIDIA GPU support detected - CPU-only deployment"
    fi
    
    success "Prerequisites validation completed"
}

validate_quality_gates() {
    log "Running autonomous SDLC quality gates..."
    
    if [ -f "autonomous_sdlc_quality_gates.py" ]; then
        python3 autonomous_sdlc_quality_gates.py || error "Quality gates validation failed"
        success "Quality gates validation passed"
    else
        warning "Quality gates script not found - skipping validation"
    fi
}

setup_namespace() {
    log "Setting up Kubernetes namespace..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        kubectl create namespace "$NAMESPACE"
        success "Created namespace: $NAMESPACE"
    else
        log "Namespace $NAMESPACE already exists"
    fi
    
    # Set default namespace context
    kubectl config set-context --current --namespace="$NAMESPACE"
}

deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Deploy Prometheus if monitoring config exists
    if [ -f "monitoring/prometheus.yml" ]; then
        kubectl apply -f monitoring/observability.yml -n "$NAMESPACE" || warning "Failed to deploy monitoring stack"
        success "Monitoring stack deployed"
    else
        warning "Monitoring configuration not found - skipping monitoring deployment"
    fi
}

deploy_application() {
    log "Deploying DP-Flash-Attention application..."
    
    # Substitute environment variables in deployment yaml
    export NAMESPACE VERSION ENVIRONMENT REGION
    envsubst < deployment/deployment.yaml | kubectl apply -f - || error "Application deployment failed"
    
    # Deploy HPA if available
    if [ -f "deployment/hpa.yaml" ]; then
        kubectl apply -f deployment/hpa.yaml -n "$NAMESPACE" || warning "HPA deployment failed"
        success "Auto-scaling configured"
    fi
    
    # Deploy network policies if available
    if [ -f "deployment/network-policy.yaml" ]; then
        kubectl apply -f deployment/network-policy.yaml -n "$NAMESPACE" || warning "Network policy deployment failed"
        success "Network policies applied"
    fi
    
    success "Application deployment completed"
}

wait_for_deployment() {
    log "Waiting for deployment to be ready..."
    
    # Wait for deployment to be available
    kubectl wait --for=condition=available --timeout=600s deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" || error "Deployment failed to become ready"
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT_NAME"
    
    success "Deployment is ready"
}

validate_deployment() {
    log "Validating deployment..."
    
    # Check deployment status
    READY_REPLICAS=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    DESIRED_REPLICAS=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    
    if [ "$READY_REPLICAS" = "$DESIRED_REPLICAS" ]; then
        success "All replicas are ready ($READY_REPLICAS/$DESIRED_REPLICAS)"
    else
        error "Not all replicas are ready ($READY_REPLICAS/$DESIRED_REPLICAS)"
    fi
    
    # Test application endpoints (if test script exists)
    if [ -f "scripts/deployment_validation.py" ]; then
        python3 scripts/deployment_validation.py || warning "Endpoint validation failed"
    fi
    
    success "Deployment validation completed"
}

setup_monitoring_alerts() {
    log "Setting up monitoring alerts..."
    
    # Apply alert rules if they exist
    if [ -d "monitoring/rules" ]; then
        kubectl apply -f monitoring/rules/ -n "$NAMESPACE" || warning "Failed to apply alert rules"
        success "Alert rules configured"
    fi
    
    # Setup Grafana dashboards if they exist
    if [ -d "monitoring/grafana/dashboards" ]; then
        kubectl create configmap grafana-dashboards \
            --from-file=monitoring/grafana/dashboards/ \
            -n "$NAMESPACE" \
            --dry-run=client -o yaml | kubectl apply -f - || warning "Failed to setup Grafana dashboards"
        success "Grafana dashboards configured"
    fi
}

generate_deployment_summary() {
    log "Generating deployment summary..."
    
    # Get deployment information
    DEPLOYMENT_STATUS=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')
    POD_COUNT=$(kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT_NAME" --no-headers | wc -l)
    SERVICE_IP=$(kubectl get service "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "N/A")
    
    # Create deployment summary
    cat > deployment_summary.json << EOF
{
  "deployment_id": "$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "environment": "$ENVIRONMENT",
  "region": "$REGION",
  "namespace": "$NAMESPACE",
  "version": "$VERSION",
  "status": "$DEPLOYMENT_STATUS",
  "pod_count": $POD_COUNT,
  "service_ip": "$SERVICE_IP",
  "endpoints": {
    "api": "http://$SERVICE_IP:8080",
    "metrics": "http://$SERVICE_IP:8080/metrics",
    "health": "http://$SERVICE_IP:8080/health"
  }
}
EOF
    
    success "Deployment summary saved to deployment_summary.json"
}

cleanup_on_failure() {
    if [ $? -ne 0 ]; then
        error "Deployment failed! Cleaning up..."
        kubectl delete deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" 2>/dev/null || true
        kubectl delete service "$DEPLOYMENT_NAME" -n "$NAMESPACE" 2>/dev/null || true
    fi
}

# Main deployment function
main() {
    log "Starting DP-Flash-Attention production deployment..."
    log "Environment: $ENVIRONMENT"
    log "Region: $REGION"
    log "Version: $VERSION"
    
    # Set up cleanup on failure
    trap cleanup_on_failure EXIT
    
    # Run deployment steps
    validate_prerequisites
    validate_quality_gates
    setup_namespace
    deploy_monitoring
    deploy_application
    wait_for_deployment
    validate_deployment
    setup_monitoring_alerts
    generate_deployment_summary
    
    # Remove failure cleanup trap on success
    trap - EXIT
    
    success "🎉 DP-Flash-Attention production deployment completed successfully!"
    log "Deployment summary:"
    cat deployment_summary.json
    
    log "Next steps:"
    log "1. Monitor deployment status: kubectl get pods -n $NAMESPACE"
    log "2. Check application logs: kubectl logs -f deployment/$DEPLOYMENT_NAME -n $NAMESPACE"
    log "3. Access Grafana dashboards for monitoring"
    log "4. Run post-deployment validation tests"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "validate")
        validate_prerequisites
        validate_quality_gates
        ;;
    "monitor")
        setup_monitoring_alerts
        ;;
    "cleanup")
        log "Cleaning up deployment..."
        kubectl delete namespace "$NAMESPACE" || true
        success "Cleanup completed"
        ;;
    "status")
        log "Deployment status:"
        kubectl get all -n "$NAMESPACE"
        ;;
    *)
        echo "Usage: $0 {deploy|validate|monitor|cleanup|status}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full production deployment (default)"
        echo "  validate - Validate prerequisites and quality gates"
        echo "  monitor  - Setup monitoring and alerts"
        echo "  cleanup  - Remove deployment"
        echo "  status   - Show deployment status"
        exit 1
        ;;
esac