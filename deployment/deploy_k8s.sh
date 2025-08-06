#!/bin/bash
# DP-Flash-Attention Kubernetes Deployment Script

set -e

echo "☸️ Deploying DP-Flash-Attention to Kubernetes..."

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed"
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster"
    exit 1
fi

# Create namespace
kubectl create namespace dp-flash-attention --dry-run=client -o yaml | kubectl apply -f -

# Apply configurations
echo "📄 Applying configurations..."
kubectl apply -f deployment.yaml -n dp-flash-attention
kubectl apply -f hpa.yaml -n dp-flash-attention  
kubectl apply -f network-policy.yaml -n dp-flash-attention

# Wait for rollout
echo "⏳ Waiting for deployment..."
kubectl rollout status deployment/dp-flash-attention -n dp-flash-attention

# Get service status
echo "📊 Service status:"
kubectl get pods -n dp-flash-attention
kubectl get services -n dp-flash-attention

echo "✅ Kubernetes deployment complete!"
