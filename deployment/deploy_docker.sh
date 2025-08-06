#!/bin/bash
# DP-Flash-Attention Docker Deployment Script

set -e

echo "🚀 Deploying DP-Flash-Attention with Docker..."

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Create necessary directories
mkdir -p logs data config monitoring

# Build the image
echo "🔨 Building Docker image..."
docker build -t dp-flash-attention:latest .

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🔍 Performing health check..."
if curl -f http://localhost:8081/health; then
    echo "✅ Deployment successful!"
    echo "📊 Metrics available at: http://localhost:8081/metrics"
    echo "🏥 Health check at: http://localhost:8081/health"
else
    echo "❌ Health check failed"
    docker-compose logs
    exit 1
fi
