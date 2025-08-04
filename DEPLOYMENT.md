# DP-Flash-Attention Deployment Guide

## 🚀 Deployment Options

### Local Development
```bash
# Install in development mode
pip install -e .

# Run with development configuration
python -m dp_flash_attention --config config/development.json
```

### Docker Deployment
```bash
# Build production image
docker build -f Dockerfile.prod --target production -t dp-flash-attention:latest .

# Run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose logs -f dp-flash-attention
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/kubernetes.yaml

# Check deployment status
kubectl get pods -n dp-flash-attention

# View logs
kubectl logs -f deployment/dp-flash-attention -n dp-flash-attention
```

## 🔧 Configuration

### Environment Variables
- `DP_FLASH_EPSILON`: Privacy epsilon parameter (default: 1.0)
- `DP_FLASH_DELTA`: Privacy delta parameter (default: 1e-5)
- `DP_FLASH_DEVICE`: Compute device (default: auto-detect)
- `DP_FLASH_LOG_LEVEL`: Logging level (default: INFO)
- `ENABLE_MONITORING`: Enable Prometheus metrics (default: false)
- `ENABLE_AUTOSCALING`: Enable auto-scaling (default: false)
- `MAX_WORKERS`: Maximum worker threads (default: 4)

### Configuration Files
- `config/production.json`: Production configuration
- `config/development.json`: Development configuration
- `config/testing.json`: Testing configuration

## 📊 Monitoring & Observability

### Prometheus Metrics
The application exposes metrics on port 8001:
- Privacy budget consumption
- Performance metrics
- System resource usage
- Error rates and latencies

### Grafana Dashboards
Pre-configured dashboards available in `monitoring/grafana/dashboards/`:
- Privacy Dashboard: Track epsilon consumption and privacy guarantees
- Performance Dashboard: Monitor latency, throughput, and resource usage
- System Dashboard: Overall system health and alerts

### Health Checks
- `/health`: Basic health check
- `/ready`: Readiness probe for Kubernetes
- `/metrics`: Prometheus metrics endpoint

## 🔒 Security Considerations

### Production Checklist
- [ ] Use HTTPS/TLS encryption
- [ ] Enable secure RNG (`use_secure_rng: true`)
- [ ] Configure proper firewall rules
- [ ] Set up monitoring and alerting
- [ ] Regular security updates
- [ ] Backup configuration and logs
- [ ] Enable audit logging
- [ ] Use non-root container user
- [ ] Implement rate limiting
- [ ] Network policies in Kubernetes

### Privacy Configuration
```json
{
  "privacy": {
    "epsilon": 1.0,           // Strong privacy: < 1.0
    "delta": 1e-5,           // Recommended: < 1e-5
    "max_grad_norm": 1.0,    // Gradient clipping bound
    "secure_rng": true       // Always true in production
  },
  "security": {
    "strict_mode": true,     // Enable for maximum security
    "validate_inputs": true, // Always validate inputs
    "privacy_audit": true    // Enable privacy auditing
  }
}
```

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check container GPU access
docker run --gpus all nvidia/cuda:12.0-runtime-ubuntu22.04 nvidia-smi
```

#### Memory Issues
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Check system memory
free -h

# Reduce batch size or sequence length in configuration
```

#### Privacy Parameter Errors
```bash
# Validate privacy parameters
python -c "
from dp_flash_attention.validation import validate_privacy_parameters_comprehensive
validate_privacy_parameters_comprehensive(1.0, 1e-5, 1.0)
"
```

### Performance Optimization

#### GPU Optimization
- Use appropriate CUDA version (12.0+)
- Enable Tensor Cores for supported models
- Optimize batch sizes for GPU memory
- Use mixed precision (float16/bfloat16)

#### CPU Optimization
- Set appropriate number of workers
- Enable auto-scaling for variable workloads
- Use memory pooling for large workloads

### Scaling Guidelines

#### Horizontal Scaling
- Use Kubernetes HPA for automatic scaling
- Configure appropriate CPU/memory thresholds
- Monitor privacy budget across replicas

#### Vertical Scaling
- Increase GPU memory for larger models
- Scale CPU cores for preprocessing
- Adjust worker threads based on workload

## 📋 Production Deployment Checklist

### Pre-deployment
- [ ] Review and test configuration
- [ ] Run comprehensive diagnostics
- [ ] Benchmark performance
- [ ] Security audit
- [ ] Backup current deployment

### Deployment
- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Monitor metrics and logs
- [ ] Verify privacy guarantees
- [ ] Performance validation

### Post-deployment
- [ ] Monitor system health
- [ ] Check error rates
- [ ] Validate privacy metrics
- [ ] Set up alerting
- [ ] Document any issues

## 🚨 Monitoring & Alerting

### Critical Alerts
- Privacy budget exhaustion
- GPU out of memory
- High error rates (>5%)
- Response time degradation
- Security audit failures

### Dashboard URLs
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Application: http://localhost:8000

### Log Locations
- Application logs: `/app/logs/`
- Nginx logs: `/var/log/nginx/`
- System logs: `/var/log/`

## 📞 Support

### Diagnostics
```bash
# Run comprehensive diagnostics
docker exec dp-flash-attention python -m dp_flash_attention diagnostics

# Export diagnostic report
docker cp dp-flash-attention:/app/logs/diagnostic_report.json ./
```

### Performance Benchmarks
```bash
# Run performance benchmarks
docker exec dp-flash-attention python -m dp_flash_attention benchmark

# View benchmark results
docker exec dp-flash-attention cat /app/logs/benchmark_results.json
```

### Log Analysis
```bash
# View application logs
docker logs dp-flash-attention

# Search for errors
docker logs dp-flash-attention 2>&1 | grep ERROR

# Monitor real-time logs
docker logs -f dp-flash-attention
```