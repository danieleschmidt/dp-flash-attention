#!/bin/bash
set -e

# DP-Flash-Attention Production Entrypoint Script

echo "🚀 Starting DP-Flash-Attention..."
echo "Environment: ${ENVIRONMENT:-production}"
echo "CUDA Visible Devices: ${CUDA_VISIBLE_DEVICES:-all}"

# Function to wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    
    echo "⏳ Waiting for $service_name at $host:$port..."
    while ! nc -z "$host" "$port"; do
        sleep 1
    done
    echo "✅ $service_name is ready"
}

# Function to check GPU availability
check_gpu() {
    echo "🔍 Checking GPU availability..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo "✅ GPU detected"
    else
        echo "⚠️  No GPU detected, using CPU fallback"
    fi
}

# Function to validate configuration
validate_config() {
    echo "🔧 Validating configuration..."
    
    # Check privacy parameters
    if [ -n "$DP_FLASH_EPSILON" ]; then
        python3 -c "
from dp_flash_attention.validation import validate_privacy_parameters_comprehensive
try:
    validate_privacy_parameters_comprehensive(
        float('$DP_FLASH_EPSILON'), 
        float('${DP_FLASH_DELTA:-1e-5}'), 
        float('${DP_FLASH_MAX_GRAD_NORM:-1.0}')
    )
    print('✅ Privacy parameters valid')
except Exception as e:
    print(f'❌ Invalid privacy parameters: {e}')
    exit(1)
"
    fi
    
    # Check system requirements
    python3 -c "
from dp_flash_attention.diagnostics import run_quick_health_check
import json
try:
    report = run_quick_health_check()
    if report['overall_status'] == 'critical':
        print('❌ Critical system issues detected')
        print(json.dumps(report['critical_issues'], indent=2))
        exit(1)
    else:
        print('✅ System health check passed')
except Exception as e:
    print(f'⚠️  Health check failed: {e}')
"
}

# Function to run migrations/setup
setup_environment() {
    echo "🔨 Setting up environment..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/cache /app/tmp
    
    # Set up monitoring if enabled
    if [ "$ENABLE_MONITORING" = "true" ]; then
        echo "📊 Enabling monitoring..."
        export DP_FLASH_PROMETHEUS=true
        export DP_FLASH_TELEMETRY=true
    fi
    
    # Configure logging
    export DP_FLASH_LOG_LEVEL=${DP_FLASH_LOG_LEVEL:-INFO}
    
    echo "✅ Environment setup complete"
}

# Function to start the main application
start_server() {
    echo "🌐 Starting DP-Flash-Attention server..."
    
    # Configure auto-scaling if enabled
    if [ "$ENABLE_AUTOSCALING" = "true" ]; then
        echo "⚖️  Auto-scaling enabled"
        export DP_FLASH_AUTO_SCALE=true
        export DP_FLASH_MAX_WORKERS=${MAX_WORKERS:-4}
        export DP_FLASH_MIN_WORKERS=${MIN_WORKERS:-1}
    fi
    
    # Start the server
    exec python3 -m dp_flash_attention.server \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8000}" \
        --workers "${WORKERS:-1}" \
        --config-file "${CONFIG_FILE:-/app/config/production.json}"
}

# Function to run diagnostic mode
run_diagnostics() {
    echo "🔍 Running comprehensive diagnostics..."
    python3 -c "
from dp_flash_attention.diagnostics import run_comprehensive_diagnostics, export_diagnostic_report
import json

print('Running full system diagnostics...')
report = run_comprehensive_diagnostics()

print(f'Overall Status: {report[\"overall_status\"]}')
print(f'Critical Issues: {len(report[\"critical_issues\"])}')
print(f'Warnings: {len(report[\"warnings\"])}')

# Export report
export_diagnostic_report(report, '/app/logs/diagnostic_report.json')
print('Diagnostic report saved to /app/logs/diagnostic_report.json')

# Print summary
if report['critical_issues']:
    print('\\n❌ Critical Issues:')
    for issue in report['critical_issues']:
        print(f'  - {issue}')

if report['warnings']:
    print('\\n⚠️  Warnings:')
    for warning in report['warnings']:
        print(f'  - {warning}')

if report['recommendations']:
    print('\\n💡 Recommendations:')
    for rec in report['recommendations']:
        print(f'  - {rec}')
"
}

# Function to run benchmarks
run_benchmarks() {
    echo "📈 Running performance benchmarks..."
    python3 -c "
from dp_flash_attention.utils import benchmark_attention_kernel
import json

configs = [
    {'batch_size': 16, 'sequence_length': 512, 'num_heads': 8, 'head_dim': 64},
    {'batch_size': 32, 'sequence_length': 1024, 'num_heads': 12, 'head_dim': 64},
    {'batch_size': 8, 'sequence_length': 2048, 'num_heads': 16, 'head_dim': 128},
]

results = []
for config in configs:
    print(f'Running benchmark: {config}')
    result = benchmark_attention_kernel(**config, num_iterations=10)
    results.append({**config, **result})
    if 'error' not in result:
        print(f'  ✅ {result[\"avg_time_ms\"]:.2f}ms avg, {result[\"throughput_samples_per_sec\"]:.1f} samples/sec')
    else:
        print(f'  ❌ {result[\"error\"]}')

# Save results
with open('/app/logs/benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Benchmark results saved to /app/logs/benchmark_results.json')
"
}

# Main execution logic
case "$1" in
    server)
        check_gpu
        validate_config
        setup_environment
        start_server
        ;;
    
    diagnostics)
        check_gpu
        run_diagnostics
        ;;
    
    benchmark)
        check_gpu
        run_benchmarks
        ;;
    
    test)
        echo "🧪 Running tests..."
        python3 -m pytest tests/ -v --tb=short
        ;;
    
    shell)
        echo "🐚 Starting interactive shell..."
        exec /bin/bash
        ;;
    
    *)
        echo "Usage: $0 {server|diagnostics|benchmark|test|shell}"
        echo ""
        echo "Commands:"
        echo "  server      - Start the DP-Flash-Attention server (default)"
        echo "  diagnostics - Run comprehensive system diagnostics"
        echo "  benchmark   - Run performance benchmarks"
        echo "  test        - Run test suite"
        echo "  shell       - Start interactive shell"
        echo ""
        echo "Environment Variables:"
        echo "  DP_FLASH_EPSILON        - Privacy epsilon parameter (default: 1.0)"
        echo "  DP_FLASH_DELTA          - Privacy delta parameter (default: 1e-5)"
        echo "  DP_FLASH_DEVICE         - Device to use (default: auto)"
        echo "  DP_FLASH_LOG_LEVEL      - Log level (default: INFO)"
        echo "  ENABLE_MONITORING       - Enable Prometheus monitoring (default: false)"
        echo "  ENABLE_AUTOSCALING      - Enable auto-scaling (default: false)"
        echo "  MAX_WORKERS             - Maximum worker threads (default: 4)"
        echo "  MIN_WORKERS             - Minimum worker threads (default: 1)"
        exit 1
        ;;
esac