# Development Environment Setup

This guide provides comprehensive instructions for setting up a development environment for DP-Flash-Attention, including CUDA requirements, privacy validation tools, and enterprise development practices.

## 🎯 Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/dp-flash-attention.git
cd dp-flash-attention
./scripts/setup_dev.sh

# Activate environment and run tests
source venv/bin/activate  # or conda activate dp-flash-attention
make test-all
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10+ (3.11 recommended for development)
- **CUDA**: 12.0+ with compute capability 7.0+ GPU
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB free space for dependencies and models

### Hardware Recommendations
- **GPU**: NVIDIA H100, A100, RTX 4090, or RTX 4080
- **CPU**: Multi-core processor (8+ cores recommended)
- **Storage**: NVMe SSD for faster builds

## 🐳 Development Environment Options

### Option 1: Docker Development (Recommended)
```bash
# Use development container
docker-compose -f docker-compose.dev.yml up -d
docker exec -it dp-flash-attention-dev bash

# Or use VSCode Dev Containers
code --folder-uri vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/yourusername/dp-flash-attention
```

### Option 2: Conda Environment
```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate dp-flash-attention

# Install development dependencies
pip install -e ".[dev,test,docs,cuda]"
```

### Option 3: Python venv
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install with development dependencies
pip install -e ".[dev,test,docs,cuda]"
```

## 🛠️ Development Tools Setup

### Essential Tools Installation
```bash
# Install development tools
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Install additional development utilities
pip install jupyterlab pytest-xdist pytest-benchmark
```

### IDE Configuration

#### VS Code Setup
Install recommended extensions:
- Python Extension Pack
- Python Docstring Generator
- CUDA support extensions
- GitLens for enhanced Git integration

#### PyCharm Configuration
1. Import project and configure Python interpreter
2. Enable pytest as test runner
3. Configure CUDA debugging if using PyCharm Professional
4. Setup code style to match Black formatting

## 🧪 Testing Framework

### Running Tests
```bash
# Quick test suite
make test

# Full test suite including GPU tests
make test-gpu

# Privacy-specific tests
make test-privacy

# Performance benchmarks
make benchmark

# Test with coverage
make test-cov
```

### Test Categories
- **Unit Tests**: Fast, isolated component testing
- **Integration Tests**: End-to-end workflow validation
- **Privacy Tests**: Differential privacy guarantee verification
- **GPU Tests**: CUDA kernel validation (requires GPU)
- **Performance Tests**: Benchmark and regression testing

## 🔒 Privacy Development Guidelines

### Privacy Parameter Validation
Always validate privacy parameters in development:
```python
# Use privacy validation utilities
from dp_flash_attention.privacy import validate_privacy_params

def validate_dp_config(epsilon, delta, sensitivity):
    """Validate privacy configuration."""
    assert epsilon > 0, "Epsilon must be positive"
    assert 0 < delta < 1, "Delta must be in (0, 1)"
    assert sensitivity > 0, "Sensitivity must be positive"
    
    return validate_privacy_params(epsilon, delta, sensitivity)
```

### Testing Privacy Guarantees
```python
# Privacy tests should verify theoretical guarantees
def test_privacy_composition():
    """Test privacy budget composition."""
    accountant = RenyiAccountant()
    
    # Multiple privacy steps
    for _ in range(10):
        accountant.add_step(epsilon=0.1, delta=1e-6, sampling_rate=0.01)
    
    total_epsilon = accountant.get_epsilon(delta=1e-5)
    assert total_epsilon < 2.0, "Privacy budget exceeded"
```

## 🚀 Development Workflow

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/privacy-enhancement

# Make changes and commit
git add .
git commit -m "feat: add privacy budget monitoring"

# Run pre-commit checks
pre-commit run --all-files

# Push and create PR
git push origin feature/privacy-enhancement
gh pr create --title "Add privacy budget monitoring" --body "Description..."
```

## 🐛 Debugging

### General Debugging
```bash
# Run with verbose output
python -m pytest tests/ -v -s

# Debug specific test
python -m pytest tests/test_privacy.py::test_epsilon_validation -v -s --pdb

# Profile performance
python -m cProfile -s cumtime examples/benchmark.py
```

### CUDA Debugging
```bash
# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run with CUDA memory checking
python -m torch.utils.collect_env
python -c "import torch; print(torch.cuda.is_available())"
```

### Privacy Debugging
```bash
# Enable privacy debug logging
export DP_DEBUG=1
python scripts/privacy_debug.py

# Validate privacy parameters
python -c "
from dp_flash_attention.privacy import validate_privacy_params
print(validate_privacy_params(epsilon=1.0, delta=1e-5, sensitivity=1.0))
"
```

## 📊 Performance Optimization

### Profiling
```bash
# Profile GPU kernels
nsys profile --trace=cuda,osrt python examples/profile_attention.py

# Memory profiling
python -m memory_profiler examples/memory_test.py

# Privacy overhead analysis
python benchmarks/privacy_overhead.py
```

---

**Next Steps**: After setup, see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and code review processes.