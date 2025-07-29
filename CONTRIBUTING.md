# Contributing to DP-Flash-Attention

Thank you for your interest in contributing to DP-Flash-Attention! This document provides guidelines and information for contributors.

## 🎯 Overview

DP-Flash-Attention is a research-oriented library implementing differential privacy in CUDA-optimized attention mechanisms. We welcome contributions in:

- **CUDA kernel optimizations** for differential privacy mechanisms
- **Privacy algorithm improvements** and new DP mechanisms
- **Performance benchmarking** and testing infrastructure
- **Documentation** and educational materials
- **Integration** with other ML frameworks

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/yourusername/dp-flash-attention.git
cd dp-flash-attention
```

2. **Set up development environment**:
```bash
# Run the setup script
./scripts/setup_dev.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,test,docs,cuda]"
pre-commit install
```

3. **Verify installation**:
```bash
make test
make test-gpu  # Requires CUDA
```

### CUDA Development Requirements

- **NVIDIA GPU** with compute capability 7.0+ (V100, A100, H100, RTX 4090)
- **CUDA Toolkit 12.0+** with development tools
- **Triton 2.3+** for kernel development
- **PyTorch 2.3+** with CUDA support

## 📋 Contribution Process

### 1. Issue Discussion

- **Check existing issues** before creating new ones
- **Use issue templates** for bug reports and feature requests
- **Discuss major changes** in issues before implementation
- **Privacy-related changes** require additional review

### 2. Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes with tests
# Run quality checks
make lint
make test
make test-privacy

# Commit with conventional commits
git commit -m "feat: add new DP mechanism for Laplacian noise"
```

### 3. Pull Request Requirements

- ✅ **All tests pass** (unit, integration, privacy tests)
- ✅ **Code quality checks** (black, ruff, mypy)
- ✅ **Privacy guarantees verified** for privacy-related changes
- ✅ **Performance benchmarks** for kernel changes
- ✅ **Documentation updated** for API changes
- ✅ **Changelog entry** for user-facing changes

## 🧪 Testing Guidelines

### Test Categories

```bash
# Unit tests - fast, isolated
make test

# GPU tests - require CUDA hardware  
make test-gpu

# Privacy tests - verify DP guarantees
make test-privacy

# Integration tests - end-to-end scenarios
make test-integration

# Performance benchmarks
make benchmark
```

### Privacy Testing Requirements

All privacy mechanism changes must include:

1. **Theoretical analysis** of privacy guarantees
2. **Empirical privacy tests** (membership inference)
3. **Composition analysis** for multi-layer usage
4. **Sensitivity analysis** for gradient clipping

### Writing Tests

```python
import pytest
import torch
from dp_flash_attention import DPFlashAttention

@pytest.mark.gpu
def test_dp_attention_privacy_guarantee():
    """Test that privacy guarantee holds under attack."""
    dp_attn = DPFlashAttention(
        embed_dim=768, 
        num_heads=12, 
        epsilon=1.0, 
        delta=1e-5
    )
    
    # Implementation details...
    assert privacy_guarantee_verified
```

## 📚 Documentation Standards

### Code Documentation

- **Docstrings** for all public APIs using Google style
- **Type hints** for all function signatures
- **Privacy parameters** clearly documented
- **CUDA kernel documentation** for low-level code

### API Documentation Example

```python
def dp_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    epsilon: float,
    delta: float,
    causal: bool = False
) -> torch.Tensor:
    """
    Differentially private Flash-Attention function.
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]  
        v: Value tensor [batch, seq_len, num_heads, head_dim]
        epsilon: Privacy budget parameter (ε)
        delta: Privacy parameter (δ)
        causal: Whether to apply causal masking
        
    Returns:
        Attention output with privacy guarantees (ε, δ)-DP
        
    Privacy:
        This function provides (ε, δ)-differential privacy through
        Gaussian noise injection calibrated to gradient sensitivity.
    """
```

## 🔒 Security and Privacy Guidelines

### Privacy-Critical Code

- **Privacy parameters** must be validated
- **Random number generation** must be cryptographically secure
- **Gradient clipping** must be applied correctly
- **Noise calibration** must account for sensitivity

### Security Best Practices

- **No hardcoded secrets** or API keys
- **Input validation** for all user-provided parameters  
- **Secure defaults** for privacy parameters
- **Audit trails** for privacy budget consumption

## 🎨 Code Style

We use automated formatting and linting:

```bash
# Format code
make format

# Check style  
make lint
```

### Style Guidelines

- **Line length**: 88 characters (Black default)
- **Import order**: isort with Black compatibility
- **Type hints**: Required for public APIs
- **Variable naming**: Descriptive names, avoid abbreviations

## 🏗️ Architecture Guidelines

### CUDA Kernel Development

```cuda
// Kernel naming convention
__global__ void dp_flash_attention_kernel(
    const float* Q,           // Input tensors
    const float* K, 
    const float* V,
    float* Out,               // Output tensor
    const PrivacyParams params, // Privacy configuration
    const AttentionConfig config // Attention configuration
);
```

### Python API Design

- **Consistent interfaces** with PyTorch conventions
- **Privacy-first design** - require explicit privacy parameters
- **Graceful degradation** when CUDA unavailable
- **Clear error messages** for common mistakes

## 🔬 Research Contributions

### Academic Contributions

We welcome research contributions including:

- **Novel DP mechanisms** for attention
- **Privacy-utility trade-off** improvements  
- **Formal verification** of privacy guarantees
- **Empirical privacy analysis** methodologies

### Publication Policy

- **Research collaborations** encouraged
- **Attribution** for significant contributions
- **Open access** to research artifacts

## 📊 Performance Contributions

### Benchmarking Standards

```python
# Benchmark template
@pytest.mark.benchmark(group="attention")
def test_dp_attention_performance(benchmark):
    def run_attention():
        return dp_attn(q, k, v)
    
    result = benchmark(run_attention)
    assert result.shape == expected_shape
```

### Performance Requirements

- **No regression** in baseline Flash-Attention performance
- **Memory efficiency** comparable to non-private versions
- **Scalability** to large sequence lengths

## 🤝 Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Research Email**: research@dp-flash-attention.org
- **Security Email**: security@dp-flash-attention.org

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/):

- **Be respectful** and inclusive
- **Focus on technical merit** 
- **Help newcomers** learn and contribute
- **Maintain professional standards**

## 📝 Release Process

### Version Numbering

We use semantic versioning (SemVer):
- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, security updates

### Changelog Format

```markdown
## [0.2.0] - 2025-08-01

### Added
- New Laplacian DP mechanism
- Support for heterogeneous privacy levels

### Changed  
- Improved noise calibration algorithm

### Fixed
- Memory leak in CUDA kernels
```

## 🏆 Recognition

### Contributor Recognition

- **Contributors list** in README
- **Release notes** attribution
- **Academic acknowledgments** for research contributions

### Contribution Types

We recognize various contribution types:
- 💻 **Code** contributions
- 📖 **Documentation** improvements  
- 🐛 **Bug** reports and fixes
- 💡 **Ideas** and feature suggestions
- 🔬 **Research** and analysis
- 🎨 **Design** and UX improvements

Thank you for contributing to privacy-preserving machine learning! 🔒🚀