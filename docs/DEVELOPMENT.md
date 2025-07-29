# Development Guide

This guide covers the development workflow, tools, and best practices for contributing to DP-Flash-Attention.

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with development headers
- **NVIDIA GPU** with CUDA 12.0+ (recommended)
- **Git** for version control
- **Docker** (optional, for containerized development)

### Development Setup

#### Option 1: Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/dp-flash-attention.git
cd dp-flash-attention

# Run automated setup
./scripts/setup_dev.sh

# Verify installation
make test
```

#### Option 2: Dev Container (Recommended)

```bash
# Open in VS Code with Dev Containers extension
code .
# VS Code will prompt to reopen in container

# Or manually with Docker
docker build -f .devcontainer/Dockerfile -t dp-flash-dev .
docker run --gpus all -it -v $(pwd):/workspaces/dp-flash-attention dp-flash-dev
```

#### Option 3: Manual Setup

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,test,docs,cuda]"

# Set up pre-commit hooks
pre-commit install
```

## 🛠️ Development Tools

### Code Quality

- **Black**: Code formatting (`make format`)
- **Ruff**: Fast linting and import sorting (`make lint`)
- **MyPy**: Static type checking (`make lint`)
- **Pre-commit**: Automated quality checks on commit

### Testing

- **Pytest**: Test framework with multiple categories
- **Coverage**: Code coverage reporting with pytest-cov
- **Benchmark**: Performance testing with pytest-benchmark
- **Hypothesis**: Property-based testing for edge cases

### Documentation

- **Sphinx**: Documentation generation (`make docs`)
- **MyST**: Markdown support in Sphinx
- **ReadTheDocs**: Hosted documentation
- **Jupyter**: Interactive examples and tutorials

## 🧪 Testing Strategy

### Test Categories

```bash
# Unit tests - fast, isolated components
make test

# Integration tests - component interactions  
make test-integration

# Privacy tests - differential privacy guarantees
make test-privacy

# GPU tests - CUDA functionality (requires GPU)
make test-gpu

# Performance benchmarks
make benchmark

# All tests
make test-all
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_privacy.py
import pytest
from dp_flash_attention.privacy import RenyiAccountant

class TestRenyiAccountant:
    def test_initialization(self):
        accountant = RenyiAccountant()
        assert len(accountant.alpha_values) > 0
        
    @pytest.mark.parametrize("epsilon,delta", [
        (1.0, 1e-5),
        (0.5, 1e-6)  
    ])
    def test_privacy_composition(self, epsilon, delta):
        accountant = RenyiAccountant()
        accountant.add_step(epsilon, delta, sampling_rate=0.01)
        total_eps = accountant.get_epsilon(delta)
        assert total_eps >= epsilon
```

#### Privacy Test Example

```python
# tests/privacy/test_guarantees.py
@pytest.mark.slow
def test_membership_inference_resistance():
    """Test that DP attention resists membership inference."""
    # Implementation of empirical privacy test
    dp_attn = DPFlashAttention(embed_dim=768, num_heads=12, 
                               epsilon=1.0, delta=1e-5)
    
    # Train shadow models and run membership inference
    privacy_violation_rate = run_membership_inference_attack(dp_attn)
    assert privacy_violation_rate < 0.1  # Less than 10% violation rate
```

#### GPU Test Example

```python
# tests/gpu/test_cuda_kernels.py
@pytest.mark.gpu
def test_cuda_kernel_correctness():
    """Test CUDA kernel produces correct results."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Compare GPU vs CPU results
    input_tensor = torch.randn(32, 512, 768)
    
    cpu_result = dp_attention_cpu(input_tensor)
    gpu_result = dp_attention_gpu(input_tensor.cuda())
    
    assert torch.allclose(cpu_result, gpu_result.cpu(), atol=1e-4)
```

### Test Markers

Use pytest markers to categorize tests:

- `@pytest.mark.slow`: Tests that take >5 seconds
- `@pytest.mark.gpu`: Tests requiring CUDA hardware
- `@pytest.mark.privacy`: Privacy-related tests
- `@pytest.mark.benchmark`: Performance benchmarks
- `@pytest.mark.integration`: Cross-component tests

## 🏗️ Project Structure

```
dp-flash-attention/
├── src/dp_flash_attention/       # Main package
│   ├── core/                     # Core DP attention implementations
│   ├── kernels/                  # CUDA kernel code
│   ├── privacy/                  # Privacy accounting and mechanisms
│   ├── utils/                    # Utility functions
│   └── integrations/             # Framework integrations
├── tests/                        # Test suites
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── privacy/                  # Privacy-specific tests
│   ├── benchmarks/               # Performance benchmarks
│   └── gpu/                      # GPU-specific tests
├── docs/                         # Documentation
│   ├── source/                   # Sphinx source files
│   ├── workflows/                # GitHub Actions documentation
│   └── examples/                 # Usage examples
├── scripts/                      # Development scripts
├── .devcontainer/                # Development container config
└── notebooks/                    # Jupyter notebooks
```

## 🔧 Development Workflow

### Feature Development

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement feature with tests**
   ```bash
   # Write code in src/
   # Write tests in tests/ 
   # Update documentation
   ```

3. **Run quality checks**
   ```bash
   make lint
   make test
   make test-privacy  # For privacy-related features
   ```

4. **Submit pull request**
   - Use the provided PR template
   - Include privacy impact assessment
   - Add performance benchmarks if applicable

### Privacy-Critical Changes

Privacy-related changes require additional validation:

1. **Theoretical analysis** of privacy guarantees
2. **Formal verification** where possible  
3. **Empirical testing** with membership inference
4. **Expert review** from privacy team
5. **Documentation** of privacy implications

### CUDA Kernel Development

For CUDA kernel modifications:

1. **Test on multiple GPU architectures** (V100, A100, H100)
2. **Benchmark performance** vs previous implementation
3. **Validate numerical accuracy** against CPU reference
4. **Check memory usage** and potential leaks
5. **Document optimization techniques** used

## 📚 Documentation

### Code Documentation

- **Docstrings**: Use Google style for all public APIs
- **Type hints**: Required for all function signatures
- **Comments**: Explain complex algorithms and privacy mechanisms

Example:
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
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        epsilon: Privacy budget parameter (ε)
        delta: Privacy parameter (δ) 
        causal: Whether to apply causal masking
        
    Returns:
        Attention output with (ε, δ)-differential privacy guarantee
        
    Privacy:
        This function provides (ε, δ)-differential privacy through
        Gaussian noise injection calibrated to gradient sensitivity.
    """
```

### Building Documentation

```bash
# Build HTML documentation
make docs

# Serve locally for preview
make docs-serve

# Documentation will be available at http://localhost:8000
```

## 🔒 Security Considerations

### Secure Development

- **Never commit secrets** or API keys
- **Validate all inputs** especially privacy parameters
- **Use secure random number generation** for DP noise
- **Follow OWASP guidelines** for secure coding

### Privacy Security

- **Validate privacy parameters** at runtime
- **Implement secure composition** of privacy mechanisms  
- **Use cryptographically secure PRNGs** for noise generation
- **Audit privacy budget consumption** with logging

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests cover new functionality
- [ ] Privacy implications documented
- [ ] Performance impact assessed
- [ ] Security vulnerabilities checked
- [ ] Documentation updated

## 🚀 Performance Optimization

### Profiling Tools

- **PyTorch Profiler**: For PyTorch-specific optimizations
- **NVIDIA Nsight**: For CUDA kernel profiling
- **cProfile**: For Python performance analysis
- **Memory Profiler**: For memory usage optimization

### Optimization Guidelines

1. **Profile first** - measure before optimizing
2. **Focus on hot paths** - optimize critical sections
3. **Memory efficiency** - minimize memory allocations
4. **CUDA best practices** - coalesced memory access, occupancy
5. **Benchmark regularly** - prevent performance regressions

### Performance Testing

```bash
# Run benchmarks with detailed output
pytest tests/benchmarks/ -v --benchmark-only

# Compare against baseline
pytest tests/benchmarks/ --benchmark-compare=baseline.json

# Profile memory usage
python -m memory_profiler your_script.py
```

## 🤝 Collaboration

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and collaboration
- **Security Email**: security@dp-flash-attention.org

### Code Review Process

1. **Automated checks** must pass (CI/CD)
2. **Manual review** by maintainer
3. **Privacy review** for privacy-impacting changes
4. **Performance review** for optimization changes
5. **Final approval** and merge

### Getting Help

- **Documentation**: Start with official docs
- **Examples**: Check notebooks/ for usage examples
- **Community**: Ask in GitHub Discussions
- **Issues**: Report bugs with reproduction steps

## 🏆 Recognition

We value all contributions:

- **Code contributions** (features, bug fixes, optimizations)
- **Documentation** improvements and examples
- **Testing** and quality assurance
- **Research** and theoretical analysis
- **Community support** and mentorship

Contributors are recognized in:
- README contributors section
- Release notes
- Academic paper acknowledgments
- Conference presentations

## 📈 Monitoring Progress

### Development Metrics

Track your contribution impact:

- **Code coverage** improvements
- **Performance** benchmark improvements
- **Privacy** guarantee enhancements
- **Documentation** completeness
- **Community** engagement

### Quality Gates

All changes must meet:

- [ ] Code coverage >90%
- [ ] All tests passing
- [ ] Performance within 5% of baseline
- [ ] Documentation updated
- [ ] Security review passed (if applicable)
- [ ] Privacy guarantees maintained (if applicable)

Happy coding! 🚀