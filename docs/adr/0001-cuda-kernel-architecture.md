# 0001. CUDA Kernel Architecture for Differential Privacy

Date: 2025-01-01

## Status

Accepted

## Context

We need to implement differential privacy directly in CUDA kernels to achieve zero-overhead privacy guarantees. The main alternatives are:

1. **Post-hoc DP**: Apply differential privacy after standard attention computation
2. **Fused DP Kernels**: Integrate DP mechanisms directly into attention kernels
3. **Hybrid Approach**: Separate kernels for attention and DP with optimized data flow

Performance requirements demand <5% overhead compared to Flash-Attention 3, which eliminates post-hoc approaches that typically add 50-100% overhead.

## Decision

We will implement **fused DP kernels** that integrate differential privacy mechanisms directly into the Flash-Attention computation pipeline.

Key architectural decisions:
- **Triton-based Implementation**: Use Triton for kernel development to enable faster iteration and easier maintenance
- **Inline Gradient Clipping**: Perform per-sample gradient clipping during attention score computation
- **Hardware-optimized Noise**: Use cuRAND for GPU-native Gaussian noise generation
- **Memory Coalescing**: Design kernel layout to maximize memory bandwidth utilization
- **Tensor Core Integration**: Utilize Tensor Cores for matrix operations where possible

## Consequences

### Positive
- **Near-zero overhead**: Eliminates data movement between separate attention and DP kernels
- **Hardware optimization**: Direct access to GPU-specific optimization techniques
- **Memory efficiency**: Reduced memory footprint by avoiding intermediate tensors
- **Maintainability**: Triton provides better maintainability than raw CUDA
- **Performance predictability**: Fused kernels provide consistent performance characteristics

### Negative
- **Development complexity**: Significantly more complex than post-hoc approaches
- **Debugging difficulty**: Harder to debug fused kernels compared to separate components
- **Hardware dependency**: Tightly coupled to NVIDIA GPU architecture
- **Testing complexity**: Requires specialized testing infrastructure for CUDA kernels

### Risks
- **Privacy guarantee validation**: More difficult to formally verify privacy in fused implementation
- **Performance optimization**: May require extensive tuning to achieve target performance
- **Maintainability**: Complex kernel code may be difficult for contributors to understand

### Mitigations
- Extensive unit testing for privacy-critical components
- Formal verification of privacy mechanisms
- Comprehensive documentation of kernel architecture
- Modular design to allow component-level testing