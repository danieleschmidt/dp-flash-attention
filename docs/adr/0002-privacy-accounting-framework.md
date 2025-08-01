# 0002. Privacy Accounting Framework Selection

Date: 2025-01-01

## Status

Accepted

## Context

We need to select a privacy accounting framework to track privacy budget consumption across multiple attention layers and training steps. The main options are:

1. **Basic (ε, δ)-DP Composition**: Simple sequential composition using basic theorems
2. **Rényi Differential Privacy (RDP)**: Advanced composition with tighter bounds
3. **Privacy Random Variables (PRV)**: Most recent approach with optimal composition
4. **Hybrid Approach**: Combine multiple frameworks for different use cases

Requirements:
- Support multi-layer composition in transformers
- Tight privacy bounds for practical privacy budgets
- Integration with existing DP libraries (Opacus, Google DP-Accounting)
- Real-time budget tracking during training

## Decision

We will use **Rényi Differential Privacy (RDP)** as our primary privacy accounting framework, with PRV integration for advanced use cases.

Implementation details:
- **Primary Framework**: RDP for all standard composition scenarios
- **Advanced Cases**: PRV integration for heterogeneous privacy levels
- **Library Integration**: Use Google's dp-accounting library for RDP computations
- **Custom Extensions**: Implement attention-specific composition rules
- **Real-time Tracking**: Efficient GPU-CPU communication for budget updates

## Consequences

### Positive
- **Tight Bounds**: RDP provides significantly tighter composition bounds than basic DP
- **Multi-layer Support**: Natural handling of transformer layer composition
- **Industry Standard**: RDP is widely accepted in the privacy research community
- **Library Support**: Excellent integration with existing privacy libraries
- **Scalability**: Efficient computation suitable for large-scale training

### Negative
- **Complexity**: More complex than basic DP composition for users to understand
- **Parameter Selection**: Requires careful selection of Rényi orders (α values)
- **Computational Overhead**: Additional computation for RDP accounting

### Risks
- **Parameter Sensitivity**: Incorrect α selection could lead to loose bounds
- **Implementation Bugs**: Complex accounting logic increases bug risk
- **User Confusion**: Users may not understand RDP parameters

### Mitigations
- Provide automated α selection based on privacy requirements
- Extensive testing of accounting logic with known test cases
- Clear documentation and examples for RDP usage
- Fallback to basic DP composition for debugging/validation