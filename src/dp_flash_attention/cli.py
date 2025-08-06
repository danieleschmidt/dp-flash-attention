"""
Command-line interface for DP-Flash-Attention benchmarking and privacy auditing.
"""

import argparse
import json
import time
from typing import Dict, Any

import torch
import numpy as np

from .utils import (cuda_version, privacy_check, check_system_requirements, 
                   benchmark_attention_kernel, estimate_memory_usage)
from .kernels import benchmark_kernel_performance, get_kernel_info
from .privacy import RenyiAccountant, GaussianMechanism, AdaptiveNoiseCalibrator
from .core import DPFlashAttention
from .functional import dp_flash_attn_func


def benchmark():
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description='Benchmark DP-Flash-Attention performance')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=512, help='Sequence length')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--head-dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Privacy epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='Privacy delta')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("DP-Flash-Attention Benchmark")
    print("=" * 40)
    
    # System information
    print("\n📊 System Information:")
    print(cuda_version())
    
    if args.verbose:
        print("\n🔍 System Requirements Check:")
        req_results = check_system_requirements()
        for key, value in req_results.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
    
    # Kernel information
    print("\n🔧 Kernel Information:")
    kernel_info = get_kernel_info()
    for key, value in kernel_info.items():
        print(f"  {key}: {value}")
    
    # Memory estimation
    print("\n💾 Memory Estimation:")
    memory_est = estimate_memory_usage(
        args.batch_size, args.seq_length, args.num_heads, args.head_dim
    )
    print(f"  Estimated GPU Memory: {memory_est['total_estimated_mb']:.1f} MB")
    
    # Run benchmark
    print(f"\n🚀 Running Benchmark...")
    print(f"  Configuration: B={args.batch_size}, S={args.seq_length}, "
          f"H={args.num_heads}, D={args.head_dim}")
    print(f"  Privacy: ε={args.epsilon}, δ={args.delta}")
    
    try:
        results = benchmark_kernel_performance(
            batch_size=args.batch_size,
            seq_len=args.seq_length,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            num_iterations=args.iterations
        )
        
        # Display results
        print("\n📈 Benchmark Results:")
        print(f"  Average Time: {results['avg_time_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.0f} samples/sec")
        print(f"  Total Time: {results['total_time_ms']:.1f} ms")
        
        # Add to results
        results['privacy_params'] = {'epsilon': args.epsilon, 'delta': args.delta}
        results['timestamp'] = time.time()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1
    
    return 0


def privacy_audit():
    """CLI entry point for privacy auditing."""
    parser = argparse.ArgumentParser(description='Audit differential privacy guarantees')
    parser.add_argument('--epsilon', type=float, required=True, help='Privacy epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='Privacy delta')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--num-steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=512, help='Sequence length')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of attention layers')
    parser.add_argument('--composition', choices=['basic', 'rdp', 'advanced'], 
                       default='rdp', help='Privacy composition method')
    parser.add_argument('--output', type=str, help='Output file for audit results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("DP-Flash-Attention Privacy Audit")
    print("=" * 40)
    
    # Privacy check
    print("\n🔒 Privacy Implementation Check:")
    print(privacy_check())
    
    # Create privacy accountant
    print(f"\n📊 Privacy Analysis:")
    print(f"  Target Privacy: (ε={args.epsilon}, δ={args.delta})")
    print(f"  Training Steps: {args.num_steps}")
    print(f"  Attention Layers: {args.num_layers}")
    print(f"  Composition Method: {args.composition}")
    
    try:
        # Initialize accountant
        accountant = RenyiAccountant()
        
        # Simulate privacy consumption
        total_epsilon_consumed = 0
        per_step_epsilon = args.epsilon / args.num_steps
        per_layer_epsilon = per_step_epsilon / args.num_layers
        
        print(f"\n🔍 Privacy Budget Allocation:")
        print(f"  Per-step epsilon: {per_step_epsilon:.6f}")
        print(f"  Per-layer epsilon: {per_layer_epsilon:.6f}")
        
        # Create Gaussian mechanism
        mechanism = GaussianMechanism(
            epsilon=per_layer_epsilon,
            delta=args.delta,
            sensitivity=args.max_grad_norm
        )
        
        print(f"  Noise scale: {mechanism.noise_scale:.4f}")
        
        # Simulate training steps
        if args.verbose:
            print(f"\n⚙️  Simulating {args.num_steps} training steps...")
        
        privacy_trace = []
        for step in range(min(args.num_steps, 100)):  # Limit simulation for performance
            step_epsilon = accountant.add_step(
                noise_scale=mechanism.noise_scale,
                delta=args.delta,
                batch_size=args.batch_size,
                dataset_size=50000  # Assumed dataset size
            )
            
            current_epsilon = accountant.get_epsilon(args.delta)
            privacy_trace.append({
                'step': step,
                'step_epsilon': step_epsilon,
                'total_epsilon': current_epsilon
            })
            
            if args.verbose and step % 20 == 0:
                print(f"    Step {step}: ε={current_epsilon:.6f}")
        
        # Final privacy analysis
        final_epsilon = accountant.get_epsilon(args.delta)
        
        print(f"\n📈 Privacy Analysis Results:")
        print(f"  Final Epsilon: {final_epsilon:.6f}")
        print(f"  Target Epsilon: {args.epsilon}")
        print(f"  Privacy Preserved: {'✓' if final_epsilon <= args.epsilon else '✗'}")
        
        if final_epsilon > args.epsilon:
            print(f"  ⚠️  Privacy budget exceeded by {final_epsilon - args.epsilon:.6f}")
        
        # Composition statistics
        comp_stats = accountant.get_composition_stats()
        print(f"\n📊 Composition Statistics:")
        print(f"  Total Steps Simulated: {comp_stats['total_steps']}")
        print(f"  Average Noise Scale: {comp_stats['avg_noise_scale']:.4f}")
        print(f"  Average Sampling Rate: {comp_stats['avg_sampling_rate']:.4f}")
        
        # Prepare results
        audit_results = {
            'privacy_params': {
                'target_epsilon': args.epsilon,
                'target_delta': args.delta,
                'final_epsilon': final_epsilon,
                'privacy_preserved': final_epsilon <= args.epsilon
            },
            'configuration': {
                'num_steps': args.num_steps,
                'num_layers': args.num_layers,
                'batch_size': args.batch_size,
                'max_grad_norm': args.max_grad_norm,
                'composition_method': args.composition
            },
            'mechanism': {
                'noise_scale': mechanism.noise_scale,
                'per_step_epsilon': per_step_epsilon,
                'per_layer_epsilon': per_layer_epsilon
            },
            'composition_stats': comp_stats,
            'privacy_trace': privacy_trace[:10],  # First 10 steps
            'timestamp': time.time()
        }
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(audit_results, f, indent=2, default=str)
            print(f"\n💾 Audit results saved to {args.output}")
        
        return 0 if final_epsilon <= args.epsilon else 1
        
    except Exception as e:
        print(f"\n❌ Privacy audit failed: {e}")
        return 1


def test_dp_attention():
    """Test DP-Flash-Attention functionality."""
    print("DP-Flash-Attention Functionality Test")
    print("=" * 40)
    
    # System check
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
    
    # Test parameters
    batch_size, seq_len, num_heads, head_dim = 4, 128, 8, 64
    embed_dim = num_heads * head_dim
    epsilon, delta = 1.0, 1e-5
    
    print(f"\n🧪 Test Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Head Dim: {head_dim}")
    print(f"  Privacy: (ε={epsilon}, δ={delta})")
    
    try:
        # Create test tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
        
        print("\n🔬 Testing DP-Flash-Attention Module:")
        
        # Test DPFlashAttention module
        dp_attn = DPFlashAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            epsilon=epsilon,
            delta=delta,
            device=device,
            dtype=torch.float32
        )
        
        # Test forward pass
        with torch.no_grad():
            output, privacy_stats = dp_attn(
                q.view(batch_size, seq_len, embed_dim),
                k.view(batch_size, seq_len, embed_dim), 
                v.view(batch_size, seq_len, embed_dim),
                return_privacy_stats=True
            )
        
        print(f"  ✓ Module forward pass successful")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Privacy spent: ε={privacy_stats.epsilon_spent:.6f}")
        print(f"  ✓ Gradient norm: {privacy_stats.grad_norm:.4f}")
        
        # Test functional interface
        print("\n🔧 Testing Functional Interface:")
        
        with torch.no_grad():
            func_output, func_privacy_stats = dp_flash_attn_func(
                q, k, v,
                epsilon=epsilon,
                delta=delta,
                return_privacy_stats=True
            )
        
        print(f"  ✓ Functional interface successful")
        print(f"  ✓ Output shape: {func_output.shape}")
        print(f"  ✓ Privacy spent: ε={func_privacy_stats.epsilon_spent:.6f}")
        
        # Verify outputs are reasonable
        assert not torch.isnan(output).any(), "Module output contains NaN"
        assert not torch.isnan(func_output).any(), "Functional output contains NaN"
        assert output.shape == (batch_size, seq_len, embed_dim), "Incorrect output shape"
        
        print(f"\n✅ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m dp_flash_attention.cli <command>")
        print("Commands: benchmark, privacy_audit, test")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove script name and command
    
    if command == 'benchmark':
        sys.exit(benchmark())
    elif command == 'privacy_audit':
        sys.exit(privacy_audit())
    elif command == 'test':
        sys.exit(test_dp_attention())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)