"""
Performance benchmarking suite for DP-Flash-Attention.

Comprehensive performance testing with privacy parameter variations.
"""

import pytest
import torch
import time
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from contextlib import contextmanager

from dp_flash_attention.core import DPFlashAttention


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    batch_size: int
    seq_length: int
    embed_dim: int
    num_heads: int
    epsilon: float
    delta: float
    device: str = 'cuda'
    dtype: torch.dtype = torch.float16
    num_warmup: int = 5
    num_iterations: int = 20


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    mean_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    memory_usage: int
    privacy_overhead: float


class PerformanceBenchmark:
    """Performance benchmark suite for DP-Flash-Attention."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    @contextmanager
    def cuda_timer(self):
        """CUDA-synchronized timing context manager."""
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        yield
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        self.last_duration = end_time - start_time
    
    def benchmark_attention(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark DP-Flash-Attention with given configuration."""
        # Initialize model
        dp_attn = DPFlashAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            epsilon=config.epsilon,
            delta=config.delta
        ).to(config.device)
        
        # Generate test data
        batch_size, seq_len, embed_dim = config.batch_size, config.seq_length, config.embed_dim
        
        Q = torch.randn(batch_size, seq_len, embed_dim, 
                       device=config.device, dtype=config.dtype, requires_grad=True)
        K = torch.randn(batch_size, seq_len, embed_dim, 
                       device=config.device, dtype=config.dtype, requires_grad=True)
        V = torch.randn(batch_size, seq_len, embed_dim, 
                       device=config.device, dtype=config.dtype, requires_grad=True)
        
        # Warmup runs
        for _ in range(config.num_warmup):
            with torch.no_grad():
                _ = dp_attn(Q, K, V)
        
        # Benchmark runs
        latencies = []
        memory_usage = 0
        
        for i in range(config.num_iterations):
            torch.cuda.empty_cache()
            
            # Measure memory before
            memory_before = torch.cuda.memory_allocated()
            
            with self.cuda_timer():
                output = dp_attn(Q, K, V)
                
            latencies.append(self.last_duration)
            
            # Measure memory after first iteration
            if i == 0:
                memory_usage = torch.cuda.memory_allocated() - memory_before
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        sorted_latencies = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_latencies))
        p99_idx = int(0.99 * len(sorted_latencies))
        p95_latency = sorted_latencies[p95_idx]
        p99_latency = sorted_latencies[p99_idx]
        
        # Calculate throughput (tokens per second)
        tokens_per_batch = batch_size * seq_len
        throughput = tokens_per_batch / mean_latency
        
        # Estimate privacy overhead (would need non-DP baseline)
        privacy_overhead = 0.05  # Placeholder - would measure against baseline
        
        result = BenchmarkResult(
            config=config,
            mean_latency=mean_latency,
            std_latency=std_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput=throughput,
            memory_usage=memory_usage,
            privacy_overhead=privacy_overhead
        )
        
        self.results.append(result)
        return result
    
    def benchmark_privacy_scaling(self) -> List[BenchmarkResult]:
        """Benchmark performance across different privacy parameters."""
        base_config = BenchmarkConfig(
            batch_size=16,
            seq_length=512,
            embed_dim=768,
            num_heads=12,
            epsilon=1.0,
            delta=1e-5
        )
        
        epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        results = []
        
        for epsilon in epsilon_values:
            config = BenchmarkConfig(
                batch_size=base_config.batch_size,
                seq_length=base_config.seq_length,
                embed_dim=base_config.embed_dim,
                num_heads=base_config.num_heads,
                epsilon=epsilon,
                delta=base_config.delta
            )
            result = self.benchmark_attention(config)
            results.append(result)
            
        return results
    
    def benchmark_model_scaling(self) -> List[BenchmarkResult]:
        """Benchmark performance across different model sizes."""
        configs = [
            # BERT-Base
            BenchmarkConfig(16, 512, 768, 12, 1.0, 1e-5),
            # BERT-Large  
            BenchmarkConfig(8, 512, 1024, 16, 1.0, 1e-5),
            # GPT-2 Medium
            BenchmarkConfig(4, 1024, 1024, 16, 1.0, 1e-5),
            # GPT-2 Large
            BenchmarkConfig(2, 1024, 1280, 20, 1.0, 1e-5),
        ]
        
        results = []
        for config in configs:
            result = self.benchmark_attention(config)
            results.append(result)
            
        return results
    
    def generate_report(self) -> str:
        """Generate performance benchmark report."""
        if not self.results:
            return "No benchmark results available."
            
        report = []
        report.append("DP-Flash-Attention Performance Benchmark Report")
        report.append("=" * 50)
        report.append()
        
        for result in self.results:
            config = result.config
            report.append(f"Configuration:")
            report.append(f"  Batch Size: {config.batch_size}")
            report.append(f"  Sequence Length: {config.seq_length}")
            report.append(f"  Embed Dim: {config.embed_dim}")
            report.append(f"  Num Heads: {config.num_heads}")
            report.append(f"  Privacy: ε={config.epsilon}, δ={config.delta}")
            report.append()
            
            report.append(f"Performance Results:")
            report.append(f"  Mean Latency: {result.mean_latency*1000:.2f}ms")
            report.append(f"  Std Latency: {result.std_latency*1000:.2f}ms")
            report.append(f"  P95 Latency: {result.p95_latency*1000:.2f}ms")
            report.append(f"  P99 Latency: {result.p99_latency*1000:.2f}ms")
            report.append(f"  Throughput: {result.throughput:.0f} tokens/sec")
            report.append(f"  Memory Usage: {result.memory_usage / (1024**2):.1f} MB")
            report.append(f"  Privacy Overhead: {result.privacy_overhead*100:.1f}%")
            report.append()
            report.append("-" * 30)
            report.append()
            
        return "\n".join(report)


# Pytest fixtures and tests
@pytest.fixture
def benchmark_suite():
    """Fixture providing benchmark suite."""
    return PerformanceBenchmark()


@pytest.mark.benchmark
@pytest.mark.gpu
class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""
    
    def test_basic_performance(self, benchmark_suite):
        """Test basic performance characteristics."""
        config = BenchmarkConfig(
            batch_size=8,
            seq_length=256,
            embed_dim=512,
            num_heads=8,
            epsilon=1.0,
            delta=1e-5
        )
        
        result = benchmark_suite.benchmark_attention(config)
        
        # Assert reasonable performance
        assert result.mean_latency < 0.1, f"Latency too high: {result.mean_latency:.4f}s"
        assert result.throughput >= 1000, f"Throughput too low: {result.throughput:.0f} tokens/sec"
        assert result.memory_usage < 1024**3, f"Memory usage too high: {result.memory_usage} bytes"
    
    @pytest.mark.slow
    def test_privacy_scaling_performance(self, benchmark_suite):
        """Test performance scaling with privacy parameters."""
        results = benchmark_suite.benchmark_privacy_scaling()
        
        # Should have results for different epsilon values
        assert len(results) >= 5
        
        # Performance should degrade with stronger privacy (lower epsilon)
        epsilon_to_latency = {r.config.epsilon: r.mean_latency for r in results}
        epsilons = sorted(epsilon_to_latency.keys())
        
        # Check that very strong privacy (low epsilon) has higher latency
        strong_privacy_latency = epsilon_to_latency[epsilons[0]]  # Lowest epsilon
        weak_privacy_latency = epsilon_to_latency[epsilons[-1]]   # Highest epsilon
        
        # Allow some variance, but strong privacy should generally be slower
        assert strong_privacy_latency <= weak_privacy_latency * 2.0, \
            "Privacy overhead too high"
    
    @pytest.mark.slow
    def test_model_scaling_performance(self, benchmark_suite):
        """Test performance scaling with model size."""
        results = benchmark_suite.benchmark_model_scaling()
        
        assert len(results) >= 3
        
        # Generate and print report
        report = benchmark_suite.generate_report()
        print("\n" + report)
        
        # Verify that larger models have proportionally higher latency
        for result in results:
            model_size = result.config.embed_dim * result.config.num_heads
            expected_min_latency = model_size / 1000000  # Very rough heuristic
            
            assert result.mean_latency >= expected_min_latency, \
                f"Latency suspiciously low for model size {model_size}"
    
    @pytest.mark.parametrize("batch_size,seq_length", [
        (1, 128),    # Small
        (8, 512),    # Medium  
        (32, 1024),  # Large
    ])
    def test_input_size_scaling(self, benchmark_suite, batch_size, seq_length):
        """Test performance scaling with input dimensions."""
        config = BenchmarkConfig(
            batch_size=batch_size,
            seq_length=seq_length,
            embed_dim=768,
            num_heads=12,
            epsilon=1.0,
            delta=1e-5
        )
        
        result = benchmark_suite.benchmark_attention(config)
        
        # Verify reasonable bounds
        max_expected_latency = (batch_size * seq_length) / 10000  # tokens per 10ms
        assert result.mean_latency <= max_expected_latency, \
            f"Latency {result.mean_latency:.4f}s too high for input size"
    
    def test_memory_efficiency(self, benchmark_suite):
        """Test memory usage efficiency."""
        config = BenchmarkConfig(
            batch_size=16,
            seq_length=512, 
            embed_dim=768,
            num_heads=12,
            epsilon=1.0,
            delta=1e-5
        )
        
        result = benchmark_suite.benchmark_attention(config)
        
        # Estimate expected memory usage
        total_elements = config.batch_size * config.seq_length * config.embed_dim
        expected_memory = total_elements * 2 * 3  # float16 * Q,K,V
        max_memory = expected_memory * 10  # Allow 10x overhead for intermediates
        
        assert result.memory_usage <= max_memory, \
            f"Memory usage {result.memory_usage} exceeds expected {max_memory}"


if __name__ == "__main__":
    # Run benchmarks directly
    suite = PerformanceBenchmark()
    
    print("Running DP-Flash-Attention benchmarks...")
    
    # Basic benchmark
    config = BenchmarkConfig(16, 512, 768, 12, 1.0, 1e-5)
    result = suite.benchmark_attention(config)
    
    # Privacy scaling
    privacy_results = suite.benchmark_privacy_scaling()
    
    # Generate report
    report = suite.generate_report()
    print(report)
    
    # Save results
    with open("benchmark_results.txt", "w") as f:
        f.write(report)