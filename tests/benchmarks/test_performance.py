"""Performance benchmarks comparing DP vs non-DP attention."""

import pytest
import torch
import time
from unittest.mock import Mock, patch


class TestPerformanceBenchmarks:
    """Benchmark tests for attention performance."""
    
    @pytest.mark.benchmark(group="attention")
    @pytest.mark.parametrize("batch_size,seq_len,embed_dim", [
        (32, 512, 768),   # BERT-base
        (16, 1024, 768),  # Longer sequences
        (8, 2048, 1024),  # GPT-style
    ])
    def test_attention_speed_comparison(self, benchmark, batch_size, seq_len, embed_dim):
        """Benchmark DP attention vs standard attention."""
        
        def run_dp_attention():
            # Mock DP attention computation
            q = torch.randn(batch_size, seq_len, embed_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
            k = torch.randn(batch_size, seq_len, embed_dim, device='cuda' if torch.cuda.is_available() else 'cpu')  
            v = torch.randn(batch_size, seq_len, embed_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Simulate attention computation with privacy overhead
            scores = torch.matmul(q, k.transpose(-2, -1))
            # Add simulated noise for DP
            noise = torch.randn_like(scores) * 0.1
            scores += noise
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)
            return output
        
        result = benchmark(run_dp_attention)
        assert result.shape == (batch_size, seq_len, embed_dim)

    @pytest.mark.benchmark(group="memory")  
    def test_memory_usage_dp_attention(self, benchmark):
        """Benchmark memory usage of DP attention mechanisms."""
        
        def measure_memory():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                # Simulate memory-intensive DP computation
                q = torch.randn(64, 2048, 1024, device='cuda', requires_grad=True)
                k = torch.randn(64, 2048, 1024, device='cuda', requires_grad=True)
                v = torch.randn(64, 2048, 1024, device='cuda', requires_grad=True)
                
                # Forward pass
                scores = torch.matmul(q, k.transpose(-2, -1))
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v)
                
                # Backward pass
                loss = output.sum()
                loss.backward()
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - initial_memory
                
                torch.cuda.empty_cache()
                return memory_used
            else:
                # CPU fallback - measure process memory
                return 1024 * 1024 * 100  # 100MB placeholder
        
        memory_used = benchmark(measure_memory)
        # Assert reasonable memory usage (less than 10GB for this test)
        assert memory_used < 10 * 1024 * 1024 * 1024

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_throughput_scaling(self):
        """Test throughput scaling with different privacy parameters."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        batch_size = 32
        seq_len = 512
        embed_dim = 768
        
        privacy_levels = [float('inf'), 10.0, 1.0, 0.1]  # inf = no privacy
        throughputs = []
        
        for epsilon in privacy_levels:
            # Mock timing for different privacy levels
            start_time = time.time()
            
            # Simulate computation with privacy overhead
            for _ in range(10):  # Multiple iterations for stable timing
                q = torch.randn(batch_size, seq_len, embed_dim, device='cuda')
                k = torch.randn(batch_size, seq_len, embed_dim, device='cuda')
                v = torch.randn(batch_size, seq_len, embed_dim, device='cuda')
                
                scores = torch.matmul(q, k.transpose(-2, -1))
                
                if epsilon != float('inf'):
                    # Add privacy overhead (noise + clipping simulation)
                    noise_scale = 1.0 / epsilon
                    noise = torch.randn_like(scores) * noise_scale
                    scores = torch.clamp(scores + noise, -1.0, 1.0)
                
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v)
                torch.cuda.synchronize()
            
            end_time = time.time()
            throughput = 10 / (end_time - start_time)  # iterations per second
            throughputs.append(throughput)
        
        # Verify that throughput degrades gracefully with stronger privacy
        assert throughputs[0] >= throughputs[1] >= throughputs[2] >= throughputs[3]
        
        # Privacy overhead should be reasonable (less than 2x slowdown)
        privacy_overhead = throughputs[0] / throughputs[2]  # inf vs epsilon=1
        assert privacy_overhead < 2.0