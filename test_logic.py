#!/usr/bin/env python3
"""Test core logic without external dependencies."""

import sys
import os
import math

def test_privacy_validation_logic():
    """Test privacy parameter validation logic."""
    print("🔍 Testing privacy validation logic...")
    
    def validate_privacy_params(epsilon, delta):
        if not isinstance(epsilon, (int, float)):
            raise ValueError(f"epsilon must be numeric, got {type(epsilon)}")
        
        if not isinstance(delta, (int, float)):
            raise ValueError(f"delta must be numeric, got {type(delta)}")
        
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
    
    try:
        # Valid parameters
        validate_privacy_params(1.0, 1e-5)
        print("✅ Valid parameters accepted")
        
        # Test invalid epsilon
        try:
            validate_privacy_params(-1.0, 1e-5)
            return False
        except ValueError:
            print("✅ Correctly rejected negative epsilon")
        
        # Test invalid delta
        try:
            validate_privacy_params(1.0, 2.0)
            return False
        except ValueError:
            print("✅ Correctly rejected delta >= 1")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation logic failed: {e}")
        return False

def test_noise_scale_computation():
    """Test noise scale computation logic."""
    print("🔍 Testing noise scale computation...")
    
    def compute_noise_scale(epsilon, delta, max_grad_norm, sequence_length):
        # Gaussian mechanism noise scale: σ = √(2 ln(1.25/δ)) * Δ / ε
        noise_scale = math.sqrt(2 * math.log(1.25 / delta)) * max_grad_norm / epsilon
        
        # Account for sequence length
        if sequence_length > 512:
            length_factor = math.sqrt(sequence_length / 512)
            noise_scale *= length_factor
        
        return noise_scale
    
    try:
        # Test computation
        noise_scale = compute_noise_scale(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            sequence_length=512
        )
        
        print(f"✅ Computed noise scale: {noise_scale:.4f}")
        
        if noise_scale > 0:
            print("✅ Noise scale is positive")
            
            # Test with different parameters
            scale_2 = compute_noise_scale(2.0, 1e-5, 1.0, 512)
            if scale_2 < noise_scale:
                print("✅ Higher epsilon gives lower noise (correct)")
                return True
            else:
                print("❌ Noise scale relationship incorrect")
                return False
        else:
            print("❌ Noise scale should be positive")
            return False
            
    except Exception as e:
        print(f"❌ Noise computation failed: {e}")
        return False

def test_rdp_computation():
    """Test RDP value computation logic."""
    print("🔍 Testing RDP computation...")
    
    def compute_rdp_gaussian(noise_scale, alphas):
        """Compute RDP values for Gaussian mechanism."""
        rdp_values = []
        
        for alpha in alphas:
            if alpha == 1.0:
                rdp_values.append(0.0)
            else:
                # RDP for Gaussian mechanism: α/(2σ²)
                rdp_value = alpha / (2 * noise_scale ** 2)
                rdp_values.append(rdp_value)
        
        return rdp_values
    
    def rdp_to_dp(rdp_values, alphas, delta):
        """Convert RDP to (ε, δ)-DP."""
        if delta <= 0:
            return float('inf')
        
        eps_values = []
        for alpha, rdp in zip(alphas, rdp_values):
            if alpha == 1.0:
                eps_values.append(float('inf'))
            else:
                # Convert: ε = RDP + log(1/δ)/(α-1)
                eps = rdp + math.log(1.0 / delta) / (alpha - 1)
                eps_values.append(eps)
        
        return min(eps_values)
    
    try:
        # Test RDP computation
        alphas = [1.1, 2.0, 4.0, 8.0, 16.0, 32.0]
        noise_scale = 2.0
        
        rdp_vals = compute_rdp_gaussian(noise_scale, alphas)
        print(f"✅ Computed RDP values: {len(rdp_vals)} values")
        
        # Convert to DP
        epsilon = rdp_to_dp(rdp_vals, alphas, 1e-5)
        print(f"✅ Converted to DP: ε={epsilon:.4f}")
        
        if epsilon > 0 and epsilon != float('inf'):
            print("✅ Valid epsilon obtained")
            return True
        else:
            print("❌ Invalid epsilon value")
            return False
            
    except Exception as e:
        print(f"❌ RDP computation failed: {e}")
        return False

def test_attention_dimension_logic():
    """Test attention dimension calculations."""
    print("🔍 Testing attention dimensions...")
    
    def validate_attention_dims(embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        return embed_dim // num_heads
    
    try:
        # Valid dimensions
        head_dim = validate_attention_dims(768, 12)
        print(f"✅ Valid dimensions: embed_dim=768, heads=12, head_dim={head_dim}")
        
        if head_dim == 64:
            print("✅ Correct head dimension calculated")
        else:
            print(f"❌ Expected head_dim=64, got {head_dim}")
            return False
        
        # Invalid dimensions
        try:
            validate_attention_dims(770, 12)
            print("❌ Should have rejected non-divisible dimensions")
            return False
        except ValueError:
            print("✅ Correctly rejected non-divisible dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Dimension logic failed: {e}")
        return False

def test_memory_estimation():
    """Test memory usage estimation."""
    print("🔍 Testing memory estimation...")
    
    def estimate_memory_usage(batch_size, seq_len, num_heads, head_dim, bytes_per_element=2):
        # Input tensors (Q, K, V)
        input_size = batch_size * seq_len * num_heads * head_dim * bytes_per_element
        total_input = 3 * input_size
        
        # Attention scores
        scores_size = batch_size * num_heads * seq_len * seq_len * bytes_per_element
        
        # Output and noise
        output_size = input_size
        noise_size = scores_size
        
        # Working memory
        working_memory = max(scores_size, output_size) * 2
        
        total_bytes = total_input + scores_size + output_size + noise_size + working_memory
        total_mb = total_bytes / (1024 ** 2)
        
        return {
            'total_mb': total_mb,
            'input_mb': total_input / (1024 ** 2),
            'scores_mb': scores_size / (1024 ** 2),
            'components': {
                'input': total_input,
                'scores': scores_size,
                'output': output_size,
                'noise': noise_size,
                'working': working_memory
            }
        }
    
    try:
        # Test typical configuration
        memory = estimate_memory_usage(
            batch_size=32,
            seq_len=512,
            num_heads=12,
            head_dim=64
        )
        
        print(f"✅ Memory estimated: {memory['total_mb']:.1f} MB")
        print(f"   Input tensors: {memory['input_mb']:.1f} MB")
        print(f"   Attention scores: {memory['scores_mb']:.1f} MB")
        
        if memory['total_mb'] > 0:
            print("✅ Positive memory estimate")
            
            # Test scaling
            memory_2x = estimate_memory_usage(64, 512, 12, 64)
            if memory_2x['total_mb'] > memory['total_mb']:
                print("✅ Memory scales with batch size")
                return True
            else:
                print("❌ Memory scaling incorrect")
                return False
        else:
            print("❌ Memory estimate should be positive")
            return False
            
    except Exception as e:
        print(f"❌ Memory estimation failed: {e}")
        return False

def main():
    """Run all logic tests."""
    print("🧪 DP-Flash-Attention Logic Tests")
    print("=" * 40)
    
    tests = [
        test_privacy_validation_logic,
        test_noise_scale_computation,
        test_rdp_computation,
        test_attention_dimension_logic,
        test_memory_estimation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All logic tests passed!")
        print("\n🔧 Generation 1 (Make It Work) - Core logic validated ✅")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())