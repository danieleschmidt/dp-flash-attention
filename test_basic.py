#!/usr/bin/env python3
"""Basic test of DP-Flash-Attention functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings("ignore")

def test_basic_imports():
    """Test that core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from dp_flash_attention import DPFlashAttention, dp_flash_attn_func
        from dp_flash_attention.privacy import RenyiAccountant, GaussianMechanism
        from dp_flash_attention.utils import validate_privacy_params, compute_noise_scale
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_privacy_validation():
    """Test privacy parameter validation."""
    print("🔍 Testing privacy validation...")
    
    try:
        from dp_flash_attention.utils import validate_privacy_params
        
        # Valid parameters
        validate_privacy_params(1.0, 1e-5)
        print("✅ Valid parameters accepted")
        
        # Invalid parameters should raise errors
        try:
            validate_privacy_params(-1.0, 1e-5)
            print("❌ Should have rejected negative epsilon")
            return False
        except ValueError:
            print("✅ Correctly rejected negative epsilon")
        
        try:
            validate_privacy_params(1.0, 2.0)
            print("❌ Should have rejected delta >= 1")
            return False
        except ValueError:
            print("✅ Correctly rejected delta >= 1")
        
        return True
    except Exception as e:
        print(f"❌ Privacy validation test failed: {e}")
        return False

def test_noise_computation():
    """Test noise scale computation."""
    print("🔍 Testing noise computation...")
    
    try:
        from dp_flash_attention.utils import compute_noise_scale
        
        noise_scale = compute_noise_scale(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            sequence_length=512
        )
        
        print(f"✅ Noise scale computed: {noise_scale:.4f}")
        
        if noise_scale > 0:
            print("✅ Noise scale is positive")
            return True
        else:
            print("❌ Noise scale should be positive")
            return False
            
    except Exception as e:
        print(f"❌ Noise computation test failed: {e}")
        return False

def test_accountant():
    """Test privacy accountant."""
    print("🔍 Testing privacy accountant...")
    
    try:
        from dp_flash_attention.privacy import RenyiAccountant
        
        accountant = RenyiAccountant()
        
        # Add some privacy steps
        step_epsilon = accountant.add_step(
            noise_scale=1.0,
            delta=1e-5,
            batch_size=32
        )
        
        print(f"✅ Added privacy step, epsilon: {step_epsilon:.6f}")
        
        total_epsilon = accountant.get_epsilon(1e-5)
        print(f"✅ Total epsilon: {total_epsilon:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Accountant test failed: {e}")
        return False

def test_mechanism():
    """Test Gaussian mechanism."""
    print("🔍 Testing Gaussian mechanism...")
    
    try:
        from dp_flash_attention.privacy import GaussianMechanism
        import torch
        
        mechanism = GaussianMechanism(
            epsilon=1.0,
            delta=1e-5,
            sensitivity=1.0
        )
        
        print(f"✅ Mechanism created, noise scale: {mechanism.noise_scale:.4f}")
        
        # Test noise addition
        tensor = torch.randn(10, 10)
        noisy_tensor = mechanism.add_noise(tensor)
        
        print(f"✅ Added noise to tensor")
        print(f"   Original std: {torch.std(tensor):.4f}")
        print(f"   Noisy std: {torch.std(noisy_tensor):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mechanism test failed: {e}")
        return False

def test_system_info():
    """Test system information functions."""
    print("🔍 Testing system information...")
    
    try:
        from dp_flash_attention.utils import cuda_version, privacy_check, check_system_requirements
        
        cuda_info = cuda_version()
        print(f"✅ CUDA info: {cuda_info.split()[0] if cuda_info else 'None'}")
        
        privacy_info = privacy_check()
        print(f"✅ Privacy check completed")
        
        sys_req = check_system_requirements()
        print(f"✅ System requirements checked: {len(sys_req)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ System info test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 DP-Flash-Attention Basic Tests")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_privacy_validation,
        test_noise_computation,
        test_accountant,
        test_mechanism,
        test_system_info,
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
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())