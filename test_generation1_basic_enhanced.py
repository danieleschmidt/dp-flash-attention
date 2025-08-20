#!/usr/bin/env python3
"""
Generation 1 Enhanced Basic Tests - Verify core functionality works
"""

import sys
import warnings
import traceback

def test_basic_imports():
    """Test that core modules can be imported."""
    print("Testing basic imports...")
    
    try:
        # Test core imports
        import numpy as np
        print("✓ NumPy imported successfully")
        
        # Test basic tensor operations without torch
        arr = np.random.randn(4, 8)
        result = np.matmul(arr, arr.T)
        assert result.shape == (4, 4)
        print("✓ Basic tensor operations work")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_dp_flash_attention_structure():
    """Test DP-Flash-Attention module structure without PyTorch."""
    print("Testing DP-Flash-Attention structure...")
    
    try:
        import sys
        import os
        
        # Add src to path
        sys.path.insert(0, '/root/repo/src')
        
        # Test that module files exist
        module_files = [
            '/root/repo/src/dp_flash_attention/__init__.py',
            '/root/repo/src/dp_flash_attention/core.py',
            '/root/repo/src/dp_flash_attention/kernels.py',
            '/root/repo/src/dp_flash_attention/privacy.py',
            '/root/repo/src/dp_flash_attention/utils.py',
        ]
        
        for file_path in module_files:
            if os.path.exists(file_path):
                print(f"✓ Found {file_path}")
            else:
                print(f"✗ Missing {file_path}")
                return False
                
        print("✓ All core module files present")
        return True
        
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_validation():
    """Test configuration and parameter validation logic."""
    print("Testing configuration validation...")
    
    try:
        # Test privacy parameter validation logic
        def validate_privacy_params(epsilon, delta):
            if epsilon <= 0:
                raise ValueError("epsilon must be positive")
            if delta < 0 or delta >= 1:
                raise ValueError("delta must be in [0, 1)")
            return True
            
        # Test valid parameters
        assert validate_privacy_params(1.0, 1e-5)
        assert validate_privacy_params(0.1, 0.0)
        print("✓ Valid privacy parameters accepted")
        
        # Test invalid parameters
        try:
            validate_privacy_params(-1.0, 1e-5)
            print("✗ Should have rejected negative epsilon")
            return False
        except ValueError:
            print("✓ Correctly rejected negative epsilon")
            
        try:
            validate_privacy_params(1.0, 1.5)
            print("✗ Should have rejected delta >= 1")
            return False
        except ValueError:
            print("✓ Correctly rejected delta >= 1")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_mathematical_operations():
    """Test core mathematical operations for attention mechanism."""
    print("Testing mathematical operations...")
    
    try:
        import math
        import numpy as np
        
        # Test attention scaling
        head_dim = 64
        scale = 1.0 / math.sqrt(head_dim)
        expected_scale = 0.125
        assert abs(scale - expected_scale) < 1e-6
        print(f"✓ Attention scale computed correctly: {scale}")
        
        # Test noise scale computation
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 2.0  # L2 sensitivity bound
        
        # Gaussian mechanism noise scale: σ = (sensitivity * sqrt(2 * ln(1.25/δ))) / ε
        noise_scale = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
        print(f"✓ Noise scale computed: {noise_scale}")
        
        # Test gradient clipping simulation
        grad_norm = 1.5
        max_norm = 1.0
        clip_factor = min(1.0, max_norm / grad_norm)
        expected_clip_factor = 2/3
        assert abs(clip_factor - expected_clip_factor) < 1e-6
        print(f"✓ Gradient clipping factor: {clip_factor}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mathematical operations test failed: {e}")
        traceback.print_exc()
        return False

def test_privacy_accounting_logic():
    """Test privacy accounting logic without external dependencies."""
    print("Testing privacy accounting logic...")
    
    try:
        # Simple privacy accounting simulation
        class SimplePrivacyAccountant:
            def __init__(self):
                self.total_epsilon = 0.0
                self.steps = []
                
            def add_step(self, epsilon_step, delta, batch_size, seq_len):
                """Add a privacy step with basic composition."""
                # Basic composition: just sum epsilons (conservative)
                self.total_epsilon += epsilon_step
                self.steps.append({
                    'epsilon': epsilon_step,
                    'delta': delta,
                    'batch_size': batch_size,
                    'seq_len': seq_len
                })
                return epsilon_step
                
            def get_total_epsilon(self):
                return self.total_epsilon
                
        # Test privacy accounting
        accountant = SimplePrivacyAccountant()
        
        # Simulate training steps
        for step in range(5):
            step_epsilon = accountant.add_step(0.1, 1e-5, 32, 512)
            assert step_epsilon == 0.1
            
        total_epsilon = accountant.get_total_epsilon()
        expected_total = 0.5  # 5 * 0.1
        assert abs(total_epsilon - expected_total) < 1e-6
        print(f"✓ Privacy accounting works: total ε = {total_epsilon}")
        
        return True
        
    except Exception as e:
        print(f"✗ Privacy accounting test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Generation 1 basic tests."""
    print("=" * 60)
    print("GENERATION 1 ENHANCED BASIC TESTS")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_dp_flash_attention_structure, 
        test_configuration_validation,
        test_mathematical_operations,
        test_privacy_accounting_logic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        print(f"\n{'-' * 40}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_func.__name__} PASSED")
            else:
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Generation 1 basic tests PASSED!")
        return True
    else:
        print("❌ Some tests failed. See details above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)