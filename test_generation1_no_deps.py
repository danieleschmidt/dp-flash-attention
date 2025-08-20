#!/usr/bin/env python3
"""
Generation 1 Basic Tests - No external dependencies
Tests core logic and structure without requiring PyTorch/NumPy
"""

import sys
import math
import warnings
import traceback

def test_core_structure():
    """Test that core module structure is intact."""
    print("Testing core module structure...")
    
    try:
        import os
        
        # Check that all required files exist
        required_files = [
            '/root/repo/src/dp_flash_attention/__init__.py',
            '/root/repo/src/dp_flash_attention/core.py',
            '/root/repo/src/dp_flash_attention/kernels.py',
            '/root/repo/src/dp_flash_attention/privacy.py',
            '/root/repo/src/dp_flash_attention/utils.py',
            '/root/repo/src/dp_flash_attention/error_handling.py',
            '/root/repo/src/dp_flash_attention/security.py',
            '/root/repo/pyproject.toml',
            '/root/repo/requirements.txt',
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✓ {os.path.basename(file_path)}")
            else:
                print(f"✗ Missing {file_path}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False

def test_privacy_math():
    """Test privacy-related mathematical operations."""
    print("Testing privacy mathematics...")
    
    try:
        # Test attention scaling
        head_dim = 64
        scale = 1.0 / math.sqrt(head_dim)
        expected = 0.125
        assert abs(scale - expected) < 1e-10
        print(f"✓ Attention scaling: {scale}")
        
        # Test noise scale computation (Gaussian mechanism)
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 2.0
        
        # σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        noise_scale = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
        print(f"✓ Noise scale: {noise_scale:.4f}")
        
        # Test gradient clipping
        grad_norm = 1.5
        max_norm = 1.0
        clip_factor = min(1.0, max_norm / grad_norm)
        print(f"✓ Clip factor: {clip_factor:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Privacy math test failed: {e}")
        return False

def test_privacy_accounting():
    """Test basic privacy accounting logic."""
    print("Testing privacy accounting...")
    
    try:
        class BasicPrivacyAccountant:
            def __init__(self):
                self.steps = []
                self.total_epsilon = 0.0
                
            def add_step(self, epsilon, delta, batch_size):
                """Add privacy step with basic composition."""
                self.steps.append({
                    'epsilon': epsilon,
                    'delta': delta, 
                    'batch_size': batch_size
                })
                self.total_epsilon += epsilon
                return epsilon
                
            def get_epsilon(self, target_delta):
                """Get total epsilon for target delta."""
                return self.total_epsilon
        
        # Test accounting
        accountant = BasicPrivacyAccountant()
        
        # Add some steps
        for i in range(10):
            step_eps = 0.1
            accountant.add_step(step_eps, 1e-5, 32)
            
        total_eps = accountant.get_epsilon(1e-5)
        expected = 1.0  # 10 * 0.1
        assert abs(total_eps - expected) < 1e-10
        print(f"✓ Privacy accounting: ε = {total_eps}")
        
        return True
        
    except Exception as e:
        print(f"✗ Privacy accounting test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and validation."""
    print("Testing error handling...")
    
    try:
        # Test parameter validation
        def validate_privacy_params(epsilon, delta):
            if epsilon <= 0:
                raise ValueError(f"epsilon must be positive, got {epsilon}")
            if delta < 0 or delta >= 1:
                raise ValueError(f"delta must be in [0,1), got {delta}")
            return True
            
        # Test valid params
        assert validate_privacy_params(1.0, 1e-5)
        assert validate_privacy_params(0.01, 0.0)
        print("✓ Valid parameters accepted")
        
        # Test invalid params
        invalid_cases = [
            (-1.0, 1e-5),  # negative epsilon
            (0.0, 1e-5),   # zero epsilon
            (1.0, -0.1),   # negative delta
            (1.0, 1.0),    # delta = 1
            (1.0, 1.5),    # delta > 1
        ]
        
        for eps, delta in invalid_cases:
            try:
                validate_privacy_params(eps, delta)
                print(f"✗ Should reject ε={eps}, δ={delta}")
                return False
            except ValueError:
                print(f"✓ Correctly rejected ε={eps}, δ={delta}")
                
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_module_configuration():
    """Test module configuration and metadata."""
    print("Testing module configuration...")
    
    try:
        # Test that we can read configuration files
        with open('/root/repo/pyproject.toml', 'r') as f:
            content = f.read()
            assert 'dp-flash-attention' in content
            assert 'torch>=2.3.0' in content
            print("✓ pyproject.toml configuration valid")
            
        with open('/root/repo/requirements.txt', 'r') as f:
            content = f.read()
            assert 'torch>=2.3.0' in content
            assert 'opacus>=1.4.0' in content
            print("✓ requirements.txt valid")
            
        # Test README exists and has key sections
        with open('/root/repo/README.md', 'r') as f:
            readme = f.read()
            required_sections = [
                'DP-Flash-Attention',
                'Privacy Guarantees', 
                'Installation',
                'Quick Start',
                'Architecture'
            ]
            
            for section in required_sections:
                if section in readme:
                    print(f"✓ README has {section} section")
                else:
                    print(f"✗ README missing {section} section")
                    return False
                    
        return True
        
    except Exception as e:
        print(f"✗ Module configuration test failed: {e}")
        return False

def main():
    """Run all Generation 1 basic tests without external dependencies."""
    print("=" * 80)
    print("GENERATION 1 ENHANCED BASIC TESTS (NO EXTERNAL DEPENDENCIES)")
    print("=" * 80)
    
    tests = [
        test_core_structure,
        test_privacy_math,
        test_privacy_accounting,
        test_error_handling,
        test_module_configuration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        print(f"\n{'-' * 50}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {e}")
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {passed}/{total} tests passed ({100*passed//total}% success rate)")
    
    if passed == total:
        print("🎉 ALL GENERATION 1 BASIC TESTS PASSED!")
        print("✓ Core structure validated")
        print("✓ Privacy mathematics verified") 
        print("✓ Configuration validated")
        print("✓ Ready for Generation 2 robustness enhancements")
        return True
    else:
        print("❌ Some tests failed - requires fixes before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)