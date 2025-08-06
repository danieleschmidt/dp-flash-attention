#!/usr/bin/env python3
"""
Generation 1 Basic Functionality Tests for DP-Flash-Attention.

Tests core functionality to ensure the system works end-to-end.
"""

import sys
import os
import traceback
import warnings
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dp_flash_attention import (
        DPFlashAttention,
        dp_flash_attn_func,
        validate_privacy_params,
        compute_noise_scale,
        cuda_version,
        privacy_check,
        RenyiAccountant,
        GaussianMechanism,
        PrivacyStats
    )
    from dp_flash_attention.validation import (
        validate_privacy_parameters_comprehensive,
        validate_system_requirements_comprehensive
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    traceback.print_exc()
    sys.exit(1)


def test_basic_imports() -> Dict[str, Any]:
    """Test that all basic imports work."""
    results = {'passed': True, 'errors': [], 'info': []}
    
    try:
        # Test core imports
        from dp_flash_attention.core import DPFlashAttention
        from dp_flash_attention.functional import dp_flash_attn_func
        from dp_flash_attention.privacy import RenyiAccountant, GaussianMechanism
        from dp_flash_attention.utils import validate_privacy_params
        from dp_flash_attention.kernels import dp_flash_attention_kernel
        
        results['info'].append("✅ All core imports successful")
        
    except ImportError as e:
        results['passed'] = False
        results['errors'].append(f"Import error: {e}")
    
    return results


def test_privacy_parameter_validation() -> Dict[str, Any]:
    """Test privacy parameter validation."""
    results = {'passed': True, 'errors': [], 'info': []}
    
    try:
        # Valid parameters should pass
        validate_privacy_params(1.0, 1e-5)
        results['info'].append("✅ Valid privacy parameters accepted")
        
        # Test comprehensive validation
        validation_result = validate_privacy_parameters_comprehensive(1.0, 1e-5, 1.0)
        if validation_result['epsilon_valid'] and validation_result['delta_valid']:
            results['info'].append("✅ Comprehensive validation working")
        else:
            results['errors'].append("Comprehensive validation failed for valid params")
        
        # Invalid parameters should raise errors
        invalid_cases = [
            (-1.0, 1e-5),  # Negative epsilon
            (1.0, 0.0),    # Zero delta
            (1.0, 1.0),    # Delta = 1.0
        ]
        
        for eps, delta in invalid_cases:
            try:
                validate_privacy_params(eps, delta)
                results['errors'].append(f"Should have failed for epsilon={eps}, delta={delta}")
                results['passed'] = False
            except ValueError:
                pass  # Expected
                
        results['info'].append("✅ Invalid privacy parameters properly rejected")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Privacy validation error: {e}")
    
    return results


def test_noise_scale_computation() -> Dict[str, Any]:
    """Test noise scale computation."""
    results = {'passed': True, 'errors': [], 'info': []}
    
    try:
        # Test basic noise scale computation
        noise_scale = compute_noise_scale(
            epsilon=1.0,
            delta=1e-5, 
            max_grad_norm=1.0,
            sequence_length=512
        )
        
        if noise_scale > 0:
            results['info'].append(f"✅ Noise scale computed: {noise_scale:.4f}")
        else:
            results['errors'].append("Noise scale should be positive")
            results['passed'] = False
        
        # Test that larger epsilon leads to smaller noise
        noise_scale_large_eps = compute_noise_scale(10.0, 1e-5, 1.0, 512)
        if noise_scale_large_eps < noise_scale:
            results['info'].append("✅ Larger epsilon produces smaller noise (correct)")
        else:
            results['errors'].append("Larger epsilon should produce smaller noise")
            results['passed'] = False
            
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Noise scale computation error: {e}")
    
    return results


def test_dp_attention_basic() -> Dict[str, Any]:
    """Test basic DP attention functionality."""
    results = {'passed': True, 'errors': [], 'info': []}
    
    try:
        # Create DP attention layer
        dp_attn = DPFlashAttention(
            embed_dim=256,
            num_heads=8,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            batch_first=True
        )
        
        results['info'].append("✅ DP attention layer created")
        
        # Test forward pass with CPU tensors
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, 256)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore CUDA warnings for CPU test
            output = dp_attn(x, x, x)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, 256)
        if output.shape == expected_shape:
            results['info'].append(f"✅ Output shape correct: {output.shape}")
        else:
            results['errors'].append(f"Output shape mismatch: {output.shape} != {expected_shape}")
            results['passed'] = False
        
        # Test with privacy stats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output, privacy_stats = dp_attn(x, x, x, return_privacy_stats=True)
        
        if isinstance(privacy_stats, PrivacyStats):
            results['info'].append("✅ Privacy statistics returned")
            results['info'].append(f"  - Epsilon spent: {privacy_stats.epsilon_spent:.6f}")
            results['info'].append(f"  - Noise scale: {privacy_stats.noise_scale:.6f}")
        else:
            results['errors'].append("Privacy statistics not returned correctly")
            results['passed'] = False
            
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"DP attention test error: {e}")
        traceback.print_exc()
    
    return results


def test_functional_interface() -> Dict[str, Any]:
    """Test functional interface."""
    results = {'passed': True, 'errors': [], 'info': []}
    
    try:
        # Create test tensors
        batch_size, seq_len, num_heads, head_dim = 2, 32, 4, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        # Test functional interface
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = dp_flash_attn_func(
                q, k, v,
                epsilon=1.0,
                delta=1e-5,
                max_grad_norm=1.0,
                causal=False
            )
        
        expected_shape = (batch_size, seq_len, num_heads, head_dim)
        if output.shape == expected_shape:
            results['info'].append(f"✅ Functional interface output shape correct: {output.shape}")
        else:
            results['errors'].append(f"Functional output shape mismatch: {output.shape} != {expected_shape}")
            results['passed'] = False
        
        # Test causal attention
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            causal_output = dp_flash_attn_func(
                q, k, v,
                epsilon=1.0,
                delta=1e-5,
                max_grad_norm=1.0,
                causal=True
            )
        
        if causal_output.shape == expected_shape:
            results['info'].append("✅ Causal attention working")
        else:
            results['errors'].append("Causal attention failed")
            results['passed'] = False
            
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Functional interface error: {e}")
        traceback.print_exc()
    
    return results


def test_privacy_accounting() -> Dict[str, Any]:
    """Test privacy accounting functionality."""
    results = {'passed': True, 'errors': [], 'info': []}
    
    try:
        # Test RenyiAccountant
        accountant = RenyiAccountant()
        results['info'].append("✅ Privacy accountant created")
        
        # Add some privacy steps
        epsilon_used = accountant.add_step(
            noise_scale=1.0,
            delta=1e-5,
            batch_size=32,
            dataset_size=50000
        )
        
        if epsilon_used > 0:
            results['info'].append(f"✅ Privacy step added, epsilon used: {epsilon_used:.6f}")
        else:
            results['errors'].append("Privacy step should consume positive epsilon")
            results['passed'] = False
        
        # Get total privacy cost
        total_epsilon = accountant.get_epsilon(delta=1e-5)
        if total_epsilon >= epsilon_used:
            results['info'].append(f"✅ Total epsilon computed: {total_epsilon:.6f}")
        else:
            results['errors'].append("Total epsilon should be >= step epsilon")
            results['passed'] = False
        
        # Test Gaussian mechanism
        mechanism = GaussianMechanism(
            epsilon=1.0,
            delta=1e-5,
            sensitivity=1.0
        )
        
        test_tensor = torch.randn(100, 100)
        noisy_tensor = mechanism.add_noise(test_tensor)
        
        if noisy_tensor.shape == test_tensor.shape:
            results['info'].append("✅ Gaussian mechanism working")
            
            # Check that noise was actually added
            if not torch.equal(test_tensor, noisy_tensor):
                results['info'].append("✅ Noise properly added to tensor")
            else:
                results['errors'].append("No noise was added")
                results['passed'] = False
        else:
            results['errors'].append("Gaussian mechanism changed tensor shape")
            results['passed'] = False
            
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Privacy accounting error: {e}")
        traceback.print_exc()
    
    return results


def test_system_compatibility() -> Dict[str, Any]:
    """Test system compatibility and requirements."""
    results = {'passed': True, 'errors': [], 'info': []}
    
    try:
        # Test CUDA version check
        cuda_info = cuda_version()
        results['info'].append(f"✅ CUDA info: {cuda_info.split()[0]}")
        
        # Test privacy check
        privacy_info = privacy_check()
        results['info'].append("✅ Privacy check completed")
        
        # Count successful checks
        success_count = privacy_info.count('✓')
        warning_count = privacy_info.count('⚠')
        error_count = privacy_info.count('✗')
        
        results['info'].append(f"  - Successful: {success_count}")
        results['info'].append(f"  - Warnings: {warning_count}")
        results['info'].append(f"  - Errors: {error_count}")
        
        # Test comprehensive system requirements
        try:
            sys_requirements = validate_system_requirements_comprehensive()
            if sys_requirements['requirements_met']:
                results['info'].append("✅ All critical system requirements met")
            else:
                results['info'].append("⚠️ Some system requirements not met (non-critical)")
                for error in sys_requirements['errors']:
                    results['info'].append(f"  - {error}")
        except Exception as e:
            results['info'].append(f"⚠️ System requirements check failed: {e}")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"System compatibility error: {e}")
    
    return results


def test_cuda_optimization() -> Dict[str, Any]:
    """Test CUDA optimization if available."""
    results = {'passed': True, 'errors': [], 'info': []}
    
    if not torch.cuda.is_available():
        results['info'].append("⚠️ CUDA not available, skipping CUDA tests")
        return results
    
    try:
        # Test CUDA DP attention
        dp_attn = DPFlashAttention(
            embed_dim=256,
            num_heads=8,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
        ).cuda()
        
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, 256, device='cuda')
        
        # Run forward pass
        output = dp_attn(x, x, x)
        
        if output.device.type == 'cuda':
            results['info'].append("✅ CUDA forward pass successful")
        else:
            results['errors'].append("Output not on CUDA device")
            results['passed'] = False
        
        # Test memory efficiency
        memory_before = torch.cuda.memory_allocated()
        
        # Run multiple forward passes
        for _ in range(10):
            output = dp_attn(x, x, x)
        
        memory_after = torch.cuda.memory_allocated()
        memory_increase = (memory_after - memory_before) / (1024**2)  # MB
        
        if memory_increase < 100:  # Less than 100MB increase is reasonable
            results['info'].append(f"✅ Memory usage stable (increase: {memory_increase:.1f}MB)")
        else:
            results['info'].append(f"⚠️ High memory usage increase: {memory_increase:.1f}MB")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"CUDA test error: {e}")
        traceback.print_exc()
    
    return results


def run_all_tests() -> None:
    """Run all Generation 1 tests."""
    print("🧪 Running Generation 1 Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Privacy Parameter Validation", test_privacy_parameter_validation),
        ("Noise Scale Computation", test_noise_scale_computation),
        ("DP Attention Basic", test_dp_attention_basic),
        ("Functional Interface", test_functional_interface), 
        ("Privacy Accounting", test_privacy_accounting),
        ("System Compatibility", test_system_compatibility),
        ("CUDA Optimization", test_cuda_optimization),
    ]
    
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            
            if result['passed']:
                print(f"✅ {test_name}: PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name}: FAILED")
                for error in result['errors']:
                    print(f"   Error: {error}")
            
            # Print info messages
            for info in result['info']:
                print(f"   {info}")
                
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Generation 1 functionality is working.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)