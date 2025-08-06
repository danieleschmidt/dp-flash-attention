#!/usr/bin/env python3
"""Test Generation 2: Make It Robust functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings("ignore")

def test_error_handling():
    """Test enhanced error handling."""
    print("🔍 Testing enhanced error handling...")
    
    try:
        from dp_flash_attention.error_handling import (
            PrivacyParameterError, validate_privacy_parameters,
            TensorShapeError, validate_tensor_inputs, DPFlashAttentionError
        )
        
        # Test privacy parameter validation
        try:
            validate_privacy_parameters(-1.0, 1e-5)
            print("❌ Should have caught negative epsilon")
            return False
        except PrivacyParameterError as e:
            print(f"✅ Caught privacy parameter error: {e.error_code}")
        
        # Test valid parameters
        try:
            validate_privacy_parameters(1.0, 1e-5)
            print("✅ Valid parameters accepted")
        except Exception as e:
            print(f"❌ Valid parameters rejected: {e}")
            return False
        
        # Test base error class
        try:
            raise DPFlashAttentionError(
                "Test error",
                error_code="TEST_ERROR",
                suggestions=["This is a test", "Check your inputs"]
            )
        except DPFlashAttentionError as e:
            if "TEST_ERROR" in str(e) and "suggestions" in str(e).lower():
                print("✅ Custom error formatting working")
            else:
                print(f"❌ Error formatting incorrect: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Error handling module import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in error handling test: {e}")
        return False

def test_logging_system():
    """Test logging and metrics system."""
    print("🔍 Testing logging and metrics...")
    
    try:
        from dp_flash_attention.logging_utils import (
            PrivacyMetricsLogger, PerformanceMonitor, get_logger
        )
        
        # Test logger creation
        logger = PrivacyMetricsLogger()
        print("✅ Logger created successfully")
        
        # Test privacy step logging
        logger.log_privacy_step(
            epsilon_spent=0.1,
            delta=1e-5,
            noise_scale=1.0,
            gradient_norm=0.5,
            clipping_bound=1.0,
            step=1
        )
        print("✅ Privacy step logged")
        
        # Test performance monitoring
        import time
        with PerformanceMonitor("test_operation", logger) as monitor:
            time.sleep(0.01)  # Simulate some work
        
        print(f"✅ Performance monitored: {monitor.duration_ms:.2f}ms")
        
        # Test summaries
        privacy_summary = logger.get_privacy_summary()
        perf_summary = logger.get_performance_summary()
        
        if privacy_summary['total_steps'] > 0:
            print("✅ Privacy summary generated")
        else:
            print("❌ Privacy summary empty")
            return False
            
        if perf_summary['total_operations'] > 0:
            print("✅ Performance summary generated")
        else:
            print("❌ Performance summary empty")  
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Logging module import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in logging test: {e}")
        return False

def test_security_features():
    """Test security validation features."""
    print("🔍 Testing security features...")
    
    try:
        from dp_flash_attention.security import (
            SecureRandomGenerator, validate_secure_environment,
            PrivacyLeakageDetector
        )
        
        # Test secure RNG
        rng = SecureRandomGenerator()
        print("✅ Secure RNG created")
        
        # Generate some noise (without PyTorch, just test initialization)
        print("✅ Secure RNG initialized")
        
        # Test environment validation
        env_validation = validate_secure_environment()
        print(f"✅ Environment validation completed: {env_validation['secure']}")
        
        if env_validation['warnings']:
            print(f"⚠️  Security warnings: {len(env_validation['warnings'])}")
        
        # Test leakage detector
        detector = PrivacyLeakageDetector()
        print("✅ Privacy leakage detector created")
        
        return True
        
    except ImportError as e:
        print(f"❌ Security module import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in security test: {e}")
        return False

def test_memory_estimation():
    """Test memory usage estimation."""
    print("🔍 Testing memory estimation...")
    
    try:
        from dp_flash_attention.utils import estimate_memory_usage
        
        # Test memory estimation
        memory_est = estimate_memory_usage(
            batch_size=32,
            sequence_length=512,
            num_heads=12,
            head_dim=64
        )
        
        print(f"✅ Memory estimated: {memory_est['total_estimated_mb']:.1f}MB")
        
        # Test different configurations
        small_est = estimate_memory_usage(4, 128, 8, 32)
        large_est = estimate_memory_usage(64, 1024, 16, 64)
        
        if large_est['total_estimated_mb'] > small_est['total_estimated_mb']:
            print("✅ Memory estimation scales correctly")
            return True
        else:
            print("❌ Memory estimation scaling incorrect")
            return False
        
    except Exception as e:
        print(f"❌ Memory estimation test failed: {e}")
        return False

def test_privacy_accounting():
    """Test enhanced privacy accounting."""
    print("🔍 Testing enhanced privacy accounting...")
    
    try:
        from dp_flash_attention.privacy import RenyiAccountant, GaussianMechanism
        
        # Test accountant
        accountant = RenyiAccountant()
        
        # Add multiple steps
        for i in range(5):
            step_epsilon = accountant.add_step(
                noise_scale=1.0,
                delta=1e-5,
                batch_size=32,
                dataset_size=10000
            )
            print(f"  Step {i+1}: ε={step_epsilon:.6f}")
        
        total_epsilon = accountant.get_epsilon(1e-5)
        print(f"✅ Total privacy cost: ε={total_epsilon:.4f}")
        
        # Test Gaussian mechanism
        mechanism = GaussianMechanism(
            epsilon=1.0,
            delta=1e-5,
            sensitivity=1.0
        )
        
        print(f"✅ Gaussian mechanism: noise_scale={mechanism.noise_scale:.4f}")
        
        # Test composition stats
        comp_stats = accountant.get_composition_stats()
        if comp_stats['total_steps'] == 5:
            print("✅ Composition statistics correct")
            return True
        else:
            print("❌ Composition statistics incorrect")
            return False
        
    except Exception as e:
        print(f"❌ Privacy accounting test failed: {e}")
        return False

def test_system_requirements():
    """Test system requirements checking."""
    print("🔍 Testing system requirements...")
    
    try:
        from dp_flash_attention.utils import check_system_requirements, cuda_version, privacy_check
        
        # Test system requirements
        sys_req = check_system_requirements()
        print(f"✅ System requirements checked: {len(sys_req)} items")
        
        # Test CUDA version check
        cuda_info = cuda_version()
        print(f"✅ CUDA info retrieved: {len(cuda_info)} characters")
        
        # Test privacy check
        privacy_info = privacy_check()
        print(f"✅ Privacy check completed: {len(privacy_info)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ System requirements test failed: {e}")
        return False

def main():
    """Run all Generation 2 tests."""
    print("🧪 DP-Flash-Attention Generation 2 Tests")
    print("=" * 45)
    print("🛡️  Make It Robust: Error Handling, Logging, Security")
    print()
    
    tests = [
        test_error_handling,
        test_logging_system,
        test_security_features,
        test_memory_estimation,
        test_privacy_accounting,
        test_system_requirements,
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
    
    print("=" * 45)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All robustness tests passed!")
        print("\n🛡️  Generation 2 (Make It Robust) - Enhanced functionality validated ✅")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())