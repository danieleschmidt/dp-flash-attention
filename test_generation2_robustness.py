#!/usr/bin/env python3
"""
Test Generation 2 robustness features: comprehensive error handling, validation, and monitoring.
"""

import sys
import os
import warnings
import time
import json
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_error_handling():
    """Test comprehensive error handling system."""
    print("Testing error handling system...")
    
    try:
        from dp_flash_attention.error_handling import (
            DPFlashAttentionError, PrivacyParameterError, CUDACompatibilityError,
            TensorShapeError, validate_privacy_parameters, handle_errors,
            safe_divide, ErrorRecovery
        )
        
        # Test custom exception classes
        try:
            raise PrivacyParameterError("Test error", epsilon=-1.0, delta=0.5)
        except PrivacyParameterError as e:
            assert "PRIVACY_PARAM_ERROR" in str(e)
            assert "epsilon > 0" in str(e)
            print("✓ PrivacyParameterError works correctly")
        
        # Test privacy parameter validation
        try:
            validate_privacy_parameters(-1.0, 1e-5)
            assert False, "Should have raised error"
        except PrivacyParameterError:
            print("✓ Privacy parameter validation catches invalid epsilon")
        
        try:
            validate_privacy_parameters(1.0, 1.5)
            assert False, "Should have raised error"
        except PrivacyParameterError:
            print("✓ Privacy parameter validation catches invalid delta")
        
        # Test valid parameters
        validate_privacy_parameters(1.0, 1e-5)
        print("✓ Privacy parameter validation accepts valid parameters")
        
        # Test safe division
        result = safe_divide(10, 2)
        assert result == 5.0
        
        result = safe_divide(10, 0, default=999)
        assert result == 999
        print("✓ Safe division works correctly")
        
        return True
        
    except ImportError as e:
        print(f"✗ Error handling test failed due to missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_logging_system():
    """Test logging and monitoring system."""
    print("Testing logging system...")
    
    try:
        from dp_flash_attention.logging_utils import (
            PrivacyMetricsLogger, PerformanceMonitor, get_logger,
            PrivacyAwareFormatter
        )
        
        # Test privacy metrics logger
        logger = PrivacyMetricsLogger()
        
        # Log some privacy steps
        logger.log_privacy_step(
            epsilon_spent=0.1,
            delta=1e-5,
            noise_scale=0.5,
            gradient_norm=1.2,
            clipping_bound=1.0,
            additional_info={'batch_size': 32}
        )
        
        logger.log_performance_metrics(
            operation="test_operation",
            duration_ms=123.45,
            memory_usage_mb=256.0,
            batch_size=32
        )
        
        logger.log_security_event(
            event_type="validation_warning",
            severity="medium",
            description="Test security event"
        )
        
        # Test summaries
        privacy_summary = logger.get_privacy_summary()
        assert privacy_summary['total_steps'] == 1
        assert privacy_summary['total_epsilon_consumed'] == 0.1
        print("✓ Privacy metrics logging works")
        
        performance_summary = logger.get_performance_summary()
        assert performance_summary['total_operations'] == 1
        print("✓ Performance metrics logging works")
        
        # Test performance monitor
        with PerformanceMonitor("test_operation", logger) as monitor:
            time.sleep(0.01)  # Small delay
        
        assert monitor.duration_ms > 5  # Should be at least a few ms
        print("✓ Performance monitoring works")
        
        return True
        
    except ImportError as e:
        print(f"✗ Logging test failed due to missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

def test_security_system():
    """Test security validation and random generation."""
    print("Testing security system...")
    
    try:
        from dp_flash_attention.security import (
            SecureRandomGenerator, PrivacyLeakageDetector,
            validate_secure_environment, create_secure_hash,
            get_secure_rng, get_privacy_auditor
        )
        
        # Test secure random generator (without PyTorch dependency)
        rng = SecureRandomGenerator(seed=42)  # Use seed for testing
        
        # Test environment validation
        env_validation = validate_secure_environment()
        assert isinstance(env_validation, dict)
        assert 'secure' in env_validation
        assert 'warnings' in env_validation
        print("✓ Security environment validation works")
        
        # Test secure hash
        hash1 = create_secure_hash("test string")
        hash2 = create_secure_hash("test string")
        hash3 = create_secure_hash("different string")
        
        assert hash1 == hash2  # Same input should give same hash
        assert hash1 != hash3  # Different input should give different hash
        assert len(hash1) == 64  # SHA256 produces 64 hex chars
        print("✓ Secure hashing works")
        
        # Test global instances
        secure_rng = get_secure_rng()
        assert isinstance(secure_rng, SecureRandomGenerator)
        
        auditor = get_privacy_auditor()
        audit_result = auditor.audit_privacy_step(
            epsilon_spent=0.1,
            delta=1e-5,
            noise_scale=0.5,
            gradient_norm=0.8,
            clipping_bound=1.0
        )
        assert isinstance(audit_result, dict)
        assert 'issues' in audit_result
        print("✓ Privacy auditor works")
        
        return True
        
    except ImportError as e:
        print(f"✗ Security test failed due to missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"✗ Security test failed: {e}")
        return False

def main():
    """Run all Generation 2 robustness tests."""
    print("🛡️  Testing DP-Flash-Attention Generation 2: ROBUSTNESS")
    print("=" * 60)
    
    tests = [
        test_error_handling,
        test_logging_system,
        test_security_system,
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n[{i}/{total}] {test_func.__name__.replace('test_', '').replace('_', ' ').title()}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Generation 2 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Generation 2 robustness tests passed!")
        print("✅ Error handling, logging, security, and validation systems are working correctly")
        return 0
    else:
        print("❌ Some Generation 2 tests failed.")
        print("🔧 Review error handling, logging, or security implementations")
        return 1

if __name__ == "__main__":
    sys.exit(main())
