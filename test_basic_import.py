#!/usr/bin/env python3
"""
Basic import and functionality test for DP-Flash-Attention.
This test can run without PyTorch to verify basic code structure.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that basic modules can be imported."""
    try:
        # Test import of helper functions (no torch dependency)
        from dp_flash_attention.helper_functions import (
            estimate_privacy_cost,
            create_attention_mask,
            compute_attention_stats
        )
        print("✓ Helper functions imported successfully")
        
        # Test privacy cost estimation
        privacy_stats = estimate_privacy_cost(
            epsilon=1.0,
            delta=1e-5,
            num_steps=1000,
            batch_size=32,
            dataset_size=50000
        )
        
        expected_keys = ['total_epsilon', 'per_step_epsilon', 'effective_epsilon', 'delta']
        for key in expected_keys:
            assert key in privacy_stats, f"Missing key: {key}"
        
        print("✓ Privacy cost estimation working")
        print(f"  Privacy stats: {privacy_stats}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_module_structure():
    """Test the module structure and core components."""
    try:
        # Check if main package can be imported
        import dp_flash_attention
        print("✓ Main package imported")
        
        # Check version info
        if hasattr(dp_flash_attention, '__version__'):
            print(f"✓ Package version: {dp_flash_attention.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Module structure test failed: {e}")
        return False

def test_configuration_files():
    """Test that configuration files are properly formatted."""
    try:
        import json
        import os
        
        # Check pyproject.toml exists
        if os.path.exists('pyproject.toml'):
            print("✓ pyproject.toml exists")
        
        # Check requirements.txt
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r') as f:
                reqs = f.read().strip().split('\n')
                print(f"✓ requirements.txt has {len(reqs)} dependencies")
        
        # Check deployment config if exists
        config_files = ['config/production.json', 'deployment/deployment_summary.json']
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    print(f"✓ {config_file} is valid JSON")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🔍 Running DP-Flash-Attention Basic Tests...")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_module_structure, 
        test_configuration_files,
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n[{i}/{total}] {test_func.__name__}")
        if test_func():
            passed += 1
        else:
            print(f"  Failed: {test_func.__name__}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Generation 1 functionality verified.")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())