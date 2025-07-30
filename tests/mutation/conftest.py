"""
Configuration for mutation testing of DP-Flash-Attention.

Sets up mutation testing with privacy-aware test selection.
"""

import pytest
from typing import List, Dict, Any
import os
import sys

# Add src to path for mutation testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def pytest_configure(config):
    """Configure pytest for mutation testing."""
    # Add custom markers
    config.addinivalue_line("markers", "mutation: marks tests suitable for mutation testing")
    config.addinivalue_line("markers", "privacy_critical: marks privacy-critical tests")
    config.addinivalue_line("markers", "fast_mutation: marks fast tests for mutation")
    
    # Configure for mutation testing environment
    if os.environ.get('MUTATION_TESTING'):
        # Reduce output during mutation testing
        config.option.quiet = True
        config.option.disable_warnings = True


def pytest_collection_modifyitems(config, items):
    """Modify test collection for mutation testing."""
    if os.environ.get('MUTATION_TESTING'):
        # Filter to only mutation-suitable tests
        mutation_items = []
        for item in items:
            # Include tests marked for mutation or privacy-critical tests
            if (item.get_closest_marker("mutation") or 
                item.get_closest_marker("privacy_critical") or
                item.get_closest_marker("fast_mutation")):
                mutation_items.append(item)
        
        # If no specific mutation tests, include all unit tests
        if not mutation_items:
            mutation_items = [item for item in items if "unit" in str(item.fspath)]
        
        items[:] = mutation_items


@pytest.fixture(scope="session")
def mutation_config():
    """Configuration for mutation testing."""
    return {
        'max_mutations': 100,
        'timeout_per_test': 30,
        'privacy_critical_modules': [
            'dp_flash_attention.core',
            'dp_flash_attention.privacy',
            'dp_flash_attention.noise',
        ],
        'exclude_patterns': [
            '# pragma: no mutate',
            'assert False',  # Don't mutate assertion failures
            'raise NotImplementedError',  # Don't mutate not implemented
        ]
    }


@pytest.fixture
def privacy_test_data():
    """Generate test data for privacy-related mutations."""
    import torch
    
    return {
        'valid_epsilon': [0.1, 1.0, 5.0, 10.0],
        'invalid_epsilon': [-1.0, 0.0, float('inf'), float('nan')],
        'valid_delta': [1e-8, 1e-6, 1e-4, 1e-3],
        'invalid_delta': [-1e-5, 0.0, 1.0, float('inf')],
        'tensor_shapes': [(8, 128, 512), (16, 256, 768), (4, 512, 1024)],
        'devices': ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    }


class MutationTestHelper:
    """Helper class for mutation testing operations."""
    
    @staticmethod
    def is_privacy_critical_mutation(mutation_info: Dict[str, Any]) -> bool:
        """Check if a mutation affects privacy-critical code."""
        critical_patterns = [
            'epsilon',
            'delta', 
            'noise',
            'clip',
            'gradient',
            'privacy'
        ]
        
        code = mutation_info.get('original_code', '').lower()
        return any(pattern in code for pattern in critical_patterns)
    
    @staticmethod
    def mutation_severity(mutation_info: Dict[str, Any]) -> str:
        """Assess severity of a mutation for privacy guarantees."""
        if MutationTestHelper.is_privacy_critical_mutation(mutation_info):
            return 'HIGH'
        elif 'test' in mutation_info.get('file_path', '').lower():
            return 'LOW'
        else:
            return 'MEDIUM'
    
    @staticmethod
    def should_skip_mutation(mutation_info: Dict[str, Any]) -> bool:
        """Determine if a mutation should be skipped."""
        # Skip mutations in test files unless specifically marked
        if 'test' in mutation_info.get('file_path', ''):
            return True
            
        # Skip mutations in __init__.py files
        if '__init__.py' in mutation_info.get('file_path', ''):
            return True
            
        # Skip mutations in configuration/setup code
        setup_patterns = ['setup.py', 'conftest.py', '__main__']
        if any(pattern in mutation_info.get('file_path', '') for pattern in setup_patterns):
            return True
            
        return False


@pytest.fixture
def mutation_helper():
    """Provide mutation testing helper."""
    return MutationTestHelper()


# Mutation testing specific hooks
def pytest_runtest_setup(item):
    """Setup for mutation test runs."""
    if os.environ.get('MUTATION_TESTING'):
        # Set shorter timeouts for mutation testing
        if hasattr(item, 'timeout'):
            item.timeout = min(item.timeout, 10)


def pytest_runtest_teardown(item, nextitem):
    """Cleanup after mutation test runs."""
    if os.environ.get('MUTATION_TESTING'):
        # Clean up any GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Custom pytest markers for mutation testing
pytest_plugins = []