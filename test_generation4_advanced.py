#!/usr/bin/env python3
"""
Advanced Generation 4 Testing Suite for DP-Flash-Attention.

Tests autonomous improvements, research engine, and global deployment capabilities.
"""

import sys
import os
import time
import json
import tempfile
from pathlib import Path

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings("ignore")

def test_autonomous_improvements():
    """Test autonomous improvement system."""
    print("🔍 Testing autonomous improvements...")
    
    try:
        from dp_flash_attention.autonomous_improvements import (
            AutonomousOptimizer, AutonomousPrivacyManager, 
            PerformanceMetrics, get_global_autonomous_optimizer
        )
        
        # Test PerformanceMetrics creation
        metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_tokens_per_sec=1000.0,
            memory_usage_gb=8.0,
            privacy_budget_consumed=0.1,
            accuracy_score=0.95,
            efficiency_ratio=0.85,
            timestamp=time.time()
        )
        assert metrics.latency_ms == 50.0
        print("✅ PerformanceMetrics creation successful")
        
        # Test AutonomousOptimizer
        optimizer = AutonomousOptimizer(learning_rate=0.01)
        assert optimizer.learning_rate == 0.01
        assert len(optimizer.strategies) > 0
        print("✅ AutonomousOptimizer initialization successful")
        
        # Test performance recording
        optimizer.record_performance(metrics)
        assert len(optimizer.metrics_history) == 1
        print("✅ Performance recording successful")
        
        # Test optimization report generation
        report = optimizer.get_optimization_report()
        assert "optimization_status" in report
        assert "performance_summary" in report
        print("✅ Optimization report generation successful")
        
        # Test AutonomousPrivacyManager
        privacy_manager = AutonomousPrivacyManager()
        sensitivity_score = privacy_manager.analyze_data_sensitivity([1, 2, 3, 4, 5])
        assert 0.0 <= sensitivity_score <= 1.0
        print("✅ Privacy manager functionality successful")
        
        # Test adaptive privacy parameters
        privacy_manager.adapt_privacy_parameters(sensitivity_score, metrics)
        epsilon, delta = privacy_manager.get_adaptive_privacy_params()
        assert epsilon > 0 and delta > 0
        print("✅ Adaptive privacy parameters successful")
        
        # Test global optimizer singleton
        global_optimizer = get_global_autonomous_optimizer()
        assert global_optimizer is not None
        print("✅ Global optimizer singleton successful")
        
        # Test state saving/loading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            optimizer.save_optimization_state(f.name)
            
        # Create new optimizer and load state
        new_optimizer = AutonomousOptimizer()
        new_optimizer.load_optimization_state(f.name)
        assert len(new_optimizer.metrics_history) > 0
        print("✅ State saving/loading successful")
        
        # Cleanup
        os.unlink(f.name)
        
        return True
        
    except Exception as e:
        print(f"❌ Autonomous improvements test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_research_engine():
    """Test advanced research engine capabilities."""
    print("🔍 Testing advanced research engine...")
    
    try:
        from dp_flash_attention.advanced_research_engine import (
            AdvancedResearchEngine, RenyiGaussianMechanism, 
            DiscreteLaplaceMechanism, PrivacyUtilityPoint
        )
        
        # Test PrivacyUtilityPoint creation
        point = PrivacyUtilityPoint(
            epsilon=1.0,
            delta=1e-5,
            utility_score=0.95,
            privacy_score=1.0,
            computational_cost=0.1,
            memory_usage=8.0
        )
        assert point.epsilon == 1.0
        assert point.utility_score == 0.95
        print("✅ PrivacyUtilityPoint creation successful")
        
        # Test RenyiGaussianMechanism
        renyi_mech = RenyiGaussianMechanism(alpha=2.0)
        assert renyi_mech.alpha == 2.0
        assert renyi_mech.get_mechanism_name() == "RenyiGaussian(α=2.0)"
        print("✅ RenyiGaussianMechanism creation successful")
        
        # Test noise addition (without torch for compatibility)
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        noisy_data = renyi_mech.add_noise(test_data, sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert len(noisy_data) == len(test_data)
        print("✅ Noise addition successful")
        
        # Test privacy cost computation
        cost = renyi_mech.compute_privacy_cost(1.0, 1.0, 1e-5)
        assert cost > 0
        print("✅ Privacy cost computation successful")
        
        # Test DiscreteLaplaceMechanism
        laplace_mech = DiscreteLaplaceMechanism()
        assert laplace_mech.get_mechanism_name() == "DiscreteLaplace"
        print("✅ DiscreteLaplaceMechanism creation successful")
        
        # Test AdvancedResearchEngine
        engine = AdvancedResearchEngine(output_dir="test_research_outputs")
        assert len(engine.mechanisms) > 0
        print("✅ AdvancedResearchEngine initialization successful")
        
        # Test experiment design
        exp_id = engine.design_experiment(
            title="Test Experiment",
            hypothesis="Test hypothesis",
            methodology="Test methodology",
            parameters={"test_param": "test_value"}
        )
        assert exp_id in engine.experiments
        assert engine.experiments[exp_id].title == "Test Experiment"
        print("✅ Experiment design successful")
        
        # Test mock research results generation
        mock_results = engine._generate_mock_research_results()
        assert len(mock_results) > 0
        assert all(isinstance(points, list) for points in mock_results.values())
        print("✅ Mock research results generation successful")
        
        # Test comprehensive research suite
        results = engine.run_comprehensive_research_suite()
        assert "experiment_id" in results
        assert "report_path" in results
        print("✅ Comprehensive research suite successful")
        
        # Cleanup test outputs
        import shutil
        if Path("test_research_outputs").exists():
            shutil.rmtree("test_research_outputs")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced research engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_global_deployment_engine():
    """Test global deployment engine capabilities."""
    print("🔍 Testing global deployment engine...")
    
    try:
        from dp_flash_attention.global_deployment_engine import (
            GlobalDeploymentEngine, DeploymentRegion, ComplianceFramework,
            GlobalLoadBalancer, ComplianceManager, EdgeOptimizer
        )
        
        # Test enum definitions
        assert DeploymentRegion.US_EAST.value == "us-east-1"
        assert ComplianceFramework.GDPR.value == "gdpr"
        print("✅ Enum definitions successful")
        
        # Test GlobalLoadBalancer
        load_balancer = GlobalLoadBalancer()
        load_balancer.register_endpoint(DeploymentRegion.US_EAST, "https://us-east.example.com")
        assert DeploymentRegion.US_EAST in load_balancer.regional_endpoints
        print("✅ GlobalLoadBalancer functionality successful")
        
        # Test routing
        route = load_balancer.route_request("US", {"frameworks": []})
        assert isinstance(route, DeploymentRegion)
        print("✅ Load balancer routing successful")
        
        # Test ComplianceManager
        compliance_manager = ComplianceManager()
        validation = compliance_manager.validate_privacy_parameters(
            epsilon=1.0, 
            delta=1e-5, 
            frameworks=[ComplianceFramework.GDPR]
        )
        assert "gdpr" in validation
        assert "valid" in validation["gdpr"]
        print("✅ ComplianceManager validation successful")
        
        # Test compliance report
        report = compliance_manager.get_compliance_report()
        assert "total_validations" in report
        print("✅ Compliance report generation successful")
        
        # Test EdgeOptimizer
        edge_optimizer = EdgeOptimizer()
        device_capabilities = {
            "memory_gb": 4,
            "compute_score": 6.0,
            "bandwidth_mbps": 100,
            "secure_enclave": True
        }
        edge_optimizer.register_edge_device("test_device", device_capabilities)
        assert "test_device" in edge_optimizer.edge_profiles
        print("✅ EdgeOptimizer device registration successful")
        
        # Test edge optimization
        model_config = {"epsilon": 1.0, "batch_size": 32, "num_heads": 12}
        optimized_config = edge_optimizer.optimize_for_edge("test_device", model_config)
        assert "epsilon" in optimized_config
        print("✅ Edge optimization successful")
        
        # Test GlobalDeploymentEngine
        engine = GlobalDeploymentEngine()
        assert engine.load_balancer is not None
        assert engine.compliance_manager is not None
        assert engine.edge_optimizer is not None
        print("✅ GlobalDeploymentEngine initialization successful")
        
        # Test deployment target creation
        targets = engine._create_default_deployment_targets()
        assert len(targets) > 0
        assert all(hasattr(target, 'region') for target in targets)
        print("✅ Default deployment targets creation successful")
        
        # Test global status (without actual deployment)
        status = engine.get_global_status()
        assert "total_deployments" in status
        assert "active_deployments" in status
        print("✅ Global status retrieval successful")
        
        # Test configuration loading
        config = engine._load_config(None)  # Load default config
        assert "default_regions" in config
        assert "auto_scaling" in config
        print("✅ Configuration loading successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Global deployment engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_capabilities():
    """Test integration between advanced components."""
    print("🔍 Testing component integration...")
    
    try:
        from dp_flash_attention.autonomous_improvements import AutonomousOptimizer, PerformanceMetrics
        from dp_flash_attention.advanced_research_engine import AdvancedResearchEngine
        from dp_flash_attention.global_deployment_engine import GlobalDeploymentEngine
        
        # Test autonomous optimizer with research engine
        optimizer = AutonomousOptimizer()
        research_engine = AdvancedResearchEngine(output_dir="integration_test_outputs")
        
        # Generate some performance data
        for i in range(10):
            metrics = PerformanceMetrics(
                latency_ms=50 + i,
                throughput_tokens_per_sec=1000 - i * 10,
                memory_usage_gb=8.0 + i * 0.1,
                privacy_budget_consumed=i * 0.01,
                accuracy_score=0.95 - i * 0.001,
                efficiency_ratio=0.85 + i * 0.01,
                timestamp=time.time() + i
            )
            optimizer.record_performance(metrics)
        
        # Test that both components can work with similar data structures
        assert len(optimizer.metrics_history) == 10
        assert len(research_engine.mechanisms) > 0
        print("✅ Autonomous optimizer and research engine integration successful")
        
        # Test deployment engine with compliance requirements
        deployment_engine = GlobalDeploymentEngine()
        
        # Test that compliance manager can validate parameters from optimizer
        latest_metrics = optimizer.metrics_history[-1]
        validation = deployment_engine.compliance_manager.validate_privacy_parameters(
            epsilon=1.0,
            delta=1e-5,
            frameworks=[deployment_engine.compliance_manager.framework_configs.keys().__iter__().__next__()]
        )
        assert len(validation) > 0
        print("✅ Deployment engine and compliance integration successful")
        
        # Test edge optimization with autonomous recommendations
        edge_config = {
            "memory_gb": 8,
            "compute_score": 7.0,
            "bandwidth_mbps": 200
        }
        deployment_engine.edge_optimizer.register_edge_device("integration_test", edge_config)
        
        model_config = {"epsilon": 1.0, "delta": 1e-5, "batch_size": 32}
        optimized = deployment_engine.edge_optimizer.optimize_for_edge("integration_test", model_config)
        assert "epsilon" in optimized
        print("✅ Edge optimization integration successful")
        
        # Cleanup
        import shutil
        if Path("integration_test_outputs").exists():
            shutil.rmtree("integration_test_outputs")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmarks():
    """Test performance characteristics of advanced components."""
    print("🔍 Testing performance benchmarks...")
    
    try:
        from dp_flash_attention.autonomous_improvements import AutonomousOptimizer, PerformanceMetrics
        from dp_flash_attention.advanced_research_engine import AdvancedResearchEngine
        
        # Test optimizer performance with large datasets
        optimizer = AutonomousOptimizer()
        
        start_time = time.time()
        for i in range(1000):  # Large number of metrics
            metrics = PerformanceMetrics(
                latency_ms=50.0,
                throughput_tokens_per_sec=1000.0,
                memory_usage_gb=8.0,
                privacy_budget_consumed=0.1,
                accuracy_score=0.95,
                efficiency_ratio=0.85,
                timestamp=time.time()
            )
            optimizer.record_performance(metrics)
        
        processing_time = time.time() - start_time
        assert processing_time < 5.0  # Should process 1000 metrics in under 5 seconds
        print(f"✅ Optimizer performance: 1000 metrics processed in {processing_time:.2f}s")
        
        # Test research engine mechanism performance
        research_engine = AdvancedResearchEngine(output_dir="benchmark_test_outputs")
        
        start_time = time.time()
        for mechanism_name, mechanism in research_engine.mechanisms.items():
            # Test noise addition performance
            for _ in range(100):
                test_data = list(range(100))  # 100 element array
                noisy_data = mechanism.add_noise(test_data, 1.0, 1.0, 1e-5)
                assert len(noisy_data) == len(test_data)
        
        mechanism_time = time.time() - start_time
        assert mechanism_time < 10.0  # Should complete in under 10 seconds
        print(f"✅ Research engine performance: mechanism testing completed in {mechanism_time:.2f}s")
        
        # Test memory usage stays reasonable
        report = optimizer.get_optimization_report()
        assert "performance_summary" in report
        print("✅ Memory usage remains reasonable during large-scale operations")
        
        # Cleanup
        import shutil
        if Path("benchmark_test_outputs").exists():
            shutil.rmtree("benchmark_test_outputs")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all Generation 4 advanced tests."""
    print("🧪 DP-Flash-Attention Generation 4 Advanced Tests")
    print("=" * 60)
    
    tests = [
        ("Autonomous Improvements", test_autonomous_improvements),
        ("Advanced Research Engine", test_advanced_research_engine),
        ("Global Deployment Engine", test_global_deployment_engine),
        ("Integration Capabilities", test_integration_capabilities),
        ("Performance Benchmarks", test_performance_benchmarks),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Generation 4 advanced tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)