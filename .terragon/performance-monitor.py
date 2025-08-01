#!/usr/bin/env python3
"""
Advanced Performance Monitoring and Optimization Discovery
Identifies GPU performance regressions and optimization opportunities
"""

import json
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PerformanceMonitor:
    """Advanced performance monitoring for CUDA/ML workloads"""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.metrics_path = self.repo_path / ".terragon" / "performance-metrics.json"
        
    def analyze_cuda_kernels(self) -> List[Dict]:
        """Analyze CUDA kernel implementations for optimization opportunities"""
        opportunities = []
        
        # Find CUDA files
        cuda_files = list(self.repo_path.glob("**/*.cu")) + list(self.repo_path.glob("**/*.cuh"))
        
        for cuda_file in cuda_files:
            try:
                with open(cuda_file, 'r') as f:
                    content = f.read()
                    
                # Check for common optimization patterns
                issues = []
                
                # Memory coalescing opportunities
                if re.search(r'threadIdx\.x.*\*.*sizeof', content):
                    issues.append({
                        'type': 'memory_coalescing',
                        'description': 'Potential non-coalesced memory access pattern',
                        'impact': 'high',
                        'effort': 'medium'
                    })
                
                # Shared memory usage
                if '__shared__' not in content and 'blockDim' in content:
                    issues.append({
                        'type': 'shared_memory',
                        'description': 'Could benefit from shared memory optimization',
                        'impact': 'medium', 
                        'effort': 'high'
                    })
                
                # Register pressure
                register_usage = len(re.findall(r'(float|int|double)\\s+\\w+', content))
                if register_usage > 20:
                    issues.append({
                        'type': 'register_pressure',
                        'description': f'High register usage ({register_usage} variables)',
                        'impact': 'medium',
                        'effort': 'high'
                    })
                
                # Divergent branches
                if re.search(r'if.*threadIdx', content):
                    issues.append({
                        'type': 'thread_divergence',
                        'description': 'Potential warp divergence from thread-dependent branches',
                        'impact': 'medium',
                        'effort': 'medium'
                    })
                
                if issues:
                    opportunities.append({
                        'file': str(cuda_file.relative_to(self.repo_path)),
                        'issues': issues,
                        'priority_score': self._calculate_cuda_priority(issues)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not analyze {cuda_file}: {e}")
        
        return opportunities
    
    def analyze_python_performance(self) -> List[Dict]:
        """Analyze Python code for performance bottlenecks"""
        opportunities = []
        
        python_files = list(self.repo_path.glob("src/**/*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                issues = []
                
                # Inefficient loops
                if re.search(r'for.*in.*range.*len\\(', content):
                    issues.append({
                        'type': 'inefficient_iteration',
                        'description': 'Using range(len()) instead of direct iteration',
                        'impact': 'low',
                        'effort': 'low'
                    })
                
                # NumPy optimizations
                if 'numpy' in content or 'np.' in content:
                    if re.search(r'for.*in.*array', content):
                        issues.append({
                            'type': 'vectorization',
                            'description': 'Loop over NumPy array could be vectorized',
                            'impact': 'high',
                            'effort': 'medium'
                        })
                
                # GPU memory transfers
                if re.search(r'\\.to\\(["\']cuda["\']\\)', content):
                    transfer_count = len(re.findall(r'\\.to\\(["\']cuda["\']\\)', content))
                    if transfer_count > 3:
                        issues.append({
                            'type': 'gpu_transfers', 
                            'description': f'Excessive GPU memory transfers ({transfer_count})',
                            'impact': 'high',
                            'effort': 'medium'
                        })
                
                # Synchronous operations
                if 'torch.cuda.synchronize()' in content:
                    issues.append({
                        'type': 'synchronization',
                        'description': 'Explicit CUDA synchronization may impact performance',
                        'impact': 'medium',
                        'effort': 'low'
                    })
                
                if issues:
                    opportunities.append({
                        'file': str(py_file.relative_to(self.repo_path)),
                        'issues': issues,
                        'priority_score': self._calculate_python_priority(issues)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not analyze {py_file}: {e}")
        
        return opportunities
    
    def check_benchmark_regressions(self) -> List[Dict]:
        """Check for potential performance regressions in benchmarks"""
        regressions = []
        
        # Look for benchmark files
        benchmark_files = (
            list(self.repo_path.glob("**/benchmark*.py")) +
            list(self.repo_path.glob("**/test_*performance*.py")) +
            list(self.repo_path.glob("**/perf_*.py"))
        )
        
        for bench_file in benchmark_files:
            try:
                with open(bench_file, 'r') as f:
                    content = f.read()
                    
                # Check for hardcoded performance expectations
                expectations = re.findall(r'assert.*<.*(\d+\.?\d*)', content)
                if expectations:
                    regressions.append({
                        'file': str(bench_file.relative_to(self.repo_path)),
                        'type': 'hardcoded_expectations',
                        'description': 'Benchmark has hardcoded performance expectations',
                        'expectations': expectations,
                        'recommendation': 'Use dynamic baselines or tolerance ranges'
                    })
                
                # Check for missing timing measurements
                if 'time.time()' not in content and 'perf_counter' not in content:
                    if 'def test_' in content or 'def benchmark_' in content:
                        regressions.append({
                            'file': str(bench_file.relative_to(self.repo_path)),
                            'type': 'missing_timing',
                            'description': 'Benchmark file missing timing measurements',
                            'recommendation': 'Add timing measurements for performance tracking'
                        })
                        
            except Exception as e:
                print(f"Warning: Could not analyze {bench_file}: {e}")
        
        return regressions
    
    def analyze_memory_usage(self) -> List[Dict]:
        """Analyze potential memory optimization opportunities"""
        opportunities = []
        
        # Check for memory-intensive operations
        py_files = list(self.repo_path.glob("src/**/*.py"))
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                issues = []
                
                # Large tensor allocations
                tensor_allocs = re.findall(r'torch\\.\\w*\\([^)]*\\)', content)
                large_allocs = [alloc for alloc in tensor_allocs if any(
                    size in alloc for size in ['1024', '2048', '4096', '8192']
                )]
                
                if large_allocs:
                    issues.append({
                        'type': 'large_allocations',
                        'description': f'Large tensor allocations found ({len(large_allocs)})',
                        'impact': 'high',
                        'effort': 'medium'
                    })
                
                # Memory leaks potential
                if 'del ' not in content and 'torch.cuda.empty_cache()' not in content:
                    if 'cuda' in content:
                        issues.append({
                            'type': 'memory_cleanup',  
                            'description': 'No explicit GPU memory cleanup',
                            'impact': 'medium',
                            'effort': 'low'
                        })
                
                # In-place operations
                inplace_ops = len(re.findall(r'\\w+_\\(', content))  # Operations ending with _
                total_ops = len(re.findall(r'torch\\.\\w+\\(', content))
                
                if total_ops > 0 and inplace_ops / total_ops < 0.3:
                    issues.append({
                        'type': 'inplace_optimization',
                        'description': f'Could use more in-place operations ({inplace_ops}/{total_ops})',
                        'impact': 'medium',
                        'effort': 'low'
                    })
                
                if issues:
                    opportunities.append({
                        'file': str(py_file.relative_to(self.repo_path)),
                        'issues': issues,
                        'priority_score': self._calculate_memory_priority(issues)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not analyze {py_file}: {e}")
        
        return opportunities
    
    def _calculate_cuda_priority(self, issues: List[Dict]) -> float:
        """Calculate priority score for CUDA optimizations"""
        impact_weights = {'high': 3.0, 'medium': 2.0, 'low': 1.0}
        effort_weights = {'low': 3.0, 'medium': 2.0, 'high': 1.0}
        
        total_score = 0
        for issue in issues:
            impact = impact_weights.get(issue['impact'], 1.0)
            effort = effort_weights.get(issue['effort'], 1.0)
            total_score += impact * effort
        
        return total_score
    
    def _calculate_python_priority(self, issues: List[Dict]) -> float:
        """Calculate priority score for Python optimizations"""
        return self._calculate_cuda_priority(issues)  # Same logic
    
    def _calculate_memory_priority(self, issues: List[Dict]) -> float:
        """Calculate priority score for memory optimizations"""
        return self._calculate_cuda_priority(issues)  # Same logic
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance analysis report"""
        print("🔍 Analyzing CUDA kernels...")
        cuda_opportunities = self.analyze_cuda_kernels()
        
        print("🐍 Analyzing Python performance...")
        python_opportunities = self.analyze_python_performance()
        
        print("📊 Checking benchmark regressions...")
        benchmark_issues = self.check_benchmark_regressions()
        
        print("💾 Analyzing memory usage...")
        memory_opportunities = self.analyze_memory_usage()
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'repository': 'dp-flash-attention',
            'cuda_optimizations': {
                'total_files': len(cuda_opportunities),
                'opportunities': cuda_opportunities
            },
            'python_optimizations': {
                'total_files': len(python_opportunities),
                'opportunities': python_opportunities
            },
            'benchmark_analysis': {
                'total_issues': len(benchmark_issues),
                'regressions': benchmark_issues
            },
            'memory_analysis': {
                'total_files': len(memory_opportunities),
                'opportunities': memory_opportunities
            },
            'summary': {
                'total_optimization_opportunities': (
                    len(cuda_opportunities) + 
                    len(python_opportunities) + 
                    len(memory_opportunities)
                ),
                'high_priority_count': self._count_high_priority_issues([
                    cuda_opportunities, python_opportunities, memory_opportunities
                ]),
                'estimated_performance_gain': self._estimate_performance_gain([
                    cuda_opportunities, python_opportunities, memory_opportunities
                ])
            }
        }
        
        return report
    
    def _count_high_priority_issues(self, opportunity_lists: List[List[Dict]]) -> int:
        """Count high priority performance issues"""
        count = 0
        for opp_list in opportunity_lists:
            for item in opp_list:
                if item.get('priority_score', 0) > 6.0:
                    count += 1
        return count
    
    def _estimate_performance_gain(self, opportunity_lists: List[List[Dict]]) -> str:
        """Estimate potential performance improvement"""
        total_score = 0
        total_items = 0
        
        for opp_list in opportunity_lists:
            for item in opp_list:
                total_score += item.get('priority_score', 0)
                total_items += 1
        
        if total_items == 0:
            return "0-5%"
        
        avg_score = total_score / total_items
        
        if avg_score > 8:
            return "25-40%"
        elif avg_score > 6:
            return "15-25%"
        elif avg_score > 4:
            return "5-15%"
        else:
            return "0-5%"
    
    def save_report(self, report: Dict) -> None:
        """Save performance analysis report"""
        self.metrics_path.parent.mkdir(exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📈 Performance report saved to {self.metrics_path}")

def main():
    """Main execution function"""
    monitor = PerformanceMonitor()
    
    print("🚀 Starting advanced performance analysis...")
    report = monitor.generate_performance_report()
    monitor.save_report(report)
    
    # Print summary
    summary = report['summary']
    print(f"\n📊 Performance Analysis Summary:")
    print(f"   Total optimization opportunities: {summary['total_optimization_opportunities']}")
    print(f"   High priority issues: {summary['high_priority_count']}")
    print(f"   Estimated performance gain: {summary['estimated_performance_gain']}")
    
    if summary['total_optimization_opportunities'] > 0:
        print(f"\n🎯 Top recommendations:")
        
        # Show top CUDA opportunities
        cuda_opps = report['cuda_optimizations']['opportunities']
        if cuda_opps:
            top_cuda = max(cuda_opps, key=lambda x: x['priority_score'])
            print(f"   CUDA: {top_cuda['file']} - {len(top_cuda['issues'])} optimization opportunities")
        
        # Show top Python opportunities  
        python_opps = report['python_optimizations']['opportunities']
        if python_opps:
            top_python = max(python_opps, key=lambda x: x['priority_score'])
            print(f"   Python: {top_python['file']} - {len(top_python['issues'])} optimization opportunities")

if __name__ == "__main__":
    main()