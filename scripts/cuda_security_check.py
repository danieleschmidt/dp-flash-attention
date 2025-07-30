#!/usr/bin/env python3
"""
CUDA Kernel Security Analysis Script

Performs security analysis on CUDA kernels for potential vulnerabilities.
Focus on differential privacy implementations and memory safety.
"""

import sys
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple


class CUDASecurityChecker:
    """Security checker for CUDA kernels."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.warnings = []
        
        # Security patterns to check
        self.security_patterns = {
            'buffer_overflow': [
                r'memcpy\s*\(',
                r'strcpy\s*\(',
                r'sprintf\s*\(',
                r'gets\s*\(',
            ],
            'uninitialized_memory': [
                r'malloc\s*\(',
                r'cudaMalloc\s*\(',
                r'__shared__\s+\w+\s+\w+\s*;',
            ],
            'race_conditions': [
                r'__syncthreads\s*\(\s*\)',
                r'atomicAdd\s*\(',
                r'atomicCAS\s*\(',
            ],
            'privacy_leaks': [
                r'printf\s*\(',
                r'cout\s*<<',
                r'stderr',
                r'stdout',
            ],
            'insecure_random': [
                r'rand\s*\(\s*\)',
                r'srand\s*\(',
                r'random\s*\(\s*\)',
            ]
        }
        
        # Privacy-specific patterns for DP libraries
        self.privacy_patterns = {
            'epsilon_validation': r'epsilon\s*[<>=]\s*0',
            'delta_validation': r'delta\s*[<>=]\s*0',
            'noise_generation': r'gaussian|laplace|exponential.*noise',
            'gradient_clipping': r'clip.*gradient|gradient.*clip',
        }
    
    def check_file(self, filepath: Path) -> Dict[str, List[str]]:
        """Check a single CUDA file for security issues."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {'errors': [f"Failed to read {filepath}: {e}"]}
        
        issues = {
            'vulnerabilities': [],
            'warnings': [],
            'privacy_issues': []
        }
        
        # Check security patterns
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues['vulnerabilities'].append(
                        f"{filepath}:{line_num}: {category} - {match.group()}"
                    )
        
        # Check privacy-specific patterns
        for category, pattern in self.privacy_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues['privacy_issues'].append(
                    f"{filepath}:{line_num}: {category} - {match.group()}"
                )
        
        # Check for memory safety
        if self._check_memory_safety(content):
            issues['warnings'].append(f"{filepath}: Potential memory safety issues")
        
        # Check for proper error handling
        if not self._check_error_handling(content):
            issues['warnings'].append(f"{filepath}: Missing CUDA error handling")
        
        return issues
    
    def _check_memory_safety(self, content: str) -> bool:
        """Check for memory safety issues."""
        # Look for malloc without corresponding free
        malloc_count = len(re.findall(r'cudaMalloc|malloc', content, re.IGNORECASE))
        free_count = len(re.findall(r'cudaFree|free', content, re.IGNORECASE))
        
        return malloc_count > free_count
    
    def _check_error_handling(self, content: str) -> bool:
        """Check for proper CUDA error handling."""
        cuda_calls = re.findall(r'cuda\w+\s*\(', content, re.IGNORECASE)
        error_checks = re.findall(r'cudaGetLastError|cudaSuccess', content, re.IGNORECASE)
        
        # Should have some error checking if there are CUDA calls
        return len(error_checks) > 0 if len(cuda_calls) > 0 else True
    
    def scan_directory(self, directory: Path) -> Dict[str, Dict]:
        """Scan all CUDA files in directory."""
        results = {}
        
        cuda_extensions = ['.cu', '.cuh', '.h', '.hpp']
        
        for ext in cuda_extensions:
            for filepath in directory.rglob(f'*{ext}'):
                if filepath.is_file():
                    results[str(filepath)] = self.check_file(filepath)
        
        return results
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate security report."""
        report = []
        report.append("CUDA Security Analysis Report")
        report.append("=" * 40)
        report.append()
        
        total_vulns = 0
        total_warnings = 0
        total_privacy = 0
        
        for filepath, issues in results.items():
            if any(issues.values()):
                report.append(f"File: {filepath}")
                report.append("-" * 20)
                
                for vuln in issues.get('vulnerabilities', []):
                    report.append(f"  VULNERABILITY: {vuln}")
                    total_vulns += 1
                
                for warning in issues.get('warnings', []):
                    report.append(f"  WARNING: {warning}")
                    total_warnings += 1
                
                for privacy in issues.get('privacy_issues', []):
                    report.append(f"  PRIVACY: {privacy}")
                    total_privacy += 1
                
                report.append()
        
        report.append("Summary:")
        report.append(f"  Total vulnerabilities: {total_vulns}")
        report.append(f"  Total warnings: {total_warnings}")
        report.append(f"  Total privacy issues: {total_privacy}")
        
        if total_vulns > 0:
            report.append()
            report.append("RECOMMENDATION: Address vulnerabilities before production use.")
        
        return "\n".join(report)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python cuda_security_check.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        sys.exit(1)
    
    checker = CUDASecurityChecker()
    results = checker.scan_directory(directory)
    report = checker.generate_report(results)
    
    print(report)
    
    # Exit with error code if vulnerabilities found
    total_vulns = sum(len(issues.get('vulnerabilities', [])) for issues in results.values())
    sys.exit(1 if total_vulns > 0 else 0)


if __name__ == '__main__':
    main()