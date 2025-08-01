#!/usr/bin/env python3
"""
Advanced Security Analysis and Compliance Monitoring
Identifies security vulnerabilities and compliance gaps
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SecurityAnalyzer:
    """Advanced security analysis for ML/CUDA repositories"""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.security_path = self.repo_path / ".terragon" / "security-analysis.json"
        
        # Security patterns to detect
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api[_-]?key\s*=\s*["\'][^"\']{16,}["\']',  
                r'secret\s*=\s*["\'][^"\']{16,}["\']',
                r'token\s*=\s*["\'][^"\']{20,}["\']',
                r'private[_-]?key\s*=\s*["\'][^"\']{32,}["\']'
            ],
            'insecure_network': [
                r'http://(?!localhost|127\.0\.0\.1)',
                r'ssl.*verify\s*=\s*False',
                r'check_hostname\s*=\s*False',
                r'urllib.*urlopen.*http://'
            ],
            'dangerous_functions': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'pickle\.loads?\s*\(',
                r'subprocess\.call.*shell\s*=\s*True',
                r'os\.system\s*\('
            ],
            'cuda_security': [
                r'cudaMalloc.*[^;]\s*$',  # Unchecked cudaMalloc
                r'__global__.*void.*\w+.*\([^)]*\*[^)]*\)',  # Kernel with raw pointers
                r'memcpy.*cudaMemcpy.*[^;]\s*$'  # Unchecked memory operations
            ],
            'privacy_violations': [
                r'print.*epsilon',  # Printing privacy parameters
                r'log.*delta',      # Logging privacy parameters  
                r'debug.*noise',    # Debug output of noise
                r'save.*privacy'    # Saving privacy-related data
            ]
        }
    
    def scan_for_secrets(self) -> List[Dict]:
        """Scan for hardcoded secrets and credentials"""
        vulnerabilities = []
        
        # Scan all text files
        file_patterns = ["*.py", "*.yaml", "*.yml", "*.json", "*.txt", "*.md", "*.sh"]
        
        for pattern in file_patterns:
            files = list(self.repo_path.glob(f"**/{pattern}"))
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Check against secret patterns
                    for secret_type, patterns in [('hardcoded_secrets', self.security_patterns['hardcoded_secrets'])]:
                        for regex_pattern in patterns:
                            matches = re.finditer(regex_pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                vulnerabilities.append({
                                    'type': 'secret_exposure',
                                    'severity': 'high',
                                    'file': str(file_path.relative_to(self.repo_path)),
                                    'line': line_num,
                                    'pattern': regex_pattern,
                                    'description': f'Potential hardcoded secret detected',
                                    'recommendation': 'Move to environment variables or secure vault'
                                })
                                
                except Exception as e:
                    continue
        
        return vulnerabilities
    
    def analyze_network_security(self) -> List[Dict]:
        """Analyze network security practices"""
        vulnerabilities = []
        
        python_files = list(self.repo_path.glob("**/*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                for pattern in self.security_patterns['insecure_network']:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'insecure_network',
                            'severity': 'medium',
                            'file': str(py_file.relative_to(self.repo_path)),
                            'line': line_num,
                            'pattern': pattern,
                            'description': 'Insecure network communication detected',
                            'recommendation': 'Use HTTPS and proper certificate validation'
                        })
                        
            except Exception:
                continue
        
        return vulnerabilities
    
    def analyze_dangerous_functions(self) -> List[Dict]:
        """Identify usage of dangerous functions"""
        vulnerabilities = []
        
        python_files = list(self.repo_path.glob("**/*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                for pattern in self.security_patterns['dangerous_functions']:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        severity = 'high'
                        if 'eval' in match.group() or 'exec' in match.group():
                            severity = 'critical'
                        
                        vulnerabilities.append({
                            'type': 'dangerous_function',
                            'severity': severity,
                            'file': str(py_file.relative_to(self.repo_path)),
                            'line': line_num,
                            'function': match.group(),
                            'description': 'Usage of potentially dangerous function',
                            'recommendation': 'Replace with safer alternatives or add input validation'
                        })
                        
            except Exception:
                continue
        
        return vulnerabilities
    
    def analyze_cuda_security(self) -> List[Dict]:
        """Analyze CUDA-specific security issues"""
        vulnerabilities = []
        
        cuda_files = list(self.repo_path.glob("**/*.cu")) + list(self.repo_path.glob("**/*.cuh"))
        
        for cuda_file in cuda_files:
            try:
                with open(cuda_file, 'r') as f:
                    content = f.read()
                
                # Check for unchecked CUDA operations
                for pattern in self.security_patterns['cuda_security']:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'cuda_security',
                            'severity': 'medium',
                            'file': str(cuda_file.relative_to(self.repo_path)),
                            'line': line_num,
                            'description': 'Potentially unsafe CUDA operation',
                            'recommendation': 'Add error checking for CUDA operations'
                        })
                
                # Check for buffer overflow potential
                if '__global__' in content:
                    # Look for array accesses without bounds checking
                    array_accesses = re.findall(r'(\w+)\[([^\]]*)\]', content)
                    for array_name, index in array_accesses:
                        if 'threadIdx' in index and 'blockDim' not in content:
                            vulnerabilities.append({
                                'type': 'buffer_overflow',
                                'severity': 'high',
                                'file': str(cuda_file.relative_to(self.repo_path)),
                                'line': 0,  # Would need more complex parsing for exact line
                                'description': f'Array access {array_name}[{index}] without bounds checking',
                                'recommendation': 'Add bounds checking for array accesses'
                            })
                            
            except Exception:
                continue
        
        return vulnerabilities
    
    def analyze_privacy_violations(self) -> List[Dict]:
        """Analyze differential privacy implementation for violations"""
        violations = []
        
        python_files = list(self.repo_path.glob("**/*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                for pattern in self.security_patterns['privacy_violations']:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        violations.append({
                            'type': 'privacy_violation',
                            'severity': 'high',
                            'file': str(py_file.relative_to(self.repo_path)),
                            'line': line_num,
                            'description': 'Potential privacy parameter leakage',
                            'recommendation': 'Avoid logging or printing privacy-sensitive information'
                        })
                
                # Check for proper noise addition
                if 'epsilon' in content and 'noise' not in content:
                    violations.append({
                        'type': 'missing_noise',
                        'severity': 'critical',
                        'file': str(py_file.relative_to(self.repo_path)),
                        'line': 0,
                        'description': 'Privacy budget (epsilon) used without noise addition',
                        'recommendation': 'Ensure proper noise is added when consuming privacy budget'
                    })
                    
            except Exception:
                continue
        
        return violations
    
    def check_dependency_vulnerabilities(self) -> List[Dict]:
        """Check for known vulnerabilities in dependencies"""
        vulnerabilities = []
        
        # Check requirements files
        req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        
        for req_file in req_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                # Simulate vulnerability check (would use safety/pip-audit in real scenario)
                vulnerabilities.append({
                    'type': 'dependency_audit',
                    'severity': 'medium',
                    'file': req_file,
                    'description': f'Dependencies in {req_file} need security audit',
                    'recommendation': 'Run pip-audit or safety check on dependencies',
                    'tools': ['pip-audit', 'safety', 'snyk']
                })
        
        return vulnerabilities
    
    def analyze_container_security(self) -> List[Dict]:
        """Analyze Docker container security"""
        vulnerabilities = []
        
        dockerfile_path = self.repo_path / "Dockerfile"
        if dockerfile_path.exists():
            try:
                with open(dockerfile_path, 'r') as f:
                    content = f.read()
                
                # Check for security best practices
                issues = []
                
                if re.search(r'FROM.*:latest', content):
                    issues.append('Using latest tag instead of specific version')
                
                if 'USER root' in content or not re.search(r'USER \w+', content):
                    issues.append('Running as root user')
                
                if re.search(r'RUN.*curl.*\|.*sh', content):
                    issues.append('Downloading and executing scripts directly')
                
                if not re.search(r'RUN.*apt-get update.*apt-get install.*--no-install-recommends', content):
                    if 'apt-get install' in content:
                        issues.append('Not using --no-install-recommends flag')
                
                for issue in issues:
                    vulnerabilities.append({
                        'type': 'container_security',
                        'severity': 'medium',
                        'file': 'Dockerfile',
                        'description': issue,
                        'recommendation': 'Follow Docker security best practices'
                    })
                    
            except Exception:
                pass
        
        return vulnerabilities
    
    def generate_security_report(self) -> Dict:
        """Generate comprehensive security analysis report"""
        print("🔒 Scanning for hardcoded secrets...")
        secrets = self.scan_for_secrets()
        
        print("🌐 Analyzing network security...")
        network_issues = self.analyze_network_security()
        
        print("⚠️  Checking dangerous functions...")
        dangerous_funcs = self.analyze_dangerous_functions()
        
        print("🔧 Analyzing CUDA security...")
        cuda_issues = self.analyze_cuda_security()
        
        print("🔐 Checking privacy violations...")
        privacy_violations = self.analyze_privacy_violations()
        
        print("📦 Checking dependency vulnerabilities...")
        dependency_vulns = self.check_dependency_vulnerabilities()
        
        print("🐳 Analyzing container security...")
        container_issues = self.analyze_container_security()
        
        all_vulnerabilities = (
            secrets + network_issues + dangerous_funcs + 
            cuda_issues + privacy_violations + dependency_vulns + container_issues
        )
        
        # Calculate severity distribution
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for vuln in all_vulnerabilities:
            severity = vuln.get('severity', 'medium')
            severity_counts[severity] += 1
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'repository': 'dp-flash-attention',
            'total_vulnerabilities': len(all_vulnerabilities),
            'severity_distribution': severity_counts,
            'vulnerability_categories': {
                'secrets': len(secrets),
                'network_security': len(network_issues),
                'dangerous_functions': len(dangerous_funcs),
                'cuda_security': len(cuda_issues),
                'privacy_violations': len(privacy_violations),
                'dependency_vulnerabilities': len(dependency_vulns),
                'container_security': len(container_issues)
            },
            'vulnerabilities': all_vulnerabilities,
            'risk_score': self._calculate_risk_score(all_vulnerabilities),
            'recommendations': self._generate_recommendations(all_vulnerabilities)
        }
        
        return report
    
    def _calculate_risk_score(self, vulnerabilities: List[Dict]) -> float:
        """Calculate overall security risk score (0-100)"""
        severity_weights = {'critical': 10, 'high': 7, 'medium': 4, 'low': 1}
        
        total_score = 0
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'medium')
            total_score += severity_weights.get(severity, 4)
        
        # Normalize to 0-100 scale
        max_possible = len(vulnerabilities) * 10
        if max_possible == 0:
            return 0
        
        return min((total_score / max_possible) * 100, 100)
    
    def _generate_recommendations(self, vulnerabilities: List[Dict]) -> List[str]:
        """Generate prioritized security recommendations"""
        recommendations = []
        
        # Group by type and severity
        critical_issues = [v for v in vulnerabilities if v.get('severity') == 'critical']
        high_issues = [v for v in vulnerabilities if v.get('severity') == 'high']
        
        if critical_issues:
            recommendations.append("CRITICAL: Address code execution vulnerabilities immediately")
        
        if high_issues:
            recommendations.append("HIGH: Fix secret exposure and privacy violations")
        
        if any(v.get('type') == 'dependency_audit' for v in vulnerabilities):
            recommendations.append("Run automated dependency security scanning (pip-audit, safety)")
        
        if any(v.get('type') == 'container_security' for v in vulnerabilities):
            recommendations.append("Implement Docker security best practices")
        
        if any(v.get('type') == 'cuda_security' for v in vulnerabilities):
            recommendations.append("Add error checking and bounds validation to CUDA code")
        
        return recommendations
    
    def save_report(self, report: Dict) -> None:
        """Save security analysis report"""
        self.security_path.parent.mkdir(exist_ok=True)
        with open(self.security_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"🛡️  Security report saved to {self.security_path}")

def main():
    """Main execution function"""
    analyzer = SecurityAnalyzer()
    
    print("🛡️  Starting comprehensive security analysis...")
    report = analyzer.generate_security_report()
    analyzer.save_report(report)
    
    # Print summary
    print(f"\n📊 Security Analysis Summary:")
    print(f"   Total vulnerabilities found: {report['total_vulnerabilities']}")
    print(f"   Risk score: {report['risk_score']:.1f}/100")
    
    severity_dist = report['severity_distribution']
    print(f"   Critical: {severity_dist['critical']}, High: {severity_dist['high']}, " +
          f"Medium: {severity_dist['medium']}, Low: {severity_dist['low']}")
    
    if report['recommendations']:
        print(f"\n🎯 Top recommendations:")
        for rec in report['recommendations'][:3]:
            print(f"   • {rec}")

if __name__ == "__main__":
    main()