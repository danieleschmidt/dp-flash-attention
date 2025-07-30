#!/usr/bin/env python3
"""
Privacy Parameter Validation Script

Validates differential privacy parameters in code for security compliance.
Ensures epsilon, delta values meet security requirements.
"""

import sys
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class PrivacyParameterChecker:
    """Checker for privacy parameter validation."""
    
    def __init__(self):
        # Privacy parameter constraints
        self.min_epsilon = 0.1  # Minimum epsilon for meaningful privacy
        self.max_epsilon = 10.0  # Maximum epsilon for reasonable privacy
        self.max_delta = 1e-3   # Maximum delta for strong privacy
        self.min_clip_norm = 0.1  # Minimum gradient clipping norm
        
        self.issues = []
    
    def check_file(self, filepath: Path) -> List[Dict]:
        """Check privacy parameters in a Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return [{'type': 'error', 'message': f"Failed to read {filepath}: {e}"}]
        
        issues = []
        
        # Check for hardcoded privacy parameters
        issues.extend(self._check_hardcoded_params(content, filepath))
        
        # Check for parameter validation
        issues.extend(self._check_parameter_validation(content, filepath))
        
        # Check for secure parameter usage
        issues.extend(self._check_secure_usage(content, filepath))
        
        # Check AST for more complex patterns
        try:
            tree = ast.parse(content)
            issues.extend(self._check_ast_patterns(tree, filepath))
        except SyntaxError:
            pass  # Skip files with syntax errors
        
        return issues
    
    def _check_hardcoded_params(self, content: str, filepath: Path) -> List[Dict]:
        """Check for hardcoded privacy parameters."""
        issues = []
        
        # Patterns for privacy parameters
        patterns = {
            'epsilon': r'epsilon\s*=\s*([0-9]*\.?[0-9]+)',
            'delta': r'delta\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            'max_grad_norm': r'max_grad_norm\s*=\s*([0-9]*\.?[0-9]+)',
            'clip_norm': r'clip_norm\s*=\s*([0-9]*\.?[0-9]+)',
        }
        
        for param_name, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                value = float(match.group(1))
                
                # Validate parameter ranges
                issue = self._validate_parameter(param_name, value, line_num, filepath)
                if issue:
                    issues.append(issue)
        
        return issues
    
    def _validate_parameter(self, param_name: str, value: float, line_num: int, filepath: Path) -> Optional[Dict]:
        """Validate a privacy parameter value."""
        if param_name == 'epsilon':
            if value <= 0:
                return {
                    'type': 'vulnerability',
                    'message': f"{filepath}:{line_num}: epsilon must be positive, got {value}"
                }
            elif value > self.max_epsilon:
                return {
                    'type': 'warning',
                    'message': f"{filepath}:{line_num}: epsilon {value} may provide insufficient privacy (> {self.max_epsilon})"
                }
            elif value < self.min_epsilon:
                return {
                    'type': 'info',
                    'message': f"{filepath}:{line_num}: epsilon {value} provides very strong privacy"
                }
        
        elif param_name == 'delta':
            if value <= 0:
                return {
                    'type': 'vulnerability',
                    'message': f"{filepath}:{line_num}: delta must be positive, got {value}"
                }
            elif value > self.max_delta:
                return {
                    'type': 'warning',
                    'message': f"{filepath}:{line_num}: delta {value} may be too large for strong privacy (> {self.max_delta})"
                }
        
        elif param_name in ['max_grad_norm', 'clip_norm']:
            if value <= 0:
                return {
                    'type': 'vulnerability',
                    'message': f"{filepath}:{line_num}: {param_name} must be positive, got {value}"
                }
            elif value < self.min_clip_norm:
                return {
                    'type': 'warning',
                    'message': f"{filepath}:{line_num}: {param_name} {value} may be too small for effective training"
                }
        
        return None
    
    def _check_parameter_validation(self, content: str, filepath: Path) -> List[Dict]:
        """Check for proper parameter validation."""
        issues = []
        
        # Look for validation patterns
        validation_patterns = [
            r'if\s+epsilon\s*<=\s*0',
            r'if\s+delta\s*<=\s*0',
            r'assert\s+epsilon\s*>\s*0',
            r'assert\s+delta\s*>\s*0',
            r'ValueError.*epsilon',
            r'ValueError.*delta',
        ]
        
        has_validation = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in validation_patterns
        )
        
        # Check if file defines privacy parameters but lacks validation
        defines_params = any(
            re.search(f'{param}\\s*=', content, re.IGNORECASE)
            for param in ['epsilon', 'delta', 'max_grad_norm']
        )
        
        if defines_params and not has_validation:
            issues.append({
                'type': 'warning',
                'message': f"{filepath}: Defines privacy parameters but lacks validation"
            })
        
        return issues
    
    def _check_secure_usage(self, content: str, filepath: Path) -> List[Dict]:
        """Check for secure usage patterns."""
        issues = []
        
        # Check for potential privacy leaks
        leak_patterns = [
            r'print.*epsilon',
            r'print.*delta',
            r'log.*epsilon',
            r'log.*delta',
            r'debug.*epsilon',
            r'debug.*delta',
        ]
        
        for pattern in leak_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append({
                    'type': 'warning',
                    'message': f"{filepath}:{line_num}: Potential privacy parameter leak in output"
                })
        
        # Check for insecure random number generation
        insecure_random = [
            r'random\.random\(\)',
            r'numpy\.random\.rand',
            r'torch\.rand\(',
        ]
        
        for pattern in insecure_random:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'type': 'info',
                    'message': f"{filepath}: Consider using cryptographically secure random for DP noise"
                })
        
        return issues
    
    def _check_ast_patterns(self, tree: ast.AST, filepath: Path) -> List[Dict]:
        """Check AST for complex patterns."""
        issues = []
        
        class PrivacyVisitor(ast.NodeVisitor):
            def __init__(self, checker):
                self.checker = checker
                self.issues = []
            
            def visit_FunctionDef(self, node):
                # Check function signatures for privacy parameters
                for arg in node.args.args:
                    if arg.arg in ['epsilon', 'delta']:
                        # Check if default value is provided
                        if not node.args.defaults:
                            self.issues.append({
                                'type': 'info',
                                'message': f"{filepath}:{node.lineno}: Function {node.name} has privacy parameter {arg.arg} without default"
                            })
                
                self.generic_visit(node)
            
            def visit_Compare(self, node):
                # Check for proper comparison operators with privacy params
                if (isinstance(node.left, ast.Name) and 
                    node.left.id in ['epsilon', 'delta']):
                    
                    for op, comparator in zip(node.ops, node.comparators):
                        if isinstance(op, ast.LtE) and isinstance(comparator, ast.Constant):
                            if comparator.value == 0:
                                self.issues.append({
                                    'type': 'vulnerability',
                                    'message': f"{filepath}:{node.lineno}: Privacy parameter should be > 0, not <= 0"
                                })
                
                self.generic_visit(node)
        
        visitor = PrivacyVisitor(self)
        visitor.visit(tree)
        issues.extend(visitor.issues)
        
        return issues
    
    def scan_directory(self, directory: Path) -> Dict[str, List[Dict]]:
        """Scan all Python files in directory."""
        results = {}
        
        for filepath in directory.rglob('*.py'):
            if filepath.is_file():
                issues = self.check_file(filepath)
                if issues:
                    results[str(filepath)] = issues
        
        return results
    
    def generate_report(self, results: Dict[str, List[Dict]]) -> str:
        """Generate privacy parameter report."""
        report = []
        report.append("Privacy Parameter Security Report")
        report.append("=" * 40)
        report.append()
        
        total_vulns = 0
        total_warnings = 0
        total_info = 0
        
        for filepath, issues in results.items():
            if issues:
                report.append(f"File: {filepath}")
                report.append("-" * 20)
                
                for issue in issues:
                    level = issue['type'].upper()
                    message = issue['message']
                    report.append(f"  {level}: {message}")
                    
                    if issue['type'] == 'vulnerability':
                        total_vulns += 1
                    elif issue['type'] == 'warning':
                        total_warnings += 1
                    else:
                        total_info += 1
                
                report.append()
        
        report.append("Summary:")
        report.append(f"  Vulnerabilities: {total_vulns}")
        report.append(f"  Warnings: {total_warnings}")
        report.append(f"  Info: {total_info}")
        
        if total_vulns > 0:
            report.append()
            report.append("CRITICAL: Fix vulnerabilities before production deployment.")
        
        return "\n".join(report)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python check_privacy_params.py <file1> [file2] ...")
        sys.exit(1)
    
    checker = PrivacyParameterChecker()
    all_results = {}
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if filepath.is_file() and filepath.suffix == '.py':
            issues = checker.check_file(filepath)
            if issues:
                all_results[str(filepath)] = issues
        elif filepath.is_dir():
            dir_results = checker.scan_directory(filepath)
            all_results.update(dir_results)
    
    if all_results:
        report = checker.generate_report(all_results)
        print(report)
        
        # Exit with error if vulnerabilities found
        total_vulns = sum(
            len([i for i in issues if i['type'] == 'vulnerability'])
            for issues in all_results.values()
        )
        sys.exit(1 if total_vulns > 0 else 0)
    else:
        print("No privacy parameter issues found.")
        sys.exit(0)


if __name__ == '__main__':
    main()