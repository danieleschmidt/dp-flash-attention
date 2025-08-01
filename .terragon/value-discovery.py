#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers, scores, and prioritizes improvement opportunities
"""

import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import logging

@dataclass
class WorkItem:
    """Represents a discovered work item with comprehensive scoring"""
    id: str
    title: str
    description: str
    category: str
    source: str
    estimated_effort: float  # in hours
    
    # Scoring components
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    
    # Impact metrics
    business_impact: int  # 1-10
    user_impact: int      # 1-10
    time_criticality: int # 1-10
    risk_reduction: int   # 1-10
    confidence: int       # 1-10
    ease: int            # 1-10
    
    # Context
    files_affected: List[str]
    related_issues: List[str] 
    created_at: str
    priority: str
    
class ValueDiscoveryEngine:
    """Main engine for autonomous value discovery"""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration"""
        # Return default config since yaml not available
        return {
            'scoring': {
                'weights': {
                    'advanced': {
                        'wsjf': 0.5,
                        'ice': 0.1,
                        'technicalDebt': 0.3,
                        'security': 0.1
                    }
                }
            },
            'repository': {
                'name': 'dp-flash-attention',
                'maturity_level': 'advanced'
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for value discovery"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.repo_path / ".terragon" / "discovery.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("terragon.discovery")
    
    def discover_work_items(self) -> List[WorkItem]:
        """Main discovery method - harvests signals from multiple sources"""
        items = []
        
        # Source 1: Git history analysis
        items.extend(self._discover_from_git_history())
        
        # Source 2: Static analysis (if tools available)
        items.extend(self._discover_from_static_analysis())
        
        # Source 3: Security vulnerabilities
        items.extend(self._discover_security_issues())
        
        # Source 4: Performance opportunities
        items.extend(self._discover_performance_issues())
        
        # Source 5: Dependency updates
        items.extend(self._discover_dependency_updates())
        
        # Source 6: Documentation gaps
        items.extend(self._discover_documentation_gaps())
        
        return items
    
    def _discover_from_git_history(self) -> List[WorkItem]:
        """Analyze git history for technical debt signals"""
        items = []
        
        try:
            # Search for TODO/FIXME in code
            result = subprocess.run([
                "git", "grep", "-n", "-i", "-E", "TODO|FIXME|HACK|XXX",
                "--", "*.py", "*.cu", "*.cpp", "*.h"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':' in line and line.strip():
                        file_path, line_num, content = line.split(':', 2)
                        item = WorkItem(
                            id=f"td-{hash(line) % 10000}",
                            title=f"Technical debt in {file_path}:{line_num}",
                            description=content.strip(),
                            category="technical-debt",
                            source="git-grep",
                            estimated_effort=0.5,
                            wsjf_score=0, ice_score=0, technical_debt_score=25,
                            composite_score=0,
                            business_impact=3, user_impact=2, time_criticality=2,
                            risk_reduction=4, confidence=8, ease=7,
                            files_affected=[file_path],
                            related_issues=[],
                            created_at=datetime.now().isoformat(),
                            priority="low"
                        )
                        items.append(item)
        except Exception as e:
            self.logger.warning(f"Git grep failed: {e}")
        
        # Analyze commit messages for quick fixes
        try:
            result = subprocess.run([
                "git", "log", "--oneline", "-50", "--grep=fix", "--grep=hack", 
                "--grep=quick", "--grep=temp", "-i"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    commit_hash, message = line.split(' ', 1)
                    item = WorkItem(
                        id=f"commit-{commit_hash}",
                        title=f"Investigate quick fix: {message[:50]}",
                        description=f"Commit {commit_hash} suggests temporary solution",
                        category="technical-debt",
                        source="git-log",
                        estimated_effort=1.0,
                        wsjf_score=0, ice_score=0, technical_debt_score=15,
                        composite_score=0,
                        business_impact=4, user_impact=3, time_criticality=3,
                        risk_reduction=5, confidence=6, ease=5,
                        files_affected=[],
                        related_issues=[],
                        created_at=datetime.now().isoformat(),
                        priority="medium"
                    )
                    items.append(item)
        except Exception as e:
            self.logger.warning(f"Git log analysis failed: {e}")
            
        return items
    
    def _discover_from_static_analysis(self) -> List[WorkItem]:
        """Run static analysis tools if available"""
        items = []
        
        # Check for Python files with high complexity
        python_files = list(self.repo_path.glob("**/*.py"))
        if python_files:
            # Simulate complexity analysis
            for py_file in python_files[:5]:  # Limit to avoid overwhelming
                relative_path = py_file.relative_to(self.repo_path)
                try:
                    with open(py_file, 'r') as f:
                        lines = f.readlines()
                        # Simple heuristic: files with many nested statements
                        complexity = sum(1 for line in lines if 
                                       re.search(r'^\\s{8,}(if|for|while|try|with)', line))
                        
                        if complexity > 5:
                            item = WorkItem(
                                id=f"complexity-{hash(str(relative_path)) % 10000}",
                                title=f"High complexity in {relative_path}",
                                description=f"File has {complexity} deeply nested statements",
                                category="technical-debt",
                                source="complexity-analysis",
                                estimated_effort=2.0,
                                wsjf_score=0, ice_score=0, technical_debt_score=complexity * 3,
                                composite_score=0,
                                business_impact=5, user_impact=4, time_criticality=2,
                                risk_reduction=6, confidence=7, ease=4,
                                files_affected=[str(relative_path)],
                                related_issues=[],
                                created_at=datetime.now().isoformat(),
                                priority="medium"
                            )
                            items.append(item)
                except Exception as e:
                    self.logger.debug(f"Could not analyze {py_file}: {e}")
        
        return items
    
    def _discover_security_issues(self) -> List[WorkItem]:
        """Identify potential security issues"""
        items = []
        
        # Check for hardcoded secrets patterns
        secret_patterns = [
            r'password\\s*=\\s*["\'][^"\']+["\']',
            r'api[_-]?key\\s*=\\s*["\'][^"\']+["\']',
            r'secret\\s*=\\s*["\'][^"\']+["\']',
            r'token\\s*=\\s*["\'][^"\']+["\']'
        ]
        
        for pattern in secret_patterns:
            try:
                result = subprocess.run([
                    "git", "grep", "-n", "-i", "-E", pattern,
                    "--", "*.py", "*.yaml", "*.yml", "*.json"
                ], cwd=self.repo_path, capture_output=True, text=True)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if ':' in line and line.strip():
                            file_path, line_num, content = line.split(':', 2)
                            item = WorkItem(
                                id=f"sec-{hash(line) % 10000}",
                                title=f"Potential secret in {file_path}:{line_num}",
                                description="Hardcoded secret detected",
                                category="security-critical",
                                source="secret-detection",
                                estimated_effort=0.5,
                                wsjf_score=0, ice_score=0, technical_debt_score=50,
                                composite_score=0,
                                business_impact=9, user_impact=8, time_criticality=8,
                                risk_reduction=9, confidence=9, ease=8,
                                files_affected=[file_path],
                                related_issues=[],
                                created_at=datetime.now().isoformat(),
                                priority="high"
                            )
                            items.append(item)
            except Exception as e:
                self.logger.debug(f"Security pattern search failed: {e}")
        
        return items
    
    def _discover_performance_issues(self) -> List[WorkItem]:
        """Identify performance optimization opportunities"""
        items = []
        
        # Look for performance-related comments
        try:
            result = subprocess.run([
                "git", "grep", "-n", "-i", "-E", 
                "slow|performance|bottleneck|optimize|inefficient",
                "--", "*.py", "*.cu"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n')[:10]:  # Limit results
                    if ':' in line and line.strip():
                        file_path, line_num, content = line.split(':', 2)
                        item = WorkItem(
                            id=f"perf-{hash(line) % 10000}",
                            title=f"Performance opportunity in {file_path}",
                            description=content.strip(),
                            category="performance-regression",
                            source="performance-grep",
                            estimated_effort=3.0,
                            wsjf_score=0, ice_score=0, technical_debt_score=20,
                            composite_score=0,
                            business_impact=7, user_impact=8, time_criticality=5,
                            risk_reduction=4, confidence=6, ease=3,
                            files_affected=[file_path],
                            related_issues=[],
                            created_at=datetime.now().isoformat(),
                            priority="medium"
                        )
                        items.append(item)
        except Exception as e:
            self.logger.debug(f"Performance grep failed: {e}")
        
        return items
    
    def _discover_dependency_updates(self) -> List[WorkItem]:
        """Find dependency update opportunities"""
        items = []
        
        # Check requirements files for outdated packages
        req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        
        for req_file in req_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                item = WorkItem(
                    id=f"dep-{req_file}",
                    title=f"Review dependencies in {req_file}",
                    description="Check for security updates and version upgrades",
                    category="dependency-update",
                    source="dependency-analysis",
                    estimated_effort=1.0,
                    wsjf_score=0, ice_score=0, technical_debt_score=10,
                    composite_score=0,
                    business_impact=4, user_impact=3, time_criticality=6,
                    risk_reduction=7, confidence=8, ease=6,
                    files_affected=[req_file],
                    related_issues=[],
                    created_at=datetime.now().isoformat(),
                    priority="medium"
                )
                items.append(item)
        
        return items
    
    def _discover_documentation_gaps(self) -> List[WorkItem]:
        """Identify documentation improvement opportunities"""
        items = []
        
        # Check for undocumented Python modules
        python_files = list(self.repo_path.glob("src/**/*.py"))
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    # Simple check for docstrings
                    if 'def ' in content and '"""' not in content:
                        relative_path = py_file.relative_to(self.repo_path)
                        item = WorkItem(
                            id=f"doc-{hash(str(relative_path)) % 10000}",
                            title=f"Add docstrings to {relative_path}",
                            description="Module contains functions without documentation",
                            category="documentation",
                            source="documentation-analysis",
                            estimated_effort=1.5,
                            wsjf_score=0, ice_score=0, technical_debt_score=5,
                            composite_score=0,
                            business_impact=3, user_impact=5, time_criticality=1,
                            risk_reduction=2, confidence=9, ease=8,
                            files_affected=[str(relative_path)],
                            related_issues=[],
                            created_at=datetime.now().isoformat(),
                            priority="low"
                        )
                        items.append(item)
            except Exception as e:
                self.logger.debug(f"Could not analyze {py_file}: {e}")
        
        return items
    
    def calculate_composite_scores(self, items: List[WorkItem]) -> List[WorkItem]:
        """Calculate composite scores using WSJF + ICE + Technical Debt"""
        weights = self.config.get('scoring', {}).get('weights', {}).get('advanced', {})
        
        for item in items:
            # WSJF Score
            cost_of_delay = (
                item.business_impact + 
                item.time_criticality + 
                item.risk_reduction
            )
            item.wsjf_score = cost_of_delay / max(item.estimated_effort, 0.1)
            
            # ICE Score  
            item.ice_score = item.business_impact * item.confidence * item.ease
            
            # Composite Score
            wsjf_weight = weights.get('wsjf', 0.5)
            ice_weight = weights.get('ice', 0.1) 
            debt_weight = weights.get('technicalDebt', 0.3)
            security_weight = weights.get('security', 0.1)
            
            # Normalize scores
            normalized_wsjf = min(item.wsjf_score / 30.0, 1.0)  # Max expected WSJF
            normalized_ice = min(item.ice_score / 1000.0, 1.0)  # Max expected ICE
            normalized_debt = min(item.technical_debt_score / 100.0, 1.0)
            
            item.composite_score = (
                wsjf_weight * normalized_wsjf * 100 +
                ice_weight * normalized_ice * 100 +
                debt_weight * normalized_debt * 100
            )
            
            # Apply category boosts
            if item.category == "security-critical":
                item.composite_score *= 2.0
            elif item.category == "privacy-violation":
                item.composite_score *= 1.8
        
        return sorted(items, key=lambda x: x.composite_score, reverse=True)
    
    def save_metrics(self, items: List[WorkItem]) -> None:
        """Save discovery metrics and backlog"""
        metrics = {
            "discovery_timestamp": datetime.now().isoformat(),
            "total_items_discovered": len(items),
            "categories": {},
            "top_items": [],
            "backlog_summary": {
                "high_priority": len([i for i in items if i.priority == "high"]),
                "medium_priority": len([i for i in items if i.priority == "medium"]),
                "low_priority": len([i for i in items if i.priority == "low"])
            }
        }
        
        # Category breakdown
        for item in items:
            if item.category not in metrics["categories"]:
                metrics["categories"][item.category] = 0
            metrics["categories"][item.category] += 1
        
        # Top 10 items
        metrics["top_items"] = [asdict(item) for item in items[:10]]
        
        # Save to file
        self.metrics_path.parent.mkdir(exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Saved {len(items)} discovered items to {self.metrics_path}")
    
    def generate_backlog(self, items: List[WorkItem]) -> str:
        """Generate formatted backlog markdown"""
        backlog = f"""# 📊 Autonomous Value Backlog

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Repository: {self.config.get('repository', {}).get('name', 'Unknown')}
Maturity Level: {self.config.get('repository', {}).get('maturity_level', 'Unknown').upper()}

## 🎯 Next Best Value Item
"""
        
        if items:
            top_item = items[0]
            backlog += f"""**[{top_item.id.upper()}] {top_item.title}**
- **Composite Score**: {top_item.composite_score:.1f}
- **WSJF**: {top_item.wsjf_score:.1f} | **ICE**: {top_item.ice_score} | **Tech Debt**: {top_item.technical_debt_score}
- **Estimated Effort**: {top_item.estimated_effort} hours
- **Category**: {top_item.category}
- **Priority**: {top_item.priority.upper()}

## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
            
            for i, item in enumerate(items[:10], 1):
                backlog += f"| {i} | {item.id.upper()} | {item.title[:50]}{'...' if len(item.title) > 50 else ''} | {item.composite_score:.1f} | {item.category} | {item.estimated_effort} |\n"
        
        backlog += f"""

## 📈 Discovery Metrics
- **Total Items Discovered**: {len(items)}
- **High Priority**: {len([i for i in items if i.priority == 'high'])}
- **Medium Priority**: {len([i for i in items if i.priority == 'medium'])}
- **Low Priority**: {len([i for i in items if i.priority == 'low'])}

### Category Breakdown
"""
        
        categories = {}
        for item in items:
            categories[item.category] = categories.get(item.category, 0) + 1
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            backlog += f"- **{category.replace('-', ' ').title()}**: {count} items\n"
        
        backlog += """

## 🔄 Continuous Discovery
This backlog is automatically updated through:
- Git history analysis for technical debt signals
- Static analysis for code quality issues  
- Security vulnerability scanning
- Performance monitoring integration
- Dependency audit and updates

## 🚀 Execution Notes
- Items are scored using WSJF + ICE + Technical Debt metrics
- Security and privacy issues receive automatic priority boost
- All changes require validation before merge
- Rollback procedures available for failed executions

---
*Generated by Terragon Autonomous SDLC Engine*
"""
        
        return backlog

def main():
    """Main execution function"""
    engine = ValueDiscoveryEngine()
    
    print("🔍 Starting autonomous value discovery...")
    
    # Discover work items
    items = engine.discover_work_items()
    print(f"📊 Discovered {len(items)} potential work items")
    
    # Calculate scores
    scored_items = engine.calculate_composite_scores(items)
    print(f"⚡ Calculated composite scores for all items")
    
    # Save metrics
    engine.save_metrics(scored_items)
    
    # Generate backlog
    backlog_content = engine.generate_backlog(scored_items)
    backlog_path = Path("/root/repo/BACKLOG.md")
    with open(backlog_path, 'w') as f:
        f.write(backlog_content)
    
    print(f"📝 Generated backlog: {backlog_path}")
    
    if scored_items:
        print(f"\n🎯 Next best value item:")
        top_item = scored_items[0]
        print(f"   [{top_item.id.upper()}] {top_item.title}")
        print(f"   Score: {top_item.composite_score:.1f} | Effort: {top_item.estimated_effort}h")
        print(f"   Category: {top_item.category} | Priority: {top_item.priority.upper()}")

if __name__ == "__main__":
    main()