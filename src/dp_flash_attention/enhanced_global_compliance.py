"""
Enhanced Global Compliance Extensions for DP-Flash-Attention.

Extends global compliance coverage to additional regulatory frameworks
and provides enhanced multi-region support for emerging privacy laws.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from .globalization import ComplianceFramework, ComplianceRequirement, Region

logger = logging.getLogger(__name__)


class ExtendedComplianceFramework(Enum):
    """Extended compliance frameworks for global coverage."""
    # Existing frameworks from base module
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore/Thailand)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    
    # Additional frameworks for extended coverage
    POPI = "popi"  # Protection of Personal Information Act (South Africa)
    DPA_UAE = "dpa_uae"  # UAE Data Protection Law
    PIPL = "pipl"  # Personal Information Protection Law (China)
    APPI = "appi"  # Act on Protection of Personal Information (Japan)
    KVKK = "kvkk"  # Kişisel Verilerin Korunması Kanunu (Turkey)
    FADP = "fadp"  # Federal Act on Data Protection (Switzerland)
    LPDP = "lpdp"  # Ley de Protección de Datos Personales (Argentina)
    CDPA = "cdpa"  # Consumer Data Protection Act (Virginia, US)
    CPA = "cpa"  # Colorado Privacy Act (Colorado, US)
    CTDPA = "ctdpa"  # Connecticut Data Privacy Act (Connecticut, US)


class ExtendedRegion(Enum):
    """Extended regional coverage for global deployment."""
    # Core regions
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"
    
    # Extended regions
    ME_SOUTH_1 = "me-south-1"  # UAE/Middle East
    AF_SOUTH_1 = "af-south-1"  # South Africa
    AP_EAST_1 = "ap-east-1"    # Hong Kong
    AP_SOUTH_1 = "ap-south-1"  # India
    EU_NORTH_1 = "eu-north-1"  # Sweden
    EU_SOUTH_1 = "eu-south-1"  # Italy
    SA_EAST_1 = "sa-east-1"    # Brazil
    CA_CENTRAL_1 = "ca-central-1"  # Canada


class EnhancedComplianceManager:
    """Enhanced compliance manager with extended global coverage."""
    
    def __init__(self):
        self.extended_frameworks = self._initialize_extended_frameworks()
        self.cross_border_rules = self._initialize_cross_border_rules()
        self.audit_logs = []
        
    def _initialize_extended_frameworks(self) -> Dict[ExtendedComplianceFramework, ComplianceRequirement]:
        """Initialize extended compliance requirements."""
        return {
            # South Africa POPI Act
            ExtendedComplianceFramework.POPI: ComplianceRequirement(
                framework=ComplianceFramework.GDPR,  # Use base enum for compatibility
                min_epsilon=0.3,
                max_delta=1e-5,
                data_residency_required=True,
                audit_log_retention_days=2555,  # 7 years
                encryption_at_rest_required=True,
                encryption_in_transit_required=True,
                consent_management_required=True,
                data_minimization_required=True,
                right_to_erasure=True,
                breach_notification_hours=72
            ),
            
            # UAE Data Protection Law
            ExtendedComplianceFramework.DPA_UAE: ComplianceRequirement(
                framework=ComplianceFramework.GDPR,  # Use base enum for compatibility
                min_epsilon=0.5,
                max_delta=1e-5,
                data_residency_required=True,
                audit_log_retention_days=1825,  # 5 years
                encryption_at_rest_required=True,
                encryption_in_transit_required=True,
                consent_management_required=True,
                data_minimization_required=True,
                right_to_erasure=False,  # Limited right
                breach_notification_hours=72
            ),
            
            # China PIPL
            ExtendedComplianceFramework.PIPL: ComplianceRequirement(
                framework=ComplianceFramework.GDPR,  # Use base enum for compatibility
                min_epsilon=0.1,  # Very strict privacy requirements
                max_delta=1e-6,
                data_residency_required=True,  # Strict localization
                audit_log_retention_days=3650,  # 10 years
                encryption_at_rest_required=True,
                encryption_in_transit_required=True,
                consent_management_required=True,
                data_minimization_required=True,
                right_to_erasure=True,
                breach_notification_hours=72
            ),
            
            # Japan APPI
            ExtendedComplianceFramework.APPI: ComplianceRequirement(
                framework=ComplianceFramework.GDPR,  # Use base enum for compatibility
                min_epsilon=0.7,
                max_delta=1e-4,
                data_residency_required=False,  # Cross-border transfers allowed
                audit_log_retention_days=1095,  # 3 years
                encryption_at_rest_required=True,
                encryption_in_transit_required=True,
                consent_management_required=True,
                data_minimization_required=True,
                right_to_erasure=True,
                breach_notification_hours=0  # No specific timeframe
            ),
            
            # Turkey KVKK
            ExtendedComplianceFramework.KVKK: ComplianceRequirement(
                framework=ComplianceFramework.GDPR,  # Use base enum for compatibility
                min_epsilon=0.4,
                max_delta=1e-5,
                data_residency_required=True,
                audit_log_retention_days=2190,  # 6 years
                encryption_at_rest_required=True,
                encryption_in_transit_required=True,
                consent_management_required=True,
                data_minimization_required=True,
                right_to_erasure=True,
                breach_notification_hours=72
            )
        }
    
    def _initialize_cross_border_rules(self) -> Dict[str, Any]:
        """Initialize cross-border data transfer rules."""
        return {
            "prohibited_transfers": [
                ("ap-east-1", "us-east-1"),  # China to US requires special approval
                ("me-south-1", "eu-west-1"),  # UAE to EU requires adequacy
            ],
            "restricted_transfers": [
                ("eu-west-1", "us-east-1"),  # EU to US requires SCCs
                ("eu-central-1", "ap-southeast-1"),  # EU to APAC requires SCCs
            ],
            "approved_mechanisms": {
                "standard_contractual_clauses": ["eu-us-scc-2021", "eu-uk-scc-2021"],
                "binding_corporate_rules": ["global-bcr-2023"],
                "adequacy_decisions": ["eu-uk", "eu-japan", "eu-canada"],
                "certification_schemes": ["iso27001", "soc2-type2"]
            }
        }
    
    def validate_extended_compliance(
        self, 
        epsilon: float, 
        delta: float, 
        frameworks: List[ExtendedComplianceFramework],
        data_origin_region: Optional[ExtendedRegion] = None,
        data_destination_region: Optional[ExtendedRegion] = None
    ) -> Dict[str, Any]:
        """Enhanced compliance validation with cross-border considerations."""
        
        validation_results = {
            "framework_compliance": {},
            "cross_border_compliance": {},
            "overall_status": "PENDING",
            "recommendations": [],
            "required_safeguards": []
        }
        
        # Validate against extended frameworks
        for framework in frameworks:
            if framework in self.extended_frameworks:
                requirement = self.extended_frameworks[framework]
                
                epsilon_valid = epsilon >= requirement.min_epsilon
                delta_valid = delta <= requirement.max_delta
                compliant = epsilon_valid and delta_valid
                
                validation_results["framework_compliance"][framework.value] = {
                    "compliant": compliant,
                    "epsilon_requirement": requirement.min_epsilon,
                    "delta_requirement": requirement.max_delta,
                    "epsilon_provided": epsilon,
                    "delta_provided": delta,
                    "data_residency_required": requirement.data_residency_required,
                    "additional_requirements": {
                        "encryption_at_rest": requirement.encryption_at_rest_required,
                        "encryption_in_transit": requirement.encryption_in_transit_required,
                        "consent_management": requirement.consent_management_required,
                        "right_to_erasure": requirement.right_to_erasure
                    }
                }
                
                if not compliant:
                    validation_results["recommendations"].append({
                        "framework": framework.value,
                        "issue": f"Privacy parameters insufficient: ε≥{requirement.min_epsilon}, δ≤{requirement.max_delta}",
                        "priority": "HIGH"
                    })
        
        # Validate cross-border transfers if regions specified
        if data_origin_region and data_destination_region:
            transfer_status = self._validate_cross_border_transfer(
                data_origin_region, data_destination_region
            )
            validation_results["cross_border_compliance"] = transfer_status
            
            if not transfer_status["allowed"]:
                validation_results["recommendations"].append({
                    "type": "cross_border",
                    "issue": transfer_status["reason"],
                    "required_safeguards": transfer_status["required_safeguards"],
                    "priority": "CRITICAL"
                })
        
        # Determine overall compliance status
        framework_compliant = all(
            result["compliant"] for result in validation_results["framework_compliance"].values()
        )
        cross_border_compliant = validation_results["cross_border_compliance"].get("allowed", True)
        
        if framework_compliant and cross_border_compliant:
            validation_results["overall_status"] = "COMPLIANT"
        elif len(validation_results["recommendations"]) == 0:
            validation_results["overall_status"] = "COMPLIANT_WITH_CONDITIONS"
        else:
            validation_results["overall_status"] = "NON_COMPLIANT"
        
        # Log comprehensive audit entry
        self._log_enhanced_compliance_check(validation_results, frameworks, epsilon, delta)
        
        return validation_results
    
    def _validate_cross_border_transfer(
        self, 
        origin: ExtendedRegion, 
        destination: ExtendedRegion
    ) -> Dict[str, Any]:
        """Validate cross-border data transfer compliance."""
        
        origin_dest_pair = (origin.value, destination.value)
        
        # Check prohibited transfers
        if origin_dest_pair in self.cross_border_rules["prohibited_transfers"]:
            return {
                "allowed": False,
                "reason": "Transfer prohibited by regulatory framework",
                "required_safeguards": ["regulatory_approval", "special_authorization"],
                "compliance_framework": "multiple"
            }
        
        # Check restricted transfers
        if origin_dest_pair in self.cross_border_rules["restricted_transfers"]:
            return {
                "allowed": True,
                "reason": "Transfer requires additional safeguards",
                "required_safeguards": ["standard_contractual_clauses", "data_mapping", "impact_assessment"],
                "compliance_framework": "gdpr"
            }
        
        # EU originating transfers (special handling)
        if origin.value.startswith("eu-"):
            if destination.value.startswith("us-"):
                return {
                    "allowed": True,
                    "reason": "EU-US transfer requires SCCs and supplementary measures",
                    "required_safeguards": ["eu_us_scc_2021", "supplementary_measures", "transfer_impact_assessment"],
                    "compliance_framework": "gdpr"
                }
            elif destination.value.startswith("ap-"):
                return {
                    "allowed": True,
                    "reason": "EU-APAC transfer requires SCCs",
                    "required_safeguards": ["standard_contractual_clauses", "adequacy_assessment"],
                    "compliance_framework": "gdpr"
                }
        
        # China originating transfers (special handling)
        if origin.value in ["ap-east-1"]:  # Hong Kong/China region
            return {
                "allowed": False,
                "reason": "China data localization requirements may apply",
                "required_safeguards": ["regulatory_approval", "security_assessment", "national_security_review"],
                "compliance_framework": "pipl"
            }
        
        # Default: transfer allowed
        return {
            "allowed": True,
            "reason": "Transfer permitted under standard terms",
            "required_safeguards": [],
            "compliance_framework": "general"
        }
    
    def _log_enhanced_compliance_check(
        self, 
        validation_results: Dict[str, Any], 
        frameworks: List[ExtendedComplianceFramework],
        epsilon: float, 
        delta: float
    ):
        """Log enhanced compliance check for audit trail."""
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "check_type": "enhanced_compliance",
            "frameworks": [f.value for f in frameworks],
            "privacy_parameters": {"epsilon": epsilon, "delta": delta},
            "overall_status": validation_results["overall_status"],
            "framework_results": validation_results["framework_compliance"],
            "cross_border_results": validation_results["cross_border_compliance"],
            "recommendations_count": len(validation_results["recommendations"]),
            "audit_id": f"EC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
        
        self.audit_logs.append(log_entry)
        
        logger.info(
            f"Enhanced compliance check [{validation_results['overall_status']}]: "
            f"{len(frameworks)} frameworks, ε={epsilon}, δ={delta}"
        )
    
    def generate_global_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive global compliance report."""
        
        now = datetime.now(timezone.utc)
        
        # Analyze recent compliance checks
        recent_checks = [
            log for log in self.audit_logs 
            if (now - datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))).days <= 30
        ]
        
        # Framework coverage analysis
        framework_coverage = {}
        for framework in ExtendedComplianceFramework:
            framework_checks = [
                check for check in recent_checks 
                if framework.value in check.get("frameworks", [])
            ]
            
            if framework_checks:
                compliant_count = sum(
                    1 for check in framework_checks 
                    if check["overall_status"] in ["COMPLIANT", "COMPLIANT_WITH_CONDITIONS"]
                )
                framework_coverage[framework.value] = {
                    "total_checks": len(framework_checks),
                    "compliant_checks": compliant_count,
                    "compliance_rate": compliant_count / len(framework_checks),
                    "last_check": framework_checks[-1]["timestamp"]
                }
        
        # Cross-border transfer analysis
        cross_border_checks = [
            check for check in recent_checks 
            if "cross_border_results" in check and check["cross_border_results"]
        ]
        
        cross_border_summary = {
            "total_transfers_evaluated": len(cross_border_checks),
            "allowed_transfers": sum(
                1 for check in cross_border_checks 
                if check["cross_border_results"].get("allowed", False)
            ),
            "restricted_transfers": sum(
                1 for check in cross_border_checks 
                if check["cross_border_results"].get("allowed", False) and 
                check["cross_border_results"].get("required_safeguards", [])
            ),
            "prohibited_transfers": sum(
                1 for check in cross_border_checks 
                if not check["cross_border_results"].get("allowed", True)
            )
        }
        
        return {
            "report_generated": now.isoformat(),
            "report_period_days": 30,
            "total_compliance_checks": len(recent_checks),
            "framework_coverage": framework_coverage,
            "cross_border_analysis": cross_border_summary,
            "compliance_trends": self._analyze_compliance_trends(recent_checks),
            "recommendations": self._generate_strategic_recommendations(framework_coverage, cross_border_summary),
            "audit_trail_entries": len(self.audit_logs),
            "supported_frameworks": [f.value for f in ExtendedComplianceFramework],
            "supported_regions": [r.value for r in ExtendedRegion]
        }
    
    def _analyze_compliance_trends(self, recent_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze compliance trends over time."""
        
        if not recent_checks:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_checks = sorted(recent_checks, key=lambda x: x["timestamp"])
        
        # Calculate weekly compliance rates
        weekly_rates = []
        current_week = []
        current_week_start = None
        
        for check in sorted_checks:
            check_date = datetime.fromisoformat(check["timestamp"].replace('Z', '+00:00'))
            week_start = check_date - timezone.utc.normalize(check_date.weekday() * 24 * 60 * 60)
            
            if current_week_start != week_start:
                if current_week:
                    compliant = sum(
                        1 for c in current_week 
                        if c["overall_status"] in ["COMPLIANT", "COMPLIANT_WITH_CONDITIONS"]
                    )
                    weekly_rates.append(compliant / len(current_week))
                
                current_week = [check]
                current_week_start = week_start
            else:
                current_week.append(check)
        
        # Add final week
        if current_week:
            compliant = sum(
                1 for c in current_week 
                if c["overall_status"] in ["COMPLIANT", "COMPLIANT_WITH_CONDITIONS"]
            )
            weekly_rates.append(compliant / len(current_week))
        
        if len(weekly_rates) < 2:
            return {"trend": "insufficient_data", "weekly_rates": weekly_rates}
        
        # Calculate trend
        if weekly_rates[-1] > weekly_rates[0]:
            trend = "improving"
        elif weekly_rates[-1] < weekly_rates[0]:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "current_rate": weekly_rates[-1],
            "previous_rate": weekly_rates[0],
            "weekly_rates": weekly_rates,
            "improvement": weekly_rates[-1] - weekly_rates[0]
        }
    
    def _generate_strategic_recommendations(
        self, 
        framework_coverage: Dict[str, Any], 
        cross_border_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate strategic compliance recommendations."""
        
        recommendations = []
        
        # Framework-specific recommendations
        for framework, stats in framework_coverage.items():
            if stats["compliance_rate"] < 0.8:
                recommendations.append({
                    "type": "framework_improvement",
                    "framework": framework,
                    "priority": "HIGH",
                    "recommendation": f"Improve {framework.upper()} compliance rate from {stats['compliance_rate']:.1%} to >80%",
                    "suggested_actions": [
                        "Review privacy parameter settings",
                        "Implement additional technical safeguards",
                        "Update data processing procedures"
                    ]
                })
        
        # Cross-border recommendations
        if cross_border_summary["prohibited_transfers"] > 0:
            recommendations.append({
                "type": "cross_border_improvement",
                "priority": "CRITICAL",
                "recommendation": f"{cross_border_summary['prohibited_transfers']} prohibited transfers detected",
                "suggested_actions": [
                    "Review data residency requirements",
                    "Implement local processing capabilities",
                    "Seek regulatory approval for necessary transfers"
                ]
            })
        
        if cross_border_summary["restricted_transfers"] > cross_border_summary["total_transfers_evaluated"] * 0.5:
            recommendations.append({
                "type": "transfer_optimization",
                "priority": "MEDIUM",
                "recommendation": "High percentage of transfers require additional safeguards",
                "suggested_actions": [
                    "Implement Standard Contractual Clauses",
                    "Conduct Transfer Impact Assessments",
                    "Consider regional data processing strategies"
                ]
            })
        
        return recommendations


# Factory function for enhanced global compliance
def create_enhanced_compliance_manager() -> EnhancedComplianceManager:
    """Create an enhanced compliance manager with global coverage."""
    return EnhancedComplianceManager()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    manager = create_enhanced_compliance_manager()
    
    # Test enhanced compliance validation
    result = manager.validate_extended_compliance(
        epsilon=0.2,
        delta=1e-6,
        frameworks=[ExtendedComplianceFramework.PIPL, ExtendedComplianceFramework.GDPR],
        data_origin_region=ExtendedRegion.AP_EAST_1,
        data_destination_region=ExtendedRegion.EU_WEST_1
    )
    
    print(f"Enhanced compliance status: {result['overall_status']}")
    print(f"Recommendations: {len(result['recommendations'])}")
    
    # Generate comprehensive report
    report = manager.generate_global_compliance_report()
    print(f"Global compliance report: {len(report['supported_frameworks'])} frameworks supported")