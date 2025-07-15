"""
Mostly score calculation functions for the AI SBOM Generator.
"""

import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from .field_registry_manager import (
    get_field_registry_manager,
    generate_field_classification,
    generate_completeness_profiles,
    generate_validation_messages,
    get_configurable_scoring_weights,
    DynamicFieldDetector  # Compatibility wrapper
)

logger = logging.getLogger(__name__)

# Validation severity levels
class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

# Registry-driven field definitions
try:
    REGISTRY_MANAGER = get_field_registry_manager()
    FIELD_CLASSIFICATION = generate_field_classification()
    COMPLETENESS_PROFILES = generate_completeness_profiles()
    VALIDATION_MESSAGES = generate_validation_messages()
    SCORING_WEIGHTS = get_configurable_scoring_weights()
    
    print(f"✅ Registry-driven configuration loaded: {len(FIELD_CLASSIFICATION)} fields")
    REGISTRY_AVAILABLE = True
    
except Exception as e:
    print(f"❌ Failed to load registry configuration: {e}")
    print("🔄 Falling back to hardcoded definitions...")
    REGISTRY_AVAILABLE = False
    
    # Hardcoded definitions as fallback
    FIELD_CLASSIFICATION = {
        # Critical fields (silently aligned with SPDX mandatory fields)
        "bomFormat": {"tier": "critical", "weight": 3, "category": "required_fields"},
        "specVersion": {"tier": "critical", "weight": 3, "category": "required_fields"},
        "serialNumber": {"tier": "critical", "weight": 3, "category": "required_fields"},
        "version": {"tier": "critical", "weight": 3, "category": "required_fields"},
        "name": {"tier": "critical", "weight": 4, "category": "component_basic"},
        "downloadLocation": {"tier": "critical", "weight": 4, "category": "external_references"},
        "primaryPurpose": {"tier": "critical", "weight": 3, "category": "metadata"},
        "suppliedBy": {"tier": "critical", "weight": 4, "category": "metadata"},
        
        # Important fields (aligned with key SPDX optional fields)
        "type": {"tier": "important", "weight": 2, "category": "component_basic"},
        "purl": {"tier": "important", "weight": 4, "category": "component_basic"},
        "description": {"tier": "important", "weight": 4, "category": "component_basic"},
        "licenses": {"tier": "important", "weight": 4, "category": "component_basic"},
        "energyConsumption": {"tier": "important", "weight": 3, "category": "component_model_card"},
        "hyperparameter": {"tier": "important", "weight": 3, "category": "component_model_card"},
        "limitation": {"tier": "important", "weight": 3, "category": "component_model_card"},
        "safetyRiskAssessment": {"tier": "important", "weight": 3, "category": "component_model_card"},
        "typeOfModel": {"tier": "important", "weight": 3, "category": "component_model_card"},
        
        # Supplementary fields (aligned with remaining SPDX optional fields)
        "modelExplainability": {"tier": "supplementary", "weight": 2, "category": "component_model_card"},
        "standardCompliance": {"tier": "supplementary", "weight": 2, "category": "metadata"},
        "domain": {"tier": "supplementary", "weight": 2, "category": "metadata"},
        "energyQuantity": {"tier": "supplementary", "weight": 2, "category": "component_model_card"},
        "energyUnit": {"tier": "supplementary", "weight": 2, "category": "component_model_card"},
        "informationAboutTraining": {"tier": "supplementary", "weight": 2, "category": "component_model_card"},
        "informationAboutApplication": {"tier": "supplementary", "weight": 2, "category": "component_model_card"},
        "metric": {"tier": "supplementary", "weight": 2, "category": "component_model_card"},
        "metricDecisionThreshold": {"tier": "supplementary", "weight": 2, "category": "component_model_card"},
        "modelDataPreprocessing": {"tier": "supplementary", "weight": 2, "category": "component_model_card"},
        "autonomyType": {"tier": "supplementary", "weight": 1, "category": "metadata"},
        "useSensitivePersonalInformation": {"tier": "supplementary", "weight": 2, "category": "component_model_card"}
    }
    
    # Completeness profiles (silently aligned with SPDX requirements)
    COMPLETENESS_PROFILES = {
        "basic": {
            "description": "Minimal fields required for identification",
            "required_fields": ["bomFormat", "specVersion", "serialNumber", "version", "name"],
            "minimum_score": 40
        },
        "standard": {
            "description": "Comprehensive fields for proper documentation",
            "required_fields": ["bomFormat", "specVersion", "serialNumber", "version", "name", 
                               "downloadLocation", "primaryPurpose", "suppliedBy"],
            "minimum_score": 70
        },
        "advanced": {
            "description": "Extensive documentation for maximum transparency",
            "required_fields": ["bomFormat", "specVersion", "serialNumber", "version", "name", 
                               "downloadLocation", "primaryPurpose", "suppliedBy",
                               "type", "purl", "description", "licenses", "hyperparameter", "limitation", 
                               "energyConsumption", "safetyRiskAssessment", "typeOfModel"],
            "minimum_score": 85
        }
    }
    
    # Validation messages framed as best practices
    VALIDATION_MESSAGES = {
        "name": {
            "missing": "Missing critical field: name - essential for model identification",
            "recommendation": "Add a descriptive name for the model"
        },
        "downloadLocation": {
            "missing": "Missing critical field: downloadLocation - needed for artifact retrieval",
            "recommendation": "Add information about where the model can be downloaded"
        },
        "primaryPurpose": {
            "missing": "Missing critical field: primaryPurpose - important for understanding model intent",
            "recommendation": "Add information about the primary purpose of this model"
        },
        "suppliedBy": {
            "missing": "Missing critical field: suppliedBy - needed for provenance tracking",
            "recommendation": "Add information about who supplied this model"
        },
        "energyConsumption": {
            "missing": "Missing important field: energyConsumption - helpful for environmental impact assessment",
            "recommendation": "Consider documenting energy consumption metrics for better transparency"
        },
        "hyperparameter": {
            "missing": "Missing important field: hyperparameter - valuable for reproducibility",
            "recommendation": "Document key hyperparameters used in training"
        },
        "limitation": {
            "missing": "Missing important field: limitation - important for responsible use",
            "recommendation": "Document known limitations of the model to guide appropriate usage"
        }
    }
    
    SCORING_WEIGHTS = {
        "tier_weights": {"critical": 3, "important": 2, "supplementary": 1},
        "category_weights": {
            "required_fields": 20, "metadata": 20, "component_basic": 20,
            "component_model_card": 30, "external_references": 10
        },
        "algorithm_config": {"type": "weighted_sum", "max_score": 100}
    }


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def generate_uuid():
    return str(uuid.uuid4())


def normalize_license_id(license_text):
    license_mappings = {
        "mit": "MIT",
        "apache": "Apache-2.0",
        "apache 2": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "apache-2": "Apache-2.0",
        "apache-2.0": "Apache-2.0",
        "gpl": "GPL-3.0-only",
        "gpl-3": "GPL-3.0-only",
        "gpl-3.0": "GPL-3.0-only",
        "gpl3": "GPL-3.0-only",
        "gpl v3": "GPL-3.0-only",
        "gpl-2": "GPL-2.0-only",
        "gpl-2.0": "GPL-2.0-only",
        "gpl2": "GPL-2.0-only",
        "gpl v2": "GPL-2.0-only",
        "lgpl": "LGPL-3.0-only",
        "lgpl-3": "LGPL-3.0-only",
        "lgpl-3.0": "LGPL-3.0-only",
        "bsd": "BSD-3-Clause",
        "bsd-3": "BSD-3-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "bsd-2": "BSD-2-Clause",
        "bsd-2-clause": "BSD-2-Clause",
        "cc": "CC-BY-4.0",
        "cc-by": "CC-BY-4.0",
        "cc-by-4.0": "CC-BY-4.0",
        "cc-by-sa": "CC-BY-SA-4.0",
        "cc-by-sa-4.0": "CC-BY-SA-4.0",
        "cc-by-nc": "CC-BY-NC-4.0",
        "cc-by-nc-4.0": "CC-BY-NC-4.0",
        "cc0": "CC0-1.0",
        "cc0-1.0": "CC0-1.0",
        "public domain": "CC0-1.0",
        "unlicense": "Unlicense",
        "proprietary": "NONE",
        "commercial": "NONE",
    }

    if not license_text:
        return None

    normalized = re.sub(r'[^\w\s-]', '', license_text.lower())

    if normalized in license_mappings:
        return license_mappings[normalized]

    for key, value in license_mappings.items():
        if key in normalized:
            return value

    return license_text


def validate_spdx(license_entry):
    spdx_licenses = [
        "MIT", "Apache-2.0", "GPL-3.0-only", "GPL-2.0-only", "LGPL-3.0-only",
        "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0",
        "Unlicense", "NONE"
    ]
    if isinstance(license_entry, list):
        return all(lic in spdx_licenses for lic in license_entry)
    return license_entry in spdx_licenses


def check_field_in_aibom(aibom: Dict[str, Any], field: str) -> bool:
    """
    Check if a field is present in the AIBOM.
    
    Args:
        aibom: The AIBOM to check
        field: The field name to check
        
    Returns:
        True if the field is present, False otherwise
    """
    if field in aibom:
        return True
    if "metadata" in aibom:
        metadata = aibom["metadata"]
        if field in metadata:
            return True
        if "properties" in metadata:
            for prop in metadata["properties"]:
                prop_name = prop.get("name", "")
                if prop_name in {field, f"spdx:{field}"}:
                    return True
    if "components" in aibom and aibom["components"]:
        component = aibom["components"][0]
        if field in component:
            return True
        if "properties" in component:
            for prop in component["properties"]:
                prop_name = prop.get("name", "")
                if prop_name in {field, f"spdx:{field}"}:
                    return True
        if "modelCard" in component:
            model_card = component["modelCard"]
            if field in model_card:
                return True
            if "modelParameters" in model_card and field in model_card["modelParameters"]:
                return True
            if "considerations" in model_card:
                considerations = model_card["considerations"]
                field_mappings = {
                    "limitation": ["technicalLimitations", "limitations"],
                    "safetyRiskAssessment": ["ethicalConsiderations", "safetyRiskAssessment"],
                    "energyConsumption": ["environmentalConsiderations", "energyConsumption"]
                }
                if field in field_mappings:
                    for section in field_mappings[field]:
                        if section in considerations and considerations[section]:
                            return True
                if field in considerations:
                    return True
    if field == "downloadLocation" and "externalReferences" in aibom:
        for ref in aibom["externalReferences"]:
            if ref.get("type") == "distribution" and ref.get("url"):
                return True
    return False



def determine_completeness_profile(aibom: Dict[str, Any], score: float) -> Dict[str, Any]:
    """
    Determine which completeness profile the AIBOM satisfies.
    
    Args:
        aibom: The AIBOM to check
        score: The calculated score
        
    Returns:
        Dictionary with profile information
    """
    satisfied_profiles = []
    
    for profile_name, profile in COMPLETENESS_PROFILES.items():
        # Check if all required fields are present
        all_required_present = all(check_field_in_aibom(aibom, field) for field in profile["required_fields"])
        
        # Check if score meets minimum
        score_sufficient = score >= profile["minimum_score"]
        
        if all_required_present and score_sufficient:
            satisfied_profiles.append(profile_name)
    
    # Return the highest satisfied profile
    if "advanced" in satisfied_profiles:
        return {
            "name": "Advanced",
            "description": COMPLETENESS_PROFILES["advanced"]["description"],
            "satisfied": True
        }
    elif "standard" in satisfied_profiles:
        return {
            "name": "Standard",
            "description": COMPLETENESS_PROFILES["standard"]["description"],
            "satisfied": True
        }
    elif "basic" in satisfied_profiles:
        return {
            "name": "Basic",
            "description": COMPLETENESS_PROFILES["basic"]["description"],
            "satisfied": True
        }
    else:
        return {
            "name": "incomplete",
            "description": "Does not satisfy any completeness profile",
            "satisfied": False
        }


def apply_completeness_penalties(original_score: float, missing_fields: Dict[str, List[str]]) -> Dict[str, Any]:

    """
    Apply penalties based on missing critical fields.
    
    Args:
        original_score: The original calculated score
        missing_fields: Dictionary of missing fields by tier
        
    Returns:
        Dictionary with penalty information
    """
    
    
    # Count missing fields by tier
    missing_critical_count = len(missing_fields["critical"])
    missing_important_count = len(missing_fields["important"])
    
    penalty_factor = 1.0
    penalty_reason = None
    
    # Calculate penalty based on missing critical fields
    if missing_critical_count > 3:
        penalty_factor *= 0.8  # 20% penalty
        penalty_reason = "Multiple critical fields missing"
    elif missing_critical_count >= 2: # if count is 2 - 3
        penalty_factor *= 0.9  # 10% penalty
        penalty_reason = "Some critical fields missing"
        
    if missing_important_count >= 5:
        penalty_factor *= 0.95  # 5% penalty
        penalty_reason = "Several important fields missing"
    
    adjusted_score = original_score * penalty_factor
    
    return {
        "adjusted_score": round(adjusted_score, 1),  # Round to 1 decimal place
        "penalty_applied": penalty_reason is not None,
        "penalty_reason": penalty_reason,
        "penalty_factor": penalty_factor
    }


def generate_field_recommendations(missing_fields: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Generate recommendations for missing fields.
    
    Args:
        missing_fields: Dictionary of missing fields by tier
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Prioritize critical fields
    for field in missing_fields["critical"]:
        if field in VALIDATION_MESSAGES:
            recommendations.append({
                "priority": "high",
                "field": field,
                "message": VALIDATION_MESSAGES[field]["missing"],
                "recommendation": VALIDATION_MESSAGES[field]["recommendation"]
            })
        else:
            recommendations.append({
                "priority": "high",
                "field": field,
                "message": f"Missing critical field: {field}",
                "recommendation": f"Add {field} to improve documentation completeness"
            })
    
    # Then important fields
    for field in missing_fields["important"]:
        if field in VALIDATION_MESSAGES:
            recommendations.append({
                "priority": "medium",
                "field": field,
                "message": VALIDATION_MESSAGES[field]["missing"],
                "recommendation": VALIDATION_MESSAGES[field]["recommendation"]
            })
        else:
            recommendations.append({
                "priority": "medium",
                "field": field,
                "message": f"Missing important field: {field}",
                "recommendation": f"Consider adding {field} for better documentation"
            })
    
    # Finally supplementary fields (limit to top 5)
    supplementary_count = 0
    for field in missing_fields["supplementary"]:
        if supplementary_count >= 5:
            break
            
        recommendations.append({
            "priority": "low",
            "field": field,
            "message": f"Missing supplementary field: {field}",
            "recommendation": f"Consider adding {field} for comprehensive documentation"
        })
        supplementary_count += 1
    
    return recommendations


def _validate_ai_requirements(aibom: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate AI-specific requirements for an AIBOM.
    
    Args:
        aibom: The AIBOM to validate
        
    Returns:
        List of validation issues
    """
    issues = []
    issue_codes = set()
    
    # Check required fields
    for field in ["bomFormat", "specVersion", "serialNumber", "version"]:
        if field not in aibom:
            issues.append({
                "severity": ValidationSeverity.ERROR.value,
                "code": f"MISSING_{field.upper()}",
                "message": f"Missing required field: {field}",
                "path": f"$.{field}"
            })
            issue_codes.add(f"MISSING_{field.upper()}")
    
    # Check bomFormat
    if "bomFormat" in aibom and aibom["bomFormat"] != "CycloneDX":
        issues.append({
            "severity": ValidationSeverity.ERROR.value,
            "code": "INVALID_BOM_FORMAT",
            "message": f"Invalid bomFormat: {aibom['bomFormat']}. Must be 'CycloneDX'",
            "path": "$.bomFormat"
        })
        issue_codes.add("INVALID_BOM_FORMAT")
    
    # Check specVersion
    if "specVersion" in aibom and aibom["specVersion"] != "1.6":
        issues.append({
            "severity": ValidationSeverity.ERROR.value,
            "code": "INVALID_SPEC_VERSION",
            "message": f"Invalid specVersion: {aibom['specVersion']}. Must be '1.6'",
            "path": "$.specVersion"
        })
        issue_codes.add("INVALID_SPEC_VERSION")
    
    # Check serialNumber
    if "serialNumber" in aibom and not aibom["serialNumber"].startswith("urn:uuid:"):
        issues.append({
            "severity": ValidationSeverity.ERROR.value,
            "code": "INVALID_SERIAL_NUMBER",
            "message": f"Invalid serialNumber format: {aibom['serialNumber']}. Must start with 'urn:uuid:'",
            "path": "$.serialNumber"
        })
        issue_codes.add("INVALID_SERIAL_NUMBER")
    
    # Check version
    if "version" in aibom:
        if not isinstance(aibom["version"], int):
            issues.append({
                "severity": ValidationSeverity.ERROR.value,
                "code": "INVALID_VERSION_TYPE",
                "message": f"Invalid version type: {type(aibom['version'])}. Must be an integer",
                "path": "$.version"
            })
            issue_codes.add("INVALID_VERSION_TYPE")
        elif aibom["version"] <= 0:
            issues.append({
                "severity": ValidationSeverity.ERROR.value,
                "code": "INVALID_VERSION_VALUE",
                "message": f"Invalid version value: {aibom['version']}. Must be positive",
                "path": "$.version"
            })
            issue_codes.add("INVALID_VERSION_VALUE")
    
    # Check metadata
    if "metadata" not in aibom:
        issues.append({
            "severity": ValidationSeverity.ERROR.value,
            "code": "MISSING_METADATA",
            "message": "Missing metadata section",
            "path": "$.metadata"
        })
        issue_codes.add("MISSING_METADATA")
    else:
        metadata = aibom["metadata"]
        
        # Check timestamp
        if "timestamp" not in metadata:
            issues.append({
                "severity": ValidationSeverity.WARNING.value,
                "code": "MISSING_TIMESTAMP",
                "message": "Missing timestamp in metadata",
                "path": "$.metadata.timestamp"
            })
            issue_codes.add("MISSING_TIMESTAMP")
        
        # Check tools
        if "tools" not in metadata or not metadata["tools"] or len(metadata["tools"]) == 0:
            issues.append({
                "severity": ValidationSeverity.WARNING.value,
                "code": "MISSING_TOOLS",
                "message": "Missing tools in metadata",
                "path": "$.metadata.tools"
            })
            issue_codes.add("MISSING_TOOLS")
        
        # Check authors
        if "authors" not in metadata or not metadata["authors"] or len(metadata["authors"]) == 0:
            issues.append({
                "severity": ValidationSeverity.WARNING.value,
                "code": "MISSING_AUTHORS",
                "message": "Missing authors in metadata",
                "path": "$.metadata.authors"
            })
            issue_codes.add("MISSING_AUTHORS")
        else:
            # Check author properties
            for i, author in enumerate(metadata["authors"]):
                if "url" in author:
                    issues.append({
                        "severity": ValidationSeverity.ERROR.value,
                        "code": "INVALID_AUTHOR_PROPERTY",
                        "message": "Author objects should not contain 'url' property, use 'email' instead",
                        "path": f"$.metadata.authors[{i}].url"
                    })
                    issue_codes.add("INVALID_AUTHOR_PROPERTY")
        
        # Check properties
        if "properties" not in metadata or not metadata["properties"] or len(metadata["properties"]) == 0:
            issues.append({
                "severity": ValidationSeverity.INFO.value,
                "code": "MISSING_PROPERTIES",
                "message": "Missing properties in metadata",
                "path": "$.metadata.properties"
            })
            issue_codes.add("MISSING_PROPERTIES")
    
    # Check components
    if "components" not in aibom or not aibom["components"] or len(aibom["components"]) == 0:
        issues.append({
            "severity": ValidationSeverity.ERROR.value,
            "code": "MISSING_COMPONENTS",
            "message": "Missing components section or empty components array",
            "path": "$.components"
        })
        issue_codes.add("MISSING_COMPONENTS")
    else:
        components = aibom["components"]
        
        # Check first component (AI model)
        component = components[0]
        
        # Check type
        if "type" not in component:
            issues.append({
                "severity": ValidationSeverity.ERROR.value,
                "code": "MISSING_COMPONENT_TYPE",
                "message": "Missing type in first component",
                "path": "$.components[0].type"
            })
            issue_codes.add("MISSING_COMPONENT_TYPE")
        elif component["type"] != "machine-learning-model":
            issues.append({
                "severity": ValidationSeverity.ERROR.value,
                "code": "INVALID_COMPONENT_TYPE",
                "message": f"Invalid type in first component: {component['type']}. Must be 'machine-learning-model'",
                "path": "$.components[0].type"
            })
            issue_codes.add("INVALID_COMPONENT_TYPE")
        
        # Check name
        if "name" not in component or not component["name"]:
            issues.append({
                "severity": ValidationSeverity.ERROR.value,
                "code": "MISSING_COMPONENT_NAME",
                "message": "Missing name in first component",
                "path": "$.components[0].name"
            })
            issue_codes.add("MISSING_COMPONENT_NAME")
        
        # Check bom-ref
        if "bom-ref" not in component or not component["bom-ref"]:
            issues.append({
                "severity": ValidationSeverity.ERROR.value,
                "code": "MISSING_BOM_REF",
                "message": "Missing bom-ref in first component",
                "path": "$.components[0].bom-ref"
            })
            issue_codes.add("MISSING_BOM_REF")
        
        # Check purl
        if "purl" not in component or not component["purl"]:
            issues.append({
                "severity": ValidationSeverity.WARNING.value,
                "code": "MISSING_PURL",
                "message": "Missing purl in first component",
                "path": "$.components[0].purl"
            })
            issue_codes.add("MISSING_PURL")
        elif not component["purl"].startswith("pkg:"):
            issues.append({
                "severity": ValidationSeverity.ERROR.value,
                "code": "INVALID_PURL_FORMAT",
                "message": f"Invalid purl format: {component['purl']}. Must start with 'pkg:'",
                "path": "$.components[0].purl"
            })
            issue_codes.add("INVALID_PURL_FORMAT")
        elif "@" not in component["purl"]:
            issues.append({
                "severity": ValidationSeverity.WARNING.value,
                "code": "MISSING_VERSION_IN_PURL",
                "message": f"Missing version in purl: {component['purl']}. Should include version after '@'",
                "path": "$.components[0].purl"
            })
            issue_codes.add("MISSING_VERSION_IN_PURL")
        
        # Check description
        if "description" not in component or not component["description"]:
            issues.append({
                "severity": ValidationSeverity.WARNING.value,
                "code": "MISSING_DESCRIPTION",
                "message": "Missing description in first component",
                "path": "$.components[0].description"
            })
            issue_codes.add("MISSING_DESCRIPTION")
        elif len(component["description"]) < 20:
            issues.append({
                "severity": ValidationSeverity.INFO.value,
                "code": "SHORT_DESCRIPTION",
                "message": f"Description is too short: {len(component['description'])} characters. Recommended minimum is 20 characters",
                "path": "$.components[0].description"
            })
            issue_codes.add("SHORT_DESCRIPTION")
        
        # Check modelCard
        if "modelCard" not in component or not component["modelCard"]:
            issues.append({
                "severity": ValidationSeverity.WARNING.value,
                "code": "MISSING_MODEL_CARD",
                "message": "Missing modelCard in first component",
                "path": "$.components[0].modelCard"
            })
            issue_codes.add("MISSING_MODEL_CARD")
        else:
            model_card = component["modelCard"]
            
            # Check modelParameters
            if "modelParameters" not in model_card or not model_card["modelParameters"]:
                issues.append({
                    "severity": ValidationSeverity.WARNING.value,
                    "code": "MISSING_MODEL_PARAMETERS",
                    "message": "Missing modelParameters in modelCard",
                    "path": "$.components[0].modelCard.modelParameters"
                })
                issue_codes.add("MISSING_MODEL_PARAMETERS")
            
            # Check considerations
            if "considerations" not in model_card or not model_card["considerations"]:
                issues.append({
                    "severity": ValidationSeverity.WARNING.value,
                    "code": "MISSING_CONSIDERATIONS",
                    "message": "Missing considerations in modelCard",
                    "path": "$.components[0].modelCard.considerations"
                })
                issue_codes.add("MISSING_CONSIDERATIONS")
    
    return issues


def _generate_validation_recommendations(issues: List[Dict[str, Any]]) -> List[str]:
    """
    Generate recommendations based on validation issues.
    
    Args:
        issues: List of validation issues
        
    Returns:
        List of recommendations
    """
    recommendations = []
    issue_codes = set(issue["code"] for issue in issues)
    
    # Generate recommendations based on issue codes
    if "MISSING_COMPONENTS" in issue_codes:
        recommendations.append("Add at least one component to the AIBOM")
        
    if "MISSING_COMPONENT_TYPE" in issue_codes or "INVALID_COMPONENT_TYPE" in issue_codes:
        recommendations.append("Ensure all AI components have type 'machine-learning-model'")
        
    if "MISSING_PURL" in issue_codes or "INVALID_PURL_FORMAT" in issue_codes:
        recommendations.append("Ensure all components have a valid PURL starting with 'pkg:'")
        
    if "MISSING_VERSION_IN_PURL" in issue_codes:
        recommendations.append("Include version information in PURLs using '@' syntax (e.g., pkg:huggingface/org/model@version)")
        
    if "MISSING_MODEL_CARD" in issue_codes:
        recommendations.append("Add a model card section to AI components")
        
    if "MISSING_MODEL_PARAMETERS" in issue_codes:
        recommendations.append("Include model parameters in the model card section")
        
    if "MISSING_CONSIDERATIONS" in issue_codes:
        recommendations.append("Add ethical considerations, limitations, and risks to the model card")
        
    if "MISSING_METADATA" in issue_codes:
        recommendations.append("Add metadata section to the AI SBOM")
        
    if "MISSING_TOOLS" in issue_codes:
        recommendations.append("Include tools information in the metadata section")
        
    if "MISSING_AUTHORS" in issue_codes:
        recommendations.append("Add authors information to the metadata section")
        
    if "MISSING_PROPERTIES" in issue_codes:
        recommendations.append("Include additional properties in the metadata section")
        
    if "INVALID_AUTHOR_PROPERTY" in issue_codes:
        recommendations.append("Remove 'url' property from author objects and use 'email' instead to comply with CycloneDX schema")
        
    return recommendations


def validate_aibom(aibom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an AIBOM against AI-specific requirements.
    
    Args:
        aibom: The AIBOM to validate
        
    Returns:
        Validation report with issues and recommendations
    """
    # Initialize validation report
    report = {
        "valid": True,
        "ai_valid": True,
        "issues": [],
        "recommendations": [],
        "summary": {
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0
        }
    }
    
    # Validate AI-specific requirements
    ai_issues = _validate_ai_requirements(aibom)
    if ai_issues:
        report["ai_valid"] = False
        report["valid"] = False
        report["issues"].extend(ai_issues)
        
    # Generate recommendations
    report["recommendations"] = _generate_validation_recommendations(report["issues"])
    
    # Update summary counts
    for issue in report["issues"]:
        if issue["severity"] == ValidationSeverity.ERROR.value:
            report["summary"]["error_count"] += 1
        elif issue["severity"] == ValidationSeverity.WARNING.value:
            report["summary"]["warning_count"] += 1
        elif issue["severity"] == ValidationSeverity.INFO.value:
            report["summary"]["info_count"] += 1
            
    return report


def get_validation_summary(report: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of the validation report.
    
    Args:
        report: Validation report
        
    Returns:
        Human-readable summary
    """
    if report["valid"]:
        summary = "✅ AIBOM is valid and complies with AI requirements.\n"
    else:
        summary = "❌ AIBOM validation failed.\n"
        
    summary += f"\nSummary:\n"
    summary += f"- Errors: {report['summary']['error_count']}\n"
    summary += f"- Warnings: {report['summary']['warning_count']}\n"
    summary += f"- Info: {report['summary']['info_count']}\n"
    
    if not report["valid"]:
        summary += "\nIssues:\n"
        for issue in report["issues"]:
            severity = issue["severity"].upper()
            code = issue["code"]
            message = issue["message"]
            path = issue["path"]
            summary += f"- [{severity}] {code}: {message} (at {path})\n"
        
        summary += "\nRecommendations:\n"
        for i, recommendation in enumerate(report["recommendations"], 1):
            summary += f"{i}. {recommendation}\n"
            
    return summary

def check_field_with_enhanced_results(aibom: Dict[str, Any], field: str, extraction_results: Optional[Dict[str, Any]] = None) -> bool:
    """
    Enhanced field detection using consolidated field registry manager.
    
    Args:
        aibom: The AIBOM to check
        field: The field name to check (must match field registry)
        extraction_results: Enhanced extraction results with confidence levels
        
    Returns:
        True if the field is present and should count toward score, False otherwise
    """
    try:
        # Initialize dynamic field detector (cached)
        if not hasattr(check_field_with_enhanced_results, '_detector'):
            try:
                if REGISTRY_AVAILABLE:
                    # Use the consolidated registry manager
                    registry_manager = get_field_registry_manager()
                    check_field_with_enhanced_results._detector = DynamicFieldDetector(registry_manager)
                    print(f"✅ Dynamic field detector initialized with registry manager")
                else:
                    # Create registry manager from path
                    from field_registry_manager import FieldRegistryManager
                    registry_path = os.path.join(current_dir, "field_registry.json")
                    registry_manager = FieldRegistryManager(registry_path)
                    check_field_with_enhanced_results._detector = DynamicFieldDetector(registry_manager)
                    print(f"✅ Dynamic field detector initialized with fallback registry manager")
                    
            except Exception as e:
                print(f"❌ Failed to initialize dynamic field detector: {e}")
                # Final fallback
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                registry_path = os.path.join(current_dir, "field_registry.json")
                try:
                    check_field_with_enhanced_results._detector = DynamicFieldDetector(registry_path)
                    print(f"🔄 Dynamic field detector initialized with emergency fallback")
                except Exception as final_error:
                    print(f"❌ Complete failure to initialize dynamic field detector: {final_error}")
                    check_field_with_enhanced_results._detector = None
        
        detector = check_field_with_enhanced_results._detector
        
        if detector is None:
            print(f"⚠️  No detector available, using fallback for {field}")
            return check_field_in_aibom(aibom, field)
        
        # First, try dynamic detection from AIBOM structure using ENHANCED REGISTRY FORMAT
        field_found_in_registry = False
        
        # Use the enhanced registry structure (registry['fields'][field_name])
        fields = detector.registry.get('fields', {})
        if field in fields:
            field_found_in_registry = True
            field_config = fields[field]
            field_path = field_config.get('jsonpath', '')
            
            if field_path:
                # Use dynamic detection
                is_present, value = detector.detect_field_presence(aibom, field_path)
                
                if is_present:
                    print(f"✅ DYNAMIC: Found {field} = {value}")
                    return True
                else:
                    print(f"❌ DYNAMIC: Missing {field} at {field_path}")
            else:
                print(f"⚠️  Field '{field}' has no jsonpath defined in registry")
        
        # If field not in registry, log warning but continue
        if not field_found_in_registry:
            print(f"⚠️  WARNING: Field '{field}' not found in field registry")
        
        # Second, check extraction results (existing logic)
        if extraction_results and field in extraction_results:
            extraction_result = extraction_results[field]
            
            # Check if this field has actual extracted data (not just placeholder)
            if hasattr(extraction_result, 'confidence'):
                # Don't count fields with 'none' confidence (placeholders like NOASSERTION)
                if extraction_result.confidence.value == 'none':
                    print(f"❌ EXTRACTION: {field} has 'none' confidence")
                    return False
                # Count fields with medium or high confidence
                is_confident = extraction_result.confidence.value in ['medium', 'high']
                print(f"{'✅' if is_confident else '❌'} EXTRACTION: {field} confidence = {extraction_result.confidence.value}")
                return is_confident
            elif hasattr(extraction_result, 'value'):
                # For simple extraction results, check if value is meaningful
                value = extraction_result.value
                if value in ['NOASSERTION', 'NOT_FOUND', None, '']:
                    print(f"❌ EXTRACTION: {field} has placeholder value: {value}")
                    return False
                print(f"✅ EXTRACTION: {field} = {value}")
                return True
        
        # Third, fallback to original AIBOM detection
        print(f"🔄 FALLBACK: Using original detection for {field}")
        return check_field_in_aibom(aibom, field)
        
    except Exception as e:
        print(f"❌ Error in enhanced field detection for {field}: {e}")
        return check_field_in_aibom(aibom, field)


def calculate_industry_neutral_score(aibom: Dict[str, Any], extraction_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate completeness score using industry best practices with proper normalization and penalties.
    
    Args:
        aibom: The AIBOM to score
        
    Returns:
        Dictionary containing score and recommendations
    """
    field_checklist = {}
    
    # Maximum points for each category (these are the "weights")
    max_scores = {
        "required_fields": 20,
        "metadata": 20,
        "component_basic": 20,
        "component_model_card": 30,
        "external_references": 10
    }
    
    # Track missing fields by tier (for penalty calculation)
    missing_fields = {
        "critical": [],
        "important": [],
        "supplementary": []
    }
    
    # Count fields by category
    fields_by_category = {category: {"total": 0, "present": 0} for category in max_scores.keys()}
    
    # Process each field and categorize
    for field, classification in FIELD_CLASSIFICATION.items():
        tier = classification["tier"]
        category = classification["category"]
        
        # Count total fields in this category
        fields_by_category[category]["total"] += 1
        
        # Enhanced field detection using extraction results
        is_present = check_field_with_enhanced_results(aibom, field, extraction_results)
        
        if is_present:
            fields_by_category[category]["present"] += 1
        else:
            missing_fields[tier].append(field)
        
        # Add to field checklist with appropriate indicators
        importance_indicator = "★★★" if tier == "critical" else "★★" if tier == "important" else "★"
        field_checklist[field] = f"{'✔' if is_present else '✘'} {importance_indicator}"
    
    # Calculate category scores using proper normalization
    category_scores = {}
    for category, counts in fields_by_category.items():
        if counts["total"] > 0:
            # Normalization: (Present Fields / Total Fields) × Maximum Points
            raw_score = (counts["present"] / counts["total"]) * max_scores[category]
            # Ensure raw_score is a number before rounding
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                category_scores[category] = round(raw_score, 1)
            else:
                category_scores[category] = 0.0

    # Log field extraction summary
    total_fields = sum(counts["total"] for counts in fields_by_category.values())
    total_present = sum(counts["present"] for counts in fields_by_category.values())
    
    print(f"📊 SCORING SUMMARY:")
    print(f"   Total fields evaluated: {total_fields}")
    print(f"   Fields successfully extracted: {total_present}")
    print(f"   Extraction success rate: {round((total_present/total_fields)*100, 1)}%")
    print(f"   Category breakdown:")
    for category, counts in fields_by_category.items():
        percentage = round((counts["present"]/counts["total"])*100, 1) if counts["total"] > 0 else 0
        print(f"     {category}: {counts['present']}/{counts['total']} ({percentage}%)")
    
    # Calculate subtotal (sum of rounded category scores)
    subtotal_score = sum(category_scores.values())
    
    # Count missing fields by tier for penalty calculation
    missing_critical_count = len(missing_fields["critical"])
    missing_important_count = len(missing_fields["important"])
    
    # Apply penalties based on missing critical and important fields
    penalty_factor = 1.0
    penalty_reasons = []
    
    # Critical field penalties
    if missing_critical_count > 3:
        penalty_factor *= 0.8  # 20% penalty
        penalty_reasons.append("Multiple critical fields missing")
    elif missing_critical_count >= 2:  # if count is 2-3
        penalty_factor *= 0.9  # 10% penalty
        penalty_reasons.append("Some critical fields missing")
    # No penalty for missing_critical_count == 1
    
    # Important field penalties (additional)
    if missing_important_count >= 5:
        penalty_factor *= 0.95  # Additional 5% penalty
        penalty_reasons.append("Several important fields missing")
    
    # Apply penalty to subtotal
    final_score = subtotal_score * penalty_factor
    final_score = round(final_score, 1)

    # Debugging calculation:
    print(f"DEBUG CATEGORIES:")
    for category, score in category_scores.items():
        print(f"  {category}: {score}")
    print(f"DEBUG: category_scores sum = {sum(category_scores.values())}")
    print(f"DEBUG: subtotal_score = {subtotal_score}")
    print(f"DEBUG: missing_critical_count = {missing_critical_count}")
    print(f"DEBUG: missing_important_count = {missing_important_count}")
    print(f"DEBUG: penalty_factor = {penalty_factor}")
    print(f"DEBUG: penalty_reasons = {penalty_reasons}")
    print(f"DEBUG: subtotal_score = {subtotal_score}")
    print(f"DEBUG: final_score calculation = {subtotal_score} × {penalty_factor} = {subtotal_score * penalty_factor}")
    print(f"DEBUG: final_score after round = {final_score}")
    
    # Ensure score is between 0 and 100
    final_score = max(0.0, min(final_score, 100.0))
    
    # Determine completeness profile
    profile = determine_completeness_profile(aibom, final_score)
    
    # Generate recommendations
    recommendations = generate_field_recommendations(missing_fields)
    
    # Prepare penalty information
    penalty_applied = penalty_factor < 1.0
    penalty_reason = " and ".join(penalty_reasons) if penalty_reasons else None
    penalty_percentage = round((1.0 - penalty_factor) * 100, 1) if penalty_applied else 0.0

    # DEBUG: Print the result structure before returning
    print("DEBUG: Final result structure:")
    print(f"  total_score: {final_score}")
    print(f"  section_scores keys: {list(category_scores.keys())}")
    
    result = {
        "total_score": final_score,
        "subtotal_score": subtotal_score,
        "section_scores": category_scores,
        "max_scores": max_scores,
        "field_checklist": field_checklist,
        "category_details": {
            "required_fields": {
                "present_fields": fields_by_category["required_fields"]["present"],
                "total_fields": fields_by_category["required_fields"]["total"],
                "percentage": round((fields_by_category["required_fields"]["present"] / fields_by_category["required_fields"]["total"]) * 100, 1)
            },
            "metadata": {
                "present_fields": fields_by_category["metadata"]["present"],
                "total_fields": fields_by_category["metadata"]["total"],
                "percentage": round((fields_by_category["metadata"]["present"] / fields_by_category["metadata"]["total"]) * 100, 1)
            },
            "component_basic": {
                "present_fields": fields_by_category["component_basic"]["present"],
                "total_fields": fields_by_category["component_basic"]["total"],
                "percentage": round((fields_by_category["component_basic"]["present"] / fields_by_category["component_basic"]["total"]) * 100, 1)
            },
            "component_model_card": {
                "present_fields": fields_by_category["component_model_card"]["present"],
                "total_fields": fields_by_category["component_model_card"]["total"],
                "percentage": round((fields_by_category["component_model_card"]["present"] / fields_by_category["component_model_card"]["total"]) * 100, 1)
            },
            "external_references": {
                "present_fields": fields_by_category["external_references"]["present"],
                "total_fields": fields_by_category["external_references"]["total"],
                "percentage": round((fields_by_category["external_references"]["present"] / fields_by_category["external_references"]["total"]) * 100, 1)
            }
        },
        "field_categorization": get_field_categorization_for_display(aibom),
        "field_tiers": {field: info["tier"] for field, info in FIELD_CLASSIFICATION.items()},
        "missing_fields": missing_fields,
        "missing_counts": {
            "critical": missing_critical_count,
            "important": missing_important_count,
            "supplementary": len(missing_fields["supplementary"])
        },
        "completeness_profile": profile,
        "penalty_applied": penalty_applied,
        "penalty_reason": penalty_reason,
        "penalty_percentage": penalty_percentage,
        "penalty_factor": penalty_factor,
        "recommendations": recommendations,
        "calculation_details": {
            "category_breakdown": {
                category: {
                    "present_fields": counts["present"],
                    "total_fields": counts["total"],
                    "percentage": round((counts["present"] / counts["total"]) * 100, 1) if counts["total"] > 0 else 0.0,
                    "points": category_scores[category],
                    "max_points": max_scores[category]
                }
                for category, counts in fields_by_category.items()
            }
        }
    }
    
    # Debug the final result
    if 'category_details' in result:
        print(f"  category_details exists: {list(result['category_details'].keys())}")
        print(f"  required_fields details: {result['category_details'].get('required_fields')}")
        print(f"  metadata details: {result['category_details'].get('metadata')}")
    else:
        print("  category_details: MISSING!")
    
    return result


def calculate_completeness_score(aibom: Dict[str, Any], validate: bool = True, use_best_practices: bool = True, extraction_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate completeness score for an AIBOM and optionally validate against AI requirements.
    Enhanced with industry best practices scoring.
    
    Args:
        aibom: The AIBOM to score and validate
        validate: Whether to perform validation
        use_best_practices: Whether to use enhanced industry best practices scoring
        
    Returns:
        Dictionary containing score and validation results
    """
    print(f"🔍 DEBUG: use_best_practices={use_best_practices}")
    print(f"🔍 DEBUG: extraction_results is None: {extraction_results is None}")
    print(f"🔍 DEBUG: extraction_results keys: {list(extraction_results.keys()) if extraction_results else 'None'}")
    
    if use_best_practices:
        print("🔍 DEBUG: Calling calculate_industry_neutral_score")
        result = calculate_industry_neutral_score(aibom, extraction_results)
    # If using best practices scoring, use the enhanced industry-neutral approach
    if use_best_practices:
        result = calculate_industry_neutral_score(aibom, extraction_results)
        
        # Add validation if requested
        if validate:
            validation_result = validate_aibom(aibom)
            result["validation"] = validation_result
            
            # Adjust score based on validation results
            if not validation_result["valid"]:
                # Count errors and warnings
                error_count = validation_result["summary"]["error_count"]
                warning_count = validation_result["summary"]["warning_count"]
                
                # Apply penalties to the score
                """
                if error_count > 0:
                    # Severe penalty for errors (up to 50% reduction)
                    error_penalty = min(0.5, error_count * 0.1)
                    result["total_score"] = round(result["total_score"] * (1 - error_penalty), 1)
                    result["validation_penalty"] = f"-{int(error_penalty * 100)}% due to {error_count} schema errors"
                elif warning_count > 0:
                    # Minor penalty for warnings (up to 20% reduction)
                    warning_penalty = min(0.2, warning_count * 0.05)
                    result["total_score"] = round(result["total_score"] * (1 - warning_penalty), 1)
                    result["validation_penalty"] = f"-{int(warning_penalty * 100)}% due to {warning_count} schema warnings"
                """
        result = add_enhanced_field_display_to_result(result, aibom)
        
        return result
    
    # Otherwise, use the original scoring method
    field_checklist = {}
    max_scores = {
        "required_fields": 20,
        "metadata": 20,
        "component_basic": 20,
        "component_model_card": 30,
        "external_references": 10
    }

    # Required Fields (20 points max)
    required_fields = ["bomFormat", "specVersion", "serialNumber", "version"]
    required_score = sum([5 if aibom.get(field) else 0 for field in required_fields])
    for field in required_fields:
        field_checklist[field] = "✔" if aibom.get(field) else "✘"

    # Metadata (20 points max)
    metadata = aibom.get("metadata", {})
    metadata_fields = ["timestamp", "tools", "authors", "component"]
    metadata_score = sum([5 if metadata.get(field) else 0 for field in metadata_fields])
    for field in metadata_fields:
        field_checklist[f"metadata.{field}"] = "✔" if metadata.get(field) else "✘"

    # Component Basic Info (20 points max)
    components = aibom.get("components", [])
    component_score = 0
    
    if components:
        # Use the first component as specified in the design
        comp = components[0]
        comp_fields = ["type", "name", "bom-ref", "purl", "description", "licenses"]
        component_score = sum([
            2 if comp.get("type") else 0,
            4 if comp.get("name") else 0,
            2 if comp.get("bom-ref") else 0,
            4 if comp.get("purl") and re.match(r'^pkg:huggingface/.+', comp["purl"]) else 0,
            4 if comp.get("description") and len(comp["description"]) > 20 else 0,
            4 if comp.get("licenses") and validate_spdx(comp["licenses"]) else 0
        ])
        for field in comp_fields:
            field_checklist[f"component.{field}"] = "✔" if comp.get(field) else "✘"
            if field == "purl" and comp.get(field) and not re.match(r'^pkg:huggingface/.+', comp["purl"]):
                field_checklist[f"component.{field}"] = "✘"
            if field == "description" and comp.get(field) and len(comp["description"]) <= 20:
                field_checklist[f"component.{field}"] = "✘"
            if field == "licenses" and comp.get(field) and not validate_spdx(comp["licenses"]):
                field_checklist[f"component.{field}"] = "✘"

    # Model Card Section (30 points max)
    model_card_score = 0
    
    if components:
        # Use the first component's model card as specified in the design
        comp = components[0]
        card = comp.get("modelCard", {})
        card_fields = ["modelParameters", "quantitativeAnalysis", "considerations"]
        model_card_score = sum([
            10 if card.get("modelParameters") else 0,
            10 if card.get("quantitativeAnalysis") else 0,
            10 if card.get("considerations") and isinstance(card["considerations"], dict) and len(str(card["considerations"])) > 50 else 0
        ])
        for field in card_fields:
            field_checklist[f"modelCard.{field}"] = "✔" if field in card else "✘"
            if field == "considerations" and field in card and (not isinstance(card["considerations"], dict) or len(str(card["considerations"])) <= 50):
                field_checklist[f"modelCard.{field}"] = "✘"

    # External References (10 points max)
    ext_refs = []
    if components and components[0].get("externalReferences"):
        ext_refs = components[0].get("externalReferences")
    ext_score = 0
    for ref in ext_refs:
        url = ref.get("url", "").lower()
        if "modelcard" in url:
            ext_score += 4
        elif "huggingface.co" in url or "github.com" in url:
            ext_score += 3
        elif "dataset" in url:
            ext_score += 3
    ext_score = min(ext_score, 10)
    field_checklist["externalReferences"] = "✔" if ext_refs else "✘"

    # Calculate total score
    section_scores = {
        "required_fields": required_score,
        "metadata": metadata_score,
        "component_basic": component_score,
        "component_model_card": model_card_score,
        "external_references": ext_score
    }
    
    # Calculate weighted total score
    total_score = (
        (section_scores["required_fields"] / max_scores["required_fields"]) * 20 +
        (section_scores["metadata"] / max_scores["metadata"]) * 20 +
        (section_scores["component_basic"] / max_scores["component_basic"]) * 20 +
        (section_scores["component_model_card"] / max_scores["component_model_card"]) * 30 +
        (section_scores["external_references"] / max_scores["external_references"]) * 10
    )
    
    # Round to one decimal place
    total_score = round(total_score, 1)
    
    # Ensure score is between 0 and 100
    total_score = max(0, min(total_score, 100))

    result = {
        "total_score": total_score,
        "section_scores": section_scores,
        "max_scores": max_scores,
        "field_checklist": field_checklist,
        "category_details": {
        "required_fields": {
            "present_fields": fields_by_category["required_fields"]["present"],
            "total_fields": fields_by_category["required_fields"]["total"],
            "percentage": round((fields_by_category["required_fields"]["present"] / fields_by_category["required_fields"]["total"]) * 100, 1)
        },
        "metadata": {
            "present_fields": fields_by_category["metadata"]["present"],
            "total_fields": fields_by_category["metadata"]["total"],
            "percentage": round((fields_by_category["metadata"]["present"] / fields_by_category["metadata"]["total"]) * 100, 1)
        },
        "component_basic": {
            "present_fields": fields_by_category["component_basic"]["present"],
            "total_fields": fields_by_category["component_basic"]["total"],
            "percentage": round((fields_by_category["component_basic"]["present"] / fields_by_category["component_basic"]["total"]) * 100, 1)
        },
        "component_model_card": {
            "present_fields": fields_by_category["component_model_card"]["present"],
            "total_fields": fields_by_category["component_model_card"]["total"],
            "percentage": round((fields_by_category["component_model_card"]["present"] / fields_by_category["component_model_card"]["total"]) * 100, 1)
        },
        "external_references": {
            "present_fields": fields_by_category["external_references"]["present"],
            "total_fields": fields_by_category["external_references"]["total"],
            "percentage": round((fields_by_category["external_references"]["present"] / fields_by_category["external_references"]["total"]) * 100, 1)
        }
     }
    }
    
    # Add validation if requested
    if validate:
        validation_result = validate_aibom(aibom)
        result["validation"] = validation_result
        
        # Adjust score based on validation results
        if not validation_result["valid"]:
            # Count errors and warnings
            error_count = validation_result["summary"]["error_count"]
            warning_count = validation_result["summary"]["warning_count"]

            """
            # Apply penalties to the score
            if error_count > 0:
                # Severe penalty for errors (up to 50% reduction)
                error_penalty = min(0.5, error_count * 0.1)
                result["total_score"] = round(result["total_score"] * (1 - error_penalty), 1)
                result["validation_penalty"] = f"-{int(error_penalty * 100)}% due to {error_count} schema errors"
            elif warning_count > 0:
                # Minor penalty for warnings (up to 20% reduction)
                warning_penalty = min(0.2, warning_count * 0.05)
                result["total_score"] = round(result["total_score"] * (1 - warning_penalty), 1)
                result["validation_penalty"] = f"-{int(warning_penalty * 100)}% due to {warning_count} schema warnings"
            """
    result = add_enhanced_field_display_to_result(result, aibom)
    
    return result


def merge_metadata(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    result = secondary.copy()
    for key, value in primary.items():
        if value is not None:
            if key in result and isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = merge_metadata(value, result[key])
            else:
                result[key] = value
    return result


def extract_model_id_parts(model_id: str) -> Dict[str, str]:
    parts = model_id.split("/")
    if len(parts) == 1:
        return {"owner": None, "name": parts[0]}
    return {"owner": parts[0], "name": "/".join(parts[1:])}


def create_purl(model_id: str) -> str:
    parts = extract_model_id_parts(model_id)
    if parts["owner"]:
        return f"pkg:huggingface/{parts['owner']}/{parts['name']}"
    return f"pkg:huggingface/{parts['name']}"


def get_field_categorization_for_display(aibom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hardcoded field categorization with dynamic status detection.
    """
    
    # Standard CycloneDX Fields
    standard_cyclonedx_definitions = {
        "bomFormat": {"json_path": "bomFormat", "importance": "Critical"},
        "specVersion": {"json_path": "specVersion", "importance": "Critical"},
        "serialNumber": {"json_path": "serialNumber", "importance": "Critical"},
        "version": {"json_path": "version", "importance": "Critical"},
        "metadata.timestamp": {"json_path": "metadata.timestamp", "importance": "Important"},
        "metadata.tools": {"json_path": "metadata.tools", "importance": "Important"},
        "metadata.component": {"json_path": "metadata.component", "importance": "Important"},
        "component.type": {"json_path": "components[].type", "importance": "Important"},
        "component.name": {"json_path": "components[].name", "importance": "Critical"},
        "component.bom-ref": {"json_path": "components[].bom-ref", "importance": "Important"},
        "component.purl": {"json_path": "components[].purl", "importance": "Important"},
        "component.description": {"json_path": "components[].description", "importance": "Important"},
        "component.licenses": {"json_path": "components[].licenses", "importance": "Important"},
        "externalReferences": {"json_path": "components[].externalReferences", "importance": "Supplementary"},
        "downloadLocation": {"json_path": "components[].externalReferences[].url", "importance": "Critical"},
    }
    
    # AI-Specific Extension Fields  
    ai_specific_definitions = {
        # Model card structure fields
        "modelCard.modelParameters": {"json_path": "components[].modelCard.modelParameters", "importance": "Important"},
        "modelCard.quantitativeAnalysis": {"json_path": "components[].modelCard.quantitativeAnalysis", "importance": "Important"},
        "modelCard.considerations": {"json_path": "components[].modelCard.considerations", "importance": "Important"},
        
        # Properties-based fields
        "primaryPurpose": {"json_path": "metadata.properties[].name=\"primaryPurpose\"", "importance": "Critical"},
        "suppliedBy": {"json_path": "metadata.properties[].name=\"suppliedBy\"", "importance": "Critical"},
        "typeOfModel": {"json_path": "components[].modelCard.properties[].name=\"typeOfModel\"", "importance": "Important"},
        "energyConsumption": {"json_path": "components[].modelCard.properties[].name=\"energyConsumption\"", "importance": "Important"},
        "hyperparameter": {"json_path": "components[].modelCard.properties[].name=\"hyperparameter\"", "importance": "Important"},
        "limitation": {"json_path": "components[].modelCard.properties[].name=\"limitation\"", "importance": "Important"},
        "safetyRiskAssessment": {"json_path": "components[].modelCard.properties[].name=\"safetyRiskAssessment\"", "importance": "Important"},
        "modelExplainability": {"json_path": "components[].modelCard.properties[].name=\"modelExplainability\"", "importance": "Supplementary"},
        "standardCompliance": {"json_path": "components[].modelCard.properties[].name=\"standardCompliance\"", "importance": "Supplementary"},
        "domain": {"json_path": "components[].modelCard.properties[].name=\"domain\"", "importance": "Supplementary"},
        "energyQuantity": {"json_path": "components[].modelCard.properties[].name=\"energyQuantity\"", "importance": "Supplementary"},
        "energyUnit": {"json_path": "components[].modelCard.properties[].name=\"energyUnit\"", "importance": "Supplementary"},
        "informationAboutTraining": {"json_path": "components[].modelCard.properties[].name=\"informationAboutTraining\"", "importance": "Supplementary"},
        "informationAboutApplication": {"json_path": "components[].modelCard.properties[].name=\"informationAboutApplication\"", "importance": "Supplementary"},
        "metric": {"json_path": "components[].modelCard.properties[].name=\"metric\"", "importance": "Supplementary"},
        "metricDecisionThreshold": {"json_path": "components[].modelCard.properties[].name=\"metricDecisionThreshold\"", "importance": "Supplementary"},
        "modelDataPreprocessing": {"json_path": "components[].modelCard.properties[].name=\"modelDataPreprocessing\"", "importance": "Supplementary"},
        "autonomyType": {"json_path": "components[].modelCard.properties[].name=\"autonomyType\"", "importance": "Supplementary"},
        "useSensitivePersonalInformation": {"json_path": "components[].modelCard.properties[].name=\"useSensitivePersonalInformation\"", "importance": "Supplementary"},
    }
    
    # DYNAMIC: Check status for each field
    def check_field_presence(field_key):
        """Simple field presence detection"""
        if field_key == "bomFormat":
            return "bomFormat" in aibom
        elif field_key == "specVersion":
            return "specVersion" in aibom
        elif field_key == "serialNumber":
            return "serialNumber" in aibom
        elif field_key == "version":
            return "version" in aibom
        elif field_key == "metadata.timestamp":
            return "metadata" in aibom and "timestamp" in aibom["metadata"]
        elif field_key == "metadata.tools":
            return "metadata" in aibom and "tools" in aibom["metadata"]
        elif field_key == "metadata.component":
            return "metadata" in aibom and "component" in aibom["metadata"]
        elif field_key == "component.type":
            return "components" in aibom and aibom["components"] and "type" in aibom["components"][0]
        elif field_key == "component.name":
            return "components" in aibom and aibom["components"] and "name" in aibom["components"][0]
        elif field_key == "component.bom-ref":
            return "components" in aibom and aibom["components"] and "bom-ref" in aibom["components"][0]
        elif field_key == "component.purl":
            return "components" in aibom and aibom["components"] and "purl" in aibom["components"][0]
        elif field_key == "component.description":
            return "components" in aibom and aibom["components"] and "description" in aibom["components"][0]
        elif field_key == "component.licenses":
            return "components" in aibom and aibom["components"] and "licenses" in aibom["components"][0]
        elif field_key == "externalReferences":
            return ("externalReferences" in aibom or 
                    ("components" in aibom and aibom["components"] and "externalReferences" in aibom["components"][0]))
        elif field_key == "downloadLocation":
            if "externalReferences" in aibom:
                for ref in aibom["externalReferences"]:
                    if ref.get("type") == "distribution":
                        return True
            if "components" in aibom and aibom["components"] and "externalReferences" in aibom["components"][0]:
                return len(aibom["components"][0]["externalReferences"]) > 0
            return False
        elif field_key == "modelCard.modelParameters":
            return ("components" in aibom and aibom["components"] and 
                    "modelCard" in aibom["components"][0] and 
                    "modelParameters" in aibom["components"][0]["modelCard"])
        elif field_key == "modelCard.quantitativeAnalysis":
            return ("components" in aibom and aibom["components"] and 
                    "modelCard" in aibom["components"][0] and 
                    "quantitativeAnalysis" in aibom["components"][0]["modelCard"])
        elif field_key == "modelCard.considerations":
            return ("components" in aibom and aibom["components"] and 
                    "modelCard" in aibom["components"][0] and 
                    "considerations" in aibom["components"][0]["modelCard"])
        elif field_key == "primaryPurpose":
            if "metadata" in aibom and "properties" in aibom["metadata"]:
                for prop in aibom["metadata"]["properties"]:
                    if prop.get("name") == "primaryPurpose":
                        return True
            return False
        elif field_key == "suppliedBy":
            if "metadata" in aibom and "properties" in aibom["metadata"]:
                for prop in aibom["metadata"]["properties"]:
                    if prop.get("name") == "suppliedBy":
                        return True
            return False
        elif field_key == "typeOfModel":
            if ("components" in aibom and aibom["components"] and 
                "modelCard" in aibom["components"][0] and 
                "properties" in aibom["components"][0]["modelCard"]):
                for prop in aibom["components"][0]["modelCard"]["properties"]:
                    if prop.get("name") == "typeOfModel":
                        return True
            return False
        else:
            # For other AI-specific fields, check in modelCard properties
            if ("components" in aibom and aibom["components"] and 
                "modelCard" in aibom["components"][0] and 
                "properties" in aibom["components"][0]["modelCard"]):
                for prop in aibom["components"][0]["modelCard"]["properties"]:
                    if prop.get("name") == field_key:
                        return True
            return False
    
    # Build result with dynamic status
    standard_fields = {}
    for field_key, field_info in standard_cyclonedx_definitions.items():
        standard_fields[field_key] = {
            "status": "✔" if check_field_presence(field_key) else "✘",
            "field_name": field_key,
            "json_path": field_info["json_path"],
            "importance": field_info["importance"]
        }
    
    ai_fields = {}
    for field_key, field_info in ai_specific_definitions.items():
        ai_fields[field_key] = {
            "status": "✔" if check_field_presence(field_key) else "✘",
            "field_name": field_key,
            "json_path": field_info["json_path"],
            "importance": field_info["importance"]
        }
    
    return {
        "standard_cyclonedx_fields": standard_fields,
        "ai_specific_extension_fields": ai_fields
    }


def add_enhanced_field_display_to_result(result: Dict[str, Any], aibom: Dict[str, Any]) -> Dict[str, Any]:
    """Add field categorization to result"""
    enhanced_result = result.copy()
    enhanced_result["field_display"] = get_field_categorization_for_display(aibom)
    return enhanced_result


def get_score_display_info(score_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate user-friendly display information for the score.
    
    Args:
        score_result: Result from calculate_industry_neutral_score
        
    Returns:
        Dictionary with display-friendly information
    """
    display_info = {
        "category_display": [],
        "penalty_display": None,
        "total_display": None
    }
    
    # Format category scores for display
    for category, score in score_result["section_scores"].items():
        max_score = score_result["max_scores"][category]
        category_name = category.replace("_", " ").title()
        
        display_info["category_display"].append({
            "name": category_name,
            "score": f"{score}/{max_score}",
            "percentage": round((score / max_score) * 100, 1) if max_score > 0 else 0.0
        })
    
    # Format penalty display
    if score_result["penalty_applied"]:
        display_info["penalty_display"] = {
            "message": f"Penalty Applied: -{score_result['penalty_percentage']}% ({score_result['penalty_reason']})",
            "subtotal": f"{score_result['subtotal_score']}/100",
            "final": f"{score_result['total_score']}/100"
        }
    
    # Format total display
    display_info["total_display"] = {
        "score": f"{score_result['total_score']}/100",
        "percentage": round(score_result['total_score'], 1)
    }
    
    return display_info


def format_score_summary(score_result: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the scoring results.
    
    Args:
        score_result: Result from calculate_industry_neutral_score
        
    Returns:
        Formatted summary string
    """
    summary = "AI SBOM Completeness Score Summary\n"
    summary += "=" * 40 + "\n\n"
    
    # Category breakdown
    summary += "Category Breakdown:\n"
    for category, score in score_result["section_scores"].items():
        max_score = score_result["max_scores"][category]
        category_name = category.replace("_", " ").title()
        percentage = round((score / max_score) * 100, 1) if max_score > 0 else 0.0
        summary += f"- {category_name}: {score}/{max_score} ({percentage}%)\n"
    
    summary += f"\nSubtotal: {score_result['subtotal_score']}/100\n"
    
    # Penalty information
    if score_result["penalty_applied"]:
        summary += f"\nPenalty Applied: -{score_result['penalty_percentage']}%\n"
        summary += f"Reason: {score_result['penalty_reason']}\n"
        summary += f"Final Score: {score_result['total_score']}/100\n"
    else:
        summary += f"Final Score: {score_result['total_score']}/100 (No penalties applied)\n"
    
    # Missing field counts
    summary += f"\nMissing Fields Summary:\n"
    summary += f"- Critical: {score_result['missing_counts']['critical']}\n"
    summary += f"- Important: {score_result['missing_counts']['important']}\n"
    summary += f"- Supplementary: {score_result['missing_counts']['supplementary']}\n"
    
    # Completeness profile
    profile = score_result["completeness_profile"]
    summary += f"\nCompleteness Profile: {profile['name']}\n"
    summary += f"Description: {profile['description']}\n"
    
    return summary

def test_consolidated_integration():
    """Test that consolidated field registry manager integration is working"""
    try:
        print("\n🧪 Testing Consolidated Integration...")
        
        # Test registry availability
        if REGISTRY_AVAILABLE:
            print("✅ Consolidated registry manager available")
            
            # Test registry manager
            manager = get_field_registry_manager()
            print(f"✅ Registry manager initialized: {manager.registry_path}")
            
            # Test field classification generation
            field_count = len(FIELD_CLASSIFICATION)
            print(f"✅ FIELD_CLASSIFICATION loaded: {field_count} fields")
            
            # Test completeness profiles
            profile_count = len(COMPLETENESS_PROFILES)
            print(f"✅ COMPLETENESS_PROFILES loaded: {profile_count} profiles")
            
            # Test validation messages
            message_count = len(VALIDATION_MESSAGES)
            print(f"✅ VALIDATION_MESSAGES loaded: {message_count} messages")
            
            # Test scoring weights
            tier_weights = SCORING_WEIGHTS.get("tier_weights", {})
            category_weights = SCORING_WEIGHTS.get("category_weights", {})
            print(f"✅ SCORING_WEIGHTS loaded: {len(tier_weights)} tiers, {len(category_weights)} categories")
            
        else:
            print("⚠️  Consolidated registry manager not available, using hardcoded definitions")
        
        # Test dynamic field detector (DynamicFieldDetector)
        if hasattr(check_field_with_enhanced_results, '_detector') and check_field_with_enhanced_results._detector:
            print(f"✅ Dynamic field detector ready")
        else:
            print(f"⚠️  Dynamic field detector not initialized")
        
        # Test field lookup
        test_fields = ["bomFormat", "primaryPurpose", "energyConsumption"]
        for field in test_fields:
            if field in FIELD_CLASSIFICATION:
                field_info = FIELD_CLASSIFICATION[field]
                print(f"✅ Field '{field}': tier={field_info['tier']}, category={field_info['category']}")
            else:
                print(f"❌ Field '{field}' not found in FIELD_CLASSIFICATION")
        
        print("🎉 Consolidated integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Consolidated integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Uncomment this line to run the test automatically when utils.py is imported
test_consolidated_integration()