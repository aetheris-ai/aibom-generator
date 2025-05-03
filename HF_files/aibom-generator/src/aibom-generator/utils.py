"""
Utility functions for the AIBOM Generator.
"""

import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Validation severity levels
class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

# Field classification based on documentation value (silently aligned with SPDX)
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
    # Check in root level
    if field in aibom:
        return True
        
    # Check in metadata
    if "metadata" in aibom:
        metadata = aibom["metadata"]
        if field in metadata:
            return True
            
        # Check in metadata properties
        if "properties" in metadata:
            for prop in metadata["properties"]:
                if prop.get("name") == f"spdx:{field}" or prop.get("name") == field:
                    return True
    
    # Check in components
    if "components" in aibom and aibom["components"]:
        component = aibom["components"][0]  # Use first component
        
        if field in component:
            return True
            
        # Check in component properties
        if "properties" in component:
            for prop in component["properties"]:
                if prop.get("name") == f"spdx:{field}" or prop.get("name") == field:
                    return True
                
        # Check in model card
        if "modelCard" in component:
            model_card = component["modelCard"]
            
            if field in model_card:
                return True
                
            # Check in model parameters
            if "modelParameters" in model_card:
                if field in model_card["modelParameters"]:
                    return True
                    
                # Check in model parameters properties
                if "properties" in model_card["modelParameters"]:
                    for prop in model_card["modelParameters"]["properties"]:
                        if prop.get("name") == f"spdx:{field}" or prop.get("name") == field:
                            return True
            
            # Check in considerations
            if "considerations" in model_card:
                if field in model_card["considerations"]:
                    return True
                
                # Check in specific consideration sections
                for section in ["technicalLimitations", "ethicalConsiderations", "environmentalConsiderations"]:
                    if section in model_card["considerations"]:
                        if field == "limitation" and section == "technicalLimitations":
                            return True
                        if field == "safetyRiskAssessment" and section == "ethicalConsiderations":
                            return True
                        if field == "energyConsumption" and section == "environmentalConsiderations":
                            return True
    
    # Check in external references
    if field == "downloadLocation" and "externalReferences" in aibom:
        for ref in aibom["externalReferences"]:
            if ref.get("type") == "distribution":
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
            "name": "advanced",
            "description": COMPLETENESS_PROFILES["advanced"]["description"],
            "satisfied": True
        }
    elif "standard" in satisfied_profiles:
        return {
            "name": "standard",
            "description": COMPLETENESS_PROFILES["standard"]["description"],
            "satisfied": True
        }
    elif "basic" in satisfied_profiles:
        return {
            "name": "basic",
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
    
    # Calculate penalty based on missing critical fields
    if missing_critical_count > 3:
        penalty_factor = 0.8  # 20% penalty
        penalty_reason = "Multiple critical fields missing"
    elif missing_critical_count > 0:
        penalty_factor = 0.9  # 10% penalty
        penalty_reason = "Some critical fields missing"
    elif missing_important_count > 5:
        penalty_factor = 0.95  # 5% penalty
        penalty_reason = "Several important fields missing"
    else:
        # No penalty
        penalty_factor = 1.0
        penalty_reason = None
    
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
        recommendations.append("Add metadata section to the AIBOM")
        
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


def calculate_industry_neutral_score(aibom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate completeness score using industry best practices without explicit standard references.
    
    Args:
        aibom: The AIBOM to score
        
    Returns:
        Dictionary containing score and recommendations
    """
    field_checklist = {}
    max_scores = {
        "required_fields": 20,
        "metadata": 20,
        "component_basic": 20,
        "component_model_card": 30,
        "external_references": 10
    }
    
    # Track missing fields by tier
    missing_fields = {
        "critical": [],
        "important": [],
        "supplementary": []
    }
    
    # Score each field based on classification
    scores_by_category = {category: 0 for category in max_scores.keys()}
    max_possible_by_category = {category: 0 for category in max_scores.keys()}
    
    for field, classification in FIELD_CLASSIFICATION.items():
        tier = classification["tier"]
        weight = classification["weight"]
        category = classification["category"]
        
        # Add to max possible score for this category
        max_possible_by_category[category] += weight
        
        # Check if field is present
        is_present = check_field_in_aibom(aibom, field)
        
        if is_present:
            scores_by_category[category] += weight
        else:
            missing_fields[tier].append(field)
        
        # Add to field checklist with appropriate indicators
        importance_indicator = "★★★" if tier == "critical" else "★★" if tier == "important" else "★"
        field_checklist[field] = f"{'✔' if is_present else '✘'} {importance_indicator}"
    
    # Normalize category scores to max_scores
    normalized_scores = {}
    for category in scores_by_category:
        if max_possible_by_category[category] > 0:
            # Normalize to the max score for this category
            normalized_score = (scores_by_category[category] / max_possible_by_category[category]) * max_scores[category]
            normalized_scores[category] = min(normalized_score, max_scores[category])
        else:
            normalized_scores[category] = 0
    
    # Calculate total score (sum of weighted normalized scores)
    total_score = 0
    for category, score in normalized_scores.items():
        # Each category contributes its percentage to the total
        category_weight = max_scores[category] / sum(max_scores.values())
        total_score += score * category_weight
    
    # Round to one decimal place
    total_score = round(total_score, 1)
    
    # Ensure score is between 0 and 100
    total_score = max(0, min(total_score, 100))
    
    # Determine completeness profile
    profile = determine_completeness_profile(aibom, total_score)
    
    # Apply penalties for missing critical fields
    penalty_result = apply_completeness_penalties(total_score, missing_fields)
    
    # Generate recommendations
    recommendations = generate_field_recommendations(missing_fields)
    
    return {
        "total_score": penalty_result["adjusted_score"],
        "section_scores": normalized_scores,
        "max_scores": max_scores,
        "field_checklist": field_checklist,
        "field_tiers": {field: info["tier"] for field, info in FIELD_CLASSIFICATION.items()},
        "missing_fields": missing_fields,
        "completeness_profile": profile,
        "penalty_applied": penalty_result["penalty_applied"],
        "penalty_reason": penalty_result["penalty_reason"],
        "recommendations": recommendations
    }


def calculate_completeness_score(aibom: Dict[str, Any], validate: bool = True, use_best_practices: bool = True) -> Dict[str, Any]:
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
    # If using best practices scoring, use the enhanced industry-neutral approach
    if use_best_practices:
        result = calculate_industry_neutral_score(aibom)
        
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
        "field_checklist": field_checklist
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

