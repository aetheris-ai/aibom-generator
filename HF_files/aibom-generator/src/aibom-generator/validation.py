"""
CycloneDX 1.6 Schema Validation for AIBOM Generator.

This module provides validation of generated AIBOMs against the official
CycloneDX 1.6 JSON schema to ensure compliance and interoperability.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from jsonschema import Draft7Validator, ValidationError, validate
from jsonschema.exceptions import SchemaError

# Module-level logger
logger = logging.getLogger(__name__)

# CycloneDX schema configuration
CYCLONEDX_1_6_SCHEMA_URL = "https://raw.githubusercontent.com/CycloneDX/specification/master/schema/bom-1.6.schema.json"
SCHEMA_CACHE_DIR = Path(__file__).parent / "schemas"
SCHEMA_CACHE_FILE = SCHEMA_CACHE_DIR / "bom-1.6.schema.json"

# Global schema cache
_cached_schema: Optional[Dict[str, Any]] = None


def _ensure_cache_dir() -> None:
    """Ensure the schema cache directory exists."""
    SCHEMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_schema_from_cache() -> Optional[Dict[str, Any]]:
    """Load schema from local cache if available."""
    if SCHEMA_CACHE_FILE.exists():
        try:
            with open(SCHEMA_CACHE_FILE, "r", encoding="utf-8") as f:
                schema = json.load(f)
                logger.debug("Loaded CycloneDX 1.6 schema from cache")
                return schema
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load cached schema: %s", e)
    return None


def _download_schema() -> Optional[Dict[str, Any]]:
    """Download the CycloneDX 1.6 schema from the official repository."""
    try:
        logger.info("Downloading CycloneDX 1.6 schema from %s", CYCLONEDX_1_6_SCHEMA_URL)
        response = requests.get(CYCLONEDX_1_6_SCHEMA_URL, timeout=30)
        response.raise_for_status()
        schema = response.json()

        # Cache the schema locally
        _ensure_cache_dir()
        with open(SCHEMA_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        logger.info("CycloneDX 1.6 schema downloaded and cached")

        return schema
    except requests.RequestException as e:
        logger.error("Failed to download CycloneDX schema: %s", e)
        return None
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to parse or cache schema: %s", e)
        return None


def load_schema(force_download: bool = False) -> Optional[Dict[str, Any]]:
    """
    Load the CycloneDX 1.6 JSON schema.

    Uses in-memory cache first, then file cache, then downloads if needed.

    Args:
        force_download: If True, download fresh schema even if cached.

    Returns:
        The schema dictionary, or None if loading failed.
    """
    global _cached_schema

    # Return in-memory cache if available
    if _cached_schema is not None and not force_download:
        return _cached_schema

    # Try loading from file cache
    if not force_download:
        schema = _load_schema_from_cache()
        if schema:
            _cached_schema = schema
            return schema

    # Download fresh schema
    schema = _download_schema()
    if schema:
        _cached_schema = schema

    return schema


def _format_validation_error(error: ValidationError) -> str:
    """Format a validation error into a readable message."""
    path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
    return f"[{path}] {error.message}"


def validate_aibom(aibom: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate an AIBOM against the CycloneDX 1.6 schema.

    Args:
        aibom: The AIBOM dictionary to validate.
        strict: If True, fail on any schema deviation. If False, collect all errors.

    Returns:
        Tuple of (is_valid, list of error messages).
        If valid, returns (True, []).
        If invalid, returns (False, [error1, error2, ...]).
    """
    schema = load_schema()

    if schema is None:
        logger.warning("Could not load CycloneDX schema - skipping validation")
        return True, ["Schema unavailable - validation skipped"]

    try:
        if strict:
            # Strict mode: raise on first error
            validate(instance=aibom, schema=schema)
            return True, []
        else:
            # Collect all validation errors
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(aibom))

            if not errors:
                return True, []

            # Format and return all errors
            error_messages = [_format_validation_error(e) for e in errors]
            logger.debug("AIBOM validation found %d issues", len(error_messages))
            return False, error_messages

    except ValidationError as e:
        return False, [_format_validation_error(e)]
    except SchemaError as e:
        logger.error("Invalid CycloneDX schema: %s", e)
        return True, [f"Schema error - validation skipped: {e.message}"]


def get_validation_summary(aibom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a validation summary suitable for inclusion in completeness reports.

    Args:
        aibom: The AIBOM dictionary to validate.

    Returns:
        Dictionary containing validation results:
        {
            "valid": bool,
            "schema_version": "1.6",
            "error_count": int,
            "errors": [str, ...],  # Only first 10 errors
            "warnings": [str, ...]  # Any non-critical issues
        }
    """
    is_valid, errors = validate_aibom(aibom, strict=False)

    # Categorize errors by severity
    critical_errors = []
    warnings = []

    for error in errors:
        # Schema unavailable is a warning, not an error
        if "Schema unavailable" in error or "Schema error" in error:
            warnings.append(error)
        else:
            critical_errors.append(error)

    return {
        "valid": is_valid and len(critical_errors) == 0,
        "schema_version": "1.6",
        "error_count": len(critical_errors),
        "errors": critical_errors[:10],  # Limit to first 10 for readability
        "warnings": warnings,
    }


def validate_minimal_requirements(aibom: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that an AIBOM meets minimal CycloneDX requirements.

    This is a lightweight check that doesn't require the full schema.
    Useful as a quick sanity check before full validation.

    Args:
        aibom: The AIBOM dictionary to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Required top-level fields for CycloneDX 1.6
    required_fields = ["bomFormat", "specVersion"]

    for field in required_fields:
        if field not in aibom:
            errors.append(f"Missing required field: {field}")

    # Validate bomFormat
    if aibom.get("bomFormat") != "CycloneDX":
        errors.append(f"Invalid bomFormat: expected 'CycloneDX', got '{aibom.get('bomFormat')}'")

    # Validate specVersion
    spec_version = aibom.get("specVersion")
    if spec_version and not spec_version.startswith("1."):
        errors.append(f"Unsupported specVersion: {spec_version}")

    # Validate serialNumber format if present
    serial = aibom.get("serialNumber")
    if serial and not serial.startswith("urn:uuid:"):
        errors.append(f"Invalid serialNumber format: '{serial}' should start with 'urn:uuid:'")

    # Validate components structure if present
    components = aibom.get("components", [])
    if components and not isinstance(components, list):
        errors.append("'components' must be an array")

    return len(errors) == 0, errors
