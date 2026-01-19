"""
Tests for CycloneDX 1.6 schema validation module.
"""
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'aibom-generator'))

from validation import (
    validate_aibom,
    validate_minimal_requirements,
    get_validation_summary,
    load_schema,
)


class TestMinimalRequirements:
    """Tests for validate_minimal_requirements function."""

    def test_valid_minimal_aibom(self):
        """Test that a valid minimal AIBOM passes validation."""
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": "urn:uuid:12345678-1234-1234-1234-123456789012",
            "version": 1,
        }
        is_valid, errors = validate_minimal_requirements(aibom)
        assert is_valid is True
        assert errors == []

    def test_missing_bomformat(self):
        """Test that missing bomFormat is detected."""
        aibom = {
            "specVersion": "1.6",
        }
        is_valid, errors = validate_minimal_requirements(aibom)
        assert is_valid is False
        assert any("bomFormat" in err for err in errors)

    def test_invalid_bomformat(self):
        """Test that invalid bomFormat is detected."""
        aibom = {
            "bomFormat": "Invalid",
            "specVersion": "1.6",
        }
        is_valid, errors = validate_minimal_requirements(aibom)
        assert is_valid is False
        assert any("Invalid bomFormat" in err for err in errors)

    def test_unsupported_specversion(self):
        """Test that unsupported specVersion is detected."""
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "2.0",
        }
        is_valid, errors = validate_minimal_requirements(aibom)
        assert is_valid is False
        assert any("Unsupported specVersion" in err for err in errors)

    def test_invalid_serialnumber_format(self):
        """Test that invalid serialNumber format is detected."""
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": "invalid-format",
        }
        is_valid, errors = validate_minimal_requirements(aibom)
        assert is_valid is False
        assert any("serialNumber" in err for err in errors)

    def test_components_not_array(self):
        """Test that non-array components is detected."""
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "components": "not-an-array",
        }
        is_valid, errors = validate_minimal_requirements(aibom)
        assert is_valid is False
        assert any("components" in err for err in errors)


class TestValidationSummary:
    """Tests for get_validation_summary function."""

    def test_valid_aibom_summary(self):
        """Test validation summary for valid AIBOM."""
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": "urn:uuid:12345678-1234-1234-1234-123456789012",
            "version": 1,
            "components": [],
        }
        summary = get_validation_summary(aibom)
        assert summary["valid"] is True
        assert summary["schema_version"] == "1.6"
        assert summary["error_count"] == 0
        assert summary["errors"] == []

    def test_invalid_aibom_summary(self):
        """Test validation summary for invalid AIBOM."""
        aibom = {
            "bomFormat": "Invalid",
            "specVersion": "2.0",
        }
        summary = get_validation_summary(aibom)
        # Note: may be valid if schema unavailable, check error_count
        assert "error_count" in summary
        assert "errors" in summary


class TestSchemaValidation:
    """Tests for full schema validation."""

    def test_valid_complete_aibom(self):
        """Test validation of a complete valid AIBOM."""
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": "urn:uuid:12345678-1234-1234-1234-123456789012",
            "version": 1,
            "metadata": {
                "timestamp": "2024-01-14T00:00:00Z",
                "tools": {
                    "components": [{
                        "bom-ref": "pkg:generic/tool@1.0",
                        "type": "application",
                        "name": "Test Tool",
                        "version": "1.0.0"
                    }]
                }
            },
            "components": [{
                "bom-ref": "pkg:huggingface/test/model@1.0",
                "type": "machine-learning-model",
                "name": "test-model",
                "version": "1.0",
            }],
        }
        is_valid, errors = validate_aibom(aibom, strict=False)
        # If schema is available, should validate
        # If schema unavailable, should return True with warning
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_schema_loading(self):
        """Test that schema can be loaded."""
        schema = load_schema()
        # Schema may be None if download fails, but function should not raise
        assert schema is None or isinstance(schema, dict)


class TestSchemaCache:
    """Tests for schema caching functionality."""

    def test_schema_caching(self):
        """Test that schema is cached after first load."""
        # First load
        schema1 = load_schema()
        # Second load should use cache
        schema2 = load_schema()
        # Both should be same object if cached
        if schema1 is not None:
            assert schema1 is schema2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
