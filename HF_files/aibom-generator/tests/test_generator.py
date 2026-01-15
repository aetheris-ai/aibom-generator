"""
Unit tests for the AIBOMGenerator class.

Tests cover:
- AIBOM generation with mocked HuggingFace API
- PURL encoding correctness
- CycloneDX structure compliance
- Error handling
"""
import pytest
import json
from unittest.mock import patch, MagicMock


class TestAIBOMGenerator:
    """Tests for AIBOMGenerator class."""

    def test_generator_initialization(self, mock_hf_api):
        """Test that generator initializes correctly."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        assert generator is not None
        assert generator.hf_api is not None

    def test_generate_aibom_basic_structure(self, mock_hf_api):
        """Test that generated AIBOM has required CycloneDX fields."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        # Check required top-level fields
        assert result["bomFormat"] == "CycloneDX"
        assert result["specVersion"] == "1.6"
        assert "serialNumber" in result
        assert result["version"] == 1
        assert "metadata" in result
        assert "components" in result

    def test_generate_aibom_has_component(self, mock_hf_api):
        """Test that generated AIBOM has at least one component."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        assert len(result["components"]) >= 1
        component = result["components"][0]
        assert "name" in component
        assert "type" in component

    def test_generate_aibom_component_type(self, mock_hf_api):
        """Test that component type is machine-learning-model."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        component = result["components"][0]
        assert component["type"] == "machine-learning-model"

    def test_generate_aibom_metadata_timestamp(self, mock_hf_api):
        """Test that metadata contains a valid timestamp."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        assert "timestamp" in result["metadata"]
        # Timestamp should be ISO format
        assert "T" in result["metadata"]["timestamp"]

    def test_generate_aibom_serial_number_format(self, mock_hf_api):
        """Test that serial number is a valid URN UUID."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        assert result["serialNumber"].startswith("urn:uuid:")

    def test_generate_aibom_with_output_file(self, mock_hf_api, tmp_path):
        """Test that AIBOM can be written to a file."""
        from src.aibom_generator.generator import AIBOMGenerator

        output_file = tmp_path / "test_aibom.json"
        generator = AIBOMGenerator()
        result = generator.generate_aibom(
            "test-org/test-model",
            output_file=str(output_file)
        )

        assert output_file.exists()
        with open(output_file) as f:
            saved = json.load(f)
        assert saved["bomFormat"] == "CycloneDX"


class TestPURLEncoding:
    """Tests for PURL (Package URL) encoding."""

    @pytest.mark.xfail(reason="PURL encoding fix pending - see Issue #13 / PR #18")
    def test_purl_contains_encoded_slash(self, mock_hf_api):
        """Test that PURL encodes slash in model ID as %2F.

        Note: This test is expected to fail until PR #18 is merged.
        It serves as a regression test to ensure the fix stays in place.
        """
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        component = result["components"][0]
        purl = component.get("purl", "")

        # The slash between org and model should be encoded as %2F
        assert "%2F" in purl, f"PURL should encode '/' as '%2F', got: {purl}"

    def test_purl_format_correct(self, mock_hf_api):
        """Test that PURL follows pkg:type/namespace/name@version format."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        component = result["components"][0]
        purl = component.get("purl", "")

        assert purl.startswith("pkg:")


class TestModelCardExtraction:
    """Tests for model card metadata extraction."""

    def test_extracts_license(self, mock_hf_api):
        """Test that license is extracted from model info."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        component = result["components"][0]
        # License should be present
        assert "licenses" in component or any(
            p.get("name") == "licenses"
            for p in component.get("modelCard", {}).get("properties", [])
        )

    def test_extracts_pipeline_tag(self, mock_hf_api):
        """Test that pipeline tag is extracted as task."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()
        result = generator.generate_aibom("test-org/test-model")

        component = result["components"][0]
        model_card = component.get("modelCard", {})
        model_params = model_card.get("modelParameters", {})

        # Task should be present
        assert "task" in model_params or any(
            p.get("name") == "primaryPurpose"
            for p in model_card.get("properties", [])
        )


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_handles_missing_model_gracefully(self):
        """Test that generator handles non-existent model.

        The generator should either raise an exception or return
        a minimal/error AIBOM structure.
        """
        from src.aibom_generator.generator import AIBOMGenerator
        from huggingface_hub.utils import RepositoryNotFoundError

        with patch('huggingface_hub.HfApi') as mock_api_class:
            mock_api = MagicMock()
            mock_api.model_info.side_effect = RepositoryNotFoundError(
                "Repository not found"
            )
            mock_api_class.return_value = mock_api

            generator = AIBOMGenerator()

            # Generator may raise or return minimal AIBOM
            try:
                result = generator.generate_aibom("nonexistent/model")
                # If it doesn't raise, it should still return valid structure
                assert "bomFormat" in result
            except Exception:
                # Exception is also acceptable behavior
                pass

    def test_handles_empty_model_id(self, mock_hf_api):
        """Test that generator handles empty model ID.

        The generator should either raise an exception or handle
        the empty input gracefully.
        """
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()

        # Generator may raise or handle gracefully
        try:
            result = generator.generate_aibom("")
            # If it doesn't raise, check it still returns structure
            assert isinstance(result, dict)
        except Exception:
            # Exception is also acceptable for empty input
            pass


class TestModelIDNormalization:
    """Tests for model ID normalization."""

    def test_normalizes_full_url(self, mock_hf_api):
        """Test that full HuggingFace URL is normalized to model ID."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()

        # The _normalise_model_id method should handle URLs
        normalized = generator._normalise_model_id(
            "https://huggingface.co/test-org/test-model"
        )
        assert normalized == "test-org/test-model"

    def test_normalizes_simple_id(self, mock_hf_api):
        """Test that simple model ID passes through."""
        from src.aibom_generator.generator import AIBOMGenerator

        generator = AIBOMGenerator()

        normalized = generator._normalise_model_id("test-org/test-model")
        assert normalized == "test-org/test-model"
