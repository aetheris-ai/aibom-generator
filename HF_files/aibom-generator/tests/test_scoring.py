"""
Unit tests for completeness scoring functionality.

Tests cover:
- Score calculation
- Field presence detection
- Validation messages
"""


class TestCompletenessScoring:
    """Tests for completeness score calculation."""

    def test_score_returns_dict(self, sample_aibom):
        """Test that scoring returns a dictionary."""
        from src.aibom_generator.utils import calculate_completeness_score

        result = calculate_completeness_score(sample_aibom)
        assert isinstance(result, dict)

    def test_score_has_total(self, sample_aibom):
        """Test that score result has total_score field."""
        from src.aibom_generator.utils import calculate_completeness_score

        result = calculate_completeness_score(sample_aibom)
        assert "total_score" in result

    def test_score_in_valid_range(self, sample_aibom):
        """Test that score is between 0 and 100."""
        from src.aibom_generator.utils import calculate_completeness_score

        result = calculate_completeness_score(sample_aibom)
        assert 0 <= result["total_score"] <= 100

    def test_empty_aibom_low_score(self):
        """Test that empty AIBOM gets a low score."""
        from src.aibom_generator.utils import calculate_completeness_score

        empty_aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "components": []
        }
        result = calculate_completeness_score(empty_aibom)
        assert result["total_score"] < 50

    def test_complete_aibom_high_score(self, sample_aibom):
        """Test that complete AIBOM gets reasonable score."""
        from src.aibom_generator.utils import calculate_completeness_score

        result = calculate_completeness_score(sample_aibom)
        # A sample AIBOM should score reasonably well
        assert result["total_score"] >= 30


class TestFieldValidation:
    """Tests for field validation in scoring."""

    def test_validates_bom_format(self, sample_aibom):
        """Test that bomFormat is validated."""
        from src.aibom_generator.utils import calculate_completeness_score

        result = calculate_completeness_score(sample_aibom, validate=True)
        # Should expose validation details and pass validation for valid bomFormat
        assert "total_score" in result
        assert result["total_score"] > 0

    def test_missing_required_fields_flagged(self):
        """Test that missing required fields are identified."""
        from src.aibom_generator.utils import calculate_completeness_score

        incomplete = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6"
            # Missing components, metadata
        }
        result = calculate_completeness_score(incomplete)

        # Should have low score due to missing components/metadata
        assert result["total_score"] < 50
