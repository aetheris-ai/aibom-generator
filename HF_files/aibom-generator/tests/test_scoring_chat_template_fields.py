from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "aibom-generator"))

from field_registry_manager import (
    get_field_registry_manager,
    generate_field_classification,
)


class TestChatTemplateFieldClassification:

    def test_chat_template_excluded_from_scoring(self):
        classification = generate_field_classification()

        assert "chat_template" not in classification

    def test_chat_template_hash_is_important(self):
        classification = generate_field_classification()

        assert "chat_template_hash" in classification
        assert classification["chat_template_hash"]["tier"] == "important"
        assert classification["chat_template_hash"]["category"] == "component_model_card"

    def test_template_security_status_is_important(self):
        classification = generate_field_classification()

        assert "template_security_status" in classification
        assert classification["template_security_status"]["tier"] == "important"

    def test_template_source_is_supplementary(self):
        classification = generate_field_classification()

        assert "template_source" in classification
        assert classification["template_source"]["tier"] == "supplementary"

    def test_model_lineage_is_supplementary(self):
        classification = generate_field_classification()

        assert "model_lineage" in classification
        assert classification["model_lineage"]["tier"] == "supplementary"

    def test_named_chat_templates_is_supplementary(self):
        classification = generate_field_classification()

        assert "named_chat_templates" in classification
        assert classification["named_chat_templates"]["tier"] == "supplementary"

    def test_total_scored_fields_is_34(self):
        classification = generate_field_classification()

        assert len(classification) == 34


class TestExcludeFromScoringFlag:

    def test_registry_has_exclude_flag_on_chat_template(self):
        manager = get_field_registry_manager()
        fields = manager.get_field_definitions()

        assert "chat_template" in fields
        assert fields["chat_template"].get("exclude_from_scoring") is True

    def test_excluded_fields_not_in_classification(self):
        manager = get_field_registry_manager()
        fields = manager.get_field_definitions()
        classification = generate_field_classification()

        excluded_fields = [
            name for name, config in fields.items()
            if config.get("exclude_from_scoring", False)
        ]

        for field in excluded_fields:
            assert field not in classification, f"{field} should be excluded from scoring"
