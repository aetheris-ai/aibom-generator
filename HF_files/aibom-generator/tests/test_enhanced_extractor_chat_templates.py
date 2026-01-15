from __future__ import annotations

from pathlib import Path
from unittest import mock
from datetime import datetime
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "aibom-generator"))

from enhanced_extractor import EnhancedExtractor, ConfidenceLevel, DataSource


class TestChatTemplateExtraction:

    @pytest.fixture
    def mock_hf_api(self):
        api = mock.MagicMock()
        api.token = None
        return api

    @pytest.fixture
    def extractor(self, mock_hf_api):
        return EnhancedExtractor(hf_api=mock_hf_api)

    def test_extracts_chat_template_hash_from_tokenizer_config(self, extractor):
        template = "{% for m in messages %}{{ m.content }}{% endfor %}"

        with mock.patch.object(extractor, '_download_and_parse_config') as mock_config:
            mock_config.side_effect = lambda model_id, filename: (
                {"chat_template": template} if filename == "tokenizer_config.json" else {}
            )
            with mock.patch.object(extractor, '_get_readme_content', return_value=""):
                result = extractor.extract_metadata("owner/model", {}, None)

        assert "chat_template_hash" in result
        assert result["chat_template_hash"].startswith("sha256:")
        assert len(result["chat_template_hash"]) == 7 + 64

    def test_extracts_template_source_provenance(self, extractor):
        template = "{{ msg }}"

        with mock.patch.object(extractor, '_download_and_parse_config') as mock_config:
            mock_config.side_effect = lambda model_id, filename: (
                {"chat_template": template} if filename == "tokenizer_config.json" else {}
            )
            with mock.patch.object(extractor, '_get_readme_content', return_value=""):
                result = extractor.extract_metadata("meta-llama/Llama-2-7b", {}, None)

        assert "template_source" in result
        source = result["template_source"]
        assert source["source_file"] == "tokenizer_config.json"
        assert "meta-llama/Llama-2-7b" in source["source_repository"]
        assert "extraction_timestamp" in source

    def test_template_security_status_defaults_to_unscanned(self, extractor):
        template = "{{ msg }}"

        with mock.patch.object(extractor, '_download_and_parse_config') as mock_config:
            mock_config.side_effect = lambda model_id, filename: (
                {"chat_template": template} if filename == "tokenizer_config.json" else {}
            )
            with mock.patch.object(extractor, '_get_readme_content', return_value=""):
                result = extractor.extract_metadata("owner/model", {}, None)

        assert "template_security_status" in result
        status = result["template_security_status"]
        assert status["status"] == "unscanned"
        assert status["scanner_name"] is None
        assert status["findings"] == []

    def test_template_attestation_is_merged(self, extractor):
        template = "{{ msg }}"
        attestation = {
            "status": "clean",
            "scanner_name": "TemplateScanner",
            "scanner_version": "1.0.0",
            "scan_timestamp": "2025-01-15T10:00:00Z",
            "findings": []
        }

        with mock.patch.object(extractor, '_download_and_parse_config') as mock_config:
            mock_config.side_effect = lambda model_id, filename: (
                {"chat_template": template} if filename == "tokenizer_config.json" else {}
            )
            with mock.patch.object(extractor, '_get_readme_content', return_value=""):
                result = extractor.extract_metadata(
                    "owner/model", {}, None,
                    template_attestation=attestation
                )

        status = result["template_security_status"]
        assert status["status"] == "clean"
        assert status["scanner_name"] == "TemplateScanner"

    def test_no_chat_template_fields_when_template_missing(self, extractor):
        with mock.patch.object(extractor, '_download_and_parse_config') as mock_config:
            mock_config.return_value = {}
            with mock.patch.object(extractor, '_get_readme_content', return_value=""):
                with mock.patch('enhanced_extractor.GGUF_AVAILABLE', False):
                    result = extractor.extract_metadata("owner/model", {}, None)

        assert "chat_template_hash" not in result
        assert "template_source" not in result
        assert "template_security_status" not in result

    def test_extraction_results_have_high_confidence(self, extractor):
        template = "{{ msg }}"

        with mock.patch.object(extractor, '_download_and_parse_config') as mock_config:
            mock_config.side_effect = lambda model_id, filename: (
                {"chat_template": template} if filename == "tokenizer_config.json" else {}
            )
            with mock.patch.object(extractor, '_get_readme_content', return_value=""):
                extractor.extract_metadata("owner/model", {}, None)

        results = extractor.get_extraction_results()

        assert "chat_template_hash" in results
        assert results["chat_template_hash"].confidence == ConfidenceLevel.HIGH


class TestGGUFExtractionFallback:

    @pytest.fixture
    def mock_hf_api(self):
        api = mock.MagicMock()
        api.token = None
        return api

    @pytest.fixture
    def extractor(self, mock_hf_api):
        return EnhancedExtractor(hf_api=mock_hf_api)

    def test_gguf_extraction_triggered_when_no_tokenizer_template(self, extractor, gguf_model_info_factory):
        gguf_info = gguf_model_info_factory(
            chat_template="{% for m in messages %}{{ m }}{% endfor %}"
        )

        with mock.patch.object(extractor, '_download_and_parse_config') as mock_config:
            mock_config.return_value = {}
            with mock.patch.object(extractor, '_get_readme_content', return_value=""):
                with mock.patch('enhanced_extractor.GGUF_AVAILABLE', True):
                    with mock.patch('enhanced_extractor.list_gguf_files', return_value=["model.gguf"]):
                        with mock.patch('enhanced_extractor.fetch_gguf_metadata_from_repo', return_value=gguf_info):
                            with mock.patch('enhanced_extractor.map_gguf_to_aibom_metadata') as mock_map:
                                mock_map.return_value = {
                                    "chat_template_hash": "sha256:abc123",
                                    "model_type": "llama",
                                }
                                result = extractor.extract_metadata("owner/model", {}, None)

        assert "chat_template_hash" in result

    def test_gguf_not_triggered_when_tokenizer_has_template(self, extractor):
        template = "{{ msg }}"

        with mock.patch.object(extractor, '_download_and_parse_config') as mock_config:
            mock_config.side_effect = lambda model_id, filename: (
                {"chat_template": template} if filename == "tokenizer_config.json" else {}
            )
            with mock.patch.object(extractor, '_get_readme_content', return_value=""):
                with mock.patch('enhanced_extractor.list_gguf_files') as mock_list:
                    result = extractor.extract_metadata("owner/model", {}, None)

        mock_list.assert_not_called()
