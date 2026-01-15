from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "aibom-generator"))

from gguf_metadata import (
    BufferUnderrunError,
    GGUFParseError,
    fetch_gguf_metadata_from_url,
    fetch_gguf_metadata_from_repo,
)


class TestFetchGGUFMetadataFromUrl:

    def test_returns_parsed_metadata(self, monkeypatch, gguf_bytes_factory, mock_httpx_client_factory):
        gguf_data = gguf_bytes_factory(architecture="llama", model_name="test-model")

        MockClient = mock_httpx_client_factory(
            head_responses=[(200, {})],
            get_responses=[(206, gguf_data)],
        )
        monkeypatch.setattr("httpx.Client", MockClient)

        result = fetch_gguf_metadata_from_url(
            "https://example.com/model.gguf",
            filename="model.gguf",
            initial_request_size=len(gguf_data) + 100,
        )

        assert result.metadata["general.architecture"] == "llama"
        assert result.filename == "model.gguf"

    def test_timeout_raises_exception(self, monkeypatch):
        class TimeoutClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def head(self, url, *, headers=None):
                raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr("httpx.Client", TimeoutClient)

        with pytest.raises(httpx.TimeoutException):
            fetch_gguf_metadata_from_url("https://example.com/model.gguf")

    def test_unparseable_data_raises_error(self, monkeypatch, mock_httpx_client_factory):
        MockClient = mock_httpx_client_factory(
            head_responses=[(200, {})],
            get_responses=[(206, b"not valid gguf data")],
        )
        monkeypatch.setattr("httpx.Client", MockClient)

        with pytest.raises((GGUFParseError, BufferUnderrunError)):
            fetch_gguf_metadata_from_url(
                "https://example.com/model.gguf",
                initial_request_size=100,
                max_request_size=200,
            )


class TestFetchGGUFMetadataFromRepo:

    def test_returns_model_info_for_valid_repo(self, monkeypatch, gguf_bytes_factory):
        gguf_data = gguf_bytes_factory(
            architecture="llama",
            model_name="Test Llama",
            chat_template="{% for m in messages %}{{ m }}{% endfor %}",
        )

        def mock_fetch(url, filename="", **kwargs):
            from gguf_metadata import parse_gguf_metadata
            return parse_gguf_metadata(gguf_data, filename)

        monkeypatch.setattr("gguf_metadata.fetch_gguf_metadata_from_url", mock_fetch)

        result = fetch_gguf_metadata_from_repo("owner/repo", "model.gguf")

        assert result.architecture == "llama"
        assert result.name == "Test Llama"
        assert result.chat_template is not None
        assert result.chat_template.has_template is True
