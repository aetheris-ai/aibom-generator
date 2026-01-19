"""
Shared pytest fixtures for AIBOM Generator tests.

Provides mock HuggingFace API responses to enable offline testing.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class MockModelInfo:
    """Mock HuggingFace ModelInfo object."""

    def __init__(
        self,
        model_id="test-org/test-model",
        author="test-org",
        pipeline_tag="text-generation",
        library_name="transformers",
        license="apache-2.0",
        tags=None,
        downloads=1000,
        likes=100,
        created_at=None,
        last_modified=None,
    ):
        self.id = model_id
        self.modelId = model_id
        self.author = author
        self.pipeline_tag = pipeline_tag
        self.library_name = library_name
        self.license = license
        self.tags = tags or ["pytorch", "text-generation"]
        self.downloads = downloads
        self.likes = likes
        self.created_at = created_at or datetime(2024, 1, 1)
        self.last_modified = last_modified or datetime(2024, 6, 1)
        self.sha = "abc123def456"
        self.private = False
        self.disabled = False
        self.gated = False
        self.siblings = []
        self.config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
        self.card_data = None
        self.safetensors = None


class MockModelCard:
    """Mock HuggingFace ModelCard object."""

    def __init__(self, content="", data=None):
        self.content = content
        self.data = data or MockModelCardData()
        self.text = content


class MockModelCardData:
    """Mock ModelCard data."""

    def __init__(self):
        self.license = "apache-2.0"
        self.language = ["en"]
        self.tags = ["text-generation"]
        self.datasets = ["test-dataset"]
        self.metrics = []
        self.model_name = None
        self.eval_results = []
        self.library_name = "transformers"
        self.pipeline_tag = "text-generation"
        self.base_model = None

    def to_dict(self):
        return {
            "license": self.license,
            "language": self.language,
            "tags": self.tags,
            "datasets": self.datasets,
        }


@pytest.fixture
def mock_model_info():
    """Return a mock ModelInfo object with default values."""
    return MockModelInfo()


@pytest.fixture
def mock_model_info_llama():
    """Return a mock ModelInfo object for a Llama-style model."""
    return MockModelInfo(
        model_id="meta-llama/Llama-2-7b",
        author="meta-llama",
        pipeline_tag="text-generation",
        library_name="transformers",
        license="llama2",
        tags=["pytorch", "llama", "text-generation"],
        downloads=1000000,
        likes=5000,
    )


@pytest.fixture
def mock_model_info_whisper():
    """Return a mock ModelInfo object for a Whisper-style model."""
    return MockModelInfo(
        model_id="openai/whisper-large-v3",
        author="openai",
        pipeline_tag="automatic-speech-recognition",
        library_name="transformers",
        license="apache-2.0",
        tags=["pytorch", "whisper", "audio"],
        downloads=500000,
        likes=2000,
    )


@pytest.fixture
def mock_model_card():
    """Return a mock ModelCard object."""
    content = """
# Test Model

This is a test model for unit testing.

## Model Details

- **Model type:** Transformer
- **Language:** English
- **License:** Apache 2.0

## Training Data

Trained on test dataset.

## Limitations

This model has some limitations.
"""
    return MockModelCard(content=content)


@pytest.fixture
def mock_hf_api(mock_model_info, mock_model_card):
    """
    Fixture that mocks the HuggingFace Hub API.

    Usage:
        def test_something(mock_hf_api):
            # HfApi calls are now mocked
            generator = AIBOMGenerator()
            result = generator.generate_aibom("test/model")
    """
    with patch('huggingface_hub.HfApi') as mock_api_class:
        mock_api = MagicMock()
        mock_api.model_info.return_value = mock_model_info
        mock_api_class.return_value = mock_api

        with patch('huggingface_hub.ModelCard') as mock_card_class:
            mock_card_class.load.return_value = mock_model_card

            yield {
                'api': mock_api,
                'api_class': mock_api_class,
                'card_class': mock_card_class,
                'model_info': mock_model_info,
                'model_card': mock_model_card,
            }


@pytest.fixture
def sample_aibom():
    """Return a sample valid AIBOM structure for testing."""
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:test-uuid",
        "version": 1,
        "metadata": {
            "timestamp": "2024-01-01T00:00:00Z",
            "tools": {
                "components": [{
                    "type": "application",
                    "name": "OWASP AIBOM Generator",
                    "version": "1.0.0"
                }]
            },
            "component": {
                "type": "application",
                "name": "test-model",
                "version": "1.0"
            }
        },
        "components": [{
            "bom-ref": "pkg:huggingface/test-org/test-model@1.0",
            "type": "machine-learning-model",
            "name": "test-model",
            "version": "1.0",
            "purl": "pkg:huggingface/test-org%2Ftest-model@1.0",
            "licenses": [{"license": {"id": "apache-2.0"}}],
            "modelCard": {
                "modelParameters": {
                    "task": "text-generation"
                }
            }
        }],
        "dependencies": []
    }


@pytest.fixture
def cyclonedx_schema():
    """Return the CycloneDX 1.6 JSON schema for validation."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["bomFormat", "specVersion"],
        "properties": {
            "bomFormat": {"type": "string", "enum": ["CycloneDX"]},
            "specVersion": {"type": "string"},
            "serialNumber": {"type": "string"},
            "version": {"type": "integer"},
            "metadata": {"type": "object"},
            "components": {"type": "array"},
            "dependencies": {"type": "array"}
        }
    }
