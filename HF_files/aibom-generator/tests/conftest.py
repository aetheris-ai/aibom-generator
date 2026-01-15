from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Dict, Optional
from unittest import mock

import httpx
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "aibom-generator"))

from gguf_metadata import (
    GGUF_MAGIC,
    GGUFValueType,
    GGUFModelInfo,
    GGUFChatTemplateInfo,
    HashValue,
)


def _write_gguf_string(buffer: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buffer.extend(struct.pack("<Q", len(encoded)))
    buffer.extend(encoded)


def _write_gguf_kv(buffer: bytearray, key: str, value_type: int, value) -> None:
    _write_gguf_string(buffer, key)
    buffer.extend(struct.pack("<I", value_type))

    if value_type == GGUFValueType.STRING:
        _write_gguf_string(buffer, value)
    elif value_type == GGUFValueType.UINT32:
        buffer.extend(struct.pack("<I", value))
    elif value_type == GGUFValueType.UINT64:
        buffer.extend(struct.pack("<Q", value))
    elif value_type == GGUFValueType.INT32:
        buffer.extend(struct.pack("<i", value))
    elif value_type == GGUFValueType.FLOAT32:
        buffer.extend(struct.pack("<f", value))
    elif value_type == GGUFValueType.BOOL:
        buffer.extend(struct.pack("<B", 1 if value else 0))
    elif value_type == GGUFValueType.ARRAY:
        elem_type, elements = value
        buffer.extend(struct.pack("<I", elem_type))
        buffer.extend(struct.pack("<Q", len(elements)))
        for elem in elements:
            if elem_type == GGUFValueType.STRING:
                _write_gguf_string(buffer, elem)
            elif elem_type == GGUFValueType.UINT32:
                buffer.extend(struct.pack("<I", elem))


@pytest.fixture
def gguf_bytes_factory() -> Callable[..., bytes]:
    def factory(
        *,
        version: int = 3,
        tensor_count: int = 0,
        metadata: Optional[Dict] = None,
        chat_template: Optional[str] = None,
        architecture: str = "test-arch",
        model_name: str = "test-model",
    ) -> bytes:
        buffer = bytearray()
        buffer.extend(struct.pack("<I", GGUF_MAGIC))
        buffer.extend(struct.pack("<I", version))
        buffer.extend(struct.pack("<Q", tensor_count))

        kv_pairs = []
        kv_pairs.append(("general.architecture", GGUFValueType.STRING, architecture))
        kv_pairs.append(("general.name", GGUFValueType.STRING, model_name))

        if chat_template:
            kv_pairs.append(("tokenizer.chat_template", GGUFValueType.STRING, chat_template))

        if metadata:
            for key, (value_type, value) in metadata.items():
                kv_pairs.append((key, value_type, value))

        buffer.extend(struct.pack("<Q", len(kv_pairs)))

        for key, value_type, value in kv_pairs:
            _write_gguf_kv(buffer, key, value_type, value)

        return bytes(buffer)

    return factory


@pytest.fixture
def sample_chat_template() -> str:
    return (
        "{% for message in messages %}\n"
        "{{ '<|' ~ message['role'] ~ '|>' ~ message['content'] }}\n"
        "{% endfor %}\n"
        "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
    )


@pytest.fixture
def sample_chat_template_hash(sample_chat_template) -> str:
    return f"sha256:{hashlib.sha256(sample_chat_template.encode('utf-8')).hexdigest()}"


@pytest.fixture
def mock_httpx_client_factory():
    def factory(*, head_responses=None, get_responses=None):
        head_iter = iter(head_responses or [(200, {})])
        get_iter = iter(get_responses or [])

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def head(self, url, *, headers=None):
                status_code, resp_headers = next(head_iter)
                response = mock.create_autospec(httpx.Response, instance=True)
                response.status_code = status_code
                response.headers = resp_headers
                return response

            def get(self, url, *, headers=None):
                status_code, content = next(get_iter)
                response = mock.create_autospec(httpx.Response, instance=True)
                response.status_code = status_code
                response.content = content
                response.raise_for_status = mock.Mock()
                return response

        return MockClient

    return factory


@pytest.fixture
def gguf_model_info_factory() -> Callable[..., GGUFModelInfo]:
    """
    Factory for creating GGUFModelInfo instances.

    Automatically computes template_hash from chat_template if not provided,
    eliminating duplicate hash computation in tests.
    """
    def factory(
        *,
        filename: str = "test.gguf",
        architecture: str = "llama",
        name: str = "Test Model",
        chat_template: Optional[str] = None,
        template_hash: Optional[str] = None,
        named_templates: Optional[Dict[str, str]] = None,
    ) -> GGUFModelInfo:
        ct_info = None
        if chat_template is not None:
            # Compute hash using HashValue (provides both formats)
            template_hash_structured = HashValue.from_content(chat_template)
            if template_hash is None:
                template_hash = template_hash_structured.to_prefixed()

            # Compute named template hashes if named templates provided
            named_template_hashes = {}
            named_template_hashes_structured = {}
            if named_templates:
                for tname, tcontent in named_templates.items():
                    h = HashValue.from_content(tcontent)
                    named_template_hashes[tname] = h.to_prefixed()
                    named_template_hashes_structured[tname] = h

            ct_info = GGUFChatTemplateInfo(
                has_template=True,
                default_template=chat_template,
                named_templates=named_templates or {},
                template_names=list(named_templates.keys()) if named_templates else [],
                template_hash=template_hash,
                template_hash_structured=template_hash_structured,
                named_template_hashes=named_template_hashes,
                named_template_hashes_structured=named_template_hashes_structured,
            )

        return GGUFModelInfo(
            filename=filename,
            architecture=architecture,
            name=name,
            chat_template=ct_info,
        )

    return factory
