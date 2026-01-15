from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "aibom-generator"))

from gguf_metadata import (
    GGUF_MAGIC,
    GGUFValueType,
    GGUFModelInfo,
    BufferUnderrunError,
    InvalidMagicError,
    HashValue,
    parse_gguf_metadata,
    extract_chat_template_info,
    extract_model_info,
    build_huggingface_url,
    map_gguf_to_aibom_metadata,
)


class TestParseGGUFMetadata:

    def test_parse_minimal_valid_gguf(self, gguf_bytes_factory):
        data = gguf_bytes_factory(architecture="llama", model_name="test-model")

        result = parse_gguf_metadata(data, filename="test.gguf")

        assert result.version == 3
        assert result.tensor_count == 0
        assert result.kv_count == 2
        assert result.metadata["general.architecture"] == "llama"
        assert result.metadata["general.name"] == "test-model"
        assert result.filename == "test.gguf"

    def test_parse_with_chat_template(self, gguf_bytes_factory, sample_chat_template):
        data = gguf_bytes_factory(chat_template=sample_chat_template)

        result = parse_gguf_metadata(data)

        assert "tokenizer.chat_template" in result.metadata
        assert result.metadata["tokenizer.chat_template"] == sample_chat_template

    def test_parse_with_various_metadata_types(self, gguf_bytes_factory):
        data = gguf_bytes_factory(
            metadata={
                "test.uint32": (GGUFValueType.UINT32, 42),
                "test.int32": (GGUFValueType.INT32, -100),
                "test.float32": (GGUFValueType.FLOAT32, 3.14),
                "test.bool_true": (GGUFValueType.BOOL, True),
                "test.bool_false": (GGUFValueType.BOOL, False),
                "test.string": (GGUFValueType.STRING, "hello world"),
            }
        )

        result = parse_gguf_metadata(data)

        assert result.metadata["test.uint32"] == 42
        assert result.metadata["test.int32"] == -100
        assert abs(result.metadata["test.float32"] - 3.14) < 0.001
        assert result.metadata["test.bool_true"] is True
        assert result.metadata["test.bool_false"] is False
        assert result.metadata["test.string"] == "hello world"

    def test_parse_invalid_magic_raises(self):
        data = b"NOT_GGUF_DATA_HERE"

        with pytest.raises(InvalidMagicError) as exc_info:
            parse_gguf_metadata(data)

        assert "invalid magic" in str(exc_info.value).lower()

    def test_parse_truncated_header_raises(self):
        data = struct.pack("<I", GGUF_MAGIC)

        with pytest.raises(BufferUnderrunError) as exc_info:
            parse_gguf_metadata(data)

        assert exc_info.value.required_bytes is not None
        assert exc_info.value.required_bytes > len(data)

    def test_parse_truncated_metadata_raises(self, gguf_bytes_factory):
        data = gguf_bytes_factory(chat_template="a" * 1000)
        truncated = data[:50]

        with pytest.raises(BufferUnderrunError) as exc_info:
            parse_gguf_metadata(truncated)

        assert exc_info.value.required_bytes > len(truncated)

    def test_buffer_underrun_carries_required_bytes(self):
        data = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 3)

        with pytest.raises(BufferUnderrunError) as exc_info:
            parse_gguf_metadata(data)

        assert exc_info.value.required_bytes is not None
        assert exc_info.value.required_bytes > 8


class TestExtractChatTemplateInfo:

    def test_extract_default_template(self, sample_chat_template, sample_chat_template_hash):
        metadata = {"tokenizer.chat_template": sample_chat_template}

        result = extract_chat_template_info(metadata)

        assert result.has_template is True
        assert result.default_template == sample_chat_template
        assert result.template_hash == sample_chat_template_hash
        assert result.named_templates == {}
        assert result.template_names == []

    def test_extract_named_templates(self):
        metadata = {
            "tokenizer.chat_template": "{{ default }}",
            "tokenizer.chat_templates": ["chatml", "plain"],
            "tokenizer.chat_template.chatml": "{{ chatml_template }}",
            "tokenizer.chat_template.plain": "{{ plain_template }}",
        }

        result = extract_chat_template_info(metadata)

        assert result.has_template is True
        assert result.default_template == "{{ default }}"
        assert "chatml" in result.named_templates
        assert "plain" in result.named_templates
        assert result.named_templates["chatml"] == "{{ chatml_template }}"
        assert result.named_templates["plain"] == "{{ plain_template }}"

    def test_extract_fallback_named_templates(self):
        metadata = {
            "tokenizer.chat_template": "{{ default }}",
            "tokenizer.chat_template.tool_use": "{{ tool_use }}",
            "tokenizer.chat_template.rag": "{{ rag }}",
        }

        result = extract_chat_template_info(metadata)

        assert "tool_use" in result.named_templates
        assert "rag" in result.named_templates

    def test_no_template(self):
        metadata = {"general.architecture": "bert", "general.name": "bert-base"}

        result = extract_chat_template_info(metadata)

        assert result.has_template is False
        assert result.default_template is None
        assert result.template_hash is None
        assert result.named_templates == {}

    def test_different_templates_different_hashes(self):
        template1 = "{{ message1 }}"
        template2 = "{{ message2 }}"

        result1 = extract_chat_template_info({"tokenizer.chat_template": template1})
        result2 = extract_chat_template_info({"tokenizer.chat_template": template2})

        assert result1.template_hash != result2.template_hash


class TestExtractModelInfo:

    def test_extract_full_model_info(self, gguf_bytes_factory, sample_chat_template):
        data = gguf_bytes_factory(
            architecture="llama",
            model_name="Llama-2-7B-Chat",
            chat_template=sample_chat_template,
            metadata={
                "tokenizer.ggml.model": (GGUFValueType.STRING, "gpt2"),
                "llama.context_length": (GGUFValueType.UINT32, 4096),
                "llama.embedding_length": (GGUFValueType.UINT32, 4096),
                "llama.block_count": (GGUFValueType.UINT32, 32),
                "general.quantization_version": (GGUFValueType.UINT32, 2),
                "general.file_type": (GGUFValueType.UINT32, 7),
            }
        )

        parsed = parse_gguf_metadata(data, filename="llama.gguf")
        result = extract_model_info(parsed)

        assert result.filename == "llama.gguf"
        assert result.architecture == "llama"
        assert result.name == "Llama-2-7B-Chat"
        assert result.tokenizer_model == "gpt2"
        assert result.context_length == 4096
        assert result.embedding_length == 4096
        assert result.block_count == 32
        assert result.quantization_version == 2
        assert result.file_type == 7
        assert result.chat_template is not None
        assert result.chat_template.has_template is True
        assert result.chat_template.default_template == sample_chat_template

    def test_extract_minimal_model_info(self, gguf_bytes_factory):
        data = gguf_bytes_factory(architecture="gpt2", model_name="GPT-2")

        parsed = parse_gguf_metadata(data)
        result = extract_model_info(parsed)

        assert result.architecture == "gpt2"
        assert result.name == "GPT-2"
        assert result.chat_template is not None
        assert result.chat_template.has_template is False


class TestBuildHuggingfaceUrl:

    def test_basic_url_construction(self):
        url = build_huggingface_url("meta-llama/Llama-2-7b-chat-hf", "model.gguf")
        assert url == "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/model.gguf"

    def test_custom_revision(self):
        url = build_huggingface_url("owner/repo", "file.gguf", revision="v1.0")
        assert "resolve/v1.0/" in url

    def test_nested_filename(self):
        url = build_huggingface_url("owner/repo", "models/quantized/model.gguf")
        assert "models/quantized/model.gguf" in url

    def test_invalid_repo_id_raises(self):
        with pytest.raises(ValueError):
            build_huggingface_url("invalid", "file.gguf")

        with pytest.raises(ValueError):
            build_huggingface_url("", "file.gguf")

    def test_special_characters_encoded(self):
        url = build_huggingface_url("owner/repo-name", "model file.gguf")
        assert "model%20file.gguf" in url


class TestMapGgufToAibomMetadata:

    def test_map_with_chat_template_hash_only_by_default(self, gguf_model_info_factory):
        template = "{% for m in messages %}{{ m }}{% endfor %}"
        info = gguf_model_info_factory(
            filename="model.gguf",
            architecture="llama",
            name="Test Model",
            chat_template=template,
        )

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert "chat_template" not in result
        assert result["chat_template_hash"].startswith("sha256:")
        assert result["model_type"] == "llama"
        assert result["gguf_filename"] == "model.gguf"

    def test_map_with_chat_template_content_opt_in(self, gguf_model_info_factory):
        template = "{% for m in messages %}{{ m }}{% endfor %}"
        info = gguf_model_info_factory(
            filename="model.gguf",
            architecture="llama",
            name="Test Model",
            chat_template=template,
        )

        result = map_gguf_to_aibom_metadata(info, "owner/repo", include_template_content=True)

        assert result["chat_template"] == template
        assert result["chat_template_hash"].startswith("sha256:")

    def test_extraction_provenance_tracks_source(self, gguf_model_info_factory):
        info = gguf_model_info_factory(filename="model.gguf", chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        prov = result["extraction_provenance"]
        assert prov["source_file"] == "model.gguf"
        assert prov["source_type"] == "gguf_embedded"
        assert prov["source_repository"] == "https://huggingface.co/owner/repo"
        assert prov["extraction_timestamp"].endswith("Z")
        assert prov["extractor_tool"] == "aibom-generator"

    def test_model_lineage_indicates_no_inheritance_for_gguf(self, gguf_model_info_factory):
        info = gguf_model_info_factory(chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        lineage = result["model_lineage"]
        assert lineage["inherited_from_base"] is False
        assert lineage["base_model"] is None
        assert lineage["derivation_method"] is None

    def test_security_status_defaults_to_unscanned(self, gguf_model_info_factory):
        info = gguf_model_info_factory(chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        status = result["template_security_status"]
        assert status["status"] == "unscanned"
        assert status["subject"]["type"] == "chat_template"
        assert status["subject"]["hash"] == result["chat_template_hash"]
        assert status["report_uri"] is None

    def test_map_without_chat_template(self, gguf_model_info_factory):
        info = gguf_model_info_factory(architecture="bert", name="BERT Base", chat_template=None)

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert "chat_template" not in result
        assert "chat_template_hash" not in result
        assert "extraction_provenance" not in result
        assert result["model_type"] == "bert"

    def test_supplementary_fields_mapped(self):
        info = GGUFModelInfo(
            filename="model.gguf",
            architecture="llama",
            name="Test Model",
            description="A test language model",
            license="MIT",
            author="Test Author",
            context_length=4096,
            embedding_length=4096,
            block_count=32,
            attention_head_count=32,
            attention_head_count_kv=8,
            quantization_version=2,
            file_type=7,
        )

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert result["description"] == "A test language model"
        assert result["gguf_license"] == "MIT"
        assert result["suppliedBy"] == "Test Author"

        assert "quantization" in result
        assert result["quantization"]["version"] == 2
        assert result["quantization"]["file_type"] == 7

        assert "hyperparameter" in result
        assert result["hyperparameter"]["context_length"] == 4096
        assert result["hyperparameter"]["embedding_length"] == 4096
        assert result["hyperparameter"]["block_count"] == 32
        assert result["hyperparameter"]["attention_head_count"] == 32
        assert result["hyperparameter"]["attention_head_count_kv"] == 8
        assert "vocab_size" not in result["hyperparameter"]


class TestCycloneDXCompatibility:

    def test_provides_namespaced_properties_for_cyclonedx_insertion(self, gguf_model_info_factory):
        info = gguf_model_info_factory(filename="model.gguf", chat_template="{{ message }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        props = result.get("cdx_component_properties", [])
        prop_dict = {p["name"]: p["value"] for p in props}

        assert any(name.startswith("aibom:") for name in prop_dict.keys())
        assert "aibom:chat_template_hash" in prop_dict

    def test_template_content_requires_explicit_opt_in(self, gguf_model_info_factory):
        template = "{{ message }}"
        info = gguf_model_info_factory(filename="model.gguf", chat_template=template)

        result_default = map_gguf_to_aibom_metadata(info, "owner/repo")
        prop_names = [p["name"] for p in result_default.get("cdx_component_properties", [])]
        assert "aibom:chat_template" not in prop_names

        result_opted = map_gguf_to_aibom_metadata(info, "owner/repo", include_template_content=True)
        prop_names = [p["name"] for p in result_opted.get("cdx_component_properties", [])]
        assert "aibom:chat_template" in prop_names

    def test_attestation_derived_from_security_status(self, gguf_model_info_factory):
        info = gguf_model_info_factory(filename="model.gguf", chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        security_status = result["template_security_status"]
        attestation = result["cdx_attestation"]

        assert attestation["map"][0]["status"] == security_status["status"]
        assert security_status["subject"]["hash"] in attestation["map"][0]["claims"]

    def test_no_cyclonedx_fields_without_chat_template(self, gguf_model_info_factory):
        info = gguf_model_info_factory(chat_template=None)

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert "cdx_component_properties" not in result
        assert "cdx_attestation" not in result


class TestTimestampInjection:

    def test_timestamp_can_be_injected_for_deterministic_output(self, gguf_model_info_factory):
        from datetime import datetime, timezone

        fixed_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        info = gguf_model_info_factory(chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo", now=fixed_time)

        assert result["extraction_provenance"]["extraction_timestamp"] == "2025-06-15T12:00:00Z"

    def test_timestamp_defaults_to_now_when_not_injected(self, gguf_model_info_factory):
        from datetime import datetime, timezone

        before = datetime.now(timezone.utc)
        info = gguf_model_info_factory(chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        after = datetime.now(timezone.utc)

        ts = result["extraction_provenance"]["extraction_timestamp"]
        assert ts.endswith("Z")
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert before <= parsed <= after


class TestHashValue:

    def test_from_content_creates_sha256_hash(self):
        content = "test content"
        h = HashValue.from_content(content)

        assert h.algorithm == "SHA-256"
        assert len(h.value) == 64
        assert h.value == hashlib.sha256(content.encode("utf-8")).hexdigest()

    def test_to_cyclonedx_produces_structured_format(self):
        h = HashValue(algorithm="SHA-256", value="abc123")

        cdx = h.to_cyclonedx()

        assert cdx == {"alg": "SHA-256", "content": "abc123"}

    def test_to_prefixed_produces_string_format(self):
        h = HashValue(algorithm="SHA-256", value="abc123def456")

        prefixed = h.to_prefixed()

        assert prefixed == "sha256:abc123def456"

    def test_roundtrip_content_to_both_formats(self):
        content = "{% for m in messages %}{{ m.content }}{% endfor %}"
        h = HashValue.from_content(content)

        prefixed = h.to_prefixed()
        cdx = h.to_cyclonedx()

        assert prefixed.startswith("sha256:")
        assert cdx["alg"] == "SHA-256"
        assert cdx["content"] == prefixed.split(":")[1]


class TestStructuredHashesInMetadata:

    def test_structured_hash_included_in_chat_template_info(self, gguf_bytes_factory):
        template = "{{ message }}"
        data = gguf_bytes_factory(chat_template=template)

        parsed = parse_gguf_metadata(data)
        ct_info = extract_chat_template_info(parsed.metadata)

        assert ct_info.template_hash_structured is not None
        assert ct_info.template_hash_structured.algorithm == "SHA-256"
        assert ct_info.template_hash == ct_info.template_hash_structured.to_prefixed()

    def test_structured_hash_in_aibom_metadata(self, gguf_model_info_factory):
        info = gguf_model_info_factory(filename="model.gguf", chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert "chat_template_hash_structured" in result
        h = result["chat_template_hash_structured"]
        assert h["alg"] == "SHA-256"
        assert h["content"] == result["chat_template_hash"].split(":")[1]

    def test_security_status_includes_structured_hash(self, gguf_model_info_factory):
        info = gguf_model_info_factory(filename="model.gguf", chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        status = result["template_security_status"]
        assert "hash_structured" in status["subject"]
        assert status["subject"]["hash_structured"]["alg"] == "SHA-256"

    def test_cdx_component_hashes_array(self, gguf_model_info_factory):
        info = gguf_model_info_factory(filename="model.gguf", chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert "cdx_component_hashes" in result
        hashes = result["cdx_component_hashes"]
        assert len(hashes) == 1
        assert hashes[0]["alg"] == "SHA-256"

    def test_named_templates_have_structured_hashes(self, gguf_model_info_factory):
        info = gguf_model_info_factory(
            filename="model.gguf",
            chat_template="{{ default }}",
            named_templates={"tool_use": "{{ tool }}", "rag": "{{ rag }}"}
        )

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert "named_chat_templates_structured" in result
        structured = result["named_chat_templates_structured"]
        assert "tool_use" in structured
        assert "rag" in structured
        assert structured["tool_use"]["alg"] == "SHA-256"
