from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "aibom-generator"))

from gguf_metadata import (
    map_gguf_to_aibom_metadata,
)


class TestChatTemplateAIBOMFields:

    def test_hash_always_available_template_content_opt_in(self, gguf_model_info_factory):
        template = "{% for m in messages %}{{ m.content }}{% endfor %}"
        info = gguf_model_info_factory(chat_template=template)

        result_default = map_gguf_to_aibom_metadata(info, "owner/repo")
        result_opted = map_gguf_to_aibom_metadata(info, "owner/repo", include_template_content=True)

        assert result_default["chat_template_hash"].startswith("sha256:")
        assert "chat_template" not in result_default
        assert result_opted["chat_template"] == template

    def test_hash_format_is_sha256_with_64_hex_chars(self, gguf_model_info_factory):
        info = gguf_model_info_factory(chat_template="{% for m in messages %}{{ m }}{% endfor %}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        hash_value = result["chat_template_hash"]
        assert hash_value.startswith("sha256:")
        assert len(hash_value) == 7 + 64

    def test_provenance_identifies_source_location(self, gguf_model_info_factory):
        info = gguf_model_info_factory(
            filename="model.Q4_K_M.gguf",
            chat_template="{{ msg }}",
        )

        result = map_gguf_to_aibom_metadata(info, "meta-llama/Llama-2-7b")

        prov = result["extraction_provenance"]
        assert prov["source_file"] == "model.Q4_K_M.gguf"
        assert "meta-llama/Llama-2-7b" in prov["source_repository"]
        assert prov["source_type"] == "gguf_embedded"

    def test_lineage_and_provenance_are_separate_concerns(self, gguf_model_info_factory):
        info = gguf_model_info_factory(chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert "extraction_provenance" in result
        assert "model_lineage" in result
        assert "source_file" in result["extraction_provenance"]
        assert "inherited_from_base" in result["model_lineage"]

    def test_security_status_starts_unscanned(self, gguf_model_info_factory):
        info = gguf_model_info_factory(chat_template="{{ msg }}")

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        status = result["template_security_status"]
        assert status["status"] == "unscanned"
        assert status["subject"]["hash"] == result["chat_template_hash"]
        assert status["findings"] == []

    def test_models_without_templates_produce_no_template_fields(self, gguf_model_info_factory):
        info = gguf_model_info_factory(chat_template=None)

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        for field in ["chat_template", "chat_template_hash", "extraction_provenance", "template_security_status"]:
            assert field not in result


class TestNamedTemplates:

    def test_named_templates_tracked_with_hashes(self):
        from gguf_metadata import extract_chat_template_info

        metadata = {
            "tokenizer.chat_template": "{{ default }}",
            "tokenizer.chat_templates": ["chatml", "tool_use", "rag"],
            "tokenizer.chat_template.chatml": "{{ chatml }}",
            "tokenizer.chat_template.tool_use": "{{ tool_use }}",
            "tokenizer.chat_template.rag": "{{ rag }}",
        }

        result = extract_chat_template_info(metadata)

        assert result.has_template is True
        assert len(result.named_templates) == 3
        assert set(result.template_names) == {"chatml", "tool_use", "rag"}
        assert len(result.named_template_hashes) == 3
        assert result.named_template_hashes["chatml"].startswith("sha256:")

    def test_named_templates_in_aibom_have_hashes(self, gguf_model_info_factory):
        info = gguf_model_info_factory(
            chat_template="{{ default }}",
            named_templates={
                "chatml": "{{ chatml }}",
                "tool_use": "{{ tool_use }}",
            },
        )

        result = map_gguf_to_aibom_metadata(info, "owner/repo")

        assert "named_chat_templates" in result
        assert result["named_chat_templates"]["chatml"].startswith("sha256:")
        assert result["named_chat_templates"]["tool_use"].startswith("sha256:")


class TestSecurityConsiderations:

    def test_template_preserves_exact_content_when_opted_in(self, gguf_model_info_factory):
        template = """{% for message in messages %}
<|{{ message['role'] }}|>
{{ message['content'] | escape }}
{% endfor %}
{% if add_generation_prompt %}<|assistant|>{% endif %}"""

        info = gguf_model_info_factory(chat_template=template)
        result = map_gguf_to_aibom_metadata(info, "owner/repo", include_template_content=True)

        assert result["chat_template"] == template

        import hashlib
        actual_hash = f"sha256:{hashlib.sha256(result['chat_template'].encode()).hexdigest()}"
        assert actual_hash == result["chat_template_hash"]

    def test_unicode_template_handled(self, gguf_model_info_factory):
        template = "{% for m in messages %}{{ m.content }}�{% endfor %}"

        info = gguf_model_info_factory(chat_template=template)
        result = map_gguf_to_aibom_metadata(info, "owner/repo", include_template_content=True)

        assert result["chat_template"] == template

        import hashlib
        expected_hash = f"sha256:{hashlib.sha256(template.encode('utf-8')).hexdigest()}"
        assert result["chat_template_hash"] == expected_hash

    def test_empty_template_not_treated_as_no_template(self):
        from gguf_metadata import extract_chat_template_info

        metadata_empty = {"tokenizer.chat_template": ""}
        result_empty = extract_chat_template_info(metadata_empty)

        metadata_none = {"general.architecture": "llama"}
        result_none = extract_chat_template_info(metadata_none)

        assert result_empty.default_template == ""
        assert result_none.default_template is None
