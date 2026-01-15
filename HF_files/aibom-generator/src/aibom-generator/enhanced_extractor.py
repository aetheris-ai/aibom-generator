#!/usr/bin/env python3
"""
Registry-Integrated (field_registry.json) Enhanced Multi-Layer Data Extraction for AIBOM Generator

This module provides a fully configurable enhanced data extraction system that
automatically picks up new fields from the JSON registry (field_registry.json) without requiring code changes.
It includes comprehensive logging, fallback mechanisms, and confidence tracking.

Key Features:
- Automatically discovers all fields from the registry (field_registry.json)
- Attempts extraction for every registry field
- Provides detailed logging for each field attempt
- Graceful error handling for individual field failures
- Maintains backward compatibility with existing code

"""

import json
import logging
import re
import requests
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse, urljoin
import time

# Import existing dependencies
from huggingface_hub import HfApi, ModelCard, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

# Import field registry manager (field_registry_manager.py)
try:
    from .field_registry_manager import get_field_registry_manager
    REGISTRY_AVAILABLE = True
except ImportError:
    try:
        from field_registry_manager import get_field_registry_manager
        REGISTRY_AVAILABLE = True
    except ImportError:
        REGISTRY_AVAILABLE = False
        print("⚠️ Field registry manager not available, falling back to legacy extraction")

try:
    from .gguf_metadata import (
        list_gguf_files,
        fetch_gguf_metadata_from_repo,
        map_gguf_to_aibom_metadata,
    )
    GGUF_AVAILABLE = True
except ImportError:
    try:
        from gguf_metadata import (
            list_gguf_files,
            fetch_gguf_metadata_from_repo,
            map_gguf_to_aibom_metadata,
        )
        GGUF_AVAILABLE = True
    except ImportError:
        GGUF_AVAILABLE = False
        logger.debug("GGUF metadata extraction not available")

# Configure logging for this module
logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Enumeration of data sources for provenance tracking"""
    HF_API = "huggingface_api"
    MODEL_CARD = "model_card_yaml"
    README_TEXT = "readme_text"
    CONFIG_FILE = "config_file"
    REPOSITORY_FILES = "repository_files"
    EXTERNAL_REFERENCE = "external_reference"
    INTELLIGENT_DEFAULT = "intelligent_default"
    PLACEHOLDER = "placeholder"
    REGISTRY_DRIVEN = "registry_driven"

class ConfidenceLevel(Enum):
    """Confidence levels for extracted data"""
    HIGH = "high"        # Direct API data, official sources
    MEDIUM = "medium"    # Inferred from reliable patterns
    LOW = "low"          # Weak inference or pattern matching
    NONE = "none"        # Placeholder values

@dataclass
class ExtractionResult:
    """Container for extraction results with full provenance"""
    value: Any
    source: DataSource
    confidence: ConfidenceLevel
    extraction_method: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    fallback_chain: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.value} (source: {self.source.value}, confidence: {self.confidence.value})"

class EnhancedExtractor:
    """
    Registry-integrated enhanced extractor that automatically picks up new fields
    from the JSON registry (field_registry.json) without requiring code changes.
    """
    
    def __init__(self, hf_api: Optional[HfApi] = None, field_registry_manager=None):
        """
        Initialize the enhanced extractor with registry integration (field_registry.json and field_registry_manager.py).
        
        Args:
            hf_api: Optional HuggingFace API instance (will create if not provided)
            field_registry_manager.py: Optional registry manager instance
        """
        self.hf_api = hf_api or HfApi()
        self.extraction_results = {}
        
        # Initialize registry manager (field_registry_manager.py)
        self.registry_manager = field_registry_manager
        if not self.registry_manager and REGISTRY_AVAILABLE:
            try:
                self.registry_manager = get_field_registry_manager()
                logger.info("✅ Registry manager initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize registry manager: {e}")
                self.registry_manager = None
        
        # Load registry fields
        self.registry_fields = {}
        if self.registry_manager:
            try:
                registry = self.registry_manager.registry
                self.registry_fields = registry.get('fields', {})
                logger.info(f"✅ Loaded {len(self.registry_fields)} fields from registry")
            except Exception as e:
                logger.error(f"❌ Error loading registry fields: {e}")
                self.registry_fields = {}
        
        # Configure logging
        self._setup_logging()
        
        # Compile regex patterns for text extraction
        self._compile_patterns()
        
        logger.info(f"Enhanced extractor initialized (registry-driven: {bool(self.registry_fields)})")
    
    def _setup_logging(self):
        """Setup logging configuration for detailed extraction tracking"""
        # Ensure a logger that will show in HF Spaces
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    def _compile_patterns(self):
        """Compile regex patterns for text extraction"""
        self.patterns = {
            'license': [
                r'license[:\s]+([a-zA-Z0-9\-\.]+)',
                r'licensed under[:\s]+([a-zA-Z0-9\-\.]+)',
                r'released under[:\s]+([a-zA-Z0-9\-\.]+)',
            ],
            'datasets': [
                r'trained on[:\s]+([a-zA-Z0-9\-\_\/]+)',
                r'dataset[:\s]+([a-zA-Z0-9\-\_\/]+)',
                r'using[:\s]+([a-zA-Z0-9\-\_\/]+)\s+dataset',
            ],
            'metrics': [
                r'([a-zA-Z]+)[:\s]+([0-9\.]+)',
                r'achieves[:\s]+([0-9\.]+)[:\s]+([a-zA-Z]+)',
            ],
            'model_type': [
                r'model type[:\s]+([a-zA-Z0-9\-]+)',
                r'architecture[:\s]+([a-zA-Z0-9\-]+)',
            ],
            'energy': [
                r'energy[:\s]+([0-9\.]+)\s*([a-zA-Z]+)',
                r'power[:\s]+([0-9\.]+)\s*([a-zA-Z]+)',
                r'consumption[:\s]+([0-9\.]+)\s*([a-zA-Z]+)',
            ],
            'limitations': [
                r'limitation[s]?[:\s]+([^\.]+)',
                r'known issue[s]?[:\s]+([^\.]+)',
                r'constraint[s]?[:\s]+([^\.]+)',
            ],
            'safety': [
                r'safety[:\s]+([^\.]+)',
                r'risk[s]?[:\s]+([^\.]+)',
                r'bias[:\s]+([^\.]+)',
            ]
        }
        
        # Compile all patterns
        for category, pattern_list in self.patterns.items():
            self.patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]


    # SPDX mappings for common licences
    LICENSE_MAPPINGS = {
        "mit": "MIT",
        "mit license": "MIT",
        "apache license version 2.0": "Apache-2.0",
        "apache license 2.0": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "apache license, version 2.0": "Apache-2.0",
        "bsd 3-clause": "BSD-3-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "bsd 2-clause": "BSD-2-Clause",
        "bsd-2-clause": "BSD-2-Clause",
        "gnu general public license v3": "GPL-3.0-only",
        "gplv3": "GPL-3.0-only",
        "gnu general public license v2": "GPL-2.0-only",
        "gplv2": "GPL-2.0-only",
    }

    def _detect_license_from_file(self, model_id: str) -> Optional[str]:
        """
        Attempt to detect a licence by looking at repository files.
        Downloads common licence filenames (e.g. LICENSE, LICENSE.md),
        reads a small snippet, and returns the matching SPDX identifier,
        or None if none match.
        """
        license_filenames = ["LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst", "COPYING"]
        for filename in license_filenames:
            try:
                file_path = hf_hub_download(repo_id=model_id, filename=filename)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    snippet = f.read(4096).lower()
                for header, spdx_id in self.LICENSE_MAPPINGS.items():
                    if header in snippet:
                        return spdx_id
            except (RepositoryNotFoundError, EntryNotFoundError):
                # file doesn’t exist; continue
                continue
            except Exception as e:
                logger.debug(f"Licence detection error reading {filename}: {e}")
                continue
        return None

        
    
    def extract_metadata(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard],
                         template_attestation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main extraction method with full registry integration.

        This method automatically discovers all fields from the registry and attempts
        to extract them without requiring code changes when new fields are added.

        Args:
            model_id: Hugging Face model identifier
            model_info: Model information from HF API
            model_card: Model card object from HF
            template_attestation: Optional external security attestation for chat template

        Returns:
            Dictionary of extracted metadata
        """
        logger.info(f"🚀 Starting registry-driven extraction for model: {model_id}")

        self._template_attestation = template_attestation
        
        # Initialize extraction results tracking
        self.extraction_results = {}
        metadata = {}
        
        if self.registry_fields:
            # Registry-driven extraction
            logger.info(f"📋 Registry-driven mode: Attempting extraction for {len(self.registry_fields)} fields")
            metadata = self._registry_driven_extraction(model_id, model_info, model_card)
        else:
            # Fallback to legacy extraction
            logger.warning("⚠️ Registry not available, falling back to legacy extraction")
            metadata = self._legacy_extraction(model_id, model_info, model_card)
        
        # Log extraction summary
        self._log_extraction_summary(model_id, metadata)
        
        # Return metadata in the same format as original method
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _registry_driven_extraction(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard]) -> Dict[str, Any]:
        """
        Registry-driven extraction that automatically processes all registry fields.
        """
        metadata = {}
        
        # Prepare extraction context
        extraction_context = {
            'model_id': model_id,
            'model_info': model_info,
            'model_card': model_card,
            'readme_content': self._get_readme_content(model_card, model_id),
            'config_data': self._download_and_parse_config(model_id, "config.json"),
            'tokenizer_config': self._download_and_parse_config(model_id, "tokenizer_config.json"),
            'template_attestation': getattr(self, '_template_attestation', None)
        }
        
        # Process each field from the registry
        successful_extractions = 0
        failed_extractions = 0
        
        for field_name, field_config in self.registry_fields.items():
            try:
                logger.info(f"🔍 Attempting extraction for field: {field_name}")
                
                # Extract field using registry configuration
                extracted_value = self._extract_registry_field(field_name, field_config, extraction_context)
                
                if extracted_value is not None:
                    metadata[field_name] = extracted_value
                    successful_extractions += 1
                    logger.info(f"✅ Successfully extracted {field_name}: {extracted_value}")
                else:
                    failed_extractions += 1
                    logger.info(f"❌ Failed to extract {field_name}")
                    
            except Exception as e:
                failed_extractions += 1
                logger.error(f"❌ Error extracting {field_name}: {e}")
                # Continue with other fields - individual failures don't stop the process
                continue
        
        logger.info(f"📊 Registry extraction complete: {successful_extractions} successful, {failed_extractions} failed")

        chat_template_fields = self._extract_chat_template_fields(
            model_id=model_id,
            tokenizer_config=extraction_context.get('tokenizer_config'),
            model_info=model_info,
            template_attestation=extraction_context.get('template_attestation')
        )
        if chat_template_fields:
            metadata.update(chat_template_fields)
            logger.info(f"✅ Extracted {len(chat_template_fields)} chat template integrity fields")

        if GGUF_AVAILABLE and not chat_template_fields:
            gguf_metadata = self._extract_gguf_metadata(model_id, extraction_context.get('template_attestation'))
            if gguf_metadata:
                metadata.update(gguf_metadata)
                logger.info(f"✅ Extracted {len(gguf_metadata)} fields from GGUF metadata")

        # Add external references
        metadata.update(self._generate_external_references(model_id, metadata))

        return metadata
    
    def _extract_registry_field(self, field_name: str, field_config: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Extract a single field based on its registry configuration.
        
        This method uses multiple extraction strategies in order of preference:
        1. Direct API extraction
        2. Model card YAML extraction
        3. Text pattern matching
        4. Intelligent inference
        5. Fallback values
        """
        extraction_methods = []
        
        # Strategy 1: Direct API extraction
        api_value = self._try_api_extraction(field_name, context)
        if api_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=api_value,
                source=DataSource.HF_API,
                confidence=ConfidenceLevel.HIGH,
                extraction_method="api_direct"
            )
            extraction_methods.append("api_direct")
            return api_value
        
        # Strategy 2: Model card YAML extraction
        yaml_value = self._try_model_card_extraction(field_name, context)
        if yaml_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=yaml_value,
                source=DataSource.MODEL_CARD,
                confidence=ConfidenceLevel.HIGH,
                extraction_method="model_card_yaml"
            )
            extraction_methods.append("model_card_yaml")
            return yaml_value
        
        # Strategy 3: Configuration file extraction
        config_value = self._try_config_extraction(field_name, context)
        if config_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=config_value,
                source=DataSource.CONFIG_FILE,
                confidence=ConfidenceLevel.HIGH,
                extraction_method="config_file"
            )
            extraction_methods.append("config_file")
            return config_value
        
        # Strategy 4: Text pattern extraction
        text_value = self._try_text_pattern_extraction(field_name, context)
        if text_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=text_value,
                source=DataSource.README_TEXT,
                confidence=ConfidenceLevel.MEDIUM,
                extraction_method="text_pattern"
            )
            extraction_methods.append("text_pattern")
            return text_value
        
        # Strategy 5: Intelligent inference
        inferred_value = self._try_intelligent_inference(field_name, context)
        if inferred_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=inferred_value,
                source=DataSource.INTELLIGENT_DEFAULT,
                confidence=ConfidenceLevel.MEDIUM,
                extraction_method="intelligent_inference"
            )
            extraction_methods.append("intelligent_inference")
            return inferred_value

        # detect licence from repository files if the field is licence/ licences
        if field_name in {"license", "licenses"}:
            detected = self._detect_license_from_file(context["model_id"])
            if detected:
                self.extraction_results[field_name] = ExtractionResult(
                    value=detected,
                    source=DataSource.REPOSITORY_FILES,
                    confidence=ConfidenceLevel.MEDIUM,
                    extraction_method="license_file",
                    fallback_chain=extraction_methods,
                )
                return detected
        
        # Strategy 6: Fallback value (if configured)
        fallback_value = self._try_fallback_value(field_name, field_config)
        if fallback_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=fallback_value,
                source=DataSource.PLACEHOLDER,
                confidence=ConfidenceLevel.NONE,
                extraction_method="fallback_placeholder",
                fallback_chain=extraction_methods
            )
            return fallback_value
        
        # No extraction successful
        self.extraction_results[field_name] = ExtractionResult(
            value=None,
            source=DataSource.PLACEHOLDER,
            confidence=ConfidenceLevel.NONE,
            extraction_method="extraction_failed",
            fallback_chain=extraction_methods
        )
        return None
    
    def _try_api_extraction(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to extract field from HuggingFace API data"""
        model_info = context.get('model_info')
        if not model_info:
            return None
        
        # Field mapping for API extraction
        api_mappings = {
            'author': lambda info: getattr(info, 'author', None) or context['model_id'].split('/')[0],
            'name': lambda info: getattr(info, 'modelId', context['model_id']).split('/')[-1],
            'tags': lambda info: getattr(info, 'tags', []),
            'pipeline_tag': lambda info: getattr(info, 'pipeline_tag', None),
            'downloads': lambda info: getattr(info, 'downloads', 0),
            'commit': lambda info: getattr(info, 'sha', '')[:7] if getattr(info, 'sha', None) else None,
            'suppliedBy': lambda info: getattr(info, 'author', None) or context['model_id'].split('/')[0],
            'primaryPurpose': lambda info: getattr(info, 'pipeline_tag', 'text-generation'),
            'downloadLocation': lambda info: f"https://huggingface.co/{context['model_id']}/tree/main",
            'license': lambda info: getattr(info.card_data, 'license', None) if hasattr(info, 'card_data') and info.card_data else None,
            'licenses': lambda info: getattr(info.card_data, 'license', None) if hasattr(info, 'card_data') and info.card_data else None
        }
        
        if field_name in api_mappings:
            try:
                return api_mappings[field_name](model_info)
            except Exception as e:
                logger.debug(f"API extraction failed for {field_name}: {e}")
                return None
        
        return None
    
    def _try_model_card_extraction(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to extract field from model card YAML frontmatter"""
        model_card = context.get('model_card')
        if not model_card or not hasattr(model_card, 'data') or not model_card.data:
            return None
        
        try:
            card_data = model_card.data.to_dict() if hasattr(model_card.data, 'to_dict') else {}
            
            # Field mapping for model card extraction
            card_mappings = {
                'license': 'license',
                'language': 'language',
                'library_name': 'library_name',
                'base_model': 'base_model',
                'datasets': 'datasets',
                'description': ['model_summary', 'description'],
                'typeOfModel': 'model_type',
                'licenses': 'license'  # Alternative mapping
            }
            
            if field_name in card_mappings:
                mapping = card_mappings[field_name]
                if isinstance(mapping, list):
                    # Try multiple keys
                    for key in mapping:
                        value = card_data.get(key)
                        if value:
                            return value
                else:
                    # Single key
                    return card_data.get(mapping)
            
            # Direct field name lookup
            return card_data.get(field_name)
            
        except Exception as e:
            logger.debug(f"Model card extraction failed for {field_name}: {e}")
            return None
    
    def _try_config_extraction(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to extract field from configuration files"""
        config_data = context.get('config_data')
        tokenizer_config = context.get('tokenizer_config')
        
        # Config file mappings
        config_mappings = {
            'model_type': ('config_data', 'model_type'),
            'architectures': ('config_data', 'architectures'),
            'vocab_size': ('config_data', 'vocab_size'),
            'tokenizer_class': ('tokenizer_config', 'tokenizer_class'),
            'typeOfModel': ('config_data', 'model_type'),
            'chat_template': ('tokenizer_config', 'chat_template')
        }
        
        if field_name in config_mappings:
            config_type, config_key = config_mappings[field_name]
            config_source = context.get(config_type)
            if config_source:
                return config_source.get(config_key)
        
        return None
    
    def _try_text_pattern_extraction(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to extract field using text pattern matching"""
        readme_content = context.get('readme_content')
        if not readme_content:
            return None
        
        # Pattern mappings for different fields
        pattern_mappings = {
            'license': 'license',
            'datasets': 'datasets',
            'energyConsumption': 'energy',
            'limitation': 'limitations',
            'safetyRiskAssessment': 'safety',
            'model_type': 'model_type'
        }
        
        if field_name in pattern_mappings:
            pattern_key = pattern_mappings[field_name]
            if pattern_key in self.patterns:
                matches = self._find_pattern_matches(readme_content, self.patterns[pattern_key])
                if matches:
                    return matches[0] if len(matches) == 1 else matches
        
        return None
    
    def _try_intelligent_inference(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to infer field value from other available data"""
        model_id = context['model_id']
        
        # Intelligent inference rules
        inference_rules = {
            'author': lambda: model_id.split('/')[0] if '/' in model_id else 'unknown',
            'suppliedBy': lambda: model_id.split('/')[0] if '/' in model_id else 'unknown',
            'name': lambda: model_id.split('/')[-1],
            'primaryPurpose': lambda: 'text-generation',  # Default for most HF models
            'typeOfModel': lambda: 'transformer',  # Default for most HF models
            'downloadLocation': lambda: f"https://huggingface.co/{model_id}/tree/main",
            'bomFormat': lambda: 'CycloneDX',
            'specVersion': lambda: '1.6',
            'serialNumber': lambda: f"urn:uuid:{model_id.replace('/', '-')}",
            'version': lambda: '1.0.0'
        }
        
        if field_name in inference_rules:
            try:
                return inference_rules[field_name]()
            except Exception as e:
                logger.debug(f"Intelligent inference failed for {field_name}: {e}")
                return None
        
        return None
    
    def _try_fallback_value(self, field_name: str, field_config: Dict[str, Any]) -> Any:
        """Try to get fallback value from field configuration"""
        # Check if field config has fallback value
        if isinstance(field_config, dict):
            fallback = field_config.get('fallback_value')
            if fallback:
                return fallback
        
        # Standard fallback values for common fields
        standard_fallbacks = {
            'license': 'NOASSERTION',
            'description': 'No description available',
            'version': '1.0.0',
            'bomFormat': 'CycloneDX',
            'specVersion': '1.6'
        }
        
        return standard_fallbacks.get(field_name)
    
    def _legacy_extraction(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard]) -> Dict[str, Any]:
        """
        Fallback to legacy extraction when registry is not available.
        This maintains backward compatibility.
        """
        logger.info("🔄 Executing legacy extraction mode")
        metadata = {}
        
        # Execute legacy extraction layers
        metadata.update(self._layer1_structured_api(model_id, model_info, model_card))
        metadata.update(self._layer2_repository_files(model_id))
        metadata.update(self._layer3_stp_extraction(model_card, model_id))
        metadata.update(self._layer4_external_references(model_id, metadata))
        metadata.update(self._layer5_intelligent_defaults(model_id, metadata))
        
        return metadata
    
    def _generate_external_references(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate external references for the model"""
        external_refs = []
        
        # Model repository
        repo_url = f"https://huggingface.co/{model_id}"
        external_refs.append({
            "type": "website",
            "url": repo_url,
            "comment": "Model repository"
        })
        
        # Model files
        files_url = f"https://huggingface.co/{model_id}/tree/main"
        external_refs.append({
            "type": "distribution",
            "url": files_url,
            "comment": "Model files"
        })
        
        # Commit URL if available
        if 'commit' in metadata:
            commit_url = f"https://huggingface.co/{model_id}/commit/{metadata['commit']}"
            external_refs.append({
                "type": "vcs",
                "url": commit_url,
                "comment": "Specific commit"
            })
        
        # Dataset references
        if 'datasets' in metadata:
            datasets = metadata['datasets']
            if isinstance(datasets, list):
                for dataset in datasets:
                    if isinstance(dataset, str):
                        dataset_url = f"https://huggingface.co/datasets/{dataset}"
                        external_refs.append({
                            "type": "distribution",
                            "url": dataset_url,
                            "comment": f"Training dataset: {dataset}"
                        })
        
        result = {'external_references': external_refs}
        
        self.extraction_results['external_references'] = ExtractionResult(
            value=external_refs,
            source=DataSource.EXTERNAL_REFERENCE,
            confidence=ConfidenceLevel.HIGH,
            extraction_method="url_generation"
        )

        return result

    def _extract_chat_template_fields(self, model_id: str, tokenizer_config: Optional[Dict[str, Any]],
                                       model_info: Dict[str, Any], template_attestation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract chat template and compute integrity metadata.

        This method extracts the chat template from tokenizer_config.json and computes:
        - chat_template: The full Jinja template string
        - chat_template_hash: SHA-256 hash for integrity verification
        - template_source: Provenance information including lineage
        - template_security_status: Security attestation (from external scanner or default unscanned)

        Args:
            model_id: Hugging Face model identifier
            tokenizer_config: Parsed tokenizer_config.json data
            model_info: Model information from HF API
            template_attestation: Optional external security attestation

        Returns:
            Dictionary containing chat template fields
        """
        result = {}

        if not tokenizer_config:
            return result

        chat_template = tokenizer_config.get("chat_template")
        if not chat_template:
            logger.debug(f"No chat_template found in tokenizer_config for {model_id}")
            return result

        logger.info(f"🔧 Found chat_template in {model_id} (length: {len(chat_template)} chars)")
        result["chat_template"] = chat_template

        template_bytes = chat_template.encode('utf-8')
        template_hash = hashlib.sha256(template_bytes).hexdigest()
        result["chat_template_hash"] = f"sha256:{template_hash}"

        base_model = None
        inherited_from_base = False

        if hasattr(model_info, 'card_data') and model_info.card_data:
            base_model = getattr(model_info.card_data, 'base_model', None)
        if not base_model and isinstance(model_info, dict):
            base_model = model_info.get('base_model')

        inherited_from_base = base_model is not None

        result["template_source"] = {
            "source_file": "tokenizer_config.json",
            "source_repository": f"https://huggingface.co/{model_id}",
            "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
            "inherited_from_base": inherited_from_base,
            "base_model": base_model
        }

        if template_attestation:
            result["template_security_status"] = {
                "status": template_attestation.get("status", "unscanned"),
                "scanner_name": template_attestation.get("scanner_name"),
                "scanner_version": template_attestation.get("scanner_version"),
                "scan_timestamp": template_attestation.get("scan_timestamp"),
                "findings": template_attestation.get("findings", [])
            }
            logger.info(f"✅ Using external security attestation: {result['template_security_status']['status']}")
        else:
            result["template_security_status"] = {
                "status": "unscanned",
                "scanner_name": None,
                "scanner_version": None,
                "scan_timestamp": None,
                "findings": []
            }

        for field_name in ["chat_template", "chat_template_hash", "template_source", "template_security_status"]:
            self.extraction_results[field_name] = ExtractionResult(
                value=result[field_name],
                source=DataSource.CONFIG_FILE if field_name in ["chat_template", "chat_template_hash"] else DataSource.REGISTRY_DRIVEN,
                confidence=ConfidenceLevel.HIGH,
                extraction_method="chat_template_extraction"
            )

        return result

    def _extract_gguf_metadata(self, model_id: str, template_attestation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract metadata from GGUF files in the repository.

        Scans ALL GGUF files and groups by (model_name, architecture) to verify
        chat template consistency across quantizations of the same model.
        Different models may legitimately have different templates.

        By default, only includes: chat_template_hash, hyperparameter, and core model fields.
        """
        if not GGUF_AVAILABLE:
            return {}

        try:
            gguf_files = list_gguf_files(model_id, hf_token=getattr(self.hf_api, 'token', None))
            if not gguf_files:
                logger.debug(f"No GGUF files found in {model_id}")
                return {}

            logger.info(f"🔧 Found {len(gguf_files)} GGUF file(s) in {model_id}")

            model_groups = {}
            primary_aibom = None

            for gguf_filename in gguf_files:
                logger.info(f"📦 Scanning GGUF: {gguf_filename}")
                try:
                    gguf_info = fetch_gguf_metadata_from_repo(
                        model_id,
                        gguf_filename,
                        hf_token=getattr(self.hf_api, 'token', None)
                    )

                    gguf_aibom = map_gguf_to_aibom_metadata(
                        gguf_info,
                        model_id,
                        include_template_content=False
                    )

                    model_name = gguf_aibom.get('name', 'unknown')
                    architecture = gguf_aibom.get('model_type', 'unknown')
                    group_key = (model_name, architecture)

                    template_hash = gguf_aibom.get('chat_template_hash')

                    if group_key not in model_groups:
                        model_groups[group_key] = []
                    model_groups[group_key].append((gguf_filename, template_hash, gguf_aibom))

                    if primary_aibom is None:
                        primary_aibom = gguf_aibom

                except Exception as e:
                    logger.warning(f"⚠️ Failed to scan {gguf_filename}: {e}")
                    continue

            if not primary_aibom:
                logger.warning(f"⚠️ No GGUF files could be parsed in {model_id}")
                return {}

            consistency_results = []
            has_warning = False

            for (model_name, architecture), files in model_groups.items():
                hashes = {f[0]: f[1] for f in files if f[1]}
                unique_hashes = set(hashes.values())

                group_result = {
                    'model_name': model_name,
                    'architecture': architecture,
                    'files_scanned': len(hashes),
                    'consistent': len(unique_hashes) <= 1,
                }

                if len(unique_hashes) > 1:
                    has_warning = True
                    group_result['unique_hashes'] = len(unique_hashes)
                    group_result['hashes_by_file'] = hashes
                    logger.warning(f"⚠️ WARNING: Chat template mismatch in {model_name} ({architecture}):")
                    for filename, hash_val in hashes.items():
                        logger.warning(f"   {filename}: {hash_val[:50]}...")
                else:
                    logger.info(f"✅ Chat template consistent for {model_name} ({architecture}) across {len(hashes)} file(s)")

                consistency_results.append(group_result)

            primary_aibom['chat_template_consistency'] = {
                'all_consistent': not has_warning,
                'files_scanned': sum(len(files) for files in model_groups.values()),
                'model_groups_checked': len(model_groups),
                'groups': consistency_results if has_warning else None,
                'warning': 'Template mismatch detected across quantizations of same model' if has_warning else None
            }

            default_fields = {
                'model_type', 'typeOfModel', 'name', 'tokenizer_class', 'vocab_size',
                'context_length', 'gguf_filename', 'quantization',
                'chat_template_hash', 'hyperparameter', 'chat_template_consistency',
            }

            gguf_aibom = {k: v for k, v in primary_aibom.items() if k in default_fields}

            for field_name, value in gguf_aibom.items():
                if value is not None:
                    self.extraction_results[field_name] = ExtractionResult(
                        value=value,
                        source=DataSource.CONFIG_FILE,
                        confidence=ConfidenceLevel.HIGH,
                        extraction_method="gguf_binary_header"
                    )

            return gguf_aibom

        except Exception as e:
            logger.warning(f"⚠️ GGUF extraction failed for {model_id}: {e}")
            return {}

    # Legacy methods for backward compatibility
    def _layer1_structured_api(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard]) -> Dict[str, Any]:
        """Legacy Layer 1: Enhanced structured data extraction from HF API and model card."""
        logger.info("📊 Executing Legacy Layer 1: Enhanced Structured API Extraction")
        metadata = {}
        
        # Enhanced model info extraction
        if model_info:
            try:
                # Extract author with fallback logic
                author = getattr(model_info, "author", None)
                if not author or author.strip() == "":
                    parts = model_id.split("/")
                    author = parts[0] if len(parts) > 1 else "unknown"
                
                metadata['author'] = author
                metadata['name'] = getattr(model_info, "modelId", model_id).split("/")[-1]
                metadata['tags'] = getattr(model_info, "tags", [])
                metadata['pipeline_tag'] = getattr(model_info, "pipeline_tag", None)
                metadata['downloads'] = getattr(model_info, "downloads", 0)
                
                # Commit information
                commit_sha = getattr(model_info, "sha", None)
                if commit_sha:
                    metadata['commit'] = commit_sha[:7]
                
            except Exception as e:
                logger.error(f"❌ Legacy Layer 1: Error extracting from model_info: {e}")
        
        # Enhanced model card extraction
        if model_card and hasattr(model_card, "data") and model_card.data:
            try:
                card_data = model_card.data.to_dict() if hasattr(model_card.data, "to_dict") else {}
                
                metadata['license'] = card_data.get("license")
                metadata['language'] = card_data.get("language")
                metadata['library_name'] = card_data.get("library_name")
                metadata['base_model'] = card_data.get("base_model")
                metadata['datasets'] = card_data.get("datasets")
                metadata['description'] = card_data.get("model_summary") or card_data.get("description")
                
            except Exception as e:
                logger.error(f"❌ Legacy Layer 1: Error extracting from model card: {e}")
        
        # Add standard AI metadata
        metadata["primaryPurpose"] = metadata.get("pipeline_tag", "text-generation")
        metadata["suppliedBy"] = metadata.get("author", "unknown")
        metadata["typeOfModel"] = "transformer"
        
        return metadata
    
    def _layer2_repository_files(self, model_id: str) -> Dict[str, Any]:
        """Legacy Layer 2: Repository file analysis"""
        logger.info("🔧 Executing Legacy Layer 2: Repository File Analysis")
        metadata = {}
        
        try:
            config_data = self._download_and_parse_config(model_id, "config.json")
            if config_data:
                metadata['model_type'] = config_data.get("model_type")
                metadata['architectures'] = config_data.get("architectures", [])
                metadata['vocab_size'] = config_data.get("vocab_size")
            
            tokenizer_config = self._download_and_parse_config(model_id, "tokenizer_config.json")
            if tokenizer_config:
                metadata['tokenizer_class'] = tokenizer_config.get("tokenizer_class")

            # try to detect licence from repository files if licence is missing
            if "license" not in metadata or not metadata["license"]:
                detected_license = self._detect_license_from_file(model_id)
                if detected_license:
                    metadata["license"] = detected_license
            
        except Exception as e:
            logger.warning(f"⚠️ Legacy Layer 2: Could not analyze repository files: {e}")
        
        return metadata
    
    def _layer3_stp_extraction(self, model_card: Optional[ModelCard], model_id: str) -> Dict[str, Any]:
        """Legacy Layer 3: Smart Text Parsing"""
        logger.info("🔍 Executing Legacy Layer 3: Smart Text Parsing")
        metadata = {}
        
        try:
            readme_content = self._get_readme_content(model_card, model_id)
            if readme_content:
                extracted_info = self._extract_from_text(readme_content)
                metadata.update(extracted_info)

                # promote licence found in README into main metadata if no licence exists yet
                license_from_text = extracted_info.get("license_from_text")
                if license_from_text and not metadata.get("license"):
                    if isinstance(license_from_text, list):
                        metadata["license"] = license_from_text[0]
                    else:
                        metadata["license"] = license_from_text
                    
        except Exception as e:
            logger.warning(f"⚠️ Legacy Layer 3: Error in Smart Text Parsing: {e}")
        
        return metadata
    
    def _layer4_external_references(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy Layer 4: External reference generation"""
        logger.info("🔗 Executing Legacy Layer 4: External Reference Generation")
        return self._generate_external_references(model_id, metadata)
    
    def _layer5_intelligent_defaults(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy Layer 5: Intelligent default generation"""
        logger.info("🧠 Executing Legacy Layer 5: Intelligent Default Generation")
        
        if 'author' not in metadata or not metadata['author']:
            parts = model_id.split("/")
            metadata['author'] = parts[0] if len(parts) > 1 else "unknown"
        
        if 'license' not in metadata or not metadata['license']:
            metadata['license'] = "NOASSERTION"
        
        return metadata
    
    # Utility methods
    def _download_and_parse_config(self, model_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Download and parse a configuration file from the model repository"""
        try:
            file_path = hf_hub_download(repo_id=model_id, filename=filename)
            with open(file_path, 'r') as f:
                return json.load(f)
        except (RepositoryNotFoundError, EntryNotFoundError, json.JSONDecodeError) as e:
            logger.debug(f"Could not download/parse {filename}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error downloading {filename}: {e}")
            return None
    
    def _get_readme_content(self, model_card: Optional[ModelCard], model_id: str) -> Optional[str]:
        """Get README content from model card or by downloading"""
        try:
            if model_card and hasattr(model_card, 'content'):
                return model_card.content
            
            readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.debug(f"Could not get README content: {e}")
            return None
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured information from unstructured text"""
        metadata = {}
        
        # Extract license information
        license_matches = self._find_pattern_matches(text, self.patterns['license'])
        if license_matches:
            metadata['license_from_text'] = license_matches[0]
        
        # Extract dataset information
        dataset_matches = self._find_pattern_matches(text, self.patterns['datasets'])
        if dataset_matches:
            metadata['datasets_from_text'] = dataset_matches
        
        # Extract performance metrics
        metric_matches = self._extract_metrics(text)
        if metric_matches:
            metadata['performance_metrics'] = metric_matches
        
        return metadata
    
    def _find_pattern_matches(self, text: str, patterns: List[re.Pattern]) -> List[str]:
        """Find matches for a list of regex patterns in text"""
        matches = []
        for pattern in patterns:
            found = pattern.findall(text)
            matches.extend(found)
        return list(set(matches))  # Remove duplicates
    
    def _extract_metrics(self, text: str) -> Dict[str, float]:
        """Extract performance metrics from text"""
        metrics = {}
        
        metric_patterns = [
            r'accuracy[:\s]+([0-9\.]+)',
            r'f1[:\s]+([0-9\.]+)',
            r'bleu[:\s]+([0-9\.]+)',
            r'rouge[:\s]+([0-9\.]+)',
        ]
        
        for pattern_str in metric_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                metric_name = pattern_str.split('[')[0]
                try:
                    metrics[metric_name] = float(matches[0])
                except ValueError:
                    continue
        
        return metrics
    
    def _log_extraction_summary(self, model_id: str, metadata: Dict[str, Any]):
        """Log comprehensive extraction summary"""
        logger.info("=" * 60)
        logger.info(f"📋 REGISTRY-DRIVEN EXTRACTION SUMMARY FOR: {model_id}")
        logger.info("=" * 60)
        
        if self.registry_fields:
            logger.info(f"📊 Registry fields available: {len(self.registry_fields)}")
            logger.info(f"📊 Total fields extracted: {len(self.extraction_results)}")
            
            # Count fields by confidence level
            confidence_counts = {}
            source_counts = {}
            
            for field_name, result in self.extraction_results.items():
                conf = result.confidence.value
                source = result.source.value
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
            
            logger.info("📈 Confidence distribution:")
            for conf, count in confidence_counts.items():
                logger.info(f"   {conf}: {count} fields")
            
            logger.info("🔍 Source distribution:")
            for source, count in source_counts.items():
                logger.info(f"   {source}: {count} fields")
            
            # Log registry field coverage
            extracted_fields = set(self.extraction_results.keys())
            registry_field_names = set(self.registry_fields.keys())
            coverage = len(extracted_fields & registry_field_names) / len(registry_field_names) * 100
            logger.info(f"📊 Registry field coverage: {coverage:.1f}%")
            
            # Log missing registry fields
            missing_fields = registry_field_names - extracted_fields
            if missing_fields:
                logger.info(f"❌ Missing registry fields: {', '.join(sorted(missing_fields))}")
        else:
            logger.info(f"📊 Legacy extraction mode: {len(metadata)} fields extracted")
        
        logger.info("=" * 60)
    
    def get_extraction_results(self) -> Dict[str, ExtractionResult]:
        """Get detailed extraction results with provenance"""
        return self.extraction_results.copy()


# Convenience function for drop-in replacement
def extract_enhanced_metadata(model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard],
                              hf_api: Optional[HfApi] = None, template_attestation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Drop-in replacement function for _extract_structured_metadata with registry integration.

    This function automatically picks up new fields from the registry without code changes.

    Args:
        model_id: Hugging Face model identifier
        model_info: Model information from HF API
        model_card: Model card object from HF
        hf_api: Optional HuggingFace API instance
        template_attestation: Optional external security attestation for chat template

    Returns:
        Dictionary of extracted metadata
    """
    extractor = EnhancedExtractor(hf_api)
    return extractor.extract_metadata(model_id, model_info, model_card, template_attestation)


if __name__ == "__main__":
    # Test the registry-integrated enhanced extractor
    import sys
    
    if len(sys.argv) > 1:
        test_model_id = sys.argv[1]
    else:
        test_model_id = "deepseek-ai/DeepSeek-R1"
    
    print(f"Testing registry-integrated enhanced extractor with model: {test_model_id}")
    
    # Initialize HF API
    hf_api = HfApi()
    
    try:
        # Fetch model info and card
        model_info = hf_api.model_info(test_model_id)
        model_card = ModelCard.load(test_model_id)
        
        # Test extraction
        extractor = EnhancedExtractor(hf_api)
        metadata = extractor.extract_metadata(test_model_id, model_info, model_card)
        
        print(f"\nExtracted {len(metadata)} metadata fields:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\nExtraction results with provenance:")
        for field, result in extractor.get_extraction_results().items():
            print(f"  {field}: {result}")
            
    except Exception as e:
        print(f"Error testing extractor: {e}")
