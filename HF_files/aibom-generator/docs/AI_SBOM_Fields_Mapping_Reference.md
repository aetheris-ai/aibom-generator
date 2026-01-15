# AI SBOM Fields Mapping Reference

## Table of Contents

- [Overview](#overview)
- [Legend](#legend)
- [Field Categories](#field-categories)
  - [Required Fields Category](#required-fields-category)
  - [Metadata Category](#metadata-category)
  - [Component Basic Category](#component-basic-category)
  - [Component Model Card Category](#component-model-card-category)
  - [External References Category](#external-references-category)
- [Scoring Summary](#scoring-summary)
- [Field Extraction Strategies](#field-extraction-strategies)
- [Standards Compatibility](#standards-compatibility)
- [Usage Notes](#usage-notes)

---

## Overview

This document provides a comprehensive mapping of all 35 fields used in the AI SBOM Generator, organized by category to match the UI structure. Each field includes its CycloneDX 1.6 location, scoring weight, tier classification, and description. SPDX 3.0 compatibility information is included for reference.

The AI SBOM Generator uses a configurable field registry to extract, validate, and score AI model documentation across multiple sources, providing comprehensive Bill of Materials for AI systems.

---

## Legend

### Tiers
- **C**: Critical - Essential fields (weight: 3x, 4.0-10.0 points)
- **I**: Important - Valuable fields (weight: 2x, 2.0-3.0 points)
- **S**: Supplementary - Additional fields (weight: 1x, 1.0-2.0 points)

### SPDX 3.0 Alignment Status (AS)
- **🎯**: Exact Match - Matched field name and type
- **✅**: Standard Field - Core SPDX compatibility
- **🔄**: Semantic Match - Same concept, different name

---

## Field Categories

### Required Fields Category

Essential CycloneDX infrastructure fields that form the foundation of every SBOM document. These fields are mandatory for proper SBOM identification and compliance.

| # | Field Name | CycloneDX Location | SPDX 3.0 Equivalent | Tier | AS | Points | Description |
|---|------------|-------------------|---------------------|------|--------|--------|-------------|
| 1 | **bomFormat** | `$.bomFormat` | Core SPDX field | C | ✅ | 4.0 | Format identifier for the SBOM (always "CycloneDX") |
| 2 | **specVersion** | `$.specVersion` | `spdxVersion` | C | ✅ | 4.0 | CycloneDX specification version (e.g., "1.6") |
| 3 | **serialNumber** | `$.serialNumber` | `spdxId` | C | ✅ | 4.0 | Unique identifier for this SBOM instance |
| 4 | **version** | `$.version` | `releaseTime` | C | ✅ | 4.0 | Version of this SBOM document |

**Category Result:** 4/4 fields • **20.0/20 points** • **100% weight**

---

### Metadata Category

AI-specific metadata and provenance information that provides context about the model's purpose, supply chain, and compliance. These fields help establish the model's intended use and regulatory context.

| # | Field Name | CycloneDX Location | SPDX 3.0 Equivalent | Tier | AS | Points | Description |
|---|------------|-------------------|---------------------|------|--------|--------|-------------|
| 5 | **primaryPurpose** | `$.metadata.properties[name="primaryPurpose"]` | `ai_intendedUse` | C | 🔄 | 4.0 | Primary intended use of the AI model |
| 6 | **suppliedBy** | `$.metadata.properties[name="suppliedBy"]` | `supplier` | C | ✅ | 4.0 | Organization or individual who supplied the model |
| 7 | **standardCompliance** | `$.metadata.properties[name="standardCompliance"]` | `ai_standardCompliance` | S | 🎯 | 1.0 | Compliance with AI/ML standards and regulations |
| 8 | **domain** | `$.metadata.properties[name="domain"]` | `ai_domain` | S | 🎯 | 1.0 | Application domain or industry vertical |
| 9 | **autonomyType** | `$.metadata.properties[name="autonomyType"]` | `ai_autonomyType` | S | 🎯 | 1.0 | Level of autonomy in decision-making |

**Category Result:** 5/5 fields • **20.0/20 points** • **100% weight**

---

### Component Basic Category

Core component identification and description fields that define the essential characteristics of the AI model. These fields provide fundamental information needed for model identification and basic documentation.

| # | Field Name | CycloneDX Location | SPDX 3.0 Equivalent | Tier | AS | Points | Description |
|---|------------|-------------------|---------------------|------|--------|--------|-------------|
| 10 | **name** | `$.components[0].name` | `name` | C | ✅ | 4.0 | Human-readable name of the model |
| 11 | **type** | `$.components[0].type` | `ai_AIPackage` type | I | ✅ | 2.0 | Component type (always "machine-learning-model") |
| 12 | **purl** | `$.components[0].purl` | `externalRefs[type="purl"]` | I | ✅ | 2.0 | Package URL for unique identification |
| 13 | **description** | `$.components[0].description` | `summary` | I | ✅ | 2.0 | Brief description of the model's purpose |
| 14 | **licenses** | `$.components[0].licenses` | `licenseConcluded` | I | ✅ | 2.0 | License information for the model |

**Category Result:** 5/5 fields • **20.0/20 points** • **100% weight**

---

### Component Model Card Category

Advanced AI model documentation fields that provide detailed information about model characteristics, training, performance, and usage considerations. This category represents the most comprehensive AI-specific documentation.

| # | Field Name | CycloneDX Location | SPDX 3.0 Equivalent | Tier | AS | Points | Description |
|---|------------|-------------------|---------------------|------|--------|--------|-------------|
| 15 | **energyConsumption** | `$.components[0].modelCard.properties[name="energyConsumption"]` | `ai_energyConsumption` | I | 🎯 | 2.0 | Energy consumption information |
| 16 | **hyperparameter** | `$.components[0].modelCard.properties[name="hyperparameter"]` | `ai_hyperparameter` | I | 🎯 | 2.0 | Key hyperparameters used in training |
| 17 | **limitation** | `$.components[0].modelCard.limitation` | `ai_limitation` | I | 🎯 | 2.0 | Known limitations and constraints |
| 18 | **safetyRiskAssessment** | `$.components[0].modelCard.properties[name="safetyRiskAssessment"]` | `ai_safetyRiskAssessment` | I | 🎯 | 2.0 | Safety and risk assessment information |
| 19 | **typeOfModel** | `$.metadata.properties[name="typeOfModel"]` | `ai_typeOfModel` | I | 🎯 | 2.0 | Technical classification of the model type |
| 20 | **modelExplainability** | `$.components[0].modelCard.properties[name="modelExplainability"]` | `ai_modelExplainability` | S | 🎯 | 1.0 | Information about model interpretability |
| 21 | **energyQuantity** | `$.components[0].modelCard.properties[name="energyQuantity"]` | `ai_energyQuantity` | S | 🎯 | 1.0 | Quantitative energy consumption metrics |
| 22 | **energyUnit** | `$.components[0].modelCard.properties[name="energyUnit"]` | `ai_energyUnit` | S | 🎯 | 1.0 | Units for energy consumption measurements |
| 23 | **informationAboutTraining** | `$.components[0].modelCard.properties[name="informationAboutTraining"]` | `ai_informationAboutTraining` | S | 🎯 | 1.0 | Details about the training process |
| 24 | **informationAboutApplication** | `$.components[0].modelCard.properties[name="informationAboutApplication"]` | `ai_informationAboutApplication` | S | 🎯 | 1.0 | Information about intended applications |
| 25 | **metric** | `$.components[0].modelCard.properties[name="metric"]` | `ai_metric` | S | 🎯 | 1.0 | Performance metrics and evaluation results |
| 26 | **metricDecisionThreshold** | `$.components[0].modelCard.properties[name="metricDecisionThreshold"]` | `ai_metricDecisionThreshold` | S | 🎯 | 1.0 | Decision thresholds for model outputs |
| 27 | **modelDataPreprocessing** | `$.components[0].modelCard.properties[name="modelDataPreprocessing"]` | `ai_modelDataPreprocessing` | S | 🎯 | 1.0 | Data preprocessing and preparation steps |
| 28 | **useSensitivePersonalInformation** | `$.components[0].modelCard.properties[name="useSensitivePersonalInformation"]` | `ai_useSensitivePersonalInformation` | S | 🎯 | 1.0 | Information about sensitive data usage |
| 29 | **chat_template** | `$.components[0].modelCard.properties[name="chat_template"]` | - | - | 🔄 | - | Full Jinja template string (opt-in content, not scored - hash is sufficient) |
| 30 | **chat_template_hash** | `$.components[0].modelCard.properties[name="chat_template_hash"]` | - | I | 🔄 | 2.5 | SHA-256 hash of chat template for integrity verification |
| 31 | **template_source** | `$.components[0].modelCard.properties[name="template_source"]` | - | S | 🔄 | 1.0 | Where/when/how the template was extracted |
| 32 | **model_lineage** | `$.components[0].modelCard.properties[name="model_lineage"]` | - | S | 🔄 | 1.0 | Template inheritance and derivation tracking |
| 33 | **template_security_status** | `$.components[0].modelCard.properties[name="template_security_status"]` | - | I | 🔄 | 2.0 | Security attestation with scanner details and findings |
| 34 | **named_chat_templates** | `$.components[0].modelCard.properties[name="named_chat_templates"]` | - | S | 🔄 | 1.0 | Hashes for named templates (tool_use, rag, etc.) |

**Category Result:** 19/20 fields scored • **37.5/37.5 points** • **100% weight**

---

### External References Category

Links and distribution information that provide access to the model and related resources. These fields enable model discovery and access.

| # | Field Name | CycloneDX Location | SPDX 3.0 Equivalent | Tier | AS | Points | Description |
|---|------------|-------------------|---------------------|------|--------|--------|-------------|
| 35 | **downloadLocation** | `$.externalReferences[type="distribution"]` | `downloadLocation` | C | ✅ | 10.0 | Primary location to download the model |

**Category Result:** 1/1 fields • **10.0/10 points** • **100% weight**

---

## Scoring Summary

The AI SBOM Generator uses a weighted scoring system to assess documentation completeness across five categories:

| Category | Fields | Max Points | Weight | Description |
|----------|--------|------------|--------|-------------|
| **Required Fields** | 4 | 20.0 | 19% | Essential CycloneDX infrastructure |
| **Metadata** | 5 | 20.0 | 19% | AI-specific metadata and provenance |
| **Component Basic** | 5 | 20.0 | 19% | Core component identification |
| **Component Model Card** | 19 | 37.5 | 35% | Advanced AI model documentation |
| **External References** | 1 | 10.0 | 9% | Distribution and reference links |
| **TOTAL** | **34** | **107.5** | **100%** | Maximum possible completeness score |

### Tier Impact on Scoring
- **Critical fields** (C) have 3x weight multiplier and significantly impact scoring
- **Important fields** (I) have 2x weight multiplier and enhance documentation quality
- **Supplementary fields** (S) have 1x weight multiplier and provide additional context

---

## Chat Template Integrity Fields

Chat templates are Jinja2 templates that format conversations before inference. Poisoned templates can influence model completions while bypassing weight-based scanning, making them a security-relevant attack surface.

Fields #29-34 track chat template integrity:
- **chat_template_hash**: SHA-256 hash (`sha256:...`) for integrity verification
- **template_source**: Source file, repository, timestamp, and extractor tool
- **template_security_status**: Security attestation (defaults to `"unscanned"`; external scanners can provide attestations via `--template-attestation`)
- **named_chat_templates**: Hash map for models with multiple templates (e.g., `tool_use`, `rag`)

For GGUF files, templates are extracted via header-only HTTP range requests without downloading full weights.

---

## Field Extraction Strategies

Extraction is attempted in priority order:

1. **HuggingFace API** (High confidence)
2. **Model Card YAML** (High confidence)
3. **Config Files** (High confidence)
4. **GGUF Metadata** (High confidence)
5. **Text Patterns** (Medium confidence)
6. **Intelligent Inference** (Medium confidence)
7. **Fallback Values** (Low confidence)

---

## Standards Compatibility

- **CycloneDX 1.6**: Primary output format with Model Card extension for AI fields
- **SPDX 3.0 AI Profile**: 17/35 fields have exact name matches; all fields semantically compatible

---

## Usage Notes

- **Registry-driven**: All fields configurable via `field_registry.json` without code changes
- **Graceful degradation**: Individual field failures don't stop extraction
- **Confidence scoring**: Each extracted value includes confidence assessment
