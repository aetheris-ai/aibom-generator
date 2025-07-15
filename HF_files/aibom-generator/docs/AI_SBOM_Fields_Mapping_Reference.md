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

This document provides a comprehensive mapping of all 29 fields used in the AI SBOM Generator, organized by category to match the UI structure. Each field includes its CycloneDX 1.6 location, scoring weight, tier classification, and description. SPDX 3.0 compatibility information is included for reference.

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

**Category Result:** 14/14 fields • **30.0/30 points** • **100% weight**

---

### External References Category

Links and distribution information that provide access to the model and related resources. These fields enable model discovery and access.

| # | Field Name | CycloneDX Location | SPDX 3.0 Equivalent | Tier | AS | Points | Description |
|---|------------|-------------------|---------------------|------|--------|--------|-------------|
| 29 | **downloadLocation** | `$.externalReferences[type="distribution"]` | `downloadLocation` | C | ✅ | 10.0 | Primary location to download the model |

**Category Result:** 1/1 fields • **10.0/10 points** • **100% weight**

---

## Scoring Summary

The AI SBOM Generator uses a weighted scoring system to assess documentation completeness across five categories:

| Category | Fields | Max Points | Weight | Description |
|----------|--------|------------|--------|-------------|
| **Required Fields** | 4 | 20.0 | 20% | Essential CycloneDX infrastructure |
| **Metadata** | 5 | 20.0 | 20% | AI-specific metadata and provenance |
| **Component Basic** | 5 | 20.0 | 20% | Core component identification |
| **Component Model Card** | 14 | 30.0 | 30% | Advanced AI model documentation |
| **External References** | 1 | 10.0 | 10% | Distribution and reference links |
| **TOTAL** | **29** | **100.0** | **100%** | Maximum possible completeness score |

### Tier Impact on Scoring
- **Critical fields** (C) have 3x weight multiplier and significantly impact scoring
- **Important fields** (I) have 2x weight multiplier and enhance documentation quality  
- **Supplementary fields** (S) have 1x weight multiplier and provide additional context

---

## Field Extraction Strategies

The AI SBOM Generator employs a multi-strategy extraction approach for each field, attempting extraction in the following priority order:

1. **HuggingFace API** → Direct metadata extraction (High confidence)
2. **Model Card** → Structured documentation parsing (Medium-high confidence)  
3. **Config Files** → Technical details from JSON files (High confidence)
4. **Text Patterns** → Regex extraction from README (Medium confidence)
5. **Intelligent Inference** → Smart defaults from context (Medium confidence)
6. **Fallback Values** → Placeholders when no data available (Low/no confidence)

This multi-strategy approach ensures maximum field coverage while maintaining confidence scoring for each extracted value.

---

## Standards Compatibility

### CycloneDX 1.6 (Primary Format)
- **Primary structure** follows CycloneDX 1.6 specification
- **Model Card extension** provides AI-specific documentation
- **Properties mechanism** allows flexible field addition
- **JSON Schema validation** ensures structural compliance

### SPDX 3.0 AI Profile (Reference Compatibility)
- **100% field coverage** with official SPDX 3.0 AI Profile specification
- **17/29 fields (59%)** have exact field name matches
- **Compatible data types** aligned with SPDX type system
- **Future dual-format support** enables SPDX 3.0 output

### Interoperability
- **Standards-compliant output** can be converted between formats
- **AI field preservation** maintains semantic meaning across standards
- **Tool compatibility** with both CycloneDX and SPDX ecosystems

---

## Usage Notes

### Configuration and Customization
- **Registry-driven extraction**: All fields are configurable via JSON registry
- **Scoring weights**: Adjustable per field and category
- **Tier assignments**: Customizable based on use case requirements
- **Extraction strategies**: Configurable priority and methods

### Field Addition and Modification
- **New fields**: Can be added to registry without code changes
- **Weight adjustments**: Modify scoring impact through configuration
- **Category organization**: Fields can be reorganized by category
- **Validation rules**: Configurable per field

### Performance Characteristics
- **Automatic field discovery**: System attempts extraction for all registry fields
- **Graceful degradation**: Individual field failures don't stop overall extraction
- **Confidence scoring**: Each field extraction includes confidence assessment
- **Comprehensive logging**: Detailed extraction results for debugging

This comprehensive field mapping serves as the definitive reference for the AI SBOM Generator's field extraction, scoring, and documentation capabilities, with full standards compatibility for future interoperability.
