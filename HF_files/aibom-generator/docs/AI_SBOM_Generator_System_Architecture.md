# AI SBOM Generator System Architecture

## Overview

The AI SBOM Generator is a configurable system that automatically generates Software Bill of Materials (SBOM) documents for AI models hosted on HuggingFace. The system uses a registry-driven architecture that allows for dynamic field configuration without code changes.

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    AI SBOM Generator                   │
├─────────────────────────────────────────────────────────────┤
│  Web Interface (FastAPI + HTML Templates)              │
├─────────────────────────────────────────────────────────────┤
│  API Layer                                             │
│  ├── Generation Endpoints                              │
│  ├── Scoring Endpoints                                 │
│  └── Batch Processing                                  │
├─────────────────────────────────────────────────────────────┤
│  Core Generation Engine                                │
│  ├── AIBOMGenerator (generator.py)                     │
│  ├── Enhanced Extractor (enhanced_extractor.py)        │
│  └── Field Registry Manager (field_registry_manager.py)│
├─────────────────────────────────────────────────────────────┤
│  Configuration Layer                                   │
│  ├── Field Registry (field_registry.json)              │
│  ├── Scoring Configuration                             │
│  └── AIBOM Generation Rules                            │
├─────────────────────────────────────────────────────────────┤
│  Data Sources                                          │
│  ├── HuggingFace API                                   │
│  ├── Model Cards                                       │
│  ├── Configuration Files                               │
│  └── README Content                                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Registry-Driven Configuration**: All fields and scoring rules defined in JSON
- **Multi-Strategy Extraction**: 6 different extraction methods per field
- **Standards Compliance**: CycloneDX 1.6 compatible output
- **Configurable Scoring**: Weighted scoring system with tier-based multipliers
- **Automatic Field Discovery**: New fields added to registry are automatically processed
- **Comprehensive Logging**: Detailed extraction and scoring logs for debugging

## Process Workflow

### 1. System Initialization

```
System Initialization Process:

    ┌───────────────────┐
    │  System Startup  │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │ Load Field       │
    │ Registry         │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │ Initialize       │
    │ Registry Manager │
    └─────────┬─────────┘
              │
              ▼
    ┌─────────────────┐
    │ Load Scoring   │
    │ Configuration  │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ Initialize     │
    │ Enhanced       │
    │ Extractor      │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │  System Ready  │
    └─────────────────┘
```

**Steps:**
1. **Load Field Registry**: Read `field_registry.json` containing all field definitions
2. **Initialize Registry Manager**: Create manager instance with loaded configuration
3. **Load Scoring Configuration**: Parse scoring weights, tiers, and category definitions
4. **Initialize Enhanced Extractor**: Create extractor with registry-driven field discovery
5. **System Ready**: All components initialized and ready for SBOM generation

### 2. SBOM Generation Process

```
SBOM Generation Workflow:

User Request ──┐
               │
               ▼
    ┌───────────────────┐      ┌────────────────────┐     ┌──────────────────┐
    │ Validate Model  │─────▶│ Fetch Model Info  │───▶│ Initialize      │
    │ ID              │      │                   │    │ Enhanced        │
    └───────────────────┘      └────────────────────┘    │ Extractor       │
                                                      └──────────┬───────┘
                                                                │
    ┌───────────────────┐     ┌──────────────────┐                 │
    │ Return SBOM +    │◀───│ Calculate       │◀────────────────┘
    │ Score            │    │ Completeness    │
    └───────────────────┘     │ Score           │
                            └──────────────────┘
                                    ▲
                                    │
                           ┌────────────────────┐
                           │ Generate AIBOM    │
                           │ Structure         │
                           └────────────────────┘
                                    ▲
                                    │
                           ┌────────────────────┐
                           │ Multi-Strategy    │
                           │ Field Processing  │
                           └────────────────────┘
                                    ▲
                                    │
                           ┌────────────────────┐
                           │ Registry-Driven   │
                           │ Extraction        │
                           └────────────────────┘
```

#### 2.1 Model Information Gathering

**Input**: HuggingFace model ID (e.g., `deepseek-ai/DeepSeek-R1`)

**Process**:
1. **Validate Model ID**: Check format and accessibility
2. **Fetch Model Info**: Retrieve metadata from HuggingFace API
3. **Download Model Card**: Get structured model documentation
4. **Fetch Configuration Files**: Download `config.json`, `tokenizer_config.json`
5. **Extract README Content**: Parse model description and documentation

#### 2.2 Registry-Driven Field Extraction

**For each of the 29 registry fields:**

```
Multi-Strategy Field Extraction:

Field from Registry
        │
        ▼
┌───────────────────┐     Success?
│ Strategy 1:      │────────┐
│ HuggingFace API  │        │
└───────────────────┘        │
        │                  │
        │ Failure          │
        ▼                  │
┌───────────────────┐        │
│ Strategy 2:      │        │
│ Model Card       │        │
└───────────────────┘        │
        │                  │
        │ Failure          │
        ▼                  │
┌───────────────────┐        │
│ Strategy 3:      │        │
│ Config Files     │        │
└───────────────────┘        │
        │                  │
        │ Failure          │
        ▼                  │
┌───────────────────┐        │
│ Strategy 4:      │        │
│ Text Patterns    │        │
└───────────────────┘        │
        │                  │
        │ Failure          │
        ▼                  │
┌───────────────────┐        │
│ Strategy 5:      │        │
│ Intelligent      │        │
│ Inference        │        │
└───────────────────┘        │
        │                  │
        │ Failure          │
        ▼                  │
┌───────────────────┐        │
│ Strategy 6:      │        │
│ Fallback Value   │        │
└───────────────────┘        │
        │                  │
        ▼                  │
┌───────────────────┐◀───────┘
│ Store Result &   │
│ Log Outcome      │
└───────────────────┘
```

**Extraction Strategies**:

1. **HuggingFace API Extraction**
   - Direct field mapping from API response
   - High confidence, structured data
   - Fields: `name`, `author`, `license`, `tags`, etc.

2. **Model Card Extraction**
   - Parse structured model card YAML/metadata
   - Medium-high confidence
   - Fields: `limitation`, `metrics`, `datasets`, etc.

3. **Configuration File Extraction**
   - Mine technical details from config files
   - High confidence for technical fields
   - Fields: `typeOfModel`, `hyperparameter`, etc.

4. **Text Pattern Extraction**
   - Regex-based extraction from README content
   - Medium confidence, requires validation
   - Fields: `safetyRiskAssessment`, `informationAboutTraining`, etc.

5. **Intelligent Inference**
   - Smart defaults based on model characteristics
   - Medium confidence, contextual
   - Fields: `primaryPurpose`, `domain`, etc.

6. **Fallback Values**
   - Placeholder values when no data available
   - Low/no confidence, maintains structure
   - Ensures complete SBOM structure

#### 2.3 AIBOM Structure Generation

**Process**:
1. **Create Base Structure**: Initialize CycloneDX 1.6 compliant structure
2. **Populate Metadata Section**: Add extracted metadata fields
3. **Build Component Section**: Create model component with extracted data
4. **Add Model Card**: Include AI-specific model card information
5. **Generate External References**: Add distribution and repository links
6. **Create Dependencies**: Define model dependencies and relationships
7. **Validate Structure**: Ensure CycloneDX compliance

**Output Structure**:
```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.6",
  "serialNumber": "urn:uuid:...",
  "version": 1,
  "metadata": {
    "timestamp": "...",
    "tools": [...],
    "component": {...},
    "properties": [...]
  },
  "components": [{
    "type": "machine-learning-model",
    "name": "...",
    "modelCard": {...},
    "properties": [...]
  }],
  "externalReferences": [...],
  "dependencies": [...]
}
```

### 3. Completeness Scoring Process

```
Completeness Scoring Process:

┌───────────────────┐
│ Extracted Fields │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Categorize       │
│ Fields           │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Apply Tier       │
│ Weights          │
│ • Critical: 3x   │
│ • Important: 2x  │
│ • Supplement: 1x │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Calculate        │
│ Category Scores  │
│ • Required: 20   │
│ • Metadata: 20   │
│ • Basic: 20      │
│ • ModelCard: 30  │
│ • ExtRefs: 10    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Sum Weighted     │
│ Scores           │
│ (Max: 100)       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Generate Score   │
│ Report           │
└───────────────────┘
```

**Scoring Algorithm**:

1. **Field Categorization**: Group fields by category (required_fields, metadata, etc.)
2. **Tier Weight Application**: Apply multipliers (Critical: 3x, Important: 2x, Supplementary: 1x)
3. **Category Score Calculation**: `(Fields Present / Total Fields) × Category Weight`
4. **Final Score**: Sum of all category scores (max 100)

**Category Weights**:
- Required Fields: 20 points
- Metadata: 20 points  
- Component Basic: 20 points
- Component Model Card: 30 points
- External References: 10 points

### 4. Output Generation

**Generated Artifacts**:
1. **AIBOM JSON**: CycloneDX 1.6 compliant SBOM document
2. **Completeness Score**: Numerical score (0-100) with breakdown
3. **Field Checklist**: Detailed field-by-field analysis
4. **Extraction Report**: Confidence levels and data sources
5. **Validation Results**: Compliance and quality checks

## Configuration Management

### Field Registry Structure

The system is driven by `field_registry.json` which defines:

- **Field Definitions**: All 29 extractable fields
- **Scoring Configuration**: Weights, tiers, and categories
- **AIBOM Generation Rules**: Structure and validation rules
- **Extraction Strategies**: How each field should be extracted

### Dynamic Configuration

**Adding New Fields**:
1. Add field definition to `field_registry.json`
2. System automatically discovers and attempts extraction
3. No code changes required

**Updating Scoring**:
1. Modify weights in registry configuration
2. Changes take effect immediately
3. Consistent scoring across all models

## Quality Assurance

### Validation Layers

1. **Input Validation**: Model ID format and accessibility
2. **Extraction Validation**: Data type and format checking
3. **Structure Validation**: CycloneDX schema compliance
4. **Scoring Validation**: Mathematical correctness
5. **Output Validation**: JSON schema and completeness

### Error Handling

- **Individual Field Failures**: Don't stop overall processing
- **Graceful Degradation**: Fallback to lower-confidence strategies
- **Comprehensive Logging**: Detailed error tracking and debugging
- **Recovery Mechanisms**: Automatic retry and alternative approaches

## Performance Characteristics

### Typical Processing Times

- **Single Model**: 2-5 seconds
- **Batch Processing**: 10-50 models/minute
- **Registry Loading**: <1 second
- **Field Extraction**: 1-3 seconds per model

### Scalability Features

- **Concurrent Processing**: Multiple models processed simultaneously
- **Caching**: Model metadata and configuration caching
- **Rate Limiting**: Respectful API usage
- **Resource Management**: Memory and connection pooling

## Integration Points

### APIs

- **Generation API**: `/api/generate` - Single model AI SBOM generation, with download URL
- **Generation with Completness Score Report API**: `/api/generate-with-report` - Generation API with completness scoring report
- **Completness Score Report Only API**: `/api/models/{model_id}/score` - Get the completeness score for a model without generating AI SBOM

### Data Sources

- **HuggingFace Hub**: Primary model metadata source
- **Model Repositories**: Direct file access for configurations
- **Model Cards**: Structured documentation parsing

### Output Formats

- **CycloneDX JSON**: Primary SBOM format
- **Field Reports**: Human-readable analysis
- **CSV Exports**: Batch processing results
- **API Responses**: Structured JSON for integration

This architecture provides a robust, configurable, and standards-compliant solution for AI model SBOM generation with comprehensive field extraction and scoring capabilities.

