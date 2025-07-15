# AI SBOM Generator API Documentation

## Overview

The AI SBOM Generator API provides a comprehensive solution for generating CycloneDX-compliant AI Software Bill of Materials (AI SBOM) for Hugging Face models. This API uses a configurable field registry system to extract and score AI SBOM fields across 5 categories, providing detailed completeness assessment and standards compliance.

---

## Table of Contents
- [Base URL](#base-url)
- [API Endpoints](#api-endpoints)
  - [API Status](#api-status)
  - [Registry Status](#registry-status)
  - [Generate AI SBOM](#generate-ai-sbom)
  - [Generate AI SBOM with Completeness Score Report](#generate-ai-sbom-with-completeness-score-report)
  - [Get Completeness Score Only](#get-completeness-score-only)
  - [Download Generated AI SBOM](#download-generated-ai-sbom)
  - [Form-Based Generation (Web UI)](#form-based-generation-web-ui)
- [Web UI](#web-ui)
- [Security Features](#security-features)
- [Field Registry System](#field-registry-system)
- [Completeness Score](#completeness-score)
- [Notes on Using the API](#notes-on-using-the-api)
- [Error Handling](#error-handling)

---

## Base URL

When deployed on Hugging Face Spaces, the base URL will be:
```
https://aetheris-ai-aibom-generator.hf.space
```

Replace this with your actual deployment URL.

---

## API Endpoints

### API Status

**Purpose**: Check if the API is operational and get version information.

**Endpoint**: `/status`

**Method**: GET

**cURL Example**:
```bash
curl -X GET "https://aetheris-ai-aibom-generator.hf.space/status"
```

**Expected Response**:
```json
{
  "status": "operational",
  "version": "1.0.0",
  "generator_version": "1.0.0"
}
```

---

### Registry Status

**Purpose**: Check the field registry configuration status and available fields.

**Endpoint**: `/api/registry/status`

**Method**: GET

**cURL Example**:
```bash
curl -X GET "https://aetheris-ai-aibom-generator.hf.space/api/registry/status"
```

**Expected Response**:
```json
{
  "registry_available": true,
  "total_fields": 29,
  "categories": [
    "required_fields",
    "metadata", 
    "component_basic",
    "component_model_card",
    "external_references"
  ],
  "field_count_by_category": {
    "required_fields": 4,
    "metadata": 5,
    "component_basic": 5,
    "component_model_card": 14,
    "external_references": 1
  },
  "registry_manager_loaded": true
}
```

---

### Generate AI SBOM

**Purpose**: Generate an AI SBOM for a specified Hugging Face model.

**Endpoint**: `/api/generate`

**Method**: POST

**Parameters**:
- `model_id` (required): The Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-chat-hf')
- `include_inference` (optional): Whether to use AI inference to enhance the AI SBOM (default: true)
- `use_best_practices` (optional): Whether to use industry best practices for scoring (default: true)
- `hf_token` (optional): Hugging Face API token for accessing private models

**cURL Example**:
```bash
curl -X POST "https://aetheris-ai-aibom-generator.hf.space/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "deepseek-ai/DeepSeek-R1",
    "include_inference": true,
    "use_best_practices": true
  }'
```

**Expected Response**: JSON containing the generated AI SBOM, model ID, timestamp, and download URL.
```json
{
  "aibom": {
    "bomFormat": "CycloneDX",
    "specVersion": "1.6",
    "serialNumber": "urn:uuid:deepseek-ai-DeepSeek-R1",
    "version": "1.0.0",
    "metadata": {
      "timestamp": "2025-07-15T18:31:18Z",
      "tools": [
        {
          "vendor": "Aetheris AI",
          "name": "AI SBOM Generator",
          "version": "1.0.0"
        }
      ],
      "properties": [
        {
          "name": "primaryPurpose",
          "value": "text-generation"
        },
        {
          "name": "suppliedBy", 
          "value": "deepseek-ai"
        }
      ]
    },
    "components": [
      {
        "type": "machine-learning-model",
        "name": "DeepSeek-R1",
        "purl": "pkg:huggingface/deepseek-ai/DeepSeek-R1",
        "description": "Advanced reasoning model with enhanced capabilities",
        "licenses": [
          {
            "license": {
              "name": "DeepSeek License"
            }
          }
        ],
        "modelCard": {
          "limitation": "Model may have limitations in certain domains"
        }
      }
    ],
    "externalReferences": [
      {
        "type": "distribution",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-R1"
      }
    ]
  },
  "model_id": "deepseek-ai/DeepSeek-R1",
  "generated_at": "2025-07-15T18:31:18Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "download_url": "/output/deepseek-ai_DeepSeek-R1_ai_sbom.json"
}
```

---

### Generate AI SBOM with Completeness Score Report

**Purpose**: Generate an AI SBOM along with a detailed completeness score report.

**Endpoint**: `/api/generate-with-report`

**Method**: POST

**Parameters**: Same as Generate AI SBOM

**cURL Example**:
```bash
curl -X POST "https://aetheris-ai-aibom-generator.hf.space/api/generate-with-report" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "deepseek-ai/DeepSeek-R1",
    "include_inference": true,
    "use_best_practices": true
  }'
```

**Expected Response**: Same as Generate AI SBOM plus completeness score details.
```json
{
  "aibom": { ... },
  "model_id": "deepseek-ai/DeepSeek-R1",
  "generated_at": "2025-07-15T18:31:18Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "download_url": "/output/deepseek-ai_DeepSeek-R1_ai_sbom.json",
  "completeness_score": {
    "total_score": 62.3,
    "section_scores": {
      "required_fields": 20.0,
      "metadata": 8.0,
      "component_basic": 20.0,
      "component_model_card": 4.3,
      "external_references": 10.0
    },
    "max_scores": {
      "required_fields": 20,
      "metadata": 20,
      "component_basic": 20,
      "component_model_card": 30,
      "external_references": 10
    },
    "field_checklist": {
      "bomFormat": "present",
      "specVersion": "present",
      "serialNumber": "present",
      "version": "present",
      "primaryPurpose": "present",
      "suppliedBy": "present",
      "standardCompliance": "missing",
      "domain": "missing",
      "autonomyType": "missing",
      "name": "present",
      "type": "present",
      "purl": "present",
      "description": "present",
      "licenses": "present",
      "energyConsumption": "missing",
      "hyperparameter": "missing",
      "limitation": "present",
      "safetyRiskAssessment": "missing",
      "typeOfModel": "present",
      "modelExplainability": "missing",
      "energyQuantity": "missing",
      "energyUnit": "missing",
      "informationAboutTraining": "missing",
      "informationAboutApplication": "missing",
      "metric": "missing",
      "metricDecisionThreshold": "missing",
      "modelDataPreprocessing": "missing",
      "useSensitivePersonalInformation": "missing",
      "downloadLocation": "present"
    },
    "category_details": {
      "required_fields": {
        "present_fields": 4,
        "total_fields": 4,
        "percentage": 100.0
      },
      "metadata": {
        "present_fields": 2,
        "total_fields": 5,
        "percentage": 40.0
      },
      "component_basic": {
        "present_fields": 5,
        "total_fields": 5,
        "percentage": 100.0
      },
      "component_model_card": {
        "present_fields": 2,
        "total_fields": 14,
        "percentage": 14.3
      },
      "external_references": {
        "present_fields": 1,
        "total_fields": 1,
        "percentage": 100.0
      }
    }
  }
}
```

---

### Get Completeness Score Only

**Purpose**: Get the completeness score for a model without generating a full AI SBOM.

**Endpoint**: `/api/models/{model_id}/score`

**Method**: GET

**Parameters**:
- `model_id` (path parameter): The Hugging Face model ID
- `use_best_practices` (query parameter, optional): Whether to use industry best practices for scoring (default: true)
- `hf_token` (query parameter, optional): Hugging Face API token for accessing private models

**cURL Example**:
```bash
curl -X GET "https://aetheris-ai-aibom-generator.hf.space/api/models/deepseek-ai/DeepSeek-R1/score?use_best_practices=true"
```

**Expected Response**:
```json
{
  "model_id": "deepseek-ai/DeepSeek-R1",
  "total_score": 62.3,
  "section_scores": {
    "required_fields": 20.0,
    "metadata": 8.0,
    "component_basic": 20.0,
    "component_model_card": 4.3,
    "external_references": 10.0
  },
  "max_scores": {
    "required_fields": 20,
    "metadata": 20,
    "component_basic": 20,
    "component_model_card": 30,
    "external_references": 10
  },
  "field_checklist": {
    "bomFormat": "present",
    "specVersion": "present",
    "name": "present",
    "downloadLocation": "present"
  },
  "generated_at": "2025-07-15T18:31:18Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### Download Generated AI SBOM

**Purpose**: Download a previously generated AI SBOM file.

**Endpoint**: `/output/{filename}`

**Method**: GET

**cURL Example**:
```bash
curl -X GET "https://aetheris-ai-aibom-generator.hf.space/output/deepseek-ai_DeepSeek-R1_ai_sbom.json" \
  -o "deepseek_r1_aibom.json"
```

---

### Form-Based Generation (Web UI)

**Purpose**: Generate AI SBOM through the web interface form submission.

**Endpoint**: `/generate`

**Method**: POST

**Content-Type**: `application/x-www-form-urlencoded`

**Parameters**:
- `model_id` (required): The Hugging Face model ID
- `g-recaptcha-response` (required): reCAPTCHA response token

**cURL Example**:
```bash
curl -X POST "https://aetheris-ai-aibom-generator.hf.space/generate" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "model_id=deepseek-ai/DeepSeek-R1&g-recaptcha-response=YOUR_RECAPTCHA_TOKEN"
```

---

## Web UI

The API also provides a user-friendly web interface accessible at the base URL. The web UI includes:

- **Model ID input field** with validation
- **reCAPTCHA protection** against automated abuse
- **Real-time generation** with progress indicators
- **Downloadable results** with completeness scoring
- **Field checklist visualization** showing extraction results
- **Category-based scoring breakdown**

---

## Security Features

### Rate Limiting
- **10 requests per minute** per IP address
- **5 concurrent requests** maximum
- **1MB request size limit**

### reCAPTCHA Protection
- **Google reCAPTCHA v2** integration for web UI
- **Automated bot detection** and prevention
- **Configurable through environment variables**

### Input Validation
- **Model ID format validation** (alphanumeric, hyphens, underscores, forward slashes)
- **XSS protection** through HTML escaping
- **SQL injection prevention** through parameterized queries

---

## Field Registry System

The AI SBOM Generator uses a configurable field registry system that enables:

### **29 Configurable Fields** across 5 categories:
- **Required Fields (4)**: bomFormat, specVersion, serialNumber, version
- **Metadata (5)**: primaryPurpose, suppliedBy, standardCompliance, domain, autonomyType  
- **Component Basic (5)**: name, type, purl, description, licenses
- **Component Model Card (14)**: energyConsumption, hyperparameter, limitation, safetyRiskAssessment, typeOfModel, modelExplainability, energyQuantity, energyUnit, informationAboutTraining, informationAboutApplication, metric, metricDecisionThreshold, modelDataPreprocessing, useSensitivePersonalInformation
- **External References (1)**: downloadLocation

### **Multi-Strategy Extraction**:
1. **HuggingFace API** → Direct metadata extraction (High confidence)
2. **Model Card** → Structured documentation parsing (Medium-high confidence)
3. **Config Files** → Technical details from JSON files (High confidence)
4. **Text Patterns** → Regex extraction from README (Medium confidence)
5. **Intelligent Inference** → Smart defaults from context (Medium confidence)
6. **Fallback Values** → Placeholders when no data available (Low confidence)

### **SPDX 3.0 Compatibility**:
- **100% field coverage** with SPDX 3.0 AI Profile specification
- **59% exact field name matches** with official SPDX 3.0 fields
- **Future dual-format support** for both CycloneDX and SPDX output
- **Current limitation** does not generate output in SPDX format

---

## Completeness Score

The completeness score is calculated using a weighted scoring system across five categories:

### **Scoring Categories**:
- **Required Fields (20%)**: Essential CycloneDX infrastructure
- **Metadata (20%)**: AI-specific metadata and provenance  
- **Component Basic (20%)**: Core component identification
- **Component Model Card (30%)**: Advanced AI model documentation
- **External References (10%)**: Distribution and reference links

### **Field Tiers**:
- **Critical (C)**: Essential fields with 3x weight multiplier
- **Important (I)**: Valuable fields with 2x weight multiplier
- **Supplementary (S)**: Additional fields with 1x weight multiplier

### **Score Interpretation**:
- **90-100**: Exceptional documentation quality
- **80-89**: Comprehensive documentation
- **70-79**: Good documentation with minor gaps
- **60-69**: Adequate documentation with some missing elements
- **50-59**: Basic documentation with significant gaps
- **Below 50**: Insufficient documentation

### **Confidence-Based Filtering**:
- Only fields extracted with **medium** or **high** confidence contribute to the score
- **Low** or **none** confidence extractions are excluded to ensure score reliability
- Individual field failures don't prevent overall SBOM generation

---

## Notes on Using the API

### **Model ID Format**
- Use the exact Hugging Face model identifier (e.g., `meta-llama/Llama-2-7b-chat-hf`)
- Model IDs are case-sensitive
- Private models require a valid `hf_token`

### **Response Times**
- **Simple models**: 5-15 seconds
- **Complex models with inference**: 30-60 seconds
- **Large models**: Up to 2 minutes

### **File Storage**
- Generated AI SBOMs are stored temporarily (7 days)
- Download URLs are valid for the file retention period
- Files are automatically cleaned up to manage storage

### **Best Practices**
- Use `use_best_practices=true` for industry-standard scoring
- Include `include_inference=true` for enhanced field extraction
- Cache results locally to avoid repeated API calls for the same model
- Use the registry status endpoint to verify system configuration

---

## Error Handling

### **Common HTTP Status Codes**:
- **200 OK**: Successful request
- **400 Bad Request**: Invalid model ID format or missing parameters
- **404 Not Found**: Model not found on Hugging Face
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server-side processing error

### **Error Response Format**:
```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2025-07-15T18:31:18Z"
}
```

### **Common Error Scenarios**:
- **Invalid Model ID**: Check format and existence on Hugging Face
- **Private Model Access**: Ensure valid `hf_token` is provided
- **Rate Limiting**: Wait before retrying or implement exponential backoff
- **Registry Unavailable**: System falls back to basic field extraction

---

## Support and Documentation

For additional support, documentation updates, or feature requests:
- **GitHub Repository**: [[Link to GitHub Isuses](https://github.com/aetheris-ai/aibom-generator/issues)]
- **API Status Page**: Use `/status` and `/api/registry/status` endpoints
- **Web Interface**: Available at the base URL for interactive testing

This API provides comprehensive AI SBOM generation capabilities with industry-leading field coverage, standards compliance, and configurable scoring systems.
