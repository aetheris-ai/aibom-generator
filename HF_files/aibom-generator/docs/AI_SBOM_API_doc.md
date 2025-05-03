# AI SBOM Generator API Documentation

## Overview

The AI SBOM Generator API provides a comprehensive solution for generating CycloneDX-compliant AI Bill of Materials (AIBOM) for Hugging Face models. This document outlines the available API endpoints, their functionality, and how to interact with them using cURL commands.

## Base URL

When deployed on Hugging Face Spaces, the base URL will be:
```
https://aetheris-ai-aibom-generator.hf.space
```

Replace this with your actual deployment URL.

## API Endpoints

### Status Endpoint

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

### Generate AIBOM Endpoint

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
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "include_inference": true,
    "use_best_practices": true
  }'
```

**Expected Response**: JSON containing the generated AIBOM, model ID, timestamp, and download URL.
```json
{
  "aibom": {
    "bomFormat": "CycloneDX",
    "specVersion": "1.6",
    "serialNumber": "urn:uuid:...",
    "version": 1,
    "metadata": { ... },
    "components": [ ... ],
    "dependencies": [ ... ]
  },
  "model_id": "meta-llama/Llama-2-7b-chat-hf",
  "generated_at": "2025-04-24T20:30:00Z",
  "request_id": "...",
  "download_url": "/output/meta-llama_Llama-2-7b-chat-hf_....json"
}
```

### Generate AIBOM with Enhancement Report

**Purpose**: Generate an AI SBOM with a detailed enhancement report.

**Endpoint**: `/api/generate-with-report`

**Method**: POST

**Parameters**: Same as `/api/generate`

**cURL Example**:
```bash
curl -X POST "https://aetheris-ai-aibom-generator.hf.space/api/generate-with-report" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "include_inference": true,
    "use_best_practices": true
  }'
```

**Expected Response**: JSON containing the generated AIBOM, model ID, timestamp, download URL, and enhancement report.
```json
{
  "aibom": { ... },
  "model_id": "meta-llama/Llama-2-7b-chat-hf",
  "generated_at": "2025-04-24T20:30:00Z",
  "request_id": "...",
  "download_url": "/output/meta-llama_Llama-2-7b-chat-hf_....json",
  "enhancement_report": {
    "ai_enhanced": true,
    "ai_model": "BERT-base-uncased",
    "original_score": {
      "total_score": 65.5,
      "completeness_score": 65.5
    },
    "final_score": {
      "total_score": 85.2,
      "completeness_score": 85.2
    },
    "improvement": 19.7
  }
}
```

### Get Model Score

**Purpose**: Get the completeness score for a model without generating a full AIBOM.

**Endpoint**: `/api/models/{model_id}/score`

**Method**: GET

**Parameters**:
- `model_id` (path parameter): The Hugging Face model ID
- `hf_token` (query parameter, optional): Hugging Face API token for accessing private models
- `use_best_practices` (query parameter, optional): Whether to use industry best practices for scoring (default: true)

**cURL Example**:
```bash
curl -X GET "https://aetheris-ai-aibom-generator.hf.space/api/models/meta-llama/Llama-2-7b-chat-hf/score?use_best_practices=true"
```

**Expected Response**: JSON containing the completeness score information.
```json
{
  "total_score": 85.2,
  "section_scores": {
    "required_fields": 20,
    "metadata": 18.5,
    "component_basic": 20,
    "component_model_card": 20.7,
    "external_references": 6
  },
  "max_scores": {
    "required_fields": 20,
    "metadata": 20,
    "component_basic": 20,
    "component_model_card": 30,
    "external_references": 10
  }
}
```

### Batch Generate AIBOMs

**Purpose**: Start a batch job to generate AIBOMs for multiple models.

**Endpoint**: `/api/batch`

**Method**: POST

**Parameters**:
- `model_ids` (required): List of Hugging Face model IDs to generate AIBOMs for
- `include_inference` (optional): Whether to use AI inference to enhance the AI SBOM (default: true)
- `use_best_practices` (optional): Whether to use industry best practices for scoring (default: true)
- `hf_token` (optional): Hugging Face API token for accessing private models

**cURL Example**:
```bash
curl -X POST "https://aetheris-ai-aibom-generator.hf.space/api/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ids": [
      "meta-llama/Llama-2-7b-chat-hf",
      "google/flan-t5-base"
    ],
    "include_inference": true,
    "use_best_practices": true
  }'
```

**Expected Response**: JSON containing the batch job ID and status information.
```json
{
  "job_id": "...",
  "status": "queued",
  "model_ids": [
    "meta-llama/Llama-2-7b-chat-hf",
    "google/flan-t5-base"
  ],
  "created_at": "2025-04-24T20:30:00Z"
}
```

### Check Batch Job Status

**Purpose**: Check the status of a batch job.

**Endpoint**: `/api/batch/{job_id}`

**Method**: GET

**Parameters**:
- `job_id` (path parameter): The ID of the batch job to check

**cURL Example**:
```bash
curl -X GET "https://aetheris-ai-aibom-generator.hf.space/api/batch/{job_id}"
```

**Expected Response**: JSON containing the current status of the batch job and results for completed models.
```json
{
  "job_id": "...",
  "status": "processing",
  "model_ids": [
    "meta-llama/Llama-2-7b-chat-hf",
    "google/flan-t5-base"
  ],
  "created_at": "2025-04-24T20:30:00Z",
  "completed": 1,
  "total": 2,
  "results": {
    "meta-llama/Llama-2-7b-chat-hf": {
      "status": "completed",
      "download_url": "/output/meta-llama_Llama-2-7b-chat-hf_....json",
      "enhancement_report": { ... }
    }
  }
}
```

### Download Generated AIBOM

**Purpose**: Download a previously generated AIBOM file.

**Endpoint**: `/download/{filename}`

**Method**: GET

**Parameters**:
- `filename` (path parameter): The filename of the AIBOM to download

**cURL Example**:
```bash
curl -X GET "https://aetheris-ai-aibom-generator.hf.space/download/{filename}" \
  -o "downloaded_aibom.json"
```

**Expected Response**: The AIBOM JSON file will be downloaded to your local machine.

### Form-Based Generation (Web UI)

**Purpose**: Generate an AIBOM using form data (typically used by the web UI).

**Endpoint**: `/generate`

**Method**: POST

**Parameters**:
- `model_id` (form field, required): The Hugging Face model ID
- `include_inference` (form field, optional): Whether to use AI inference to enhance the AI SBOM
- `use_best_practices` (form field, optional): Whether to use industry best practices for scoring

**cURL Example**:
```bash
curl -X POST "https://aetheris-ai-aibom-generator.hf.space/generate" \
  -F "model_id=meta-llama/Llama-2-7b-chat-hf" \
  -F "include_inference=true" \
  -F "use_best_practices=true"
```

**Expected Response**: HTML page with the generated AIBOM results.

## Web UI

The API also provides a web user interface for generating AIBOMs without writing code:

**URL**: `https://aetheris-ai-aibom-generator.hf.space/`

The web UI allows you to:
1. Enter a Hugging Face model ID
2. Configure generation options
3. Generate an AIBOM
4. View the results in a human-friendly format
5. Download the generated AIBOM as a JSON file

## Understanding the Field Checklist

In the Field Checklist tab of the results page, you'll see a list of fields with check marks (✔/✘) and stars (★). Here's what they mean:

- **Check marks**:
  - ✔: Field is present in the AIBOM
  - ✘: Field is missing from the AIBOM

- **Stars** (importance level):
  - ★★★ (three stars): Critical fields - Essential for a valid and complete AIBOM
  - ★★ (two stars): Important fields - Valuable information that enhances completeness
  - ★ (one star): Supplementary fields - Additional context and details (optional)

## Notes on Using the API

1. When deployed on Hugging Face Spaces, use the correct URL format as shown in the examples.
2. Some endpoints may have rate limiting or require authentication.
3. For large responses, consider adding appropriate timeout settings in your requests.
4. If you encounter CORS issues, you may need to add appropriate headers.
5. For downloading files, specify the output file name in your client code.

## Error Handling

The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (resource not found)
- 500: Internal Server Error (server-side error)

Error responses include a detail message explaining the error:
```json
{
  "detail": "Error generating AI SBOM: Model not found"
}
```

## Completeness Score

The completeness score is calculated based on the presence and quality of various fields in the AIBOM. The score is broken down into sections:

1. **Required Fields** (20 points): Basic required fields for a valid AIBOM
2. **Metadata** (20 points): Information about the AIBOM itself
3. **Component Basic Info** (20 points): Basic information about the AI model component
4. **Model Card** (30 points): Detailed model card information
5. **External References** (10 points): Links to external resources

The total score is a weighted sum of these section scores, with a maximum of 100 points.

## Enhancement Report

When AI enhancement is enabled, the API uses an inference model to extract additional information from the model card and other sources. The enhancement report shows:

1. **Original Score**: The completeness score before enhancement
2. **Enhanced Score**: The completeness score after enhancement
3. **Improvement**: The point increase from enhancement
4. **AI Model Used**: The model used for enhancement

This helps you understand how much the AI enhancement improved the AIBOM's completeness.
