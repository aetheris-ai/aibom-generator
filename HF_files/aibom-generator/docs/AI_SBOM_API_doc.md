# AI SBOM Generator API Documentation

## Overview

The AI SBOM Generator API provides a comprehensive solution for generating CycloneDX-compliant AI Bill of Materials (AI SBOM) for Hugging Face models. This document outlines the available API endpoints, their functionality, and how to interact with them using cURL commands.

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

### Generate AI SBOM Endpoint

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

**Expected Response**: JSON containing the generated AI SBOM, model ID, timestamp, and download URL.
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

### Generate AI SBOM with Enhancement Report

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

**Expected Response**: JSON containing the generated AI SBOM, model ID, timestamp, download URL, and enhancement report.
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

**Purpose**: Get the completeness score for a model without generating a full AI SBOM.

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

### Download Generated AI SBOM

**Purpose**: Download a previously generated AI SBOM file.

**Endpoint**: `/download/{filename}`

**Method**: GET

**Parameters**:
- `filename` (path parameter): The filename of the AI SBOM to download

**cURL Example**:
```bash
curl -X GET "https://aetheris-ai-aibom-generator.hf.space/download/{filename}" \
  -o "downloaded_aibom.json"
```

**Expected Response**: The AI SBOM JSON file will be downloaded to your local machine.

### Form-Based Generation (Web UI)

**Purpose**: Generate an AI SBOM using form data (typically used by the web UI).

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

**Expected Response**: HTML page with the generated AI SBOM results.

## Web UI

The API also provides a web user interface for generating AI SBOMs without writing code:

**URL**: `https://aetheris-ai-aibom-generator.hf.space/`

The web UI allows you to:
1. Enter a Hugging Face model ID
2. Configure generation options
3. Generate an AI SBOM
4. View the results in a human-friendly format
5. Download the generated AI SBOM as a JSON file

## Understanding the Field Checklist

In the Field Checklist tab of the results page, you'll see a list of fields with check marks (✔/✘) and stars (★). Here's what they mean:

- **Check marks**:
  - ✔: Field is present in the AI SBOM
  - ✘: Field is missing from the AI SBOM

- **Stars** (importance level):
  - ★★★ (three stars): Critical fields - Essential for a valid and complete AI SBOM
  - ★★ (two stars): Important fields - Valuable information that enhances completeness
  - ★ (one star): Supplementary fields - Additional context and details (optional)

## Security Features

The API includes several security features to protect against Denial of Service (DoS) attacks:

1. **Rate Limiting**: Limits the number of requests a single IP address can make within a specific time window.

2. **Concurrency Limiting**: Restricts the total number of simultaneous requests being processed to prevent resource exhaustion.

3. **Request Size Limiting**: Prevents attackers from sending extremely large payloads that could consume memory or processing resources.

4. **API Key Authentication** (optional): When configured, requires an API key for accessing API endpoints, enabling tracking and control of API usage.

5. **CAPTCHA Verification** (optional): When configured for the web interface, helps ensure requests come from humans rather than bots.

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
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error (server-side error)
- 503: Service Unavailable (server at capacity)

Error responses include a detail message explaining the error:
```json
{
  "detail": "Error generating AI SBOM: Model not found"
}
```

## Completeness Score

The completeness score is calculated based on the presence and quality of various fields in the AI SBOM. The score is broken down into sections:

1. **Required Fields** (20 points): Basic required fields for a valid AI SBOM
2. **Metadata** (20 points): Information about the AI SBOM itself
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

This helps you understand how much the AI enhancement improved the AI SBOM's completeness.
