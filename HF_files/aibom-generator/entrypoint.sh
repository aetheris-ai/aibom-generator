#!/bin/bash
set -e

# Default inference URL for internal inference model service
DEFAULT_INFERENCE_URL="http://localhost:8000/extract"
export AIBOM_INFERENCE_URL=${AIBOM_INFERENCE_URL:-$DEFAULT_INFERENCE_URL}

echo "Using AIBOM_INFERENCE_URL: $AIBOM_INFERENCE_URL"

# Check if command-line arguments are provided
if [ -n "$1" ]; then
  case "$1" in
    server)
      # Start the API server explicitly (recommended for Hugging Face Spaces)
      echo "Starting AIBOM Generator API server..."
      exec uvicorn src.aibom_generator.api:app --host 0.0.0.0 --port ${PORT:-7860}
      ;;
    worker)
      # Start the background worker
      echo "Starting AIBOM Generator background worker..."
      exec python -m src.aibom_generator.worker
      ;;
    inference)
      # Start the inference model server
      echo "Starting AIBOM Generator inference model server..."
      exec python -m src.aibom_generator.inference_model --host 0.0.0.0 --port ${PORT:-8000}
      ;;
    *)
      # Run as CLI with provided arguments
      echo "Running AIBOM Generator CLI..."
      exec python -m src.aibom_generator.cli "$@"
      ;;
  esac
else
  # Default behavior (if no arguments): start API server (web UI mode)
  echo "Starting AIBOM Generator API server (web UI)..."
  exec uvicorn src.aibom_generator.api:app --host 0.0.0.0 --port ${PORT:-7860}
fi
