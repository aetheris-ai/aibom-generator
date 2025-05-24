FROM python:3.10-slim

WORKDIR /app

# This ensures models are cached properly and don’t re-download every time
RUN mkdir /.cache && chmod 777 /.cache
ENV TRANSFORMERS_CACHE=/.cache
ENV HF_HOME=/.cache
# Optional: Set default values for API configuration
# ENV AIBOM_USE_INFERENCE=true
# ENV AIBOM_CACHE_DIR=/.cache

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (including setup.py)
COPY . /app

# Safety check to ensure correct directory naming
RUN if [ -d "/app/src/aibom-generator" ] && [ ! -d "/app/src/aibom_generator" ]; then \
    mv /app/src/aibom-generator /app/src/aibom_generator; \
    echo "Renamed directory to match Python import conventions"; \
fi

# Creates a directory called "output" inside application directory, sets permissions so that the application can write files to this directory
# RUN mkdir -p /app/output && chmod 777 /app/output

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app

# Create entrypoint script
RUN chmod +x /app/entrypoint.sh

# Command to run the application
ENTRYPOINT ["/app/entrypoint.sh"]
