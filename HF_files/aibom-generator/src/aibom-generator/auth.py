from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
import os
import logging

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
API_KEY = os.environ.get("API_KEY")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not API_KEY:
        # If no API key is set, don't enforce authentication
        logger.warning("API_KEY environment variable not set. API authentication disabled.")
        return None
        
    if api_key_header == API_KEY:
        return api_key_header
    
    logger.warning(f"Invalid API key attempt")
    raise HTTPException(status_code=403, detail="Invalid API Key")
