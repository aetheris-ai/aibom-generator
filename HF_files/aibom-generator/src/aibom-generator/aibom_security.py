"""
Security module for AIBOM generator implementation.

This module provides security functions that can be integrated
into the AIBOM generator to improve input validation, error handling,
and protection against common web vulnerabilities.
"""

import re
import os
import json
import logging
from typing import Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def validate_model_id(model_id: str) -> str:
    """
    Validate model ID to prevent injection attacks.
    
    Args:
        model_id: The model ID to validate
        
    Returns:
        The validated model ID
        
    Raises:
        ValueError: If the model ID contains invalid characters
    """
    # Only allow alphanumeric characters, hyphens, underscores, and forward slashes
    if not model_id or not isinstance(model_id, str):
        raise ValueError("Model ID must be a non-empty string")
        
    if not re.match(r'^[a-zA-Z0-9_\-/]+$', model_id):
        raise ValueError(f"Invalid model ID format: {model_id}")
    
    # Prevent path traversal attempts
    if '..' in model_id:
        raise ValueError(f"Invalid model ID - contains path traversal sequence: {model_id}")
        
    return model_id

def safe_path_join(directory: str, filename: str) -> str:
    """
    Safely join directory and filename to prevent path traversal attacks.
    
    Args:
        directory: Base directory
        filename: Filename to append
        
    Returns:
        Safe file path
    """
    # Ensure filename doesn't contain path traversal attempts
    filename = os.path.basename(filename)
    return os.path.join(directory, filename)

def safe_json_parse(json_string: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely parse JSON with error handling.
    
    Args:
        json_string: JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    if default is None:
        default = {}
        
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Invalid JSON: {e}")
        return default

def sanitize_html_output(text: str) -> str:
    """
    Sanitize text for safe HTML output to prevent XSS attacks.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Replace HTML special characters with their entities
    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '/': '&#x2F;',
    }
    
    for char, entity in replacements.items():
        text = text.replace(char, entity)
        
    return text

def secure_file_operations(file_path: str, operation: str, content: Optional[str] = None) -> Union[str, bool]:
    """
    Perform secure file operations with proper error handling.
    
    Args:
        file_path: Path to the file
        operation: Operation to perform ('read', 'write', 'append')
        content: Content to write (for 'write' and 'append' operations)
        
    Returns:
        File content for 'read' operation, True for successful 'write'/'append', False otherwise
    """
    try:
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif operation == 'write' and content is not None:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        elif operation == 'append' and content is not None:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            logger.error(f"Invalid file operation: {operation}")
            return False
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return "" if operation == 'read' else False

def validate_url(url: str) -> bool:
    """
    Validate URL format to prevent malicious URL injection.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    # Basic URL validation
    url_pattern = re.compile(
        r'^(https?):\/\/'  # http:// or https://
        r'(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*'  # domain segments
        r'([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])'  # last domain segment
        r'(:\d+)?'  # optional port
        r'(\/[-a-zA-Z0-9%_.~#+]*)*'  # path
        r'(\?[;&a-zA-Z0-9%_.~+=-]*)?'  # query string
        r'(\#[-a-zA-Z0-9%_.~+=/]*)?$'  # fragment
    )
    
    return bool(url_pattern.match(url))

def secure_template_rendering(template_content: str, context: Dict[str, Any]) -> str:
    """
    Render templates securely with auto-escaping enabled.
    
    This is a placeholder function. In a real implementation, you would use
    a template engine like Jinja2 with auto-escaping enabled.
    
    Args:
        template_content: Template content
        context: Context variables for rendering
        
    Returns:
        Rendered template
    """
    try:
        from jinja2 import Template
        template = Template(template_content, autoescape=True)
        return template.render(**context)
    except ImportError:
        logger.error("Jinja2 not available, falling back to basic rendering")
        # Very basic fallback (not recommended for production)
        result = template_content
        for key, value in context.items():
            if isinstance(value, str):
                placeholder = "{{" + key + "}}"
                result = result.replace(placeholder, sanitize_html_output(value))
        return result
    except Exception as e:
        logger.error(f"Template rendering failed: {e}")
        return ""

def implement_rate_limiting(user_id: str, action: str, limit: int, period: int) -> bool:
    """
    Implement basic rate limiting to prevent abuse.
    
    This is a placeholder function. In a real implementation, you would use
    a database or cache to track request counts.
    
    Args:
        user_id: Identifier for the user
        action: Action being performed
        limit: Maximum number of actions allowed
        period: Time period in seconds
        
    Returns:
        True if action is allowed, False if rate limit exceeded
    """
    # In a real implementation, you would:
    # 1. Check if user has exceeded limit in the given period
    # 2. If not, increment counter and allow action
    # 3. If yes, deny action
    
    # Placeholder implementation always allows action
    logger.info(f"Rate limiting check for user {user_id}, action {action}")
    return True

# Integration example for the AIBOM generator
def secure_aibom_generation(model_id: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Example of how to integrate security improvements into AIBOM generation.
    
    Args:
        model_id: Model ID to generate AIBOM for
        output_file: Optional output file path
        
    Returns:
        Generated AIBOM data
    """
    try:
        # Validate input
        validated_model_id = validate_model_id(model_id)
        
        # Process model ID securely
        # (This would call your actual AIBOM generation logic)
        aibom_data = {"message": f"AIBOM for {validated_model_id}"}
        
        # Handle output file securely if provided
        if output_file:
            safe_output_path = safe_path_join(os.path.dirname(output_file), os.path.basename(output_file))
            secure_file_operations(safe_output_path, 'write', json.dumps(aibom_data, indent=2))
            
        return aibom_data
        
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {e}")
        return {"error": "Invalid input parameters"}
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"AIBOM generation failed: {e}")
        return {"error": "An internal error occurred"}
