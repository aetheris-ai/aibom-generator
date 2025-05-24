import requests
import os
import logging

logger = logging.getLogger(__name__)

RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")
RECAPTCHA_VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"

def verify_recaptcha(recaptcha_response ):
    """Verify reCAPTCHA response."""
    if not RECAPTCHA_SECRET_KEY:
        # If no secret key is set, don't enforce CAPTCHA
        logger.warning("RECAPTCHA_SECRET_KEY environment variable not set. CAPTCHA verification disabled.")
        return True
        
    if not recaptcha_response:
        logger.warning("No reCAPTCHA response provided")
        return False
        
    try:
        response = requests.post(
            RECAPTCHA_VERIFY_URL,
            data={
                "secret": RECAPTCHA_SECRET_KEY,
                "response": recaptcha_response
            }
        )
        result = response.json()
        return result.get("success", False)
    except Exception as e:
        logger.error(f"reCAPTCHA verification error: {str(e)}")
        # On error, allow the request to proceed
        return True
