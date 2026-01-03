"""
WSGI entry point for Gunicorn
This file allows Gunicorn to properly import and run the Flask app
"""
import sys
import os
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    sys.path.insert(0, current_dir)
    sys.path.insert(0, parent_dir)
    
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Python path: {sys.path}")
    
    # Import the Flask app
    logger.info("Attempting to import app from app module...")
    from app import app
    
    logger.info("Successfully imported Flask app")
    logger.info(f"App routes: {[str(rule) for rule in app.url_map.iter_rules()]}")
    
    # This is what Gunicorn will use
    application = app
    
except Exception as e:
    logger.error(f"Failed to import app: {e}", exc_info=True)
    raise

if __name__ == "__main__":
    app.run()

