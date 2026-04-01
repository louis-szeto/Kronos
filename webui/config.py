"""Configuration, environment variables, and API key management."""

import os
import secrets
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Security: structured logging, no credentials in output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Security: allowed data directory for file operations
DATA_DIR = Path(os.environ.get(
    'KRONOS_DATA_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
)).resolve()
RESULTS_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')).resolve()

# API key file for persistence
_API_KEY_FILE = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.api_key'))

# Security: API key for endpoint protection
def _load_or_generate_api_key():
    """Load API key from file → env → generate + persist."""
    # 1. Environment variable takes precedence
    env_key = os.environ.get('KRONOS_API_KEY')
    if env_key:
        return env_key

    # 2. Persisted file
    if _API_KEY_FILE.exists():
        try:
            key = _API_KEY_FILE.read_text().strip()
            if key:
                logger.info("Loaded persisted API key from %s", _API_KEY_FILE)
                return key
        except OSError:
            pass

    # 3. Generate and persist
    key = secrets.token_urlsafe(32)
    try:
        _API_KEY_FILE.write_text(key)
        logger.info("Generated and persisted API key to %s", _API_KEY_FILE)
    except OSError as e:
        logger.warning("Could not persist API key: %s", e)
    logger.warning("No KRONOS_API_KEY env var set. Using generated key (persistent across restarts). "
                   "Set KRONOS_API_KEY env var for production!")
    return key

API_KEY = _load_or_generate_api_key()

# Security: allowed data directory for file operations
DATA_DIR = Path(os.environ.get(
    'KRONOS_DATA_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
)).resolve()
RESULTS_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')).resolve()
