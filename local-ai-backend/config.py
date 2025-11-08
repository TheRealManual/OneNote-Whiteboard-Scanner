"""
Configuration settings for the AI backend (Hybrid Extractor Mode)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[CONFIG] Loaded environment from {env_path}")
else:
    print(f"[CONFIG] No .env file found at {env_path}")
    print(f"[CONFIG] Create .env from .env.example and add your Azure credentials")

# Environment mode
PRODUCTION = os.getenv('PRODUCTION', 'false').lower() == 'true'

# Server configuration
HOST = "127.0.0.1"  # Localhost only for security
PORT = 5000

# CORS allowed origins (add your frontend URLs)
ALLOWED_ORIGINS = [
    "https://localhost:3000",  # React dev server
    "https://localhost:3001",  # Alternate port
    "https://127.0.0.1:3000",
    "https://127.0.0.1:3001",
    "https://85dabf938f5e.ngrok-free.app",  # ngrok tunnel
    "https://onenote.officeapps.live.com",  # OneNote Online
]

# Image processing settings
MAX_IMAGE_SIZE = (960, 540)  # Optimized for Intel CPU/iGPU
MIN_CONTOUR_AREA = 50  # Minimum stroke area in pixels
MAX_CONTOUR_AREA = 50000  # Maximum stroke area

# Processing timeouts
PROCESSING_TIMEOUT = 60  # seconds

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': LOG_FORMAT,
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': LOG_LEVEL,
        },
    },
    'root': {
        'level': LOG_LEVEL,
        'handlers': ['console'],
    },
}

# User configuration file location
USER_CONFIG_FILE = Path.home() / '.onenote_scanner' / 'config.json'

# OneNote OAuth Configuration
ONENOTE_CLIENT_ID = os.getenv('ONENOTE_CLIENT_ID', '')
ONENOTE_CLIENT_SECRET = os.getenv('ONENOTE_CLIENT_SECRET', '')
OAUTH_REDIRECT_URI = os.getenv('OAUTH_REDIRECT_URI', 'http://localhost:5000/onenote/callback')
OAUTH_SCOPES = ['Notes.ReadWrite', 'Notes.Create', 'offline_access']

def validate_config():
    """Validate configuration settings"""
    if not HOST or not PORT:
        raise ValueError("HOST and PORT must be configured")
    if PORT < 1 or PORT > 65535:
        raise ValueError("PORT must be between 1 and 65535")
    return True
