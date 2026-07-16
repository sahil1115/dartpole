# document_processing/config_handler.py
# version: 1.2.0
# file : 1


import logging
import os

# Attempt to import user_config, otherwise use DummyConfig
try:
    import config as user_config # Assuming config.py is in the parent directory (src)
except ImportError:
    user_config = None # Will be handled below
    
# 
try:
    import config as user_config # Assuming config.py is in the parent directory (src)
except ImportError:
    user_config = None

class DummyConfig:
    APP_NAME = "DocProcessor"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    PDF_MIN_TEXT_LENGTH = 30
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc']
    ENABLE_SEMANTIC_CHUNKING = False
    SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    ENABLE_FORM_EXTRACTION = True
    ENABLE_TABLE_SUMMARIZATION = False # Original was False
    ENABLE_GRAPH_EXTRACTION = True    # Original was True
    ENABLE_AUTO_ROTATION = True
    OCR_CONFIDENCE_THRESHOLD = 2.0
    ENABLE_OCR_ROTATION = True # Added from original document_processor.py


# Initialize a config object (instance of DummyConfig to start)
config = DummyConfig()

# Populate the config object from user_config or use DummyConfig defaults
if user_config:
    default_config_keys = {k: v for k, v in DummyConfig.__dict__.items() if not k.startswith("__")}
    for key, default_value in default_config_keys.items():
        setattr(config, key, getattr(user_config, key, default_value))
        # Original document_processor.py had a warning for missing non-critical keys,
        # this can be re-added if necessary.
else:
    # This case means user_config.py (src/config.py) was not found or importable
    # The 'config' object will retain DummyConfig defaults.
    # A warning about this situation was present in the original file.
    print("Warning: src/config.py not found or not importable at this level. Using default settings for document_processing module.")


# Setup Logger (can be centralized here if all modules use the same logger name base)
# However, the original document_processor.py used config.APP_NAME for the logger.
# It's often better for each module to get its own logger: logger = logging.getLogger(__name__)
# For now, we replicate the original logger fetching based on APP_NAME from the loaded config.

logger = logging.getLogger(config.APP_NAME) # Uses APP_NAME from the config instance

# Original feature availability logging can be moved to the main DocumentProcessor or relevant modules
# Example:
# logger.info(f"--- Feature Availability (from config_handler perspective) ---")
# logger.info(f"PDFPlumber (Tables): {PDFPLUMBER_AVAILABLE}") # These need to be checked where used
# logger.info(f"Table Summarization: {'ENABLED' if config.ENABLE_TABLE_SUMMARIZATION else 'DISABLED'}")