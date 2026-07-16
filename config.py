# config.py
# version : 1.1.5
import torch
import os
import logging

# --- General Configuration ---
APP_NAME = "DartPole"
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Document Processing ---
DEFAULT_DOCUMENTS_DIR = r"C:\read_doc"
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md']
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PDF_MIN_TEXT_LENGTH = 20
# Semantic (topic-clustering) chunking loads a second transformer model and embeds
# every paragraph during processing — much slower, and it can reorder paragraphs.
# The default recursive splitter is faster and respects document order.
ENABLE_SEMANTIC_CHUNKING = False
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENABLE_FORM_EXTRACTION = True
ENABLE_TABLE_SUMMARIZATION = True
ENABLE_GRAPH_EXTRACTION = False
ENABLE_AUTO_ROTATION = True
OCR_CONFIDENCE_THRESHOLD = 2.0
ENABLE_OCR_ROTATION = True

# --- Local Embeddings Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VECTOR_STORE_DIR = "vectorstore_dartpole"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZE_EMBEDDINGS = True
ENABLE_MINICPM_IMAGE_PROCESSING = True
MINICPM_MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"

# --- LLM Provider Selection ---
LLM_PROVIDER = "ollama"

# --- Ollama LLM Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = ""
OLLAMA_REQUEST_TIMEOUT = 1000
OLLAMA_TEMPERATURE = 0.1
OLLAMA_STOP_SEQUENCES = ["\nObservation:", "\nContext:", "\nQuestion:"]
# Context window: must be large enough to hold RETRIEVER_K chunks + prompt + answer,
# otherwise Ollama silently truncates the retrieved context (default is only 2048).
OLLAMA_NUM_CTX = 4096
# Keep the model loaded in Ollama between queries (default is 5m) so follow-up
# questions don't pay the model reload cost.
OLLAMA_KEEP_ALIVE = "30m"

# --- RAG Chain Parameters ---
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_K = 5
CHAIN_TYPE = "stuff"
RAG_PROMPT_TEMPLATE = """You are an AI assistant specialized in analyzing the provided documents. Use the following pieces of context extracted from the documents to answer the question accurately and concisely. If you don't know the answer based *only* on the context provided, explicitly state that the information is not available in the documents. Do not make up information or use external knowledge.

Context:
{context}

Question: {question}

Answer:"""

# --- API Server ---
API_HOST = "localhost"
API_PORT = 5000
API_DEBUG_MODE = True

# --- UI Server ---
UI_HOST = "localhost"
UI_PORT = 8000
UI_DEBUG_MODE = True
UI_RELOADER = False
AUTO_OPEN_BROWSER = True

# --- Cleanup Behavior ---
CLEAR_VECTOR_STORE_ON_CLEANUP = True

# --- Helper function to get API keys safely ---
helper_logger = logging.getLogger(__name__ + ".config_helper")

def get_api_key(provider_name, env_var_name, default_value=""):
    """Gets API key from environment or config, logging a warning if not found or seems like a placeholder."""
    key = os.getenv(env_var_name, default_value)
    placeholder_texts = ["YOUR_", "_HERE", "API_KEY"]

    if not key or key == default_value or any(pt in key for pt in placeholder_texts):
        message = f"{provider_name} API key not found or appears to be a placeholder in environment variable '{env_var_name}' or config.py."
        if LLM_PROVIDER.lower() == provider_name.lower():
            helper_logger.error(message)
            raise ValueError(message + f" Please set the API key for the selected provider '{provider_name}'.")
        else:
            helper_logger.warning(message + " Using default/placeholder value.")
            return key

    helper_logger.info(f"Found API key for {provider_name} via '{env_var_name}' or config.")
    return key