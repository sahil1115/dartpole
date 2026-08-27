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
# No legacy '.doc': docx2txt only reads zip-based .docx; old OLE .doc files
# would be listed but fail silently during processing.
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md']
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

# --- Document Insights (executive summary + key entities) ---
# Budget is chars, not tokens, but ~4 chars/token keeps the sample well under
# OLLAMA_NUM_CTX once the prompt wrapper and the answer itself are accounted for.
INSIGHTS_CONTEXT_CHARS = 6000
INSIGHTS_MAX_ENTITIES = 25
# Stop sequences (OLLAMA_STOP_SEQUENCES above) cut generation on "\nContext:" and
# "\nQuestion:" — these prompts deliberately avoid those labels.
INSIGHTS_SUMMARY_PROMPT = """You are an AI assistant specialized in analyzing documents. Below are representative excerpts drawn from a folder of documents ({document_names}). Write a 1-page executive summary of the whole folder: what these documents are, their main topics, and their most important points. Base the summary *only* on the excerpts provided. Write in clear prose, no headings, no bullet points.

DOCUMENTS: {document_names}

EXCERPTS:
{excerpts}

EXECUTIVE SUMMARY:"""
INSIGHTS_ENTITIES_PROMPT = """Read the excerpts below and list the key named entities they mention: people, organizations, locations, dates, monetary amounts, and products. Respond with *only* a JSON array, no other text, in this exact format:
[{"text": "Acme Corp", "type": "organization"}, {"text": "March 3, 2025", "type": "date"}]
Allowed "type" values: person, organization, location, date, amount, product, other. Include at most 25 entities, most important first.

EXCERPTS:
{excerpts}

JSON:"""

# --- Bill Analysis & Forecasting ---
# Text budget for the per-document extraction prompt. Kept well under
# OLLAMA_NUM_CTX so a long candidate list and the JSON response still fit.
BILLS_MAX_DOC_CHARS = 3000
# Cap how many candidates go into the prompt/schema enum per document, so a
# noisy statement (e.g. a full year's transaction history) can't blow the
# context budget or produce an unwieldy enum.
BILLS_MAX_CANDIDATES = 20
BILLS_FORECAST_WINDOW_DAYS = 30
BILLS_CATEGORIES = [
    "Utilities", "Rent/Mortgage", "Insurance", "Subscriptions",
    "Telecom", "Credit Card", "Loan", "Medical", "Other",
]
# The model chooses only from candidates already found by exact text
# matching (enforced via the request's JSON-schema "enum") — it is never
# asked to read or compute a number itself, since money can't tolerate the
# error rate that free-form generation would produce.
BILLS_EXTRACTION_PROMPT = """You are analyzing ONE document to decide whether it is a bill, invoice, or financial statement that requires payment.

Below is the document text, followed by amount candidates and date candidates already found in it by exact text matching. You must choose your answers for total_due, due_date, and issue_date *only* from these exact candidate strings — never invent, calculate, or reformat a value. If the correct value is not among the candidates, leave that field as an empty string.

DOCUMENT:
{document_text}

AMOUNT CANDIDATES:
{amount_candidates}

DATE CANDIDATES:
{date_candidates}

Decide: is this a bill/invoice/statement requiring payment? Who is the vendor? What spending category does it belong to? Which candidate is the total amount due? Which candidate is the due date? Which candidate is the issue or statement date?"""

# --- API Server ---
API_HOST = "localhost"
API_PORT = 5000
API_DEBUG_MODE = False

# --- UI Server ---
UI_HOST = "localhost"
UI_PORT = 8000
UI_DEBUG_MODE = False
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