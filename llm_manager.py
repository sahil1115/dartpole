
# llm_manager.py
# version 1.0.1
import os
import logging
import torch
import gc
import time # For cleanup delay
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# LLM Imports - Import necessary classes based on potential providers
from langchain_community.llms import Ollama # For Ollama
# --- Uncomment imports for API providers when needed ---
# from langchain_openai import ChatOpenAI # For OpenAI
# from langchain_anthropic import ChatAnthropic # For Anthropic
# from langchain_google_genai import ChatGoogleGenerativeAI # For Google AI
# ----------------------------------------------------

# Import config and helper function
import config
# from config import get_api_key # Uncomment if using the helper
from chromadb.api.client import SharedSystemClient

# Setup logger
logger = logging.getLogger(__name__)
# Configure logging once, preferably in main.py or the entry point
# logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Session-level embedding model (in-memory only) ---
# Loading the HuggingFace embedding model takes 10-30s on CPU. It is stateless,
# so a single instance is shared by every request for the lifetime of the server
# process. Nothing is persisted to disk; the model is released when the app exits.
_shared_embeddings = None

def get_shared_embeddings():
    """Returns the process-wide embedding model, loading it on first use."""
    global _shared_embeddings
    if _shared_embeddings is None:
        logger.info(f"Loading embedding model (once per session): {config.EMBEDDING_MODEL_NAME} on {config.EMBEDDING_DEVICE}")
        _shared_embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': config.NORMALIZE_EMBEDDINGS}
        )
        logger.info("Embedding model loaded and will be reused for this session.")
    return _shared_embeddings

class LLMManager:
    def __init__(self, db_directory=config.VECTOR_STORE_DIR):
        """Initialize LLM Manager"""
        self.db_directory = db_directory
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.llm_client = None # Store the initialized LLM client

        logger.info(f"LLMManager initialized. Configured LLM Provider: {config.LLM_PROVIDER}")
        logger.info(f"Embedding device: {config.EMBEDDING_DEVICE}")
        logger.info(f"Vector store directory: {self.db_directory}")


    def initialize_embeddings(self):
        """Attaches the shared session embedding model (loads it on first use)."""
        if self.embeddings:
            logger.debug("Embeddings model already attached.")
            return self.embeddings
        try:
            self.embeddings = get_shared_embeddings()
            return self.embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model '{config.EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
            self.embeddings = None # Ensure state reflects failure
            raise RuntimeError(f"Failed to initialize embeddings: {e}") from e


    def process_documents(self, documents):
        """
        Creates a vector store from processed document chunks.
        Assumes embeddings are already initialized.
        """
        if not documents:
            logger.warning("No documents provided to process_documents. Cannot create vector store.")
            # Consider creating an empty store structure if required downstream
            self.vector_store = None
            return None
        if not self.embeddings:
            logger.error("Embeddings not initialized before calling process_documents.")
            raise ValueError("Embeddings must be initialized before processing documents.")

        logger.info(f"Creating/updating vector store at: {self.db_directory}")
        logger.info(f"Processing {len(documents)} document chunks into vector store...")
        try:
            # Extract texts and metadatas from the list of dicts
            texts = [doc['page_content'] for doc in documents if 'page_content' in doc]
            metadatas = [doc['metadata'] for doc in documents if 'metadata' in doc]

            if len(texts) != len(documents):
                 logger.warning("Some documents were missing 'page_content' or 'metadata'.")
            if not texts:
                 logger.warning("No text content found in documents to add to vector store.")
                 # Create empty store structure?
                 self.vector_store = Chroma.from_texts( # Create with dummy data and immediately delete? Or just initialize?
                        texts=[""], # Add a dummy text to initialize the collection
                        embedding=self.embeddings,
                        metadatas=[{'source': 'dummy'}],
                        persist_directory=self.db_directory
                    )
                 self.vector_store.delete(ids=self.vector_store.get()['ids']) # Delete the dummy entry
                 self.vector_store.persist()
                 logger.info(f"Empty vector store structure created due to no input texts.")
                 return self.vector_store

            # Create Chroma vector store from texts
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=self.db_directory
            )
            self.vector_store.persist() # Ensure data is written to disk
            count = self.vector_store._collection.count() if self.vector_store and hasattr(self.vector_store, '_collection') else 'N/A'
            logger.info(f"Vector store created/updated with {count} document chunks.")
            return self.vector_store

        except Exception as e:
            logger.error(f"Failed to create/process vector store: {e}", exc_info=True)
            self.vector_store = None # Ensure state reflects failure
            raise RuntimeError(f"Failed to create vector store: {e}") from e


    def initialize_llm(self, model_name=None):
        """
        Initialize the selected LLM client and QA Chain.
        Requires vector_store to be loaded/initialized first.
        Accepts an optional model_name override (primarily for Ollama).
        """
        if self.qa_chain:
            logger.info(f"{config.LLM_PROVIDER} LLM ({getattr(self.llm_client, 'model', 'N/A')}) and QA Chain already initialized.")
            return self.qa_chain

        if not self.vector_store:
            logger.error("Vector store not initialized or loaded. Cannot initialize LLM QA chain.")
            raise ValueError("Vector store must be initialized/loaded before initializing the LLM.")

        logger.info(f"Initializing LLM client for provider: {config.LLM_PROVIDER}")
        self.llm_client = None # Reset client

        # Determine the effective model name, especially for Ollama
        effective_ollama_model = model_name if model_name and config.LLM_PROVIDER.lower() == "ollama" else config.OLLAMA_MODEL
        if config.LLM_PROVIDER.lower() == "ollama" and not effective_ollama_model:
             logger.error("Ollama provider selected, but no model name specified via parameter or config.")
             raise ValueError("Ollama model name is required but missing.")
        if config.LLM_PROVIDER.lower() == "ollama":
             logger.info(f"Target Ollama model for initialization: {effective_ollama_model}")


        try:
            # --- Conditional LLM Initialization ---
            provider = config.LLM_PROVIDER.lower()

            if provider == "ollama":
                self.llm_client = Ollama(
                    base_url=config.OLLAMA_BASE_URL,
                    model=effective_ollama_model, # Use the determined model name
                    timeout=config.OLLAMA_REQUEST_TIMEOUT,
                    temperature=config.OLLAMA_TEMPERATURE, # Pass temperature from config
                    stop=config.OLLAMA_STOP_SEQUENCES, # Pass stop sequences
                    num_ctx=getattr(config, 'OLLAMA_NUM_CTX', 4096), # Fit retrieved chunks without truncation
                    keep_alive=getattr(config, 'OLLAMA_KEEP_ALIVE', "30m"), # Keep model hot between queries
                )
                # Optional: Add a simple connection check here if needed
                try:
                     # A light check, like listing models again or a specific health endpoint if available
                     self.llm_client.invoke("Respond with OK.") # Simple check
                     logger.info(f"Successfully connected to Ollama model {effective_ollama_model}.")
                except Exception as ollama_conn_e:
                     logger.error(f"Failed to connect or communicate with Ollama model {effective_ollama_model} at {config.OLLAMA_BASE_URL}: {ollama_conn_e}", exc_info=False)
                     raise ConnectionError(f"Failed to connect to Ollama model '{effective_ollama_model}'. Is it running and accessible at {config.OLLAMA_BASE_URL}?") from ollama_conn_e


            # === Add other providers within elif blocks ===
            elif provider == "openai":
                 # --- Check if OpenAI config seems valid before trying ---
                 if not config.OPENAI_API_KEY or "YOUR_" in config.OPENAI_API_KEY:
                      raise ValueError("OpenAI API Key not configured correctly in config.py or environment variables.")
                 try:
                      from langchain_openai import ChatOpenAI # Import here to avoid error if not installed
                      logger.info(f"Initializing OpenAI client for model: {config.OPENAI_MODEL_NAME}")
                      self.llm_client = ChatOpenAI(
                          openai_api_key=config.OPENAI_API_KEY,
                          model_name=config.OPENAI_MODEL_NAME,
                          temperature=config.OPENAI_TEMPERATURE,
                          max_tokens=config.OPENAI_MAX_TOKENS
                      )
                      # Add a simple check if needed (e.g., invoke)
                 except ImportError:
                      logger.error("Missing 'langchain-openai' package. Please install it.")
                      raise ImportError("OpenAI provider selected, but 'langchain-openai' is not installed.")

            # Add elif blocks for anthropic, google, etc. similar to openai

            else:
                logger.error(f"Unsupported LLM provider configured: {config.LLM_PROVIDER}")
                raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")
            # --------------------------------------

            # --- Create QA chain using the selected llm_client ---
            if not self.llm_client:
                 logger.error("LLM client failed to initialize.")
                 raise RuntimeError("LLM client could not be initialized for the selected provider.")

            logger.debug(f"Creating RetrievalQA chain with {config.LLM_PROVIDER} LLM.")
            retriever = self.vector_store.as_retriever(
                search_type=config.RETRIEVER_SEARCH_TYPE,
                search_kwargs={"k": config.RETRIEVER_K}
            )
            prompt_template = self._create_prompt_template()

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_client,
                chain_type=config.CHAIN_TYPE,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )

            llm_model_name = getattr(self.llm_client, 'model', effective_ollama_model if provider == 'ollama' else 'N/A')
            logger.info(f"{config.LLM_PROVIDER} LLM client ({llm_model_name}) and RetrievalQA chain initialized successfully.")
            return self.qa_chain

        except ImportError as e:
             # Catch missing LangChain integration packages
             logger.error(f"Missing required LangChain components for provider '{config.LLM_PROVIDER}'. "
                          f"Ensure necessary packages (e.g., 'langchain-openai') are installed. Error: {e}", exc_info=False)
             raise ImportError(f"Missing dependencies for LLM provider {config.LLM_PROVIDER}. Please install required packages.") from e
        except ValueError as e: # Catch configuration errors (like missing keys, invalid model names)
             logger.error(f"Configuration or Value error for provider '{config.LLM_PROVIDER}': {e}", exc_info=False)
             raise
        except ConnectionError as e: # Catch connection errors (Ollama, APIs)
             logger.error(f"Connection error for provider '{config.LLM_PROVIDER}': {e}", exc_info=False)
             raise
        except NotImplementedError as e: # Catch if a provider block is commented out
             logger.error(f"Provider '{config.LLM_PROVIDER}' is selected but not fully implemented/enabled: {e}", exc_info=False)
             raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error initializing LLM ({config.LLM_PROVIDER}) or QA chain: {e}", exc_info=True)
            self.cleanup() # Attempt cleanup on unexpected failure
            raise RuntimeError(f"Unexpected error during LLM/QA setup: {e}") from e


    def _create_prompt_template(self):
        """Creates the prompt template for the RAG chain."""
        # Use the template from config file
        template = config.RAG_PROMPT_TEMPLATE
        logger.debug(f"Using RAG prompt template:\n{template}")
        return PromptTemplate(template=template, input_variables=["context", "question"])


    def query(self, question):
        """Performs a query using the initialized RAG chain."""
        if not self.qa_chain:
            logger.error("Query attempted before QA chain was initialized.")
            raise ValueError("LLM QA chain not initialized. Initialize system and select model first.")

        llm_model_name = getattr(self.llm_client, 'model', 'N/A')
        logger.info(f"Processing query with {config.LLM_PROVIDER} ({llm_model_name}): '{question}'")
        try:
            result = self.qa_chain.invoke({"query": question}) # Use invoke for LCEL chains
            logger.debug(f"Raw query result: {result}") # Log raw result for debugging
            logger.info("Query processed successfully.")
            return result # Return the dictionary result
        except Exception as e:
            logger.error(f"Error during query processing with {config.LLM_PROVIDER} ({llm_model_name}): {e}", exc_info=True)
            raise RuntimeError(f"Query failed: {e}") from e


    def cleanup_embeddings_only(self):
        """Detaches this manager from the shared session embedding model.

        The model itself stays loaded for the rest of the session so the next
        operation doesn't pay the (10-30s) load cost again; it is released
        when the server process exits.
        """
        if self.embeddings:
            logger.debug("Detaching from shared embedding model (model stays loaded for this session).")
            self.embeddings = None
        else:
            logger.debug("No embedding model reference to detach.")


# In llm_manager.py

    def cleanup(self):
        """Cleans up all resources managed by LLMManager."""
        logger.info("Cleaning up LLMManager resources...")

        # 1. Cleanup QA Chain
        if self.qa_chain:
             logger.debug("Deleting QA chain object.")
             try: del self.qa_chain; self.qa_chain = None
             except Exception as e: logger.error(f"Error cleaning up QA chain: {e}", exc_info=True)

        # 2. Cleanup LLM Client
        if self.llm_client:
             logger.debug(f"Deleting LLM client object ({getattr(self.llm_client, 'model', 'N/A')}).")
             try: del self.llm_client; self.llm_client = None
             except Exception as e: logger.error(f"Error cleaning up LLM client: {e}", exc_info=True)

        # --- START MODIFICATION ---


        # 3. Cleanup Vector Store object reference FIRST (shuts down Chroma client)
        # if self.vector_store:
        #     logger.debug("Persisting & shutting down Chroma vector store client...")
        #     try:
        #         self.vector_store.persist()
        #         client_attr = getattr(self.vector_store, "_client", getattr(self.vector_store, "client", None))
        #         if client_attr:
        #             client_attr.shutdown()
        #         self.vector_store = None
        #         logger.info("Chroma vector store client shut down successfully.")
        #         time.sleep(0.1)
        #     except Exception as err:
        #         logger.error(f"Error during Chroma client shutdown: {err}", exc_info=True)
        # else:
        #     logger.info("No active Chroma vector store to shut down.")


        # 3. Shutdown Chroma vector store client
        if self.vector_store:
            logger.debug("Persisting vector store to disk…")
            try:
                self.vector_store.persist()
            except Exception:
                pass

            # grab underlying chromadb.Client (PersistentClient or HttpClient)
            client = getattr(self.vector_store, "_client", None) or getattr(self.vector_store, "client", None)
            if client:
                try:
                    # hack: stop its internal background threads
                    if hasattr(client, "_system"):
                        client._system.stop()
                    # remove its entry from the shared registry so it can truly be GC’d
                    SharedSystemClient._identifier_to_system.pop(client._identifier, None)
                    logger.debug("Chroma client threads stopped via SharedSystemClient hack.")
                except Exception as err:
                    logger.warning(f"Could not fully stop Chroma client threads: {err}", exc_info=True)

            # drop our reference
            self.vector_store = None

            # force a few GC runs + small delays so Windows has time to release the files
            import gc, time
            for _ in range(3):
                gc.collect()
                time.sleep(0.5)

            logger.info("Vector store torn down and in-memory references cleared.")
        else:
            logger.info("No vector store instance to shut down.")

        # 4. Cleanup Embeddings Model (calls cleanup_embeddings_only)
        self.cleanup_embeddings_only()
        # --- END MODIFICATION ---

        # Garbage collect just in case
        gc.collect()
        logger.info("LLMManager resource cleanup sequence finished.")

    # Ensure 'import time' and 'import gc' are at the top of llm_manager.py