# api_server.py
# version : 1.2.0 (Restored full functionality with ImportError fix)
# author : Sahil Malik

import os
import sys
import atexit
import logging
import platform
import stat
import subprocess
import shutil
import time
# import random # Not explicitly used in restored logic, but was present
import gc
import psutil
from datetime import date
from flask import Flask, request, jsonify, make_response
from langchain_community.vectorstores import Chroma # Assuming Chroma is still used directly here
from threading import Thread # Used for shutdown and the bills background scan
import config # Assuming config.py is in the same directory or accessible
# Ensure document_processing.core_processor and llm_manager are found
from document_processing.core_processor import DocumentProcessor
from llm_manager import LLMManager, get_shared_embeddings
import insights
import bills

# MODIFIED: Added Tkinter imports
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    # This print will happen at server startup if tkinter is missing.
    print("WARNING: tkinter module not found. Folder Browse from UI will be disabled for the API.")


# Setup logger
# Logging will be configured by main.py if this is imported.
# If run directly, the __main__ block's setup_logging will be used.
logger = logging.getLogger(config.APP_NAME + ".API")

# --- Global State ---
app_state = {
    "llm_manager": None,
    "documents_directory": config.DEFAULT_DOCUMENTS_DIR, # Initial default from config
    "active_documents_directory": None, # Path used in the current successful session
    "selected_ollama_model": None,
    "is_initialized": False, # LLM + Vector Store ready state
    "docs_processed": False, # Vector store created/loaded state
    "is_processing": False, # Lock for long operations
    # "initial_docs_path_is_fixed": False, # Removed, as browse button allows override
    "insights": None, # Cached {summary, entities, ...} for the active session
    "insights_generating": False, # Guards concurrent /insights calls
    "bills_job": None, # {status, done, total, current_file, records, error} for /bills/scan
    "bills_detected": False # Cheap regex pre-scan result: does this corpus look bill-like?
}

# ---------------------------------------------------------------------------------------------

def _clear_readonly_and_retry(func, path, _exc_info):
    """rmtree error handler: Windows raises WinError 5 on read-only files/dirs."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def rmtree_robust(path):
    """shutil.rmtree that also removes read-only entries (needed on Windows)."""
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_clear_readonly_and_retry)
    else:
        shutil.rmtree(path, onerror=_clear_readonly_and_retry)

def release_chroma_clients():
    """Stops chromadb's process-global clients so their SQLite/HNSW file handles
    are closed. Required on Windows before the vector store directory can be
    deleted (open handles cause WinError 32)."""
    try:
        from chromadb.api.client import SharedSystemClient
        for identifier in list(SharedSystemClient._identifier_to_system.keys()):
            system = SharedSystemClient._identifier_to_system.pop(identifier, None)
            if system is not None:
                try:
                    system.stop()
                except Exception as stop_err:
                    logger.debug(f"Chroma system '{identifier}' stop error (ignored): {stop_err}")
        gc.collect()
    except Exception as e:
        logger.warning(f"Could not release chromadb clients: {e}")

def reset_vector_store_on_startup():
    """Clean-slate policy: delete any vector store left over from a previous
    session (e.g. after a crash or force-kill) so document data never
    accumulates on disk between sessions."""
    try:
        if os.path.exists(config.VECTOR_STORE_DIR) and os.listdir(config.VECTOR_STORE_DIR):
            logger.info(f"Startup clean slate: removing leftover vector store at '{config.VECTOR_STORE_DIR}'.")
            rmtree_robust(config.VECTOR_STORE_DIR)
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to reset vector store directory on startup: {e}", exc_info=True)

def _run_bills_scan(job, llm_client, vector_store, sources):
    """Background worker for POST /bills/scan. Runs in its own thread since
    a single document takes 10s-70s to extract (observed) and a folder can
    hold dozens — far too slow to be a blocking request like /insights.
    `job` is the dict stored at app_state["bills_job"]; mutating it in place
    is how progress reaches concurrent GET /api/bills polls.
    """
    job["status"] = "running"
    try:
        for source in sources:
            job["current_file"] = os.path.basename(source)
            try:
                record = bills.extract_bill_from_document(llm_client, vector_store, source)
                if record and record.get("is_bill"):
                    job["records"].append(record)
            except Exception as e:
                logger.error(f"Bill extraction failed for '{source}': {e}", exc_info=True)
            job["done"] += 1
        job["status"] = "done"
    except Exception as e:
        logger.error(f"Bills scan failed: {e}", exc_info=True)
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        job["current_file"] = None


def create_api_app():
    """Creates the Flask API application."""
    # No CORS: the UI is served same-origin (DispatcherMiddleware mounts this API
    # under /api on the same port), and the API must not be callable cross-origin —
    # it exposes local document content and cleanup/shutdown controls.
    api_instance = Flask(__name__)

    # --- Helper functions ---
    def get_llm_manager():
        return app_state.get("llm_manager")

    def set_llm_manager(manager):
        app_state["llm_manager"] = manager
        app_state["is_initialized"] = (manager is not None and app_state["docs_processed"])

    def _bills_scan_running():
        # A running scan holds direct references to the current llm_client and
        # vector_store (see start_bills_scan below); re-indexing, switching
        # models, or cleaning up while it runs would tear those out from under
        # the background thread.
        job = app_state.get("bills_job")
        return bool(job and job.get("status") == "running")


    # --- Internal Cleanup Function (defined within create_api_app or passed app_state) ---
    def cleanup_internal():
        """Internal function to perform cleanup actions."""
        logger.info("Attempting to clean up resources (internal)...")
        current_llm_manager = get_llm_manager()
        if current_llm_manager:
            try:
                current_llm_manager.cleanup()
                logger.info("LLM Manager resources cleanup initiated.")
            except Exception as e:
                logger.error(f"Error during LLM Manager cleanup call: {e}", exc_info=True)
        set_llm_manager(None)

        time.sleep(0.2)

        app_state["is_initialized"] = False
        app_state["docs_processed"] = False
        app_state["selected_ollama_model"] = None
        app_state["active_documents_directory"] = None
        app_state["insights"] = None
        app_state["bills_job"] = None
        # app_state["is_processing"] = False; # Caller should manage this

        if config.CLEAR_VECTOR_STORE_ON_CLEANUP and os.path.exists(config.VECTOR_STORE_DIR):
            logger.info(f"Clearing vector store directory: {config.VECTOR_STORE_DIR}")
            release_chroma_clients()
            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries):
                try:
                    rmtree_robust(config.VECTOR_STORE_DIR)
                    logger.info(f"Vector store directory cleared successfully on attempt {attempt + 1}.")
                    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True) # Recreate for next session
                    logger.info(f"Recreated empty directory: {config.VECTOR_STORE_DIR}")
                    break
                except OSError as e:
                    logger.error(f"Attempt {attempt + 1}/{max_retries} failed to clear vector store: {e}", exc_info=False)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logger.error("Giving up on vector store deletion.", exc_info=True)
        elif not config.CLEAR_VECTOR_STORE_ON_CLEANUP:
            logger.info("Vector store directory retained as per config.")
        else:
            logger.info(f"Vector store directory ({config.VECTOR_STORE_DIR}) not found, no need to clear.")
        
        gc.collect()
        logger.info("Application state reset and internal cleanup finished.")


    # --- Error Handler ---
    @api_instance.errorhandler(Exception)
    def handle_generic_error(error):
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        if app_state.get("is_processing", False):
            app_state["is_processing"] = False
            logger.warning("Released processing lock due to unhandled exception.")
        response = jsonify({
            "status": "error",
            "message": "An internal server error occurred.",
            "error_type": type(error).__name__
        })
        return make_response(response, 500)

    # --- API Routes ---
    @api_instance.route('/initialize', methods=['POST'])
    def initialize_documents():
        if app_state["is_processing"]:
            return make_response(jsonify({"status": "error", "message": "Busy with another task."}), 409)
        if _bills_scan_running():
            return make_response(jsonify({"status": "error", "message": "A bills scan is running. Wait for it to finish first."}), 409)

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return make_response(jsonify({"status": "error", "message": "Request body must be a JSON object."}), 400)
        doc_dir = data.get('directory_path')
        if not doc_dir or not isinstance(doc_dir, str) or not os.path.isdir(doc_dir):
            msg = f"Invalid or non-existent directory path provided: '{doc_dir}'"
            logger.error(msg)
            return make_response(jsonify({"status": "error", "message": msg}), 400)
        try:
            os.listdir(doc_dir) # Check readability
        except OSError as e:
            msg = f"Cannot read directory (permissions?): '{doc_dir}'. Error: {e}"
            logger.error(msg)
            return make_response(jsonify({"status": "error", "message": msg}), 403)

        app_state["is_processing"] = True
        if get_llm_manager(): get_llm_manager().cleanup()
        set_llm_manager(None)
        app_state["docs_processed"] = False
        app_state["is_initialized"] = False
        app_state["active_documents_directory"] = None
        app_state["selected_ollama_model"] = None
        app_state["insights"] = None
        app_state["bills_job"] = None
        app_state["bills_detected"] = False

        vector_store_path = config.VECTOR_STORE_DIR
        processed_document_chunks = []
        doc_chunk_count = 0
        embeddings_model = None

        try:
            logger.info(f"Starting document processing from: {doc_dir}")
            doc_processor = DocumentProcessor(doc_dir)
            processed_document_chunks = doc_processor.process_all_documents()
            doc_chunk_count = len(processed_document_chunks)
            logger.info(f"Document processing complete. Found {doc_chunk_count} chunks.")
            if doc_chunk_count == 0:
                logger.warning("No document chunks were generated. Vector store may be empty or not created.")

            # Shared session model: loaded once on first use, reused afterwards.
            embeddings_model = get_shared_embeddings()
            if not embeddings_model:
                raise RuntimeError("Failed to initialize embedding model.")

            if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
                logger.info(f"Deleting existing vector store before rebuild: {vector_store_path}")
                # Close any open chromadb file handles first (Windows cannot
                # delete files that are still held open).
                release_chroma_clients()
                # Robust deletion logic
                max_retries_del = 5; retry_delay_del = 2; deleted_vs = False
                for attempt_del in range(max_retries_del):
                    try:
                        gc.collect(); rmtree_robust(vector_store_path); time.sleep(0.5)
                        logger.info(f"Vector store deleted on attempt {attempt_del + 1}.")
                        deleted_vs = True; break
                    except OSError as e_del:
                        logger.error(f"Attempt {attempt_del + 1} to delete vector store failed: {e_del}")
                        if attempt_del < max_retries_del - 1: time.sleep(retry_delay_del)
                if not deleted_vs:
                    raise RuntimeError(f"Failed to delete existing vector store at '{vector_store_path}'. Check permissions or locked files.")
            
            os.makedirs(vector_store_path, exist_ok=True)

            if doc_chunk_count > 0:
                logger.info("Creating new vector store...")
                texts_for_store = [doc['page_content'] for doc in processed_document_chunks]
                metadatas_for_store = [doc['metadata'] for doc in processed_document_chunks]
                
                Chroma.from_texts(
                    texts=texts_for_store, embedding=embeddings_model,
                    metadatas=metadatas_for_store, persist_directory=vector_store_path
                ).persist()
                logger.info(f"New vector store created with {doc_chunk_count} items.")
            else:
                # Ensure an empty Chroma DB is initialized if no documents
                # This involves creating with a dummy, then deleting, or initializing empty structure
                Chroma(persist_directory=vector_store_path, embedding_function=embeddings_model).persist()
                logger.info("Empty vector store structure initialized as no documents were processed.")


            app_state["docs_processed"] = True
            app_state["active_documents_directory"] = doc_dir
            # Cheap regex-only pre-scan (no LLM cost) over the chunks already in
            # memory, so the UI knows whether to offer the Bills tab at all
            # without the user having to run a scan first to find out.
            try:
                sample_text = "\n".join(d.get('page_content', '') for d in processed_document_chunks[:200])
                app_state["bills_detected"] = bool(sample_text.strip()) and bills.looks_like_bill(sample_text)
            except Exception as e:
                logger.warning(f"Bills pre-scan failed (non-fatal): {e}")
                app_state["bills_detected"] = False
            return jsonify({
                "status": "success",
                "message": f"Documents processed from '{os.path.basename(doc_dir)}' ({doc_chunk_count} chunks). Vector store ready.",
                "bills_detected": app_state["bills_detected"]
            })

        except Exception as e:
            logger.error(f"Error during initialization from '{doc_dir}': {e}", exc_info=True)
            app_state["docs_processed"] = False
            return make_response(jsonify({"status": "error", "message": f"Initialization failed: {str(e)}"}), 500)
        finally:
            processed_document_chunks = None; embeddings_model = None; gc.collect()
            app_state["is_processing"] = False
            
    @api_instance.route('/models', methods=['GET'])
    def get_ollama_models():
        command = ["ollama", "list"]
        models = {}  # name -> size string (e.g. "4.7 GB")
        try:
            logger.info(f"Executing command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', shell=False)
            output = result.stdout.strip()
            logger.debug(f"'ollama list' output:\n{output}")
            lines = output.split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split()
                    # Columns: NAME  ID  SIZE  UNIT  MODIFIED...  -> size = "SIZE UNIT"
                    if parts and ':' in parts[0]:
                        name = parts[0].strip()
                        size = ""
                        if len(parts) >= 4:
                            size = f"{parts[2]} {parts[3]}"
                        models[name] = size
            model_list = [{"name": n, "size": models[n]} for n in sorted(models.keys())]
            logger.info(f"Found models: {[m['name'] for m in model_list]}")
            return jsonify({"status": "success", "models": model_list})
        except FileNotFoundError:
            msg = "'ollama' command not found. Ensure Ollama is installed and in system PATH."
            logger.error(msg)
            return make_response(jsonify({"status": "error", "message": msg}), 500)
        except subprocess.CalledProcessError as e:
            msg = f"Error executing 'ollama list': {e.stderr or e.stdout or str(e)}"
            logger.error(msg)
            return make_response(jsonify({"status": "error", "message": msg}), 500)
        except Exception as e:
            msg = f"An unexpected error occurred while listing Ollama models: {str(e)}"
            logger.error(msg, exc_info=True)
            return make_response(jsonify({"status": "error", "message": msg}), 500)

    @api_instance.route('/select_model', methods=['POST'])
    def select_and_initialize_model():
        if app_state["is_processing"]:
            return make_response(jsonify({"status": "error", "message": "Busy with another task."}), 409)
        if _bills_scan_running():
            return make_response(jsonify({"status": "error", "message": "A bills scan is running. Wait for it to finish first."}), 409)
        if not app_state["docs_processed"]:
            return make_response(jsonify({"status": "error", "message": "Initialize System (process documents) first."}), 400)

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return make_response(jsonify({"status": "error", "message": "Request body must be a JSON object."}), 400)
        selected_model = data.get('model_name')
        if not selected_model:
            return make_response(jsonify({"status": "error", "message": "Request missing 'model_name'."}), 400)

        app_state["is_processing"] = True
        try:
            logger.info(f"Attempting to select and initialize model: {selected_model}")
            if get_llm_manager(): get_llm_manager().cleanup()
            
            llm_manager_instance = LLMManager(db_directory=config.VECTOR_STORE_DIR)
            embeddings = llm_manager_instance.initialize_embeddings()
            if not embeddings: raise RuntimeError("Failed to initialize embeddings for LLM setup.")
            
            vector_store_path = llm_manager_instance.db_directory
            if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
                raise RuntimeError(f"Vector store not found or empty at {vector_store_path}. Please Initialize System again.")
            
            llm_manager_instance.vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            vs_count = llm_manager_instance.vector_store._collection.count()
            logger.info(f"Loaded vector store with {vs_count} items for model setup.")
            if vs_count == 0: logger.warning("Vector store is empty. QA functionality will be limited.")

            qa_chain = llm_manager_instance.initialize_llm(model_name=selected_model)
            if not qa_chain: raise RuntimeError(f"Failed to initialize QA chain with model: {selected_model}.")
            
            set_llm_manager(llm_manager_instance)
            app_state["selected_ollama_model"] = selected_model
            app_state["insights"] = None
            app_state["bills_job"] = None
            
            logger.info(f"LLM '{selected_model}' and QA Chain initialized successfully.")
            return jsonify({"status": "success", "message": f"System ready. Model: {selected_model}."})

        except Exception as e:
            logger.error(f"Failed to select/initialize model '{selected_model}': {e}", exc_info=True)
            if get_llm_manager(): get_llm_manager().cleanup()
            set_llm_manager(None); app_state["selected_ollama_model"] = None; app_state["is_initialized"] = False
            return make_response(jsonify({"status": "error", "message": f"Model setup failed: {str(e)}"}), 500)
        finally:
            app_state["is_processing"] = False; gc.collect()

    @api_instance.route('/query', methods=['POST'])
    def query_llm():
        # Reject queries while indexing / model-load / cleanup is in flight —
        # they would race with the very state they depend on.
        if app_state["is_processing"]:
            return make_response(jsonify({"status": "error", "message": "Busy with another task."}), 409)

        llm_manager = get_llm_manager()
        if not llm_manager or not app_state["is_initialized"]:
            msg = "System not ready. " + ("Please select a model." if app_state["docs_processed"] else "Please initialize and select a model.")
            return make_response(jsonify({"status": "error", "message": msg}), 400)

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return make_response(jsonify({"status": "error", "message": "Request body must be a JSON object."}), 400)
        question = data.get('question')
        if not question or not question.strip():
            return make_response(jsonify({"status": "error", "message": "Request missing 'question'."}), 400)

        logger.info(f"Query (model '{app_state.get('selected_ollama_model', 'N/A')}'): '{question[:100]}...'") # Log truncated query
        try:
            result = llm_manager.query(question)
            answer = result.get('result', "No answer was generated.")
            source_documents_data = []
            if result.get('source_documents'):
                for doc in result['source_documents']:
                    metadata = {}
                    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict): metadata = doc.metadata
                    elif isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict): metadata = doc['metadata']
                    safe_metadata = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v for k, v in metadata.items()}
                    # Include the retrieved passage text so the UI can display the exact
                    # quote each answer was drawn from (Sources panel citation cards).
                    quote = ""
                    if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                        quote = doc.page_content
                    elif isinstance(doc, dict):
                        quote = doc.get('page_content', "") or ""
                    source_documents_data.append({"metadata": safe_metadata, "quote": quote})

            logger.debug(f"Query successful. Answer length: {len(answer)}, Sources found: {len(source_documents_data)}")
            return jsonify({"status": "success", "answer": answer, "sources": source_documents_data})
        except Exception as e:
            logger.error(f"Error during query processing: {e}", exc_info=True)
            return make_response(jsonify({"status": "error", "message": f"Query processing failed: {str(e)}"}), 500)

    @api_instance.route('/insights', methods=['POST'])
    def get_corpus_insights():
        # Checks (does not take) the processing lock, same as /query: insights
        # generation must not race an in-flight initialize/select_model/cleanup,
        # but it should not block the user from chatting while it runs.
        if app_state["is_processing"]:
            return make_response(jsonify({"status": "error", "message": "Busy with another task."}), 409)
        if app_state["insights_generating"]:
            return make_response(jsonify({"status": "error", "message": "Insights are already being generated."}), 409)

        llm_manager = get_llm_manager()
        if not llm_manager or not app_state["is_initialized"]:
            msg = "System not ready. " + ("Please select a model." if app_state["docs_processed"] else "Please initialize and select a model.")
            return make_response(jsonify({"status": "error", "message": msg}), 400)

        data = request.get_json(silent=True) or {}
        refresh = bool(isinstance(data, dict) and data.get('refresh'))

        cached = app_state.get("insights")
        if cached and not refresh:
            return jsonify({"status": "success", "cached": True, **cached})

        app_state["insights_generating"] = True
        try:
            logger.info(f"Generating corpus insights (model '{app_state.get('selected_ollama_model', 'N/A')}')...")
            result = insights.generate_insights(llm_manager.llm_client, llm_manager.vector_store)
            payload = {
                "summary": result["summary"],
                "entities": result["entities"],
                "entity_source": result["entity_source"],
                "document_count": len(result["documents"]),
                "documents": result["documents"],
                "model": app_state.get("selected_ollama_model"),
            }
            app_state["insights"] = payload
            logger.info(f"Insights generated: {len(payload['entities'])} entities ({payload['entity_source']}), {payload['document_count']} documents.")
            return jsonify({"status": "success", "cached": False, **payload})
        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            return make_response(jsonify({"status": "error", "message": f"Insights generation failed: {str(e)}"}), 500)
        finally:
            app_state["insights_generating"] = False

    @api_instance.route('/bills/scan', methods=['POST'])
    def start_bills_scan():
        if app_state["is_processing"]:
            return make_response(jsonify({"status": "error", "message": "Busy with another task."}), 409)
        if _bills_scan_running():
            return make_response(jsonify({"status": "error", "message": "A bills scan is already running."}), 409)

        llm_manager = get_llm_manager()
        if not llm_manager or not app_state["is_initialized"]:
            msg = "System not ready. " + ("Please select a model." if app_state["docs_processed"] else "Please initialize and select a model.")
            return make_response(jsonify({"status": "error", "message": msg}), 400)

        vector_store_path = config.VECTOR_STORE_DIR
        if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
            return make_response(jsonify({"status": "error", "message": "Vector store not found. Initialize System again."}), 400)

        try:
            # Read-only metadata scan (no embedding model needed), same pattern
            # as /documents, just to enumerate the distinct source paths.
            vs_meta = Chroma(persist_directory=vector_store_path, embedding_function=None)
            results = vs_meta.get(include=['metadatas'])
            del vs_meta
            sources = sorted({m.get('source') for m in results.get('metadatas', []) if isinstance(m, dict) and m.get('source')})
        except Exception as e:
            logger.error(f"Failed to list documents for bills scan: {e}", exc_info=True)
            return make_response(jsonify({"status": "error", "message": f"Failed to list documents: {e}"}), 500)

        if not sources:
            return make_response(jsonify({"status": "error", "message": "No documents found to scan."}), 400)

        job = {"status": "queued", "done": 0, "total": len(sources), "current_file": None, "records": [], "error": None}
        app_state["bills_job"] = job
        # Captured here as direct references so the thread keeps working even
        # if app_state["llm_manager"] is later reassigned; _bills_scan_running()
        # is what actually prevents a concurrent re-index/cleanup from tearing
        # the underlying client out from under it.
        thread = Thread(
            target=_run_bills_scan,
            args=(job, llm_manager.llm_client, llm_manager.vector_store, sources),
            daemon=True,
        )
        thread.start()
        logger.info(f"Bills scan started for {len(sources)} document(s).")
        return jsonify({"status": "success", "message": f"Bills scan started for {len(sources)} document(s).", "total": len(sources)})

    @api_instance.route('/bills', methods=['GET'])
    def get_bills():
        job = app_state.get("bills_job") or {}
        records = job.get("records", [])
        return jsonify({
            "status": "success",
            "job_status": job.get("status", "idle"),
            "done": job.get("done", 0),
            "total": job.get("total", 0),
            "current_file": job.get("current_file"),
            "error": job.get("error"),
            "records": records,
            "forecast": bills.compute_forecast(records),
            "categories": config.BILLS_CATEGORIES,
        })

    _BILLS_EDITABLE_FIELDS = {"vendor", "category", "amount", "due_date", "issue_date"}

    @api_instance.route('/bills/<record_id>', methods=['PATCH'])
    def update_bill(record_id):
        job = app_state.get("bills_job")
        record = next((r for r in job["records"] if r.get("id") == record_id), None) if job else None
        if not record:
            return make_response(jsonify({"status": "error", "message": f"No bill record with id '{record_id}'."}), 404)

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return make_response(jsonify({"status": "error", "message": "Request body must be a JSON object."}), 400)

        for field, value in data.items():
            if field not in _BILLS_EDITABLE_FIELDS:
                continue
            if field == "amount":
                try:
                    record["amount"] = float(value)
                except (TypeError, ValueError):
                    return make_response(jsonify({"status": "error", "message": "amount must be a number."}), 400)
                record["amount_confident"] = True
            elif field == "category":
                if value not in config.BILLS_CATEGORIES:
                    return make_response(jsonify({"status": "error", "message": f"category must be one of {config.BILLS_CATEGORIES}."}), 400)
                record["category"] = value
            elif field in ("due_date", "issue_date"):
                if value:
                    try:
                        date.fromisoformat(value)
                    except ValueError:
                        return make_response(jsonify({"status": "error", "message": f"{field} must be an ISO date (YYYY-MM-DD)."}), 400)
                    record[field] = value
                    record[f"{field}_confident"] = True
                else:
                    record[field] = None
                    record[f"{field}_confident"] = False
            elif field == "vendor":
                record["vendor"] = (str(value).strip() or "Unknown")

        return jsonify({"status": "success", "record": record})

    @api_instance.route('/bills/<record_id>', methods=['DELETE'])
    def delete_bill(record_id):
        job = app_state.get("bills_job")
        if not job:
            return make_response(jsonify({"status": "error", "message": "No bills scan has been run."}), 400)
        before = len(job.get("records", []))
        job["records"] = [r for r in job.get("records", []) if r.get("id") != record_id]
        if len(job["records"]) == before:
            return make_response(jsonify({"status": "error", "message": f"No bill record with id '{record_id}'."}), 404)
        return jsonify({"status": "success", "message": "Bill record removed."})

    @api_instance.route('/documents', methods=['GET'])
    def get_documents_list_from_store():
        vector_store_path = config.VECTOR_STORE_DIR
        if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
            logger.info("Document list requested, but vector store not found or is empty.")
            return jsonify({"status": "success", "total_chunk_count": 0, "unique_source_count": 0, "documents": [], "active_directory": app_state.get("active_documents_directory")})

        doc_list_output = []; total_chunks = 0
        try:
            logger.info("Loading vector store metadata for document list...")
            # Reading metadata does not require an embedding model.
            vector_store_instance = Chroma(persist_directory=vector_store_path, embedding_function=None)
            results = vector_store_instance.get(include=['metadatas'])
            del vector_store_instance

            all_metadatas = results.get('metadatas', [])
            total_chunks = len(all_metadatas)
            
            source_info_map = {}
            for meta in all_metadatas:
                if not isinstance(meta, dict): continue
                source_path = meta.get('source', 'Unknown Source')
                page_num = meta.get('page') 
                if source_path not in source_info_map:
                    source_info_map[source_path] = {"source": os.path.basename(source_path), "pages": set(), "chunk_count": 0, "full_path": source_path}
                source_info_map[source_path]["chunk_count"] += 1
                if page_num is not None and str(page_num).isdigit(): source_info_map[source_path]["pages"].add(int(page_num))
            
            for data in source_info_map.values():
                data['pages'] = ", ".join(map(str, sorted(list(data['pages'])))) if data['pages'] else "N/A"
                doc_list_output.append(data)
            
            doc_list_output.sort(key=lambda x: x['source'])
            logger.info(f"Returning {len(doc_list_output)} unique sources from {total_chunks} total chunks.")
        except Exception as e:
            logger.error(f"Error reading document list from vector store: {e}", exc_info=True)
            return make_response(jsonify({"status": "error", "message": f"Failed to list documents: {e}"}), 500)
        return jsonify({"status": "success", "total_chunk_count": total_chunks, "unique_source_count": len(doc_list_output), "documents": doc_list_output, "active_directory": app_state.get("active_documents_directory")})

    @api_instance.route('/status', methods=['GET'])
    def get_system_status():
        vector_store_path = config.VECTOR_STORE_DIR
        vector_store_exists_and_populated = os.path.exists(vector_store_path) and bool(os.listdir(vector_store_path))
        
        status_data = {
            "llm_initialized": app_state["is_initialized"],
            "docs_processed": app_state["docs_processed"],
            "is_processing": app_state["is_processing"],
            "selected_ollama_model": app_state.get("selected_ollama_model"),
            "initial_config_documents_directory": app_state.get("documents_directory"), # Path from config or --docs
            "active_documents_directory": app_state.get("active_documents_directory"), # Path used after UI init
            "vector_store_exists": vector_store_exists_and_populated,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "ollama_base_url": config.OLLAMA_BASE_URL,
            "bills_detected": app_state.get("bills_detected", False),
            # "is_initial_docs_path_fixed": app_state.get("initial_docs_path_is_fixed", False) # Removed this flag
        }
        logger.debug(f"Reporting system status: {status_data}")
        return jsonify({"status": "success", "system_status": status_data})

    # --- MODIFIED: New endpoint for Browse folders ---
    @api_instance.route('/browse-folder', methods=['GET'])
    def browse_for_folder_route():
        if not TKINTER_AVAILABLE:
            logger.error("Browse folder endpoint called, but tkinter is not available on the server.")
            return make_response(jsonify({
                "status": "error", 
                "message": "Folder Browse capability (tkinter) is not available on the server."
            }), 501)

        root_tk = None
        try:
            logger.info("Attempting to open folder selection dialog via API call.")
            root_tk = tk.Tk()
            root_tk.withdraw()
            root_tk.attributes("-topmost", True)

            initial_dir_path = app_state.get("active_documents_directory") or \
                               app_state.get("documents_directory") or \
                               os.path.expanduser("~")
            if not os.path.isdir(initial_dir_path): # Fallback if path isn't a dir
                initial_dir_path = os.path.expanduser("~")

            folder_path = filedialog.askdirectory(
                parent=root_tk, # Good practice to set parent
                title="Select Documents Directory",
                initialdir=initial_dir_path
            )
            
            # Ensuring destroy is called
            if root_tk:
                root_tk.destroy()
                root_tk = None # Avoid trying to destroy it again in finally if it worked

            if folder_path:
                logger.info(f"Folder selected via dialog: {folder_path}")
                return jsonify({"status": "success", "path": folder_path})
            else:
                logger.info("Folder selection was cancelled by the user.")
                return jsonify({"status": "cancelled", "message": "Folder selection cancelled."})
        except Exception as e:
            logger.error(f"Error opening folder dialog: {e}", exc_info=True)
            if root_tk:
                try: root_tk.destroy()
                except: pass # Ignore errors during cleanup destroy
            return make_response(jsonify({
                "status": "error", 
                "message": f"An error occurred while trying to open the folder dialog: {str(e)}"
            }), 500)
        finally:
            # Double ensure root_tk is destroyed if it still exists (e.g. exception before destroy)
            if root_tk:
                try: root_tk.destroy()
                except: pass


    # --- Folder file listing (used by the indexing view to show real filenames) ---
    @api_instance.route('/list-folder', methods=['GET'])
    def list_folder_files():
        doc_dir = request.args.get('path') or app_state.get("active_documents_directory") \
            or app_state.get("documents_directory")
        if not doc_dir or not os.path.isdir(doc_dir):
            return make_response(jsonify({"status": "error", "message": f"Not a directory: {doc_dir}"}), 400)

        def file_type(ext):
            ext = ext.lower()
            if ext == '.pdf':
                return 'pdf'
            if ext == '.docx':
                return 'doc'
            if ext == '.md':
                return 'md'
            return 'txt'

        files = []
        try:
            # Top level only — must mirror DocumentProcessor.process_all_documents
            # (os.listdir, directories skipped) so this view never lists a file
            # that indexing won't actually process.
            supported = tuple(config.SUPPORTED_EXTENSIONS)
            for name in os.listdir(doc_dir):
                if not os.path.isfile(os.path.join(doc_dir, name)):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in supported:
                    files.append({"name": name, "type": file_type(ext)})
            files.sort(key=lambda f: f["name"].lower())
        except OSError as e:
            return make_response(jsonify({"status": "error", "message": f"Cannot read directory: {e}"}), 403)
        return jsonify({"status": "success", "directory": doc_dir, "files": files, "count": len(files)})

    # --- Individual System Stats Endpoints ---
    @api_instance.route('/stats/cpu', methods=['GET'])
    def get_cpu_stats():
        try:
            cpu_usage = psutil.cpu_percent(interval=0.5) # Reduced interval for quicker response
            return jsonify({"status": "success", "cpu_usage": cpu_usage})
        except Exception as e:
            logger.warning(f"Could not fetch CPU stats: {e}") # Changed to warning
            return jsonify({"status": "success", "cpu_usage": 0, "error": str(e)}) # Return 0 on error


    @api_instance.route('/stats/ram', methods=['GET'])
    def get_ram_stats():
        try:
            ram = psutil.virtual_memory()
            return jsonify({
                "status": "success",
                "ram_total_gb": round(ram.total / (1024**3), 2),
                "ram_available_gb": round(ram.available / (1024**3), 2),
                "ram_used_gb": round(ram.used / (1024**3), 2),
                "ram_usage_percent": ram.percent
            })
        except Exception as e:
            logger.warning(f"Could not fetch RAM stats: {e}")
            return jsonify({"status": "success", "ram_usage_percent": 0, "error": str(e)})


    @api_instance.route('/stats/disk', methods=['GET'])
    def get_disk_stats():
        disk_path_checked = ""
        try:
            # Check preferred path first, then fallback
            preferred_path = app_state.get("active_documents_directory") or app_state.get("documents_directory")
            if preferred_path and os.path.exists(os.path.dirname(preferred_path) or preferred_path): # Check parent or self
                 disk_path_checked = os.path.dirname(preferred_path) or preferred_path
                 if not os.path.isdir(disk_path_checked): # if preferred_path is a file path, get its dir
                     disk_path_checked = os.path.dirname(disk_path_checked)
            elif platform.system() == "Windows" and os.path.exists("C:\\"):
                disk_path_checked = "C:\\"
            elif os.path.exists("/"):
                disk_path_checked = "/"
            else:
                disk_path_checked = os.getcwd()
            
            if not disk_path_checked or not os.path.isdir(disk_path_checked): # Final fallback if all else fails
                 disk_path_checked = os.getcwd()

            disk = psutil.disk_usage(disk_path_checked)
            return jsonify({
                "status": "success",
                "disk_path_checked": disk_path_checked,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_usage_percent": disk.percent
            })
        except Exception as e:
            logger.warning(f"Could not fetch Disk stats for '{disk_path_checked}': {e}")
            return jsonify({"status": "success", "disk_usage_percent": 0, "error": str(e), "disk_path_checked": disk_path_checked or "unknown"})


    @api_instance.route('/cleanup', methods=['POST'])
    def cleanup_route():
        if app_state["is_processing"]:
            return make_response(jsonify({"status": "error", "message": "Busy with another task, cannot cleanup."}), 409)
        if _bills_scan_running():
            return make_response(jsonify({"status": "error", "message": "A bills scan is running. Wait for it to finish first."}), 409)

        app_state["is_processing"] = True
        try:
            cleanup_internal()
            return jsonify({"status": "success", "message": "Resources cleaned." })
        except Exception as e:
            logger.error(f"Cleanup endpoint error: {e}", exc_info=True)
            return make_response(jsonify({"status": "error", "message": f"Cleanup failed: {str(e)}"}), 500)
        finally:
            app_state["is_processing"] = False

    @api_instance.route('/shutdown', methods=['POST'])
    def shutdown_route():
        logger.warning("Received shutdown request. Preparing to terminate server.")
        try:
            cleanup_internal() # Attempt cleanup before shutdown
        except Exception as e:
            logger.error(f"Error during pre-shutdown cleanup: {e}", exc_info=True)

        def do_shutdown_thread():
            time.sleep(0.1) # Short delay for response to be sent
            logger.info("Executing server shutdown (os._exit)...")
            os._exit(0)

        shutdown_thread = Thread(target=do_shutdown_thread)
        shutdown_thread.daemon = True
        shutdown_thread.start()
        return jsonify({"status": "success", "message": "Shutdown initiated. Server will terminate shortly."})

    # Expose for callers outside this closure (e.g. the direct-run atexit handler).
    api_instance.cleanup_internal = cleanup_internal
    return api_instance
# --- End of create_api_app ---

# --- Create the Flask app instance at the module level ---
api_app = create_api_app()

# --- Logging and Exit Handling (Only for direct execution of api_server.py) ---
def direct_run_setup_logging():
    log_dir = getattr(config, 'LOG_DIR', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').upper()
    log_level_val = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level_val,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{config.APP_NAME.lower()}_api_direct.log")),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logger.info("Logging setup for direct execution of api_server.py.")


def direct_run_cleanup_on_exit(): # This is the atexit handler for direct run
    logger.info("Direct run: Initiating cleanup before exit...")
    try:
        # Call the same cleanup_internal as routes use for consistency
        api_app.cleanup_internal()
        logger.info("Direct run: Cleanup on exit completed.")
    except Exception as e:
        logger.error(f"Error during direct run cleanup on exit: {e}", exc_info=True)

if __name__ == '__main__':
    direct_run_setup_logging()
    atexit.register(direct_run_cleanup_on_exit)
    reset_vector_store_on_startup()

    logger.info(f"Starting {config.APP_NAME} API server (DIRECT EXECUTION of api_server.py)...")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {sys.version}")
    if not TKINTER_AVAILABLE:
        logger.warning("tkinter module is not available. The '/browse-folder' API endpoint will not function.")


    try:
        api_host = getattr(config, 'API_HOST', 'localhost')
        api_port = getattr(config, 'API_PORT', 5000)
        api_debug = getattr(config, 'API_DEBUG_MODE', True)
        
        api_app.run(
            host=api_host,
            port=api_port,
            debug=api_debug,
            use_reloader=False # Reloader can cause issues with atexit and resource cleanup in some cases
        )
    except Exception as e:
        logger.critical(f"Failed to start API server directly: {e}", exc_info=True)
        sys.exit(1)