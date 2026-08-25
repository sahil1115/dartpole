# src/main.py
import os
import sys
import webbrowser
import argparse
import logging
from threading import Timer
from flask import Flask, send_from_directory
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

# Import config and the API app instance
import config
from api_server import api_app, app_state, reset_vector_store_on_startup

# Setup logger (Configure root logger once)
# Ensure log directory from config exists
log_dir_from_config = getattr(config, 'LOG_DIR', 'logs') # Get LOG_DIR, default to 'logs'
os.makedirs(log_dir_from_config, exist_ok=True)

logging.basicConfig(
    level=config.LOG_LEVEL, 
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    handlers=[ # Explicitly add handlers if needed, e.g. file handler
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir_from_config, f"{config.APP_NAME.lower()}_main.log"))
    ],
    force=True
)
logger = logging.getLogger(config.APP_NAME + ".Main")
logging.getLogger("werkzeug").setLevel(logging.WARNING) # Quieten Werkzeug


# Define the main Flask app for serving the UI
ui_app = Flask(__name__, static_folder=None)

@ui_app.route('/')
def index():
    logger.debug("Serving index.html")
    ui_dir = os.path.dirname(os.path.abspath(__file__)) # Use abspath for robustness
    return send_from_directory(ui_dir, 'index.html')

@ui_app.route('/<path:filename>')
def serve_static(filename):
    if ".." in filename or filename.startswith(("/", "\\")):
        logger.warning(f"Attempted invalid static file access: {filename}")
        return "Invalid path", 404
    logger.debug(f"Serving static file: {filename}")
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        return send_from_directory(ui_dir, filename)
    except Exception: # Catch generic exception for send_from_directory
        logger.error(f"Static file not found or error serving: {filename}", exc_info=False) # Keep log less verbose
        return "File not found", 404

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description=f'{config.APP_NAME} - Local RAG System')
    parser.add_argument('--model', type=str, default=None,
                        help=f'Ollama model override (primarily for non-UI use or future extension)')
    parser.add_argument('--docs', type=str, default=config.DEFAULT_DOCUMENTS_DIR,
                        help=f'Initial path for documents directory (can be changed in UI) (default: {config.DEFAULT_DOCUMENTS_DIR})')
    parser.add_argument('--port', type=int, default=config.UI_PORT,
                        help=f'Port for the UI server (default: {config.UI_PORT})')
    parser.add_argument('--api-port', type=int, default=config.API_PORT,
                        help=f'Internal port for backend API if run separately (default: {config.API_PORT})')
    parser.add_argument('--no-browser', action='store_true', default=not config.AUTO_OPEN_BROWSER,
                        help='Prevent automatic browser opening.')
    parser.add_argument('--log-level', type=str, default=config.LOG_LEVEL,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help=f'Logging level (default: {config.LOG_LEVEL})')

    args = parser.parse_args()

    log_level_arg = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_arg) # Set root logger level
    for handler in logging.getLogger().handlers: handler.setLevel(log_level_arg) # Update handlers
    logger.info(f"Log level set to {args.log_level}")

    if args.model:
         logger.warning(f"Command line --model='{args.model}' provided. Note: Ollama model is typically selected via UI.")
    
    return args

# --- Main Execution ---
if __name__ == '__main__':
    args = parse_arguments()

    # Set the initial documents_directory in app_state from args (which defaults to config.DEFAULT_DOCUMENTS_DIR)
    # This value will be used by the UI as the initial content for the docsPathInput field.
    app_state["documents_directory"] = args.docs
    # The "initial_docs_path_is_fixed" flag is removed from app_state as the UI browse button allows changing it.

    initial_docs_dir_to_check = args.docs
    if not os.path.exists(initial_docs_dir_to_check):
        logger.info(f"Initial documents directory specified does not exist: {initial_docs_dir_to_check}. Please specify a valid path in the UI.")
    elif not os.path.isdir(initial_docs_dir_to_check):
         logger.error(f"The initial path specified for documents is not a directory: {initial_docs_dir_to_check}. Please specify a valid path in the UI.")

    vector_store_dir = config.VECTOR_STORE_DIR
    # Clean slate on every start: wipe any leftover store from a previous session.
    reset_vector_store_on_startup()
    if not os.path.isdir(vector_store_dir):
        logger.error(f"Failed to create/access vector store directory: {vector_store_dir}.")
        sys.exit(1)
    logger.info(f"Vector store directory is: {vector_store_dir} (session-only, cleared on start and shutdown)")

    application = DispatcherMiddleware(ui_app.wsgi_app, {
        '/api': api_app.wsgi_app
    })

    def open_browser():
        url = f'http://{config.UI_HOST}:{args.port}'
        try: webbrowser.open(url); logger.info(f"Attempted browser open at {url}")
        except Exception as e: logger.warning(f"Could not open browser: {e}")

    print("\n" + "=" * 80)
    print(f" Starting {config.APP_NAME} ".center(80, "="))
    print("=" * 80)
    print(f" Initial Documents Dir: {args.docs} (Changeable in UI)")
    print(f" Vector Store Dir    : {config.VECTOR_STORE_DIR}")
    print(f" (Ollama Model selected via UI after initialization)")
    print(f" UI Server Running at: http://{config.UI_HOST}:{args.port}")
    print(f" API Endpoints under : http://{config.UI_HOST}:{args.port}/api") # Corrected path for API
    print(f" Log Level           : {args.log_level}")
    print("=" * 80)

    if not args.no_browser:
        logger.info("Opening browser in ~1.5 seconds...")
        Timer(1.5, open_browser).start()

    try:
        logger.info(f"Starting Werkzeug development server on {config.UI_HOST}:{args.port}")
        run_simple(
            hostname=config.UI_HOST,
            port=args.port,
            application=application,
            use_reloader=config.UI_RELOADER, # Typically False for production or when debugging complex states
            use_debugger=config.UI_DEBUG_MODE,
            threaded=True # Flask default for run_simple, good for handling multiple requests like API + UI
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
             logger.error(f"Port {args.port} is already in use. Stop other process or use --port argument.")
        else: logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error running server: {e}", exc_info=True)
        sys.exit(1)
    finally:
         logger.info("Server shutting down.")