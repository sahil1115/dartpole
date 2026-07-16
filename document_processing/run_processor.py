# document_processing/run_processor.py
# Version 2.0.0
# This script is the main entry point for processing documents.
# It includes pre-flight checks and robust error handling to prevent common
# multiprocessing and initialization issues.

import os
import sys
from enhanced_core_processor import EnhancedDocumentProcessor

def pre_flight_checks():
    """
    Attempts to initialize components that may require network access for setup.
    This helps ensure that any one-time downloads happen before multiprocessing starts.
    """
    print("--- Running Pre-flight Checks ---")
    print("This may take a moment as libraries might be downloading required models for the first time.")
    
    try:
        # Attempt to trigger the download for the sentence-transformer model if needed.
        # Replace 'all-MiniLM-L6-v2' if your SemanticChunker uses a different model.
        print("Initializing sentence-transformer model...")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence-transformer model is available.")

        # You can add other checks here, e.g., for pytesseract language data if needed.

        print("--- Pre-flight Checks Passed ---")
        return True
    except ImportError as ie:
        print(f"\n❌ Pre-flight Check Failed: A required library is missing.")
        print(f"Error: {ie}")
        print("Please install it, e.g., 'pip install sentence-transformers'.")
        return False
    except Exception as e:
        print(f"\n❌ Pre-flight Check Failed: Could not download or initialize a required component.")
        print(f"Error: {e}")
        print("Please ensure you have a stable internet connection and that your firewall is not blocking Python.")
        return False

def run_document_processing(directory_path: str, filter_for_english: bool):
    """
    Initializes and runs the document processor on a given directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The specified directory does not exist: '{directory_path}'")
        return

    print("\n--- Starting Document Processing ---")
    print(f"Target Directory: {directory_path}")
    print(f"Filter for English in Scanned PDFs: {'Yes' if filter_for_english else 'No'}")
    print("-" * 35)

    try:
        processor = EnhancedDocumentProcessor(
            documents_directory_path=directory_path,
            english_only_filter=filter_for_english
        )

        all_docs = processor.process_all_documents()

        print("\n--- Processing Complete ---")
        if all_docs:
            print(f"Successfully generated a total of {len(all_docs)} document chunks.")
            # Optional: Print a summary
            # for i, doc in enumerate(all_docs):
            #     content = doc.get('page_content', 'No Content').strip()
            #     source = doc.get('metadata', {}).get('source', 'Unknown Source')
            #     print(f"\nChunk {i+1} from '{os.path.basename(source)}'")
            #     print(f"{content[:150]}...")
        else:
            print("No documents were processed or no content could be extracted.")
        print("-" * 35)

    except ValueError as ve:
        print(f"Initialization Error: {ve}")
    except Exception as e:
        # This will now provide a more specific error message.
        print(f"\n--- An Unexpected Error Occurred During Processing ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("\nThis might be due to a corrupted file, a library issue, or a memory problem.")
        print("Please check the file being processed when the error occurred.")


# --- Main Execution Guard ---
# This is the most important part of the script for preventing multiprocessing errors.
# It ensures that the processing code only runs when the script is executed directly.
if __name__ == '__main__':
    # Add this check to prevent issues on systems that fork processes.
    if sys.version_info >= (3, 8) and sys.platform == 'win32':
         # This is required for some multiprocessing libraries to work correctly on Windows
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            # The start method might already be set, which is fine.
            pass

    # --- Configuration ---
    # IMPORTANT: Update this to a valid path on your system.
    DOCS_DIRECTORY = os.path.expanduser('~/path/to/your/documents')
    FILTER_SCANNED_FOR_ENGLISH = True
    # --- End of Configuration ---

    # 1. Run pre-flight checks first to handle downloads.
    if pre_flight_checks():
        # 2. If checks pass, run the main processing function.
        run_document_processing(
            directory_path=DOCS_DIRECTORY,
            filter_for_english=FILTER_SCANNED_FOR_ENGLISH
        )
    else:
        print("\nHalting execution due to failed pre-flight checks.")