# document_processing/core_processor.py
# version : 1.2.0
# File: 9
# This file will contain the main DocumentProcessor class.



import os
import logging
from typing import List, Dict, Any

# LangChain components for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
# from langchain.schema import Document # Not directly used for output format, but good to be aware of

# Import from other modules in this package
from .config_handler import config, logger # Use the logger from config_handler
from .image_utils import ImageAnalyzer
from .text_chunker import SemanticChunker
from .content_handler import ContentBlock, extract_content_structure # Assuming ContentBlock is used

# Import specific file processors
from .pdf_processor import process_pdf_file
from .docx_processor import process_docx_file
from .txt_processor import process_txt_file


class DocumentProcessor:
    def __init__(self, documents_directory_path: str):
        if not os.path.isdir(documents_directory_path):
            logger.error(f"Initialization failed: Directory not found at '{documents_directory_path}'")
            raise ValueError(f"Directory not found: {documents_directory_path}")
        
        self.dir_path = documents_directory_path
        
        # Initialize components (these will log their own status)
        self.image_analyzer = ImageAnalyzer()
        # Semantic chunking loads a second transformer model and embeds every
        # paragraph — only pay that cost if it is explicitly enabled in config.
        if getattr(config, 'ENABLE_SEMANTIC_CHUNKING', False):
            self.semantic_chunker = SemanticChunker() # Uses config.SEMANTIC_MODEL internally
        else:
            self.semantic_chunker = None

        # Initialize text splitters from LangChain using config values
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.token_splitter = TokenTextSplitter( # Used as fallback or specific strategy
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
            # model_name can be specified if using a tokenizer tied to a model, e.g., for tiktoken
            # For generic token splitting, it might use a default or character-based split if model_name omitted
        )
        
        # Determine if semantic splitting should be the default strategy
        self.use_semantic_splitting_default = bool(self.semantic_chunker and self.semantic_chunker.is_available())

        logger.info(f"DocumentProcessor initialized for directory: {self.dir_path}.")
        logger.info(f"Default text splitting strategy: {'Semantic' if self.use_semantic_splitting_default else 'Recursive'}.")
        self._log_feature_availability()


    def _log_feature_availability(self):
        # Log status of optional features based on component availability
        # (Mirrors the logging from original document_processor.py)
        logger.info(f"--- DocumentProcessor Feature Availability ---")
        # PDFPlumber/PDF2Image availability is implicitly handled by pdf_processor module
        logger.info(f"PDF Processing: (pdfplumber: {bool(getattr(config, 'PDFPLUMBER_AVAILABLE', True))}, pdf2image: {bool(getattr(config, 'PDF2IMAGE_AVAILABLE', True))})") # Approximation
        logger.info(f"OCR (via ImageAnalyzer/Pytesseract): {self.image_analyzer._ocr_available if self.image_analyzer else 'Unknown'}")
        logger.info(f"Semantic Splitting (via SemanticChunker): {self.semantic_chunker.is_available() if self.semantic_chunker else 'DISABLED (config)'}")
        # PDFMiner for forms is handled in pdf_processor
        logger.info(f"Graph/Image Analysis (via ImageAnalyzer/OpenCV): {self.image_analyzer.is_graph_available() if self.image_analyzer else 'Unknown'}")
        logger.info(f"Table Summarization in PDFs (Configurable): {'ENABLED' if config.ENABLE_TABLE_SUMMARIZATION else 'DISABLED'}")
        logger.info(f"Form Extraction in PDFs (Configurable): {'ENABLED' if config.ENABLE_FORM_EXTRACTION else 'DISABLED'}")
        logger.info(f"---------------------------------------------")


    def _split_text_into_documents(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a larger text into smaller document chunks suitable for a vector store.
        Each chunk is a dictionary with 'page_content' and 'metadata'.
        """
        if not text or not text.strip():
            return []

        chunks_text: List[str] = []
        split_method_used = 'unknown'

        # Try semantic splitting if explicitly enabled and available
        if self.use_semantic_splitting_default and self.semantic_chunker and self.semantic_chunker.is_available():
            try:
                chunks_text = self.semantic_chunker.semantic_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                split_method_used = 'semantic'
                if not chunks_text and text.strip(): # If semantic split returns nothing for non-empty text
                    logger.warning(f"Semantic split returned no chunks for non-empty text (source: {metadata.get('source', 'N/A')}, page: {metadata.get('page', 'N/A')}). Falling back.")
                    # Force fallback by clearing chunks_text or setting a flag
                    chunks_text = []
            except Exception as sem_e:
                logger.error(f"Semantic splitting failed: {sem_e}. Falling back to recursive splitter.", exc_info=True)
                chunks_text = [] # Ensure fallback

        # Default strategy: recursive character splitting (fast, respects
        # paragraph/sentence boundaries, preserves document order).
        if not chunks_text:
            try:
                chunks_text = self.recursive_splitter.split_text(text)
                split_method_used = 'recursive'
                if not chunks_text and text.strip():
                    logger.warning(f"Recursive splitter returned no chunks for non-empty text (source: {metadata.get('source')}, page: {metadata.get('page')}). Trying token splitter.")
                    chunks_text = []
            except Exception as rec_e:
                logger.error(f"Recursive splitting failed: {rec_e}. Falling back to token splitter.", exc_info=True)
                chunks_text = []

        # Last resort: token-based splitting
        if not chunks_text:
            try:
                chunks_text = self.token_splitter.split_text(text)
                split_method_used = 'token'
                if not chunks_text and text.strip():
                     logger.warning(f"Token splitter also returned no chunks for non-empty text (source: {metadata.get('source')}, page: {metadata.get('page')}). This text may be lost.")
            except Exception as tok_e:
                 logger.error(f"Token splitting also failed: {tok_e}. Text from this section might be lost.", exc_info=True)
                 return [] # No chunks could be made


        # Format chunks into the dictionary structure expected by vector stores
        output_documents: List[Dict[str, Any]] = []
        for i, chunk_str in enumerate(chunks_text):
            chunk_meta = metadata.copy() # Start with base metadata from the source (e.g., file path, page number)
            chunk_meta['chunk_index'] = i
            chunk_meta['split_method'] = split_method_used
            
            # Add context from headings if possible (from original logic)
            # This requires extract_content_structure to identify headings effectively.
            content_blocks_in_chunk = extract_content_structure(chunk_str) # Analyze the chunk's own structure
            headings_in_chunk = [cb.text for cb in content_blocks_in_chunk if cb.content_type == 'heading']
            if headings_in_chunk:
                # Use last few headings from the chunk as its immediate context
                chunk_meta['context_headings'] = ' > '.join(headings_in_chunk[-3:]) 
            
            # Store content types present in the chunk
            if content_blocks_in_chunk:
                chunk_meta['content_types_in_chunk'] = ", ".join(sorted(list(set(cb.content_type for cb in content_blocks_in_chunk))))
            else:
                chunk_meta['content_types_in_chunk'] = "unknown"

            output_documents.append({'page_content': chunk_str, 'metadata': chunk_meta})
            
        return output_documents


    def process_all_documents(self) -> List[Dict[str, Any]]:
        """
        Processes all supported files in the initialized directory.
        Returns a list of document chunks, each as a dictionary.
        """
        all_processed_docs: List[Dict[str, Any]] = []
        logger.info(f"Starting document processing in directory: {self.dir_path}")

        try:
            for filename in os.listdir(self.dir_path):
                file_path = os.path.join(self.dir_path, filename)
                if not os.path.isfile(file_path):
                    continue # Skip directories

                _, file_extension = os.path.splitext(filename.lower())
                
                # Use SUPPORTED_EXTENSIONS from the loaded config object
                if file_extension not in config.SUPPORTED_EXTENSIONS:
                    logger.debug(f"Skipping unsupported file type: {filename}")
                    continue

                logger.info(f"--- Processing file: {filename} ---")
                file_specific_docs: List[Dict[str, Any]] = []
                try:
                    if file_extension == '.pdf':
                        file_specific_docs = process_pdf_file(file_path, self.image_analyzer, self._split_text_into_documents)
                    elif file_extension in ['.docx', '.doc']: # .doc might need additional handling or conversion
                        file_specific_docs = process_docx_file(file_path, self.image_analyzer, self._split_text_into_documents)
                    elif file_extension in ['.txt', '.md']:
                        file_specific_docs = process_txt_file(file_path, self._split_text_into_documents)
                    # Add handlers for other supported types if necessary
                    
                    all_processed_docs.extend(file_specific_docs)
                    logger.info(f"Finished processing {filename}, generated {len(file_specific_docs)} document chunks.")

                except Exception as e_file_proc:
                    logger.error(f"Failed to process file {filename}: {e_file_proc}", exc_info=True)
                    # Optionally, add a placeholder document indicating failure for this file
                    # all_processed_docs.append({'page_content': f"Error processing file: {filename}", 
                    #                            'metadata': {'source': file_path, 'error': str(e_file_proc)}})
            
        except OSError as e_dir_read:
            logger.error(f"Directory read error for {self.dir_path}: {e_dir_read}", exc_info=True)
            # Depending on desired behavior, either raise or return what's processed so far
            raise # Or return all_processed_docs

        logger.info(f"Overall document processing finished for directory {self.dir_path}. Total chunks generated: {len(all_processed_docs)}")
        
        # Sanitize metadata (from original logic, important for vector stores)
        sanitized_docs_final: List[Dict[str, Any]] = []
        for i, doc_item_dict in enumerate(all_processed_docs):
            if not isinstance(doc_item_dict, dict) or \
               'page_content' not in doc_item_dict or \
               'metadata' not in doc_item_dict:
                logger.warning(f"Skipping invalid document item at index {i} (not a dict or missing keys): {str(doc_item_dict)[:150]}")
                continue
            
            current_metadata = doc_item_dict.get('metadata', {})
            if isinstance(current_metadata, dict):
                cleaned_meta = {}
                for k, v in current_metadata.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        cleaned_meta[k] = v
                    elif isinstance(v, (list, tuple)): # Ensure list/tuple items are also simple types
                        try:
                            cleaned_meta[k] = [item if isinstance(item, (str, int, float, bool, type(None))) else str(item) for item in v]
                        except Exception as list_conv_e:
                            logger.debug(f"Could not convert list item to string for metadata key '{k}' in {doc_item_dict.get('metadata', {}).get('source', 'N/A')}: {list_conv_e}. Stringifying whole list.")
                            cleaned_meta[k] = str(v) # Fallback: stringify the whole list/tuple
                    else: # For other complex types, try to convert to string
                        try:
                            cleaned_meta[k] = str(v)
                        except Exception as str_conv_e:
                            logger.debug(f"Could not convert metadata value to string for key '{k}' in {doc_item_dict.get('metadata', {}).get('source', 'N/A')}: {str_conv_e}. Skipping key.")
                            # Do not add the key if conversion fails, or add a placeholder like 'conversion_failed'
                doc_item_dict['metadata'] = cleaned_meta
                sanitized_docs_final.append(doc_item_dict)
            else:
                # If metadata is not a dict, create a minimal one
                logger.warning(f"Document item at index {i} has non-dict metadata: {type(current_metadata)}. Source: {doc_item_dict.get('metadata', {}).get('source', 'N/A')}. Creating minimal metadata.")
                source_val = 'unknown_source_malformed_meta'
                if hasattr(current_metadata, 'get'): # Try to get source if metadata was some object with .get
                    source_val = str(current_metadata.get('source', 'unknown_source_in_malformed_object_meta'))
                elif current_metadata is not None:
                    source_val = str(current_metadata) # As a last resort

                doc_item_dict['metadata'] = {'source': source_val, 'original_metadata_type': str(type(current_metadata))}
                sanitized_docs_final.append(doc_item_dict)
                
        return sanitized_docs_final