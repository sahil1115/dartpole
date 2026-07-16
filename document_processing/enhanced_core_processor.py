# document_processing/enhanced_core_processor.py
# version: 1.4.0
# Enhanced core processor with better scanned PDF support and configurable English-only filtering

import os
import logging
from typing import List, Dict, Any

# LangChain components for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

# Import from other modules in this package
from .config_handler import config, logger
from .image_utils import ImageAnalyzer
from .text_chunker import SemanticChunker
from .content_handler import ContentBlock, extract_content_structure

# Import specific file processors
from .pdf_processor import process_pdf_file
from .scanned_pdf_processor import process_scanned_pdf_file, ScannedPDFProcessor
from .docx_processor import process_docx_file
from .txt_processor import process_txt_file


class EnhancedDocumentProcessor:
    """Enhanced document processor with better support for scanned PDFs."""
    
    def __init__(self, documents_directory_path: str, english_only_filter: bool = False):
        """
        Initializes the processor.

        Args:
            documents_directory_path (str): The path to the directory containing documents.
            english_only_filter (bool): If True, filters content from scanned PDFs to keep only English text.
        """
        if not os.path.isdir(documents_directory_path):
            logger.error(f"Initialization failed: Directory not found at '{documents_directory_path}'")
            raise ValueError(f"Directory not found: {documents_directory_path}")
        
        self.dir_path = documents_directory_path
        self.english_only_filter = english_only_filter
        
        # Initialize components
        self.image_analyzer = ImageAnalyzer()
        self.semantic_chunker = SemanticChunker()
        self.scanned_pdf_processor = ScannedPDFProcessor()

        # Initialize text splitters from LangChain using config values
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.token_splitter = TokenTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        self.use_semantic_splitting_default = self.semantic_chunker.is_available()
        
        logger.info(f"EnhancedDocumentProcessor initialized for directory: {self.dir_path}")
        logger.info(f"Filter for English-only text in scanned PDFs: {'Enabled' if self.english_only_filter else 'Disabled'}")
        logger.info(f"Default text splitting strategy: {'Semantic' if self.use_semantic_splitting_default else 'Token/Recursive'}")
        self._log_feature_availability()

    def _log_feature_availability(self):
        """Log status of optional features based on component availability."""
        logger.info(f"--- EnhancedDocumentProcessor Feature Availability ---")
        logger.info(f"PDF Processing (Regular): Available")
        logger.info(f"PDF Processing (Scanned/OCR): {'Available' if self.scanned_pdf_processor.ocr_available else 'Unavailable (missing pytesseract/pdf2image)'}")
        logger.info(f"OCR (via ImageAnalyzer/Pytesseract): {self.image_analyzer._ocr_available if self.image_analyzer else 'Unknown'}")
        logger.info(f"Semantic Splitting: {self.semantic_chunker.is_available() if self.semantic_chunker else 'Unknown'}")
        logger.info(f"Graph/Image Analysis: {self.image_analyzer.is_graph_available() if self.image_analyzer else 'Unknown'}")
        logger.info(f"Language Detection (for scanned PDFs): {'Available' if self.scanned_pdf_processor.lang_detect_available else 'Basic'}")
        logger.info(f"Advanced Image Preprocessing: {'Available' if self.scanned_pdf_processor.cv2_available else 'Basic'}")
        logger.info(f"Table Summarization: {'ENABLED' if config.ENABLE_TABLE_SUMMARIZATION else 'DISABLED'}")
        logger.info(f"Form Extraction: {'ENABLED' if config.ENABLE_FORM_EXTRACTION else 'DISABLED'}")
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

        if self.use_semantic_splitting_default and self.semantic_chunker.is_available():
            try:
                chunks_text = self.semantic_chunker.semantic_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                split_method_used = 'semantic'
                if not chunks_text and text.strip():
                    logger.warning(f"Semantic split returned no chunks for non-empty text (source: {metadata.get('source', 'N/A')}, page: {metadata.get('page', 'N/A')}). Falling back.")
                    chunks_text = []
            except Exception as sem_e:
                logger.error(f"Semantic splitting failed: {sem_e}. Falling back to token/recursive splitter.", exc_info=True)
                chunks_text = []

        if not chunks_text:
            try:
                chunks_text = self.token_splitter.split_text(text)
                split_method_used = 'token'
                if not chunks_text and text.strip():
                    logger.warning(f"Token splitter returned no chunks for non-empty text. Trying recursive splitter.")
                    chunks_text = []
            except Exception as tok_e:
                logger.error(f"Token splitting failed: {tok_e}. Falling back to recursive.", exc_info=True)
                chunks_text = []
        
        if not chunks_text:
            try:
                chunks_text = self.recursive_splitter.split_text(text)
                split_method_used = 'recursive'
                if not chunks_text and text.strip():
                    logger.warning(f"Recursive splitter also returned no chunks for non-empty text. This text may be lost.")
            except Exception as rec_e:
                logger.error(f"Recursive splitting also failed: {rec_e}. Text from this section might be lost.", exc_info=True)
                return []

        output_documents: List[Dict[str, Any]] = []
        for i, chunk_str in enumerate(chunks_text):
            chunk_meta = metadata.copy()
            chunk_meta['chunk_index'] = i
            chunk_meta['split_method'] = split_method_used
            
            content_blocks_in_chunk = extract_content_structure(chunk_str)
            headings_in_chunk = [cb.text for cb in content_blocks_in_chunk if cb.content_type == 'heading']
            if headings_in_chunk:
                chunk_meta['context_headings'] = ' > '.join(headings_in_chunk[-3:])
            
            if content_blocks_in_chunk:
                chunk_meta['content_types_in_chunk'] = ", ".join(sorted(list(set(cb.content_type for cb in content_blocks_in_chunk))))
            else:
                chunk_meta['content_types_in_chunk'] = "unknown"

            output_documents.append({'page_content': chunk_str, 'metadata': chunk_meta})
            
        return output_documents

    def _detect_pdf_processing_method(self, file_path: str) -> str:
        """Detect whether a PDF should be processed as regular text-based PDF or scanned PDF."""
        try:
            pdf_type = self.scanned_pdf_processor.detect_pdf_type(file_path)
            
            if pdf_type == "text_based":
                return 'regular'
            elif pdf_type == "scanned_image":
                return 'scanned'
            else:
                return 'hybrid'
                
        except Exception as e:
            logger.warning(f"Could not detect PDF processing method for {file_path}: {e}")
            return 'hybrid'

    def _filter_docs_for_english(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Helper method to filter a list of documents for English content."""
        if not docs:
            return []
            
        original_count = len(docs)
        filtered_docs = []
        for doc in docs:
            content = doc.get('page_content', '')
            if content and self._is_primarily_english(content):
                filtered_docs.append(doc)
            else:
                logger.debug(f"Filtered out non-English content chunk: {content[:100]}...")
        
        if original_count > len(filtered_docs):
            logger.info(f"Filtered {original_count} -> {len(filtered_docs)} document chunks for English content.")
            
        return filtered_docs

    def process_pdf_intelligently(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Intelligently process PDF files by detecting if they're text-based or scanned.
        Applies English-only filter for scanned content if configured.
        """
        processing_method = self._detect_pdf_processing_method(file_path)
        logger.info(f"PDF processing method for {os.path.basename(file_path)}: {processing_method}")
        
        processed_docs = []
        
        if processing_method == 'regular':
            try:
                processed_docs = process_pdf_file(file_path, self.image_analyzer, self._split_text_into_documents)
                if processed_docs:
                    logger.info(f"Successfully processed {file_path} using regular PDF processing")
                    return processed_docs
            except Exception as e:
                logger.warning(f"Regular PDF processing failed for {file_path}: {e}")
        
        elif processing_method == 'scanned':
            if self.scanned_pdf_processor.ocr_available and self.scanned_pdf_processor.pdf2image_available:
                try:
                    processed_docs = process_scanned_pdf_file(
                        file_path, 
                        self._split_text_into_documents,
                        force_ocr=True,
                        dpi=300
                    )
                    if processed_docs:
                        logger.info(f"Successfully processed {file_path} using OCR-based processing")
                        if self.english_only_filter:
                            processed_docs = self._filter_docs_for_english(processed_docs)
                        return processed_docs
                except Exception as e:
                    logger.error(f"OCR-based PDF processing failed for {file_path}: {e}")
            else:
                logger.error(f"Cannot process scanned PDF {file_path}: OCR libraries not available")
        
        elif processing_method == 'hybrid':
            try:
                processed_docs = process_pdf_file(file_path, self.image_analyzer, self._split_text_into_documents)
                
                total_content_length = sum(len(doc.get('page_content', '')) for doc in processed_docs)
                
                if total_content_length < 200:
                    logger.info(f"Insufficient content from regular processing ({total_content_length} chars). Trying OCR...")
                    
                    if self.scanned_pdf_processor.ocr_available and self.scanned_pdf_processor.pdf2image_available:
                        try:
                            ocr_docs = process_scanned_pdf_file(
                                file_path, 
                                self._split_text_into_documents,
                                force_ocr=True,
                                dpi=300
                            )
                            
                            if self.english_only_filter:
                                ocr_docs = self._filter_docs_for_english(ocr_docs)
                            
                            ocr_content_length = sum(len(doc.get('page_content', '')) for doc in ocr_docs)
                            if ocr_content_length > total_content_length * 2:
                                logger.info(f"OCR processing found significantly more content ({ocr_content_length} vs {total_content_length} chars)")
                                processed_docs = ocr_docs
                        except Exception as ocr_e:
                            logger.warning(f"OCR fallback failed for {file_path}: {ocr_e}")
                
                return processed_docs
                
            except Exception as e:
                logger.error(f"Hybrid PDF processing failed for {file_path}: {e}")
        
        return processed_docs

    def process_all_documents(self) -> List[Dict[str, Any]]:
        """
        Processes all supported files in the initialized directory with enhanced PDF handling.
        Returns a list of document chunks, each as a dictionary.
        """
        all_processed_docs: List[Dict[str, Any]] = []
        logger.info(f"Starting enhanced document processing in directory: {self.dir_path}")

        try:
            for filename in os.listdir(self.dir_path):
                file_path = os.path.join(self.dir_path, filename)
                if not os.path.isfile(file_path):
                    continue

                _, file_extension = os.path.splitext(filename.lower())
                
                if file_extension not in config.SUPPORTED_EXTENSIONS:
                    logger.debug(f"Skipping unsupported file type: {filename}")
                    continue

                logger.info(f"--- Processing file: {filename} ---")
                file_specific_docs: List[Dict[str, Any]] = []
                
                try:
                    if file_extension == '.pdf':
                        file_specific_docs = self.process_pdf_intelligently(file_path)
                    elif file_extension in ['.docx', '.doc']:
                        file_specific_docs = process_docx_file(file_path, self.image_analyzer, self._split_text_into_documents)
                    elif file_extension == '.txt':
                        file_specific_docs = process_txt_file(file_path, self._split_text_into_documents)
                    
                    all_processed_docs.extend(file_specific_docs)
                    logger.info(f"Finished processing {filename}, generated {len(file_specific_docs)} document chunks.")

                except Exception as e_file_proc:
                    logger.error(f"Failed to process file {filename}: {e_file_proc}", exc_info=True)
            
        except OSError as e_dir_read:
            logger.error(f"Directory read error for {self.dir_path}: {e_dir_read}", exc_info=True)
            raise

        logger.info(f"Overall document processing finished for directory {self.dir_path}. Total chunks generated: {len(all_processed_docs)}")
        
        sanitized_docs_final: List[Dict[str, Any]] = []
        for i, doc_item_dict in enumerate(all_processed_docs):
            if not isinstance(doc_item_dict, dict) or \
               'page_content' not in doc_item_dict or \
               'metadata' not in doc_item_dict:
                logger.warning(f"Skipping invalid document item at index {i}: {str(doc_item_dict)[:150]}")
                continue
            
            current_metadata = doc_item_dict.get('metadata', {})
            if isinstance(current_metadata, dict):
                cleaned_meta = {}
                for k, v in current_metadata.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        cleaned_meta[k] = v
                    elif isinstance(v, (list, tuple)):
                        try:
                            cleaned_meta[k] = [item if isinstance(item, (str, int, float, bool, type(None))) else str(item) for item in v]
                        except Exception as list_conv_e:
                            logger.debug(f"Could not convert list item to string for metadata key '{k}': {list_conv_e}")
                            cleaned_meta[k] = str(v)
                    else:
                        try:
                            cleaned_meta[k] = str(v)
                        except Exception as str_conv_e:
                            logger.debug(f"Could not convert metadata value to string for key '{k}': {str_conv_e}")
                doc_item_dict['metadata'] = cleaned_meta
                sanitized_docs_final.append(doc_item_dict)
            else:
                logger.warning(f"Document item at index {i} has non-dict metadata. Creating minimal metadata.")
                source_val = 'unknown_source_malformed_meta'
                if hasattr(current_metadata, 'get'):
                    source_val = str(current_metadata.get('source', 'unknown_source_in_malformed_object_meta'))
                elif current_metadata is not None:
                    source_val = str(current_metadata)

                doc_item_dict['metadata'] = {'source': source_val, 'original_metadata_type': str(type(current_metadata))}
                sanitized_docs_final.append(doc_item_dict)
                
        return sanitized_docs_final

    def process_single_scanned_pdf(
        self, 
        file_path: str, 
        dpi: int = 300, 
        max_pages: int = None,
        force_english_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Process a single scanned PDF file with custom settings."""
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        if not self.scanned_pdf_processor.ocr_available or not self.scanned_pdf_processor.pdf2image_available:
            logger.error("Cannot process scanned PDFs: Required libraries not available")
            return []
        
        logger.info(f"Processing single scanned PDF: {os.path.basename(file_path)}")
        logger.info(f"Settings: DPI={dpi}, Max Pages={max_pages}, English Only={force_english_only}")
        
        try:
            processed_docs = process_scanned_pdf_file(
                file_path=file_path,
                text_splitting_func=self._split_text_into_documents,
                force_ocr=True,
                dpi=dpi,
                max_pages=max_pages
            )
            
            if force_english_only and processed_docs:
                processed_docs = self._filter_docs_for_english(processed_docs)
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing scanned PDF {file_path}: {e}", exc_info=True)
            return []

    def _is_primarily_english(self, text: str, threshold: float = 0.8) -> bool:
        """Check if text is primarily English based on character analysis."""
        if not text.strip():
            return False
        
        english_chars = sum(1 for c in text if c.isascii() and (c.isalnum() or c.isspace() or c in '.,!?;:-()[]{}\"\''))
        total_chars = len(text)
        
        if total_chars == 0:
            return False
        
        english_ratio = english_chars / total_chars
        return english_ratio >= threshold

def process_scanned_pdf_directory(
    directory_path: str,
    dpi: int = 300,
    max_pages_per_file: int = None,
    english_only: bool = True
) -> List[Dict[str, Any]]:
    """Process all PDF files in a directory as scanned documents."""
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    
    # The main processor is now configured with the english_only filter.
    processor = EnhancedDocumentProcessor(directory_path, english_only_filter=english_only)
    all_docs = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            # Use the single-file method for this directory-level convenience function
            docs = processor.process_single_scanned_pdf(
                file_path=file_path,
                dpi=dpi,
                max_pages=max_pages_per_file,
                force_english_only=english_only
            )
            all_docs.extend(docs)
    
    return all_docs