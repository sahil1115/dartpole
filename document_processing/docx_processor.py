# document_processing/docx_processor.py
# version : 1.2.0
# File: 7
# This file will handle DOCX-specific processing.


import logging
import os
import shutil
import numpy as np # For unique temp dir name generation
from PIL import Image # For image handling
from typing import List, Dict, Any

# Config and logger
from .config_handler import config, logger
# Image utilities
from .image_utils import ImageAnalyzer # No need for GRAPH_EXTRACTION_AVAILABLE here, ImageAnalyzer handles it

try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False
    logger.warning("docx2txt not found. DOCX file processing will be unavailable.")


def process_docx_file(
    file_path: str, 
    image_analyzer: ImageAnalyzer,
    text_splitting_func # Function like _split_text_into_documents from core_processor
) -> List[Dict[str, Any]]:

    if not DOCX2TXT_AVAILABLE:
        logger.error(f"Cannot process DOCX {file_path}, docx2txt library is missing.")
        return []

    all_docs: List[Dict[str, Any]] = []
    try:
        logger.debug(f"Processing DOCX: {file_path}")
        
        doc_basename = os.path.splitext(os.path.basename(file_path))[0]
        # Create a unique temporary directory for images for this DOCX file
        # Use a combination of basename and random int to ensure uniqueness
        temp_img_dir_name = f"temp_docx_imgs_{doc_basename}_{np.random.randint(10000, 99999)}"
        # Place it in a standard temp location or a sub-directory of CWD
        # For example, in the same directory as the script for simplicity here:
        temp_img_dir_path = os.path.join(os.getcwd(), temp_img_dir_name) # Adjust as needed
        
        text_content = ""
        image_extraction_path_used = None

        try:
            os.makedirs(temp_img_dir_path, exist_ok=True)
            text_content = docx2txt.process(file_path, img_dir=temp_img_dir_path)
            image_extraction_path_used = temp_img_dir_path
            logger.debug(f"DOCX images (if any) extracted to temporary folder: {temp_img_dir_path}")
        except Exception as e_docx2txt_with_img:
            logger.warning(f"docx2txt failed to extract images to specific temp dir '{temp_img_dir_path}': {e_docx2txt_with_img}. Trying without explicit img_dir or default behavior.")
            # Fallback: Try processing without explicit img_dir (docx2txt might use a default location relative to file)
            # Or, if that's not desired, just get text.
            try:
                text_content = docx2txt.process(file_path) # Let docx2txt handle image location internally if it can
                # If images are needed and the above doesn't provide a clear path, this part becomes complex.
                # The original code used "word/media" which is typical for unzipped .docx files.
                # This implies one might need to unzip the docx to reliably get images if img_dir fails.
                # For now, we proceed with text if image_dir fails.
                logger.info(f"Processed DOCX {file_path} for text content; image path not explicitly set after fallback.")
            except Exception as e_docx2txt_text_only:
                logger.error(f"docx2txt failed to extract even text content from {file_path}: {e_docx2txt_text_only}")
                return []


        metadata = {'source': file_path, 'file_type': 'docx'}
        embedded_visuals_summary = ""
        images_processed_count = 0

        if image_extraction_path_used and os.path.isdir(image_extraction_path_used) and \
           getattr(config, 'ENABLE_GRAPH_EXTRACTION', False): # Process images if path is valid and feature enabled
            
            for img_filename in os.listdir(image_extraction_path_used):
                img_full_path = os.path.join(image_extraction_path_used, img_filename)
                # Check if it's a common image file type
                if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                    try:
                        pil_image = Image.open(img_full_path)
                        is_graph = False
                        if image_analyzer and image_analyzer.is_graph_available(): # Check if analyzer can do graph detection
                            is_graph = image_analyzer.is_graph_image(pil_image)
                        
                        images_processed_count += 1
                        img_placeholder_id = f"VisualRef-{images_processed_count}"
                        
                        if is_graph:
                            graph_meta = image_analyzer.extract_graph_data(pil_image)
                            g_type = graph_meta.get('graph_type', 'unknown')
                            g_title = graph_meta.get('title', 'N/A')
                            embedded_visuals_summary += f"\n[{img_placeholder_id}: Graph (Type: {g_type}, Title: '{g_title}', SourceFile: {img_filename})]"
                        else:
                            embedded_visuals_summary += f"\n[{img_placeholder_id}: Image (SourceFile: {img_filename})]"
                    
                    except Exception as e_img_proc:
                        logger.warning(f"Error processing image {img_filename} from DOCX {file_path}: {e_img_proc}")
            
            if embedded_visuals_summary:
                text_content += "\n\n--- Embedded Visuals Information ---" + embedded_visuals_summary

        # Cleanup the temporary image directory
        if image_extraction_path_used and os.path.exists(image_extraction_path_used):
            try:
                shutil.rmtree(image_extraction_path_used)
                logger.debug(f"Successfully removed temporary DOCX image directory: {image_extraction_path_used}")
            except Exception as e_rmtree:
                logger.warning(f"Failed to remove temporary DOCX image directory {image_extraction_path_used}: {e_rmtree}")
        
        if text_content.strip():
            all_docs = text_splitting_func(text_content, metadata)
        
    except FileNotFoundError:
        logger.error(f"DOCX file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error processing DOCX file {file_path}: {e}", exc_info=True)
        return []
        
    return all_docs