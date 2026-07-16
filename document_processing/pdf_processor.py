# document_processing/pdf_processor.py
# version : 1.2.0
# File: 6
# This file will handle PDF-specific processing



import logging
import os
from typing import List, Dict, Any

# Config and logger
from .config_handler import config, logger
# Image utilities
from .image_utils import ImageAnalyzer, OCR_AVAILABLE, GRAPH_EXTRACTION_AVAILABLE
# ContentBlock for structuring extracted data (if used directly here)
from .content_handler import ContentBlock # Or pass structured text

# PDF processing libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not found. PDF text and table extraction will be limited.")

try:
    import pdf2image
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not found. PDF image extraction and OCR fallback will be disabled.")

# Pytesseract availability is checked in image_utils (OCR_AVAILABLE), but the module
# itself is needed here for the OCR fallback call.
try:
    import pytesseract
except ImportError:
    pytesseract = None

# PDFMiner for form extraction (as in original)
try:
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdftypes import resolve1, PDFObjRef
    from pdfminer.psparser import PSLiteral  # PDF name objects (pdfminer.six has no PDFName)
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger.warning("pdfminer.six not found. PDF form field extraction using pdfminer will be unavailable.")


# Forward declaration for split_text_into_documents if it's in core_processor
# from .core_processor import split_text_into_documents # This creates a circular dependency if core_processor imports this.
# Alternative: Pass the splitter function as an argument, or have core_processor call this module's functions.
# For now, assume split_text_into_documents will be passed or called by the main DocumentProcessor.


def _extract_form_fields_pdfminer(pdf_path: str) -> Dict[str, Any]:
    if not PDFMINER_AVAILABLE:
        return {}
    
    form_fields: Dict[str, Any] = {}
    try:
        with open(pdf_path, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)

            if 'AcroForm' not in doc.catalog:
                logger.debug(f"pdfminer: No AcroForm dictionary found in {os.path.basename(pdf_path)}.")
                return {}

            acro_form = resolve1(doc.catalog['AcroForm'])
            if not acro_form or 'Fields' not in acro_form:
                logger.debug(f"pdfminer: AcroForm found but no 'Fields' entry in {os.path.basename(pdf_path)}.")
                return {}

            pdf_fields_refs = resolve1(acro_form.get('Fields', []))
            field_count = 0
            for field_ref in pdf_fields_refs:
                field = resolve1(field_ref)
                if not field: continue

                name = field.get('T') # Field name
                value = field.get('V') # Field value
                field_type_raw = field.get('FT') # Field type

                # Decode name (often bytes)
                try:
                    if name and isinstance(name, bytes): name = name.decode('utf-8', errors='ignore')
                except Exception: continue # Skip if name cannot be decoded
                if not name: continue # Skip if no name

                # Decode value
                try:
                    if value and isinstance(value, bytes): value = value.decode('utf-8', errors='ignore')
                    if isinstance(value, PDFObjRef): value = resolve1(value) # Resolve reference
                    if isinstance(value, PSLiteral): value = value.name # PDF name objects
                    # Handle common boolean representations for checkboxes/radio
                    if value == 'Yes': value = True
                    elif value == 'Off': value = False
                    # Handle lists (e.g., for multi-select choices)
                    if isinstance(value, list):
                        value = [v.name if isinstance(v, PSLiteral) else v for v in value]
                except Exception as val_err:
                    logger.warning(f"Could not fully decode value for pdfminer field '{name}': {val_err}")


                # Determine field type string (from original logic)
                type_str = "Unknown"
                if field_type_raw and isinstance(field_type_raw, PSLiteral):
                    type_str = field_type_raw.name
                    if type_str == 'Tx': type_str = 'Text'
                    elif type_str == 'Btn': # Button can be checkbox, radio, or pushbutton
                        flags = field.get('Ff', 0) # Field flags
                        if flags & (1 << 15): type_str = 'Radio'      # Radio button (bit 16 is 1-based index)
                        elif flags & (1 << 16): type_str = 'Pushbutton' # Push button (bit 17)
                        else: type_str = 'Checkbox' # Default for Btn if not radio/push
                    elif type_str == 'Ch': type_str = 'Choice'
                    elif type_str == 'Sig': type_str = 'Signature'
                
                form_fields[name] = {"value": value, "type": type_str, "page": "N/A (pdfminer)"}
                field_count +=1
        
        if field_count > 0:
            logger.info(f"pdfminer extracted {field_count} form fields from AcroForm for {os.path.basename(pdf_path)}.")
        else:
            logger.debug(f"pdfminer found no form fields in AcroForm for {os.path.basename(pdf_path)}.")
            
    except Exception as e:
        logger.error(f"Error in pdfminer AcroForm extraction for {pdf_path}: {e}")
    return form_fields


def process_pdf_file(
    file_path: str, 
    image_analyzer: ImageAnalyzer, 
    text_splitting_func # Function like _split_text_into_documents from core_processor
) -> List[Dict[str, Any]]: # Returns list of document dicts for vector store

    all_docs: List[Dict[str, Any]] = []
    pdf_plumber_doc = None
    page_images_cache = [] # Cache images if multiple operations need them (OCR, rotation)

    try:
        # Determine number of pages
        num_pages = 0
        if PDFPLUMBER_AVAILABLE:
            try:
                pdf_plumber_doc = pdfplumber.open(file_path)
                num_pages = len(pdf_plumber_doc.pages)
            except Exception as e_plumber_open:
                logger.warning(f"pdfplumber failed to open {file_path}: {e_plumber_open}. Will try pdf2image for page count if available.")
                if PDF2IMAGE_AVAILABLE: # Fallback for page count
                    try:
                        # Get info instead of full images just for page count to save memory
                        info = pdf2image.pdfinfo_from_path(file_path)
                        num_pages = info['Pages']
                    except Exception as e_pdf2image_info:
                        logger.error(f"pdf2image also failed to get page count for {file_path}: {e_pdf2image_info}")
                        return [] # Cannot proceed without page count
                else:
                    logger.error(f"No PDF library available to determine page count for {file_path}.")
                    return []
        elif PDF2IMAGE_AVAILABLE: # pdfplumber not available, use pdf2image for page count
            try:
                info = pdf2image.pdfinfo_from_path(file_path)
                num_pages = info['Pages']
            except Exception as e_pdf2image_info:
                logger.error(f"pdf2image failed to get page count for {file_path}: {e_pdf2image_info}")
                return []
        else:
            logger.error(f"Neither pdfplumber nor pdf2image available for PDF processing of {file_path}.")
            return []

        # Form Field Extraction (using pdfminer as per original)
        if getattr(config, 'ENABLE_FORM_EXTRACTION', False): # Check config
            form_fields = _extract_form_fields_pdfminer(file_path)
            if form_fields:
                form_content = "Document Form Fields:\n" + "\n".join(
                    f"- Field '{k}' (Type: {v.get('type', 'N/A')}, Page: {v.get('page', 'N/A')}): '{v.get('value', '')}'"
                    for k, v in form_fields.items()
                )
                all_docs.append({
                    'page_content': form_content,
                    'metadata': {'source': file_path, 'content_type': 'form_data_summary', 'total_pages': num_pages, 'page': 0} # Assign page 0 for forms
                })
        
        # Per-page processing
        for page_num_idx in range(num_pages): # 0-indexed
            page_num_display = page_num_idx + 1 # 1-indexed for metadata/logging
            page_text_content = ""
            page_meta = {'source': file_path, 'page': page_num_display, 'total_pages': num_pages}
            content_extraction_log = [] # Log what was extracted from the page

            # 1. Text and Table Extraction (pdfplumber)
            if pdf_plumber_doc and page_num_idx < len(pdf_plumber_doc.pages):
                try:
                    page = pdf_plumber_doc.pages[page_num_idx]
                    extracted_text = page.extract_text(x_tolerance=1, y_tolerance=3) # Params from common usage
                    if extracted_text and extracted_text.strip():
                        page_text_content += extracted_text.strip() + "\n\n"
                        content_extraction_log.append("Text(pdfplumber)")

                    if getattr(config, 'ENABLE_TABLE_SUMMARIZATION', False): # Check config for tables
                        tables = page.extract_tables()
                        if tables:
                            content_extraction_log.append(f"Tables({len(tables)})")
                            for table_idx, table_data in enumerate(tables):
                                if table_data:
                                    # Convert table data to a string format (e.g., CSV-like or Markdown)
                                    table_str_representation = "\n".join(["\t".join(str(cell).strip().replace('\n', ' ') if cell else "" for cell in row) for row in table_data])
                                    if table_str_representation.strip():
                                        page_text_content += f"--- Table {page_num_display}.{table_idx + 1} ---\n{table_str_representation}\n---\n\n"
                except Exception as e_plumber_page:
                    logger.error(f"pdfplumber error on page {page_num_display} of {os.path.basename(file_path)}: {e_plumber_page}")

            # 2. Image Processing (pdf2image, ImageAnalyzer for rotation, graph)
            # Only load images when rotation or graph extraction is explicitly enabled.
            # OCR fallback is handled lazily in step 3 below.
            pil_page_image = None
            needs_image_for_features = getattr(config, 'ENABLE_AUTO_ROTATION', False) or \
                                       getattr(config, 'ENABLE_GRAPH_EXTRACTION', False)
            if PDF2IMAGE_AVAILABLE and needs_image_for_features:
                try:
                    if not page_images_cache:
                        page_images_cache = convert_from_path(file_path, dpi=150, first_page=1, last_page=num_pages)

                    if page_images_cache and page_num_idx < len(page_images_cache):
                        pil_page_image = page_images_cache[page_num_idx]
                        content_extraction_log.append("Image(pdf2image)")

                        # Auto-rotation
                        if getattr(config, 'ENABLE_AUTO_ROTATION', False) and image_analyzer and pil_page_image:
                            try:
                                rotation_angle = image_analyzer.detect_rotation(pil_page_image)
                                if rotation_angle != 0:
                                    pil_page_image = pil_page_image.rotate(-rotation_angle, expand=True)
                                    page_meta['rotated_degrees'] = rotation_angle
                                    content_extraction_log.append(f"Rotated({rotation_angle})")
                            except Exception as rot_e:
                                logger.warning(f"Auto-rotation error on page {page_num_display}: {rot_e}")

                        # Graph Detection/Placeholder
                        if getattr(config, 'ENABLE_GRAPH_EXTRACTION', False) and image_analyzer and image_analyzer.is_graph_available() and pil_page_image:
                            img_id_ref = f"ImageRef-{page_num_display}.1"
                            if image_analyzer.is_graph_image(pil_page_image):
                                content_extraction_log.append("GraphDetected")
                                graph_meta_data = image_analyzer.extract_graph_data(pil_page_image)
                                graph_type = graph_meta_data.get('graph_type', 'unknown')
                                graph_title = graph_meta_data.get('title', 'N/A')
                                page_text_content += f"--- {img_id_ref} (Type: Graph - {graph_type}, Title: '{graph_title}') ---\n"
                                page_text_content += "[Graph data information would be summarized here if fully extracted.]\n\n"
                            else:
                                page_text_content += f"--- {img_id_ref} (Type: Image) ---\n[Image content would be described or referenced here.]\n\n"

                except Exception as img_err:
                    logger.warning(f"Image processing error for page {page_num_display} of {os.path.basename(file_path)}: {img_err}")

            # 3. OCR Fallback — only load page image here if text is actually insufficient
            final_text_strip = page_text_content.strip()
            min_text_len_for_ocr_skip = getattr(config, 'PDF_MIN_TEXT_LENGTH', 30)
            if OCR_AVAILABLE and \
               (not final_text_strip or len(final_text_strip) < min_text_len_for_ocr_skip):
                # Lazily load page image only when OCR is actually needed
                if PDF2IMAGE_AVAILABLE and pil_page_image is None:
                    try:
                        if not page_images_cache:
                            page_images_cache = convert_from_path(file_path, dpi=300, first_page=1, last_page=num_pages)
                        if page_images_cache and page_num_idx < len(page_images_cache):
                            pil_page_image = page_images_cache[page_num_idx]
                    except Exception as img_err:
                        logger.warning(f"Could not load page image for OCR fallback on page {page_num_display} of {os.path.basename(file_path)}: {img_err}")
            if OCR_AVAILABLE and pil_page_image and \
               (not final_text_strip or len(final_text_strip) < min_text_len_for_ocr_skip):
                logger.info(f"Attempting OCR for page {page_num_display} of {os.path.basename(file_path)} (Text length: {len(final_text_strip)})")
                try:
                    # Image should be rotated by now if auto-rotation was enabled
                    ocr_text = pytesseract.image_to_string(pil_page_image) # Use the potentially rotated image
                    if ocr_text and ocr_text.strip():
                        page_text_content += f"--- OCR Extracted Text ---\n{ocr_text.strip()}\n---\n"
                        content_extraction_log.append("Text(OCR)")
                except Exception as ocr_err:
                    logger.error(f"OCR error on page {page_num_display} of {os.path.basename(file_path)}: {ocr_err}")

            # Finalize page content and split
            page_meta['content_extracted'] = ", ".join(sorted(list(set(content_extraction_log)))) or "None"
            final_page_text_to_split = page_text_content.strip()

            if final_page_text_to_split and len(final_page_text_to_split) >= min_text_len_for_ocr_skip:
                # Use the passed text_splitting_func (e.g., _split_text_into_documents from core_processor)
                page_docs = text_splitting_func(final_page_text_to_split, page_meta)
                all_docs.extend(page_docs)
            elif not final_page_text_to_split:
                logger.debug(f"No text content extracted from page {page_num_display} of {os.path.basename(file_path)}.")
        
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True)
        # Ensure partial docs are not returned on major error, or handle as per requirements
        return [] 
    finally:
        if pdf_plumber_doc:
            pdf_plumber_doc.close()
        page_images_cache = [] # Clear cache

    return all_docs