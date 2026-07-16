# document_processing/txt_processor.py
# version : 1.2.0
# File: 8
# This file will handle TXT-specific processing.


import logging
import os
import numpy as np # For tabular data detection std dev if used
from typing import List, Dict, Any

# Config and logger
from .config_handler import config, logger

def process_txt_file(
    file_path: str,
    text_splitting_func # Function like _split_text_into_documents from core_processor
) -> List[Dict[str, Any]]:

    all_docs: List[Dict[str, Any]] = []
    try:
        logger.debug(f"Processing TXT: {file_path}")
        
        # Try common encodings (from original)
        possible_encodings = ['utf-8', 'latin-1', 'cp1252']
        text_content = None
        detected_encoding = None

        for enc in possible_encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    text_content = f.read()
                detected_encoding = enc
                logger.debug(f"TXT file {file_path} read successfully with encoding: {enc}")
                break
            except (UnicodeDecodeError, OSError):
                logger.debug(f"Failed to read {file_path} with encoding {enc}.")
                continue
        
        # Fallback if common encodings fail (from original)
        if text_content is None:
            try:
                with open(file_path, 'rb') as f: # Read as bytes
                    raw_bytes = f.read()
                # Attempt to decode with utf-8, replacing errors
                text_content = raw_bytes.decode('utf-8', errors='replace')
                detected_encoding = 'utf-8 (with replacements)'
                logger.warning(f"Decoded TXT {file_path} using utf-8 with error replacement after other encodings failed.")
            except Exception as fb_e:
                logger.error(f"Cannot read/decode TXT file {file_path} even with fallback: {fb_e}")
                return []
        
        metadata = {'source': file_path, 'encoding': detected_encoding, 'file_type': 'txt'}

        # Tabular data detection logic (simplified from original)
        # This could be enhanced or made configurable.
        lines = text_content.strip().split('\n')
        if len(lines) > 5: # Only check for tables if more than a few lines
            delimiters_to_check = [',', '\t', ';', '|']
            delimiter_counts: Dict[str, int] = {d: 0 for d in delimiters_to_check}
            consistent_delimiter = None
            
            # Check first few lines for consistent delimiters
            sample_lines = lines[:min(20, len(lines))] 
            for line_sample in sample_lines:
                if line_sample.strip(): # Skip empty lines
                    for delim in delimiters_to_check:
                        if delim in line_sample:
                            delimiter_counts[delim] += 1
            
            # Find the most frequent delimiter if it appears consistently
            # Original had a more complex check involving std dev of counts per line
            # Simplified: if a delimiter appears in a good portion of sample lines.
            min_line_occurrence = max(1, int(len(sample_lines) * 0.5)) # Must appear in at least 50% of sample lines
            
            best_delim = None
            max_occurrences = 0
            for delim, count in delimiter_counts.items():
                if count >= min_line_occurrence and count > max_occurrences:
                    # Further check: does this delimiter create multiple fields consistently?
                    # (e.g. average number of splits per line is > 1)
                    num_fields_per_line = [line_s.count(delim) + 1 for line_s in sample_lines if delim in line_s]
                    if num_fields_per_line and np.mean(num_fields_per_line) > 1.5 : # Avg more than 1.5 fields
                        max_occurrences = count
                        best_delim = delim
            
            if best_delim:
                metadata['content_format'] = 'tabular_text'
                metadata['likely_delimiter'] = best_delim
                logger.info(f"Likely tabular TXT file detected: {os.path.basename(file_path)} with delimiter '{best_delim}'")

        if text_content.strip():
            all_docs = text_splitting_func(text_content, metadata)
            
    except FileNotFoundError:
        logger.error(f"TXT file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error processing TXT file {file_path}: {e}", exc_info=True)
        return []
        
    return all_docs