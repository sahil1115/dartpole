# document_processing/content_handler.py
# version : 1.2.0
# File: 5
# This file will contain the ContentBlock dataclass and the _extract_content_structure function.


import re
from dataclasses import dataclass, field # field for default_factory
from typing import List, Dict, Any

@dataclass
class ContentBlock:
    text: str
    content_type: str # e.g., 'paragraph', 'heading', 'list_item', 'table_row_text'
    level: int = 0 # Optional: for heading levels or list indentation
    page_num: int = 0 # Optional: if content is page-specific
    # Ensure metadata is mutable and defaults to an empty dict correctly
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_content_structure(text: str) -> List[ContentBlock]:
    """
    Analyzes text and breaks it down into logical ContentBlocks.
    This is a simplified version of the original logic.
    """
    if not text or not text.strip():
        return []
    
    lines = text.split('\n')
    blocks = []
    current_block_lines = []
    current_type = 'paragraph' # Default type

    for line in lines:
        stripped_line = line.strip()

        # If line is empty, it might signify end of current block
        if not stripped_line:
            if current_block_lines:
                blocks.append(ContentBlock(text='\n'.join(current_block_lines), content_type=current_type))
                current_block_lines = []
            # Reset type or carry over based on context (simplified: reset to paragraph)
            current_type = 'paragraph' 
            continue

        # Basic type detection (can be enhanced significantly)
        # Original logic for detecting headings, lists, tables was more detailed.
        is_heading_candidate = len(stripped_line) < 100 and \
                               (stripped_line.isupper() or \
                                stripped_line.endswith(':') or \
                                re.match(r'^(?:[IVXLC]+\.|\d+\.|[A-Z]\.)\s+', stripped_line))
        
        is_list_item_candidate = re.match(r'^[\*\-•+]|\d+[\.)]', stripped_line)
        
        # Table detection was based on multiple spaces/tabs; this is harder to do reliably
        # without more context or structure from the source (e.g., PDF table extraction).
        # Simplified: if it's not a heading or list, it's a paragraph for now.

        line_type_determined = 'paragraph' # Default
        if is_heading_candidate:
            line_type_determined = 'heading'
        elif is_list_item_candidate:
            line_type_determined = 'list_item'
        # Add more type detections (e.g., for 'table_text' if applicable from source)

        # If type changes, finalize previous block
        if line_type_determined != current_type and current_block_lines:
            blocks.append(ContentBlock(text='\n'.join(current_block_lines), content_type=current_type))
            current_block_lines = [line] # Start new block with current line
            current_type = line_type_determined
        else:
            current_block_lines.append(line)
            # If it's the first line of a new potential block, set its type
            if len(current_block_lines) == 1:
                 current_type = line_type_determined
    
    # Add any remaining lines as the last block
    if current_block_lines:
        blocks.append(ContentBlock(text='\n'.join(current_block_lines), content_type=current_type))
        
    return blocks