# document_processing/image_utils.py
# version : 1.3.0 (updated with new chart extraction functions)

import logging
import numpy as np
import math
from PIL import Image
from typing import Dict, Any, Tuple, List # Added List for new type hints

# Import config from the new config_handler
from .config_handler import config, logger # Use the logger from config_handler or get a new one

try:
    import cv2
    GRAPH_EXTRACTION_AVAILABLE = True
except ImportError:
    GRAPH_EXTRACTION_AVAILABLE = False
    logger.warning("OpenCV (cv2) not found. Some image analysis features (graph detection, advanced rotation) will be disabled.")

try:
    import pytesseract
    # The Tesseract path might need to be configurable or handled more robustly
    # For now, keeping the original hardcoded path as a fallback.
    # Consider moving this to config.py if it needs to be user-set.
    try:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    except Exception as tess_path_ex:
        logger.warning(f"Could not set tesseract_cmd path: {tess_path_ex}. Ensure Tesseract is in PATH or path is correctly set if OCR is needed.")
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("Pytesseract not found. OCR features will be disabled.")


class ImageAnalyzer:
    def __init__(self):
        self._graph_available = GRAPH_EXTRACTION_AVAILABLE and hasattr(config, 'ENABLE_GRAPH_EXTRACTION') and config.ENABLE_GRAPH_EXTRACTION
        self._ocr_available = OCR_AVAILABLE
        self._enable_ocr_rotation = hasattr(config, 'ENABLE_OCR_ROTATION') and config.ENABLE_OCR_ROTATION
        self._ocr_confidence_threshold = getattr(config, 'OCR_CONFIDENCE_THRESHOLD', 2.0)
        logger.info(f"ImageAnalyzer initialized: Graph detection {'enabled' if self._graph_available else 'disabled'}, OCR {'enabled' if self._ocr_available else 'disabled'}.")

    def is_graph_available(self) -> bool:
        return self._graph_available

    def _pil_to_cv2(self, pil_image: Image.Image):
        if not GRAPH_EXTRACTION_AVAILABLE: return None # cv2 might not be imported
        try:
            return cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"PIL to CV2 error: {e}")
            return None

    def _cv2_to_pil(self, cv2_image):
        if not GRAPH_EXTRACTION_AVAILABLE: return None # cv2 might not be imported
        try:
            return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"CV2 to PIL error: {e}")
            return None

    def detect_rotation(self, image_data) -> int:
        if not self._ocr_available and not self._graph_available:
            return 0

        pil_img = None
        if isinstance(image_data, Image.Image):
            pil_img = image_data
        elif GRAPH_EXTRACTION_AVAILABLE and isinstance(image_data, np.ndarray): # cv2 image
            pil_img = self._cv2_to_pil(image_data)
        
        if pil_img is None:
             logger.warning("detect_rotation: Could not process image_data to PIL Image.")
             return 0

        # OCR-based rotation detection
        if self._ocr_available and self._enable_ocr_rotation:
            try:
                min_size = 500
                if min(pil_img.size) < min_size:
                    scale = min_size / min(pil_img.size)
                    new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
                    resized_pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    resized_pil_img = pil_img

                if resized_pil_img.size[0] * resized_pil_img.size[1] < 10000:
                    logger.debug("Skipping OCR orientation: Image too small (%dx%d)", resized_pil_img.size[0], resized_pil_img.size[1])
                else:
                    osd = pytesseract.image_to_osd(
                        resized_pil_img,
                        output_type=pytesseract.Output.DICT,
                        config='--dpi 300 --psm 0'
                    )
                    if 'rotate' in osd and 'confidence' in osd and osd['confidence'] > self._ocr_confidence_threshold:
                        logger.debug("OCR orientation detected: Rotate %d degrees (confidence: %.2f)", osd['rotate'], osd['confidence'])
                        return osd['rotate']
                    else:
                        logger.debug("OCR orientation skipped: Insufficient confidence or missing OSD data (OSD: %s)", osd)
            except Exception as e:
                logger.warning(f"OCR orientation error for image (size: {pil_img.size}): {e}")

        # Line-based rotation detection (Graph-based)
        if self._graph_available:
            try:
                image_np = self._pil_to_cv2(pil_img)
                if image_np is None or image_np.size == 0:
                    return 0
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
                if lines is None:
                    return 0
                h_votes, v_votes = 0, 0
                for line_segment in lines:
                    for x1, y1, x2, y2 in line_segment:
                        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                        if angle < 10 or angle > 170:
                            h_votes += 1
                        elif 80 < angle < 100:
                            v_votes += 1
                if v_votes > h_votes * 1.5:
                    return 90
            except Exception as e:
                logger.error(f"Line rotation detection error: {e}")
        return 0

    # =================================
    # MISSING HELPER FUNCTION STUBS
    # =================================
    def _extract_all_text(self, pil_image: Image.Image) -> List[Dict]:
        # Placeholder: Implement OCR logic to extract text with bounding boxes/positions
        # Each dict should have 'text', 'center_x', 'center_y', 'x', 'y', 'w', 'h' (or similar)
        logger.warning("Placeholder: _extract_all_text called. Needs implementation.")
        if self._ocr_available:
            # Example using pytesseract.image_to_data
            # texts = []
            # data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            # for i in range(len(data['text'])):
            #     if data['text'][i].strip(): # only non-empty text
            #         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            #         texts.append({
            #             'text': data['text'][i],
            #             'x': x, 'y': y, 'w': w, 'h': h,
            #             'center_x': x + w/2,
            #             'center_y': y + h/2
            #         })
            # return texts
            pass
        return []

    def _analyze_chart_structure(self, pil_image: Image.Image, texts: List[Dict]) -> Dict[str, Any]:
        # Placeholder: Implement logic to identify title, axes, legend, etc.
        logger.warning("Placeholder: _analyze_chart_structure called. Needs implementation.")
        return {
            'title': 'N/A (Placeholder)',
            'x_label': '',
            'y_label': '',
            'legend': [],
            'axes_bounds': {}, # e.g., {'x_min_pixel': 50, 'x_max_pixel': 450, 'y_min_pixel': 50, 'y_max_pixel': 350, 'x_min_val':0, ...}
            'confidence': 0.0
        }

    def _detect_bars(self, gray_image) -> List[Tuple[int, int, int, int]]:
        # Placeholder: Implement OpenCV logic to detect rectangular bars
        logger.warning("Placeholder: _detect_bars called. Needs implementation.")
        return [] # List of (x, y, w, h)

    def _estimate_value_from_position(self, position_pixel: int, axes_bounds: Dict, orientation: str) -> float:
        # Placeholder: Implement logic to map pixel to data value
        # Needs axes_bounds like: {'y_pixel_min': Y_IMG_COORD_FOR_Y_AXIS_MIN, 'y_pixel_max': Y_IMG_COORD_FOR_Y_AXIS_MAX,
        #                         'y_val_min': Y_AXIS_VALUE_MIN, 'y_val_max': Y_AXIS_VALUE_MAX}
        # Similar for horizontal
        logger.warning("Placeholder: _estimate_value_from_position called. Needs implementation.")
        return 0.0

    def _distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        # Placeholder: Implement Euclidean distance
        # logger.debug("Placeholder: _distance called. Needs implementation.")
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)


    def _extract_axis_values(self, texts: List[Dict], image_size: Tuple[int, int], axis_type: str) -> List[Dict]:
        # Placeholder: Extract and parse axis values from text
        # e.g., [{'value': 10, 'position': 50_pixel_coord}, ...]
        logger.warning("Placeholder: _extract_axis_values called. Needs implementation.")
        return []

    def _detect_line_points(self, cv_image) -> List[Tuple[int, int]]:
        # Placeholder: Detect key points on a line graph
        logger.warning("Placeholder: _detect_line_points called. Needs implementation.")
        return [] # List of (x,y) tuples

    def _detect_scatter_points(self, cv_image) -> List[Tuple[int, int]]:
        # Placeholder: Detect individual points in a scatter plot
        logger.warning("Placeholder: _detect_scatter_points called. Needs implementation.")
        return [] # List of (x,y) tuples

    def _has_radial_lines(self, gray_image, center: Tuple[int, int], radius: int) -> bool:
        # Placeholder: Check for lines radiating from the center (for pie charts)
        logger.warning("Placeholder: _has_radial_lines called. Needs implementation.")
        return False

    def _are_rectangles_aligned(self, rectangles: List[Tuple[int, int, int, int]]) -> bool:
        # Placeholder: Check if rectangles are aligned (horizontally or vertically)
        logger.warning("Placeholder: _are_rectangles_aligned called. Needs implementation.")
        # Basic check: more than 2 rectangles, and either their x-coords or y-coords are similar
        if len(rectangles) < 2: return False
        # Simple alignment check (can be much more sophisticated)
        aligned_x = len(set(r[0] for r in rectangles)) < len(rectangles) / 2
        aligned_y = len(set(r[1] for r in rectangles)) < len(rectangles) / 2
        return aligned_x or aligned_y


    def _has_connected_lines(self, lines) -> bool: # lines from HoughLinesP
        # Placeholder: Analyze line segments to see if they form continuous paths
        logger.warning("Placeholder: _has_connected_lines called. Needs implementation.")
        return lines is not None and len(lines) > 2 # very basic placeholder

    def _has_many_points(self, gray_image) -> bool:
        # Placeholder: Detect if there are many small, distinct points/blobs
        logger.warning("Placeholder: _has_many_points called. Needs implementation.")
        # Example using blob detection
        # params = cv2.SimpleBlobDetector_Params()
        # params.filterByArea = True
        # params.minArea = 5
        # params.maxArea = 100
        # detector = cv2.SimpleBlobDetector_create(params)
        # keypoints = detector.detect(gray_image)
        # return len(keypoints) > 10 # Arbitrary threshold
        return False

    def _extract_generic_chart_data(self, image: Image.Image, texts: List[Dict], structure: Dict) -> List[Dict]:
        logger.warning("Placeholder: _extract_generic_chart_data called. Needs implementation.")
        return []
    # ======================
    # Improved Graph Detection (NEW - replacing old is_graph_image)
    # ======================
    def is_graph_image(self, image_data) -> bool:
        """Determine if image contains a chart or graph"""
        if not self._graph_available:
            return False
        
        try:
            # Convert to cv2 format if needed
            if isinstance(image_data, Image.Image):
                image_cv = self._pil_to_cv2(image_data)
            elif isinstance(image_data, np.ndarray):
                image_cv = image_data
            else:
                logger.warning("is_graph_image: Unsupported image_data type.")
                return False
            
            if image_cv is None:
                return False
            
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # 1. Check for axes lines
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                h_lines = 0
                v_lines = 0
                
                for line_arr in lines: # Iterate over the outer array
                    for x1, y1, x2, y2 in line_arr: # Iterate points in segment
                        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                        if angle < 10 or angle > 170: # Horizontal
                            h_lines += 1
                        elif 80 < angle < 100: # Vertical
                            v_lines += 1
                
                # Likely has axes if both horizontal and vertical lines present
                if h_lines >= 1 and v_lines >= 1:
                    logger.debug("is_graph_image: Detected axes lines.")
                    return True
            
            # 2. Check for circular patterns (pie charts)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100,
                                       param1=50, param2=30, minRadius=30) # minRadius from original code
            if circles is not None and circles.ndim == 3 and circles.shape[1] > 0: # Check circles structure
                logger.debug("is_graph_image: Detected circular patterns.")
                return True
            
            # 3. Check for regular patterns (bars, data points)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shape_sizes = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Reasonable size range
                    shape_sizes.append(area)
            
            # Check if multiple similar-sized shapes (bars, points)
            if len(shape_sizes) > 3:
                shape_sizes_sorted = sorted(shape_sizes)
                # Check if sizes are somewhat uniform
                size_ratios = [shape_sizes_sorted[i+1] / shape_sizes_sorted[i] 
                               for i in range(len(shape_sizes_sorted)-1) 
                               if shape_sizes_sorted[i] > 0]
                
                if size_ratios and max(size_ratios) < 3:  # Similar sizes (heuristic)
                    logger.debug("is_graph_image: Detected regular patterns of similar shapes.")
                    return True
            
            # 4. Check text patterns
            if self._ocr_available:
                pil_img_for_ocr = self._cv2_to_pil(image_cv)
                if pil_img_for_ocr:
                    texts = self._extract_all_text(pil_img_for_ocr) # Ensure _extract_all_text is implemented
                    
                    # Check for numeric patterns (axis values)
                    numeric_count = sum(1 for t in texts if t.get('text') and any(c.isdigit() for c in t['text']))
                    
                    # Check for chart-like text patterns
                    has_percentage = any('%' in t['text'] for t in texts if t.get('text'))
                    has_axis_numbers = numeric_count > 3 # Heuristic: more than 3 numbers might indicate axes/data
                    
                    if has_percentage or has_axis_numbers:
                        logger.debug("is_graph_image: Detected chart-like text patterns.")
                        return True
        
        except Exception as e:
            logger.error(f"Error in is_graph_image detection: {e}", exc_info=True)
        
        return False

    # ======================
    # Advanced Chart Data Extraction (NEW - replacing old extract_graph_data)
    # ======================
    def extract_graph_data(self, image_data) -> Dict[str, Any]:
        """Extract actual data from charts/graphs"""
        if not self._graph_available:
            return {"error": "Graph extraction not available (cv2 missing or disabled).", "extracted": False}
        
        try:
            # Convert to PIL Image if needed
            pil_image = None
            if isinstance(image_data, np.ndarray): # cv2 image
                pil_image = self._cv2_to_pil(image_data)
            elif isinstance(image_data, Image.Image): # PIL image
                pil_image = image_data
            else:
                return {"error": "Unsupported image data type for graph extraction.", "extracted": False}

            if pil_image is None:
                return {"error": "Image conversion failed for graph extraction.", "extracted": False}
            
            # Detect chart type
            chart_type = self._detect_graph_type(pil_image) # Uses the new _detect_graph_type
            
            # Extract text elements from the chart
            # CRITICAL: _extract_all_text needs to be implemented
            text_elements = self._extract_all_text(pil_image)
            
            # Analyze chart structure
            # CRITICAL: _analyze_chart_structure needs to be implemented
            chart_structure = self._analyze_chart_structure(pil_image, text_elements)
            
            data = []
            # Extract data based on chart type
            if chart_type == "bar_chart":
                # CRITICAL: _extract_bar_chart_data needs to be fully implemented with its helpers
                data = self._extract_bar_chart_data(pil_image, text_elements, chart_structure)
            elif chart_type == "line_chart":
                # CRITICAL: _extract_line_chart_data needs to be fully implemented with its helpers
                data = self._extract_line_chart_data(pil_image, text_elements, chart_structure)
            elif chart_type == "pie_chart":
                # CRITICAL: _extract_pie_chart_data needs to be fully implemented with its helpers
                data = self._extract_pie_chart_data(pil_image, text_elements, chart_structure)
            elif chart_type == "scatter_plot":
                # CRITICAL: _extract_scatter_plot_data needs to be fully implemented with its helpers
                data = self._extract_scatter_plot_data(pil_image, text_elements, chart_structure)
            else: # unknown_chart or other types
                # CRITICAL: _extract_generic_chart_data needs to be implemented
                logger.info(f"Attempting generic data extraction for chart type: {chart_type}")
                data = self._extract_generic_chart_data(pil_image, text_elements, chart_structure)
            
            return {
                "graph_type": chart_type,
                "title": chart_structure.get('title', 'N/A'),
                "x_label": chart_structure.get('x_label', ''),
                "y_label": chart_structure.get('y_label', ''),
                "data": data,
                "legend": chart_structure.get('legend', []),
                "extracted": True if data else False, # Consider extracted True only if data is found
                "message": f"Data extraction attempted for {chart_type}.",
                "confidence": chart_structure.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error during graph data extraction: {e}", exc_info=True)
            return {"error": f"Extraction failed: {e}", "extracted": False}

    # ======================
    # Specific Chart Data Extractors (NEW)
    # ======================
    def _extract_bar_chart_data(self, image: Image.Image, texts: List[Dict], structure: Dict) -> List[Dict]:
        """Extract data from bar chart"""
        # CRITICAL: Needs _detect_bars and _estimate_value_from_position
        data = []
        img_cv = self._pil_to_cv2(image)
        if img_cv is None: return []
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        bars = self._detect_bars(gray) # Needs implementation
        
        for bar in bars:
            x, y_bar, w, h = bar # y_bar is the top of the bar
            bar_center_x = x + w/2
            # bar_top_y = y_bar # for vertical bars growing upwards
            # bar_bottom_y = y_bar + h

            # Find label (usually below or beside bar, depends on orientation)
            # This logic needs to be robust to bar orientation (vertical/horizontal)
            label_texts = [t for t in texts if 
                           abs(t['center_x'] - bar_center_x) < w * 1.5 and # Wider search for labels
                           t['center_y'] > y_bar + h ] # Assuming vertical bars, labels below
            
            label = label_texts[0]['text'] if label_texts else f"Bar_{len(data)+1}"
            
            # Find value (usually above, on bar, or requires estimation)
            value_texts = [t for t in texts if 
                           abs(t['center_x'] - bar_center_x) < w and
                           abs(t['center_y'] - y_bar) < h/2 + 20] # Search near the top of the bar
            
            value = None
            if value_texts:
                try:
                    value = float(value_texts[0]['text'].replace(',', '').strip())
                except ValueError:
                    logger.debug(f"Could not parse float from bar value text: {value_texts[0]['text']}")
                    pass
            
            if value is None and structure.get('axes_bounds') and structure['axes_bounds'].get('y_val_min') is not None:
                 # Assuming vertical bar, value is proportional to height
                value = self._estimate_value_from_position(y_bar, structure['axes_bounds'], 'vertical_bar_top')
            
            data.append({
                'label': label,
                'value': value,
                'bar_position': {'x': x, 'y': y_bar, 'width': w, 'height': h}
            })
        
        return data

    def _extract_pie_chart_data(self, image: Image.Image, texts: List[Dict], structure: Dict) -> List[Dict]:
        """Extract data from pie chart"""
        # CRITICAL: Needs robust _distance and possibly color segmentation for slices if no text
        data = []
        
        percentage_texts = [t for t in texts if t.get('text') and '%' in t['text']]
        
        for pct_text_info in percentage_texts:
            pct_text = pct_text_info['text']
            try:
                pct_str = pct_text.replace('%', '').strip()
                percentage = float(pct_str)
                
                # Find nearest non-percentage text as label
                # The distance threshold (100) is arbitrary and might need tuning or be made dynamic
                # based on image size or text size.
                nearby_texts = sorted(
                    [t for t in texts if
                     t != pct_text_info and t.get('text') and '%' not in t['text'] and
                     self._distance(pct_text_info['center_x'], pct_text_info['center_y'],
                                    t['center_x'], t['center_y']) < 100], # 100 is a magic number
                    key=lambda t: self._distance(pct_text_info['center_x'], pct_text_info['center_y'],
                                                 t['center_x'], t['center_y'])
                )
                
                label = nearby_texts[0]['text'] if nearby_texts else f"Slice_{len(data)+1}"
                
                data.append({
                    'label': label,
                    'percentage': percentage,
                    'value': percentage # Can be converted to absolute if total is known or other values are present
                })
                
            except ValueError:
                logger.debug(f"Could not parse float from pie chart percentage: {pct_text}")
                continue
        
        data.sort(key=lambda x: x['percentage'], reverse=True)
        
        return data

    def _extract_line_chart_data(self, image: Image.Image, texts: List[Dict], structure: Dict) -> List[Dict]:
        """Extract data from line chart"""
        # CRITICAL: Needs _extract_axis_values, _detect_line_points, _estimate_value_from_position
        data = []
        
        # Extract x-axis labels (usually at bottom)
        # _extract_axis_values needs implementation
        x_axis_info = self._extract_axis_values(texts, image.size, 'x') # [{'value': val, 'position_pixel': px}, ...]
        
        # Extract y-axis values (usually at left) - primarily for scale, not direct data usually
        # y_axis_info = self._extract_axis_values(texts, image.size, 'y')

        # Detect line points using image processing
        img_cv = self._pil_to_cv2(image)
        if img_cv is None: return []
        
        points = self._detect_line_points(img_cv) # Needs implementation: list of (x_pixel, y_pixel)
        
        if not structure.get('axes_bounds'):
            logger.warning("Cannot extract line chart data without axes_bounds in structure.")
            return []

        # Match points with x-labels or interpolate
        for p_idx, point_px in enumerate(points): # Assuming points are somewhat ordered or represent distinct data points
            x_pixel, y_pixel = point_px

            # Find closest x-axis label by pixel position or interpolate
            # This part is complex: you might need to map pixel x to an x-value
            # then find if that x-value exists in x_axis_info or interpolate
            x_value_label = f"Point_{p_idx + 1}" # Placeholder
            
            current_x_val = self._estimate_value_from_position(x_pixel, structure['axes_bounds'], 'horizontal')

            # Try to find a matching text label for x_value if available from x_axis_info
            closest_x_label = None
            min_dist = float('inf')
            if x_axis_info:
                for x_lab in x_axis_info:
                    dist = abs(x_lab['position_pixel'] - x_pixel)
                    if dist < min_dist and dist < 20: # 20px threshold, heuristic
                        min_dist = dist
                        closest_x_label = x_lab['value']
                if closest_x_label:
                    x_value_label = closest_x_label
                else: # Use estimated value if no text label is close
                    x_value_label = current_x_val


            y_value = self._estimate_value_from_position(y_pixel, structure['axes_bounds'], 'vertical')
            
            data.append({
                'x': x_value_label, # Could be numeric or categorical
                'y': y_value,
                'position': {'x': x_pixel, 'y': y_pixel}
            })
        
        # Sort data if x values are numeric and can be sorted
        try:
            data.sort(key=lambda item: float(item['x']))
        except (ValueError, TypeError):
            logger.debug("Cannot sort line chart data by x-value as it's not consistently numeric.")
            pass # Keep original order if x is not numeric

        return data

    def _extract_scatter_plot_data(self, image: Image.Image, texts: List[Dict], structure: Dict) -> List[Dict]:
        """Extract data from scatter plot"""
        # CRITICAL: Needs _detect_scatter_points, _estimate_value_from_position, _extract_axis_values
        data = []
        
        img_cv = self._pil_to_cv2(image)
        if img_cv is None: return []

        points = self._detect_scatter_points(img_cv) # Needs implementation: list of (x_pixel, y_pixel)

        if not structure.get('axes_bounds'):
            logger.warning("Cannot extract scatter plot data without axes_bounds in structure.")
            return []
            
        for point_px in points:
            x_pixel, y_pixel = point_px
            x_val = self._estimate_value_from_position(x_pixel, structure['axes_bounds'], 'horizontal')
            y_val = self._estimate_value_from_position(y_pixel, structure['axes_bounds'], 'vertical')
            
            data.append({
                'x': x_val,
                'y': y_val,
                'position': {'x': x_pixel, 'y': y_pixel}
            })
        
        return data

    # ======================
    # Improved Graph Type Detection (NEW - replacing old _detect_graph_type)
    # ======================
    def _detect_graph_type(self, image: Image.Image) -> str: # Parameter changed to PIL.Image to match usage
        """Detect the type of chart/graph"""
        img_cv = self._pil_to_cv2(image) # Convert PIL to CV2
        if img_cv is None:
            logger.warning("_detect_graph_type: CV2 image conversion failed.")
            return "unknown_chart"
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect circles (pie charts)
        # Adjusted minDist for HoughCircles to be relative to image size
        # Params for HoughCircles often need tuning
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=max(width, height)//4, # min dist between centers
                                   param1=50, param2=30, # Canny edge detector thresholds
                                   minRadius=int(min(width, height) * 0.1), # Smallest radius
                                   maxRadius=int(min(width,height) * 0.45)) # Largest radius
                                   
        if circles is not None and len(circles[0]) > 0:
            # Check if it's likely a pie chart by looking for radial lines or characteristic text
            # CRITICAL: _has_radial_lines needs implementation
            # For now, a strong circle detection might be enough for a basic pie_chart guess
            # Consider checking text elements for percentages near the circle center
            logger.debug(f"HoughCircles detected {len(circles[0])} circle(s).")
            # Assuming the largest detected circle is the pie chart
            main_circle = circles[0][0] # x, y, radius
            if self._has_radial_lines(gray, (int(main_circle[0]), int(main_circle[1])), int(main_circle[2])): # Needs implementation
                 return "pie_chart"
            # Fallback if _has_radial_lines is not robust:
            # If many texts nearby have '%', it could be a pie chart.
            # This check could also be part of _analyze_chart_structure
            logger.info("Detected circle(s), potentially a pie chart. Radial line check pending implementation.")
            # For now, let's assume a dominant circle is a pie chart if other checks fail
            # This could be refined by checking for percentage signs in OCR text near the circle.


        # Detect rectangles (bar charts)
        edges = cv2.Canny(gray, 50, 150) # Standard Canny thresholds
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        min_bar_area = (width * height) * 0.001 # Bar area should be at least 0.1% of image
        max_bar_area = (width * height) * 0.20  # Bar area no more than 20% of image
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4: # It's a quadrilateral
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(approx)
                # Basic aspect ratio check for bars (can be vertical or horizontal)
                aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0
                area = w_rect * h_rect
                
                # Filter for bar-like rectangles: not too small, not too large, reasonable aspect ratio
                if (0.05 < aspect_ratio < 20.0) and (min_bar_area < area < max_bar_area) :
                    # Check if it's filled or mostly solid (advanced)
                    rectangles.append((x_rect, y_rect, w_rect, h_rect))
        
        if len(rectangles) > 2: # Need at least 3 bars to be confident
            # CRITICAL: _are_rectangles_aligned needs implementation
            if self._are_rectangles_aligned(rectangles):
                logger.debug(f"Detected {len(rectangles)} aligned rectangles, likely a bar chart.")
                return "bar_chart"
            else:
                logger.debug(f"Detected {len(rectangles)} rectangles, but not clearly aligned for bar chart.")

        # Detect continuous lines (line charts)
        # HoughLinesP parameters might need tuning based on typical line thickness and length in charts
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=max(width, height)//20, maxLineGap=10)
        if lines is not None and len(lines) > 1 : # Need more than one line segment
            # CRITICAL: _has_connected_lines needs implementation
            # This is tricky: needs to distinguish data lines from axis lines or grid lines
            # Look for non-horizontal and non-vertical dominant lines
            non_hv_lines = 0
            for line_arr in lines:
                for x1,y1,x2,y2 in line_arr:
                    angle_rad = math.atan2(y2-y1, x2-x1)
                    angle_deg = math.degrees(angle_rad)
                    # Check if line is not strictly horizontal or vertical (within a tolerance)
                    if not (abs(angle_deg) < 5 or abs(angle_deg-180) < 5 or \
                            abs(angle_deg-90) < 5 or abs(angle_deg-270) < 5):
                        non_hv_lines +=1
            
            if non_hv_lines > 1 and self._has_connected_lines(lines): # Needs better _has_connected_lines
                logger.debug(f"Detected {len(lines)} line segments, with {non_hv_lines} non-HV lines, potentially a line chart.")
                return "line_chart"

        # Check for scatter plot (many small circles or points)
        # CRITICAL: _has_many_points needs implementation
        if self._has_many_points(gray): # Needs implementation
            logger.debug("Detected many small points, likely a scatter plot.")
            return "scatter_plot"
        
        # If a dominant circle was found earlier and other types weren't strongly identified
        if circles is not None and len(circles[0]) > 0:
            logger.info("Reconsidering pie_chart due to circle detection and no other strong match.")
            return "pie_chart"

        logger.info("Could not determine specific chart type, defaulting to unknown_chart.")
        return "unknown_chart"

    # _extract_axis_labels and _extract_data_points are effectively replaced by the new
    # chart-specific extraction methods and _analyze_chart_structure.
    # Keeping them as stubs for now if any old code path might call them, but they should be deprecated.
    def _extract_axis_labels(self, image_np) -> Tuple[str, str]:
        # Placeholder - This functionality should be part of _analyze_chart_structure
        logger.warning("Deprecated: _extract_axis_labels called. Use _analyze_chart_structure.")
        return "", ""

    def _extract_data_points(self, image_np, graph_type) -> list: # list of Dict[str, Any]
        # Placeholder - This functionality is now handled by specific extract_*_chart_data methods
        logger.warning("Deprecated: _extract_data_points called. Use specific extract_*_chart_data methods.")
        return []