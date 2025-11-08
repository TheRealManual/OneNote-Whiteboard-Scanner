"""
InkML export module
Converts strokes to InkML format for OneNote integration
"""

import logging
from typing import List
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from ai.stroke_extract import Stroke

logger = logging.getLogger(__name__)


def strokes_to_inkml(
    strokes: List[Stroke],
    width: int,
    height: int
) -> str:
    """
    Convert list of strokes to InkML format for OneNote
    
    InkML is the native format used by OneNote for digital ink,
    which makes it fully editable after import.
    
    Args:
        strokes: List of Stroke objects
        width: Canvas width
        height: Canvas height
        
    Returns:
        InkML XML string
    """
    try:
        # Create InkML root with proper namespace
        ink = Element('ink')
        ink.set('xmlns', 'http://www.w3.org/2003/InkML')
        
        # Add canvas dimensions as annotation
        annotation = SubElement(ink, 'annotation')
        annotation.set('type', 'canvas')
        annotation.text = f'{width}x{height}'
        
        # Add definitions for brushes
        definitions = SubElement(ink, 'definitions')
        
        # Track unique colors and create brushes
        unique_colors = {}
        for i, stroke in enumerate(strokes):
            if stroke.color not in unique_colors:
                brush_id = f'brush-{len(unique_colors)}'
                unique_colors[stroke.color] = brush_id
                
                # Create brush definition
                brush = SubElement(definitions, 'brush')
                brush.set('xml:id', brush_id)
                
                # Color
                color_elem = SubElement(brush, 'annotation')
                color_elem.set('type', 'color')
                color_elem.text = stroke.color
                
                # Width
                width_elem = SubElement(brush, 'annotation')
                width_elem.set('type', 'width')
                width_elem.text = str(stroke.thickness)
        
        # Add trace group for all strokes
        trace_group = SubElement(ink, 'traceGroup')
        
        # Add each stroke as a trace
        for i, stroke in enumerate(strokes):
            trace = SubElement(trace_group, 'trace')
            trace.set('id', f'trace-{i}')
            
            # Reference the brush
            brush_id = unique_colors[stroke.color]
            trace.set('brushRef', f'#{brush_id}')
            
            # Convert points to InkML format (space-separated X Y pairs)
            points_str = ' '.join(f'{int(p[0])} {int(p[1])}' for p in stroke.points)
            trace.text = points_str
        
        # Convert to pretty-printed XML string
        inkml_string = prettify_xml(ink)
        
        logger.info(f"Generated InkML with {len(strokes)} traces ({len(unique_colors)} colors)")
        return inkml_string
        
    except Exception as e:
        logger.error(f"Error generating InkML: {e}")
        return ""


def prettify_xml(elem: Element) -> str:
    """Return a pretty-printed XML string"""
    rough_string = tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def validate_inkml(inkml_string: str) -> bool:
    """
    Validate InkML format
    
    Args:
        inkml_string: InkML XML string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Basic validation - check for required elements
        required_elements = ['ink', 'trace']
        for elem in required_elements:
            if f'<{elem}' not in inkml_string and f'<{elem} ' not in inkml_string:
                logger.error(f"InkML validation failed: missing <{elem}> element")
                return False
        
        # Check namespace
        if 'http://www.w3.org/2003/InkML' not in inkml_string:
            logger.warning("InkML missing proper namespace")
        
        return True
        
    except Exception as e:
        logger.error(f"InkML validation error: {e}")
        return False
