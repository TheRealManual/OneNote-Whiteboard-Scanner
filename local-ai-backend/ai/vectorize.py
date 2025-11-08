"""
Vectorization module
Converts extracted strokes to SVG format
"""

import numpy as np
from typing import List, Tuple
import logging
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from ai.stroke_extract import Stroke

logger = logging.getLogger(__name__)


def strokes_to_svg(
    strokes: List[Stroke],
    width: int,
    height: int,
    background_color: str = "none"  # Changed from "#ffffff" to "none" for transparency
) -> str:
    """
    Convert list of strokes to SVG format
    
    Args:
        strokes: List of Stroke objects
        width: Canvas width
        height: Canvas height
        background_color: Background color (default transparent)
        
    Returns:
        SVG string
    """
    try:
        # Create SVG root element
        svg = Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f'0 0 {width} {height}')
        
        # Only add background if not transparent
        if background_color != "none":
            background = SubElement(svg, 'rect')
            background.set('width', str(width))
            background.set('height', str(height))
            background.set('fill', background_color)
        
        # Create group for strokes
        strokes_group = SubElement(svg, 'g')
        strokes_group.set('id', 'whiteboard-strokes')
        
        # Add each stroke as a path
        for i, stroke in enumerate(strokes):
            path_element = stroke_to_path_element(stroke, stroke_id=f'stroke-{i}')
            strokes_group.append(path_element)
        
        # Convert to pretty-printed string
        svg_string = prettify_xml(svg)
        
        logger.info(f"Generated SVG with {len(strokes)} strokes ({width}x{height})")
        return svg_string
        
    except Exception as e:
        logger.error(f"Error generating SVG: {e}")
        return ""


def stroke_to_path_element(stroke: Stroke, stroke_id: str = None) -> Element:
    """
    Convert a single stroke to an SVG path element
    
    Args:
        stroke: Stroke object
        stroke_id: Optional ID for the path element
        
    Returns:
        SVG path Element
    """
    path = Element('path')
    
    if stroke_id:
        path.set('id', stroke_id)
    
    # Generate path data string
    path_data = points_to_path_data(stroke.points)
    path.set('d', path_data)
    
    # Set stroke attributes
    path.set('stroke', stroke.color)
    path.set('stroke-width', f'{stroke.thickness:.2f}')
    path.set('stroke-linecap', 'round')
    path.set('stroke-linejoin', 'round')
    path.set('fill', 'none')
    
    # Add opacity for lighter colors
    if should_add_opacity(stroke.color):
        path.set('opacity', '0.9')
    
    return path


def points_to_path_data(points: np.ndarray, smooth: bool = True) -> str:
    """
    Convert array of points to SVG path data string
    
    Args:
        points: Array of (x, y) coordinates
        smooth: Whether to use cubic Bezier curves for smoothing
        
    Returns:
        SVG path data string
    """
    if len(points) == 0:
        return ""
    
    if len(points) == 1:
        # Single point - draw a small circle
        x, y = points[0]
        return f"M {x:.2f},{y:.2f} L {x+0.1:.2f},{y:.2f}"
    
    # Start with Move command
    path_parts = [f"M {points[0][0]:.2f},{points[0][1]:.2f}"]
    
    if smooth and len(points) > 2:
        # Use cubic Bezier curves for smooth paths
        bezier_commands = points_to_bezier(points)
        path_parts.extend(bezier_commands)
    else:
        # Use simple line segments
        for point in points[1:]:
            path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")
    
    return " ".join(path_parts)


def points_to_bezier(points: np.ndarray) -> List[str]:
    """
    Convert points to smooth cubic Bezier curves
    
    Uses Catmull-Rom to Bezier conversion for smooth interpolation
    """
    bezier_commands = []
    
    if len(points) < 3:
        for point in points[1:]:
            bezier_commands.append(f"L {point[0]:.2f},{point[1]:.2f}")
        return bezier_commands
    
    # For each segment, create a cubic Bezier curve
    for i in range(len(points) - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(len(points) - 1, i + 2)]
        
        # Calculate control points
        cp1, cp2 = catmull_rom_to_bezier(p0, p1, p2, p3)
        
        # Create cubic Bezier command
        bezier_commands.append(
            f"C {cp1[0]:.2f},{cp1[1]:.2f} {cp2[0]:.2f},{cp2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}"
        )
    
    return bezier_commands


def catmull_rom_to_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    tension: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Catmull-Rom spline segment to cubic Bezier control points
    
    Args:
        p0, p1, p2, p3: Four consecutive points
        tension: Tension parameter (0.5 = Catmull-Rom)
        
    Returns:
        Two control points (cp1, cp2)
    """
    # Calculate control points
    cp1 = p1 + (p2 - p0) / (6.0 / tension)
    cp2 = p2 - (p3 - p1) / (6.0 / tension)
    
    return cp1, cp2


def should_add_opacity(color: str) -> bool:
    """
    Determine if a color should have reduced opacity
    (e.g., for highlighter effect on light colors)
    """
    # Convert hex to RGB
    hex_color = color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Check if color is light (high brightness)
    brightness = (r + g + b) / 3
    
    return brightness > 200


def prettify_xml(elem: Element) -> str:
    """
    Return a pretty-printed XML string
    """
    rough_string = tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def strokes_to_inkml(strokes: List[Stroke]) -> str:
    """
    Convert strokes to InkML format (alternative to SVG)
    InkML is specifically designed for digital ink
    
    Args:
        strokes: List of Stroke objects
        
    Returns:
        InkML XML string
    """
    # Create InkML root
    ink = Element('ink')
    ink.set('xmlns', 'http://www.w3.org/2003/InkML')
    
    # Add definitions for colors
    definitions = SubElement(ink, 'definitions')
    
    # Track unique colors
    unique_colors = set(stroke.color for stroke in strokes)
    
    for i, color in enumerate(unique_colors):
        brush = SubElement(definitions, 'brush')
        brush.set('xml:id', f'brush-{i}')
        
        color_elem = SubElement(brush, 'color')
        color_elem.text = color
        
        width_elem = SubElement(brush, 'width')
        width_elem.text = '2.0'
    
    # Add traces (strokes)
    for i, stroke in enumerate(strokes):
        trace = SubElement(ink, 'trace')
        trace.set('id', f'trace-{i}')
        
        # Find brush ID for this color
        brush_id = f"brush-{list(unique_colors).index(stroke.color)}"
        trace.set('brushRef', f'#{brush_id}')
        
        # Add points as space-separated X Y pairs
        points_str = ' '.join(f'{p[0]:.2f} {p[1]:.2f}' for p in stroke.points)
        trace.text = points_str
    
    return prettify_xml(ink)


def optimize_svg(svg_string: str) -> str:
    """
    Optimize SVG by removing unnecessary whitespace and reducing precision
    """
    # This is a placeholder for SVG optimization
    # Could use libraries like scour or svgo
    return svg_string


def add_svg_metadata(svg_string: str, metadata: dict) -> str:
    """
    Add metadata to SVG for tracking processing information
    """
    # Could add processing metadata like:
    # - timestamp
    # - processing parameters
    # - source image info
    return svg_string
