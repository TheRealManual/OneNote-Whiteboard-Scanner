"""
Stroke extraction module
Extracts individual strokes from color-segmented masks
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from config import MIN_CONTOUR_AREA, MAX_CONTOUR_AREA

# Stroke processing constants (moved from config.py)
MIN_STROKE_LENGTH = 10  # Minimum stroke length in pixels
EPSILON_FACTOR = 0.002  # Curve approximation factor

logger = logging.getLogger(__name__)


class Stroke:
    """Represents a single stroke with its properties"""
    
    def __init__(self, points: np.ndarray, color: str, thickness: float = 2.0):
        self.points = points  # Array of (x, y) coordinates
        self.color = color    # Hex color string
        self.thickness = thickness
        self.length = self._calculate_length()
        self.bbox = self._calculate_bbox()
    
    def _calculate_length(self) -> float:
        """Calculate total stroke length"""
        if len(self.points) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(self.points) - 1):
            dx = self.points[i+1][0] - self.points[i][0]
            dy = self.points[i+1][1] - self.points[i][1]
            total += np.sqrt(dx*dx + dy*dy)
        return total
    
    def _calculate_bbox(self) -> Tuple[int, int, int, int]:
        """Calculate bounding box (x, y, width, height)"""
        if len(self.points) == 0:
            return (0, 0, 0, 0)
        
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stroke to dictionary for serialization"""
        return {
            "points": self.points.tolist(),
            "color": self.color,
            "thickness": self.thickness,
            "length": self.length,
            "bbox": self.bbox
        }


def extract_strokes(mask: np.ndarray, color: str) -> List[Stroke]:
    """
    Extract individual strokes from a binary mask
    
    Args:
        mask: Binary mask of a single color
        color: Hex color string for this mask
        
    Returns:
        List of Stroke objects
    """
    try:
        strokes = []
        
        # Find contours in the mask (CHAIN_APPROX_NONE = keep all points)
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE  # Changed from CHAIN_APPROX_SIMPLE - keep all contour points
        )
        
        logger.info(f"Found {len(contours)} contours for color {color}")
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            
            # Convert to stroke points (all contour points)
            points = contour.reshape(-1, 2).astype(float)
            
            # Skip if too few points
            if len(points) < 10:
                continue
            
            # Apply HEAVY multi-pass Gaussian smoothing to create silky smooth curves
            if len(points) >= 10:
                # Pass 1: Strong smoothing (window 11)
                points = smooth_stroke(points, window_size=11)
                # Pass 2: Medium smoothing (window 7)
                points = smooth_stroke(points, window_size=7)
                # Pass 3: Light smoothing (window 5) to polish
                points = smooth_stroke(points, window_size=5)
            
            # SIMPLIFIED TOO AGGRESSIVELY - DON'T simplify smooth curves!
            # The Bezier curve fitting in vectorize.py will handle reduction
            # Keeping smooth points preserves curve quality
            # 
            # OLD: epsilon = 0.5 threw away 90% of points and re-introduced jaggedness
            # NEW: Skip simplification, let SVG Bezier curves do the work
            
            # Skip if too short
            if len(points) < 2:
                continue
            
            # Estimate stroke thickness from contour area and perimeter
            perimeter = cv2.arcLength(contour, True)
            thickness = estimate_stroke_thickness(area, perimeter)
            
            # Create stroke object
            stroke = Stroke(points, color, thickness)
            
            # Filter by minimum length
            if stroke.length >= MIN_STROKE_LENGTH:
                strokes.append(stroke)
        
        logger.info(f"Extracted {len(strokes)} valid strokes for color {color}")
        return strokes
        
    except Exception as e:
        logger.error(f"Error extracting strokes: {e}")
        return []


def estimate_stroke_thickness(area: float, perimeter: float) -> float:
    """
    Estimate stroke thickness based on contour geometry
    
    Uses the relationship: thickness ≈ 2 * area / perimeter
    """
    if perimeter == 0:
        return 2.0
    
    thickness = (2.0 * area) / perimeter
    
    # Clamp to reasonable range (1-10 pixels)
    thickness = max(1.0, min(10.0, thickness))
    
    return thickness


def smooth_stroke(points: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth stroke points using Gaussian-weighted moving average
    
    Args:
        points: Array of (x, y) coordinates
        window_size: Size of smoothing window
        
    Returns:
        Smoothed points array
    """
    if len(points) < window_size:
        return points
    
    smoothed = np.copy(points).astype(float)
    
    # Create Gaussian weights for better smoothing
    sigma = window_size / 3.0
    weights = np.exp(-0.5 * ((np.arange(window_size) - window_size // 2) / sigma) ** 2)
    weights = weights / weights.sum()
    
    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        
        # Get window and corresponding weights
        window = points[start:end]
        window_weights = weights[window_size // 2 - (i - start):window_size // 2 + (end - i)]
        window_weights = window_weights / window_weights.sum()
        
        # Weighted average
        smoothed[i] = np.average(window, axis=0, weights=window_weights)
    
    return smoothed


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Apply skeletonization to get centerline of strokes
    Useful for thick marker strokes
    """
    from skimage.morphology import skeletonize
    
    # Convert to binary
    binary = mask > 0
    
    # Skeletonize
    skeleton = skeletonize(binary)
    
    # Convert back to uint8
    return (skeleton * 255).astype(np.uint8)


def extract_strokes_from_skeleton(skeleton: np.ndarray, color: str) -> List[Stroke]:
    """
    Extract strokes from skeletonized image
    Better for thick markers
    """
    strokes = []
    
    # Find branch points and endpoints
    kernel = np.ones((3, 3), np.uint8)
    
    # Dilate slightly to connect nearby points
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Find contours on skeleton
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if len(contour) < 2:
            continue
        
        points = contour.reshape(-1, 2).astype(float)
        
        # Smooth the skeleton path
        smoothed = smooth_stroke(points)
        
        stroke = Stroke(smoothed, color, thickness=3.0)
        
        if stroke.length >= MIN_STROKE_LENGTH:
            strokes.append(stroke)
    
    return strokes


def simplify_stroke(stroke: Stroke, tolerance: float = 2.0) -> Stroke:
    """
    Simplify stroke by reducing number of points while preserving shape
    Uses Ramer-Douglas-Peucker algorithm
    """
    if len(stroke.points) < 3:
        return stroke
    
    # Convert to contour format
    contour = stroke.points.reshape(-1, 1, 2).astype(np.float32)
    
    # Simplify
    epsilon = tolerance
    simplified = cv2.approxPolyDP(contour, epsilon, False)
    
    # Convert back to points
    new_points = simplified.reshape(-1, 2)
    
    return Stroke(new_points, stroke.color, stroke.thickness)


def connect_broken_strokes(strokes: List[Stroke], max_gap: float = 10.0) -> List[Stroke]:
    """
    Connect strokes that appear to be broken parts of the same line
    
    Args:
        strokes: List of stroke objects
        max_gap: Maximum pixel distance to consider strokes connected
        
    Returns:
        List of strokes with broken parts connected
    """
    if len(strokes) < 2:
        return strokes
    
    connected = []
    used = set()
    
    for i, stroke1 in enumerate(strokes):
        if i in used:
            continue
        
        # Start with this stroke
        merged_points = list(stroke1.points)
        used.add(i)
        
        # Look for nearby endpoints
        while True:
            found_connection = False
            endpoint = merged_points[-1]
            
            for j, stroke2 in enumerate(strokes):
                if j in used or stroke1.color != stroke2.color:
                    continue
                
                # Check distance to start of stroke2
                start_point = stroke2.points[0]
                distance = np.linalg.norm(endpoint - start_point)
                
                if distance <= max_gap:
                    # Connect strokes
                    merged_points.extend(stroke2.points[1:])
                    used.add(j)
                    found_connection = True
                    break
            
            if not found_connection:
                break
        
        # Create merged stroke
        merged = Stroke(
            np.array(merged_points),
            stroke1.color,
            stroke1.thickness
        )
        connected.append(merged)
    
    logger.info(f"Connected {len(strokes)} strokes into {len(connected)} strokes")
    return connected
