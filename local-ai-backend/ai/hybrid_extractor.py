"""
Hybrid Stroke Extractor - Fast & Robust Whiteboard Processing
Combines classical CV with tiny AI for 1-3 second processing
Optimized for Intel Iris Xe Graphics with OpenVINO
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging
from pathlib import Path
import json

# Optional AI imports
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
    # Set threading early to avoid conflicts
    core = ov.Core()
    core.set_property({"INFERENCE_NUM_THREADS": 4})
except ImportError:
    OPENVINO_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from skimage.morphology import skeletonize, remove_small_objects
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / 'config_hybrid.json'
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)
else:
    # Default config
    CONFIG = {
        "resize_to": [960, 540],
        "rectified_canvas": [1280, 720],
        "illumination": {"enabled": True, "blur_sigma": 21, "clahe": True},
        "stroke_extract": {
            "use_classical": True,
            "use_u2net": False,  # Disabled by default until model downloaded
            "color_hsv_masks": True,
            "canny": [80, 200],  # Increased thresholds to ignore weak edges
            "morph_open": 3,  # Increased to remove more noise
            "morph_close": 3,
            "min_area_px": 500  # Much higher - only keep substantial strokes
        },
        "vectorize": {
            "skeletonize": True,
            "gap_close_px": 4,
            "rdp_epsilon_px": 1.0,
            "stroke_width_px": 2.5,
            "transparent_bg": True
        }
    }


class HybridStrokeExtractor:
    """Fast hybrid stroke extraction using classical CV + optional tiny AI"""
    
    def __init__(self):
        self.u2net_session = None
        self.backend_type = None
        
        if CONFIG['stroke_extract'].get('use_u2net', False):
            self._init_u2net()
    
    def _init_u2net(self):
        """Initialize U2-Net model with best available backend"""
        model_path = Path(__file__).parent.parent / 'models' / 'u2netp.onnx'
        
        if not model_path.exists():
            logger.warning(f"U2-Net model not found at {model_path}")
            logger.info("Download from: https://github.com/xuebinqin/U-2-Net")
            CONFIG['stroke_extract']['use_u2net'] = False
            return
        
        # Try backends in order: OpenVINO GPU > ONNX DML > ONNX CPU
        backend = CONFIG['stroke_extract'].get('u2net_backend', 'auto')
        
        if backend == 'auto' or backend == 'openvino':
            if OPENVINO_AVAILABLE:
                try:
                    # Try OpenVINO first
                    logger.info("Initializing U2-Net with OpenVINO...")
                    self.backend_type = 'openvino'
                    # Would need converted model here
                    logger.warning("OpenVINO model conversion needed, falling back to ONNX")
                except Exception as e:
                    logger.warning(f"OpenVINO init failed: {e}")
        
        # Fallback to ONNX Runtime
        if self.u2net_session is None and ONNX_AVAILABLE:
            try:
                providers = []
                
                # Try DirectML for Intel GPU
                if 'DmlExecutionProvider' in ort.get_available_providers():
                    providers.append('DmlExecutionProvider')
                    logger.info("U2-Net using DirectML (Intel GPU)")
                
                # Fallback to CPU
                providers.append('CPUExecutionProvider')
                
                self.u2net_session = ort.InferenceSession(
                    str(model_path),
                    providers=providers
                )
                self.backend_type = 'onnx'
                logger.info(f"✓ U2-Net loaded with {providers[0]}")
                
            except Exception as e:
                logger.error(f"Failed to load U2-Net: {e}")
                CONFIG['stroke_extract']['use_u2net'] = False
    
    def process_image(self, img: np.ndarray) -> Dict:
        """
        Main processing pipeline
        
        Returns:
            Dict with 'strokes', 'mask', 'metadata'
        """
        import time
        start = time.time()
        
        logger.info("="*60)
        logger.info("STARTING IMAGE PROCESSING PIPELINE")
        logger.info("="*60)
        logger.info(f"Input image shape: {img.shape} (H×W×C)")
        logger.info(f"Input image dtype: {img.dtype}")
        logger.info(f"Input image size: {img.shape[1]}×{img.shape[0]}")
        
        # Store original for color sampling
        self.original_img = img.copy()
        logger.info("Stored original image for color sampling")
        
        # Step 1: Ingest & Normalize
        logger.info("="*60)
        logger.info("Step 1: Normalizing image...")
        logger.info("="*60)
        normalized = self._normalize_image(img)
        logger.info(f"After normalization: {normalized.shape}")
        
        # Step 2: Whiteboard Detection & Rectification
        logger.info("="*60)
        logger.info("Step 2: Detecting whiteboard...")
        logger.info("="*60)
        rectified, transform_matrix = self._detect_and_rectify(normalized)
        logger.info(f"After rectification: {rectified.shape}")
        logger.info(f"Transform matrix: {'Applied' if transform_matrix is not None else 'None (used resize)'}")
        
        # Step 3: Stroke Isolation (Hybrid)
        logger.info("="*60)
        logger.info("Step 3: Extracting strokes...")
        logger.info("="*60)
        logger.info(f"Input to stroke extraction: {rectified.shape}")
        stroke_mask = self._extract_strokes_hybrid(rectified)
        logger.info(f"Stroke mask shape: {stroke_mask.shape}")
        logger.info(f"Stroke mask dtype: {stroke_mask.dtype}")
        logger.info(f"Non-zero pixels in mask: {np.count_nonzero(stroke_mask)}")
        
        # Step 4: Vectorize
        logger.info("="*60)
        logger.info("Step 4: Vectorizing...")
        logger.info("="*60)
        strokes = self._vectorize_mask(stroke_mask, rectified)
        logger.info(f"Number of strokes generated: {len(strokes)}")
        
        elapsed = time.time() - start
        logger.info("="*60)
        logger.info(f"✓ PROCESSING COMPLETE in {elapsed:.2f}s")
        logger.info("="*60)
        
        return {
            'strokes': strokes,
            'mask': stroke_mask,
            'rectified': rectified,
            'metadata': {
                'processing_time': elapsed,
                'num_strokes': len(strokes),
                'backend': self.backend_type or 'classical_only'
            }
        }
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image with white balance and illumination correction"""
        logger.info(f"Input to normalize: {img.shape}")
        
        # Resize - CONFIG stores [width, height]
        target_w, target_h = CONFIG['resize_to']
        target_size = (target_w, target_h)  # cv2.resize wants (width, height)
        logger.info(f"Target resize: {target_size} (W×H)")
        img = cv2.resize(img, target_size)
        logger.info(f"After resize: {img.shape} (H×W×C)")
        
        if not CONFIG['illumination']['enabled']:
            logger.info("Illumination correction disabled")
            return img
        
        logger.info("Applying white balance (GrayWorld)...")
        # White balance (GrayWorld)
        img_float = img.astype(np.float32)
        avg_b, avg_g, avg_r = cv2.mean(img_float)[:3]
        avg = (avg_b + avg_g + avg_r) / 3.0
        logger.info(f"Channel averages - B: {avg_b:.1f}, G: {avg_g:.1f}, R: {avg_r:.1f}, Avg: {avg:.1f}")
        
        img_float[:,:,0] *= avg / avg_b if avg_b > 0 else 1.0
        img_float[:,:,1] *= avg / avg_g if avg_g > 0 else 1.0
        img_float[:,:,2] *= avg / avg_r if avg_r > 0 else 1.0
        img = np.clip(img_float, 0, 255).astype(np.uint8)
        logger.info("White balance applied")
        
        # Illumination flattening with stronger background removal
        logger.info("Applying illumination flattening...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_sigma = CONFIG['illumination']['blur_sigma']
        logger.info(f"Gaussian blur sigma: {blur_sigma} (increased for better flattening)")
        bg = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
        flat = cv2.divide(gray, bg, scale=255)
        
        if CONFIG['illumination']['clahe']:
            logger.info("Applying CLAHE histogram equalization...")
            flat = cv2.equalizeHist(flat)
        
        # Apply back to color channels
        img_flat = img.copy()
        for i in range(3):
            img_flat[:,:,i] = cv2.divide(img[:,:,i], bg, scale=255)
        
        logger.info("Illumination correction complete")
        return img_flat
    
    def _detect_and_rectify(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect whiteboard and rectify perspective"""
        logger.info(f"Detecting quad in image of size: {img.shape}")
        
        # Try contour-based quad detection first
        logger.info("Trying contour-based quad detection...")
        quad = self._find_quad_contour(img)
        
        if quad is None:
            logger.info("Contour detection failed, trying Hough lines...")
            # Try Hough lines
            quad = self._find_quad_hough(img)
        
        if quad is None:
            logger.info("No quad detected - using full image as whiteboard")
            # Use full image dimensions but respect configured canvas aspect ratio
            canvas_w, canvas_h = CONFIG['rectified_canvas']
            logger.info(f"Configured canvas size: {canvas_w}×{canvas_h}")
            
            if CONFIG.get('preserve_aspect_ratio', True):
                src_h, src_w = img.shape[:2]
                aspect = src_w / src_h
                target_aspect = canvas_w / canvas_h
                logger.info(f"Source aspect ratio: {aspect:.3f} ({src_w}×{src_h})")
                logger.info(f"Target aspect ratio: {target_aspect:.3f} ({canvas_w}×{canvas_h})")
                
                if aspect > target_aspect:
                    # Width constrained
                    canvas_size = (canvas_w, int(canvas_w / aspect))
                    logger.info(f"Width constrained: {canvas_size}")
                else:
                    # Height constrained
                    canvas_size = (int(canvas_h * aspect), canvas_h)
                    logger.info(f"Height constrained: {canvas_size}")
            else:
                canvas_size = (canvas_w, canvas_h)
                logger.info(f"Aspect ratio preservation disabled, using: {canvas_size}")
            
            # Resize to canvas size
            logger.info(f"Resizing from {img.shape[1]}×{img.shape[0]} to {canvas_size[0]}×{canvas_size[1]}")
            rectified = cv2.resize(img, canvas_size)
            logger.info(f"✓ Resized to {canvas_size} (aspect preserved)")
            return rectified, None
        
        logger.info(f"Quad detected with points: {quad}")
        
        # Calculate aspect ratio from the detected quad
        canvas_w, canvas_h = CONFIG['rectified_canvas']
        logger.info(f"Configured canvas size: {canvas_w}×{canvas_h}")
        
        # Preserve aspect ratio if configured
        if CONFIG.get('preserve_aspect_ratio', True):
            # Calculate the quad's width and height
            # Quad points are in order: top-left, top-right, bottom-right, bottom-left
            width1 = np.linalg.norm(quad[1] - quad[0])
            width2 = np.linalg.norm(quad[2] - quad[3])
            height1 = np.linalg.norm(quad[3] - quad[0])
            height2 = np.linalg.norm(quad[2] - quad[1])
            
            src_w = (width1 + width2) / 2
            src_h = (height1 + height2) / 2
            aspect = src_w / src_h
            target_aspect = canvas_w / canvas_h
            
            logger.info(f"Quad dimensions - W1: {width1:.1f}, W2: {width2:.1f}, H1: {height1:.1f}, H2: {height2:.1f}")
            logger.info(f"Quad average: {src_w:.1f}×{src_h:.1f}")
            logger.info(f"Quad aspect ratio: {aspect:.3f}")
            logger.info(f"Target aspect ratio: {target_aspect:.3f}")
            
            if aspect > target_aspect:
                # Width constrained
                canvas_size = (canvas_w, int(canvas_w / aspect))
                logger.info(f"Width constrained: {canvas_size}")
            else:
                # Height constrained
                canvas_size = (int(canvas_h * aspect), canvas_h)
                logger.info(f"Height constrained: {canvas_size}")
        else:
            canvas_size = (canvas_w, canvas_h)
            logger.info(f"Aspect ratio preservation disabled, using: {canvas_size}")
        
        # Four-point perspective transform
        dst_pts = np.array([
            [0, 0],
            [canvas_size[0] - 1, 0],
            [canvas_size[0] - 1, canvas_size[1] - 1],
            [0, canvas_size[1] - 1]
        ], dtype=np.float32)
        
        logger.info(f"Destination points for perspective transform: {dst_pts}")
        logger.info(f"Applying perspective transform...")
        transform = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_pts)
        rectified = cv2.warpPerspective(img, transform, canvas_size)
        
        logger.info(f"✓ Rectified to {canvas_size} (quad aspect: {aspect:.2f})")
        return rectified, transform
    
    def _find_quad_contour(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Find whiteboard quadrilateral using contours"""
        logger.info("Converting to grayscale and detecting edges...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        logger.info(f"Edge pixels detected: {np.count_nonzero(edges)}")
        
        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        logger.info(f"After dilation: {np.count_nonzero(edges)} edge pixels")
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Found {len(contours)} contours")
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        logger.info(f"Checking top 10 contours by area...")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            logger.info(f"  Contour {i+1}: area={area:.0f} ({area/(img.shape[0]*img.shape[1])*100:.1f}% of image), "
                       f"perimeter={perimeter:.0f}, vertices={len(approx)}")
            
            if len(approx) == 4:
                # Check if it's roughly rectangular (lowered threshold)
                min_area = img.shape[0] * img.shape[1] * 0.1
                if area > min_area:  # At least 10% of image
                    logger.info(f"  ✓ Found quad! Area {area:.0f} > min {min_area:.0f}")
                    # Order points: top-left, top-right, bottom-right, bottom-left
                    pts = approx.reshape(4, 2)
                    ordered = self._order_points(pts)
                    logger.info(f"  Ordered points: {ordered}")
                    return ordered
                else:
                    logger.info(f"  ✗ Quad too small: {area:.0f} < {min_area:.0f}")
        
        logger.info("No suitable quad found")
        return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting from top-left"""
        # Sort by y-coordinate
        sorted_pts = pts[np.argsort(pts[:, 1])]
        
        # Top two points (lowest y)
        top_pts = sorted_pts[:2]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
        
        # Bottom two points (highest y)
        bottom_pts = sorted_pts[2:]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
        
        # Return as: top-left, top-right, bottom-right, bottom-left
        return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
    
    def _find_quad_hough(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Find whiteboard quadrilateral using Hough lines"""
        # Simplified implementation - would need full Hough line intersection logic
        return None
    
    def _extract_strokes_hybrid(self, img: np.ndarray) -> np.ndarray:
        """Extract strokes using classical + optional AI"""
        cfg = CONFIG['stroke_extract']
        
        logger.info("Using classical stroke extraction...")
        # Classical branch
        classical_mask = self._extract_classical(img)
        logger.info(f"Classical mask: {classical_mask.shape}, non-zero: {np.count_nonzero(classical_mask)}")
        
        # AI branch (if enabled)
        if cfg.get('use_u2net', False) and self.u2net_session:
            logger.info("Using U2-Net AI extraction...")
            ai_mask = self._extract_u2net(img)
            logger.info(f"AI mask: {ai_mask.shape}, non-zero: {np.count_nonzero(ai_mask)}")
            # Fuse masks
            fused = cv2.bitwise_or(classical_mask, ai_mask)
            logger.info(f"Fused mask non-zero: {np.count_nonzero(fused)}")
        else:
            logger.info("AI extraction disabled, using classical only")
            fused = classical_mask
        
        # Clean up
        logger.info("Cleaning mask...")
        fused = self._clean_mask(fused)
        
        # Additional refinement: smooth jaggies and remove random pixels
        fused = cv2.medianBlur(fused, 3)
        _, fused = cv2.threshold(fused, 127, 255, cv2.THRESH_BINARY)
        
        logger.info(f"After cleaning: {np.count_nonzero(fused)} non-zero pixels")
        
        return fused
    
    def _extract_classical(self, img: np.ndarray) -> np.ndarray:
        """Classical CV stroke extraction with advanced techniques"""
        cfg = CONFIG['stroke_extract']
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Background subtraction (new technique)
        if CONFIG['illumination'].get('background_subtract', True):
            logger.info("Applying background subtraction...")
            median_blur_size = CONFIG['illumination'].get('median_blur_size', 31)
            bg = cv2.medianBlur(gray, median_blur_size)
            gray = cv2.absdiff(gray, bg)
            logger.info(f"Background subtracted with median blur kernel: {median_blur_size}")
        
        # Step 2: Denoise
        logger.info("Denoising with fastNlMeansDenoising...")
        gray = cv2.fastNlMeansDenoising(gray, h=7)
        gray = cv2.equalizeHist(gray)
        
        # Step 3: Auto-calibrate adaptive threshold (new technique)
        block_size = cfg.get('adaptive_block', 15)
        c_value = cfg.get('adaptive_c', 18)
        
        if cfg.get('auto_calibrate_threshold', True):
            logger.info("Auto-calibrating threshold based on image statistics...")
            mean, std = cv2.meanStdDev(gray)
            mean_val = mean[0][0]
            std_val = std[0][0]
            
            # Adjust adaptive_c based on image variance
            # Higher std = more contrast = can be stricter
            # Lower std = faint marks = need gentler threshold
            c_value = int(np.clip(std_val * 0.5, 8, 20))
            logger.info(f"  Image mean: {mean_val:.1f}, std: {std_val:.1f}")
            logger.info(f"  Auto-calibrated adaptive_c: {c_value} (original: {cfg.get('adaptive_c', 18)})")
        
        # Step 4: Adaptive threshold for dark ink
        logger.info(f"Adaptive threshold: block_size={block_size}, c={c_value}")
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, c_value
        )
        
        # Step 5: Canny edges (for structure reinforcement only)
        canny_low, canny_high = cfg['canny']
        edges = cv2.Canny(gray, canny_low, canny_high)
        logger.info(f"Canny edges: [{canny_low}, {canny_high}]")
        
        # Step 6: Color markers (HSV)
        mask_colors = None
        if cfg.get('color_hsv_masks', True):
            logger.info("Extracting colored markers (HSV)...")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Red (wraps around hue)
            mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
            mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Blue
            mask_blue = cv2.inRange(hsv, (100, 70, 50), (130, 255, 255))
            
            # Green
            mask_green = cv2.inRange(hsv, (40, 70, 50), (80, 255, 255))
            
            # Combine color masks
            mask_colors = cv2.bitwise_or(mask_red, mask_blue)
            mask_colors = cv2.bitwise_or(mask_colors, mask_green)
            
            logger.info(f"Color mask pixels: {np.count_nonzero(mask_colors)}")
        
        # Step 7: Intelligent combination strategy
        if cfg.get('canny_structure_only', True):
            # Use Canny to reinforce structure, not add noise
            # AND edges with adaptive threshold to keep only structured lines
            logger.info("Combining: Adaptive threshold AND Canny (structure only)...")
            structured_ink = cv2.bitwise_and(thresh, edges)
            
            # Combine with original threshold (to not lose too much)
            combined = cv2.bitwise_or(thresh, structured_ink)
            
            # Add color markers if available
            if mask_colors is not None:
                # AND color masks with edges to ensure they're real strokes
                color_structured = cv2.bitwise_and(mask_colors, edges)
                combined = cv2.bitwise_or(combined, color_structured)
                logger.info("Added structured color markers")
        else:
            # Old method: OR everything together
            logger.info("Combining: Adaptive threshold OR Canny OR Colors...")
            if mask_colors is not None:
                mask_colors_edges = cv2.bitwise_and(mask_colors, edges)
                combined = cv2.bitwise_or(thresh, mask_colors_edges)
            else:
                combined = thresh
        
        logger.info(f"Combined mask pixels: {np.count_nonzero(combined)}")
        return combined
    
    def _extract_u2net(self, img: np.ndarray) -> np.ndarray:
        """Extract strokes using U2-Net saliency"""
        # Placeholder - would need actual U2-Net inference
        logger.warning("U2-Net inference not yet implemented")
        return np.zeros(img.shape[:2], dtype=np.uint8)
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean mask with morphology and smart filtering"""
        cfg = CONFIG['stroke_extract']
        
        # Morphological operations (improved values)
        morph_open = cfg.get('morph_open', 4)
        morph_close = cfg.get('morph_close', 5)
        
        if morph_open > 0:
            logger.info(f"Morphological opening: kernel size {morph_open} (removes noise)")
            kernel = np.ones((morph_open, morph_open), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if morph_close > 0:
            logger.info(f"Morphological closing: kernel size {morph_close} (connects gaps)")
            kernel = np.ones((morph_close, morph_close), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small areas with dynamic scaling
        min_area = cfg.get('min_area_px', 250)
        
        # Optional: Scale min_area based on image size
        img_area = mask.shape[0] * mask.shape[1]
        min_area_ratio = min_area / (960 * 540)  # Ratio relative to default size
        scaled_min_area = int(img_area * min_area_ratio)
        
        logger.info(f"Filtering small contours: min_area={min_area} px (scaled: {scaled_min_area})")
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        
        kept_count = 0
        removed_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
                kept_count += 1
            else:
                removed_count += 1
        
        logger.info(f"Contour filtering: kept {kept_count}, removed {removed_count} small regions")
        
        return clean_mask
    
    def _vectorize_mask(self, mask: np.ndarray, img: np.ndarray) -> List[Dict]:
        """Convert mask to vector strokes"""
        cfg = CONFIG['vectorize']
        
        logger.info(f"Vectorizing mask: {mask.shape}")
        
        # Skeletonize if enabled
        if cfg.get('skeletonize', True) and SKIMAGE_AVAILABLE:
            logger.info("Skeletonizing mask...")
            skel = skeletonize(mask > 0)
            mask = (skel * 255).astype(np.uint8)
            logger.info(f"After skeletonization: {np.count_nonzero(mask)} pixels")
        else:
            logger.info("Skeletonization disabled or skimage not available")
        
        # Find contours
        logger.info("Finding contours...")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Found {len(contours)} contours")
        
        strokes = []
        for i, contour in enumerate(contours):
            # Simplify with RDP
            epsilon = cfg.get('rdp_epsilon_px', 1.0)
            approx = cv2.approxPolyDP(contour, epsilon, False)
            
            if len(approx) < 2:
                logger.info(f"  Contour {i+1}: Skipped (too few points: {len(approx)})")
                continue
            
            # Extract color from original image
            points = approx.reshape(-1, 2)
            colorize = cfg.get('colorize_from_source')
            logger.info(f"  Config colorize_from_source: {colorize}")
            
            if colorize:
                color = self._get_stroke_color(points, img)
            else:
                color = '#000000'
            
            logger.info(f"  Contour {i+1}: {len(approx)} points, color={color}, "
                       f"width={cfg.get('stroke_width_px', 2.5)}")
            
            strokes.append({
                'points': points.tolist(),
                'color': color,
                'width': cfg.get('stroke_width_px', 2.5)
            })
        
        logger.info(f"Generated {len(strokes)} strokes")
        return strokes
    
    def _get_stroke_color(self, points: np.ndarray, img: np.ndarray) -> str:
        """Sample dominant color along stroke path"""
        colors = []
        for x, y in points:
            if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
                colors.append(img[int(y), int(x)])
        
        if not colors:
            return '#000000'
        
        # Median color (BGR from OpenCV)
        median_color = np.median(colors, axis=0).astype(int)
        
        # Convert BGR to RGB (no inversion - use actual colors from image)
        r = median_color[2]  # B→R
        g = median_color[1]  # G→G
        b = median_color[0]  # R→B
        
        return f"#{r:02x}{g:02x}{b:02x}"


# Singleton instance
_extractor = None

def get_extractor() -> HybridStrokeExtractor:
    """Get or create singleton extractor"""
    global _extractor
    if _extractor is None:
        _extractor = HybridStrokeExtractor()
    return _extractor
