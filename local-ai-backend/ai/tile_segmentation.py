"""
Tile-Aware Lightweight Segmentation
Uses DeepLabV3-MobileNetV3 for uncertain regions only
Supports both ONNX (.onnx) and PyTorch (.pts) models
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TileSegmentation:
    """Tile-aware semantic segmentation for stroke/smudge/shadow classification"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize tile segmentation model
        
        Args:
            model_path: Path to model (.onnx for ONNX Runtime, .pts for PyTorch)
        """
        self.session = None
        self.enabled = False
        self.model_type = None
        
        if model_path is None:
            # Try PyTorch model first (easier to work with), then ONNX
            base_path = Path(__file__).parent.parent / 'models'
            candidates = [
                base_path / 'whiteboard_seg.pts',  # PyTorch TorchScript
                base_path / 'whiteboard_seg_int8.onnx',  # Quantized ONNX
                base_path / 'whiteboard_seg.onnx',  # Regular ONNX
            ]
            model_path = None
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break
        
        if model_path is None:
            logger.info("No segmentation model found")
            logger.info("Tile segmentation disabled - will use classical CV only")
            return
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.info(f"Segmentation model not found at {model_path}")
            logger.info("Tile segmentation disabled - will use classical CV only")
            return
        
        try:
            # Determine model type and load
            if model_path.suffix == '.pts':
                # PyTorch TorchScript model
                if not TORCH_AVAILABLE:
                    logger.warning("PyTorch not available - cannot load .pts model")
                    return
                
                from pytorch_inference import create_inference_session
                self.session = create_inference_session(str(model_path))
                self.model_type = 'pytorch'
                logger.info(f"Loaded PyTorch model: {model_path}")
                
            elif model_path.suffix == '.onnx':
                # ONNX Runtime model
                if not ONNX_AVAILABLE:
                    logger.warning("ONNX Runtime not available - cannot load .onnx model")
                    return
                
                # Try DirectML first for Intel GPU acceleration, fall back to CPU
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(str(model_path), providers=providers)
                self.model_type = 'onnx'
                
                active_provider = self.session.get_providers()[0]
                logger.info(f"Tile segmentation initialized with {active_provider}")
                logger.info(f"Model: {model_path.name}")
            
            self.enabled = True
            self.input_size = (640, 480)  # Model input size (W, H)
            
        except Exception as e:
            logger.error(f"Failed to initialize tile segmentation: {e}")
            self.enabled = False
    
    def find_uncertain_tiles(self, 
                            classical_mask: np.ndarray, 
                            img: np.ndarray,
                            tile_size: int = 128,
                            overlap: int = 16) -> List[Tuple[int, int, int, int]]:
        """
        Identify uncertain regions that need ML verification
        
        Args:
            classical_mask: Binary mask from classical CV pipeline
            img: Original image for intensity analysis
            tile_size: Size of tiles (64, 128, or 256)
            overlap: Overlap between tiles to avoid edge artifacts
        
        Returns:
            List of (x, y, w, h) tile coordinates
        """
        H, W = classical_mask.shape[:2]
        tiles = []
        
        # Convert to grayscale for uncertainty analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate stride with overlap
        stride = tile_size - overlap
        
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                tile_mask = classical_mask[y:y+tile_size, x:x+tile_size]
                tile_gray = gray[y:y+tile_size, x:x+tile_size]
                
                # Check if tile is uncertain
                if self._is_uncertain(tile_mask, tile_gray):
                    tiles.append((x, y, tile_size, tile_size))
        
        # Handle edge tiles
        if W % stride != 0:
            for y in range(0, H - tile_size + 1, stride):
                x = W - tile_size
                tile_mask = classical_mask[y:y+tile_size, x:x+tile_size]
                tile_gray = gray[y:y+tile_size, x:x+tile_size]
                if self._is_uncertain(tile_mask, tile_gray):
                    tiles.append((x, y, tile_size, tile_size))
        
        if H % stride != 0:
            for x in range(0, W - tile_size + 1, stride):
                y = H - tile_size
                tile_mask = classical_mask[y:y+tile_size, x:x+tile_size]
                tile_gray = gray[y:y+tile_size, x:x+tile_size]
                if self._is_uncertain(tile_mask, tile_gray):
                    tiles.append((x, y, tile_size, tile_size))
        
        logger.info(f"Found {len(tiles)} uncertain tiles (size={tile_size}x{tile_size})")
        return tiles
    
    def _is_uncertain(self, tile_mask: np.ndarray, tile_gray: np.ndarray) -> bool:
        """
        Determine if a tile contains uncertain regions
        
        Uncertain characteristics:
        - Medium pixel density (10-60% filled)
        - Medium intensity variance (shadows/smudges have gradient)
        - Moderate edge strength (not crisp strokes, not empty)
        """
        density = np.count_nonzero(tile_mask) / tile_mask.size
        
        # Skip empty or very full tiles
        if density < 0.1 or density > 0.6:
            return False
        
        # Check intensity variance (shadows/smudges create gradients)
        std = np.std(tile_gray)
        
        # Skip very uniform regions (likely clean strokes or empty space)
        if std < 15:
            return False
        
        # Check for medium-strength edges (smudges have softer edges)
        edges = cv2.Canny(tile_gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Uncertain if moderate edge density
        if 0.02 < edge_density < 0.15:
            return True
        
        # Uncertain if high variance + medium density (likely smudge/shadow)
        if std > 25 and 0.15 < density < 0.5:
            return True
        
        return False
    
    def infer_tiles(self, img: np.ndarray, tiles: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Run segmentation inference on uncertain tiles only
        
        Args:
            img: Original image (BGR)
            tiles: List of (x, y, w, h) tile coordinates
        
        Returns:
            Full-resolution stroke mask (0=background, 255=stroke)
        """
        if not self.enabled or len(tiles) == 0:
            logger.warning("Tile segmentation not enabled or no tiles to process")
            return np.zeros(img.shape[:2], dtype=np.uint8)
        
        H, W = img.shape[:2]
        mask_full = np.zeros((H, W), dtype=np.uint8)
        
        logger.info(f"Processing {len(tiles)} tiles with DeepLabV3-MobileNetV3 INT8...")
        
        for idx, (x, y, w, h) in enumerate(tiles):
            # Extract and preprocess tile
            tile_img = img[y:y+h, x:x+w]
            
            # Resize to model input size
            tile_resized = cv2.resize(tile_img, self.input_size)
            
            # Normalize to [0, 1] and convert to CHW format
            tile_normalized = tile_resized.astype(np.float32) / 255.0
            tile_chw = tile_normalized.transpose(2, 0, 1)
            tile_batch = tile_chw[np.newaxis, ...]  # Add batch dimension
            
            # Run inference
            try:
                if self.model_type == 'pytorch':
                    # PyTorch inference session (uses pytorch_inference.py adapter)
                    input_name = self.session.get_inputs()[0].name
                    outputs = self.session.run(None, {input_name: tile_batch})
                    pred = outputs[0][0]  # Shape: [num_classes, H, W]
                else:
                    # ONNX Runtime inference
                    outputs = self.session.run(None, {'input': tile_batch})
                    pred = outputs[0][0]  # Shape: [num_classes, H, W]
                
                # Extract stroke class (class 1)
                # Classes: 0=background, 1=stroke, 2=smudge, 3=shadow
                stroke_mask = (np.argmax(pred, axis=0) == 1).astype(np.uint8) * 255
                
                # Resize back to tile size
                stroke_mask_resized = cv2.resize(stroke_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Merge into full mask (take maximum to preserve strokes)
                mask_full[y:y+h, x:x+w] = np.maximum(
                    mask_full[y:y+h, x:x+w], 
                    stroke_mask_resized
                )
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(tiles)} tiles")
                
            except Exception as e:
                logger.error(f"Inference failed for tile {idx}: {e}")
                continue
        
        logger.info(f"Tile segmentation complete: {np.count_nonzero(mask_full)} stroke pixels")
        return mask_full
    
    def refine_mask(self, 
                    classical_mask: np.ndarray, 
                    img: np.ndarray,
                    tile_size: int = 128) -> np.ndarray:
        """
        Refine classical CV mask using tile-based segmentation on uncertain regions
        
        Args:
            classical_mask: Initial mask from classical CV pipeline
            img: Original image
            tile_size: Size of processing tiles (64, 128, or 256)
        
        Returns:
            Refined stroke mask with ML-verified regions
        """
        if not self.enabled:
            logger.info("Tile segmentation disabled - returning classical mask")
            return classical_mask
        
        # Find uncertain regions
        uncertain_tiles = self.find_uncertain_tiles(classical_mask, img, tile_size=tile_size)
        
        if len(uncertain_tiles) == 0:
            logger.info("No uncertain regions found - classical mask is confident")
            return classical_mask
        
        # Run ML on uncertain tiles only
        ml_mask = self.infer_tiles(img, uncertain_tiles)
        
        # Merge: Keep confident classical regions, replace uncertain with ML
        refined_mask = classical_mask.copy()
        
        # Create uncertainty map
        uncertainty_map = np.zeros(classical_mask.shape, dtype=np.uint8)
        for (x, y, w, h) in uncertain_tiles:
            uncertainty_map[y:y+h, x:x+w] = 255
        
        # Replace uncertain regions with ML predictions
        refined_mask = np.where(uncertainty_map > 0, ml_mask, classical_mask)
        
        # Log statistics
        classical_pixels = np.count_nonzero(classical_mask)
        refined_pixels = np.count_nonzero(refined_mask)
        change_pct = abs(refined_pixels - classical_pixels) / max(classical_pixels, 1) * 100
        
        logger.info(f"Mask refinement: {classical_pixels} → {refined_pixels} pixels ({change_pct:.1f}% change)")
        
        return refined_mask.astype(np.uint8)
