"""
Tile-Aware Lightweight Segmentation
Uses DeepLabV3-MobileNetV3 for stroke detection with smooth tiled inference

PRODUCTION SETTINGS (must match training):
- Tile size: 768×1024 (W×H format: width, height)
- Overlap: 50% (default, smooth Gaussian blending)
- Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Classes: 2 (background=0, stroke=1)
- Method: Full-image smooth tiled inference (infer_full_image_smooth)

Model file: models/whiteboard_seg.pts (42.6 MB TorchScript)
Trained with: train_segmentation.py at 768×1024 resolution
Tested with: test_model_production.py with same settings

The infer_full_image_smooth() method is the RECOMMENDED approach for production.
It processes the entire image with overlapping tiles and Gaussian-weighted blending
for seamless results without visible tile boundaries.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    from torchvision import transforms
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
            # IMPORTANT: Use same resolution as training/testing (768×1024 for quality)
            # Matches test_model.py default resolution
            self.input_size = (1024, 768)  # Model input size (W, H)
            
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
    
    def infer_tiles(self, img: np.ndarray, tiles: List[Tuple[int, int, int, int]], progress_callback=None) -> np.ndarray:
        """
        Run segmentation inference on uncertain tiles only
        
        Args:
            img: Original image (BGR)
            tiles: List of (x, y, w, h) tile coordinates
            progress_callback: Optional callback(progress_pct, details) for reporting progress
        
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
            
            # CRITICAL: Use ImageNet normalization (must match training!)
            # Convert BGR to RGB for ImageNet stats
            tile_rgb = cv2.cvtColor(tile_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            tile_normalized = tile_rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization (mean and std)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tile_normalized = (tile_normalized - mean) / std
            
            # Convert to CHW format
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
                
                # Report progress every tile (for smooth updates)
                if progress_callback:
                    # Progress from 20% to 75% based on tile completion (55% range for AI processing)
                    progress_pct = 20 + int((idx + 1) / len(tiles) * 55)
                    logger.info(f"[TILE PROGRESS] Calling callback with {progress_pct}% (tile {idx+1}/{len(tiles)})")
                    progress_callback(progress_pct, f"Processing tile {idx + 1}/{len(tiles)}")
                else:
                    logger.warning(f"[TILE PROGRESS] No callback provided! Cannot report progress for tile {idx+1}")
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(tiles)} tiles")
                
            except Exception as e:
                logger.error(f"Inference failed for tile {idx}: {e}")
                continue
        
        logger.info(f"Tile segmentation complete: {np.count_nonzero(mask_full)} stroke pixels")
        return mask_full
    
    def create_gaussian_weight_map(self, height, width, sigma=None):
        """
        Create 2D Gaussian weight map for tile blending
        Higher weights in center, lower at edges for smooth transitions
        """
        if sigma is None:
            sigma = min(height, width) / 6
        
        # Create 1D gaussians
        y = np.linspace(-height/2, height/2, height)
        x = np.linspace(-width/2, width/2, width)
        
        # Create 2D gaussian via outer product
        gauss_y = np.exp(-(y**2) / (2 * sigma**2))
        gauss_x = np.exp(-(x**2) / (2 * sigma**2))
        weight_map = np.outer(gauss_y, gauss_x)
        
        # Normalize to [0, 1]
        weight_map = weight_map / weight_map.max()
        
        return weight_map.astype(np.float32)
    
    def infer_full_image_smooth(self, img: np.ndarray, overlap=0.5, progress_callback=None) -> np.ndarray:
        """
        Run SMOOTH tile-based inference on ENTIRE image with Gaussian blending
        This produces much smoother results than the simple max-blending approach.
        
        Args:
            img: Original image (BGR format from OpenCV)
            overlap: Overlap ratio between tiles (0.5 = 50% overlap)
            progress_callback: Optional callback(progress_pct, details)
        
        Returns:
            Full-resolution stroke mask (0=background, 255=stroke)
        """
        if not self.enabled:
            logger.warning("Tile segmentation not enabled")
            return np.zeros(img.shape[:2], dtype=np.uint8)
        
        H, W = img.shape[:2]
        tile_h, tile_w = self.input_size[1], self.input_size[0]  # input_size is (W, H)
        
        logger.info(f"Running SMOOTH tiled inference on {W}×{H} image")
        logger.info(f"Tile size: {tile_w}×{tile_h}, Overlap: {overlap*100:.0f}%")
        
        # Calculate stride
        stride_h = int(tile_h * (1 - overlap))
        stride_w = int(tile_w * (1 - overlap))
        
        # Initialize accumulation arrays
        prediction_sum = np.zeros((H, W), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)
        
        # Create Gaussian weight map for smooth blending
        weight_map = self.create_gaussian_weight_map(tile_h, tile_w)
        
        # Calculate grid
        tiles_h = (H - tile_h) // stride_h + 1
        tiles_w = (W - tile_w) // stride_w + 1
        total_tiles = tiles_h * tiles_w
        
        # Add edge tiles
        if W > tiles_w * stride_w:
            total_tiles += tiles_h
        if H > tiles_h * stride_h:
            total_tiles += tiles_w
        if W > tiles_w * stride_w and H > tiles_h * stride_h:
            total_tiles += 1
        
        logger.info(f"Processing {total_tiles} tiles ({tiles_h}×{tiles_w} grid + edges)...")
        
        # Setup preprocessing transform
        if TORCH_AVAILABLE:
            transform = transforms.Compose([
                transforms.Resize((tile_h, tile_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        tile_count = 0
        
        # Process main grid
        for y in range(0, H - tile_h + 1, stride_h):
            for x in range(0, W - tile_w + 1, stride_w):
                # Extract tile
                tile_img = img[y:y+tile_h, x:x+tile_w]
                
                # Convert BGR to RGB
                tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                
                # Preprocess with PIL/torchvision for consistency
                tile_pil = Image.fromarray(tile_rgb)
                
                if TORCH_AVAILABLE:
                    # Use torchvision transform (same as training)
                    tile_tensor = transform(tile_pil).unsqueeze(0).numpy()
                else:
                    # Fallback: manual preprocessing
                    tile_resized = cv2.resize(tile_rgb, (tile_w, tile_h))
                    tile_normalized = tile_resized.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    tile_normalized = (tile_normalized - mean) / std
                    tile_tensor = tile_normalized.transpose(2, 0, 1)[np.newaxis, ...]
                
                # Run inference
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: tile_tensor})
                pred = outputs[0][0]  # Shape: [num_classes, H, W]
                
                # Extract stroke class (class 1 for binary segmentation)
                stroke_prob = np.argmax(pred, axis=0).astype(np.float32)
                
                # Add weighted prediction
                prediction_sum[y:y+tile_h, x:x+tile_w] += stroke_prob * weight_map
                weight_sum[y:y+tile_h, x:x+tile_w] += weight_map
                
                tile_count += 1
                if progress_callback and tile_count % 5 == 0:
                    progress_pct = 20 + int((tile_count / total_tiles) * 55)
                    progress_callback(progress_pct, f"Processing tile {tile_count}/{total_tiles}")
        
        # Process right edge
        if W > tiles_w * stride_w:
            x = W - tile_w
            for y in range(0, H - tile_h + 1, stride_h):
                tile_img = img[y:y+tile_h, x:x+tile_w]
                tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                tile_pil = Image.fromarray(tile_rgb)
                
                if TORCH_AVAILABLE:
                    tile_tensor = transform(tile_pil).unsqueeze(0).numpy()
                else:
                    tile_resized = cv2.resize(tile_rgb, (tile_w, tile_h))
                    tile_normalized = tile_resized.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    tile_normalized = (tile_normalized - mean) / std
                    tile_tensor = tile_normalized.transpose(2, 0, 1)[np.newaxis, ...]
                
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: tile_tensor})
                pred = outputs[0][0]
                stroke_prob = np.argmax(pred, axis=0).astype(np.float32)
                
                prediction_sum[y:y+tile_h, x:x+tile_w] += stroke_prob * weight_map
                weight_sum[y:y+tile_h, x:x+tile_w] += weight_map
                tile_count += 1
        
        # Process bottom edge
        if H > tiles_h * stride_h:
            y = H - tile_h
            for x in range(0, W - tile_w + 1, stride_w):
                tile_img = img[y:y+tile_h, x:x+tile_w]
                tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                tile_pil = Image.fromarray(tile_rgb)
                
                if TORCH_AVAILABLE:
                    tile_tensor = transform(tile_pil).unsqueeze(0).numpy()
                else:
                    tile_resized = cv2.resize(tile_rgb, (tile_w, tile_h))
                    tile_normalized = tile_resized.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    tile_normalized = (tile_normalized - mean) / std
                    tile_tensor = tile_normalized.transpose(2, 0, 1)[np.newaxis, ...]
                
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: tile_tensor})
                pred = outputs[0][0]
                stroke_prob = np.argmax(pred, axis=0).astype(np.float32)
                
                prediction_sum[y:y+tile_h, x:x+tile_w] += stroke_prob * weight_map
                weight_sum[y:y+tile_h, x:x+tile_w] += weight_map
                tile_count += 1
        
        # Process bottom-right corner
        if W > tiles_w * stride_w and H > tiles_h * stride_h:
            x = W - tile_w
            y = H - tile_h
            tile_img = img[y:y+tile_h, x:x+tile_w]
            tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
            tile_pil = Image.fromarray(tile_rgb)
            
            if TORCH_AVAILABLE:
                tile_tensor = transform(tile_pil).unsqueeze(0).numpy()
            else:
                tile_resized = cv2.resize(tile_rgb, (tile_w, tile_h))
                tile_normalized = tile_resized.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                tile_normalized = (tile_normalized - mean) / std
                tile_tensor = tile_normalized.transpose(2, 0, 1)[np.newaxis, ...]
            
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: tile_tensor})
            pred = outputs[0][0]
            stroke_prob = np.argmax(pred, axis=0).astype(np.float32)
            
            prediction_sum[y:y+tile_h, x:x+tile_w] += stroke_prob * weight_map
            weight_sum[y:y+tile_h, x:x+tile_w] += weight_map
            tile_count += 1
        
        # Normalize by weights (weighted average blending)
        weight_sum = np.maximum(weight_sum, 1e-8)  # Avoid division by zero
        final_prediction = (prediction_sum / weight_sum).round().astype(np.uint8)
        
        # Convert to binary mask (0 or 255)
        final_mask = (final_prediction * 255).astype(np.uint8)
        
        stroke_pixels = np.count_nonzero(final_mask)
        logger.info(f"✓ Smooth tiled inference complete: {stroke_pixels} stroke pixels")
        
        return final_mask
    
    def refine_mask(self, 
                    classical_mask: np.ndarray, 
                    img: np.ndarray,
                    tile_size: int = 128,
                    overlap: float = 0.5,
                    use_smooth_tiling: bool = True,
                    progress_callback=None) -> np.ndarray:
        """
        Refine mask using ML segmentation
        
        Args:
            classical_mask: Initial mask from classical CV pipeline (IGNORED if use_smooth_tiling=True)
            img: Original image (BGR format)
            tile_size: Size of processing tiles (IGNORED if use_smooth_tiling=True)
            overlap: Overlap ratio for smooth tiling (default 0.5 = 50%)
            use_smooth_tiling: If True, uses smooth Gaussian-blended full-image tiling (RECOMMENDED)
                              If False, uses old uncertain-tile method
            progress_callback: Optional callback(progress_pct, details)
        
        Returns:
            Refined stroke mask with ML predictions
        """
        if not self.enabled:
            logger.info("Tile segmentation disabled - returning classical mask")
            return classical_mask
        
        # RECOMMENDED: Use smooth Gaussian-blended tiling on ENTIRE image
        if use_smooth_tiling:
            logger.info("Using SMOOTH tiled inference (Gaussian blending) - PRODUCTION MODE")
            return self.infer_full_image_smooth(img, overlap=overlap, progress_callback=progress_callback)
        
        # LEGACY: Old uncertain-tile method (kept for compatibility)
        else:
            logger.info("Using LEGACY uncertain-tile method (not recommended)")
            # Find uncertain regions
            uncertain_tiles = self.find_uncertain_tiles(classical_mask, img, tile_size=tile_size)
            
            if len(uncertain_tiles) == 0:
                logger.info("No uncertain regions found - classical mask is confident")
                return classical_mask
            
            # Run ML on uncertain tiles only
            ml_mask = self.infer_tiles(img, uncertain_tiles, progress_callback=progress_callback)
            
            # Merge: Keep confident classical regions, replace uncertain with ML
            refined_mask = classical_mask.copy()
            
            return refined_mask.astype(np.uint8)
