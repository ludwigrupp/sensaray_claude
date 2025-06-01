import numba
import numpy as np
import cv2
import os
import time
import shutil
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import logging
from collections import deque

logger = logging.getLogger(__name__)

class HSVProcessingError(Exception):
    """Raised when HSV processing encounters an error."""
    pass

class HSVCalculator:
    """HSV-based contamination detection with optimized processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.work_dir = Path(config["work_dir"])
        self.hsv_work_dir = self.work_dir / "HSV_masks"
        self.hsv_work_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for metallic color definitions
        self._metallic_colors = self._initialize_metallic_colors()

    def _initialize_metallic_colors(self) -> List[Tuple]:
        """Initialize metallic color definitions."""
        return [
            (0, 180, 0, 25, 204, 255, "Silver"),
            (0, 180, 0, 50, 76, 204, "Gray Metal"),
            (20, 28, 128, 230, 153, 255, "Gold"),
            (10, 18, 128, 230, 128, 230, "Copper"),
            (100, 120, 51, 128, 128, 230, "Steel Blue"),
            (29, 35, 128, 204, 102, 204, "Bronze")
        ]

    def load_hsv_mask(self) -> Tuple[np.ndarray, float]:
        """Load HSV mask with proper error handling and validation."""
        hsv_file_path = self.hsv_work_dir / f"HSV_mask_{self.config['line_name']}.npy"
        
        try:
            if hsv_file_path.exists():
                hsv_mask = np.load(hsv_file_path).astype(np.uint8)
                
                # Validate mask dimensions
                if hsv_mask.shape != (180, 256, 256):
                    raise HSVProcessingError(f"Invalid HSV mask dimensions: {hsv_mask.shape}")
                
                pixel_count = np.sum(hsv_mask > 0)
                total_pixels = 180 * 256 * 256
                percentage = round((pixel_count / total_pixels) * 100, 2)
                
                logger.info(f"Loaded HSV mask: {hsv_file_path}")
                return hsv_mask, percentage
                
            else:
                if self.config.get("build_hsv_mask", False):
                    hsv_mask = np.zeros((180, 256, 256), dtype=np.uint8)
                    logger.warning(f"HSV mask not found at {hsv_file_path}, created empty mask")
                    waiter = input('.............waiting here as HSV_mask was not found')
                    return hsv_mask, 0.0
                else:
                    # Create a basic mask for trial mode
                    hsv_mask = np.zeros((180, 256, 256), dtype=np.uint8)
                    # Fill with some basic "good" values for yellow/normal colors
                    hsv_mask[10:40, 50:255, 80:255] = 1  # Yellow range
                    logger.warning(f"Created basic HSV mask for trial/demo mode as {hsv_file_path} not found! ")
                    waiter = input('.............waiting here as HSV_mask was not found') 
                    return hsv_mask, 5.0
                    
        except Exception as e:
            logger.error(f"Error loading HSV mask: {e}")
            # Return basic mask instead of failing
            hsv_mask = np.zeros((180, 256, 256), dtype=np.uint8)
            hsv_mask[10:40, 50:255, 80:255] = 1  # Basic yellow range
            return hsv_mask, 1.0

    def save_hsv_mask(self, hsv_mask: np.ndarray, percentage: float) -> None:
        """Save HSV mask with backup and validation."""
        try:
            # Validate input
            if hsv_mask.shape != (180, 256, 256):
                raise ValueError(f"Invalid HSV mask shape: {hsv_mask.shape}")
            
            hsv_file_path = self.hsv_work_dir / f"HSV_mask_{self.config['line_name']}.npy"
            
            # Save main mask
            np.save(hsv_file_path, hsv_mask)
            
            # Save timestamped version
            timestamp = datetime.now().strftime('D_%Y-%m-%d_T_%H_%M_%S')
            timestamped_path = self.hsv_work_dir / f"HSV_mask_{self.config['line_name']}_{percentage}_{timestamp}.npy"
            np.save(timestamped_path, hsv_mask)
            
            logger.info(f"HSV mask saved: {percentage}% pixels active")
            
        except Exception as e:
            logger.error(f"Error saving HSV mask: {e}")
            raise HSVProcessingError(f"Failed to save HSV mask: {e}")

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _check_contamination_vectorized(
        hsv_frame: np.ndarray, 
        hsv_mask: np.ndarray
    ) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """Optimized contamination check using Numba compilation."""
        height, width = hsv_frame.shape[:2]
        no_match_counter = 0
        max_detections = 10000  # Limit to prevent memory issues
        
        # Pre-allocate arrays
        xy_positions = []
        hsv_values = []
        
        for row in numba.prange(height):
            if no_match_counter >= max_detections:
                break
            for col in range(width):
                if no_match_counter >= max_detections:
                    break
                    
                h = hsv_frame[row, col, 0]
                s = hsv_frame[row, col, 1]
                v = hsv_frame[row, col, 2]
                
                if hsv_mask[h, s, v] == 0:
                    xy_positions.append((col, row))
                    hsv_values.append((h, s, v))
                    no_match_counter += 1
        
        return no_match_counter, xy_positions, hsv_values

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _build_hsv_mask_vectorized(
        hsv_frame: np.ndarray, 
        hsv_mask: np.ndarray, 
        s_min: int, 
        v_min: int
    ) -> Tuple[np.ndarray, int]:
        """Build HSV mask using vectorized operations."""
        height, width = hsv_frame.shape[:2]
        new_pixels = 0
        
        for row in numba.prange(height):
            for col in range(width):
                h = hsv_frame[row, col, 0]
                s = hsv_frame[row, col, 1]
                v = hsv_frame[row, col, 2]
                
                # Apply S and V thresholds
                if (s > s_min and v > v_min) or (s > s_min and v < 30):
                    if hsv_mask[h, s, v] == 0:
                        hsv_mask[h, s, v] = 1
                        new_pixels += 1
        
        return hsv_mask, new_pixels

    def check_frame_for_contamination(
        self, 
        bgr_frame: np.ndarray, 
        hsv_mask: np.ndarray
    ) -> Tuple[int, np.ndarray, List[Tuple[int, int, int]]]:
        """Check frame for contamination with optimized processing."""
        try:
            if bgr_frame is None or bgr_frame.size == 0:
                raise ValueError("BGR frame is empty or None")
            
            # Convert to HSV
            hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
            
            # Use optimized contamination check
            no_match_count, xy_positions, hsv_values = self._check_contamination_vectorized(
                hsv_frame, hsv_mask
            )
            
            # Create visualization mask
            contamination_mask = np.zeros_like(bgr_frame)
            mask_radius = self.config.get("mask_radius", 200)
            
            for position in xy_positions[:1000]:  # Limit visualizations
                cv2.circle(
                    contamination_mask, 
                    position, 
                    mask_radius, 
                    (255, 255, 255), 
                    -1
                )
                cv2.rectangle(
                    contamination_mask,
                    (position[0] - 10, position[1] - 10),
                    (position[0] + 10, position[1] + 10),
                    (0, 0, 0),
                    1
                )
            
            return no_match_count, contamination_mask, hsv_values
            
        except Exception as e:
            logger.error(f"Error in contamination check: {e}")
            # Return safe defaults
            return 0, np.zeros_like(bgr_frame), []

    def build_hsv_mask_from_frame(
        self, 
        bgr_frame: np.ndarray, 
        hsv_mask: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Build HSV mask from frame with validation."""
        try:
            if bgr_frame is None or bgr_frame.size == 0:
                raise ValueError("BGR frame is empty or None")
            
            hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
            
            s_min = self.config.get("s_min", 30)
            v_min = self.config.get("v_min", 70)
            
            updated_mask, new_pixels = self._build_hsv_mask_vectorized(
                hsv_frame, hsv_mask, s_min, v_min
            )
            
            return updated_mask, new_pixels
            
        except Exception as e:
            logger.error(f"Error building HSV mask: {e}")
            return hsv_mask, 0

    def remove_blue_values(
        self, 
        hsv_mask: np.ndarray,
        h_min: int = 100, 
        h_max: int = 135,
        s_min: int = 156, 
        s_max: int = 256, 
        v_min: int = 156, 
        v_max: int = 256
    ) -> np.ndarray:
        """Remove blue HSV values from mask for white belt processing."""
        try:
            # Validate input
            if hsv_mask.shape != (180, 256, 256):
                raise ValueError(f"Invalid HSV mask shape: {hsv_mask.shape}")
            
            # Remove blue values
            hsv_mask[h_min:h_max, s_min:s_max, v_min:v_max] = 0
            logger.info(f"Removed blue values from HSV mask: H[{h_min}:{h_max}], S[{s_min}:{s_max}], V[{v_min}:{v_max}]")
            
            return hsv_mask
            
        except Exception as e:
            logger.error(f"Error removing blue values: {e}")
            return hsv_mask

    @staticmethod
    @numba.jit(nopython=True)
    def _identify_metallic_color(h: int, s: int, v: int) -> int:
        """Identify metallic color using compiled function."""
        # Metallic color definitions (encoded as tuples)
        metallic_ranges = [
            (0, 180, 0, 25, 204, 255),    # Silver
            (0, 180, 0, 50, 76, 204),     # Gray Metal  
            (20, 28, 128, 230, 153, 255), # Gold
            (10, 18, 128, 230, 128, 230), # Copper
            (100, 120, 51, 128, 128, 230), # Steel Blue
            (29, 35, 128, 204, 102, 204)   # Bronze
        ]
        
        for i, (h_min, h_max, s_min, s_max, v_min, v_max) in enumerate(metallic_ranges):
            if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
                return i
        return -1  # No metallic color

    def check_metallic_colors(self, bgr_frame: np.ndarray) -> Tuple[int, List[Tuple[int, int]], List[str]]:
        """Check frame for metallic colors with optimized processing."""
        try:
            hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
            height, width = hsv_frame.shape[:2]
            
            metallic_positions = []
            metallic_types = []
            metallic_count = 0
            max_detections = 200
            
            color_names = ["Silver", "Gray Metal", "Gold", "Copper", "Steel Blue", "Bronze"]
            
            for row in range(height):
                if metallic_count >= max_detections:
                    break
                for col in range(width):
                    if metallic_count >= max_detections:
                        break
                        
                    h, s, v = hsv_frame[row, col]
                    color_index = self._identify_metallic_color(h, s, v)
                    
                    if color_index >= 0:
                        metallic_positions.append((col, row))
                        metallic_types.append(color_names[color_index])
                        metallic_count += 1
            
            return metallic_count, metallic_positions, metallic_types
            
        except Exception as e:
            logger.error(f"Error checking metallic colors: {e}")
            return 0, [], []

    def apply_morphological_fill(
        self, 
        hsv_mask_slice: np.ndarray, 
        radius: int = 10
    ) -> np.ndarray:
        """Apply morphological operations to fill HSV mask regions."""
        try:
            # Create disk kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            
            # Apply closing operation to fill gaps
            filled_mask = cv2.morphologyEx(hsv_mask_slice, cv2.MORPH_CLOSE, kernel)
            
            return filled_mask.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error in morphological fill: {e}")
            return hsv_mask_slice

    def fill_area_mask(self, hsv_mask: np.ndarray, distance: int = 1) -> np.ndarray:
        """Fill HSV mask areas within specified distance."""
        try:
            if distance <= 0:
                return hsv_mask
                
            # Create a larger kernel for dilation
            kernel_size = 2 * distance + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Process each H slice
            for h in range(hsv_mask.shape[0]):
                hsv_mask[h] = cv2.dilate(hsv_mask[h], kernel, iterations=1)
            
            return hsv_mask
            
        except Exception as e:
            logger.error(f"Error filling area mask: {e}")
            return hsv_mask

    def rolling_ball_fill(
        self, 
        slice_2d_orig: np.ndarray, 
        seed: Tuple[int, int] = (255, 255), 
        radius: int = 10
    ) -> np.ndarray:
        """Apply rolling ball flood fill algorithm to HSV mask slice."""
        try:
            H, W = slice_2d_orig.shape

            # 1. Mirror-pad the original matrix
            padded = np.pad(slice_2d_orig, pad_width=radius, mode='edge')

            # 2. Adjust seed to padded coordinates
            seed_padded = (seed[0] + radius, seed[1] + radius)

            # 3. Prepare result and visited arrays
            visited = np.zeros_like(padded, dtype=bool)
            result = np.ones_like(padded, dtype=np.uint8)

            # 4. Build disk offsets
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            disk_coords = np.argwhere(disk == 1) - radius

            # 5. Start flood fill
            queue = deque()
            queue.append(seed_padded)
            visited[seed_padded] = True
            result[seed_padded] = 0

            while queue:
                s, v = queue.popleft()

                for ds in range(-radius, radius + 1):
                    for dv in range(-radius, radius + 1):
                        ns, nv = s + ds, v + dv

                        # Skip if the disk would go outside bounds
                        if ns - radius < 0 or ns + radius >= padded.shape[0] or nv - radius < 0 or nv + radius >= padded.shape[1]:
                            continue

                        if visited[ns, nv]:
                            continue

                        all_inside = True
                        for offset in disk_coords:
                            ys, yv = ns + offset[0], nv + offset[1]
                            if padded[ys, yv] == 1:
                                all_inside = False
                                break

                        if all_inside:
                            visited[ns, nv] = True
                            result[ns, nv] = 0
                            queue.append((ns, nv))

            # 6. Crop result back to original shape
            return result[radius:H+radius, radius:W+radius]
            
        except Exception as e:
            logger.error(f"Error in rolling ball fill: {e}")
            return slice_2d_orig

    def write_hsv_mask_to_dir(
        self, 
        hsv_file_name: str, 
        hsv_mask: np.ndarray, 
        hsv_file_name_archive: str, 
        old_hsv_mask: np.ndarray
    ) -> None:
        """Write HSV mask to directory with archival."""
        try:
            # Check disk space (minimum 1GB)
            free_space_gb = shutil.disk_usage(self.work_dir).free / (1024**3)
            
            if free_space_gb > 1.0:
                np.save(hsv_file_name, hsv_mask)
                logger.info(f"HSV mask saved to {hsv_file_name}")
            else:
                logger.warning("Insufficient disk space for HSV mask saving")
                
            if free_space_gb > 2.0:  # More space needed for archive
                timestamp = datetime.now()
                timestamp_string = timestamp.strftime('D_%Y-%m-%d_T_%H_%M_%S_%f')[:-3]
                archive_filename = f"{hsv_file_name_archive}_{timestamp_string}.npy"
                np.save(archive_filename, old_hsv_mask)
                logger.info(f"HSV mask archived to {archive_filename}")
            else:
                logger.warning("Insufficient disk space for HSV mask archiving")
                
        except Exception as e:
            logger.error(f"Error writing HSV mask to directory: {e}")


# Legacy compatibility class for existing code
class HSV_calc:
    """Legacy compatibility class for existing code."""
    
    @staticmethod
    def load_HSV_mask(config, mylogs):
        """Legacy compatibility method."""
        calculator = HSVCalculator(config)
        return calculator.load_hsv_mask()

    @staticmethod
    def write_HSV_mask_to_dir(hsv_file_name, hsv_mask, hsv_file_name_archive, old_hsv_mask):
        """Legacy compatibility method."""
        # This would need a config object to work properly
        logger.warning("Using legacy write_HSV_mask_to_dir - migrate to HSVCalculator")
        
    @staticmethod
    def save_HSV_mask(hsv_mask, perc_of_pixels, config, mylogs):
        """Legacy compatibility method."""
        calculator = HSVCalculator(config)
        calculator.save_hsv_mask(hsv_mask, perc_of_pixels)

    @staticmethod
    @numba.jit(nopython=True)
    def check_pic_for_contanimation_raw(hsv_frame, hsv_mask):
        """Legacy compatibility method."""
        return HSVCalculator._check_contamination_vectorized(hsv_frame, hsv_mask)

    @staticmethod
    @numba.jit(nopython=True)
    def build_HSV_mask_from_HSV_frame(hsv_frame, hsv_mask, s_min, v_min):
        """Legacy compatibility method."""
        return HSVCalculator._build_hsv_mask_vectorized(hsv_frame, hsv_mask, s_min, v_min)

    @staticmethod
    def check_pic_for_contanimation(counter, bgr_frame, hsv_mask):
        """Legacy compatibility method."""
        logger.warning("Using legacy check_pic_for_contanimation - migrate to HSVCalculator")
        config = {'mask_radius': 200}  # Default config
        calculator = HSVCalculator(config)
        no_match_count, contamination_mask, hsv_values = calculator.check_frame_for_contamination(
            bgr_frame, hsv_mask
        )
        return no_match_count, [], hsv_values  # Note: xy_mask format changed

    @staticmethod
    def delete_blue_HSV_values(hsv_mask, h_min=100, h_max=135, s_min=156, s_max=256, v_min=156, v_max=256):
        """Legacy compatibility method."""
        config = {'line_name': 'legacy'}
        calculator = HSVCalculator(config)
        return calculator.remove_blue_values(hsv_mask, h_min, h_max, s_min, s_max, v_min, v_max)

    @staticmethod
    def rolling_ball_fill(slice_2d_orig, seed=(255, 255), radius=10):
        """Legacy compatibility method."""
        config = {'line_name': 'legacy'}
        calculator = HSVCalculator(config)
        return calculator.rolling_ball_fill(slice_2d_orig, seed, radius)

    @staticmethod
    @numba.jit(nopython=True)
    def identify_metallic_HSV_color(h, s, v):
        """Legacy compatibility method."""
        color_index = HSVCalculator._identify_metallic_color(h, s, v)
        color_names = ["Silver", "Gray Metal", "Gold", "Copper", "Steel Blue", "Bronze"]
        if color_index >= 0:
            return color_names[color_index]
        return "no metallic color"

    @staticmethod
    def check_pic_for_metallic_HSV_color(bgr_frame):
        """Legacy compatibility method."""
        config = {'line_name': 'legacy'}
        calculator = HSVCalculator(config)
        return calculator.check_metallic_colors(bgr_frame)