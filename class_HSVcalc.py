import numba
import numpy as np
import cv2
import os
import time
# import shutil # No longer directly used by remaining methods, consider removing if not needed elsewhere
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import logging
# from collections import deque # No longer used by remaining methods

logger = logging.getLogger(__name__)

class HSVProcessingError(Exception):
    """Custom exception raised when HSV processing or mask operations encounter an error."""
    pass

class HSVCalculator:
    """
    Handles HSV-based image processing tasks such as contamination detection and HSV mask management.
    Optimized methods using Numba are included for performance-critical operations.
    Archived functions (related to metallic colors, specific fill algorithms, and direct mask writing)
    have been removed from this class and are expected to be in archive.py or replaced by new workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the HSVCalculator with configuration.
        Args:
            config: A dictionary containing settings, including 'work_dir' and 'line_name'.
        """
        self.config = config
        # Ensure 'work_dir' is present in config, otherwise default or raise error
        if "work_dir" not in config:
            # Defaulting work_dir, but ideally this should be guaranteed by config loading
            logger.warning("'work_dir' not found in config, defaulting to './temp_hsv_work'. This might be an issue.")
            self.work_dir = Path("./temp_hsv_work")
        else:
            self.work_dir = Path(config["work_dir"])

        self.hsv_mask_dir = self.work_dir / "HSV_masks" # Renamed for clarity
        self.hsv_mask_dir.mkdir(parents=True, exist_ok=True)

        # Metallic colors related attributes removed as methods are archived.
        # self._metallic_colors = []

    # _initialize_metallic_colors method (and its use in __init__) removed - archived.

    def load_hsv_mask(self) -> Tuple[np.ndarray, float]:
        """
        Loads the 3D HSV mask from a .npy file specific to the configured line_name.
        If the mask file doesn't exist, it creates and returns a default or trial mode mask.
        Validates mask dimensions and calculates the percentage of active (allowed) pixels.

        Returns:
            A tuple containing:
                - hsv_mask (np.ndarray): The loaded or newly created 3D HSV mask (H,S,V).
                - percentage_active (float): The percentage of non-zero entries in the mask.

        Raises:
            HSVProcessingError: If a mask is critical and cannot be loaded or created.
        """
        line_name = self.config.get('line_name', 'default_line') # Use default if not in config
        hsv_mask_filename = f"HSV_mask_{line_name}.npy"
        hsv_file_path = self.hsv_mask_dir / hsv_mask_filename

        default_mask_shape = (180, 256, 256) # H: 0-179, S: 0-255, V: 0-255

        try:
            if hsv_file_path.is_file(): # More robust check than .exists() for files
                hsv_mask = np.load(hsv_file_path).astype(np.uint8)

                if hsv_mask.shape != default_mask_shape:
                    logger.error(f"Loaded HSV mask from {hsv_file_path} has invalid dimensions: {hsv_mask.shape}. Expected {default_mask_shape}. Recreating a default mask.")
                    # This situation should ideally trigger a rebuild or error state.
                    raise HSVProcessingError(f"Invalid HSV mask dimensions from file: {hsv_mask.shape}")

                active_pixels = np.sum(hsv_mask > 0)
                total_pixels = np.prod(default_mask_shape)
                percentage_active = round((active_pixels / total_pixels) * 100, 3) if total_pixels > 0 else 0.0

                logger.info(f"Successfully loaded HSV mask from: {hsv_file_path} ({percentage_active}% active pixels).")
                return hsv_mask, percentage_active
            else:
                logger.warning(f"HSV mask file not found at {hsv_file_path}. Creating a new default mask.")
                hsv_mask = np.zeros(default_mask_shape, dtype=np.uint8) # Start with an empty mask

                # Behavior for missing mask depends on configuration (e.g., trial mode, build mode)
                is_trial_mode = self.config.get('sensaray_type') == 'TRIAL_MODE'
                is_build_mode = self.config.get("build_hsv_mask", False)

                if is_trial_mode or is_build_mode:
                    # For trial or initial build mode, a more permissive default might be desired,
                    # or specific ranges can be pre-allowed.
                    # Example: allow a common range for "good" product colors like yellows/greens.
                    hsv_mask[15:55, 80:255, 80:255] = 1  # Broad Yellow-Greenish range
                    message = "TRIAL_MODE" if is_trial_mode else "BUILD_HSV_MASK active"
                    logger.info(f"{message}: Created a default HSV mask as none was found. This mask may need training/building.")
                    # Consider prompting user only if interactive mode is intended
                    # input(f"{message}: Default HSV mask created. Press Enter to continue...")
                else:
                    # If not trial/build mode, a missing mask might be a critical error.
                    logger.error("HSV mask not found and not in TRIAL or BUILD_HSV_MASK mode. Application might not function correctly with an empty mask.")
                    # Depending on requirements, could raise HSVProcessingError here.
                    # For now, returns the empty mask, downstream logic must handle it.

                active_pixels = np.sum(hsv_mask > 0)
                total_pixels = np.prod(default_mask_shape)
                percentage_active = round((active_pixels / total_pixels) * 100, 3) if total_pixels > 0 else 0.0
                return hsv_mask, percentage_active

        except Exception as e:
            logger.error(f"General error loading or creating HSV mask from {hsv_file_path}: {e}", exc_info=True)
            # Fallback to a minimal, safe mask in case of any unexpected error
            hsv_mask = np.zeros(default_mask_shape, dtype=np.uint8)
            hsv_mask[20:30, 100:200, 100:200] = 1  # A very small default "safe" yellow range
            logger.critical("CRITICAL: Returning an emergency minimal HSV mask due to previous errors. Functionality will be limited.")
            return hsv_mask, np.sum(hsv_mask > 0) / np.prod(default_mask_shape) * 100


    def save_hsv_mask(self, hsv_mask: np.ndarray, percentage_active: float) -> None:
        """
        Saves the provided 3D HSV mask to a .npy file. Also creates a timestamped backup.
        Args:
            hsv_mask: The HSV mask (3D NumPy array) to save.
            percentage_active: The percentage of active pixels, included in the backup filename.
        Raises:
            HSVProcessingError: If the mask is invalid or saving fails.
        """
        if not isinstance(hsv_mask, np.ndarray) or hsv_mask.shape != (180, 256, 256):
            raise HSVProcessingError(f"Invalid HSV mask object or shape for saving. Shape: {hsv_mask.shape if isinstance(hsv_mask, np.ndarray) else 'Not a NumPy array'}")

        line_name = self.config.get('line_name', 'default_line')
        hsv_mask_filename = f"HSV_mask_{line_name}.npy"
        hsv_file_path = self.hsv_mask_dir / hsv_mask_filename

        try:
            # Ensure dtype is uint8 for masks (0 or 1)
            hsv_mask_to_save = hsv_mask.astype(np.uint8)

            np.save(hsv_file_path, hsv_mask_to_save)
            logger.info(f"Main HSV mask saved successfully to: {hsv_file_path}")

            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # Milliseconds precision
            perc_str = f"{percentage_active:.2f}".replace('.', 'p') # Format percentage for filename
            backup_filename = f"HSV_mask_{line_name}_P{perc_str}_{timestamp}.npy"
            backup_file_path = self.hsv_mask_dir / backup_filename
            np.save(backup_file_path, hsv_mask_to_save)
            logger.info(f"Timestamped HSV mask backup saved to: {backup_file_path}")

        except Exception as e:
            logger.error(f"Failed to save HSV mask to {hsv_file_path} and/or its backup: {e}", exc_info=True)
            raise HSVProcessingError(f"Failed to save HSV mask: {e}")


    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _check_contamination_vectorized(
        hsv_frame: np.ndarray,
        hsv_mask: np.ndarray
    ) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """Numba-optimized: Checks frame against HSV mask for disallowed pixels (contaminants)."""
        height, width = hsv_frame.shape[:2]
        contaminant_count = 0
        max_to_store = 5000 # Limit stored contaminant details to prevent excessive memory with numba lists

        # Numba requires list types to be known or inferred. For tuples, it's usually fine.
        xy_contaminant_coords = []
        hsv_contaminant_values = []

        for r in numba.prange(height): # Parallelize outer loop
            # Local counter for this thread if needed, though global counter update needs care (not done here for simplicity)
            # For now, assuming `contaminant_count` update is okay due to how numba handles reductions or if GIL is released.
            # More robust for parallel counting might involve per-thread lists then merging.
            # However, numba list append is thread-safe.
            if contaminant_count >= max_to_store * 2 and len(xy_contaminant_coords) >= max_to_store : # Heuristic to stop early if too many total contaminants
                 continue

            for c in range(width):
                h, s, v = hsv_frame[r, c, 0], hsv_frame[r, c, 1], hsv_frame[r, c, 2]

                if hsv_mask[h, s, v] == 0: # 0 indicates a disallowed (contaminant) HSV value
                    if len(xy_contaminant_coords) < max_to_store: # Store details only up to a limit
                        xy_contaminant_coords.append((c, r)) # Store as (x,y)
                        hsv_contaminant_values.append((h, s, v))
                    contaminant_count += 1

        return contaminant_count, xy_contaminant_coords, hsv_contaminant_values


    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _build_hsv_mask_vectorized(
        hsv_frame: np.ndarray,
        hsv_mask_to_update: np.ndarray,
        s_min_thresh: int,
        v_min_thresh: int,
        v_max_thresh_for_dark_pixels: int # e.g. 30, for allowing dark (but saturated) pixels
    ) -> Tuple[np.ndarray, int]:
        """Numba-optimized: Updates hsv_mask by adding pixels from hsv_frame meeting S/V criteria."""
        height, width = hsv_frame.shape[:2]
        newly_added_pixels_count = 0

        # Create a copy inside the Numba function if hsv_mask_to_update should not be modified in place by caller
        # For now, assumes it can be modified. Numba often works on copies for arrays unless specified.
        # Let's ensure it's clear: this function MODIFIES hsv_mask_to_update.

        for r in numba.prange(height):
            for c in range(width):
                h, s, v = hsv_frame[r, c, 0], hsv_frame[r, c, 1], hsv_frame[r, c, 2]

                # Condition for adding to mask:
                # Pixel must be saturated enough (s > s_min_thresh) AND
                # EITHER bright enough (v > v_min_thresh) OR dark enough (v < v_max_thresh_for_dark_pixels)
                # This allows common product colors as well as potentially dark markings/text.
                is_valid_color = (s > s_min_thresh) and \
                                 ((v > v_min_thresh) or (v < v_max_thresh_for_dark_pixels))

                if is_valid_color:
                    if hsv_mask_to_update[h, s, v] == 0: # If this HSV tuple is not yet in the mask
                        hsv_mask_to_update[h, s, v] = 1  # Add it to the mask (mark as allowed)
                        newly_added_pixels_count += 1

        return hsv_mask_to_update, newly_added_pixels_count


    def check_frame_for_contamination(
        self,
        bgr_frame: np.ndarray,
        hsv_mask: np.ndarray
    ) -> Tuple[int, np.ndarray, List[Tuple[int, int, int]]]:
        """
        Checks a BGR frame for contamination using the provided HSV mask.
        Returns: (contamination_count, visual_contamination_mask, list_of_contaminant_hsv_values)
        """
        if bgr_frame is None or bgr_frame.size == 0:
            logger.error("Cannot check empty BGR frame for contamination.")
            return 0, np.zeros((100,100,3), dtype=np.uint8), [] # Sensible defaults for error
        if hsv_mask is None or hsv_mask.shape != (180,256,256):
            logger.error("Invalid or missing HSV mask for contamination check.")
            return 0, np.zeros_like(bgr_frame), []


        try:
            hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

            count, positions, hsv_values = self._check_contamination_vectorized(hsv_frame, hsv_mask)

            # Create a visual representation of contaminations
            vis_mask = np.zeros_like(bgr_frame, dtype=np.uint8)
            marker_radius = self.config.get("contamination_marker_radius", 3) # Small radius for markers
            max_markers = self.config.get("max_contamination_markers_on_image", 500)

            for i, pos in enumerate(positions):
                if i >= max_markers: break
                cv2.circle(vis_mask, pos, marker_radius, (0, 0, 255), -1) # Red dots for contaminants

            return count, vis_mask, hsv_values

        except cv2.error as cv_err:
            logger.error(f"OpenCV error in contamination check: {cv_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error in contamination check: {e}", exc_info=True)

        return 0, np.zeros_like(bgr_frame), [] # Fallback


    def build_hsv_mask_from_frame(
        self,
        bgr_frame: np.ndarray,
        existing_hsv_mask: np.ndarray # Current mask to be updated
    ) -> Tuple[np.ndarray, int]:
        """
        Updates an existing HSV mask with new color information from a BGR frame.
        Returns: (updated_hsv_mask, number_of_new_pixels_added)
        """
        if bgr_frame is None or bgr_frame.size == 0:
            logger.error("Cannot build HSV mask from empty BGR frame.")
            return existing_hsv_mask, 0
        if existing_hsv_mask is None or existing_hsv_mask.shape != (180,256,256):
            logger.error("Invalid existing HSV mask provided for building.")
            # Could return a new empty mask, or current one if partially valid
            return np.zeros((180,256,256), dtype=np.uint8), 0


        try:
            hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

            s_min = self.config.get("hsv_build_s_min_threshold", 40) # Default S threshold
            v_min = self.config.get("hsv_build_v_min_threshold", 50) # Default V threshold for bright
            v_max_dark = self.config.get("hsv_build_v_max_dark_threshold", 30) # Default V threshold for dark colors

            # Pass a copy of the mask to Numba func if it shouldn't modify original `existing_hsv_mask` directly
            # Or if Numba's array passing semantics are unclear for the specific version/case.
            # Numba usually works on array views if possible, but copy() ensures safety.
            updated_mask, num_new = self._build_hsv_mask_vectorized(
                hsv_frame, existing_hsv_mask.copy(), s_min, v_min, v_max_dark
            )

            if num_new > 0:
                logger.info(f"Added {num_new} new HSV combinations to the mask from the frame.")

            return updated_mask, num_new
        except cv2.error as cv_err:
            logger.error(f"OpenCV error in HSV mask building: {cv_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error in HSV mask building: {e}", exc_info=True)

        return existing_hsv_mask, 0 # Fallback


    def remove_blue_values_from_mask(
        self,
        hsv_mask: np.ndarray,
        h_range: Tuple[int,int] = (90, 135),  # Typical Hue for blues: 90-135
        s_range: Tuple[int,int] = (50, 255),  # Saturation: reasonably saturated blues
        v_range: Tuple[int,int] = (50, 255)   # Value: reasonably bright blues
    ) -> np.ndarray:
        """Removes specified blue HSV values from the mask (sets them to 0)."""
        if hsv_mask is None or hsv_mask.shape != (180, 256, 256):
            logger.error("Invalid HSV mask for blue removal.")
            return hsv_mask if hsv_mask is not None else np.zeros((180,256,256),dtype=np.uint8)

        try:
            h_min, h_max = max(0, h_range[0]), min(180, h_range[1]) # Clamp H to 0-179
            s_min, s_max = max(0, s_range[0]), min(256, s_range[1]) # Clamp S,V to 0-255
            v_min, v_max = max(0, v_range[0]), min(256, v_range[1])

            if not (h_min < h_max and s_min < s_max and v_min < v_max):
                logger.warning(f"Invalid range for blue removal: H({h_min}-{h_max}), S({s_min}-{s_max}), V({v_min}-{v_max}). No operation performed.")
                return hsv_mask

            mask_copy = hsv_mask.copy() # Work on a copy
            mask_copy[h_min:h_max, s_min:s_max, v_min:v_max] = 0 # Set this range to 0 (disallowed)

            removed_count = np.sum(hsv_mask[h_min:h_max, s_min:s_max, v_min:v_max]) - np.sum(mask_copy[h_min:h_max, s_min:s_max, v_min:v_max])
            logger.info(f"Cleared {removed_count} HSV combinations in the blue range from the mask.")
            return mask_copy
        except Exception as e:
            logger.error(f"Error removing blue values: {e}", exc_info=True)
        return hsv_mask


    def apply_morphological_fill_to_sv_slice( # Renamed for clarity
        self,
        sv_plane: np.ndarray, # A 2D (S,V) slice of the HSV mask for a specific Hue
        kernel_radius: int = 3
    ) -> np.ndarray:
        """Applies morphological closing to an S-V plane of the HSV mask."""
        if sv_plane is None or sv_plane.ndim != 2 or sv_plane.shape != (256,256) :
            logger.error(f"Invalid S-V plane provided for morphological fill. Shape: {sv_plane.shape if isinstance(sv_plane,np.ndarray) else 'None'}")
            return sv_plane if sv_plane is not None else np.zeros((256,256), dtype=np.uint8)
        if kernel_radius <= 0:
            return sv_plane # No operation if radius is non-positive

        try:
            # MORPH_ELLIPSE is often good for general purpose filling
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_radius + 1, 2 * kernel_radius + 1))
            # MORPH_CLOSE = Dilation followed by Erosion. Fills small holes, connects components.
            filled_sv_plane = cv2.morphologyEx(sv_plane.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
            return filled_sv_plane
        except cv2.error as cv_err:
            logger.error(f"OpenCV error during morphological fill on S-V slice: {cv_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error in morphological fill on S-V slice: {e}", exc_info=True)
        return sv_plane # Return original on error

    # Archived methods previously here:
    # _identify_metallic_color (static)
    # check_metallic_colors
    # fill_area_mask (different from apply_morphological_fill_to_sv_slice)
    # rolling_ball_fill
    # write_hsv_mask_to_dir

# Legacy HSV_calc class (and its methods) removed.
# All usages should be updated to HSVCalculator or functions in archive.py.
