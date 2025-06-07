# misc_utils.py
# This file has been refactored from the original class_Misc.py.
# All static methods from the former Misc class have been converted to regular functions.
# Class-level constants have been moved to module-level constants.
# Internal calls have been updated accordingly.
# This file is created anew after deleting the previous version due to tool issues.

import cv2
import os
import shutil
import numpy as np
import time
from datetime import datetime
from typing import Tuple, Optional, Union, Dict, Any
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Module-level constants
MIN_DISK_SPACE_GB: float = 10.0 # Minimum free disk space in Gigabytes required for operations.
JPEG_QUALITY: int = 90         # Default JPEG quality for image compression (0-100).
PNG_COMPRESSION: int = 1       # Default PNG compression level (0-9, 0 is none, 1 is fast, 9 is max).

class DiskSpaceError(Exception):
    """Custom exception raised when insufficient disk space is available for an operation."""
    pass

def setup_logger(
    logger_name: str,
    log_dir: Union[str, Path] = "./logs",
    log_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024, # 10MB, increased default
    backup_count: int = 7 # Increased backup count
) -> logging.Logger:
    """
    Set up a robust rotating file logger with console output.

    Features:
    - Rotating log files to manage disk space.
    - Separate log levels for file and console.
    - UTF-8 encoding for log files and console.
    - Detailed formatting for log messages.
    - Clears existing handlers to prevent duplication.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(min(log_level, console_level)) # Logger must be at least as permissive as its handlers

    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    log_file = log_path / f'{logger_name}.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    try:
        if hasattr(console_handler.stream, 'reconfigure') and hasattr(console_handler.stream, 'encoding') and console_handler.stream.encoding != 'utf-8':
            console_handler.stream.reconfigure(encoding='utf-8')
    except Exception as e:
        # Use a generic logger here, or print, as the primary logger is what's being set up.
        print(f"Debug: Could not reconfigure console stream for UTF-8 in setup_logger: {e}")


    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def reduce_frame_as_jpg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> np.ndarray:
    """
    Compresses an image frame using JPEG encoding and then decodes it.
    This can be used to simulate JPEG compression artifacts or to reduce frame data size if needed.
    """
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Input frame must be a non-empty NumPy array.")
    if not (0 <= quality <= 100):
        raise ValueError("JPEG quality must be between 0 and 100.")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, buffer = cv2.imencode('.jpg', frame, encode_param)
    if not result:
        raise RuntimeError("Failed to encode frame as JPEG.")

    jpeg_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if jpeg_image is None:
        raise RuntimeError("Failed to decode JPEG buffer.")
    return jpeg_image

def check_disk_space(directory: Union[str, Path], min_gb: float = MIN_DISK_SPACE_GB) -> bool:
    """
    Checks if the specified directory has at least `min_gb` of free disk space.
    Returns True if sufficient space is available, False otherwise.
    Uses module-level logger for errors.
    """
    logger = logging.getLogger(__name__) # Use a logger for messages
    try:
        dir_path = Path(directory)
        usage = shutil.disk_usage(dir_path)
        free_space_gb = usage.free / (1024**3)
        return free_space_gb >= min_gb
    except FileNotFoundError:
        logger.error(f"Directory not found for disk space check: {directory}")
        return False
    except OSError as e:
        logger.error(f"OS error during disk space check for {directory}: {e}")
        return False

def write_image_to_dir(
    image: np.ndarray,
    directory: Union[str, Path],
    filename: str,
    format_type: str = 'JPEG',
    quality: Optional[int] = None
) -> bool:
    """
    Writes an image to the specified directory with a disk space check.
    Supports JPEG and PNG formats. Uses module-level logger.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(image, np.ndarray) or image.size == 0:
        logger.error("Cannot write an empty or invalid image.")
        return False

    target_directory = Path(directory)

    required_space_check = max(0.1, MIN_DISK_SPACE_GB / 10)
    if not check_disk_space(target_directory, required_space_check):
        logger.error(f"Insufficient disk space in {target_directory} (requires > {required_space_check}GB). Image '{filename}' not written.")
        return False

    try:
        target_directory.mkdir(parents=True, exist_ok=True)
        filepath = target_directory / filename

        current_quality: int
        if format_type.upper() == 'JPEG':
            current_quality = quality if quality is not None else JPEG_QUALITY
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), current_quality]
        elif format_type.upper() == 'PNG':
            current_quality = quality if quality is not None else PNG_COMPRESSION
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), current_quality]
        else:
            logger.info(f"Writing image {filepath} with default OpenCV parameters for format {format_type}.")
            encode_params = []

        success = cv2.imwrite(str(filepath), image, encode_params)
        if not success:
            logger.error(f"OpenCV failed to write image: {filepath}")
        else:
            logger.debug(f"Successfully wrote image {filepath}")
        return success

    except Exception as e:
        logger.error(f"Exception writing image {filename} to {directory}: {e}", exc_info=True)
        return False

def write_live_pic_to_dir(frame: np.ndarray, filename: str, work_dir: Union[str, Path]) -> bool:
    """
    Convenience function to write a 'live' picture (frame from a camera) to an 'upload' subdirectory.
    Uses JPEG format by default.
    """
    upload_dir = Path(work_dir) / "upload"
    return write_image_to_dir(frame, upload_dir, filename, 'JPEG', quality=JPEG_QUALITY)

def reboot_windows() -> None:
    """
    Initiates a forced reboot of the Windows system after a 60-second delay.
    Logs the action and creates a temporary marker file.
    Caution: This will interrupt system operations. Uses module-level logger.
    """
    logger = logging.getLogger(__name__)
    try:
        marker_dir = Path(os.getenv("TEMP", "C:/Temp")) / "reboot_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        reboot_marker_file = marker_dir / f"reboot_initiated_{timestamp}.log"
        with open(reboot_marker_file, "w") as f:
            f.write(f"Windows reboot initiated by application at {timestamp}.")

        logger.critical(f"CRITICAL: Initiating Windows system reboot in 60 seconds. Marker: {reboot_marker_file}")
        os.system("shutdown /r /f /t 60 /c \"Application requested system reboot\"")
    except Exception as e:
        logger.error(f"Error preparing for Windows reboot: {e}", exc_info=True)

def resize_image(image: np.ndarray, scale_percent: int) -> np.ndarray: # Moved up to be available for show_camera_image
    """
    Resizes an image by a given percentage.
    Uses INTER_AREA for shrinking and INTER_LINEAR for enlarging.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image to resize must be a non-empty NumPy array.")
    if not (1 <= scale_percent <= 2000):
        raise ValueError("Scale percent must be between 1 and 2000.")

    new_width = int(image.shape[1] * scale_percent / 100)
    new_height = int(image.shape[0] * scale_percent / 100)

    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"Calculated resize dimensions are invalid: {new_width}x{new_height} from scale {scale_percent}%.")

    interpolation = cv2.INTER_AREA if scale_percent < 100 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

def add_text_overlay( # Moved up to be available for show_camera_image
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 25),
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA
) -> np.ndarray:
    """
    Adds a text overlay to an image. Works on a copy of the image.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image for text overlay must be a non-empty NumPy array.")

    image_with_text = image.copy()

    cv2.putText(
        image_with_text,
        text,
        position,
        font_face,
        font_scale,
        color,
        thickness,
        line_type
    )
    return image_with_text

def show_camera_image(
    image: np.ndarray,
    scale_percent: int,
    window_name: str,
    text_overlay: str,
    display_available: bool = True
) -> np.ndarray:
    """
    Displays an image in an OpenCV window with scaling and text overlay.
    If display is unavailable, it logs a warning and can save the image as a fallback.
    Returns the (potentially modified) image. Uses module-level logger.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(image, np.ndarray) or image.size == 0:
        logger.warning("Empty or invalid image provided to show_camera_image. Returning a black placeholder.")
        return np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        scaled_image = resize_image(image, scale_percent)
        image_with_text = add_text_overlay(scaled_image, text_overlay)

        if display_available:
            try:
                cv2.imshow(window_name, image_with_text)
                cv2.waitKey(1)
            except cv2.error as e:
                logger.warning(f"OpenCV display error for window '{window_name}': {e}. Display might not be available.")
                fallback_dir = Path("./logs/display_fallbacks")
                fallback_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime('%Y%m%d%H%M%S')
                fallback_filename = f"display_fallback_{window_name}_{timestamp}.jpg"
                write_image_to_dir(image_with_text, fallback_dir, fallback_filename) # Uses the main write_image_to_dir
                logger.info(f"Saved image to {fallback_dir / fallback_filename} as display fallback.")
        else:
            logger.debug(f"Display for window '{window_name}' is not enabled. Image prepared but not shown.")

        return image_with_text

    except Exception as e:
        logger.error(f"General error in show_camera_image for '{window_name}': {e}", exc_info=True)
        return image.copy() if image is not None and image.size > 0 else np.zeros((100,100,3), dtype=np.uint8)

def get_region_around_pixel(
    pixel: Tuple[int, int],
    frame: np.ndarray,
    radius: int
) -> np.ndarray:
    """
    Extracts a square region of `(2*radius+1)x(2*radius+1)` around a specified pixel from a frame.
    Handles boundary conditions gracefully.
    """
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Input frame must be a non-empty NumPy array.")
    if not (isinstance(pixel, tuple) and len(pixel) == 2 and all(isinstance(coord, int) for coord in pixel)):
        raise ValueError("Pixel must be a tuple of two integers (x, y).")
    if not (isinstance(radius, int) and radius > 0):
        raise ValueError("Radius must be a positive integer.")

    x, y = pixel
    height, width = frame.shape[:2]

    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Pixel coordinates ({x},{y}) are outside frame dimensions ({width}x{height}).")

    row_min = max(y - radius, 0)
    row_max = min(y + radius + 1, height)
    col_min = max(x - radius, 0)
    col_max = min(x + radius + 1, width)

    region = frame[row_min:row_max, col_min:col_max]
    if region.size == 0:
        logging.getLogger(__name__).warning(f"Extracted region is empty for pixel {pixel}, radius {radius}. This may indicate an issue.")
    return region

def extract_center_slice(
    image: np.ndarray,
    slice_width_pixels: int,
    scale_percent: int
) -> np.ndarray:
    """
    Extracts a vertical slice from the center of an image, then resizes it.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image must be a non-empty NumPy array.")
    if not (isinstance(slice_width_pixels, int) and slice_width_pixels > 0):
        raise ValueError("Slice width (pixels) must be a positive integer.")

    img_height, img_width = image.shape[:2]
    center_x = img_width // 2

    if slice_width_pixels > img_width:
        logger.warning(f"Slice width {slice_width_pixels}px exceeds image width {img_width}px. Clamping to image width.")
        slice_width_pixels = img_width

    left_bound = max(center_x - slice_width_pixels // 2, 0)
    right_bound = min(center_x + slice_width_pixels // 2, img_width)

    center_slice = image[:, left_bound:right_bound]

    if center_slice.size == 0:
        raise ValueError("Extracted center slice is empty. Check image dimensions and slice_width_pixels.")

    return resize_image(center_slice, scale_percent)

def create_empty_mask(frame: np.ndarray, num_channels: int = 1) -> np.ndarray:
    """
    Creates an empty (all zeros) mask with the same height and width as the input frame.
    The mask will have `num_channels` (default 1 for grayscale mask).
    """
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Input frame to create mask from must be a non-empty NumPy array.")
    if not (num_channels == 1 or num_channels == 3):
        raise ValueError("Number of channels for mask must be 1 (grayscale) or 3 (color).")

    height, width = frame.shape[:2]
    if num_channels == 1:
        return np.zeros((height, width), dtype=np.uint8)
    else:
        return np.zeros((height, width, 3), dtype=np.uint8)

def blend_frame_with_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blends a frame with a mask using bitwise AND.
    If the frame is color and mask is grayscale, the mask is converted to color before blending.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(frame, np.ndarray) or not isinstance(mask, np.ndarray):
        raise ValueError("Frame and mask must be NumPy arrays.")
    if frame.size == 0 or mask.size == 0:
        raise ValueError("Frame or mask is empty.")

    frame_channels = frame.shape[2] if len(frame.shape) == 3 else 1
    mask_channels = mask.shape[2] if len(mask.shape) == 3 else 1

    compatible_mask = mask
    if frame_channels == 3 and mask_channels == 1:
        if frame.shape[:2] != mask.shape[:2]:
             raise ValueError(f"Frame (H,W {frame.shape[:2]}) and mask (H,W {mask.shape[:2]}) dimensions must match for grayscale mask conversion.")
        compatible_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif frame.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Frame (H,W {frame.shape[:2]}) and mask (H,W {mask.shape[:2]}) dimensions must match.")
    elif frame_channels != mask_channels and not (frame_channels == 3 and mask_channels == 1) :
         raise ValueError(f"Frame channels ({frame_channels}) and mask channels ({mask_channels}) are incompatible.")

    if frame.shape != compatible_mask.shape:
         logger.error(f"Shape mismatch before blending: Frame {frame.shape}, Mask {compatible_mask.shape}")
         raise ValueError(f"After processing, frame shape {frame.shape} and mask shape {compatible_mask.shape} are still incompatible.")

    return cv2.bitwise_and(frame, compatible_mask)

def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculates the Shannon entropy of an image.
    The image is converted to grayscale if it's in color.
    Entropy is a measure of randomness or information content.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image for entropy calculation must be a non-empty NumPy array.")

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray_image = image
    else:
        raise ValueError(f"Unsupported image shape for entropy: {image.shape}. Must be 2D or 3D.")

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    if hist is None or hist.sum() == 0:
        return 0.0

    hist_normalized = hist.ravel() / hist.sum()

    entropy_val = -np.sum(hist_normalized * np.log2(hist_normalized + np.finfo(float).eps))
    return round(float(entropy_val), 3)

def check_yellow_percentage(
    hsv_image: np.ndarray,
    lower_hsv_bound: Tuple[int, int, int],
    upper_hsv_bound: Tuple[int, int, int]
) -> float:
    """
    Calculates the maximum percentage of pixels within a given HSV range, analyzed in three vertical sections of the image.
    This is useful for detecting color presence in specific parts of an image.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(hsv_image, np.ndarray) or hsv_image.size == 0:
        raise ValueError("Input HSV image for yellow check must be a non-empty NumPy array.")
    if not (len(hsv_image.shape) == 3 and hsv_image.shape[2] == 3):
        raise ValueError("Input image must be a 3-channel HSV image.")
    if not (isinstance(lower_hsv_bound, tuple) and len(lower_hsv_bound) == 3 and isinstance(upper_hsv_bound, tuple) and len(upper_hsv_bound) == 3):
        raise ValueError("Lower and upper HSV bounds must be tuples of 3 integers.")

    height, width = hsv_image.shape[:2]
    if width == 0 or height == 0: return 0.0

    third_width = width // 3
    if third_width == 0 and width > 0 : # Handle very narrow images
        logger.debug("Image too narrow for thirds, analyzing as a single section.")
        third_width = width

    section_percentages = []

    for i in range(3):
        if third_width == width and i > 0: break # Analyzed as one section

        start_col = i * third_width
        end_col = ((i + 1) * third_width) if (i < 2 and third_width != width) else width

        if start_col >= end_col: continue

        image_section = hsv_image[:, start_col:end_col]

        if image_section.size == 0:
            section_percentages.append(0.0)
            continue

        mask = cv2.inRange(image_section, np.array(lower_hsv_bound), np.array(upper_hsv_bound))

        positive_pixels = np.count_nonzero(mask)
        total_pixels_in_section = image_section.shape[0] * image_section.shape[1]

        if total_pixels_in_section == 0:
            section_percentages.append(0.0)
        else:
            percentage = (positive_pixels / total_pixels_in_section) * 100.0
            section_percentages.append(percentage)

    return round(max(section_percentages) if section_percentages else 0.0, 2)

def calculate_brightness(bgr_frame: np.ndarray) -> float:
    """
    Calculates the normalized brightness of a BGR frame.
    Brightness is defined as the mean pixel intensity of the grayscale equivalent, scaled to [0, 1].
    """
    if not isinstance(bgr_frame, np.ndarray) or bgr_frame.size == 0:
        raise ValueError("Input BGR frame for brightness calculation must be a non-empty NumPy array.")
    if not (len(bgr_frame.shape) == 3 and bgr_frame.shape[2] == 3):
        raise ValueError("Input frame must be a 3-channel BGR image.")

    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    brightness_val = np.mean(gray_frame) / 255.0
    return round(brightness_val, 3)

def get_frame_statistics(
    bgr_frame: np.ndarray,
    temperature_str: str,
    config: Dict[str, Any]
) -> Tuple[str, float, float, float]:
    """
    Calculates and aggregates several statistics for a BGR frame:
    entropy, yellow percentage (based on HSV), and brightness.
    Returns a formatted string of these statistics and the raw values.
    Requires a configuration dictionary for various thresholds and parameters.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(bgr_frame, np.ndarray) or bgr_frame.size == 0:
        raise ValueError("Input BGR frame for statistics must be a non-empty NumPy array.")
    if not isinstance(config, dict):
        raise TypeError("Configuration 'config' must be a dictionary.")

    required_keys = [
        "hsv_yellow_lower_bound", "hsv_yellow_upper_bound", "entropy_threshold",
        "hsv_yellow_percent_threshold", "brightness_threshold", "max_product_temperature"
    ]
    for key in required_keys:
        if key not in config:
            # Log error and raise, as this is a critical configuration issue
            err_msg = f"Missing required key '{key}' in configuration dictionary for frame statistics."
            logger.error(err_msg)
            raise ValueError(err_msg)

    entropy = calculate_entropy(bgr_frame)

    try:
        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        logger.error(f"Failed to convert BGR frame to HSV for statistics: {e}", exc_info=True)
        error_message = "Frame Conv. Error"
        return error_message, entropy, 0.0, 0.0

    yellow_percentage = check_yellow_percentage(
        hsv_frame,
        tuple(config["hsv_yellow_lower_bound"]),
        tuple(config["hsv_yellow_upper_bound"])
    )

    brightness = calculate_brightness(bgr_frame)

    stats_info_str = (
        f"Entropy: {entropy:.2f} (Th: {config['entropy_threshold']}), "
        f"Yellow: {yellow_percentage:.2f}% (Th: {config['hsv_yellow_percent_threshold']}), "
        f"Brightness: {brightness:.2f} (Th: {config['brightness_threshold']}), "
        f"Temp: {temperature_str} (Max: {config['max_product_temperature']})"
    )

    return stats_info_str, entropy, yellow_percentage, brightness
