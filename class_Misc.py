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

class DiskSpaceError(Exception):
    """Raised when insufficient disk space is available."""
    pass

class Misc:
    # Configuration constants
    MIN_DISK_SPACE_GB = 10
    JPEG_QUALITY = 100
    PNG_COMPRESSION = 0

    @staticmethod
    def reduce_frame_as_jpg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> np.ndarray:
        """Compress frame as JPEG and return decompressed version."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")
            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        jpeg_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return jpeg_image
    
    @staticmethod
    def check_disk_space(directory: Union[str, Path], min_gb: float = MIN_DISK_SPACE_GB) -> bool:
        """Check if sufficient disk space is available."""
        try:
            free_space_gb = shutil.disk_usage(directory).free / (1024**3)
            return free_space_gb > min_gb
        except OSError as e:
            logging.error(f"Error checking disk space: {e}")
            return False
    
    @staticmethod
    def write_image_to_dir(
        image: np.ndarray, 
        directory: Union[str, Path], 
        filename: str, 
        format_type: str = 'JPEG',
        quality: int = JPEG_QUALITY
    ) -> bool:
        """Write image to directory with disk space check."""
        directory = Path(directory)
        
        if not Misc.check_disk_space(directory, Misc.MIN_DISK_SPACE_GB - 5):
            raise DiskSpaceError("Insufficient disk space for image writing")
        
        try:
            directory.mkdir(parents=True, exist_ok=True)
            filepath = directory / filename
            
            if format_type.upper() == 'JPEG':
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            elif format_type.upper() == 'PNG':
                encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), Misc.PNG_COMPRESSION]
            else:
                encode_params = []
                
            success = cv2.imwrite(str(filepath), image, encode_params)
            if not success:
                logging.error(f"Failed to write image: {filepath}")
            return success
            
        except Exception as e:
            logging.error(f"Error writing image {filename}: {e}")
            return False
        
    @staticmethod
    def write_live_pic_to_dir(frame: np.ndarray, filename: str, work_dir: Union[str, Path]) -> bool:
        """Write live picture with error handling."""
        upload_dir = Path(work_dir) / "upload"
        return Misc.write_image_to_dir(frame, upload_dir, filename, 'JPEG')
                  
    @staticmethod
    def reboot_windows() -> None:
        """Reboot Windows system with logging."""
        try:
            dummy = np.zeros((1, 1), dtype=np.uint8)
            timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")
            reboot_file = f"C:/Reboot_windows_{timestamp}.jpg"
            cv2.imwrite(reboot_file, dummy)
            
            logging.info("Initiating Windows reboot in 60 seconds")
            os.system("shutdown /r /f /t 60")
        except Exception as e:
            logging.error(f"Error during reboot: {e}")
    
    @staticmethod
    def show_camera_image(
        image: np.ndarray, 
        scale_percent: int, 
        window_name: str, 
        text_overlay: str,
        display_available: bool = True
    ) -> np.ndarray:
        """Display camera image with text overlay and scaling."""
        if image is None or image.size == 0:
            logging.warning("Empty image provided to show_camera_image")
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        try:
            # Calculate new dimensions
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            
            # Resize image
            resized_image = cv2.resize(image, (width, height))
            
            # Add text overlay
            image_with_text = Misc.add_text_overlay(resized_image, text_overlay)
            
            # Only display if GUI is available
            if display_available:
                try:
                    cv2.imshow(window_name, image_with_text)
                    cv2.waitKey(1)
                except cv2.error as e:
                    logging.warning(f"OpenCV display not available: {e}")
                    # Save image instead of displaying
                    Misc.write_image_to_dir(image_with_text, "./logs/", f"display_{window_name}.jpg")
            else:
                logging.debug(f"Display disabled - image prepared for {window_name}")
            
            return image_with_text
            
        except Exception as e:
            logging.error(f"Error processing image for display: {e}")
            return image
    
    @staticmethod
    def resize_image(image: np.ndarray, scale_percent: int) -> np.ndarray:
        """Resize image by percentage with validation."""
        if image is None or image.size == 0:
            raise ValueError("Image is empty or None")
            
        if not 1 <= scale_percent <= 500:
            raise ValueError("Scale percent must be between 1 and 500")
            
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        
        return cv2.resize(image, (width, height))

    @staticmethod
    def get_region_around_pixel(
        pixel: Tuple[int, int], 
        frame: np.ndarray, 
        radius: int
    ) -> np.ndarray:
        """Extract region around pixel with bounds checking."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")
            
        x, y = pixel
        height, width = frame.shape[:2]
        
        # Calculate bounds with safety checks
        row_min = max(y - radius, 0)
        row_max = min(y + radius, height)
        col_min = max(x - radius, 0)
        col_max = min(x + radius, width)
        
        return frame[row_min:row_max, col_min:col_max]

    @staticmethod
    def extract_center_slice(
        image: np.ndarray, 
        slice_width: int, 
        scale_percent: int
    ) -> np.ndarray:
        """Extract center slice of frame and resize."""
        if image is None or image.size == 0:
            raise ValueError("Image is empty or None")
            
        height, width = image.shape[:2]
        center_x = width // 2
        
        # Extract center slice
        left_bound = max(center_x - slice_width // 2, 0)
        right_bound = min(center_x + slice_width // 2, width)
        
        slice_img = image[:, left_bound:right_bound]
        
        # Resize slice
        return Misc.resize_image(slice_img, scale_percent)
        
    @staticmethod
    def add_text_overlay(
        image: np.ndarray, 
        text: str, 
        position: Tuple[int, int] = (5, 20),
        font_scale: float = 0.5,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """Add text overlay to image."""
        if image is None or image.size == 0:
            raise ValueError("Image is empty or None")
            
        image_copy = image.copy()
        cv2.putText(
            image_copy, 
            text, 
            position, 
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        return image_copy
 
    @staticmethod
    def setup_logger(
        logger_name: str, 
        log_dir: Union[str, Path] = "./logs",
        max_bytes: int = 100000,
        backup_count: int = 3
    ) -> logging.Logger:
        """Set up rotating file logger with console output."""
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation and UTF-8 encoding
        log_file = log_dir / f'{logger_name}.log'
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'  # Add UTF-8 encoding to handle emojis
        )
        file_formatter = logging.Formatter(
            "%(asctime)s:%(levelname)s:%(message)s", 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(file_formatter)
        
        # Try to set UTF-8 encoding for console (Windows fix)
        try:
            if hasattr(console_handler.stream, 'reconfigure'):
                console_handler.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Ignore if reconfigure not available

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    @staticmethod
    def create_empty_mask(frame: np.ndarray) -> np.ndarray:
        """Create empty mask matching frame dimensions."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")
        return np.zeros(frame.shape, dtype=np.uint8)

    @staticmethod
    def blend_frame_with_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Blend frame with mask using bitwise AND."""
        if frame.shape != mask.shape:
            raise ValueError("Frame and mask dimensions must match")
        return cv2.bitwise_and(frame, mask)
        
    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """Calculate image entropy."""
        if image is None or image.size == 0:
            raise ValueError("Image is empty or None")
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / hist.sum()  # Normalize
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
        return round(entropy, 2)

    @staticmethod
    def check_yellow_percentage(
        hsv_image: np.ndarray, 
        lower_bound: Tuple[int, int, int], 
        upper_bound: Tuple[int, int, int]
    ) -> float:
        """Check percentage of yellow pixels in HSV image using vectorized operations."""
        if hsv_image is None or hsv_image.size == 0:
            raise ValueError("HSV image is empty or None")
            
        # Split image into thirds for analysis
        height, width = hsv_image.shape[:2]
        third_width = width // 3
        
        percentages = []
        
        for i in range(3):
            start_col = i * third_width
            end_col = (i + 1) * third_width if i < 2 else width
            
            section = hsv_image[:, start_col:end_col]
            
            # Create mask for yellow range
            mask = cv2.inRange(section, np.array(lower_bound), np.array(upper_bound))
            
            # Calculate percentage
            yellow_pixels = np.sum(mask > 0)
            total_pixels = section.shape[0] * section.shape[1]
            percentage = (yellow_pixels / total_pixels) * 100
            
            percentages.append(percentage)
        
        return round(max(percentages), 2)

    @staticmethod
    def calculate_brightness(bgr_frame: np.ndarray) -> float:
        """Calculate normalized brightness of BGR frame."""
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("BGR frame is empty or None")
            
        # Convert to grayscale for brightness calculation
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        return round(brightness, 2)
    
    @staticmethod
    def get_frame_statistics(
        bgr_frame: np.ndarray, 
        temperature_str: str,
        config: Dict[str, Any]
    ) -> Tuple[str, float, float, float]:
        """Calculate comprehensive frame statistics."""
        entropy = Misc.calculate_entropy(bgr_frame)
        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        
        yellow_percentage = Misc.check_yellow_percentage(
            hsv_frame,
            config["hsv_yellow_lower_bound"],
            config["hsv_yellow_upper_bound"]
        )
        
        brightness = Misc.calculate_brightness(bgr_frame)
        
        # Create info string
        info_parts = [
            f'Entropy: {entropy:.2f} (threshold: {config["entropy_threshold"]})',
            f'Yellow: {yellow_percentage:.2f}% (threshold: {config["hsv_yellow_percent_threshold"]})',
            f'Brightness: {brightness:.2f} (threshold: {config["brightness_threshold"]})',
            f'Temperature: {temperature_str} (max: {config["max_product_temperature"]})'
        ]
        
        frame_info = ', '.join(info_parts)
        
        return frame_info, entropy, yellow_percentage, brightness