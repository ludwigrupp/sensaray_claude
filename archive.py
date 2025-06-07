import numpy as np
import cv2
import logging
from pathlib import Path
import shutil
from datetime import datetime
import os
import requests
from ruamel.yaml import YAML

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Placeholder for constants that might be needed
# Example: METALLIC_COLORS_HSV = {}

# ArenaCamera functions
def device_context(camera_instance):
    """
    Context manager for device handling.
    Adapted from ArenaCamera.device_context.
    """
    # Content of __enter__
    # This might need adjustment if camera_instance methods are not directly usable
    # or if specific attributes are expected to be set on the instance itself.
    logger.info("Entering device context")
    # Assuming camera_instance has a way to initialize or get devices
    # This is a placeholder for the actual logic from ArenaCamera
    try:
        # camera_instance.nodemap = camera_instance.device.nodemap
        # camera_instance.tl_stream_nodemap = camera_instance.device.tl_stream_nodemap
        # camera_instance.device.start_stream(1)
        logger.info("Stream started")
        yield camera_instance # Or whatever the original context manager returned
    except Exception as e:
        logger.error(f"Error in device_context: {e}")
        # Perform cleanup if necessary
        # if hasattr(camera_instance, 'device') and camera_instance.device:
        #     camera_instance.device.stop_stream()
        # ArenaSDK.Arena.close_device(camera_instance.device)
        # ArenaSDK.System.close_system()
        raise # Re-raise the exception if handling it here is not appropriate

    # Content of __exit__
    finally:
        logger.info("Exiting device context")
        # if hasattr(camera_instance, 'device') and camera_instance.device:
        #     camera_instance.device.stop_stream()
        #     logger.info("Stream stopped in device_context")
        #     # Potentially call cleanup_devices if it's a method on camera_instance
        #     if hasattr(camera_instance, 'cleanup_devices'):
        #         camera_instance.cleanup_devices()
        #     else:
        #         # Fallback to a more general cleanup if cleanup_devices is not available
        #         # ArenaSDK.Arena.close_device(camera_instance.device)
        #         # ArenaSDK.System.close_system()
        #         logger.info("Arena system and device closed in device_context (fallback)")
        # else:
        #     logger.info("No active device to clean up in device_context")
        pass # Placeholder for actual cleanup

def destroy_all_devices():
    """
    Destroys all devices.
    Adapted from ArenaCamera.destroy_all_devices (static method).
    """
    logger.info("Attempting to destroy all Arena devices.")
    # This would typically involve Arena SDK calls
    # from arena_api import ArenaSDK
    # ArenaSDK.System.close_system()
    # logger.info("Arena system closed by destroy_all_devices.")
    pass # Placeholder for Arena SDK calls

# EvoThermalSensor functions
def is_running(sensor_instance):
    """
    Checks if the EvoThermal sensor is running.
    Adapted from EvoThermalSensor.is_running.
    """
    # Assuming sensor_instance has an attribute like _is_running
    if hasattr(sensor_instance, '_is_running'):
        return sensor_instance._is_running
    elif hasattr(sensor_instance, 'is_running_flag'): # Alternative attribute name
        return sensor_instance.is_running_flag
    logger.warning("Sensor instance does not have a recognized 'is_running' attribute.")
    return False

def get_average_temperature(sensor_instance, region=None):
    """
    Gets the average temperature from the EvoThermal sensor.
    Adapted from EvoThermalSensor.get_average_temperature.
    """
    # Assuming sensor_instance has a method like 'get_temp' or similar
    # and handles regions internally or via parameters.
    if hasattr(sensor_instance, 'get_temp_frame_data_average'): # from the original EvoThermalSensor
        try:
            if region:
                # This part depends heavily on how the original class handled regions.
                # It might involve calling a method on sensor_instance that accepts a region,
                # or processing a full frame from sensor_instance and then applying the region.
                # For now, let's assume it might have a way to pass the region.
                # This is a placeholder for actual region processing.
                # temperatures = sensor_instance.get_temperatures(region=region)
                # return np.mean(temperatures) if temperatures else None

                # If the original method took a frame and a region:
                # frame = sensor_instance.get_temp_frame_data_average() # or similar
                # if frame is not None:
                #     x, y, w, h = region
                #     roi = frame[y:y+h, x:x+w]
                #     return np.mean(roi) if roi.size > 0 else None
                # return None
                logger.info(f"Region parameter provided to get_average_temperature: {region}. Region processing is illustrative.")
                # Fallback to average of the whole frame if region specific logic isn't clear
                return sensor_instance.get_temp_frame_data_average()

            return sensor_instance.get_temp_frame_data_average()
        except Exception as e:
            logger.error(f"Error getting average temperature: {e}")
            return None
    logger.warning("Sensor instance does not have 'get_temp_frame_data_average' method.")
    return None

# HSVCalculator functions

# Define metallic colors data directly in the archive script
# This was originally loaded from a YAML file in HSVCalculator
METALLIC_COLORS_HSV_ARCHIVED = {
    'gold': {'lower': [15, 100, 100], 'upper': [35, 255, 255]},
    'silver': {'lower': [0, 0, 150], 'upper': [180, 50, 255]},
    'copper': {'lower': [5, 100, 100], 'upper': [25, 255, 255]}, # Similar to orange/bronze
    'bronze': {'lower': [10, 100, 50], 'upper': [30, 255, 200]}
}

def _initialize_metallic_colors_archived():
    """
    Returns the predefined metallic colors.
    Adapted from HSVCalculator._initialize_metallic_colors.
    Now directly returns the hardcoded dict.
    """
    return METALLIC_COLORS_HSV_ARCHIVED

def _identify_metallic_color_archived(h, s, v):
    """
    Identifies a metallic color based on HSV values.
    Adapted from HSVCalculator._identify_metallic_color.
    Uses METALLIC_COLORS_HSV_ARCHIVED.
    """
    metallic_colors = _initialize_metallic_colors_archived()
    for color_name, ranges in metallic_colors.items():
        lower = np.array(ranges['lower'])
        upper = np.array(ranges['upper'])
        if lower[0] <= h <= upper[0] and \
           lower[1] <= s <= upper[1] and \
           lower[2] <= v <= upper[2]:
            return color_name
    return None

def check_metallic_colors(hsv_calculator_instance, bgr_frame):
    """
    Checks for metallic colors in a BGR frame.
    Adapted from HSVCalculator.check_metallic_colors.
    """
    if bgr_frame is None:
        logger.error("BGR frame is None in check_metallic_colors.")
        return {}

    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    # It's assumed hsv_calculator_instance might have attributes like self.h_avg, self.s_avg, self.v_avg
    # For a standalone function, we'd typically calculate these averages from the frame or a region.
    # If the original intent was to use pre-calculated averages from the instance,
    # that logic needs to be clarified. For now, let's average the whole frame.

    # If hsv_calculator_instance stores these as properties after some calculation:
    # h_avg = getattr(hsv_calculator_instance, 'h_avg', np.mean(hsv_frame[:,:,0]))
    # s_avg = getattr(hsv_calculator_instance, 's_avg', np.mean(hsv_frame[:,:,1]))
    # v_avg = getattr(hsv_calculator_instance, 'v_avg', np.mean(hsv_frame[:,:,2]))

    # For simplicity, let's assume we are checking the average HSV of the entire frame
    # or that the instance is expected to have these values set by another method.
    # This part is a bit ambiguous without knowing how h_avg, s_avg, v_avg were set in the original class.
    # Let's assume for now these are passed in or calculated if not on instance.
    # Fallback: calculate average from the frame if not available on instance
    h_avg = getattr(hsv_calculator_instance, 'h_avg', np.median(hsv_frame[:,:,0]))
    s_avg = getattr(hsv_calculator_instance, 's_avg', np.median(hsv_frame[:,:,1]))
    v_avg = getattr(hsv_calculator_instance, 'v_avg', np.median(hsv_frame[:,:,2]))

    identified_color = _identify_metallic_color_archived(h_avg, s_avg, v_avg)

    results = {
        'H_avg': h_avg,
        'S_avg': s_avg,
        'V_avg': v_avg,
        'identified_color': identified_color
    }
    if identified_color:
        logger.info(f"Identified metallic color: {identified_color} with HSV: ({h_avg:.2f}, {s_avg:.2f}, {v_avg:.2f})")
    else:
        logger.info(f"No specific metallic color identified for HSV: ({h_avg:.2f}, {s_avg:.2f}, {v_avg:.2f})")

    return results

def fill_area_mask(hsv_calculator_instance, hsv_mask, distance=1):
    """
    Fills the area mask using morphological operations.
    Adapted from HSVCalculator.fill_area_mask.
    hsv_calculator_instance is not directly used here but kept for consistency.
    """
    if hsv_mask is None:
        logger.error("HSV mask is None in fill_area_mask.")
        return None

    # Ensure mask is binary
    _, binary_mask = cv2.threshold(hsv_mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2 * distance + 1, 2 * distance + 1), np.uint8)
    filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel)

    # Optionally, use hsv_calculator_instance attributes if they control kernel size, iterations etc.
    # e.g., distance = getattr(hsv_calculator_instance, 'fill_distance', distance)

    logger.info(f"Area mask filled with distance: {distance}.")
    return filled_mask

def rolling_ball_fill(hsv_calculator_instance, slice_2d_orig, seed=(255, 255), radius=10):
    """
    Performs a rolling ball fill (flood fill) on a 2D slice.
    Adapted from HSVCalculator.rolling_ball_fill.
    hsv_calculator_instance is not directly used here but kept for consistency.
    """
    if slice_2d_orig is None:
        logger.error("Input slice_2d_orig is None in rolling_ball_fill.")
        return None

    h, w = slice_2d_orig.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Ensure seed point is within image boundaries
    seed_x, seed_y = seed
    if not (0 <= seed_x < w and 0 <= seed_y < h):
        logger.warning(f"Seed point {seed} is outside image dimensions ({w}x{h}). Using center as fallback.")
        seed_x, seed_y = w // 2, h // 2

    # The new value for floodFill is typically a scalar.
    # For a binary mask, this would be 255 (white).
    # The loDiff and upDiff control the tolerance. For filling a specific region based on seed color.
    # If slice_2d_orig is already a binary mask, loDiff and upDiff might be 0.
    # If it's a grayscale image, these values matter.

    # Assuming slice_2d_orig is a binary or grayscale image where we want to fill.
    # If it's binary and we want to fill a black area connected to seed with white:
    # The seed point value in slice_2d_orig[seed_y, seed_x] is important.
    # Let's assume we are filling based on the seed point's color.

    # Check if image is already binary (0 or 255)
    unique_values = np.unique(slice_2d_orig)
    is_binary = len(unique_values) <= 2 and (0 in unique_values or 255 in unique_values)

    fill_val = 255 # Color to fill with

    # For floodFill, loDiff and upDiff define the range relative to the seed pixel's value
    # within which other pixels will be filled.
    # If the image is binary and we want to fill a region of 0s (black) starting from a seed which is 0:
    # loDiff = 0, upDiff = 0. We are filling pixels with the same color as seed.
    # If the image is grayscale, these might need to be larger.

    # This part might need access to attributes on hsv_calculator_instance if they define these params
    lo_diff = getattr(hsv_calculator_instance, 'floodfill_lo_diff', 0 if is_binary else 10)
    up_diff = getattr(hsv_calculator_instance, 'floodfill_up_diff', 0 if is_binary else 10)

    # cv2.floodFill modifies the input image directly. Create a copy if original should be preserved.
    slice_2d_filled = slice_2d_orig.copy()

    cv2.floodFill(slice_2d_filled, mask, (seed_x, seed_y), fill_val, loDiff=lo_diff, upDiff=up_diff)

    # The 'radius' parameter was in the original function name, but not used in typical floodFill.
    # It might imply a morphological operation post-fill or a different algorithm.
    # If 'rolling_ball' implies a specific morphological structuring element:
    if radius > 0:
        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # This part is speculative based on "rolling ball" name.
        # If the fill was meant to be constrained or expanded by this.
        # For example, an opening or closing operation.
        # slice_2d_filled = cv2.morphologyEx(slice_2d_filled, cv2.MORPH_OPEN, kernel)
        logger.info(f"Rolling ball 'radius' {radius} might imply further morphological operations not fully implemented here.")

    logger.info(f"Rolling ball fill performed with seed: {seed}, radius: {radius} (radius effect speculative).")
    return slice_2d_filled


def write_hsv_mask_to_dir(hsv_calculator_instance, hsv_file_name, hsv_mask, hsv_file_name_archive, old_hsv_mask):
    """
    Writes the HSV mask to a directory and archives the old mask if it has changed.
    Adapted from HSVCalculator.write_hsv_mask_to_dir.
    Uses attributes from hsv_calculator_instance for paths.
    """
    # Attributes like 'hsv_dir_path_captured_images_masks', 'hsv_dir_path_archived_images_masks'
    # are expected on hsv_calculator_instance.

    hsv_dir_path_captured = getattr(hsv_calculator_instance, 'hsv_dir_path_captured_images_masks', Path("captured_masks"))
    hsv_dir_path_archived = getattr(hsv_calculator_instance, 'hsv_dir_path_archived_images_masks', Path("archived_masks"))

    # Ensure directories exist
    Path(hsv_dir_path_captured).mkdir(parents=True, exist_ok=True)
    Path(hsv_dir_path_archived).mkdir(parents=True, exist_ok=True)

    full_hsv_file_path = Path(hsv_dir_path_captured) / hsv_file_name
    full_hsv_archive_path = Path(hsv_dir_path_archived) / hsv_file_name_archive

    if hsv_mask is None:
        logger.error(f"HSV mask is None. Cannot write to {full_hsv_file_path}.")
        return False

    # Check if the new mask is different from the old one
    changed = True # Assume changed if old_hsv_mask is None
    if old_hsv_mask is not None:
        if old_hsv_mask.shape == hsv_mask.shape:
            difference = cv2.subtract(old_hsv_mask, hsv_mask)
            changed = np.any(difference) # True if any pixel is different
        else: # Shapes are different, so it definitely changed
            changed = True

    if changed:
        logger.info(f"HSV mask has changed or is new. Saving to {full_hsv_file_path}.")
        if old_hsv_mask is not None and Path(full_hsv_file_path).exists():
            # Archive the previous version if it exists
            try:
                shutil.move(str(full_hsv_file_path), str(full_hsv_archive_path))
                logger.info(f"Archived old HSV mask to {full_hsv_archive_path}.")
            except Exception as e:
                logger.error(f"Error archiving old HSV mask: {e}")

        try:
            cv2.imwrite(str(full_hsv_file_path), hsv_mask)
            logger.info(f"Successfully wrote new HSV mask to {full_hsv_file_path}.")
            # Update the instance's attribute for the old mask with the current one
            if hasattr(hsv_calculator_instance, 'last_saved_hsv_mask'):
                 setattr(hsv_calculator_instance, 'last_saved_hsv_mask', hsv_mask.copy())
            return True
        except Exception as e:
            logger.error(f"Error writing HSV mask to {full_hsv_file_path}: {e}")
            return False
    else:
        logger.info(f"HSV mask at {full_hsv_file_path} has not changed. No new file written.")
        return False


# Settings functions
def save_yaml_config_archived(data, config_file_path_str):
    """
    Saves data to a YAML file.
    Adapted from Settings.save_yaml_config (static method).
    """
    config_file_path = Path(config_file_path_str)
    try:
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file_path, 'w') as f:
            yaml.dump(data, f)
        logger.info(f"Successfully saved YAML config to {config_file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving YAML config to {config_file_path}: {e}")
        return False

if __name__ == '__main__':
    # Example usage or tests can be added here
    logger.info("archive.py executed as main.")

    # Example for save_yaml_config_archived
    # test_data = {'key': 'value', 'list': [1, 2, 3]}
    # save_yaml_config_archived(test_data, 'test_config.yaml')

    # To test other functions, you would need to create mock instances
    # of ArenaCamera, EvoThermalSensor, HSVCalculator.

    # Mock HSVCalculator instance for testing
    class MockHSVCalculator:
        def __init__(self):
            self.hsv_dir_path_captured_images_masks = "temp_captured_masks"
            self.hsv_dir_path_archived_images_masks = "temp_archived_masks"
            self.last_saved_hsv_mask = None # Store the last mask here
            # Attributes for floodfill_lo_diff, floodfill_up_diff if needed for rolling_ball_fill
            self.floodfill_lo_diff = 5
            self.floodfill_up_diff = 5
            # Attributes for average HSV if needed for check_metallic_colors
            self.h_avg = 0
            self.s_avg = 0
            self.v_avg = 0


    mock_hsv_calc = MockHSVCalculator()

    # Test write_hsv_mask_to_dir
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask1, (50,50), 20, 255, -1)

    # First save
    write_hsv_mask_to_dir(mock_hsv_calc, "test_mask.png", mask1, "archive_mask_v1.png", None)

    # Modify mask and save again (should archive the first one)
    mask2 = mask1.copy()
    cv2.rectangle(mask2, (10,10), (30,30), 128, -1) # Change the mask
    # The old_hsv_mask should ideally be what was previously at "test_mask.png" before this save
    # For this test, we pass what would have been the content of that file
    write_hsv_mask_to_dir(mock_hsv_calc, "test_mask.png", mask2, f"archive_mask_v2_{datetime.now().strftime('%Y%m%d%H%M%S')}.png", mask1)

    # Test fill_area_mask
    # Create a mask with a gap
    mask_with_gap = np.zeros((100,100), dtype=np.uint8)
    mask_with_gap[40:60, 20:40] = 255 # Part 1
    mask_with_gap[40:60, 60:80] = 255 # Part 2 (gap between 40 and 60)
    filled_mask_result = fill_area_mask(mock_hsv_calc, mask_with_gap, distance=10) # distance to bridge the gap
    if filled_mask_result is not None:
        logger.info(f"fill_area_mask executed. Check 'filled_mask_result.png' if saved.")
        # cv2.imwrite("filled_mask_result.png", filled_mask_result)

    # Test rolling_ball_fill
    slice_to_fill = np.zeros((100,100), dtype=np.uint8)
    slice_to_fill[40:60, 40:60] = 128 # A gray area
    # Seed inside the gray area
    filled_slice_result = rolling_ball_fill(mock_hsv_calc, slice_to_fill, seed=(50,50), radius=5)
    if filled_slice_result is not None:
        logger.info(f"rolling_ball_fill executed. Check 'filled_slice_result.png' if saved.")
        # cv2.imwrite("filled_slice_result.png", filled_slice_result)

    # Test check_metallic_colors
    # Create a dummy BGR frame (e.g., gold-like color)
    # Gold HSV is approx H=25, S=200, V=200
    # For BGR, cvtColor(np.uint8([[[25,200,200]]]), cv2.COLOR_HSV2BGR) -> [[[ 50 200 200]]]
    # This is not right, HSV values are H:0-179, S:0-255, V:0-255 in OpenCV
    # Gold: H ~22-38 (OpenCV: H/2 -> 11-19), S ~100-255, V ~100-255
    # Let's use H=15 (OpenCV), S=200, V=200
    # hsv_gold_color = np.uint8([[[15, 200, 200]]])
    # bgr_gold_frame = cv2.cvtColor(hsv_gold_color, cv2.COLOR_HSV2BGR)
    # print(f"BGR for gold-like: {bgr_gold_frame}") # [[[ 49 200 200]]]

    # Create a 10x10 frame of this color
    # bgr_frame_test = np.full((10, 10, 3), bgr_gold_frame[0,0,:], dtype=np.uint8)
    # Or just set the h_avg, s_avg, v_avg on the mock instance
    mock_hsv_calc.h_avg = 25 # Using typical H range 0-360 for this example, _identify_metallic_color_archived expects this
    mock_hsv_calc.s_avg = 200
    mock_hsv_calc.v_avg = 200
    # metallic_results = check_metallic_colors(mock_hsv_calc, bgr_frame_test) # Pass a dummy frame, averages will be from instance
    # logger.info(f"Metallic color check results: {metallic_results}")

    # Clean up test directories
    # if Path(mock_hsv_calc.hsv_dir_path_captured_images_masks).exists():
    #     shutil.rmtree(mock_hsv_calc.hsv_dir_path_captured_images_masks)
    # if Path(mock_hsv_calc.hsv_dir_path_archived_images_masks).exists():
    #     shutil.rmtree(mock_hsv_calc.hsv_dir_path_archived_images_masks)
    # if Path('test_config.yaml').exists():
    #     os.remove('test_config.yaml')
    pass
