"""
Sensaray Industrial Vision Inspection System v2.0
Improved version with better error handling, performance, and maintainability.
Refactored to use utility modules (misc_utils, config_utils) instead of classes (Misc, Settings).
"""

import time
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List # Added List
import cv2
import numpy as np
import logging
from pathlib import Path
import signal
import os
import platform

# Refactored imports
from class_Arena import ArenaCamera, CameraError
from class_EvoThermal import EvoThermalSensor, ThermalSensorError # EvoThermalSensor.THERMAL_RESOLUTION is used
from class_HSVcalc import HSVCalculator, HSVProcessingError
import misc_utils # Replaces `from class_Misc import Misc`
from misc_utils import DiskSpaceError # Specific error class from misc_utils
import config_utils # Replaces `from class_Settings import Settings`
from config_utils import ConfigurationError # Specific error class from config_utils

def setup_opencv_environment():
    """
    Configures OpenCV environment settings, especially for headless systems or VMs.
    Returns:
        bool: True if a display is likely available, False otherwise.
    """
    display_available = True
    # Use a local logger instance for this utility function to avoid dependency on global setup
    logger = logging.getLogger(__name__ + ".opencv_env_setup")
    # Basic console logging for this function if it's called before main logger setup
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    try:
        if platform.system() == "Linux" and not os.environ.get('DISPLAY'):
            display_available = False
            logger.info("Linux environment without DISPLAY variable detected.")
        elif platform.system() == "Darwin": # macOS
            # Check for WindowServer process as an indicator of GUI session
            import subprocess
            try:
                result = subprocess.run(['pgrep', '-x', 'WindowServer'], capture_output=True, text=True, check=False)
                if result.returncode != 0 or not result.stdout.strip(): # No WindowServer process found
                    display_available = False
                    logger.info("macOS WindowServer process not found. Assuming no GUI.")
            except FileNotFoundError:
                logger.warning("'pgrep' command not found on macOS. Cannot accurately determine GUI availability. Assuming display is available.")
            except Exception as e_pgrep:
                logger.warning(f"Error checking for WindowServer on macOS: {e_pgrep}. Assuming display is available.")
        elif platform.system() == "Windows":
            # Attempt to create and destroy a window as a test for GUI availability
            try:
                cv2.namedWindow("opencv_gui_availability_test", cv2.WINDOW_AUTOSIZE)
                cv2.destroyWindow("opencv_gui_availability_test")
            except cv2.error: # Specific OpenCV error if GUI operations fail
                display_available = False
                logger.info("OpenCV GUI test failed on Windows. Assuming no display or GUI context.")
            except Exception as e_cv_win_test:
                logger.warning(f"Unexpected error during OpenCV GUI test on Windows: {e_cv_win_test}. Assuming display for now.")
                # display_available = False # Or assume true if unsure

    except Exception as e_platform_check:
        logger.error(f"Error detecting display availability: {e_platform_check}. Assuming no display.", exc_info=True)
        display_available = False

    if not display_available:
        logger.warning("No active display detected or GUI functions are unavailable. OpenCV GUI features will be disabled.")
        # This environment variable is specific to some OpenCV backends (e.g., MSMF on Windows)
        # and might help avoid errors or warnings when running headless.
        os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

    return display_available

class SensaraySystem:
    """Main controller for the Sensaray vision inspection system."""

    def __init__(self, config_file_path_override: Optional[str] = None):
        """
        Initializes the Sensaray system.
        Args:
            config_file_path_override: Optional path to a config file to use instead of the default.
        """
        self.display_available: bool = setup_opencv_environment()

        # Logging is set up first, using the function from misc_utils
        self.logger = misc_utils.setup_logger('sensaray_main_log') # Changed logger name

        if not self.display_available:
            self.logger.warning("System running in headless mode. All GUI-based display features will be disabled.")

        try:
            # Configuration loading using config_utils module
            self.config: Dict[str, Any] = config_utils.load_yaml_config(config_file_path=config_file_path_override)
            if not self.config: # Should be handled by load_yaml_config raising error
                 self.logger.critical("Critical: Configuration loading returned empty. System cannot proceed.")
                 raise ConfigurationError("Failed to load any configuration.")

            # If display is not available, ensure 'show_pics' is False in the config
            if not self.display_available and self.config.get('show_pics', False):
                self.config['show_pics'] = False
                self.logger.info("Configuration updated: 'show_pics' automatically set to False due to unavailable display.")

            # Initialize main components
            self.camera_system = ArenaCamera(self.config)
            self.hsv_calculator = HSVCalculator(self.config)
            self.thermal_sensor: Optional[EvoThermalSensor] = None # Type hint for clarity
            self.devices: Dict[str, Any] = {} # To store initialized device objects (cameras, etc.)

            self.is_running: bool = False # Controls the main processing loop
            self.stats: Dict[str, Any] = { # For tracking operational statistics
                'frames_processed_total': 0,
                'anomalies_detected_total': 0,
                'last_anomaly_timestamp': None, # Changed from last_anomaly_time
                'anomalies_this_hour_count': 0,    # Renamed for clarity
                'current_processing_hour_start_time': time.monotonic() # For hourly rate limiting
            }

            # Setup signal handlers for graceful shutdown on SIGINT (Ctrl+C) and SIGTERM
            signal.signal(signal.SIGINT, self._graceful_signal_handler)
            signal.signal(signal.SIGTERM, self._graceful_signal_handler)

            self.logger.info(f"SensaraySystem initialized. Type: {self.config.get('sensaray_type', 'NOT_SPECIFIED')}")

        except ConfigurationError as e_conf:
            self.logger.critical(f"Initialization failed due to ConfigurationError: {e_conf}", exc_info=True)
            raise # Re-raise to stop execution if essential config is missing/invalid
        except Exception as e_init:
            self.logger.critical(f"Unexpected error during SensaraySystem initialization: {e_init}", exc_info=True)
            raise


    def _graceful_signal_handler(self, signum: int, frame: Optional[Any]):
        """Handles SIGINT and SIGTERM for a graceful shutdown."""
        signal_name = signal.Signals(signum).name
        self.logger.info(f"Signal {signal_name} ({signum}) received. Initiating graceful shutdown...")
        self.is_running = False # This flag will be checked by processing loops to exit


    def initialize_sensors(self) -> bool:
        """Initializes sensors based on the 'sensaray_type' configuration."""
        try:
            sensor_config_type = self.config.get('sensaray_type')
            buffer_count = self.config.get('buffersize', 10) # Default camera buffer size

            self.logger.info(f"Initializing sensors for system type: {sensor_config_type}")

            if sensor_config_type == 'BO3_400_noIR':
                cam_dev_0, cam_dev_1 = self.camera_system.initialize_dual_camera(buffer_count)
                self.devices = {'camera_0': cam_dev_0, 'camera_1': cam_dev_1}
            elif sensor_config_type == 'BO2_400_EVO':
                cam_dev_0 = self.camera_system.initialize_single_camera(buffer_count)
                self.devices = {'camera_0': cam_dev_0}
                self.thermal_sensor = EvoThermalSensor() # auto_connect=True by default
                if not self.thermal_sensor.is_connected:
                    self.logger.warning("EvoThermalSensor failed to auto-connect. Thermal data might be unavailable.")
            elif sensor_config_type == 'BO2_400_noIR':
                cam_dev_0 = self.camera_system.initialize_single_camera(buffer_count)
                self.devices = {'camera_0': cam_dev_0}
            elif sensor_config_type == 'BO2_400_iTec': # Corrected type from original
                cam_dev_0 = self.camera_system.initialize_single_camera(buffer_count)
                self.devices = {'camera_0': cam_dev_0}
                self.logger.warning("iTecIRSensor specific integration not implemented. No thermal data for BO2_400_iTec.")
            elif sensor_config_type == 'TRIAL_MODE':
                self.logger.info("TRIAL_MODE: Mock sensor initialization. No physical hardware needed.")
                return True # No actual hardware to initialize
            else:
                raise ConfigurationError(f"Unsupported 'sensaray_type' in configuration: {sensor_config_type}")

            self._warm_up_sensors_after_init() # Renamed for clarity
            self.logger.info("All configured sensors initialized successfully.")
            return True

        # Specific exceptions from components
        except CameraError as e_cam:
            self.logger.error(f"CameraError during sensor initialization: {e_cam}", exc_info=True)
        except ThermalSensorError as e_therm:
             self.logger.error(f"ThermalSensorError during sensor initialization: {e_therm}", exc_info=True)
        # General configuration or other errors
        except ConfigurationError as e_conf: # Re-raise if it's a config issue found here
            self.logger.error(f"ConfigurationError during sensor initialization: {e_conf}", exc_info=True)
            raise
        except Exception as e_gen_init:
            self.logger.error(f"Unexpected error during sensor initialization: {e_gen_init}", exc_info=True)

        self.cleanup_sensors() # Attempt to clean up any partially initialized resources
        return False


    def _warm_up_sensors_after_init(self, num_warm_up_frames: int = 3) -> None: # Reduced default
        self.logger.info(f"Warming up sensors by reading and discarding {num_warm_up_frames} frames...")
        for i in range(num_warm_up_frames):
            try:
                _, _ = self.read_sensor_data() # Read and discard the data
                self.logger.debug(f"Warm-up frame {i + 1}/{num_warm_up_frames} processed.")
            except Exception as e_warmup:
                self.logger.warning(f"Error during sensor warm-up frame {i + 1}: {e_warmup}. Continuing warm-up sequence.", exc_info=False) # Keep log clean
                time.sleep(0.2) # Small delay before next attempt if one fails


    def read_sensor_data(self) -> Tuple[Optional[np.ndarray], np.ndarray]: # Thermal frame always returns at least zeros
        """Reads data from connected sensors based on configuration."""
        current_frame: Optional[np.ndarray] = None
        current_thermal_frame: np.ndarray = np.zeros(EvoThermalSensor.THERMAL_RESOLUTION, dtype=np.float32) # Default empty
        sensor_type = self.config.get('sensaray_type')

        try:
            if sensor_type == 'BO3_400_noIR':
                current_frame = self.camera_system.read_dual_camera_frame(
                    self.devices['camera_0'], self.devices['camera_1']
                )
            elif sensor_type in ['BO2_400_EVO', 'BO2_400_noIR', 'BO2_400_iTec']:
                current_frame = self.camera_system.read_single_camera_frame(self.devices['camera_0'])
                if self.thermal_sensor and sensor_type == 'BO2_400_EVO' and self.thermal_sensor.is_connected:
                    read_thermal = self.thermal_sensor.read_thermal_frame()
                    if read_thermal is not None: current_thermal_frame = read_thermal
                # elif self.thermal_sensor and sensor_type == 'BO2_400_iTec': ...
            elif sensor_type == 'TRIAL_MODE': # This method is overridden in TrialSensaraySystem
                self.logger.debug("TRIAL_MODE: read_sensor_data called on base class. This should be handled by subclass.")
                # Return dummy data for base class trial mode if not overridden (though it is)
                current_frame = np.full((480, 640, 3), 128, dtype=np.uint8) # Grey image
                current_thermal_frame = np.full(EvoThermalSensor.THERMAL_RESOLUTION, 25.0, dtype=np.float32)
            else:
                # This should ideally be caught during init, but as a safeguard:
                raise ConfigurationError(f"Sensor reading not implemented for sensaray_type: {sensor_type}")
        except CameraError as e:
            self.logger.error(f"CameraError while reading sensor data: {e}", exc_info=True) # Log with stack trace
            # Optional: Implement retry logic or specific error handling here
        except ThermalSensorError as e: # Assuming EvoThermalSensor raises this
            self.logger.error(f"ThermalSensorError while reading data: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error during sensor data acquisition: {e}", exc_info=True)

        return current_frame, current_thermal_frame


    def process_frame(
        self,
        frame: Optional[np.ndarray],
        thermal_frame: np.ndarray, # Expect thermal_frame to always be provided (even if zeros)
        frame_counter: int
    ) -> Optional[Dict[str, Any]]:
        """Processes a single frame of data from the sensors."""
        if frame is None:
            self.logger.error(f"Frame {frame_counter}: Input camera frame is None. Cannot process.")
            return None # Critical error if no camera frame

        try:
            current_timestamp = datetime.now()

            # Temperature processing (region of interest, correction)
            temp_roi_coords = ( # x1, y1, x2, y2
                self.config.get('thermal_roi_x1', 0), 0,
                EvoThermalSensor.THERMAL_RESOLUTION[1] - self.config.get('thermal_roi_x2_offset', 0), # width - offset_from_right
                EvoThermalSensor.THERMAL_RESOLUTION[0] # Full height
            )
            # Ensure ROI coords are valid
            temp_roi_coords = (
                max(0, temp_roi_coords[0]), max(0, temp_roi_coords[1]),
                min(EvoThermalSensor.THERMAL_RESOLUTION[1], temp_roi_coords[2]),
                min(EvoThermalSensor.THERMAL_RESOLUTION[0], temp_roi_coords[3])
            )

            avg_temp_raw = 0.0
            if temp_roi_coords[0] < temp_roi_coords[2] and temp_roi_coords[1] < temp_roi_coords[3]: # Valid ROI
                avg_temp_raw = np.mean(thermal_frame[
                    temp_roi_coords[1]:temp_roi_coords[3], temp_roi_coords[0]:temp_roi_coords[2]
                ])

            avg_temp_corrected = avg_temp_raw + self.config.get('temperature_correction_offset', 0.0)
            temp_str_for_filename = f"{avg_temp_corrected:05.2f}".replace(".", "_") # Format for filenames

            # Filename generation
            timestamp_for_filename = current_timestamp.strftime('D_%Y-%m-%d_T_%H_%M_%S_%f')[:-3]
            line_id_prefix = self.config.get('lineID', 'LINE_UNKNOWN') # From config, or default
            base_filename_prefix = f"{line_id_prefix}_{timestamp_for_filename}_TEMP_{temp_str_for_filename}"

            generated_filenames = {
                'anomaly': f"{base_filename_prefix}_AD.jpg",
                'normal': f"{base_filename_prefix}_ND.jpg", # Potentially PNG for quality
                'diagnostic': f"{base_filename_prefix}_DC.jpg",
                'live': f"{self.config.get('line_name', 'SENSARAY_LIVE')}_live_pic_LP.jpg"
            }

            cropped_frame = self._crop_frame_based_on_config(frame) # Renamed for clarity

            # Frame statistics using misc_utils
            frame_stats_info_str, entropy_val, yellow_pct_val, brightness_val = misc_utils.get_frame_statistics(
                cropped_frame, temp_str_for_filename, self.config # Pass relevant config
            )

            # Assess frame quality based on statistics
            current_frame_quality = self._assess_frame_quality_against_thresholds(
                entropy_val, yellow_pct_val, brightness_val, avg_temp_corrected
            )

            return {
                'timestamp': current_timestamp, 'frame_counter': frame_counter,
                'temperature_avg': avg_temp_corrected, 'temperature_str': temp_str_for_filename,
                'filenames': generated_filenames, 'frame_info_str': frame_stats_info_str, # Renamed for clarity
                'entropy': entropy_val, 'yellow_percentage': yellow_pct_val, 'brightness': brightness_val,
                'frame_quality_assessment': current_frame_quality, # Renamed for clarity
                'anomaly_detected_flag': False, 'anomaly_pixel_count': 0, # Renamed for clarity
                'frame_cropped_data': cropped_frame # Renamed for clarity
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during frame {frame_counter} processing: {e}", exc_info=True)
            return None # Indicate failure


    def _crop_frame_based_on_config(self, frame: np.ndarray) -> np.ndarray: # Renamed
        """Crops the frame based on 'cfc_left_cutoff' and 'cfc_right_cutoff' from config."""
        left_cutoff = self.config.get('cfc_left_cutoff', 0)
        right_cutoff = self.config.get('cfc_right_cutoff', 0)
        frame_height, frame_width, _ = frame.shape

        # Validate cutoffs
        left = max(0, left_cutoff)
        right = max(0, right_cutoff)

        if (left + right) >= frame_width:
            self.logger.warning(f"Sum of left ({left}) and right ({right}) cutoffs exceeds or equals frame width ({frame_width}). Returning uncropped frame.")
            return frame
        return frame[:, left : frame_width - right, :]


    def _assess_frame_quality_against_thresholds(self, entropy, yellow_pct, brightness, temperature) -> Dict[str, bool]: # Renamed
        """Assesses frame quality by comparing statistics against configured thresholds."""
        # Use .get(key, default_value) for all threshold lookups to prevent KeyError
        is_entropy_ok = entropy > self.config.get('entropy_threshold', 5.0)
        is_yellow_ok = yellow_pct > self.config.get('hsv_yellow_percent_threshold', 20.0)
        is_brightness_ok = brightness > self.config.get('brightness_threshold', 0.6)
        # Temperature check: frame is good for processing if product is NOT hot (i.e., temp is low)
        is_temp_ok_for_processing = temperature < self.config.get('max_product_temperature', 25.0)

        quality_assessment = {
            'entropy_ok': is_entropy_ok,
            'yellow_ok': is_yellow_ok,
            'brightness_ok': is_brightness_ok,
            'temperature_ok_for_processing': is_temp_ok_for_processing # Renamed for clarity
        }
        quality_assessment['overall_good_for_processing'] = all(quality_assessment.values()) # Renamed
        return quality_assessment


    def check_for_contamination(self, frame_processing_result: Dict[str, Any], hsv_mask: np.ndarray) -> Dict[str, Any]:
        """Performs contamination check on the processed frame result."""
        # Validate input
        if not frame_processing_result or not frame_processing_result.get('frame_quality_assessment', {}).get('overall_good_for_processing'):
            self.logger.debug(f"Frame {frame_processing_result.get('frame_counter', 'N/A')}: Quality not good or invalid data. Skipping contamination check. Info: {frame_processing_result.get('frame_info_str', 'N/A')}")
            return frame_processing_result if frame_processing_result else {}

        try:
            cropped_frame_data = frame_processing_result['frame_cropped_data']
            # Call HSVCalculator method for the core logic
            anomaly_px_count, visual_cont_mask, _ = self.hsv_calculator.check_frame_for_contamination(cropped_frame_data, hsv_mask)

            self.stats['frames_processed_total'] += 1

            # Anomaly detection logic based on pixel count thresholds
            anomaly_threshold_min = self.config.get('hsv_pixel_threshold', 20)
            anomaly_threshold_max = self.config.get('hsv_pixel_threshold_max', 20000) # Upper bound to ignore massive changes

            if anomaly_threshold_min <= anomaly_px_count < anomaly_threshold_max:
                if self._should_record_anomaly_based_on_rate_limits(): # Renamed
                    frame_processing_result['anomaly_detected_flag'] = True
                    frame_processing_result['contamination_visualization_mask'] = visual_cont_mask # Store for saving
                    self._update_anomaly_statistics() # Renamed
                    self.logger.error(f"Frame {frame_processing_result['frame_counter']}: ANOMALY DETECTED - Contaminant pixels: {anomaly_px_count} (Threshold: {anomaly_threshold_min}) - Anomalies/hr: {self.stats['anomalies_this_hour_count']} - Info: {frame_processing_result['frame_info_str']}")
                else:
                    self.logger.warning(f"Frame {frame_processing_result['frame_counter']}: ANOMALY RATE LIMITED - Contaminant pixels: {anomaly_px_count} - Anomalies/hr: {self.stats['anomalies_this_hour_count']} - Last anomaly: {self.stats.get('last_anomaly_timestamp')}")
            else:
                # Log level depends on whether count is just below threshold or way off (e.g. > max_threshold)
                log_msg_level = logging.INFO if anomaly_px_count < anomaly_threshold_min else logging.WARNING
                self.logger.log(log_msg_level, f"Frame {frame_processing_result['frame_counter']}: {'CLEAN' if anomaly_px_count < anomaly_threshold_min else 'HIGH COUNT (ignored)'} - Contaminant pixels: {anomaly_px_count} (Threshold: {anomaly_threshold_min}, Max: {anomaly_threshold_max}) - Info: {frame_processing_result['frame_info_str']}")

            frame_processing_result['anomaly_pixel_count'] = anomaly_px_count
        except HSVProcessingError as e_hsv: # Catch specific errors from HSVCalculator
            self.logger.error(f"HSVProcessingError during contamination check for frame {frame_processing_result.get('frame_counter','N/A')}: {e_hsv}", exc_info=True)
        except Exception as e_contam:
            self.logger.error(f"Unexpected error in contamination check for frame {frame_processing_result.get('frame_counter','N/A')}: {e_contam}", exc_info=True)
        return frame_processing_result


    def _should_record_anomaly_based_on_rate_limits(self) -> bool: # Renamed
        """Checks if an anomaly should be recorded based on configured rate limits."""
        current_time = datetime.now()
        # Time since last anomaly
        if self.stats['last_anomaly_timestamp']:
            seconds_since_last = (current_time - self.stats['last_anomaly_timestamp']).total_seconds()
            if seconds_since_last < self.config.get('time_delay_between_anomalies_sec', 60): # Renamed key
                return False # Too soon since the last one

        # Anomalies per hour
        monotonic_now = time.monotonic()
        if (monotonic_now - self.stats['current_processing_hour_start_time']) > 3600: # Check if hour has passed
            self.stats['anomalies_this_hour_count'] = 0 # Reset counter
            self.stats['current_processing_hour_start_time'] = monotonic_now # Start new hour window

        if self.stats['anomalies_this_hour_count'] >= self.config.get('max_anomalies_per_hour_limit', 5): # Renamed key
            return False # Exceeded hourly limit
        return True

    def _update_anomaly_statistics(self) -> None: # Renamed
        """Updates statistics when a new anomaly is recorded."""
        self.stats['last_anomaly_timestamp'] = datetime.now()
        self.stats['anomalies_this_hour_count'] += 1
        self.stats['anomalies_detected_total'] += 1


    def save_frames(self, frame_result: Optional[Dict[str, Any]]) -> None:
        """Saves frames (live, diagnostic, anomaly) based on conditions and configuration."""
        if not frame_result: return # Nothing to save if processing failed

        try:
            base_work_dir = Path(self.config['work_dir'])
            upload_destination_dir = base_work_dir / "upload" # Renamed for clarity
            upload_destination_dir.mkdir(parents=True, exist_ok=True)

            frame_counter = frame_result['frame_counter']
            filenames = frame_result['filenames']

            # Save live picture (periodically) using misc_utils
            if frame_counter % self.config.get('live_pic_save_interval_frames', 150) == 0: # Approx every 5min at 1fps * 30 = 150
                live_display_frame = self._create_display_frame_with_overlay(frame_result) # Renamed
                misc_utils.write_image_to_dir(live_display_frame, upload_destination_dir, filenames['live'])
                self.logger.info(f"Saved live frame: {filenames['live']}")

            # Save diagnostic/temperature frames
            if self.config.get('write_temp_diagnostic_frames', False) and self._should_save_diagnostic_frame(frame_result): # Renamed key & func
                diagnostic_display_frame = self._create_display_frame_with_overlay(frame_result)
                misc_utils.write_image_to_dir(diagnostic_display_frame, upload_destination_dir, filenames['diagnostic'])
                self.logger.info(f"Saved diagnostic frame: {filenames['diagnostic']}")

            # Save anomaly frames
            if self.config.get('send_alerts_on_anomaly', False) and frame_result.get('anomaly_detected_flag', False): # Renamed keys
                original_cropped_img = frame_result['frame_cropped_data']
                visual_contamination_mask = frame_result.get('contamination_visualization_mask')

                if visual_contamination_mask is not None and visual_contamination_mask.shape == original_cropped_img.shape:
                    # Blend original with mask using misc_utils
                    masked_visualization = misc_utils.blend_frame_with_mask(original_cropped_img, visual_contamination_mask)
                    # Concatenate original and masked images vertically for comparison
                    combined_anomaly_image = cv2.vconcat([original_cropped_img, masked_visualization])
                    save_status = misc_utils.write_image_to_dir(combined_anomaly_image, upload_destination_dir, filenames['anomaly'])
                else: # Fallback if mask is missing or mismatched
                     self.logger.warning(f"Contamination mask issue for anomaly {filenames['anomaly']}. Saving only original cropped image.")
                     save_status = misc_utils.write_image_to_dir(original_cropped_img, upload_destination_dir, filenames['anomaly'])

                if save_status: self.logger.error(f"ANOMALY IMAGE SAVED: {filenames['anomaly']}")
                else: self.logger.error(f"FAILED TO SAVE ANOMALY IMAGE (check logs for details): {filenames['anomaly']}")

            # Save raw frames for analysis (get_pics mode)
            if self.config.get('save_raw_analysis_frames', False) and (frame_counter % self.config.get('raw_analysis_frame_interval',1000) == 0): # Renamed keys
                analysis_pics_dir = base_work_dir / "analysis_pics" # Renamed for clarity
                analysis_pics_dir.mkdir(parents=True, exist_ok=True)
                misc_utils.write_image_to_dir(frame_result['frame_cropped_data'], analysis_pics_dir, filenames['normal'], format_type='PNG') # Save as PNG for quality
                self.logger.info(f"Saved raw analysis PNG frame: {filenames['normal']}")

        except DiskSpaceError: # Catch specific error from misc_utils
            self.logger.critical("DiskSpaceError: Insufficient disk space. Frame saving operations suspended. Please free up disk space.")
            # Optionally, disable further saving attempts for a period to avoid repeated errors/checks
            # self.config['save_frames_enabled'] = False # Example dynamic config change
        except Exception as e:
            self.logger.error(f"Unexpected error during frame saving: {e}", exc_info=True)


    def _should_save_diagnostic_frame(self, frame_result: Dict[str, Any]) -> bool: # Renamed
        if self.config.get('sensaray_type') == 'TRIAL_MODE': # Check type from config
            return frame_result['frame_counter'] % self.config.get('trial_diag_frame_interval', 10) == 0
        return frame_result['frame_counter'] % self.config.get('prod_diag_frame_interval', 120) == 0


    def _create_display_frame_with_overlay(self, frame_result: Dict[str, Any]) -> np.ndarray: # Renamed
        """Creates a frame with informational overlays for display or saving as live/diagnostic."""
        # Use the cropped frame data as the base
        base_frame_for_display = frame_result['frame_cropped_data'].copy()
        try:
            # Resize using misc_utils. Scale percentage from config or default.
            display_scale_percent = self.config.get('display_image_scale_percent', 20) # e.g. 20%
            display_frame = misc_utils.resize_image(base_frame_for_display, display_scale_percent)

            # Add overlay lines for cutoffs, scaled to the display image size
            h, w, _ = display_frame.shape
            left_cutoff_scaled = int(self.config.get('cfc_left_cutoff', 0) * (display_scale_percent / 100.0))
            right_cutoff_scaled = int(self.config.get('cfc_right_cutoff', 0) * (display_scale_percent / 100.0))

            # Draw lines (e.g., cyan color, thickness 1)
            cv2.line(display_frame, (left_cutoff_scaled, 0), (left_cutoff_scaled, h), (255, 255, 0), 1)
            cv2.line(display_frame, (w - right_cutoff_scaled, 0), (w - right_cutoff_scaled, h), (255, 255, 0), 1)

            # Construct text for overlay
            # Use frame_info_str which already has most details
            overlay_text = f"{frame_result['filenames']['diagnostic']} | {frame_result['frame_info_str']}"
            # Add text overlay using misc_utils
            display_frame_with_text = misc_utils.add_text_overlay(display_frame, overlay_text)
            return display_frame_with_text
        except Exception as e:
            self.logger.error(f"Error creating display frame with overlay: {e}", exc_info=True)
            # Fallback to returning the (resized, if successful) base frame without text/lines
            return misc_utils.resize_image(base_frame_for_display, self.config.get('display_image_scale_percent', 20)) if base_frame_for_display is not None else np.zeros((100,100,3), dtype=np.uint8)


    def display_frames(self, frame_result: Optional[Dict[str, Any]]) -> None:
        """Displays frames if 'show_pics' is enabled and a display is available."""
        if not frame_result or not self.config.get('show_pics', False) or not self.display_available:
            return # Skip if no result, display disabled in config, or no display environment

        try:
            # Display frame periodically based on 'show_every_x_pic' config
            if frame_result['frame_counter'] % self.config.get('show_every_x_pic', 5) == 0:
                frame_for_display = self._create_display_frame_with_overlay(frame_result)
                window_title = f"Sensaray Output: {self.config.get('sensaray_type','N/A')} - Line: {self.config.get('line_name', 'N/A')}"

                cv2.imshow(window_title, frame_for_display)
                cv2.waitKey(1) # Crucial for OpenCV GUI event processing
        except cv2.error as e_cv_gui: # Catch specific OpenCV GUI errors
            self.logger.warning(f"OpenCV GUI error during display_frames (imshow/waitKey): {e_cv_gui}. Disabling 'show_pics' to prevent further errors.")
            self.config['show_pics'] = False # Dynamically disable to avoid repeated errors
            self.display_available = False # Assume display context is lost or problematic
        except Exception as e_disp:
            self.logger.error(f"Unexpected error in display_frames: {e_disp}", exc_info=True)
            self.config['show_pics'] = False # Also disable on other unexpected errors


    def update_hsv_mask(self, frame_result: Optional[Dict[str, Any]], current_hsv_mask: np.ndarray) -> np.ndarray:
        """Updates the HSV mask if 'build_hsv_mask' is enabled and frame quality is good."""
        if not frame_result or not self.config.get('build_hsv_mask', False):
            return current_hsv_mask # Return current mask if no result or building is disabled
        if not frame_result.get('frame_quality_assessment',{}).get('overall_good_for_processing'):
            return current_hsv_mask # Don't update mask with poor quality frames

        try:
            # Call HSVCalculator method to update the mask
            updated_mask, num_new_pixels = self.hsv_calculator.build_hsv_mask_from_frame(
                frame_result['frame_cropped_data'], current_hsv_mask
            )
            if num_new_pixels > 0:
                # Recalculate active pixel percentage for logging
                active_pixels = np.sum(updated_mask > 0)
                total_mask_entries = np.prod(updated_mask.shape) # H*S*V
                percentage_active = round((active_pixels / total_mask_entries) * 100, 2) if total_mask_entries > 0 else 0
                self.logger.info(f"Frame {frame_result['frame_counter']}: HSV MASK UPDATED - {num_new_pixels} new HSV combinations added. Total active combinations: {percentage_active}%")
            return updated_mask
        except HSVProcessingError as e_hsv_build: # Catch specific errors from HSVCalculator
            self.logger.error(f"HSVProcessingError during HSV mask update for frame {frame_result.get('frame_counter','N/A')}: {e_hsv_build}", exc_info=True)
        except Exception as e_update_mask:
            self.logger.error(f"Unexpected error updating HSV mask for frame {frame_result.get('frame_counter','N/A')}: {e_update_mask}", exc_info=True)
        return current_hsv_mask # Return original mask if update fails


    def run_processing_loop(self) -> None:
        """Main loop for continuous frame processing."""
        try:
            # Load initial HSV mask
            hsv_mask_current, initial_mask_percentage = self.hsv_calculator.load_hsv_mask()
            if self.config.get('white_belt_mode', False): # Specific mode for white belts
                hsv_mask_current = self.hsv_calculator.remove_blue_values_from_mask(hsv_mask_current)
            self.logger.info(f"Initial HSV mask loaded. Active pixel combinations: {initial_mask_percentage}%.")

            loop_start_time_monotonic = time.monotonic()
            # Configurable loop duration, default to 1 hour (3600 seconds)
            configured_loop_duration_sec = self.config.get('processing_loop_duration_seconds', 3600)
            frame_count = 0 # Renamed from frame_counter for local scope
            self.is_running = True # Set flag to start the loop

            self.logger.info(f"Starting main processing loop. Configured duration: {configured_loop_duration_sec} seconds.")

            while self.is_running and (time.monotonic() - loop_start_time_monotonic) < configured_loop_duration_sec:
                frame_count += 1
                current_iteration_start_time = time.perf_counter() # For FPS calculation/control

                try:
                    cam_frame, thermal_frame_data = self.read_sensor_data()
                    if cam_frame is None: # Critical if no camera frame
                        self.logger.warning(f"Frame {frame_count}: Failed to acquire valid camera frame. Skipping this iteration.")
                        time.sleep(0.5) # Brief pause before trying next frame
                        continue

                    # Process the acquired frame data
                    processed_frame_info = self.process_frame(cam_frame, thermal_frame_data, frame_count)
                    if not processed_frame_info:
                        self.logger.warning(f"Frame {frame_count}: Frame processing failed. Skipping further actions for this frame.")
                        continue # Skip to next frame if processing returns None

                    # Contamination check (if enabled)
                    if self.config.get('enable_contamination_check', True): # Renamed key
                        processed_frame_info = self.check_for_contamination(processed_frame_info, hsv_mask_current)

                    # Update HSV mask (if enabled)
                    hsv_mask_current = self.update_hsv_mask(processed_frame_info, hsv_mask_current)

                    # Save relevant frames (anomaly, live, diagnostic)
                    self.save_frames(processed_frame_info)

                    # Display frames (if enabled and display available)
                    self.display_frames(processed_frame_info)

                    # Periodic logging
                    if frame_count % self.config.get('log_status_interval_frames', 50) == 0:
                        self.logger.info(f"Iter {frame_count}: {processed_frame_info.get('frame_info_str','N/A')}")

                    # Less frequent detailed status and config update check
                    if frame_count % self.config.get('config_update_check_interval_frames', 1000) == 0:
                        self.logger.info(f"Status @ Frame {frame_count}: Total Anomalies: {self.stats['anomalies_detected_total']}, Anomalies/Hour: {self.stats['anomalies_this_hour_count']}")
                        try:
                            # Update system configuration from server using config_utils
                            updated_config = config_utils.update_settings_from_server(self.config) # Pass current config
                            if updated_config and id(updated_config) != id(self.config): # Check if a new object was returned
                                self.config = updated_config
                                self.logger.info("System configuration successfully updated from server.")
                        except ConfigurationError as e_cfg_upd_conf:
                            self.logger.warning(f"Configuration update from server failed (ConfigurationError): {e_cfg_upd_conf}")
                        except Exception as e_cfg_upd_gen:
                             self.logger.warning(f"Unexpected error during server configuration update: {e_cfg_upd_gen}", exc_info=True)

                except KeyboardInterrupt: # Handle Ctrl+C within the loop
                    self.logger.info("User interruption (KeyboardInterrupt) detected within processing loop.")
                    self.is_running = False # Signal to stop all loops
                except Exception as e_loop_iter: # Catch errors for a single iteration
                    self.logger.error(f"Unhandled error in processing loop iteration {frame_count}: {e_loop_iter}", exc_info=True)
                    time.sleep(1.0) # Pause briefly after an iteration error before continuing

                # Optional: Frame rate control
                target_fps = self.config.get('target_fps', 2) # Default target FPS
                if target_fps > 0:
                    elapsed_iteration_time = time.perf_counter() - current_iteration_start_time
                    sleep_time = max(0, (1.0 / target_fps) - elapsed_iteration_time)
                    if sleep_time > 0: time.sleep(sleep_time)

            # --- End of while self.is_running loop ---

            # After loop finishes (either by duration or is_running flag)
            if self.config.get('build_hsv_mask', False) and hsv_mask_current is not None:
                self.logger.info("Processing loop concluded. Finalizing and saving HSV mask as 'build_hsv_mask' is enabled.")
                try:
                    # Apply morphological fill to each S-V plane of the 3D HSV mask
                    for h_plane_index in range(hsv_mask_current.shape[0]): # Iterate H dimension (0-179)
                        sv_plane = hsv_mask_current[h_plane_index, :, :]
                        filled_sv_plane = self.hsv_calculator.apply_morphological_fill_to_sv_slice(sv_plane)
                        hsv_mask_current[h_plane_index, :, :] = filled_sv_plane

                    final_active_count = np.sum(hsv_mask_current > 0)
                    total_mask_size = np.prod(hsv_mask_current.shape)
                    final_fill_percentage = round((final_active_count / total_mask_size) * 100, 2) if total_mask_size > 0 else 0.0
                    self.hsv_calculator.save_hsv_mask(hsv_mask_current, final_fill_percentage)
                    self.logger.info(f"Final HSV mask (after morphological fill) saved. Active combinations: {final_fill_percentage}%.")
                except Exception as e_final_save:
                    self.logger.error(f"Error during final save of built HSV mask: {e_final_save}", exc_info=True)

            self.logger.info(f"Main processing loop finished. Total frames processed in this session: {frame_count}.")

        except Exception as e_outer_loop: # Catch critical errors that might break the setup of the loop
            self.logger.critical(f"A critical error occurred outside the main iteration logic of run_processing_loop: {e_outer_loop}", exc_info=True)
            # Consider re-raising if this indicates a non-recoverable state for the calling `run` method
            raise


    def cleanup_sensors(self) -> None:
        """Cleans up (disconnects, releases) all initialized sensors and resources."""
        self.logger.info("Initiating cleanup of all sensors and system resources...")
        try:
            if self.thermal_sensor: # Check if instance exists
                self.thermal_sensor.disconnect()
                self.logger.info("EvoThermalSensor disconnected.")

            if hasattr(self, 'camera_system') and self.camera_system: # Check if camera_system was initialized
                self.camera_system.cleanup_devices()
                self.logger.info("ArenaCamera system cleaned up.")

            # Destroy any remaining OpenCV windows if display was available
            if self.display_available:
                try:
                    cv2.destroyAllWindows()
                    self.logger.debug("OpenCV windows destroyed via destroyAllWindows().")
                except cv2.error as e_cv_destroy: # Catch if GUI context already lost
                    self.logger.debug(f"cv2.destroyAllWindows() encountered a minor error (ignorable): {e_cv_destroy}")

            self.logger.info("All sensor and resource cleanup procedures completed.")
        except Exception as e_cleanup:
            self.logger.error(f"An error occurred during sensor cleanup: {e_cleanup}", exc_info=True)


    def run(self) -> None:
        """The main execution method for the Sensaray system."""
        # Max runtime for the entire application, e.g., 2 days. Configurable.
        max_system_duration = timedelta(days=self.config.get('max_total_system_runtime_days', 2))
        application_start_time = datetime.now()

        try:
            self.logger.info(f"Sensaray System run method started. System Type: {self.config.get('sensaray_type', 'UNKNOWN')}. Max total runtime: {max_system_duration}.")

            # This loop allows the system to run for 'max_system_duration', potentially restarting the processing loop.
            while (datetime.now() - application_start_time) < max_system_duration:
                # Check `is_running` at the start of each major cycle.
                # It could be set to False by a signal handler at any time.
                if not self.is_running and (datetime.now() - application_start_time).total_seconds() > 5 : # Avoid immediate exit if is_running was false from init
                     self.logger.info("System run loop detected is_running is false. Exiting normally.")
                     break # Exit if shutdown was signaled

                self.is_running = True # Set true for the current operational cycle
                try:
                    if not self.initialize_sensors():
                        self.logger.error("Sensor initialization failed. Retrying after a 60-second delay.")
                        time.sleep(60)
                        continue # Skip to the next iteration of the outer while loop to retry initialization

                    self.run_processing_loop() # This is the main operational loop for frame processing

                except KeyboardInterrupt: # User pressed Ctrl+C
                    self.logger.info("KeyboardInterrupt caught in main run method. Initiating shutdown.")
                    self.is_running = False # Signal all loops to stop
                    break # Exit the outer while loop
                except Exception as e_operational_cycle:
                    self.logger.error(f"An unhandled error occurred in an operational cycle (initialize or processing loop): {e_operational_cycle}", exc_info=True)
                    self.logger.info("Attempting to cleanup sensors and will retry the cycle after a 30-second delay.")
                    # self.cleanup_sensors() # Cleanup before retrying
                    # time.sleep(30) # Wait before the next attempt of the outer while loop
                    # For critical errors, may be better to exit than retry indefinitely.
                    # For now, let's make it exit to avoid error loops.
                    self.is_running = False; break

                # If run_processing_loop finishes (e.g., due to its own duration),
                # this outer loop will restart it, unless self.is_running is set to False.
                if self.is_running:
                    self.logger.info("Current processing loop finished. Cleaning up sensors before potentially restarting cycle.")
                    self.cleanup_sensors()
                    time.sleep(self.config.get('delay_between_processing_cycles_sec', 5)) # Brief pause
                else:
                    self.logger.info("Shutdown signaled during processing. Exiting main run method.")
                    break

            # After the main while loop (either max runtime reached or is_running became false)
            if (datetime.now() - application_start_time) >= max_system_duration:
                 self.logger.info(f"System maximum runtime of {max_system_duration} has been reached.")
                 if self.config.get('auto_reboot_at_max_runtime', False): # More specific config key
                    self.logger.info("Initiating system reboot as configured due to max runtime.")
                    misc_utils.reboot_windows() # Use refactored misc_utils
            else:
                self.logger.info("System run completed before reaching maximum runtime.")

        except Exception as e_critical_run:
            self.logger.critical(f"A critical unrecoverable error occurred in SensaraySystem.run(): {e_critical_run}", exc_info=True)
            # Consider specific exit codes or further actions for critical failures
        finally:
            self.logger.info("SensaraySystem.run() is concluding. Performing final cleanup procedures...")
            self.cleanup_sensors() # Ensure cleanup is always called
            self.logger.info("Sensaray system shutdown finalized.")


class TrialSensaraySystem(SensaraySystem):
    """A subclass of SensaraySystem for trial mode using test images."""

    def __init__(self, config_file_path_override: Optional[str] = None):
        """Initializes the TrialSensaraySystem."""
        super().__init__(config_file_path_override=config_file_path_override) # Call base class __init__
        self.test_image_paths: List[Path] = [] # Explicitly typed list of Path objects
        self._load_and_verify_test_images() # Renamed method
        self.current_image_idx: int = 0 # Renamed for clarity
        self.images_processed_in_cycle: int = 0 # Renamed for clarity
        self.total_test_images_in_set: int = len(self.test_image_paths) # Renamed

    def _load_and_verify_test_images(self): # Renamed
        """Loads test images from the directory specified in config, creates samples if dir is empty."""
        test_img_dir_path_str = self.config.get('test_image_dir', './data/trial_images/') # Default path
        test_img_dir = Path(test_img_dir_path_str)

        if not test_img_dir.is_dir(): # If directory doesn't exist or is not a dir
            self.logger.info(f"Test image directory '{test_img_dir.resolve()}' not found. Creating it and sample images.")
            self._create_sample_test_images(test_img_dir) # Create directory and sample images

        # Load images (jpg and png)
        self.test_image_paths = sorted(list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png')))
        self.total_test_images_in_set = len(self.test_image_paths)

        if not self.test_image_paths:
            self.logger.warning(f"No test images (*.jpg, *.png) found in {test_img_dir.resolve()}. Creating sample images now.")
            self._create_sample_test_images(test_img_dir) # Attempt to create again if still none
            self.test_image_paths = sorted(list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png')))
            self.total_test_images_in_set = len(self.test_image_paths)
            if not self.test_image_paths:
                 self.logger.error(f"CRITICAL: No test images available in {test_img_dir.resolve()} even after sample creation attempt. Trial mode cannot function.")
                 # Consider raising an error or setting a state that prevents run()

    def _create_sample_test_images(self, test_dir: Path): # Renamed
        """Creates a few sample images in the specified directory for trial mode."""
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Creating 3 sample PNG images in {test_dir.resolve()} for trial mode...")
            for i in range(3): # Create a few distinct sample images
                # Image dimensions (e.g., 640x480)
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                # Add some features for visual inspection and processing
                text = f"Sample Image {i+1}"
                color = [(0,255,255), (255,0,255), (255,255,0)][i % 3] # Yellow, Magenta, Cyan text
                cv2.putText(img, text, (30, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                # Add a "defect" to one of the images for testing anomaly detection
                if i == 1:
                    cv2.circle(img, (320, 240), 50, (0,0,255), -1) # Red circle as a defect

                cv2.imwrite(str(test_dir / f'sample_trial_image_{i:02d}.png'), img) # Save as PNG
            self.logger.info(f"Successfully created sample images in {test_dir.resolve()}.")
        except Exception as e_create_sample:
            self.logger.error(f"Failed to create sample test images in {test_dir.resolve()}: {e_create_sample}", exc_info=True)

    def initialize_sensors(self) -> bool: # Overridden from base class
        self.logger.info("TRIAL MODE: Mock sensor initialization. No actual hardware access.")
        self.devices = {'mock_camera_active': True} # Minimal placeholder
        return True # Always successful for trial mode

    def read_sensor_data(self) -> Tuple[Optional[np.ndarray], np.ndarray]: # Overridden
        """Reads the next test image in a cycle for trial mode."""
        if not self.test_image_paths: # No images loaded
            self.logger.error("TRIAL MODE: No test images are available. Stopping run.")
            self.is_running = False # Signal to stop the main loop
            return None, np.zeros(EvoThermalSensor.THERMAL_RESOLUTION, dtype=np.float32)

        # Check if a full cycle of images has been processed
        if self.images_processed_in_cycle >= self.total_test_images_in_set and self.total_test_images_in_set > 0 :
            self._prompt_user_after_image_cycle() # Renamed method
            if not self.is_running: # User chose to quit
                 return None, np.zeros(EvoThermalSensor.THERMAL_RESOLUTION, dtype=np.float32)
            # Reset counters for the new cycle if user continues
            self.current_image_idx = 0
            self.images_processed_in_cycle = 0

        current_image_file_path = self.test_image_paths[self.current_image_idx]
        frame_data = cv2.imread(str(current_image_file_path))

        if frame_data is None: # Failed to load image
            self.logger.error(f"TRIAL MODE: Failed to load test image: {current_image_file_path}. Skipping.")
            # Advance index and attempt to load next image recursively, or handle error
            self.current_image_idx = (self.current_image_idx + 1) % self.total_test_images_in_set
            self.images_processed_in_cycle +=1 # Count as processed attempt
            # To prevent infinite recursion if all images are bad, add a check or limit retries.
            # For now, just try next one.
            if self.images_processed_in_cycle < self.total_test_images_in_set * 2 : # Avoid too deep recursion
                return self.read_sensor_data()
            else: # Exhausted retry attempts for bad images
                self.logger.error("TRIAL_MODE: Multiple image load failures. Stopping.")
                self.is_running = False
                return None, np.zeros(EvoThermalSensor.THERMAL_RESOLUTION, dtype=np.float32)

        # Successfully loaded an image
        self.current_image_idx = (self.current_image_idx + 1) % self.total_test_images_in_set
        self.images_processed_in_cycle += 1

        # Mock thermal data using a configurable temperature
        mock_thermal_value = self.config.get('trial_mode_mock_temperature', 22.0)
        thermal_frame_data = np.full(EvoThermalSensor.THERMAL_RESOLUTION, mock_thermal_value, dtype=np.float32)

        self.logger.debug(f"TRIAL MODE: Loaded image '{current_image_file_path.name}' ({self.images_processed_in_cycle}/{self.total_test_images_in_set} this cycle).")
        # Configurable delay to simulate real-world camera frame rate
        time.sleep(self.config.get('trial_mode_frame_processing_delay_sec', 0.1))
        return frame_data, thermal_frame_data

    def _prompt_user_after_image_cycle(self): # Renamed
        """Waits for user input after a full cycle of test images has been processed."""
        self.logger.info(f"TRIAL MODE: Finished processing all {self.total_test_images_in_set} test images in the current cycle.")
        # Provide a summary of the cycle
        summary_message = [
            "\n" + "="*70,
            f"TRIAL MODE: IMAGE CYCLE COMPLETED ({self.total_test_images_in_set} images processed this cycle).",
            f"  Total anomalies detected this cycle: {self.stats['anomalies_detected_total']}", # This should be reset per cycle
            f"  Check output directory '{self.config.get('work_dir')}/upload/' for any generated anomaly images.",
            "="*70 + "\n"
        ]
        print("\n".join(summary_message)) # Print to console for user

        try:
            # Reset cycle-specific stats before waiting for input
            self.stats['anomalies_detected_total'] = 0 # Reset for the next cycle
            self.stats['anomalies_this_hour_count'] = 0
            self.stats['last_anomaly_timestamp'] = None
            self.stats['current_processing_hour_start_time'] = time.monotonic()


            user_choice = input("Enter 'q' (or 'quit') to exit, or press ENTER to run another cycle with test images: ").strip().lower()
            if user_choice in ['q', 'quit']:
                self.logger.info("TRIAL MODE: User chose to quit after completing a cycle.")
                self.is_running = False # Signal main loop to terminate
        except KeyboardInterrupt: # Handle Ctrl+C during input prompt
            self.logger.info("TRIAL MODE: User interrupted input prompt (Ctrl+C). Quitting.")
            self.is_running = False

    def cleanup_sensors(self) -> None: # Overridden
        self.logger.info("TRIAL MODE: Mock sensor cleanup. No hardware actions needed.")
        # Any trial-specific cleanup can go here (e.g., deleting temp files if any were created)


def main():
    """Main entry point for the Sensaray application."""
    # Setup a very basic logger for messages before the main application logger is configured
    # This helps catch errors during early initialization (e.g., config loading)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)
    bootstrap_logger = logging.getLogger('SensarayBootstrap') # Distinct logger name

    try:
        # Load configuration using the refactored config_utils module
        # load_yaml_config will use its internal default path if no argument is given
        app_config = config_utils.load_yaml_config()

        # Determine system mode (Trial or Production) based on configuration
        # Use .get() for safety, defaulting to False or a non-trial type if key is missing
        is_trial_mode_active = app_config.get('sensaray_type') == 'TRIAL_MODE'

        sensaray_system_instance: SensaraySystem # Type hint for the system instance

        if is_trial_mode_active:
            bootstrap_logger.info("Configuration indicates TRIAL_MODE. Initializing TrialSensaraySystem.")
            # Pass the loaded config to the constructor if it's designed to accept it,
            # otherwise, it will load its own config (current design).
            # To pass: TrialSensaraySystem(config_override=app_config)
            sensaray_system_instance = TrialSensaraySystem()
        else:
            bootstrap_logger.info("Configuration indicates PRODUCTION_MODE. Initializing SensaraySystem.")
            sensaray_system_instance = SensaraySystem()

        # Start the main run loop of the initialized system instance
        sensaray_system_instance.run()

    except ConfigurationError as e_config_fatal:
        bootstrap_logger.critical(f"FATAL: Configuration Error during startup: {e_config_fatal}. System cannot start.", exc_info=True)
        sys.exit(2) # Specific exit code for configuration-related fatal errors
    except Exception as e_main_fatal:
        bootstrap_logger.critical(f"FATAL: An unhandled critical error occurred in main(): {e_main_fatal}", exc_info=True)
        sys.exit(1) # General fatal error exit code
    finally:
        bootstrap_logger.info("Sensaray application main function is concluding.")


if __name__ == '__main__':
    main()
