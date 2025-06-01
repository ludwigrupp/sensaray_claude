"""
Sensaray Industrial Vision Inspection System v2.0
Improved version with better error handling, performance, and maintainability.
"""

import time
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import logging
from pathlib import Path
import signal
import os
import platform

from class_Arena import ArenaCamera, CameraError
from class_EvoThermal import EvoThermalSensor, ThermalSensorError
from class_HSVcalc import HSVCalculator, HSVProcessingError
from class_Misc import Misc, DiskSpaceError
from class_Settings import Settings, ConfigurationError

def setup_opencv_environment():
    """Setup OpenCV for headless/Parallels environments."""
    display_available = True
    
    try:
        # Check if we're in a headless environment
        if platform.system() == "Linux":
            display_available = bool(os.environ.get('DISPLAY'))
        elif platform.system() == "Darwin":  # macOS
            # Check if we're in a GUI session
            import subprocess
            try:
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                display_available = 'WindowServer' in result.stdout
            except:
                display_available = False
        elif platform.system() == "Windows":
            # In Parallels, Windows might not have proper GUI support
            try:
                # Test if we can create a window
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.namedWindow('test', cv2.WINDOW_NORMAL)
                cv2.destroyWindow('test')
            except:
                display_available = False
    except Exception as e:
        logging.warning(f"Could not detect display availability: {e}")
        display_available = False
    
    if not display_available:
        logging.warning("No display detected - disabling OpenCV GUI features")
        # Set environment variable to disable GUI
        os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
        
    return display_available

class SensaraySystem:
    """Main Sensaray system controller with improved architecture."""
    
    def __init__(self, config_file: Optional[str] = None):
        # Check display availability first
        self.display_available = setup_opencv_environment()
        
        # Initialize logging first
        self.logger = Misc.setup_logger('sensaray_main')
        
        if not self.display_available:
            self.logger.warning("Running in headless mode - GUI features disabled")
        
        try:
            # Load configuration
            if config_file:
                # Custom config loading would go here
                pass
            self.config = Settings.load_yaml_config()
            
            # Disable show_pics if no display available
            if not self.display_available:
                self.config['show_pics'] = False
                self.logger.info("Automatically disabled show_pics due to headless environment")
            
            # Initialize components
            self.camera_system = ArenaCamera(self.config)
            self.hsv_calculator = HSVCalculator(self.config)
            self.thermal_sensor = None
            self.devices = {}
            
            # System state
            self.is_running = False
            self.stats = {
                'frames_processed': 0,
                'anomalies_detected': 0,
                'last_anomaly_time': None,
                'anomalies_per_hour': 0
            }
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info(f"Sensaray system initialized: {self.config['sensaray_type']}")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False

    def initialize_sensors(self) -> bool:
        """Initialize all sensors based on system type."""
        try:
            sensor_type = self.config['sensaray_type']
            buffer_size = self.config.get('buffersize', 10)
            
            self.logger.info(f"Initializing sensors for {sensor_type}")
            
            if sensor_type == 'BO3_400_noIR':
                device_0, device_1 = self.camera_system.initialize_dual_camera(buffer_size)
                self.devices = {'camera_0': device_0, 'camera_1': device_1}
                self.thermal_sensor = None
                
            elif sensor_type == 'BO2_400_EVO':
                device_0 = self.camera_system.initialize_single_camera(buffer_size)
                self.devices = {'camera_0': device_0}
                self.thermal_sensor = EvoThermalSensor()
                
            elif sensor_type == 'BO2_400_noIR':
                device_0 = self.camera_system.initialize_single_camera(buffer_size)
                self.devices = {'camera_0': device_0}
                self.thermal_sensor = None
                
            elif sensor_type == 'BO2_400_iTec':
                device_0 = self.camera_system.initialize_single_camera(buffer_size)
                self.devices = {'camera_0': device_0}
                # self.thermal_sensor = iTecIRSensor()  # Would need implementation
                self.thermal_sensor = None
                
            elif sensor_type == 'TRIAL_MODE':
                self.logger.info("Trial mode - no hardware initialization required")
                return True
                
            else:
                raise ConfigurationError(f"Unknown sensor type: {sensor_type}")
            
            # Warm up system
            self._warm_up_sensors()
            
            self.logger.info("All sensors initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Sensor initialization failed: {e}")
            self.cleanup_sensors()
            return False

    def _warm_up_sensors(self, warm_up_frames: int = 10) -> None:
        """Warm up sensors by reading initial frames."""
        self.logger.info(f"Warming up sensors with {warm_up_frames} frames")
        
        for i in range(warm_up_frames):
            try:
                frame, thermal_frame = self.read_sensor_data()
                if i % 5 == 0:
                    self.logger.debug(f"Warm-up frame {i + 1}/{warm_up_frames}")
            except Exception as e:
                self.logger.warning(f"Warm-up frame {i + 1} failed: {e}")

    def read_sensor_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read data from all sensors based on system type."""
        try:
            sensor_type = self.config['sensaray_type']
            
            if sensor_type == 'BO3_400_noIR':
                frame = self.camera_system.read_dual_camera_frame(
                    self.devices['camera_0'], 
                    self.devices['camera_1']
                )
                thermal_frame = np.zeros((32, 32), dtype=np.float32)
                
            elif sensor_type in ['BO2_400_EVO', 'BO2_400_noIR', 'BO2_400_iTec']:
                frame = self.camera_system.read_single_camera_frame(self.devices['camera_0'])
                
                if self.thermal_sensor:
                    if sensor_type == 'BO2_400_EVO':
                        thermal_frame = self.thermal_sensor.read_thermal_frame()
                    elif sensor_type == 'BO2_400_iTec':
                        # temp_value = self.thermal_sensor.read_temperature()
                        temp_value = 15.0  # Mock for now
                        thermal_frame = np.full((32, 32), temp_value, dtype=np.float32)
                    else:
                        thermal_frame = np.zeros((32, 32), dtype=np.float32)
                else:
                    thermal_frame = np.zeros((32, 32), dtype=np.float32)
                    
            elif sensor_type == 'TRIAL_MODE':
                # This will be overridden in trial subclass
                frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
                thermal_frame = np.zeros((32, 32), dtype=np.float32)
                
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")
            
            return frame, thermal_frame
            
        except Exception as e:
            self.logger.error(f"Error reading sensor data: {e}")
            raise

    def process_frame(
        self, 
        frame: np.ndarray, 
        thermal_frame: np.ndarray, 
        frame_counter: int
    ) -> Dict[str, Any]:
        """Process a single frame and return results."""
        try:
            timestamp = datetime.now()
            
            # Calculate average temperature
            temp_region = (
                self.config.get('cft_left_cutoff', 0),
                0,
                31 - self.config.get('cft_right_cutoff', 0),
                31
            )
            avg_temp = np.mean(thermal_frame[
                temp_region[1]:temp_region[3], 
                temp_region[0]:temp_region[2]
            ]) + self.config.get('temperature_correction', 0)
            
            temp_str = f"{avg_temp:05.2f}".replace(".", "_")
            
            # Generate filenames
            timestamp_str = (
                f"{self.config.get('lineID', 'LINE_1')}_"
                f"{timestamp.strftime('D_%Y-%m-%d_T_%H_%M_%S_%f')[:-3]}_"
                f"TEMP_{temp_str}"
            )
            
            filenames = {
                'anomaly': f"{timestamp_str}_AD.jpg",
                'normal': f"{timestamp_str}_ND.jpg",
                'diagnostic': f"{timestamp_str}_DC.jpg",
                'live': f"{self.config.get('line_name', 'TRIAL')}_live_pic_LP.jpg"
            }
            
            # Process frame for contamination
            frame_cropped = self._crop_frame(frame)
            
            # Calculate frame statistics
            frame_info, entropy, yellow_pct, brightness = Misc.get_frame_statistics(
                frame_cropped, temp_str, self.config
            )
            
            # Check if frame is suitable for processing
            frame_quality = self._assess_frame_quality(
                entropy, yellow_pct, brightness, avg_temp
            )
            
            result = {
                'timestamp': timestamp,
                'frame_counter': frame_counter,
                'temperature': avg_temp,
                'temperature_str': temp_str,
                'filenames': filenames,
                'frame_info': frame_info,
                'entropy': entropy,
                'yellow_percentage': yellow_pct,
                'brightness': brightness,
                'frame_quality': frame_quality,
                'anomaly_detected': False,
                'anomaly_count': 0,
                'frame_cropped': frame_cropped
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_counter}: {e}")
            raise

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame based on configuration."""
        left_cutoff = self.config.get('cfc_left_cutoff', 0)
        right_cutoff = self.config.get('cfc_right_cutoff', 0)
        
        return frame[:, left_cutoff:frame.shape[1] - right_cutoff, :]

    def _assess_frame_quality(
        self, 
        entropy: float, 
        yellow_pct: float, 
        brightness: float, 
        temperature: float
    ) -> Dict[str, bool]:
        """Assess frame quality against thresholds."""
        # Define default values as constants for clarity
        DEFAULT_MAX_PRODUCT_TEMP = 25.0  # Higher default to be more permissive
        
        max_temp = self.config.get('max_product_temperature', DEFAULT_MAX_PRODUCT_TEMP)
        
        return {
            'entropy_ok': entropy > self.config.get('entropy_threshold', 5.0),
            'yellow_ok': yellow_pct > self.config.get('hsv_yellow_percent_threshold', 20),
            'brightness_ok': brightness > self.config.get('brightness_threshold', 0.6),
            # Temperature logic: process frames when no hot product is present
            'temperature_ok': temperature < max_temp,
            'overall_good': (
                entropy > self.config.get('entropy_threshold', 5.0) and
                yellow_pct > self.config.get('hsv_yellow_percent_threshold', 20) and
                brightness > self.config.get('brightness_threshold', 0.6) and
                temperature < max_temp
            )
        }

    def check_for_contamination(
        self, 
        frame_result: Dict[str, Any], 
        hsv_mask: np.ndarray
    ) -> Dict[str, Any]:
        """Check frame for contamination using HSV analysis."""
        try:
            if not frame_result['frame_quality']['overall_good']:
                self.logger.warning(f"Frame {frame_result['frame_counter']}: POOR QUALITY - {frame_result['frame_info']}")
                return frame_result
            
            frame_cropped = frame_result['frame_cropped']
            
            # Perform contamination check
            anomaly_count, contamination_mask, hsv_values = (
                self.hsv_calculator.check_frame_for_contamination(
                    frame_cropped, hsv_mask
                )
            )
            
            # Update statistics
            self.stats['frames_processed'] += 1
            
            # Check if anomaly threshold is exceeded
            threshold = self.config.get('hsv_pixel_threshold', 20)
            max_threshold = self.config.get('hsv_pixel_threshold_max', 20000)
            
            if threshold < anomaly_count < max_threshold:
                # Check rate limiting
                if self._should_record_anomaly():
                    frame_result['anomaly_detected'] = True
                    frame_result['contamination_mask'] = contamination_mask
                    self._record_anomaly()
                    
                    self.logger.error(f"Frame {frame_result['frame_counter']}: ANOMALY DETECTED - HSV no match: {anomaly_count} (threshold: {threshold}) - AD/h: {self.stats['anomalies_per_hour']} - {frame_result['frame_info']}")
                else:
                    self.logger.warning(f"Frame {frame_result['frame_counter']}: ANOMALY RATE LIMITED - HSV no match: {anomaly_count} - AD/h: {self.stats['anomalies_per_hour']} - time delay: {(frame_result['timestamp'] - self.stats['last_anomaly_time']).total_seconds() if self.stats['last_anomaly_time'] else 'N/A'}s < {self.config.get('time_delay_anomaly', 60)}s")
            else:
                self.logger.info(f"Frame {frame_result['frame_counter']}: CLEAN - HSV no match: {anomaly_count}/{threshold} - maxloop: {max(anomaly_count, getattr(self, '_max_anomaly_count', 0))} - {frame_result['frame_info']}")
                
                # Track maximum anomaly count seen
                self._max_anomaly_count = max(getattr(self, '_max_anomaly_count', 0), anomaly_count)
            
            frame_result['anomaly_count'] = anomaly_count
            return frame_result
            
        except Exception as e:
            self.logger.error(f"Error in contamination check: {e}")
            return frame_result

    def _should_record_anomaly(self) -> bool:
        """Check if anomaly should be recorded based on rate limiting."""
        now = datetime.now()
        
        # Check time delay
        if self.stats['last_anomaly_time']:
            time_since_last = now - self.stats['last_anomaly_time']
            if time_since_last.total_seconds() < self.config.get('time_delay_anomaly', 60):
                return False
        
        # Check hourly limit
        if self.stats['anomalies_per_hour'] >= self.config.get('max_anomalies_per_hour', 5):
            return False
        
        return True

    def _record_anomaly(self) -> None:
        """Record anomaly in statistics."""
        now = datetime.now()
        
        # Reset hourly counter if needed
        if (self.stats['last_anomaly_time'] and 
            (now - self.stats['last_anomaly_time']).total_seconds() > 3600):
            self.stats['anomalies_per_hour'] = 0
        
        self.stats['last_anomaly_time'] = now
        self.stats['anomalies_per_hour'] += 1
        self.stats['anomalies_detected'] += 1

    def save_frames(self, frame_result: Dict[str, Any]) -> None:
        """Save frames based on configuration and detection results."""
        try:
            work_dir = Path(self.config['work_dir'])
            
            # Ensure upload directory exists
            upload_dir = work_dir / "upload"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Save live picture periodically
            if frame_result['frame_counter'] % (self.config.get('show_every_x_pic', 5) * 30) == 0:
                live_frame = self._create_display_frame(frame_result)
                Misc.write_image_to_dir(
                    live_frame,
                    upload_dir,
                    frame_result['filenames']['live']
                )
                self.logger.info(f"Saved live frame: {frame_result['filenames']['live']}")
            
            # Save temperature frames
            if self.config.get('write_temperatures', False):
                if self._should_save_temperature_frame(frame_result):
                    display_frame = self._create_display_frame(frame_result)
                    Misc.write_image_to_dir(
                        display_frame,
                        upload_dir,
                        frame_result['filenames']['diagnostic']
                    )
                    self.logger.info(f"Saved temperature frame: {frame_result['filenames']['diagnostic']}")
            
            # Save anomaly frames - THIS IS THE KEY PART!
            if (self.config.get('send_alerts', False) and 
                frame_result.get('anomaly_detected', False) and 
                'contamination_mask' in frame_result):
                
                # Create combined frame (original + masked)
                original = frame_result['frame_cropped']
                masked = Misc.blend_frame_with_mask(
                    original, frame_result['contamination_mask']
                )
                combined = cv2.vconcat([original, masked])
                
                # Save the anomaly image
                success = Misc.write_image_to_dir(
                    combined,
                    upload_dir,
                    frame_result['filenames']['anomaly']
                )
                
                if success:
                    self.logger.error(f"ANOMALY IMAGE SAVED: {frame_result['filenames']['anomaly']}")
                else:
                    self.logger.error(f"FAILED TO SAVE ANOMALY IMAGE: {frame_result['filenames']['anomaly']}")
            
            # Save frames for analysis (if enabled)
            if self.config.get('get_pics', False):
                if frame_result['frame_counter'] % 1000 == 0:
                    get_pics_dir = work_dir / "get_pics"
                    get_pics_dir.mkdir(parents=True, exist_ok=True)
                    Misc.write_image_to_dir(
                        frame_result['frame_cropped'],
                        get_pics_dir,
                        frame_result['filenames']['normal'],
                        'PNG'
                    )
                    self.logger.info(f"Saved analysis frame: {frame_result['filenames']['normal']}")
                    
        except DiskSpaceError:
            self.logger.error("Insufficient disk space for frame saving")
        except Exception as e:
            self.logger.error(f"Error saving frames: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _should_save_temperature_frame(self, frame_result: Dict[str, Any]) -> bool:
        """Determine if temperature frame should be saved."""
        # In trial mode, save more frequently for testing
        if self.config.get('trial_mode', False):
            # Save every 10 frames in trial mode
            return frame_result['frame_counter'] % 10 == 0
        else:
            # Production mode - save every 2 minutes (120 frames at ~1 FPS)
            return frame_result['frame_counter'] % 120 == 0

    def _create_display_frame(self, frame_result: Dict[str, Any]) -> np.ndarray:
        """Create frame with overlay information for display."""
        try:
            frame = frame_result['frame_cropped'].copy()
            
            # Resize for display
            display_frame = Misc.resize_image(frame, 20)
            
            # Add overlay lines
            left_cutoff = int(self.config.get('cfc_left_cutoff', 0) * 0.2)
            right_cutoff = int(self.config.get('cfc_right_cutoff', 0) * 0.2)
            height = display_frame.shape[0]
            width = display_frame.shape[1]
            
            cv2.line(display_frame, (left_cutoff, 0), (left_cutoff, height), (255, 255, 255), 3)
            cv2.line(display_frame, (width - right_cutoff, 0), (width - right_cutoff, height), (255, 255, 255), 3)
            
            # Add text overlay
            text = f"{frame_result['filenames']['diagnostic']} {frame_result['frame_info']}"
            display_frame = Misc.add_text_overlay(display_frame, text)
            
            return display_frame
            
        except Exception as e:
            self.logger.error(f"Error creating display frame: {e}")
            return frame_result['frame_cropped']

    def display_frames(self, frame_result: Dict[str, Any]) -> None:
        """Display frames if enabled in configuration."""
        if not self.config.get('show_pics', False) or not self.display_available:
            return
            
        try:
            if frame_result['frame_counter'] % self.config.get('show_every_x_pic', 5) == 0:
                display_frame = self._create_display_frame(frame_result)
                
                window_title = f"Sensaray {self.config['sensaray_type']} - {self.config.get('line_name', 'TRIAL')}"
                
                try:
                    cv2.imshow(window_title, display_frame)
                    cv2.waitKey(1)
                except cv2.error as e:
                    self.logger.warning(f"OpenCV display error: {e}")
                    # Save frame instead of displaying
                    save_path = Path(self.config['work_dir']) / "logs" / f"display_frame_{frame_result['frame_counter']}.jpg"
                    save_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(save_path), display_frame)
                    # Disable further display attempts
                    self.config['show_pics'] = False
                    self.display_available = False
                
        except Exception as e:
            self.logger.error(f"Error in display_frames: {e}")
            # Disable display on any error
            self.config['show_pics'] = False

    def update_hsv_mask(
        self, 
        frame_result: Dict[str, Any], 
        hsv_mask: np.ndarray
    ) -> np.ndarray:
        """Update HSV mask if building is enabled."""
        if not self.config.get('build_hsv_mask', False):
            return hsv_mask
            
        if not frame_result['frame_quality']['overall_good']:
            return hsv_mask
        
        try:
            updated_mask, new_pixels = self.hsv_calculator.build_hsv_mask_from_frame(
                frame_result['frame_cropped'], hsv_mask
            )
            
            if new_pixels > 0:
                pixel_count = np.sum(updated_mask > 0)
                total_pixels = 180 * 256 * 256
                percentage = round((pixel_count / total_pixels) * 100, 2)
                
                self.logger.info(f"Frame {frame_result['frame_counter']}: HSV MASK UPDATED - {new_pixels} new pixels added - total: {pixel_count} ({percentage}% coverage)")
            
            return updated_mask
            
        except Exception as e:
            self.logger.error(f"Error updating HSV mask: {e}")
            return hsv_mask
            
        except Exception as e:
            self.logger.error(f"Error updating HSV mask: {e}")
            return hsv_mask

    def run_processing_loop(self) -> None:
        """Main processing loop with improved error handling."""
        try:
            # Load HSV mask
            hsv_mask, mask_percentage = self.hsv_calculator.load_hsv_mask()
            
            # Apply blue value removal for white belt systems
            if self.config.get('white_belt', False):
                hsv_mask = self.hsv_calculator.remove_blue_values(hsv_mask)
            
            self.logger.info(f"HSV mask loaded: {mask_percentage}% pixels active")
            
            # Processing loop variables
            loop_start_time = time.time()
            loop_duration = self.config.get('looptime', 3600)
            frame_counter = 0
            
            self.is_running = True
            self.logger.info("Starting main processing loop")
            
            while self.is_running and (time.time() - loop_start_time) < loop_duration:
                try:
                    frame_counter += 1
                    
                    # Read sensor data
                    frame, thermal_frame = self.read_sensor_data()
                    
                    # Process frame
                    frame_result = self.process_frame(frame, thermal_frame, frame_counter)
                    
                    # Check for contamination
                    if self.config.get('check_for_contamination', True):
                        frame_result = self.check_for_contamination(frame_result, hsv_mask)
                    
                    # Update HSV mask
                    hsv_mask = self.update_hsv_mask(frame_result, hsv_mask)
                    
                    # Save frames
                    self.save_frames(frame_result)
                    
                    # Display frames
                    self.display_frames(frame_result)
                    
                    # Periodic logging and status output
                    if frame_counter % 50 == 0 or frame_counter <= 10:
                        self.logger.info(f'{frame_counter} {frame_result["frame_info"]}')
                        
                    # Additional status every 1000 frames
                    if frame_counter % 1000 == 0:
                        self.logger.info(f"Processed {frame_counter} frames - Total anomalies: {self.stats['anomalies_detected']}, Rate: {self.stats['anomalies_per_hour']}/hour")
                    
                    # Update configuration periodically
                    if frame_counter % 10000 == 0:
                        try:
                            new_config = Settings.update_settings_from_server()
                            if new_config and new_config != self.config:
                                self.config = new_config
                                self.logger.info("Configuration updated from server")
                        except Exception as e:
                            self.logger.warning(f"Config update failed: {e}")
                    
                except KeyboardInterrupt:
                    self.logger.info("Processing interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in processing loop iteration {frame_counter}: {e}")
                    continue
            
            # Save HSV mask if building was enabled
            if self.config.get('build_hsv_mask', False):
                try:
                    # Apply morphological operations
                    for h in range(hsv_mask.shape[0]):
                        hsv_mask[h] = self.hsv_calculator.apply_morphological_fill(hsv_mask[h])
                    
                    pixel_count = np.sum(hsv_mask > 0)
                    total_pixels = 180 * 256 * 256
                    final_percentage = round((pixel_count / total_pixels) * 100, 2)
                    
                    self.hsv_calculator.save_hsv_mask(hsv_mask, final_percentage)
                    self.logger.info("HSV mask saved after processing loop")
                    
                except Exception as e:
                    self.logger.error(f"Error saving HSV mask: {e}")
            
            self.logger.info(f"Processing loop completed: {frame_counter} frames processed")
            
        except Exception as e:
            self.logger.error(f"Critical error in processing loop: {e}")
            raise

    def cleanup_sensors(self) -> None:
        """Clean up all sensors and resources."""
        try:
            self.logger.info("Cleaning up sensors and resources")
            
            if self.thermal_sensor:
                self.thermal_sensor.disconnect()
            
            self.camera_system.cleanup_devices()
            
            # Safely destroy OpenCV windows
            if self.display_available:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass  # Ignore errors when destroying windows
            
            self.logger.info("Sensor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during sensor cleanup: {e}")

    def run(self) -> None:
        """Main system run method with comprehensive error handling."""
        start_time = datetime.now()
        max_runtime_days = 2
        
        try:
            self.logger.info(f"Starting Sensaray system: {self.config['sensaray_type']}")
            
            while datetime.now() < start_time + timedelta(days=max_runtime_days):
                try:
                    # Initialize sensors
                    if not self.initialize_sensors():
                        self.logger.error("Sensor initialization failed, retrying in 60 seconds")
                        time.sleep(60)
                        continue
                    
                    # Run main processing loop
                    self.run_processing_loop()
                    
                    # Cleanup for reinitialization
                    self.cleanup_sensors()
                    
                    if not self.is_running:  # Graceful shutdown requested
                        break
                        
                except KeyboardInterrupt:
                    self.logger.info("System shutdown requested by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    self.cleanup_sensors()
                    time.sleep(30)  # Wait before retry
            
            self.logger.info("System runtime limit reached, initiating reboot")
            
        except Exception as e:
            self.logger.critical(f"Critical system error: {e}")
            raise
        finally:
            self.cleanup_sensors()
            
            # Optional system reboot
            if self.config.get('auto_reboot', False):
                self.logger.info("Initiating system reboot")
                Misc.reboot_windows()


class TrialSensaraySystem(SensaraySystem):
    """Trial version that works without hardware and waits for input after processing all samples."""
    
    def __init__(self):
        super().__init__()
        self.test_images = self._load_test_images()
        self.current_image_index = 0
        self.images_processed_count = 0
        self.total_test_images = len(self.test_images)
        self.cycle_completed = False
    
    def _load_test_images(self):
        """Load test images from directory."""
        test_dir = Path(self.config.get('test_image_dir', './data/test_images/'))
        if not test_dir.exists():
            # Create sample test images
            self._create_sample_images(test_dir)
        
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        return sorted(image_files)
    
    def _create_sample_images(self, test_dir):
        """Create sample images for trial."""
        test_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Creating sample test images in {test_dir}")
        
        # Create simple test images
        for i in range(10):
            # Create a colored image with some noise
            img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            
            # Add some colored regions
            cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 255), -1)  # Yellow
            cv2.circle(img, (500, 500), 100, (255, 0, 0), -1)  # Blue
            
            # Add some "contamination" to certain images
            if i % 3 == 0:
                cv2.rectangle(img, (700, 700), (800, 800), (0, 0, 255), -1)  # Red spot
                self.logger.debug(f"Added contamination to test image {i}")
            
            cv2.imwrite(str(test_dir / f'test_image_{i:03d}.jpg'), img)
        
        self.logger.info(f"Created {len(list(test_dir.glob('*.jpg')))} sample test images")
    
    def initialize_sensors(self):
        """Mock sensor initialization for trial."""
        self.logger.info("Initializing TRIAL MODE - no hardware required")
        self.devices = {'mock_camera': True}
        return True
    
    def read_sensor_data(self):
        """Read from test images with cycling and waiting for user input."""
        if not self.test_images:
            raise ValueError("No test images available")
        
        # Check if we've completed a full cycle
        if self.current_image_index == 0 and self.images_processed_count > 0:
            self.cycle_completed = True
            self._wait_for_user_input()
            
            if not self.is_running:
                return np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((32, 32), dtype=np.float32)
        
        # Load next image
        image_path = self.test_images[self.current_image_index]
        self.current_image_index = (self.current_image_index + 1) % len(self.test_images)
        self.images_processed_count += 1
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not load test image: {image_path}")
        
        # Mock thermal data
        mock_temp = self.config.get('mock_thermal_temp', 15.0)
        thermal_frame = np.full((32, 32), mock_temp, dtype=np.float32)
        
        # Add some delay to simulate real camera
        time.sleep(0.1)
        
        self.logger.debug(f"Loaded test image: {image_path.name} ({self.current_image_index}/{len(self.test_images)})")
        
        return frame, thermal_frame
    
    def _wait_for_user_input(self):
        """Wait for user input after completing all test images."""
        self.logger.info(f"Completed processing all {self.total_test_images} test images")
        
        # Create summary
        summary = [
            "=" * 60,
            f"CYCLE COMPLETED - Processed {self.total_test_images} test images",
            f"Total anomalies detected: {self.stats['anomalies_detected']}",
            f"Anomaly rate: {self.stats['anomalies_per_hour']}/hour",
            f"Check ./data/upload/ for saved images",
            "=" * 60,
        ]
        
        for line in summary:
            print(line)
        
        try:
            user_input = input("Press ENTER to process images again, or 'q' to quit: ")
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                self.logger.info("User requested exit")
                self.is_running = False
                return
            
            # Reset statistics for new cycle
            self.stats['anomalies_detected'] = 0
            self.stats['anomalies_per_hour'] = 0
            self.stats['last_anomaly_time'] = None
            self.images_processed_count = 0
            self.cycle_completed = False
            
            print("Starting new processing cycle...\n")
            self.logger.info("Starting new processing cycle")
            
        except KeyboardInterrupt:
            self.logger.info("User interrupted with Ctrl+C")
            self.is_running = False
    
    def cleanup_sensors(self):
        """Mock cleanup for trial."""
        self.logger.info("Trial mode cleanup completed")


def main():
    """Main entry point with trial mode support."""
    try:
        config = Settings.load_yaml_config()
        
        if config.get('trial_mode', False) or config.get('sensaray_type') == 'TRIAL_MODE':
            logger = logging.getLogger('sensaray_main')
            logger.info("Starting Sensaray in TRIAL MODE")
            sensaray = TrialSensaraySystem()
        else:
            logger = logging.getLogger('sensaray_main')
            logger.info("Starting Sensaray in PRODUCTION MODE")
            sensaray = SensaraySystem()
        
        sensaray.run()
        
    except ConfigurationError as e:
        logger = logging.getLogger('sensaray_main')
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger('sensaray_main')
        logger.critical(f"Critical system error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()