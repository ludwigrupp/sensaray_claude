import numpy as np
import time
from typing import Tuple, Optional, List, Dict, Any
import logging
# from contextlib import contextmanager # No longer needed as device_context is removed

try:
    from arena_api import enums
    from arena_api.buffer import BufferFactory
    from arena_api.system import system
    ARENA_AVAILABLE = True
except ImportError:
    logging.warning("Arena API not available - camera functionality disabled")
    system = None # Explicitly set system to None if import fails
    ARENA_AVAILABLE = False
    # Define enums and other necessary ArenaSDK parts as mock objects if needed for type hinting or basic structure
    # This helps in maintaining code structure even if the API is not available at development time.
    class MockEnums:
        PixelFormat_BGR8 = 'BGR8' # Example mock value
    enums = MockEnums()


import cv2

logger = logging.getLogger(__name__)

class CameraError(Exception):
    """Custom exception raised when camera-specific operations fail."""
    pass

class ArenaCamera:
    """
    Improved Arena camera interface focusing on initialization, configuration,
    frame capture, and cleanup, with enhanced error handling and clarity.
    Archived functions (device_context, destroy_all_devices) are removed.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ArenaCamera instance.
        Args:
            config: A dictionary containing configuration parameters for the camera(s).
        """
        self.config = config
        self.devices: List[Any] = [] # Type hint for list of devices
        self.is_initialized: bool = False

        # Camera configuration constants from config or defaults
        self.MAX_RETRY_ATTEMPTS: int = self.config.get('max_retry_attempts', 5) # More configurable
        self.RETRY_DELAY_SECONDS: int = self.config.get('retry_delay_seconds', 10)
        self.DEFAULT_PIXEL_FORMAT: str = self.config.get('pixel_format', 'BGR8') # enums.PixelFormat_BGR8 if using real enums

    # Note: The @contextmanager device_context was removed as per refactoring.
    # Users should call initialize and cleanup_devices explicitly.

    def create_devices_with_retry(self) -> List[Any]:
        """
        Attempts to create Arena camera devices with retry logic.
        Retries are logged, and errors are raised upon final failure.
        """
        if not ARENA_AVAILABLE or system is None:
            logger.error("Arena API or system object is not available. Cannot create devices.")
            raise CameraError("Arena API not available, cannot create devices.")

        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                # system.update_devices(100) # Optional: Force update of device list with a timeout
                created_devices = system.create_device() # Creates all available devices
                if created_devices:
                    logger.info(f"Successfully created {len(created_devices)} Arena device(s) on attempt {attempt + 1}.")
                    self.devices = created_devices
                    return self.devices # Return the list of created devices

                logger.warning(
                    f"Attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}: No Arena devices found. "
                    f"Waiting {self.RETRY_DELAY_SECONDS}s before next attempt."
                )
                time.sleep(self.RETRY_DELAY_SECONDS)

            except Exception as e: # Catching broader exceptions from Arena API
                logger.error(f"Device creation attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt >= self.MAX_RETRY_ATTEMPTS - 1: # Check if it's the last attempt
                    raise CameraError(f"Failed to create Arena devices after {self.MAX_RETRY_ATTEMPTS} attempts. Last error: {e}")
                time.sleep(self.RETRY_DELAY_SECONDS) # Wait before the next retry

        # This part is reached if loop completes without returning (i.e. no devices found in any attempt)
        raise CameraError(f"No Arena devices found after {self.MAX_RETRY_ATTEMPTS} attempts.")

    def _configure_stream_settings(self, stream_nodemap: Any) -> None:
        """
        Configures stream-specific settings for a device.
        Args:
            stream_nodemap: The stream nodemap object for the device.
        Raises:
            CameraError: If configuration of stream settings fails.
        """
        try:
            # Example: stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
            # Using .set_value as it's often safer or required by some APIs
            stream_nodemap.get_node('StreamAutoNegotiatePacketSize').value = True
            stream_nodemap.get_node("StreamBufferHandlingMode").value = enums.StreamBufferHandlingMode_NewestOnly if ARENA_AVAILABLE else "NewestOnly"
            stream_nodemap.get_node('StreamPacketResendEnable').value = True
            logger.debug("Stream settings configured: AutoPacketSize=True, BufferMode=NewestOnly, PacketResend=True.")
        except Exception as e: # More specific exception type if known from Arena API
            logger.error(f"Error configuring stream settings: {e}", exc_info=True)
            raise CameraError(f"Stream configuration failed: {e}")

    def _configure_camera_node(
        self,
        device: Any, # Pass the whole device
        camera_id: int # 0 for single/first, 1 for second in dual, etc.
    ) -> None:
        """
        Configures a specific camera device node with parameters from the config.
        Args:
            device: The Arena device object to configure.
            camera_id: An identifier (e.g., 0 or 1) to fetch specific settings from config.
        Raises:
            CameraError: If configuration fails or essential nodes are missing.
        """
        nodemap = device.nodemap
        stream_nodemap = device.tl_stream_nodemap
        try:
            self._configure_stream_settings(stream_nodemap)

            cam_prefix = f"cam{camera_id}"
            cam_config_keys = {
                'width': f"{cam_prefix}_width", 'height': f"{cam_prefix}_height",
                'offset_x': f"{cam_prefix}_offset_x", 'offset_y': f"{cam_prefix}_offset_y",
                'gain_auto': f"{cam_prefix}_gain_auto", 'gain': f"{cam_prefix}_gain",
                'exposure_auto': f"{cam_prefix}_exposure_auto", 'exposure_time': f"{cam_prefix}_exposure_time"
            }

            # Check for missing essential configuration keys for this camera_id
            missing_cfg_keys = [key for key, conf_key_name in cam_config_keys.items() if conf_key_name not in self.config and not (key in ['gain', 'exposure_time'] and self.config.get(cam_config_keys[key+'_auto']) != "Off")]
            if missing_cfg_keys:
                raise CameraError(f"Missing essential configuration items for {cam_prefix}: {', '.join(missing_cfg_keys)}")

            # Configure camera parameters
            nodes_to_set = {
                'Width': self.config[cam_config_keys['width']],
                'Height': self.config[cam_config_keys['height']],
                'OffsetX': self.config[cam_config_keys['offset_x']],
                'OffsetY': self.config[cam_config_keys['offset_y']],
                'PixelFormat': self.DEFAULT_PIXEL_FORMAT
            }
            for node_name, value in nodes_to_set.items():
                nodemap.get_node(node_name).value = value

            # Gain settings
            gain_auto_mode = self.config[cam_config_keys['gain_auto']]
            nodemap.get_node('GainAuto').value = gain_auto_mode
            if gain_auto_mode == "Off":
                nodemap.get_node('Gain').value = float(self.config[cam_config_keys['gain']])

            # Exposure settings
            exposure_auto_mode = self.config[cam_config_keys['exposure_auto']]
            nodemap.get_node('ExposureAuto').value = exposure_auto_mode
            if exposure_auto_mode == "Off":
                nodemap.get_node('ExposureTime').value = float(self.config[cam_config_keys['exposure_time']])

            logger.info(f"Camera device (ID {camera_id}, SN: {device.nodemap['DeviceSerialNumber'].value if ARENA_AVAILABLE else 'N/A'}) configured.")

        except Exception as e: # More specific exception type if known
            logger.error(f"Error configuring camera node for device ID {camera_id}: {e}", exc_info=True)
            raise CameraError(f"Camera (ID {camera_id}) node configuration failed: {e}")

    def _get_frame_from_device(self, device: Any, device_id_log: str) -> np.ndarray: # device_id_log for logging
        """
        Retrieves a frame from the specified device. Handles incomplete frames and requeues buffer.
        Args:
            device: The Arena device object.
            device_id_log: A string identifier for logging (e.g., "0" or "Primary").
        Returns:
            A NumPy array representing the frame.
        Raises:
            CameraError: If frame capture fails repeatedly or other errors occur.
        """
        try:
            buffer = device.get_buffer(timeout=2000) # Timeout in ms for getting a buffer

            if buffer is None: # Check if buffer is None (timeout)
                raise CameraError(f"Timeout getting buffer from device {device_id_log}")

            if hasattr(buffer, 'is_incomplete') and buffer.is_incomplete:
                logger.warning(f"Received incomplete frame (ID: {buffer.frame_id}) from device {device_id_log}. Requeuing.")
                device.requeue_buffer(buffer)
                buffer = device.get_buffer(timeout=2000) # Retry once
                if buffer is None:
                    raise CameraError(f"Timeout on retry getting buffer from device {device_id_log} after incomplete frame.")
                if hasattr(buffer, 'is_incomplete') and buffer.is_incomplete:
                    device.requeue_buffer(buffer) # Requeue even if it's still bad
                    raise CameraError(f"Consecutive incomplete frames from device {device_id_log}. Gave up.")

            # Assuming BGR8 format, 3 channels. Adjust if pixel format can vary.
            # This is a common way to reshape, ensure it matches expected buffer layout.
            frame_data = np.ctypeslib.as_array(
                buffer.pdata,
                shape=(buffer.height, buffer.width, int(buffer.bits_per_pixel / 8)) # Calculate components from BPP
            )

            frame_copy = frame_data.copy() # Important: copy data from buffer before requeueing
            device.requeue_buffer(buffer)
            return frame_copy

        except Exception as e: # More specific (e.g. ArenaSDK specific exception)
            logger.error(f"Error getting frame from device {device_id_log}: {e}", exc_info=True)
            raise CameraError(f"Frame capture failed for device {device_id_log}: {e}")

    def initialize_single_camera(self, buffer_count: Optional[int] = None) -> Any:
        """Initializes a single camera system (e.g., BO2 configuration)."""
        effective_buffer_count = buffer_count if buffer_count is not None else self.config.get('buffersize', 10)
        try:
            logger.info(f"Initializing single camera system with {effective_buffer_count} buffers.")
            if not self.create_devices_with_retry(): # Ensure devices are created
                 raise CameraError("Failed to create devices for single camera system.")

            if len(self.devices) < 1:
                raise CameraError("No cameras found/created for single camera system.")

            # Assume the first device in the list is the one to use
            primary_device = self.devices[0]
            self._configure_camera_node(primary_device, 0) # camera_id 0 for cam0_... config keys
            primary_device.start_stream(effective_buffer_count)

            # Warm-up: capture a few frames to ensure stability
            for i in range(5):
                _ = self._get_frame_from_device(primary_device, "Primary (Warm-up)")
                logger.debug(f"Warm-up frame {i+1}/5 captured from single camera.")

            logger.info("Single camera system initialized successfully.")
            self.is_initialized = True
            return primary_device # Return the initialized device object

        except Exception as e:
            logger.error(f"Single camera initialization failed: {e}", exc_info=True)
            self.cleanup_devices() # Attempt cleanup on failure
            raise CameraError(f"Single camera system initialization failed: {e}")


    def initialize_dual_camera(self, buffer_count: Optional[int] = None) -> Tuple[Any, Any]:
        """Initializes a dual camera system (e.g., BO3 configuration)."""
        effective_buffer_count = buffer_count if buffer_count is not None else self.config.get('buffersize', 10)
        try:
            logger.info(f"Initializing dual camera system with {effective_buffer_count} buffers per camera.")
            if not self.create_devices_with_retry():
                 raise CameraError("Failed to create devices for dual camera system.")

            if len(self.devices) < 2:
                raise CameraError(f"Insufficient cameras found/created for dual system: found {len(self.devices)}, need 2.")

            # Assign devices based on a convention or configuration (e.g., serial number)
            # For now, assuming order from create_device or specific serials if configured.
            # This example assumes devices[0] is cam1 and devices[1] is cam0 based on original logic.
            # It's safer to use serial numbers from config if available.
            device_cam0 = self.devices[1]
            device_cam1 = self.devices[0]
            # Add logic here to map devices based on serial numbers from config if they exist for cam0 and cam1

            self._configure_camera_node(device_cam0, 0) # cam_id=0 for "cam0_..." keys
            self._configure_camera_node(device_cam1, 1) # cam_id=1 for "cam1_..." keys

            device_cam0.start_stream(effective_buffer_count)
            device_cam1.start_stream(effective_buffer_count)

            # Warm-up for dual cameras
            for i in range(5):
                _, _ = self._get_frame_from_device(device_cam0, "Cam0 (Warm-up)"), self._get_frame_from_device(device_cam1, "Cam1 (Warm-up)")
                logger.debug(f"Warm-up frame {i+1}/5 captured from dual cameras.")

            logger.info("Dual camera system initialized successfully.")
            self.is_initialized = True
            return device_cam0, device_cam1 # Return tuple of device objects

        except Exception as e:
            logger.error(f"Dual camera initialization failed: {e}", exc_info=True)
            self.cleanup_devices()
            raise CameraError(f"Dual camera system initialization failed: {e}")

    def read_single_camera_frame(self, device: Any) -> np.ndarray:
        """Reads a frame from an initialized single camera."""
        if not self.is_initialized or device is None:
            raise CameraError("Single camera not initialized or device is None.")
        return self._get_frame_from_device(device, "SingleCam")

    def read_dual_camera_frame(self, device_0: Any, device_1: Any) -> Optional[np.ndarray]:
        """Reads frames from two initialized cameras and concatenates them horizontally."""
        if not self.is_initialized or device_0 is None or device_1 is None:
            raise CameraError("Dual camera system not initialized or one/both devices are None.")
        try:
            frame_0 = self._get_frame_from_device(device_0, "Cam0")
            frame_1 = self._get_frame_from_device(device_1, "Cam1")

            if frame_0 is None or frame_1 is None: # Should be caught by _get_frame_from_device raising error
                logger.error("Failed to get frame from one or both dual cameras.")
                return None

            # Ensure frames have same height for hconcat
            if frame_0.shape[0] != frame_1.shape[0]:
                # Resize one to match the other, or log error and return.
                # This indicates a configuration mismatch.
                logger.error(f"Frame height mismatch for dual cameras: Cam0 H={frame_0.shape[0]}, Cam1 H={frame_1.shape[0]}. Cannot concatenate.")
                # Consider resizing strategy or raising specific error
                # For now, attempt to resize frame_1 to frame_0's height
                target_height = frame_0.shape[0]
                aspect_ratio_1 = frame_1.shape[1] / frame_1.shape[0]
                new_width_1 = int(target_height * aspect_ratio_1)
                frame_1 = cv2.resize(frame_1, (new_width_1, target_height))
                logger.warning(f"Resized Cam1 frame to {frame_1.shape} to match Cam0 height for concatenation.")


            return cv2.hconcat([frame_0, frame_1])

        except CameraError as e: # Catch specific CameraErrors from _get_frame_from_device
            logger.error(f"Failed to read from dual camera system: {e}", exc_info=True)
            raise # Re-raise the caught CameraError
        except Exception as e: # Catch other unexpected errors like OpenCV issues
            logger.error(f"Unexpected error during dual camera frame read/concatenation: {e}", exc_info=True)
            raise CameraError(f"Dual camera read/concat failed: {e}")


    def cleanup_devices(self) -> None:
        """
        Stops streams and destroys all created Arena devices.
        This method should be called to free resources.
        """
        logger.info(f"Cleaning up Arena devices. Currently {len(self.devices)} device(s) tracked.")
        # Stop stream for each device first
        for i, device_obj in enumerate(self.devices):
            try:
                if device_obj and hasattr(device_obj, 'stop_stream'): # Check if device object is valid and has stop_stream
                    logger.debug(f"Stopping stream for device {i}...")
                    device_obj.stop_stream()
                    logger.info(f"Stream stopped for device {i}.")
            except Exception as e: # Arena API specific exception if available
                logger.error(f"Error stopping stream for device {i}: {e}", exc_info=True)

        # Destroy all devices known by the system instance
        if ARENA_AVAILABLE and system is not None and self.devices: # Only if system and devices exist
            try:
                system.destroy_device() # Destroys all devices created by this 'system' instance
                logger.info("All Arena devices destroyed via system.destroy_device().")
            except Exception as e: # Arena API specific exception
                logger.error(f"Error during system.destroy_device(): {e}", exc_info=True)

        self.devices = [] # Clear the list of devices
        self.is_initialized = False
        logger.info("ArenaCamera cleanup complete.")

    # Note: The staticmethod destroy_all_devices was removed as per refactoring.
    # Cleanup is now instance-based via cleanup_devices(). For a global cleanup,
    # one might need to manage 'system' object externally if multiple ArenaCamera instances are not desired.

# Note: The legacy 'Arena' class was removed as per refactoring.
# Ensure all usages are updated to ArenaCamera or archived functions.
