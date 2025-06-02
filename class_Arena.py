import numpy as np
import time
from typing import Tuple, Optional, List, Dict, Any
import logging
from contextlib import contextmanager

try:
    from arena_api import enums
    from arena_api.buffer import BufferFactory
    from arena_api.system import system
    ARENA_AVAILABLE = True
except ImportError:
    logging.warning("Arena API not available - camera functionality disabled")
    system = None
    ARENA_AVAILABLE = False

import cv2

logger = logging.getLogger(__name__)

class CameraError(Exception):
    """Raised when camera operations fail."""
    pass

class ArenaCamera:
    """Improved Arena camera interface with better error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.devices = []
        self.is_initialized = False
        
        # Camera configuration constants
        self.MAX_RETRY_ATTEMPTS = 6
        self.RETRY_DELAY_SECONDS = 10
        self.DEFAULT_PIXEL_FORMAT = 'BGR8'

    @contextmanager
    def device_context(self):
        """Context manager for device lifecycle."""
        try:
            yield
        finally:
            self.cleanup_devices()

    def create_devices_with_retry(self) -> List:
        """Create devices with retry logic and proper error handling."""
        if not ARENA_AVAILABLE or system is None:
            raise CameraError("Arena API not available")
            
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                devices = system.create_device()
                if devices:
                    logger.info(f"Successfully created {len(devices)} device(s)")
                    self.devices = devices
                    return devices
                    
                # Wait before retry
                logger.warning(
                    f"Attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}: "
                    f"No devices found, waiting {self.RETRY_DELAY_SECONDS}s"
                )
                
                for second in range(self.RETRY_DELAY_SECONDS):
                    time.sleep(1)
                    if second % 5 == 0:  # Log every 5 seconds
                        logger.debug(f"Waiting... {second + 1}/{self.RETRY_DELAY_SECONDS}s")
                        
            except Exception as e:
                logger.error(f"Device creation attempt {attempt + 1} failed: {e}")
                if attempt == self.MAX_RETRY_ATTEMPTS - 1:
                    raise CameraError(f"Failed to create devices after {self.MAX_RETRY_ATTEMPTS} attempts")
                time.sleep(self.RETRY_DELAY_SECONDS)
        
        raise CameraError("No devices found after maximum retry attempts")

    def _configure_stream_settings(self, stream_nodemap) -> None:
        """Configure stream-specific settings."""
        try:
            stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
            stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
            stream_nodemap['StreamPacketResendEnable'].value = True
            logger.debug("Stream settings configured successfully")
        except Exception as e:
            logger.error(f"Error configuring stream settings: {e}")
            raise CameraError(f"Stream configuration failed: {e}")

    def _configure_camera_node(
        self, 
        nodemap, 
        stream_nodemap, 
        camera_id: int
    ) -> None:
        """Configure camera node with validation."""
        try:
            # Configure stream settings
            self._configure_stream_settings(stream_nodemap)
            
            # Get camera-specific config
            cam_prefix = f"cam{camera_id}"
            cam_config = {
                'width': self.config[f"{cam_prefix}_width"],
                'height': self.config[f"{cam_prefix}_height"],
                'offset_x': self.config[f"{cam_prefix}_offset_x"],
                'offset_y': self.config[f"{cam_prefix}_offset_y"],
                'gain_auto': self.config[f"{cam_prefix}_gain_auto"],
                'gain': self.config.get(f"{cam_prefix}_gain", 20.0),
                'exposure_auto': self.config[f"{cam_prefix}_exposure_auto"],
                'exposure_time': self.config.get(f"{cam_prefix}_exposure_time", 1000.0)
            }
            
            # Get required nodes
            node_names = [
                'Width', 'Height', 'PixelFormat', 'OffsetX', 'OffsetY',
                'GainAuto', 'Gain', 'ExposureAuto', 'ExposureTime'
            ]
            nodes = nodemap.get_node(node_names)
            
            # Validate nodes exist
            missing_nodes = [name for name in node_names if name not in nodes]
            if missing_nodes:
                raise CameraError(f"Missing camera nodes: {missing_nodes}")
            
            # Configure camera parameters
            nodes['Width'].value = cam_config['width']
            nodes['Height'].value = cam_config['height']
            nodes['OffsetX'].value = cam_config['offset_x']
            nodes['OffsetY'].value = cam_config['offset_y']
            nodes['PixelFormat'].value = self.DEFAULT_PIXEL_FORMAT
            
            # Configure gain
            nodes['GainAuto'].value = cam_config['gain_auto']
            if cam_config['gain_auto'] == "Off":
                nodes['Gain'].value = cam_config['gain']
            
            # Configure exposure
            nodes['ExposureAuto'].value = cam_config['exposure_auto']
            if cam_config['exposure_auto'] == "Off":
                nodes['ExposureTime'].value = cam_config['exposure_time']
            
            logger.info(
                f"Camera {camera_id} configured: "
                f"Resolution: {cam_config['width']}x{cam_config['height']}, "
                f"Offset: ({cam_config['offset_x']}, {cam_config['offset_y']}), "
                f"Gain: {cam_config['gain_auto']} ({cam_config['gain']}), "
                f"Exposure: {cam_config['exposure_auto']} ({cam_config['exposure_time']})"
            )
            
        except Exception as e:
            logger.error(f"Error configuring camera {camera_id}: {e}")
            raise CameraError(f"Camera {camera_id} configuration failed: {e}")

    def _get_frame_from_device(self, device, device_id: int) -> np.ndarray:
        """Get frame from device with error handling."""
        try:
            buffer = device.get_buffer()
            
            # Check for incomplete frames
            if hasattr(buffer, 'is_incomplete') and buffer.is_incomplete:
                logger.warning(f"Received incomplete frame from device {device_id}: {buffer.frame_id}")
                device.requeue_buffer(buffer)
                # Retry once
                buffer = device.get_buffer()
                if hasattr(buffer, 'is_incomplete') and buffer.is_incomplete:
                    device.requeue_buffer(buffer)
                    raise CameraError(f"Consecutive incomplete frames from device {device_id}")
            
            # Convert buffer to numpy array
            frame = np.ctypeslib.as_array(
                buffer.pdata,
                shape=(buffer.height, buffer.width, 3)
            ).reshape(buffer.height, buffer.width, 3)
            
            # Copy data to avoid buffer issues
            frame_copy = frame.copy()
            
            # Requeue buffer
            device.requeue_buffer(buffer)
            
            return frame_copy
            
        except Exception as e:
            logger.error(f"Error getting frame from device {device_id}: {e}")
            raise CameraError(f"Frame capture failed for device {device_id}: {e}")

    def initialize_single_camera(self, buffer_size: int) -> object:
        """Initialize single camera system (BO2)."""
        try:
            logger.info("Initializing single camera system")
            devices = self.create_devices_with_retry()
            
            if len(devices) < 1:
                raise CameraError("No cameras found for single camera system")
            
            device = devices[0]
            self._configure_camera_node(device.nodemap, device.tl_stream_nodemap, 0)
            device.start_stream(buffer_size)
            
            # Warm up camera
            for _ in range(10):
                self._get_frame_from_device(device, 0)
            
            logger.info("Single camera system initialized successfully")
            self.is_initialized = True
            return device
            
        except Exception as e:
            logger.error(f"Single camera initialization failed: {e}")
            self.cleanup_devices()
            raise CameraError(f"Single camera initialization failed: {e}")

    def initialize_dual_camera(self, buffer_size: int) -> Tuple[object, object]:
        """Initialize dual camera system (BO3)."""
        try:
            logger.info("Initializing dual camera system")
            devices = self.create_devices_with_retry()
            
            if len(devices) < 2:
                raise CameraError(f"Insufficient cameras for dual system: found {len(devices)}, need 2")
            
            # Configure devices (note: order might be swapped)
            device_0 = devices[1]  # Based on original code
            device_1 = devices[0]
            
            self._configure_camera_node(device_0.nodemap, device_0.tl_stream_nodemap, 0)
            self._configure_camera_node(device_1.nodemap, device_1.tl_stream_nodemap, 1)
            
            # Start streams
            device_0.start_stream(buffer_size)
            device_1.start_stream(buffer_size)
            
            # Warm up cameras
            for _ in range(10):
                self.read_dual_camera_frame(device_0, device_1)
            
            logger.info("Dual camera system initialized successfully")
            self.is_initialized = True
            return device_0, device_1
            
        except Exception as e:
            logger.error(f"Dual camera initialization failed: {e}")
            self.cleanup_devices()
            raise CameraError(f"Dual camera initialization failed: {e}")

    def read_single_camera_frame(self, device) -> np.ndarray:
        """Read frame from single camera with error handling."""
        return self._get_frame_from_device(device, 0)

    def read_dual_camera_frame(self, device_0, device_1) -> np.ndarray:
        """Read and concatenate frames from dual cameras."""
        try:
            frame_0 = self._get_frame_from_device(device_0, 0)
            frame_1 = self._get_frame_from_device(device_1, 1)
            
            # Concatenate horizontally
            combined_frame = cv2.hconcat([frame_0, frame_1])
            return combined_frame
            
        except Exception as e:
            logger.error(f"Dual camera frame read failed: {e}")
            raise CameraError(f"Dual camera frame read failed: {e}")

    def cleanup_devices(self) -> None:
        """Clean up camera devices and resources."""
        try:
            if ARENA_AVAILABLE and system and self.is_initialized:
                system.destroy_device()
                logger.info("Camera devices cleaned up successfully")
            self.is_initialized = False
            self.devices = []
        except Exception as e:
            logger.error(f"Error during device cleanup: {e}")

    @staticmethod
    def destroy_all_devices() -> None:
        """Static method to destroy all devices."""
        try:
            if ARENA_AVAILABLE and system:
                system.destroy_device()
                logger.info("All camera devices destroyed")
        except Exception as e:
            logger.error(f"Error destroying devices: {e}")


# Legacy compatibility functions for existing code
class Arena:
    """Legacy compatibility class."""
    
    @staticmethod
    def destroy_devices():
        ArenaCamera.destroy_all_devices()

    @staticmethod
    def create_devices_with_tries():
        """Legacy compatibility method."""
        camera = ArenaCamera({'buffersize': 10})
        return camera.create_devices_with_retry()

    @staticmethod
    def configure_node_0(nodemap, stream_nodemap):
        """Legacy method - use ArenaCamera instead."""
        logger.warning("Using legacy configure_node_0 - migrate to ArenaCamera")

    @staticmethod
    def configure_node_1(nodemap, stream_nodemap):
        """Legacy method - use ArenaCamera instead."""
        logger.warning("Using legacy configure_node_1 - migrate to ArenaCamera")

    @staticmethod
    def get_frame_0_BGR(device_0):
        """Legacy method - use ArenaCamera instead."""
        logger.warning("Using legacy get_frame_0_BGR - migrate to ArenaCamera")
        camera = ArenaCamera({})
        return camera._get_frame_from_device(device_0, 0)
    
    @staticmethod
    def get_frame_1_BGR(device_1):
        """Legacy method - use ArenaCamera instead."""
        logger.warning("Using legacy get_frame_1_BGR - migrate to ArenaCamera")
        camera = ArenaCamera({})
        return camera._get_frame_from_device(device_1, 1)

    @staticmethod
    def initialize_cams_BO3(buffernumber):
        """Legacy method - use ArenaCamera instead."""
        logger.warning("Using legacy initialize_cams_BO3 - migrate to ArenaCamera")
        config = {
            'buffersize': buffernumber,
            'cam0_width': 5472, 'cam0_height': 1000,
            'cam0_offset_x': 0, 'cam0_offset_y': 2000,
            'cam0_gain_auto': 'Off', 'cam0_gain': 20.0,
            'cam0_exposure_auto': 'Off', 'cam0_exposure_time': 1000.0,
            'cam1_width': 5472, 'cam1_height': 1000,
            'cam1_offset_x': 0, 'cam1_offset_y': 2000,
            'cam1_gain_auto': 'Off', 'cam1_gain': 20.0,
            'cam1_exposure_auto': 'Off', 'cam1_exposure_time': 1000.0
        }
        camera = ArenaCamera(config)
        return camera.initialize_dual_camera(buffernumber)

    @staticmethod
    def read_cams_BO3(device_0, device_1):
        """Legacy method - use ArenaCamera instead."""
        logger.warning("Using legacy read_cams_BO3 - migrate to ArenaCamera")
        camera = ArenaCamera({})
        return camera.read_dual_camera_frame(device_0, device_1)

    @staticmethod
    def initialize_cams_BO2(buffernumber):
        """Legacy method - use ArenaCamera instead."""
        logger.warning("Using legacy initialize_cams_BO2 - migrate to ArenaCamera")
        config = {
            'buffersize': buffernumber,
            'cam0_width': 5472, 'cam0_height': 1000,
            'cam0_offset_x': 0, 'cam0_offset_y': 2000,
            'cam0_gain_auto': 'Off', 'cam0_gain': 20.0,
            'cam0_exposure_auto': 'Off', 'cam0_exposure_time': 1000.0
        }
        camera = ArenaCamera(config)
        return camera.initialize_single_camera(buffernumber)

    @staticmethod
    def read_cams_BO2(device_0):
        """Legacy method - use ArenaCamera instead."""
        logger.warning("Using legacy read_cams_BO2 - migrate to ArenaCamera")
        camera = ArenaCamera({})
        return camera.read_single_camera_frame(device_0)