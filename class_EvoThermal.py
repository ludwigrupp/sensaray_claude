import numpy as np
import crcmod.predefined
from struct import unpack
import serial
import serial.tools.list_ports
import threading
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ThermalSensorError(Exception):
    """Raised when thermal sensor operations fail."""
    pass

class EvoThermalSensor:
    """Improved EvoThermal sensor interface with better error handling."""
    
    # Class constants
    VENDOR_ID = ":5740"
    BAUDRATE = 115200
    FRAME_HEADER = 13
    ACK_HEADER = 20
    FRAME_SIZE = 2068
    DATA_SIZE = 2064
    THERMAL_DIMENSIONS = (32, 32)
    KELVIN_TO_CELSIUS = 273.15
    DATA_SCALE_FACTOR = 10.0
    
    def __init__(self, auto_connect: bool = True):
        self.port: Optional[serial.Serial] = None
        self.serial_lock = threading.Lock()
        self.is_connected = False
        
        # Initialize CRC functions
        self.crc32 = crcmod.predefined.mkPredefinedCrcFun('crc-32-mpeg')
        self.crc8 = crcmod.predefined.mkPredefinedCrcFun('crc-8')
        
        # Command definitions
        self.activate_command = bytes([0x00, 0x52, 0x02, 0x01, 0xDF])
        self.deactivate_command = bytes([0x00, 0x52, 0x02, 0x00, 0xD8])
        
        if auto_connect:
            self.connect()

    def _find_thermal_port(self) -> Optional[str]:
        """Find EvoThermal device port."""
        try:
            ports = list(serial.tools.list_ports.comports())
            for port_info in ports:
                if self.VENDOR_ID in port_info.hwid:
                    logger.info(f"EvoThermal found on port {port_info.device}")
                    return port_info.device
            
            logger.warning("EvoThermal device not found")
            return None
            
        except Exception as e:
            logger.error(f"Error scanning for thermal sensor: {e}")
            return None

    def connect(self) -> bool:
        """Connect to thermal sensor with error handling."""
        try:
            port_name = self._find_thermal_port()
            if not port_name:
                logger.warning("No thermal sensor found")
                return False
            
            self.port = serial.Serial(
                port=port_name,
                baudrate=self.BAUDRATE,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1.0
            )
            
            if self.port.is_open:
                self.is_connected = True
                logger.info(f"Connected to thermal sensor on {port_name}")
                return True
            else:
                logger.error("Failed to open thermal sensor port")
                return False
                
        except serial.SerialException as e:
            logger.error(f"Serial connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to thermal sensor: {e}")
            return False

    def is_running(self) -> bool:
        """Check if thermal sensor is available and running."""
        if not self.is_connected or not self.port:
            return False
            
        try:
            # Quick check if port is still available
            port_name = self._find_thermal_port()
            return port_name == self.port.name
        except Exception:
            return False

    def _send_command(self, command: bytes) -> bool:
        """Send command to thermal sensor with proper error handling."""
        if not self.is_connected or not self.port:
            logger.warning("Thermal sensor not connected")
            return False
            
        try:
            with self.serial_lock:
                # Send command
                self.port.write(command)
                self.port.flush()
                
                # Read acknowledgment
                ack = self.port.read(1)
                if not ack:
                    logger.warning("No acknowledgment received")
                    return False
                
                # Wait for ACK header
                while len(ack) == 1 and ack[0] != self.ACK_HEADER:
                    ack = self.port.read(1)
                    if not ack:
                        logger.warning("Timeout waiting for ACK header")
                        return False
                
                # Read rest of ACK
                ack += self.port.read(3)
                if len(ack) != 4:
                    logger.warning("Incomplete ACK received")
                    return False
                
                # Verify CRC
                calculated_crc = self.crc8(ack[:3])
                if calculated_crc != ack[3]:
                    logger.warning("ACK CRC mismatch")
                    return False
                
                # Check ACK/NACK
                if ack[2] == 0:
                    return True
                else:
                    logger.warning("Command not acknowledged (NACK)")
                    return False
                    
        except serial.SerialTimeoutException:
            logger.error("Timeout during command transmission")
            return False
        except serial.SerialException as e:
            logger.error(f"Serial error during command: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending command: {e}")
            return False

    def read_thermal_frame(self) -> np.ndarray:
        """Read thermal frame with comprehensive error handling."""
        if not self.is_connected or not self.port:
            logger.warning("Thermal sensor not connected, returning empty frame")
            return np.zeros(self.THERMAL_DIMENSIONS, dtype=np.float32)
        
        try:
            # Activate sensor
            max_activation_attempts = 3
            for attempt in range(max_activation_attempts):
                if self._send_command(self.activate_command):
                    break
                if attempt == max_activation_attempts - 1:
                    logger.error("Failed to activate thermal sensor")
                    return np.zeros(self.THERMAL_DIMENSIONS, dtype=np.float32)
                time.sleep(0.1)
            
            # Read frame data
            frame_data = self._read_frame_data()
            if frame_data is None:
                return np.zeros(self.THERMAL_DIMENSIONS, dtype=np.float32)
            
            # Deactivate sensor
            self._send_command(self.deactivate_command)
            
            return frame_data
            
        except Exception as e:
            logger.error(f"Error reading thermal frame: {e}")
            return np.zeros(self.THERMAL_DIMENSIONS, dtype=np.float32)

    def _read_frame_data(self) -> Optional[np.ndarray]:
        """Read and validate frame data."""
        max_frame_attempts = 5
        
        for attempt in range(max_frame_attempts):
            try:
                with self.serial_lock:
                    # Look for frame header
                    header_bytes = self.port.read(2)
                    if len(header_bytes) != 2:
                        continue
                    
                    header = unpack('H', header_bytes)[0]
                    if header != self.FRAME_HEADER:
                        continue
                    
                    # Read frame data
                    data_bytes = self.port.read(self.FRAME_SIZE)
                    if len(data_bytes) != self.FRAME_SIZE:
                        logger.warning(f"Incomplete frame data: {len(data_bytes)}/{self.FRAME_SIZE}")
                        continue
                    
                    # Validate CRC
                    calculated_crc = self.crc32(data_bytes[:self.DATA_SIZE])
                    data_unpacked = unpack("H" * (self.FRAME_SIZE // 2), data_bytes)
                    
                    received_crc = ((data_unpacked[1032] & 0xFFFF) << 16) | (data_unpacked[1033] & 0xFFFF)
                    
                    if calculated_crc != received_crc:
                        logger.warning(f"CRC mismatch: calculated={calculated_crc:08x}, received={received_crc:08x}")
                        continue
                    
                    # Extract temperature data
                    temp_data = np.array(data_unpacked[:1024], dtype=np.float32)
                    temp_data = temp_data.reshape(self.THERMAL_DIMENSIONS)
                    
                    # Convert from decikelvin to Celsius
                    temp_data = (temp_data / self.DATA_SCALE_FACTOR) - self.KELVIN_TO_CELSIUS
                    
                    # Flush input buffer
                    self.port.reset_input_buffer()
                    
                    return temp_data
                    
            except Exception as e:
                logger.warning(f"Frame read attempt {attempt + 1} failed: {e}")
                if attempt == max_frame_attempts - 1:
                    logger.error("All frame read attempts failed")
        
        return None

    def get_average_temperature(self, region: Optional[Tuple[int, int, int, int]] = None) -> float:
        """Get average temperature from specified region or entire frame."""
        frame = self.read_thermal_frame()
        
        if region:
            x1, y1, x2, y2 = region
            # Validate region bounds
            x1 = max(0, min(x1, self.THERMAL_DIMENSIONS[1] - 1))
            x2 = max(x1 + 1, min(x2, self.THERMAL_DIMENSIONS[1]))
            y1 = max(0, min(y1, self.THERMAL_DIMENSIONS[0] - 1))
            y2 = max(y1 + 1, min(y2, self.THERMAL_DIMENSIONS[0]))
            
            region_data = frame[y1:y2, x1:x2]
        else:
            region_data = frame
        
        return float(np.mean(region_data))

    def disconnect(self) -> bool:
        """Disconnect from thermal sensor."""
        try:
            if self.is_connected and self.port:
                # Send deactivate command
                self._send_command(self.deactivate_command)
                
                # Close port
                self.port.close()
                self.port = None
                self.is_connected = False
                
                logger.info("Thermal sensor disconnected successfully")
                return True
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting thermal sensor: {e}")
            return False

    def __del__(self):
        """Cleanup on object destruction."""
        self.disconnect()


# Legacy compatibility class
class EvoThermal:
    """Legacy compatibility class for existing code."""
    
    def __init__(self):
        self.sensor = EvoThermalSensor()
        self.rounded_array = np.zeros((32, 32))

    def running(self) -> bool:
        """Legacy compatibility method."""
        return self.sensor.is_running()

    def get_thermals(self) -> np.ndarray:
        """Legacy compatibility method."""
        return self.sensor.read_thermal_frame()

    def get_frame(self) -> np.ndarray:
        """Legacy compatibility method."""
        frame = self.sensor.read_thermal_frame()
        self.rounded_array = np.round(frame, 0)
        return self.rounded_array

    def stop(self) -> bool:
        """Legacy compatibility method."""
        return self.sensor.disconnect()

    def send_command(self, command) -> bool:
        """Legacy compatibility method."""
        return self.sensor._send_command(command)

    def run(self):
        """Legacy compatibility method."""
        frame = self.get_thermals()
        self.rounded_array = np.round(frame, 0)
        # Note: update_GUI method was removed as it's not defined in original