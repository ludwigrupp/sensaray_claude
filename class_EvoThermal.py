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
    """
    Improved EvoThermal sensor interface with better error handling and clarity.
    Archived functions (is_running, get_average_temperature) are removed from this class.
    """

    # Class constants
    VENDOR_ID = ":5740" # Used to identify the EvoThermal device by its hardware ID
    BAUDRATE = 115200   # Serial communication baud rate
    FRAME_HEADER_VALUE = 0x000D # Expected value for a data frame header (e.g., after unpack('<H', bytes))
                                # This might need adjustment based on exact protocol (e.g. if it's just one byte 0x0D)
    ACK_HEADER_VALUE = 0x14     # Expected value for an acknowledgment packet header (first byte)

    FRAME_PACKET_SIZE = 2068 # Total size of a data frame packet (header + data + crc)
    THERMAL_DATA_SIZE_BYTES = 2064 # Size of the actual thermal data payload in bytes (e.g., 32x32 pixels * 2 bytes/pixel)
    CRC_IN_FRAME_SIZE_BYTES = 4 # Size of CRC in the data frame (e.g., CRC32)

    THERMAL_RESOLUTION = (32, 32) # Native resolution of the thermal sensor
    KELVIN_TO_CELSIUS_OFFSET = 273.15 # Offset for Kelvin to Celsius conversion
    DECIKELVIN_SCALE_FACTOR = 10.0    # Sensor data is in deciKelvin, divide by this for Kelvin

    ACK_PACKET_SIZE = 4 # Expected size of an ACK packet (header, len, status, crc8)

    def __init__(self, auto_connect: bool = True, port_path: Optional[str] = None):
        """
        Initializes the EvoThermalSensor.
        Args:
            auto_connect: If True, attempts to connect to the sensor upon initialization.
            port_path: Optional specific serial port path to connect to. If None, scans for sensor.
        """
        self.port: Optional[serial.Serial] = None
        self.serial_lock = threading.Lock()
        self.is_connected: bool = False
        self._port_path_override: Optional[str] = port_path # Store user-specified port

        self.crc32_func = crcmod.predefined.mkPredefinedCrcFun('crc-32-mpeg')
        self.crc8_func = crcmod.predefined.mkPredefinedCrcFun('crc-8')

        self.CMD_ACTIVATE_STREAM = bytes([0x00, 0x52, 0x02, 0x01, 0xDF])
        self.CMD_DEACTIVATE_STREAM = bytes([0x00, 0x52, 0x02, 0x00, 0xD8])

        if auto_connect:
            try:
                self.connect()
            except Exception as e:
                 logger.error(f"Auto-connect failed for EvoThermalSensor: {e}", exc_info=True)


    def _find_thermal_port(self) -> Optional[str]:
        """Scans COM ports for the EvoThermal sensor unless a port_path was specified."""
        if self._port_path_override:
            logger.info(f"Attempting to use specified port: {self._port_path_override}")
            # Basic check if port exists in list of available ports
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if self._port_path_override in available_ports:
                return self._port_path_override
            else:
                logger.warning(f"Specified port {self._port_path_override} not found in available ports: {available_ports}.")
                return None

        logger.debug("Scanning for EvoThermal sensor by VENDOR_ID...")
        try:
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                if p.hwid and self.VENDOR_ID in p.hwid:
                    logger.info(f"EvoThermal sensor found on port: {p.device} (HWID: {p.hwid})")
                    return p.device
            logger.warning(f"EvoThermal sensor with VENDOR_ID '{self.VENDOR_ID}' not found.")
            return None
        except Exception as e:
            logger.error(f"Error occurred while scanning for serial ports: {e}", exc_info=True)
            return None

    def connect(self) -> bool:
        """Establishes connection to the thermal sensor."""
        if self.is_connected and self.port and self.port.is_open:
            logger.info(f"Already connected to thermal sensor on {self.port.name}.")
            return True

        # Disconnect if already initialized but not open, to be safe
        if self.port: self.disconnect()

        port_to_try = self._find_thermal_port()
        if not port_to_try:
            return False

        try:
            with self.serial_lock:
                self.port = serial.Serial(
                    port=port_to_try,
                    baudrate=self.BAUDRATE,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=0.5, # Shorter timeout for general ops, specific timeouts for blocking reads
                    write_timeout=0.5
                )

            if self.port.is_open:
                self.is_connected = True
                logger.info(f"Successfully connected to EvoThermal sensor on {port_to_try}.")
                # Optionally send a benign command to test communication here
                return True
            else:
                logger.error(f"Failed to open serial port {port_to_try} (is_open is False after init).")
                self.is_connected = False # Ensure state
                return False
        except serial.SerialException as e:
            logger.error(f"SerialException while connecting to {port_to_try}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error connecting to {port_to_try}: {e}", exc_info=True)

        self.is_connected = False # Ensure state if any error
        self.port = None
        return False

    # is_running() method removed and archived.

    def _send_command(self, command: bytes, expect_ack: bool = True) -> bool:
        """Sends a command and optionally checks for ACK. Returns True on success."""
        if not self.is_connected or not self.port or not self.port.is_open:
            logger.warning(f"Sensor not connected. Cannot send command {command.hex()}.")
            return False

        try:
            with self.serial_lock:
                self.port.reset_input_buffer() # Clear stale data
                bytes_written = self.port.write(command)
                self.port.flush()
                if bytes_written != len(command):
                    logger.warning(f"Not all bytes of command {command.hex()} written: {bytes_written}/{len(command)}")
                    return False
                logger.debug(f"Command {command.hex()} sent ({bytes_written} bytes).")

                if not expect_ack:
                    return True

                # Read ACK (expected: ACK_PACKET_SIZE bytes, e.g. 4 bytes)
                ack_data = self.port.read(self.ACK_PACKET_SIZE)
                if len(ack_data) != self.ACK_PACKET_SIZE:
                    logger.warning(f"ACK for command {command.hex()} incomplete or timeout. Received: {ack_data.hex()}")
                    return False

                if ack_data[0] != self.ACK_HEADER_VALUE:
                    logger.warning(f"Invalid ACK header for {command.hex()}: {ack_data[0]:02X} (expected {self.ACK_HEADER_VALUE:02X}). Full ACK: {ack_data.hex()}")
                    return False

                # CRC8 validation for ACK (first 3 bytes, 4th is CRC)
                if self.crc8_func(ack_data[:self.ACK_PACKET_SIZE-1]) != ack_data[self.ACK_PACKET_SIZE-1]:
                    logger.warning(f"ACK CRC8 mismatch for {command.hex()}. ACK: {ack_data.hex()}")
                    return False

                # Check status byte in ACK (e.g., byte 2, 0 means success)
                if ack_data[2] == 0: # Assuming 0 is success
                    logger.debug(f"Command {command.hex()} acknowledged successfully.")
                    return True
                else:
                    logger.warning(f"Command {command.hex()} NACK'd. Status byte: {ack_data[2]:02X}. ACK: {ack_data.hex()}")
                    return False
        except serial.SerialTimeoutException:
            logger.error(f"Write timeout sending command {command.hex()}.", exc_info=True)
        except serial.SerialException as e:
            logger.error(f"SerialException sending command {command.hex()}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error sending command {command.hex()}: {e}", exc_info=True)
        return False

    def read_thermal_frame(self) -> Optional[np.ndarray]:
        """Reads a thermal frame. Activates, reads, then deactivates stream."""
        if not self.is_connected:
            logger.warning("Not connected, cannot read thermal frame.")
            # Attempt re-connect if not connected
            # if not self.connect(): return None
            return None

        if not self._send_command(self.CMD_ACTIVATE_STREAM):
            logger.error("Failed to activate stream for frame read.")
            self._send_command(self.CMD_DEACTIVATE_STREAM, expect_ack=False) # Attempt to clean up
            return None

        frame = self._read_raw_frame_packet()

        if not self._send_command(self.CMD_DEACTIVATE_STREAM, expect_ack=True): # Expect ACK for deactivate
            logger.warning("Failed to deactivate stream after frame read attempt.")
            # This is not ideal, sensor might keep streaming. Consider port reset or re-connect.

        return frame


    def _read_raw_frame_packet(self) -> Optional[np.ndarray]:
        """Reads and processes one raw frame data packet from the sensor."""
        if not self.port or not self.port.is_open: return None

        # Increased timeout specific for frame reading
        # Original timeout for port is 0.5s. Frame reading needs more.
        # Temporarily set longer timeout for this read operation if possible, or ensure port timeout is sufficient.
        # For now, relying on multiple reads if data comes in chunks.

        raw_frame_buffer = bytearray()
        # Frame header is 2 bytes (e.g. 0x00 0x0D)
        # FRAME_PACKET_SIZE includes these 2 header bytes, then data, then CRC
        expected_bytes_total = self.FRAME_PACKET_SIZE

        try:
            with self.serial_lock:
                self.port.reset_input_buffer() # Clear buffer before reading frame
                # Read until we get the full packet or timeout
                # This loop tries to accumulate the full packet.
                read_start_time = time.monotonic()
                while len(raw_frame_buffer) < expected_bytes_total and (time.monotonic() - read_start_time) < 2.0: # 2s timeout for full frame
                    bytes_to_read = min(self.port.in_waiting, expected_bytes_total - len(raw_frame_buffer)) if self.port.in_waiting > 0 else 1
                    if bytes_to_read > 0:
                         raw_frame_buffer.extend(self.port.read(bytes_to_read))
                    if len(raw_frame_buffer) == expected_bytes_total: break
                    time.sleep(0.005) # Short sleep to avoid pegging CPU

                if len(raw_frame_buffer) != expected_bytes_total:
                    logger.warning(f"Incomplete frame packet: got {len(raw_frame_buffer)}/{expected_bytes_total} bytes. Data: {raw_frame_buffer.hex()}")
                    return None

            # Validate header (first 2 bytes)
            # Assuming header is little-endian short (H) and its value is FRAME_HEADER_VALUE
            frame_header = unpack('<H', raw_frame_buffer[:2])[0]
            if frame_header != self.FRAME_HEADER_VALUE: # e.g. 0x000D
                logger.warning(f"Frame header mismatch: {frame_header:#06x} (expected {self.FRAME_HEADER_VALUE:#06x}).")
                return None

            # CRC32 validation (on the thermal data part, before CRC itself)
            # Thermal data starts after 2-byte header. Ends before 4-byte CRC.
            data_for_crc_start_idx = 2
            data_for_crc_end_idx = 2 + self.THERMAL_DATA_SIZE_BYTES
            thermal_data_payload = raw_frame_buffer[data_for_crc_start_idx : data_for_crc_end_idx]

            calculated_crc = self.crc32_func(thermal_data_payload)

            # Received CRC is the last 4 bytes of the packet
            received_crc_bytes = raw_frame_buffer[data_for_crc_end_idx : data_for_crc_end_idx + self.CRC_IN_FRAME_SIZE_BYTES]
            received_crc = unpack('<I', received_crc_bytes)[0] # Little-endian unsigned int

            if calculated_crc != received_crc:
                logger.warning(f"Frame CRC32 mismatch. Calc: {calculated_crc:08X}, Recv: {received_crc:08X}.")
                return None

            # Extract thermal data (shorts), convert to Celsius
            # Data is THERMAL_DATA_SIZE_BYTES, which is 1024 shorts (32*32*2)
            num_shorts = self.THERMAL_RESOLUTION[0] * self.THERMAL_RESOLUTION[1]
            thermal_values_short = unpack('<' + 'H' * num_shorts, thermal_data_payload[:num_shorts*2])

            thermal_values_float = np.array(thermal_values_short, dtype=np.float32)
            thermal_values_celsius = (thermal_values_float / self.DECIKELVIN_SCALE_FACTOR) - self.KELVIN_TO_CELSIUS_OFFSET

            return thermal_values_celsius.reshape(self.THERMAL_RESOLUTION)

        except serial.SerialException as e:
            logger.error(f"SerialException during raw frame read: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during raw frame processing: {e}", exc_info=True)
        return None

    # get_average_temperature() method removed and archived.

    def disconnect(self) -> bool:
        """Disconnects from the sensor. Returns True if successful or already disconnected."""
        if not self.is_connected and (self.port is None or not self.port.is_open):
            logger.debug("Already disconnected.")
            return True

        logger.info("Disconnecting from EvoThermal sensor...")
        if self.port and self.port.is_open:
            try:
                # Try to deactivate stream, but don't let it block final port closure
                self._send_command(self.CMD_DEACTIVATE_STREAM, expect_ack=False) # Send deactivate, don't wait for ACK if problematic
                with self.serial_lock:
                    self.port.close()
                logger.info("Serial port closed.")
            except Exception as e:
                logger.error(f"Error closing serial port: {e}", exc_info=True)

        self.port = None
        self.is_connected = False
        return True # Assume disconnected even if errors occurred during close

    def __del__(self):
        """Destructor to ensure resources are released."""
        logger.debug(f"EvoThermalSensor instance for port '{self._port_path_override or 'auto'}' being deleted. Cleaning up...")
        self.disconnect()

# Legacy EvoThermal class (and its methods) removed.
# All usages should be updated to EvoThermalSensor or functions in archive.py.
