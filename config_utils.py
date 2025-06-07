# config_utils.py
# This file has been refactored from the original class_Settings.py.
# All static methods from the former Settings class have been converted to regular functions.
# Class-level constants have been moved to module-level constants.
# The save_yaml_config function has been removed as its archived version is in archive.py.
# Internal calls have been updated accordingly.
# This file is created anew after deleting the previous version due to tool issues.

import requests
from ruamel.yaml import YAML
from pprint import pprint
from typing import Dict, Any, Optional, Union # Added Union
import logging
from pathlib import Path

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Initialize the YAML object with settings to preserve formatting and comments if possible
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)
# yaml.explicit_start = True # Optionally add --- at the beginning of the file
# yaml.width = 120 # Example: Set line width for nicer output

# Define CONFIG_FILE path at module level
CONFIG_FILE = Path("./LDBG_L18_config.yaml")
# For testing: CONFIG_FILE = Path("./test_config.yaml") # This can be uncommented for tests

class ConfigurationError(Exception):
    """Custom exception raised when configuration loading, validation, or related operations fail."""
    pass

# Module-level constants (formerly class-level in Settings)
REQUIRED_KEYS = [
    'work_dir', 'sensaray_type', 'line_number', 'line_name',
    'hsv_pixel_threshold', 'buffersize', 'entropy_threshold'
    # Add other critical keys as the application evolves
]

VALID_SENSARAY_TYPES = [
    'BO2_400_noIR', 'BO2_400_EVO', 'BO2_400_iTEC', 'BO3_400_noIR',
    'TRIAL_MODE' # Trial mode might have relaxed validation rules
]

# --- Internal Helper Functions ---
def _validate_config(config_data: Dict[str, Any]) -> None:
    """
    Validates the structure and content of the configuration dictionary.
    Raises ConfigurationError if critical validation fails.
    This is intended as an internal helper function.
    """
    if not isinstance(config_data, dict):
        raise ConfigurationError("Configuration data must be a dictionary.")
    if not config_data: # Check if the dictionary is empty
        raise ConfigurationError("Configuration data is empty.")

    # Conditional validation: if not in 'TRIAL_MODE', enforce required keys.
    if config_data.get('sensaray_type') != 'TRIAL_MODE':
        missing_keys = [key for key in REQUIRED_KEYS if key not in config_data]
        if missing_keys:
            raise ConfigurationError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    # Validate 'sensaray_type' value
    sensaray_type = config_data.get('sensaray_type')
    if sensaray_type not in VALID_SENSARAY_TYPES:
        raise ConfigurationError(
            f"Invalid sensaray_type: '{sensaray_type}'. "
            f"Valid types are: {', '.join(VALID_SENSARAY_TYPES)}"
        )

    # Example: Validate data types and ranges for specific keys if they exist
    if 'hsv_pixel_threshold' in config_data:
        threshold = config_data['hsv_pixel_threshold']
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ConfigurationError("hsv_pixel_threshold must be a non-negative number.")

    if 'buffersize' in config_data:
        buffersize = config_data['buffersize']
        if not isinstance(buffersize, int) or buffersize <= 0:
            raise ConfigurationError("buffersize must be a positive integer.")

    if 'work_dir' in config_data and not isinstance(config_data['work_dir'], str):
        raise ConfigurationError("work_dir must be a string representing a valid path.")

    # One could add more specific checks, e.g., for URL formats, path writability, etc.
    logger.debug("Configuration data passed validation.")


# --- Public API Functions ---
def load_yaml_config(config_file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration from the specified file path or the default module-level CONFIG_FILE.
    After loading, it validates the configuration.

    Args:
        config_file_path: Optional path to the configuration file. If None, uses default.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        ConfigurationError: If the file is not found, cannot be parsed, or fails validation.
    """
    active_config_file = Path(config_file_path) if config_file_path else CONFIG_FILE
    logger.info(f"Loading configuration from: {active_config_file.resolve()}")

    try:
        with open(active_config_file, 'r', encoding='utf-8') as file_stream:
            # Using yaml.load for ruamel.yaml
            config_data = yaml.load(file_stream)

        if config_data is None: # Handles empty YAML file resulting in None
            logger.warning(f"Configuration file {active_config_file} is empty or contains only comments.")
            raise ConfigurationError(f"Configuration file {active_config_file} is empty or invalid.")

        _validate_config(config_data) # Validate the loaded data
        logger.info(f"Configuration successfully loaded and validated from {active_config_file}.")
        return config_data
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {active_config_file.resolve()}")
        raise ConfigurationError(f"Configuration file not found: {active_config_file.resolve()}")
    except Exception as e: # Catch ruamel.yaml.YAMLError and other potential exceptions
        logger.error(f"Error loading or parsing YAML configuration from {active_config_file.resolve()}: {e}", exc_info=True)
        raise ConfigurationError(f"Error loading/parsing {active_config_file.resolve()}: {e}")


def update_settings_from_server(current_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Updates settings by fetching them from a configured Directus server.
    If 'current_config' is not provided, it loads the default configuration first.
    The function attempts to merge server settings into the local configuration.

    Args:
        current_config: Optional dictionary with current application settings including server details.

    Returns:
        The updated configuration dictionary. If the update fails, it returns the original
        (or freshly loaded) configuration, logging appropriate errors.
    """
    if current_config is None:
        try:
            config_to_use = load_yaml_config() # Load default config if none is passed
        except ConfigurationError as e:
            logger.error(f"Cannot update from server: Failed to load initial configuration. Error: {e}")
            # Depending on desired behavior, could return an empty dict or re-raise
            return {} # Return empty or a minimal default config
    else:
        # Work on a copy to avoid modifying the original dict in place if it's mutable
        config_to_use = current_config.copy()

    # Extract server connection details from the configuration
    server_address = config_to_use.get('server_address')
    directus_token = config_to_use.get('directus_token')
    line_number = config_to_use.get('line_number') # Used to filter settings for the specific line

    if not all([server_address, directus_token, line_number]):
        logger.warning("Server address, Directus token, or line number is missing in configuration. "
                       "Skipping settings update from server.")
        return config_to_use

    # Construct the API request URL
    # Example: "http://server/items/Settings?limit=-1&filter[production_line][id][_eq]=LINE_NUM&filter[active][_eq]=true&access_token=TOKEN"
    url_settings = (
        f"{server_address.rstrip('/')}/items/Settings?limit=1" # limit=1 if only one active setting expected
        f"&filter[_and][0][production_line][id][_eq]={line_number}"
        f"&filter[_and][1][active][_eq]=true" # Assuming a field 'active' marks usable settings
        f"&access_token={directus_token}" # Token for authentication
    )
    # Log URL without token for security:
    log_url = url_settings.split('&access_token=')[0] + "&access_token=***" if directus_token else url_settings
    logger.info(f"Fetching settings from server: {log_url}")

    try:
        with requests.Session() as http_session:
            response = http_session.get(url_settings, timeout=15) # Increased timeout
            response.raise_for_status() # Check for HTTP errors (4xx or 5xx)

            response_data = response.json()

            if 'data' not in response_data or not isinstance(response_data['data'], list) or not response_data['data']:
                logger.warning(f"No active settings found on server for production line '{line_number}' or unexpected response structure. Using existing configuration.")
                return config_to_use

            server_settings = response_data['data'][0] # Assuming the first item is the one
            logger.info(f"Received settings from server. Data: { {k: server_settings.get(k) for k in ['cfc_left_cutoff', 'send_alerts']} } (sample keys)") # Log sample

            # Merge server settings into the local configuration dictionary
            # Only update keys that are expected to be controlled by the server.
            # Use .get() with fallback to existing value to prevent deleting keys or adding Nones if server data is sparse.
            keys_to_update = [
                "cfc_left_cutoff", "cfc_right_cutoff", "send_alerts",
                "hsv_pixel_threshold", "build_hsv_mask", "temperature_correction"
                # Add other server-configurable keys here
            ]
            server_key_map = { # If server keys differ from local keys
                "hsv_pixel_threshold": "HSV_pixel_thres"
            }

            for local_key in keys_to_update:
                server_key = server_key_map.get(local_key, local_key) # Get mapped server key or use local key
                if server_key in server_settings:
                    config_to_use[local_key] = server_settings[server_key]
                # else: logger.debug(f"Key '{server_key}' not in server response, local value for '{local_key}' preserved.")

            logger.info("Configuration successfully updated with settings from server.")
            # Optionally, re-validate the configuration if server settings could be malformed
            # _validate_config(config_to_use)

    except requests.exceptions.Timeout:
        logger.error("Timeout occurred while fetching settings from server.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code} while fetching settings: {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error occurred while fetching settings: {e}")
    except ValueError as e: # Includes JSONDecodeError
        logger.error(f"Error decoding JSON response from server: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while updating settings from server: {e}", exc_info=True)

    return config_to_use # Return config_to_use, which is updated or original if errors occurred

# --- Main Execution Block (for testing) ---
if __name__ == '__main__':
    # Basic logging setup for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a dummy config file for testing purposes
    dummy_config_filename = "test_suite_config.yaml"
    CONFIG_FILE = Path(dummy_config_filename) # Override default for this test run

    dummy_data_for_file = {
        'work_dir': '/test/sensaray_data',
        'sensaray_type': 'BO2_400_EVO', # A valid type
        'line_number': "L101", # Example line number
        'line_name': 'Test Line Alpha',
        'hsv_pixel_threshold': 120,
        'buffersize': 25,
        'entropy_threshold': 3.2,
        'server_address': 'http://mock-server.example.com:8055', # Mock server address
        'directus_token': 'test_token_12345',
        # Include keys that might be updated from server
        'cfc_left_cutoff': 5,
        'cfc_right_cutoff': 5,
        'send_alerts': False,
        'build_hsv_mask': True,
        'temperature_correction': -0.5
    }

    try:
        # Write the dummy data to the test config file
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f_stream:
            yaml.dump(dummy_data_for_file, f_stream)
        logger.info(f"Created dummy configuration file: {CONFIG_FILE.resolve()}")

        # Test 1: Load the dummy configuration
        loaded_settings = load_yaml_config() # Uses the overridden CONFIG_FILE
        logger.info("--- Test Load YAML Config ---")
        pprint(loaded_settings)
        assert loaded_settings['line_name'] == 'Test Line Alpha'

        # Test 2: Attempt to update from server (expected to log errors if mock server isn't running)
        logger.info("--- Test Update Settings From Server (mock) ---")
        updated_settings = update_settings_from_server(loaded_settings)
        pprint(updated_settings)
        # If server call fails, updated_settings should be same as loaded_settings
        assert updated_settings['work_dir'] == loaded_settings['work_dir']

        # Test 3: Validation failure - missing key (if not TRIAL_MODE)
        logger.info("--- Test Validation Failure (Missing Key) ---")
        invalid_data = dummy_data_for_file.copy()
        del invalid_data['work_dir']
        # invalid_data['sensaray_type'] = 'BO2_400_noIR' # Ensure not TRIAL_MODE
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f_stream:
            yaml.dump(invalid_data, f_stream)
        try:
            load_yaml_config()
        except ConfigurationError as e:
            logger.info(f"Caught expected validation error: {e}")
            assert "Missing required configuration keys" in str(e)

        # Test 4: Validation failure - invalid sensaray_type
        logger.info("--- Test Validation Failure (Invalid Type) ---")
        invalid_data_type = dummy_data_for_file.copy()
        invalid_data_type['sensaray_type'] = 'INVALID_TYPE_XYZ'
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f_stream:
            yaml.dump(invalid_data_type, f_stream)
        try:
            load_yaml_config()
        except ConfigurationError as e:
            logger.info(f"Caught expected validation error for type: {e}")
            assert "Invalid sensaray_type" in str(e)


    except ConfigurationError as ce_test:
        logger.error(f"A ConfigurationError occurred during testing: {ce_test}", exc_info=True)
    except Exception as e_test:
        logger.error(f"An unexpected error occurred during testing: {e_test}", exc_info=True)
    finally:
        # Clean up the dummy config file
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
            logger.info(f"Cleaned up dummy configuration file: {CONFIG_FILE.resolve()}")
