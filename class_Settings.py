import requests
from ruamel.yaml import YAML
from pprint import pprint
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialize the YAML object with round_trip_load and round_trip_dump to preserve formatting
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

# Use Path for better file handling
CONFIG_FILE = Path("./LDBG_L18_config.yaml")
# CONFIG_FILE = Path("./test_config.yaml")
class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass

class Settings:
    REQUIRED_KEYS = [
        'work_dir', 'sensaray_type', 'line_number', 'line_name',
        'hsv_pixel_threshold', 'buffersize', 'entropy_threshold'
    ]
    
    VALID_SENSARAY_TYPES = [
        'BO2_400_noIR', 'BO2_400_EVO', 'BO2_400_iTEC', 'BO3_400_noIR', 'TRIAL_MODE'
    ]

    @staticmethod
    def load_yaml_config() -> Optional[Dict[str, Any]]:
        """Load and validate YAML configuration."""
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
                config = yaml.load(file)
                Settings._validate_config(config)
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {CONFIG_FILE}")
            raise ConfigurationError(f"Configuration file not found: {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            raise ConfigurationError(f"Error loading YAML config: {e}")

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        if not config:
            raise ConfigurationError("Configuration is empty")
            
        # Check required keys - skip validation for trial mode
        if config.get('sensaray_type') != 'TRIAL_MODE':
            missing_keys = [key for key in Settings.REQUIRED_KEYS if key not in config]
            if missing_keys:
                raise ConfigurationError(f"Missing required config keys: {missing_keys}")
            
        # Validate sensaray type
        if config.get('sensaray_type') not in Settings.VALID_SENSARAY_TYPES:
            raise ConfigurationError(f"Invalid sensaray_type: {config.get('sensaray_type')}")
            
        # Validate numeric ranges
        if config.get('hsv_pixel_threshold', 0) < 0:
            raise ConfigurationError("hsv_pixel_threshold must be non-negative")
            
        if config.get('buffersize', 10) <= 0:
            raise ConfigurationError("buffersize must be positive")

    @staticmethod
    def save_yaml_config(data: Dict[str, Any]) -> None:
        """Save configuration with validation."""
        try:
            Settings._validate_config(data)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
        except Exception as e:
            logger.error(f"Error saving YAML config: {e}")
            raise ConfigurationError(f"Error saving YAML config: {e}")

    @staticmethod
    def update_settings_from_server() -> Optional[Dict[str, Any]]:
        """Update settings from server with proper error handling."""
        config = Settings.load_yaml_config()
        if not config:
            logger.error("Failed to load initial configuration.")
            return None
        
        try:
            server_address = config.get('server_address')
            directus_token = config.get('directus_token')
            line_number = config.get('line_number')
            
            if not all([server_address, directus_token, line_number]):
                logger.warning("Missing server configuration, using local config")
                return config
                
            url_settings = (
                f"{server_address}/items/Settings?limit=-1"
                f"&filter[_and][0][production_line][id][_eq]={line_number}"
                f"&filter[_and][1][active][_eq]=true&access_token={directus_token}"
            )

            with requests.Session() as session:
                response = session.get(url_settings, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'data' not in data or not data['data']:
                    logger.warning("No active settings found for production line")
                    return config
                
                # Update specific settings
                server_settings = data['data'][0]
                config.update({
                    "cfc_left_cutoff": server_settings.get('cfc_left_cutoff', config.get('cfc_left_cutoff')),
                    "cfc_right_cutoff": server_settings.get('cfc_right_cutoff', config.get('cfc_right_cutoff')),
                    "send_alerts": server_settings.get('send_alerts', config.get('send_alerts')),
                    "hsv_pixel_threshold": server_settings.get('HSV_pixel_thres', config.get('hsv_pixel_threshold')),
                    "build_hsv_mask": server_settings.get('build_HSV_mask', config.get('build_hsv_mask')),
                    "temperature_correction": server_settings.get('temperature_correction', config.get('temperature_correction')),
                })

                logger.info("Successfully updated settings from server")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error updating settings: {e}")
        except requests.exceptions.Timeout:
            logger.error("Timeout while updating settings from server")
        except ValueError as e:
            logger.error(f"Invalid response data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating settings: {e}")
        
        return config