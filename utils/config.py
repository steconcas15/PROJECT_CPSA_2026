"""
Module: config_loader.py

This module acts as the "control center" for the project's settings. 
It automatically finds the 'config.yaml' file, loads the settings, 
and provides helper functions to access them.

Scope:
    1. It searches for the config file in different folders
       making it easy to run the code from anywhere.
"""

import os
from pathlib import Path
import yaml

# ---- CONFIG LOCATION (robust) ----
CONFIG_FILENAME = "config.yaml"

def _discover_config_path() -> Path:
    # 1) Check if the user set a specific path in an environment variable
    env_path = os.getenv("STOPME_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p

    # 2) Walk up the folders starting from where this script is located
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent / CONFIG_FILENAME
        if candidate.is_file():
            return candidate

    # 3) Check the folder where the user started the program
    cwd_candidate = Path.cwd() / CONFIG_FILENAME
    if cwd_candidate.is_file():
        return cwd_candidate

    # Fallback to a default parent directory if nothing else is found
    return here.parents[1] / CONFIG_FILENAME

CONFIG_PATH = _discover_config_path()

# ---- LOAD CONFIG ----
# Try to open the file and read it as a dictionary.
# Else use an empty dictionary
try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f) or {}
    if not isinstance(CONFIG, dict):
        CONFIG = {}
except Exception:
    CONFIG = {}

# LOGGER
def get_log_path():
    """
    Returns the resolved base log path, expanding "~" to the user's home directory.
    """
    default_base = str(Path.home() / "Documents" / "STOPME" / "logs")
    # Corretto: aggiunto os.path.expanduser (in alternativa si poteva usare Path(...).expanduser())
    return str(Path(os.path.expanduser(CONFIG.get("log_base_path", default_base))))

def actuation_details_enabled() -> bool:
    return CONFIG.get("enable_actuation_detail", True)

def system_log_enabled() -> bool:
    """ Returns True if we want detailed logs for actions taken. """
    return CONFIG.get("enable_system_log", True)

def debug_system_console_enabled() -> bool:
    """ Returns True if the system logs should be active. """
    return CONFIG.get("debug_system_console", False)

def debug_event_console_enabled() -> bool:
    return CONFIG.get("debug_event_console", False)

# METAMOTION
def get_metamotion_config() -> dict:
    return CONFIG.get("metamotion", {})

# SPEAKER
def get_speaker_config() -> dict:
    """ Gets settings for the speaker. """
    return CONFIG.get("speaker", {})

# BLUECOIN
def get_bluecoin_config() -> list[dict]:
    """ Gets settings for the bluecoins. """
    return CONFIG.get("bluecoins", []) or []

# IMU CONFIGURATION
def get_sync_config() -> dict:
    """ 
    Returns timing settings. 
    'max_skew_ms' is how much we allow sensors to be out of sync.
    """
    sync_cfg = CONFIG.get("sync", {}) or {}
    return {
        "max_skew_ms": int(sync_cfg.get("max_skew_ms", 25)),
        "stale_ms": int(sync_cfg.get("stale_ms", 100)),
    }

# BUFFER CONFIGURATION
def get_buffer_config() -> dict:
    """ Settings for how we capture and process chunks of data. """
    buff_cfg = CONFIG.get("buffer", {}) or {}
    return {
        "window_size": int(buff_cfg.get("window_size", 150)),
        "overlap": int(buff_cfg.get("overlap", 75)),
        "debug_print_buffer": bool(buff_cfg.get("debug_print_buffer", True)),
        "debug_print_features": bool(buff_cfg.get("debug_print_features", True)),
    }

# POLICY CONFIGURATION
def get_policy_attempts() -> int:
    policy_config = CONFIG.get("policy", {}) or {}
    return int(policy_config.get("attempts", 5))

# QUEUE SIZE
def get_event_queue_size() -> int:
    return int(CONFIG.get("event_queue_size", 5))

# YOLO XMODEL PATH
def get_yolo_path() -> str:
    return str(Path(CONFIG["yolo_model_name"]).expanduser())

# RESNET18 XMODEL PATH
def get_resnet_path() -> str:
    return str(Path(CONFIG["resnet_model_name"]).expanduser())
