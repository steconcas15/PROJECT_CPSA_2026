import os
from pathlib import Path
import yaml

# ---- CONFIG LOCATION (robust) ----
CONFIG_FILENAME = "config.yaml"

def _discover_config_path() -> Path:
    # 1) Environment variable override
    env_path = os.getenv("STOPME_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p

    # 2) Walk up parents from this file
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent / CONFIG_FILENAME
        if candidate.is_file():
            return candidate

    # 3) Check CWD as a last resort
    cwd_candidate = Path.cwd() / CONFIG_FILENAME
    if cwd_candidate.is_file():
        return cwd_candidate

    # Corretto a parents[1] perché utils.py è dentro la cartella utils/
    return here.parents[1] / CONFIG_FILENAME

CONFIG_PATH = _discover_config_path()

# ---- LOAD CONFIG ----
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
    return CONFIG.get("enable_system_log", True)

def debug_system_console_enabled() -> bool:
    return CONFIG.get("debug_system_console", False)

def debug_event_console_enabled() -> bool:
    return CONFIG.get("debug_event_console", False)

# SPEAKER
def get_speaker_config() -> dict:
    return CONFIG.get("speaker", {})

# BLUECOIN
def get_bluecoin_config() -> list[dict]:
    return CONFIG.get("bluecoins", []) or []

# IMU CONFIGURATION
def get_sync_config() -> dict:
    sync_cfg = CONFIG.get("sync", {}) or {}
    return {
        "max_skew_ms": int(sync_cfg.get("max_skew_ms", 25)),
        "stale_ms": int(sync_cfg.get("stale_ms", 100)),
    }

# BUFFER CONFIGURATION
def get_buffer_config() -> dict:
    buff_cfg = CONFIG.get("buffer", {}) or {}
    return {
        "window_size": int(buff_cfg.get("window_size", 150)),
        "overlap": int(buff_cfg.get("overlap", 75)),
        "debug_print_buffer": bool(buff_cfg.get("debug_print_buffer", True)),
        "debug_print_features": bool(buff_cfg.get("debug_print_features", True)),
    }

# QUEUE SIZE
def get_event_queue_size() -> int:
    return int(CONFIG.get("event_queue_size", 5))

# YOLO XMODEL PATH
def get_yolo_path() -> str:
    return str(Path(CONFIG["yolo_model_name"]).expanduser())

# RESNET18 XMODEL PATH
def get_resnet_path() -> str:
    return str(Path(CONFIG["resnet_model_name"]).expanduser())
