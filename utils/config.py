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

# YOLO XMODEL PATH
def get_yolo_path() -> str:
    return str(Path(CONFIG["yolo_model_name"]).expanduser())

# RESNET18 XMODEL PATH
def get_resnet_path() -> str:
    return str(Path(CONFIG["resnet_model_name"]).expanduser())
