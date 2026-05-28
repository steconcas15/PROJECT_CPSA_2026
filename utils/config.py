import os  # Import the operating system module to interact with environment variables and file paths
from pathlib import Path  # Import Path from pathlib for robust, object-oriented file system paths management
import yaml  # Import the yaml module to parse and load data from YAML configuration files

# ---- CONFIG LOCATION (robust) ----
CONFIG_FILENAME = "config.yaml"  # Define a constant for the target configuration filename

def _discover_config_path() -> Path:  # Define a private function that returns a Path object pointing to config.yaml
    # 1) Environment variable override
    env_path = os.getenv("STOPME_CONFIG")  # Check if the environment variable 'STOPME_CONFIG' is set on the system
    if env_path:  # If the environment variable exists and is not empty
        p = Path(env_path).expanduser()  # Convert the env string to a Path object and resolve any '~' user home shortcuts
        if p.is_file():  # Check if the path actually points to an existing file
            return p  # Return this verified path immediately, overriding any other discovery method

    # 2) Walk up parents from this file
    here = Path(__file__).resolve()  # Get the absolute, real file path of this specific script file (config.py)
    for parent in [here.parent] + list(here.parents)]:  # Loop through the current directory and all its parent directories moving upwards
        candidate = parent / CONFIG_FILENAME  # Construct a potential path by appending 'config.yaml' to the current parent directory
        if candidate.is_file():  # Check if 'config.yaml' actually exists in this specific directory
            return candidate  # Return the path immediately as soon as the file is found

    # 3) Check CWD as a last resort
    cwd_candidate = Path.cwd() / CONFIG_FILENAME  # Construct a path looking for 'config.yaml' in the Current Working Directory
    if cwd_candidate.is_file():  # Check if the file exists right inside the directory where the terminal ran the script
        return cwd_candidate  # Return this path if found

    # Fallback default path if the file wasn't found anywhere else
    return here.parents[1] / CONFIG_FILENAME  # Go up 2 levels (out of utils/) and look for config.yaml in the project root directory

CONFIG_PATH = _discover_config_path()  # Call the discovery function and store the resolved path to config.yaml in a global constant

# ---- LOAD CONFIG ----
try:  # Start a try block to handle potential errors safely while opening or parsing the file
    with open(CONFIG_PATH, "r") as f:  # Open the discovered config.yaml file in read-only mode using a secure file context
        CONFIG = yaml.safe_load(f) or {}  # Parse the YAML content into a Python dictionary, defaulting to an empty dict if the file is empty
    if not isinstance(CONFIG, dict):  # Verify if the parsed object is a valid Python dictionary structure
        CONFIG = {}  # If it is not a dictionary (e.g., if it parsed as a string), reset it to a safe empty dictionary
except Exception:  # Catch any exception or error that occurs during file opening or parsing (e.g., FileNotFoundError, SyntaxError)
    CONFIG = {}  # Fallback to an empty dictionary so the program won't crash if configuration loading fails

# YOLO XMODEL PATH
def get_yolo_path() -> str:  # Define a public function to retrieve the YOLO model path as a string
    return str(Path(CONFIG["yolo_model_name"]).expanduser())  # Extract 'yolo_model_name' from CONFIG dict, expand any '~', and return it as a clean string

# RESNET18 XMODEL PATH
def get_resnet_path() -> str:  # Define a public function to retrieve the ResNet18 model path as a string
    return str(Path(CONFIG["resnet_model_name"]).expanduser())  # Extract 'resnet_model_name' from CONFIG dict, expand any '~', and return it as a clean string
