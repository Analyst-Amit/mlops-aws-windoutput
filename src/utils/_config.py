"""Config Environment Configuration

    Raises:
        AttributeError: Failure to retrieve expected attribute value.

    Classes:
        Env: Enumeration Representing Environments.
        Defaults: Enumeration Representing Defaults.

    Functions:
        get_file() Get File.
        get_json() Get Json.
        get_pickle() Get Pickle.

    ConfigParser:
        [section][option]: returns value.
        get(section, option): returns value.
        getint(section, option): returns integer value.
        getfloat(section, option): returns float value.
        getboolean(section, option): returns boolean value.
        getlist(section, option): returns list value.
        getsecret(section, option): returns secret value.
"""

import json
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict

import joblib


PACKAGE_ROOT = Path(__file__).parents[2]
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.ini"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "src/trained_models/model.bin"


def get_argv_config(cfg_file_path=CONFIG_FILE_PATH):
    # Create a ConfigParser object
    config = ConfigParser()

    # Check if the file exists
    if not CONFIG_FILE_PATH.is_file():
        print(f"Config file not found: {cfg_file_path}")
        return None

    config.read(cfg_file_path)

    # Print available sections for debugging
    print("Available sections:", config.sections())

    return config


def get_file(path: str) -> str:
    """Get File and Return Contents.

    Returns:
        str: File Contents.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_json(path: str) -> Dict[str, Any]:
    """Get Json and Return Contents.

    Returns:
        Dict[str, Any]: File Contents.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pickle(path: str = TRAINED_MODEL_DIR) -> Any:
    """Get Pickle and Return Contents.

    Args:
        path (str): The path to the pickle file.

    Returns:
        Any: The loaded object from the pickle file.
    """
    # Convert the path to a Pathlib object
    path = Path(path)

    # Ensure the directory exists before attempting to open the file
    if not path.exists():
        raise FileNotFoundError(f"The file at {path} does not exist.")

    # Open the file and load the pickle content
    with path.open("rb") as f:
        return joblib.load(f)


def save_model(model: Any, model_path: str = TRAINED_MODEL_DIR) -> None:
    """Save the trained model to a file.

    Args:
        model (Any): The trained machine learning model to be saved.
        model_path (str): The path where the model file will be saved.
    """
    # Ensure the directory exists before saving the model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model to the specified file
    with model_path.open("wb") as f_out:
        joblib.dump(model, f_out)

    print(f"Model saved to {model_path}")
