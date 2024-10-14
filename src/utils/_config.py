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
import tempfile
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict
import os
from dotenv import load_dotenv

import boto3
import joblib
from botocore.exceptions import ClientError


PACKAGE_ROOT = Path(__file__).parents[2]
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.ini"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "src/trained_models/model.bin"


from pathlib import Path
from dotenv import load_dotenv
import argparse

def load_env_file(env: str):
    """
    Load the .env file based on the environment (e.g., 'dev', 'prod').
    
    Args:
        env (str): The environment to load (e.g., 'dev', 'prod').
    """
    env_path = Path(__file__).resolve().parents[2] / f'conf/{env}.env'
    
    # Load the .env file
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    
    load_dotenv(dotenv_path=env_path)
    print(f"Environment: {env}")

def parse_args():
    """
    Parse command-line arguments for environment and user.
    
    Returns:
        Namespace: Parsed arguments containing 'env' and 'user'.
    """
    parser = argparse.ArgumentParser(description="Process environment configurations.")
    
    # Define named arguments
    parser.add_argument('--env', required=True, help="Environment to use (e.g., 'dev' or 'prod')")
    # parser.add_argument('--user', required=True, help="Username to use")
    
    # Parse the arguments
    return parser.parse_args()


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


def save_model_to_s3(model: Any, bucket_name: str) -> None:
    """Save the model to S3."""
    s3_client = boto3.client("s3")
    key = "Artifacts/model.bin"
    try:
        with tempfile.TemporaryFile() as fp:
            joblib.dump(model, fp)
            fp.seek(0)
            s3_client.put_object(Body=fp.read(), Bucket=bucket_name, Key=key)
            print(f"Model saved to s3://{bucket_name}/{key}")
    except Exception as e:
        print(f"Failed to save model to S3: {e}")


def load_model_from_s3(bucket_name: str) -> Any:
    s3_client = boto3.client("s3")
    key = "Artifacts/model.bin"
    """Load the model from S3."""
    model = None
    try:
        with tempfile.TemporaryFile() as fp:
            s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
            fp.seek(0)
            model = joblib.load(fp)
            print(f"Model loaded from s3://{bucket_name}/{key}")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print(error_code)
        print(f"Failed to load model from S3: Model not found at s3://{bucket_name}/Artifacts")
        return "404"

    return model
