"""
This module handles the loading and pulling of data for further processing.

It includes functions to load data from various sources, such as CSV files,
and prepares the data for further use in the pipeline.
"""
from pathlib import Path

import boto3
import pandas as pd

from utils._config import PACKAGE_ROOT, get_argv_config


def load_data(file_name: str) -> pd.DataFrame:
    """
    Load data from a CSV file and print information about the data.

    This function reads a CSV file from the specified file path using
    pandas, loads the data into a DataFrame, and prints the number
    of rows and columns in the DataFrame. It then returns the DataFrame.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The data loaded from the CSV file.

    Prints:
        str: A message indicating the data has been successfully
        pulled from the source, along with the number of rows and columns.
    """
    file_path = f"{PACKAGE_ROOT}/data/{file_name}"
    df = pd.read_csv(file_path)
    print(
        "Pulled successfully from Source and saved to Output\n"
        f"Rows: {df.shape[0]} Columns: {df.shape[1]}"
    )
    return df


def download_from_s3():
    """
    Download both training and testing files from an S3 bucket to a local directory.

    This function downloads the training and testing files from an S3 bucket to a local
    directory defined by the PACKAGE_ROOT variable.
    """
    # Load configuration once at the module level
    config = get_argv_config()
    s3_config = config["S3Configs"]
    bucket_name = s3_config["bucket_name"]
    files_config = config["Files"]

    s3 = boto3.client("s3")

    # Define the file keys mapping for training and testing data
    file_keys = {
        "training_data": f"data/{files_config['training_data']}",
        "test_data": f"data/{files_config['test_data']}",
    }

    for key_name, file_key in file_keys.items():
        download_path = f"{PACKAGE_ROOT}/{file_key}"

        # Ensure the directory exists before saving the file
        Path(download_path).parent.mkdir(parents=True, exist_ok=True)

        # Download the file from the S3 bucket and save it to the specified local path
        s3.download_file(bucket_name, file_key, download_path)

        # Inform the user that the file has been downloaded successfully
        print(f"Downloaded {files_config[key_name]} from {bucket_name}/data to {download_path}")


if __name__ == "__main__":
    download_from_s3()
