"""
This module handles the loading and pulling of data for further processing.

It includes functions to load data from various sources, such as CSV files,
and prepares the data for further use in the pipeline.
"""
import pandas as pd
from utils._config import PACKAGE_ROOT


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
