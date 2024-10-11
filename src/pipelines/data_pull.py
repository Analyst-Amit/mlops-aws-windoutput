"""
This module handles the loading and pulling of data for further processing.

It includes functions to load data from various sources, such as CSV files,
and prepares the data for further use in the pipeline.
"""
from io import StringIO

import boto3
import pandas as pd


def load_data(file_name: str, bucket_name: str) -> pd.DataFrame:
    """
    Load data from an S3 bucket and return it as a DataFrame.

    This function fetches a CSV file from an Amazon S3 bucket, reads its contents
    into a pandas DataFrame, and returns the DataFrame. It also prints the number
    of rows and columns in the data.

    Args:
        file_name (str): The name of the CSV file to load from the S3 bucket.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.

    Prints:
        str: A message indicating the data was successfully loaded, with the
        number of rows and columns in the DataFrame.
    """
    s3 = boto3.client("s3")

    # Retrieve CSV file from S3
    obj = s3.get_object(Bucket=bucket_name, Key=f"data/{file_name}")
    data = obj["Body"].read().decode("utf-8")

    # Load data into a DataFrame
    df = pd.read_csv(StringIO(data))

    print("Data loaded successfully from S3")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df
