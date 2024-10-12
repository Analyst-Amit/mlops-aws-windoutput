"""Postprocess"""

from io import StringIO

import boto3
import pandas as pd


def publish_data(df: pd.DataFrame, bucket_name, file_name: str = "result") -> None:
    """
    Save a DataFrame as a CSV file and upload it to an S3 bucket.

    This function converts a DataFrame to CSV format and uploads the file to
    a specified Amazon S3 bucket.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        file_name (str): The name of the file to save in the S3 bucket.

    Prints:
        str: A message indicating the CSV file has been successfully saved to S3.
    """
    s3 = boto3.client("s3")

    # Convert DataFrame to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload the CSV to S3
    s3.put_object(
        Bucket=bucket_name, Key=f"output_files/{file_name}.csv", Body=csv_buffer.getvalue()
    )

    print(f"Data successfully uploaded to S3 as {file_name}")
