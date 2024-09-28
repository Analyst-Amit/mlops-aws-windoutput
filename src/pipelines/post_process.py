"""Postprocess"""

import pandas as pd

from utils._config import PACKAGE_ROOT


def publish_data(df: pd.DataFrame, output_csv_path: str = PACKAGE_ROOT) -> None:
    """Save the DataFrame to a new CSV file, ensuring the directory exists.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        output_csv_path (str): The path where the CSV file will be saved.
    """
    # Convert the path to a Pathlib object
    output_csv_path = output_csv_path / "src/model_output/result.csv"

    # Create the directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
