"""Postprocess"""

import pandas as pd
from pathlib import Path


def publish_data(df: pd.DataFrame, output_csv_path: str) -> None:
    """Save the DataFrame to a new CSV file, ensuring the directory exists.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        output_csv_path (str): The path where the CSV file will be saved.
    """
    # Convert the path to a Pathlib object
    output_csv_path = Path(output_csv_path)
    
    # Create the directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the DataFrame to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
