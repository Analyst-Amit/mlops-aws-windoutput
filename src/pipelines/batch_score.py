"""Batch Score Artifacts

This script performs batch scoring on a dataset using a pre-trained model.
It includes functions to load data, preprocess it, score it, and save the results.

Dependencies:
- sys
- pathlib.Path
- src.datapull.data_pull.load_data
- src.preprocess.pre_process.prepare_data
- src.utils._config.get_argv_config
- src.utils._config.get_pickle
"""

from typing import Any

import pandas as pd

from pipelines.data_pull import load_data
from pipelines.post_process import publish_data
from pipelines.pre_process import prepare_data
from pipelines.train import main as model_train
from utils._config import get_argv_config, load_model_from_s3, load_env_file, parse_args
import os


def score_model(
    model: Any,
    wind_speed: float,
    theoretical_power: float,
    wind_direction: float,
    month: int,
    hour: int,
) -> float:
    """Score a single set of inputs using the loaded model.

    Args:
        model (Any): The machine learning model used for prediction.
        wind_speed (float): The wind speed value.
        theoretical_power (float): The theoretical power curve value.
        wind_direction (float): The wind direction value.
        month (int): The month of the observation.
        hour (int): The hour of the observation.

    Returns:
        float: The predicted value from the model.
    """
    return model.predict([[wind_speed, theoretical_power, wind_direction, month, hour]])[0]


def batch_score(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Apply batch scoring to the dataset using the provided model.

    Args:
        df (pd.DataFrame): The input DataFrame containing features for scoring.
        model (Any): The machine learning model used for batch scoring.

    Returns:
        pd.DataFrame: The DataFrame with an additional column for the predicted scores.
    """
    df["score"] = df.apply(
        lambda row: score_model(
            model,
            row["Wind Speed (m/s)"],
            row["Theoretical_Power_Curve (KWh)"],
            row["Wind Direction (°)"],
            row["Month"],
            row["Hour"],
        ),
        axis=1,
    )
    return df


def main() -> None:
    """Main function to load model, score data, and save results.

    This function:
    1. Loads the configuration.
    2. Loads the pre-trained model.
    3. Loads and preprocesses the test data.
    4. Scores the data using the model.
    5. Performs the post processing.
    """
    config = get_argv_config()
    files_config = config["Files"]

    # Parse arguments
    args = parse_args()

    # Load the environment variables from the .env file
    load_env_file(args.env)

    # Access the environment variables
    bucket_name = os.getenv('s3_bucket')

    # Try to load the model from S3
    try:
        model = load_model_from_s3(bucket_name)

        # Check if the model load returned a 400 error (model not found)
        if model == "404":
            print("Model not found, triggering model re-training...")
            model_train()

            # After re-training, attempt to load the model again
            model = load_model_from_s3(bucket_name)
            print("Model Loaded!! Continuing Model Inferencing")
            if model == "404":
                raise Exception("Model re-training failed, unable to load model after re-train.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Exiting process.")

    else:
        # Load the test data from S3 or local files
        df = load_data(files_config["test_data"], bucket_name)

        # Preprocess the data for scoring
        df = prepare_data(df, mode="score")

        # Score the data using the model
        scored_df = batch_score(df, model)

        # Perform the post-processing and save the results
        publish_data(scored_df, bucket_name)


if __name__ == "__main__":
    main()
