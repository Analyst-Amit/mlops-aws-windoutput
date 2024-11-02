"""This module provides functionality for training and evaluating
 a machine learning model using the ExtraTreesRegressor.

"""
import os
from typing import Any, Tuple

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import ExtraTreesRegressor

from pipelines.data_pull import load_data
from pipelines.experiment import (
    mlflow_initial_tags_aliases,
    run_mlflow_model_update,
    setup_mlflow_experiment,
)
from pipelines.pre_process import split_data
from utils._config import get_argv_config, load_env_file, parse_args, save_model_to_s3


def evaluate_performance(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Evaluate the performance of a machine learning model on training and
      test datasets.

    This function computes the accuracy of the given model on both the trainin
    g and test datasets and logs the accuracy metrics to the console.
      It returns the accuracy scores for both datasets.

    Args:
        model (Any): The machine learning model to be evaluated. It should
          implement a `score` method that takes features and target labels
            as input and returns the accuracy score.
        X_train (pd.DataFrame): The feature matrix for the training data.
        y_train (pd.DataFrame): The target labels for the training data.
        X_test (pd.DataFrame): The feature matrix for the test data.
        y_test (pd.DataFrame): The target labels for the test data.

    Returns:
        Tuple[float, float]: A tuple containing the training accuracy and test
          accuracy, in that order.
    """
    print("Model evaluation...")
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Accuracy on train: {train_accuracy:.0%}")
    print(f"Accuracy on test : {test_accuracy:.0%}")

    return train_accuracy, test_accuracy


def main(config) -> None:
    """
    Main function for training a machine learning model using ExtraTreesRegressor.

    This function:
    1. Loads the configuration settings.
    2. Sets up an MLflow experiment.
    3. Loads and prepares the training data.
    4. Trains an ExtraTreesRegressor model.
    5. Logs the model and its performance metrics to MLflow.
    6. Saves the trained model to a specified file path.

    The function does not return any values but performs actions such as training
      the model,
    logging metrics, and saving the model artifact.

    Returns:
        None: This function does not return a value.
    """

    mlflow_config = config["MLflow"]
    files_config = config["Files"]
    model_config = config["ModelParameters"]

    # Parse arguments
    args = parse_args()

    # Load the environment variables from the .env file
    load_env_file(args.env)

    # Access the environment variables
    bucket_name = os.getenv("s3_bucket")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    setup_mlflow_experiment(MLFLOW_TRACKING_URI, mlflow_config["experiment_name"])

    # Load data
    dataDF = load_data(files_config["training_data"], bucket_name)

    with mlflow.start_run(run_name=mlflow_config["model_run_name"]):
        # Prepare data
        print("Preparing data...")

        X_train, y_train, X_test, y_test = split_data(dataDF, test_size=0.2, mode="train")

        # Train model
        print("Model training...")
        model_params = {
            "n_estimators": int(model_config["n_estimators"]),
            "min_samples_split": float(model_config["min_samples_split"]),
            "random_state": int(model_config["random_state"]),
        }

        mlflow.log_params(model_params)

        model = ExtraTreesRegressor(**model_params)

        model.fit(X_train, y_train)

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model with MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=bucket_name,
            signature=signature,
            registered_model_name=mlflow_config["registered_model_name"],
        )

        mlflow_initial_tags_aliases(mlflow_config["registered_model_name"])

        # Evaluate and log performance
        print("Model evaluation...")
        train_accuracy, test_accuracy = evaluate_performance(
            model, X_train, y_train, X_test, y_test
        )

        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Persist model to file
        print("Persisting model...")
        save_model_to_s3(model, bucket_name=bucket_name)
        print("Model training completed.")


if __name__ == "__main__":
    config = get_argv_config()
    main(config)
    run_mlflow_model_update(config)
