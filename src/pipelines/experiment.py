"""
Module for MLflow experiment setup and management.

This module contains functions for configuring and managing MLflow experiments,
ensuring that experiments exist and are properly set up for logging.

Dependencies:
- `mlflow`: For interacting with MLflow's tracking server and managing experiments.
- `os`: For accessing environment variables.

Functions:
- `setup_mlflow_experiment(experiment_name: str) -> None`: Ensures that an MLflow
 experiment exists and sets it for logging.
"""

import os

import mlflow
from mlflow.exceptions import MlflowException


def setup_mlflow_experiment(MLFLOW_TRACKING_URI: str, experiment_name: str) -> None:
    """Ensure MLflow experiment exists and set it for logging.

    Args:
        experiment_name (str): The name of the MLflow experiment to be checked or
          created.

    Raises:
        MlflowException: If there's an error while interacting with MLflow.
    """
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Check if the experiment exists
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            mlflow.create_experiment(experiment_name)
            print(f'Experiment "{experiment_name}" created.')
        else:
            print(f'Experiment "{experiment_name}" already exists.')

        # Set the experiment for MLflow logging
        mlflow.set_experiment(experiment_name)

    except MlflowException as e:
        print(f"Error setting experiment: {e}")
