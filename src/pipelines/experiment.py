"""
MLflow Experiment Setup and Model Evaluation Module

This module provides functions to:
1. Set up and manage MLflow experiments.
2. Initialize model tags and aliases.
3. Evaluate challenger models and update the model registry based on performance.

Dependencies:
- `mlflow`: For MLflow experiment and model management.
- `os`: For environment variable access.
- `sklearn`: For model performance evaluation metrics.
"""

from time import sleep

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import root_mean_squared_error

from utils._config import load_model_by_alias


def setup_mlflow_tracking(uri="http://localhost:5000"):
    """
    Set the MLflow tracking URI and initialize the client.

    Args:
        uri (str): MLflow tracking server URI.

    Returns:
        MlflowClient: Initialized MLflow client.
    """
    mlflow.set_tracking_uri(uri)
    print("Tracking URI:", mlflow.get_tracking_uri())
    return MlflowClient()


def setup_mlflow_experiment(mlflow_tracking_uri, experiment_name):
    """
    Ensure MLflow experiment exists and set it for logging.

    Args:
        mlflow_tracking_uri (str): MLflow tracking server URI.
        experiment_name (str): The name of the experiment to create or use.

    Raises:
        MlflowException: If there's an issue interacting with MLflow.
    """
    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            print(f'Experiment "{experiment_name}" created.')
        else:
            print(f'Experiment "{experiment_name}" already exists.')
        mlflow.set_experiment(experiment_name)
    except MlflowException as e:
        print(f"Error setting experiment: {e}")


def mlflow_initial_tags_aliases(registered_model_name):
    """
    Set initial 'Candidate' alias for the latest version of a newly registered model.

    Args:
        registered_model_name (str): The name of the registered model.
    """
    client = MlflowClient()

    # Get all versions of the model and sort by version number to find the latest
    all_versions = client.search_model_versions(f"name='{registered_model_name}'")
    if all_versions:
        latest_version = max(all_versions, key=lambda v: int(v.version))

        # Set the 'Candidate' alias for the latest version
        client.set_registered_model_alias(
            registered_model_name, "candidate", latest_version.version
        )
        print(
            f"Alias 'candidate' set for version {latest_version.version}"
            f"of model '{registered_model_name}'."
        )
    else:
        print("No versions found for the specified model.")


def update_model_alias(client, model_name, new_alias, version, old_alias=None):
    """
    Set a new alias for a model version and optionally delete an old alias.
    """
    client.set_registered_model_alias(model_name, new_alias, version)
    if old_alias:
        client.delete_registered_model_alias(model_name, old_alias)
    print(f"Alias '{new_alias}' set for version {version} of model '{model_name}'.")
    if old_alias:
        print(f"Alias '{old_alias}' removed from model '{model_name}'.")


def calculate_rmse(predictions, true_values):
    """
    Calculate the Root Mean Squared Error (RMSE) between predictions and true values.

    Args:
        predictions (array-like): Predicted values.
        true_values (array-like): Actual values.

    Returns:
        float: Computed RMSE.
    """
    # return np.sqrt(mean_squared_error(true_values, predictions))
    return root_mean_squared_error(true_values, predictions)


def evaluate_and_update_champion(client, model_name, data, true_values):
    """
    Evaluate challenger and champion models, updating aliases based on RMSE comparison.
    """
    # Load models
    challenger_model = load_model_by_alias(model_name, "challenger")
    champion_model = load_model_by_alias(model_name, "champion")

    # Generate predictions and calculate RMSE
    challenger_rmse = calculate_rmse(challenger_model.predict(data), true_values)
    champion_rmse = calculate_rmse(champion_model.predict(data), true_values)

    # Determine whether to update champion alias
    if challenger_rmse < champion_rmse:
        print("Challenger model is better. Updating Champion model...")
        challenger_version = client.get_model_version_by_alias(model_name, "challenger").version
        champion_version = client.get_model_version_by_alias(model_name, "champion").version

        # Update aliases: set 'challenger' as 'champion' and archive previous champion
        update_model_alias(
            client, model_name, "champion", challenger_version, old_alias="challenger"
        )
        update_model_alias(client, model_name, "archived", champion_version)
    else:
        print("Current Champion model is better; no update performed.")
        challenger_version = client.get_model_version_by_alias(model_name, "challenger").version
        update_model_alias(
            client, model_name, "archived", challenger_version, old_alias="challenger"
        )


def prepare_evaluation_data():
    """
    Prepare a sample data frame for evaluation.
    """
    data = pd.DataFrame(
        {
            "Wind Speed (m/s)": [8.218296051, 4.995032787, 2.212670088, 2.190548897, 4.157712936],
            "Theoretical_Power_Curve (KWh)": [1657.373187, 334.7802577, 0, 0, 152.8500671],
            "Wind Direction (°)": [78.12586975, 17.07011986, 127.6598969, 329.9155884, 81.20819855],
            "Month": [8, 5, 4, 7, 4],
            "Hour": [21, 11, 19, 11, 10],
        }
    )
    true_values = [1339.537964, 277.7774048, 0, 0, 87.54579926]
    return data, true_values


def run_mlflow_model_update(config):
    mlflow_config = config["MLflow"]
    model_name = mlflow_config["registered_model_name"]

    client = setup_mlflow_tracking()
    print("MLflow updating 'candidate' to 'challenger'...")

    candidate_version = client.get_model_version_by_alias(model_name, "candidate").version
    update_model_alias(client, model_name, "challenger", candidate_version, old_alias="candidate")
    sleep(15)

    # Prepare evaluation data and evaluate models
    data, true_values = prepare_evaluation_data()
    evaluate_and_update_champion(client, model_name, data, true_values)


if __name__ == "__main__":
    run_mlflow_model_update()
