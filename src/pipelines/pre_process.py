"""
Data processing module for handling and preparing data for machine learning.

This module provides functions for:
- Validating required columns in a DataFrame.
- Removing rows with invalid power values.
- Preparing and cleaning the data.
- Splitting the data into training and testing sets.

Dependencies:
- pandas: For data manipulation and analysis.
- sklearn: For splitting the dataset into training and testing sets.

Functions:
- validate_columns: Ensures the DataFrame contains all required columns.
- remove_invalid_power_rows: Removes rows where LV ActivePower is 0 but
 Theoretical_Power_Curve is not 0.
- prepare_data: Prepares and cleans the data.
- split_data: Splits the data into training and testing sets.
"""

from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def validate_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Ensure the DataFrame contains all required columns.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (list): A list of required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


def remove_invalid_power_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where LV ActivePower is 0 but Theoretical_Power_Curve is not 0.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df[~((df["LV ActivePower (kW)"] == 0) & (df["Theoretical_Power_Curve (KWh)"] != 0))]
    return df


def prepare_data(df: pd.DataFrame, mode: Optional[str] = None) -> pd.DataFrame:
    """Prepare and clean the data.

    Args:
        df (pd.DataFrame): The DataFrame to prepare.
        mode (Optional[str]): Optional mode to determine the required columns.
                              If "score", "LV ActivePower (kW)" will be excluded.

    Returns:
        pd.DataFrame: The prepared DataFrame.
    """
    required_columns = [
        "Date/Time",
        "LV ActivePower (kW)",
        "Wind Speed (m/s)",
        "Theoretical_Power_Curve (KWh)",
        "Wind Direction (°)",
    ]
    if mode == "score":
        required_columns.remove("LV ActivePower (kW)")

    validate_columns(df, required_columns)

    df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")
    df["Month"] = df["Date/Time"].dt.month
    df["Hour"] = df["Date/Time"].dt.hour
    df.drop("Date/Time", axis=1, inplace=True)

    # Remove rows with months January or December
    df = df[~df["Month"].isin([1, 12])]

    # Remove outliers based on wind speed
    Q1 = df["Wind Speed (m/s)"].quantile(0.25)
    Q3 = df["Wind Speed (m/s)"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[
        ~((df["Wind Speed (m/s)"] < (Q1 - 1.5 * IQR)) | (df["Wind Speed (m/s)"] > (Q3 + 1.5 * IQR)))
    ]
    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, mode: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): Proportion of the dataset to include in the test split.
        mode (Optional[str]): Optional mode to determine the required columns.
                              If "score", "LV ActivePower (kW)" will be excluded.

    Returns:
        Tuple[pd.DataFrame]: The training features, training targets, testing features, and
          testing targets.
    """
    df = prepare_data(df, mode)
    df = remove_invalid_power_rows(df)
    trainDF, testDF = train_test_split(df, test_size=test_size, random_state=1234)

    X_train = trainDF.drop(columns=["LV ActivePower (kW)"]).values
    y_train = trainDF["LV ActivePower (kW)"].values
    X_test = testDF.drop(columns=["LV ActivePower (kW)"]).values
    y_test = testDF["LV ActivePower (kW)"].values

    return X_train, y_train, X_test, y_test
