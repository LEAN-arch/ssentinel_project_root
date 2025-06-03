# sentinel_project_root/data_processing/helpers.py
# Helper utilities for data processing tasks.

import pandas as pd
import numpy as np
import logging
import json
import hashlib
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans DataFrame column names: converts to lowercase, replaces non-alphanumeric
    characters (except underscore) with underscores, collapses multiple underscores,
    and strips leading/trailing underscores.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return pd.DataFrame() # Return empty DataFrame on invalid input

    df_copy = df.copy()
    try:
        df_copy.columns = (
            df_copy.columns.str.lower()
            .str.replace('[^0-9a-zA-Z_]', '_', regex=True)
            .str.replace('_+', '_', regex=True)
            .str.strip('_')
        )
    except Exception as e:
        logger.error(f"Error cleaning column names: {e}. Columns: {df.columns.tolist()}")
        # Return with original columns if error during cleaning specific ones
        return df
    return df_copy

def convert_to_numeric(
    series: pd.Series,
    default_value: Any = np.nan,
    target_type: Union[type(float), type(int), None] = float
) -> pd.Series:
    """
    Robustly converts a pandas Series to a numeric type (float or int).
    Non-convertible values are replaced with default_value.
    """
    if not isinstance(series, pd.Series):
        # Attempt to convert input to a Series if it's not already one
        try:
            # Determine a sensible dtype if possible, else object
            if isinstance(series, (list, np.ndarray, tuple)):
                # If all elements look numeric or are None/NaN, try float
                if all(isinstance(x, (int, float)) or x is None or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) for x in series if x is not None):
                    series_dtype = float
                else:
                    series_dtype = object # Fallback for mixed types
            else:
                series_dtype = object
            series = pd.Series(series, dtype=series_dtype)
        except Exception as e:
            logger.warning(f"Could not convert input to Series in convert_to_numeric: {e}. Returning series of default values.")
            length = len(series) if hasattr(series, '__len__') else 1
            return pd.Series([default_value] * length, dtype=type(default_value) if default_value is not np.nan else float)

    # Attempt conversion to numeric
    numeric_series = pd.to_numeric(series, errors='coerce')

    # Fill NaNs introduced by 'coerce'
    if default_value is not np.nan:
        numeric_series = numeric_series.fillna(default_value)
    
    # Convert to integer if requested and possible
    if target_type == int and pd.notna(default_value):
        try:
            # If default_value is float and has decimal, cannot directly cast to int without losing info
            if isinstance(default_value, float) and not default_value.is_integer():
                 logger.debug(f"Cannot convert to int due to float default_value {default_value} with decimal part. Returning float.")
            else:
                # Attempt to convert to integer. If any NaNs remain (if default_value was NaN), this can fail.
                # So, ensure NaNs are handled if default_value was NaN.
                if numeric_series.isnull().any() and default_value is np.nan:
                    # Cannot convert to int if NaNs are present and no int-compatible default. Keep as float.
                    pass
                else:
                    # If default_value was NaN, but all values are now non-NaN (e.g. all original were convertible),
                    # it might be possible to convert to int if they are whole numbers.
                    # However, to be safe, if NaNs *could* have been present, stick to float.
                    # A more robust int conversion requires all values to be finite.
                    if not numeric_series.isnull().any(): # Only if no NaNs at all
                         if (numeric_series % 1 == 0).all(): # Check if all are whole numbers
                            numeric_series = numeric_series.astype(pd.Int64Dtype()) # Use nullable int type
                         else:
                            logger.debug("Series contains non-integer values after numeric conversion. Returning float.")
    elif target_type == int and default_value is np.nan:
        # If target is int but default is NaN, it's tricky.
        # We can only convert to Int64Dtype if there are NO NaNs after conversion.
        if not numeric_series.isnull().any() and (numeric_series % 1 == 0).all():
            numeric_series = numeric_series.astype(pd.Int64Dtype())
        else:
             logger.debug("Cannot convert to int as NaNs are present or values are not whole numbers. Returning float.")


    return numeric_series


def robust_json_load(file_path: str) -> Optional[Union[Dict, List]]:
    """
    Loads a JSON file robustly, handling file not found and JSON decode errors.
    """
    if not os.path.exists(file_path):
        logger.error(f"JSON file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading JSON from {file_path}: {e}")
        return None

def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Creates a SHA256 hash of a Pandas DataFrame's string representation.
    Handles None or empty DataFrames gracefully.
    Designed to replace `hash_geodataframe` by working with any DataFrame.
    """
    if df is None:
        return None
    if df.empty:
        return "empty_dataframe" # Consistent hash for empty DataFrames
    try:
        # Convert to string, then hash. Ensure consistent sorting for reproducibility.
        # Sort by index and columns to ensure order doesn't affect hash.
        df_sorted = df.sort_index().sort_index(axis=1)
        df_string = df_sorted.to_csv(index=True, header=True, na_rep='_NaN_') # Using CSV for robust string rep
        return hashlib.sha256(df_string.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Could not create hash for DataFrame: {e}. Returning None.")
        return None

def convert_date_columns(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Converts specified columns in a DataFrame to datetime objects."""
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to datetime: {e}. Leaving as is or with NaT.")
        else:
            logger.debug(f"Date column '{col}' not found in DataFrame for conversion.")
    return df_copy

def standardize_missing_values(df: pd.DataFrame, string_cols: list[str], numeric_cols_defaults: dict[str, Any]) -> pd.DataFrame:
    """Standardizes missing values for string and numeric columns."""
    df_copy = df.copy()
    common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'Null', 'NULL']

    for col in string_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.strip()
            df_copy[col] = df_copy[col].replace(common_na_strings, 'Unknown', regex=False)
            df_copy[col] = df_copy[col].fillna('Unknown') # Ensure actual NaN also becomes 'Unknown'
        else:
            logger.debug(f"String column '{col}' for standardization not found in DataFrame.")

    for col, default_val in numeric_cols_defaults.items():
        if col in df_copy.columns:
            df_copy[col] = convert_to_numeric(df_copy[col], default_value=default_val)
        else:
            logger.debug(f"Numeric column '{col}' for standardization not found in DataFrame. Creating with default.")
            df_copy[col] = default_val # Add column with default if missing
    return df_copy
