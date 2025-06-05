# sentinel_project_root/data_processing/helpers.py
# Helper utilities for data processing tasks.

import pandas as pd
import numpy as np
import logging
import json
import hashlib
import os # For robust_json_load
from typing import Any, Optional, Union, List, Dict, Type

logger = logging.getLogger(__name__)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans DataFrame column names: converts to lowercase, replaces non-alphanumeric
    characters (except underscore) with underscores, collapses multiple underscores,
    and strips leading/trailing underscores.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return pd.DataFrame()

    df_copy = df.copy()
    try:
        new_columns = (
            df_copy.columns.astype(str) # Ensure all column names are strings
            .str.lower()
            .str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
            .str.replace(r'_+', '_', regex=True)
            .str.strip('_')
        )
        # Handle potential duplicate column names after cleaning
        if new_columns.duplicated().any():
            logger.warning("Duplicate column names found after cleaning. Appending suffixes.")
            new_columns = pd.Series(new_columns).groupby(level=0).transform(lambda x: x + '_' + pd.Series(range(len(x))).astype(str) if len(x) > 1 else x).tolist()
        df_copy.columns = new_columns
    except Exception as e:
        logger.error(f"Error cleaning column names: {e}. Columns: {df.columns.tolist()}", exc_info=True)
        return df # Return original on error
    return df_copy

def convert_to_numeric(
    series: pd.Series,
    default_value: Any = np.nan,
    target_type: Optional[Type[Union[float, int]]] = float
) -> pd.Series:
    """
    Robustly converts a pandas Series to a numeric type (float or int).
    Non-convertible values are replaced with default_value.
    """
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series, dtype=object) # Convert to Series if not already
        except Exception as e:
            logger.warning(f"Could not convert input to Series in convert_to_numeric: {e}. Returning series of default values.")
            length = len(series) if hasattr(series, '__len__') else 1
            return pd.Series([default_value] * length, dtype=type(default_value) if default_value is not np.nan else float)

    numeric_series = pd.to_numeric(series, errors='coerce')

    if default_value is not np.nan:
        numeric_series = numeric_series.fillna(default_value)
    
    if target_type == int:
        # Attempt conversion to nullable integer type if no NaNs or if default_value is int-compatible
        if not numeric_series.isnull().any() or (default_value is not np.nan and isinstance(default_value, int)):
            try:
                # Check if all values are whole numbers before casting to Int64
                if (numeric_series.dropna() % 1 == 0).all():
                    numeric_series = numeric_series.astype(pd.Int64Dtype())
                else:
                    logger.debug("Series contains non-integer values; cannot convert to int. Returning float.")
                    if numeric_series.dtype != float : numeric_series = numeric_series.astype(float) # Ensure float if not int
            except (TypeError, ValueError): # E.g., if default_value was float with decimal
                logger.debug("Could not convert series to int type. Returning float.")
                if numeric_series.dtype != float : numeric_series = numeric_series.astype(float)
        else: # NaNs present and default_value is NaN, cannot convert to int
            logger.debug("NaNs present and default_value is NaN; cannot convert to int. Returning float.")
            if numeric_series.dtype != float : numeric_series = numeric_series.astype(float)
    elif numeric_series.dtype != float and target_type == float: # Ensure it's float if target_type is float
        numeric_series = numeric_series.astype(float)

    return numeric_series


def robust_json_load(file_path: str) -> Optional[Union[Dict, List]]:
    """
    Loads a JSON file robustly, handling file not found and JSON decode errors.
    """
    if not os.path.exists(file_path): # os.path is fine here for simple check
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
        logger.error(f"An unexpected error occurred while loading JSON from {file_path}: {e}", exc_info=True)
        return None

def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Creates a SHA256 hash of a Pandas DataFrame's CSV string representation.
    Handles None or empty DataFrames gracefully. Sorts by index and columns.
    """
    if df is None:
        return None
    if df.empty:
        return hashlib.sha256("empty_dataframe".encode('utf-8')).hexdigest()
    try:
        df_sorted = df.sort_index().sort_index(axis=1)
        df_string = df_sorted.to_csv(index=True, header=True, na_rep='_NaN_', lineterminator='\n')
        return hashlib.sha256(df_string.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Could not create hash for DataFrame: {e}.", exc_info=True)
        return None

def convert_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Converts specified columns in a DataFrame to datetime objects (pd.Timestamp)."""
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to datetime: {e}. Original data kept or NaT if unparseable.", exc_info=True)
        else:
            logger.debug(f"Date column '{col}' not found in DataFrame for conversion.")
    return df_copy

def standardize_missing_values(
    df: pd.DataFrame,
    string_cols_defaults: Dict[str, str], # Column name to default string value
    numeric_cols_defaults: Dict[str, Any] # Column name to default numeric value (or np.nan)
) -> pd.DataFrame:
    """Standardizes missing values for specified string and numeric columns."""
    df_copy = df.copy()
    common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'Null', 'NULL', 'unknown']

    for col, default_str_val in string_cols_defaults.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.strip()
            df_copy[col] = df_copy[col].replace(common_na_strings, default_str_val, regex=False)
            df_copy[col] = df_copy[col].fillna(default_str_val)
        else:
            logger.debug(f"String column '{col}' for standardization not found. Creating with default '{default_str_val}'.")
            df_copy[col] = default_str_val

    for col, default_num_val in numeric_cols_defaults.items():
        if col in df_copy.columns:
            df_copy[col] = convert_to_numeric(df_copy[col], default_value=default_num_val)
        else:
            logger.debug(f"Numeric column '{col}' for standardization not found. Creating with default {default_num_val}.")
            df_copy[col] = default_num_val
    return df_copy
