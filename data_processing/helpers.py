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
    and strips leading/trailing underscores. Handles potential duplicate names.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return pd.DataFrame()

    df_copy = df.copy()
    try:
        new_columns_series = (
            pd.Series(df_copy.columns).astype(str) # Ensure all column names are strings
            .str.lower()
            .str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
            .str.replace(r'_+', '_', regex=True)
            .str.strip('_')
        )
        
        # Handle potential duplicate column names after cleaning
        if new_columns_series.duplicated().any():
            logger.warning("Duplicate column names found after cleaning. Appending suffixes to make them unique.")
            # Create a list of unique column names
            cols = []
            counts = {}
            for item in new_columns_series:
                if item in counts:
                    counts[item] += 1
                    cols.append(f"{item}_{counts[item]}")
                else:
                    counts[item] = 0
                    cols.append(item)
            new_columns = cols
        else:
            new_columns = new_columns_series.tolist()
            
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
    Robustly converts a pandas Series to a numeric type (float or nullable int).
    Non-convertible values are replaced with default_value.
    """
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series, dtype=object) # Convert to Series if not already
        except Exception as e:
            logger.warning(f"Could not convert input to Series in convert_to_numeric: {e}. Returning series of default values.")
            length = len(series) if hasattr(series, '__len__') else 1
            # Determine dtype for the default series based on default_value
            default_series_dtype = type(default_value) if default_value is not np.nan and not isinstance(default_value, (list, tuple, dict)) else float
            try:
                return pd.Series([default_value] * length, dtype=default_series_dtype)
            except TypeError: # Fallback if default_value itself is complex
                 return pd.Series([default_value] * length, dtype=object)


    # Coerce to numeric, making non-convertible values NaN
    numeric_series = pd.to_numeric(series, errors='coerce')

    # Fill NaNs (created by 'coerce' or originally present) with default_value if it's not np.nan
    if default_value is not np.nan:
        numeric_series = numeric_series.fillna(default_value)
    
    if target_type == int:
        # Attempt conversion to nullable integer type (Int64) if values are suitable
        # Suitable means: no NaNs left (unless default_value was NaN and all original NaNs were filled by it)
        # AND all non-NaN values are whole numbers.
        if numeric_series.isnull().any() and default_value is np.nan:
            # If NaNs are allowed (default_value is NaN) and still exist, cannot convert to standard int.
            # Keep as float, or if Int64 is acceptable and all non-NaN are whole numbers:
            if (numeric_series.dropna() % 1 == 0).all():
                 try:
                    numeric_series = numeric_series.astype(pd.Int64Dtype())
                 except Exception: # Broad exception for casting issues
                    logger.debug("Could not cast to Int64Dtype, likely due to mixed types or unhandled NaNs. Keeping as float.")
                    if numeric_series.dtype != float: numeric_series = numeric_series.astype(float) # Ensure float if not Int64
            else:
                # Contains non-integers or unfillable NaNs
                if numeric_series.dtype != float: numeric_series = numeric_series.astype(float)
        elif not numeric_series.isnull().any(): # No NaNs present
            if (numeric_series % 1 == 0).all(): # All are whole numbers
                try:
                    numeric_series = numeric_series.astype(pd.Int64Dtype())
                except Exception:
                    logger.debug("Could not cast to Int64Dtype. Keeping as float.")
                    if numeric_series.dtype != float: numeric_series = numeric_series.astype(float)
            else: # Contains non-integers
                logger.debug("Series contains non-integer values; cannot convert to int. Returning float.")
                if numeric_series.dtype != float: numeric_series = numeric_series.astype(float)
        else: # Has NaNs, but default_value was not NaN (so NaNs are filled)
              # This case implies all values are now non-NaN and default_value was int-like
            if (numeric_series % 1 == 0).all():
                try:
                    numeric_series = numeric_series.astype(pd.Int64Dtype())
                except Exception:
                    logger.debug("Could not cast to Int64Dtype. Keeping as float.")
                    if numeric_series.dtype != float: numeric_series = numeric_series.astype(float)
            else:
                logger.debug("Series contains non-integer values after fill; cannot convert to int. Returning float.")
                if numeric_series.dtype != float: numeric_series = numeric_series.astype(float)

    elif target_type == float and numeric_series.dtype != float : # Ensure it's float if target_type is float explicitly
        try:
            numeric_series = numeric_series.astype(float)
        except Exception as e:
             logger.warning(f"Could not cast series to float: {e}. Current dtype: {numeric_series.dtype}", exc_info=True)
             # Potentially return series as is or raise, depending on strictness required

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
        return None # Consistent hash for None
    if df.empty:
        # Consistent hash for any empty DataFrame (regardless of columns)
        return hashlib.sha256("empty_dataframe".encode('utf-8')).hexdigest()
    try:
        # Sort by index first, then by columns for canonical representation
        df_sorted_index = df.sort_index()
        df_sorted_columns = df_sorted_index.sort_index(axis=1)
        
        # Convert to CSV string. Using a consistent line terminator.
        # na_rep ensures NaNs are represented consistently.
        df_string = df_sorted_columns.to_csv(index=True, header=True, na_rep='_NaN_', lineterminator='\n')
        return hashlib.sha256(df_string.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Could not create hash for DataFrame: {e}.", exc_info=True)
        # Fallback hash for error cases, or return None if preferred
        return hashlib.sha256(f"dataframe_hashing_error:{str(e)}".encode('utf-8')).hexdigest()


def convert_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Converts specified columns in a DataFrame to datetime objects (pd.Timestamp).
       Columns not found are skipped with a debug log.
    """
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            try:
                # pd.to_datetime handles various formats, including already datetime objects
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to datetime: {e}. Original data in column may be preserved or set to NaT if unparseable.", exc_info=True)
        else:
            logger.debug(f"Date column '{col}' not found in DataFrame for conversion.")
    return df_copy

def standardize_missing_values(
    df: pd.DataFrame,
    string_cols_defaults: Dict[str, str], # Column name to default string value
    numeric_cols_defaults: Dict[str, Any] # Column name to default numeric value (or np.nan)
) -> pd.DataFrame:
    """Standardizes missing values for specified string and numeric columns.
       If a specified column does not exist, it's created with its default value.
    """
    df_copy = df.copy()
    # Broader list of common string representations of missing/null values
    common_na_strings = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'nil', 'na', 'undefined', 'unknown']

    for col, default_str_val in string_cols_defaults.items():
        if col in df_copy.columns:
            # Ensure the column is treated as string type first for robust replacement
            df_copy[col] = df_copy[col].astype(str).str.strip()
            # Replace common NA strings (case-insensitive for the strings in common_na_strings)
            # This requires regex=True for flags=re.IGNORECASE or a loop of str.replace
            # Simpler: convert column to lower, then replace, then restore (if case matters).
            # Or, use a regex that handles case for the search terms.
            # For now, use a loop for exact common_na_strings replacement.
            for na_s in common_na_strings:
                df_copy[col] = df_copy[col].str.replace(f"^{re.escape(na_s)}$", default_str_val, case=False, regex=True)
            df_copy[col] = df_copy[col].fillna(default_str_val) # Handle actual np.nan
        else:
            logger.debug(f"String column '{col}' for standardization not found. Creating with default '{default_str_val}'.")
            df_copy[col] = default_str_val

    for col, default_num_val in numeric_cols_defaults.items():
        target_num_type = int if isinstance(default_num_val, int) and default_num_val is not np.nan else float
        if col in df_copy.columns:
            df_copy[col] = convert_to_numeric(df_copy[col], default_value=default_num_val, target_type=target_num_type)
        else:
            logger.debug(f"Numeric column '{col}' for standardization not found. Creating with default {default_num_val}.")
            # Create series with default and attempt to cast to target_num_type if possible
            temp_series = pd.Series([default_num_val] * len(df_copy), index=df_copy.index)
            if target_num_type == int and default_num_val is not np.nan:
                try:
                    df_copy[col] = temp_series.astype(pd.Int64Dtype())
                except:
                    df_copy[col] = temp_series.astype(float) # Fallback to float
            else:
                 df_copy[col] = temp_series.astype(float)

    return df_copy
