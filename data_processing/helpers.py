# ssentinel_project_root/data_processing/helpers.py
"""
A collection of reusable, low-level helper functions for data cleaning and preparation.
These functions are designed to be generic, robust, and used by the data loaders.
"""
import pandas as pd
import numpy as np
import logging
import json
import hashlib
import re
from typing import Any, Optional, Union, List, Dict, Type
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Constants for Data Cleaning ---
# A single, case-insensitive regex pattern to find common "not available" strings
# or empty/whitespace-only strings.
NA_REGEX_PATTERN = r'(?i)^\s*(nan|none|n/a|#n/a|np.nan|nat|<na>|null|nu|nil|na|undefined|unknown|-|)\s*$'


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans DataFrame column names: converts to snake_case, handles special characters,
    and de-duplicates resulting names. Returns a new DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    df_cleaned = df.copy()
    
    new_cols = {}
    for col in df_cleaned.columns:
        # 1. Convert to string and lowercase
        s = str(col).lower()
        # 2. Replace any non-alphanumeric character with an underscore
        s = re.sub(r'[^a-z0-9]+', '_', s)
        # 3. Collapse consecutive underscores
        s = re.sub(r'_+', '_', s)
        # 4. Remove leading/trailing underscores
        new_cols[col] = s.strip('_')

    df_cleaned.rename(columns=new_cols, inplace=True)

    # Handle potential duplicate column names after cleaning
    cols = pd.Series(df_cleaned.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df_cleaned.columns = cols
    
    return df_cleaned


def convert_to_numeric(
    data_input: Any,
    default_value: Any = np.nan,
    target_type: Optional[Type[Union[float, int]]] = None
) -> Any:
    """
    Robustly converts input (scalar or Series) to a numeric type, handling common NA strings.
    """
    is_series = isinstance(data_input, pd.Series)
    series = data_input if is_series else pd.Series([data_input])

    if pd.api.types.is_object_dtype(series.dtype):
        series = series.replace(NA_REGEX_PATTERN, np.nan, regex=True)

    numeric_series = pd.to_numeric(series, errors='coerce')
    numeric_series.fillna(default_value, inplace=True)

    if target_type is int:
        # Use nullable integer type if NaNs are possible, otherwise standard int
        if numeric_series.isnull().any():
            numeric_series = numeric_series.astype(pd.Int64Dtype())
        else:
            numeric_series = numeric_series.astype(int)
    
    return numeric_series if is_series else numeric_series.iloc[0]


def convert_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Converts specified columns in a DataFrame to datetime objects, coercing errors."""
    if not isinstance(df, pd.DataFrame):
        return df
        
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    return df_copy


def standardize_missing_values(
    df: pd.DataFrame,
    string_cols_defaults: Optional[Dict[str, str]] = None,
    numeric_cols_defaults: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Standardizes missing values in specified columns using defaults and regex replacement.
    """
    if not isinstance(df, pd.DataFrame):
        return df
        
    df_copy = df.copy()

    if numeric_cols_defaults:
        for col, default in numeric_cols_defaults.items():
            if col not in df_copy.columns:
                df_copy[col] = default
            else:
                target_type = int if isinstance(default, int) else float
                df_copy[col] = convert_to_numeric(df_copy[col], default_value=default, target_type=target_type)

    if string_cols_defaults:
        for col, default in string_cols_defaults.items():
            if col not in df_copy.columns:
                df_copy[col] = default
            else:
                # First, ensure it's an object type to allow for np.nan
                series = df_copy[col].astype(object)
                # Replace all NA-like strings with a true np.nan
                series.replace(NA_REGEX_PATTERN, np.nan, regex=True, inplace=True)
                # Now, fill any remaining np.nan values with the desired string default
                df_copy[col] = series.fillna(default).astype(str)
            
    return df_copy


def robust_json_load(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """Loads JSON data from a file with robust error handling."""
    path_obj = Path(file_path).resolve()
    if not path_obj.is_file():
        logger.error(f"JSON file not found: {path_obj}")
        return None
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading or parsing JSON from {path_obj}: {e}")
        return None


def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Creates a consistent and stable SHA256 hash for a DataFrame, suitable for caching.
    This method is more stable across library versions than other hashing methods.
    """
    if not isinstance(df, pd.DataFrame):
        return None
        
    # Sorting columns ensures a consistent hash regardless of original column order
    df_sorted = df.reindex(sorted(df.columns), axis=1)

    try:
        # Convert to a canonical JSON representation before hashing
        json_str = df_sorted.to_json(orient='split', date_format='iso', default_handler=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.error(f"Could not create a stable hash for the DataFrame: {e}", exc_info=True)
        return None
