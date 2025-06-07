# sentinel_project_root/data_processing/helpers.py
# Core helper utilities for data processing tasks in the Sentinel project.

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
COMMON_NA_STRINGS_SET = frozenset([
    '', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'na', 'nat', '<na>',
    'null', 'nil', 'undefined', 'unknown', '-', 'not available'
])
NA_REGEX_PATTERN = r'^\s*$' + (
    r'|^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_SET if s) + r')$'
)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans DataFrame column names: converts to lowercase, replaces non-alphanumeric
    characters with underscores, and handles potential duplicate names.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return pd.DataFrame()

    new_cols, col_counts = [], {}
    for col in df.columns:
        s = str(col).lower().strip()
        s = re.sub(r'[^0-9a-z_]+', '_', s)
        s = re.sub(r'_+', '_', s).strip('_')
        if not s: s = "unnamed_col"
        
        if s in col_counts:
            col_counts[s] += 1
            new_cols.append(f"{s}_{col_counts[s]}")
        else:
            col_counts[s] = 0
            new_cols.append(s)
            
    df_cleaned = df.copy()
    df_cleaned.columns = new_cols
    return df_cleaned


def convert_to_numeric(
    data_input: Any,
    default_value: Any = np.nan,
    target_type: Optional[Type[Union[float, int]]] = None
) -> Any:
    """
    Robustly converts input data to a numeric type (int or float).
    Handles strings, NaNs, and ensures type consistency.
    """
    is_series = isinstance(data_input, pd.Series)
    series = data_input if is_series else pd.Series(data_input)
    
    # Use pre-compiled regex for efficient replacement of NA-like strings
    if pd.api.types.is_object_dtype(series.dtype):
        series = series.replace(NA_REGEX_PATTERN, np.nan, regex=True)

    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Fill any new NaNs with the specified default value
    numeric_series.fillna(default_value, inplace=True)
    
    # Attempt to cast to a specific integer or float type if requested
    if target_type is int:
        # Check if conversion to int is safe (no fractional parts)
        if (numeric_series % 1 == 0).all():
            return numeric_series.astype(int) if is_series else int(numeric_series.iloc[0])
    elif target_type is float:
        return numeric_series.astype(float) if is_series else float(numeric_series.iloc[0])
    
    return numeric_series if is_series else numeric_series.iloc[0]


def robust_json_load(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """Loads JSON data from a file with robust error handling."""
    path_obj = Path(file_path)
    if not path_obj.is_file():
        logger.error(f"JSON file not found: {path_obj.resolve()}")
        return None
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse JSON from {path_obj.resolve()}: {e}", exc_info=True)
        return None


def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Creates a consistent SHA256 hash for a Pandas DataFrame for caching.
    """
    if not isinstance(df, pd.DataFrame):
        return None
    try:
        # Sort by columns and use pandas' utility for a stable hash
        df_sorted = df.sort_index(axis=1)
        return str(pd.util.hash_pandas_object(df_sorted, index=True).sum())
    except Exception as e:
        logger.warning(f"Standard DataFrame hashing failed: {e}. Falling back to a less precise hash.", exc_info=True)
        try:
            # Fallback to a string representation of key dataframe properties
            summary = str(df.shape) + str(df.columns.tolist()) + str(df.head(1).to_dict())
            return hashlib.sha256(summary.encode('utf-8')).hexdigest()
        except Exception as e_fallback:
            logger.error(f"Fallback hashing also failed: {e_fallback}. Returning None.")
            return None


def standardize_missing_values(
    df: pd.DataFrame,
    string_cols_defaults: Optional[Dict[str, str]] = None,
    numeric_cols_defaults: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Standardizes missing values in a DataFrame using defined defaults.
    """
    if not isinstance(df, pd.DataFrame): return df
    df_copy = df.copy()

    if string_cols_defaults:
        for col, default_val in string_cols_defaults.items():
            if col not in df_copy.columns: df_copy[col] = default_val
            series = df_copy[col].astype(object).replace(NA_REGEX_PATTERN, np.nan, regex=True)
            df_copy[col] = series.fillna(default_val).astype(str)

    if numeric_cols_defaults:
        for col, default_val in numeric_cols_defaults.items():
            target_type = int if isinstance(default_val, int) else float
            if col not in df_copy.columns: df_copy[col] = default_val
            df_copy[col] = convert_to_numeric(df_copy[col], default_value=default_val, target_type=target_type)
            
    return df_copy
