import pandas as pd
import numpy as np
import logging
import json
import hashlib
import os
import re
from typing import Any, Optional, Union, List, Dict, Type
from pathlib import Path

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)

# --- Constants for Data Cleaning ---
# Common NA strings for robust replacement, defined as a frozenset for efficiency
COMMON_NA_STRINGS_SET = frozenset(['nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'nil', 'na', 'undefined', 'unknown', '-'])

# FIXED: Reconstructed and valid regex pattern from the corrupted original.
# This single regex is case-insensitive and matches either an empty/whitespace string OR any of the common NA strings.
_na_pattern_parts = [re.escape(s) for s in COMMON_NA_STRINGS_SET]
NA_REGEX_PATTERN = r'(?i)^\s*(?:' + '|'.join(_na_pattern_parts) + r'|)\s*$' if _na_pattern_parts else r'^\s*$'


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans DataFrame column names: converts to lowercase, replaces non-alphanumeric
    characters with underscores, and handles potential duplicate names.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}. Returning empty DataFrame.")
        return pd.DataFrame()
    if df.empty:
        return df.copy()

    df_cleaned = df.copy()
    try:
        original_columns = df_cleaned.columns.tolist()
        new_columns_base = []
        for col in original_columns:
            s = str(col).lower()
            s = re.sub(r'[^0-9a-zA-Z_]+', '_', s)
            s = re.sub(r'_+', '_', s)
            s = s.strip('_')
            s = s or f"unnamed_col_{original_columns.index(col)}"
            new_columns_base.append(s)
        
        final_new_columns = []
        col_counts: Dict[str, int] = {}
        for col_name in new_columns_base:
            if col_name in col_counts:
                col_counts[col_name] += 1
                final_new_columns.append(f"{col_name}_{col_counts[col_name]}")
            else:
                col_counts[col_name] = 0
                final_new_columns.append(col_name)
                
        df_cleaned.columns = final_new_columns
    except Exception as e:
        logger.error(f"Error cleaning column names: {e}. Original columns: {df.columns.tolist()}", exc_info=True)
        return df
    return df_cleaned


def convert_to_numeric(
    data_input: Any,
    default_value: Any = np.nan,
    target_type: Optional[Type[Union[float, int]]] = None
) -> Any:
    """
    Robustly converts input to a numeric pandas Series or scalar. Handles common NA strings.
    """
    is_series = isinstance(data_input, pd.Series)
    if not is_series:
        series_to_process = pd.Series(data_input, dtype=object)
    else:
        series_to_process = data_input.copy()

    if pd.api.types.is_object_dtype(series_to_process.dtype) and NA_REGEX_PATTERN:
        series_to_process.replace(NA_REGEX_PATTERN, np.nan, regex=True, inplace=True)

    numeric_series = pd.to_numeric(series_to_process, errors='coerce')

    if not pd.isna(default_value):
        numeric_series = numeric_series.fillna(default_value)

    if target_type is int:
        if numeric_series.isnull().any():
            try:
                numeric_series = numeric_series.astype(pd.Int64Dtype())
            except (TypeError, ValueError):
                pass  # Keep as float if conversion to nullable int fails
        else:
            try:
                non_na_series = numeric_series.dropna()
                if non_na_series.empty or (non_na_series % 1 == 0).all():
                    numeric_series = numeric_series.astype(int)
            except (TypeError, ValueError):
                pass # Keep as float if conversion to standard int fails
    elif target_type is float and not pd.api.types.is_float_dtype(numeric_series.dtype):
        numeric_series = numeric_series.astype(float)

    return numeric_series if is_series else (numeric_series.iloc[0] if not numeric_series.empty else default_value)


def robust_json_load(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """Loads JSON data from a file with robust error handling."""
    path_obj = Path(file_path)
    if not path_obj.is_file():
        logger.error(f"JSON file not found: {path_obj.resolve()}")
        return None
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {path_obj.resolve()}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading JSON from {path_obj.resolve()}: {e}", exc_info=True)
        return None


def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Creates a consistent SHA256 hash for a DataFrame for caching."""
    if df is None: return None
    if not isinstance(df, pd.DataFrame):
        return hashlib.sha256(str(df).encode('utf-8')).hexdigest()
    if df.empty:
        return hashlib.sha256(f"empty_dataframe_cols:{'_'.join(sorted(df.columns))}".encode()).hexdigest()
    try:
        df_sorted = df.reindex(sorted(df.columns), axis=1)
        return str(pd.util.hash_pandas_object(df_sorted, index=True).sum())
    except Exception as e:
        logger.warning(f"Standard DataFrame hashing failed: {e}. Falling back to less precise hash.", exc_info=True)
        try:
            summary = str(df.head(2).to_dict()) + str(df.shape) + str(df.columns.tolist())
            return hashlib.sha256(summary.encode('utf-8')).hexdigest()
        except Exception as e_fallback:
            logger.error(f"Fallback hashing also failed: {e_fallback}. Returning None.")
            return None


def convert_date_columns(df: pd.DataFrame, date_columns: List[str], errors: str = 'coerce') -> pd.DataFrame:
    """Converts specified columns in a DataFrame to datetime objects."""
    if not isinstance(df, pd.DataFrame): return df
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors=errors)
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to datetime: {e}.")
    return df_copy


def standardize_missing_values(
    df: pd.DataFrame,
    string_cols_defaults: Optional[Dict[str, str]] = None,
    numeric_cols_defaults: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Standardizes missing values in specified columns using defaults and regex replacement.
    """
    if not isinstance(df, pd.DataFrame): return df
    df_copy = df.copy()

    if string_cols_defaults:
        for col, default in string_cols_defaults.items():
            if col not in df_copy.columns:
                df_copy[col] = default
                continue
            series = df_copy[col].astype(object) # Ensure object type for replacement
            if NA_REGEX_PATTERN:
                series.replace(NA_REGEX_PATTERN, np.nan, regex=True, inplace=True)
            df_copy[col] = series.fillna(default).astype(str).str.strip()

    if numeric_cols_defaults:
        for col, default in numeric_cols_defaults.items():
            target_type = int if isinstance(default, int) and not pd.isna(default) else float
            if col not in df_copy.columns:
                df_copy[col] = default
            df_copy[col] = convert_to_numeric(df_copy[col], default_value=default, target_type=target_type)
            
    return df_copy
