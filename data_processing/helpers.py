# sentinel_project_root/data_processing/helpers.py
# Helper utilities for data processing tasks.

import pandas as pd
import numpy as np
import logging
import json
import hashlib
import os 
import re 
from typing import Any, Optional, Union, List, Dict, Type
from pathlib import Path # <--- ADDED THIS IMPORT

logger = logging.getLogger(__name__)

# Common NA strings for robust replacement, defined as a frozenset for efficiency
COMMON_NA_STRINGS_SET = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'nil', 'na', 'undefined', 'unknown', '-'])
VALID_NA_FOR_REGEX = [s for s in COMMON_NA_STRINGS_SET if s] # Filter out empty string if problematic for regex
NA_REGEX_PATTERN = r'^(?:' + '|'.join(re.escape(s) for s in VALID_NA_FOR_REGEX) + r')$' if VALID_NA_FOR_REGEX else None


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans DataFrame column names: converts to lowercase, replaces non-alphanumeric
    characters (except underscore) with underscores, collapses multiple underscores,
    and strips leading/trailing underscores. Handles potential duplicate names by appending suffixes.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}. Returning empty DataFrame.")
        return pd.DataFrame() 
    
    if df.empty:
        logger.debug("clean_column_names received an empty DataFrame. Returning as is.")
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
            if not s: 
                s = f"unnamed_col_{original_columns.index(col)}"
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
    Converts input to numeric. If input is a Series, applies pd.to_numeric.
    Handles NaNs and attempts conversion to target_type (int or float).
    For int, uses nullable Int64Dtype if NaNs are present after coercion (and default_value is NaN).
    """
    is_series = isinstance(data_input, pd.Series)
    # Ensure series_to_process is a Series, even if input is scalar or list-like
    if not is_series:
        try:
            series_to_process = pd.Series(data_input, dtype=object if not isinstance(data_input, (int, float, bool)) else None)
        except Exception as e_series_create: # Handle potential errors creating series from unusual input
            logger.warning(f"Could not robustly convert input to Series in convert_to_numeric: {e_series_create}. Input type: {type(data_input)}. Returning default.")
            return default_value if not is_series else pd.Series([default_value] * (len(data_input) if hasattr(data_input, '__len__') else 1))
    else:
        series_to_process = data_input


    numeric_series = pd.to_numeric(series_to_process, errors='coerce')

    # Fill NaNs from coercion if default_value is not NaN itself
    if default_value is not np.nan:
        numeric_series = numeric_series.fillna(default_value)

    if target_type is int:
        if numeric_series.isnull().any() and default_value is np.nan: # NaNs exist and should be preserved
            try:
                numeric_series = numeric_series.astype(pd.Int64Dtype())
            except Exception:
                logger.debug("Cannot convert to Int64Dtype (possibly non-integer floats present even after coercion). Keeping as float.")
                if not pd.api.types.is_float_dtype(numeric_series.dtype):
                    numeric_series = numeric_series.astype(float)
        else: # No NaNs, or NaNs were filled
            try:
                # Check if all *non-NaN* values are whole numbers before attempting int conversion
                # This handles cases where default_value might be a float like 0.0
                non_na_series = numeric_series.dropna()
                if non_na_series.empty or (non_na_series % 1 == 0).all():
                    # If default_value was float (e.g. 0.0) and filled NaNs, astype(int) is fine
                    # If default_value was int (e.g. 0), astype(int) is fine
                    # If series originally had floats that are whole numbers, astype(int) is fine
                    numeric_series = numeric_series.astype(int) 
                else: # Contains non-integer numbers that were not NaNs
                    if not pd.api.types.is_float_dtype(numeric_series.dtype):
                         numeric_series = numeric_series.astype(float) 
            except Exception: 
                logger.debug("Failed to convert to standard int. Keeping as float.")
                if not pd.api.types.is_float_dtype(numeric_series.dtype):
                    numeric_series = numeric_series.astype(float)
                    
    elif target_type is float: 
        if not pd.api.types.is_float_dtype(numeric_series.dtype):
            try:
                numeric_series = numeric_series.astype(float)
            except Exception as e_float_cast:
                logger.warning(f"Could not cast series to float: {e_float_cast}. Dtype: {numeric_series.dtype}", exc_info=True)
    
    # If default_value was np.nan and it wasn't filled (e.g. target_type was float or Int64Dtype handled it)
    # and NaNs still exist from coercion, fill them now. This is somewhat redundant given the earlier fillna
    # if default_value is not np.nan, but ensures that if default_value IS np.nan, it's the final state for NaNs.
    if default_value is np.nan and numeric_series.isnull().any() and not pd.api.types.is_integer_dtype(numeric_series.dtype): # Don't fill pd.NA from Int64Dtype
        numeric_series = numeric_series.fillna(default_value)


    return numeric_series if is_series else numeric_series.iloc[0]


def robust_json_load(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """Loads JSON data from a file with robust error handling."""
    path_obj = Path(file_path) # Path() is now defined due to import
    if not path_obj.is_file(): 
        logger.error(f"JSON file not found: {path_obj.resolve()}")
        return None
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e_json_decode:
        logger.error(f"Error decoding JSON from {path_obj.resolve()}: {e_json_decode}")
        return None
    except Exception as e_json_load:
        logger.error(f"Unexpected error loading JSON from {path_obj.resolve()}: {e_json_load}", exc_info=True)
        return None


def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Creates a SHA256 hash for a Pandas DataFrame for caching purposes.
    Sorts by index and columns to ensure hash consistency.
    """
    if df is None:
        return None 
    if not isinstance(df, pd.DataFrame):
        logger.warning(f"hash_dataframe_safe received non-DataFrame input of type {type(df)}. Hashing string representation.")
        return hashlib.sha256(str(df).encode('utf-8')).hexdigest()
    if df.empty:
        empty_df_representation = "empty_dataframe_cols:" + "_".join(sorted(df.columns.astype(str)))
        return hashlib.sha256(empty_df_representation.encode('utf-8')).hexdigest()
    
    try:
        df_sorted_cols = df.reindex(sorted(df.columns), axis=1)
        df_sorted_all = df_sorted_cols.sort_index().sort_index(axis=1) 
        
        return str(pd.util.hash_pandas_object(df_sorted_all, index=True).sum())
    except Exception as e:
        logger.warning(f"Could not create standard hash for DataFrame: {e}. Falling back to less precise hash.", exc_info=True)
        try:
            summary_repr = str(df.head(2).to_dict()) + str(df.shape) + str(df.columns.tolist()) + str(df.dtypes.to_dict())
            return hashlib.sha256(summary_repr.encode('utf-8')).hexdigest()
        except Exception as e_fallback:
            logger.error(f"Fallback hashing also failed: {e_fallback}. Returning None.")
            return None


def convert_date_columns(df: pd.DataFrame, date_columns: List[str], errors: str = 'coerce') -> pd.DataFrame:
    """Converts specified columns in a DataFrame to datetime objects."""
    if not isinstance(df, pd.DataFrame):
        logger.warning("convert_date_columns: Input is not a DataFrame. Returning as is.")
        return df
    
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            if df_copy[col].isnull().all(): 
                logger.debug(f"Date column '{col}' contains all NaNs. Assigning pd.NaT.")
                df_copy[col] = pd.NaT 
                continue
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors=errors)
                # Optional: Make timezone-naive after conversion
                # if pd.api.types.is_datetime64_any_dtype(df_copy[col]) and df_copy[col].dt.tz is not None:
                #    logger.debug(f"Converting column '{col}' to timezone-naive.")
                #    df_copy[col] = df_copy[col].dt.tz_localize(None)
            except Exception as e_date_conv:
                logger.warning(f"Could not convert column '{col}' to datetime: {e_date_conv}. Original data kept if coercion failed.", exc_info=False) 
        else:
            logger.debug(f"Date column '{col}' not found for conversion in DataFrame.")
    return df_copy


def standardize_missing_values(
    df: pd.DataFrame,
    string_cols_defaults: Optional[Dict[str, str]] = None, 
    numeric_cols_defaults: Optional[Dict[str, Any]] = None 
) -> pd.DataFrame:
    """
    Standardizes missing values in specified columns.
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning("standardize_missing_values: Input is not a DataFrame. Returning as is.")
        return df
        
    df_copy = df.copy()
    
    if string_cols_defaults:
        for col, default_str_val in string_cols_defaults.items():
            if col not in df_copy.columns:
                 df_copy[col] = default_str_val
                 logger.debug(f"String col '{col}' not found in standardize_missing_values. Creating with default '{default_str_val}'.")
                 continue

            series = df_copy[col].astype(object) 
            if NA_REGEX_PATTERN:
                try:
                    series = series.replace(NA_REGEX_PATTERN, np.nan, regex=True) 
                except Exception as e_regex_str:
                    logger.warning(f"Regex NA replacement for string col '{col}' failed: {e_regex_str}. Proceeding.")
            df_copy[col] = series.fillna(default_str_val).astype(str).str.strip()

    if numeric_cols_defaults:
        for col, default_num_val in numeric_cols_defaults.items():
            target_num_type = int if isinstance(default_num_val, int) and default_num_val is not np.nan else float
            
            if col not in df_copy.columns:
                logger.debug(f"Numeric col '{col}' not found in standardize_missing_values. Creating with default {default_num_val}.")
                temp_series = pd.Series([default_num_val] * len(df_copy), index=df_copy.index)
                if target_num_type == int and default_num_val is not np.nan:
                    try: df_copy[col] = temp_series.astype(pd.Int64Dtype())
                    except: df_copy[col] = temp_series.astype(float) 
                else: df_copy[col] = temp_series.astype(float)
                continue

            series = df_copy[col]
            if pd.api.types.is_object_dtype(series.dtype) and NA_REGEX_PATTERN: 
                try:
                    series = series.replace(NA_REGEX_PATTERN, np.nan, regex=True)
                except Exception as e_regex_num:
                    logger.warning(f"Regex NA replacement for numeric col '{col}' failed: {e_regex_num}. Proceeding.")
            
            df_copy[col] = convert_to_numeric(series, default_value=default_num_val, target_type=target_num_type)
            
    return df_copy
