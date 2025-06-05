# sentinel_project_root/data_processing/helpers.py
# Helper utilities for data processing tasks.

import pandas as pd
import numpy as np
import logging
import json
import hashlib
import os 
import re # Ensure 're' is imported for regular expressions
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
    if df.empty: # Handle empty DataFrame gracefully
        return pd.DataFrame(columns=df.columns) # Return with original columns if empty

    df_copy = df.copy()
    try:
        new_columns_series = (
            pd.Series(df_copy.columns).astype(str)
            .str.lower()
            .str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
            .str.replace(r'_+', '_', regex=True)
            .str.strip('_')
        )
        
        if new_columns_series.duplicated().any():
            logger.warning("Duplicate column names found after cleaning. Appending suffixes to make them unique.")
            cols = []
            counts: Dict[str, int] = {}
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
    if not isinstance(series, pd.Series):
        try: series = pd.Series(series, dtype=object)
        except Exception as e:
            logger.warning(f"Could not convert input to Series in convert_to_numeric: {e}.")
            length = len(series) if hasattr(series, '__len__') else 1
            default_dtype = type(default_value) if default_value is not np.nan and not isinstance(default_value, (list,dict,tuple)) else float
            try: return pd.Series([default_value] * length, dtype=default_dtype)
            except: return pd.Series([default_value] * length, dtype=object)

    numeric_series = pd.to_numeric(series, errors='coerce')
    if default_value is not np.nan: numeric_series = numeric_series.fillna(default_value)
    
    if target_type == int:
        can_convert_to_int64 = False
        # Check if all non-NaN values are whole numbers
        if not numeric_series.isnull().any(): 
            if (numeric_series % 1 == 0).all(): can_convert_to_int64 = True
        # If NaNs were filled by an integer default_value, it might be convertible
        elif default_value is not np.nan and isinstance(default_value, int):
             if (numeric_series % 1 == 0).all(): can_convert_to_int64 = True
        
        if can_convert_to_int64:
            try: numeric_series = numeric_series.astype(pd.Int64Dtype())
            except Exception: 
                logger.debug("Could not cast to Int64Dtype, keeping as float or original if already float.")
                if numeric_series.dtype != float : numeric_series = numeric_series.astype(float) # Ensure float if not Int64
        else: # Cannot convert to int (e.g. NaNs remain with NaN default, or non-integers present)
            logger.debug("Cannot convert to int (NaNs/non-integers). Returning float.")
            if numeric_series.dtype != float: numeric_series = numeric_series.astype(float)
    elif target_type == float and numeric_series.dtype != float : # Ensure float if target is float
        try: numeric_series = numeric_series.astype(float)
        except Exception as e: logger.warning(f"Could not cast series to float: {e}. Dtype: {numeric_series.dtype}", exc_info=True)
    return numeric_series

def robust_json_load(file_path: str) -> Optional[Union[Dict, List]]:
    if not os.path.exists(file_path): logger.error(f"JSON file not found: {file_path}"); return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        return data
    except json.JSONDecodeError as e: logger.error(f"Error decoding JSON from {file_path}: {e}"); return None
    except Exception as e: logger.error(f"Unexpected error loading JSON from {file_path}: {e}", exc_info=True); return None

def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None: return None
    if df.empty: return hashlib.sha256("empty_dataframe".encode('utf-8')).hexdigest()
    try:
        df_sorted = df.sort_index().sort_index(axis=1)
        df_string = df_sorted.to_csv(index=True, header=True, na_rep='_NaN_', lineterminator='\n')
        return hashlib.sha256(df_string.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Could not create hash for DataFrame: {e}.", exc_info=True)
        return hashlib.sha256(f"dataframe_hashing_error:{str(e)}".encode('utf-8')).hexdigest()

def convert_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            try: df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except Exception as e: logger.warning(f"Could not convert '{col}' to datetime: {e}.", exc_info=True)
        else: logger.debug(f"Date column '{col}' not found for conversion.")
    return df_copy

def standardize_missing_values(
    df: pd.DataFrame,
    string_cols_defaults: Dict[str, str],
    numeric_cols_defaults: Dict[str, Any]
) -> pd.DataFrame:
    df_copy = df.copy()
    common_na_strings = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'nil', 'na', 'undefined', 'unknown']
    valid_na_for_regex = [s for s in common_na_strings if s] # Filter out empty string if it was causing issues for regex
    na_regex_pattern = r'^(?:' + '|'.join(re.escape(s) for s in valid_na_for_regex) + r')$' if valid_na_for_regex else None

    for col, default_str_val in string_cols_defaults.items():
        if col not in df_copy.columns: # Create column if it doesn't exist
             df_copy[col] = default_str_val
             logger.debug(f"String col '{col}' not found. Creating with default '{default_str_val}'.")
             continue # Skip further processing for this new column

        # For existing columns
        df_copy[col] = df_copy[col].astype(str).str.strip() # Ensure string type and strip
        if na_regex_pattern: # Use regex for case-insensitive exact match of NA strings
            df_copy[col] = df_copy[col].replace(na_regex_pattern, default_str_val, regex=True)
        df_copy[col] = df_copy[col].fillna(default_str_val) # Handle actual np.nan after potential replacements

    for col, default_num_val in numeric_cols_defaults.items():
        target_num_type = int if isinstance(default_num_val, int) and default_num_val is not np.nan else float
        if col not in df_copy.columns: # Create column if it doesn't exist
            logger.debug(f"Numeric col '{col}' not found. Creating with default {default_num_val}.")
            # Create series with default and attempt to cast to target_num_type if possible
            temp_series_for_new_col = pd.Series([default_num_val] * len(df_copy), index=df_copy.index)
            if target_num_type == int and default_num_val is not np.nan:
                try: df_copy[col] = temp_series_for_new_col.astype(pd.Int64Dtype())
                except: df_copy[col] = temp_series_for_new_col.astype(float) # Fallback
            else: df_copy[col] = temp_series_for_new_col.astype(float)
            continue

        # For existing columns
        # If object type, first replace NA strings with np.nan for convert_to_numeric
        if pd.api.types.is_object_dtype(df_copy[col].dtype) and na_regex_pattern:
             df_copy[col] = df_copy[col].replace(na_regex_pattern, np.nan, regex=True)
        df_copy[col] = convert_to_numeric(df_copy[col], default_value=default_num_val, target_type=target_num_type)
    return df_copy
