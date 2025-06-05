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
    if not isinstance(df, pd.DataFrame):
        logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return pd.DataFrame()
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
            logger.warning("Duplicate column names after cleaning. Appending suffixes.")
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
        return df
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
        can_convert = False
        if not numeric_series.isnull().any(): # No NaNs at all
            if (numeric_series % 1 == 0).all(): can_convert = True
        elif default_value is not np.nan and isinstance(default_value, int): # NaNs filled with int
             if (numeric_series % 1 == 0).all(): can_convert = True
        
        if can_convert:
            try: numeric_series = numeric_series.astype(pd.Int64Dtype())
            except: logger.debug("Could not cast to Int64Dtype, keeping float."); numeric_series = numeric_series.astype(float) if numeric_series.dtype != float else numeric_series
        else:
            if numeric_series.dtype != float: numeric_series = numeric_series.astype(float)
    elif target_type == float and numeric_series.dtype != float:
        try: numeric_series = numeric_series.astype(float)
        except Exception as e: logger.warning(f"Could not cast to float: {e}. Dtype: {numeric_series.dtype}", exc_info=True)
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
    valid_na_for_regex = [s for s in common_na_strings if s]
    na_regex = r'^(?:' + '|'.join(re.escape(s) for s in valid_na_for_regex) + r')$' if valid_na_for_regex else None

    for col, default_str in string_cols_defaults.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.strip()
            if na_regex: df_copy[col] = df_copy[col].replace(na_regex, default_str, regex=True) # Case-insensitive handled by common_na having lower
            df_copy[col] = df_copy[col].fillna(default_str)
        else:
            logger.debug(f"String col '{col}' not found. Creating with default '{default_str}'.")
            df_copy[col] = default_str

    for col, default_num in numeric_cols_defaults.items():
        num_type = int if isinstance(default_num, int) and default_num is not np.nan else float
        if col in df_copy.columns:
            if pd.api.types.is_object_dtype(df_copy[col].dtype) and na_regex:
                 df_copy[col] = df_copy[col].replace(na_regex, np.nan, regex=True)
            df_copy[col] = convert_to_numeric(df_copy[col], default_value=default_num, target_type=num_type)
        else:
            logger.debug(f"Numeric col '{col}' not found. Creating with default {default_num}.")
            series_default = pd.Series([default_num] * len(df_copy), index=df_copy.index)
            if num_type == int and default_num is not np.nan:
                try: df_copy[col] = series_default.astype(pd.Int64Dtype())
                except: df_copy[col] = series_default.astype(float)
            else: df_copy[col] = series_default.astype(float)
    return df_copy
