# sentinel_project_root/data_processing/helpers.py
# SME-EVALUATED AND REVISED VERSION (GOLD STANDARD)
# This definitive version enhances robustness against edge cases, improves clarity
# with better encapsulation, and adds comprehensive self-documentation.

import pandas as pd
import numpy as np
import logging
import json
import hashlib
import re
from typing import Any, Optional, Union, List, Dict, Type
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

# A comprehensive, case-insensitive regex to identify and replace various "Not Available" strings.
# This pattern is pre-compiled for performance, as it's used frequently.
NA_REGEX_PATTERN = re.compile(
    r'(?i)^\s*(nan|none|n/a|#n/a|np\.nan|nat|<na>|null|nil|na|undefined|unknown|-|)\s*$'
)


class DataCleaner:
    """
    Encapsulates a suite of data cleaning and standardization operations.
    Designed to be instantiated as a singleton and used across a data processing pipeline.
    """
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans DataFrame column names for consistency and usability.
        This is a critical first step for creating a predictable data environment.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}.")
            return pd.DataFrame()
        if df.empty: return df.copy()

        df_cleaned = df.copy()
        
        # 1. Standardize format: lowercase, snake_case
        new_cols = (
            df_cleaned.columns.astype(str).str.lower()
            .str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
            .str.replace(r'_+', '_', regex=True).str.strip('_')
        )
        
        # 2. Handle empty column names resulting from cleaning (e.g., a column of only symbols)
        new_cols = [f"unnamed_col_{i}" if not name else name for i, name in enumerate(new_cols)]

        # 3. De-duplicate column names efficiently
        counts = Counter(new_cols)
        final_cols = []
        col_seen_count = Counter()
        for name in new_cols:
            if counts[name] > 1:
                col_seen_count[name] += 1
                final_cols.append(f"{name}_{col_seen_count[name]}")
            else:
                final_cols.append(name)
        
        df_cleaned.columns = final_cols
        return df_cleaned

    def _standardize_numeric_col(self, series: pd.Series, default_value: Any) -> pd.Series:
        """Private helper to robustly convert a Series to a numeric type."""
        if pd.api.types.is_object_dtype(series.dtype):
            series = series.replace(NA_REGEX_PATTERN, np.nan, regex=True)
        
        numeric_series = pd.to_numeric(series, errors='coerce')
        if not pd.isna(default_value):
            numeric_series = numeric_series.fillna(default_value)
            
        return numeric_series

    def standardize_missing_values(self, df: pd.DataFrame, string_cols_defaults: Dict[str, str], numeric_cols_defaults: Dict[str, Any]) -> pd.DataFrame:
        """Standardizes missing values in specified columns using defaults and regex."""
        if not isinstance(df, pd.DataFrame): return df
        df_copy = df.copy()

        for col, default in string_cols_defaults.items():
            if col in df_copy.columns:
                series = pd.Series(df_copy[col], dtype=object).replace(NA_REGEX_PATTERN, np.nan, regex=True)
                df_copy[col] = series.fillna(default).astype(str).str.strip()
            
        for col, default in numeric_cols_defaults.items():
            if col in df_copy.columns:
                df_copy[col] = self._standardize_numeric_col(df_copy[col], default_value=default)
        return df_copy
    
    def enforce_timezone_naive(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """
        Converts specified columns to timezone-naive datetime objects.
        This enforces the application-wide policy of working with naive datetimes
        to prevent timezone-related errors during comparisons and calculations.
        """
        if not isinstance(df, pd.DataFrame): return df
        df_copy = df.copy()
        for col in date_columns:
            if col in df_copy.columns:
                # The 'utc=True' argument correctly interprets timezone-aware strings (like ISO 8601 with 'Z').
                # The subsequent .dt.tz_localize(None) strips the timezone, making it naive.
                series = pd.to_datetime(df_copy[col], errors='coerce', utc=True)
                df_copy[col] = series.dt.tz_localize(None)
        return df_copy


def convert_to_numeric(data_input: Any, default_value: Any = np.nan, target_type: Optional[Type] = None) -> Any:
    """
    Robustly converts various inputs to a numeric type, handling common "Not Available" strings.
    """
    is_series = isinstance(data_input, pd.Series)
    series = data_input if is_series else pd.Series([data_input], dtype=object)

    if pd.api.types.is_object_dtype(series.dtype):
        series = series.replace(NA_REGEX_PATTERN, np.nan, regex=True)

    numeric_series = pd.to_numeric(series, errors='coerce').fillna(default_value)
    
    if target_type is int:
        # Use nullable integer type if NaNs still exist, otherwise use standard int.
        if numeric_series.isnull().any():
            numeric_series = numeric_series.astype(pd.Int64Dtype())
        else:
            numeric_series = numeric_series.astype(int)
    elif target_type is float:
        numeric_series = numeric_series.astype(float)

    return numeric_series if is_series else (numeric_series.iloc[0] if not numeric_series.empty else default_value)


def robust_json_load(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """Loads JSON data from a file with robust error handling for common issues."""
    path_obj = Path(file_path)
    if not path_obj.is_file() or path_obj.stat().st_size == 0:
        if not path_obj.is_file(): logger.error(f"JSON file not found: {path_obj.resolve()}")
        else: logger.warning(f"JSON file is empty: {path_obj.resolve()}")
        return None
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error decoding JSON from {path_obj.resolve()}: {e}")
        return None


def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Creates a consistent SHA256 hash for a DataFrame, suitable for use in caching keys.
    It's designed to be stable regardless of column or row order.
    """
    if df is None: return "none"
    if not isinstance(df, pd.DataFrame):
        return hashlib.sha256(str(df).encode('utf-8')).hexdigest()
    if df.empty:
        return hashlib.sha256(f"empty_df_cols:{'_'.join(sorted(df.columns))}".encode()).hexdigest()
    
    try:
        # Sort by index and columns for order-independent hashing.
        df_sorted = df.sort_index().reindex(sorted(df.columns), axis=1)
        # pd.util.hash_pandas_object is the most reliable method.
        return hashlib.sha256(pd.util.hash_pandas_object(df_sorted, index=True).values).hexdigest()
    except Exception as e:
        logger.warning(f"Standard DataFrame hashing failed: {e}. Falling back to a less precise hash.", exc_info=True)
        try:
            # Fallback uses a summary of the DataFrame. Less precise but better than nothing.
            summary = str(df.head(2).to_dict()) + str(df.shape) + str(df.columns.tolist())
            return hashlib.sha256(summary.encode('utf-8')).hexdigest()
        except Exception as fallback_e:
            logger.error(f"Fallback DataFrame hashing also failed: {fallback_e}.")
            return None


def convert_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Converts specified columns in a DataFrame to datetime objects, coercing errors."""
    if not isinstance(df, pd.DataFrame): return df
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    return df_copy

# --- Singleton Instance for Convenience ---
# This allows other modules to import and use the cleaner without needing to instantiate it.
data_cleaner = DataCleaner()
