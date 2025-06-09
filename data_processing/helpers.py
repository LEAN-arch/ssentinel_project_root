# sentinel_project_root/data_processing/helpers.py
# SME-EVALUATED AND REVISED VERSION (GOLD STANDARD V2)
# This version optimizes the performance of column name de-duplication and
# adds clarifying comments for complex logic and potential edge cases.

"""
A collection of robust, high-performance utility functions for common data
processing tasks like cleaning, type conversion, and file I/O.
"""
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

# <<< SME REVISION >>> Added a detailed comment to clarify the regex behavior and potential edge cases.
# A comprehensive, case-insensitive regex to identify and replace various "Not Available" strings.
# NOTE: This pattern is intentionally broad. The inclusion of `'-'` will match a hyphen only when
# it is the sole character in a cell (surrounded by optional whitespace). This is a common
# placeholder but could conflict with data where a single hyphen is a meaningful value.
# Use with awareness of the source data's conventions.
NA_REGEX_PATTERN = r'(?i)^\s*(nan|none|n/a|#n/a|np\.nan|nat|<na>|null|nil|na|undefined|unknown|-|)\s*$'


class DataCleaner:
    """
    Encapsulates a suite of data cleaning and standardization operations.
    Designed to be instantiated once and used across a data processing pipeline.
    """

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans DataFrame column names for consistency and usability.

        Transformations:
        - Converts to lowercase.
        - Replaces non-alphanumeric characters with underscores.
        - Collapses multiple underscores into one.
        - Strips leading/trailing underscores.
        - Appends a numeric suffix to de-duplicate column names.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with cleaned column names.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error(f"clean_column_names expects a pandas DataFrame, got {type(df)}.")
            return pd.DataFrame()
        if df.empty:
            return df.copy()

        df_cleaned = df.copy()
        try:
            new_cols = (df_cleaned.columns.astype(str).str.lower()
                        .str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
                        .str.replace(r'_+', '_', regex=True).str.strip('_'))

            new_cols = [f"unnamed_col_{i}" if not name else name for i, name in enumerate(new_cols)]

            # <<< SME REVISION >>> Replaced O(N^2) de-duplication with a more performant O(N) approach.
            # This is significantly faster for DataFrames with many columns or many duplicates.
            counts = Counter(new_cols)
            seen_counts = Counter()
            final_cols = []
            for name in new_cols:
                if counts[name] > 1:
                    seen_counts[name] += 1
                    final_cols.append(f"{name}_{seen_counts[name]}")
                else:
                    final_cols.append(name)

            df_cleaned.columns = final_cols
        except Exception as e:
            logger.error(f"Error cleaning column names: {e}", exc_info=True)
            return df
        return df_cleaned

    def standardize_missing_values(self, df: pd.DataFrame, string_cols_defaults: Dict[str, str], numeric_cols_defaults: Dict[str, Any]) -> pd.DataFrame:
        """
        Standardizes missing values in specified columns using defaults and regex replacement.

        Args:
            df (pd.DataFrame): The input DataFrame.
            string_cols_defaults (Dict[str, str]): A mapping of string columns to their default values.
            numeric_cols_defaults (Dict[str, Any]): A mapping of numeric columns to their default values.

        Returns:
            pd.DataFrame: A new DataFrame with standardized missing values.
        """
        if not isinstance(df, pd.DataFrame):
            return df
        df_copy = df.copy()

        for col, default in string_cols_defaults.items():
            if col in df_copy.columns:
                series = df_copy[col].astype(object).replace(NA_REGEX_PATTERN, np.nan, regex=True)
                df_copy[col] = series.fillna(default).astype(str).str.strip()

        for col, default in numeric_cols_defaults.items():
            if col in df_copy.columns:
                target_type = int if isinstance(default, int) else float
                df_copy[col] = convert_to_numeric(df_copy[col], default_value=default, target_type=target_type)
        return df_copy


def convert_to_numeric(data_input: Any, default_value: Any = np.nan, target_type: Optional[Type] = None) -> Any:
    """
    Robustly converts various inputs to a numeric pandas Series or scalar,
    handling common "Not Available" string representations.

    Args:
        data_input (Any): The data to convert (can be a scalar, list, or pandas Series).
        default_value (Any): The value to use for elements that cannot be converted. Defaults to np.nan.
        target_type (Optional[Type]): If `int` or `float`, attempts to cast the result to this type.

    Returns:
        Any: The converted data in the same format as the input (scalar or Series).
    """
    is_series = isinstance(data_input, pd.Series)
    series = data_input if is_series else pd.Series(data_input, dtype=object)

    if pd.api.types.is_object_dtype(series.dtype):
        series = series.replace(NA_REGEX_PATTERN, np.nan, regex=True)

    numeric_series = pd.to_numeric(series, errors='coerce')
    if not pd.isna(default_value):
        numeric_series = numeric_series.fillna(default_value)

    if target_type is int and pd.api.types.is_numeric_dtype(numeric_series.dtype):
        if numeric_series.isnull().any():
            # Use nullable Int64 if NaNs are present after filling.
            numeric_series = numeric_series.astype(pd.Int64Dtype())
        else:
            # Otherwise, use standard, more performant int.
            numeric_series = numeric_series.astype(int)
    elif target_type is float:
        numeric_series = numeric_series.astype(float)

    # <<< SME REVISION >>> Added a comment to clarify the return logic.
    # Return a Series if the input was a Series, otherwise extract the scalar value.
    return numeric_series if is_series else (numeric_series.iloc[0] if not numeric_series.empty else default_value)


def robust_json_load(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """Loads JSON data from a file with robust error handling and UTF-8 encoding."""
    path_obj = Path(file_file_path)
    if not path_obj.is_file():
        logger.error(f"JSON file not found: {path_obj.resolve()}")
        return None
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error decoding JSON from {path_obj.resolve()}: {e}")
        return None


def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Creates a consistent SHA256 hash for a DataFrame, suitable for caching."""
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        return hashlib.sha256(str(df).encode('utf-8')).hexdigest()
    if df.empty:
        return hashlib.sha256(f"empty_df_cols:{'_'.join(sorted(df.columns))}".encode()).hexdigest()
    try:
        # Use pandas' built-in, C-optimized hashing on sorted columns for consistency.
        df_sorted = df.reindex(sorted(df.columns), axis=1)
        return hashlib.sha256(pd.util.hash_pandas_object(df_sorted, index=True).values).hexdigest()
    except Exception as e:
        logger.warning(f"Standard DataFrame hashing failed: {e}. Falling back to a less precise hash.", exc_info=True)
        try:
            # Fallback is less precise but robust against unhashable dtypes.
            summary = str(df.head(2).to_dict()) + str(df.shape) + str(df.columns.tolist())
            return hashlib.sha256(summary.encode('utf-8')).hexdigest()
        except Exception as fallback_e:
            logger.error(f"Fallback DataFrame hashing also failed: {fallback_e}.")
            return None


def convert_date_columns(df: pd.DataFrame, date_columns: List[str], errors: str = 'coerce') -> pd.DataFrame:
    """Converts specified columns in a DataFrame to datetime objects, handling errors."""
    if not isinstance(df, pd.DataFrame):
        return df
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors=errors)
    return df_copy


# --- Singleton Instance for Convenience ---
data_cleaner = DataCleaner()
