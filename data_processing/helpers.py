# sentinel_project_root/data_processing/helpers.py
# SME PLATINUM STANDARD (V4 - Fluent API & Architectural Refinement)
# This version introduces a chainable `DataPipeline` class for more expressive
# and readable data processing workflows. It also simplifies the API for
# standardizing missing values.

"""
A collection of robust, high-performance utility functions and a fluent
DataPipeline class for common data processing tasks.
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

# --- Standalone Utility Functions ---

NA_REGEX_PATTERN = re.compile(
    r'(?i)^\s*(nan|none|n/a|#n/a|np\.nan|nat|<na>|null|nil|na|undefined|unknown|-|)\s*$'
)

def convert_to_numeric(data_input: Any, default_value: Any = np.nan, target_type: Optional[Type] = None) -> Any:
    """
    Robustly converts various inputs to a numeric pandas Series or scalar,
    handling common "Not Available" string representations.
    """
    # <<< SME REVISION V4 >>> This function is a true utility and remains standalone.
    # Its logic is already excellent.
    is_series = isinstance(data_input, pd.Series)
    series = data_input if is_series else pd.Series([data_input], dtype=object)

    if pd.api.types.is_object_dtype(series.dtype):
        # Using a pre-compiled regex is slightly more performant.
        series = series.replace(NA_REGEX_PATTERN, np.nan, regex=True)

    numeric_series = pd.to_numeric(series, errors='coerce')
    if not pd.isna(default_value):
        numeric_series = numeric_series.fillna(default_value)

    if target_type is int and pd.api.types.is_numeric_dtype(numeric_series.dtype):
        # Use pandas' nullable integer type if NaNs could exist, otherwise use standard int.
        numeric_series = numeric_series.astype(pd.Int64Dtype() if numeric_series.isnull().any() else int)
    elif target_type is float:
        numeric_series = numeric_series.astype(float)

    return numeric_series if is_series else (numeric_series.iloc[0] if not numeric_series.empty else default_value)


def robust_json_load(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """Loads JSON data from a file with robust error handling and UTF-8 encoding."""
    path_obj = Path(file_path)
    if not path_obj.is_file():
        logger.error(f"JSON load failed: File not found at {path_obj.resolve()}")
        return None
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error decoding JSON from {path_obj.resolve()}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {path_obj.resolve()}: {e}", exc_info=True)
        return None


def hash_dataframe_safe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Creates a consistent SHA256 hash for a DataFrame, suitable for caching."""
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        logger.warning(f"hash_dataframe_safe expected a DataFrame, got {type(df)}. Hashing string representation.")
        return hashlib.sha256(str(df).encode('utf-8')).hexdigest()
    if df.empty:
        # Consistent hash for an empty dataframe based on its columns
        col_string = '_'.join(sorted(df.columns))
        return hashlib.sha256(f"empty_df:{col_string}".encode()).hexdigest()
    try:
        # Use pandas' built-in, C-optimized hashing on sorted columns for consistency.
        df_sorted = df.reindex(sorted(df.columns), axis=1)
        return hashlib.sha256(pd.util.hash_pandas_object(df_sorted, index=True).values).hexdigest()
    except Exception as e:
        logger.warning(f"Standard DataFrame hashing failed: {e}. Falling back to a less precise hash.")
        try:
            # Fallback is less precise but robust against unhashable dtypes.
            summary = str(df.head(2).to_dict()) + str(df.shape) + str(df.columns.tolist())
            return hashlib.sha256(summary.encode('utf-8')).hexdigest()
        except Exception as fallback_e:
            logger.error(f"Fallback DataFrame hashing also failed: {fallback_e}.")
            return None


# <<< SME REVISION V4 >>> Introduce a fluent DataPipeline class for method chaining.
class DataPipeline:
    """
    A fluent interface for applying a sequence of data processing operations.

    Enables expressive, readable, and chainable cleaning pipelines.

    Usage:
        processed_df = (DataPipeline(raw_df)
                        .clean_column_names()
                        .convert_date_columns(['event_date'])
                        .standardize_missing_values({'age': 0, 'notes': 'unknown'})
                        .to_df())
    """
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataPipeline must be initialized with a pandas DataFrame.")
        self.df = df.copy()

    def to_df(self) -> pd.DataFrame:
        """Returns the processed DataFrame."""
        return self.df

    def clean_column_names(self) -> 'DataPipeline':
        """
        Cleans DataFrame column names for consistency and usability.
        (Lowercase, underscore-separated, no duplicates).
        """
        if self.df.empty:
            return self

        try:
            new_cols = (self.df.columns.astype(str).str.lower()
                        .str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
                        .str.replace(r'__+', '_', regex=True).str.strip('_'))
            
            # Handle potentially empty column names after cleaning
            new_cols = [f"unnamed_col_{i}" if not name else name for i, name in enumerate(new_cols)]

            # High-performance O(N) de-duplication
            counts = Counter(new_cols)
            if max(counts.values()) > 1:
                seen_counts = Counter()
                final_cols = []
                for name in new_cols:
                    if counts[name] > 1:
                        seen_counts[name] += 1
                        final_cols.append(f"{name}_{seen_counts[name]-1}") # 0-indexed suffix
                    else:
                        final_cols.append(name)
                self.df.columns = final_cols
            else:
                self.df.columns = new_cols

        except Exception as e:
            logger.error(f"Error cleaning column names: {e}", exc_info=True)
        return self

    def standardize_missing_values(self, default_values: Dict[str, Any]) -> 'DataPipeline':
        """
        Standardizes various "Not Available" formats to np.nan and then fills
        with provided defaults, inferring type from the default value.
        """
        # <<< SME REVISION V4 >>> Simplified API. No longer requires separate dicts.
        if not default_values:
            return self

        for col, default in default_values.items():
            if col in self.df.columns:
                if isinstance(default, (int, float, np.number)):
                    # Handle numeric columns
                    target_type = int if isinstance(default, int) else float
                    self.df[col] = convert_to_numeric(self.df[col], default_value=default, target_type=target_type)
                else:
                    # Handle string/object columns
                    series = self.df[col].astype(object).replace(NA_REGEX_PATTERN, np.nan, regex=True)
                    self.df[col] = series.fillna(str(default)).astype(str).str.strip()
        return self

    def convert_date_columns(self, date_columns: List[str], errors: str = 'coerce') -> 'DataPipeline':
        """Converts specified columns to datetime objects, coercing errors to NaT."""
        if not date_columns:
            return self
            
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors=errors)
        return self
