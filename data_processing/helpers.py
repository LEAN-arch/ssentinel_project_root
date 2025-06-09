# sentinel_project_root/data_processing/helpers.py
# SME PLATINUM STANDARD - CORE DATA UTILITIES & FLUENT PIPELINE

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Standalone Utility Functions ---

NA_VALUES_REGEX = re.compile(
    r'^\s*(-|n/a|na|none|null|nan|nat|<na>|undefined|unknown)\s*$',
    re.IGNORECASE
)

def convert_to_numeric(
    data: Any,
    target_type: Type = float,
    default_value: Optional[Any] = np.nan
) -> Union[pd.Series, Any]:
    """
    Robustly converts an object or Series to a specified numeric type.
    """
    is_series = isinstance(data, pd.Series)
    series = data if is_series else pd.Series([data])

    if pd.api.types.is_object_dtype(series.dtype):
        series = series.replace(NA_VALUES_REGEX, np.nan, regex=True)

    numeric_series = pd.to_numeric(series, errors='coerce')

    if pd.notna(default_value):
        numeric_series = numeric_series.fillna(default_value)
    
    if target_type is int:
        if numeric_series.isnull().any():
            numeric_series = numeric_series.astype(pd.Int64Dtype())
        else:
            numeric_series = numeric_series.astype(int)
    else:
        numeric_series = numeric_series.astype(float)
        
    return numeric_series if is_series else (numeric_series.iloc[0] if not numeric_series.empty else default_value)


def robust_json_load(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """Loads JSON from a file with robust error handling."""
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"JSON load failed: File not found at {path.resolve()}")
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
        logger.error(f"Error decoding JSON from {path.resolve()}: {e}")
        return None


def hash_dataframe(df: pd.DataFrame) -> str:
    """Creates a consistent SHA256 hash for a DataFrame, suitable for caching."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


# --- Fluent DataPipeline Class ---

class DataPipeline:
    """
    A fluent interface for applying a sequence of data processing operations.
    Enables expressive, readable, and chainable cleaning pipelines.
    """
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataPipeline must be initialized with a pandas DataFrame.")
        self.df = df.copy()

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the processed DataFrame."""
        return self.df

    def clean_column_names(self) -> 'DataPipeline':
        """
        Standardizes column names to snake_case, removes special characters,
        and ensures uniqueness with numbered suffixes.
        """
        if self.df.empty:
            return self

        cols = (self.df.columns.astype(str)
                .str.lower()
                .str.replace(r'[\s\W]+', '_', regex=True)
                .str.strip('_'))
        
        if cols.duplicated().any():
            counts = Counter()
            new_cols = []
            for name in cols:
                if counts[name] > 0:
                    new_cols.append(f"{name}_{counts[name]}")
                else:
                    new_cols.append(name)
                counts[name] += 1
            self.df.columns = new_cols
        else:
            self.df.columns = cols
            
        return self

    def standardize_missing_values(self, column_defaults: Dict[str, Any]) -> 'DataPipeline':
        """
        Replaces various NA representations with np.nan and then fills with
        provided defaults, inferring type from the default value.
        """
        for col, default_val in column_defaults.items():
            if col in self.df.columns:
                series = self.df[col]
                if isinstance(default_val, (int, float)):
                    target_type = int if isinstance(default_val, int) else float
                    self.df[col] = convert_to_numeric(series, target_type, default_val)
                else:
                    self.df[col] = series.replace(NA_VALUES_REGEX, np.nan, regex=True).fillna(str(default_val))
        return self

    def convert_date_columns(self, date_cols: List[str], errors: str = 'coerce') -> 'DataPipeline':
        """Converts specified columns to datetime objects, coercing errors to NaT."""
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors=errors)
        return self

    def rename_columns(self, rename_map: Dict[str, str]) -> 'DataPipeline':
        """Renames columns based on a provided mapping."""
        self.df.rename(columns=rename_map, inplace=True)
        return self

    def cast_column_types(self, dtype_map: Dict[str, Any]) -> 'DataPipeline':
        """Casts columns to specified data types with error handling."""
        for col, dtype in dtype_map.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not cast column '{col}' to {dtype}. Skipping. Error: {e}")
        return self

    def pipe(self, func: Callable, *args, **kwargs) -> 'DataPipeline':
        """Applies a custom function to the DataFrame in the pipeline."""
        self.df = func(self.df, *args, **kwargs)
        return self
