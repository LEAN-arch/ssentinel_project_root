# sentinel_project_root/data_processing/cached.py
# SME PLATINUM STANDARD - STREAMLIT CACHING LAYER

import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any

from .helpers import hash_dataframe
from .logic import calculate_clinic_kpis, calculate_environmental_kpis, calculate_trend

CACHE_TTL_SECONDS = 3600

@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_clinic_kpis(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Cached wrapper for calculate_clinic_kpis."""
    return calculate_clinic_kpis(df)

@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_environmental_kpis(iot_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Cached wrapper for calculate_environmental_kpis."""
    return calculate_environmental_kpis(iot_df)

@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_trend(df: Optional[pd.DataFrame], value_col: str, date_col: str, freq: str, agg_func: str) -> pd.Series:
    """Cached wrapper for calculating trends. Only accepts hashable (string) aggregations."""
    return calculate_trend(df, value_col, date_col, freq, agg_func)
