# sentinel_project_root/pages/chw_components/epi_signals.py
Extracts epidemiological signals from CHW daily data for Sentinel Health Co-Pilot.
import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime
try:
from config import settings
from data_processing.helpers import convert_to_numeric
except ImportError as e:
logging.basicConfig(level=logging.ERROR)
logger_init = logging.getLogger(name)
logger_init.error(f"Critical import error in epi_signals.py: {e}. Ensure paths/dependencies are correct.")
raise
logger = logging.getLogger(name)
Common NA strings for robust replacement
COMMON_NA_STRINGS_EPI = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_EPI_PATTERN = r'^\s*
′
+
(
r
′
∣
(
?
:
′
+
′
∣
′
.
j
o
i
n
(
r
e
.
e
s
c
a
p
e
(
s
)
f
o
r
s
i
n
C
O
M
M
O
N
N
A
S
T
R
I
N
G
S
E
P
I
i
f
s
)
+
r
′
)
′
 +(r 
′
 ∣ 
(
 ?: 
′
 + 
′
 ∣ 
′
 .join(re.escape(s)forsinCOMMON 
N
​
 A 
S
​
 TRINGS 
E
​
 PIifs)+r 
′
 )
' if any(COMMON_NA_STRINGS_EPI) else '')
Pre-compile common regex patterns
SYMPTOM_KEYWORDS_PATTERN_EPI = re.compile(
r"\b(fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|short\s+of\s+breath|fatigue|dizzy|nausea)\b",
re.IGNORECASE
)
MALARIA_PATTERN_EPI = re.compile(r"\bmalaria\b", re.IGNORECASE)
TB_PATTERN_EPI = re.compile(r"\btb\b|tuberculosis", re.IGNORECASE)
def _prepare_epi_dataframe(
df: pd.DataFrame,
cols_config: Dict[str, Dict[str, Any]],
log_prefix: str
) -> pd.DataFrame:
"""Prepares the DataFrame for epi signal extraction: ensures columns exist, correct types, and handles NAs."""
df_prepared = df.copy()
for col_name, config in cols_config.items():
default_value = config["default"]
target_type_str = config["type"]
if col_name not in df_prepared.columns:
        if target_type_str == "datetime" and default_value is pd.NaT:
             df_prepared[col_name] = pd.NaT
        elif isinstance(default_value, (list, dict)):
             df_prepared[col_name] = [default_value.copy() for _ in range(len(df_prepared))]
        else:
             df_prepared[col_name] = default_value
    
    if target_type_str in [float, int, "datetime"] and pd.api.types.is_object_dtype(df_prepared[col_name].dtype):
        if NA_REGEX_EPI_PATTERN:
            try:
                df_prepared[col_name].replace(NA_REGEX_EPI_PATTERN, np.nan, regex=True, inplace=True)
            except Exception as e_regex:
                logger.warning(f"({log_prefix}) Regex NA replacement failed for '{col_name}': {e_regex}. Proceeding.")
    
    try:
        if target_type_str == "datetime":
            df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
        elif target_type_str == float:
            df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=float)
        elif target_type_str == int:
            df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=int)
        elif target_type_str == str:
            series = df_prepared[col_name].fillna(str(default_value))
            df_prepared[col_name] = series.astype(str).str.strip()
    except Exception as e_conv:
        logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
        if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
        else: df_prepared[col_name] = default_value
        
return df_prepared
