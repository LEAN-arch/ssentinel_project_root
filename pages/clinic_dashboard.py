# sentinel_project_root/pages/clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta # Import date directly
from typing import Optional, Dict, Any, Tuple, List
import os
from pathlib import Path # For path operations

# --- Sentineldashboard.py` is now occurring in `clinic_dashboard.py`.

The traceback snippet isn't showing the exact failing import line within `clinic_dashboard.py`, but the error message "attempted relative import with no known parent package System Imports (Absolute Imports from Project Root) ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_" is the key. It means an import like `from .some_module import ...` or `from ..some_other_module import ...` is being used in a context where Python doesn't recognize the current file's directory (`pages/`) as part of a package that can resolve such relative paths.

**The Fix (Same as for `chw_dashboard.py`):**

We need to change the relative imports within `ssentinel_project_root/pages/clinic_dashboard.py` to absolute imports, assuming the project root (`ssentinel_project_root`) has been added to `sys.path` by `app.py`.

For example, an import like:
`from .clinic_components.env_details import prepare_clinic_environmental_detail_data`

should become:
`from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data`

Let's apply this tobar_chart
    
    # Clinic specific components using absolute imports from 'pages' package
    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e_clinic_dash_abs: # Unique exception variable name
    import sys 
    _current_file_clinic = Path(__file__).resolve()
    _pages_dir_clinic = _current_file_clinic.parent
    _project_root_clinic_assumption = _pages_dir_clinic.parent

    error_msg_clinic_detail = (
        f"Clinic Dashboard Import Error (using absolute imports): {e_ `clinic_dashboard.py`.

File 50: `ssentinel_project_root/pages/clinic_dashboard.py`
```python
# sentinel_project_root/pages/clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta # Keep 'date' for type hints
from typing import Optional, Dict, Any, Tuple, List
import os # For os.path.exists
from pathlib import Path # For path operations

# --- Sentinel System Imports (Absolute Imports from Project Root) ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart
    
    # Clinic specific components using absolute imports from 'pages' package
    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_clinic_dash_abs}. "
        f"Ensure project root ('{_project_root_clinic_assumption}') is in sys.path (done by app.py) "
        f"and all modules/packages (e.g., 'pages', 'pages.clinic_components') have `__init__.py` files. "
        f"Check for typos in import paths. Current Python Path: {sys.path}"
    )
    try:
        st.error(error_msg_clinic_detail)
        st.stop()
    except NameError: 
        print(error_msg_clinic_detail, file=sys.stderr)
        raise

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Page Title and Introduction ---data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e_clinic_dash_abs: # Unique exception variable name
    import sys
    _current_file_clinic = Path(__file__).resolve()
    _pages_dir_clinic = _current_file_clinic.parent
    _project_root_clinic_assumption = _pages_dir_clinic.parent 

    error_msg_clinic_detail = (
        f"Clinic Dashboard Import Error (using absolute imports): {e_clinic_dash_abs}. "
        f"Ensure project root ('{_project_root_clinic_assumption}') is in sys.path (done by app.py) "
        f"and all modules/packages (e.g., 'pages', 'pages.clinic_components') have `__init__.py` files. "
        f"Check for typos in import paths. Current Python Path: {sys.path}"
    )
    try:
        st.error(error_msg_clinic_detail)
        st.stop()

st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

# --- Data Loading Function for this Dashboard ---
@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading comprehensive clinic operational dataset...",
    hash_funcs={pd.DataFrame: lambda df_cache: pd.util.hash_pandas_object(df_cache, index=True) if isinstance(df_cache, pd.DataFrame) else hash(df_cache)}
)
def get_clinic_console_processed_data(
    selected_period_start_date: date, # Use 'date' directly
    selected_period_end_date: date    # Use 'date'
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], bool]:
    log_ctx = "ClinicConsoleDataLoad"
    logger.info(f"({log_ctx}) Loading data for period: {selected_period_start_date.isoformat()} to {selected_period_end_date.isoformat()}")
    
    raw_health_df = load_health_records(source_context=f"{log_ctx}/LoadRawHealthRecs")
    raw_iot_df = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadRawIoTData")
    
    iot_source_file_path = Path(settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH)
    if not iot_source_file_path.is_absolute(): iot_source_file_path = (Path(settings.PROJECT_ROOT_DIR) / settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH).resolve()
    iot_source_file_exists = iot_source_file_path.exists()
    iot_data_loaded_ok = isinstance(raw_iot_df, pd.DataFrame) and not raw_iot_df.empty
    is_iot_data_available_flag = iot_source_file_exists and iot_data_loaded_ok

    ai_enriched_health_df_full: Optional[pd.DataFrame] = None
    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        ai_enriched_health_df_full, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{    except NameError:
        print(error_msg_clinic_detail, file=sys.stderr)
        raise

logger = logging.getLogger(__name__)

st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading comprehensive clinic operational dataset...",
    hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df, index=True) if isinstance(df, pd.DataFrame) else hash(df)}
)
def get_clinic_console_processed_data(
    selected_period_start_date: date,
    selected_period_end_date: date
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], bool]:
    log_ctx = "ClinicConsoleDataLoad"
    logger.info(f"({log_ctx}) Loading data for period: {selected_period_start_date.isoformat()} to {selected_period_end_date.isoformat()}")
    
    raw_health_df = load_health_records(source_context=f"{log_ctx}/LoadRawHealthRecs")
    raw_iot_df = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadRawIoTData")
    
    iot_source_file = Path(settings.PROJECT_ROOT_DIR) / settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH
    iot_source_exists = iot_source_file.exists() and iot_source_file.is_file()
    iot_data_loaded = isinstance(raw_iot_df, pd.DataFrame) and not raw_iot_df.empty
    is_iot_available = iot_source_exists and iot_data_loaded

    ai_enriched_health_df_full: Optional[pd.DataFrame] = None
    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        ai_enriched_health_df_full, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrichHealth")
    else:
        logger.warning(f"({log_ctx}) Raw health data for clinic empty. AI enrichment skipped.")
        ai_enriched_health_df_full = pd.DataFrame()

    df_health_period: pd.DataFrame = pd.DataFrame()
    if isinstance(ai_enriched_health_df_full, pd.DataFrame) and not ai_enriched_health_df_full.empty and 'encounter_date' in ai_enriched_health_df_full.columns:
        if not pd.api.types.log_ctx}/AIEnrichHealth")
    else:
        logger.warning(f"({log_ctx}) Raw health data empty. AI enrichment skipped.")
        ai_enriched_health_df_full = pd.DataFrame()

    df_health_period: pd.DataFrame = pd.DataFrame()
    if isinstance(ai_enriched_health_df_full, pd.DataFrame) and not ai_enriched_health_df_full.empty and 'encounter_date' in ai_enriched_health_df_full.columns:
        if not pd.api.types.is_datetime64_any_dtype(ai_enriched_health_df_full['encounter_date']):
            ai_enriched_health_df_full['encounter_date'] = pd.to_datetime(ai_enriched_health_df_full['encounter_date'], errors='coerce')
        df_health_period = ai_enriched_health_df_full[
            (ai_enriched_health_df_full['encounter_date'].notna()) &
            (ai_enriched_health_df_full['encounter_date'].dt.date >= selected_period_start_date) &
            (ai_enriched_health_df_full['encounter_date'].dt.date <= selected_period_end_date)
        ].copy()
    
    df_iot_period: pd.DataFrame = pd.DataFrame()
    if is_iot_data_available_flag and isinstance(raw_iot_df, pd.DataFrame) and 'timestamp' in raw_iot_df.columns: # Check raw_iot_df type
        if not pd.api.types.is_datetime64_any_dtype(raw_iot_df['timestamp']):
            raw_iot_df['timestamp'] = pd.to_datetime(raw_iot_df['timestamp'], errors='coerce')
        dfis_datetime64_any_dtype(ai_enriched_health_df_full['encounter_date']):
            ai_enriched_health_df_full['encounter_date'] = pd.to_datetime(ai_enriched_health_df_full['encounter_date'], errors='coerce')
        df_health_period = ai_enriched_health_df_full[
            (ai_enriched_health_df_full['encounter_date'].notna()) &
            (ai_enriched_health_df_full['encounter_date'].dt.date >= selected_period_start_date) &
            (ai_enriched_health_df_full['encounter_date'].dt.date <= selected_period_end_date)
        ].copy()
    
    df_iot_period: pd.DataFrame = pd.DataFrame()
    if is_iot_available and isinstance(raw_iot_df, pd.DataFrame) and 'timestamp' in raw_iot_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(raw_iot_df['timestamp']):
            raw_iot_df['timestamp'] = pd.to_datetime(raw_iot_df['timestamp'], errors='coerce')
        df_iot_period = raw_iot_df[
            (raw_iot_df['timestamp'].notna()) &
            (raw_iot_df['timestamp'].dt.date >= selected_period_start_date) &
            (raw_iot_df['timestamp'].dt.date <= selected_period_end_date)
        ].copy()

    clinic_summary_kpis: Dict[str, Any] = {"test_summary_details": {}} # Default structure
    if not df_health_period.empty:
        try:
            clinic_summary_kpis = get_clinic_summary_kpis(df_health_period, f"{log_ctx}/PeriodSummaryKPIs")
        except Exception as e_sum_kpi:
            logger.error(f"({log_ctx}) Error calculating clinic summary KPIs: {e_sum_kpi}", exc_info=True)
    else: logger.info(f"({log_ctx}) No health data in period for clinic summary KPIs.")
    return ai_enriched_health_df_full, df_health_period, df_iot_period, clinic_summary_kpis, is_iot_available

# --- Sidebar Filters ---
logo_path_sidebar_clinic = Path(settings.PROJECT_ROOT_DIR) / settings.APP_LOGO_SMALL_PATH
if logo_path_sidebar_clinic.exists() and logo_path_sidebar_clinic.is_file(): st.sidebar.image(str(logo_path_sidebar_clinic), width=120)
st.sidebar.header("Console Filters")

abs_min_clinic = date.today() - timedelta(days=365); abs_max_clinic = date.today()
def_end_clinic = abs_max_clinic
def_start_clinic = max(abs_min_clinic, def_end_clinic -_iot_period = raw_iot_df[
            (raw_iot_df['timestamp'].notna()) &
            (raw_iot_df['timestamp'].dt.date >= selected_period_start_date) &
            (raw_iot_df['timestamp'].dt.date <= selected_period_end_date)
        ].copy()

    clinic_kpis_period: Dict[str, Any] = {"test_summary_details": {}} # Default structure
    if not df_health_period.empty:
        try: clinic_kpis_period = get_clinic_summary_kpis(df_health_period, f"{log_ctx}/PeriodSummaryKPIs")
        except Exception as e_kpi: logger.error(f"({log_ctx}) Error calculating clinic summary KPIs: {e_kpi}", exc_info=True)
    else: logger.info(f"({log_ctx}) No health data in period for clinic summary KPIs.")
    
    return ai_enriched_health_df_full, df_health_period, df_iot_period, clinic_kpis_period, is_iot_data_available_flag

# --- Sidebar Filters ---
logo_path_sidebar_clinic = Path(settings.APP_LOGO_SMALL_PATH)
if not logo_path_sidebar_clinic.is_absolute(): logo_path_sidebar_clinic = (Path(settings.PROJECT_ROOT_DIR) / settings.APP_LOGO_SMALL_PATH).resolve()
if logo_path_sidebar_clinic.exists(): st.sidebar.image(str(logo_path_sidebar_clinic), width=120)
st.sidebar.header("Console Filters")

abs_min_clinic = date.today() - timedelta(days=365); abs_max_clinic = date.today()
def_end_clinic = abs_max_clinic
def_start_clinic = max(abs_min_clinic, def_end_clinic - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1))

date_range_key_clinic_ss = "clinic_console_date_range_v2" # Unique session state key
if date_range_key_clinic_ss not in st.session_state: st.session_state[date_range_key_clinic_ss] = [def_start_clinic, def_end_clinic]
selected_range_clinic = st.sidebar.date_input("Select Date Range for Clinic Review:", value=st.session_state[date_range_key_clinic_ss], min_value=abs_min_clinic, max_value=abs_max_clinic, key=f"{date_range_key_clinic_ss}_widget")

start_date_clinic_filt: date; end_date_clinic_filt: date # Explicitly type
if isinstance(selected_range timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1))
date_range_key_clinic_ss = "clinic_console_date_range_v2"
if date_range_key_clinic_ss not in st.session_state: st.session_state[date_range_key_clinic_ss] = [def_start_clinic, def_end_clinic]

selected_range_clinic_ui = st.sidebar.date_input("Select Date Range for Clinic Review:", value=st.session_state[date_range_key_clinic_ss], min_value=abs_min_clinic, max_value=abs_max_clinic, key=f"{date_range_key_clinic_ss}_widget")
start_date_clinic_filt: date; end_date_clinic_filt: date # Type hint
if isinstance(selected_range_clinic_ui, (list, tuple)) and len(selected_range_clinic_ui) == 2:
    st.session_state[date_range_key_clinic_ss] = selected_range_clinic_ui
    start_date_clinic_filt, end_date_clinic_filt = selected_range_clinic_ui
else: 
    start_date_clinic_filt, end_date_clinic_filt = st.session_state[date_range_key_clinic_ss]
    st.sidebar.warning("Date range selection error. Using previous/default.")
if start_date_clinic_filt > end_date_clinic_filt:
    st.sidebar.error("Start date must be <= end date. Adjusting end date."); end_date_clinic_filt = start_date_clinic_filt
    st.session_state[date_range_key_clinic_ss][1] = end_date_clinic_filt

MAX_QUERY_DAYS_CLINIC_CONSOLE = 90 # Max query range
if (end_date_clinic_filt - start_date_clinic_filt).days >= MAX_QUERY_DAYS_CLINIC_CONSOLE : # Use >= to include the max day itself
    st.sidebar.warning(f"Date range large. Limiting to {MAX_QUERY_DAYS_CLINIC_CONSOLE} days from start for performance.")
    end_date_clinic_filt = start_date_clinic_filt + timedelta(days=MAX_QUERY_DAYS_CLINIC_CONSOLE -1)
    if end_date_clinic_filt > abs_max_clinic: end_date_clinic_filt = abs_max_clinic
    st.session_state[date_range_key_clinic_ss] = [start_date_clinic_filt, end_date_clinic_filt]

# --- Load Data ---
current_period_str_clinic = f"{start_date_clinic_filt.strftime('%d %b %Y')} - {end_date_clinic_filt.strftime('%d %b %Y')}"
full_hist_health_df, health_df_period, iot_df_period, clinic_summary_kpis_data, iot_available = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"test_summary_details": {}}, False
try:
    full_hist_health_df, health_df_period, iot_df_period, clinic_summary_kpis_data, iot_available = get_clinic_console_processed_data(start_date_clinic_filt, end_date_clinic_filt)
except Exception as e_load_clinic:
    logger.error(f"Clinic Dashboard: Main data loading failed: {e_load_clinic}", exc_info=True)
    st.error(f"Error loading clinic dashboard data: {e_load_clinic}. Check logs.")
if not iot_available: st.sidebar.warning("IoT environmental data source unavailable. Some metrics may be missing.")
st.info(f"Displaying Clinic Console data for period: **{current_period_str_clinic}**")

# --- Section 1: Top-Level KPIs ---
st.header("🚀 Clinic Performance & Environment Snapshot")
_clinic, (list, tuple)) and len(selected_range_clinic) == 2:
    st.session_state[date_range_key_clinic_ss] = selected_range_clinic
    start_date_clinic_filt, end_date_clinic_filt = selected_range_clinic
else: start_date_clinic_filt, end_date_clinic_filt = st.session_state[date_range_key_clinic_ss]; st.sidebar.warning("Date range error.")
if start_date_clinic_filt > end_date_clinic_filt: st.sidebar.error("Start date must be <= end date."); end_date_clinic_filt = start_date_clinic_filt; st.session_state[date_range_key_clinic_ss][1] = end_date_clinic_filt

MAX_QUERY_DAYS_CLINIC_CONSOLE = 90 
if (end_date_clinic_filt - start_date_clinic_filt).days +1 > MAX_QUERY_DAYS_CLINIC_CONSOLE: # Ensure range includes end date
    st.sidebar.warning(f"Date range limited to {MAX_QUERY_DAYS_CLINIC_CONSOLE} days for performance.")
    end_date_clinic_filt = start_date_clinic_filt + timedelta(days=MAX_QUERY_DAYS_CLINIC_CONSOLE -1)
    if end_date_clinic_filt > abs_max_clinic: end_date_clinic_filt = abs_max_clinic
    st.session_state[date_range_key_clinic_ss] = [start_date_clinic_filt, end_date_clinic_filt]

# --- Load Data ---
current_period_str_clinic = f"{start_date_clinic_filt.strftime('%d %b %Y')} - {end_date_clinic_filt.strftime('%d %b %Y')}"
full_hist_health_df, health_df_period_clinic, iot_df_period_clinic, clinic_summary_kpis, iot_available = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"test_summary_details": {}}, False
try:
    full_hist_health_df, health_df_period_clinic, iot_df_period_clinic, clinic_summary_kpis, iot_available = get_clinic_console_processed_data(start_date_clinic_filt, end_date_clinic_filt)
except Exception as e_load_clinic:
    logger.error(f"Clinic Dashboard: Main data loading failed: {e_load_clinic}", exc_info=True)
    st.error(f"Error loading clinic dashboard data: {e_load_clinic}. Please check logs.")
if not iot_available: st.sidebar.warning("IoT environmental data source unavailable. Some metrics may be missing.")
st.info(f"Displaying Clinic Console data for period: **{current_period_str_clinic}**")

# --- Section 1: Top-Level KPIs ---
st.header("🚀 Clinic Performance & Environment Snapshot")
if clinic_summary_kpis and isinstance(clinic_summary_kpis.get("test_summary_details"), dict):
    main_kpis_clinic = structure_main_clinic_kpis(clinic_summary_kpis, current_period_str_clinic)
    disease_kpis_clinic = structure_disease_specific_clinic_kpis(clinic_summary_kpis, current_period_str_clinic)
    if main_kpis_clinic:
        st.markdown("##### **Overall Service Performance:**"); cols_main = st.columns(min(len(main_kpis_clinic), 4))
        for i, kpi in enumerate(main_kpis_clinic): 
            with cols_main[i % 4]: render_kpi_card(**kpi)
    if disease_kpis_clinic:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**"); cols_disease = st.columns(min(len(disease_kpis_clinic), 4))
        for i, kpi in enumerate(disease_kif clinic_summary_kpis_data and isinstance(clinic_summary_kpis_data.get("test_summary_details"), dict):
    main_kpis_clinic = structure_main_clinic_kpis(clinic_summary_kpis_data, current_period_str_clinic)
    disease_kpis_clinic = structure_disease_specific_clinic_kpis(clinic_summary_kpis_data, current_period_str_clinic)
    if main_kpis_clinic:
        st.markdown("##### **Overall Service Performance:**"); kpi_cols_main = st.columns(min(len(main_kpis_clinic), 4))
        for i, kpi_data in enumerate(main_kpis_clinic):
            with kpi_cols_main[i % 4]: render_kpi_card(**kpi_data)
    if disease_kpis_clinic:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**"); kpi_cols_disease = st.columns(min(len(disease_kpis_clinic), 4))
        for i, kpi_data in enumerate(disease_kpis_clinic):
            with kpi_cols_disease[i % 4]: render_kpi_card(**kpi_data)
else: st.warning(f"Core clinic KPIs could not be generated for {current_period_str_clinic}.")

st.markdown("##### **Clinic Environment Quick Check:**")
env_summary_kpis_qc = get_clinic_environmental_summary_kpis(iot_df_period, "ClinicDash/EnvQuickCheck")
has_env_data_qc = env_summary_kpis_qc and any(pd.notna(v) and (v != 0 if "count" in k else True) for k,v in env_summary_kpis_qc.items() if isinstance(v, (int,float)) and ("avg_" in k or "_count" in k or "_flag" in k))
if has_env_data_qc:
    env_kpi_cols_qc = st.columns(4)
    co2_val = env_summary_kpis_qc.get('avg_co2_overall_ppm', np.nan)
    co2_stat = "HIGH_RISK" if pd.notna(co2_val) and co2_val > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM else ("MODERATE_CONCERN" if pd.notna(co2_val) and co2_val > settings.ALERT_AMBIENT_CO2_HIGH_PPM else "ACCEPTABLE")
    with env_kpi_cols_qc[0]: render_kpi_card("Avg. CO2", f"{co2_val:.0f}" if pd.notna(co2_val) else "N/A", "ppm", "💨", co2_stat, help_text=f"Avg CO2. Target < {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm.")
    pm25_val = env_summary_kpis_qc.get('avg_pm25_overall_ugm3', np.nan)
    pm25_stat = "HIGH_RISK" if pd.notna(pm25_val) and pm25_val > settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 else ("MODERATE_CONCERN" if pd.notna(pm25_val) and pm25_val > settings.ALERT_AMBIENT_PM25_HIGH_UGM3 else "ACCEPTABLE")
    with env_kpi_cols_qc[1]: render_kpi_card("Avg. PM2.5", f"{pm25_val:.1f}" if pd.notna(pm25_val) else "N/A", "µg/m³", "🌫️", pm25_stat, help_text=f"Avg PM2.5. Target < {settings.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³.")
    occup_val = env_summary_kpis_qc.get('avg_waiting_room_occupancy_overall_persons', np.nan)
    occup_stat = "MODERATE_CONCERN" if pd.notna(occup_val) and occup_val > settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX else "ACCEPTABLE"
    with env_kpi_cols_qc[2]: render_kpi_card("Avg. Waiting Occupancy", f"{occup_val:.1f}" if pd.notna(occup_val) else "N/A", "persons", "👨‍👩‍👧‍👦", occup_stat, help_text=f"Avg waiting area occupancy. Target < {settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons.")
    noise_alerts_val = env_summary_kpis_qc.get('rooms_noise_high_alert_latest_count', 0)
    noise_stat = "HIGH_CONCERN" if noise_alerts_val > 1 else ("MODERATE_CONCERN" if noise_alerts_val == 1 else "ACCEPTABLE")
    with env_kpi_cols_qc[3]:pis_clinic): 
            with cols_disease[i % 4]: render_kpi_card(**kpi)
else: st.warning(f"Core clinic KPIs could not be generated for {current_period_str_clinic}.")

st.markdown("##### **Clinic Environment Quick Check:**")
env_summary_quick = get_clinic_environmental_summary_kpis(iot_df_period_clinic, "ClinicDash/EnvQuickCheck")
has_env_data = env_summary_quick and any(pd.notna(v) and (v != 0 if "count" in k else True) for k,v in env_summary_quick.items() if isinstance(v, (int, float)) and ("avg_" in k or "_count" in k or "_flag" in k))
if has_env_data:
    cols_env = st.columns(4)
    co2_val, co2_stat = env_summary_quick.get('avg_co2_overall_ppm', np.nan), "ACCEPTABLE"
    if pd.notna(co2_val): co2_stat = "HIGH_RISK" if co2_val > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM else ("MODERATE_CONCERN" if co2_val > settings.ALERT_AMBIENT_CO2_HIGH_PPM else "ACCEPTABLE")
    with cols_env[0]: render_kpi_card("Avg. CO2", f"{co2_val:.0f}" if pd.notna(co2_val) else "N/A", "ppm", "💨", co2_stat, help_text=f"Avg CO2. Target < {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm.")
    pm25_val, pm25_stat = env_summary_quick.get('avg_pm25_overall_ugm3', np.nan), "ACCEPTABLE"
    if pd.notna(pm25_val): pm25_stat = "HIGH_RISK" if pm25_val > settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 else ("MODERATE_CONCERN" if pm25_val > settings.ALERT_AMBIENT_PM25_HIGH_UGM3 else "ACCEPTABLE")
    with cols_env[1]: render_kpi_card("Avg. PM2.5", f"{pm25_val:.1f}" if pd.notna(pm25_val) else "N/A", "µg/m³", "🌫️", pm25_stat, help_text=f"Avg PM2.5. Target < {settings.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³.")
    occup_val, occup_stat = env_summary_quick.get('avg_waiting_room_occupancy_overall_persons', np.nan), "ACCEPTABLE"
    if pd.notna(occup_val) and occup_val > settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX: occup_stat = "MODERATE_CONCERN"
    with cols_env[2]: render_kpi_card("Avg. Waiting Occupancy", f"{occup_val:.1f}" if pd.notna(occup_val) else "N/A", "persons", "👨‍👩‍👧‍👦", occup_stat, help_text=f"Avg waiting area occupancy. Target < {settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX}.")
    noise_alerts_count = env_summary_quick.get('rooms_noise_high_alert_latest_count', 0)
    noise_stat = "HIGH_CONCERN" if noise_alerts_count > 1 else ("MODERATE_CONCERN" if noise_alerts_count == 1 else "ACCEPTABLE")
    with cols_env[3]: render_kpi_card("High Noise Alerts", str(noise_alerts_count), "areas", "🔊", noise_stat, help_text=f"Areas with noise > {settings.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA.")
else: st.info("No significant environmental IoT data for this period." if iot_available else "Environmental IoT data source unavailable.")
st.divider()

# --- Tabbed Interface ---
st.header("🛠️ Operational Areas Deep Dive")
tab_titles = ["📈 Local Epidemiology", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_epi, tab_test, tab_supply, tab_patient, tab_env = st.tabs(tab_titles)

with tab_epi:
    st.subheader(f"Local Epidemiological Intelligence ({current_period_str_clinic})")
    if not health_df_period_clinic.empty:
        epi_data = calculate_clinic_epidemiological_data(health_df_period_clinic, current_period_str_clinic)
        sympt_trends_df = epi_data.get("symptom_trends_weekly_top_n_df")
        if isinstance(sympt_trends_df, pd.DataFrame) and not sympt_trends_df.empty render_kpi_card("High Noise Alerts", str(noise_alerts_val), "areas", "🔊", noise_stat, help_text=f"Areas with noise > {settings.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA (latest).")
else: st.info("No significant environmental IoT data for this period for snapshot KPIs." if iot_available else "Environmental IoT data source generally unavailable for snapshot.")
st.divider()

# --- Tabbed Interface ---
st.header("🛠️ Operational Areas Deep Dive")
tab_titles_clinic = ["📈 Local Epidemiology", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_epi_clinic, tab_testing_clinic, tab_supply_clinic, tab_patient_clinic, tab_env_clinic = st.tabs(tab_titles_clinic)

with tab_epi_clinic:
    st.subheader(f"Local Epidemiological Intelligence ({current_period_str_clinic})")
    if not health_df_period.empty:
        epi_data_clinic = calculate_clinic_epidemiological_data(health_df_period, current_period_str_clinic)
        symptom_trends_df = epi_data_clinic.get("symptom_trends_weekly_top_n_df")
        if isinstance(symptom_trends_df, pd.DataFrame) and not symptom_trends_df.empty:
            st.plotly_chart(plot_bar_chart(symptom_trends_df, 'week_start_date', 'count', "Weekly Symptom Frequency (Top Reported)", 'symptom', 'group', True, "Week Starting", "Symptom Encounters"), use_container_width=True)
        
        malaria_rdt_name = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT")
        malaria_pos_trend = epi_data_clinic.get("key_test_positivity_trends", {}).get(malaria_rdt_name)
        if isinstance(malaria_pos_trend, pd.Series) and not malaria_pos_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(malaria_pos_trend, f"Weekly {malaria_rdt_name} Positivity Rate", "Positivity %", target_ref_line_val=settings.TARGET_MALARIA_POSITIVITY_RATE, y_values_are_counts=False), use_container_width=True)
        for note in epi_data_clinic.get("calculation_notes", []): st.caption(f"Note (Epi Tab): {note}")
    else: st.info("No health data in period for epidemiological analysis.")

with tab_testing_clinic:
    st.subheader(f"Testing & Diagnostics Performance ({current_period_str_clinic})")
    testing_insights_map = prepare_clinic_lab_testing_insights_data(health_df_period, clinic_summary_kpis_data, current_period_str_clinic, "All Critical Tests Summary")
    crit_tests_summary_df = testing_insights_map.get("all_critical_tests_summary_table_df")
    if isinstance(crit_tests_summary_df, pd.DataFrame) and not crit_tests_summary_df.empty:
        st.markdown("###### **Critical Tests Performance Summary:**"); st.dataframe(crit_tests_summary_df, use_container_width=True, hide_index=True)
    overdue_tests_df = testing_insights_map.get("overdue_pending_tests_list_df")
    if isinstance(overdue_tests_df, pd.DataFrame) and not overdue_tests_df.empty:
        st.markdown("###### **Overdue Pending Tests (Top 15):**"); st.dataframe(overdue_tests_df.head(15), use_container_width=True, hide_index=True)
    elif isinstance(overdue_tests_df, pd.DataFrame): st.success("✅ No tests currently flagged as overdue.")
    for note in testing_insights_map.get("processing_notes", []): st.caption(f"Note (Testing Tab): {note}")

with tab_supply_clinic:
    st.subheader(f"Medical Supply Forecast & Status ({current_period_str_clinic})")
    use_ai_supply = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="clinic_supply_ai_toggle")
    supply_forecast_map = prepare_clinic_supply_forecast_overview_data(full_hist_health_df, current_period_str_clinic, use_ai_supply_forecasting_model=use_ai_supply) # Pass full_hist_health_df
    st.markdown(f"**Forecast Model Used:** `{supply_forecast_map.get('forecast_model_type_used', 'N/A')}`")
    supply_overview_list = supply_forecast_map.get("forecast_items_overview_list", [])
    if supply_overview_list:
        st.dataframe(pd.DataFrame(supply_overview_list), use_container_width=True, hide_index=True, column_config={"estimated_stockout_date": st.column_config.TextColumn("Est. Stockout Date")})
    else: st.info("No supply forecast data generated.")
    for note in supply_forecast_map.get("data_processing_notes", []): st.caption(f"Note (Supply Tab): {note}")

with tab_patient_clinic:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_period_str_clinic})")
    if not health_df_period.empty:
        patient_focus_map = prepare_clinic_patient_focus_overview_data(health_df_period, current_period_str_clinic)
        patient_load_plot_df = patient_focus_map.get("patient_load_by_key_condition_df")
        if isinstance(patient_load_plot_df, pd.DataFrame) and not patient_load_plot_df.empty:
            st.markdown("###### **Patient Load by Key Condition (Weekly):**")
            st.plotly_chart(plot_bar_chart(patient_load_plot_df, 'period_start_date:
            st.plotly_chart(plot_bar_chart(sympt_trends_df, 'week_start_date', 'count', "Weekly Symptom Frequency (Top Reported)", 'symptom', 'group', y_values_are_counts_flag=True, x_axis_label_text="Week Starting", y_axis_label_text="Symptom Encounters"), use_container_width=True)
        malaria_rdt_name = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT")
        malaria_pos_trend = epi_data.get("key_test_positivity_trends", {}).get(malaria_rdt_name)
        if isinstance(malaria_pos_trend, pd.Series) and not malaria_pos_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(malaria_pos_trend, f"Weekly {malaria_rdt_name} Positivity Rate", "Positivity %", target_ref_line_val=settings.TARGET_MALARIA_POSITIVITY_RATE), use_container_width=True)
        for note in epi_data.get("calculation_notes", []): st.caption(f"Note (Epi): {note}")
    else: st.info("No health data in period for epidemiological analysis.")

with tab_test:
    st.subheader(f"Testing & Diagnostics Performance ({current_period_str_clinic})")
    test_insights = prepare_clinic_lab_testing_insights_data(health_df_period_clinic, clinic_summary_kpis, current_period_str_clinic)
    crit_summary_df = test_insights.get("all_critical_tests_summary_table_df")
    if isinstance(crit_summary_df, pd.DataFrame) and not crit_summary_df.empty: st.markdown("###### **Critical Tests Performance Summary:**"); st.dataframe(crit_summary_df, use_container_width=True, hide_index=True)
    overdue_df = test_insights.get("overdue_pending_tests_list_df")
    if isinstance(overdue_df, pd.DataFrame) and not overdue_df.empty: st.markdown("###### **Overdue Pending Tests (Top 15):**"); st.dataframe(overdue_df.head(15), use_container_width=True, hide_index=True)
    elif isinstance(overdue_df, pd.DataFrame): st.success("✅ No tests currently flagged as overdue.")
    for note in test_insights.get("processing_notes", []): st.caption(f"Note (Testing): {note}")

with tab_supply:
    st.subheader(f"Medical Supply Forecast & Status ({current_period_str_clinic})")
    use_ai_supply = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="clinic_supply_ai_toggle_v2")
    supply_data = prepare_clinic_supply_forecast_overview_data(full_hist_health_df, current_period_str_clinic, use_ai_supply_forecasting_model=use_ai_supply)
    st.markdown(f"**Forecast Model Used:** `{supply_data.get('forecast_model_type_used', 'N/A')}`")
    supply_overview_list = supply_data.get("forecast_items_overview_list", [])
    if supply_overview_list: st.dataframe(pd.DataFrame(supply_overview_list), use_container_width=True, hide_index=True, column_config={"estimated_stockout_date": st.column_config.TextColumn("Est. Stockout Date")})
    else: st.info("No supply forecast data generated.")
    for note in supply_data.get("data_processing_notes", []): st.caption(f"Note (Supply): {note}")

with tab_patient:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_period_str_clinic})")
    if not health_df_period_clinic.empty:
        patient_focus_data = prepare_clinic_patient_focus_overview_data(health_df_period_clinic, current_period_str_clinic)
        load_plot_df = patient_focus_data.get("patient_load_by_key_condition_df")
        if isinstance(load_plot_df, pd.DataFrame) and not load_plot_df.empty:
            st.markdown("###### **Patient Load by Key Condition (Weekly):**")
            st.plotly_chart(plot_bar_chart(load_plot_df, 'period_start_date', 'unique_patients_count', "Patient Load by Key Condition", 'condition', 'stack', y_values_are_counts_flag=True, x_axis_label_text="Week Starting", y_axis_label_text="Unique Patients Seen"), use_container_width=True)
        flagged_patients_df = patient_focus_data.get("flagged_patients_for_review_df")
        if isinstance(flagged_patients_df, pd.DataFrame) and not flagged_patients_df.empty:
            st.markdown("###### **Flagged Patients for Clinical Review (Top Priority):**"); st.dataframe(flagged_patients_df.head(15), use_container_width=True, hide_index=True)
        elif isinstance(flagged_patients_df, pd.DataFrame): st.info("No patients currently flagged for clinical review.")
        for note in patient_focus_data.get("processing_notes", []): st.caption(f"Note (Patient Focus): {note}")
    else: st.info("No health data in period for patient focus analysis.")

with tab_env:
    st.subheader(f"Facility Environment Detailed Monitoring ({current_period_str_clinic})")
    env_details = prepare_clinic_environmental_detail_data(iot_df_period_clinic, iot_available, current_period_str_clinic)
    env_alerts = env_details.get("current_environmental_alerts_list", [])
    if env_alerts:
        st.markdown("###### **Current Environmental Alerts (Latest Readings):**"); non_acceptable_found = False
        for alert in env_alerts:
            if alert.get("level") != "ACCEPTABLE": non_acceptable_found = True; render_traffic_light_indicator(alert.get('message', 'Issue detected.'), alert.get('level', 'UNKNOWN'), alert.get('alert_type', 'Env Alert'))
        if not non_acceptable_found and len(env_alerts) == 1 and env_alerts[0].get("level") == "ACCEPTABLE": st.success(f"✅ {env_alerts[0].get('message', 'Environment appears normal.')}")
        elif not non_acceptable_found and len(env_alerts) > 1: st.info("Multiple environmental parameters checked; all appear within acceptable limits.")
    co2_trend_clinic = env_details.get("hourly_avg_co2_trend")
    if isinstance(co2_trend_clinic, pd.Series) and not co2_trend_clinic.empty: st.plotly_chart(plot_annotated_line_chart(co2_trend_clinic, "Hourly Avg. CO2 Levels (Clinic-wide)", "CO2 (ppm)", date_format_hover="%H:%M (%d-%b)", target_ref_line_val=settings.ALERT_AMBIENT_CO2_HIGH_PPM), use_container_width=True)
    latest_room_df = env_details.get("latest_room_sensor_readings_df")
    if isinstance(latest_room_df, pd.DataFrame) and not latest_room_df.empty: st.markdown("###### **Latest Sensor Readings by Room (End of Period):**"); st.dataframe(latest_room_df, use_container_width=True, hide_index=True)
    for note in env_details.get("processing_notes", []): st.caption(f"Note (Env. Detail): {note}")
    if not iot_available and (not isinstance(iot_df_period_clinic, pd.DataFrame) or iot_df_period_clinic.empty):
        st.warning("IoT environmental data source generally unavailable. Detailed monitoring not possible.")

logger.info(f"Clinic Operations Console page loaded for period: {current_period_str_clinic}")
