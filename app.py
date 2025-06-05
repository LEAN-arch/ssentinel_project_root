# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import sys 
from pathlib import Path 
import logging 
import html 
import importlib.util 

# --- CRITICAL PATH SETUP ---
_this_app_file_path = Path(__file__).resolve()
_project_root_dir_app = _this_app_file_path.parent    

# Initial diagnostic prints to stderr
print(f"DEBUG_APP_PY (L19): __file__ in app.py = {__file__}", file=sys.stderr)
print(f"DEBUG_APP_PY (L20): _this_app_file_path_app = {_this_app_file_path}", file=sys.stderr)
print(f"DEBUG_APP_PY (L21): Calculated _project_root_dir_app = {_project_root_dir_app}", file=sys.stderr)
print(f"DEBUG_APP_PY (L22): Initial sys.path before any modification in app.py = {sys.path}", file=sys.stderr)

project_root_str_app = str(_project_root_dir_app)
if project_root_str_app not in sys.path:
    sys.path.insert(0, project_root_str_app)
    print(f"DEBUG_APP_PY (L29): Added project root '{project_root_str_app}' to sys.path.", file=sys.stderr)
elif sys.path[0] != project_root_str_app:
    try:
        sys.path.remove(project_root_str_app)
        print(f"DEBUG_APP_PY (L34): Removed '{project_root_str_app}' from interior of sys.path.", file=sys.stderr)
    except ValueError:
        print(f"DEBUG_APP_PY (L37): '{project_root_str_app}' reported in sys.path but not found by remove(). Will insert at [0].", file=sys.stderr)
    sys.path.insert(0, project_root_str_app)
    print(f"DEBUG_APP_PY (L40): Moved '{project_root_str_app}' to start of sys.path.", file=sys.stderr)
else:
    print(f"DEBUG_APP_PY (L43): Project root '{project_root_str_app}' is already at start of sys.path.", file=sys.stderr)

print(f"DEBUG_APP_PY (L46): sys.path JUST BEFORE 'from config import settings' = {sys.path}", file=sys.stderr)

# --- Import Settings ---
try:
    from config import settings 
except ImportError as e_cfg_app_final_corrected_v4:
    print(f"FATAL_APP_PY (L53): STILL FAILED to import config.settings: {e_cfg_app_final_corrected_v4}", file=sys.stderr)
    print(f"FINAL sys.path at import failure: {sys.path}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings_final_corrected_v4: 
    print(f"FATAL_APP_PY (L58): AttributeError on 'config.settings' (likely circular import OR settings.py error): {e_attr_settings_final_corrected_v4}", file=sys.stderr)
    print(f"FINAL sys.path at attribute error: {sys.path}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg_final_corrected_v4:
    print(f"FATAL_APP_PY (L63): Generic error during 'config.settings' import: {e_generic_cfg_final_corrected_v4}", file=sys.stderr)
    print(f"FINAL sys.path at generic error: {sys.path}", file=sys.stderr)
    sys.exit(1)

import streamlit as st # Import Streamlit after settings

# --- Global Logging Configuration ---
valid_log_levels_app_final_cfg_v5 = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str_final_cfg_v5 = str(settings.LOG_LEVEL).upper()
if log_level_app_str_final_cfg_v5 not in valid_log_levels_app_final_cfg_v5:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str_final_cfg_v5}'. Using INFO.", file=sys.stderr); log_level_app_str_final_cfg_v5 = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str_final_cfg_v5, logging.INFO), 
                    format=settings.LOG_FORMAT, datefmt=settings.LOGFile 57 (Corrected - `NameError` for `Dict`): `ssentinel_project_root/pages/04_population_dashboard.py` (assuming you've renamed it with a prefix)
```python
# sentinel_project_root/pages/04_population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
import html 
import plotly.express as px 
from pathlib import Path 
import re 
from typing import Optional, Dict, Any, Tuple, List # Added Dict, Tuple, List here

try:
    from config import settings
    from data_processing.loaders import load_health_records, load_zone_data
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe, convert_to_numeric
    from visualization.plots import plot_bar_chart, create_empty_figure, plot_annotated_line_chart
    from visualization.ui_elements import display_custom_styled_kpi_box
except ImportError as e_pop_dash_final_fix: # Unique exception variable name
    import sys
    _current_file_pop_final_fix = Path(__file__).resolve()
    _project_root_pop_assumption_final_fix = _current_file_pop_final_fix.parent.parent
    error_msg_pop_detail_final_fix = (
        f"Population Dashboard Import Error: {e_pop_dash_final_fix}. "
        f"Ensure project root ('{_project_root_pop_assumption_final_fix}') is in sys.path and modules are correct. "
        f"Path: {sys.path}"
    )
    try: st.error(error_msg_pop_detail_final_fix); st.stop()
    except NameError: print(error_msg_pop_detail_final_fix, file=sys.stderr); raise

logger = logging.getLogger(__name__)

st.title(f"📊 {settings.APP_NAME} - Population Health Analytics & Research Console")
st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.divider()

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe}, show_spinner="Loading population_DATE_FORMAT, 
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__) 
logger.info(f"INFO (app.py): Successfully imported and accessed config.settings. APP_NAME: {settings.APP_NAME}")

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30_APP_V4 = False 
STREAMLIT_PAGE_LINK_AVAILABLE_APP_V4 = False
try:
    from packaging import version 
    st_version_obj_v4 = version.parse(st.__version__) 
    if st_version_obj_v4 >= version.parse("1.30.0"): STREAMLIT_VERSION_GE_1_30_APP_V4 = True
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE_APP_V4 = True
    if not STREAMLIT_VERSION_GE_1_30_APP_V4: logger.warning(f"Streamlit version {st.__version__} < 1.30.0. Some UI features might use fallbacks.")
except Exception as e_st_ver_final_cfg_val_app_v4: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver_final_cfg_val_app_v4}")

if not importlib.util.find_spec("plotly"): logger.warning("Plotly not installed. Visualization features may fail.")

# --- Page Configuration ---
page_icon_path_obj_app_main_cfg_final_val_v4 = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon_str_app_main_cfg_final_val_v4: str = str(page_icon_path_obj_app_main_cfg_final_val_v4) if page_icon_path_obj_app_main_cfg_final_val_v4.is_file() else "🌍"
if final_page_icon_str_app_main_cfg_final_val_v4 == "🌍": logger.warning(f"Page icon not found: '{page_icon_path_obj_app_main_cfg_final_val_v4}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str_app_main_cfg_final_val_v4,
    layout="wide", initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}\n\nEdge-First Health Intelligence Co-Pilot."
    }
)

# --- Apply Plotly Theme & CSS ---
 analytics dataset...")
def get_population_analytics_datasets_pop_fix(log_ctx: str = "PopAnalyticsConsole/LoadData") -> tuple[pd.DataFrame, pd.DataFrame]: # Renamed func
    logger.info(f"({log_ctx}) Loading population health records and zone attributes.")
    raw_health_df_pop_fix = load_health_records(source_context=f"{log_ctx}/HealthRecs")
    enriched_health_df_pop_fix: pd.DataFrame
    base_health_cols_schema_pop_fix = ['patient_id', 'encounter_date', 'condition', 'age', 'gender', 'zone_id', 'ai_risk_score', 'ai_followup_priority_score']
    if isinstance(raw_health_df_pop_fix, pd.DataFrame) and not raw_health_df_pop_fix.empty:
        enriched_health_df_pop_fix, _ = apply_ai_models(raw_health_df_pop_fix.copy(), source_context=f"{log_ctx}/AIEnrich")
    else:
        logger.warning(f"({log_ctx}) Raw health records empty/invalid. AI enrichment skipped.")
        enriched_health_df_pop_fix = pd.DataFrame(columns=base_health_cols_schema_pop_fix)

    zone_data_full_pop_fix = load_zone_data(source_context=f"{log_ctx}/ZoneData")
    zone_attributes_df_pop_fix: pd.DataFrame
    sdoh_cols_pop_fix = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min', 'predominant_hazard_type', 'primary_livelihood', 'water_source_main', 'area_sqkm']
    if isinstance(zone_data_full_pop_fix, pd.DataFrame) and not zone_data_full_pop_fix.empty:
        cols_keep_pop_fix = [col_pop for col_pop in sdoh_cols_pop_fix if col_pop in zone_data_full_pop_fix.columns]
        if 'zone_id' not in cols_keep_pop_fix and 'zone_id' in zone_data_full_pop_fix.columns: cols_keep_pop_fix.append('zone_id')
        if cols_keep_pop_fix:
            zone_attributes_df_pop_fix = zone_data_full_pop_fix[list(set(cols_keep_pop_fix))].copy()
            for sdoh_col_pop_fix in sdoh_cols_pop_fix:
                if sdoh_col_pop_fix not in zone_attributes_df_pop_fix.columns: zone_attributes_df_pop_fix[sdoh_col_pop_fix] = np.nan
            logger.info(f"({log_ctx}) Loaded {len(zone_attributes_df_pop_fix)} zone attributes.")
        else: zone_attributes_df_pop_fix = pd.DataFrame(columns=sdoh_cols_pop_fix); logger.warning(f"({log_ctx}) No SDOH columns in zone data.")
    else: zone_attributes_df_pop_fix = pd.DataFrame(columns=sdoh_cols_pop_fix); logger.warning(f"({log_ctx}) Zone attributes data unavailable.")
    if enriched_health_df_pop_fix.empty: logger.error(f"({log_ctx}) CRITICAL: Health data empty after processing.")
    return enriched_health_df_pop_fix, zone_attributes_df_pop_fix

# --- Load Datasets ---
health_df_pop_main_fix, zone_attr_pop_main_fix = pd.DataFrame(), pd.DataFrame() 
try: health_df_pop_main_fix, zone_attr_pop_main_fix = get_population_analytics_datasets_pop_fix() # Use renamed func
except Exception as e_pop_load_main_fix: 
    logger.error(f"Population Dashboard: Dataset loading failed: {e_pop_load_main_fix}", exc_info=True)
    st.error(f"Error loading population analytics data: {str(e_pop_load_main_fix)}. Dashboard functionality will be severely limited. Please check console logs and ensure data files (e.g., health_records_expanded.csv) are correctly placed and accessible.")
if health_df_pop_main_fix.empty: 
    st.error("🚨 Critical Data Failure: Primary health dataset for population analytics is empty. Most console features will be unavailable. Ensure `health_records_expanded.csv` is in `data_sources/` and is not empty.")

# --- Sidebar Filters ---
# Assuming settings.PROJECT_ROOT_DIR is correctly defined in settings.py
project_root_pop_fix = Path(settings.PROJECT_ROOT_DIR) 
logo_path_pop_sidebar_final_fix = project_root_pop_fix / settings.APP_LOGO_SMALL_PATH 
if logo_path_pop_sidebar_final_fix.exists() and logo_path_pop_sidebar_final_fix.is_file(): st.sidebar.image(str(logo_path_pop_sidebar_final_fix), width=120)
else: logger.warning(f"Sidebar logo not found for Population Dashboard: {logo_path_pop_sidebar_final_fix}")
st.sidebar.header("🔎 Analytics Filters")

min_date_pop_final_fix, max_date_pop_final_fix = date.today() - timedelta(days=365*3), date.today()
if isinstance(health_df_pop_main_fix, pd.DataFrame) and 'encounter_date' in health_df_pop_main_fix.columns and health_df_pop_main_fix['encounter_date'].notna().any():
    if not pd.api.types.is_datetime64_any_dtype(health_df_pop_main_fix['encounter_date']): health_df_pop_main_fix['encounter_date'] = pd.to_datetime(health_df_pop_main_fix['encounter_date'], errors='coerce')
    if health_df_pop_main_fix['encounter_date'].notna().any():
        min_date_pop_final_fix, max_date_pop_final_fix = health_df_pop_main_fix['encounter_date'].min().date(), health_df_pop_main_fix['encounter_date'].max().date()
if min_date_pop_final_fix > max_date_pop_final_fix: min_date_pop_final_fix = max_date_pop_final_fix 

pop_date_key_final_ss_fix = "pop_dashboard_date_range_v4" # Ensure unique key
if pop_date_key_final_ss_fix not in st.session_state: st.session_state[pop_date_key_final_ss_fix] = [min_date_pop_final_fix, max_date_pop_final_fix]
selected_date_range_pop_ui_val_final_fix = st.sidebar.date_input("Select Date Range for Analysis:", value=st.session_state[try:
    from visualization.plots import set_sentinel_plotly_theme
    set_sentinel_plotly_theme(); logger.debug("Sentinel Plotly theme applied.")
except Exception as e_theme_main_app_cfg_final_val_app_v4: logger.error(f"Error applying Plotly theme: {e_theme_main_app_cfg_final_val_app_v4}", exc_info=True); st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles_app_final_cfg_val_ui_app_v4(css_path_str_app_final_cfg_val_ui_app_v4: str):
    css_path_app_final_cfg_val_ui_app_v4 = Path(css_path_str_app_final_cfg_val_ui_app_v4)
    if css_path_app_final_cfg_val_ui_app_v4.is_file():
        try:
            with open(css_path_app_final_cfg_val_ui_app_v4, "r", encoding="utf-8") as f_css_app_final_cfg_val_ui_app_v4: st.markdown(f'<style>{f_css_app_final_cfg_val_ui_app_v4.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path_app_final_cfg_val_ui_app_v4}")
        except Exception as e_css_main_app_final_cfg_val_ui_app_v4: logger.error(f"Error applying CSS {css_path_app_final_cfg_val_ui_app_v4}: {e_css_main_app_final_cfg_val_ui_app_v4}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path_app_final_cfg_val_ui_app_v4}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles_app_final_cfg_val_ui_app_v4(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols_app_ui_final_cfg_val_ui_val_app_v4 = st.columns([0.12, 0.88])
with header_cols_app_ui_final_cfg_val_ui_val_app_v4[0]:
    l_logo_path_app_final_cfg_val_app_v4 = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_app_final_cfg_val_app_v4 = Path(settings.APP_LOGO_SMALL_PATH)
    if l_logo_path_app_final_cfg_val_app_v4.is_file(): st.image(str(l_logo_path_app_final_cfg_val_app_v4), width=100)
    elif s_logo_path_app_final_cfg_val_app_v4.is_file(): st.image(str(s_logo_path_app_final_cfg_val_app_v4), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_app_final_cfg_val_app_v4}', S: '{s_logo_path_app_final_cfg_val_app_v4}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_ui_final_cfg_val_ui_val_app_v4[1]: st.title(html.escape(settings.APP_NAME)); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description (Full content from prompt used here) ---
st.markdown(f"""## Welcome to the {html.escape(settings.APP_NAME)} Demonstrator
Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
operational actionability** in resource-limited, high-risk environments. It aims to convert 
diverse data sources into life-saving, workflow-integrated decisions, even with 
**minimal or intermittent internet connectivity.**""")
st.markdown("#### Core Design Principles:")
core_principles_main_app_v5_val_app_v4 = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_core_principles_v5_val_app_v4 = min(len(core_principles_main_app_v5_val_app_v4), 2)
if num_cols_core_principles_v5_val_app_v4 > 0:
    cols_core_principles_ui_v5_val_app_v4 = st.columns(num_cols_core_principles_v5_val_app_v4)
    for idx_core_v5_val_app_v4, (title_core_v5_val_app_v4, desc_core_v5_val_app_v4) in enumerate(core_principles_main_app_v5_val_app_v4):
        with cols_core_principles_ui_v5_val_app_v4[idx_core_v5_val_app_v4 % num_cols_core_principles_v5_val_app_v4]:
            st.markdown(f"##### {html.escape(title_core_v5_val_app_v4)}"); st.markdown(f"<small>{html.escape(desc_core_v5_val_app_v4)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. These views represent perspectives of **Supervisors, Clinic Managers, or District Health Officers (DHOs)**. The primary interface for frontline workers (e.g., CHWs) is a dedicated native application on their Personal Edge Device (PED), tailored for their specific operational context.")
st.info("💡 **Note:** This web application serves as a high-level demonstrator for the Sentinel system's data processing capabilities and the types of aggregated views available to management and strategic personnel.")
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_directory_obj_app_final_cfg_val_app_v4 = _project_root_dir_app / "pages" 
role_navigation_config_app_final_cfg_list_val_app_v4 = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data...", "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2)...", "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs)...", "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis...", "page_filename": "04_population_dashboard.py", "icon": "📊"},
] 

num_nav_cols_final_app_cfg_val_ui_v4_app_v4 = min(len(role_navigation_config_app_final_cfg_list_val_app_v4), 2)
if num_nav_cols_final_app_cfg_val_ui_v4_app_v4 > 0:
    nav_cols_ui_final_app_cfg_val_ui_v4_app_v4 = st.columns(num_nav_cols_final_app_cfg_val_ui_v4_app_v4)
    current_col_idx_nav_final_cfg_val_ui_v4_app_v4 = 0
    for nav_item_final_app_cfg_item_val_v4_app_v4 in role_navigation_config_app_final_cfg_list_val_app_v4:
        page_link_target_app_cfg_item_val_v4_app_v4 = nav_item_final_app_cfg_item_val_v4_app_v4['page_filename'] 
        physical_page_full_path_app_cfg_item_val_v4_app_v4 = pages_directory_obj_app_final_cfg_val_app_v4 / nav_item_final_app_cfg_item_val_v4_app_v4["page_filename"]
        if not physical_page_full_path_app_cfg_item_val_v4_app_v4.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app_cfg_item_val_v4_app_v4['title']}' not found: {physical_page_full_path_app_cfg_item_val_v4_app_v4}")
            continue
        with nav_cols_ui_final_app_cfg_val_ui_v4_app_v4[current_col_idx_nav_final_cfg_val_ui_v4_app_v4 % num_nav_cols_final_app_cfg_val_ui_v4_app_v4]:
            container_args_final_app_cfg_val_v4_app_v4 = {"border": True} if STREAMLIT_VERSION_GE_1_30_APP_V4 else {}
            with st.container(**container_args_final_app_cfg_val_v4_app_v4):
                st.subheader(f"{nav_item_final_app_cfg_item_val_v4_app_v4['icon']} {html.escape(nav_item_final_app_cfg_item_val_v4_app_v4['title'])}")
                st.markdown(f"<small>{nav_item_final_app_cfg_item_val_v4_app_v4['desc']}</small>", unsafe_allow_html=True) 
                link_label_final_app_cfg_val_v4_app_v4 = f"Explore {nav_item_final_app_cfg_item_val_v4_app_v4['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE_APP_V4:
                    link_kwargs_final_app_cfg_val_v4_app_v4 = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30_APP_V4 else {}
                    st.page_link(page_link_target_app_cfg_item_val_v4_app_v4, label=link_label_final_app_cfg_val_v4_app_v4, icon="➡️", **link_kwargs_final_app_cfg_val_v4_app_v4)
                else: 
                    st.markdown(f'<a href="{nav_item_final_app_cfg_item_val_v4_app_v4["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label_final_app_cfg_val_v4_app_v4} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final_cfg_val_ui_v4_app_v4 += 1
st.divider()

st.header(f"{html.escape(settings.APP_NAME)} - Key Capabilities Reimagined")
capabilities_data_app_final_cfg_full_v4_app_v4 = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_final_app_cfg_val_final_v4_app_v4 = min(len(capabilities_data_app_final_cfg_full_v4_app_v4), 3)
if num_cap_cols_final_app_cfg_val_final_v4_app_v4 > 0:
    cap_cols_ui_final_app_cfg_val_final_v4_app_v4 = st.columns(num_cap_cols_final_app_cfg_val_final_v4_app_v4)
    for i_cap_final_cfg_final_v4_app_v4, (cap_t_final_cfg_final_v4_app_v4, cap_d_final_cfg_final_v4_app_v4) in enumerate(capabilities_data_app_final_cfg_full_v4_app_v4):
        with cap_cols_ui_final_app_cfg_val_final_v4_app_v4[i_cap_final_cfg_final_v4_app_v4 % num_cap_cols_final_app_cfg_val_final_v4_app_v4]: 
            st.markdown(f"##### {html.escape(cap_t_final_cfg_final_v4_app_v4)}"); st.markdown(f"<small>{html.escape(cap_d_final_cfg_final_v4_app_v4)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

st.sidebar.header(f"{html.escape(settings.APP_NAME)} v{settings.APP_VERSION}")
st.sidebar.divider(); st.sidebar.markdown("#### About This Demonstrator:"); st.sidebar.info("Web app simulates higher-level dashboards...")
st.sidebar.divider()
glossary_filename_sidebar_cfg_final_val_v4_app_v4 = "05_glossary_page.py" 
glossary_link_target_sidebar_cfg_final_val_v4_app_v4 = glossary_filename_sidebar_cfg_final_val_v4_app_v4 
glossary_physical_path_final_sb_cfg_final_val_v4_app_v4 = pages_directory_obj_app_final_cfg_val_app_v4 / glossary_filename_sidebar_cfg_final_val_v4_app_v4
if glossary_physical_path_final_sb_cfg_final_val_v4_app_v4.exists():
    if STREAMLIT_PAGE_LINK_AVAILABLE_APP_V4: st.sidebar.page_link(glossary_link_target_sidebar_cfg_final_val_v4_app_v4, label="📜 System Glossary", icon="📚")
    else: st.sidebar.markdown(f'<a href="{glossary_filename_sidebar_cfg_final_val_v4_app_v4}" target="_self">📜 System Glossary</a>', unsafe_allow_html=True)
else: logger.warning(f"Glossary page for sidebar (expected: {glossary_physical_path_final_sb_cfg_final_val_v4_app_v4}) not found.")
st.sidebar.divider()
st.sidebar.markdown(f"**{html.escape(settings.ORGANIZATION_NAME)}**"); st.sidebar.markdown(f"Support: [{html.escape(settings.SUPPORT_CONTACT_INFO)}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(html.escape(settings.APP_FOOTER_TEXT))
logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
