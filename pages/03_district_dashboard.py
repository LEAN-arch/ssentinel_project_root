# sentinel_project_root/pages/03_district_dashboard.py
# District Health Strategic Command Center for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple
import os 
from pathlib import Path # For path operations

# --- Sentinel System Imports (Absolute Imports from Project Root) ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data, load_zone_data
    from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
    from data_processing.aggregation import get_district_summary_kpis
    from data_processing.helpers import hash_dataframe_safe
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart
    
    # District Component specific imports using absolute paths from 'pages'
    from pages.district_components.kpi_structuring import structure_district_summary_kpis
    from pages.district_components.map_display import render_district_map_visualization, _get_district_map_metric_options_config
    from pages.district_components.trend_analysis import calculate_district_wide_trends
    from pages.district_components.comparison_data import prepare_district_zonal_comparison_data, get_district_comparison_metrics_config
    from pages.district_components.intervention_planning import (
        get_district_intervention_criteria_options,
        identify_priority_zones_for_intervention_planning
    )
except ImportError as e_dist_dash_abs: # Unique exception variable name
    import sys
    _current_file_dist = Path(__file__).resolve()
    _pages_dir_dist = _current_file_dist.parent
    _project_root_dist_assumption = _pages_dir_dist.parent 

    error_msg_dist_detail = (
        f"District Dashboard Import Error (using absolute imports): {e_dist_dash_abs}. "
        f"Ensure project root ('{_project_root_dist_assumption}') is in sys.path (done by app.py) "
        f"and all modules/packages (e.g., 'pages', 'pages.district_components') have `__init__.py` files. "
        f"Check for typos in import paths. Current Python Path: {sys.path}"
    )
    try:
        st.error(error_msg_dist_detail)
        st.stop()
    except NameError:
        print(error_msg_dist_detail, file=sys.stderr)
        raise

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

# --- Page Config ---
try:
    page_icon_value = "🗺️"
    app_logo_small_path_str = _get_setting('APP_LOGO_SMALL_PATH', None)
    if app_logo_small_path_str:
        favicon_path = Path(app_logo_small_path_str)
        if favicon_path.is_file():
            page_icon_value = str(favicon_path)
        else:
            logger.warning(f"Favicon for District Dashboard not found at path from setting APP_LOGO_SMALL_PATH: {favicon_path}")
    
    page_layout_value = _get_setting('APP_LAYOUT', "wide")
    
    st.set_page_config(
        page_title=f"District Command - {_get_setting('APP_NAME', 'Sentinel App')}",
        page_icon=page_icon_value, layout=page_layout_value
    )
except Exception as e_page_config:
    logger.error(f"Error applying page configuration for District Dashboard: {e_page_config}", exc_info=True)
    st.set_page_config(page_title="District Command", page_icon="🗺️", layout="wide")


# --- Page Title and Introduction ---
st.title(f"🗺️ {settings.APP_NAME} - District Health Strategic Command Center")
st.markdown(f"**Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring.**")
st.divider()


# --- Data Aggregation and Preparation for DHO View (Cached) ---
@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    hash_funcs={pd.DataFrame: hash_dataframe_safe},
    show_spinner="Aggregating district-level operational data..."
)
def get_dho_command_center_processed_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Loads and processes core data. Returns only serializable (cacheable) objects.
    The intervention criteria options (which may contain non-serializable lambdas) are generated outside this cached function.
    """
    log_ctx = "DHODatasetPrep"
    logger.info(f"({log_ctx}) Initializing full data pipeline for DHO view...")
    
    raw_health_df_dho = load_health_records(source_context=f"{log_ctx}/LoadHealth")
    raw_iot_df_dho = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadIoT")
    base_zone_df_dho = load_zone_data(source_context=f"{log_ctx}/LoadZoneData")

    if not isinstance(base_zone_df_dho, pd.DataFrame) or base_zone_df_dho.empty or 'zone_id' not in base_zone_df_dho.columns:
        logger.error(f"({log_ctx}) Base zone DataFrame failed to load or is invalid. DHO dashboard heavily impacted.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    ai_enriched_health_df_dho: Optional[pd.DataFrame] = pd.DataFrame() # Initialize
    if isinstance(raw_health_df_dho, pd.DataFrame) and not raw_health_df_dho.empty:
        ai_enriched_health_df_dho, _ = apply_ai_models(raw_health_df_dho.copy(), source_context=f"{log_ctx}/AIEnrichHealth")
    else:
        logger.warning(f"({log_ctx}) Raw health data for DHO view empty/invalid. AI enrichment skipped.")

    enriched_zone_df_for_dho = enrich_zone_geodata_with_health_aggregates(
        zone_df=base_zone_df_dho, health_df=ai_enriched_health_df_dho,
        iot_df=raw_iot_df_dho, source_context=f"{log_ctx}/EnrichZoneDF"
    )
    if not isinstance(enriched_zone_df_for_dho, pd.DataFrame) or enriched_zone_df_for_dho.empty: # Check after enrichment
        logger.warning(f"({log_ctx}) Zone DataFrame enrichment resulted in empty DataFrame. Using base zone data if available.")
        enriched_zone_df_for_dho = base_zone_df_dho if not base_zone_df_dho.empty else pd.DataFrame()

    district_summary_kpis_map = get_district_summary_kpis(enriched_zone_df_for_dho, f"{log_ctx}/CalcDistrictKPIs")
    
    # DEBUG FIX: Removed generation of intervention_criteria_opts from the cached function.
    
    df_shape_log = enriched_zone_df_for_dho.shape if isinstance(enriched_zone_df_for_dho, pd.DataFrame) else 'N/A'
    logger.info(f"({log_ctx}) DHO data preparation complete. Enriched Zone DF shape: {df_shape_log}")
    return enriched_zone_df_for_dho, ai_enriched_health_df_dho, raw_iot_df_dho, district_summary_kpis_map

# --- Load and Prepare Data for the Dashboard ---
# DEBUG FIX: Initialize variables to match the new return signature of the cached function.
enriched_zone_df_display, historical_health_df_for_trends, historical_iot_df_for_trends, district_kpis_summary_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
intervention_criteria_options_data = {} # Will be populated after data loading.

try:
    # DEBUG FIX: Unpack fewer variables, as the cached function returns one less item.
    (enriched_zone_df_display, historical_health_df_for_trends, historical_iot_df_for_trends,
     district_kpis_summary_data) = get_dho_command_center_processed_datasets()
     
    # DEBUG FIX: Generate the (potentially non-serializable) intervention options *after* loading data from the cache.
    intervention_criteria_options_data = get_district_intervention_criteria_options(
        district_zone_df_sample_check=enriched_zone_df_display.head(2) if isinstance(enriched_zone_df_display, pd.DataFrame) and not enriched_zone_df_display.empty else None
    )

except Exception as e_dho_data_main_load:
    logger.error(f"DHO Dashboard: Failed to load/process main datasets: {e_dho_data_main_load}", exc_info=True)
    st.error(f"Error loading DHO dashboard data: {e_dho_data_main_load}. Check logs.")

data_as_of_ts = pd.Timestamp('now') # Placeholder; ideally from data source
st.caption(f"Data presented as of: {data_as_of_ts.strftime('%d %b %Y, %H:%M %Z')}")

# --- Sidebar Filters ---
logo_path_sidebar_dho = Path(settings.PROJECT_ROOT_DIR) / settings.APP_LOGO_SMALL_PATH
if logo_path_sidebar_dho.exists() and logo_path_sidebar_dho.is_file(): st.sidebar.image(str(logo_path_sidebar_dho), width=240)
else: logger.warning(f"Sidebar logo not found for DHO dashboard: {logo_path_sidebar_dho}")
st.sidebar.header("Analysis Filters")

abs_min_date_dho = data_as_of_ts.date() - timedelta(days=365 * 2)
abs_max_date_dho = data_as_of_ts.date()
def_end_dho = abs_max_date_dho
def_start_dho = max(abs_min_date_dho, def_end_dho - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3 - 1))

dho_trend_date_key_ss = "dho_dashboard_trend_date_range_v2" # Unique session state key
if dho_trend_date_key_ss not in st.session_state: st.session_state[dho_trend_date_key_ss] = [def_start_dho, def_end_dho]
selected_trend_range_dho = st.sidebar.date_input("Select Date Range for Trend Analysis:", value=st.session_state[dho_trend_date_key_ss], min_value=abs_min_date_dho, max_value=abs_max_date_dho, key=f"{dho_trend_date_key_ss}_widget")

selected_start_date_trends: date; selected_end_date_trends: date # Type hint
if isinstance(selected_trend_range_dho, (list,tuple)) and len(selected_trend_range_dho) == 2:
    st.session_state[dho_trend_date_key_ss] = selected_trend_range_dho
    selected_start_date_trends, selected_end_date_trends = selected_trend_range_dho
else: 
    selected_start_date_trends, selected_end_date_trends = st.session_state[dho_trend_date_key_ss]
    st.sidebar.warning("Trend date range selection error. Using previous/default.")
if selected_start_date_trends > selected_end_date_trends:
    st.sidebar.error("DHO Trends: Start date must be <= end date. Adjusting."); selected_end_date_trends = selected_start_date_trends
    st.session_state[dho_trend_date_key_ss][1] = selected_end_date_trends

# --- Section 1: District Performance Dashboard (KPIs) ---
st.header("📊 District Performance Dashboard")
if district_kpis_summary_data and isinstance(district_kpis_summary_data, dict) and \
   any(pd.notna(v_kpi) and v_kpi != 0 for k_kpi,v_kpi in district_kpis_summary_data.items() if k_kpi != 'total_zones_in_df' and isinstance(v_kpi, (int, float))): 
    structured_dho_kpi_list_val = structure_district_summary_kpis(district_kpis_summary_data, enriched_zone_df_display, f"Snapshot as of {data_as_of_ts.strftime('%d %b %Y')}")
    if structured_dho_kpi_list_val:
        num_kpis_dho = len(structured_dho_kpi_list_val); kpis_per_row = 4
        for i_row_dho in range(0, num_kpis_dho, kpis_per_row):
            cols_kpi_dho = st.columns(kpis_per_row)
            for idx_kpi_in_row, kpi_item_dho in enumerate(structured_dho_kpi_list_val[i_row_dho : i_row_dho + kpis_per_row]):
                with cols_kpi_dho[idx_kpi_in_row % kpis_per_row]: render_kpi_card(**kpi_item_dho)
    else: st.info("District KPIs could not be structured from available summary data.")
else: st.warning("District-wide summary KPIs unavailable or contain no significant data.")
st.divider()

# --- Section 2: In-Depth District Analysis Modules (Tabs) ---
st.header("🔍 In-Depth District Analysis Modules")
dho_tab_names_list = ["🗺️ Geospatial Overview", "📈 District Trends", "🆚 Zonal Comparison", "🎯 Intervention Planning"]
tab_map, tab_trends, tab_compare, tab_intervene = st.tabs(dho_tab_names_list)

with tab_map:
    st.subheader("Interactive District Health & Environmental Map")
    if isinstance(enriched_zone_df_display, pd.DataFrame) and not enriched_zone_df_display.empty and \
       ('geometry_obj' in enriched_zone_df_display.columns or 'geometry' in enriched_zone_df_display.columns):
        render_district_map_visualization(enriched_zone_df_display, 'avg_risk_score', f"Zonal Data as of {data_as_of_ts.strftime('%d %b %Y')}")
    else: st.warning("Map unavailable: Enriched zone data missing, empty, or lacks geometry.")

with tab_trends:
    trend_period_str_dho = f"{selected_start_date_trends.strftime('%d %b %Y')} - {selected_end_date_trends.strftime('%d %b %Y')}"
    st.subheader(f"District-Wide Health & Environmental Trends ({trend_period_str_dho})")
    
    df_health_trends_dho: pd.DataFrame = pd.DataFrame()
    if isinstance(historical_health_df_for_trends, pd.DataFrame) and not historical_health_df_for_trends.empty and 'encounter_date' in historical_health_df_for_trends.columns:
        df_health_trends_dho = historical_health_df_for_trends[
            (pd.to_datetime(historical_health_df_for_trends['encounter_date']).dt.date >= selected_start_date_trends) &
            (pd.to_datetime(historical_health_df_for_trends['encounter_date']).dt.date <= selected_end_date_trends)
        ].copy()

    df_iot_trends_dho: pd.DataFrame = pd.DataFrame()
    if isinstance(historical_iot_df_for_trends, pd.DataFrame) and not historical_iot_df_for_trends.empty and 'timestamp' in historical_iot_df_for_trends.columns:
        df_iot_trends_dho = historical_iot_df_for_trends[
            (pd.to_datetime(historical_iot_df_for_trends['timestamp']).dt.date >= selected_start_date_trends) &
            (pd.to_datetime(historical_iot_df_for_trends['timestamp']).dt.date <= selected_end_date_trends)
        ].copy()

    if df_health_trends_dho.empty and df_iot_trends_dho.empty: st.info(f"No health or IoT data for trend period: {trend_period_str_dho}")
    else:
        district_trends_data = calculate_district_wide_trends(df_health_trends_dho, df_iot_trends_dho, selected_start_date_trends, selected_end_date_trends, trend_period_str_dho)
        disease_inc_trends = district_trends_data.get("disease_incidence_trends", {})
        if disease_inc_trends:
            st.markdown("###### Key Disease Incidence (Weekly New/Active Cases - Unique Patients):")
            max_charts_row = 2; disease_keys = list(disease_inc_trends.keys())
            for i_dis_row in range(0, len(disease_keys), max_charts_row):
                cols_dis_trend = st.columns(max_charts_row)
                for j_dis_idx, key_dis_plot in enumerate(disease_keys[i_dis_row : i_dis_row + max_charts_row]):
                    series_dis_data = disease_inc_trends[key_dis_plot]
                    if isinstance(series_dis_data, pd.Series) and not series_dis_data.empty:
                        with cols_dis_trend[j_dis_idx % max_charts_row]: st.plotly_chart(plot_annotated_line_chart(series_dis_data, f"{key_dis_plot} New/Active Cases (Weekly)", y_values_are_counts=True, y_axis_label="# Unique Patients"), use_container_width=True)
        
        other_trends_cfg = {"Avg. Patient AI Risk Score Trend": (district_trends_data.get("avg_patient_ai_risk_trend"), "AI Risk Score"),
                            "Avg. Patient Daily Steps Trend": (district_trends_data.get("avg_patient_daily_steps_trend"), "Steps/Day"),
                            "Avg. Clinic CO2 Levels Trend (District)": (district_trends_data.get("avg_clinic_co2_trend"), "CO2 (ppm)")}
        for title_trend, (series_data, y_label) in other_trends_cfg.items():
            if isinstance(series_data, pd.Series) and not series_data.empty:
                y_is_count = "Steps" in y_label or "#" in y_label
                st.plotly_chart(plot_annotated_line_chart(series_data, title_trend, y_label, y_values_are_counts=y_is_count), use_container_width=True)
        for note_trend in district_trends_data.get("data_availability_notes", []): st.caption(f"Note (Trends Tab): {note_trend}")

with tab_compare:
    st.subheader("Comparative Zonal Analysis")
    if isinstance(enriched_zone_df_display, pd.DataFrame) and not enriched_zone_df_display.empty:
        zonal_compare_data = prepare_district_zonal_comparison_data(enriched_zone_df_display, f"Data as of {data_as_of_ts.strftime('%d %b %Y')}")
        compare_table_df = zonal_compare_data.get("zonal_comparison_table_df")
        if isinstance(compare_table_df, pd.DataFrame) and not compare_table_df.empty:
            st.markdown("###### **Aggregated Zonal Metrics Comparison Table:**")
            st.dataframe(compare_table_df, height=min(600, len(compare_table_df)*38 + 76), use_container_width=True)
            
            compare_metrics_cfg = zonal_compare_data.get("comparison_metrics_config", {})
            default_bar_metric_name = "Avg. AI Risk Score (Zone)"
            if default_bar_metric_name in compare_metrics_cfg:
                metric_details_bar = compare_metrics_cfg[default_bar_metric_name]
                risk_col_bar = metric_details_bar["col"]
                df_bar_compare = compare_table_df.reset_index()
                zone_id_col_bar = df_bar_compare.columns[0] # Assumes first col is zone identifier
                if risk_col_bar in df_bar_compare.columns and zone_id_col_bar in df_bar_compare.columns:
                    st.plotly_chart(plot_bar_chart(df_bar_compare, zone_id_col_bar, risk_col_bar, f"{default_bar_metric_name} by Zone", sort_by_col=risk_col_bar, sort_ascending_flag=False, x_axis_label_text="Zone", y_axis_label_text="Avg. AI Risk Score"), use_container_width=True)
        else: st.info("No data for zonal comparison table with current enriched zone data.")
        for note_compare in zonal_compare_data.get("data_availability_notes", []): st.caption(f"Note (Comparison Tab): {note_compare}")
    else: st.warning("Zonal comparison cannot be performed: Enriched district zone data missing or empty.")

with tab_intervene:
    st.subheader("Targeted Intervention Planning Assistant")
    if isinstance(enriched_zone_df_display, pd.DataFrame) and not enriched_zone_df_display.empty and intervention_criteria_options_data:
        crit_keys_list = list(intervention_criteria_options_data.keys())
        default_crit_sel = crit_keys_list[:min(2, len(crit_keys_list))]
        
        # Session state for multiselect
        interv_criteria_key_ss = "dho_intervention_criteria_multiselect_v2" # Unique key
        if interv_criteria_key_ss not in st.session_state: st.session_state[interv_criteria_key_ss] = default_crit_sel
        # Ensure default is valid if options changed
        current_selection_interv = [opt for opt in st.session_state[interv_criteria_key_ss] if opt in crit_keys_list]
        if not current_selection_interv and default_crit_sel: current_selection_interv = default_crit_sel
        elif not current_selection_interv and crit_keys_list : current_selection_interv = [crit_keys_list[0]] if crit_keys_list else []


        selected_criteria_interv_ui = st.multiselect(
            "Select Criteria to Identify Priority Zones (zones meeting ANY selected criterion shown):",
            options=crit_keys_list, default=current_selection_interv, key=f"{interv_criteria_key_ss}_widget"
        )
        st.session_state[interv_criteria_key_ss] = selected_criteria_interv_ui # Update session state
        
        interv_results = identify_priority_zones_for_intervention_planning(enriched_zone_df_display, selected_criteria_interv_ui, intervention_criteria_options_data, f"Data as of {data_as_of_ts.strftime('%d %b %Y')}")
        priority_zones_df_interv = interv_results.get("priority_zones_for_intervention_df")
        applied_criteria_list_interv = interv_results.get("applied_criteria_display_names", [])

        if isinstance(priority_zones_df_interv, pd.DataFrame) and not priority_zones_df_interv.empty:
            st.markdown(f"###### **{len(priority_zones_df_interv)} Zone(s) Flagged Based On: {', '.join(applied_criteria_list_interv) if applied_criteria_list_interv else 'Selected Criteria'}**")
            st.dataframe(priority_zones_df_interv, use_container_width=True, height=min(500, len(priority_zones_df_interv)*40 + 76), hide_index=True)
        elif selected_criteria_interv_ui: st.success(f"✅ No zones currently meet criteria: {', '.join(applied_criteria_list_interv) if applied_criteria_list_interv else 'None applied'}.")
        else: st.info("Please select one or more criteria above to identify priority zones.")
        for note_intervene in interv_results.get("data_availability_notes", []): st.caption(f"Note (Intervention Tab): {note_intervene}")
    else: st.warning("Intervention planning tools unavailable: Enriched zone data or criteria definitions missing/invalid.")

logger.info(f"DHO Strategic Command Center page loaded/refreshed. Data timestamp context: {data_as_of_ts.isoformat()}")
