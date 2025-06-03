# sentinel_project_root/pages/district_dashboard.py
# District Health Strategic Command Center for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np # For np.nan
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple
import os # For os.path.exists for logo

# --- Sentinel System Imports from Refactored Structure ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data, load_zone_data
    from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
    from data_processing.aggregation import get_district_summary_kpis
    from data_processing.helpers import hash_dataframe_safe # For caching
    from analytics.orchestrator import apply_ai_models # For AI enrichment before aggregation
    from visualization.ui_elements import render_kpi_card
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart
    
    # District Component specific imports
    from .district_components.kpi_structuring import structure_district_summary_kpis
    from .district_components.map_display import render_district_map_visualization
    from .district_components.trend_analysis import calculate_district_wide_trends
    from .district_components.comparison_data import prepare_district_zonal_comparison_data
    from .district_components.intervention_planning import (
        get_district_intervention_criteria_options,
        identify_priority_zones_for_intervention_planning
    )
except ImportError as e_dist_dash:
    import sys
    st.error(
        f"District Dashboard Import Error: {e_dist_dash}. "
        f"Please ensure all modules are correctly placed and dependencies installed. "
        f"Relevant Python Path: {sys.path}"
    )
    logger_dist = logging.getLogger(__name__)
    logger_dist.error(f"District Dashboard Import Error: {e_dist_dash}", exc_info=True)
    st.stop()

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Page Title and Introduction ---
st.title(f"🗺️ {settings.APP_NAME} - District Health Strategic Command Center")
st.markdown(f"**Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring.**")
st.divider()


# --- Data Aggregation and Preparation for DHO View (Cached) ---
@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    hash_funcs={pd.DataFrame: hash_dataframe_safe}, # Use safe hasher for DataFrames
    show_spinner="Aggregating district-level operational data..."
)
def get_dho_command_center_processed_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    """
    Loads, enriches, and aggregates data for the DHO Command Center.
    Returns:
        - Enriched Zone DataFrame (pandas DataFrame, not GeoDataFrame).
        - Full historical health DataFrame (AI enriched).
        - Full historical IoT DataFrame.
        - District summary KPIs dictionary.
        - Intervention criteria options dictionary.
    """
    log_ctx = "DHODatasetPrep"
    logger.info(f"({log_ctx}) Initializing full data pipeline simulation for DHO view...")
    
    # 1. Load Raw Data
    raw_health_df_dho = load_health_records(source_context=f"{log_ctx}/LoadHealth")
    raw_iot_df_dho = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadIoT")
    # load_zone_data now returns a pandas DataFrame with 'geometry_obj' and 'crs' columns
    base_zone_df_dho = load_zone_data(source_context=f"{log_ctx}/LoadZoneData")

    if base_zone_df_dho.empty or 'zone_id' not in base_zone_df_dho.columns:
        logger.error(f"({log_ctx}) Base zone attribute/geometry DataFrame failed to load or is invalid. DHO dashboard heavily impacted.")
        # Return empty structures with expected keys/columns if possible
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    # 2. Enrich Health Data with AI models
    if isinstance(raw_health_df_dho, pd.DataFrame) and not raw_health_df_dho.empty:
        ai_enriched_health_df_dho, _ = apply_ai_models(raw_health_df_dho.copy(), source_context=f"{log_ctx}/AIEnrichHealth")
    else:
        logger.warning(f"({log_ctx}) Raw health data for DHO view is empty or invalid. AI enrichment skipped.")
        ai_enriched_health_df_dho = pd.DataFrame()

    # 3. Enrich Zone DataFrame with Health and IoT Aggregates
    # This function now takes pandas DataFrames and returns a pandas DataFrame.
    enriched_zone_df_for_dho = enrich_zone_geodata_with_health_aggregates(
        zone_df=base_zone_df_dho, 
        health_df=ai_enriched_health_df_dho,
        iot_df=raw_iot_df_dho, 
        source_context=f"{log_ctx}/EnrichZoneDF"
    )
    if enriched_zone_df_for_dho.empty:
        logger.warning(f"({log_ctx}) Zone DataFrame enrichment resulted in empty DataFrame. Using base zone data if available for map context.")
        # Fallback for map: if enrichment fails but base_zone_df_dho has geometry, map might still show zones
        enriched_zone_df_for_dho = base_zone_df_dho if not base_zone_df_dho.empty else pd.DataFrame()


    # 4. Calculate District Summary KPIs
    district_summary_kpis_map = get_district_summary_kpis(
        enriched_zone_df=enriched_zone_df_for_dho, 
        source_context=f"{log_ctx}/CalcDistrictKPIs"
    )
    
    # 5. Get Intervention Criteria Options (based on columns available in enriched_zone_df)
    intervention_criteria_options_map = get_district_intervention_criteria_options(
        district_zone_df_sample_check=enriched_zone_df_for_dho.head(2) if not enriched_zone_df_for_dho.empty else None
    )
    
    df_shape_log_dho = enriched_zone_df_for_dho.shape if isinstance(enriched_zone_df_for_dho, pd.DataFrame) else 'N/A'
    logger.info(f"({log_ctx}) DHO data preparation complete. Enriched Zone DF shape: {df_shape_log_dho}")
    return enriched_zone_df_for_dho, ai_enriched_health_df_dho, raw_iot_df_dho, district_summary_kpis_map, intervention_criteria_options_map


# --- Load and Prepare Data for the Dashboard ---
# Use session state to load data once per session or if filters change (more complex state mgmt needed for filter-based reload)
# For simplicity of this refactor, caching is primary; complex session state for re-filtering is omitted here.
try:
    (
        enriched_zone_df_display,
        historical_health_df_for_trends, # Full AI-enriched health data for trend calculations
        historical_iot_df_for_trends,    # Full IoT data for trend calculations
        district_kpis_summary_data,
        intervention_criteria_options_data
    ) = get_dho_command_center_processed_datasets()
except Exception as e_dho_data_main:
    logger.error(f"DHO Dashboard: Failed to load or process main dashboard datasets: {e_dho_data_main}", exc_info=True)
    st.error(f"Error loading DHO dashboard data: {str(e_dho_data_main)}. Please check logs or contact support.")
    # Provide empty defaults to allow UI to render parts of the page without crashing
    enriched_zone_df_display, historical_health_df_for_trends, historical_iot_df_for_trends = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    district_kpis_summary_data, intervention_criteria_options_data = {}, {}
    # st.stop() # Consider stopping if data is absolutely critical

# Display data timestamp context
data_as_of_timestamp = pd.Timestamp('now') # Placeholder; ideally from data source's latest date
st.caption(f"Data presented as of: {data_as_of_timestamp.strftime('%d %b %Y, %H:%M %Z')}")


# --- Sidebar Filters ---
if os.path.exists(settings.APP_LOGO_SMALL_PATH):
    st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
st.sidebar.header("Analysis Filters")

# Date Range Picker for Trend Analysis
abs_min_date_dho_trends_ui = data_as_of_timestamp.date() - timedelta(days=365 * 2) # Up to 2 years back
abs_max_date_dho_trends_ui = data_as_of_timestamp.date()
default_end_date_dho_ui = abs_max_date_dho_trends_ui
default_start_date_dho_ui = default_end_date_dho_ui - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3 - 1) # ~90 days
if default_start_date_dho_ui < abs_min_date_dho_trends_ui:
    default_start_date_dho_ui = abs_min_date_dho_trends_ui

# Session state for DHO trend date range
dho_trend_date_session_key = "dho_dashboard_trend_date_range"
if dho_trend_date_session_key not in st.session_state:
    st.session_state[dho_trend_date_session_key] = [default_start_date_dho_ui, default_end_date_dho_ui]

selected_trend_date_range_dho_ui = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:",
    value=st.session_state[dho_trend_date_session_key],
    min_value=abs_min_date_dho_trends_ui,
    max_value=abs_max_date_dho_trends_ui,
    key=f"{dho_trend_date_session_key}_widget"
)
if isinstance(selected_trend_date_range_dho_ui, (list,tuple)) and len(selected_trend_date_range_dho_ui) == 2:
    st.session_state[dho_trend_date_session_key] = selected_trend_date_range_dho_ui
    selected_start_date_for_trends, selected_end_date_for_trends = selected_trend_date_range_dho_ui
else:
    selected_start_date_for_trends, selected_end_date_for_trends = default_start_date_dho_ui, default_end_date_dho_ui
    st.session_state[dho_trend_date_session_key] = [selected_start_date_for_trends, selected_end_date_for_trends]


if selected_start_date_for_trends > selected_end_date_for_trends:
    st.sidebar.error("DHO Trends: Start date must be on or before the end date. Adjusting end date.")
    selected_end_date_for_trends = selected_start_date_for_trends
    st.session_state[dho_trend_date_session_key][1] = selected_end_date_for_trends


# --- Section 1: District Performance Dashboard (KPIs) ---
st.header("📊 District Performance Dashboard")
if district_kpis_summary_data and isinstance(district_kpis_summary_data, dict) and \
   any(pd.notna(v) and v != 0 for k,v in district_kpis_summary_data.items() if k != 'total_zones_in_df' and isinstance(v, (int, float))): 
    
    structured_dho_kpi_list = structure_district_summary_kpis(
        district_kpis_summary_input=district_kpis_summary_data,
        district_enriched_zone_df_context=enriched_zone_df_display, # Pass for context like total zones
        reporting_period_context_str=f"Snapshot as of {data_as_of_timestamp.strftime('%d %b %Y')}"
    )
    if structured_dho_kpi_list:
        num_total_dho_kpis = len(structured_dho_kpi_list)
        kpis_per_row_dho = 4 # Max 4 KPIs per row
        for i_row_kpi_dho in range(0, num_total_dho_kpis, kpis_per_row_dho):
            kpi_cols_for_this_row = st.columns(kpis_per_row_dho)
            for kpi_idx_in_row, kpi_data_item_dho in enumerate(structured_dho_kpi_list[i_row_kpi_dho : i_row_kpi_dho + kpis_per_row_dho]):
                with kpi_cols_for_this_row[kpi_idx_in_row]:
                    render_kpi_card(**kpi_data_item_dho) # Use new renderer
    else: 
        st.info("District KPIs could not be structured from the available summary data. Check component logs.")
else:
    st.warning("District-wide summary KPIs are currently unavailable or contain no significant data to display.")
st.divider()


# --- Section 2: In-Depth District Analysis Modules (Tabs) ---
st.header("🔍 In-Depth District Analysis Modules")
dho_tab_titles = ["🗺️ Geospatial Overview", "📈 District Trends", "🆚 Zonal Comparison", "🎯 Intervention Planning"]
tab_map_view, tab_trends_view, tab_comparison_view, tab_intervention_view = st.tabs(dho_tab_titles)

with tab_map_view:
    st.subheader("Interactive District Health & Environmental Map")
    if isinstance(enriched_zone_df_display, pd.DataFrame) and not enriched_zone_df_display.empty and \
       ('geometry_obj' in enriched_zone_df_display.columns or 'geometry' in enriched_zone_df_display.columns): # Check for geometry data
        
        render_district_map_visualization( # Uses component from .district_components
            enriched_district_zone_df=enriched_zone_df_display,
            default_metric_col_for_map_display='avg_risk_score', # Default metric to show
            reporting_period_context_str=f"Zonal Data as of {data_as_of_timestamp.strftime('%d %b %Y')}"
        )
    else:
        st.warning(
            "Map visualization unavailable: Enriched district zone data is missing, empty, "
            "or lacks necessary geometry information ('geometry_obj' or 'geometry' column)."
        )

with tab_trends_view:
    trend_period_display_dho = f"{selected_start_date_for_trends.strftime('%d %b %Y')} - {selected_end_date_for_trends.strftime('%d %b %Y')}"
    st.subheader(f"District-Wide Health & Environmental Trends ({trend_period_display_dho})")
    
    # Filter historical data for the selected trend period
    df_health_for_trends_tab_dho = pd.DataFrame()
    if isinstance(historical_health_df_for_trends, pd.DataFrame) and not historical_health_df_for_trends.empty and \
       'encounter_date' in historical_health_df_for_trends.columns:
        df_health_for_trends_tab_dho = historical_health_df_for_trends[
            (pd.to_datetime(historical_health_df_for_trends['encounter_date']).dt.date >= selected_start_date_for_trends) &
            (pd.to_datetime(historical_health_df_for_trends['encounter_date']).dt.date <= selected_end_date_for_trends)
        ].copy()

    df_iot_for_trends_tab_dho = pd.DataFrame()
    if isinstance(historical_iot_df_for_trends, pd.DataFrame) and not historical_iot_df_for_trends.empty and \
       'timestamp' in historical_iot_df_for_trends.columns:
        df_iot_for_trends_tab_dho = historical_iot_df_for_trends[
            (pd.to_datetime(historical_iot_df_for_trends['timestamp']).dt.date >= selected_start_date_for_trends) &
            (pd.to_datetime(historical_iot_df_for_trends['timestamp']).dt.date <= selected_end_date_for_trends)
        ].copy()

    if df_health_for_trends_tab_dho.empty and df_iot_for_trends_tab_dho.empty:
        st.info(f"No health or IoT data available for the selected trend period: {trend_period_display_dho}")
    else:
        district_trends_data_map = calculate_district_wide_trends( # Uses component
            health_df_filtered_for_period=df_health_for_trends_tab_dho,
            iot_df_filtered_for_period=df_iot_for_trends_tab_dho,
            trend_start_date_context=selected_start_date_for_trends,
            trend_end_date_context=selected_end_date_for_trends,
            reporting_period_display_str=trend_period_display_dho
        )
        
        disease_incidence_trends_map = district_trends_data_map.get("disease_incidence_trends", {})
        if disease_incidence_trends_map:
            st.markdown("###### Key Disease Incidence (Weekly New/Active Cases - Unique Patients):")
            # Display disease trends, perhaps 2 per row
            max_disease_charts_per_row = 2
            disease_trend_keys_list = list(disease_incidence_trends_map.keys())
            for i_disease_row in range(0, len(disease_trend_keys_list), max_disease_charts_per_row):
                cols_disease_trend_row = st.columns(max_disease_charts_per_row)
                for j_disease_idx_in_row, key_disease_trend_plot in enumerate(disease_trend_keys_list[i_disease_row : i_disease_row + max_disease_charts_per_row]):
                    series_disease_data_plot = disease_incidence_trends_map[key_disease_trend_plot]
                    if isinstance(series_disease_data_plot, pd.Series) and not series_disease_data_plot.empty:
                        with cols_disease_trend_row[j_disease_idx_in_row]:
                            st.plotly_chart(
                                plot_annotated_line_chart(
                                    series_disease_data_plot, 
                                    chart_title=f"{key_disease_trend_plot} New/Active Cases (Weekly)", 
                                    y_values_are_counts=True, y_axis_label="# Unique Patients"
                                ), use_container_width=True
                            )
        
        # Plot other general district trends
        other_district_trends_to_display = {
            "Avg. Patient AI Risk Score Trend": (district_trends_data_map.get("avg_patient_ai_risk_trend"), "AI Risk Score"),
            "Avg. Patient Daily Steps Trend": (district_trends_data_map.get("avg_patient_daily_steps_trend"), "Steps/Day"),
            "Avg. Clinic CO2 Levels Trend (District)": (district_trends_data_map.get("avg_clinic_co2_trend"), "CO2 (ppm)")
        }
        for trend_plot_title, (trend_series_data_val, y_axis_label_val) in other_district_trends_to_display.items():
            if isinstance(trend_series_data_val, pd.Series) and not trend_series_data_val.empty:
                y_is_count_for_trend = "Steps" in y_axis_label_val or "#" in y_axis_label_val # Heuristic
                st.plotly_chart(
                    plot_annotated_line_chart(
                        trend_series_data_val, chart_title=trend_plot_title, 
                        y_axis_label=y_axis_label_val, y_values_are_counts=y_is_count_for_trend
                    ), use_container_width=True
                )
        
        if district_trends_data_map.get("data_availability_notes"):
            for note_trend_tab in district_trends_data_map["data_availability_notes"]: st.caption(f"Note (Trends Tab): {note_trend_tab}")

with tab_comparison_view:
    st.subheader("Comparative Zonal Analysis")
    if isinstance(enriched_zone_df_display, pd.DataFrame) and not enriched_zone_df_display.empty:
        zonal_comparison_data_map = prepare_district_zonal_comparison_data( # Uses component
            enriched_district_zone_df=enriched_zone_df_display,
            reporting_period_context_str=f"Data as of {data_as_of_timestamp.strftime('%d %b %Y')}"
        )
        
        df_comparison_table_for_display = zonal_comparison_data_map.get("zonal_comparison_table_df")
        if isinstance(df_comparison_table_for_display, pd.DataFrame) and not df_comparison_table_for_display.empty:
            st.markdown("###### **Aggregated Zonal Metrics Comparison Table:**")
            # Displaying the table - index is Zone Name
            st.dataframe(
                df_comparison_table_for_display, 
                height=min(600, len(df_comparison_table_for_display)*38 + 76), # Dynamic height with header
                use_container_width=True
            )
            
            # Example Bar Chart for a key comparison metric (e.g., Avg. AI Risk Score by Zone)
            comparison_metrics_config_map = zonal_comparison_data_map.get("comparison_metrics_config", {})
            default_bar_metric_display_name_comp = "Avg. AI Risk Score (Zone)" # Default metric to chart
            
            if default_bar_metric_display_name_comp in comparison_metrics_config_map:
                metric_details_for_bar_chart = comparison_metrics_config_map[default_bar_metric_display_name_comp]
                actual_risk_col_name_for_bar = metric_details_for_bar_chart["col"]
                
                # Table might have Zone Name as index, need it as a column for px.bar
                df_for_bar_chart_comparison = df_comparison_table_for_display.reset_index() 
                # Identify the zone identifier column (likely 'Zone / Sector' from index name or 'name'/'zone_id')
                zone_id_col_for_bar_chart = df_for_bar_chart_comparison.columns[0] # Assuming first col after reset_index is the zone identifier
                
                if actual_risk_col_name_for_bar in df_for_bar_chart_comparison.columns and \
                   zone_id_col_for_bar_chart in df_for_bar_chart_comparison.columns:
                    st.plotly_chart(
                        plot_bar_chart(
                            df_for_bar_chart_comparison, x_col_name=zone_id_col_for_bar_chart, y_col_name=actual_risk_col_name_for_bar, 
                            chart_title=f"{default_bar_metric_display_name_comp} by Zone", 
                            sort_by_col=actual_risk_col_name_for_bar, sort_ascending_flag=False, # Highest risk first
                            x_axis_label_text="Zone", y_axis_label_text="Avg. AI Risk Score"
                        ), use_container_width=True
                    )
        else:
            st.info("No data available for the zonal comparison table with current enriched zone data.")

        if zonal_comparison_data_map.get("data_availability_notes"):
            for note_compare_tab in zonal_comparison_data_map["data_availability_notes"]: st.caption(f"Note (Comparison Tab): {note_compare_tab}")
    else:
        st.warning("Zonal comparison cannot be performed: Enriched district zone data is missing or empty.")

with tab_intervention_view:
    st.subheader("Targeted Intervention Planning Assistant")
    if isinstance(enriched_zone_df_display, pd.DataFrame) and not enriched_zone_df_display.empty and \
       intervention_criteria_options_data: # Check if criteria options were loaded
        
        intervention_criteria_keys_list = list(intervention_criteria_options_data.keys())
        # Default selection: first two criteria if available, else first one, else empty
        default_criteria_selection_ui = intervention_criteria_keys_list[:min(2, len(intervention_criteria_keys_list))]
        
        selected_criteria_for_intervention_ui = st.multiselect(
            "Select Criteria to Identify Priority Zones (zones meeting ANY selected criterion will be shown):",
            options=intervention_criteria_keys_list,
            default=default_criteria_selection_ui,
            key="dho_intervention_criteria_multiselect_main"
        )
        
        intervention_results_map = identify_priority_zones_for_intervention_planning( # Uses component
            enriched_district_zone_df=enriched_zone_df_display,
            selected_criteria_display_names_list=selected_criteria_for_intervention_ui,
            available_intervention_criteria_config=intervention_criteria_options_data, 
            reporting_period_context_str=f"Data as of {data_as_of_timestamp.strftime('%d %b %Y')}"
        )
        
        df_priority_zones_for_display_interv = intervention_results_map.get("priority_zones_for_intervention_df")
        applied_criteria_names_list_interv = intervention_results_map.get("applied_criteria_display_names", [])

        if isinstance(df_priority_zones_for_display_interv, pd.DataFrame) and not df_priority_zones_for_display_interv.empty:
            st.markdown(
                f"###### **{len(df_priority_zones_for_display_interv)} Zone(s) Flagged for Intervention Based On: "
                f"{', '.join(applied_criteria_names_list_interv) if applied_criteria_names_list_interv else 'Selected Criteria'}**"
            )
            # Display table of flagged zones
            st.dataframe(
                df_priority_zones_for_display_interv, 
                use_container_width=True, 
                height=min(500, len(df_priority_zones_for_display_interv)*40 + 76), # Dynamic height
                hide_index=True # Assuming index is not meaningful for direct display here
            )
        elif selected_criteria_for_intervention_ui: # Criteria selected but no zones met them
             st.success(
                 f"✅ No zones currently meet the selected intervention criteria: "
                 f"{', '.join(applied_criteria_names_list_interv) if applied_criteria_names_list_interv else 'None applied due to data issues'}."
            )
        else: # No criteria selected by user
             st.info("Please select one or more criteria above to identify potential priority zones for intervention.")

        if intervention_results_map.get("data_availability_notes"):
            for note_intervene_tab in intervention_results_map["data_availability_notes"]: st.caption(f"Note (Intervention Tab): {note_intervene_tab}")
    else:
        st.warning(
            "Intervention planning tools unavailable: Enriched district zone data or "
            "intervention criteria definitions are missing/invalid."
        )

logger.info(f"DHO Strategic Command Center page loaded/refreshed. Data timestamp context: {data_as_of_timestamp.isoformat()}")
