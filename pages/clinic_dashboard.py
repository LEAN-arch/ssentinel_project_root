# sentinel_project_root/pages/clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np # For np.nan
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple, List
import os # For checking logo path

# --- Sentinel System Imports from Refactored Structure ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis
    from analytics.orchestrator import apply_ai_models # For potential AI-enriched data if needed by components
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart
    
    # Clinic specific components from the new structure
    from .clinic_components.env_details import prepare_clinic_environmental_detail_data
    from .clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from .clinic_components.epi_data import calculate_clinic_epidemiological_data
    from .clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from .clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from .clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e_clinic_dash:
    import sys
    st.error(
        f"Clinic Dashboard Import Error: {e_clinic_dash}. "
        f"Please ensure all modules are correctly placed and dependencies installed. "
        f"Relevant Python Path: {sys.path}"
    )
    logger = logging.getLogger(__name__) # Attempt to get logger
    logger.error(f"Clinic Dashboard Import Error: {e_clinic_dash}", exc_info=True)
    st.stop()

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Page Title and Introduction ---
# Page config is set in app.py
st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

# --- Data Loading Function for this Dashboard ---
@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading comprehensive clinic operational dataset...",
    hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df, index=True)} # More robust hash for DFs
)
def get_clinic_console_processed_data(
    selected_period_start_date: date,
    selected_period_end_date: date
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], bool]:
    """
    Loads, enriches (with AI scores), and filters data for the Clinic Console.
    Returns:
        - Full historical health DataFrame (AI enriched, not period filtered).
        - Period-filtered health DataFrame (AI enriched).
        - Period-filtered IoT DataFrame.
        - Clinic summary KPIs for the period.
        - Flag indicating IoT data source availability.
    """
    log_ctx = "ClinicConsoleDataLoad"
    logger.info(f"({log_ctx}) Loading data for period: {selected_period_start_date.isoformat()} to {selected_period_end_date.isoformat()}")
    
    # Load raw data
    raw_health_df = load_health_records(source_context=f"{log_ctx}/LoadRawHealthRecs")
    raw_iot_df = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadRawIoTData")
    
    iot_source_file_exists = os.path.exists(settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH) # Check if source file is there
    iot_data_actually_loaded = isinstance(raw_iot_df, pd.DataFrame) and not raw_iot_df.empty
    is_iot_data_available = iot_source_file_exists and iot_data_actually_loaded # Both file exists and data loaded

    # Enrich health data with AI models (risk scores, etc.)
    # This is done on the full historical set before period filtering for consistency if AI models need broader context.
    # However, for dashboard performance, if AI models can run on period data, that might be better.
    # Current `apply_ai_models` is okay with full or partial.
    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        ai_enriched_health_df_full, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrichHealth")
    else:
        logger.warning(f"({log_ctx}) Raw health data for clinic is empty or invalid. AI enrichment skipped. Dashboard may lack some insights.")
        ai_enriched_health_df_full = pd.DataFrame() # Empty DF with schema if raw load failed

    # Filter AI-enriched health data for the selected period
    df_health_for_period_display = pd.DataFrame() # Default empty
    if not ai_enriched_health_df_full.empty and 'encounter_date' in ai_enriched_health_df_full.columns:
        # Ensure encounter_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(ai_enriched_health_df_full['encounter_date']):
            ai_enriched_health_df_full['encounter_date'] = pd.to_datetime(ai_enriched_health_df_full['encounter_date'], errors='coerce')
        
        df_health_for_period_display = ai_enriched_health_df_full[
            (ai_enriched_health_df_full['encounter_date'].notna()) & # Ensure date is not NaT
            (ai_enriched_health_df_full['encounter_date'].dt.date >= selected_period_start_date) &
            (ai_enriched_health_df_full['encounter_date'].dt.date <= selected_period_end_date)
        ].copy()
    
    # Filter IoT data for the selected period
    df_iot_for_period_display = pd.DataFrame() # Default empty
    if is_iot_data_available and 'timestamp' in raw_iot_df.columns:
         # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(raw_iot_df['timestamp']):
            raw_iot_df['timestamp'] = pd.to_datetime(raw_iot_df['timestamp'], errors='coerce')

        df_iot_for_period_display = raw_iot_df[
            (raw_iot_df['timestamp'].notna()) &
            (raw_iot_df['timestamp'].dt.date >= selected_period_start_date) &
            (raw_iot_df['timestamp'].dt.date <= selected_period_end_date)
        ].copy()

    # Get clinic summary KPIs for the period using the period-filtered health data
    clinic_summary_kpis_for_period: Dict[str, Any] = {}
    if not df_health_for_period_display.empty:
        try:
            clinic_summary_kpis_for_period = get_clinic_summary_kpis(
                health_df_period=df_health_for_period_display,
                source_context=f"{log_ctx}/PeriodSummaryKPIs"
            )
        except Exception as e_summary_kpi:
            logger.error(f"({log_ctx}) Error calculating clinic summary KPIs: {e_summary_kpi}", exc_info=True)
            clinic_summary_kpis_for_period = {"test_summary_details": {}} # Ensure key exists
    else:
        logger.info(f"({log_ctx}) No health data in selected period for clinic summary KPIs. Using defaults.")
        clinic_summary_kpis_for_period = {"test_summary_details": {}} # Default structure

    return ai_enriched_health_df_full, df_health_for_period_display, df_iot_for_period_display, clinic_summary_kpis_for_period, is_iot_data_available


# --- Sidebar Filters Setup ---
if os.path.exists(settings.APP_LOGO_SMALL_PATH):
    st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
st.sidebar.header("Console Filters")

# Date Range Picker for Clinic Console
abs_min_date_clinic_console = date.today() - timedelta(days=365 * 1) # Allow up to 1 year back
abs_max_date_clinic_console = date.today()

default_end_date_console_ui = abs_max_date_clinic_console
default_start_date_console_ui = default_end_date_console_ui - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1) # Default 30 days
if default_start_date_console_ui < abs_min_date_clinic_console:
    default_start_date_console_ui = abs_min_date_clinic_console

# Session state for date range picker
date_range_session_key_clinic = "clinic_console_date_range_selection"
if date_range_session_key_clinic not in st.session_state:
    st.session_state[date_range_session_key_clinic] = [default_start_date_console_ui, default_end_date_console_ui]

selected_date_range_clinic_ui = st.sidebar.date_input(
    "Select Date Range for Clinic Review:",
    value=st.session_state[date_range_session_key_clinic],
    min_value=abs_min_date_clinic_console,
    max_value=abs_max_date_clinic_console,
    key=f"{date_range_session_key_clinic}_widget"
)
# Update session state and unpack
if isinstance(selected_date_range_clinic_ui, (list, tuple)) and len(selected_date_range_clinic_ui) == 2:
    st.session_state[date_range_session_key_clinic] = selected_date_range_clinic_ui
    start_date_query_clinic, end_date_query_clinic = selected_date_range_clinic_ui
else: # Fallback
    start_date_query_clinic, end_date_query_clinic = default_start_date_console_ui, default_end_date_console_ui
    st.session_state[date_range_session_key_clinic] = [start_date_query_clinic, end_date_query_clinic] # Correct session state

# Validate date range
if start_date_query_clinic > end_date_query_clinic:
    st.sidebar.error("Clinic Console: Start date must be on or before end date. Adjusting end date.")
    end_date_query_clinic = start_date_query_clinic
    st.session_state[date_range_session_key_clinic][1] = end_date_query_clinic # Update session state

# Limit query range to avoid performance issues (e.g., 90 days max)
MAX_QUERY_DAYS_CLINIC = 90 
if (end_date_query_clinic - start_date_query_clinic).days > MAX_QUERY_DAYS_CLINIC:
    st.sidebar.warning(f"Date range too large. Limiting to {MAX_QUERY_DAYS_CLINIC} days from start date for performance.")
    end_date_query_clinic = start_date_query_clinic + timedelta(days=MAX_QUERY_DAYS_CLINIC -1)
    if end_date_query_clinic > abs_max_date_clinic_console: end_date_query_clinic = abs_max_date_clinic_console
    st.session_state[date_range_session_key_clinic] = [start_date_query_clinic, end_date_query_clinic]


# --- Load Data Based on Selected Filters ---
current_reporting_period_display_str = f"{start_date_query_clinic.strftime('%d %b %Y')} - {end_date_query_clinic.strftime('%d %b %Y')}"

try:
    (full_historical_health_df_clinic, # AI enriched, not period filtered (for supply forecast context)
     health_df_for_period_clinic_tabs,    # AI enriched, period filtered
     iot_df_for_period_clinic_tabs,       # Period filtered
     clinic_summary_kpis_for_period_data, # From aggregation using period health data
     iot_data_source_is_available) = get_clinic_console_processed_data(start_date_query_clinic, end_date_query_clinic)
except Exception as e_main_clinic_data_load:
    logger.error(f"Clinic Dashboard: Main data loading failed: {e_main_clinic_data_load}", exc_info=True)
    st.error(f"Error loading clinic dashboard data: {str(e_main_clinic_data_load)}. Please contact support.")
    # Provide empty defaults to allow UI to render without crashing
    full_historical_health_df_clinic, health_df_for_period_clinic_tabs, iot_df_for_period_clinic_tabs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    clinic_summary_kpis_for_period_data = {"test_summary_details": {}}
    iot_data_source_is_available = False
    # st.stop() # Consider stopping if data is absolutely critical for any display

if not iot_data_source_is_available:
    st.sidebar.warning("IoT environmental data source appears unavailable. Some environmental metrics may be missing.")

st.info(f"Displaying Clinic Console data for period: **{current_reporting_period_display_str}**")

# --- Section 1: Top-Level KPIs ---
st.header("🚀 Clinic Performance & Environment Snapshot")

# Structure and display Main Service KPIs
if clinic_summary_kpis_for_period_data and isinstance(clinic_summary_kpis_for_period_data.get("test_summary_details"), dict):
    main_kpi_cards_list_clinic = structure_main_clinic_kpis(
        clinic_service_kpis_summary_data=clinic_summary_kpis_for_period_data,
        reporting_period_context_str=current_reporting_period_display_str
    )
    disease_specific_kpi_cards_list_clinic = structure_disease_specific_clinic_kpis(
        clinic_service_kpis_summary_data=clinic_summary_kpis_for_period_data,
        reporting_period_context_str=current_reporting_period_display_str
    )
    
    if main_kpi_cards_list_clinic:
        st.markdown("##### **Overall Service Performance:**")
        main_kpi_cols = st.columns(min(len(main_kpi_cards_list_clinic), 4)) # Max 4 KPIs per row
        for i, kpi_data_main_item in enumerate(main_kpi_cards_list_clinic):
            with main_kpi_cols[i % 4]: # Cycle through columns
                render_kpi_card(**kpi_data_main_item) # Use new renderer
    
    if disease_specific_kpi_cards_list_clinic:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**")
        disease_kpi_cols = st.columns(min(len(disease_specific_kpi_cards_list_clinic), 4))
        for i, kpi_data_disease_item in enumerate(disease_specific_kpi_cards_list_clinic):
            with disease_kpi_cols[i % 4]:
                render_kpi_card(**kpi_data_disease_item)
else:
    st.warning(f"Core clinic performance KPIs could not be generated for {current_reporting_period_display_str}. Check data sources or component logs.")

# Clinic Environment Quick Check KPIs
st.markdown("##### **Clinic Environment Quick Check:**")
env_summary_kpis_quick_check = get_clinic_environmental_summary_kpis(
    iot_df_period=iot_df_for_period_clinic_tabs, # Use period-filtered IoT data
    source_context="ClinicDash/EnvQuickCheckKPIs"
)

has_meaningful_env_data_for_kpis = env_summary_kpis_quick_check and any(
    pd.notna(val) and (val != 0 if "count" in key else True) # Check if count is non-zero, or other values are non-NaN
    for key, val in env_summary_kpis_quick_check.items()
    if isinstance(val, (int, float)) and ("avg_" in key or "_count" in key or "_flag" in key or "_ppm" in key or "_ugm3" in key) # Heuristic for relevant numeric KPIs
)

if has_meaningful_env_data_for_kpis:
    env_kpi_cols_quick_check = st.columns(4)
    with env_kpi_cols_quick_check[0]:
        co2_val_qc = env_summary_kpis_quick_check.get('avg_co2_overall_ppm', np.nan)
        co2_status_qc = "HIGH_RISK" if pd.notna(co2_val_qc) and co2_val_qc > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM else \
                        ("MODERATE_CONCERN" if pd.notna(co2_val_qc) and co2_val_qc > settings.ALERT_AMBIENT_CO2_HIGH_PPM else "ACCEPTABLE")
        render_kpi_card(
            title="Avg. CO2", value_str=f"{co2_val_qc:.0f}" if pd.notna(co2_val_qc) else "N/A",
            units="ppm", icon="💨", status_level=co2_status_qc,
            help_text=f"Average CO2 in monitored areas. Target < {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm."
        )
    with env_kpi_cols_quick_check[1]:
        pm25_val_qc = env_summary_kpis_quick_check.get('avg_pm25_overall_ugm3', np.nan)
        pm25_status_qc = "HIGH_RISK" if pd.notna(pm25_val_qc) and pm25_val_qc > settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 else \
                         ("MODERATE_CONCERN" if pd.notna(pm25_val_qc) and pm25_val_qc > settings.ALERT_AMBIENT_PM25_HIGH_UGM3 else "ACCEPTABLE")
        render_kpi_card(
            title="Avg. PM2.5", value_str=f"{pm25_val_qc:.1f}" if pd.notna(pm25_val_qc) else "N/A",
            units="µg/m³", icon="🌫️", status_level=pm25_status_qc,
            help_text=f"Average PM2.5 particulate matter. Target < {settings.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³."
        )
    with env_kpi_cols_quick_check[2]:
        occupancy_val_qc = env_summary_kpis_quick_check.get('avg_waiting_room_occupancy_overall_persons', np.nan)
        occupancy_status_qc = "MODERATE_CONCERN" if pd.notna(occupancy_val_qc) and occupancy_val_qc > settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX else "ACCEPTABLE"
        render_kpi_card(
            title="Avg. Waiting Occupancy", value_str=f"{occupancy_val_qc:.1f}" if pd.notna(occupancy_val_qc) else "N/A",
            units="persons", icon="👨‍👩‍👧‍👦", status_level=occupancy_status_qc,
            help_text=f"Average occupancy in waiting areas. Target < {settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons."
        )
    with env_kpi_cols_quick_check[3]:
        noise_alerts_qc_count = env_summary_kpis_quick_check.get('rooms_noise_high_alert_latest_count', 0)
        noise_status_qc = "HIGH_CONCERN" if noise_alerts_qc_count > 1 else ("MODERATE_CONCERN" if noise_alerts_qc_count == 1 else "ACCEPTABLE")
        render_kpi_card(
            title="High Noise Alerts", value_str=str(noise_alerts_qc_count),
            units="areas", icon="🔊", status_level=noise_status_qc,
            help_text=f"Areas with sustained noise > {settings.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA based on latest readings."
        )
else: # No meaningful IoT data for quick check KPIs
    if iot_data_source_is_available: # Source exists, but no data for this period
        st.info("No significant environmental IoT data available for this period to display snapshot KPIs.")
    else: # Source itself is likely missing/unavailable
        st.caption("Environmental IoT data source is generally unavailable for this clinic. Monitoring snapshot is limited.")
st.divider()


# --- Tabbed Interface for Detailed Operational Areas ---
st.header("🛠️ Operational Areas Deep Dive")
clinic_console_tab_names = ["📈 Local Epidemiology", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_epi, tab_testing, tab_supply, tab_patient, tab_env = st.tabs(clinic_console_tab_names)

with tab_epi:
    st.subheader(f"Local Epidemiological Intelligence ({current_reporting_period_display_str})")
    if not health_df_for_period_clinic_tabs.empty:
        epi_data_for_tab_display = calculate_clinic_epidemiological_data(
            filtered_health_df_clinic_period=health_df_for_period_clinic_tabs,
            reporting_period_context_str=current_reporting_period_display_str,
            # condition_filter_for_demographics can be made a selectbox here if needed for this tab
        )
        
        df_symptom_trends_for_plot = epi_data_for_tab_display.get("symptom_trends_weekly_top_n_df")
        if isinstance(df_symptom_trends_for_plot, pd.DataFrame) and not df_symptom_trends_for_plot.empty:
            st.plotly_chart(plot_bar_chart(
                df_input=df_symptom_trends_for_plot, x_col_name='week_start_date', y_col_name='count',
                chart_title="Weekly Symptom Frequency (Top Reported)", color_col_name='symptom',
                bar_mode_style='group', y_values_are_counts_flag=True,
                x_axis_label_text="Week Starting", y_axis_label_text="Symptom Encounters"
            ), use_container_width=True)
        
        malaria_rdt_name_from_config = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT")
        malaria_positivity_trend_series = epi_data_for_tab_display.get("key_test_positivity_trends", {}).get(malaria_rdt_name_from_config)
        if isinstance(malaria_positivity_trend_series, pd.Series) and not malaria_positivity_trend_series.empty:
            st.plotly_chart(plot_annotated_line_chart(
                data_series=malaria_positivity_trend_series, 
                chart_title=f"Weekly {malaria_rdt_name_from_config} Positivity Rate",
                y_axis_label="Positivity %", 
                target_ref_line_val=settings.TARGET_MALARIA_POSITIVITY_RATE,
                y_values_are_counts=False # It's a rate/percentage
            ), use_container_width=True)
        
        if epi_data_for_tab_display.get("calculation_notes"):
            for note_epi_tab in epi_data_for_tab_display["calculation_notes"]:
                st.caption(f"Note (Epi Tab): {note_epi_tab}")
    else:
        st.info("No health data available in the selected period for epidemiological analysis in this tab.")

with tab_testing:
    st.subheader(f"Testing & Diagnostics Performance ({current_reporting_period_display_str})")
    # Option for user to select a specific test group for detailed trends, defaulting to "All Critical"
    # For now, hardcoding focus for simplicity, can be made a selectbox.
    focus_test_group_for_details = "All Critical Tests Summary" 
    
    testing_insights_data_map = prepare_clinic_lab_testing_insights_data(
        filtered_health_df_for_clinic_period=health_df_for_period_clinic_tabs,
        clinic_overall_kpis_summary=clinic_summary_kpis_for_period_data, # Pass the pre-aggregated summary
        reporting_period_context_str=current_reporting_period_display_str,
        focus_test_group_display_name=focus_test_group_for_details
    )
    
    df_critical_tests_summary_display = testing_insights_data_map.get("all_critical_tests_summary_table_df")
    if isinstance(df_critical_tests_summary_display, pd.DataFrame) and not df_critical_tests_summary_display.empty:
        st.markdown("###### **Critical Tests Performance Summary:**")
        st.dataframe(df_critical_tests_summary_display, use_container_width=True, hide_index=True)
    
    df_overdue_tests_display = testing_insights_data_map.get("overdue_pending_tests_list_df")
    if isinstance(df_overdue_tests_display, pd.DataFrame) and not df_overdue_tests_display.empty:
        st.markdown("###### **Overdue Pending Tests (Top 15 by Days Pending):**")
        st.dataframe(df_overdue_tests_display.head(15), use_container_width=True, hide_index=True)
    elif isinstance(df_overdue_tests_display, pd.DataFrame): # Empty DF means no overdue tests
        st.success("✅ No tests currently flagged as overdue based on defined criteria.")

    if testing_insights_data_map.get("processing_notes"):
        for note_test_tab in testing_insights_data_map["processing_notes"]:
            st.caption(f"Note (Testing Tab): {note_test_tab}")

with tab_supply:
    st.subheader(f"Medical Supply Forecast & Status ({current_reporting_period_display_str})")
    use_ai_supply_forecast_toggle = st.checkbox(
        "Use Advanced AI Supply Forecast (Simulated)", value=False, # Default to simple linear
        key="clinic_dashboard_supply_ai_toggle"
    )
    
    supply_forecast_data_map = prepare_clinic_supply_forecast_overview_data(
        clinic_historical_health_df_for_supply=full_historical_health_df_clinic, # Use full history for rates
        reporting_period_context_str=current_reporting_period_display_str,
        use_ai_supply_forecasting_model=use_ai_supply_forecast_toggle
        # items_list_to_forecast can be added as a multiselect here
    )
    st.markdown(f"**Forecast Model Used:** `{supply_forecast_data_map.get('forecast_model_type_used', 'N/A')}`")
    
    list_supply_overview_for_display = supply_forecast_data_map.get("forecast_items_overview_list", [])
    if list_supply_overview_for_display:
        df_supply_overview_display = pd.DataFrame(list_supply_overview_for_display)
        st.dataframe(
            df_supply_overview_display, use_container_width=True, hide_index=True,
            column_config={ # Example: format date column if it's datetime object, else it's string
                "estimated_stockout_date": st.column_config.TextColumn("Est. Stockout Date") 
            }
        )
    else:
        st.info("No supply forecast data generated for the selected items or model type.")
    
    if supply_forecast_data_map.get("data_processing_notes"):
        for note_supply_tab in supply_forecast_data_map["data_processing_notes"]:
            st.caption(f"Note (Supply Tab): {note_supply_tab}")

with tab_patient:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_reporting_period_display_str})")
    if not health_df_for_period_clinic_tabs.empty:
        patient_focus_data_map = prepare_clinic_patient_focus_overview_data(
            filtered_health_df_for_clinic_period=health_df_for_period_clinic_tabs,
            reporting_period_context_str=current_reporting_period_display_str
        )
        
        df_patient_load_plot_data = patient_focus_data_map.get("patient_load_by_key_condition_df")
        if isinstance(df_patient_load_plot_data, pd.DataFrame) and not df_patient_load_plot_data.empty:
            st.markdown("###### **Patient Load by Key Condition (Aggregated Weekly):**")
            st.plotly_chart(plot_bar_chart(
                df_input=df_patient_load_plot_data, x_col_name='period_start_date', y_col_name='unique_patients_count',
                chart_title="Patient Load by Key Condition", color_col_name='condition', bar_mode_style='stack',
                y_values_are_counts_flag=True,
                x_axis_label_text="Week Starting", y_axis_label_text="Unique Patients Seen"
            ), use_container_width=True)
        
        df_flagged_patients_for_review = patient_focus_data_map.get("flagged_patients_for_review_df")
        if isinstance(df_flagged_patients_for_review, pd.DataFrame) and not df_flagged_patients_for_review.empty:
            st.markdown("###### **Flagged Patients for Clinical Review (Top Priority):**")
            st.dataframe(df_flagged_patients_for_review.head(15), use_container_width=True, hide_index=True)
        elif isinstance(df_flagged_patients_for_review, pd.DataFrame): # Empty DF means no flagged
            st.info("No patients currently flagged for clinical review in this period based on criteria.")
        
        if patient_focus_data_map.get("processing_notes"):
            for note_patient_tab in patient_focus_data_map["processing_notes"]:
                st.caption(f"Note (Patient Focus Tab): {note_patient_tab}")
    else:
        st.info("No health data available in the selected period for patient focus analysis in this tab.")

with tab_env:
    st.subheader(f"Facility Environment Detailed Monitoring ({current_reporting_period_display_str})")
    env_details_data_map = prepare_clinic_environmental_detail_data(
        filtered_iot_df_for_period=iot_df_for_period_clinic_tabs, # Use period-filtered IoT
        iot_data_source_is_generally_available=iot_data_source_is_available,
        reporting_period_context_str=current_reporting_period_display_str
    )
    
    list_current_env_alerts_for_display = env_details_data_map.get("current_environmental_alerts_list", [])
    if list_current_env_alerts_for_display:
        st.markdown("###### **Current Environmental Alerts (from Latest Readings in Period):**")
        non_acceptable_alert_found_env = False
        for alert_item_env in list_current_env_alerts_for_display:
            if alert_item_env.get("level") != "ACCEPTABLE":
                non_acceptable_alert_found_env = True
                render_traffic_light_indicator( # Use new renderer
                    message=alert_item_env.get('message', 'Environmental issue detected.'),
                    status_level=alert_item_env.get('level', 'UNKNOWN'), # Pass Pythonic status
                    details_text=alert_item_env.get('alert_type', 'Environmental Alert')
                )
        if not non_acceptable_alert_found_env and len(list_current_env_alerts_for_display) == 1 and \
           list_current_env_alerts_for_display[0].get("level") == "ACCEPTABLE":
            st.success(f"✅ {list_current_env_alerts_for_display[0].get('message', 'Environment appears normal based on latest checks.')}")
        elif not non_acceptable_alert_found_env and len(list_current_env_alerts_for_display) > 1:
            st.info("Multiple environmental parameters checked; all appear within acceptable limits based on latest readings.")

    co2_trend_series_clinic = env_details_data_map.get("hourly_avg_co2_trend")
    if isinstance(co2_trend_series_clinic, pd.Series) and not co2_trend_series_clinic.empty:
        st.plotly_chart(plot_annotated_line_chart(
            data_series=co2_trend_series_clinic, chart_title="Hourly Avg. CO2 Levels (Clinic-wide)",
            y_axis_label="CO2 (ppm)", date_format_hover="%H:%M (%d-%b)", y_values_are_counts=False,
            target_ref_line_val=settings.ALERT_AMBIENT_CO2_HIGH_PPM # Show high CO2 threshold
        ), use_container_width=True)
    
    df_latest_room_readings_display = env_details_data_map.get("latest_room_sensor_readings_df")
    if isinstance(df_latest_room_readings_display, pd.DataFrame) and not df_latest_room_readings_display.empty:
        st.markdown("###### **Latest Sensor Readings by Room (End of Period):**")
        st.dataframe(df_latest_room_readings_display, use_container_width=True, hide_index=True)
    
    if env_details_data_map.get("processing_notes"):
        for note_env_tab in env_details_data_map["processing_notes"]:
            st.caption(f"Note (Env. Detail Tab): {note_env_tab}")
    
    if not iot_data_source_is_available and \
       (not isinstance(iot_df_for_period_clinic_tabs, pd.DataFrame) or iot_df_for_period_clinic_tabs.empty):
        st.warning("IoT environmental data source appears generally unavailable. Detailed environmental monitoring is not possible.")

logger.info(f"Clinic Operations & Management Console page loaded for period: {current_reporting_period_display_str}")
