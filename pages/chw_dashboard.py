# sentinel_project_root/pages/chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
import numpy as np # For np.nan
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple
import os # For checking logo path in sidebar

# --- Sentinel System Imports from Refactored Structure ---
try:
    from config import settings
    from data_processing.loaders import load_health_records
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart
    
    # CHW specific components from the new structure
    from .chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from .chw_components.alert_generation import generate_chw_alerts # Note: was alert_generator.py
    from .chw_components.epi_signals import extract_chw_epi_signals # Note: was epi_signal_extractor.py
    from .chw_components.task_processing import generate_chw_tasks # Note: was task_processor.py
    from .chw_components.activity_trends import calculate_chw_activity_trends_data # Note: was activity_trend_calculator.py
except ImportError as e_chw_dash:
    import sys
    # Provide a more informative error message in Streamlit if imports fail
    st.error(
        f"CHW Dashboard Import Error: {e_chw_dash}. "
        f"Please ensure all modules are correctly placed and dependencies installed. "
        f"Relevant Python Path: {sys.path}"
    )
    logger = logging.getLogger(__name__) # Attempt to get a logger for console output
    logger.error(f"CHW Dashboard Import Error: {e_chw_dash}", exc_info=True)
    st.stop() # Halt execution of this page if essential imports fail

# --- Page Specific Logger ---
logger = logging.getLogger(__name__) # Get logger for this specific page

# --- Page Title and Introduction ---
# Page config is set in app.py
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring & Field Support - {settings.APP_NAME}**")
st.divider()

# --- Utility Function for Filter Options ---
# (Could be moved to a shared utils if used by multiple dashboard pages)
def _create_filter_dropdown_options(
    df_for_options: pd.DataFrame,
    column_name_in_df: str,
    default_fallback_options: List[str],
    display_name_plural_for_all: str
) -> List[str]:
    """Creates a list of filter options for a selectbox, including an 'All ...' option."""
    all_option_label = f"All {display_name_plural_for_all}"
    options_list = [all_option_label] # 'All' is always the first default option
    
    if isinstance(df_for_options, pd.DataFrame) and not df_for_options.empty and column_name_in_df in df_for_options.columns:
        unique_values_from_df = sorted(df_for_options[column_name_in_df].dropna().unique().tolist())
        if unique_values_from_df:
            options_list.extend(unique_values_from_df)
        else:
            logger.warning(f"Filter options: Column '{column_name_in_df}' for '{display_name_plural_for_all}' has no unique, non-null values. Using defaults.")
            options_list.extend(default_fallback_options) # Use fallback if column empty after dropna
    else:
        logger.warning(f"Filter options: Column '{column_name_in_df}' not found or DataFrame empty for '{display_name_plural_for_all}'. Using default options.")
        options_list.extend(default_fallback_options) # Use fallback if column not found or df empty
    return options_list

# --- Data Loading Function for this Dashboard ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data...")
def get_chw_dashboard_display_data(
    selected_view_date: date,
    selected_trend_start_date: date,
    selected_trend_end_date: date,
    selected_chw_id_filter: Optional[str] = None,
    selected_zone_id_filter: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads and filters health records for the CHW dashboard for a specific date and trend period.
    Returns daily data, period data, and any pre-calculated daily KPIs.
    """
    log_ctx = "CHWDashboardDataLoad"
    logger.info(
        f"({log_ctx}) Loading data for view: {selected_view_date}, trend: {selected_trend_start_date}-{selected_trend_end_date}, "
        f"CHW: {selected_chw_id_filter or 'All'}, Zone: {selected_zone_id_filter or 'All'}"
    )

    try:
        # Load all health records once; filtering will be applied subsequently.
        # For very large datasets, consider server-side filtering if possible.
        all_health_records_df = load_health_records(source_context=f"{log_ctx}/LoadAllRecs")
    except Exception as e_load_all:
        logger.error(f"({log_ctx}) CRITICAL FAILURE: Could not load health records: {e_load_all}", exc_info=True)
        st.error("Fatal Error: Could not load primary health data for the dashboard. Please contact support.")
        return pd.DataFrame(), pd.DataFrame(), {} # Return empty structures to prevent further errors

    if all_health_records_df.empty:
        logger.warning(f"({log_ctx}) No health records were loaded. Dashboard will display no data.")
        st.warning("No health data is currently available. Please check data sources or try again later.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Critical check: 'encounter_date' must exist and be datetime
    if 'encounter_date' not in all_health_records_df.columns or \
       not pd.api.types.is_datetime64_any_dtype(all_health_records_df['encounter_date']):
        logger.error(f"({log_ctx}) 'encounter_date' column is missing or not in datetime format. Cannot filter data effectively.")
        st.error("Data Integrity Error: 'encounter_date' in health records is invalid. Dashboard cannot proceed.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # --- Filter for Daily Snapshot ---
    df_for_daily_snapshot = all_health_records_df[
        all_health_records_df['encounter_date'].dt.date == selected_view_date
    ].copy() # Use .copy() to avoid SettingWithCopyWarning on subsequent filters
    
    if selected_chw_id_filter and 'chw_id' in df_for_daily_snapshot.columns:
        df_for_daily_snapshot = df_for_daily_snapshot[df_for_daily_snapshot['chw_id'] == selected_chw_id_filter]
    if selected_zone_id_filter and 'zone_id' in df_for_daily_snapshot.columns:
        df_for_daily_snapshot = df_for_daily_snapshot[df_for_daily_snapshot['zone_id'] == selected_zone_id_filter]

    # --- Filter for Trend Period ---
    df_for_trend_period = all_health_records_df[
        (all_health_records_df['encounter_date'].dt.date >= selected_trend_start_date) &
        (all_health_records_df['encounter_date'].dt.date <= selected_trend_end_date)
    ].copy()
    
    if selected_chw_id_filter and 'chw_id' in df_for_trend_period.columns:
        df_for_trend_period = df_for_trend_period[df_for_trend_period['chw_id'] == selected_chw_id_filter]
    if selected_zone_id_filter and 'zone_id' in df_for_trend_period.columns:
        df_for_trend_period = df_for_trend_period[df_for_trend_period['zone_id'] == selected_zone_id_filter]

    # Pre-calculate specific KPIs if needed (e.g., worker fatigue from self-check for the day)
    # This can optimize calls to summary metric calculators if they accept pre-calculated values.
    pre_calculated_daily_kpis_map: Dict[str, Any] = {}
    if selected_chw_id_filter and not df_for_daily_snapshot.empty:
        # Example: Max fatigue score from WORKER_SELF_CHECK encounters for the selected CHW on that day
        worker_self_check_df = df_for_daily_snapshot[
            (df_for_daily_snapshot.get('chw_id') == selected_chw_id_filter) & # Ensure chw_id column exists
            (df_for_daily_snapshot.get('encounter_type', pd.Series(dtype=str)) == 'WORKER_SELF_CHECK') # Ensure encounter_type column exists
        ]
        if not worker_self_check_df.empty:
            # Try to find a fatigue-related score column
            fatigue_score_col_options = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            actual_fatigue_col_found = next((col for col in fatigue_score_col_options if col in worker_self_check_df.columns and worker_self_check_df[col].notna().any()), None)
            
            if actual_fatigue_col_found:
                pre_calculated_daily_kpis_map['worker_self_fatigue_index_today'] = worker_self_check_df[actual_fatigue_col_found].max()
            else: # No relevant score column found in self-check data
                pre_calculated_daily_kpis_map['worker_self_fatigue_index_today'] = np.nan # Explicitly NaN
        else: # No self-check records found for this CHW today
            pre_calculated_daily_kpis_map['worker_self_fatigue_index_today'] = np.nan
    
    logger.info(
        f"({log_ctx}) Data loading and filtering complete. Daily Snapshot: {len(df_for_daily_snapshot)} records, "
        f"Trend Period: {len(df_for_trend_period)} records."
    )
    return df_for_daily_snapshot, df_for_trend_period, pre_calculated_daily_kpis_map


# --- Sidebar Filters Setup ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS) # Cache data for filter population
def load_dropdown_filter_data():
    logger.debug("CHW Dashboard: Loading minimal data for filter dropdowns.")
    # Optimize by loading only 'chw_id' and 'zone_id' if data source allows partial loads.
    # For current CSV loader, it loads all then we select.
    return load_health_records(source_context="CHWDash/FilterDataPopulation")

df_for_filter_options = load_dropdown_filter_data()

# Sidebar Header with Logo
if os.path.exists(settings.APP_LOGO_SMALL_PATH):
    st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120) # Slightly smaller logo
else:
    st.sidebar.markdown(f"#### {settings.APP_NAME}", help="App Logo Placeholder")

st.sidebar.header("Dashboard Filters")

# CHW ID Filter
chw_filter_options = _create_filter_dropdown_options(df_for_filter_options, 'chw_id', ["CHW001", "CHW002", "CHW003"], "CHWs")
# Session state key for CHW ID filter
chw_id_session_key = "chw_dashboard_chw_id_selection"
if chw_id_session_key not in st.session_state:
    st.session_state[chw_id_session_key] = chw_filter_options[0] # Default to "All CHWs"

selected_chw_id_ui = st.sidebar.selectbox(
    "Filter by CHW ID:", options=chw_filter_options,
    key=f"{chw_id_session_key}_widget", # Widget key
    index=chw_filter_options.index(st.session_state[chw_id_session_key]) # Set from session state
)
st.session_state[chw_id_session_key] = selected_chw_id_ui # Update session state
actual_chw_id_query_param = None if selected_chw_id_ui.startswith("All ") else selected_chw_id_ui

# Zone Filter
zone_filter_options = _create_filter_dropdown_options(df_for_filter_options, 'zone_id', ["ZoneA", "ZoneB", "ZoneC"], "Zones")
zone_id_session_key = "chw_dashboard_zone_id_selection"
if zone_id_session_key not in st.session_state:
    st.session_state[zone_id_session_key] = zone_filter_options[0]

selected_zone_id_ui = st.sidebar.selectbox(
    "Filter by Zone:", options=zone_filter_options,
    key=f"{zone_id_session_key}_widget",
    index=zone_filter_options.index(st.session_state[zone_id_session_key])
)
st.session_state[zone_id_session_key] = selected_zone_id_ui
actual_zone_id_query_param = None if selected_zone_id_ui.startswith("All ") else selected_zone_id_ui


# Date Pickers for Dashboard Scope
# Define overall min/max allowable dates for date pickers
abs_min_date_for_pickers = date.today() - timedelta(days=max(180, settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 5)) # Wider historical range
abs_max_date_for_pickers = date.today() # Cannot select future dates

# Daily Snapshot Date Picker
# Session state for daily date picker
daily_date_session_key = "chw_dashboard_daily_date_selection"
if daily_date_session_key not in st.session_state:
    st.session_state[daily_date_session_key] = abs_max_date_for_pickers # Default to today

selected_daily_date_ui = st.sidebar.date_input(
    "View Daily Activity For:",
    value=st.session_state[daily_date_session_key],
    min_value=abs_min_date_for_pickers,
    max_value=abs_max_date_for_pickers,
    key=f"{daily_date_session_key}_widget"
)
st.session_state[daily_date_session_key] = selected_daily_date_ui # Update session state


# Trend Date Range Picker
# Session state for trend date range picker (stores a list/tuple of two dates)
trend_date_range_session_key = "chw_dashboard_trend_date_range_selection"
# Default trend range calculation
default_trend_end_ui = selected_daily_date_ui # Align with daily snapshot initially
default_trend_start_ui = default_trend_end_ui - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_trend_start_ui < abs_min_date_for_pickers: # Ensure start doesn't go before absolute min
    default_trend_start_ui = abs_min_date_for_pickers

if trend_date_range_session_key not in st.session_state:
    st.session_state[trend_date_range_session_key] = [default_trend_start_ui, default_trend_end_ui]

selected_trend_date_range_ui = st.sidebar.date_input(
    "Select Trend Date Range:",
    value=st.session_state[trend_date_range_session_key], # Must be a list/tuple of two dates
    min_value=abs_min_date_for_pickers,
    max_value=abs_max_date_for_pickers,
    key=f"{trend_date_range_session_key}_widget"
)
# Ensure it's a list/tuple of two dates after selection
if isinstance(selected_trend_date_range_ui, (list, tuple)) and len(selected_trend_date_range_ui) == 2:
    st.session_state[trend_date_range_session_key] = selected_trend_date_range_ui
    trend_start_date_query_param, trend_end_date_query_param = selected_trend_date_range_ui
else: # Fallback if date_input for range doesn't return two dates (should not happen)
    st.sidebar.warning("Trend date range selection error. Reverting to default range.")
    st.session_state[trend_date_range_session_key] = [default_trend_start_ui, default_trend_end_ui]
    trend_start_date_query_param, trend_end_date_query_param = default_trend_start_ui, default_trend_end_ui

# Validate that trend start is not after trend end
if trend_start_date_query_param > trend_end_date_query_param:
    st.sidebar.error("Trend Start Date must be on or before Trend End Date. Adjusting to single day.")
    trend_end_date_query_param = trend_start_date_query_param # Make it a single day range


# --- Load Data Based on Selected Filters ---
# This is the main data loading call for the dashboard content.
try:
    daily_df_for_display, period_df_for_display, daily_pre_calculated_kpis_map = get_chw_dashboard_display_data(
        selected_view_date=selected_daily_date_ui,
        selected_trend_start_date=trend_start_date_query_param,
        selected_trend_end_date=trend_end_date_query_param,
        selected_chw_id_filter=actual_chw_id_query_param,
        selected_zone_id_filter=actual_zone_id_query_param
    )
except Exception as e_main_data_load:
    logger.error(f"CHW Dashboard: Main data loading/processing failed: {e_main_data_load}", exc_info=True)
    st.error(f"An error occurred while loading CHW dashboard data: {str(e_main_data_load)}. Please check logs or contact support.")
    # Stop execution for this page if main data fails to load, providing empty placeholders.
    daily_df_for_display, period_df_for_display, daily_pre_calculated_kpis_map = pd.DataFrame(), pd.DataFrame(), {}
    # st.stop() # Could stop here, or let sections below handle empty DFs.

# --- Display Filter Context to User ---
filter_context_display_parts = [f"Snapshot Date: **{selected_daily_date_ui.strftime('%d %b %Y')}**"]
if actual_chw_id_query_param:
    filter_context_display_parts.append(f"CHW: **{actual_chw_id_query_param}**")
if actual_zone_id_query_param:
    filter_context_display_parts.append(f"Zone: **{actual_zone_id_query_param}**")
st.info(f"Displaying data for: {', '.join(filter_context_display_parts)}")


# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not daily_df_for_display.empty:
    try:
        chw_daily_summary_metrics_map = calculate_chw_daily_summary_metrics(
            chw_daily_encounter_df=daily_df_for_display,
            for_date=selected_daily_date_ui,
            chw_daily_kpi_input_data=daily_pre_calculated_kpis_map, # Pass pre-calcs
            source_context="CHWDash/DailySummary"
        )
    except Exception as e_daily_summary:
        logger.error(f"Error calculating CHW daily summary metrics for dashboard: {e_daily_summary}", exc_info=True)
        st.warning("Could not calculate daily summary metrics. Some KPIs may be missing or inaccurate.")
        chw_daily_summary_metrics_map = {} # Fallback to empty dict on error

    # Display KPIs in columns
    kpi_cols_daily_snapshot = st.columns(4)
    with kpi_cols_daily_snapshot[0]:
        render_kpi_card(
            title="Visits Today", value_str=str(chw_daily_summary_metrics_map.get("visits_count", 0)),
            icon="👥", help_text="Total unique patients encountered by the CHW/team for the selected date."
        )
    
    high_prio_followups_val = chw_daily_summary_metrics_map.get("high_ai_prio_followups_count", 0)
    prio_status_level = "ACCEPTABLE" if high_prio_followups_val <= 2 else \
                        ("MODERATE_CONCERN" if high_prio_followups_val <= 5 else "HIGH_CONCERN")
    with kpi_cols_daily_snapshot[1]:
        render_kpi_card(
            title="High Prio Follow-ups", value_str=str(high_prio_followups_val),
            icon="🎯", status_level=prio_status_level,
            help_text=f"Patients needing urgent follow-up based on AI priority scores (e.g., score ≥ {settings.FATIGUE_INDEX_HIGH_THRESHOLD})." # Example threshold in help
        )
    
    critical_spo2_cases_val = chw_daily_summary_metrics_map.get("critical_spo2_cases_identified_count", 0)
    spo2_status_level = "HIGH_CONCERN" if critical_spo2_cases_val > 0 else "ACCEPTABLE"
    with kpi_cols_daily_snapshot[2]:
        render_kpi_card(
            title="Critical SpO2 Cases", value_str=str(critical_spo2_cases_val),
            icon="💨", status_level=spo2_status_level,
            help_text=f"Patients identified with SpO2 < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}% on the selected date."
        )

    high_fever_cases_val = chw_daily_summary_metrics_map.get("high_fever_cases_identified_count", 0)
    fever_status_level = "HIGH_CONCERN" if high_fever_cases_val > 0 else "ACCEPTABLE"
    with kpi_cols_daily_snapshot[3]:
        render_kpi_card(
            title="High Fever Cases", value_str=str(high_fever_cases_val),
            icon="🔥", status_level=fever_status_level,
            help_text=f"Patients identified with temperature ≥ {settings.ALERT_BODY_TEMP_HIGH_FEVER_C}°C on the selected date."
        )
else: # daily_df_for_display is empty
    st.markdown("_No activity data available for the selected date and/or filters to display daily performance snapshot._")
st.divider()


# --- Section 2: Key Alerts & Actionable Tasks ---
st.header("🚦 Key Alerts & Tasks")
# Key Alerts Display
try:
    chw_alerts_list_for_display = generate_chw_alerts(
        patient_encounter_data_df=daily_df_for_display, # Use daily data for today's alerts
        for_date=selected_daily_date_ui,
        chw_zone_context_str=actual_zone_id_query_param or "All Zones",
        max_alerts_to_return=10 # Limit for UI display
    )
except ValueError as ve_alerts_display: # Catch specific error if component raises for no data
    logger.warning(f"CHW Dashboard: Value error generating alerts for display: {ve_alerts_display}")
    chw_alerts_list_for_display = []
    st.caption(f"Alert generation note: {str(ve_alerts_display)}")
except Exception as e_alerts_display:
    logger.error(f"CHW Dashboard: Error generating patient alerts for display: {e_alerts_display}", exc_info=True)
    st.warning("Could not generate patient alerts. Alert list may be incomplete.")
    chw_alerts_list_for_display = []

if chw_alerts_list_for_display:
    st.subheader("Priority Patient Alerts (Today):")
    critical_alerts_exist = False
    for alert_item_disp in chw_alerts_list_for_display:
        if alert_item_disp.get("alert_level") == "CRITICAL":
            critical_alerts_exist = True
            render_traffic_light_indicator(
                message=f"Pt. {alert_item_disp.get('patient_id', 'N/A')}: {alert_item_disp.get('primary_reason', 'Critical Alert')}",
                status_level="HIGH_RISK", # This should map to CSS classes for red
                details_text=(
                    f"Details: {alert_item_disp.get('brief_details','N/A')} | "
                    f"Context: {alert_item_disp.get('context_info','N/A')} | "
                    f"Action Suggestion: {alert_item_disp.get('suggested_action_code','REVIEW_IMMEDIATELY')}"
                )
            )
    if not critical_alerts_exist:
        st.info("No CRITICAL patient alerts identified for this selection today.")

    warning_alerts_for_display = [a for a in chw_alerts_list_for_display if a.get("alert_level") == "WARNING"]
    if warning_alerts_for_display:
        st.markdown("###### Warning Level Alerts:")
        for warn_item_disp in warning_alerts_for_display:
            render_traffic_light_indicator(
                message=f"Pt. {warn_item_disp.get('patient_id', 'N/A')}: {warn_item_disp.get('primary_reason', 'Warning')}",
                status_level="MODERATE_CONCERN", # Maps to amber/yellow
                details_text=f"Details: {warn_item_disp.get('brief_details','N/A')} | Context: {warn_item_disp.get('context_info','N/A')}"
            )
    elif not critical_alerts_exist : # No critical AND no warnings
        st.info("Only informational alerts (if any) were generated. No urgent patient alerts requiring immediate attention.")
elif not daily_df_for_display.empty: # Data exists, but alert list is empty
    st.success("✅ No significant patient alerts needing immediate attention were generated for today's selection.")
else:
    st.markdown("_No activity data to generate patient alerts for today._")

# Actionable Tasks Display (integrating the refactored component)
try:
    chw_tasks_list_for_display = generate_chw_tasks(
        source_patient_data_df=daily_df_for_display, # Based on today's findings
        for_date=selected_daily_date_ui,
        chw_id_context=actual_chw_id_query_param,
        zone_context_str=actual_zone_id_query_param or "All Zones",
        max_tasks_to_return_for_summary=10
    )
except ValueError as ve_tasks_display:
    logger.warning(f"CHW Dashboard: Value error generating tasks for display: {ve_tasks_display}")
    chw_tasks_list_for_display = []
    st.caption(f"Task generation note: {str(ve_tasks_display)}")
except Exception as e_tasks_display:
    logger.error(f"CHW Dashboard: Error generating CHW tasks for display: {e_tasks_display}", exc_info=True)
    st.warning("Could not generate the prioritized tasks list for display.")
    chw_tasks_list_for_display = []

if chw_tasks_list_for_display:
    st.subheader("Top Priority Tasks (Today/Next Day):")
    tasks_df_for_table = pd.DataFrame(chw_tasks_list_for_display)
    # Select and order columns for the task table display
    task_table_cols_order = ['patient_id', 'task_description', 'priority_score', 'due_date', 
                             'status', 'key_patient_context', 'assigned_chw_id'] 
    # Filter to only those columns that actually exist in the DataFrame
    actual_cols_for_task_table = [col for col in task_table_cols_order if col in tasks_df_for_table.columns]
    
    if not tasks_df_for_table.empty and actual_cols_for_task_table:
        st.dataframe(
            tasks_df_for_table[actual_cols_for_task_table],
            use_container_width=True,
            height=min(420, len(tasks_df_for_table) * 38 + 58), # Dynamic height with a max
            hide_index=True,
            column_config={ # Example of specific column configurations for better display
                "priority_score": st.column_config.NumberColumn(format="%.1f"),
                "due_date": st.column_config.DateColumn(format="YYYY-MM-DD")
            }
        )
    elif not tasks_df_for_table.empty: # Columns mismatch
        st.warning("Task data available but cannot be displayed due to column configuration issues.")
        logger.debug(f"Task table display issue: Expected cols {task_table_cols_order}, available {tasks_df_for_table.columns.tolist()}")
elif not daily_df_for_display.empty: # Data exists, but task list is empty
     st.info("No high-priority tasks identified requiring action today or tomorrow based on current data.")
else:
     st.markdown("_No activity data to generate tasks for today._")
st.divider()


# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if not daily_df_for_display.empty:
    try:
        chw_epi_signals_map = extract_chw_epi_signals(
            chw_daily_encounter_df=daily_df_for_display, # Today's data
            pre_calculated_chw_kpis=daily_pre_calculated_kpis_map,
            for_date=selected_daily_date_ui,
            chw_zone_context=actual_zone_id_query_param or "All Zones",
            max_symptom_clusters_to_report=3
        )
    except ValueError as ve_epi_display:
        logger.warning(f"CHW Dashboard: Value error extracting epi signals for display: {ve_epi_display}")
        chw_epi_signals_map = {}
        st.caption(f"Epi signal extraction note: {str(ve_epi_display)}")
    except Exception as e_epi_display:
        logger.error(f"CHW Dashboard: Error extracting epi signals for display: {e_epi_display}", exc_info=True)
        st.warning("Could not extract local epidemiological signals for display.")
        chw_epi_signals_map = {}

    # Display Epi KPIs
    epi_kpi_cols_display = st.columns(3)
    with epi_kpi_cols_display[0]:
        sympt_key_cond_val = chw_epi_signals_map.get("symptomatic_patients_key_conditions_count", 0)
        render_kpi_card(
            title="Symptomatic (Key Cond.)", value_str=str(sympt_key_cond_val),
            icon="🤒", units="cases today",
            help_text=f"Patients seen today presenting with symptoms related to key conditions (e.g., {', '.join(settings.KEY_CONDITIONS_FOR_ACTION[:2])}, etc.)."
        )
    with epi_kpi_cols_display[1]:
        new_malaria_val = chw_epi_signals_map.get("newly_identified_malaria_patients_count", 0)
        malaria_status_level = "HIGH_CONCERN" if new_malaria_val > 1 else ("MODERATE_CONCERN" if new_malaria_val == 1 else "ACCEPTABLE")
        render_kpi_card(
            title="New Malaria Cases", value_str=str(new_malaria_val),
            icon="🦟", units="cases today", status_level=malaria_status_level,
            help_text="New malaria cases (e.g., based on RDT positive or clinical diagnosis) identified today."
        )
    with epi_kpi_cols_display[2]:
        pending_tb_contacts_val = chw_epi_signals_map.get("pending_tb_contact_tracing_tasks_count", 0)
        tb_contact_status_level = "MODERATE_CONCERN" if pending_tb_contacts_val > 0 else "ACCEPTABLE"
        render_kpi_card(
            title="Pending TB Contacts", value_str=str(pending_tb_contacts_val),
            icon="👥", units="to trace", status_level=tb_contact_status_level,
            help_text="Number of TB contacts (related to today's CHW activity or existing tasks) still requiring follow-up."
        )

    # Display Detected Symptom Clusters
    detected_symptom_clusters_list = chw_epi_signals_map.get("detected_symptom_clusters", [])
    if detected_symptom_clusters_list:
        st.markdown("###### Detected Symptom Clusters (Requires Verification by Supervisor):")
        for cluster_item_data in detected_symptom_clusters_list:
            st.warning( # Use st.warning for visibility
                f"⚠️ **Pattern: {cluster_item_data.get('symptoms_pattern', 'Unknown Symptom Pattern')}** - "
                f"{cluster_item_data.get('patient_count', 'N/A')} cases identified in "
                f"{cluster_item_data.get('location_hint', 'CHW operational area')}. Supervisor to verify and potentially escalate."
            )
    elif 'patient_reported_symptoms' in daily_df_for_display.columns and daily_df_for_display['patient_reported_symptoms'].notna().any():
        st.info("No significant symptom clusters detected today based on current data and criteria.")
else: # daily_df_for_display is empty
    st.markdown("_No activity data available for the selected date/filters to extract local epi signals._")
st.divider()


# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_text = f"{trend_start_date_query_param.strftime('%d %b %Y')} - {trend_end_date_query_param.strftime('%d %b %Y')}"
trend_filter_context_text = ""
if actual_chw_id_query_param: trend_filter_context_text += f" for CHW **{actual_chw_id_query_param}**"
if actual_zone_id_query_param: trend_filter_context_text += f" in Zone **{actual_zone_id_query_param}**"
st.markdown(f"Displaying trends from **{trend_period_display_text}**{trend_filter_context_text if trend_filter_context_text else ' (All CHWs/Zones in selected period)'}.")

if not period_df_for_display.empty:
    try:
        chw_activity_trends_map = calculate_chw_activity_trends_data(
            chw_historical_health_df=period_df_for_display, # Pass the period-filtered data
            trend_start_date_input=trend_start_date_query_param, # For context or internal use by component
            trend_end_date_input=trend_end_date_query_param,
            zone_filter=actual_zone_id_query_param, # Allow component to re-filter if necessary (though period_df is already filtered)
            time_period_aggregation='D' # Daily trends are common for CHW supervisor view
        )
    except Exception as e_trends_display:
        logger.error(f"CHW Dashboard: Error calculating activity trends for display: {e_trends_display}", exc_info=True)
        st.warning("Could not calculate CHW activity trends. Trend charts may be unavailable.")
        chw_activity_trends_map = {} # Fallback to empty dict

    trend_plot_cols_display = st.columns(2) # Two columns for trend plots
    with trend_plot_cols_display[0]:
        patient_visits_trend_series = chw_activity_trends_map.get("patient_visits_trend")
        if isinstance(patient_visits_trend_series, pd.Series) and not patient_visits_trend_series.empty:
            st.plotly_chart(
                plot_annotated_line_chart( # Use new plotting function
                    data_series=patient_visits_trend_series, chart_title="Daily Patient Visits Trend",
                    y_axis_label="# Unique Patients Visited", y_values_are_counts=True
                ), use_container_width=True
            )
        else:
            st.caption("No patient visit trend data available for this selection.")
            
    with trend_plot_cols_display[1]:
        high_prio_followups_trend_srs = chw_activity_trends_map.get("high_priority_followups_trend")
        if isinstance(high_prio_followups_trend_srs, pd.Series) and not high_prio_followups_trend_srs.empty:
            st.plotly_chart(
                plot_annotated_line_chart(
                    data_series=high_prio_followups_trend_srs, chart_title="Daily High Prio. Follow-ups Trend",
                    y_axis_label="# High Prio. Follow-ups", y_values_are_counts=True
                ), use_container_width=True
            )
        else:
            st.caption("No high-priority follow-up trend data available for this selection.")
else: # period_df_for_display is empty
    st.markdown("_No historical data available for the selected trend period and/or filters._")

logger.info(
    f"CHW Supervisor Dashboard page loaded. Filters: Date={selected_daily_date_ui}, "
    f"CHW={actual_chw_id_query_param or 'All'}, Zone={actual_zone_id_query_param or 'All'}, "
    f"Trend Range=({trend_start_date_query_param} to {trend_end_date_query_param})."
)
