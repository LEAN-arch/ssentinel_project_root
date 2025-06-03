# sentinel_project_root/pages/chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
import logging
from datetime import date, timedelta, datetime
from typing import Optional, Dict, Any, List, Tuple
import os # For checking logo path

# --- Sentinel System Imports ---
# Ensure robust pathing for imports if this file is run directly or by Streamlit
# (Streamlit usually handles paths well if app.py is at project root)
try:
    from config import settings # Use new settings module
    from data_processing.loaders import load_health_records
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator # Use new UI elements
    from visualization.plots import plot_annotated_line_chart # Use new plotting functions
    # CHW specific components - ensure paths are correct if they are moved/refactored
    from .chw_components.summary_metrics import calculate_chw_daily_summary_metrics # Adjusted import
    from .chw_components.alert_generation import generate_chw_alerts # Adjusted import, renamed component
    from .chw_components.epi_signals import extract_chw_epi_signals # Adjusted import
    # from .chw_components.task_processing import generate_chw_tasks # Adjusted import, renamed component
    from .chw_components.activity_trends import calculate_chw_activity_trends_data # Adjusted import
except ImportError as e:
    # This structure helps diagnose import errors better
    import sys
    st.error(f"Import Error in CHW Dashboard: {e}. Check module paths and ensure all dependencies are installed. Python sys.path: {sys.path}")
    st.stop() # Stop execution if core modules can't be loaded

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Page Configuration (Handled by Streamlit based on filename & app.py) ---
# st.set_page_config is usually called once in the main app.py.
# Individual pages inherit global settings but can override some via st.title, etc.
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring for {settings.APP_NAME}**")
st.divider()

# --- Utility Function for Filter Options (can be moved to a shared utils if used elsewhere) ---
def _create_filter_options(df: pd.DataFrame, column_name: str, default_options: List[str], display_name_plural: str) -> List[str]:
    """Creates a list of filter options for a selectbox, including an 'All' option."""
    options = [f"All {display_name_plural}"] # "All" is always the first option
    if isinstance(df, pd.DataFrame) and not df.empty and column_name in df.columns:
        unique_values = sorted(df[column_name].dropna().unique().tolist())
        if unique_values:
            options.extend(unique_values)
        else: # Column exists but no unique values (e.g., all NaN or empty after dropna)
            logger.warning(f"Column '{column_name}' for {display_name_plural} filter has no unique, non-null values. Using defaults.")
            options.extend(default_options)
    else:
        logger.warning(f"Column '{column_name}' not found or DataFrame empty for {display_name_plural} filter. Using default options.")
        # st.sidebar.caption(f"No {display_name_plural} data for filter.") # User feedback
        options.extend(default_options)
    return options

# --- Data Loading Functions ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data...")
def get_chw_dashboard_processed_data(
    selected_date_view: date, # Renamed for clarity
    trend_range_start_date: date, # Renamed
    trend_range_end_date: date,   # Renamed
    filter_chw_id: Optional[str] = None, # Renamed
    filter_zone_id: Optional[str] = None  # Renamed
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads and filters health records for the CHW dashboard for a specific date and trend period.
    """
    log_prefix = "CHWDashboardData"
    logger.info(f"({log_prefix}) Loading data for view_date: {selected_date_view}, trend: {trend_range_start_date}-{trend_range_end_date}, CHW: {filter_chw_id}, Zone: {filter_zone_id}")

    try:
        # Load all health records once. Filtering will happen below.
        # Specify only necessary columns to reduce memory, if known and stable.
        # For dynamic dashboards, loading more columns and then selecting is often safer.
        health_records_df_all = load_health_records(source_context=f"{log_prefix}/LoadAllHealthRecs")
    except Exception as e_load:
        logger.error(f"({log_prefix}) CRITICAL: Failed to load health records: {e_load}", exc_info=True)
        st.error("Fatal Error: Could not load primary health data. Dashboard cannot proceed.")
        return pd.DataFrame(), pd.DataFrame(), {} # Return empty structures

    if health_records_df_all.empty:
        logger.warning(f"({log_prefix}) No health records loaded. Dashboard will be empty.")
        st.warning("No health data available to display for the CHW dashboard.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Ensure 'encounter_date' is datetime
    if 'encounter_date' not in health_records_df_all.columns or not pd.api.types.is_datetime64_any_dtype(health_records_df_all['encounter_date']):
        logger.error(f"({log_prefix}) 'encounter_date' column missing or not datetime. Cannot filter data.")
        st.error("Data Error: 'encounter_date' is missing or invalid in health records.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # --- Filter for Daily Snapshot ---
    daily_snapshot_df = health_records_df_all[health_records_df_all['encounter_date'].dt.date == selected_date_view].copy()
    if filter_chw_id:
        daily_snapshot_df = daily_snapshot_df[daily_snapshot_df.get('chw_id', pd.Series(dtype=str)) == filter_chw_id]
    if filter_zone_id:
        daily_snapshot_df = daily_snapshot_df[daily_snapshot_df.get('zone_id', pd.Series(dtype=str)) == filter_zone_id]

    # --- Filter for Trend Period ---
    period_trend_df = health_records_df_all[
        (health_records_df_all['encounter_date'].dt.date >= trend_range_start_date) &
        (health_records_df_all['encounter_date'].dt.date <= trend_range_end_date)
    ].copy()
    if filter_chw_id:
        period_trend_df = period_trend_df[period_trend_df.get('chw_id', pd.Series(dtype=str)) == filter_chw_id]
    if filter_zone_id:
        period_trend_df = period_trend_df[period_trend_df.get('zone_id', pd.Series(dtype=str)) == filter_zone_id]

    # Pre-calculate specific KPIs if needed by components (e.g., worker fatigue if not from encounter_type)
    # This was in original logic, can be useful if direct calculation is more efficient than full DF pass in component.
    pre_calculated_kpis_for_day: Dict[str, Any] = {}
    if filter_chw_id and not daily_snapshot_df.empty:
        # Example: Max fatigue score from WORKER_SELF_CHECK encounters for the selected CHW on that day
        worker_self_check_records = daily_snapshot_df[
            (daily_snapshot_df.get('chw_id') == filter_chw_id) &
            (daily_snapshot_df.get('encounter_type') == 'WORKER_SELF_CHECK')
        ]
        if not worker_self_check_records.empty:
            # Use a relevant score column that indicates fatigue
            fatigue_score_col = next((col for col in ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score'] if col in worker_self_check_records.columns and worker_self_check_records[col].notna().any()), None)
            if fatigue_score_col:
                pre_calculated_kpis_for_day['worker_self_fatigue_index_today'] = worker_self_check_records[fatigue_score_col].max()
            else:
                pre_calculated_kpis_for_day['worker_self_fatigue_index_today'] = np.nan
        else:
            pre_calculated_kpis_for_day['worker_self_fatigue_index_today'] = np.nan # No self-check data
    
    logger.info(f"({log_prefix}) Data loaded and filtered: Daily Snapshot - {len(daily_snapshot_df)} recs, Trend Period - {len(period_trend_df)} recs.")
    return daily_snapshot_df, period_trend_df, pre_calculated_kpis_for_day


# --- Sidebar Filters ---
# Use a cached function to load data just for populating filter options, if full dataset is large.
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_data_for_filters():
    logger.debug("Loading minimal data for CHW dashboard filters.")
    # Load only 'chw_id' and 'zone_id' for efficiency if possible, or a sample.
    # For simplicity here, still loads more, but ideally optimized.
    return load_health_records(source_context="CHWDashboard/SidebarFilterPopulation")

filter_data_df = load_data_for_filters()

# Sidebar Header with Logo
if os.path.exists(settings.APP_LOGO_SMALL_PATH):
    st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=150)
else:
    st.sidebar.markdown("🌍 Sentinel", help="Sentinel Health Co-Pilot Logo") # Fallback text/icon

st.sidebar.header("🗓️ View Filters")

# CHW ID Filter
chw_id_options_list = _create_filter_options(filter_data_df, 'chw_id', ["CHW001", "CHW002", "CHW003"], "CHWs")
# Ensure session state for selectbox to prevent reset on rerun if not explicitly handled by key changes
if 'chw_dashboard_selected_chw_id' not in st.session_state:
    st.session_state.chw_dashboard_selected_chw_id = chw_id_options_list[0] # Default to "All CHWs"

selected_chw_id_filter_val = st.sidebar.selectbox(
    "Filter by CHW ID:",
    options=chw_id_options_list,
    key="chw_dashboard_selected_chw_id_widget", # Unique key for the widget
    # index=chw_id_options_list.index(st.session_state.chw_dashboard_selected_chw_id) # Set index from session state
)
st.session_state.chw_dashboard_selected_chw_id = selected_chw_id_filter_val # Update session state
actual_chw_id_for_query = None if selected_chw_id_filter_val.startswith("All ") else selected_chw_id_filter_val

# Zone Filter
zone_id_options_list = _create_filter_options(filter_data_df, 'zone_id', ["ZoneA", "ZoneB"], "Zones")
if 'chw_dashboard_selected_zone_id' not in st.session_state:
    st.session_state.chw_dashboard_selected_zone_id = zone_id_options_list[0]

selected_zone_id_filter_val = st.sidebar.selectbox(
    "Filter by Zone:",
    options=zone_id_options_list,
    key="chw_dashboard_selected_zone_id_widget",
    # index=zone_id_options_list.index(st.session_state.chw_dashboard_selected_zone_id)
)
st.session_state.chw_dashboard_selected_zone_id = selected_zone_id_filter_val
actual_zone_id_for_query = None if selected_zone_id_filter_val.startswith("All ") else selected_zone_id_filter_val


# Date Pickers
# Determine overall min/max dates from the data if possible, else use wide range.
min_allowable_date = date.today() - timedelta(days=max(180, settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 4))
max_allowable_date = date.today()

# Daily Snapshot Date Picker
selected_daily_snapshot_date = st.sidebar.date_input(
    "View Daily Activity For:",
    value=max_allowable_date, # Default to today or most recent data
    min_value=min_allowable_date,
    max_value=max_allowable_date,
    key="chw_dashboard_daily_snapshot_datepicker"
)

# Trend Date Range Picker
default_trend_end_date = selected_daily_snapshot_date # Align trend end with snapshot date initially
default_trend_start_date = default_trend_end_date - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_trend_start_date < min_allowable_date:
    default_trend_start_date = min_allowable_date

# Ensure date_input for range gets a list/tuple of two dates
trend_date_range_selection = st.sidebar.date_input(
    "Select Trend Date Range:",
    value=[default_trend_start_date, default_trend_end_date], # Pass as list/tuple
    min_value=min_allowable_date,
    max_value=max_allowable_date,
    key="chw_dashboard_trend_datepicker"
)
# Unpack the date range selection
selected_trend_start_date_val: date
selected_trend_end_date_val: date
if isinstance(trend_date_range_selection, (list, tuple)) and len(trend_date_range_selection) == 2:
    selected_trend_start_date_val, selected_trend_end_date_val = trend_date_range_selection
else: # Fallback if somehow it's not a range (should not happen with st.date_input for range)
    selected_trend_start_date_val = default_trend_start_date
    selected_trend_end_date_val = default_trend_end_date
    st.sidebar.warning("Trend date range error. Using defaults.")

if selected_trend_start_date_val > selected_trend_end_date_val:
    st.sidebar.error("Trend Start Date must be on or before Trend End Date.")
    # Potentially revert to defaults or stop, here we just show error and proceed with invalid range for now
    # For robustness, could force end_date = start_date or use previous valid range.
    selected_trend_end_date_val = selected_trend_start_date_val # Simple fix: make range one day

# --- Load Data Based on Filters ---
try:
    daily_df_chw, period_df_chw, daily_pre_calc_kpis = get_chw_dashboard_processed_data(
        selected_date_view=selected_daily_snapshot_date,
        trend_range_start_date=selected_trend_start_date_val,
        trend_range_end_date=selected_trend_end_date_val,
        filter_chw_id=actual_chw_id_for_query,
        filter_zone_id=actual_zone_id_for_query
    )
except Exception as e_data_load_main:
    logger.error(f"CHW Dashboard: Failed to load or process main dashboard data: {e_data_load_main}", exc_info=True)
    st.error(f"Error loading CHW dashboard data: {str(e_data_load_main)}. Please try again or contact support.")
    st.stop() # Stop execution if data loading fails catastrophically

# --- Display Filter Context ---
filter_context_parts = [f"Date: **{selected_daily_snapshot_date.strftime('%d %b %Y')}**"]
if actual_chw_id_for_query:
    filter_context_parts.append(f"CHW: **{actual_chw_id_for_query}**")
if actual_zone_id_for_query:
    filter_context_parts.append(f"Zone: **{actual_zone_id_for_query}**")
st.info(f"Displaying data for: {', '.join(filter_context_parts)}")


# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not daily_df_chw.empty:
    try:
        # Pass the pre-calculated KPIs (like fatigue) to the summary metrics function
        daily_summary_metrics_chw = calculate_chw_daily_summary_metrics(
            chw_daily_encounter_df=daily_df_chw, # Pass the already filtered daily_df
            for_date=selected_daily_snapshot_date, # Pass the target date
            chw_daily_kpi_input_data=daily_pre_calc_kpis # Pass pre-calculated values
        )
    except Exception as e_metrics:
        logger.error(f"Error calculating CHW daily summary metrics: {e_metrics}", exc_info=True)
        st.warning("Could not calculate daily summary metrics. Some KPIs may be missing.")
        daily_summary_metrics_chw = {} # Fallback to empty dict

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi_card(
            title="Visits Today", value_str=str(daily_summary_metrics_chw.get("visits_count", 0)),
            icon="👥", help_text="Total unique patients encountered by the CHW/team today."
        )
    
    high_prio_followups = daily_summary_metrics_chw.get("high_ai_prio_followups_count", 0)
    prio_status = "ACCEPTABLE" if high_prio_followups <= 2 else ("MODERATE_CONCERN" if high_prio_followups <= 5 else "HIGH_CONCERN")
    with kpi_cols[1]:
        render_kpi_card(
            title="High Prio Follow-ups", value_str=str(high_prio_followups),
            icon="🎯", status_level=prio_status,
            help_text=f"Patients needing urgent follow-up based on AI priority scores (≥{settings.FATIGUE_INDEX_HIGH_THRESHOLD})."
        )
    
    critical_spo2_cases = daily_summary_metrics_chw.get("critical_spo2_cases_identified_count", 0)
    spo2_status = "HIGH_CONCERN" if critical_spo2_cases > 0 else "ACCEPTABLE"
    with kpi_cols[2]:
        render_kpi_card(
            title="Critical SpO2 Cases", value_str=str(critical_spo2_cases),
            icon="💨", status_level=spo2_status,
            help_text=f"Patients identified with SpO2 < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}%."
        )

    high_fever_cases = daily_summary_metrics_chw.get("high_fever_cases_identified_count", 0) # Ensure this key matches output
    fever_status = "HIGH_CONCERN" if high_fever_cases > 0 else "ACCEPTABLE"
    with kpi_cols[3]:
        render_kpi_card(
            title="High Fever Cases", value_str=str(high_fever_cases),
            icon="🔥", status_level=fever_status,
            help_text=f"Patients identified with temperature ≥ {settings.ALERT_BODY_TEMP_HIGH_FEVER_C}°C."
        )
else:
    st.markdown("No activity data available for the selected date/filters to display daily performance snapshot.")
st.divider()


# --- Section 2: Key Alerts & Actionable Tasks ---
st.header("🚦 Key Alerts & Tasks")
# Key Alerts
# The component generate_chw_alerts should handle empty df internally
try:
    chw_patient_alerts_list = generate_chw_alerts( # Renamed component
        patient_encounter_data_df=daily_df_chw,
        for_date=selected_daily_snapshot_date,
        chw_zone_context_str=actual_zone_id_for_query or "All Zones", # Pass actual zone or "All"
        max_alerts_to_return=8 # Limit for dashboard display
    )
except ValueError as ve_alerts: # Catch specific error if component raises it for no data
    logger.warning(f"Value error generating CHW alerts: {ve_alerts}")
    chw_patient_alerts_list = []
    st.caption(f"Alert generation note: {ve_alerts}")
except Exception as e_alerts:
    logger.error(f"Error generating CHW patient alerts: {e_alerts}", exc_info=True)
    st.warning("Could not generate patient alerts. Alert list may be incomplete.")
    chw_patient_alerts_list = []


if chw_patient_alerts_list:
    st.subheader("Priority Patient Alerts:")
    critical_alerts_found_today = False
    for alert_item in chw_patient_alerts_list:
        if alert_item.get("alert_level") == "CRITICAL":
            critical_alerts_found_today = True
            render_traffic_light_indicator(
                message=f"Pt. {alert_item.get('patient_id', 'N/A')}: {alert_item.get('primary_reason', 'Critical Alert')}",
                status_level="HIGH_RISK", # Use a status level understood by the renderer
                details_text=(
                    f"Details: {alert_item.get('brief_details','N/A')} | "
                    f"Context: {alert_item.get('context_info','N/A')} | "
                    f"Action: {alert_item.get('suggested_action_code','REVIEW_IMMEDIATELY')}"
                )
            )
    if not critical_alerts_found_today:
        st.info("No CRITICAL patient alerts for this selection.")

    warning_alerts_list = [a for a in chw_patient_alerts_list if a.get("alert_level") == "WARNING"]
    if warning_alerts_list:
        st.markdown("###### Warning Alerts:")
        for warn_item in warning_alerts_list:
            render_traffic_light_indicator(
                message=f"Pt. {warn_item.get('patient_id', 'N/A')}: {warn_item.get('primary_reason', 'Warning')}",
                status_level="MODERATE_CONCERN", # Use a status level for renderer
                details_text=f"Details: {warn_item.get('brief_details','N/A')} | Context: {warn_item.get('context_info','N/A')}"
            )
    elif not critical_alerts_found_today : # No critical and no warnings
        st.info("Only informational alerts (if any) were generated. No urgent patient alerts.")

elif not daily_df_chw.empty: # Data exists but no alerts generated
    st.success("✅ No significant patient alerts generated for the current selection.")
else: # No data to generate from
    st.markdown("No activity data to generate patient alerts.")


# Actionable Tasks (Placeholder - Task generation logic to be refined)
# try:
#     prioritized_chw_tasks_list = generate_chw_tasks(
#         source_patient_data_df=daily_df_chw, # Use daily data for today's tasks
#         for_date=selected_daily_snapshot_date,
#         chw_id_context=actual_chw_id_for_query,
#         zone_context_str=actual_zone_id_for_query or "All Zones",
#         max_tasks_to_return_for_summary=10
#     )
# except ValueError as ve_tasks: # If component raises ValueError for no data
#     logger.warning(f"Value error generating CHW tasks: {ve_tasks}")
#     prioritized_chw_tasks_list = []
#     st.caption(f"Task generation note: {ve_tasks}")
# except Exception as e_tasks:
#     logger.error(f"Error generating CHW prioritized tasks: {e_tasks}", exc_info=True)
#     st.warning("Could not generate prioritized tasks list.")
#     prioritized_chw_tasks_list = []

# if prioritized_chw_tasks_list:
#     st.subheader("Top Priority Tasks:")
#     # Display tasks in a more user-friendly way, perhaps a styled table or list
#     tasks_df_display = pd.DataFrame(prioritized_chw_tasks_list)
#     # Define columns to show in the dashboard task table
#     task_display_cols = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id', 'alert_source_info']
#     actual_task_cols_to_show = [col for col in task_display_cols if col in tasks_df_display.columns]
    
#     if not tasks_df_display.empty and actual_task_cols_to_show:
#         st.dataframe(
#             tasks_df_display[actual_task_cols_to_show],
#             use_container_width=True,
#             height=min(380, len(tasks_df_display) * 40 + 60), # Dynamic height
#             hide_index=True
#         )
#     else:
#         st.info("Task data is available but could not be displayed (column mismatch or empty after selection).")
# elif not daily_df_chw.empty:
#      st.info("No high-priority tasks identified for the current selection.")
# else:
#      st.markdown("No activity data to generate tasks.")
# st.divider()
# Temporarily disabling task section display due to ongoing refactor of task_processor
st.subheader("Prioritized Tasks (Under Review)")
st.caption("Task generation and display component is currently under review and will be updated.")
st.divider()


# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch")
if not daily_df_chw.empty:
    try:
        chw_epi_signals_data = extract_chw_epi_signals(
            chw_daily_encounter_df=daily_df_chw,
            pre_calculated_chw_kpis=daily_pre_calc_kpis, # Pass pre-calculated KPIs
            for_date=selected_daily_snapshot_date,
            chw_zone_context=actual_zone_id_for_query or "All Zones",
            max_symptom_clusters_to_report=3
        )
    except ValueError as ve_epi: # If component raises this
        logger.warning(f"Value error extracting CHW epi signals: {ve_epi}")
        chw_epi_signals_data = {}
        st.caption(f"Epi signal extraction note: {ve_epi}")
    except Exception as e_epi:
        logger.error(f"Error extracting CHW epi signals: {e_epi}", exc_info=True)
        st.warning("Could not extract local epidemiological signals.")
        chw_epi_signals_data = {}

    epi_kpi_cols = st.columns(3)
    with epi_kpi_cols[0]:
        sympt_key_cond = chw_epi_signals_data.get("symptomatic_patients_key_conditions_count", 0)
        render_kpi_card(
            title="Symptomatic (Key Cond.)", value_str=str(sympt_key_cond),
            icon="🤒", units="cases today",
            help_text=f"Patients presenting with symptoms related to key conditions (e.g., {', '.join(settings.KEY_CONDITIONS_FOR_ACTION[:2])}, etc.)."
        )
    with epi_kpi_cols[1]:
        new_malaria = chw_epi_signals_data.get("newly_identified_malaria_patients_count", 0)
        malaria_status = "HIGH_CONCERN" if new_malaria > 1 else ("MODERATE_CONCERN" if new_malaria == 1 else "ACCEPTABLE")
        render_kpi_card(
            title="New Malaria Cases", value_str=str(new_malaria),
            icon="🦟", units="cases today", status_level=malaria_status,
            help_text="New malaria cases (e.g., RDT positive) identified today."
        )
    with epi_kpi_cols[2]:
        pending_tb_contacts = chw_epi_signals_data.get("pending_tb_contact_tracing_tasks_count", 0)
        tb_contact_status = "MODERATE_CONCERN" if pending_tb_contacts > 0 else "ACCEPTABLE"
        render_kpi_card(
            title="Pending TB Contacts", value_str=str(pending_tb_contacts),
            icon="👥", units="to trace", status_level=tb_contact_status,
            help_text="Number of TB contacts still requiring follow-up tracing."
        )

    detected_symptom_clusters = chw_epi_signals_data.get("detected_symptom_clusters", [])
    if detected_symptom_clusters:
        st.markdown("###### Detected Symptom Clusters (Requires Verification):")
        for cluster_item in detected_symptom_clusters:
            st.warning(
                f"⚠️ **{cluster_item.get('symptoms_pattern', 'Unknown Pattern')}**: "
                f"{cluster_item.get('patient_count', 'N/A')} cases in {cluster_item.get('location_hint', 'area')}. "
                f"Consider investigation."
            )
    elif 'patient_reported_symptoms' in daily_df_chw.columns and daily_df_chw['patient_reported_symptoms'].notna().any(): # Only if relevant col exists
        st.info("No significant symptom clusters detected based on current data and criteria.")
else:
    st.markdown("No activity data available for the selected date/filters to extract epi signals.")
st.divider()


# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_str = f"{selected_trend_start_date_val.strftime('%d %b %Y')} - {selected_trend_end_date_val.strftime('%d %b %Y')}"
trend_filter_context_str = ""
if actual_chw_id_for_query: trend_filter_context_str += f" for CHW **{actual_chw_id_for_query}**"
if actual_zone_id_for_query: trend_filter_context_str += f" in Zone **{actual_zone_id_for_query}**"
st.markdown(f"Trends from **{trend_period_display_str}**{trend_filter_context_str if trend_filter_context_str else ' (All CHWs/Zones in Period)'}.")

if not period_df_chw.empty:
    try:
        chw_activity_trends = calculate_chw_activity_trends_data(
            chw_historical_health_df=period_df_chw, # Pass the already filtered period_df
            trend_start_date_input=selected_trend_start_date_val, # For context if needed by component
            trend_end_date_input=selected_trend_end_date_val,     # For context if needed by component
            zone_filter=actual_zone_id_for_query, # Pass zone filter for component's internal use
            time_period_aggregation='D' # Daily trends typically useful for CHW supervisor
        )
    except Exception as e_trends:
        logger.error(f"Error calculating CHW activity trends: {e_trends}", exc_info=True)
        st.warning("Could not calculate activity trends. Trend charts may be unavailable.")
        chw_activity_trends = {} # Fallback

    trend_plot_cols = st.columns(2)
    with trend_plot_cols[0]:
        visits_trend_series = chw_activity_trends.get("patient_visits_trend")
        if isinstance(visits_trend_series, pd.Series) and not visits_trend_series.empty:
            st.plotly_chart(
                plot_annotated_line_chart(
                    data_series=visits_trend_series, chart_title="Daily Patient Visits Trend",
                    y_axis_label="# Patients Visited", y_values_are_counts=True
                ), use_container_width=True
            )
        else:
            st.caption("No patient visit trend data available for this selection.")
            
    with trend_plot_cols[1]:
        high_prio_followups_trend_series = chw_activity_trends.get("high_priority_followups_trend")
        if isinstance(high_prio_followups_trend_series, pd.Series) and not high_prio_followups_trend_series.empty:
            st.plotly_chart(
                plot_annotated_line_chart(
                    data_series=high_prio_followups_trend_series, chart_title="Daily High Prio. Follow-ups Trend",
                    y_axis_label="# High Prio Follow-ups", y_values_are_counts=True
                ), use_container_width=True
            )
        else:
            st.caption("No high-priority follow-up trend data available for this selection.")
else:
    st.markdown("No historical data available for the selected trend period/filters.")

logger.info(
    f"CHW Supervisor View loaded for Date: {selected_daily_snapshot_date}, "
    f"CHW: {actual_chw_id_for_query or 'All'}, Zone: {actual_zone_id_for_query or 'All'}"
)
