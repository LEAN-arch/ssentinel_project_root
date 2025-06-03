# sentinel_project_root/pages/population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np # For np.nan and other numeric ops
import logging
from datetime import date, timedelta
import html # For escaping HTML in custom markdown
import plotly.express as px # For direct use of complex plots like histograms if needed

# --- Sentinel System Imports from Refactored Structure ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_zone_data
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe # For caching
    from visualization.plots import plot_bar_chart, create_empty_figure # Use new plot functions
    from visualization.ui_elements import display_custom_styled_kpi_box # Specific KPI box style
except ImportError as e_pop_dash:
    import sys
    st.error(
        f"Population Dashboard Import Error: {e_pop_dash}. "
        f"Please ensure all modules are correctly placed and dependencies installed. "
        f"Relevant Python Path: {sys.path}"
    )
    logger_pop = logging.getLogger(__name__)
    logger_pop.error(f"Population Dashboard Import Error: {e_pop_dash}", exc_info=True)
    st.stop()

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Page Title and Introduction ---
st.title(f"📊 {settings.APP_NAME} - Population Health Analytics & Research Console")
st.markdown(
    "In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, "
    "and health system factors using aggregated population-level data."
)
st.divider()

# --- Data Loading Function for Population Analytics Console (Cached) ---
@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    hash_funcs={pd.DataFrame: hash_dataframe_safe},
    show_spinner="Loading and preparing population analytics dataset..."
)
def get_population_analytics_datasets_processed(log_source_context: str = "PopAnalyticsConsole/LoadData") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and prepares the primary datasets for the Population Analytics Console:
    1. AI-enriched health records.
    2. Zone attributes (without geometry, for SDOH context).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (enriched_health_df, zone_attributes_df)
    """
    logger.info(f"({log_source_context}) Loading population health records and zone attribute data.")
    
    # 1. Load and Enrich Health Records
    raw_health_df_for_pop = load_health_records(source_context=f"{log_source_context}/HealthRecsLoad")
    
    enriched_health_df_for_pop: pd.DataFrame
    if isinstance(raw_health_df_for_pop, pd.DataFrame) and not raw_health_df_for_pop.empty:
        # Apply AI models (risk, priority scores) to the health data
        enriched_health_df_for_pop, _ = apply_ai_models(raw_health_df_for_pop.copy(), source_context=f"{log_source_context}/AIEnrichHealth")
    else:
        logger.warning(f"({log_source_context}) Raw health records for population analytics are empty or invalid. AI enrichment skipped.")
        # Define schema for empty DF if raw load fails, to match expected structure after AI enrichment
        base_cols_pop = ['patient_id', 'encounter_date', 'condition', 'age', 'gender', 'zone_id'] # Example base
        ai_cols_pop = ['ai_risk_score', 'ai_followup_priority_score']
        enriched_health_df_for_pop = pd.DataFrame(columns=list(set(base_cols_pop + ai_cols_pop)))

    # 2. Load Zone Attributes (for SDOH context, no geometry needed for this dashboard typically)
    # `load_zone_data` returns a pandas DataFrame; we'll drop geometry-related columns if present.
    zone_data_full_df = load_zone_data(source_context=f"{log_source_context}/ZoneDataLoad")
    
    zone_attributes_df_for_pop: pd.DataFrame
    # Define expected SDOH-related attribute columns from zone data
    expected_sdoh_cols_from_zone_data = [
        'zone_id', 'name', 'population', 'socio_economic_index', 
        'avg_travel_time_clinic_min', 'predominant_hazard_type', 
        'primary_livelihood', 'water_source_main', 'area_sqkm' # area_sqkm for density if needed
    ]
    if isinstance(zone_data_full_df, pd.DataFrame) and not zone_data_full_df.empty:
        # Select only the attribute columns, exclude geometry-related ones
        cols_to_keep_from_zone_data = [col for col in expected_sdoh_cols_from_zone_data if col in zone_data_full_df.columns]
        if 'zone_id' not in cols_to_keep_from_zone_data and 'zone_id' in zone_data_full_df.columns:
            cols_to_keep_from_zone_data.append('zone_id') # Ensure zone_id is always there for merges

        if cols_to_keep_from_zone_data :
            zone_attributes_df_for_pop = zone_data_full_df[list(set(cols_to_keep_from_zone_data))].copy()
            logger.info(f"({log_source_context}) Loaded {len(zone_attributes_df_for_pop)} zone attributes records for SDOH context.")
        else:
            logger.warning(f"({log_source_context}) No expected SDOH attribute columns found in loaded zone data. SDOH analysis will be limited.")
            zone_attributes_df_for_pop = pd.DataFrame(columns=expected_sdoh_cols_from_zone_data)
    else:
        logger.warning(f"({log_source_context}) Zone attributes data unavailable or empty. SDOH analytics will be limited.")
        zone_attributes_df_for_pop = pd.DataFrame(columns=expected_sdoh_cols_from_zone_data) # Empty with schema

    if enriched_health_df_for_pop.empty: 
        logger.error(f"({log_source_context}) CRITICAL FAILURE: Health data for population analytics remains empty after all processing steps.")
        # Streamlit error will be shown by calling code if this is critical
        
    return enriched_health_df_for_pop, zone_attributes_df_for_pop


# --- Load Datasets for the Console ---
try:
    health_df_pop_main, zone_attributes_pop_sdoh = get_population_analytics_datasets_processed()
except Exception as e_pop_data_main:
    logger.error(f"Population Dashboard: Failed to load main datasets: {e_pop_data_main}", exc_info=True)
    st.error(f"Error loading Population Analytics data: {str(e_pop_data_main)}. Dashboard functionality will be limited.")
    health_df_pop_main, zone_attributes_pop_sdoh = pd.DataFrame(), pd.DataFrame() # Fallback to empty

if health_df_pop_main.empty: 
    st.error(
        "🚨 **Critical Data Failure:** Primary health dataset for population analytics could not be loaded or is empty. "
        "Most console features will be unavailable. Please check data sources and application logs."
    )
    # st.stop() # Optionally stop if no data makes the page unusable.

# --- Sidebar Filters ---
if os.path.exists(settings.APP_LOGO_SMALL_PATH):
    st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
st.sidebar.header("🔎 Analytics Filters")

# Determine min/max dates from the loaded health data for date pickers
min_date_for_pop_analysis = date.today() - timedelta(days=365*3) # Default 3 years back
max_date_for_pop_analysis = date.today()

if 'encounter_date' in health_df_pop_main.columns and health_df_pop_main['encounter_date'].notna().any():
    # Ensure it's datetime (loader should handle this, but defensive check)
    if not pd.api.types.is_datetime64_any_dtype(health_df_pop_main['encounter_date']):
        health_df_pop_main['encounter_date'] = pd.to_datetime(health_df_pop_main['encounter_date'], errors='coerce')
    
    if health_df_pop_main['encounter_date'].notna().any(): # Check again after coercion
        min_date_for_pop_analysis = health_df_pop_main['encounter_date'].min().date()
        max_date_for_pop_analysis = health_df_pop_main['encounter_date'].max().date()

if min_date_for_pop_analysis > max_date_for_pop_analysis: # Safety if min somehow ends up after max
    min_date_for_pop_analysis = max_date_for_pop_analysis 

# Date Range Filter
# Session state for date range picker
pop_date_range_session_key = "pop_dashboard_date_range_selection"
if pop_date_range_session_key not in st.session_state:
    st.session_state[pop_date_range_session_key] = [min_date_for_pop_analysis, max_date_for_pop_analysis] # Default to full range

selected_date_range_pop_ui = st.sidebar.date_input(
    "Select Date Range for Analysis:",
    value=st.session_state[pop_date_range_session_key],
    min_value=min_date_for_pop_analysis,
    max_value=max_date_for_pop_analysis,
    key=f"{pop_date_range_session_key}_widget"
)
if isinstance(selected_date_range_pop_ui, (list, tuple)) and len(selected_date_range_pop_ui) == 2:
    st.session_state[pop_date_range_session_key] = selected_date_range_pop_ui
    start_date_filter_pop, end_date_filter_pop = selected_date_range_pop_ui
else: # Fallback if UI component returns unexpected value
    start_date_filter_pop, end_date_filter_pop = st.session_state[pop_date_range_session_key]
    st.sidebar.warning("Date range selection error. Using previous/default range.")

if start_date_filter_pop > end_date_filter_pop:
    st.sidebar.error("Start date must be on or before end date. Adjusting end date.")
    end_date_filter_pop = start_date_filter_pop
    st.session_state[pop_date_range_session_key][1] = end_date_filter_pop


# Filter the main health DataFrame based on selected date range
analytics_df_filtered_by_date: pd.DataFrame
if 'encounter_date' not in health_df_pop_main.columns:
    st.error("Critical error: 'encounter_date' column is missing from the health dataset. Cannot filter by date.")
    analytics_df_filtered_by_date = pd.DataFrame() # Empty DF if date col missing
    # st.stop() # Might be too disruptive, allow page to render with warning.
else:
    # Ensure 'encounter_date' is datetime before filtering (should be from loader)
    if not pd.api.types.is_datetime64_any_dtype(health_df_pop_main['encounter_date']):
         health_df_pop_main['encounter_date'] = pd.to_datetime(health_df_pop_main['encounter_date'], errors='coerce')

    analytics_df_filtered_by_date = health_df_pop_main[
        (health_df_pop_main['encounter_date'].notna()) & 
        (health_df_pop_main['encounter_date'].dt.date >= start_date_filter_pop) &
        (health_df_pop_main['encounter_date'].dt.date <= end_date_filter_pop)
    ].copy() # Use .copy() for subsequent modifications

# Condition Filter
selected_condition_filter_ui = "All Conditions" # Default
if 'condition' in analytics_df_filtered_by_date.columns:
    unique_conditions_for_filter = ["All Conditions"] + sorted(analytics_df_filtered_by_date['condition'].dropna().unique().tolist())
    # Session state for condition filter
    pop_condition_session_key = "pop_dashboard_condition_selection"
    if pop_condition_session_key not in st.session_state:
        st.session_state[pop_condition_session_key] = unique_conditions_for_filter[0] # Default to "All Conditions"
    
    selected_condition_filter_ui = st.sidebar.selectbox(
        "Filter by Condition Group (Optional):", options=unique_conditions_for_filter,
        key=f"{pop_condition_session_key}_widget",
        index=unique_conditions_for_filter.index(st.session_state[pop_condition_session_key])
    )
    st.session_state[pop_condition_session_key] = selected_condition_filter_ui
    
    if selected_condition_filter_ui != "All Conditions":
        analytics_df_filtered_by_date = analytics_df_filtered_by_date[
            analytics_df_filtered_by_date['condition'] == selected_condition_filter_ui
        ]
else:
    st.sidebar.caption("Condition filter unavailable ('condition' column missing in data).")


# Zone Filter (using zone_attributes_pop_sdoh for options)
selected_zone_display_ui = "All Zones" # Default
zone_display_to_id_map_pop_filter: Dict[str, str] = {}
zone_options_for_pop_filter = ["All Zones"]

if isinstance(zone_attributes_pop_sdoh, pd.DataFrame) and not zone_attributes_pop_sdoh.empty and \
   'zone_id' in zone_attributes_pop_sdoh.columns:
    
    temp_zone_options = []
    for _, zone_row_filter_opt in zone_attributes_pop_sdoh.iterrows():
        zone_id_val_opt = str(zone_row_filter_opt['zone_id']).strip()
        zone_name_val_opt = str(zone_row_filter_opt.get('name', zone_id_val_opt)).strip() # Use name if available
        
        display_option_str = f"{zone_name_val_opt} ({zone_id_val_opt})" if zone_name_val_opt != zone_id_val_opt and zone_name_val_opt != "Unknown" else zone_id_val_opt
        if display_option_str not in zone_display_to_id_map_pop_filter: # Avoid duplicate display options
            temp_zone_options.append(display_option_str)
            zone_display_to_id_map_pop_filter[display_option_str] = zone_id_val_opt
    
    if temp_zone_options:
        zone_options_for_pop_filter.extend(sorted(list(set(temp_zone_options)))) # Add unique sorted options
else:
    st.sidebar.caption("Zone filter options limited (zone attributes data missing or invalid).")

if len(zone_options_for_pop_filter) > 1 : # Only show if there are actual zone options beyond "All Zones"
    pop_zone_session_key = "pop_dashboard_zone_selection"
    if pop_zone_session_key not in st.session_state:
        st.session_state[pop_zone_session_key] = zone_options_for_pop_filter[0] # Default "All Zones"

    selected_zone_display_ui = st.sidebar.selectbox(
        "Filter by Operational Zone (Optional):", options=zone_options_for_pop_filter, 
        key=f"{pop_zone_session_key}_widget",
        index=zone_options_for_pop_filter.index(st.session_state[pop_zone_session_key])
    )
    st.session_state[pop_zone_session_key] = selected_zone_display_ui
    
    if selected_zone_display_ui != "All Zones":
        actual_zone_id_to_filter_by = zone_display_to_id_map_pop_filter.get(selected_zone_display_ui, selected_zone_display_ui) # Fallback if map key missing
        if 'zone_id' in analytics_df_filtered_by_date.columns:
            analytics_df_filtered_by_date = analytics_df_filtered_by_date[
                analytics_df_filtered_by_date['zone_id'] == actual_zone_id_to_filter_by
            ]
        else:
            st.sidebar.caption("Cannot filter by zone ('zone_id' column missing in health data).")


# Final check if DataFrame is empty after all filters
if analytics_df_filtered_by_date.empty and \
   (selected_condition_filter_ui != "All Conditions" or selected_zone_display_ui != "All Zones" or \
    start_date_filter_pop != min_date_for_pop_analysis or end_date_filter_pop != max_date_for_pop_analysis):
    st.warning(f"No health data available for the selected filters. Please broaden your filter criteria or check data sources.")


# --- Population Health Snapshot KPIs (using custom styled boxes) ---
st.subheader(
    f"Population Health Snapshot "
    f"({start_date_filter_pop.strftime('%d %b %Y')} - {end_date_filter_pop.strftime('%d %b %Y')}, "
    f"Condition: {selected_condition_filter_ui}, Zone: {selected_zone_display_ui})"
)

if analytics_df_filtered_by_date.empty:
    st.info("Insufficient data after filtering to display population summary KPIs.")
else:
    # Calculate KPIs for the filtered data
    kpi_cols_pop_summary = st.columns(4) # Use 4 columns for KPIs
    
    num_total_unique_patients_pop = 0
    if 'patient_id' in analytics_df_filtered_by_date.columns:
        num_total_unique_patients_pop = analytics_df_filtered_by_date['patient_id'].nunique()
    
    mean_ai_risk_score_val_pop = np.nan
    if 'ai_risk_score' in analytics_df_filtered_by_date.columns and analytics_df_filtered_by_date['ai_risk_score'].notna().any():
        mean_ai_risk_score_val_pop = analytics_df_filtered_by_date['ai_risk_score'].mean()
    
    count_high_risk_patients_pop_val = 0
    percent_high_risk_patients_pop_val = 0.0
    if 'ai_risk_score' in analytics_df_filtered_by_date.columns and num_total_unique_patients_pop > 0:
        df_high_risk_patients_pop = analytics_df_filtered_by_date[
            analytics_df_filtered_by_date['ai_risk_score'] >= settings.RISK_SCORE_HIGH_THRESHOLD
        ]
        if 'patient_id' in df_high_risk_patients_pop.columns:
            count_high_risk_patients_pop_val = df_high_risk_patients_pop['patient_id'].nunique()
        
        percent_high_risk_patients_pop_val = (count_high_risk_patients_pop_val / num_total_unique_patients_pop) * 100 if num_total_unique_patients_pop > 0 else 0.0
    
    top_condition_name_pop_kpi, num_top_condition_encounters_pop_kpi = "N/A", 0
    if 'condition' in analytics_df_filtered_by_date.columns and analytics_df_filtered_by_date['condition'].notna().any():
        # Count encounters for top condition, not unique patients here for this specific KPI
        condition_encounter_counts_pop = analytics_df_filtered_by_date['condition'].value_counts()
        if not condition_encounter_counts_pop.empty:
            top_condition_name_pop_kpi = condition_encounter_counts_pop.idxmax()
            num_top_condition_encounters_pop_kpi = condition_encounter_counts_pop.max()

    with kpi_cols_pop_summary[0]:
        display_custom_styled_kpi_box(label="Total Unique Patients", value=num_total_unique_patients_pop)
    with kpi_cols_pop_summary[1]:
        display_custom_styled_kpi_box(label="Avg. AI Risk Score", value=f"{mean_ai_risk_score_val_pop:.1f}" if pd.notna(mean_ai_risk_score_val_pop) else "N/A")
    with kpi_cols_pop_summary[2]:
        display_custom_styled_kpi_box(
            label="% High AI Risk Patients",
            value=f"{percent_high_risk_patients_pop_val:.1f}%",
            sub_text=f"({count_high_risk_patients_pop_val:,} patients)"
        )
    with kpi_cols_pop_summary[3]:
        display_custom_styled_kpi_box(
            label="Top Condition (Encounters)",
            value=html.escape(str(top_condition_name_pop_kpi)), # Escape for safety
            sub_text=f"{num_top_condition_encounters_pop_kpi:,} encounters",
            highlight_edge_color=settings.COLOR_RISK_MODERATE # Example highlight
        )

# --- Tabs for Detailed Population Analytics ---
pop_analytics_tab_titles_list = ["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"]
tab_pop_epi, tab_pop_demog_sdoh, tab_pop_clinical, tab_pop_systems = st.tabs(pop_analytics_tab_titles_list)

with tab_pop_epi:
    st.header(f"Epidemiological Overview (Filters: {selected_condition_filter_ui} | {selected_zone_display_ui})")
    if analytics_df_filtered_by_date.empty:
        st.info("No data available for Epidemiological Overview with the current filter selections.")
    else:
        # Top Conditions by Unique Patient Count
        if 'condition' in analytics_df_filtered_by_date.columns and 'patient_id' in analytics_df_filtered_by_date.columns:
            df_condition_unique_patient_counts_plot = analytics_df_filtered_by_date.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients') # Top 10
            if not df_condition_unique_patient_counts_plot.empty:
                st.plotly_chart(plot_bar_chart(
                    df_input=df_condition_unique_patient_counts_plot, x_col_name='condition', y_col_name='unique_patients', 
                    chart_title="Top Conditions by Unique Patient Count (Filtered Set)", 
                    orientation_bar='h', y_values_are_counts_flag=True, chart_height=450,
                    x_axis_label_text="Unique Patient Count", y_axis_label_text="Condition"
                ), use_container_width=True)
            else:
                st.caption("No aggregated condition counts found for unique patients with current filters.")
        
        # Patient AI Risk Score Distribution
        if 'ai_risk_score' in analytics_df_filtered_by_date.columns and analytics_df_filtered_by_date['ai_risk_score'].notna().any():
            try:
                fig_ai_risk_dist_pop = px.histogram( # Using Plotly Express directly for histograms
                    analytics_df_filtered_by_date.dropna(subset=['ai_risk_score']), 
                    x="ai_risk_score", nbins=25, # Adjust nbins as needed
                    title="Patient AI Risk Score Distribution (Filtered Set)",
                    labels={'ai_risk_score': 'AI Risk Score Bins', 'count': 'Number of Records'} 
                )
                fig_ai_risk_dist_pop.update_layout(
                    bargap=0.1, height=settings.WEB_PLOT_COMPACT_HEIGHT,
                    title_x=0.05 # Align title left
                )
                st.plotly_chart(fig_ai_risk_dist_pop, use_container_width=True)
            except Exception as e_hist:
                logger.error(f"Population Dashboard: Error plotting AI risk histogram: {e_hist}", exc_info=True)
                st.warning("Could not display AI Risk Score Distribution chart.")
        
        st.caption(
            "Note: True incidence/prevalence trends require careful definition of 'new case' vs 'active case' and appropriate population denominators. "
            "This section provides overview counts and distributions based on available encounter data for the selected period."
        )

with tab_pop_demog_sdoh:
    st.header("Demographics & Social Determinants of Health (SDOH) Context")
    if analytics_df_filtered_by_date.empty:
        st.info("No health data available for Demographics & SDOH analysis with the current filter selections.")
    else:
        st.markdown("_(Placeholder: Detailed charts for Age Distribution, Gender Distribution. For SDOH insights, "
                    "the `analytics_df_filtered_by_date` can be merged with `zone_attributes_pop_sdoh` "
                    "on `zone_id` to correlate health outcomes like AI Risk Score with Zone SES, "
                    "Travel Time to Clinic, Primary Livelihood, Water Source, etc. "
                    "This would involve creating scatter plots, box plots by SDOH category, or correlation matrices.)_")
        if zone_attributes_pop_sdoh.empty:
            st.caption("Zone attribute data (required for SDOH-specific analysis) is currently unavailable or empty.")

with tab_pop_clinical:
    st.header("Clinical Insights & Diagnostic Patterns")
    if analytics_df_filtered_by_date.empty:
        st.info("No data available for Clinical Insights with the current filter selections.")
    else:
        st.markdown("_(Placeholder: Analyses for Top Reported Symptoms (frequency, trends if data supports), "
                    "Test Result Distributions (e.g., % Positive for key tests over time or by demographic strata), "
                    "and deeper dives into Test Positivity Trends for specific conditions or risk groups.)_")

with tab_pop_systems:
    st.header("Health Systems Performance & Equity Lens")
    if analytics_df_filtered_by_date.empty:
        st.info("No data available for Health Systems & Equity analysis with the current filter selections.")
    else:
        st.markdown("_(Placeholder: Analyses on Patient Encounters by Clinic/Zone (if applicable and data allows disaggregation), "
                    "Referral Pathway Completion Rates and Bottlenecks (if referral outcome data is comprehensive and linked), "
                    "and investigation of AI Risk Score Variations when stratified by SDOH factors "
                    "like Zone SES or Primary Livelihood to explore potential health equity considerations.)_")


st.divider()
st.caption(settings.APP_FOOTER_TEXT)
logger.info(
    f"Population Health Analytics Console page loaded. Filters: "
    f"Period=({start_date_filter_pop.isoformat()} to {end_date_filter_pop.isoformat()}), "
    f"Condition='{selected_condition_filter_ui}', Zone='{selected_zone_display_ui}'"
)
