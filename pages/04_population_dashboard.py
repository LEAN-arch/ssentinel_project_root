# sentinel_project_root/pages/04_population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# Plotting and other utilities
import plotly.express as px
# import html # Keep if you anticipate needing to escape HTML for custom components
# import re   # Keep if you need regex for text processing

# --- Configuration and Custom Module Imports ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_zone_data # Assuming load_zone_data exists
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe, convert_to_numeric # convert_to_numeric can be useful
    # Assuming these plot functions are generic enough or adaptable
    from visualization.plots import plot_bar_chart, create_empty_figure, plot_annotated_line_chart
    from visualization.ui_elements import display_custom_styled_kpi_box # If you have such a function
except ImportError as e_pop_dash_import:
    import sys
    current_file_path = Path(__file__).resolve()
    project_root_dir = current_file_path.parent.parent
    error_message = (
        f"Population Dashboard Import Error: {e_pop_dash_import}. "
        f"Ensure project root ('{project_root_dir}') is in sys.path and all modules are correct. "
        f"Current sys.path: {sys.path}"
    )
    try:
        st.error(error_message)
        st.stop()
    except NameError:
        print(error_message, file=sys.stderr)
        raise

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Page Configuration (Call this early) ---
try:
    page_icon_value = "🌍" # Default icon
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_FAVICON_PATH'):
        favicon_path = Path(settings.PROJECT_ROOT_DIR) / settings.APP_FAVICON_PATH
        if favicon_path.is_file():
            page_icon_value = str(favicon_path)
        else:
            logger.warning(f"Favicon for Population Dashboard not found: {favicon_path}")

    page_layout_value = "wide"
    if hasattr(settings, 'APP_LAYOUT'):
        page_layout_value = settings.APP_LAYOUT
        
    st.set_page_config(
        page_title=f"Population Analytics - {settings.APP_NAME if hasattr(settings, 'APP_NAME') else 'App'}",
        page_icon=page_icon_value,
        layout=page_layout_value
    )
except Exception as e_page_config:
    logger.error(f"Error applying page configuration for Population Dashboard: {e_page_config}", exc_info=True)
    st.set_page_config(page_title="Population Analytics", page_icon="🌍", layout="wide") # Fallback


# --- Page Title and Introduction ---
st.title(f"📊 {settings.APP_NAME if hasattr(settings, 'APP_NAME') else 'Sentinel Health Co-Pilot'} - Population Health Analytics & Research Console")
st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.divider()

# --- Data Loading and Caching ---
@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS if hasattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS') else 300,
    hash_funcs={pd.DataFrame: hash_dataframe_safe},
    show_spinner="Loading population analytics dataset..."
)
def get_population_analytics_datasets(log_ctx: str = "PopAnalyticsConsole/LoadData") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses health records and zone attributes for population analytics.
    Returns a tuple of (enriched_health_df, zone_attributes_df).
    DataFrames are empty if data is unavailable, not None.
    """
    logger.info(f"({log_ctx}) Initiating load for population health records and zone attributes.")

    # Load Health Records
    raw_health_df = load_health_records(source_context=f"{log_ctx}/HealthRecs")
    enriched_health_df = pd.DataFrame() # Default to empty
    base_health_cols_schema = ['patient_id', 'encounter_date', 'condition', 'age', 'gender', 'zone_id', 
                               'ai_risk_score', 'ai_followup_priority_score'] # Define expected schema

    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        logger.info(f"({log_ctx}) Raw health records loaded: {raw_health_df.shape[0]} rows. Applying AI models.")
        try:
            enriched_data, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrich")
            if isinstance(enriched_data, pd.DataFrame):
                enriched_health_df = enriched_data
            else:
                logger.warning(f"({log_ctx}) AI model application did not return a DataFrame. Using raw data for schema alignment.")
                # Ensure enriched_health_df has at least the base columns even if AI models fail
                enriched_health_df = raw_health_df.reindex(columns=base_health_cols_schema, fill_value=np.nan)
        except Exception as e_ai:
            logger.error(f"({log_ctx}) Error during AI model application: {e_ai}. Using raw data for schema alignment.", exc_info=True)
            enriched_health_df = raw_health_df.reindex(columns=base_health_cols_schema, fill_value=np.nan)
    else:
        logger.warning(f"({log_ctx}) Raw health records are empty or invalid. AI enrichment skipped.")
        enriched_health_df = pd.DataFrame(columns=base_health_cols_schema)


    # Load Zone Data
    zone_data_full = load_zone_data(source_context=f"{log_ctx}/ZoneData") # Assuming this loader exists
    zone_attributes_df = pd.DataFrame() # Default to empty
    # Define expected SDOH columns for schema consistency
    sdoh_cols = ['zone_id', 'name', 'population', 'socio_economic_index', 
                 'avg_travel_time_clinic_min', 'predominant_hazard_type', 
                 'primary_livelihood', 'water_source_main', 'area_sqkm']

    if isinstance(zone_data_full, pd.DataFrame) and not zone_data_full.empty:
        logger.info(f"({log_ctx}) Full zone data loaded: {zone_data_full.shape[0]} rows. Extracting attributes.")
        # Select only the desired SDOH columns that exist in the loaded data
        cols_to_keep = [col for col in sdoh_cols if col in zone_data_full.columns]
        
        if 'zone_id' not in cols_to_keep and 'zone_id' in zone_data_full.columns: # Ensure zone_id is present if available
            cols_to_keep.append('zone_id')
        
        if cols_to_keep:
            zone_attributes_df = zone_data_full[list(set(cols_to_keep))].copy() # Use set to avoid duplicate zone_id
            # Ensure all expected SDOH columns exist, fill with NaN if not present in source
            for sdoh_col_expected in sdoh_cols:
                if sdoh_col_expected not in zone_attributes_df.columns:
                    zone_attributes_df[sdoh_col_expected] = np.nan
            logger.info(f"({log_ctx}) Processed zone attributes: {zone_attributes_df.shape[0]} zones, {len(zone_attributes_df.columns)} attributes.")
        else:
            logger.warning(f"({log_ctx}) No relevant SDOH columns found in loaded zone data. Using empty DataFrame with SDOH schema.")
            zone_attributes_df = pd.DataFrame(columns=sdoh_cols)
    else:
        logger.warning(f"({log_ctx}) Zone attributes data is unavailable or empty. Using empty DataFrame with SDOH schema.")
        zone_attributes_df = pd.DataFrame(columns=sdoh_cols)

    if enriched_health_df.empty:
        logger.error(f"({log_ctx}) CRITICAL: Enriched health dataset is empty after all processing steps.")
    
    return enriched_health_df, zone_attributes_df

# Attempt to load main datasets
health_df_main, zone_attr_main = pd.DataFrame(), pd.DataFrame() # Initialize
try:
    health_df_main, zone_attr_main = get_population_analytics_datasets()
except Exception as e_main_load:
    logger.error(f"Population Dashboard: Critical dataset loading failed: {e_main_load}", exc_info=True)
    st.error(
        f"🛑 Error loading population analytics data: {str(e_main_load)}. "
        "Dashboard functionality will be severely limited. Check application logs and data sources."
    )

if health_df_main.empty:
    data_dir_path_str = str(Path(settings.DATA_DIR).resolve()) if hasattr(settings, 'DATA_DIR') else "data_sources/"
    health_records_filename = Path(settings.HEALTH_RECORDS_CSV_PATH).name if hasattr(settings, 'HEALTH_RECORDS_CSV_PATH') else "health_records_expanded.csv"
    st.error(
        "🚨 Critical Data Failure: Primary health dataset is empty. Most features will be unavailable. "
        f"Ensure `{health_records_filename}` is in `{data_dir_path_str}` and is valid."
    )
    # Consider st.stop() here if the dashboard is truly unusable without health_df_main

# --- Sidebar Setup ---
st.sidebar.markdown("---") # Visual separator

# Logo Display (Simplified with fixed width)
try:
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_LOGO_SMALL_PATH'):
        project_root = Path(settings.PROJECT_ROOT_DIR)
        logo_path_sidebar = project_root / settings.APP_LOGO_SMALL_PATH
        if logo_path_sidebar.is_file():
            st.sidebar.image(str(logo_path_sidebar.resolve()), width=150) # Fixed width, adjust as needed
        else:
            logger.warning(f"Sidebar logo for Population Dashboard not found at: {logo_path_sidebar.resolve()}")
            st.sidebar.caption("Logo not found.")
    else:
        logger.warning("PROJECT_ROOT_DIR or APP_LOGO_SMALL_PATH missing in settings for sidebar logo.")
        st.sidebar.caption("Logo config error.")
except Exception as e_logo:
    logger.error(f"Unexpected error displaying sidebar logo: {e_logo}", exc_info=True)
    st.sidebar.caption("Error loading logo.")
st.sidebar.markdown("---") # Visual separator

st.sidebar.header("🔎 Analytics Filters")

# Date Range Filter
# Determine overall min/max dates from the loaded health_df_main
abs_min_fallback_date_pop = date.today() - timedelta(days=3*365) # Fallback
abs_max_fallback_date_pop = date.today()    # Fallback

min_data_date_pop, max_data_date_pop = abs_min_fallback_date_pop, abs_max_fallback_date_pop

if isinstance(health_df_main, pd.DataFrame) and 'encounter_date' in health_df_main.columns:
    # Ensure 'encounter_date' is datetime and timezone-naive for min/max calculation
    if not pd.api.types.is_datetime64_any_dtype(health_df_main['encounter_date']):
        health_df_main['encounter_date'] = pd.to_datetime(health_df_main['encounter_date'], errors='coerce')
    if health_df_main['encounter_date'].dt.tz is not None:
            health_df_main['encounter_date'] = health_df_main['encounter_date'].dt.tz_localize(None)
            
    valid_dates_pop = health_df_main['encounter_date'].dropna()
    if not valid_dates_pop.empty:
        min_from_data_pop = valid_dates_pop.min().date()
        max_from_data_pop = valid_dates_pop.max().date()
        if min_from_data_pop <= max_from_data_pop: # Ensure logical range
            min_data_date_pop = min_from_data_pop
            max_data_date_pop = max_from_data_pop
        else: logger.warning("Min date from health data is after max date. Using fallback date range for Population Dashboard.")
    else: logger.info("No valid (non-NaT) encounter dates in health data for Population Dashboard. Using fallback date range.")
else: logger.info("Health data or 'encounter_date' column not available for Population Dashboard. Using fallback date range.")


date_range_ss_key_pop = "pop_dashboard_date_range_v8" # Use a new key if state structure changes
if date_range_ss_key_pop not in st.session_state:
    # Default to full available range or fallback if data is sparse
    st.session_state[date_range_ss_key_pop] = [min_data_date_pop, max_data_date_pop]
else: # Validate persisted state against current data bounds
    persisted_start_pop, persisted_end_pop = st.session_state[date_range_ss_key_pop]
    current_start_pop = min(max(persisted_start_pop, min_data_date_pop), max_data_date_pop)
    current_end_pop = min(max(persisted_end_pop, min_data_date_pop), max_data_date_pop)
    if current_start_pop > current_end_pop: current_start_pop = current_end_pop
    st.session_state[date_range_ss_key_pop] = [current_start_pop, current_end_pop]

selected_date_range_pop = st.sidebar.date_input(
    "Select Date Range for Analysis:",
    value=st.session_state[date_range_ss_key_pop],
    min_value=min_data_date_pop, max_value=max_data_date_pop,
    key=f"{date_range_ss_key_pop}_widget"
)

start_date_filter_pop, end_date_filter_pop = min_data_date_pop, max_data_date_pop # Initialize
if isinstance(selected_date_range_pop, (list, tuple)) and len(selected_date_range_pop) == 2:
    start_date_filter_pop, end_date_filter_pop = selected_date_range_pop
    if start_date_filter_pop > end_date_filter_pop:
        st.sidebar.error("Start date cannot be after end date. Adjusting.")
        end_date_filter_pop = start_date_filter_pop 
    st.session_state[date_range_ss_key_pop] = [start_date_filter_pop, end_date_filter_pop]
else: # Fallback if date_input returns single date or unexpected
    start_date_filter_pop, end_date_filter_pop = st.session_state.get(date_range_ss_key_pop, [min_data_date_pop, max_data_date_pop])
    # st.sidebar.warning("Date range selection error for Population Dashboard. Using previous/default.") # Optional warning


# Condition Filter (from health_df_main)
available_conditions_pop = ["All Conditions"]
if isinstance(health_df_main, pd.DataFrame) and 'condition' in health_df_main.columns:
    unique_conditions_pop = health_df_main['condition'].dropna().astype(str).unique().tolist()
    if unique_conditions_pop: available_conditions_pop.extend(sorted(unique_conditions_pop))
selected_condition_filter_pop = st.sidebar.selectbox("Filter by Condition:", options=available_conditions_pop, index=0, key="pop_cond_filter_v4")

# Zone Filter (from zone_attr_main for names, fallback to health_df_main for IDs)
available_zones_pop = ["All Zones/Regions"]
zone_name_to_id_map_pop = {}
if isinstance(zone_attr_main, pd.DataFrame) and 'name' in zone_attr_main.columns and 'zone_id' in zone_attr_main.columns:
    valid_zones_pop = zone_attr_main.dropna(subset=['name', 'zone_id'])
    if not valid_zones_pop.empty:
        # Handle potential duplicate names by taking the first zone_id for a given name
        zone_name_to_id_map_pop = valid_zones_pop.groupby('name')['zone_id'].first().to_dict()
        available_zones_pop.extend(sorted(valid_zones_pop['name'].astype(str).unique().tolist()))
elif isinstance(health_df_main, pd.DataFrame) and 'zone_id' in health_df_main.columns: # Fallback if no zone names
    logger.info("Zone names not available from zone_attr_main, using zone_ids from health data for filter.")
    available_zones_pop.extend(sorted(health_df_main['zone_id'].dropna().astype(str).unique().tolist()))
selected_zone_filter_display_pop = st.sidebar.selectbox("Filter by Zone/Region:", options=available_zones_pop, index=0, key="pop_zone_filter_v4")


# --- Apply Filters to Data ---
filtered_pop_analytics_df = pd.DataFrame() # Initialize as empty
if isinstance(health_df_main, pd.DataFrame) and not health_df_main.empty:
    temp_pop_df = health_df_main.copy()
    
    # Date Filter (ensure encounter_date is datetime again, as it might be reloaded from cache)
    if 'encounter_date' in temp_pop_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(temp_pop_df['encounter_date']):
            temp_pop_df['encounter_date'] = pd.to_datetime(temp_pop_df['encounter_date'], errors='coerce')
        if temp_pop_df['encounter_date'].dt.tz is not None:
            temp_pop_df['encounter_date'] = temp_pop_df['encounter_date'].dt.tz_localize(None)

        start_dt_pop = pd.to_datetime(start_date_filter_pop) # Convert date to datetime for comparison
        end_dt_pop = pd.to_datetime(end_date_filter_pop)
        
        temp_pop_df = temp_pop_df[
            (temp_pop_df['encounter_date'].notna()) &
            (temp_pop_df['encounter_date'] >= start_dt_pop) &
            (temp_pop_df['encounter_date'] <= end_dt_pop)
        ]
    else: st.warning("⚠️ 'encounter_date' missing from health data. Date filtering not applied to Population Dashboard.")

    # Condition Filter
    if selected_condition_filter_pop != "All Conditions" and 'condition' in temp_pop_df.columns:
        temp_pop_df = temp_pop_df[temp_pop_df['condition'] == selected_condition_filter_pop]

    # Zone Filter
    if selected_zone_filter_display_pop != "All Zones/Regions":
        if zone_name_to_id_map_pop and selected_zone_filter_display_pop in zone_name_to_id_map_pop: # Filtering by name via map
            selected_zone_id_filter_pop = zone_name_to_id_map_pop[selected_zone_filter_display_pop]
            # Ensure zone_id in temp_pop_df is comparable (e.g., string if map keys are string)
            if 'zone_id' in temp_pop_df.columns:
                 temp_pop_df = temp_pop_df[temp_pop_df['zone_id'].astype(str) == str(selected_zone_id_filter_pop)]
        elif 'zone_id' in temp_pop_df.columns: # Fallback to filtering by ID if name wasn't in map or no map exists
            temp_pop_df = temp_pop_df[temp_pop_df['zone_id'].astype(str) == str(selected_zone_filter_display_pop)]
            
    filtered_pop_analytics_df = temp_pop_df
else:
    logger.info("Initial health_df_main for Population Dashboard is empty. No data to filter.")


# --- Main Page Content ---
# Display filter context clearly
filter_context_str_pop = (
    f"({start_date_filter_pop.strftime('%d %b %Y')} - {end_date_filter_pop.strftime('%d %b %Y')}, "
    f"Condition: {selected_condition_filter_pop}, Zone: {selected_zone_filter_display_pop})"
)
st.subheader(f"Population Health Snapshot {filter_context_str_pop}")

if filtered_pop_analytics_df.empty:
    st.info("ℹ️ Insufficient data after filtering to display population summary KPIs or detailed views.")
else:
    # --- KPIs ---
    kpi_cols_pop = st.columns(4) # Adjust number of columns as needed
    with kpi_cols_pop[0]:
        total_encounters_pop = filtered_pop_analytics_df.shape[0]
        st.metric("Total Encounters", f"{total_encounters_pop:,}")
    with kpi_cols_pop[1]:
        unique_patients_pop = filtered_pop_analytics_df['patient_id'].nunique() if 'patient_id' in filtered_pop_analytics_df else 0
        st.metric("Unique Patients", f"{unique_patients_pop:,}")
    with kpi_cols_pop[2]:
        avg_age_pop = filtered_pop_analytics_df['age'].mean() if 'age' in filtered_pop_analytics_df else np.nan
        st.metric("Avg. Patient Age", f"{avg_age_pop:.1f}" if pd.notna(avg_age_pop) else "N/A")
    with kpi_cols_pop[3]:
        avg_risk_pop = filtered_pop_analytics_df['ai_risk_score'].mean() if 'ai_risk_score' in filtered_pop_analytics_df else np.nan
        st.metric("Avg. AI Risk Score", f"{avg_risk_pop:.2f}" if pd.notna(avg_risk_pop) else "N/A")
    st.markdown("---")


# --- Tabbed Interface ---
tab_titles_pop = ["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"]
tabs_pop = st.tabs(tab_titles_pop)

with tabs_pop[0]: # Epi Overview
    st.header(f"Epidemiological Overview {filter_context_str_pop}")
    if filtered_pop_analytics_df.empty: st.info("No data for Epi Overview with current filters.")
    else:
        if 'condition' in filtered_pop_analytics_df.columns:
            top_conditions_pop = filtered_pop_analytics_df['condition'].value_counts().nlargest(10)
            if not top_conditions_pop.empty:
                fig_cond_pop = px.bar(top_conditions_pop, y=top_conditions_pop.index, x=top_conditions_pop.values, 
                                 orientation='h', title="Top 10 Conditions by Encounters", 
                                 labels={'y':'Condition', 'x':'Number of Encounters'})
                fig_cond_pop.update_layout(yaxis={'categoryorder':'total ascending'}) # Show highest bar at top
                st.plotly_chart(fig_cond_pop, use_container_width=True)
            else: st.caption("No condition data to display.")
        
        if 'encounter_date' in filtered_pop_analytics_df.columns:
            # Resample to weekly, ensure index is datetime
            enc_trend_pop_df = filtered_pop_analytics_df.set_index('encounter_date').resample('W-MON').size().reset_index(name='count')
            if not enc_trend_pop_df.empty:
                fig_trend_pop = px.line(enc_trend_pop_df, x='encounter_date', y='count', 
                                   title="Weekly Encounters Trend", markers=True,
                                   labels={'encounter_date': 'Week Starting (Monday)', 'count': 'Number of Encounters'})
                st.plotly_chart(fig_trend_pop, use_container_width=True)
            else: st.caption("No encounter trend data to display.")


with tabs_pop[1]: # Demographics & SDOH
    st.header(f"Demographics & Socio-demographic Health (SDOH) {filter_context_str_pop}")
    if filtered_pop_analytics_df.empty and zone_attr_main.empty:
        st.info("No health or zone attribute data available for Demographics & SDOH analysis.")
    else:
        # Age Distribution (from unique patients in filtered health data)
        if 'age' in filtered_pop_analytics_df.columns and 'patient_id' in filtered_pop_analytics_df.columns:
            unique_patient_ages_pop = filtered_pop_analytics_df.drop_duplicates(subset=['patient_id'])['age'].dropna()
            if not unique_patient_ages_pop.empty:
                fig_age_pop = px.histogram(unique_patient_ages_pop, nbins=20, title="Age Distribution (Unique Patients)")
                st.plotly_chart(fig_age_pop, use_container_width=True)
            else: st.caption("No age data for unique patients to display.")

        # Gender Distribution (from unique patients)
        if 'gender' in filtered_pop_analytics_df.columns and 'patient_id' in filtered_pop_analytics_df.columns:
            unique_patient_genders_pop = filtered_pop_analytics_df.drop_duplicates(subset=['patient_id'])['gender'].value_counts()
            if not unique_patient_genders_pop.empty:
                fig_gender_pop = px.pie(unique_patient_genders_pop, values=unique_patient_genders_pop.values, 
                                   names=unique_patient_genders_pop.index, title="Gender Distribution (Unique Patients)")
                st.plotly_chart(fig_gender_pop, use_container_width=True)
            else: st.caption("No gender data for unique patients to display.")
        
        # SDOH from zone_attr_main (if available and filtered or all zones)
        display_zone_attr_df = zone_attr_main.copy()
        if selected_zone_filter_display_pop != "All Zones/Regions" and not zone_attr_main.empty:
            if zone_name_to_id_map_pop and selected_zone_filter_display_pop in zone_name_to_id_map_pop:
                selected_zone_id = zone_name_to_id_map_pop[selected_zone_filter_display_pop]
                display_zone_attr_df = zone_attr_main[zone_attr_main['zone_id'].astype(str) == str(selected_zone_id)]
            else: # If filtering by ID directly (e.g., if names weren't mapped)
                display_zone_attr_df = zone_attr_main[zone_attr_main['zone_id'].astype(str) == str(selected_zone_filter_display_pop)]
        
        if not display_zone_attr_df.empty:
            st.markdown("---")
            st.subheader("Zone Attributes Overview" + (f" for {selected_zone_filter_display_pop}" if selected_zone_filter_display_pop != "All Zones/Regions" else ""))
            if 'population' in display_zone_attr_df.columns and display_zone_attr_df['population'].notna().any():
                pop_by_zone_display = display_zone_attr_df.dropna(subset=['population', 'name']).sort_values('population', ascending=False).head(15)
                if not pop_by_zone_display.empty:
                    fig_pop_zone = px.bar(pop_by_zone_display, x='name', y='population', title="Population by Zone (Top 15 if 'All Zones')")
                    st.plotly_chart(fig_pop_zone, use_container_width=True)
            
            if 'socio_economic_index' in display_zone_attr_df.columns and display_zone_attr_df['socio_economic_index'].notna().any():
                sei_by_zone_display = display_zone_attr_df.dropna(subset=['socio_economic_index', 'name']).sort_values('socio_economic_index')
                if not sei_by_zone_display.empty:
                    fig_sei_zone = px.bar(sei_by_zone_display, x='name', y='socio_economic_index', title="Socio-Economic Index by Zone (Lower is better)")
                    st.plotly_chart(fig_sei_zone, use_container_width=True)
            
            # Display selected SDOH columns as a table if specific zone is selected
            if selected_zone_filter_display_pop != "All Zones/Regions" and not display_zone_attr_df.empty:
                st.dataframe(display_zone_attr_df.set_index('name')[['population', 'socio_economic_index', 'avg_travel_time_clinic_min', 'predominant_hazard_type', 'primary_livelihood', 'water_source_main']].T, use_container_width=True)
            elif display_zone_attr_df.shape[0] > 15: # If many zones, show a sample
                st.dataframe(display_zone_attr_df[['name', 'population', 'socio_economic_index']].head(15), use_container_width=True)


with tabs_pop[2]: # Clinical Insights
    st.header(f"Clinical Insights {filter_context_str_pop}")
    if filtered_pop_analytics_df.empty: st.info("No data for Clinical Insights with current filters.")
    else:
        if 'ai_risk_score' in filtered_pop_analytics_df.columns:
            risk_scores_pop = filtered_pop_analytics_df['ai_risk_score'].dropna()
            if not risk_scores_pop.empty:
                fig_risk_pop = px.histogram(risk_scores_pop, title="AI Risk Score Distribution (All encounters in period)")
                st.plotly_chart(fig_risk_pop, use_container_width=True)
            else: st.caption("No AI risk score data to display.")

        if 'ai_followup_priority_score' in filtered_pop_analytics_df.columns:
            prio_scores_pop = filtered_pop_analytics_df['ai_followup_priority_score'].dropna()
            if not prio_scores_pop.empty:
                fig_prio_pop = px.histogram(prio_scores_pop, title="AI Follow-up Priority Score Distribution (All encounters)")
                st.plotly_chart(fig_prio_pop, use_container_width=True)
            else: st.caption("No AI follow-up priority score data to display.")
        
        # Further clinical insights could be added here, e.g., outcomes if available


with tabs_pop[3]: # Systems & Equity
    st.header(f"Systems & Equity Insights {filter_context_str_pop}")
    if filtered_pop_analytics_df.empty and zone_attr_main.empty:
        st.info("No health or zone attribute data available for Systems & Equity analysis.")
    else:
        # Encounter distribution by zone
        if 'zone_id' in filtered_pop_analytics_df.columns:
            encounters_by_zone_pop = filtered_pop_analytics_df['zone_id'].value_counts().nlargest(20)
            if not encounters_by_zone_pop.empty:
                # Try to map zone_id to name for better display
                if not zone_attr_main.empty and 'zone_id' in zone_attr_main and 'name' in zone_attr_main:
                    zone_id_name_map = zone_attr_main.drop_duplicates(subset=['zone_id']).set_index('zone_id')['name']
                    encounters_by_zone_pop.index = encounters_by_zone_pop.index.map(lambda x: f"{zone_id_name_map.get(x, x)} ({x})")

                fig_enc_zone_pop = px.bar(encounters_by_zone_pop, y=encounters_by_zone_pop.index, x=encounters_by_zone_pop.values,
                                     orientation='h', title="Encounter Distribution by Zone (Top 20)",
                                     labels={'y':'Zone', 'x':'Number of Encounters'})
                fig_enc_zone_pop.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_enc_zone_pop, use_container_width=True)
            else: st.caption("No encounter data by zone to display.")

        # Access equity (e.g., avg travel time vs. utilization - requires merging health and zone data)
        if not filtered_pop_analytics_df.empty and not zone_attr_main.empty and \
           'zone_id' in filtered_pop_analytics_df and 'zone_id' in zone_attr_main and \
           'avg_travel_time_clinic_min' in zone_attr_main and 'population' in zone_attr_main:
            
            try:
                # Ensure zone_id types match for merging
                df1 = filtered_pop_analytics_df[['zone_id', 'patient_id']].copy()
                df1['zone_id'] = df1['zone_id'].astype(str)
                df2 = zone_attr_main[['zone_id', 'name', 'avg_travel_time_clinic_min', 'population']].copy()
                df2['zone_id'] = df2['zone_id'].astype(str)

                merged_equity_df = pd.merge(df1, df2, on='zone_id', how='left')
                
                if not merged_equity_df.empty and 'avg_travel_time_clinic_min' in merged_equity_df:
                    # Calculate utilization per capita (encounters / population)
                    zone_utilization = merged_equity_df.groupby('zone_id').agg(
                        name=('name', 'first'),
                        encounters=('patient_id', 'size'), # Using size of encounters in the period
                        avg_travel_time=('avg_travel_time_clinic_min', 'mean'),
                        population=('population', 'first') # Assuming population is consistent per zone_id
                    ).reset_index()
                    zone_utilization = zone_utilization.dropna(subset=['population', 'avg_travel_time', 'encounters'])
                    zone_utilization = zone_utilization[zone_utilization['population'] > 0] # Avoid division by zero
                    if not zone_utilization.empty:
                        zone_utilization['utilization_per_1000_pop'] = (zone_utilization['encounters'] / zone_utilization['population']) * 1000
                        
                        fig_equity_pop = px.scatter(zone_utilization.dropna(subset=['utilization_per_1000_pop']), 
                                               x='avg_travel_time', y='utilization_per_1000_pop',
                                               size='population', hover_name='name',
                                               title="Service Utilization vs. Avg Travel Time by Zone",
                                               labels={'avg_travel_time': 'Avg. Travel Time to Clinic (min)', 
                                                       'utilization_per_1000_pop': 'Encounters per 1,000 Population'})
                        st.plotly_chart(fig_equity_pop, use_container_width=True)
                    else: st.caption("Insufficient data to plot utilization vs. travel time.")
            except Exception as e_equity_plot:
                logger.error(f"Error generating equity plot: {e_equity_plot}", exc_info=True)
                st.caption("Could not generate service utilization vs. travel time plot.")


# --- Footer ---
st.divider()
footer_text = settings.APP_FOOTER_TEXT if hasattr(settings, 'APP_FOOTER_TEXT') else "Sentinel Health Co-Pilot."
st.caption(footer_text)

logger.info(
    f"Population Health Analytics Console fully rendered. Period: {filter_context_str_pop}. "
    f"FilteredDataRows: {filtered_pop_analytics_df.shape[0] if isinstance(filtered_pop_analytics_df, pd.DataFrame) else 0}."
)
