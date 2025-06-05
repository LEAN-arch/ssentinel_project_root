# sentinel_project_root/pages/population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
import html # For escaping HTML in custom markdown
import plotly.express as px 
from pathlib import Path 
import re # Added import for regular expressions

try:
    from config import settings
    from data_processing.loaders import load_health_records, load_zone_data
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe, convert_to_numeric
    from visualization.plots import plot_bar_chart, create_empty_figure, plot_annotated_line_chart
    from visualization.ui_elements import display_custom_styled_kpi_box
except ImportError as e_pop_dash_final:
    import sys
    _current_file_pop_final = Path(__file__).resolve()
    _project_root_pop_assumption_final = _current_file_pop_final.parent.parent
    error_msg_pop_detail_final = (
        f"Population Dashboard Import Error: {e_pop_dash_final}. "
        f"Ensure project root ('{_project_root_pop_assumption_final}') is in sys.path and modules are correct. "
        f"Path: {sys.path}"
    )
    try: st.error(error_msg_pop_detail_final); st.stop()
    except NameError: print(error_msg_pop_detail_final, file=sys.stderr); raise

logger = logging.getLogger(__name__)

st.title(f"📊 {settings.APP_NAME} - Population Health Analytics & Research Console")
st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.divider()

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe}, show_spinner="Loading population analytics dataset...")
def get_population_analytics_datasets(log_ctx: str = "PopAnalyticsConsole/LoadData") -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"({log_ctx}) Loading population health records and zone attributes.")
    raw_health_df = load_health_records(source_context=f"{log_ctx}/HealthRecs")
    enriched_health_df: pd.DataFrame
    base_health_cols_schema = ['patient_id', 'encounter_date', 'condition', 'age', 'gender', 'zone_id', 'ai_risk_score', 'ai_followup_priority_score']
    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        enriched_health_df, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrich")
    else:
        logger.warning(f"({log_ctx}) Raw health records empty/invalid. AI enrichment skipped.")
        enriched_health_df = pd.DataFrame(columns=base_health_cols_schema)

    zone_data_full = load_zone_data(source_context=f"{log_ctx}/ZoneData")
    zone_attributes_df: pd.DataFrame
    sdoh_cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min', 'predominant_hazard_type', 'primary_livelihood', 'water_source_main', 'area_sqkm']
    if isinstance(zone_data_full, pd.DataFrame) and not zone_data_full.empty:
        cols_keep = [col for col in sdoh_cols if col in zone_data_full.columns]
        if 'zone_id' not in cols_keep and 'zone_id' in zone_data_full.columns: cols_keep.append('zone_id')
        if cols_keep:
            zone_attributes_df = zone_data_full[list(set(cols_keep))].copy()
            for sdoh_col in sdoh_cols:
                if sdoh_col not in zone_attributes_df.columns: zone_attributes_df[sdoh_col] = np.nan
            logger.info(f"({log_ctx}) Loaded {len(zone_attributes_df)} zone attributes.")
        else: zone_attributes_df = pd.DataFrame(columns=sdoh_cols); logger.warning(f"({log_ctx}) No SDOH columns in zone data.")
    else: zone_attributes_df = pd.DataFrame(columns=sdoh_cols); logger.warning(f"({log_ctx}) Zone attributes data unavailable.")
    if enriched_health_df.empty: logger.error(f"({log_ctx}) CRITICAL: Health data empty after processing.")
    return enriched_health_df, zone_attributes_df

# --- Load Datasets ---
health_df_pop, zone_attr_pop = pd.DataFrame(), pd.DataFrame()
try: health_df_pop, zone_attr_pop = get_population_analytics_datasets()
except Exception as e_pop_load: 
    logger.error(f"Population Dashboard: Dataset loading failed: {e_pop_load}", exc_info=True)
    st.error(f"Error loading population analytics data: {str(e_pop_load)}. Dashboard functionality will be severely limited. Please check console logs and ensure data files (e.g., health_records_expanded.csv) are correctly placed and accessible.")
# Do not st.stop() here, allow UI to render with warnings if possible.

if health_df_pop.empty: 
    st.error("🚨 Critical Data Failure: Primary health dataset for population analytics is empty. Most console features will be unavailable. Ensure `health_records_expanded.csv` is in `data_sources/` and is not empty.")

# --- Sidebar Filters ---
project_root = Path(settings.PROJECT_ROOT_DIR)
logo_path_pop_sidebar_final = project_root / settings.APP_LOGO_SMALL_PATH # Corrected path construction
if logo_path_pop_sidebar_final.exists() and logo_path_pop_sidebar_final.is_file(): st.sidebar.image(str(logo_path_pop_sidebar_final), width=120)
else: logger.warning(f"Sidebar logo not found: {logo_path_pop_sidebar_final}")
st.sidebar.header("🔎 Analytics Filters")

min_date_pop_filt_default, max_date_pop_filt_default = date.today() - timedelta(days=365*3), date.today()
if isinstance(health_df_pop, pd.DataFrame) and 'encounter_date' in health_df_pop.columns and health_df_pop['encounter_date'].notna().any():
    # Ensure 'encounter_date' is datetime before min/max
    if not pd.api.types.is_datetime64_any_dtype(health_df_pop['encounter_date']): 
        health_df_pop['encounter_date'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce')
    if health_df_pop['encounter_date'].notna().any(): # Check again after coercion
        min_date_pop_filt_default = health_df_pop['encounter_date'].min().date()
        max_date_pop_filt_default = health_df_pop['encounter_date'].max().date()
if min_date_pop_filt_default > max_date_pop_filt_default: min_date_pop_filt_default = max_date_pop_filt_default 

pop_date_key_final = "pop_dashboard_date_range_v3" # Ensure unique session state key
if pop_date_key_final not in st.session_state: st.session_state[pop_date_key_final] = [min_date_pop_filt_default, max_date_pop_filt_default]
selected_date_range_pop_ui_val = st.sidebar.date_input("Select Date Range for Analysis:", value=st.session_state[pop_date_key_final], min_value=min_date_pop_filt_default, max_value=max_date_pop_filt_default, key=f"{pop_date_key_final}_widget")
start_date_pop_filt_final, end_date_pop_filt_final = selected_date_range_pop_ui_val if isinstance(selected_date_range_pop_ui_val, (list,tuple)) and len(selected_date_range_pop_ui_val)==2 else st.session_state[pop_date_key_final]
if start_date_pop_filt_final > end_date_pop_filt_final: st.sidebar.error("Start date must be <= end date."); end_date_pop_filt_final = start_date_pop_filt_final
st.session_state[pop_date_key_final] = [start_date_pop_filt_final, end_date_pop_filt_final]

analytics_df_pop_display_final: pd.DataFrame = pd.DataFrame() # Initialize
if isinstance(health_df_pop, pd.DataFrame) and 'encounter_date' in health_df_pop.columns:
    # Ensure 'encounter_date' is datetime before filtering
    if not pd.api.types.is_datetime64_any_dtype(health_df_pop['encounter_date']):
        health_df_pop['encounter_date'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce')
    analytics_df_pop_display_final = health_df_pop[(health_df_pop['encounter_date'].notna()) & (health_df_pop['encounter_date'].dt.date >= start_date_pop_filt_final) & (health_df_pop['encounter_date'].dt.date <= end_date_pop_filt_final)].copy()
elif isinstance(health_df_pop, pd.DataFrame): # Health_df exists but no encounter_date
    logger.error("'encounter_date' column missing from health_df_pop. Cannot filter by date.")
    st.error("Data Error: 'encounter_date' column is missing from the health dataset. Date filtering disabled.")
    analytics_df_pop_display_final = health_df_pop.copy() # Use unfiltered data if no date column

selected_cond_pop_ui_val = "All Conditions (Aggregated)"
if isinstance(analytics_df_pop_display_final, pd.DataFrame) and 'condition' in analytics_df_pop_display_final.columns and analytics_df_pop_display_final['condition'].notna().any():
    unique_conds_pop_val = ["All Conditions (Aggregated)"] + sorted(analytics_df_pop_display_final['condition'].dropna().unique().tolist())
    pop_cond_key_final = "pop_dashboard_condition_v3"
    if pop_cond_key_final not in st.session_state or st.session_state[pop_cond_key_final] not in unique_conds_pop_val: st.session_state[pop_cond_key_final] = unique_conds_pop_val[0]
    selected_cond_pop_ui_val = st.sidebar.selectbox("Filter by Condition:", options=unique_conds_pop_val, key=f"{pop_cond_key_final}_widget", index=unique_conds_pop_val.index(st.session_state[pop_cond_key_final]))
    st.session_state[pop_cond_key_final] = selected_cond_pop_ui_val
    if selected_cond_pop_ui_val != "All Conditions (Aggregated)": analytics_df_pop_display_final = analytics_df_pop_display_final[analytics_df_pop_display_final['condition'] == selected_cond_pop_ui_val]
else: st.sidebar.caption("Condition filter unavailable (no 'condition' data).")

selected_zone_pop_ui_val = "All Zones (Aggregated)"; zone_map_pop_ui_val: Dict[str,str] = {}; zone_opts_pop_ui_val = ["All Zones (Aggregated)"]
if isinstance(zone_attr_pop, pd.DataFrame) and not zone_attr_pop.empty and 'zone_id' in zone_attr_pop.columns:
    temp_opts_pop_val = []
    for _, z_row_val in zone_attr_pop.iterrows():
        z_id_val = str(z_row_val['zone_id']).strip()
        z_name_val = str(z_row_val.get('name', z_id_val)).strip() # Use name if available, else ID
        disp_opt_val = f"{z_name_val} ({z_id_val})" if z_name_val and z_name_val != z_id_val and z_name_val.lower() != "unknown" else z_id_val
        if disp_opt_val not in zone_map_pop_ui_val: # Ensure display option is unique before adding
            temp_opts_pop_val.append(disp_opt_val)
            zone_map_pop_ui_val[disp_opt_val] = z_id_val # Map display string to actual zone_id
    if temp_opts_pop_val: zone_opts_pop_ui_val.extend(sorted(list(set(temp_opts_pop_val))))
else: st.sidebar.caption("Zone filter options limited (zone attributes data missing).")

if len(zone_opts_pop_ui_val) > 1:
    pop_zone_key_final = "pop_dashboard_zone_v3"
    if pop_zone_key_final not in st.session_state or st.session_state[pop_zone_key_final] not in zone_opts_pop_ui_val: st.session_state[pop_zone_key_final] = zone_opts_pop_ui_val[0]
    selected_zone_pop_ui_val = st.sidebar.selectbox("Filter by Zone:", options=zone_opts_pop_ui_val, key=f"{pop_zone_key_final}_widget", index=zone_opts_pop_ui_val.index(st.session_state[pop_zone_key_final]))
    st.session_state[pop_zone_key_final] = selected_zone_pop_ui_val
    if selected_zone_pop_ui_val != "All Zones (Aggregated)" and 'zone_id' in analytics_df_pop_display_final.columns:
        zone_id_to_filter = zone_map_pop_ui_val.get(selected_zone_pop_ui_val, selected_zone_pop_ui_val) # Get actual ID from map
        analytics_df_pop_display_final = analytics_df_pop_display_final[analytics_df_pop_display_final['zone_id'] == zone_id_to_filter]
    elif selected_zone_pop_ui_val != "All Zones (Aggregated)": st.sidebar.caption("'zone_id' missing in health data for zone filtering.")

if analytics_df_pop_display_final.empty and (start_date_pop_filt_final != min_date_pop_filt_default or end_date_pop_filt_final != max_date_pop_filt_default or selected_cond_pop_ui_val != "All Conditions (Aggregated)" or selected_zone_pop_ui_val != "All Zones (Aggregated)"):
    st.warning("No health data available for the selected filters. Please broaden your filter criteria or check data sources.")

# --- Population Health Snapshot KPIs ---
st.subheader(f"Population Health Snapshot ({start_date_pop_filt_final.strftime('%d %b %Y')} - {end_date_pop_filt_final.strftime('%d %b %Y')}, Cond: {selected_cond_pop_ui_val}, Zone: {selected_zone_pop_ui_val})")
if analytics_df_pop_display_final.empty: st.info("Insufficient data after filtering to display population summary KPIs.")
else:
    cols_kpi_pop_sum_final = st.columns(4)
    total_unique_pats_final = analytics_df_pop_display_final['patient_id'].nunique() if 'patient_id' in analytics_df_pop_display_final.columns else 0
    mean_risk_final = analytics_df_pop_display_final['ai_risk_score'].mean() if 'ai_risk_score' in analytics_df_pop_display_final.columns and analytics_df_pop_display_final['ai_risk_score'].notna().any() else np.nan
    high_risk_count_final = 0; high_risk_perc_final = 0.0
    if 'ai_risk_score' in analytics_df_pop_display_final.columns and total_unique_pats_final > 0:
        high_risk_df_final = analytics_df_pop_display_final[convert_to_numeric(analytics_df_pop_display_final['ai_risk_score'], default_value=-1) >= settings.RISK_SCORE_HIGH_THRESHOLD]
        high_risk_count_final = high_risk_df_final['patient_id'].nunique() if 'patient_id' in high_risk_df_final.columns else 0
        high_risk_perc_final = (high_risk_count_final / total_unique_pats_final) * 100
    top_cond_final, top_cond_enc_final = "N/A", 0
    if 'condition' in analytics_df_pop_display_final.columns and analytics_df_pop_display_final['condition'].notna().any():
        cond_counts_final = analytics_df_pop_display_final['condition'].value_counts()
        if not cond_counts_final.empty: top_cond_final, top_cond_enc_final = cond_counts_final.idxmax(), cond_counts_final.max()

    with cols_kpi_pop_sum_final[0]: display_custom_styled_kpi_box("Total Unique Patients", total_unique_pats_final)
    with cols_kpi_pop_sum_final[1]: display_custom_styled_kpi_box("Avg. AI Risk Score", f"{mean_risk_final:.1f}" if pd.notna(mean_risk_final) else "N/A")
    with cols_kpi_pop_sum_final[2]: display_custom_styled_kpi_box("% High AI Risk Patients", f"{high_risk_perc_final:.1f}%", f"({high_risk_count_final:,} patients)")
    with cols_kpi_pop_sum_final[3]: display_custom_styled_kpi_box("Top Condition (Encounters)", html.escape(str(top_cond_final)), f"{top_cond_enc_final:,} encounters", settings.COLOR_RISK_MODERATE)

# --- Tabs for Detailed Population Analytics ---
pop_tab_titles_final = ["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"]
tab_epi_pop_final, tab_demog_sdoh_pop_final, tab_clinical_pop_final, tab_systems_pop_final = st.tabs(pop_tab_titles_final)

with tab_epi_pop_final:
    st.header(f"Epidemiological Overview (Filters: {selected_cond_pop_ui_val} | {selected_zone_pop_ui_val})")
    if analytics_df_pop_display_final.empty: st.info("No data for Epi Overview with current filters.")
    else:
        if 'condition' in analytics_df_pop_display_final.columns and 'patient_id' in analytics_df_pop_display_final.columns:
            cond_unique_pat_counts_df_final = analytics_df_pop_display_final.groupby('condition')['patient_id'].nunique().nlargest(12).reset_index(name='unique_patients')
            if not cond_unique_pat_counts_df_final.empty:
                st.plotly_chart(plot_bar_chart(cond_unique_pat_counts_df_final, 'condition', 'unique_patients', "Top Conditions by Unique Patient Count", orientation_bar='h', y_values_are_counts_flag=True, chart_height=450, x_axis_label_text="Unique Patient Count", y_axis_label_text="Condition"), use_container_width=True)
        if 'ai_risk_score' in analytics_df_pop_display_final.columns and analytics_df_pop_display_final['ai_risk_score'].notna().any():
            fig_risk_dist_pop_final = px.histogram(analytics_df_pop_display_final.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=25, title="Patient AI Risk Score Distribution", labels={'ai_risk_score': 'AI Risk Score', 'count': 'Records'})
            fig_risk_dist_pop_final.update_layout(bargap=0.1, height=settings.WEB_PLOT_COMPACT_HEIGHT, title_x=0.05)
            st.plotly_chart(fig_risk_dist_pop_final, use_container_width=True)
        st.caption("Note: True incidence/prevalence trends require careful case definitions and population denominators.")

with tab_demog_sdoh_pop_final:
    st.header("Demographics & Social Determinants of Health (SDOH) Context")
    if analytics_df_pop_display_final.empty: st.info("No data for Demographics & SDOH analysis with current filters.")
    else:
        if 'age' in analytics_df_pop_display_final.columns and analytics_df_pop_display_final['age'].notna().any():
            age_df_pop_final = analytics_df_pop_display_final.dropna(subset=['age'])
            if not age_df_pop_final.empty:
                age_bins_final = [0, 5, 18, 35, 50, 65, np.inf]
                age_labels_final = ['0-4', '5-17', '18-34', '35-49', '50-64', '65+']
                age_df_pop_final['age_group'] = pd.cut(convert_to_numeric(age_df_pop_final['age'], default_value=np.nan), bins=age_bins_final, labels=age_labels_final, right=False)
                age_group_counts_final = age_df_pop_final['age_group'].value_counts().sort_index().reset_index()
                age_group_counts_final.columns = ['Age Group', 'Count']
                st.plotly_chart(plot_bar_chart(age_group_counts_final, 'Age Group', 'Count', "Patient Age Distribution", y_values_are_counts_flag=True), use_container_width=True)
        if not zone_attr_pop.empty and 'zone_id' in analytics_df_pop_display_final.columns and 'zone_id' in zone_attr_pop.columns:
            merged_sdoh_df_final = pd.merge(analytics_df_pop_display_final, zone_attr_pop, on='zone_id', how='left', suffixes=('_health', '_zone'))
            if 'socio_economic_index' in merged_sdoh_df_final.columns and 'ai_risk_score' in merged_sdoh_df_final.columns and \
               merged_sdoh_df_final['socio_economic_index'].notna().any() and merged_sdoh_df_final['ai_risk_score'].notna().any():
                st.markdown("###### AI Risk Score vs. Socio-Economic Index (Zone Level)")
                fig_sdoh_risk_final = px.scatter(merged_sdoh_df_final.dropna(subset=['socio_economic_index', 'ai_risk_score']), 
                                           x='socio_economic_index', y='ai_risk_score', trendline="ols", 
                                           hover_name=merged_sdoh_df_final.get('name_zone', merged_sdoh_df_final.get('zone_id')), # Prefer zone name
                                           labels={'socio_economic_index': 'Socio-Economic Index (Higher is better)', 'ai_risk_score': 'Avg. AI Risk Score'})
                st.plotly_chart(fig_sdoh_risk_final, use_container_width=True)
        if zone_attr_pop.empty: st.caption("Zone attribute data (for SDOH analysis) unavailable.")

with tab_clinical_pop_final:
    st.header("Clinical Insights & Diagnostic Patterns")
    if analytics_df_pop_display_final.empty: st.info("No data for Clinical Insights with current filters.")
    else: st.markdown("_(Placeholder: Analyses on Top Symptoms, other Test Result Distributions, Test Positivity Trends.)_")

with tab_systems_pop_final:
    st.header("Health Systems Performance & Equity Lens")
    if analytics_df_pop_display_final.empty: st.info("No data for Health Systems & Equity analysis with current filters.")
    else: st.markdown("_(Placeholder: Analyses on Encounters by Clinic/Zone, Referral Completion, AI Risk Variations by SDOH factors.)_")

st.divider()
st.caption(settings.APP_FOOTER_TEXT)
logger.info(f"Population Health Analytics Console loaded. Filters: Period=({start_date_pop_filt_final.isoformat()} to {end_date_pop_filt_final.isoformat()}), Cond='{selected_cond_pop_ui_val}', Zone='{selected_zone_pop_ui_val}'")
