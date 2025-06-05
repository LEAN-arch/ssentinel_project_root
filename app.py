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
except ImportError as e_pop_dash_final_fix_again: # Unique exception variable name
    import sys
    _current_file_pop_final_fix_again = Path(__file__).resolve()
    _project_root_pop_assumption_final_fix_again = _current_file_pop_final_fix_again.parent.parent
    error_msg_pop_detail_final_fix_again = (
        f"Population Dashboard Import Error: {e_pop_dash_final_fix_again}. "
        f"Ensure project root ('{_project_root_pop_assumption_final_fix_again}') is in sys.path and modules are correct. "
        f"Path: {sys.path}"
    )
    try: st.error(error_msg_pop_detail_final_fix_again); st.stop()
    except NameError: print(error_msg_pop_detail_final_fix_again, file=sys.stderr); raise

logger = logging.getLogger(__name__)

st.title(f"📊 {settings.APP_NAME} - Population Health Analytics & Research Console")
st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.divider()

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe}, show_spinner="Loading population analytics dataset...")
def get_population_analytics_datasets_pop_fix_again(log_ctx: str = "PopAnalyticsConsole/LoadData") -> tuple[pd.DataFrame, pd.DataFrame]: # Renamed func
    logger.info(f"({log_ctx}) Loading population health records and zone attributes.")
    raw_health_df_pop_fix_again = load_health_records(source_context=f"{log_ctx}/HealthRecs")
    enriched_health_df_pop_fix_again: pd.DataFrame
    base_health_cols_schema_pop_fix_again = ['patient_id', 'encounter_date', 'condition', 'age', 'gender', 'zone_id', 'ai_risk_score', 'ai_followup_priority_score']
    if isinstance(raw_health_df_pop_fix_again, pd.DataFrame) and not raw_health_df_pop_fix_again.empty:
        enriched_health_df_pop_fix_again, _ = apply_ai_models(raw_health_df_pop_fix_again.copy(), source_context=f"{log_ctx}/AIEnrich")
    else:
        logger.warning(f"({log_ctx}) Raw health records empty/invalid. AI enrichment skipped.")
        enriched_health_df_pop_fix_again = pd.DataFrame(columns=base_health_cols_schema_pop_fix_again)

    zone_data_full_pop_fix_again = load_zone_data(source_context=f"{log_ctx}/ZoneData")
    zone_attributes_df_pop_fix_again: pd.DataFrame
    sdoh_cols_pop_fix_again = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min', 'predominant_hazard_type', 'primary_livelihood', 'water_source_main', 'area_sqkm']
    if isinstance(zone_data_full_pop_fix_again, pd.DataFrame) and not zone_data_full_pop_fix_again.empty:
        cols_keep_pop_fix_again = [col_pop_again for col_pop_again in sdoh_cols_pop_fix_again if col_pop_again in zone_data_full_pop_fix_again.columns]
        if 'zone_id' not in cols_keep_pop_fix_again and 'zone_id' in zone_data_full_pop_fix_again.columns: cols_keep_pop_fix_again.append('zone_id')
        if cols_keep_pop_fix_again:
            zone_attributes_df_pop_fix_again = zone_data_full_pop_fix_again[list(set(cols_keep_pop_fix_again))].copy()
            for sdoh_col_pop_fix_again in sdoh_cols_pop_fix_again:
                if sdoh_col_pop_fix_again not in zone_attributes_df_pop_fix_again.columns: zone_attributes_df_pop_fix_again[sdoh_col_pop_fix_again] = np.nan
            logger.info(f"({log_ctx}) Loaded {len(zone_attributes_df_pop_fix_again)} zone attributes.")
        else: zone_attributes_df_pop_fix_again = pd.DataFrame(columns=sdoh_cols_pop_fix_again); logger.warning(f"({log_ctx}) No SDOH columns in zone data.")
    else: zone_attributes_df_pop_fix_again = pd.DataFrame(columns=sdoh_cols_pop_fix_again); logger.warning(f"({log_ctx}) Zone attributes data unavailable.")
    if enriched_health_df_pop_fix_again.empty: logger.error(f"({log_ctx}) CRITICAL: Health data empty after processing.")
    return enriched_health_df_pop_fix_again, zone_attributes_df_pop_fix_again

# --- Load Datasets ---
health_df_pop_main_fix_again, zone_attr_pop_main_fix_again = pd.DataFrame(), pd.DataFrame() 
try: health_df_pop_main_fix_again, zone_attr_pop_main_fix_again = get_population_analytics_datasets_pop_fix_again()
except Exception as e_pop_load_main_fix_again: 
    logger.error(f"Population Dashboard: Dataset loading failed: {e_pop_load_main_fix_again}", exc_info=True)
    st.error(f"Error loading population analytics data: {str(e_pop_load_main_fix_again)}. Dashboard functionality will be severely limited. Please check console logs and ensure data files (e.g., health_records_expanded.csv) are correctly placed and accessible.")
if health_df_pop_main_fix_again.empty: 
    st.error("🚨 Critical Data Failure: Primary health dataset for population analytics is empty. Most console features will be unavailable. Ensure `health_records_expanded.csv` is in `data_sources/` and is not empty.")

# --- Sidebar Filters ---
project_root_pop_fix_again = Path(settings.PROJECT_ROOT_DIR) 
logo_path_pop_sidebar_final_fix_again = project_root_pop_fix_again / settings.APP_LOGO_SMALL_PATH 
if logo_path_pop_sidebar_final_fix_again.exists() and logo_path_pop_sidebar_final_fix_again.is_file(): st.sidebar.image(str(logo_path_pop_sidebar_final_fix_again), width=120)
else: logger.warning(f"Sidebar logo not found for Population Dashboard: {logo_path_pop_sidebar_final_fix_again}")
st.sidebar.header("🔎 Analytics Filters")

min_date_pop_final_fix_again, max_date_pop_final_fix_again = date.today() - timedelta(days=365*3), date.today()
if isinstance(health_df_pop_main_fix_again, pd.DataFrame) and 'encounter_date' in health_df_pop_main_fix_again.columns and health_df_pop_main_fix_again['encounter_date'].notna().any():
    if not pd.api.types.is_datetime64_any_dtype(health_df_pop_main_fix_again['encounter_date']): health_df_pop_main_fix_again['encounter_date'] = pd.to_datetime(health_df_pop_main_fix_again['encounter_date'], errors='coerce')
    if health_df_pop_main_fix_again['encounter_date'].notna().any():
        min_date_pop_final_fix_again, max_date_pop_final_fix_again = health_df_pop_main_fix_again['encounter_date'].min().date(), health_df_pop_main_fix_again['encounter_date'].max().date()
if min_date_pop_final_fix_again > max_date_pop_final_fix_again: min_date_pop_final_fix_again = max_date_pop_final_fix_again 

pop_date_key_final_ss_fix_again = "pop_dashboard_date_range_v4" # Ensure unique key
if pop_date_key_final_ss_fix_again not in st.session_state: st.session_state[pop_date_key_final_ss_fix_again] = [min_date_pop_final_fix_again, max_date_pop_final_fix_again]
selected_date_range_pop_ui_val_final_fix_again = st.sidebar.date_input("Select Date Range for Analysis:", value=st.session_state[pop_date_key_final_ss_fix_again], min_value=min_date_pop_final_fix_again, max_value=max_date_pop_final_fix_again, key=f"{pop_date_key_final_ss_fix_again}_widget")
start_date_pop_filt_final_ui_again, end_date_pop_filt_final_ui_again = selected_date_range_pop_ui_val_final_fix_again if isinstance(selected_date_range_pop_ui_val_final_fix_again, (list,tuple)) and len(selected_date_range_pop_ui_val_final_fix_again)==2 else st.session_state[pop_date_key_final_ss_fix_again]
if start_date_pop_filt_final_ui_again > end_date_pop_filt_final_ui_again: st.sidebar.error("Start date <= end date."); end_date_pop_filt_final_ui_again = start_date_pop_filt_final_ui_again
st.session_state[pop_date_key_final_ss_fix_again] = [start_date_pop_filt_final_ui_again, end_date_pop_filt_final_ui_again]

analytics_df_pop_display_final_df_again: pd.DataFrame = pd.DataFrame() 
if isinstance(health_df_pop_main_fix_again, pd.DataFrame) and 'encounter_date' in health_df_pop_main_fix_again.columns:
    if not pd.api.types.is_datetime64_any_dtype(health_df_pop_main_fix_again['encounter_date']): health_df_pop_main_fix_again['encounter_date'] = pd.to_datetime(health_df_pop_main_fix_again['encounter_date'], errors='coerce')
    analytics_df_pop_display_final_df_again = health_df_pop_main_fix_again[(health_df_pop_main_fix_again['encounter_date'].notna()) & (health_df_pop_main_fix_again['encounter_date'].dt.date >= start_date_pop_filt_final_ui_again) & (health_df_pop_main_fix_again['encounter_date'].dt.date <= end_date_pop_filt_final_ui_again)].copy()
elif isinstance(health_df_pop_main_fix_again, pd.DataFrame): 
    logger.error("'encounter_date' column missing from health_df_pop_main_fix_again. Date filtering disabled.")
    st.error("Data Error: 'encounter_date' missing. Date filtering disabled.")
    analytics_df_pop_display_final_df_again = health_df_pop_main_fix_again.copy()

selected_cond_pop_ui_final_again = "All Conditions (Aggregated)"
if isinstance(analytics_df_pop_display_final_df_again, pd.DataFrame) and 'condition' in analytics_df_pop_display_final_df_again.columns and analytics_df_pop_display_final_df_again['condition'].notna().any():
    unique_conds_pop_final_again = ["All Conditions (Aggregated)"] + sorted(analytics_df_pop_display_final_df_again['condition'].dropna().unique().tolist())
    pop_cond_key_final_ss_again = "pop_dashboard_condition_v4"
    if pop_cond_key_final_ss_again not in st.session_state or st.session_state[pop_cond_key_final_ss_again] not in unique_conds_pop_final_again: st.session_state[pop_cond_key_final_ss_again] = unique_conds_pop_final_again[0]
    selected_cond_pop_ui_final_again = st.sidebar.selectbox("Filter by Condition:", options=unique_conds_pop_final_again, key=f"{pop_cond_key_final_ss_again}_widget", index=unique_conds_pop_final_again.index(st.session_state[pop_cond_key_final_ss_again]))
    st.session_state[pop_cond_key_final_ss_again] = selected_cond_pop_ui_final_again
    if selected_cond_pop_ui_final_again != "All Conditions (Aggregated)": analytics_df_pop_display_final_df_again = analytics_df_pop_display_final_df_again[analytics_df_pop_display_final_df_again['condition'] == selected_cond_pop_ui_final_again]
else: st.sidebar.caption("Condition filter unavailable (no 'condition' data).")

selected_zone_pop_ui_final_again = "All Zones (Aggregated)"; zone_map_pop_ui_final_again: Dict[str,str] = {}; zone_opts_pop_ui_final_again = ["All Zones (Aggregated)"]
if isinstance(zone_attr_pop_main_fix_again, pd.DataFrame) and not zone_attr_pop_main_fix_again.empty and 'zone_id' in zone_attr_pop_main_fix_again.columns:
    temp_opts_pop_final_again = []
    for _, z_row_final_again in zone_attr_pop_main_fix_again.iterrows():
        z_id_final_again = str(z_row_final_again['zone_id']).strip()
        z_name_final_again = str(z_row_final_again.get('name', z_id_final_again)).strip()
        disp_opt_final_again = f"{z_name_final_again} ({z_id_final_again})" if z_name_final_again and z_name_final_again != z_id_final_again and z_name_final_again.lower() != "unknown" else z_id_final_again
        if disp_opt_final_again not in zone_map_pop_ui_final_again: temp_opts_pop_final_again.append(disp_opt_final_again); zone_map_pop_ui_final_again[disp_opt_final_again] = z_id_final_again
    if temp_opts_pop_final_again: zone_opts_pop_ui_final_again.extend(sorted(list(set(temp_opts_pop_final_again))))
else: st.sidebar.caption("Zone filter options limited (zone attributes data missing).")
if len(zone_opts_pop_ui_final_again) > 1:
    pop_zone_key_final_ss_again = "pop_dashboard_zone_v4"
    if pop_zone_key_final_ss_again not in st.session_state or st.session_state[pop_zone_key_final_ss_again] not in zone_opts_pop_ui_final_again: st.session_state[pop_zone_key_final_ss_again] = zone_opts_pop_ui_final_again[0]
    selected_zone_pop_ui_final_again = st.sidebar.selectbox("Filter by Zone:", options=zone_opts_pop_ui_final_again, key=f"{pop_zone_key_final_ss_again}_widget", index=zone_opts_pop_ui_final_again.index(st.session_state[pop_zone_key_final_ss_again]))
    st.session_state[pop_zone_key_final_ss_again] = selected_zone_pop_ui_final_again
    if selected_zone_pop_ui_final_again != "All Zones (Aggregated)" and 'zone_id' in analytics_df_pop_display_final_df_again.columns:
        zone_id_to_filter_final_again = zone_map_pop_ui_final_again.get(selected_zone_pop_ui_final_again, selected_zone_pop_ui_final_again)
        analytics_df_pop_display_final_df_again = analytics_df_pop_display_final_df_again[analytics_df_pop_display_final_df_again['zone_id'] == zone_id_to_filter_final_again]
    elif selected_zone_pop_ui_final_again != "All Zones (Aggregated)": st.sidebar.caption("'zone_id' missing in health data for zone filtering.")

if analytics_df_pop_display_final_df_again.empty and (start_date_pop_filt_final_ui_again != min_date_pop_final_fix_again or end_date_pop_filt_final_ui_again != max_date_pop_final_fix_again or selected_cond_pop_ui_final_again != "All Conditions (Aggregated)" or selected_zone_pop_ui_final_again != "All Zones (Aggregated)"):
    st.warning("No health data for selected filters. Broaden criteria or check data sources.")

st.subheader(f"Population Health Snapshot ({start_date_pop_filt_final_ui_again.strftime('%d %b %Y')} - {end_date_pop_filt_final_ui_again.strftime('%d %b %Y')}, Cond: {selected_cond_pop_ui_final_again}, Zone: {selected_zone_pop_ui_final_again})")
if analytics_df_pop_display_final_df_again.empty: st.info("Insufficient data for population summary KPIs.")
else:
    cols_kpi_pop_final_again = st.columns(4)
    total_unique_pats_final_val_again = analytics_df_pop_display_final_df_again['patient_id'].nunique() if 'patient_id' in analytics_df_pop_display_final_df_again.columns else 0
    mean_risk_final_val_again = analytics_df_pop_display_final_df_again['ai_risk_score'].mean() if 'ai_risk_score' in analytics_df_pop_display_final_df_again.columns and analytics_df_pop_display_final_df_again['ai_risk_score'].notna().any() else np.nan
    high_risk_count_final_val_again = 0; high_risk_perc_final_val_again = 0.0
    if 'ai_risk_score' in analytics_df_pop_display_final_df_again.columns and total_unique_pats_final_val_again > 0:
        risk_series_numeric_again = convert_to_numeric(analytics_df_pop_display_final_df_again['ai_risk_score'], default_value=-1)
        high_risk_df_final_val_again = analytics_df_pop_display_final_df_again[risk_series_numeric_again >= settings.RISK_SCORE_HIGH_THRESHOLD]
        high_risk_count_final_val_again = high_risk_df_final_val_again['patient_id'].nunique() if 'patient_id' in high_risk_df_final_val_again.columns else 0
        high_risk_perc_final_val_again = (high_risk_count_final_val_again / total_unique_pats_final_val_again) * 100
    top_cond_final_val_again, top_cond_enc_final_val_again = "N/A", 0
    if 'condition' in analytics_df_pop_display_final_df_again.columns and analytics_df_pop_display_final_df_again['condition'].notna().any():
        cond_counts_final_val_again = analytics_df_pop_display_final_df_again['condition'].value_counts()
        if not cond_counts_final_val_again.empty: top_cond_final_val_again, top_cond_enc_final_val_again = cond_counts_final_val_again.idxmax(), cond_counts_final_val_again.max()

    with cols_kpi_pop_final_again[0]: display_custom_styled_kpi_box("Total Unique Patients", total_unique_pats_final_val_again)
    with cols_kpi_pop_final_again[1]: display_custom_styled_kpi_box("Avg. AI Risk Score", f"{mean_risk_final_val_again:.1f}" if pd.notna(mean_risk_final_val_again) else "N/A")
    with cols_kpi_pop_final_again[2]: display_custom_styled_kpi_box("% High AI Risk Patients", f"{high_risk_perc_final_val_again:.1f}%", f"({high_risk_count_final_val_again:,} patients)")
    with cols_kpi_pop_final_again[3]: display_custom_styled_kpi_box("Top Condition (Encounters)", html.escape(str(top_cond_final_val_again)), f"{top_cond_enc_final_val_again:,} encounters", settings.COLOR_RISK_MODERATE)

pop_tab_titles_final_list_val_again = ["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"]
tab_epi_pop_final_val_again, tab_demog_sdoh_pop_final_val_again, tab_clinical_pop_final_val_again, tab_systems_pop_final_val_again = st.tabs(pop_tab_titles_final_list_val_again)

with tab_epi_pop_final_val_again:
    st.header(f"Epidemiological Overview (Filters: {selected_cond_pop_ui_final_again} | {selected_zone_pop_ui_final_again})")
    if analytics_df_pop_display_final_df_again.empty: st.info("No data for Epi Overview with current filters.")
    else:
        # ... (Epi Overview content - same as previous correct version, truncated for brevity) ...
        if 'condition' in analytics_df_pop_display_final_df_again.columns and 'patient_id' in analytics_df_pop_display_final_df_again.columns:
            cond_unique_pat_counts_df_final_val_again = analytics_df_pop_display_final_df_again.groupby('condition')['patient_id'].nunique().nlargest(12).reset_index(name='unique_patients')
            if not cond_unique_pat_counts_df_final_val_again.empty:
                st.plotly_chart(plot_bar_chart(cond_unique_pat_counts_df_final_val_again, 'condition', 'unique_patients', "Top Conditions by Unique Patient Count", orientation_bar='h', y_values_are_counts_flag=True, chart_height=450, x_axis_label_text="Unique Patient Count", y_axis_label_text="Condition"), use_container_width=True)
        if 'ai_risk_score' in analytics_df_pop_display_final_df_again.columns and analytics_df_pop_display_final_df_again['ai_risk_score'].notna().any():
            fig_risk_dist_pop_final_val_again = px.histogram(analytics_df_pop_display_final_df_again.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=25, title="Patient AI Risk Score Distribution", labels={'ai_risk_score': 'AI Risk Score', 'count': 'Records'})
            fig_risk_dist_pop_final_val_again.update_layout(bargap=0.1, height=settings.WEB_PLOT_COMPACT_HEIGHT, title_x=0.05)
            st.plotly_chart(fig_risk_dist_pop_final_val_again, use_container_width=True)
        st.caption("Note: True incidence/prevalence trends require careful case definitions and population denominators.")


with tab_demog_sdoh_pop_final_val_again:
    st.header("Demographics & Social Determinants of Health (SDOH) Context")
    if analytics_df_pop_display_final_df_again.empty: st.info("No data for Demographics & SDOH analysis with current filters.")
    else:
        # ... (Demographics & SDOH content - same as previous correct version, truncated for brevity) ...
        if 'age' in analytics_df_pop_display_final_df_again.columns and analytics_df_pop_display_final_df_again['age'].notna().any():
            age_df_pop_final_val_again = analytics_df_pop_display_final_df_again.dropna(subset=['age'])
            if not age_df_pop_final_val_again.empty:
                age_bins_final_val_again = [0, 5, 18, 35, 50, 65, np.inf]
                age_labels_final_val_again = ['0-4', '5-17', '18-34', '35-49', '50-64', '65+']
                age_series_for_cut_final_again = convert_to_numeric(age_df_pop_final_val_again['age'], default_value=np.nan)
                if age_series_for_cut_final_again.notna().any():
                    # Use .loc with a temporary column to avoid SettingWithCopyWarning
                    temp_age_df = age_df_pop_final_val_again.copy()
                    temp_age_df.loc[:, 'age_group'] = pd.cut(age_series_for_cut_final_again.dropna(), bins=age_bins_final_val_again, labels=age_labels_final_val_again, right=False)
                    age_group_counts_final_val_again = temp_age_df['age_group'].value_counts().sort_index().reset_index()
                    age_group_counts_final_val_again.columns = ['Age Group', 'Count']
                    st.plotly_chart(plot_bar_chart(age_group_counts_final_val_again, 'Age Group', 'Count', "Patient Age Distribution", y_values_are_counts_flag=True), use_container_width=True)
        if not zone_attr_pop_main_fix_again.empty and 'zone_id' in analytics_df_pop_display_final_df_again.columns and 'zone_id' in zone_attr_pop_main_fix_again.columns:
            merged_sdoh_df_final_val_again = pd.merge(analytics_df_pop_display_final_df_again, zone_attr_pop_main_fix_again, on='zone_id', how='left', suffixes=('_health', '_zone'))
            if 'socio_economic_index' in merged_sdoh_df_final_val_again.columns and 'ai_risk_score' in merged_sdoh_df_final_val_again.columns and \
               merged_sdoh_df_final_val_again['socio_economic_index'].notna().any() and merged_sdoh_df_final_val_again['ai_risk_score'].notna().any():
                st.markdown("###### AI Risk Score vs. Socio-Economic Index (Zone Level)")
                hover_name_col_sdoh_again = merged_sdoh_df_final_val_again.get('name_zone', merged_sdoh_df_final_val_again.get('zone_id')) 
                fig_sdoh_risk_final_val_again = px.scatter(merged_sdoh_df_final_val_again.dropna(subset=['socio_economic_index', 'ai_risk_score']), 
                                           x='socio_economic_index', y='ai_risk_score', trendline="ols", hover_name=hover_name_col_sdoh_again,
                                           labels={'socio_economic_index': 'Socio-Economic Index (Higher is better)', 'ai_risk_score': 'Avg. AI Risk Score'})
                st.plotly_chart(fig_sdoh_risk_final_val_again, use_container_width=True)
        if zone_attr_pop_main_fix_again.empty: st.caption("Zone attribute data (for SDOH analysis) unavailable.")


with tab_clinical_pop_final_val_again:
    st.header("Clinical Insights & Diagnostic Patterns")
    if analytics_df_pop_display_final_df_again.empty: st.info("No data for Clinical Insights with current filters.")
    else: st.markdown("_(Placeholder: Analyses on Top Symptoms, other Test Result Distributions, Test Positivity Trends.)_")

with tab_systems_pop_final_val_again:
    st.header("Health Systems Performance & Equity Lens")
    if analytics_df_pop_display_final_df_again.empty: st.info("No data for Health Systems & Equity analysis with current filters.")
    else: st.markdown("_(Placeholder: Analyses on Encounters by Clinic/Zone, Referral Completion, AI Risk Variations by SDOH factors.)_")

st.divider()
st.caption(settings.APP_FOOTER_TEXT)
logger.info(f"Population Health Analytics Console loaded. Filters: Period=({start_date_pop_filt_final_ui_again.isoformat()} to {end_date_pop_filt_final_ui_again.isoformat()}), Cond='{selected_cond_pop_ui_final_again}', Zone='{selected_zone_pop_ui_final_again}'")
