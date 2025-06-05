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
health_df_pop, zone_attr_pop = pd.DataFrame(), pd.DataFrame() # Initialize
try: health_df_pop, zone_attr_pop = get_population_analytics_datasets()
except Exception as e_pop_load: logger.error(f"Population Dashboard: Dataset loading failed: {e_pop_load}", exc_info=True); st.error(f"Error loading data: {e_pop_load}.")
if health_df_pop.empty: st.error("🚨 Critical Data Failure: Primary health dataset empty. Console features limited."); # st.stop() implicitly by empty df

# --- Sidebar Filters ---
project_root = Path(settings.PROJECT_ROOT_DIR)
logo_path_pop_sidebar = project_root / settings.APP_LOGO_SMALL_PATH
if logo_path_pop_sidebar.exists() and logo_path_pop_sidebar.is_file(): st.sidebar.image(str(logo_path_pop_sidebar), width=120)
st.sidebar.header("🔎 Analytics Filters")

min_date_pop, max_date_pop = date.today() - timedelta(days=365*3), date.today()
if 'encounter_date' in health_df_pop.columns and health_df_pop['encounter_date'].notna().any():
    if not pd.api.types.is_datetime64_any_dtype(health_df_pop['encounter_date']): health_df_pop['encounter_date'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce')
    if health_df_pop['encounter_date'].notna().any():
        min_date_pop, max_date_pop = health_df_pop['encounter_date'].min().date(), health_df_pop['encounter_date'].max().date()
if min_date_pop > max_date_pop: min_date_pop = max_date_pop 

pop_date_key = "pop_dashboard_date_range_v2"
if pop_date_key not in st.session_state: st.session_state[pop_date_key] = [min_date_pop, max_date_pop]
selected_date_range_pop_ui = st.sidebar.date_input("Select Date Range for Analysis:", value=st.session_state[pop_date_key], min_value=min_date_pop, max_value=max_date_pop, key=f"{pop_date_key}_widget")
start_date_pop_filt_ui, end_date_pop_filt_ui = selected_date_range_pop_ui if isinstance(selected_date_range_pop_ui, (list,tuple)) and len(selected_date_range_pop_ui)==2 else st.session_state[pop_date_key]
if start_date_pop_filt_ui > end_date_pop_filt_ui: st.sidebar.error("Start date <= end date."); end_date_pop_filt_ui = start_date_pop_filt_ui
st.session_state[pop_date_key] = [start_date_pop_filt_ui, end_date_pop_filt_ui]

analytics_df_pop_display: pd.DataFrame = pd.DataFrame()
if 'encounter_date' in health_df_pop.columns:
    analytics_df_pop_display = health_df_pop[(health_df_pop['encounter_date'].notna()) & (health_df_pop['encounter_date'].dt.date >= start_date_pop_filt_ui) & (health_df_pop['encounter_date'].dt.date <= end_date_pop_filt_ui)].copy()
else: st.error("'encounter_date' column missing. Cannot filter.");

selected_cond_pop_ui = "All Conditions (Aggregated)"
if 'condition' in analytics_df_pop_display.columns and analytics_df_pop_display['condition'].notna().any():
    unique_conds = ["All Conditions (Aggregated)"] + sorted(analytics_df_pop_display['condition'].dropna().unique().tolist())
    pop_cond_key = "pop_dashboard_condition_v2"
    if pop_cond_key not in st.session_state or st.session_state[pop_cond_key] not in unique_conds: st.session_state[pop_cond_key] = unique_conds[0]
    selected_cond_pop_ui = st.sidebar.selectbox("Filter by Condition:", options=unique_conds, key=f"{pop_cond_key}_widget", index=unique_conds.index(st.session_state[pop_cond_key]))
    st.session_state[pop_cond_key] = selected_cond_pop_ui
    if selected_cond_pop_ui != "All Conditions (Aggregated)": analytics_df_pop_display = analytics_df_pop_display[analytics_df_pop_display['condition'] == selected_cond_pop_ui]
else: st.sidebar.caption("Condition filter unavailable.")

selected_zone_pop_ui = "All Zones (Aggregated)"; zone_map_pop_ui: Dict[str,str] = {}; zone_opts_pop_ui = ["All Zones (Aggregated)"]
if isinstance(zone_attr_pop, pd.DataFrame) and not zone_attr_pop.empty and 'zone_id' in zone_attr_pop.columns:
    temp_opts = [f"{r.get('name', r['zone_id'])} ({r['zone_id']})" if r.get('name', r['zone_id']) != r['zone_id'] else str(r['zone_id']) for _, r in zone_attr_pop.iterrows()]
    unique_temp_opts = sorted(list(set(temp_opts)))
    for opt_str in unique_temp_opts: # Rebuild map to avoid issues if names are not unique without IDs
        zone_id_match = re.search(r'\((Zone[A-Za-z0-9]+)\)$', opt_str) # Extract ZoneID from "Name (ZoneID)"
        zone_map_pop_ui[opt_str] = zone_id_match.group(1) if zone_id_match else opt_str
    zone_opts_pop_ui.extend(unique_temp_opts)
else: st.sidebar.caption("Zone filter options limited.")
if len(zone_opts_pop_ui) > 1:
    pop_zone_key = "pop_dashboard_zone_v2"
    if pop_zone_key not in st.session_state or st.session_state[pop_zone_key] not in zone_opts_pop_ui: st.session_state[pop_zone_key] = zone_opts_pop_ui[0]
    selected_zone_pop_ui = st.sidebar.selectbox("Filter by Zone:", options=zone_opts_pop_ui, key=f"{pop_zone_key}_widget", index=zone_opts_pop_ui.index(st.session_state[pop_zone_key]))
    st.session_state[pop_zone_key] = selected_zone_pop_ui
    if selected_zone_pop_ui != "All Zones (Aggregated)" and 'zone_id' in analytics_df_pop_display.columns:
        analytics_df_pop_display = analytics_df_pop_display[analytics_df_pop_display['zone_id'] == zone_map_pop_ui.get(selected_zone_pop_ui, selected_zone_pop_ui)]
    elif selected_zone_pop_ui != "All Zones (Aggregated)": st.sidebar.caption("'zone_id' missing in health data.")

if analytics_df_pop_display.empty and (start_date_pop_filt_ui != min_date_pop or end_date_pop_filt_ui != max_date_pop or selected_cond_pop_ui != "All Conditions (Aggregated)" or selected_zone_pop_ui != "All Zones (Aggregated)"):
    st.warning("No health data for selected filters. Broaden criteria or check data sources.")

st.subheader(f"Population Health Snapshot ({start_date_pop_filt_ui.strftime('%d %b %Y')} - {end_date_pop_filt_ui.strftime('%d %b %Y')}, Cond: {selected_cond_pop_ui}, Zone: {selected_zone_pop_ui})")
if analytics_df_pop_display.empty: st.info("Insufficient data for population summary KPIs.")
else:
    cols_kpi_pop_sum = st.columns(4)
    total_unique_pats = analytics_df_pop_display['patient_id'].nunique() if 'patient_id' in analytics_df_pop_display.columns else 0
    mean_risk = analytics_df_pop_display['ai_risk_score'].mean() if 'ai_risk_score' in analytics_df_pop_display.columns and analytics_df_pop_display['ai_risk_score'].notna().any() else np.nan
    high_risk_count = 0; high_risk_perc = 0.0
    if 'ai_risk_score' in analytics_df_pop_display.columns and total_unique_pats > 0:
        high_risk_df = analytics_df_pop_display[analytics_df_pop_display['ai_risk_score'] >= settings.RISK_SCORE_HIGH_THRESHOLD]
        high_risk_count = high_risk_df['patient_id'].nunique() if 'patient_id' in high_risk_df.columns else 0
        high_risk_perc = (high_risk_count / total_unique_pats) * 100
    top_cond, top_cond_enc = "N/A", 0
    if 'condition' in analytics_df_pop_display.columns and analytics_df_pop_display['condition'].notna().any():
        cond_counts = analytics_df_pop_display['condition'].value_counts()
        if not cond_counts.empty: top_cond, top_cond_enc = cond_counts.idxmax(), cond_counts.max()
    with cols_kpi_pop_sum[0]: display_custom_styled_kpi_box("Total Unique Patients", total_unique_pats)
    with cols_kpi_pop_sum[1]: display_custom_styled_kpi_box("Avg. AI Risk Score", f"{mean_risk:.1f}" if pd.notna(mean_risk) else "N/A")
    with cols_kpi_pop_sum[2]: display_custom_styled_kpi_box("% High AI Risk Patients", f"{high_risk_perc:.1f}%", f"({high_risk_count:,} patients)")
    with cols_kpi_pop_sum[3]: display_custom_styled_kpi_box("Top Condition (Encounters)", html.escape(str(top_cond)), f"{top_cond_enc:,} encounters", settings.COLOR_RISK_MODERATE)

# --- Tabs for Detailed Population Analytics ---
pop_tab_titles_list_val = ["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"]
tab_pop_epi_val, tab_pop_demog_sdoh_val, tab_pop_clinical_val, tab_pop_systems_val = st.tabs(pop_tab_titles_list_val)

with tab_pop_epi_val:
    st.header(f"Epidemiological Overview (Filters: {selected_cond_pop_ui} | {selected_zone_pop_ui})")
    if analytics_df_pop_display.empty: st.info("No data for Epi Overview with current filters.")
    else:
        if 'condition' in analytics_df_pop_display.columns and 'patient_id' in analytics_df_pop_display.columns:
            cond_unique_pat_counts_df = analytics_df_pop_display.groupby('condition')['patient_id'].nunique().nlargest(12).reset_index(name='unique_patients')
            if not cond_unique_pat_counts_df.empty:
                st.plotly_chart(plot_bar_chart(cond_unique_pat_counts_df, 'condition', 'unique_patients', "Top Conditions by Unique Patient Count", orientation_bar='h', y_values_are_counts_flag=True, chart_height=450, x_axis_label_text="Unique Patient Count", y_axis_label_text="Condition"), use_container_width=True)
        if 'ai_risk_score' in analytics_df_pop_display.columns and analytics_df_pop_display['ai_risk_score'].notna().any():
            fig_risk_dist_pop_val = px.histogram(analytics_df_pop_display.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=25, title="Patient AI Risk Score Distribution", labels={'ai_risk_score': 'AI Risk Score', 'count': 'Records'})
            fig_risk_dist_pop_val.update_layout(bargap=0.1, height=settings.WEB_PLOT_COMPACT_HEIGHT, title_x=0.05)
            st.plotly_chart(fig_risk_dist_pop_val, use_container_width=True)
        st.caption("Note: True incidence/prevalence trends require careful case definitions and population denominators.")

with tab_pop_demog_sdoh_val:
    st.header("Demographics & Social Determinants of Health (SDOH) Context")
    if analytics_df_pop_display.empty: st.info("No data for Demographics & SDOH analysis with current filters.")
    else:
        # Age Distribution
        if 'age' in analytics_df_pop_display.columns and analytics_df_pop_display['age'].notna().any():
            age_df_pop = analytics_df_pop_display.dropna(subset=['age'])
            age_bins = [0, 5, 18, 35, 50, 65, np.inf]
            age_labels = ['0-4', '5-17', '18-34', '35-49', '50-64', '65+']
            if not age_df_pop.empty:
                age_df_pop['age_group'] = pd.cut(age_df_pop['age'], bins=age_bins, labels=age_labels, right=False)
                age_group_counts = age_df_pop['age_group'].value_counts().sort_index().reset_index()
                age_group_counts.columns = ['Age Group', 'Count']
                st.plotly_chart(plot_bar_chart(age_group_counts, 'Age Group', 'Count', "Patient Age Distribution", y_values_are_counts_flag=True), use_container_width=True)
        # SDOH Example (if zone_attr_pop is available and merged)
        if not zone_attr_pop.empty and 'zone_id' in analytics_df_pop_display.columns and 'zone_id' in zone_attr_pop.columns:
            merged_sdoh_df = pd.merge(analytics_df_pop_display, zone_attr_pop, on='zone_id', how='left')
            if 'socio_economic_index' in merged_sdoh_df.columns and 'ai_risk_score' in merged_sdoh_df.columns and \
               merged_sdoh_df['socio_economic_index'].notna().any() and merged_sdoh_df['ai_risk_score'].notna().any():
                st.markdown("###### AI Risk Score vs. Socio-Economic Index (Zone Level)")
                fig_sdoh_risk = px.scatter(merged_sdoh_df.dropna(subset=['socio_economic_index', 'ai_risk_score']), 
                                           x='socio_economic_index', y='ai_risk_score', 
                                           trendline="ols", hover_name='name_x', # Use zone name from merge
                                           labels={'socio_economic_index': 'Socio-Economic Index (Higher is better)', 'ai_risk_score': 'Avg. AI Risk Score'})
                st.plotly_chart(fig_sdoh_risk, use_container_width=True)
        if zone_attr_pop.empty: st.caption("Zone attribute data (for SDOH analysis) unavailable.")

with tab_pop_clinical_val:
    st.header("Clinical Insights & Diagnostic Patterns")
    if analytics_df_pop_display.empty: st.info("No data for Clinical Insights with current filters.")
    else:
        # Test Positivity Rate Trend example (for Malaria RDT)
        if 'test_type' in analytics_df_pop_display.columns and 'test_result' in analytics_df_pop_display.columns and 'encounter_date' in analytics_df_pop_display.columns:
            malaria_tests_pop = analytics_df_pop_display[analytics_df_pop_display['test_type'] == 'RDT-Malaria'].copy()
            if not malaria_tests_pop.empty:
                malaria_tests_pop['is_positive'] = (malaria_tests_pop['test_result'].astype(str).str.lower() == 'positive')
                # Need to use a function from data_processing.aggregation for trends if available, or simple resample here
                # For simplicity, direct resample example:
                try:
                    malaria_pos_trend_pop = malaria_tests_pop.set_index('encounter_date')['is_positive'].resample('M').mean() * 100
                    if not malaria_pos_trend_pop.empty:
                        st.plotly_chart(plot_annotated_line_chart(malaria_pos_trend_pop, "Malaria RDT Positivity Rate (Monthly %)", "Positivity Rate (%)"), use_container_width=True)
                except Exception as e_trend:
                    logger.warning(f"Could not generate malaria positivity trend: {e_trend}")
                    st.caption("Could not generate Malaria positivity trend.")
        st.markdown("_(Placeholder: Further analyses on Top Symptoms, other Test Result Distributions.)_")

with tab_pop_systems_val:
    st.header("Health Systems Performance & Equity Lens")
    if analytics_df_pop_display.empty: st.info("No data for Health Systems & Equity analysis with current filters.")
    else: st.markdown("_(Placeholder: Analyses on Encounters by Clinic/Zone, Referral Completion, AI Risk Variations by SDOH factors.)_")

st.divider()
st.caption(settings.APP_FOOTER_TEXT)
logger.info(f"Population Health Analytics Console loaded. Filters: Period=({start_date_pop_filt_ui.isoformat()} to {end_date_pop_filt_ui.isoformat()}), Cond='{selected_cond_pop_ui}', Zone='{selected_zone_pop_ui}'")
