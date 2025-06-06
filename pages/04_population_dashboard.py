# sentinel_project_root/pages/04_population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import re 
import html 

try:
    from config import settings
    from data_processing.loaders import load_health_records, load_zone_data 
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe, convert_to_numeric 
    from visualization.plots import plot_bar_chart, create_empty_figure, plot_annotated_line_chart
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

logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

try:
    page_icon_value = "🌍" 
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_FAVICON_PATH'):
        favicon_path = Path(_get_setting('PROJECT_ROOT_DIR','.')) / _get_setting('APP_FAVICON_PATH','assets/favicon.ico')
        if favicon_path.is_file():
            page_icon_value = str(favicon_path)
        else:
            logger.warning(f"Favicon for Population Dashboard not found: {favicon_path}")
    page_layout_value = _get_setting('APP_LAYOUT', "wide")
    st.set_page_config(
        page_title=f"Population Analytics - {_get_setting('APP_NAME', 'Sentinel App')}",
        page_icon=page_icon_value,
        layout=page_layout_value
    )
except Exception as e_page_config:
    logger.error(f"Error applying page configuration for Population Dashboard: {e_page_config}", exc_info=True)
    st.set_page_config(page_title="Population Analytics", page_icon="🌍", layout="wide") 

st.title(f"📊 {_get_setting('APP_NAME', 'Sentinel Health Co-Pilot')} - Population Health Analytics & Research Console")
st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.divider()

@st.cache_data(
    ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 300),
    hash_funcs={pd.DataFrame: hash_dataframe_safe},
    show_spinner="Loading population analytics dataset..."
)
def get_population_analytics_datasets(log_ctx: str = "PopAnalyticsConsole/LoadData") -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"({log_ctx}) Initiating load for population health records and zone attributes.")
    raw_health_df = load_health_records(source_context=f"{log_ctx}/HealthRecs")
    enriched_health_df = pd.DataFrame() 
    base_health_cols_schema = ['patient_id', 'encounter_date', 'condition', 'age', 'gender', 'zone_id', 
                               'ai_risk_score', 'ai_followup_priority_score'] 
    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        try:
            enriched_data, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrich")
            if isinstance(enriched_data, pd.DataFrame): enriched_health_df = enriched_data
            else:
                logger.warning(f"({log_ctx}) AI model app did not return DataFrame. Reverting to raw schema.")
                enriched_health_df = raw_health_df.reindex(columns=base_health_cols_schema, fill_value=np.nan)
        except Exception as e_ai:
            logger.error(f"({log_ctx}) Error during AI model app: {e_ai}. Reverting to raw schema.", exc_info=True)
            enriched_health_df = raw_health_df.reindex(columns=base_health_cols_schema, fill_value=np.nan)
    else:
        logger.warning(f"({log_ctx}) Raw health records empty/invalid. AI enrichment skipped.")
        enriched_health_df = pd.DataFrame(columns=base_health_cols_schema)

    zone_data_full = load_zone_data(source_context=f"{log_ctx}/ZoneData") 
    zone_attributes_df = pd.DataFrame() 
    sdoh_cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min', 
                 'predominant_hazard_type', 'primary_livelihood', 'water_source_main', 'area_sqkm', 
                 'num_clinics', 'num_chws'] 
    if isinstance(zone_data_full, pd.DataFrame) and not zone_data_full.empty:
        cols_to_keep = [col for col in sdoh_cols if col in zone_data_full.columns]
        if 'zone_id' not in cols_to_keep and 'zone_id' in zone_data_full.columns: cols_to_keep.append('zone_id')
        if cols_to_keep:
            zone_attributes_df = zone_data_full[list(set(cols_to_keep))].copy() 
            for sdoh_col_expected in sdoh_cols:
                if sdoh_col_expected not in zone_attributes_df.columns: zone_attributes_df[sdoh_col_expected] = np.nan
        else: zone_attributes_df = pd.DataFrame(columns=sdoh_cols)
    else: zone_attributes_df = pd.DataFrame(columns=sdoh_cols)
    if enriched_health_df.empty: logger.error(f"({log_ctx}) CRITICAL: Enriched health dataset empty.")
    return enriched_health_df, zone_attributes_df

health_df_main, zone_attr_main = pd.DataFrame(), pd.DataFrame() 
data_load_error_flag = False
try:
    health_df_main, zone_attr_main = get_population_analytics_datasets()
    if health_df_main.empty: data_load_error_flag = True 
except Exception as e_main_load:
    data_load_error_flag = True; logger.error(f"Pop Dash: Dataset loading failed: {e_main_load}", exc_info=True)
    st.error(f"🛑 Error loading data: {str(e_main_load)}. Limited functionality.")
if data_load_error_flag or health_df_main.empty: 
    data_dir = _get_setting('DATA_DIR', "data_sources/")
    health_file = Path(_get_setting('HEALTH_RECORDS_CSV_PATH', "health_records_expanded.csv")).name
    st.error(f"🚨 Critical Data Failure: Primary health data empty. Ensure `{health_file}` is in `{str(Path(data_dir).resolve())}` and valid.")

st.sidebar.markdown("---") 
try:
    logo_path = Path(_get_setting('PROJECT_ROOT_DIR', '.')) / _get_setting('APP_LOGO_SMALL_PATH', 'assets/logo_placeholder.png')
    if logo_path.is_file(): st.sidebar.image(str(logo_path.resolve()), width=230)
    else: logger.warning(f"Pop Dash: Sidebar logo not found: {logo_path.resolve()}"); st.sidebar.caption("Logo missing.")
except Exception as e_logo: logger.error(f"Pop Dash: Sidebar logo error: {e_logo}", exc_info=True); st.sidebar.caption("Logo error.")
st.sidebar.markdown("---") 
st.sidebar.header("🔎 Analytics Filters")

min_fallback_pop, max_fallback_pop = date.today() - timedelta(days=3*365), date.today()
min_data_date_pop, max_data_date_pop = min_fallback_pop, max_fallback_pop
if isinstance(health_df_main, pd.DataFrame) and 'encounter_date' in health_df_main.columns:
    try:
        if not pd.api.types.is_datetime64_any_dtype(health_df_main['encounter_date']):
            health_df_main['encounter_date'] = pd.to_datetime(health_df_main['encounter_date'], errors='coerce')
        if health_df_main['encounter_date'].dt.tz is not None:
            health_df_main['encounter_date'] = health_df_main['encounter_date'].dt.tz_localize(None) 
        valid_dates_s = health_df_main['encounter_date'].dropna()
        if not valid_dates_s.empty:
            min_calc, max_calc = valid_dates_s.min().date(), valid_dates_s.max().date()
            if min_calc <= max_calc: min_data_date_pop, max_data_date_pop = min_calc, max_calc
    except Exception as e_dt_range: logger.warning(f"Pop Dash: Error setting date range: {e_dt_range}")

ss_key_date_pop = "pop_dashboard_date_range_v10"
if ss_key_date_pop not in st.session_state or \
   not (isinstance(st.session_state[ss_key_date_pop], list) and len(st.session_state[ss_key_date_pop]) == 2 and \
        all(isinstance(d, date) for d in st.session_state[ss_key_date_pop]) and \
        min_data_date_pop <= st.session_state[ss_key_date_pop][0] <= max_data_date_pop and \
        min_data_date_pop <= st.session_state[ss_key_date_pop][1] <= max_data_date_pop and \
        st.session_state[ss_key_date_pop][0] <= st.session_state[ss_key_date_pop][1]):
    st.session_state[ss_key_date_pop] = [min_data_date_pop, max_data_date_pop]

selected_date_range_pop_val = st.sidebar.date_input("Select Date Range:", value=st.session_state[ss_key_date_pop], min_value=min_data_date_pop, max_value=max_data_date_pop, key=f"{ss_key_date_pop}_widget")
# Initialize filter dates from session state (which has been validated or defaulted)
start_filter_pop, end_filter_pop = st.session_state[ss_key_date_pop]
# If date_input returns a new valid range, update them
if isinstance(selected_date_range_pop_val, (list, tuple)) and len(selected_date_range_pop_val) == 2:
    start_ui, end_ui = selected_date_range_pop_val
    # Clamp to actual data min/max before assigning
    start_filter_pop = min(max(start_ui, min_data_date_pop), max_data_date_pop)
    end_filter_pop = min(max(end_ui, min_data_date_pop), max_data_date_pop)
    if start_filter_pop > end_filter_pop: end_filter_pop = start_filter_pop 
    st.session_state[ss_key_date_pop] = [start_filter_pop, end_filter_pop] # Persist validated selection

cond_options_list = ["All Conditions"]
if isinstance(health_df_main, pd.DataFrame) and 'condition' in health_df_main.columns:
    unique_c = health_df_main['condition'].dropna().astype(str).unique()
    if len(unique_c) > 0 : cond_options_list.extend(sorted(list(unique_c)))
selected_condition_filter_pop_val = st.sidebar.selectbox("Filter by Condition:", options=cond_options_list, index=0, key="pop_cond_v6_final") # Use unique key

zone_options_list = ["All Zones/Regions"]
zone_name_id_map = {} # Use a consistent name
if isinstance(zone_attr_main, pd.DataFrame) and 'name' in zone_attr_main.columns and 'zone_id' in zone_attr_main.columns:
    valid_zones = zone_attr_main.dropna(subset=['name', 'zone_id'])
    if not valid_zones.empty: zone_name_id_map = valid_zones.groupby('name')['zone_id'].first().to_dict(); zone_options_list.extend(sorted(list(zone_name_id_map.keys())))
elif isinstance(health_df_main, pd.DataFrame) and 'zone_id' in health_df_main.columns: # Check health_df_main as fallback
    zone_options_list.extend(sorted(list(health_df_main['zone_id'].dropna().astype(str).unique())))
selected_zone_display_filter_pop_val = st.sidebar.selectbox("Filter by Zone/Region:", options=zone_options_list, index=0, key="pop_zone_v6_final") # Use unique key

# --- Apply Filters to Data ---
df_filtered_final = pd.DataFrame()
if not data_load_error_flag and isinstance(health_df_main, pd.DataFrame) and not health_df_main.empty:
    df_processing = health_df_main.copy()
    if 'encounter_date' in df_processing: 
        start_dt_norm_filter = pd.to_datetime(start_filter_pop).normalize()
        end_dt_norm_filter = pd.to_datetime(end_filter_pop).normalize()
        df_processing = df_processing[(df_processing['encounter_date'].notna()) & (df_processing['encounter_date'].dt.normalize() >= start_dt_norm_filter) & (df_processing['encounter_date'].dt.normalize() <= end_dt_norm_filter)]
    
    if selected_condition_filter_pop_val != "All Conditions" and 'condition' in df_processing.columns: # Check column exists
        df_processing = df_processing[df_processing['condition'] == selected_condition_filter_pop_val]
    
    if selected_zone_display_filter_pop_val != "All Zones/Regions":
        if zone_name_id_map and selected_zone_display_filter_pop_val in zone_name_id_map: # Use zone_name_id_map
            zone_id_to_filter = zone_name_id_map[selected_zone_display_filter_pop_val]
            if 'zone_id' in df_processing.columns: # Check column exists
                df_processing = df_processing[df_processing['zone_id'].astype(str) == str(zone_id_to_filter)]
        elif 'zone_id' in df_processing.columns: # Check column exists for direct filter
            df_processing = df_processing[df_processing['zone_id'].astype(str) == str(selected_zone_display_filter_pop_val)]
    df_filtered_final = df_processing
else:
    if not data_load_error_flag: 
        logger.info("Population Dashboard: Main health data is empty after initial load. No data to filter.")

# --- Main Page Content ---
# DEFINE filter_context_display_str USING THE CORRECT, FINALIZED FILTER VARIABLES
filter_context_display_str = (
    f"({start_filter_pop.strftime('%d %b %Y')} - {end_filter_pop.strftime('%d %b %Y')}, "
    f"Cond: {selected_condition_filter_pop_val}, Zone: {selected_zone_display_filter_pop_val})"
)
st.subheader(f"Population Health Snapshot {filter_context_display_str}")

if data_load_error_flag or df_filtered_final.empty:
    st.info("ℹ️ Insufficient data for population summary KPIs or detailed views.")
else:
    kpi_cols_main_display = st.columns(4) 
    with kpi_cols_main_display[0]: st.metric("Total Encounters", f"{df_filtered_final.shape[0]:,}")
    with kpi_cols_main_display[1]: st.metric("Unique Patients", f"{df_filtered_final.get('patient_id', pd.Series(dtype=str)).nunique():,}")
    with kpi_cols_main_display[2]: 
        avg_age_kpi = convert_to_numeric(df_filtered_final.get('age', pd.Series(dtype=float)), np.nan).mean()
        st.metric("Avg. Patient Age", f"{avg_age_kpi:.1f}" if pd.notna(avg_age_kpi) else "N/A")
    with kpi_cols_main_display[3]: 
        avg_risk_kpi = convert_to_numeric(df_filtered_final.get('ai_risk_score', pd.Series(dtype=float)), np.nan).mean()
        st.metric("Avg. AI Risk Score", f"{avg_risk_kpi:.2f}" if pd.notna(avg_risk_kpi) else "N/A")
    st.markdown("---")

tab_titles_main_population = ["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"] 
tabs_rendered = st.tabs(tab_titles_main_population) 

with tabs_rendered[0]: 
    st.header(f"Epidemiological Overview {filter_context_display_str}")
    if data_load_error_flag or df_filtered_final.empty: st.info("No data for Epi Overview.")
    else:
        if 'condition' in df_filtered_final.columns:
            top_conds_data_epi = df_filtered_final['condition'].value_counts().nlargest(10)
            if not top_conds_data_epi.empty:
                fig_top_conds = px.bar(top_conds_data_epi, y=top_conds_data_epi.index, x=top_conds_data_epi.values, orientation='h', title="Top 10 Conditions by Encounters", labels={'y':'Condition', 'x':'Number of Encounters'})
                fig_top_conds.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat='d', xaxis_rangemode='tozero') 
                if not top_conds_data_epi.empty: # Check again before accessing max()
                    max_val = top_conds_data_epi.max()
                    if pd.notna(max_val) and max_val < 30 and max_val > 0 : fig_top_conds.update_xaxes(dtick=1)
                    elif pd.notna(max_val) and max_val == 0 : fig_top_conds.update_xaxes(dtick=1, range=[0,1])
                st.plotly_chart(fig_top_conds, use_container_width=True)
            else: st.caption("No condition data for top conditions chart.")
        else: st.caption("'condition' column missing for top conditions chart.")
        
        if 'encounter_date' in df_filtered_final.columns:
            df_trend_epi_source = df_filtered_final.set_index('encounter_date')
            if pd.api.types.is_datetime64_any_dtype(df_trend_epi_source.index) and not df_trend_epi_source.empty:
                try:
                    weekly_enc_trend_data = df_trend_epi_source.resample('W-MON').size().reset_index(name='count')
                    if not weekly_enc_trend_data.empty:
                        weekly_enc_trend_data['count'] = weekly_enc_trend_data['count'].astype(int)
                        fig_weekly_trend = plot_annotated_line_chart(data_series=weekly_enc_trend_data.set_index('encounter_date')['count'], chart_title="Weekly Encounters Trend", y_axis_label="Number of Encounters", y_values_are_counts=True)
                        st.plotly_chart(fig_weekly_trend, use_container_width=True)
                    else: st.caption("No weekly encounter trend data after resampling.")
                except Exception as e_resample_epi: logger.error(f"Epi Trend Error: {e_resample_epi}", exc_info=True); st.caption("Trend error.")
            else: st.caption("Encounter dates unsuitable for trend.")
        else: st.caption("'encounter_date' column missing for weekly trend.")

with tabs_rendered[1]: 
    st.header(f"Demographics & Socio-demographic Health (SDOH) {filter_context_display_str}")
    if data_load_error_flag or (df_filtered_final.empty and (not isinstance(zone_attr_main, pd.DataFrame) or zone_attr_main.empty)): st.info("No health or zone data.")
    else:
        if 'age' in df_filtered_final.columns and 'patient_id' in df_filtered_final.columns:
            unique_ages_dist = convert_to_numeric(df_filtered_final.drop_duplicates(subset=['patient_id'])['age'], np.nan).dropna()
            if not unique_ages_dist.empty:
                fig_age_dist_plot = px.histogram(unique_ages_dist, nbins=20, title="Age Distribution (Unique Patients)")
                fig_age_dist_plot.update_layout(yaxis_title="Patient Count", xaxis_title="Age", yaxis_tickformat='d', yaxis_dtick=1, yaxis_rangemode='tozero')
                st.plotly_chart(fig_age_dist_plot, use_container_width=True)
            else: st.caption("No age data for unique patients.")
        else: st.caption("Required columns for age distribution missing ('age' or 'patient_id').")

        if 'gender' in df_filtered_final.columns and 'patient_id' in df_filtered_final.columns:
            unique_genders_dist = df_filtered_final.drop_duplicates(subset=['patient_id'])['gender'].astype(str).value_counts()
            if not unique_genders_dist.empty:
                fig_gender_dist_plot = px.pie(unique_genders_dist, values=unique_genders_dist.values, names=unique_genders_dist.index, title="Gender Distribution (Unique Patients)")
                fig_gender_dist_plot.update_traces(texttemplate='%{value:d} (%{percent})', hoverinfo='label+percent+value')
                st.plotly_chart(fig_gender_dist_plot, use_container_width=True)
            else: st.caption("No gender data for unique patients.")
        else: st.caption("Required columns for gender distribution missing ('gender' or 'patient_id').")
        
        zone_attr_display_df = zone_attr_main.copy() if isinstance(zone_attr_main, pd.DataFrame) else pd.DataFrame()
        if not zone_attr_display_df.empty and selected_zone_display_filter_pop_val != "All Zones/Regions": 
            if zone_name_id_map and selected_zone_display_filter_pop_val in zone_name_id_map: 
                zone_id_filter_val = zone_name_id_map[selected_zone_display_filter_pop_val] 
                zone_attr_display_df = zone_attr_display_df[zone_attr_display_df['zone_id'].astype(str) == str(zone_id_filter_val)]
            elif 'zone_id' in zone_attr_display_df.columns: 
                zone_attr_display_df = zone_attr_display_df[zone_attr_display_df['zone_id'].astype(str) == str(selected_zone_display_filter_pop_val)]
        
        if not zone_attr_display_df.empty:
            st.markdown("---"); st.subheader("Zone Attributes" + (f" for {selected_zone_display_filter_pop_val}" if selected_zone_display_filter_pop_val != "All Zones/Regions" else ""))
            if 'population' in zone_attr_display_df.columns and zone_attr_display_df['population'].notna().any():
                pop_by_zone_plot_data = zone_attr_display_df.dropna(subset=['population', 'name']).sort_values('population', ascending=False).head(15)
                if not pop_by_zone_plot_data.empty:
                    fig_pop_by_zone = px.bar(pop_by_zone_plot_data, x='name', y='population', title="Population by Zone")
                    fig_pop_by_zone.update_layout(yaxis_tickformat='d', yaxis_rangemode='tozero')
                    if not pop_by_zone_plot_data.empty: 
                        max_pop = pop_by_zone_plot_data['population'].max()
                        if pd.notna(max_pop) and max_pop < 30 and max_pop > 0 : fig_pop_by_zone.update_yaxes(dtick=1)
                        elif pd.notna(max_pop) and max_pop == 0 : fig_pop_by_zone.update_yaxes(dtick=1, range=[0,1])
                    st.plotly_chart(fig_pop_by_zone, use_container_width=True)
            if 'socio_economic_index' in zone_attr_display_df.columns and zone_attr_display_df['socio_economic_index'].notna().any():
                sei_by_zone_plot_data = zone_attr_display_df.dropna(subset=['socio_economic_index', 'name']).sort_values('socio_economic_index')
                if not sei_by_zone_plot_data.empty: st.plotly_chart(px.bar(sei_by_zone_plot_data, x='name', y='socio_economic_index', title="Socio-Economic Index (Lower is better)"), use_container_width=True)
            if selected_zone_display_filter_pop_val != "All Zones/Regions" and not zone_attr_display_df.empty and 'name' in zone_attr_display_df.columns: 
                sdoh_cols_for_table = [c for c in ['population', 'socio_economic_index', 'avg_travel_time_clinic_min', 'predominant_hazard_type', 'primary_livelihood', 'water_source_main'] if c in zone_attr_display_df]
                if sdoh_cols_for_table : st.dataframe(zone_attr_display_df.set_index('name')[sdoh_cols_for_table].T.dropna(axis=1, how='all'), use_container_width=True)
            elif not zone_attr_display_df.empty and zone_attr_display_df.shape[0] > 15: 
                sdoh_sample_cols_for_table = [c for c in ['name', 'population', 'socio_economic_index'] if c in zone_attr_display_df]
                if sdoh_sample_cols_for_table: st.dataframe(zone_attr_display_df[sdoh_sample_cols_for_table].head(15), use_container_width=True)
        elif not data_load_error_flag and isinstance(zone_attr_main, pd.DataFrame) and not zone_attr_main.empty: st.caption("No zone attributes for selected zone.")

with tabs_rendered[2]: 
    st.header(f"Clinical Insights {filter_context_display_str}")
    if data_load_error_flag or df_filtered_final.empty: st.info("No data for Clinical Insights.")
    else:
        if 'ai_risk_score' in df_filtered_final.columns:
            risk_scores_data_clin = convert_to_numeric(df_filtered_final['ai_risk_score'], np.nan).dropna()
            if not risk_scores_data_clin.empty:
                fig_risk_hist = px.histogram(risk_scores_data_clin, title="AI Risk Score Distribution")
                fig_risk_hist.update_layout(yaxis_title="Frequency", yaxis_tickformat='d', yaxis_dtick=1, yaxis_rangemode='tozero')
                st.plotly_chart(fig_risk_hist, use_container_width=True)
            else: st.caption("No AI risk score data to display.")
        else: st.caption("'ai_risk_score' column missing.")

        if 'ai_followup_priority_score' in df_filtered_final.columns:
            prio_scores_data_clin = convert_to_numeric(df_filtered_final['ai_followup_priority_score'], np.nan).dropna()
            if not prio_scores_data_clin.empty:
                fig_prio_hist = px.histogram(prio_scores_data_clin, title="AI Follow-up Priority Score Distribution")
                fig_prio_hist.update_layout(yaxis_title="Frequency", yaxis_tickformat='d', yaxis_dtick=1, yaxis_rangemode='tozero')
                st.plotly_chart(fig_prio_hist, use_container_width=True)
            else: st.caption("No AI follow-up priority score data to display.")
        else: st.caption("'ai_followup_priority_score' column missing.")

with tabs_rendered[3]: 
    st.header(f"Systems & Equity Insights {filter_context_display_str}")
    if data_load_error_flag or (df_filtered_final.empty and (not isinstance(zone_attr_main, pd.DataFrame) or zone_attr_main.empty)): st.info("No health or zone data.")
    else:
        if 'zone_id' in df_filtered_final.columns:
            enc_by_zone_sys = df_filtered_final['zone_id'].value_counts().nlargest(20)
            if not enc_by_zone_sys.empty:
                if isinstance(zone_attr_main, pd.DataFrame) and 'zone_id' in zone_attr_main.columns and 'name' in zone_attr_main.columns: 
                    zone_map_sys_equity = zone_attr_main.dropna(subset=['zone_id', 'name']).drop_duplicates(subset=['zone_id']).set_index('zone_id')['name']
                    enc_by_zone_sys.index = enc_by_zone_sys.index.map(lambda x_val: f"{zone_map_sys_equity.get(x_val, str(x_val))} ({str(x_val)})")
                fig_enc_zone_plot = px.bar(enc_by_zone_sys, y=enc_by_zone_sys.index, x=enc_by_zone_sys.values, orientation='h', title="Encounter Distribution by Zone", labels={'y':'Zone', 'x':'Encounters'})
                fig_enc_zone_plot.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat='d', xaxis_rangemode='tozero')
                if not enc_by_zone_sys.empty: 
                    max_enc = enc_by_zone_sys.max()
                    if pd.notna(max_enc) and max_enc < 30 and max_enc > 0 : fig_enc_zone_plot.update_xaxes(dtick=1)
                    elif pd.notna(max_enc) and max_enc == 0 : fig_enc_zone_plot.update_xaxes(dtick=1, range=[0,1])
                st.plotly_chart(fig_enc_zone_plot, use_container_width=True)
            else: st.caption("No encounter by zone data to display.")
        else: st.caption("'zone_id' column missing for encounter distribution.")
        
        required_cols_equity = ['zone_id', 'name', 'avg_travel_time_clinic_min', 'population']
        if not df_filtered_final.empty and isinstance(zone_attr_main, pd.DataFrame) and not zone_attr_main.empty and \
           all(c in zone_attr_main.columns for c in required_cols_equity) and 'zone_id' in df_filtered_final.columns: 
            try:
                df_h_eq_plot = df_filtered_final[['zone_id', 'patient_id']].copy(); df_h_eq_plot['zone_id'] = df_h_eq_plot['zone_id'].astype(str)
                df_z_eq_plot = zone_attr_main[required_cols_equity].copy(); df_z_eq_plot['zone_id'] = df_z_eq_plot['zone_id'].astype(str)
                df_z_eq_plot['avg_travel_time_clinic_min'] = convert_to_numeric(df_z_eq_plot['avg_travel_time_clinic_min'], np.nan)
                df_z_eq_plot['population'] = convert_to_numeric(df_z_eq_plot['population'], 0.0)
                merged_eq_plot_df = pd.merge(df_h_eq_plot, df_z_eq_plot, on='zone_id', how='left')
                if not merged_eq_plot_df.empty and 'avg_travel_time_clinic_min' in merged_eq_plot_df.columns and merged_eq_plot_df['avg_travel_time_clinic_min'].notna().any():
                    zone_util_plot_df = merged_eq_plot_df.groupby('zone_id').agg(name=('name', 'first'), encounters=('patient_id', 'size'), avg_travel_time=('avg_travel_time_clinic_min', 'mean'), population=('population', 'first')).reset_index()
                    zone_util_plot_df = zone_util_plot_df.dropna(subset=['population', 'avg_travel_time', 'encounters']); zone_util_plot_df = zone_util_plot_df[zone_util_plot_df['population'] > 0] 
                    if not zone_util_plot_df.empty:
                        zone_util_plot_df['utilization_per_1000_pop'] = (zone_util_plot_df['encounters'] / zone_util_plot_df['population']) * 1000
                        zone_util_plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        zone_util_plot_df.dropna(subset=['utilization_per_1000_pop'], inplace=True)
                        if not zone_util_plot_df.empty:
                            fig_equity_plot_final = px.scatter(
                                zone_util_plot_df, 
                                x='avg_travel_time', y='utilization_per_1000_pop', 
                                size='population', 
                                hover_name='name',
                                title="Service Utilization vs. Avg Travel Time by Zone",
                                labels={'avg_travel_time': 'Avg. Travel Time (min)', 
                                        'utilization_per_1000_pop': 'Encounters per 1,000 Pop'}
                            )
                            st.plotly_chart(fig_equity_plot_final, use_container_width=True)
                        else: st.caption("Insufficient valid data for utilization vs. travel time plot after calculations.")
                    else: st.caption("Insufficient data for utilization vs. travel time plot after aggregation.")
                else: st.caption("Merged data for equity plot missing key columns or travel time data.")
            except Exception as e_eq_plot_final: 
                logger.error(f"Equity plot error: {e_eq_plot_final}", exc_info=True)
                st.caption(f"Could not generate service utilization vs. travel time plot: {e_eq_plot_final}")
        else: st.caption("Required data for Service Utilization vs. Travel Time plot is missing or incomplete.")

st.divider()
footer_text_val = _get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot.")
st.caption(footer_text_val)
logger.info(f"Population Dashboard rendered. Filters: {filter_context_display_str}. FilteredRows: {df_filtered_final.shape[0] if isinstance(df_filtered_final, pd.DataFrame) else 0}.")
