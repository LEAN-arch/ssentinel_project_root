# sentinel_project_root/pages/03_district_dashboard.py
# District Health Strategic Command Center for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple
import os 
from pathlib import Path

# --- Sentinel System Imports ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data, load_zone_data
    from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
    from data_processing.aggregation import get_district_summary_kpis, get_trend_data
    from data_processing.helpers import hash_dataframe_safe
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart, plot_choropleth_map
except ImportError as e:
    st.error(f"Fatal Error: A required module could not be imported.\nDetails: {e}\nThis may be due to an incorrect project structure or dependencies.")
    st.stop()

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Self-Contained Component Logic for this Page ---

def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

def structure_district_summary_kpis(kpis: Dict, **kwargs) -> List[Dict[str, Any]]:
    """Structures district-level KPIs for display."""
    structured_kpis = []
    if not kpis: return []
    
    structured_kpis.append({"title": "Total Population", "value_str": f"{kpis.get('total_population_district', 0):,}", "icon": "👥"})
    structured_kpis.append({"title": "Zones Monitored", "value_str": str(kpis.get('total_zones_in_df', 0)), "icon": "🗺️"})
    structured_kpis.append({"title": "Avg. AI Risk Score", "value_str": f"{kpis.get('population_weighted_avg_ai_risk_score', 0):.1f}", "icon": "🔥", "help_text": "Population-weighted average risk."})
    structured_kpis.append({"title": "High-Risk Zones", "value_str": str(kpis.get('zones_meeting_high_risk_criteria_count', 0)), "icon": "🚨", "help_text": f"Zones with avg. risk > {settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE}."})
    
    return structured_kpis

def render_district_map_visualization(map_df: pd.DataFrame, metric_key: str, display_name: str):
    """Renders the main choropleth map."""
    if 'geometry_obj' not in map_df.columns:
        st.warning("Map unavailable: Zone geometry data is missing.")
        return
    if metric_key not in map_df.columns:
        st.warning(f"Metric '{display_name}' not available in the data for mapping.")
        return

    map_df[metric_key] = pd.to_numeric(map_df[metric_key], errors='coerce').fillna(0)
    
    geojson_data = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": row['geometry_obj'], "id": str(row['zone_id']), "properties": {}}
        for _, row in map_df.iterrows()
    ]}
    
    fig = plot_choropleth_map(map_df, geojson=geojson_data, locations="zone_id", color=metric_key,
                              hover_name="name", labels={metric_key: display_name},
                              title=f"<b>{display_name} by Zone</b>")
    st.plotly_chart(fig, use_container_width=True)

def calculate_district_wide_trends(health_df: pd.DataFrame, iot_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates various district-wide trends."""
    results = {"disease_incidence_trends": {}, "avg_patient_ai_risk_trend": pd.Series(), "avg_clinic_co2_trend": pd.Series()}
    if not health_df.empty:
        results["avg_patient_ai_risk_trend"] = get_trend_data(health_df, 'ai_risk_score', 'encounter_date', 'W-MON')
    if not iot_df.empty:
        results["avg_clinic_co2_trend"] = get_trend_data(iot_df, 'avg_co2_ppm', 'timestamp', 'h')
    return results

def get_district_intervention_criteria_options(df_check: Optional[pd.DataFrame]) -> Dict[str, Dict]:
    options = {
        "High Avg. Patient Risk": {"col": "avg_risk_score", "threshold": settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE, "comparison": "ge"},
        "Low Facility Coverage": {"col": "facility_coverage_score", "threshold": settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT, "comparison": "le"},
        "High TB Burden": {"col": "active_tb_cases", "threshold": settings.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS, "comparison": "ge"}
    }
    if df_check is not None:
        return {name: config for name, config in options.items() if config["col"] in df_check.columns}
    return options

def identify_priority_zones_for_intervention(df: pd.DataFrame, criteria: List[str], options: Dict) -> pd.DataFrame:
    if df.empty or not criteria: return pd.DataFrame()
    final_mask = pd.Series(False, index=df.index)
    for criterion in criteria:
        config = options.get(criterion)
        if config and config["col"] in df.columns:
            col, thr, comp = config["col"], config["threshold"], config["comparison"]
            if comp == "ge": final_mask |= (df[col] >= thr)
            elif comp == "le": final_mask |= (df[col] <= thr)
    return df[final_mask]

# --- Page Title and Introduction ---
st.title(f"🗺️ {settings.APP_NAME} - District Health Strategic Command Center")
st.markdown(f"**Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring.**")
st.divider()

# --- Data Aggregation and Preparation ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Aggregating district-level operational data...")
def get_dho_processed_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    raw_health_df = load_health_records()
    raw_iot_df = load_iot_clinic_environment_data()
    base_zone_df = load_zone_data()

    if base_zone_df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    ai_enriched_health_df, _ = apply_ai_models(raw_health_df.copy())
    enriched_zone_df = enrich_zone_geodata_with_health_aggregates(base_zone_df, ai_enriched_health_df, raw_iot_df)
    
    if enriched_zone_df.empty: return base_zone_df, ai_enriched_health_df, raw_iot_df, {}

    # --- DEFINITIVE FIX FOR UserHashError ---
    df_for_kpis = enriched_zone_df.copy()
    if 'geometry_obj' in df_for_kpis.columns:
        df_for_kpis = df_for_kpis.drop(columns=['geometry_obj'])
    
    district_summary_kpis = get_district_summary_kpis(df_for_kpis)
    return enriched_zone_df, ai_enriched_health_df, raw_iot_df, district_summary_kpis

# --- Main Logic ---
enriched_zone_df_display, historical_health_df, historical_iot_df, district_kpis = get_dho_processed_datasets()

if enriched_zone_df_display.empty:
    st.error("Could not load or process the necessary zone and health data. The dashboard cannot be displayed.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Analysis Filters")
if os.path.exists(settings.APP_LOGO_SMALL_PATH): st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
abs_min_date, abs_max_date = date.today() - timedelta(days=365*2), date.today()
if not historical_health_df.empty and 'encounter_date' in historical_health_df.columns:
    abs_min_date = historical_health_df['encounter_date'].min().date()
    abs_max_date = historical_health_df['encounter_date'].max().date()

default_start = max(abs_min_date, abs_max_date - timedelta(days=90))
start_date, end_date = st.sidebar.date_input("Select Date Range for Trend Analysis:", value=(default_start, abs_max_date), min_value=abs_min_date, max_value=abs_max_date)

# --- KPI Section ---
st.header("📊 District Performance Dashboard")
structured_kpis = structure_district_summary_kpis(district_kpis)
if structured_kpis:
    cols = st.columns(len(structured_kpis))
    for col, kpi in zip(cols, structured_kpis):
        with col: render_kpi_card(**kpi)
else:
    st.warning("District-wide summary KPIs are unavailable.")
st.divider()

# --- Tabbed Section ---
tab_map, tab_trends, tab_compare, tab_intervene = st.tabs(["🗺️ Geospatial Overview", "📈 District Trends", "🆚 Zonal Comparison", "🎯 Intervention Planning"])

with tab_map:
    st.subheader("Interactive District Health Map")
    map_metric_options = {
        "Average AI Risk Score": "avg_risk_score",
        "Prevalence per 1,000 Pop.": "prevalence_per_1000",
        "Total Patient Encounters": "total_patient_encounters"
    }
    selected_metric_name = st.selectbox("Select Map Metric:", options=list(map_metric_options.keys()))
    render_district_map_visualization(enriched_zone_df_display, enriched_zone_df_display, map_metric_options[selected_metric_name], selected_metric_name)

with tab_trends:
    st.subheader("District-Wide Health & Environmental Trends")
    df_health_trends = historical_health_df[historical_health_df['encounter_date'].dt.date.between(start_date, end_date)]
    if not df_health_trends.empty:
        risk_trend = get_trend_data(df_health_trends, 'ai_risk_score', 'encounter_date', 'W-MON')
        st.plotly_chart(plot_annotated_line_chart(risk_trend, "Weekly Average Patient Risk Score", "Avg. Risk Score"), use_container_width=True)
    else:
        st.info("No health data available for the selected trend period.")

with tab_compare:
    st.subheader("Comparative Zonal Analysis")
    if not enriched_zone_df_display.empty:
        comparison_cols = ['name', 'population', 'avg_risk_score', 'prevalence_per_1000', 'total_patient_encounters']
        display_cols = [col for col in comparison_cols if col in enriched_zone_df_display.columns]
        st.dataframe(enriched_zone_df_display[display_cols].sort_values('avg_risk_score', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.warning("Zonal comparison cannot be performed.")

with tab_intervene:
    st.subheader("Targeted Intervention Planning Assistant")
    criteria_options = get_district_intervention_criteria_options(enriched_zone_df_display)
    selected_criteria = st.multiselect("Select Criteria to Identify Priority Zones:", options=list(criteria_options.keys()))
    if selected_criteria:
        priority_zones_df = identify_priority_zones(enriched_zone_df_display, selected_criteria, criteria_options)
        st.markdown(f"###### **{len(priority_zones_df)} Zone(s) Flagged for Intervention**")
        if not priority_zones_df.empty:
            st.dataframe(priority_zones_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No zones currently meet the selected criteria.")
    else:
        st.info("Select one or more criteria to identify priority zones.")

logger.info("DHO Strategic Command Center page loaded/refreshed.")
