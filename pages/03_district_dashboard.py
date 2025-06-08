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
    from data_processing.aggregation import get_district_summary_kpis
    from data_processing.helpers import hash_dataframe_safe
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart, plot_choropleth_map
except ImportError as e_dist_dash_abs:
    st.error(f"District Dashboard Import Error: {e_dist_dash_abs}. Please ensure all modules are correctly placed and dependencies installed.")
    st.stop()

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Self-Contained Component Logic for this Page ---

def structure_district_summary_kpis(kpis: Dict, **kwargs) -> List[Dict[str, Any]]:
    """Structures district-level KPIs for display."""
    structured_kpis = []
    if not kpis: return []
    
    structured_kpis.append({"title": "Total Population", "value_str": f"{kpis.get('total_population_district', 0):,}", "icon": "👥"})
    structured_kpis.append({"title": "Zones Monitored", "value_str": str(kpis.get('total_zones_in_df', 0)), "icon": "🗺️"})
    structured_kpis.append({"title": "Avg. AI Risk Score", "value_str": f"{kpis.get('population_weighted_avg_ai_risk_score', 0):.1f}", "icon": "🔥", "help_text": "Population-weighted average risk."})
    structured_kpis.append({"title": "High-Risk Zones", "value_str": str(kpis.get('zones_meeting_high_risk_criteria_count', 0)), "icon": "🚨", "help_text": f"Zones with avg. risk > {settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE}."})
    
    return structured_kpis

def render_district_map_visualization(map_df: pd.DataFrame, geojson_data: Dict, metric_to_plot: str, selected_map_metric: str):
    """Renders the main choropleth map."""
    if metric_to_plot not in map_df.columns:
        st.warning(f"Metric '{selected_map_metric}' not available in the data.")
        return
    
    fig = plot_choropleth_map(
        map_df,
        geojson=geojson_data,
        locations="zone_id",
        color=metric_to_plot,
        title="",
        hover_name="name",
        hover_data={"zone_id": False, metric_to_plot: ":.2f"},
        labels={metric_to_plot: selected_map_metric}
    )
    st.plotly_chart(fig, use_container_width=True)

def calculate_district_wide_trends(health_df, iot_df, start_date, end_date, **kwargs) -> Dict[str, Any]:
    """Calculates various district-wide trends."""
    results = {"disease_incidence_trends": {}, "data_availability_notes": []}
    if not health_df.empty:
        results["avg_patient_ai_risk_trend"] = get_trend_data(health_df, 'ai_risk_score', 'encounter_date', 'W-MON')
    if not iot_df.empty:
        results["avg_clinic_co2_trend"] = get_trend_data(iot_df, 'avg_co2_ppm', 'timestamp', 'h')
    return results

def prepare_district_zonal_comparison_data(df, **kwargs):
    if df.empty: return {"zonal_comparison_table_df": pd.DataFrame()}
    return {"zonal_comparison_table_df": df[['name', 'avg_risk_score', 'prevalence_per_1000', 'total_patient_encounters']].sort_values('avg_risk_score', ascending=False)}

def get_district_intervention_criteria_options(**kwargs):
    return {"High Average Risk": "avg_risk_score", "High Prevalence": "prevalence_per_1000", "Low Facility Coverage": "facility_coverage_score"}

def identify_priority_zones_for_intervention_planning(df, selected_criteria, criteria_options, **kwargs):
    if df.empty or not selected_criteria:
        return {"priority_zones_for_intervention_df": pd.DataFrame(), "applied_criteria_display_names": []}
    
    # Example simple logic: flag if in top 25% for any selected metric
    priority_zones = pd.DataFrame()
    for criterion in selected_criteria:
        metric_col = criteria_options.get(criterion)
        if metric_col in df.columns:
            threshold = df[metric_col].quantile(0.75)
            flagged = df[df[metric_col] >= threshold]
            priority_zones = pd.concat([priority_zones, flagged]).drop_duplicates()
            
    return {"priority_zones_for_intervention_df": priority_zones, "applied_criteria_display_names": selected_criteria}


# --- Page Title and Introduction ---
st.title(f"🗺️ {settings.APP_NAME} - District Health Strategic Command Center")
st.markdown(f"**Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring.**")
st.divider()


# --- Data Aggregation and Preparation ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Aggregating district-level operational data...")
def get_dho_command_center_processed_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log_ctx = "DHODatasetPrep"
    logger.info(f"({log_ctx}) Initializing full data pipeline for DHO view...")
    raw_health_df, raw_iot_df, base_zone_df = load_health_records(), load_iot_clinic_environment_data(), load_zone_data()

    if base_zone_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    ai_enriched_health_df, _ = apply_ai_models(raw_health_df.copy())
    
    # DEFINITIVE FIX: The source_context argument is removed.
    enriched_zone_df = enrich_zone_geodata_with_health_aggregates(
        zone_df=base_zone_df, 
        health_df=ai_enriched_health_df,
        iot_df=raw_iot_df
    )
    
    district_summary_kpis = get_district_summary_kpis(enriched_zone_df)
    return enriched_zone_df, ai_enriched_health_df, raw_iot_df, district_summary_kpis

# --- Main Logic ---
enriched_zone_df_display, historical_health_df_for_trends, historical_iot_df_for_trends, district_kpis_summary_data = get_dho_command_center_processed_datasets()

if enriched_zone_df_display.empty:
    st.error("Could not load or process the necessary zone and health data. The dashboard cannot be displayed.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Analysis Filters")
if os.path.exists(settings.APP_LOGO_SMALL_PATH): st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
abs_min_date_dho, abs_max_date_dho = date.today() - timedelta(days=365*2), date.today()
def_end_dho = abs_max_date_dho
def_start_dho = max(abs_min_date_dho, def_end_dho - timedelta(days=90))
selected_start_date_trends, selected_end_date_trends = st.sidebar.date_input("Select Date Range for Trend Analysis:", value=(def_start_dho, def_end_dho), min_value=abs_min_date_dho, max_value=abs_max_date_dho)

# --- KPI Section ---
st.header("📊 District Performance Dashboard")
structured_dho_kpi_list_val = structure_district_summary_kpis(district_kpis_summary_data)
cols = st.columns(len(structured_dho_kpi_list_val))
for i, kpi_item_dho in enumerate(structured_dho_kpi_list_val):
    with cols[i]: render_kpi_card(**kpi_item_dho)
st.divider()

# --- Tabbed Section ---
tab_map, tab_trends, tab_compare, tab_intervene = st.tabs(["🗺️ Geospatial Overview", "📈 District Trends", "🆚 Zonal Comparison", "🎯 Intervention Planning"])

with tab_map:
    st.subheader("Interactive District Health & Environmental Map")
    if not enriched_zone_df_display.empty and 'geometry_obj' in enriched_zone_df_display.columns:
        map_metric_options = _get_district_map_metric_options_config(enriched_zone_df_display)
        selected_map_metric_name = st.selectbox("Select Map Metric:", options=list(map_metric_options.keys()))
        render_district_map_visualization(enriched_zone_df_display, selected_map_metric_name, "Zonal Health Metrics")
    else:
        st.warning("Map unavailable: Enriched zone data missing, empty, or lacks geometry.")

with tab_trends:
    st.subheader("District-Wide Health & Environmental Trends")
    # ... (trend logic as in the original file) ...
    
with tab_compare:
    st.subheader("Comparative Zonal Analysis")
    # ... (comparison logic as in the original file) ...

with tab_intervene:
    st.subheader("Targeted Intervention Planning Assistant")
    # ... (intervention logic as in the original file) ...

logger.info("DHO Strategic Command Center page loaded/refreshed.")
