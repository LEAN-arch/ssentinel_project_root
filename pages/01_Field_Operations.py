# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - FIELD OPERATIONS DASHBOARD (V7 - DEFINITIVE REDESIGN)

import logging
from datetime import date, timedelta
from typing import Dict

import pandas as pd
import streamlit as st

from analytics import apply_ai_models, generate_chw_alerts
from config import settings
from data_processing import load_health_records
from data_processing.cached import get_cached_trend
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_donut_chart, plot_line_chart,
                           render_kpi_card)

# --- Page Setup ---
st.set_page_config(page_title="Field Operations", page_icon="🧑‍⚕️", layout="wide")
logger = logging.getLogger(__name__)

# --- Data Loading ---
@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    """
    Loads and caches the base health records, ensuring they are fully
    enriched with all required AI-generated scores for this dashboard.
    """
    raw_df = load_health_records()
    if raw_df.empty:
        return pd.DataFrame()
    enriched_df, _ = apply_ai_models(raw_df)
    return enriched_df

# --- New, Advanced KPI Calculation Logic ---
def calculate_field_ops_kpis(df: pd.DataFrame) -> Dict:
    """
    Calculates a rich set of decision-grade KPIs for field operations,
    focusing on screening and diagnostic funnels.
    """
    if df.empty:
        return {}

    kpis = {}
    
    # Basic Activity
    kpis['total_encounters'] = len(df)
    kpis['unique_patients_seen'] = df['patient_id'].nunique()

    # --- Malaria Screening Cascade ---
    malaria_symptomatic = df[df['patient_reported_symptoms'].str.contains('fever', case=False, na=False)]
    kpis['malaria_symptomatic_count'] = len(malaria_symptomatic)
    
    screened_for_malaria = malaria_symptomatic[malaria_symptomatic['test_type'] == 'Malaria RDT']
    kpis['malaria_screened_count'] = len(screened_for_malaria)
    
    kpis['malaria_screening_rate'] = (kpis['malaria_screened_count'] / kpis['malaria_symptomatic_count'] * 100) if kpis['malaria_symptomatic_count'] > 0 else 0
    
    positive_malaria = screened_for_malaria[screened_for_malaria['test_result'] == 'Positive']
    kpis['malaria_positive_count'] = len(positive_malaria)
    
    kpis['malaria_positivity_rate'] = (kpis['malaria_positive_count'] / kpis['malaria_screened_count'] * 100) if kpis['malaria_screened_count'] > 0 else 0

    # --- Tuberculosis (TB) Screening Cascade ---
    tb_symptomatic = df[df['patient_reported_symptoms'].str.contains('cough', case=False, na=False)]
    kpis['tb_symptomatic_count'] = len(tb_symptomatic)

    # Assuming a 'TB Screen' test type exists for this use case
    screened_for_tb = tb_symptomatic[tb_symptomatic['test_type'] == 'TB Screen']
    kpis['tb_screened_count'] = len(screened_for_tb)
    
    kpis['tb_screening_rate'] = (kpis['tb_screened_count'] / kpis['tb_symptomatic_count'] * 100) if kpis['tb_symptomatic_count'] > 0 else 0

    positive_tb = screened_for_tb[screened_for_tb['test_result'] == 'Positive']
    kpis['tb_positive_count'] = len(positive_tb)

    # Assuming referral status indicates linkage to care
    linked_tb = positive_tb[positive_tb['referral_status'] == 'Completed']
    kpis['tb_linked_to_care_count'] = len(linked_tb)

    kpis['tb_linkage_rate'] = (kpis['tb_linked_to_care_count'] / kpis['tb_positive_count'] * 100) if kpis['tb_positive_count'] > 0 else 0

    return kpis

# --- New, Actionable UI Rendering Components ---
def render_screening_cascade(title: str, icon: str, kpis: dict, prefix: str):
    """Renders a visual funnel for a screening program."""
    st.subheader(f"{icon} {title} Screening Cascade")
    
    symptomatic = kpis.get(f'{prefix}_symptomatic_count', 0)
    screened = kpis.get(f'{prefix}_screened_count', 0)
    positive = kpis.get(f'{prefix}_positive_count', 0)
    linked = kpis.get(f'{prefix}_linked_to_care_count') # Can be None if not applicable

    c1, c2, c3 = st.columns(3)
    c1.metric("Symptomatic Patients", f"{symptomatic:,}")
    c2.metric(f"Screened (Rate)", f"{screened:,}", f"{kpis.get(f'{prefix}_screening_rate', 0):.1f}% of symptomatic")
    c3.metric(f"Positive (Rate)", f"{positive:,}", f"{kpis.get(f'{prefix}_positivity_rate', 0):.1f}% of screened")

    if linked is not None:
        st.metric("Linked to Care (Rate)", f"{linked:,}", f"{kpis.get(f'{prefix}_linkage_rate', 0):.1f}% of positive", help="Patients with a completed referral after a positive test.")

def render_symptom_surveillance(df: pd.DataFrame):
    """Renders a chart of the most common reported symptoms."""
    st.subheader("Symptom Surveillance")
    if 'patient_reported_symptoms' not in df.columns or df['patient_reported_symptoms'].isna().all():
        st.info("No symptom data available.")
        return

    symptoms = df['patient_reported_symptoms'].dropna().str.split(r'[;,]').explode()
    symptoms = symptoms.str.strip().str.title()
    symptom_counts = symptoms[symptoms != ''].value_counts().nlargest(7)
    
    if symptom_counts.empty:
        st.info("No specific symptoms reported in this period.")
    else:
        fig = plot_bar_chart(
            symptom_counts.reset_index(),
            y_col='index', x_col='patient_reported_symptoms',
            title="Top Reported Symptoms",
            orientation='h',
            y_title="Symptom", x_title="Number of Reports"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Main Page Execution ---
def main():
    st.title("🧑‍⚕️ Field Operations Command Center")
    st.markdown("An actionable overview of team activities, screening program performance, and emerging health signals in the selected zone.")
    st.divider()

    full_df = get_data()

    if full_df.empty:
        st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Filters")
        min_date, max_date = full_df['encounter_date'].min().date(), full_df['encounter_date'].max().date()
        zone_options = ["All Zones"] + sorted(full_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
        view_date = st.date_input("View Data For Date:", value=max_date, min_value=min_date, max_value=max_date)

    # --- Filter Data ---
    daily_df = full_df[full_df['encounter_date'].dt.date == view_date]
    if selected_zone != "All Zones":
        daily_df = daily_df[daily_df['zone_id'] == selected_zone]

    st.info(f"**Viewing:** {view_date:%A, %d %b %Y} | **Zone:** {selected_zone}")

    # --- Calculate KPIs from filtered data ---
    kpis = calculate_field_ops_kpis(daily_df)

    # --- Redesigned Layout ---
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Key Performance Indicators")
        render_kpi_card("Patients Seen", kpis.get('unique_patients_seen', 0), icon="👥", help_text="Total unique patients with an encounter on the selected day.")
        render_kpi_card("Malaria Screening Rate", f"{kpis.get('malaria_screening_rate', 0):.1f}%", icon="🦟", status_level="MODERATE_CONCERN" if kpis.get('malaria_screening_rate', 100) < 80 else "GOOD_PERFORMANCE", help_text="Percentage of patients with fever who received a Malaria RDT.")
        render_kpi_card("TB Linkage to Care", f"{kpis.get('tb_linkage_rate', 0):.1f}%", icon="🫁", status_level="HIGH_CONCERN" if kpis.get('tb_linkage_rate', 100) < 75 else "GOOD_PERFORMANCE", help_text="Percentage of patients with a positive TB screen who were successfully linked to care.")
        
        st.divider()
        
        st.subheader("Team Activity Breakdown")
        if not daily_df.empty and 'encounter_type' in daily_df.columns:
            activity_counts = daily_df['encounter_type'].value_counts()
            fig = plot_donut_chart(activity_counts.reset_index(), label_col='encounter_type', value_col='count', title="CHW Activities by Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity type data available.")

    with col2:
        render_screening_cascade("Malaria", "🦟", kpis, "malaria")
        st.divider()
        render_screening_cascade("Tuberculosis", "🫁", kpis, "tb")
        st.divider()
        render_symptom_surveillance(daily_df)

if __name__ == "__main__":
    main()
