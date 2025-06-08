# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
import re
from datetime import date, timedelta
from typing import Dict, Any, Tuple, List, Optional
import os
import sys

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Sentinel System Imports ---
# This ensures all necessary functions from your project are available.
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis, get_trend_data
    from analytics.orchestrator import apply_ai_models
    from analytics.supply_forecasting import forecast_supply_levels_advanced, generate_simple_supply_forecast
    from analytics.alerting import get_patient_alerts_for_clinic
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart
except ImportError as e:
    st.error(f"Fatal Error: A required module could not be imported.\nDetails: {e}\nThis may be due to an incorrect project structure. Please ensure all files and '__init__.py' files are correctly placed.")
    logger.critical(f"Clinic Dashboard - Unrecoverable Import Error: {e}", exc_info=True)
    st.stop()


# --- Self-Contained Page Components (to prevent circular imports) ---

class _KPIStructurer:
    """A data-driven class to structure raw KPI data into a display-ready format."""
    def __init__(self, kpis_summary: Optional[Dict[str, Any]]):
        self.summary_data = kpis_summary if isinstance(kpis_summary, dict) else {}
        self._MAIN_KPI_DEFS = [
            {"title": "Overall Avg. TAT", "source_key": "overall_avg_test_turnaround_conclusive_days", "target_setting": "TARGET_TEST_TURNAROUND_DAYS", "default_target": 2.0, "units": "days", "icon": "⏱️", "help_template": "Avg. Turnaround Time. Target: ~{target:.1f} days.", "status_logic": "lower_is_better", "precision": 1},
            {"title": "% Critical Tests TAT Met", "source_key": "perc_critical_tests_tat_met", "target_setting": "TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY", "default_target": 85.0, "units": "%", "icon": "🎯", "help_template": "Critical tests meeting TAT. Target: ≥{target:.1f}%.", "status_logic": "higher_is_better", "precision": 1},
            {"title": "Pending Critical Tests", "source_key": "total_pending_critical_tests_patients", "target_setting": "TARGET_PENDING_CRITICAL_TESTS", "default_target": 0, "units": "patients", "icon": "⏳", "help_template": "Patients with pending critical tests. Target: {target}.", "status_logic": "lower_is_better_count", "is_count": True},
            {"title": "Sample Rejection Rate", "source_key": "sample_rejection_rate_perc", "target_setting": "TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY", "default_target": 5.0, "units": "%", "icon": "🧪", "help_template": "Rate of rejected lab samples. Target: <{target:.1f}%.", "status_logic": "lower_is_better", "precision": 1},
        ]

    # --- DEFINITIVE FIX for TypeError ---
    # The method signature is corrected to accept all arguments being passed to it.
    def _format_kpi_value(self, value: Any, precision: int, suffix: str, default_str: str, is_count: bool) -> str:
        if pd.isna(value): return default_str
        try:
            num_value = pd.to_numeric(value)
            return f"{int(num_value):,}" if is_count else f"{num_value:,.{precision}f}{suffix}"
        except (ValueError, TypeError): return str(value)
    
    def _get_status(self, v, t, l):
        if pd.isna(v): return "NO_DATA"
        if l == "lower_is_better": return "GOOD_PERFORMANCE" if v <= t else "MODERATE_CONCERN" if v <= t * 1.5 else "HIGH_CONCERN"
        if l == "higher_is_better": return "GOOD_PERFORMANCE" if v >= t else "ACCEPTABLE" if v >= t * 0.8 else "HIGH_CONCERN"
        if l == "lower_is_better_count": return "GOOD_PERFORMANCE" if v == t else "ACCEPTABLE" if v <= t + 2 else "HIGH_CONCERN"
        return "NEUTRAL"

    def _get_nested_value(self, key_path: str):
        keys = key_path.split('.')
        value = self.summary_data
        for key in keys:
            if isinstance(value, dict): value = value.get(key)
            else: return None
        return value

    def _build_kpi(self, conf):
        value = self._get_nested_value(conf["source_key"])
        target = conf.get("target_value", getattr(settings, conf.get("target_setting", ""), conf.get("default_target")))
        value_num = pd.to_numeric(value, errors='coerce')
        status = self._get_status(value_num, target, conf["status_logic"])
        val_str = self._format_kpi_value(
            value, 
            conf.get("precision", 1), 
            conf.get("units", "") if conf.get("units") == "%" else "", 
            "N/A", 
            conf.get("is_count", False)
        )
        help_text = conf["help_template"].format(target=target, days_remaining=getattr(settings, 'CRITICAL_SUPPLY_DAYS_REMAINING', 7))
        units = conf.get("units", "")
        return {"title": conf["title"], "value_str": val_str, "units": units if units != "%" else "", "icon": conf["icon"], "status_level": status, "help_text": help_text}

    def structure_main_kpis(self): return [self._build_kpi(c) for c in self._MAIN_KPI_DEFS]
    
    def structure_disease_and_supply_kpis(self):
        kpi_defs = []
        key_tests = getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {})
        for test_name, config in key_tests.items():
            if isinstance(config, dict):
                kpi_defs.append({"title": f"{config.get('display_name', test_name)} Positivity", "source_key": f"test_summary_details.{test_name}.positive_rate_perc", "target_value": float(config.get("target_max_positivity_pct", 10.0)), "units": "%", "icon": config.get("icon", "🔬"), "help_template": "Positivity rate. Target: <{target:.1f}%.", "status_logic": "lower_is_better", "precision": 1})
        kpi_defs.append({"title": "Key Drug Stockouts", "source_key": "key_drug_stockouts_count", "target_setting": "TARGET_DRUG_STOCKOUTS", "default_target": 0, "units": "items", "icon": "💊", "help_template": "Key drugs with <{days_remaining} days of stock. Target: {target}.", "status_logic": "lower_is_better_count", "is_count": True, "precision": 0})
        return [self._build_kpi(conf) for conf in kpi_defs]


# --- Page Title ---
st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

# --- Data Loading ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading and processing all operational data...")
def get_dashboard_data() -> Tuple[pd.DataFrame, pd.DataFrame, bool, date, date]:
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()
    iot_available = isinstance(iot_df, pd.DataFrame) and not iot_df.empty
    min_date, max_date = date.today() - timedelta(days=365), date.today()
    if not health_df.empty and 'encounter_date' in health_df.columns:
        valid_dates = health_df['encounter_date'].dropna()
        if not valid_dates.empty: min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    ai_enriched_health_df, _ = apply_ai_models(health_df)
    return ai_enriched_health_df, iot_df, iot_available, min_date, max_date

# --- Main App Logic ---
full_health_df, full_iot_df, iot_available, abs_min_date, abs_max_date = get_dashboard_data()

# --- Sidebar ---
st.sidebar.header("Console Filters")
if os.path.exists(settings.APP_LOGO_SMALL_PATH): st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
default_date_range_days = getattr(settings, 'WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
default_start = max(abs_min_date, abs_max_date - timedelta(days=default_date_range_days - 1))
session_key = "clinic_date_range"
if session_key not in st.session_state: st.session_state[session_key] = (default_start, abs_max_date)
start_date, end_date = st.sidebar.date_input("Select Date Range:", value=st.session_state[session_key], min_value=abs_min_date, max_value=abs_max_date)
if start_date > end_date: end_date = start_date
st.session_state[session_key] = (start_date, end_date)

# --- Filter Data ---
period_health_df = full_health_df[full_health_df['encounter_date'].dt.date.between(start_date, end_date)]
period_iot_df = full_iot_df[full_iot_df['timestamp'].dt.date.between(start_date, end_date)] if iot_available and not full_iot_df.empty else pd.DataFrame()
period_kpis = get_clinic_summary_kpis(period_health_df) if not period_health_df.empty else {}

period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
st.info(f"**Displaying Clinic Console for:** `{period_str}`")

# --- KPI Section ---
st.header("🚀 Performance & Environment Snapshot")
kpi_structurer = _KPIStructurer(period_kpis)
main_kpis = kpi_structurer.structure_main_kpis()
disease_kpis = kpi_structurer.structure_disease_and_supply_kpis()

if main_kpis or disease_kpis:
    st.markdown("##### **Overall Service Performance**")
    cols = st.columns(len(main_kpis))
    for i, kpi in enumerate(main_kpis):
        with cols[i]: render_kpi_card(**kpi)
    st.markdown("##### **Key Disease & Supply Indicators**")
    cols = st.columns(len(disease_kpis))
    for i, kpi in enumerate(disease_kpis):
        with cols[i]: render_kpi_card(**kpi)
else:
    st.info("No service performance data available for the selected period.")
st.divider()

# --- Tabbed Section ---
st.header("🛠️ Operational Areas Deep Dive")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Epidemiology", "🔬 Testing", "💊 Supply Chain", "🧍 Patients", "🌿 Environment"])

with tab1:
    st.subheader("Local Epidemiological Intelligence")
    if period_health_df.empty:
        st.info("No data for epidemiological analysis.")
    else:
        st.markdown("###### Top Reported Symptoms")
        symptoms = period_health_df['patient_reported_symptoms'].dropna().str.split(r'[;,|]').explode().str.strip().str.title()
        symptom_counts = symptoms[symptoms != ''].value_counts().nlargest(10).reset_index()
        symptom_counts.columns = ['Symptom', 'Count']
        st.dataframe(symptom_counts, hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Testing & Diagnostics Performance")
    if period_health_df.empty:
        st.info("No data for testing analysis.")
    else:
        st.markdown("###### Test Result Distribution")
        st.dataframe(period_health_df['test_result'].value_counts())

with tab3:
    st.subheader("Medical Supply Forecast")
    use_ai = st.checkbox("Use Advanced AI Forecast", key="supply_ai_toggle")
    forecast_df = generate_simple_supply_forecast(full_health_df) if not use_ai else forecast_supply_levels_advanced(full_health_df)
    if not forecast_df.empty:
        st.markdown("###### Forecasted Days of Supply")
        st.dataframe(forecast_df.sort_values('forecast_date').drop_duplicates('item', keep='first'))
    else:
        st.info("Could not generate supply forecast.")

with tab4:
    st.subheader("High-Interest Patient Cases")
    if period_health_df.empty:
        st.info("No data for patient analysis.")
    else:
        flagged_patients = get_patient_alerts_for_clinic(health_df_period=period_health_df)
        st.markdown("###### Flagged Patients for Clinical Review")
        if not flagged_patients.empty:
            st.dataframe(flagged_patients[['patient_id', 'age', 'gender', 'condition', 'ai_risk_score', 'Alert Reason']].head(15))
        else:
            st.success("✅ No patients currently flagged for review.")

with tab5:
    st.subheader("Facility Environment Monitoring")
    if period_iot_df.empty:
        st.info("No environmental data for this period.")
    else:
        st.markdown("###### Latest Sensor Readings")
        st.dataframe(period_iot_df.sort_values('timestamp', ascending=False).drop_duplicates('room_name', keep='first'))
