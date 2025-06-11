# sentinel_project_root/pages/04_Population_Analytics.py
# SME PLATINUM STANDARD - POPULATION ANALYTICS (V9 - VISUALIZATION ENHANCED)

import logging
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import apply_ai_models
from config import settings
from data_processing.loaders import load_health_records, load_zone_data
from data_processing.enrichment import enrich_zone_data_with_aggregates
from data_processing.helpers import convert_to_numeric
from visualization import (plot_bar_chart, plot_choropleth_map,
                           plot_donut_chart, render_custom_kpi)

st.set_page_config(page_title="Population Analytics", page_icon="📊", layout="wide")
logger = logging.getLogger(__name__)

# --- SME VISUALIZATION UPGRADE: Constants for professional, consistent styling ---
PLOTLY_TEMPLATE = "plotly_white"
RISK_COLORS = {'Low Risk': '#28a745', 'Moderate Risk': '#ffc107', 'High Risk': '#dc3545'}
GENDER_COLORS = {"Female": "#E1396C", "Male": "#1f77b4", "Unknown": "#7f7f7f"}
CHOROPLETH_FONT_COLOR = "white"

# --- Mock AI Functions (Unchanged) ---
def predict_risk_transition(df: pd.DataFrame) -> dict:
    if df.empty: return {'nodes': [], 'source_indices': [], 'target_indices': [], 'values': [], 'upward_transitions': 0}
    low_to_mod = int(df[df['ai_risk_score'] < 40].shape[0] * 0.10)
    mod_to_high = int(df[(df['ai_risk_score'] >= 40) & (df['ai_risk_score'] < 65)].shape[0] * 0.05)
    return {
        'nodes': ['Low Risk', 'Moderate Risk', 'High Risk', 'Low Risk (Future)', 'Moderate Risk (Future)', 'High Risk (Future)'],
        'source_indices': [0, 1], 'target_indices': [4, 5],
        'values': [low_to_mod, mod_to_high], 'upward_transitions': low_to_mod + mod_to_high
    }

def calculate_preventive_opportunity(zone_df: pd.DataFrame, health_df: pd.DataFrame) -> pd.DataFrame:
    if health_df.empty:
        zone_df['preventive_opportunity_index'] = 0
        return zone_df
    moderate_risk = health_df[health_df['ai_risk_score'].between(40, 64, inclusive='both')]
    opportunity = moderate_risk.groupby('zone_id').size().reset_index(name='preventive_opportunity_index')
    zone_df = pd.merge(zone_df, opportunity, on='zone_id', how='left').fillna(0)
    return zone_df

def detect_emerging_threats(health_df: pd.DataFrame, lookback_days: int = 90, threshold: float = 2.0) -> pd.DataFrame:
    if health_df.empty: return pd.DataFrame()
    end_date = health_df['encounter_date'].max()
    start_date = end_date - timedelta(days=lookback_days)
    recent_df = health_df[health_df['encounter_date'] > start_date]
    top_diagnoses = recent_df['diagnosis'].value_counts().nlargest(5).index
    if top_diagnoses.empty: return pd.DataFrame()
    threat_diagnosis = np.random.choice(top_diagnoses)
    baseline_avg = np.random.uniform(5, 10)
    recent_avg = baseline_avg * (threshold + np.random.uniform(0, 1))
    threats = [{'diagnosis': threat_diagnosis, 'recent_avg_cases': recent_avg, 'baseline_avg_cases': baseline_avg, 'z_score': threshold + np.random.uniform(0, 1)}]
    return pd.DataFrame(threats)

@st.cache_data(ttl=3600, show_spinner="Loading population datasets...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_health_df = load_health_records()
    if raw_health_df.empty: return pd.DataFrame(), pd.DataFrame()
    health_df, _ = apply_ai_models(raw_health_df)
    if 'risk_factors' not in health_df.columns:
        factors = ['Hypertension', 'Diabetes', 'Smoking', 'Obesity', 'Malnutrition']
        health_df['risk_factors'] = health_df['patient_id'].apply(lambda x: list(np.random.choice(factors, size=np.random.randint(0, 4), replace=False)))
    zone_df = load_zone_data()
    enriched_zone_df = enrich_zone_data_with_aggregates(zone_df, health_df)
    return health_df, enriched_zone_df

@st.cache_data
def get_risk_stratification(df: pd.DataFrame) -> dict:
    if df.empty or 'ai_risk_score' not in df.columns: return {'pyramid_data': pd.DataFrame()}
    def assign_tier(score):
        if score >= settings.ANALYTICS.risk_score_moderate_threshold: return 'High Risk'
        if score >= settings.ANALYTICS.risk_score_low_threshold: return 'Moderate Risk'
        return 'Low Risk'
    df_copy = df.copy().drop_duplicates('patient_id')
    df_copy['risk_tier'] = convert_to_numeric(df_copy['ai_risk_score']).apply(assign_tier)
    pyramid_data = df_copy['risk_tier'].value_counts().reset_index()
    pyramid_data.columns = ['risk_tier', 'patient_count']
    return {'pyramid_data': pyramid_data}

# --- Main Logic ---
def main():
    st.title("📊 Population Health Analytics")
    st.markdown("A strategic command center for understanding population dynamics, predicting future health trends, and targeting interventions.")
    st.divider()

    health_df, zone_df = get_data()
    if health_df.empty or zone_df.empty:
        st.error("Could not load necessary data. Dashboard cannot be rendered.")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        min_date, max_date = health_df['encounter_date'].min().date(), health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range:", value=(max(min_date, max_date - timedelta(days=89)), max_date), min_value=min_date, max_value=max_date, key="pop_date_range")
        zone_options = ["All Zones"] + sorted(zone_df['zone_name'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options, key="pop_zone_filter")

    df_filtered = health_df[health_df['encounter_date'].dt.date.between(start_date, end_date)]
    zone_df_filtered = zone_df.copy()
    if selected_zone != "All Zones":
        if 'zone_id' in zone_df.columns and 'zone_name' in zone_df.columns:
            zone_id_series = zone_df.loc[zone_df['zone_name'] == selected_zone, 'zone_id']
            if not zone_id_series.empty:
                zone_id = zone_id_series.iloc[0]
                df_filtered = df_filtered[df_filtered['zone_id'] == zone_id]
                zone_df_filtered = zone_df[zone_df['zone_id'] == zone_id]

    st.subheader("Population Snapshot")
    cols = st.columns(4)
    unique_patients = df_filtered['patient_id'].nunique()
    with cols[0]: render_custom_kpi("Unique Patients", unique_patients, "In selected period")
    with cols[1]: render_custom_kpi("Avg. Risk Score", df_filtered.get('ai_risk_score').mean(), "0-100 scale")
    high_risk_count = (df_filtered.get('ai_risk_score', pd.Series(dtype=float)) >= settings.ANALYTICS.risk_score_moderate_threshold).sum()
    with cols[2]: render_custom_kpi("High-Risk Patients", high_risk_count, f"Score ≥ {settings.ANALYTICS.risk_score_moderate_threshold}", highlight_status='high-risk')
    with cols[3]: render_custom_kpi("Median Patient Age", df_filtered.get('age').median(), "Years")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Risk Stratification", "🗺️ Geospatial Analysis", "🧑‍🤝‍🧑 Demographics", "🔬 Emerging Threats"])

    with tab1:
        st.header("Population Risk Stratification")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Current Risk Pyramid")
            risk_data = get_risk_stratification(df_filtered)
            pyramid_data = risk_data.get('pyramid_data')
            if not pyramid_data.empty:
                # --- SME VISUALIZATION UPGRADE: Funnel to Horizontal Bar Chart ---
                pyramid_data = pyramid_data.set_index('risk_tier').reindex(['High Risk', 'Moderate Risk', 'Low Risk']).reset_index()
                fig = px.bar(
                    pyramid_data,
                    y='risk_tier',
                    x='patient_count',
                    orientation='h',
                    color='risk_tier',
                    color_discrete_map=RISK_COLORS,
                    text='patient_count',
                    title="Current Population Risk Distribution"
                )
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside')
                fig.update_layout(
                    template=PLOTLY_TEMPLATE,
                    xaxis_title="Number of Patients",
                    yaxis_title="Risk Tier",
                    showlegend=False,
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available to generate a risk pyramid for this period.")
        
        with col2:
            st.subheader("🔮 AI-Predicted Risk Transition (Next 90 Days)")
            st.markdown("This model predicts how many patients are likely to transition between risk tiers, highlighting future high-risk burdens.")
            if not df_filtered.empty:
                predicted_transitions = predict_risk_transition(df_filtered)
                # --- SME VISUALIZATION UPGRADE: Sankey Diagram Polish ---
                fig = go.Figure(go.Sankey(
                    node=dict(
                        pad=25, thickness=20, line=dict(color="black", width=0.5),
                        label=predicted_transitions['nodes'],
                        color=[RISK_COLORS.get(label.split(' ')[0], '#7f7f7f') for label in predicted_transitions['nodes']]
                    ),
                    link=dict(
                        source=predicted_transitions['source_indices'], target=predicted_transitions['target_indices'],
                        value=predicted_transitions['values'],
                        color='rgba(0, 123, 255, 0.5)',
                        hovertemplate='Predicted Transitions: %{value:,.0f}<extra></extra>'
                    )
                ))
                fig.update_layout(title_text="Predicted 90-Day Patient Risk Flow", font_size=12, template=PLOTLY_TEMPLATE, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
                
                upward_transitions = predicted_transitions['upward_transitions']
                st.warning(f"**Actionable Insight:** The model predicts **{upward_transitions:,}** patients will transition to a higher risk tier in the next 90 days. Focus preventive efforts on the 'Moderate Risk' cohort.")
            else:
                st.info("Insufficient data to predict risk transitions.")

    with tab2:
        st.header("Geospatial Analysis")
        if 'geometry' not in zone_df_filtered.columns or zone_df_filtered['geometry'].isnull().all():
            st.warning("Geospatial data is unavailable for the selected zone(s).")
        else:
            geojson_data = {"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {"zone_id": row['zone_id']}, "geometry": row['geometry']}
                for _, row in zone_df_filtered.iterrows() if pd.notna(row.get('geometry'))
            ]}
            map_metric = st.selectbox("Select Map Metric:", ["Average AI Risk Score", "Prevalence per 1,000 People", "Preventive Opportunity Index"])
            
            # --- SME VISUALIZATION UPGRADE: Enhanced Map Styling ---
            if "Risk" in map_metric:
                color_metric, scale, title = 'avg_risk_score', 'Reds', 'Zonal Average Risk Score'
            elif "Prevalence" in map_metric:
                color_metric, scale, title = 'prevalence_per_1000_pop', 'Reds', 'Zonal Disease Prevalence'
            else: 
                zone_df_filtered = calculate_preventive_opportunity(zone_df_filtered, df_filtered)
                color_metric, scale, title = 'preventive_opportunity_index', 'Greens', 'Zonal Preventive Opportunity Index'
                st.info("💡 **Preventive Opportunity Index** highlights zones with a high concentration of moderate-risk individuals who are most amenable to cost-effective preventive care.")

            fig = plot_choropleth_map(
                zone_df_filtered, geojson=geojson_data, value_col=color_metric, title=title, 
                hover_name='zone_name', color_continuous_scale=scale
            )
            fig.update_layout(
                title_x=0.5,
                coloraxis_colorbar=dict(
                    title=map_metric.replace(" ", "<br>"),
                    lenmode="fraction", len=0.8,
                    yanchor="middle", y=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Demographic Insights & Risk Driver Analysis")
        df_unique = df_filtered.drop_duplicates(subset=['patient_id'])
        
        st.subheader("Risk Factor Hotspot Analysis")
        st.markdown("Select a specific risk factor to see which demographic groups are most affected, enabling targeted health campaigns.")
        
        all_factors = sorted(list(set([factor for sublist in df_unique['risk_factors'] for factor in sublist])))
        if all_factors:
            selected_factor = st.selectbox("Select a Risk Factor to Analyze:", all_factors)
            
            if selected_factor:
                factor_df = df_unique[df_unique['risk_factors'].apply(lambda x: selected_factor in x)].copy()
                if not factor_df.empty:
                    age_bins = [0, 18, 40, 65, 150]; age_labels = ['0-17', '18-39', '40-64', '65+']
                    factor_df['age_group'] = pd.cut(factor_df['age'], bins=age_bins, labels=age_labels, right=False)
                    driver_data = factor_df.groupby(['age_group', 'gender']).size().reset_index(name='count')
                    
                    # --- SME VISUALIZATION UPGRADE: Bar Chart Polish ---
                    fig = px.bar(
                        driver_data, x='age_group', y='count', color='gender', barmode='group',
                        title=f"Who is most affected by '{selected_factor}'?",
                        labels={'count': 'Number of Patients', 'age_group': 'Age Group', 'gender': 'Gender'},
                        category_orders={'age_group': age_labels},
                        color_discrete_map=GENDER_COLORS,
                        template=PLOTLY_TEMPLATE,
                        text='count'
                    )
                    fig.update_traces(textposition='outside')
                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No patients with the risk factor '{selected_factor}' in the selected period.")
        else:
            st.info("No risk factor data available for this analysis.")

    with tab4:
        st.header("🔬 Emerging Health Threats Analysis")
        st.markdown("This module uses anomaly detection to identify significant increases in diagnosis rates that may signal an outbreak or emerging public health issue.")
        
        if not health_df.empty:
            threats_df = detect_emerging_threats(health_df, lookback_days=90, threshold=2.0)
            
            if not threats_df.empty:
                st.warning(f"🚨 **Emerging Threats Detected: {len(threats_df)}**")
                for _, row in threats_df.iterrows():
                    with st.container(border=True):
                        st.subheader(f"Anomaly Detected: {row['diagnosis']}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Recent Weekly Average", f"{row['recent_avg_cases']:.1f} cases/wk")
                        col2.metric("Historical Baseline", f"{row['baseline_avg_cases']:.1f} cases/wk")
                        col3.metric("Spike (Std Devs)", f"{row['z_score']:.1f}σ", "Above Baseline", delta_color="inverse")
                        
                        # --- SME VISUALIZATION UPGRADE: Line Chart Polish ---
                        threat_trend_df = health_df[health_df['diagnosis'] == row['diagnosis']]
                        threat_trend_ts = threat_trend_df.set_index('encounter_date').resample('W-MON').size()
                        fig = go.Figure()
                        # Baseline Area
                        fig.add_trace(go.Scatter(x=threat_trend_ts.index, y=[row['baseline_avg_cases']] * len(threat_trend_ts), fill=None, mode='lines', line_color='#28a745', line_dash='dash', name='Historical Baseline'))
                        # Recent Avg Line
                        fig.add_trace(go.Scatter(x=threat_trend_ts.index, y=[row['recent_avg_cases']] * len(threat_trend_ts), fill=None, mode='lines', line_color='#dc3545', line_dash='dash', name='Recent Average'))
                        # Actual Cases Line
                        fig.add_trace(go.Scatter(x=threat_trend_ts.index, y=threat_trend_ts.values, mode='lines+markers', line_color='#007bff', name='Weekly Cases'))
                        
                        fig.update_layout(
                            title=f"Weekly Case Trend for '{row['diagnosis']}'",
                            xaxis_title='Week', yaxis_title='Number of Cases',
                            template=PLOTLY_TEMPLATE, showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No significant emerging health threats detected based on current data.")
        else:
            st.info("Not enough historical data to analyze emerging threats.")

if __name__ == "__main__":
    main()
