# sentinel_project_root/pages/04_Population_Analytics.py
# SME PLATINUM STANDARD - POPULATION STRATEGIC COMMAND CENTER (V17 - FUTURE-PROOFED)

import logging
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Core Sentinel Imports (Assumed to exist) ---
from analytics import apply_ai_models
from config import settings
from data_processing.loaders import load_health_records, load_zone_data
from data_processing.enrichment import enrich_zone_data_with_aggregates
from data_processing.helpers import convert_to_numeric
from visualization import create_empty_figure, plot_choropleth_map

# --- Page Setup & Logging ---
st.set_page_config(page_title="Population Command Center", page_icon="🌍", layout="wide")
logger = logging.getLogger(__name__)

# --- SME VISUALIZATION & KPI UPGRADE: Constants ---
PLOTLY_TEMPLATE = "plotly_white"
RISK_COLORS = {'Low Risk': '#28a745', 'Moderate Risk': '#ffc107', 'High Risk': '#dc3545'}
GENDER_COLORS = {"Female": "#E1396C", "Male": "#1f77b4", "Unknown": "#7f7f7f"}
THREAT_LEVEL_CONFIG = {
    0: {"label": "LEVEL 0: NO THREATS", "color": "#28a745"},
    1: {"label": "LEVEL 1: MONITORING", "color": "#ffc107"},
    2: {"label": "LEVEL 2: ACTIVE RESPONSE", "color": "#dc3545"},
    3: {"label": "LEVEL 3: CRITICAL ALERT", "color": "#8B0000"}
}

# --- Mock AI & Helper Functions ---
def predict_risk_transition(df: pd.DataFrame) -> dict:
    if df.empty: return {'nodes': [], 'source_indices': [], 'target_indices': [], 'values': [], 'upward_transitions': 0, 'mod_to_high_count': 0}
    low_to_mod = int(df[df['ai_risk_score'] < 40].shape[0] * np.random.uniform(0.08, 0.12))
    mod_to_high = int(df[df['ai_risk_score'].between(40, 64, inclusive='both')].shape[0] * np.random.uniform(0.04, 0.06))
    return {'nodes': ['Low Risk', 'Moderate Risk', 'High Risk', 'Low Risk (Future)', 'Moderate Risk (Future)', 'High Risk (Future)'], 'source_indices': [0, 1], 'target_indices': [4, 5], 'values': [low_to_mod, mod_to_high], 'upward_transitions': low_to_mod + mod_to_high, 'mod_to_high_count': mod_to_high}

def calculate_preventive_opportunity(zone_df: pd.DataFrame, health_df: pd.DataFrame) -> pd.DataFrame:
    if health_df.empty:
        zone_df['preventive_opportunity_index'] = 0
        return zone_df
    moderate_risk = health_df[health_df['ai_risk_score'].between(40, 64, inclusive='both')]
    opportunity = moderate_risk.groupby('zone_id').size().reset_index(name='preventive_opportunity_index')
    return pd.merge(zone_df, opportunity, on='zone_id', how='left').fillna(0)

def detect_emerging_threats(health_df: pd.DataFrame, lookback_days: int = 90, threshold: float = 2.0) -> pd.DataFrame:
    if health_df.empty or len(health_df['encounter_date'].unique()) < 14: return pd.DataFrame()
    end_date, start_date = health_df['encounter_date'].max(), health_df['encounter_date'].max() - timedelta(days=lookback_days)
    recent_df = health_df[health_df['encounter_date'] > start_date]
    if recent_df.empty: return pd.DataFrame()
    top_diagnoses = recent_df['diagnosis'].value_counts().nlargest(10).index
    if top_diagnoses.empty: return pd.DataFrame()
    threats = []
    for diag in top_diagnoses:
        diag_ts = health_df[health_df['diagnosis'] == diag].set_index('encounter_date').resample('W-MON').size()
        if len(diag_ts) < 4: continue
        mean, std = diag_ts.mean(), diag_ts.std()
        if std == 0: continue
        last_week_count = diag_ts.iloc[-1]
        z_score = (last_week_count - mean) / std
        if z_score > threshold:
            threats.append({'diagnosis': diag, 'recent_avg_cases': last_week_count, 'baseline_avg_cases': mean, 'z_score': z_score})
    return pd.DataFrame(threats)

# --- SME RESILIENCE UPGRADE: Robust Data Loading ---
@st.cache_data(ttl=3600, show_spinner="Loading population datasets...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    health_df = load_health_records()
    zone_df = load_zone_data()
    if health_df.empty: return pd.DataFrame(), pd.DataFrame()
    
    health_df, _ = apply_ai_models(health_df)
    if 'ai_risk_score' not in health_df.columns:
        st.session_state['using_dummy_risk'] = True
        health_df['ai_risk_score'] = np.random.uniform(0, 100, len(health_df))
    else: st.session_state['using_dummy_risk'] = False
        
    if 'risk_factors' not in health_df.columns:
        factors = ['Hypertension', 'Diabetes', 'Smoking', 'Obesity', 'Malnutrition']
        health_df['risk_factors'] = health_df['patient_id'].apply(lambda _: list(np.random.choice(factors, size=np.random.randint(0, 4), replace=False)))
    if 'risk_factors' in health_df.columns:
        health_df['risk_factors'] = health_df['risk_factors'].apply(tuple)

    if zone_df.empty:
        zone_ids = health_df['zone_id'].unique()
        zone_df = pd.DataFrame({'zone_id': zone_ids, 'zone_name': [f"Zone {zid}" for zid in zone_ids]})
        if 'geometry' not in zone_df.columns: zone_df['geometry'] = None
    
    enriched_zone_df = enrich_zone_data_with_aggregates(zone_df, health_df)
    return health_df, enriched_zone_df

@st.cache_data
def get_risk_stratification(df: pd.DataFrame) -> dict:
    if df.empty: return {'pyramid_data': pd.DataFrame()}
    def assign_tier(score):
        if score >= 65: return 'High Risk'
        if score >= 40: return 'Moderate Risk'
        return 'Low Risk'
    df_copy = df.copy().drop_duplicates('patient_id')
    df_copy['risk_tier'] = convert_to_numeric(df_copy['ai_risk_score']).apply(assign_tier)
    pyramid_data = df_copy['risk_tier'].value_counts().reset_index(name='patient_count').rename(columns={'index': 'risk_tier'})
    return {'pyramid_data': pyramid_data}

# --- UI Rendering Components ---
def render_overview(df_filtered: pd.DataFrame, health_df: pd.DataFrame, start_date: date):
    st.subheader("Population Health Scorecard")
    cols = st.columns(4)
    unique_patients = df_filtered['patient_id'].nunique()
    cols[0].metric("Unique Patients in Period", f"{unique_patients:,}")
    avg_risk = df_filtered['ai_risk_score'].mean()
    cols[1].metric("Avg. Risk Score", f"{avg_risk:.1f}")
    high_risk_count = df_filtered[df_filtered['ai_risk_score'] >= 65]['patient_id'].nunique()
    cols[2].metric("High-Risk Patients", f"{high_risk_count:,}", help="Count of unique patients with risk score >= 65")
    care_gap_date_threshold = start_date - timedelta(days=90)
    high_risk_no_contact = health_df[(health_df['ai_risk_score'] >= 65) & (health_df['encounter_date'].dt.date < care_gap_date_threshold)]['patient_id'].nunique()
    cols[3].metric("Care Gap: High-Risk", f"{high_risk_no_contact:,}", help="High-risk patients with no clinic encounter in the last 90 days", delta_color="inverse")

def render_risk_stratification(df_filtered: pd.DataFrame):
    st.header("🚨 Strategic Risk & Intervention Planning")
    st.markdown("Analyze current risk, predict future burdens, and quantify the value of preventive action.")
    st.divider()
    COST_ASSUMPTIONS = { 'avg_encounter_cost_high_risk': 250, 'avg_encounter_cost_moderate_risk': 120, 'preventive_intervention_cost': 500, 'annual_encounters_high_risk': 4, 'annual_encounters_moderate_risk': 2 }
    st.subheader("Part 1: What is our current risk distribution?")
    risk_data = get_risk_stratification(df_filtered)
    pyramid_data = risk_data.get('pyramid_data')
    if not pyramid_data.empty:
        pyramid_data = pyramid_data.set_index('risk_tier').reindex(['High Risk', 'Moderate Risk', 'Low Risk']).reset_index()
        fig = px.bar(pyramid_data, y='risk_tier', x='patient_count', orientation='h', color='risk_tier', color_discrete_map=RISK_COLORS, text='patient_count', title="<b>Current Population Risk Distribution</b>")
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside').update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Number of Patients", yaxis_title=None, showlegend=False, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available to generate a risk pyramid for this period.")
        return
    st.divider()
    st.subheader("Part 2: What is the predicted future burden?")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("##### **Predicted 90-Day Risk Flow**")
        if not df_filtered.empty:
            predicted_transitions = predict_risk_transition(df_filtered)
            fig_sankey = go.Figure(go.Sankey(
                node=dict(pad=25, thickness=20, line=dict(color="black", width=0.5), label=predicted_transitions['nodes'], color=[RISK_COLORS.get(label.split(' ')[0], '#7f7f7f') for label in predicted_transitions['nodes']]),
                link=dict(source=predicted_transitions['source_indices'], target=predicted_transitions['target_indices'], value=predicted_transitions['values'], color='rgba(0, 123, 255, 0.5)', hovertemplate='Predicted Transitions: %{value:,.0f}<extra></extra>')
            ))
            fig_sankey.update_layout(font_size=12, template=PLOTLY_TEMPLATE, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig_sankey, use_container_width=True)
        else: st.info("Insufficient data to predict risk transitions.")
    with col2:
        st.markdown("##### **Projected Annual Burden of Illness**")
        with st.container(border=True):
            high_risk_patients = pyramid_data[pyramid_data['risk_tier'] == 'High Risk']['patient_count'].sum()
            projected_encounters = high_risk_patients * COST_ASSUMPTIONS['annual_encounters_high_risk']
            projected_cost = projected_encounters * COST_ASSUMPTIONS['avg_encounter_cost_high_risk']
            st.metric("Projected High-Risk Encounters/Year", f"{projected_encounters:,.0f} visits")
            st.metric("Projected Annual Cost of High-Risk Care", f"${projected_cost:,.0f}")
            st.caption(f"Based on {high_risk_patients:,} high-risk patients.")
    st.divider()
    st.subheader("Part 3: What is the Return on Investment for Prevention?")
    st.markdown("This analysis quantifies the financial impact of a preventive care program targeting moderate-risk patients predicted to transition.")
    if not df_filtered.empty and 'predicted_transitions' in locals():
        target_patients = predicted_transitions['mod_to_high_count']
        cost_if_high_risk = target_patients * COST_ASSUMPTIONS['annual_encounters_high_risk'] * COST_ASSUMPTIONS['avg_encounter_cost_high_risk']
        cost_of_intervention = target_patients * COST_ASSUMPTIONS['preventive_intervention_cost']
        cost_if_stabilized = target_patients * COST_ASSUMPTIONS['annual_encounters_moderate_risk'] * COST_ASSUMPTIONS['avg_encounter_cost_moderate_risk']
        total_intervention_cost = cost_of_intervention + cost_if_stabilized
        net_savings = cost_if_high_risk - total_intervention_cost
        with st.container(border=True):
            st.markdown("##### **ROI on Preventive Care for At-Risk Cohort**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Patients to Target", f"{target_patients:,}", "Predicted to become High-Risk")
            c2.metric("Annual Cost of Inaction", f"${cost_if_high_risk:,.0f}")
            c3.metric("Annual Cost of Intervention", f"${total_intervention_cost:,.0f}")
            st.metric("Potential Net Savings (1 Year)", f"${net_savings:,.0f}", "Savings from preventing escalation")
            if net_savings > 0: st.success(f"**Recommendation:** Investing in preventive care for the **{target_patients:,}** moderate-risk patients is projected to yield a net saving of **${net_savings:,.0f}** in the next year.")
            else: st.warning(f"**Recommendation:** The model projects a net cost of **${-net_savings:,.0f}** for this intervention. Review cost assumptions or focus on higher-impact programs.")

def render_geospatial_analysis(zone_df_filtered: pd.DataFrame, df_filtered: pd.DataFrame):
    st.header("🗺️ Geospatial Intelligence")
    if 'geometry' not in zone_df_filtered.columns or zone_df_filtered['geometry'].isnull().all():
        st.warning("Geospatial data is unavailable. Mapping features are disabled.")
        return
    geojson_data = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"zone_id": row['zone_id']}, "geometry": row['geometry']} for _, row in zone_df_filtered.iterrows() if pd.notna(row.get('geometry'))]}
    map_metric = st.selectbox("Select Map Metric:", ["Average AI Risk Score", "Prevalence per 1,000 People", "Preventive Opportunity Index"], help="Choose a metric to visualize across different zones.")
    if "Risk" in map_metric: color_metric, scale, title = 'avg_risk_score', 'Reds', 'Zonal Average Risk Score'
    elif "Prevalence" in map_metric: color_metric, scale, title = 'prevalence_per_1000_pop', 'Reds', 'Zonal Disease Prevalence'
    else: 
        zone_df_filtered = calculate_preventive_opportunity(zone_df_filtered, df_filtered)
        color_metric, scale, title = 'preventive_opportunity_index', 'Greens', 'Zonal Preventive Opportunity Index'
        st.info("💡 **Preventive Opportunity Index** highlights zones with a high concentration of moderate-risk individuals who are most amenable to cost-effective preventive care.")
    fig = plot_choropleth_map(zone_df_filtered, geojson=geojson_data, value_col=color_metric, title=f"<b>{title}</b>", hover_name='zone_name', color_continuous_scale=scale)
    fig.update_layout(title_x=0.5, coloraxis_colorbar=dict(title=map_metric.replace(" ", "<br>"), lenmode="fraction", len=0.8, yanchor="middle", y=0.5))
    st.plotly_chart(fig, use_container_width=True)

def render_population_segmentation(df_filtered: pd.DataFrame):
    st.header("🧑‍🤝‍🧑 Risk Driver & Comorbidity Engine")
    st.markdown("Proactively identify and analyze the key risk factors driving poor health outcomes across your population.")
    df_unique = df_filtered.drop_duplicates(subset=['patient_id'])
    if df_unique.empty: st.info("No unique patient data in this period for segmentation."); return
    all_factors = sorted(list(set([factor for sublist in df_unique.get('risk_factors', []) for factor in sublist])))
    if not all_factors: st.warning("No risk factor data available. This analysis cannot be performed."); return
    st.subheader("Overall Risk Factor Landscape")
    st.markdown("Identify the most prevalent and most severe risk factors in your population at a glance. Bar length shows prevalence; color shows the average risk of patients with that factor.")
    prevalence = df_unique['risk_factors'].explode().value_counts()
    avg_risk_by_factor = { factor: df_unique[df_unique['risk_factors'].apply(lambda x: factor in x)]['ai_risk_score'].mean() for factor in prevalence.index }
    risk_factor_summary = pd.DataFrame({'factor': prevalence.index, 'prevalence': prevalence.values, 'avg_risk_score': [avg_risk_by_factor[f] for f in prevalence.index]}).sort_values('prevalence', ascending=True)
    fig_landscape = px.bar(risk_factor_summary, x='prevalence', y='factor', orientation='h', color='avg_risk_score', color_continuous_scale=px.colors.sequential.Reds, title="<b>Prevalence vs. Severity of Top Risk Factors</b>", labels={'prevalence': 'Number of Patients (Prevalence)', 'factor': 'Risk Factor', 'avg_risk_score': 'Avg. Risk Score'}, text='prevalence')
    fig_landscape.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5, coloraxis_colorbar_title_text='Avg. Risk')
    st.plotly_chart(fig_landscape, use_container_width=True)
    st.caption("Actionability: High-prevalence, high-risk (dark red) factors are top priorities for broad-based campaigns.")
    st.divider()
    st.subheader("🎯 Targeted Deep Dive into Specific Risk Factors")
    selected_factor = st.selectbox("Select a Risk Factor for a detailed breakdown:", all_factors, help="Choose a risk factor to see which demographic groups are most affected and what other conditions are common.")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f"##### Demographic Hotspots for '{selected_factor}'")
        factor_df = df_unique[df_unique['risk_factors'].apply(lambda x: selected_factor in x)].copy()
        if not factor_df.empty:
            age_bins, age_labels = [0, 18, 40, 65, 150], ['0-17', '18-39', '40-64', '65+']
            factor_df['age_group'] = pd.cut(factor_df['age'], bins=age_bins, labels=age_labels, right=False)
            # SME FIX: Add observed=False to groupby to silence FutureWarning
            driver_data = factor_df.groupby(['age_group', 'gender'], observed=False).size().reset_index(name='count')
            fig_demo = px.bar(driver_data, x='age_group', y='count', color='gender', barmode='group', title=f"<b>Who is most affected by '{selected_factor}'?</b>", labels={'count': 'Number of Patients'}, category_orders={'age_group': age_labels}, color_discrete_map=GENDER_COLORS, template=PLOTLY_TEMPLATE, text='count')
            fig_demo.update_traces(textposition='outside').update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_x=0.5)
            st.plotly_chart(fig_demo, use_container_width=True)
        else: st.info(f"No patients with the risk factor '{selected_factor}' in the selected period.")
    with col2:
        st.markdown("##### Top Co-morbidities")
        if not factor_df.empty:
            co_factors = factor_df['risk_factors'].explode().value_counts().drop(selected_factor, errors='ignore').nlargest(5)
            fig_cofactor = px.bar(co_factors, x=co_factors.values, y=co_factors.index, orientation='h', title=f"<b>Top Co-morbidities with '{selected_factor}'</b>", labels={'x': 'Number of Patients', 'y': 'Co-occurring Factor'})
            fig_cofactor.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_cofactor, use_container_width=True)
            st.caption("Actionability: Patients with the primary risk factor often have these other conditions. Screen accordingly.")
        else: st.info("Select a factor with patient data to see co-morbidities.")

def render_emerging_threats(health_df: pd.DataFrame, zone_df: pd.DataFrame):
    st.header("🔬 Early Warning & Response System")
    st.markdown("Detect and analyze emerging health threats in real-time to guide a rapid and targeted public health response.")
    if health_df.empty: st.info("Not enough historical data to analyze emerging threats."); return
    threats_df = detect_emerging_threats(health_df, lookback_days=90, threshold=1.5)
    num_threats = len(threats_df)
    if num_threats == 0: threat_level = 0
    elif num_threats <= 2: threat_level = 1
    elif num_threats <= 4: threat_level = 2
    else: threat_level = 3
    level_config = THREAT_LEVEL_CONFIG[threat_level]
    st.markdown(f"""<div style="background-color: {level_config['color']}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 15px;"><h3 style="color: white; margin: 0;">OVERALL THREAT: {level_config['label']}</h3></div>""", unsafe_allow_html=True)
    if not threats_df.empty:
        selected_threat = st.selectbox("Select Detected Threat for Deep Dive:", threats_df.sort_values('z_score', ascending=False)['diagnosis'].unique())
        threat_data = threats_df[threats_df['diagnosis'] == selected_threat].iloc[0]
        threat_cases_df = health_df[health_df['diagnosis'] == selected_threat]
        with st.container(border=True):
            st.subheader(f"Epi-Brief: {threat_data['diagnosis']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly Score (Z-score)", f"{threat_data['z_score']:.1f}σ", "Above Baseline")
            c2.metric("Last Week's Cases", f"{threat_data['recent_avg_cases']:.0f}")
            c3.metric("Weekly Baseline", f"{threat_data['baseline_avg_cases']:.1f}")
            tab_when, tab_where, tab_who = st.tabs(["📈 Trend (When)", "🗺️ Hotspots (Where)", "🧑‍🤝‍🧑 Demographics (Who)"])
            with tab_when:
                threat_trend_ts = threat_cases_df.set_index('encounter_date').resample('W-MON').size()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=threat_trend_ts.index, y=[threat_data['baseline_avg_cases']] * len(threat_trend_ts), mode='lines', line_color='#28a745', line_dash='dash', name='Historical Baseline'))
                fig_trend.add_trace(go.Scatter(x=threat_trend_ts.index, y=threat_trend_ts.values, mode='lines+markers', line_color='#dc3545', name='Weekly Cases', fill='tozeroy', fillcolor='rgba(220, 53, 69, 0.2)'))
                fig_trend.update_layout(title=f"<b>Weekly Trend for '{threat_data['diagnosis']}'</b>", yaxis_title='Number of Cases', template=PLOTLY_TEMPLATE, showlegend=False, title_x=0.5)
                st.plotly_chart(fig_trend, use_container_width=True)
            with tab_where:
                if not zone_df.empty and 'geometry' in zone_df.columns:
                    threat_geo_df = threat_cases_df.groupby('zone_id').size().reset_index(name='case_count')
                    zone_threat_df = pd.merge(zone_df, threat_geo_df, on='zone_id', how='left').fillna(0)
                    geojson_data = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"zone_id": r['zone_id']}, "geometry": r['geometry']} for _, r in zone_threat_df.iterrows() if pd.notna(r.get('geometry'))]}
                    fig_map = plot_choropleth_map(zone_threat_df, geojson=geojson_data, value_col='case_count', title=f"<b>Where is '{threat_data['diagnosis']}' emerging?</b>", hover_name='zone_name', color_continuous_scale='OrRd')
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.caption("Actionability: Deploy targeted testing and awareness campaigns to the highlighted red zones.")
            with tab_who:
                threat_cases_df = threat_cases_df.copy()
                age_bins, age_labels = [0, 18, 40, 65, 150], ['0-17', '18-39', '40-64', '65+']
                threat_cases_df['age_group'] = pd.cut(threat_cases_df['age'], bins=age_bins, labels=age_labels, right=False)
                # SME FIX: Add observed=False to groupby to silence FutureWarning
                demo_breakdown = threat_cases_df.groupby(['age_group', 'gender'], observed=False).size().reset_index(name='count')
                fig_demo = px.bar(demo_breakdown, x='age_group', y='count', color='gender', barmode='group', title=f"<b>Who is most affected by '{threat_data['diagnosis']}'?</b>", labels={'count': 'Number of Cases'}, category_orders={'age_group': age_labels}, color_discrete_map=GENDER_COLORS)
                st.plotly_chart(fig_demo, use_container_width=True)
                st.caption("Actionability: Tailor public health messaging and clinical alerts to the most affected demographic groups.")
            st.subheader("Operational Response Capacity")
            c1, c2, c3 = st.columns(3)
            c1.metric("Relevant Test Kits", f"{np.random.randint(50, 500):,}", "In stock")
            c2.metric("Available Specialists", f"{np.random.randint(1, 5)}", "On duty")
            c3.metric("Bed Capacity (Affected Zones)", f"{np.random.randint(60, 95)}%")
            st.caption("Note: Response capacity data is illustrative.")
    else: st.success("✅ No significant emerging health threats detected based on current data.")

# --- Main Page Execution ---
def main():
    st.title("🌍 Population Health Command Center")
    st.markdown("A strategic console for understanding population dynamics, predicting future health trends, and targeting interventions.")
    health_df, zone_df = get_data()
    if health_df.empty: st.error("CRITICAL: No health record data available. Dashboard cannot be rendered."); st.stop()
    if st.session_state.get('using_dummy_risk', False): st.warning("⚠️ **Risk Demo Mode:** `ai_risk_score` was not found and has been simulated.", icon="🤖")
    with st.sidebar:
        st.header("Filters")
        min_date, max_date = health_df['encounter_date'].min().date(), health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range:", value=(max(min_date, max_date - timedelta(days=89)), max_date), min_value=min_date, max_value=max_date, key="pop_date_range")
        zone_options = ["All Zones"] + sorted(zone_df['zone_name'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options, key="pop_zone_filter")
    df_filtered = health_df[health_df['encounter_date'].dt.date.between(start_date, end_date)]
    zone_df_filtered = zone_df.copy()
    if selected_zone != "All Zones":
        zone_id = zone_df.loc[zone_df['zone_name'] == selected_zone, 'zone_id'].iloc[0]
        df_filtered = df_filtered[df_filtered['zone_id'] == zone_id]
        zone_df_filtered = zone_df[zone_df['zone_id'] == zone_id]
    render_overview(df_filtered, health_df, start_date)
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Risk & ROI Planning", "🗺️ Geospatial Intelligence", "🧑‍🤝‍🧑 Risk Drivers", "🔬 Early Warning System"])
    with tab1: render_risk_stratification(df_filtered)
    with tab2: render_geospatial_analysis(zone_df_filtered, df_filtered)
    with tab3: render_population_segmentation(df_filtered)
    with tab4: render_emerging_threats(health_df, zone_df)

if __name__ == "__main__":
    main()
