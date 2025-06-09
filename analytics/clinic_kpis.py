# sentinel_project_root/analytics/clinic_kpis.py
# SME PLATINUM STANDARD (V1 - NEW MODULE)
# This new module encapsulates all business logic for calculating the Clinic Dashboard's KPI analysis table.

import pandas as pd
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any

# Use a try-except block for Plotly for environments where it might not be needed (e.g., backend processing)
try:
    import plotly.graph_objects as go
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

# <<< SME INTEGRATION >>> Use the new Pydantic settings object.
from config import settings
from data_processing.aggregation import get_clinic_summary_kpis, get_trend_data

logger = logging.getLogger(__name__)

def _create_sparkline_bytes(data: pd.Series, color: str) -> Optional[bytes]:
    """Creates a compact sparkline chart and returns it as PNG bytes if plotly/kaleido are installed."""
    if not KALEIDO_AVAILABLE or data is None or data.empty or data.isna().all():
        return None
    # ... (function code is identical to previous review)
    fig = go.Figure(go.Scatter(x=data.index, y=data, mode='lines', line=dict(color=color, width=2.5), fill='tozeroy'))
    fig.update_layout(width=150, height=50, margin=dict(l=0, r=0, t=5, b=5), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    try:
        return fig.to_image(format="png", engine="kaleido")
    except Exception as e:
        logger.warning(f"Could not generate sparkline image. Is 'kaleido' installed? Error: {e}")
        return None

def generate_kpi_analysis_table(full_df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Performs a period-over-period KPI analysis, calculates change, and generates trend sparklines.
    ASSUMPTION: The input `full_df` has been pre-processed to include boolean flag columns.
    """
    if full_df.empty: return pd.DataFrame()

    current_period_df = full_df[full_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_days = max((end_date - start_date).days + 1, 1)
    prev_start_date = start_date - timedelta(days=period_days)
    previous_period_df = full_df[full_df['encounter_date'].dt.date.between(prev_start_date, prev_start_date + timedelta(days=period_days-1))]

    kpi_current = get_clinic_summary_kpis(current_period_df)
    kpi_previous = get_clinic_summary_kpis(previous_period_df)

    kpi_defs = {
        "Avg. Test TAT (Days)": ("overall_avg_test_turnaround_conclusive_days", "test_turnaround_conclusive_days", "mean"),
        "Sample Rejection (%)": ("sample_rejection_rate_perc", "is_rejected", "mean"),
        "Pending Critical Tests": ("total_pending_critical_tests_patients", "is_critical_and_pending", "sum"),
        "Key Drug Stockouts": ("key_drug_stockouts_count", "is_stockout", "sum")
    }
    
    analysis_data, trend_df_subset = [], full_df[full_df['encounter_date'].dt.date >= (end_date - timedelta(days=90))]

    for name, (kpi_key, trend_col, trend_agg) in kpi_defs.items():
        current_val, prev_val = kpi_current.get(kpi_key), kpi_previous.get(kpi_key)
        change_str = "N/A"
        if pd.notna(current_val) and pd.notna(prev_val) and prev_val != 0:
            change_str = f"{((current_val - prev_val) / prev_val) * 100:+.1f}%"
        
        trend_series = get_trend_data(trend_df_subset, value_col=trend_col, period='W-MON', agg_func=trend_agg)
        if kpi_key == "sample_rejection_rate_perc": trend_series *= 100

        analysis_data.append({
            "Metric": name, "Current Period": current_val, "Previous Period": prev_val,
            "Change": change_str, "90-Day Trend": _create_sparkline_bytes(trend_series, settings.COLOR_ACTION_PRIMARY)
        })
        
    return pd.DataFrame(analysis_data)
