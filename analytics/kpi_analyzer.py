# sentinel_project_root/analytics/kpi_analyzer.py
# SME PLATINUM STANDARD - KPI ANALYSIS & TRENDING

import logging
from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd

# Plotly is an optional dependency for sparkline generation
try:
    import plotly.graph_objects as go
    KALEIDO_INSTALLED = True
except ImportError:
    KALEIDO_INSTALLED = False
    
from config import settings
from data_processing.aggregation import get_cached_clinic_kpis, get_cached_trend

logger = logging.getLogger(__name__)

def _create_sparkline(
    series: pd.Series,
    color: str,
    fill_color: str,
    is_good_change: Optional[bool] = None
) -> Optional[bytes]:
    """Creates a compact sparkline chart as PNG bytes if plotly/kaleido are installed."""
    if not KALEIDO_INSTALLED or not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return None

    # Determine trend line color based on change direction if specified
    if is_good_change is not None:
        final_color = settings.COLOR_DELTA_POSITIVE if is_good_change else settings.COLOR_DELTA_NEGATIVE
    else:
        final_color = color

    fig = go.Figure(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        line=dict(color=final_color, width=3),
        fill='tozeroy',
        fillcolor=fill_color
    ))
    fig.update_layout(
        width=180, height=50,
        margin=dict(l=0, r=0, t=5, b=5),
        xaxis_visible=False, yaxis_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    try:
        # Use kaleido to convert figure to a static image (PNG)
        return fig.to_image(format="png", engine="kaleido", scale=2)
    except Exception as e:
        logger.warning(f"Could not generate sparkline. Is 'kaleido' installed? Error: {e}")
        return None

def generate_kpi_analysis_table(
    full_df: pd.DataFrame,
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Performs a period-over-period KPI analysis, calculates change, and generates trend sparklines.
    Expects a DataFrame with pre-calculated boolean flags from enrichment.
    """
    if full_df.empty or 'encounter_date' not in full_df.columns:
        return pd.DataFrame()

    # Define current and previous periods
    current_period_df = full_df[full_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_days = max((end_date - start_date).days, 0)
    prev_start_date = start_date - timedelta(days=period_days + 1)
    prev_end_date = start_date - timedelta(days=1)
    previous_period_df = full_df[full_df['encounter_date'].dt.date.between(prev_start_date, prev_end_date)]

    # Get KPIs for both periods
    kpi_current = get_cached_clinic_kpis(current_period_df)
    kpi_previous = get_cached_clinic_kpis(previous_period_df)

    # Define KPIs to analyze
    # (kpi_key, trend_column, trend_agg_func, higher_is_better)
    kpi_definitions: Dict[str, Tuple[str, str, str, bool]] = {
        "Avg. Test TAT (Days)": ("avg_test_tat_days", "test_turnaround_days", "mean", False),
        "% Tests in Target": ("perc_tests_within_tat", "test_turnaround_days", lambda s: (s <= 2).mean() * 100, True),
        "Sample Rejection (%)": ("sample_rejection_rate_perc", "is_rejected", "mean", False),
        "Pending Critical Tests": ("pending_critical_tests_count", "is_critical_and_pending", "sum", False),
        "Key Supply At Risk": ("key_items_at_risk_count", "is_supply_at_risk", "sum", False),
    }
    
    analysis_data = []
    # Use a 90-day window for trend calculation for smoother lines
    trend_df_subset = full_df[full_df['encounter_date'].dt.date >= (end_date - timedelta(days=90))]

    for name, (kpi_key, trend_col, trend_agg, higher_is_better) in kpi_definitions.items():
        current_val = kpi_current.get(kpi_key)
        prev_val = kpi_previous.get(kpi_key)
        
        # Calculate percentage change
        change_str = "N/A"
        is_good_change = None
        if pd.notna(current_val) and pd.notna(prev_val) and prev_val != 0:
            change = ((current_val - prev_val) / abs(prev_val)) * 100
            change_str = f"{change:+.1f}%"
            is_good_change = (change > 0) if higher_is_better else (change < 0)

        # Generate trend sparkline
        trend_series = get_cached_trend(df=trend_df_subset, value_col=trend_col, date_col='encounter_date', freq='W', agg_func=trend_agg)
        if trend_agg == "mean" and trend_col == "is_rejected": trend_series *= 100 # scale rate to percentage
            
        sparkline_bytes = _create_sparkline(
            trend_series,
            color=settings.COLOR_PRIMARY,
            fill_color=settings.COLOR_ACCENT + '33', # Add alpha for fill
            is_good_change=is_good_change
        )
        
        analysis_data.append({
            "Metric": name,
            "Current": current_val,
            "Previous": prev_val,
            "Change": change_str,
            "Trend (90d)": sparkline_bytes,
        })
        
    return pd.DataFrame(analysis_data)
