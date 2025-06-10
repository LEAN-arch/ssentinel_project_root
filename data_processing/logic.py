df_trend = df[[date_col, value_col]].copy()
df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')

# --- SME FIX: Only convert to numeric for numeric aggregation functions ---
numeric_aggs = ['mean', 'sum', 'median', 'std', 'var']
if isinstance(agg_func, str) and agg_func in numeric_aggs:
    df_trend[value_col] = convert_to_numeric(df_trend[value_col])

# This will now correctly handle strings for 'nunique' and 'count'
# without converting them to NaN.
df_trend.dropna(subset=[date_col, value_col], inplace=True)

if df_trend.empty:
    return pd.Series(dtype=np.float64)
    
try:
    trend_series = df_trend.set_index(date_col)[value_col].resample(freq).agg(agg_func)
    if isinstance(agg_func, str) and agg_func in ['count', 'size', 'nunique', 'sum']:
        trend_series = trend_series.fillna(0).astype(int)
    return trend_series
except Exception as e:
    logger.error(f"Error generating trend for '{value_col}': {e}", exc_info=True)
    return pd.Series(dtype=np.float64)
