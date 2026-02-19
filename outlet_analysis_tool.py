import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize
import os

# Page config
st.set_page_config(page_title="Outlet Analysis Tool", layout="wide", page_icon="")

# Load data with caching
@st.cache_data
def load_parquet_files():
    """Load all parquet files from step3_filtered_engineered folder"""
    try:
        script_dir = Path(__file__).resolve().parent
        candidate_paths = [
            script_dir / "step3_filtered_engineered",
            Path.cwd() / "step3_filtered_engineered",
            Path.cwd() / "final_prep" / "step3_filtered_engineered"
        ]
        folder_path = next((p for p in candidate_paths if p.exists()), None)
        
        if folder_path is None:
            checked = "\n".join([f"- {p}" for p in candidate_paths])
            st.error(
                "Could not find 'step3_filtered_engineered' folder. Checked paths:\n"
                f"{checked}"
            )
            return None

        parquet_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.parquet')])
        
        if not parquet_files:
            st.error("No parquet files found in step3_filtered_engineered folder")
            return None
        
        # Load all files and combine
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(folder_path / file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Ensure Date is datetime
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        
        return combined_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def calculate_rfm(df, recency_days=90, frequency_order_threshold=20):
    """Calculate RFM metrics with clustering."""
    
    # Get max date in dataset
    max_date = df['Date'].max()
    
    # Create order-level data (aggregate by Outlet, Date, Invoice)
    order_level = df.groupby(['Outlet_ID', 'Date', 'Bill_No', 'Final_State']).agg({
        'Quantity': 'sum',
        'Net_Amt': 'sum'
    }).reset_index()
    
    # Calculate RFM at outlet level
    rfm = order_level.groupby(['Outlet_ID', 'Final_State']).agg({
        'Date': ['min', 'max', 'nunique'],
        'Bill_No': 'count',  # Total orders count (kept for reporting)
        'Net_Amt': 'mean'    # Monetary: average order value (AOV)
    }).reset_index()
    
    # Flatten column names
    rfm.columns = ['Outlet_ID', 'Final_State', 'first_order', 'last_order', 'unique_order_days', 'orders_count', 'AOV']
    
    # Calculate Recency
    rfm['Recency_days'] = (max_date - rfm['last_order']).dt.days
    rfm['Recency_flag'] = (rfm['Recency_days'] <= recency_days).astype(int)
    rfm['R_label'] = rfm['Recency_flag'].map({1: 'Recent', 0: 'Stale'})

    # Frequency rate to avoid penalizing outlets with shorter history.
    # orders_per_day = orders_count / active_days, where active_days = last_order - first_order + 1.
    rfm['active_days'] = (rfm['last_order'] - rfm['first_order']).dt.days + 1
    rfm['active_days'] = rfm['active_days'].clip(lower=1)
    rfm['orders_per_day'] = rfm['orders_count'] / rfm['active_days']

    # Fixed frequency rule requested:
    # High if outlet has >= threshold unique order dates in filtered period, else Low.
    rfm['F_label'] = np.where(rfm['unique_order_days'] >= int(frequency_order_threshold), 'High', 'Low')
    rfm['F_cluster_id'] = np.where(rfm['unique_order_days'] >= int(frequency_order_threshold), 1, 0)
    
    # Cluster Monetary (M) using K-means
    valid_m = rfm['AOV'].notna()
    if valid_m.sum() >= 10:
        m_values = np.log1p(rfm.loc[valid_m, 'AOV'].values).reshape(-1, 1)
        scaler_m = StandardScaler()
        m_scaled = scaler_m.fit_transform(m_values)
        
        kmeans_m = KMeans(n_clusters=2, random_state=42, n_init=20)
        m_clusters = kmeans_m.fit_predict(m_scaled)
        
        cluster_means = [rfm.loc[valid_m, 'AOV'][m_clusters == i].mean() for i in range(2)]
        high_cluster_m = np.argmax(cluster_means)
        
        rfm.loc[valid_m, 'M_cluster_id'] = m_clusters
        rfm.loc[valid_m, 'M_label'] = ['High' if c == high_cluster_m else 'Low' for c in m_clusters]
    else:
        median_m = rfm['AOV'].median()
        rfm['M_label'] = rfm['AOV'].apply(lambda x: 'High' if x >= median_m else 'Low')
        rfm['M_cluster_id'] = np.nan
    
    # Keep monetary semantic consistency only.
    m_means = rfm.groupby('M_label', dropna=False)['AOV'].mean()
    if 'High' in m_means.index and 'Low' in m_means.index:
        if pd.notna(m_means['High']) and pd.notna(m_means['Low']) and m_means['High'] < m_means['Low']:
            rfm['M_label'] = rfm['M_label'].replace({'High': 'Low', 'Low': 'High'})

    # Create RFM Segment (2x2x2 = 8 segments)
    rfm['RFM_Segment'] = rfm['R_label'] + '-' + rfm['F_label'] + '-' + rfm['M_label']

    # Cluster summary tables for UI/debug
    freq_cluster_summary = (
        rfm.groupby('F_label', dropna=False)['orders_per_day']
        .agg(['count', 'min', 'max', 'mean'])
        .reset_index()
        .rename(columns={
            'F_label': 'Frequency_Cluster',
            'count': 'Outlets',
            'min': 'Min_Orders_Per_Day',
            'max': 'Max_Orders_Per_Day',
            'mean': 'Mean_Orders_Per_Day'
        })
        .sort_values('Frequency_Cluster')
    )

    monetary_cluster_summary = (
        rfm.groupby('M_label', dropna=False)['AOV']
        .agg(['count', 'min', 'max', 'mean'])
        .reset_index()
        .rename(columns={
            'M_label': 'Monetary_Cluster',
            'count': 'Outlets',
            'min': 'Min_AOV',
            'max': 'Max_AOV',
            'mean': 'Mean_AOV'
        })
        .sort_values('Monetary_Cluster')
    )

    cluster_summary = {
        'frequency': freq_cluster_summary,
        'monetary': monetary_cluster_summary
    }

    return rfm, max_date, cluster_summary


def backfill_rfm_fields(rfm):
    """Backfill RFM columns for backward compatibility with older session/cache data."""
    rfm = rfm.copy()

    if 'active_days' not in rfm.columns and {'first_order', 'last_order'}.issubset(rfm.columns):
        first_order = pd.to_datetime(rfm['first_order'], errors='coerce')
        last_order = pd.to_datetime(rfm['last_order'], errors='coerce')
        rfm['active_days'] = ((last_order - first_order).dt.days + 1).clip(lower=1)

    if 'orders_per_day' not in rfm.columns and 'orders_count' in rfm.columns:
        if 'active_days' in rfm.columns:
            rfm['orders_per_day'] = rfm['orders_count'] / rfm['active_days'].replace(0, np.nan)
        else:
            # Fallback for very old payloads: keep behavior safe and non-breaking.
            rfm['orders_per_day'] = rfm['orders_count']

    # Enforce current fixed frequency rule for backward-compatible payloads too.
    if 'unique_order_days' in rfm.columns:
        freq_base = pd.to_numeric(rfm['unique_order_days'], errors='coerce').fillna(0)
    elif 'orders_count' in rfm.columns:
        # Fallback for very old payloads that don't have unique_order_days.
        freq_base = pd.to_numeric(rfm['orders_count'], errors='coerce').fillna(0)
    else:
        freq_base = pd.Series(0, index=rfm.index, dtype=float)
    rfm['F_label'] = np.where(freq_base >= 20, 'High', 'Low')
    rfm['F_cluster_id'] = np.where(freq_base >= 20, 1, 0)

    if {'M_label', 'AOV'}.issubset(rfm.columns):
        m_means = rfm.groupby('M_label', dropna=False)['AOV'].mean()
        if 'High' in m_means.index and 'Low' in m_means.index:
            if pd.notna(m_means['High']) and pd.notna(m_means['Low']) and m_means['High'] < m_means['Low']:
                rfm['M_label'] = rfm['M_label'].replace({'High': 'Low', 'Low': 'High'})

    if {'R_label', 'F_label', 'M_label'}.issubset(rfm.columns):
        rfm['RFM_Segment'] = rfm['R_label'] + '-' + rfm['F_label'] + '-' + rfm['M_label']

    return rfm


def build_cluster_summary_from_rfm(rfm):
    """Build frequency/monetary cluster range tables from current RFM dataframe."""
    rfm = backfill_rfm_fields(rfm)

    if {'F_label', 'orders_per_day'}.issubset(rfm.columns):
        freq_cluster_summary = (
            rfm.groupby('F_label', dropna=False)['orders_per_day']
            .agg(['count', 'min', 'max', 'mean'])
            .reset_index()
            .rename(columns={
                'F_label': 'Frequency_Cluster',
                'count': 'Outlets',
                'min': 'Min_Orders_Per_Day',
                'max': 'Max_Orders_Per_Day',
                'mean': 'Mean_Orders_Per_Day'
            })
            .sort_values('Frequency_Cluster')
        )
    else:
        freq_cluster_summary = pd.DataFrame(
            columns=['Frequency_Cluster', 'Outlets', 'Min_Orders_Per_Day', 'Max_Orders_Per_Day', 'Mean_Orders_Per_Day']
        )

    if {'M_label', 'AOV'}.issubset(rfm.columns):
        monetary_cluster_summary = (
            rfm.groupby('M_label', dropna=False)['AOV']
            .agg(['count', 'min', 'max', 'mean'])
            .reset_index()
            .rename(columns={
                'M_label': 'Monetary_Cluster',
                'count': 'Outlets',
                'min': 'Min_AOV',
                'max': 'Max_AOV',
                'mean': 'Mean_AOV'
            })
            .sort_values('Monetary_Cluster')
        )
    else:
        monetary_cluster_summary = pd.DataFrame(
            columns=['Monetary_Cluster', 'Outlets', 'Min_AOV', 'Max_AOV', 'Mean_AOV']
        )

    return {'frequency': freq_cluster_summary, 'monetary': monetary_cluster_summary}

def aggregate_by_time_period(df, time_agg):
    """
    Aggregate transaction data by time period (Daily, Weekly, Monthly)
    
    Parameters:
    - df: DataFrame with transaction data
    - time_agg: 'Daily', 'Weekly', or 'Monthly'
    
    Returns:
    - Aggregated DataFrame with time period column
    """
    # Shallow copy avoids huge memory spikes on large datasets.
    df = df.copy(deep=False)
    
    if time_agg == 'Daily':
        # Group by Date
        group_col = 'Date'
        df['Period'] = df['Date']
        
    elif time_agg == 'Weekly':
        # Group by year-week
        df['Period'] = df['Date'].dt.to_period('W').apply(lambda x: x.start_time)
        group_col = 'Period'
        
    elif time_agg == 'Monthly':
        # Group by year-month
        df['Period'] = df['Date'].dt.to_period('M').apply(lambda x: x.start_time)
        group_col = 'Period'
    
    # Aggregate
    agg_df = df.groupby(group_col).agg({
        'Bill_No': 'nunique',
        'Quantity': 'sum',
        'Net_Amt': 'sum',
        'TotalDiscount': 'sum',
        'SalesValue_atBasicRate': 'sum',
        'Store_ID': 'nunique'  # Add Store_Count
    }).reset_index()
    
    agg_df.columns = ['Period', 'Orders', 'Quantity', 'Net_Amt', 'TotalDiscount', 'SalesValue_atBasicRate', 'Store_Count']
    
    # Calculate Discount %
    agg_df['Discount_Pct'] = (agg_df['TotalDiscount'] / agg_df['SalesValue_atBasicRate'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    return agg_df.sort_values('Period', ascending=False)

def get_slab_baseline_discount(slab):
    """Map slab to baseline discount percentage."""
    slab_key = str(slab).lower()
    if slab_key == "slab0":
        return 8.0
    if slab_key == "slab1":
        return 11.0
    return 14.0

def aggregate_for_roi(df, time_agg):
    """Aggregate transactions for two-stage OLS + ROI calculations."""
    # Shallow copy avoids huge memory spikes on large datasets.
    df = df.copy(deep=False)
    
    if time_agg == 'Daily':
        group_col = 'Date'
        df['Period'] = df['Date']
    elif time_agg == 'Weekly':
        df['Period'] = df['Date'].dt.to_period('W').apply(lambda x: x.start_time)
        group_col = 'Period'
    elif time_agg == 'Monthly':
        df['Period'] = df['Date'].dt.to_period('M').apply(lambda x: x.start_time)
        group_col = 'Period'
    else:
        group_col = 'Date'
        df['Period'] = df['Date']
    
    agg_df = df.groupby(group_col).agg({
        'Store_ID': 'nunique',
        'Bill_No': 'nunique',
        'Quantity': 'sum',
        'Net_Amt': 'sum',
        'TotalDiscount': 'sum',
        'SalesValue_atBasicRate': 'sum'
    }).reset_index()
    
    agg_df.columns = [
        'Period', 'Store_Count', 'Orders', 'Quantity', 'Net_Amt',
        'TotalDiscount', 'SalesValue_atBasicRate'
    ]
    agg_df['Base_Price'] = (
        agg_df['SalesValue_atBasicRate'] / agg_df['Quantity'].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    agg_df['Discount_Pct'] = (agg_df['TotalDiscount'] / agg_df['SalesValue_atBasicRate'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    return agg_df.sort_values('Period', ascending=False)


def _safe_coef_standard_errors(X_with_intercept, mse):
    """Return coefficient SEs with pseudoinverse fallback for singular matrices."""
    xtx = X_with_intercept.T @ X_with_intercept
    used_pinv = False
    try:
        inv_xtx = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        inv_xtx = np.linalg.pinv(xtx)
        used_pinv = True

    var_coef = mse * np.diag(inv_xtx)
    var_coef = np.where(var_coef < 0, np.nan, var_coef)
    se_coef = np.sqrt(var_coef)
    return se_coef, used_pinv


def _safe_t_stat(coef, se):
    if pd.isna(se) or se <= 0:
        return np.nan
    return coef / se


def _safe_two_tailed_pvalue(t_value, dof):
    if pd.isna(t_value) or dof <= 0:
        return np.nan
    return 2 * (1 - stats.t.cdf(abs(t_value), dof))


def round_discount_for_display(series, step=0.5):
    """Round discount percentages for smoother chart display only."""
    s = pd.Series(series, copy=False)
    return (np.round(s.astype(float) / step) * step).astype(float)


def _validate_discount_shift_up(future_values, current_base, threshold_pp, min_share=0.6):
    """Validate sustained upward shift in discount regime."""
    fut = pd.Series(future_values).dropna().astype(float)
    if fut.empty:
        return False
    needed = int(np.ceil(len(fut) * min_share))
    return (fut >= (current_base + threshold_pp * 0.6)).sum() >= needed


def _validate_discount_shift_down(future_values, current_base, threshold_pp, min_share=0.6):
    """Validate sustained downward shift in discount regime."""
    fut = pd.Series(future_values).dropna().astype(float)
    if fut.empty:
        return False
    needed = int(np.ceil(len(fut) * min_share))
    return (fut <= (current_base - threshold_pp * 0.6)).sum() >= needed


def estimate_base_discount_series(
    discount_series,
    rolling_periods=12,
    validation_periods=8,
    up_threshold_pp=1.0,
    down_threshold_pp=1.0,
    percentile=35.0,
    cooldown_periods=None
):
    """
    Estimate base discount with transition logic and validation.
    Returns:
    - base_discount (np.ndarray)
    - is_transition (np.ndarray[bool])
    """
    s = pd.Series(discount_series, copy=False).astype(float).replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return np.array([]), np.array([], dtype=bool)

    s = s.interpolate(limit_direction='both').bfill().ffill()
    arr = s.to_numpy()
    n = len(arr)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    rolling_periods = max(3, min(int(rolling_periods), n))
    validation_periods = max(3, int(validation_periods))
    cooldown = int(cooldown_periods) if cooldown_periods is not None else rolling_periods
    cooldown = max(1, cooldown)

    base = np.empty(n, dtype=float)
    transitions = np.zeros(n, dtype=bool)

    current_base = float(np.nanpercentile(arr[:rolling_periods], percentile))
    last_transition = -cooldown
    max_step_per_period = 0.15

    for i in range(n):
        window_start = max(0, i - rolling_periods + 1)
        history = arr[window_start:i + 1]
        candidate = float(np.nanpercentile(history, percentile))
        future = arr[i:min(n, i + validation_periods)]

        shifted = False
        if len(future) >= validation_periods and (i - last_transition) >= cooldown:
            if (
                candidate >= current_base + up_threshold_pp and
                _validate_discount_shift_up(future, current_base, up_threshold_pp)
            ):
                current_base = candidate
                transitions[i] = True
                last_transition = i
                shifted = True
            elif (
                candidate <= current_base - down_threshold_pp and
                _validate_discount_shift_down(future, current_base, down_threshold_pp)
            ):
                current_base = candidate
                transitions[i] = True
                last_transition = i
                shifted = True

        if not shifted:
            # Gentle drift to adapt slowly and avoid overreacting to noise.
            delta = np.clip(candidate - current_base, -max_step_per_period, max_step_per_period)
            current_base = current_base + delta

        base[i] = np.clip(current_base, 0.0, 100.0)

    return base, transitions


def estimate_base_discount_daily_blocks(
    period_series,
    discount_series,
    min_upward_jump_pp=1.0,
    min_downward_drop_pp=1.0,
    round_step=0.5
):
    """
    Daily-only base discount:
    - one constant base per calendar month
    - split month into 3 segments: days 1-10, 11-20, 21-end
    - candidate = minimum of the 3 segment medians
    - upward month-to-month jump allowed only if >= min_upward_jump_pp
    - downward month-to-month drop allowed only if >= min_downward_drop_pp
    - discounts are rounded to 0.5 steps before estimation
    """
    periods_raw = pd.to_datetime(pd.Series(period_series), errors='coerce')
    discounts_raw = pd.Series(discount_series).astype(float).replace([np.inf, -np.inf], np.nan)
    input_len = len(periods_raw)

    valid = periods_raw.notna() & discounts_raw.notna()
    if valid.sum() == 0:
        return np.array([]), np.array([], dtype=bool)

    # Sort valid rows by calendar date so 1-10 / 11-20 / 21-end slices are true day-order slices.
    valid_positions = np.where(valid.to_numpy())[0]
    work = pd.DataFrame({
        'Orig_Pos': valid_positions,
        'Period': periods_raw[valid].to_numpy(),
        'Discount': discounts_raw[valid].to_numpy(dtype=float)
    }).sort_values('Period', kind='stable').reset_index(drop=True)

    periods = pd.to_datetime(work['Period'], errors='coerce')
    discounts = pd.Series(work['Discount'], dtype=float)
    discounts = discounts.interpolate(limit_direction='both').bfill().ffill()
    discounts = round_discount_for_display(discounts, step=round_step)

    n = len(discounts)
    base = np.empty(n, dtype=float)
    transitions = np.zeros(n, dtype=bool)

    min_upward_jump_pp = max(0.0, float(min_upward_jump_pp))
    min_downward_drop_pp = max(0.0, float(min_downward_drop_pp))
    block_id = periods.dt.to_period('M')

    def _seg_median(vals):
        return float(np.nanmedian(vals)) if len(vals) > 0 else np.nan

    prev_base = None
    for bid in sorted(block_id.unique()):
        mask = (block_id == bid).to_numpy()
        block_vals = discounts[mask].to_numpy(dtype=float)

        # Remove only true outliers within month before segment medians.
        valid_vals = block_vals[~np.isnan(block_vals)]
        if len(valid_vals) >= 8:
            q1, q3 = np.nanquantile(valid_vals, [0.25, 0.75])
            iqr = q3 - q1
            if np.isfinite(iqr) and iqr > 0:
                lo_fence = q1 - 1.5 * iqr
                hi_fence = q3 + 1.5 * iqr
                block_vals = np.where(
                    (block_vals < lo_fence) | (block_vals > hi_fence),
                    np.nan,
                    block_vals
                )

        seg1 = block_vals[:10]
        seg2 = block_vals[10:20]
        seg3 = block_vals[20:]
        medians = [m for m in [_seg_median(seg1), _seg_median(seg2), _seg_median(seg3)] if np.isfinite(m)]
        candidate = float(np.min(medians)) if medians else np.nan

        if prev_base is None:
            block_base = candidate
        else:
            if not np.isfinite(candidate):
                candidate = prev_base
            if candidate > prev_base and (candidate - prev_base) < min_upward_jump_pp:
                block_base = prev_base
            elif candidate < prev_base and (prev_base - candidate) < min_downward_drop_pp:
                block_base = prev_base
            else:
                block_base = candidate

            first_idx = int(np.where(mask)[0][0])
            if abs(block_base - prev_base) > 1e-9:
                transitions[first_idx] = True

        base[mask] = np.clip(block_base, 0.0, 100.0)
        prev_base = block_base

    # Re-expand to original length/order.
    full_base = np.full(input_len, np.nan, dtype=float)
    full_transitions = np.zeros(input_len, dtype=bool)
    full_base[work['Orig_Pos'].to_numpy(dtype=int)] = base
    full_transitions[work['Orig_Pos'].to_numpy(dtype=int)] = transitions

    return full_base, full_transitions

def run_ols_regression(x, y, x_name="X", y_name="Y", periods=None):
    """
    Run OLS regression: Y = 0 + 1*X
    
    Parameters:
    - x: Independent variable (e.g., Discount %)
    - y: Dependent variable (e.g., Quantity)
    - x_name: Name of X variable for display
    - y_name: Name of Y variable for display
    - periods: Optional Series of period/date labels
    
    Returns:
    - Dictionary with regression results
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Also filter periods if provided
    if periods is not None:
        periods_clean = periods[mask]
    else:
        periods_clean = None
    
    if len(x_clean) < 3:
        return None
    
    # Reshape for sklearn
    X = x_clean.values.reshape(-1, 1)
    Y = y_clean.values
    
    # Fit OLS model
    model = LinearRegression()
    model.fit(X, Y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Calculate statistics
    n = len(Y)
    k = 1  # number of predictors
    
    # R-squared
    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Adjusted R-squared
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else 0
    
    # Standard error
    mse = ss_res / (n - k - 1) if n > k + 1 else 0
    se = np.sqrt(mse)
    
    # Coefficient standard error
    X_with_intercept = np.column_stack([np.ones(n), X])
    se_coef, used_pinv = _safe_coef_standard_errors(X_with_intercept, mse)
    
    # T-statistics and p-values
    dof = n - k - 1
    t_intercept = _safe_t_stat(model.intercept_, se_coef[0])
    t_slope = _safe_t_stat(model.coef_[0], se_coef[1])
    
    p_intercept = _safe_two_tailed_pvalue(t_intercept, dof)
    p_slope = _safe_two_tailed_pvalue(t_slope, dof)
    
    # F-statistic
    f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1)) if (1 - r_squared) > 0 and n > k + 1 else 0
    f_pvalue = 1 - stats.f.cdf(f_stat, k, n - k - 1) if f_stat > 0 else 1
    
    return {
        'intercept': model.intercept_,
        'slope': model.coef_[0],
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'se': se,
        'se_intercept': se_coef[0],
        'se_slope': se_coef[1],
        't_intercept': t_intercept,
        't_slope': t_slope,
        'p_intercept': p_intercept,
        'p_slope': p_slope,
        'f_stat': f_stat,
        'f_pvalue': f_pvalue,
        'n': n,
        'x_name': x_name,
        'y_name': y_name,
        'y_pred': y_pred,
        'x_clean': x_clean,
        'y_clean': y_clean,
        'periods_clean': periods_clean,
        'used_pinv_for_se': used_pinv
    }


class CustomConstrainedRidge:
    """Ridge regression with sign-constrained coefficients."""

    def __init__(self, l2_penalty=1.0, non_negative_indices=None, non_positive_indices=None, maxiter=2000):
        self.l2_penalty = float(l2_penalty)
        self.non_negative_indices = tuple(non_negative_indices or [])
        self.non_positive_indices = tuple(non_positive_indices or [])
        self.maxiter = int(maxiter)
        self.intercept_ = 0.0
        self.coef_ = None
        self.n_features_in_ = 0
        self.success_ = False
        self.message_ = ""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std = np.where(x_std <= 1e-12, 1.0, x_std)
        Xs = (X - x_mean) / x_std

        y_mean = float(np.mean(y))
        theta0 = np.zeros(p + 1, dtype=float)
        theta0[0] = y_mean

        bounds = [(None, None)] * (p + 1)
        for idx in self.non_negative_indices:
            if 0 <= idx < p:
                bounds[idx + 1] = (0.0, None)
        for idx in self.non_positive_indices:
            if 0 <= idx < p:
                bounds[idx + 1] = (None, 0.0)

        lam = self.l2_penalty

        def obj(theta):
            b0 = theta[0]
            w = theta[1:]
            resid = y - (Xs @ w + b0)
            return float(np.mean(resid * resid) + lam * np.sum(w * w))

        def grad(theta):
            b0 = theta[0]
            w = theta[1:]
            resid = (Xs @ w + b0) - y
            g_b0 = 2.0 * np.mean(resid)
            g_w = 2.0 * (Xs.T @ resid) / n + 2.0 * lam * w
            return np.concatenate([[g_b0], g_w])

        res = minimize(
            obj,
            theta0,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.maxiter}
        )
        self.success_ = bool(res.success)
        self.message_ = str(res.message)

        theta = np.asarray(res.x, dtype=float)
        b0_s = float(theta[0])
        w_s = theta[1:]

        coef = w_s / x_std
        intercept = b0_s - float(np.sum((w_s * x_mean) / x_std))

        if self.non_negative_indices:
            coef[list(self.non_negative_indices)] = np.maximum(coef[list(self.non_negative_indices)], 0.0)
        if self.non_positive_indices:
            coef[list(self.non_positive_indices)] = np.minimum(coef[list(self.non_positive_indices)], 0.0)

        self.intercept_ = float(intercept)
        self.coef_ = coef.astype(float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def run_two_stage_ols(discount_pct, quantity, store_count, periods=None, base_discount_pct=None):
    """
    Run two-stage model:
    - Stage 1: OLS (Store_Count ~ Actual_Discount)
    - Stage 2: Custom Constrained Ridge
      Quantity ~ Residual(Store) + Structural_Discount + Tactical_Discount + Lag1_Structural_Discount
    """
    disc_s = pd.Series(discount_pct, copy=False).astype(float)
    qty_s = pd.Series(quantity, copy=False).astype(float)
    store_s = pd.Series(store_count, copy=False).astype(float)

    if base_discount_pct is None:
        base_s = pd.Series(np.zeros(len(disc_s)), index=disc_s.index, dtype=float)
    elif np.isscalar(base_discount_pct):
        base_s = pd.Series(np.full(len(disc_s), float(base_discount_pct)), index=disc_s.index, dtype=float)
    else:
        base_s = pd.Series(base_discount_pct, copy=False).astype(float)

    if len(base_s) != len(disc_s):
        return None

    mask = disc_s.notna() & qty_s.notna() & store_s.notna() & base_s.notna()
    if mask.sum() < 3:
        return None

    disc_clean = disc_s[mask].to_numpy(dtype=float)
    qty_clean = qty_s[mask].to_numpy(dtype=float)
    store_clean = store_s[mask].to_numpy(dtype=float)
    base_clean = base_s[mask].to_numpy(dtype=float)

    if periods is not None:
        periods_s = pd.Series(periods)
        periods_clean = periods_s[mask].reset_index(drop=True) if len(periods_s) == len(mask) else None
        if periods_clean is not None:
            period_dt = pd.to_datetime(periods_clean, errors='coerce')
            order = np.argsort(period_dt.to_numpy(dtype='datetime64[ns]'), kind='mergesort')
            disc_clean = disc_clean[order]
            qty_clean = qty_clean[order]
            store_clean = store_clean[order]
            base_clean = base_clean[order]
            periods_clean = periods_clean.iloc[order].reset_index(drop=True)
    else:
        periods_clean = None

    tactical_clean = np.clip(disc_clean - base_clean, 0.0, None)
    below_base_clean = np.clip(base_clean - disc_clean, 0.0, None)
    prev_structural_clean = np.roll(base_clean, 1)
    if len(prev_structural_clean) > 0:
        prev_structural_clean[0] = base_clean[0]
    lag1_structural_clean = prev_structural_clean

    # SIMPLE MODEL: Quantity ~ Actual Discount%
    X_simple = disc_clean.reshape(-1, 1)
    Y = qty_clean
    model_simple = LinearRegression()
    model_simple.fit(X_simple, Y)
    y_pred_simple = model_simple.predict(X_simple)
    ss_res_simple = np.sum((Y - y_pred_simple) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared_simple = 1 - (ss_res_simple / ss_tot) if ss_tot > 0 else 0

    # STAGE 1: Store_Count ~ Actual Discount
    X_stage1 = disc_clean.reshape(-1, 1)
    Y_stage1 = store_clean
    model_stage1 = LinearRegression()
    model_stage1.fit(X_stage1, Y_stage1)
    store_pred = model_stage1.predict(X_stage1)
    residuals_stage1 = Y_stage1 - store_pred
    ss_res_s1 = np.sum(residuals_stage1 ** 2)
    ss_tot_s1 = np.sum((Y_stage1 - np.mean(Y_stage1)) ** 2)
    r_squared_s1 = 1 - (ss_res_s1 / ss_tot_s1) if ss_tot_s1 > 0 else 0

    # STAGE 2: Constrained Ridge
    # Constraints requested:
    # residual >= 0, structural >= 0, tactical >= 0, lag1_structural <= 0
    X_stage2 = np.column_stack([residuals_stage1, base_clean, tactical_clean, lag1_structural_clean])
    Y_stage2 = qty_clean
    model_stage2 = CustomConstrainedRidge(
        l2_penalty=1.0,
        non_negative_indices=[0, 1, 2],
        non_positive_indices=[3],
        maxiter=4000
    )
    model_stage2.fit(X_stage2, Y_stage2)
    qty_pred_enhanced = model_stage2.predict(X_stage2)

    n = len(Y_stage2)
    k = 4
    ss_res_s2 = np.sum((Y_stage2 - qty_pred_enhanced) ** 2)
    r_squared_s2 = 1 - (ss_res_s2 / ss_tot) if ss_tot > 0 else 0
    adj_r_squared_s2 = 1 - (1 - r_squared_s2) * (n - 1) / (n - k - 1) if n > k + 1 else 0
    mse_s2 = ss_res_s2 / (n - k - 1) if n > k + 1 else 0
    se_s2 = np.sqrt(mse_s2)

    # Classical OLS inferential statistics are not directly valid for constrained ridge.
    used_pinv = False
    se_coef = np.full(5, np.nan, dtype=float)
    t_intercept = np.nan
    t_residual = np.nan
    t_structural = np.nan
    t_tactical = np.nan
    t_lag1 = np.nan
    p_intercept = np.nan
    p_residual = np.nan
    p_structural = np.nan
    p_tactical = np.nan
    p_lag1 = np.nan
    f_stat = np.nan
    f_pvalue = np.nan

    return {
        'simple_intercept': model_simple.intercept_,
        'simple_slope': model_simple.coef_[0],
        'simple_r_squared': r_squared_simple,
        'simple_y_pred': y_pred_simple,
        'stage1_intercept': model_stage1.intercept_,
        'stage1_coef_discount': model_stage1.coef_[0],
        'stage1_coef_structural': np.nan,
        'stage1_coef_tactical': np.nan,
        'stage1_slope': model_stage1.coef_[0],
        'stage1_r_squared': r_squared_s1,
        'stage1_residuals': residuals_stage1,
        'stage1_store_pred': store_pred,
        'stage1_model': model_stage1,
        'intercept': model_stage2.intercept_,
        'coef_residual': model_stage2.coef_[0],
        'coef_structural': model_stage2.coef_[1],
        'coef_tactical': model_stage2.coef_[2],
        'coef_lag1_structural': model_stage2.coef_[3],
        'coef_lag1_stepup_structural': model_stage2.coef_[3],
        'coef_lag1_combined': model_stage2.coef_[3],
        'coef_discount': model_stage2.coef_[2],
        'r_squared': r_squared_s2,
        'adj_r_squared': adj_r_squared_s2,
        'se': se_s2,
        'se_intercept': se_coef[0],
        'se_residual': se_coef[1],
        'se_structural': se_coef[2],
        'se_tactical': se_coef[3],
        'se_lag1_structural': se_coef[4],
        'se_lag1_stepup_structural': se_coef[4],
        'se_lag1_combined': se_coef[4],
        'se_discount': se_coef[3],
        't_intercept': t_intercept,
        't_residual': t_residual,
        't_structural': t_structural,
        't_tactical': t_tactical,
        't_lag1_structural': t_lag1,
        't_lag1_stepup_structural': t_lag1,
        't_lag1_combined': t_lag1,
        't_discount': t_tactical,
        'p_intercept': p_intercept,
        'p_residual': p_residual,
        'p_structural': p_structural,
        'p_tactical': p_tactical,
        'p_lag1_structural': p_lag1,
        'p_lag1_stepup_structural': p_lag1,
        'p_lag1_combined': p_lag1,
        'p_discount': p_tactical,
        'f_stat': f_stat,
        'f_pvalue': f_pvalue,
        'n': n,
        'y_pred': qty_pred_enhanced,
        'x_clean': disc_clean,
        'y_clean': qty_clean,
        'store_clean': store_clean,
        'base_discount_clean': base_clean,
        'tactical_discount_clean': tactical_clean,
        'lag1_structural_clean': lag1_structural_clean,
        'prev_structural_clean': prev_structural_clean,
        'lag1_stepup_structural_clean': lag1_structural_clean,
        'lag1_discount_clean': lag1_structural_clean,
        'below_base_clean': below_base_clean,
        'periods_clean': periods_clean,
        'stage2_model': model_stage2,
        'stage2_model_type': 'CustomConstrainedRidge',
        'used_pinv_for_se': used_pinv
    }

def calculate_roi_metrics(ols_result, daily_agg, baseline_discount_pct=None):
    """
    Calculate ROI metrics using structural vs tactical discount logic.
    
    Parameters:
    - ols_result: Result from run_two_stage_ols
    - daily_agg: DataFrame with aggregated data (must include Base_Price)
    - baseline_discount_pct: Backward-compatible scalar baseline if structural series is unavailable
    
    Returns:
    - DataFrame with ROI calculations
    """
    if ols_result is None:
        return None
    
    if len(ols_result['y_pred']) == 0:
        return None

    # Create ROI dataframe
    roi_df = pd.DataFrame({
        'Period': ols_result['periods_clean'],
        'Actual_Discount_Pct': ols_result['x_clean'],
        'Actual_Store_Count': ols_result['store_clean'],
        'Actual_Quantity': ols_result['y_clean']
    })
    roi_df['Period'] = pd.to_datetime(roi_df['Period'], errors='coerce')
    daily_ref = daily_agg.copy(deep=False)
    if 'Period' in daily_ref.columns:
        daily_ref['Period'] = pd.to_datetime(daily_ref['Period'], errors='coerce')
    if 'base_discount_clean' in ols_result:
        structural_discount_pct = np.asarray(ols_result['base_discount_clean'], dtype=float)
    else:
        baseline_fallback = float(baseline_discount_pct) if baseline_discount_pct is not None else 0.0
        structural_discount_pct = np.full(len(roi_df), baseline_fallback, dtype=float)

    lag1_clean = ols_result.get(
        'lag1_structural_clean',
        ols_result.get('lag1_stepup_structural_clean', ols_result.get('lag1_discount_clean', None))
    )
    if lag1_clean is not None and len(lag1_clean) == len(roi_df):
        lag1_struct = np.asarray(lag1_clean, dtype=float)
    else:
        prev_struct = pd.Series(structural_discount_pct).shift(1).fillna(pd.Series(structural_discount_pct)).to_numpy(dtype=float)
        lag1_struct = prev_struct
    roi_df['Lag1_Structural_Pct'] = lag1_struct

    tactical_discount_pct = np.clip(roi_df['Actual_Discount_Pct'].to_numpy(dtype=float) - structural_discount_pct, 0.0, None)
    below_base_pct = np.clip(structural_discount_pct - roi_df['Actual_Discount_Pct'].to_numpy(dtype=float), 0.0, None)

    roi_df['Structural_Discount_Pct'] = structural_discount_pct
    roi_df['Baseline_Discount_Pct'] = structural_discount_pct
    roi_df['Tactical_Discount_Pct'] = tactical_discount_pct
    roi_df['Extra_Discount_Pct'] = tactical_discount_pct
    roi_df['Below_Base_Pct'] = below_base_pct

    base_price_map = {}
    fallback_base_price = pd.to_numeric(daily_ref.get('Base_Price', pd.Series(dtype=float)), errors='coerce').dropna()
    fallback_base_price = float(fallback_base_price.median()) if not fallback_base_price.empty else 100.0
    for period in roi_df['Period']:
        matching_rows = daily_ref[daily_ref['Period'] == period]
        if not matching_rows.empty and 'Base_Price' in matching_rows.columns:
            base_price_map[period] = matching_rows['Base_Price'].iloc[0]
        else:
            base_price_map[period] = fallback_base_price

    roi_df['Base_Price'] = roi_df['Period'].map(base_price_map).astype(float)

    stage1_model = ols_result['stage1_model']
    stage2_model = ols_result['stage2_model']
    n_rows = len(roi_df)

    # Residual anchor must be observed residual from Stage-1 at actual discount.
    # Prefer precomputed Observed_Store_Residual by period when available.
    obs_resid_map = None
    if 'Observed_Store_Residual' in daily_ref.columns and 'Period' in daily_ref.columns:
        obs_resid_df = (
            daily_ref[['Period', 'Observed_Store_Residual']]
            .drop_duplicates(subset=['Period'])
            .copy()
        )
        obs_resid_map = dict(zip(obs_resid_df['Period'], obs_resid_df['Observed_Store_Residual']))
        roi_df['Observed_Store_Residual'] = pd.to_numeric(
            roi_df['Period'].map(obs_resid_map),
            errors='coerce'
        )
    else:
        roi_df['Observed_Store_Residual'] = np.nan

    X_stage1_actual = roi_df['Actual_Discount_Pct'].to_numpy(dtype=float).reshape(-1, 1)

    observed_store_pred = stage1_model.predict(X_stage1_actual)
    fallback_residuals = roi_df['Actual_Store_Count'].to_numpy(dtype=float) - observed_store_pred
    residual_fixed = np.where(
        pd.to_numeric(roi_df['Observed_Store_Residual'], errors='coerce').notna(),
        pd.to_numeric(roi_df['Observed_Store_Residual'], errors='coerce'),
        fallback_residuals
    ).astype(float)

    tactical_vals = roi_df['Tactical_Discount_Pct'].to_numpy(dtype=float)
    structural_vals = roi_df['Structural_Discount_Pct'].to_numpy(dtype=float)
    lag1_vals = roi_df['Lag1_Structural_Pct'].to_numpy(dtype=float)

    stage2_n_features = getattr(stage2_model, 'n_features_in_', 4)
    if stage2_n_features == 4:
        X_predicted = np.column_stack([residual_fixed, structural_vals, tactical_vals, lag1_vals])
        X_baseline = np.column_stack([residual_fixed, structural_vals, np.zeros(n_rows), lag1_vals])
    elif stage2_n_features == 3:
        X_predicted = np.column_stack([residual_fixed, structural_vals, tactical_vals])
        X_baseline = np.column_stack([residual_fixed, structural_vals, np.zeros(n_rows)])
    elif stage2_n_features == 2:
        # Backward compatibility (older two-feature model)
        X_predicted = np.column_stack([residual_fixed, roi_df['Actual_Discount_Pct'].to_numpy(dtype=float)])
        X_baseline = np.column_stack([residual_fixed, structural_vals])
    else:
        X_predicted = residual_fixed.reshape(-1, 1)
        X_baseline = residual_fixed.reshape(-1, 1)

    predicted_quantity = stage2_model.predict(X_predicted)
    baseline_quantity = stage2_model.predict(X_baseline)

    roi_df['Predicted_Store_Count'] = observed_store_pred
    roi_df['Baseline_Store_Count'] = observed_store_pred
    roi_df['Predicted_Quantity'] = predicted_quantity
    roi_df['Baseline_Quantity'] = baseline_quantity

    roi_df['Actual_Discounted_Price'] = roi_df['Base_Price'] * (1 - roi_df['Actual_Discount_Pct'] / 100)
    roi_df['Baseline_Discounted_Price'] = roi_df['Base_Price'] * (1 - roi_df['Structural_Discount_Pct'] / 100)
    roi_df['Structural_Discounted_Price'] = roi_df['Baseline_Discounted_Price']

    roi_df['Predicted_Revenue'] = roi_df['Predicted_Quantity'] * roi_df['Actual_Discounted_Price']
    roi_df['Baseline_Revenue'] = roi_df['Baseline_Quantity'] * roi_df['Baseline_Discounted_Price']
    roi_df['Spend'] = roi_df['Base_Price'] * (roi_df['Tactical_Discount_Pct'] / 100) * roi_df['Predicted_Quantity']

    # Diagnostics: why incremental revenue can be negative despite positive tactical beta.
    roi_df['Qty_Lift_Pct'] = (
        roi_df['Predicted_Quantity'] / roi_df['Baseline_Quantity'].replace(0, np.nan) - 1
    ) * 100
    roi_df['Breakeven_Qty_Lift_Pct'] = (
        roi_df['Baseline_Discounted_Price'] / roi_df['Actual_Discounted_Price'].replace(0, np.nan) - 1
    ) * 100
    roi_df['Lift_Minus_Breakeven_Pct'] = roi_df['Qty_Lift_Pct'] - roi_df['Breakeven_Qty_Lift_Pct']

    roi_df['Incremental_Revenue'] = roi_df['Predicted_Revenue'] - roi_df['Baseline_Revenue']
    roi_df['ROI_1mo'] = roi_df['Incremental_Revenue'] / roi_df['Spend'].replace(0, np.nan)

    roi_df['Predicted_Revenue_Next'] = roi_df['Predicted_Revenue'].shift(-1)
    roi_df['Baseline_Revenue_Next'] = roi_df['Baseline_Revenue'].shift(-1)
    roi_df['Predicted_Revenue_2mo'] = roi_df['Predicted_Revenue'] + roi_df['Predicted_Revenue_Next'].fillna(0)
    roi_df['Baseline_Revenue_2mo'] = roi_df['Baseline_Revenue'] + roi_df['Baseline_Revenue_Next'].fillna(0)
    roi_df['Incremental_Revenue_2mo'] = roi_df['Predicted_Revenue_2mo'] - roi_df['Baseline_Revenue_2mo']
    roi_df['ROI_2mo'] = roi_df['Incremental_Revenue_2mo'] / roi_df['Spend'].replace(0, np.nan)

    # ROI is defined only where tactical discount exists.
    roi_df = roi_df[roi_df['Tactical_Discount_Pct'] > 0].copy()
    return roi_df


def calculate_structural_roi_metrics(ols_result, monthly_agg):
    """
    Structural ROI from base-regime step-up episodes.
    For each step-up (Base_1 -> Base_2), ROI is computed over the full hold period
    where Base_2 remains active, excluding tactical effects.
    """
    if ols_result is None or 'base_discount_clean' not in ols_result:
        return None

    periods = ols_result.get('periods_clean', None)
    if periods is None:
        return None

    structural = np.asarray(ols_result['base_discount_clean'], dtype=float)
    actual_store = np.asarray(ols_result['store_clean'], dtype=float)
    actual_discount = np.asarray(ols_result['x_clean'], dtype=float)
    if len(structural) == 0:
        return None

    period_df = pd.DataFrame({
        'Period': periods,
        'Structural_Discount_Pct': structural,
        'Actual_Store_Count': actual_store,
        'Actual_Discount_Pct': actual_discount
    }).sort_values('Period', ascending=True).reset_index(drop=True)
    period_df['Period'] = pd.to_datetime(period_df['Period'], errors='coerce')
    monthly_ref = monthly_agg.copy(deep=False)
    if 'Period' in monthly_ref.columns:
        monthly_ref['Period'] = pd.to_datetime(monthly_ref['Period'], errors='coerce')
    prev_structural = (
        period_df['Structural_Discount_Pct']
        .shift(1)
        .fillna(period_df['Structural_Discount_Pct'])
        .astype(float)
    )
    period_df['Lag1_Structural_Pct'] = prev_structural

    base_price_map = {}
    fallback_base_price = pd.to_numeric(monthly_ref.get('Base_Price', pd.Series(dtype=float)), errors='coerce').dropna()
    fallback_base_price = float(fallback_base_price.median()) if not fallback_base_price.empty else 100.0
    for period in period_df['Period']:
        matching_rows = monthly_ref[monthly_ref['Period'] == period]
        if not matching_rows.empty and 'Base_Price' in matching_rows.columns:
            base_price_map[period] = matching_rows['Base_Price'].iloc[0]
        else:
            base_price_map[period] = fallback_base_price
    period_df['Base_Price'] = period_df['Period'].map(base_price_map).astype(float)

    # Fixed residual anchor: use observed Stage-1 residual for the same month.
    obs_resid = None
    if 'Observed_Store_Residual' in monthly_ref.columns:
        period_resid_map = (
            monthly_ref[['Period', 'Observed_Store_Residual']]
            .drop_duplicates(subset=['Period'])
            .copy()
        )
        period_resid_map['Period'] = pd.to_datetime(period_resid_map['Period'], errors='coerce')
        period_df = period_df.merge(period_resid_map, on='Period', how='left')
        obs_resid = pd.to_numeric(period_df['Observed_Store_Residual'], errors='coerce')

    if obs_resid is None or obs_resid.isna().any():
        X_obs_stage1 = period_df['Actual_Discount_Pct'].to_numpy(dtype=float).reshape(-1, 1)
        pred_store_obs = ols_result['stage1_model'].predict(X_obs_stage1)
        fallback_resid = period_df['Actual_Store_Count'].to_numpy(dtype=float) - pred_store_obs
        period_df['Observed_Store_Residual'] = np.where(
            pd.to_numeric(period_df.get('Observed_Store_Residual', np.nan), errors='coerce').notna(),
            pd.to_numeric(period_df['Observed_Store_Residual'], errors='coerce'),
            fallback_resid
        )

    # Identify contiguous structural-base regimes.
    regime_break = period_df['Structural_Discount_Pct'].diff().abs().fillna(0) > 1e-9
    period_df['Regime_ID'] = regime_break.cumsum()
    period_df['Row_Idx'] = np.arange(len(period_df))
    regimes = (
        period_df.groupby('Regime_ID', as_index=False)
        .agg(
            Start_Idx=('Row_Idx', 'min'),
            End_Idx=('Row_Idx', 'max'),
            Start_Period=('Period', 'min'),
            End_Period=('Period', 'max'),
            Structural_Discount_Pct=('Structural_Discount_Pct', 'first')
        )
        .sort_values('Start_Idx')
        .reset_index(drop=True)
    )

    stage1_model = ols_result['stage1_model']
    stage2_model = ols_result['stage2_model']
    stage2_n_features = getattr(stage2_model, 'n_features_in_', 4)

    def _predict_qty(store_residuals, struct_vals, tactical_vals, lag1_vals=None):
        if stage2_n_features == 4:
            if lag1_vals is None:
                lag1_vals = np.zeros_like(struct_vals, dtype=float)
            X_s2 = np.column_stack([store_residuals, struct_vals, tactical_vals, lag1_vals])
        elif stage2_n_features == 3:
            X_s2 = np.column_stack([store_residuals, struct_vals, tactical_vals])
        elif stage2_n_features == 2:
            X_s2 = np.column_stack([store_residuals, struct_vals])
        else:
            X_s2 = store_residuals.reshape(-1, 1)
        return stage2_model.predict(X_s2)

    episode_rows = []
    for i in range(1, len(regimes)):
        prev_base = float(regimes.loc[i - 1, 'Structural_Discount_Pct'])
        curr_base = float(regimes.loc[i, 'Structural_Discount_Pct'])
        step_up_pp = curr_base - prev_base
        if step_up_pp <= 0:
            continue

        start_idx = int(regimes.loc[i, 'Start_Idx'])
        end_idx = int(regimes.loc[i, 'End_Idx'])
        hold_df = period_df.iloc[start_idx:end_idx + 1].copy()
        n_rows = len(hold_df)
        if n_rows == 0:
            continue

        actual_store_np = hold_df['Actual_Store_Count'].to_numpy(dtype=float)
        base_price_np = hold_df['Base_Price'].to_numpy(dtype=float)
        residual_anchor = hold_df['Observed_Store_Residual'].to_numpy(dtype=float)

        prev_struct = np.full(n_rows, prev_base, dtype=float)
        curr_struct = np.full(n_rows, curr_base, dtype=float)
        zeros = np.zeros(n_rows, dtype=float)
        # Scenario-specific lag of structural base:
        # - Baseline world (old base held): lag stays old base for all months.
        # - New world (step-up at episode start): first month lag is old base, then new base.
        lag1_prev = np.full(n_rows, prev_base, dtype=float)
        lag1_curr = np.full(n_rows, curr_base, dtype=float)
        lag1_curr[0] = prev_base

        # Counterfactual baseline/current with SAME observed month residual anchor.
        qty_prev = _predict_qty(residual_anchor, prev_struct, zeros, lag1_prev)
        qty_curr = _predict_qty(residual_anchor, curr_struct, zeros, lag1_curr)

        baseline_price = base_price_np * (1 - prev_base / 100.0)
        current_price = base_price_np * (1 - prev_base / 100.0)
        baseline_revenue = qty_prev * baseline_price
        predicted_revenue = qty_curr * current_price
        incremental_revenue = predicted_revenue - baseline_revenue
        spend = base_price_np * (step_up_pp / 100.0) * qty_curr

        spend_sum = float(np.nansum(spend))
        incr_sum = float(np.nansum(incremental_revenue))
        baseline_rev_sum = float(np.nansum(baseline_revenue))
        predicted_rev_sum = float(np.nansum(predicted_revenue))
        baseline_qty_sum = float(np.nansum(qty_prev))
        predicted_qty_sum = float(np.nansum(qty_curr))
        roi_1 = incr_sum / spend_sum if spend_sum > 0 else np.nan

        episode_rows.append({
            'Period': hold_df['Period'].iloc[0],
            'Episode_End_Period': hold_df['Period'].iloc[-1],
            'Hold_Months': int(n_rows),
            'Base_Price': float(np.nanmean(base_price_np)),
            'Prev_Structural_Discount_Pct': prev_base,
            'Structural_Discount_Pct': curr_base,
            'Structural_Step_Up_Pct': step_up_pp,
            'Structural_Step_Down_Pct': 0.0,
            'Baseline_Quantity': baseline_qty_sum,
            'Predicted_Quantity': predicted_qty_sum,
            'Baseline_Revenue': baseline_rev_sum,
            'Predicted_Revenue': predicted_rev_sum,
            'Incremental_Revenue': incr_sum,
            'Spend': spend_sum,
            'ROI_1mo': roi_1,
            'Incremental_Revenue_2mo': np.nan,
            'ROI_2mo': np.nan
        })

    return pd.DataFrame(episode_rows)


def calculate_tactical_roi_metrics(ols_result, monthly_agg, min_tactical_pp=0.5):
    """
    Tactical ROI on monthly level.
    Uses actual monthly discount vs base discount, with the same observed store residual
    in both baseline and predicted scenarios. Reports only months where
    (Actual_Discount - Base_Discount) > min_tactical_pp.
    """
    if ols_result is None:
        return None
    if len(ols_result.get('y_clean', [])) == 0:
        return None

    periods = pd.to_datetime(ols_result['periods_clean'])
    actual_discount = np.asarray(ols_result['x_clean'], dtype=float)
    actual_store = np.asarray(ols_result['store_clean'], dtype=float)
    actual_qty = np.asarray(ols_result['y_clean'], dtype=float)

    if 'base_discount_clean' in ols_result:
        base_discount = np.asarray(ols_result['base_discount_clean'], dtype=float)
    else:
        base_discount = np.asarray(actual_discount, dtype=float)

    tactical_discount = np.clip(actual_discount - base_discount, 0.0, None)

    roi_df = pd.DataFrame({
        'Period': periods,
        'Actual_Discount_Pct': actual_discount,
        'Base_Discount_Pct': base_discount,
        'Tactical_Discount_Pct': tactical_discount,
        'Actual_Store_Count': actual_store,
        'Actual_Quantity': actual_qty
    }).sort_values('Period', ascending=True).reset_index(drop=True)
    prev_structural = (
        roi_df['Base_Discount_Pct']
        .shift(1)
        .fillna(roi_df['Base_Discount_Pct'])
        .astype(float)
    )
    roi_df['Lag1_Structural_Pct'] = prev_structural

    monthly_ref = monthly_agg.copy(deep=False)
    monthly_ref['Period'] = pd.to_datetime(monthly_ref['Period'], errors='coerce')
    base_cols = ['Period']
    if 'Base_Price' in monthly_ref.columns:
        base_cols.append('Base_Price')
    if 'Observed_Store_Residual' in monthly_ref.columns:
        base_cols.append('Observed_Store_Residual')
    roi_df = roi_df.merge(
        monthly_ref[base_cols].drop_duplicates(subset=['Period']),
        on='Period',
        how='left'
    )

    fallback_base_price = pd.to_numeric(monthly_ref.get('Base_Price', pd.Series(dtype=float)), errors='coerce').dropna()
    fallback_base_price = float(fallback_base_price.median()) if not fallback_base_price.empty else 100.0
    roi_df['Base_Price'] = pd.to_numeric(roi_df.get('Base_Price', np.nan), errors='coerce').fillna(fallback_base_price)

    stage1_model = ols_result['stage1_model']
    stage2_model = ols_result['stage2_model']
    n_rows = len(roi_df)

    obs_resid = pd.to_numeric(roi_df.get('Observed_Store_Residual', np.nan), errors='coerce')
    if obs_resid.isna().any():
        X_obs_stage1 = roi_df['Actual_Discount_Pct'].to_numpy(dtype=float).reshape(-1, 1)
        pred_store_obs = stage1_model.predict(X_obs_stage1)
        fallback_resid = roi_df['Actual_Store_Count'].to_numpy(dtype=float) - pred_store_obs
        roi_df['Observed_Store_Residual'] = np.where(obs_resid.notna(), obs_resid, fallback_resid)
    else:
        roi_df['Observed_Store_Residual'] = obs_resid

    residual_fixed = roi_df['Observed_Store_Residual'].to_numpy(dtype=float)
    structural_vals = roi_df['Base_Discount_Pct'].to_numpy(dtype=float)
    tactical_vals = roi_df['Tactical_Discount_Pct'].to_numpy(dtype=float)

    stage2_n_features = getattr(stage2_model, 'n_features_in_', 4)
    lag1_vals = roi_df['Lag1_Structural_Pct'].to_numpy(dtype=float)
    if stage2_n_features == 4:
        X_pred = np.column_stack([residual_fixed, structural_vals, tactical_vals, lag1_vals])
        X_base = np.column_stack([residual_fixed, structural_vals, np.zeros(n_rows), lag1_vals])
    elif stage2_n_features == 3:
        X_pred = np.column_stack([residual_fixed, structural_vals, tactical_vals])
        X_base = np.column_stack([residual_fixed, structural_vals, np.zeros(n_rows)])
    elif stage2_n_features == 2:
        X_pred = np.column_stack([residual_fixed, roi_df['Actual_Discount_Pct'].to_numpy(dtype=float)])
        X_base = np.column_stack([residual_fixed, structural_vals])
    else:
        X_pred = residual_fixed.reshape(-1, 1)
        X_base = residual_fixed.reshape(-1, 1)

    pred_qty = stage2_model.predict(X_pred)
    base_qty = stage2_model.predict(X_base)

    roi_df['Predicted_Quantity'] = pred_qty
    roi_df['Baseline_Quantity'] = base_qty
    roi_df['Predicted_Discounted_Price'] = roi_df['Base_Price'] * (1 - roi_df['Actual_Discount_Pct'] / 100.0)
    roi_df['Baseline_Discounted_Price'] = roi_df['Base_Price'] * (1 - roi_df['Base_Discount_Pct'] / 100.0)
    roi_df['Predicted_Revenue'] = roi_df['Predicted_Quantity'] * roi_df['Predicted_Discounted_Price']
    roi_df['Baseline_Revenue'] = roi_df['Baseline_Quantity'] * roi_df['Baseline_Discounted_Price']
    roi_df['Spend'] = roi_df['Base_Price'] * (roi_df['Tactical_Discount_Pct'] / 100.0) * roi_df['Predicted_Quantity']
    roi_df['Incremental_Revenue'] = roi_df['Predicted_Revenue'] - roi_df['Baseline_Revenue']
    roi_df['ROI_1mo'] = roi_df['Incremental_Revenue'] / roi_df['Spend'].replace(0, np.nan)

    roi_df = roi_df[roi_df['Tactical_Discount_Pct'] > float(min_tactical_pp)].copy()
    if roi_df.empty:
        return roi_df

    return roi_df.sort_values('Period', ascending=True).reset_index(drop=True)

# Main app
def main():
    st.title(" Outlet Analysis Tool")
    st.markdown("### Analyze Filtered Outlet Data with RFM Segmentation")
    
    # Load data once per session (avoid showing loading spinner on every widget change)
    if 'base_df' not in st.session_state or st.session_state.base_df is None:
        with st.spinner("Loading parquet files from step3_filtered_engineered..."):
            st.session_state.base_df = load_parquet_files()
    df = st.session_state.base_df
    
    if df is None:
        st.stop()
    
    # Initialize session state for RFM
    if 'rfm_data' not in st.session_state:
        st.session_state.rfm_data = None
    if 'rfm_max_date' not in st.session_state:
        st.session_state.rfm_max_date = None
    if 'rfm_source_df' not in st.session_state:
        st.session_state.rfm_source_df = None
    if 'rfm_filters' not in st.session_state:
        st.session_state.rfm_filters = {}
    if 'rfm_cluster_summary' not in st.session_state:
        st.session_state.rfm_cluster_summary = None
    if 'rfm_input_rows' not in st.session_state:
        st.session_state.rfm_input_rows = None
    if 'rfm_input_outlets' not in st.session_state:
        st.session_state.rfm_input_outlets = None
    
    st.success(f" Loaded {len(df):,} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # STEP 1: RFM Calculation Filters
    step1_section = st.sidebar.expander(" Step 1: RFM Calculation Filters", expanded=True)
    step1_section.markdown("*Select filters to define the dataset for RFM analysis*")
    
    # State filter (multiselect)
    states = sorted(df['Final_State'].dropna().unique().tolist())
    selected_states = step1_section.multiselect(
        "Select State(s)",
        options=states,
        default=states,
        help="Select one or more states for RFM calculation"
    )

    # Cascading option logic (lightweight): only updates selectable values.
    # Actual dataframe filtering for RFM still happens only on button click.
    option_mask = pd.Series(True, index=df.index)
    if selected_states:
        option_mask &= df['Final_State'].isin(selected_states)

    categories = sorted(df.loc[option_mask, 'Category'].dropna().unique().tolist())
    selected_categories = step1_section.multiselect(
        "Select Category(ies)",
        options=categories,
        default=[],
        help="Select categories for RFM calculation (leave empty for all)"
    )

    if selected_categories:
        option_mask &= df['Category'].isin(selected_categories)

    # Subcategory filter (multiselect)
    subcategories = sorted(df.loc[option_mask, 'Subcategory'].dropna().unique().tolist())
    selected_subcategories = step1_section.multiselect(
        "Select Subcategory(ies)",
        options=subcategories,
        default=[],
        help="Select subcategories for RFM calculation (leave empty for all)"
    )

    if selected_subcategories:
        option_mask &= df['Subcategory'].isin(selected_subcategories)

    # Brand filter (multiselect)
    brands = sorted(df.loc[option_mask, 'Brand'].dropna().unique().tolist())
    selected_brands = step1_section.multiselect(
        "Select Brand(s)",
        options=brands,
        default=[],
        help="Select brands for RFM calculation (leave empty for all)"
    )

    if selected_brands:
        option_mask &= df['Brand'].isin(selected_brands)

    # Size filter (multiselect)
    sizes = sorted(df.loc[option_mask, 'Sizes'].dropna().unique().tolist())
    selected_sizes = step1_section.multiselect(
        "Select Size(s)",
        options=sizes,
        default=[],
        help="Select sizes for RFM calculation (leave empty for all)"
    )

    if st.session_state.rfm_input_rows is not None and st.session_state.rfm_input_outlets is not None:
        step1_section.metric("Rows for RFM", f"{st.session_state.rfm_input_rows:,}")
        step1_section.metric("Outlets for RFM", f"{st.session_state.rfm_input_outlets:,}")
    else:
        step1_section.metric("Dataset Rows", f"{len(df):,}")
        step1_section.metric("Dataset Outlets", f"{df['Outlet_ID'].nunique():,}")
    step1_section.caption("Selections are applied when you click 'Calculate RFM'.")
    
    step1_section.divider()
    
    # RFM Configuration
    step1_section.markdown("** RFM Configuration**")
    
    recency_threshold = step1_section.slider(
        "Recency Threshold (days)",
        min_value=30,
        max_value=180,
        value=90,
        step=10,
        help="Outlets with last order within this many days are considered 'Recent'"
    )
    
    if step1_section.button(" Calculate RFM", type="primary", use_container_width=True):
        with st.spinner("Calculating RFM metrics and clustering..."):
            # Start with reference; filtering below materializes only required subsets.
            df_for_rfm = df
            
            if selected_states:
                df_for_rfm = df_for_rfm[df_for_rfm['Final_State'].isin(selected_states)]
            if selected_categories:
                df_for_rfm = df_for_rfm[df_for_rfm['Category'].isin(selected_categories)]
            if selected_subcategories:
                df_for_rfm = df_for_rfm[df_for_rfm['Subcategory'].isin(selected_subcategories)]
            if selected_brands:
                df_for_rfm = df_for_rfm[df_for_rfm['Brand'].isin(selected_brands)]
            if selected_sizes:
                df_for_rfm = df_for_rfm[df_for_rfm['Sizes'].isin(selected_sizes)]
            
            if df_for_rfm.empty:
                st.session_state.rfm_data = None
                st.session_state.rfm_max_date = None
                st.session_state.rfm_source_df = None
                st.session_state.rfm_filters = {}
                st.session_state.rfm_cluster_summary = None
                st.session_state.rfm_input_rows = 0
                st.session_state.rfm_input_outlets = 0
                st.warning("No rows match selected filters. Adjust filters and click Calculate RFM again.")
            else:
                calc_result = calculate_rfm(df_for_rfm, recency_days=recency_threshold)
                if isinstance(calc_result, tuple) and len(calc_result) == 3:
                    rfm_result, max_date, cluster_summary = calc_result
                else:
                    # Backward compatibility for any stale cached payload shape.
                    rfm_result, max_date = calc_result
                    cluster_summary = build_cluster_summary_from_rfm(rfm_result)

                rfm_result = backfill_rfm_fields(rfm_result)
                st.session_state.rfm_data = rfm_result
                st.session_state.rfm_max_date = max_date
                st.session_state.rfm_source_df = df_for_rfm
                st.session_state.rfm_cluster_summary = cluster_summary
                st.session_state.rfm_input_rows = len(df_for_rfm)
                st.session_state.rfm_input_outlets = df_for_rfm['Outlet_ID'].nunique()
                st.session_state.rfm_filters = {
                    'states': selected_states,
                    'categories': selected_categories,
                    'subcategories': selected_subcategories,
                    'brands': selected_brands,
                    'sizes': selected_sizes
                }
                st.success(" RFM Analysis Complete!")
    
    # Use last calculated RFM dataset for downstream analysis.
    # If RFM not calculated yet, show full dataset in overview.
    if st.session_state.get('rfm_source_df') is not None:
        df_filtered = st.session_state.rfm_source_df
    else:
        df_filtered = df
    
    # STEP 2: Outlet Selection Filters (only show if RFM is calculated)
    if 'rfm_data' in st.session_state and st.session_state.rfm_data is not None:
        step2_section = st.sidebar.expander(" Step 2: Outlet Selection Filters", expanded=False)
        step2_section.markdown("*Narrow down outlets for detailed analysis*")
        
        rfm = backfill_rfm_fields(st.session_state.rfm_data)
        st.session_state.rfm_data = rfm
        if st.session_state.get('rfm_cluster_summary') is None:
            st.session_state.rfm_cluster_summary = build_cluster_summary_from_rfm(rfm)
        
        # RFM Segment filter (multiselect)
        all_segments = sorted(rfm['RFM_Segment'].unique())
        selected_segments = step2_section.multiselect(
            "Select RFM Segment(s)",
            options=all_segments,
            default=[],
            help="Choose one or more segments (leave empty for all)"
        )
        
        # Filter outlets based on RFM segment
        if selected_segments:
            segment_rfm = rfm[rfm['RFM_Segment'].isin(selected_segments)].copy()
        else:
            segment_rfm = rfm.copy()
        
        # Outlet Classification filter (multiselect)
        classifications = sorted(df_filtered['Final_Outlet_Classification'].dropna().unique().tolist())
        selected_classifications = step2_section.multiselect(
            "Select Outlet Type(s)",
            options=classifications,
            default=[],
            help="Filter by outlet classification (leave empty for all)"
        )
        
        if selected_classifications:
            classification_outlets = df_filtered[
                df_filtered['Final_Outlet_Classification'].isin(selected_classifications)
            ]['Outlet_ID'].unique()
            segment_rfm = segment_rfm[segment_rfm['Outlet_ID'].isin(classification_outlets)]
        
        # Slab filter (multiselect)
        slabs = sorted(df_filtered['Slab'].dropna().unique().tolist())
        slab_options = ['All Slabs'] + slabs
        selected_slabs_ui = step2_section.multiselect(
            "Select Slab(s)",
            options=slab_options,
            default=[],
            help="Filter by quantity slab. Choose 'All Slabs' to explicitly include every slab."
        )
        
        if 'All Slabs' in selected_slabs_ui:
            selected_slabs = slabs.copy()
            step2_section.info(f" All slabs selected ({len(selected_slabs)})")
        else:
            selected_slabs = selected_slabs_ui
        
        if selected_slabs:
            slab_outlets = df_filtered[
                df_filtered['Slab'].isin(selected_slabs)
            ]['Outlet_ID'].unique()
            segment_rfm = segment_rfm[segment_rfm['Outlet_ID'].isin(slab_outlets)]
        
        # Calculate total net amount per outlet
        outlet_totals = df_filtered.groupby('Outlet_ID')['Net_Amt'].sum().reset_index()
        outlet_totals.columns = ['Outlet_ID', 'Total_Net_Amt']
        
        # Merge and sort
        segment_rfm = segment_rfm.merge(outlet_totals, on='Outlet_ID', how='inner')
        segment_rfm = segment_rfm.sort_values('Total_Net_Amt', ascending=False)
        
        if len(segment_rfm) > 0:
            # Add "Select All" checkbox
            select_all_outlets = step2_section.checkbox(
                " Select All Outlets",
                value=False,
                help="Check to select all filtered outlets"
            )
            
            if select_all_outlets:
                # Select all outlets
                selected_outlets = segment_rfm['Outlet_ID'].tolist()
                step2_section.info(f" All {len(selected_outlets)} outlets selected")
            else:
                # Create outlet list for multiselect
                outlet_options = [
                    f"{row['Outlet_ID']} ({row['Total_Net_Amt']:,.0f})" 
                    for _, row in segment_rfm.iterrows()
                ]
                
                selected_outlets_display = step2_section.multiselect(
                    f"Select Outlet(s) ({len(segment_rfm)} outlets)",
                    options=outlet_options,
                    default=[],
                    help="Select one or more outlets. Sorted by highest net amount"
                )
                
                if selected_outlets_display:
                    selected_outlets = [int(o.split(' (')[0]) for o in selected_outlets_display]
                else:
                    selected_outlets = None
        else:
            selected_outlets = None
            step2_section.warning("No outlets match the selected filters")
    else:
        selected_segments = []
        selected_outlets = None
        selected_slabs = []
    
    # STEP 3: View Filters (only show if outlets are selected AND multiple items in Step 1)
    # This allows narrowing down what's shown in graphs/tables
    if selected_outlets:
        st.sidebar.divider()
        
        # Check if multiple items selected in Step 1 filters
        multiple_step1_items = (
            (len(selected_categories) > 1) or
            (len(selected_subcategories) > 1) or
            (len(selected_brands) > 1) or
            (len(selected_sizes) > 1)
        )
        
        if multiple_step1_items:
            st.sidebar.header(" Step 3: Outlet View Filters")
            st.sidebar.markdown("*Narrow down what's displayed in outlet graphs/tables*")
            st.sidebar.info(" These filters only affect the visualization, not the outlet selection")
            
            # Get transactions for selected outlets
            df_view_base = df_filtered[df_filtered['Outlet_ID'].isin(selected_outlets)]
            
            # View Category filter
            view_categories = sorted(df_view_base['Category'].dropna().unique().tolist())
            if len(view_categories) > 1:
                view_selected_categories = st.sidebar.multiselect(
                    "View: Category(ies)",
                    options=view_categories,
                    default=view_categories,
                    help="Filter what's shown in graphs (doesn't change outlet list)"
                )
                if view_selected_categories:
                    df_view_base = df_view_base[df_view_base['Category'].isin(view_selected_categories)]
            
            # View Subcategory filter
            view_subcategories = sorted(df_view_base['Subcategory'].dropna().unique().tolist())
            if len(view_subcategories) > 1:
                view_selected_subcategories = st.sidebar.multiselect(
                    "View: Subcategory(ies)",
                    options=view_subcategories,
                    default=view_subcategories,
                    help="Filter what's shown in graphs (doesn't change outlet list)"
                )
                if view_selected_subcategories:
                    df_view_base = df_view_base[df_view_base['Subcategory'].isin(view_selected_subcategories)]
            
            # View Brand filter
            view_brands = sorted(df_view_base['Brand'].dropna().unique().tolist())
            if len(view_brands) > 1:
                view_selected_brands = st.sidebar.multiselect(
                    "View: Brand(s)",
                    options=view_brands,
                    default=view_brands,
                    help="Filter what's shown in graphs (doesn't change outlet list)"
                )
                if view_selected_brands:
                    df_view_base = df_view_base[df_view_base['Brand'].isin(view_selected_brands)]
            
            # View Size filter
            view_sizes = sorted(df_view_base['Sizes'].dropna().unique().tolist())
            if len(view_sizes) > 1:
                view_selected_sizes = st.sidebar.multiselect(
                    "View: Size(s)",
                    options=view_sizes,
                    default=view_sizes,
                    help="Filter what's shown in graphs (doesn't change outlet list)"
                )
                if view_selected_sizes:
                    df_view_base = df_view_base[df_view_base['Sizes'].isin(view_selected_sizes)]
            
            df_view = df_view_base
            st.sidebar.metric("View Rows", f"{len(df_view):,}")
        else:
            # No Step 3 filters needed - use all data for selected outlets
            df_view = df_filtered[df_filtered['Outlet_ID'].isin(selected_outlets)]
    else:
        # No outlets selected - use filtered data
        df_view = df_filtered
    
    # Apply slab filter to df_view if slabs were selected in Step 2
    if selected_outlets and selected_slabs:
        df_view = df_view[df_view['Slab'].isin(selected_slabs)]
        st.sidebar.info(f" Filtered to slab(s): {', '.join(selected_slabs)}")
    
    # Tabs
    tab1, tab2 = st.tabs([" RFM Analysis", " Data Overview"])
    
    # TAB 1: RFM Analysis
    with tab1:
        st.header(" RFM Analysis - Outlet Classification")
        
        # Display RFM results if available
        if st.session_state.rfm_data is not None:
            rfm = st.session_state.rfm_data
            max_date = st.session_state.rfm_max_date
            
            # Key Metrics
            st.subheader(" RFM Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Outlets", f"{len(rfm):,}")
                recent_pct = (rfm['R_label'] == 'Recent').sum() / len(rfm) * 100
                st.metric("Recent Outlets", f"{recent_pct:.1f}%")
            
            with col2:
                high_f_pct = (rfm['F_label'] == 'High').sum() / len(rfm) * 100
                st.metric("High Frequency", f"{high_f_pct:.1f}%")
                avg_order_days_all = rfm['unique_order_days'].mean() if 'unique_order_days' in rfm.columns else rfm['orders_count'].mean()
                st.metric("Avg Order Days", f"{avg_order_days_all:.1f}")
                avg_orders_per_day = rfm['orders_per_day'].mean() if 'orders_per_day' in rfm.columns else rfm['orders_count'].mean()
                st.metric("Avg Orders/Day", f"{avg_orders_per_day:.3f}")
            
            with col3:
                high_m_pct = (rfm['M_label'] == 'High').sum() / len(rfm) * 100
                st.metric("High Monetary", f"{high_m_pct:.1f}%")
                st.metric("Avg AOV", f"{rfm['AOV'].mean():.2f}")
            
            with col4:
                st.metric("Analysis Date", str(max_date.date()))
                st.metric("RFM Segments", rfm['RFM_Segment'].nunique())
            
            st.divider()

            cluster_summary = st.session_state.get('rfm_cluster_summary')
            if cluster_summary is not None:
                with st.expander("Frequency and Monetary Cluster Ranges", expanded=False):
                    st.markdown("**Frequency Clusters (orders/day):**")
                    st.dataframe(
                        cluster_summary['frequency'].style.format({
                            'Outlets': '{:,.0f}',
                            'Min_Orders_Per_Day': '{:.4f}',
                            'Max_Orders_Per_Day': '{:.4f}',
                            'Mean_Orders_Per_Day': '{:.4f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.markdown("**Monetary Clusters (AOV):**")
                    st.dataframe(
                        cluster_summary['monetary'].style.format({
                            'Outlets': '{:,.0f}',
                            'Min_AOV': '{:,.2f}',
                            'Max_AOV': '{:,.2f}',
                            'Mean_AOV': '{:,.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                st.divider()
            
            # Beautiful 8-Grid Segment View
            st.subheader(" RFM Segment Grid View")
            
            # Define all 8 segments in order
            all_segments = [
                'Recent-High-High', 'Recent-High-Low',
                'Recent-Low-High', 'Recent-Low-Low',
                'Stale-High-High', 'Stale-High-Low',
                'Stale-Low-High', 'Stale-Low-Low'
            ]
            
            # Calculate segment stats with state breakdown
            total_sales_value = df_filtered['SalesValue_atBasicRate'].sum()
            
            segment_details = []
            for seg in all_segments:
                seg_data = rfm[rfm['RFM_Segment'] == seg]
                total_outlets = len(seg_data)
                pct_total = (total_outlets / len(rfm) * 100) if len(rfm) > 0 else 0
                
                # State breakdown
                state_breakdown = seg_data['Final_State'].value_counts().to_dict()
                mah_count = state_breakdown.get('MAH', 0)
                up_count = state_breakdown.get('UP', 0)
                
                # Averages
                avg_order_days = (
                    seg_data['unique_order_days'].mean()
                    if ('unique_order_days' in seg_data.columns and len(seg_data) > 0)
                    else (seg_data['orders_count'].mean() if len(seg_data) > 0 else 0)
                )
                avg_aov = seg_data['AOV'].mean() if len(seg_data) > 0 else 0
                avg_recency = seg_data['Recency_days'].mean() if len(seg_data) > 0 else 0
                
                # Market share by SalesValue_atBasicRate
                seg_outlets = seg_data['Outlet_ID'].tolist()
                seg_sales_value = df_filtered[df_filtered['Outlet_ID'].isin(seg_outlets)]['SalesValue_atBasicRate'].sum()
                market_share_pct = (seg_sales_value / total_sales_value * 100) if total_sales_value > 0 else 0
                
                segment_details.append({
                    'segment': seg,
                    'total': total_outlets,
                    'pct': pct_total,
                    'mah': mah_count,
                    'up': up_count,
                    'avg_order_days': avg_order_days,
                    'avg_aov': avg_aov,
                    'avg_recency': avg_recency,
                    'market_share': market_share_pct
                })
            
            # Create 4x2 grid (4 columns, 2 rows)
            for row in range(2):
                cols = st.columns(4)
                for col_idx in range(4):
                    seg_idx = row * 4 + col_idx
                    seg_info = segment_details[seg_idx]
                    
                    with cols[col_idx]:
                        # Determine color based on segment quality
                        if 'Recent-High-High' in seg_info['segment']:
                            color = '#28a745'  # Green - Best
                            emoji = ''
                        elif 'Stale-Low-Low' in seg_info['segment']:
                            color = '#dc3545'  # Red - Worst
                            emoji = ''
                        elif 'Recent' in seg_info['segment']:
                            color = '#17a2b8'  # Blue - Good
                            emoji = ''
                        else:
                            color = '#ffc107'  # Yellow - At Risk
                            emoji = ''
                        
                        # Create card
                        st.markdown(f"""
                        <div style="
                            border: 2px solid {color};
                            border-radius: 10px;
                            padding: 12px;
                            background-color: {color}15;
                            min-height: 280px;
                            margin-bottom: 10px;
                        ">
                            <div style="text-align: center; margin-bottom: 8px;">
                                <div style="font-size: 28px; line-height: 1;">{emoji}</div>
                            </div>
                            <div style="text-align: center; margin-bottom: 12px;">
                                <div style="font-weight: bold; font-size: 13px; color: {color}; line-height: 1.3;">
                                    {seg_info['segment']}
                                </div>
                            </div>
                            <div style="text-align: center; margin-bottom: 12px;">
                                <div style="font-size: 36px; font-weight: bold; color: {color}; line-height: 1;">
                                    {seg_info['total']:,}
                                </div>
                                <div style="font-size: 16px; color: {color}; margin-top: 4px;">
                                    {seg_info['pct']:.1f}%
                                </div>
                                <div style="font-size: 12px; color: #666; margin-top: 4px; font-weight: bold;">
                                    Market Share: {seg_info['market_share']:.1f}%
                                </div>
                            </div>
                            <div style="
                                margin-bottom: 10px;
                                padding: 8px;
                                background-color: {color}10;
                                border-radius: 5px;
                            ">
                                <div style="font-size: 11px; color: #666; text-align: center; margin-bottom: 4px;">
                                    <strong>Avg Order Days:</strong> {seg_info['avg_order_days']:.1f}
                                </div>
                                <div style="font-size: 11px; color: #666; text-align: center; margin-bottom: 4px;">
                                    <strong>Avg AOV:</strong> {seg_info['avg_aov']:.0f}
                                </div>
                                <div style="font-size: 11px; color: #666; text-align: center;">
                                    <strong>Avg Recency:</strong> {seg_info['avg_recency']:.0f} days
                                </div>
                            </div>
                            <div style="
                                display: grid; 
                                grid-template-columns: 1fr 1fr; 
                                gap: 8px; 
                                padding-top: 8px;
                                border-top: 1px solid {color}40;
                            ">
                                <div style="text-align: center;">
                                    <div style="font-weight: bold; color: #666; font-size: 11px;">MAH</div>
                                    <div style="color: {color}; font-weight: bold; font-size: 16px;">{seg_info['mah']:,}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-weight: bold; color: #666; font-size: 11px;">UP</div>
                                    <div style="color: {color}; font-weight: bold; font-size: 16px;">{seg_info['up']:,}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()
            
            # Show outlet analysis if outlets are selected from sidebar
            if selected_outlets:
                st.markdown(f"###  Analyzing {len(selected_outlets)} Selected Outlet(s)")
                
                # Add controls in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Add view toggle if multiple outlets selected
                    if len(selected_outlets) > 1:
                        view_mode = st.radio(
                            "Select View Mode:",
                            options=["Combined View", "Individual View"],
                            horizontal=True,
                            help="Combined View: Aggregate all outlets together | Individual View: Show each outlet separately"
                        )
                    else:
                        view_mode = "Individual View"
                
                with col2:
                    # Add time aggregation option
                    time_agg = st.radio(
                        "Time Aggregation:",
                        options=["Daily", "Weekly", "Monthly"],
                        horizontal=True,
                        help="Choose how to aggregate data over time"
                    )
                
                st.divider()
                

                # COMBINED VIEW - Aggregate all outlets together
                if view_mode == "Combined View" and len(selected_outlets) > 1:
                    st.markdown(f"### Combined Analysis - {len(selected_outlets)} Outlets ({time_agg})")

                    combined_transactions = df_view.copy()

                    if time_agg == 'Daily':
                        daily_agg_combined = combined_transactions.groupby('Date').agg({
                            'Outlet_ID': 'nunique',
                            'Store_ID': 'nunique',
                            'Bill_No': 'nunique',
                            'Quantity': 'sum',
                            'Net_Amt': 'sum',
                            'TotalDiscount': 'sum',
                            'SalesValue_atBasicRate': 'sum'
                        }).reset_index()
                        daily_agg_combined.columns = ['Period', 'Active_Outlets', 'Store_Count', 'Orders', 'Quantity', 'Net_Amt', 'TotalDiscount', 'SalesValue_atBasicRate']

                    elif time_agg == 'Weekly':
                        combined_transactions['Period'] = combined_transactions['Date'].dt.to_period('W').apply(lambda x: x.start_time)
                        daily_agg_combined = combined_transactions.groupby('Period').agg({
                            'Outlet_ID': 'nunique',
                            'Store_ID': 'nunique',
                            'Bill_No': 'nunique',
                            'Quantity': 'sum',
                            'Net_Amt': 'sum',
                            'TotalDiscount': 'sum',
                            'SalesValue_atBasicRate': 'sum'
                        }).reset_index()
                        daily_agg_combined.columns = ['Period', 'Active_Outlets', 'Store_Count', 'Orders', 'Quantity', 'Net_Amt', 'TotalDiscount', 'SalesValue_atBasicRate']

                    elif time_agg == 'Monthly':
                        combined_transactions['Period'] = combined_transactions['Date'].dt.to_period('M').apply(lambda x: x.start_time)
                        daily_agg_combined = combined_transactions.groupby('Period').agg({
                            'Outlet_ID': 'nunique',
                            'Store_ID': 'nunique',
                            'Bill_No': 'nunique',
                            'Quantity': 'sum',
                            'Net_Amt': 'sum',
                            'TotalDiscount': 'sum',
                            'SalesValue_atBasicRate': 'sum'
                        }).reset_index()
                        daily_agg_combined.columns = ['Period', 'Active_Outlets', 'Store_Count', 'Orders', 'Quantity', 'Net_Amt', 'TotalDiscount', 'SalesValue_atBasicRate']

                    daily_agg_combined['Discount_Pct'] = (daily_agg_combined['TotalDiscount'] / daily_agg_combined['SalesValue_atBasicRate'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
                    daily_agg_combined['Discount_Pct_Display'] = round_discount_for_display(daily_agg_combined['Discount_Pct'], step=0.5)
                    daily_agg_combined['Base_Price'] = (
                        daily_agg_combined['SalesValue_atBasicRate'] / daily_agg_combined['Quantity'].replace(0, np.nan)
                    ).replace([np.inf, -np.inf], np.nan)
                    daily_agg_combined = daily_agg_combined.sort_values('Period', ascending=False)

                    corr_qty_disc_pct = daily_agg_combined['Quantity'].corr(daily_agg_combined['Discount_Pct'])
                    corr_sales_disc_pct = daily_agg_combined['SalesValue_atBasicRate'].corr(daily_agg_combined['Discount_Pct'])

                    monthly_model_agg = aggregate_for_roi(combined_transactions, 'Monthly')
                    monthly_model_agg = monthly_model_agg.sort_values('Period', ascending=True).reset_index(drop=True).copy()
                    daily_base_plot_df = aggregate_for_roi(combined_transactions, 'Daily')
                    daily_base_plot_df = daily_base_plot_df.sort_values('Period', ascending=True).reset_index(drop=True).copy()
                    with st.expander("Base Discount Estimation Settings (Monthly Base from Daily 10-day Medians)", expanded=False):
                        s1, s2 = st.columns(2)
                        base_min_up_jump_pp = float(s1.number_input(
                            "Min monthly base step-up (pp)",
                            min_value=0.0,
                            max_value=5.0,
                            value=1.0,
                            step=0.1,
                            key="base_min_up_jump_monthly_model"
                        ))
                        base_min_down_drop_pp = float(s2.number_input(
                            "Min monthly base step-down (pp)",
                            min_value=0.0,
                            max_value=5.0,
                            value=1.0,
                            step=0.1,
                            key="base_min_down_drop_monthly_model"
                        ))
                        st.caption("Per month: compute medians for days 1-10, 11-20, 21-end; candidate base is minimum median. Base changes only if step-up or step-down magnitude meets threshold.")

                    # Estimate base on DAILY series with monthly-block logic, then roll up to one base per month.
                    daily_base_discount, daily_base_transitions = estimate_base_discount_daily_blocks(
                        daily_base_plot_df['Period'],
                        daily_base_plot_df['Discount_Pct'],
                        min_upward_jump_pp=base_min_up_jump_pp,
                        min_downward_drop_pp=base_min_down_drop_pp,
                        round_step=0.5
                    )
                    daily_base_discount = round_discount_for_display(daily_base_discount, step=0.5)
                    daily_base_plot_df['Base_Discount_Pct'] = daily_base_discount
                    daily_base_plot_df['Is_Base_Transition'] = daily_base_transitions
                    daily_base_plot_df['Month_Key'] = daily_base_plot_df['Period'].dt.to_period('M')

                    monthly_base_map = (
                        daily_base_plot_df.groupby('Month_Key', as_index=False)
                        .agg(
                            Base_Discount_Pct=('Base_Discount_Pct', 'first'),
                            Is_Base_Transition=('Is_Base_Transition', 'max')
                        )
                    )
                    monthly_model_agg['Month_Key'] = monthly_model_agg['Period'].dt.to_period('M')
                    monthly_model_agg = monthly_model_agg.merge(
                        monthly_base_map,
                        on='Month_Key',
                        how='left'
                    )
                    monthly_model_agg['Base_Discount_Pct'] = monthly_model_agg['Base_Discount_Pct'].ffill().bfill()
                    monthly_model_agg['Is_Base_Transition'] = monthly_model_agg['Is_Base_Transition'].fillna(False)

                    daily_base_plot_df['Discount_Pct_Display'] = round_discount_for_display(daily_base_plot_df['Discount_Pct'], step=0.5)
                    daily_base_plot_df['Base_Discount_Pct_Display'] = round_discount_for_display(daily_base_plot_df['Base_Discount_Pct'], step=0.5)

                    st.markdown("#### Base Discount vs Actual Discount")

                    fig_base_discount = go.Figure()
                    fig_base_discount.add_trace(go.Scatter(
                        x=daily_base_plot_df['Period'],
                        y=daily_base_plot_df['Discount_Pct_Display'],
                        mode='lines+markers',
                        name='Actual Discount %',
                        line=dict(color='#E63946', width=2),
                        marker=dict(size=5)
                    ))
                    fig_base_discount.add_trace(go.Scatter(
                        x=daily_base_plot_df['Period'],
                        y=daily_base_plot_df['Base_Discount_Pct_Display'],
                        mode='lines+markers',
                        name='Estimated Base Discount %',
                        line=dict(color='#1D3557', width=3),
                        marker=dict(size=5)
                    ))

                    transition_points = (
                        daily_base_plot_df[daily_base_plot_df['Is_Base_Transition'].fillna(False)]
                        .groupby('Month_Key', as_index=False)
                        .first()
                    )
                    if not transition_points.empty:
                        fig_base_discount.add_trace(go.Scatter(
                            x=transition_points['Period'],
                            y=transition_points['Base_Discount_Pct_Display'],
                            mode='markers',
                            name='Detected Transition',
                            marker=dict(color='#FF8C00', size=10, symbol='diamond')
                        ))

                    fig_base_discount.update_layout(
                        height=360,
                        xaxis_title='Daily',
                        yaxis_title='Discount %',
                        hovermode='x unified',
                        showlegend=True
                    )
                    st.plotly_chart(fig_base_discount, use_container_width=True)
                    st.caption("This chart is always daily actual discount with monthly estimated base discount mapped to each day. Modeling and ROI are always monthly.")

                    slab_scope = selected_slabs if selected_slabs else sorted(combined_transactions['Slab'].dropna().unique().tolist())
                    with st.expander("Individual Slab Base vs Actual Discount", expanded=False):
                        if not slab_scope:
                            st.info("No slabs available after current filters.")
                        else:
                            for slab in slab_scope:
                                slab_label = str(slab).strip() if pd.notna(slab) else "Unknown"
                                if slab_label == "":
                                    slab_label = "Unknown"
                                st.markdown(f"**Slab: {slab_label}**")
                                try:
                                    slab_txn = combined_transactions[combined_transactions['Slab'] == slab].copy()
                                    if slab_txn.empty:
                                        st.info(f"No data for slab {slab_label} after current filters.")
                                        continue

                                    slab_daily_plot = aggregate_for_roi(slab_txn, 'Daily').sort_values('Period', ascending=True).reset_index(drop=True)
                                    if slab_daily_plot.empty:
                                        st.info(f"No daily data available for slab {slab_label}.")
                                        continue

                                    slab_daily_base_discount, slab_daily_transitions = estimate_base_discount_daily_blocks(
                                        slab_daily_plot['Period'],
                                        slab_daily_plot['Discount_Pct'],
                                        min_upward_jump_pp=base_min_up_jump_pp,
                                        min_downward_drop_pp=base_min_down_drop_pp,
                                        round_step=0.5
                                    )
                                    slab_daily_plot['Base_Discount_Pct'] = round_discount_for_display(slab_daily_base_discount, step=0.5)
                                    slab_daily_plot['Is_Base_Transition'] = slab_daily_transitions
                                    slab_daily_plot['Month_Key'] = slab_daily_plot['Period'].dt.to_period('M')
                                    slab_daily_plot['Discount_Pct_Display'] = round_discount_for_display(slab_daily_plot['Discount_Pct'], step=0.5)
                                    slab_daily_plot['Base_Discount_Pct_Display'] = round_discount_for_display(slab_daily_plot['Base_Discount_Pct'], step=0.5)

                                    fig_slab_base = make_subplots(specs=[[{"secondary_y": True}]])
                                    fig_slab_base.add_trace(go.Scatter(
                                        x=slab_daily_plot['Period'],
                                        y=slab_daily_plot['Discount_Pct_Display'],
                                        mode='lines+markers',
                                        name='Actual Discount %',
                                        line=dict(color='#E63946', width=2),
                                        marker=dict(size=4)
                                    ), secondary_y=False)
                                    fig_slab_base.add_trace(go.Scatter(
                                        x=slab_daily_plot['Period'],
                                        y=slab_daily_plot['Base_Discount_Pct_Display'],
                                        mode='lines+markers',
                                        name='Estimated Base Discount %',
                                        line=dict(color='#1D3557', width=3),
                                        marker=dict(size=4)
                                    ), secondary_y=False)
                                    fig_slab_base.add_trace(go.Scatter(
                                        x=slab_daily_plot['Period'],
                                        y=slab_daily_plot['Quantity'],
                                        mode='lines+markers',
                                        name='Actual Quantity',
                                        line=dict(color='#2A9D8F', width=2, dash='dot'),
                                        marker=dict(size=4)
                                    ), secondary_y=True)

                                    slab_transition_points = (
                                        slab_daily_plot[slab_daily_plot['Is_Base_Transition'].fillna(False)]
                                        .groupby('Month_Key', as_index=False)
                                        .first()
                                    )
                                    if not slab_transition_points.empty:
                                        fig_slab_base.add_trace(go.Scatter(
                                            x=slab_transition_points['Period'],
                                            y=slab_transition_points['Base_Discount_Pct_Display'],
                                            mode='markers',
                                            name='Detected Transition',
                                            marker=dict(color='#FF8C00', size=9, symbol='diamond')
                                        ), secondary_y=False)

                                    fig_slab_base.update_layout(
                                        title=f"Slab {slab_label}",
                                        height=340,
                                        xaxis_title='Daily',
                                        hovermode='x unified',
                                        showlegend=True
                                    )
                                    fig_slab_base.update_yaxes(title_text='Discount %', secondary_y=False)
                                    fig_slab_base.update_yaxes(title_text='Actual Quantity', secondary_y=True)
                                    st.plotly_chart(fig_slab_base, use_container_width=True)
                                    st.caption("Daily actual discount and estimated base discount on left axis, with actual quantity on right axis for this slab.")
                                except Exception as e:
                                    st.warning(f"Could not render slab {slab_label}: {e}")
                    st.divider()
                    
                    slab_results = []
                    for slab in slab_scope:
                        slab_txn = combined_transactions[combined_transactions['Slab'] == slab].copy()
                        slab_result = {
                            'slab': slab,
                            'qty_weight': slab_txn['Quantity'].sum() if not slab_txn.empty else 0,
                            'valid': False,
                            'reason': None,
                            'agg': None,
                            'ols': None,
                            'roi_structural_df': None,
                            'roi_tactical_df': None,
                            'avg_structural_discount_pct': np.nan,
                            'tactical_periods': 0,
                            'below_base_periods': 0
                        }

                        if slab_txn.empty:
                            slab_result['reason'] = 'No data after filters'
                            slab_results.append(slab_result)
                            continue

                        slab_agg = aggregate_for_roi(slab_txn, 'Monthly')
                        if len(slab_agg) < 3:
                            slab_result['reason'] = 'Insufficient periods for regression (need at least 3)'
                            slab_results.append(slab_result)
                            continue

                        slab_daily_agg = aggregate_for_roi(slab_txn, 'Daily').sort_values('Period', ascending=True).reset_index(drop=True)
                        slab_daily_base_discount, _ = estimate_base_discount_daily_blocks(
                            slab_daily_agg['Period'],
                            slab_daily_agg['Discount_Pct'],
                            min_upward_jump_pp=base_min_up_jump_pp,
                            min_downward_drop_pp=base_min_down_drop_pp,
                            round_step=0.5
                        )
                        slab_daily_agg['Base_Discount_Pct'] = round_discount_for_display(slab_daily_base_discount, step=0.5)
                        slab_daily_agg['Month_Key'] = slab_daily_agg['Period'].dt.to_period('M')
                        slab_monthly_base_map = (
                            slab_daily_agg.groupby('Month_Key', as_index=False)
                            .agg(Base_Discount_Pct=('Base_Discount_Pct', 'first'))
                        )

                        slab_agg_model = slab_agg.sort_values('Period', ascending=True).reset_index(drop=True).copy()
                        slab_agg_model['Month_Key'] = slab_agg_model['Period'].dt.to_period('M')
                        slab_agg_model = slab_agg_model.merge(
                            slab_monthly_base_map,
                            on='Month_Key',
                            how='left'
                        )
                        slab_agg_model['Base_Discount_Pct'] = slab_agg_model['Base_Discount_Pct'].ffill().bfill()
                        slab_agg_model['Tactical_Discount_Pct'] = (slab_agg_model['Discount_Pct'] - slab_agg_model['Base_Discount_Pct']).clip(lower=0)
                        slab_agg_model['Below_Base_Pct'] = (slab_agg_model['Base_Discount_Pct'] - slab_agg_model['Discount_Pct']).clip(lower=0)

                        ols_slab = run_two_stage_ols(
                            slab_agg_model['Discount_Pct'],
                            slab_agg_model['Quantity'],
                            slab_agg_model['Store_Count'],
                            periods=slab_agg_model['Period'],
                            base_discount_pct=slab_agg_model['Base_Discount_Pct']
                        )
                        if ols_slab is None:
                            slab_result['reason'] = 'Model could not be fitted'
                            slab_results.append(slab_result)
                            continue

                        # Keep observed Stage-1 month residual for fixed-residual structural ROI.
                        residual_map = pd.DataFrame({
                            'Period': pd.to_datetime(ols_slab['periods_clean']),
                            'Observed_Store_Residual': np.asarray(ols_slab['stage1_residuals'], dtype=float)
                        })
                        slab_agg_model['Period'] = pd.to_datetime(slab_agg_model['Period'])
                        slab_agg_model = slab_agg_model.merge(
                            residual_map,
                            on='Period',
                            how='left'
                        )

                        roi_structural_df = calculate_structural_roi_metrics(ols_slab, slab_agg_model)
                        roi_tactical_df = calculate_tactical_roi_metrics(
                            ols_slab,
                            slab_agg_model,
                            min_tactical_pp=0.5
                        )

                        slab_result.update({
                            'valid': True,
                            'agg': slab_agg_model,
                            'ols': ols_slab,
                            'roi_structural_df': roi_structural_df,
                            'roi_tactical_df': roi_tactical_df,
                            'avg_structural_discount_pct': float(slab_agg_model['Base_Discount_Pct'].mean()),
                            'tactical_periods': int((slab_agg_model['Tactical_Discount_Pct'] > 0).sum()),
                            'below_base_periods': int((slab_agg_model['Below_Base_Pct'] > 0).sum())
                        })
                        slab_results.append(slab_result)

                    valid_slab_results = [r for r in slab_results if r['valid']]
                    structural_roi_ready_results = [r for r in valid_slab_results if r['roi_structural_df'] is not None and not r['roi_structural_df'].empty]
                    tactical_roi_ready_results = [r for r in valid_slab_results if r['roi_tactical_df'] is not None and not r['roi_tactical_df'].empty]

                    graphs_tab, models_tab, roi_tab = st.tabs(['Graphs', 'Models', 'ROI'])

                    with graphs_tab:
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=(
                                f'Combined Quantity vs Discount % (Corr: {corr_qty_disc_pct:.3f})',
                                f'Combined Sales Value vs Discount % (Corr: {corr_sales_disc_pct:.3f})'
                            ),
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                            vertical_spacing=0.12,
                            shared_xaxes=True
                        )

                        fig.add_trace(
                            go.Scatter(x=daily_agg_combined['Period'], y=daily_agg_combined['Quantity'], mode='lines+markers', name='Quantity', line=dict(color='#F18F01', width=2)),
                            row=1, col=1, secondary_y=False
                        )
                        fig.add_trace(
                            go.Scatter(x=daily_agg_combined['Period'], y=daily_agg_combined['Discount_Pct_Display'], mode='lines+markers', name='Discount %', line=dict(color='#E63946', width=2, dash='dash')),
                            row=1, col=1, secondary_y=True
                        )
                        fig.add_trace(
                            go.Scatter(x=daily_agg_combined['Period'], y=daily_agg_combined['SalesValue_atBasicRate'], mode='lines+markers', name='Sales Value', line=dict(color='#20C997', width=2), showlegend=True),
                            row=2, col=1, secondary_y=False
                        )
                        fig.add_trace(
                            go.Scatter(x=daily_agg_combined['Period'], y=daily_agg_combined['Discount_Pct_Display'], mode='lines+markers', name='Discount %', line=dict(color='#E63946', width=2, dash='dash'), showlegend=False),
                            row=2, col=1, secondary_y=True
                        )
                        fig.update_xaxes(title_text=time_agg, row=2, col=1)
                        fig.update_yaxes(title_text='Quantity', row=1, col=1, secondary_y=False)
                        fig.update_yaxes(title_text='Discount %', row=1, col=1, secondary_y=True)
                        fig.update_yaxes(title_text='Sales Value', row=2, col=1, secondary_y=False)
                        fig.update_yaxes(title_text='Discount %', row=2, col=1, secondary_y=True)
                        fig.update_layout(height=750, hovermode='x', showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                        csv = combined_transactions.to_csv(index=False)
                        st.download_button(
                            label=f"Download Combined Data ({len(selected_outlets)} outlets)",
                            data=csv,
                            file_name=f"combined_{len(selected_outlets)}_outlets.csv",
                            mime='text/csv',
                            key='download_combined'
                        )

                    with models_tab:
                        if not valid_slab_results:
                            st.info('No slab has enough data to build models.')
                        elif len(valid_slab_results) == 1:
                            model_tabs = [models_tab]
                            model_payload = valid_slab_results
                        else:
                            model_tabs = st.tabs([f"Slab {r['slab']}" for r in valid_slab_results])
                            model_payload = valid_slab_results

                        for mtab, res in zip(model_tabs, model_payload):
                            with mtab:
                                slab = res['slab']
                                ols_slab = res['ols']
                                slab_agg = res['agg']
                                corr_qty_disc_slab = slab_agg['Quantity'].corr(slab_agg['Discount_Pct'])
                                corr_sales_disc_slab = slab_agg['SalesValue_atBasicRate'].corr(slab_agg['Discount_Pct'])

                                st.markdown(f"### Slab {slab} (Monthly Model)")
                                st.caption(f"Enhanced Model R2: {ols_slab['r_squared']:.4f}")
                                stage2_n_features = getattr(ols_slab.get('stage2_model', None), 'n_features_in_', 3)
                                if stage2_n_features == 4:
                                    eq_text = "Quantity = beta0 + beta1*Residual(Store) + beta2*Structural_Discount + beta3*Tactical_Discount + beta4*Lag1_Structural_Discount"
                                else:
                                    eq_text = "Quantity = beta0 + beta1*Residual(Store) + beta2*Structural_Discount + beta3*Tactical_Discount"

                                st.markdown(eq_text)
                                st.caption(
                                    f"beta1 residual(store)={ols_slab['coef_residual']:.2f}, "
                                    f"beta2 structural={ols_slab['coef_structural']:.2f}, "
                                    f"beta3 tactical={ols_slab['coef_tactical']:.2f}"
                                    + (
                                        f", beta4 lag1_structural={ols_slab.get('coef_lag1_structural', ols_slab.get('coef_lag1_stepup_structural', ols_slab.get('coef_lag1_combined', np.nan))):.2f}."
                                        if stage2_n_features == 4 else "."
                                    )
                                    + " If actual discount is below base, tactical is treated as 0."
                                )
                                if str(ols_slab.get('stage2_model_type', '')).lower() == 'customconstrainedridge':
                                    st.caption("Stage 2 uses constrained ridge, so classical p-values are not reported.")
                                st.caption(
                                    f"Stage 1 store model: Store_Count = alpha + gamma*Discount%, gamma={ols_slab['stage1_coef_discount']:.2f}"
                                )
                                with st.expander("Model Coefficients (Validation View)", expanded=False):
                                    stage1_df = pd.DataFrame([
                                        {'Term': 'alpha (intercept)', 'Coefficient': ols_slab['stage1_intercept']},
                                        {'Term': 'gamma (Discount%)', 'Coefficient': ols_slab['stage1_coef_discount']}
                                    ])
                                    stage2_df = pd.DataFrame([
                                        {'Term': 'beta0 (intercept)', 'Coefficient': ols_slab['intercept'], 'P_Value': ols_slab['p_intercept']},
                                        {'Term': 'beta1 Residual(Store)', 'Coefficient': ols_slab['coef_residual'], 'P_Value': ols_slab['p_residual']},
                                        {'Term': 'beta2 Structural_Discount', 'Coefficient': ols_slab['coef_structural'], 'P_Value': ols_slab['p_structural']},
                                        {'Term': 'beta3 Tactical_Discount', 'Coefficient': ols_slab['coef_tactical'], 'P_Value': ols_slab['p_tactical']}
                                    ])
                                    if stage2_n_features == 4:
                                        stage2_df = pd.concat([
                                            stage2_df,
                                            pd.DataFrame([{
                                                'Term': 'beta4 Lag1_Structural_Discount',
                                                'Coefficient': ols_slab.get('coef_lag1_structural', ols_slab.get('coef_lag1_stepup_structural', ols_slab.get('coef_lag1_combined', np.nan))),
                                                'P_Value': ols_slab.get('p_lag1_structural', ols_slab.get('p_lag1_stepup_structural', ols_slab.get('p_lag1_combined', np.nan)))
                                            }])
                                        ], ignore_index=True)
                                    st.markdown("**Stage 1 Coefficients**")
                                    st.dataframe(
                                        stage1_df.style.format({'Coefficient': '{:.4f}'}),
                                        use_container_width=True,
                                        height=130
                                    )
                                    st.markdown("**Stage 2 Coefficients**")
                                    st.dataframe(
                                        stage2_df.style.format({'Coefficient': '{:.4f}', 'P_Value': '{:.6f}'}),
                                        use_container_width=True,
                                        height=220
                                    )

                                fig_stage1 = go.Figure()
                                fig_stage1.add_trace(go.Scatter(
                                    x=ols_slab['periods_clean'], y=ols_slab['store_clean'],
                                    mode='lines+markers', name='Actual Store Count',
                                    line=dict(color='#8B4513', width=3), marker=dict(size=8)
                                ))
                                fig_stage1.add_trace(go.Scatter(
                                    x=ols_slab['periods_clean'], y=ols_slab['stage1_store_pred'],
                                    mode='lines+markers', name=f"Predicted Store Count (R2={ols_slab['stage1_r_squared']:.4f})",
                                    line=dict(color='#FFA500', width=2, dash='dash'), marker=dict(size=6)
                                ))
                                fig_stage1.update_layout(
                                    xaxis_title='Month', yaxis_title='Store Count', height=350,
                                    hovermode='x unified', showlegend=True,
                                    title=f"Slab {slab}: Stage 1 (Store_Count ~ Discount%)"
                                )
                                st.plotly_chart(fig_stage1, use_container_width=True)

                                fig_ols = go.Figure()
                                fig_ols.add_trace(go.Scatter(
                                    x=ols_slab['periods_clean'], y=ols_slab['y_clean'],
                                    mode='lines+markers', name='Actual Quantity',
                                    line=dict(color='#000000', width=3), marker=dict(size=8)
                                ))
                                fig_ols.add_trace(go.Scatter(
                                    x=ols_slab['periods_clean'], y=ols_slab['y_pred'],
                                    mode='lines+markers', name=f"Enhanced Model (R2={ols_slab['r_squared']:.4f})",
                                    line=dict(color='#2ca02c', width=2, dash='dot'), marker=dict(size=6)
                                ))
                                fig_ols.update_layout(
                                    xaxis_title='Month', yaxis_title='Quantity', height=420,
                                    hovermode='x unified', showlegend=True,
                                    title=f"Slab {slab}: Actual vs Predicted Quantity"
                                )
                                st.plotly_chart(fig_ols, use_container_width=True)

                        invalid_models = [r for r in slab_results if not r['valid']]
                        if invalid_models:
                            with st.expander('Slabs Skipped in Modeling', expanded=False):
                                for r in invalid_models:
                                    st.markdown(f"- Slab {r['slab']}: {r['reason']}")

                    with roi_tab:
                        if not structural_roi_ready_results:
                            st.info('Structural ROI unavailable. It is computed only on base step-up episodes.')
                        else:
                            st.caption('Tactical ROI is hidden for now. Structural ROI is episode-based: for each base step-up (Base_1 -> Base_2), ROI is computed across the full hold period where Base_2 stays active.')

                            if len(structural_roi_ready_results) > 1:
                                st.markdown('### Combined Structural ROI (Quantity-Weighted Across Slabs)')
                                summary_rows = []
                                for r in structural_roi_ready_results:
                                    roi_df = r['roi_structural_df']
                                    summary_rows.append({
                                        'Slab': r['slab'],
                                        'Weight_Qty': r['qty_weight'],
                                        'Avg_Structural_%': r['avg_structural_discount_pct'],
                                        'Episodes': len(roi_df),
                                        'Avg_ROI_Structural': roi_df['ROI_1mo'].mean(),
                                        'Spend': roi_df['Spend'].sum(),
                                        'Incremental_Revenue': roi_df['Incremental_Revenue'].sum()
                                    })
                                summary_df = pd.DataFrame(summary_rows)
                                weights = summary_df['Weight_Qty'].astype(float).values
                                roi1_vals = summary_df['Avg_ROI_Structural'].astype(float).values
                                weighted_roi_1 = np.average(roi1_vals, weights=weights) if weights.sum() > 0 else np.nan

                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    st.metric('Weighted Avg Structural ROI', f"{weighted_roi_1:.2f}x")
                                with c2:
                                    st.metric('Episodes', f"{int(summary_df['Episodes'].sum())}")
                                with c3:
                                    st.metric('Total Structural Spend', f"{summary_df['Spend'].sum():,.0f}")
                                with c4:
                                    st.metric('Total Structural Incremental Revenue', f"{summary_df['Incremental_Revenue'].sum():,.0f}")

                                st.dataframe(
                                    summary_df.style.format({
                                        'Weight_Qty': '{:,.0f}',
                                        'Avg_Structural_%': '{:.2f}%',
                                        'Episodes': '{:,.0f}',
                                        'Avg_ROI_Structural': '{:.2f}x',
                                        'Spend': '{:,.0f}',
                                        'Incremental_Revenue': '{:,.0f}'
                                    }),
                                    use_container_width=True,
                                    height=220
                                )

                            if len(structural_roi_ready_results) == 1:
                                structural_tabs = [roi_tab]
                            else:
                                structural_tabs = st.tabs([f"Slab {r['slab']} Structural ROI" for r in structural_roi_ready_results])

                            for rtab, res in zip(structural_tabs, structural_roi_ready_results):
                                with rtab:
                                    slab = res['slab']
                                    roi_df = res['roi_structural_df']
                                    ols_slab = res['ols']
                                    slab_agg = res['agg'].sort_values('Period').reset_index(drop=True).copy()
                                    tactical_df = res.get('roi_tactical_df', None)
                                    if tactical_df is not None and not tactical_df.empty:
                                        tactical_df = tactical_df.sort_values('Period').reset_index(drop=True).copy()

                                    st.markdown(f"### Structural ROI - Slab {slab} (Base Step-up Episodes)")
                                    c1, c2, c3, c4 = st.columns(4)
                                    with c1:
                                        st.metric('Avg Structural ROI', f"{roi_df['ROI_1mo'].mean():.2f}x")
                                    with c2:
                                        st.metric('Episodes', f"{len(roi_df)}")
                                    with c3:
                                        st.metric('Total Structural Spend', f"{roi_df['Spend'].sum():,.0f}")
                                    with c4:
                                        st.metric('Total Structural Incremental Revenue', f"{roi_df['Incremental_Revenue'].sum():,.0f}")

                                    roi_month_rows = []
                                    for _, epi in roi_df.iterrows():
                                        epi_start = pd.to_datetime(epi['Period'])
                                        epi_end = pd.to_datetime(epi['Episode_End_Period'])
                                        hold_mask = (slab_agg['Period'] >= epi_start) & (slab_agg['Period'] <= epi_end)
                                        hold_periods = slab_agg.loc[hold_mask, 'Period']
                                        for p in hold_periods:
                                            roi_month_rows.append({
                                                'Period': p,
                                                'ROI_1mo': float(epi['ROI_1mo'])
                                            })
                                    roi_month_df = pd.DataFrame(roi_month_rows)
                                    base_line_col = 'Base_Discount_Pct' if 'Base_Discount_Pct' in slab_agg.columns else 'Discount_Pct'

                                    structural_bar_df = (
                                        roi_month_df.groupby('Period', as_index=False)['ROI_1mo']
                                        .mean()
                                        .sort_values('Period')
                                    ) if not roi_month_df.empty else pd.DataFrame(columns=['Period', 'ROI_1mo'])
                                    tactical_bar_df = (
                                        tactical_df.groupby('Period', as_index=False)['ROI_1mo']
                                        .mean()
                                        .sort_values('Period')
                                    ) if tactical_df is not None and not tactical_df.empty else pd.DataFrame(columns=['Period', 'ROI_1mo'])

                                    # Build ROI + Profit ROI in one combined subplot later.

                                    st.markdown("#### Profit View (COGS-based)")
                                    base_price_series = pd.to_numeric(slab_agg.get('Base_Price', pd.Series(dtype=float)), errors='coerce').dropna()
                                    first_base_price = float(base_price_series.iloc[0]) if not base_price_series.empty else 100.0
                                    default_cogs = float(round(first_base_price * 0.5))
                                    cogs_per_unit = float(st.number_input(
                                        "Assumed COGS per unit",
                                        min_value=0.0,
                                        value=default_cogs,
                                        step=1.0,
                                        key=f"cogs_per_unit_{slab}"
                                    ))
                                    st.caption(
                                        f"Default COGS = round(first base price x 50%) = round({first_base_price:.2f} x 0.5) = {default_cogs:.2f}"
                                    )

                                    struct_profit_df = roi_df.copy()
                                    struct_profit_df['Baseline_Profit'] = struct_profit_df['Baseline_Revenue'] - (cogs_per_unit * struct_profit_df['Baseline_Quantity'])
                                    struct_profit_df['Predicted_Profit'] = struct_profit_df['Predicted_Revenue'] - (cogs_per_unit * struct_profit_df['Predicted_Quantity'])
                                    struct_profit_df['Incremental_Profit'] = struct_profit_df['Predicted_Profit'] - struct_profit_df['Baseline_Profit']
                                    struct_profit_df['Profit_ROI'] = struct_profit_df['Incremental_Profit'] / struct_profit_df['Spend'].replace(0, np.nan)

                                    tactical_profit_df = pd.DataFrame()
                                    if tactical_df is not None and not tactical_df.empty:
                                        tactical_profit_df = tactical_df.copy()
                                        tactical_profit_df['Baseline_Profit'] = tactical_profit_df['Baseline_Revenue'] - (cogs_per_unit * tactical_profit_df['Baseline_Quantity'])
                                        tactical_profit_df['Predicted_Profit'] = tactical_profit_df['Predicted_Revenue'] - (cogs_per_unit * tactical_profit_df['Predicted_Quantity'])
                                        tactical_profit_df['Incremental_Profit'] = tactical_profit_df['Predicted_Profit'] - tactical_profit_df['Baseline_Profit']
                                        tactical_profit_df['Profit_ROI'] = tactical_profit_df['Incremental_Profit'] / tactical_profit_df['Spend'].replace(0, np.nan)

                                    struct_plot_df = (
                                        struct_profit_df[['Period', 'Profit_ROI']]
                                        .groupby('Period', as_index=False)
                                        .agg(Profit_ROI=('Profit_ROI', 'mean'))
                                        .sort_values('Period')
                                    )
                                    tactical_plot_df = (
                                        tactical_profit_df[['Period', 'Profit_ROI']]
                                        .groupby('Period', as_index=False)
                                        .agg(Profit_ROI=('Profit_ROI', 'mean'))
                                        .sort_values('Period')
                                    ) if not tactical_profit_df.empty else pd.DataFrame(columns=['Period', 'Profit_ROI'])

                                    combined_fig = make_subplots(
                                        rows=2,
                                        cols=1,
                                        shared_xaxes=True,
                                        vertical_spacing=0.16,
                                        specs=[[{"secondary_y": True}], [{}]],
                                        subplot_titles=(
                                            "ROI View: Structural vs Tactical + Base Discount",
                                            "Profit ROI View (COGS-adjusted)"
                                        )
                                    )

                                    if not structural_bar_df.empty:
                                        combined_fig.add_trace(
                                            go.Bar(
                                                x=structural_bar_df['Period'],
                                                y=structural_bar_df['ROI_1mo'],
                                                name='Structural ROI',
                                                marker_color='#4B99B8',
                                                opacity=0.9,
                                                offsetgroup='struct',
                                                text=structural_bar_df['ROI_1mo'].map(lambda v: f"{v:.2f}x"),
                                                textposition='outside',
                                                textfont=dict(size=12, color='#1D3557'),
                                                cliponaxis=False,
                                                hovertemplate='Month: %{x|%b %Y}<br>Structural ROI: %{y:.2f}x<extra></extra>'
                                            ),
                                            row=1,
                                            col=1,
                                            secondary_y=False
                                        )
                                    if not tactical_bar_df.empty:
                                        combined_fig.add_trace(
                                            go.Bar(
                                                x=tactical_bar_df['Period'],
                                                y=tactical_bar_df['ROI_1mo'],
                                                name='Tactical ROI',
                                                marker_color='#F4A025',
                                                opacity=0.95,
                                                offsetgroup='tact',
                                                text=tactical_bar_df['ROI_1mo'].map(lambda v: f"{v:.2f}x"),
                                                textposition='outside',
                                                textfont=dict(size=12, color='#7A4100'),
                                                cliponaxis=False,
                                                hovertemplate='Month: %{x|%b %Y}<br>Tactical ROI: %{y:.2f}x<extra></extra>'
                                            ),
                                            row=1,
                                            col=1,
                                            secondary_y=False
                                        )
                                    combined_fig.add_trace(
                                        go.Scatter(
                                            x=slab_agg['Period'],
                                            y=slab_agg[base_line_col],
                                            mode='lines+markers',
                                            name='Estimated Base Discount %',
                                            line=dict(color='#1D3557', width=3, shape='hv'),
                                            marker=dict(size=6),
                                            hovertemplate='Month: %{x|%b %Y}<br>Base Discount: %{y:.2f}%<extra></extra>'
                                        ),
                                        row=1,
                                        col=1,
                                        secondary_y=True
                                    )

                                    if not struct_plot_df.empty:
                                        combined_fig.add_trace(
                                            go.Bar(
                                                x=struct_plot_df['Period'],
                                                y=struct_plot_df['Profit_ROI'],
                                                name='Structural Profit ROI',
                                                marker_color='#1D3557',
                                                opacity=0.9,
                                                offsetgroup='struct_profit',
                                                text=struct_plot_df['Profit_ROI'].map(lambda v: f"{v:.2f}x"),
                                                textposition='outside',
                                                textfont=dict(size=12, color='#1D3557'),
                                                cliponaxis=False,
                                                hovertemplate='Month: %{x|%b %Y}<br>Structural Profit ROI: %{y:.2f}x<extra></extra>'
                                            ),
                                            row=2,
                                            col=1
                                        )
                                    if not tactical_plot_df.empty:
                                        combined_fig.add_trace(
                                            go.Bar(
                                                x=tactical_plot_df['Period'],
                                                y=tactical_plot_df['Profit_ROI'],
                                                name='Tactical Profit ROI',
                                                marker_color='#A86000',
                                                opacity=0.9,
                                                offsetgroup='tact_profit',
                                                text=tactical_plot_df['Profit_ROI'].map(lambda v: f"{v:.2f}x"),
                                                textposition='outside',
                                                textfont=dict(size=12, color='#7A4100'),
                                                cliponaxis=False,
                                                hovertemplate='Month: %{x|%b %Y}<br>Tactical Profit ROI: %{y:.2f}x<extra></extra>'
                                            ),
                                            row=2,
                                            col=1
                                        )

                                    combined_fig.update_layout(
                                        title=f"Slab {slab}: ROI and Profit ROI (Combined Subplots)",
                                        height=820,
                                        hovermode='x unified',
                                        showlegend=True,
                                        barmode='group',
                                        uniformtext_minsize=11,
                                        uniformtext_mode='show',
                                        margin=dict(t=95)
                                    )
                                    combined_fig.update_xaxes(title_text='Month', row=2, col=1)
                                    combined_fig.update_yaxes(title_text='ROI (Revenue / Spend)', row=1, col=1, secondary_y=False)
                                    combined_fig.update_yaxes(title_text='Estimated Base Discount %', row=1, col=1, secondary_y=True)
                                    combined_fig.update_yaxes(title_text='Profit ROI (Profit / Spend)', row=2, col=1)
                                    st.plotly_chart(combined_fig, use_container_width=True)
                                    st.caption("Top subplot: ROI with base discount. Bottom subplot: Profit ROI using selected COGS.")

                                    st.markdown("#### 12-Month Promo Calendar (Apr-Mar)")
                                    plan_df = slab_agg.sort_values('Period').reset_index(drop=True).copy()
                                    if not plan_df.empty:
                                        plan_df['Period'] = pd.to_datetime(plan_df['Period'], errors='coerce')
                                        plan_df = plan_df.dropna(subset=['Period']).reset_index(drop=True)
                                        plan_df['Month'] = plan_df['Period'].dt.to_period('M')
                                        plan_df['Base_Price'] = pd.to_numeric(plan_df.get('Base_Price', np.nan), errors='coerce')
                                        plan_df['Discount_Pct'] = pd.to_numeric(plan_df.get('Discount_Pct', np.nan), errors='coerce')
                                        plan_df['Store_Count'] = pd.to_numeric(plan_df.get('Store_Count', np.nan), errors='coerce')
                                        plan_df['Observed_Store_Residual'] = pd.to_numeric(plan_df.get('Observed_Store_Residual', np.nan), errors='coerce')
                                        plan_df['Structural_Discount_Pct'] = pd.to_numeric(
                                            plan_df.get('Base_Discount_Pct', plan_df.get('Discount_Pct', np.nan)),
                                            errors='coerce'
                                        )

                                        available_months = sorted(plan_df['Month'].dropna().unique())
                                        if not available_months:
                                            st.info("No valid monthly data available for planning.")
                                            plan_months = []
                                        else:
                                            # Planner is fixed to FY2026 for now: Apr-2025 to Mar-2026.
                                            plan_start_year = 2025
                                            start_m = pd.Period(f"{plan_start_year}-04", freq='M')
                                            plan_months = list(pd.period_range(start=start_m, periods=12, freq='M'))

                                        if len(plan_months) < 3:
                                            st.info("Need at least 3 monthly points for planning view.")
                                        else:
                                            rows = []
                                            for m in plan_months:
                                                mm = plan_df[plan_df['Month'] == m]
                                                if mm.empty:
                                                    rows.append({
                                                        'Month': m,
                                                        'Period': m.to_timestamp(how='start'),
                                                        'Base_Price': np.nan,
                                                        'Discount_Pct': np.nan,
                                                        'Store_Count': np.nan,
                                                        'Observed_Store_Residual': np.nan,
                                                        'Structural_Discount_Pct': np.nan
                                                    })
                                                else:
                                                    r = mm.iloc[-1]
                                                    rows.append({
                                                        'Month': m,
                                                        'Period': m.to_timestamp(how='start'),
                                                        'Base_Price': float(r['Base_Price']) if pd.notna(r['Base_Price']) else np.nan,
                                                        'Discount_Pct': float(r['Discount_Pct']) if pd.notna(r['Discount_Pct']) else np.nan,
                                                        'Store_Count': float(r['Store_Count']) if pd.notna(r['Store_Count']) else np.nan,
                                                        'Observed_Store_Residual': float(r['Observed_Store_Residual']) if pd.notna(r['Observed_Store_Residual']) else np.nan,
                                                        'Structural_Discount_Pct': float(r['Structural_Discount_Pct']) if pd.notna(r['Structural_Discount_Pct']) else np.nan
                                                    })
                                            plan_template = pd.DataFrame(rows).sort_values('Period').reset_index(drop=True)
                                            plan_template['Structural_Discount_Pct'] = plan_template['Structural_Discount_Pct'].ffill().bfill()

                                            # Planning base price starts from latest known base price (e.g., Nov latest),
                                            # then cascades month-by-month with manual override.
                                            base_price_series_plan = pd.to_numeric(plan_df.sort_values('Period')['Base_Price'], errors='coerce').dropna()
                                            default_bp_plan = float(base_price_series_plan.iloc[-1]) if not base_price_series_plan.empty else first_base_price
                                            default_bp_plan = float(np.round(default_bp_plan * 2.0) / 2.0)
                                            plan_template['Base_Price'] = plan_template['Base_Price'].fillna(default_bp_plan)

                                            missing_resid = plan_template['Observed_Store_Residual'].isna()
                                            if missing_resid.any():
                                                x_actual = plan_template['Discount_Pct'].fillna(plan_template['Structural_Discount_Pct']).to_numpy(dtype=float).reshape(-1, 1)
                                                pred_store = ols_slab['stage1_model'].predict(x_actual)
                                                fallback_resid = plan_template['Store_Count'].fillna(0.0).to_numpy(dtype=float) - pred_store
                                                plan_template.loc[missing_resid, 'Observed_Store_Residual'] = fallback_resid[missing_resid.to_numpy()]
                                            plan_template['Observed_Store_Residual'] = plan_template['Observed_Store_Residual'].fillna(0.0)
                                            observed_struct = plan_template['Structural_Discount_Pct'].to_numpy(dtype=float)

                                            prev_m = plan_months[0] - 1
                                            prev_row = plan_df[plan_df['Month'] == prev_m]
                                            prev_struct = float(prev_row['Structural_Discount_Pct'].iloc[-1]) if not prev_row.empty and pd.notna(prev_row['Structural_Discount_Pct'].iloc[-1]) else float(observed_struct[0])

                                            n_plan = len(plan_template)
                                            # Default planner promo path from previous complete FY (Apr-2024 to Mar-2025),
                                            # mapped month-to-month into FY2026 plan months.
                                            ref_start = pd.Period("2024-04", freq='M')
                                            ref_months = list(pd.period_range(start=ref_start, periods=12, freq='M'))
                                            ref_df = (
                                                plan_df[plan_df['Month'].isin(ref_months)]
                                                .sort_values('Period')
                                                .drop_duplicates(subset=['Month'], keep='last')
                                            )
                                            ref_struct_map = {
                                                m: float(v)
                                                for m, v in zip(
                                                    ref_df['Month'],
                                                    pd.to_numeric(ref_df['Structural_Discount_Pct'], errors='coerce')
                                                )
                                                if pd.notna(v)
                                            }
                                            if len(ref_struct_map) >= 12:
                                                default_struct_plan = np.asarray(
                                                    [float(ref_struct_map.get((m - 12), observed_struct[i])) for i, m in enumerate(plan_months)],
                                                    dtype=float
                                                )
                                            else:
                                                default_struct_plan = np.empty(n_plan, dtype=float)
                                                default_struct_plan[0] = prev_struct
                                                if n_plan > 1:
                                                    default_struct_plan[1:] = observed_struct[:-1]
                                            default_struct_plan = np.clip(default_struct_plan, 0.0, 60.0)
                                            # In planner view, "Current" path is the default FY-based plan path.
                                            # This ensures Reset aligns Current and Planned lines.
                                            baseline_struct = default_struct_plan.copy()

                                            ui_seed_key = f"plan_ui_seed_{slab}"
                                            if ui_seed_key not in st.session_state:
                                                st.session_state[ui_seed_key] = 0
                                            if st.button("Reset To Default", key=f"plan_reset_btn_{slab}"):
                                                # Clear existing planner widget states for this slab so both
                                                # promo sliders and base-price inputs rehydrate to defaults.
                                                reset_prefixes = (
                                                    f"plan_struct_{slab}_",
                                                    f"plan_bp_{slab}_",
                                                )
                                                for k in list(st.session_state.keys()):
                                                    if any(str(k).startswith(pref) for pref in reset_prefixes):
                                                        del st.session_state[k]
                                                st.session_state[ui_seed_key] += 1
                                                st.rerun()
                                            ui_seed = int(st.session_state[ui_seed_key])

                                            left_panel, right_panel = st.columns([1.25, 1.75], gap="large")
                                            planned_struct = []
                                            planned_base_price = []

                                            with left_panel:
                                                st.markdown("**Promo Calendar Inputs**")
                                                h1, h2, h3, h4 = st.columns([1.1, 1.0, 1.9, 1.2])
                                                with h1:
                                                    st.caption("Month")
                                                with h2:
                                                    st.caption("Default %")
                                                with h3:
                                                    st.caption("Planned Promo %")
                                                with h4:
                                                    st.caption("Base Price")

                                                prev_bp_input = float(default_bp_plan)
                                                try:
                                                    inputs_container = st.container(height=520)
                                                except TypeError:
                                                    inputs_container = st.container()
                                                with inputs_container:
                                                    for i, row in plan_template.iterrows():
                                                        label = pd.to_datetime(row['Period']).strftime('%b')
                                                        month_id = str(row['Month'])
                                                        struct_key = f"plan_struct_{slab}_{month_id}_{ui_seed}"
                                                        row_cols = st.columns([1.1, 1.0, 1.9, 1.2])
                                                        with row_cols[0]:
                                                            st.markdown(label)
                                                        with row_cols[1]:
                                                            st.markdown(f"{default_struct_plan[i]:.1f}%")
                                                        with row_cols[2]:
                                                            promo_val = float(st.slider(
                                                                f"Planned Promo % {label}",
                                                                min_value=0.0,
                                                                max_value=60.0,
                                                                value=float(default_struct_plan[i]),
                                                                step=0.5,
                                                                key=struct_key,
                                                                label_visibility="collapsed"
                                                            ))
                                                        with row_cols[3]:
                                                            bp_key = f"plan_bp_{slab}_{month_id}_{ui_seed}_{round(prev_bp_input, 1)}"
                                                            bp_val_raw = float(st.number_input(
                                                                f"Base Price {label}",
                                                                min_value=0.0,
                                                                value=float(np.round(prev_bp_input * 2.0) / 2.0),
                                                                step=0.5,
                                                                key=bp_key,
                                                                label_visibility="collapsed",
                                                                format="%.1f"
                                                            ))
                                                            bp_val = float(np.round(bp_val_raw * 2.0) / 2.0)
                                                        planned_struct.append(promo_val)
                                                        planned_base_price.append(bp_val)
                                                        prev_bp_input = bp_val
                                            planned_struct = np.asarray(planned_struct, dtype=float)
                                            planned_base_price = np.asarray(planned_base_price, dtype=float)

                                            lag_old = np.empty(n_plan, dtype=float)
                                            lag_new = np.empty(n_plan, dtype=float)
                                            lag_old[0] = prev_struct
                                            lag_new[0] = prev_struct
                                            if n_plan > 1:
                                                lag_old[1:] = baseline_struct[:-1]
                                                lag_new[1:] = planned_struct[:-1]

                                            residual_arr = plan_template['Observed_Store_Residual'].to_numpy(dtype=float)
                                            base_price_arr = planned_base_price
                                            zeros_arr = np.zeros(n_plan, dtype=float)

                                            stage2_n_features = getattr(ols_slab.get('stage2_model', None), 'n_features_in_', 4)
                                            if stage2_n_features == 4:
                                                x_old = np.column_stack([residual_arr, baseline_struct, zeros_arr, lag_old])
                                                x_new = np.column_stack([residual_arr, planned_struct, zeros_arr, lag_new])
                                            elif stage2_n_features == 3:
                                                x_old = np.column_stack([residual_arr, baseline_struct, zeros_arr])
                                                x_new = np.column_stack([residual_arr, planned_struct, zeros_arr])
                                            elif stage2_n_features == 2:
                                                x_old = np.column_stack([residual_arr, baseline_struct])
                                                x_new = np.column_stack([residual_arr, planned_struct])
                                            else:
                                                x_old = residual_arr.reshape(-1, 1)
                                                x_new = residual_arr.reshape(-1, 1)

                                            qty_old = np.maximum(ols_slab['stage2_model'].predict(x_old), 0.0)
                                            qty_new = np.maximum(ols_slab['stage2_model'].predict(x_new), 0.0)
                                            price_old = base_price_arr * (1.0 - baseline_struct / 100.0)
                                            price_new = base_price_arr * (1.0 - planned_struct / 100.0)
                                            rev_old = qty_old * price_old
                                            rev_new = qty_new * price_new
                                            prof_old = qty_old * (price_old - cogs_per_unit)
                                            prof_new = qty_new * (price_new - cogs_per_unit)

                                            total_rev_old = float(np.nansum(rev_old))
                                            total_rev_new = float(np.nansum(rev_new))
                                            total_prof_old = float(np.nansum(prof_old))
                                            total_prof_new = float(np.nansum(prof_new))
                                            total_qty_old = float(np.nansum(qty_old))
                                            total_qty_new = float(np.nansum(qty_new))
                                            rev_delta = total_rev_new - total_rev_old
                                            prof_delta = total_prof_new - total_prof_old
                                            qty_delta = total_qty_new - total_qty_old
                                            rev_pct = (rev_delta / total_rev_old * 100.0) if abs(total_rev_old) > 1e-12 else np.nan
                                            prof_pct = (prof_delta / total_prof_old * 100.0) if abs(total_prof_old) > 1e-12 else np.nan
                                            qty_pct = (qty_delta / total_qty_old * 100.0) if abs(total_qty_old) > 1e-12 else np.nan
                                            rev_delta_text = f"{rev_delta:+,.0f} ({rev_pct:+.1f}%)" if pd.notna(rev_pct) else f"{rev_delta:+,.0f}"
                                            prof_delta_text = f"{prof_delta:+,.0f} ({prof_pct:+.1f}%)" if pd.notna(prof_pct) else f"{prof_delta:+,.0f}"

                                            with right_panel:
                                                avg_promo_old = float(np.nanmean(baseline_struct))
                                                avg_promo_new = float(np.nanmean(planned_struct))
                                                promo_delta_pp = avg_promo_new - avg_promo_old
                                                promo_pct = (promo_delta_pp / avg_promo_old * 100.0) if abs(avg_promo_old) > 1e-12 else np.nan
                                                step_up_pp = np.clip(planned_struct - baseline_struct, 0.0, None)
                                                spend_monthly = base_price_arr * (step_up_pp / 100.0) * qty_new
                                                total_spend_plan = float(np.nansum(spend_monthly))
                                                roi_revenue = (rev_delta / total_spend_plan) if total_spend_plan > 1e-12 else np.nan
                                                roi_revenue_pct = roi_revenue * 100.0 if pd.notna(roi_revenue) else np.nan

                                                def _fmt_pct_change(pct_val):
                                                    if pd.isna(pct_val):
                                                        return "NA"
                                                    if pct_val > 1e-9:
                                                        return f"Increase +{pct_val:.1f}%"
                                                    if pct_val < -1e-9:
                                                        return f"Decrease {pct_val:.1f}%"
                                                    return "No Change 0.0%"

                                                b1, b2, b3, b4, b5 = st.columns(5)
                                                with b1:
                                                    with st.container(border=True):
                                                        st.caption("Volume")
                                                        st.metric(
                                                            "Volume Change",
                                                            _fmt_pct_change(qty_pct)
                                                        )
                                                with b2:
                                                    with st.container(border=True):
                                                        st.caption("Revenue")
                                                        st.metric(
                                                            "Revenue Change",
                                                            _fmt_pct_change(rev_pct)
                                                        )
                                                with b3:
                                                    with st.container(border=True):
                                                        st.caption("Profit")
                                                        st.metric(
                                                            "Profit Change",
                                                            _fmt_pct_change(prof_pct)
                                                        )
                                                with b4:
                                                    with st.container(border=True):
                                                        st.caption("Promo")
                                                        st.metric(
                                                            "Promo Change",
                                                            _fmt_pct_change(promo_pct)
                                                        )
                                                with b5:
                                                    with st.container(border=True):
                                                        st.caption("ROI")
                                                        st.metric(
                                                            "ROI Change",
                                                            _fmt_pct_change(roi_revenue_pct)
                                                        )

                                                path_fig = go.Figure()
                                                path_fig.add_trace(go.Scatter(
                                                    x=plan_template['Period'],
                                                    y=baseline_struct,
                                                    mode='lines',
                                                    name='Current Promo %',
                                                    line=dict(color='#6B7280', width=2, dash='dot', shape='hv')
                                                ))
                                                path_fig.add_trace(go.Scatter(
                                                    x=plan_template['Period'],
                                                    y=planned_struct,
                                                    mode='lines',
                                                    name='Planned Promo %',
                                                    line=dict(color='#1D4E89', width=3, shape='hv')
                                                ))
                                                path_fig.update_layout(
                                                    title=f"Slab {slab}: Promo Calendar",
                                                    height=300,
                                                    hovermode='x unified',
                                                    xaxis_title='Month',
                                                    yaxis_title='Structural Discount %',
                                                    template='plotly_white',
                                                    legend=dict(orientation='h', y=-0.33, x=0.5, xanchor='center'),
                                                    margin=dict(b=95)
                                                )
                                                st.plotly_chart(
                                                    path_fig,
                                                    use_container_width=True,
                                                    key=f"plan_calendar_path_chart_{slab}"
                                                )
                                                st.caption("Boxes show total change for the 12-month plan. ROI uses sum(Incremental Revenue) / sum(Plan Spend), where plan spend is based on month-wise step-up vs default.")
                                    with st.expander(f"How Structural ROI Is Calculated (Step-by-Step) - Slab {slab}", expanded=False):
                                        beta0 = float(ols_slab['intercept'])
                                        beta1 = float(ols_slab['coef_residual'])
                                        beta2 = float(ols_slab['coef_structural'])
                                        beta3 = float(ols_slab['coef_tactical'])
                                        beta4 = float(
                                            ols_slab.get(
                                                'coef_lag1_structural',
                                                ols_slab.get('coef_lag1_stepup_structural', ols_slab.get('coef_lag1_combined', 0.0))
                                            )
                                        )
                                        alpha = float(ols_slab['stage1_intercept'])
                                        gamma = float(ols_slab['stage1_coef_discount'])
                                        stage2_n_features = getattr(ols_slab.get('stage2_model', None), 'n_features_in_', 3)

                                        st.markdown("**Model coefficients used in this slab**")
                                        st.markdown(
                                            f"- Store model: `Store = alpha + gamma x Discount%` where `alpha={alpha:.4f}`, `gamma={gamma:.4f}`  \n"
                                            + (
                                                f"- Quantity model: `Qty = beta0 + beta1 x StoreResidual + beta2 x Structural + beta3 x Tactical + beta4 x Lag1Structural` where "
                                                f"`beta0={beta0:.4f}`, `beta1={beta1:.4f}`, `beta2={beta2:.4f}`, `beta3={beta3:.4f}`, `beta4={beta4:.4f}`"
                                                if stage2_n_features == 4
                                                else f"- Quantity model: `Qty = beta0 + beta1 x StoreResidual + beta2 x Structural + beta3 x Tactical` where "
                                                     f"`beta0={beta0:.4f}`, `beta1={beta1:.4f}`, `beta2={beta2:.4f}`, `beta3={beta3:.4f}`"
                                            )
                                        )
                                        st.caption("For Structural ROI, Tactical is fixed to 0 and the observed month store residual is kept same in both old-base and new-base scenarios.")

                                        episodes_df = roi_df.reset_index(drop=True).copy()
                                        if episodes_df.empty:
                                            st.info("No structural step-up episodes available.")
                                        else:
                                            episode_options = []
                                            for epi_idx, epi in episodes_df.iterrows():
                                                epi_start = pd.to_datetime(epi['Period'])
                                                epi_end = pd.to_datetime(epi['Episode_End_Period'])
                                                prev_base = float(epi['Prev_Structural_Discount_Pct'])
                                                curr_base = float(epi['Structural_Discount_Pct'])
                                                episode_options.append(
                                                    f"Episode {epi_idx + 1} | {epi_start.strftime('%Y-%m-%d')} to {epi_end.strftime('%Y-%m-%d')} | {prev_base:.2f}% -> {curr_base:.2f}%"
                                                )

                                            selected_episode_label = st.selectbox(
                                                "Select episode for detailed calculation",
                                                options=episode_options,
                                                key=f"structural_episode_select_{slab}"
                                            )
                                            selected_idx = episode_options.index(selected_episode_label)
                                            epi = episodes_df.iloc[selected_idx]

                                            epi_start = pd.to_datetime(epi['Period'])
                                            epi_end = pd.to_datetime(epi['Episode_End_Period'])
                                            prev_base = float(epi['Prev_Structural_Discount_Pct'])
                                            curr_base = float(epi['Structural_Discount_Pct'])
                                            step_up = float(epi['Structural_Step_Up_Pct'])
                                            hold = slab_agg[(slab_agg['Period'] >= epi_start) & (slab_agg['Period'] <= epi_end)].copy()

                                            if hold.empty:
                                                st.info("No monthly rows found for selected episode.")
                                            else:
                                                hold = hold.sort_values('Period').reset_index(drop=True)
                                                st.markdown(
                                                    f"**Selected episode:** {epi_start.strftime('%Y-%m-%d')} to {epi_end.strftime('%Y-%m-%d')} "
                                                    f"| Old Base {prev_base:.2f}% -> New Base {curr_base:.2f}%"
                                                )
                                                st.markdown(
                                                    f"**Episode constants**  \n"
                                                    f"- Old Base = {prev_base:.2f}%  \n"
                                                    f"- New Base = {curr_base:.2f}%  \n"
                                                    f"- Step Up = {step_up:.2f}%"
                                                )

                                                for month_idx, h in hold.iterrows():
                                                    store_actual = float(h['Store_Count'])
                                                    observed_discount = float(h['Discount_Pct']) if pd.notna(h['Discount_Pct']) else curr_base
                                                    lag1_old = prev_base
                                                    lag1_new = prev_base if month_idx == 0 else curr_base
                                                    base_price = float(h['Base_Price']) if pd.notna(h['Base_Price']) else 100.0

                                                    pred_store_observed = alpha + gamma * observed_discount
                                                    if 'Observed_Store_Residual' in h and pd.notna(h['Observed_Store_Residual']):
                                                        resid_fixed = float(h['Observed_Store_Residual'])
                                                    else:
                                                        resid_fixed = store_actual - pred_store_observed

                                                    q_base_intercept = beta0
                                                    q_base_resid = beta1 * resid_fixed
                                                    q_base_struct = beta2 * prev_base
                                                    q_base_lag = beta4 * lag1_old if stage2_n_features == 4 else 0.0
                                                    q_base = q_base_intercept + q_base_resid + q_base_struct + q_base_lag

                                                    q_pred_intercept = beta0
                                                    q_pred_resid = beta1 * resid_fixed
                                                    q_pred_struct = beta2 * curr_base
                                                    q_pred_lag = beta4 * lag1_new if stage2_n_features == 4 else 0.0
                                                    q_pred = q_pred_intercept + q_pred_resid + q_pred_struct + q_pred_lag

                                                    price_base = base_price * (1 - prev_base / 100.0)
                                                    price_pred = base_price * (1 - curr_base / 100.0)
                                                    rev_base = q_base * price_base
                                                    rev_pred = q_pred * price_pred
                                                    spend = base_price * (step_up / 100.0) * q_pred
                                                    inc_rev = rev_pred - rev_base

                                                    month_str = pd.to_datetime(h['Period']).strftime('%Y-%m-%d')
                                                    st.markdown(f"**Month: {month_str}**")
                                                    st.markdown(
                                                        f"1. **Store model at observed month discount**  \n"
                                                        f"`Pred Stores (Observed) = alpha + gamma x Observed Discount`  \n"
                                                        f"`= {alpha:.4f} + ({gamma:.4f} x {observed_discount:.2f}) = {pred_store_observed:.2f}`"
                                                    )
                                                    st.markdown(
                                                        f"2. **Fixed residual (same for old/new scenarios)**  \n"
                                                        f"`Fixed Residual = Actual Stores - Pred Stores (Observed)`  \n"
                                                        f"`= {store_actual:.2f} - {pred_store_observed:.2f} = {resid_fixed:.2f}`"
                                                    )
                                                    st.markdown(
                                                        f"3. **Quantity model (old/new base; tactical fixed 0; same residual)**  \n"
                                                        + (
                                                            f"`Old Qty = beta0 + beta1 x Fixed Residual + beta2 x Old Base + beta3 x 0 + beta4 x Lag1Structural(old)`  \n"
                                                            f"`= {q_base_intercept:.2f} + {q_base_resid:.2f} + {q_base_struct:.2f} + 0 + {q_base_lag:.2f} = {q_base:.2f}`  \n"
                                                            f"`New Qty = beta0 + beta1 x Fixed Residual + beta2 x New Base + beta3 x 0 + beta4 x Lag1Structural(new)`  \n"
                                                            f"`= {q_pred_intercept:.2f} + {q_pred_resid:.2f} + {q_pred_struct:.2f} + 0 + {q_pred_lag:.2f} = {q_pred:.2f}`"
                                                            if stage2_n_features == 4 else
                                                            f"`Old Qty = beta0 + beta1 x Fixed Residual + beta2 x Old Base + beta3 x 0`  \n"
                                                            f"`= {q_base_intercept:.2f} + {q_base_resid:.2f} + {q_base_struct:.2f} + 0 = {q_base:.2f}`  \n"
                                                            f"`New Qty = beta0 + beta1 x Fixed Residual + beta2 x New Base + beta3 x 0`  \n"
                                                            f"`= {q_pred_intercept:.2f} + {q_pred_resid:.2f} + {q_pred_struct:.2f} + 0 = {q_pred:.2f}`"
                                                        )
                                                    )
                                                    st.markdown(
                                                        f"4. **Price and revenue**  \n"
                                                        f"`Old Price = Base Price x (1 - Old Base/100) = {price_base:.2f}`  \n"
                                                        f"`New Price = Base Price x (1 - New Base/100) = {price_pred:.2f}`  \n"
                                                        f"`Old Revenue = Old Qty x Old Price = {rev_base:.2f}`  \n"
                                                        f"`New Revenue = New Qty x New Price = {rev_pred:.2f}`"
                                                    )
                                                    st.markdown(
                                                        f"5. **Spend and incremental revenue**  \n"
                                                        f"`Structural Spend = Base Price x (Step Up/100) x New Qty = {spend:.2f}`  \n"
                                                        f"`Incremental Revenue = New Revenue - Old Revenue = {rev_pred:.2f} - {rev_base:.2f} = {inc_rev:.2f}`"
                                                    )
                                                    st.divider()

                                    with st.expander(f"Structural ROI Episode Summary - Slab {slab}", expanded=False):
                                        display_roi = roi_df.copy()
                                        display_roi['Period'] = display_roi['Period'].dt.strftime('%Y-%m-%d')
                                        display_roi['Episode_End_Period'] = display_roi['Episode_End_Period'].dt.strftime('%Y-%m-%d')
                                        display_roi = display_roi.rename(columns={
                                            'Period': 'Episode Start',
                                            'Episode_End_Period': 'Episode End',
                                            'Hold_Months': 'Months in Episode',
                                            'Base_Price': 'Avg Base Price',
                                            'Prev_Structural_Discount_Pct': 'Old Base Discount %',
                                            'Structural_Discount_Pct': 'New Base Discount %',
                                            'Structural_Step_Up_Pct': 'Step Up %',
                                            'Predicted_Quantity': 'New-Base Quantity',
                                            'Baseline_Quantity': 'Old-Base Quantity',
                                            'Predicted_Revenue': 'New-Base Revenue',
                                            'Baseline_Revenue': 'Old-Base Revenue',
                                            'Spend': 'Structural Spend',
                                            'Incremental_Revenue': 'Incremental Revenue',
                                            'ROI_1mo': 'Structural ROI'
                                        })
                                        st.dataframe(
                                            display_roi[[
                                                'Episode Start', 'Episode End', 'Months in Episode',
                                                'Avg Base Price', 'Old Base Discount %', 'New Base Discount %',
                                                'Step Up %', 'New-Base Quantity', 'Old-Base Quantity',
                                                'New-Base Revenue', 'Old-Base Revenue', 'Structural Spend',
                                                'Incremental Revenue', 'Structural ROI'
                                            ]].style.format({
                                                'Months in Episode': '{:,.0f}',
                                                'Avg Base Price': '{:.2f}',
                                                'Old Base Discount %': '{:.2f}%',
                                                'New Base Discount %': '{:.2f}%',
                                                'Step Up %': '{:.2f}%',
                                                'New-Base Quantity': '{:,.0f}',
                                                'Old-Base Quantity': '{:,.0f}',
                                                'New-Base Revenue': '{:,.2f}',
                                                'Old-Base Revenue': '{:,.2f}',
                                                'Structural Spend': '{:,.2f}',
                                                'Incremental Revenue': '{:,.2f}',
                                                'Structural ROI': '{:.2f}x'
                                            }),
                                            use_container_width=True,
                                            height=320
                                        )

                # INDIVIDUAL VIEW - Show each outlet separately

                elif view_mode == "Individual View":
                    # Loop through each selected outlet
                    for outlet_idx, selected_outlet in enumerate(selected_outlets):
                        if outlet_idx > 0:
                            st.markdown("---")
                        
                        # Get outlet RFM details
                        outlet_rfm = rfm[rfm['Outlet_ID'] == selected_outlet].iloc[0]
                        
                        # Get segment name
                        outlet_segment = outlet_rfm['RFM_Segment']
                        
                        # Display outlet summary
                        st.markdown(f"###  Outlet {selected_outlet} - {outlet_segment} ({time_agg})")
                        
                        st.divider()
                        
                        # Get detailed transaction data
                        outlet_transactions = df_view[df_view['Outlet_ID'] == selected_outlet].copy()
                        outlet_transactions = outlet_transactions.sort_values('Date', ascending=False)
                        
                        # Aggregate by time period
                        if time_agg == 'Daily':
                            daily_agg = outlet_transactions.groupby('Date').agg({
                                'Bill_No': 'nunique',
                                'Quantity': 'sum',
                                'Net_Amt': 'sum',
                                'TotalDiscount': 'sum',
                                'SalesValue_atBasicRate': 'sum'
                            }).reset_index()
                            daily_agg.columns = ['Period', 'Orders', 'Quantity', 'Net_Amt', 'TotalDiscount', 'SalesValue_atBasicRate']
                            
                        elif time_agg == 'Weekly':
                            outlet_transactions['Period'] = outlet_transactions['Date'].dt.to_period('W').apply(lambda x: x.start_time)
                            daily_agg = outlet_transactions.groupby('Period').agg({
                                'Bill_No': 'nunique',
                                'Quantity': 'sum',
                                'Net_Amt': 'sum',
                                'TotalDiscount': 'sum',
                                'SalesValue_atBasicRate': 'sum'
                            }).reset_index()
                            daily_agg.columns = ['Period', 'Orders', 'Quantity', 'Net_Amt', 'TotalDiscount', 'SalesValue_atBasicRate']
                            
                        elif time_agg == 'Monthly':
                            outlet_transactions['Period'] = outlet_transactions['Date'].dt.to_period('M').apply(lambda x: x.start_time)
                            daily_agg = outlet_transactions.groupby('Period').agg({
                                'Bill_No': 'nunique',
                                'Quantity': 'sum',
                                'Net_Amt': 'sum',
                                'TotalDiscount': 'sum',
                                'SalesValue_atBasicRate': 'sum'
                            }).reset_index()
                            daily_agg.columns = ['Period', 'Orders', 'Quantity', 'Net_Amt', 'TotalDiscount', 'SalesValue_atBasicRate']
                        
                        # Calculate Discount %
                        daily_agg['Discount_Pct'] = (daily_agg['TotalDiscount'] / daily_agg['SalesValue_atBasicRate'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
                        daily_agg['Discount_Pct_Display'] = round_discount_for_display(daily_agg['Discount_Pct'], step=0.5)
                        daily_agg = daily_agg.sort_values('Period', ascending=False)
                        
                        # Outlet Summary in expander (collapsed by default)
                        with st.expander(f" Outlet {selected_outlet} Summary - RFM & Transaction Metrics (Click to expand)", expanded=False):
                            st.markdown("**RFM Metrics:**")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("State", outlet_rfm['Final_State'])
                            with col2:
                                st.metric("Total Orders", f"{outlet_rfm['orders_count']:.0f}")
                            with col3:
                                st.metric("Avg AOV", f"{outlet_rfm['AOV']:.2f}")
                            with col4:
                                st.metric("Recency (days)", f"{outlet_rfm['Recency_days']:.0f}")
                            with col5:
                                st.metric("First Order", str(outlet_rfm['first_order'].date()))
                            
                            st.divider()
                            
                            st.markdown("**Transaction Summary (Filtered Data):**")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric(f"Total {time_agg} Periods", len(daily_agg))
                            with col2:
                                st.metric("Total Quantity", f"{daily_agg['Quantity'].sum():,.0f}")
                            with col3:
                                st.metric("Total Net Amount", f"{daily_agg['Net_Amt'].sum():,.2f}")
                            with col4:
                                st.metric("Total Discount", f"{daily_agg['TotalDiscount'].sum():,.2f}")
                            with col5:
                                avg_discount_pct = (daily_agg['TotalDiscount'].sum() / daily_agg['SalesValue_atBasicRate'].sum() * 100) if daily_agg['SalesValue_atBasicRate'].sum() > 0 else 0
                                st.metric("Avg Discount %", f"{avg_discount_pct:.2f}%")
                        
                        st.divider()
                        
                        # Timeline visualization (MAIN FOCUS - Always visible)
                        st.subheader(f" Purchase Timeline - Outlet {selected_outlet} ({time_agg})")
                        
                        # Calculate correlations
                        corr_qty_disc_pct = daily_agg['Quantity'].corr(daily_agg['Discount_Pct'])
                        corr_sales_disc_pct = daily_agg['SalesValue_atBasicRate'].corr(daily_agg['Discount_Pct'])
                        
                        # Run OLS Regression: Quantity = 0 + 1*Discount%
                        ols_qty = run_ols_regression(
                            daily_agg['Discount_Pct'], 
                            daily_agg['Quantity'],
                            x_name="Discount %",
                            y_name="Quantity",
                            periods=daily_agg['Period']
                        )
                        
                        # Display Correlation and OLS Results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("** Correlation Analysis:**")
                            st.markdown(f"""
                            - Quantity vs Discount %: **{corr_qty_disc_pct:.3f}**
                            - Sales Value vs Discount %: **{corr_sales_disc_pct:.3f}**
                            """)
                        
                        with col2:
                            if ols_qty:
                                st.markdown("** OLS Regression: Quantity ~ Discount %**")
                                st.markdown(f"""
                                - **Equation:** Quantity = {ols_qty['intercept']:.2f} + {ols_qty['slope']:.2f}  Discount%
                                - **R:** {ols_qty['r_squared']:.4f} | **Adj. R:** {ols_qty['adj_r_squared']:.4f}
                                - **Slope p-value:** {ols_qty['p_slope']:.4f} {' Significant' if ols_qty['p_slope'] < 0.05 else ' Not significant'}
                                - **F-statistic:** {ols_qty['f_stat']:.2f} (p={ols_qty['f_pvalue']:.4f})
                                """)
                            else:
                                st.markdown("** OLS Regression:** Insufficient data")
                        
                        st.divider()
                        
                        # Create subplots (2 rows now)
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=(
                                f'Quantity vs Discount % (Corr: {corr_qty_disc_pct:.3f})',
                                f'Sales Value vs Discount % (Corr: {corr_sales_disc_pct:.3f})'
                            ),
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                            vertical_spacing=0.12,
                            shared_xaxes=True
                        )
                        
                        # Subplot 1: Quantity vs Discount %
                        fig.add_trace(
                            go.Scatter(x=daily_agg['Period'], y=daily_agg['Quantity'],
                                     mode='lines+markers', name='Quantity',
                                     line=dict(color='#F18F01', width=2)),
                            row=1, col=1, secondary_y=False
                        )
                        fig.add_trace(
                            go.Scatter(x=daily_agg['Period'], y=daily_agg['Discount_Pct_Display'],
                                     mode='lines+markers', name='Discount %',
                                     line=dict(color='#E63946', width=2, dash='dash')),
                            row=1, col=1, secondary_y=True
                        )
                        
                        # Subplot 2: Sales Value vs Discount %
                        fig.add_trace(
                            go.Scatter(x=daily_agg['Period'], y=daily_agg['SalesValue_atBasicRate'],
                                     mode='lines+markers', name='Sales Value ()',
                                     line=dict(color='#20C997', width=2), showlegend=True),
                            row=2, col=1, secondary_y=False
                        )
                        fig.add_trace(
                            go.Scatter(x=daily_agg['Period'], y=daily_agg['Discount_Pct_Display'],
                                     mode='lines+markers', name='Discount %',
                                     line=dict(color='#E63946', width=2, dash='dash'), showlegend=False),
                            row=2, col=1, secondary_y=True
                        )
                        
                        # Update layout
                        fig.update_xaxes(title_text=time_agg, row=2, col=1)
                        fig.update_yaxes(title_text="Quantity", row=1, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="Discount %", row=1, col=1, secondary_y=True)
                        fig.update_yaxes(title_text="Sales Value ()", row=2, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="Discount %", row=2, col=1, secondary_y=True)
                        
                        fig.update_layout(height=750, hovermode='x', showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        
                        # OLS Actual vs Predicted Plot (Time Series)
                        if ols_qty:
                            st.subheader(f" OLS Regression: Actual vs Predicted Quantity Over Time - Outlet {selected_outlet}")
                            
                            # Create time series plot
                            fig_ols = go.Figure()
                            
                            # Actual Quantity line
                            fig_ols.add_trace(go.Scatter(
                                x=ols_qty['periods_clean'],
                                y=ols_qty['y_clean'],
                                mode='lines+markers',
                                name='Actual Quantity',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=8)
                            ))
                            
                            # Predicted Quantity line
                            fig_ols.add_trace(go.Scatter(
                                x=ols_qty['periods_clean'],
                                y=ols_qty['y_pred'],
                                mode='lines+markers',
                                name='Predicted Quantity',
                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                marker=dict(size=8)
                            ))
                            
                            fig_ols.update_layout(
                                xaxis_title=time_agg,
                                yaxis_title="Quantity",
                                height=500,
                                hovermode='x unified',
                                showlegend=True,
                                title=f"R = {ols_qty['r_squared']:.4f} | Equation: Quantity = {ols_qty['intercept']:.2f} + {ols_qty['slope']:.2f}  Discount%"
                            )
                            
                            st.plotly_chart(fig_ols, use_container_width=True)
                            
                            st.divider()
                        
                        # Summary table in expander (collapsed by default)
                        with st.expander(f" Outlet {selected_outlet} - {time_agg} Summary (Click to expand)", expanded=False):
                            display_df = daily_agg.copy()
                            display_df['Period'] = display_df['Period'].dt.strftime('%Y-%m-%d')
                            st.dataframe(
                                display_df[['Period', 'Orders', 'Quantity', 'Net_Amt', 'TotalDiscount', 'SalesValue_atBasicRate', 'Discount_Pct']].style.format({
                                    'Orders': '{:.0f}',
                                    'Quantity': '{:,.0f}',
                                    'Net_Amt': '{:,.2f}',
                                    'TotalDiscount': '{:,.2f}',
                                    'SalesValue_atBasicRate': '{:,.2f}',
                                    'Discount_Pct': '{:.2f}%'
                                }),
                                use_container_width=True,
                                height=400
                            )
                        
                        # Detailed line-level transactions in expander (collapsed by default)
                        with st.expander(f" Outlet {selected_outlet} - Detailed Transactions - Line Level (Click to expand)", expanded=False):
                            # Select columns to display
                            detail_cols = [
                                'Date', 'Bill_No', 'Final_State', 'Sku_Code', 
                                'Category', 'Subcategory', 'Brand', 'Sizes',
                                'MRP', 'Quantity', 
                                'SalesValue_atBasicRate', 'TotalDiscount',
                                'Net_Amt', 'Basic_Rate_Per_PC'
                            ]
                            
                            st.dataframe(
                                outlet_transactions[detail_cols],
                                use_container_width=True,
                                height=400
                            )
                        
                        # Download button
                        csv = outlet_transactions.to_csv(index=False)
                        st.download_button(
                            label=f" Download Outlet {selected_outlet} Data",
                            data=csv,
                            file_name=f"outlet_{selected_outlet}_{outlet_segment}.csv",
                            mime="text/csv",
                            key=f"download_{selected_outlet}"
                        )
            else:
                st.info(" Use the sidebar filters to select RFM segment(s) and outlet(s) for detailed analysis")
        
        else:
            st.info(" Click 'Calculate RFM' button in the sidebar to start the analysis")
            
            st.markdown("""
            ###  About RFM Analysis
            
            **RFM** stands for:
            - **R (Recency)**: How recently did the outlet make a purchase->
              - Recent: Last order within threshold days (default: 90 days)
              - Stale: Last order beyond threshold days
            
            - **F (Frequency)**: How often does the outlet purchase->
              - Clustered into High/Low using K-means on orders per day
            
            - **M (Monetary)**: How much does the outlet spend per order->
              - Clustered into High/Low using K-means on Average Order Value (AOV)
            
            **Result**: 8 segments (222) like:
            - Recent-High-High (Best customers)
            - Stale-Low-Low (At-risk customers)
            - And 6 more combinations
            """)
    
    # TAB 2: Data Overview
    with tab2:
        st.header(" Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df_filtered):,}")
        with col2:
            st.metric("Unique Outlets", f"{df_filtered['Outlet_ID'].nunique():,}")
        with col3:
            st.metric("Unique Invoices", f"{df_filtered['Bill_No'].nunique():,}")
        with col4:
            st.metric("Date Range", f"{(df_filtered['Date'].max() - df_filtered['Date'].min()).days} days")
        
        st.divider()
        
        # Robust invoice key for contribution/AOV calculations.
        # Helps avoid undercount if Bill_No repeats across outlets or dates.
        df_overview = df_filtered.copy(deep=False)
        df_overview['Invoice_Key'] = (
            df_overview['Outlet_ID'].astype(str)
            + '|'
            + df_overview['Bill_No'].astype(str)
            + '|'
            + pd.to_datetime(df_overview['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        )

        # Summary by State
        st.subheader(" Summary by State")
        
        state_summary = (
            df_overview.groupby('Final_State', as_index=False).agg(
                Outlets=('Outlet_ID', 'nunique'),
                Invoices=('Invoice_Key', 'nunique'),
                Quantity=('Quantity', 'sum'),
                AOQ=('Quantity', 'mean'),  # line-level average quantity
                Sales_Value=('SalesValue_atBasicRate', 'sum'),
                Total_Discount=('TotalDiscount', 'sum')
            )
            .rename(columns={'Final_State': 'State'})
        )
        state_summary['AOV'] = (
            state_summary['Sales_Value'] / state_summary['Invoices'].replace(0, np.nan)
        ).round(2)
        state_summary['AOQ'] = pd.to_numeric(state_summary['AOQ'], errors='coerce').round(2)
        state_summary['Discount_Pct'] = (state_summary['Total_Discount'] / state_summary['Sales_Value'] * 100).round(2)
        state_summary = state_summary[
            ['State', 'Outlets', 'Invoices', 'Quantity', 'AOQ', 'AOV', 'Sales_Value', 'Total_Discount', 'Discount_Pct']
        ]
        
        st.dataframe(state_summary, use_container_width=True)
        
        st.divider()
        
        # Summary by Slab
        st.subheader(" Summary by Slab")
        
        slab_summary = (
            df_overview.groupby('Slab', as_index=False).agg(
                Outlets=('Outlet_ID', 'nunique'),
                Invoices=('Invoice_Key', 'nunique'),
                Quantity=('Quantity', 'sum'),
                AOQ=('Quantity', 'mean'),  # line-level average quantity
                Sales_Value=('SalesValue_atBasicRate', 'sum'),
                Total_Discount=('TotalDiscount', 'sum')
            )
        )
        slab_summary['AOV'] = (
            slab_summary['Sales_Value'] / slab_summary['Invoices'].replace(0, np.nan)
        ).round(2)
        slab_summary['AOQ'] = pd.to_numeric(slab_summary['AOQ'], errors='coerce').round(2)
        slab_summary['Discount_Pct'] = (slab_summary['Total_Discount'] / slab_summary['Sales_Value'] * 100).round(2)

        # Show slab criteria in brackets based on quantity range observed in current filtered data.
        slab_criteria = (
            df_overview.groupby('Slab')['Quantity']
            .agg(Min_Qty='min', Max_Qty='max')
            .reset_index()
        )
        slab_summary = slab_summary.merge(slab_criteria, on='Slab', how='left')

        def _slab_with_criteria(row):
            slab_name = str(row['Slab']) if pd.notna(row['Slab']) else 'Unknown'
            qmin = pd.to_numeric(row.get('Min_Qty', np.nan), errors='coerce')
            qmax = pd.to_numeric(row.get('Max_Qty', np.nan), errors='coerce')
            if pd.isna(qmin) or pd.isna(qmax):
                return slab_name
            if abs(float(qmax) - float(qmin)) < 1e-9:
                criteria = f"Qty={float(qmin):g}"
            else:
                criteria = f"Qty {float(qmin):g}-{float(qmax):g}"
            return f"{slab_name} ({criteria})"

        slab_summary['Slab'] = slab_summary.apply(_slab_with_criteria, axis=1)
        total_slab_invoices = slab_summary['Invoices'].sum()
        total_slab_sales = slab_summary['Sales_Value'].sum()
        slab_summary['Invoice_Contribution_%'] = (
            slab_summary['Invoices'] / total_slab_invoices * 100.0
        ).round(2) if total_slab_invoices > 0 else np.nan
        slab_summary['Sales_Contribution_%'] = (
            slab_summary['Sales_Value'] / total_slab_sales * 100.0
        ).round(2) if total_slab_sales > 0 else np.nan
        slab_summary = slab_summary[
            [
                'Slab', 'Outlets', 'Invoices', 'Invoice_Contribution_%',
                'Quantity', 'AOQ', 'AOV', 'Sales_Value', 'Sales_Contribution_%',
                'Total_Discount', 'Discount_Pct'
            ]
        ]
        
        st.dataframe(slab_summary, use_container_width=True)
        
        st.divider()
        
        # Data preview
        st.subheader(" Data Preview")
        st.dataframe(df_filtered.head(100), use_container_width=True, height=400)

if __name__ == "__main__":
    main()


