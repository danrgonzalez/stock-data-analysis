import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from pathlib import Path
import openpyxl

# Page configuration
st.set_page_config(
    page_title="Stock Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(file_path):
    """Load the stock data from Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns
        for col in ['EPS', 'Revenue', 'Price', 'DivAmt']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean Report column and create Date
        if 'Report' in df.columns:
            df['Report'] = df['Report'].astype(str)
            
            # Extract year and quarter from Report (format Q1'11, Q2'11, etc.)
            df['Year'] = df['Report'].str.extract(r"'(\d{2})").astype(str)
            df['Year'] = df['Year'].apply(lambda x: '20' + x if x.isdigit() and len(x) == 2 else None)
            
            df['Quarter'] = df['Report'].str.extract(r"Q(\d)").astype(str)
            
            # Create a proper date column
            valid_dates = (~df['Year'].isna()) & (~df['Quarter'].isna())
            if valid_dates.any():
                quarter_to_month = {'1': '01', '2': '04', '3': '07', '4': '10'}
                df.loc[valid_dates, 'Date'] = pd.to_datetime(
                    df.loc[valid_dates, 'Year'] + '-' + 
                    df.loc[valid_dates, 'Quarter'].map(quarter_to_month) + '-01',
                    format='%Y-%m-%d',
                    errors='coerce'
                )
        
        # Remove rows with no financial data
<<<<<<< HEAD
        df = df.dropna(subset=['Ticker'])
=======
        df = df.dropna(subset=['Ticker'])
        
        # Calculate TTM values first (before QoQ changes)
        df = calculate_ttm_values(df)
        
        # Calculate Multiple (P/E Ratio) - Price divided by EPS TTM
        df = calculate_multiple(df)
        
        # Calculate quarter-over-quarter percentage changes
        df = calculate_qoq_changes(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_ttm_values(df):
    """Calculate Trailing Twelve Months (TTM) values for EPS and Revenue"""
    # Sort by ticker and date to ensure proper chronological order
    df = df.sort_values(['Ticker', 'Date'])
    
    # Calculate TTM for EPS and Revenue
    ttm_metrics = ['EPS', 'Revenue']
    
    for metric in ttm_metrics:
        if metric in df.columns:
            # Calculate rolling sum of last 4 quarters (including current quarter)
            df[f'{metric}_TTM'] = df.groupby('Ticker')[metric].rolling(window=4, min_periods=1).sum().reset_index(0, drop=True)
    
    return df

def calculate_multiple(df):
    """Calculate Multiple (P/E Ratio) = Current Price / EPS TTM"""
    if 'Price' in df.columns and 'EPS_TTM' in df.columns:
        # Calculate P/E ratio (Price / EPS TTM)
        df['Multiple'] = df['Price'] / df['EPS_TTM']
        
        # Handle infinite values (when EPS_TTM is zero or very close to zero)
        df['Multiple'] = df['Multiple'].replace([float('inf'), -float('inf')], float('nan'))
        
        # Filter out extremely high multiples (likely data errors) - typically P/E > 1000 is unrealistic
        df.loc[df['Multiple'] > 1000, 'Multiple'] = float('nan')
        df.loc[df['Multiple'] < -1000, 'Multiple'] = float('nan')
    
    return df

def calculate_qoq_changes(df):
    """Calculate quarter-over-quarter percentage changes for each ticker"""
    # Metrics to calculate QoQ changes for (including TTM values and Multiple)
    base_metrics = ['EPS', 'Revenue', 'Price', 'DivAmt']
    ttm_metrics = ['EPS_TTM', 'Revenue_TTM']
    multiple_metrics = ['Multiple']
    all_metrics = base_metrics + ttm_metrics + multiple_metrics
    
    # Sort by ticker and date to ensure proper chronological order
    df = df.sort_values(['Ticker', 'Date'])
    
    # Calculate QoQ changes for each metric
    for metric in all_metrics:
        if metric in df.columns:
            # Calculate percentage change from previous quarter for each ticker
            df[f'{metric}_QoQ_Change'] = df.groupby('Ticker')[metric].pct_change() * 100
            
            # Calculate absolute change as well
            df[f'{metric}_QoQ_Abs_Change'] = df.groupby('Ticker')[metric].diff()
    
    return df

def get_available_metrics(df):
    """Get list of available metrics including QoQ changes, TTM values, and Multiple"""
    base_metrics = [col for col in ['EPS', 'Revenue', 'Price', 'DivAmt'] if col in df.columns]
    ttm_metrics = [col for col in df.columns if col.endswith('_TTM')]
    multiple_metrics = [col for col in df.columns if col == 'Multiple']
    qoq_metrics = [col for col in df.columns if col.endswith('_QoQ_Change')]
    abs_change_metrics = [col for col in df.columns if col.endswith('_QoQ_Abs_Change')]
    
    return {
        'base_metrics': base_metrics,
        'ttm_metrics': ttm_metrics,
        'multiple_metrics': multiple_metrics,
        'qoq_metrics': qoq_metrics,
        'abs_change_metrics': abs_change_metrics,
        'all_metrics': base_metrics + ttm_metrics + multiple_metrics + qoq_metrics + abs_change_metrics
    }

def create_time_series_plot(df, tickers, metrics):
    """Create interactive time series plot for selected tickers and metrics"""
    if not tickers or not metrics:
        return None
    
    # Filter data for selected tickers
    filtered_df = df[df['Ticker'].isin(tickers)].copy()
    
    if filtered_df.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=len(metrics), 
        cols=1,
        subplot_titles=[format_metric_name(metric) for metric in metrics],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(metrics):
        if metric not in filtered_df.columns:
            continue
            
        for j, ticker in enumerate(tickers):
            ticker_data = filtered_df[filtered_df['Ticker'] == ticker].sort_values('Date')
            
            if not ticker_data.empty and not ticker_data[metric].isna().all():
                # Add suffix for different metric types
                suffix = ""
                if "_QoQ_Change" in metric:
                    suffix = " (%)"
                elif "_QoQ_Abs_Change" in metric:
                    suffix = " (Abs)"
                elif "_TTM" in metric:
                    suffix = " (TTM)"
                elif metric == "Multiple":
                    suffix = " (P/E)"
                
                # Format the metric name for hover
                formatted_metric = format_metric_name(metric)
                
                fig.add_trace(
                    go.Scatter(
                        x=ticker_data['Date'],
                        y=ticker_data[metric],
                        name=f"{ticker}{suffix}",
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 0),  # Only show legend for first subplot
                        hovertemplate=(
                            f"<b>{ticker}</b><br>"
                            "Date: %{x}<br>"
                            f"{formatted_metric}: %{{y:.2f}}{suffix}<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=i+1, col=1
                )
    
    fig.update_layout(
        height=300 * len(metrics),
        title_text="Stock Metrics Over Time",
        hovermode='x unified'
    )
    
    # Add zero line for QoQ change charts
    for i, metric in enumerate(metrics):
        if "_QoQ_Change" in metric or "_QoQ_Abs_Change" in metric:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i+1, col=1)
    
    return fig

def format_metric_name(metric):
    """Format metric names for display"""
    if metric == 'Multiple':
        return "Multiple (P/E Ratio)"
    elif metric.endswith('_TTM'):
        base_metric = metric.replace('_TTM', '')
        return f"{base_metric} (TTM)"
    elif metric.endswith('_QoQ_Change'):
        base_metric = metric.replace('_QoQ_Change', '')
        if base_metric.endswith('_TTM'):
            base_metric = base_metric.replace('_TTM', '') + ' TTM'
        elif base_metric == 'Multiple':
            base_metric = 'Multiple (P/E)'
        return f"{base_metric} QoQ Change (%)"
    elif metric.endswith('_QoQ_Abs_Change'):
        base_metric = metric.replace('_QoQ_Abs_Change', '')
        if base_metric.endswith('_TTM'):
            base_metric = base_metric.replace('_TTM', '') + ' TTM'
        elif base_metric == 'Multiple':
            base_metric = 'Multiple (P/E)'
        return f"{base_metric} QoQ Absolute Change"
    else:
        return metric

def create_comparison_chart(df, tickers, metric, chart_type='bar'):
    """Create comparison charts for selected tickers"""
    if not tickers or metric not in df.columns:
        return None
    
    # Calculate average metric by ticker
    comparison_data = df[df['Ticker'].isin(tickers)].groupby('Ticker')[metric].agg(['mean', 'std', 'count']).reset_index()
    comparison_data.columns = ['Ticker', 'Average', 'StdDev', 'Count']
    
    # Format the title and labels
    metric_display = format_metric_name(metric)
    
    if chart_type == 'bar':
        fig = px.bar(
            comparison_data, 
            x='Ticker', 
            y='Average',
            title=f'Average {metric_display} by Ticker',
            error_y='StdDev'
        )
        
        # Add zero line for QoQ metrics
        if "_QoQ_Change" in metric or "_QoQ_Abs_Change" in metric:
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
    else:  # box plot
        filtered_df = df[df['Ticker'].isin(tickers)]
        fig = px.box(
            filtered_df, 
            x='Ticker', 
            y=metric,
            title=f'{metric_display} Distribution by Ticker'
        )
        
        # Add zero line for QoQ metrics
        if "_QoQ_Change" in metric or "_QoQ_Abs_Change" in metric:
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_correlation_heatmap(df, tickers):
    """Create correlation heatmap for selected tickers"""
    if not tickers:
        return None
    
    filtered_df = df[df['Ticker'].isin(tickers)]
    
    # Get all numeric columns including QoQ changes, TTM values, and Multiple
    numeric_cols = ['EPS', 'Revenue', 'Price', 'DivAmt']
    ttm_cols = [col for col in filtered_df.columns if col.endswith('_TTM')]
    multiple_cols = ['Multiple'] if 'Multiple' in filtered_df.columns else []
    qoq_cols = [col for col in filtered_df.columns if col.endswith('_QoQ_Change')]
    all_cols = numeric_cols + ttm_cols + multiple_cols + qoq_cols
    
    available_cols = [col for col in all_cols if col in filtered_df.columns]
    
    if len(available_cols) < 2:
        return None
    
    corr_matrix = filtered_df[available_cols].corr()
    
    # Format column names for display
    display_names = [format_metric_name(col) for col in available_cols]
    corr_matrix.index = display_names
    corr_matrix.columns = display_names
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix (Including TTM, Multiple & QoQ Changes)",
        color_continuous_scale='RdBu'
    )
    
    return fig

def create_qoq_summary_table(df, tickers):
    """Create summary table for QoQ changes"""
    if not tickers:
        return pd.DataFrame()
    
    filtered_df = df[df['Ticker'].isin(tickers)]
    
    # Get QoQ change columns including Multiple
    qoq_cols = [col for col in filtered_df.columns if col.endswith('_QoQ_Change')]
    
    if not qoq_cols:
        return pd.DataFrame()
    
    # Calculate summary statistics for QoQ changes
    summary_stats = filtered_df.groupby('Ticker')[qoq_cols].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(2)
    
    # Flatten column names
    summary_stats.columns = [f"{col[0].replace('_QoQ_Change', '')}_{col[1]}" for col in summary_stats.columns.values]
    
    return summary_stats

def create_ttm_summary_table(df, tickers):
    """Create summary table for TTM values"""
    if not tickers:
        return pd.DataFrame()
    
    filtered_df = df[df['Ticker'].isin(tickers)]
    
    # Get TTM columns
    ttm_cols = [col for col in filtered_df.columns if col.endswith('_TTM')]
    
    if not ttm_cols:
        return pd.DataFrame()
    
    # Calculate summary statistics for TTM values
    summary_stats = filtered_df.groupby('Ticker')[ttm_cols].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(2)
    
    # Flatten column names
    summary_stats.columns = [f"{col[0].replace('_TTM', '')}_{col[1]}" for col in summary_stats.columns.values]
    
    return summary_stats

def create_multiple_summary_table(df, tickers):
    """Create summary table for Multiple (P/E Ratio) values"""
    if not tickers or 'Multiple' not in df.columns:
        return pd.DataFrame()
    
    filtered_df = df[df['Ticker'].isin(tickers)]
    
    # Calculate summary statistics for Multiple values
    summary_stats = filtered_df.groupby('Ticker')['Multiple'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).round(2)
    
    # Rename columns for clarity
    summary_stats.columns = ['Mean_PE', 'Median_PE', 'StdDev_PE', 'Min_PE', 'Max_PE', 'Count']
    
    return summary_stats

def main():
    st.title("📈 Interactive Stock Data Dashboard")
    st.markdown("---")
    
    # Sidebar for file upload and filters
    st.sidebar.header("Data Source")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Excel file", 
        type=['xlsx', 'xls'],
        help="Upload the StockData.xlsx file"
    )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        # Try to load from default location
        default_path = Path('data/StockData.xlsx')
        if default_path.exists():
            df = load_data(default_path)
        else:
            st.warning("Please upload your StockData.xlsx file to begin analysis.")
            st.stop()
    
    if df.empty:
        st.error("No data available. Please check your file.")
        st.stop()
    
    # Display basic info
    st.sidebar.success(f"✅ Data loaded: {len(df)} records, {df['Ticker'].nunique()} unique tickers")
    
    # Get available metrics
    metrics_dict = get_available_metrics(df)
    
    # Filters
    st.sidebar.header("Filters")
    
    # Ticker selection
    all_tickers = sorted(df['Ticker'].unique())
    default_tickers = all_tickers[:5] if len(all_tickers) >= 5 else all_tickers
    
    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=all_tickers,
        default=default_tickers,
        help="Choose one or more stock tickers to analyze"
    )
    
    # Date range filter
    if 'Date' in df.columns and not df['Date'].isna().all():
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter by date range
        if len(date_range) == 2:
            df = df[(df['Date'] >= pd.Timestamp(date_range[0])) & 
                   (df['Date'] <= pd.Timestamp(date_range[1]))]
    
    # Metric selection with categories
    st.sidebar.subheader("Select Metrics")
    
    # Base metrics
    selected_base_metrics = st.sidebar.multiselect(
        "Base Metrics",
        options=metrics_dict['base_metrics'],
        default=metrics_dict['base_metrics'][:2] if len(metrics_dict['base_metrics']) >= 2 else metrics_dict['base_metrics'],
        help="Choose base financial metrics"
    )
    
    # TTM metrics
    selected_ttm_metrics = st.sidebar.multiselect(
        "Trailing Twelve Months (TTM)",
        options=metrics_dict['ttm_metrics'],
        default=[],
        help="Choose TTM metrics (sum of last 4 quarters)"
    )
    
    # Multiple metrics
    selected_multiple_metrics = st.sidebar.multiselect(
        "Multiple (P/E Ratio)",
        options=metrics_dict['multiple_metrics'],
        default=metrics_dict['multiple_metrics'] if metrics_dict['multiple_metrics'] else [],
        help="Multiple = Current Price / EPS TTM (P/E Ratio)"
    )
    
    # QoQ Change metrics
    selected_qoq_metrics = st.sidebar.multiselect(
        "Quarter-over-Quarter % Changes",
        options=metrics_dict['qoq_metrics'],
        default=[],
        help="Choose QoQ percentage change metrics"
    )
    
    # Combine selected metrics
    selected_metrics = selected_base_metrics + selected_ttm_metrics + selected_multiple_metrics + selected_qoq_metrics
    
    # Main content area
    if not selected_tickers:
        st.warning("Please select at least one ticker to display charts.")
        st.stop()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Time Series", 
        "📈 Comparison", 
        "🔗 Correlation", 
        "📋 Data Table",
        "📉 QoQ Analysis",
        "📆 TTM Analysis",
        "🔢 Multiple Analysis"
    ])
    
    with tab1:
        st.header("Time Series Analysis")
        
        if selected_metrics and 'Date' in df.columns:
            fig = create_time_series_plot(df, selected_tickers, selected_metrics)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected combination.")
        else:
            st.warning("Please select metrics and ensure date data is available.")
    
    with tab2:
        st.header("Ticker Comparison")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            comparison_metric = st.selectbox(
                "Select Metric",
                options=metrics_dict['all_metrics'],
                key="comparison_metric"
            )
            
            chart_type = st.radio(
                "Chart Type",
                options=['bar', 'box'],
                format_func=lambda x: 'Bar Chart' if x == 'bar' else 'Box Plot'
            )
        
        with col1:
            if comparison_metric:
                fig = create_comparison_chart(df, selected_tickers, comparison_metric, chart_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for comparison.")
    
    with tab3:
        st.header("Correlation Analysis")
        
        if len(selected_tickers) > 0:
            fig = create_correlation_heatmap(df, selected_tickers)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 metrics for correlation analysis.")
        else:
            st.warning("Please select tickers for correlation analysis.")
    
    with tab4:
        st.header("Raw Data")
        
        # Filter data for selected tickers
        filtered_df = df[df['Ticker'].isin(selected_tickers)].copy()
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        if not filtered_df.empty:
            summary_stats = filtered_df.groupby('Ticker')[metrics_dict['all_metrics']].agg(['mean', 'std', 'min', 'max', 'count']).round(3)
            st.dataframe(summary_stats, use_container_width=True)
        
        # Display raw data
        st.subheader("Raw Data")
        if not filtered_df.empty:
            # Sort by Ticker and Date
            display_df = filtered_df.sort_values(['Ticker', 'Date'] if 'Date' in filtered_df.columns else ['Ticker'])
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"stock_data_{'_'.join(selected_tickers)}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data to display.")
    
    with tab5:
        st.header("Quarter-over-Quarter Analysis")
        
        if not selected_tickers:
            st.warning("Please select tickers to view QoQ analysis.")
        else:
            # QoQ Summary Statistics
            st.subheader("QoQ Change Summary Statistics")
            qoq_summary = create_qoq_summary_table(df, selected_tickers)
            if not qoq_summary.empty:
                st.dataframe(qoq_summary, use_container_width=True)
                
                st.markdown("**Legend**: mean = average change, std = volatility, min/max = range, count = number of quarters")
            else:
                st.warning("No QoQ data available.")
            
            # Recent QoQ Changes
            st.subheader("Most Recent Quarter Changes")
            filtered_df = df[df['Ticker'].isin(selected_tickers)]
            
            if not filtered_df.empty and 'Date' in filtered_df.columns:
                # Get most recent quarter for each ticker
                latest_data = filtered_df.loc[filtered_df.groupby('Ticker')['Date'].idxmax()]
                
                # Select QoQ change columns
                qoq_cols = [col for col in latest_data.columns if col.endswith('_QoQ_Change')]
                display_cols = ['Ticker', 'Report', 'Date'] + qoq_cols
                
                recent_changes = latest_data[display_cols].round(2)
                st.dataframe(recent_changes, use_container_width=True)
            
            # QoQ Volatility Analysis
            st.subheader("QoQ Volatility Analysis")
            if metrics_dict['qoq_metrics']:
                volatility_metric = st.selectbox(
                    "Select QoQ Metric for Volatility Analysis",
                    options=metrics_dict['qoq_metrics'],
                    key="volatility_metric"
                )
                
                if volatility_metric:
                    # Calculate volatility (standard deviation of QoQ changes)
                    volatility_data = filtered_df.groupby('Ticker')[volatility_metric].agg(['std', 'mean']).round(2)
                    volatility_data.columns = ['Volatility (Std Dev)', 'Average Change']
                    volatility_data = volatility_data.sort_values('Volatility (Std Dev)', ascending=False)
                    
                    st.dataframe(volatility_data, use_container_width=True)
                    
                    # Volatility chart
                    volatility_chart_data = pd.DataFrame({
                        'Ticker': volatility_data.index,
                        'Volatility': volatility_data['Volatility (Std Dev)']
                    })
                    
                    fig = px.bar(
                        volatility_chart_data,
                        x='Ticker',
                        y='Volatility',
                        title=f"QoQ Volatility: {format_metric_name(volatility_metric)}",
                        labels={'Volatility': 'Volatility (Standard Deviation)'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("Trailing Twelve Months (TTM) Analysis")
        
        if not selected_tickers:
            st.warning("Please select tickers to view TTM analysis.")
        else:
            # TTM Summary Statistics
            st.subheader("TTM Summary Statistics")
            ttm_summary = create_ttm_summary_table(df, selected_tickers)
            if not ttm_summary.empty:
                st.dataframe(ttm_summary, use_container_width=True)
                
                st.markdown("**Legend**: TTM values are the sum of the last 4 quarters (including current quarter)")
            else:
                st.warning("No TTM data available.")
            
            # Recent TTM Values
            st.subheader("Most Recent TTM Values")
            filtered_df = df[df['Ticker'].isin(selected_tickers)]
            
            if not filtered_df.empty and 'Date' in filtered_df.columns:
                # Get most recent quarter for each ticker
                latest_data = filtered_df.loc[filtered_df.groupby('Ticker')['Date'].idxmax()]
                
                # Select TTM columns
                ttm_cols = [col for col in latest_data.columns if col.endswith('_TTM')]
                display_cols = ['Ticker', 'Report', 'Date'] + ttm_cols
                
                recent_ttm = latest_data[display_cols].round(2)
                st.dataframe(recent_ttm, use_container_width=True)
            
            # TTM Growth Analysis
            st.subheader("TTM Growth Analysis")
            if metrics_dict['ttm_metrics']:
                ttm_metric = st.selectbox(
                    "Select TTM Metric for Growth Analysis",
                    options=metrics_dict['ttm_metrics'],
                    key="ttm_growth_metric"
                )
                
                if ttm_metric:
                    # Calculate TTM growth (YoY comparison)
                    # Get TTM values from 4 quarters ago vs current TTM
                    ttm_growth_data = []
                    
                    for ticker in selected_tickers:
                        ticker_data = filtered_df[filtered_df['Ticker'] == ticker].sort_values('Date')
                        if len(ticker_data) >= 5:  # Need at least 5 quarters to compare YoY TTM
                            current_ttm = ticker_data[ttm_metric].iloc[-1]
                            year_ago_ttm = ticker_data[ttm_metric].iloc[-5]  # 4 quarters ago
                            
                            if pd.notna(current_ttm) and pd.notna(year_ago_ttm) and year_ago_ttm != 0:
                                growth_rate = ((current_ttm - year_ago_ttm) / year_ago_ttm) * 100
                                ttm_growth_data.append({
                                    'Ticker': ticker,
                                    'Current TTM': current_ttm,
                                    'Year Ago TTM': year_ago_ttm,
                                    'YoY Growth (%)': growth_rate
                                })
                    
                    if ttm_growth_data:
                        growth_df = pd.DataFrame(ttm_growth_data)
                        growth_df = growth_df.sort_values('YoY Growth (%)', ascending=False)
                        st.dataframe(growth_df.round(2), use_container_width=True)
                        
                        # Growth chart
                        fig = px.bar(
                            growth_df,
                            x='Ticker',
                            y='YoY Growth (%)',
                            title=f"Year-over-Year TTM Growth: {format_metric_name(ttm_metric)}",
                            color='YoY Growth (%)',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Insufficient data for TTM growth analysis. Need at least 5 quarters of data.")
    
    with tab7:
        st.header("Multiple (P/E Ratio) Analysis")
        
        if not selected_tickers:
            st.warning("Please select tickers to view Multiple analysis.")
        elif 'Multiple' not in df.columns:
            st.warning("Multiple data not available. Ensure Price and EPS TTM data are present.")
        else:
            # Multiple Summary Statistics
            st.subheader("Multiple (P/E Ratio) Summary Statistics")
            multiple_summary = create_multiple_summary_table(df, selected_tickers)
            if not multiple_summary.empty:
                st.dataframe(multiple_summary, use_container_width=True)
                
                st.markdown("**Legend**: Multiple = Current Price / EPS TTM (P/E Ratio). Mean/Median show central tendency, StdDev shows volatility")
            else:
                st.warning("No Multiple data available.")
            
            # Current Multiple Values
            st.subheader("Most Recent Multiple (P/E Ratio) Values")
            filtered_df = df[df['Ticker'].isin(selected_tickers)]
            
            if not filtered_df.empty and 'Date' in filtered_df.columns:
                # Get most recent quarter for each ticker
                latest_data = filtered_df.loc[filtered_df.groupby('Ticker')['Date'].idxmax()]
                
                # Select relevant columns
                display_cols = ['Ticker', 'Report', 'Date', 'Price', 'EPS_TTM', 'Multiple']
                available_cols = [col for col in display_cols if col in latest_data.columns]
                
                recent_multiples = latest_data[available_cols].round(2)
                recent_multiples = recent_multiples.sort_values('Multiple', ascending=True)
                st.dataframe(recent_multiples, use_container_width=True)
                
                # Highlight insights
                if not recent_multiples['Multiple'].isna().all():
                    low_pe = recent_multiples.loc[recent_multiples['Multiple'].idxmin()]
                    high_pe = recent_multiples.loc[recent_multiples['Multiple'].idxmax()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"Lowest P/E: {low_pe['Ticker']}", 
                            value=f"{low_pe['Multiple']:.2f}",
                            help="Potentially undervalued (lower P/E ratios may indicate better value)"
                        )
                    with col2:
                        st.metric(
                            label=f"Highest P/E: {high_pe['Ticker']}", 
                            value=f"{high_pe['Multiple']:.2f}",
                            help="Higher P/E ratios may indicate growth expectations or overvaluation"
                        )
            
            # Multiple Distribution Analysis
            st.subheader("P/E Ratio Distribution")
            if not filtered_df.empty and 'Multiple' in filtered_df.columns:
                # Filter out extreme values for better visualization
                clean_multiples = filtered_df[filtered_df['Multiple'].between(0, 100)]  # Focus on reasonable P/E range
                
                if not clean_multiples.empty:
                    # Box plot
                    fig_box = px.box(
                        clean_multiples,
                        x='Ticker',
                        y='Multiple',
                        title="P/E Ratio Distribution by Ticker (Filtered: 0-100 range)",
                        labels={'Multiple': 'P/E Ratio'}
                    )
                    fig_box.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Histogram
                    fig_hist = px.histogram(
                        clean_multiples,
                        x='Multiple',
                        title="P/E Ratio Distribution (All Selected Tickers)",
                        labels={'Multiple': 'P/E Ratio', 'count': 'Frequency'},
                        nbins=30
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.warning("No valid P/E ratios in the 0-100 range for visualization.")
            
            # P/E Ratio Categories
            st.subheader("P/E Ratio Categories")
            if not filtered_df.empty and 'Multiple' in filtered_df.columns:
                # Categorize P/E ratios
                def categorize_pe(pe_ratio):
                    if pd.isna(pe_ratio):
                        return "N/A"
                    elif pe_ratio < 0:
                        return "Negative (Loss-making)"
                    elif pe_ratio < 15:
                        return "Low (< 15) - Potentially Undervalued"
                    elif pe_ratio < 25:
                        return "Moderate (15-25) - Fair Value"
                    elif pe_ratio < 50:
                        return "High (25-50) - Growth Premium"
                    else:
                        return "Very High (> 50) - High Growth/Overvalued"
                
                # Get latest P/E for each ticker
                latest_pe = filtered_df.loc[filtered_df.groupby('Ticker')['Date'].idxmax()]
                latest_pe['PE_Category'] = latest_pe['Multiple'].apply(categorize_pe)
                
                # Count by category
                pe_categories = latest_pe['PE_Category'].value_counts()
                
                # Display category breakdown
                fig_pie = px.pie(
                    values=pe_categories.values,
                    names=pe_categories.index,
                    title="P/E Ratio Categories Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Show tickers in each category
                st.subheader("Tickers by P/E Category")
                for category in pe_categories.index:
                    tickers_in_category = latest_pe[latest_pe['PE_Category'] == category]['Ticker'].tolist()
                    st.write(f"**{category}**: {', '.join(tickers_in_category)}")
            
            # Multiple vs Other Metrics Correlation
            st.subheader("P/E Ratio Relationships")
            if not filtered_df.empty and 'Multiple' in filtered_df.columns:
                # Scatter plot: P/E vs EPS TTM
                if 'EPS_TTM' in filtered_df.columns:
                    # Filter for reasonable ranges
                    scatter_data = filtered_df[
                        (filtered_df['Multiple'].between(0, 100)) & 
                        (filtered_df['EPS_TTM'] > 0)
                    ]
                    
                    if not scatter_data.empty:
                        fig_scatter = px.scatter(
                            scatter_data,
                            x='EPS_TTM',
                            y='Multiple',
                            color='Ticker',
                            title="P/E Ratio vs EPS TTM",
                            labels={'Multiple': 'P/E Ratio', 'EPS_TTM': 'EPS TTM'},
                            hover_data=['Ticker', 'Report']
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        st.markdown("""
                        **Interpretation**: 
                        - Lower P/E with higher EPS TTM may indicate undervalued opportunities
                        - Higher P/E with lower EPS TTM may indicate overvaluation or high growth expectations
                        """)

if __name__ == "__main__":
    main()
>>>>>>> 183cf39 (adding multiple)
