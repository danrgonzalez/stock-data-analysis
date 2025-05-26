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
        df = df.dropna(subset=['Ticker'])
        
        # Calculate quarter-over-quarter percentage changes
        df = calculate_qoq_changes(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_qoq_changes(df):
    """Calculate quarter-over-quarter percentage changes for each ticker"""
    # Metrics to calculate QoQ changes for
    metrics = ['EPS', 'Revenue', 'Price', 'DivAmt']
    
    # Sort by ticker and date to ensure proper chronological order
    df = df.sort_values(['Ticker', 'Date'])
    
    # Calculate QoQ changes for each metric
    for metric in metrics:
        if metric in df.columns:
            # Calculate percentage change from previous quarter for each ticker
            df[f'{metric}_QoQ_Change'] = df.groupby('Ticker')[metric].pct_change() * 100
            
            # Calculate absolute change as well
            df[f'{metric}_QoQ_Abs_Change'] = df.groupby('Ticker')[metric].diff()
    
    return df

def get_available_metrics(df):
    """Get list of available metrics including QoQ changes"""
    base_metrics = [col for col in ['EPS', 'Revenue', 'Price', 'DivAmt'] if col in df.columns]
    qoq_metrics = [col for col in df.columns if col.endswith('_QoQ_Change')]
    abs_change_metrics = [col for col in df.columns if col.endswith('_QoQ_Abs_Change')]
    
    return {
        'base_metrics': base_metrics,
        'qoq_metrics': qoq_metrics,
        'abs_change_metrics': abs_change_metrics,
        'all_metrics': base_metrics + qoq_metrics + abs_change_metrics
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
                # Add suffix for QoQ metrics
                suffix = ""
                if "_QoQ_Change" in metric:
                    suffix = " (%)"
                elif "_QoQ_Abs_Change" in metric:
                    suffix = " (Abs)"
                
                fig.add_trace(
                    go.Scatter(
                        x=ticker_data['Date'],
                        y=ticker_data[metric],
                        name=f"{ticker}{suffix}",
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 0),  # Only show legend for first subplot
                        hovertemplate=f"<b>{ticker}</b><br>" +
                                    "Date: %{x}<br>" +
                                    f"{format_metric_name(metric)}: %{y:.2f}{suffix}<br>" +
                                    "<extra></extra>"
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
    if metric.endswith('_QoQ_Change'):
        base_metric = metric.replace('_QoQ_Change', '')
        return f"{base_metric} QoQ Change (%)"
    elif metric.endswith('_QoQ_Abs_Change'):
        base_metric = metric.replace('_QoQ_Abs_Change', '')
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
    
    # Get all numeric columns including QoQ changes
    numeric_cols = ['EPS', 'Revenue', 'Price', 'DivAmt']
    qoq_cols = [col for col in filtered_df.columns if col.endswith('_QoQ_Change')]
    all_cols = numeric_cols + qoq_cols
    
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
        title="Correlation Matrix (Including QoQ Changes)",
        color_continuous_scale='RdBu'
    )
    
    return fig

def create_qoq_summary_table(df, tickers):
    """Create summary table for QoQ changes"""
    if not tickers:
        return pd.DataFrame()
    
    filtered_df = df[df['Ticker'].isin(tickers)]
    
    # Get QoQ change columns
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
    
    # QoQ Change metrics
    selected_qoq_metrics = st.sidebar.multiselect(
        "Quarter-over-Quarter % Changes",
        options=metrics_dict['qoq_metrics'],
        default=[],
        help="Choose QoQ percentage change metrics"
    )
    
    # Combine selected metrics
    selected_metrics = selected_base_metrics + selected_qoq_metrics
    
    # Main content area
    if not selected_tickers:
        st.warning("Please select at least one ticker to display charts.")
        st.stop()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Time Series", 
        "📈 Comparison", 
        "🔗 Correlation", 
        "📋 Data Table",
        "📉 QoQ Analysis"
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
                    fig = px.bar(
                        x=volatility_data.index,
                        y=volatility_data['Volatility (Std Dev)'],
                        title=f"QoQ Volatility: {format_metric_name(volatility_metric)}",
                        labels={'x': 'Ticker', 'y': 'Volatility (Standard Deviation)'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
