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
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

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
        subplot_titles=metrics,
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
                fig.add_trace(
                    go.Scatter(
                        x=ticker_data['Date'],
                        y=ticker_data[metric],
                        name=f"{ticker} - {metric}",
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=i+1, col=1
                )
    
    fig.update_layout(
        height=300 * len(metrics),
        title_text="Stock Metrics Over Time",
        hovermode='x unified'
    )
    
    return fig

def create_comparison_chart(df, tickers, metric, chart_type='bar'):
    """Create comparison charts for selected tickers"""
    if not tickers or metric not in df.columns:
        return None
    
    # Calculate average metric by ticker
    comparison_data = df[df['Ticker'].isin(tickers)].groupby('Ticker')[metric].agg(['mean', 'std', 'count']).reset_index()
    comparison_data.columns = ['Ticker', 'Average', 'StdDev', 'Count']
    
    if chart_type == 'bar':
        fig = px.bar(
            comparison_data, 
            x='Ticker', 
            y='Average',
            title=f'Average {metric} by Ticker',
            error_y='StdDev'
        )
    else:  # box plot
        filtered_df = df[df['Ticker'].isin(tickers)]
        fig = px.box(
            filtered_df, 
            x='Ticker', 
            y=metric,
            title=f'{metric} Distribution by Ticker'
        )
    
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_correlation_heatmap(df, tickers):
    """Create correlation heatmap for selected tickers"""
    if not tickers:
        return None
    
    filtered_df = df[df['Ticker'].isin(tickers)]
    numeric_cols = ['EPS', 'Revenue', 'Price', 'DivAmt']
    available_cols = [col for col in numeric_cols if col in filtered_df.columns]
    
    if len(available_cols) < 2:
        return None
    
    corr_matrix = filtered_df[available_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    
    return fig

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
    
    # Metric selection
    available_metrics = [col for col in ['EPS', 'Revenue', 'Price', 'DivAmt'] if col in df.columns]
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics",
        options=available_metrics,
        default=available_metrics[:2] if len(available_metrics) >= 2 else available_metrics,
        help="Choose metrics to display in time series"
    )
    
    # Main content area
    if not selected_tickers:
        st.warning("Please select at least one ticker to display charts.")
        st.stop()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Time Series", "📈 Comparison", "🔗 Correlation", "📋 Data Table"])
    
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
                options=available_metrics,
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
            summary_stats = filtered_df.groupby('Ticker')[available_metrics].agg(['mean', 'std', 'min', 'max', 'count']).round(3)
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

if __name__ == "__main__":
    main()
