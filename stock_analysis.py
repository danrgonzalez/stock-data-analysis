import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_data(file_path):
    """
    Load the stock data CSV file into a pandas DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Convert numeric columns
    for col in ['EPS', 'Revenue']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Price and DivAmt to numeric values
    for col in ['Price', 'DivAmt']:
        if col in df.columns:
            # First convert to string to handle any non-string values
            df[col] = df[col].astype(str)
            # Then remove any non-numeric characters except decimal points
            df[col] = df[col].replace(r'[^\d.]', '', regex=True)
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def clean_data(df):
    """
    Clean the DataFrame by handling missing values and data types
    """
    # Drop the unnamed column if it exists
    if '' in df.columns:
        df = df.drop(columns=[''])
    
    # Convert Report column to datetime if possible
    if 'Report' in df.columns:
        # Convert Report to string type first to handle any non-string values
        df['Report'] = df['Report'].astype(str)
        
        # Extract year and quarter from Report (format Q1'11, Q2'11, etc.)
        df['Year'] = df['Report'].str.extract(r"'(\d{2})").astype(str)
        df['Year'] = df['Year'].apply(lambda x: '20' + x if x.isdigit() and len(x) == 2 else None)
        
        df['Quarter'] = df['Report'].str.extract(r"Q(\d)").astype(str)
        
        # Filter out invalid quarters
        df.loc[~df['Quarter'].isin(['1', '2', '3', '4']), 'Quarter'] = None
        
        # Create a proper date column (set to first day of the quarter)
        valid_dates = (~df['Year'].isna()) & (~df['Quarter'].isna())
        if valid_dates.any():
            df.loc[valid_dates, 'Date'] = pd.to_datetime(
                df.loc[valid_dates, 'Year'] + '-' + df.loc[valid_dates, 'Quarter'] + '-1', 
                format='%Y-%m-%d', 
                errors='coerce'
            )
    
    # Remove rows with all missing financial data
    df = df.dropna(subset=['EPS', 'Revenue', 'Price', 'DivAmt'], how='all')
    
    return df

def add_latest_value_alignment(df):
    """
    Add columns for latest value alignment across all tickers
    This function ensures that the latest available data point for each ticker
    can be compared at the same time point, regardless of actual reporting dates.
    """
    if 'Date' not in df.columns:
        return df
    
    # For each ticker, find their latest (most recent) data point
    latest_data_by_ticker = df.loc[df.groupby('Ticker')['Date'].idxmax()].copy()
    
    # Find the overall latest date across all tickers to use as alignment point
    global_latest_date = df['Date'].max()
    
    # Create aligned date for latest values
    latest_data_by_ticker['Aligned_Latest_Date'] = global_latest_date
    
    # Create a flag to identify latest values
    latest_data_by_ticker['Is_Latest_Value'] = True
    
    # Add these flags back to the main dataframe
    df['Is_Latest_Value'] = False
    df['Aligned_Latest_Date'] = pd.NaT
    
    # Mark the latest values for each ticker
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker]
        if not ticker_data.empty and 'Date' in ticker_data.columns:
            ticker_latest_idx = ticker_data['Date'].idxmax()
            if pd.notna(ticker_latest_idx):
                df.loc[ticker_latest_idx, 'Is_Latest_Value'] = True
                df.loc[ticker_latest_idx, 'Aligned_Latest_Date'] = global_latest_date
    
    return df

def calculate_ttm_values(df):
    """
    Calculate Trailing Twelve Months (TTM) values for EPS and Revenue
    """
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
    """
    Calculate Multiple (P/E Ratio) = Current Price / EPS TTM
    """
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
    """
    Calculate quarter-over-quarter percentage changes for each ticker
    """
    # Metrics to calculate QoQ changes for (including TTM values and Multiple)
    base_metrics = ['EPS', 'Revenue', 'Price', 'DivAmt']
    ttm_metrics = [col for col in df.columns if col.endswith('_TTM')]
    multiple_metrics = ['Multiple'] if 'Multiple' in df.columns else []
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

def basic_stats(df):
    """
    Calculate basic statistics for the dataset
    """
    # Ensure we have numeric columns for statistics
    numeric_cols = []
    for col in ['EPS', 'Revenue', 'Price', 'DivAmt']:
        if col in df.columns:
            # Convert to numeric if not already
            df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_cols.append(col)
    
    # Add TTM and Multiple columns if they exist
    ttm_cols = [col for col in df.columns if col.endswith('_TTM')]
    multiple_cols = ['Multiple'] if 'Multiple' in df.columns else []
    numeric_cols.extend(ttm_cols + multiple_cols)
    
    # If no numeric columns, return empty DataFrame
    if not numeric_cols:
        return pd.DataFrame()
    
    # Group by ticker and calculate stats
    ticker_stats = df.groupby('Ticker')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
    
    return ticker_stats

def plot_top_stocks(df, metric='EPS', top_n=10):
    """
    Plot the top N stocks by a given metric
    """
    if metric not in df.columns:
        print(f"Metric {metric} not found in the data")
        return
    
    # Ensure the metric is numeric
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    # Calculate average of the metric by ticker
    avg_metric = df.groupby('Ticker')[metric].mean().sort_values(ascending=False)
    
    # Get top N tickers
    top_tickers = avg_metric.head(top_n)
    
    # Skip if no data
    if len(top_tickers) == 0:
        print(f"No valid data for metric: {metric}")
        return
    
    # Plot
    plt.figure(figsize=(12, 6))
    top_tickers.plot(kind='bar')
    plt.title(f'Top {top_n} Stocks by Average {metric}')
    plt.ylabel(metric)
    plt.xlabel('Ticker')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'top_{top_n}_{metric.lower()}_stocks.png')
    plt.close()

def plot_aligned_latest_values(df, metric='EPS', top_n=10):
    """
    Plot the latest values for top N stocks, aligned for comparison
    """
    if 'Is_Latest_Value' not in df.columns:
        print("Latest value alignment not available. Run add_latest_value_alignment() first.")
        return
    
    if metric not in df.columns:
        print(f"Metric {metric} not found in the data")
        return
    
    # Get only the latest values for each ticker
    latest_data = df[df['Is_Latest_Value'] == True].copy()
    
    if latest_data.empty:
        print("No latest values data available")
        return
    
    # Ensure the metric is numeric and remove NaN values
    latest_data[metric] = pd.to_numeric(latest_data[metric], errors='coerce')
    metric_data = latest_data[['Ticker', metric, 'Report', 'Date']].dropna(subset=[metric])
    
    if metric_data.empty:
        print(f"No valid data for {metric}")
        return
    
    # Sort by metric value and take top N
    metric_data = metric_data.sort_values(metric, ascending=False).head(top_n)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(metric_data)), metric_data[metric])
    
    # Customize the plot
    plt.title(f'Top {top_n} Stocks by Latest {metric} Values (Aligned Comparison)', fontsize=16, fontweight='bold')
    plt.xlabel('Tickers', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    
    # Set x-axis labels
    plt.xticks(range(len(metric_data)), metric_data['Ticker'], rotation=45)
    
    # Add value labels on bars and report periods
    for i, (bar, value, ticker, report) in enumerate(zip(bars, metric_data[metric], metric_data['Ticker'], metric_data['Report'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        # Add report period as text below ticker
        plt.text(i, -abs(height)*0.05, f'({report})', ha='center', va='top', fontsize=6, alpha=0.7)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'top_{top_n}_latest_{metric.lower()}_aligned.png')
    print(f"Saved aligned latest values plot: top_{top_n}_latest_{metric.lower()}_aligned.png")
    plt.close()

def analyze_time_trends(df, ticker, metrics=['EPS', 'Price'], include_latest_alignment=True):
    """
    Analyze time trends for a specific ticker, optionally showing latest value alignment
    """
    if 'Date' not in df.columns or ticker not in df['Ticker'].unique():
        print(f"Unable to analyze time trends for {ticker}")
        return
    
    # Filter for the specific ticker
    ticker_df = df[df['Ticker'] == ticker].sort_values('Date')
    
    # Ensure we have valid date data
    if ticker_df['Date'].isna().all():
        print(f"No valid date data for {ticker}")
        return
    
    # Convert metrics to numeric
    for metric in metrics:
        if metric in ticker_df.columns:
            ticker_df[metric] = pd.to_numeric(ticker_df[metric], errors='coerce')
    
    # Plot the metrics over time
    plt.figure(figsize=(14, 7))
    
    for i, metric in enumerate(metrics):
        if metric in ticker_df.columns:
            valid_data = ticker_df.dropna(subset=['Date', metric])
            if len(valid_data) > 0:
                plt.subplot(len(metrics), 1, i+1)
                plt.plot(valid_data['Date'], valid_data[metric], marker='o', linewidth=2)
                
                # Highlight the latest value if alignment is available
                if include_latest_alignment and 'Is_Latest_Value' in ticker_df.columns:
                    latest_point = ticker_df[ticker_df['Is_Latest_Value'] == True]
                    if not latest_point.empty and not latest_point[metric].isna().all():
                        plt.scatter(latest_point['Date'], latest_point[metric], 
                                  color='red', s=100, marker='D', zorder=5,
                                  label=f"Latest ({latest_point['Report'].iloc[0]})")
                        plt.legend()
                
                plt.title(f'{ticker} - {metric} Over Time')
                plt.ylabel(metric)
                plt.grid(True, alpha=0.3)
            else:
                print(f"No valid data for {ticker} - {metric}")
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_time_trends.png')
    plt.close()

def correlation_analysis(df, metrics=['EPS', 'Revenue', 'Price', 'DivAmt']):
    """
    Analyze correlations between different metrics
    """
    # Ensure metrics are numeric
    for metric in metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    # Calculate correlation matrix
    valid_metrics = [m for m in metrics if m in df.columns]
    
    # Skip if no valid metrics
    if not valid_metrics:
        print("No valid metrics for correlation analysis")
        return pd.DataFrame()
    
    # Calculate correlation
    corr_matrix = df[valid_metrics].corr().round(2)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Stock Metrics')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    return corr_matrix

def create_latest_values_summary(df, filename='latest_values_summary.csv'):
    """
    Create a comprehensive summary of latest values for all tickers
    """
    if 'Is_Latest_Value' not in df.columns:
        print("Latest value alignment not available. Run add_latest_value_alignment() first.")
        return pd.DataFrame()
    
    # Get only the latest values for each ticker
    latest_data = df[df['Is_Latest_Value'] == True].copy()
    
    if latest_data.empty:
        print("No latest values data available")
        return pd.DataFrame()
    
    # Select relevant columns
    summary_cols = ['Ticker', 'Report', 'Date']
    metric_cols = ['EPS', 'Revenue', 'Price', 'DivAmt']
    ttm_cols = [col for col in latest_data.columns if col.endswith('_TTM')]
    multiple_cols = ['Multiple'] if 'Multiple' in latest_data.columns else []
    qoq_cols = [col for col in latest_data.columns if col.endswith('_QoQ_Change')]
    
    all_cols = summary_cols + metric_cols + ttm_cols + multiple_cols + qoq_cols
    available_cols = [col for col in all_cols if col in latest_data.columns]
    
    summary_df = latest_data[available_cols].round(2)
    summary_df = summary_df.sort_values('Ticker')
    
    # Save to CSV
    summary_df.to_csv(filename, index=False)
    print(f"Saved latest values summary to {filename}")
    
    return summary_df

def main():
    """
    Main function to run the analysis with latest value alignment
    """
    # Ensure the data directory exists
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir()
        print("Created 'data' directory")
    
    # Check if the file exists
    file_path = data_dir / 'StockData.csv'
    if not file_path.exists():
        print(f"Error: {file_path} does not exist")
        return
    
    # Load and clean the data
    print("Loading data...")
    df = load_data(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    df = clean_data(df)
    print(f"After cleaning: {len(df)} rows")
    
    # Calculate TTM values
    print("Calculating TTM values...")
    df = calculate_ttm_values(df)
    df = calculate_multiple(df)
    
    # Calculate QoQ changes
    print("Calculating QoQ changes...")
    df = calculate_qoq_changes(df)
    
    # Add latest value alignment
    print("Adding latest value alignment...")
    df = add_latest_value_alignment(df)
    
    # Basic statistics
    print("Calculating basic statistics...")
    stats = basic_stats(df)
    print("Top 5 stocks by average EPS:")
    if 'EPS' in df.columns:
        top_eps = df.groupby('Ticker')['EPS'].mean().sort_values(ascending=False).head()
        print(top_eps)
    
    # Create latest values summary
    print("Creating latest values summary...")
    create_latest_values_summary(df)
    
    # Plotting
    print("Creating plots...")
    plot_top_stocks(df, metric='EPS', top_n=10)
    plot_top_stocks(df, metric='Revenue', top_n=10)
    
    # Plot aligned latest values
    print("Creating aligned latest value plots...")
    plot_aligned_latest_values(df, metric='EPS', top_n=10)
    plot_aligned_latest_values(df, metric='Revenue', top_n=10)
    
    if 'EPS_TTM' in df.columns:
        plot_aligned_latest_values(df, metric='EPS_TTM', top_n=10)
    
    if 'Multiple' in df.columns:
        plot_aligned_latest_values(df, metric='Multiple', top_n=10)
    
    # Time trends for a top performer
    if 'EPS' in df.columns:
        top_eps = df.groupby('Ticker')['EPS'].mean().sort_values(ascending=False)
        if len(top_eps) > 0:
            top_ticker = top_eps.index[0]
            print(f"Analyzing time trends for {top_ticker}...")
            analyze_time_trends(df, top_ticker, include_latest_alignment=True)
        else:
            print("No valid EPS data for time trend analysis")
    
    # Correlation analysis
    print("Performing correlation analysis...")
    # Include TTM and Multiple columns in correlation
    correlation_metrics = ['EPS', 'Revenue', 'Price', 'DivAmt']
    if 'EPS_TTM' in df.columns:
        correlation_metrics.append('EPS_TTM')
    if 'Revenue_TTM' in df.columns:
        correlation_metrics.append('Revenue_TTM')
    if 'Multiple' in df.columns:
        correlation_metrics.append('Multiple')
    
    correlation_analysis(df, metrics=correlation_metrics)
    
    print("\nAnalysis complete! Check the output directory for plots and summary files.")
    print("\nKey features added:")
    print("- Latest value alignment ensures fair comparison across all tickers")
    print("- TTM (Trailing Twelve Months) calculations for more stable metrics")
    print("- P/E Multiple calculations (Price/EPS TTM)")
    print("- Quarter-over-Quarter change analysis")
    print("- Aligned latest values plots show most recent performance comparison")

if __name__ == "__main__":
    main()