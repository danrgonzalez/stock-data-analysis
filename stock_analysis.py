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
    # Convert Price and DivAmt to numeric values
    for col in ['Price', 'DivAmt']:
        if col in df.columns:
            # Remove any non-numeric characters except decimal points
            df[col] = pd.to_numeric(df[col].replace(r'[^\d.]', '', regex=True), errors='coerce')
    
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

def analyze_time_trends(df, ticker, metrics=['EPS', 'Price']):
    """
    Analyze time trends for a specific ticker
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
                plt.plot(valid_data['Date'], valid_data[metric])
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

def main():
    """
    Main function to run the analysis
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
    
    # Basic statistics
    print("Calculating basic statistics...")
    stats = basic_stats(df)
    print("Top 5 stocks by average EPS:")
    top_eps = df.groupby('Ticker')['EPS'].mean().sort_values(ascending=False).head()
    print(top_eps)
    
    # Plotting
    print("Creating plots...")
    plot_top_stocks(df, metric='EPS', top_n=10)
    plot_top_stocks(df, metric='Revenue', top_n=10)
    
    # Time trends for a top performer
    if len(top_eps) > 0:
        top_ticker = top_eps.index[0]
        print(f"Analyzing time trends for {top_ticker}...")
        analyze_time_trends(df, top_ticker)
    else:
        print("No valid EPS data for time trend analysis")
    
    # Correlation analysis
    print("Performing correlation analysis...")
    correlation_analysis(df)
    
    print("Analysis complete! Check the output directory for plots.")

if __name__ == "__main__":
    main()
