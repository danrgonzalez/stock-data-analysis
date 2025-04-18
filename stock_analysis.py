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
        # Extract year and quarter from Report (format Q1'11, Q2'11, etc.)
        df['Year'] = df['Report'].str.extract(r"'(\d{2})").astype(str).apply(lambda x: '20' + x if x.isdigit() else None)
        df['Quarter'] = df['Report'].str.extract(r"Q(\d)").astype(str)
        
        # Create a proper date column (set to first day of the quarter)
        df['Date'] = pd.to_datetime(df['Year'] + '-' + df['Quarter'] + '-1', format='%Y-%m-%d', errors='coerce')
    
    # Remove rows with all missing financial data
    df = df.dropna(subset=['EPS', 'Revenue', 'Price', 'DivAmt'], how='all')
    
    return df

def basic_stats(df):
    """
    Calculate basic statistics for the dataset
    """
    # Group by ticker and calculate stats
    ticker_stats = df.groupby('Ticker').agg({
        'EPS': ['mean', 'std', 'min', 'max', 'count'],
        'Revenue': ['mean', 'std', 'min', 'max', 'count'],
        'Price': ['mean', 'std', 'min', 'max', 'count'],
        'DivAmt': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    return ticker_stats

def plot_top_stocks(df, metric='EPS', top_n=10):
    """
    Plot the top N stocks by a given metric
    """
    if metric not in df.columns:
        print(f"Metric {metric} not found in the data")
        return
    
    # Calculate average of the metric by ticker
    avg_metric = df.groupby('Ticker')[metric].mean().sort_values(ascending=False)
    
    # Get top N tickers
    top_tickers = avg_metric.head(top_n)
    
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
    
    # Plot the metrics over time
    plt.figure(figsize=(14, 7))
    
    for i, metric in enumerate(metrics):
        if metric in ticker_df.columns:
            plt.subplot(len(metrics), 1, i+1)
            plt.plot(ticker_df['Date'], ticker_df[metric])
            plt.title(f'{ticker} - {metric} Over Time')
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_time_trends.png')
    plt.close()

def correlation_analysis(df, metrics=['EPS', 'Revenue', 'Price', 'DivAmt']):
    """
    Analyze correlations between different metrics
    """
    # Calculate correlation matrix
    valid_metrics = [m for m in metrics if m in df.columns]
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
    top_ticker = top_eps.index[0]
    print(f"Analyzing time trends for {top_ticker}...")
    analyze_time_trends(df, top_ticker)
    
    # Correlation analysis
    print("Performing correlation analysis...")
    correlation_analysis(df)
    
    print("Analysis complete! Check the output directory for plots.")

if __name__ == "__main__":
    main()
