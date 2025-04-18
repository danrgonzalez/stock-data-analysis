#!/usr/bin/env python3
"""
Command-line interface for stock data analysis.
"""

import argparse
import os
from pathlib import Path
import pandas as pd

from stock_analysis import (
    load_data, 
    clean_data, 
    basic_stats, 
    plot_top_stocks, 
    analyze_time_trends, 
    correlation_analysis
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Data Analysis CLI')
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default='data/StockData.csv',
        help='Path to the stock data CSV file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=10,
        help='Number of top stocks to analyze'
    )
    
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        help='Specific ticker to analyze'
    )
    
    parser.add_argument(
        '--metrics', '-m',
        type=str,
        default='EPS,Revenue,Price,DivAmt',
        help='Comma-separated list of metrics to analyze'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only output statistics, no plots'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the CLI."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created output directory: {output_dir}")
    
    # Check if the input file exists
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: Input file {file_path} does not exist")
        return 1
    
    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(',')]
    
    # Load and clean the data
    print(f"Loading data from {file_path}...")
    df = load_data(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    df = clean_data(df)
    print(f"After cleaning: {len(df)} rows")
    
    # Change working directory to output
    os.chdir(output_dir)
    
    # Basic statistics
    print("Calculating basic statistics...")
    stats = basic_stats(df)
    
    # Save statistics to CSV
    stats_file = 'basic_stats.csv'
    stats.to_csv(stats_file)
    print(f"Saved basic statistics to {output_dir / stats_file}")
    
    # Print top N stocks by each metric
    for metric in metrics:
        if metric in df.columns:
            top_metric = df.groupby('Ticker')[metric].mean().sort_values(ascending=False).head(args.top_n)
            print(f"\nTop {args.top_n} stocks by average {metric}:")
            print(top_metric)
            
            # Save to CSV
            top_metric_file = f'top_{args.top_n}_{metric.lower()}_stocks.csv'
            top_metric.to_csv(top_metric_file)
            print(f"Saved to {output_dir / top_metric_file}")
    
    # Skip plots if stats_only is True
    if args.stats_only:
        print("Skipping plots (--stats-only flag was used)")
        return 0
    
    # Create plots
    print("\nCreating plots...")
    for metric in metrics:
        if metric in df.columns:
            print(f"Plotting top stocks by {metric}...")
            plot_top_stocks(df, metric=metric, top_n=args.top_n)
            print(f"Saved plot to {output_dir / f'top_{args.top_n}_{metric.lower()}_stocks.png'}")
    
    # Time trends for a specific ticker
    if args.ticker:
        if args.ticker in df['Ticker'].unique():
            print(f"Analyzing time trends for {args.ticker}...")
            analyze_time_trends(df, args.ticker, metrics=metrics)
            print(f"Saved time trends plot to {output_dir / f'{args.ticker}_time_trends.png'}")
        else:
            print(f"Warning: Ticker {args.ticker} not found in the data")
    else:
        # Use the top performer by EPS
        top_eps_ticker = df.groupby('Ticker')['EPS'].mean().sort_values(ascending=False).index[0]
        print(f"Analyzing time trends for top performer {top_eps_ticker}...")
        analyze_time_trends(df, top_eps_ticker, metrics=metrics)
        print(f"Saved time trends plot to {output_dir / f'{top_eps_ticker}_time_trends.png'}")
    
    # Correlation analysis
    print("Performing correlation analysis...")
    corr_matrix = correlation_analysis(df, metrics=metrics)
    
    # Save correlation matrix to CSV
    corr_file = 'correlation_matrix.csv'
    corr_matrix.to_csv(corr_file)
    print(f"Saved correlation matrix to {output_dir / corr_file}")
    print(f"Saved correlation heatmap to {output_dir / 'correlation_matrix.png'}")
    
    print("\nAnalysis complete! Check the output directory for results.")
    return 0

if __name__ == "__main__":
    exit(main())
