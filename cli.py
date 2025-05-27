#!/usr/bin/env python3
"""
Command-line interface for stock data analysis with latest value alignment.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from stock_analysis import (
    load_data, 
    clean_data, 
    basic_stats, 
    plot_top_stocks, 
    analyze_time_trends, 
    correlation_analysis,
    add_latest_value_alignment,
    calculate_ttm_values,
    calculate_multiple,
    calculate_qoq_changes,
    plot_aligned_latest_values,
    create_latest_values_summary
)

def plot_aligned_latest_values_cli(df, metrics, output_dir):
    """Create a bar chart showing latest values for all tickers aligned"""
    if 'Is_Latest_Value' not in df.columns:
        print("Latest value alignment not available")
        return
    
    # Get only the latest values for each ticker
    latest_data = df[df['Is_Latest_Value'] == True].copy()
    
    if latest_data.empty:
        print("No latest values data available")
        return
    
    # Create plots for each metric
    for metric in metrics:
        if metric not in latest_data.columns:
            continue
        
        # Get the data for this metric, removing NaN values
        metric_data = latest_data[['Ticker', metric, 'Report', 'Date']].dropna(subset=[metric])
        
        if metric_data.empty:
            print(f"No data available for {metric}")
            continue
        
        # Sort by metric value for better visualization
        metric_data = metric_data.sort_values(metric, ascending=False)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(metric_data)), metric_data[metric])
        
        # Customize the plot
        plt.title(f'Latest {metric} Values (Aligned Comparison)', fontsize=16, fontweight='bold')
        plt.xlabel('Tickers', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        
        # Set x-axis labels
        plt.xticks(range(len(metric_data)), metric_data['Ticker'], rotation=45)
        
        # Add value labels on bars
        for i, (bar, value, ticker, report) in enumerate(zip(bars, metric_data[metric], metric_data['Ticker'], metric_data['Report'])):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            # Add report period as text below ticker
            plt.text(i, -abs(height)*0.05, f'({report})', ha='center', va='top', fontsize=6, alpha=0.7)
        
        # Add zero line if needed (for QoQ metrics)
        if "_QoQ_Change" in metric or "_QoQ_Abs_Change" in metric:
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        filename = f'latest_{metric.lower()}_aligned_comparison.png'
        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
        print(f"Saved aligned latest values plot: {filename}")
        plt.close()

def create_latest_values_summary_cli(df, output_dir):
    """Create a summary table of latest values for all tickers"""
    if 'Is_Latest_Value' not in df.columns:
        print("Latest value alignment not available")
        return
    
    # Get only the latest values for each ticker
    latest_data = df[df['Is_Latest_Value'] == True].copy()
    
    if latest_data.empty:
        print("No latest values data available")
        return
    
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
    filename = 'latest_values_aligned_summary.csv'
    summary_df.to_csv(Path(output_dir) / filename, index=False)
    print(f"Saved latest values summary: {filename}")
    
    # Print summary to console
    print("\nLatest Values Summary (Aligned):")
    print("=" * 50)
    print(f"Total tickers: {len(summary_df)}")
    
    if 'Date' in summary_df.columns:
        date_range = summary_df['Date'].agg(['min', 'max'])
        print(f"Date range of latest values: {date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}")
    
    if 'Report' in summary_df.columns:
        unique_reports = summary_df['Report'].unique()
        print(f"Latest report periods: {', '.join(sorted(unique_reports))}")
    
    print("\nTop 5 tickers by latest EPS (if available):")
    if 'EPS' in summary_df.columns:
        top_eps = summary_df.nlargest(5, 'EPS')[['Ticker', 'EPS', 'Report']]
        print(top_eps.to_string(index=False))
    
    return summary_df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Data Analysis CLI with Latest Value Alignment')
    
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
    
    parser.add_argument(
        '--aligned-latest',
        action='store_true',
        help='Create aligned latest values comparison charts and summary'
    )
    
    parser.add_argument(
        '--include-ttm',
        action='store_true',
        help='Include TTM (Trailing Twelve Months) calculations'
    )
    
    parser.add_argument(
        '--include-qoq',
        action='store_true',
        help='Include QoQ (Quarter-over-Quarter) change calculations'
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
    
    # Add TTM calculations if requested
    if args.include_ttm:
        print("Calculating TTM values...")
        df = calculate_ttm_values(df)
        df = calculate_multiple(df)
    
    # Add QoQ calculations if requested
    if args.include_qoq:
        print("Calculating QoQ changes...")
        df = calculate_qoq_changes(df)
    
    # Add latest value alignment
    print("Adding latest value alignment...")
    df = add_latest_value_alignment(df)
    
    # Change working directory to output
    os.chdir(output_dir)
    
    # Basic statistics
    print("Calculating basic statistics...")
    stats = basic_stats(df)
    
    # Save statistics to CSV
    if not stats.empty:
        stats_file = 'basic_stats.csv'
        stats.to_csv(stats_file)
        print(f"Saved basic statistics to {output_dir / stats_file}")
    
    # Create aligned latest values summary
    if args.aligned_latest:
        print("Creating aligned latest values analysis...")
        create_latest_values_summary_cli(df, '.')
    
    # Print top N stocks by each metric
    for metric in metrics:
        if metric in df.columns:
            # Ensure metric is numeric
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
            # Calculate and get top stocks
            top_metric = df.groupby('Ticker')[metric].mean().sort_values(ascending=False).head(args.top_n)
            
            # Only proceed if we have data
            if len(top_metric) > 0:
                print(f"\nTop {args.top_n} stocks by average {metric}:")
                print(top_metric)
                
                # Save to CSV
                top_metric_file = f'top_{args.top_n}_{metric.lower()}_stocks.csv'
                top_metric.to_csv(top_metric_file)
                print(f"Saved to {output_dir / top_metric_file}")
            else:
                print(f"\nNo valid data for metric: {metric}")
    
    # Skip plots if stats_only is True
    if args.stats_only:
        print("Skipping plots (--stats-only flag was used)")
        return 0
    
    # Create standard plots
    print("\nCreating plots...")
    for metric in metrics:
        if metric in df.columns:
            # Ensure metric is numeric
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
            # Check if we have valid data
            if df[metric].notna().any():
                print(f"Plotting top stocks by {metric}...")
                plot_top_stocks(df, metric=metric, top_n=args.top_n)
                print(f"Saved plot to {output_dir / f'top_{args.top_n}_{metric.lower()}_stocks.png'}")
            else:
                print(f"No valid data for metric: {metric}, skipping plot")
    
    # Create aligned latest values plots if requested
    if args.aligned_latest and not args.stats_only:
        print("Creating aligned latest values plots...")
        
        # Include TTM and QoQ metrics if they were calculated
        plot_metrics = metrics.copy()
        if args.include_ttm:
            ttm_metrics = [col for col in df.columns if col.endswith('_TTM')]
            plot_metrics.extend(ttm_metrics)
            if 'Multiple' in df.columns:
                plot_metrics.append('Multiple')
        
        if args.include_qoq:
            qoq_metrics = [col for col in df.columns if col.endswith('_QoQ_Change')]
            plot_metrics.extend(qoq_metrics)
        
        plot_aligned_latest_values_cli(df, plot_metrics, '.')
    
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
        # Ensure EPS is numeric
        if 'EPS' in df.columns:
            df['EPS'] = pd.to_numeric(df['EPS'], errors='coerce')
            
            # Calculate top performers by EPS
            top_eps = df.groupby('Ticker')['EPS'].mean().sort_values(ascending=False)
            
            # Check if we have any valid data
            if len(top_eps) > 0 and pd.notna(top_eps.iloc[0]):
                top_eps_ticker = top_eps.index[0]
                print(f"Analyzing time trends for top performer {top_eps_ticker}...")
                analyze_time_trends(df, top_eps_ticker, metrics=metrics)
                print(f"Saved time trends plot to {output_dir / f'{top_eps_ticker}_time_trends.png'}")
            else:
                print("No valid EPS data for time trend analysis")
    
    # Correlation analysis
    print("Performing correlation analysis...")
    # Ensure metrics are numeric
    for metric in metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    # Include additional metrics for correlation if calculated
    correlation_metrics = metrics.copy()
    if args.include_ttm:
        ttm_metrics = [col for col in df.columns if col.endswith('_TTM')]
        correlation_metrics.extend(ttm_metrics)
        if 'Multiple' in df.columns:
            correlation_metrics.append('Multiple')
    
    if args.include_qoq:
        qoq_metrics = [col for col in df.columns if col.endswith('_QoQ_Change')]
        correlation_metrics.extend(qoq_metrics)
    
    corr_matrix = correlation_analysis(df, metrics=correlation_metrics)
    
    # Only save correlation matrix if we have data
    if not corr_matrix.empty:
        corr_file = 'correlation_matrix.csv'
        corr_matrix.to_csv(corr_file)
        print(f"Saved correlation matrix to {output_dir / corr_file}")
        if not args.stats_only:
            print(f"Saved correlation heatmap to {output_dir / 'correlation_matrix.png'}")
    else:
        print("No valid data for correlation analysis")
    
    # Print summary of latest values alignment
    if args.aligned_latest:
        print("\n" + "="*60)
        print("LATEST VALUES ALIGNMENT SUMMARY")
        print("="*60)
        
        latest_data = df[df['Is_Latest_Value'] == True]
        if not latest_data.empty:
            print(f"Total tickers with latest values: {len(latest_data)}")
            
            if 'Date' in latest_data.columns:
                date_range = latest_data['Date'].agg(['min', 'max'])
                print(f"Date range of latest values: {date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}")
            
            if 'Report' in latest_data.columns:
                unique_reports = latest_data['Report'].value_counts()
                print("\nDistribution of latest report periods:")
                for report, count in unique_reports.items():
                    print(f"  {report}: {count} tickers")
            
            print("\nThis alignment allows fair comparison of all tickers' most recent performance,")
            print("regardless of when their latest data was reported.")
    
    print("\nAnalysis complete! Check the output directory for results.")
    return 0

if __name__ == "__main__":
    exit(main())