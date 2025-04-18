#!/usr/bin/env python3
"""
Example script demonstrating how to use the stock_analysis module programmatically.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from stock_analysis import (
    load_data,
    clean_data,
    basic_stats,
    plot_top_stocks,
    analyze_time_trends,
    correlation_analysis
)

def example_custom_analysis():
    """
    Example of custom analysis using the stock_analysis module.
    """
    # File path
    file_path = Path('data/StockData.csv')
    if not file_path.exists():
        print(f"Error: {file_path} does not exist")
        return
    
    # Create output directory
    output_dir = Path('custom_analysis')
    if not output_dir.exists():
        output_dir.mkdir()
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_data(file_path)
    df = clean_data(df)
    
    # Change to output directory
    os.chdir(output_dir)
    
    # --- Example 1: Compare EPS progression for top 3 performers ---
    print("\nExample 1: Comparing EPS progression for top performers")
    
    # Get top 3 performers by average EPS
    top_eps = df.groupby('Ticker')['EPS'].mean().sort_values(ascending=False).head(3)
    top_tickers = top_eps.index.tolist()
    
    # Plot EPS over time for each top ticker
    plt.figure(figsize=(14, 8))
    
    for ticker in top_tickers:
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
        if 'Date' in ticker_data.columns and not ticker_data['Date'].isna().all():
            plt.plot(ticker_data['Date'], ticker_data['EPS'], marker='o', linewidth=2, label=ticker)
    
    plt.title('EPS Progression for Top 3 Performers')
    plt.xlabel('Date')
    plt.ylabel('EPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('top_performers_eps_comparison.png')
    plt.close()
    
    # --- Example 2: Price to EPS ratio analysis ---
    print("\nExample 2: Price to EPS ratio analysis")
    
    # Calculate P/E ratio (Price / EPS)
    df_pe = df.copy()
    df_pe['PE_Ratio'] = df_pe['Price'] / df_pe['EPS']
    
    # Handle infinite values (when EPS is zero)
    df_pe['PE_Ratio'] = df_pe['PE_Ratio'].replace([float('inf'), -float('inf')], float('nan'))
    
    # Calculate average P/E ratio by ticker
    pe_by_ticker = df_pe.groupby('Ticker')['PE_Ratio'].mean().sort_values()
    
    # Get top and bottom 5 P/E ratios (excluding NaNs)
    pe_by_ticker = pe_by_ticker.dropna()
    lowest_pe = pe_by_ticker.head(5)
    highest_pe = pe_by_ticker.tail(5)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Lowest P/E
    plt.subplot(2, 1, 1)
    lowest_pe.plot(kind='bar')
    plt.title('5 Stocks with Lowest Average P/E Ratio')
    plt.ylabel('P/E Ratio')
    plt.xlabel('Ticker')
    plt.grid(axis='y', alpha=0.3)
    
    # Highest P/E
    plt.subplot(2, 1, 2)
    highest_pe.plot(kind='bar')
    plt.title('5 Stocks with Highest Average P/E Ratio')
    plt.ylabel('P/E Ratio')
    plt.xlabel('Ticker')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pe_ratio_analysis.png')
    plt.close()
    
    # --- Example 3: Revenue vs EPS correlation by ticker ---
    print("\nExample 3: Revenue vs EPS correlation by ticker")
    
    # Calculate correlation between Revenue and EPS for each ticker
    correlations = []
    
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker]
        
        # Need at least 3 points with both Revenue and EPS data
        valid_data = ticker_data.dropna(subset=['Revenue', 'EPS'])
        
        if len(valid_data) >= 3:
            corr = valid_data['Revenue'].corr(valid_data['EPS'])
            correlations.append({
                'Ticker': ticker,
                'Correlation': corr,
                'DataPoints': len(valid_data)
            })
    
    # Convert to DataFrame and sort
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    # Plot top and bottom 10 correlations
    plt.figure(figsize=(14, 10))
    
    # Top correlations
    plt.subplot(2, 1, 1)
    top_corr = corr_df.head(10)
    plt.bar(top_corr['Ticker'], top_corr['Correlation'])
    plt.title('Top 10 Tickers by Revenue-EPS Correlation')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.ylim(-1, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Bottom correlations
    plt.subplot(2, 1, 2)
    bottom_corr = corr_df.tail(10)
    plt.bar(bottom_corr['Ticker'], bottom_corr['Correlation'])
    plt.title('Bottom 10 Tickers by Revenue-EPS Correlation')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.ylim(-1, 1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('revenue_eps_correlation.png')
    plt.close()
    
    # Save correlation data to CSV
    corr_df.to_csv('revenue_eps_correlation.csv', index=False)
    
    print(f"\nCustom analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    example_custom_analysis()
