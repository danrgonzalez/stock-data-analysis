# Stock Data Analysis

This repository contains code and data for analyzing stock market information, including:
- Earnings Per Share (EPS)
- Revenue
- Stock Price
- Dividend Amounts

## Repository Structure

- `data/StockData.csv`: The dataset containing stock information
- `stock_analysis.py`: Python script with functions for data loading, cleaning, and analysis
- `stock_analysis_demo.ipynb`: Jupyter notebook demonstrating the analysis process
- `cli.py`: Command-line interface for stock analysis
- `example.py`: Example script showing custom analysis using the module
- `requirements.txt`: List of required Python packages

## Dataset Description

The `StockData.csv` file contains the following columns:
- `Ticker`: Stock symbol
- `Report`: Reporting period (e.g., Q1'11, Q2'11)
- `EPS`: Earnings Per Share
- `Revenue`: Company revenue for the period
- `Price`: Stock price
- `DivAmt`: Dividend amount

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/danrgonzalez/stock-data-analysis.git
cd stock-data-analysis
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

The easiest way to analyze the stock data is through the command-line interface:

```bash
python cli.py
```

This will run the analysis with default settings and save the results in the `output` directory.

#### CLI Options

```
usage: cli.py [-h] [--file FILE] [--output OUTPUT] [--top-n TOP_N] [--ticker TICKER] [--metrics METRICS] [--stats-only]

Stock Data Analysis CLI

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Path to the stock data CSV file
  --output OUTPUT, -o OUTPUT
                        Directory to save output files
  --top-n TOP_N, -n TOP_N
                        Number of top stocks to analyze
  --ticker TICKER, -t TICKER
                        Specific ticker to analyze
  --metrics METRICS, -m METRICS
                        Comma-separated list of metrics to analyze
  --stats-only          Only output statistics, no plots
```

Examples:

1. Analyze top 5 stocks by EPS and Revenue:
```bash
python cli.py --top-n 5 --metrics EPS,Revenue
```

2. Analyze a specific ticker (e.g., AAPL):
```bash
python cli.py --ticker AAPL
```

3. Generate only statistics (no plots):
```bash
python cli.py --stats-only
```

### Example Custom Analysis

For a demonstration of how to use the `stock_analysis.py` module programmatically to create custom analyses, run:

```bash
python example.py
```

This script shows three custom analyses:
1. Comparing EPS progression for top performers
2. Price to EPS ratio analysis
3. Revenue vs EPS correlation analysis by ticker

The results will be saved in the `custom_analysis` directory.

### Using the Stock Analysis Module

You can import functions from the module in your own Python scripts:

```python
from stock_analysis import load_data, clean_data, plot_top_stocks

# Load and clean data
df = load_data('data/StockData.csv')
df = clean_data(df)

# Custom analysis
# ...
```

### Using the Jupyter Notebook

For interactive analysis, you can use the Jupyter notebook:

```bash
jupyter notebook stock_analysis_demo.ipynb
```

The notebook contains step-by-step analysis with visualizations:
1. Data loading and exploration
2. Data cleaning
3. Basic statistics
4. Visualization of top stocks
5. Time series analysis
6. Correlation analysis
7. Sector-based analysis (simulated)
8. Analysis of EPS and price relationship
9. Dividend analysis

## Extending the Analysis

Here are some ways you can extend this analysis:

1. Add sector information to analyze performance by industry
2. Implement time series forecasting for stock prices or EPS
3. Create a portfolio optimization algorithm based on the metrics
4. Add risk analysis using volatility measures
5. Incorporate external economic indicators

## License

This project is provided for educational purposes only. The data and code are not intended for production use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
