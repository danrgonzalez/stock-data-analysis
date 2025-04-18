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
pip install pandas matplotlib seaborn numpy jupyter
```

## Usage

### Running the Analysis Script

You can run the analysis script directly:

```bash
python stock_analysis.py
```

This will generate several analysis outputs and save plots in the current directory.

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
