# Stock Data Analysis

Up to TTM

This repository contains code and data for analyzing stock market information, including:
- Earnings Per Share (EPS)
- Revenue
- Stock Price
- Dividend Amounts

## 🆕 Interactive Dashboard

**NEW**: An interactive web dashboard built with Streamlit for exploring your stock data!

### Quick Start with Dashboard

1. **Clone and Setup**:
```bash
git clone https://github.com/danrgonzalez/stock-data-analysis.git
cd stock-data-analysis
```

2. **Run Dashboard** (Easy method):
```bash
python run_dashboard.py
```
This will automatically install requirements and launch the dashboard.

3. **Manual Dashboard Launch**:
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

4. **Upload Your Data**: When the dashboard opens, upload your `StockData.xlsx` file using the sidebar file uploader.

### Dashboard Features

- 📊 **Time Series Analysis**: Interactive plots showing stock metrics over time
- 📈 **Ticker Comparison**: Compare multiple stocks with bar charts and box plots
- 🔗 **Correlation Analysis**: Heatmaps showing relationships between metrics
- 📋 **Data Tables**: View raw data and summary statistics
- 🎛️ **Interactive Filters**: Select tickers, date ranges, and metrics
- 💾 **Export Options**: Download filtered data as CSV

The dashboard supports:
- Multiple ticker selection and comparison
- Date range filtering
- Metric selection (EPS, Revenue, Price, DivAmt)
- Various chart types and visualizations
- Responsive design that works on desktop and mobile

## Repository Structure

- `dashboard.py`: **NEW** Interactive Streamlit dashboard
- `run_dashboard.py`: **NEW** Easy dashboard launcher script
- `data/StockData.csv`: The dataset containing stock information
- `stock_analysis.py`: Python script with functions for data loading, cleaning, and analysis
- `stock_analysis_demo.ipynb`: Jupyter notebook demonstrating the analysis process
- `cli.py`: Command-line interface for stock analysis
- `example.py`: Example script showing custom analysis using the module
- `requirements.txt`: List of required Python packages (updated with dashboard dependencies)

## Dataset Description

The stock data contains the following columns:
- `Ticker`: Stock symbol (149 unique tickers)
- `Report`: Reporting period (e.g., Q1'11, Q2'11)
- `EPS`: Earnings Per Share
- `Revenue`: Company revenue for the period
- `Price`: Stock price
- `DivAmt`: Dividend amount

Data spans from 2011 onwards with quarterly financial data.

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

## Usage Options

### Option 1: Interactive Dashboard (Recommended)
```bash
python run_dashboard.py
```
Then upload your Excel file and start exploring!

### Option 2: Command-Line Interface

```bash
python cli.py
```

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

### Option 3: Custom Analysis

For a demonstration of custom programmatic analysis:

```bash
python example.py
```

### Option 4: Jupyter Notebook

For interactive analysis:

```bash
jupyter notebook stock_analysis_demo.ipynb
```

## Dashboard Screenshots

The dashboard provides multiple views:

1. **Time Series**: Compare metrics over time for selected tickers
2. **Comparison**: Bar charts and box plots comparing different stocks
3. **Correlation**: Heatmap showing relationships between financial metrics
4. **Data Table**: Raw data view with summary statistics and export options

## Using the Stock Analysis Module

You can import functions from the module in your own Python scripts:

```python
from stock_analysis import load_data, clean_data, plot_top_stocks

# Load and clean data
df = load_data('data/StockData.csv')
df = clean_data(df)

# Custom analysis
# ...
```

## Extending the Analysis

Here are some ways you can extend this analysis:

1. Add sector information to analyze performance by industry
2. Implement time series forecasting for stock prices or EPS
3. Create a portfolio optimization algorithm based on the metrics
4. Add risk analysis using volatility measures
5. Incorporate external economic indicators
6. Add more dashboard features like technical indicators

## Dependencies

Updated requirements include:
- pandas>=1.3.0
- matplotlib>=3.4.0  
- seaborn>=0.11.0
- numpy>=1.20.0
- jupyter>=1.0.0
- notebook>=6.4.0
- **streamlit>=1.28.0** (for dashboard)
- **plotly>=5.15.0** (for interactive plots)
- **openpyxl>=3.0.0** (for Excel file reading)

## License

This project is provided for educational purposes only. The data and code are not intended for production use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
