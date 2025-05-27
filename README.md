# Stock Data Analysis

This repository contains code and data for analyzing stock market information, including:
- Earnings Per Share (EPS)
- Revenue
- Stock Price
- Dividend Amounts

## 🆕 Latest Value Alignment Feature

**NEW**: Enhanced dashboard and analysis tools with **Latest Value Alignment** - ensuring all tickers' most recent values are compared fairly regardless of different reporting periods!

### Key Enhancement: Fair Comparison Across Tickers

Previously, if one ticker's latest data was Q1'25 and another's was Q3'24, they would appear at different time points making comparison difficult. Now:

- ✅ **All latest values align to the same time point** for fair comparison
- 💎 **Diamond markers** show aligned latest values in time series charts
- 📊 **New "Latest Values" tab** in dashboard for direct comparison
- 🔗 **Aligned comparison charts** in CLI and analysis tools

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

### Enhanced Dashboard Features

- 📊 **Time Series Analysis**: Interactive plots with latest value alignment
- 🔄 **Latest Values Comparison**: NEW tab showing aligned latest values
- 📈 **Ticker Comparison**: Compare multiple stocks with bar charts and box plots
- 🔗 **Correlation Analysis**: Heatmaps showing relationships between metrics
- 📋 **Data Tables**: View raw data and summary statistics
- 🎛️ **Interactive Filters**: Select tickers, date ranges, and metrics
- 💾 **Export Options**: Download filtered data as CSV
- 📉 **QoQ Analysis**: Quarter-over-Quarter change analysis
- 📆 **TTM Analysis**: Trailing Twelve Months calculations
- 🔢 **Multiple Analysis**: P/E ratio analysis and categorization

The dashboard supports:
- Multiple ticker selection and comparison with **latest value alignment**
- Date range filtering
- Metric selection (EPS, Revenue, Price, DivAmt, TTM values, P/E ratios, QoQ changes)
- Various chart types and visualizations
- **Diamond markers showing latest aligned values**
- Responsive design that works on desktop and mobile

## Repository Structure

- `dashboard.py`: **ENHANCED** Interactive Streamlit dashboard with latest value alignment
- `run_dashboard.py`: Easy dashboard launcher script
- `data/StockData.csv`: The dataset containing stock information
- `stock_analysis.py`: **ENHANCED** Python script with latest value alignment functions
- `stock_analysis_demo.ipynb`: Jupyter notebook demonstrating the analysis process
- `cli.py`: **ENHANCED** Command-line interface with alignment features
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
Then upload your Excel file and start exploring with **latest value alignment**!

### Option 2: Enhanced Command-Line Interface

```bash
python cli.py --aligned-latest --include-ttm --include-qoq
```

#### Enhanced CLI Options

```
usage: cli.py [-h] [--file FILE] [--output OUTPUT] [--top-n TOP_N] [--ticker TICKER] 
              [--metrics METRICS] [--stats-only] [--aligned-latest] [--include-ttm] [--include-qoq]

Stock Data Analysis CLI with Latest Value Alignment

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
  --aligned-latest      Create aligned latest values comparison charts and summary
  --include-ttm         Include TTM (Trailing Twelve Months) calculations
  --include-qoq         Include QoQ (Quarter-over-Quarter) change calculations
```

#### New CLI Examples:

1. **Full analysis with latest value alignment**:
```bash
python cli.py --aligned-latest --include-ttm --include-qoq
```

2. **Analyze top 5 stocks with TTM metrics**:
```bash
python cli.py --top-n 5 --include-ttm --aligned-latest
```

3. **QoQ analysis for specific ticker**:
```bash
python cli.py --ticker AAPL --include-qoq --aligned-latest
```

### Option 3: Custom Analysis

For a demonstration of custom programmatic analysis with latest value alignment:

```bash
python example.py
```

### Option 4: Jupyter Notebook

For interactive analysis:

```bash
jupyter notebook stock_analysis_demo.ipynb
```

## New Features Explained

### Latest Value Alignment

The **Latest Value Alignment** feature solves a key problem in cross-ticker comparison:

**Problem**: Different tickers report at different times
- Ticker A: Latest data from Q1'25
- Ticker B: Latest data from Q3'24
- Ticker C: Latest data from Q4'24

**Solution**: Align all latest values to the same comparison point
- All tickers' latest values appear aligned for fair comparison
- Diamond markers in charts show these aligned latest values
- Dedicated "Latest Values" analysis tab in dashboard
- Summary tables show actual reporting periods alongside aligned comparisons

### TTM (Trailing Twelve Months) Calculations

- **EPS TTM**: Sum of last 4 quarters of EPS
- **Revenue TTM**: Sum of last 4 quarters of Revenue
- **P/E Multiple**: Current Price ÷ EPS TTM (more stable than quarterly P/E)

### QoQ (Quarter-over-Quarter) Analysis

- **Percentage Changes**: QoQ growth rates for all metrics
- **Absolute Changes**: Raw QoQ differences
- **Volatility Analysis**: Standard deviation of QoQ changes
- **Latest QoQ Values**: Most recent quarter's changes

## Dashboard Screenshots

The enhanced dashboard now provides:

1. **Time Series**: Compare metrics over time with **latest value alignment diamonds**
2. **Latest Values**: NEW - Direct comparison of aligned latest values
3. **Comparison**: Bar charts and box plots comparing different stocks
4. **Correlation**: Heatmap showing relationships between financial metrics (including TTM and QoQ)
5. **Data Table**: Raw data view with summary statistics and export options
6. **QoQ Analysis**: Quarter-over-quarter change analysis with volatility metrics
7. **TTM Analysis**: Trailing twelve months analysis with growth calculations
8. **Multiple Analysis**: P/E ratio analysis with categorization and distribution

## Using the Enhanced Stock Analysis Module

You can import functions from the enhanced module in your own Python scripts:

```python
from stock_analysis import (
    load_data, clean_data, plot_top_stocks,
    add_latest_value_alignment, calculate_ttm_values,
    calculate_multiple, calculate_qoq_changes,
    plot_aligned_latest_values, create_latest_values_summary
)

# Load and clean data
df = load_data('data/StockData.csv')
df = clean_data(df)

# Add enhanced features
df = calculate_ttm_values(df)
df = calculate_multiple(df)
df = calculate_qoq_changes(df)
df = add_latest_value_alignment(df)

# Create aligned latest values plot
plot_aligned_latest_values(df, metric='EPS', top_n=10)

# Custom analysis with latest value alignment
latest_summary = create_latest_values_summary(df)
print(latest_summary)
```

## Extending the Analysis

Here are some ways you can extend this enhanced analysis:

1. **Sector Analysis**: Add sector information to analyze performance by industry with alignment
2. **Forecasting**: Implement time series forecasting using TTM values
3. **Portfolio Optimization**: Use aligned latest values for portfolio construction
4. **Risk Analysis**: Leverage QoQ volatility measures for risk assessment
5. **Peer Comparison**: Use latest value alignment for industry peer analysis
6. **Performance Attribution**: Analyze what drives QoQ changes
7. **Valuation Models**: Build models using P/E multiples and TTM metrics

## Dependencies

Updated requirements include:
- pandas>=1.3.0
- matplotlib>=3.4.0  
- seaborn>=0.11.0
- numpy>=1.20.0
- jupyter>=1.0.0
- notebook>=6.4.0
- **streamlit>=1.28.0** (for enhanced dashboard)
- **plotly>=5.15.0** (for interactive plots with alignment features)
- **openpyxl>=3.0.0** (for Excel file reading)

## Key Benefits of Latest Value Alignment

1. **Fair Comparisons**: No more comparing Q1'25 data against Q3'24 data
2. **Better Decision Making**: Make investment decisions based on truly comparable metrics
3. **Clear Visualizations**: Diamond markers clearly show which values are being compared
4. **Comprehensive Analysis**: TTM values provide more stable metrics than quarterly snapshots
5. **Enhanced Insights**: QoQ analysis reveals momentum and volatility patterns
6. **Professional Reporting**: Generate reports with properly aligned comparison tables

## License

This project is provided for educational purposes only. The data and code are not intended for production use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.