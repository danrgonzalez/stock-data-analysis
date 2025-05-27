import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from pathlib import Path
import openpyxl

# Page configuration
st.set_page_config(
    page_title="Stock Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(file_path):
    """Load the stock data from Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns
        for col in ['EPS', 'Revenue', 'Price', 'DivAmt']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean Report column and create Date
        if 'Report' in df.columns:
            df['Report'] = df['Report'].astype(str)
            
            # Extract year and quarter from Report (format Q1'11, Q2'11, etc.)
            df['Year'] = df['Report'].str.extract(r"'(\d{2})").astype(str)
            df['Year'] = df['Year'].apply(lambda x: '20' + x if x.isdigit() and len(x) == 2 else None)
            
            df['Quarter'] = df['Report'].str.extract(r"Q(\d)").astype(str)
            
            # Create a proper date column
            valid_dates = (~df['Year'].isna()) & (~df['Quarter'].isna())
            if valid_dates.any():
                quarter_to_month = {'1': '01', '2': '04', '3': '07', '4': '10'}
                df.loc[valid_dates, 'Date'] = pd.to_datetime(
                    df.loc[valid_dates, 'Year'] + '-' + 
                    df.loc[valid_dates, 'Quarter'].map(quarter_to_month) + '-01',
                    format='%Y-%m-%d',
                    errors='coerce'
                )
        
        # Remove rows with no financial data
        df = df.dropna(subset=['Ticker'])