import streamlit as st
import requests
import json
from operator import itemgetter
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Any
import numpy as np

# Constants
API_ENDPOINTS = {
    "Top Performers": st.secrets["topPerformers "],
    "Top Picks": st.secrets["topPicks "],
    "ETFs": st.secrets["ETFs"]
}

YAHOO_METRIC_CATEGORIES = {
    "Valuation Metrics": {
        "marketCap": ("Market Cap", lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else "N/A"),
        "enterpriseValue": ("Enterprise Value", lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else "N/A"),
        "forwardPE": ("Forward P/E", lambda x: f"{x:.2f}x" if isinstance(x, (int, float)) else "N/A"),
        "priceToBook": ("Price/Book", lambda x: f"{x:.2f}x" if isinstance(x, (int, float)) else "N/A"),
        "enterpriseToRevenue": ("EV/Revenue", lambda x: f"{x:.2f}x" if isinstance(x, (int, float)) else "N/A"),
        "enterpriseToEbitda": ("EV/EBITDA", lambda x: f"{x:.2f}x" if isinstance(x, (int, float)) else "N/A"),
    },
    "Financial Metrics": {
        "totalRevenue": ("Revenue (TTM)", lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else "N/A"),
        "grossProfits": ("Gross Profit", lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else "N/A"),
        "operatingMargins": ("Operating Margin", lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else "N/A"),
        "profitMargins": ("Profit Margin", lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else "N/A"),
        "returnOnEquity": ("ROE", lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else "N/A"),
        "returnOnAssets": ("ROA", lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else "N/A"),
    },
    "Trading Information": {
        "fiftyTwoWeekHigh": ("52W High", lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else "N/A"),
        "fiftyTwoWeekLow": ("52W Low", lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else "N/A"),
        "volume": ("Volume", lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else "N/A"),
        "averageVolume": ("Avg Volume", lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else "N/A"),
        "beta": ("Beta", lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"),
    },
    "Growth & Dividends": {
        "revenueGrowth": ("Revenue Growth", lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else "N/A"),
        "earningsGrowth": ("Earnings Growth", lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else "N/A"),
        "dividendRate": ("Dividend Rate", lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else "N/A"),
        "dividendYield": ("Dividend Yield", lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else "N/A"),
        "payoutRatio": ("Payout Ratio", lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else "N/A"),
    },
    "Balance Sheet": {
        "totalCash": ("Total Cash", lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else "N/A"),
        "totalDebt": ("Total Debt", lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else "N/A"),
        "debtToEquity": ("Debt/Equity", lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"),
        "currentRatio": ("Current Ratio", lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"),
        "quickRatio": ("Quick Ratio", lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"),
    }
}

@st.cache_data(ttl=300)
def fetch_data(url: str) -> List[Dict]:
    """Fetch data from API with error handling."""
    try:
        response = requests.post(
            url,
            json={'page': 1, 'count': 100, 'search': '', 'user_id': ''}
        )
        response.raise_for_status()
        return response.json().get("stocks", [])
    except Exception as e:
        st.error(f"Failed to retrieve data from API: {str(e)}")
        return []

def filter_stocks(stocks: List[Dict]) -> List[Dict]:
    """Filter and sort stocks based on criteria."""
    return sorted(
        [stock for stock in stocks if stock["reco"] == "BUY" and stock["perf"] > 3],
        key=itemgetter("days_since_reco")
    )

@st.cache_data(ttl=300)
def get_yahoo_finance_data(symbol: str) -> Tuple[Dict[str, Dict[str, str]], pd.DataFrame]:
    """Fetch and process Yahoo Finance data with comprehensive metrics."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Process metrics by category
        metrics = {}
        for category, category_metrics in YAHOO_METRIC_CATEGORIES.items():
            metrics[category] = {
                display_name: formatter(info.get(metric_key))
                for metric_key, (display_name, formatter) in category_metrics.items()
            }
        
        # Get historical data
        hist = stock.history(period='6mo')
        
        return metrics, hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def create_price_volume_chart(hist_data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a combined price and volume chart."""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            name='Price',
            line=dict(color='blue')
        )
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Price and Volume History",
        yaxis=dict(title='Price', side='left'),
        yaxis2=dict(title='Volume', side='right', overlaying='y'),
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(page_title="Dashboard", layout="wide")
    st.title("📈 Stock Dashboard")
    
    # Dataset selection
    dataset = st.selectbox("Select Data Source", list(API_ENDPOINTS.keys()))
    stocks = fetch_data(API_ENDPOINTS[dataset])
    filtered_stocks = filter_stocks(stocks)
    
    if not filtered_stocks:
        st.warning("No stocks found matching the criteria.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(filtered_stocks)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Stock Data", "📈 Detailed Analysis"])
    
    with tab1:
        st.subheader(f"Filtered Stocks ({dataset})")
        st.dataframe(
            df[["stock_name", "stk", "reco", "price", "change", "days_since_reco", "sm_ann"]],
            use_container_width=True
        )
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Data",
            csv,
            "stock_data.csv",
            "text/csv",
            key='download-csv'
        )
    
    with tab2:
        selected_stock = st.selectbox(
            "Select Stock for Detailed Analysis",
            df['stock_name'].tolist()
        )
        
        symbol = df[df['stock_name'] == selected_stock]['stk'].iloc[0]
        metrics, hist_data = get_yahoo_finance_data(symbol)
        
        if metrics and hist_data is not None:
            # Display metrics by category
            for category, category_metrics in metrics.items():
                st.subheader(category)
                cols = st.columns(len(category_metrics))
                for col, (metric_name, value) in zip(cols, category_metrics.items()):
                    col.metric(metric_name, value)
            
            # Display combined price and volume chart
            st.plotly_chart(
                create_price_volume_chart(hist_data, symbol),
                use_container_width=True
            )
    
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()