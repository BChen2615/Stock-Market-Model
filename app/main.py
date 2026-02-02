import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)

from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
BASE_DIR = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_universal.pkl')
DB_FULL_PATH = os.path.join(BASE_DIR, 'data', 'twstock.db')

# Page Config
st.set_page_config(page_title="AI Stock Predictor", layout="wide")

# --- CACHED FUNCTIONS (To speed up the app) ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def get_stock_list():
    import sqlite3
    conn = sqlite3.connect(DB_FULL_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Stock_ID FROM tw_stock_prices")
    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return [s for s in stocks if len(s) == 4]

def predict_single_stock(stock_id, model):
    try:
        # Load data with raw prices for display
        df = prepare_training_data_v2(stock_id, target_days=1, path=DB_FULL_PATH, keep_raw_prices=True)
        if df is None or df.empty:
            return None, None
            
        # Prepare Features for Prediction
        drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target', 
                     'Open', 'High', 'Low', 'Close', 'Volume']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Predict Probability
        probs = model.predict_proba(X)[:, 1]
        
        # Add prob to df for visualization
        df['Prob_Up'] = probs
        
        return df, probs[-1] # Return full df and latest probability
    except Exception as e:
        st.error(f"Error predicting {stock_id}: {e}")
        return None, None

# --- UI LAYOUT ---

st.title("📈 AI Stock Prediction Dashboard")

model = load_model()
if model is None:
    st.error("Model not found! Please run Train_Universal_Model.py first.")
    st.stop()

# Sidebar
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Select Mode", ["Market Scan", "Single Stock Analysis"])

# --- MODE 1: MARKET SCAN ---
if mode == "Market Scan":
    st.header("🔍 Market Radar: Top Picks for Tomorrow")
    
    threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.60, 0.01)
    
    if st.button("Start Scan (This may take time)"):
        all_stocks = get_stock_list()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        # Scan loop (Limit to first 100 for demo speed, remove limit for full scan)
        # In production, you might want to pre-calculate this and save to CSV
        scan_list = all_stocks # [:100] # Uncomment slice for testing
        
        for i, stock in enumerate(scan_list):
            if i % 10 == 0:
                progress_bar.progress(i / len(scan_list))
                status_text.text(f"Scanning {stock}...")
                
            df, last_prob = predict_single_stock(stock, model)
            
            if last_prob is not None:
                last_row = df.iloc[-1]
                results.append({
                    "Stock": stock,
                    "Date": last_row.name.date(),
                    "Close": last_row['Close'],
                    "Prob_Up": last_prob
                })
        
        progress_bar.empty()
        status_text.text("Scan Complete!")
        
        # Display Results
        if results:
            res_df = pd.DataFrame(results)
            # Filter
            top_picks = res_df[res_df['Prob_Up'] > threshold].sort_values(by='Prob_Up', ascending=False)
            
            st.success(f"Found {len(top_picks)} stocks with Prob > {threshold}")
            st.dataframe(top_picks.style.format({"Prob_Up": "{:.2%}", "Close": "{:.2f}"}))
            
            # Download
            csv = top_picks.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "top_picks.csv", "text/csv")
        else:
            st.warning("No data found.")

# --- MODE 2: SINGLE STOCK ANALYSIS ---
elif mode == "Single Stock Analysis":
    st.header("📊 Single Stock Deep Dive")
    
    stock_input = st.sidebar.text_input("Enter Stock ID (e.g., 2330)", "2330")
    
    if st.button("Analyze"):
        with st.spinner(f"Analyzing {stock_input}..."):
            df, last_prob = predict_single_stock(stock_input, model)
            
            if df is not None:
                # 1. Summary Metrics
                col1, col2, col3 = st.columns(3)
                last_date = df.index[-1].date()
                last_close = df['Close'].iloc[-1]
                
                col1.metric("Date", str(last_date))
                col2.metric("Close Price", f"{last_close:.2f}")
                
                # Color code probability
                prob_color = "normal"
                if last_prob > 0.6: prob_color = "inverse" # Highlight
                col3.metric("AI Prediction (Up Prob)", f"{last_prob:.2%}", delta=None)
                
                if last_prob > 0.6:
                    st.success("🚀 Strong Buy Signal!")
                elif last_prob < 0.4:
                    st.error("🔻 Bearish Signal")
                else:
                    st.warning("⚖️ Neutral / Hold")
                
                # 2. Interactive Chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, row_heights=[0.7, 0.3],
                                    subplot_titles=("Price History", "AI Confidence"))
                
                # Candlestick (Simplified to Close line for speed, or use OHLC if available)
                # We have OHLC in df if keep_raw_prices=True
                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
                
                # Probability
                fig.add_trace(go.Scatter(x=df.index, y=df['Prob_Up'], name='Prob Up', 
                                         line=dict(color='purple', width=1), fill='tozeroy'), row=2, col=1)
                
                # Thresholds
                fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)
                fig.add_hline(y=0.4, line_dash="dot", line_color="red", row=2, col=1)
                
                fig.update_layout(height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. Recent Data Table
                st.subheader("Recent Data")
                st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Prob_Up']].tail(10).style.format("{:.2f}"))
                
            else:
                st.error("Stock not found or insufficient data.")
