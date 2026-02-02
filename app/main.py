import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import plotly.graph_objects as go
from datetime import datetime

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)

from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
st.set_page_config(page_title="AI Stock Radar", layout="wide", page_icon="📈")

# --- SESSION STATE INIT ---
if 'page' not in st.session_state:
    st.session_state.page = 'radar'
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = '2330'

# --- DATABASE FUNCTIONS ---
def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_latest_prediction_date():
    conn = get_db_connection()
    try:
        query = "SELECT MAX(Date) FROM daily_predictions"
        date = conn.execute(query).fetchone()[0]
        return date
    except:
        return None
    finally:
        conn.close()

def get_top_picks(date, sort_by='Prob_1d', limit=200):
    conn = get_db_connection()
    query = f"""
        SELECT * FROM daily_predictions 
        WHERE Date = ? 
        ORDER BY {sort_by} DESC 
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(date, limit))
    conn.close()
    return df

def get_stock_history(stock_id, days=200): # Increased days for MA calculation
    try:
        df = prepare_training_data_v2(stock_id, target_days=1, path=DB_PATH, keep_raw_prices=True)
        if df is not None:
            return df.tail(days)
    except:
        pass
    return None

# --- NAVIGATION FUNCTIONS ---
def go_to_analysis(stock_id):
    st.session_state.selected_stock = stock_id
    st.session_state.page = 'analysis'

def go_to_radar():
    st.session_state.page = 'radar'

# --- PAGE 1: MARKET RADAR ---
def render_radar():
    st.title("📈 AI Stock Radar")
    
    latest_date = get_latest_prediction_date()
    if not latest_date:
        st.error("No data found. Run Batch_Predict.py.")
        return

    # Status Bar
    today = datetime.now().date()
    pred_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
    if (today - pred_date).days > 3:
        st.warning(f"⚠️ Data Outdated: {latest_date}")
    else:
        st.success(f"✅ Data Updated: {latest_date}")

    # --- Search Bar ---
    col_search, _ = st.columns([1, 2])
    with col_search:
        search_input = st.text_input("🔍 Search Stock ID (Press Enter)", placeholder="e.g. 2330")
        if search_input:
            go_to_analysis(search_input)
            st.rerun()

    st.divider()

    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        sort_col = st.selectbox("Sort Strategy", 
                               ['Prob_1d', 'Prob_2d', 'Prob_3d', 'Prob_7d', 'Prob_14d'], 
                               index=0)
    with col2:
        threshold = st.slider("Confidence >", 0.5, 0.9, 0.6, 0.01)
    
    # Data
    df = get_top_picks(latest_date, sort_by=sort_col, limit=500)
    df_filtered = df[df[sort_col] > threshold].reset_index(drop=True)
    
    st.markdown(f"### Top Opportunities ({len(df_filtered)})")
    
    # Header
    cols = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
    headers = ["Stock", "Price", "1-Day %", "3-Day %", "7-Day %", "14-Day %", "Action"]
    for col, h in zip(cols, headers):
        col.markdown(f"**{h}**")
    
    st.divider()
    
    # Rows (Limit to top 50 to prevent lag)
    for index, row in df_filtered.head(50).iterrows():
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
        
        c1.write(f"**{row['Stock_ID']}**")
        c2.write(f"{row['Close_Price']:.2f}")
        
        # Color coding helper
        def color_prob(p):
            if p > 0.6:
                return f":green[{p:.1%}]"
            elif p < 0.4:
                return f":red[{p:.1%}]"
            else:
                return f"{p:.1%}"

        c3.markdown(color_prob(row['Prob_1d']))
        c4.markdown(color_prob(row['Prob_3d']))
        c5.markdown(color_prob(row['Prob_7d']))
        c6.markdown(color_prob(row['Prob_14d']))
        
        # The Button
        if c7.button("👉", key=f"btn_{row['Stock_ID']}"):
            go_to_analysis(row['Stock_ID'])
            st.rerun()

# --- PAGE 2: STOCK ANALYSIS ---
def render_analysis():
    stock_id = st.session_state.selected_stock
    
    # Header with Back Button
    c1, c2 = st.columns([1, 10])
    if c1.button("← Back"):
        go_to_radar()
        st.rerun()
        
    c2.title(f"📊 Analysis: {stock_id}")
    
    # Fetch Data
    latest_date = get_latest_prediction_date()
    conn = get_db_connection()
    row = pd.read_sql("SELECT * FROM daily_predictions WHERE Date = ? AND Stock_ID = ?", 
                      conn, params=(latest_date, stock_id))
    conn.close()
    
    if row.empty:
        st.error(f"No prediction data found for {stock_id}.")
        return

    data = row.iloc[0]
    
    # 1. Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("1-Day Prob", f"{data['Prob_1d']:.1%}")
    m2.metric("2-Day Prob", f"{data['Prob_2d']:.1%}")
    m3.metric("3-Day Prob", f"{data['Prob_3d']:.1%}")
    m4.metric("7-Day Prob", f"{data['Prob_7d']:.1%}")
    m5.metric("14-Day Prob", f"{data['Prob_14d']:.1%}")
    
    st.divider()
    
    # 2. Charts Layout
    col_chart1, col_chart2 = st.columns([1, 2]) # Give more space to price chart
    
    with col_chart1:
        st.subheader("AI Confidence")
        horizons = [1, 2, 3, 7, 14]
        probs = [data[f'Prob_{h}d'] for h in horizons]
        
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=horizons, y=probs, mode='lines+markers+text', 
                                   text=[f"{p:.1%}" for p in probs], textposition="top center",
                                   line=dict(color='#636EFA', width=4)))
        fig_h.add_hline(y=0.6, line_dash="dot", line_color="green", annotation_text="Strong Buy")
        fig_h.add_hline(y=0.5, line_dash="dot", line_color="gray")
        fig_h.update_layout(xaxis_title="Days Ahead", yaxis_title="Probability", yaxis_range=[0, 1], height=400)
        st.plotly_chart(fig_h, use_container_width=True)
        
    with col_chart2:
        st.subheader("Price History")
        
        # --- Chart Controls ---
        cc1, cc2 = st.columns(2)
        with cc1:
            chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
        with cc2:
            ma_options = st.multiselect("Indicators", ["SMA 5", "SMA 10", "SMA 20", "SMA 60"], default=["SMA 5", "SMA 20"])
            
        df_hist = get_stock_history(stock_id, days=150) # Get enough data for MA
        
        if df_hist is not None:
            fig_p = go.Figure()
            
            # 1. Base Chart
            if chart_type == "Candlestick":
                fig_p.add_trace(go.Candlestick(x=df_hist.index,
                    open=df_hist['Open'], high=df_hist['High'],
                    low=df_hist['Low'], close=df_hist['Close'], name='OHLC'))
            else:
                fig_p.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], 
                                           mode='lines', name='Close', line=dict(color='blue')))
            
            # 2. Moving Averages
            colors = {"SMA 5": "orange", "SMA 10": "cyan", "SMA 20": "purple", "SMA 60": "gray"}
            
            for ma in ma_options:
                window = int(ma.split()[1])
                # Calculate MA on the fly
                ma_series = df_hist['Close'].rolling(window=window).mean()
                fig_p.add_trace(go.Scatter(x=df_hist.index, y=ma_series, 
                                           mode='lines', name=ma, line=dict(color=colors[ma], width=1.5)))
            
            fig_p.update_layout(xaxis_rangeslider_visible=False, height=400, 
                                margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_p, use_container_width=True)

# --- MAIN ROUTER ---
if st.session_state.page == 'radar':
    render_radar()
else:
    render_analysis()
