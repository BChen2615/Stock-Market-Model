import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import subprocess
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)

from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH
import auth_system

# --- CONFIG ---
st.set_page_config(page_title="AI Stock Radar", layout="wide", page_icon="📈")

# --- SESSION STATE INIT ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'radar'
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = '2330'
if 'display_date' not in st.session_state:
    st.session_state.display_date = None
# Pagination State
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1
if 'page_size' not in st.session_state:
    st.session_state.page_size = 20

# --- DATABASE FUNCTIONS ---
def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_latest_db_date():
    conn = get_db_connection()
    try:
        query = "SELECT MAX(Date) FROM daily_predictions"
        date_str = conn.execute(query).fetchone()[0]
        return date_str
    except:
        return None
    finally:
        conn.close()

def get_available_dates():
    conn = get_db_connection()
    try:
        query = "SELECT DISTINCT Date FROM daily_predictions ORDER BY Date DESC"
        dates = [row[0] for row in conn.execute(query).fetchall()]
        return dates
    except:
        return []
    finally:
        conn.close()

def get_top_picks(date_str, sort_by='Prob_1d', limit=1000): # Increased limit for pagination
    conn = get_db_connection()
    # Check if Avg_Volume_5d exists (backward compatibility)
    try:
        query = f"""
            SELECT Stock_ID, Close_Price, Avg_Volume_5d, Prob_1d, Prob_2d, Prob_3d, Prob_7d, Prob_14d 
            FROM daily_predictions 
            WHERE Date = ? 
            ORDER BY {sort_by} DESC 
            LIMIT ?
        """
        df = pd.read_sql(query, conn, params=(date_str, limit))
    except:
        # Fallback if column missing
        query = f"""
            SELECT Stock_ID, Close_Price, Prob_1d, Prob_2d, Prob_3d, Prob_7d, Prob_14d 
            FROM daily_predictions 
            WHERE Date = ? 
            ORDER BY {sort_by} DESC 
            LIMIT ?
        """
        df = pd.read_sql(query, conn, params=(date_str, limit))
        df['Avg_Volume_5d'] = 0 # Default
        
    conn.close()
    return df

@st.cache_data(ttl=3600)
def get_stock_history(stock_id, days=200):
    try:
        df = prepare_training_data_v2(stock_id, target_days=1, path=DB_PATH, 
                                      keep_raw_prices=True, is_prediction_mode=True)
        if df is not None:
            return df.tail(days)
    except:
        pass
    return None

def get_historical_predictions(stock_id, days=60):
    conn = get_db_connection()
    query = """
        SELECT Date, Prob_1d, Close_Price 
        FROM daily_predictions 
        WHERE Stock_ID = ? 
        ORDER BY Date ASC
    """
    df = pd.read_sql(query, conn, params=(stock_id,))
    conn.close()
    
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        return df.tail(days)
    return None

# --- NAVIGATION ---
def go_to_analysis(stock_id):
    st.session_state.selected_stock = stock_id
    st.session_state.page = 'analysis'
    auth_system.log_access(st.session_state.username, "view_stock", stock_id)

def go_to_radar():
    st.session_state.page = 'radar'

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

# --- AUTH PAGES ---
def render_login():
    st.title("🔐 Login to AI Stock Radar")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                if auth_system.login_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    auth_system.log_access(username, "login", "success")
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submit_reg = st.form_submit_button("Register")
            if submit_reg:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_pass) < 4:
                    st.error("Password too short")
                else:
                    success, msg = auth_system.register_user(new_user, new_pass)
                    if success: st.success(msg)
                    else: st.error(msg)

# --- APP PAGES ---
def render_radar():
    st.title("📈 AI Stock Radar")
    st.caption(f"Welcome, {st.session_state.username} | [Logout]")
    if st.button("Logout"): logout()
    
    # 1. Check Data Freshness
    db_latest_date = get_latest_db_date()
    
    if not db_latest_date:
        st.error("System initializing... No data yet.")
        if st.button("🔧 Initialize Data"):
            batch_script = os.path.join(core_dir, 'Batch_Predict.py')
            subprocess.Popen([sys.executable, batch_script, '--force'])
            st.info("Initialization started...")
        return

    # Initialize display date
    if st.session_state.display_date is None:
        st.session_state.display_date = db_latest_date

    # 2. Status Card
    today = datetime.now().date()
    
    if db_latest_date > st.session_state.display_date:
        st.info(f"✨ New market data available ({db_latest_date})!")
        if st.button("🔄 Load New Data"):
            st.session_state.display_date = db_latest_date
            st.rerun()

    view_date_str = st.session_state.display_date
    view_dt = datetime.strptime(view_date_str, '%Y-%m-%d').date()
    view_diff = (today - view_dt).days
    
    if view_diff > 3:
        st.warning(f"⚠️ Viewing Old Data: {view_date_str} (Outdated by {view_diff} days)")
    else:
        st.success(f"✅ Data Up-to-Date: {view_date_str}")

    st.divider()

    # 3. Controls
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c1:
        available_dates = get_available_dates()
        if available_dates:
            selected_date = st.selectbox(
                "📅 Select Date", 
                options=available_dates,
                index=available_dates.index(st.session_state.display_date) if st.session_state.display_date in available_dates else 0
            )
            if selected_date != st.session_state.display_date:
                st.session_state.display_date = selected_date
                st.rerun()

    with c2:
        sort_col = st.selectbox("Sort Strategy", 
                               ['Prob_1d', 'Prob_2d', 'Prob_3d', 'Prob_7d', 'Prob_14d'], 
                               index=0)
    with c3:
        threshold = st.slider("Confidence >", 0.5, 0.9, 0.6, 0.01)

    # Admin Zone
    with st.sidebar:
        st.divider()
        st.markdown("### Admin Zone")
        if st.button("🔧 Force Background Update"):
            batch_script = os.path.join(core_dir, 'Batch_Predict.py')
            subprocess.Popen([sys.executable, batch_script, '--force'])
            st.toast("Background update started!")

    # Search
    col_search, _ = st.columns([1, 2])
    with col_search:
        search_input = st.text_input("🔍 Search Stock ID", placeholder="e.g. 2330")
        if search_input:
            go_to_analysis(search_input)
            st.rerun()

    st.divider()
    
    # Data
    df = get_top_picks(st.session_state.display_date, sort_by=sort_col, limit=1000)
    
    if df.empty:
        st.warning(f"No predictions found for {st.session_state.display_date}.")
        return

    df_filtered = df[df[sort_col] > threshold].reset_index(drop=True)
    
    # --- Pagination Logic ---
    total_rows = len(df_filtered)
    page_size = st.session_state.page_size
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Ensure page number is valid
    if st.session_state.page_number > total_pages:
        st.session_state.page_number = max(1, total_pages)
        
    start_idx = (st.session_state.page_number - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    df_page = df_filtered.iloc[start_idx:end_idx]
    
    st.markdown(f"### Top Opportunities ({total_rows} found)")
    
    # Header
    cols = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8])
    headers = ["Stock", "Price", "Avg Vol (5d)", "1-Day %", "3-Day %", "7-Day %", "14-Day %", "Action"]
    for col, h in zip(cols, headers):
        col.markdown(f"**{h}**")
    
    st.divider()
    
    # Rows
    for index, row in df_page.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8])
        
        c1.write(f"**{row['Stock_ID']}**")
        c2.write(f"{row['Close_Price']:.2f}")
        
        # Format Volume (e.g. 1.2M or 500K)
        vol = row['Avg_Volume_5d']
        if vol > 1_000_000:
            vol_str = f"{vol/1_000_000:.1f}M"
        elif vol > 1_000:
            vol_str = f"{vol/1_000:.0f}K"
        else:
            vol_str = f"{vol:.0f}"
        c3.write(vol_str)
        
        def color_prob(p):
            if p > 0.6: return f":green[{p:.1%}]"
            elif p < 0.4: return f":red[{p:.1%}]"
            else: return f"{p:.1%}"

        c4.markdown(color_prob(row['Prob_1d']))
        c5.markdown(color_prob(row['Prob_3d']))
        c6.markdown(color_prob(row['Prob_7d']))
        c7.markdown(color_prob(row['Prob_14d']))
        
        if c8.button("👉", key=f"btn_{row['Stock_ID']}"):
            go_to_analysis(row['Stock_ID'])
            st.rerun()

    st.divider()
    
    # Pagination Controls
    p1, p2, p3, p4, p5 = st.columns([1, 1, 2, 1, 1])
    
    with p1:
        if st.button("Previous") and st.session_state.page_number > 1:
            st.session_state.page_number -= 1
            st.rerun()
            
    with p3:
        st.write(f"Page {st.session_state.page_number} of {total_pages}")
        
    with p5:
        if st.button("Next") and st.session_state.page_number < total_pages:
            st.session_state.page_number += 1
            st.rerun()

def render_analysis():
    stock_id = st.session_state.selected_stock
    
    c1, c2 = st.columns([1, 10])
    if c1.button("← Back"):
        go_to_radar()
        st.rerun()
        
    c2.title(f"📊 Analysis: {stock_id}")
    
    display_date = st.session_state.display_date
    
    conn = get_db_connection()
    row = pd.read_sql("SELECT * FROM daily_predictions WHERE Date = ? AND Stock_ID = ?", 
                      conn, params=(display_date, stock_id))
    conn.close()
    
    if row.empty:
        st.error(f"No prediction data found for {stock_id} on {display_date}.")
        return

    data = row.iloc[0]
    
    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("1-Day Prob", f"{data['Prob_1d']:.1%}")
    m2.metric("2-Day Prob", f"{data['Prob_2d']:.1%}")
    m3.metric("3-Day Prob", f"{data['Prob_3d']:.1%}")
    m4.metric("7-Day Prob", f"{data['Prob_7d']:.1%}")
    m5.metric("14-Day Prob", f"{data['Prob_14d']:.1%}")
    
    st.divider()
    
    # Charts
    col_chart1, col_chart2 = st.columns([1, 2])
    
    with col_chart1:
        st.subheader("AI Confidence Term Structure")
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
        st.subheader("Price History & Indicators")
        
        cc1, cc2 = st.columns(2)
        with cc1:
            chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
        with cc2:
            ma_options = st.multiselect("Indicators", ["SMA 5", "SMA 10", "SMA 20", "SMA 60"], default=["SMA 5", "SMA 20"])
            
        df_hist = get_stock_history(stock_id, days=150)
        
        if df_hist is not None:
            fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                  vertical_spacing=0.05, row_heights=[0.7, 0.3],
                                  subplot_titles=("Price", "Historical AI Prediction"))
            
            if chart_type == "Candlestick":
                fig_p.add_trace(go.Candlestick(x=df_hist.index,
                    open=df_hist['Open'], high=df_hist['High'],
                    low=df_hist['Low'], close=df_hist['Close'], name='OHLC'), row=1, col=1)
            else:
                fig_p.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], 
                                           mode='lines', name='Close', line=dict(color='blue')), row=1, col=1)
            
            colors = {"SMA 5": "orange", "SMA 10": "cyan", "SMA 20": "purple", "SMA 60": "gray"}
            for ma in ma_options:
                window = int(ma.split()[1])
                ma_series = df_hist['Close'].rolling(window=window).mean()
                fig_p.add_trace(go.Scatter(x=df_hist.index, y=ma_series, 
                                           mode='lines', name=ma, line=dict(color=colors[ma], width=1.5)), row=1, col=1)
            
            df_preds = get_historical_predictions(stock_id)
            if df_preds is not None:
                fig_p.add_trace(go.Scatter(x=df_preds['Date'], y=df_preds['Prob_1d'], 
                                           name='AI Prob (1d)', line=dict(color='magenta', width=1), fill='tozeroy'), row=2, col=1)
                fig_p.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)
                fig_p.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1)
            
            fig_p.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_p, use_container_width=True)

# --- MAIN ENTRY ---
if not st.session_state.authenticated:
    render_login()
else:
    if st.session_state.page == 'radar':
        render_radar()
    else:
        render_analysis()
