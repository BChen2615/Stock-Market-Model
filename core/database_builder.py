import twstock
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import requests
import os
from datetime import datetime, timedelta

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWDB_DIR = os.path.join(BASE_DIR, 'data', 'twstock.db')
START_DATE_DEFAULT = "2020-01-01"

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(TWDB_DIR)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tw_stock_prices (
            Date DATETIME,
            Stock_ID TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER,
            Type TEXT,
            PRIMARY KEY (Date, Stock_ID)
        );
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_meta (
            Stock_ID TEXT PRIMARY KEY,
            Name TEXT,
            Market TEXT,
            Industry TEXT,
            Updated_At DATETIME
        );
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_audit_log (
            Date DATETIME,
            Stock_ID TEXT,
            Reason TEXT,
            Raw_Data TEXT
        );
    """)
    conn.commit()
    return conn

# --- HELPER FUNCTION: Data Cleaning ---
def clean_and_validate_data(df, stock_id):
    if df.empty: return None, None
    df = df.reset_index()
    if 'Date' not in df.columns: return None, None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] <= datetime.now() + timedelta(days=1)]
    
    na_rows = df[df[['Open', 'High', 'Low', 'Close']].isna().any(axis=1)].copy()
    if not na_rows.empty: na_rows['Reason'] = 'Missing Values'
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    if df.empty: return None, na_rows
    
    mask_zero = (df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0) | (df['Close'] <= 0)
    mask_logic = (df['High'] < df['Low']) | (df['High'] < df['Open']) | (df['High'] < df['Close']) | (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
    bad_data = df[mask_zero | mask_logic].copy()
    if not bad_data.empty: bad_data['Reason'] = 'Logic Error'
    df = df[~(mask_zero | mask_logic)].copy()
    
    df['pct_change'] = df['Close'].pct_change().abs()
    mask_extreme = df['pct_change'] > 0.5
    extreme_data = df[mask_extreme].copy()
    if not extreme_data.empty: extreme_data['Reason'] = 'Extreme Volatility'
    df = df[~mask_extreme]
    
    audit_logs = pd.concat([na_rows, bad_data, extreme_data]) if 'na_rows' in locals() else None
    if 'pct_change' in df.columns: df = df.drop(columns=['pct_change'])
    
    return df, audit_logs

# --- HELPER: Get Last Date in DB ---
def get_last_dates_map(conn):
    """
    Returns a dict {Stock_ID: Last_Date (datetime)} for all stocks.
    """
    try:
        df = pd.read_sql("SELECT Stock_ID, MAX(Date) as Last_Date FROM tw_stock_prices GROUP BY Stock_ID", conn)
        df['Last_Date'] = pd.to_datetime(df['Last_Date'])
        return df.set_index('Stock_ID')['Last_Date'].to_dict()
    except:
        return {}

# --- BULK DOWNLOAD LOGIC ---
def process_stocks_bulk(stock_list, stock_type, conn):
    print(f"[WAIT] Starting BULK processing for {len(stock_list)} {stock_type} stocks...")
    
    # Pre-load existing dates to filter duplicates BEFORE insert
    last_dates_map = get_last_dates_map(conn)
    
    suffix = ".TW" if stock_type == "TW" else ".TWO"
    tickers = [f"{code}{suffix}" for code in stock_list]
    
    CHUNK_SIZE = 100
    total_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i+CHUNK_SIZE]
        print(f"  Processing chunk {i//CHUNK_SIZE + 1}/{total_chunks} ({len(chunk)} stocks)...")
        
        try:
            data = yf.download(chunk, start=START_DATE_DEFAULT, group_by="ticker", auto_adjust=False, progress=False, threads=True)
            
            if data.empty: continue
            
            # Handle single ticker case
            if len(chunk) == 1:
                ticker = chunk[0]
                code = ticker.replace(suffix, "")
                clean_df, _ = clean_and_validate_data(data, code)
                if clean_df is not None and not clean_df.empty:
                    # Filter new data only
                    last_date = last_dates_map.get(code)
                    if last_date:
                        clean_df = clean_df[clean_df['Date'] > last_date]
                    
                    if not clean_df.empty:
                        clean_df["Type"] = stock_type
                        clean_df["Stock_ID"] = code
                        save_to_db(clean_df, conn)
                continue

            # Handle multi ticker case
            downloaded_tickers = data.columns.levels[0]
            
            for ticker in downloaded_tickers:
                df_ticker = data[ticker].copy()
                df_ticker = df_ticker.dropna(how='all')
                
                if df_ticker.empty: continue
                
                code = ticker.replace(suffix, "")
                clean_df, error_log = clean_and_validate_data(df_ticker, code)
                
                if clean_df is not None and not clean_df.empty:
                    # --- CRITICAL FIX: Filter out existing dates ---
                    last_date = last_dates_map.get(code)
                    if last_date:
                        clean_df = clean_df[clean_df['Date'] > last_date]
                    
                    if not clean_df.empty:
                        clean_df["Type"] = stock_type
                        clean_df["Stock_ID"] = code
                        save_to_db(clean_df, conn)
                
        except Exception as e:
            print(f"  [ERROR] Chunk failed: {e}")

def save_to_db(df, conn):
    cols_to_keep = ["Date", "Stock_ID", "Open", "High", "Low", "Close", "Volume", "Type"]
    df = df[[c for c in cols_to_keep if c in df.columns]]
    try:
        # Now we can safely append because we filtered duplicates
        df.to_sql("tw_stock_prices", conn, if_exists="append", index=False)
    except sqlite3.IntegrityError:
        pass

# --- STOCK LIST MANAGEMENT ---
def get_stock_codes_list(conn, force_update=False):
    if not force_update:
        try:
            df_meta = pd.read_sql("SELECT Stock_ID, Market FROM stock_meta", conn)
            if not df_meta.empty:
                print(f"[OK] Loaded {len(df_meta)} stocks from DB cache.")
                tw_codes = df_meta[df_meta['Market'] == 'TW']['Stock_ID'].tolist()
                two_codes = df_meta[df_meta['Market'] == 'TWO']['Stock_ID'].tolist()
                return tw_codes, two_codes
        except Exception:
            pass 

    print("[WAIT] Crawling stock codes from TWSE/TPEX...")
    urls = {
        "TW": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "TWO": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    }
    
    meta_data = []
    import urllib3
    urllib3.disable_warnings()
    
    for m_type, url in urls.items():
        try:
            r = requests.get(url, verify=False)
            dfs = pd.read_html(r.text)
            if not dfs: continue
            df = dfs[0]
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            
            if 'CFICode' in df.columns: df = df[df['CFICode'] == 'ESVUFR']
            
            df[['Code', 'Name']] = df['有價證券代號及名稱'].str.split(pat=r'\s+', n=1, expand=True)
            
            for _, row in df.iterrows():
                code = row['Code']
                name = row['Name']
                industry = row['產業別'] if '產業別' in row else None
                
                if code.isdigit() and len(code) == 4:
                    meta_data.append({
                        'Stock_ID': code,
                        'Name': name,
                        'Market': m_type,
                        'Industry': industry,
                        'Updated_At': datetime.now()
                    })
                    
        except Exception as e:
            print(f"[ERROR] {m_type}: {e}")
            
    if meta_data:
        df_meta = pd.DataFrame(meta_data)
        print(f"[OK] Saving {len(df_meta)} stocks to stock_meta table...")
        df_meta.to_sql('stock_meta', conn, if_exists='replace', index=False)
        
        tw_codes = df_meta[df_meta['Market'] == 'TW']['Stock_ID'].tolist()
        two_codes = df_meta[df_meta['Market'] == 'TWO']['Stock_ID'].tolist()
        return tw_codes, two_codes
    
    return [], []

if __name__ == "__main__":
    conn = init_db()
    TW_CODES, TWO_CODES = get_stock_codes_list(conn, force_update=False)
    
    if TW_CODES: process_stocks_bulk(TW_CODES, "TW", conn)
    if TWO_CODES: process_stocks_bulk(TWO_CODES, "TWO", conn)
    
    conn.close()
    print("[OK] Database build complete.")
