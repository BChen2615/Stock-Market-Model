import yfinance as yf
import pandas as pd
import sqlite3
import os

# --- CONFIG ---
DB_PATH = '../data/twstock.db'
SYMBOLS = {
    'TSM': 'TSM_ADR',      # TSMC ADR
    '^SOX': 'SOX_Index',   # PHLX Semiconductor Sector
    '^TWII': 'TWII_Index', # TAIEX
    '^GSPC': 'SP500_Index' # S&P 500
}

def fetch_and_store_external_data():
    print("--- Fetching External Market Data ---")
    
    # Connect to DB
    # Ensure path is absolute to avoid errors
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_full_path = os.path.join(base_dir, 'data', 'twstock.db')
    
    conn = sqlite3.connect(db_full_path)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS external_market_data (
            Date DATETIME,
            Symbol TEXT,
            Close REAL,
            Volume INTEGER,
            PRIMARY KEY (Date, Symbol)
        )
    """)
    conn.commit()
    
    for ticker, name in SYMBOLS.items():
        print(f"Downloading {ticker} ({name})...")
        try:
            # Download data (last 5 years to match stock data)
            df = yf.download(ticker, period="5y", progress=False, auto_adjust=False)
            
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue
                
            # Clean data
            df = df.reset_index()
            # yfinance might return multi-level columns, flatten them if necessary
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df['Date'] = pd.to_datetime(df['Date'])
            df['Symbol'] = name
            
            # Select columns
            cols_to_keep = ['Date', 'Symbol', 'Close', 'Volume']
            # Ensure columns exist (Indices might not have Volume)
            if 'Volume' not in df.columns:
                df['Volume'] = 0
                
            df_save = df[cols_to_keep].dropna()
            
            # Write to DB
            df_save.to_sql('external_market_data', conn, if_exists='append', index=False)
            print(f"Saved {len(df_save)} rows for {name}.")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            # If table already has data, to_sql might fail on unique constraint.
            # We can ignore this for now or use a more complex upsert logic.
            # For simplicity, we assume this is a one-time fetch or we catch the error.
            if "UNIQUE constraint failed" in str(e):
                print("Data already exists. Skipping.")
            
    conn.close()
    print("External data fetch complete.")

if __name__ == "__main__":
    fetch_and_store_external_data()
