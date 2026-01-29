import twstock
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# --- CONFIG ---
TWDB_DIR = '../data/twstock.db'
START_DATE = "2020-01-01"  # Recommended to set a specific date; "5y" changes over time
# Set threshold for abnormal volatility (e.g., >20% daily change is suspicious.
# Taiwan's limit is 10%, but set wider to account for dividends/splits).
OUTLIER_THRESHOLD = 0.20

# --- DATABASE SETUP ---
conn = sqlite3.connect(TWDB_DIR)
cursor = conn.cursor()

# Create main data table
cursor.execute("""
               CREATE TABLE IF NOT EXISTS tw_stock_prices
               (
                   Date
                   DATETIME,
                   Stock_ID
                   TEXT,
                   Open
                   REAL,
                   High
                   REAL,
                   Low
                   REAL,
                   Close
                   REAL,
                   Volume
                   INTEGER,
                   Type
                   TEXT,
                   PRIMARY
                   KEY
               (
                   Date,
                   Stock_ID
               )
                   );
               """)

# Create an "Error Log Table" to record filtered abnormal data for future inspection
cursor.execute("""
               CREATE TABLE IF NOT EXISTS data_audit_log
               (
                   Date
                   DATETIME,
                   Stock_ID
                   TEXT,
                   Reason
                   TEXT,
                   Raw_Data
                   TEXT
               );
               """)
conn.commit()


# --- HELPER FUNCTION: Data Cleaning and Validation ---
def clean_and_validate_data(df, stock_id):
    """
    Cleans and validates the DataFrame, returning clean data and a list of anomalies.
    """
    if df.empty:
        return None, None

    # 1. Basic Cleaning: Reset index and ensure date format
    df = df.reset_index()
    if 'Date' not in df.columns:  # yfinance sometimes uses different index names
        return None, None
    df['Date'] = pd.to_datetime(df['Date'])

    # 2. Remove future data (YF occasionally has future dates due to timezone issues)
    df = df[df['Date'] <= datetime.now()]

    # 3. Check for missing values (Drop NaNs in OHLC)
    # Record rows to be dropped for logging
    na_rows = df[df[['Open', 'High', 'Low', 'Close']].isna().any(axis=1)].copy()
    if not na_rows.empty:
        na_rows['Reason'] = 'Missing Values (NaN)'

    # Actually drop the rows
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    if df.empty: return None, na_rows

    # 4. Price Logic Check (Sanity Check)
    # Rule A: Price must be > 0
    # Rule B: High must be the daily maximum (High >= Open, Close, Low)
    # Rule C: Low must be the daily minimum (Low <= Open, Close, High)

    mask_zero = (df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0) | (df['Close'] <= 0)
    mask_logic_error = (df['High'] < df['Low']) | (df['High'] < df['Open']) | (df['High'] < df['Close']) | (
                df['Low'] > df['Open']) | (df['Low'] > df['Close'])

    bad_data = df[mask_zero | mask_logic_error].copy()
    if not bad_data.empty:
        bad_data['Reason'] = 'Logic Error (Zero or H/L invalid)'

    # Filter out bad data
    df = df[~(mask_zero | mask_logic_error)].copy()

    # 5. Outlier Detection
    # While deletion isn't mandatory, we can flag them.
    # Calculate daily percentage change
    df['pct_change'] = df['Close'].pct_change().abs()

    # If daily volatility exceeds 50% (extreme anomaly), it's usually a data error
    # (even with dividends/splits, a 50% single-day drop is rare).
    # We are conservative here, only recording "extreme" glitches.
    mask_extreme = df['pct_change'] > 0.5
    extreme_data = df[mask_extreme].copy()
    if not extreme_data.empty:
        extreme_data['Reason'] = 'Extreme Volatility (>50%)'
        # Optional: Decide whether to delete these; demonstrating deletion here.
        df = df[~mask_extreme]

    # Merge all bad data to write to Log
    audit_logs = pd.concat([na_rows, bad_data, extreme_data]) if 'na_rows' in locals() else None

    # Clean up temporary columns
    if 'pct_change' in df.columns:
        df = df.drop(columns=['pct_change'])

    return df, audit_logs


# --- MAIN LOOP ---

def process_stocks(stock_list, stock_type):
    print(f"Starting processing for {stock_type} stocks...")
    total = len(stock_list)

    for i, code in enumerate(stock_list):
        if len(code) != 4: continue

        # Progress display
        if i % 10 == 0:
            print(f"Processing {code}.{stock_type} ({i}/{total})...")

        try:
            # Download data
            ticker = f"{code}.{stock_type}"
            df = yf.download(ticker, period="5y", auto_adjust=False, multi_level_index=False, progress=False)

            # --- Core Change: Add validation mechanism ---
            clean_df, error_log = clean_and_validate_data(df, code)

            # If data exists, write to database
            if clean_df is not None and not clean_df.empty:
                clean_df["Type"] = stock_type
                clean_df["Stock_ID"] = code

                # Remove unnecessary columns (Adj Close is handled by yf params, but just to be safe)
                cols_to_keep = ["Date", "Stock_ID", "Open", "High", "Low", "Close", "Volume", "Type"]
                # Ensure columns exist before selection
                clean_df = clean_df[[c for c in cols_to_keep if c in clean_df.columns]]

                # Write to DB (use try-except to avoid duplicate Primary Key errors)
                try:
                    clean_df.to_sql("tw_stock_prices", conn, if_exists="append", index=False)
                except sqlite3.IntegrityError:
                    # If data already exists, skip or consider updating
                    # print(f"Data for {code} already exists, skipping duplicates.")
                    pass

            # If there is abnormal data, write to Log table
            if error_log is not None and not error_log.empty:
                # Convert the entire row to string for storage
                error_log['Raw_Data'] = error_log.apply(lambda x: str(x.to_dict()), axis=1)
                error_log['Stock_ID'] = code
                error_log[['Date', 'Stock_ID', 'Reason', 'Raw_Data']].to_sql("data_audit_log", conn, if_exists="append",
                                                                             index=False)

        except Exception as e:
            print(f"Error processing {code}: {e}")


# Execute for TW (Listed on TWSE)
TW_CODES = [c for c in twstock.twse.keys() if len(c) == 4]
process_stocks(TW_CODES, "TW")

# Execute for TWO (Listed on TPEX)
TWO_CODES = [c for c in twstock.tpex.keys() if len(c) == 4]
process_stocks(TWO_CODES, "TWO")

conn.close()
print("Database build complete.")