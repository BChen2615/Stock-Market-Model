import os
import pandas as pd
import numpy as np
import joblib
import sqlite3
import datetime
import sys
import pytz # Timezone support

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH
# Import database builder to trigger update
import database_builder

# --- CONFIG ---
BASE_DIR = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
HORIZONS = [1, 2, 3, 7, 14]

def init_prediction_db():
    """Create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists and has the new column
    try:
        cursor.execute("SELECT Avg_Volume_5d FROM daily_predictions LIMIT 1")
    except sqlite3.OperationalError:
        # Table doesn't exist or column missing, drop and recreate
        cursor.execute("DROP TABLE IF EXISTS daily_predictions")
        cursor.execute("""
            CREATE TABLE daily_predictions (
                Date DATE,
                Stock_ID TEXT,
                Close_Price REAL,
                Avg_Volume_5d REAL, -- New Column
                Prob_1d REAL,
                Prob_2d REAL,
                Prob_3d REAL,
                Prob_7d REAL,
                Prob_14d REAL,
                Updated_At DATETIME,
                PRIMARY KEY (Date, Stock_ID)
            )
        """)
        print("Prediction DB initialized (Recreated).")
        
    conn.commit()
    conn.close()

def load_models():
    """Load all horizon models."""
    models = {}
    for h in HORIZONS:
        path = os.path.join(MODELS_DIR, f'xgb_universal_{h}d.pkl')
        if os.path.exists(path):
            models[h] = joblib.load(path)
        else:
            print(f"Warning: Model {h}d not found at {path}")
    return models

def get_all_stock_ids():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Stock_ID FROM tw_stock_prices")
    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return [s for s in stocks if len(s) == 4]

def update_data_if_needed():
    """
    Checks if data needs update based on Taiwan time.
    If it's past 13:30 TW time, we should have today's data.
    """
    tw_tz = pytz.timezone('Asia/Taipei')
    now_tw = datetime.datetime.now(tw_tz)
    
    print(f"Current Taiwan Time: {now_tw}")
    
    # Check latest date in DB
    conn = sqlite3.connect(DB_PATH)
    try:
        last_date_str = conn.execute("SELECT MAX(Date) FROM tw_stock_prices").fetchone()[0]
        last_date = datetime.datetime.strptime(last_date_str, '%Y-%m-%d %H:%M:%S').date()
    except:
        last_date = datetime.date(2000, 1, 1)
    
    print(f"Latest DB Date: {last_date}")
    
    # Logic:
    # If today is weekday AND time > 13:30 AND last_date < today: UPDATE
    # If today is weekend: We don't strictly need update, but good to check Friday.
    
    is_weekday = now_tw.weekday() < 5
    is_market_closed = now_tw.hour >= 14 or (now_tw.hour == 13 and now_tw.minute >= 30)
    
    today_date = now_tw.date()
    
    if is_weekday and is_market_closed and last_date < today_date:
        print("⚠️ Data is outdated. Triggering download...")
        
        # Re-fetch codes to be safe
        tw_codes, two_codes = database_builder.get_stock_codes_list(conn)
        
        if tw_codes: database_builder.process_stocks_bulk(tw_codes, "TW", conn)
        if two_codes: database_builder.process_stocks_bulk(two_codes, "TWO", conn)
        
        print("✅ Data update complete.")
    else:
        print("✅ Data is up-to-date (or market not closed yet).")
    
    conn.close()

def run_batch_prediction(force_update=False):
    print("--- Starting Batch Pipeline ---")
    
    # 1. Update Data First
    if force_update:
        print("Force updating data...")
        conn = sqlite3.connect(DB_PATH)
        tw_codes, two_codes = database_builder.get_stock_codes_list(conn)
        
        if tw_codes: database_builder.process_stocks_bulk(tw_codes, "TW", conn)
        if two_codes: database_builder.process_stocks_bulk(two_codes, "TWO", conn)
        conn.close()
    else:
        update_data_if_needed()
    
    # 2. Prediction
    init_prediction_db()
    models = load_models()
    if not models: return

    all_stocks = get_all_stock_ids()
    print(f"Scanning {len(all_stocks)} stocks for prediction...")
    
    results = []
    
    # Use a counter for progress
    total = len(all_stocks)
    for i, stock in enumerate(all_stocks):
        if i % 50 == 0: print(f"Predicting {i}/{total}...")
            
        try:
            # Load data with raw prices AND prediction mode
            df = prepare_training_data_v2(stock, target_days=1, path=DB_PATH, 
                                          keep_raw_prices=True, is_prediction_mode=True)
            
            if df is not None and not df.empty:
                last_row = df.iloc[[-1]]
                last_date = last_row.index[0].date()
                last_close = last_row['Close'].values[0]
                
                # Calculate 5-day Avg Volume
                # We need at least 5 days of data
                if len(df) >= 5:
                    avg_vol = df['Volume'].tail(5).mean()
                else:
                    avg_vol = df['Volume'].mean()
                
                drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target', 
                             'Open', 'High', 'Low', 'Close', 'Volume']
                X = last_row.drop(columns=[c for c in drop_cols if c in last_row.columns])
                X = X.select_dtypes(include=[np.number])
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                row_result = {
                    'Date': str(last_date),
                    'Stock_ID': stock,
                    'Close_Price': float(last_close),
                    'Avg_Volume_5d': float(avg_vol),
                    'Updated_At': datetime.datetime.now().isoformat()
                }
                
                for h, model in models.items():
                    prob = model.predict_proba(X)[0, 1]
                    row_result[f'Prob_{h}d'] = float(prob)
                
                results.append(row_result)
                
        except Exception as e:
            if i < 5:
                print(f"Error on {stock}: {e}")
            continue
            
    if results:
        print(f"Saving {len(results)} predictions...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        sql = """
            INSERT OR REPLACE INTO daily_predictions 
            (Date, Stock_ID, Close_Price, Avg_Volume_5d, Prob_1d, Prob_2d, Prob_3d, Prob_7d, Prob_14d, Updated_At)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data_to_insert = []
        for r in results:
            p1 = r.get('Prob_1d', None)
            p2 = r.get('Prob_2d', None)
            p3 = r.get('Prob_3d', None)
            p7 = r.get('Prob_7d', None)
            p14 = r.get('Prob_14d', None)
            
            data_to_insert.append((
                r['Date'], r['Stock_ID'], r['Close_Price'], r['Avg_Volume_5d'],
                p1, p2, p3, p7, p14, r['Updated_At']
            ))
            
        cursor.executemany(sql, data_to_insert)
        conn.commit()
        conn.close()
        print("Batch prediction complete.")

if __name__ == "__main__":
    # Check for command line arg to force update
    force = len(sys.argv) > 1 and sys.argv[1] == '--force'
    run_batch_prediction(force_update=force)
