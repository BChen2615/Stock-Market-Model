import os
import pandas as pd
import numpy as np
import joblib
import sqlite3
import datetime
import sys

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
BASE_DIR = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
HORIZONS = [1, 2, 3, 7, 14]

def init_prediction_db():
    """Create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_predictions (
            Date DATE,
            Stock_ID TEXT,
            Close_Price REAL,
            Prob_1d REAL,
            Prob_2d REAL,
            Prob_3d REAL,
            Prob_7d REAL,
            Prob_14d REAL,
            Updated_At DATETIME,
            PRIMARY KEY (Date, Stock_ID)
        )
    """)
    conn.commit()
    conn.close()
    print("Prediction DB initialized.")

def load_models():
    """Load all horizon models."""
    models = {}
    for h in HORIZONS:
        path = os.path.join(MODELS_DIR, f'xgb_universal_{h}d.pkl')
        if os.path.exists(path):
            models[h] = joblib.load(path)
            print(f"Loaded model: {h}d")
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

def run_batch_prediction():
    print("--- Starting Batch Prediction ---")
    init_prediction_db()
    
    models = load_models()
    if not models:
        print("No models found. Aborting.")
        return

    all_stocks = get_all_stock_ids()
    print(f"Scanning {len(all_stocks)} stocks...")
    
    results = []
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    
    # We use a counter to print progress
    count = 0
    
    for stock in all_stocks:
        count += 1
        if count % 50 == 0: print(f"Processed {count} stocks...")
            
        try:
            # Load data (we need raw prices to get the latest Close)
            # We use target_days=1 just to get the feature set structure, 
            # the actual target shift doesn't matter for inference on the LAST row.
            df = prepare_training_data_v2(stock, target_days=1, path=DB_PATH, keep_raw_prices=True)
            
            if df is not None and not df.empty:
                # Get the very last row (latest data)
                last_row = df.iloc[[-1]]
                last_date = last_row.index[0].date()
                last_close = last_row['Close'].values[0]
                
                # Prepare Features
                drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target', 
                             'Open', 'High', 'Low', 'Close', 'Volume']
                X = last_row.drop(columns=[c for c in drop_cols if c in last_row.columns])
                X = X.select_dtypes(include=[np.number])
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Predict for all horizons
                row_result = {
                    'Date': str(last_date), # Use the data date, not today's date
                    'Stock_ID': stock,
                    'Close_Price': float(last_close),
                    'Updated_At': datetime.datetime.now().isoformat()
                }
                
                for h, model in models.items():
                    prob = model.predict_proba(X)[0, 1]
                    row_result[f'Prob_{h}d'] = float(prob)
                
                results.append(row_result)
                
        except Exception as e:
            # print(f"Error {stock}: {e}")
            continue
            
    # Save to DB
    if results:
        print(f"Saving {len(results)} predictions to database...")
        conn = sqlite3.connect(DB_PATH)
        
        # We use a loop or executemany to Insert/Replace
        # Using pandas to_sql with 'append' might fail on duplicates if we don't handle PKs.
        # Better to use SQL INSERT OR REPLACE
        
        cursor = conn.cursor()
        sql = """
            INSERT OR REPLACE INTO daily_predictions 
            (Date, Stock_ID, Close_Price, Prob_1d, Prob_2d, Prob_3d, Prob_7d, Prob_14d, Updated_At)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        data_to_insert = []
        for r in results:
            # Ensure all probs exist (handle missing models)
            p1 = r.get('Prob_1d', None)
            p2 = r.get('Prob_2d', None)
            p3 = r.get('Prob_3d', None)
            p7 = r.get('Prob_7d', None)
            p14 = r.get('Prob_14d', None)
            
            data_to_insert.append((
                r['Date'], r['Stock_ID'], r['Close_Price'],
                p1, p2, p3, p7, p14, r['Updated_At']
            ))
            
        cursor.executemany(sql, data_to_insert)
        conn.commit()
        conn.close()
        print("Batch prediction complete.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_batch_prediction()
