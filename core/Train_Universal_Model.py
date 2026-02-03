import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sqlite3
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
# Use absolute path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_FULL_PATH = os.path.join(DATA_DIR, 'twstock.db')

# --- STOCK SELECTION (SECTOR LEADERS) ---
TRAIN_STOCKS = [
    '2330', '2454', '2303', '3711', '3034', # Semi
    '2881', '2882', '2891', '2886', '2884', # Fin
    '2317', '2308', '2382', '2357', '3231', # Tech
    '2002', '1301', '1303', '1101',         # Trad
    '2603', '2609', '2615'                  # Ship
]

# --- DYNAMIC TEST SET SELECTION ---
def get_all_stock_ids():
    conn = sqlite3.connect(DB_FULL_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Stock_ID FROM tw_stock_prices")
    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return [s for s in stocks if len(s) == 4]

# Get all stocks, exclude training stocks
ALL_STOCKS = get_all_stock_ids()
NON_TRAIN_STOCKS = [s for s in ALL_STOCKS if s not in TRAIN_STOCKS]

# Randomly sample 10 stocks for testing
if len(NON_TRAIN_STOCKS) >= 10:
    TEST_STOCKS = random.sample(NON_TRAIN_STOCKS, 10)
else:
    TEST_STOCKS = NON_TRAIN_STOCKS # Fallback if DB is small

print(f"Training on {len(TRAIN_STOCKS)} Sector Leaders.")
print(f"Testing on {len(TEST_STOCKS)} Random Unseen Stocks: {TEST_STOCKS}")

BUFFER_THRESHOLD = 0.005

def load_and_stack_data(stock_list, target_days, label="Training"):
    print(f"\n--- Loading {label} Data (Target: {target_days} days) ---")
    df_list = []
    
    total_stocks = len(stock_list)
    for i, stock_id in enumerate(stock_list):
        if i % 5 == 0:
            print(f"Processing {i}/{total_stocks}: {stock_id}...")
            
        try:
            df = prepare_training_data_v2(stock_id, target_days=target_days, path=DB_FULL_PATH)
            if df is not None and not df.empty:
                df['Stock_ID'] = stock_id
                df_list.append(df)
        except Exception as e:
            # print(f"Error loading {stock_id}: {e}")
            pass
            
    if not df_list:
        return None
        
    full_df = pd.concat(df_list, axis=0)
    full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Total {label} Samples: {len(full_df)}")
    return full_df

def train_model_for_horizon(days):
    print(f"\n=== Training Model for {days}-Day Horizon ===")
    
    # 1. Prepare Data
    df_train_all = load_and_stack_data(TRAIN_STOCKS, days, "Training Pool")
    if df_train_all is None: return

    # 2. Label Engineering
    threshold = BUFFER_THRESHOLD * np.sqrt(days)
    print(f"Using Threshold: {threshold:.2%}")
    
    mask_significant = (df_train_all['Future_Return'] > threshold) | (df_train_all['Future_Return'] < -threshold)
    df_clean = df_train_all[mask_significant].copy()
    df_clean['Target'] = (df_clean['Future_Return'] > 0).astype(int)
    
    # 3. Feature Selection
    drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target']
    X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
    X = X.select_dtypes(include=[np.number])
    y = df_clean['Target']
    
    # Scale Pos Weight
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos
    
    # 4. Train
    X = X.sort_index()
    y = y.sort_index()
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training XGBoost ({days}d)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # 5. Evaluate
    print(f"--- Validation Performance ({days}d) ---")
    evaluate_model(model, X_val, y_val)
    
    # 6. Generalization Test
    print(f"--- Generalization Test ({days}d) ---")
    df_test_all = load_and_stack_data(TEST_STOCKS, days, "Unseen Test Pool")
    if df_test_all is not None:
        mask_sig_test = (df_test_all['Future_Return'] > threshold) | (df_test_all['Future_Return'] < -threshold)
        df_test_clean = df_test_all[mask_sig_test].copy()
        df_test_clean['Target'] = (df_test_clean['Future_Return'] > 0).astype(int)
        
        X_unseen = df_test_clean.drop(columns=[c for c in drop_cols if c in df_test_clean.columns])
        X_unseen = X_unseen.select_dtypes(include=[np.number])
        y_unseen = df_test_clean['Target']
        
        evaluate_model(model, X_unseen, y_unseen)
    
    # 7. Save
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    model_name = f'xgb_universal_{days}d.pkl'
    joblib.dump(model, os.path.join(MODELS_DIR, model_name))
    print(f"Model saved: {model_name}")

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y, y_pred, target_names=['Down', 'Up']))

if __name__ == "__main__":
    # Train for multiple horizons
    for d in [1, 2, 3, 7, 14]:
        train_model_for_horizon(d)
