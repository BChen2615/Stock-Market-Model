import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import twstock
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_FULL_PATH = os.path.join(DATA_DIR, 'twstock.db')

# --- STOCK SELECTION ---
# We want to train on a large, representative set of stocks.
# Strategy: Use Top 100 stocks by market cap (or just a large list of known tickers)
# For simplicity, we can use the twstock library to get a list, or define a manual list.
# Here we define a manual list of ~50 major stocks across sectors to ensure diversity without OOM.

TRAIN_STOCKS = [
    # Semiconductor
    '2330', '2454', '2303', '3711', '3034', '2379', '3443', '3035',
    # Financial
    '2881', '2882', '2891', '2886', '2884', '2892', '2880', '2885', '2883', '2890',
    # Tech / Electronics
    '2317', '2308', '2382', '2357', '3231', '2395', '4938', '2356', '2353', '2324',
    # Traditional / Materials
    '2002', '1301', '1303', '1326', '1101', '1102', '2105',
    # Shipping
    '2603', '2609', '2615',
    # Others
    '2912', '9910', '5871', '5876', '2412', '3045', '2408'
]

# Stocks to test generalization (Model has NEVER seen these)
# We pick some mid-cap stocks that are not in the top list
TEST_STOCKS = [
    '2344', # Winbond (Tech)
    '2834', # TBB (Finance)
    '1402', # Far Eastern New Century (Textile)
    '2618', # Eva Air (Aviation)
]

TARGET_DAYS = 1
BUFFER_THRESHOLD = 0.005

def load_and_stack_data(stock_list, label="Training"):
    print(f"\n--- Loading {label} Data ---")
    df_list = []
    
    total_stocks = len(stock_list)
    for i, stock_id in enumerate(stock_list):
        if i % 5 == 0:
            print(f"Processing {i}/{total_stocks}: {stock_id}...")
            
        try:
            df = prepare_training_data_v2(stock_id, target_days=TARGET_DAYS, path=DB_FULL_PATH)
            if df is not None and not df.empty:
                # Add Stock_ID column back just for tracking (will drop before training)
                df['Stock_ID'] = stock_id
                df_list.append(df)
        except Exception as e:
            print(f"Error loading {stock_id}: {e}")
            
    if not df_list:
        return None
        
    full_df = pd.concat(df_list, axis=0)
    
    # --- CRITICAL FIX: Handle Infinite Values ---
    # Replace inf/-inf with NaN, then drop NaNs
    full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Total {label} Samples: {len(full_df)}")
    return full_df

def train_universal_model():
    # 1. Prepare Training Data
    print(f"Training on {len(TRAIN_STOCKS)} stocks...")
    df_train_all = load_and_stack_data(TRAIN_STOCKS, "Training Pool")
    if df_train_all is None: return

    # 2. Label Engineering (Same as before)
    mask_significant = (df_train_all['Future_Return'] > BUFFER_THRESHOLD) | (df_train_all['Future_Return'] < -BUFFER_THRESHOLD)
    df_clean = df_train_all[mask_significant].copy()
    df_clean['Target'] = (df_clean['Future_Return'] > 0).astype(int)
    
    print(f"Cleaned Training Samples: {len(df_clean)}")
    print(f"Class Distribution:\n{df_clean['Target'].value_counts(normalize=True)}")
    
    # 3. Feature Selection
    drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target']
    # Ensure we only drop columns that exist
    X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
    X = X.select_dtypes(include=[np.number])
    y = df_clean['Target']
    
    # Calculate Scale Pos Weight
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos
    
    # 4. Train Model
    # Sort by index (Date) to ensure time-based split
    X = X.sort_index()
    y = y.sort_index()
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("Training Universal XGBoost...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=500, # More trees for more data
        learning_rate=0.02,
        max_depth=6, # Slightly deeper for complex patterns
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    # Remove early_stopping_rounds to avoid potential issues with validation set size/distribution
    # Or keep it if validation set is robust enough
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # 5. Evaluate on Validation Set (Seen Stocks, Unseen Time)
    print("\n--- Validation Set Performance (Seen Stocks, Future Time) ---")
    evaluate_model(model, X_val, y_val)
    
    # 6. Evaluate on Test Stocks (Unseen Stocks)
    print("\n--- Generalization Test (Unseen Stocks) ---")
    df_test_all = load_and_stack_data(TEST_STOCKS, "Unseen Test Pool")
    
    if df_test_all is not None:
        mask_sig_test = (df_test_all['Future_Return'] > BUFFER_THRESHOLD) | (df_test_all['Future_Return'] < -BUFFER_THRESHOLD)
        df_test_clean = df_test_all[mask_sig_test].copy()
        df_test_clean['Target'] = (df_test_clean['Future_Return'] > 0).astype(int)
        
        X_unseen = df_test_clean.drop(columns=[c for c in drop_cols if c in df_test_clean.columns])
        X_unseen = X_unseen.select_dtypes(include=[np.number])
        y_unseen = df_test_clean['Target']
        
        evaluate_model(model, X_unseen, y_unseen)
    
    # 7. Save
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(model, os.path.join(MODELS_DIR, 'xgb_universal.pkl'))
    print("\nUniversal Model Saved.")

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y, y_pred, target_names=['Down', 'Up']))
    
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    train_universal_model()
