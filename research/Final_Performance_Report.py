import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)

from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
BASE_DIR = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_universal_1d.pkl')
DB_FULL_PATH = os.path.join(BASE_DIR, 'data', 'twstock.db')

# Test Stocks (Unseen during training)
TEST_STOCKS = ['2303', '2882', '2344', '2834', '1402', '2618']
TARGET_DAYS = 1

def get_all_stock_ids():
    conn = sqlite3.connect(DB_FULL_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Stock_ID FROM tw_stock_prices")
    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return [s for s in stocks if len(s) == 4]

def evaluate_model_performance(model):
    print("\n=== 1. Model Performance Evaluation (Unseen Stocks) ===")
    
    X_list, y_list = [], []
    
    for stock in TEST_STOCKS:
        try:
            df = prepare_training_data_v2(stock, target_days=TARGET_DAYS, path=DB_FULL_PATH)
            if df is not None and not df.empty:
                # Label Engineering (Same as training)
                mask_sig = (df['Future_Return'] > 0.005) | (df['Future_Return'] < -0.005)
                df = df[mask_sig].copy()
                df['Target'] = (df['Future_Return'] > 0).astype(int)
                
                drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target']
                X = df.drop(columns=[c for c in drop_cols if c in df.columns])
                X = X.select_dtypes(include=[np.number])
                y = df['Target']
                
                X_list.append(X)
                y_list.append(y)
        except Exception:
            pass
            
    if not X_list:
        print("No test data available.")
        return None

    X_test = pd.concat(X_list)
    y_test = pd.concat(y_list)
    
    # --- CRITICAL FIX: Handle Infinite Values ---
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # --- Threshold Analysis (Precision-Recall Curve) ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Find optimal threshold for F1 Score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\n--- Optimal Threshold Analysis ---")
    print(f"Best Threshold (Max F1): {best_threshold:.4f}")
    print(f"Precision at Best Threshold: {precisions[best_idx]:.4f}")
    print(f"Recall at Best Threshold: {recalls[best_idx]:.4f}")
    
    # Plot PR Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, marker='.', label='XGBoost')
    plt.scatter(recalls[best_idx], precisions[best_idx], marker='o', color='red', label='Best Threshold')
    plt.xlabel('Recall (Catching Opportunities)')
    plt.ylabel('Precision (Avoiding False Alarms)')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_threshold

def scan_market(model, threshold):
    print("\n=== 2. Market Scan: Predicting Tomorrow's Movers ===")
    
    all_stocks = get_all_stock_ids()
    print(f"Scanning {len(all_stocks)} stocks... (This may take a moment)")
    
    results = []
    
    for i, stock in enumerate(all_stocks):
        if i % 100 == 0: print(f"Processed {i} stocks...")
            
        try:
            # We only need the latest data point
            # But prepare_training_data calculates rolling windows, so we need history.
            # We load data, calculate features, and take the LAST row.
            df = prepare_training_data_v2(stock, target_days=TARGET_DAYS, path=DB_FULL_PATH, keep_raw_prices=True)
            
            if df is not None and not df.empty:
                last_row = df.iloc[[-1]] # Keep as DataFrame
                last_date = last_row.index[0]
                last_close = last_row['Close'].values[0]
                
                # Prepare Features
                drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target', 
                             'Open', 'High', 'Low', 'Close', 'Volume']
                X = last_row.drop(columns=[c for c in drop_cols if c in last_row.columns])
                X = X.select_dtypes(include=[np.number])
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Predict
                prob = model.predict_proba(X)[0, 1]
                
                results.append({
                    'Stock_ID': stock,
                    'Date': last_date,
                    'Close': last_close,
                    'Prob_Up': prob
                })
                
        except Exception:
            continue
            
    # Create DataFrame
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("No results found.")
        return

    # Filter by Threshold
    top_picks = res_df[res_df['Prob_Up'] > threshold].sort_values(by='Prob_Up', ascending=False)
    
    print(f"\n--- Top Picks (Prob > {threshold:.2f}) ---")
    print(top_picks.head(20))
    
    # Visualization
    if not top_picks.empty:
        plt.figure(figsize=(12, 6))
        top_10 = top_picks.head(10)
        sns.barplot(x='Prob_Up', y='Stock_ID', data=top_10, palette='viridis')
        plt.title(f"Top 10 Stocks with Highest Up Probability (Date: {top_10['Date'].iloc[0].date()})")
        plt.xlabel("Probability of Up")
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.legend()
        plt.show()
        
        # Save to CSV
        save_path = os.path.join(BASE_DIR, 'market_scan_results.csv')
        top_picks.to_csv(save_path, index=False)
        print(f"\nFull scan results saved to: {save_path}")

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return
        
    model = joblib.load(MODEL_PATH)
    
    # 1. Evaluate and find best threshold
    best_threshold = evaluate_model_performance(model)
    
    if best_threshold is None:
        best_threshold = 0.55 # Fallback
        
    # 2. Scan Market using that threshold
    scan_market(model, best_threshold)

if __name__ == "__main__":
    main()
