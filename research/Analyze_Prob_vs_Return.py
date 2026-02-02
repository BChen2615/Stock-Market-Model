import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)

from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
BASE_DIR = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_universal_14d.pkl')
DB_FULL_PATH = os.path.join(BASE_DIR, 'data', 'twstock.db')

# Use the same test stocks as before to ensure out-of-sample validity
TEST_STOCKS = ['2303', '2882', '2344', '2834', '1402', '2618', '2603', '2330'] # Added a few more for more data points
TARGET_DAYS = 1

def analyze_prob_vs_return():
    print("--- Analyzing Prediction Probability vs Actual Return ---")
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    model = joblib.load(MODEL_PATH)
    
    all_probs = []
    all_returns = []
    
    print("Collecting data from test stocks...")
    for stock in TEST_STOCKS:
        try:
            # Keep raw prices=True just to be safe, though we only need Future_Return
            df = prepare_training_data_v2(stock, target_days=TARGET_DAYS, path=DB_FULL_PATH, keep_raw_prices=True)
            
            if df is not None and not df.empty:
                # Prepare Features
                drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target', 
                             'Open', 'High', 'Low', 'Close', 'Volume']
                X = df.drop(columns=[c for c in drop_cols if c in df.columns])
                X = X.select_dtypes(include=[np.number])
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Predict Probabilities
                probs = model.predict_proba(X)[:, 1]
                returns = df['Future_Return'].values
                
                all_probs.extend(probs)
                all_returns.extend(returns)
                
        except Exception as e:
            print(f"Skipping {stock}: {e}")
            
    if not all_probs:
        print("No data collected.")
        return

    # Create Analysis DataFrame
    df_res = pd.DataFrame({
        'Probability': all_probs,
        'Actual_Return': all_returns
    })
    
    # Create Bins (e.g., 0.0-0.1, 0.1-0.2, ...)
    # We use 10 bins (Deciles)
    df_res['Prob_Bin'] = pd.cut(df_res['Probability'], bins=np.linspace(0, 1, 11), labels=False)
    
    # Group by Bin
    bin_stats = df_res.groupby('Prob_Bin')['Actual_Return'].agg(['mean', 'count', 'std'])
    bin_stats['mean'] = bin_stats['mean'] * 100 # Convert to %
    
    # Rename index for plotting
    bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    # Ensure we have all bins even if empty
    bin_stats = bin_stats.reindex(range(10), fill_value=0)
    bin_stats.index = bin_labels
    
    print("\n--- Bin Statistics ---")
    print(bin_stats)
    
    # --- Visualization ---
    plt.figure(figsize=(12, 6))
    
    # Bar chart for Average Return
    bars = plt.bar(bin_stats.index, bin_stats['mean'], color='skyblue', edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom' if height > 0 else 'top')
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel('Model Prediction Probability (Confidence)')
    plt.ylabel('Average Next-Day Return (%)')
    plt.title('Does Higher Confidence Lead to Higher Returns?')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add sample count on secondary axis (optional, but good to know)
    # For simplicity, just printing counts in console is enough, or add as text
    
    plt.show()
    
    # Scatter Plot (for granular view)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Probability', y='Actual_Return', data=df_res, alpha=0.3, s=10)
    # Add trend line
    sns.regplot(x='Probability', y='Actual_Return', data=df_res, scatter=False, color='red')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Actual Return')
    plt.title('Scatter Plot: Probability vs Return')
    plt.ylim(-0.1, 0.1) # Limit y-axis to focus on normal moves
    plt.show()

if __name__ == "__main__":
    analyze_prob_vs_return()
