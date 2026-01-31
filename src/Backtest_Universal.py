import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sqlite3
from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_universal.pkl')
DB_FULL_PATH = os.path.join(BASE_DIR, 'data', 'twstock.db')

INITIAL_CAPITAL = 100000
TRANSACTION_FEE = 0.001425 # 0.1425%
TAX_RATE = 0.003 # 0.3%
CONFIDENCE_THRESHOLD = 0.70 # Only buy if prob > 0.55

def get_all_stock_ids():
    conn = sqlite3.connect(DB_FULL_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Stock_ID FROM tw_stock_prices")
    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return stocks

def run_backtest(stock_id, model):
    print(f"\n--- Backtesting {stock_id} ---")
    
    # 1. Prepare Data
    # We need the raw price data for simulation, and features for prediction
    try:
        df = prepare_training_data_v2(stock_id, target_days=1, path=DB_FULL_PATH, keep_raw_prices=True)
        if df is None or df.empty:
            print("No data.")
            return None
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

    # 2. Predict
    # Drop non-feature columns AND raw price columns that were kept for backtesting logic
    drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 'Target', 
                 'Open', 'High', 'Low', 'Close', 'Volume']
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    
    # Handle infs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    probs = model.predict_proba(X)[:, 1] # Prob of Up
    
    # 3. Simulation Loop
    dates = df.index
    returns = df['Future_Return'].values
    
    equity = [INITIAL_CAPITAL]
    position = 0 # 0: Cash, 1: Held
    
    # Benchmark (Buy & Hold)
    benchmark_equity = [INITIAL_CAPITAL]
    
    for i in range(len(probs)):
        current_equity = equity[-1]
        prob = probs[i]
        ret = returns[i] # Return of the next day
        
        # Strategy Logic
        if position == 0:
            if prob > CONFIDENCE_THRESHOLD:
                # Buy Signal
                # Cost: Fee
                cost = current_equity * TRANSACTION_FEE
                position = 1
                # We enter the market. The change in value will be (1 + ret)
                # But we pay fee upfront.
                new_equity = (current_equity - cost) * (1 + ret)
            else:
                # Stay in Cash
                new_equity = current_equity
        else:
            if prob < 0.5: # Sell Signal (Weak condition, can be tuned)
                # Sell Signal
                # Cost: Fee + Tax
                cost = current_equity * (TRANSACTION_FEE + TAX_RATE)
                position = 0
                new_equity = current_equity - cost
            else:
                # Hold
                new_equity = current_equity * (1 + ret)
        
        equity.append(new_equity)
        
        # Benchmark Logic
        # Always invested
        bench_equity = benchmark_equity[-1] * (1 + ret)
        benchmark_equity.append(bench_equity)
        
    # Create Result DataFrame
    res_df = pd.DataFrame({
        'Date': dates,
        'Strategy': equity[:-1], # Align length
        'Benchmark': benchmark_equity[:-1]
    }).set_index('Date')
    
    # Calculate Stats
    total_return = (res_df['Strategy'].iloc[-1] / INITIAL_CAPITAL) - 1
    bench_return = (res_df['Benchmark'].iloc[-1] / INITIAL_CAPITAL) - 1
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Benchmark Return: {bench_return:.2%}")
    
    return res_df

def main():
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run Train_Universal_Model.py first.")
        return
    
    model = joblib.load(MODEL_PATH)
    print("Universal Model Loaded.")
    
    # Pick Random Stocks
    all_stocks = get_all_stock_ids()
    # Filter out some known bad data or indices if any
    all_stocks = [s for s in all_stocks if len(s) == 4] 
    
    if len(all_stocks) < 4:
        print("Not enough stocks in DB.")
        return
        
    selected_stocks = random.sample(all_stocks, 4)
    print(f"Selected Stocks for Backtest: {selected_stocks}")
    
    # Run Backtests
    results = {}
    for stock in selected_stocks:
        res = run_backtest(stock, model)
        if res is not None:
            results[stock] = res
            
    # Visualization
    if results:
        plt.figure(figsize=(15, 10))
        
        for i, (stock, df) in enumerate(results.items()):
            plt.subplot(2, 2, i+1)
            
            # Normalize to 100 for comparison
            strat_norm = df['Strategy'] / df['Strategy'].iloc[0] * 100
            bench_norm = df['Benchmark'] / df['Benchmark'].iloc[0] * 100
            
            plt.plot(strat_norm, label='AI Strategy', color='purple', linewidth=2)
            plt.plot(bench_norm, label='Buy & Hold', color='gray', alpha=0.5)
            
            final_ret = (strat_norm.iloc[-1] - 100)
            plt.title(f"{stock} (Return: {final_ret:.1f}%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
