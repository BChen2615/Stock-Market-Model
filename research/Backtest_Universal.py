import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)

from Feature_Engineering_V2 import prepare_training_data_v2, DB_PATH

# --- CONFIG ---
BASE_DIR = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_universal.pkl')
DB_FULL_PATH = os.path.join(BASE_DIR, 'data', 'twstock.db')

INITIAL_CAPITAL = 1_000_000
TRANSACTION_FEE = 0.001425 # 0.1425%
TAX_RATE = 0.003 # 0.3%
CONFIDENCE_THRESHOLD = 0.60 # Only buy if prob > 0.55

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
    try:
        df = prepare_training_data_v2(stock_id, target_days=1, path=DB_FULL_PATH, keep_raw_prices=True)
        if df is None or df.empty:
            print("No data.")
            return None
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

    # 2. Predict
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
                cost = current_equity * TRANSACTION_FEE
                position = 1
                new_equity = (current_equity - cost) * (1 + ret)
            else:
                # Stay in Cash
                new_equity = current_equity
        else:
            if prob < 0.5: # Sell Signal
                # Sell Signal
                cost = current_equity * (TRANSACTION_FEE + TAX_RATE)
                position = 0
                new_equity = current_equity - cost
            else:
                # Hold
                new_equity = current_equity * (1 + ret)
        
        equity.append(new_equity)
        
        # Benchmark Logic
        bench_equity = benchmark_equity[-1] * (1 + ret)
        benchmark_equity.append(bench_equity)
        
    # Create Result DataFrame
    res_df = pd.DataFrame({
        'Date': dates,
        'Strategy': equity[:-1],
        'Benchmark': benchmark_equity[:-1],
        'Prob_Up': probs
    }).set_index('Date')
    
    # Calculate Stats
    total_return = (res_df['Strategy'].iloc[-1] / INITIAL_CAPITAL) - 1
    bench_return = (res_df['Benchmark'].iloc[-1] / INITIAL_CAPITAL) - 1
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Benchmark Return: {bench_return:.2%}")
    
    return res_df

def plot_interactive_results(results):
    for stock, df in results.items():
        # Create subplots: 2 rows (Equity, Probability)
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{stock} Equity Curve", "Model Confidence (Prob Up)"))

        # 1. Equity Curve (Top)
        # Normalize to 100
        strat_norm = df['Strategy'] / df['Strategy'].iloc[0] * 100
        bench_norm = df['Benchmark'] / df['Benchmark'].iloc[0] * 100
        
        fig.add_trace(go.Scatter(x=df.index, y=strat_norm, name='AI Strategy', line=dict(color='purple', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bench_norm, name='Buy & Hold', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        
        # 2. Probability (Bottom)
        fig.add_trace(go.Scatter(x=df.index, y=df['Prob_Up'], name='Prob Up', line=dict(color='blue', width=1), fill='tozeroy'), row=2, col=1)
        
        # Add Threshold Lines
        fig.add_hline(y=CONFIDENCE_THRESHOLD, line_dash="dot", line_color="green", annotation_text="Buy Threshold", row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Sell Threshold", row=2, col=1)
        
        # Layout
        final_ret = (strat_norm.iloc[-1] - 100)
        fig.update_layout(
            title=f"Backtest Result: {stock} (Total Return: {final_ret:.1f}%)",
            height=700,
            hovermode="x unified"
        )
        
        fig.show()

def main():
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run Train_Universal_Model.py first.")
        return
    
    model = joblib.load(MODEL_PATH)
    print("Universal Model Loaded.")
    
    # Pick Random Stocks
    all_stocks = get_all_stock_ids()
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
        plot_interactive_results(results)

if __name__ == "__main__":
    main()
