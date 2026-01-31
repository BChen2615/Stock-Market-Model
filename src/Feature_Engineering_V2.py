import sqlite3
import pandas as pd
import numpy as np

# --- CONFIG ---
DB_PATH = '../data/twstock.db'

def load_data(stock_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM tw_stock_prices WHERE Stock_ID = '{stock_id}' ORDER BY Date"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        return df
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

def load_external_data(db_path=DB_PATH):
    """
    Loads external market data (TSM ADR, SOX, etc.) from DB.
    Returns a pivoted DataFrame with Date index and columns like 'TSM_ADR_Close', 'SOX_Index_Close'.
    """
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT * FROM external_market_data"
        df = pd.read_sql(query, conn)
    except Exception:
        # Table might not exist yet
        conn.close()
        return pd.DataFrame()
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot to get columns for each symbol
    # We only care about Close and Volume
    df_pivot = df.pivot(index='Date', columns='Symbol', values=['Close', 'Volume'])
    
    # Flatten columns: e.g., ('Close', 'TSM_ADR') -> 'TSM_ADR_Close'
    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]
    
    return df_pivot

def add_stationary_features(df):
    """
    Generates ONLY stationary features (ratios, percentages).
    No absolute prices allowed.
    """
    df = df.copy()
    
    # 1. Basic Price Changes (Normalized)
    # Instead of Raw Open/High/Low, use their relation to Prev Close
    prev_close = df['Close'].shift(1)
    
    df['Open_Gap'] = (df['Open'] - prev_close) / prev_close
    df['High_Chg'] = (df['High'] - prev_close) / prev_close
    df['Low_Chg'] = (df['Low'] - prev_close) / prev_close
    df['Close_Chg'] = (df['Close'] - prev_close) / prev_close # This is Daily Return
    
    # Volume Change
    df['Vol_Chg'] = df['Volume'].pct_change()
    
    # 2. Moving Average Bias (乖离率)
    # Close / SMA - 1
    for window in [5, 10, 20, 60]:
        sma = df['Close'].rolling(window=window).mean()
        df[f'Bias_{window}'] = (df['Close'] / sma) - 1
        
    # 3. Volatility (Normalized)
    # ATR / Close (Relative ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=14).mean()
    df['NATR'] = atr / df['Close'] # Normalized ATR
    
    # 4. Momentum (Oscillators are naturally stationary)
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, min_periods=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (We use the Histogram, which is stationary-ish)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    # Normalize MACD by price to make it comparable across years
    df['MACD_Hist_Norm'] = (macd - signal) / df['Close']
    
    # 5. Interaction Features
    # Price Range (High - Low) / Close
    df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Close']
    
    # Close location within High-Low range (Stochastic K concept for single day)
    # 1.0 = Closed at High, 0.0 = Closed at Low
    df['Close_Loc'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    return df

def add_external_features(df, df_ext):
    """
    Merges external data and calculates stationary features (daily returns).
    """
    if df_ext.empty:
        return df
        
    # Merge on Date
    # Note: df index is Date, df_ext index is Date
    df = df.join(df_ext, how='left')
    
    # Calculate Returns for External Data
    # We only care about the Change %, not absolute levels
    ext_cols = [c for c in df_ext.columns if 'Close' in c]
    
    for col in ext_cols:
        # Daily Return
        df[f'{col}_Chg'] = df[col].pct_change()
        
        # Drop the absolute price column immediately
        df = df.drop(columns=[col])
        
    # Drop Volume columns from external data (usually less useful than price)
    vol_cols = [c for c in df.columns if 'Volume' in c and c != 'Volume'] # Keep original Volume
    df = df.drop(columns=vol_cols)
    
    return df

def add_lag_features_v2(df, lags=5):
    """
    Add lags of ONLY the stationary features.
    """
    df = df.copy()
    
    # Features to lag
    # Include the new External Change columns
    ext_chg_cols = [c for c in df.columns if '_Chg' in c and c != 'Close_Chg'] # Close_Chg is already in list below
    
    cols_to_lag = ['Close_Chg', 'Vol_Chg', 'RSI', 'MACD_Hist_Norm', 'Bias_5', 'Bias_20'] + ext_chg_cols
    
    for col in cols_to_lag:
        if col not in df.columns: continue
        
        for lag in range(1, lags + 1):
            df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
            
    return df

def prepare_training_data_v2(stock_id, target_days=1, path=DB_PATH, keep_raw_prices=False):
    # 1. Load Stock Data
    df = load_data(stock_id, path)
    if df.empty: return None
    
    # 2. Load External Data
    df_ext = load_external_data(path)
    
    # 3. Add Features (Strictly Stationary)
    df = add_stationary_features(df)
    
    # 4. Add External Features
    df = add_external_features(df, df_ext)
    
    # 5. Add Lags (Short memory is enough for day trading)
    df = add_lag_features_v2(df, lags=5)
    
    # 6. Create Target
    # Target: Next Day's Return
    future_close = df['Close'].shift(-target_days)
    df['Future_Return'] = (future_close - df['Close']) / df['Close']
    
    # 7. Clean up
    # Drop raw price columns to prevent leakage/overfitting on absolute levels
    # UNLESS keep_raw_prices is True (for backtesting)
    if not keep_raw_prices:
        raw_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Type', 'Stock_ID']
        df = df.drop(columns=[c for c in raw_cols if c in df.columns])
    
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Test
    data = prepare_training_data_v2('2330')
    if data is not None:
        print("V2 Feature Engineering Successful.")
        print(f"Shape: {data.shape}")
        print("Columns:", data.columns.tolist())
        print(data.head())
