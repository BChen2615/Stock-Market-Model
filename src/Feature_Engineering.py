import sqlite3
import pandas as pd
import numpy as np

# --- CONFIG ---
DB_PATH = '../data/twstock.db'

def load_data(stock_id, db_path=DB_PATH):
    """
    Load stock data from SQLite database for a specific stock ID.
    """
    conn = sqlite3.connect(db_path)
    # Select all columns
    query = f"SELECT * FROM tw_stock_prices WHERE Stock_ID = '{stock_id}' ORDER BY Date"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        return df
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

def add_technical_indicators(df):
    """
    Add comprehensive technical indicators to the DataFrame.
    Includes Trend, Momentum, Volatility, and Volume indicators.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # --- 1. Trend Indicators ---
    
    # Simple Moving Averages (SMA)
    # Short-term: 5, 10; Medium-term: 20; Long-term: 60
    for window in [5, 10, 20, 60]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
    # Exponential Moving Averages (EMA)
    # Common periods: 12, 26 (used for MACD)
    for window in [12, 26]:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # --- 2. Momentum Indicators ---
    
    # RSI (Relative Strength Index) - 14 days
    # Using Wilder's Smoothing (alpha = 1/n) which is ewm(com=n-1)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator (KD)
    # K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    # D = 3-day SMA of K
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # ROC (Rate of Change) - 12 days
    df['ROC'] = df['Close'].pct_change(periods=12) * 100
    
    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    
    # --- 3. Volatility Indicators ---
    
    # Bollinger Bands (20 days, 2 std dev)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma_20
    
    # ATR (Average True Range) - 14 days
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # --- 4. Volume Indicators ---
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # --- 5. Pattern Features (New) ---

    # Bullish Alignment (SMA 5 > 10 > 20 > 60)
    # Returns 1 if True, 0 if False
    df['Bullish_Alignment'] = (
        (df['SMA_5'] > df['SMA_10']) & 
        (df['SMA_10'] > df['SMA_20']) & 
        (df['SMA_20'] > df['SMA_60'])
    ).astype(int)

    # MA Entanglement (Standard Deviation of SMAs / Close Price)
    # Lower value means lines are closer together (entangled)
    ma_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_60']
    # Calculate std dev across columns for each row
    ma_std = df[ma_cols].std(axis=1)
    # Normalize by price so it's comparable across different price levels
    df['MA_Entanglement'] = ma_std / df['Close']
    
    return df

def add_lag_features(df, lags=20):
    """
    Add lag features for Close price, Volume, and Returns.
    This creates columns for t-1 to t-20.
    """
    df = df.copy()
    
    # Calculate Daily Return first as it's a normalized feature
    df['Return'] = df['Close'].pct_change()
    
    for lag in range(1, lags + 1):
        # Price Lags
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        # Volume Lags
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        # Return Lags
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
        
    return df

def prepare_training_data(stock_id, target_days=3, path=DB_PATH):
    """
    Full pipeline to generate features and prepare (X, y) for training.
    
    Args:
        stock_id (str): The stock ID to process.
        target_days (int): The number of days to look ahead for the target (default 3).
        path (str): Path to the database.
                            
    Returns:
        df_final (pd.DataFrame): The dataframe with features and target.
                                 Rows with NaN are dropped.
    """
    # 1. Load Data
    df = load_data(stock_id, path)
    
    if df.empty:
        print(f"No data found for {stock_id}")
        return None

    # 2. Add Technical Indicators
    df = add_technical_indicators(df)
    
    # 3. Add Lag Features (0-20 days history context)
    df = add_lag_features(df, lags=20)
    
    # 4. Create Target: Future N-Day Cumulative Return
    # Formula: (Close[t+N] - Close[t]) / Close[t]
    # We use shift(-N) to bring future price to current row
    future_close = df['Close'].shift(-target_days)
    df['Future_Return'] = (future_close - df['Close']) / df['Close']
    
    # NOTE: We do NOT create the 'Target' classification column here anymore.
    # We leave 'Future_Return' as is, so the training script can define its own labels.
    
    # 5. Drop NaN values
    # We lose initial rows due to rolling windows (max 60) and lags (20)
    # We lose final rows due to target shift (target_days)
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Example usage for testing
    # Assuming '2330' (TSMC) exists in the DB
    stock_code = "2330" 
    print(f"Generating features for {stock_code}...")
    
    try:
        data = prepare_training_data(stock_code, target_days=3)
        if data is not None and not data.empty:
            print("Feature generation successful.")
            print(f"Data shape: {data.shape}")
            print("Columns:", data.columns.tolist())
            print("\nLast 5 rows:")
            print(data[['Close', 'Future_Return', 'Bullish_Alignment']].tail())
        else:
            print("No data available or not enough data to generate features.")
    except Exception as e:
        print(f"An error occurred: {e}")
