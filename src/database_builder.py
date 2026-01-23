import twstock
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# --- CONFIG ---
TWDB_DIR = '../data/twstock.db'
START_DATE = "2020-01-01"  # 建議明確設定日期，"5y" 會隨時間變動
# 設定異常波動閾值 (例如單日漲跌超過 20% 視為可疑，台股正常限制是 10%，但考量除權息，設寬一點)
OUTLIER_THRESHOLD = 0.20

# --- DATABASE SETUP ---
conn = sqlite3.connect(TWDB_DIR)
cursor = conn.cursor()

# 建立主資料表
cursor.execute("""
               CREATE TABLE IF NOT EXISTS tw_stock_prices
               (
                   Date
                   DATETIME,
                   Stock_ID
                   TEXT,
                   Open
                   REAL,
                   High
                   REAL,
                   Low
                   REAL,
                   Close
                   REAL,
                   Volume
                   INTEGER,
                   Type
                   TEXT,
                   PRIMARY
                   KEY
               (
                   Date,
                   Stock_ID
               )
                   );
               """)

# 建立一個「錯誤日誌表」，用來記錄被過濾掉的異常數據，方便日後檢查
cursor.execute("""
               CREATE TABLE IF NOT EXISTS data_audit_log
               (
                   Date
                   DATETIME,
                   Stock_ID
                   TEXT,
                   Reason
                   TEXT,
                   Raw_Data
                   TEXT
               );
               """)
conn.commit()


# --- HELPER FUNCTION: 數據清洗與審查 ---
def clean_and_validate_data(df, stock_id):
    """
    清洗並驗證 DataFrame，返回乾淨的數據和異常數據列表。
    """
    if df.empty:
        return None, None

    # 1. 基礎清洗：移除索引並確保日期格式
    df = df.reset_index()
    if 'Date' not in df.columns:  # yfinance 有時 index 名稱不同
        return None, None
    df['Date'] = pd.to_datetime(df['Date'])

    # 2. 移除未來日期的數據 (YF 偶爾會有時區錯亂的未來數據)
    df = df[df['Date'] <= datetime.now()]

    # 3. 檢查缺失值 (Drop NaNs in OHLC)
    # 記錄下要被刪除的行以便 Log
    na_rows = df[df[['Open', 'High', 'Low', 'Close']].isna().any(axis=1)].copy()
    if not na_rows.empty:
        na_rows['Reason'] = 'Missing Values (NaN)'

    # 真正刪除
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    if df.empty: return None, na_rows

    # 4. 檢查價格邏輯 (Sanity Check)
    # 規則 A: 價格必須 > 0
    # 規則 B: High 必須是當日最高 (High >= Open, Close, Low)
    # 規則 C: Low 必須是當日最低 (Low <= Open, Close, High)

    mask_zero = (df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0) | (df['Close'] <= 0)
    mask_logic_error = (df['High'] < df['Low']) | (df['High'] < df['Open']) | (df['High'] < df['Close']) | (
                df['Low'] > df['Open']) | (df['Low'] > df['Close'])

    bad_data = df[mask_zero | mask_logic_error].copy()
    if not bad_data.empty:
        bad_data['Reason'] = 'Logic Error (Zero or H/L invalid)'

    # 過濾掉錯誤數據
    df = df[~(mask_zero | mask_logic_error)].copy()

    # 5. 異常波動偵測 (Outlier Detection)
    # 雖然不一定要刪除，但我們可以標記。
    # 計算單日漲跌幅
    df['pct_change'] = df['Close'].pct_change().abs()

    # 如果單日波動超過 50% (極端異常)，通常是數據錯誤 (台股即使除權息也很少單日腰斬)
    # 這裡我們保守一點，只記錄「極度異常」的 glitch
    mask_extreme = df['pct_change'] > 0.5
    extreme_data = df[mask_extreme].copy()
    if not extreme_data.empty:
        extreme_data['Reason'] = 'Extreme Volatility (>50%)'
        # 選擇性：你可以決定要不要刪除這些資料，這裡示範刪除
        df = df[~mask_extreme]

    # 合併所有的壞數據準備寫入 Log
    audit_logs = pd.concat([na_rows, bad_data, extreme_data]) if 'na_rows' in locals() else None

    # 清理暫存欄位
    if 'pct_change' in df.columns:
        df = df.drop(columns=['pct_change'])

    return df, audit_logs


# --- MAIN LOOP ---

def process_stocks(stock_list, stock_type):
    print(f"Starting processing for {stock_type} stocks...")
    total = len(stock_list)

    for i, code in enumerate(stock_list):
        if len(code) != 4: continue

        # 進度條顯示
        if i % 10 == 0:
            print(f"Processing {code}.{stock_type} ({i}/{total})...")

        try:
            # 下載數據
            ticker = f"{code}.{stock_type}"
            df = yf.download(ticker, period="5y", auto_adjust=False, multi_level_index=False, progress=False)

            # --- 核心修改：加入審查機制 ---
            clean_df, error_log = clean_and_validate_data(df, code)

            # 如果有數據，寫入資料庫
            if clean_df is not None and not clean_df.empty:
                clean_df["Type"] = stock_type
                clean_df["Stock_ID"] = code

                # 移除不必要的欄位 (Adj Close 已經在 yf 參數處理，但保險起見)
                cols_to_keep = ["Date", "Stock_ID", "Open", "High", "Low", "Close", "Volume", "Type"]
                # 確保欄位存在再選取
                clean_df = clean_df[[c for c in cols_to_keep if c in clean_df.columns]]

                # 寫入 DB (使用 try-except 避免重複 Primary Key 報錯)
                try:
                    clean_df.to_sql("tw_stock_prices", conn, if_exists="append", index=False)
                except sqlite3.IntegrityError:
                    # 如果資料已經存在，這裡選擇忽略，或者你可以改成 update
                    # print(f"Data for {code} already exists, skipping duplicates.")
                    pass

            # 如果有異常數據，寫入 Log 表
            if error_log is not None and not error_log.empty:
                # 把整個 row 轉成 string 方便存儲
                error_log['Raw_Data'] = error_log.apply(lambda x: str(x.to_dict()), axis=1)
                error_log['Stock_ID'] = code
                error_log[['Date', 'Stock_ID', 'Reason', 'Raw_Data']].to_sql("data_audit_log", conn, if_exists="append",
                                                                             index=False)

        except Exception as e:
            print(f"Error processing {code}: {e}")


# 執行 TW (上市)
TW_CODES = [c for c in twstock.twse.keys() if len(c) == 4]
process_stocks(TW_CODES, "TW")

# 執行 TWO (上櫃)
TWO_CODES = [c for c in twstock.tpex.keys() if len(c) == 4]
process_stocks(TWO_CODES, "TWO")

conn.close()
print("Database build complete.")