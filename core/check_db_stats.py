import sqlite3
import pandas as pd
import os

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'twstock.db')

def check_db_stats():
    print(f"--- Database Statistics ({DB_PATH}) ---")
    
    if not os.path.exists(DB_PATH):
        print("Error: Database file not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 1. Total Stocks
        query_stocks = "SELECT COUNT(DISTINCT Stock_ID) FROM tw_stock_prices"
        total_stocks = conn.execute(query_stocks).fetchone()[0]
        print(f"Total Stocks: {total_stocks}")
        
        # 2. Total Rows
        query_rows = "SELECT COUNT(*) FROM tw_stock_prices"
        total_rows = conn.execute(query_rows).fetchone()[0]
        print(f"Total Data Points: {total_rows}")
        
        # 3. Date Range
        query_date = "SELECT MIN(Date), MAX(Date) FROM tw_stock_prices"
        min_date, max_date = conn.execute(query_date).fetchone()
        print(f"Date Range: {min_date} to {max_date}")
        
        # 4. Data Freshness Check
        # Get the latest date for each stock
        print("\n--- Freshness Check ---")
        query_freshness = """
            SELECT Stock_ID, MAX(Date) as Last_Date, COUNT(*) as Count 
            FROM tw_stock_prices 
            GROUP BY Stock_ID
        """
        df_stats = pd.read_sql(query_freshness, conn)
        df_stats['Last_Date'] = pd.to_datetime(df_stats['Last_Date'])
        
        # Check how many are up-to-date (assuming max_date in DB is the target)
        # Note: max_date is a string from SQL, convert to datetime
        target_date = pd.to_datetime(max_date)
        
        up_to_date_count = (df_stats['Last_Date'] == target_date).sum()
        print(f"Stocks updated to {target_date.date()}: {up_to_date_count} ({up_to_date_count/total_stocks:.1%})")
        
        # Check for stale stocks (older than 7 days from target)
        stale_threshold = target_date - pd.Timedelta(days=7)
        stale_stocks = df_stats[df_stats['Last_Date'] < stale_threshold]
        print(f"Stale Stocks (Not updated in last 7 days): {len(stale_stocks)}")
        if not stale_stocks.empty:
            print(f"Sample Stale Stocks: {stale_stocks['Stock_ID'].head(5).tolist()}")
            
        # 5. Data Volume Check
        print("\n--- Data Volume Check ---")
        min_count = df_stats['Count'].min()
        max_count = df_stats['Count'].max()
        avg_count = df_stats['Count'].mean()
        print(f"Rows per Stock: Min={min_count}, Max={max_count}, Avg={avg_count:.0f}")
        
        # Stocks with too few data points (e.g., < 100 days)
        thin_stocks = df_stats[df_stats['Count'] < 100]
        print(f"Stocks with < 100 days of data: {len(thin_stocks)}")
        
        # 6. External Data Check
        print("\n--- External Data Check ---")
        try:
            query_ext = "SELECT Symbol, COUNT(*) FROM external_market_data GROUP BY Symbol"
            ext_stats = pd.read_sql(query_ext, conn)
            if not ext_stats.empty:
                print(ext_stats)
            else:
                print("External data table is empty.")
        except Exception:
            print("External data table does not exist.")

    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_db_stats()
