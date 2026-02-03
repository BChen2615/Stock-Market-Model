import sqlite3
import pandas as pd
import os
from datetime import datetime

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'twstock.db')

def check_freshness():
    print(f"--- Checking Database Freshness ({DB_PATH}) ---")
    
    if not os.path.exists(DB_PATH):
        print("Error: Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 1. Get Max Date for each stock
        query = """
            SELECT Stock_ID, MAX(Date) as Last_Date 
            FROM tw_stock_prices 
            GROUP BY Stock_ID
        """
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("Database is empty.")
            return

        df['Last_Date'] = pd.to_datetime(df['Last_Date'])
        
        # 2. Statistics
        total_stocks = len(df)
        print(f"Total Stocks in DB: {total_stocks}")
        
        # Group by Date
        date_counts = df['Last_Date'].value_counts().sort_index(ascending=False)
        
        print("\n--- Update Status by Date ---")
        print(date_counts.head(10)) # Show top 10 most recent dates
        
        # 3. Check for Laggards
        latest_date = df['Last_Date'].max()
        print(f"\nLatest Date in DB: {latest_date.date()}")
        
        # Stocks that are NOT on the latest date
        laggards = df[df['Last_Date'] < latest_date]
        print(f"Stocks NOT updated to latest date: {len(laggards)} ({len(laggards)/total_stocks:.1%})")
        
        if not laggards.empty:
            print("\nTop 10 Laggards (Oldest Data):")
            print(laggards.sort_values('Last_Date').head(10))
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_freshness()
