import sqlite3
import pandas as pd
import bcrypt
import datetime
import os

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH_DB_PATH = os.path.join(BASE_DIR, 'data', 'users.db')

def init_auth_db():
    """Initialize the user and logs database."""
    if not os.path.exists(os.path.dirname(AUTH_DB_PATH)):
        os.makedirs(os.path.dirname(AUTH_DB_PATH))
        
    conn = sqlite3.connect(AUTH_DB_PATH)
    c = conn.cursor()
    
    # Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BLOB,
            created_at DATETIME
        )
    ''')
    
    # Logs Table (Analytics)
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT,
            details TEXT,
            timestamp DATETIME
        )
    ''')
    
    conn.commit()
    conn.close()

def register_user(username, password):
    """Register a new user with hashed password."""
    conn = sqlite3.connect(AUTH_DB_PATH)
    c = conn.cursor()
    
    # Check if exists
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if c.fetchone():
        conn.close()
        return False, "Username already exists."
    
    # Hash password
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        c.execute('INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)', 
                  (username, hashed, datetime.datetime.now()))
        conn.commit()
        conn.close()
        return True, "Registration successful! Please login."
    except Exception as e:
        conn.close()
        return False, f"Error: {e}"

def login_user(username, password):
    """Verify credentials."""
    conn = sqlite3.connect(AUTH_DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    
    if data:
        stored_hash = data[0]
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return True
    return False

def log_access(username, action, details=""):
    """Record user activity for analytics."""
    try:
        conn = sqlite3.connect(AUTH_DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO access_logs (username, action, details, timestamp) VALUES (?, ?, ?, ?)',
                  (username, action, details, datetime.datetime.now()))
        conn.commit()
        conn.close()
    except:
        pass # Don't crash app if logging fails

# Initialize on import
init_auth_db()
