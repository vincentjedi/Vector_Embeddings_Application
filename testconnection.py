# test_connection.py
from models import engine
from sqlalchemy import text

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        print("Database connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")