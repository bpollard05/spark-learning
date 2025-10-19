# test_db.py
from sqlalchemy import create_engine, text

# Connect to database
engine = create_engine("postgresql://user:password@localhost:5432/learning_companion")

# Test connection
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM users"))
    count = result.scalar()
    print(f"✅ Database connected! Users table has {count} rows")
    
    # List all tables
    result = conn.execute(text("""
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public'
    """))
    tables = [row[0] for row in result]
    print(f"📊 Tables created: {', '.join(tables)}")