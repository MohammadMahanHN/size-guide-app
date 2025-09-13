import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from a .env file (for local development)
load_dotenv()

# Get the database URL from environment variables
# For Render, this will be the internal PostgreSQL URL
# For local dev, it will fall back to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./decision_tree.db")

# If using PostgreSQL, the engine needs no extra arguments
if DATABASE_URL.startswith("postgres"):
    engine = create_engine(DATABASE_URL)
else:
    # SQLite requires connect_args
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
