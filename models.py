# models.py
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL with validation
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please check your .env file")

# Configure engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text)
    embedding = Column(Vector(1536))  # For text-embedding-3-small

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    
    document_id = Column(Integer, primary_key=True)
    embedding = Column(Vector(1536))

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database tables created successfully")