# semantic_search_azure.py
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pgvector.sqlalchemy import Vector
from openai import AzureOpenAI
import numpy as np
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
class AzureEmbeddingService:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",  # Use latest stable version
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        self.executor = ThreadPoolExecutor(max_workers=4)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with retry logic"""
        response = self.client.embeddings.create(
            input=text,
            model=self.deployment
        )
        return response.data[0].embedding

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Async wrapper for embedding generation"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.generate_embedding,
            text
        )

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectordb")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text)
    embedding = Column(Vector(1536))  # For text-embedding-3-small

Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()
embedding_service = AzureEmbeddingService()

@app.get("/documents/similar/")
async def find_similar_documents(query: str, limit: int = 5):
    """Find similar documents using vector search"""
    db = SessionLocal()
    try:
        # Generate query embedding
        embedding = await embedding_service.generate_embedding_async(query)
        
        # Vector similarity search
        similar_docs = db.query(Document)\
            .order_by(Document.embedding.cosine_distance(embedding))\
            .limit(limit)\
            .all()
        
        # Format response with similarity scores
        results = []
        for doc in similar_docs:
            similarity = 1 - np.dot(embedding, doc.embedding) / (np.linalg.norm(embedding) * np.linalg.norm(doc.embedding))
            results.append({
                "id": doc.id,
                "content": doc.content,
                "similarity_score": float(similarity)
            })
        
        return sorted(results, key=lambda x: x["similarity_score"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/documents/")
async def create_document(content: str):
    """Add a document with generated embedding"""
    db = SessionLocal()
    try:
        embedding = await embedding_service.generate_embedding_async(content)
        document = Document(content=content, embedding=embedding)
        db.add(document)
        db.commit()
        db.refresh(document)
        return {"id": document.id, "status": "created"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Test data loader
def load_test_data():
    test_documents = [
        "PostgreSQL is an advanced open-source relational database",
        "Vector embeddings enable semantic search capabilities",
        "Azure offers managed PostgreSQL with pgvector extension",
        "Azure OpenAI generates 1536-dimensional vectors",
        "Cosine similarity measures the angle between vectors",
        "HNSW indexing accelerates vector search operations",
        "FastAPI is a modern Python web framework",
        "SQLAlchemy provides ORM capabilities for Python",
        "Semantic search understands meaning beyond keywords",
        "Artificial intelligence transforms data analysis"
    ]
    
    db = SessionLocal()
    try:
        for content in test_documents:
            if not db.query(Document).filter(Document.content == content).first():
                embedding = embedding_service.generate_embedding(content)
                db.add(Document(content=content, embedding=embedding))
        db.commit()
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    
    # Load test data
    load_test_data()
    print("Test data loaded successfully")
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)