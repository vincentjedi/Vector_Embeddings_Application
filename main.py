# main.py
from fastapi import FastAPI, HTTPException
from models import SessionLocal, Document, DocumentEmbedding
from embeddings import EmbeddingService
from typing import List
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
embedding_service = EmbeddingService()
executor = ThreadPoolExecutor(max_workers=4)

class DocumentCreate(BaseModel):
    content: str

class DocumentResponse(BaseModel):
    id: int
    content: str

@app.post("/documents/", response_model=DocumentResponse)
async def create_document(doc: DocumentCreate):
    """Create document and generate embedding in one transaction"""
    db = SessionLocal()
    try:
        # Generate embedding asynchronously
        embedding = await asyncio.get_event_loop().run_in_executor(
            executor,
            embedding_service.generate_embedding,
            doc.content
        )
        
        # Store document and embedding
        document = Document(content=doc.content, embedding=embedding)
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Also store in separate table
        db.add(DocumentEmbedding(
            document_id=document.id,
            embedding=embedding
        ))
        db.commit()
        
        return document
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/documents/batch/")
async def create_documents_batch(docs: List[DocumentCreate]):
    """Batch processing for better performance"""
    db = SessionLocal()
    try:
        # Extract texts for batch embedding
        texts = [doc.content for doc in docs]
        
        # Generate all embeddings in one API call
        embeddings = await asyncio.get_event_loop().run_in_executor(
            executor,
            embedding_service.batch_generate_embeddings,
            texts
        )
        
        # Create documents with embeddings
        documents = []
        for doc, embedding in zip(docs, embeddings):
            document = Document(content=doc.content, embedding=embedding)
            db.add(document)
            documents.append(document)
        
        db.commit()
        
        # Refresh to get IDs
        for doc in documents:
            db.refresh(doc)
        
        # Store in embeddings table
        for doc in documents:
            db.add(DocumentEmbedding(
                document_id=doc.id,
                embedding=doc.embedding
            ))
        
        db.commit()
        return {"message": f"Successfully created {len(documents)} documents"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/documents/similar/")
async def find_similar_documents(query: str, limit: int = 5):
    """Find similar documents using vector search"""
    db = SessionLocal()
    try:
        # Generate query embedding
        embedding = await asyncio.get_event_loop().run_in_executor(
            executor,
            embedding_service.generate_embedding,
            query
        )
        
        # Vector similarity search
        similar_docs = db.query(Document)\
            .order_by(Document.embedding.cosine_distance(embedding))\
            .limit(limit)\
            .all()
        
        return similar_docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()