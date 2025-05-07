# worker.py
import asyncio
from models import SessionLocal, Document, DocumentEmbedding
from embeddings import EmbeddingService
from sqlalchemy import and_, not_
from concurrent.futures import ThreadPoolExecutor

embedding_service = EmbeddingService()
executor = ThreadPoolExecutor(max_workers=4)

async def process_pending_embeddings(batch_size: int = 50):
    db = SessionLocal()
    try:
        # Updated query that matches your schema
        pending_docs = db.query(Document)\
            .outerjoin(DocumentEmbedding, Document.id == DocumentEmbedding.document_id)\
            .filter(
                DocumentEmbedding.document_id.is_(None),
                Document.content.isnot(None)  # Use the correct column name here
            )\
            .limit(batch_size)\
            .all()
        
        if not pending_docs:
            return 0
        
        
        # Batch generate embeddings
        texts = [doc.content for doc in pending_docs]
        embeddings = await asyncio.get_event_loop().run_in_executor(
            executor,
            embedding_service.batch_generate_embeddings,
            texts
        )
        
        # Store embeddings
        for doc, embedding in zip(pending_docs, embeddings):
            db.add(DocumentEmbedding(
                document_id=doc.id,
                embedding=embedding
            ))
            doc.embedding = embedding  # Update main document too
        
        db.commit()
        return len(pending_docs)
    except Exception as e:
        db.rollback()
        print(f"Error processing embeddings: {e}")
        return 0
    finally:
        db.close()

async def run_periodically(interval: int = 60):
    """Run the worker periodically"""
    while True:
        processed = await process_pending_embeddings()
        print(f"Processed {processed} documents")
        await asyncio.sleep(interval)

if __name__ == "__main__":
    asyncio.run(run_periodically())