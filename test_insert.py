from models import Document, SessionLocal

db = SessionLocal()
db.add(Document(content="Tell us about you"))
db.commit()



