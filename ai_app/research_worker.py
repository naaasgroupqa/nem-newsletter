from datetime import datetime
from app.db import Base, engine, SessionLocal
from app.models import ResearchTopic
from app.ai import answer, auto_extract_learnings

Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    topics=db.query(ResearchTopic).filter(ResearchTopic.active == True).all()
    for t in topics:
        q=f"Research the latest reliable updates about: {t.topic}. Focus on durable facts and meaningful changes."
        text, sources = answer(db, q, use_web=True)
        auto_extract_learnings(db, q, text, sources)
        t.last_run_at=datetime.utcnow(); db.commit()
        print(f"updated: {t.topic}")
