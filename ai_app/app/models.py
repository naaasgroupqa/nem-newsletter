from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base

class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(32), default="manual")
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_text: Mapped[str] = mapped_column(Text)
    assistant_text: Mapped[str] = mapped_column(Text)
    used_web: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class ResearchTopic(Base):
    __tablename__ = "research_topics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
