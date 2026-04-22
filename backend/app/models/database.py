"""
SQLAlchemy database models
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime

from app.config import get_settings


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass


class Analysis(Base):
    """Model for storing skin analysis results."""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    image_hash = Column(String(64), index=True, nullable=True)
    disease = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    recommendations = Column(Text, nullable=True)
    next_steps = Column(Text, nullable=True)
    tips = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database engine and session
settings = get_settings()
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
