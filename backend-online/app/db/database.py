from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

from app.core.config import settings

# Create declarative base for models
Base = declarative_base()

# Database initialization and cleanup functions
async def init_db():
    """Initialize database connections and perform startup tasks"""
    # In production, we don't create tables here - use Alembic migrations instead
    # This function is primarily for establishing connections and initialization
    global engine
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        future=True,
        poolclass=NullPool  # Using NullPool for better control over connection lifecycle
    )
    # Create async session factory
    global async_session
    async_session = sessionmaker(
        engine, 
        class_=AsyncSession, 
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )

async def close_db():
    """Close database connections and perform cleanup tasks"""
    # Ensure all connections are properly closed
    await engine.dispose()

# Dependency for getting DB session
async def get_db():
    """Dependency for getting async database session"""
    session = async_session()
    try:
        yield session
    finally:
        await session.close()
