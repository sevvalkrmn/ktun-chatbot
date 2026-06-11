"""
OAuth token deposu için SQLAlchemy async motoru (varsayılan: SQLite).

.env içinde DATABASE_URL ile PostgreSQL'e geçilebilir:
  SQLite (varsayılan): sqlite+aiosqlite:///./tokens.db
  PostgreSQL          : postgresql+asyncpg://user:pass@host:5432/db
"""
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./tokens.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Tabloları oluşturur (uygulama açılışında çağrılır)."""
    import db_models  # noqa: F401  — modelleri Base'e kaydetmek için
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
