"""Initialize the database with tables"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
from app.database.base import Base
from app.models import User, Chapter, ChatSession, PersonalizationSetting, Translation

async def init_db():
    """Create all tables in the database"""
    print(f"Creating database tables...")
    print(f"Database URL: {settings.DATABASE_URL}")

    engine = create_async_engine(
        str(settings.DATABASE_URL),
        echo=True,
    )

    async with engine.begin() as conn:
        # Drop all tables (use with caution in production!)
        # await conn.run_sync(Base.metadata.drop_all)

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("✅ Database tables created successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())
