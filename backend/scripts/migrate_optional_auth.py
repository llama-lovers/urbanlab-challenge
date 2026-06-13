"""
One-time migration: make chat_sessions.user_id nullable to support anonymous sessions.
Run once against an existing database:

    uv run python scripts/migrate_optional_auth.py
"""

import asyncio

from sqlalchemy import text

from app.database import engine


async def main() -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("ALTER TABLE chat_sessions ALTER COLUMN user_id DROP NOT NULL")
        )
    print("Done: chat_sessions.user_id is now nullable.")


asyncio.run(main())
