import asyncio
from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.services import rag_engine

async def main():
    async with AsyncSessionLocal() as db:
        res = await rag_engine.run(db, "what is this document about?", None)
        print(res)

asyncio.run(main())
