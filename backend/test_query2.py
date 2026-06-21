import asyncio
from app.db.session import AsyncSessionLocal
from app.services import rag_engine

async def main():
    async with AsyncSessionLocal() as db:
        res = await rag_engine.run(db, "What is the capital of France?", None)
        print(res)

asyncio.run(main())
