import asyncio
from app.db.session import AsyncSessionLocal
from app.services import rag_engine, embedder, reranker

embedder.initialise()
reranker.initialise()

async def main():
    async with AsyncSessionLocal() as db:
        try:
            res = await rag_engine.run(db, "What is the capital of France?", None)
            print(res)
        except Exception as e:
            import traceback
            traceback.print_exc()

asyncio.run(main())
