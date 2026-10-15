import asyncio
import app.qdrant as qd
from ollama import AsyncClient
from qdrant_client import AsyncQdrantClient
from app.settings import settings

QUESTION = "что такое программа?"
MARKER = "Что такое программа"   # кусок текста нужного чанка

async def main():
    qd.qdrant = AsyncQdrantClient(settings.qdrant_host, port=settings.qdrant_port,
                                  check_compatibility=False)
    qd.ollama = AsyncClient(host=settings.ollama_host)

    emb = await qd.ollama.embed(model=settings.embedding_model,
                                input=f"search_query: {QUESTION}")
    res = await qd.qdrant.query_points(settings.collection_name,
                                       query=emb["embeddings"][0],
                                       limit=15, with_payload=True)
    for rank, p in enumerate(res.points, 1):
        mark = "  <== ЦЕЛЬ" if MARKER in p.payload["text"] else ""
        print(f"{rank:2} {p.score:.3f} {p.payload['section'][:70]}{mark}")

asyncio.run(main())