from ollama import AsyncClient
from app.utils import load_files
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from app.logger import logger
from app.settings import settings

async def qdrant_startup():
    global qdrant, ollama

    # Инициализация моделей и клиентов (один раз при старте)
    qdrant = AsyncQdrantClient(settings.qdrant_host, port=settings.qdrant_port)
    ollama = AsyncClient(host=settings.ollama_host)

    if await qdrant.collection_exists(settings.collection_name):
        await qdrant.delete_collection(settings.collection_name)
    
    await qdrant.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    try: 
        docs = load_files(['docs/agents.md', 'docs/fastapi.md', 'docs/rag.md'])
    except Exception as e:
        logger.error(f'Не удалось загрузить файлы: {e}')
        raise FileNotFoundError('Не удалось загрузить файлы')

    embeddings = await ollama.embed(model=settings.embedding_model, input=docs)
    points = [
        PointStruct(id=i, vector=emb, payload={"text": doc})
        for i, (doc, emb) in enumerate(zip(docs, embeddings['embeddings']))
    ]
    
    await qdrant.upsert(settings.collection_name, points)
    logger.info(f"✅ Коллекция '{settings.collection_name}' создана и заполнена.")

async def qdrant_stop():
    await qdrant.delete_collection(settings.collection_name)
    await qdrant.close()

async def qdrant_search(question):
    # 1. Эмбеддинг запроса
    embedding_response = await ollama.embed(model=settings.embedding_model, input=question)
    query_vector = embedding_response['embeddings'][0]

    # 2. Поиск в Qdrant (асинхронно)
    search_result = await qdrant.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=1
    )
    context = search_result.points[0].payload["text"]

    return context