from ollama import AsyncClient
from app.utils import load_files
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.chunking import load_chunks

from app.logger import logger
from app.settings import settings

async def qdrant_startup() -> None:
    '''Загрузка чанков в qdrant'''
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

    # Загрузка чанков
    try: 
        chunks = load_chunks('docs')
    except Exception as e:
        logger.error(f'Не удалось загрузить файлы: {e}')
        raise FileNotFoundError('Не удалось загрузить файлы')
    
    logger.info(f"Сделано {len(chunks)} чанков")

    embeddings = await ollama.embed(model=settings.embedding_model, input=[f"search document: {c['section']}\n{c['text']}" for c in chunks])
    points = [
        PointStruct(id=c['id'],
                    vector=emb,
                    payload={'text': c['text'], 'source': c['source'], 'section': c['section']})
        for c, emb in enumerate(zip(chunks, embeddings['embeddings']))
    ]
    
    await qdrant.upsert(settings.collection_name, points)
    logger.info(f"✅ Коллекция '{settings.collection_name}' создана и заполнена.")

async def qdrant_stop():
    await qdrant.delete_collection(settings.collection_name)
    await qdrant.close()

async def qdrant_search(question) -> tuple[str, list[dict]]:
    # 1. Эмбеддинг запроса
    embedding_response = await ollama.embed(model=settings.embedding_model, input=f'search_query: {question}')
    query_vector = embedding_response['embeddings'][0]

    # 2. Поиск в Qdrant (асинхронно)
    search_result = await qdrant.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=3,
        with_payload=True
    )
    hits = [
        {
            "text": p.payload["text"],
            "source": p.payload["source"],
            "section": p.payload["section"],
            "score": p.score,
        }
        for p in search_result.points
    ]

    context = "\n\n---\n\n".join(
        f"[{h['source']} > {h['section']}]\n{h['text']}" for h in hits
    )
    return context, hits