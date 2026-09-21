from ollama import AsyncClient
from fastapi import FastAPI, Body, HTTPException
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.qdrant import qdrant_startup, qdrant_stop
from app.session import init_db, get_state, save_state, r
from app.agent import llm_app
from app.schemas import QueryCheck

from app.logger import logger
from app.settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при запуске и очистка при завершении"""
    try:
        await qdrant_startup()
        logger.info('Qdrant загружен')
    except Exception as e:
        logger.exception(f'Qdrant не смог загрузиться: {e}')
        raise

    try:
        init_db()
        logger.info('База данных SQL создана')
    except Exception as e:
        logger.exception(f'Не удалось создать таблицу sql: {e}')
        raise
    yield  # <-- запуск приложения

    # --- Shutdown (опционально) ---
    # Например, закрытие соединений
    await qdrant_stop()

# Старт сервиса
app = FastAPI(title="RAG Service", lifespan=lifespan)

@app.post("/query")
async def query(request: QueryCheck):
    """
    RAG-эндпоинт: получает запрос → выполняет действие через LLM.
    """
    # Загрузка сессии
    try:
        state = get_state(request.user_id)
    except Exception as e:
        logger.exception(f'Не удалось загрузить сессию для {request.user_id}: {e}')
        raise 

    # Запрос к LLM
    try:
        result = await llm_app.ainvoke({
            **state,
            "messages": state['messages'] + [HumanMessage(content=request.query)]
        })
        response = result['messages'][-1]
        logger.info(f'Ответ LLM: {response}')
    except Exception as e:
        logger.exception(f'Не удалось получить запрос от LLM: {e}')
        raise

    # Сохранение сессии
    try:
        save_state(request.user_id, result)
    except Exception as e:
        logger.error(f'Не удалось сохранить сессию: {e}')
    return {'response': response}

@app.get("/health")
def health_check():
    return {'status': 'healthy'}