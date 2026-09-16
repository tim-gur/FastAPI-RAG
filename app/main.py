from ollama import AsyncClient
from fastapi import FastAPI, Body
from contextlib import asynccontextmanager
import logging
from qdrant import qdrant_startup, qdrant_stop
from session import get_state, save_state
from agent import llm_app
from langchain_core.messages import HumanMessage
from logging import setup_logging
from pydantic import BaseModel

# логирование
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при запуске и очистка при завершении"""
    await qdrant_startup()

    yield  # <-- запуск приложения

    # --- Shutdown (опционально) ---
    # Например, закрытие соединений
    await qdrant_stop()

# Старт сервиса
app = FastAPI(title="RAG Service", lifespan=lifespan)

class QueryCheck(BaseModel):
    user_id: int
    query: str

@app.post("/ask")
async def ask(request: QueryCheck):
    """
    RAG-эндпоинт: получает запрос → выполняет действие через LLM.
    """
    state = get_state(request.user_id)

    result = await llm_app.ainvoke({
        **state,
        "messages": [HumanMessage(content=request.query)]
    })

    save_state(request.user_id, result)

    return {'response': result['messages'][-1]}

@app.get("/health")
async def health_check():
    try:
        # Тестовый запрос к LLM
        test_response = await llm_app.chat({
            "messages":[HumanMessage(content="Верни {'task_key': 'TEST-1', 'summary': 'тест'}")]
        })
        # Пытаемся валидировать
        QueryCheck.model_validate_json(test_response.message)
        return {"status": "healthy"}
    except Exception as e:
        logger.exception(f'FastAPI error: {e}')
        return {"status": "unhealthy", "error": str(e)}