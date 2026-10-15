import asyncio
from openai import AsyncOpenAI
import sys
import types
from pathlib import Path
import json
# ragas 0.4.3 imports a module that langchain-community 0.4 removed.
# A stub is enough: it's only referenced for Vertex AI model checks.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRelevance
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import AsyncQdrantClient
from ollama import AsyncClient
from pydantic import BaseModel

from app.settings import settings
from app.logger import logger
from app.qdrant import qdrant_search
import app.qdrant as qd
from app.main import qdrant_startup

gen_llm = ChatOllama(
    model=settings.llm_model,
    temperature=0,
    base_url=settings.ollama_host
)

async def answer(question: str) -> dict:
    '''Ответ LLM с контекстом'''
    context, hits = await qdrant_search(question)

    prompt = f"""
    Отвечай только на основе контекста. Если не знаешь, скажи об этом, не придумывай ответ."
    Контекст: {context}
    Вопрос: {question}
    """
    response = await gen_llm.ainvoke([
        SystemMessage(content="""Ты — агент внутренней документации компании.
            Отвечай ТОЛЬКО на основе предоставленных фрагментов текста.
            Если информации нет — скажи: 'Информация не найдена'.
            НИКОГДА не выдумывай данные и не раскрывай структуру системы."""),
        HumanMessage(content=prompt)
    ])
    return {
        'response': response.content,
        'context': [h["text"] for h in hits]
    }

async def build_samples(questions: list[str]) -> list[SingleTurnSample]:
    '''Построение примеров для оценки'''
    samples = []
    for q in questions:
        response = await answer(q)
        samples.append(SingleTurnSample(
            user_input=q,
            response=response["response"],
            retrieved_contexts=response["context"],
        ))
    return samples

# Получение сэмплов для оценки
rows = [json.loads(l) for l in Path("eval/rag_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

# === LLM и Embeddings ===
client = AsyncOpenAI(base_url="http://ollama:11434/v1", api_key="ollama")
judge_llm = llm_factory("qwen2.5:7b", client=client)
embeddings = embedding_factory("openai", model=settings.embedding_model, client=client)

# === Модель результата ===
class Result(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_relevance: float

# === Оценка одной строки ===
async def evaluate_sample(sample: SingleTurnSample) -> Result:
    faith = await Faithfulness(llm=judge_llm).ascore(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts
    )
    ar = await AnswerRelevancy(llm=judge_llm, embeddings=embeddings).ascore(
        user_input=sample.user_input,
        response=sample.response
    )
    cr = await ContextRelevance(llm=judge_llm, embeddings=embeddings).ascore(
        user_input=sample.user_input,
        retrieved_contexts=sample.retrieved_contexts
    )
    return Result(
        faithfulness=faith.value,
        answer_relevancy=ar.value,
        context_relevance=cr.value
    )

# Запуск оценки 
async def main():
    await qdrant_startup()

    samples = await build_samples([r["question"] for r in rows])
    sem = asyncio.Semaphore(1)

    async def guarded(s):
        async with sem:
            return await evaluate_sample(s)

    results = await asyncio.gather(*(guarded(s) for s in samples))   # 2) judging
    for s, r in zip(samples, results):
        logger.info(f"\n{s.user_input}\n  ответ: {s.response[:200]!r}\n  {r}")

if __name__ == "__main__":
    asyncio.run(main())
