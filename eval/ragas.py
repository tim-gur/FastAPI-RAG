import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRelevance
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.settings import settings
from app.qdrant import qdrant_search

llm = ChatOllama(
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
    response = await llm.ainvoke([
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
            retrieved_contexts=response["contexts"],
        ))
    return samples

# Получение сэмплов для оценки
questions = [
    'расскажи про валидацию данных в FastAPI',
    'Что такое эмбеддинги?',
    'Как обрабатывать ошибки в FastAPI?',
    'Чем dense-поиск отличается от гибридного?',
    'для чего нужна векторная база данных?'
    ]
samples = build_samples(questions)

dataset = EvaluationDataset(samples=samples)
# === LLM и Embeddings ===
client = AsyncOpenAI(base_url="http://ollama:11434/v1", api_key="ollama")
llm = llm_factory("qwen2.5:7b", client=client)
embeddings = embedding_factory("openai", model=settings.embedding_model, client=client)

# === Модель результата ===
class Result(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_relevance: float

# === Оценка одной строки ===
async def evaluate_sample(sample: SingleTurnSample) -> Result:
    faith = await Faithfulness(llm=llm).ascore(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts
    )
    ar = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
        user_input=sample.user_input,
        response=sample.response
    )
    cr = await ContextRelevance(llm=llm, embeddings=embeddings).ascore(
        user_input=sample.user_input,
        retrieved_contexts=sample.retrieved_contexts
    )
    return Result(
        faithfulness=faith.value,
        answer_relevancy=ar.value,
        context_relevance=cr.value
    )

# Запуск оценки 
async def evaluate():
    tasks = [evaluate_sample(sample) for sample in dataset.samples]
    results = await asyncio.gather(*tasks)
    for i, res in enumerate(results):
        print(f"Sample {i+1}: {res}")
