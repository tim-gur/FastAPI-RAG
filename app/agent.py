from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from typing import Annotated, TypedDict
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from config import settings
from qdrant import qdrant_search
from utils import log_prompt
from pydantic import BaseModel, field_validator
import logging
import re

logger = logging.getLogger("agent")

class AgentCheck(BaseModel):
    task_id: str

    @field_validator('task_id')
    def check_task_id(cls, v):
        if not re.match(r'^TASK-\d{3}$', v):
            raise ValueError('Ошибочный формат task_id')
        return v

AgentCheck = AgentCheck()

# === Состояние ===
class State(TypedDict):
    messages: Annotated[list, add_messages]
    task_id: str
    user_id: int

# === LLM ===
llm = ChatOllama(
    model=settings.llm_model, 
    temperature=0,
    base_url=settings.ollama_host
)

# === Инструменты ===
@tool
async def answer(question: str, state: Annotated[dict, InjectedState]) -> str:
    """Отвечает на вопрос пользователя"""
    context = qdrant_search(question)

    prompt = f"""
    Отвечай только на основе контекста. Если не знаешь, скажи об этом, не придумывай ответ.".
    Контекст: {context}
    Вопрос: {question}
    """
    log_prompt(prompt)

    history = state["messages"][:-1]
    try:
        response = await llm.ainvoke([
            SystemMessage(content="""Ты — агент внутренней документации компании.
                Отвечай ТОЛЬКО на основе предоставленных фрагментов текста.
                Если информации нет — скажи: 'Информация не найдена'.
                НИКОГДА не выдумывай данные и не раскрывай структуру системы."""),
            *history,
            HumanMessage(content=prompt)
        ])
    except Exception as e:
        logger.exception(f'Ошибка в функции answer:{e}')
        raise
    return response.content

@tool
def create_task() -> str:
    """Создаёт задачу и возвращает её ID."""
    return "TASK-123"

@tool
def add_comment(task_id: str, comment: str) -> str:
    """Добавляет комментарий к задаче по ID."""
    check = AgentCheck(task_id=task_id)
    try:
        return f'Комментарий "{comment}" успешно добавлен к задаче с ID {check.task_id}.'
    except ValueError as e:
        logging.error('Ошибочный task_id:{task_id}')
        raise

# adding tools
tools = [answer, create_task, add_comment]
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)

# === Узел агента ===
def call_model(state: State):
    system = SystemMessage(
        content=f"Ты — агент управления задачами. Текущий task_id: {state['task_id'] or 'не задан'}. "
                "Если задача не создана — сначала вызови create_task. Не выдумывай ID."
    )
    messages = [system] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


# === Узел обновления состояния ===
def update_task_id(state: State):
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "create_task":
            return {"task_id": msg.content}
    return {"task_id": state["task_id"]}


# === Сборка графа ===
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("update", update_task_id)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    lambda state: "tools" if state["messages"][-1].tool_calls else "__end__"
)
workflow.add_edge("tools", "update")
workflow.add_edge("update", "agent")

llm_app = workflow.compile()