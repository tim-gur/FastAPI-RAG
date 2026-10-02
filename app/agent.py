from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.graph import StateGraph, START
from typing import Annotated
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from app.qdrant import qdrant_search
from app.utils import log_prompt
from app.schemas import State
from app.session import create_task_sql, add_comment_sql

from app.logger import logger
from app.settings import settings

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
    context, hits = await qdrant_search(question)

    # Логирование контекста
    logger.info(
        "retrieval | query=%r | hits=%s",
        question,
        [(h["source"], h["section"], round(h["score"], 3)) for h in hits],
    )
    logger.debug("retrieved context: %r", context)

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
def create_task(user_id: int) -> str:
    """Создаёт задачу и возвращает её ID."""
    try:
        return create_task_sql(user_id)
    except Exception as e:
        logger.error(f'Не получилось добавить задание: {e}')
        raise

@tool
def add_comment(user_id: int, task_id: int, comment: str) -> str:
    """Добавляет комментарий к задаче по ID."""
    try:
        add_comment_sql(user_id, task_id, comment)
        return f'Комментарий добавлен к задаче {task_id}.'
    except Exception as e:
        logger.error(f'Не получилось добавить комментарий: {e}')
        raise

tools = [answer, create_task, add_comment]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# === Узел агента: решает, вызывать ли инструмент ===
def call_model(state: State):
    system = SystemMessage(
        content=f"""Ты — агент, который либо отвечает на вопросы пользователя по документации, 
                    либо помогает управлять задачами (создание задачи, добавление комментариев).

                    Текущий task_id: {state['task_id'] or 'не задан'}.

                    Правила:
                    - Вопрос по документации → вызови answer.
                    - Просьба создать задачу → вызови create_task (если task_id ещё не задан).
                    - Просьба добавить комментарий → вызови add_comment с текущим task_id.
                    - Не вызывай create_task, если пользователь не просил создать задачу."""
    )
    messages = [system] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


# === Узел агента: формирует финальный текстовый ответ, БЕЗ доступа к инструментам ===
def respond(state: State):
    """Генерирует финальный текстовый ответ на основе результата инструмента.
    Использует обычный llm (без bind_tools) — физически не может вызвать инструмент снова."""
    system = SystemMessage(
        content="""Сформулируй краткий, понятный ответ пользователю на основе результата 
        выполненного действия (последнее сообщение в истории). Отвечай только текстом."""
    )
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


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
workflow.add_node("respond", respond)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    lambda state: "tools" if state["messages"][-1].tool_calls else "__end__"
)
workflow.add_edge("tools", "update")
workflow.add_edge("update", "respond")
workflow.add_edge("respond", "__end__")

llm_app = workflow.compile()