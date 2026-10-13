from prometheus_client import Counter, Histogram

ROUTING = Counter("agent_routing_total", "Выбранное действие", ["action"])
TOOL_ERRORS = Counter("agent_tool_errors_total", "Ошибки tools", ["tool"])
LLM_SECONDS = Histogram("llm_call_seconds", "Ожидание запроса LLM", ["stage"],
                        buckets=(1, 2, 5, 10, 20, 30, 60, 120))
RETRIEVAL_SECONDS = Histogram("retrieval_seconds", "Qdrant поиск с embedding",
                              buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5))
TOP_SCORE = Histogram("retrieval_top_score", "Оценка лучших чанков",
                      buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
SESSION_ERRORS = Counter("session_errors_total", "Ошибки сессии", ["op"])