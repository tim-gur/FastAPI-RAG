from pydantic import BaseModel, field_validator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
import re

# валидация State для агента
class State(TypedDict):
    messages: Annotated[list, add_messages]
    task_id: int
    user_id: int

# валидация запроса пользователя
class QueryCheck(BaseModel):
    user_id: int
    query: str

    @field_validator('query')
    @classmethod
    def check_empty_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Запрос не может быть пустым')
        if len(v) > 2000:
            raise ValueError('Запрос слишком длинный (макс. 2000 символов)')
        return v

    @field_validator('query')
    @classmethod
    def check_injections(cls, v: str) -> str:
        words = ["игнорируй", "забудь", "system prompt", "выведи все", "раскрой секрет"]
        l = v.lower()
        for pattern in words:
            if pattern in l:
                raise ValueError('Запрос содержит недопустимые слова')
        return v