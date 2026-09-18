from pydantic import BaseModel, field_validator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
import re

# валидация State для агента
class State(TypedDict):
    messages: Annotated[list, add_messages]
    task_id: int
    user_id: int