import json
import redis
from config import settings
from langchain_core.messages import messages_to_dict, messages_from_dict

r = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)

def get_state(user_id: int) -> dict:
    raw = r.get(f"session:{user_id}")
    if raw is None:
        return {"messages": [], "task_id": ""}
    data = json.loads(raw)
    data["messages"] = messages_from_dict(data["messages"])
    return data

def save_state(user_id: int, state: dict) -> None:
    serializable = {**state, "messages": messages_to_dict(state["messages"])}
    r.set(f"session:{user_id}", json.dumps(serializable))