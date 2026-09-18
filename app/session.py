import json
import redis
from langchain_core.messages import messages_to_dict, messages_from_dict
import sqlite3

from app.logger import logger
from app.settings import settings

# инициализация redis
r = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)

# загрузка сессии
def get_state(user_id: int) -> dict:
    raw = r.get(f"session:{user_id}")
    if raw is None:
        return {"messages": [], "task_id": ""}
    data = json.loads(raw)
    data["messages"] = messages_from_dict(data["messages"])
    return data

# сохранение сессии
def save_state(user_id: int, state: dict) -> None:
    serializable = {**state, "messages": messages_to_dict(state["messages"])}
    r.set(f"session:{user_id}", json.dumps(serializable))

# инициализация базы данных
def init_db():
    conn = sqlite3.connect(settings.db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            status TEXT,
            comment TEXT
        )
    """)
    conn.commit()
    conn.close()

def create_task_sql(user_id: int) -> int:
    conn = sqlite3.connect(settings.db_path)
    cursor = conn.execute(
        'INSERT INTO tasks (user_id, status) VALUES (?, ?)',
        (user_id, 'created')
    )
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    return task_id

def add_comment_sql(user_id: int, task_id: int, comment: str) -> None:
    conn = sqlite3.connect(settings.db_path)
    cursor = conn.execute(
        'UPDATE tasks SET comment = ? WHERE user_id = ? AND task_id = ?',
        (comment, user_id, task_id)
    )
    conn.commit()
    conn.close()