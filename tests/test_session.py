# tests/test_session.py
import sqlite3
import pytest
from app.session import create_task_sql, add_comment_sql

def test_create_task_sql(temp_db):
    task_id = create_task_sql(user_id=42)
    assert isinstance(task_id, int)

    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT user_id, status FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row == (42, "created")

def test_add_comment_sql(temp_db):
    task_id = create_task_sql(user_id=42)
    add_comment_sql(user_id=42, task_id=task_id, comment="test comment")

    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT comment FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row[0] == "test comment"