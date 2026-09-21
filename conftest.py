import pytest
import sqlite3
from app.qdrant import qdrant_startup
from app.main import app
from fastapi.testclient import TestClient

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_tasks.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            status TEXT,
            comment TEXT
        )
    """)
    conn.commit()
    conn.close()
    monkeypatch.setattr("app.settings.settings.db_path", str(db_path))
    return db_path

@pytest.fixture(scope='session', autouse=True)
async def init_qdrant_for_tests():
    await qdrant_startup()

@pytest.fixture()
def client():
    return TestClient(app)